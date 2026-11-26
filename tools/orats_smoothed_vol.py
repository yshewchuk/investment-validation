#!/usr/bin/env python3
"""Re-ingest ORATS smvVol, rates, borrow and dividend fields; see tools/ARCHITECTURE.md."""
import argparse
import contextlib
import gzip
import hashlib
import inspect
import io
import json
import logging
import os
import resource
import shutil
import sys
import time
import zlib
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

# ruff: noqa: E402
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from engine import paths
from engine.data.fetch import iter_cached
from engine.data.normalize.common import mask_sentinels
from engine.data.normalize.fetch_store import payload_shape

TOOL_VERSION, LOG = "4", logging.getLogger("smvvol")
MAX_META_BYTES, MAX_BODY_BYTES = 1 << 20, 1 << 28
QKEYS, RKEYS = ["ticker", "obs_date", "expiry", "strike"], ["ticker", "obs_date"]
QVALS, RATES = ["smvVol", "callMidIv", "putMidIv", "stockPrice"], ["iRate5wk", "iRateLt"]
CARRY = ["borrow30", "borrow2yr", "annIdiv", "annActDiv", "divAmt", "divDate", "assetType"]
TABLES = {"quotes": ("hist/strikes", QKEYS, QVALS, ["smvVol"]),
          "rates": ("hist/cores", RKEYS, [*RATES, *CARRY], RATES)}
UNITS = {"iRate5wk": "percent", "iRateLt": "percent", "borrow30": "percent",
         "borrow2yr": "percent", "annIdiv": "dollars_per_share", "annActDiv": "dollars_per_share",
         "divAmt": "dollars_per_share", "divDate": "date", "assetType": "code"}
DATE = "datetime64[ns]"
BAD, PROGRESS_S = ("unreadable", "unrecognized"), 30.0


def _source_bytes(path) -> bytes:
    with open(path, "rb") as fh:
        data = fh.read(MAX_META_BYTES + 1)
    if len(data) > MAX_META_BYTES:
        raise SystemExit(f"source {path} exceeds {MAX_META_BYTES} bytes")
    return data

CODE_DIGEST = hashlib.sha256(b"".join(_source_bytes(f) for f in (
    __file__, inspect.getfile(payload_shape), inspect.getfile(mask_sentinels)))).hexdigest()


def check_out_dir(path: Path, raw_root: Path, label: str) -> Path:
    resolved = path.resolve()
    for target in (paths.RAW, paths.CURATED, paths.FEATURES, raw_root):
        t = target.resolve()
        if resolved == t or t in resolved.parents or resolved in t.parents:
            raise SystemExit(f"{label} {resolved} overlaps {t}; refusing to write")
    return resolved


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write(tmp)
    os.replace(tmp, path)


def _atomic_json(path: Path, obj) -> None:
    text = json.dumps(obj, sort_keys=True, indent=2) + "\n"
    _atomic(path, lambda t: t.write_text(text, encoding="utf-8"))


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


@contextmanager
def _stage(name: str):
    LOG.info("[smvvol] %s start", name)
    t0 = time.monotonic()
    yield
    LOG.info("[smvvol] %s end: %.1f s, rss %.0f MB", name, time.monotonic() - t0, _rss_mb())


def _eta(done: int, total: int, start: float) -> float:
    return (total - done) * (time.monotonic() - start) / done


def _parse_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="%Y-%m-%d")
    in_range = parsed.between(pd.Timestamp.min, pd.Timestamp.max)
    return parsed.where(in_range).astype(DATE)


def parse_entry(table: str, entry, blob: bytes) -> tuple:
    _, keys, values, required = TABLES[table]
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as fh:
        body = fh.read(MAX_BODY_BYTES + 1)
    if len(body) > MAX_BODY_BYTES:
        raise ValueError(f"decompressed body exceeds {MAX_BODY_BYTES} bytes")
    shape, rows = payload_shape(json.loads(body))
    if shape == "unrecognized":
        return "unrecognized", None, {}, {}
    if not rows:
        return "empty", None, {"empty": 1}, {}
    raw = pd.DataFrame.from_records(rows)
    divs = {}
    if table == "rates":
        divs = {str(c): int(raw[c].notna().sum()) for c in raw.columns if "div" in str(c).lower()}
    if not raw.reindex(columns=required).notna().to_numpy().any():
        return "ok", None, {"without_required_fields": 1}, divs
    raw = raw.reindex(columns=["ticker", "tradeDate", "expirDate", "strike", *values])
    full = pd.DataFrame({
        "ticker": raw["ticker"].astype(str).where(raw["ticker"].notna()),
        "obs_date": pd.to_datetime(raw["tradeDate"], errors="coerce").astype(DATE),
        "expiry": pd.to_datetime(raw["expirDate"], errors="coerce").astype(DATE),
        "strike": mask_sentinels(pd.to_numeric(raw["strike"], errors="coerce")),
        **{v: (_parse_date(raw[v])
               if v == "divDate" else mask_sentinels(pd.to_numeric(raw[v], errors="coerce")))
           for v in values},
    })
    fetched = pd.to_datetime((entry.meta or {}).get("fetched_at"), utc=True,
                             errors="coerce", format="ISO8601")
    full["fetched_at"] = pd.Timestamp(0, tz="UTC") if pd.isna(fetched) else fetched
    full["cache_key"] = entry.key
    full["row_no"] = np.arange(len(full), dtype=np.int64)
    bad = full[keys].isna().any(axis=1).to_numpy()
    kept = full.loc[~bad].reset_index(drop=True)
    return "ok", kept, {"rows_bad_key": int(bad.sum()), "rows_kept": len(kept)}, divs


def _shard_bytes(path: Path) -> bytes:
    with open(path, "rb") as fh:
        blob = fh.read(MAX_BODY_BYTES + 1)
    if len(blob) > MAX_BODY_BYTES:
        raise ValueError(f"shard {path} exceeds {MAX_BODY_BYTES} bytes")
    return blob


def _shard_digest(path: Path) -> str:
    return hashlib.sha256(_shard_bytes(path)).hexdigest()


def _seal(marker: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in marker.items() if k != "seal"},
                                     sort_keys=True).encode()).hexdigest()


def _reusable_marker(marker_path: Path, stamp: list, shard_dir: Path, key: str):
    try:
        with open(marker_path, "rb") as fh:
            raw = fh.read(MAX_META_BYTES + 1)
        marker = json.loads(raw) if len(raw) <= MAX_META_BYTES else None
        if marker is None or marker["stamp"] != stamp or marker["seal"] != _seal(marker):
            return None
        for month, digest in marker["months"].items():
            if _shard_digest(shard_dir / month / f"{key}.parquet") != digest:
                return None
        return marker
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _shard_item(table: str, entry, work: Path) -> tuple:
    marker_path = work / "markers" / table / f"{entry.key}.json"
    try:
        with open(entry.path, "rb") as fh:
            blob = fh.read(MAX_BODY_BYTES + 1)
        if len(blob) > MAX_BODY_BYTES:
            raise ValueError(f"compressed body exceeds {MAX_BODY_BYTES} bytes")
        stamp = [CODE_DIGEST, (entry.meta or {}).get("fetched_at"), hashlib.sha256(blob).hexdigest()]
        marker = _reusable_marker(marker_path, stamp, work / "shards" / table, entry.key)
        if marker is not None:
            return "resumed", marker, ""
        status, frame, counters, divs = parse_entry(table, entry, blob)
    except (OSError, EOFError, ValueError, zlib.error) as exc:
        return "unreadable", None, str(exc)
    if status == "unrecognized":
        return "unrecognized", None, "payload shape unrecognized"
    months = {}
    if frame is not None and not frame.empty:
        for month, part in frame.groupby(frame["obs_date"].dt.strftime("%Y-%m")):
            path = work / "shards" / table / month / f"{entry.key}.parquet"
            _atomic(path, lambda t, p=part: p.to_parquet(t, index=False, engine="pyarrow"))
            months[month] = _shard_digest(path)
    marker = {"stamp": stamp, "months": months, "counters": counters, "dividend_fields": divs}
    marker["seal"] = _seal(marker)
    _atomic_json(marker_path, marker)
    return status, marker, ""


def shard_stage(entries: list, work: Path) -> tuple:
    markers, counts = {name: {} for name in TABLES}, Counter()
    start = last = time.monotonic()
    for done, (table, entry) in enumerate(entries, 1):
        status, marker, detail = _shard_item(table, entry, work)
        counts[(table, status)] += 1
        if marker is not None:
            markers[table][entry.key] = marker
        if status in BAD and sum(n for (_, s), n in counts.items() if s in BAD) <= 20:
            LOG.warning("[smvvol] %s %s %s: %s", status, table, entry.key, detail)
        now = time.monotonic()
        if now - last >= PROGRESS_S:
            last, rate = now, done / (now - start)
            resumed = sum(n for (_, s), n in counts.items() if s == "resumed")
            LOG.info("[smvvol] shard %d/%d, %.1f/s, ETA %.0f s, resumed-skipped %d",
                     done, len(entries), rate, (len(entries) - done) / rate, resumed)
    LOG.info("[smvvol] shard outcomes: %s", dict(counts))
    return markers, counts


def _read_shards(work: Path, table: str, month: str, entry_keys: list, markers: dict) -> list:
    shards, total = [], len(entry_keys)
    start = last = time.monotonic()
    for done, k in enumerate(entry_keys, 1):
        path = work / "shards" / table / month / f"{k}.parquet"
        blob = _shard_bytes(path)
        if hashlib.sha256(blob).hexdigest() != markers[table][k]["months"][month]:
            raise ValueError(f"shard {path} changed after it was written")
        shards.append(pd.read_parquet(io.BytesIO(blob)))
        now = time.monotonic()
        if now - last >= PROGRESS_S:
            last = now
            rate = done / max(now - start, 1e-9)
            LOG.info("[smvvol] merge %s %s: shards %d/%d, %.1f shards/s, ETA %.0f s, rss %.0f MB",
                     table, month, done, total, rate, _eta(done, total, start), _rss_mb())
    return shards


def merge_stage(work: Path, out: Path, markers: dict) -> dict:
    result = {}
    for table in TABLES:
        _, keys, values, _ = TABLES[table]
        cols = [*keys, *values]
        index: dict = {}
        for key in sorted(markers[table]):
            for month in markers[table][key]["months"]:
                index.setdefault(month, []).append(key)
        if (out / table).exists():
            shutil.rmtree(out / table)
        totals = Counter(rows_out=0, conflicting_keys=0, conflicting_keys_all_columns=0)
        non_null = Counter(dict.fromkeys(values, 0))
        start = time.monotonic()
        for done, (month, entry_keys) in enumerate(sorted(index.items()), 1):
            shards = _read_shards(work, table, month, entry_keys, markers)
            df = pd.concat(shards, ignore_index=True)
            LOG.info("[smvvol] merge %s %s: dedupe %d rows, rss %.0f MB",
                     table, month, len(df), _rss_mb())
            nc_cols = [*keys, *(v for v in values if v not in CARRY)]
            conflicts = (df[nc_cols].drop_duplicates().groupby(keys, sort=False).size() > 1).sum()
            conflicts_all = conflicts if len(nc_cols) == len(cols) else (
                df[cols].drop_duplicates().groupby(keys, sort=False).size() > 1).sum()
            final = df.sort_values([*keys, "fetched_at", "cache_key", "row_no"], kind="mergesort")
            final = final.drop_duplicates(subset=keys, keep="last")[cols].reset_index(drop=True)
            path = out / table / f"year={month[:4]}" / f"month={month[5:]}.parquet"
            _atomic(path, lambda t: final.to_parquet(t, index=False, engine="pyarrow"))
            totals.update({"rows_out": len(final), "conflicting_keys": int(conflicts),
                           "conflicting_keys_all_columns": int(conflicts_all)})
            non_null.update({v: int(final[v].notna().sum()) for v in values})
            LOG.info("[smvvol] merge %s %s: %d/%d months, rows_out %d, %.1f months/s, ETA %.0f s, rss %.0f MB",
                     table, month, done, len(index), len(final),
                     done / max(time.monotonic() - start, 1e-9), _eta(done, len(index), start),
                     _rss_mb())
        result[table] = {**totals, "non_null": dict(non_null)}
    return result


def _table_counts(entries: list, markers: dict, counts: Counter, merged: dict) -> dict:
    result = {}
    for table in TABLES:
        c = sum((Counter(m["counters"]) for m in markers[table].values()), Counter())
        result[table] = {
            "entries": sum(t == table for t, _ in entries), "entries_parsed": len(markers[table]),
            "unreadable": counts[(table, "unreadable")],
            "unrecognized": counts[(table, "unrecognized")],
            **{k: c[k] for k in ("empty", "without_required_fields", "rows_bad_key")},
            "rows_in": c["rows_kept"], "rows_out": merged[table]["rows_out"],
            "duplicates_dropped": c["rows_kept"] - merged[table]["rows_out"],
            "conflicting_keys": merged[table]["conflicting_keys"],
            "conflicting_keys_all_columns": merged[table]["conflicting_keys_all_columns"],
            "non_null": merged[table]["non_null"],
        }
    return result


def _run(entries: list, out: Path, work: Path) -> int:
    work.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").unlink(missing_ok=True)
    with _stage("shard"):
        markers, counts = shard_stage(entries, work)
    with _stage("merge"):
        merged = merge_stage(work, out, markers)
    tables = _table_counts(entries, markers, counts, merged)
    divs: Counter = Counter()
    for m in markers["rates"].values():
        divs.update(m["dividend_fields"])
    manifest = {"version": TOOL_VERSION, "units": UNITS, "tables": tables,
                "dividend_fields": divs}
    _atomic_json(out / "manifest.json", manifest)
    for name in ("shards", "markers"):
        shutil.rmtree(work / name, ignore_errors=True)
    with contextlib.suppress(OSError):
        work.rmdir()  # kept when --work-dir holds other files
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 1 if any(s in BAD for _, s in counts) else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw-root", type=Path, default=paths.RAW_FETCH)
    p.add_argument("--out", type=Path, default=paths.DATA / "experiment" / "orats_smoothed_vol")
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/orats_smoothed_vol_work"))
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    raw_root = args.raw_root.resolve()
    out = check_out_dir(args.out, raw_root, "--out")
    work = check_out_dir(args.work_dir, raw_root, "--work-dir")
    if out == work or out in work.parents or work in out.parents:
        raise SystemExit("--out and --work-dir must not equal or contain each other")
    found = [(name, e) for name, spec in TABLES.items()
             for e in iter_cached("orats", spec[0], root=raw_root, max_meta_bytes=MAX_META_BYTES)]
    entries = sorted(found, key=lambda it: (it[0], it[1].key))
    if not any(table == "quotes" for table, _ in entries):
        raise SystemExit("zero hist/strikes entries in the raw cache")
    return _run(entries, out, work)


if __name__ == "__main__":
    raise SystemExit(main())
