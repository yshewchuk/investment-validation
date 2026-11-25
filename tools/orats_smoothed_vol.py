#!/usr/bin/env python3
"""Re-ingest ORATS smvVol and rates from the raw cache into experiment parquet.

Never writes raw, curated or feature stores. Resumable via --work-dir.
"""
import argparse
import contextlib
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
from datetime import datetime
from pathlib import Path

# ruff: noqa: E402
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from engine import paths
from engine.data.fetch import iter_cached
from engine.data.normalize.common import mask_sentinels
from engine.data.normalize.fetch_store import payload_shape

TOOL_VERSION, LOG = "1", logging.getLogger("smvvol")
QKEYS, RKEYS = ["ticker", "obs_date", "expiry", "strike"], ["ticker", "obs_date"]
QVALS, RATES = ["smvVol", "callMidIv", "putMidIv", "stockPrice"], ["iRate5wk", "iRateLt"]
# table: (endpoint, keys, values, required); a row carries the table if any value is set
TABLES = {"quotes": ("hist/strikes", QKEYS, QVALS, ["smvVol"]),
          "rates": ("hist/cores", RKEYS, RATES, RATES)}
DATE = "datetime64[ns]"
DTYPES = {"ticker": object, "obs_date": DATE, "expiry": DATE, "strike": float}
BAD, YEARS, PROGRESS_S = ("unreadable", "unrecognized"), range(2017, 2027), 30.0


def check_out_dir(path: Path, raw_root: Path, curated: Path, label: str) -> Path:
    resolved = path.resolve()
    for target in (paths.RAW, paths.CURATED, paths.FEATURES, raw_root, curated):
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


@contextmanager
def _stage(name: str):
    LOG.info("[smvvol] %s start", name)
    t0 = time.monotonic()
    yield
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    LOG.info("[smvvol] %s end: %.1f s, rss %.0f MB", name, time.monotonic() - t0, rss)


def _eta(done: int, total: int, start: float) -> float:
    return (total - done) * (time.monotonic() - start) / done


def parse_entry(table: str, entry) -> tuple:
    _, keys, values, required = TABLES[table]
    shape, rows = payload_shape(entry.json())
    if shape == "unrecognized":
        return "unrecognized", None, 0, False, {}
    if not rows:
        return "empty", None, 0, False, {}
    raw = pd.DataFrame.from_records(rows)
    divs = {}
    if table == "rates":
        divs = {str(c): int(raw[c].notna().sum()) for c in raw.columns if "div" in str(c).lower()}
    if not raw.reindex(columns=required).notna().to_numpy().any():
        return "ok", None, 0, True, divs
    raw = raw.reindex(columns=["ticker", "tradeDate", "expirDate", "strike", *values])
    full = pd.DataFrame({
        "ticker": raw["ticker"].astype(str).where(raw["ticker"].notna()),
        "obs_date": pd.to_datetime(raw["tradeDate"], errors="coerce").astype(DATE),
        "expiry": pd.to_datetime(raw["expirDate"], errors="coerce").astype(DATE),
        "strike": mask_sentinels(pd.to_numeric(raw["strike"], errors="coerce")),
        **{v: mask_sentinels(pd.to_numeric(raw[v], errors="coerce")) for v in values},
    })
    fetched = pd.to_datetime((entry.meta or {}).get("fetched_at"), utc=True,
                             errors="coerce", format="ISO8601")
    full["fetched_at"] = pd.Timestamp(0, tz="UTC") if pd.isna(fetched) else fetched
    full["cache_key"] = entry.key
    full["row_no"] = np.arange(len(full), dtype=np.int64)
    bad = full[keys].isna().any(axis=1).to_numpy()
    return "ok", full.loc[~bad].reset_index(drop=True), int(bad.sum()), False, divs


def _shard_item(table: str, entry, work: Path) -> tuple:
    stamp = [TOOL_VERSION, (entry.meta or {}).get("fetched_at"), entry.path.stat().st_size]
    marker_path = work / "markers" / table / f"{entry.key}.json"
    try:
        old = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = None
    if isinstance(old, dict) and old.get("stamp") == stamp:
        return "resumed", old, ""
    try:
        status, frame, bad_keys, no_req, divs = parse_entry(table, entry)
    except (OSError, EOFError, ValueError, zlib.error) as exc:
        return "unreadable", None, str(exc)
    if status == "unrecognized":
        return "unrecognized", None, "payload shape unrecognized"
    months = []
    if frame is not None and not frame.empty:
        for month, part in frame.groupby(frame["obs_date"].dt.strftime("%Y-%m")):
            path = work / "shards" / table / month / f"{entry.key}.parquet"
            _atomic(path, lambda t, p=part: p.to_parquet(t, index=False, engine="pyarrow"))
            months.append(month)
    counters = {"rows_bad_key": bad_keys, "rows_kept": 0 if frame is None else len(frame),
                "without_required_fields": int(no_req), "empty": int(status == "empty")}
    marker = {"stamp": stamp, "months": months, "counters": counters, "dividend_fields": divs}
    _atomic_json(marker_path, marker)
    return status, marker, ""


def shard_stage(entries: list, work: Path) -> tuple:
    markers, counts, failures = {name: {} for name in TABLES}, Counter(), Counter()
    start = last = time.monotonic()
    for done, (table, entry) in enumerate(entries, 1):
        status, marker, detail = _shard_item(table, entry, work)
        counts[status] += 1
        if marker is not None:
            markers[table][entry.key] = marker
        if status in BAD:
            failures[(table, status)] += 1
            if sum(failures.values()) <= 20:
                LOG.warning("[smvvol] %s %s %s: %s", status, table, entry.key, detail)
        now = time.monotonic()
        if now - last >= PROGRESS_S:
            last, rate = now, done / (now - start)
            LOG.info("[smvvol] shard %d/%d, %.1f/s, ETA %.0f s, resumed-skipped %d",
                     done, len(entries), rate, (len(entries) - done) / rate, counts["resumed"])
    LOG.info("[smvvol] shard outcomes: %s", dict(counts))
    return markers, failures


def merge_stage(work: Path, out: Path, markers: dict) -> dict:
    result = {}
    for table in TABLES:
        _, keys, values, _ = TABLES[table]
        cols = [*keys, *values]
        index: dict = {}
        for key in sorted(markers[table]):
            for month in markers[table][key]["months"]:
                index.setdefault(month, []).append(key)
        shutil.rmtree(out / table, ignore_errors=True)
        totals = Counter(rows_out=0, conflicting_keys=0)
        start = time.monotonic()
        for done, (month, entry_keys) in enumerate(sorted(index.items()), 1):
            shards = [pd.read_parquet(work / "shards" / table / month / f"{k}.parquet")
                      for k in entry_keys]
            df = pd.concat(shards, ignore_index=True)
            conflicts = (df[cols].drop_duplicates().groupby(keys, sort=False).size() > 1).sum()
            final = df.sort_values([*keys, "fetched_at", "cache_key", "row_no"], kind="mergesort")
            final = final.drop_duplicates(subset=keys, keep="last")[cols].reset_index(drop=True)
            path = out / table / f"year={month[:4]}" / f"month={month[5:]}.parquet"
            _atomic(path, lambda t: final.to_parquet(t, index=False, engine="pyarrow"))
            totals.update({"rows_out": len(final), "conflicting_keys": int(conflicts)})
            LOG.info("[smvvol] merge %s %s: %d/%d months, rows_out %d, ETA %.0f s",
                     table, month, done, len(index), len(final),
                     _eta(done, len(index), start))
        result[table] = dict(totals)
    return result


def _read_chains(curated: Path, year: int, month: int):
    start, end = datetime(year, month, 1), datetime(year + (month == 12), month % 12 + 1, 1)
    filters = [("obs_date", ">=", start), ("obs_date", "<", end)]
    frames = [pq.read_table(f, columns=[*QKEYS, "chain_kind"], filters=filters).to_pandas()
              for f in sorted(curated.glob(f"year={year}/part-*.parquet"))]
    if not sum(map(len, frames)):
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.astype({"ticker": str, "strike": float, "obs_date": DATE, "expiry": DATE})
    df["chain_kind"] = df["chain_kind"].astype(object).fillna("unknown").astype(str)
    return df


def _read_out(out: Path, table: str, year: int, month: int, cols: list) -> pd.DataFrame:
    path = out / table / f"year={year}" / f"month={month:02d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame({c: pd.Series(dtype=DTYPES.get(c, float)) for c in cols})


def coverage_stage(curated: Path, out: Path) -> list[dict]:
    tallies: dict = {}
    start = time.monotonic()
    for i, year in enumerate(YEARS, 1):
        for month in range(1, 13):
            chains = _read_chains(curated, year, month)
            if chains is None:
                continue
            quotes = _read_out(out, "quotes", year, month, QKEYS + ["smvVol"])
            rates = _read_out(out, "rates", year, month, RKEYS + RATES)
            m = chains.merge(quotes, on=QKEYS, how="left", indicator=True, validate="many_to_one")
            r = m.merge(rates, on=RKEYS, how="left", validate="many_to_one")
            smv = m["smvVol"].to_numpy(dtype=float)
            flags = pd.DataFrame({"chain_kind": m["chain_kind"].to_numpy(),
                                  "key": (m["_merge"] == "both").to_numpy(),
                                  "smv": np.isfinite(smv) & (smv > 0),
                                  "rates": r[RATES].notna().any(axis=1).to_numpy()})
            for kind, g in flags.groupby("chain_kind"):
                add = np.array([len(g), *g[["key", "smv", "rates"]].sum()], dtype=np.int64)
                tallies[(year, kind)] = tallies.get((year, kind), 0) + add
        LOG.info("[smvvol] coverage %d done: %d/%d years, ETA %.0f s",
                 year, i, len(YEARS), _eta(i, len(YEARS), start))
    rows = []
    for year in YEARS:
        kinds = sorted(k for y, k in tallies if y == year)
        total = sum((tallies[(year, k)] for k in kinds), np.zeros(4, dtype=np.int64))
        for kind, v in [(k, tallies[(year, k)]) for k in kinds] + [("all", total)]:
            n = dict(zip(["rows", "key_matched", "smv_matched", "rates_matched"], map(int, v)))
            share = round(n["smv_matched"] / n["rows"], 6) if n["rows"] else None
            rows.append({"year": year, "chain_kind": kind, **n, "smv_share": share})
    return rows


def _table_counts(entries: list, markers: dict, failures: Counter, merged: dict) -> dict:
    result = {}
    for table in TABLES:
        c = Counter()
        for marker in markers[table].values():
            c.update(marker["counters"])
        result[table] = {
            "entries": sum(t == table for t, _ in entries), "entries_parsed": len(markers[table]),
            "unreadable": failures[(table, "unreadable")],
            "unrecognized": failures[(table, "unrecognized")],
            **{k: c[k] for k in ("empty", "without_required_fields", "rows_bad_key")},
            "rows_in": c["rows_kept"], "rows_out": merged[table]["rows_out"],
            "duplicates_dropped": c["rows_kept"] - merged[table]["rows_out"],
            "conflicting_keys": merged[table]["conflicting_keys"],
        }
    return result


def _run(entries: list, curated: Path, out: Path, work: Path) -> int:
    (out / "manifest.json").unlink(missing_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    with _stage("shard"):
        markers, failures = shard_stage(entries, work)
    with _stage("merge"):
        merged = merge_stage(work, out, markers)
    counts = _table_counts(entries, markers, failures, merged)
    divs: Counter = Counter()
    for m in markers["rates"].values():
        divs.update(m["dividend_fields"])
    dividends = dict(sorted(divs.items()))
    with _stage("coverage"):
        rows = coverage_stage(curated, out)
        _atomic_json(out / "coverage.json", {"coverage": rows, "dividend_fields": dividends})
    manifest = {"version": TOOL_VERSION, "tables": counts}
    _atomic_json(out / "manifest.json", manifest)
    for name in ("shards", "markers"):
        shutil.rmtree(work / name, ignore_errors=True)
    with contextlib.suppress(OSError):
        work.rmdir()  # kept when --work-dir holds other files
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 1 if sum(failures.values()) else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--raw-root", type=Path, default=paths.RAW_FETCH)
    p.add_argument("--curated-chains", type=Path, default=paths.CURATED / "option_chains")
    p.add_argument("--out", type=Path, default=paths.DATA / "experiment" / "orats_smoothed_vol")
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/orats_smoothed_vol_work"))
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    raw_root, curated = args.raw_root.resolve(), args.curated_chains.resolve()
    out = check_out_dir(args.out, raw_root, curated, "--out")
    work = check_out_dir(args.work_dir, raw_root, curated, "--work-dir")
    if out == work or out in work.parents or work in out.parents:
        raise SystemExit("--out and --work-dir must not equal or contain each other")
    found = [(name, e) for name, spec in TABLES.items()
             for e in iter_cached("orats", spec[0], root=raw_root)]
    entries = sorted(found, key=lambda it: (it[0], it[1].key))
    if not list(curated.glob("year=*/part-*.parquet")):
        raise SystemExit(f"no year=*/part-*.parquet under {curated}")
    if not any(table == "quotes" for table, _ in entries):
        raise SystemExit("zero hist/strikes entries in the raw cache")
    return _run(entries, curated, out, work)


if __name__ == "__main__":
    raise SystemExit(main())
