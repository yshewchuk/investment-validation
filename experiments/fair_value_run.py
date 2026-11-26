"""Fill model F chunked pricing smoke; usage and refusals: the `fair_value_run` row of experiments/README.md."""

import argparse
import hashlib
import io
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet

from engine.fills import MIN_MEANINGFUL_COST
from experiments import fair_value_fills as fvf
from experiments import fair_value_inputs as fvi
from experiments import fair_value_legs as fvl

BATCH_ROWS, LEG_KEYS = 50_000, ["structure_id", "ticker", "entry_date", "exit_date", "expiry"]
SOURCE_FILES = ("fair_value_carry.py", "fair_value_fills.py", "fair_value_inputs.py", "fair_value_legs.py",
                "fair_value_pricer.py")
SUMMED = ("rows", "put_rows", "expiries", "expiries_missing", "load_s", "price_s")
EXPIRY, LOG = fvl.EXPIRY_KEY, logging.getLogger("fair_value_run")


def fvf_id(root=Path(__file__).parent) -> str:
    files = {n: hashlib.sha256(fvi._read_capped(Path(root) / n, 1 << 20, n)).hexdigest() for n in SOURCE_FILES}
    blob = {"files": files, "constants": {"engine.fills.MIN_MEANINGFUL_COST": MIN_MEANINGFUL_COST}}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require_fvf_id(expected: str) -> str:
    if (current := fvf_id()) != expected:
        raise fvi.FairValueRefusal("fvf_id_mismatch", "differs", expected=expected, current=current)
    return current


def _peak_mb() -> float:
    return int(Path("/proc/self/status").read_text(encoding="utf-8").split("VmHWM:")[1].split()[0]) / 1024.0


def requested_expiries(legs: pd.DataFrame) -> pd.DataFrame:
    if any(c not in legs.columns for c in LEG_KEYS) or len(legs) == 0 or legs[LEG_KEYS].isna().any().any():
        raise fvi.FairValueRefusal("legs_invalid", "legs lack columns, are empty or hold a null")
    entry, leg_exit = (pd.to_datetime(legs[c], errors="coerce") for c in ("entry_date", "exit_date"))
    if entry.isna().any() or leg_exit.isna().any() or (leg_exit < entry).any():
        raise fvi.FairValueRefusal("legs_invalid", "dates do not convert or exit_date precedes entry_date")
    keys = [fvi._norm(legs.assign(obs_date=d.to_numpy()), EXPIRY) for d in (entry, leg_exit)]
    return pd.concat(keys).drop_duplicates().sort_values(EXPIRY, kind="mergesort").reset_index(drop=True)


def _check_rss(cfg, quarter) -> None:
    if (peak := _peak_mb()) > cfg.max_rss_mb:
        raise fvi.FairValueRefusal("rss_over_limit", "peak RSS is over the limit", code="RESOURCE_LIMIT_EXCEEDED",
                                   peak_rss_mb=peak, quarter=str(quarter))


def _run_chunk(cfg, quarter, wanted: pd.DataFrame):
    started = time.monotonic()
    limit = int(cfg.max_rss_mb * 2**20)  # one limit bounds tables and quote file
    Path("/proc/self/clear_refs").write_text("5", encoding="utf-8")  # reset VmHWM: peak_rss_mb is per chunk
    lagged = pd.concat([wanted["obs_date"], wanted["obs_date"] - pd.Timedelta(days=fvi.RATE_LOOKBACK_DAYS)])
    rels = sorted({f"quotes/year={m.year}/month={m.month:02d}.parquet" for m in wanted["obs_date"].dt.to_period("M")}
                  | {f"rates/year={m.year}/month={m.month:02d}.parquet" for m in lagged.dt.to_period("M")})
    if missing := [rel for rel in rels if rel not in cfg.pins["files"]]:
        raise fvi.FairValueRefusal("table_absent", "no pin for a needed table", paths=missing)
    tables = fvi.load_tables(cfg.tables_root, manifest_sha256=cfg.pins["manifest_sha256"], vol_scale=cfg.vol_scale,
                             file_sha256={r: cfg.pins["files"][r] for r in rels}, rate_scale=cfg.rate_scale,
                             max_total_bytes=limit)
    path = Path(cfg.quotes_dir) / f"{quarter}.parquet"
    blob = fvi._read_capped(path, limit, path.name)
    meta = pyarrow.parquet.read_metadata(io.BytesIO(blob))
    size = sum(meta.row_group(g).total_byte_size for g in range(meta.num_row_groups))
    if (size := max(size, meta.num_rows * fvi.DECODED_ROW_BYTES)) > limit:  # admit from the footer, before decoding
        raise fvi.FairValueRefusal("input_too_large", "quote file estimate over max_rss_mb",
                                   code="RESOURCE_LIMIT_EXCEEDED", path=path.name, decoded_bytes=size)
    frame = pd.read_parquet(io.BytesIO(blob), columns=[c for c in fvi.QUOTE_COLUMNS if c in meta.schema.names])
    rows = fvi.bind_quotes(frame, snapshot_id=cfg.snapshot_id,
                           expected_snapshot_id=cfg.expected_snapshot_id).reset_index(drop=True)
    rows[fvi.VOL_KEYS] = fvi._norm(rows, fvi.VOL_KEYS)
    rows = rows.merge(wanted, on=EXPIRY, how="inner").reset_index(drop=True)
    rows = rows.join(fvi.lookup_inputs(rows, tables.rates))
    load_s = time.monotonic() - started
    _check_rss(cfg, quarter)
    ids = (rows.groupby(EXPIRY).size().cumsum().shift(fill_value=0) // BATCH_ROWS).to_numpy()  # whole expiries
    batches = [p for _, p in rows.groupby(ids[rows.groupby(EXPIRY).ngroup()])] if len(rows) else []
    price_s = 0.0
    for part in batches:
        t0 = time.monotonic()
        fvf.leg_fills(part, tables.vols)  # priced table discarded: this smoke times pricing
        price_s += time.monotonic() - t0
        _check_rss(cfg, quarter)
    return {"quarter": str(quarter), "rows": len(rows), "put_rows": int((rows["right"] == "P").sum()),
            "expiries": len(wanted), "expiries_missing": len(wanted) - len(rows[EXPIRY].drop_duplicates()),
            "batches": len(batches), "load_s": load_s, "price_s": price_s,
            "total_s": time.monotonic() - started, "peak_rss_mb": _peak_mb()}


def run_smoke(legs, *, quotes_dir, tables_root, pins, snapshot_id, expected_snapshot_id, vol_scale, rate_scale,
              max_rss_mb, expected_fvf_id=None) -> dict:
    started, current = time.monotonic(), require_fvf_id(expected_fvf_id) if expected_fvf_id is not None else fvf_id()
    if not (isinstance(max_rss_mb, (int, float)) and 0 < max_rss_mb < float("inf")):
        raise fvi.FairValueRefusal("rss_limit", "max_rss_mb must be a finite number > 0", name="max_rss_mb")
    if not (isinstance(pins, dict) and isinstance(pins.get("manifest_sha256"), str)
            and isinstance(pins.get("files"), dict)):
        raise fvi.FairValueRefusal("pins_invalid", "pins need a manifest_sha256 string and a files object")
    expiries = requested_expiries(legs)
    groups = {q: g for q, g in expiries.groupby(expiries["obs_date"].dt.to_period("Q"))}
    if absent := [str(q) for q in groups if not (Path(quotes_dir) / f"{q}.parquet").is_file()]:
        raise fvi.FairValueRefusal("quotes_file_absent", "a needed quarter has no quote file", quarters=absent)
    cfg = argparse.Namespace(quotes_dir=quotes_dir, tables_root=tables_root, pins=pins, snapshot_id=snapshot_id,
                             expected_snapshot_id=expected_snapshot_id, vol_scale=vol_scale, rate_scale=rate_scale,
                             max_rss_mb=float(max_rss_mb))
    LOG.info("[fv_smoke] start expiries=%d quarters=%d", len(expiries), len(groups))
    chunks = []
    for done, (quarter, wanted) in enumerate(groups.items(), 1):
        chunks.append(record := _run_chunk(cfg, quarter, wanted))
        LOG.info("[fv_smoke] %d/%d %s rows=%d %.1fs rows/s=%.0f eta=%.0fs peak_rss=%.0fMB", done, len(groups),
                 quarter, record["rows"], record["total_s"], record["rows"] / max(record["total_s"], 1e-9),
                 (time.monotonic() - started) / done * (len(groups) - done), record["peak_rss_mb"])
    totals = {**{k: sum(c[k] for c in chunks) for k in SUMMED},
              "elapsed_s": time.monotonic() - started, "peak_rss_mb": max(c["peak_rss_mb"] for c in chunks)}
    LOG.info("[fv_smoke] end elapsed=%.1fs", totals["elapsed_s"])
    return {"fvf_id": current, "chunks": chunks, "totals": totals}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fill model F chunked pricing smoke; one JSON line on stdout.")
    for flag in ("--legs", "--quotes-dir", "--tables-root", "--pins", "--snapshot-id", "--expected-snapshot-id"):
        parser.add_argument(flag, required=True)
    for flag in ("--vol-scale", "--rate-scale", "--max-rss-mb"):
        parser.add_argument(flag, type=float, required=True)
    parser.add_argument("--expected-fvf-id")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    try:
        args.pins = json.loads(fvi._read_capped(Path(args.pins), fvi.MAX_MANIFEST_BYTES, "pins"))
        legs = io.BytesIO(fvi._read_capped(Path(args.legs), fvi.MAX_TABLE_BYTES, "legs"))
        report = run_smoke(pd.read_csv(legs), **{k: v for k, v in vars(args).items() if k != "legs"})
    except (ValueError, OSError, pyarrow.ArrowException) as err:  # refusals, and JSON, CSV and Arrow read errors
        refusal = err if hasattr(err, "details") else fvi.FairValueRefusal("input_unreadable", "", error=repr(err))
        print(json.dumps({"refusal": refusal.code, "details": refusal.details}, default=str))
        return 2
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
