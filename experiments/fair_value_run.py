"""Fill model F chunked PRICING smoke (entry/exit evaluation not timed, nothing stored). Run with
`python3 -m experiments.fair_value_run`; quotes_dir holds <year>Q<n>.parquet, every listed row of each expiry."""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from engine.fills import MIN_MEANINGFUL_COST
from experiments import fair_value_fills as fvf
from experiments import fair_value_inputs as fvi
from experiments import fair_value_legs as fvl

BATCH_ROWS, LEG_KEYS = 50_000, ["structure_id", "ticker", "entry_date", "exit_date", "expiry"]
SOURCE_FILES = ("fair_value_fills.py", "fair_value_inputs.py", "fair_value_legs.py", "fair_value_pricer.py")
SUMMED = ("rows", "put_rows", "expiries", "expiries_missing", "load_s", "price_s")
EXPIRY, LOG = fvl.EXPIRY_KEY, logging.getLogger("fair_value_run")


def fvf_id(root=Path(__file__).parent) -> str:
    files = {n: hashlib.sha256((Path(root) / n).read_bytes()).hexdigest() for n in SOURCE_FILES}
    blob = {"files": files, "constants": {"engine.fills.MIN_MEANINGFUL_COST": MIN_MEANINGFUL_COST}}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require_fvf_id(expected: str) -> str:
    current = fvf_id()
    if current != expected:
        raise fvi.FairValueRefusal("fvf_id_mismatch", "differs", expected=expected, current=current)
    return current


def _peak_mb() -> float:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    return int(status.split("VmHWM:")[1].split()[0]) / 1024.0


def requested_expiries(legs: pd.DataFrame) -> pd.DataFrame:
    if any(c not in legs.columns for c in LEG_KEYS) or len(legs) == 0 or legs[LEG_KEYS].isna().any().any():
        raise fvi.FairValueRefusal("legs_invalid", "legs lack columns, are empty or hold a null")
    entry, leg_exit = (pd.to_datetime(legs[c], errors="coerce") for c in ("entry_date", "exit_date"))
    if entry.isna().any() or leg_exit.isna().any() or (leg_exit < entry).any():
        raise fvi.FairValueRefusal("legs_invalid", "dates do not convert or exit_date precedes entry_date")
    keys = [fvi._norm(legs.assign(obs_date=d.to_numpy()), EXPIRY) for d in (entry, leg_exit)]
    return pd.concat(keys).drop_duplicates().sort_values(EXPIRY, kind="mergesort").reset_index(drop=True)


def _batches(rows: pd.DataFrame, batch_rows: int) -> list:
    start = (rows.groupby(EXPIRY).size().cumsum().shift(fill_value=0) // batch_rows).to_numpy()
    return [part for _, part in rows.groupby(start[rows.groupby(EXPIRY).ngroup()], sort=True)]


def _check_rss(cfg, quarter) -> None:
    if (peak := _peak_mb()) > cfg.max_rss_mb:
        raise fvi.FairValueRefusal("rss_over_limit", "peak RSS is over the limit", code="RESOURCE_LIMIT_EXCEEDED",
                                   peak_rss_mb=peak, quarter=str(quarter))


def _run_chunk(cfg, quarter, wanted: pd.DataFrame):
    started = time.monotonic()
    Path("/proc/self/clear_refs").write_text("5", encoding="utf-8")  # reset VmHWM: peak_rss_mb is per chunk
    lagged = pd.concat([wanted["obs_date"], wanted["obs_date"] - pd.Timedelta(days=fvi.RATE_LOOKBACK_DAYS)])
    rels = sorted({f"quotes/year={m.year}/month={m.month:02d}.parquet" for m in wanted["obs_date"].dt.to_period("M")}
                  | {f"rates/year={m.year}/month={m.month:02d}.parquet" for m in lagged.dt.to_period("M")})
    if missing := [rel for rel in rels if rel not in cfg.pins["files"]]:
        raise fvi.FairValueRefusal("table_absent", "no pin for a needed table", paths=missing)
    tables = fvi.load_tables(cfg.tables_root, manifest_sha256=cfg.pins["manifest_sha256"], vol_scale=cfg.vol_scale,
                             file_sha256={r: cfg.pins["files"][r] for r in rels}, rate_scale=cfg.rate_scale,
                             max_total_bytes=cfg.max_table_bytes)
    rows = fvi.bind_quotes(pd.read_parquet(Path(cfg.quotes_dir) / f"{quarter}.parquet"), snapshot_id=cfg.snapshot_id,
                           expected_snapshot_id=cfg.expected_snapshot_id).reset_index(drop=True)
    rows[fvi.VOL_KEYS] = fvi._norm(rows, fvi.VOL_KEYS)
    rows = rows.merge(wanted, on=EXPIRY, how="inner").reset_index(drop=True)
    rows["rate"] = fvi.lookup_rate(rows, tables.rates)
    load_s = time.monotonic() - started
    _check_rss(cfg, quarter)
    batches, price_s = (_batches(rows, cfg.batch_rows) if len(rows) else []), 0.0
    for part in batches:
        t0 = time.monotonic()
        fvf.leg_fills(part, tables.vols)  # priced table discarded: this smoke times pricing
        price_s += time.monotonic() - t0
        _check_rss(cfg, quarter)
    record = {"quarter": str(quarter), "rows": len(rows), "put_rows": int((rows["right"] == "P").sum()),
              "expiries": len(wanted), "expiries_missing": len(wanted) - len(rows[EXPIRY].drop_duplicates()),
              "batches": len(batches), "load_s": load_s, "price_s": price_s,
              "total_s": time.monotonic() - started, "peak_rss_mb": _peak_mb()}
    return record


def run_smoke(legs, *, quotes_dir, tables_root, pins, snapshot_id, expected_snapshot_id, vol_scale, rate_scale,
              max_rss_mb, expected_fvf_id=None, max_table_bytes=None, batch_rows=BATCH_ROWS) -> dict:
    started, current = time.monotonic(), require_fvf_id(expected_fvf_id) if expected_fvf_id is not None else fvf_id()
    if not (isinstance(max_rss_mb, (int, float)) and 0 < max_rss_mb < float("inf")):
        raise fvi.FairValueRefusal("rss_limit", "max_rss_mb must be a finite number > 0", name="max_rss_mb")
    if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or batch_rows <= 0:
        raise fvi.FairValueRefusal("batch_rows", "batch_rows must be an int > 0", name="batch_rows")
    expiries = requested_expiries(legs)
    groups = {q: g for q, g in expiries.groupby(expiries["obs_date"].dt.to_period("Q"))}
    if absent := [str(q) for q in groups if not (Path(quotes_dir) / f"{q}.parquet").is_file()]:
        raise fvi.FairValueRefusal("quotes_file_absent", "a needed quarter has no quote file", quarters=absent)
    cfg = argparse.Namespace(quotes_dir=quotes_dir, tables_root=tables_root, pins=pins, snapshot_id=snapshot_id,
                             expected_snapshot_id=expected_snapshot_id, vol_scale=vol_scale, rate_scale=rate_scale,
                             max_rss_mb=float(max_rss_mb), max_table_bytes=max_table_bytes, batch_rows=batch_rows)
    LOG.info("[fv_smoke] start structures=%d expiries=%d quarters=%d",
             legs.structure_id.nunique(), len(expiries), len(groups))
    chunks = []
    for done, (quarter, wanted) in enumerate(groups.items(), 1):
        record = _run_chunk(cfg, quarter, wanted)
        chunks.append(record)
        LOG.info("[fv_smoke] %d/%d %s rows=%d %.1fs rows/s=%.0f eta=%.0fs peak_rss=%.0fMB", done, len(groups),
                 quarter, record["rows"], record["total_s"], record["rows"] / max(record["total_s"], 1e-9),
                 (time.monotonic() - started) / done * (len(groups) - done), record["peak_rss_mb"])
    totals = {k: sum(c[k] for c in chunks) for k in SUMMED}
    totals.update(structures=int(legs.structure_id.nunique()), chunks=len(chunks),
                  elapsed_s=time.monotonic() - started, peak_rss_mb=max(c["peak_rss_mb"] for c in chunks),
                  rows_per_price_s=totals["rows"] / totals["price_s"] if totals["price_s"] > 0 else None)
    LOG.info("[fv_smoke] end elapsed=%.1fs", totals["elapsed_s"])
    return {"fvf_id": current, "chunks": chunks, "totals": totals}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fill model F chunked pricing smoke; one JSON line on stdout.")
    for flag in ("--legs", "--quotes-dir", "--tables-root", "--pins", "--snapshot-id", "--expected-snapshot-id"):
        parser.add_argument(flag, required=True)
    for flag in ("--vol-scale", "--rate-scale", "--max-rss-mb"):
        parser.add_argument(flag, type=float, required=True)
    parser.add_argument("--expected-fvf-id")
    parser.add_argument("--max-table-bytes", type=int)
    parser.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
    args = parser.parse_args(argv)
    args.pins = json.loads(Path(args.pins).read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    try:
        report = run_smoke(pd.read_csv(args.legs), **{k: v for k, v in vars(args).items() if k != "legs"})
    except (fvi.FairValueRefusal, fvl.FairValueRefusal) as err:
        print(json.dumps({"refusal": err.code, "details": err.details}, default=str))
        return 2
    print(json.dumps(report, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
