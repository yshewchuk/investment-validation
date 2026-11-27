"""EXP-190 Q6c: `python -m experiments.size_quantile_run --sources-config CFG --out-dir DIR [--tickers 50]`. Writes
run_inputs.json (size_quantile_run_inputs.v1) before any fit, then fits/ and stage_a_report.json; a smoke writes fits/
and smoke_record.json (size_quantile_smoke.v1: timing, no verdict, score or ledger row). Manual-only."""
import argparse
import functools
import hashlib
import json
import resource
import sqlite3
import sys
import time
from pathlib import Path

from engine.v2.contracts import ArtifactRef
from engine.v2.data.errors import DataError
from engine.v2.foundation import DocumentError, content_hash, from_document
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from engine.v2.research.progress import decision, stage
from experiments import size_pnl_sources as sources, size_pnl_tables
from experiments.memory_guard import rss_watchdog
from experiments.size_pnl_readers import admit_events, read_labels, read_production
from experiments.size_quantile_prereg import SPEC_PATH, load_spec, require_preregistration
from experiments.size_quantile_sources import load_sealed_panel
from experiments.size_quantile_stage_a import _publish, hash_labels, run_stage_a
from experiments.size_quantile_walk import fit_walk_forward
from experiments.size_quantile_y0 import y0_report

CAP_BYTES, SMOKE_SALT, END_YEAR = 8 * 2**30, "exp190-smoke", 2025


@stage("quantile inputs")
def load_inputs(cfg, conn, store):
    try:
        spec = experiment_spec_from_document(sources._read_json(cfg["spec"], "input_unreadable"))
        ref = from_document(ArtifactRef, sources._read_json(cfg["plan_ref"], "input_unreadable"))
    except (DocumentError, TypeError):
        raise fail("INVALID_EXPERIMENT_SPEC", "input unreadable", details={"reason": "input_unreadable"}) from None
    sealed = load_sealed_panel(conn, store, spec, ref)
    repo, snap, cap = sealed["repository"], sealed["snapshot"], sources.MAX_OBJECT_BYTES
    if snap.snapshot_id != cfg["snapshot_id"]:
        raise sources._snapshot_mismatch()
    sources._bound_snapshot(conn, cfg["crush_snapshot_id"])
    crush = sources._snapshot(repo, cfg["crush_snapshot_id"])
    admitted = admit_events(repo, snap, sealed["as_of_month"], max_object_bytes=cap)
    tables = size_pnl_tables.tables(repo, snap, admitted, crush_snapshot=crush, max_object_bytes=cap,
                                    expected_hash=cfg["production_hash"])
    cached = functools.cache(tables.fetch)
    def fetch(columns, ids):  # one read per key: the scored labels are the ones bound before the fit
        return cached(tuple(columns), tuple(ids)).copy()
    production = read_production(fetch, admitted, expected_hash=cfg["production_hash"], table_hash=tables.table_hash)
    return {"sealed": sealed, "admitted": admitted, "fetch": fetch, "production": production,
            "table_hash": tables.table_hash, "b0_labels_hash": hash_labels(read_labels(fetch, admitted))}


def smoke_subset(sealed, count):
    ranked = sorted({e["ticker"] for e in sealed["events"]},
                    key=lambda t: (hashlib.sha256(f"{t}|{SMOKE_SALT}".encode()).hexdigest(), t))
    chosen = ranked[:count]
    events = [e for e in sealed["events"] if e["ticker"] in chosen]
    ids = {e["event_id"] for e in events}
    rows = [r for r in sealed["rows"] if r["event_id"] in ids]
    decision("quantile smoke", f"tickers={len(ranked)} kept={len(chosen)} events={len(events)} rows={len(rows)}")
    return {**sealed, "events": events, "rows": rows,
            "binding": content_hash({"sealed": sealed["binding"], "smoke_tickers": chosen})}, chosen


def _run(sources_config, out_dir, tickers, **paths):
    guard, out_dir = require_preregistration(**paths), Path(out_dir)
    if tickers is not None and (type(tickers) is not int or tickers < 1):
        raise fail("INVALID_EXPERIMENT_SPEC", "tickers invalid", details={"reason": "smoke_tickers_invalid"})
    if tickers is not None and out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise fail("INVALID_EXPERIMENT_SPEC", "out dir not fresh", details={"reason": "smoke_out_dir_not_fresh"})
    spec, digest = load_spec(paths.get("spec_path", SPEC_PATH))
    if digest != guard["spec_hash"]:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "spec changed", details={"reason": "spec_changed"})
    cfg = sources._config(sources_config)
    conn, store = sources._open(cfg)
    try:
        inputs = load_inputs(cfg, conn, store)
        y0 = y0_report(inputs["sealed"], spec)
        years = [str(y) for y in range(int(y0["y0"]), END_YEAR + 1)]
        sealed, chosen = (inputs["sealed"], None) if tickers is None else smoke_subset(inputs["sealed"], tickers)
        body = {"schema_version": "size_quantile_run_inputs.v1", "mode": "full" if tickers is None else "smoke",
                "spec_hash": guard["spec_hash"], "preregistered_at": guard["preregistered_at"],
                "plan_id": sealed["plan"]["plan_id"], "sealed_binding": sealed["binding"], "years": years, "y0": y0,
                "production_binding": inputs["production"]["binding"], "production_table_hash": inputs["table_hash"],
                "b0_labels_hash": inputs["b0_labels_hash"], "smoke_tickers": chosen}
        record = _publish(out_dir, body, "run_inputs.json", "run_inputs_changed")
        if tickers is None:
            doc = run_stage_a(sealed, years, inputs["admitted"], inputs["fetch"], out_dir,
                              production_hash=cfg["production_hash"], table_hash=inputs["table_hash"],
                              run_inputs_hash=record["content_hash"], **paths)
            return {"mode": "full", "status": doc["status"], "reasons": doc["reasons"],
                    "content_hash": doc["content_hash"], "report": str(out_dir / "stage_a_report.json")}
        started = time.monotonic()
        receipts = fit_walk_forward(sealed, spec, years, out_dir / "fits")
        smoke = {"schema_version": "size_quantile_smoke.v1", "run_inputs_hash": record["content_hash"],
                 "events": len(sealed["events"]), "tickers": len(chosen), "fit_seconds": time.monotonic() - started,
                 "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 "fit_status": {y: r["status"] for y, r in receipts.items()},
                 "fit_receipt_hashes": {y: r["content_hash"] for y, r in receipts.items()}}
        return _publish(out_dir, smoke, "smoke_record.json", "smoke_record_changed")
    finally:
        conn.close()


def run(sources_config, out_dir, tickers=None, *, cap_bytes=CAP_BYTES, **paths):  # paths: the guard's path overrides
    try:
        with rss_watchdog(cap_bytes), stage("quantile stage A run"):
            return _run(sources_config, out_dir, tickers, **paths)
    except MemoryError:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "stage A ran out of memory") from None
    except sqlite3.Error:
        raise fail("RESOURCE_UNAVAILABLE", "catalog unavailable", details={"reason": "catalog_unavailable"}) from None


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tickers", type=int, default=None, help="smoke: this many tickers (smallest sha256 ranks)")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.sources_config, args.out_dir, args.tickers)))
    except (OpsError, DataError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
