"""EXP-189 A1 W7c3 part a: report-input producers (design A1.2, A1.7, A1.10 "Report inputs"; F design section 8). Nothing calls
this yet; part b calls it from the run stages. Aggregates are integer counts (shares: count / the group's ``rows``), so chunk
results add up with ``merge_counts``. Follow-up: smile minus smvVol, |V - mid|/half-spread, c quantiles, no-arbitrage bound.
Schema: ``out/report_inputs/<stage>/<name>.parquet`` (index=False); ``out/receipts/report_inputs_<stage>.json`` with body
``{"stage", "inputs_hash", "scope", "tables": {name: {"sha256", "rows"}}}``. Failure rules: a partial failure leaves files but no
receipt (nothing bound); an exact replay completes it; changed bytes refuse ``report_input_changed`` (old bytes kept); a changed
receipt body refuses ``tuning_receipt_changed``; files are never replaced; concurrent writers are safe via create-only links.
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd

from experiments import fair_value_inputs as fvi
from experiments import size_pnl_decisions as decisions
from experiments import size_pnl_receipts as receipts
from experiments.size_pnl_shortvol import Refusal
from experiments.size_pnl_straddle import EXIT_UNQUOTED

EXIT_METRICS = ("trades", "exit_fair_mid", "exit_unquoted_worst_case", "exit_negative", "excluded_rung3", "excluded_unpriceable")
POINT_COLS = ["point", "year", "chain_kind", "metric", "value", "count"]
FLAG_COLS = ["repaired", "unknown_bid_size", "unknown_ask_size", "unknown_oi", "carry_fallback", "dividend_fallback",
             "dividend_applied"]
LEG_COLS = ["obs_date", "chain_kind", "status", "source", "clamp_side", *FLAG_COLS, "annIdiv", "annActDiv",
            "eff_bid", "eff_ask", "fill_buy", "fill_sell"]


def _flag(series):
    return series.fillna(False).astype(bool).to_numpy()


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def _text(series, null):
    return series.astype(object).where(series.notna(), null).astype(str)


def exit_flag_counts(rows):
    """Long frame [arm, structure, year, metric, count] of F taken rows: every metric for every group, zeros included."""
    if len(rows) == 0: return pd.DataFrame(columns=["arm", "structure", "year", "metric", "count"])
    rows, keys = rows.reset_index(drop=True), ["arm", "structure", "year"]
    wide = pd.DataFrame({"arm": rows["arm"], "structure": rows["structure"],
                         "year": pd.to_datetime(rows["event_date"]).dt.year, "trades": 1,
                         "exit_fair_mid": _flag(rows["exit_fair_mid"]), "exit_negative": _flag(rows["exit_negative"]),
                         "exit_unquoted_worst_case": (rows["rung"] == EXIT_UNQUOTED).to_numpy(),
                         "excluded_rung3": _flag(rows["excluded_rung3"]),
                         "excluded_unpriceable": _flag(rows["excluded_unpriceable"])})
    out = wide.groupby(keys)[list(EXIT_METRICS)].sum().reset_index().melt(id_vars=keys, var_name="metric", value_name="count")
    return out.sort_values(keys + ["metric"], kind="mergesort", ignore_index=True)


def leg_counts(legs, *, point):
    """Long frame [point, year, chain_kind, metric, value, count] of F leg rows, each row once: categorical metrics per observed
    value; count metrics (value "") for every (year, chain_kind) group, zeros included."""
    if point not in ("entry", "exit"): raise ValueError(f"point must be 'entry' or 'exit', got {point!r}")
    missing = [c for c in LEG_COLS if c not in legs.columns]
    if missing: raise fvi.FairValueRefusal("column_missing", "leg table lacks required columns", columns=missing)
    if len(legs) == 0: return pd.DataFrame(columns=POINT_COLS)
    legs = legs.reset_index(drop=True)
    year, chain, keys = pd.to_datetime(legs["obs_date"]).dt.year, _text(legs["chain_kind"], "null"), ["year", "chain_kind"]
    parts = [pd.DataFrame({"year": year, "chain_kind": chain, "value": _text(legs[n], null)}).groupby(keys + ["value"]).size()
             .rename("count").reset_index().assign(metric=n)
             for n, null in (("status", "null"), ("source", "none"), ("clamp_side", "null"))]
    mid = (_num(legs["eff_bid"]) + _num(legs["eff_ask"])) / 2
    ind = pd.DataFrame({"year": year, "chain_kind": chain, "rows": 1, **{c: _flag(legs[c]) for c in FLAG_COLS},
                        "annIdiv_known": _num(legs["annIdiv"]).abs().lt(float("inf")).to_numpy(),
                        "annActDiv_known": _num(legs["annActDiv"]).abs().lt(float("inf")).to_numpy(),
                        "better_than_mid_buy": (_num(legs["fill_buy"]) < mid).to_numpy(),
                        "better_than_mid_sell": (_num(legs["fill_sell"]) > mid).to_numpy()})
    counts = ind.groupby(keys).sum().reset_index().melt(id_vars=keys, var_name="metric", value_name="count").assign(value="")
    out = pd.concat([*parts, counts], ignore_index=True).assign(point=point)
    return out[POINT_COLS].sort_values(keys + ["metric", "value"], kind="mergesort", ignore_index=True)


def merge_counts(frames):
    """Sum ``count`` over the other columns of count frames of one shape; all-empty input keeps the columns."""
    if not frames: raise ValueError("merge_counts needs at least one frame")
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty: return frame.iloc[0:0].reset_index(drop=True)
    return frame.groupby([c for c in frame.columns if c != "count"], sort=True, as_index=False)["count"].sum()


def _publish(directory, name, frame):
    """Create-only publish of ``frame`` as ``directory/<name>.parquet``; returns its sha256."""
    final, tmp = directory / f"{name}.parquet", decisions._unique_tmp(directory)
    try:
        frame.to_parquet(tmp, index=False)
        with open(tmp, "r+b") as handle: os.fsync(handle.fileno())
        sha = decisions._file_sha256(tmp)
        try: os.link(tmp, final)
        except FileExistsError: pass
        if decisions._file_sha256(final) != sha:
            raise Refusal("EXPERIMENT_IDENTITY_CONFLICT", f"report input {name} differs", {"reason": "report_input_changed"})
        receipts._fsync_dir(directory)
    finally:
        tmp.unlink(missing_ok=True)
    return sha


def write(out, stage, inputs_hash, scope, tables):
    """Publish each table, then write the stage's receipt once; returns the receipt (failure rules: module docstring)."""
    if stage not in ("inner", "outer"): raise ValueError(f"stage must be 'inner' or 'outer', got {stage!r}")
    if not tables: raise ValueError("write needs a non-empty dict of tables")
    started, out = time.monotonic(), Path(out)
    directory = out / "report_inputs" / stage
    directory.mkdir(parents=True, exist_ok=True)
    for path in (out, out / "report_inputs", directory): receipts._fsync_dir(path)
    entries = {n: {"sha256": _publish(directory, n, tables[n]), "rows": len(tables[n])} for n in sorted(tables)}
    body = {"stage": stage, "inputs_hash": inputs_hash, "scope": scope, "tables": entries}
    receipt = receipts.write_receipt(out / "receipts" / f"report_inputs_{stage}.json", body)
    rows = sum(e["rows"] for e in entries.values())
    print(f"[f_report_inputs] {stage} tables={len(entries)} rows={rows} "
          f"elapsed={time.monotonic() - started:.1f}s", file=sys.stderr)
    return receipt
