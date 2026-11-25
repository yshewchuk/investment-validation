"""EXP-189 arm E per-event choices: the month's arm-E decisions with the marks behind them, one row per event and structure.

Persisted only, no analysis, no gate or objective reads it. ``write_month`` freezes an output month to
``e_choices/<period>.parquet`` (immutable, like ``size_pnl_decisions.freeze_month``); ``write_receipt`` binds a stage's months.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from engine.portfolio import CONTRACT_MULTIPLIER
from experiments import size_pnl_decisions as decisions
from experiments import size_pnl_receipts as receipts
from experiments.size_pnl_shortvol import Refusal
from experiments.size_width_success import LADDER

_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
_OBJECT = ("event_id", "structure", "status", "reason")
ALPHA_COLS = [f"{p}_a{a:.2f}" for a in LADDER for p in ("entry", "exit", "pnl")]
COLS = ["event_id", "event_date", "structure", "k", "exp_pnl_sim", "exp_pnl_sim_k1", "status", "reason", "entry_cost",
        "exit_value", "exit_value_mid", "pnl_dollars", "pnl_dollars_mid", "lots", "debit_dollars", "secured_per_contract",
        *ALPHA_COLS]


def _text(values):
    """``values`` as an object array of str (event ids, structures, statuses)."""
    return np.array([str(v) for v in values], dtype=object)


def _num(values):
    """``values`` as a float64 array; a non-numeric or missing entry is NaN."""
    return pd.to_numeric(pd.Series(list(values), dtype=object), errors="coerce").to_numpy(dtype="float64")


def _by_id(frame, col, ids):
    """``col`` of ``frame`` for each id in ``ids`` as float64: NaN where ``frame`` is None or holds no row for the id."""
    if frame is None:
        return np.full(len(ids), np.nan)
    got = dict(zip(_text(frame["event_id"]), _num(frame[col])))
    return np.array([got.get(i, np.nan) for i in ids], dtype="float64")


def choice_rows(decided, k1, primary, mid, secured, ladder):
    """One row per row of ``decided`` (this month's arm-E decisions of one structure, every status), with the marks of the TAKEN rows.

    ``k1`` is this month's score_events frame at k = 1.0 (None when 1.0 is not in the grid); ``primary`` and ``mid`` are the score
    frames of the taken rows (primary marks, and mid_marks), ``secured`` the secured frame and ``ladder`` {alpha: repriced score frame}
    ({} when the sweep has no ladder). Every mark is NaN for a row that is not taken."""
    ids, status = _text(decided["event_id"]), _text(decided["status"])
    taken = np.array([s == "taken" for s in status], dtype=bool)

    def mark(frame, col):
        return np.where(taken, _by_id(frame, col, ids), np.nan)

    lots, entry = mark(primary, "lots"), _num(decided["entry_cost"])
    cols = {"event_id": ids, "event_date": pd.to_datetime(decided["event_date"]).to_numpy(),
            "structure": _text(decided["structure"]), "k": _num(decided["k"]), "exp_pnl_sim": _num(decided["exp_pnl_sim"]),
            "exp_pnl_sim_k1": _by_id(k1, "exp_pnl_sim", ids), "status": status,
            "reason": np.array(list(decided["reason"]), dtype=object), "entry_cost": entry,
            "exit_value": mark(primary, "exit_value"), "exit_value_mid": mark(mid, "exit_value"),
            "pnl_dollars": mark(primary, "pnl_dollars"), "pnl_dollars_mid": mark(mid, "pnl_dollars"), "lots": lots,
            "debit_dollars": entry * CONTRACT_MULTIPLIER * lots, "secured_per_contract": mark(secured, "secured_per_contract")}
    for a in LADDER:
        lad = ladder.get(a)
        cols[f"entry_a{a:.2f}"] = mark(lad, "entry_cost")
        cols[f"exit_a{a:.2f}"] = mark(lad, "exit_value")
        cols[f"pnl_a{a:.2f}"] = mark(lad, "pnl_dollars")
    return pd.DataFrame(cols, columns=COLS)


def _frame(frames):
    """The month's rows: the concatenated frames, or an empty frame with the dtypes ``choice_rows`` gives."""
    if frames:
        return pd.concat(frames, ignore_index=True)
    dtypes = {c: "object" if c in _OBJECT else "datetime64[ns]" if c == "event_date" else "float64" for c in COLS}
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in dtypes.items()}, columns=COLS)


def write_month(out, period, frames):
    """Freeze one output month's rows to ``out/e_choices/<period>.parquet``, sorted by (structure, event_id), and return its receipt entry.
    Immutable like ``size_pnl_decisions.freeze_month``: a file with other bytes refuses and is left as it was."""
    directory = Path(out) / "e_choices"
    directory.mkdir(exist_ok=True)
    final, tmp = directory / f"{period}.parquet", decisions._unique_tmp(directory)
    try:
        frame = _frame(frames).sort_values(["structure", "event_id"], kind="mergesort")
        frame.to_parquet(tmp, index=False)
        with open(tmp, "r+b") as handle: os.fsync(handle.fileno())
        sha = decisions._file_sha256(tmp)
        if not os.path.lexists(final): os.replace(tmp, final); receipts._fsync_dir(directory)
        elif decisions._file_sha256(final) != sha: raise Refusal(_CONFLICT, f"e_choices month {period} differs from this run's rows", {"reason": "month_freeze_changed"})
    finally:
        tmp.unlink(missing_ok=True)
    return {"period": str(period), "sha256": sha, "rows": len(frame), "taken": int((frame["status"] == "taken").sum())}


def write_receipt(out, stage, inputs_hash, scope, months):
    """The stage's receipt ``receipts/e_choices_<stage>.json`` binding its month files (``months`` from ``write_month``); logs one stderr line."""
    rows, taken = sum(m["rows"] for m in months), sum(m["taken"] for m in months)
    body = {"stage": stage, "inputs_hash": inputs_hash, "scope": scope,
            "months": {m["period"]: {"sha256": m["sha256"], "rows": m["rows"], "taken": m["taken"]} for m in months},
            "rows": rows, "taken": taken}
    receipt = receipts.write_receipt(Path(out) / "receipts" / f"e_choices_{stage}.json", body)
    print(f"[width_choices] {stage} months={len(months)} rows={rows} taken={taken}", file=sys.stderr)
    return receipt


def report_line(inner_receipt, outer_receipt):
    """The T5b markdown section: each stage's rows and taken rows, or 'not persisted' for a stage with no receipt."""
    def part(stage, receipt):
        if receipt is None:
            return f"{stage}: not persisted"
        return f"{stage}: {receipt['rows']} rows ({receipt['taken']} taken)"
    return ("### T5b arm E per-event choices (persisted, not analysed here)\n\n"
            f"{part('inner', inner_receipt)}; {part('outer', outer_receipt)}. "
            "Files e_choices/<month>.parquet, bound by receipts/e_choices_inner.json and receipts/e_choices_outer.json.")
