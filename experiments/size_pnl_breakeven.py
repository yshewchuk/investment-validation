"""EXP-188 Amendment 7: the breakeven alpha of the report (guides/size_decision_pnl_design.md section 18, "Headline marks").

For the SAME taken trades, the P&L is taken at alpha 0 (worst fills) and at alpha 1 (best fills), every leg entry and exit
at that alpha (mid_marks=True). Whole lots are held at the mid-mark lots, so the P&L is linear in alpha, and the breakeven
is ``engine.fills.breakeven_alpha`` on the two totals. Research-only and report-only: it never changes a decision.
"""
import numpy as np
import pandas as pd

from engine.fills import breakeven_alpha
from engine.portfolio import CONTRACT_MULTIPLIER
from experiments.size_pnl_report import md_table

ENDPOINT_COLS = ["pnl_dollars_a0", "pnl_dollars_a1"]
HEADERS = ["model", "book", "rule", "year", "trades", "pnl_a0", "pnl_a1", "breakeven_alpha"]
TITLE = "### Breakeven alpha per model and book (fraction of the bid-ask spread to capture to break even)"
NOTE = ("breakeven_alpha of the totals at alpha 0 and alpha 1 (same trades, lots held at mid); "
        "0.5 = break even at mid; below 0.5 profitable at mid.")
NOT_TESTABLE = "### Breakeven alpha per model and book\n\nnot_testable: breakeven_endpoints_not_in_decisions"


def with_endpoints(frame, taken_a0, taken_a1):
    """``frame`` plus pnl_dollars_a0 / pnl_dollars_a1: each taken row's per-share P&L at alpha 0 / alpha 1 x CONTRACT_MULTIPLIER x
    its lots, merged on event_id (NaN where the event is not taken or not merged)."""
    ids, lots = frame["event_id"].astype(str), frame["lots"].to_numpy(dtype=float)
    out = {}
    for col, taken in zip(ENDPOINT_COLS, (taken_a0, taken_a1)):
        per_share = taken.assign(event_id=taken["event_id"].astype(str)).drop_duplicates("event_id").set_index("event_id")["pnl_per_share"].astype(float)
        out[col] = ids.map(per_share).to_numpy(dtype=float) * CONTRACT_MULTIPLIER * lots
    return frame.assign(**out)


def read(a0, a1):
    """One cell for a pair of totals: the breakeven alpha, or what the two endpoints say when there is none."""
    a0, a1 = float(a0), float(a1)
    if not (np.isfinite(a0) and np.isfinite(a1)):
        return "n/a"
    x = breakeven_alpha(a0, a1)
    if x is not None:
        return f"{x:.3f}"
    if a0 == 0 and a1 == 0:  # a flat line at exactly 0 has no crossing to report
        return "n/a"
    if a0 > 0 or (a0 >= 0 and a1 >= 0):
        return "profitable at alpha 0 (breakeven 0)"
    return "loses even at alpha 1 (no breakeven)"


def section(frame):
    """The breakeven table of the evaluated ``frame`` over its TAKEN rows, one row per (model, book, rule, calendar year); raises ValueError on a lone endpoint column or a non-finite taken endpoint."""
    present = [col in frame.columns for col in ENDPOINT_COLS]
    if not any(present):
        return NOT_TESTABLE
    if not all(present):
        raise ValueError("breakeven endpoints come as a pair: pnl_dollars_a0 and pnl_dollars_a1")
    taken_mask = frame["taken"].astype(bool).to_numpy()
    finite = np.isfinite(frame[ENDPOINT_COLS].to_numpy(dtype=float)).all(axis=1)
    if (taken_mask & ~finite).any():
        raise ValueError("a taken row has a non-finite pnl_dollars_a0 or pnl_dollars_a1")
    if frame[ENDPOINT_COLS].isna().all().all():
        return NOT_TESTABLE
    taken = frame[taken_mask].copy()
    taken["year"] = pd.to_datetime(taken["event_date"]).dt.year
    keys = ["model_id", "book", "arm_or_rule", "year"]
    grouped = taken.groupby(keys)
    totals = pd.DataFrame({"trades": grouped.size(), "a0": grouped["pnl_dollars_a0"].sum(), "a1": grouped["pnl_dollars_a1"].sum()}).reset_index()  # sums ignore NaN
    rows = []
    for rec in totals.sort_values(keys).to_dict("records"):
        a0, a1 = float(rec["a0"]), float(rec["a1"])
        rows.append([rec["model_id"], rec["book"], rec["arm_or_rule"], int(rec["year"]), int(rec["trades"]),
                     f"{a0:.2f}", f"{a1:.2f}", read(a0, a1)])
    return "\n\n".join([TITLE, NOTE, md_table(HEADERS, rows)])
