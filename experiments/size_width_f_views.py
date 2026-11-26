"""EXP-189 Amendment 1, slice W7c1: beside-F tables and the POST-HOC label (design A1.1, A1.2, A1.10).

f_views is {label: {period: {view: cell}}} (inner 2023H2, 2024; outer 2025), built by view_tables from an F sweep.
No run stage calls view_tables yet (issue #2996); the report refuses an F run's receipts until then."""
import math
import resource
import sys
import time

import numpy as np
import pandas as pd

from experiments.fair_value_fills import VARIANTS
from experiments.size_pnl_report import md_table
from experiments.size_pnl_shortvol import NAMES
from experiments.size_width_account import simulate_account
from experiments.size_width_sweep import E_ARM, admit_k

LABEL = "POST-HOC (Amendment 1, F fills)"
ACCOUNT_VIEWS = ("F", "mid", "harsh")
VIEWS = (*ACCOUNT_VIEWS, *VARIANTS)
VIEW_COLS = {"F": ("entry_cost", "exit_value"), "mid": ("entry_cost_mid", "exit_value_mid"), "harsh": ("entry_cost_harsh", "exit_value_harsh"), **{v: (f"entry_cost_{v}", f"exit_value_{v}") for v in VARIANTS}}


def _frame(rows, view):
    """``rows`` as ``view``'s account frame: its own debit (finite and > 0, else ValueError) and its own exit."""
    debit, exit_col = VIEW_COLS[view]
    cost = pd.to_numeric(rows[debit], errors="coerce").to_numpy(dtype=float)
    if not (np.isfinite(cost) & (cost > 0)).all(): raise ValueError(f"view {view} debit must be finite and > 0")
    return rows[["entry_date", "exit_date", "secured_per_contract"]].assign(entry_cost=cost, exit_value=pd.to_numeric(rows[exit_col], errors="coerce").to_numpy(dtype=float))


def _cell(sweep, pairs, window, rows, view):
    """One view's cell over ``rows``; the ACCOUNT_VIEWS add each structure's own stand-alone account."""
    frame = _frame(rows, view)
    per_trade = None if frame.empty else float(((frame["exit_value"] - frame["entry_cost"]) / frame["entry_cost"]).mean())
    cell = {"trades": len(frame), "return_per_trade": per_trade, "account_return": simulate_account(frame, "primary")["total_return"]}
    if view in ACCOUNT_VIEWS: cell["structures"] = {n: simulate_account(_frame(sweep.rows([(n, arm)], window), view), "primary")["total_return"] for n, arm in pairs}
    return cell


def view_tables(sweep, by_label, windows):
    """{label: {period: {view: cell}}} from an F sweep; one taken set per label and period, shared by every view."""
    if not sweep.f: raise ValueError("view_tables needs an F sweep")
    out, total, done, start = {}, len(by_label) * len(windows), 0, time.monotonic()
    for label, by in by_label.items():
        pairs = [(n, by[n] if by[n] == E_ARM else admit_k(by[n])) for n in NAMES]
        out[label] = {}
        for period, window in windows.items():
            rows = sweep.rows(pairs, window)
            out[label][period] = {view: _cell(sweep, pairs, window, rows, view) for view in VIEWS}
            done += 1
            elapsed = time.monotonic() - start
            print(f"[width_f_views] {label} {period} trades={len(rows)} done {done}/{total} " f"rate={60 * done / elapsed:.2f}/min eta={(total - done) * elapsed / done:.0f}s elapsed={elapsed:.1f}s "
                  f"rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}MB", file=sys.stderr)
    return out


def _real(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def admit(block, results, labels, periods):
    """True only when ``block`` holds exactly the views of ``labels`` and ``periods`` and its F cells agree with ``results``."""
    try:
        if not isinstance(block, dict) or set(block) != set(labels): return False
        for label in labels:
            if set(block[label]) != set(periods): return False
            for period in periods:
                cells = block[label][period]
                if set(cells) != set(VIEWS) or len({cells[v]["trades"] for v in VIEWS}) != 1: return False
                for view in VIEWS:
                    cell, t = cells[view], cells[view]["trades"]
                    if isinstance(t, bool) or not isinstance(t, int) or t < 0 or not _real(cell["account_return"]): return False
                    if (cell["return_per_trade"] is None) != (t == 0) or (t > 0 and not _real(cell["return_per_trade"])): return False
                    if view in ACCOUNT_VIEWS and (set(cell["structures"]) != set(NAMES) or not all(map(_real, cell["structures"].values()))): return False
                f, res = cells["F"], results[label][period]
                if f["trades"] != res["account"]["trades"] or not math.isclose(f["account_return"], res["account"]["total_return"], rel_tol=1e-12, abs_tol=1e-12): return False
                if not all(math.isclose(f["structures"][n], res["structures"][n]["total_return"], rel_tol=1e-12, abs_tol=1e-12) for n in NAMES): return False
        return True
    except (KeyError, TypeError, AttributeError):
        return False


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}"


def t5_markdown(labels, periods, views, stats):
    """T5 at F fills (the deciding cell), then the beside-F account and per-trade tables by view."""
    keys = [(label, p) for label in labels for p in periods]
    deciding = [[label, p, views[label][p]["F"]["trades"], _pct(views[label][p]["F"]["account_return"]),
                 _pct(views[label][p]["F"]["return_per_trade"]), "n/a" if stats[label][p]["x"] is None else f"{stats[label][p]['x']:.3f}"]
                for label, p in keys]
    def section(title, key): return f"### {title}\n\n" + md_table(["arm", "period", *VIEWS], [[label, p, *(_pct(views[label][p][v][key]) for v in VIEWS)] for label, p in keys])
    return "\n\n".join([
        "### T5 production versus arms ($200,000 account at F fills; 2025 NOT blind for widths)\n\n"
        + md_table(["arm", "period", "trades", "account return %", "return per trade %", "breakeven alpha"], deciding),
        section("T5 beside F: seven-structure $200,000 account return % by view (reported, never deciding; 2025 NOT blind for widths)", "account_return"),
        section("T5 beside F: return per trade % by view (reported, never deciding; 2025 NOT blind for widths)", "return_per_trade")])


def t6_markdown(labels, views):
    """T6 per-structure stand-alone accounts in 2025 at mid and at harsh (reported, never deciding)."""
    return "\n\n".join(
        f"### T6 beside F: per-structure stand-alone accounts at {view} (2025, return %; reported, never deciding; 2025 NOT blind for widths)\n\n"
        + md_table(["structure", *labels], [[n, *(_pct(views[label]["2025"][view]["structures"][n]) for label in labels)] for n in NAMES])
        for view in ("mid", "harsh"))


def label_headings(text):
    """``text`` with the label appended to every heading line (a line starting with "#")."""
    return "\n".join(f"{line} [{LABEL}]" if line.startswith("#") else line for line in text.split("\n"))
