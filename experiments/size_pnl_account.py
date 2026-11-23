"""EXP-188 Amendment 2: cash-secured $100,000 account sensitivity (guides/size_decision_pnl_design.md section 15).

Research-only and report-only: it reads the taken rows of decisions.parquet and produces sensitivity tables. It is
not the pre-registered verdict, it computes no interval, and it never changes a decision. The margin rule is EXP-134's
``margin.py``, loaded by path and never altered: every short put is secured in full at strike x 100, long puts secure
nothing, and the account is walked by ``margin.simulate`` under POLICY (cap 66% of marked equity, target share 25% of
headroom, expensive-first on entry-date ties). Returns are quoted on the debit, as in EXP-134.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.size_pnl_report import _md_table

_MARGIN = Path(__file__).resolve().parents[1] / "experiments" / "EXP-134_priced_right_funded_and_held_structure_s" / "margin.py"


def _load(name, path):
    """Import a module by PATH (as EXP-136's run.py does), so the EXP-134 file is used unchanged."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


margin = _load("exp188_margin", _MARGIN)

START_EQUITY = 100_000.0
POLICY = dict(start_equity=100_000.0, cap=0.66, target_share=0.25, expensive_first=True)
NOTE = ("Policy: start $100,000, secured cap 66% of marked equity, target share 25% of headroom, expensive-first on entry-date ties; "
        "returns are quoted on the debit (EXP-134 margin rule, long puts secure nothing). No interval is computed, and no contrast is a verdict.")
_ACCOUNT_HEADERS = ["kind", "model", "key", "wanted", "funded", "final_equity", "total_return", "max_drawdown", "peak_open"]


def secured_per_contract(legs):
    """Cash one contract of these resolved legs ties up: EXP-134's rule on the traded legs (qty > 0), reference legs excluded."""
    return margin.secured_per_contract({"entry": [{"side": leg["side"], "qty": leg["qty"], "strike": leg["strike"]}
                                                  for leg in legs if leg["qty"] > 0]})


def simulate_account(rows, mark):
    """Metrics of one account over the TAKEN ``rows``; ``mark`` "mid" walks the Amendment 1 exit marks instead of the primary ones."""
    start = POLICY["start_equity"]
    if rows.empty:
        return {"wanted": 0, "funded": 0, "final_equity": start, "total_return": 0.0, "max_drawdown": 0.0, "peak_open": 0}
    exit_col = "exit_value_mid" if mark == "mid" else "exit_value"
    for col in ("entry_cost", "secured_per_contract", exit_col):  # a missing or non-finite input is refused, never simulated
        if col not in rows.columns or not np.isfinite(rows[col].to_numpy(dtype=float)).all():
            raise ValueError(f"account inputs must be finite: {col}")
    for col in ("entry_date", "exit_date"):  # a missing or unparseable date, or NaT for any row, is refused, never simulated
        try:
            parsed = pd.to_datetime(rows[col])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("account inputs must carry valid entry_date and exit_date") from exc
        if parsed.isna().any():
            raise ValueError("account inputs must carry valid entry_date and exit_date")
    frame = rows.drop(columns=["exit_value"]).rename(columns={"exit_value_mid": "exit_value"}) if mark == "mid" else rows
    t = margin.simulate(frame[["entry_date", "exit_date", "entry_cost", "exit_value", "secured_per_contract"]], **POLICY)
    final = float(t.attrs["final_equity"])
    done, max_dd = t[t["funded"]], 0.0
    if not done.empty:  # the realised curve at EXIT DATES: same-date P&L is summed before any peak is taken
        per_date = done.assign(exit_date=pd.to_datetime(done["exit_date"])).groupby("exit_date")["pnl_usd"].sum().sort_index()
        curve = start + np.concatenate([[0.0], np.cumsum(per_date.to_numpy(dtype=float))])
        max_dd = float(np.min(curve / np.maximum.accumulate(curve) - 1.0))
    return {"wanted": len(t), "funded": int(t["funded"].sum()), "final_equity": final, "total_return": final / start - 1.0,
            "max_drawdown": max_dd, "peak_open": int(t["concurrency"].max()) if len(t) else 0}


def straddle_rows(rows):
    """Copy with the straddle's secured cash = debit x 100 (the cash a long straddle needs; a judgement call, design section 15)."""
    out = rows.copy()
    out["secured_per_contract"] = out["entry_cost"].astype(float) * 100.0
    return out


def accounts(taken, universe=None):
    """Account metrics per (mark, kind, model, key). ``taken`` holds the taken rows. ``universe`` holds ALL decision rows of
    any status and discovers the account keys, so a valid account that takes no trade is reported with its empty metrics;
    None takes the keys from ``taken``. The mid mark is skipped when the exit_value_mid column is absent or all NaN."""
    universe = taken if universe is None else universe
    marks = ["primary"]
    if "exit_value_mid" in taken.columns and taken["exit_value_mid"].notna().any():
        marks.append("mid")
    short, long = taken[taken["book"] == "short_vol"], straddle_rows(taken[taken["book"] == "straddle"])
    plans = [("book", "short_vol", short, ["model_id", "arm_or_rule"], lambda k: k[1]),
             ("structure", "short_vol", short, ["model_id", "arm_or_rule", "structure"], lambda k: f"{k[1]}/{k[2]}"),
             ("straddle", "straddle", long, ["model_id", "arm_or_rule"], lambda k: k[1])]
    out = []
    for mark in marks:
        for kind, book, rows, keys, key_of in plans:
            keyed = universe[universe["book"] == book]
            taken_by_key = {k: g for k, g in rows.groupby(keys)}
            for k, _ in keyed.groupby(keys):  # keys from the universe; a key with no taken row simulates the empty account
                group = taken_by_key.get(k, rows.iloc[:0])
                out.append({"mark": mark, "kind": kind, "model": k[0], "key": key_of(k), **simulate_account(group, mark)})
    return out


def differences(results, pairs, labels):
    """Paired contrasts of final_equity. ``pairs`` maps a pair id to (mean_label, median_label). Each row has a = the mean
    (or M0) value, b = the median (or other) value, and diff = b - a: for a pair, median minus mean; for M0 versus another
    label, other minus M0. Positive diff means the median-type (or the other) model did better. A contrast with a missing side is skipped."""
    value = {(r["mark"], r["kind"], r["key"], r["model"]): r["final_equity"] for r in results}
    out = []
    for mark, kind, key in dict.fromkeys((r["mark"], r["kind"], r["key"]) for r in results):
        contrasts = [(pid, mean, median) for pid, (mean, median) in pairs.items()]
        contrasts += [("M0-vs-" + label, "M0", label) for label in labels if label != "M0"]
        for name, a_label, b_label in contrasts:
            a, b = value.get((mark, kind, key, a_label)), value.get((mark, kind, key, b_label))
            if a is None or b is None:
                continue
            out.append({"mark": mark, "kind": kind, "key": key, "contrast": name, "a": a, "b": b, "diff": b - a})
    return out


def tables(results, diffs):
    """Markdown sections, one per mark present: the account table and the differences table."""
    out = []
    for mark in ("primary", "mid"):
        rows = [r for r in results if r["mark"] == mark]
        if not rows:
            continue
        title = f"### Sensitivity (Amendment 2), not the pre-registered verdict: cash-secured $100,000 accounts, {mark} marks"
        # Dollar values are pre-formatted with cents: _cell would render a float at 4 significant digits (100049.0 -> "1e+05").
        account = _md_table(_ACCOUNT_HEADERS, [[f"{r[h]:.2f}" if h == "final_equity" else r[h] for h in _ACCOUNT_HEADERS] for r in rows])
        diff_rows = [[f"{d[h]:.2f}" if h == "diff" else d[h] for h in ("mark", "kind", "key", "contrast", "diff")] for d in diffs if d["mark"] == mark]
        out.append("\n\n".join([title, NOTE, account, _md_table(["mark", "kind", "key", "contrast", "diff"], diff_rows)]))
    return out
