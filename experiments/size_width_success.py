"""EXP-189 W6a: the section 7 success table, recommendation order, proposed board rule and T7 rendering. Pure functions."""
import math

from experiments import size_pnl_breakeven as breakeven
from experiments.size_pnl_report import md_table
from experiments.size_pnl_shortvol import NAMES

PERIODS = ("2025", "2023H2", "2024")
MARGIN = 0.05
MAX_LOST = 2
PREFERENCE = ("A", "D", "C", "B", "E")
TITLE = "### T7 success table (2025 NOT blind for widths)"
HEADERS = ["arm", "beats 2025", "beats 2023H2", "beats 2024", "breakeven x", "breakeven production",
           "lost structures", "passed", "fails"]
RULE_HEADERS = ["structure", "k", "divisor", "flatness_se", "declined_share"]


def _num(x):
    """``x`` to 3 decimals, or n/a for None."""
    return "n/a" if x is None else f"{x:.3f}"


def breakeven_alpha(a0, a1, trades):
    """Breakeven alpha of the alpha-0 and alpha-1 totals over ``trades`` taken trades, or None (not_available)."""
    if trades == 0 or not (math.isfinite(a0) and math.isfinite(a1)) or (a0 == 0 and a1 == 0):
        return None
    x = breakeven.breakeven_alpha(a0, a1)
    if x is not None:
        return x
    return 0.0 if a0 > 0 or (a0 >= 0 and a1 >= 0) else 1.0


def _lost(arm, production):
    """Structures whose 2025 total_return in ``arm`` is below production's; all of them when 2025 is missing."""
    if "2025" not in arm or "2025" not in production:
        return len(NAMES)
    a, p = arm["2025"]["structures"], production["2025"]["structures"]
    return sum(a[n]["total_return"] < p[n]["total_return"] for n in NAMES)


def _row(label, arm, production, breakevens, e_summary):
    """One arm's success row: periods beaten, the 2025 breakeven margin, the structure guard and the same-k flag."""
    fails, beats = [], {}
    for p in PERIODS:
        if p not in arm or p not in production:
            beats[p] = False
            fails.append(f"missing_{p}")
            continue
        beats[p] = arm[p]["account"]["total_return"] > production[p]["account"]["total_return"]
        if not beats[p]:
            fails.append(f"not_beat_{p}")
    x = breakevens.get(label, {}).get("2025")
    base = breakevens.get("production", {}).get("2025")
    margin_ok = x is not None and base is not None and x <= base + MARGIN
    if x is None or base is None:
        fails.append("breakeven_not_available")
    elif not margin_ok:
        fails.append("breakeven_margin")
    lost = _lost(arm, production)
    guard_ok = lost <= MAX_LOST
    if not guard_ok:
        fails.append("guard")
    same_k = bool(e_summary) and label == "E" and any(v["same_k"] for v in e_summary.values())
    if same_k:
        fails.append("same_k")
    passed = all(beats.values()) and margin_ok and guard_ok and not same_k
    return {"beats": beats, "breakeven": {"x": x, "production": base, "margin_ok": margin_ok},
            "lost_structures": lost, "guard_ok": guard_ok, "same_k": same_k, "passed": passed, "fails": fails}


def success_table(production, arms, breakevens, e_summary=None):
    """{arm: row} of the section 7 success rule for each arm in ``arms`` against production."""
    return {label: _row(label, arm, production, breakevens, e_summary) for label, arm in arms.items()}


def recommend(table):
    """The first arm in PREFERENCE whose row passed, or None (recommend no change)."""
    return next((a for a in PREFERENCE if a in table and table[a]["passed"]), None)


def board_rule(label, factors, flatness, declined):
    """The proposed board width rule: divisors per structure for A-D, the expectation rule for E, or no change."""
    if label is None:
        return {"recommendation": "no_change", "rows": []}
    if label == "E":
        return {"recommendation": "E", "rule": "pick k maximising exp_pnl_sim per event", "rows": []}
    rows = [{"structure": n, "k": factors[n], "divisor": f"divisor / {factors[n]}",
             "flatness_se": flatness.get(n), "declined_share": declined.get(n)} for n in NAMES]
    return {"recommendation": label, "rows": rows}


def render_t7(table):
    """T7 as markdown: the success table over its arms, then one recommendation line."""
    rows = [[label, r["beats"]["2025"], r["beats"]["2023H2"], r["beats"]["2024"], _num(r["breakeven"]["x"]),
             _num(r["breakeven"]["production"]), r["lost_structures"], r["passed"], ";".join(r["fails"])]
            for label, r in table.items()]
    rec = recommend(table)
    line = "Recommendation: no change recommended." if rec is None else f"Recommendation: arm {rec}."
    return "\n\n".join([TITLE, md_table(HEADERS, rows), line])


def render_rule(rule):
    """The board rule as markdown: a divisor table for A-D, a sentence for E, or the no-change line."""
    if rule["recommendation"] == "no_change":
        return "No change recommended."
    if rule["recommendation"] == "E":
        return f"Recommendation: E, {rule['rule']}."
    rows = [[r["structure"], _num(r["k"]), r["divisor"], _num(r["flatness_se"]), _num(r["declined_share"])]
            for r in rule["rows"]]
    return "\n\n".join([f"### Proposed board width rule (arm {rule['recommendation']})", md_table(RULE_HEADERS, rows)])
