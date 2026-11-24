"""EXP-189 W6a: the section 7 success table, recommendation order, proposed board rule and T7 rendering. Pure functions."""
import math
import numbers

from engine.evaluate import breakeven_alpha_from_sweep
from experiments.size_pnl_report import md_table
from experiments.size_pnl_shortvol import NAMES
from experiments.size_width_sizing import GRID

PERIODS = ("2025", "2023H2", "2024")
LADDER = (0.0, 0.25, 0.5, 0.75, 1.0)
MARGIN = 0.05
MAX_LOST = 2
PREFERENCE = ("A", "D", "C", "B", "E")
TITLE = "### T7 success table (2025 NOT blind for widths)"
HEADERS = ["arm", "beats 2025", "beats 2023H2", "beats 2024", "breakeven x", "breakeven production",
           "lost structures", "passed", "fails"]
RULE_HEADERS = ["structure", "k", "divisor", "flatness_se", "declined_share"]


def _num(x):
    """``x`` as a float if it is a finite real number (bool and str excluded), else None."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _fmt(x):
    """``x`` to 3 decimals, or n/a for None."""
    return "n/a" if x is None else f"{x:.3f}"


def _get(obj, *keys):
    """``obj`` indexed by ``keys`` in turn; None when a key is missing or a level is not a dict."""
    for k in keys:
        if not isinstance(obj, dict) or k not in obj:
            return None
        obj = obj[k]
    return obj


def breakeven_alpha(sweep):
    """The EXP-187 measure of design section 7: breakeven alpha of mean return per trade over LADDER, or None (not_available).

    ``sweep`` is {alpha: {"mean": mean return per trade}} and must hold every ladder alpha with a finite real mean;
    otherwise None. A crossing of zero is interpolated by ``engine.evaluate.breakeven_alpha_from_sweep``. With no
    crossing: 0.0 if every mean is >= 0 (always positive), 1.0 if any is negative (always negative). All means exactly
    zero is None.
    """
    if not isinstance(sweep, dict):
        return None
    means = []
    for a in LADDER:
        point = sweep.get(a)
        m = _num(point.get("mean")) if isinstance(point, dict) else None
        if m is None:
            return None
        means.append(m)
    if all(m == 0.0 for m in means):
        return None
    x = breakeven_alpha_from_sweep({a: {"mean": m} for a, m in zip(LADDER, means)})
    if x is not None:
        return x
    return 0.0 if min(means) >= 0 else 1.0


def _breakeven_2025(breakevens, label):
    """The 2025 breakeven alpha of ``label``, or None when absent, non-dict or not a finite real number."""
    return _num(_get(breakevens, label, "2025"))


def _lost(arm, production):
    """Structures whose 2025 total_return in ``arm`` is below production's. A structure whose return is missing or
    not a finite number (NaN included) in either counts as lost; all of them count when 2025 structures are missing."""
    a, p = _get(arm, "2025", "structures"), _get(production, "2025", "structures")
    if not isinstance(a, dict) or not isinstance(p, dict):
        return len(NAMES)
    lost = 0
    for n in NAMES:
        arm_ret = _num(_get(a, n, "total_return"))
        prod_ret = _num(_get(p, n, "total_return"))
        if arm_ret is None or prod_ret is None or arm_ret < prod_ret:
            lost += 1
    return lost


def _row(label, arm, production, breakevens, e_summary):
    """One arm's success row: periods beaten, the 2025 breakeven margin, the structure guard and the same-k flag."""
    fails, beats = [], {}
    for p in PERIODS:
        arm_ret = _num(_get(arm, p, "account", "total_return"))
        prod_ret = _num(_get(production, p, "account", "total_return"))
        if arm_ret is None or prod_ret is None:
            beats[p] = False
            fails.append(f"missing_{p}")
            continue
        beats[p] = arm_ret > prod_ret
        if not beats[p]:
            fails.append(f"not_beat_{p}")
    x = _breakeven_2025(breakevens, label)
    base = _breakeven_2025(breakevens, "production")
    margin_ok = x is not None and base is not None and x <= base + MARGIN
    if x is None or base is None:
        fails.append("breakeven_not_available")
    elif not margin_ok:
        fails.append("breakeven_margin")
    lost = _lost(arm, production)
    guard_ok = lost <= MAX_LOST
    if not guard_ok:
        fails.append("guard")
    e_ok = True
    same_k = False
    if label == "E":
        complete = (isinstance(e_summary, dict) and set(e_summary) == set(NAMES)
                    and all(isinstance(v, dict) and isinstance(v.get("same_k"), bool) for v in e_summary.values()))
        if not complete:
            e_ok = False
            fails.append("e_summary_missing")
        else:
            same_k = any(v["same_k"] for v in e_summary.values())
            if same_k:
                fails.append("same_k")
    passed = all(beats.values()) and margin_ok and guard_ok and e_ok and not same_k
    return {"beats": beats, "breakeven": {"x": x, "production": base, "margin_ok": margin_ok},
            "lost_structures": lost, "guard_ok": guard_ok, "same_k": same_k, "passed": passed, "fails": fails}


def success_table(production, arms, breakevens, e_summary=None):
    """{arm: row} of the section 7 success rule for each arm in ``arms`` against production; ValueError unless both are dicts."""
    if not isinstance(production, dict) or not isinstance(arms, dict):
        raise ValueError("production and arms must be dicts")
    return {label: _row(label, arm, production, breakevens, e_summary) for label, arm in arms.items()}


def recommend(table):
    """The first arm in PREFERENCE whose row passed, or None (recommend no change)."""
    return next((a for a in PREFERENCE if a in table and table[a]["passed"]), None)


def _check_factors(factors):
    """ValueError unless ``factors`` is a dict holding every structure in NAMES with a finite real k in GRID (bool excluded)."""
    if not isinstance(factors, dict):
        raise ValueError("factors must be a dict of structure -> k")
    for n in NAMES:
        if n not in factors:
            raise ValueError(f"missing factor for structure {n!r}")
        k = factors[n]
        kf = None if isinstance(k, bool) else _num(k)
        if kf is None:
            raise ValueError(f"factor for structure {n!r} must be a finite real number, got {k!r}")
        if kf not in GRID:
            raise ValueError(f"factor for structure {n!r} is {k!r}, not in size_width_sizing.GRID {GRID}")


def _check_values(name, d):
    """ValueError unless ``d`` is a dict whose values are each None or a finite real (bool excluded)."""
    if not isinstance(d, dict):
        raise ValueError(f"{name} must be a dict of structure -> value")
    for n, v in d.items():
        if v is not None and _num(v) is None:
            raise ValueError(f"{name} for structure {n!r} must be a finite real number or None, got {v!r}")


def board_rule(label, factors, flatness, declined):
    """The proposed board width rule: divisors per structure for A-D, the expectation rule for E, or no change.

    ValueError for a label outside None and PREFERENCE. For A-D, ``factors`` must hold every structure in NAMES with
    a k in GRID, and ``flatness`` and ``declined`` must be dicts whose values are None or finite reals.
    """
    if label is None:
        return {"recommendation": "no_change", "rows": []}
    if label not in PREFERENCE:
        raise ValueError(f"unknown arm {label!r}: label must be None or one of {PREFERENCE}")
    if label == "E":
        return {"recommendation": "E", "rule": "pick k maximising exp_pnl_sim per event", "rows": []}
    _check_factors(factors)
    _check_values("flatness", flatness)
    _check_values("declined", declined)
    rows = [{"structure": n, "k": factors[n], "divisor": f"divisor / {factors[n]}",
             "flatness_se": flatness.get(n), "declined_share": declined.get(n)} for n in NAMES]
    return {"recommendation": label, "rows": rows}


def render_t7(table):
    """T7 as markdown: the success table over its arms, then one recommendation line."""
    rows = [[label, r["beats"]["2025"], r["beats"]["2023H2"], r["beats"]["2024"], _fmt(r["breakeven"]["x"]),
             _fmt(r["breakeven"]["production"]), r["lost_structures"], r["passed"], ";".join(r["fails"])]
            for label, r in table.items()]
    rec = recommend(table)
    line = "Recommendation: no change recommended." if rec is None else f"Recommendation: arm {rec}."
    return "\n\n".join([TITLE, md_table(HEADERS, rows), line])


def render_rule(rule):
    """The board rule as markdown: a divisor table for A-D, a sentence for E, or the no-change line.

    ValueError unless ``rule`` is a dict with a known recommendation; an A-D rule also needs a list ``rows`` of
    dicts holding every key in RULE_HEADERS.
    """
    rec = rule.get("recommendation") if isinstance(rule, dict) else None
    if not isinstance(rule, dict):
        raise ValueError("rule must be a dict")
    if rec not in ("no_change", "E") + PREFERENCE:
        raise ValueError(f"unknown recommendation {rec!r}: expected no_change, E or one of {PREFERENCE}")
    if rec == "no_change":
        return "No change recommended."
    if rec == "E":
        return f"Recommendation: E, {rule['rule']}."
    rows_in = rule.get("rows")
    if not isinstance(rows_in, list) or not all(isinstance(r, dict) and all(k in r for k in RULE_HEADERS)
                                                for r in rows_in):
        raise ValueError(f"arm {rec} rule needs a list 'rows' of dicts with keys {', '.join(RULE_HEADERS)}")
    rows = [[r["structure"], _fmt(r["k"]), r["divisor"], _fmt(r["flatness_se"]), _fmt(r["declined_share"])]
            for r in rows_in]
    return "\n\n".join([f"### Proposed board width rule (arm {rec})", md_table(RULE_HEADERS, rows)])
