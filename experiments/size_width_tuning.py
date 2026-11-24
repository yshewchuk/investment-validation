"""EXP-189 W4: objective, tie rule and tunings A, B, D, arm E's summary, tuning receipts, 2025 evaluation (design sections 3, 5, 6).

All tuning reads the inner window (2023H2-2024) of the sweep only. Objective: stand-alone cash-secured $200,000 account return at
marks_mid. Standard error: sd of quarterly returns / sqrt(quarter count) (0.0 with fewer than two quarters).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import size_pnl_receipts as receipts
from experiments import size_width_account as account
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_sizing import GRID
from experiments.size_width_sweep import E_ARM, INNER, WINDOWS

MIN_TRADES = 300
EDGES = (min(GRID), max(GRID))
TWIN = ("TWIN-P", "TWIN-P5")
SAME_K_SHARE = 0.8
_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"


def _quarters(window):
    return pd.period_range(window[0], window[1] - pd.Timedelta(days=1), freq="Q")


def unit_score(rows, window):
    """{"ret", "se", "trades"} of one account over the taken ``rows`` of ``window`` at marks_mid."""
    q = account.quarter_returns(rows, _quarters(window))
    se = float(np.std(q, ddof=1) / math.sqrt(len(q))) if len(q) > 1 else 0.0
    return {"ret": float(account.simulate_account(rows, "mid")["total_return"]), "se": se, "trades": int(len(rows))}


def _pick(scored):
    """Tie rule: among k within one standard error of the best return, the one nearest 1, then the smaller."""
    best = max(scored.values(), key=lambda s: s["ret"])
    near = [k for k, s in scored.items() if s["ret"] >= best["ret"] - best["se"]]
    return min(near, key=lambda k: (abs(k - 1.0), k))


def choose(scores):
    """{"k", "flags"} from {k: unit_score}: a k under MIN_TRADES is not scored; none scored -> k 1.0 "k_defaulted"; a pick on a grid edge -> "k_at_edge" and the best interior k instead."""
    scored = {k: s for k, s in scores.items() if s["trades"] >= MIN_TRADES}
    if not scored:
        return {"k": 1.0, "flags": ["k_defaulted"]}
    k = _pick(scored)
    if k not in EDGES:
        return {"k": float(k), "flags": []}
    interior = {j: s for j, s in scored.items() if j not in EDGES}
    if not interior:
        return {"k": 1.0, "flags": ["k_at_edge", "k_defaulted"]}
    return {"k": float(_pick(interior)), "flags": ["k_at_edge"]}


def units(arm):
    """Unit name -> member structures: A one account, B one per structure, D the TWIN pair and the five centre-peaked structures."""
    return {"A": {"all": NAMES}, "B": {n: (n,) for n in NAMES},
            "D": {"T": TWIN, "C": tuple(n for n in NAMES if n not in TWIN)}}[arm]


def tune(sweep, arm, window=INNER):
    """Arm A, B or D on the inner window: {unit: {"k", "flags", "members", "scores": {str(k): {ret, se, trades, not_scored}}}}."""
    out = {}
    for unit, members in units(arm).items():
        scores = {k: unit_score(sweep.rows([(n, k) for n in members], window), window) for k in sweep.ks}
        out[unit] = {**choose(scores), "members": list(members),
                     "scores": {str(k): {**s, "not_scored": s["trades"] < MIN_TRADES} for k, s in scores.items()}}
    return out


def factors(arm, body):
    """structure -> k (a float) from a tuned body; arm E maps every structure to E_ARM."""
    if arm == E_ARM:
        return {n: E_ARM for n in NAMES}
    return {n: unit["k"] for unit in body.values() for n in unit["members"]}


def e_summary(sweep, window=INNER):
    """Arm E needs no tuning. Per structure: the share of candidate events at each chosen k, the modal share, ``same_k`` (modal share >= SAME_K_SHARE: E then equals a global factor and fails), and the mean exp_pnl_sim by k and for E (T3)."""
    out = {}
    for name in NAMES:
        rows = sweep.scored_rows(name, E_ARM, window)
        share = rows["k"].value_counts(normalize=True) if not rows.empty else pd.Series(dtype=float)
        modal = float(share.max()) if len(share) else 0.0
        out[name] = {"k_share": {str(k): float(v) for k, v in share.items()}, "modal_share": modal,
                     "same_k": modal >= SAME_K_SHARE, "candidates": int(len(rows)), "curve": sweep.curves(name, window)}
    return out


def evaluate(sweep, by_structure, window):
    """Seven-structure $200,000 account at marks_mid and each structure's stand-alone account in ``window``; ``by_structure``: structure -> k or E_ARM."""
    pairs = [(n, by_structure[n]) for n in NAMES]
    per = {}
    for pair in pairs:
        one = sweep.rows([pair], window)
        per[pair[0]] = {**account.simulate_account(one, "mid"), "trades": int(len(one))}
    rows = sweep.rows(pairs, window)
    return {"account": {**account.simulate_account(rows, "mid"), "trades": int(len(rows))}, "structures": per}


def _path(out, arm):
    return Path(out) / "receipts" / f"tuning_{arm}.json"


def run_inner(sweep, out, inputs_hash):
    """Tune A, B, D and summarise E on 2023-24, write tuning_A/B/D/E receipts (atomic, immutable, bound to ``inputs_hash``), and return
    {"tuned": ..., "results": {arm_or_production: {"2023H2": evaluate, "2024": evaluate}}}."""
    tuned = {arm: tune(sweep, arm) for arm in ("A", "B", "D")}
    tuned["E"] = e_summary(sweep)
    for arm, body in tuned.items():
        receipts.write_receipt(_path(out, arm), {"experiment": "EXP-189", "arm": arm, "inputs_hash": inputs_hash,
                                                 "window": "2023H2-2024", "body": body})
    labels = {"production": {n: 1.0 for n in NAMES}, **{arm: factors(arm, tuned[arm]) for arm in ("A", "B", "D", "E")}}
    results = {label: {w: evaluate(sweep, by, WINDOWS[w]) for w in ("2023H2", "2024")} for label, by in labels.items()}
    return {"tuned": tuned, "results": results}


def frozen(out, arm, inputs_hash):
    """The factors of ``arm`` from its receipt. A missing receipt refuses tuning_receipt_missing (load_receipt); one bound to other inputs refuses tuning_receipt_changed."""
    receipt = receipts.load_receipt(_path(out, arm))
    if receipt["inputs_hash"] != inputs_hash:
        raise Refusal(_CONFLICT, f"tuning receipt {arm} is bound to other inputs", {"reason": "tuning_receipt_changed"})
    return factors(arm, receipt["body"])


def run_outer(sweep, out, arms, inputs_hash, window=WINDOWS["2025"]):
    """2025 (2025 NOT blind for widths): every arm's receipt is loaded first (a missing one refuses before any account is walked), then production
    (k = 1) and each arm are evaluated on its frozen factors. Returns {label: evaluate}."""
    by = {arm: frozen(out, arm, inputs_hash) for arm in arms}
    by = {"production": {n: 1.0 for n in NAMES}, **by}
    return {label: evaluate(sweep, f, window) for label, f in by.items()}
