"""EXP-189 W4: tuning A, B, D on the inner window, arm E summary, receipts, 2025 evaluation (design sections 3, 5, 6)."""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import size_pnl_receipts as receipts
from experiments import size_width_account as account
from experiments.size_pnl_shortvol import NAMES, Refusal
from experiments.size_width_errmodel import MIN_EVENTS
from experiments.size_width_sizing import GRID
from experiments.size_width_sweep import E_ARM, INNER, WINDOWS, admit_k

MIN_TRADES, SAME_K_SHARE = 300, 0.8
FAKE_DRAWS = 20
FAKE_SEED_TAG = "exp189-v1"
EDGES = (min(GRID), max(GRID))
TWIN = ("TWIN-P", "TWIN-P5")
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
    top = max(scored, key=lambda j: scored[j]["ret"])
    k = _pick(scored)
    if top not in EDGES and k not in EDGES:
        return {"k": float(k), "flags": []}
    interior = {j: s for j, s in scored.items() if j not in EDGES}
    if not interior:
        return {"k": 1.0, "flags": ["k_at_edge", "k_defaulted"]}
    return {"k": float(_pick(interior)), "flags": ["k_at_edge"]}


def units(arm):
    """Unit name -> member structures: A one account, B and C one per structure, D the TWIN pair and the five centre-peaked structures."""
    return {"A": {"all": NAMES}, "B": {n: (n,) for n in NAMES}, "C": {n: (n,) for n in NAMES},
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
    return {n: E_ARM for n in NAMES} if arm == E_ARM else {n: unit["k"] for unit in body.values() for n in unit["members"]}


def e_summary(sweep, window=INNER):
    """Arm E needs no tuning. Per structure: the share of ALL candidate events of the window (scored or not, so events with no feasible k count in the denominator) at each chosen k, the modal share, ``same_k`` (modal share >= SAME_K_SHARE: E then equals a global factor and fails), the candidate and no-feasible-k counts, and the mean exp_pnl_sim by k and for E (T3)."""
    out = {}
    for name in NAMES:
        everything = sweep.scored_rows(name, E_ARM, window, scored_only=False)
        rows = sweep.scored_rows(name, E_ARM, window)
        share = rows["k"].value_counts() / len(everything) if len(everything) else pd.Series(dtype=float)
        modal = float(share.max()) if len(share) else 0.0
        out[name] = {"k_share": {str(k): float(v) for k, v in share.items()}, "modal_share": modal,
                     "same_k": modal >= SAME_K_SHARE, "candidates": int(len(everything)),
                     "no_feasible_k": int(len(everything) - len(rows)), "curve": sweep.curves(name, window)}
    return out


def evaluate(sweep, by_structure, window):
    """Seven-structure $200,000 account at marks_mid and each structure's own account in ``window``; ``by_structure``: structure -> k or E_ARM."""
    pairs = [(n, by_structure[n] if by_structure[n] == E_ARM else admit_k(by_structure[n])) for n in NAMES]
    per = {}
    for pair in pairs:
        one = sweep.rows([pair], window)
        per[pair[0]] = {**account.simulate_account(one, "mid"), "trades": int(len(one))}
    rows = sweep.rows(pairs, window)
    return {"account": {**account.simulate_account(rows, "mid"), "trades": int(len(rows))}, "structures": per}


def run_inner(sweep, out, inputs_hash):
    """Tune A, B, D, summarise E on 2023-24, write the atomic immutable receipts bound to ``inputs_hash``; returns
    {"tuned": ..., "results": {arm_or_production: {"2023H2": evaluate, "2024": evaluate}}}."""
    tuned = {arm: tune(sweep, arm) for arm in ("A", "B", "D")}
    tuned["E"] = e_summary(sweep)
    for arm, body in tuned.items():
        receipts.write_receipt(Path(out) / "receipts" / f"tuning_{arm}.json",
                               {"experiment": "EXP-189", "arm": arm, "inputs_hash": inputs_hash, "window": "2023H2-2024", "body": body})
    labels = {"production": {n: 1.0 for n in NAMES}, **{arm: factors(arm, tuned[arm]) for arm in ("A", "B", "D", "E")}}
    results = {label: {w: evaluate(sweep, by, WINDOWS[w]) for w in ("2023H2", "2024")} for label, by in labels.items()}
    return {"tuned": tuned, "results": results}


def admit_draws(n):
    """The draw count ``n`` when it is the protocol N (FAKE_DRAWS, read at call time) as a plain int; anything else is refused."""
    if type(n) is int and n == FAKE_DRAWS:
        return n
    raise Refusal(_CONFLICT, "tuning C draw count is not the protocol N", {"reason": "fake_draws_not_protocol", "n": n})


def _finite_real(x):
    """True for a finite int or float that is not a bool (a bool is not a number here)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    try:
        return math.isfinite(float(x))
    except OverflowError:
        return False


def _fake_model_ok(model):
    """True when ``model`` is a dict whose L, b, s and mean_log_y are finite real numbers, s >= 0, and n an int >= MIN_EVENTS."""
    if not isinstance(model, dict):
        return False
    if not all(_finite_real(model.get(key)) for key in ("L", "b", "s", "mean_log_y")):
        return False
    return model["s"] >= 0 and type(model.get("n")) is int and model["n"] >= MIN_EVENTS


def _valid(arm, receipt):
    """True when ``receipt`` is EXP-189 arm ``arm`` on the inner window with a dict body; A, B, C and D bodies hold exactly the units
    of ``arm``, each with its members and a grid k (a bool is not a k). An arm C receipt also needs a calibrated fake with the
    protocol N, the seed tag, and a valid fake model."""
    body = receipt.get("body")
    if not (receipt.get("experiment") == "EXP-189" and receipt.get("arm") == arm and receipt.get("window") == "2023H2-2024"
            and isinstance(body, dict)):
        return False
    if arm not in ("A", "B", "C", "D"):
        return True
    if arm == "C":
        fake = receipt.get("fake")
        cal = fake.get("calibration") if isinstance(fake, dict) else None
        if not (isinstance(cal, dict) and cal.get("ok") is True):
            return False
        try:
            admit_draws(fake.get("N"))
        except Refusal:
            return False
        if fake.get("seed_tag") != FAKE_SEED_TAG or not _fake_model_ok(fake.get("model")):
            return False
    try:
        return set(body) == set(units(arm)) and all(
            isinstance(body[u], dict) and body[u].get("members") == list(m) and admit_k(body[u].get("k")) is not None
            for u, m in units(arm).items())
    except ValueError:
        return False


def frozen(out, arm, inputs_hash):
    """The factors of ``arm`` from its receipt, validated before they are read. A missing receipt refuses tuning_receipt_missing
    (load_receipt); a failed identity or shape check, or other inputs, refuses tuning_receipt_changed."""
    receipt = receipts.load_receipt(Path(out) / "receipts" / f"tuning_{arm}.json")
    if not _valid(arm, receipt):
        raise Refusal(_CONFLICT, f"tuning receipt {arm} does not match EXP-189 arm {arm} on the inner window",
                      {"reason": "tuning_receipt_changed"})
    if receipt["inputs_hash"] != inputs_hash:
        raise Refusal(_CONFLICT, f"tuning receipt {arm} is bound to other inputs", {"reason": "tuning_receipt_changed"})
    return factors(arm, receipt["body"])


def run_outer(sweep, out, arms, inputs_hash, window=WINDOWS["2025"]):
    """2025 (2025 NOT blind for widths): every arm's receipt is loaded first (a missing one refuses before any account is walked), then production
    (k = 1) and each arm are evaluated on its frozen factors. Returns {label: evaluate}."""
    by = {arm: frozen(out, arm, inputs_hash) for arm in arms}
    by = {"production": {n: 1.0 for n in NAMES}, **by}
    return {label: evaluate(sweep, f, window) for label, f in by.items()}
