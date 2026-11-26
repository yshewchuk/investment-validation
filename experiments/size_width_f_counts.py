"""EXP-189 Amendment 1 (A1.4, A1.5): F views and counted non-trades, added in place to an F sweep's evaluate() result.
Keys: ``views``, ``nontrades`` (any_k_f_skipped_share on E), ``nonpositive_debit`` (ladder sweeps: NaN-return trades per alpha).
"""
import pandas as pd

from experiments import size_width_account as account
from experiments import size_width_f_fills as ff
from experiments.fair_value_fills import VARIANTS
from experiments.size_width_sweep import E_ARM, RET_COLS, in_window
from experiments.size_width_success import LADDER


def _in(frames, window):
    return pd.concat([in_window(f, window) for f in frames] or [pd.DataFrame()], ignore_index=True)


def _skipped(cands):
    return set() if cands.empty else set(cands.loc[cands["reason"].isin(ff.ENTRY_REASONS), "event_id"])


def views(rows, names):
    """{view: simulate_account metrics + mean_ret on the view's own columns (None when empty)}; F is the primary pair."""
    pair = {v: ("entry_cost", "exit_value") if v == "F" else (f"entry_cost_{v}", f"exit_value_{v}") for v in names}
    return {v: {**account.simulate_account(rows, "primary" if v == "F" else v),
                "mean_ret": None if rows.empty else float((rows[pair[v][1]] / rows[pair[v][0]] - 1.0).mean())} for v in names}


def nontrades(sweep, name, arm, window):
    """Counts of the candidates of (name, arm) by event date in ``window``, against production (name, 1.0)."""
    key, prod = (name, arm), (name, 1.0)
    cands = _in(sweep.scored.get(key, []), window)
    skip, has = _skipped(cands), prod in sweep.scored
    prod_skip = _skipped(_in(sweep.scored[prod], window)) if has else set()
    out = {"candidates": len(cands), "f_skipped": len(skip),
           "reasons": {str(r): int(c) for r, c in _in(sweep.declined_frames.get(key, []), window).get("reason", pd.Series(dtype=object)).fillna("none").value_counts().items()},
           "only_arm_f_skipped": len(skip - prod_skip) if has else None,
           "only_production_f_skipped": len(prod_skip - skip) if has else None}
    if arm == E_ARM:  # the share of E's candidate events with at least one grid k F-skipped
        full = len(cands) > 0 and all((name, k) in sweep.scored for k in sweep.ks)
        hit = set().union(*(_skipped(_in(sweep.scored[(name, k)], window)) for k in sweep.ks)) if full else set()
        out["any_k_f_skipped_share"] = len(hit & set(cands["event_id"])) / len(cands) if full else None
    return out


def add(result, sweep, pairs, window):
    """Adds the views and counts of ``pairs`` over ``window`` to ``result`` (evaluate's dict), in place."""
    for name, arm in pairs:
        result["structures"][name].update(views=views(sweep.rows([(name, arm)], window), ("F", "mid", "harsh")),
                                          nontrades=nontrades(sweep, name, arm, window))
    rows = sweep.rows(pairs, window)
    result["account"]["views"] = views(rows, ("F", "mid", "harsh", *VARIANTS))
    if getattr(sweep, "ladder", False):  # a ladder return is NaN exactly when the repriced debit is not positive
        result["account"]["nonpositive_debit"] = {f"{a:.2f}": 0 if rows.empty else int(rows[c].isna().sum()) for a, c in zip(LADDER, RET_COLS)}
