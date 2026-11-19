"""EXP-188 S4: per-model residual pools, the linear fast path, and its equivalence check.

Each model gets its own ``ResidualPool`` built from its prediction table and the shared labels;
nothing here names a model kind, so a synthetic prediction source can pass through the same call.
``put_means`` simulates the exits once and prices a single put at every strike of a ladder, so any
structure on that ladder is a weighted sum of the same means. ``check_equivalence`` holds that path
to ``expected_pnl`` on a sample of events, and a mismatch refuses the run.
"""
import hashlib
import sys

import numpy as np
import pandas as pd

from engine.pnl_sim import (DRAWS, MIN_POOL, MIN_SPOT_FRACTION, ResidualPool, black_scholes_put,
                            expected_pnl, trailing_cutoff)
from engine.v2.ops.errors import fail

COMMON_START = pd.Timestamp("2023-01-01")
MIN_WINDOW = 100
TOLERANCE = 1e-9
_CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
_TABLE_COLS = ("event_date", "pred_move", "pred_crush")
_LABEL_COLS = ("abs_move", "crush_pct_iv30")


def build_pool(table, labels, *, floor=COMMON_START):
    """Residual pool for one model: paired move and crush errors, from ``floor`` on if given."""
    missing = [c for c in _TABLE_COLS if c not in table.columns]
    missing += [c for c in _LABEL_COLS if c not in labels.columns]
    if missing:
        raise fail(_CONFLICT, "the pool inputs lack required columns",
                   details={"reason": "pool_columns_missing", "missing": missing})
    frame = table[list(_TABLE_COLS)].join(labels[list(_LABEL_COLS)], how="inner")
    dates = pd.to_datetime(frame["event_date"])
    if floor is not None:
        keep = dates >= pd.Timestamp(floor)
        frame, dates = frame[keep], dates[keep]
    history = pd.DataFrame({
        "event_date": dates.to_numpy(),
        "pred_abs_move": frame["pred_move"].to_numpy(dtype=float),
        "err_move": (frame["abs_move"] - frame["pred_move"]).to_numpy(dtype=float),
        "err_crush": (frame["crush_pct_iv30"] - frame["pred_crush"]).to_numpy(dtype=float),
    })
    pool = ResidualPool(history)
    floor_text = "none" if floor is None else pd.Timestamp(floor).date()
    print(f"[pnl_pool] pool rows={len(pool)} floor={floor_text}", file=sys.stderr)
    return pool


def event_seed(key, event_date):
    """Seed for one event's draws: the hash ``expected_pnl`` uses, so both paths agree."""
    return int.from_bytes(hashlib.sha256(f"{key}|{event_date}".encode()).digest()[:8], "big")


def put_means(pool, *, key, event_date, spot, pre_iv30, pred_move, pred_crush, dte_exit,
              strikes, draws=DRAWS):
    """Mean exit value of a single put at each strike, or ``None`` if unsimulable.

    Validation, seed and draw order match ``expected_pnl``, so the same simulated exits price every
    strike. A non-finite spot, vol or forecast returns None, as in expected_pnl. Allocates the draws
    vectors and one (draws, strikes) grid transiently; only the (strikes,) means are returned.
    """
    strikes = np.asarray(strikes, dtype=float)
    if strikes.ndim != 1 or not np.all(np.isfinite(strikes)) or np.any(np.diff(strikes) <= 0):
        raise ValueError("strikes must be a finite, strictly increasing 1-D array")
    values = (spot, pre_iv30, pred_move, pred_crush, dte_exit)
    if not all(v is not None and np.isfinite(v) for v in values):
        return None
    if pre_iv30 <= 0 or dte_exit < 0:
        return None
    rng = np.random.default_rng(event_seed(key, event_date))
    err_move, err_crush = pool.draw(event_date, pred_move, draws, rng)
    if err_move.size == 0:
        return None
    move = np.maximum(pred_move + err_move, 0.0)
    crush = pred_crush + err_crush
    sign = rng.choice((-1.0, 1.0), size=move.size)
    spot_exit = spot * np.maximum(1.0 + sign * move / 100.0, MIN_SPOT_FRACTION)
    vol_exit = (pre_iv30 / 100.0) * (1.0 + crush / 100.0)
    puts = black_scholes_put(spot_exit[:, None], strikes[None, :], dte_exit / 365.0,
                             vol_exit[:, None])
    return puts.mean(axis=0)


def structure_value(means, strikes, legs):
    """Value per unit of a structure: sum of side * qty * mean put at each leg's strike.

    Legs are skipped as ``expected_pnl`` skips them: no strike, a non-finite strike, or zero qty.
    """
    grid = np.asarray(strikes, dtype=float)
    total = 0.0
    for leg in legs:
        strike = leg.get("strike")
        if strike is None or not np.isfinite(float(strike)):
            continue
        qty = float(leg.get("qty", 0.0))
        if not np.isfinite(qty):
            raise fail(_CONFLICT, "a leg quantity is not finite",
                       details={"reason": "leg_invalid"})
        if qty == 0:
            continue
        hit = np.flatnonzero(grid == float(strike))
        if hit.size != 1:
            raise fail(_CONFLICT, "a leg strike is off the ladder",
                       details={"reason": "leg_off_ladder"})
        # `sell` means the position is LONG that leg and receives its value, as in expected_pnl.
        side = 1.0 if str(leg.get("side", "")).lower() == "sell" else -1.0
        total += side * qty * float(means[hit[0]])
    return total


def structure_pnl(means, strikes, legs, entry_cost):
    """Simulated return on entry cost, or ``None`` when entry cost is not a positive finite number."""
    if entry_cost is None or not np.isfinite(entry_cost) or entry_cost <= 0:
        return None
    return (structure_value(means, strikes, legs) - entry_cost) / entry_cost


def undetermined_reason(pool, history, event_date):
    """Why an event cannot be scored under a model: ``thin_pool``, ``thin_window``, or ``None``."""
    if pool.before(event_date) < MIN_POOL:
        return "thin_pool"
    if trailing_cutoff(history, event_date, min_window=MIN_WINDOW) is None:
        return "thin_window"
    return None


def paired_events(undetermined, event_dates):
    """Kept event ids and dropped counts by quarter.

    An event undetermined under ANY model leaves for ALL, so every model is scored on one population.
    """
    dropped = set()
    for ids in undetermined.values():
        if not set(ids) <= set(event_dates.index):
            raise ValueError("undetermined ids are not in the candidate population")
        dropped |= set(ids)
    dates = pd.to_datetime(event_dates)
    kept = sorted(eid for eid in event_dates.index if eid not in dropped)
    quarters = {}
    for eid in dropped:
        stamp = pd.Timestamp(dates[eid])
        label = f"{stamp.year}Q{stamp.quarter}"
        quarters[label] = quarters.get(label, 0) + 1
    return kept, dict(sorted(quarters.items()))


def check_equivalence(pool, sample, *, tolerance=TOLERANCE, draws=DRAWS):
    """Hold the fast path to ``expected_pnl`` on sampled structures; a mismatch refuses the run."""
    if not (np.isfinite(tolerance) and tolerance > 0):
        raise ValueError("tolerance must be a positive finite number")
    if not sample:
        raise ValueError("equivalence needs a non-empty sample")
    worst = 0.0
    compared = 0
    for item in sample:
        means = put_means(pool, key=item["key"], event_date=item["event_date"], spot=item["spot"],
                          pre_iv30=item["pre_iv30"], pred_move=item["pred_move"],
                          pred_crush=item["pred_crush"], dte_exit=item["dte_exit"],
                          strikes=item["strikes"], draws=draws)
        legs = item["legs"]
        fast = None if means is None or not legs else structure_pnl(
            means, item["strikes"], legs, item["entry_cost"])
        direct = expected_pnl(exit_legs=legs, spot=item["spot"], entry_cost=item["entry_cost"],
                              pre_iv30=item["pre_iv30"], pred_abs_move=item["pred_move"],
                              pred_iv_crush=item["pred_crush"], dte_exit=item["dte_exit"],
                              event_date=item["event_date"], pool=pool, key=item["key"],
                              draws=draws)
        ref = None if direct is None else direct["exp_pnl_sim"]
        if fast is None and ref is None:
            continue
        if fast is None or ref is None or not abs(fast - ref) <= tolerance:
            raise fail(_CONFLICT, "the fast path disagrees with expected_pnl",
                       details={"reason": "fast_path_mismatch"})
        compared += 1
        worst = max(worst, abs(fast - ref))
    if compared == 0:
        raise fail(_CONFLICT, "no sample item was simulable, so nothing was compared",
                   details={"reason": "equivalence_empty"})
    skipped = len(sample) - compared
    print(f"[pnl_pool] equivalence checked={compared} skipped={skipped} "
          f"max_abs_diff={worst:.3e}", file=sys.stderr)
    return {"checked": compared, "skipped": skipped, "max_abs_diff": worst}
