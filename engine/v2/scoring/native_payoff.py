"""Native, answer-free reproduction of legacy's payoff-calibration/model layer.

Legacy computes ``exp_pnl_model``/``win_model`` in two pieces
(``engine/score.py::Scorer._score_model``, :2067-2218):

1. A payoff line -- ``exit_value / spot ~= intercept + slope * driver`` --
   fitted causally (only trades closed before the decision's evidence
   cutoff) on real, already-closed trades. Pure math in
   ``engine/payoff.py::fit_payoff`` (:296-363): OLS via ``np.polyfit``, plus
   the fitted line's own residuals, capped to ``MAX_RESIDUALS`` via a fixed
   subsample seed (:289-293, :343-350).
2. The champion driver model's OWN held-out residual population, optionally
   bucketed by the decile the point prediction falls in
   (``engine/models/registry.py::bucket_residuals``, :242-278, and
   ``ModelArtifact.residual_pool``/``residual_draws``, :197-231).
3. Both draws feed one Monte Carlo return distribution
   (``engine/payoff.py::simulate_returns``, :445-466), drawn from a SHARED
   rng in a fixed order: the driver's own residual pool first, then the
   payoff line's residuals (engine/score.py:2164-2208) -- order matters
   because ``rng.choice`` consumes the generator's state sequentially.

STR-RUNUP (R4-17) is the two-driver exception (EXP-149): its exit value is
not a line through one driver but a surface through the predicted T-1
implied move AND exit moneyness, fitted by
``engine/payoff.py::fit_runup_payoff``/``RunupPayoffSurface`` and simulated
by ``engine/score.py::Scorer._score_runup_model`` (:2270-2440) +
``engine/payoff.py::simulate_runup_returns`` (:469-491). The draw order there
is fixed too, from one shared rng: the implied-move model's own residual
pool FIRST, the runup-move model's own residual pool (at its native D14
scale) SECOND, the +/-1 sign draw THIRD, the payoff surface's own residuals
FOURTH (engine/score.py:2385-2421).

``engine/v2`` has no dependency on legacy ``engine/`` code (see
``stages.py``'s ``ATM_TOLERANCE_PCT`` precedent), so this module reproduces
the pure math as literal, cited functions rather than importing
``engine.payoff``/``engine.models.registry``. Tests MAY import the legacy
modules to pin parity; this module must not.
"""
from __future__ import annotations

from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "MIN_TRADES", "MAX_RESIDUALS", "RESIDUAL_SEED", "DECILES", "MIN_POOL",
    "MODEL_DRAWS", "RUNUP_TERMS", "RUNUP_BASE_DAYS",
    "fit_payoff_line", "cap_residuals", "bucket_residual_pool",
    "residual_pool_for", "driver_residual_pool", "payoff_exit_value",
    "simulate_model_returns",
    "scale_runup_move", "runup_payoff_design", "fit_runup_payoff_surface",
    "runup_exit_value_per_spot", "simulate_runup_model_returns",
]

#: engine/payoff.py:285
MIN_TRADES = 200
#: engine/payoff.py:289
MAX_RESIDUALS = 5000
#: engine/payoff.py:293
RESIDUAL_SEED = 20260829
#: engine/models/registry.py:243 (``bucket_residuals`` default)
DECILES = 10
#: engine/models/registry.py:243 (``bucket_residuals`` default ``min_pool``)
MIN_POOL = 250
#: engine/score.py:207
MODEL_DRAWS = 4000


def _parse_day(value: Any) -> np.datetime64 | None:
    if value is None:
        return None
    try:
        day = np.datetime64(str(value)[:10])
    except ValueError:
        return None
    return None if np.isnat(day) else day


def fit_payoff_line(
    rows: Sequence[Mapping[str, Any]] | None,
    *,
    before: Any = None,
    min_trades: int = MIN_TRADES,
    max_residuals: int = MAX_RESIDUALS,
    residual_seed: int = RESIDUAL_SEED,
) -> dict[str, Any] | None:
    """engine/payoff.py:296-341 (``fit_payoff``), from raw source rows.

    Each row is one PRIOR, already-closed trade: ``driver`` (the driver
    value realized at entry), ``spot_entry``, ``exit_value`` and
    ``exit_date``. ``before`` restricts to trades closed strictly before it
    (engine/payoff.py:317-319, ``exit_date < before``) -- the same causal
    rule legacy applies, so a row dated on or after the cutoff never reaches
    the fit. Returns ``None`` (mirroring ``PayoffError``) when fewer than
    ``min_trades`` rows survive filtering.
    """
    cutoff = _parse_day(before) if before is not None else None
    driver: list[float] = []
    spot: list[float] = []
    exit_value: list[float] = []
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        if cutoff is not None:
            exit_day = _parse_day(row.get("exit_date"))
            if exit_day is None or not (exit_day < cutoff):
                continue
        try:
            d = float(row["driver"])
            s = float(row["spot_entry"])
            v = float(row["exit_value"])
        except (KeyError, TypeError, ValueError):
            continue
        driver.append(d)
        spot.append(s)
        exit_value.append(v)
    driver_arr = np.asarray(driver, dtype=float)
    spot_arr = np.asarray(spot, dtype=float)
    exit_arr = np.asarray(exit_value, dtype=float)
    ok = (
        np.isfinite(driver_arr) & np.isfinite(spot_arr) & np.isfinite(exit_arr)
        & (spot_arr > 0)
    )
    x = driver_arr[ok]
    y = exit_arr[ok] / spot_arr[ok]
    n = int(x.size)
    if n < min_trades:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    r = (
        float(np.corrcoef(x, y)[0, 1])
        if x.std() > 0 and y.std() > 0
        else None
    )
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "resid_sd": float(resid.std(ddof=2)) if resid.size > 2 else float("nan"),
        "n": n,
        "r": r,
        "residuals": cap_residuals(resid, max_residuals=max_residuals, seed=residual_seed),
    }


def cap_residuals(
    residuals: np.ndarray,
    *,
    max_residuals: int = MAX_RESIDUALS,
    seed: int = RESIDUAL_SEED,
) -> np.ndarray:
    """engine/payoff.py:343-350 -- cap and sort the fitted line's residuals.

    Order matters: the capped array is later sampled by ``rng.choice`` on
    matching indices, so the native and legacy arrays must be sorted
    identically to draw the same values from the same seed.
    """
    kept = np.asarray(residuals, dtype=float)
    if kept.size > max_residuals:
        kept = np.random.default_rng(seed).choice(
            kept, size=max_residuals, replace=False,
        )
    return np.sort(kept)


def bucket_residual_pool(
    predictions: Sequence[float],
    residuals: Sequence[float],
    *,
    deciles: int = DECILES,
    min_pool: int = MIN_POOL,
) -> dict[str, Any] | None:
    """engine/models/registry.py:242-278 (``bucket_residuals``), verbatim math.

    Returns ``None`` when the sample cannot support the split -- the caller
    then falls back to the flat pool, exactly as legacy's
    ``ModelArtifact.residual_pool`` does when ``residual_buckets`` is absent.
    """
    pred = np.asarray(predictions, dtype=float)
    res = np.asarray(residuals, dtype=float)
    if pred.shape != res.shape:
        raise ValueError(
            f"predictions {pred.shape} and residuals {res.shape} differ",
        )
    ok = np.isfinite(pred) & np.isfinite(res)
    pred, res = pred[ok], res[ok]
    if pred.size < deciles * min_pool:
        return None
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, deciles + 1)))
    if edges.size < 3:
        return None
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    index = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, edges.size - 2)
    pools = [res[index == i] for i in range(edges.size - 1)]
    return {"edges": edges, "pools": pools, "min_pool": int(min_pool)}


def residual_pool_for(
    buckets: Mapping[str, Any] | None,
    prediction: float | None,
    flat_residuals: np.ndarray,
) -> tuple[np.ndarray, str]:
    """engine/models/registry.py:197-221 (``ModelArtifact.residual_pool``)."""
    if not buckets or prediction is None or not isfinite(prediction):
        return flat_residuals, "flat"
    edges = buckets["edges"]
    pools = buckets["pools"]
    index = int(np.clip(np.searchsorted(edges, prediction, side="right") - 1, 0, len(pools) - 1))
    pool = pools[index]
    if pool.size < int(buckets.get("min_pool", 0)):
        return flat_residuals, f"flat (bucket {index} thin: {pool.size})"
    return pool, f"bucket {index}"


def driver_residual_pool(
    rows: Sequence[Mapping[str, Any]] | None,
    prediction: float | None,
    *,
    deciles: int = DECILES,
    min_pool: int = MIN_POOL,
) -> np.ndarray | None:
    """The champion driver model's own held-out residual pool for one point.

    ``rows`` are the model's FULL held-out (prediction, residual) population
    -- a fixed, artifact-owned asset (engine/models/registry.py's
    ``ModelArtifact.residuals``/``residual_buckets``), not filtered per
    request the way the payoff rows are: legacy never re-filters a trained
    artifact's own residual store at scoring time. Returns ``None`` when no
    valid rows are supplied at all (the caller's cue that the model's
    calibration state is missing, not merely thin).
    """
    predictions: list[float] = []
    residuals: list[float] = []
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        try:
            p = float(row["prediction"])
            r = float(row["residual"])
        except (KeyError, TypeError, ValueError):
            continue
        if isfinite(p) and isfinite(r):
            predictions.append(p)
            residuals.append(r)
    if not residuals:
        return None
    pred_arr = np.asarray(predictions, dtype=float)
    res_arr = np.asarray(residuals, dtype=float)
    buckets = bucket_residual_pool(pred_arr, res_arr, deciles=deciles, min_pool=min_pool)
    pool, _ = residual_pool_for(buckets, prediction, res_arr)
    return pool


def payoff_exit_value(
    driver_values: Sequence[float], spot: float, intercept: float, slope: float,
) -> np.ndarray:
    """engine/payoff.py:138-145 (``PayoffMap.exit_value``)."""
    values = np.asarray(driver_values, dtype=float)
    return np.maximum(0.0, (intercept + slope * values) * float(spot))


def simulate_model_returns(
    driver: float,
    driver_pool: Sequence[float],
    payoff_residuals: Sequence[float],
    intercept: float,
    slope: float,
    spot: float,
    cost: float,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """engine/score.py:2164-2208 + engine/payoff.py:445-466, in one call.

    Draw order is fixed and matches legacy exactly: the driver's own
    residual pool is drawn from ``rng`` FIRST, then the payoff line's
    residuals SECOND -- both from the same generator, so reordering would
    silently change every draw downstream.
    """
    model_draws = float(driver) + rng.choice(
        np.asarray(driver_pool, dtype=float), size=int(draws), replace=True,
    )
    noise = rng.choice(
        np.asarray(payoff_residuals, dtype=float), size=int(draws), replace=True,
    )
    pnl = payoff_exit_value(model_draws, spot, intercept, slope) - float(cost)
    pnl = pnl + noise * float(spot)
    if cost <= 0:
        return np.full(np.shape(pnl), np.nan)
    return pnl / float(cost)


# ---------------------------------------------------------------------------
# STR-RUNUP -- the two-driver payoff surface (R4-17)
# ---------------------------------------------------------------------------

#: engine/payoff.py:172-180 RUNUP_TERMS / RUNUP_BASE_DAYS
RUNUP_TERMS = (
    "intercept",
    "implied_move",
    "abs_moneyness",
    "moneyness_sq_div10",
    "signed_moneyness",
    "implied_x_abs_moneyness_div10",
)
RUNUP_BASE_DAYS = 14.0


def scale_runup_move(values: Sequence[float] | float,
                     days_before_print: float) -> np.ndarray:
    """engine/payoff.py:183-186 (``scale_runup_move``), verbatim."""
    scale = float(days_before_print) / RUNUP_BASE_DAYS
    return np.asarray(values, dtype=float) * scale


def runup_payoff_design(implied_move: Sequence[float],
                        moneyness: Sequence[float]) -> np.ndarray:
    """engine/payoff.py:189-203 (``runup_payoff_design``), verbatim."""
    implied = np.asarray(implied_move, dtype=float)
    money = np.asarray(moneyness, dtype=float)
    absolute = np.abs(money)
    return np.column_stack([
        np.ones(len(implied)),
        implied,
        absolute,
        np.square(money) / 10.0,
        money,
        implied * absolute / 10.0,
    ])


def fit_runup_payoff_surface(
    rows: Sequence[Mapping[str, Any]] | None,
    *,
    before: Any = None,
    min_trades: int = MIN_TRADES,
    max_residuals: int = MAX_RESIDUALS,
    residual_seed: int = RESIDUAL_SEED,
) -> dict[str, Any] | None:
    """engine/payoff.py:366-442 (``fit_runup_payoff``), from raw source rows.

    Each row is one PRIOR, already-closed STR-RUNUP trade: ``driver`` (the
    implied T-1 move realized at entry, legacy's ``im_t1``), ``spot_entry``,
    ``spot_exit``, ``strike`` and ``exit_value``. ``before`` restricts to
    trades closed strictly before it, the same causal rule
    :func:`fit_payoff_line` applies. Returns ``None`` (mirroring
    ``PayoffError``) when fewer than ``min_trades`` rows survive filtering.
    """
    cutoff = _parse_day(before) if before is not None else None
    implied: list[float] = []
    spot_entry: list[float] = []
    spot_exit: list[float] = []
    strike: list[float] = []
    exit_value: list[float] = []
    for row in rows or ():
        if not isinstance(row, Mapping):
            continue
        if cutoff is not None:
            exit_day = _parse_day(row.get("exit_date"))
            if exit_day is None or not (exit_day < cutoff):
                continue
        try:
            im = float(row["driver"])
            se = float(row["spot_entry"])
            sx = float(row["spot_exit"])
            k = float(row["strike"])
            ev = float(row["exit_value"])
        except (KeyError, TypeError, ValueError):
            continue
        implied.append(im)
        spot_entry.append(se)
        spot_exit.append(sx)
        strike.append(k)
        exit_value.append(ev)
    implied_arr = np.asarray(implied, dtype=float)
    se_arr = np.asarray(spot_entry, dtype=float)
    sx_arr = np.asarray(spot_exit, dtype=float)
    k_arr = np.asarray(strike, dtype=float)
    ev_arr = np.asarray(exit_value, dtype=float)
    ok = (
        np.isfinite(implied_arr) & np.isfinite(se_arr) & np.isfinite(sx_arr)
        & np.isfinite(k_arr) & np.isfinite(ev_arr)
        & (se_arr > 0) & (sx_arr > 0) & (k_arr > 0)
    )
    implied_arr, se_arr, sx_arr, k_arr, ev_arr = (
        implied_arr[ok], se_arr[ok], sx_arr[ok], k_arr[ok], ev_arr[ok],
    )
    n = int(implied_arr.size)
    if n < min_trades:
        return None
    moneyness = 100.0 * np.log(sx_arr / k_arr)
    design = runup_payoff_design(implied_arr, moneyness)
    target = ev_arr / se_arr
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted = design @ coefficients
    residuals = target - fitted
    r = (
        float(np.corrcoef(fitted, target)[0, 1])
        if fitted.std() > 0 and target.std() > 0
        else None
    )
    return {
        "coefficients": tuple(float(value) for value in coefficients),
        "resid_sd": float(residuals.std(ddof=2)) if residuals.size > 2 else float("nan"),
        "n": n,
        "r": r,
        "residuals": cap_residuals(residuals, max_residuals=max_residuals, seed=residual_seed),
    }


def runup_exit_value_per_spot(
    implied_move: Sequence[float],
    signed_move: Sequence[float],
    coefficients: Sequence[float],
    *,
    spot: float,
    strike: float,
) -> np.ndarray:
    """engine/payoff.py:225-256 (``RunupPayoffSurface.exit_value``/``value_per_spot``).

    Unfloored -- the caller applies the zero floor after adding the surface's
    own residual noise, exactly as ``simulate_runup_returns`` does
    (engine/payoff.py:480-488).
    """
    implied = np.asarray(implied_move, dtype=float)
    move = np.asarray(signed_move, dtype=float)
    implied, move = np.broadcast_arrays(implied, move)
    exit_spot = float(spot) * np.exp(move / 100.0)
    moneyness = 100.0 * np.log(exit_spot / float(strike))
    design = runup_payoff_design(implied.ravel(), moneyness.ravel())
    value_per_spot = design @ np.asarray(coefficients, dtype=float)
    return value_per_spot.reshape(implied.shape)


def simulate_runup_model_returns(
    point_implied: float,
    point_move_d14: float,
    implied_pool: Sequence[float],
    move_pool: Sequence[float],
    coefficients: Sequence[float],
    payoff_residuals: Sequence[float],
    spot: float,
    strike: float,
    cost: float,
    days_before_print: float,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """engine/score.py:2377-2430 + engine/payoff.py:469-491, in one call.

    Draw order is fixed and matches legacy exactly, all from the SAME rng:
    the implied-move model's own residual pool FIRST, the runup-move
    model's own residual pool (at its native D14 scale) SECOND, the +/-1
    sign draw THIRD, the payoff surface's own residuals FOURTH. Reordering
    any of the four would silently change every draw downstream.
    """
    implied_draws = float(point_implied) + rng.choice(
        np.asarray(implied_pool, dtype=float), size=int(draws), replace=True,
    )
    implied_draws = np.maximum(implied_draws, 0.0)
    move_draws_d14 = float(point_move_d14) + rng.choice(
        np.asarray(move_pool, dtype=float), size=int(draws), replace=True,
    )
    move_draws = scale_runup_move(
        np.maximum(move_draws_d14, 0.0), days_before_print,
    )
    signed_moves = rng.choice((-1.0, 1.0), size=int(draws)) * move_draws
    noise = rng.choice(
        np.asarray(payoff_residuals, dtype=float), size=int(draws), replace=True,
    )
    value_per_spot = runup_exit_value_per_spot(
        implied_draws, signed_moves, coefficients, spot=spot, strike=strike,
    )
    value_per_spot = value_per_spot + noise
    value = np.maximum(value_per_spot, 0.0) * float(spot)
    if cost <= 0:
        return np.full(np.shape(value), np.nan)
    return (value - float(cost)) / float(cost)
