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
    "MODEL_DRAWS",
    "fit_payoff_line", "cap_residuals", "bucket_residual_pool",
    "residual_pool_for", "driver_residual_pool", "payoff_exit_value",
    "simulate_model_returns",
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
