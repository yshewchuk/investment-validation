"""Derived gate and chooser feature columns, natively (R4-20 gaps 3 and 5).

A frozen gate may name columns that are not in the base feature frame. The
forecast-analog STR-THRU champion (``gate_midfill_str_thru_forecast_analog``)
names eight of them, which legacy computes on demand in
``engine/score.py`` ``Scorer._gate_feature_frame``:

* ``pred_abs_move`` and its band ``pred_abs_move_p10``/``_p90``/``_sd``: the
  Tier-4 SIZE fold served at ``tier4.serving_fold(event_date, as_of)``
  (``Scorer._forecast_for_gate``), its band from ``ServingModel.interval``
  (``tier4.interval_for`` over the fold's own held-out ``pool_pred``/
  ``pool_res``);
* ``forecast_edge``: that forecast minus the row's ``im`` feature;
* ``analog_mean``/``analog_win_rate``/``analog_n``: the bucket-analog
  layer's ``exp_pnl_analog``/``win_analog``/``n_analogs`` for this row.

Here each is derived from declared inputs only: the forecast from a frozen
size-fold binding the gate recipe names, the band from that fold's pool as
declared source rows, the analog columns from the native analog stage's own
outputs. The arithmetic is a line-for-line port (``interval_for``,
``_pool_stats``, ``_floored``; ``registry.bucket_residuals`` is
``native_payoff.bucket_residual_pool``), so the gate row equals legacy's bit
for bit, NaN where legacy has NaN -- and the frozen stage executor then
declines a non-finite column MISSING_FEATURES, as ``_score_gate`` does.

The DYN-SV chooser's own ``analog_*`` columns are a DIFFERENT construction
(the k-NN block, ``Scorer._chooser_analogs``) under the same names, so these
gate columns are applied to the gate's inputs only, never published.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from engine.v2.scoring.native_payoff import DECILES, MIN_POOL, bucket_residual_pool

__all__ = [
    "CHOOSER_MENU",
    "GATE_ANALOG_COLUMNS",
    "GATE_FORECAST_COLUMNS",
    "MIN_RESIDUALS",
    "chooser_direct_columns",
    "forecast_interval",
    "gate_analog_columns",
    "gate_forecast_columns",
]

#: engine/score.py ``Scorer._GATE_FORECAST_COLUMNS``/``_GATE_ANALOG_COLUMNS``.
GATE_FORECAST_COLUMNS = (
    "pred_abs_move", "pred_abs_move_p10", "pred_abs_move_p90",
    "pred_abs_move_sd", "forecast_edge",
)
GATE_ANALOG_COLUMNS = ("analog_mean", "analog_win_rate", "analog_n")
#: engine/data/features/tier4.py ``MIN_RESIDUALS``.
MIN_RESIDUALS = 250
#: engine/score.py ``DYNAMIC_MENU`` -- the families the chooser ranks.
CHOOSER_MENU = ("TWIN-P", "TWIN-P5", "CND-PS", "BFLY-P", "BFLY-P5", "RAMP7", "CTR5")

_NAN = float("nan")


def _pool_stats(residuals: np.ndarray) -> tuple[float, float, float, int]:
    """engine/data/features/tier4.py ``_pool_stats``."""
    clean = residuals[np.isfinite(residuals)]
    if clean.size < MIN_RESIDUALS:
        return _NAN, _NAN, _NAN, int(clean.size)
    return (
        float(np.quantile(clean, 0.10)),
        float(np.quantile(clean, 0.90)),
        float(clean.std(ddof=1)),
        int(clean.size),
    )


def _floored(values: np.ndarray, floor: float | None) -> np.ndarray:
    return values if floor is None else np.maximum(values, floor)


def forecast_interval(
    predictions: Sequence[float],
    pool_pred: Sequence[float],
    pool_res: Sequence[float],
    *,
    floor: float | None = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """engine/data/features/tier4.py ``interval_for``, verbatim math.

    ``(p10, p90, sd, n)`` for ``predictions`` from a fold's held-out pool:
    the flat pool's stats unless the prediction-decile buckets
    (``registry.bucket_residuals`` = ``native_payoff.bucket_residual_pool``)
    exist, then the bucket's stats (flat when the bucket is thin).
    """
    predictions = np.asarray(predictions, dtype=float)
    p10 = np.full(predictions.shape, np.nan)
    p90 = np.full(predictions.shape, np.nan)
    sd = np.full(predictions.shape, np.nan)
    n = np.full(predictions.shape, np.nan)

    flat = np.asarray(pool_res, dtype=float)
    flat_stats = _pool_stats(flat)
    if flat_stats[3] < MIN_RESIDUALS:
        return p10, p90, sd, n

    buckets = bucket_residual_pool(pool_pred, pool_res, deciles=DECILES, min_pool=MIN_POOL)
    if buckets is None:
        q10, q90, spread, count = flat_stats
        known = np.isfinite(predictions)
        p10[known] = _floored(predictions[known] + q10, floor)
        p90[known] = _floored(predictions[known] + q90, floor)
        sd[known] = spread
        n[known] = count
        return p10, p90, sd, n

    edges, pools = buckets["edges"], buckets["pools"]
    min_pool = int(buckets.get("min_pool", 0))
    stats = [
        _pool_stats(pool) if pool.size >= min_pool else flat_stats for pool in pools
    ]
    index = np.clip(
        np.searchsorted(edges, predictions, side="right") - 1, 0, len(pools) - 1
    )
    for i, (q10, q90, spread, count) in enumerate(stats):
        rows = np.isfinite(predictions) & (index == i)
        if not rows.any():
            continue
        p10[rows] = _floored(predictions[rows] + q10, floor)
        p90[rows] = _floored(predictions[rows] + q90, floor)
        sd[rows] = spread
        n[rows] = count
    return p10, p90, sd, n


def _present(value: Any) -> float | None:
    """legacy ``_feature_value``: ``None`` for absent/None/NaN, else the float
    (an infinite value is present, as ``pd.notna`` says)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def gate_forecast_columns(
    forecast: float | None,
    pool: Mapping[str, Any] | None,
    implied: Any,
) -> dict[str, float]:
    """engine/score.py ``Scorer._forecast_for_gate``, from a served forecast.

    ``forecast`` is the size fold's prediction (``None``/NaN when the fold
    declined: every column stays NaN, as legacy's early return leaves them).
    ``pool`` is the fold's ``{"predictions", "residuals", "interval_floor"}``;
    without it the band stays NaN (legacy's ``except: pass``).
    """
    out = {name: _NAN for name in GATE_FORECAST_COLUMNS}
    if forecast is None or forecast != forecast:
        return out
    out["pred_abs_move"] = float(forecast)
    if pool is not None:
        try:
            p10, p90, sd, _ = forecast_interval(
                [forecast], pool["predictions"], pool["residuals"],
                floor=pool.get("interval_floor", 0.0),
            )
            if np.isfinite(sd[0]):
                out["pred_abs_move_p10"] = float(p10[0])
                out["pred_abs_move_p90"] = float(p90[0])
                out["pred_abs_move_sd"] = float(sd[0])
        except (KeyError, TypeError, ValueError):
            pass
    im = _present(implied)
    if im is not None:
        out["forecast_edge"] = float(forecast) - im
    return out


def gate_analog_columns(values: Mapping[str, Any]) -> dict[str, float]:
    """engine/score.py ``Scorer._gate_feature_frame``'s analog block, read
    off this row's native analog-stage outputs.

    Legacy's ``n_analogs`` defaults to 0 because its analog layer always
    runs; natively an analog stage that did not run leaves ``n_analogs``
    unset, and that is NaN here (a missing input), never an invented 0.
    """
    mean, win, count = (values.get(name) for name in
                        ("exp_pnl_analog", "win_analog", "n_analogs"))
    return {
        "analog_mean": _NAN if mean is None else float(mean),
        "analog_win_rate": _NAN if win is None else float(win),
        "analog_n": _NAN if count is None else float(count),
    }


def chooser_direct_columns(strategy: str, values: Mapping[str, Any],
                           flags: Sequence[str]) -> dict[str, float]:
    """The chooser columns engine/score.py ``Scorer._chooser_frame`` reads
    straight off the scoring pass, by the same formulas.

    ``exp_pnl_sim``/``exp_pnl_sim_select`` (both the full-draw mean),
    the ``is_<member>`` one-hots, ``quote_repaired`` (training hard-codes
    0.0) and ``wide_market``. Every other chooser column is a declared
    feature (see the R4-20 row in the remaining-work guide).
    """
    sim = values.get("exp_pnl_sim")
    out: dict[str, float] = {"exp_pnl_sim": _NAN if sim is None else float(sim)}
    out["exp_pnl_sim_select"] = out["exp_pnl_sim"]
    for member in CHOOSER_MENU:
        out[f"is_{member.lower().replace('-', '_')}"] = 1.0 if member == strategy else 0.0
    out["quote_repaired"] = 0.0
    out["wide_market"] = 1.0 if "WIDE_MARKET" in flags else 0.0
    return out

