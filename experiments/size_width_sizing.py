"""EXP-189 k sizing module: k multiplies production's forecast_sizing width; WIDTH_MIN/WIDTH_MAX apply after k.

``width = k * (forecast / 100) / divisor`` for the seven put-side short-vol structures, with the divisors of
``engine.forecast_sizing``. k = 1 reproduces production exactly (``production_params``). A width outside the
bounds is declined (None), and the bounds are tested AFTER k, so an unscaled width outside them may be
brought inside by k. Pure: no I/O and no logging.
"""
import math

from engine.forecast_sizing import (
    FIVE_STRIKE_PEAK,
    PLATEAU_CENTRE,
    SHORT_VOL_OUTER,
    WIDTH_MAX,
    WIDTH_MIN,
    forecast_params,
)

GRID = (0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)

DIVISORS = {"TWIN-P": PLATEAU_CENTRE, "TWIN-P5": FIVE_STRIKE_PEAK, **SHORT_VOL_OUTER}


def k_params(strategy, pred_abs_move, k):
    """``{"width_moneyness": width}`` at multiple ``k`` of production's width, or None (declined)."""
    try:
        k_value = float(k)
    except (TypeError, ValueError):
        raise ValueError("k must be a positive finite number") from None
    if not (math.isfinite(k_value) and k_value > 0):
        raise ValueError("k must be a positive finite number")
    if strategy not in DIVISORS:
        return None
    try:
        forecast = float(pred_abs_move)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(forecast) and forecast > 0):  # NaN, inf and non-positive are not forecasts
        return None
    width = k_value * (forecast / 100.0) / DIVISORS[strategy]
    if not WIDTH_MIN <= width <= WIDTH_MAX:
        return None
    return {"width_moneyness": width}


def production_params(strategy, pred):
    """Production's own sizing, unchanged: ``forecast_params(strategy, pred)``."""
    return forecast_params(strategy, pred)
