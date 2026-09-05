"""Sizing a structure from a feature model's forecast.

Step 4 of ``guides/tier4_feature_models.md``: the bridge between the Tier-4
forecast and :attr:`engine.score.ScoreRequest.structure_params`.

The whole reason Tier 4 exists is on the other side of this module. EXP-125
sized TWIN-P's tent from the size model's predicted move and it was the only
lever in five experiments that moved the number that mattered — the share of
prints landing beyond a wing fell 37.5% → 16.7%. Putting that on the live board
hit a dependency cycle::

    structure needs w  →  w = f(pred_abs_move)  →  prediction needs features
       ↑                                                    │
       └──── pricing needs the structure ←── _features needs entry_cost, spot

The break is that the size model's fourteen features are **entirely
pricing-free** — all Tier-3 panel columns, none from a chain. So the forecast
was never downstream of pricing; it was only *computed* there. This module is
where the ordering is made explicit: forecast first, structure second, price
third.

Sizing is a per-strategy rule, kept here rather than in
``engine.structures``, because it is a claim about what a forecast MEANS for a
shape, not about the shape itself. A structure factory should stay ignorant of
which model happens to be pointed at it.
"""
from __future__ import annotations

from typing import Callable, Mapping

__all__ = [
    "PLATEAU_CENTRE",
    "WIDTH_MIN",
    "WIDTH_MAX",
    "FORECAST_SIZED",
    "twin_p_params",
    "twin_p5_params",
    "forecast_params",
]

#: TWIN-P's payoff is flat at its maximum for a move between ``w`` and ``2w``,
#: so the plateau's centre sits at ``1.5w``. Sizing puts the FORECAST at that
#: centre — ``w = forecast / 1.5`` — rather than at a wing or at the peak, so
#: the structure pays its maximum across the band the forecast actually
#: brackets rather than only if the forecast is exactly right. Registered in
#: EXP-125's spec.yaml before the experiment ran.
PLATEAU_CENTRE = 1.5

#: Bounds on the resulting width, as a share of spot. Below the floor the seven
#: strikes collapse onto too few ladder positions to be distinct; above the
#: ceiling the wings sit ±60% of spot away, where nothing is listed. A forecast
#: landing outside them yields NO structure rather than a clipped one — clipping
#: would silently trade a shape the forecast did not ask for.
WIDTH_MIN, WIDTH_MAX = 0.005, 0.15


def twin_p_params(pred_abs_move: float) -> dict | None:
    """``width_moneyness`` for TWIN-P, or ``None`` if the forecast cannot size it."""
    try:
        forecast = float(pred_abs_move)
    except (TypeError, ValueError):
        return None
    if not forecast > 0 or forecast != forecast:  # NaN is not a forecast
        return None
    width = (forecast / 100.0) / PLATEAU_CENTRE
    if not WIDTH_MIN <= width <= WIDTH_MAX:
        return None
    return {"width_moneyness": width}


#: TWIN-P5 peaks at exactly ``+/-a`` rather than across a plateau, so the
#: forecast goes on the peak itself and the divisor is 1, not
#: :data:`PLATEAU_CENTRE`. Getting this wrong is not a rounding: sizing the
#: five-strike shape through 1.5 would put its peak at two thirds of the
#: predicted move and its wings at twice it, which is a different trade from
#: the one EXP-126 measured.
FIVE_STRIKE_PEAK = 1.0


def twin_p5_params(pred_abs_move: float) -> dict | None:
    """``width_moneyness`` for TWIN-P5, or ``None`` if the forecast cannot size it."""
    try:
        forecast = float(pred_abs_move)
    except (TypeError, ValueError):
        return None
    if not forecast > 0 or forecast != forecast:
        return None
    width = (forecast / 100.0) / FIVE_STRIKE_PEAK
    if not WIDTH_MIN <= width <= WIDTH_MAX:
        return None
    return {"width_moneyness": width}


#: Strategies whose shape is set per event by a Tier-4 forecast, and the rule
#: that turns the forecast into structure parameters.
FORECAST_SIZED: dict[str, Callable[[float], dict | None]] = {
    "TWIN-P": twin_p_params,
    "TWIN-P5": twin_p5_params,
}


def forecast_params(strategy: str, pred_abs_move: float) -> dict | None:
    """Structure parameters for ``strategy`` at this forecast, or ``None``.

    ``None`` means *decline to size*, and every caller must treat it that way.
    A NULL forecast is not a zero-width tent; it is the absence of a decision.
    """
    rule = FORECAST_SIZED.get(strategy)
    return None if rule is None else rule(pred_abs_move)


def sized_strategies() -> tuple[str, ...]:
    return tuple(sorted(FORECAST_SIZED))


def describe(strategy: str, params: Mapping | None) -> str:
    """One-line human summary of a sizing decision, for a score's detail line."""
    if not params:
        return f"{strategy}: no forecast, not sized"
    if "width_moneyness" in params:
        return f"{strategy}: tent width {float(params['width_moneyness']) * 100:.2f}% of spot"
    return f"{strategy}: {dict(params)}"
