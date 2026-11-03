"""Per-opportunity proper scores and common frozen-bridge decisions."""
from math import isfinite

from engine.v2.research.magnitude_distribution_math import brier
from engine.v2.research.magnitude_distribution_payoff import (
    bridge_summary,
    empirical_bridge_summary,
)
from experiments.distribution_prediction_scores import arm_forecasts, score_forecasts
from experiments.distribution_run_request import number, require

__all__ = ["arm_forecasts", "score_forecasts", "economic_forecast", "profit_scores"]


def economic_forecast(value, bridge, decision, *, scale, round_trip_cost):
    """P(net profit) and net EV use identical per-share bridge/cost in every arm."""
    scale, round_trip_cost = number(scale, positive=True), number(round_trip_cost)
    if value is None or decision is None or (isinstance(value, dict) and value["tail"]["policy"] == "unresolved"):
        return None
    args = dict(spot=decision.spot, entry_cost=decision.cost + round_trip_cost / scale,
                intercept=bridge.intercept, slope=bridge.slope, payoff_residuals=list(bridge.residuals))
    result = bridge_summary(value, **args) if isinstance(value, dict) else empirical_bridge_summary(value, **args)
    return {**result, "expected_net_profit_usd": result["expected_profit"] * scale}


def profit_scores(intents, quotes):
    """Score decision P(net profit) on all quoted opportunities, not selected survivors."""
    for row in intents:
        quote = quotes.get(row["event_id"])
        require(quote is None or all(value is None or type(value) is int
            or (type(value) is float and isfinite(value))
            for value in (quote["entry_cash"], quote["exit_cash"])))
        observed = None if quote is None or quote["entry_cash"] is None or quote["exit_cash"] is None else quote["exit_cash"] > quote["entry_cash"]
        for arm, forecast in row["forecasts"].items():
            # Validate every present probability; unknown truth still withholds the loss.
            loss = None if forecast is None else brier(forecast["probability_profit"], observed is True)
            yield {"event_id": row["event_id"], "arm": arm, "observed_profit": observed,
                "probability_profit": None if forecast is None else forecast["probability_profit"],
                "brier": None if observed is None else loss}
