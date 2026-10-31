"""Per-opportunity proper scores and common frozen-bridge decisions."""
from math import fsum, isfinite

from engine.v2.research.magnitude_distribution import bin_index, histogram, recipe
from engine.v2.research.magnitude_distribution_math import bin_scores, brier, crps, empirical_crps
from engine.v2.research.magnitude_distribution_payoff import (
    bridge_summary,
    empirical_bridge_summary,
)
from engine.v2.research.progress import Progress, stage
from experiments.distribution_oof import control_for_prediction
from experiments.distribution_run_request import number, require


def _require_pool_binding(receipt, pool):
    require(type(receipt) is dict and receipt.get("status") == "completed"
            and type(receipt.get("claim")) is dict and type(pool) is dict
            and type(pool.get("destination")) is dict)
    claim, destination = receipt["claim"], pool["destination"]
    require(type(claim.get("trial")) is dict)
    for source, keys in ((claim, ("plan_id", "admission_id", "role", "outer_year", "inner_year")),
            (claim["trial"], ("recipe_id", "context_id", "fold_id", "inputs_id", "budget_id", "horizon", "role"))):
        require(all(key in source and key in destination and source[key] == destination[key] for key in keys))


def arm_forecasts(row, tail, pool, benchmark=None, *, receipt):
    _require_pool_binding(receipt, pool)
    document = {**recipe(), "probabilities": row["probabilities"], "tail": tail}
    controls = control_for_prediction(pool, document, row["point_prediction"])
    values = {"histogram": document,
        "same_mean": controls["histogram"].get("magnitudes"), "matched_point": controls["point"].get("magnitudes")}
    if benchmark is not None:
        values["incumbent"] = benchmark["incumbent_magnitudes"]
    return values


def economic_forecast(value, bridge, decision, *, scale, round_trip_cost):
    """P(net profit) and net EV use identical per-share bridge/cost in every arm."""
    scale, round_trip_cost = number(scale, positive=True), number(round_trip_cost)
    if value is None or decision is None or (isinstance(value, dict) and value["tail"]["policy"] == "unresolved"):
        return None
    args = dict(spot=decision.spot, entry_cost=decision.cost + round_trip_cost / scale,
                intercept=bridge.intercept, slope=bridge.slope, payoff_residuals=list(bridge.residuals))
    result = bridge_summary(value, **args) if isinstance(value, dict) else empirical_bridge_summary(value, **args)
    return {**result, "expected_net_profit_usd": result["expected_profit"] * scale}


def score_forecasts(result, pool, targets, *, outer_year, horizon):
    """Targets arrive only after decisions; missing controls retain explicit rows."""
    require(type(result) is dict)
    _require_pool_binding(result.get("receipt"), pool)
    destination = pool["destination"]
    require(type(outer_year) is str and type(horizon) is int
            and destination["outer_year"] == outer_year and destination["horizon"] == horizon
            and destination["role"] == "outer" and destination["inner_year"] is None)
    progress = Progress("distribution proper scores", len(result["rows"]))
    with stage("distribution proper scores"):
        for row in result["rows"]:
            observation = targets[row["event_id"]]
            values = arm_forecasts(row, result["tail"], pool, receipt=result["receipt"])
            for arm, value in values.items():
                output = {"event_id": row["event_id"], "outer_year": outer_year, "horizon": horizon,
                          "arm": arm, "observation": observation, "status": "unavailable"}
                if isinstance(observation, dict):
                    output.update(status="missing_target", observation=None, reason=observation["omitted"])
                elif value is not None:
                    masses = value["probabilities"] if isinstance(value, dict) else [0] * 81
                    if not isinstance(value, dict):
                        for sample in value:
                            masses[bin_index(sample)] += 1
                        masses = [count / len(value) for count in masses]
                    scores = bin_scores(histogram(masses), observation)
                    loss = scores["log_loss"]
                    output.update(status="available", rps=scores["rps"], log_loss=loss if isfinite(loss) else "infinite",
                        crps=(None if value["tail"]["policy"] == "unresolved" else crps(value, observation))
                             if isinstance(value, dict) else empirical_crps(value, observation),
                        tails=[{"threshold": threshold, "probability_ge": fsum(masses[bin_index(threshold):]) / fsum(masses),
                                "observed_ge": observation >= threshold} for threshold in (5, 10, 20)])
                yield output
            progress.tick()


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
