"""Source-free forecast scores; callers own admission and target access order."""
from math import fsum, isfinite

from engine.v2.ops.errors import fail
from engine.v2.research.magnitude_distribution import bin_index, histogram, recipe
from engine.v2.research.magnitude_distribution_math import bin_scores, crps, empirical_crps
from engine.v2.research.progress import Progress, stage
from experiments.distribution_oof import control_for_prediction

__all__ = ["arm_forecasts", "score_forecasts"]


def _require(condition):
    if not condition:
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution run contract")


def _require_pool_binding(receipt, pool):
    _require(type(receipt) is dict and receipt.get("status") == "completed"
            and type(receipt.get("claim")) is dict and type(pool) is dict
            and type(pool.get("destination")) is dict)
    claim, destination = receipt["claim"], pool["destination"]
    _require(type(claim.get("trial")) is dict)
    for source, keys in ((claim, ("plan_id", "admission_id", "role", "outer_year", "inner_year")),
            (claim["trial"], ("recipe_id", "context_id", "fold_id", "inputs_id", "budget_id", "horizon", "role"))):
        _require(all(key in source and key in destination and source[key] == destination[key] for key in keys))


def arm_forecasts(row, tail, pool, benchmark=None, *, receipt):
    _require_pool_binding(receipt, pool)
    document = {**recipe(), "probabilities": row["probabilities"], "tail": tail}
    controls = control_for_prediction(pool, document, row["point_prediction"])
    values = {"histogram": document,
        "same_mean": controls["histogram"].get("magnitudes"), "matched_point": controls["point"].get("magnitudes")}
    if benchmark is not None:
        values["incumbent"] = benchmark["incumbent_magnitudes"]
    return values


def score_forecasts(result, pool, targets, *, outer_year, horizon):
    """Score supplied outcomes; missing targets and controls retain explicit rows."""
    _require(type(result) is dict)
    _require_pool_binding(result.get("receipt"), pool)
    destination = pool["destination"]
    _require(type(outer_year) is str and type(horizon) is int
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
