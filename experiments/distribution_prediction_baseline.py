"""Training-only empirical reference from freshly verified distribution fit evidence."""
from copy import deepcopy
from math import fsum, isfinite

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.ops.errors import fail
from engine.v2.research.magnitude_distribution import bin_index, histogram
from engine.v2.research.magnitude_distribution_math import bin_scores, prepare_empirical_crps
from engine.v2.research.progress import Progress, stage
from experiments.magnitude_search_evidence import OMISSION_REASONS, fit_evidence

__all__ = ["training_baseline", "score_training_baseline"]
_SCHEMA = "distribution_training_baseline.v1"


def _require(condition):
    if not condition:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution training baseline differs")


def training_baseline(fits, outer_year, horizon):
    """Reverify stored training and completed fit; never open evaluation outcomes.

    The returned hash detects mutation, not forged admission. Persist this
    evidence before targets; only this admitted builder establishes provenance.
    """
    context = fits.trials.read_fit_context("outer", outer_year, None)
    plan, fold, admission, inputs = (context[key] for key in ("plan", "fold", "admission", "inputs"))
    _require(type(horizon) is int and horizon in plan["recipe"]["horizons"])
    result = fits.read("outer", outer_year, horizon)
    receipt, training, events = result["receipt"], inputs["training"], fold["evaluation_event_ids"]
    train_ids = [row["event_id"] for row in training]
    _require(bool(training) and bool(events) and len(set(train_ids)) == len(train_ids)
             and len(set(events)) == len(events) and not set(train_ids).intersection(events))
    evidence = fit_evidence(inputs, fold, {"horizons": [horizon]},
                            {"features": plan["recipe"]["point_recipe"]["features"]})[0]
    expected = {"plan_id": plan["plan_id"], "admission_id": content_hash(admission),
        "trial": next(item for item in admission["trials"] if item["horizon"] == horizon),
        "role": "outer", "outer_year": outer_year, "inner_year": None,
        "evidence": {key: evidence[key] for key in ("horizon_sessions", "training_matrix_id", "prediction_matrix_id")},
        "ordered_events_id": content_hash([train_ids, events])}
    _require(receipt["status"] == "completed" and canonical_json(receipt["claim"]) == canonical_json(expected)
             and receipt["result_id"] == content_hash({key: value for key, value in receipt.items() if key != "result_id"})
             and [row["event_id"] for row in result["rows"]] == events)
    # Same admitted label rows and order as the fitted pair, never residuals or outer labels.
    progress = Progress("distribution training baseline verification", len(training))
    for row in training:
        bin_index(row["absolute_move_pct"])
        progress.tick()
    body = {"schema_version": _SCHEMA, "policy": "same_admitted_training_equal_weight.v1",
        "fit_claim": expected, "result_id": receipt["result_id"], "training": training,
        "evaluation_event_ids": events, "training_labels_before": fold["training_labels_before"],
        "final_read_authority": "none"}
    return deepcopy({**body, "baseline_id": content_hash(body)})


def score_training_baseline(baseline, targets, *, receipt):
    """Score a saved admitted baseline against supplied complete target/status rows.

    Bind the matching fit before target access; sort/bin the shared reference
    once, not per event. This numeric replay grants no source-read authority.
    """
    try:
        _require(type(baseline) is dict and baseline.get("schema_version") == _SCHEMA
                 and baseline.get("policy") == "same_admitted_training_equal_weight.v1"
                 and baseline["baseline_id"] == content_hash({key: value for key, value in baseline.items() if key != "baseline_id"}))
        _require(type(receipt) is dict and receipt["status"] == "completed"
                 and receipt["result_id"] == baseline["result_id"]
                 and canonical_json(receipt["claim"]) == canonical_json(baseline["fit_claim"]))
        events, claim, training = baseline["evaluation_event_ids"], baseline["fit_claim"], baseline["training"]
        _require(type(events) is list and bool(events) and all(type(event) is str and event for event in events)
                 and len(events) == len(set(events)) and type(claim) is dict and type(claim["trial"]) is dict
                 and type(claim["outer_year"]) is str and bool(claim["outer_year"])
                 and type(claim["trial"]["horizon"]) is int and claim["trial"]["horizon"] in (1, 3, 5))
        _require(type(training) is list and bool(training)
                 and all(type(row) is dict and set(row) == {"event_id", "absolute_move_pct"} for row in training))
        samples = [row["absolute_move_pct"] for row in training]
        _require(type(targets) is dict and set(targets) == set(events))
        for value in targets.values():
            if type(value) is dict:
                _require(set(value) == {"omitted"} and type(value["omitted"]) is str and value["omitted"] in OMISSION_REASONS)
            else:
                _require(type(value) in (int, float) and isfinite(value) and value >= 0)
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution training baseline differs") from None
    score = prepare_empirical_crps(samples)
    counts = [0] * 81
    for value in samples:
        counts[bin_index(value)] += 1
    masses = [count / len(samples) for count in counts]
    document = histogram(masses)
    tails = [fsum(masses[bin_index(threshold):]) / fsum(masses) for threshold in (5, 10, 20)]
    progress = Progress("distribution empirical baseline scores", len(events))
    with stage("distribution empirical baseline scores"):
        for event in events:
            observation = targets[event]
            row = {"event_id": event, "outer_year": claim["outer_year"], "horizon": claim["trial"]["horizon"],
                "arm": "training_empirical", "observation": observation, "status": "available"}
            if isinstance(observation, dict):
                row.update(status="missing_target", observation=None, reason=observation["omitted"])
            else:
                scores = bin_scores(document, observation)
                row.update(crps=score(observation), rps=scores["rps"],
                    log_loss=scores["log_loss"] if isfinite(scores["log_loss"]) else "infinite",
                    tails=[{"threshold": threshold, "probability_ge": probability, "observed_ge": observation >= threshold}
                        for threshold, probability in zip((5, 10, 20), tails)])
            yield row
            progress.tick()
