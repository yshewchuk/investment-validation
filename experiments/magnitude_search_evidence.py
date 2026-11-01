"""Shared native fold inputs and pure evidence checks for inner/outer search."""
import math
from numbers import Real

from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import canonical_json, content_hash, from_document
from engine.v2.research.magnitude_history_store import read_stored_for_events
from engine.v2.research.regression_evidence import normalize_forecasts as normalize_regression_forecasts
from engine.v2.research.magnitude_inputs import (
    magnitude_recipe,
    preflight_magnitude_targets,
    qualify_magnitude_training_targets,
)
from experiments.magnitude_history_inputs import history_binding

OMISSION_REASONS = frozenset({"skipped", "not_orats", "no_session", "no_price_window",
    "no_price_history", "ticker_too_few_events"})


def admit_fold_inputs(repository, binding, fold, *, history_failure, strict_json=False, plan=None):
    """Load preflighted native inputs; the caller supplies its history-conflict envelope."""
    if binding.get("schema_version") == "native_experiment_registration.v1.8":
        from experiments.regression_search_binding import fold_inputs
        return fold_inputs(repository, binding, fold, plan=plan)
    if binding.get("schema_version") in ("native_experiment_registration.v1.6", "native_experiment_registration.v1.7"):
        from experiments.size_search_binding import fold_inputs
        return fold_inputs(repository, binding, fold)
    snapshot = from_document(SnapshotRef, binding["snapshot"])
    month = binding["holdouts"]["holdout_as_of_month"]
    history = history_binding(repository, snapshot, binding["magnitude_plan"],
        binding["magnitude_history_request"], as_of_month=month, event_ids=binding["event_ids"])
    same = (canonical_json(history) == canonical_json(binding["magnitude_history"])) if strict_json else history == binding["magnitude_history"]
    if not same:
        raise history_failure
    participants = set(fold["training_candidate_event_ids"] + fold["evaluation_event_ids"])
    preflight = preflight_magnitude_targets(repository, snapshot, as_of_month=month, event_ids=sorted(participants))
    stored = read_stored_for_events(repository, from_document(SnapshotRef, history["feature_snapshot"]),
        history["context"], event_ids=participants, columns=binding["magnitude_execution"]["features"])
    train = qualify_magnitude_training_targets(repository, snapshot, as_of_month=month,
        labels_before=fold["training_labels_before"], event_ids=fold["training_candidate_event_ids"])
    return {"preflight": preflight, "stored_inputs_id": stored["inputs_id"], "training_inputs": dict(train.attrs),
            "training": train[["event_id", "absolute_move_pct"]].to_dict("records"),
            "features": stored["rows"]}


def fit_evidence(inputs, fold, plan, execution):
    """Bind ordered native matrices and training targets without fitting."""
    indexed = {(row["event_id"], row["horizon_sessions"]): row for row in inputs["features"]}
    train = [row["event_id"] for row in inputs["training"]]
    result, target = [], inputs.get("target", {}).get("column", "absolute_move_pct")
    for horizon in plan["horizons"]:
        documents = []
        for events in (train, fold["evaluation_event_ids"]):
            rows = [indexed[event, horizon] for event in events]
            documents.append({"columns": execution["features"], "row_keys": [
                [row[name] for name in ("event_id", "context_id", "horizon_sessions")] for row in rows],
                "values": [[row[name] for name in execution["features"]] for row in rows]})
        documents[0][target] = [row[target] for row in inputs["training"]]
        result.append({"horizon_sessions": horizon, "training_event_ids": train,
            "evaluation_event_ids": fold["evaluation_event_ids"], "training_matrix_id": content_hash(documents[0]),
            "prediction_matrix_id": content_hash(documents[1])})
    return result


def normalize_forecasts(rows, fold, horizons, *, strict_json=False, allow_incomplete=False):
    """Require a complete raw row grid; explicit None values are dropped only in the opt-in path."""
    return normalize_regression_forecasts(rows, fold, horizons,
        prediction_column="predicted_absolute_move_pct",
        target_domain="signed" if allow_incomplete else "nonnegative",
        strict_json=strict_json, allow_abstentions=bool(allow_incomplete))


def validate_targets(preflight, fold, evaluation_inputs, targets, *, strict_json=False):
    """Reconstruct the protected receipt and exact ordered targets; malformed evidence raises ValueError."""
    events = set(fold["evaluation_event_ids"])
    consumed = [row for row in preflight["planned_inputs"] if row["event_id"] in events]
    ordered = [row["event_id"] for row in consumed]
    if len(ordered) != len(events) or set(ordered) != events:
        raise ValueError
    omitted = evaluation_inputs.get("omitted_labels", []) if isinstance(evaluation_inputs, dict) else []
    if not isinstance(omitted, list):
        raise ValueError
    omitted_ids = []
    for row in omitted:
        if (not isinstance(row, dict) or set(row) != {"event_id", "reason"}
                or not (type(row["reason"]) is str and row["reason"] in OMISSION_REASONS)
                or row["event_id"] not in events):
            raise ValueError
        omitted_ids.append(row["event_id"])
    if omitted_ids != sorted(omitted_ids) or len(set(omitted_ids)) != len(omitted_ids):
        raise ValueError
    recipe = magnitude_recipe()
    receipt = {"schema_version": "research_magnitude_targets.v1",
        **{name: preflight[name] for name in ("snapshot_id", "holdout_as_of_month", "random_membership_version",
            "rolling_membership_version", "population_use", "purpose", "computed_moves_version")},
        "labels_before": fold["evaluation_labels_before"], "target_recipe": recipe,
        "target_recipe_id": content_hash(recipe), "consumed_inputs": consumed, "excluded_future_events": []}
    if omitted:
        receipt["omitted_labels"] = omitted
    expected = {**receipt, "inputs_id": content_hash(receipt), "holdout_exclusions": []}
    same = (canonical_json(evaluation_inputs) == canonical_json(expected)) if strict_json else evaluation_inputs == expected
    if not same:
        raise ValueError
    if [row["event_id"] for row in targets] != [event for event in ordered if event not in set(omitted_ids)]:
        raise ValueError
    for row in targets:
        value = row["absolute_move_pct"]
        if (set(row) != {"event_id", "absolute_move_pct"} or isinstance(value, bool)
                or not isinstance(value, Real) or value < 0 or not math.isfinite(value)
                or (strict_json and type(value) not in (int, float))):
            raise ValueError
