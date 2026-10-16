"""Sealed fixed-mean or fitted magnitude candidates over stored history."""
import json
import math
from statistics import mean

from engine.v2.contracts import SnapshotRef
from engine.v2.data.errors import DataError
from engine.v2.foundation import canonical_json, content_hash, from_document
from engine.v2.ops.errors import fail
from engine.v2.ops.native_registration import (
    validate_magnitude_fitted_execution,
    validate_magnitude_history_execution,
)
from engine.v2.research.magnitude_history_store import read_stored
from engine.v2.research.magnitude_history_table import FEATURES, METADATA, TABLE
from engine.v2.research.magnitude_inputs import (
    load_magnitude_targets,
    preflight_magnitude_targets,
    qualify_magnitude_training_targets,
)
from experiments.magnitude_fitted import fitted_forecasts
from experiments.magnitude_history_inputs import history_binding
from experiments.magnitude_report import magnitude_metrics, validate_prediction_spec
from experiments.prediction_report import _years


def _validate(binding):
    """Select the exact sealed policy without broadening older schema admission."""
    if not isinstance(binding, dict) or binding.get("schema_version") not in (
            "native_experiment_registration.v1.4", "native_experiment_registration.v1.5"):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "history execution requires a sealed v1.4 plan")
    fitted = binding["schema_version"] == "native_experiment_registration.v1.5"
    (validate_magnitude_fitted_execution if fitted else validate_magnitude_history_execution)(binding)
    return fitted


def prediction_result(repository, registration):
    """Consume pinned features only; freeze each fold's forecasts before its labels."""
    binding = registration.document
    fitted = _validate(binding)
    _years(binding["execution_plan"])
    plan, execution, history = (binding[k] for k in ("magnitude_plan", "magnitude_execution", "magnitude_history"))
    snapshot = from_document(SnapshotRef, binding["snapshot"])
    feature = from_document(SnapshotRef, history["feature_snapshot"])
    month = binding["holdouts"]["holdout_as_of_month"]
    admitted = history_binding(repository, snapshot, plan, binding["magnitude_history_request"],
        as_of_month=month, event_ids=binding["event_ids"])
    if admitted != history:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "history producer binding changed before execution")
    participants = sorted({event for fold in plan["folds"] for field in (
        "training_candidate_event_ids", "evaluation_event_ids") for event in fold[field]})
    preflight = preflight_magnitude_targets(repository, snapshot, as_of_month=month, event_ids=participants)
    try:
        stored = read_stored(repository, feature, history["context"])
    except DataError as error:
        if error.code == "IDENTITY_CONFLICT":
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "stored feature identity changed after admission") from None
        raise
    indexed = {(row["event_id"], row["horizon_sessions"]): row for row in stored["rows"]}
    events = {event["event_id"]: event for event in plan["events"]}
    folds, predictions = [], []
    for fold in plan["folds"]:
        train = qualify_magnitude_training_targets(repository, snapshot, as_of_month=month,
            labels_before=fold["training_labels_before"], event_ids=fold["training_candidate_event_ids"])
        if fitted:
            estimates, fit_evidence = fitted_forecasts(indexed, train, fold, plan, execution)
            fit_details = {"fits": fit_evidence}
        else:
            fallback = mean(train.absolute_move_pct.tolist())
            if not math.isfinite(fallback):
                raise fail("EXPERIMENT_VARIANT_FAILED", "magnitude fallback mean must be finite")
            fit_details = {"mean_absolute_move_pct": fallback}
        frozen = []
        for event_id in fold["evaluation_event_ids"]:
            for clock in events[event_id]["forecasts"]:
                row = indexed[event_id, clock["horizon_sessions"]]
                frozen.append({"event_id": event_id, "fold": fold["year"],
                    "horizon_sessions": clock["horizon_sessions"], "forecast": clock,
                    "predicted_absolute_move_pct": estimates[event_id, clock["horizon_sessions"]] if fitted
                        else row["mean_prior_abs_move"] if row["n_prior"] else fallback,
                    "source": execution["model"] if fitted else "stored_history" if row["n_prior"] else "fold_mean",
                    "n_prior": row["n_prior"]})
        test = load_magnitude_targets(repository, snapshot, as_of_month=month,
            labels_before=execution["evaluation_labels_before"], event_ids=fold["evaluation_event_ids"])
        labels = test.set_index("event_id")
        scored = [{**row, "absolute_move_pct": float(labels.loc[row["event_id"], "absolute_move_pct"]),
            "population_use": labels.loc[row["event_id"], "population_use"]} for row in frozen]
        predictions.extend(scored)
        folds.append({**fold, "train_count": len(train), "test_count": len(test),
            "train_event_ids": train.event_id.tolist(), "test_event_ids": test.event_id.tolist(),
            **fit_details, "training_inputs": dict(train.attrs),
            "evaluation_inputs": dict(test.attrs), "metrics": magnitude_metrics(scored)})
    consumed = {"inputs_id": stored["inputs_id"], "feature_snapshot": history["feature_snapshot"],
        "columns": list((*METADATA, *FEATURES)), "row_keys": [
            [row[key] for key in ("event_id", "context_id", "horizon_sessions")] for row in stored["rows"]],
        "objects": [{"fragment_id": r.fragment_id, "object_id": r.object_ref.object_id,
                     "content_hash": r.object_ref.content_hash} for r in repository.fragment_records(feature, TABLE)]}
    result = {"schema_version": "native_magnitude_fitted_result.v1" if fitted else "native_magnitude_history_result.v1",
        "target": plan["target_recipe"],
        "model": execution["model"], "features": execution["features"], "attempted_variants": 1,
        "variant_id": registration.variant_id, "run_id": registration.run_id, "provenance": binding,
        "preflight": preflight, "stored_inputs": consumed, "registered_population_count": len(binding["event_ids"]),
        "population_labels": sorted({row["population_use"] for row in predictions}),
        "event_count": sum(fold["test_count"] for fold in folds), "forecast_count": len(predictions),
        "weighting": execution["weighting"], "metrics": magnitude_metrics(predictions),
        "per_horizon": [{"horizon_sessions": horizon, **magnitude_metrics([
            row for row in predictions if row["horizon_sessions"] == horizon])} for horizon in plan["horizons"]],
        "holdout_results": {"random": "excluded; not evaluated", "rolling": "excluded; not evaluated"},
        "quantile_calibration": "not_implemented", "folds": folds, "predictions": predictions}
    if fitted:
        result["fit_recipe"] = execution["fit_recipe"]
    return {**result, "result_id": content_hash(result)}


def render_prediction_report(registration, result, *, spec, no_ledger):
    """Render deterministic complete native evidence without publishing anything."""
    validate_prediction_spec(spec)
    fitted = _validate(registration.document)
    schema = "native_magnitude_fitted_result.v1" if fitted else "native_magnitude_history_result.v1"
    execution = registration.document["magnitude_execution"]
    provenance_matches = (canonical_json(result["provenance"]) == canonical_json(registration.document)
                          if fitted else result["provenance"] == registration.document)
    if (spec.spec_hash != registration.document["spec_hash"] or result["variant_id"] != registration.variant_id
            or not provenance_matches or result["run_id"] != registration.run_id
            or result["schema_version"] != schema
            or (fitted and any(canonical_json(result.get(key)) != canonical_json(execution[key])
                               for key in ("model", "features", "fit_recipe", "weighting")))
            or (fitted and canonical_json(result.get("target")) != canonical_json(
                registration.document["magnitude_plan"]["target_recipe"]))
            or result["result_id"] != content_hash({k: v for k, v in result.items() if k != "result_id"})):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "history report differs from its registered result")
    metrics = result["metrics"]
    title = "# Native fitted-history magnitude report" if fitted else "# Native stored-history magnitude report"
    return "\n".join([title, "", spec.hypothesis, "",
        f"Recording mode: {'no-ledger smoke; not recorded' if no_ledger else 'recorded evaluation'}.",
        f"Run: {registration.run_id}; variant: {registration.variant_id}.",
        "Target: absolute split-neutral, non-dividend-adjusted earnings close-close return in percentage points.",
        (f"Conditional mean: fixed {execution['model']} fitted independently per fold and horizon; no search."
         if fitted else "Conditional mean: stored ticker arithmetic mean; empty history uses the strictly qualified fold mean."),
        "Forecasts: 5, 3 and 1 full trading sessions; equal event and horizon weights, never average fold RMSEs.",
        f"OOF events: {result['event_count']}; forecasts: {result['forecast_count']}; RMSE: {metrics['rmse']:.8g}; MAE: {metrics['mae']:.8g}.",
        "Random and rolling holdouts excluded; not evaluated. Schedules/publication are unverified reconstructions.",
        "Quantiles, search, comparative winner selection and trading returns are not implemented.",
        "", "## Complete metrics and provenance", "```json",
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False), "```", ""])
