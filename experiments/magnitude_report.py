"""Read-only fixed mean-magnitude baseline over an admitted native temporal plan."""
import json
import math
from statistics import mean

from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import content_hash, from_document
from engine.v2.ops.errors import fail
from engine.v2.ops.native_registration import validate_magnitude_execution
from engine.v2.research.magnitude_inputs import (
    load_magnitude_targets,
    preflight_magnitude_targets,
    qualify_magnitude_training_targets,
)
from engine.v2.research.regression_evidence import error_metrics
from experiments.prediction_report import _years
from experiments.prediction_report import validate_prediction_spec as validate_base_spec


def validate_prediction_spec(spec):
    """Refuse nonfixed declarations before registration or any label access."""
    plan = validate_base_spec(spec)
    if plan.seed != 0:
        raise fail("INVALID_EXPERIMENT_SPEC", "magnitude mean baseline supports only seed zero")
    return plan


def magnitude_metrics(predictions):
    """Equal event weights; scale before squaring to preserve finite large errors."""
    errors = [abs(row["predicted_absolute_move_pct"] - row["absolute_move_pct"]) for row in predictions]
    try:
        return error_metrics(errors)
    except ValueError:
        message = "magnitude metrics require scored rows" if not errors else "magnitude metrics must be finite"
        raise fail("EXPERIMENT_VARIANT_FAILED", message) from None


def prediction_result(repository, registration):
    """Evaluate an admitted fixed plan; no writes, head lookup or final-read option."""
    binding = registration.document
    validate_magnitude_execution(binding)
    _years(binding["execution_plan"])
    plan = binding["magnitude_plan"]
    execution = binding["magnitude_execution"]
    snapshot = from_document(SnapshotRef, binding["snapshot"])
    month = binding["holdouts"]["holdout_as_of_month"]
    participants = sorted({event for fold in plan["folds"] for field in (
        "training_candidate_event_ids", "evaluation_event_ids") for event in fold[field]})
    preflight = preflight_magnitude_targets(repository, snapshot, as_of_month=month, event_ids=participants)
    events = {event["event_id"]: event for event in plan["events"]}
    folds, predictions = [], []
    for fold in plan["folds"]:
        train = qualify_magnitude_training_targets(repository, snapshot, as_of_month=month,
            labels_before=fold["training_labels_before"], event_ids=fold["training_candidate_event_ids"])
        fitted_mean = mean(train.absolute_move_pct.tolist())
        if not math.isfinite(fitted_mean):
            raise fail("EXPERIMENT_VARIANT_FAILED", "magnitude fitted mean must be finite")
        test = load_magnitude_targets(repository, snapshot, as_of_month=month,
            labels_before=execution["evaluation_labels_before"], event_ids=fold["evaluation_event_ids"])
        scored = [{"event_id": row.event_id, "fold": fold["year"], "absolute_move_pct": row.absolute_move_pct,
            "predicted_absolute_move_pct": fitted_mean, "population_use": row.population_use,
            "forecasts": events[row.event_id]["forecasts"]} for row in test.itertuples(index=False)]
        predictions.extend(scored)
        folds.append({**fold, "train_count": len(train), "test_count": len(test),
            "train_event_ids": train.event_id.tolist(), "test_event_ids": test.event_id.tolist(),
            "mean_absolute_move_pct": fitted_mean, "training_inputs": dict(train.attrs),
            "evaluation_inputs": dict(test.attrs), "metrics": magnitude_metrics(scored)})
    metrics = magnitude_metrics(predictions)
    result = {"schema_version": "native_magnitude_result.v1", "target": plan["target_recipe"],
        "model": execution["model"], "features": execution["features"], "attempted_variants": 1,
        "variant_id": registration.variant_id, "run_id": registration.run_id, "provenance": binding,
        "preflight": preflight, "registered_population_count": len(binding["event_ids"]),
        "population_labels": sorted({row["population_use"] for row in predictions}),
        "forecast_count": len(predictions) * len(plan["horizons"]),
        "weighting": "equal_event_weight; identical_intercept_for_all_horizons",
        "holdout_results": {"random": "excluded; not evaluated", "rolling": "excluded; not evaluated"},
        "metrics": metrics, "per_horizon": [{"horizon_sessions": horizon, **metrics} for horizon in plan["horizons"]],
        "quantile_calibration": "not_implemented", "folds": folds, "predictions": predictions}
    return {**result, "result_id": content_hash(result)}


def render_prediction_report(registration, result, *, spec, no_ledger):
    """Render deterministic native evidence without publishing or claiming trading returns."""
    validate_prediction_spec(spec)
    binding = registration.document
    if (spec.spec_hash != binding["spec_hash"] or result["variant_id"] != registration.variant_id
            or result["provenance"] != binding or result["run_id"] != registration.run_id
            or result["result_id"] != content_hash({key: value for key, value in result.items() if key != "result_id"})):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "magnitude report differs from its registered result")
    metrics = result["metrics"]
    lines = ["# Native magnitude prediction report", "", spec.hypothesis, "",
        f"Recording mode: {'no-ledger smoke; not recorded' if no_ledger else 'recorded evaluation'}.",
        f"Run: {registration.run_id}; variant: {registration.variant_id}.",
        f"Snapshot: {binding['snapshot']['snapshot_id']}; spec: {binding['spec_hash']}.",
        "Target: absolute earnings close-close return, percentage points; split-neutral, non-dividend-adjusted.",
        "Estimand: conditional mean. Baseline: fold-local arithmetic mean with a constant intercept.",
        "Forecast horizons: 5, 3 and 1 full trading sessions; identical intercept, one weight per event.",
        "Random holdout: excluded; not evaluated. Rolling holdout: excluded; not evaluated.",
        f"OOF events: {metrics['count']}; forecasts: {result['forecast_count']}; RMSE: {metrics['rmse']:.8g}; MAE: {metrics['mae']:.8g}.",
        "", "## Folds", "| Year | Train | Test | Mean magnitude | RMSE | MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    lines.extend(f"| {fold['year']} | {fold['train_count']} | {fold['test_count']} | {fold['mean_absolute_move_pct']:.8g} | {fold['metrics']['rmse']:.8g} | {fold['metrics']['mae']:.8g} |"
                 for fold in result["folds"])
    lines += ["", "## Limitations",
        "Training labels are qualified strictly before the sealed fit boundary; evaluation labels must be complete before the explicit evaluation cutoff.",
        "RMSE pools event losses; it is not an average of fold RMSEs. MAE and each horizon use the same event population.",
        "Schedules and publication availability remain unverified reconstructed date bounds, not observed point-in-time evidence.",
        "Quantile calibration, market/history features, search, holdout evaluation and trading-return claims are not implemented.",
        "", "## Complete metrics and provenance", "```json", json.dumps(result, sort_keys=True, indent=2, allow_nan=False), "```", ""]
    return "\n".join(lines)
