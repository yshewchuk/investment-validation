"""Causal fourteen-input incumbent refits on admitted arrays, never production rows."""
import math
import time
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from statistics import mean

import numpy as np
from threadpoolctl import threadpool_limits

from engine.v2.foundation import content_hash
from engine.v2.models.frozen_state_io import BoundedStateReadError
from engine.v2.models.training.estimators import fit_recipe_estimator
from engine.v2.models.training.recipes import RecipeKey, current_recipes
from engine.v2.ops.errors import fail
from experiments.magnitude_search_evidence import fit_evidence as candidate_evidence
from experiments.magnitude_search_selection import history_mean_prediction

FEATURES = ["has_implied_quote", "mean_prior_abs_move", "abs_dist_ema", "ema12r_abs",
    "mean_prior_move", "signed_streak", "dist_high", "dist_ema", "spy_vol20", "spy_dd252",
    "mean_prior_or_implied", "or_implied", "or_rvol30", "mcap_log"]
COLUMNS = list(dict.fromkeys(["n_prior", *[{'or_implied': 'implied_move',
    'or_rvol30': 'rvol30'}.get(name, name) for name in FEATURES], "current_quote_status"]))


def native_recipe():
    """Refuse incumbent drift; reuse native constructors, not the live-panel loader."""
    try:
        recipe = current_recipes()[RecipeKey("size", "*", "champion")]
    except BoundedStateReadError as error:
        raise fail("RESOURCE_LIMIT_EXCEEDED" if error.code == "SIZE_LIMIT" else "RESOURCE_UNAVAILABLE",
                   "incumbent registry failed bounded admission") from None
    if (list(recipe.features) != FEATURES or recipe.estimator.kind != "ols_mlp_blend"
            or recipe.estimator.seeds != (20260829,) or recipe.folds.min_train_rows != 500
            or recipe.estimator.params != {"mlp": {"hidden_layer_sizes": [64, 32], "max_iter": 400,
                "early_stopping": True, "n_iter_no_change": 15}, "scale_mlp_inputs": True,
                "blend": "equal_weight_mean(ols, mlp)"}):
        raise ValueError("incumbent recipe changed; explicit reviewed migration required")
    return replace(recipe, dataset_source="qualified stored research panel; causal refit",
                   clock_id="earnings_forecast_cutoff.v1")


def recipe():
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    complete = {"ols": LinearRegression().get_params(), "mlp": MLPRegressor(hidden_layer_sizes=(64, 32),
        max_iter=400, early_stopping=True, n_iter_no_change=15, random_state=20260829).get_params()}
    complete["mlp"]["hidden_layer_sizes"] = list(complete["mlp"]["hidden_layer_sizes"])
    return {"schema_version": "size_causal_comparator.v1", "native": native_recipe().as_dict(),
        "component_parameters": complete, "features": list(FEATURES), "minimum_prior": 4, "minimum_training_rows": 500,
        "masks": {"or_implied": [0, 60], "or_rvol30": [0, 700], "training_target": [0, 200]},
        "prediction_projection": "none", "missing": "abstain", "quote_value": "stored_without_reconstruction", "refit": "fresh_per_outer_horizon",
        "history_mean": "stored_mean_else_qualified_training_mean", "production_reproduction": False}


def candidate_recipe(comparison):
    """Derive the fixed fallback candidate from an already-verified comparator."""
    return {**deepcopy(comparison), "schema_version": "size_history_mean_candidate.v1",
            "missing": "history_mean", "refit": "fresh_per_fold_horizon", "prediction_domain": "signed"}


def _values(row):
    if (row["current_quote_status"] not in ("present", "zero", "absent", "unavailable")
            or row["has_implied_quote"] != int(row["current_quote_status"] == "present")
            or type(row["n_prior"]) is not int or row["n_prior"] < 0):
        raise ValueError("invalid admitted quote/history metadata")
    values = {**row, "or_rvol30": row["rvol30"], "or_implied": row["implied_move"]}
    if row["n_prior"] < 4:
        return None, "insufficient_history"
    for name, hi in (("or_implied", 60), ("or_rvol30", 700)):
        value = values[name]
        if value is not None and not 0 <= value <= hi:
            return None, "predictor_mask:" + name
    if any(values[name] is None or not np.isfinite(values[name]) for name in FEATURES):
        return None, "unavailable_input"
    return [values[name] for name in FEATURES], None


def fit_evidence(inputs, fold, plan):
    """Bind eligible numeric arrays; markers do not prove estimator entry."""
    indexed = {(r["event_id"], r["horizon_sessions"]): r for r in inputs["features"]}
    result = []
    for horizon in plan["horizons"]:
        groups = [[(r["event_id"], _values(indexed[r["event_id"], horizon])[0], float(r["absolute_move_pct"]))
                   for r in inputs["training"] if 0 <= r["absolute_move_pct"] <= 200],
                  [(event, _values(indexed[event, horizon])[0], None) for event in fold["evaluation_event_ids"]]]
        groups = [[item for item in group if item[1] is not None] for group in groups]
        documents = [{"columns": FEATURES, "row_keys": [[e, indexed[e, horizon]["context_id"], horizon]
            for e, _, _ in group], "values": [[float(v) for v in values] for _, values, _ in group]} for group in groups]
        documents[0]["absolute_move_pct"] = [label for _, _, label in groups[0]]
        result.append({"horizon_sessions": horizon, "matrix_scope": "eligible_incumbent_arrays",
            "fit_eligible": len(groups[0]) >= 500, "training_event_ids": [e for e, _, _ in groups[0]],
            "evaluation_event_ids": [e for e, _, _ in groups[1]], "training_matrix_id": content_hash(documents[0]),
            "prediction_matrix_id": content_hash(documents[1])})
    return result


def _fit_rows(indexed, training, evaluation_ids, horizon):
    training = [(r["event_id"], _values(indexed[r["event_id"], horizon])[0], r["absolute_move_pct"])
                for r in training if 0 <= r["absolute_move_pct"] <= 200]
    return ([row for row in training if row[1] is not None],
            [(event, *_values(indexed[event, horizon])) for event in evaluation_ids])


def _details(training, evaluated):
    fitted = int(len(training) >= 500)
    reasons = [reason or "insufficient_training" for _, _, reason in evaluated if reason or not fitted]
    return {"training_event_ids": [e for e, _, _ in training], "logical_fits": fitted,
            "component_fits": 2 * fitted, "fallback_counts": dict(sorted(Counter(reasons).items()))}


def candidate_fit_details(inputs, fold, horizons):
    """Reconstruct candidate accounting from admitted features/training, never outcomes."""
    indexed = {(r["event_id"], r["horizon_sessions"]): r for r in inputs["features"]}
    return [{"horizon_sessions": h, **_details(*_fit_rows(indexed, inputs["training"],
                fold["evaluation_event_ids"], h))} for h in horizons]


def forecasts(indexed, train, fold, plan, execution, *, progress, inputs=None):
    """Causal candidate fallback; the standalone comparator still abstains."""
    candidate = execution.get("model") == "size_v1_4"
    if execution["fit_recipe"] != (candidate_recipe(recipe()) if candidate else recipe()):
        raise ValueError("comparator recipe differs")
    inputs = inputs or {"features": list(indexed.values()), "training": train.to_dict("records")}
    predictions, details = [], []
    fallback = (math.fsum(r["absolute_move_pct"] for r in inputs["training"]) / len(inputs["training"])
                if candidate else mean(r["absolute_move_pct"] for r in inputs["training"]))
    evidence = candidate_evidence(inputs, fold, plan, execution) if candidate else fit_evidence(inputs, fold, plan)
    for horizon, fit in zip(plan["horizons"], evidence):
        progress("started", fit)
        training, evaluated = _fit_rows(indexed, inputs["training"], fold["evaluation_event_ids"], horizon)
        model = None
        started = time.monotonic()
        try:
            if len(training) >= 500:
                with threadpool_limits(limits=1):
                    model = fit_recipe_estimator(native_recipe(), np.array([v for _, v, _ in training], dtype=float),
                                                np.array([y for _, _, y in training]))
            complete = [(event, values) for event, values, reason in evaluated if reason is None]
            estimates = {} if model is None or not complete else dict(zip([e for e, _ in complete],
                model.predict(np.array([v for _, v in complete], dtype=float))))
        except (ValueError, ArithmeticError):
            if not candidate:
                raise
            raise fail("EXPERIMENT_VARIANT_FAILED", "incumbent candidate estimator failed") from None
        elapsed = max(0.0, time.monotonic() - started)
        for event, _, reason in evaluated:
            reason = reason or ("insufficient_training" if model is None else None)
            row = indexed[event, horizon]
            history = history_mean_prediction(row, fallback) if candidate else (
                row["mean_prior_abs_move"] if row["n_prior"] else fallback)
            value = (history if candidate else None) if reason else float(estimates[event])
            if value is not None and not np.isfinite(value):
                if candidate:
                    raise fail("EXPERIMENT_VARIANT_FAILED", "incumbent candidate prediction is nonfinite")
                raise ValueError("nonfinite incumbent prediction")
            predictions.append({"event_id": event, "horizon_sessions": horizon,
                "predicted_absolute_move_pct": value, "abstention": None if candidate else reason,
                "history_mean": history, "quote_status": row["current_quote_status"]})
        detail = _details(training, evaluated)
        if not candidate:
            detail.pop("fallback_counts")
        details.append({"horizon_sessions": horizon, **detail})
        progress("completed", {**fit, "fit_seconds": elapsed} if candidate else fit)
    return {"event": "succeeded", "forecasts": sorted(predictions, key=lambda r: (r["event_id"], r["horizon_sessions"])),
            "fit_details": details}
