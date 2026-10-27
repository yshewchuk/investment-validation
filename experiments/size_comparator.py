"""Causal fourteen-input incumbent refits on admitted arrays, never production rows."""
from dataclasses import replace
from statistics import mean

import numpy as np
from threadpoolctl import threadpool_limits

from engine.v2.foundation import content_hash
from engine.v2.models.frozen_state_io import BoundedStateReadError
from engine.v2.models.training.estimators import fit_recipe_estimator
from engine.v2.models.training.recipes import RecipeKey, current_recipes
from engine.v2.ops.errors import fail

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


def forecasts(indexed, train, fold, plan, execution, *, progress):
    """Preserve abstention and negative blends; never apply target masks to evaluation."""
    if execution["fit_recipe"] != recipe():
        raise ValueError("comparator recipe differs")
    predictions, details = [], []
    train_ids, test_ids = train.event_id.tolist(), fold["evaluation_event_ids"]
    labels = train.absolute_move_pct.to_numpy(dtype=float)
    fallback = mean(labels.tolist())
    for horizon in plan["horizons"]:
        documents = [{"columns": execution["features"], "row_keys": [[event, indexed[event, horizon]["context_id"], horizon]
            for event in ids], "values": [[indexed[event, horizon][name] for name in execution["features"]]
            for event in ids]} for ids in (train_ids, test_ids)]
        documents[0]["absolute_move_pct"] = labels.tolist()
        fit = {"horizon_sessions": horizon, "training_event_ids": train_ids,
            "evaluation_event_ids": test_ids, "training_matrix_id": content_hash(documents[0]),
            "prediction_matrix_id": content_hash(documents[1])}
        progress("started", fit)
        training = [(event, _values(indexed[event, horizon])[0], label) for event, label in zip(train_ids, labels)]
        training = [(event, values, label) for event, values, label in training if values is not None and 0 <= label <= 200]
        evaluated = [(event, *_values(indexed[event, horizon])) for event in test_ids]
        model = None
        if len(training) >= 500:
            with threadpool_limits(limits=1):
                model = fit_recipe_estimator(native_recipe(), np.array([v for _, v, _ in training]),
                                            np.array([y for _, _, y in training]))
        complete = [(event, values) for event, values, reason in evaluated if reason is None]
        estimates = {} if model is None or not complete else dict(zip([e for e, _ in complete],
            model.predict(np.array([v for _, v in complete]))))
        for event, values, reason in evaluated:
            reason = reason or ("insufficient_training" if model is None else None)
            value = None if reason else float(estimates[event])
            if value is not None and not np.isfinite(value):
                raise ValueError("nonfinite incumbent prediction")
            row = indexed[event, horizon]
            predictions.append({"event_id": event, "horizon_sessions": horizon,
                "predicted_absolute_move_pct": value, "abstention": reason,
                "history_mean": row["mean_prior_abs_move"] if row["n_prior"] else fallback,
                "quote_status": row["current_quote_status"]})
        details.append({"horizon_sessions": horizon, "training_event_ids": [e for e, _, _ in training],
                        "logical_fits": int(model is not None), "component_fits": 2 * int(model is not None)})
        progress("completed", fit)
    return {"event": "succeeded", "forecasts": sorted(predictions, key=lambda r: (r["event_id"], r["horizon_sessions"])),
            "fit_details": details}
