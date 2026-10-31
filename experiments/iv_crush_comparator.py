"""Causal forecast-cutoff IV incumbent refits on admitted arrays, not live panels."""
from copy import deepcopy
from dataclasses import replace
from statistics import mean

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from engine.v2.foundation import content_hash
from engine.v2.models.frozen_state_io import BoundedStateReadError
from engine.v2.models.training.estimators import fit_recipe_estimator
from engine.v2.models.training.fit_routes import check_fitting_allowed
from engine.v2.models.training.recipes import RecipeKey, current_recipes
from engine.v2.ops.errors import fail
from engine.v2.research.horizon_fit_inputs import prepare_horizon_inputs
from engine.v2.research.progress import Progress, decision, stage
from experiments.iv_crush_plan import target

FEATURES = ["n_prior", "mean_prior_move", "mean_prior_abs_move", "ema2_prior_move",
    "ema4_prior_move", "ema8_prior_move", "ema12_prior_move", "ema2_prior_abs_move",
    "ema4_prior_abs_move", "ema8_prior_abs_move", "ema12_prior_abs_move", "mean_prior_or_implied",
    "spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol20", "signed_streak", "ema12r_abs",
    "dist_high", "dist_ema", "ret5", "ret10", "ret20", "or_implied", "or_skewing", "or_contango",
    "or_rvol30", "or_exern30", "or_iv30", "or_iee", "or_fwd90_30", "or_fexern90_30", "mcap_log",
    "abs_dist_high", "abs_dist_ema", "has_implied_quote", "pre_iv30", "pre_iv10",
    "pre_exern_iv30", "pre_exern_iv10"]
_ALIASES = {"or_implied": "implied_move", "or_skewing": "skew", "or_contango": "contango",
    "or_rvol30": "rvol30", "or_exern30": "exern_iv30", "or_iv30": "iv30", "or_iee": "iee",
    "or_fwd90_30": "fwd90_30", "or_fexern90_30": "fexern90_30", "pre_iv30": "iv30",
    "pre_iv10": "iv10", "pre_exern_iv30": "exern_iv30", "pre_exern_iv10": "exern_iv10",
    "abs_dist_high": "dist_high"}
COLUMNS = list(dict.fromkeys(_ALIASES.get(name, name) for name in FEATURES))
_NATIVE_ID = "sha256:59c73e3ac0488252e502a40447e19938fac0d926eec2aac59a6db2eec2147d70"


def native_recipe():
    """Pin the complete native recipe; never import its historical panel/target loader."""
    try:
        native = current_recipes().get(RecipeKey("iv_crush", "*", "champion"))
    except BoundedStateReadError as error:
        raise fail("RESOURCE_LIMIT_EXCEEDED" if error.code == "SIZE_LIMIT" else "RESOURCE_UNAVAILABLE",
                   "incumbent registry failed bounded admission") from None
    if native is None or content_hash(native.as_dict()) != _NATIVE_ID or list(native.features) != FEATURES:
        raise ValueError("IV incumbent recipe changed; explicit reviewed migration required")
    return replace(native, dataset_source="qualified stored research panel; causal refit",
                   clock_id="earnings_forecast_cutoff.v1")


def _recipe(native):
    return {"schema_version": "iv_crush_causal_comparator.v1", "native": native.as_dict(), "target": target(),
        "component_parameters": HistGradientBoostingRegressor(**native.estimator.params,
            random_state=native.estimator.seeds[0]).get_params(),
        "features": list(FEATURES), "source_columns": list(COLUMNS),
        "feature_mapping": {name: "abs(dist_high)" if name == "abs_dist_high" else _ALIASES.get(name, name)
                            for name in FEATURES},
        "minimum_training_rows": 500, "missing": "abstain", "prediction_projection": "none",
        "refit": "fresh_per_outer_horizon", "training_mean": "all_admitted_training_labels",
        "production_reproduction": False}


def recipe():
    """Declare the signed causal refit, not reproduction of the legacy source population."""
    return _recipe(native_recipe())


def _training(rows):
    labels = [row["crush_pct_iv30"] for row in rows]
    if not labels or any(isinstance(y, bool) or not isinstance(y, (int, float)) or not np.isfinite(y) for y in labels):
        raise fail("INVALID_EXPERIMENT_SPEC", "IV comparator requires nonempty finite admitted training labels")
    return labels


def _horizon(indexed, training, evaluated, horizon):
    mapped = {}
    events = list(dict.fromkeys([row["event_id"] for row in training] + list(evaluated)))
    meter = Progress("IV incumbent mapping", len(events))
    for event in events:
        meter.tick()
        row = indexed[event, horizon]
        values = {name: row[_ALIASES.get(name, name)] for name in FEATURES}
        if any(value is None or not np.isfinite(value) for value in values.values()):
            continue
        values["abs_dist_high"] = abs(values["abs_dist_high"])
        mapped[event, horizon] = {**values, **{key: row[key] for key in ("event_id", "context_id", "horizon_sessions")}}
    eligible = [row for row in training if (row["event_id"], horizon) in mapped]
    tests = [event for event in evaluated if (event, horizon) in mapped]
    labels = np.array([row["crush_pct_iv30"] for row in eligible], dtype=float)
    matrix, prediction, evidence = prepare_horizon_inputs(mapped, [row["event_id"] for row in eligible],
        tests, horizon, FEATURES, labels, target_column="crush_pct_iv30")
    return matrix, labels, prediction, {**evidence, "matrix_scope": "eligible_incumbent_arrays",
                                      "fit_eligible": len(eligible) >= 500}


def fit_evidence(inputs, fold, plan):
    """Pure ordered eligible-array evidence; markers do not assert estimator entry."""
    _training(inputs["training"])
    indexed = {(row["event_id"], row["horizon_sessions"]): row for row in inputs["features"]}
    return [_horizon(indexed, inputs["training"], fold["evaluation_event_ids"], horizon)[3]
            for horizon in plan["horizons"]]


def forecasts(indexed, train, fold, plan, execution, *, progress):
    """Fit training-only labels; incomplete predictors abstain, signed values stay signed."""
    check_fitting_allowed("iv_crush_comparator.forecasts")
    native = native_recipe()
    if execution["fit_recipe"] != _recipe(native):
        raise ValueError("IV comparator recipe differs")
    training = train[["event_id", "crush_pct_iv30"]].to_dict("records")
    baseline = mean(_training(training))
    predictions, details = [], []
    for horizon in plan["horizons"]:
        matrix, labels, prediction, evidence = _horizon(indexed, training, fold["evaluation_event_ids"], horizon)
        decision("IV incumbent", f"horizon {horizon}: {len(labels)} complete training rows; fit={evidence['fit_eligible']}")
        progress("started", deepcopy(evidence))
        estimates = {}
        if evidence["fit_eligible"]:
            with stage(f"IV incumbent horizon {horizon}"), threadpool_limits(limits=1):
                model = fit_recipe_estimator(native, matrix, labels)
                if len(prediction):
                    values = model.predict(prediction)
                    if len(values) != len(prediction) or not np.isfinite(values).all():
                        raise ValueError("invalid IV incumbent predictions")
                    estimates = dict(zip(evidence["evaluation_event_ids"], map(float, values)))
        eligible_evaluation = set(evidence["evaluation_event_ids"])
        for event in fold["evaluation_event_ids"]:
            reason = ("unavailable_input" if event not in eligible_evaluation else
                      "insufficient_training" if not evidence["fit_eligible"] else None)
            predictions.append({"event_id": event, "horizon_sessions": horizon,
                "predicted_crush_pct_iv30": estimates.get(event), "abstention": reason, "training_mean": baseline})
        details.append({"horizon_sessions": horizon, "training_event_ids": evidence["training_event_ids"],
                        "logical_fits": int(evidence["fit_eligible"]), "component_fits": int(evidence["fit_eligible"])})
        progress("completed", deepcopy(evidence))
    return {"event": "succeeded", "forecasts": sorted(predictions, key=lambda row: (row["event_id"], row["horizon_sessions"])),
            "fit_details": details}
