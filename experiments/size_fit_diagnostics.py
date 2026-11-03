"""Closed numeric-fit diagnostics retained in immutable v2 worker receipts."""
import math
from datetime import date

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.models.magnitude_mlp_recipe import validate_mlp_recipe
from engine.v2.ops.errors import fail


def keys(binding):
    return {"diagnostics"} if binding.get("schema_version") in ("native_experiment_registration.v1.6", "native_experiment_registration.v1.7", "native_experiment_registration.v1.8") else set()


def _training_event_date_ids(inputs, horizons):
    if (type(inputs) is not dict or type(inputs.get("training")) is not list
            or type(inputs.get("features")) is not list):
        raise ValueError
    training_ids = []
    for item in inputs["training"]:
        if type(item) is not dict or type(item.get("event_id")) is not str:
            raise ValueError
        training_ids.append(item["event_id"])
    training_set, requested = set(training_ids), set(horizons)
    indexed = {}
    for row in inputs["features"]:
        if type(row) is not dict:
            raise ValueError
        horizon = row.get("horizon_sessions")
        if horizon in requested:
            value = row.get("event_date")
            try:
                if type(value) is not str or date.fromisoformat(value).isoformat() != value:
                    raise ValueError
            except ValueError:
                raise ValueError from None
        key = (row.get("event_id"), horizon)
        if key[0] in training_set and key[1] in requested:
            if key in indexed:
                raise ValueError
            indexed[key] = row
    result = {}
    for horizon in horizons:
        try:
            dates = [indexed[(event_id, horizon)]["event_date"] for event_id in training_ids]
        except KeyError:
            raise ValueError from None
        result[horizon] = content_hash(dates)
    return result


def _mlp_diagnostics(diagnostics, recipe, horizons, training_event_dates_ids=None):
    """Validate worker claims, not convergence by refitting or reading source data."""
    recipe = validate_mlp_recipe(recipe)
    recipe_id = content_hash(recipe)
    max_iter = recipe["estimator"]["params"]["max_iter"]
    fixed_budget = recipe["nonconvergence"] == "epoch_budget_success_no_retry"
    expected = {"schema_version", "recipe_id", "horizon_sessions", "component_fits",
                "converged", "n_iter", "input_features", "loss", "loss_scale"}
    requires_event_dates = recipe.get("requires_event_dates") is True
    if requires_event_dates:
        if type(training_event_dates_ids) is not dict:
            raise ValueError
        expected.add("training_event_dates_id")
    if type(diagnostics) is not list or len(diagnostics) != len(horizons):
        raise ValueError
    for item, horizon in zip(diagnostics, horizons):
        if (type(item) is not dict or set(item) != expected
                or item["schema_version"] != "magnitude_mlp_fit.v1"
                or item["recipe_id"] != recipe_id
                or type(item["horizon_sessions"]) is not int or item["horizon_sessions"] != horizon
                or type(item["component_fits"]) is not int or item["component_fits"] != 1
                or type(item["converged"]) is not bool
                or (item["converged"] is False
                    and (not fixed_budget or item["n_iter"] != max_iter))
                or item["loss_scale"] != "standardized_target"
                or type(item["n_iter"]) is not int
                or not 1 <= item["n_iter"] <= max_iter
                or type(item["input_features"]) is not int
                or not 1 <= item["input_features"] <= recipe["preprocessing"]["union_features"]
                or type(item["loss"]) not in (int, float) or item["loss"] < 0
                or (requires_event_dates and (
                    type(item.get("training_event_dates_id")) is not str
                    or len(item["training_event_dates_id"]) != 71
                    or not item["training_event_dates_id"].startswith("sha256:")
                    or any(c not in "0123456789abcdef" for c in item["training_event_dates_id"][7:])
                    or item["training_event_dates_id"] != training_event_dates_ids.get(horizon)))):
            raise ValueError
        try:
            if not math.isfinite(item["loss"]):
                raise ValueError
        except OverflowError:
            raise ValueError from None
    return {"diagnostics": diagnostics}


def evidence(result, execution, horizons, inputs=None, fold=None):
    if execution.get("schema_version") not in ("size_candidate_execution.v1", "regression_candidate_execution.v1") or result.get("event") != "succeeded":
        return {}
    try:
        diagnostics, recipe = result["diagnostics"], execution["fit_recipe"]
        if recipe.get("model") == "mlp" or recipe.get("schema_version") == "magnitude_mlp_recipe.v1":
            if execution["schema_version"] != "size_candidate_execution.v1":
                raise ValueError
            training_event_dates_ids = (_training_event_date_ids(inputs, horizons)
                if recipe.get("requires_event_dates") is True else None)
            return _mlp_diagnostics(diagnostics, recipe, horizons, training_event_dates_ids)
        if recipe.get("schema_version") == "size_history_mean_candidate.v1":
            from experiments.size_comparator import candidate_fit_details
            expected = [{"schema_version": "size_v1_4_fit.v1", "recipe_id": content_hash(recipe), **item}
                        for item in candidate_fit_details(inputs, fold, horizons)]
            if canonical_json(diagnostics) != canonical_json(expected):
                raise ValueError
            return {"diagnostics": diagnostics}
        if len(diagnostics) != len(horizons):
            raise ValueError
        for item, horizon in zip(diagnostics, horizons):
            expected = {"schema_version", "recipe_id", "horizon_sessions"}
            elastic = recipe["model"] == "elasticnet"
            if elastic:
                expected |= {"coefficients", "coefficient_scale", "selected_columns", "intercept", "converged", "n_iter", "dual_gap"}
            prefix = "regression" if execution["schema_version"] == "regression_candidate_execution.v1" else "magnitude"
            schema = prefix + ("_elasticnet_fit.v1" if elastic else "_ridge_hgb_fit.v1")
            if (set(item) != expected or item["horizon_sessions"] != horizon or item["schema_version"] != schema
                    or item["recipe_id"] != content_hash(recipe)):
                raise ValueError
            if elastic:
                entries = item["coefficients"]
                layout = [(kind, column) for kind in ("value", "missing_indicator") for column in recipe["features"]]
                if ([(e["kind"], e["column"]) for e in entries] != layout or item["converged"] is not True
                        or item["coefficient_scale"] != "standardized_inputs" or type(item["n_iter"]) is not int
                        or not 0 <= item["n_iter"] <= 10000):
                    raise ValueError
                numbers = [item["intercept"], item["dual_gap"], *[e["coefficient"] for e in entries]]
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers):
                    raise ValueError
                selected = {e["column"] for e in entries if e["coefficient"] != 0}
                if item["selected_columns"] != [c for c in recipe["features"] if c in selected]:
                    raise ValueError
        return {"diagnostics": diagnostics}
    except (KeyError, TypeError, ValueError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "v2 fit diagnostics differ from bound recipe") from None
