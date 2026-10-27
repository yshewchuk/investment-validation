"""Closed numeric-fit diagnostics retained in immutable v2 worker receipts."""
import math

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail


def keys(binding):
    return {"diagnostics"} if binding.get("schema_version") == "native_experiment_registration.v1.6" else set()


def evidence(result, execution, horizons):
    if execution.get("schema_version") != "size_candidate_execution.v1" or result.get("event") != "succeeded":
        return {}
    try:
        diagnostics, recipe = result["diagnostics"], execution["fit_recipe"]
        if len(diagnostics) != len(horizons):
            raise ValueError
        for item, horizon in zip(diagnostics, horizons):
            expected = {"schema_version", "recipe_id", "horizon_sessions"}
            elastic = recipe["model"] == "elasticnet"
            if elastic:
                expected |= {"coefficients", "coefficient_scale", "selected_columns", "intercept", "converged", "n_iter", "dual_gap"}
            schema = "magnitude_elasticnet_fit.v1" if elastic else "magnitude_ridge_hgb_fit.v1"
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
