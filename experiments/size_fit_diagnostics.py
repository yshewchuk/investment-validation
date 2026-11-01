"""Closed numeric-fit diagnostics retained in immutable v2 worker receipts."""
import math

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import fail
from experiments.size_comparator import _values


def keys(binding):
    return {"diagnostics"} if binding.get("schema_version") in ("native_experiment_registration.v1.6", "native_experiment_registration.v1.7", "native_experiment_registration.v1.8") else set()


def evidence(result, execution, horizons, *, inputs=None, fold=None):
    if execution.get("schema_version") not in ("size_candidate_execution.v1", "regression_candidate_execution.v1") or result.get("event") != "succeeded":
        return {}
    try:
        diagnostics, recipe = result["diagnostics"], execution["fit_recipe"]
        if len(diagnostics) != len(horizons):
            raise ValueError
        for item, horizon in zip(diagnostics, horizons):
            expected = {"schema_version", "recipe_id", "horizon_sessions"}
            elastic = recipe.get("model") == "elasticnet"
            size = recipe.get("schema_version") == "size_causal_comparator.v1"
            if elastic:
                expected |= {"coefficients", "coefficient_scale", "selected_columns", "intercept", "converged", "n_iter", "dual_gap"}
            elif size:
                expected |= {"training_event_ids", "logical_fits", "component_fits", "abstention_counts", "training_abstention_counts"}
            prefix = "regression" if execution["schema_version"] == "regression_candidate_execution.v1" else "magnitude"
            schema = "size_v1_4_fit.v1" if size else prefix + ("_elasticnet_fit.v1" if elastic else "_ridge_hgb_fit.v1")
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
            elif size:
                if inputs is None or fold is None:
                    raise ValueError
                indexed = {(r["event_id"], r["horizon_sessions"]): r for r in inputs["features"]}
                training_ids, training_abstention_counts = [], {}
                for row in inputs["training"]:
                    if 0 <= row["absolute_move_pct"] <= 200:
                        _, reason = _values(indexed[row["event_id"], horizon])
                        if reason is None:
                            training_ids.append(row["event_id"])
                        else:
                            training_abstention_counts[reason] = training_abstention_counts.get(reason, 0) + 1
                abstention_counts, eligible = {}, []
                for event in fold["evaluation_event_ids"]:
                    _, reason = _values(indexed[event, horizon])
                    if reason is None:
                        eligible.append(event)
                    else:
                        abstention_counts[reason] = abstention_counts.get(reason, 0) + 1
                logical_fits = int(len(training_ids) >= 500)
                if not logical_fits:
                    for _ in eligible:
                        abstention_counts["insufficient_training"] = abstention_counts.get("insufficient_training", 0) + 1
                expected_size = {"training_event_ids": training_ids, "logical_fits": logical_fits,
                                 "component_fits": 2 * logical_fits, "abstention_counts": abstention_counts,
                                 "training_abstention_counts": training_abstention_counts}
                if (type(item["logical_fits"]) is not int or type(item["component_fits"]) is not int
                        or not isinstance(item["abstention_counts"], dict)
                        or not isinstance(item["training_abstention_counts"], dict)
                        or any(type(v) is not int for v in item["abstention_counts"].values())
                        or any(type(v) is not int for v in item["training_abstention_counts"].values())):
                    raise ValueError
                if any(item[k] != expected_size[k] for k in expected_size):
                    raise ValueError
        return {"diagnostics": diagnostics}
    except (KeyError, TypeError, ValueError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "v2 fit diagnostics differ from bound recipe") from None
