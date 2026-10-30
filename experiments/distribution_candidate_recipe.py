"""Source-free distribution declarations, not native admission or runnable plans.

This initial raw-probability candidate is not preregistration. Source, fold,
feature and resource references remain caller declarations until the shared
platform admits them. No registration transport, rows or holdout logic live here.
"""
import json
import re
import sys

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.models.magnitude_ridge_hgb_recipe import ridge_hgb_recipe
from engine.v2.ops.errors import OpsError, fail
from engine.v2.research.magnitude_distribution import recipe as output_recipe
from engine.v2.research.magnitude_panel_qualified_table import recipe as feature_recipe
from experiments.magnitude_search_execution import validate_execution_limits

__all__ = ["candidate_recipe", "validate_candidate_recipe", "fit_budget", "trial_identity"]

_HORIZONS = [5, 3, 1]


def _require(condition):
    if not condition:
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution declaration")


def _identified(document, key):
    return {**document, key: content_hash(document)}


def _hash(value):
    return type(value) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _count(value):
    return type(value) is int and 0 < value < sys.maxsize


def candidate_recipe(feature_set):
    """Declare one explicit H/HM/HMO pair; no feature or parameter search.

    Reuse native HGB knobs with seven leaves, changing only the classifier's
    estimator/loss and removing the regressor-only quantile parameter. Both
    arms use identical ordered v3 features and preprocessing. Missing classes
    get zero mass; no smoothing, class weighting or calibration is claimed.
    """
    _require(type(feature_set) is str and feature_set in ("H", "HM", "HMO"))
    features = feature_recipe(feature_set)
    point = ridge_hgb_recipe(model="history_hgb", columns=features["model_columns"],
        feature_dictionary_hash=features["dictionary_hash"], max_leaf_nodes=7)
    classifier = {key: value for key, value in point["estimator"]["params"].items()
                  if key != "quantile"}
    classifier.update(loss="log_loss", class_weight=None)
    document = {"schema_version": "distribution_candidate_recipe.v1", "feature_set": feature_set,
        "feature_schema": features["schema_version"], "point_recipe": point,
        "classifier": {"class": "HistGradientBoostingClassifier", "params": classifier,
                       "preprocessing": dict(point["preprocessing"])},
        "output_recipe": output_recipe(), "horizons": list(_HORIZONS), "primary_horizon": 1,
        "class_policy": {"axis": list(range(81)), "mapping": "explicit_estimator_class_labels",
            "minimum_observed_classes": 2, "absent_class_mass": 0, "smoothing": "none"},
        "calibration": {"identity": "uncalibrated", "fit_count": 0},
        "tail": {"policy": "training_only_empirical_excess", "no_samples": "unresolved"},
        "comparison": "raw_histogram_and_same_feature_point", "retries": 0}
    return _identified(document, "recipe_id")


def validate_candidate_recipe(recipe):
    """Return detached installed semantics, refusing altered/unknown fields.

    JSON-equivalent containers/numbers are accepted as in native model recipes.
    Identity equality proves metadata consistency, not source eligibility.
    """
    try:
        json.dumps(recipe, allow_nan=False)
        expected = candidate_recipe(recipe["feature_set"])
        if canonical_json(recipe) == canonical_json(expected):
            return expected
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError, OpsError):
        pass
    raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution candidate recipe")


def fit_budget(recipe, *, inner_folds, outer_folds, max_fits, max_trees, child_limits):
    """Count declared raw-pair work, not a complete experiment/resource admission.

    Each declared fold has all three horizons and two estimator fits. The
    classifier ceiling uses 81 trees per iteration; actual binary fits use one.
    Counts include the pair's declared inner OOF fits, not separate calibration,
    incumbent or payoff refits. No pooling, preprocessing, memory feasibility,
    parent/job containment, source feasibility or automatic retry is implied.
    """
    recipe = validate_candidate_recipe(recipe)
    _require(all(_count(value) for value in (inner_folds, outer_folds, max_fits, max_trees)))
    _require(type(child_limits) is dict)
    validate_execution_limits(child_limits)
    fits_per_arm = len(_HORIZONS) * (inner_folds + outer_folds)
    iterations = recipe["classifier"]["params"]["max_iter"]
    classes = len(recipe["class_policy"]["axis"])
    classifier_trees = fits_per_arm * iterations * classes
    point_trees = fits_per_arm * recipe["point_recipe"]["estimator"]["params"]["max_iter"]
    counts = {"classifier_fits": fits_per_arm, "point_fits": fits_per_arm,
        "total_fits": 2 * fits_per_arm, "classifier_trees_upper": classifier_trees,
        "point_trees_upper": point_trees, "total_trees_upper": classifier_trees + point_trees}
    if counts["total_fits"] > max_fits or counts["total_trees_upper"] > max_trees:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "distribution raw-pair fit or tree cap is insufficient")
    return _identified({"schema_version": "distribution_fit_budget.v1",
        "scope": "raw_histogram_and_matched_point_only", "recipe_id": recipe["recipe_id"],
        "inner_folds": inner_folds, "outer_folds": outer_folds, "horizons": list(_HORIZONS),
        "counts": counts, "max_fits": max_fits, "max_trees": max_trees,
        "child_limits": dict(child_limits), "retries": 0,
        "excluded": ["additional_calibration_fits", "incumbent_refits", "payoff_refits",
                     "feature_preparation", "parent_and_job_containment"]}, "budget_id")


def trial_identity(recipe, *, context_id, fold_id, inputs_id, budget_id, horizon, role):
    """Bind opaque references for one pair/horizon; authenticate none of them.

    The later shared platform must admit the actual context, grouped fold,
    cohort and resource plan before fitting. This declaration is not a claim,
    sealed receipt, registration, journal entry, retry grant or source proof.
    """
    recipe = validate_candidate_recipe(recipe)
    _require(all(_hash(value) for value in (context_id, fold_id, inputs_id, budget_id)))
    _require(type(horizon) is int and horizon in _HORIZONS
             and type(role) is str and role in ("inner", "outer"))
    return _identified({"schema_version": "distribution_trial_declaration.v1",
        "recipe_id": recipe["recipe_id"], "context_id": context_id, "fold_id": fold_id,
        "inputs_id": inputs_id, "budget_id": budget_id, "horizon": horizon, "role": role,
        "calibration_identity": "uncalibrated", "retries": 0}, "trial_id")
