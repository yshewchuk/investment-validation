"""Independent raw-pair declarations; no source, fitting or registration work."""
import copy
import math
import socket
import subprocess
import sys

import pytest

from engine.v2.foundation import content_hash
from engine.v2.models.magnitude_ridge_hgb_recipe import ridge_hgb_recipe
from engine.v2.ops.errors import OpsError
from engine.v2.research.magnitude_distribution import recipe as output_recipe
from engine.v2.research.magnitude_panel_qualified_table import recipe as feature_recipe
from experiments import distribution_candidate_recipe as declaration

LIMITS = {"memory_bytes": 1024, "cpu_seconds": 3, "wall_seconds": 5}
REFS = {name: "sha256:" + digit * 64 for name, digit in
        (("context_id", "1"), ("fold_id", "2"), ("inputs_id", "3"), ("budget_id", "4"))}


def _refuses(function, *args, code="INVALID_EXPERIMENT_SPEC", message="invalid distribution declaration", **kwargs):
    with pytest.raises(OpsError) as caught:
        function(*args, **kwargs)
    assert caught.value.code == code
    assert caught.value.problem.message == message
    assert caught.value.problem.retryable is False


def _budget(**changes):
    kwargs = {"inner_folds": 6, "outer_folds": 3, "max_fits": 54,
              "max_trees": 221400, "child_limits": LIMITS}
    return declaration.fit_budget(declaration.candidate_recipe("H"), **{**kwargs, **changes})


@pytest.mark.parametrize("subset,width", [("H", 11), ("HM", 28), ("HMO", 43)])
def test_pair_reuses_exact_native_feature_order_and_point_policy(subset, width):
    result = declaration.candidate_recipe(subset)
    features = feature_recipe(subset)
    point = ridge_hgb_recipe(model="history_hgb", columns=features["model_columns"],
        feature_dictionary_hash=features["dictionary_hash"], max_leaf_nodes=7)
    assert result["point_recipe"] == point
    assert len(point["features"]) == width
    assert result["feature_schema"] == "research_magnitude_panel.v3"
    assert result["feature_set"] == subset
    assert result["classifier"]["preprocessing"] == point["preprocessing"]
    shared = dict(point["estimator"]["params"])
    del shared["quantile"]
    shared.update(loss="log_loss", class_weight=None)
    assert result["classifier"] == {"class": "HistGradientBoostingClassifier",
                                    "params": shared, "preprocessing": point["preprocessing"]}
    assert shared["early_stopping"] is False
    assert shared["max_iter"] == 100
    assert shared["max_leaf_nodes"] == 7
    assert result["recipe_id"] == content_hash({k: v for k, v in result.items() if k != "recipe_id"})


def test_initial_raw_policy_and_detached_results_are_explicit():
    result = declaration.candidate_recipe("H")
    assert result["output_recipe"] == output_recipe()
    assert result["horizons"] == [5, 3, 1] and result["primary_horizon"] == 1
    assert result["class_policy"] == {"axis": list(range(81)), "mapping": "explicit_estimator_class_labels",
        "minimum_observed_classes": 2, "absent_class_mass": 0, "smoothing": "none"}
    assert result["calibration"] == {"identity": "uncalibrated", "fit_count": 0}
    assert result["tail"] == {"policy": "training_only_empirical_excess", "no_samples": "unresolved"}
    assert result["retries"] == 0
    before = copy.deepcopy(result)
    detached = declaration.validate_candidate_recipe(result)
    result["point_recipe"]["features"].clear()
    result["classifier"]["params"]["max_iter"] = 1
    result["class_policy"]["axis"].clear()
    assert detached == before == declaration.candidate_recipe("H")
    assert len({declaration.candidate_recipe(s)["recipe_id"] for s in ("H", "HM", "HMO")}) == 3


@pytest.mark.parametrize("bad", [None, [], {}, 1, True, "h", "H/HM/HMO", "HMO "])
def test_invalid_subset_refuses(bad):
    _refuses(declaration.candidate_recipe, bad)


@pytest.mark.parametrize("path,value", [
    (("schema_version",), "other"), (("extra",), 1), (("recipe_id",), "sha256:" + "a" * 64),
    (("point_recipe", "features"), ["future_label"]),
    (("point_recipe", "feature_dictionary_hash"), "sha256:" + "b" * 64),
    (("classifier", "params", "early_stopping"), True),
    (("classifier", "params", "max_iter"), 99), (("classifier", "params", "quantile"), None),
    (("classifier", "params", "class_weight"), "balanced"),
    (("class_policy", "absent_class_mass"), 0.001), (("calibration", "identity"), "isotonic"),
    (("tail", "policy"), "outer_test_empirical"), (("horizons",), [1]),
    (("output_recipe", "finite_bin_edges"), [0, 20]), (("retries",), 1),
])
def test_changed_semantics_refuse_even_if_caller_rehashes(path, value):
    recipe = declaration.candidate_recipe("H")
    node = recipe
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    if path != ("recipe_id",):
        recipe["recipe_id"] = content_hash({k: v for k, v in recipe.items() if k != "recipe_id"})
    _refuses(declaration.validate_candidate_recipe, recipe, message="invalid distribution candidate recipe")


@pytest.mark.parametrize("bad", [None, [], {}, "recipe", {"feature_set": "H"},
    {"feature_set": object()}, {"feature_set": "H", "other": math.nan}])
def test_malformed_recipe_has_safe_typed_refusal(bad):
    _refuses(declaration.validate_candidate_recipe, bad, message="invalid distribution candidate recipe")


def test_exact_boundary_counts_include_each_fold_horizon_and_both_arms():
    result = _budget()
    # Nine blocks x three horizons = 27 fits/arm; classifier ceiling81*100 each.
    assert result["counts"] == {"classifier_fits": 27, "point_fits": 27, "total_fits": 54,
        "classifier_trees_upper": 218700, "point_trees_upper": 2700, "total_trees_upper": 221400}
    assert result["scope"] == "raw_histogram_and_matched_point_only"
    assert result["excluded"] == ["additional_calibration_fits", "incumbent_refits", "payoff_refits",
                                  "feature_preparation", "parent_and_job_containment"]
    assert _budget(inner_folds=1, outer_folds=1, max_fits=12, max_trees=49200)["counts"]["total_trees_upper"] == 49200
    assert result["budget_id"] == content_hash({k: v for k, v in result.items() if k != "budget_id"})
    result["child_limits"].clear()
    assert LIMITS == {"memory_bytes": 1024, "cpu_seconds": 3, "wall_seconds": 5}


@pytest.mark.parametrize("change", [{"max_fits": 53}, {"max_trees": 221399},
                                    {"inner_folds": 7}, {"outer_folds": 4}])
def test_insufficient_raw_pair_caps_refuse_before_any_execution(change):
    _refuses(_budget, **change, code="RESOURCE_LIMIT_EXCEEDED",
             message="distribution raw-pair fit or tree cap is insufficient")


@pytest.mark.parametrize("field", ["inner_folds", "outer_folds", "max_fits", "max_trees"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.0, "3", None, sys.maxsize])
def test_counts_require_positive_platform_integers(field, bad):
    _refuses(_budget, **{field: bad})


def test_larger_declared_counts_cannot_overflow_into_an_admitted_budget():
    _refuses(_budget, inner_folds=sys.maxsize - 1, max_fits=sys.maxsize - 1,
             max_trees=sys.maxsize - 1, code="RESOURCE_LIMIT_EXCEEDED",
             message="distribution raw-pair fit or tree cap is insufficient")


@pytest.mark.parametrize("bad", [None, [], {"memory_bytes": 1},
    {**LIMITS, "cpu_id": 0}, {**LIMITS, "wall_seconds": True}, {**LIMITS, "memory_bytes": 0}])
def test_child_limit_owner_refusal_is_preserved(bad):
    message = ("invalid distribution declaration" if type(bad) is not dict else
               "search execution requires explicit positive integer resource limits")
    _refuses(_budget, child_limits=bad, message=message)


def test_each_resource_change_changes_identity_without_allocating_resources():
    baseline = _budget()["budget_id"]
    for changes in ({"max_fits": 55}, {"max_trees": 221401}, {"inner_folds": 5},
                    {"outer_folds": 2}, {"child_limits": {**LIMITS, "cpu_seconds": 4}}):
        assert _budget(**changes)["budget_id"] != baseline


@pytest.mark.parametrize("role", ["inner", "outer"])
@pytest.mark.parametrize("horizon", [5, 3, 1])
def test_trial_binds_every_reference_role_and_horizon_but_not_source_authority(role, horizon):
    recipe = declaration.candidate_recipe("H")
    result = declaration.trial_identity(recipe, **REFS, role=role, horizon=horizon)
    assert result == {"schema_version": "distribution_trial_declaration.v1", **REFS,
        "recipe_id": recipe["recipe_id"], "role": role, "horizon": horizon,
        "calibration_identity": "uncalibrated", "retries": 0,
        "trial_id": content_hash({k: v for k, v in result.items() if k != "trial_id"})}
    for key in REFS:
        changed = declaration.trial_identity(recipe, **{**REFS, key: "sha256:" + "9" * 64},
                                            role=role, horizon=horizon)
        assert changed["trial_id"] != result["trial_id"]
    assert declaration.trial_identity(declaration.candidate_recipe("HM"), **REFS,
                                     role=role, horizon=horizon)["trial_id"] != result["trial_id"]


@pytest.mark.parametrize("field", list(REFS))
@pytest.mark.parametrize("bad", [None, 1, "a" * 64, "sha256:" + "A" * 64, "sha256:" + "a" * 63])
def test_malformed_opaque_references_refuse(field, bad):
    _refuses(declaration.trial_identity, declaration.candidate_recipe("H"),
             **{**REFS, field: bad}, role="inner", horizon=1)


def test_all_role_horizon_declarations_have_distinct_identity():
    recipe = declaration.candidate_recipe("H")
    identities = {declaration.trial_identity(recipe, **REFS, role=role, horizon=horizon)["trial_id"]
                  for role in ("inner", "outer") for horizon in (5, 3, 1)}
    assert len(identities) == 6


@pytest.mark.parametrize("role,horizon", [("fit", 1), (None, 1), ("inner", True),
                                        ("outer", 1.0), ("inner", 2)])
def test_invalid_trial_role_or_horizon_refuses(role, horizon):
    _refuses(declaration.trial_identity, declaration.candidate_recipe("H"),
             **REFS, role=role, horizon=horizon)


def test_public_inventory_and_no_io(monkeypatch):
    assert declaration.__all__ == ["candidate_recipe", "validate_candidate_recipe", "fit_budget", "trial_identity"]
    def forbidden(*args, **kwargs):
        pytest.fail("declaration attempted source/execution I/O")
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    recipe = declaration.candidate_recipe("H")
    assert declaration.validate_candidate_recipe(recipe) == recipe
    assert _budget()["counts"]["total_fits"] == 54
    assert declaration.trial_identity(recipe, **REFS, horizon=1, role="inner")["retries"] == 0


@pytest.mark.parametrize("function", [declaration.candidate_recipe, declaration.validate_candidate_recipe,
                                     declaration.fit_budget, declaration.trial_identity])
def test_omitted_required_arguments_preserve_python_type_error(function):
    with pytest.raises(TypeError):
        function()


def test_parameters_match_installed_sklearn_without_fitting():
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    result = declaration.candidate_recipe("H")
    for cls, params in ((HistGradientBoostingClassifier, result["classifier"]["params"]),
                        (HistGradientBoostingRegressor, result["point_recipe"]["estimator"]["params"])):
        assert cls(**params).get_params() == params
