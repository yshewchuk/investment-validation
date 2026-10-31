"""Independent closed-policy and source-free metadata checks for stage-2 fits."""
import copy
import json
import subprocess
import sys

import pytest

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.models.magnitude_ridge_hgb_recipe import ridge_hgb_recipe, validate_ridge_hgb_recipe

HASH = "sha256:" + "a" * 64
RIDGE = {"alpha": 1.0, "fit_intercept": True, "copy_X": True, "max_iter": None,
         "tol": 0.0001, "solver": "svd", "positive": False, "random_state": 0}
HGB = {"loss": "squared_error", "quantile": None, "learning_rate": 0.1,
       "max_iter": 100, "max_leaf_nodes": 7, "max_depth": None,
       "min_samples_leaf": 20, "l2_regularization": 0.0, "max_features": 1.0,
       "max_bins": 255, "categorical_features": None, "monotonic_cst": None,
       "interaction_cst": None, "warm_start": False, "early_stopping": False,
       "scoring": "loss", "validation_fraction": 0.1, "n_iter_no_change": 10,
       "tol": 1e-7, "verbose": 0, "random_state": 0}
HGB_V2_KNOBS = {"learning_rate": 0.05, "max_leaf_nodes": 7,
                "min_samples_leaf": 20, "l2_regularization": 0.0}
HGB_V2 = {**HGB, **HGB_V2_KNOBS, "max_iter": 500, "early_stopping": True}
CASES = [("history_ridge", {"alpha": value}) for value in (0.1, 1, 10)] + [
    ("history_hgb", {"max_leaf_nodes": value}) for value in (7, 15)]


@pytest.mark.parametrize("model,knobs", CASES)
def test_exact_closed_policy_and_detached_canonical_roundtrip(model, knobs):
    columns = ["one", "two", "three"]
    ridge = model == "history_ridge"
    expected = {
        "schema_version": "magnitude_ridge_hgb_recipe.v1", "model": model,
        "features": columns.copy(), "feature_dictionary_hash": HASH,
        "estimator": {"class": "Ridge" if ridge else "HistGradientBoostingRegressor",
                      "params": {**(RIDGE if ridge else HGB), **knobs}},
        "preprocessing": {"imputer": {"strategy": "median", "keep_empty_features": True},
            "missing_indicator": {"features": "all", "sparse": False},
            "scaler": {"with_mean": True, "with_std": True}, "output_features": 6}
            if ridge else {"missing_values": "native", "scaling": "none",
                           "all_empty_training_columns": "constant_zero_train_and_predict"},
        "target_transform": "none", "prediction_projection": "clip_zero",
        "fit_scope": "fold_and_horizon", "numerical_threads": 1,
    }
    actual = ridge_hgb_recipe(model=model, columns=columns, feature_dictionary_hash=HASH, **knobs)
    assert actual == expected
    assert validate_ridge_hgb_recipe(json.loads(canonical_json(actual))) == expected
    detached = validate_ridge_hgb_recipe(actual)
    columns.clear()
    actual["features"].clear()
    actual["estimator"]["params"].clear()
    actual["preprocessing"].clear()
    assert detached == expected


@pytest.mark.parametrize("alpha", [1e-6, 0.001, 0.003, 0.03, 100.0, 1000.0])
def test_v3_ridge_alpha_interval_is_accepted_and_validated(alpha):
    policy = ridge_hgb_recipe(model="history_ridge", columns=["a", "b"],
                              feature_dictionary_hash=HASH, alpha=alpha)
    assert policy["estimator"]["params"]["alpha"] == float(alpha)
    assert validate_ridge_hgb_recipe(json.loads(canonical_json(policy))) == policy


def test_v1_hgb_recipe_keeps_fixed_iteration_policy():
    recipe = ridge_hgb_recipe(model="history_hgb", columns=["a"], feature_dictionary_hash=HASH,
                              max_leaf_nodes=7)
    params = recipe["estimator"]["params"]
    assert recipe["schema_version"] == "magnitude_ridge_hgb_recipe.v1"
    assert params["early_stopping"] is False
    assert params["max_iter"] == 100
    assert params["learning_rate"] == 0.1
    assert params["min_samples_leaf"] == 20
    assert params["l2_regularization"] == 0.0


def test_v2_hgb_recipe_declares_early_stopping_and_all_axes():
    recipe = ridge_hgb_recipe(model="history_hgb", columns=["a", "b"],
                              feature_dictionary_hash=HASH, **HGB_V2_KNOBS)
    params = recipe["estimator"]["params"]
    assert recipe["schema_version"] == "magnitude_ridge_hgb_recipe.v2"
    assert params["early_stopping"] is True
    assert params["max_iter"] == 500
    assert params["n_iter_no_change"] == 10
    assert params["validation_fraction"] == 0.1
    assert params["scoring"] == "loss"
    assert params["random_state"] == 0
    assert params["tol"] == 1e-7
    assert params["learning_rate"] == 0.05 and type(params["learning_rate"]) is float
    assert params["min_samples_leaf"] == 20 and type(params["min_samples_leaf"]) is int
    assert params["l2_regularization"] == 0.0 and type(params["l2_regularization"]) is float
    assert recipe["numerical_threads"] == 1
    assert params == HGB_V2
    assert validate_ridge_hgb_recipe(json.loads(canonical_json(recipe))) == recipe


def test_v1_and_v2_hgb_recipes_for_same_leaf_count_have_different_identity():
    v1 = ridge_hgb_recipe(model="history_hgb", columns=["a"], feature_dictionary_hash=HASH,
                          max_leaf_nodes=7)
    v2 = ridge_hgb_recipe(model="history_hgb", columns=["a"], feature_dictionary_hash=HASH,
                          **HGB_V2_KNOBS)
    assert v1["estimator"]["params"]["max_leaf_nodes"] == v2["estimator"]["params"]["max_leaf_nodes"]
    assert content_hash(v1) != content_hash(v2)


def test_v2_validation_refuses_altered_fixed_early_stopping_policy():
    original = ridge_hgb_recipe(model="history_hgb", columns=["a"], feature_dictionary_hash=HASH,
                                **HGB_V2_KNOBS)
    for key, value in (("early_stopping", False), ("max_iter", 99), ("n_iter_no_change", 1),
                       ("validation_fraction", 0.2), ("learning_rate", 2.0)):
        altered = copy.deepcopy(original)
        altered["estimator"]["params"][key] = value
        with pytest.raises(ValueError, match="closed"):
            validate_ridge_hgb_recipe(altered)


@pytest.mark.parametrize("knobs", [
    {"learning_rate": 0.05},
    {"min_samples_leaf": 20},
    {"l2_regularization": 0.0},
    {"learning_rate": 0.05, "min_samples_leaf": 20},
    {"learning_rate": 0.05, "l2_regularization": 0.0},
    {"min_samples_leaf": 20, "l2_regularization": 0.0},
])
def test_partial_new_hgb_arguments_refuse(knobs):
    with pytest.raises(ValueError):
        ridge_hgb_recipe(model="history_hgb", columns=["a"], feature_dictionary_hash=HASH,
                         max_leaf_nodes=7, **knobs)


@pytest.mark.parametrize("knobs", [
    {**HGB_V2_KNOBS, "learning_rate": 0.0},
    {**HGB_V2_KNOBS, "learning_rate": -0.1},
    {**HGB_V2_KNOBS, "learning_rate": 1.0001},
    {**HGB_V2_KNOBS, "learning_rate": float("nan")},
    {**HGB_V2_KNOBS, "learning_rate": float("inf")},
    {**HGB_V2_KNOBS, "learning_rate": True},
    {**HGB_V2_KNOBS, "min_samples_leaf": 0},
    {**HGB_V2_KNOBS, "min_samples_leaf": 1001},
    {**HGB_V2_KNOBS, "min_samples_leaf": 20.0},
    {**HGB_V2_KNOBS, "min_samples_leaf": True},
    {**HGB_V2_KNOBS, "l2_regularization": -1.0},
    {**HGB_V2_KNOBS, "l2_regularization": 1000.0001},
    {**HGB_V2_KNOBS, "l2_regularization": float("nan")},
    {**HGB_V2_KNOBS, "l2_regularization": True},
    {**HGB_V2_KNOBS, "max_leaf_nodes": 1},
    {**HGB_V2_KNOBS, "max_leaf_nodes": 65},
    {**HGB_V2_KNOBS, "max_leaf_nodes": 7.0},
    {**HGB_V2_KNOBS, "max_leaf_nodes": True},
])
def test_out_of_bounds_v2_hgb_arguments_refuse(knobs):
    with pytest.raises(ValueError):
        ridge_hgb_recipe(model="history_hgb", columns=["a"], feature_dictionary_hash=HASH, **knobs)


def test_new_hgb_arguments_are_refused_for_ridge():
    with pytest.raises(ValueError, match="Ridge requires"):
        ridge_hgb_recipe(model="history_ridge", columns=["a"], feature_dictionary_hash=HASH,
                         alpha=1.0, **HGB_V2_KNOBS)


@pytest.mark.parametrize("model,knobs", CASES)
def test_identity_binds_parameters_dictionary_order_and_width(model, knobs):
    def recipe(columns, digest=HASH):
        return ridge_hgb_recipe(model=model, columns=columns, feature_dictionary_hash=digest, **knobs)

    policies = [recipe(["a", "b"]), recipe(["b", "a"]), recipe(["a"]),
                recipe(["a", "b"], "sha256:" + "b" * 64)]
    assert len({content_hash(policy) for policy in policies}) == len(policies)
    for policy in policies:
        assert validate_ridge_hgb_recipe(policy) == policy
    assert len({content_hash(ridge_hgb_recipe(model=m, columns=["a", "b"],
                feature_dictionary_hash=HASH, **k)) for m, k in CASES}) == 5


@pytest.mark.parametrize("columns", [[], (), "one", {"one"}, ["a", "a"], [""],
                                   ["a", 1], [None], ["a-b"], ["A"], ["a\n"], [["a"]]])
def test_bad_columns_refuse(columns):
    with pytest.raises(ValueError, match="columns"):
        ridge_hgb_recipe(model="history_ridge", columns=columns, feature_dictionary_hash=HASH, alpha=1)


@pytest.mark.parametrize("digest", [None, "latest", "a" * 64, "sha256:" + "A" * 64, HASH + "\n"])
def test_bad_dictionary_hash_refuses(digest):
    with pytest.raises(ValueError, match="hash"):
        ridge_hgb_recipe(model="history_ridge", columns=["a"], feature_dictionary_hash=digest, alpha=1)


@pytest.mark.parametrize("model,knobs", [
    ("ridge", {"alpha": 1}), (None, {}), ("history_ridge", {}), ("history_hgb", {}),
    ("history_ridge", {"alpha": True}), ("history_ridge", {"alpha": 1e-7}),
    ("history_ridge", {"alpha": 1000.0001}), ("history_ridge", {"alpha": 0}),
    ("history_ridge", {"alpha": -1}), ("history_ridge", {"alpha": "1"}),
    ("history_ridge", {"alpha": float("nan")}), ("history_ridge", {"alpha": float("inf")}),
    ("history_ridge", {"alpha": 1, "max_leaf_nodes": 7}),
    ("history_hgb", {"max_leaf_nodes": 7, "alpha": 1}),
    ("history_hgb", {"max_leaf_nodes": 7.0}), ("history_hgb", {"max_leaf_nodes": True}),
    ("history_hgb", {"max_leaf_nodes": 8}),
])
def test_bad_or_wrong_family_parameters_refuse(model, knobs):
    message = ("Ridge requires" if model == "history_ridge" else
               "HGB requires" if model == "history_hgb" else "unknown variable-width magnitude model")
    with pytest.raises(ValueError, match=message):
        ridge_hgb_recipe(model=model, columns=["a"], feature_dictionary_hash=HASH, **knobs)


@pytest.mark.parametrize("model,knobs", CASES)
def test_any_fixed_policy_change_or_unknown_field_refuses(model, knobs):
    original = ridge_hgb_recipe(model=model, columns=["a", "b"], feature_dictionary_hash=HASH, **knobs)
    changes = [(("schema_version",), "magnitude_fitted_recipe.v1"),
               (("numerical_threads",), True), (("fit_scope",), "global"),
               (("target_transform",), "log"), (("prediction_projection",), "none"),
               (("estimator", "class"), "other"),
               (("estimator", "params", "random_state"), 7),
               (("preprocessing",), {}), (("extra",), 1),
               (("estimator", "params", "extra"), None),
               (("preprocessing", "extra"), float("inf")),
               (("preprocessing", "extra"), object())]
    for path, value in changes:
        altered = copy.deepcopy(original)
        parent = altered
        for key in path[:-1]:
            parent = parent[key]
        parent[path[-1]] = value
        with pytest.raises(ValueError, match="closed"):
            validate_ridge_hgb_recipe(altered)
    for key in original:
        altered = copy.deepcopy(original)
        del altered[key]
        with pytest.raises(ValueError, match="closed"):
            validate_ridge_hgb_recipe(altered)


@pytest.mark.parametrize("recipe", [None, [], "recipe", 1, {"estimator": {"params": []}}])
def test_malformed_envelopes_refuse(recipe):
    with pytest.raises(ValueError, match="closed"):
        validate_ridge_hgb_recipe(recipe)



def test_non_json_object_cannot_masquerade_as_valid_canonical_policy():
    class LooksLikeNone:
        def __str__(self):
            return "none"

    recipe = ridge_hgb_recipe(model="history_ridge", columns=["a"], feature_dictionary_hash=HASH, alpha=1)
    recipe["target_transform"] = LooksLikeNone()
    with pytest.raises(ValueError, match="invalid closed variable-width magnitude recipe"):
        validate_ridge_hgb_recipe(recipe)


@pytest.mark.parametrize("params", [[], None, "parameters", 1])
def test_malformed_parameters_inside_otherwise_complete_envelope_refuse(params):
    recipe = ridge_hgb_recipe(model="history_ridge", columns=["a"], feature_dictionary_hash=HASH, alpha=1)
    recipe["estimator"]["params"] = params
    with pytest.raises(ValueError, match="invalid closed variable-width magnitude recipe"):
        validate_ridge_hgb_recipe(recipe)


def test_excessive_recipe_nesting_is_a_closed_refusal():
    nested = []
    for _ in range(2000):
        nested = [nested]
    recipe = ridge_hgb_recipe(model="history_ridge", columns=["a"], feature_dictionary_hash=HASH, alpha=1)
    recipe["extra"] = nested
    with pytest.raises(ValueError, match="invalid closed variable-width magnitude recipe"):
        validate_ridge_hgb_recipe(recipe)


def test_metadata_never_imports_training_sklearn_or_source_admission():
    probe = '''
import importlib.abc
import sys
class RejectSources(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith(("engine.v2.models.training", "sklearn",
                                        "engine.v2.research", "engine.v2.ops")), fullname
sys.meta_path.insert(0, RejectSources())
from engine.v2.models.magnitude_ridge_hgb_recipe import ridge_hgb_recipe, validate_ridge_hgb_recipe
for model, knobs in (("history_ridge", {"alpha": 10}), ("history_hgb", {"max_leaf_nodes": 15})):
    recipe = ridge_hgb_recipe(model=model, columns=["arbitrary"],
                             feature_dictionary_hash="sha256:" + "a" * 64, **knobs)
    assert validate_ridge_hgb_recipe(recipe) == recipe
'''
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
