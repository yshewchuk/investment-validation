"""Closed log1p target-transform recipes, fitters and diagnostics."""
import copy
from pathlib import Path

import numpy as np
import pytest
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

from engine.v2.foundation import content_hash
from engine.v2.models.magnitude_candidate import magnitude_candidate_family
from engine.v2.models.magnitude_elasticnet_recipe import (
    LOG1P_TARGET, elasticnet_recipe, validate_elasticnet_recipe)
from engine.v2.models.magnitude_elasticnet_spline_recipe import (
    SPLINE_COLUMNS, elasticnet_spline_recipe, validate_elasticnet_spline_recipe)
from engine.v2.models.magnitude_mlp_recipe import mlp_recipe, validate_mlp_recipe
from engine.v2.models.training import _magnitude_elasticnet_estimator as estimator_module
from engine.v2.models.training.magnitude_elasticnet import fit_elasticnet_predictions
from engine.v2.models.training.magnitude_mlp import fit_mlp_predictions
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import source_closure
from experiments.magnitude_candidate_recipe import candidate_recipe
from experiments.size_fit_diagnostics import evidence

HASH = "sha256:" + "a" * 64
COLS = ["x0", "x1", "x2"]
SPLINE_COLS = list(SPLINE_COLUMNS) + ["extra_col"]
EN = {"alpha": 0.01, "l1_ratio": 0.75}
TWIN_EN = {**EN, "target": "log1p_median"}


def en_target(columns=COLS, alpha=0.01):
    return candidate_recipe("elasticnet", {"alpha": alpha, "l1_ratio": 0.75,
                                           "target": "log1p_median"},
                            columns=columns, feature_dictionary_hash=HASH)


def spline_target(columns=SPLINE_COLS):
    return candidate_recipe("elasticnet", {**EN, "spline_knots": 6,
                                           "target": "log1p_median"},
                            columns=columns, feature_dictionary_hash=HASH)


def mlp_target(columns=COLS, profile="relu32_l2_0p01"):
    return candidate_recipe("mlp", {"profile": profile, "target": "log1p_median"},
                            columns=columns, feature_dictionary_hash=HASH)


def matrices(rows=60, width=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((rows, width)) + 0.1, rng.random((5, width)) + 0.1


def test_base_recipes_are_byte_identical():
    direct = elasticnet_recipe(columns=COLS, feature_dictionary_hash=HASH, **EN)
    assert direct["schema_version"] == "magnitude_elasticnet_recipe.v1"
    assert direct["target_transform"] == "none"
    assert candidate_recipe("elasticnet", EN, columns=COLS,
                            feature_dictionary_hash=HASH) == direct
    assert content_hash(candidate_recipe("elasticnet", EN, columns=COLS,
                                         feature_dictionary_hash=HASH)) == content_hash(direct)


def test_twin_recipes_are_distinct_v2_schemas():
    base = candidate_recipe("elasticnet", EN, columns=COLS, feature_dictionary_hash=HASH)
    twin = candidate_recipe("elasticnet", TWIN_EN, columns=COLS,
                            feature_dictionary_hash=HASH)
    assert twin["schema_version"] == "magnitude_elasticnet_recipe.v2"
    assert twin["target_transform"]["kind"] == "log1p"
    assert content_hash(twin) != content_hash(base)

    base = candidate_recipe("elasticnet", {**EN, "spline_knots": 6},
                            columns=SPLINE_COLS, feature_dictionary_hash=HASH)
    twin = spline_target()
    assert twin["schema_version"] == "magnitude_elasticnet_spline_recipe.v2"
    assert twin["target_transform"]["kind"] == "log1p"
    assert content_hash(twin) != content_hash(base)

    base = candidate_recipe("mlp", {"profile": "chrono10_relu64x32_l2_1"},
                            columns=COLS, feature_dictionary_hash=HASH)
    twin = mlp_target(profile="chrono10_relu64x32_l2_1")
    assert twin["schema_version"] == "magnitude_mlp_recipe.v2"
    assert twin["target_transform"]["kind"] == "log1p"
    assert twin["target_transform"]["then"] == "StandardScaler"
    assert content_hash(twin) != content_hash(base)

    zero = candidate_recipe("elasticnet", {**EN, "spline_knots": 0,
                                           "target": "log1p_median"},
                            columns=COLS, feature_dictionary_hash=HASH)
    assert zero["schema_version"] == "magnitude_elasticnet_recipe.v2"
    assert "spline" not in zero["preprocessing"]


def test_validators_round_trip_and_refuse_drift():
    plain, spline, mlp = en_target(), spline_target(), mlp_target()
    assert validate_elasticnet_recipe(plain) == plain
    assert validate_elasticnet_spline_recipe(spline) == spline
    assert validate_mlp_recipe(mlp) == mlp

    for validate, recipe in (
            (validate_elasticnet_recipe, plain),
            (validate_elasticnet_spline_recipe, spline),
            (validate_mlp_recipe, mlp)):
        with pytest.raises(ValueError):
            validate({**recipe, "target_transform": "none"})
    for make, validate in (
            (lambda: candidate_recipe("elasticnet", EN, columns=COLS,
                                      feature_dictionary_hash=HASH),
             validate_elasticnet_recipe),
            (lambda: elasticnet_spline_recipe(columns=SPLINE_COLS,
                                              feature_dictionary_hash=HASH,
                                              n_knots=6, **EN),
             validate_elasticnet_spline_recipe),
            (lambda: mlp_recipe(columns=COLS, feature_dictionary_hash=HASH,
                                profile="relu32_l2_0p01"), validate_mlp_recipe)):
        with pytest.raises(ValueError):
            validate({**make(), "target_transform": dict(LOG1P_TARGET)})

    with pytest.raises(ValueError):
        candidate_recipe("hgb", {"max_leaf_nodes": 7, "target": "log1p_median"},
                         columns=COLS, feature_dictionary_hash=HASH)
    with pytest.raises(ValueError):
        candidate_recipe("ridge", {"alpha": 1, "target": "x"},
                         columns=COLS, feature_dictionary_hash=HASH)
    with pytest.raises(ValueError, match="candidate target must be 'log1p_median'"):
        candidate_recipe("elasticnet", {**EN, "target": "other"},
                         columns=COLS, feature_dictionary_hash=HASH)


def test_candidate_family_accepts_v2_schemas():
    assert magnitude_candidate_family(en_target()) == "elasticnet"
    assert magnitude_candidate_family(spline_target()) == "elasticnet"
    assert magnitude_candidate_family(mlp_target()) == "mlp"


def test_elasticnet_log1p_fit_predicts_on_original_scale():
    train, prediction = matrices(seed=1)
    recipe = en_target()
    labels = np.random.default_rng(2).random(60) + 0.5
    result = fit_elasticnet_predictions(recipe, train, labels, prediction, columns=COLS)
    predictions = np.asarray(result["predictions"])
    assert result["target_transform"] == "log1p"
    assert result["coefficient_scale"] == "standardized_inputs_log1p_target"
    assert np.isfinite(predictions).all() and (predictions >= 0).all()

    constant = fit_elasticnet_predictions(
        en_target(alpha=0.001), train, np.full(60, 2.0), prediction, columns=COLS)
    assert np.allclose(constant["predictions"], 2.0, atol=1e-6)


def test_zero_labels_stay_zero_without_retry():
    train, prediction = matrices(seed=3)
    labels = np.zeros(60)
    result = fit_elasticnet_predictions(en_target(), train, labels, prediction, columns=COLS)
    assert result["predictions"] == [0.0] * len(prediction)
    assert all(entry["coefficient"] == 0.0 for entry in result["coefficients"])

    result = fit_mlp_predictions(mlp_target(), train, labels, prediction, columns=COLS)
    predictions = np.asarray(result["predictions"])
    assert result["target_transform"] == "log1p"
    assert np.isfinite(predictions).all() and (predictions < 0.5).all()


def test_expm1_overflow_raises_finite_prediction_error(monkeypatch):
    monkeypatch.setattr(estimator_module.np, "expm1",
                        lambda value: np.full_like(value, np.inf))
    train, prediction = matrices(seed=4)
    with pytest.raises(ValueError, match="predictions require one finite value"):
        fit_elasticnet_predictions(en_target(), train, np.ones(60) * 2.0,
                                   prediction, columns=COLS)


def _diagnostics(recipe, result):
    detail = {"horizon_sessions": 1,
              **{key: value for key, value in result.items() if key != "predictions"}}
    return {"event": "succeeded", "diagnostics": [detail]}


def _execution(recipe):
    return {"schema_version": "size_candidate_execution.v1", "fit_recipe": recipe}


def test_log1p_diagnostics_admitted_and_tampering_refused():
    train, prediction = matrices(seed=5)
    recipe = en_target()
    labels = np.random.default_rng(6).random(60) + 0.5
    document = _diagnostics(recipe, fit_elasticnet_predictions(
        recipe, train, labels, prediction, columns=COLS))
    assert evidence(document, _execution(recipe), [1]) == {"diagnostics": document["diagnostics"]}

    for field, value in (("target_transform", None),
                         ("coefficient_scale", "standardized_inputs")):
        tampered = copy.deepcopy(document)
        if value is None:
            del tampered["diagnostics"][0][field]
        else:
            tampered["diagnostics"][0][field] = value
        with pytest.raises(OpsError) as caught:
            evidence(tampered, _execution(recipe), [1])
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"

    recipe = mlp_target()
    document = _diagnostics(recipe, fit_mlp_predictions(
        recipe, train, np.zeros(60), prediction, columns=COLS))
    assert evidence(document, _execution(recipe), [1]) == {"diagnostics": document["diagnostics"]}


def test_en_log1p_prediction_matches_independent_oracle():
    train, prediction = matrices(seed=7)
    labels = np.random.default_rng(8).exponential(3.0, 60)
    recipe = en_target()
    result = fit_elasticnet_predictions(recipe, train, labels, prediction, columns=COLS)

    params = recipe["estimator"]["params"]
    oracle = make_pipeline(
        FeatureUnion([
            ("i", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("m", MissingIndicator(features="all", sparse=False)),
        ]),
        StandardScaler(),
        ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"],
                   max_iter=10000, tol=0.0001, precompute=False,
                   random_state=0, selection="cyclic"),
    )
    oracle.fit(train, np.log1p(labels))
    expected = np.maximum(np.expm1(oracle.predict(prediction)), 0)
    assert np.allclose(result["predictions"], expected, rtol=1e-9, atol=1e-9)

    none = candidate_recipe("elasticnet", {"alpha": params["alpha"],
                                           "l1_ratio": params["l1_ratio"]},
                            columns=COLS, feature_dictionary_hash=HASH)
    baseline = fit_elasticnet_predictions(none, train, labels, prediction, columns=COLS)
    difference = np.abs(np.asarray(result["predictions"])
                        - np.asarray(baseline["predictions"])).max()
    assert difference > 1e-3


def _fit_mlp_pair(profile, train, labels, prediction):
    none = candidate_recipe("mlp", {"profile": profile}, columns=COLS,
                            feature_dictionary_hash=HASH)
    log1p = candidate_recipe("mlp", {"profile": profile, "target": "log1p_median"},
                             columns=COLS, feature_dictionary_hash=HASH)
    return (fit_mlp_predictions(none, train, labels, prediction, columns=COLS),
            fit_mlp_predictions(log1p, train, labels, prediction, columns=COLS))


def test_mlp_log1p_changes_fitted_predictions():
    train, prediction = matrices(seed=10)
    labels = np.random.default_rng(9).exponential(3.0, 60)
    try:
        none_result, log1p_result = _fit_mlp_pair("relu32_l2_0p01", train, labels, prediction)
    except ValueError as error:
        if "convergence" not in str(error):
            raise
        none_result, log1p_result = _fit_mlp_pair(
            "fixed200_relu64x32_l2_1", train, labels, prediction)

    none = np.asarray(none_result["predictions"])
    log1p = np.asarray(log1p_result["predictions"])
    assert np.isfinite(none).all() and np.isfinite(log1p).all()
    assert np.abs(none - log1p).max() > 1e-3


def test_mlp_log1p_matches_none_recipe_fitted_on_log1p_labels():
    train, prediction = matrices(seed=11)
    labels = np.random.default_rng(12).exponential(3.0, 60)
    profile = "fixed200_relu64x32_l2_1"
    none_recipe = candidate_recipe("mlp", {"profile": profile}, columns=COLS,
                                   feature_dictionary_hash=HASH)
    log1p_recipe = candidate_recipe("mlp", {"profile": profile,
                                            "target": "log1p_median"},
                                    columns=COLS, feature_dictionary_hash=HASH)

    log1p = np.asarray(fit_mlp_predictions(
        log1p_recipe, train, labels, prediction, columns=COLS)["predictions"])
    expected = np.expm1(np.asarray(fit_mlp_predictions(
        none_recipe, train, np.log1p(labels), prediction, columns=COLS)["predictions"]))
    assert np.allclose(log1p, expected, rtol=1e-6, atol=1e-8)
    assert expected.max() > 0.5


def test_invalid_target_transform_refused():
    with pytest.raises(ValueError, match="target_transform"):
        elasticnet_recipe(columns=COLS, feature_dictionary_hash=HASH,
                          target_transform="bad", **EN)
    with pytest.raises(ValueError, match="target_transform"):
        mlp_recipe(columns=COLS, feature_dictionary_hash=HASH,
                   profile="relu32_l2_0p01", target_transform="bad")
    with pytest.raises(ValueError, match="target_transform"):
        elasticnet_spline_recipe(columns=SPLINE_COLS, feature_dictionary_hash=HASH,
                                 n_knots=6, target_transform="bad", **EN)


def test_log1p_diagnostics_wrong_values_refused():
    train, prediction = matrices(seed=11)
    recipe = en_target()
    labels = np.random.default_rng(12).random(60) + 0.5
    document = _diagnostics(recipe, fit_elasticnet_predictions(
        recipe, train, labels, prediction, columns=COLS))
    tampered = copy.deepcopy(document)
    tampered["diagnostics"][0]["target_transform"] = "none"
    with pytest.raises(OpsError) as caught:
        evidence(tampered, _execution(recipe), [1])
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"

    recipe = mlp_target()
    document = _diagnostics(recipe, fit_mlp_predictions(
        recipe, train, np.zeros(60), prediction, columns=COLS))
    for mutate in (lambda item: item.pop("target_transform"),
                   lambda item: item.__setitem__("target_transform", "none")):
        tampered = copy.deepcopy(document)
        mutate(tampered["diagnostics"][0])
        with pytest.raises(OpsError) as caught:
            evidence(tampered, _execution(recipe), [1])
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_panel_closure_is_unchanged():
    closure = source_closure(Path(__file__).resolve().parents[3],
                             ["experiments/size_panel.py"])
    assert len(closure) == 150
    assert content_hash(closure) == (
        "sha256:b286c0f39239f8d603b54a7907e3f9eacfcd3b9b2216c59781040d33defed05d")
    edited = {
        "engine/v2/models/magnitude_elasticnet_recipe.py",
        "engine/v2/models/magnitude_elasticnet_spline_recipe.py",
        "engine/v2/models/magnitude_mlp_recipe.py",
        "engine/v2/models/magnitude_candidate.py",
        "experiments/magnitude_candidate_recipe.py",
        "engine/v2/models/training/_magnitude_elasticnet_estimator.py",
        "engine/v2/models/training/_magnitude_elasticnet_spline_fit.py",
        "engine/v2/models/training/magnitude_elasticnet.py",
        "engine/v2/models/training/_magnitude_mlp_estimator.py",
        "engine/v2/models/training/magnitude_mlp.py",
        "experiments/size_fit_diagnostics.py",
        "tests/v2/research/test_size_log_target_recipes.py",
    }
    assert not edited & set(closure)
