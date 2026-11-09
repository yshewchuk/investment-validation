"""v10 HGB quantile-loss axis, quantile recipes and estimator construction.

The v10 request copies ``v9_request`` and only swaps the schema version and the
HMOS qualified dictionary hash that v10 requires.
"""
import inspect

import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from engine.v2.foundation import content_hash
from engine.v2.models.magnitude_ridge_hgb_recipe import (
    ridge_hgb_recipe, validate_ridge_hgb_recipe)
from engine.v2.models.training._magnitude_estimators import build_estimator
from engine.v2.ops.errors import OpsError
from engine.v2.research.magnitude_panel_qualified_table import recipe as qualified_recipe
from experiments import magnitude_search_request as compiler
from experiments.magnitude_candidate_recipe import candidate_recipe
from experiments.magnitude_search_request_versions import compile_request
from tests.v2.research.test_magnitude_search_request_v9_hgb_loss import v9_request

HASH = "sha256:" + "a" * 64
HGB_BASE = {"learning_rate": 0.05, "max_leaf_nodes": 7,
            "min_samples_leaf": 20, "l2_regularization": 0.0}


def v10_request(**hgb_axes):
    request = v9_request(**hgb_axes)
    request["schema_version"] = "magnitude_search_request.v10"
    request["feature_dictionary_hash"] = qualified_recipe("HMOS")["dictionary_hash"]
    return request


def quantile_recipe(quantile=0.5):
    return ridge_hgb_recipe(model="history_hgb", columns=["x0"],
                            feature_dictionary_hash=HASH, loss="quantile",
                            quantile=quantile, **HGB_BASE)


def hgb_specs(result):
    return [spec for spec in result["candidate_specs"] if spec["family"] == "hgb"]


def test_v10_quantile_axis_expands_hgb_candidates_and_fits():
    baseline = compile_request(v10_request())
    assert baseline["counts"]["candidates"] == 35
    hgb_baseline = len(hgb_specs(baseline))

    single = compile_request(v10_request(loss=["quantile"], quantile=[0.5]))
    assert single["counts"]["candidates"] == 35
    assert single["request"]["space"]["hgb"]["loss"] == ["quantile"]
    assert single["request"]["space"]["hgb"]["quantile"] == [0.5]

    request = v10_request(loss=["quantile"], quantile=[0.25, 0.5, 0.75])
    compiled = compile_request(request)
    expected = 35 - hgb_baseline + 3 * hgb_baseline
    assert compiled["counts"]["candidates"] == expected
    assert len(hgb_specs(compiled)) == 3 * hgb_baseline

    counts, baseline_counts = compiled["counts"], baseline["counts"]
    assert counts["search_fits"] == 3 * (
        expected * baseline_counts["inner_folds"] + baseline_counts["outer_folds"])
    assert counts["search_fits"] - baseline_counts["search_fits"] == 3 * (
        expected - 35) * baseline_counts["inner_folds"]

    request["max_fits"] = counts["search_fits"] - 1
    with pytest.raises(OpsError) as caught:
        compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("overrides", [
    {"loss": ["quantile"]},
    {"quantile": [0.5]},
    {"loss": ["gamma"], "quantile": [0.5]},
    {"loss": ["gamma", "quantile"], "quantile": [0.5]},
    {"loss": ["quantile"], "quantile": [0.0]},
    {"loss": ["quantile"], "quantile": [1.0]},
    {"loss": ["quantile"], "quantile": [-0.1]},
    {"loss": ["quantile"], "quantile": [1.5]},
    {"loss": ["quantile"], "quantile": [True]},
    {"loss": ["quantile"], "quantile": [0.7, 0.5]},
    {"loss": ["quantile"], "quantile": [0.5, 0.5]},
    {"loss": ["quantile"], "quantile": []},
    {"loss": ["quantile"], "quantile": ["0.5"]},
])
def test_v10_refuses_invalid_quantile_axes_before_expansion(monkeypatch, overrides):
    monkeypatch.setattr(compiler, "_specifications",
                        lambda *args, **kwargs: pytest.fail("invalid axes expanded"))
    with pytest.raises(OpsError) as caught:
        compile_request(v10_request(**overrides))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_v10_quantile_recipe_round_trips_and_is_quantile_distinct():
    recipe = quantile_recipe()
    params = recipe["estimator"]["params"]
    assert params["loss"] == "quantile"
    assert params["quantile"] == 0.5
    assert "target_floor_pct" not in recipe
    assert validate_ridge_hgb_recipe(recipe) == recipe
    assert content_hash(quantile_recipe(0.25)) != content_hash(quantile_recipe(0.75))


def test_v10_quantile_recipe_refuses_invalid_pairings():
    with pytest.raises(ValueError):
        ridge_hgb_recipe(model="history_hgb", columns=["x0"],
                         feature_dictionary_hash=HASH, loss="quantile", **HGB_BASE)
    with pytest.raises(ValueError):
        ridge_hgb_recipe(model="history_hgb", columns=["x0"],
                         feature_dictionary_hash=HASH, loss="squared_error",
                         quantile=0.5, **HGB_BASE)
    with pytest.raises(ValueError):
        ridge_hgb_recipe(model="history_ridge", columns=["x0"],
                         feature_dictionary_hash=HASH, quantile=0.5, **HGB_BASE)
    with pytest.raises(ValueError):
        quantile_recipe(1.0)


def test_v10_candidate_recipe_carries_quantile_loss():
    recipe = candidate_recipe("hgb", {**HGB_BASE, "loss": "quantile", "quantile": 0.5},
                              columns=["x0"], feature_dictionary_hash=HASH)
    assert recipe["estimator"]["params"]["loss"] == "quantile"
    assert recipe["estimator"]["params"]["quantile"] == 0.5


def test_v10_build_estimator_constructs_quantile_loss():
    recipe = quantile_recipe()
    signature = inspect.signature(build_estimator)
    required = [parameter for parameter in signature.parameters.values()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind in (parameter.POSITIONAL_ONLY,
                                       parameter.POSITIONAL_OR_KEYWORD)]
    if len(required) > 1:
        pytest.skip("build_estimator requires more than the recipe")
    estimator = build_estimator(recipe)
    assert isinstance(estimator, HistGradientBoostingRegressor)
    params = estimator.get_params()
    assert params["loss"] == "quantile"
    assert params["quantile"] == 0.5


def test_v9_default_request_stays_at_35_candidates_without_quantile():
    compiled = compile_request(v9_request())
    assert compiled["counts"]["candidates"] == 35
    assert "quantile" not in compiled["request"]["space"]["hgb"]
