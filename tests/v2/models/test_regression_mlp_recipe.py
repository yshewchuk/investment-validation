"""Signed/nonnegative MLP recipe and cross-schema validation."""
import pytest

from engine.v2.foundation import canonical_json
from engine.v2.models.magnitude_mlp_recipe import _PROFILES, mlp_recipe
from engine.v2.models.regression_mlp_recipe import (
    regression_mlp_recipe, validate_regression_mlp_recipe)

COLS = ["a", "b", "c", "d", "e", "f"]
H = "sha256:" + "0" * 64
PROFILE = "relu32_l2_0p01"


def _recipe(**overrides):
    kwargs = {"target_domain": "signed", "columns": COLS,
              "feature_dictionary_hash": H, "profile": PROFILE, **overrides}
    return regression_mlp_recipe(**kwargs)


def _unsigned(**overrides):
    kwargs = {"columns": COLS, "feature_dictionary_hash": H, "profile": PROFILE, **overrides}
    return mlp_recipe(**kwargs)


def test_signed_recipe_mirrors_the_magnitude_neural_block():
    result = _recipe()
    base = _unsigned()
    assert result["schema_version"] == "regression_mlp_recipe.v1"
    assert result["target_domain"] == "signed"
    assert result["prediction_projection"] == "none"
    for key in ("estimator", "preprocessing", "features", "profile"):
        assert result[key] == base[key]


def test_nonnegative_recipe_binds_zero_clipping():
    result = _recipe(target_domain="nonnegative")
    assert result["target_domain"] == "nonnegative"
    assert result["prediction_projection"] == "clip_zero"


def test_magnitude_recipe_identity_is_unchanged():
    base = _unsigned()
    assert base["schema_version"] == "magnitude_mlp_recipe.v1"
    assert base["prediction_projection"] == "clip_zero"
    assert "target_domain" not in base


def test_validate_round_trips_and_refuses_drift():
    assert canonical_json(validate_regression_mlp_recipe(_recipe())) == \
        canonical_json(_recipe())
    mutations = [
        lambda r: r.__setitem__("extra", 1),
        lambda r: r.__setitem__("prediction_projection", "clip_zero"),
        lambda r: r.__setitem__("target_domain", "nonnegative"),
        lambda r: r.__setitem__("profile", "relu64x32_l2_0p01"),
    ]
    for mutate in mutations:
        candidate = _recipe()
        mutate(candidate)
        with pytest.raises(ValueError, match="^invalid closed regression MLP recipe$"):
            validate_regression_mlp_recipe(candidate)


@pytest.mark.parametrize("domain", [None, "SIGNED", True, []])
def test_builder_requires_a_signed_or_nonnegative_domain(domain):
    with pytest.raises(
            ValueError,
            match="^regression MLP recipe requires a signed or nonnegative target domain$"):
        _recipe(target_domain=domain)


def test_all_magnitude_profiles_build_and_validate():
    for profile in _PROFILES:
        recipe = _recipe(profile=profile)
        assert canonical_json(validate_regression_mlp_recipe(recipe)) == \
            canonical_json(recipe)
