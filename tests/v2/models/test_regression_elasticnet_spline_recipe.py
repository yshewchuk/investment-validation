"""Signed spline ElasticNet recipe and cross-schema validation."""
import pytest

from engine.v2.foundation import canonical_json
from engine.v2.models.magnitude_elasticnet_spline_recipe import (
    SPLINE_COLUMNS, elasticnet_spline_recipe, validate_elasticnet_spline_recipe)
from engine.v2.models.regression_elasticnet_spline_recipe import (
    regression_elasticnet_spline_recipe, validate_regression_elasticnet_spline_recipe)

COLS = list(SPLINE_COLUMNS) + ["x1", "x2"]
H = "sha256:" + "0" * 64
PARAMS = {"alpha": 0.1, "l1_ratio": 0.25}


def _signed(**overrides):
    kwargs = {"target_domain": "signed", "columns": COLS, "feature_dictionary_hash": H,
              "n_knots": 6, **PARAMS}
    kwargs.update(overrides)
    return regression_elasticnet_spline_recipe(**kwargs)


def _unsigned(**overrides):
    kwargs = {"columns": COLS, "feature_dictionary_hash": H, "n_knots": 6, **PARAMS}
    kwargs.update(overrides)
    return elasticnet_spline_recipe(**kwargs)


def test_recipe_is_the_signed_twin_with_an_unchanged_spline_block():
    result = _signed()
    base = _unsigned()
    assert result["schema_version"] == "regression_elasticnet_spline_recipe.v1"
    assert result["target_domain"] == "signed"
    assert result["prediction_projection"] == "none"
    for key in ("preprocessing", "features", "estimator"):
        assert result[key] == base[key]
    assert result["preprocessing"]["spline"]["columns"] == list(SPLINE_COLUMNS)


def test_validate_round_trips_and_refuses_drift():
    assert canonical_json(validate_regression_elasticnet_spline_recipe(_signed())) == \
        canonical_json(_signed())
    mutations = [
        lambda r: r["preprocessing"]["spline"].__setitem__("n_knots", 7),
        lambda r: r.__setitem__("prediction_projection", "clip_zero"),
        lambda r: r.__setitem__("extra", 1),
        lambda r: r.__setitem__("target_domain", "nonnegative"),
    ]
    for mutate in mutations:
        candidate = _signed()
        mutate(candidate)
        with pytest.raises(
                ValueError,
                match="^invalid closed regression ElasticNet spline recipe$"):
            validate_regression_elasticnet_spline_recipe(candidate)


@pytest.mark.parametrize("domain", [None, "nonnegative", "SIGNED", True, []])
def test_builder_requires_the_signed_domain(domain):
    with pytest.raises(
            ValueError,
            match="^regression spline recipe requires the signed target domain$"):
        _signed(target_domain=domain)


def test_unsigned_spline_recipe_is_unchanged_and_schemas_do_not_cross_validate():
    unsigned = _unsigned()
    assert unsigned["schema_version"] == "magnitude_elasticnet_spline_recipe.v1"
    assert unsigned["prediction_projection"] == "clip_zero"
    assert canonical_json(validate_elasticnet_spline_recipe(unsigned)) == \
        canonical_json(unsigned)
    with pytest.raises(ValueError, match="^invalid closed ElasticNet spline recipe$"):
        validate_elasticnet_spline_recipe(_signed())
    with pytest.raises(
            ValueError,
            match="^invalid closed regression ElasticNet spline recipe$"):
        validate_regression_elasticnet_spline_recipe(unsigned)
