"""Closed ElasticNet spline recipe, validation and candidate dispatch."""
import json

import pytest

from engine.v2.foundation import canonical_json
from engine.v2.models.magnitude_elasticnet_recipe import elasticnet_recipe
from engine.v2.models.magnitude_elasticnet_spline_recipe import (
    SPLINE_COLUMNS, elasticnet_spline_recipe, validate_elasticnet_spline_recipe)
from experiments.magnitude_candidate_recipe import candidate_recipe

COLS = list(SPLINE_COLUMNS) + ["x1", "x2"]
H = "sha256:" + "0" * 64
PARAMS = {"alpha": 0.1, "l1_ratio": 0.25}
SPLINE = {"columns": list(SPLINE_COLUMNS), "n_knots": 6, "degree": 3,
          "knot_rule": "quantile", "extrapolation": "constant",
          "include_bias": False, "input": "imputed", "output": "append"}
PLAIN = json.loads("""{
  "schema_version": "magnitude_elasticnet_recipe.v1",
  "model": "elasticnet",
  "features": ["mean_prior_abs_move", "ema2_prior_abs_move", "ema4_prior_abs_move",
    "ema8_prior_abs_move", "ema12_prior_abs_move", "iv10", "iv30", "exern_iv10",
    "exern_iv30", "fwd90_30", "fexern90_30", "rvol30", "spy_vol5", "spy_vol20",
    "spy_vol60", "season_heat_mean_abs_5", "season_heat_mean_abs_20",
    "season_heat_ratio_20_252", "tail_max_abs", "tail_std_abs", "tail_p75_abs",
    "tail_p90_abs", "tail_max4_abs", "tail_std_over_mean", "tail_max_over_mean",
    "iv10_30", "iv_rv", "ex_ratio", "iv30_chg5", "iv10_chg5", "own_rvol5",
    "own_rvol10", "own_rvol60", "own_absret1", "x1", "x2"],
  "feature_dictionary_hash": "__HASH__",
  "estimator": {"class": "ElasticNet", "params": {"alpha": 0.1, "l1_ratio": 0.25,
    "fit_intercept": true, "precompute": false, "max_iter": 10000, "copy_X": true,
    "tol": 0.0001, "warm_start": false, "positive": false, "random_state": 0,
    "selection": "cyclic"}},
  "preprocessing": {
    "imputer": {"strategy": "median", "keep_empty_features": true},
    "missing_indicator": {"features": "all", "sparse": false},
    "scaler": {"with_mean": true, "with_std": true},
    "output_features": 72},
  "target_transform": "none", "prediction_projection": "clip_zero",
  "fit_scope": "fold_and_horizon", "numerical_threads": 1,
  "nonconvergence": "variant_failure_no_retry"
}""".replace("__HASH__", H))


def _spline_recipe(**overrides):
    kwargs = {"columns": COLS, "feature_dictionary_hash": H, "n_knots": 6, **PARAMS}
    kwargs.update(overrides)
    return elasticnet_spline_recipe(**kwargs)


def test_recipe_pins_precompute_append_and_plain_policy():
    result = _spline_recipe()
    base = elasticnet_recipe(columns=COLS, feature_dictionary_hash=H, **PARAMS)
    params = result["estimator"]["params"]
    assert result["schema_version"] == "magnitude_elasticnet_spline_recipe.v1"
    assert params["precompute"] is True
    assert result["preprocessing"]["spline"] == SPLINE
    assert result["preprocessing"]["spline"]["columns"] == list(SPLINE_COLUMNS)
    assert result["preprocessing"]["output_features"] == 2 * len(COLS) + len(SPLINE_COLUMNS) * 7
    assert {k: v for k, v in params.items() if k != "precompute"} == {
        k: v for k, v in base["estimator"]["params"].items() if k != "precompute"}
    assert params["max_iter"] == base["estimator"]["params"]["max_iter"]
    for key in ("numerical_threads", "nonconvergence"):
        assert result[key] == base[key]


def test_validate_round_trips_and_refuses_drift():
    assert canonical_json(validate_elasticnet_spline_recipe(_spline_recipe())) == \
        canonical_json(_spline_recipe())
    mutations = [
        lambda r: r["estimator"]["params"].__setitem__("precompute", False),
        lambda r: r["preprocessing"]["spline"].__setitem__("n_knots", 7),
        lambda r: r["preprocessing"]["spline"].__setitem__("include_bias", True),
        lambda r: r["preprocessing"]["spline"]["columns"].reverse(),
        lambda r: r["preprocessing"].__setitem__("extra", 1),
        lambda r: r.__setitem__("schema_version", "magnitude_elasticnet_recipe.v1"),
    ]
    for mutate in mutations:
        candidate = _spline_recipe()
        mutate(candidate)
        with pytest.raises(ValueError, match="^invalid closed ElasticNet spline recipe$"):
            validate_elasticnet_spline_recipe(candidate)


@pytest.mark.parametrize("n_knots", [3, 9, 6.0, True, "6"])
def test_n_knots_must_be_a_strict_integer_in_range(n_knots):
    with pytest.raises(ValueError, match="^spline n_knots must be an integer in 4..8$"):
        _spline_recipe(n_knots=n_knots)


def test_columns_missing_a_spline_column_refuse():
    with pytest.raises(ValueError,
                       match="^spline columns must be a subset of the feature columns$"):
        _spline_recipe(columns=COLS[1:])


def test_plain_recipe_bytes_stay_frozen():
    recipe = elasticnet_recipe(columns=COLS, feature_dictionary_hash=H, **PARAMS)
    assert recipe == PLAIN
    assert canonical_json(recipe) == canonical_json(PLAIN)
    assert "spline" not in recipe["preprocessing"]
    assert recipe["estimator"]["params"]["precompute"] is False


def test_dispatch_zero_knots_is_the_plain_recipe_and_positive_knots_select_spline():
    base = candidate_recipe("elasticnet", dict(PARAMS), columns=COLS,
                            feature_dictionary_hash=H)
    zero = candidate_recipe("elasticnet", {**PARAMS, "spline_knots": 0}, columns=COLS,
                            feature_dictionary_hash=H)
    assert canonical_json(zero) == canonical_json(base)
    spline = candidate_recipe("elasticnet", {**PARAMS, "spline_knots": 6}, columns=COLS,
                              feature_dictionary_hash=H)
    assert canonical_json(spline) == canonical_json(_spline_recipe())


@pytest.mark.parametrize("knots", [3, True, 6.0])
def test_dispatch_refuses_invalid_spline_knots(knots):
    with pytest.raises(ValueError, match="^spline n_knots must be an integer in 4..8$"):
        candidate_recipe("elasticnet", {**PARAMS, "spline_knots": knots}, columns=COLS,
                         feature_dictionary_hash=H)


def test_other_families_still_refuse_spline_knots():
    with pytest.raises(ValueError,
                       match="^candidate requires a supported family and exact parameter keys$"):
        candidate_recipe("ridge", {"alpha": 1, "spline_knots": 6}, columns=COLS,
                         feature_dictionary_hash=H)
