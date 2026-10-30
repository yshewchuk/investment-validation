"""Independent causal feature/matrix, signed native-fit and training-only mean oracles."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from engine.models.no_fit import RuntimeFitForbidden as LegacyFitForbidden
from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.foundation import content_hash
from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.models.training.recipes import RecipeKey
from engine.v2.ops.errors import OpsError
from engine.v2.research.magnitude_panel_qualified_table import recipe as panel_recipe
from experiments import iv_crush_comparator as comparator

FEATURES = ("n_prior mean_prior_move mean_prior_abs_move ema2_prior_move ema4_prior_move "
    "ema8_prior_move ema12_prior_move ema2_prior_abs_move ema4_prior_abs_move ema8_prior_abs_move "
    "ema12_prior_abs_move mean_prior_or_implied spy_ret21 spy_ret63 spy_ret252 spy_dd252 spy_vol20 "
    "signed_streak ema12r_abs dist_high dist_ema ret5 ret10 ret20 or_implied or_skewing or_contango "
    "or_rvol30 or_exern30 or_iv30 or_iee or_fwd90_30 or_fexern90_30 mcap_log abs_dist_high "
    "abs_dist_ema has_implied_quote pre_iv30 pre_iv10 pre_exern_iv30 pre_exern_iv10").split()
ALIASES = dict(zip("or_implied or_skewing or_contango or_rvol30 or_exern30 or_iv30 or_iee or_fwd90_30 "
    "or_fexern90_30 pre_iv30 pre_iv10 pre_exern_iv30 pre_exern_iv10".split(),
    "implied_move skew contango rvol30 exern_iv30 iv30 iee fwd90_30 fexern90_30 iv30 iv10 exern_iv30 exern_iv10".split()))


def row(event, horizon):
    value = {name: float(i + horizon) for i, name in enumerate(panel_recipe("HMO")["model_columns"])}
    value.update(event_id=event, context_id="context-" + event, horizon_sessions=horizon,
                 n_prior=1, has_implied_quote=1, dist_high=-7.0)
    # These future-pre-print aliases must never replace forecast-cutoff IV state.
    value.update(pre_iv30=9999., pre_iv10=9999., pre_exern_iv30=9999., pre_exern_iv10=9999.)
    return value


def case(n=500):
    indexed = {(f"train{i}", h): row(f"train{i}", h) for i in range(n) for h in (5, 3, 1)}
    indexed.update({("test", h): row("test", h) for h in (5, 3, 1)})
    training = pd.DataFrame({"event_id": [f"train{i}" for i in range(n)],
                             "crush_pct_iv30": [float(i % 7 - 4) for i in range(n)]})
    return indexed, training, {"evaluation_event_ids": ["test"]}, {"horizons": [5, 3, 1]}, {
        "features": comparator.COLUMNS, "fit_recipe": comparator.recipe()}


def expected_values(value):
    return [abs(value["dist_high"]) if name == "abs_dist_high" else value[ALIASES.get(name, name)] for name in FEATURES]


def test_exact_native_policy_and_causal_mapping():
    declared = comparator.recipe()
    assert comparator.FEATURES == FEATURES and len(FEATURES) == 41
    assert len(comparator.COLUMNS) == 38
    assert set(comparator.COLUMNS) <= set(panel_recipe("HMO")["model_columns"])
    native = declared["native"]
    assert native["estimator"] == {"kind": "hgb", "params": {"learning_rate": .06, "max_iter": 300}, "seeds": [20260829]}
    assert native["filters"] == native["value_masks"] == []
    assert native["missing_mask"] == "drop_rows_with_nonfinite_feature_or_target"
    assert native["target"]["signed"] and native["target"]["transform"] is None
    assert declared["minimum_training_rows"] == 500 and declared["production_reproduction"] is False
    assert declared["prediction_projection"] == "none"
    assert declared["feature_mapping"]["pre_iv30"] == "iv30"
    assert declared["feature_mapping"]["abs_dist_high"] == "abs(dist_high)"
    declared["features"].clear()
    assert comparator.recipe()["features"] == FEATURES


def test_exact_complete_case_arrays_evidence_and_training_mean(monkeypatch):
    indexed, training, fold, plan, execution = case(502)
    training.loc[0, "crush_pct_iv30"] = -80.
    training.loc[1, "crush_pct_iv30"] = 120.
    for h in plan["horizons"]:
        indexed["train0", h]["iv30"] = None
        indexed["train1", h]["mcap_log"] = np.nan
    indexed["test", 3]["skew"] = np.inf
    calls, progress, before = [], [], deepcopy(training)
    def fit(native, X, y):
        calls.append((X.copy(), y.copy()))
        return SimpleNamespace(predict=lambda values: np.full(len(values), -12.5))
    monkeypatch.setattr(comparator, "fit_recipe_estimator", fit)
    result = comparator.forecasts(indexed, training, fold, plan, execution,
                                  progress=lambda state, fit: progress.append((state, fit)))
    assert len(calls) == 3
    for h, (matrix, labels) in zip(plan["horizons"], calls):
        np.testing.assert_array_equal(matrix, [expected_values(indexed[f"train{i}", h]) for i in range(2, 502)])
        np.testing.assert_array_equal(labels, training.crush_pct_iv30.to_numpy()[2:])
    pd.testing.assert_frame_equal(training, before)
    assert [(r["horizon_sessions"], r["predicted_crush_pct_iv30"], r["abstention"]) for r in result["forecasts"]] == [
        (1, -12.5, None), (3, None, "unavailable_input"), (5, -12.5, None)]
    assert all(r["training_mean"] == pytest.approx(training.crush_pct_iv30.mean()) for r in result["forecasts"])
    evidence = comparator.fit_evidence({"features": list(indexed.values()), "training": training.to_dict("records")}, fold, plan)
    assert progress == [(state, fit) for fit in evidence for state in ("started", "completed")]
    for fit in evidence:
        h = fit["horizon_sessions"]
        ids = [f"train{i}" for i in range(2, 502)]
        expected = {"columns": FEATURES, "row_keys": [[e, "context-" + e, h] for e in ids],
            "values": [expected_values(indexed[e, h]) for e in ids],
            "crush_pct_iv30": training.crush_pct_iv30.tolist()[2:]}
        assert fit["training_matrix_id"] == content_hash(expected)
        tests = [] if h == 3 else ["test"]
        assert fit["prediction_matrix_id"] == content_hash({"columns": FEATURES,
            "row_keys": [[e, "context-" + e, h] for e in tests], "values": [expected_values(indexed[e, h]) for e in tests]})
        assert fit["fit_eligible"] and fit["training_event_ids"] == ids
    assert all(d["logical_fits"] == d["component_fits"] == 1 for d in result["fit_details"])


@pytest.mark.parametrize("n", [1, 499])
def test_insufficient_complete_training_abstains_but_mean_uses_all_labels(monkeypatch, n):
    args = case(n)
    monkeypatch.setattr(comparator, "fit_recipe_estimator", lambda *a: pytest.fail("unexpected fit"))
    result = comparator.forecasts(*args, progress=lambda *a: None)
    assert all(r["abstention"] == "insufficient_training" and r["predicted_crush_pct_iv30"] is None
               and r["training_mean"] == pytest.approx(args[1].crush_pct_iv30.mean()) for r in result["forecasts"])
    assert all(d["logical_fits"] == d["component_fits"] == 0 for d in result["fit_details"])


@pytest.mark.parametrize("bad", [None, np.nan, np.inf, -np.inf, True])
def test_invalid_training_labels_refuse_before_fit(monkeypatch, bad):
    args = case(1)
    args[1]["crush_pct_iv30"] = pd.Series([bad], dtype=object)
    monkeypatch.setattr(comparator, "fit_recipe_estimator", lambda *a: pytest.fail("unexpected fit"))
    with pytest.raises(OpsError) as caught:
        comparator.forecasts(*args, progress=lambda *a: None)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_empty_training_refuses_before_fit():
    with pytest.raises(OpsError, match="nonempty finite"):
        comparator.forecasts(*case(0), progress=lambda *a: None)


@pytest.mark.parametrize("guard,error", [(no_fit_guard, RuntimeFitForbidden), (legacy_guard, LegacyFitForbidden)])
def test_both_no_fit_guards_precede_all_input_and_registry_inspection(monkeypatch, guard, error):
    monkeypatch.setattr(comparator, "native_recipe", lambda: pytest.fail("registry inspected"))
    with guard(), pytest.raises(error):
        comparator.forecasts(None, None, None, None, None, progress=None)


@pytest.mark.parametrize("change", ["seed", "features", "params", "target", "minimum", "mask"])
def test_native_recipe_drift_refuses(monkeypatch, change):
    native = comparator.current_recipes()[RecipeKey("iv_crush", "*", "champion")]
    if change == "seed":
        native = replace(native, estimator=replace(native.estimator, seeds=(7,)))
    elif change == "features":
        native = replace(native, features=tuple(reversed(native.features)))
    elif change == "params":
        native = replace(native, estimator=replace(native.estimator, params={"learning_rate": .07, "max_iter": 300}))
    elif change == "target":
        native = replace(native, target=replace(native.target, signed=False))
    elif change == "minimum":
        native = replace(native, folds=replace(native.folds, min_train_rows=499))
    else:
        native = replace(native, missing_mask="impute")
    monkeypatch.setattr(comparator, "current_recipes", lambda: {RecipeKey("iv_crush", "*", "champion"): native})
    with pytest.raises(ValueError, match="recipe changed"):
        comparator.native_recipe()


def test_real_native_refits_are_fresh_single_threaded_and_signed(monkeypatch):
    args = case()
    args[1]["crush_pct_iv30"] = -17.25
    actual_fit, models, threads = comparator.fit_recipe_estimator, [], []
    from threadpoolctl import threadpool_info
    def fitted(native, X, y):
        threads.extend(item["num_threads"] for item in threadpool_info())
        model = actual_fit(native, X, y)
        models.append(model)
        with threadpool_limits(limits=1):
            expected = HistGradientBoostingRegressor(learning_rate=.06, max_iter=300, random_state=20260829).fit(X, y)
        np.testing.assert_array_equal(model.predict(X[:2]), expected.predict(X[:2]))
        return model
    monkeypatch.setattr(comparator, "fit_recipe_estimator", fitted)
    result = comparator.forecasts(*args, progress=lambda *a: None)
    assert len({id(model) for model in models}) == 3 and set(threads) == {1}
    assert all(r["predicted_crush_pct_iv30"] == -17.25 and r["training_mean"] == -17.25 for r in result["forecasts"])


def test_nonconstant_native_math_matches_independent_hgb():
    indexed, training, fold, plan, execution = case()
    plan["horizons"] = [3]
    for i, event in enumerate(training.event_id):
        indexed[event, 3]["mcap_log"] = i / 50.
    training["crush_pct_iv30"] = np.sin(np.arange(500) / 25.) * 20 - 10
    indexed["test", 3]["mcap_log"] = 2.75
    with threadpool_limits(limits=1):
        expected = HistGradientBoostingRegressor(learning_rate=.06, max_iter=300, random_state=20260829).fit(
            np.array([expected_values(indexed[e, 3]) for e in training.event_id]), training.crush_pct_iv30.to_numpy())
        value = float(expected.predict(np.array([expected_values(indexed["test", 3])]))[0])
    result = comparator.forecasts(indexed, training, fold, plan, execution, progress=lambda *a: None)
    assert result["forecasts"][0]["predicted_crush_pct_iv30"] == value
    assert value != pytest.approx(training.crush_pct_iv30.mean())


def test_evaluation_labels_and_future_pre_fields_are_never_read(monkeypatch):
    args = case(1)
    class Poison:
        def __iter__(self):
            pytest.fail("evaluation labels inspected")
    class CutoffRow(dict):
        def __getitem__(self, key):
            if key.startswith("pre_") or key == "crush_pct_iv30":
                pytest.fail("future target-side field inspected")
            return super().__getitem__(key)
    for key, value in args[0].items():
        args[0][key] = CutoffRow(value)
    args[2]["evaluation_targets"] = Poison()
    monkeypatch.setattr(comparator, "fit_recipe_estimator", lambda *a: pytest.fail("unexpected fit"))
    result = comparator.forecasts(*args, progress=lambda *a: None)
    assert len(result["forecasts"]) == 3


def test_quote_indicator_is_required_not_inferred():
    args = case(1)
    del args[0]["test", 5]["has_implied_quote"]
    with pytest.raises(KeyError, match="has_implied_quote"):
        comparator.forecasts(*args, progress=lambda *a: None)


def test_execution_recipe_drift_precedes_input_inspection():
    execution = {"fit_recipe": comparator.recipe()}
    execution["fit_recipe"]["minimum_training_rows"] = 1
    with pytest.raises(ValueError, match="recipe differs"):
        comparator.forecasts(None, None, None, None, execution, progress=None)


@pytest.mark.parametrize("code,expected", [("SIZE_LIMIT", "RESOURCE_LIMIT_EXCEEDED"),
                                           ("NOT_REGULAR_FILE", "RESOURCE_UNAVAILABLE")])
def test_bounded_registry_refusal_is_typed(monkeypatch, code, expected):
    from engine.v2.models.frozen_state_io import BoundedStateReadError
    def refused():
        raise BoundedStateReadError(code, "test refused registry")
    monkeypatch.setattr(comparator, "current_recipes", refused)
    with pytest.raises(OpsError) as caught:
        comparator.native_recipe()
    assert caught.value.code == expected


def test_progress_callback_cannot_mutate_later_evidence():
    args = case(1)
    def progress(state, evidence):
        evidence["training_event_ids"].clear()
        evidence["evaluation_event_ids"].clear()
    result = comparator.forecasts(*args, progress=progress)
    assert all(item["training_event_ids"] == ["train0"] for item in result["fit_details"])
    assert all(row["abstention"] == "insufficient_training" for row in result["forecasts"])


def test_missing_native_recipe_refuses_as_drift(monkeypatch):
    monkeypatch.setattr(comparator, "current_recipes", lambda: {})
    with pytest.raises(ValueError, match="recipe changed"):
        comparator.native_recipe()
