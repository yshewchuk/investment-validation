"""Independent fourteen-column/mask/refit and native numerical comparator oracles."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from experiments import size_comparator as comparator
from experiments.magnitude_search_evidence import fit_evidence


def row(event="test", horizon=1, **changes):
    value = dict.fromkeys(comparator.COLUMNS, 1.0)
    value.update(event_id=event, context_id="context", horizon_sessions=horizon, n_prior=5,
                 current_quote_status="present", has_implied_quote=1, implied_move=10.0, rvol30=20.0)
    return {**value, **changes}


def case(n=500):
    indexed = {(f"train{i}", h): row(f"train{i}", h) for i in range(n) for h in (5, 3, 1)}
    indexed.update({("test", h): row("test", h) for h in (5, 3, 1)})
    train = pd.DataFrame({"event_id": [f"train{i}" for i in range(n)], "absolute_move_pct": [3.0] * n})
    fold = {"evaluation_event_ids": ["test"]}
    plan = {"horizons": [5, 3, 1]}
    execution = {"features": comparator.COLUMNS, "fit_recipe": comparator.recipe()}
    return indexed, train, fold, plan, execution


def test_exact_active_columns_and_native_policy():
    expected = ["has_implied_quote", "mean_prior_abs_move", "abs_dist_ema", "ema12r_abs",
        "mean_prior_move", "signed_streak", "dist_high", "dist_ema", "spy_vol20", "spy_dd252",
        "mean_prior_or_implied", "or_implied", "or_rvol30", "mcap_log"]
    assert comparator.FEATURES == expected
    recipe = comparator.recipe()
    assert recipe["native"]["estimator"]["seeds"] == [20260829]
    assert recipe["native"]["estimator"]["params"]["mlp"] == {
        "hidden_layer_sizes": [64, 32], "max_iter": 400, "early_stopping": True, "n_iter_no_change": 15}
    assert recipe["prediction_projection"] == "none"
    assert recipe["minimum_training_rows"] == 500
    assert recipe["production_reproduction"] is False


@pytest.mark.parametrize("changes,reason", [({"n_prior": 3}, "insufficient_history"),
    ({"mcap_log": None}, "unavailable_input"), ({"implied_move": 61}, "predictor_mask:or_implied"),
    ({"rvol30": 701}, "predictor_mask:or_rvol30"),
    ({"current_quote_status": "absent", "has_implied_quote": 0, "implied_move": None}, "unavailable_input")])
def test_abstention_masks_without_substitution(changes, reason):
    assert comparator._values(row(**changes)) == (None, reason)


@pytest.mark.parametrize("status", ["zero", "absent", "unavailable", "present"])
def test_missing_quote_never_reconstructed_from_status(status):
    assert comparator._values(row(current_quote_status=status,
        has_implied_quote=int(status == "present"), implied_move=None)) == (None, "unavailable_input")


def test_observed_numeric_zero_is_distinct_from_missing_native_predictor():
    values, reason = comparator._values(row(current_quote_status="zero", has_implied_quote=0, implied_move=0.))
    assert reason is None and values[0] == 0 and values[11] == 0
    # This array-level zero is admitted by the native bounds, not synthesized from lineage.
    assert comparator.native_recipe().features[11] == "or_implied"
    with pytest.raises(ValueError):
        comparator._values(row(current_quote_status="zero", has_implied_quote=1, implied_move=None))


def test_complete_input_training_masks_and_raw_negative_predictions(monkeypatch):
    indexed, train, fold, plan, execution = case(503)
    train.loc[0, "absolute_move_pct"] = 201
    for h in plan["horizons"]:
        indexed["train1", h]["n_prior"] = 3
        indexed["train2", h]["mcap_log"] = None
    calls = []
    def fit(recipe, X, y):
        calls.append((X.copy(), y.copy()))
        return SimpleNamespace(predict=lambda X: np.full(len(X), -2.0))
    monkeypatch.setattr(comparator, "fit_recipe_estimator", fit)
    progress = []
    original = deepcopy(indexed)
    result = comparator.forecasts(indexed, train, fold, plan, execution,
                                  progress=lambda state, fit: progress.append((state, fit)))
    assert len(calls) == 3
    assert all(X.shape == (500, 14) and np.array_equal(y, np.full(500, 3.0)) for X, y in calls)
    assert all(r["predicted_absolute_move_pct"] == -2.0 for r in result["forecasts"])
    assert all(r["component_fits"] == 2 and r["logical_fits"] == 1 for r in result["fit_details"])
    inputs = {"features": list(indexed.values()), "training": train.to_dict("records")}
    expected = fit_evidence(inputs, fold, plan, execution)
    assert progress == [(state, fit) for fit in expected for state in ("started", "completed")]
    assert indexed == original


def test_499_rows_abstains_without_constructor_and_history_mean_keeps_coverage(monkeypatch):
    args = case(499)
    for h in args[3]["horizons"]:
        args[0]["test", h]["n_prior"] = 0
        args[0]["test", h]["mean_prior_abs_move"] = None
    monkeypatch.setattr(comparator, "fit_recipe_estimator", lambda *a: pytest.fail("unexpected fit"))
    result = comparator.forecasts(*args, progress=lambda *a: None)
    assert all(r["predicted_absolute_move_pct"] is None and r["history_mean"] == 3 for r in result["forecasts"])
    assert all(r["component_fits"] == 0 for r in result["fit_details"])


def test_native_constructor_matches_legacy_math_on_synthetic_arrays():
    from engine.models.training.size_model import fit
    from engine.v2.models.training.estimators import fit_recipe_estimator
    rng = np.random.default_rng(11)
    X, y, test = rng.normal(size=(500, 14)), rng.uniform(0, 20, 500), rng.normal(size=(12, 14))
    with threadpool_limits(limits=1):
        expected = fit(X, y, seed=20260829).predict(test)
        actual = fit_recipe_estimator(comparator.native_recipe(), X, y).predict(test)
    np.testing.assert_array_equal(actual, expected)


def test_recipe_drift_refused_before_constructor(monkeypatch):
    args = case(1)
    args[-1]["fit_recipe"]["minimum_training_rows"] = 1
    monkeypatch.setattr(comparator, "fit_recipe_estimator", lambda *a: pytest.fail("unexpected fit"))
    with pytest.raises(ValueError, match="differs"):
        comparator.forecasts(*args, progress=lambda *a: None)


def test_positive_native_comparator_refit_on_synthetic_admitted_fold(monkeypatch):
    """Exercise the comparator itself with real native fits, not a substituted model."""
    indexed, train, fold, plan, execution = case(500)
    fold.update(year="2024", training_labels_before="2024-01-01", evaluation_labels_before="2025-01-01")
    rng = np.random.default_rng(44)
    train["absolute_move_pct"] = rng.uniform(1, 15, 500)
    for index, event in enumerate(train.event_id):
        for h in plan["horizons"]:
            indexed[event, h].update(event_date="2023-06-01", source_session_ceiling="2023-05-20",
                mean_prior_abs_move=float(train.absolute_move_pct.iloc[index]),
                mean_prior_move=float(rng.normal()), dist_ema=float(rng.normal()))
    for h in plan["horizons"]:
        indexed["test", h].update(event_date="2024-06-01", source_session_ceiling="2024-05-20")
    from engine.v2.models.training.estimators import fit_recipe_estimator
    calls = []
    def native_fit(recipe, X, y):
        calls.append((X.shape, recipe.estimator.seeds))
        return fit_recipe_estimator(recipe, X, y)
    monkeypatch.setattr(comparator, "fit_recipe_estimator", native_fit)
    before = deepcopy(indexed)
    first = comparator.forecasts(indexed, train, fold, plan, execution, progress=lambda *a: None)
    assert calls == [((500, 14), (20260829,))] * 3
    assert all(r["abstention"] is None and np.isfinite(r["predicted_absolute_move_pct"]) for r in first["forecasts"])
    assert all(r["logical_fits"] == 1 and r["component_fits"] == 2 for r in first["fit_details"])
    assert all(indexed[event, 1]["event_date"] < fold["training_labels_before"] for event in train.event_id)
    assert indexed == before
    assert comparator.forecasts(indexed, train, fold, plan, execution, progress=lambda *a: None) == first


def test_source_zero_normalization_drops_training_and_reports_abstention(monkeypatch):
    from engine.v2.research.magnitude_panel_daily import _DAILY, _daily
    from experiments.size_paired_report import paired_report
    source = dict.fromkeys(_DAILY, 1.)
    source.update(ticker="SYNTH", date="2024-06-03", mcap_asof="2024-06-03", implied_move=0.)
    market = _daily([source], [source["date"]], "2024-06-03")
    assert market["quote_status"] == "zero" and market["values"]["implied_move"] is None
    args = case(500)
    for horizon in args[3]["horizons"]:
        for event in ("train0", "test"):
            args[0][event, horizon].update(implied_move=market["values"]["implied_move"],
                current_quote_status=market["quote_status"], has_implied_quote=0)
    monkeypatch.setattr(comparator, "fit_recipe_estimator", lambda *a: pytest.fail("499 complete rows cannot fit"))
    result = comparator.forecasts(*args, progress=lambda *a: None)
    assert all(len(f["training_event_ids"]) == 499 and f["component_fits"] == 0 for f in result["fit_details"])
    assert all(f["abstention"] == "unavailable_input" for f in result["forecasts"])
    challenger = [dict(event_id="test", horizon_sessions=h, predicted_absolute_move_pct=2.) for h in (1, 3, 5)]
    report = paired_report([dict(event_id="test", ticker="SYNTH", event_date="2024-06-04")],
                           challenger, result["forecasts"], {"test": 3.})
    assert report["paired_events"] == report["forecast_matched_events"] == 0
    assert report["broad_coverage_metrics"]["challenger"]["count"] == 3
    assert report["comparison_status"] == "unavailable_no_complete_matched_events"


def test_returned_recipe_features_are_detached_and_preserve_identity():
    from engine.v2.foundation import canonical_json
    from experiments import size_comparator
    original = list(size_comparator.FEATURES)
    result = size_comparator.recipe()
    expected = canonical_json(result)
    try:
        result["features"].append("caller-change")
        fresh = size_comparator.recipe()
        assert size_comparator.FEATURES == original
        assert fresh["features"] == original
        assert fresh["features"] is not size_comparator.FEATURES
        assert canonical_json(fresh) == expected
    finally:
        size_comparator.FEATURES[:] = original
