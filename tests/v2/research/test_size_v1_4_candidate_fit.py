"""Fixed SIZE fallback policy: independent numeric, refusal and worker oracles."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import size_comparator as comparator
from experiments.magnitude_search_evidence import fit_evidence, forecast_target, normalize_forecasts
from experiments.size_fit_diagnostics import evidence as diagnostic_evidence
from experiments.size_search_binding import _panel_columns


def _row(event, *, prior=4, **updates):
    return {"event_id": event, "horizon_sessions": 1, "context_id": "ctx-" + event,
        "n_prior": prior, "current_quote_status": "absent", "has_implied_quote": 0,
        "mean_prior_abs_move": 3.0, "abs_dist_ema": 1.0, "ema12r_abs": 2.0,
        "mean_prior_move": 1.0, "signed_streak": 1.0, "dist_high": 1.0,
        "dist_ema": 1.0, "spy_vol20": 1.0, "spy_dd252": 1.0,
        "mean_prior_or_implied": 1.0, "implied_move": 5.0, "rvol30": 30.0,
        "mcap_log": 1.0, **updates}


def _execution():
    return {"schema_version": "size_candidate_execution.v1", "features": comparator.COLUMNS,
            "model": "size_v1_4", "fit_recipe": comparator.candidate_recipe(comparator.recipe())}


def _inputs(count=500, evaluation=None):
    training = [{"event_id": f"train-{i}", "absolute_move_pct": 10.0} for i in range(count)]
    evaluation = evaluation or [_row("eval")]
    return {"training": training, "features": [_row(r["event_id"]) for r in training] + evaluation}, {
        "evaluation_event_ids": [r["event_id"] for r in evaluation]}


def _run(inputs, fold, *, execution=None, progress=lambda *args: None):
    return comparator.forecasts({(r["event_id"], 1): r for r in inputs["features"]},
        pd.DataFrame(inputs["training"]), fold, {"horizons": [1]}, execution or _execution(),
        progress=progress, inputs=inputs)


def test_candidate_policy_is_distinct_fixed_and_detached():
    comparison = comparator.recipe()
    recipe = comparator.candidate_recipe(comparison)
    assert recipe == {**comparison, "schema_version": "size_history_mean_candidate.v1",
        "missing": "history_mean", "refit": "fresh_per_fold_horizon", "prediction_domain": "signed"}
    assert recipe["minimum_prior"] == 4 and recipe["minimum_training_rows"] == 500
    assert recipe["masks"] == {"or_implied": [0, 60], "or_rvol30": [0, 700], "training_target": [0, 200]}
    assert recipe["history_mean"] == "stored_mean_else_qualified_training_mean"
    assert recipe["production_reproduction"] is False and recipe["prediction_projection"] == "none"
    assert content_hash(recipe) != content_hash(comparison)
    recipe["component_parameters"]["mlp"]["hidden_layer_sizes"].clear()
    assert comparison["component_parameters"]["mlp"]["hidden_layer_sizes"] == [64, 32]


def test_fitted_candidate_keeps_negative_prediction_and_falls_back_without_changing_population(monkeypatch):
    evaluation = [_row("complete"), _row("low", prior=3), _row("missing", implied_move=None),
                  _row("masked", rvol30=701), _row("zero", prior=0, mean_prior_abs_move=None)]
    inputs, fold = _inputs(evaluation=evaluation)
    # Excluded training rows must neither enter the fit nor disappear from the baseline's admitted mean.
    inputs["training"] += [{"event_id": "target-mask", "absolute_move_pct": 250.0},
                           {"event_id": "history-mask", "absolute_move_pct": 7.0},
                           {"event_id": "input-mask", "absolute_move_pct": 9.0}]
    inputs["features"] += [_row("target-mask"), _row("history-mask", prior=3),
                           _row("input-mask", mcap_log=None)]
    captured = []

    class Model:
        def predict(self, matrix):
            assert matrix.shape == (1, 14)
            return np.array([-2.5])

    def fit(recipe, matrix, labels):
        assert matrix.shape == (500, 14)
        assert labels.tolist() == [10.0] * 500
        assert matrix[0].tolist() == [0, 3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 5, 30, 1]
        return Model()

    monkeypatch.setattr(comparator, "fit_recipe_estimator", fit)
    result = _run(inputs, fold, progress=lambda *pair: captured.append(pair))
    values = {r["event_id"]: r["predicted_absolute_move_pct"] for r in result["forecasts"]}
    assert values == {"complete": -2.5, "low": 3, "missing": 3, "masked": 3,
                      "zero": pytest.approx((5000 + 250 + 7 + 9) / 503)}
    assert all(r["abstention"] is None for r in result["forecasts"])
    detail = result["fit_details"][0]
    assert detail == {"horizon_sessions": 1, "training_event_ids": [f"train-{i}" for i in range(500)],
        "logical_fits": 1, "component_fits": 2, "fallback_counts": {
            "insufficient_history": 2, "predictor_mask:or_rvol30": 1, "unavailable_input": 1}}
    expected = fit_evidence(inputs, fold, {"horizons": [1]}, _execution())[0]
    assert captured[0] == ("started", expected)
    assert captured[1][0] == "completed"
    assert {k: v for k, v in captured[1][1].items() if k != "fit_seconds"} == expected
    assert captured[1][1]["fit_seconds"] >= 0


def test_insufficient_training_falls_back_but_standalone_comparator_still_abstains(monkeypatch):
    inputs, fold = _inputs(499)
    monkeypatch.setattr(comparator, "fit_recipe_estimator", lambda *args: pytest.fail("ineligible fit"))
    result = _run(inputs, fold)
    assert result["forecasts"][0]["predicted_absolute_move_pct"] == 3.0
    assert result["fit_details"][0]["fallback_counts"] == {"insufficient_training": 1}
    assert result["fit_details"][0]["logical_fits"] == result["fit_details"][0]["component_fits"] == 0
    baseline = _run(inputs, fold, execution={"schema_version": "size_comparison_execution.v1",
        "features": comparator.COLUMNS, "fit_recipe": comparator.recipe()})
    assert baseline["forecasts"][0]["predicted_absolute_move_pct"] is None
    assert baseline["forecasts"][0]["abstention"] == "insufficient_training"
    assert "fallback_counts" not in baseline["fit_details"][0]


def test_zero_prior_fallback_exactly_matches_selection_baseline_at_rounding_edge():
    from experiments.magnitude_search_selection import blended_event_score, history_mean_rows

    inputs, fold = _inputs(3, [_row("eval", prior=0, mean_prior_abs_move=None)])
    for row, value in zip(inputs["training"], [0.1, 0.2, 0.3]):
        row["absolute_move_pct"] = value
    actual = 0.19999999999999998  # fsum/n, unlike statistics.mean's 0.2.
    prediction = _run(inputs, fold)["forecasts"][0]["predicted_absolute_move_pct"]
    assert prediction == actual
    history = history_mean_rows(inputs, {"targets": [{"event_id": "eval", "absolute_move_pct": actual}]})
    candidate = [{**history[0], "predicted_absolute_move_pct": prediction}]
    score = blended_event_score(candidate, history, metric_weights={"rmse": 0.6, "mae": 0.3,
        "large_move_underprediction": 0.1}, large_move_threshold_pct=10.0)
    assert score["metrics"]["rmse"] == score["metrics"]["mae"] == 0.0
    assert score["blend_score"] == 1.0  # Equal-to-baseline policy, not a false ZeroComparatorError.
    baseline = _run(inputs, fold, execution={"schema_version": "size_comparison_execution.v1",
        "features": comparator.COLUMNS, "fit_recipe": comparator.recipe()})
    assert baseline["forecasts"][0]["history_mean"] == 0.2  # Standalone comparator unchanged.


@pytest.mark.parametrize("mean", [None, float("nan"), float("inf"), True, "3"])
def test_invalid_history_mean_is_not_a_fallback(mean):
    inputs, fold = _inputs(1, [_row("eval", prior=3, mean_prior_abs_move=mean)])
    with pytest.raises(ValueError, match="history_mean"):
        _run(inputs, fold)


def test_zero_prior_requires_finite_admitted_training_mean():
    inputs, fold = _inputs(1, [_row("eval", prior=0, mean_prior_abs_move=None)])
    inputs["training"][0]["absolute_move_pct"] = float("nan")
    with pytest.raises(ValueError, match="history_mean training mean"):
        _run(inputs, fold)


@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("phase", ["fit", "predict"])
@pytest.mark.parametrize("error", [ValueError, ArithmeticError, OSError, MemoryError, KeyError])
def test_estimator_failures_never_become_fallback(monkeypatch, candidate, phase, error):
    inputs, fold = _inputs()
    class Model:
        def predict(self, matrix):
            raise error("sentinel")
    def fit(*args):
        if phase == "fit":
            raise error("sentinel")
        return Model()
    monkeypatch.setattr(comparator, "fit_recipe_estimator", fit)
    execution = (_execution() if candidate else {"schema_version": "size_comparison_execution.v1",
        "features": comparator.COLUMNS, "fit_recipe": comparator.recipe()})
    if candidate and error in (ValueError, ArithmeticError):
        with pytest.raises(OpsError) as caught:
            _run(inputs, fold, execution=execution)
        assert caught.value.code == "EXPERIMENT_VARIANT_FAILED"
    else:
        with pytest.raises(error):
            _run(inputs, fold, execution=execution)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("candidate", [False, True])
def test_nonfinite_estimator_output_is_never_fallback_or_infrastructure(monkeypatch, value, candidate):
    from types import SimpleNamespace

    inputs, fold = _inputs()
    monkeypatch.setattr(comparator, "fit_recipe_estimator",
        lambda *args: SimpleNamespace(predict=lambda matrix: np.array([value])))
    execution = _execution() if candidate else {"schema_version": "size_comparison_execution.v1",
        "features": comparator.COLUMNS, "fit_recipe": comparator.recipe()}
    if candidate:
        with pytest.raises(OpsError) as caught:
            _run(inputs, fold, execution=execution)
        assert caught.value.code == "EXPERIMENT_VARIANT_FAILED"
        assert caught.value.problem.retryable is False
    else:
        with pytest.raises(ValueError, match="^nonfinite incumbent prediction$"):
            _run(inputs, fold, execution=execution)


def test_complete_grid_policy_accepts_signed_size_but_never_partial_or_none():
    fold = {"evaluation_event_ids": ["a", "b"]}
    rows = [{"event_id": event, "horizon_sessions": 1, "predicted_absolute_move_pct": value}
            for event, value in (("a", -2.5), ("b", 3.0))]
    policy = forecast_target({}, _execution()["fit_recipe"])
    assert normalize_forecasts(rows, fold, [1], target=policy) == rows
    for invalid in (rows[:1], [rows[0], {**rows[1], "predicted_absolute_move_pct": None}]):
        with pytest.raises(ValueError):
            normalize_forecasts(invalid, fold, [1], target=policy)
    with pytest.raises(ValueError):
        normalize_forecasts(rows, fold, [1])
    target = {"prediction_column": "predicted_crush", "domain": "signed"}
    assert forecast_target({"target": target}, {}) == target


def test_diagnostics_reconstruct_exact_counts_and_reject_poison():
    inputs, fold = _inputs(1)
    execution = _execution()
    result = _run(inputs, fold, execution=execution)
    worker = {"event": "succeeded", "diagnostics": [{"schema_version": "size_v1_4_fit.v1",
        "recipe_id": content_hash(execution["fit_recipe"]), **result["fit_details"][0]}]}
    assert diagnostic_evidence(worker, execution, [1], inputs, fold) == {"diagnostics": worker["diagnostics"]}
    for field, poison in (("training_event_ids", []), ("logical_fits", False),
                         ("component_fits", False), ("fallback_counts", {}),
                         ("fallback_counts", {"insufficient_training": True})):
        changed = deepcopy(worker)
        changed["diagnostics"][0][field] = poison
        with pytest.raises(OpsError) as caught:
            diagnostic_evidence(changed, execution, [1], inputs, fold)
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_v5_panel_projection_retains_all_other_family_columns():
    columns = ["ema2_prior_move", "ret5", "iv10"]
    assert _panel_columns("magnitude_search_request.v5", columns) == list(dict.fromkeys(
        c for c in comparator.COLUMNS + columns if c != "current_quote_status"))
    assert _panel_columns("magnitude_search_request.v4", columns) == columns


@pytest.mark.parametrize("fails", [False, True])
def test_worker_uses_shared_timing_failure_envelope_and_serializes_fallbacks(tmp_path, monkeypatch, capsys, fails):
    import json
    import math
    import sys
    from types import SimpleNamespace

    from engine.v2.ops.errors import fail
    from experiments import magnitude_search_worker as worker
    execution = _execution()
    request = {"store_root": "/synthetic", "manifest": {}, "inputs_id": "inputs",
        "inner": {"evaluation_event_ids": ["event"]}, "plan": {"horizons": [1]}, "execution": execution}
    path = tmp_path / "worker-request.json"
    path.write_text(json.dumps(request))
    limits = {"memory_bytes": 1024 ** 2, "cpu_seconds": 30, "cpu_id": 0}
    monkeypatch.setattr(sys, "argv", ["worker", str(path), json.dumps(limits)])
    monkeypatch.setattr(worker.os, "sched_setaffinity", lambda *args: None)
    monkeypatch.setattr(worker.resource, "setrlimit", lambda *args: None)
    monkeypatch.setattr(worker, "start_rss_watchdog", lambda *args: None)
    rss, usage = iter((10240, 20480)), iter((10, 15))
    monkeypatch.setattr(worker, "rss_bytes", lambda: next(rss))
    monkeypatch.setattr(worker.resource, "getrusage", lambda *args: SimpleNamespace(ru_maxrss=next(usage)))
    monkeypatch.setattr("engine.v2.foundation.ArtifactStore", lambda *args: object())
    monkeypatch.setattr("experiments.magnitude_search_inputs.load_fit_inputs",
                        lambda *args: {"features": [], "training": []})
    detail = {"horizon_sessions": 1, "training_event_ids": [], "logical_fits": 0,
              "component_fits": 0, "fallback_counts": {"insufficient_history": 1}}
    def forecasts(*args, **kwargs):
        assert set(kwargs["inputs"]) == {"features", "training"}
        if fails:
            raise fail("EXPERIMENT_VARIANT_FAILED", "sentinel")
        return {"event": "succeeded", "forecasts": [{"event_id": "event", "horizon_sessions": 1,
            "predicted_absolute_move_pct": 3.0}], "fit_details": [detail]}
    monkeypatch.setattr(comparator, "forecasts", forecasts)
    worker.main()
    result = json.loads(capsys.readouterr().out)
    assert result["event"] == ("failed" if fails else "succeeded")
    assert type(result["score_seconds"]) is float and math.isfinite(result["score_seconds"]) and result["score_seconds"] >= 0
    assert result["rss_delta_kib"] == 10
    if not fails:
        assert result["forecasts"] == [{"event_id": "event", "horizon_sessions": 1, "predicted_absolute_move_pct": 3.0}]
        assert result["diagnostics"] == [{"schema_version": "size_v1_4_fit.v1",
            "recipe_id": content_hash(execution["fit_recipe"]), **detail}]
    else:
        assert set(result) == {"event", "score_seconds", "rss_delta_kib"}


@pytest.mark.parametrize("changes", [
    {"current_quote_status": "invalid"}, {"has_implied_quote": 1},
    {"n_prior": -1}, {"n_prior": True}, {"n_prior": 3.0},
])
@pytest.mark.parametrize("candidate", [False, True])
def test_target_mask_does_not_skip_existing_metadata_refusal(candidate, changes):
    inputs, fold = _inputs(1)
    inputs["training"][0]["absolute_move_pct"] = 250.0
    inputs["features"][0].update(changes)
    execution = _execution() if candidate else {"schema_version": "size_comparison_execution.v1",
        "features": comparator.COLUMNS, "fit_recipe": comparator.recipe()}
    with pytest.raises(ValueError, match="^invalid admitted quote/history metadata$"):
        _run(inputs, fold, execution=execution)
