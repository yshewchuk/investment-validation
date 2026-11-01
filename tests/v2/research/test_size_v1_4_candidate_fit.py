"""SIZE v1.4 candidate fits use only admitted fold training rows and count abstentions."""
import numpy as np
import pandas as pd
import pytest
from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments.magnitude_search_trials import _verify_size_forecast_replay
from experiments.size_fit_diagnostics import evidence as diagnostic_evidence
from experiments.size_search_binding import _panel_columns

from experiments import size_comparator as comparator
from experiments.magnitude_search_evidence import fit_evidence as shared_fit_evidence
from experiments.magnitude_search_evidence import normalize_forecasts


def _row(event_id, *, n_prior=4, missing=False):
    return {
        "event_id": event_id, "horizon_sessions": 5, "context_id": "ctx-" + event_id,
        "n_prior": n_prior, "current_quote_status": "absent", "has_implied_quote": 0,
        "mean_prior_abs_move": 1.0, "abs_dist_ema": 1.0, "ema12r_abs": 1.0,
        "mean_prior_move": 1.0, "signed_streak": 1.0, "dist_high": 1.0,
        "dist_ema": 1.0, "spy_vol20": 1.0, "spy_dd252": 1.0,
        "mean_prior_or_implied": 1.0, "implied_move": None if missing else 5.0,
        "rvol30": 30.0, "mcap_log": 1.0,
    }


def test_inner_fold_fit_uses_only_pre_cutoff_training_rows_and_counts_abstentions(monkeypatch):
    training_ids = [f"train-{index}" for index in range(500)]
    evaluation_ids = ["low-prior", "missing-input", "complete"]
    features = [_row(event_id) for event_id in training_ids]
    features.extend([_row("low-prior", n_prior=3), _row("missing-input", missing=True), _row("complete")])
    train = pd.DataFrame({"event_id": training_ids, "absolute_move_pct": [10.0] * 500,
                          "label_end": ["2024-12-31"] * 500})
    fold = {"evaluation_event_ids": evaluation_ids,
            "training_labels_before": "2025-01-01"}
    seen = {}

    class Predictor:
        def predict(self, matrix):
            return np.full(len(matrix), 12.0)

    def fit_spy(recipe, train_features, train_labels):
        seen["labels"] = np.asarray(train_labels).copy()
        seen["matrix"] = np.asarray(train_features).copy()
        return Predictor()

    monkeypatch.setattr(comparator, "fit_recipe_estimator", fit_spy)
    assert all(label_end < fold["training_labels_before"] for label_end in train["label_end"])
    plan = {"horizons": [5]}
    execution = {"schema_version": "size_candidate_execution.v1", "features": comparator.COLUMNS,
                 "model": "size_v1_4", "fit_recipe": comparator.recipe()}
    inputs = {"training": train.to_dict("records"), "features": features}
    expected = shared_fit_evidence(inputs, fold, plan, execution)
    captured = []
    result = comparator.forecasts(
        {(row["event_id"], 5): row for row in features}, train, fold,
        plan, execution, progress=lambda *pair: captured.append(pair))

    assert captured == [("started", expected[0]), ("completed", expected[0])]
    assert seen["labels"].tolist() == [10.0] * 500
    assert seen["matrix"].shape == (500, len(comparator.FEATURES))
    assert result["fit_details"][0]["training_event_ids"] == training_ids
    assert result["fit_details"][0]["training_abstention_counts"] == {}
    assert result["fit_details"][0]["abstention_counts"] == {
        "insufficient_history": 1, "unavailable_input": 1}
    forecasts = {row["event_id"]: row for row in result["forecasts"]}
    assert forecasts["low-prior"]["predicted_absolute_move_pct"] is None
    assert forecasts["low-prior"]["abstention"] == "insufficient_history"
    assert forecasts["missing-input"]["predicted_absolute_move_pct"] is None
    assert forecasts["missing-input"]["abstention"] == "unavailable_input"
    assert forecasts["complete"]["predicted_absolute_move_pct"] == 12.0


def test_normalize_forecasts_allows_incomplete_nonempty_subset_but_strict_path_requires_complete():
    fold = {"evaluation_event_ids": ["a", "b"]}
    rows = [
        {"event_id": "a", "horizon_sessions": 5, "predicted_absolute_move_pct": -2.5},
        {"event_id": "b", "horizon_sessions": 5, "predicted_absolute_move_pct": None},
    ]

    assert normalize_forecasts(rows, fold, [5], allow_incomplete=True) == [
        {"event_id": "a", "horizon_sessions": 5, "predicted_absolute_move_pct": -2.5}]

    with pytest.raises(ValueError):
        normalize_forecasts(rows, fold, [5])

    with pytest.raises(ValueError):
        normalize_forecasts(rows[:1], fold, [5], allow_incomplete=True)


def test_insufficient_training_abstentions_are_counted():
    training_ids = [f"train-{index}" for index in range(499)]
    features = [_row(event_id) for event_id in training_ids + ["held-out"]]
    train = pd.DataFrame({"event_id": training_ids, "absolute_move_pct": [10.0] * 499})
    fold = {"evaluation_event_ids": ["held-out"],
            "training_labels_before": "2025-01-01"}
    plan = {"horizons": [5]}
    execution = {"schema_version": "size_candidate_execution.v1", "features": comparator.COLUMNS,
                 "model": "size_v1_4", "fit_recipe": comparator.recipe()}
    result = comparator.forecasts(
        {(row["event_id"], 5): row for row in features}, train, fold,
        plan, execution, progress=lambda *pair: None)

    assert result["fit_details"][0]["abstention_counts"] == {"insufficient_training": 1}
    forecasts = {row["event_id"]: row for row in result["forecasts"]}
    assert forecasts["held-out"]["predicted_absolute_move_pct"] is None
    assert forecasts["held-out"]["abstention"] == "insufficient_training"


@pytest.mark.parametrize("fails", [False, True])
def test_size_candidate_worker_emits_timing_and_maps_fit_failure(tmp_path, monkeypatch, capsys, fails):
    import json
    import math
    import sys
    from types import SimpleNamespace

    from engine.v2.ops.errors import fail
    from experiments import magnitude_search_worker as worker

    request = {"store_root": "/synthetic", "manifest": {}, "inputs_id": "inputs",
        "inner": {"evaluation_event_ids": ["event"]}, "plan": {"horizons": [1]},
        "execution": {"schema_version": "size_candidate_execution.v1", "features": [],
                      "fit_recipe": {"schema_version": "size_causal_comparator.v1"}}}
    path = tmp_path / "worker-request.json"
    path.write_text(json.dumps(request))
    limits = {"memory_bytes": 1024 ** 2, "cpu_seconds": 30, "cpu_id": 0}
    monkeypatch.setattr(sys, "argv", ["worker", str(path), json.dumps(limits)])
    monkeypatch.setattr(worker.os, "sched_setaffinity", lambda *args: None)
    monkeypatch.setattr(worker.resource, "setrlimit", lambda *args: None)
    monkeypatch.setattr(worker, "start_rss_watchdog", lambda *args: None)
    rss = iter((10240, 20480))
    monkeypatch.setattr(worker, "rss_bytes", lambda: next(rss))
    usage = iter((10, 15))
    monkeypatch.setattr(worker.resource, "getrusage",
                        lambda *args: SimpleNamespace(ru_maxrss=next(usage)))
    monkeypatch.setattr("engine.v2.foundation.ArtifactStore", lambda *args: object())
    monkeypatch.setattr("experiments.magnitude_search_inputs.load_fit_inputs",
                        lambda *args: {"features": [], "training": []})

    def forecasts(*args, **kwargs):
        if fails:
            raise fail("EXPERIMENT_VARIANT_FAILED", "synthetic candidate fit failure")
        return {"event": "succeeded", "forecasts": [{"event_id": "event", "horizon_sessions": 1,
            "predicted_absolute_move_pct": 1.0}], "fit_details": [{"horizon_sessions": 1,
            "training_event_ids": [], "logical_fits": 0, "component_fits": 0,
            "abstention_counts": {}, "training_abstention_counts": {}}]}

    monkeypatch.setattr("experiments.size_comparator.forecasts", forecasts)
    worker.main()
    result = json.loads(capsys.readouterr().out)
    assert result["event"] == ("failed" if fails else "succeeded")
    assert type(result["score_seconds"]) is float and math.isfinite(result["score_seconds"])
    assert result["score_seconds"] >= 0
    assert type(result["rss_delta_kib"]) is int and result["rss_delta_kib"] == 10
    assert set(result) == ({"event", "score_seconds", "rss_delta_kib"} if fails else
        {"event", "forecasts", "diagnostics", "score_seconds", "rss_delta_kib"})


def _size_execution():
    return {"schema_version": "size_candidate_execution.v1", "features": comparator.COLUMNS,
            "model": "size_v1_4", "fit_recipe": comparator.recipe()}


def test_size_terminal_replay_validates_raw_grid_and_rejects_empty_storage():
    from pytest import raises
    fold, horizons = {"evaluation_event_ids": ["a", "b"]}, [5]
    raw = [{"event_id": "a", "horizon_sessions": 5, "predicted_absolute_move_pct": -2.5},
           {"event_id": "b", "horizon_sessions": 5, "predicted_absolute_move_pct": None}]
    stored = [{"event_id": "a", "horizon_sessions": 5, "predicted_absolute_move_pct": -2.5}]
    assert _verify_size_forecast_replay(raw, stored, fold, horizons) == stored
    with raises(ValueError):
        _verify_size_forecast_replay(raw, [], fold, horizons)
    with raises(ValueError):
        _verify_size_forecast_replay(raw[:1], stored, fold, horizons)
    with raises(ValueError):
        _verify_size_forecast_replay([{**row, "predicted_absolute_move_pct": None} for row in raw], [], fold, horizons)


def test_size_diagnostics_are_reconstructed_from_admitted_rows():
    training = pd.DataFrame([{"event_id": "train", "absolute_move_pct": 10.0}])
    features, fold, plan = [_row("train"), _row("eval")], {"evaluation_event_ids": ["eval"]}, {"horizons": [5]}
    execution = _size_execution()
    indexed = {(row["event_id"], 5): row for row in features}
    result = comparator.forecasts(indexed, training, fold, plan, execution, progress=lambda *args: None)
    worker = {"event": "succeeded", "diagnostics": [{"schema_version": "size_v1_4_fit.v1",
        "recipe_id": content_hash(execution["fit_recipe"]), **result["fit_details"][0]}]}
    assert worker["diagnostics"][0]["abstention_counts"] == {"insufficient_training": 1}
    inputs = {"features": features, "training": training.to_dict("records")}
    assert diagnostic_evidence(worker, execution, [5], inputs=inputs, fold=fold) == {"diagnostics": worker["diagnostics"]}
    for field, poisoned in (("training_event_ids", ["invented"]), ("abstention_counts", {})):
        corrupted = {"event": "succeeded", "diagnostics": [dict(worker["diagnostics"][0])]}
        corrupted["diagnostics"][0][field] = poisoned
        with pytest.raises(OpsError):
            diagnostic_evidence(corrupted, execution, [5], inputs=inputs, fold=fold)


@pytest.mark.parametrize("stage,error_type", [("fit", ValueError), ("fit", ArithmeticError),
                                                ("predict", ValueError), ("predict", ArithmeticError)])
def test_size_candidate_estimator_errors_are_candidate_failures(monkeypatch, stage, error_type):
    ids = [f"train-{i}" for i in range(500)]
    training = pd.DataFrame({"event_id": ids, "absolute_move_pct": [10.0] * 500})
    features = [_row(event) for event in ids + ["eval"]]
    fold, plan, execution = {"evaluation_event_ids": ["eval"]}, {"horizons": [5]}, _size_execution()
    if stage == "fit":
        def failing_fit(*args, **kwargs):
            raise error_type()
        monkeypatch.setattr(comparator, "fit_recipe_estimator", failing_fit)
    else:
        class Predictor:
            def predict(self, matrix):
                raise error_type()
        monkeypatch.setattr(comparator, "fit_recipe_estimator", lambda *args, **kwargs: Predictor())
    with pytest.raises(OpsError) as error:
        comparator.forecasts({(row["event_id"], 5): row for row in features}, training, fold, plan,
                             execution, progress=lambda *args: None)
    assert error.value.code == "EXPERIMENT_VARIANT_FAILED"


def test_v5_panel_projection_includes_both_candidate_families():
    model_columns = ["model_feature", comparator.COLUMNS[0], "model_feature"]
    projected = _panel_columns("magnitude_search_request.v5", model_columns)
    assert len(projected) == len(set(projected))
    assert set(projected) == set(comparator.COLUMNS) | {"model_feature"}
    assert _panel_columns("magnitude_search_request.v4", model_columns) is model_columns
