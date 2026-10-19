"""Consumed-only child inputs retain ordered fit evidence and the hard wire cap."""
import copy

import pandas as pd
import pytest

from engine.v2.foundation import canonical_json
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.native_registration import magnitude_fitted_execution_document
from experiments import magnitude_search_execution as execution
from experiments.magnitude_fitted import fitted_forecasts
from tests.v2.research.test_magnitude_search_execution import LIMITS
from tests.v2.research.test_magnitude_search_protocol import wire_child


def inputs_and_plan(model="history_ridge"):
    policy = magnitude_fitted_execution_document("2025-02-01", model)
    fold = {"training_candidate_event_ids": ["train-a", "omitted", "train-b"],
            "evaluation_event_ids": ["test-b", "test-a"]}
    plan = {"horizons": [5, 1]}
    features = []
    for index, event in enumerate(("test-a", "train-a", "omitted", "test-b", "train-b", "unused")):
        for horizon in (1, 3, 5):
            values = {name: index + 1 if name == "n_prior" else (index + 1) * (column + horizon) / 11
                      for column, name in enumerate(policy["features"])}
            values[policy["features"][3]] = None
            features.append({"event_id": event, "context_id": "sha256:" + "c" * 64,
                "horizon_sessions": horizon, "history_json": "unused provenance " * 100,
                "event_date": "2024-01-01", "source_session_ceiling": "2023-12-20",
                "unused_feature": 987.0, **values})
    inputs = {"features": features, "training": [
        {"event_id": "train-b", "absolute_move_pct": 8.0, "unconsumed_label_field": "private"},
        {"event_id": "train-a", "absolute_move_pct": 2.0, "unconsumed_label_field": "private"}],
        "preflight": {"not": "child data"}, "stored_inputs_id": "not child data"}
    return inputs, fold, plan, policy


def expected_inputs():
    """Literal row/value oracle, independent of either production projection path."""
    columns = ("mean_prior_move", "mean_prior_abs_move", "ema2_prior_move",
        "ema2_prior_abs_move", "ema4_prior_move", "ema4_prior_abs_move", "ema8_prior_move",
        "ema8_prior_abs_move", "ema12_prior_move", "ema12_prior_abs_move")
    rows = [
        ("test-a", 1, 1, (2, 3, None, 5, 6, 7, 8, 9, 10, 11)),
        ("test-a", 5, 1, (6, 7, None, 9, 10, 11, 12, 13, 14, 15)),
        ("train-a", 1, 2, (4, 6, None, 10, 12, 14, 16, 18, 20, 22)),
        ("train-a", 5, 2, (12, 14, None, 18, 20, 22, 24, 26, 28, 30)),
        ("test-b", 1, 4, (8, 12, None, 20, 24, 28, 32, 36, 40, 44)),
        ("test-b", 5, 4, (24, 28, None, 36, 40, 44, 48, 52, 56, 60)),
        ("train-b", 1, 5, (10, 15, None, 25, 30, 35, 40, 45, 50, 55)),
        ("train-b", 5, 5, (30, 35, None, 45, 50, 55, 60, 65, 70, 75)),
    ]
    return {"features": [{"event_id": event, "context_id": "sha256:" + "c" * 64,
        "horizon_sessions": horizon, "n_prior": count,
        **{name: None if value is None else value / 11
           for name, value in zip(columns, numerators, strict=True)}}
        for event, horizon, count, numerators in rows],
        "training": [{"event_id": "train-b", "absolute_move_pct": 8.0},
                     {"event_id": "train-a", "absolute_move_pct": 2.0}]}


def expected_request(fold, plan, policy):
    return {"inputs": expected_inputs(), "inner": fold,
            "plan": {"horizons": plan["horizons"]}, "execution": policy}


def run(inputs, fold, plan, policy, progress=lambda *args: None):
    return execution.run_fitted_trial(inputs, fold, plan, policy,
                                      execution.execution_limits(LIMITS), progress)


@pytest.mark.parametrize("model", ["history_ridge", "history_hgb"])
def test_projected_payload_keeps_only_consumed_columns_rows_and_original_order(monkeypatch, model):
    inputs, fold, plan, policy = inputs_and_plan(model)
    original = copy.deepcopy(inputs)
    observed = wire_child(monkeypatch, 'print(\'{"event":"succeeded","forecasts":[]}\')')
    assert run(inputs, fold, plan, policy) == {"event": "succeeded", "forecasts": []}
    child, scratch, payload = observed[0]
    assert payload == expected_request(fold, plan, policy)
    assert inputs == original
    assert child.returncode == 0
    assert not scratch.exists()


def test_projection_does_not_read_unconsumed_values():
    inputs, fold, plan, policy = inputs_and_plan()
    for row in inputs["features"]:
        row["history_json"] = object()
        if row["event_id"] in {"omitted", "unused"} or row["horizon_sessions"] == 3:
            for name in policy["features"]:
                del row[name]
    projected = execution._fit_inputs(inputs, fold, plan, policy)
    assert projected == expected_inputs()
    canonical_json(projected)


@pytest.mark.parametrize("model", ["history_ridge", "history_hgb"])
def test_real_child_projection_preserves_exact_forecasts_and_fit_evidence(model):
    inputs, fold, plan, policy = inputs_and_plan(model)
    original = copy.deepcopy(inputs)
    indexed = {(row["event_id"], row["horizon_sessions"]): row for row in inputs["features"]}
    expected_progress = []
    expected, evidence = fitted_forecasts(indexed, pd.DataFrame(inputs["training"]), fold, plan, policy,
        progress=lambda state, fit: expected_progress.append((state, fit)), strict_failures=True)
    progress = []
    result = run(inputs, fold, plan, policy, lambda state, fit: progress.append((state, fit)))
    assert result == {"event": "succeeded", "forecasts": [{"event_id": event, "horizon_sessions": horizon,
        "predicted_absolute_move_pct": value} for (event, horizon), value in sorted(expected.items())]}
    assert progress == expected_progress
    assert [fit for state, fit in progress if state == "completed"] == evidence
    assert [fit["horizon_sessions"] for fit in evidence] == [5, 1]
    assert all(fit["training_event_ids"] == ["train-b", "train-a"] for fit in evidence)
    assert all(fit["evaluation_event_ids"] == ["test-b", "test-a"] for fit in evidence)
    assert inputs == original


@pytest.mark.parametrize("excess", [0, 1], ids=["exact-cap", "cap-plus-one"])
def test_cap_applies_inclusively_to_projected_payload_before_launch(monkeypatch, excess):
    inputs, fold, plan, policy = inputs_and_plan()
    projected = expected_request(fold, plan, policy)
    size = len(canonical_json(projected).encode())
    cap = size - excess
    assert len(canonical_json({**projected, "inputs": inputs}).encode()) > cap
    monkeypatch.setattr(execution, "MAX_BYTES", cap)
    observed = wire_child(monkeypatch, 'print(\'{"event":"succeeded","forecasts":[]}\')')
    if excess:
        with pytest.raises(OpsError) as caught:
            run(inputs, fold, plan, policy)
        assert caught.value.problem == fail(
            "RESOURCE_LIMIT_EXCEEDED", "search execution input exceeds its byte limit").problem
        assert observed == []
    else:
        assert run(inputs, fold, plan, policy) == {"event": "succeeded", "forecasts": []}
        child, scratch, payload = observed[0]
        assert payload == projected
        assert child.returncode == 0
        assert not scratch.exists()
