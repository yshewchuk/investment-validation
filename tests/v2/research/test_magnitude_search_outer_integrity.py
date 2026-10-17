"""No-launch prerequisites and valid-hash negatives for the outer evidence chain."""
# ruff: noqa: F811
import copy
from fractions import Fraction

import numpy as np
import pytest

from engine.v2.foundation import content_hash
from experiments import magnitude_search_outer as outer
from experiments import magnitude_search_trials as trials
from tests.v2.research.magnitude_trial_support import trial_cases, trial_forecasts
from tests.v2.research.test_magnitude_search_execution import LIMITS, inject_child
from tests.v2.research.test_magnitude_search_outer import (
    CONFLICT,
    base_source,  # noqa: F401
    independent_metrics,
    known_targets,
    native_outer_source,  # noqa: F401
    outer_case,  # noqa: F401
    outer_problem,
    run,
    slot,
)
from tests.v2.research.test_magnitude_search_trials import case  # noqa: F401


@pytest.mark.parametrize("horizon, value", [(np.int64(1), 2.5), (1, np.float32(2.5)), (1, Fraction(5, 2))])
def test_shared_forecast_validation_preserves_inner_and_outer_numeric_policies(horizon, value):
    fold = {"evaluation_event_ids": ["synthetic-event"]}
    forecasts = [{"event_id": "synthetic-event", "horizon_sessions": horizon, "predicted_absolute_move_pct": value}]
    normalized = trials.SearchTrials._forecasts(forecasts, fold, [1])
    assert normalized == [{"event_id": "synthetic-event", "horizon_sessions": 1, "predicted_absolute_move_pct": 2.5}]
    assert type(normalized[0]["horizon_sessions"]) is int
    assert type(normalized[0]["predicted_absolute_move_pct"]) is float
    with outer_problem("EXPERIMENT_IDENTITY_CONFLICT", CONFLICT):
        outer._forecasts(forecasts, fold, [1])


@pytest.mark.parametrize("missing", ["selection", "execution", "terminal"])
def test_missing_or_submitted_only_native_evidence_cannot_launch(outer_case, monkeypatch, missing):
    case = outer_case
    original = outer.read_record
    def read(conn, store, run_id, name):
        if name == missing or name.startswith(missing + ":"):
            return None
        return original(conn, store, run_id, name)
    monkeypatch.setattr(outer, "read_record", read)
    monkeypatch.setattr(trials.SearchTrials, "execute", lambda *a, **kw: pytest.fail("attempted inner execute"))
    monkeypatch.setattr(outer, "_inputs", lambda *a: pytest.fail("opened outer inputs"))
    before = case.conn.total_changes
    message = ("outer evaluation requires an existing frozen selection" if missing == "selection" else
               "outer evaluation requires completed native inner trials")
    with outer_problem("EXPERIMENT_IDENTITY_CONFLICT", message):
        run(case)
    assert case.conn.total_changes == before


def test_valid_native_candidate_failure_is_excluded_without_blocking_outer(case, monkeypatch):  # noqa: F811
    original = trials.run_fitted_trial
    def failed(inputs, fold, plan, execution, limits, progress):
        first = trials.SearchTrials._fit_evidence(inputs, fold, plan, execution)[0]
        progress("started", first)
        return {"event": "failed"}
    cases = trial_cases(case)
    monkeypatch.setattr(trials, "run_fitted_trial", failed)
    case.recorder.execute(*cases[0], limits=LIMITS)
    monkeypatch.setattr(trials, "run_fitted_trial", original)
    for args in cases[1:]:
        case.recorder.execute(*args, limits=LIMITS)
    selected = case.recorder.freeze()
    r = case.recorder
    monkeypatch.setattr(trials, "run_fitted_trial", lambda *a: pytest.fail("replayed native failure refitted"))
    result = outer.execute_outer_fold(r.conn, r.store, r.repository, r.registration,
                                     r.request, r.plan_ref, "2024", limits=LIMITS)
    assert result["candidate"]["candidate_id"] == selected["selections"][0]["selected_candidate_id"]
    assert result["candidate"]["candidate_id"] != cases[0][2]


MUTATIONS = ["start_selection", "start_sources", "inputs_features", "inputs_training",
    "fit_started", "fit_completed", "execution_count", "execution_extra", "execution_forecasts",
    "forecasts_value", "forecasts_missing", "forecasts_boolean", "validation_cutoff", "validation_purpose",
    "validation_targets", "terminal_metrics", "terminal_candidate", "terminal_prospective", "terminal_extra"]


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_valid_hash_outer_evidence_tampering_refuses_replay(outer_case, monkeypatch, mutation):
    case = outer_case
    original = outer.append_record
    def append(conn, store, run_id, name, body, **kwargs):
        body = copy.deepcopy(body)
        if name.startswith("outer-start:"):
            if mutation == "start_selection":
                body["selection_id"] = "wrong"
            if mutation == "start_sources":
                body["outer_sources"] = {}
        if name.startswith("outer-inputs:"):
            if mutation == "inputs_features":
                body["features"][0]["n_prior"] += 1
            if mutation == "inputs_training":
                body["training"][0]["absolute_move_pct"] += 1
        if name.startswith("outer-fit-started:") and mutation == "fit_started":
            body["prediction_matrix_id"] = "wrong"
        if name.startswith("outer-fit-completed:") and mutation == "fit_completed":
            return body
        if name.startswith("outer-execution:"):
            if mutation == "execution_count":
                body["completed_fit_count"] = True
            if mutation == "execution_extra":
                body["extra"] = True
            if mutation == "execution_forecasts":
                body["forecasts_id"] = "wrong"
        if name.startswith("outer-forecasts:"):
            if mutation == "forecasts_value":
                body[0]["predicted_absolute_move_pct"] += 1
            if mutation == "forecasts_missing":
                body.pop()
            if mutation == "forecasts_boolean":
                body[0]["horizon_sessions"] = True
        if name.startswith("outer-validation:"):
            if mutation == "validation_cutoff":
                body["evaluation_inputs"]["labels_before"] = "2026-01-01"
            if mutation == "validation_purpose":
                body["evaluation_inputs"]["purpose"] = "final"
            if mutation == "validation_targets":
                body["targets"][0]["absolute_move_pct"] += 1
        if name.startswith("outer-terminal:"):
            if mutation == "terminal_metrics":
                body["metrics"]["rmse"] += 1
            if mutation == "terminal_candidate":
                body["candidate"]["candidate_id"] = "wrong"
            if mutation == "terminal_prospective":
                body["prospective_candidate_id"] = "wrong"
            if mutation == "terminal_extra":
                body["extra"] = True
            body["result_id"] = content_hash({k: v for k, v in body.items() if k != "result_id"})
        return original(conn, store, run_id, name, body, **kwargs)
    monkeypatch.setattr(outer, "append_record", append)
    # Valid content-addressed record bytes are published, not hash-corrupt files.
    with outer_problem("EXPERIMENT_IDENTITY_CONFLICT", CONFLICT):
        run(case)
    assert slot(case, "terminal") is not None
    before = case.conn.total_changes
    monkeypatch.setattr(outer, "run_fitted_trial", lambda *a: pytest.fail("tampered replay fitted"))
    monkeypatch.setattr(outer, "load_magnitude_targets", lambda *a, **kw: pytest.fail("tampered replay opened labels"))
    with outer_problem("EXPERIMENT_IDENTITY_CONFLICT", CONFLICT):
        run(case)
    assert case.conn.total_changes == before


@pytest.mark.parametrize("kind", ["missing", "extra", "nonfinite", "boolean"])
def test_bad_child_forecasts_refuse_before_outer_labels(outer_case, monkeypatch, kind):
    original = outer.run_fitted_trial
    def fitted(*args):
        result = original(*args)
        rows = result["forecasts"]
        if kind == "missing":
            rows.pop()
        elif kind == "extra":
            rows.append(rows[0])
        else:
            rows[0]["predicted_absolute_move_pct"] = float("nan") if kind == "nonfinite" else True
        return result
    monkeypatch.setattr(outer, "run_fitted_trial", fitted)
    monkeypatch.setattr(outer, "load_magnitude_targets", lambda *a, **kw: pytest.fail("invalid forecast opened labels"))
    with outer_problem("EXPERIMENT_IDENTITY_CONFLICT", CONFLICT):
        run(outer_case)
    assert slot(outer_case, "forecasts") is None
    assert slot(outer_case, "terminal") is None


def test_outer_child_receives_only_prior_training_labels(outer_case, monkeypatch):
    case = outer_case
    children = inject_child(monkeypatch, """
def injected(recipe, train, labels, test):
    assert len(labels) == 10
    assert sorted(labels.tolist()) == [2, 2, 3, 3, 3, 3, 4, 6, 8, 12]
    assert len(test) == 8
    return [5.0] * len(test)
""")
    result = run(case)
    assert {row["predicted_absolute_move_pct"] for row in result["predictions"]} == {5.0}
    assert result["metrics"] == {"count": 24, "rmse": pytest.approx((276 / 8) ** 0.5), "mae": 4.0}
    expected = [{"event_id": event, "predicted_absolute_move_pct": 5.0}
                for event in known_targets() if ":2024-" in event]
    assert result["per_horizon"] == [{"horizon_sessions": horizon, **independent_metrics(expected)}
                                      for horizon in [5, 3, 1]]
    assert all(child.poll() is not None and not path.exists() for child, path in children)



def test_actual_submitted_forecast_selection_has_no_outer_execution_authority(case, monkeypatch):
    for args in trial_cases(case):
        case.recorder.start(*args)
        case.recorder.record(*args, forecasts=trial_forecasts(case, args))
    case.recorder.freeze()
    r = case.recorder
    before = r.conn.total_changes
    monkeypatch.setattr(trials.SearchTrials, "execute", lambda *a, **kw: pytest.fail("submitted trial executed"))
    monkeypatch.setattr(outer, "_inputs", lambda *a: pytest.fail("submitted selection opened outer inputs"))
    with outer_problem("EXPERIMENT_IDENTITY_CONFLICT", "outer evaluation requires completed native inner trials"):
        outer.execute_outer_fold(r.conn, r.store, r.repository, r.registration,
                                 r.request, r.plan_ref, "2024", limits=LIMITS)
    assert r.conn.total_changes == before
