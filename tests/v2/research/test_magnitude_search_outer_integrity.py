"""No-launch prerequisites and valid-hash negatives for the outer evidence chain."""
# ruff: noqa: F811
import copy
import sqlite3
from fractions import Fraction
from unittest.mock import Mock

import numpy as np
import pytest

from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.ops.migrations import Migration, migrate
from experiments import magnitude_search_outer as outer
from experiments import magnitude_search_records as records
from experiments import magnitude_search_trials as trials
from tests.v2.models.training.test_magnitude_estimators import FEATURES, RIDGE_PARAMS
from tests.v2.research.magnitude_trial_support import history_event, trial_cases, trial_forecasts
from tests.v2.research.test_magnitude_search_execution import LIMITS, inject_child
from tests.v2.research.test_magnitude_search_outer import (
    CONFLICT,
    TARGET_RECIPE,
    base_source,  # noqa: F401
    expected_fits,
    independent_metrics,
    journal_slots,
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


@pytest.mark.parametrize("value, json_native", [
    (2, True), (2.5, True), (np.int64(2), False), (np.float32(2.5), False), (Fraction(5, 2), False)])
def test_shared_target_validation_preserves_inner_and_outer_numeric_policies(value, json_native):
    common = {"snapshot_id": "snapshot", "holdout_as_of_month": "2026-01",
        "random_membership_version": "canonical-event-sha256.v1", "rolling_membership_version": "calendar-months.v1",
        "population_use": "post-release selection", "purpose": "selection", "computed_moves_version": "moves"}
    physical = [{"event_id": "event", "fragment_id": "fragment", "object_id": "object", "content_hash": "hash"}]
    cohort = {"preflight": {**common, "planned_inputs": physical}}
    identity = {"schema_version": "research_magnitude_targets.v1", **common,
        "labels_before": "2025-02-01", "target_recipe": TARGET_RECIPE, "target_recipe_id": content_hash(TARGET_RECIPE),
        "consumed_inputs": physical, "excluded_future_events": []}
    validation = {"evaluation_inputs": {**identity, "inputs_id": content_hash(identity), "holdout_exclusions": []},
                  "targets": [{"event_id": "event", "absolute_move_pct": value}]}
    fold = {"evaluation_event_ids": ["event"], "evaluation_labels_before": "2025-02-01"}
    admitted = {"inputs_id": content_hash(cohort)}
    trials.SearchTrials._validation(fold, admitted, cohort, {**validation, **admitted})
    if json_native:
        outer._validate_targets(cohort, validation, fold)
    else:
        with outer_problem("EXPERIMENT_IDENTITY_CONFLICT", CONFLICT):
            outer._validate_targets(cohort, validation, fold)


@pytest.mark.parametrize("operation", ["execute_outer_fold", "collect_outer_results"])
@pytest.mark.parametrize("schema", ["missing", "older", "newer", "checksum"])
def test_outer_migrations_refuse_before_plan_evidence_or_native_access(monkeypatch, operation, schema):
    conn = sqlite3.connect(":memory:", isolation_level=None)
    try:
        if schema != "missing":
            conn.execute("CREATE TABLE experiment_runs (run_id TEXT PRIMARY KEY)")
            migrations = records.MIGRATIONS + ((Migration(version=2, name="future", statements=()),)
                                               if schema == "newer" else ())
            migrate(conn, records.OWNER, migrations, clock=SystemClock())
        if schema == "older":
            conn.execute("DELETE FROM schema_versions WHERE owner='magnitude_search'")
        elif schema == "checksum":
            conn.execute("UPDATE schema_versions SET checksum='different' WHERE owner='magnitude_search'")
        forbidden = Mock(side_effect=AssertionError("opened plan, evidence or native inputs before migration admission"))
        for name in ("read_search_plan", "read_record", "_inputs", "load_magnitude_targets", "run_fitted_trial"):
            monkeypatch.setattr(outer, name, forbidden)
        monkeypatch.setattr(ArtifactStore, "read_verified", forbidden)
        message, details = "magnitude_search schema is missing migrations", {
            "reason": "schema_older", "owner": "magnitude_search", "missing": [1]}
        if schema == "newer":
            message, details = "magnitude_search schema is newer than this code supports", {
                "reason": "schema_newer", "owner": "magnitude_search", "applied": 2, "supported": 1}
        elif schema == "checksum":
            message, details = "magnitude_search migration differs from the one applied", {
                "reason": "checksum_mismatch", "owner": "magnitude_search", "version": 1}
        before = list(conn.iterdump())
        with outer_problem("INTEGRITY_FAILED", message, category="integrity", details=details):
            args = [conn, None, None, None, None, None]
            if operation == "execute_outer_fold":
                outer.execute_outer_fold(*args, "2024", limits=LIMITS)
            else:
                outer.collect_outer_results(*args)
        forbidden.assert_not_called()
        assert list(conn.iterdump()) == before
    finally:
        conn.close()


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
    recipe = {"schema_version": "magnitude_fitted_recipe.v1", "model": "history_ridge", "features": FEATURES,
        "estimator": {"class": "Ridge", "params": {**RIDGE_PARAMS, "alpha": 0.5}},
        "preprocessing": {"imputer": {"strategy": "median", "keep_empty_features": True},
            "missing_indicator": {"features": "all", "sparse": False},
            "scaler": {"with_mean": True, "with_std": True}, "output_features": 22},
        "target_transform": "none", "prediction_projection": "clip_zero",
        "fit_scope": "fold_and_horizon", "numerical_threads": 1}
    winner = "sha256:121c0427eab3f4ca0034a44991aaa8147cc67385cc66dfda27f0a55be5084d91"
    assert case.selected["selections"][0]["selected_candidate_id"] == winner
    assert recipe != case.recorder.registration.document["magnitude_execution"]["fit_recipe"]
    assert all(recipe != candidate["recipe"] for candidate in case.plan["candidates"]
               if candidate["candidate_id"] != winner)
    children = inject_child(monkeypatch, f"""
count = 0
def injected(recipe, train, labels, test):
    global count
    count += 1
    assert recipe == {recipe!r}
    assert len(labels) == 10
    assert sorted(labels.tolist()) == [2, 2, 3, 3, 3, 3, 4, 6, 8, 12]
    assert len(test) == 8
    return [10 * recipe['estimator']['params']['alpha'] + count + index for index in range(len(test))]
""")
    result = run(case)
    events = sorted([history_event(day, ticker=ticker) for day in
        ("2024-06-12", "2024-06-17", "2024-12-27", "2024-12-30") for ticker in ("SYNTH_A", "SYNTH_B")],
        key=lambda event: (event["event_date"], event["event_id"]))
    expected = sorted([{"event_id": event["event_id"], "horizon_sessions": horizon,
        "predicted_absolute_move_pct": 5.0 + {5: 1, 3: 2, 1: 3}[horizon] + index,
        "absolute_move_pct": known_targets()[event["event_id"]]}
        for index, event in enumerate(events) for horizon in (5, 3, 1)],
        key=lambda row: (row["event_id"], row["horizon_sessions"]))
    assert result["predictions"] == expected
    assert result["metrics"] == independent_metrics(expected)
    assert result["per_horizon"] == [{"horizon_sessions": horizon, **independent_metrics([
        row for row in expected if row["horizon_sessions"] == horizon])}
                                      for horizon in [5, 3, 1]]
    assert len(children) == 1
    assert all(child.poll() is not None and not path.exists() for child, path in children)


def assert_live_child_refusal(case, started, completed):
    names = ["outer-start", "outer-inputs"]
    for index, fit in enumerate(expected_fits(case)):
        assert slot(case, "fit-started:" + str(fit["horizon_sessions"])) == (fit if index < started else None)
        assert slot(case, "fit-completed:" + str(fit["horizon_sessions"])) == (fit if index < completed else None)
        for state, count in (("started", started), ("completed", completed)):
            if index < count:
                names.append("outer-fit-" + state + ":" + str(fit["horizon_sessions"]))
    suffix = ":" + content_hash([case.plan["plan_id"], "2024"])
    assert sorted(name for name, _ in journal_slots(case) if name.startswith("outer-")) == sorted(name + suffix for name in names)
    assert all(slot(case, name) is None for name in ("execution", "forecasts", "validation", "terminal"))


@pytest.mark.parametrize("kind, started, completed", [
    ("wrong_matrix", 0, 0), ("wrong_horizon", 0, 0), ("completed_first", 0, 0),
    ("duplicate_start", 1, 0), ("unknown_state", 1, 0), ("wrong_completion", 1, 0), ("overflow", 3, 3)])
def test_fresh_child_progress_refuses_before_bad_event_publication(outer_case, monkeypatch, kind, started, completed):
    case = outer_case
    def fitted(inputs, fold, plan, execution, limits, progress):
        fits = expected_fits(case)
        for fit in fits[:completed]:
            progress("started", fit)
            progress("completed", fit)
        if started > completed:
            progress("started", fits[completed])
        state, fit = "started", fits[0]
        if kind in ("wrong_matrix", "wrong_completion"):
            fit = {**fit, "prediction_matrix_id": "wrong"}
        if kind == "wrong_horizon":
            fit = {**fit, "horizon_sessions": 1}
        if kind in ("completed_first", "wrong_completion"):
            state = "completed"
        if kind == "unknown_state":
            state = "unexpected"
        progress(state, fit)
        pytest.fail("malformed live progress was accepted")
    monkeypatch.setattr(outer, "run_fitted_trial", fitted)
    monkeypatch.setattr(outer, "load_magnitude_targets", lambda *a, **kw: pytest.fail("bad progress opened labels"))
    with outer_problem("EXPERIMENT_IDENTITY_CONFLICT", CONFLICT):
        run(case)
    assert_live_child_refusal(case, started, completed)


@pytest.mark.parametrize("kind, completed, pending", [
    ("no_fits", 0, False), ("partial", 1, False), ("pending", 2, True), ("success_extra", 3, False),
    ("failure_without_start", 0, False), ("failure_extra", 1, True), ("unknown_event", 1, True)])
def test_fresh_child_completion_refuses_before_execution_publication(outer_case, monkeypatch, kind, completed, pending):
    case = outer_case
    def fitted(inputs, fold, plan, execution, limits, progress):
        fits = expected_fits(case)
        for fit in fits[:completed]:
            progress("started", fit)
            progress("completed", fit)
        if pending:
            progress("started", fits[completed])
        if kind.startswith("failure") or kind == "unknown_event":
            result = {"event": "unknown" if kind == "unknown_event" else "failed"}
        else:
            result = {"event": "succeeded", "forecasts": [{"event_id": event, "horizon_sessions": horizon,
                "predicted_absolute_move_pct": 5.0} for event in fold["evaluation_event_ids"] for horizon in plan["horizons"]]}
        if kind.endswith("extra"):
            result["extra"] = "unexpected"
        return result
    monkeypatch.setattr(outer, "run_fitted_trial", fitted)
    monkeypatch.setattr(outer, "load_magnitude_targets", lambda *a, **kw: pytest.fail("bad completion opened labels"))
    with outer_problem("EXPERIMENT_IDENTITY_CONFLICT", CONFLICT):
        run(case)
    assert_live_child_refusal(case, completed + int(pending), completed)


@pytest.mark.parametrize("kind", ["cutoff", "purpose", "receipt_id", "reordered", "missing", "negative", "nonfinite", "boolean"])
def test_fresh_targets_refuse_before_validation_publication(outer_case, monkeypatch, kind):
    original = outer.load_magnitude_targets
    def labels(*args, **kwargs):
        frame = original(*args, **kwargs).copy(deep=True)
        if kind in ("cutoff", "purpose", "receipt_id"):
            key, value = {"cutoff": ("labels_before", "2026-01-01"), "purpose": ("purpose", "final"),
                          "receipt_id": ("inputs_id", "wrong")}[kind]
            frame.attrs[key] = value
            if kind != "receipt_id":
                frame.attrs["inputs_id"] = content_hash({key: value for key, value in frame.attrs.items()
                                                        if key not in ("inputs_id", "holdout_exclusions")})
        elif kind == "reordered":
            frame = frame.iloc[::-1].copy()
        elif kind == "missing":
            frame = frame.iloc[1:].copy()
        else:
            frame["absolute_move_pct"] = frame["absolute_move_pct"].astype(object)
            frame.loc[frame.index[0], "absolute_move_pct"] = {"negative": -1.0, "nonfinite": float("nan"), "boolean": True}[kind]
        return frame
    monkeypatch.setattr(outer, "load_magnitude_targets", labels)
    with outer_problem("EXPERIMENT_IDENTITY_CONFLICT", CONFLICT):
        run(outer_case)
    assert slot(outer_case, "forecasts") is not None
    assert slot(outer_case, "execution")["status"] == "succeeded"
    assert slot(outer_case, "validation") is None
    assert slot(outer_case, "terminal") is None



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
