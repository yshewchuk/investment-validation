from unittest.mock import Mock

import pytest

from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_outer as outer
from experiments.magnitude_search_refits import failed_terminal, refit_key

MESSAGE = "refit unit is outside the sealed selection"


def _selection():
    return {"selections": [{"outer_year": "2024", "selected_candidate_id": "overall",
                            "winners": {}, "distinct_winner_ids": ["overall", "rmse"]}],
            "prospective_candidate_id": "overall"}


def _plan(*, winner):
    return {"plan_id": "plan",
            "selection": {"winner_rules": ["overall", "rmse", "mae"]} if winner else {},
            "candidates": [{"candidate_id": value} for value in ("overall", "rmse", "extra")],
            "folds": [{"outer": {"year": "2024"}}]}


def _call(monkeypatch, plan, candidate_id):
    admitted = Mock(return_value={"admitted": True})
    monkeypatch.setattr(outer, "validate_execution_limits", lambda limits: None)
    monkeypatch.setattr(outer, "require_current", lambda *args: None)
    monkeypatch.setattr(outer, "read_search_plan", lambda *args: plan)
    monkeypatch.setattr(outer, "SearchTrials", lambda *args: object())
    monkeypatch.setattr(outer, "_native_selection", lambda *args: _selection())
    monkeypatch.setattr(outer, "_execute_admitted", admitted)
    result = outer.execute_outer_fold(None, None, None, None, {}, None, "2024",
                                      limits={}, candidate_id=candidate_id)
    return result, admitted


def test_execute_outer_fold_passes_a_sealed_refit_unit_to_admission(monkeypatch):
    result, admitted = _call(monkeypatch, _plan(winner=True), "rmse")
    assert result == {"admitted": True}
    assert admitted.call_args.kwargs["candidate_id"] == "rmse"


def test_execute_outer_fold_refuses_a_candidate_outside_the_refit_units(monkeypatch):
    with pytest.raises(OpsError) as caught:
        _call(monkeypatch, _plan(winner=True), "extra")
    assert (caught.value.problem.code, caught.value.problem.message) == (
        "INVALID_EXPERIMENT_SPEC", MESSAGE)


def test_execute_outer_fold_refuses_a_missing_candidate_on_a_winner_plan(monkeypatch):
    with pytest.raises(OpsError) as caught:
        _call(monkeypatch, _plan(winner=True), None)
    assert (caught.value.problem.code, caught.value.problem.message) == (
        "INVALID_EXPERIMENT_SPEC", MESSAGE)


def test_execute_outer_fold_refuses_a_candidate_on_a_legacy_plan(monkeypatch):
    with pytest.raises(OpsError) as caught:
        _call(monkeypatch, _plan(winner=False), "overall")
    assert (caught.value.problem.code, caught.value.problem.message) == (
        "INVALID_EXPERIMENT_SPEC", MESSAGE)


def test_collect_outer_results_refits_every_winner_unit(monkeypatch):
    plan = {"plan_id": "plan",
            "selection": {"winner_rules": ["overall", "rmse", "mae"]},
            "candidates": [{"candidate_id": value} for value in ("a", "b")],
            "folds": [{"outer": {"year": "2024"}}],
            "horizons": [1]}
    stored = {"selections": [{"outer_year": "2024", "selected_candidate_id": "a",
                              "distinct_winner_ids": ["a", "b"]}],
              "prospective_candidate_id": "a"}
    key_a, key_b = refit_key(plan, "2024", "a"), refit_key(plan, "2024", "b")
    records = {"selection": stored,
               "outer-start:" + key_a: {"limits": {}},
               "outer-start:" + key_b: {"limits": {}},
               "outer-terminal:" + key_a: {"status": "succeeded"},
               "outer-terminal:" + key_b: failed_terminal(key_b, "2024", "b")}
    fold_a = {"outer_year": "2024",
              "predictions": [{"event_id": "event", "horizon_sessions": 1,
                               "predicted_absolute_move_pct": 1.0,
                               "absolute_move_pct": 2.0}],
              "event_count": 1, "forecast_count": 1,
              "prospective_candidate_id": "a",
              "holdout_results": {"random": "excluded", "rolling": "excluded"}}
    failed_b = failed_terminal(key_b, "2024", "b")

    def execute(conn, store, repository, registration, plan, fold, selection, limits, *,
                replay_only=False, input_callback=None, candidate_id=None):
        return fold_a if candidate_id == "a" else failed_b

    monkeypatch.setattr(outer, "require_current", lambda *args: None)
    monkeypatch.setattr(outer, "read_search_plan", lambda *args: plan)
    monkeypatch.setattr(outer, "read_record",
                        lambda conn, store, run_id, slot: records[slot])
    monkeypatch.setattr(outer, "validate_execution_limits", lambda limits: None)
    monkeypatch.setattr(outer, "SearchTrials", lambda *args: object())
    monkeypatch.setattr(outer, "_native_selection", lambda *args: stored)
    monkeypatch.setattr(outer, "_execute_admitted", execute)
    monkeypatch.setattr(outer, "magnitude_metrics", lambda *args, **kwargs: {})

    result = outer.collect_outer_results(
        None, None, None, Mock(run_id="run"), {}, None)

    assert result["refits"] == [
        {"outer_year": "2024", "candidate_id": "a", "status": "completed",
         "result": fold_a},
        {"outer_year": "2024", "candidate_id": "b", "status": "failed",
         "result": failed_b},
    ]
    assert result["folds"] == [fold_a]


def _failed_journal():
    plan = {"plan_id": "plan"}
    key = refit_key(plan, "2024", "rmse")
    started = {"outer_id": key}
    fit = {"horizon_sessions": 1}
    attempted = {"horizon_sessions": 2}
    records = {"start": started,
               "terminal": failed_terminal(key, "2024", "rmse"),
               "worker-result": {"event": "failed"},
               "execution": {"status": "failed", "forecasts_id": None, "fits": [fit],
                             "completed_fit_count": 1, "fit_attempt_count": 2},
               "fit-started:1": fit, "fit-completed:1": {**fit, "fit_seconds": 1.0},
               "fit-started:2": attempted}
    return records, key, started


def _replay_failed(records, key, started):
    return outer._replay_failed(records.get, started, key, "2024", "rmse", [1, 2])


def test_replay_failed_returns_the_terminal_for_a_valid_journal():
    records, key, started = _failed_journal()
    assert _replay_failed(records, key, started) == failed_terminal(key, "2024", "rmse")


def test_replay_failed_rejects_an_invalid_completed_fit_marker():
    records, key, started = _failed_journal()
    records["fit-completed:1"] = {}
    with pytest.raises(OpsError) as caught:
        _replay_failed(records, key, started)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_replay_failed_rejects_a_missing_attempted_fit_start():
    records, key, started = _failed_journal()
    del records["fit-started:2"]
    with pytest.raises(OpsError) as caught:
        _replay_failed(records, key, started)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_replay_failed_rejects_a_wrong_fit_attempt_count():
    records, key, started = _failed_journal()
    records["execution"]["fit_attempt_count"] = 1
    with pytest.raises(OpsError) as caught:
        _replay_failed(records, key, started)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_replay_failed_rejects_a_non_failed_worker_result():
    records, key, started = _failed_journal()
    records["worker-result"] = {"event": "succeeded"}
    with pytest.raises(OpsError) as caught:
        _replay_failed(records, key, started)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
