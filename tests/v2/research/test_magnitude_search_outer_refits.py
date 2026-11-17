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
               "outer-terminal:" + key_b: failed_terminal(key_b, "2024", "b", "inputs-b")}
    fold_a = {"outer_year": "2024",
              "predictions": [{"event_id": "event", "horizon_sessions": 1,
                               "predicted_absolute_move_pct": 1.0,
                               "absolute_move_pct": 2.0}],
              "event_count": 1, "forecast_count": 1,
              "prospective_candidate_id": "a",
              "holdout_results": {"random": "excluded", "rolling": "excluded"}}
    failed_b = failed_terminal(key_b, "2024", "b", "inputs-b")

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


EXPECTED = [{"horizon_sessions": 5}, {"horizon_sessions": 3}]


def _failed_journal(*, fits=None, attempt_count=None):
    plan = {"plan_id": "plan"}
    key = refit_key(plan, "2024", "rmse")
    started = {"outer_id": key}
    if fits is None:
        fits = [EXPECTED[0]]
    if attempt_count is None:
        attempt_count = len(fits) + 1
    inputs_id = "inputs"
    records = {"start": started,
               "inputs": {"manifest": "inputs"},
               "terminal": failed_terminal(key, "2024", "rmse", inputs_id),
               "worker-result": {"event": "failed"},
               "execution": {"status": "failed", "forecasts_id": None, "fits": fits,
                             "completed_fit_count": len(fits), "fit_attempt_count": attempt_count}}
    for fit in fits:
        slot = ":" + str(fit["horizon_sessions"])
        records["fit-started" + slot] = fit
        records["fit-completed" + slot] = {**fit, "fit_seconds": 1.0}
    if attempt_count == len(fits) + 1 and len(fits) < len(EXPECTED):
        records["fit-started:" + str(EXPECTED[len(fits)]["horizon_sessions"])] = EXPECTED[len(fits)]
    return records, key, started


def _replay_failed(records, key, started, monkeypatch):
    monkeypatch.setattr(outer, "load_inputs",
                        lambda store, record, inputs_id: {"inputs_id": inputs_id})
    monkeypatch.setattr(outer, "fit_evidence",
                        lambda inputs, fold, plan, execution: EXPECTED)
    return outer._replay_failed(records.get, None, started, key, "2024", "rmse", {}, {},
                                records["execution"])


def test_replay_failed_returns_the_terminal_for_a_valid_journal(monkeypatch):
    records, key, started = _failed_journal()
    assert _replay_failed(records, key, started, monkeypatch) == failed_terminal(
        key, "2024", "rmse", "inputs")


def test_replay_failed_accepts_a_failure_before_the_first_horizon(monkeypatch):
    records, key, started = _failed_journal(fits=[], attempt_count=0)
    assert _replay_failed(records, key, started, monkeypatch) == failed_terminal(
        key, "2024", "rmse", "inputs")


def test_replay_failed_rejects_receipt_fits_that_differ_from_the_recomputed_prefix(monkeypatch):
    records, key, started = _failed_journal(fits=[{"horizon_sessions": 9}], attempt_count=1)
    with pytest.raises(OpsError) as caught:
        _replay_failed(records, key, started, monkeypatch)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_replay_failed_rejects_an_invalid_completed_fit_marker(monkeypatch):
    records, key, started = _failed_journal()
    records["fit-completed:5"] = {}
    with pytest.raises(OpsError) as caught:
        _replay_failed(records, key, started, monkeypatch)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_replay_failed_rejects_a_missing_attempted_fit_start(monkeypatch):
    records, key, started = _failed_journal()
    del records["fit-started:3"]
    with pytest.raises(OpsError) as caught:
        _replay_failed(records, key, started, monkeypatch)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_replay_failed_rejects_an_attempt_count_beyond_the_attempted_fit(monkeypatch):
    records, key, started = _failed_journal(attempt_count=3)
    with pytest.raises(OpsError) as caught:
        _replay_failed(records, key, started, monkeypatch)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_replay_failed_rejects_a_non_failed_worker_result(monkeypatch):
    records, key, started = _failed_journal()
    records["worker-result"] = {"event": "succeeded"}
    with pytest.raises(OpsError) as caught:
        _replay_failed(records, key, started, monkeypatch)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
