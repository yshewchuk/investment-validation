from unittest.mock import Mock

import pytest

from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_outer as outer

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
