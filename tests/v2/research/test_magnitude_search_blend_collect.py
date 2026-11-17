from unittest.mock import Mock

import pytest

from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_blend_outer as blend_outer
from experiments import magnitude_search_outer as outer
from experiments.magnitude_search_refits import failed_terminal, refit_key
from experiments.size_distribution_report import collect_distribution

BLEND = {"blend_id": "blend_ab", "name": "ab", "component_candidate_ids": ["b", "a"]}


def _plan(extra_candidates=()):
    return {"plan_id": "plan",
            "selection": {"winner_rules": ["overall", "rmse", "mae"]},
            "candidates": [{"candidate_id": value} for value in ("a", "b", "rmse", *extra_candidates)],
            "blends": [BLEND],
            "folds": [{"outer": {"year": "2024"}}],
            "horizons": [1]}


def _fold(predicted, candidate_id):
    return {"outer_year": "2024", "result_id": "result-" + candidate_id,
            "predictions": [{"event_id": "event", "horizon_sessions": 1,
                             "predicted_absolute_move_pct": predicted, "absolute_move_pct": 2.0}],
            "event_count": 1, "forecast_count": 1, "fit_attempt_count": 1,
            "evaluation_inputs": {}, "prospective_candidate_id": "blend_ab",
            "holdout_results": {"random": "excluded", "rolling": "excluded"}}


def _collect(monkeypatch, plan, stored, outcomes):
    """Run collect_outer_results with per-unit outcomes and record the callback's unit."""
    keys = {unit: refit_key(plan, "2024", unit) for unit in outcomes}
    records = {"selection": stored}
    for unit, outcome in outcomes.items():
        records["outer-start:" + keys[unit]] = {"limits": {}}
        records["outer-terminal:" + keys[unit]] = {"status": "succeeded"}
    callbacks = []

    def execute(conn, store, repository, registration, plan, fold, selection, limits, *,
                replay_only=False, input_callback=None, candidate_id=None):
        if input_callback is not None:
            callbacks.append(candidate_id)
        return outcomes[candidate_id]

    monkeypatch.setattr(outer, "require_current", lambda *args: None)
    monkeypatch.setattr(outer, "read_search_plan", lambda *args: plan)
    monkeypatch.setattr(outer, "read_record", lambda conn, store, run_id, slot: records[slot])
    monkeypatch.setattr(outer, "validate_execution_limits", lambda limits: None)
    monkeypatch.setattr(outer, "SearchTrials", lambda *args: object())
    monkeypatch.setattr(outer, "_native_selection", lambda *args: stored)
    monkeypatch.setattr(outer, "_execute_admitted", execute)
    monkeypatch.setattr(outer, "magnitude_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr(blend_outer, "magnitude_metrics", lambda rows, **kwargs: {"count": len(rows)})
    result = outer.collect_outer_results(None, None, None, Mock(run_id="run"), {}, None,
                                         input_callback=lambda fold, inputs: None)
    return result, callbacks


def test_blend_overall_yields_a_blend_fold_and_refit_rows(monkeypatch):
    plan = _plan()
    stored = {"selections": [{"outer_year": "2024", "selected_candidate_id": "blend_ab",
                              "distinct_winner_ids": ["blend_ab"], "best_non_blend_id": "rmse"}],
              "prospective_candidate_id": "blend_ab"}
    outcomes = {"a": _fold(1.0, "a"), "b": _fold(3.0, "b"), "rmse": _fold(9.0, "rmse")}
    result, callbacks = _collect(monkeypatch, plan, stored, outcomes)

    blend_fold = result["folds"][0]
    assert blend_fold["candidate"]["candidate_id"] == "blend_ab"
    assert blend_fold["candidate"]["component_candidate_ids"] == ["a", "b"]
    assert blend_fold["predictions"][0]["predicted_absolute_move_pct"] == 2.0
    assert callbacks == ["a"]
    assert result["refits"] == [
        {"outer_year": "2024", "candidate_id": "a", "status": "completed", "result": outcomes["a"]},
        {"outer_year": "2024", "candidate_id": "b", "status": "completed", "result": outcomes["b"]},
        {"outer_year": "2024", "candidate_id": "rmse", "status": "completed", "result": outcomes["rmse"]},
        {"outer_year": "2024", "candidate_id": "blend_ab", "kind": "blend",
         "component_candidate_ids": ["a", "b"], "status": "completed", "result": blend_fold},
    ]


def test_failed_component_of_a_non_overall_blend_fails_only_its_blend_row(monkeypatch):
    plan = _plan()
    stored = {"selections": [{"outer_year": "2024", "selected_candidate_id": "a",
                              "distinct_winner_ids": ["a", "blend_ab"], "best_non_blend_id": "a"}],
              "prospective_candidate_id": "a"}
    failed_b = failed_terminal(refit_key(plan, "2024", "b"), "2024", "b", "inputs-b")
    outcomes = {"a": _fold(1.0, "a"), "b": failed_b}
    result, _ = _collect(monkeypatch, plan, stored, outcomes)

    assert result["folds"] == [outcomes["a"]]
    assert result["refits"][-1] == {"outer_year": "2024", "candidate_id": "blend_ab", "kind": "blend",
                                    "component_candidate_ids": ["a", "b"], "status": "failed",
                                    "result": None}
    assert result["refits"][1] == {"outer_year": "2024", "candidate_id": "b", "status": "failed",
                                   "result": failed_b}


def _replay(monkeypatch, candidate_id):
    plan = _plan()
    registration = Mock(run_id="run", document={"magnitude_execution": {"evaluation_labels_before": None}})
    stored = {"selections": [{"outer_year": "2024", "selected_candidate_id": "blend_ab",
                              "distinct_winner_ids": ["blend_ab", "rmse"], "best_non_blend_id": "rmse"}],
              "prospective_candidate_id": "blend_ab"}
    terminal = {"status": "failed"}
    records = {"outer-terminal:" + refit_key(plan, "2024", candidate_id): terminal}
    replay_failed = Mock(return_value=terminal)
    monkeypatch.setattr(outer, "source_closure", lambda *args: {})
    monkeypatch.setattr(outer, "execution_for_candidate", lambda *args: {})
    monkeypatch.setattr(outer, "read_record", lambda conn, store, run_id, slot: records.get(slot))
    monkeypatch.setattr(outer, "_replay_failed", replay_failed)
    return lambda: outer._execute_admitted(
        None, None, None, registration, plan, {"year": "2024"}, stored, {},
        replay_only=True, candidate_id=candidate_id), replay_failed


def test_failed_terminal_of_an_overall_blend_component_refuses(monkeypatch):
    run, replay_failed = _replay(monkeypatch, "a")
    with pytest.raises(OpsError) as caught:
        run()
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    replay_failed.assert_not_called()


def test_failed_terminal_of_a_non_required_unit_replays(monkeypatch):
    run, replay_failed = _replay(monkeypatch, "rmse")
    assert run() == {"status": "failed"}
    replay_failed.assert_called_once()


def test_distribution_refuses_a_blend_selection_before_validation():
    plan = {"horizons": [1], "folds": [], "candidates": [], "blends": [BLEND]}
    result = collect_distribution(plan, {}, {}, {"2024": "blend_ab"})
    assert result == {"schema_version": "size_distribution.v1", "status": "blend_not_distributed"}
