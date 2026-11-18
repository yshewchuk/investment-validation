"""Unit tests for blend winner rows in report-only winner rows (fake paired callable)."""
import math

from engine.v2.models.magnitude_elasticnet_recipe import elasticnet_recipe
from experiments.magnitude_search_blend_outer import average_forecasts
from experiments.magnitude_search_refits import refit_key
from experiments.size_search_winners import refit_forecast_map, winner_rows

_HASH = "sha256:" + "0" * 64
PAIRED_KWARGS = {"horizons": [1], "implied_moves": None, "implied_scales": {}, "label_reasons": {}}


class FakePaired:
    def __call__(self, ev, fc, cmp, tg, **kwargs):
        return {}


def _plan():
    return {"plan_id": "p", "horizons": [1],
            "candidates": [
                {"candidate_id": "a", "recipe": elasticnet_recipe(
                    columns=["a"], feature_dictionary_hash=_HASH, alpha=0.1, l1_ratio=0.5)}],
            "blends": [{"blend_id": "B", "name": "b1", "component_candidate_ids": ["a", "c"]}],
            "folds": [{"outer": {"year": "2024", "evaluation_event_ids": ["e1"]}}],
            "selection": {"winner_rules": ["overall"]}}


SELECTION = {"selections": [
    {"outer_year": "2024", "scores": [],
     "winners": {"overall": {"candidate_id": "B", "tie_band": 0.1}},
     "selected_candidate_id": "B"}]}

EVENTS = [{"event_id": "e1"}]
TARGETS = {"e1": 2.0}
COMPARISONS = []


def _blend_refit(status="completed", result=None):
    return {"outer_year": "2024", "candidate_id": "B", "kind": "blend", "status": status, "result": result}


def _blend_predictions():
    return {"predictions": [{"event_id": "e1", "horizon_sessions": 1,
                             "predicted_absolute_move_pct": 1.0, "absolute_move_pct": 2.0}]}


def _forecast_rows():
    return [{"event_id": "e1", "horizon_sessions": 1, "predicted_absolute_move_pct": 1.0}]


def _by_id(rows):
    return {row["candidate_id"]: row for row in rows}


def test_blend_winner_row_has_blend_kind_components_and_mixed_transform():
    refits = [_blend_refit("completed", _blend_predictions())]
    rows = _by_id(winner_rows(_plan(), SELECTION, refits, {("2024", "B"): _forecast_rows()},
                              EVENTS, COMPARISONS, TARGETS, paired=FakePaired(), paired_kwargs=PAIRED_KWARGS))
    row = rows["B"]
    assert row["kind"] == "blend"
    assert row["components"] == ["a", "c"]
    assert row["target_transform"] == "mixed"
    assert row["inner_scores_optimistic"] is True
    assert row["status"] == "completed"
    assert row["error_code"] is None


def test_failed_blend_refit_marks_row_component_failed():
    refits = [_blend_refit("failed", None)]
    rows = _by_id(winner_rows(_plan(), SELECTION, refits, {}, EVENTS, COMPARISONS, TARGETS,
                              paired=FakePaired(), paired_kwargs=PAIRED_KWARGS))
    row = rows["B"]
    assert row["kind"] == "blend"
    assert row["status"] == "failed"
    assert row["error_code"] == "component_failed"
    assert row["paired"] is None


def test_best_single_that_wins_no_rule_gets_a_completed_row():
    selection = {"selections": [
        {"outer_year": "2024", "scores": [],
         "winners": {"overall": {"candidate_id": "B", "tie_band": 0.1}},
         "selected_candidate_id": "B", "best_non_blend_id": "a"}]}
    refits = [_blend_refit("completed", _blend_predictions()),
              {"outer_year": "2024", "candidate_id": "a", "kind": "candidate", "status": "completed", "result": None}]
    forecasts = {("2024", "B"): _forecast_rows(), ("2024", "a"): _forecast_rows()}
    rows = _by_id(winner_rows(_plan(), selection, refits, forecasts, EVENTS, COMPARISONS, TARGETS,
                              paired=FakePaired(), paired_kwargs=PAIRED_KWARGS))
    row = rows["a"]
    assert row["rules"] == []
    assert row["primary"] is False
    assert row["kind"] == "candidate"
    assert row["status"] == "completed"
    assert row["error_code"] is None
    assert row["outer_years"] == ["2024"]
    assert row["inner"] == [{"outer_year": "2024", "rules": [], "scores": {}, "tie_band": {}}]


PLAN_FM = {"plan_id": "p"}


def _stored(predicted_by_event):
    return [{"event_id": event_id, "horizon_sessions": 1, "predicted_absolute_move_pct": value}
            for event_id, value in predicted_by_event]


def test_refit_forecast_map_averages_blend_component_forecasts_and_skips_unwanted():
    stored = {
        refit_key(PLAN_FM, "2024", "a"): _stored([("e1", 1.0), ("e2", 4.0)]),
        refit_key(PLAN_FM, "2024", "c"): _stored([("e1", 3.0), ("e2", 6.0)]),
        refit_key(PLAN_FM, "2024", "z"): _stored([("e1", 7.0), ("e2", 9.0)]),
    }
    refits = [
        {"outer_year": "2024", "candidate_id": "B", "kind": "blend", "status": "completed",
         "component_candidate_ids": ["a", "c"]},
        {"outer_year": "2024", "candidate_id": "z", "kind": "candidate", "status": "completed"},
        # Not completed: skipped even though its components are stored.
        {"outer_year": "2024", "candidate_id": "B2", "kind": "blend", "status": "failed",
         "component_candidate_ids": ["a", "c"]},
        # Completed but not wanted for this year: skipped (no stored forecasts, so read would raise).
        {"outer_year": "2024", "candidate_id": "q", "kind": "candidate", "status": "completed"},
        {"outer_year": "2023", "candidate_id": "B", "kind": "blend", "status": "completed",
         "component_candidate_ids": ["a", "c"]},
    ]
    wanted = {"2024": {"B", "B2", "z"}}
    result = refit_forecast_map(PLAN_FM, refits, wanted, lambda key: stored[key], average_forecasts)
    assert result == {
        ("2024", "B"): _stored([("e1", 2.0), ("e2", 5.0)]),
        ("2024", "z"): _stored([("e1", 7.0), ("e2", 9.0)]),
    }


def test_refit_forecast_map_blend_average_is_fsum_of_component_values():
    stored = {
        refit_key(PLAN_FM, "2024", "a"): _stored([("e1", 0.1)]),
        refit_key(PLAN_FM, "2024", "c"): _stored([("e1", 0.2)]),
    }
    refits = [{"outer_year": "2024", "candidate_id": "B", "kind": "blend", "status": "completed",
               "component_candidate_ids": ["a", "c"]}]
    result = refit_forecast_map(PLAN_FM, refits, {"2024": {"B"}}, lambda key: stored[key], average_forecasts)
    assert result[("2024", "B")][0]["predicted_absolute_move_pct"] == math.fsum([0.1, 0.2]) / 2
