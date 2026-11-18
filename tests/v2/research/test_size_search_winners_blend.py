"""Unit tests for blend winner rows in report-only winner rows (fake paired callable)."""
import math

from engine.v2.foundation import content_hash
from engine.v2.models.magnitude_elasticnet_recipe import elasticnet_recipe
from experiments.magnitude_search_blend_outer import average_forecasts
from experiments.magnitude_search_refits import refit_key
from experiments.size_search_winners import refit_forecast_map, selected_forecasts, winner_rows

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


def test_selected_forecasts_blend_is_fsum_average_of_component_refits():
    plan = _plan()
    stored = {
        refit_key(plan, "2024", "a"): _stored([("e1", 1.0), ("e2", 4.0)]),
        refit_key(plan, "2024", "c"): _stored([("e1", 3.0), ("e2", 6.0)]),
    }
    result = selected_forecasts(plan, "2024", "B", lambda key: stored[key], average_forecasts, True)
    # e1: fsum([1.0, 3.0]) / 2 = 2.0 ; e2: fsum([4.0, 6.0]) / 2 = 5.0
    assert result == _stored([("e1", 2.0), ("e2", 5.0)])


def test_selected_forecasts_plain_candidate_reads_its_own_refit():
    plan = _plan()
    stored = {refit_key(plan, "2024", "a"): _stored([("e1", 1.0), ("e2", 4.0)])}

    def no_average(lists):
        raise AssertionError("average must not be called for a plain candidate")

    result = selected_forecasts(plan, "2024", "a", lambda key: stored[key], no_average, True)
    assert result == _stored([("e1", 1.0), ("e2", 4.0)])


def test_selected_forecasts_legacy_reads_fold_key():
    plan = _plan()
    legacy_key = content_hash(["p", "2024"])
    stored = {legacy_key: _stored([("e1", 7.0)])}
    keys = []

    def read(key):
        keys.append(key)
        return stored[key]

    result = selected_forecasts(plan, "2024", "a", read, average_forecasts, False)
    assert keys == [legacy_key]
    assert result == _stored([("e1", 7.0)])


DISTINCT_PAIRED = {"sentinel": "distinct-paired-result"}


class RecordingPaired:
    def __init__(self):
        self.calls = []

    def __call__(self, ev, fc, cmp, tg, **kwargs):
        self.calls.append({"events": ev, "forecasts": fc, "comparisons": cmp, "targets": tg, "kwargs": kwargs})
        return DISTINCT_PAIRED


def test_completed_blend_winner_row_pairs_averaged_fold_forecasts_and_share():
    plan = _plan()
    blend_refit = {"outer_year": "2024", "candidate_id": "B", "kind": "blend", "status": "completed",
                   "component_candidate_ids": ["a", "c"], "result": _blend_predictions()}
    # Stored component forecasts. e2 is outside the fold (and outside TARGETS) and must not reach paired.
    stored = {
        refit_key(plan, "2024", "a"): _stored([("e1", 1.0), ("e2", 4.0)]),
        refit_key(plan, "2024", "c"): _stored([("e1", 2.0), ("e2", 6.0)]),
    }
    forecasts = refit_forecast_map(plan, [blend_refit], {"2024": {"B"}}, lambda key: stored[key], average_forecasts)
    paired = RecordingPaired()
    rows = _by_id(winner_rows(plan, SELECTION, [blend_refit], forecasts, EVENTS, COMPARISONS, TARGETS,
                              paired=paired, paired_kwargs=PAIRED_KWARGS))
    row = rows["B"]
    assert row["status"] == "completed"
    assert row["paired"] is DISTINCT_PAIRED
    assert len(paired.calls) == 1
    # Averaged e1 forecast: fsum([1.0, 2.0]) / 2 = 1.5 (e2 averaged to 5.0 but is outside the fold, so dropped).
    assert paired.calls[0]["forecasts"] == _stored([("e1", 1.5)])
    # Hand computation: lowest horizon is 1. Only e1 is in TARGETS (2.0). Averaged e1 forecast 1.5 < target 2.0,
    # so the one scored row is above forecast: share = 1 / 1 = 1.0.
    assert row["actual_above_forecast_share"] == 1.0
