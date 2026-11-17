"""Unit tests for blend winner rows in report-only winner rows (fake paired callable)."""
from engine.v2.models.magnitude_elasticnet_recipe import elasticnet_recipe
from experiments.size_search_winners import winner_rows

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
