"""Hand-computed oracles for per-event blended SIZE selection."""
import copy
import math

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments.magnitude_search_selection import (
    blended_event_score,
    history_mean_rows,
    selection_tie_band,
)
from experiments.magnitude_search_request_versions import compile_request
from tests.v2.research.test_magnitude_search_request_v3 import v3_request
from tests.v2.research.test_magnitude_search_request_v4 import v4_request


WEIGHTS = {"rmse": 0.6, "mae": 0.3, "large_move_underprediction": 0.1}
POLICY = {"metric_weights": WEIGHTS, "lead_weights": {"1": 1.0, "3": 0.0, "5": 0.0},
    "large_move_threshold_pct": 10.0, "tie_band_floor": 0.005}


def _rows(values):
    return [{"event_id": f"e{index}", "horizon_sessions": 1,
        "absolute_move_pct": actual, "predicted_absolute_move_pct": predicted}
        for index, (actual, predicted) in enumerate(values)]


def test_blend_normalizes_three_terms_against_same_lead1_events():
    candidate = _rows([(12, 9), (4, 5), (8, 6)])
    history = _rows([(12, 10), (4, 6), (8, 10)])
    for rows in (candidate, history):
        rows.extend([
            {"event_id": "e0", "horizon_sessions": 3, "absolute_move_pct": 12,
             "predicted_absolute_move_pct": 1000},
            {"event_id": "e0", "horizon_sessions": 5, "absolute_move_pct": 12,
             "predicted_absolute_move_pct": 1000},
        ])
    score = blended_event_score(candidate, history, metric_weights=WEIGHTS,
        large_move_threshold_pct=10.0)
    assert score["metrics"] == {"count": 3, "large_move_count": 1,
        "rmse": pytest.approx(math.sqrt(14 / 3)), "mae": pytest.approx(2.0),
        "large_move_underprediction": pytest.approx(3.0)}
    assert score["normalized"] == {"rmse": pytest.approx(math.sqrt(14 / 3) / 2),
        "mae": pytest.approx(1.0), "large_move_underprediction": pytest.approx(1.5)}
    assert score["blend_score"] == pytest.approx(0.6 * math.sqrt(14 / 3) / 2 + 0.3 + 0.1 * 1.5)
    with pytest.raises(ValueError, match="one row per event"):
        blended_event_score(candidate + [candidate[0]], history, metric_weights=WEIGHTS,
            large_move_threshold_pct=10.0)


def test_zero_comparator_terms_have_defined_parity_or_refuse_undefined_ratio():
    perfect = _rows([(12.0, 12.0), (4.0, 4.0)])
    parity = blended_event_score(perfect, perfect, metric_weights=WEIGHTS,
        large_move_threshold_pct=10.0)
    assert parity["normalized"] == {"rmse": 1.0, "mae": 1.0,
        "large_move_underprediction": 1.0}
    worse = _rows([(12.0, 11.0), (4.0, 4.0)])
    with pytest.raises(ValueError, match="zero history_mean error"):
        blended_event_score(worse, perfect, metric_weights=WEIGHTS,
            large_move_threshold_pct=10.0)


def test_history_mean_uses_inner_training_fallback_and_causal_feature():
    rows = history_mean_rows(
        {"training": [{"absolute_move_pct": 2.0}, {"absolute_move_pct": 4.0}],
         "features": [
             {"event_id": "prior", "horizon_sessions": 1, "n_prior": 1,
              "mean_prior_abs_move": 7.0},
             {"event_id": "empty", "horizon_sessions": 1, "n_prior": 0,
              "mean_prior_abs_move": None},
         ]},
        {"targets": [{"event_id": "prior", "absolute_move_pct": 12.0},
                     {"event_id": "empty", "absolute_move_pct": 4.0}]})
    assert rows == [
        {"event_id": "prior", "horizon_sessions": 1, "absolute_move_pct": 12.0,
         "predicted_absolute_move_pct": 7.0},
        {"event_id": "empty", "horizon_sessions": 1, "absolute_move_pct": 4.0,
         "predicted_absolute_move_pct": 3.0},
    ]


def test_tie_band_uses_max_sample_se_floor_then_candidate_id():
    scores = [
        {"candidate_id": "z", "blend_score": 0.5},
        {"candidate_id": "a", "blend_score": 0.504},
        {"candidate_id": "b", "blend_score": 0.507},
    ]
    yearly = [
        {"candidate_id": "z", "yearly": {"2022": {"blend_score": 0.49},
            "2023": {"blend_score": 0.51}, "2024": {"blend_score": 0.50}}},
        {"candidate_id": "a", "yearly": {}}, {"candidate_id": "b", "yearly": {}},
    ]
    selected, band = selection_tie_band(scores, yearly, floor=0.005)
    assert band["standard_error"] == pytest.approx(0.005773502691896258)
    assert band["width"] == pytest.approx(band["standard_error"])
    assert band["tied_candidate_ids"] == ["a", "z"]
    assert selected == "a"

    floor_scores = [{"candidate_id": "z", "blend_score": 0.5},
                    {"candidate_id": "a", "blend_score": 0.504},
                    {"candidate_id": "b", "blend_score": 0.506}]
    floor_yearly = [{"candidate_id": "z", "yearly": {
        "2023": {"blend_score": 0.5}, "2024": {"blend_score": 0.5}}}]
    selected, band = selection_tie_band(floor_scores, floor_yearly, floor=0.005)
    assert band["width"] == 0.005
    assert band["tied_candidate_ids"] == ["a", "z"]
    assert selected == "a"


def test_v3_policy_is_request_declared_validated_and_identity_bound():
    legacy = compile_request(v3_request())
    request = v3_request()
    request["selection"] = copy.deepcopy(POLICY)
    blended = compile_request(request)
    assert blended["request"]["selection"] == POLICY
    assert blended["request_id"] != legacy["request_id"]
    request["selection"]["lead_weights"]["1"] = 0.5
    with pytest.raises(OpsError) as caught:
        compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_v4_hgb_request_accepts_the_same_hashed_selection_declaration():
    request = v4_request()
    request["selection"] = copy.deepcopy(POLICY)
    compiled = compile_request(request)
    assert compiled["request"]["schema_version"] == "magnitude_search_request.v4"
    assert compiled["request"]["selection"] == POLICY
    assert compiled["request_id"] == content_hash(compiled["request"])
