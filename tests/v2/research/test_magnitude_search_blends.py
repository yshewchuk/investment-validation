"""Blend inner-scoring oracles: averaged component forecasts scored by the real blended metric."""
import math

import pytest

from engine.v2.ops.errors import OpsError
from experiments.magnitude_search_blends import average_component_rows, score_blends
from experiments.magnitude_search_selection import ZeroComparatorError, blended_event_score


WEIGHTS = {"rmse": 0.6, "mae": 0.3, "large_move_underprediction": 0.1}
THRESHOLD = 10.0
INNER_YEARS = ["2023", "2024"]


def _lead(event_id, actual, predicted):
    return {"event_id": event_id, "horizon_sessions": 1,
            "absolute_move_pct": actual, "predicted_absolute_move_pct": predicted}


def _other(event_id):
    return {"event_id": event_id, "horizon_sessions": 3,
            "absolute_move_pct": 12.0, "predicted_absolute_move_pct": 1000.0}


def _mean(values):
    return math.fsum(values) / len(values)


# Distinct event ids per inner year, as real inner folds have.
TARGETS_2023 = {"a0": 12.0, "a1": 4.0, "a2": 8.0}
TARGETS_2024 = {"b0": 10.0, "b1": 6.0, "b2": 14.0}
PREDICTIONS = {
    "c-alpha": {"a0": 9.0, "a1": 5.0, "a2": 6.0, "b0": 9.0, "b1": 5.0, "b2": 12.0},
    "c-beta": {"a0": 11.0, "a1": 4.0, "a2": 9.0, "b0": 8.0, "b1": 6.0, "b2": 13.0},
    "c-gamma": {"a0": 10.0, "a1": 7.0, "a2": 7.0, "b0": 11.0, "b1": 4.0, "b2": 11.0},
}
HISTORY_2023 = [_lead("a0", 12.0, 10.0), _lead("a1", 4.0, 6.0), _lead("a2", 8.0, 10.0)]
HISTORY_2024 = [_lead("b0", 10.0, 7.0), _lead("b1", 6.0, 5.0), _lead("b2", 14.0, 9.0)]
BLEND = {"name": "blend-x", "blend_id": "blend:x", "component_candidate_ids": ["c-gamma", "c-alpha", "c-beta"]}


def _component_rows(candidate, targets):
    return [_lead(event_id, targets[event_id], PREDICTIONS[candidate][event_id])
            for event_id in targets] + [_other(event_id) for event_id in targets]


def _component_years():
    return {candidate: {"2023": _component_rows(candidate, TARGETS_2023),
                        "2024": _component_rows(candidate, TARGETS_2024)}
            for candidate in PREDICTIONS}


def _score_fn(candidate_rows, history_rows):
    return blended_event_score(candidate_rows, history_rows, metric_weights=WEIGHTS,
                               large_move_threshold_pct=THRESHOLD)


def _explicit_average(year_targets):
    """Hand-averaged lead-1 rows over the three components, in event order."""
    return [_lead(event_id, target,
                  _mean([PREDICTIONS[c][event_id] for c in ("c-alpha", "c-beta", "c-gamma")]))
            for event_id, target in year_targets.items()]


def test_average_component_rows_equals_explicit_mean_drops_nonlead_and_sorts():
    components = [
        [_other("a1"), _lead("a1", 4.0, 5.0), _lead("a0", 12.0, 9.0)],
        [_lead("a0", 12.0, 11.0), _lead("a1", 4.0, 4.0), _other("a0")],
        [_lead("a1", 4.0, 7.0), _lead("a0", 12.0, 10.0)],
    ]
    averaged = average_component_rows(components)
    assert averaged == [
        {"event_id": "a0", "horizon_sessions": 1, "absolute_move_pct": 12.0,
         "predicted_absolute_move_pct": math.fsum([9.0, 11.0, 10.0]) / 3},
        {"event_id": "a1", "horizon_sessions": 1, "absolute_move_pct": 4.0,
         "predicted_absolute_move_pct": math.fsum([5.0, 4.0, 7.0]) / 3},
    ]


def test_score_blends_three_component_matches_real_blended_score_pooled_and_yearly():
    scores = score_blends([BLEND], _component_years(), INNER_YEARS,
        {"2023": HISTORY_2023, "2024": HISTORY_2024}, _score_fn, set())
    assert len(scores) == 1
    result = scores[0]
    avg_2023 = _explicit_average(TARGETS_2023)
    avg_2024 = _explicit_average(TARGETS_2024)
    expected_pooled = _score_fn(avg_2023 + avg_2024, HISTORY_2023 + HISTORY_2024)
    expected_2023 = _score_fn(avg_2023, HISTORY_2023)
    expected_2024 = _score_fn(avg_2024, HISTORY_2024)

    assert result["candidate_id"] == "blend:x"
    assert result["kind"] == "blend"
    assert result["blend_name"] == "blend-x"
    assert result["metrics"] == pytest.approx(expected_pooled["metrics"])
    assert result["normalized"] == pytest.approx(expected_pooled["normalized"])
    assert result["blend_score"] == pytest.approx(expected_pooled["blend_score"])
    assert list(result["yearly"]) == INNER_YEARS
    assert result["yearly"]["2023"]["blend_score"] == pytest.approx(expected_2023["blend_score"])
    assert result["yearly"]["2024"]["blend_score"] == pytest.approx(expected_2024["blend_score"])
    assert result["yearly"]["2023"]["metrics"] == pytest.approx(expected_2023["metrics"])
    assert result["yearly"]["2024"]["metrics"] == pytest.approx(expected_2024["metrics"])


def test_failed_component_excludes_blend_with_metrics_none():
    scores = score_blends([BLEND], _component_years(), INNER_YEARS,
        {"2023": HISTORY_2023, "2024": HISTORY_2024}, _score_fn, {"c-beta"})
    assert scores == [{"candidate_id": "blend:x", "metrics": None,
                       "kind": "blend", "blend_name": "blend-x"}]


def test_misaligned_event_ids_raise_identity_conflict():
    years = _component_years()
    years["c-gamma"]["2024"] = [row for row in years["c-gamma"]["2024"]
                                if row["event_id"] != "b2" or row["horizon_sessions"] != 1]
    with pytest.raises(OpsError) as caught:
        score_blends([BLEND], years, INNER_YEARS,
            {"2023": HISTORY_2023, "2024": HISTORY_2024}, _score_fn, set())
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_differing_absolute_move_pct_raises_identity_conflict():
    years = _component_years()
    years["c-beta"]["2023"] = [dict(row, absolute_move_pct=5.0)
                               if row["event_id"] == "a1" and row["horizon_sessions"] == 1 else row
                               for row in years["c-beta"]["2023"]]
    with pytest.raises(OpsError) as caught:
        score_blends([BLEND], years, INNER_YEARS,
            {"2023": HISTORY_2023, "2024": HISTORY_2024}, _score_fn, set())
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_empty_component_rows_raise_identity_conflict():
    with pytest.raises(OpsError) as caught:
        average_component_rows([])
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_duplicate_lead1_event_ids_raise_identity_conflict():
    components = [[_lead("a0", 12.0, 9.0), _lead("a0", 12.0, 10.0)], [_lead("a0", 12.0, 11.0)]]
    with pytest.raises(OpsError) as caught:
        average_component_rows(components)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_zero_comparator_year_gives_yearly_none_while_pooled_is_scored():
    perfect_2023 = [_lead("a0", 12.0, 12.0), _lead("a1", 4.0, 4.0), _lead("a2", 8.0, 8.0)]
    scores = score_blends([BLEND], _component_years(), INNER_YEARS,
        {"2023": perfect_2023, "2024": HISTORY_2024}, _score_fn, set())
    result = scores[0]
    assert result["yearly"]["2023"] is None
    assert result["yearly"]["2024"] is not None
    assert math.isfinite(result["blend_score"])
    # Hand-computed pooled blend: 0.6/sqrt(3) + 0.3*(23/27) + 0.1*(7/12) = 0.2*sqrt(3) + 113/360.
    assert result["blend_score"] == pytest.approx(0.2 * math.sqrt(3.0) + 113.0 / 360.0)
    with pytest.raises(ZeroComparatorError):
        _score_fn(_explicit_average(TARGETS_2023), perfect_2023)


def test_score_blends_hand_computed_numeric_oracle_single_year():
    weights = {"rmse": 0.5, "mae": 0.3, "large_move_underprediction": 0.2}
    threshold = 10.0
    component_years = {
        "c-one": {"2023": [_lead("e1", 4.0, 2.0), _lead("e2", 12.0, 8.0)]},
        "c-two": {"2023": [_lead("e1", 4.0, 4.0), _lead("e2", 12.0, 10.0)]},
    }
    history_rows_by_year = {"2023": [_lead("e1", 4.0, 5.0), _lead("e2", 12.0, 5.0)]}
    blend = {"name": "blend-oracle", "blend_id": "blend:oracle",
             "component_candidate_ids": ["c-one", "c-two"]}

    def score_fn(candidate_rows, history_rows):
        return blended_event_score(candidate_rows, history_rows, metric_weights=weights,
                                   large_move_threshold_pct=threshold)

    # Averaged candidate forecasts: e1 = (2+4)/2 = 3.0, e2 = (8+10)/2 = 9.0.
    # Candidate errors [1, 3]: rmse = sqrt(5), mae = 2.0, large (actual >= 10: e2) under = 12-9 = 3.0.
    # History errors [1, 7]: rmse = sqrt(25) = 5.0, mae = 4.0, large under = 12-5 = 7.0.
    # Normalized: rmse sqrt(5)/5, mae 2/4 = 0.5, large 3/7.
    expected_rmse = math.sqrt(5.0)
    expected_mae = 2.0
    expected_large = 3.0
    expected_norm_rmse = math.sqrt(5.0) / 5.0
    expected_norm_mae = 0.5
    expected_norm_large = 3.0 / 7.0
    expected_blend = 0.5 * expected_norm_rmse + 0.3 * expected_norm_mae + 0.2 * expected_norm_large

    scores = score_blends([blend], component_years, ["2023"], history_rows_by_year, score_fn, set())
    assert len(scores) == 1
    result = scores[0]
    assert result["candidate_id"] == "blend:oracle"
    assert result["kind"] == "blend"
    assert result["blend_name"] == "blend-oracle"

    assert result["metrics"]["rmse"] == pytest.approx(expected_rmse)
    assert result["metrics"]["mae"] == pytest.approx(expected_mae)
    assert result["metrics"]["large_move_underprediction"] == pytest.approx(expected_large)
    assert result["normalized"]["rmse"] == pytest.approx(expected_norm_rmse)
    assert result["normalized"]["mae"] == pytest.approx(expected_norm_mae)
    assert result["normalized"]["large_move_underprediction"] == pytest.approx(expected_norm_large)
    assert result["blend_score"] == pytest.approx(expected_blend)

    yearly_2023 = result["yearly"]["2023"]
    assert list(result["yearly"]) == ["2023"]
    assert yearly_2023["metrics"]["rmse"] == pytest.approx(expected_rmse)
    assert yearly_2023["metrics"]["mae"] == pytest.approx(expected_mae)
    assert yearly_2023["metrics"]["large_move_underprediction"] == pytest.approx(expected_large)
    assert yearly_2023["normalized"]["rmse"] == pytest.approx(expected_norm_rmse)
    assert yearly_2023["normalized"]["mae"] == pytest.approx(expected_norm_mae)
    assert yearly_2023["normalized"]["large_move_underprediction"] == pytest.approx(expected_norm_large)
    assert yearly_2023["blend_score"] == pytest.approx(expected_blend)
