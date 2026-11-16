"""Hand-computed oracles for signed per-event training-mean selection."""
import math

import pytest

from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_trials as trials
from experiments.iv_selection import signed_event_score, training_mean_rows
from experiments.magnitude_search_plan import PLAN_V3, cohort_key, trial_key
from experiments.magnitude_search_selection import ZeroComparatorError

TARGET = {"column": "crush_pct_iv30", "prediction_column": "predicted_crush_pct_iv30"}
WEIGHTS = {"rmse": 0.67, "mae": 0.33}
POLICY = {"metric_weights": {"rmse": 0.6, "mae": 0.3,
                             "large_move_underprediction": 0.1},
          "lead_weights": {"1": 1.0, "3": 0.0, "5": 0.0},
          "large_move_threshold_pct": 10.0, "tie_band_floor": 0.005}


def _row(event_id, actual, predicted, horizon=1):
    return {"event_id": event_id, "horizon_sessions": horizon,
            "crush_pct_iv30": actual, "predicted_crush_pct_iv30": predicted}


def _rows(actuals, predictions, horizon=1):
    return [_row(event_id, actual, predictions[event_id], horizon)
            for event_id, actual in actuals.items()]


def test_signed_score_hand_oracle():
    actuals = {"e0": -20.0, "e1": -5.0, "e2": 10.0}
    candidate = _rows(actuals, {"e0": -18.0, "e1": -10.0, "e2": 4.0})
    baseline = _rows(actuals, {"e0": -5.0, "e1": -5.0, "e2": -5.0})
    # Candidate errors are -2, 5, 6; baseline errors are -15, 0, 15.
    rmse = math.sqrt((2.0**2 + 5.0**2 + 6.0**2) / 3)
    mae = (2.0 + 5.0 + 6.0) / 3
    rmse0 = math.sqrt((15.0**2 + 0.0**2 + 15.0**2) / 3)
    mae0 = (15.0 + 0.0 + 15.0) / 3
    score = signed_event_score(candidate, baseline, metric_weights=WEIGHTS,
                               target=TARGET)
    assert score["metrics"] == {"count": 3, "rmse": pytest.approx(rmse),
                                "mae": pytest.approx(mae)}
    assert score["normalized"] == {"rmse": pytest.approx(rmse / rmse0),
                                   "mae": pytest.approx(mae / mae0)}
    assert score["blend_score"] == pytest.approx(
        0.67 * rmse / rmse0 + 0.33 * mae / mae0)


def test_training_mean_rows_uses_fold_training_mean():
    inputs = {"training": [{"event_id": "t0", "crush_pct_iv30": -10.0},
                           {"event_id": "t1", "crush_pct_iv30": -20.0},
                           {"event_id": "t2", "crush_pct_iv30": -30.0}]}
    validation = {"targets": [{"event_id": "e0", "crush_pct_iv30": -7.0},
                              {"event_id": "e1", "crush_pct_iv30": 12.0}]}
    rows = training_mean_rows(inputs, validation, target=TARGET)
    assert rows == [
        {"event_id": "e0", "horizon_sessions": 1,
         "crush_pct_iv30": -7.0, "predicted_crush_pct_iv30": -20.0},
        {"event_id": "e1", "horizon_sessions": 1,
         "crush_pct_iv30": 12.0, "predicted_crush_pct_iv30": -20.0},
    ]


def test_training_mean_rows_refuses_empty_training():
    with pytest.raises(ValueError):
        training_mean_rows({"training": []}, {"targets": []}, target=TARGET)


def test_training_mean_rows_refuses_nonfinite_label():
    with pytest.raises(ValueError):
        training_mean_rows({"training": [{"crush_pct_iv30": float("nan")}]},
                           {"targets": []}, target=TARGET)


def test_zero_baseline_nonzero_error_raises():
    actuals = {"e0": -20.0, "e1": -5.0, "e2": 10.0}
    baseline = _rows(actuals, actuals)
    candidate = _rows(actuals, {"e0": -18.0, "e1": -10.0, "e2": 4.0})
    with pytest.raises(ZeroComparatorError):
        signed_event_score(candidate, baseline, metric_weights=WEIGHTS,
                           target=TARGET)
    parity = signed_event_score(baseline, baseline, metric_weights=WEIGHTS,
                                target=TARGET)
    assert parity["normalized"] == {"rmse": 1.0, "mae": 1.0}
    assert parity["blend_score"] == pytest.approx(1.0)


def test_only_lead_one_rows_score():
    actuals = {"e0": -20.0, "e1": -5.0, "e2": 10.0}
    candidate = _rows(actuals, {"e0": -18.0, "e1": -10.0, "e2": 4.0})
    baseline = _rows(actuals, {"e0": -5.0, "e1": -5.0, "e2": -5.0})
    expected = signed_event_score(candidate, baseline, metric_weights=WEIGHTS,
                                  target=TARGET)
    for rows in (candidate, baseline):
        rows.extend([
            _row("e0", -20.0, 1000.0, horizon=3),
            _row("e0", -20.0, -1000.0, horizon=5),
        ])
    assert signed_event_score(candidate, baseline, metric_weights=WEIGHTS,
                              target=TARGET) == expected


@pytest.mark.parametrize("candidate,baseline,weights", [
    ([], [], WEIGHTS),
    ([_row("e0", -20.0, -18.0, horizon=3)],
     [_row("e0", -20.0, -5.0, horizon=3)], WEIGHTS),
    ([_row("e0", -20.0, -18.0), _row("e0", -20.0, -18.0)],
     [_row("e0", -20.0, -5.0)], WEIGHTS),
    ([_row("e0", -20.0, -18.0)], [_row("e1", -20.0, -5.0)], WEIGHTS),
    ([_row("e0", -20.0, -18.0)], [_row("e0", -19.0, -5.0)], WEIGHTS),
    ([_row("e0", -20.0, math.nan)], [_row("e0", -20.0, -5.0)], WEIGHTS),
    ([_row("e0", -20.0, -18.0)], [_row("e0", -20.0, -5.0)],
     {"rmse": 0.67, "mae": 0.33, "large_move_underprediction": 0.0}),
    ([_row("e0", -20.0, -18.0)], [_row("e0", -20.0, -5.0)], {"rmse": 0.67}),
])
def test_signed_event_score_refuses_malformed_inputs(candidate, baseline, weights):
    with pytest.raises(ValueError):
        signed_event_score(candidate, baseline, metric_weights=weights,
                           target=TARGET)


def test_negative_and_signed_targets_normalize_symmetrically():
    actuals = {"e0": -20.0, "e1": -5.0, "e2": 10.0}
    predictions = {"e0": -18.0, "e1": -10.0, "e2": 4.0}
    baseline = {"e0": -5.0, "e1": -5.0, "e2": -5.0}
    original = signed_event_score(_rows(actuals, predictions),
                                  _rows(actuals, baseline),
                                  metric_weights=WEIGHTS, target=TARGET)
    negated = signed_event_score(
        _rows({key: -value for key, value in actuals.items()},
              {key: -value for key, value in predictions.items()}),
        _rows({key: -value for key, value in actuals.items()},
              {key: -value for key, value in baseline.items()}),
        metric_weights=WEIGHTS, target=TARGET)
    assert negated["blend_score"] == pytest.approx(original["blend_score"])


def test_freeze_dispatches_v2_plan_to_signed_blend(monkeypatch):
    plan = {"schema_version": PLAN_V3, "plan_id": "iv-plan",
        "request": {"selection": {"metric_weights": dict(WEIGHTS),
            "lead_weights": {"1": 1.0, "3": 0.0, "5": 0.0},
            "tie_band_floor": 0.005}},
        "selection": {"metric": "blend_rmse_mae",
            "ties": "lowest_score_then_candidate_id_ascending"},
        "target": TARGET,
        "candidates": [{"candidate_id": "a"}, {"candidate_id": "b"}],
        "folds": [{"outer": {"year": "2024"}, "inner": [{"year": "2023"}]}]}
    cohort_id = cohort_key(plan, "2024", "2023")
    actuals = {"e0": -20.0, "e1": -5.0, "e2": 10.0}
    predictions = {
        "a": {"e0": -18.0, "e1": -10.0, "e2": 4.0},
        "b": {"e0": 0.0, "e1": 0.0, "e2": 0.0},
    }
    inputs = {"training": [{"event_id": "t0", "crush_pct_iv30": -10.0},
                           {"event_id": "t1", "crush_pct_iv30": -30.0}]}
    validation = {"targets": [{"event_id": event_id, "crush_pct_iv30": actual}
                              for event_id, actual in actuals.items()]}
    recorder = object.__new__(trials.SearchTrials)
    monkeypatch.setattr(recorder, "_admit", lambda: plan)

    def terminal(plan_arg, outer_year, inner_year, candidate_id):
        rows = _rows(actuals, predictions[candidate_id])
        return trial_key(plan_arg, outer_year, inner_year, candidate_id), {
            "status": "succeeded", "predictions": rows}

    def read(slot):
        if slot.startswith("inputs:"):
            return {"cohort": cohort_id}
        if slot == "validation:" + cohort_id:
            return validation
        raise AssertionError(slot)

    monkeypatch.setattr(recorder, "_terminal", terminal)
    monkeypatch.setattr(recorder, "_read", read)
    monkeypatch.setattr(recorder, "_cohort", lambda admitted: inputs)
    monkeypatch.setattr(recorder, "_validation", lambda *args: None)
    monkeypatch.setattr(recorder, "_append", lambda slot, body, **kwargs: body)

    frozen = recorder.freeze()
    selection = frozen["selections"][0]
    # The training mean is -20, so baseline errors are 0, 15, 30.
    # Candidate a errors are -2, 5, 6; candidate b errors are -20, -5, 10.
    rmse0 = math.sqrt((0.0**2 + 15.0**2 + 30.0**2) / 3)
    mae0 = (0.0 + 15.0 + 30.0) / 3
    expected_a = (0.67 * math.sqrt((2.0**2 + 5.0**2 + 6.0**2) / 3) / rmse0
                  + 0.33 * ((2.0 + 5.0 + 6.0) / 3) / mae0)
    expected_b = (0.67 * math.sqrt((20.0**2 + 5.0**2 + 10.0**2) / 3) / rmse0
                  + 0.33 * ((20.0 + 5.0 + 10.0) / 3) / mae0)
    scores = {item["candidate_id"]: item for item in selection["scores"]}
    assert expected_a < expected_b
    assert selection["selected_candidate_id"] == "a"
    assert scores["a"]["blend_score"] == pytest.approx(expected_a)
    assert scores["b"]["blend_score"] == pytest.approx(expected_b)
    assert selection["history_mean"]["blend_score"] == pytest.approx(1.0)
    for item in scores.values():
        assert "large_move_underprediction" not in item["normalized"]


def test_freeze_zero_baseline_refuses_variant_failed(monkeypatch):
    plan = {"schema_version": PLAN_V3, "plan_id": "iv-zero-plan",
        "request": {"selection": {"metric_weights": dict(WEIGHTS),
            "lead_weights": {"1": 1.0, "3": 0.0, "5": 0.0},
            "tie_band_floor": 0.005}},
        "selection": {"metric": "blend_rmse_mae",
            "ties": "lowest_score_then_candidate_id_ascending"},
        "target": TARGET,
        "candidates": [{"candidate_id": "a"}],
        "folds": [{"outer": {"year": "2024"}, "inner": [{"year": "2023"}]}]}
    cohort_id = cohort_key(plan, "2024", "2023")
    actuals = {"e0": -20.0, "e1": -20.0}
    predictions = {"a": {"e0": -10.0, "e1": -20.0}}
    inputs = {"training": [{"event_id": "t0", "crush_pct_iv30": -10.0},
                           {"event_id": "t1", "crush_pct_iv30": -30.0}]}
    validation = {"targets": [{"event_id": event_id, "crush_pct_iv30": actual}
                              for event_id, actual in actuals.items()]}
    recorder = object.__new__(trials.SearchTrials)
    monkeypatch.setattr(recorder, "_admit", lambda: plan)

    def terminal(plan_arg, outer_year, inner_year, candidate_id):
        rows = _rows(actuals, predictions[candidate_id])
        return trial_key(plan_arg, outer_year, inner_year, candidate_id), {
            "status": "succeeded", "predictions": rows}

    def read(slot):
        if slot.startswith("inputs:"):
            return {"cohort": cohort_id}
        if slot == "validation:" + cohort_id:
            return validation
        raise AssertionError(slot)

    monkeypatch.setattr(recorder, "_terminal", terminal)
    monkeypatch.setattr(recorder, "_read", read)
    monkeypatch.setattr(recorder, "_cohort", lambda admitted: inputs)
    monkeypatch.setattr(recorder, "_validation", lambda *args: None)
    monkeypatch.setattr(recorder, "_append", lambda slot, body, **kwargs: body)

    with pytest.raises(OpsError) as caught:
        recorder.freeze()
    assert caught.value.problem.code == "EXPERIMENT_VARIANT_FAILED"


def test_v1_plan_still_uses_magnitude_blend(monkeypatch):
    plan = {"schema_version": PLAN_V3, "plan_id": "magnitude-plan",
        "request": {"selection": dict(POLICY)},
        "selection": {"ties": "lowest_score_then_candidate_id_ascending"},
        "candidates": [{"candidate_id": "a-near-tie"}, {"candidate_id": "z-best"}],
        "folds": [{"outer": {"year": "2024"}, "inner": [{"year": "2023"}]}]}
    cohort_id = cohort_key(plan, "2024", "2023")
    actuals = {"e0": 12.0, "e1": 4.0, "e2": 8.0}
    predictions = {
        "a-near-tie": {"e0": 10.99, "e1": 6.0, "e2": 10.0},
        "z-best": {"e0": 11.0, "e1": 6.0, "e2": 10.0},
    }
    inputs = {"training": [{"absolute_move_pct": 2.0}], "features": [
        {"event_id": event_id, "horizon_sessions": 1, "n_prior": 1,
         "mean_prior_abs_move": value}
        for event_id, value in (("e0", 10.0), ("e1", 6.0), ("e2", 10.0))]}
    validation = {"targets": [
        {"event_id": event_id, "absolute_move_pct": actual}
        for event_id, actual in actuals.items()]}
    recorder = object.__new__(trials.SearchTrials)
    monkeypatch.setattr(recorder, "_admit", lambda: plan)

    def terminal(plan_arg, outer_year, inner_year, candidate_id):
        rows = [{"event_id": event_id, "horizon_sessions": 1,
                 "absolute_move_pct": actuals[event_id],
                 "predicted_absolute_move_pct": predictions[candidate_id][event_id]}
                for event_id in actuals]
        return trial_key(plan_arg, outer_year, inner_year, candidate_id), {
            "status": "succeeded", "predictions": rows}

    def read(slot):
        if slot.startswith("inputs:"):
            return {"cohort": cohort_id}
        if slot == "validation:" + cohort_id:
            return validation
        raise AssertionError(slot)

    monkeypatch.setattr(recorder, "_terminal", terminal)
    monkeypatch.setattr(recorder, "_read", read)
    monkeypatch.setattr(recorder, "_cohort", lambda admitted: inputs)
    monkeypatch.setattr(recorder, "_validation", lambda *args: None)
    monkeypatch.setattr(recorder, "_append", lambda slot, body, **kwargs: body)

    frozen = recorder.freeze()
    selection = frozen["selections"][0]
    assert "large_move_underprediction" in selection["scores"][0]["normalized"]
    assert selection["selected_candidate_id"] == "z-best"
