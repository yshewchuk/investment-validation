import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments.magnitude_report import magnitude_metrics
from experiments.magnitude_search_blend_outer import average_forecasts, blend_outer_result


def _row(event_id, predicted, actual=None, horizon=1):
    row = {"event_id": event_id, "horizon_sessions": horizon, "predicted_absolute_move_pct": predicted}
    if actual is not None:
        row["absolute_move_pct"] = actual
    return row


def _conflict(call):
    with pytest.raises(OpsError) as caught:
        call()
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_average_forecasts_is_the_fsum_mean_per_key():
    result = average_forecasts([[_row("a", 1.0)], [_row("a", 3.0)], [_row("a", 2.0)]])
    assert result == [_row("a", 2.0)]


def test_average_forecasts_keeps_a_negative_mean():
    result = average_forecasts([[_row("a", -1.0)], [_row("a", -2.0)]])
    assert result[0]["predicted_absolute_move_pct"] == -1.5


def test_average_forecasts_keeps_the_first_list_order_and_other_fields():
    first = [_row("b", 1.0, actual=5.0), _row("a", 2.0, actual=6.0)]
    second = [_row("a", 4.0, actual=6.0), _row("b", 2.0, actual=5.0)]
    assert average_forecasts([first, second]) == [
        _row("b", 1.5, actual=5.0),
        _row("a", 3.0, actual=6.0),
    ]


def test_average_forecasts_refuses_misaligned_event_keys():
    _conflict(lambda: average_forecasts([[_row("a", 1.0)], [_row("b", 1.0)]]))


def test_average_forecasts_refuses_a_target_mismatch():
    _conflict(lambda: average_forecasts([[_row("a", 1.0, actual=2.0)], [_row("a", 1.0, actual=3.0)]]))


def test_average_forecasts_refuses_duplicate_keys():
    _conflict(lambda: average_forecasts([[_row("a", 1.0), _row("a", 2.0)], [_row("a", 1.0)]]))


def _component(candidate_id, predicted, event_count=1, fit_count=2, rows=None):
    return {"result_id": "result-" + candidate_id,
            "predictions": rows if rows is not None else [_row("e", predicted, actual=2.0)],
            "event_count": event_count,
            "evaluation_inputs": {"omitted_labels": []},
            "prospective_candidate_id": "a",
            "fit_attempt_count": fit_count,
            "holdout_results": {"random": "excluded", "rolling": "excluded"}}


PLAN = {"plan_id": "plan", "horizons": [1]}
BLEND = {"blend_id": "blend_ab", "name": "ab", "component_candidate_ids": ["b", "a"]}


def test_blend_outer_metrics_are_real_metrics_of_averaged_two_event_forecasts():
    first = [_row("e1", 1.0, actual=2.0), _row("e2", 5.0, actual=4.0)]
    second = [_row("e1", 3.0, actual=2.0), _row("e2", 7.0, actual=4.0)]
    result = blend_outer_result(PLAN, BLEND, "2024",
                                [_component("a", None, event_count=2, rows=first),
                                 _component("b", None, event_count=2, rows=second)])
    # Hand-averaged: e1 = (1+3)/2 = 2.0 vs actual 2.0; e2 = (5+7)/2 = 6.0 vs actual 4.0.
    expected_predictions = [_row("e1", 2.0, actual=2.0), _row("e2", 6.0, actual=4.0)]
    expected_metrics = magnitude_metrics(expected_predictions)
    assert result["predictions"] == expected_predictions
    assert result["metrics"] == expected_metrics
    # Independent oracle: errors are 0 and 2, so RMSE = sqrt(2), MAE = 1, count = 2.
    assert result["metrics"]["count"] == 2
    assert result["metrics"]["rmse"] == pytest.approx(2 ** 0.5)
    assert result["metrics"]["mae"] == pytest.approx(1.0)
    assert result["per_horizon"][0] == {"horizon_sessions": 1, **expected_metrics}


def test_blend_outer_result_binds_components_and_averages_forecasts():
    result = blend_outer_result(PLAN, BLEND, "2024", [_component("a", 1.0), _component("b", 3.0)])
    assert result["schema_version"] == "magnitude_search_outer_blend_result.v1"
    assert result["plan_id"] == "plan"
    assert result["outer_year"] == "2024"
    assert result["candidate"] == {"candidate_id": "blend_ab", "kind": "blend", "blend_name": "ab",
                                   "component_candidate_ids": ["a", "b"]}
    assert result["component_refits"] == [{"candidate_id": "a", "result_id": "result-a"},
                                          {"candidate_id": "b", "result_id": "result-b"}]
    assert result["predictions"] == [_row("e", 2.0, actual=2.0)]
    assert result["forecasts_id"] == content_hash([_row("e", 2.0)])
    assert result["event_count"] == 1
    assert result["forecast_count"] == 1
    assert result["fit_attempt_count"] == 4
    single = [_row("e", 2.0, actual=2.0)]
    assert result["metrics"] == magnitude_metrics(single)
    assert result["per_horizon"] == [{"horizon_sessions": 1, **magnitude_metrics(single)}]
    assert result["prospective_candidate_id"] == "a"
    assert result["evaluation_inputs"] == {"omitted_labels": []}
    assert result["holdout_results"] == {"random": "excluded", "rolling": "excluded"}
    assert result["evaluation_authority"] == "selected_outer_fold_only"
    assert result["final_read_authority"] == "none"
    assert result["result_id"] == content_hash({k: v for k, v in result.items() if k != "result_id"})


def test_blend_outer_result_refuses_component_event_count_mismatch():
    _conflict(lambda: blend_outer_result(
        PLAN, BLEND, "2024", [_component("a", 1.0), _component("b", 3.0, event_count=2)]))
