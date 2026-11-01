"""Independent signed paired/broad loss and complete-event coverage expectations."""
from copy import deepcopy

import pytest

from engine.v2.foundation import content_hash
from experiments.iv_crush_paired_report import paired_report


def case():
    events = [{"event_id": e, "ticker": e.upper(), "event_date": f"202{year}-06-02"}
              for e, year in (("a", 3), ("b", 4))]
    targets = {"a": -140., "b": 20.}
    challenger = [{"event_id": e, "horizon_sessions": h, "predicted_crush_pct_iv30": targets[e] + 2.}
                  for e in targets for h in (1, 3, 5)]
    incumbent = [{"event_id": e, "horizon_sessions": h, "predicted_crush_pct_iv30": targets[e] + 4.,
                  "abstention": None, "training_mean": -60.} for e in targets for h in (1, 3, 5)]
    return events, challenger, incumbent, targets


def test_signed_matched_losses_are_independent_of_order_and_have_no_promotion_rule():
    args = case()
    result = paired_report(*args)
    assert result["result_id"] == content_hash({key: value for key, value in result.items() if key != "result_id"})
    assert result["metrics"]["challenger"] == {"count": 6, "mae": 2., "rmse": 2., "bias": 2.}
    assert result["metrics"]["incumbent"] == {"count": 6, "mae": 4., "rmse": 4., "bias": 4.}
    assert result["metrics"]["training_mean"] == {"count": 6, "mae": 80., "rmse": 80., "bias": 0.}
    assert result["difference"] == {"mae": -2., "rmse": -2.}
    assert result["paired_events"] == result["eligible_events"] == 2
    assert result["horizon_weight"] == 1 / 3 and result["event_total_weight"] == 1
    assert result["final_read_authority"] == "none"
    assert not {"winner", "promotion", "large_move_threshold_pct", "quarter_bootstrap"} & set(result)
    shuffled = ([*reversed(args[0])], [*reversed(args[1])], [*reversed(args[2])], args[3])
    assert paired_report(*shuffled) == result
    assert all(row["forecast_status"] == "predicted" and row["paired_forecasts"] for row in result["coverage"])


def test_one_horizon_abstention_excludes_whole_event_without_hiding_broad_losses():
    args = case()
    args[2][0].update(abstention="unavailable_input", predicted_crush_pct_iv30=None)
    result = paired_report(*args)
    assert result["paired_events"] == result["forecast_matched_events"] == 1
    assert {row["event_id"] for row in result["paired_rows"]} == {"b"}
    assert result["metrics"]["challenger"]["count"] == 3
    assert result["broad_coverage_metrics"]["challenger"]["count"] == 6
    assert sum(row["count"] for row in result["coverage"] if row["arm"] == "incumbent") == 6
    assert sum(row["count"] for row in result["coverage"] if row["forecast_status"] == "unavailable_input") == 1


def test_no_matched_events_remains_descriptive_and_retains_training_mean():
    args = case()
    for row in args[2]:
        row.update(abstention="insufficient_training", predicted_crush_pct_iv30=None)
    result = paired_report(*args)
    assert result["paired_events"] == 0 and result["paired_rows"] == []
    assert result["comparison_status"] == "unavailable_no_complete_matched_events"
    assert "metrics" not in result and "difference" not in result
    assert result["broad_coverage_metrics"]["training_mean"]["mae"] == 80.


def test_selected_horizon_weight_and_original_inputs_preserved():
    args = case()
    args = (args[0], [row for row in args[1] if row["horizon_sessions"] == 1],
            [row for row in args[2] if row["horizon_sessions"] == 1], args[3])
    before = deepcopy(args)
    result = paired_report(*args, horizons=[1])
    assert result["horizon_weight"] == 1 and result["metrics"]["challenger"]["count"] == 2
    assert list(result["per_horizon"]) == ["1"] and args == before


@pytest.mark.parametrize("mutation", ["missing_event", "duplicate_event", "missing_target", "extra_target",
    "bad_target", "bool_target", "duplicate_challenger", "duplicate_incumbent", "missing_challenger",
    "missing_incumbent", "wrong_horizon", "bool_horizon", "missing_prediction", "unknown_reason",
    "unexpected_abstention_value", "bad_mean", "bool_mean", "bad_prediction"])
def test_malformed_or_incomplete_reports_refuse(mutation):
    events, challenger, incumbent, targets = case()
    if mutation == "missing_event":
        events.pop()
    elif mutation == "duplicate_event":
        events.append(deepcopy(events[0]))
    elif mutation == "missing_target":
        del targets["a"]
    elif mutation == "extra_target":
        targets["c"] = 0.
    elif mutation in ("bad_target", "bool_target"):
        targets["a"] = float("inf") if mutation == "bad_target" else True
    elif mutation == "duplicate_challenger":
        challenger[1] = deepcopy(challenger[0])
    elif mutation == "duplicate_incumbent":
        incumbent[1] = deepcopy(incumbent[0])
    elif mutation == "missing_challenger":
        challenger.pop()
    elif mutation == "missing_incumbent":
        incumbent.pop()
    elif mutation in ("wrong_horizon", "bool_horizon"):
        incumbent[0]["horizon_sessions"] = 9 if mutation == "wrong_horizon" else True
    elif mutation == "missing_prediction":
        incumbent[0]["predicted_crush_pct_iv30"] = None
    elif mutation == "unknown_reason":
        incumbent[0].update(abstention="other", predicted_crush_pct_iv30=None)
    elif mutation == "unexpected_abstention_value":
        incumbent[0]["abstention"] = "unavailable_input"
    elif mutation in ("bad_mean", "bool_mean"):
        incumbent[0]["training_mean"] = float("nan") if mutation == "bad_mean" else True
    else:
        incumbent[0]["predicted_crush_pct_iv30"] = float("inf")
    with pytest.raises(ValueError):
        paired_report(events, challenger, incumbent, targets)
