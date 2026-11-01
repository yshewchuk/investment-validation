"""Independent matched-weight, clustered-bootstrap, abstention and poison oracles."""
import math
from copy import deepcopy

import numpy as np
import pytest

from experiments.size_paired_report import paired_report


def fixture():
    events = [{"event_id": f"e{i}", "ticker": "A" if i < 2 else "B", "event_date": day}
              for i, day in enumerate(["2023-01-20", "2023-03-14", "2023-04-17", "2024-07-05"])]
    targets = {f"e{i}": 2.0 + i * 4 for i in range(4)}
    challenger, incumbent = [], []
    for i, event in enumerate(events):
        for h in (1, 3, 5):
            key = {"event_id": event["event_id"], "horizon_sessions": h}
            challenger.append({**key, "predicted_absolute_move_pct": targets[event["event_id"]] + h / 10})
            incumbent.append({**key, "predicted_absolute_move_pct": targets[event["event_id"]] - 1,
                "abstention": None, "history_mean": 4.0, "quote_status": "present"})
    return events, challenger, incumbent, targets


def test_exact_pooled_difference_equal_event_weights_and_bootstrap_oracle():
    args = fixture()
    report = paired_report(*args)
    expected_rmse = math.sqrt((0.1 ** 2 + 0.3 ** 2 + 0.5 ** 2) / 3)
    assert report["difference"] == {"rmse": pytest.approx(expected_rmse - 1), "mae": pytest.approx(-0.7)}
    assert report["paired_events"] == 4 and len(report["paired_rows"]) == 12
    assert report["event_total_weight"] == 1 and report["horizon_weight"] == 1 / 3
    assert report["quarter_bootstrap"]["blocks"] == 3 and report["issuer_sensitivity"]["blocks"] == 2
    for interval in (report["quarter_bootstrap"], report["issuer_sensitivity"]):
        assert interval["resamples"] == 2000 and interval["seed"] == 0
        assert interval["percentile_95"]["mae"] == pytest.approx([-0.7, -0.7])
        assert interval["percentile_95"]["rmse"] == pytest.approx([expected_rmse - 1] * 2)
    assert report["metrics"]["challenger"]["bias"] == pytest.approx(0.3)
    assert report["metrics"]["incumbent"]["large_move_underprediction"] == 1
    assert paired_report(*args) == report


def test_cluster_sampling_recomputes_pooled_losses_on_identical_blocks():
    args = fixture()
    for i, row in enumerate(args[1]):
        row["predicted_absolute_move_pct"] += i / 3
    report = paired_report(*args)
    rows = report["paired_rows"]
    blocks = sorted({r["quarter"] for r in rows})
    rng = np.random.default_rng(0)
    rmse, mae = [], []
    for _ in range(2000):
        sample = [r for i in rng.integers(0, len(blocks), size=len(blocks)) for r in rows if r["quarter"] == blocks[i]]
        a = [r["challenger"] - r["absolute_move_pct"] for r in sample]
        b = [r["incumbent"] - r["absolute_move_pct"] for r in sample]
        rmse.append(math.sqrt(sum(v * v for v in a) / len(a)) - math.sqrt(sum(v * v for v in b) / len(b)))
        mae.append(sum(abs(v) for v in a) / len(a) - sum(abs(v) for v in b) / len(b))
    assert report["quarter_bootstrap"]["percentile_95"]["rmse"] == pytest.approx(np.percentile(rmse, [2.5, 97.5]))
    assert report["quarter_bootstrap"]["percentile_95"]["mae"] == pytest.approx(np.percentile(mae, [2.5, 97.5]))


def test_one_missing_horizon_excludes_whole_event_not_broad_sanity_reference():
    args = fixture()
    args[2][0].update(predicted_absolute_move_pct=None, abstention="unavailable_input")
    del args[3]["e1"]
    report = paired_report(*args)
    assert report["eligible_events"] == 4 and report["forecast_matched_events"] == 3
    assert report["paired_events"] == 2 and len(report["paired_rows"]) == 6
    assert {r["event_id"] for r in report["paired_rows"]} == {"e2", "e3"}
    assert report["label_unavailable_events"] == ["e1"]
    assert report["broad_coverage_metrics"]["history_mean"]["count"] == 9
    assert report["broad_coverage_metrics"]["challenger"]["count"] == 9
    assert report["final_read_authority"] == "none"


def test_challenger_abstention_is_counted_and_paired_metrics_keep_complete_events():
    args = fixture()
    args[1].pop(0)
    report = paired_report(*args)
    assert report["forecast_matched_events"] == 3
    assert report["paired_events"] == 3
    assert len(report["paired_rows"]) == 9
    assert report["challenger_abstentions"] == 1
    assert report["broad_coverage_metrics"]["challenger"]["count"] == 11
    assert report["metrics"]["challenger"]["count"] == 9
    assert sum(row["count"] for row in report["coverage"]
               if row["arm"] == "challenger" and row["forecast_status"] == "abstained") == 1


def test_large_evaluation_labels_never_use_training_mask_and_negative_incumbent_stays_raw():
    args = fixture()
    args[3]["e0"] = 250
    args[2][0]["predicted_absolute_move_pct"] = -10
    args[1][0]["predicted_absolute_move_pct"] = -2.0
    report = paired_report(*args)
    first = next(r for r in report["paired_rows"] if r["event_id"] == "e0" and r["horizon_sessions"] == 1)
    assert first["absolute_move_pct"] == 250 and first["incumbent"] == -10
    assert first["challenger"] == -2.0
    assert report["paired_events"] == 4


@pytest.mark.parametrize("kind", ["duplicate", "foreign", "nan", "bool"])
def test_poisoned_challenger_or_population_refused(kind):
    args = fixture()
    if kind == "duplicate":
        args[1].append(deepcopy(args[1][0]))
    elif kind == "foreign":
        args[1][0]["event_id"] = "not-admitted"
    else:
        args[1][0]["predicted_absolute_move_pct"] = {"nan": float("nan"), "bool": True}[kind]
    with pytest.raises(ValueError):
        paired_report(*args)


def test_empty_paired_population_is_reported_not_replaced():
    args = fixture()
    for row in args[2]:
        row.update(predicted_absolute_move_pct=None, abstention="insufficient_training")
    report = paired_report(*args)
    assert report["paired_events"] == 0
    assert report["comparison_status"] == "unavailable_no_complete_matched_events"
    assert "difference" not in report and "quarter_bootstrap" not in report
    assert report["broad_coverage_metrics"]["history_mean"]["count"] == 12


def test_single_lead_horizon_report_covers_only_horizon_one():
    events = [{"event_id": "e0", "ticker": "A", "event_date": "2023-01-20"}]
    targets = {"e0": 2.0}
    challenger = [{"event_id": "e0", "horizon_sessions": 1,
                   "predicted_absolute_move_pct": 2.1}]
    incumbent = [{"event_id": "e0", "horizon_sessions": 1,
                  "predicted_absolute_move_pct": 1.0, "abstention": None,
                  "history_mean": 4.0, "quote_status": "present"}]
    report = paired_report(events, challenger, incumbent, targets, horizons=(1,))
    assert report["forecast_matched_events"] == 1
    assert report["paired_events"] == 1
    assert report["metrics"]["challenger"]["rmse"] == pytest.approx(0.1)
    assert report["metrics"]["incumbent"]["rmse"] == pytest.approx(1.0)
    assert report["difference"]["rmse"] == pytest.approx(-0.9)
    assert report["difference"]["mae"] == pytest.approx(-0.9)
    assert {row["horizon_sessions"] for row in report["paired_rows"]} == {1}
    assert report["horizon_weight"] == 1
    assert set(report["per_horizon"]) == {"1"}


def test_two_lead_horizon_subset_report_weights_and_metrics_independently():
    events = [{"event_id": "e0", "ticker": "A", "event_date": "2023-01-20"}]
    targets = {"e0": 2.0}
    challenger = [
        {"event_id": "e0", "horizon_sessions": 1, "predicted_absolute_move_pct": 2.1},
        {"event_id": "e0", "horizon_sessions": 3, "predicted_absolute_move_pct": 2.6},
    ]
    incumbent = [
        {"event_id": "e0", "horizon_sessions": 1, "predicted_absolute_move_pct": 1.0,
         "abstention": None, "history_mean": 4.0, "quote_status": "present"},
        {"event_id": "e0", "horizon_sessions": 3, "predicted_absolute_move_pct": 1.0,
         "abstention": None, "history_mean": 4.0, "quote_status": "present"},
    ]
    report = paired_report(events, challenger, incumbent, targets, horizons=(1, 3))
    rows = [r for r in report["paired_rows"] if r["event_id"] == "e0"]
    expected_rmse = math.sqrt((0.1 ** 2 + 0.6 ** 2) / 2)
    assert len(rows) == 2
    assert report["horizon_weight"] == 0.5
    assert set(report["per_horizon"]) == {"1", "3"}
    assert report["metrics"]["challenger"]["rmse"] == pytest.approx(expected_rmse)
    assert report["metrics"]["incumbent"]["rmse"] == pytest.approx(1.0)
    assert report["difference"]["rmse"] == pytest.approx(expected_rmse - 1)
    assert report["difference"]["mae"] == pytest.approx(-0.65)


def test_legacy_horizon_order_accepted_and_per_horizon_emitted_ascending():
    args = fixture()
    report = paired_report(*args, horizons=(5, 3, 1))
    assert list(report["per_horizon"]) == ["1", "3", "5"]
    assert paired_report(*args) == report
