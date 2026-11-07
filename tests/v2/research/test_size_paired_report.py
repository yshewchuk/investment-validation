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


def test_large_evaluation_labels_never_use_training_mask_and_negative_incumbent_stays_raw():
    args = fixture()
    args[3]["e0"] = 250
    args[2][0]["predicted_absolute_move_pct"] = -10
    report = paired_report(*args)
    first = next(r for r in report["paired_rows"] if r["event_id"] == "e0" and r["horizon_sessions"] == 1)
    assert first["absolute_move_pct"] == 250 and first["incumbent"] == -10
    assert report["paired_events"] == 4


@pytest.mark.parametrize("kind", ["duplicate", "missing", "foreign", "nan", "bool"])
def test_poisoned_challenger_or_population_refused(kind):
    args = fixture()
    if kind == "duplicate":
        args[1].append(deepcopy(args[1][0]))
    elif kind == "missing":
        args[1].pop()
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


def test_negative_challenger_is_preserved_but_negative_target_refuses():
    args = fixture()
    args[1][0]["predicted_absolute_move_pct"] = -2.0
    result = paired_report(*args)
    assert result["paired_rows"][0]["challenger"] == -2.0
    args[3]["e0"] = -2.0
    with pytest.raises(ValueError, match="invalid magnitude target"):
        paired_report(*args)


def test_implied_move_uses_percent_points_and_keeps_full_metrics_and_bootstrap():
    events, challenger, incumbent, targets = fixture()
    implied = {(event, horizon): value
               for event, value in (("e0", 2.5), ("e1", 5.5)) for horizon in (1, 3, 5)}
    old = paired_report(events, challenger, incumbent, targets)
    report = paired_report(events, challenger, incumbent, targets, implied_moves=implied)
    rows = {(row["event_id"], row["horizon_sessions"]): row for row in report["paired_rows"]}

    assert report["schema_version"] == "size_paired_report.v2"
    assert report["implied_move_coverage"] == {"available_rows": 6, "total_paired_rows": 12}
    assert rows["e0", 1]["implied_move"] == 2.5
    assert rows["e1", 3]["implied_move"] == 5.5
    assert rows["e2", 1]["implied_move"] is None
    implied_metrics = report["metrics"]["implied_move"]
    assert implied_metrics["count"] == 6
    assert implied_metrics["rmse"] == pytest.approx(0.5)
    assert implied_metrics["mae"] == pytest.approx(0.5)
    assert implied_metrics["bias"] == pytest.approx(0.0)

    like = report["like_for_like"]
    assert set(like["metrics"]) == {"challenger", "incumbent", "history_mean", "implied_move"}
    assert like["metrics"]["implied_move"]["rmse"] == pytest.approx(0.5)
    assert like["metrics"]["implied_move"]["mae"] == pytest.approx(0.5)
    assert like["metrics"]["implied_move"]["bias"] == pytest.approx(0.0)
    assert like["metrics"]["challenger"]["rmse"] == pytest.approx(math.sqrt(0.35 / 3))
    assert like["metrics"]["challenger"]["mae"] == pytest.approx(0.3)
    assert like["metrics"]["incumbent"]["rmse"] == pytest.approx(1.0)
    assert like["metrics"]["history_mean"]["rmse"] == pytest.approx(2.0)
    assert like["metrics"]["implied_move"]["count"] == 6

    bootstrap = report["quarter_bootstrap"]["implied_move"]
    assert bootstrap == like["quarter_bootstrap"]
    assert bootstrap["blocks"] == 1 and bootstrap["events"] == 2
    assert bootstrap["resamples"] == 2000 and bootstrap["seed"] == 0
    assert set(bootstrap["percentile_95"]) == {"challenger", "incumbent", "history_mean", "implied_move"}
    assert bootstrap["percentile_95"]["implied_move"]["rmse"] == pytest.approx([0.5, 0.5])
    assert bootstrap["percentile_95"]["implied_move"]["mae"] == pytest.approx([0.5, 0.5])
    assert bootstrap["difference_percentile_95"]["implied_move_minus_incumbent"] == {
        "rmse": pytest.approx([-0.5, -0.5]), "mae": pytest.approx([-0.5, -0.5])}

    for arm in ("challenger", "incumbent", "history_mean"):
        assert report["metrics"][arm] == old["metrics"][arm]
    assert report["difference"] == old["difference"]
    for key in ("cluster", "blocks", "events", "resamples", "seed"):
        assert report["quarter_bootstrap"][key] == old["quarter_bootstrap"][key]
    assert report["quarter_bootstrap"]["percentile_95"] == old["quarter_bootstrap"]["percentile_95"]
    assert report["issuer_sensitivity"] == old["issuer_sensitivity"]
    for year in old["per_year"]:
        for arm in ("challenger", "incumbent", "history_mean"):
            assert report["per_year"][year]["arms"][arm] == old["per_year"][year]["arms"][arm]
    assert report["per_year"]["2023"]["arms"]["implied_move"]["rmse"] == pytest.approx(0.5)
    assert report["per_year"]["2024"]["arms"]["implied_move"] is None
    assert report["per_year"]["2024"]["implied_move_coverage"] == {
        "available_rows": 0, "total_paired_rows": 3}
    assert report == paired_report(events, challenger, incumbent, targets, implied_moves=implied)


def test_implied_move_nulls_abstain_and_zero_coverage_is_explicit():
    events, challenger, incumbent, targets = fixture()
    implied = {(f"e{i}", horizon): None for i in range(4) for horizon in (1, 3, 5)}
    report = paired_report(events, challenger, incumbent, targets, implied_moves=implied)

    assert report["implied_move_coverage"] == {"available_rows": 0, "total_paired_rows": 12}
    assert len(report["paired_rows"]) == 12
    assert all(row["implied_move"] is None for row in report["paired_rows"])
    assert report["metrics"]["implied_move"] is None
    assert report["like_for_like"]["status"] == "unavailable_no_implied_move_rows"
    assert report["like_for_like"]["metrics"] is None
    assert report["like_for_like"]["quarter_bootstrap"] is None
    assert report["quarter_bootstrap"]["implied_move"] is None
    for breakdown in report["per_year"].values():
        assert breakdown["arms"]["implied_move"] is None
        assert breakdown["implied_move_coverage"]["available_rows"] == 0
    for arm in ("challenger", "incumbent", "history_mean"):
        assert report["metrics"][arm]["count"] == 12
    assert report["metrics"] == paired_report(events, challenger, incumbent, targets)["metrics"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, True, "2"])
def test_implied_move_rejects_nonfinite_negative_or_non_numeric_values(value):
    events, challenger, incumbent, targets = fixture()
    with pytest.raises(ValueError, match="invalid implied move"):
        paired_report(events, challenger, incumbent, targets,
                      implied_moves={("e0", 1): value})


def test_implied_move_rejects_keys_outside_the_paired_population():
    events, challenger, incumbent, targets = fixture()
    with pytest.raises(ValueError, match="implied move coverage differs"):
        paired_report(events, challenger, incumbent, targets,
                      implied_moves={("not-admitted", 1): 2.5})


def test_implied_scale_is_known_and_changed_test_rows_cannot_alter_it():
    from experiments.size_search_report import fit_implied_scale

    class SentinelLabels(dict):
        def get(self, key, default=None):
            if key == "test":
                raise AssertionError("test label accessed")
            return super().get(key, default)

    features = [
        {"event_id": "train-a", "horizon_sessions": 1, "implied_move": 1.0},
        {"event_id": "train-a", "horizon_sessions": 3, "implied_move": 3.0},
        {"event_id": "train-b", "horizon_sessions": 5, "implied_move": 2.0},
        {"event_id": "test", "horizon_sessions": 1, "implied_move": 100.0},
    ]
    labels = SentinelLabels({"train-a": 2.0, "train-b": 8.0, "test": 1000.0})
    calibration = fit_implied_scale(["train-a", "train-b"], labels, features, (1, 3, 5))
    assert calibration["fit_n"] == 3
    assert calibration["scale"] == pytest.approx(12 / 7)

    features[-1]["implied_move"] = 999.0
    labels["test"] = 0.0
    assert fit_implied_scale(["train-a", "train-b"], labels, features, (1, 3, 5)) == calibration
    assert fit_implied_scale(["train-a"], {"train-a": 2.0},
                             [{"event_id": "train-a", "horizon_sessions": 1, "implied_move": 0.0}],
                             (1,)) == {"scale": None, "fit_n": 1}


def test_implied_scale_never_accesses_test_row_feature_or_label_values():
    from experiments.size_search_report import fit_implied_scale

    class GuardedLabels(dict):
        def get(self, key, default=None):
            if key == "test":
                raise AssertionError("test label accessed")
            return super().get(key, default)

    class GuardedRow(dict):
        def __getitem__(self, key):
            if key == "implied_move" and self["event_id"] == "test":
                raise AssertionError("test feature accessed")
            return super().__getitem__(key)

    features = [GuardedRow(event_id="train", horizon_sessions=1, implied_move=2.0),
                GuardedRow(event_id="test", horizon_sessions=1, implied_move=100.0)]
    calibration = fit_implied_scale(["train"], GuardedLabels(train=4.0, test=1000.0),
                                    features, (1,))
    assert calibration == {"scale": 2.0, "fit_n": 1}


def test_scaled_implied_reports_coverage_metrics_per_year_and_subset():
    events, challenger, incumbent, targets = fixture()
    implied = {("e0", h): 2.5 for h in (1, 3, 5)}
    implied.update({("e1", h): 5.5 for h in (1, 3, 5)})
    implied.update({("e3", h): 4.0 for h in (1, 3, 5)})
    scales = {"2023": {"scale": 2.0, "fit_n": 9}, "2024": {"scale": None, "fit_n": 0}}
    raw = paired_report(events, challenger, incumbent, targets, implied_moves=implied)
    report = paired_report(events, challenger, incumbent, targets,
                           implied_moves=implied, implied_scales=scales)
    rows = {(row["event_id"], row["horizon_sessions"]): row for row in report["paired_rows"]}

    assert report["implied_move_scaled_calibration"] == scales
    assert report["implied_move_scaled_coverage"] == {"available_rows": 6, "total_paired_rows": 12}
    assert rows["e0", 1]["implied_move_scaled"] == pytest.approx(5.0)
    assert rows["e1", 3]["implied_move_scaled"] == pytest.approx(11.0)
    assert rows["e3", 1]["implied_move"] == 4.0
    assert rows["e3", 1]["implied_move_scaled"] is None

    scaled = report["metrics"]["implied_move_scaled"]
    assert scaled["count"] == 6
    assert scaled["rmse"] == pytest.approx(math.sqrt(17.0))
    assert scaled["mae"] == pytest.approx(4.0)
    assert scaled["bias"] == pytest.approx(4.0)

    like = report["like_for_like_implied_move_scaled"]
    assert set(like["metrics"]) == {
        "challenger", "incumbent", "history_mean", "implied_move", "implied_move_scaled"}
    assert like["metrics"]["implied_move_scaled"] == scaled
    assert like["metrics"]["implied_move"]["count"] == 6
    assert like["metrics"]["implied_move"]["rmse"] == pytest.approx(0.5)
    assert like["available_rows"] == 6
    assert like["total_paired_rows"] == 12

    bootstrap = report["quarter_bootstrap"]["implied_move_scaled"]
    assert bootstrap == like["quarter_bootstrap"]
    assert bootstrap["blocks"] == 1 and bootstrap["events"] == 2
    assert set(bootstrap["percentile_95"]) == set(like["metrics"])
    assert bootstrap["percentile_95"]["implied_move_scaled"]["rmse"] == pytest.approx(
        [math.sqrt(17.0)] * 2)
    assert bootstrap["difference_percentile_95"]["implied_move_scaled_minus_implied_move"] == {
        "rmse": pytest.approx([math.sqrt(17.0) - 0.5] * 2),
        "mae": pytest.approx([4.0 - 0.5] * 2)}

    assert report["per_year"]["2023"]["arms"]["implied_move_scaled"]["count"] == 6
    assert report["per_year"]["2023"]["implied_move_scaled_coverage"] == {
        "available_rows": 6, "total_paired_rows": 9}
    assert report["per_year"]["2024"]["arms"]["implied_move_scaled"] is None
    assert report["per_year"]["2024"]["implied_move_scaled_coverage"] == {
        "available_rows": 0, "total_paired_rows": 3}

    assert report["implied_move_coverage"] == raw["implied_move_coverage"]
    assert report["metrics"]["implied_move"] == raw["metrics"]["implied_move"]
    assert report["like_for_like"] == raw["like_for_like"]
    assert report["quarter_bootstrap"]["implied_move"] == raw["quarter_bootstrap"]["implied_move"]
    for arm in ("challenger", "incumbent", "history_mean"):
        assert report["metrics"][arm] == raw["metrics"][arm]


def test_scaled_implied_without_calibration_is_explicit_zero_coverage():
    events, challenger, incumbent, targets = fixture()
    implied = {("e0", h): 2.5 for h in (1, 3, 5)}
    report = paired_report(events, challenger, incumbent, targets, implied_moves=implied)

    assert report["implied_move_coverage"] == {"available_rows": 3, "total_paired_rows": 12}
    assert report["implied_move_scaled_calibration"] == {}
    assert report["implied_move_scaled_coverage"] == {"available_rows": 0, "total_paired_rows": 12}
    assert all(row["implied_move_scaled"] is None for row in report["paired_rows"])
    assert report["metrics"]["implied_move_scaled"] is None
    assert report["like_for_like_implied_move_scaled"]["status"] == "unavailable_no_scaled_implied_move_rows"
    assert report["like_for_like_implied_move_scaled"]["metrics"] is None
    assert report["quarter_bootstrap"]["implied_move_scaled"] is None


@pytest.mark.parametrize("record", [
    {"scale": True, "fit_n": 1}, {"scale": -1.0, "fit_n": 1},
    {"scale": float("nan"), "fit_n": 1}, {"scale": 1.0, "fit_n": True}, {"scale": 1.0},
])
def test_implied_scale_calibration_rejects_malformed_records(record):
    events, challenger, incumbent, targets = fixture()
    with pytest.raises(ValueError, match="implied scale"):
        paired_report(events, challenger, incumbent, targets,
                      implied_moves={}, implied_scales={"2023": record})
