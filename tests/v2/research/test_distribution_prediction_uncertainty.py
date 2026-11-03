"""Full-roster resampling oracles; every draw keeps arms/horizons together."""
import json
import subprocess
import sys
from copy import deepcopy
from datetime import date

import numpy as np
import pytest

from engine.v2.ops.errors import OpsError
from experiments.distribution_prediction_uncertainty import prediction_uncertainty
from tests.v2.research.test_distribution_prediction_report import ARMS, events, scores

POLICY = dict(seed=17, resamples=128, block_days=7, block_origin="2026-01-01", confidence=.95)


def report(rows=None, metadata=None, **changes):
    return prediction_uncertainty(scores() if rows is None else rows, events() if metadata is None else metadata,
                                  **(POLICY | changes))


def oracle(values, field, metadata=None, **changes):
    policy = POLICY | changes
    metadata = events() if metadata is None else metadata
    clusters = {row["event_id"]: row["ticker"] if field == "ticker" else
                (date.fromisoformat(row["event_date"]) - date.fromisoformat(policy["block_origin"])).days // policy["block_days"]
                for row in metadata}
    labels = sorted(set(clusters.values()))
    rng, samples = np.random.default_rng(policy["seed"]), []
    for _ in range(policy["resamples"]):
        draw = rng.integers(0, len(labels), size=len(labels))
        # Expand whole cluster event lists rather than using the implementation's matrix reducer.
        sample = [values[event] for position in draw for event, label in clusters.items()
                  if label == labels[position] and event in values]
        if sample:
            samples.append(sum(sample) / len(sample))
    quantiles = [(1 - policy["confidence"]) / 2, (1 + policy["confidence"]) / 2]
    return np.quantile(samples, quantiles).tolist(), len(samples)


def test_all_contrasts_and_tail_gaps_match_independent_full_cluster_oracles():
    result = report()
    policy = result["uncertainty"]
    assert {key: policy[key] for key in POLICY} == POLICY
    assert policy["resampling"] == "joint_full_population_clusters" and policy["calendar_clock"] == "event_date"
    assert policy["interpretation"] == "conditional_on_frozen_fits_and_nonempty_scoreable_resamples"
    assert result["primary"] == "h1_same_mean_minus_histogram_crps"
    for horizon in ("1", "3", "5"):
        for arm, offset in zip(ARMS[1:], (0, 1, 2)):
            contrast = result["horizons"][horizon]["contrasts"][arm]
            values = {str(i): delta + offset for i, delta in enumerate((1, -1, 2, 4))}
            for field, clusters in (("ticker", 2), ("calendar", 3)):
                interval = contrast["conditional_intervals"][field]
                expected, used = oracle(values, field)
                assert interval == dict(status="available", clusters=clusters, scoreable_clusters=clusters,
                    scoreable_events=4, used_resamples=used, empty_resamples=0, interval=pytest.approx(expected))
        for arm in ARMS:
            for threshold, tail in zip((5, 10, 20), result["horizons"][horizon]["arms"][arm]["tails"]):
                probability = {5: .75, 10: .5, 20: .25}[threshold]
                values = {str(i): probability - int(observation >= threshold) for i, observation in enumerate((0, 5, 10, 20))}
                assert tail["mean_calibration_gap"] == 0 and tail["gap_direction"] == "probability_minus_observed"
                for field in ("ticker", "calendar"):
                    expected, _ = oracle(values, field)
                    assert tail["conditional_intervals"][field]["interval"] == pytest.approx(expected)


def test_joint_draws_retain_full_cluster_roster_with_different_horizon_masks():
    rows, metadata = scores(), events()
    for index, event in enumerate(metadata):
        event["ticker"] = str(index)  # Four clusters, including entirely unscoreable clusters.
    for row in rows:
        if row["horizon"] == 3 and row["event_id"] in ("0", "1") and row["arm"] == "same_mean":
            row["status"] = "unavailable"
    result = report(rows, metadata)
    primary = result["horizons"]["1"]["contrasts"]["same_mean"]
    sparse = result["horizons"]["3"]["contrasts"]["same_mean"]
    assert primary["scoreable_events"] == 4 and sparse["scoreable_events"] == 2
    assert sparse["full_population_mean_crps_improvement"] is None
    for field in ("ticker", "calendar"):
        expected, used = oracle({"2": 2, "3": 4}, field, metadata)
        interval = sparse["conditional_intervals"][field]
        assert interval["interval"] == pytest.approx(expected)
        assert interval["used_resamples"] == used and interval["empty_resamples"] == POLICY["resamples"] - used
        assert interval["scoreable_clusters"] == 2 and interval["clusters"] > 2
        assert interval["empty_resamples"] > 0


def test_missing_targets_unresolved_crps_and_missing_rows_do_not_become_zero_deltas():
    rows = scores((1,))
    for row in rows:
        if row["event_id"] == "0":
            row.update(status="missing_target", observation=None)
        if row["event_id"] == "1" and row["arm"] == "histogram":
            row["crps"] = None
    result = report(rows)
    primary = result["horizons"]["1"]["contrasts"]["same_mean"]
    assert primary["scoreable_mean_crps_improvement"] == 3
    assert primary["conditional_intervals"]["ticker"]["status"] == "unavailable_fewer_than_two_scoreable_clusters"
    expected, used = oracle({"2": 2, "3": 4}, "calendar")
    assert primary["conditional_intervals"]["calendar"]["interval"] == pytest.approx(expected)
    assert primary["conditional_intervals"]["calendar"]["used_resamples"] == used
    tail = result["horizons"]["1"]["arms"]["histogram"]["tails"][0]
    assert tail["count"] == 3 and tail["conditional_intervals"]["ticker"]["scoreable_events"] == 3
    assert tail["mean_calibration_gap"] == -.25
    absent = result["horizons"]["5"]["contrasts"]["same_mean"]["conditional_intervals"]["calendar"]
    assert absent["scoreable_events"] == 0 and absent["empty_resamples"] == POLICY["resamples"]
    assert "interval" not in absent


def test_empty_population_and_single_clusters_remain_unavailable():
    result = report([], [])
    interval = result["horizons"]["1"]["contrasts"]["same_mean"]["conditional_intervals"]["ticker"]
    assert interval["clusters"] == interval["scoreable_clusters"] == interval["used_resamples"] == 0
    assert interval["empty_resamples"] == POLICY["resamples"] and "interval" not in interval
    metadata = events()
    for row in metadata:
        row.update(ticker="AAA", event_date="2026-01-01")
    result = report(metadata=metadata)
    for interval in result["horizons"]["1"]["contrasts"]["same_mean"]["conditional_intervals"].values():
        assert interval["clusters"] == interval["scoreable_clusters"] == 1 and "interval" not in interval


def test_before_origin_fixed_calendar_blocks_use_floor_not_truncation():
    result = report(block_origin="2026-01-05")
    expected, _ = oracle({str(i): delta for i, delta in enumerate((1, -1, 2, 4))}, "calendar", block_origin="2026-01-05")
    assert result["horizons"]["1"]["contrasts"]["same_mean"]["conditional_intervals"]["calendar"]["interval"] == pytest.approx(expected)


def test_repeat_order_invariance_no_input_mutation_and_json_safe_evidence():
    rows, metadata = scores(), events()
    before = deepcopy((rows, metadata))
    result = report(iter(rows), metadata)
    assert result == report(list(reversed(rows)), list(reversed(metadata)))
    assert (rows, metadata) == before
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("change", [dict(seed=True), dict(seed=-1), dict(seed="1"), dict(resamples=True),
    dict(resamples=0), dict(resamples=1.5), dict(block_days=0), dict(block_days=True), dict(block_days=1.5),
    dict(block_origin="20260101"), dict(block_origin="2026-02-30"), dict(block_origin=None),
    dict(confidence=True), dict(confidence=0), dict(confidence=1), dict(confidence=float("nan"))])
def test_invalid_policy_refuses_before_score_consumption(change):
    def forbidden():
        raise AssertionError("scores must not be consumed")
        yield
    with pytest.raises(OpsError) as caught:
        report(forbidden(), **change)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable


def test_arithmetic_overflow_refuses_without_partial_return_or_mutation():
    rows = scores((1,))
    for row in rows:
        row["crps"] = 1e308 if row["arm"] == "same_mean" else 0
    before = deepcopy(rows)
    with pytest.raises(OpsError) as caught:
        report(rows)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and rows == before


def test_clean_import_does_not_reach_strategy_book_market_or_payoff():
    code = """
import sys
from experiments.distribution_prediction_uncertainty import prediction_uncertainty
assert not any(name in sys.modules for name in (
    'experiments.distribution_report', 'experiments.distribution_book',
    'experiments.distribution_market', 'experiments.distribution_run',
    'engine.v2.research.magnitude_distribution_payoff'))
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
