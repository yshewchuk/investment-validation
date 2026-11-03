"""Independent empirical-law interval/PIT report oracles and refusal checks."""
import json
import subprocess
import sys
from copy import deepcopy

import pytest

from engine.v2.ops.errors import OpsError
from experiments.distribution_prediction_diagnostic_report import (
    LEVELS,
    prediction_diagnostic_report,
)
from experiments.distribution_prediction_report import ARMS
from tests.v2.research.test_distribution_prediction_report import events


def diagnostics(horizons=(1, 3, 5)):
    # Independent exact law: five equally weighted draws [0, 0, 5, 10, 20].
    rows = []
    for horizon in horizons:
        for i, (observation, lower, upper) in enumerate(((0, 0, .4), (5, .4, .6), (10, .6, .8), (20, .8, 1))):
            for arm in ARMS:
                intervals = [dict(level=level, lower=0, upper=end, width=end, covered=0 <= observation <= end)
                             for level, end in zip(LEVELS, (10, 20, 20, 20))]
                rows.append(dict(event_id=str(i), outer_year="2026", horizon=horizon, arm=arm,
                    observation=observation, status="available", intervals=intervals, pit_lower=lower, pit_upper=upper))
    return rows


def test_all_arms_horizons_report_coverage_and_width_together_with_count():
    result = prediction_diagnostic_report(diagnostics(), events())
    assert result["schema_version"] == "distribution_prediction_diagnostic_report.v1"
    assert result["population"] == 4 and set(result["horizons"]) == {"1", "3", "5"}
    assert result["pit_policy"] == "brackets_only_unrandomized"
    assert result["calibration_identity"] == "uncalibrated" and result["final_read_authority"] == "none"
    for summary in result["horizons"].values():
        assert set(summary["arms"]) == set(ARMS)
        for report in summary["arms"].values():
            assert report["population"] == 4 and report["diagnostic_population"] == "available_only"
            assert report["counts"] == dict(available=4, unavailable=0, missing_target=0, missing_row=0)
            assert report["pit_count"] == 4
            assert report["intervals"] == [dict(level=level, count=4, covered_count=covered,
                observed_coverage=covered / 4, mean_width=width)
                for level, covered, width in zip(LEVELS, (3, 4, 4, 4), (10, 20, 20, 20))]
    histogram = [row for row in result["rows"] if row["arm"] == "histogram" and row["horizon"] == 1]
    assert [(row["pit_lower"], row["pit_upper"]) for row in histogram] == [(0, .4), (.4, .6), (.6, .8), (.8, 1)]


def test_missing_unavailable_rows_never_turn_into_failed_or_zero_width_intervals():
    rows = diagnostics((1,))
    for row in rows:
        if row["event_id"] == "0":
            row.update(status="missing_target", observation=None, reason="not_available")
            del row["intervals"]
        if row["event_id"] == "1" and row["arm"] == "histogram":
            row.update(status="unavailable", reason="unresolved_tail")
            del row["pit_lower"]
    rows = [row for row in rows if not (row["event_id"] == "2" and row["arm"] == "histogram")]
    result = prediction_diagnostic_report(rows, events())
    report = result["horizons"]["1"]["arms"]["histogram"]
    assert report["population"] == 4 and report["pit_count"] == 1
    assert report["counts"] == dict(available=1, unavailable=1, missing_target=1, missing_row=1)
    assert report["intervals"][0] == dict(level=.5, count=1, covered_count=0, observed_coverage=0, mean_width=10)
    assert report["intervals"][1] == dict(level=.8, count=1, covered_count=1, observed_coverage=1, mean_width=20)
    assert any(row.get("reason") == "unresolved_tail" for row in result["rows"])
    for horizon in ("3", "5"):
        for report in result["horizons"][horizon]["arms"].values():
            assert report["counts"]["missing_row"] == 4
            assert report["intervals"][0] == dict(level=.5, count=0, covered_count=0, observed_coverage=None, mean_width=None)


def test_zero_width_point_mass_and_inclusive_boundaries_remain_valid():
    row = diagnostics((1,))[0]
    row.update(observation=0, pit_lower=0, pit_upper=1,
               intervals=[dict(level=level, lower=0, upper=0, width=0, covered=True) for level in LEVELS])
    report = prediction_diagnostic_report([row], events()[:1])["horizons"]["1"]["arms"]["histogram"]
    assert all(interval["observed_coverage"] == 1 and interval["mean_width"] == 0 for interval in report["intervals"])


def test_single_pass_deterministic_detached_evidence_with_no_midpoint_randomization():
    rows, metadata = diagnostics(), events()
    before = deepcopy((rows, metadata))
    class Once:
        called = False
        def __iter__(self):
            assert not self.called
            self.called = True
            return iter(rows)
    result = prediction_diagnostic_report(Once(), metadata)
    assert result == prediction_diagnostic_report(list(reversed(rows)), list(reversed(metadata)))
    assert (rows, metadata) == before
    json.dumps(result, allow_nan=False)
    rows[0]["intervals"][0]["width"] = 999
    metadata[0]["source"]["snapshot"] = "changed"
    assert result == prediction_diagnostic_report(*before)


@pytest.mark.parametrize("change", [dict(intervals=None), dict(intervals=[]), dict(pit_lower=-.1),
    dict(pit_lower=.5), dict(pit_lower=True), dict(pit_lower=float("nan")), dict(pit_upper=1.1),
    dict(pit_upper="1"), dict(pit_upper=None), dict(pit_upper=float("inf")), dict(observation=None),
    dict(observation=1), dict(horizon=True), dict(outer_year="2025"), dict(arm="incumbent"), dict(status="unknown")])
def test_bad_row_identity_pit_and_available_contracts_refuse(change):
    rows = diagnostics((1,))
    rows[0].update(change)
    with pytest.raises(OpsError) as caught:
        prediction_diagnostic_report(rows, events())
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable


@pytest.mark.parametrize("change", [dict(level=.6), dict(lower=-1), dict(lower=True), dict(lower=float("nan")),
    dict(upper=9), dict(upper=float("inf")), dict(width=11), dict(width=True), dict(width=None),
    dict(covered=False), dict(covered=1), dict(lower=20, upper=10, width=0)])
def test_invalid_interval_arithmetic_or_coverage_refuses(change):
    rows = diagnostics((1,))
    rows[0]["intervals"][0].update(change)
    with pytest.raises(OpsError):
        prediction_diagnostic_report(rows, events())


def test_reversed_intervals_or_non_nested_equal_tail_levels_refuse():
    rows = diagnostics((1,))
    rows[0]["intervals"].reverse()
    with pytest.raises(OpsError):
        prediction_diagnostic_report(rows, events())
    rows = diagnostics((1,))
    rows[0]["intervals"][1].update(upper=9, width=9)
    with pytest.raises(OpsError):
        prediction_diagnostic_report(rows, events())


@pytest.mark.parametrize("field", ["intervals", "pit_lower", "pit_upper"])
def test_available_rows_require_diagnostics(field):
    rows = diagnostics((1,))
    del rows[0][field]
    with pytest.raises(OpsError):
        prediction_diagnostic_report(rows, events())


@pytest.mark.parametrize("status", ["unavailable", "missing_target"])
@pytest.mark.parametrize("change", [dict(intervals=None), dict(intervals=[]), dict(intervals=[{}] * 4),
    dict(pit_lower=-.1), dict(pit_lower=1.1), dict(pit_lower=True), dict(pit_lower="0"),
    dict(pit_upper=-.1), dict(pit_upper=1.1), dict(pit_upper=False), dict(pit_upper=None),
    dict(pit_lower=.8, pit_upper=.2)])
def test_nonavailable_retained_row_fields_refuse_finite_malformed_values(status, change):
    row, metadata = diagnostics((1,))[0], events()[:1]
    row.update(status=status, observation=None if status == "missing_target" else 0, **change)
    before = deepcopy(([row], metadata))
    with pytest.raises(OpsError) as caught:
        prediction_diagnostic_report([row], metadata)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable
    assert ([row], metadata) == before


@pytest.mark.parametrize("status", ["unavailable", "missing_target"])
@pytest.mark.parametrize("field", ["pit_lower", "pit_upper"])
@pytest.mark.parametrize("value", [-.1, 1.1, "0", True])
def test_nonavailable_single_retained_pit_bound_refuses_invalid_values(status, field, value):
    row, metadata = diagnostics((1,))[0], events()[:1]
    row.update(status=status, observation=None if status == "missing_target" else 0)
    del row["pit_upper" if field == "pit_lower" else "pit_lower"]
    row[field] = value
    before = deepcopy(([row], metadata))
    with pytest.raises(OpsError) as caught:
        prediction_diagnostic_report([row], metadata)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable
    assert ([row], metadata) == before


@pytest.mark.parametrize("status", ["unavailable", "missing_target"])
@pytest.mark.parametrize("change", [dict(level=.6), dict(lower=-1), dict(lower=True), dict(lower="0"),
    dict(upper=9), dict(upper=None), dict(width=11), dict(width=True), dict(width="10"),
    dict(covered=1), dict(covered=None), dict(lower=20, upper=10, width=0)])
def test_nonavailable_retained_interval_fields_refuse(status, change):
    row, metadata = diagnostics((1,))[0], events()[:1]
    row.update(status=status, observation=None if status == "missing_target" else 0)
    row["intervals"][0].update(change)
    before = deepcopy(([row], metadata))
    with pytest.raises(OpsError) as caught:
        prediction_diagnostic_report([row], metadata)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable
    assert ([row], metadata) == before


@pytest.mark.parametrize("status", ["unavailable", "missing_target"])
@pytest.mark.parametrize("change", ["short", "long", "reverse", "lower_nesting", "upper_nesting"])
def test_nonavailable_retained_intervals_require_exact_nested_levels(status, change):
    row, metadata = diagnostics((1,))[0], events()[:1]
    row.update(status=status, observation=None if status == "missing_target" else 0)
    if change == "short":
        row["intervals"].pop()
    elif change == "long":
        row["intervals"].append(deepcopy(row["intervals"][-1]))
    elif change == "reverse":
        row["intervals"].reverse()
    elif change == "lower_nesting":
        row["intervals"][1].update(lower=1, width=19, covered=False)
    else:
        row["intervals"][1].update(upper=9, width=9)
    before = deepcopy(([row], metadata))
    with pytest.raises(OpsError) as caught:
        prediction_diagnostic_report([row], metadata)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable
    assert ([row], metadata) == before


@pytest.mark.parametrize("status", ["unavailable", "missing_target"])
@pytest.mark.parametrize("mask", range(8))
def test_nonavailable_absent_or_valid_partial_fields_preserve_raw_evidence_and_counts(status, mask):
    row, metadata = diagnostics((1,))[0], events()[:1]
    row.update(status=status, observation=None if status == "missing_target" else 0)
    if status == "missing_target":
        # Both boolean coverage values are legal when no observation exists.
        for index, interval in enumerate(row["intervals"]):
            interval["covered"] = bool(index % 2)
    for index, field in enumerate(("intervals", "pit_lower", "pit_upper")):
        if not mask & (1 << index):
            del row[field]
    before = deepcopy(([row], metadata))
    result = prediction_diagnostic_report([row], metadata)
    assert ([row], metadata) == before and result["rows"] == [row]
    bare = {key: value for key, value in row.items() if key not in ("intervals", "pit_lower", "pit_upper")}
    expected = prediction_diagnostic_report([bare], metadata)
    assert {key: value for key, value in result.items() if key != "rows"} == {
        key: value for key, value in expected.items() if key != "rows"}
    summary = result["horizons"]["1"]["arms"]["histogram"]
    assert summary["counts"][status] == 1 and summary["pit_count"] == 0
    assert all(item["count"] == 0 and item["mean_width"] is None for item in summary["intervals"])
    json.dumps(result, allow_nan=False)
    if "intervals" in row:
        row["intervals"][0]["width"] = 999
        assert result["rows"] == before[0]


def test_unavailable_retained_coverage_still_matches_present_observation():
    row, metadata = diagnostics((1,))[0], events()[:1]
    row["status"] = "unavailable"
    row["intervals"][0]["covered"] = False
    before = deepcopy(([row], metadata))
    with pytest.raises(OpsError) as caught:
        prediction_diagnostic_report([row], metadata)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable
    assert ([row], metadata) == before


def test_duplicates_and_unknown_metadata_refuse_empty_cohort_is_explicit():
    rows = diagnostics((1,))
    for supplied, metadata in ((rows + [rows[0]], events()), (rows, events()[:-1]), ([], events() + [events()[0]])):
        with pytest.raises(OpsError):
            prediction_diagnostic_report(supplied, metadata)
    result = prediction_diagnostic_report([], [])
    assert result["population"] == 0
    assert result["horizons"]["1"]["arms"]["histogram"]["intervals"][0]["mean_width"] is None


def test_clean_import_does_not_reach_strategy_book_market_or_payoff():
    code = """
import sys
from experiments.distribution_prediction_diagnostic_report import prediction_diagnostic_report
assert not any(name in sys.modules for name in (
    'experiments.distribution_report', 'experiments.distribution_book',
    'experiments.distribution_market', 'experiments.distribution_run',
    'engine.v2.research.magnitude_distribution_payoff'))
prediction_diagnostic_report([], [])
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
