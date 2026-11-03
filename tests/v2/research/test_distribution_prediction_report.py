"""Hand-calculated forecast report oracles; no fit, source or strategy inputs."""
import json
import subprocess
import sys
from copy import deepcopy

import pytest

from engine.v2.ops.errors import OpsError
from experiments.distribution_prediction_report import ARMS, prediction_report


def events():
    return [dict(event_id=str(i), ticker="AAA" if i < 2 else "BBB", event_date=day,
                 source={"snapshot": "frozen", "ordinal": i})
            for i, day in enumerate(("2026-01-01", "2026-01-02", "2026-01-09", "2026-01-16"))]


def scores(horizons=(1, 3, 5)):
    rows = []
    for horizon in horizons:
        for i, (histogram, control, observation) in enumerate(zip([1, 2, 3, 4], [2, 1, 5, 8], [0, 5, 10, 20])):
            for arm, crps in zip(ARMS, (histogram, control, control + 1, control + 2)):
                rows.append(dict(event_id=str(i), outer_year="2026", horizon=horizon, arm=arm,
                    observation=observation, status="available", crps=crps, rps=crps * 2,
                    log_loss="infinite" if i == 0 else i / 10,
                    tails=[dict(threshold=threshold, probability_ge=p, observed_ge=observation >= threshold)
                           for threshold, p in ((5, .75), (10, .5), (20, .25))]))
    return rows


def test_all_arms_horizons_and_primary_have_independent_event_weighted_oracles():
    result = prediction_report(scores(), events())
    assert result["primary"] == "h1_same_mean_minus_histogram_crps"
    assert result["calibration_identity"] == "uncalibrated"
    assert result["final_read_authority"] == "none"
    assert result["population"] == 4 and set(result["horizons"]) == {"1", "3", "5"}
    for horizon, summary in result["horizons"].items():
        assert set(summary["arms"]) == set(ARMS)
        for arm, expected in zip(ARMS, (2.5, 4, 5, 6)):
            report = summary["arms"][arm]
            assert report["population"] == 4 and report["summary_population"] == "scoreable_only"
            assert report["counts"] == dict(available=4, unavailable=0, missing_target=0, missing_row=0)
            assert report["scores"]["crps"] == dict(mean=expected, count=4)
            assert report["scores"]["rps"] == dict(mean=2 * expected, count=4)
            assert report["scores"]["log_loss"] == dict(mean="infinite", count=4,
                finite_mean=pytest.approx(.2), finite_count=3, infinite_count=1)
            assert report["tails"] == [dict(threshold=t, count=4, observed_count=n,
                observed_frequency=p, mean_probability_ge=p) for t, n, p in ((5, 3, .75), (10, 2, .5), (20, 1, .25))]
        for arm, expected in zip(ARMS[1:], (1.5, 2.5, 3.5)):
            report = summary["contrasts"][arm]
            assert report["role"] == ("primary" if horizon == "1" and arm == "same_mean" else "descriptive")
            assert report["status"] == "complete" and report["scoreable_events"] == 4
            assert report["scoreable_mean_crps_improvement"] == expected
            assert report["full_population_mean_crps_improvement"] == expected
            assert report["excluded_event_ids"] == []
    assert result["horizons"]["1"]["contrasts"]["same_mean"]["paired_deltas"] == [
        dict(event_id=str(i), delta=value) for i, value in enumerate((1, -1, 2, 4))]


def test_missing_targets_controls_crps_and_rows_keep_exact_denominator_and_provenance():
    rows = scores((1,))
    for row in rows:
        if row["event_id"] == "0":
            row.update(status="missing_target", observation=None, reason="not_available")
        elif row["event_id"] == "1" and row["arm"] == "same_mean":
            row.update(status="unavailable", reason="no_causal_residuals")
        elif row["event_id"] == "2" and row["arm"] == "histogram":
            row["crps"] = None
    rows = [row for row in rows if not (row["event_id"] == "3" and row["arm"] == "training_empirical")]
    result = prediction_report(rows, events())
    summary = result["horizons"]["1"]
    assert summary["arms"]["histogram"]["counts"] == dict(available=3, unavailable=0, missing_target=1, missing_row=0)
    assert summary["arms"]["histogram"]["unresolved_crps"] == 1
    assert summary["arms"]["histogram"]["scores"]["crps"] == dict(mean=3, count=2)
    assert summary["arms"]["histogram"]["scores"]["rps"] == dict(mean=6, count=3)
    assert summary["arms"]["same_mean"]["counts"]["unavailable"] == 1
    assert summary["arms"]["training_empirical"]["counts"]["missing_row"] == 1
    primary = summary["contrasts"]["same_mean"]
    assert primary["population"] == 4 and primary["scoreable_events"] == 1
    assert primary["excluded_event_ids"] == ["0", "1", "2"]
    assert primary["full_population_mean_crps_improvement"] is None
    assert primary["scoreable_mean_crps_improvement"] == 4
    assert any(row.get("reason") == "no_causal_residuals" for row in result["rows"])
    assert result["events"] == events()
    for horizon in ("3", "5"):
        for arm in ARMS:
            assert result["horizons"][horizon]["arms"][arm]["counts"]["missing_row"] == 4
        assert result["horizons"][horizon]["contrasts"]["same_mean"]["scoreable_mean_crps_improvement"] is None


def test_empty_inputs_and_infinite_only_loss_do_not_invent_zero_scores():
    result = prediction_report([], events())
    for summary in result["horizons"].values():
        for report in summary["arms"].values():
            assert report["scores"]["crps"] == dict(mean=None, count=0)
            assert report["tails"][0]["observed_frequency"] is None
        assert summary["contrasts"]["same_mean"]["excluded_event_ids"] == ["0", "1", "2", "3"]
    rows = scores((1,))[:4]
    report = prediction_report(rows, events()[:1])["horizons"]["1"]["arms"]["histogram"]
    assert report["scores"]["log_loss"] == dict(mean="infinite", count=1,
        finite_mean=None, finite_count=0, infinite_count=1)
    assert prediction_report([], [])["population"] == 0


def test_single_pass_order_invariance_detached_inputs_and_json_safety():
    rows, metadata = scores(), events()
    before = deepcopy((rows, metadata))
    class Once:
        called = False
        def __iter__(self):
            assert not self.called
            self.called = True
            return iter(rows)
    result = prediction_report(Once(), metadata)
    assert result == prediction_report(list(reversed(rows)), list(reversed(metadata)))
    assert (rows, metadata) == before
    json.dumps(result, allow_nan=False)
    rows[0]["tails"][0]["probability_ge"] = 0
    metadata[0]["source"]["snapshot"] = "changed"
    assert result == prediction_report(*before)


@pytest.mark.parametrize("change", [dict(event_id="unknown"), dict(event_id=[]), dict(horizon=2), dict(horizon=True),
    dict(arm="incumbent"), dict(arm=[]), dict(status="invented"), dict(outer_year=None), dict(outer_year=""),
    dict(crps=float("nan")), dict(crps=float("inf")), dict(crps=-1), dict(crps=True), dict(crps="missing"),
    dict(rps=float("nan")), dict(rps=-1), dict(rps=True), dict(log_loss="other"), dict(log_loss=None),
    dict(log_loss=float("inf")), dict(log_loss=True), dict(tails=None), dict(tails=[]),
    dict(observation=None), dict(observation=-1), dict(observation=True), dict(observation=float("inf")),
    dict(status="missing_target"), dict(outer_year="2025"), dict(observation=1)])
def test_malformed_rows_and_cross_arm_observation_or_year_mismatches_refuse(change):
    rows = scores((1,))
    rows[0].update(change)
    before = deepcopy(rows)
    with pytest.raises(OpsError) as caught:
        prediction_report(rows, events())
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable
    assert rows == before


@pytest.mark.parametrize("field", ["event_id", "outer_year", "horizon", "arm", "observation", "status", "crps", "rps", "log_loss", "tails"])
def test_required_row_fields_have_typed_refusal(field):
    rows = scores((1,))
    del rows[0][field]
    with pytest.raises(OpsError):
        prediction_report(rows, events())


@pytest.mark.parametrize("field,value", [("threshold", 6), ("observed_ge", 1), ("observed_ge", True),
    ("probability_ge", -1), ("probability_ge", 1.1), ("probability_ge", float("nan")),
    ("probability_ge", True), ("probability_ge", "0.5")])
def test_invalid_tail_contract_refuses(field, value):
    rows = scores((1,))
    rows[0]["tails"][0][field] = value
    with pytest.raises(OpsError):
        prediction_report(rows, events())


@pytest.mark.parametrize("change", [dict(event_id=""), dict(event_id=[]), dict(ticker=""), dict(ticker=None),
    dict(event_date="20260101"), dict(event_date="2026-02-30"), dict(event_date=None)])
def test_bad_frozen_metadata_refuses(change):
    metadata = events()
    metadata[0].update(change)
    with pytest.raises(OpsError):
        prediction_report(scores(), metadata)


def test_duplicate_rows_events_and_cross_horizon_year_conflicts_refuse():
    rows, metadata = scores(), events()
    for candidate, cohort in ((rows + [rows[0]], metadata), (rows, metadata + [metadata[0]])):
        with pytest.raises(OpsError):
            prediction_report(candidate, cohort)
    for row in rows:
        if row["horizon"] == 3:
            row["outer_year"] = "2025"
    with pytest.raises(OpsError):
        prediction_report(rows, metadata)


def test_extreme_finite_scores_use_scaled_mean_without_sum_overflow():
    rows = scores((1,))
    for row in rows:
        row["crps"] = 1e308 if row["arm"] == "histogram" else 0
    summary = prediction_report(rows, events())["horizons"]["1"]
    assert summary["arms"]["histogram"]["scores"]["crps"]["mean"] == 1e308
    assert summary["contrasts"]["same_mean"]["scoreable_mean_crps_improvement"] == -1e308


def test_clean_import_does_not_reach_strategy_book_market_or_payoff():
    code = """
import sys
from experiments.distribution_prediction_report import prediction_report
assert not any(name in sys.modules for name in (
    'experiments.distribution_report', 'experiments.distribution_book',
    'experiments.distribution_market', 'experiments.distribution_run',
    'engine.v2.research.magnitude_distribution_payoff'))
prediction_report([], [])
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def retained_rows(status):
    rows = scores((1,))
    for row in rows:
        row["status"] = status
        if status == "missing_target":
            row["observation"] = None
    return rows


@pytest.mark.parametrize("status", ["available", "unavailable", "missing_target"])
@pytest.mark.parametrize("location", ["row_value", "row_key", "event_value", "event_key"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_retained_payload_requires_finite_json_for_all_statuses(status, location, invalid):
    rows, metadata = retained_rows(status), events()
    target = rows[0] if location.startswith("row") else metadata[0]
    target["provenance"] = {invalid: "value"} if location.endswith("key") else {"nested": [{"value": invalid}]}
    with pytest.raises(OpsError) as caught:
        prediction_report(rows, metadata)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable


@pytest.mark.parametrize("status", ["available", "unavailable", "missing_target"])
@pytest.mark.parametrize("location", ["row", "event"])
@pytest.mark.parametrize("invalid", [b"bytes", {1, 2}, {(1, 2): "tuple-key"}])
def test_non_json_retained_values_and_keys_have_typed_refusals(status, location, invalid):
    rows, metadata = retained_rows(status), events()
    target = rows[0] if location == "row" else metadata[0]
    target["provenance"] = {"nested": [invalid]}
    with pytest.raises(OpsError) as caught:
        prediction_report(rows, metadata)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable


@pytest.mark.parametrize("status", ["available", "unavailable", "missing_target"])
@pytest.mark.parametrize("change", [dict(crps="bad"), dict(crps=-1), dict(crps=True),
    dict(rps="bad"), dict(rps=None), dict(rps=-1), dict(log_loss="bad"), dict(log_loss=None),
    dict(log_loss=True), dict(tails="bad"), dict(tails=[]), dict(crps=float("nan")), dict(rps=float("inf"))])
def test_present_score_fields_are_validated_even_when_not_aggregated(status, change):
    rows = retained_rows(status)
    rows[0].update(change)
    with pytest.raises(OpsError) as caught:
        prediction_report(rows, events())
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable


@pytest.mark.parametrize("status", ["available", "unavailable", "missing_target"])
@pytest.mark.parametrize("field,value", [("threshold", 6), ("observed_ge", 1),
    ("probability_ge", "bad"), ("probability_ge", 2), ("probability_ge", float("nan"))])
def test_present_tails_are_validated_even_without_an_observed_target(status, field, value):
    rows = retained_rows(status)
    rows[0]["tails"][0][field] = value
    with pytest.raises(OpsError) as caught:
        prediction_report(rows, events())
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable


@pytest.mark.parametrize("location", ["row", "event"])
@pytest.mark.parametrize("shape", ["cycle", "too_deep"])
def test_recursive_retained_payload_has_typed_refusal(location, shape):
    rows, metadata = retained_rows("unavailable"), events()
    payload = []
    if shape == "cycle":
        payload.append(payload)
    else:
        for _ in range(sys.getrecursionlimit() + 10):
            payload = [payload]
    (rows[0] if location == "row" else metadata[0])["provenance"] = payload
    with pytest.raises(OpsError) as caught:
        prediction_report(rows, metadata)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC" and not caught.value.problem.retryable


@pytest.mark.parametrize("status", ["available", "unavailable", "missing_target"])
def test_valid_retained_fields_and_unknown_provenance_are_preserved_json_safe(status):
    rows, metadata = retained_rows(status), events()
    rows[0]["provenance"] = {"nested": [{"value": .5, "note": "opaque", "optional": None}]}
    if status == "missing_target":
        rows[0]["tails"][0]["observed_ge"] = True  # No target exists against which to compare it.
    before = deepcopy((rows, metadata))
    result = prediction_report(rows, metadata)
    assert result["rows"] == sorted(rows, key=lambda row: (row["event_id"], row["horizon"], row["arm"]))
    assert result["events"] == metadata and (rows, metadata) == before
    json.dumps(result, allow_nan=False)
    histogram = result["horizons"]["1"]["arms"]["histogram"]
    assert histogram["counts"][status] == 4
    assert histogram["scores"]["crps"] == dict(mean=2.5 if status == "available" else None,
                                               count=4 if status == "available" else 0)


@pytest.mark.parametrize("status", ["unavailable", "missing_target"])
@pytest.mark.parametrize("keep", [(), ("crps",), ("rps",), ("log_loss",), ("tails",)])
def test_nonavailable_rows_may_omit_any_unapplicable_score_fields(status, keep):
    rows = retained_rows(status)
    for row in rows:
        for field in ("crps", "rps", "log_loss", "tails"):
            if field not in keep:
                del row[field]
    result = prediction_report(rows, events())
    assert result["horizons"]["1"]["arms"]["histogram"]["counts"][status] == 4
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("missing_target", [False, True])
def test_real_scorer_omits_unavailable_metrics_and_retains_numerical_status_parity(missing_target):
    from experiments.distribution_prediction_scores import score_forecasts
    from tests.v2.research.test_distribution_scores import forecast, pool

    targets = {"e": {"omitted": "no_price_window"} if missing_target else .125}
    rows = list(score_forecasts(forecast(), pool(status="no_earlier_inner"), targets, outer_year="2026", horizon=1))
    for row in rows:
        if row["status"] != "available":
            assert not {"crps", "rps", "log_loss", "tails"}.intersection(row)
    result = prediction_report(rows, [dict(event_id="e", ticker="SYN", event_date="2026-01-01")])
    histogram = result["horizons"]["1"]["arms"]["histogram"]
    assert histogram["scores"]["crps"] == dict(mean=None if missing_target else pytest.approx(1 / 48),
                                               count=0 if missing_target else 1)
    assert result["horizons"]["1"]["arms"]["same_mean"]["counts"]["missing_target" if missing_target else "unavailable"] == 1
    json.dumps(result, allow_nan=False)
