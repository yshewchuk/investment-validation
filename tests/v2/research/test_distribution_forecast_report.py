"""Pure forecast summaries; independent of the later runner and market adapter."""
from copy import deepcopy

import numpy as np
import pytest

from engine.v2.ops.errors import OpsError
from experiments.distribution_report import forecast_report


def metadata():
    return [dict(event_id=str(i), ticker="AAA" if i < 2 else "BBB", decision_date=day)
        for i, day in enumerate(("2026-01-01", "2026-01-02", "2026-01-09", "2026-01-16"))]


def score_rows(horizons=(1, 3, 5)):
    rows = []
    for horizon in horizons:
        for i, (histogram, control, observation) in enumerate(zip([1, 2, 3, 4], [2, 1, 5, 8], [0, 5, 10, 20])):
            for arm, crps in (("histogram", histogram), ("same_mean", control), ("matched_point", control + 1)):
                rows.append(dict(event_id=str(i), outer_year="2026", horizon=horizon, arm=arm,
                    observation=observation, status="available", crps=crps, rps=crps * 2,
                    log_loss="infinite" if i == 0 else i / 10,
                    tails=[dict(threshold=threshold, probability_ge=p, observed_ge=observation >= threshold)
                        for threshold, p in ((5, .75), (10, .5), (20, .25))]))
    return rows


def forecast_summary(rows, opportunities=None, **changes):
    if opportunities is None:
        opportunities = metadata()
    kwargs = dict(seed=17, resamples=128, block_days=7, block_origin="2026-01-01", confidence=.95)
    return forecast_report(iter(rows), opportunities, **(kwargs | changes))


def test_forecast_report_primary_is_event_weighted_h1_not_best_horizon():
    result = forecast_summary(score_rows())
    assert result["primary"] == "h1_same_mean_minus_histogram_crps"
    assert result["calibration_identity"] == "uncalibrated" and result["final_read_authority"] == "none"
    for horizon, summary in result["horizons"].items():
        assert summary["role"] == ("primary" if horizon == "1" else "descriptive")
        assert summary["mean_crps_improvement"] == 1.5
        assert summary["scoreable_events"] == summary["opportunities"] == 4
        assert summary["excluded_events"] == 0 and summary["status"] == "available"
        assert summary["coverage"] == {arm: {"available": 4} for arm in ("histogram", "same_mean")}
    histogram = result["horizons"]["1"]["histogram"]
    assert histogram["crps_mean"] == 2.5 and histogram["rps_mean"] == 5
    assert histogram["log_loss_mean"] == "infinite" and histogram["infinite_log_loss_count"] == 1
    assert histogram["finite_log_loss_mean"] == pytest.approx(.2)
    assert histogram["tails"] == [dict(threshold=t, count=4, observed_count=n, mean_probability_ge=p)
        for t, n, p in ((5, 3, .75), (10, 2, .5), (20, 1, .25))]


def test_forecast_calendar_and_ticker_intervals_match_paired_cluster_oracle():
    result = forecast_summary(score_rows((1,)))["horizons"]["1"]
    for field, aggregates in (("calendar", [[0, 2], [2, 1], [4, 1]]), ("ticker", [[0, 2], [6, 2]])):
        values, samples, rng = np.array(aggregates), [], np.random.default_rng(17)
        for _ in range(128):
            total = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
            samples.append(total[0] / total[1])
        assert result["conditional_intervals"][field]["interval"] == pytest.approx(np.quantile(samples, [.025, .975]))


def test_forecast_common_usable_mask_preserves_all_exclusion_counts():
    rows = score_rows((1,))
    for row in rows:
        if row["event_id"] == "0":
            row.update(status="missing_target", observation=None, reason="unavailable")
            row.pop("crps")
        if row["event_id"] == "1" and row["arm"] == "same_mean":
            row.update(status="unavailable")
            row.pop("crps")
        if row["event_id"] == "2" and row["arm"] == "histogram":
            row["crps"] = None
    result = forecast_summary(rows)["horizons"]["1"]
    assert result["opportunities"] == 4 and result["scoreable_events"] == 1 and result["excluded_events"] == 3
    assert result["mean_crps_improvement"] == 4
    assert result["coverage"]["histogram"] == dict(missing_target=1, available=2, unresolved_crps=1)
    assert result["coverage"]["same_mean"] == dict(missing_target=1, unavailable=1, available=2)
    assert result["conditional_intervals"]["calendar"]["status"] == "unavailable_fewer_than_two_clusters"
    assert result["histogram"]["crps_mean"] == 3  # Only histogram events1,3; no zeros filled in.


def test_forecast_missing_rows_no_pairs_and_absent_h1_do_not_invent_effects():
    rows = [row for row in score_rows((3,)) if row["arm"] == "histogram"]
    result = forecast_summary(rows)
    assert "1" not in result["horizons"] and result["horizons"]["3"]["role"] == "descriptive"
    summary = result["horizons"]["3"]
    assert summary["mean_crps_improvement"] is None and summary["status"] == "unavailable"
    assert summary["coverage"]["same_mean"] == {"missing_row": 4}
    assert summary["excluded_events"] == 4


@pytest.mark.parametrize("change", [dict(event_id="unknown"), dict(horizon=2), dict(horizon=True),
    dict(crps=float("nan")), dict(crps=float("inf")), dict(crps=-1), dict(crps=True)])
def test_forecast_invalid_cohort_horizon_or_crps_refuses(change):
    rows = score_rows((1,))
    rows[0].update(change)
    with pytest.raises(OpsError):
        forecast_summary(rows)


def test_forecast_duplicate_rows_refuse_even_if_cas_positions_differ():
    rows = score_rows((1,))
    rows.append(rows[0] | {"position": 999})
    with pytest.raises(OpsError):
        forecast_summary(rows)


def test_forecast_report_single_pass_cas_iterator_and_no_mutation():
    rows, opportunities = score_rows((1,)), metadata()
    before = deepcopy((rows, opportunities))
    class Once:
        called = False
        def __iter__(self):
            assert not self.called
            self.called = True
            return iter(rows)
    kwargs = dict(seed=17, resamples=128, block_days=7, block_origin="2026-01-01", confidence=.95)
    result = forecast_report(Once(), opportunities, **kwargs)
    assert result == forecast_summary(list(reversed(rows)), list(reversed(opportunities)))
    assert (rows, opportunities) == before
    assert "event_id" not in str(result)


def test_forecast_report_real_scorer_output_and_missing_target_rows():
    from experiments.distribution_scores import score_forecasts
    from tests.v2.research.test_distribution_scores import forecast, pool as make_pool

    result, pool = forecast(), make_pool()
    meta = [dict(event_id="e", ticker="SYN", decision_date="2026-01-01")]
    rows = list(score_forecasts(result, pool, {"e": .125}, outer_year="2026", horizon=1))
    summary = forecast_summary(rows, meta)["horizons"]["1"]
    expected = next(row["crps"] for row in rows if row["arm"] == "same_mean") - rows[0]["crps"]
    assert summary["mean_crps_improvement"] == expected
    missing = list(score_forecasts(result, pool, {"e": {"omitted": "no_label"}}, outer_year="2026", horizon=1))
    summary = forecast_summary(missing, meta)["horizons"]["1"]
    assert summary["mean_crps_improvement"] is None and summary["histogram"]["crps_mean"] is None


@pytest.mark.parametrize("change", [dict(arm="invented"), dict(status="invented"),
    dict(rps=float("nan")), dict(rps=-1), dict(rps=True), dict(log_loss="other"),
    dict(log_loss=float("inf")), dict(log_loss=None), dict(tails=None), dict(tails=[]),
    dict(observation=None), dict(observation=-1), dict(observation=True), dict(observation=float("inf"))])
def test_forecast_report_refuses_malformed_consumed_score_values(change):
    rows = score_rows((1,))
    rows[0].update(change)
    with pytest.raises(OpsError) as caught:
        forecast_summary(rows)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False


@pytest.mark.parametrize("bad_row", [None, [], {}, {"horizon": 1}, {"horizon": 1, "event_id": []}])
def test_forecast_report_malformed_rows_have_typed_refusals(bad_row):
    with pytest.raises(OpsError) as caught:
        forecast_summary([bad_row])
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("field,value", [("threshold", 6), ("observed_ge", 1),
    ("probability_ge", -1), ("probability_ge", 1.1), ("probability_ge", float("nan")),
    ("probability_ge", True), ("probability_ge", "0.5")])
def test_forecast_tail_report_admits_only_fixed_finite_probabilities_and_boolean_truth(field, value):
    rows = score_rows((1,))
    rows[0]["tails"][0][field] = value
    with pytest.raises(OpsError) as caught:
        forecast_summary(rows)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("field", ["observation", "crps", "rps", "log_loss", "tails", "status"])
def test_forecast_available_rows_require_consumed_contract_fields(field):
    rows = score_rows((1,))
    del rows[0][field]
    with pytest.raises(OpsError) as caught:
        forecast_summary(rows)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_forecast_paired_scores_require_the_same_observed_target():
    rows = score_rows((1,))
    rows[1]["observation"] = 1
    with pytest.raises(OpsError) as caught:
        forecast_summary(rows)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("unresolved_arm", [None, "histogram", "same_mean", "matched_point"])
@pytest.mark.parametrize("changed_arm", ["histogram", "same_mean", "matched_point"])
@pytest.mark.parametrize("reverse", [False, True])
def test_available_target_identity_precedes_crps_mask(unresolved_arm, changed_arm, reverse):
    rows = score_rows((1,))[:3]
    for row in rows:
        if row["arm"] == unresolved_arm:
            row["crps"] = None
        if row["arm"] == changed_arm:
            row["observation"] = 1  # Same tail indicators as zero: isolate target identity.
    before = deepcopy(rows)
    with pytest.raises(OpsError) as caught:
        forecast_summary(rows[::-1] if reverse else rows, metadata()[:1])
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False
    assert caught.value.problem.message == "invalid distribution paired report inputs or arithmetic"
    assert rows == before


@pytest.mark.parametrize("absent_arm", ["histogram", "same_mean", "matched_point"])
@pytest.mark.parametrize("reverse", [False, True])
def test_remaining_available_arms_still_require_one_target(absent_arm, reverse):
    rows = [row for row in score_rows((1,))[:3] if row["arm"] != absent_arm]
    rows[0]["observation"] = 1
    for row in rows:
        row["crps"] = None
    with pytest.raises(OpsError) as caught:
        forecast_summary(rows[::-1] if reverse else rows, metadata()[:1])
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False
    assert caught.value.problem.message == "invalid distribution paired report inputs or arithmetic"


@pytest.mark.parametrize("status", ["unavailable", "missing_target"])
@pytest.mark.parametrize("placeholder", [None, "absent", ["not consumed"], {"omitted": "no_label"}, 1])
def test_unavailable_target_placeholders_remain_coverage_only(status, placeholder):
    rows = score_rows((1,))[:3]
    rows[1] = {key: rows[1][key] for key in ("event_id", "outer_year", "horizon", "arm")}
    rows[1].update(status=status, observation=placeholder)
    if placeholder == "absent":
        rows[1].pop("observation")
    before = deepcopy(rows)
    summary = forecast_summary(rows, metadata()[:1])["horizons"]["1"]
    assert summary["scoreable_events"] == 0 and summary["mean_crps_improvement"] is None
    assert summary["opportunities"] == summary["excluded_events"] == 1
    assert summary["coverage"] == {"histogram": {"available": 1}, "same_mean": {status: 1}}
    assert summary["histogram"]["crps_mean"] == 1
    assert rows == before


@pytest.mark.parametrize("present_arms", [("histogram",), ("same_mean",), ("matched_point",),
    ("histogram", "same_mean"), ("histogram", "matched_point"), ("same_mean", "matched_point")])
def test_matching_available_targets_allow_absent_arms_and_unresolved_crps(present_arms):
    rows = [row for row in score_rows((1,))[:3] if row["arm"] in present_arms]
    for row in rows:
        row["crps"] = None
    result = forecast_summary(rows, metadata()[:1])
    assert result == forecast_summary(rows[::-1], metadata()[:1])
    summary = result["horizons"]["1"]
    assert summary["scoreable_events"] == 0 and summary["mean_crps_improvement"] is None
    assert summary["opportunities"] == summary["excluded_events"] == 1
    assert summary["coverage"] == {arm: {"unresolved_crps" if arm in present_arms else "missing_row": 1}
        for arm in ("histogram", "same_mean")}


@pytest.mark.parametrize("settings", [dict(seed=3, resamples=13, confidence=.8, block_days=8),
    dict(seed=29, resamples=37, confidence=.5, block_days=1)])
def test_nondefault_uncertainty_settings_and_schema_have_independent_oracles(settings):
    result = forecast_summary(score_rows((1,)), **settings)
    assert result["schema_version"] == "distribution_forecast_report.v1"
    assert result["uncertainty"] == settings | dict(block_origin="2026-01-01",
        calendar_blocks="occupied_fixed_nonoverlapping", interpretation="conditional_on_frozen_fits_no_refitting")
    values = np.array([[0, 2], [6, 2]]) if settings["block_days"] == 8 else np.array([[1, 1], [-1, 1], [2, 1], [4, 1]])
    samples, rng = [], np.random.default_rng(settings["seed"])
    for _ in range(settings["resamples"]):
        total = values[rng.integers(0, len(values), size=len(values))].sum(axis=0)
        samples.append(total[0] / total[1])
    q = (1 - settings["confidence"]) / 2
    assert result["horizons"]["1"]["conditional_intervals"]["calendar"]["interval"] == pytest.approx(np.quantile(samples, [q, 1 - q]))


def test_real_scorer_report_runs_without_source_reads_or_fitting(monkeypatch):
    from pathlib import Path

    from engine.models.no_fit import no_fit_guard as legacy_guard
    from engine.v2.models.no_fit import no_fit_guard as native_guard
    from experiments.distribution_scores import score_forecasts
    from tests.v2.research.test_distribution_scores import forecast, pool

    result, control = forecast(event="0"), pool()
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected source read")
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with legacy_guard(), native_guard():
        rows = score_forecasts(result, control, {"0": .125}, outer_year="2026", horizon=1)
        summary = forecast_summary(rows, metadata()[:1])
    assert summary["horizons"]["1"]["histogram"]["crps_mean"] == pytest.approx(1 / 48)
    assert summary["horizons"]["1"]["mean_crps_improvement"] == pytest.approx(1 / 24)


def test_forecast_tail_truth_must_match_the_supplied_observation():
    rows = score_rows((1,))
    rows[0]["tails"][0]["observed_ge"] = True
    with pytest.raises(OpsError) as caught:
        forecast_summary(rows)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
