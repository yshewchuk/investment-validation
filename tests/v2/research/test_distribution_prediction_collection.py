"""Real CAS replay plus collection roster, compactness and arithmetic oracles."""
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.errors import OpsError
from experiments import distribution_prediction_collection as collection
from experiments import distribution_prediction_evaluation as evaluation
from tests.v2.research.test_distribution_prediction_diagnostic_report import diagnostics
from tests.v2.research.test_distribution_prediction_evaluation import evaluated as evaluated
from tests.v2.research.test_distribution_prediction_plan import case as case
from tests.v2.research.test_distribution_prediction_plan import request
from tests.v2.research.test_distribution_prediction_report import ARMS, events, scores


@pytest.fixture
def replay(evaluated, monkeypatch):
    evaluated.run.trials.source.document["size_panel"] = {"context": {"target_events": [
        dict(event_id="Z", ticker="AAA", event_date="2024-01-01"),
        dict(event_id="A", ticker="BBB", event_date="2024-03-01"),
        dict(event_id="training", ticker="CCC", event_date="2023-01-01")]}}
    evaluated.saved = evaluation.evaluate_outer(evaluated.run, evaluated.fits, "2024")
    monkeypatch.setattr(evaluation, "evaluation_targets", lambda *args: pytest.fail("new source outcome read"))
    evaluated.publish = evaluated.run.publish
    monkeypatch.setattr(evaluated.run, "publish", lambda *args, **kwargs: pytest.fail("collection publication"))
    return evaluated


def test_real_cas_replay_compact_summaries_and_independent_numeric_oracles(replay):
    before = replay.case.conn.total_changes
    with no_fit_guard():
        groups, summaries = collection.report_evidence(replay.run, replay.fits)
    assert replay.case.conn.total_changes == before
    assert groups == [{"outer_year": "2024", "evaluation_id": content_hash(replay.saved)}]
    forecast, diagnostic = summaries["forecast"], summaries["diagnostics"]
    assert forecast["schema_version"] == "distribution_prediction_score_summary.v1"
    assert diagnostic["schema_version"] == "distribution_prediction_diagnostic_summary.v1"
    assert forecast["population"] == diagnostic["population"] == 2
    assert forecast["primary"] == "h1_same_mean_minus_histogram_crps"
    assert diagnostic["pit_policy"] == "brackets_only_unrandomized"
    for horizon in ("1", "3", "5"):
        for arm in ARMS:
            expected = dict(available=1, unavailable=0, missing_target=1, missing_row=0)
            assert forecast["horizons"][horizon]["arms"][arm]["counts"] == expected
            assert diagnostic["horizons"][horizon]["arms"][arm]["counts"] == expected
        histogram = forecast["horizons"][horizon]["arms"]["histogram"]
        assert histogram["scores"]["crps"] == dict(mean=pytest.approx(1 / 48), count=1)
        baseline = forecast["horizons"][horizon]["contrasts"]["training_empirical"]
        assert baseline["scoreable_mean_crps_improvement"] == pytest.approx(11 / 48)
        assert baseline["full_population_mean_crps_improvement"] is None
        assert baseline["excluded_events"] == 1 and baseline["scoreable_events"] == 1
        assert baseline["conditional_intervals"]["ticker"]["scoreable_clusters"] == 1
        assert histogram["tails"][0]["count"] == 1 and histogram["tails"][0]["mean_calibration_gap"] == 0
        intervals = diagnostic["horizons"][horizon]["arms"]["histogram"]["intervals"]
        assert intervals[0] == dict(level=.5, count=1, covered_count=1, observed_coverage=1, mean_width=.125)
    def compact(value):
        if isinstance(value, dict):
            assert not {"rows", "events", "paired_deltas", "excluded_event_ids"}.intersection(value)
            for item in value.values():
                compact(item)
        elif isinstance(value, list):
            for item in value:
                compact(item)
    compact(summaries)
    assert canonical_json(summaries) == canonical_json(collection.report_evidence(replay.run, replay.fits)[1])
    assert replay.case.conn.total_changes == before


def test_unresolved_tail_statuses_survive_collection(evaluated):
    evaluated.run.trials.source.document["size_panel"] = {"context": {"target_events": [
        dict(event_id=event, ticker=event, event_date="2024-01-01") for event in ("Z", "A")]}}
    # Use the exact admitted unresolved-tail representation, not a report stub.
    from tests.v2.research.test_distribution_scores import distribution
    for value in evaluated.values.values():
        value["result"]["tail"] = distribution(tail=None)["tail"]
    evaluation.evaluate_outer(evaluated.run, evaluated.fits, "2024")
    _, summaries = collection.report_evidence(evaluated.run, evaluated.fits)
    histogram = summaries["forecast"]["horizons"]["1"]["arms"]["histogram"]
    assert histogram["unresolved_crps"] == 1 and histogram["scores"]["crps"]["mean"] is None
    diagnostic = summaries["diagnostics"]["horizons"]["1"]["arms"]["histogram"]
    assert diagnostic["counts"] == dict(available=0, unavailable=1, missing_target=1, missing_row=0)
    assert diagnostic["intervals"][0]["mean_width"] is None
    assert summaries["forecast"]["horizons"]["1"]["arms"]["same_mean"]["counts"]["unavailable"] == 1


@pytest.mark.parametrize("section", ["scores", "diagnostics", "targets"])
def test_rehashed_tampering_refuses_in_real_cas_replay(replay, monkeypatch, section):
    changed = deepcopy(replay.saved)
    descriptor = changed if section == "targets" else changed["horizons"][0]
    rows = list(replay.run.read_section(descriptor[section]))
    rows[0][{"scores": "crps", "diagnostics": "pit_lower", "targets": "absolute_move_pct"}[section]] = 999
    descriptor[section] = replay.publish("distribution_prediction_tampered", rows, count=len(rows))
    original = replay.run.record
    monkeypatch.setattr(replay.run, "record", lambda slot, body=None:
        changed if slot == "evaluated:2024" else original(slot, body))
    with pytest.raises(OpsError):
        collection.report_evidence(replay.run, replay.fits)


def test_later_collection_rechecks_native_admission_without_stale_cache(replay):
    collection.report_evidence(replay.run, replay.fits)
    replay.case.state["closure"] = content_hash("changed installation")
    with pytest.raises(OpsError) as caught:
        collection.report_evidence(replay.run, replay.fits)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_missing_evaluated_completion_cannot_produce_partial_summaries(replay, monkeypatch):
    original = replay.run.record
    monkeypatch.setattr(replay.run, "record", lambda slot, body=None:
        None if slot == "evaluated:2024" else original(slot, body))
    with pytest.raises(OpsError):
        collection.report_evidence(replay.run, replay.fits)


@pytest.fixture
def two_years(monkeypatch):
    calls, metadata = [], events()
    native = {"folds": [{"outer": {"year": "2025", "evaluation_event_ids": ["0"]}},
                        {"outer": {"year": "2026", "evaluation_event_ids": ["1", "2", "3"]}}]}
    sealed = dict(admission_plan_id="admitted", run_plan_id="sealed", event_count=4,
                  evaluation_event_ids=["0", "1", "2", "3"], request=request())
    metadata[0]["event_date"] = "2025-01-01"
    values = {}
    for year, ids in (("2025", ["0"]), ("2026", ["1", "2", "3"])):
        values[year] = (dict(run_plan_id="sealed", outer_year=year),
            [dict(row, outer_year=year) for row in scores() if row["event_id"] in ids],
            [dict(row, outer_year=year) for row in diagnostics() if row["event_id"] in ids])
    def admit():
        calls.append("admit")
        return deepcopy(sealed)
    def plans():
        calls.append("plans")
        return {"plan_id": "admitted"}, deepcopy(native)
    def read(run, fits, year):
        calls.append(year)
        return deepcopy(values[year])
    run = SimpleNamespace(admit=admit, plans=plans,
        trials=SimpleNamespace(source=SimpleNamespace(document={"size_panel": {"context": {"target_events": metadata}}})))
    monkeypatch.setattr(collection, "read_evaluated_outer", read)
    return SimpleNamespace(run=run, fits=object(), calls=calls, metadata=metadata, native=native, sealed=sealed, values=values)


def test_all_years_event_weighted_not_fold_weighted_and_sealed_policy_forwarded(two_years):
    groups, summaries = collection.report_evidence(two_years.run, two_years.fits)
    assert two_years.calls == ["admit", "plans", "2025", "2026"]
    assert groups == [dict(outer_year=year, evaluation_id=content_hash(two_years.values[year][0])) for year in ("2025", "2026")]
    assert summaries["forecast"]["horizons"]["1"]["arms"]["histogram"]["scores"]["crps"] == dict(mean=2.5, count=4)
    assert {key: summaries["forecast"]["uncertainty"][key] for key in request()["uncertainty"]} == request()["uncertainty"]
    assert summaries["forecast"]["horizons"]["1"]["contrasts"]["same_mean"]["excluded_events"] == 0


@pytest.mark.parametrize("change", ["missing", "duplicate", "unselected_duplicate", "date", "ticker"])
def test_frozen_metadata_missing_duplicate_or_malformed_refuses(two_years, change):
    if change == "missing":
        two_years.metadata.pop()
    if change == "duplicate":
        two_years.metadata.append(deepcopy(two_years.metadata[0]))
    if change == "unselected_duplicate":
        two_years.metadata.extend([dict(event_id="training"), dict(event_id="training")])
    if change == "date":
        two_years.metadata[0]["event_date"] = "20250101"
    if change == "ticker":
        two_years.metadata[0]["ticker"] = ""
    with pytest.raises(OpsError):
        collection.report_evidence(two_years.run, two_years.fits)


@pytest.mark.parametrize("change", ["missing_year", "duplicate_year", "duplicate_event", "count", "plan"])
def test_changed_whole_sealed_population_refuses_before_evaluated_reads(two_years, change):
    if change == "missing_year":
        two_years.native["folds"].pop()
    if change == "duplicate_year":
        two_years.native["folds"][1]["outer"]["year"] = "2025"
    if change == "duplicate_event":
        two_years.native["folds"][1]["outer"]["evaluation_event_ids"][0] = "0"
    if change == "count":
        two_years.sealed["event_count"] = 3
    if change == "plan":
        two_years.sealed["admission_plan_id"] = "foreign"
    with pytest.raises(OpsError) as caught:
        collection.report_evidence(two_years.run, two_years.fits)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert two_years.calls == ["admit", "plans"]


@pytest.mark.parametrize("section", [1, 2])
@pytest.mark.parametrize("change", ["missing", "duplicate", "foreign", "horizon", "year", "observation"])
def test_complete_all_arm_horizon_rows_and_cross_report_observations_required(two_years, section, change):
    rows = two_years.values["2025"][section]
    if change == "missing":
        rows.pop()
    if change == "duplicate":
        rows[-1] = deepcopy(rows[0])
    if change == "foreign":
        rows[0]["event_id"] = "1"
    if change == "horizon":
        rows[0]["horizon"] = True
    if change == "year":
        rows[0]["outer_year"] = "2026"
    if change == "observation":
        for row in rows:
            row["observation"] = 1
    with pytest.raises(OpsError) as caught:
        collection.report_evidence(two_years.run, two_years.fits)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize("field,value", [("outer_year", "2027"), ("run_plan_id", "foreign")])
def test_evaluation_record_binds_correct_year_and_run(two_years, field, value):
    two_years.values["2025"][0][field] = value
    with pytest.raises(OpsError):
        collection.report_evidence(two_years.run, two_years.fits)


def test_clean_import_excludes_strategy_and_future_runner():
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname in ('experiments.distribution_report', 'experiments.distribution_pipeline',
                        'experiments.distribution_book', 'experiments.distribution_market',
                        'experiments.distribution_run', 'experiments.distribution_prediction_run',
                        'engine.v2.research.magnitude_distribution_payoff'):
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
from experiments.distribution_prediction_collection import report_evidence
assert callable(report_evidence)
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, timeout=15)
