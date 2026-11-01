"""Stored cohort validation is loaded once and reused by later trials, never reloaded."""
# ruff: noqa: F811
import json

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import content_hash, from_document
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_trials as trials
from tests.v2.research.test_magnitude_search_execution import LIMITS
from tests.v2.research.test_magnitude_search_execution_integrity import native  # noqa: F401
from tests.v2.research.test_magnitude_search_trial_integrity import (
    journal,  # noqa: F401
    native_labels,
    prepare_recording,
    rows,
    validation_evidence,
)

COHORT = content_hash(["plan", "2024", "2023"])


def counting_labels(monkeypatch):
    calls = []
    def counted(*args, **kwargs):
        calls.append(kwargs["labels_before"])
        return native_labels()
    monkeypatch.setattr(trials, "load_magnitude_targets", counted)
    return calls


def forbid_labels(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("stored validation reload opened labels")
    monkeypatch.setattr(trials, "load_magnitude_targets", forbidden)


def corrupt_stored_record(search_journal, slot):
    ref_json = search_journal.conn.execute(
        "SELECT ref_json FROM magnitude_search_records WHERE slot=?", (slot,)).fetchone()[0]
    path = search_journal.store.root / from_document(ArtifactRef, json.loads(ref_json)).storage_key
    path.chmod(0o644)
    path.write_bytes(b"tampered validation bytes")


def validation_rows(search_journal):
    return search_journal.conn.execute(
        "SELECT count(*) FROM magnitude_search_records WHERE slot LIKE 'validation:%'").fetchone()[0]


def add_inner_fold(search_journal):
    search_journal.plan["folds"][0]["inner"].append({"year": "2022", "evaluation_event_ids": ["event"],
                                                       "evaluation_labels_before": "2024-01-01"})


def test_cohort_labels_load_once_across_all_candidate_trials(journal, monkeypatch):
    prepare_recording(journal, monkeypatch)
    calls = counting_labels(monkeypatch)
    append = journal.recorder._append
    validation_appends = []
    def counted_append(slot, body, **kwargs):
        if slot.startswith("validation:"):
            validation_appends.append(body)
        return append(slot, body, **kwargs)
    monkeypatch.setattr(journal.recorder, "_append", counted_append)
    journal.plan["candidates"].append({"candidate_id": "m-mid", "recipe": {}})
    for candidate in ("z-last", "m-mid", "a-first"):
        journal.recorder.start("2024", "2023", candidate)
        assert journal.recorder.record("2024", "2023", candidate, forecasts=rows(4))["status"] == "succeeded"
    assert len(calls) == 1  # three trials of one cohort load labels for only the first
    assert validation_appends == [validation_evidence()] * 3  # each trial still replays the same append
    assert validation_rows(journal) == 1
    assert journal.recorder._read("validation:" + COHORT) == validation_evidence()


def test_reused_terminal_matches_a_labels_reloading_trial(journal, monkeypatch):
    prepare_recording(journal, monkeypatch)
    add_inner_fold(journal)
    journal.recorder.start("2024", "2023", "z-last")
    loaded = journal.recorder.record("2024", "2023", "z-last", forecasts=rows(4))  # first trial loads labels
    journal.recorder.start("2024", "2023", "a-first")
    reused = journal.recorder.record("2024", "2023", "a-first", forecasts=rows(4))  # reuse path
    journal.recorder.start("2024", "2022", "a-first")
    reloaded = journal.recorder.record("2024", "2022", "a-first", forecasts=rows(4))  # fresh cohort reloads
    assert reused["validation_id"] == loaded["validation_id"] == reloaded["validation_id"]
    assert reused["validation_id"] == content_hash(validation_evidence())
    assert reused["predictions"] == loaded["predictions"] == reloaded["predictions"]
    assert reused["metrics"] == loaded["metrics"] == reloaded["metrics"]
    assert reused["inputs_id"] == loaded["inputs_id"] == reloaded["inputs_id"]
    assert reused["predictions"]
    assert reused["metrics"]["count"] == 3


def test_corrupt_stored_validation_refuses_the_next_trial_without_reload(journal, monkeypatch):
    prepare_recording(journal, monkeypatch)
    journal.recorder.start("2024", "2023", "z-last")
    journal.recorder.record("2024", "2023", "z-last", forecasts=rows(4))
    corrupt_stored_record(journal, "validation:" + COHORT)
    forbid_labels(monkeypatch)
    started = journal.recorder.start("2024", "2023", "a-first")
    with pytest.raises(OpsError) as caught:
        journal.recorder.record("2024", "2023", "a-first", forecasts=rows(4))
    assert caught.value.problem == fail(
        "EXPERIMENT_IDENTITY_CONFLICT", "search journal artifact could not be verified").problem
    assert journal.recorder._read("terminal:" + started["trial_id"]) is None


def test_tampered_inputs_id_validation_refuses_before_any_reload(journal, monkeypatch):
    tampered = validation_evidence()
    tampered["inputs_id"] = content_hash(["tampered cohort"])
    journal.recorder._append("validation:" + COHORT, tampered)
    forbid_labels(monkeypatch)
    started = journal.recorder.start("2024", "2023", "z-last")
    with pytest.raises(OpsError) as caught:
        journal.recorder.record("2024", "2023", "z-last", forecasts=rows(4))
    assert caught.value.problem == fail(
        "EXPERIMENT_IDENTITY_CONFLICT", "search validation evidence differs").problem
    assert journal.recorder._read("terminal:" + started["trial_id"]) is None
    assert journal.recorder._read("validation:" + COHORT) == tampered  # refused, never replaced by a reload


def test_corrupt_stored_validation_refuses_native_execute_without_reload(native, monkeypatch):
    native.recorder.execute("2024", "2023", "z-last", limits=LIMITS)
    corrupt_stored_record(native, "validation:" + COHORT)
    forbid_labels(monkeypatch)
    key = content_hash(["plan", "2024", "2023", "a-first"])
    with pytest.raises(OpsError) as caught:
        native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    assert caught.value.problem == fail(
        "EXPERIMENT_IDENTITY_CONFLICT", "search journal artifact could not be verified").problem
    assert native.recorder._read("terminal:" + key) is None


def test_each_cohort_loads_labels_exactly_once(journal, monkeypatch):
    prepare_recording(journal, monkeypatch)
    add_inner_fold(journal)
    calls = counting_labels(monkeypatch)
    for outer_year, inner_year, candidate in (("2024", "2023", "z-last"), ("2024", "2022", "z-last"),
                                              ("2024", "2023", "a-first"), ("2024", "2022", "a-first")):
        journal.recorder.start(outer_year, inner_year, candidate)
        assert journal.recorder.record(outer_year, inner_year, candidate,
                                       forecasts=rows(4))["status"] == "succeeded"
    assert len(calls) == 2  # one load per cohort; revisits reuse the stored record
    assert validation_rows(journal) == 2
    assert journal.recorder._read("validation:" + COHORT) == validation_evidence()
    assert (journal.recorder._read("validation:" + content_hash(["plan", "2024", "2022"]))
            == validation_evidence())
