"""Same-call forecast reuse preserves bytes, error order and native checks."""
# ruff: noqa: F811
import copy
from fractions import Fraction

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_execution as execution
from tests.v2.research.test_magnitude_search_execution import LIMITS
from tests.v2.research.test_magnitude_search_execution_integrity import native  # noqa: F401
from tests.v2.research.test_magnitude_search_trial_integrity import (
    durable_evidence,
    journal,  # noqa: F401
    prepare_recording,
    rows,
)


@pytest.fixture
def recording(journal, monkeypatch):
    prepare_recording(journal, monkeypatch)
    started = journal.recorder.start("2024", "2023", "a-first")
    return journal, started["trial_id"]


def track(recorder, monkeypatch):
    original, calls = recorder._forecasts, []

    def normalize(value, *args):
        calls.append(copy.deepcopy(value))
        return original(value, *args)

    monkeypatch.setattr(recorder, "_forecasts", normalize)
    return calls


@pytest.mark.parametrize("native", [False, True])
def test_record_normalizes_once_and_preserves_canonical_evidence(recording, monkeypatch, native):
    journal, key = recording
    recorder = journal.recorder
    submitted = rows(Fraction(7, 2))[::-1]
    before = copy.deepcopy(submitted)
    if native:
        recorder._append("execution-start:" + key, {"synthetic": True})
        recorder._append("execution:" + key, {"status": "succeeded", "forecasts_id": content_hash(rows(3.5))})
    calls = track(recorder, monkeypatch)
    terminal = recorder._record("2024", "2023", "a-first", forecasts=submitted, native=native)
    assert calls == [before]
    assert submitted == before
    assert recorder._read("forecasts:" + key) == rows(3.5)
    assert terminal["predictions"] == [{**row, "absolute_move_pct": 4.0} for row in rows(3.5)]
    assert terminal["metrics"] == {"count": 3, "rmse": 0.5, "mae": 0.5}
    submitted[0]["predicted_absolute_move_pct"] = 999
    assert recorder._read("forecasts:" + key) == rows(3.5)


@pytest.mark.parametrize("completed", [None, [], {}, {"status": "failed"}])
def test_bad_native_completion_precedes_forecast_normalization(recording, monkeypatch, completed):
    journal, key = recording
    recorder = journal.recorder
    recorder._append("execution-start:" + key, {"synthetic": True})
    if completed is not None:
        recorder._append("execution:" + key, completed)
    before = durable_evidence(journal)
    calls = track(recorder, monkeypatch)
    with pytest.raises(OpsError) as caught:
        recorder._record("2024", "2023", "a-first", forecasts=[{}], native=True)
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "native completion differs from recording").problem
    assert calls == []
    assert durable_evidence(journal) == before


@pytest.mark.parametrize("forecasts,expected", [
    ([{}], ("INVALID_EXPERIMENT_SPEC", "trial forecasts require complete finite event-horizon coverage")),
    (rows(7), ("EXPERIMENT_IDENTITY_CONFLICT", "native completion differs from recording")),
])
def test_native_validation_and_digest_still_refuse_before_publication(recording, monkeypatch, forecasts, expected):
    journal, key = recording
    recorder = journal.recorder
    recorder._append("execution-start:" + key, {"synthetic": True})
    recorder._append("execution:" + key, {"status": "succeeded", "forecasts_id": content_hash(rows(3.5))})
    before = durable_evidence(journal)
    calls = track(recorder, monkeypatch)
    with pytest.raises(OpsError) as caught:
        recorder._record("2024", "2023", "a-first", forecasts=forecasts, native=True)
    assert caught.value.problem == fail(*expected).problem
    assert calls == [forecasts]
    assert durable_evidence(journal) == before


def test_repeated_submitted_record_rechecks_forecasts(recording, monkeypatch):
    journal, key = recording
    recorder = journal.recorder
    first = recorder.record("2024", "2023", "a-first", forecasts=rows(3.5))
    before = durable_evidence(journal)
    calls = track(recorder, monkeypatch)
    assert recorder.record("2024", "2023", "a-first", forecasts=rows(3.5)) == first
    assert calls  # This later call cannot reuse a previous normalized submission.
    with pytest.raises(OpsError) as caught:
        recorder.record("2024", "2023", "a-first", forecasts=rows(4))
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "terminal trial cannot accept another submission").problem
    assert durable_evidence(journal) == before


def test_failed_record_does_not_normalize_forecasts(recording, monkeypatch):
    journal, _ = recording
    calls = track(journal.recorder, monkeypatch)
    result = journal.recorder.record("2024", "2023", "a-first",
        failure=fail("EXPERIMENT_VARIANT_FAILED", "synthetic failure").problem)
    assert result["status"] == "failed"
    assert calls == []


def test_public_native_execute_reuses_only_record_local_normalization(native, monkeypatch):
    monkeypatch.setattr(execution.os, "sched_getaffinity", lambda pid: {0}, raising=False)
    recorder = native.recorder
    original_record, original_forecasts = recorder._record, recorder._forecasts
    recording, calls = [], []

    def normalize(value, *args):
        if recording:
            calls.append(copy.deepcopy(value))
        return original_forecasts(value, *args)

    def record(*args, **kwargs):
        recording.append(True)
        try:
            return original_record(*args, **kwargs)
        finally:
            recording.pop()

    monkeypatch.setattr(recorder, "_record", record)
    monkeypatch.setattr(recorder, "_forecasts", normalize)
    receipt = recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    assert calls == [rows(2.0)]
    assert receipt["status"] == "succeeded"
    assert receipt["fit_attempt_count"] == receipt["completed_fit_count"] == 3
    before = durable_evidence(native)
    assert recorder.execute("2024", "2023", "a-first", limits=LIMITS) == receipt
    assert durable_evidence(native) == before
