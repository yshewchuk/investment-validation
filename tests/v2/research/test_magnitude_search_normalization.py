"""Same-call forecast reuse preserves bytes, error order and native checks."""
# ruff: noqa: F811
import copy
from fractions import Fraction
from math import sqrt

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_execution as execution
from experiments import magnitude_search_trials as trials
from tests.v2.research.test_magnitude_search_execution import LIMITS
from tests.v2.research.test_magnitude_search_execution_integrity import native  # noqa: F401
from tests.v2.research.test_magnitude_search_trial_integrity import (
    durable_evidence,
    expected_problem,
    journal,  # noqa: F401
    native_inputs,
    native_inputs_id,
    prepare_recording,
    rows,
    validation_evidence,
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


def expected_terminal(inputs, forecasts):
    # Independent fixture contract, never a result returned by the recorder.
    validation = validation_evidence()
    validation["inputs_id"] = native_inputs_id(inputs)
    errors = [row["predicted_absolute_move_pct"] - 4.0 for row in forecasts]
    return {"status": "succeeded", "inputs_id": native_inputs_id(inputs),
        "validation_id": content_hash(validation), "evaluation_inputs": validation["evaluation_inputs"],
        "predictions": [{**row, "absolute_move_pct": 4.0} for row in forecasts],
        "metrics": {"count": 3, "rmse": pytest.approx(sqrt(sum(error ** 2 for error in errors) / 3), rel=1e-14, abs=1e-14),
                    "mae": sum(abs(error) for error in errors) / 3}}


@pytest.mark.parametrize("values", [(Fraction(7, 2), Fraction(9, 2), Fraction(11, 2)),
                                   (Fraction(1, 2), Fraction(3, 2), Fraction(5, 2))])
@pytest.mark.parametrize("native", [False, True])
def test_record_normalizes_once_and_preserves_canonical_evidence(recording, monkeypatch, native, values):
    journal, key = recording
    recorder = journal.recorder
    submitted = [{**row, "predicted_absolute_move_pct": value}
                 for row, value in zip(rows(), values, strict=True)][::-1]
    expected = [{**row, "predicted_absolute_move_pct": float(value)}
                for row, value in zip(rows(), values, strict=True)]
    before = copy.deepcopy(submitted)
    if native:
        recorder._append("execution-start:" + key, {"synthetic": True})
        recorder._append("execution:" + key, {"status": "succeeded", "forecasts_id": content_hash(expected)})
    calls = track(recorder, monkeypatch)
    terminal = recorder._record("2024", "2023", "a-first", forecasts=submitted, native=native)
    assert calls == [before]
    assert submitted == before
    assert recorder._read("forecasts:" + key) == expected
    assert terminal == expected_terminal(native_inputs(), expected)
    submitted[0]["predicted_absolute_move_pct"] = 999
    assert recorder._read("forecasts:" + key) == expected


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
    first = expected_terminal(native_inputs(), rows(3.5))
    assert recorder.record("2024", "2023", "a-first", forecasts=rows(3.5)) == first
    before = durable_evidence(journal)
    original, calls = recorder._forecasts, []
    matching, conflicting = rows(3.5), rows(4)
    def normalize(value, *args):
        # Receipt verification also normalizes stored rows. Only identity proves
        # that this call admitted the caller's own new submission.
        if value is matching or value is conflicting:
            calls.append(value)
        return original(value, *args)
    monkeypatch.setattr(recorder, "_forecasts", normalize)
    assert recorder.record("2024", "2023", "a-first", forecasts=matching) == first
    assert calls == [matching]
    assert calls[0] is matching
    with pytest.raises(OpsError) as caught:
        recorder.record("2024", "2023", "a-first", forecasts=conflicting)
    assert calls == [matching, conflicting]
    assert calls[1] is conflicting
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "terminal trial cannot accept another submission").problem
    assert durable_evidence(journal) == before


def test_failed_record_does_not_normalize_forecasts(recording, monkeypatch):
    journal, key = recording
    calls = track(journal.recorder, monkeypatch)
    result = journal.recorder.record("2024", "2023", "a-first",
        failure=fail("EXPERIMENT_VARIANT_FAILED", "synthetic failure").problem)
    expected = {"status": "failed", "problem": expected_problem("EXPERIMENT_VARIANT_FAILED")}
    assert result == expected
    assert journal.recorder._read("terminal:" + key) == expected
    before = durable_evidence(journal)
    assert journal.recorder.record("2024", "2023", "a-first",
        failure=fail("EXPERIMENT_VARIANT_FAILED", "different private diagnostic").problem) == expected
    assert durable_evidence(journal) == before
    assert calls == []


@pytest.mark.parametrize("values", [(1.0, 3.0, 5.0), (2.0, 4.0, 6.0)])
def test_public_native_execute_reuses_only_record_local_normalization(native, monkeypatch, values):
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
    forecasts = [{**row, "predicted_absolute_move_pct": value}
                 for row, value in zip(rows(), values, strict=True)]
    stub_fit = trials.run_fitted_trial
    def fit(*args):
        result = stub_fit(*args)  # Fixture emits synthetic matrix progress, no fitting.
        return {**result, "forecasts": copy.deepcopy(forecasts)}
    monkeypatch.setattr(trials, "run_fitted_trial", fit)
    inputs = recorder._inputs(None)  # Test fixture's supplied inputs, before execution.
    columns = recorder.registration.document["magnitude_execution"]["features"]
    fits = []
    for horizon in (5, 3, 1):
        training = {"columns": columns, "row_keys": [["train", "history", horizon]],
                    "values": [[0.0] * 11], "absolute_move_pct": [2.0]}
        prediction = {"columns": columns, "row_keys": [["event", "history", horizon]], "values": [[0.0] * 11]}
        fits.append({"horizon_sessions": horizon, "training_event_ids": ["train"],
            "evaluation_event_ids": ["event"], "training_matrix_id": content_hash(training),
            "prediction_matrix_id": content_hash(prediction)})
    key = content_hash(["plan", "2024", "2023", "a-first"])
    expected = {"attempt": {"limits": LIMITS, "candidate_id": "a-first", "trial_id": key},
        "fit_attempt_count": 3, "completed_fit_count": 3, "fits": fits, "status": "succeeded",
        "forecasts_id": content_hash(forecasts), "terminal": expected_terminal(inputs, forecasts),
        "evaluation_authority": "none"}
    receipt = recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    assert calls == [forecasts]
    stable_receipt = {name: value for name, value in receipt.items()
                      if name not in ("score_seconds", "rss_delta_kib")}
    stable_receipt["terminal"] = {name: value for name, value in stable_receipt["terminal"].items()
                                  if name not in ("score_seconds", "rss_delta_kib", "inner_metrics")}
    assert stable_receipt == expected
    assert type(receipt["score_seconds"]) is float and receipt["score_seconds"] >= 1.5
    assert type(receipt["rss_delta_kib"]) is int and receipt["rss_delta_kib"] == 12
    durable_result = recorder._read("execution-result:" + key)["result"]
    assert receipt["score_seconds"] == durable_result["score_seconds"]
    assert receipt["rss_delta_kib"] == durable_result["rss_delta_kib"]
    assert receipt["terminal"]["score_seconds"] >= durable_result["score_seconds"]
    assert receipt["terminal"]["rss_delta_kib"] == durable_result["rss_delta_kib"]
    for fit in fits:
        marker = recorder._read("fit-completed:" + key + ":" + str(fit["horizon_sessions"]))
        assert {name: value for name, value in marker.items() if name != "fit_seconds"} == fit
        assert type(marker["fit_seconds"]) is float and marker["fit_seconds"] >= 0.0
    before = durable_evidence(native)
    assert recorder.execute("2024", "2023", "a-first", limits=LIMITS) == receipt
    assert durable_evidence(native) == before

    original_read, reads = recorder._read, []
    key = content_hash(["plan", "2024", "2023", "a-first"])
    def corrupt_fresh_read(slot):
        reads.append(slot)
        value = original_read(slot)
        if slot == "execution:" + key:
            value = {**value, "completed_fit_count": 0}
        return value
    monkeypatch.setattr(recorder, "_read", corrupt_fresh_read)
    with pytest.raises(OpsError) as caught:
        recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    assert "execution:" + key in reads
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search execution accounting differs").problem
    assert durable_evidence(native) == before
