"""A failed failure-journal write must never replace the consumed attempt's cause."""
import json
import sys
import time

import pytest

from engine.v2.data.errors import fail as data_fail
from engine.v2.foundation import ArtifactError, to_document
from engine.v2.ops.errors import OpsError, fail
from experiments import distribution_run as runner
from tests.v2.research.test_distribution_run import case as case
from tests.v2.research.test_distribution_run_acceptance import snapshot

DIAGNOSTIC = "[distribution] failure journal unavailable; original failure retained; attempt remains consumed"


def primary_error(kind):
    if kind in ("ops", "retryable"):
        return fail("RESOURCE_LIMIT_EXCEEDED" if kind == "ops" else "RESOURCE_UNAVAILABLE",
            "SECRET primary ops", stage="synthetic_fit", details={"cause": "SECRET detail"},
            retry_after_seconds=7, dependency_refs=("synthetic-input",))
    if kind == "data":
        return data_fail("CONTRACT_MISMATCH", "SECRET primary data", stage="synthetic_fit", details={"cause": "SECRET detail"})
    return {"memory": MemoryError, "interrupt": KeyboardInterrupt, "exit": SystemExit}[kind]("SECRET primary " + kind)


def secondary_error(kind):
    if kind == "typed":
        return fail("RESOURCE_UNAVAILABLE", "SECRET secondary storage")
    if kind == "integrity":
        return ArtifactError("INTEGRITY_FAILED", "SECRET secondary artifact path")
    return {"os": OSError, "memory": MemoryError, "interrupt": KeyboardInterrupt, "exit": SystemExit}[kind]("SECRET secondary " + kind)


def install_failure(case, monkeypatch, primary, secondary):
    run = runner.DistributionRun(case.trials, case.request)
    run.seal(clock=case.clock)
    case.state["failure"] = primary
    source_tracebacks, writes = [], []
    fit, publish = runner.DistributionFits.fit, case.store.publish_bytes
    def observed_fit(*args, **kwargs):
        try:
            return fit(*args, **kwargs)
        except BaseException as error:
            assert error is primary
            source_tracebacks.append(error.__traceback__)
            raise
    def failed_publication(payload, **kwargs):
        if kwargs.get("schema_ref") == "magnitude_search_record.v1":
            record = json.loads(payload)
            if record["slot"] == "distribution:run-failed":
                writes.append(record)
                if secondary is not None:
                    raise secondary
        return publish(payload, **kwargs)
    monkeypatch.setattr(runner.DistributionFits, "fit", observed_fit)
    monkeypatch.setattr(case.store, "publish_bytes", failed_publication)
    return run, source_tracebacks, writes


def assert_original(caught, primary, before, tracebacks):
    assert caught.value is primary and len(tracebacks) == 1
    assert (to_document(primary.problem) if hasattr(primary, "problem") else primary.args) == before
    trace = caught.value.__traceback__
    frames = []
    while trace is not None:
        frames.append(trace)
        trace = trace.tb_next
    assert any(trace is tracebacks[0] for trace in frames)  # Original throwing-frame chain survives.
    assert sum(trace.tb_frame.f_code.co_name == "_run_contained" for trace in frames) == 1
    assert not any(trace.tb_frame.f_code.co_name == "failed_publication" for trace in frames)


def assert_consumed(case, run, writes):
    assert len(writes) == 1 and run.record("started") is not None
    assert run.record("failed") is None and run.record("completed") is None
    assert sum(call[0] == "fit" for call in case.calls) == 1
    before = snapshot(case)
    calls = [call for call in case.calls if call[0] != "contained"]
    case.state["failure"] = None
    for attempt in (run, runner.DistributionRun(case.trials, case.request)):
        with pytest.raises(OpsError) as retry:
            attempt._run_contained(case.clock, time.monotonic())
        assert retry.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem
    assert snapshot(case) == before and len(writes) == 1
    assert [call for call in case.calls if call[0] != "contained"] == calls


@pytest.mark.parametrize("primary_kind", ["ops", "retryable", "data", "memory", "interrupt", "exit"])
@pytest.mark.parametrize("secondary_kind", ["typed", "os", "integrity", "memory", "interrupt", "exit"])
def test_failure_journal_errors_preserve_primary_identity_traceback_and_consumed_attempt(case, monkeypatch, capsys, primary_kind, secondary_kind):
    primary = primary_error(primary_kind)
    before = to_document(primary.problem) if hasattr(primary, "problem") else primary.args
    run, tracebacks, writes = install_failure(case, monkeypatch, primary, secondary_error(secondary_kind))
    with pytest.raises(BaseException) as caught:
        run.run(clock=case.clock)
    assert_original(caught, primary, before, tracebacks)
    assert writes[0]["body"]["code"] == getattr(primary, "code", "INTERRUPTED")
    assert_consumed(case, run, writes)
    captured = capsys.readouterr()
    assert captured.out == "" and [line.strip() for line in captured.err.splitlines()].count(DIAGNOSTIC) == 1
    assert "SECRET" not in captured.err


@pytest.mark.parametrize("diagnostic_kind", ["os", "memory", "interrupt", "exit"])
def test_diagnostic_write_failure_cannot_replace_original_exception(case, monkeypatch, capsys, diagnostic_kind):
    primary = primary_error("data")
    before = to_document(primary.problem)
    run, tracebacks, writes = install_failure(case, monkeypatch, primary, secondary_error("os"))
    emitted = []
    class BrokenStderr:
        def write(self, text):
            emitted.append(text)
            raise secondary_error(diagnostic_kind)
    # Isolate this best-effort diagnostic from public run()'s outer progress stage.
    with monkeypatch.context() as patch:
        patch.setattr(sys, "stderr", BrokenStderr())
        with pytest.raises(BaseException) as caught:
            run._run_contained(case.clock, time.monotonic())
    assert_original(caught, primary, before, tracebacks)
    assert [text.strip() for text in emitted] == [DIAGNOSTIC]  # No retry/log amplification when the sink itself fails.
    assert_consumed(case, run, writes)
    assert "SECRET" not in capsys.readouterr().err


def test_successful_failure_journal_keeps_original_without_unavailable_diagnostic(case, monkeypatch, capsys):
    primary = primary_error("ops")
    run, tracebacks, writes = install_failure(case, monkeypatch, primary, None)
    with pytest.raises(BaseException) as caught:
        run.run(clock=case.clock)
    assert_original(caught, primary, to_document(primary.problem), tracebacks)
    assert len(writes) == 1 and run.record("failed")["code"] == primary.code
    assert run.record("started") is not None and run.record("completed") is None
    captured = capsys.readouterr()
    assert captured.out == "" and DIAGNOSTIC not in captured.err and "SECRET" not in captured.err
