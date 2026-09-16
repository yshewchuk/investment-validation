"""Coordinator completion errors keep structural causes without data values."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.ops import worker
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.cli import explain_command
from engine.v2.ops.errors import OpsError, make_problem
from engine.v2.ops.submission import get_job
from engine.v2.ops.supervisor import Service
from tests.ops_support import REGISTRY, TEST_POLICY, catalog, enqueue_claim


def _finish(tmp_path, monkeypatch, raise_error):
    conn, clock, supervisor = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)
    service = Service(conn, tmp_path, REGISTRY, TEST_POLICY, clock=clock,
                      code_source=Path(__file__).resolve().parents[1])
    monkeypatch.setattr(service, "_commit_success", raise_error)
    service._progress_state[claim.attempt_id] = object()
    service._finish(SimpleNamespace(claim=claim), {"exit_code": 0})
    assert claim.attempt_id not in service._progress_state
    return service, conn, get_job(conn, claim.job_id)


def test_completion_preserves_redacted_exception_chain(tmp_path, monkeypatch):
    secret = "sk" + "_live_" + "".join(chr(97 + i % 26) for i in range(32))
    env_value = "".join(chr(65 + i % 26) for i in range(28))
    monkeypatch.setenv("COMPLETION_TEST_TOKEN", env_value)

    def fail(*args):
        try:
            raise KeyError(secret)
        except KeyError as cause:
            raise ValueError(env_value) from cause

    service, conn, job = _finish(tmp_path, monkeypatch, fail)
    assert job.state == "failed"
    assert job.failure.code == "VALIDATION_FAILED"
    assert job.failure.retryable is False
    assert job.failure.message == "coordinator completion raised ValueError"
    assert job.failure.details == {}
    assert job.failure.diagnostic_ref is not None
    ref = artifact(conn, service.store, job.failure.diagnostic_ref)
    details_text = service.store.read_verified(ref).decode()
    details = json.loads(details_text)
    assert details["exception_type"] == "ValueError"
    assert details["causes"][0]["exception_type"] == "KeyError"
    for item in [details, *details["causes"]]:
        path, _, line = item["location"].rpartition(":")
        assert path.endswith("test_v2_ops_completion_diagnostics.py") and line.isdigit()
    explained = explain_command(SimpleNamespace(job_id=job.job_id), conn, service.root)
    assert explained["failure"]["diagnostic_ref"] == ref.artifact_id
    raw = json.dumps(explained) + details_text
    raw += "".join(row[0] for row in conn.execute(
        "SELECT failure_json FROM attempts WHERE failure_json IS NOT NULL"))
    assert secret not in raw and env_value not in raw


def test_completion_preserves_typed_problem(tmp_path, monkeypatch):
    problem = make_problem("INPUT_CHANGED", "captured inputs changed", stage="export")

    def fail(*args):
        raise OpsError(problem)

    service, conn, job = _finish(tmp_path, monkeypatch, fail)
    assert job.failure == problem
    assert conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0


@pytest.mark.parametrize("suppressed", [False, True])
def test_completion_honors_suppressed_context(tmp_path, monkeypatch, suppressed):
    def fail(*args):
        try:
            raise KeyError("private detail")
        except KeyError:
            if suppressed:
                raise RuntimeError("private detail") from None
            raise RuntimeError("private detail")

    service, conn, job = _finish(tmp_path, monkeypatch, fail)
    details = json.loads(service.store.read_verified(
        artifact(conn, service.store, job.failure.diagnostic_ref)))
    assert ("causes" in details) is not suppressed


def test_completion_diagnostic_write_failure_does_not_lose_failure(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("private detail")

    def no_space(*args):
        raise OSError("private filesystem detail")

    monkeypatch.setattr(worker, "_write_failure_details", no_space)
    service, conn, job = _finish(tmp_path, monkeypatch, fail)
    assert job.state == "failed"
    assert job.failure.message == "coordinator completion raised ValueError"
    assert job.failure.diagnostic_ref is None
