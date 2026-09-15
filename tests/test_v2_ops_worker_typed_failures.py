"""Real nightly attempt 9 defect: a worker's typed ``OpsError`` (an
``engine.v2.ops.legacy_adapter._action_score`` ``VALIDATION_FAILED``, in the
real incident) was flattened to a generic retryable ``WORKER_FAILED``.
``engine/v2/ops/worker.py::main`` caught ``BaseException`` and wrote only
``{"failure": "WORKER_FAILED"}`` to the result pipe; ``supervisor.py::
_commit_success`` then raised a made-up generic problem without ever looking
at the worker's own result bytes -- the deterministic validation failure was
retried (wasting an attempt) and its ``missing``/``unplanned`` details were
only ever visible in the private ``worker.stderr``.

Two layers, tested separately:

* ``engine/v2/ops/worker.py`` -- direct, in-process ``main()`` calls (no
  subprocess) with ``dispatch`` monkeypatched to raise, following the
  pattern of testing the fixed entrypoint's exception handling in isolation.
* ``engine/v2/ops/supervisor.py`` -- a real ``Service``/catalog/``tick()``
  loop against a real subprocess, whose argv is swapped for a tiny stub (the
  seam ``tests/test_v2_ops_snapshot_stages.py`` and
  ``tests/test_v2_ops_checkpoint_output_names.py`` use): same env, same
  stdin envelope, same result pipe, same ``Service`` finish path. The stub
  writes exactly the result shape the real (now-fixed) ``worker.py`` would
  produce for a caught ``OpsError``, so this exercises the real recording,
  retry, diagnostic-publishing and dependent-blocking logic end to end.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import SystemClock, content_hash
from engine.v2.ops import executor, worker
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.checkpoints import artifact as artifact_ref
from engine.v2.ops.cli import explain_command
from engine.v2.ops.errors import OpsError, make_problem
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.profiles import profile_named
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, get_job, job_id_for, submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


# --------------------------------------------------------------------------
# 1. engine/v2/ops/worker.py -- direct, in-process
# --------------------------------------------------------------------------


def _run_worker_main(tmp_path, monkeypatch, *, dispatch_raises):
    staging = tmp_path / "staging"
    staging.mkdir()
    read_fd, write_fd = os.pipe()
    envelope = {"worker": "artifact_check", "parameters": {"expected_ids": []},
               "staging": str(staging), "job_id": "job_x", "attempt_id": "att_x",
               "fence": 1, "cpu_ids": list(os.sched_getaffinity(0)), "result_fd": str(write_fd)}
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(
        buffer=io.BytesIO(json.dumps(envelope).encode() + b"\n")))

    def fake_dispatch(*a, **k):
        raise dispatch_raises
    monkeypatch.setattr(worker, "dispatch", fake_dispatch)
    # worker.main() writes the result and closes write_fd itself (mirrors
    # what the real subprocess does over its inherited pipe).
    exit_code = worker.main()
    data = b""
    while chunk := os.read(read_fd, 65536):
        data += chunk
    os.close(read_fd)
    return exit_code, json.loads(data), staging


def test_worker_writes_typed_problem_for_caught_opserror(tmp_path, monkeypatch):
    exc = OpsError(make_problem(
        "VALIDATION_FAILED", "score population differs from planned inputs",
        details={"missing": ["AAA"], "unplanned": ["ZZZ"]}))
    exit_code, result, staging = _run_worker_main(tmp_path, monkeypatch, dispatch_raises=exc)
    assert exit_code == 1
    assert result["failure"] == "VALIDATION_FAILED"
    assert result["problem"] == {"code": "VALIDATION_FAILED", "category": "validation",
                                 "retryable": False,
                                 "message": "score population differs from planned inputs"}
    details = json.loads((staging / "diagnostics" / "failure_details.json").read_text())
    assert details == {"missing": ["AAA"], "unplanned": ["ZZZ"]}
    stderr = (staging / "diagnostics" / "worker.stderr").read_text()
    assert "OpsError" in stderr or "VALIDATION_FAILED" in stderr


def test_worker_plain_exception_stays_worker_failed(tmp_path, monkeypatch):
    """Extended by the ops-explain-from-logs task: an untyped exception now
    stays classified WORKER_FAILED (unchanged) but carries ``exception_type``
    and a code ``location`` in its details, so ``ops explain`` has something
    better than "worker did not complete its contract" without reading
    ``worker.stderr``. Its own message ("boom") is deliberately NOT included
    -- only the three narrow, reviewed exception branches above ever put a
    message in the details; see ``test_o31_...`` for why."""
    exit_code, result, staging = _run_worker_main(
        tmp_path, monkeypatch, dispatch_raises=RuntimeError("boom"))
    assert exit_code == 1
    assert result["failure"] == "WORKER_FAILED"
    assert result["problem"]["code"] == "WORKER_FAILED"
    assert result["problem"]["category"] == "internal"
    assert result["problem"]["retryable"] is True
    assert "boom" not in result["problem"]["message"]
    details = json.loads((staging / "diagnostics" / "failure_details.json").read_text())
    assert details["exception_type"] == "RuntimeError"
    path, _, line = details["location"].rpartition(":")
    assert path.endswith(".py") and line.isdigit()
    assert "boom" not in json.dumps(details)
    assert "boom" in (staging / "diagnostics" / "worker.stderr").read_text()


def test_worker_module_not_found_is_typed_nonretryable(tmp_path, monkeypatch):
    """Real shadow nightly attempt 13: a pinned model artifact's pickle
    named ``engine.models.ensemble``, absent from the code snapshot.
    ``import_module`` raised ``ModuleNotFoundError`` deep inside
    ``joblib.load`` (via ``score.py::_score_chooser``), which used to fall
    into the plain-``BaseException`` branch above and come back as a
    retryable ``WORKER_FAILED`` -- retried once for nothing, since a missing
    module never resolves itself. It must now be typed, non-retryable, and
    name the module."""
    exc = ModuleNotFoundError("No module named 'engine.models.ensemble'",
                              name="engine.models.ensemble")
    exit_code, result, staging = _run_worker_main(tmp_path, monkeypatch, dispatch_raises=exc)
    assert exit_code == 1
    assert result["failure"] == "INPUT_CHANGED"
    assert result["problem"]["code"] == "INPUT_CHANGED"
    assert result["problem"]["category"] == "dependency"
    assert result["problem"]["retryable"] is False
    assert "engine.models.ensemble" in result["problem"]["message"]
    details = json.loads((staging / "diagnostics" / "failure_details.json").read_text())
    assert details == {"module": "engine.models.ensemble"}
    assert "ModuleNotFoundError" in (staging / "diagnostics" / "worker.stderr").read_text()


def test_worker_value_error_is_typed_nonretryable(tmp_path, monkeypatch):
    """Real shadow nightly attempt 14: ``legacy_adapter._write_action``'s
    ``json.dumps(value, allow_nan=False)`` raised ``ValueError: Out of range
    float values are not JSON compliant: nan`` on a real (legacy-produced)
    NaN (``model_evidence.py``'s ``magnitude_spearman``), on two separate
    fresh attempts (``att_4a4bc10ceed84fe44e5bc03bd96d5660``,
    ``att_94c281762ed961ef3606eb573796a688``) -- the exact same deterministic
    exception both times, because retrying re-runs the exact same inputs
    through the exact same code. It used to fall into the plain-
    ``BaseException`` branch and come back as a retryable ``WORKER_FAILED``.
    It must now be typed VALIDATION_FAILED and non-retryable, the same shape
    the ModuleNotFoundError/INPUT_CHANGED precedent above already gets."""
    exc = ValueError("Out of range float values are not JSON compliant: nan")
    exit_code, result, staging = _run_worker_main(tmp_path, monkeypatch, dispatch_raises=exc)
    assert exit_code == 1
    assert result["failure"] == "VALIDATION_FAILED"
    assert result["problem"]["code"] == "VALIDATION_FAILED"
    assert result["problem"]["category"] == "validation"
    assert result["problem"]["retryable"] is False
    assert "ValueError" in result["problem"]["message"]
    assert "ValueError" in (staging / "diagnostics" / "worker.stderr").read_text()


# --------------------------------------------------------------------------
# 2. engine/v2/ops/supervisor.py -- real Service, stub worker subprocess
# --------------------------------------------------------------------------

#: Mirrors exactly what the fixed worker.py writes for a caught OpsError
#: (scenario "validation_failed"), a retryable typed problem (scenario
#: "retryable"), an unregistered code (scenario "malformed"), and a bare
#: WORKER_FAILED with no problem doc, i.e. the pre-existing crash path
#: (scenario "crash") -- selected by the job's own ``expected_ids`` marker so
#: one stub source covers every scenario without touching worker.py itself.
_STUB = r"""
import json, os, sys
envelope = json.loads(sys.stdin.buffer.readline())
staging = envelope["staging"]
scenario = envelope["parameters"]["expected_ids"][0]
fd = int(envelope["result_fd"])
base = dict(schema_version="worker_result.v1.0", job_id=envelope["job_id"],
            attempt_id=envelope["attempt_id"], fence=envelope["fence"])
if scenario == "validation_failed":
    diagdir = os.path.join(staging, "diagnostics")
    os.makedirs(diagdir, exist_ok=True)
    with open(os.path.join(diagdir, "failure_details.json"), "w") as fh:
        json.dump({"missing": ["AAA"], "unplanned": ["ZZZ"]}, fh)
    result = dict(base, failure="VALIDATION_FAILED",
                  problem=dict(code="VALIDATION_FAILED", category="validation",
                               retryable=False,
                               message="score population differs from planned inputs"))
elif scenario == "retryable":
    result = dict(base, failure="TRANSIENT_SOURCE",
                  problem=dict(code="TRANSIENT_SOURCE", category="source",
                               retryable=True, message="upstream feed timed out"))
elif scenario == "malformed":
    result = dict(base, failure="TOTALLY_UNKNOWN_CODE",
                  problem=dict(code="TOTALLY_UNKNOWN_CODE", category="bogus",
                               retryable=True, message="not a registered code"))
elif scenario == "crash":
    result = dict(base, failure="WORKER_FAILED")
else:
    raise SystemExit("unknown scenario: " + scenario)
os.write(fd, json.dumps(result).encode() + b"\n")
sys.exit(1)
"""


def _service(tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    store_root = root / "prod"
    store_root.mkdir()
    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    service = Service(conn, root, registry(), TEST_POLICY, clock=clock,
                      code_source=REPO, store_root=store_root)
    service.start()
    return service, conn, clock


def _install_stub(monkeypatch):
    real = subprocess.Popen

    def popen(args, **kwargs):
        if list(args[-2:]) == ["-m", "engine.v2.ops.worker"]:
            args = [sys.executable, "-u", "-c", _STUB]
        return real(args, **kwargs)
    monkeypatch.setattr(executor.subprocess, "Popen", popen)


def _submit(conn, clock, *, key, scenario, dependency_job_ids=()):
    profile = profile_named(TEST_POLICY, "delivery")
    job = JobSpec(kind="artifact_check",
                  implementation_ref=content_hash(worker_source_manifest(REPO)), spec_hash=None,
                  environment_ref=content_hash(environment_identity(
                      profile.thread_count or profile.cpu_count)),
                  parameters={"expected_ids": [scenario]},
                  input_refs=(), dependency_job_ids=dependency_job_ids,
                  output_namespace="shadow", resource_class="delivery",
                  retry_policy_ref="bounded", checkpoint_contract_ref="receipt.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key=key, principal="operator", job=job), clock=clock)
    return job_id_for("shadow", key)


def _run_until(conn, service, job_id, states, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        service.tick()
        if get_job(conn, job_id).state in states:
            return get_job(conn, job_id)
        time.sleep(0.05)
    raise AssertionError(f"{job_id} did not reach {states} within {timeout}s "
                         f"(last state: {get_job(conn, job_id).state})")


def test_nonretryable_typed_failure_fails_on_first_attempt_and_blocks_dependents(
        tmp_path, monkeypatch):
    service, conn, clock = _service(tmp_path, "svc")
    _install_stub(monkeypatch)
    parent = _submit(conn, clock, key="parent", scenario="validation_failed")
    child = _submit(conn, clock, key="child", scenario="validation_failed",
                    dependency_job_ids=(parent,))
    job = _run_until(conn, service, parent, {"failed", "succeeded"})
    assert job.state == "failed"
    assert job.attempt_count == 1
    assert job.failure.code == "VALIDATION_FAILED"
    assert job.failure.message == "score population differs from planned inputs"
    assert job.failure.retryable is False
    assert job.failure.details == {}
    assert job.failure.diagnostic_ref is not None

    raw = conn.execute("SELECT failure_json FROM jobs WHERE job_id = ?", (parent,)).fetchone()[0]
    assert "missing" not in raw and "unplanned" not in raw

    ref = artifact_ref(conn, service.store, job.failure.diagnostic_ref)
    details = json.loads(service.store.read_verified(ref))
    assert details == {"missing": ["AAA"], "unplanned": ["ZZZ"]}

    child_job = get_job(conn, child)
    assert child_job.state == "blocked"


def test_retryable_typed_failure_is_retried(tmp_path, monkeypatch):
    service, conn, clock = _service(tmp_path, "svc")
    _install_stub(monkeypatch)
    job_id = _submit(conn, clock, key="parent", scenario="retryable")
    first = _run_until(conn, service, job_id, {"retry_wait", "failed"})
    assert first.state == "retry_wait"
    assert first.attempt_count == 1
    assert first.failure.code == "TRANSIENT_SOURCE"
    assert first.failure.retryable is True
    # Proof this was actually retried, not just left non-failed: a second
    # attempt launches once the retry delay (1s, artifact_check's policy)
    # elapses.
    deadline = time.monotonic() + 10
    while get_job(conn, job_id).attempt_count < 2 and time.monotonic() < deadline:
        service.tick()
        time.sleep(0.05)
    assert get_job(conn, job_id).attempt_count >= 2


def test_malformed_and_unknown_problem_fall_back_to_worker_failed(tmp_path, monkeypatch):
    service, conn, clock = _service(tmp_path, "svc")
    _install_stub(monkeypatch)
    for scenario in ("malformed", "crash"):
        job_id = _submit(conn, clock, key=scenario, scenario=scenario)
        job = _run_until(conn, service, job_id, {"retry_wait", "failed"})
        assert job.failure.code == "WORKER_FAILED"
        assert job.failure.diagnostic_ref is None


def test_explain_shows_diagnostic_ref_when_present(tmp_path, monkeypatch):
    service, conn, clock = _service(tmp_path, "svc")
    _install_stub(monkeypatch)
    job_id = _submit(conn, clock, key="parent", scenario="validation_failed")
    job = _run_until(conn, service, job_id, {"failed"})
    document = explain_command(SimpleNamespace(job_id=job_id), conn, service.root)
    assert document["failure"]["diagnostic_ref"] == job.failure.diagnostic_ref
    assert document["failure"]["diagnostic_ref"] is not None
