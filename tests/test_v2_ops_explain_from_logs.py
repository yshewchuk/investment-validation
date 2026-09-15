"""``ops explain``/``ops logs`` show an attempt's own step timeline, memory
and failure detail without anyone querying the catalog, staging files or an
external monitor by hand (task: make a v2 ops job explain itself from its
own logs).

Two seams, mirroring ``tests/test_v2_ops_worker_typed_failures.py``:

* A real ``Service``/catalog/``tick()`` loop against a real subprocess whose
  argv is swapped for a tiny stub (same env, same stdin envelope, same
  result pipe, same finish path) -- used for step ordering and the resource
  kill, both of which need the executor's real ≤1s memory sampling.
* ``engine/v2/ops/worker.py::main()`` called directly, in-process, for the
  crash/redaction case -- the same pattern that file uses to test the fixed
  entrypoint's exception handling in isolation.
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
from engine.v2.ops import executor, worker as worker_module
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.cli import _render_explain_text, _render_progress_text, explain_command
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, get_job, job_id_for, submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import POLICY as TINY_POLICY
from tests.ops_support import REGISTRY as TINY_REGISTRY
from tests.ops_support import TEST_POLICY, catalog, request, sample

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})

#: Three ordered steps, then a deterministic non-retryable failure -- the
#: job reaching "failed" on the first attempt is what keeps this test fast
#: and exact (no retry timing to wait out).
_STUB_STEPS = r"""
import json, os, sys, time
envelope = json.loads(sys.stdin.buffer.readline())
diag = os.path.join(envelope["staging"], "diagnostics")
os.makedirs(diag, exist_ok=True)
fd = os.open(os.path.join(diag, "steps.ndjson"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
t0 = time.monotonic()
for name in ("alpha", "beta", "gamma"):
    os.write(fd, json.dumps({"event": "start", "step": name,
             "elapsed_seconds": time.monotonic() - t0, "rss_bytes": 11 * (1 << 20)}).encode() + b"\n")
    time.sleep(0.03)
    os.write(fd, json.dumps({"event": "end", "step": name,
             "elapsed_seconds": time.monotonic() - t0, "rss_bytes": 12 * (1 << 20),
             "units": 1}).encode() + b"\n")
os.close(fd)
result = dict(schema_version="worker_result.v1.0", job_id=envelope["job_id"],
             attempt_id=envelope["attempt_id"], fence=envelope["fence"],
             failure="VALIDATION_FAILED",
             problem=dict(code="VALIDATION_FAILED", category="validation", retryable=False,
                          message="synthetic, deterministic"))
os.write(int(envelope["result_fd"]), json.dumps(result).encode() + b"\n")
sys.exit(1)
"""

#: Opens one step ("ramp"), then allocates real memory past the "delivery"
#: profile's 256 MiB cap and holds it -- the supervisor's own watchdog must
#: detect and kill this, no cooperation from the stub beyond not exiting.
_STUB_RAMP = r"""
import json, os, sys, time
envelope = json.loads(sys.stdin.buffer.readline())
diag = os.path.join(envelope["staging"], "diagnostics")
os.makedirs(diag, exist_ok=True)
fd = os.open(os.path.join(diag, "steps.ndjson"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
os.write(fd, json.dumps({"event": "start", "step": "ramp", "elapsed_seconds": 0.0,
         "rss_bytes": 0}).encode() + b"\n")
os.close(fd)
blob = bytearray(300 * (1 << 20))
blob[0] = 1
blob[-1] = 1
time.sleep(30)
"""


def _install_stub(monkeypatch, source):
    real = subprocess.Popen

    def popen(args, **kwargs):
        if list(args[-2:]) == ["-m", "engine.v2.ops.worker"]:
            args = [sys.executable, "-u", "-c", source]
        return real(args, **kwargs)
    monkeypatch.setattr(executor.subprocess, "Popen", popen)


def _service(tmp_path):
    root = tmp_path / "svc"
    root.mkdir()
    store_root = root / "prod"
    store_root.mkdir()
    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    service = Service(conn, root, registry(), TEST_POLICY, clock=clock,
                      code_source=REPO, store_root=store_root)
    service.start()
    return service, conn, clock


def _submit(conn, clock, *, key):
    profile = profile_named(TEST_POLICY, "delivery")
    job = JobSpec(kind="artifact_check",
                  implementation_ref=content_hash(worker_source_manifest(REPO)), spec_hash=None,
                  environment_ref=content_hash(environment_identity(
                      profile.thread_count or profile.cpu_count)),
                  parameters={"expected_ids": []}, input_refs=(), dependency_job_ids=(),
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
        time.sleep(0.02)
    raise AssertionError(f"{job_id} did not reach {states} within {timeout}s "
                         f"(last: {get_job(conn, job_id).state})")


# --------------------------------------------------------------------------
# 1. step timeline, in order, with durations
# --------------------------------------------------------------------------


def test_steps_appear_in_order_with_durations(tmp_path, monkeypatch):
    service, conn, clock = _service(tmp_path)
    _install_stub(monkeypatch, _STUB_STEPS)
    job_id = _submit(conn, clock, key="steps")
    job = _run_until(conn, service, job_id, {"failed"})
    assert job.attempt_count == 1

    document = explain_command(SimpleNamespace(job_id=job_id), conn, service.root)
    steps = document["attempts"][0]["steps"]
    assert [s["step"] for s in steps] == ["alpha", "beta", "gamma"]
    for step in steps:
        assert step["duration_seconds"] is not None and step["duration_seconds"] >= 0
        assert step["rss_at_end_bytes"] == 12 * (1 << 20)
        assert step["units"] == 1
    assert document["attempts"][0]["step_events_recorded"] is True

    text = _render_explain_text(document)
    assert text.index("alpha") < text.index("beta") < text.index("gamma")

    for row in document["attempts"][0]["steps"]:
        assert row["step"] in ("alpha", "beta", "gamma")


# --------------------------------------------------------------------------
# 2. resource kill: step, peak, limit, elapsed -- caught under 10s
# --------------------------------------------------------------------------


def test_resource_kill_names_step_peak_limit_and_elapsed_under_10s(tmp_path, monkeypatch):
    service, conn, clock = _service(tmp_path)
    _install_stub(monkeypatch, _STUB_RAMP)
    job_id = _submit(conn, clock, key="ramp")

    wall_start = time.monotonic()
    job = _run_until(conn, service, job_id, {"failed", "retry_wait"})
    wall_elapsed = time.monotonic() - wall_start

    assert job.failure.code == "RESOURCE_LIMIT_EXCEEDED"
    details = job.failure.details
    assert details["step"] == "ramp"
    assert details["limit_bytes"] == 256 * (1 << 20)
    assert details["peak_bytes"] > details["limit_bytes"]
    assert 0 <= details["elapsed_s"] < 10
    # The watchdog's own ≤1s sampling caught this well inside one 10s
    # heartbeat interval -- the real-world defect this task fixes.
    assert wall_elapsed < 10

    document = explain_command(SimpleNamespace(job_id=job_id), conn, service.root)
    attempt = document["attempts"][-1]
    assert attempt["failure"]["details"]["step"] == "ramp"
    text = _render_explain_text(document)
    assert "ramp" in text and "RESOURCE_LIMIT_EXCEEDED" in text


# --------------------------------------------------------------------------
# 3. crash detail + redaction: exception_type/location/diagnostic_ref, and
#    a secret-like value / price-like float never reach the pipe, the
#    published details, or explain's structured output.
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
    monkeypatch.setattr(worker_module, "dispatch", fake_dispatch)
    exit_code = worker_module.main()
    data = b""
    while chunk := os.read(read_fd, 65536):
        data += chunk
    os.close(read_fd)
    return exit_code, json.loads(data), staging


def test_crash_worker_carries_exception_type_location_and_diagnostic_ref(tmp_path, monkeypatch):
    """Real shadow nightly attempt 16: a missing code asset surfaced as a
    bare ``FileNotFoundError``, which fell into the pre-fix untyped branch
    and produced NOTHING beyond ``WORKER_FAILED`` -- the real cause lived
    only in the private ``worker.stderr``."""
    exc = FileNotFoundError(2, "No such file or directory", "/root/investing-plan/data/x.json")
    exit_code, result, staging = _run_worker_main(tmp_path, monkeypatch, dispatch_raises=exc)
    assert exit_code == 1
    assert result["failure"] == "WORKER_FAILED"
    assert result["problem"]["message"] == "worker raised an unhandled exception"
    assert "details" not in result["problem"]
    details = json.loads((staging / "diagnostics" / "failure_details.json").read_text())
    assert details["exception_type"] == "FileNotFoundError"
    path, _, line = details["location"].rpartition(":")
    assert path.endswith(".py") and line.isdigit()


def test_redaction_secret_and_price_stay_out_of_pipe_and_details(tmp_path, monkeypatch):
    """Decide-and-state-the-rule (task §5 bullet 5): an exception message is
    NEVER included for an unreviewed exception type -- class name and code
    location only -- because it cannot be vetted for a credential, an env
    var value or a price. The one place a secret-bearing message legitimately
    still lands is the private, per-attempt ``worker.stderr``, which never
    crosses the result pipe, is never published as a catalog artifact, and
    is read by ``ops explain`` only as a clearly-labelled, bounded local
    file excerpt -- never folded into ``failure.details``."""
    secret = "sk_live_51H8xJ2SECRETVALUE0000000000"
    price = "1234.56"
    exc = RuntimeError(f"upstream token {secret} failed for price {price}")

    exit_code, result, staging = _run_worker_main(tmp_path, monkeypatch, dispatch_raises=exc)
    assert exit_code == 1
    raw_pipe = json.dumps(result)
    assert secret not in raw_pipe and price not in raw_pipe

    details_text = (staging / "diagnostics" / "failure_details.json").read_text()
    assert secret not in details_text and price not in details_text
    details = json.loads(details_text)
    assert details == {"exception_type": "RuntimeError", "location": details["location"]}

    # The one documented exception: the raw private stderr file legitimately
    # carries it (never published, never sent over the pipe).
    stderr_text = (staging / "diagnostics" / "worker.stderr").read_text()
    assert secret in stderr_text and price in stderr_text


def test_progress_event_fields_carry_no_business_value(tmp_path, monkeypatch):
    """Structural guard: every ``ProgressEvent`` field the supervisor
    actually fills is a step name, a count, a byte size, a duration or a
    fixed enum-like message -- never a value read out of legacy-computed
    data (a price, a score, a PnL number). A future field added to carry
    something else would need to change this whitelist, on purpose."""
    from dataclasses import fields

    from engine.v2.contracts import ProgressEvent
    allowed = {"job_id", "attempt_id", "stage_id", "sequence", "recorded_at", "kind",
              "elapsed_seconds", "message", "completed_units", "total_units",
              "memory_current_bytes", "memory_peak_bytes", "checkpoint_ref", "eta_seconds",
              "latest_error_code", "step", "step_duration_seconds", "step_units",
              "schema_version"}
    assert {f.name for f in fields(ProgressEvent)} == allowed


# --------------------------------------------------------------------------
# 4. old-format attempts (no step events) still render
# --------------------------------------------------------------------------


def test_explain_text_and_json_on_old_format_events(tmp_path):
    """An attempt from before this task -- real shadow attempt 16's own
    shape: 11 heartbeat rows, no ``step``/``step_duration_seconds``/
    ``step_units`` keys at all in the stored JSON (not merely ``null`` --
    absent, as a pre-task writer would have left them) -- must still render,
    both as text ("no step events recorded") and as JSON, not crash."""
    conn, clock, supervisor = catalog(tmp_path)
    submit(conn, TINY_REGISTRY, TINY_POLICY, request("old"), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor,
                       clock=clock, registry=TINY_REGISTRY)
    old_body = json.dumps({
        "schema_version": "progress_event.v1.0", "job_id": claim.job_id,
        "attempt_id": claim.attempt_id, "stage_id": "tiny", "sequence": 0,
        "recorded_at": "2026-09-14T12:00:00.000000Z", "kind": "heartbeat",
        "elapsed_seconds": 82.0, "message": "worker observed",
        "completed_units": None, "total_units": None,
        "memory_current_bytes": 2498 * (1 << 20), "memory_peak_bytes": 4060 * (1 << 20),
        "checkpoint_ref": None, "eta_seconds": None, "latest_error_code": None})
    with transaction(conn):
        conn.execute("INSERT INTO progress_events (attempt_id, sequence, job_id, kind, "
                     "recorded_at, body_json) VALUES (?,?,?,?,?,?)",
                     (claim.attempt_id, 0, claim.job_id, "heartbeat",
                      "2026-09-14T12:00:00.000000Z", old_body))
    from engine.v2.ops.errors import make_problem
    commit_attempt(conn, claim.attempt_id, claim.fence,
                   Outcome(False, "verified_dead", 137,
                           make_problem("RESOURCE_LIMIT_EXCEEDED",
                                       "worker did not complete its contract")),
                   clock=clock)

    document = explain_command(SimpleNamespace(job_id=claim.job_id), conn, Path(supervisor_root(conn)))
    attempt = document["attempts"][0]
    assert attempt["steps"] == []
    assert attempt["step_events_recorded"] is False
    text = _render_explain_text(document)
    assert "no step events recorded" in text
    json.dumps(document)  # --json path never raises on the old shape


def supervisor_root(conn):
    # ``ops_support.catalog`` opens the connection directly on a file path
    # without keeping the parent directory around; recover it from sqlite's
    # own view of the open database file.
    path = conn.execute("PRAGMA database_list").fetchone()[2]
    return str(Path(path).parent)


def test_logs_follow_renders_step_events_as_text(tmp_path, monkeypatch):
    service, conn, clock = _service(tmp_path)
    _install_stub(monkeypatch, _STUB_STEPS)
    job_id = _submit(conn, clock, key="steps-follow")
    _run_until(conn, service, job_id, {"failed"})
    rows = conn.execute("SELECT body_json FROM progress_events WHERE job_id = ? "
                        "ORDER BY recorded_at, sequence", (job_id,)).fetchall()
    events = [json.loads(row[0]) for row in rows]
    lines = [_render_progress_text(event) for event in events]
    starts = [line for line in lines if "started" in line]
    ends = [line for line in lines if "done" in line]
    assert any("alpha" in line for line in starts)
    assert any("gamma" in line for line in ends)
