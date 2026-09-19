"""Small real-process fault controls for the watchdog and admission gates."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from engine.v2.contracts import ProcessIdentity
from engine.v2.foundation import SystemClock
from engine.v2.ops import worker as worker_module
from engine.v2.ops.discovery import sample_capacity
from engine.v2.ops.errors import OpsError
from engine.v2.ops.executor_cgroup import probe
from engine.v2.ops.executor_watchdog import find_owners, observe, process_info, process_table, signal_owned
from engine.v2.ops.health import health
from engine.v2.ops.lifecycle import record_launch
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.provider_budget import configure_account
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.resources import decide
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.stages import registry, validate_result
from engine.v2.ops.submission import submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import POLICY, TEST_POLICY, catalog, request, sample

# Real subprocess spawn/kill/signal, a real cgroup probe, and real watchdog
# calls against this host's process table (see tests/conftest.py's grouping
# rule).
pytestmark = pytest.mark.xdist_group("serial")


def _child(source):
    return subprocess.Popen([sys.executable, "-u", "-c", source],
                            start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _gone(process):
    for _ in range(30):
        if process.poll() is not None:
            return True
        time.sleep(0.01)
    return process.poll() is not None


def test_term_refusal_escalates_to_kill(tmp_path):
    ready = tmp_path / "ready"
    child = _child("import pathlib,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                   "pathlib.Path(%r).write_text(\"ready\"); time.sleep(30)" % str(ready))
    try:
        for _ in range(50):
            if ready.exists():
                break
            time.sleep(0.01)
        assert ready.exists()
        identity = process_info(child.pid, "boot")[0]
        signal_owned((identity,), "boot", hard=False)
        time.sleep(0.05)
        assert child.poll() is None
        signal_owned((identity,), "boot", hard=True)
        assert _gone(child)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_process_identity_rejects_pid_reuse():
    identity = ProcessIdentity(boot_id="b", pid=7, start_ticks=10, process_group=7)
    replacement = ProcessIdentity(boot_id="b", pid=7, start_ticks=11, process_group=7)
    table = {7: (replacement, 1, "S", 1)}
    observed, alive, memory = observe((identity,), "b", table=table)
    assert not alive
    assert memory == 0
    assert observed == (identity,)


def test_parent_exit_does_not_release_unknown_descendant(tmp_path):
    child_pid = tmp_path / "child.pid"
    source = (
        "import pathlib,subprocess,sys; "
        "p=subprocess.Popen([sys.executable, \"-c\", \"import time; time.sleep(30)\"], "
        "start_new_session=True); pathlib.Path(%r).write_text(str(p.pid)); raise SystemExit"
        % str(child_pid)
    )
    parent = _child(source)
    try:
        parent_identity = process_info(parent.pid, "boot")[0]
        assert _gone(parent)
        for _ in range(50):
            if child_pid.exists():
                break
            time.sleep(0.01)
        assert child_pid.exists()
        observed, alive, _ = observe((parent_identity,), "boot")
        assert not alive
        descendant = process_info(int(child_pid.read_text()), "boot")[0]
        signal_owned((descendant,), "boot", hard=True)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait()


def test_cgroup_probe_reports_real_host_fallback():
    result = probe(Path("/sys/fs/cgroup"))
    assert result["mode"] in {"watchdog", "cgroup"}
    if not result["available"]:
        assert result["mode"] == "watchdog"


def test_bad_worker_result_cannot_succeed(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    job = submit(conn, registry(), POLICY,
                 request(kind="artifact_check", checkpoint_contract_ref="receipt.v1.0",
                         parameters={"expected_ids": []}),
                 clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                       supervisor=supervisor, clock=clock)
    with pytest.raises(OpsError):
        validate_result(claim, {"schema_version": "worker_result.v1.0",
                                "job_id": job.job_id, "attempt_id": claim.attempt_id,
                                "fence": claim.fence, "completed_ids": [],
                                "outputs": []})


def test_provider_shortage_stays_queued_without_attempt(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    configure_account(conn, "acct", "generation-1", 1, 1)
    submit(conn, registry(), POLICY, request(kind="artifact_check",
           checkpoint_contract_ref="receipt.v1.0", provider_budget_ref="acct",
           parameters={"expected_ids": []}),
           clock=clock)
    assert claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                      supervisor=supervisor, clock=clock) is None
    assert conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
    reason = conn.execute("SELECT queue_reason_json FROM jobs").fetchone()[0]
    assert "PROVIDER_BUDGET" in reason


def test_restart_quarantines_same_boot_unobserved_tree(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    boot = __import__("engine.v2.ops.recovery", fromlist=["read_boot_id"]).read_boot_id()
    epoch = begin_epoch(conn, clock=clock, boot_id=boot, pid=1)
    supervisor = __import__("engine.v2.ops.scheduler", fromlist=["Supervisor"]).Supervisor(epoch, boot)
    submit(conn, registry(), POLICY, request(kind="artifact_check",
           checkpoint_contract_ref="receipt.v1.0", parameters={"expected_ids": []},
           implementation_ref="code", environment_ref="env"), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                       supervisor=supervisor, clock=clock)
    child = _child("import time; time.sleep(30)")
    try:
        record_launch(conn, claim.attempt_id, claim.fence, process_info(child.pid, boot)[0],
                      clock=clock, lease_seconds=120)
        service = Service(conn, tmp_path, registry(), TEST_POLICY, clock=clock,
                          code_source=Path(__file__).resolve().parents[1])
        service.boot = boot
        service.start()
        state = conn.execute("SELECT process_state FROM attempts WHERE attempt_id=?",
                             (claim.attempt_id,)).fetchone()[0]
        assert state == "quarantined"
        assert conn.execute("SELECT released_at FROM resource_reservations").fetchone()[0] is None
        service.close()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_secret_value_is_rejected_before_catalog_write(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    with pytest.raises(OpsError):
        submit(conn, registry(), POLICY, request(kind="artifact_check",
               checkpoint_contract_ref="receipt.v1.0",
               parameters={"expected_ids": ["https://example/?token=synthetic-secret"]}),
                clock=clock)
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_o14_watchdog_containment_is_declared_not_implied(tmp_path):
    probe_result = probe(Path("/sys/fs/cgroup"))
    sample_ = sample_capacity(tmp_path, clock=SystemClock())
    assert sample_.executor_mode in ("watchdog", "cgroup")
    if not probe_result["available"]:
        assert sample_.executor_mode == "watchdog"
        assert sample_.containment == "best_effort"
    profile = profile_named(DEFAULT_POLICY, "legacy_score")
    admission = decide(DEFAULT_POLICY, profile, sample_, [])
    if admission.admitted:
        assert admission.resources.executor_mode == sample_.executor_mode
        assert admission.resources.containment == sample_.containment
    else:
        assert admission.reason is not None
    conn, clock, _ = catalog(tmp_path)
    document = health(conn, clock=clock)
    assert document["executor_mode"] == "watchdog"
    assert document["containment"] == "best_effort"
    document = health(conn, clock=clock, executor_mode="cgroup")
    assert document["containment"] == "kernel"


def test_o31_worker_failure_never_carries_exception_text(tmp_path, monkeypatch):
    """O31's guarantee (guides/rearchitecture_phase1_operations.md) is that a
    secret VALUE never reaches a receipt/log/status -- checked below directly
    against the secret substrings, in the full pipe payload. Since the
    ops-explain-from-logs task, an untyped exception's result legitimately
    carries a curated, reviewed ``details`` (``exception_type``, code
    ``location``) whose field NAMES happen to contain the substring
    "exception"; the old proxy assertion banning that substring anywhere in
    the payload is retired in favour of the real property -- no message text,
    typed or not, and no secret substring, anywhere in the bytes actually
    written to the pipe."""
    secret_url = "https://user:S3CRET-VALUE@api.example.invalid/v1?api_key=ANOTHER-SECRET"
    read_fd, write_fd = os.pipe()
    envelope = {
        "worker": "artifact_check",
        "parameters": {"expected_ids": [], "note": secret_url},
        "staging": str(tmp_path),
        "result_fd": write_fd,
        "job_id": "job_x",
        "attempt_id": "att_x",
        "fence": 1,
        "cpu_ids": sorted(os.sched_getaffinity(0)),
    }

    def boom(*args, **kwargs):
        raise RuntimeError(f"request failed: {secret_url}")

    monkeypatch.setenv("INVESTING_PLAN_ROOT", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(
        buffer=io.BytesIO(json.dumps(envelope).encode() + b"\n")))
    monkeypatch.setattr(worker_module, "dispatch", boom)
    try:
        rc = worker_module.main()
        data = os.read(read_fd, 1 << 20)
    finally:
        os.close(read_fd)
    assert rc == 1
    payload = json.loads(data)
    assert payload["failure"] == "WORKER_FAILED"
    assert payload["schema_version"] == "worker_result.v1.0"
    assert b"S3CRET-VALUE" not in data
    assert b"ANOTHER-SECRET" not in data
    assert "request failed" not in json.dumps(payload)
    assert "message" not in payload
    assert payload["problem"]["message"] == "worker raised an unhandled exception"
    assert "details" not in payload["problem"]
    # details land only in the private staging file, never the pipe.
    details = json.loads((tmp_path / "diagnostics" / "failure_details.json").read_text())
    assert details == {"exception_type": "RuntimeError",
                       "location": details["location"]}
    assert details["location"].endswith("test_v2_ops_executor_faults.py:235")
    assert "S3CRET-VALUE" not in json.dumps(details)


def _fake_stat_line(pid: int, *, start_ticks: int = 12345, rss_pages: int = 10, pgrp: int = 111,
                    session: int = 111, ppid: int = 1) -> str:
    """A syntactically real ``/proc/<pid>/stat`` line — enough fields that
    ``process_info`` finds ``state``..``rss`` at the same positions the
    kernel does, after ``rfind(")")`` skips past a parenthesized ``comm``.
    """
    extra = ["S", str(ppid), str(pgrp), str(session), "0", "-1", "0", "0", "0", "0", "0",
            "0", "0", "0", "0", "0", "0", "1", "0", str(start_ticks), "0", str(rss_pages)]
    return f"{pid} (fake) " + " ".join(extra)


def test_process_table_skips_a_pid_that_disappears_mid_scan(tmp_path):
    """``process_table``'s ``except (OSError, ValueError, IndexError): continue``
    guards a real kernel race: a pid ``iterdir()`` lists can exit, or have a
    ``stat`` file mid-write, before it is read. Live-subprocess tests can
    provoke this only probabilistically (A1/A7's real-process fault tests
    above do launch and reap real children, but never reliably in this exact
    window); a synthetic proc tree makes the same guard deterministic
    instead of leaving it to timing luck.
    """
    proc = tmp_path / "proc"
    good = proc / "111"
    good.mkdir(parents=True)
    (good / "stat").write_text(_fake_stat_line(111))
    bad = proc / "222"
    bad.mkdir()
    (bad / "stat").write_text("222 (fake) S")  # far too few fields: IndexError

    table = process_table("boot", proc=proc)
    assert set(table) == {111}
    assert table[111][0].start_ticks == 12345


def test_find_owners_ignores_a_foreign_uid_process_but_still_blocks_on_same_uid(tmp_path):
    """§ B1's "unreadable environ never excludes" rule is what makes recovery
    hang forever on a shared, non-root CI runner: every OTHER user's and the
    init system's own processes have an environ this process cannot read,
    so ``find_owners`` never returns empty and the attempt never proves dead.

    ``ci-quarantine-same-uid``'s fix scopes the rule to same-uid processes: a
    fork/exec descendant of our own launch always keeps our uid, so a
    foreign-uid process cannot be our escaper regardless of environ
    readability, while a same-uid process with an unreadable environ must
    still block exactly as before -- this test proves both halves with a
    synthetic ``/proc`` tree (no real permission enforcement needed: root can
    read anything, so the foreign-uid case is proved by ownership alone, the
    same way the real fix decides it).
    """
    boot = "boot-uid-test"
    self_uid = os.getuid()
    foreign_uid = self_uid + 1
    session = 222  # deliberately not launch_pid, isolating proof (c)

    proc = tmp_path / "proc"
    same_uid_pid, foreign_uid_pid = 40001, 40002
    for pid, uid in ((same_uid_pid, self_uid), (foreign_uid_pid, foreign_uid)):
        d = proc / str(pid)
        d.mkdir(parents=True)
        (d / "stat").write_text(_fake_stat_line(pid, session=session, ppid=1))
        # No "environ" file at all: read_bytes() raises FileNotFoundError,
        # the same OSError subclass a real permission-denied read raises.
        os.chown(d, uid, os.getgid())

    table = process_table(boot, proc=proc)
    assert set(table) == {same_uid_pid, foreign_uid_pid}

    blockers = {pid for pid, _ in find_owners(
        boot, launch_pid=1, launch_start_ticks=0, marker="attempt-marker-xyz",
        table=table, proc=proc)}
    assert same_uid_pid in blockers, "same-uid unreadable environ must still block"
    assert foreign_uid_pid not in blockers, "a foreign-uid process must be ignored"
