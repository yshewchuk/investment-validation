"""Timeout stderr belongs to the Service's artifact store, not its catalog."""
from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from engine.v2.ops.executor import persist_members
from engine.v2.ops.executor_watchdog import process_info
from engine.v2.ops.lifecycle import attempt_receipts, record_launch
from engine.v2.ops.supervisor import Service
from tests import ops_support
from tests.ops_support import REGISTRY, TEST_POLICY, catalog, enqueue_claim

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def separate_roots(tmp_path, monkeypatch):
    catalog_root = tmp_path / "catalog"
    catalog_root.mkdir()
    conn, clock, _ = catalog(catalog_root)
    service = Service(conn, tmp_path / "store", REGISTRY, TEST_POLICY,
                      clock=clock, code_source=REPO, store_root=tmp_path / "legacy")
    try:
        service.root.mkdir()
        service.start()
        claim = enqueue_claim(conn, clock, service.identity)
        # Hold the admitted job nonterminal; launch/liveness and diagnostics
        # are exercised separately, without unrelated supervisor sidecars.
        monkeypatch.setattr(service, "tick", lambda: None)
        yield conn, service, claim
    finally:
        service.close()
        conn.close()


def _stderr_path(root, attempt_id):
    return root / "attempts" / attempt_id / "staging" / "diagnostics" / "worker.stderr"


@pytest.mark.skipif(
    not callable(getattr(os, "pidfd_open", None)) or
    not callable(getattr(signal, "pidfd_send_signal", None)),
    reason="real traceback signaling requires both pidfd APIs")
def test_timeout_captures_real_subprocess_stderr_from_separate_store(
        separate_roots, tmp_path, monkeypatch):
    conn, service, claim = separate_roots
    path = _stderr_path(service.store.root, claim.attempt_id)
    path.parent.mkdir(parents=True)
    child = tmp_path / "private_worker.py"
    child.write_text(
        "import faulthandler, resource, sys, threading, traceback\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        "faulthandler.enable()\n"
        "sys.stderr.write('OutsideTailError: old message\\n' + 'x' * 5000 + '\\n')\n"
        "try:\n"
        "    raise RuntimeError('private-credential-590')\n"
        "except RuntimeError:\n"
        "    traceback.print_exc()\n"
        "sys.stderr.flush()\n"
        "print('ready', flush=True)\n"
        "threading.Event().wait()\n")
    observed = []
    await_growth = ops_support._await_worker_stderr_growth

    def record_growth(path, baseline):
        observed.append((path, baseline))
        await_growth(path, baseline)

    monkeypatch.setattr(ops_support, "_await_worker_stderr_growth", record_growth)
    with path.open("ab") as stderr:
        proc = subprocess.Popen([sys.executable, str(child)], stderr=stderr,
                                stdout=subprocess.PIPE, start_new_session=True)
    try:
        assert select.select([proc.stdout], [], [], 10)[0], "worker did not become ready"
        assert proc.stdout.readline() == b"ready\n"
        baseline = path.stat().st_size
        identity = process_info(proc.pid, service.boot)[0]
        record_launch(conn, claim.attempt_id, claim.fence, identity,
                      clock=service.clock, lease_seconds=120)
        persist_members(conn, claim.attempt_id, (identity,))
        with pytest.raises(ops_support.RunUntilTimeout) as raised:
            ops_support.run_until(service, conn, claim.job_id, timeout=0.05)
        message = str(raised.value)
        assert observed == [(path, baseline)], message
        assert "traceback signal sent" in message
        assert "RuntimeError: <message redacted>" in message
        assert 'File "<path>"' in message
        assert "Current thread <thread id redacted>" in message
        assert "OutsideTailError" not in message
        assert "private-credential-590" not in message
        assert "raise RuntimeError" not in message
        assert str(tmp_path) not in message
        assert "after 0.1s" in message
        assert "not retried or extended" in message
        assert len(attempt_receipts(conn, claim.job_id)) == 1
        proc.wait(timeout=10)
        assert proc.returncode == -signal.SIGABRT
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        proc.stdout.close()


def test_stderr_tail_is_latest_4096_bytes_replacement_decoded_and_sanitized(separate_roots):
    conn, service, claim = separate_roots
    path = _stderr_path(service.root, claim.attempt_id)
    path.parent.mkdir(parents=True)
    # The 4096-byte boundary cuts the UTF-8 character; replacement decoding
    # must still retain the safe traceback that follows it.
    ending = (b'\nTraceback (most recent call last):\n'
              b'  File "/private/stage.py", line 9, in stage\n'
              b'    secret_source()\nValueError: private-message\n')
    payload = (b"OutsideTailError: old message\n" + b"x" * 5000 + b"\xc3\xa9" +
               b"x" * (4095 - len(ending)) + ending)
    path.write_bytes(payload)
    tail = ops_support._worker_stderr_tail(
        conn, attempt_receipts(conn, claim.job_id), root=service.root)
    assert tail == ('<diagnostic text redacted>\nTraceback (most recent call last):\n'
                    '  File "<path>", line 9, in stage\n<diagnostic text redacted>\n'
                    'ValueError: <message redacted>')


@pytest.mark.parametrize("unreadable", ["missing", "directory"])
def test_timeout_does_not_fall_back_to_catalog_when_store_stderr_is_unreadable(
        separate_roots, tmp_path, unreadable):
    conn, service, claim = separate_roots
    decoy = _stderr_path(tmp_path / "catalog", claim.attempt_id)
    decoy.parent.mkdir(parents=True)
    decoy.write_text("WrongRootError: unrelated diagnostic\n")
    if unreadable == "directory":
        _stderr_path(service.root, claim.attempt_id).mkdir(parents=True)
    with pytest.raises(ops_support.RunUntilTimeout) as raised:
        ops_support.run_until(service, conn, claim.job_id, timeout=0)
    message = str(raised.value)
    error = "FileNotFoundError" if unreadable == "missing" else "IsADirectoryError"
    assert f"worker stderr read failed: {error}" in message
    assert "WrongRootError" not in message
    assert str(tmp_path) not in message
    assert len(attempt_receipts(conn, claim.job_id)) == 1


@pytest.mark.parametrize("state", ops_support.TERMINAL_STATES)
def test_terminal_jobs_do_not_read_timeout_diagnostics(separate_roots, monkeypatch, state):
    conn, service, claim = separate_roots
    conn.execute("UPDATE jobs SET state=?,active_attempt_id=NULL WHERE job_id=?",
                 (state, claim.job_id))

    def unexpected(*args, **kwargs):
        pytest.fail("terminal job requested timeout diagnostics")

    monkeypatch.setattr(ops_support, "_run_until_deadline_message", unexpected)
    assert ops_support.run_until(service, conn, claim.job_id, timeout=0) == state
