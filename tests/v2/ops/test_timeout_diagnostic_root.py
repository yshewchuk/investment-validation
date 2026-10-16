"""Timeout stderr belongs to the Service's artifact store, not its catalog."""
from __future__ import annotations

import os
import select
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _write_worker(tmp_path):
    child = tmp_path / "private_worker.py"
    child.write_text(
        "import faulthandler, resource, sys, threading, traceback\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        "faulthandler.enable()\n"
        "try:\n"
        "    raise RuntimeError('private-credential-590')\n"
        "except RuntimeError:\n"
        "    ending = traceback.format_exc().encode()\n"
        "boundary = b'File \\\"/private/boundary.py\\\", line 4096, in exact_boundary\\n'\n"
        "tail = boundary + b'x' * (4095 - len(boundary) - len(ending)) + b'\\n' + ending\n"
        "assert len(tail) == 4096\n"
        "sys.stderr.buffer.write(b'OutsideTailError: old message\\n' + b'x' * 5000 + b'\\nX' + tail)\n"
        "sys.stderr.flush()\n"
        "print('ready', flush=True)\n"
        "if len(sys.argv) > 1:\n"
        "    threading.Event().wait()\n")
    return child


def test_real_subprocess_timeout_pins_exact_tail_boundary(separate_roots, tmp_path):
    conn, service, claim = separate_roots
    path = _stderr_path(service.root, claim.attempt_id)
    path.parent.mkdir(parents=True)
    with path.open("ab") as stderr:
        subprocess.run([sys.executable, str(_write_worker(tmp_path))], stderr=stderr,
                       stdout=subprocess.PIPE, check=True, timeout=10)
    with pytest.raises(ops_support.RunUntilTimeout) as raised:
        ops_support.run_until(service, conn, claim.job_id, timeout=0)
    message = str(raised.value)
    # F starts exactly 4096 bytes before EOF. A 4095-byte read loses the F;
    # 4097 or larger includes the X prefix, making this frame unrecognizable.
    assert 'worker stderr tail: File "<path>", line 4096, in exact_boundary\n' in message
    assert "RuntimeError: <message redacted>" in message
    assert "OutsideTailError" not in message
    assert "private-credential-590" not in message
    assert "raise RuntimeError" not in message
    assert str(tmp_path) not in message


@pytest.mark.skipif(
    not callable(getattr(os, "pidfd_open", None)) or
    not callable(getattr(signal, "pidfd_send_signal", None)),
    reason="real traceback signaling requires both pidfd APIs")
def test_timeout_captures_real_subprocess_stderr_from_separate_store(
        separate_roots, tmp_path, monkeypatch):
    conn, service, claim = separate_roots
    path = _stderr_path(service.root, claim.attempt_id)
    path.parent.mkdir(parents=True)
    child = _write_worker(tmp_path)
    observed = []
    stat = Path.stat

    def record_stat(path, *args, **kwargs):
        result = stat(path, *args, **kwargs)
        if path.name == "worker.stderr":
            observed.append((path, result.st_size))
        return result

    with path.open("ab") as stderr:
        proc = subprocess.Popen([sys.executable, str(child), "wait"], stderr=stderr,
                                stdout=subprocess.PIPE, start_new_session=True)
    try:
        assert select.select([proc.stdout], [], [], 10)[0], "worker did not become ready"
        assert proc.stdout.readline() == b"ready\n"
        baseline = path.stat().st_size
        monkeypatch.setattr(Path, "stat", record_stat)
        identity = process_info(proc.pid, service.boot)[0]
        record_launch(conn, claim.attempt_id, claim.fence, identity,
                      clock=service.clock, lease_seconds=120)
        persist_members(conn, claim.attempt_id, (identity,))
        with pytest.raises(ops_support.RunUntilTimeout) as raised:
            ops_support.run_until(service, conn, claim.job_id, timeout=0.05)
        message = str(raised.value)
        assert len(observed) >= 2, message
        assert observed[0] == (path, baseline)
        assert all(observed_path == path for observed_path, _ in observed)
        assert observed[-1][1] > baseline
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
    with pytest.raises(ops_support.RunUntilTimeout) as raised:
        ops_support.run_until(service, conn, claim.job_id, timeout=0)
    tail = str(raised.value).split("worker stderr tail: ", 1)[1].split(
        "; tracked process diagnostics:", 1)[0]
    assert tail == ('<diagnostic text redacted>\nTraceback (most recent call last):\n'
                    '  File "<path>", line 9, in stage\n<diagnostic text redacted>\n'
                    'ValueError: <message redacted>')


@pytest.mark.parametrize("rootless", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_short_or_empty_stderr_from_explicit_and_rootless_services(
        separate_roots, tmp_path, rootless, empty):
    conn, service, claim = separate_roots
    root = tmp_path / "catalog" if rootless else service.root
    path = _stderr_path(root, claim.attempt_id)
    path.parent.mkdir(parents=True)
    path.write_text("" if empty else
                    'Traceback (most recent call last):\n'
                    '  File "/private/stage.py", line 9, in stage\n'
                    '    secret_source()\nValueError: private-message\n')
    if rootless:
        service = SimpleNamespace(tick=lambda: None)
    with pytest.raises(ops_support.RunUntilTimeout) as raised:
        ops_support.run_until(service, conn, claim.job_id, timeout=0)
    tail = str(raised.value).split("worker stderr tail: ", 1)[1].split(
        "; tracked process diagnostics:", 1)[0]
    expected = ("" if empty else
                'Traceback (most recent call last):\n'
                '  File "<path>", line 9, in stage\n<diagnostic text redacted>\n'
                'ValueError: <message redacted>')
    assert tail == expected


def test_rootless_service_without_database_path_reports_unavailable(separate_roots):
    conn, _, claim = separate_roots
    # Copy the admitted attempt into an in-memory catalog with no backing
    # path, so the rootless double has no directory for its diagnostics.
    memory = sqlite3.connect(":memory:")
    try:
        conn.backup(memory)
        memory.row_factory = sqlite3.Row
        service = SimpleNamespace(tick=lambda: None)
        assert len(attempt_receipts(memory, claim.job_id)) == 1
        with pytest.raises(ops_support.RunUntilTimeout) as raised:
            ops_support.run_until(service, memory, claim.job_id, timeout=0)
        assert "worker stderr tail: unavailable (no catalog database path);" in str(raised.value)
    finally:
        memory.close()


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
def test_terminal_jobs_return_without_timeout(separate_roots, state):
    conn, service, claim = separate_roots
    conn.execute("UPDATE jobs SET state=?,active_attempt_id=NULL WHERE job_id=?",
                 (state, claim.job_id))

    assert ops_support.run_until(service, conn, claim.job_id, timeout=0) == state
