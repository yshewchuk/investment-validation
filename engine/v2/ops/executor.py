"""Trusted, gated subprocess launch and whole-tree watchdog observation."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field

from engine.v2.foundation import ensure_directory, safe_relative_path
from engine.v2.ops.catalog import dumps, transaction
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.executor_watchdog import observe, process_info, signal_owned
from engine.v2.ops.lifecycle import record_launch

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
                    "LOKY_MAX_CPU_COUNT")


@dataclass
class Running:
    claim: object
    process: subprocess.Popen
    result_fd: int
    identities: tuple
    started: float
    data: bytearray = field(default_factory=bytearray)
    peak: int = 0
    stop_at: float | None = None
    failure: str | None = None


def launch(conn, claim, kind, store, code_root, *, clock, boot_id, lease_seconds=120):
    staging = store.staging_dir(claim.attempt_id)
    _materialize_inputs(conn, claim, store, staging)
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(code_root), "PYTHONUNBUFFERED": "1",
           "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8",
           "INVESTING_PLAN_ROOT": str(staging / "legacy")}
    env.update({key: str(claim.resources.thread_count) for key in THREAD_VARIABLES})
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    # A7: raw stderr goes to a private per-attempt file, never DEVNULL and
    # never the result pipe — a crash is otherwise a bare WORKER_FAILED.
    diagnostics = staging / "diagnostics"
    ensure_directory(diagnostics)
    stderr_fd = os.open(diagnostics / "worker.stderr",
                        os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        child = subprocess.Popen([sys.executable, "-u", "-m", "engine.v2.ops.worker"],
                                 cwd=code_root, env=env, stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=stderr_fd,
                                 pass_fds=(write_fd,), start_new_session=True)
    except BaseException:
        os.close(read_fd)
        raise
    finally:
        os.close(write_fd)
        os.close(stderr_fd)
    try:
        identity = process_info(child.pid, boot_id)[0]
        record_launch(conn, claim.attempt_id, claim.fence, identity,
                      clock=clock, lease_seconds=lease_seconds)
        persist_members(conn, claim.attempt_id, (identity,))
        envelope = dict(worker=kind.worker, parameters=claim.spec.parameters,
                        staging=str(staging), result_fd=write_fd, job_id=claim.job_id,
                        attempt_id=claim.attempt_id, fence=claim.fence,
                        cpu_ids=list(claim.resources.assigned_cpu_ids))
        child.stdin.write(json.dumps(envelope).encode() + b"\n")
        child.stdin.flush()
        child.stdin.close()
    except BaseException:
        child.kill()
        child.wait()
        os.close(read_fd)
        raise
    return Running(claim, child, read_fd, (identity,), clock.monotonic())


def _materialize_inputs(conn, claim, store, staging):
    """Copy verified predecessor artifacts into this fresh attempt root."""
    bindings = claim.spec.parameters.get("input_bindings") or {}
    for name, artifact_id in bindings.items():
        parts = safe_relative_path(name)
        if str(artifact_id).startswith("job_"):
            dependency, separator, output_name = str(artifact_id).partition("#")
            if dependency not in claim.spec.dependency_job_ids:
                raise ValueError("input binding names an undeclared dependency")
            row = conn.execute(
                "SELECT ao.artifact_id FROM attempts a JOIN attempt_outputs ao "
                "ON ao.attempt_id=a.attempt_id WHERE a.job_id=? "
                "AND a.state=? AND (?='' OR ao.name=?) "
                "ORDER BY a.attempt_number DESC LIMIT 1",
                (dependency, "succeeded", output_name, output_name)).fetchone()
            if row is None:
                raise ValueError("predecessor output is not committed")
            artifact_id = row[0]
        ref = artifact(conn, store, artifact_id)
        destination = staging.joinpath(*parts)
        ensure_directory(destination.parent)
        if destination.exists() or destination.is_symlink():
            raise ValueError("input binding destination already exists")
        destination.write_bytes(store.read_verified(ref))
        destination.chmod(0o444)


def persist_members(conn, attempt_id, identities):
    with transaction(conn):
        for identity in identities:
            conn.execute("INSERT OR IGNORE INTO process_members VALUES (?,?,?,?)",
                         (attempt_id, identity.pid, identity.start_ticks, dumps(identity)))


def poll(conn, running, *, boot_id, clock, grace_seconds=2):
    running.identities, alive, memory = observe(running.identities, boot_id)
    persist_members(conn, running.claim.attempt_id, running.identities)
    running.peak = max(running.peak, memory)
    _read_result(running)
    if memory > running.claim.resources.reserved_memory_bytes:
        running.failure = "RESOURCE_LIMIT_EXCEEDED"
    if running.failure and alive:
        stop(running, boot_id=boot_id, clock=clock, grace_seconds=grace_seconds)
    exit_code = running.process.poll()
    if exit_code is not None and alive:
        running.failure = running.failure or "WORKER_FAILED"
        stop(running, boot_id=boot_id, clock=clock, grace_seconds=grace_seconds)
    return dict(done=exit_code is not None and not alive, memory=memory, exit_code=exit_code)


def _read_result(running):
    try:
        while chunk := os.read(running.result_fd, 65536):
            running.data.extend(chunk)
            if len(running.data) > 1 << 20:
                running.failure = "VALIDATION_FAILED"
                break
    except BlockingIOError:
        pass


def stop(running, *, boot_id, clock, grace_seconds=2):
    if running.stop_at is None:
        running.stop_at = clock.monotonic()
    hard = clock.monotonic() - running.stop_at >= grace_seconds
    signal_owned(running.identities, boot_id, hard=hard)
