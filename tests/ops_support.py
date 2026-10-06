"""Small operations fixtures. No market data, numerical imports or network."""
import json
import os
import re
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from engine.v2.foundation import content_hash
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import dumps, transaction
from engine.v2.ops.diagnostics import process_family_liveness
from engine.v2.ops.lifecycle import attempt_receipts
from engine.v2.ops.outbox import enqueue, watermark
from engine.v2.ops.profiles import DEFAULT_POLICY, MIB
from engine.v2.ops.recovery import begin_epoch, read_boot_id
from engine.v2.ops.scheduler import Supervisor
from engine.v2.ops.submission import get_job
from tools.v2_ops_fixtures import Parameters, POLICY, REGISTRY, enqueue_claim, request, sample  # noqa: F401

GIB = 1 << 30


class FakeClock:
    def __init__(self):
        self.value = datetime(2026, 9, 12, tzinfo=timezone.utc)
        self.elapsed = 0.0

    def now(self):
        return self.value

    def monotonic(self):
        return self.elapsed

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)
        self.elapsed += seconds


#: ``Service.tick()`` samples the host's real free memory and this process's
#: CPU affinity, so DEFAULT_POLICY's heavy profiles (up to 11 GiB, 4-5 CPUs)
#: make admission race whatever else is using the box -- or never fit at all:
#: admission compares cpu_count against this process's affinity minus the
#: reserved CPU, so a 4-CPU host (a standard GitHub runner) cannot launch the
#: 4- and 5-CPU profiles at all. TEST_POLICY mirrors DEFAULT_POLICY with each
#: profile's memory_bytes capped small and its cpu_count at
#: ``_TEST_PROFILE_CPUS`` (3 fits any host with 4+ CPUs); thread_count stays
#: DEFAULT_POLICY's effective ``thread_count or cpu_count`` so an
#: environment_ref built from DEFAULT_POLICY still matches what launch
#: resolves under this policy.
_TEST_PROFILE_MEMORY_BYTES = 384 * MIB
_TEST_PROFILE_CPUS = 3
TEST_POLICY = replace(DEFAULT_POLICY, profiles=tuple(
    replace(p, memory_bytes=min(p.memory_bytes, _TEST_PROFILE_MEMORY_BYTES),
            cpu_count=min(p.cpu_count, _TEST_PROFILE_CPUS),
            thread_count=p.thread_count or p.cpu_count)
    for p in DEFAULT_POLICY.profiles))


def catalog(tmp_path):
    clock = FakeClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    epoch = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
    return conn, clock, Supervisor(epoch, "boot")


def seed_delivered_health_release(conn, *, release_id, requested_session, resolved_session):
    """Seed one delivered ``releases`` row and its complete delivered
    ``release_intent`` -> ``export`` receipt chain in a catalog transaction,
    then write the scope's ``releases/<scope>/CURRENT`` pointer and its
    ``nightly``/``publication`` watermark. This is synthetic test setup; the
    production producer is never called here. The release occurrence and the
    export receipt ``session`` are both the explicit ``resolved_session``;
    sessions never default."""
    scope = "shadow"
    validation = content_hash({"release_id": release_id, "scope": scope,
                               "requested_session": requested_session,
                               "session": resolved_session})
    release_key = content_hash([scope, resolved_session, validation])
    manifest = {"schema_version": "release_manifest.v1.0", "release_id": release_id,
                "occurrence": resolved_session}
    export_receipt = {"schema_version": "ledger_export_receipt.v1.0", "scope": scope,
                      "requested_session": requested_session, "session": resolved_session}
    intent_receipt = {"release_id": release_id, "bound_at": resolved_session}
    with transaction(conn):
        conn.execute(
            "INSERT INTO releases(release_id,occurrence,manifest_json,manifest_hash,"
            "expected_current,eligible,published_at,delivered_at) VALUES (?,?,?,?,?,?,?,?)",
            (release_id, resolved_session, dumps(manifest), content_hash(manifest),
             None, 1, resolved_session, resolved_session))
        enqueue(conn, "export", release_key, {"validation": validation})
        conn.execute("UPDATE outbox SET state='delivered',attempts=attempts+1,receipt_json=? "
                     "WHERE kind='export' AND logical_key=?", (dumps(export_receipt), release_key))
        enqueue(conn, "release_intent", release_key, {"validation": validation})
        conn.execute("UPDATE outbox SET state='delivered',attempts=attempts+1,receipt_json=? "
                     "WHERE kind='release_intent' AND logical_key=?",
                     (dumps(intent_receipt), release_key))
    database_file = conn.execute("PRAGMA database_list").fetchone()["file"]
    release_dir = Path(database_file).parent / "releases" / scope
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "CURRENT").write_text(release_id + "\n")
    with transaction(conn):
        watermark(conn, "nightly", scope, "publication", resolved_session, release_id,
                  clock=FakeClock())


# -- bounded admission waits ---------------------------------------------------
#
# A test driving a real ``Service.tick()`` loop depends on the HOST: ``tick()``
# samples this process's CPU affinity, MemAvailable and free disk
# (``discovery.sample_capacity``), and a job the sample cannot admit just stays
# ``queued`` with a ``queue_reason_json``. ``run_until`` and ``AdmissionWatch``
# fail the test -- as ``RESOURCE WAIT``, never a skip -- as soon as the wait is
# known to be environmental:
#
# * ``PROFILE_EXCEEDS_CAPACITY`` (capacity the host can never offer) fails on
#   the first sample.
# * The other host-dependent reasons (``HOST_RESOURCE_REASONS``) fail once the
#   job has sat continuously queued on them for ``ADMISSION_WAIT_SECONDS``. A
#   shortage that clears inside the window still just waits, as the production
#   scheduler does. That window is clamped to the caller's polling deadline
#   (``AdmissionWatch(deadline=...)``): no later than it, so a window outlasting
#   a short budget cannot fail with RESOURCE WAIT at that deadline and hide the
#   ``RunUntilTimeout`` diagnostics, which report the same queue reason.
# * Reasons from the test's OWN catalog (a dependency, a held heavy slot, a
#   store lease) are never environmental: they fall through to the caller's
#   ``run_until`` deadline, which raises ``RunUntilTimeout`` (an
#   ``AssertionError``, so a deadline assertion still fires) with the job
#   diagnostics.

TERMINAL_STATES = ("succeeded", "failed", "blocked", "cancelled")

#: ``tests/README.md`` gives every test a 600 s phase budget, so ``run_until``
#: must finish -- return or fail -- strictly inside it. The cap only ever
#: SHORTENS the wait: a caller timeout below it is preserved unchanged.
RUN_UNTIL_TIMEOUT_CAP_SECONDS = 540.0

#: Queue reasons (``engine.v2.ops.resources.decide``) whose outcome depends on
#: the host sample rather than on other jobs in the test's own catalog.
HOST_RESOURCE_REASONS = frozenset({
    "PROFILE_EXCEEDS_CAPACITY", "MEMORY_HEADROOM", "RESERVATION_BUDGET",
    "CPU_UNAVAILABLE", "DISK_SPACE"})

ADMISSION_WAIT_SECONDS = float(os.environ.get("OPS_TEST_ADMISSION_WAIT_S", "60"))


def _gib(value) -> str:
    return f"{value / GIB:.2f} GiB" if isinstance(value, (int, float)) else "?"


def _mem_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def admission_message(conn, job_id, reason: dict, waited: float, *, policy=TEST_POLICY) -> str:
    """What the job waits for, what it needs, what the host has, and what to
    do about it."""
    row = conn.execute("SELECT spec_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    spec = json.loads(row[0]) if row and row[0] else {}
    kind, profile = spec.get("kind", "?"), spec.get("resource_class", "?")
    code = reason.get("code", "?")
    needed, available = reason.get("needed") or {}, reason.get("available") or {}
    head = (f"RESOURCE WAIT (environment, not a code failure): job {job_id} "
            f"(kind {kind}, profile {profile}) stayed queued {waited:.0f}s on {code}. ")
    margin = policy.free_margin_bytes
    if code == "MEMORY_HEADROOM":
        want = needed.get("memory_bytes", 0) + needed.get("owed_unconsumed_bytes", 0)
        body = (f"Waiting for memory: needs {_gib(want)} of headroom and has "
                f"{_gib(available.get('headroom_bytes'))} (headroom = MemAvailable - "
                f"{_gib(margin)} free margin), so it needs {_gib(want + margin)} free; "
                f"MemAvailable is {_gib(_mem_available_bytes())} now. Free memory "
                f"(other agents, heavy jobs, fewer xdist workers) and rerun this file.")
    elif code == "PROFILE_EXCEEDS_CAPACITY":
        cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        body = (f"The profile needs {needed.get('cpus')} worker CPUs and "
                f"{_gib(needed.get('memory_bytes'))}. This process's CPU affinity has "
                f"{len(cpus)} CPUs, {available.get('worker_cpus')} after the "
                f"{policy.reserved_cpu_count} reserved, and memory capacity is "
                f"{_gib(available.get('capacity_bytes'))}. That cannot change during "
                f"the run: give pytest at least "
                f"{(needed.get('cpus') or 0) + policy.reserved_cpu_count} CPUs in its "
                f"affinity (no taskset, bounded_run --cores or --cpu-set narrower "
                f"than that).")
    elif code == "CPU_UNAVAILABLE":
        body = (f"Waiting for CPUs: needs {needed.get('cpus')} free worker CPUs, "
                f"{available.get('free_cpus')} free in this process's affinity.")
    elif code == "DISK_SPACE":
        body = (f"Waiting for disk: needs {_gib(needed.get('scratch_bytes'))} of scratch, "
                f"{_gib(available.get('spare_bytes'))} spare above the "
                f"{_gib(policy.min_free_disk_bytes)} floor.")
    elif code == "RESERVATION_BUDGET":
        body = (f"Waiting for reservation budget: needs {_gib(needed.get('memory_bytes'))}, "
                f"{_gib(available.get('reserved_bytes'))} of "
                f"{_gib(available.get('capacity_bytes'))} already reserved.")
    else:
        body = f"needed={needed} available={available}."
    return head + body


class AdmissionWatch:
    """Fail the test once one job's host-resource wait is known to be
    hopeless (see the block comment above). Call :meth:`check` once per poll,
    and with ``final=True`` at the caller's own deadline. A watch given the
    caller's polling ``deadline`` goes silent at that deadline: it is never
    later than it, so a wait the caller budgets past cannot steal the
    ``RunUntilTimeout`` diagnostics the caller raises there."""

    def __init__(self, conn, job_id, *, policy=TEST_POLICY, wait_seconds=None,
                 deadline=None):
        self.conn, self.job_id, self.policy = conn, job_id, policy
        self.wait_seconds = ADMISSION_WAIT_SECONDS if wait_seconds is None else wait_seconds
        #: the caller's absolute ``time.monotonic()`` polling deadline, or
        #: ``None`` for a watch whose caller has no deadline of its own
        self.deadline = deadline
        self.since = None

    def check(self, *, final=False) -> None:
        import pytest

        row = self.conn.execute("SELECT state, queue_reason_json FROM jobs WHERE job_id=?",
                                (self.job_id,)).fetchone()
        reason = json.loads(row[1]) if row and row[0] == "queued" and row[1] else None
        if reason is None or reason.get("code") not in HOST_RESOURCE_REASONS:
            self.since = None
            return
        now = time.monotonic()
        if self.deadline is not None and now >= self.deadline:
            return  # at/after the polling deadline the caller's timeout speaks
        self.since = now if self.since is None else self.since
        waited = now - self.since
        if reason["code"] == "PROFILE_EXCEEDS_CAPACITY" or final or waited >= self.wait_seconds:
            pytest.fail(admission_message(self.conn, self.job_id, reason, waited,
                                          policy=self.policy), pytrace=False)


def job_state(conn, job_id) -> str:
    return conn.execute("SELECT state FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]


def _read_progress_rows(conn, job_id):
    """This job's ``progress_events`` bodies, decoded in the same order as
    ``engine.v2.ops.cli._progress_rows`` reads them, kept local so this shared
    test helper never imports the production CLI module graph."""
    rows = conn.execute("SELECT body_json FROM progress_events WHERE job_id = ? "
                        "ORDER BY recorded_at, sequence", (job_id,)).fetchall()
    return [json.loads(row[0]) for row in rows]


_TRACEBACK_MARKER = re.compile(r"^Traceback \(most recent call last\):$")
_TRACEBACK_FRAME = re.compile(
    r"^(?P<indent>\s*)File \"(?P<path>.*)\", line (?P<line>\d+)"
    r"(?:, in (?P<func>\S+))?$")
_EXCEPTION_CLASS = re.compile(
    r"^(?P<cls>[A-Za-z_][A-Za-z0-9_.]*"
    r"(?:Error|Exception|Warning|Exit|Interrupt|Failure|Iteration))"
    r"(?:: (?P<msg>.*))?$")


def _sanitize_worker_stderr_tail(text: str) -> str:
    """Keep only safe Python traceback structure from a worker stderr tail:
    the ``Traceback (most recent call last):`` marker, frame lines with every
    frame path replaced by ``<path>`` (line number and function name kept),
    and the exception class with its message replaced by
    ``<message redacted>``. Every source-code line and every other
    unstructured stderr line becomes ``<diagnostic text redacted>``, so no
    raw paths, exception messages, source lines or free text survive. A
    non-empty tail from which no safe traceback structure remains returns
    that placeholder alone; empty input stays empty."""
    if not text:
        return text
    kept = False
    out = []
    for line in text.splitlines():
        frame = _TRACEBACK_FRAME.match(line)
        if frame is not None:
            kept = True
            rendered = (f"{frame.group('indent')}File \"<path>\", "
                        f"line {frame.group('line')}")
            if frame.group("func") is not None:
                rendered += f", in {frame.group('func')}"
            out.append(rendered)
            continue
        exception = _EXCEPTION_CLASS.match(line)
        if exception is not None:
            kept = True
            out.append(exception.group("cls") + (
                ": <message redacted>" if exception.group("msg") is not None else ""))
            continue
        if _TRACEBACK_MARKER.match(line):
            kept = True
            out.append(line)
            continue
        out.append("<diagnostic text redacted>")
    return "\n".join(out) if kept else "<diagnostic text redacted>"


def _worker_stderr_tail(conn, attempts) -> str:
    """Bounded tail (latest 4096 bytes, replacement-decoded) of the latest
    attempt's ``worker.stderr``, sanitized to safe traceback structure only
    (:func:`_sanitize_worker_stderr_tail`), the path never printed;
    ``unavailable (...)`` for an absent attempt, database path, file or read
    error. Never raises."""
    try:
        if attempts is None or not len(attempts) or attempts[-1] is None:
            return "unavailable (no attempt recorded)"
        attempt_id = getattr(attempts[-1], "attempt_id", None)
        if attempt_id is None:
            return "unavailable (no attempt id on the latest attempt receipt)"
        database_file = None
        for row in conn.execute("PRAGMA database_list").fetchall():
            if row["file"]:
                database_file = row["file"]
                break
        if not database_file:
            return "unavailable (no catalog database path)"
        path = (Path(database_file).parent / "attempts" / str(attempt_id) /
                "staging" / "diagnostics" / "worker.stderr")
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 4096))
            data = fh.read(4096)
        return _sanitize_worker_stderr_tail(data.decode("utf-8", errors="replace"))
    except Exception as exc:
        return f"unavailable (worker stderr read failed: {type(exc).__name__})"


def _tracked_process_diagnostics(conn, job_id) -> str:
    """Per tracked process identity of this job -- the distinct
    ``(pid, start_ticks, attempts.host_boot_id)`` tuples -- the delta of
    ``utime + stime`` between two complete ``/proc/<pid>/stat`` samples one
    second apart, the final state, ``/proc/<pid>/wchan`` and
    ``/proc/<pid>/task/<pid>/children``. A record is reported only when its
    stored boot id equals the live boot id and both samples' start time
    equals the stored ``start_ticks``, so a reused pid is never reported as
    this job's worker; a record whose stat sample is unreadable or that fails
    either identity check renders ``unavailable`` with its pid, no samples.
    No tracked processes reports that without sleeping. Never raises and
    never prints command lines, absolute paths, environment values, boot ids
    or raw exception text."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT pm.pid, pm.start_ticks, a.host_boot_id "
            "FROM process_members pm "
            "JOIN attempts a ON a.attempt_id = pm.attempt_id "
            "WHERE a.job_id = ?", (job_id,)).fetchall()
        identities = sorted(
            {(row[0], row[1], row[2]) for row in rows if row[0] is not None},
            key=lambda identity: (identity[0], str(identity[1]), str(identity[2])))
        if not identities:
            return "unavailable (no tracked pids)"

        def read_stat(pid):
            try:
                with open(f"/proc/{pid}/stat", "rb") as fh:
                    text = fh.read().decode("utf-8", errors="replace")
            except Exception:
                return None
            if ")" not in text:
                return None
            fields = text[text.rfind(")") + 1:].split()
            if len(fields) < 20:
                return None
            try:
                return {"state": fields[0],
                        "ticks": int(fields[11]) + int(fields[12]),
                        "start_ticks": int(fields[19])}
            except ValueError:
                return None

        def read_small(path):
            try:
                with open(path, "rb") as fh:
                    return fh.read().decode("utf-8", errors="replace").strip()
            except Exception:
                return None

        pids = sorted({identity[0] for identity in identities})

        def sample_all():
            return {pid: read_stat(pid) for pid in pids}

        boot_id = read_boot_id()
        first = sample_all()
        time.sleep(1.0)
        second = sample_all()
        parts = []
        for index, identity in enumerate(identities, start=1):
            pid, stored_start_ticks, stored_boot_id = identity
            label = f"tracked process {index} of {len(identities)} (pid {pid})"
            if stored_boot_id != boot_id:
                parts.append(f"{label}: unavailable (boot id mismatch)")
                continue
            earlier = first.get(pid)
            final = second.get(pid)
            if (earlier is None or final is None
                    or earlier.get("start_ticks") != stored_start_ticks
                    or final.get("start_ticks") != stored_start_ticks):
                parts.append(f"{label}: unavailable (stat unreadable or start time mismatch)")
                continue
            wchan = read_small(f"/proc/{pid}/wchan")
            children = read_small(f"/proc/{pid}/task/{pid}/children")
            parts.append(
                f"{label}: cpu {final['ticks'] - earlier['ticks']} ticks, "
                f"state {final['state']}, "
                f"wchan {wchan if wchan is not None else 'unavailable'}, "
                f"children {children if children is not None else 'unavailable'}")
        return "; ".join(parts)
    except Exception as exc:
        return f"unavailable (tracked process diagnostics failed: {type(exc).__name__})"


def _run_until_deadline_message(conn, job_id, state, states, timeout) -> str:
    """Deadline diagnostics from the existing ops read helpers (``get_job``,
    ``attempt_receipts``, ``diagnostics.process_family_liveness``) plus the
    module-local :func:`_read_progress_rows` reading of ``progress_events``,
    no other ad-hoc SQL: every required label always prints, an unavailable
    one rendering ``unavailable`` alone (catching ``Exception``, never
    ``BaseException``).

    The two heartbeats never blur: ``attempts.heartbeat_at`` is the fenced
    lease-renewal stamp (``lifecycle.heartbeat``), while a ``progress_events``
    row of ``kind="heartbeat"`` is only a throttled supervisor observation
    event; the lease stamp, that observation event, the latest meaningful
    non-heartbeat progress step/event and the diagnostic-only process-family
    live/tracked counts each get their own labelled field.
    """
    def field(render_one, *, missing, label):
        """Render one label; ``unavailable`` for THIS field only on error/``None``.
        The ``str()`` conversion is guarded too, so a value whose ``__str__``
        raises fails THIS field alone instead of the whole formatter."""
        try:
            value = render_one()
            if value is None:
                return missing
            return value if isinstance(value, str) else str(value)
        except Exception as exc:
            return f"unavailable ({label} failed: {type(exc).__name__})"

    def job_attr(name):
        if job is None:
            return None
        return getattr(job, name)  # AttributeError on a bare receipt -> this field unavailable

    def render_reason(reason):
        if reason is None:
            return None
        return f"{reason.code} (needed={reason.needed}, available={reason.available})"

    def render_progress(progress):
        if progress is None:
            return None
        return f"{progress.kind} at {progress.recorded_at}: {progress.message[:200]}"

    def render_latest_progress(progress):
        if progress is None:
            return None
        when = (progress.recorded_at if progress.recorded_at is not None
                else "unavailable (no recorded progress timestamp)")
        stamp = f"{progress.kind} at {when}"
        if progress.kind == "heartbeat":
            stamp += " (progress/observation event, not a lease signal)"
        return stamp

    def render_lease(attempts):
        if attempts is None or not len(attempts) or attempts[-1] is None:
            return None
        last = attempts[-1]
        heartbeat = (last.heartbeat_at if last.heartbeat_at is not None
                     else "unavailable (no recorded lease heartbeat)")
        expiry = (last.lease_expires_at if last.lease_expires_at is not None
                  else "unavailable (no recorded lease expiry)")
        return (f"attempt {last.attempt_number} ({last.state}): heartbeat at "
                f"{heartbeat}, lease expires at {expiry}")

    def render_step(events):
        steps = [event for event in events if event.get("kind") != "heartbeat"]
        if not steps:
            return None
        last = steps[-1]
        step = f" (step {last['step']})" if last.get("step") else ""
        message = f": {last['message'][:120]}" if last.get("message") else ""
        return f"{last.get('kind')} at {last.get('recorded_at')}{step}{message}"

    def render_family():
        summary = process_family_liveness(conn, job_id=job_id, boot_id=read_boot_id())
        return f"{summary['live']} live / {summary['tracked']} tracked (diagnostic only)"

    # The labels that must never vanish -- job id, last state and its
    # interpretation -- are guarded through ``field`` too, so the head below
    # interpolates strings only and cannot itself raise.
    job_text = field(lambda: str(job_id), missing="<unprintable>", label="job id")
    state_text = field(lambda: repr(state), missing="unavailable", label="state")
    classified = field(lambda: {
        "queued": "queued (never admitted)",
        "retry_wait": "retrying (waiting for re-admission)",
        "running": "admitted (running)",
        "cancelling": "admitted (cancelling)",
    }.get(state, state_text), missing=state_text, label="state interpretation")
    elapsed_text = field(lambda: f"{float(timeout):.1f}s", missing="unavailable",
                         label="deadline")
    reached_text = field(lambda: repr(list(states)), missing="unavailable",
                         label="target states")
    head = (f"run_until deadline expired after {elapsed_text}: job {job_text} "
            f"ended in state {state_text} -- {classified} -- without reaching any of "
            f"{reached_text}; the wait was bounded and is not retried or extended. ")
    try:
        try:
            job, job_note = get_job(conn, job_id), None
        except Exception as exc:
            job, job_note = None, f"unavailable (get_job failed: {type(exc).__name__})"
        count = field(lambda: job_attr("attempt_count"),
                      missing=job_note or "unavailable (no attempt count)",
                      label="attempt count")
        reason_text = field(lambda: render_reason(job_attr("queue_reason")),
                            missing=job_note or "unavailable (no queue reason recorded)",
                            label="queue/admission reason")
        progress_text = field(lambda: render_latest_progress(job_attr("latest_progress")),
                              missing=job_note or "unavailable (no progress event recorded)",
                              label="latest progress event")
        tail = field(lambda: render_progress(job_attr("latest_progress")),
                     missing=job_note or "unavailable (no progress recorded)",
                     label="log tail")
        step_text = field(lambda: render_step(_read_progress_rows(conn, job_id)),
                          missing="unavailable (no non-heartbeat progress event recorded)",
                          label="progress rows")
        try:
            attempts = attempt_receipts(conn, job_id)
        except Exception as exc:
            lease = f"unavailable (attempt_receipts failed: {type(exc).__name__})"
        else:
            lease = field(lambda: render_lease(attempts),
                          missing="unavailable (no attempts recorded)",
                          label="attempt lease heartbeat")
        family = field(render_family, missing="unavailable (no liveness summary)",
                       label="process family liveness")
        stderr_tail = field(lambda: _worker_stderr_tail(conn, attempts),
                            missing="unavailable (worker stderr unavailable)",
                            label="worker stderr tail")
        process_details = field(lambda: _tracked_process_diagnostics(conn, job_id),
                                missing="unavailable (tracked process diagnostics unavailable)",
                                label="tracked process diagnostics")
        return head + "; ".join([
            f"attempt count: {count}",
            f"queue/admission reason: {reason_text}",
            f"attempt lease heartbeat: {lease}",
            f"latest progress event: {progress_text}",
            f"latest non-heartbeat progress event: {step_text}",
            f"process family liveness: {family}",
            f"log tail: {tail}",
            f"worker stderr tail: {stderr_tail}",
            f"tracked process diagnostics: {process_details}",
        ]) + "."
    except Exception as exc:  # last resort: a diagnostic bug must never blank the failure
        na = f"unavailable (diagnostics failed: {type(exc).__name__})"
        return (head + f"attempt count: {na}; queue/admission reason: {na}; "
                f"attempt lease heartbeat: {na}; latest progress event: {na}; "
                f"latest non-heartbeat progress event: {na}; process family "
                f"liveness: {na}; log tail: {na}; worker stderr tail: {na}; "
                f"tracked process diagnostics: {na}.")


class RunUntilTimeout(AssertionError):
    """``run_until``'s per-call budget, or the cap on it, expired while the job
    was still nonterminal. The message is :func:`_run_until_deadline_message`:
    the last observed state plus the job's queue/admission reason, attempt count
    and last lease heartbeat, read through the existing ops query helpers
    (``get_job``, ``attempt_receipts``, ``process_family_liveness``). It
    subclasses ``AssertionError``, so a caller asserting on the deadline failure
    still fires. Observational only: it retries nothing, extends nothing and
    sleeps nowhere."""


def run_until(service, conn, job_id, *, timeout, states=TERMINAL_STATES, poll=0.05) -> str:
    """Tick ``service`` until ``job_id`` reaches one of ``states``, capping the wait
    at ``RUN_UNTIL_TIMEOUT_CAP_SECONDS``; an unadmittable job fails ``RESOURCE
    WAIT``, and whichever of the caller's budget and that cap expires first
    raises ``RunUntilTimeout`` with :func:`_run_until_deadline_message`, never a
    returned nonterminal state."""
    effective = min(float(timeout), RUN_UNTIL_TIMEOUT_CAP_SECONDS)
    deadline = time.monotonic() + effective
    # The watch shares this polling deadline: its admission window is never
    # later than it, so it cannot fail the test past the deadline and hide the
    # timeout diagnostics below (its own window, when it fits inside the
    # budget, still fails fast with RESOURCE WAIT as before).
    watch = AdmissionWatch(conn, job_id, policy=getattr(service, "policy", TEST_POLICY),
                           deadline=deadline)
    state = job_state(conn, job_id)
    while time.monotonic() < deadline:
        service.tick()
        state = job_state(conn, job_id)
        if state in states:
            return state
        watch.check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break  # the deadline arrived during the check above: never sleep past it
        time.sleep(min(poll, remaining))
    # A job completing during the final sleep (after the deadline elapsed) is a
    # success, not a timeout: refresh once more before failing.
    state = job_state(conn, job_id)
    if state in states:
        return state
    raise RunUntilTimeout(_run_until_deadline_message(conn, job_id, state, states, effective))
