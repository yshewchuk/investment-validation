"""Required-case proof for ``engine.v2.ops.nightly_trigger`` (Phase 6 slice 12).

Every branch is driven with a fake clock, fake provider/plan/submit/serve
seams -- no network, no catalog, no real systemd timer. The negative controls
are explicit assertions, not status strings: a not-final probe never submits, a
past-deadline run never submits, and a rerun of a decided date never probes or
submits again. The mutual-exclusion lock is proven with a REAL
``fcntl.flock``: BUSY_LEGACY against a lock held by the test, and the hold
itself across the whole probe->submit->serve run.
"""
from __future__ import annotations

import fcntl
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.ops import nightly_trigger  # noqa: E402
from engine.v2.ops.errors import OpsError, fail  # noqa: E402
from engine.v2.ops.nightly_trigger import (  # noqa: E402
    TriggerReceipt,
    default_as_of,
    full_population,
    load_state,
    probe_finality,
    run_trigger,
    state_path,
    write_state,
)

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
AS_OF = "2026-09-25"
IN_WINDOW = datetime(2026, 9, 26, 2, 0, tzinfo=ET)
BEFORE_WINDOW = datetime(2026, 9, 25, 23, 30, tzinfo=ET)
AT_0602 = datetime(2026, 9, 26, 6, 2, tzinfo=ET)
AFTER_DEADLINE = datetime(2026, 9, 26, 6, 30, tzinfo=ET)


class FakeClock:
    def __init__(self, moment):
        self.moment = moment

    def now(self):
        return self.moment

    def monotonic(self):
        return 0.0


class FakeProvider:
    """A provider callable: fixed verdict, records every call."""

    def __init__(self, is_final, detail=None):
        self.is_final = is_final
        self.detail = detail if detail is not None else ("final" if is_final else "not final")
        self.calls = []

    def __call__(self, as_of, tickers):
        self.calls.append((as_of, tuple(tickers)))
        return self.is_final, self.detail


class FakePlan:
    """The injected plan seam: returns a fixed plan_ref and records its call."""

    def __init__(self, plan_ref="plan_ref_1"):
        self.plan_ref = plan_ref
        self.calls = []

    def __call__(self, root, as_of, tickers, context_tickers, clock, *, full_run=True,
                 expected_shadow_snapshot_id=None):
        self.calls.append({"root": Path(root), "as_of": as_of, "tickers": tuple(tickers),
                           "context_tickers": tuple(context_tickers), "full_run": full_run,
                           "expected_shadow_snapshot_id": expected_shadow_snapshot_id})
        return self.plan_ref


class FakeSubmit:
    """The injected (idempotent) submit seam; asserts the run lock is held."""

    def __init__(self, fail_times=0, crash_times=0):
        self.fail_times, self.crash_times = fail_times, crash_times
        self.calls = []

    def __call__(self, root, as_of, plan_ref, clock):
        assert _lock_held(root), "submit ran without holding the legacy nightly lock"
        self.calls.append((as_of, plan_ref))
        if self.crash_times:
            self.crash_times -= 1
            raise RuntimeError("simulated crash between submit and the state write")
        if self.fail_times:
            self.fail_times -= 1
            raise OSError("submit failed")
        return {"run_id": "run_test", "jobs": []}


class FakeServe:
    """The injected serve seam: fixed final status, asserts the lock is held."""

    def __init__(self, final="completed", fail_times=0):
        self.final, self.fail_times = final, fail_times
        self.calls = []

    def __call__(self, root, plan_ref, clock):
        assert _lock_held(root), "serve ran without holding the legacy nightly lock"
        self.calls.append((plan_ref, Path(root)))
        if self.fail_times:
            self.fail_times -= 1
            raise OSError("serve failed")
        return self.final


class FakeEnsureSnapshot:
    """The injected ``ensure_snapshot_fn`` seam (Cutover PR-7b): a fixed
    ``(readiness, snapshot_id)`` outcome, records every call.
    """

    def __init__(self, readiness="ready", snapshot_id="snap_default"):
        self.readiness, self.snapshot_id = readiness, snapshot_id
        self.calls = []

    def __call__(self, root, as_of, clock, attempt):
        self.calls.append({"root": Path(root), "as_of": as_of, "attempt": attempt})
        if self.readiness == "ready":
            return self.readiness, self.snapshot_id
        return self.readiness, None


def _lock_held(root) -> bool:
    """True while another handle holds the run lock (same-process flock)."""
    path = Path(root) / "reports" / ".nightly.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


def _run(root, clock, provider, plan, submit, serve, *, as_of=AS_OF,
         tickers=("AAA", "BBB"), context_tickers=None, ensure_snapshot_fn=None, **overrides):
    kwargs = dict(tickers=tickers,
                  context_tickers=tickers if context_tickers is None else context_tickers,
                  deadline_et="06:00", window_start_et="00:00",
                  provider=provider, clock=clock, plan_fn=plan, submit_fn=submit,
                  serve_fn=serve, ensure_snapshot_fn=ensure_snapshot_fn or FakeEnsureSnapshot())
    kwargs.update(overrides)
    return run_trigger(root, as_of, **kwargs)


# --------------------------------------------------------------------------
# 1. before the window
# --------------------------------------------------------------------------


def test_before_window_is_not_yet_without_probing_or_submitting(tmp_path):
    provider, plan, submit, serve = FakeProvider(True), FakePlan(), FakeSubmit(), FakeServe()
    receipt = _run(tmp_path, FakeClock(BEFORE_WINDOW), provider, plan, submit, serve)
    assert receipt.status == "not_yet"
    assert provider.calls == [] and plan.calls == []
    assert submit.calls == [] and serve.calls == []
    assert not state_path(tmp_path, AS_OF).exists()


# --------------------------------------------------------------------------
# 2 + 4. in window, not final: state recorded, NEVER a submit (negative control)
# --------------------------------------------------------------------------


def test_in_window_not_final_is_not_yet_and_never_submits(tmp_path):
    provider, plan, submit, serve = FakeProvider(False), FakePlan(), FakeSubmit(), FakeServe()
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, submit, serve)
    assert receipt.status == "not_yet" and receipt.detail == "not final"
    assert provider.calls == [(AS_OF, ("AAA", "BBB"))]
    assert submit.calls == [] and plan.calls == []  # the negative control
    stored = load_state(tmp_path, AS_OF)
    assert stored is not None and stored.status == "not_yet"
    assert _lock_held(tmp_path) is False  # released on exit


def test_not_final_probe_leaves_submit_untouched_as_a_shared_fake(tmp_path):
    submit = FakeSubmit()
    _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(False), FakePlan(), submit, FakeServe())
    _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(False), FakePlan(), submit, FakeServe())
    assert submit.calls == []


# --------------------------------------------------------------------------
# 3. final, in window: plan, submit, serve; full population declared
# --------------------------------------------------------------------------


def test_final_in_window_plans_submits_serves_and_completes(tmp_path):
    provider, plan = FakeProvider(True), FakePlan("plan_ref_9")
    submit, serve = FakeSubmit(), FakeServe("completed")
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, submit, serve)
    assert receipt.status == "completed" and receipt.plan_ref == "plan_ref_9"
    assert len(plan.calls) == 1
    assert plan.calls[0]["full_run"] is True
    assert plan.calls[0]["tickers"] == ("AAA", "BBB")
    assert plan.calls[0]["context_tickers"] == ("AAA", "BBB")
    assert plan.calls[0]["expected_shadow_snapshot_id"] == "snap_default"
    assert submit.calls == [(AS_OF, "plan_ref_9")]
    assert serve.calls == [("plan_ref_9", tmp_path)]
    assert load_state(tmp_path, AS_OF) == receipt
    assert not state_path(tmp_path, AS_OF).with_name(AS_OF + ".json.tmp").exists()


def test_a_serve_failure_is_recorded_as_failed(tmp_path):
    plan, submit, serve = FakePlan("plan_f"), FakeSubmit(), FakeServe("failed")
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit, serve)
    assert receipt.status == "failed" and receipt.plan_ref == "plan_f"


# --------------------------------------------------------------------------
# 5. past deadline: MISSED for both provider verdicts, no probe, no submit
# --------------------------------------------------------------------------


@pytest.mark.parametrize("is_final", [True, False])
def test_after_deadline_is_missed_and_never_submits(tmp_path, is_final):
    provider, plan, submit, serve = (FakeProvider(is_final), FakePlan(), FakeSubmit(),
                                     FakeServe())
    receipt = _run(tmp_path, FakeClock(AFTER_DEADLINE), provider, plan, submit, serve)
    assert receipt.status == "missed"
    assert provider.calls == []  # the deadline gates before any probe
    assert submit.calls == [] and plan.calls == [] and serve.calls == []
    rerun = _run(tmp_path, FakeClock(AFTER_DEADLINE), provider, plan, submit, serve)
    assert rerun.status == "idle" and rerun.plan_ref == receipt.plan_ref
    assert provider.calls == []


def test_deadline_grace_keeps_the_0602_tick_inside_and_misses_0610(tmp_path):
    # 06:02 is inside the five-minute grace: the probe still runs.
    inside = _run(tmp_path / "inside", FakeClock(AT_0602), FakeProvider(False),
                  FakePlan(), FakeSubmit(), FakeServe())
    assert inside.status == "not_yet"
    # 06:02 with a final provider completes the run.
    final = _run(tmp_path / "final", FakeClock(AT_0602), FakeProvider(True),
                 FakePlan("plan_0602"), FakeSubmit(), FakeServe())
    assert final.status == "completed"
    # 06:10 is past the grace: MISSED without a probe.
    provider = FakeProvider(True)
    late = _run(tmp_path / "late", FakeClock(datetime(2026, 9, 26, 6, 10, tzinfo=ET)),
                provider, FakePlan(), FakeSubmit(), FakeServe())
    assert late.status == "missed" and provider.calls == []


# --------------------------------------------------------------------------
# 6. rerun of a terminal date: IDLE, no probe, no write, no journal spam
# --------------------------------------------------------------------------


def test_terminal_state_is_idle_without_probing_or_writing(tmp_path):
    provider, plan, submit, serve = FakeProvider(True), FakePlan(), FakeSubmit(), FakeServe()
    first = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, submit, serve)
    assert first.status == "completed"
    path = state_path(tmp_path, AS_OF)
    before = path.read_text()
    second = _run(tmp_path, FakeClock(AFTER_DEADLINE), provider, plan, submit, serve)
    assert second.status == "idle" and second.plan_ref == first.plan_ref
    assert provider.calls == [(AS_OF, ("AAA", "BBB"))]  # never probed again
    assert len(plan.calls) == 1 and len(submit.calls) == 1 and len(serve.calls) == 1
    assert path.read_text() == before  # nothing written


def test_missed_state_is_terminal_and_never_reprobes(tmp_path):
    provider, plan, submit, serve = FakeProvider(True), FakePlan(), FakeSubmit(), FakeServe()
    first = _run(tmp_path, FakeClock(AFTER_DEADLINE), provider, plan, submit, serve)
    assert first.status == "missed"
    second = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, submit, serve)
    assert second.status == "idle"
    assert provider.calls == [] and submit.calls == []


# --------------------------------------------------------------------------
# 7. corrupt / missing / mismatched state is no prior state
# --------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [
    "{not json",
    json.dumps(["submitted"]),
    json.dumps({"as_of": "2026-01-01", "status": "submitted", "detail": "", "checked_at": ""}),
    json.dumps({"as_of": AS_OF, "status": "not-a-status"}),
    json.dumps({"as_of": AS_OF, "status": "error", "error_count": "many", "plan_ref": 7}),
])
def test_corrupt_state_falls_through_to_the_normal_decision(tmp_path, payload):
    path = state_path(tmp_path, AS_OF)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    plan, submit, serve = FakePlan("plan_fresh"), FakeSubmit(), FakeServe()
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit, serve)
    assert receipt.status == "completed" and receipt.plan_ref == "plan_fresh"
    assert len(plan.calls) == 1


def test_missing_state_falls_through_to_the_normal_decision(tmp_path):
    plan, submit, serve = FakePlan(), FakeSubmit(), FakeServe()
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit, serve)
    assert receipt.status == "completed" and len(plan.calls) == 1


def test_run_trigger_refuses_a_non_iso_as_of(tmp_path):
    with pytest.raises(OpsError):
        _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), FakePlan(), FakeSubmit(),
             FakeServe(), as_of="not-a-date")


# --------------------------------------------------------------------------
# 8. probe_finality on its own, both branches
# --------------------------------------------------------------------------


def test_probe_finality_delegates_both_branches_to_the_provider():
    final = FakeProvider(True, "market-wide published")
    assert probe_finality(AS_OF, ["AAA"], provider=final) == (True, "market-wide published")
    assert final.calls == [(AS_OF, ("AAA",))]
    pending = FakeProvider(False, "not published")
    assert probe_finality(AS_OF, ("AAA",), provider=pending) == (False, "not published")


# --------------------------------------------------------------------------
# 9. mutual exclusion: BUSY_LEGACY, and the lock held across the whole run
# --------------------------------------------------------------------------


def test_busy_legacy_with_a_real_flock_records_retry_and_never_probes(tmp_path):
    lock = tmp_path / "reports" / ".nightly.lock"
    lock.parent.mkdir(parents=True)
    holder = lock.open("a+")
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        provider, plan, submit, serve = (FakeProvider(True), FakePlan(), FakeSubmit(),
                                         FakeServe())
        receipt = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, submit, serve)
        assert receipt.status == "busy_legacy"
        assert provider.calls == []  # the lock is taken before the probe
        assert plan.calls == [] and submit.calls == [] and serve.calls == []
        assert not holder.closed  # the test's own lock is untouched
        stored = load_state(tmp_path, AS_OF)
        assert stored is not None and stored.status == "busy_legacy"
    finally:
        holder.close()
    plan, submit, serve = FakePlan("plan_after_release"), FakeSubmit(), FakeServe()
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit, serve)
    assert receipt.status == "completed" and submit.calls == [(AS_OF, "plan_after_release")]


def test_busy_legacy_does_not_overwrite_a_resumable_submitted_state(tmp_path):
    # Tick 1: a normal run reaches "submitted" with a plan_ref and the process
    # dies before serving -- the durable state on disk is "submitted".
    plan, submit = FakePlan("plan_resume"), FakeSubmit()

    def die_after_submit(root, plan_ref, clock):
        raise RuntimeError("simulated process death after submit")

    with pytest.raises(RuntimeError):
        _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit,
             die_after_submit)
    first = load_state(tmp_path, AS_OF)
    assert first is not None and first.status == "submitted"
    assert first.plan_ref == "plan_resume"
    assert len(plan.calls) == 1
    first = replace(first, error_count=1)
    write_state(tmp_path, first)

    # Tick 2: the test holds the legacy lock, so _LegacyLock returns held=False.
    lock = tmp_path / "reports" / ".nightly.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    holder = lock.open("a+")
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit,
                    die_after_submit)
    finally:
        holder.close()
    assert busy.status == "busy_legacy"
    assert busy.plan_ref == "plan_resume"  # carried forward, never None
    assert busy.error_count == 1
    assert len(plan.calls) == 1  # the busy tick never re-planned
    unchanged = load_state(tmp_path, AS_OF)
    assert unchanged == first  # nothing was persisted by the busy tick
    assert unchanged.error_count == 1

    # Tick 3: the lock is free -> resume the SAME plan_ref, no new plan.
    serve = FakeServe("completed")
    resumed = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit, serve)
    assert resumed.status == "completed" and resumed.plan_ref == "plan_resume"
    assert len(plan.calls) == 1  # the plan was built exactly once, on tick 1
    assert submit.calls == [(AS_OF, "plan_resume"), (AS_OF, "plan_resume")]
    assert serve.calls == [("plan_resume", tmp_path)]


def test_busy_legacy_with_no_resumable_state_still_persists_as_before(tmp_path):
    lock = tmp_path / "reports" / ".nightly.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    holder = lock.open("a+")
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), FakePlan(),
                       FakeSubmit(), FakeServe())
    finally:
        holder.close()
    assert receipt.status == "busy_legacy"
    assert receipt.plan_ref is None and receipt.error_count == 0
    stored = load_state(tmp_path, AS_OF)
    assert stored == receipt  # the ordinary busy path still writes to disk


def test_the_lock_is_held_for_the_whole_run_and_released_after(tmp_path):
    # FakeSubmit/FakeServe assert the lock is held during their calls; the
    # busy-tick check below proves it is released when the run returns.
    plan, submit, serve = FakePlan("plan_lock"), FakeSubmit(), FakeServe()
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit, serve)
    assert receipt.status == "completed"
    assert _lock_held(tmp_path) is False


# --------------------------------------------------------------------------
# 10. no double submit: crash between submit and the state write
# --------------------------------------------------------------------------


def test_crash_between_submit_and_state_write_resubmits_the_same_plan_ref(tmp_path):
    plan, submit, serve = FakePlan("plan_A"), FakeSubmit(crash_times=1), FakeServe()
    with pytest.raises(RuntimeError):
        _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit, serve)
    stored = load_state(tmp_path, AS_OF)
    assert stored is not None and stored.status == "submitting" and stored.plan_ref == "plan_A"
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit, serve)
    assert receipt.status == "completed"
    assert len(plan.calls) == 1  # never re-planned
    assert submit.calls == [(AS_OF, "plan_A"), (AS_OF, "plan_A")]  # same ref, no new plan


def test_crash_between_plan_fn_and_the_submitting_write_replans_but_never_double_submits(
    tmp_path, monkeypatch,
):
    """Issue #186: a crash between plan_fn returning and _ensure_plan_ref's
    "submitting" receipt write leaves no durable record of the built
    plan_ref. The next tick's fresh _decide calls plan_fn again -- proving
    the accepted-risk behavior engine/v2/ops/ARCHITECTURE.md's
    nightly_trigger.py table now documents: the first plan is orphaned
    (never referenced again), but the run still completes cleanly under the
    SECOND plan_ref, with no double-submission."""
    plan = FakePlan("plan_ORPHANED")
    real_record = nightly_trigger._record
    crash_armed = {"on": True}

    def crashing_record(root, receipt):
        if crash_armed["on"] and receipt.status == "submitting":
            crash_armed["on"] = False
            raise RuntimeError("simulated crash before the submitting receipt lands")
        return real_record(root, receipt)

    monkeypatch.setattr(nightly_trigger, "_record", crashing_record)
    with pytest.raises(RuntimeError):
        _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, FakeSubmit(), FakeServe())
    assert load_state(tmp_path, AS_OF) is None  # nothing durable recorded the orphaned plan_ref
    assert len(plan.calls) == 1

    plan.plan_ref = "plan_SECOND"  # the resumed tick's fresh plan differs from the orphan
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, FakeSubmit(), FakeServe())
    assert receipt.status == "completed" and receipt.plan_ref == "plan_SECOND"
    assert len(plan.calls) == 2  # plan_fn ran twice: the orphaned build, then the real one
    stored = load_state(tmp_path, AS_OF)
    assert stored is not None and stored.plan_ref == "plan_SECOND"  # the orphan is never referenced again


def test_submitted_state_resumes_serving_without_replanning(tmp_path):
    write_state(tmp_path, TriggerReceipt(as_of=AS_OF, status="submitted", detail="crash",
                                         checked_at="2026-09-26T06:00:00Z", plan_ref="plan_X"))
    provider, plan, submit, serve = FakeProvider(True), FakePlan(), FakeSubmit(), FakeServe()
    ensure_snapshot = FakeEnsureSnapshot()
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, submit, serve,
                   ensure_snapshot_fn=ensure_snapshot)
    assert receipt.status == "completed" and receipt.plan_ref == "plan_X"
    assert provider.calls == [] and plan.calls == []
    assert submit.calls == [(AS_OF, "plan_X")]  # idempotent no-op resubmission
    assert serve.calls == [("plan_X", tmp_path)]
    assert ensure_snapshot.calls == []  # the resume branch never re-verifies the snapshot


def test_error_after_a_failed_submit_keeps_the_plan_ref(tmp_path):
    plan, submit, serve = FakePlan("plan_err"), FakeSubmit(fail_times=1), FakeServe()
    first = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit, serve)
    assert first.status == "error" and first.plan_ref == "plan_err" and first.error_count == 1
    second = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit, serve)
    assert second.status == "completed"
    assert len(plan.calls) == 1
    assert submit.calls == [(AS_OF, "plan_err"), (AS_OF, "plan_err")]


# --------------------------------------------------------------------------
# 11. error -> failed_setup after three consecutive errors, exit-once semantics
# --------------------------------------------------------------------------


def test_three_consecutive_setup_errors_become_failed_setup_once(tmp_path):
    provider, plan = FakeProvider(True), FakePlan("plan_boom")
    submit, serve = FakeSubmit(fail_times=3), FakeServe()
    statuses, counts = [], []
    for _ in range(3):
        receipt = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, submit, serve)
        statuses.append(receipt.status)
        counts.append(receipt.error_count)
    assert statuses == ["error", "error", "failed_setup"]
    assert counts == [1, 2, 3]
    assert provider.calls == [(AS_OF, ("AAA", "BBB"))]  # probed once, then resumed
    assert len(plan.calls) == 1 and len(submit.calls) == 3
    terminal = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, submit, serve)
    assert terminal.status == "idle"
    assert len(submit.calls) == 3  # a terminal tick never submits again


# --------------------------------------------------------------------------
# 11b. timed_out: resumable like the crash/error states, terminal after three
# --------------------------------------------------------------------------


def test_a_timed_out_serve_is_recorded_as_timed_out_and_releases_the_lock(tmp_path):
    plan, submit, serve = FakePlan("plan_tmid"), FakeSubmit(), FakeServe("timed_out")
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit, serve)
    assert receipt.status == "timed_out" and receipt.plan_ref == "plan_tmid"
    stored = load_state(tmp_path, AS_OF)
    assert stored is not None and stored.status == "timed_out" and stored.error_count == 1
    assert _lock_held(tmp_path) is False


def test_a_timed_out_run_resumes_the_same_plan_ref_next_tick(tmp_path):
    first_plan = FakePlan("plan_X")
    first = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), first_plan,
                 FakeSubmit(), FakeServe("timed_out"))
    assert first.status == "timed_out" and first.plan_ref == "plan_X"
    second_plan = FakePlan("plan_OTHER")
    second = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), second_plan,
                  FakeSubmit(), FakeServe("completed"))
    assert second.status == "completed" and second.plan_ref == "plan_X"
    assert second_plan.calls == []


def test_a_pre_plan_timeout_resumes_past_the_window_close(tmp_path):
    # Cutover PR-7b-2 gate-round-4 fix: a pre-plan ensure_snapshot_fn timeout
    # carries no plan_ref, so it cannot rely on the same "plan_ref is set"
    # signal the post-plan timed_out case uses to skip _decide's window
    # check. Tick 1 (in window) times out waiting on the snapshot import.
    # Tick 2 runs well after the retry window has closed (06:05 ET) --
    # proving the fix resumes anyway instead of falling into _decide and
    # recording a terminal "missed".
    provider = FakeProvider(True)
    first_ensure = FakeEnsureSnapshot(readiness="timed_out")
    first = _run(tmp_path, FakeClock(IN_WINDOW), provider, FakePlan("plan_should_not_build"),
                 FakeSubmit(), FakeServe(), ensure_snapshot_fn=first_ensure)
    assert first.status == "timed_out" and first.plan_ref is None
    assert first_ensure.calls == [{"root": tmp_path, "as_of": AS_OF, "attempt": 0}]

    plan, submit, serve = FakePlan("plan_late_snapshot"), FakeSubmit(), FakeServe("completed")
    second_ensure = FakeEnsureSnapshot(readiness="ready", snapshot_id="snap_late")
    second = _run(tmp_path, FakeClock(AFTER_DEADLINE), provider, plan, submit, serve,
                  ensure_snapshot_fn=second_ensure)
    assert second.status == "completed" and second.plan_ref == "plan_late_snapshot"
    assert second_ensure.calls == [{"root": tmp_path, "as_of": AS_OF, "attempt": 0}]
    assert provider.calls == [(AS_OF, ("AAA", "BBB"))]  # probed once, on tick 1, never again
    assert plan.calls[0]["expected_shadow_snapshot_id"] == "snap_late"


def test_busy_legacy_does_not_overwrite_a_pending_pre_plan_timeout(tmp_path):
    ensure = FakeEnsureSnapshot(readiness="timed_out")
    first = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), FakePlan(), FakeSubmit(),
                 FakeServe(), ensure_snapshot_fn=ensure)
    assert first.status == "timed_out" and first.plan_ref is None

    lock = tmp_path / "reports" / ".nightly.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    holder = lock.open("a+")
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = _run(tmp_path, FakeClock(AFTER_DEADLINE), FakeProvider(True), FakePlan(),
                    FakeSubmit(), FakeServe(), ensure_snapshot_fn=FakeEnsureSnapshot())
    finally:
        holder.close()
    assert busy.status == "busy_legacy" and busy.plan_ref is None
    unchanged = load_state(tmp_path, AS_OF)
    assert unchanged is not None and unchanged.status == "timed_out"  # preserved, not overwritten


def test_three_consecutive_pre_plan_timeouts_become_failed(tmp_path):
    # Mirrors test_three_consecutive_timeouts_become_failed, but for the
    # PRE-plan ensure_snapshot_fn timeout instead of the post-plan serve_fn
    # one -- each retry after the first is reached through run_trigger's
    # (round-4-broadened) resume branch, since plan_ref is None throughout.
    provider = FakeProvider(True)
    ensure = FakeEnsureSnapshot(readiness="timed_out")
    plan = FakePlan("plan_should_not_build")
    statuses, counts = [], []
    for _ in range(3):
        receipt = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, FakeSubmit(),
                       FakeServe(), ensure_snapshot_fn=ensure)
        statuses.append(receipt.status)
        counts.append(receipt.error_count)
    assert statuses == ["timed_out", "timed_out", "failed"]
    assert counts == [1, 2, 3]
    assert provider.calls == [(AS_OF, ("AAA", "BBB"))]  # probed once, on tick 1, never again
    assert plan.calls == []  # the plan was never built -- the snapshot import never finished
    assert ensure.calls == [
        {"root": tmp_path, "as_of": AS_OF, "attempt": 0},
        {"root": tmp_path, "as_of": AS_OF, "attempt": 0},
        {"root": tmp_path, "as_of": AS_OF, "attempt": 0},
    ]
    terminal = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, FakeSubmit(), FakeServe(),
                    ensure_snapshot_fn=ensure)
    assert terminal.status == "idle"
    assert len(ensure.calls) == 3  # a terminal tick never re-checks the snapshot


def test_a_pre_plan_terminal_failure_resumes_past_the_window_close(tmp_path):
    # Gate-round-5 fix: the SAME defect round 4 fixed for "timed_out" also
    # applies to a pre-plan "error" (plan_ref=None) -- e.g. ensure_snapshot_fn
    # raising a terminal INPUT_CHANGED failure late in the day, after the
    # retry window has already closed. Tick 1 (in window) fails terminally.
    # Tick 2 runs well after the window's close -- proving the fix resumes
    # anyway (a fresh attempt, new snapshot_attempt) instead of falling into
    # _decide and recording a terminal "missed".
    provider = FakeProvider(True)

    class ExplodingEnsure:
        def __init__(self):
            self.calls = []

        def __call__(self, root, as_of, clock, attempt):
            self.calls.append(attempt)
            raise fail("INPUT_CHANGED", "boom")

    first = _run(tmp_path, FakeClock(IN_WINDOW), provider, FakePlan("plan_should_not_build"),
                 FakeSubmit(), FakeServe(), ensure_snapshot_fn=ExplodingEnsure())
    assert first.status == "error" and first.plan_ref is None
    assert first.snapshot_attempt == 1  # INPUT_CHANGED bumps it

    plan, submit, serve = FakePlan("plan_late_recovery"), FakeSubmit(), FakeServe("completed")
    second_ensure = FakeEnsureSnapshot(readiness="ready", snapshot_id="snap_recovery")
    second = _run(tmp_path, FakeClock(AFTER_DEADLINE), provider, plan, submit, serve,
                  ensure_snapshot_fn=second_ensure)
    assert second.status == "completed" and second.plan_ref == "plan_late_recovery"
    assert second_ensure.calls == [{"root": tmp_path, "as_of": AS_OF, "attempt": 1}]
    assert provider.calls == [(AS_OF, ("AAA", "BBB"))]  # probed once, on tick 1, never again


def test_a_pre_plan_not_yet_after_an_error_stays_resumable_past_the_deadline(tmp_path):
    # CodeRabbit finding (round 7), the resume/not_yet gap: a PRE-plan resume
    # whose retried ensure_snapshot_fn call comes back "not_yet" (the legacy
    # store still has not caught up) used to write the plain, NON-resumable
    # "not_yet" status, so the next tick fell through to _decide's window
    # check -- already closed -- and recorded a terminal "missed", discarding
    # the retry the round-5 fix exists to allow. Tick 1 fails terminally
    # (INPUT_CHANGED bumps snapshot_attempt to 1); tick 2, past the deadline,
    # hits "not_yet" on that SAME attempt and must record "snapshot_not_yet";
    # tick 3, still past the deadline, resumes straight through to
    # "completed" -- _decide's window check never re-gates this as_of.
    provider = FakeProvider(True)

    class ExplodingEnsure:
        def __init__(self):
            self.calls = []

        def __call__(self, root, as_of, clock, attempt):
            self.calls.append(attempt)
            raise fail("INPUT_CHANGED", "boom")

    first = _run(tmp_path, FakeClock(IN_WINDOW), provider, FakePlan("plan_should_not_build"),
                 FakeSubmit(), FakeServe(), ensure_snapshot_fn=ExplodingEnsure())
    assert first.status == "error" and first.plan_ref is None
    assert first.snapshot_attempt == 1  # INPUT_CHANGED bumps it

    second_ensure = FakeEnsureSnapshot(readiness="not_yet")
    second = _run(tmp_path, FakeClock(AFTER_DEADLINE), provider, FakePlan(), FakeSubmit(),
                  FakeServe(), ensure_snapshot_fn=second_ensure)
    assert second.status == "snapshot_not_yet"  # NOT "missed" -- still resumable
    assert second_ensure.calls == [{"root": tmp_path, "as_of": AS_OF, "attempt": 1}]
    assert second.snapshot_attempt == 1  # a "not_yet" outcome never bumps it

    plan, submit, serve = (FakePlan("plan_after_snapshot_not_yet"), FakeSubmit(),
                           FakeServe("completed"))
    third_ensure = FakeEnsureSnapshot(readiness="ready", snapshot_id="snap_after_wait")
    third = _run(tmp_path, FakeClock(AFTER_DEADLINE), provider, plan, submit, serve,
                 ensure_snapshot_fn=third_ensure)
    assert third.status == "completed" and third.plan_ref == "plan_after_snapshot_not_yet"
    assert provider.calls == [(AS_OF, ("AAA", "BBB"))]  # probed once, on tick 1, never again


def test_three_consecutive_pre_plan_terminal_failures_become_failed_setup(tmp_path):
    # Mirrors test_three_consecutive_setup_errors_become_failed_setup_once,
    # but for a PRE-plan ensure_snapshot_fn terminal failure (plan_ref=None
    # throughout) instead of a post-plan submit_fn failure -- each retry
    # after the first is reached through run_trigger's (round-5-generalized)
    # resume branch.
    provider = FakeProvider(True)

    class ExplodingEnsure:
        def __init__(self):
            self.calls = []

        def __call__(self, root, as_of, clock, attempt):
            self.calls.append(attempt)
            raise fail("INPUT_CHANGED", "boom")

    ensure = ExplodingEnsure()
    plan = FakePlan("plan_should_not_build")
    statuses, counts = [], []
    for _ in range(3):
        receipt = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, FakeSubmit(),
                       FakeServe(), ensure_snapshot_fn=ensure)
        statuses.append(receipt.status)
        counts.append(receipt.error_count)
    assert statuses == ["error", "error", "failed_setup"]
    assert counts == [1, 2, 3]
    assert provider.calls == [(AS_OF, ("AAA", "BBB"))]  # probed once, on tick 1, never again
    assert plan.calls == []  # the plan was never built -- the snapshot never came ready
    assert ensure.calls == [0, 1, 2]  # each terminal failure mints a genuinely new attempt
    terminal = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, FakeSubmit(), FakeServe(),
                    ensure_snapshot_fn=ensure)
    assert terminal.status == "idle"
    assert len(ensure.calls) == 3  # a terminal tick never re-checks the snapshot


def test_a_pre_plan_resume_preserves_the_original_ticker_selection(tmp_path):
    # Gate-round-6 fix: run_trigger's resume branch used to hardcode empty
    # tickers/context_tickers, discarding whatever the caller explicitly
    # passed. Since a pre-plan resume (plan_ref=None) now actually calls
    # plan_fn for real, that selection must survive the resume.
    explicit_tickers = ("XOM", "CVX")
    explicit_context = ("XOM", "CVX", "SPY")
    provider = FakeProvider(True)
    first_ensure = FakeEnsureSnapshot(readiness="timed_out")
    first = _run(tmp_path, FakeClock(IN_WINDOW), provider, FakePlan("plan_should_not_build"),
                 FakeSubmit(), FakeServe(), tickers=explicit_tickers,
                 context_tickers=explicit_context, ensure_snapshot_fn=first_ensure)
    assert first.status == "timed_out" and first.plan_ref is None

    plan, submit, serve = FakePlan("plan_kept_selection"), FakeSubmit(), FakeServe("completed")
    second_ensure = FakeEnsureSnapshot(readiness="ready", snapshot_id="snap_kept")
    second = _run(tmp_path, FakeClock(AFTER_DEADLINE), provider, plan, submit, serve,
                  tickers=explicit_tickers, context_tickers=explicit_context,
                  ensure_snapshot_fn=second_ensure)
    assert second.status == "completed" and second.plan_ref == "plan_kept_selection"
    assert plan.calls[0]["tickers"] == explicit_tickers
    assert plan.calls[0]["context_tickers"] == explicit_context


def test_three_consecutive_timeouts_become_failed(tmp_path):
    provider, plan = FakeProvider(True), FakePlan("plan_3t")
    submit, serve = FakeSubmit(), FakeServe("timed_out")
    statuses, counts = [], []
    for _ in range(3):
        receipt = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, submit, serve)
        statuses.append(receipt.status)
        counts.append(receipt.error_count)
    assert statuses == ["timed_out", "timed_out", "failed"]
    assert counts == [1, 2, 3]
    assert provider.calls == [(AS_OF, ("AAA", "BBB"))]  # probed once, then resumed
    assert len(plan.calls) == 1 and len(submit.calls) == 3
    terminal = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, submit, serve)
    assert terminal.status == "idle"
    assert len(submit.calls) == 3


def test_default_serve_passes_todays_et_cutoff(tmp_path, monkeypatch):
    from engine.v2.ops import bootstrap, cli, supervisor

    monkeypatch.setattr(bootstrap, "open_catalog", lambda *a, **k: _DummyConn())
    monkeypatch.setattr(cli, "_submit_command",
                        lambda args, root, conn, clock: {"jobs": [{"job_id": "job_1"}]})
    monkeypatch.setattr(supervisor, "Service", lambda *a, **k: object())
    captured = {}

    def fake_serve(service, *, until=None, deadline_at=None):
        captured["deadline_at"] = deadline_at
        return "deadline_exceeded"

    monkeypatch.setattr(nightly_trigger, "serve", fake_serve, raising=False)
    monkeypatch.setattr(supervisor, "serve", fake_serve)
    clock = FakeClock(IN_WINDOW)
    assert nightly_trigger._default_serve(tmp_path, "plan_cutoff", clock) == "timed_out"
    assert captured["deadline_at"] == nightly_trigger._serve_deadline(FakeClock(IN_WINDOW))


def test_serve_deadline_is_the_same_absolute_cutoff_for_a_morning_or_evening_start():
    morning = FakeClock(datetime(2026, 9, 26, 6, 5, tzinfo=ET))
    evening = FakeClock(datetime(2026, 9, 26, 20, 30, tzinfo=ET))
    expected = datetime(2026, 9, 26, 20, 0, tzinfo=ET)
    assert nightly_trigger._serve_deadline(morning) == expected
    assert nightly_trigger._serve_deadline(evening) == expected
    assert nightly_trigger._serve_deadline(morning) == nightly_trigger._serve_deadline(evening)


def test_a_late_evening_resume_times_out_on_its_first_tick_not_after_a_fresh_budget(tmp_path, monkeypatch):
    first_plan = FakePlan("plan_late")
    first = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), first_plan,
                 FakeSubmit(), FakeServe("timed_out"))
    assert first.status == "timed_out" and first.plan_ref == "plan_late"

    # The resumed tick, at 20:30 ET the same day -- after DEFAULT_SERVE_DEADLINE_ET. Drive the
    # REAL _default_serve -> serve() path (same stubbing pattern as
    # test_default_serve_passes_todays_et_cutoff) rather than a FakeServe, so this proves the
    # actual deadline check stops it on the first tick, not just that _serve_deadline's own
    # return value looks right in isolation.
    from engine.v2.ops import bootstrap, cli, supervisor

    late_moment = datetime(2026, 9, 26, 20, 30, tzinfo=ET)
    monkeypatch.setattr(bootstrap, "open_catalog", lambda *a, **k: _DummyConn())
    monkeypatch.setattr(cli, "_submit_command",
                        lambda args, root, conn, clock: {"jobs": [{"job_id": "job_1"}]})
    monkeypatch.setattr(supervisor, "Service", lambda *a, **k: object())
    tick_calls = []

    def real_serve_stub(service, *, until=None, deadline_at=None):
        # Mirrors supervisor.serve's own loop shape closely enough to prove the deadline check
        # fires on the very first iteration for an already-past deadline, without needing a
        # real Service/catalog: no service.tick()/service.start() call here since `service` is
        # the bare stub object() above -- this checks the SAME condition serve() itself checks.
        tick_calls.append(1)
        if deadline_at is not None and late_moment >= deadline_at:
            return "deadline_exceeded"
        raise AssertionError("expected the deadline to have already passed")

    monkeypatch.setattr(nightly_trigger, "serve", real_serve_stub, raising=False)
    monkeypatch.setattr(supervisor, "serve", real_serve_stub)
    late_clock = FakeClock(late_moment)
    result = nightly_trigger._default_serve(tmp_path, "plan_late", late_clock)
    assert result == "timed_out"
    assert len(tick_calls) == 1  # stopped on the first (only) check, no further cycle

    second_plan = FakePlan("plan_OTHER")
    second = _run(tmp_path, late_clock, FakeProvider(True), second_plan,
                  FakeSubmit(), FakeServe("timed_out"))
    assert second.plan_ref == "plan_late"  # resumes, no re-plan
    assert second_plan.calls == []


# --------------------------------------------------------------------------
# 12. default as-of: most recent completed trading session before today ET
# --------------------------------------------------------------------------


@pytest.mark.parametrize("moment, expected", [
    (datetime(2026, 9, 26, 0, 30, tzinfo=ET), "2026-09-25"),   # Saturday -> Friday
    (datetime(2026, 9, 28, 0, 30, tzinfo=ET), "2026-09-25"),   # Monday -> Friday
    (datetime(2026, 11, 1, 0, 30, tzinfo=ET), "2026-10-30"),   # DST end (Sunday)
    (datetime(2026, 3, 8, 0, 30, tzinfo=ET), "2026-03-06"),    # DST start (Sunday)
    (datetime(2026, 11, 26, 0, 30, tzinfo=ET), "2026-11-25"),  # Thanksgiving skipped
])
def test_default_as_of_is_the_previous_trading_session(moment, expected):
    assert default_as_of(FakeClock(moment)) == expected


def test_default_as_of_reads_converted_clocks_in_america_new_york():
    # 04:30 UTC on the fall-back date is 00:30 EDT: still Sunday Nov 1.
    assert default_as_of(FakeClock(datetime(2026, 11, 1, 4, 30, tzinfo=UTC))) == "2026-10-30"
    # 07:30 UTC is 02:30 EST after the transition: still Sunday Nov 1.
    assert default_as_of(FakeClock(datetime(2026, 11, 1, 7, 30, tzinfo=UTC))) == "2026-10-30"
    # 07:30 UTC on the spring-forward date is 03:30 EDT: still Sunday Mar 8.
    assert default_as_of(FakeClock(datetime(2026, 3, 8, 7, 30, tzinfo=UTC))) == "2026-03-06"


# --------------------------------------------------------------------------
# 13. population: the native plan's full default population, never empty
# --------------------------------------------------------------------------


def _write_population(root: Path, keys) -> Path:
    path = root / "reports" / "phase6" / "nightly_trigger" / "expected_population.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(keys)))
    return path


def test_full_population_is_the_operator_document(tmp_path):
    document = _write_population(
        tmp_path, ["AAA|STR-THRU|2026-09-25", "BBB|STR-THRU|2026-09-25",
                   "AAA|EXP-1|2026-09-25"])
    assert full_population(tmp_path) == (("AAA", "BBB"), ("AAA", "BBB"))
    assert document.is_file()
    assert full_population(tmp_path / "absent") == ((), ())


class _DummyConn:
    def close(self):
        return None


def test_default_plan_passes_full_run_and_the_full_population(tmp_path, monkeypatch):
    document = _write_population(
        tmp_path, ["AAA|STR-THRU|2026-09-25", "BBB|STR-THRU|2026-09-25"])
    captured = {}
    from engine.v2.ops import bootstrap, cli

    monkeypatch.setattr(bootstrap, "open_catalog", lambda *a, **k: _DummyConn())
    manifest_path = tmp_path / "captured_manifest.json"
    monkeypatch.setattr(nightly_trigger, "_capture_input_manifest",
                        lambda *a, **k: manifest_path)

    def fake_plan(args, root, conn, clock):
        captured["args"] = args
        return {"plan_ref": "plan_full", "plan": {}}

    monkeypatch.setattr(cli, "_plan_command", fake_plan)
    plan_ref = nightly_trigger._default_plan(
        tmp_path, AS_OF, (), (), None, expected_shadow_snapshot_id="snap_verified")
    assert plan_ref == "plan_full"
    args = captured["args"]
    assert args.full_run is True
    assert args.expected_population == document
    assert args.tickers == "AAA,BBB" and args.context_tickers == "AAA,BBB"
    assert args.input_manifest == manifest_path
    assert args.input_mode == "snapshot" and args.snapshot_scope == "shadow"
    assert args.expected_snapshot_id == "snap_verified"


def test_scheduled_run_declares_the_full_population(tmp_path):
    plan, submit, serve = FakePlan("plan_pop"), FakeSubmit(), FakeServe()
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit, serve,
                   tickers=("AAA", "BBB"))
    assert receipt.status == "completed"
    assert plan.calls[0]["full_run"] is True and plan.calls[0]["tickers"] == ("AAA", "BBB")


# --------------------------------------------------------------------------
# 14. CLI: one JSON receipt line; exit 0 IDLE, 1 on the first failure
# --------------------------------------------------------------------------


def test_main_prints_one_receipt_and_exits_zero_for_not_yet(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(nightly_trigger, "SystemClock", lambda: FakeClock(IN_WINDOW))
    monkeypatch.setattr(nightly_trigger, "_orats_probe",
                        lambda as_of, tickers: (False, "not published"))
    code = nightly_trigger.main(["--as-of", AS_OF, "--root", str(tmp_path)])
    lines = capsys.readouterr().out.strip().splitlines()
    assert code == 0 and len(lines) == 1
    assert json.loads(lines[0])["status"] == "not_yet"


def test_main_exits_one_for_missed_then_zero_for_idle(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(nightly_trigger, "SystemClock", lambda: FakeClock(AFTER_DEADLINE))
    assert nightly_trigger.main(["--as-of", AS_OF, "--root", str(tmp_path)]) == 1
    document = json.loads(capsys.readouterr().out.strip())
    assert document["status"] == "missed"
    assert nightly_trigger.main(["--as-of", AS_OF, "--root", str(tmp_path)]) == 0
    idle = json.loads(capsys.readouterr().out.strip())
    assert idle["status"] == "idle"


def test_main_records_the_submit_refusal_when_qualification_inputs_are_absent(
        tmp_path, monkeypatch, capsys):
    """End-to-end production wiring: an unconfigured scheduled run builds the
    real (blocked) plan in-process and records ``ops submit``'s typed refusal
    as an ``error`` receipt -- it never bypasses the planned-population gate."""
    monkeypatch.setattr(nightly_trigger, "SystemClock", lambda: FakeClock(IN_WINDOW))
    monkeypatch.setattr(nightly_trigger, "_orats_probe",
                        lambda as_of, tickers: (True, "final"))
    # Cutover PR-7b-2: the wiring makes _submit_plan ensure the shadow snapshot
    # BEFORE planning; this end-to-end test is about the plan/submit refusal on
    # absent qualification inputs, not the snapshot import, so stub readiness --
    # the flipped _default_plan literals then surface the typed refusal.
    monkeypatch.setattr(nightly_trigger, "_ensure_shadow_snapshot",
                        lambda root, as_of, clock, attempt: ("ready", "snap_stub"))
    code = nightly_trigger.main(["--as-of", AS_OF, "--root", str(tmp_path)])
    document = json.loads(capsys.readouterr().out.strip())
    assert code == 1 and document["status"] == "error"
    assert "INVALID_REQUEST" in document["detail"]
    stored = load_state(tmp_path, AS_OF)
    assert stored is not None and stored.status == "error"


# --------------------------------------------------------------------------
# 15. paths
# --------------------------------------------------------------------------


def test_lock_and_state_paths_live_under_the_root(tmp_path):
    assert nightly_trigger.legacy_lock_path(tmp_path) == tmp_path / "reports" / ".nightly.lock"
    assert state_path(tmp_path, AS_OF) == (
        tmp_path / "reports" / "phase6" / "nightly_trigger" / f"{AS_OF}.json")


# --------------------------------------------------------------------------
# 16. issue #104: per-as_of input manifest capture and year derivation
# --------------------------------------------------------------------------


def test_derive_years_matches_legacy_context_years_formula():
    # 35 days after 2026-01-15 is 2026-02-19: the horizon stays in 2026.
    assert nightly_trigger._derive_years("2026-01-15") == (2025, 2026)
    # 35 days after 2026-11-30 is 2027-01-04: a late-November as_of pulls in
    # the following year, which the fixed 2024/2026 pair could never express.
    assert nightly_trigger._derive_years("2026-11-30") == (2025, 2027)


def test_capture_input_manifest_writes_a_per_as_of_path_and_returns_it(tmp_path, monkeypatch):
    from engine.v2.ops import capture_inputs

    class FakeManifest:
        def __init__(self, selected_session):
            self.selected_session = selected_session

    def fake_capture(root, *, as_of, tickers, context_tickers, year_start, year_end):
        return FakeManifest(as_of)

    def fake_write_manifest(manifest, output):
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(manifest.selected_session)
        return path

    monkeypatch.setattr(capture_inputs, "write_manifest", fake_write_manifest)
    first = nightly_trigger._capture_input_manifest(
        tmp_path, "2026-01-15", ("AAA",), ("AAA",), 2025, 2026, capture_fn=fake_capture)
    second = nightly_trigger._capture_input_manifest(
        tmp_path, "2026-11-30", ("AAA",), ("AAA",), 2025, 2027, capture_fn=fake_capture)
    target = tmp_path / "reports" / "phase6" / "nightly_trigger"
    assert first == target / "2026-01-15.input_manifest.json"
    assert second == target / "2026-11-30.input_manifest.json"
    assert first.is_file() and second.is_file()
    assert first.read_text() == "2026-01-15"
    assert second.read_text() == "2026-11-30"


def test_capture_input_manifest_refuses_a_selected_session_mismatch(tmp_path, monkeypatch):
    from engine.v2.ops import capture_inputs

    class FakeManifest:
        selected_session = "2026-01-14"

    def fake_capture(root, *, as_of, tickers, context_tickers, year_start, year_end):
        return FakeManifest()

    def explode(manifest, output):
        raise AssertionError("a mismatched manifest must never be written")

    monkeypatch.setattr(capture_inputs, "write_manifest", explode)
    with pytest.raises(OpsError) as raised:
        nightly_trigger._capture_input_manifest(
            tmp_path, "2026-01-15", ("AAA",), ("AAA",), 2025, 2026, capture_fn=fake_capture)
    assert raised.value.code == "INPUT_CHANGED"
    assert not (tmp_path / "reports" / "phase6" / "nightly_trigger"
                / "2026-01-15.input_manifest.json").exists()


def test_default_plan_has_no_input_manifest_when_there_is_no_population(tmp_path, monkeypatch):
    captured = {}
    from engine.v2.ops import bootstrap, cli

    monkeypatch.setattr(bootstrap, "open_catalog", lambda *a, **k: _DummyConn())

    def should_not_be_called(*args, **kwargs):
        raise AssertionError("should not be called")

    monkeypatch.setattr(nightly_trigger, "_capture_input_manifest", should_not_be_called)

    def fake_plan(args, root, conn, clock):
        captured["args"] = args
        return {"plan_ref": "plan_no_pop", "plan": {}}

    monkeypatch.setattr(cli, "_plan_command", fake_plan)
    assert nightly_trigger._default_plan(tmp_path, AS_OF, (), (), None) == "plan_no_pop"
    args = captured["args"]
    assert args.input_manifest is None
    assert (args.year_start, args.year_end) == nightly_trigger._derive_years(AS_OF)
    assert args.input_mode == "snapshot" and args.snapshot_scope == "shadow"
    assert args.expected_snapshot_id is None  # no ensure_snapshot_fn call in this path


# --------------------------------------------------------------------------
# 17. Cutover PR-7b: _ensure_shadow_snapshot (added unused; not yet wired
# into _submit_plan -- these tests exercise it directly)
# --------------------------------------------------------------------------


class _FakeManifest:
    def __init__(self, selected_session):
        self.selected_session = selected_session


class _FakePlan:
    def __init__(self, selected_session):
        self.legacy_input_manifest = _FakeManifest(selected_session)


def _open_ops_catalog(root, clock):
    from engine.v2.ops.bootstrap import open_catalog

    ops_root = nightly_trigger._ops_root(root)
    ops_root.mkdir(parents=True, exist_ok=True)
    return open_catalog(ops_root / "catalog.sqlite", clock=clock)


def _seed_snapshot_import_job(conn, clock, *, as_of, attempt, state):
    """A minimal, real ``snapshot_import`` job row under the exact key
    ``_ensure_shadow_snapshot`` would look up -- the same "submit through
    real submission machinery, then set state directly" shape PR-7a's own
    ``_mark_score_succeeded`` fixture uses, simplified to a raw INSERT since
    this job kind takes no dependencies. Returns the row's job_id.
    """
    from engine.v2.ops.submission import job_id_for

    key = f"shadow_snapshot_import:{as_of}:{attempt}"
    job_id = job_id_for("shadow", key)
    stamp = clock.now().isoformat()
    conn.execute(
        "INSERT INTO jobs (job_id, namespace, idempotency_key, request_digest, principal, "
        "kind, spec_json, resource_class, checkpoint_contract_ref, retry_json, state, "
        "priority, max_attempts, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (job_id, "shadow", key, "digest", "operator", "snapshot_import", "{}",
         "legacy_rebuild", "snapshot_import_inspections.v1.0", "{}", state, 0, 1, stamp, stamp))
    conn.commit()
    return job_id


def _explode(*args, **kwargs):
    raise AssertionError("must not be called on this branch")


def test_ensure_shadow_snapshot_succeeded_under_current_attempt_is_a_noop(tmp_path, monkeypatch):
    """R2/R6, and the cache-hit branch reading resulting_head_snapshot_id off
    an already-succeeded job rather than re-resolving the mutable head: a job
    already ``succeeded`` under this exact ``attempt`` key is a pure cache
    hit -- plan_import_fn/submit_import_fn/serve_fn are never called, and the
    returned snapshot_id comes off the job's own committed receipt via
    ``_resulting_head_snapshot_id``, not a fresh ``data_snapshot_heads``
    read (this test never seeds that table at all, so any such read would
    surface as None, not the value asserted below).
    """
    clock = FakeClock(IN_WINDOW)
    conn = _open_ops_catalog(tmp_path, clock)
    _seed_snapshot_import_job(conn, clock, as_of=AS_OF, attempt=2, state="succeeded")
    conn.close()
    monkeypatch.setattr(nightly_trigger, "_resulting_head_snapshot_id",
                        lambda root, conn, job_id: "snap_from_receipt")

    result = nightly_trigger._ensure_shadow_snapshot(
        tmp_path, AS_OF, clock, 2, plan_import_fn=_explode, submit_import_fn=_explode,
        serve_fn=_explode)

    assert result == ("ready", "snap_from_receipt")


def test_ensure_shadow_snapshot_terminal_failed_job_raises_without_resubmitting(tmp_path):
    """R1(b)/R3: a job that is terminal but not succeeded under this exact
    attempt key raises INPUT_CHANGED immediately -- it is never resubmitted
    under the same key."""
    clock = FakeClock(IN_WINDOW)
    conn = _open_ops_catalog(tmp_path, clock)
    _seed_snapshot_import_job(conn, clock, as_of=AS_OF, attempt=0, state="failed")
    conn.close()

    with pytest.raises(OpsError) as raised:
        nightly_trigger._ensure_shadow_snapshot(
            tmp_path, AS_OF, clock, 0, plan_import_fn=_explode, submit_import_fn=_explode,
            serve_fn=_explode)
    assert raised.value.code == "INPUT_CHANGED"


def test_ensure_shadow_snapshot_session_mismatch_returns_not_yet(tmp_path):
    """R1(a): the legacy store has not caught up to as_of yet -- returned as
    not_yet, no submission attempted, no attempt consumed."""
    clock = FakeClock(IN_WINDOW)

    def fake_plan_import(root, expected_head_snapshot_id, expected_head_generation):
        assert expected_head_snapshot_id is None and expected_head_generation == 0
        return _FakePlan("2026-09-24")  # a different session than AS_OF

    result = nightly_trigger._ensure_shadow_snapshot(
        tmp_path, AS_OF, clock, 0, plan_import_fn=fake_plan_import,
        submit_import_fn=_explode, serve_fn=_explode)

    assert result == ("not_yet", None)


def test_ensure_shadow_snapshot_clean_round_trip_returns_ready(tmp_path):
    """A clean plan-import -> submit -> serve round trip: no existing row,
    a matching session, a fresh submission, and a fake serve_fn returning
    ("ready", snapshot_id) directly."""
    clock = FakeClock(IN_WINDOW)
    calls = {}

    def fake_plan_import(root, expected_head_snapshot_id, expected_head_generation):
        return _FakePlan(AS_OF)

    def fake_submit_import(root, conn, plan, idempotency_key, clock_arg):
        calls["submit_key"] = idempotency_key
        return "job_fresh"

    def fake_serve(root, conn, job_id, clock_arg, deadline_at):
        calls["serve"] = (job_id, deadline_at)
        return "ready", "snap_committed"

    result = nightly_trigger._ensure_shadow_snapshot(
        tmp_path, AS_OF, clock, 3, plan_import_fn=fake_plan_import,
        submit_import_fn=fake_submit_import, serve_fn=fake_serve)

    assert result == ("ready", "snap_committed")
    assert calls["submit_key"] == f"shadow_snapshot_import:{AS_OF}:3"
    assert calls["serve"] == ("job_fresh", nightly_trigger._serve_deadline(clock))


def test_ensure_shadow_snapshot_fresh_attempt_after_terminal_failure_retries(tmp_path):
    """R3/R6: a terminal failure under attempt 0 must never be retried under
    that SAME key (a bare as_of-only key would keep matching the dead row
    forever); a fresh attempt=1 mints a genuinely new key and succeeds."""
    clock = FakeClock(IN_WINDOW)
    conn = _open_ops_catalog(tmp_path, clock)
    _seed_snapshot_import_job(conn, clock, as_of=AS_OF, attempt=0, state="failed")
    conn.close()

    with pytest.raises(OpsError):
        nightly_trigger._ensure_shadow_snapshot(
            tmp_path, AS_OF, clock, 0, plan_import_fn=_explode, submit_import_fn=_explode,
            serve_fn=_explode)

    def fake_plan_import(root, expected_head_snapshot_id, expected_head_generation):
        return _FakePlan(AS_OF)

    def fake_submit_import(root, conn, plan, idempotency_key, clock_arg):
        assert idempotency_key == f"shadow_snapshot_import:{AS_OF}:1"
        return "job_retry"

    def fake_serve(root, conn, job_id, clock_arg, deadline_at):
        return "ready", "snap_retry"

    result = nightly_trigger._ensure_shadow_snapshot(
        tmp_path, AS_OF, clock, 1, plan_import_fn=fake_plan_import,
        submit_import_fn=fake_submit_import, serve_fn=fake_serve)

    assert result == ("ready", "snap_retry")


def test_ensure_shadow_snapshot_reattaches_to_a_non_terminal_job_without_resubmitting(tmp_path):
    """A non-terminal existing row under the current attempt key (queued,
    running, retry_wait, ...) is reattached to and waited on directly --
    plan_import_fn/submit_import_fn are never called a second time for it."""
    clock = FakeClock(IN_WINDOW)
    conn = _open_ops_catalog(tmp_path, clock)
    job_id = _seed_snapshot_import_job(conn, clock, as_of=AS_OF, attempt=0, state="queued")
    conn.close()

    def fake_serve(root, conn, seen_job_id, clock_arg, deadline_at):
        assert seen_job_id == job_id
        return "timed_out", None

    result = nightly_trigger._ensure_shadow_snapshot(
        tmp_path, AS_OF, clock, 0, plan_import_fn=_explode, submit_import_fn=_explode,
        serve_fn=fake_serve)

    assert result == ("timed_out", None)


def test_default_serve_snapshot_import_maps_deadline_exceeded_to_timed_out(tmp_path, monkeypatch):
    """The production serve_fn default: a "deadline_exceeded" drive-to-
    terminal outcome maps to ("timed_out", None); the still-running job is
    left exactly as it is (this default issues no UPDATE/DELETE against
    `jobs` -- _drive_jobs_to_terminal and get_job only ever read it)."""
    from engine.v2.ops import supervisor

    monkeypatch.setattr(supervisor, "Service", lambda *a, **k: object())
    captured = {}

    def fake_serve(service, *, until=None, deadline_at=None):
        captured["deadline_at"] = deadline_at
        return "deadline_exceeded"

    monkeypatch.setattr(supervisor, "serve", fake_serve)
    clock = FakeClock(IN_WINDOW)
    conn = _open_ops_catalog(tmp_path, clock)
    job_id = _seed_snapshot_import_job(conn, clock, as_of=AS_OF, attempt=0, state="queued")
    deadline_at = nightly_trigger._serve_deadline(clock)

    result = nightly_trigger._default_serve_snapshot_import(tmp_path, conn, job_id, clock,
                                                            deadline_at)
    state = conn.execute("SELECT state FROM jobs WHERE job_id = ?", (job_id,)).fetchone()["state"]
    conn.close()

    assert result == ("timed_out", None)
    assert captured["deadline_at"] == deadline_at
    assert state == "queued"


def _seed_snapshot_import_receipt_output(conn, clock, root, job_id, *, name, document):
    """A real attempt + a real, content-addressed artifact registered as
    that attempt's named output -- the shape a succeeded snapshot_import
    job's coordinator effect actually produces
    (``snapshot_promotion.snapshot_import_effect``), built through the real
    ``ArtifactStore``/``register_artifact`` calls rather than a hand-invented
    artifacts row. Reuses the job's existing attempt when there is one, so
    several named outputs land on the SAME attempt -- the real publishing
    shape, and the only shape that exercises name-vs-index selection.
    """
    import json as _json

    from engine.v2.foundation import ArtifactStore
    from engine.v2.ops.catalog import transaction
    from engine.v2.ops.checkpoints import register_artifact
    from engine.v2.ops.recovery import begin_epoch

    store = ArtifactStore(nightly_trigger._ops_root(root))
    ref = store.publish_bytes(_json.dumps(document).encode("utf-8"), schema_ref="test.v1.0")
    existing = conn.execute("SELECT attempt_id FROM attempts WHERE job_id = ?",
                            (job_id,)).fetchone()
    if existing is None:
        attempt_id = f"att_{job_id}_1"
        epoch_id = begin_epoch(conn, clock=clock, boot_id="test", pid=1)
        stamp = clock.now().isoformat()
        conn.execute(
            "INSERT INTO attempts (attempt_id, job_id, attempt_number, fence, supervisor_epoch, "
            "host_boot_id, state, process_state, resources_json, created_at, lease_expires_at) "
            "VALUES (?, ?, 1, 1, ?, 'test', 'succeeded', 'exited', '{}', ?, ?)",
            (attempt_id, job_id, epoch_id, stamp, stamp))
    else:
        attempt_id = existing["attempt_id"]
    with transaction(conn):
        register_artifact(conn, ref, attempt_id, clock)
    conn.execute("INSERT INTO attempt_outputs (attempt_id, name, artifact_id) VALUES (?, ?, ?)",
                (attempt_id, name, ref.artifact_id))
    conn.commit()


def test_resulting_head_snapshot_id_selects_the_named_receipt_over_other_outputs(tmp_path):
    """CodeRabbit round 1 (real): get_job().output_refs is ordered by
    artifact_id, not by output name, so a naive output_refs[0] pick can
    return the wrong artifact when a succeeded attempt published more than
    one named output. CodeRabbit round 2 (real): a hardcoded pair of
    document values happened to make the RECEIPT's own artifact_id sort
    first anyway, so the old, buggy implementation passed this test by
    coincidence. This version publishes both candidate documents first,
    determines which one's artifact_id actually sorts first -- the one an
    output_refs[0] pick would return -- and deliberately assigns THAT value
    to the non-receipt-named output, so the test fails against the old
    implementation regardless of what the two content hashes happen to be,
    and passes only when the receipt is genuinely selected by name."""
    import json as _json

    from engine.v2.foundation import ArtifactStore

    clock = FakeClock(IN_WINDOW)
    conn = _open_ops_catalog(tmp_path, clock)
    job_id = _seed_snapshot_import_job(conn, clock, as_of=AS_OF, attempt=0, state="succeeded")

    store = ArtifactStore(nightly_trigger._ops_root(tmp_path))
    id_a = store.publish_bytes(_json.dumps({"resulting_head_snapshot_id": "A"}).encode(),
                               schema_ref="test.v1.0").artifact_id
    id_b = store.publish_bytes(_json.dumps({"resulting_head_snapshot_id": "B"}).encode(),
                               schema_ref="test.v1.0").artifact_id
    # Whichever of "A"/"B" has the artifact_id that sorts FIRST is the value
    # an output_refs[0] pick (ORDER BY artifact_id) would return -- give
    # that value to the non-receipt-named output, and the OTHER value to
    # the real receipt-named output.
    first_value, second_value = ("A", "B") if id_a < id_b else ("B", "A")
    _seed_snapshot_import_receipt_output(conn, clock, tmp_path, job_id, name="snapshot_import",
                                        document={"resulting_head_snapshot_id": first_value})
    _seed_snapshot_import_receipt_output(conn, clock, tmp_path, job_id,
                                        name="snapshot_import_receipt",
                                        document={"resulting_head_snapshot_id": second_value})

    result = nightly_trigger._resulting_head_snapshot_id(tmp_path, conn, job_id)
    conn.close()

    assert result == second_value


def test_resulting_head_snapshot_id_raises_when_the_receipt_has_no_snapshot_id(tmp_path):
    """A receipt document present but missing (or carrying an empty)
    resulting_head_snapshot_id must raise INPUT_CHANGED, never return None
    silently."""
    clock = FakeClock(IN_WINDOW)
    conn = _open_ops_catalog(tmp_path, clock)
    job_id = _seed_snapshot_import_job(conn, clock, as_of=AS_OF, attempt=0, state="succeeded")
    _seed_snapshot_import_receipt_output(conn, clock, tmp_path, job_id,
                                        name="snapshot_import_receipt",
                                        document={"status": "committed"})

    with pytest.raises(OpsError) as raised:
        nightly_trigger._resulting_head_snapshot_id(tmp_path, conn, job_id)
    conn.close()

    assert raised.value.code == "INPUT_CHANGED"


def test_resulting_head_snapshot_id_raises_when_there_is_no_receipt_artifact_at_all(tmp_path):
    """A succeeded job with no attempt_outputs row at all (no receipt ever
    published) must raise INPUT_CHANGED rather than crash or return None."""
    clock = FakeClock(IN_WINDOW)
    conn = _open_ops_catalog(tmp_path, clock)
    job_id = _seed_snapshot_import_job(conn, clock, as_of=AS_OF, attempt=0, state="succeeded")

    with pytest.raises(OpsError) as raised:
        nightly_trigger._resulting_head_snapshot_id(tmp_path, conn, job_id)
    conn.close()

    assert raised.value.code == "INPUT_CHANGED"


# --------------------------------------------------------------------------
# 18. Cutover PR-7b: wiring ensure_snapshot_fn into _submit_plan
# --------------------------------------------------------------------------


def _call_submit_plan(root, clock, *, ensure_snapshot_fn, plan=None, submit=None, serve=None,
                      prior=None, plan_ref=None):
    # _submit_plan is normally reached only through run_trigger, which holds the
    # legacy lock for the whole run; FakeSubmit/FakeServe assert exactly that, so
    # this direct-entry helper takes the same lock around the call.
    lock_path = Path(root) / "reports" / ".nightly.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    holder = lock_path.open("a+")
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return nightly_trigger._submit_plan(
            root, AS_OF, tickers=("AAA", "BBB"), context_tickers=("AAA", "BBB"), clock=clock,
            plan_fn=plan or FakePlan(), submit_fn=submit or FakeSubmit(),
            serve_fn=serve or FakeServe(), ensure_snapshot_fn=ensure_snapshot_fn,
            full_run=True, prior=prior, plan_ref=plan_ref)
    finally:
        holder.close()


def test_submit_plan_not_yet_readiness_records_not_yet_never_submitting(tmp_path):
    plan, submit, serve = FakePlan(), FakeSubmit(), FakeServe()
    ensure_snapshot = FakeEnsureSnapshot(readiness="not_yet")
    receipt = _call_submit_plan(tmp_path, FakeClock(IN_WINDOW), ensure_snapshot_fn=ensure_snapshot,
                                plan=plan, submit=submit, serve=serve)
    assert receipt.status == "not_yet"
    assert plan.calls == [] and submit.calls == [] and serve.calls == []
    stored = load_state(tmp_path, AS_OF)
    assert stored is not None and stored.status == "not_yet"  # never recorded as "submitting"


def test_submit_plan_timed_out_readiness_returns_timed_out_then_failed_after_three(tmp_path):
    plan, submit, serve = FakePlan(), FakeSubmit(), FakeServe()
    ensure_snapshot = FakeEnsureSnapshot(readiness="timed_out")
    prior = None
    for expected_count in (1, 2):
        receipt = _call_submit_plan(tmp_path, FakeClock(IN_WINDOW),
                                    ensure_snapshot_fn=ensure_snapshot, plan=plan, submit=submit,
                                    serve=serve, prior=prior)
        assert receipt.status == "timed_out" and receipt.error_count == expected_count
        assert receipt.snapshot_attempt == 0  # a pre-plan timeout never bumps snapshot_attempt
        prior = receipt
    third = _call_submit_plan(tmp_path, FakeClock(IN_WINDOW), ensure_snapshot_fn=ensure_snapshot,
                              plan=plan, submit=submit, serve=serve, prior=prior)
    assert third.status == "failed" and third.error_count == 3
    assert plan.calls == [] and submit.calls == [] and serve.calls == []


def test_submit_plan_ready_readiness_calls_plan_fn_with_the_verified_snapshot_id(tmp_path):
    plan, submit, serve = FakePlan("plan_ready"), FakeSubmit(), FakeServe("completed")
    ensure_snapshot = FakeEnsureSnapshot(readiness="ready", snapshot_id="snap_xyz")
    receipt = _call_submit_plan(tmp_path, FakeClock(IN_WINDOW), ensure_snapshot_fn=ensure_snapshot,
                                plan=plan, submit=submit, serve=serve)
    assert receipt.status == "completed" and receipt.plan_ref == "plan_ready"
    assert plan.calls[0]["expected_shadow_snapshot_id"] == "snap_xyz"


def test_ensure_snapshot_fn_terminal_failure_bumps_snapshot_attempt_and_survives_a_timeout(
        tmp_path):
    """Opus-gate regression, e900074: a fail-then-timeout-then-retry sequence
    must not reset snapshot_attempt back to 0 on the timeout tick."""
    plan, submit, serve = FakePlan(), FakeSubmit(), FakeServe()

    class ExplodingThenTimingOut:
        def __init__(self):
            self.calls = []

        def __call__(self, root, as_of, clock, attempt):
            self.calls.append(attempt)
            if len(self.calls) in (1, 3):
                raise fail("INPUT_CHANGED", "boom")
            return "timed_out", None

    ensure_snapshot = ExplodingThenTimingOut()
    first = _call_submit_plan(tmp_path, FakeClock(IN_WINDOW), ensure_snapshot_fn=ensure_snapshot,
                              plan=plan, submit=submit, serve=serve)
    assert first.status == "error" and first.snapshot_attempt == 1
    second = _call_submit_plan(tmp_path, FakeClock(IN_WINDOW), ensure_snapshot_fn=ensure_snapshot,
                              plan=plan, submit=submit, serve=serve, prior=first)
    assert second.status == "timed_out" and second.snapshot_attempt == 1
    third = _call_submit_plan(tmp_path, FakeClock(IN_WINDOW), ensure_snapshot_fn=ensure_snapshot,
                              plan=plan, submit=submit, serve=serve, prior=second)
    assert ensure_snapshot.calls == [0, 1, 1]  # NOT reset to 0 by the interleaved timeout
    assert third.status == "error" and third.snapshot_attempt == 2


def test_ensure_snapshot_fn_transient_os_error_does_not_bump_snapshot_attempt(tmp_path):
    """CodeRabbit round 1: only a terminal INPUT_CHANGED refusal mints a new
    idempotency key; a transient OSError must not, since the underlying
    import job (if any) may still be live under the OLD attempt's key."""
    plan, submit, serve = FakePlan(), FakeSubmit(), FakeServe()

    def transient_os_error(root, as_of, clock, attempt):
        raise OSError("transient catalog hiccup")

    first = _call_submit_plan(tmp_path, FakeClock(IN_WINDOW),
                              ensure_snapshot_fn=transient_os_error, plan=plan, submit=submit,
                              serve=serve)
    assert first.status == "error" and first.snapshot_attempt == 0

    seen_attempts = []

    def records_attempt(root, as_of, clock, attempt):
        seen_attempts.append(attempt)
        raise OSError("still transient")

    second = _call_submit_plan(tmp_path, FakeClock(IN_WINDOW),
                               ensure_snapshot_fn=records_attempt, plan=plan, submit=submit,
                               serve=serve, prior=first)
    assert seen_attempts == [0]  # the SAME attempt, not bumped by the prior OSError
    assert second.status == "error" and second.snapshot_attempt == 0


def test_busy_legacy_between_submit_plan_entries_leaves_snapshot_attempt_unchanged(tmp_path):
    """Opus-gate regression, fb7d31e's finding 1 on e900074: a busy_legacy tick
    sandwiched between two _submit_plan entries must not touch snapshot_attempt."""

    def always_explodes(root, as_of, clock, attempt):
        raise fail("INPUT_CHANGED", "boom")

    plan, submit, serve = FakePlan(), FakeSubmit(), FakeServe()
    first = _call_submit_plan(tmp_path, FakeClock(IN_WINDOW), ensure_snapshot_fn=always_explodes,
                              plan=plan, submit=submit, serve=serve)
    assert first.status == "error" and first.snapshot_attempt == 1

    lock = tmp_path / "reports" / ".nightly.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    holder = lock.open("a+")
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        busy = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), plan, submit, serve,
                    ensure_snapshot_fn=FakeEnsureSnapshot())
    finally:
        holder.close()
    assert busy.status == "busy_legacy" and busy.snapshot_attempt == 1

    second = _call_submit_plan(tmp_path, FakeClock(IN_WINDOW), ensure_snapshot_fn=always_explodes,
                               plan=plan, submit=submit, serve=serve, prior=first)
    assert second.status == "error" and second.snapshot_attempt == 2


def test_alternating_terminal_failure_and_timeout_still_reaches_failed_setup_after_three(
        tmp_path):
    """Opus-gate regression, fb7d31e: an alternating terminal-failure/timed_out
    sequence for the SAME as_of must still reach failed_setup after exactly
    MAX_CONSECUTIVE_ERRORS terminal failures, however many timed_out ticks are
    interleaved between them -- the OLD error_count-only check resets on every
    non-"error" status and would never give up on its own."""
    plan, submit, serve = FakePlan(), FakeSubmit(), FakeServe()

    class AlternatingFailThenTimeout:
        def __init__(self):
            self.attempt_seen = []

        def __call__(self, root, as_of, clock, attempt):
            self.attempt_seen.append(attempt)
            raise fail("INPUT_CHANGED", "boom")

    ensure_snapshot = AlternatingFailThenTimeout()
    prior = None
    for _ in range(2):
        failed = _call_submit_plan(tmp_path, FakeClock(IN_WINDOW),
                                   ensure_snapshot_fn=ensure_snapshot, plan=plan, submit=submit,
                                   serve=serve, prior=prior)
        assert failed.status == "error"
        timed_out_ensure = FakeEnsureSnapshot(readiness="timed_out")
        prior = _call_submit_plan(tmp_path, FakeClock(IN_WINDOW),
                                  ensure_snapshot_fn=timed_out_ensure, plan=plan, submit=submit,
                                  serve=serve, prior=failed)
        assert prior.status == "timed_out" and prior.snapshot_attempt == failed.snapshot_attempt
    third = _call_submit_plan(tmp_path, FakeClock(IN_WINDOW), ensure_snapshot_fn=ensure_snapshot,
                              plan=plan, submit=submit, serve=serve, prior=prior)
    assert third.status == "failed_setup"
    assert "3 consecutive times" in third.detail  # reports snapshot_attempt (3), not error_count
    assert third.snapshot_attempt == 3
