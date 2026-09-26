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
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.ops import nightly_trigger  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402
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

    def __call__(self, root, as_of, tickers, context_tickers, clock, *, full_run=True):
        self.calls.append({"root": Path(root), "as_of": as_of, "tickers": tuple(tickers),
                           "context_tickers": tuple(context_tickers), "full_run": full_run})
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
         tickers=("AAA", "BBB"), context_tickers=None, **overrides):
    kwargs = dict(tickers=tickers,
                  context_tickers=tickers if context_tickers is None else context_tickers,
                  deadline_et="06:00", window_start_et="00:00",
                  provider=provider, clock=clock, plan_fn=plan, submit_fn=submit,
                  serve_fn=serve)
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


def test_submitted_state_resumes_serving_without_replanning(tmp_path):
    write_state(tmp_path, TriggerReceipt(as_of=AS_OF, status="submitted", detail="crash",
                                         checked_at="2026-09-26T06:00:00Z", plan_ref="plan_X"))
    provider, plan, submit, serve = FakeProvider(True), FakePlan(), FakeSubmit(), FakeServe()
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), provider, plan, submit, serve)
    assert receipt.status == "completed" and receipt.plan_ref == "plan_X"
    assert provider.calls == [] and plan.calls == []
    assert submit.calls == [(AS_OF, "plan_X")]  # idempotent no-op resubmission
    assert serve.calls == [("plan_X", tmp_path)]


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

    def fake_plan(args, root, conn, clock):
        captured["args"] = args
        return {"plan_ref": "plan_full", "plan": {}}

    monkeypatch.setattr(cli, "_plan_command", fake_plan)
    plan_ref = nightly_trigger._default_plan(tmp_path, AS_OF, (), (), None)
    assert plan_ref == "plan_full"
    args = captured["args"]
    assert args.full_run is True
    assert args.expected_population == document
    assert args.tickers == "AAA,BBB" and args.context_tickers == "AAA,BBB"


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
