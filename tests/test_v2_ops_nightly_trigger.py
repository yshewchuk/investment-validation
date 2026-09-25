"""Required-case proof for ``engine.v2.ops.nightly_trigger`` (Phase 6 slice 12).

Every branch is driven with a fake clock, a fake provider callable and a fake
submit callable -- no network, no catalog, no real systemd timer. The three
negative controls the slice brief names are explicit assertions, not status
strings: a not-final probe never submits, a past-deadline run never submits,
and a rerun of a decided date never submits again. BUSY_LEGACY is proven with
a REAL ``fcntl.flock`` held by the test against the legacy lock file.
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
from engine.v2.ops.nightly_trigger import (  # noqa: E402
    load_state,
    probe_finality,
    run_trigger,
    state_path,
)

ET = ZoneInfo("America/New_York")
AS_OF = "2026-09-25"
IN_WINDOW = datetime(2026, 9, 25, 2, 0, tzinfo=ET)
BEFORE_WINDOW = datetime(2026, 9, 25, 1, 0, tzinfo=ET)
AFTER_DEADLINE = datetime(2026, 9, 25, 6, 30, tzinfo=ET)


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


class FakeSubmit:
    """A submit callable: records calls, returns a fixed plan_ref."""

    def __init__(self, plan_ref="plan_ref_1"):
        self.plan_ref = plan_ref
        self.call_count = 0
        self.calls = []

    def __call__(self, root, as_of, tickers, context_tickers, clock):
        self.call_count += 1
        self.calls.append({"root": root, "as_of": as_of, "tickers": tuple(tickers),
                           "context_tickers": tuple(context_tickers), "clock": clock})
        return self.plan_ref


def _run(root, clock, provider, submit, *, as_of=AS_OF, **overrides):
    kwargs = dict(tickers=("AAA", "BBB"), context_tickers=("AAA", "BBB"),
                  deadline_et="06:00", window_start_et="00:00",
                  provider=provider, clock=clock, submit_fn=submit)
    kwargs.update(overrides)
    return run_trigger(root, as_of, **kwargs)


@pytest.fixture(autouse=True)
def _isolated_legacy_lock(tmp_path, monkeypatch):
    """Keep the real repo's ``reports/.nightly.lock`` out of every test.

    Without this, each in-window run would flock the worktree's own lock file,
    and two tests running in parallel could momentarily see each other as a
    legacy nightly -- a flaky BUSY_LEGACY in the middle of an unrelated case.
    """
    monkeypatch.setattr(nightly_trigger, "repo_root", lambda: tmp_path)


# --------------------------------------------------------------------------
# 1. before the window
# --------------------------------------------------------------------------


def test_before_window_is_not_yet_without_probing_or_submitting(tmp_path):
    provider, submit = FakeProvider(True), FakeSubmit()
    # The default window starts at 00:00, so "before" is exercised with a
    # one-hour window: 01:00 is before a 02:00 start.
    receipt = _run(tmp_path, FakeClock(BEFORE_WINDOW), provider, submit,
                   window_start_et="02:00")
    assert receipt.status == "not_yet"
    assert provider.calls == []
    assert submit.call_count == 0
    assert not state_path(tmp_path, AS_OF).exists()


# --------------------------------------------------------------------------
# 2 + 4. in window, not final: state recorded, NEVER a submit (negative control)
# --------------------------------------------------------------------------


def test_in_window_not_final_is_not_yet_and_never_submits(tmp_path):
    provider, submit = FakeProvider(False), FakeSubmit()
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), provider, submit)
    assert receipt.status == "not_yet" and receipt.detail == "not final"
    assert provider.calls == [(AS_OF, ("AAA", "BBB"))]
    assert submit.call_count == 0  # the negative control, not just the status
    stored = load_state(tmp_path, AS_OF)
    assert stored is not None and stored.status == "not_yet"


def test_not_final_probe_leaves_submit_untouched_as_a_shared_fake(tmp_path):
    submit = FakeSubmit()
    _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(False), submit)
    _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(False), submit)
    assert submit.call_count == 0


# --------------------------------------------------------------------------
# 3. final, in window: one submit, plan_ref recorded, state written
# --------------------------------------------------------------------------


def test_final_in_window_submits_exactly_once_and_records_plan_ref(tmp_path):
    submit = FakeSubmit("plan_ref_9")
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), submit)
    assert receipt.status == "submitted" and receipt.plan_ref == "plan_ref_9"
    assert submit.call_count == 1
    assert submit.calls[0]["as_of"] == AS_OF
    assert submit.calls[0]["tickers"] == ("AAA", "BBB")
    assert submit.calls[0]["context_tickers"] == ("AAA", "BBB")
    assert load_state(tmp_path, AS_OF) == receipt
    assert state_path(tmp_path, AS_OF).is_file()
    assert not state_path(tmp_path, AS_OF).with_name(AS_OF + ".json.tmp").exists()


# --------------------------------------------------------------------------
# 5. past deadline: MISSED for both provider verdicts, one state write
# --------------------------------------------------------------------------


@pytest.mark.parametrize("is_final", [True, False])
def test_after_deadline_is_missed_and_never_submits(tmp_path, is_final):
    provider, submit = FakeProvider(is_final), FakeSubmit()
    receipt = _run(tmp_path, FakeClock(AFTER_DEADLINE), provider, submit)
    assert receipt.status == "missed"
    assert provider.calls == []  # the deadline gates before any probe
    assert submit.call_count == 0
    before = state_path(tmp_path, AS_OF).read_text()
    again = _run(tmp_path, FakeClock(AFTER_DEADLINE), provider, submit)
    assert again == receipt
    assert state_path(tmp_path, AS_OF).read_text() == before
    assert submit.call_count == 0


# --------------------------------------------------------------------------
# 6. rerun of a submitted date: stored receipt, no second submit
# --------------------------------------------------------------------------


def test_rerun_after_submit_never_submits_again(tmp_path):
    provider, submit = FakeProvider(True), FakeSubmit("plan_ref_7")
    first = _run(tmp_path, FakeClock(IN_WINDOW), provider, submit)
    assert first.status == "submitted" and submit.call_count == 1
    # Even a later clock (past the deadline) may not resurrect the date.
    second = _run(tmp_path, FakeClock(AFTER_DEADLINE), provider, submit)
    assert second == first and second.plan_ref == "plan_ref_7"
    assert submit.call_count == 1


# --------------------------------------------------------------------------
# 7. rerun of a missed date: terminal, no re-probe, no re-submit
# --------------------------------------------------------------------------


def test_missed_state_is_terminal_and_never_reprobes(tmp_path):
    provider, submit = FakeProvider(True), FakeSubmit()
    first = _run(tmp_path, FakeClock(AFTER_DEADLINE), provider, submit)
    assert first.status == "missed"
    probes_before = len(provider.calls)
    second = _run(tmp_path, FakeClock(IN_WINDOW), provider, submit)
    assert second == first and second.status == "missed"
    assert len(provider.calls) == probes_before
    assert submit.call_count == 0


# --------------------------------------------------------------------------
# 8. corrupt / missing / mismatched state is no prior state
# --------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [
    "{not json",
    json.dumps(["submitted"]),
    json.dumps({"as_of": "2026-01-01", "status": "submitted", "detail": "", "checked_at": ""}),
    json.dumps({"as_of": AS_OF, "status": "not-a-status"}),
])
def test_corrupt_state_falls_through_to_the_normal_decision(tmp_path, payload):
    path = state_path(tmp_path, AS_OF)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload)
    submit = FakeSubmit("plan_ref_fresh")
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), submit)
    assert receipt.status == "submitted" and receipt.plan_ref == "plan_ref_fresh"
    assert submit.call_count == 1


def test_missing_state_falls_through_to_the_normal_decision(tmp_path):
    submit = FakeSubmit()
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), FakeProvider(True), submit)
    assert receipt.status == "submitted" and submit.call_count == 1


# --------------------------------------------------------------------------
# 9. probe_finality on its own, both branches
# --------------------------------------------------------------------------


def test_probe_finality_delegates_both_branches_to_the_provider():
    final = FakeProvider(True, "market-wide published")
    assert probe_finality(AS_OF, ["AAA"], provider=final) == (True, "market-wide published")
    assert final.calls == [(AS_OF, ("AAA",))]
    pending = FakeProvider(False, "not published")
    assert probe_finality(AS_OF, ("AAA",), provider=pending) == (False, "not published")


# --------------------------------------------------------------------------
# BUSY_LEGACY: a real flock, held by this test, on the legacy lock file
# --------------------------------------------------------------------------


def test_busy_legacy_with_a_real_flock_records_retry_and_releases_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly_trigger, "repo_root", lambda: tmp_path)
    lock = tmp_path / "reports" / ".nightly.lock"
    lock.parent.mkdir(parents=True)
    holder = lock.open("a+")
    try:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        provider, submit = FakeProvider(True), FakeSubmit()
        receipt = _run(tmp_path, FakeClock(IN_WINDOW), provider, submit)
        assert receipt.status == "busy_legacy"
        assert submit.call_count == 0  # never submit beside a heavy legacy run
        assert not holder.closed  # the test's own lock is untouched
        stored = load_state(tmp_path, AS_OF)
        assert stored is not None and stored.status == "busy_legacy"
    finally:
        holder.close()
    submit = FakeSubmit("plan_ref_after_release")
    receipt = _run(tmp_path, FakeClock(IN_WINDOW), provider, submit)
    assert receipt.status == "submitted" and submit.call_count == 1


def test_legacy_lock_path_is_the_repo_reports_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(nightly_trigger, "repo_root", lambda: tmp_path)
    assert nightly_trigger.legacy_lock_path() == tmp_path / "reports" / ".nightly.lock"
    assert state_path(tmp_path, AS_OF) == (
        tmp_path / "reports" / "phase6" / "nightly_trigger" / f"{AS_OF}.json")


# --------------------------------------------------------------------------
# CLI: one JSON receipt line; exit 0 for not_yet, 1 for missed/error
# --------------------------------------------------------------------------


def test_main_prints_one_receipt_and_exits_zero_for_not_yet(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(nightly_trigger, "SystemClock", lambda: FakeClock(IN_WINDOW))
    monkeypatch.setattr(nightly_trigger, "_orats_probe",
                        lambda as_of, tickers: (False, "not published"))
    code = nightly_trigger.main(["--as-of", AS_OF, "--root", str(tmp_path)])
    lines = capsys.readouterr().out.strip().splitlines()
    assert code == 0 and len(lines) == 1
    assert json.loads(lines[0])["status"] == "not_yet"


def test_main_exits_one_for_missed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(nightly_trigger, "SystemClock", lambda: FakeClock(AFTER_DEADLINE))
    code = nightly_trigger.main(["--as-of", AS_OF, "--root", str(tmp_path)])
    document = json.loads(capsys.readouterr().out.strip())
    assert code == 1 and document["status"] == "missed"


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
