"""Run-scoped nightly completion reports (slice 7, ``nightly_completion.py``).

The outcomes are derived only from explicit required/optional/qualification
dispositions; supervisor idleness never implies completion. The integration
test drives the *real* session store, the actual ``Service`` method, the real
atomic writer and the real JSON reader -- none of those layers is stubbed.
"""
from __future__ import annotations

import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly_completion import Disposition, build_completion_report
from engine.v2.ops.nightly_session import (
    STATE_DIR,
    SessionIdentity,
    ensure_session,
    mark_started,
    request_rerun,
)
from engine.v2.ops.stages import registry
from engine.v2.ops.supervisor import Service

IDENT = SessionIdentity("2026-10-08", "shadow", "sel-1", "cat-1")
RUN_ID = "a" * 64


class _Clock:
    """The minimal clock contract ``Service`` needs: wall time and monotonic time."""

    def __init__(self) -> None:
        self._mono = 0.0

    def now(self) -> str:
        return "2026-10-08T00:00:00+00:00"

    def monotonic(self) -> float:
        self._mono += 0.001
        return self._mono


def _report(expected, observed, run_id=RUN_ID):
    return build_completion_report(run_id, expected, observed)


def _names(report):
    return [row["name"] for row in report["dispositions"]]


def _row(report, name):
    return next(row for row in report["dispositions"] if row["name"] == name)


def _service(root):
    conn = sqlite3.connect(":memory:")
    service = Service(conn, root, registry(), None, clock=_Clock(), code_source=root,
                      store_root=root, materialization_base=root / "materializations")
    return conn, service


def test_clean_required_and_succeeded_qualification_is_completed():
    expected = {"qualifier": "qualification", "primary": "required"}
    observed = {"primary": Disposition("succeeded", count=2),
                "qualifier": Disposition("succeeded")}
    report = _report(expected, observed)
    assert report["status"] == "completed"
    assert report["required_outcome"] == "completed"
    assert report["degraded_outcome"] == "clean"
    assert report["qualification_outcome"] == "qualified"
    assert _names(report) == ["primary", "qualifier"]  # stable sorted, not insertion order
    assert report["sidecars_terminal"] == 1
    assert _row(report, "primary")["count"] == 2


def test_required_failure_is_never_clean_success_even_with_clean_sidecars():
    expected = {"primary": "required", "side": "optional", "quality": "qualification"}
    observed = {"primary": Disposition("failed", reason_code="WORK_FAILED"),
                "side": Disposition("succeeded"), "quality": Disposition("succeeded")}
    report = _report(expected, observed)
    assert report["required_outcome"] == "failed"
    assert report["status"] == "failed"
    assert report["degraded_outcome"] == "clean"


def test_missing_selected_sidecar_waits_and_is_never_inferred_from_idleness():
    expected = {"primary": "required", "side": "optional"}
    report = _report(expected, {"primary": Disposition("succeeded")})
    assert report["status"] == "waiting"
    assert report["required_outcome"] == "completed"
    assert report["sidecars_terminal"] == 0
    assert _row(report, "side")["status"] == "pending"
    assert report["status"] != "completed"


def test_qualification_stays_pending_until_every_selected_sidecar_is_terminal():
    expected = {"primary": "required", "quality": "qualification", "optional": "optional"}
    observed = {"primary": Disposition("succeeded"), "quality": Disposition("succeeded"),
                "optional": Disposition("pending")}
    report = _report(expected, observed)
    assert report["status"] == "waiting"
    assert report["qualification_outcome"] == "pending"
    assert report["sidecars_terminal"] == 1


def test_optional_sidecar_failure_is_completed_degraded_without_qualification():
    expected = {"primary": "required", "side": "optional"}
    observed = {"primary": Disposition("succeeded"),
                "side": Disposition("failed", reason_code="SIDE_FAILED")}
    report = _report(expected, observed)
    assert report["status"] == "completed_degraded"
    assert report["required_outcome"] == "completed"
    assert report["qualification_outcome"] == "not_requested"


def test_valid_qualification_refusal_yields_degraded_with_explicit_count_and_code():
    expected = {"primary": "required", "quality": "qualification"}
    observed = {"primary": Disposition("succeeded"),
                "quality": Disposition("degraded", reason_code="REFUSED", count=3)}
    report = _report(expected, observed)
    assert report["status"] == "completed_degraded"
    assert report["qualification_outcome"] == "degraded"
    row = _row(report, "quality")
    assert row["count"] == 3 and row["reason_code"] == "REFUSED"


def test_qualification_failure_is_reported_separately_from_completed_required():
    expected = {"primary": "required", "quality": "qualification"}
    observed = {"primary": Disposition("succeeded"),
                "quality": Disposition("failed", reason_code="QUALITY_FAILED")}
    report = _report(expected, observed)
    assert report["qualification_outcome"] == "failed"
    assert report["required_outcome"] == "completed"
    assert report["status"] == "completed_degraded"


def test_invalid_evidence_requires_recovery_and_is_not_terminal():
    expected = {"primary": "required", "side": "optional"}
    observed = {"primary": Disposition("succeeded", evidence_valid=False),
                "side": Disposition("succeeded", evidence_valid=False)}
    report = _report(expected, observed)
    assert report["status"] == "recovery_required"
    assert report["required_outcome"] == "recovery_required"
    assert report["sidecars_terminal"] == 0


@pytest.mark.parametrize("required,sidecar,required_status,overall", [
    (Disposition("pending"), Disposition("recovery_required"), "waiting", "recovery_required"),
    (Disposition("pending"), Disposition("succeeded", evidence_valid=False),
     "waiting", "recovery_required"),
    (Disposition("timed_out"), Disposition("succeeded"), "timed_out", "timed_out"),
    (Disposition("degraded"), Disposition("succeeded"), "failed", "failed"),
])
def test_uncertain_sidecar_and_required_timeout_remain_non_success(
        required, sidecar, required_status, overall):
    expected = {"primary": "required", "quality": "qualification"}
    observed = {"primary": required, "quality": sidecar}
    report = _report(expected, observed)
    assert report["required_outcome"] == required_status
    assert report["status"] == overall
    assert report["status"] not in {"completed", "completed_degraded"}


@pytest.mark.parametrize("expected,observed", [
    ({"primary": "bogus"}, {"primary": Disposition("succeeded")}),
    ({"side": "optional"}, {"side": Disposition("succeeded")}),
    ({"primary": "required"},
     {"primary": Disposition("succeeded"), "extra": Disposition("succeeded")}),
    ({"primary": "required"}, {"primary": "succeeded"}),
])
def test_malformed_no_required_extra_or_unselected_inputs_refuse(expected, observed):
    with pytest.raises(OpsError) as exc:
        _report(expected, observed)
    assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("raw", ["/tmp/secret path", "FileNotFoundError: it broke", "lowercase"])
def test_raw_message_or_path_is_not_a_valid_reason_code(raw):
    expected = {"primary": "required"}
    observed = {"primary": Disposition("failed", reason_code=raw)}
    with pytest.raises(OpsError) as exc:
        _report(expected, observed)
    assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("run_id,expected", [
    ("/tmp/private", {"primary": "required"}),
    ("a" * 64, {"/tmp/private": "optional", "primary": "required"}),
])
def test_report_rejects_pathlike_identifiers(run_id, expected):
    with pytest.raises(OpsError) as exc:
        _report(expected, {}, run_id=run_id)
    assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("sidecar,qualification,overall,terminal_count", [
    (Disposition("recovery_required"), "invalid", "recovery_required", 1),
    (Disposition("succeeded", evidence_valid=False), "invalid", "recovery_required", 1),
    (Disposition("failed", reason_code="SIDE_FAILED"), "failed", "completed_degraded", 2),
    (Disposition("timed_out", reason_code="SIDE_TIMEOUT"), "failed", "completed_degraded", 2),
])
def test_optional_sidecar_disposition_is_part_of_requested_qualification(
        sidecar, qualification, overall, terminal_count):
    expected = {"primary": "required", "quality": "qualification", "optional": "optional"}
    observed = {"primary": Disposition("succeeded"), "quality": Disposition("succeeded"),
                "optional": sidecar}
    report = _report(expected, observed)
    assert report["qualification_outcome"] == qualification
    assert report["status"] == overall
    assert report["sidecars_terminal"] == terminal_count


@pytest.mark.parametrize("disposition", [
    Disposition("succeeded", count=True),
    Disposition("succeeded", evidence_valid=1),
])
def test_boolean_count_and_evidence_flags_are_rejected(disposition):
    with pytest.raises(OpsError) as exc:
        _report({"primary": "required"}, {"primary": disposition})
    assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("expected", [
    {"primary": "invalid-role"},
    {"bad/name": "required"},
])
def test_invalid_work_names_and_roles_have_a_named_work_refusal(expected):
    with pytest.raises(OpsError) as exc:
        _report(expected, {})
    assert exc.value.code == "INVALID_REQUEST"
    assert exc.value.problem.message == "completion needs named expected work"


def test_recording_missing_or_unstarted_session_refuses(tmp_path):
    conn, service = _service(tmp_path)
    try:
        with pytest.raises(OpsError) as missing:
            service.record_nightly_completion(IDENT, 1, {"primary": "required"}, {})
        assert missing.value.code == "CHECKPOINT_INCOMPATIBLE"
        ensure_session(tmp_path, IDENT)  # allocated, but not started
        with pytest.raises(OpsError) as allocated:
            service.record_nightly_completion(IDENT, 1, {"primary": "required"}, {})
        assert allocated.value.code == "CHECKPOINT_INCOMPATIBLE"
    finally:
        conn.close()


def test_service_records_a_real_run_scoped_report_and_is_byte_stable(tmp_path):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    run_id = state.active.run_id
    conn, service = _service(tmp_path)
    try:
        expected = {"primary": "required", "side": "optional", "quality": "qualification"}
        observed = {"primary": Disposition("succeeded"), "side": Disposition("succeeded"),
                    "quality": Disposition("succeeded")}
        report = service.record_nightly_completion(IDENT, 1, expected, observed)
        path = tmp_path.joinpath(*STATE_DIR, f"{run_id}.completion.json")
        text = path.read_text()
        assert json.loads(text) == report
        assert report["run_id"] == run_id  # scoped to this run, not its date or key
        assert report["status"] == "completed"
        assert str(tmp_path) not in text  # redacted: no local path leaked
        assert "Traceback" not in text and "FileNotFoundError" not in text
        again = service.record_nightly_completion(IDENT, 1, expected, observed)
        assert again == report
        assert path.read_text() == text == (
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    finally:
        conn.close()


def test_overlapping_recorders_and_rerun_share_the_session_lock(tmp_path, monkeypatch):
    ensure_session(tmp_path, IDENT)
    mark_started(tmp_path, IDENT, 1)
    conn, service = _service(tmp_path)
    expected = {"primary": "required"}
    observed = {"primary": Disposition("succeeded")}
    entered, release = Event(), Event()
    second_replace, rerun_replace = Event(), Event()
    second_started, rerun_started = Event(), Event()
    real_replace = os.replace

    def pause_completion_replace(source, destination):
        target = str(destination)
        if target.endswith(".completion.json"):
            if not entered.is_set():
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("test writer was not released")
            else:
                second_replace.set()
        elif target.endswith(f"{IDENT.session_key}.json"):
            rerun_replace.set()
        return real_replace(source, destination)

    def record_again():
        second_started.set()
        return service.record_nightly_completion(IDENT, 1, expected, observed)

    def rerun():
        rerun_started.set()
        return request_rerun(tmp_path, IDENT)

    monkeypatch.setattr(os, "replace", pause_completion_replace)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            first = pool.submit(service.record_nightly_completion, IDENT, 1, expected, observed)
            assert entered.wait(5)
            second = pool.submit(record_again)
            next_generation = pool.submit(rerun)
            assert second_started.wait(5) and rerun_started.wait(5)
            assert not second_replace.wait(0.1) and not rerun_replace.wait(0.1)
            assert not second.done() and not next_generation.done()
            release.set()
            report = first.result(timeout=5)
            try:
                second_report = second.result(timeout=5)
            except OpsError as exc:
                assert exc.code == "CHECKPOINT_INCOMPATIBLE"
                second_report = report
            active = next_generation.result(timeout=5)
        assert active.active.generation == 2
        old_path = tmp_path.joinpath(*STATE_DIR, f"{report['run_id']}.completion.json")
        before = old_path.read_bytes()
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(IDENT, 1, expected, observed)
        assert exc.value.code == "CHECKPOINT_INCOMPATIBLE"
        assert old_path.read_bytes() == before
        assert json.loads(before) in (report, second_report)
    finally:
        release.set()
        conn.close()