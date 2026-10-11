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
from threading import Event, Thread

import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly_completion import Disposition, build_completion_report
from engine.v2.ops.nightly_session import (
    STATE_DIR,
    SessionIdentity,
    ensure_session,
    mark_started,
    request_rerun,
    session_path,
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


@pytest.mark.parametrize("required_status", ["failed", "degraded"])
def test_required_failure_or_degradation_is_never_clean_success_even_with_clean_sidecars(
        required_status):
    expected = {"primary": "required", "side": "optional", "quality": "qualification"}
    observed = {"primary": Disposition(
                    required_status,
                    reason_code="WORK_FAILED" if required_status == "failed" else "WORK_DEGRADED"),
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


def test_optional_pending_sidecar_waits_without_changing_qualification():
    expected = {"primary": "required", "quality": "qualification", "optional": "optional"}
    observed = {"primary": Disposition("succeeded"), "quality": Disposition("succeeded"),
                "optional": Disposition("pending", count=1)}
    report = _report(expected, observed)
    assert report["status"] == "waiting"
    assert report["degraded_outcome"] == "degraded"
    assert report["qualification_outcome"] == "qualified"
    assert report["sidecars_terminal"] == 1


def test_missing_qualification_disposition_is_pending():
    expected = {"primary": "required", "quality": "qualification"}
    report = _report(expected, {"primary": Disposition("succeeded")})
    assert report["qualification_outcome"] == "pending"
    assert report["status"] == "waiting"


@pytest.mark.parametrize("quality", [
    Disposition("recovery_required", reason_code="RECOVERY_REQUIRED"),
    Disposition("succeeded", evidence_valid=False),
    Disposition("pending", evidence_valid=False),
])
def test_recovery_or_invalid_qualification_evidence_is_invalid(quality):
    expected = {"primary": "required", "quality": "qualification"}
    report = _report(expected, {"primary": Disposition("succeeded"), "quality": quality})
    assert report["qualification_outcome"] == "invalid"
    assert report["status"] == "recovery_required"


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


@pytest.mark.parametrize("sidecar,overall,terminal_count", [
    (Disposition("recovery_required"), "recovery_required", 1),
    (Disposition("succeeded", evidence_valid=False), "recovery_required", 1),
    (Disposition("failed", reason_code="SIDE_FAILED"), "completed_degraded", 2),
    (Disposition("timed_out", reason_code="SIDE_TIMEOUT"), "completed_degraded", 2),
])
def test_optional_sidecar_does_not_change_qualification_outcome(
        sidecar, overall, terminal_count):
    expected = {"primary": "required", "quality": "qualification", "optional": "optional"}
    observed = {"primary": Disposition("succeeded"), "quality": Disposition("succeeded"),
                "optional": sidecar}
    report = _report(expected, observed)
    assert report["qualification_outcome"] == "qualified"
    assert report["status"] == overall
    assert report["sidecars_terminal"] == terminal_count


def test_optional_count_degrades_report_but_required_count_is_report_only():
    expected = {"primary": "required", "quality": "qualification", "optional": "optional"}
    observed = {"primary": Disposition("succeeded", count=7),
                "quality": Disposition("succeeded"),
                "optional": Disposition("succeeded", count=2)}
    report = _report(expected, observed)
    assert report["required_outcome"] == "completed"
    assert report["degraded_outcome"] == "degraded"
    assert report["qualification_outcome"] == "qualified"
    assert report["status"] == "completed_degraded"


def test_qualification_count_marks_qualification_degraded():
    report = _report({"primary": "required", "quality": "qualification"},
                     {"primary": Disposition("succeeded"),
                      "quality": Disposition("succeeded", count=1)})
    assert report["qualification_outcome"] == "degraded"
    assert report["degraded_outcome"] == "degraded"
    assert report["status"] == "completed_degraded"


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


@pytest.mark.parametrize("identity", [None, "bad", {}])
def test_recording_malformed_identity_refuses_without_report_write(tmp_path, identity):
    conn, service = _service(tmp_path)
    try:
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(identity, 1, {"primary": "required"}, {})
        assert exc.value.code == "INVALID_REQUEST"
        report_dir = tmp_path.joinpath(*STATE_DIR)
        assert not report_dir.exists() or not list(report_dir.glob("*.completion.json"))
    finally:
        conn.close()


@pytest.mark.parametrize("expected, observed", [
    ({"primary": "invalid-role"}, {}),
    ({"primary": "required"}, {"primary": object()}),
])
def test_recording_malformed_work_refuses_without_report_write(tmp_path, expected, observed):
    ensure_session(tmp_path, IDENT)
    mark_started(tmp_path, IDENT, 1)
    conn, service = _service(tmp_path)
    try:
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(IDENT, 1, expected, observed)
        assert exc.value.code == "INVALID_REQUEST"
        assert not list(tmp_path.joinpath(*STATE_DIR).glob("*.completion.json"))
    finally:
        conn.close()


def test_recording_missing_or_unstarted_session_refuses(tmp_path):
    conn, service = _service(tmp_path)
    malformed = {"primary": "invalid-role"}
    try:
        with pytest.raises(OpsError) as missing:
            service.record_nightly_completion(IDENT, 1, {"primary": "required"}, {})
        assert missing.value.code == "CHECKPOINT_INCOMPATIBLE"
        with pytest.raises(OpsError) as missing_malformed:
            service.record_nightly_completion(IDENT, 1, malformed, {})
        assert missing_malformed.value.code == "CHECKPOINT_INCOMPATIBLE"
        ensure_session(tmp_path, IDENT)  # allocated, but not started
        with pytest.raises(OpsError) as allocated:
            service.record_nightly_completion(IDENT, 1, {"primary": "required"}, {})
        assert allocated.value.code == "CHECKPOINT_INCOMPATIBLE"
        with pytest.raises(OpsError) as allocated_malformed:
            service.record_nightly_completion(IDENT, 1, malformed, {})
        assert allocated_malformed.value.code == "CHECKPOINT_INCOMPATIBLE"
        assert not list(tmp_path.joinpath(*STATE_DIR).glob("*.completion.json"))
    finally:
        conn.close()


def test_service_records_a_real_run_scoped_report_and_is_byte_stable(tmp_path):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    run_id = state.active.run_id
    conn, service = _service(tmp_path)
    try:
        expected = {"primary": "required", "secondary": "required",
                    "side": "optional", "quality": "qualification"}
        observed = {"primary": Disposition("succeeded"),
                    "secondary": Disposition("succeeded"),
                    "side": Disposition("succeeded"),
                    "quality": Disposition("succeeded")}
        report = service.record_nightly_completion(IDENT, 1, expected, observed)
        path = tmp_path.joinpath(*STATE_DIR, f"{run_id}.completion.json")
        text = path.read_text()
        persisted = json.loads(text)
        assert persisted == report
        assert persisted["schema_version"] == "nightly_completion.v1"
        assert persisted["required_count"] == 2
        assert persisted["sidecar_count"] == 2
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


def test_service_persists_expected_roles_and_evidence_validity(tmp_path):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    conn, service = _service(tmp_path)
    expected = {"primary": "required", "side": "optional",
                "quality": "qualification"}
    observed = {"primary": Disposition("succeeded"),
                "side": Disposition("succeeded", evidence_valid=False),
                "quality": Disposition("succeeded")}
    try:
        service.record_nightly_completion(IDENT, 1, expected, observed)
        path = tmp_path.joinpath(*STATE_DIR, f"{state.active.run_id}.completion.json")
        persisted = json.loads(path.read_text())
        assert persisted["dispositions"] == [
            {"name": "primary", "role": "required", "status": "succeeded",
             "reason_code": None, "count": 0, "evidence_valid": True},
            {"name": "quality", "role": "qualification", "status": "succeeded",
             "reason_code": None, "count": 0, "evidence_valid": True},
            {"name": "side", "role": "optional", "status": "succeeded",
             "reason_code": None, "count": 0, "evidence_valid": False},
        ]
    finally:
        conn.close()


@pytest.mark.parametrize("target, unsafe", [
    ("session", "symlink"), ("session", "fifo"),
    ("lock", "symlink"), ("lock", "fifo"),
])
def test_service_rejects_unsafe_session_and_lock_files_without_blocking(
        tmp_path, target, unsafe):
    ensure_session(tmp_path, IDENT)
    mark_started(tmp_path, IDENT, 1)
    session_file = session_path(tmp_path, IDENT)
    unsafe_path = (session_file if target == "session" else
                   session_file.with_name(session_file.name + ".lock"))
    sentinel = tmp_path / f"{target}-sentinel"
    if unsafe == "symlink":
        if target == "session":
            session_file.rename(sentinel)
        else:
            sentinel.write_text("lock sentinel")
            unsafe_path.unlink()
        unsafe_path.symlink_to(sentinel)
    else:
        unsafe_path.unlink(missing_ok=True)
        os.mkfifo(unsafe_path)

    conn, service = _service(tmp_path)
    errors = []

    def record():
        try:
            service.record_nightly_completion(
                IDENT, 1, {"primary": "required"},
                {"primary": Disposition("succeeded")})
        except Exception as exc:
            errors.append(exc)

    worker = Thread(target=record, daemon=True)
    try:
        worker.start()
        worker.join(timeout=3)
        assert not worker.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], OpsError)
        assert errors[0].code == "INTEGRITY_FAILED"
        report_dir = tmp_path.joinpath(*STATE_DIR)
        assert not list(report_dir.glob("*.completion.json"))
        if unsafe == "symlink":
            assert unsafe_path.is_symlink()
            assert sentinel.exists()
    finally:
        conn.close()


def test_service_rejects_oversized_session_state_without_report(tmp_path):
    ensure_session(tmp_path, IDENT)
    mark_started(tmp_path, IDENT, 1)
    session_path(tmp_path, IDENT).write_bytes(b"x" * (1024 * 1024 + 1))
    conn, service = _service(tmp_path)
    try:
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(
                IDENT, 1, {"primary": "required"},
                {"primary": Disposition("succeeded")})
        assert exc.value.code == "INTEGRITY_FAILED"
        assert not list(tmp_path.joinpath(*STATE_DIR).glob("*.completion.json"))
    finally:
        conn.close()


def test_service_rejects_preexisting_completion_temp_symlink(tmp_path):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    report_path = tmp_path.joinpath(
        *STATE_DIR, f"{state.active.run_id}.completion.json")
    external = tmp_path / "external-sentinel"
    external.write_text("preserve me")
    temp_link = report_path.with_name(report_path.name + ".tmp")
    temp_link.symlink_to(external)
    conn, service = _service(tmp_path)
    try:
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(
                IDENT, 1, {"primary": "required"},
                {"primary": Disposition("succeeded")})
        assert exc.value.code == "INTEGRITY_FAILED"
        assert not report_path.exists()
        assert external.read_text() == "preserve me"
        assert temp_link.is_symlink()
    finally:
        conn.close()


def test_service_rejects_symlinked_session_directory_before_report(tmp_path):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    session_dir = session_path(tmp_path, IDENT).parent
    moved_dir = session_dir.with_name("sessions-moved")
    session_dir.rename(moved_dir)
    session_dir.symlink_to(moved_dir, target_is_directory=True)
    conn, service = _service(tmp_path)
    try:
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(
                IDENT, 1, {"primary": "required"},
                {"primary": Disposition("succeeded")})
        assert exc.value.code == "INTEGRITY_FAILED"
        assert not moved_dir.joinpath(f"{state.active.run_id}.completion.json").exists()
        assert not list(moved_dir.glob("*.tmp"))
    finally:
        conn.close()


def test_service_rejects_directory_at_completion_destination(tmp_path):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    report_path = tmp_path.joinpath(
        *STATE_DIR, f"{state.active.run_id}.completion.json")
    report_path.mkdir()
    conn, service = _service(tmp_path)
    try:
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(
                IDENT, 1, {"primary": "required"},
                {"primary": Disposition("succeeded")})
        assert exc.value.code == "INTEGRITY_FAILED"
        assert report_path.is_dir()
        assert not report_path.with_name(report_path.name + ".tmp").exists()
    finally:
        conn.close()


def test_service_reclaims_stale_regular_completion_temp(tmp_path):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    report_path = tmp_path.joinpath(
        *STATE_DIR, f"{state.active.run_id}.completion.json")
    stale_temp = report_path.with_name(report_path.name + ".tmp")
    stale_temp.write_text("left by a prior process stop")
    conn, service = _service(tmp_path)
    try:
        report = service.record_nightly_completion(
            IDENT, 1, {"primary": "required"},
            {"primary": Disposition("succeeded")})
        assert report["status"] == "completed"
        assert json.loads(report_path.read_text()) == report
        assert not stale_temp.exists()
    finally:
        conn.close()


def test_same_generation_record_rebuilds_instead_of_using_report_cache(tmp_path):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    conn, service = _service(tmp_path)
    expected = {"primary": "required", "side": "optional"}
    path = tmp_path.joinpath(*STATE_DIR, f"{state.active.run_id}.completion.json")
    try:
        first = service.record_nightly_completion(
            IDENT, 1, expected,
            {"primary": Disposition("succeeded"), "side": Disposition("succeeded")})
        updated = service.record_nightly_completion(
            IDENT, 1, expected,
            {"primary": Disposition("succeeded"),
             "side": Disposition("failed", reason_code="OPTIONAL_FAILED")})
        assert first["status"] == "completed"
        assert updated["status"] == "completed_degraded"
        assert updated != first
        assert json.loads(path.read_text()) == updated
    finally:
        conn.close()


@pytest.mark.parametrize("has_previous", [False, True])
def test_failed_completion_replace_keeps_previous_report_or_none(
        tmp_path, monkeypatch, has_previous):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    conn, service = _service(tmp_path)
    expected = {"primary": "required"}
    path = tmp_path.joinpath(*STATE_DIR, f"{state.active.run_id}.completion.json")
    try:
        if has_previous:
            service.record_nightly_completion(
                IDENT, 1, expected, {"primary": Disposition("succeeded")})
        previous = path.read_bytes() if has_previous else None
        real_replace = os.replace

        def fail_completion_replace(source, destination):
            if str(destination).endswith(".completion.json"):
                raise OSError("replace failed")
            return real_replace(source, destination)

        monkeypatch.setattr(os, "replace", fail_completion_replace)
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(
                IDENT, 1, expected,
                {"primary": Disposition("failed", reason_code="STEP_FAILED")})
        assert exc.value.code == "INTEGRITY_FAILED"
        if previous is None:
            assert not path.exists()
        else:
            assert path.read_bytes() == previous
        assert not path.with_name(path.name + ".tmp").exists()
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
                assert second.result(timeout=5) == report
            except OpsError as exc:
                assert exc.code == "CHECKPOINT_INCOMPATIBLE"
            active = next_generation.result(timeout=5)
        assert active.active.generation == 2
        old_path = tmp_path.joinpath(*STATE_DIR, f"{report['run_id']}.completion.json")
        before = old_path.read_bytes()
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(IDENT, 1, expected, observed)
        assert exc.value.code == "CHECKPOINT_INCOMPATIBLE"
        assert old_path.read_bytes() == before
        assert json.loads(before) == report
    finally:
        release.set()
        conn.close()