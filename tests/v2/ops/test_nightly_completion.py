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
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread

import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly_completion import (
    Disposition, build_completion_report, record_completion)
from engine.v2.ops import nightly_session
from engine.v2.ops.nightly_session import (
    MAX_SESSION_BYTES,
    atomic_write,
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
    assert report["required_count"] == 1
    assert report["sidecar_count"] == 1


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


@pytest.mark.parametrize(("quality_status", "reason_code"), [
    ("failed", "QUALITY_FAILED"),
    ("timed_out", "QUALITY_TIMEOUT"),
])
def test_qualification_failure_is_reported_separately_from_completed_required(
        quality_status, reason_code):
    expected = {"primary": "required", "quality": "qualification"}
    observed = {"primary": Disposition("succeeded"),
                "quality": Disposition(quality_status, reason_code=reason_code)}
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
    (Disposition("recovery_required"), Disposition("succeeded"),
     "recovery_required", "recovery_required"),
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


@pytest.mark.parametrize("run_id,expected,observed", [
    pytest.param(RUN_ID, {"primary": "required"}, {"primary": Disposition("unknown")},
                 id="unknown-status"),
    pytest.param(RUN_ID, {"primary": "required"}, {"primary": Disposition([])},
                 id="nonstring-status"),
    pytest.param(RUN_ID, {"primary": "required"},
                 {"primary": Disposition("succeeded", count=-1)}, id="negative-count"),
    pytest.param(RUN_ID, {"primary": "required"},
                 {"primary": Disposition("succeeded", count=1.5)}, id="noninteger-count"),
    pytest.param(RUN_ID, {"primary": "required"},
                 {"primary": Disposition("succeeded", count=10**5000)},
                 id="count-exceeds-json-integer-limit"),
    pytest.param(RUN_ID, {"primary": "required"},
                 {"primary": Disposition("failed", reason_code=7)}, id="nonstring-reason"),
    pytest.param(RUN_ID, None, {}, id="nonmapping-expected"),
    pytest.param(RUN_ID, {"primary": "required"}, None, id="nonmapping-observed"),
    pytest.param(RUN_ID, {}, {}, id="empty-expected-work"),
    pytest.param(RUN_ID, {7: "required"}, {}, id="nonstring-work-name"),
    pytest.param(RUN_ID, {"primary": []}, {}, id="nonstring-work-role"),
    pytest.param("a" * 63, {"primary": "required"}, {}, id="short-run-id"),
    pytest.param("a" * 65, {"primary": "required"}, {}, id="long-run-id"),
    pytest.param(RUN_ID, {"n" * 65: "required"}, {}, id="long-work-name"),
])
def test_each_independent_validation_guard_returns_invalid_request(
        run_id, expected, observed):
    with pytest.raises(OpsError) as exc:
        _report(expected, observed, run_id=run_id)
    assert exc.value.code == "INVALID_REQUEST"


def test_64_character_run_and_work_names_are_valid():
    name = "n" * 64
    report = _report({name: "required"}, {}, run_id=RUN_ID)
    assert report["status"] == "waiting"
    assert report["dispositions"][0]["name"] == name


@pytest.mark.parametrize(("count", "accepted"), [
    pytest.param(2**63 - 1, True, id="maximum-count"),
    pytest.param(2**63, False, id="one-above-maximum"),
])
def test_count_limit_has_an_inclusive_boundary(count, accepted):
    expected = {"primary": "required"}
    observed = {"primary": Disposition("succeeded", count=count)}
    if accepted:
        report = _report(expected, observed)
        assert report["dispositions"][0]["count"] == 2**63 - 1
    else:
        with pytest.raises(OpsError) as exc:
            _report(expected, observed)
        assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize(("reason_code", "accepted"), [
    pytest.param("A" * 64, True, id="maximum-reason-code"),
    pytest.param("A" * 65, False, id="oversized-reason-code"),
])
def test_reason_code_limit_has_an_inclusive_boundary(reason_code, accepted):
    expected = {"primary": "required"}
    observed = {"primary": Disposition("failed", reason_code=reason_code)}
    if accepted:
        report = _report(expected, observed)
        assert report["dispositions"][0]["reason_code"] == reason_code
    else:
        with pytest.raises(OpsError) as exc:
            _report(expected, observed)
        assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("run_id", [None, [], 7])
def test_nonstring_run_id_is_invalid_request(run_id):
    with pytest.raises(OpsError) as exc:
        _report({"primary": "required"}, {}, run_id=run_id)
    assert exc.value.code == "INVALID_REQUEST"
    assert exc.value.problem.message == "completion needs a run ID"


@pytest.mark.parametrize("expected,observed", [
    (None, {}),
    ({"primary": "required"}, None),
])
def test_nonmapping_inputs_take_the_typed_mapping_refusal(expected, observed):
    with pytest.raises(OpsError) as exc:
        _report(expected, observed)
    assert exc.value.code == "INVALID_REQUEST"
    assert exc.value.problem.message == "completion dispositions must be mappings"


def test_shared_atomic_writer_default_preserves_large_sibling_artifact_capacity(tmp_path):
    path = tmp_path / "history-manifest.json"
    payload = "x" * (MAX_SESSION_BYTES + 1)
    atomic_write(path, payload)
    assert path.read_text() == payload


def test_expected_work_requires_at_least_one_required_item():
    with pytest.raises(OpsError) as exc:
        _report({"only-sidecar": "optional"}, {})
    assert exc.value.code == "INVALID_REQUEST"
    assert exc.value.problem.message == "completion needs required work"


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
    (Disposition("degraded", reason_code="SIDE_DEGRADED"), "completed_degraded", 2),
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
    assert report["required_count"] == 1
    assert report["sidecar_count"] == 2


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


@pytest.mark.parametrize("quality", [
    Disposition("succeeded", count=1),
    Disposition("degraded"),
])
def test_qualification_count_or_status_marks_qualification_degraded(quality):
    report = _report({"primary": "required", "quality": "qualification"},
                     {"primary": Disposition("succeeded"), "quality": quality})
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


def test_real_recording_rejects_integer_beyond_json_limit_before_publication(tmp_path):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    report_path = tmp_path.joinpath(
        *STATE_DIR, f"{state.active.run_id}.completion.json")
    conn, service = _service(tmp_path)
    try:
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(
                IDENT, 1, {"primary": "required"},
                {"primary": Disposition("succeeded", count=10**5000)})
        assert exc.value.code == "INVALID_REQUEST"
        assert not report_path.exists()
        assert not report_path.with_name(report_path.name + ".tmp").exists()
    finally:
        conn.close()


@pytest.mark.parametrize("identity", [None, "bad", {}])
def test_record_completion_identity_guard_is_typed_before_lock(tmp_path, identity):
    with pytest.raises(OpsError) as exc:
        record_completion(tmp_path, identity, 1, {"primary": "required"}, {})
    assert exc.value.code == "INVALID_REQUEST"
    assert exc.value.problem.message == "completion needs a session identity"
    assert not tmp_path.joinpath(*STATE_DIR).exists()


@pytest.mark.parametrize("identity", [None, "bad", {}])
def test_recording_malformed_identity_refuses_without_report_write(tmp_path, identity):
    conn, service = _service(tmp_path)
    try:
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(identity, 1, {"primary": "required"}, {})
        assert exc.value.code == "INVALID_REQUEST"
        report_dir = tmp_path.joinpath(*STATE_DIR)
        assert not report_dir.exists()  # malformed identity is rejected before the lock creates it
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


@pytest.mark.parametrize("generation", [True, 1.0, "1", 2])
def test_record_completion_rejects_invalid_generation_with_exact_code(
        tmp_path, generation):
    ensure_session(tmp_path, IDENT)
    mark_started(tmp_path, IDENT, 1)
    with pytest.raises(OpsError) as exc:
        record_completion(
            tmp_path, IDENT, generation, {"primary": "required"}, {})
    assert exc.value.code == "CHECKPOINT_INCOMPATIBLE"
    assert exc.value.problem.message == (
        "completion generation is not the active started run")
    assert not list(tmp_path.joinpath(*STATE_DIR).glob("*.completion.json"))


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


def test_service_keeps_pending_sidecar_waiting_without_idle_inference(tmp_path):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    conn, service = _service(tmp_path)
    expected = {"primary": "required", "side": "optional"}
    observed = {"primary": Disposition("succeeded")}
    try:
        report = service.record_nightly_completion(IDENT, 1, expected, observed)
        path = tmp_path.joinpath(
            *STATE_DIR, f"{state.active.run_id}.completion.json")
        persisted = json.loads(path.read_text())
        assert persisted == report
        assert persisted["status"] == "waiting"
        assert persisted["required_outcome"] == "completed"
        assert persisted["sidecars_terminal"] == 0
        assert _row(persisted, "side")["status"] == "pending"
        assert persisted["status"] not in {"completed", "completed_degraded"}
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
    observed = {"primary": Disposition("failed", reason_code="REQUIRED_FAILED", count=3),
                "side": Disposition("degraded", reason_code="SIDECAR_DEGRADED", count=2,
                                    evidence_valid=False),
                "quality": Disposition("timed_out", reason_code="QUALIFICATION_TIMEOUT",
                                        count=4)}
    try:
        service.record_nightly_completion(IDENT, 1, expected, observed)
        path = tmp_path.joinpath(*STATE_DIR, f"{state.active.run_id}.completion.json")
        persisted = json.loads(path.read_text())
        assert persisted["dispositions"] == [
            {"name": "primary", "role": "required", "status": "failed",
             "reason_code": "REQUIRED_FAILED", "count": 3, "evidence_valid": True},
            {"name": "quality", "role": "qualification", "status": "timed_out",
             "reason_code": "QUALIFICATION_TIMEOUT", "count": 4, "evidence_valid": True},
            {"name": "side", "role": "optional", "status": "degraded",
             "reason_code": "SIDECAR_DEGRADED", "count": 2, "evidence_valid": False},
        ]
        assert persisted["required_outcome"] == "failed"
        assert persisted["qualification_outcome"] == "failed"
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
    state_file = session_path(tmp_path, IDENT)
    valid_json = state_file.read_bytes()
    assert json.loads(valid_json)  # the fixture is a valid state document
    state_file.write_bytes(valid_json + b" " * (MAX_SESSION_BYTES + 1 - len(valid_json)))
    assert json.loads(state_file.read_bytes())
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


def test_initial_oversized_session_state_is_refused_before_publication(tmp_path):
    large_identity = SessionIdentity(
        IDENT.as_of, IDENT.scope, "s" * MAX_SESSION_BYTES, IDENT.catalog_identity)
    with pytest.raises(OpsError) as exc:
        ensure_session(tmp_path, large_identity)
    assert exc.value.code == "INTEGRITY_FAILED"
    state_file = session_path(tmp_path, large_identity)
    assert not state_file.exists()
    assert not state_file.with_name(state_file.name + ".tmp").exists()


def test_rerun_growth_is_refused_and_preserves_prior_session_state(tmp_path):
    ensure_session(tmp_path, IDENT)
    mark_started(tmp_path, IDENT, 1)
    state_file = session_path(tmp_path, IDENT)
    before = state_file.read_bytes()
    with pytest.raises(OpsError) as exc:
        request_rerun(tmp_path, IDENT, ("x" * MAX_SESSION_BYTES,))
    assert exc.value.code == "INTEGRITY_FAILED"
    assert state_file.read_bytes() == before
    assert json.loads(before)["generations"][-1]["status"] == "started"
    assert not state_file.with_name(state_file.name + ".tmp").exists()


def test_service_rejects_oversized_valid_completion_report_before_publication(tmp_path):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    expected = {"primary": "required"}
    expected.update({f"side-{index:05d}-{'x' * 50}": "optional"
                     for index in range(12000)})
    observed = {"primary": Disposition("succeeded")}
    report = build_completion_report(state.active.run_id, expected, observed)
    encoded_size = len((json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
                       .encode("utf-8"))
    assert encoded_size > MAX_SESSION_BYTES
    conn, service = _service(tmp_path)
    report_path = tmp_path.joinpath(
        *STATE_DIR, f"{state.active.run_id}.completion.json")
    try:
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(IDENT, 1, expected, observed)
        assert exc.value.code == "INTEGRITY_FAILED"
        assert not report_path.exists()
        assert not report_path.with_name(report_path.name + ".tmp").exists()
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


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_service_rejects_symlink_or_fifo_completion_destination(tmp_path, kind):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    report_path = tmp_path.joinpath(*STATE_DIR, f"{state.active.run_id}.completion.json")
    sentinel = tmp_path / "destination-sentinel"
    sentinel.write_text("preserve")
    if kind == "symlink":
        report_path.symlink_to(sentinel)
    else:
        os.mkfifo(report_path)
    conn, service = _service(tmp_path)
    try:
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(
                IDENT, 1, {"primary": "required"},
                {"primary": Disposition("succeeded")})
        assert exc.value.code == "INTEGRITY_FAILED"
        if kind == "symlink":
            assert report_path.is_symlink()
        else:
            assert report_path.exists() and not report_path.is_file()
        assert sentinel.read_text() == "preserve"
        assert not report_path.with_name(report_path.name + ".tmp").exists()
    finally:
        conn.close()


def test_publication_stays_in_open_directory_when_path_is_swapped(tmp_path, monkeypatch):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    report_dir = session_path(tmp_path, IDENT).parent
    report_path = report_dir / f"{state.active.run_id}.completion.json"
    moved_dir = report_dir.with_name("sessions-moved")
    redirected_dir = report_dir.with_name("sessions-redirected")
    redirected_dir.mkdir()
    conn, service = _service(tmp_path)
    real_replace = os.replace
    swapped = False

    def swap_then_replace(source, destination, *args, **kwargs):
        nonlocal swapped
        if destination == report_path.name and not swapped:
            report_dir.rename(moved_dir)
            report_dir.symlink_to(redirected_dir, target_is_directory=True)
            swapped = True
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", swap_then_replace)
    try:
        report = service.record_nightly_completion(
            IDENT, 1, {"primary": "required"},
            {"primary": Disposition("succeeded")})
        assert swapped
        assert json.loads((moved_dir / report_path.name).read_text()) == report
        assert not (redirected_dir / report_path.name).exists()
        assert not (moved_dir / (report_path.name + ".tmp")).exists()
        assert not (redirected_dir / (report_path.name + ".tmp")).exists()
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


@pytest.mark.parametrize("failure", ["mkdir", "flock"])
def test_transition_lock_filesystem_failures_are_typed_and_redacted(
        tmp_path, monkeypatch, failure):
    ensure_session(tmp_path, IDENT)
    mark_started(tmp_path, IDENT, 1)
    conn, service = _service(tmp_path)
    if failure == "mkdir":
        real_mkdir = Path.mkdir

        def fail_session_mkdir(path, *args, **kwargs):
            if path == session_path(tmp_path, IDENT).parent:
                raise OSError("private path detail")
            return real_mkdir(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fail_session_mkdir)
    else:
        def fail_flock(*args, **kwargs):
            raise OSError("private lock detail")

        monkeypatch.setattr(nightly_session.fcntl, "flock", fail_flock)
    try:
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(
                IDENT, 1, {"primary": "required"},
                {"primary": Disposition("succeeded")})
        assert exc.value.code == "INTEGRITY_FAILED"
        assert "private" not in str(exc.value)
        assert not list(tmp_path.joinpath(*STATE_DIR).glob("*.completion.json"))
    finally:
        conn.close()


@pytest.mark.parametrize("failure", ["file_fsync", "directory_fsync"])
def test_completion_fsync_failures_are_typed_and_preserve_publication_semantics(
        tmp_path, monkeypatch, failure):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    conn, service = _service(tmp_path)
    expected = {"primary": "required"}
    path = tmp_path.joinpath(*STATE_DIR, f"{state.active.run_id}.completion.json")
    try:
        old_report = service.record_nightly_completion(
            IDENT, 1, expected, {"primary": Disposition("succeeded")})
        previous = path.read_bytes()
        real_fsync = os.fsync
        calls = 0

        def fail_selected_fsync(fd):
            nonlocal calls
            calls += 1
            target = 1 if failure == "file_fsync" else 2
            if calls == target:
                raise OSError("private filesystem detail")
            return real_fsync(fd)

        monkeypatch.setattr(os, "fsync", fail_selected_fsync)
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(
                IDENT, 1, expected,
                {"primary": Disposition("failed", reason_code="STEP_FAILED")})
        assert exc.value.code == "INTEGRITY_FAILED"
        assert "private filesystem detail" not in str(exc.value)
        if failure == "file_fsync":
            assert path.read_bytes() == previous
            assert json.loads(previous) == old_report
        else:
            assert json.loads(path.read_text())["required_outcome"] == "failed"
        assert not path.with_name(path.name + ".tmp").exists()
    finally:
        conn.close()


def test_cleanup_failure_does_not_mask_publication_failure(tmp_path, monkeypatch):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    report_path = tmp_path.joinpath(
        *STATE_DIR, f"{state.active.run_id}.completion.json")
    temp_path = report_path.with_name(report_path.name + ".tmp")
    conn, service = _service(tmp_path)
    real_replace, real_unlink = os.replace, os.unlink

    def fail_replace(source, destination, *args, **kwargs):
        if str(destination).endswith(".completion.json"):
            raise OSError("private publication detail")
        return real_replace(source, destination, *args, **kwargs)

    def fail_temp_unlink(name, *args, **kwargs):
        if str(name).endswith(".completion.json.tmp"):
            raise OSError("private cleanup detail")
        return real_unlink(name, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fail_replace)
    monkeypatch.setattr(os, "unlink", fail_temp_unlink)
    try:
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(
                IDENT, 1, {"primary": "required"},
                {"primary": Disposition("succeeded")})
        assert exc.value.code == "INTEGRITY_FAILED"
        assert exc.value.problem.message == "session publication destination is unsafe"
        assert "private" not in str(exc.value)
        assert not report_path.exists()
        assert temp_path.exists()  # injected cleanup failure did not mask replace failure
    finally:
        conn.close()


def test_session_reader_standalone_close_failure_is_typed(tmp_path, monkeypatch):
    ensure_session(tmp_path, IDENT)
    real_close = os.close

    def fail_close(fd):
        real_close(fd)
        raise OSError("private close detail")

    monkeypatch.setattr(os, "close", fail_close)
    with pytest.raises(OpsError) as exc:
        nightly_session.load_session(tmp_path, IDENT)
    assert exc.value.code == "INTEGRITY_FAILED"
    assert exc.value.problem.message == "session state close failed"
    assert "private" not in str(exc.value)


def test_session_reader_close_failure_does_not_mask_read_error(tmp_path, monkeypatch):
    ensure_session(tmp_path, IDENT)
    real_close = os.close

    def fail_read(fd, size):
        raise OSError("private read detail")

    def fail_close(fd):
        real_close(fd)
        raise OSError("private close detail")

    monkeypatch.setattr(os, "read", fail_read)
    monkeypatch.setattr(os, "close", fail_close)
    with pytest.raises(OpsError) as exc:
        nightly_session.load_session(tmp_path, IDENT)
    assert exc.value.code == "INTEGRITY_FAILED"
    assert exc.value.problem.message == "session state is unreadable"
    assert "private" not in str(exc.value)


def test_failed_open_cleanup_close_does_not_mask_regular_file_refusal(
        tmp_path, monkeypatch):
    monkeypatch.setattr(nightly_session, "session_path", lambda *_: tmp_path)
    real_close = os.close

    def fail_close(fd):
        real_close(fd)
        raise OSError("private cleanup close detail")

    monkeypatch.setattr(os, "close", fail_close)
    with pytest.raises(OpsError) as exc:
        nightly_session.load_session(tmp_path, IDENT)
    assert exc.value.code == "INTEGRITY_FAILED"
    assert exc.value.problem.message == "session state is not a regular file"
    assert "private" not in str(exc.value)


def test_standalone_publication_directory_close_failure_is_typed(tmp_path, monkeypatch):
    ensure_session(tmp_path, IDENT)
    state = mark_started(tmp_path, IDENT, 1)
    report_path = tmp_path.joinpath(
        *STATE_DIR, f"{state.active.run_id}.completion.json")
    conn, service = _service(tmp_path)
    real_close = os.close

    def fail_directory_close(fd):
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            real_close(fd)
            raise OSError("private directory close detail")
        return real_close(fd)

    monkeypatch.setattr(os, "close", fail_directory_close)
    try:
        with pytest.raises(OpsError) as exc:
            service.record_nightly_completion(
                IDENT, 1, {"primary": "required"},
                {"primary": Disposition("succeeded")})
        assert exc.value.code == "INTEGRITY_FAILED"
        assert "private directory close detail" not in str(exc.value)
        assert json.loads(report_path.read_text())["status"] == "completed"
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

        def fail_completion_replace(source, destination, *args, **kwargs):
            if str(destination).endswith(".completion.json"):
                raise OSError("replace failed")
            return real_replace(source, destination, *args, **kwargs)

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

    def pause_completion_replace(source, destination, *args, **kwargs):
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
        return real_replace(source, destination, *args, **kwargs)

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