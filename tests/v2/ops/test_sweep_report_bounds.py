"""Existing native reports are compared within their verified byte boundary."""
import errno
import os
from pathlib import Path

import pytest

from engine.v2.ops.errors import OpsError
from experiments import native_outcomes as outcomes
from tests.v2.ops.native_bounds_support import (
    assert_identity_refusal,
    assert_objects_preserved,
    catalog_rows,
    object_bytes,
    register_native_case,
)
from tests.v2.ops.test_sweep_native_bounds import ACTIONS, CAP, REPORT, forbid, invoke
from tests.v2.research.test_native_registration import source as source


def report_path(source, registration, no_ledger=False):
    mode = "smoke" if no_ledger else "recorded"
    return source.store.root / "native_reports" / registration.run_id / mode / "REPORT.md"


def forbid_report_reads(monkeypatch, path):
    """Allow artifact verification, but reject either reader on the report inode."""
    real_read_bytes, real_read = Path.read_bytes, os.read
    identity = path.stat().st_dev, path.stat().st_ino

    def read_bytes(candidate):
        if candidate == path:
            forbid()
        return real_read_bytes(candidate)

    def read(fd, size):
        metadata = os.fstat(fd)
        if (metadata.st_dev, metadata.st_ino) == identity:
            forbid()
        return real_read(fd, size)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(outcomes.os, "read", read)


@pytest.mark.parametrize("action,completed", [
    (ACTIONS[0], False), (ACTIONS[0], True), (ACTIONS[1], True), (ACTIONS[2], True),
], ids=["fresh-publication", "repeated-publication", "replay", "export"])
@pytest.mark.parametrize("no_ledger", [False, True], ids=["recorded", "smoke"])
@pytest.mark.parametrize("excess", [1, CAP * 32], ids=["one-extra-byte", "large-file"])
def test_oversized_existing_report_refuses_before_reads_or_effects(
        source, tmp_path, monkeypatch, action, completed, no_ledger, excess):
    registration = register_native_case(source)
    ledger = tmp_path / "ledger.csv"
    if completed:
        invoke(ACTIONS[0], source, registration, ledger, no_ledger=no_ledger)
    path = report_path(source, registration, no_ledger)
    path.parent.mkdir(parents=True)
    with path.open("wb") as report:
        report.write(REPORT.encode())
        report.truncate(len(REPORT.encode()) + excess)
    catalog, objects = catalog_rows(source.conn), object_bytes(source.store)
    ledger_bytes = ledger.read_bytes() if ledger.exists() else None
    real_publish, schemas = source.store.publish_bytes, []

    def refusal_only(data, *, schema_ref):
        assert schema_ref == "native_experiment_refusal.v1.0"
        schemas.append(schema_ref)
        return real_publish(data, schema_ref=schema_ref)

    with monkeypatch.context() as patch:
        forbid_report_reads(patch, path)
        patch.setattr(source.store, "publish_bytes", refusal_only)
        patch.setattr(outcomes, "ledger_append", forbid)
        patch.setattr(outcomes, "_reserve", forbid)
        with pytest.raises(OpsError) as captured:
            invoke(action, source, registration, ledger, no_ledger=no_ledger,
                   max_evidence_bytes=CAP)
    assert_identity_refusal(source, captured, registration)
    assert schemas == ["native_experiment_refusal.v1.0"]
    assert catalog_rows(source.conn) == catalog
    assert_objects_preserved(source, objects)
    assert (ledger.read_bytes() if ledger.exists() else None) == ledger_bytes
    assert path.stat().st_size == len(REPORT.encode()) + excess
    assert not source.conn.in_transaction


@pytest.mark.parametrize("collision", ["oversize", "fifo", "symlink"])
@pytest.mark.parametrize("no_ledger", [False, True], ids=["recorded", "smoke"])
def test_late_export_collision_is_refused_without_reading_or_replacing_it(
        source, tmp_path, monkeypatch, collision, no_ledger):
    registration = register_native_case(source)
    ledger = tmp_path / "ledger.csv"
    invoke(ACTIONS[0], source, registration, ledger, no_ledger=no_ledger)
    path = report_path(source, registration, no_ledger)
    foreign = tmp_path / "foreign-report.md"
    foreign.write_text(REPORT)
    catalog, objects = catalog_rows(source.conn), object_bytes(source.store)
    ledger_bytes = ledger.read_bytes() if ledger.exists() else None
    real_link, collisions = os.link, []

    def collide(staged, destination, *args, **kwargs):
        if Path(destination) == path:
            assert not path.exists(), "collision must arrive after report prechecks"
            if collision == "oversize":
                path.write_bytes(REPORT.encode() + b"x")
            elif collision == "fifo":
                os.mkfifo(path)
            else:
                path.symlink_to(foreign)
            forbid_report_reads(patch, path)
            collisions.append(path.lstat())
        return real_link(staged, destination, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(outcomes.os, "link", collide)
        with pytest.raises(OpsError) as captured:
            invoke(ACTIONS[2], source, registration, ledger, no_ledger=no_ledger,
                   max_evidence_bytes=CAP)
    assert_identity_refusal(source, captured, registration)
    assert len(collisions) == 1
    assert path.lstat() == collisions[0]
    assert catalog_rows(source.conn) == catalog
    assert_objects_preserved(source, objects)
    assert (ledger.read_bytes() if ledger.exists() else None) == ledger_bytes
    assert foreign.read_text() == REPORT
    assert not list(path.parent.glob(".REPORT.*"))


@pytest.mark.parametrize("kind", ["oversize", "fifo", "symlink", "directory"])
def test_report_descriptor_rejects_unsafe_objects_before_payload_reads(tmp_path, monkeypatch, kind):
    path = tmp_path / "REPORT.md"
    expected = b"expected"
    if kind == "oversize":
        path.write_bytes(expected + b"x")
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "symlink":
        target = tmp_path / "target.md"
        target.write_bytes(expected)
        path.symlink_to(target)
    else:
        path.mkdir()
        # Isolate S_ISREG: the directory must pass the independent size bound.
        expected = b"x" * max(len(expected), path.stat().st_size)
        assert path.stat().st_size <= len(expected)
    real_open, opened = os.open, []

    def safe_open(candidate, flags):
        assert candidate == path
        assert flags & os.O_NOFOLLOW
        assert flags & os.O_NONBLOCK
        fd = real_open(candidate, flags)
        opened.append(fd)
        return fd

    with monkeypatch.context() as patch:
        patch.setattr(outcomes.os, "open", safe_open)
        patch.setattr(outcomes.os, "read", forbid)
        message = ("report must not be a symbolic link" if kind == "symlink"
                   else "report exceeds its expected regular-file size")
        with pytest.raises(ValueError, match=f"^{message}$"):
            outcomes._report_bytes(path, expected, CAP)
    assert len(opened) == (0 if kind == "symlink" else 1)
    for fd in opened:
        with pytest.raises(OSError) as captured:
            os.fstat(fd)
        assert captured.value.errno == errno.EBADF


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("replacement", ["directory", "symlink"])
def test_report_replacement_after_path_prechecks_preserves_public_r6(
        source, tmp_path, monkeypatch, action, replacement):
    registration = register_native_case(source)
    ledger = tmp_path / "ledger.csv"
    directory = tmp_path / "replacement-directory"
    directory.mkdir()
    report = REPORT + "x" * directory.stat().st_size
    if action != ACTIONS[0]:
        invoke(ACTIONS[0], source, registration, ledger, report=report)
    path = report_path(source, registration)
    path.parent.mkdir(parents=True)
    path.write_text(report)
    original = tmp_path / "original-report.md"
    target = tmp_path / "symlink-target.md"
    target.write_text(report)
    catalog, objects = catalog_rows(source.conn), object_bytes(source.store)
    ledger_bytes = ledger.read_bytes() if ledger.exists() else None
    real_open, real_publish, replacements = os.open, source.store.publish_bytes, []

    def replace_at_open(candidate, *args, **kwargs):
        if Path(candidate) == path:
            assert not replacements
            path.rename(original)
            if replacement == "directory":
                directory.rename(path)
                assert path.stat().st_size <= len(report.encode())
            else:
                path.symlink_to(target)
            replacements.append(path.lstat())
            forbid_report_reads(patch, path)
        return real_open(candidate, *args, **kwargs)

    def refusal_only(data, *, schema_ref):
        assert schema_ref == "native_experiment_refusal.v1.0"
        return real_publish(data, schema_ref=schema_ref)

    options = {"report": report} if action == ACTIONS[0] else {}
    with monkeypatch.context() as patch:
        patch.setattr(outcomes.os, "open", replace_at_open)
        patch.setattr(source.store, "publish_bytes", refusal_only)
        patch.setattr(outcomes, "ledger_append", forbid)
        patch.setattr(outcomes, "_reserve", forbid)
        with pytest.raises(OpsError) as captured:
            invoke(action, source, registration, ledger, max_evidence_bytes=CAP, **options)
    assert_identity_refusal(source, captured, registration)
    assert len(replacements) == 1
    assert path.lstat() == replacements[0]
    assert original.read_text() == target.read_text() == report
    assert catalog_rows(source.conn) == catalog
    assert_objects_preserved(source, objects)
    assert (ledger.read_bytes() if ledger.exists() else None) == ledger_bytes
    assert not source.conn.in_transaction


def test_report_growth_after_descriptor_precheck_reads_only_expected_plus_one(tmp_path, monkeypatch):
    path = tmp_path / "REPORT.md"
    expected = b"x" * (65536 + 7)
    path.write_bytes(expected)
    real_fstat, real_read = os.fstat, os.read
    descriptors, requests, received = [], [], []

    def grow_after_stat(fd):
        metadata = real_fstat(fd)
        descriptors.append(fd)
        with path.open("ab") as report:
            report.write(b"y" * (len(expected) + 1))
        return metadata

    def bounded_read(fd, size):
        remaining = len(expected) + 1 - sum(map(len, received))
        assert 0 < size <= min(65536, remaining)
        requests.append(size)
        # A short first read must not be mistaken for EOF.
        chunk = real_read(fd, min(size, 17) if len(requests) == 1 else size)
        received.append(chunk)
        return chunk

    with monkeypatch.context() as patch:
        patch.setattr(outcomes.os, "fstat", grow_after_stat)
        patch.setattr(outcomes.os, "read", bounded_read)
        with pytest.raises(ValueError, match="^report exceeds its expected regular-file size$"):
            outcomes._report_bytes(path, expected, CAP)
    assert len(descriptors) == 1
    assert len(requests) > 1
    assert sum(map(len, received)) == len(expected) + 1
    assert path.stat().st_size > len(expected) + 1
    with pytest.raises(OSError) as captured:
        os.fstat(descriptors[0])
    assert captured.value.errno == errno.EBADF


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("operation", ["open", "fstat", "read"])
def test_report_io_errors_stay_retryable_without_publication_effects(
        source, tmp_path, monkeypatch, action, operation):
    registration = register_native_case(source)
    ledger = tmp_path / "ledger.csv"
    if action != ACTIONS[0]:
        invoke(ACTIONS[0], source, registration, ledger)
    path = report_path(source, registration)
    path.parent.mkdir(parents=True)
    path.write_text(REPORT)
    identity = path.stat().st_dev, path.stat().st_ino
    catalog, objects = catalog_rows(source.conn), object_bytes(source.store)
    ledger_bytes = ledger.read_bytes() if ledger.exists() else None
    real_operation, real_fstat, failed = getattr(os, operation), os.fstat, []

    def fail_report_only(first, *args, **kwargs):
        if operation == "open":
            matches = Path(first) == path
        else:
            metadata = real_fstat(first)
            matches = (metadata.st_dev, metadata.st_ino) == identity
        if matches:
            failed.append(first)
            raise OSError(errno.EIO, "synthetic report storage interruption")
        return real_operation(first, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(outcomes.os, operation, fail_report_only)
        patch.setattr(outcomes, "_reserve", forbid)
        patch.setattr(outcomes, "ledger_append", forbid)
        patch.setattr(source.store, "publish_bytes", forbid)
        with pytest.raises(OpsError) as captured:
            invoke(action, source, registration, ledger, max_evidence_bytes=CAP)
    assert captured.value.code == "RESOURCE_UNAVAILABLE"
    assert captured.value.problem.retryable is True
    assert len(failed) == 1
    assert catalog_rows(source.conn) == catalog
    assert object_bytes(source.store) == objects
    assert (ledger.read_bytes() if ledger.exists() else None) == ledger_bytes
    assert path.read_text() == REPORT
    if operation != "open":
        with pytest.raises(OSError) as closed:
            os.fstat(failed[0])
        assert closed.value.errno == errno.EBADF


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("options", [{}, {"max_evidence_bytes": None}], ids=["omitted", "explicit-none"])
def test_default_report_comparison_keeps_path_reader(source, tmp_path, monkeypatch, action, options):
    registration = register_native_case(source)
    ledger = tmp_path / "ledger.csv"
    first = invoke(ACTIONS[0], source, registration, ledger)
    path = invoke(ACTIONS[2], source, registration, ledger)
    ledger_bytes = ledger.read_bytes()
    real_read, real_open, reads = Path.read_bytes, os.open, []

    def read_bytes(candidate):
        if candidate == path:
            reads.append(candidate)
        return real_read(candidate)

    def no_descriptor_report(candidate, *args, **kwargs):
        assert Path(candidate) != path, "default report comparisons must keep the original reader"
        return real_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(outcomes.os, "open", no_descriptor_report)
    result = invoke(action, source, registration, ledger, **options)
    assert result == (path if action == ACTIONS[2] else first)
    assert reads
    assert ledger.read_bytes() == ledger_bytes and path.read_text() == REPORT


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("mismatch", ["short", "same-length"])
def test_bounded_report_comparison_refuses_different_content_before_effects(
        source, tmp_path, monkeypatch, action, mismatch):
    registration = register_native_case(source)
    ledger = tmp_path / "ledger.csv"
    if action != ACTIONS[0]:
        invoke(ACTIONS[0], source, registration, ledger)
    path = report_path(source, registration)
    path.parent.mkdir(parents=True)
    data = REPORT.encode()[:-1] if mismatch == "short" else b"!" + REPORT.encode()[1:]
    path.write_bytes(data)
    catalog, objects = catalog_rows(source.conn), object_bytes(source.store)
    ledger_bytes = ledger.read_bytes() if ledger.exists() else None
    real_publish = source.store.publish_bytes

    def refusal_only(payload, *, schema_ref):
        assert schema_ref == "native_experiment_refusal.v1.0"
        return real_publish(payload, schema_ref=schema_ref)

    with monkeypatch.context() as patch:
        patch.setattr(source.store, "publish_bytes", refusal_only)
        patch.setattr(outcomes, "ledger_append", forbid)
        patch.setattr(outcomes, "_reserve", forbid)
        with pytest.raises(OpsError) as captured:
            invoke(action, source, registration, ledger, max_evidence_bytes=CAP)
    assert_identity_refusal(source, captured, registration)
    assert path.read_bytes() == data
    assert catalog_rows(source.conn) == catalog
    assert_objects_preserved(source, objects)
    assert (ledger.read_bytes() if ledger.exists() else None) == ledger_bytes


def test_exact_existing_report_can_exceed_metadata_budget(source, tmp_path):
    registration = register_native_case(source)
    ledger = tmp_path / "ledger.csv"
    report = REPORT + "漢" * CAP
    path = report_path(source, registration)
    path.parent.mkdir(parents=True)
    path.write_text(report)
    first = invoke(ACTIONS[0], source, registration, ledger, report=report, max_evidence_bytes=CAP)
    assert invoke(ACTIONS[1], source, registration, ledger, max_evidence_bytes=CAP) == first
    assert invoke(ACTIONS[2], source, registration, ledger, max_evidence_bytes=CAP) == path
    assert path.read_text() == report
