"""Local immutable backup and isolated restore controls."""
from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from engine.v2.foundation import ArtifactStore, to_document
from engine.v2.ops import backup as backup_module
from engine.v2.ops import catalog as catalog_module
from engine.v2.ops.backup import prepare_backup, restore_backup, run_backup
from engine.v2.ops.catalog import integrity_errors
from engine.v2.ops.effects_graph import backup_effect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.fingerprints import file_hash
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from tests.ops_support import catalog
from tests.v2.ops.test_v2_ops_effects_graph import (
    SESSION,
    _commit,
    _open,
    _params,
    _row,
    _seed_decisions,
    _submit_and_claim,
)


class _InjectedCrash(Exception):
    pass


def _row_counts(conn):
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name")]
    return {name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            for name in tables}


def test_backup_copies_artifacts_and_restores_new_root(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b1", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="b1", owner="worker", target=backup, clock=clock, store=store)
    assert manifest["artifacts"]["evidence.bin"]["content_hash"] == ref.content_hash
    restored = restore_backup(backup, tmp_path / "restored")
    assert (restored / "artifacts" / "evidence.bin").read_bytes() == b"evidence"
    assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == "delivered"
    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")
    assert refused.value.code == "INVALID_REQUEST"
    assert refused.value.problem.retryable is False


@pytest.mark.parametrize("stale_bytes", [b"", b"SQLite format 3\x00"],
                         ids=["zero-byte", "truncated-header"])
def test_backup_publishes_verified_database_beside_unreferenced_stale_path(
        tmp_path, stale_bytes):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b7", {}, clock=clock)
    backup = tmp_path / "backup"
    backup.mkdir()
    stale = backup / "b7.sqlite"
    stale.write_bytes(stale_bytes)

    manifest = run_backup(
        conn, key="b7", owner="worker", target=backup, clock=clock, store=store)

    # The stale key-named path is never the published database: the manifest
    # selects a fresh immutable file, whatever remains at <key>.sqlite.
    assert manifest["database_file"] != "b7.sqlite"
    assert stale.read_bytes() == stale_bytes
    published = backup / manifest["database_file"]
    assert file_hash(published) == manifest["database"]
    assert manifest["database_content_hash"].startswith("sha256:")
    rebuilt = sqlite3.connect(published)
    try:
        assert integrity_errors(rebuilt) == []
        assert _row_counts(rebuilt) == _row_counts(conn)
    finally:
        rebuilt.close()
    assert manifest["database_rows"] == _row_counts(conn)
    assert conn.execute(
        "SELECT state FROM outbox WHERE logical_key='b7'").fetchone()[0] == "delivered"


def test_staged_row_count_mismatch_is_refused_with_table_details(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b8", {}, clock=clock)
    real_backup_to = backup_module.backup_to

    def corrupt(conn, destination, *, progress=None):
        real_backup_to(conn, destination, progress=progress)
        damaged = sqlite3.connect(destination)
        try:
            damaged.execute("DELETE FROM outbox")
            damaged.commit()
        finally:
            damaged.close()

    monkeypatch.setattr(backup_module, "backup_to", corrupt)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b8", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)
    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert refused.value.problem.details["tables"] == ["outbox"]


def test_lost_ack_retry_reuses_prior_database(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b9", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    calls = []
    real_backup_to = backup_module.backup_to

    def counted(conn, destination, *, progress=None):
        calls.append(destination)
        real_backup_to(conn, destination, progress=progress)

    monkeypatch.setattr(backup_module, "backup_to", counted)

    def crash(point):
        if point == "after_manifest_before_ack":
            raise _InjectedCrash("ack lost")

    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="b9", owner="worker", target=backup, clock=clock,
                   store=store, fault=crash)
    assert len(calls) == 1
    clock.advance(301)
    run_backup(conn, key="b9", owner="retry", target=backup, clock=clock, store=store)
    assert len(calls) == 1


def test_lost_ack_retry_rebuilds_when_source_changed(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b10", {}, clock=clock)
    conn.execute("CREATE TABLE synthetic_change (value INTEGER)")
    conn.commit()
    backup = tmp_path / "backup"
    calls = []
    real_backup_to = backup_module.backup_to

    def counted(conn, destination, *, progress=None):
        calls.append(destination)
        real_backup_to(conn, destination, progress=progress)

    monkeypatch.setattr(backup_module, "backup_to", counted)

    def crash(point):
        if point == "after_manifest_before_ack":
            raise _InjectedCrash("ack lost")

    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="b10", owner="worker", target=backup, clock=clock,
                   store=store, fault=crash)
    first = json.loads((backup / "b10.manifest.json").read_text())
    assert len(calls) == 1

    conn.execute("INSERT INTO synthetic_change VALUES (1)")
    conn.commit()
    clock.advance(301)
    second = run_backup(conn, key="b10", owner="retry", target=backup, clock=clock, store=store)

    assert len(calls) == 2
    assert second["database"] != first["database"]
    assert second["database_rows"]["synthetic_change"] == 1


def test_backup_counts_and_copy_share_one_snapshot(tmp_path, monkeypatch):
    """A writer that commits after the source counts are read must not leak
    its row into the backup: counts and ``backup_to`` use one read snapshot."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    conn.execute("CREATE TABLE synthetic_change (value INTEGER)")
    conn.commit()
    prepare_backup(conn, "b13", {}, clock=clock)
    before = _row_counts(conn)
    database = tmp_path / "ops.sqlite"
    real_backup_to = backup_module.backup_to
    observed = []

    def racing_backup_to(source, destination, *, progress=None):
        writer = sqlite3.connect(database)
        try:
            writer.execute("INSERT INTO synthetic_change VALUES (1)")
            writer.commit()
        finally:
            writer.close()
        observed.append(_row_counts(source))
        real_backup_to(source, destination, progress=progress)

    monkeypatch.setattr(backup_module, "backup_to", racing_backup_to)
    manifest = run_backup(conn, key="b13", owner="worker", target=tmp_path / "backup",
                          clock=clock, store=store)

    assert observed == [before]
    assert _row_counts(conn)["synthetic_change"] == 1
    backup = sqlite3.connect(tmp_path / "backup" / manifest["database_file"])
    try:
        assert _row_counts(backup) == before
        assert backup.execute("SELECT COUNT(*) FROM synthetic_change").fetchone()[0] == 0
    finally:
        backup.close()
    assert manifest["database_rows"] == before


def test_oversized_prior_manifest_is_ignored_and_rebuilt(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b11", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    calls = []
    real_backup_to = backup_module.backup_to

    def counted(conn, destination, *, progress=None):
        calls.append(destination)
        real_backup_to(conn, destination, progress=progress)

    monkeypatch.setattr(backup_module, "backup_to", counted)

    def crash(point):
        if point == "after_manifest_before_ack":
            raise _InjectedCrash("ack lost")

    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="b11", owner="worker", target=backup, clock=clock,
                   store=store, fault=crash)
    manifest_path = backup / "b11.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert len(calls) == 1

    manifest["artifacts"]["evidence.bin"]["padding"] = (
        "x" * (backup_module.MAX_MANIFEST_BYTES + 1))
    # The manifest is still shape-valid: only the byte cap distinguishes it.
    backup_module._validated_manifest_shape(manifest, current=True)
    manifest_path.write_text(json.dumps(manifest))
    assert manifest_path.stat().st_size > backup_module.MAX_MANIFEST_BYTES

    clock.advance(301)
    rebuilt = run_backup(conn, key="b11", owner="retry", target=backup, clock=clock,
                         store=store)

    assert len(calls) == 2
    assert manifest_path.stat().st_size <= backup_module.MAX_MANIFEST_BYTES
    assert json.loads(manifest_path.read_text()) == rebuilt
    assert rebuilt["backup_key"] == "b11"
    assert "padding" not in rebuilt["artifacts"]["evidence.bin"]


def test_prior_manifest_with_extra_field_is_invalid_and_rebuilt(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b14", {}, clock=clock)
    backup = tmp_path / "backup"
    calls = []
    real_backup_to = backup_module.backup_to

    def counted(conn, destination, *, progress=None):
        calls.append(destination)
        real_backup_to(conn, destination, progress=progress)

    monkeypatch.setattr(backup_module, "backup_to", counted)

    def crash(point):
        if point == "after_manifest_before_ack":
            raise _InjectedCrash("ack lost")

    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="b14", owner="worker", target=backup, clock=clock,
                   store=store, fault=crash)
    manifest_path = backup / "b14.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["irrelevant"] = "field"
    manifest_path.write_text(json.dumps(manifest))

    clock.advance(301)
    rebuilt = run_backup(conn, key="b14", owner="retry", target=backup, clock=clock,
                         store=store)

    assert len(calls) == 2
    assert "irrelevant" not in rebuilt
    assert json.loads(manifest_path.read_text()) == rebuilt


def test_retry_removes_only_matching_crash_staging_files(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b12", {}, clock=clock)
    backup = tmp_path / "backup"
    backup.mkdir()
    stale = backup / f"{backup_module._stage_prefix('b12')}crash.staging.sqlite"
    other = backup / f"{backup_module._stage_prefix('b13')}crash.staging.sqlite"
    stale.write_bytes(b"junk")
    other.write_bytes(b"other")

    run_backup(conn, key="b12", owner="worker", target=backup, clock=clock, store=store)

    assert not stale.exists()
    assert other.exists()


def test_backup_copy_failure_remains_retryable(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b2", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    old_destination = backup / "artifacts" / "evidence.bin"
    old_destination.parent.mkdir(parents=True)
    old_destination.write_bytes(b"old evidence")
    def crash(_name):
        raise _InjectedCrash("copy interrupted")
    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="b2", owner="worker", target=backup, clock=clock,
                   store=store, fault=crash)
    assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == "running"
    assert old_destination.read_bytes() == b"old evidence"
    assert list(old_destination.parent.glob(".evidence.bin.*.pending")) == []
    clock.advance(301)
    run_backup(conn, key="b2", owner="retry", target=backup, clock=clock, store=store)
    assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == "delivered"


def test_manifest_success_before_ack_is_idempotent_and_keys_are_safe(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    with pytest.raises(OpsError) as refused:
        prepare_backup(conn, "../escape", {"evidence.bin": ref}, clock=clock)
    assert refused.value.code == "INVALID_REQUEST"
    assert refused.value.problem.retryable is False
    prepare_backup(conn, "b3", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    def crash(point):
        if point == "after_manifest_before_ack":
            raise _InjectedCrash("ack lost")
    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="b3", owner="worker", target=backup, clock=clock, store=store,
                   fault=crash)
    first = (backup / "b3.manifest.json").read_bytes()
    clock.advance(301)
    second = run_backup(conn, key="b3", owner="retry", target=backup, clock=clock, store=store)
    assert (backup / "b3.manifest.json").read_bytes() == first
    assert second["cutoff"] == "2026-09-12T00:00:00.000000Z"
    assert conn.execute("SELECT state FROM outbox WHERE logical_key='b3'").fetchone()[0] == "delivered"


def test_prepare_backup_retry_returns_same_effect_without_rebuilding_payload(tmp_path):
    """P2-C07: ``prepare_backup`` called again for a key that already has a
    pending/running ``backup`` effect must return that SAME effect, not
    build a new payload -- a new payload (fresh ``cutoff``) for the same
    logical key would make ``outbox.enqueue`` raise ``IDEMPOTENCY_CONFLICT``."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    first_id = prepare_backup(conn, "b6", {"evidence.bin": ref}, clock=clock)
    original = conn.execute("SELECT payload_json FROM outbox WHERE effect_id=?", (first_id,)).fetchone()[0]

    clock.advance(3600)
    second_id = prepare_backup(conn, "b6", {"evidence.bin": ref}, clock=clock)
    assert second_id == first_id
    assert conn.execute("SELECT payload_json FROM outbox WHERE effect_id=?",
                        (first_id,)).fetchone()[0] == original
    assert conn.execute("SELECT COUNT(*) FROM outbox WHERE kind='backup' AND logical_key='b6'"
                        ).fetchone()[0] == 1


def test_staged_content_mismatch_is_refused_without_row_count_change(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    conn.execute("CREATE TABLE synthetic_content (value TEXT)")
    conn.execute("INSERT INTO synthetic_content VALUES ('original')")
    conn.commit()
    prepare_backup(conn, "b15", {}, clock=clock)
    real_backup_to = backup_module.backup_to

    def corrupt_content(source, destination, *, progress=None):
        real_backup_to(source, destination, progress=progress)
        damaged = sqlite3.connect(destination)
        try:
            damaged.execute("UPDATE synthetic_content SET value='tampered'")
            damaged.commit()
        finally:
            damaged.close()

    monkeypatch.setattr(backup_module, "backup_to", corrupt_content)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b15", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert refused.value.problem.details["reason"] == "content_digest"


def test_same_row_count_altered_prior_with_updated_file_hash_is_rebuilt(
        tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    conn.execute("CREATE TABLE synthetic_content (value TEXT)")
    conn.execute("INSERT INTO synthetic_content VALUES ('original')")
    conn.commit()
    prepare_backup(conn, "b16", {}, clock=clock)
    backup = tmp_path / "backup"
    calls = []
    real_backup_to = backup_module.backup_to

    def counted(conn, destination, *, progress=None):
        calls.append(destination)
        real_backup_to(conn, destination, progress=progress)

    monkeypatch.setattr(backup_module, "backup_to", counted)

    def crash(point):
        if point == "after_manifest_before_ack":
            raise _InjectedCrash("ack lost")

    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="b16", owner="worker", target=backup, clock=clock,
                   store=store, fault=crash)
    first = json.loads((backup / "b16.manifest.json").read_text())
    assert len(calls) == 1

    prior_database = backup / first["database_file"]
    altered = sqlite3.connect(prior_database)
    try:
        altered.execute("UPDATE synthetic_content SET value='altered'")
        altered.commit()
    finally:
        altered.close()
    manifest = json.loads((backup / "b16.manifest.json").read_text())
    manifest["database"] = file_hash(prior_database)
    (backup / "b16.manifest.json").write_text(json.dumps(manifest))

    clock.advance(301)
    rebuilt = run_backup(conn, key="b16", owner="retry", target=backup,
                         clock=clock, store=store)

    assert len(calls) == 2
    assert rebuilt["database_file"] != first["database_file"]
    assert rebuilt["database_content_hash"] != first["database_content_hash"]


@pytest.mark.parametrize("corruption", ["integrity", "file-hash"])
def test_corrupt_prior_database_falls_back_to_rebuild(tmp_path, monkeypatch, corruption):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b17", {}, clock=clock)
    backup = tmp_path / "backup"
    calls = []
    real_backup_to = backup_module.backup_to

    def counted(conn, destination, *, progress=None):
        calls.append(destination)
        real_backup_to(conn, destination, progress=progress)

    monkeypatch.setattr(backup_module, "backup_to", counted)

    def crash(point):
        if point == "after_manifest_before_ack":
            raise _InjectedCrash("ack lost")

    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="b17", owner="worker", target=backup, clock=clock,
                   store=store, fault=crash)
    first = json.loads((backup / "b17.manifest.json").read_text())
    prior_database = backup / first["database_file"]
    manifest_path = backup / "b17.manifest.json"
    if corruption == "integrity":
        prior_database.write_bytes(b"not a sqlite database")
        first["database"] = file_hash(prior_database)
    else:
        first["database"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(first))

    clock.advance(301)
    rebuilt = run_backup(conn, key="b17", owner="retry", target=backup,
                         clock=clock, store=store)

    assert len(calls) == 2
    assert rebuilt["database_file"] != first["database_file"]


def test_structurally_valid_prior_with_different_artifact_metadata_is_refused(
        tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b18", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"

    def crash(point):
        if point == "after_manifest_before_ack":
            raise _InjectedCrash("ack lost")

    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="b18", owner="worker", target=backup, clock=clock,
                   store=store, fault=crash)
    manifest_path = backup / "b18.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["evidence.bin"]["content_hash"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    clock.advance(301)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b18", owner="retry", target=backup, clock=clock, store=store)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False


def test_artifact_hash_mismatch_refuses_non_retryable_integrity_failure(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"good bytes", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b19", {"evidence.bin": ref}, clock=clock)
    wrong = tmp_path / "wrong.bin"
    wrong.write_bytes(b"different bytes")
    monkeypatch.setattr(store, "verify", lambda _ref: wrong)

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b19", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False


def test_deeply_nested_prior_manifest_is_invalid_and_rebuilt(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b20", {}, clock=clock)
    backup = tmp_path / "backup"
    calls = []
    real_backup_to = backup_module.backup_to

    def counted(conn, destination, *, progress=None):
        calls.append(destination)
        real_backup_to(conn, destination, progress=progress)

    monkeypatch.setattr(backup_module, "backup_to", counted)

    def crash(point):
        if point == "after_manifest_before_ack":
            raise _InjectedCrash("ack lost")

    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="b20", owner="worker", target=backup, clock=clock,
                   store=store, fault=crash)
    (backup / "b20.manifest.json").write_text("[" * 100_000)

    clock.advance(301)
    rebuilt = run_backup(conn, key="b20", owner="retry", target=backup,
                         clock=clock, store=store)

    assert len(calls) == 2
    assert rebuilt["backup_key"] == "b20"


def test_unsafe_artifact_name_is_refused_before_database_install(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b21", {"../escape": ref}, clock=clock)
    backup = tmp_path / "backup"

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b21", owner="worker", target=backup, clock=clock, store=store)

    assert refused.value.code == "INVALID_REQUEST"
    assert refused.value.problem.retryable is False
    assert not list(backup.glob("*.sqlite"))


def test_per_key_lock_serializes_two_threads(tmp_path):
    target = tmp_path / "backup"
    target.mkdir()
    first_entered = threading.Event()
    release = threading.Event()
    order = []

    def first():
        with backup_module._backup_lock(target, "same-key"):
            order.append("first-enter")
            first_entered.set()
            release.wait(5)
            order.append("first-exit")

    def second():
        first_entered.wait(5)
        with backup_module._backup_lock(target, "same-key"):
            order.append("second-enter")

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert first_entered.wait(5)
    second_thread.start()
    assert second_thread.is_alive()
    assert "second-enter" not in order
    release.set()
    first_thread.join(5)
    second_thread.join(5)

    assert order == ["first-enter", "first-exit", "second-enter"]


def test_different_key_locks_are_independent(tmp_path):
    target = tmp_path / "backup"
    target.mkdir()
    entered = threading.Event()

    def other():
        with backup_module._backup_lock(target, "other-key"):
            entered.set()

    with backup_module._backup_lock(target, "one-key"):
        thread = threading.Thread(target=other)
        thread.start()
        assert entered.wait(5)
        thread.join(5)


def test_restore_reads_manifest_within_the_existing_bound(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "cap", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="cap", owner="worker", target=backup,
                          clock=clock, store=store)
    (backup / manifest["database_file"]).rename(backup / "cap.sqlite")
    legacy = {name: manifest[name] for name in
              ("schema_version", "backup_key", "cutoff", "database", "artifacts")}
    legacy["artifacts"]["evidence.bin"]["padding"] = (
        "x" * (backup_module.MAX_MANIFEST_BYTES + 1))
    # The manifest is still shape-valid: only the byte cap distinguishes it.
    backup_module._validated_manifest_shape(legacy, current=False)
    (backup / "cap.manifest.json").write_text(json.dumps(legacy))
    assert (backup / "cap.manifest.json").stat().st_size > \
        backup_module.MAX_MANIFEST_BYTES
    destination = tmp_path / "restored"

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, destination)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert not destination.exists()


def test_restore_catches_deeply_nested_manifest_json(tmp_path):
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "nested.manifest.json").write_text("[" * 100_000)

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False


def test_restore_legacy_manifest_falls_back_to_key_named_database(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "legacy", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="legacy", owner="worker", target=backup,
                          clock=clock, store=store)
    source_rows = _row_counts(conn)
    (backup / manifest["database_file"]).rename(backup / "legacy.sqlite")
    legacy = {name: manifest[name] for name in
              ("schema_version", "backup_key", "cutoff", "database", "artifacts")}
    (backup / "legacy.manifest.json").write_text(json.dumps(legacy))

    restored = restore_backup(backup, tmp_path / "restored")

    on_disk = json.loads((backup / "legacy.manifest.json").read_text())
    assert set(on_disk) == {"schema_version", "backup_key", "cutoff", "database",
                            "artifacts"}
    assert (restored / "ops.sqlite").is_file()
    restored_db = sqlite3.connect(restored / "ops.sqlite")
    try:
        assert integrity_errors(restored_db) == []
        assert _row_counts(restored_db) == source_rows
    finally:
        restored_db.close()
    assert (restored / "artifacts" / "evidence.bin").read_bytes() == b"evidence"


def test_restore_rejects_unsafe_database_filename(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "unsafe", {}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="unsafe", owner="worker", target=backup,
                          clock=clock, store=store)
    manifest["database_file"] = "../escape.sqlite"
    (backup / "unsafe.manifest.json").write_text(json.dumps(manifest))
    destination = tmp_path / "restored"

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, destination)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert not destination.exists()


@pytest.mark.parametrize("referenced", ["database", "artifact"])
@pytest.mark.parametrize("damage", ["missing", "symlink"])
def test_restore_missing_or_symlinked_source_is_an_integrity_failure(
        tmp_path, referenced, damage):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "restore", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="restore", owner="worker", target=backup,
                          clock=clock, store=store)
    source = (backup / manifest["database_file"] if referenced == "database"
              else backup / "artifacts" / "evidence.bin")
    if damage == "missing":
        source.unlink()
    else:
        replacement = tmp_path / "replacement.bin"
        replacement.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(replacement)

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False


def test_catalog_backup_to_reports_page_progress(tmp_path):
    conn, _, _ = catalog(tmp_path)
    destination = tmp_path / "copy.sqlite"
    seen = []

    backup_module.backup_to(conn, destination, progress=lambda *args: seen.append(args))

    assert destination.is_file()
    assert seen
    _status, remaining, total = seen[-1]
    assert remaining == 0
    assert total > 0


def test_backup_rejects_symlink_target_before_writing(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b4", {"evidence.bin": ref}, clock=clock)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = tmp_path / "backup"
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b4", owner="worker", target=target, clock=clock, store=store)
    assert refused.value.code == "INVALID_REQUEST"
    assert refused.value.problem.retryable is False
    assert not list(outside.glob("b4.*.sqlite"))
    assert not (outside / "b4.sqlite").exists()


@pytest.mark.parametrize("key", ["../escape", "/absolute", "a/b", "", ".", "a\\b"])
def test_run_backup_rejects_unsafe_key_before_creating_target(tmp_path, key):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    target = tmp_path / "backup"

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key=key, owner="worker", target=target, clock=clock,
                   store=store)

    assert refused.value.code == "INVALID_REQUEST"
    assert refused.value.problem.retryable is False
    assert not target.exists()


def test_run_backup_retry_after_the_effect_already_delivered_succeeds(tmp_path):
    """#98: a retry after a crash between run_backup's own complete() (state=delivered)
    and the caller's later watermark/attempt-commit step must succeed as an idempotent
    replay, not fail permanently with STALE_EXPECTATION."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b4", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    first = run_backup(conn, key="b4", owner="worker", target=backup, clock=clock, store=store)
    assert conn.execute("SELECT state FROM outbox WHERE logical_key='b4'").fetchone()[0] == "delivered"
    manifest_before_retry = (backup / "b4.manifest.json").read_bytes()

    # Simulate the retry that happens when the caller crashed after run_backup's own
    # complete() but before it wrote the watermark/committed the attempt: run_backup is
    # invoked again with the SAME key, while the outbox row is already delivered.
    second = run_backup(conn, key="b4", owner="worker", target=backup, clock=clock, store=store)
    assert second == first
    assert conn.execute("SELECT state FROM outbox WHERE logical_key='b4'").fetchone()[0] == "delivered"
    # No second backup was performed: the manifest file on disk is unchanged.
    assert (backup / "b4.manifest.json").read_bytes() == manifest_before_retry


def test_backup_coordinator_retry_after_delivery_before_commit_attempt_succeeds(tmp_path,
                                                                                 monkeypatch):
    """#98 at the COORDINATOR level: the fault here is injected after
    ``backup_effect``'s own ``run_backup`` delivered the effect (outbox row
    ``delivered``, manifest durable) but before the attempt's own
    ``commit_attempt`` runs — the exact boundary ``backup.py``'s
    ``_delivered_or_stale`` replay branch targets. The call-level test above
    only simulates the crash at the ``run_backup`` call site, and the
    existing coordinator tests only inject faults INSIDE ``backup_effect``
    (before it returns); neither exercises this window, which the nightly
    job-retry path traverses for real. The retry re-attempts the SAME job
    through the scheduler's own retry machinery (a retryable ``commit_attempt``
    failure settles the crashed attempt and moves the job to ``retry_wait``;
    a fresh claim after the backoff delay starts attempt 2), and must
    complete with exactly ONE delivered backup effect, the manifest
    unchanged, the backup watermark written, the second attempt committed
    and the first attempt settled as failed -- nothing left dangling."""
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        _seed_decisions(conn, clock, scope, SESSION, predictions=[_row("evt-1", "pred")])

        first = _submit_and_claim(conn, clock, supervisor, kind="backup", key="backup-1",
                                  parameters=_params("backup", SESSION, scope))
        result = backup_effect(conn, store, first, root, clock=clock)
        assert result == (None, ())
        key = conn.execute("SELECT logical_key FROM outbox WHERE kind='backup'").fetchone()[0]
        assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == \
            "delivered"
        manifest_path = root / "backups" / scope / (key + ".manifest.json")
        manifest_before = manifest_path.read_bytes()

        # The fault: this attempt's commit_attempt never completes — the
        # coordinator dies right after backup_effect returned, leaving the
        # delivered effect durably DONE but the attempt uncommitted.
        def crash(*args, **kwargs):
            raise _InjectedCrash("crash before commit_attempt")

        monkeypatch.setattr("tests.v2.ops.test_v2_ops_effects_graph.commit_attempt", crash)
        with pytest.raises(_InjectedCrash, match="crash before commit_attempt"):
            _commit(conn, clock, first, result)
        monkeypatch.undo()
        assert conn.execute("SELECT state FROM attempts WHERE attempt_id=?",
                            (first.attempt_id,)).fetchone()[0] not in ("succeeded", "failed")

        # Retry the SAME job, through the mechanism this suite's own
        # retry-after-failure tests use (test_v2_ops_stages_core_kinds.py's
        # ``_fail_once``): ``commit_attempt`` with a retryable failure settles
        # the crashed attempt as failed and moves the job to ``retry_wait``;
        # once the backup kind's first backoff step (bounded, (5, 30)) has
        # elapsed, a fresh claim of the same idempotency key starts attempt 2
        # on the same job_id. Before #98 that second attempt's run_backup
        # found only the delivered row and failed permanently with
        # STALE_EXPECTATION.
        state = commit_attempt(
            conn, first.attempt_id, first.fence,
            Outcome(succeeded=False, process_state="verified_dead", exit_code=1,
                    failure=fail("WORKER_FAILED", "coordinator died before commit").problem),
            clock=clock)
        assert state == "retry_wait"
        clock.advance(5)

        second = _submit_and_claim(conn, clock, supervisor, kind="backup", key="backup-1",
                                   parameters=_params("backup", SESSION, scope))
        assert second.job_id == first.job_id
        assert second.attempt_number == 2
        retried = backup_effect(conn, store, second, root, clock=clock)
        assert retried == (None, ())
        _commit(conn, clock, second, retried)

        # Exactly one delivered effect — no duplicate backup ran.
        rows = conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchall()
        assert [r["state"] for r in rows] == ["delivered"]
        # The manifest on disk is the FIRST attempt's, byte for byte.
        assert manifest_path.read_bytes() == manifest_before
        backup_wm = conn.execute("SELECT occurrence FROM watermarks WHERE pipeline='nightly' "
                                 "AND scope=? AND stage='backup'", (scope,)).fetchone()
        assert backup_wm["occurrence"] == SESSION
        # The retried attempt is ultimately committed successfully...
        assert conn.execute("SELECT state FROM attempts WHERE attempt_id=?",
                            (second.attempt_id,)).fetchone()[0] == "succeeded"
        assert conn.execute("SELECT state FROM jobs WHERE job_id=?",
                            (second.job_id,)).fetchone()[0] == "succeeded"
        # ...and the superseded first attempt is settled, not left dangling
        # (same assertion the lease-recovery retry test makes for the
        # attempt it supersedes, test_v2_ops_coordinator_lease.py).
        crashed = conn.execute("SELECT state, process_state FROM attempts WHERE attempt_id=?",
                               (first.attempt_id,)).fetchone()
        assert (crashed["state"], crashed["process_state"]) == ("failed", "verified_dead")
    finally:
        conn.close()


def test_manifest_database_path_uses_current_name_or_legacy_fallback(tmp_path):
    assert backup_module.manifest_database_path(
        tmp_path, {"backup_key": "b1", "database_file": "b1.abc.sqlite"}) == \
        tmp_path / "b1.abc.sqlite"
    assert backup_module.manifest_database_path(
        tmp_path, {"backup_key": "b1"}) == tmp_path / "b1.sqlite"


def test_single_filename_rejects_unsafe_names(tmp_path):
    assert backup_module._single_filename("plain.sqlite") == "plain.sqlite"
    for name in ("../escape", "/absolute", "a/b", "", ".", "a\\b"):
        with pytest.raises(ValueError):
            backup_module._single_filename(name)
        with pytest.raises(ValueError):
            backup_module.manifest_database_path(
                tmp_path, {"backup_key": "safe", "database_file": name})
        with pytest.raises(ValueError):
            backup_module.manifest_database_path(tmp_path, {"backup_key": name})


def test_file_backed_snapshot_uses_a_separate_read_only_connection(tmp_path):
    conn, _, _ = catalog(tmp_path)
    try:
        with backup_module._snapshot(conn) as (snapshot, separate):
            assert separate is True
            assert snapshot is not conn
            with pytest.raises(sqlite3.OperationalError):
                snapshot.execute("CREATE TABLE snapshot_probe (value INTEGER)")
            assert snapshot.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0
            conn.execute("CREATE TABLE snapshot_probe (value INTEGER)")
            conn.commit()
        assert not conn.in_transaction
    finally:
        conn.close()


def test_in_memory_snapshot_reuses_the_connection_without_keepalive():
    conn = sqlite3.connect(":memory:")
    try:
        with backup_module._snapshot(conn) as (snapshot, separate):
            assert separate is False
            assert snapshot is conn
            assert snapshot.execute("SELECT 1").fetchone()[0] == 1
        assert not conn.in_transaction
    finally:
        conn.close()


def test_expected_dump_lines_includes_schema_statements(tmp_path):
    conn, _, _ = catalog(tmp_path)
    try:
        baseline = backup_module._schema_statement_count(conn)
        conn.execute("CREATE TABLE synthetic_indexed (value INTEGER)")
        conn.execute("CREATE INDEX synthetic_indexed_value ON synthetic_indexed(value)")
        conn.commit()
        rows = _row_counts(conn)
        statements = backup_module._schema_statement_count(conn)
        assert statements == baseline + 1
        assert backup_module._expected_dump_lines(
            rows, statements, backup_module._sqlite_sequence_lines(conn)) == len(
            list(conn.iterdump()))
    finally:
        conn.close()


def test_full_logical_digest_changes_on_schema_change():
    conn = sqlite3.connect(":memory:")
    try:
        before = backup_module._full_logical_digest(conn, stage="schema-test")
        conn.execute("CREATE TABLE schema_probe (value TEXT)")
        after = backup_module._full_logical_digest(conn, stage="schema-test")
    finally:
        conn.close()
    assert after != before


def test_restore_refuses_database_over_the_byte_cap(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "cap", {}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="cap", owner="worker", target=backup,
                          clock=clock, store=store)
    size = (backup / manifest["database_file"]).stat().st_size
    monkeypatch.setattr(backup_module, "MAX_BACKUP_FILE_BYTES", size - 1)
    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")
    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False


def test_restore_refuses_a_published_database_size_mismatch(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "size", {}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="size", owner="worker", target=backup,
                          clock=clock, store=store)
    manifest["database_size"] += 1
    (backup / "size.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")
    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False


def test_fresh_copy_outbox_state_tamper_is_refused_as_content_digest(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b22", {}, clock=clock)
    real_backup_to = backup_module.backup_to

    def tamper(source, destination, *, progress=None):
        real_backup_to(source, destination, progress=progress)
        damaged = sqlite3.connect(destination)
        try:
            damaged.execute("UPDATE outbox SET attempts=attempts+1")
            damaged.commit()
        finally:
            damaged.close()

    monkeypatch.setattr(backup_module, "backup_to", tamper)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b22", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert refused.value.problem.details["reason"] == "content_digest"


def test_malformed_prior_artifact_metadata_is_rebuilt(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b23", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    calls = []
    real_backup_to = backup_module.backup_to

    def counted(conn, destination, *, progress=None):
        calls.append(destination)
        real_backup_to(conn, destination, progress=progress)

    monkeypatch.setattr(backup_module, "backup_to", counted)

    def crash(point):
        if point == "after_manifest_before_ack":
            raise _InjectedCrash("ack lost")

    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="b23", owner="worker", target=backup, clock=clock,
                   store=store, fault=crash)
    manifest_path = backup / "b23.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["evidence.bin"]["content_hash"] = 123
    manifest_path.write_text(json.dumps(manifest))

    clock.advance(301)
    rebuilt = run_backup(conn, key="b23", owner="retry", target=backup,
                         clock=clock, store=store)

    assert len(calls) == 2
    assert rebuilt["backup_key"] == "b23"
    assert isinstance(
        json.loads(manifest_path.read_text())["artifacts"]["evidence.bin"]["content_hash"], str)


@pytest.mark.parametrize("mutation", [
    {"artifact_id": 123},
    {"storage_key": None},
    {"unexpected": "field"},
], ids=["wrong-type", "null-storage-key", "unknown-field"])
def test_malformed_payload_artifact_metadata_is_refused_before_backup(
        tmp_path, mutation):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    malformed = {**to_document(ref), **mutation}
    prepare_backup(conn, "bad-ref", {"evidence.bin": malformed}, clock=clock)
    backup = tmp_path / "backup"

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="bad-ref", owner="worker", target=backup,
                   clock=clock, store=store)

    assert refused.value.code == "INVALID_REQUEST"
    assert refused.value.problem.retryable is False
    assert not list(backup.glob("*.sqlite"))
    assert not list(backup.glob("*.manifest.json"))


def test_restore_refuses_a_published_database_hash_mismatch(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "hash", {}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="hash", owner="worker", target=backup,
                          clock=clock, store=store)
    published = backup / manifest["database_file"]
    data = bytearray(published.read_bytes())
    data[-1] ^= 0xFF
    published.write_bytes(data)

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert "mismatch" in str(refused.value)
    assert not (tmp_path / "restored" / "ops.sqlite").exists()


def test_sparse_file_over_the_byte_cap_is_refused_without_a_destination(
        tmp_path, monkeypatch):
    monkeypatch.setattr(backup_module, "MAX_BACKUP_FILE_BYTES", 1024)
    sparse = tmp_path / "sparse.bin"
    with sparse.open("wb") as stream:
        stream.truncate(1025)
    destination = tmp_path / "copy.bin"

    for call in (
        lambda: backup_module._hash_file(sparse, stage="sparse hash"),
        lambda: backup_module._stream_copy(sparse, destination, stage="sparse copy"),
    ):
        with pytest.raises(OpsError) as refused:
            call()
        assert refused.value.code == "INTEGRITY_FAILED"
        assert refused.value.problem.retryable is False

    assert not destination.exists()


def test_file_backed_backup_renews_keepalive_during_the_copy(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b25", {}, clock=clock)
    renewals = []
    in_backup = {"active": False}
    returned = {"value": False}
    real_backup_to = backup_module.backup_to

    def wrapped(source, destination, *, progress=None):
        def marked(*args):
            in_backup["active"] = True
            try:
                progress(*args)
            finally:
                in_backup["active"] = False

        real_backup_to(source, destination, progress=marked)
        returned["value"] = True

    monkeypatch.setattr(backup_module, "backup_to", wrapped)
    monkeypatch.setattr(backup_module, "_HEARTBEAT_SECONDS", 0)
    # Force many bounded page batches so more than one in-copy renewal occurs.
    monkeypatch.setattr(catalog_module, "_BACKUP_PAGE_BATCH", 1)

    run_backup(conn, key="b25", owner="worker", target=tmp_path / "backup",
               clock=clock, store=store,
               keepalive=lambda: renewals.append(
                   (in_backup["active"], returned["value"])))

    in_copy = [active for active, done in renewals if active and not done]
    assert len(in_copy) > 1


def test_table_row_counts_keepalive_fires_inside_one_long_count(monkeypatch):
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE synthetic_rows (value INTEGER)")
        conn.executemany("INSERT INTO synthetic_rows VALUES (?)",
                         ((i,) for i in range(5000)))
        conn.commit()
        monkeypatch.setattr(backup_module, "_HEARTBEAT_SECONDS", 0)
        calls = []

        counts = backup_module._table_row_counts(
            conn, keepalive=lambda: calls.append(1))

        assert counts["synthetic_rows"] == 5000
        assert len(calls) > 1
    finally:
        conn.close()


def test_superseded_uuid_database_is_removed_after_the_new_manifest(
        tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    conn.execute("CREATE TABLE synthetic_change (value INTEGER)")
    conn.commit()
    prepare_backup(conn, "b26", {}, clock=clock)
    backup = tmp_path / "backup"

    def crash(point):
        if point == "after_manifest_before_ack":
            raise _InjectedCrash("ack lost")

    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="b26", owner="worker", target=backup,
                   clock=clock, store=store, fault=crash)
    first = json.loads((backup / "b26.manifest.json").read_text())
    first_path = backup / first["database_file"]
    assert first_path.exists()

    conn.execute("INSERT INTO synthetic_change VALUES (1)")
    conn.commit()
    clock.advance(301)
    stale = backup / f"b26.{'0' * 32}.sqlite"
    stale.write_bytes(b"stale")
    assert stale.name != first["database_file"]

    observed = []
    real_remove = backup_module._remove_superseded_databases

    def checked_remove(target, key, keep):
        published = json.loads((backup / "b26.manifest.json").read_text())
        observed.append((published["database_file"], (backup / keep).exists(),
                         first_path.exists(), stale.exists()))
        real_remove(target, key, keep)

    monkeypatch.setattr(backup_module, "_remove_superseded_databases",
                        checked_remove)
    second = run_backup(conn, key="b26", owner="retry", target=backup,
                        clock=clock, store=store)

    assert second["database_file"] != first["database_file"]
    assert observed == [(second["database_file"], True, True, True)]
    assert not first_path.exists()
    assert not stale.exists()
    assert (backup / second["database_file"]).exists()


def _current_manifest(**overrides):
    """A current-shaped manifest: every current key, with non-null database
    fields and an artifact byte_size."""
    manifest = {
        "schema_version": "catalog_backup.v1.0",
        "backup_key": "b-shape",
        "cutoff": "2026-09-12T00:00:00.000000Z",
        "database": "sha256:" + "0" * 64,
        "database_file": "b-shape.abc.sqlite",
        "database_content_hash": "sha256:" + "1" * 64,
        "database_rows": {"outbox": 1},
        "database_size": 4096,
        "artifacts": {"evidence.bin": {"content_hash": "sha256:" + "2" * 64,
                                       "byte_size": 8}},
    }
    manifest.update(overrides)
    return manifest


def _legacy_manifest(**overrides):
    """A legacy-shaped manifest that predates the current-only optional keys."""
    manifest = {
        "schema_version": "catalog_backup.v1.0",
        "backup_key": "b-shape",
        "cutoff": "2026-09-12T00:00:00.000000Z",
        "database": "sha256:" + "0" * 64,
        "artifacts": {"evidence.bin": {"content_hash": "sha256:" + "2" * 64}},
    }
    manifest.update(overrides)
    return manifest


@pytest.mark.parametrize("field", ["database_file", "database_content_hash",
                                   "database_size"])
def test_current_manifest_rejects_explicit_null_database_field(field):
    manifest = _current_manifest()
    manifest[field] = None

    with pytest.raises(ValueError):
        backup_module._validated_manifest_shape(manifest, current=True)


@pytest.mark.parametrize("artifact", [
    {"content_hash": "sha256:" + "2" * 64},
    {"content_hash": "sha256:" + "2" * 64, "byte_size": None},
], ids=["missing", "null"])
def test_current_manifest_requires_artifact_byte_size(artifact):
    manifest = _current_manifest(artifacts={"evidence.bin": artifact})

    with pytest.raises(ValueError):
        backup_module._validated_manifest_shape(manifest, current=True)


@pytest.mark.parametrize("field", ["database_file", "database_content_hash",
                                   "database_size"])
def test_legacy_manifest_present_null_optional_field_is_rejected(field):
    manifest = _legacy_manifest()
    manifest[field] = None

    with pytest.raises(ValueError):
        backup_module._validated_manifest_shape(manifest, current=False)


def test_legacy_manifest_omitting_optional_fields_validates():
    manifest = _legacy_manifest()

    name, artifacts, size = backup_module._validated_manifest_shape(
        manifest, current=False)

    assert name == "b-shape.sqlite"
    assert size is None
    assert artifacts == {"evidence.bin": {"content_hash": "sha256:" + "2" * 64}}
