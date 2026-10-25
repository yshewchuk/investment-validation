"""Retention pruning of verified backups and unreferenced materializations."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

import engine.v2.ops.retention as retention_module
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import file_hash
from engine.v2.ops.retention import enforce_retention
from engine.v2.ops.snapshot_roots import default_materialization_base

SCHEMA = "catalog_backup.v1.0"
REQUEST_BINDING = "materialization_request.json"
LIVE_HASH = "sha256:" + "ab" * 32
ZERO_HASH = "sha256:" + "0" * 64


def _files(root):
    root = Path(root)
    if not root.exists():
        return {}
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def _sqlite(path, *, broken_fk=False):
    conn = sqlite3.connect(path)
    try:
        if broken_fk:
            conn.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY)")
            conn.execute("CREATE TABLE child (id INTEGER PRIMARY KEY, "
                         "parent_id INTEGER REFERENCES parent(id))")
            conn.execute("INSERT INTO child VALUES (1, 99)")
        else:
            conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("INSERT INTO sample(value) VALUES ('kept')")
        conn.commit()
    finally:
        conn.close()


def _write_backup(scope, key, cutoff, artifacts=None, *, broken_fk=False):
    scope.mkdir(parents=True, exist_ok=True)
    database = scope / (key + ".sqlite")
    _sqlite(database, broken_fk=broken_fk)
    artifacts_dir = scope / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    metadata = {}
    for name, payload in (artifacts or {}).items():
        path = artifacts_dir / name
        path.write_bytes(payload)
        metadata[name] = {"content_hash": file_hash(path), "byte_size": path.stat().st_size}
    manifest = {
        "schema_version": SCHEMA,
        "backup_key": key,
        "cutoff": cutoff,
        "database": file_hash(database),
        "artifacts": metadata,
    }
    (scope / (key + ".manifest.json")).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _manifest(scope, key):
    return json.loads((scope / (key + ".manifest.json")).read_text(encoding="utf-8"))


def _save_manifest(scope, key, manifest):
    (scope / (key + ".manifest.json")).write_text(json.dumps(manifest), encoding="utf-8")


def _catalog_schema(conn):
    conn.execute("CREATE TABLE jobs (job_id TEXT PRIMARY KEY, kind TEXT, spec_json TEXT)")
    conn.execute("CREATE TABLE attempts "
                 "(attempt_id TEXT PRIMARY KEY, job_id TEXT, state TEXT)")
    conn.execute("CREATE TABLE attempt_input_bindings "
                 "(attempt_id TEXT, name TEXT, artifact_id TEXT)")


@pytest.fixture
def catalog():
    conn = sqlite3.connect(":memory:")
    _catalog_schema(conn)
    yield conn
    conn.close()


def _live_attempt(conn, bindings=(), *, kind="legacy_materialize",
                  attempt_id="att_live", job_id="job_live"):
    spec = {"kind": kind, "parameters": {}}
    if kind == "legacy_materialize":
        spec["parameters"]["input_mode"] = "snapshot"
    conn.execute("INSERT INTO jobs VALUES (?, ?, ?)",
                 (job_id, kind, json.dumps(spec)))
    conn.execute("INSERT INTO attempts VALUES (?, ?, ?)",
                 (attempt_id, job_id, "running"))
    conn.executemany("INSERT INTO attempt_input_bindings VALUES (?, ?, ?)",
                     [(attempt_id, name, artifact_id) for name, artifact_id in bindings])


def _refusal(snapshot_root, ops_root, conn, reason, **kwargs):
    before = _files(snapshot_root)
    with pytest.raises(OpsError) as caught:
        enforce_retention(ops_root, conn=conn, store=object(), **kwargs)
    assert caught.value.problem.details["reason_key"] == reason
    if reason == "live_request_binding_invalid":
        expected_code = "INPUT_CHANGED"
    elif reason.endswith("_keep_invalid"):
        expected_code = "INVALID_REQUEST"
    else:
        expected_code = "INTEGRITY_FAILED"
    assert caught.value.code == expected_code
    assert _files(snapshot_root) == before


def _materialization_dir(base, name, mtime_ns):
    path = base / name
    path.mkdir()
    (path / "sentinel.bin").write_bytes(b"kept")
    os.utime(path, ns=(mtime_ns, mtime_ns))
    path.chmod(0o555)
    return path


def test_verified_backups_prune_past_keep_and_rerun_is_noop(tmp_path, catalog):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b2", "2026-01-01T00:00:00+00:00", {"old.bin": b"old"})
    _write_backup(scope, "b1", "2026-01-02T00:00:00+00:00", {"new.bin": b"new"})

    deleted = enforce_retention(ops, conn=catalog, store=object(),
                                keep=1, materialization_keep=1)
    assert deleted == ["b2"]
    assert sorted(path.name for path in scope.iterdir() if path.is_file()) == [
        "b1.manifest.json", "b1.sqlite"]
    assert (scope / "artifacts" / "new.bin").read_bytes() == b"new"
    assert not (scope / "artifacts" / "old.bin").exists()

    before = _files(ops)
    assert enforce_retention(ops, conn=catalog, store=object(),
                             keep=1, materialization_keep=1) == []
    assert _files(ops) == before


@pytest.mark.parametrize(("damage", "reason"), [
    ("empty", "backup_database_invalid"),
    ("hash", "backup_database_hash_mismatch"),
    ("integrity", "backup_database_integrity_failed"),
])
def test_invalid_backup_databases_refuse_and_leave_every_file(
        tmp_path, catalog, damage, reason):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00",
                  {"a.bin": b"artifact"}, broken_fk=damage == "integrity")
    if damage == "empty":
        (scope / "b1.sqlite").write_bytes(b"")
    elif damage == "hash":
        manifest = _manifest(scope, "b1")
        manifest["database"] = ZERO_HASH
        _save_manifest(scope, "b1", manifest)

    _refusal(ops, ops, catalog, reason, keep=1, materialization_keep=1)


@pytest.mark.parametrize(("damage", "reason"), [
    ("size", "backup_artifact_size_mismatch"),
    ("hash", "backup_artifact_hash_mismatch"),
    ("extra", "backup_artifacts_mismatch"),
])
def test_invalid_backup_artifacts_refuse_and_leave_every_file(
        tmp_path, catalog, damage, reason):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00", {"a.bin": b"artifact"})
    if damage == "size":
        manifest = _manifest(scope, "b1")
        manifest["artifacts"]["a.bin"]["byte_size"] += 1
        _save_manifest(scope, "b1", manifest)
    elif damage == "hash":
        manifest = _manifest(scope, "b1")
        manifest["artifacts"]["a.bin"]["content_hash"] = ZERO_HASH
        _save_manifest(scope, "b1", manifest)
    else:
        (scope / "artifacts" / "extra.bin").write_bytes(b"extra")

    _refusal(ops, ops, catalog, reason, keep=1, materialization_keep=1)


@pytest.mark.parametrize(("env", "value", "reason"), [
    (retention_module.KEEP_ENV, "0", "backup_keep_invalid"),
    (retention_module.KEEP_ENV, "not-an-int", "backup_keep_invalid"),
    (retention_module.MATERIALIZATION_KEEP_ENV, "0", "materialization_keep_invalid"),
    (retention_module.MATERIALIZATION_KEEP_ENV, "-2", "materialization_keep_invalid"),
    (retention_module.MATERIALIZATION_KEEP_ENV, "not-an-int", "materialization_keep_invalid"),
])
def test_invalid_environment_limits_refuse_without_deleting(
        tmp_path, catalog, monkeypatch, env, value, reason):
    monkeypatch.delenv(retention_module.KEEP_ENV, raising=False)
    monkeypatch.delenv(retention_module.MATERIALIZATION_KEEP_ENV, raising=False)
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00", {"a.bin": b"artifact"})
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    _materialization_dir(base, "root-old", 1)
    _materialization_dir(base, "root-new", 2)
    monkeypatch.setenv(env, value)

    _refusal(tmp_path, ops, catalog, reason)


def test_materialization_keep_defaults_to_eleven(tmp_path, catalog, monkeypatch):
    monkeypatch.delenv(retention_module.MATERIALIZATION_KEEP_ENV, raising=False)
    ops = tmp_path / "ops"
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    roots = [_materialization_dir(base, f"root-{index:02d}", index + 1)
             for index in range(12)]

    enforce_retention(ops, conn=catalog, store=object())

    assert not roots[0].exists()
    assert all(root.is_dir() for root in roots[1:])


@pytest.mark.parametrize("kind", [
    "legacy_materialize", "legacy_finality", "legacy_features"])
def test_newest_materializations_and_live_request_root_survive(
        tmp_path, catalog, monkeypatch, kind):
    ops = tmp_path / "ops"
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    live = _materialization_dir(base, LIVE_HASH.removeprefix("sha256:"), 1)
    _materialization_dir(base, "root-b", 2)
    _materialization_dir(base, "root-c", 3)
    _materialization_dir(base, "root-d", 4)
    newest = _materialization_dir(base, "root-e", 5)
    _live_attempt(catalog, [(REQUEST_BINDING, "art_live")], kind=kind)
    monkeypatch.setattr(
        retention_module, "request_from_artifact",
        lambda conn, store, artifact_id: SimpleNamespace(request_hash=LIVE_HASH))

    enforce_retention(ops, conn=catalog, store=object(),
                      keep=1, materialization_keep=2)

    assert live.is_dir() and (live / "sentinel.bin").is_file()
    assert (base / "root-d").is_dir() and newest.is_dir()
    assert not (base / "root-b").exists()
    assert not (base / "root-c").exists()


@pytest.mark.parametrize("bindings", [
    (),
    ((REQUEST_BINDING, "art_a"), (REQUEST_BINDING, "art_b")),
])
def test_live_attempt_without_single_request_binding_refuses_and_deletes_nothing(
        tmp_path, catalog, bindings):
    ops = tmp_path / "ops"
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    _materialization_dir(base, LIVE_HASH.removeprefix("sha256:"), 1)
    _materialization_dir(base, "root-old", 2)
    _live_attempt(catalog, bindings)

    _refusal(tmp_path, ops, catalog, "live_request_binding_invalid",
             keep=1, materialization_keep=1)


def test_absent_materialization_base_still_refuses_invalid_live_binding(tmp_path, catalog):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "old", "2026-01-01T00:00:00+00:00")
    _write_backup(scope, "new", "2026-01-02T00:00:00+00:00")
    assert not default_materialization_base(ops).exists()
    _live_attempt(catalog, bindings=())

    _refusal(tmp_path, ops, catalog, "live_request_binding_invalid",
             keep=1, materialization_keep=1)


def test_live_job_spec_oversized_refuses_and_deletes_nothing(
        tmp_path, catalog, monkeypatch):
    ops = tmp_path / "ops"
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    _materialization_dir(base, "root-old", 1)
    _materialization_dir(base, "root-new", 2)
    _live_attempt(catalog)
    monkeypatch.setattr(retention_module, "_LIVE_SPEC_LIMIT", 16)

    _refusal(tmp_path, ops, catalog, "live_job_spec_oversized",
             keep=1, materialization_keep=1)


def test_live_attempt_count_exceeded_refuses_and_deletes_nothing(
        tmp_path, catalog, monkeypatch):
    ops = tmp_path / "ops"
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    _materialization_dir(base, "root-old", 1)
    _materialization_dir(base, "root-new", 2)
    _live_attempt(catalog, kind="other", attempt_id="att_a", job_id="job_a")
    _live_attempt(catalog, kind="other", attempt_id="att_b", job_id="job_b")
    monkeypatch.setattr(retention_module, "_LIVE_ATTEMPT_LIMIT", 1)

    _refusal(tmp_path, ops, catalog, "live_attempt_count_exceeded",
             keep=1, materialization_keep=1)


def test_unbound_current_publication_is_reported_and_only_older_roots_pruned(
        tmp_path, catalog, capsys):
    ops = tmp_path / "ops"
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    oldest = _materialization_dir(base, "root-a", 1)
    older = _materialization_dir(base, "root-b", 2)
    newer = _materialization_dir(base, "root-c", 3)
    newest = _materialization_dir(base, "root-d", 4)

    enforce_retention(ops, conn=catalog, store=object(),
                      keep=1, materialization_keep=2)

    events = [json.loads(line)["event"] for line in capsys.readouterr().err.splitlines()
              if line.startswith("{")]
    assert "RETENTION_CURRENT_PUBLICATION_UNBOUND" in events
    assert not oldest.exists() and not older.exists()
    assert newer.is_dir() and newest.is_dir()


def test_oversized_manifest_refuses_before_any_backup_is_deleted(tmp_path, catalog):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "old", "2026-01-01T00:00:00+00:00")
    _write_backup(scope, "new", "2026-01-02T00:00:00+00:00")
    (scope / "old.manifest.json").write_bytes(
        b"x" * (retention_module._MANIFEST_LIMIT + 1))

    _refusal(ops, ops, catalog, "backup_manifest_oversized",
             keep=1, materialization_keep=1)


def test_backup_keep_defaults_to_seven(tmp_path, catalog, monkeypatch):
    monkeypatch.delenv(retention_module.KEEP_ENV, raising=False)
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    for index in range(8):
        _write_backup(scope, f"backup-{index:02d}",
                      f"2026-01-{index + 1:02d}T00:00:00+00:00")

    assert enforce_retention(ops, conn=catalog, store=object(),
                             materialization_keep=1) == ["backup-00"]
    assert (scope / "backup-07.sqlite").exists()


@pytest.mark.parametrize(("damage", "reason"), [
    ("keys", "backup_manifest_keys"),
    ("cutoff", "backup_cutoff_invalid"),
])
def test_invalid_backup_manifest_shape_refuses_and_deletes_nothing(
        tmp_path, catalog, damage, reason):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "old", "2026-01-01T00:00:00+00:00")
    _write_backup(scope, "new", "2026-01-02T00:00:00+00:00")
    manifest = _manifest(scope, "old")
    if damage == "keys":
        manifest["unexpected"] = True
    else:
        manifest["cutoff"] = "not-a-time"
    _save_manifest(scope, "old", manifest)

    _refusal(ops, ops, catalog, reason, keep=1, materialization_keep=1)


def test_naive_manifest_cutoff_refuses_and_preserves_backup_scope(tmp_path, catalog):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "old", "2026-01-01T00:00:00+00:00")
    _write_backup(scope, "new", "2026-01-02T00:00:00+00:00")
    manifest = _manifest(scope, "old")
    manifest["cutoff"] = "2026-01-01T00:00:00"
    _save_manifest(scope, "old", manifest)

    _refusal(ops, ops, catalog, "backup_cutoff_invalid", keep=1, materialization_keep=1)


def test_keepalive_callback_covers_verification_and_deletion(tmp_path, catalog):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b2", "2026-01-01T00:00:00+00:00", {"old.bin": b"old"})
    _write_backup(scope, "b1", "2026-01-02T00:00:00+00:00", {"new.bin": b"new"})
    calls = []

    def keepalive():
        calls.append((scope / "b2.sqlite").exists())

    deleted = enforce_retention(ops, conn=catalog, store=object(),
                                keep=1, materialization_keep=1,
                                keepalive=keepalive)

    assert deleted == ["b2"]
    assert len(calls) >= 5
    assert True in calls
    assert False in calls
    assert not (scope / "b2.sqlite").exists()
    assert not (scope / "b2.manifest.json").exists()
    assert not (scope / "artifacts" / "old.bin").exists()
    assert (scope / "b1.sqlite").exists()
    assert (scope / "b1.manifest.json").exists()
    assert (scope / "artifacts" / "new.bin").exists()


def test_keepalive_callback_failure_propagates_before_deletion(tmp_path, catalog):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b2", "2026-01-01T00:00:00+00:00", {"old.bin": b"old"})
    _write_backup(scope, "b1", "2026-01-02T00:00:00+00:00", {"new.bin": b"new"})
    before = _files(scope)

    def keepalive():
        raise RuntimeError("distinctive keepalive failure")

    with pytest.raises(RuntimeError, match="distinctive keepalive failure"):
        enforce_retention(ops, conn=catalog, store=object(),
                          keep=1, materialization_keep=1,
                          keepalive=keepalive)

    assert _files(scope) == before


def test_database_descriptor_stable_when_pathname_replaced_after_open(
        tmp_path, monkeypatch):
    database = tmp_path / "b1.sqlite"
    _sqlite(database)
    expected = file_hash(database)
    replacement = b"invalid bytes swapped in at the pathname"
    real_open = retention_module.os.open
    swapped = False

    def open_with_swap(target, flags, *args, **kwargs):
        nonlocal swapped
        fd = real_open(target, flags, *args, **kwargs)
        if str(target) == str(database) and not swapped:
            swapped = True
            retained = database.with_name(database.name + ".retained")
            os.rename(database, retained)
            sink = real_open(database, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(sink, replacement)
            finally:
                os.close(sink)
        return fd

    monkeypatch.setattr(retention_module.os, "open", open_with_swap)

    retention_module._database(database, expected)

    assert database.read_bytes() == replacement


def test_digest_keepalive_runs_once_per_chunk_and_matches_file_hash(tmp_path):
    path = tmp_path / "chunked.bin"
    path.write_bytes(b"a" * (2 * 1024 * 1024))
    calls = []
    fd = os.open(path, os.O_RDONLY)
    try:
        digest = retention_module._digest(fd, lambda: calls.append(None))
    finally:
        os.close(fd)

    assert digest == file_hash(path)
    assert len(calls) == 2


def test_live_ordinary_attempt_without_bindings_does_not_block_backup_pruning(
        tmp_path, catalog):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "old", "2026-01-01T00:00:00+00:00")
    _write_backup(scope, "new", "2026-01-02T00:00:00+00:00")
    _live_attempt(catalog, bindings=(), kind="ordinary")

    deleted = enforce_retention(ops, conn=catalog, store=object(), keep=1)

    assert deleted == ["old"]
    assert not (scope / "old.sqlite").exists()
    assert not (scope / "old.manifest.json").exists()
    assert (scope / "new.sqlite").exists()
    assert (scope / "new.manifest.json").exists()
