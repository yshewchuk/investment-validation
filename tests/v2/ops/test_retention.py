"""Retention pruning of verified backups and unreferenced materializations."""
from __future__ import annotations

import json
import os
import sqlite3
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import engine.v2.ops.retention as retention_module
from engine.v2.ops.errors import OpsError, fail
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


def _live_attempt(conn, bindings=()):
    spec = {"kind": "legacy_materialize", "parameters": {"input_mode": "snapshot"}}
    conn.execute("INSERT INTO jobs VALUES (?, ?, ?)",
                 ("job_live", "legacy_materialize", json.dumps(spec)))
    conn.execute("INSERT INTO attempts VALUES (?, ?, ?)",
                 ("att_live", "job_live", "running"))
    conn.executemany("INSERT INTO attempt_input_bindings VALUES (?, ?, ?)",
                     [("att_live", name, artifact_id) for name, artifact_id in bindings])


def _refusal(snapshot_root, ops_root, conn, reason, *, code="INTEGRITY_FAILED", **kwargs):
    before = _files(snapshot_root)
    with pytest.raises(OpsError) as caught:
        enforce_retention(ops_root, conn=conn, store=object(), **kwargs)
    assert caught.value.code == code
    assert caught.value.problem.details["reason_key"] == reason
    assert _files(snapshot_root) == before
    return caught.value


def _materialization_dir(base, name, mtime_ns):
    path = base / name
    path.mkdir()
    (path / "sentinel.bin").write_bytes(b"kept")
    os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def test_verified_backups_prune_past_keep_and_rerun_is_noop(tmp_path, catalog):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00", {"old.bin": b"old"})
    _write_backup(scope, "b2", "2026-01-02T00:00:00+00:00", {"new.bin": b"new"})

    deleted = enforce_retention(ops, conn=catalog, store=object(),
                                keep=1, materialization_keep=1)
    assert deleted == ["b1"]
    assert sorted(path.name for path in scope.iterdir() if path.is_file()) == [
        "b2.manifest.json", "b2.sqlite"]
    assert (scope / "artifacts" / "new.bin").read_bytes() == b"new"
    assert not (scope / "artifacts" / "old.bin").exists()

    before = _files(ops)
    assert enforce_retention(ops, conn=catalog, store=object(),
                             keep=1, materialization_keep=1) == []
    assert _files(ops) == before


def test_verification_creates_no_scope_entries(tmp_path, catalog):
    """Read-only verification must not add SQLite sidecars to the backup scope.

    Production backups come from a WAL-mode catalog. Opening such a database
    read-only without ``immutable=1`` can create a ``-shm``/``-wal`` sidecar
    after ``_scope_files`` enumerates the scope; the next retention pass then
    refuses that sidecar as a stray file. Verification must leave the scope
    unchanged and a second pass must still succeed.
    """
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00", {"a.bin": b"artifact"})
    database = scope / "b1.sqlite"
    conn = sqlite3.connect(database)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("INSERT INTO sample(value) VALUES ('wal')")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    manifest = _manifest(scope, "b1")
    manifest["database"] = file_hash(database)
    _save_manifest(scope, "b1", manifest)

    before = _files(ops)
    assert enforce_retention(ops, conn=catalog, store=object(),
                             keep=1, materialization_keep=1) == []
    assert _files(ops) == before
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


def test_corrupt_backup_refusal_names_the_failing_backup_key(tmp_path, catalog):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00", {"old.bin": b"old"})
    _write_backup(scope, "b2", "2026-01-02T00:00:00+00:00", {"new.bin": b"new"})
    manifest = _manifest(scope, "b2")
    manifest["database"] = ZERO_HASH
    _save_manifest(scope, "b2", manifest)

    error = _refusal(ops, ops, catalog, "backup_database_hash_mismatch",
                     keep=1, materialization_keep=1)

    assert error.problem.details["backup_key"] == "b2"
    assert error.problem.details["reason_key"] == "backup_database_hash_mismatch"
    assert set(error.problem.details) == {"reason_key", "backup_key"}


def test_database_pathname_replaced_after_pinned_hash_refuses_with_changed_reason(
        tmp_path, catalog, monkeypatch):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00", {"old.bin": b"old"})
    _write_backup(scope, "b2", "2026-01-02T00:00:00+00:00", {"new.bin": b"new"})
    older = scope / "b1.sqlite"
    swap = tmp_path / "swap.sqlite"
    swap.write_bytes(older.read_bytes())
    real_file_hash = retention_module.file_hash
    state = {"replaced": False}

    def file_hash(path, **kwargs):
        digest = real_file_hash(path, **kwargs)
        if not state["replaced"] and str(path).startswith("/proc/self/fd/"):
            state["replaced"] = True
            os.replace(swap, older)
        return digest

    monkeypatch.setattr(retention_module, "file_hash", file_hash)

    _refusal(ops, ops, catalog, "backup_database_changed",
             keep=1, materialization_keep=1)
    assert state["replaced"]


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

    _refusal(tmp_path, ops, catalog, reason, code="INVALID_REQUEST")


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


def test_backup_keep_defaults_to_seven(tmp_path, catalog, monkeypatch):
    monkeypatch.delenv(retention_module.KEEP_ENV, raising=False)
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    for index in range(8):
        _write_backup(scope, f"b{index}", f"2026-01-{index + 1:02d}T00:00:00+00:00",
                      {f"a{index}.bin": b"artifact"})

    enforce_retention(ops, conn=catalog, store=object())

    assert sorted(path.name for path in scope.iterdir() if path.is_file()) == [
        "b1.manifest.json", "b1.sqlite", "b2.manifest.json", "b2.sqlite",
        "b3.manifest.json", "b3.sqlite", "b4.manifest.json", "b4.sqlite",
        "b5.manifest.json", "b5.sqlite", "b6.manifest.json", "b6.sqlite",
        "b7.manifest.json", "b7.sqlite"]
    assert not (scope / "artifacts" / "a0.bin").exists()


def test_newest_materializations_and_live_request_root_survive(tmp_path, catalog, monkeypatch):
    ops = tmp_path / "ops"
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    live = _materialization_dir(base, LIVE_HASH.removeprefix("sha256:"), 1)
    _materialization_dir(base, "root-b", 2)
    _materialization_dir(base, "root-c", 3)
    _materialization_dir(base, "root-d", 4)
    newest = _materialization_dir(base, "root-e", 5)
    _live_attempt(catalog, [(REQUEST_BINDING, "art_live")])
    captured = []

    def request_from_artifact(conn, store, artifact_id):
        captured.append(artifact_id)
        return SimpleNamespace(request_hash=LIVE_HASH)

    monkeypatch.setattr(retention_module, "request_from_artifact", request_from_artifact)

    enforce_retention(ops, conn=catalog, store=object(),
                      keep=1, materialization_keep=2)

    assert captured == ["art_live"]
    assert live.is_dir() and (live / "sentinel.bin").is_file()
    assert (base / "root-d").is_dir() and newest.is_dir()
    assert not (base / "root-b").exists()
    assert not (base / "root-c").exists()


def test_live_crosscheck_bound_request_root_survives(tmp_path, catalog, monkeypatch):
    ops = tmp_path / "ops"
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    bound = _materialization_dir(base, LIVE_HASH.removeprefix("sha256:"), 1)
    _materialization_dir(base, "root-old", 2)
    newest = _materialization_dir(base, "root-new", 3)
    spec = {"kind": "legacy_finality", "parameters": {}}
    catalog.execute("INSERT INTO jobs VALUES (?, ?, ?)",
                    ("job_check", "legacy_finality", json.dumps(spec)))
    catalog.execute("INSERT INTO attempts VALUES (?, ?, ?)",
                    ("att_check", "job_check", "running"))
    catalog.execute("INSERT INTO attempt_input_bindings VALUES (?, ?, ?)",
                    ("att_check", REQUEST_BINDING, "art_check"))
    monkeypatch.setattr(
        retention_module, "request_from_artifact",
        lambda conn, store, artifact_id: SimpleNamespace(request_hash=LIVE_HASH))

    enforce_retention(ops, conn=catalog, store=object(),
                      keep=1, materialization_keep=1)

    assert bound.is_dir() and (bound / "sentinel.bin").is_file()
    assert newest.is_dir()
    assert not (base / "root-old").exists()


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
             code="INPUT_CHANGED", keep=1, materialization_keep=1)


def test_live_attempt_without_materialization_intent_is_skipped(tmp_path, catalog,
                                                                 monkeypatch):
    ops = tmp_path / "ops"
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    _materialization_dir(base, "root-old", 1)
    newest = _materialization_dir(base, "root-new", 2)
    spec = {"kind": "legacy_finality", "parameters": {}}
    catalog.execute("INSERT INTO jobs VALUES (?, ?, ?)",
                    ("job_skip", "legacy_finality", json.dumps(spec)))
    catalog.execute("INSERT INTO attempts VALUES (?, ?, ?)",
                    ("att_skip", "job_skip", "running"))
    captured = []
    monkeypatch.setattr(
        retention_module, "request_from_artifact",
        lambda conn, store, artifact_id: captured.append(artifact_id))

    enforce_retention(ops, conn=catalog, store=object(),
                      keep=1, materialization_keep=1)

    assert captured == []
    assert newest.is_dir() and (newest / "sentinel.bin").is_file()
    assert not (base / "root-old").exists()


def test_oversized_live_spec_refuses_in_preflight_and_deletes_nothing(tmp_path, catalog):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00", {"old.bin": b"old"})
    _write_backup(scope, "b2", "2026-01-02T00:00:00+00:00", {"new.bin": b"new"})
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    _materialization_dir(base, "root-old", 1)
    _materialization_dir(base, "root-new", 2)
    spec = {"kind": "legacy_materialize",
            "parameters": {"input_mode": "snapshot"},
            "padding": "x" * (retention_module.MAX_LIVE_SPEC_BYTES + 1)}
    catalog.execute("INSERT INTO jobs VALUES (?, ?, ?)",
                    ("job_big", "legacy_materialize", json.dumps(spec)))
    catalog.execute("INSERT INTO attempts VALUES (?, ?, ?)",
                    ("att_big", "job_big", "running"))

    error = _refusal(tmp_path, ops, catalog, "live_job_spec_too_large",
                     keep=1, materialization_keep=1)

    assert error.code == "INTEGRITY_FAILED"
    assert error.problem.details["attempt_id"] == "att_big"
    assert error.problem.details["spec_bytes"] > retention_module.MAX_LIVE_SPEC_BYTES
    assert error.problem.details["byte_limit"] == retention_module.MAX_LIVE_SPEC_BYTES


def test_false_non_dict_live_parameters_refuse_with_attempt_id(tmp_path, catalog):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00", {"old.bin": b"old"})
    _write_backup(scope, "b2", "2026-01-02T00:00:00+00:00", {"new.bin": b"new"})
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    _materialization_dir(base, "root-old", 1)
    _materialization_dir(base, "root-new", 2)
    spec = {"kind": "legacy_materialize", "parameters": []}
    catalog.execute("INSERT INTO jobs VALUES (?, ?, ?)",
                    ("job_bad", "legacy_materialize", json.dumps(spec)))
    catalog.execute("INSERT INTO attempts VALUES (?, ?, ?)",
                    ("att_bad", "job_bad", "running"))

    error = _refusal(tmp_path, ops, catalog, "live_job_parameters_malformed",
                     keep=1, materialization_keep=1)

    assert error.code == "INTEGRITY_FAILED"
    assert error.problem.details["reason_key"] == "live_job_parameters_malformed"
    assert error.problem.details["attempt_id"] == "att_bad"


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


@pytest.mark.parametrize("kind", ["database", "artifact"])
def test_over_limit_backup_stream_refuses_and_deletes_nothing(
        tmp_path, catalog, monkeypatch, kind):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00", {"old.bin": b"old"})
    _write_backup(scope, "b2", "2026-01-02T00:00:00+00:00", {"new.bin": b"new"})
    monkeypatch.setattr(retention_module, "MAX_BACKUP_STREAM_BYTES", 10)
    if kind == "database":
        database = scope / "b1.sqlite"
        with database.open("wb") as stream:
            stream.truncate(11)
        manifest = _manifest(scope, "b1")
        manifest["database"] = "sha256:" + "0" * 64
        _save_manifest(scope, "b1", manifest)
    else:
        artifact = scope / "artifacts" / "old.bin"
        with artifact.open("wb") as stream:
            stream.truncate(11)
        manifest = _manifest(scope, "b1")
        manifest["artifacts"]["old.bin"] = {
            "content_hash": "sha256:" + "0" * 64, "byte_size": 11}
        _save_manifest(scope, "b1", manifest)
    error = _refusal(ops, ops, catalog, "stream_too_large",
                     keep=1, materialization_keep=1)
    assert error.code == "INTEGRITY_FAILED"


def test_file_hash_stream_cap_reads_no_chunk_past_the_limit(tmp_path, monkeypatch):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"x" * ((1 << 20) + 64))
    limit = 10
    reads = {"total": 0}
    real_open = Path.open

    class _CountedStream:
        def __init__(self, stream):
            self._stream = stream

        def read(self, size=-1):
            chunk = self._stream.read(size)
            reads["total"] += len(chunk)
            return chunk

        def tell(self):
            return self._stream.tell()

        def __enter__(self):
            self._stream.__enter__()
            return self

        def __exit__(self, *exc_info):
            return self._stream.__exit__(*exc_info)

    def counting_open(self, *args, **kwargs):
        return _CountedStream(real_open(self, *args, **kwargs))

    monkeypatch.setattr(Path, "open", counting_open)

    with pytest.raises(OpsError) as caught:
        file_hash(path, max_bytes=limit)

    assert caught.value.code == "INTEGRITY_FAILED"
    assert caught.value.problem.details["reason_key"] == "stream_too_large"
    assert reads["total"] <= limit + 1


def test_file_hash_negative_max_bytes_refuses_empty_file(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"")

    with pytest.raises(OpsError) as caught:
        file_hash(path, max_bytes=-1)

    assert caught.value.code == "INVALID_REQUEST"
    assert caught.value.problem.details["max_bytes"] == -1


def test_file_hash_reports_cumulative_bytes_to_on_progress(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"x" * 2304000)
    seen = []
    file_hash(path, on_progress=seen.append)
    assert seen == [1 << 20, 2 << 20, 2304000]


def test_oversized_manifest_refuses_with_typed_error(tmp_path, catalog, monkeypatch):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00", {"a.bin": b"artifact"})
    monkeypatch.setattr(retention_module, "_MAX_MANIFEST_BYTES", 10)
    error = _refusal(ops, ops, catalog, "backup_manifest_too_large",
                     keep=1, materialization_keep=1)
    assert error.code == "INTEGRITY_FAILED"


def test_scope_listing_permission_error_is_typed(tmp_path, catalog, monkeypatch):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00", {"old.bin": b"old"})
    _write_backup(scope, "b2", "2026-01-02T00:00:00+00:00", {"new.bin": b"new"})
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    _materialization_dir(base, "root-old", 1)
    _materialization_dir(base, "root-new", 2)
    real_iterdir = Path.iterdir

    def deny_scope_listing(self, *args, **kwargs):
        if self == scope:
            raise PermissionError("denied")
        return real_iterdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "iterdir", deny_scope_listing)

    error = _refusal(tmp_path, ops, catalog, "retention_io_failed",
                     keep=1, materialization_keep=1)
    assert error.code == "INTEGRITY_FAILED"


def test_injected_deletion_oserror_is_typed(tmp_path, catalog, monkeypatch):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00", {"a.bin": b"old"})
    _write_backup(scope, "b2", "2026-01-02T00:00:00+00:00", {"b.bin": b"new"})
    before = _files(ops)
    monkeypatch.setattr(Path, "unlink",
                        lambda self: (_ for _ in ()).throw(OSError("denied")))
    with pytest.raises(OpsError) as caught:
        enforce_retention(ops, conn=catalog, store=object(),
                          keep=1, materialization_keep=1)
    assert caught.value.code == "INTEGRITY_FAILED"
    assert caught.value.problem.details["reason_key"] == "backup_delete_failed"
    assert "denied" not in str(caught.value.problem)
    assert _files(ops) == before


def test_backup_deletion_failure_is_typed_without_root(tmp_path, catalog, monkeypatch):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00", {"a.bin": b"old"})
    _write_backup(scope, "b2", "2026-01-02T00:00:00+00:00", {"b.bin": b"new"})
    before = _files(ops)
    monkeypatch.setattr(
        Path, "unlink",
        lambda self: (_ for _ in ()).throw(PermissionError("denied")))
    with pytest.raises(OpsError) as caught:
        enforce_retention(ops, conn=catalog, store=object(),
                          keep=1, materialization_keep=1)
    assert caught.value.code == "INTEGRITY_FAILED"
    assert caught.value.problem.details["reason_key"] == "backup_delete_failed"
    assert _files(ops) == before


def test_read_only_materialization_root_is_removed(tmp_path, catalog, monkeypatch):
    ops = tmp_path / "ops"
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    root = _materialization_dir(base, "root-old", 1)
    os.chmod(root, 0o555)
    _materialization_dir(base, "root-new", 2)
    real_unlink = os.unlink
    writable_parents = []

    def unlink(path, *args, **kwargs):
        parent = Path(path).parent
        writable_parents.append(bool(os.stat(parent).st_mode & stat.S_IWUSR))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(retention_module.os, "unlink", unlink)
    enforce_retention(ops, conn=catalog, store=object(),
                      keep=1, materialization_keep=1)
    assert not root.exists()
    assert writable_parents and all(writable_parents)


def test_materialization_deletion_failure_is_typed_without_root(tmp_path, catalog, monkeypatch):
    ops = tmp_path / "ops"
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    _materialization_dir(base, "root-new", 2)
    old = _materialization_dir(base, "root-old", 1)
    monkeypatch.setattr(
        retention_module.os, "unlink",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")))
    with pytest.raises(OpsError) as caught:
        enforce_retention(ops, conn=catalog, store=object(),
                          keep=1, materialization_keep=1)
    assert caught.value.code == "INTEGRITY_FAILED"
    assert caught.value.problem.details["reason_key"] == "materialization_delete_failed"
    assert old.is_dir()


def test_remove_tree_lease_lost_propagates_and_stops_further_removal(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (root / "one.txt").write_bytes(b"one")
    (root / "sub" / "two.txt").write_bytes(b"two")
    error = fail("LEASE_LOST", "the attempt lost its lease during materialization removal")
    calls = {"count": 0}

    def keepalive():
        calls["count"] += 1
        if calls["count"] >= 5:
            raise error

    with pytest.raises(OpsError) as caught:
        retention_module._remove_tree(root, keepalive=keepalive)

    assert caught.value is error
    assert caught.value.code == "LEASE_LOST"
    assert not (root / "sub" / "two.txt").exists()
    assert (root / "one.txt").is_file()
    assert (root / "sub").is_dir()
    assert root.is_dir()


def test_configured_materialization_base_is_used_for_retention(tmp_path, catalog):
    ops = tmp_path / "ops"
    default_base = default_materialization_base(ops)
    default_base.mkdir(parents=True)
    default_root = _materialization_dir(default_base, "default-root", 1)

    configured_base = tmp_path / "configured-base"
    configured_base.mkdir()
    oldest = _materialization_dir(configured_base, "root-old", 1)
    newest = _materialization_dir(configured_base, "root-new", 2)

    enforce_retention(ops, conn=catalog, store=object(),
                      keep=1, materialization_keep=1,
                      materialization_base=configured_base)

    assert not oldest.exists()
    assert newest.is_dir()
    assert default_root.is_dir()


def test_integrity_progress_handler_lease_lost_propagates_and_deletes_nothing(
        tmp_path, catalog, monkeypatch):
    ops = tmp_path / "ops"
    scope = ops / "backups" / "shadow"
    _write_backup(scope, "b1", "2026-01-01T00:00:00+00:00", {"old.bin": b"old"})
    _write_backup(scope, "b2", "2026-01-02T00:00:00+00:00", {"new.bin": b"new"})
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    _materialization_dir(base, "root-old", 1)
    _materialization_dir(base, "root-new", 2)
    before = _files(ops)
    state = {"integrity": False, "calls": 0}
    real_integrity_errors = retention_module.integrity_errors

    def integrity_errors(conn):
        state["integrity"] = True
        try:
            return real_integrity_errors(conn)
        finally:
            state["integrity"] = False

    def keepalive():
        if state["integrity"]:
            state["calls"] += 1
            raise fail("LEASE_LOST",
                       "the attempt lost its lease during backup verification")

    monkeypatch.setattr(retention_module, "integrity_errors", integrity_errors)
    monkeypatch.setattr(retention_module, "_INTEGRITY_PROGRESS_INTERVAL", 1)

    with pytest.raises(OpsError) as caught:
        enforce_retention(ops, conn=catalog, store=object(),
                          keep=1, materialization_keep=1, keepalive=keepalive)

    assert caught.value.code == "LEASE_LOST"
    assert caught.value.problem.details.get("reason_key") != "backup_database_verify_failed"
    assert state["calls"] > 0
    assert _files(ops) == before


def test_hidden_in_progress_materialization_root_survives_while_old_visible_roots_prune(
        tmp_path, catalog):
    ops = tmp_path / "ops"
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    hidden = base / ("." + "ef" * 32 + ".partial-att_partial")
    hidden.mkdir()
    (hidden / "partial.bin").write_bytes(b"in flight")
    (hidden / "nested").mkdir()
    (hidden / "nested" / "part.bin").write_bytes(b"partial")
    os.utime(hidden, ns=(1, 1))
    old = _materialization_dir(base, "root-old", 2)
    newest = _materialization_dir(base, "root-new", 3)

    enforce_retention(ops, conn=catalog, store=object(),
                      keep=1, materialization_keep=1)

    assert hidden.is_dir()
    assert (hidden / "partial.bin").read_bytes() == b"in flight"
    assert (hidden / "nested" / "part.bin").read_bytes() == b"partial"
    assert newest.is_dir() and (newest / "sentinel.bin").is_file()
    assert not old.exists()


def test_completed_attempt_snapshot_binding_does_not_retain_old_root(
        tmp_path, catalog, monkeypatch):
    ops = tmp_path / "ops"
    base = default_materialization_base(ops)
    base.mkdir(parents=True)
    bound = _materialization_dir(base, "cd" * 32, 1)
    newest = _materialization_dir(base, "root-new", 2)
    spec = {"kind": "legacy_materialize", "parameters": {"input_mode": "snapshot"}}
    catalog.execute("INSERT INTO jobs VALUES (?, ?, ?)",
                    ("job_done", "legacy_materialize", json.dumps(spec)))
    catalog.execute("INSERT INTO attempts VALUES (?, ?, ?)",
                    ("att_done", "job_done", "succeeded"))
    catalog.execute("INSERT INTO attempt_input_bindings VALUES (?, ?, ?)",
                    ("att_done", REQUEST_BINDING, "art_done"))
    captured = []

    def request_from_artifact(conn, store, artifact_id):
        captured.append(artifact_id)
        return SimpleNamespace(request_hash="sha256:" + "cd" * 32)

    monkeypatch.setattr(retention_module, "request_from_artifact", request_from_artifact)

    enforce_retention(ops, conn=catalog, store=object(),
                      keep=1, materialization_keep=1)

    assert captured == []
    assert newest.is_dir() and (newest / "sentinel.bin").is_file()
    assert not bound.exists()
