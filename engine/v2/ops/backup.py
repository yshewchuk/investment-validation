"""Durable local backup effect; remote mirroring is deliberately out of scope."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    format_timestamp,
    from_document,
    fsync_directory,
    safe_relative_path,
)
from engine.v2.ops.catalog import backup_to, integrity_errors, transaction
from engine.v2.ops.errors import fail
from engine.v2.ops.outbox import claim, complete, enqueue

#: Prior manifests are advisory metadata; reads beyond this bound are invalid.
MAX_MANIFEST_BYTES = 1024 * 1024

#: The exact top-level keys a current manifest carries. Extra or missing keys
#: make prior metadata invalid (rebuild); they never become a refusal.
_MANIFEST_KEYS = frozenset({
    "schema_version", "backup_key", "cutoff", "database", "database_file",
    "database_content_hash", "database_rows", "artifacts",
})

_CHUNK = 1 << 20
_LOG_INTERVAL_SECONDS = 5.0


def _rss_mb():
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except (ImportError, OSError):
        return None


def _log(message):
    print(f"[backup] {message}", file=sys.stderr, flush=True)


class _Progress:
    """Bounded stderr progress: start/end plus a periodic rate and ETA."""

    def __init__(self, stage, total=None):
        self.stage = stage
        self.total = total
        self.count = 0
        self.started = None
        self.last = None

    def start(self):
        self.started = self.last = time.monotonic()
        total = "?" if self.total is None else self.total
        _log(f"{self.stage}: start total={total} rss_mb={_rss_mb()}")

    def advance(self, amount=1):
        self.count += amount
        now = time.monotonic()
        if self.last is not None and now - self.last >= _LOG_INTERVAL_SECONDS:
            self._emit(now)
            self.last = now

    def _emit(self, now):
        elapsed = max(now - self.started, 1e-9)
        rate = self.count / elapsed
        if self.total:
            remaining = max(self.total - self.count, 0)
            eta = f"{remaining / rate:.1f}s" if rate > 0 else "?"
        else:
            eta = "?"
        total = "?" if self.total is None else self.total
        _log(f"{self.stage}: {self.count}/{total} rate={rate:.1f}/s ETA={eta} "
             f"rss_mb={_rss_mb()}")

    def end(self):
        now = time.monotonic()
        elapsed = max(now - self.started, 1e-9)
        rate = self.count / elapsed
        _log(f"{self.stage}: end count={self.count} elapsed={elapsed:.1f}s "
             f"rate={rate:.1f}/s rss_mb={_rss_mb()}")


def _single_filename(value):
    """A safe, plain single path component (no separators or traversal)."""
    if (not isinstance(value, str) or not value or value in (".", "..")
            or "/" in value or "\\" in value or "\x00" in value):
        raise ValueError("not a safe single filename")
    return value


def manifest_database_path(root, manifest):
    """The database path a manifest names, falling back to legacy ``<key>.sqlite``."""
    name = manifest.get("database_file")
    if name is None:
        name = _single_filename(manifest.get("backup_key")) + ".sqlite"
    else:
        name = _single_filename(name)
    return Path(root) / name


def _stage_token(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _stage_prefix(key):
    return f".bkp-{_stage_token(key)}."


_DIGEST_SKIP_PREFIXES = ("INSERT INTO outbox", 'INSERT INTO "outbox"',)
_DIGEST_OUTBOX_COLUMNS = "effect_id, kind, logical_key, payload_hash, payload_json"


def _logical_digest(conn, *, stage, expected_lines=None):
    """SHA-256 over the connection's logical SQL dump, streamed line by line.

    Outbox rows are replaced by a stable identity/payload projection: claiming
    or retrying an effect rewrites only its volatile state, attempt count,
    receipt, token, owner and lease, which must not make a lost-ack retry look
    like new source data. The physical file hash still covers the whole
    database.
    """
    progress = _Progress(stage, total=expected_lines)
    digest = hashlib.sha256()
    progress.start()
    try:
        for line in conn.iterdump():
            if line.startswith(_DIGEST_SKIP_PREFIXES):
                continue
            digest.update(line.encode("utf-8"))
            progress.advance()
        for row in conn.execute(
                f"SELECT {_DIGEST_OUTBOX_COLUMNS} FROM outbox ORDER BY effect_id"):
            digest.update(json.dumps(list(row), separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8") + b"\n")
            progress.advance()
    finally:
        progress.end()
    return "sha256:" + digest.hexdigest()


def _hash_file(path, *, stage):
    progress = _Progress(stage, total=path.stat().st_size)
    digest = hashlib.sha256()
    progress.start()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_CHUNK), b""):
                digest.update(chunk)
                progress.advance(len(chunk))
    finally:
        progress.end()
    return "sha256:" + digest.hexdigest()


def _stream_copy(source, destination, *, expected_hash=None, expected_size=None,
                 stage, fault=None):
    """Copy ``source`` to ``destination`` in bounded chunks, durably and atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = source.open("rb")
    try:
        progress = _Progress(stage, total=os.fstat(stream.fileno()).st_size)
        fd, temporary_name = tempfile.mkstemp(
            dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".pending")
        temporary = Path(temporary_name)
        try:
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(fd, "wb") as out:
                progress.start()
                for chunk in iter(lambda: stream.read(_CHUNK), b""):
                    out.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    progress.advance(len(chunk))
                out.flush()
                os.fsync(out.fileno())
                progress.end()
            actual_hash = "sha256:" + digest.hexdigest()
            if expected_size is not None and size != expected_size:
                raise fail("INTEGRITY_FAILED", f"{stage} size mismatch")
            if expected_hash is not None and actual_hash != expected_hash:
                raise fail("INTEGRITY_FAILED", f"{stage} hash mismatch")
            if fault is not None:
                fault(destination.name)
            os.replace(temporary, destination)
            temporary = None
            fsync_directory(destination.parent)
            return size
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
    finally:
        stream.close()


def prepare_backup(conn, key, artifacts, *, clock):
    """Enqueue one ``backup`` effect for ``key``, idempotent across attempts.

    A retry (after ``fail_effect`` returns the effect to pending) must call
    this again before re-running ``run_backup``, but the effect already
    fixed its own ``cutoff`` and artifact list on the FIRST attempt — a
    retry that rebuilt the payload with the clock's current time would hand
    ``outbox.enqueue`` a different payload hash for the same logical key and
    fail with ``IDEMPOTENCY_CONFLICT`` (P2-C07). So an existing ``backup``
    effect for this key is returned as-is, with no new payload built at all.
    """
    if len(safe_relative_path(key)) != 1:
        raise fail("INVALID_REQUEST", "unsafe backup key")
    with transaction(conn):
        existing = conn.execute("SELECT effect_id FROM outbox WHERE kind='backup' AND logical_key=?",
                                (key,)).fetchone()
        if existing is not None:
            return existing["effect_id"]
        payload = {"backup_key": key, "artifacts": artifacts,
                   "cutoff": format_timestamp(clock.now()), "schema_version": "catalog_backup.v1.0"}
        return enqueue(conn, "backup", key, payload)


def _no_keepalive():
    return None


def _delivered_receipt(conn, kind, logical_key):
    """The stored receipt for an already-``delivered`` outbox row under
    ``(kind, logical_key)``, or ``None`` if no such row is delivered yet.

    #98: lets a retry after a crash between the effect's own commit and the
    caller's attempt commit treat an already-completed effect as success,
    instead of failing permanently with STALE_EXPECTATION because
    ``claim()`` no longer matches a delivered row.
    """
    row = conn.execute(
        "SELECT receipt_json FROM outbox WHERE kind=? AND logical_key=? AND state='delivered'",
        (kind, logical_key)).fetchone()
    return None if row is None else json.loads(row["receipt_json"])


def _delivered_or_stale(conn, kind, logical_key):
    """``_delivered_receipt`` or STALE_EXPECTATION: the claim-failure branch."""
    delivered = _delivered_receipt(conn, kind, logical_key)
    if delivered is None:
        raise fail("STALE_EXPECTATION", "backup effect is not pending")
    return delivered


@contextmanager
def _read_snapshot(conn):
    """One explicit deferred read transaction, so counts, prior validation and
    a backup agree.

    ``BEGIN`` (not ``BEGIN IMMEDIATE``) only takes a read lock on first access,
    leaving writers free in WAL mode. Commit on success, roll back on anything
    else. Callers keep this open only around counting, logical hashing, prior
    validation and ``backup_to``: never around artifact copying or manifest
    publication.
    """
    if conn.in_transaction:
        raise RuntimeError("nested catalog transaction: backup needs a clean connection")
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _table_row_counts(conn, *, stage="row counts"):
    """Deterministic row count per non-internal table, keyed and ordered by name."""
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name")]
    progress = _Progress(stage, total=len(tables))
    counts = {}
    progress.start()
    try:
        for name in tables:
            quoted = '"' + name.replace('"', '""') + '"'
            counts[name] = conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            progress.advance()
    finally:
        progress.end()
    return counts


def _read_manifest(manifest_path):
    """Parse a prior manifest within ``MAX_MANIFEST_BYTES``; a missing,
    symlinked, non-regular, oversize or invalid prior is reported as ``None``."""
    from engine.v2.ops.workflows.commands import read_bounded_bytes

    try:
        return json.loads(read_bounded_bytes(manifest_path, MAX_MANIFEST_BYTES))
    except (OSError, ValueError, RecursionError):
        return None


def _expected_dump_lines(rows):
    return sum(rows.values()) + len(rows) + 2


def _verified_prior_backup(manifest_path, target, source_rows, source_digest, key, payload):
    """The existing manifest when it and its named database are a verified prior
    result for this payload, else ``None``.

    Row counts alone do not prove content, so the prior database's file hash,
    integrity and logical dump digest are all checked inside the source read
    snapshot. A prior without the current key set or ``database_file`` /
    ``database_content_hash`` is legacy or malformed and is rebuilt, never
    refused."""
    manifest = _read_manifest(manifest_path)
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        return None
    if not isinstance(manifest.get("database_file"), str):
        return None
    if (manifest.get("backup_key") != key
            or manifest.get("schema_version") != payload["schema_version"]
            or manifest.get("cutoff") != payload["cutoff"]
            or manifest.get("database_rows") != source_rows
            or manifest.get("database_content_hash") != source_digest):
        return None
    try:
        database = manifest_database_path(target, manifest)
        if not database.is_file():
            return None
        probe = sqlite3.connect(database)
        try:
            if integrity_errors(probe):
                return None
            if _table_row_counts(probe, stage="prior-row-counts") != source_rows:
                return None
            if _logical_digest(probe, stage="prior-logical-digest",
                               expected_lines=_expected_dump_lines(source_rows)) != source_digest:
                return None
        finally:
            probe.close()
        if _hash_file(database, stage="prior-file-hash") != manifest.get("database"):
            return None
    except (OSError, sqlite3.Error, ValueError):
        return None
    return manifest


def _verify_staged(staged, source_rows, source_digest):
    """Verify a staged copy's rows and logical content against the source.

    ``backup_to`` already owns the integrity/foreign-key check.
    """
    probe = sqlite3.connect(staged)
    try:
        staged_rows = _table_row_counts(probe, stage="staged-row-counts")
        if staged_rows != source_rows:
            differing = [name for name in sorted(set(source_rows) | set(staged_rows))
                         if source_rows.get(name) != staged_rows.get(name)]
            raise fail("INTEGRITY_FAILED", "backup row counts differ from source",
                       details={"tables": differing})
        staged_digest = _logical_digest(probe, stage="staged-logical-digest",
                                        expected_lines=_expected_dump_lines(source_rows))
    finally:
        probe.close()
    if staged_digest != source_digest:
        raise fail("INTEGRITY_FAILED", "backup content digest differs from source",
                   details={"reason": "content_digest"})


@contextmanager
def _backup_lock(target, key):
    """Serialize overlapping attempts for one target/key while they inspect,
    install and publish that key's database and manifest."""
    path = target / f".{key}.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _remove_stale_stages(target, key):
    """Remove crash-leftover staging databases for this key only."""
    prefix = _stage_prefix(key)
    suffix = ".staging.sqlite"
    for entry in target.iterdir():
        if entry.name.startswith(prefix) and entry.name.endswith(suffix):
            try:
                entry.unlink()
            except FileNotFoundError:
                pass


def _stage_copy(conn, target, key):
    """Write the source's current snapshot into a fresh staging database."""
    stage_fd, stage_name = tempfile.mkstemp(
        dir=str(target), prefix=_stage_prefix(key), suffix=".staging.sqlite")
    os.close(stage_fd)
    os.unlink(stage_name)
    staged = Path(stage_name)
    progress = _Progress("sqlite-backup-pages")
    try:
        progress.start()

        def page_progress(_status, remaining, total):
            progress.total = total
            progress.count = max(total - remaining, 0)
            progress.advance(0)

        backup_to(conn, staged, progress=page_progress)
    except BaseException:
        _discard_staged(staged)
        raise
    finally:
        progress.end()
    return staged


def _discard_staged(staged):
    try:
        staged.unlink()
    except FileNotFoundError:
        pass


def _unique_database_name(target, key):
    for _ in range(10):
        name = f"{key}.{uuid.uuid4().hex}.sqlite"
        if not (target / name).exists():
            return name
    raise fail("WORKER_FAILED", "could not allocate a unique backup database name")


def _install_stage(staged, target, key, source_rows, source_digest):
    """Verify and durably install a staged copy under a unique immutable name.

    The currently published database path is never replaced: a crash before the
    manifest replace leaves the old manifest naming the old intact database,
    while the newly installed database is simply unreferenced."""
    try:
        _verify_staged(staged, source_rows, source_digest)
        with staged.open("rb") as stream:
            os.fsync(stream.fileno())
        name = _unique_database_name(target, key)
        installed = target / name
        os.replace(staged, installed)
        staged = None
        fsync_directory(target)
        return name
    finally:
        if staged is not None:
            _discard_staged(staged)


def _copy_artifact(store, target, name, ref, fault):
    """Copy one artifact durably, verify its published hash and run the
    per-artifact fault hook before publication."""
    source = store.verify(ref)
    destination = target / "artifacts" / name
    _stream_copy(source, destination, expected_hash=ref.content_hash,
                 expected_size=ref.byte_size, stage=f"artifact {name}", fault=fault)
    if _hash_file(destination, stage=f"artifact-hash {name}") != ref.content_hash:
        raise fail("INTEGRITY_FAILED", "backup artifact hash mismatch")
    return {"content_hash": ref.content_hash, "byte_size": ref.byte_size}


def _copy_artifacts(payload, store, target, fault, keepalive):
    """Copy every payload artifact into the target and return their manifest entries."""
    copied = {}
    for name, document in payload.get("artifacts", {}).items():
        keepalive()
        ref = from_document(ArtifactRef, document)
        copied[name] = _copy_artifact(store, target, name, ref, fault)
    return copied


def _validated_artifact_names(payload):
    """Validate every artifact name before any database is installed."""
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise fail("INVALID_REQUEST", "unsafe backup artifact name")
    for name in artifacts:
        try:
            _single_filename(name)
        except ValueError:
            raise fail("INVALID_REQUEST", "unsafe backup artifact name") from None
    return artifacts


def _publish_manifest(manifest_path, manifest, prior):
    """Match a prior manifest, or atomically write a new one with a directory fsync."""
    if prior is not None:
        if prior != manifest:
            raise fail("INTEGRITY_FAILED", "backup manifest differs")
        return
    fd, name = tempfile.mkstemp(
        dir=str(manifest_path.parent), prefix=f".{manifest_path.name}.",
        suffix=".pending")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
        temporary = None
        fsync_directory(manifest_path.parent)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def run_backup(conn, *, key, owner, target, clock, store, fault=None, keepalive=_no_keepalive):
    effect = claim(conn, "backup", owner=owner, clock=clock, logical_key=key)
    if effect is None:
        return _delivered_or_stale(conn, "backup", key)
    payload = json.loads(effect["payload_json"])
    target = Path(target)
    if target.exists() and target.is_symlink():
        raise fail("INVALID_REQUEST", "backup target is a symlink")
    target.mkdir(parents=True, exist_ok=True)
    _validated_artifact_names(payload)
    manifest_path = target / (key + ".manifest.json")
    started = time.monotonic()
    _log(f"run_backup {key}: start target={target} rss_mb={_rss_mb()}")
    with _backup_lock(target, key):
        _remove_stale_stages(target, key)
        staged = None
        try:
            with _read_snapshot(conn):
                source_rows = _table_row_counts(conn, stage="source-row-counts")
                source_digest = _logical_digest(
                    conn, stage="source-logical-digest",
                    expected_lines=_expected_dump_lines(source_rows))
                prior = _verified_prior_backup(
                    manifest_path, target, source_rows, source_digest, key, payload)
                if prior is None:
                    _log(f"run_backup {key}: prior invalid or absent, rebuilding")
                    staged = _stage_copy(conn, target, key)
                else:
                    _log(f"run_backup {key}: reusing verified prior database "
                         f"{prior['database_file']}")
        except BaseException:
            if staged is not None:
                _discard_staged(staged)
            raise
        if staged is not None:
            database_file = _install_stage(staged, target, key, source_rows, source_digest)
        else:
            database_file = Path(prior["database_file"]).name
        database = target / database_file
        copied = _copy_artifacts(payload, store, target, fault, keepalive)
        manifest = {"schema_version": payload["schema_version"], "backup_key": key,
                    "cutoff": payload["cutoff"], "database": _hash_file(
                        database, stage="database-file-hash"),
                    "database_file": database_file,
                    "database_content_hash": source_digest,
                    "database_rows": source_rows, "artifacts": copied}
        _publish_manifest(manifest_path, manifest, prior)
    if fault:
        fault("after_manifest_before_ack")
    _log(f"run_backup {key}: complete database={manifest['database_file']} "
         f"elapsed={time.monotonic() - started:.1f}s rss_mb={_rss_mb()}")
    complete(conn, effect["effect_id"], manifest, owner=owner,
             claim_token=effect["claim_token"], clock=clock)
    return manifest


def _read_restore_manifest(manifest_path):
    """Parse one backup manifest as typed INTEGRITY_FAILED when unreadable."""
    from engine.v2.ops.workflows.commands import read_bounded_bytes

    try:
        manifest = json.loads(read_bounded_bytes(manifest_path, MAX_MANIFEST_BYTES))
    except (OSError, ValueError, RecursionError) as exc:
        raise fail("INTEGRITY_FAILED", "backup manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise fail("INTEGRITY_FAILED", "backup manifest is unreadable")
    return manifest


def _validated_restore_manifest(manifest):
    """Validate manifest shape and every filename before writing anything."""
    try:
        artifacts = manifest["artifacts"]
        if not isinstance(artifacts, dict):
            raise ValueError("artifacts is not a mapping")
        for name, info in artifacts.items():
            _single_filename(name)
            if not isinstance(info, dict) or not isinstance(info.get("content_hash"), str):
                raise ValueError("artifact metadata is invalid")
        if not isinstance(manifest["database"], str):
            raise ValueError("database hash is invalid")
        database = manifest_database_path(Path("/"), manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise fail("INTEGRITY_FAILED", "backup manifest is invalid") from exc
    return database.name, artifacts


def restore_backup(source, destination):
    """Restore into a new isolated root and verify every database/artifact byte."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise fail("INVALID_REQUEST", "restore destination already exists")
    manifests = list(source.glob("*.manifest.json"))
    if len(manifests) != 1:
        raise fail("INTEGRITY_FAILED", "backup manifest is missing or ambiguous")
    manifest = _read_restore_manifest(manifests[0])
    database_name, artifacts = _validated_restore_manifest(manifest)
    started = time.monotonic()
    _log(f"restore_backup: start source={source} database={database_name} "
         f"rss_mb={_rss_mb()}")
    destination.mkdir(parents=True)
    database = destination / "ops.sqlite"
    _stream_copy(source / database_name, database,
                 expected_hash=manifest["database"], stage="restore database")
    probe = sqlite3.connect(database)
    try:
        if integrity_errors(probe):
            raise fail("INTEGRITY_FAILED", "restored catalog failed verification")
    finally:
        probe.close()
    for name, info in artifacts.items():
        path = destination / "artifacts" / name
        expected_size = info.get("byte_size")
        _stream_copy(source / "artifacts" / name, path,
                     expected_hash=info["content_hash"],
                     expected_size=expected_size if isinstance(expected_size, int) else None,
                     stage=f"restore artifact {name}")
    _log(f"restore_backup: complete destination={destination} "
         f"elapsed={time.monotonic() - started:.1f}s rss_mb={_rss_mb()}")
    return destination
