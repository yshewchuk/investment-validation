"""Durable local backup effect; remote mirroring is deliberately out of scope."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    ArtifactError,
    ensure_directory,
    format_timestamp,
    from_document,
    fsync_directory,
    safe_relative_path,
)
from engine.v2.ops.catalog import backup_to, integrity_errors, transaction
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.outbox import claim, complete, enqueue

#: Prior manifests are advisory metadata; reads beyond this bound are invalid.
MAX_MANIFEST_BYTES = 1024 * 1024

#: The largest database or artifact byte stream backup/restore will read.
MAX_BACKUP_FILE_BYTES = 1 << 40

#: The exact top-level keys a current manifest carries. Extra or missing keys
#: make prior metadata invalid (rebuild); they never become a refusal.
_MANIFEST_KEYS = frozenset({
    "schema_version", "backup_key", "cutoff", "database", "database_file",
    "database_content_hash", "database_rows", "database_size", "artifacts",
})

#: Keys every manifest carries, including legacy manifests that predate
#: ``database_file``, ``database_content_hash`` and ``database_size``.
_BASE_MANIFEST_KEYS = frozenset({
    "schema_version", "backup_key", "cutoff", "database", "database_rows",
    "artifacts",
})

_CHUNK = 1 << 20
_LOG_INTERVAL_SECONDS = 5.0
_HEARTBEAT_SECONDS = 30.0


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

    def __init__(self, stage, total=None, keepalive=None):
        self.stage = stage
        self.total = total
        self.count = 0
        self.started = None
        self.last = None
        self.keepalive = keepalive
        self.last_keepalive = None

    def start(self):
        self.started = self.last = time.monotonic()
        total = "?" if self.total is None else self.total
        _log(f"{self.stage}: start total={total} rss_mb={_rss_mb()}")

    def advance(self, amount=1):
        self.count += amount
        now = time.monotonic()
        if self.keepalive is not None and (
                self.last_keepalive is None
                or now - self.last_keepalive >= _HEARTBEAT_SECONDS):
            self.keepalive()
            self.last_keepalive = now
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
    try:
        parts = safe_relative_path(value)
    except ArtifactError:
        raise ValueError("not a safe single filename") from None
    if len(parts) != 1:
        raise ValueError("not a safe single filename")
    return parts[0]


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


_DIGEST_OUTBOX_INSERT = re.compile(r'^INSERT INTO (?:"outbox"|outbox) VALUES')
_DIGEST_STABLE_COLUMNS = ("effect_id", "kind", "logical_key", "payload_hash",
                          "payload_json")


def _full_logical_digest(conn, *, stage, expected_lines=None, keepalive=None):
    """SHA-256 over the whole logical SQL dump, including every outbox field.

    This is the digest a staged or prior database must match byte-for-byte at
    the logical level: it describes the database that was actually copied.
    """
    progress = _Progress(stage, total=expected_lines, keepalive=keepalive)
    digest = hashlib.sha256()
    progress.start()
    try:
        for line in conn.iterdump():
            digest.update(line.encode("utf-8"))
            progress.advance()
    finally:
        progress.end()
    return "sha256:" + digest.hexdigest()


def _retry_logical_digest(conn, key, *, stage, expected_lines=None, keepalive=None):
    """SHA-256 that ignores this backup effect's own volatile outbox state.

    A lost-ack retry changes the backup row's claim state, attempts, receipt,
    token, owner and lease. Only that row is reduced to its stable identity and
    payload fields; every other outbox row and the rest of the database stay
    fully represented, so unrelated source changes still invalidate a prior.
    """
    progress = _Progress(stage, total=expected_lines, keepalive=keepalive)
    digest = hashlib.sha256()
    progress.start()
    try:
        for line in conn.iterdump():
            if _DIGEST_OUTBOX_INSERT.match(line):
                continue
            digest.update(line.encode("utf-8"))
            progress.advance()
        columns = [row[1] for row in conn.execute("PRAGMA table_info(outbox)")]
        selected = ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
        for row in conn.execute(f"SELECT {selected} FROM outbox ORDER BY effect_id"):
            record = dict(zip(columns, row))
            if record.get("kind") == "backup" and record.get("logical_key") == key:
                projection = [record[name] for name in _DIGEST_STABLE_COLUMNS]
            else:
                projection = [record[name] for name in columns]
            digest.update(json.dumps(projection, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8") + b"\n")
            progress.advance()
    finally:
        progress.end()
    return "sha256:" + digest.hexdigest()


def _open_checked(path, *, stage, expected_size=None):
    """Open a regular non-symlink file within the byte cap; refuse otherwise."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise fail("INTEGRITY_FAILED", f"{stage} is not a regular file")
        size = info.st_size
        if size > MAX_BACKUP_FILE_BYTES:
            raise fail("INTEGRITY_FAILED", f"{stage} exceeds the file size limit")
        if expected_size is not None and size != expected_size:
            raise fail("INTEGRITY_FAILED", f"{stage} size mismatch")
        return os.fdopen(fd, "rb"), size
    except BaseException:
        os.close(fd)
        raise


def _hash_file(path, *, stage, keepalive=None):
    stream, size = _open_checked(path, stage=stage)
    digest = hashlib.sha256()
    progress = _Progress(stage, total=size, keepalive=keepalive)
    read = 0
    progress.start()
    try:
        while True:
            chunk = stream.read(min(_CHUNK, size - read + 1))
            if not chunk:
                break
            if len(chunk) > size - read:
                raise fail("INTEGRITY_FAILED", f"{stage} grew while hashing")
            digest.update(chunk)
            read += len(chunk)
            progress.advance(len(chunk))
        if read != size:
            raise fail("INTEGRITY_FAILED", f"{stage} shrank while hashing")
    finally:
        stream.close()
        progress.end()
    return "sha256:" + digest.hexdigest()


def _stream_copy(source, destination, *, expected_hash=None, expected_size=None,
                 stage, fault=None, keepalive=None):
    """Copy ``source`` to ``destination`` in bounded chunks, durably and atomically."""
    stream, size = _open_checked(source, stage=stage, expected_size=expected_size)
    try:
        ensure_directory(destination.parent)
        progress = _Progress(stage, total=size, keepalive=keepalive)
        fd, temporary_name = tempfile.mkstemp(
            dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".pending")
        temporary = Path(temporary_name)
        try:
            digest = hashlib.sha256()
            copied = 0
            with os.fdopen(fd, "wb") as out:
                progress.start()
                try:
                    while True:
                        chunk = stream.read(min(_CHUNK, size - copied + 1))
                        if not chunk:
                            break
                        if len(chunk) > size - copied:
                            raise fail("INTEGRITY_FAILED", f"{stage} grew while copying")
                        out.write(chunk)
                        digest.update(chunk)
                        copied += len(chunk)
                        progress.advance(len(chunk))
                    out.flush()
                    os.fsync(out.fileno())
                finally:
                    progress.end()
            if copied != size:
                raise fail("INTEGRITY_FAILED", f"{stage} shrank while copying")
            actual_hash = "sha256:" + digest.hexdigest()
            if expected_hash is not None and actual_hash != expected_hash:
                raise fail("INTEGRITY_FAILED", f"{stage} hash mismatch")
            if fault is not None:
                fault(destination.name)
            os.replace(temporary, destination)
            temporary = None
            fsync_directory(destination.parent)
            return copied
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


def _table_row_counts(conn, *, stage="row counts", keepalive=None):
    """Deterministic row count per non-internal table, keyed and ordered by name."""
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name")]
    progress = _Progress(stage, total=len(tables), keepalive=keepalive)
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


def _validated_artifacts_shape(artifacts):
    """Validate an artifact mapping (shared by prior admission and restore)."""
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts is not a mapping")
    for name, info in artifacts.items():
        _single_filename(name)
        if not isinstance(info, dict):
            raise ValueError("artifact metadata is invalid")
        if not isinstance(info.get("content_hash"), str):
            raise ValueError("artifact content hash is invalid")
        size = info.get("byte_size")
        if size is not None and (
                not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise ValueError("artifact byte size is invalid")
    return artifacts


def _validated_row_counts(rows):
    if not isinstance(rows, dict):
        raise ValueError("manifest row counts are invalid")
    for name, count in rows.items():
        if (not isinstance(name, str) or not isinstance(count, int)
                or isinstance(count, bool) or count < 0):
            raise ValueError("manifest row counts are invalid")
    return rows


def _validated_optional_manifest_fields(manifest):
    """Validate current-only fields when present; return the database name."""
    name = manifest.get("database_file")
    if name is not None:
        _single_filename(name)
    content_hash = manifest.get("database_content_hash")
    if content_hash is not None and not isinstance(content_hash, str):
        raise ValueError("manifest database content hash is invalid")
    size = manifest.get("database_size")
    if size is not None and (
            not isinstance(size, int) or isinstance(size, bool) or size < 0):
        raise ValueError("manifest database size is invalid")
    return name, size


def _validated_manifest_shape(manifest, *, current):
    """Validate a current or legacy manifest shape without doing I/O.

    ``current`` requires exactly the current key set. A legacy restore manifest
    may omit ``database_file``, ``database_content_hash`` and
    ``database_size``; it then reads ``<backup_key>.sqlite`` and hashes/sizes
    that file directly.
    """
    if not isinstance(manifest, dict):
        raise ValueError("manifest is not a mapping")
    keys = set(manifest)
    if current:
        if keys != _MANIFEST_KEYS:
            raise ValueError("manifest key set is not current")
    elif not _BASE_MANIFEST_KEYS <= keys <= _MANIFEST_KEYS:
        raise ValueError("manifest key set is not a legacy manifest")
    for name in ("schema_version", "backup_key", "cutoff", "database"):
        if not isinstance(manifest.get(name), str):
            raise ValueError(f"manifest {name} is invalid")
    _validated_row_counts(manifest.get("database_rows"))
    artifacts = _validated_artifacts_shape(manifest.get("artifacts"))
    name, size = _validated_optional_manifest_fields(manifest)
    if name is None:
        name = _single_filename(manifest["backup_key"]) + ".sqlite"
    return name, artifacts, size


def _expected_dump_lines(rows):
    return sum(rows.values()) + len(rows) + 2


def _verified_prior_backup(manifest_path, target, source_rows, source_retry_digest,
                           key, payload, keepalive=None):
    """The existing manifest when it and its named database are a verified prior
    result for this payload, else ``None``.

    The prior's full dump digest must equal its own recorded
    ``database_content_hash`` (the database that was stored), while its
    retry-stable outbox projection must equal the current source's. Comparing
    only the latter would let unrelated source changes reuse a stale database.
    A prior without the current key set or ``database_file`` /
    ``database_content_hash`` / ``database_size`` is legacy or malformed and is
    rebuilt, never refused."""
    manifest = _read_manifest(manifest_path)
    try:
        database_name, _artifacts, database_size = _validated_manifest_shape(
            manifest, current=True)
    except ValueError:
        return None
    if (manifest.get("backup_key") != key
            or manifest.get("schema_version") != payload["schema_version"]
            or manifest.get("cutoff") != payload["cutoff"]
            or manifest.get("database_rows") != source_rows):
        return None
    database = target / database_name
    try:
        stream, actual_size = _open_checked(database, stage="prior database")
        stream.close()
        if actual_size != database_size:
            return None
        probe = sqlite3.connect(database)
        try:
            if integrity_errors(probe):
                return None
            if _table_row_counts(probe, stage="prior-row-counts",
                                 keepalive=keepalive) != source_rows:
                return None
            if _full_logical_digest(
                    probe, stage="prior-logical-digest",
                    expected_lines=_expected_dump_lines(source_rows),
                    keepalive=keepalive) != manifest.get("database_content_hash"):
                return None
            if _retry_logical_digest(
                    probe, key, stage="prior-retry-digest",
                    expected_lines=_expected_dump_lines(source_rows),
                    keepalive=keepalive) != source_retry_digest:
                return None
        finally:
            probe.close()
        if _hash_file(database, stage="prior-file-hash",
                      keepalive=keepalive) != manifest.get("database"):
            return None
    except (OSError, OpsError, sqlite3.Error, ValueError):
        return None
    return manifest


def _verify_staged(staged, source_rows, source_digest, keepalive=None):
    """Verify a staged copy's rows and logical content against the source.

    ``backup_to`` already owns the integrity/foreign-key check.
    """
    probe = sqlite3.connect(staged)
    try:
        staged_rows = _table_row_counts(probe, stage="staged-row-counts",
                                        keepalive=keepalive)
        if staged_rows != source_rows:
            differing = [name for name in sorted(set(source_rows) | set(staged_rows))
                         if source_rows.get(name) != staged_rows.get(name)]
            raise fail("INTEGRITY_FAILED", "backup row counts differ from source",
                       details={"tables": differing})
        staged_digest = _full_logical_digest(
            probe, stage="staged-logical-digest",
            expected_lines=_expected_dump_lines(source_rows), keepalive=keepalive)
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


def _stage_copy(conn, target, key, keepalive=None):
    """Write the source's current snapshot into a fresh staging database."""
    stage_fd, stage_name = tempfile.mkstemp(
        dir=str(target), prefix=_stage_prefix(key), suffix=".staging.sqlite")
    os.close(stage_fd)
    os.unlink(stage_name)
    staged = Path(stage_name)
    progress = _Progress("sqlite-backup-pages", keepalive=keepalive)
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


def _install_stage(staged, target, key, source_rows, source_digest, keepalive=None):
    """Verify and durably install a staged copy under a unique immutable name.

    The currently published database path is never replaced: a crash before the
    manifest replace leaves the old manifest naming the old intact database,
    while the newly installed database is simply unreferenced."""
    try:
        _verify_staged(staged, source_rows, source_digest, keepalive=keepalive)
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


def _remove_superseded_databases(target, key, keep):
    """Remove only this key's unique SQLite files, keeping the published one."""
    removed = False
    for entry in target.iterdir():
        if entry.name == keep or not entry.name.startswith(key + "."):
            continue
        if re.fullmatch(r"\.[0-9a-f]{32}\.sqlite", entry.name[len(key):]) is None:
            continue
        try:
            entry.unlink()
            removed = True
        except FileNotFoundError:
            continue
        except OSError as exc:
            _log(f"superseded database cleanup failed for {entry.name}: {exc}")
    if removed:
        fsync_directory(target)


def _copy_artifact(store, target, name, ref, fault, keepalive=None):
    """Copy one artifact durably, verify its published hash and run the
    per-artifact fault hook before publication."""
    source = store.verify(ref)
    destination = target / "artifacts" / name
    _stream_copy(source, destination, expected_hash=ref.content_hash,
                 expected_size=ref.byte_size, stage=f"artifact {name}",
                 fault=fault, keepalive=keepalive)
    if _hash_file(destination, stage=f"artifact-hash {name}",
                  keepalive=keepalive) != ref.content_hash:
        raise fail("INTEGRITY_FAILED", "backup artifact hash mismatch")
    return {"content_hash": ref.content_hash, "byte_size": ref.byte_size}


def _copy_artifacts(payload, store, target, fault, keepalive):
    """Copy every payload artifact into the target and return their manifest entries."""
    copied = {}
    for name, document in payload.get("artifacts", {}).items():
        keepalive()
        ref = from_document(ArtifactRef, document)
        copied[name] = _copy_artifact(store, target, name, ref, fault, keepalive)
    return copied


def _validated_artifact_names(payload):
    """Validate every artifact name before any database is installed."""
    artifacts = payload.get("artifacts", {})
    try:
        _validated_artifacts_shape(artifacts)
    except ValueError:
        raise fail("INVALID_REQUEST", "unsafe backup artifact name") from None
    return artifacts


def _publish_manifest(manifest_path, manifest, prior):
    """Match a prior manifest, or atomically write a new one with a directory fsync."""
    if prior is not None:
        if prior != manifest:
            raise fail("INTEGRITY_FAILED", "backup manifest differs")
        return
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise fail("INTEGRITY_FAILED", "backup manifest is too large")
    ensure_directory(manifest_path.parent)
    fd, name = tempfile.mkstemp(
        dir=str(manifest_path.parent), prefix=f".{manifest_path.name}.",
        suffix=".pending")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
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


def _main_database_path(conn):
    for row in conn.execute("PRAGMA database_list"):
        if row[1] == "main" and row[2]:
            return row[2]
    return None


@contextmanager
def _snapshot(conn):
    """A read snapshot on a separate read-only connection when file-backed.

    The source connection stays free, so the coordinator keepalive can renew
    the attempt lease while page copy, counts, digests and prior verification
    run. An in-memory catalog has no second connection; its keepalive stays off
    because a lease write cannot share the open read transaction.
    """
    path = _main_database_path(conn)
    if path is None:
        with _read_snapshot(conn) as active:
            yield active, False
        return
    snapshot = sqlite3.connect(f"file:{quote(path, safe='/')}?mode=ro", uri=True)
    try:
        snapshot.row_factory = conn.row_factory
        with _read_snapshot(snapshot) as active:
            yield active, True
    finally:
        snapshot.close()


def run_backup(conn, *, key, owner, target, clock, store, fault=None, keepalive=_no_keepalive):
    effect = claim(conn, "backup", owner=owner, clock=clock, logical_key=key)
    if effect is None:
        return _delivered_or_stale(conn, "backup", key)
    payload = json.loads(effect["payload_json"])
    target = Path(target)
    if target.exists() and target.is_symlink():
        raise fail("INVALID_REQUEST", "backup target is a symlink")
    ensure_directory(target)
    _validated_artifact_names(payload)
    manifest_path = target / (key + ".manifest.json")
    started = time.monotonic()
    _log(f"run_backup {key}: start target={target} rss_mb={_rss_mb()}")
    with _backup_lock(target, key):
        _remove_stale_stages(target, key)
        staged = None
        try:
            with _snapshot(conn) as (snapshot, separate):
                heartbeat = keepalive if separate else None
                source_rows = _table_row_counts(
                    snapshot, stage="source-row-counts", keepalive=heartbeat)
                source_digest = _full_logical_digest(
                    snapshot, stage="source-logical-digest",
                    expected_lines=_expected_dump_lines(source_rows),
                    keepalive=heartbeat)
                source_retry_digest = _retry_logical_digest(
                    snapshot, key, stage="source-retry-digest",
                    expected_lines=_expected_dump_lines(source_rows),
                    keepalive=heartbeat)
                prior = _verified_prior_backup(
                    manifest_path, target, source_rows, source_retry_digest, key,
                    payload, keepalive=heartbeat)
                if prior is None:
                    _log(f"run_backup {key}: prior invalid or absent, rebuilding")
                    staged = _stage_copy(snapshot, target, key, keepalive=heartbeat)
                else:
                    _log(f"run_backup {key}: reusing verified prior database "
                         f"{prior['database_file']}")
        except BaseException:
            if staged is not None:
                _discard_staged(staged)
            raise
        if staged is not None:
            database_file = _install_stage(
                staged, target, key, source_rows, source_digest,
                keepalive=keepalive)
        else:
            database_file = Path(prior["database_file"]).name
        database = target / database_file
        copied = _copy_artifacts(payload, store, target, fault, keepalive)
        manifest = {"schema_version": payload["schema_version"], "backup_key": key,
                    "cutoff": payload["cutoff"], "database": _hash_file(
                        database, stage="database-file-hash", keepalive=keepalive),
                    "database_file": database_file,
                    "database_content_hash": (
                        prior["database_content_hash"] if prior is not None
                        else source_digest),
                    "database_rows": source_rows,
                    "database_size": database.stat().st_size,
                    "artifacts": copied}
        _publish_manifest(manifest_path, manifest, prior)
        _remove_superseded_databases(target, key, database_file)
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
        return _validated_manifest_shape(manifest, current=False)
    except ValueError as exc:
        raise fail("INTEGRITY_FAILED", "backup manifest is invalid") from exc


def restore_backup(source, destination):
    """Restore into a new isolated root and verify every database/artifact byte."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise fail("INVALID_REQUEST", "restore destination already exists")
    manifests = list(source.glob("*.manifest.json"))
    if len(manifests) != 1:
        raise fail("INTEGRITY_FAILED", "backup manifest is missing or ambiguous")
    manifest = _read_restore_manifest(manifests[0])
    database_name, artifacts, database_size = _validated_restore_manifest(manifest)
    source_database = manifest_database_path(source, manifest)
    started = time.monotonic()
    _log(f"restore_backup: start source={source} database={database_name} "
         f"rss_mb={_rss_mb()}")
    ensure_directory(destination)
    database = destination / "ops.sqlite"
    _stream_copy(source_database, database,
                 expected_hash=manifest["database"], expected_size=database_size,
                 stage="restore database")
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
