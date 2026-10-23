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
    CONTENT_HASH_PREFIX,
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

#: Keys every manifest carries, including the pre-change legacy schema that
#: predates ``database_rows``, ``database_file``, ``database_content_hash`` and
#: ``database_size``.
_BASE_MANIFEST_KEYS = frozenset({
    "schema_version", "backup_key", "cutoff", "database", "artifacts",
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

    def __init__(self, stage, total=None, keepalive=None, announce=True):
        self.stage = stage
        self.total = total
        self.count = 0
        self.started = None
        self.last = None
        self.keepalive = keepalive
        self.last_keepalive = None
        self.announce = announce

    def start(self):
        self.started = self.last = time.monotonic()
        if not self.announce:
            return
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
        if not self.announce:
            return
        now = time.monotonic()
        elapsed = max(now - self.started, 1e-9)
        rate = self.count / elapsed
        _log(f"{self.stage}: end count={self.count} elapsed={elapsed:.1f}s "
             f"rate={rate:.1f}/s rss_mb={_rss_mb()}")


def _single_filename(value):
    """A safe, plain single path component (no separators or traversal)."""
    if not isinstance(value, str):
        raise ValueError("not a safe single filename")
    try:
        parts = safe_relative_path(value)
    except ArtifactError:
        raise ValueError("not a safe single filename") from None
    if len(parts) != 1:
        raise ValueError("not a safe single filename")
    return parts[0]


def _validated_key(key):
    """Refuse a key that is not exactly one safe path component."""
    try:
        parts = safe_relative_path(key)
    except ArtifactError:
        raise fail("INVALID_REQUEST", "unsafe backup key") from None
    if len(parts) != 1:
        raise fail("INVALID_REQUEST", "unsafe backup key")
    return key


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


def _open_checked(path, *, stage, expected_size=None, oserror_code=None):
    """Open a regular non-symlink file within the byte cap; refuse otherwise."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        if oserror_code is not None:
            raise fail(oserror_code, f"{stage} is missing or unreadable") from None
        raise
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


def _hash_file(path, *, stage, keepalive=None, announce=True):
    stream, size = _open_checked(path, stage=stage)
    digest = hashlib.sha256()
    progress = _Progress(stage, total=size, keepalive=keepalive, announce=announce)
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
                 stage, fault=None, keepalive=None, oserror_code=None,
                 announce=True):
    """Copy ``source`` to ``destination`` in bounded chunks, durably and atomically."""
    stream, size = _open_checked(source, stage=stage, expected_size=expected_size,
                                 oserror_code=oserror_code)
    try:
        ensure_directory(destination.parent)
        progress = _Progress(stage, total=size, keepalive=keepalive, announce=announce)
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
    _validated_key(key)
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
    captured = []
    last_poll = []
    progress.start()
    if keepalive is not None:
        last_poll.append(time.monotonic())

        def poll_keepalive():
            try:
                now = time.monotonic()
                if now - last_poll[0] < _HEARTBEAT_SECONDS:
                    return 0
                keepalive()
                last_poll[0] = now
                return 0
            except BaseException as exc:
                captured.append(exc)
                return 1

        conn.set_progress_handler(poll_keepalive, 1000)
    try:
        for name in tables:
            quoted = '"' + name.replace('"', '""') + '"'
            try:
                counts[name] = conn.execute(
                    f"SELECT COUNT(1) FROM {quoted}").fetchone()[0]
            except BaseException:
                if not captured:
                    raise
            if captured:
                raise captured[0]
            progress.advance()
    finally:
        if keepalive is not None:
            conn.set_progress_handler(None, 0)
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


def _validated_artifacts_shape(artifacts, *, current):
    """Validate an artifact mapping (shared by prior admission and restore).

    A current manifest requires ``byte_size`` on every artifact. A legacy
    manifest may omit it, but a present value is validated the same way: only
    genuine absence is an omission, never a null value.
    """
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts is not a mapping")
    for name, info in artifacts.items():
        _single_filename(name)
        if not isinstance(info, dict):
            raise ValueError("artifact metadata is invalid")
        if not isinstance(info.get("content_hash"), str):
            raise ValueError("artifact content hash is invalid")
        size = info.get("byte_size")
        if "byte_size" not in info:
            if current:
                raise ValueError("artifact byte size is missing")
        elif size is None or (
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


def _validated_optional_manifest_fields(manifest, *, current):
    """Validate the current-only database fields; return the name and size.

    A current manifest must carry a valid non-null ``database_file``,
    ``database_content_hash`` and ``database_size``. A legacy manifest may
    omit any of them, but a present key is validated the same way: a null or
    invalid value is rejected, never treated as an omission.
    """
    name = manifest.get("database_file")
    if "database_file" in manifest:
        _single_filename(name)
    elif current:
        raise ValueError("manifest database file is missing")
    content_hash = manifest.get("database_content_hash")
    if "database_content_hash" in manifest:
        if not isinstance(content_hash, str):
            raise ValueError("manifest database content hash is invalid")
    elif current:
        raise ValueError("manifest database content hash is missing")
    size = manifest.get("database_size")
    if "database_size" in manifest:
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("manifest database size is invalid")
    elif current:
        raise ValueError("manifest database size is missing")
    return name, size


def _validated_manifest_shape(manifest, *, current):
    """Validate a current or legacy manifest shape without doing I/O.

    ``current`` requires exactly the current key set and valid non-null
    ``database_file``, ``database_content_hash`` and ``database_size`` fields.
    A legacy restore manifest may omit ``database_rows``, ``database_file``,
    ``database_content_hash`` and ``database_size``; it then reads
    ``<backup_key>.sqlite`` and hashes/sizes that file directly. A present
    optional field is validated even in a legacy manifest, so a null value is
    never mistaken for an omission.
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
    if "database_rows" in manifest:
        _validated_row_counts(manifest["database_rows"])
    artifacts = _validated_artifacts_shape(manifest.get("artifacts"), current=current)
    name, size = _validated_optional_manifest_fields(manifest, current=current)
    if name is None:
        name = _single_filename(manifest["backup_key"]) + ".sqlite"
    return name, artifacts, size


def _schema_statement_count(conn):
    """The number of index/view/trigger statements ``iterdump`` will emit."""
    return conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type IN ('index', 'view', 'trigger') "
        "AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%'").fetchone()[0]


def _sqlite_sequence_lines(conn):
    """The ``sqlite_sequence`` dump output: the DELETE statement when the
    internal table exists, plus one INSERT per sequence row (``_row_counts``
    excludes ``sqlite_%`` tables, so neither appears elsewhere)."""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'"
    ).fetchone()
    if exists is None:
        return 0
    return 1 + conn.execute("SELECT COUNT(*) FROM sqlite_sequence").fetchone()[0]


def _expected_dump_lines(rows, schema_statements=0, sequence_lines=0):
    """Expected logical-dump lines: BEGIN/COMMIT, CREATE TABLE per table,
    each index/view/trigger statement, the sqlite_sequence DELETE/INSERT
    output and one INSERT per row."""
    return sum(rows.values()) + len(rows) + schema_statements + sequence_lines + 2


def _verified_prior_backup(manifest_path, target, source_rows, source_retry_digest,
                           key, payload, expected_lines, keepalive=None):
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
    except (AttributeError, TypeError, ValueError):
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
                    expected_lines=expected_lines,
                    keepalive=keepalive) != manifest.get("database_content_hash"):
                return None
            if _retry_logical_digest(
                    probe, key, stage="prior-retry-digest",
                    expected_lines=expected_lines,
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


def _verify_staged(staged, source_rows, source_digest, expected_lines, keepalive=None):
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
            expected_lines=expected_lines, keepalive=keepalive)
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


def _refuse_oversized_snapshot(conn, *, stage="source database"):
    """Refuse a snapshot whose page allocation already exceeds the byte cap.

    ``backup_to`` copies and the database file hash/verification happen later,
    so read the snapshot's own ``page_count * page_size`` up front: a database
    that can never fit the cap is rejected non-retryably before any pages are
    copied.
    """
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    if page_count * page_size > MAX_BACKUP_FILE_BYTES:
        raise fail("INTEGRITY_FAILED", f"{stage} exceeds the file size limit")


def _stage_copy(conn, target, key, keepalive=None):
    """Write the source's current snapshot into a fresh staging database."""
    _refuse_oversized_snapshot(conn)
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


def _install_stage(staged, target, key, source_rows, source_digest, expected_lines,
                   keepalive=None):
    """Verify and durably install a staged copy under a unique immutable name.

    The currently published database path is never replaced: a crash before the
    manifest replace leaves the old manifest naming the old intact database,
    while the newly installed database is simply unreferenced."""
    try:
        _verify_staged(staged, source_rows, source_digest, expected_lines,
                       keepalive=keepalive)
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
        except OSError:
            _log(f"superseded database cleanup failed for {entry.name}")
    if removed:
        fsync_directory(target)


def _artifact_object_path(store, ref):
    """The validated object path ``ref`` names, without reading its bytes.

    Mirrors ``ArtifactStore._object_parts`` so the preflight opens exactly the
    object ``store.verify`` would: the storage key must be the store's
    ``objects/<hash prefix>/<hash>`` layout for the reference's own content hash.
    """
    try:
        parts = safe_relative_path(ref.storage_key)
    except ArtifactError:
        raise fail("INTEGRITY_FAILED", "backup artifact storage key is unsafe") from None
    digest = ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
    if (len(parts) != 3 or parts[0] != "objects"
            or parts[1] != digest[:2] or parts[2] != digest):
        raise fail("INTEGRITY_FAILED", "backup artifact storage key is invalid")
    return store.root.joinpath(*parts)


def _preflight_artifact(store, ref):
    """Refuse a backing object before ``ArtifactStore.verify`` scans it whole.

    Opens the object as a regular, no-follow file bounded by
    ``MAX_BACKUP_FILE_BYTES`` and matched to ``ref.byte_size``, so an oversized
    or size-mismatched object is rejected without a full content read;
    ``store.verify`` then re-hashes it for content-address integrity.
    """
    path = _artifact_object_path(store, ref)
    stream, _size = _open_checked(
        path, stage="artifact-preflight", expected_size=ref.byte_size,
        oserror_code="INTEGRITY_FAILED")
    stream.close()


def _copy_artifact(store, target, name, ref, fault, keepalive=None):
    """Copy one artifact durably, verify its published hash and run the
    per-artifact fault hook before publication."""
    _preflight_artifact(store, ref)
    source = store.verify(ref)
    destination = target / "artifacts" / name
    _stream_copy(source, destination, expected_hash=ref.content_hash,
                 expected_size=ref.byte_size, stage="artifact-copy",
                 fault=fault, keepalive=keepalive, announce=False)
    if _hash_file(destination, stage="artifact-hash", keepalive=keepalive,
                  announce=False) != ref.content_hash:
        raise fail("INTEGRITY_FAILED", "backup artifact hash mismatch")
    return {"content_hash": ref.content_hash, "byte_size": ref.byte_size}


def _copy_artifacts(refs, store, target, fault, keepalive):
    """Copy every parsed payload artifact into the target and return its entries.

    One bounded aggregate progress stage covers the whole artifact set, so a
    large set renews its lease between files without logging per file.
    """
    copied = {}
    progress = _Progress("artifact-copy", total=len(refs), keepalive=keepalive)
    progress.start()
    try:
        for name, ref in refs.items():
            copied[name] = _copy_artifact(store, target, name, ref, fault, keepalive)
            progress.advance()
    finally:
        progress.end()
    return copied


def _validated_artifact_names(payload):
    """Parse every payload artifact reference before any database is installed."""
    artifacts = payload.get("artifacts", {})
    try:
        _validated_artifacts_shape(artifacts, current=True)
        return {name: from_document(ArtifactRef, document)
                for name, document in artifacts.items()}
    except (ValueError, TypeError):
        raise fail("INVALID_REQUEST", "invalid backup artifact metadata") from None


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
    _validated_key(key)
    effect = claim(conn, "backup", owner=owner, clock=clock, logical_key=key)
    if effect is None:
        return _delivered_or_stale(conn, "backup", key)
    payload = json.loads(effect["payload_json"])
    artifact_refs = _validated_artifact_names(payload)
    target = Path(target)
    if target.exists() and target.is_symlink():
        raise fail("INVALID_REQUEST", "backup target is a symlink")
    ensure_directory(target)
    manifest_path = target / (key + ".manifest.json")
    started = time.monotonic()
    _log(f"run_backup {key}: start rss_mb={_rss_mb()}")
    with _backup_lock(target, key):
        _remove_stale_stages(target, key)
        staged = None
        try:
            with _snapshot(conn) as (snapshot, separate):
                heartbeat = keepalive if separate else None
                _refuse_oversized_snapshot(snapshot)
                source_rows = _table_row_counts(
                    snapshot, stage="source-row-counts", keepalive=heartbeat)
                expected_lines = _expected_dump_lines(
                    source_rows, _schema_statement_count(snapshot),
                    _sqlite_sequence_lines(snapshot))
                source_digest = _full_logical_digest(
                    snapshot, stage="source-logical-digest",
                    expected_lines=expected_lines,
                    keepalive=heartbeat)
                source_retry_digest = _retry_logical_digest(
                    snapshot, key, stage="source-retry-digest",
                    expected_lines=expected_lines,
                    keepalive=heartbeat)
                prior = _verified_prior_backup(
                    manifest_path, target, source_rows, source_retry_digest, key,
                    payload, expected_lines, keepalive=heartbeat)
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
                staged, target, key, source_rows, source_digest, expected_lines,
                keepalive=keepalive)
        else:
            database_file = Path(prior["database_file"]).name
        database = target / database_file
        copied = _copy_artifacts(artifact_refs, store, target, fault, keepalive)
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
    _log(f"run_backup {key}: complete elapsed={time.monotonic() - started:.1f}s "
         f"rss_mb={_rss_mb()}")
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
    _log(f"restore_backup: start database={database_name} rss_mb={_rss_mb()}")
    ensure_directory(destination)
    database = destination / "ops.sqlite"
    _stream_copy(source_database, database,
                 expected_hash=manifest["database"], expected_size=database_size,
                 stage="restore database", oserror_code="INTEGRITY_FAILED")
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
                     stage=f"restore artifact {name}",
                     oserror_code="INTEGRITY_FAILED")
    _log(f"restore_backup: complete elapsed={time.monotonic() - started:.1f}s "
         f"rss_mb={_rss_mb()}")
    return destination
