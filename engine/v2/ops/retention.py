from __future__ import annotations

import json
import os
import resource
import sqlite3
import stat
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from engine.v2.foundation import safe_relative_path
from engine.v2.ops.catalog import decode_bounded_text, integrity_errors, sql_text_budget
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.fingerprints import file_hash
from engine.v2.ops.reclaim import remove_tree
from engine.v2.ops.snapshot_roots import default_materialization_base, materialization_root
from engine.v2.ops.snapshot_stages import request_from_artifact

KEEP_ENV, DEFAULT_KEEP = "OPS_BACKUP_RETENTION_KEEP", 7
MATERIALIZATION_KEEP_ENV, DEFAULT_MATERIALIZATION_KEEP = "OPS_MATERIALIZATION_RETENTION_KEEP", 11
_SCHEMA, _DB, _MANIFEST = "catalog_backup.v1.0", ".sqlite", ".manifest.json"
_MANIFEST_KEYS = frozenset({"schema_version", "backup_key", "cutoff", "database", "artifacts"})
_ARTIFACT_KEYS = frozenset({"content_hash", "byte_size"})
_LIVE_STATES = ("starting", "running", "cancelling", "recovery_pending")
_REQUEST_BINDING = "materialization_request.json"
_MAX_MANIFEST_BYTES = 10 * 1024 * 1024
MAX_BACKUP_STREAM_BYTES = 4 * 1024 * 1024 * 1024
MAX_LIVE_SPEC_BYTES = 1024 * 1024
_INTEGRITY_PROGRESS_INTERVAL = 1000

def _stderr(event, **fields):
    sys.stderr.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")

def _rss_peak_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)

def _stage(name, total):
    _stderr("RETENTION_STAGE_START", stage=name, total=total, rss_peak_mb=_rss_peak_mb())
    return time.monotonic()

def _stage_end(name, started, done, total):
    _stderr("RETENTION_STAGE_END", stage=name, done=done, total=total,
            elapsed_s=round(time.monotonic() - started, 3), rss_peak_mb=_rss_peak_mb())

class _Progress:
    def __init__(self, stage, total, interval=60.0):
        self.stage, self.total = stage, total
        self.interval = interval
        self.started = self.last = time.monotonic()
    def tick(self, done):
        now = time.monotonic()
        if now - self.last < self.interval:
            return
        elapsed = now - self.started
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 and self.total > done else None
        self.last = now
        _stderr("RETENTION_PROGRESS", stage=self.stage, done=done, total=self.total,
                rate_per_s=round(rate, 3),
                eta_s=None if eta is None else round(eta, 1), rss_peak_mb=_rss_peak_mb())

def _renew(keepalive):
    keepalive() if keepalive is not None else None

def _limit(keep, env, default, label):
    if keep is None:
        raw = os.environ.get(env)
        try:
            keep = int(str(default) if raw is None else raw.strip())
        except ValueError:
            raise fail("INVALID_REQUEST", f"{label} keep must be a positive integer",
                       details={"reason_key": f"{label}_keep_invalid"}) from None
    if isinstance(keep, bool) or not isinstance(keep, int) or keep <= 0:
        raise fail("INVALID_REQUEST", f"{label} keep must be a positive integer",
                   details={"reason_key": f"{label}_keep_invalid"})
    return keep

def _safe_name(name, *, artifact=False):
    bad = not isinstance(name, str) or not name or name in (".", "..")
    if artifact:
        bad = bad or "/" in name or "\\" in name
    elif not bad:
        bad = len(safe_relative_path(name)) != 1
    if bad:
        reason = "backup_artifact_name_unsafe" if artifact else "backup_name_unsafe"
        raise fail("INTEGRITY_FAILED", "backup name is unsafe",
                   details={"reason_key": reason})
    return name

def _digest(path, keepalive=None):
    try:
        total = path.stat().st_size
        started = time.monotonic()
        last = started

        def on_progress(bytes_read):
            nonlocal last
            now = time.monotonic()
            if now - last < 60.0:
                return
            last = now
            elapsed = now - started
            rate = bytes_read / elapsed if elapsed > 0 else 0.0
            eta = (total - bytes_read) / rate if rate > 0 and total > bytes_read else None
            _stderr("RETENTION_PROGRESS", stage="verify_backup_file", done=bytes_read,
                    total=total, bytes_per_s=round(rate, 3),
                    eta_s=None if eta is None else round(eta, 1),
                    rss_peak_mb=_rss_peak_mb())

        return file_hash(path, keepalive=keepalive, max_bytes=MAX_BACKUP_STREAM_BYTES,
                         on_progress=on_progress)
    except OSError:
        raise fail("INTEGRITY_FAILED", "backup file could not be read",
                   details={"reason_key": "backup_file_unreadable"}) from None

def _database(path, expected, keepalive=None):
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
                     | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        raise fail("INTEGRITY_FAILED", "backup database is missing, empty or not a regular file",
                   details={"reason_key": "backup_database_invalid"}) from None
    try:
        pinned = os.fstat(fd)
        if not stat.S_ISREG(pinned.st_mode) or pinned.st_size <= 0:
            raise fail("INTEGRITY_FAILED", "backup database is missing, empty or not a regular file",
                       details={"reason_key": "backup_database_invalid"})
        proc_fd = Path(f"/proc/self/fd/{fd}")
        if _digest(proc_fd, keepalive) != expected:
            raise fail("INTEGRITY_FAILED", "backup database hash does not match its manifest",
                       details={"reason_key": "backup_database_hash_mismatch"})
        callback_error = None

        def _progress():
            nonlocal callback_error
            try:
                keepalive()
            except Exception as error:
                callback_error = error
                return 1
            return 0

        try:
            conn = sqlite3.connect(proc_fd.as_uri() + "?mode=ro&immutable=1", uri=True)
            try:
                if keepalive is not None:
                    conn.set_progress_handler(_progress, _INTEGRITY_PROGRESS_INTERVAL)
                problems = integrity_errors(conn)
            finally:
                try:
                    conn.set_progress_handler(None, 0)
                finally:
                    conn.close()
        except sqlite3.Error:
            if callback_error is not None:
                raise callback_error from None
            raise fail("INTEGRITY_FAILED", "backup database could not be verified",
                       details={"reason_key": "backup_database_verify_failed"}) from None
        if callback_error is not None:
            raise callback_error from None
        if problems:
            raise fail("INTEGRITY_FAILED", "backup database failed integrity or foreign-key checks",
                       details={"reason_key": "backup_database_integrity_failed"})
        try:
            current = os.lstat(path)
        except OSError:
            current = None
        if current is None or (current.st_dev, current.st_ino) != (pinned.st_dev, pinned.st_ino):
            raise fail("INTEGRITY_FAILED", "backup database changed during verification",
                       details={"reason_key": "backup_database_changed"})
    finally:
        os.close(fd)

def _artifact(directory, name, info, keepalive=None):
    _safe_name(name, artifact=True)
    if not isinstance(info, dict) or set(info) != _ARTIFACT_KEYS:
        raise fail("INTEGRITY_FAILED", "backup artifact metadata has unexpected keys",
                   details={"reason_key": "backup_artifact_metadata_keys"})
    digest, size = info["content_hash"], info["byte_size"]
    if not isinstance(digest, str) or isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise fail("INTEGRITY_FAILED", "backup artifact metadata is invalid",
                   details={"reason_key": "backup_artifact_metadata_invalid"})
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise fail("INTEGRITY_FAILED", "backup artifact is invalid",
                   details={"reason_key": "backup_artifact_invalid"})
    if path.stat().st_size != size:
        raise fail("INTEGRITY_FAILED", "backup artifact size does not match its manifest",
                   details={"reason_key": "backup_artifact_size_mismatch"})
    if _digest(path, keepalive) != digest:
        raise fail("INTEGRITY_FAILED", "backup artifact hash does not match its manifest",
                   details={"reason_key": "backup_artifact_hash_mismatch"})

def _manifest(key, path, database, artifacts_dir, keepalive=None):
    _renew(keepalive)
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_MANIFEST_BYTES + 1)
        if len(raw) > _MAX_MANIFEST_BYTES:
            raise fail("INTEGRITY_FAILED", "backup manifest is too large",
                       details={"reason_key": "backup_manifest_too_large"})
        document = json.loads(raw)
    except (OSError, ValueError, RecursionError):
        raise fail("INTEGRITY_FAILED", "backup manifest could not be read",
                   details={"reason_key": "backup_manifest_unreadable"}) from None
    if not isinstance(document, dict) or set(document) != _MANIFEST_KEYS:
        raise fail("INTEGRITY_FAILED", "backup manifest has unexpected keys",
                   details={"reason_key": "backup_manifest_keys"})
    if document["schema_version"] != _SCHEMA or document["backup_key"] != key:
        raise fail("INTEGRITY_FAILED", "invalid backup manifest",
                   details={"reason_key": "backup_manifest_invalid"})
    try:
        when = datetime.fromisoformat(document["cutoff"])
    except (TypeError, ValueError):
        when = None
    if when is None or when.tzinfo is None or when.utcoffset() is None:
        raise fail("INTEGRITY_FAILED", "backup cutoff is not a timezone-aware ISO timestamp",
                   details={"reason_key": "backup_cutoff_invalid"})
    _database(database, document["database"], keepalive)
    artifacts = document["artifacts"]
    if not isinstance(artifacts, dict):
        raise fail("INTEGRITY_FAILED", "backup artifact metadata is not a mapping",
                   details={"reason_key": "backup_artifacts_not_mapping"})
    for name, info in artifacts.items():
        _artifact(artifacts_dir, name, info, keepalive)
    return {"key": key, "when": when, "artifacts": set(artifacts)}

def _scope_files(scope, keepalive=None):
    manifests, databases = {}, {}
    entries = []
    for path in scope.iterdir():
        _renew(keepalive)
        entries.append(path)
    for path in sorted(entries, key=lambda item: item.name):
        _renew(keepalive)
        if path.name == "artifacts":
            continue
        if path.is_symlink() or not path.is_file():
            raise fail("INTEGRITY_FAILED", "backup scope contains a stray entry",
                       details={"reason_key": "backup_scope_stray_entry"})
        if path.name.endswith(_MANIFEST):
            manifests[_safe_name(path.name[:-len(_MANIFEST)])] = path
        elif path.name.endswith(_DB):
            databases[_safe_name(path.name[:-len(_DB)])] = path
        else:
            raise fail("INTEGRITY_FAILED", "backup scope contains a stray file",
                       details={"reason_key": "backup_scope_stray_file"})
    return manifests, databases

def _present_artifacts(artifacts_dir, keepalive=None):
    present = set()
    if artifacts_dir.exists():
        for path in artifacts_dir.iterdir():
            _renew(keepalive)
            if path.is_symlink() or not path.is_file():
                raise fail("INTEGRITY_FAILED", "stray backup artifact",
                           details={"reason_key": "backup_artifact_stray"})
            present.add(path.name)
    return present

def _scope_artifacts_dir(scope):
    artifacts_dir = scope / "artifacts"
    if artifacts_dir.is_symlink() or (artifacts_dir.exists() and not artifacts_dir.is_dir()):
        raise fail("INTEGRITY_FAILED", "backup artifacts path is not a directory",
                   details={"reason_key": "backup_artifacts_path_invalid"})
    return artifacts_dir

def _scope_records(manifests, databases, artifacts_dir, keepalive=None):
    if set(manifests) != set(databases):
        raise fail("INTEGRITY_FAILED", "incomplete backup",
                   details={"reason_key": "backup_incomplete"})
    records = []
    for key in sorted(manifests):
        try:
            records.append(
                _manifest(key, manifests[key], databases[key], artifacts_dir, keepalive))
        except OpsError as error:
            problem = error.problem
            raise OpsError(replace(problem, details={**problem.details,
                                                     "backup_key": key})) from None
    return records

def _read_scope(scope, keepalive=None):
    artifacts_dir = _scope_artifacts_dir(scope)
    manifests, databases = _scope_files(scope, keepalive)
    records = _scope_records(manifests, databases, artifacts_dir, keepalive)
    referenced = set().union(*(record["artifacts"] for record in records)) if records else set()
    present = _present_artifacts(artifacts_dir, keepalive)
    if present != referenced:
        raise fail("INTEGRITY_FAILED", "backup artifacts are stray or incomplete",
                   details={"reason_key": "backup_artifacts_mismatch"})
    return records, artifacts_dir, present

def _plan(scope, keep, keepalive=None):
    records, artifacts_dir, present = _read_scope(scope, keepalive)
    ordered = sorted(records, key=lambda r: (r["when"], r["key"]), reverse=True)
    retained = ordered[:keep]
    kept = set().union(*(r["artifacts"] for r in retained)) if retained else set()
    return scope, ordered[keep:], artifacts_dir, present - kept

def _live_spec_too_large(attempt_id, spec_bytes):
    return fail("INTEGRITY_FAILED", "live job spec exceeds its byte bound",
                details={"reason_key": "live_job_spec_too_large",
                         "attempt_id": attempt_id,
                         "spec_bytes": spec_bytes,
                         "byte_limit": MAX_LIVE_SPEC_BYTES})

def _live_spec_malformed(attempt_id):
    return fail("INTEGRITY_FAILED", "live job spec is malformed",
                details={"reason_key": "live_job_spec_malformed",
                         "attempt_id": attempt_id})

def _live_spec(attempt_id, spec_bytes, spec_json, encoding, limit):
    if spec_json is None:
        if spec_bytes is not None and spec_bytes > limit:
            raise _live_spec_too_large(attempt_id, spec_bytes)
        raise _live_spec_malformed(attempt_id)
    try:
        text = decode_bounded_text(spec_json, encoding, MAX_LIVE_SPEC_BYTES)
    except UnicodeError:
        text = None
    except ValueError:
        raise _live_spec_too_large(attempt_id, spec_bytes) from None
    try:
        spec = json.loads(text) if text is not None else None
    except (TypeError, ValueError, RecursionError):
        spec = None
    if not isinstance(spec, dict):
        raise _live_spec_malformed(attempt_id)
    return spec

def _live_roots(base, conn, store, keepalive=None):
    encoding, limit = sql_text_budget(conn, MAX_LIVE_SPEC_BYTES)
    _renew(keepalive)
    total = conn.execute(
        "SELECT COUNT(*) FROM attempts a JOIN jobs j ON j.job_id = a.job_id "
        "WHERE a.state IN (?,?,?,?)", _LIVE_STATES).fetchone()[0]
    _renew(keepalive)
    cursor = conn.execute(
        "SELECT a.attempt_id, j.kind, length(CAST(j.spec_json AS BLOB)) AS spec_bytes, "
        "CASE WHEN typeof(j.spec_json)='text' AND length(CAST(j.spec_json AS BLOB))<=? "
        "THEN CAST(j.spec_json AS BLOB) END AS spec_json, "
        "EXISTS(SELECT 1 FROM attempt_input_bindings b "
        "WHERE b.attempt_id = a.attempt_id AND b.name = ?) AS bound "
        "FROM attempts a JOIN jobs j ON j.job_id = a.job_id "
        "WHERE a.state IN (?,?,?,?)", (limit, _REQUEST_BINDING, *_LIVE_STATES))
    started = _stage("resolve_live_roots", total)
    progress = _Progress("resolve_live_roots", total)
    done = 0
    try:
        active = []
        for attempt_id, kind, spec_bytes, spec_json, bound in cursor:
            _renew(keepalive)
            spec = _live_spec(attempt_id, spec_bytes, spec_json, encoding, limit)
            parameters = spec["parameters"] if "parameters" in spec else {}
            if not isinstance(parameters, dict):
                raise fail("INTEGRITY_FAILED", "live job parameters are malformed",
                           details={"reason_key": "live_job_parameters_malformed",
                                    "attempt_id": attempt_id})
            if kind != "legacy_materialize" \
                    and spec.get("kind") != "legacy_materialize" \
                    and parameters.get("input_mode") != "snapshot" \
                    and not bound:
                done += 1
                progress.tick(done)
                continue
            active.append(attempt_id)
            done += 1
            progress.tick(done)
        if not active:
            return set()
        roots = set()
        for attempt_id in active:
            _renew(keepalive)
            bound = conn.execute(
                "SELECT artifact_id FROM attempt_input_bindings "
                "WHERE attempt_id = ? AND name = ?",
                (attempt_id, _REQUEST_BINDING)).fetchall()
            if len(bound) != 1:
                raise fail("INPUT_CHANGED",
                           "live materialization attempt has no single request binding",
                           details={"reason_key": "live_request_binding_invalid",
                                    "attempt_id": attempt_id})
            request = request_from_artifact(conn, store, bound[0][0])
            _renew(keepalive)
            roots.add(materialization_root(base, request.request_hash).name)
        return roots
    finally:
        _stage_end("resolve_live_roots", started, done, total)

def _materialization_plan(ops_root, keep, conn, store, base=None, keepalive=None):
    base = base or default_materialization_base(ops_root)
    if base.is_symlink() or (base.exists() and not base.is_dir()):
        raise fail("INTEGRITY_FAILED", "materialization root is not a directory",
                   details={"reason_key": "materialization_root_invalid"})
    if not base.exists():
        return []
    _stderr("RETENTION_CURRENT_PUBLICATION_UNBOUND")
    all_entries = []
    for item in base.iterdir():
        _renew(keepalive)
        all_entries.append(item)
    started = _stage("verify_materialization", len(all_entries))
    progress = _Progress("verify_materialization", len(all_entries))
    done = 0
    try:
        entries = []
        for item in all_entries:
            _renew(keepalive)
            if item.name.startswith("."):
                done += 1
                progress.tick(done)
                continue
            if item.is_symlink() or not item.is_dir():
                raise fail("INTEGRITY_FAILED", "materialization entry is not a directory",
                           details={"reason_key": "materialization_entry_invalid"})
            entries.append(item)
            done += 1
            progress.tick(done)

        def _mtime(item):
            _renew(keepalive)
            return item.stat().st_mtime_ns

        ordered = sorted(entries, key=_mtime, reverse=True)
    finally:
        _stage_end("verify_materialization", started, done, len(all_entries))
    live = _live_roots(base, conn, store, keepalive)
    retained = {item.name for item in ordered[:keep]} | live
    return [item for item in ordered if item.name not in retained]

def _unlink(path):
    try:
        path.unlink()
    except OSError:
        raise fail("INTEGRITY_FAILED", "backup deletion failed",
                   details={"reason_key": "backup_delete_failed"}) from None

def _remove_tree(path, keepalive=None):
    """Remove one materialization tree with the shared reclaim walk.

    Raw ``OSError`` maps to materialization_delete_failed; an ``OpsError``
    raised by ``keepalive`` propagates unchanged."""
    try:
        remove_tree(path, lambda: _renew(keepalive))
    except OSError:
        raise fail("INTEGRITY_FAILED", "materialization deletion failed",
                   details={"reason_key": "materialization_delete_failed"}) from None

def enforce_retention(ops_root, *, conn, store, keep=None, materialization_keep=None, keepalive=None, materialization_base=None):
    """Keep verified backups and unreferenced roots; return deleted backup keys only."""
    try:
        return _enforce_retention(ops_root, conn=conn, store=store, keep=keep,
                                  materialization_keep=materialization_keep,
                                  keepalive=keepalive,
                                  materialization_base=materialization_base)
    except OSError:
        raise fail("INTEGRITY_FAILED", "retention filesystem verification failed",
                   details={"reason_key": "retention_io_failed"}) from None

def _enforce_retention(ops_root, *, conn, store, keep=None, materialization_keep=None, keepalive=None, materialization_base=None):
    keep = _limit(keep, KEEP_ENV, DEFAULT_KEEP, "backup")
    materialization_keep = _limit(materialization_keep, MATERIALIZATION_KEEP_ENV,
                                  DEFAULT_MATERIALIZATION_KEEP, "materialization")
    root, plans = Path(ops_root) / "backups", []
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise fail("INTEGRITY_FAILED", "backup root is not a directory",
                       details={"reason_key": "backup_root_invalid"})
        scopes = []
        for item in root.iterdir():
            _renew(keepalive)
            scopes.append(item)
        scopes.sort(key=lambda item: item.name)
        if any(scope.is_symlink() or not scope.is_dir() for scope in scopes):
            raise fail("INTEGRITY_FAILED", "backup scope is not a directory",
                       details={"reason_key": "backup_scope_not_directory"})
        started, progress, done = _stage("verify_backups", len(scopes)), _Progress("verify_backups", len(scopes)), 0
        try:
            for scope in scopes:
                plans.append(_plan(scope, keep, keepalive))
                progress.tick(done := done + 1)
        finally:
            _stage_end("verify_backups", started, done, len(scopes))
    roots = _materialization_plan(ops_root, materialization_keep, conn, store,
                                  base=materialization_base, keepalive=keepalive)
    deleted = []
    backup_total = sum(len(discarded) + len(artifact_names)
                       for _, discarded, _, artifact_names in plans)
    _stderr("RETENTION_DELETE_PLAN", backup_total=backup_total,
            materialization_total=len(roots), keep=keep,
            materialization_keep=materialization_keep, scopes=len(plans))
    started, progress, done = _stage("delete_backups", backup_total), _Progress("delete_backups", backup_total), 0
    try:
        for scope, discarded, artifacts_dir, artifact_names in plans:
            for record in discarded:
                _renew(keepalive)
                _unlink(scope / (record["key"] + _DB))
                _renew(keepalive)
                _unlink(scope / (record["key"] + _MANIFEST))
                deleted.append(record["key"])
                progress.tick(done := done + 1)
            for name in sorted(artifact_names):
                _renew(keepalive)
                _unlink(artifacts_dir / name)
                progress.tick(done := done + 1)
    finally:
        _stage_end("delete_backups", started, done, backup_total)
    started, progress, done = _stage("delete_materialization", len(roots)), _Progress("delete_materialization", len(roots)), 0
    try:
        for path in roots:
            _renew(keepalive)
            _remove_tree(path, keepalive)
            progress.tick(done := done + 1)
    finally:
        _stage_end("delete_materialization", started, done, len(roots))
    return deleted
