"""Prune verified catalog backups and unreferenced materialization roots.

A scope under ``ops_root/backups`` holds ``<key>.sqlite``, ``<key>.manifest.json``
and a shared ``artifacts/`` directory. Everything is verified before deletion;
any malformed, stray or incomplete file refuses the whole call. Roots beside ``ops_root`` are kept by mtime, plus every root a live attempt still binds."""
from __future__ import annotations

import json
import os
import resource
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

from engine.v2.foundation import safe_relative_path
from engine.v2.ops.catalog import integrity_errors
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import file_hash
from engine.v2.ops.snapshot_roots import default_materialization_base, materialization_root
from engine.v2.ops.snapshot_stages import request_from_artifact

KEEP_ENV, DEFAULT_KEEP = "OPS_BACKUP_RETENTION_KEEP", 7
MATERIALIZATION_KEEP_ENV, DEFAULT_MATERIALIZATION_KEEP = "OPS_MATERIALIZATION_RETENTION_KEEP", 11
_SCHEMA, _DB, _MANIFEST = "catalog_backup.v1.0", ".sqlite", ".manifest.json"
_MANIFEST_LIMIT, _FILE_LIMIT = 10 * 1024 * 1024, 4 * 1024**3
_MANIFEST_KEYS = frozenset({"schema_version", "backup_key", "cutoff", "database", "artifacts"})
_ARTIFACT_KEYS = frozenset({"content_hash", "byte_size"})
_LIVE_STATES = ("starting", "running", "cancelling", "recovery_pending")
_REQUEST_BINDING = "materialization_request.json"
_LIVE_ATTEMPT_LIMIT = 10_000
_LIVE_SPEC_LIMIT = 16 * 1024
_PROTECTED_KINDS = frozenset({"legacy_materialize", "legacy_finality", "legacy_features"})

def _stderr(event, **fields):
    sys.stderr.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")

def _no_keepalive():
    pass

def _rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)

def _stage(name, total):
    _stderr("RETENTION_STAGE_START", stage=name, total=total, rss_mb=_rss_mb())
    return time.monotonic()

def _stage_end(name, started, done, total):
    _stderr("RETENTION_STAGE_END", stage=name, done=done, total=total,
            elapsed_s=round(time.monotonic() - started, 3), rss_mb=_rss_mb())

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
                eta_s=None if eta is None else round(eta, 1), rss_mb=_rss_mb())

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

def _digest(path):
    try:
        return file_hash(path)
    except OSError:
        raise fail("INTEGRITY_FAILED", "backup file could not be read",
                   details={"reason_key": "backup_file_unreadable"}) from None

def _database(path, expected, keepalive=_no_keepalive):
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= _FILE_LIMIT:
        raise fail("INTEGRITY_FAILED", "backup database is missing, empty, oversized or not a regular file",
                   details={"reason_key": "backup_database_invalid"})
    keepalive()
    if _digest(path) != expected:
        raise fail("INTEGRITY_FAILED", "backup database hash does not match its manifest",
                   details={"reason_key": "backup_database_hash_mismatch"})
    keepalive()
    try:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            problems = integrity_errors(conn)
        finally:
            conn.close()
    except sqlite3.Error:
        raise fail("INTEGRITY_FAILED", "backup database could not be verified",
                   details={"reason_key": "backup_database_verify_failed"}) from None
    if problems:
        raise fail("INTEGRITY_FAILED", "backup database failed integrity or foreign-key checks",
                   details={"reason_key": "backup_database_integrity_failed"})

def _artifact(directory, name, info, keepalive=_no_keepalive):
    _safe_name(name, artifact=True)
    if not isinstance(info, dict) or set(info) != _ARTIFACT_KEYS:
        raise fail("INTEGRITY_FAILED", "backup artifact metadata has unexpected keys",
                   details={"reason_key": "backup_artifact_metadata_keys"})
    digest, size = info["content_hash"], info["byte_size"]
    if not isinstance(digest, str) or isinstance(size, bool) or not isinstance(size, int) or size < 0 or size > _FILE_LIMIT:
        raise fail("INTEGRITY_FAILED", "backup artifact metadata is invalid",
                   details={"reason_key": "backup_artifact_metadata_invalid"})
    path = directory / name
    if path.is_symlink() or not path.is_file():
        raise fail("INTEGRITY_FAILED", "backup artifact is invalid",
                   details={"reason_key": "backup_artifact_invalid"})
    if path.stat().st_size != size:
        raise fail("INTEGRITY_FAILED", "backup artifact size does not match its manifest",
                   details={"reason_key": "backup_artifact_size_mismatch"})
    keepalive()
    if _digest(path) != digest:
        raise fail("INTEGRITY_FAILED", "backup artifact hash does not match its manifest",
                   details={"reason_key": "backup_artifact_hash_mismatch"})

def _manifest(key, path, database, artifacts_dir, keepalive=_no_keepalive):
    try:
        if path.stat().st_size > _MANIFEST_LIMIT:
            raise fail("INTEGRITY_FAILED", "backup manifest exceeds the 10 MiB limit",
                       details={"reason_key": "backup_manifest_oversized"})
        document = json.loads(path.read_text())
    except (OSError, ValueError):
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
    _database(database, document["database"], keepalive=keepalive)
    artifacts = document["artifacts"]
    if not isinstance(artifacts, dict):
        raise fail("INTEGRITY_FAILED", "backup artifact metadata is not a mapping",
                   details={"reason_key": "backup_artifacts_not_mapping"})
    for name, info in artifacts.items():
        _artifact(artifacts_dir, name, info, keepalive=keepalive)
    return {"key": key, "when": when, "artifacts": set(artifacts)}

def _scope_files(scope):
    manifests, databases = {}, {}
    for path in sorted(scope.iterdir(), key=lambda item: item.name):
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

def _present_artifacts(artifacts_dir):
    present = set()
    if artifacts_dir.exists():
        for path in artifacts_dir.iterdir():
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

def _scope_records(manifests, databases, artifacts_dir, keepalive=_no_keepalive):
    if set(manifests) != set(databases):
        raise fail("INTEGRITY_FAILED", "incomplete backup",
                   details={"reason_key": "backup_incomplete"})
    records = []
    for key in sorted(manifests):
        keepalive()
        records.append(_manifest(key, manifests[key], databases[key], artifacts_dir,
                                 keepalive=keepalive))
    return records

def _read_scope(scope, keepalive=_no_keepalive):
    artifacts_dir = _scope_artifacts_dir(scope)
    manifests, databases = _scope_files(scope)
    records = _scope_records(manifests, databases, artifacts_dir, keepalive=keepalive)
    referenced = set().union(*(record["artifacts"] for record in records)) if records else set()
    present = _present_artifacts(artifacts_dir)
    if present != referenced:
        raise fail("INTEGRITY_FAILED", "backup artifacts are stray or incomplete",
                   details={"reason_key": "backup_artifacts_mismatch"})
    return records, artifacts_dir, present

def _plan(scope, keep, keepalive=_no_keepalive):
    records, artifacts_dir, present = _read_scope(scope, keepalive=keepalive)
    ordered = sorted(records, key=lambda r: (r["when"], r["key"]), reverse=True)
    retained = ordered[:keep]
    kept = set().union(*(r["artifacts"] for r in retained)) if retained else set()
    return scope, ordered[keep:], artifacts_dir, present - kept

def _live_roots(base, conn, store, keepalive=_no_keepalive):
    keepalive()
    rows = conn.execute(
        "SELECT a.attempt_id, j.kind, "
        "CASE WHEN length(CAST(j.spec_json AS BLOB)) <= ? THEN j.spec_json END, "
        "length(CAST(j.spec_json AS BLOB)) FROM attempts a "
        "JOIN jobs j ON j.job_id = a.job_id WHERE a.state IN (?,?,?,?) "
        "LIMIT ?",
        (_LIVE_SPEC_LIMIT, *_LIVE_STATES, _LIVE_ATTEMPT_LIMIT + 1)).fetchall()
    total = len(rows)
    started = _stage("resolve_live_roots", total)
    progress = _Progress("resolve_live_roots", total)
    done = 0
    try:
        if total > _LIVE_ATTEMPT_LIMIT:
            raise fail("INTEGRITY_FAILED", "live attempt count exceeds the 10,000 limit",
                       details={"reason_key": "live_attempt_count_exceeded"})
        active = []
        for attempt_id, kind, spec_json, spec_bytes in rows:
            keepalive()
            if spec_bytes is not None and spec_bytes > _LIVE_SPEC_LIMIT:
                raise fail("INTEGRITY_FAILED", "live job spec exceeds the 16 KiB limit",
                           details={"reason_key": "live_job_spec_oversized",
                                    "attempt_id": attempt_id})
            try:
                spec = json.loads(spec_json)
            except (TypeError, ValueError):
                spec = None
            if not isinstance(spec, dict):
                raise fail("INTEGRITY_FAILED", "live job spec is malformed",
                           details={"reason_key": "live_job_spec_malformed",
                                    "attempt_id": attempt_id})
            parameters = spec.get("parameters") or {}
            if not isinstance(parameters, dict):
                raise fail("INTEGRITY_FAILED", "live job parameters are malformed",
                           details={"reason_key": "live_job_parameters_malformed",
                                    "attempt_id": attempt_id})
            if kind not in _PROTECTED_KINDS \
                    and spec.get("kind") not in _PROTECTED_KINDS \
                    and parameters.get("input_mode") != "snapshot":
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
            keepalive()
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
            roots.add(materialization_root(base, request.request_hash).name)
        return roots
    finally:
        _stage_end("resolve_live_roots", started, done, total)

def _materialization_plan(ops_root, keep, conn, store, keepalive=_no_keepalive):
    base = default_materialization_base(ops_root)
    if base.is_symlink() or (base.exists() and not base.is_dir()):
        raise fail("INTEGRITY_FAILED", "materialization root is not a directory",
                   details={"reason_key": "materialization_root_invalid"})
    if not base.exists():
        return []
    _stderr("RETENTION_CURRENT_PUBLICATION_UNBOUND")
    all_entries = list(base.iterdir())
    started = _stage("verify_materialization", len(all_entries))
    progress = _Progress("verify_materialization", len(all_entries))
    done = 0
    try:
        entries = []
        for item in all_entries:
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
        ordered = sorted(entries, key=lambda item: item.stat().st_mtime_ns, reverse=True)
    finally:
        _stage_end("verify_materialization", started, done, len(all_entries))
    live = _live_roots(base, conn, store, keepalive=keepalive)
    retained = {item.name for item in ordered[:keep]} | live
    return [item for item in ordered if item.name not in retained]

def _remove_tree(path, keepalive=_no_keepalive):
    for directory, _, _ in os.walk(path):
        keepalive()
        os.chmod(directory, 0o755)
    shutil.rmtree(path)


def enforce_retention(ops_root, *, conn, store, keep=None, materialization_keep=None,
                      keepalive=None):
    """Keep the newest verified backups and materialization roots; return pruned keys.

    ``conn`` and ``store`` are the coordinator's already-open operations catalog and artifact store; live root resolution never guesses or re-opens them."""
    keepalive = keepalive or _no_keepalive
    keep = _limit(keep, KEEP_ENV, DEFAULT_KEEP, "backup")
    materialization_keep = _limit(materialization_keep, MATERIALIZATION_KEEP_ENV,
                                  DEFAULT_MATERIALIZATION_KEEP, "materialization")
    root = Path(ops_root) / "backups"
    plans = []
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise fail("INTEGRITY_FAILED", "backup root is not a directory",
                       details={"reason_key": "backup_root_invalid"})
        scopes = sorted(root.iterdir(), key=lambda item: item.name)
        if any(scope.is_symlink() or not scope.is_dir() for scope in scopes):
            raise fail("INTEGRITY_FAILED", "backup scope is not a directory",
                       details={"reason_key": "backup_scope_not_directory"})
        started = _stage("verify_backups", len(scopes))
        progress = _Progress("verify_backups", len(scopes))
        done = 0
        try:
            for scope in scopes:
                plans.append(_plan(scope, keep, keepalive=keepalive))
                done += 1
                progress.tick(done)
        finally:
            _stage_end("verify_backups", started, done, len(scopes))
    roots = _materialization_plan(ops_root, materialization_keep, conn, store,
                                  keepalive=keepalive)
    deleted = []
    backup_total = sum(len(discarded) + len(artifact_names)
                       for _, discarded, _, artifact_names in plans)
    started = _stage("delete_backups", backup_total)
    progress = _Progress("delete_backups", backup_total)
    done = 0
    try:
        for scope, discarded, artifacts_dir, artifact_names in plans:
            for record in discarded:
                keepalive()
                (scope / (record["key"] + _DB)).unlink()
                keepalive()
                (scope / (record["key"] + _MANIFEST)).unlink()
                deleted.append(record["key"])
                done += 1
                progress.tick(done)
            for name in sorted(artifact_names):
                keepalive()
                (artifacts_dir / name).unlink()
                done += 1
                progress.tick(done)
    finally:
        _stage_end("delete_backups", started, done, backup_total)
    started = _stage("delete_materialization", len(roots))
    progress = _Progress("delete_materialization", len(roots))
    done = 0
    try:
        for path in roots:
            keepalive()
            _remove_tree(path, keepalive=keepalive)
            done += 1
            progress.tick(done)
    finally:
        _stage_end("delete_materialization", started, done, len(roots))
    return deleted
