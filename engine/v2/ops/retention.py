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

from engine.v2.foundation import ArtifactError, fsync_directory, safe_relative_path
from engine.v2.ops.catalog import decode_bounded_text, integrity_errors, sql_text_budget
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.fingerprints import NotRegularFile, atomic_file_info, file_hash
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
_MAX_JOURNAL_BYTES = 4 * 1024 * 1024
_JOURNAL_NAME = ".retention-journal.json"
_JOURNAL_SCHEMA = "retention_journal.v1.1"
_JOURNAL_KEYS = frozenset({"schema_version", "files", "roots", "backup_keep"})
_JOURNAL_FILE_KEYS = frozenset({"path", "content_hash", "byte_size"})
MAX_BACKUP_STREAM_BYTES = 4 * 1024 * 1024 * 1024
MAX_LIVE_SPEC_BYTES = 1024 * 1024
_INTEGRITY_PROGRESS_INTERVAL = 1000
_INTEGRITY_PROGRESS_LOG_SECONDS = 60.0

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
    bad = not isinstance(name, str) or not name or name in (".", "..") or "\x00" in name
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

def _journal_invalid():
    return fail("INTEGRITY_FAILED", "retention journal is invalid",
                details={"reason_key": "retention_journal_invalid"})

def _journal_progress(stage):
    started = time.monotonic()
    last = started

    def on_progress(bytes_read, total):
        nonlocal last
        now = time.monotonic()
        if now - last < 60.0:
            return
        last = now
        elapsed = now - started
        rate = bytes_read / elapsed if elapsed > 0 else 0.0
        eta = (total - bytes_read) / rate if rate > 0 and total > bytes_read else None
        _stderr("RETENTION_PROGRESS", stage=stage, done=bytes_read,
                total=total, bytes_per_s=round(rate, 3),
                eta_s=None if eta is None else round(eta, 1),
                rss_peak_mb=_rss_peak_mb())

    return on_progress

def _journal_record(path, ops_root, keepalive=None):
    try:
        digest, size = atomic_file_info(
            path, keepalive=keepalive, max_bytes=MAX_BACKUP_STREAM_BYTES,
            on_progress=_journal_progress("verify_backup_file"))
    except NotRegularFile:
        raise _journal_invalid() from None
    return {"path": path.relative_to(ops_root).as_posix(),
            "content_hash": digest, "byte_size": size}

def _journal_backup_parts(relative):
    if not isinstance(relative, str):
        return None
    parts = relative.split("/")
    if len(parts) not in (3, 4) or parts[0] != "backups":
        return None
    try:
        safe_relative_path(relative)
    except ArtifactError:
        return None
    _safe_name(parts[1])
    if len(parts) == 3:
        if parts[2].endswith(_DB):
            _safe_name(parts[2][:-len(_DB)])
        elif parts[2].endswith(_MANIFEST):
            _safe_name(parts[2][:-len(_MANIFEST)])
        else:
            return None
    elif parts[2] == "artifacts":
        _safe_name(parts[3], artifact=True)
    else:
        return None
    return parts

def _journal_tree_directories(value):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value) \
            or len(set(value)) != len(value) or "" not in value:
        raise _journal_invalid()
    for relative in value:
        if relative == "":
            continue
        parts = relative.split("/")
        if "\\" in relative or "\x00" in relative \
                or any(part in ("", ".", "..") for part in parts):
            raise _journal_invalid()
    return value

def _journal_relative_path_safe(parts, relative):
    if "\\" in relative or "\x00" in relative:
        return False
    return all(part not in ("", ".", "..") for part in parts)

def _journal_tree_file_entry(item, seen):
    if not isinstance(item, dict) or set(item) != _JOURNAL_FILE_KEYS:
        raise _journal_invalid()
    relative = item["path"]
    if not isinstance(relative, str) or relative in seen:
        raise _journal_invalid()
    parts = relative.split("/")
    digest, size = item["content_hash"], item["byte_size"]
    if not _journal_relative_path_safe(parts, relative) \
            or not isinstance(digest, str) or not digest.startswith("sha256:") \
            or isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise _journal_invalid()
    seen.add(relative)
    return {"path": relative, "content_hash": digest, "byte_size": size}

def _journal_tree_files(value):
    if not isinstance(value, list):
        raise _journal_invalid()
    files, seen = [], set()
    for item in value:
        files.append(_journal_tree_file_entry(item, seen))
    return files

def _journal_entry(entry):
    if not isinstance(entry, dict) or set(entry) != _JOURNAL_FILE_KEYS:
        raise _journal_invalid()
    parts = _journal_backup_parts(entry["path"])
    digest, size = entry["content_hash"], entry["byte_size"]
    if parts is None or not isinstance(digest, str) or not digest.startswith("sha256:") \
            or isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise _journal_invalid()
    return parts, digest, size

def _journal_tree_record(root, base, keepalive=None):
    directories, files = [""], []

    def walk_error(error):
        raise error

    try:
        for directory, names, filenames in os.walk(
                root, topdown=True, followlinks=False, onerror=walk_error):
            _renew(keepalive)
            parent = Path(directory)
            for name in names:
                child = parent / name
                info = child.lstat()
                if not stat.S_ISDIR(info.st_mode):
                    raise _journal_invalid()
                directories.append(child.relative_to(root).as_posix())
            for name in filenames:
                _renew(keepalive)
                child = parent / name
                try:
                    digest, size = atomic_file_info(
                        child, keepalive=keepalive,
                        max_bytes=MAX_BACKUP_STREAM_BYTES,
                        on_progress=_journal_progress("verify_backup_file"))
                except NotRegularFile:
                    raise _journal_invalid() from None
                files.append({"path": child.relative_to(root).as_posix(),
                              "content_hash": digest, "byte_size": size})
    except OSError:
        raise fail("INTEGRITY_FAILED", "materialization root could not be verified",
                   details={"reason_key": "materialization_root_incomplete"}) from None
    return {"base": str(Path(base).resolve()), "name": root.name,
            "mtime_ns": root.stat().st_mtime_ns,
            "directories": sorted(directories), "files": sorted(files, key=lambda x: x["path"])}

def _journal_tree_entry(entry):
    keys = {"base", "name", "mtime_ns", "directories", "files"}
    if not isinstance(entry, dict) or set(entry) != keys:
        raise _journal_invalid()
    base, name, mtime = entry["base"], entry["name"], entry["mtime_ns"]
    if not isinstance(base, str) or not Path(base).is_absolute() \
            or not isinstance(name, str) or name in ("", ".", "..") \
            or "/" in name or "\\" in name or "\x00" in name \
            or isinstance(mtime, bool) or not isinstance(mtime, int):
        raise _journal_invalid()
    return {"base": base, "name": name, "mtime_ns": mtime,
            "directories": _journal_tree_directories(entry["directories"]),
            "files": _journal_tree_files(entry["files"])}

def _journal_document_backup_keep(document, raw):
    backup_keep = document.get("backup_keep") if isinstance(document, dict) else None
    if len(raw) > _MAX_JOURNAL_BYTES or not isinstance(document, dict) \
            or set(document) != _JOURNAL_KEYS \
            or document["schema_version"] != _JOURNAL_SCHEMA \
            or not isinstance(document["files"], list) \
            or not isinstance(document["roots"], list) \
            or isinstance(backup_keep, bool) or not isinstance(backup_keep, int) \
            or backup_keep <= 0:
        raise _journal_invalid()
    return backup_keep

def _journal_files(journal):
    try:
        info = journal.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise fail("INTEGRITY_FAILED", "retention journal could not be read",
                   details={"reason_key": "retention_journal_unreadable"}) from None
    if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_JOURNAL_BYTES:
        raise _journal_invalid()
    try:
        with journal.open("rb") as stream:
            raw = stream.read(_MAX_JOURNAL_BYTES + 1)
        document = json.loads(raw)
    except (OSError, ValueError, RecursionError):
        raise fail("INTEGRITY_FAILED", "retention journal could not be read",
                   details={"reason_key": "retention_journal_unreadable"}) from None
    backup_keep = _journal_document_backup_keep(document, raw)
    return ([_journal_entry(entry) for entry in document["files"]],
            [_journal_tree_entry(entry) for entry in document["roots"]],
            backup_keep)

def _write_journal(journal, files, roots, backup_keep):
    payload = json.dumps({"schema_version": _JOURNAL_SCHEMA,
                          "backup_keep": backup_keep,
                          "files": files, "roots": roots},
                         sort_keys=True).encode("utf-8")
    if len(payload) > _MAX_JOURNAL_BYTES:
        raise fail("INTEGRITY_FAILED", "retention journal is too large",
                   details={"reason_key": "retention_journal_too_large"})
    pending = journal.with_name(journal.name + ".pending")
    try:
        journal.parent.mkdir(parents=True, exist_ok=True)
        with pending.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, journal)
        fsync_directory(journal.parent)
    except OSError:
        raise fail("BACKUP_FAILED", "retention journal could not be written",
                   details={"reason_key": "retention_journal_write_failed"}) from None

def _clear_journal(journal):
    try:
        journal.unlink()
        fsync_directory(journal.parent)
    except FileNotFoundError:
        return
    except OSError:
        raise fail("BACKUP_FAILED", "retention journal could not be cleared",
                   details={"reason_key": "retention_journal_clear_failed"}) from None

def _journal_tree_file_matches(path, info, expected, keepalive):
    return expected is not None and stat.S_ISREG(info.st_mode) \
        and info.st_size == expected["byte_size"] \
        and _digest(path, keepalive) == expected["content_hash"]

def _verify_journal_tree(entry, keepalive=None):
    root = Path(entry["base"]) / entry["name"]
    try:
        info = root.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise fail("INTEGRITY_FAILED", "materialization journal root could not be read",
                   details={"reason_key": "materialization_root_incomplete"}) from None
    if not stat.S_ISDIR(info.st_mode):
        raise _journal_invalid()
    expected_dirs = set(entry["directories"])
    expected_files = {item["path"]: item for item in entry["files"]}
    actual_dirs, actual_files = {""}, set()

    def walk_error(error):
        raise error

    try:
        for directory, names, filenames in os.walk(
                root, topdown=True, followlinks=False, onerror=walk_error):
            _renew(keepalive)
            parent = Path(directory)
            for name in names:
                child = parent / name
                if not stat.S_ISDIR(child.lstat().st_mode):
                    raise _journal_invalid()
                actual_dirs.add(child.relative_to(root).as_posix())
            for name in filenames:
                _renew(keepalive)
                child = parent / name
                file_info = child.lstat()
                relative = child.relative_to(root).as_posix()
                expected = expected_files.get(relative)
                if not _journal_tree_file_matches(child, file_info, expected, keepalive):
                    raise _journal_invalid()
                actual_files.add(relative)
    except OSError:
        raise fail("INTEGRITY_FAILED", "materialization journal root could not be verified",
                   details={"reason_key": "materialization_root_incomplete"}) from None
    if not actual_dirs <= expected_dirs or not actual_files <= set(expected_files):
        raise _journal_invalid()
    return root

def _read_journal_or_discard_pending(journal):
    entries = _journal_files(journal)
    if entries is not None:
        return entries
    pending = journal.with_name(journal.name + ".pending")
    try:
        info = pending.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise fail("INTEGRITY_FAILED", "retention journal could not be read",
                   details={"reason_key": "retention_journal_unreadable"}) from None
    if not stat.S_ISREG(info.st_mode):
        raise _journal_invalid()
    try:
        pending.unlink()
        fsync_directory(journal.parent)
    except OSError:
        raise fail("BACKUP_FAILED", "retention journal temporary file could not be cleared",
                   details={"reason_key": "retention_journal_clear_failed"}) from None
    return None

def _validated_journal_base(materialization_base, trees):
    base_path = Path(materialization_base)
    if trees and (base_path.is_symlink() or (base_path.exists() and not base_path.is_dir())):
        raise fail("INTEGRITY_FAILED", "materialization root is not a directory",
                   details={"reason_key": "materialization_root_invalid"})
    base = base_path.resolve()
    if any(Path(entry["base"]) != base for entry in trees):
        raise _journal_invalid()
    return base

def _check_journal_roots_live(base, trees, conn, store, keepalive):
    if not trees:
        return
    live = _live_roots(base, conn, store, keepalive)
    if live.intersection(entry["name"] for entry in trees):
        raise fail("INTEGRITY_FAILED", "journaled materialization root is in use",
                   details={"reason_key": "materialization_root_in_use"})

def _journal_root_candidates(base, tree_by_name):
    if not tree_by_name or not base.exists():
        return []
    candidates = []
    for item in base.iterdir():
        if item.name.startswith("."):
            continue
        if item.is_symlink() or not item.is_dir():
            raise fail("INTEGRITY_FAILED", "materialization entry is not a directory",
                       details={"reason_key": "materialization_entry_invalid"})
        if item.name not in tree_by_name \
                and stat.S_IMODE(os.lstat(item).st_mode) != 0o555:
            raise fail("INTEGRITY_FAILED", "materialization root is incomplete",
                       details={"reason_key": "materialization_root_incomplete"})
        candidates.append(item)
    candidates.sort(key=lambda item: tree_by_name.get(item.name, {}).get(
        "mtime_ns", item.stat().st_mtime_ns), reverse=True)
    return candidates

def _check_journal_roots_not_retained(candidates, tree_by_name, keep):
    if {item.name for item in candidates[:keep]}.intersection(tree_by_name):
        raise fail("INTEGRITY_FAILED", "journaled materialization root is now retained",
                   details={"reason_key": "materialization_root_retained"})

def _verify_journal_files(root, files, keepalive):
    verified = []
    for parts, digest, size in files:
        path = root.joinpath(*parts)
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise fail("INTEGRITY_FAILED", "retention journal entry could not be read",
                       details={"reason_key": "retention_journal_entry_unreadable"}) from None
        if not stat.S_ISREG(info.st_mode) or info.st_size != size \
                or _digest(path, keepalive) != digest:
            raise fail("INTEGRITY_FAILED", "retention journal entry does not match its record",
                       details={"reason_key": "retention_journal_entry_mismatch"})
        verified.append(path)
    return verified

def _replay_journal(journal, files, trees, keepalive):
    for path in files:
        _unlink(path)
    for path in trees:
        if path is not None:
            _remove_tree(path, keepalive)
    _clear_journal(journal)

def _recover_journal(ops_root, materialization_base, conn, store, keep, backup_keep=None, keepalive=None):
    root = Path(ops_root)
    journal = root / _JOURNAL_NAME
    entries = _read_journal_or_discard_pending(journal)
    if entries is None:
        return False
    files, trees, recorded_backup_keep = entries
    if backup_keep > recorded_backup_keep:
        raise fail("INTEGRITY_FAILED", "backup retention limit increased during interrupted prune",
                   details={"reason_key": "retention_journal_policy_changed",
                            "recorded_keep": recorded_backup_keep,
                            "current_keep": backup_keep})
    base = _validated_journal_base(materialization_base, trees)
    _check_journal_roots_live(base, trees, conn, store, keepalive)
    tree_by_name = {entry["name"]: entry for entry in trees}
    candidates = _journal_root_candidates(base, tree_by_name)
    _check_journal_roots_not_retained(candidates, tree_by_name, keep)
    verified_files = _verify_journal_files(root, files, keepalive)
    verified_trees = [_verify_journal_tree(entry, keepalive) for entry in trees]
    _replay_journal(journal, verified_files, verified_trees, keepalive)
    return True


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
        state = {"batches": 0, "started": time.monotonic(), "last_log": time.monotonic()}

        def _progress():
            nonlocal callback_error
            state["batches"] += 1
            if keepalive is not None:
                try:
                    keepalive()
                except Exception as error:
                    callback_error = error
                    return 1
            now = time.monotonic()
            if now - state["last_log"] >= _INTEGRITY_PROGRESS_LOG_SECONDS:
                elapsed = now - state["started"]
                state["last_log"] = now
                rate = state["batches"] / elapsed if elapsed > 0 else 0.0
                _stderr("RETENTION_PROGRESS", stage="verify_backup_database",
                        done=state["batches"], total=None,
                        rate_per_s=round(rate, 3), eta_s=None,
                        elapsed_s=round(elapsed, 3), rss_peak_mb=_rss_peak_mb())
            return 0

        try:
            conn = sqlite3.connect(proc_fd.as_uri() + "?mode=ro&immutable=1", uri=True)
            try:
                conn.set_progress_handler(_progress, _INTEGRITY_PROGRESS_INTERVAL)
                started = _stage("verify_backup_database", None)
                try:
                    problems = integrity_errors(conn)
                finally:
                    _stage_end("verify_backup_database", started, state["batches"], None)
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

def _residue_keys(scope, keepalive):
    names = set()
    for path in scope.iterdir():
        _renew(keepalive)
        names.add(path.name)
    manifests = {name[:-len(_MANIFEST)] for name in names
                 if name.endswith(_MANIFEST)}
    databases = {name[:-len(_DB)] for name in names if name.endswith(_DB)}
    pending_suffix = ".manifest.pending"
    pending = {name[:-len(pending_suffix)] for name in names
               if name.endswith(pending_suffix)}
    return (databases - manifests) | (pending - manifests)

def _residue_row(conn, key, limit):
    _safe_name(key)
    try:
        row = conn.execute(
            "SELECT state, length(CAST(payload_json AS BLOB)) AS payload_bytes, "
            "CASE WHEN typeof(payload_json)='text' AND "
            "length(CAST(payload_json AS BLOB))<=? "
            "THEN CAST(payload_json AS BLOB) END AS payload_json "
            "FROM outbox WHERE kind='backup' AND logical_key=?",
            (limit, key)).fetchone()
    except sqlite3.Error:
        row = None
    if row is None or row[0] not in ("pending", "failed"):
        raise fail("INTEGRITY_FAILED", "backup creation residue is not verifiably abandoned",
                   details={"reason_key": "backup_creation_residue_unverified",
                            "backup_key": key})
    if row[2] is None or row[1] > limit:
        raise fail("INTEGRITY_FAILED", "backup creation payload is too large or invalid",
                   details={"reason_key": "backup_creation_payload_invalid",
                            "backup_key": key})
    return row

def _residue_payload(row, encoding, limit, key):
    try:
        text = decode_bounded_text(row[2], encoding, limit)
        payload = json.loads(text)
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise fail("INTEGRITY_FAILED", "backup creation payload is invalid",
                   details={"reason_key": "backup_creation_payload_invalid",
                            "backup_key": key}) from None
    if not isinstance(payload, dict) or payload.get("backup_key") != key \
            or not isinstance(payload.get("artifacts"), dict):
        raise fail("INTEGRITY_FAILED", "backup creation payload does not match residue",
                   details={"reason_key": "backup_creation_payload_invalid",
                            "backup_key": key})
    return payload

def _failed_creation_residue(scope, conn, keepalive=None):
    keys = _residue_keys(scope, keepalive)
    if not keys:
        return set(), set()
    encoding, limit = sql_text_budget(conn, MAX_LIVE_SPEC_BYTES)
    artifacts = set()
    for key in sorted(keys):
        row = _residue_row(conn, key, limit)
        payload = _residue_payload(row, encoding, limit, key)
        for name in payload["artifacts"]:
            _safe_name(name, artifact=True)
            artifacts.add(name)
    return keys, artifacts

def _scope_files(scope, keepalive=None, residue_keys=()):
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
            key = _safe_name(path.name[:-len(_DB)])
            if key in residue_keys and key not in manifests:
                continue
            databases[key] = path
        elif path.name.endswith(".manifest.pending"):
            key = _safe_name(path.name[:-len(".manifest.pending")])
            if key in residue_keys and key not in manifests:
                continue
            raise fail("INTEGRITY_FAILED", "backup scope contains a stray file",
                       details={"reason_key": "backup_scope_stray_file"})
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

def _read_scope(scope, keepalive=None, conn=None):
    residue_keys, residue_artifacts = _failed_creation_residue(scope, conn, keepalive)
    artifacts_dir = _scope_artifacts_dir(scope)
    manifests, databases = _scope_files(scope, keepalive, residue_keys)
    records = _scope_records(manifests, databases, artifacts_dir, keepalive)
    referenced = set().union(*(record["artifacts"] for record in records)) if records else set()
    present = _present_artifacts(artifacts_dir, keepalive)
    unrecognized = present - referenced - residue_artifacts
    if unrecognized:
        raise fail("INTEGRITY_FAILED", "backup artifacts are stray or incomplete",
                   details={"reason_key": "backup_artifacts_mismatch"})
    if residue_keys:
        _stderr("RETENTION_BACKUP_RESIDUE_KEPT", backup_count=len(residue_keys),
                artifact_count=len(residue_artifacts))
    return records, artifacts_dir, present, residue_artifacts

def _plan(scope, keep, keepalive=None, conn=None):
    records, artifacts_dir, present, residue_artifacts = _read_scope(
        scope, keepalive, conn)
    ordered = sorted(records, key=lambda r: (r["when"], r["key"]), reverse=True)
    retained = ordered[:keep]
    kept = set().union(*(r["artifacts"] for r in retained)) if retained else set()
    return scope, ordered[keep:], artifacts_dir, present - kept - residue_artifacts

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

def _live_request_artifact_invalid(attempt_id):
    return fail("INTEGRITY_FAILED",
                "live materialization request artifact is invalid",
                details={"reason_key": "live_request_artifact_invalid",
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
            try:
                request = request_from_artifact(conn, store, bound[0][0],
                                                max_bytes=MAX_LIVE_SPEC_BYTES)
            except OpsError as error:
                if error.code != "INPUT_CHANGED":
                    raise
                raise _live_request_artifact_invalid(attempt_id) from None
            except ArtifactError:
                raise _live_request_artifact_invalid(attempt_id) from None
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
            if stat.S_IMODE(os.lstat(item).st_mode) != 0o555:
                raise fail("INTEGRITY_FAILED", "materialization root is not mode 0555",
                           details={"reason_key": "materialization_root_incomplete"})
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
        raise fail("BACKUP_FAILED", "backup deletion failed",
                   details={"reason_key": "backup_delete_failed"}) from None

def _remove_tree(path, keepalive=None):
    """Remove one materialization tree with the shared reclaim walk.

    Raw ``OSError`` maps to materialization_delete_failed; an ``OpsError``
    raised by ``keepalive`` propagates unchanged."""
    progress = None

    def on_progress(done, total):
        nonlocal progress
        if progress is None:
            progress = _Progress("delete_materialization_root", total)
        progress.tick(done)

    try:
        remove_tree(path, lambda: _renew(keepalive), on_progress=on_progress)
    except OSError:
        raise fail("BACKUP_FAILED", "materialization deletion failed",
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

def _verified_backup_plans(root, keep, keepalive, conn):
    if not root.exists():
        return []
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
    started = _stage("verify_backups", len(scopes))
    progress, done, plans = _Progress("verify_backups", len(scopes)), 0, []
    try:
        for scope in scopes:
            plans.append(_plan(scope, keep, keepalive, conn=conn))
            done += 1
            progress.tick(done)
    finally:
        _stage_end("verify_backups", started, done, len(scopes))
    return plans

def _write_deletion_journal(ops_root, plans, roots, base, backup_keep=None, keepalive=None):
    files = []
    for scope, discarded, artifacts_dir, artifact_names in plans:
        for record in discarded:
            files.append(_journal_record(scope / (record["key"] + _DB), ops_root, keepalive))
            files.append(_journal_record(scope / (record["key"] + _MANIFEST), ops_root, keepalive))
        for name in sorted(artifact_names):
            files.append(_journal_record(artifacts_dir / name, ops_root, keepalive))
    trees = [_journal_tree_record(path, base, keepalive) for path in roots]
    journal = ops_root / _JOURNAL_NAME
    active = bool(files or trees)
    if active:
        _write_journal(journal, files, trees, backup_keep)
    return journal, active

def _delete_backup_plans(plans, keepalive):
    total = sum(len(discarded) + len(artifact_names)
                for _, discarded, _, artifact_names in plans)
    started = _stage("delete_backups", total)
    progress, done, deleted = _Progress("delete_backups", total), 0, []
    try:
        for scope, discarded, artifacts_dir, artifact_names in plans:
            for record in discarded:
                _renew(keepalive)
                _unlink(scope / (record["key"] + _DB))
                _renew(keepalive)
                _unlink(scope / (record["key"] + _MANIFEST))
                deleted.append(record["key"])
                done += 1
                progress.tick(done)
            for name in sorted(artifact_names):
                _renew(keepalive)
                _unlink(artifacts_dir / name)
                done += 1
                progress.tick(done)
    finally:
        _stage_end("delete_backups", started, done, total)
    return deleted

def _delete_materialization_plans(roots, keepalive):
    started = _stage("delete_materialization", len(roots))
    progress, done = _Progress("delete_materialization", len(roots)), 0
    try:
        for path in roots:
            _renew(keepalive)
            _remove_tree(path, keepalive)
            done += 1
            progress.tick(done)
    finally:
        _stage_end("delete_materialization", started, done, len(roots))

def _enforce_retention(ops_root, *, conn, store, keep=None, materialization_keep=None, keepalive=None, materialization_base=None):
    keep = _limit(keep, KEEP_ENV, DEFAULT_KEEP, "backup")
    materialization_keep = _limit(materialization_keep, MATERIALIZATION_KEEP_ENV,
                                  DEFAULT_MATERIALIZATION_KEEP, "materialization")
    ops_root = Path(ops_root)
    base = Path(materialization_base or default_materialization_base(ops_root))
    _recover_journal(ops_root, base, conn, store, materialization_keep, keep, keepalive)
    plans = _verified_backup_plans(ops_root / "backups", keep, keepalive, conn)
    roots = _materialization_plan(ops_root, materialization_keep, conn, store,
                                  base=base, keepalive=keepalive)
    backup_total = sum(len(discarded) + len(artifact_names)
                       for _, discarded, _, artifact_names in plans)
    _stderr("RETENTION_DELETE_PLAN", backup_total=backup_total,
            materialization_total=len(roots), keep=keep,
            materialization_keep=materialization_keep, scopes=len(plans))
    journal, journal_active = _write_deletion_journal(
        ops_root, plans, roots, base, keep, keepalive)
    deleted = _delete_backup_plans(plans, keepalive)
    _delete_materialization_plans(roots, keepalive)
    if journal_active:
        _clear_journal(journal)
    _stderr("RETENTION_DELETE_COMPLETE", backup_count=len(deleted),
            materialization_count=len(roots))
    return deleted
