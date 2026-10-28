"""Isolated experiment research workspace: a write sandbox over a read-only pinned snapshot.

An overlay, not a copy. The pinned snapshot is READ from the production catalog (opened
``mode=ro`` plus ``query_only``) and the production store's immutable objects, resolved by
snapshot id only, never a head. Every WRITE lands under the workspace root: its own catalog,
store, scratch and output, plus the read pin (the resolved ``SnapshotRef`` published as a
workspace artifact) and ``workspace.json``. The workspace root carries ``MARKER`` so the
supervisor and job launcher refuse to run from it (``workspace.guard``).

Refusals are typed: ``WORKSPACE_REFUSED`` (``details["reason"]``: production_root,
symlink_path, parent_missing, root_not_empty, production_missing, production_mismatch,
layout_invalid, identity_unreadable, source_method_refused, pin_name_invalid) and ``SNAPSHOT_CLOSURE_BROKEN`` (from
``workspace.closure``, plus ``closure_changed``/``manifest_mismatch``/``pin_mismatch``
when the open-time closure, the stored identity or the pin disagree).
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from engine.v2.data import errors as data_errors
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactError, ArtifactStore, canonical_json, fsync_directory
from engine.v2.ops.bootstrap import open_catalog, require_compatible_catalog
from engine.v2.ops.catalog import dumps, register_artifact, transaction
from engine.v2.ops.checkpoints import registered_artifact
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.workspace.closure import (
    _MAX_VALUE_BYTES,
    SnapshotClosure,
    _BoundedConn,
    verify_snapshot_closure,
)
from engine.v2.ops.workspace.guard import MARKER

__all__ = ["IDENTITY_FILE", "MARKER", "ResearchWorkspace", "SnapshotOnlyRepository",
           "WorkspacePaths", "create_workspace", "open_workspace"]

SCHEMA_VERSION = "research_workspace.v1.0"
PIN_SCHEMA_REF = "research_workspace_pin.v1.0"
READ_PIN_SCHEMA_REF = "research_workspace_read_pin.v1.0"
IDENTITY_FILE = "workspace.json"
_DIRECTORIES = ("store", "scratch", "output")
_MAX_METADATA_BYTES = 64 * 1024
_SOURCE_METHODS = frozenset({
    "resolve", "resolve_full", "scan", "table_contract", "fragment_records", "get_event",
    "get_chain", "get_price_series", "get_close", "scan_population_bound", "explain_dependencies",
    "pins_snapshots", "read_counters"})
_IDENTITY_KEYS = ("schema_version", "snapshot_id", "manifest_hash", "identity_digest",
                  "production_catalog", "production_store", "pin_artifact_id")


def _refuse(reason: str, **details):
    return fail("WORKSPACE_REFUSED", "research workspace request refused",
                details={"reason": reason, **details})


def _changed(reason: str, **details):
    return data_errors.fail("SNAPSHOT_CLOSURE_BROKEN", "snapshot closure is incomplete or changed",
                            details={"reason": reason, **details})


def _log(message: str) -> None:
    print("[workspace] " + message, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    catalog: Path
    store: Path
    scratch: Path
    output: Path


def _paths(root: Path) -> WorkspacePaths:
    return WorkspacePaths(root=root, catalog=root / "catalog.sqlite", store=root / "store",
                          scratch=root / "scratch", output=root / "output")


class SnapshotOnlyRepository(Repository):
    """The production source view: an allowlist of read methods, each bound to the pinned snapshot id."""

    def __init__(self, conn, store, snapshot_id: str) -> None:
        super().__init__(conn, store, pin_snapshots=True)
        self._only = snapshot_id

    def __getattribute__(self, name):
        if name.startswith("_") or name in _SOURCE_METHODS:
            return super().__getattribute__(name)
        raise fail("WORKSPACE_REFUSED", "method not available on the pinned source view",
                   details={"reason": "source_method_refused", "method": name})

    def _check(self, snapshot_id: str) -> None:
        if snapshot_id != self._only:
            raise data_errors.fail("SNAPSHOT_NOT_FOUND", "only the pinned snapshot resolves here",
                                   details={"snapshot_id": snapshot_id})

    def _check_ref(self, snapshot_ref) -> None:
        """A ref must be the pinned snapshot exactly, not merely carry its id."""
        self._check(getattr(snapshot_ref, "snapshot_id", None))
        if snapshot_ref != super().resolve(self._only):
            raise data_errors.fail("SNAPSHOT_NOT_FOUND", "reference is not the pinned snapshot",
                                   details={"snapshot_id": snapshot_ref.snapshot_id})

    def resolve(self, snapshot_id: str):
        self._check(snapshot_id)
        return super().resolve(snapshot_id)

    def resolve_full(self, snapshot_id: str):
        self._check(snapshot_id)
        return super().resolve_full(snapshot_id)

    def table_contract(self, snapshot_ref, table_name):
        self._check_ref(snapshot_ref)
        return super().table_contract(snapshot_ref, table_name)

    def fragment_records(self, snapshot_ref, table_name):
        self._check_ref(snapshot_ref)
        return super().fragment_records(snapshot_ref, table_name)

    def get_event(self, event_ref, snapshot_ref):
        self._check_ref(snapshot_ref)
        return super().get_event(event_ref, snapshot_ref)

    def get_chain(self, query, snapshot_ref):
        self._check_ref(snapshot_ref)
        return super().get_chain(query, snapshot_ref)

    def get_price_series(self, query, snapshot_ref):
        self._check_ref(snapshot_ref)
        return super().get_price_series(query, snapshot_ref)

    def get_close(self, ticker, date, observation_ceiling, snapshot_ref):
        self._check_ref(snapshot_ref)
        return super().get_close(ticker, date, observation_ceiling, snapshot_ref)

    def scan(self, query, *, table_name, max_object_bytes=None):
        self._check(getattr(query, "snapshot_id", None))
        return super().scan(query, table_name=table_name, max_object_bytes=max_object_bytes)

    def explain_dependencies(self, query, *, table_name=None, snapshot_ref=None):
        if snapshot_ref is not None:
            self._check_ref(snapshot_ref)
        query_snapshot = getattr(query, "snapshot_id", None)
        if snapshot_ref is None or query_snapshot is not None:
            self._check(query_snapshot)
        return super().explain_dependencies(query, table_name=table_name, snapshot_ref=snapshot_ref)

    def scan_population_bound(self, snapshot_id, **kwargs):
        self._check(snapshot_id)
        return super().scan_population_bound(snapshot_id, **kwargs)


def _lstat(path: Path):
    try:
        return os.lstat(path)
    except OSError:
        return None


def _related(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def _production(catalog, store_root) -> tuple[Path, Path]:
    catalog, store_root = Path(os.path.realpath(catalog)), Path(os.path.realpath(store_root))
    if not catalog.is_file() or not store_root.is_dir():
        raise _refuse("production_missing")
    return catalog, store_root


def _checked_root(root, catalog: Path, store_root: Path) -> Path:
    literal = Path(os.path.abspath(root))
    if Path(os.path.realpath(literal)) != literal:
        raise _refuse("symlink_path")
    if _related(literal, store_root) or _related(literal, catalog.parent):
        raise _refuse("production_root")
    if not literal.parent.is_dir():
        raise _refuse("parent_missing")
    return literal


def _check_layout(paths: WorkspacePaths, production_catalog: Path) -> None:
    root_info = _lstat(paths.root)
    if root_info is not None and stat.S_IMODE(root_info.st_mode) & 0o077:
        raise _refuse("layout_invalid", path=paths.root.name)
    for directory in (paths.root, paths.store, paths.scratch, paths.output):
        info = _lstat(directory)
        if info is None or not stat.S_ISDIR(info.st_mode):
            raise _refuse("layout_invalid", path=directory.name)
    for file in (paths.catalog, paths.root / IDENTITY_FILE, paths.root / MARKER):
        info = _lstat(file)
        if info is None or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise _refuse("layout_invalid", path=file.name)
    if os.path.samefile(paths.catalog, production_catalog):
        raise _refuse("production_root")


def _open_readonly(catalog: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(catalog.as_uri() + "?mode=ro", uri=True, isolation_level=None,
                           timeout=5.0)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, _MAX_VALUE_BYTES)
        require_compatible_catalog(_BoundedConn(conn))
    except BaseException:
        conn.close()
        raise
    return conn


def _read_identity(root: Path) -> dict:
    try:
        fd = os.open(root / IDENTITY_FILE, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError
            data = os.read(fd, _MAX_METADATA_BYTES + 1)
        finally:
            os.close(fd)
        if len(data) > _MAX_METADATA_BYTES:
            raise ValueError
        identity = json.loads(data.decode("utf-8"))
        if (not isinstance(identity, dict) or any(key not in identity for key in _IDENTITY_KEYS)
                or any(not isinstance(identity[key], str)
                       for key in _IDENTITY_KEYS if key != "schema_version")):
            raise ValueError
    except (OSError, ValueError):
        raise _refuse("identity_unreadable") from None
    return identity


class ResearchWorkspace:
    """Open handles. ``conn``/``store`` write to the workspace only; ``source`` reads the
    pinned production snapshot through a read-only connection and never writes."""

    def __init__(self, paths, identity, closure, conn, store, source_conn, source_store):
        self.paths, self.identity, self.closure = paths, identity, closure
        self.snapshot_id, self.manifest_hash = identity["snapshot_id"], identity["manifest_hash"]
        self.conn, self.store = conn, store
        self._source_conn, self._source_store = source_conn, source_store
        self.source = SnapshotOnlyRepository(_BoundedConn(source_conn), source_store, self.snapshot_id)

    def source_snapshot(self):
        """The pinned ``SnapshotRef``; its id and manifest hash must equal the pin."""
        ref = self.source.resolve(self.snapshot_id)
        if ref.snapshot_id != self.snapshot_id or ref.manifest_hash != self.manifest_hash:
            raise _changed("manifest_mismatch", snapshot_id=self.snapshot_id)
        return ref

    def verify_before_seal(self, *, deep: bool = False) -> SnapshotClosure:
        """Re-verify the closure; it must equal the one verified at open, stat included."""
        closure = verify_snapshot_closure(self._source_conn, self._source_store, self.snapshot_id,
                                          self.manifest_hash, deep=deep)
        if (closure.identity_digest != self.closure.identity_digest
                or closure.stat_digest != self.closure.stat_digest):
            raise _changed("closure_changed", snapshot_id=self.snapshot_id)
        return closure

    def record_read_pin(self, name: str, document: dict, *, clock):
        """Preflight hook: publish a read-pin document to the WORKSPACE store and catalog only."""
        if not isinstance(name, str) or not 0 < len(name) <= 128:
            raise _refuse("pin_name_invalid")
        body = canonical_json({"schema_version": READ_PIN_SCHEMA_REF, "name": name,
                               "snapshot_id": self.snapshot_id,
                               "manifest_hash": self.manifest_hash, "pin": document})
        ref = self.store.publish_bytes(body.encode("utf-8"), schema_ref=READ_PIN_SCHEMA_REF)
        with transaction(self.conn):
            register_artifact(self.conn, ref, None, clock)
        return ref

    def close(self) -> None:
        self.conn.close()
        self._source_conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _clear(root: Path, created: bool) -> None:
    if created:
        shutil.rmtree(root, ignore_errors=True)
        return
    for child in root.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def create_workspace(root, *, production_catalog, production_store, snapshot_id: str,
                     manifest_hash: str, clock, deep: bool = False) -> ResearchWorkspace:
    """Verify the pinned snapshot's closure, then lay out and open a new workspace."""
    if (not isinstance(root, (str, os.PathLike)) or not isinstance(snapshot_id, str)
            or not isinstance(manifest_hash, str)):
        raise _refuse("argument_invalid")
    catalog, store_root = _production(production_catalog, production_store)
    base = _checked_root(root, catalog, store_root)
    if base.exists() and (not base.is_dir() or any(base.iterdir())):
        raise _refuse("root_not_empty")
    created, paths = not base.exists(), _paths(base)
    _log(f"stage=create start snapshot={snapshot_id}")
    source_conn, conn = _open_readonly(catalog), None
    try:
        source_store = ArtifactStore(store_root)
        closure = verify_snapshot_closure(source_conn, source_store, snapshot_id, manifest_hash,
                                          deep=deep)
        snapshot = Repository(_BoundedConn(source_conn)).resolve(snapshot_id)
        base.mkdir(mode=0o700, exist_ok=True)
        os.chmod(base, 0o700)
        (base / MARKER).write_text(SCHEMA_VERSION + "\n", encoding="utf-8")
        for name in _DIRECTORIES:
            (base / name).mkdir(mode=0o700)
        conn, store = open_catalog(paths.catalog, clock=clock), ArtifactStore(paths.store)
        pin = store.publish_bytes(dumps(snapshot).encode("utf-8"), schema_ref=PIN_SCHEMA_REF)
        with transaction(conn):
            register_artifact(conn, pin, None, clock)
        identity = {"schema_version": SCHEMA_VERSION, "snapshot_id": snapshot_id,
                    "manifest_hash": manifest_hash, "identity_digest": closure.identity_digest,
                    "production_catalog": str(catalog), "production_store": str(store_root),
                    "pin_artifact_id": pin.artifact_id}
        staged = base / (IDENTITY_FILE + ".tmp")
        fd = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, json.dumps(identity, sort_keys=True).encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(staged, base / IDENTITY_FILE)
        fsync_directory(base)
    except BaseException:
        source_conn.close()
        if conn is not None:
            conn.close()
        if base.exists():
            _clear(base, created)
        raise
    _log("stage=create done")
    return ResearchWorkspace(paths, identity, closure, conn, store, source_conn, source_store)


def open_workspace(root, *, production_catalog, production_store, clock,
                   deep: bool = False) -> ResearchWorkspace:
    """Re-open a workspace: layout, production binding, closure and pin must all still hold."""
    catalog, store_root = _production(production_catalog, production_store)
    base = _checked_root(root, catalog, store_root)
    paths = _paths(base)
    _check_layout(paths, catalog)
    identity = _read_identity(base)
    if identity["schema_version"] != SCHEMA_VERSION:
        raise _refuse("schema_mismatch")
    if (identity["production_catalog"] != str(catalog)
            or identity["production_store"] != str(store_root)):
        raise _refuse("production_mismatch")
    _log(f"stage=open start snapshot={identity['snapshot_id']}")
    source_conn, conn = _open_readonly(catalog), None
    try:
        source_store = ArtifactStore(store_root)
        closure = verify_snapshot_closure(source_conn, source_store, identity["snapshot_id"],
                                          identity["manifest_hash"], deep=deep)
        if closure.identity_digest != identity["identity_digest"]:
            raise _changed("closure_changed", snapshot_id=identity["snapshot_id"])
        conn, store = open_catalog(paths.catalog, clock=clock), ArtifactStore(paths.store)
        snapshot = Repository(_BoundedConn(source_conn)).resolve(identity["snapshot_id"])
        expected = dumps(snapshot).encode("utf-8")
        try:
            ref = registered_artifact(conn, identity["pin_artifact_id"],
                                      max_metadata_bytes=_MAX_METADATA_BYTES)
            pinned = store.read_verified(ref) if ref.byte_size == len(expected) else None
        except (ArtifactError, OpsError):
            pinned = None
        if pinned != expected:
            raise _changed("pin_mismatch", snapshot_id=identity["snapshot_id"])
    except BaseException:
        source_conn.close()
        if conn is not None:
            conn.close()
        raise
    _log("stage=open done")
    return ResearchWorkspace(paths, identity, closure, conn, store, source_conn, source_store)
