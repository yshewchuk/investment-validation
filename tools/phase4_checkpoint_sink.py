"""Durable, bounded storage for Phase 4 diagnostic checkpoints.

The sink keeps only compact resource and case indexes in memory.  Resource
bytes remain shared files beneath the bundle root; each case is serialized,
fsynced, and atomically installed as an independent JSON document.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator

from engine.v2.foundation import content_hash

#: Shared with ``checks/phase4_checkpoints.validate_bundle`` -- one schema id,
#: one contract. The manifest this sink finalizes is exactly what that
#: validator expects: ``release_id`` and ``coverage`` are real, top-level,
#: independently checkable fields (not opaque caller metadata), so a bundle
#: is auditable after the fact without trusting the writer's say-so.
SCHEMA_VERSION = "phase4_diagnostic_checkpoints.v1.0"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CHUNK = 1 << 20


class CheckpointSinkError(ValueError):
    """The checkpoint bundle is unsafe, malformed, or internally inconsistent."""


def _fail(label: str, reason: str) -> None:
    raise CheckpointSinkError(label + ": " + reason)


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or value in {".", ".."} or not _IDENTIFIER.fullmatch(value):
        _fail(label, "expected a safe identifier")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        _fail(label, "expected sha256:<64 lowercase hex characters>")
    return value


def _validate_json(value: Any, label: str = "document") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise CheckpointSinkError(label + ": invalid Unicode string") from exc
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(label, "non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, label + "[" + str(index) + "]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(label, "object keys must be strings")
            _validate_json(key, label + ".key")
            _validate_json(item, label + "." + key)
        return
    _fail(label, "value is not JSON-safe")


def _json_bytes(value: Any) -> bytes:
    _validate_json(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"


#: The exact `json.dumps` settings `_json_bytes` uses, reused so the two
#: never drift.
_CASE_ENCODER_KWARGS = dict(
    allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True,
)


def _iter_json_chunks(value: Any) -> Iterator[bytes]:
    """Stream the exact same bytes `_json_bytes(value)` would return, as
    UTF-8 chunks, so a large case document (a chooser's multi-member
    checkpoint) is never held as one big `str`/`bytes` object. Byte-identical
    to `_json_bytes` -- both are `json.dumps` with the same settings; this
    one is fed to the file/hash piece by piece instead of joined first
    (`json.JSONEncoder.iterencode` yields incrementally with or without the C
    accelerator -- see `tests/test_phase4_checkpoint_sink.py`).
    """
    _validate_json(value)
    encoder = json.JSONEncoder(**_CASE_ENCODER_KWARGS)
    for piece in encoder.iterencode(value):
        yield piece.encode("utf-8")
    yield b"\n"


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _strict_load(path: Path, label: str) -> Any:
    def reject_constant(value: str) -> None:
        _fail(label, "non-finite JSON constant " + value)

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=reject_constant)
    except CheckpointSinkError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointSinkError(label + ": malformed JSON") from exc
    _validate_json(value, label)
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_stream(path: Path, chunks: Iterable[bytes]) -> str:
    """`_atomic_write`, but for a `chunks` iterator instead of one `bytes`
    object already in hand: each chunk is written and hashed as it arrives,
    so the full document is never assembled in memory just to write it.
    Returns the `sha256:<hex>` digest over the concatenated chunks -- the
    same value `_sha256_bytes(b"".join(chunks))` would have returned.
    """
    digest = hashlib.sha256()
    temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for chunk in chunks:
                handle.write(chunk)
                digest.update(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return "sha256:" + digest.hexdigest()


class DiskCheckpointSink:
    """Incrementally publish a resumable Phase 4 checkpoint bundle."""

    def __init__(self, root: Path) -> None:
        supplied = Path(root)
        supplied.mkdir(parents=True, exist_ok=True)
        self.root = supplied.resolve()
        self.cases_dir = self.root / "cases"
        self.cases_dir.mkdir(exist_ok=True)
        if self.cases_dir.is_symlink() or not self.cases_dir.is_dir():
            _fail("root/cases", "must be a real directory")
        self._cleanup_temps()
        self._resources: dict[str, dict[str, str]] = {}
        self._cases: dict[str, str] = {}
        #: case_id -> {"strategy": ..., "branches": ...} as found in each
        #: case document, refreshed by ``_scan_cases``. Bounded by case
        #: count, not case content: only these two leaf fields are kept, so
        #: ``finalize`` can derive honest coverage without retaining the
        #: full corpus in memory.
        self._coverage_index: dict[str, dict[str, Any]] = {}
        manifest = self.root / "manifest.json"
        if manifest.exists():
            if manifest.is_symlink() or not manifest.is_file():
                _fail("manifest", "must be a regular file")
            self._load_manifest(manifest)
        self._scan_cases()

    def _cleanup_temps(self) -> None:
        for directory in (self.root, self.cases_dir):
            for path in directory.glob(".*.tmp"):
                if path.is_dir() and not path.is_symlink():
                    _fail(str(path), "stale temporary path is a directory")
                path.unlink(missing_ok=True)

    def _relative_resource(self, source_path: Path) -> tuple[str, Path]:
        source = Path(source_path)
        candidate = source if source.is_absolute() else self.root / source
        try:
            lexical = candidate.absolute().relative_to(self.root)
        except ValueError as exc:
            raise CheckpointSinkError("resource.path: escapes bundle root") from exc
        if not lexical.parts or any(part in {"", ".", ".."} for part in lexical.parts):
            _fail("resource.path", "expected a safe root-relative path")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.root)
        except (OSError, ValueError) as exc:
            raise CheckpointSinkError("resource.path: missing or escapes bundle root") from exc
        info = resolved.stat()
        if not stat.S_ISREG(info.st_mode):
            _fail("resource.path", "expected a regular file")
        return lexical.as_posix(), resolved

    def write_resource(self, resource_id: str, source_path: Path) -> dict[str, str]:
        """Record one immutable shared resource by root-relative path and hash."""
        resource_id = _identifier(resource_id, "resource_id")
        relative, resolved = self._relative_resource(source_path)
        definition = {
            "resource_id": resource_id,
            "path": relative,
            "sha256": _sha256_file(resolved),
        }
        existing = self._resources.get(resource_id)
        if existing is not None and existing != definition:
            _fail("resource_id", "conflicting redefinition for " + resource_id)
        self._resources[resource_id] = definition
        return dict(definition)

    def write_case(self, case_id: str, case_document: Any) -> str:
        """Atomically write one finite JSON case and return its content hash.

        The common case -- a brand new case_id, e.g. a chooser's multi-member
        checkpoint that repeats a shared fold pool once per member -- streams
        straight to disk (`_atomic_write_stream`/`_iter_json_chunks`): no
        whole-document `str`/`bytes` object is built just to write or hash it.
        A resumed run's duplicate case_id is rare and its documents are not
        the multi-GB ones, so that path keeps the original whole-document
        comparison (`_json_bytes`), unchanged, as the stricter check.
        """
        case_id = _identifier(case_id, "case_id")
        path = self.cases_dir / (case_id + ".json")
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                _fail("case_id", "existing case path is unsafe")
            data = _json_bytes(case_document)
            digest = _sha256_bytes(data)
            actual = _sha256_file(path)
            if actual != digest or path.read_bytes() != data:
                _fail("case_id", "conflicting duplicate for " + case_id)
        else:
            digest = _atomic_write_stream(path, _iter_json_chunks(case_document))
        recorded = self._cases.get(case_id)
        if recorded is not None and recorded != digest:
            _fail("case_id", "hash conflict for " + case_id)
        self._cases[case_id] = digest
        return digest

    def completed_case_ids(self) -> tuple[str, ...]:
        """Return all durable, valid case IDs available for resume."""
        self._scan_cases()
        return tuple(sorted(self._cases))

    def _derive_coverage(self) -> dict[str, dict[str, list[str]]]:
        """Recompute strategy/branch coverage from the cases actually on disk.

        Coverage is never taken on the caller's word: a case only counts once
        its own document is parsed and its ``strategy``/``branches`` fields
        are read back, the same way resource hashes are re-verified against
        the file rather than trusted from the caller.
        """
        strategies: dict[str, list[str]] = {}
        branches: dict[str, list[str]] = {}
        for case_id in sorted(self._cases):
            info = self._coverage_index.get(case_id, {})
            strategy = info.get("strategy")
            if isinstance(strategy, str) and strategy.strip():
                strategies.setdefault(strategy, []).append(case_id)
            case_branches = info.get("branches")
            if isinstance(case_branches, list):
                for branch in case_branches:
                    if isinstance(branch, str) and branch.strip():
                        branches.setdefault(branch, []).append(case_id)
        return {"strategies": strategies, "branches": branches}

    def finalize(self, metadata: Any) -> dict[str, Any]:
        """Verify all references and atomically write a deterministic manifest.

        ``metadata`` must carry a nonempty ``release_id`` -- the one field
        that makes a bundle traceable to the release it came from. Anything
        else the caller puts in ``metadata`` (a source snapshot, a status
        label, ...) rides along verbatim under the ``metadata`` key of the
        finalized manifest; nothing supplied here is dropped.
        """
        _validate_json(metadata, "metadata")
        if not isinstance(metadata, dict):
            _fail("metadata", "expected an object")
        release_id = metadata.get("release_id")
        if not isinstance(release_id, str) or not release_id.strip():
            _fail("metadata.release_id", "expected nonempty string")
        self._scan_cases()
        for resource_id, resource in sorted(self._resources.items()):
            relative, resolved = self._relative_resource(Path(resource["path"]))
            if relative != resource["path"] or _sha256_file(resolved) != resource["sha256"]:
                _fail("resource_id", "hash conflict for " + resource_id)
        body = {
            "schema_version": SCHEMA_VERSION,
            "release_id": release_id,
            "metadata": metadata,
            "resources": [dict(self._resources[key]) for key in sorted(self._resources)],
            "coverage": self._derive_coverage(),
            "cases": [
                {"case_id": key, "sha256": self._cases[key]}
                for key in sorted(self._cases)
            ],
        }
        manifest = dict(body)
        # The same canonicalization ``checks/phase4_checkpoints.validate_bundle``
        # uses to re-check this hash -- plain ``json.dumps(sort_keys=True)``
        # is not it (see engine/v2/foundation/canonical.py's docstring): two
        # independent re-serializations of the same dict only agree if both
        # use the one real JCS implementation.
        manifest["manifest_hash"] = content_hash(body)
        _atomic_write(self.root / "manifest.json", _json_bytes(manifest))
        return manifest

    def _scan_cases(self) -> None:
        discovered: dict[str, str] = {}
        coverage_index: dict[str, dict[str, Any]] = {}
        for path in sorted(self.cases_dir.glob("*.json")):
            if path.is_symlink() or not path.is_file():
                _fail("case", "unsafe case path " + path.name)
            case_id = _identifier(path.stem, "case_id")
            document = _strict_load(path, "case " + case_id)
            discovered[case_id] = _sha256_file(path)
            if isinstance(document, dict):
                coverage_index[case_id] = {
                    "strategy": document.get("strategy"),
                    "branches": document.get("branches"),
                }
        for case_id, digest in self._cases.items():
            if case_id not in discovered:
                _fail("case_id", "manifest case is missing: " + case_id)
            if discovered[case_id] != digest:
                _fail("case_id", "hash conflict for " + case_id)
        self._cases = discovered
        self._coverage_index = coverage_index

    def _load_manifest(self, path: Path) -> None:
        manifest = _strict_load(path, "manifest")
        expected = {
            "schema_version", "release_id", "metadata", "resources", "coverage",
            "cases", "manifest_hash",
        }
        if not isinstance(manifest, dict) or set(manifest) != expected:
            _fail("manifest", "unexpected or missing fields")
        if manifest["schema_version"] != SCHEMA_VERSION:
            _fail("manifest.schema_version", "unsupported")
        body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
        if content_hash(body) != _digest(manifest["manifest_hash"], "manifest_hash"):
            _fail("manifest_hash", "mismatch")
        if not isinstance(manifest["resources"], list):
            _fail("manifest.resources", "expected list")
        for index, row in enumerate(manifest["resources"]):
            label = "manifest.resources[" + str(index) + "]"
            if not isinstance(row, dict) or set(row) != {"resource_id", "path", "sha256"}:
                _fail(label, "malformed")
            resource_id = _identifier(row["resource_id"], label + ".resource_id")
            if resource_id in self._resources:
                _fail(label, "duplicate resource ID")
            _digest(row["sha256"], label + ".sha256")
            relative, resolved = self._relative_resource(Path(row["path"]))
            if relative != row["path"] or _sha256_file(resolved) != row["sha256"]:
                _fail(label, "resource hash mismatch")
            self._resources[resource_id] = dict(row)
        if not isinstance(manifest["cases"], list):
            _fail("manifest.cases", "expected list")
        for index, row in enumerate(manifest["cases"]):
            label = "manifest.cases[" + str(index) + "]"
            if not isinstance(row, dict) or set(row) != {"case_id", "sha256"}:
                _fail(label, "malformed")
            case_id = _identifier(row["case_id"], label + ".case_id")
            if case_id in self._cases:
                _fail(label, "duplicate case ID")
            self._cases[case_id] = _digest(row["sha256"], label + ".sha256")


__all__ = ["CheckpointSinkError", "DiskCheckpointSink", "SCHEMA_VERSION"]
