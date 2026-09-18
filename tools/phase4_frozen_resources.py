"""Package verified model artifacts for Phase 4 frozen native scoring.

The packager accepts source-owned model binding metadata and copies only the
referenced artifacts into a release-local resource directory.  It emits the
strict sidecar consumed by :mod:`checks.phase4_frozen_bridge`; prediction
values and scoring answers never enter the package.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from checks.phase4_frozen_bridge import FROZEN_RELEASE_SCHEMA, FROZEN_TRACE_SCHEMA
from engine.v2.foundation import content_hash

_HASH_PREFIX = "sha256:"
_DEFAULT_ADAPTER = "joblib-estimator.v1"
_CHUNK_SIZE = 1 << 20


class FrozenResourceError(ValueError):
    """Captured model bindings cannot form a safe frozen resource package."""


@dataclass(frozen=True)
class FrozenResourcePackage:
    """Files and declarations required by the strict frozen replay bridge."""

    resource_rows: tuple[dict[str, Any], ...]
    sidecar_document: dict[str, Any]
    trace_declaration: dict[str, Any]
    request_refs: tuple[str, ...]


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrozenResourceError(f"{label}: expected nonempty string")
    return value.strip()


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise FrozenResourceError(f"{label}: expected nonempty list")
    result = tuple(_nonempty(item, f"{label}[]") for item in value)
    if len(result) != len(set(result)):
        raise FrozenResourceError(f"{label}: duplicate values")
    return result


def _alias(
    row: Mapping[str, Any],
    primary: str,
    alternate: str,
    label: str,
) -> Any:
    first = row.get(primary)
    second = row.get(alternate)
    if first is not None and second is not None and first != second:
        raise FrozenResourceError(f"{label}: conflicting {primary} and {alternate}")
    value = first if first is not None else second
    if value is None:
        raise FrozenResourceError(f"{label}.{primary}: missing")
    return value


def _digest(value: Any, label: str) -> str:
    text = _nonempty(value, label).lower()
    if text.startswith(_HASH_PREFIX):
        text = text[len(_HASH_PREFIX) :]
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise FrozenResourceError(f"{label}: expected sha256 digest")
    return _HASH_PREFIX + text


def _hex_hash(value: Any) -> str:
    digest = content_hash(value)
    if not digest.startswith(_HASH_PREFIX):
        raise FrozenResourceError("content hash did not use sha256")
    return digest[len(_HASH_PREFIX) :]


def _safe_source(raw: Any, source_root: Path, label: str) -> Path:
    value = _nonempty(raw, label)
    root = source_root.resolve()
    path = Path(value)
    unresolved = path if path.is_absolute() else root / path
    if unresolved.is_symlink():
        raise FrozenResourceError(f"{label}: symbolic links are unsupported")
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FrozenResourceError(f"{label}: escapes source root") from exc
    if not candidate.is_file():
        raise FrozenResourceError(f"{label}: artifact missing")
    return candidate


def _safe_destination(release_root: Path, relative: str) -> Path:
    root = release_root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FrozenResourceError("generated resource path escapes release root") from exc
    return candidate


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK_SIZE):
            digest.update(block)
    return _HASH_PREFIX + digest.hexdigest()


def _copy_verified(source: Path, destination: Path, expected: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".phase4-artifact-", dir=destination.parent, delete=False
        ) as target:
            temporary_name = target.name
            digest = hashlib.sha256()
            with source.open("rb") as origin:
                while block := origin.read(_CHUNK_SIZE):
                    digest.update(block)
                    target.write(block)
            target.flush()
            os.fsync(target.fileno())
        actual = _HASH_PREFIX + digest.hexdigest()
        if actual != expected:
            raise FrozenResourceError(
                f"artifact sha256 mismatch: expected {expected}, found {actual}"
            )
        if destination.exists():
            if not destination.is_file() or _file_digest(destination) != expected:
                raise FrozenResourceError(
                    f"release artifact already exists with different bytes: {destination.name}"
                )
            return
        os.replace(temporary_name, destination)
        temporary_name = None
        if _file_digest(destination) != expected:
            raise FrozenResourceError("copied artifact failed sha256 verification")
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _write_sidecar(path: Path, document: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")
    digest = _HASH_PREFIX + hashlib.sha256(raw).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != raw:
            raise FrozenResourceError(
                f"frozen release sidecar already exists with different bytes: {path.name}"
            )
        return digest
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".phase4-sidecar-", dir=path.parent, delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return digest


def _normalized_binding(
    raw: Any,
    *,
    index: int,
    source_root: Path,
) -> dict[str, Any]:
    label = f"model_bindings[{index}]"
    if not isinstance(raw, Mapping):
        raise FrozenResourceError(f"{label}: expected object")
    artifact_value = _alias(raw, "artifact", "artifact_path", label)
    digest_value = _alias(raw, "artifact_sha256", "artifact_hash", label)
    strategy_value = _alias(raw, "strategy", "strategy_id", label)
    clock_value = _alias(raw, "decision_clock", "decision_clock_id", label)
    digest = _digest(digest_value, f"{label}.artifact_sha256")
    source = _safe_source(artifact_value, source_root, f"{label}.artifact")
    adapter = _nonempty(raw.get("adapter", _DEFAULT_ADAPTER), f"{label}.adapter")
    if adapter not in {"joblib-estimator.v1", "json-linear.v1"}:
        raise FrozenResourceError(f"{label}.adapter: unsupported {adapter}")
    return {
        "model_id": _nonempty(raw.get("model_id"), f"{label}.model_id"),
        "role": _nonempty(raw.get("role"), f"{label}.role"),
        "strategy_id": _nonempty(strategy_value, f"{label}.strategy"),
        "decision_clock_id": _nonempty(clock_value, f"{label}.decision_clock"),
        "adapter": adapter,
        "feature_order": _strings(raw.get("feature_order"), f"{label}.feature_order"),
        "output_names": _strings(raw.get("output_names"), f"{label}.output_names"),
        "artifact_sha256": digest,
        "source": source,
    }


def package_frozen_resources(
    *,
    model_bindings: Sequence[Mapping[str, Any]],
    deployment_id: str,
    release_root: Path,
    source_root: Path | None = None,
) -> FrozenResourcePackage:
    """Copy frozen artifacts and emit a strict Phase 4 release sidecar.

    Relative captured artifact paths are resolved beneath ``source_root``
    (the current working directory by default).  Generated resource paths are
    always relative to ``release_root`` and contain only verified digests.
    """
    deployment = _nonempty(deployment_id, "deployment_id")
    if not isinstance(model_bindings, (list, tuple)) or not model_bindings:
        raise FrozenResourceError("model_bindings: expected nonempty list")
    source_base = Path.cwd() if source_root is None else Path(source_root)
    release_base = Path(release_root)
    release_base.mkdir(parents=True, exist_ok=True)

    normalized = [
        _normalized_binding(row, index=index, source_root=source_base)
        for index, row in enumerate(model_bindings)
    ]
    strategies = {item["strategy_id"] for item in normalized}
    decision_clocks = {item["decision_clock_id"] for item in normalized}
    if len(strategies) != 1:
        raise FrozenResourceError("model_bindings: mixed strategies")
    if len(decision_clocks) != 1:
        raise FrozenResourceError("model_bindings: mixed decision clocks")
    bindings: list[dict[str, Any]] = []
    artifact_rows: dict[str, dict[str, Any]] = {}
    base_roles: set[tuple[str, str, str]] = set()
    binding_ids: set[str] = set()
    for item in normalized:
        role_key = (
            item["strategy_id"],
            item["decision_clock_id"],
            item["role"].split(":", 1)[0],
        )
        if role_key in base_roles:
            raise FrozenResourceError(
                "model_bindings: duplicate role for strategy and decision clock"
            )
        base_roles.add(role_key)
        identity = {
            key: item[key]
            for key in (
                "model_id",
                "role",
                "strategy_id",
                "decision_clock_id",
                "adapter",
                "feature_order",
                "output_names",
                "artifact_sha256",
            )
        }
        identity_hex = _hex_hash(identity)
        binding_id = f"phase4-binding-{identity_hex}"
        request_ref = f"model:{identity_hex}"
        if binding_id in binding_ids:
            raise FrozenResourceError("model_bindings: duplicate binding")
        binding_ids.add(binding_id)

        artifact_hex = item["artifact_sha256"][len(_HASH_PREFIX) :]
        artifact_id = f"phase4-artifact-{artifact_hex}"
        artifact_relative = f"resources/models/{artifact_hex}.artifact"
        artifact_path = _safe_destination(release_base, artifact_relative)
        _copy_verified(item["source"], artifact_path, item["artifact_sha256"])
        artifact_rows.setdefault(
            artifact_id,
            {
                "resource_id": artifact_id,
                "ref": f"artifact:{artifact_hex}",
                "kind": "artifact",
                "path": artifact_relative,
                "sha256": item["artifact_sha256"],
            },
        )
        bindings.append(
            {
                "binding_id": binding_id,
                "model_id": item["model_id"],
                "request_ref": request_ref,
                "role": item["role"],
                "strategy_id": item["strategy_id"],
                "decision_clock_id": item["decision_clock_id"],
                "adapter": item["adapter"],
                "feature_order": list(item["feature_order"]),
                "output_names": list(item["output_names"]),
                "members": [{"name": "estimator", "resource_id": artifact_id}],
            }
        )

    bindings.sort(key=lambda row: row["binding_id"])
    release_identity = {
        "deployment_id": deployment,
        "bindings": bindings,
    }
    release_id = f"phase4-frozen-{_hex_hash(release_identity)}"
    sidecar = {
        "schema_version": FROZEN_RELEASE_SCHEMA,
        "release_id": release_id,
        "deployment_id": deployment,
        "bindings": bindings,
    }
    sidecar_hex = _hex_hash(sidecar)
    sidecar_id = f"phase4-frozen-release-{sidecar_hex}"
    sidecar_relative = f"resources/frozen/{sidecar_hex}.json"
    sidecar_path = _safe_destination(release_base, sidecar_relative)
    sidecar_sha = _write_sidecar(sidecar_path, sidecar)
    sidecar_row = {
        "resource_id": sidecar_id,
        "ref": f"frozen-release:{release_id}",
        "kind": "sidecar",
        "path": sidecar_relative,
        "sha256": sidecar_sha,
        "content_hash": content_hash(sidecar),
    }
    rows = (sidecar_row, *(artifact_rows[key] for key in sorted(artifact_rows)))
    ordered_binding_ids = tuple(row["binding_id"] for row in bindings)
    declaration = {
        "schema_version": FROZEN_TRACE_SCHEMA,
        "release_resource_id": sidecar_id,
        "binding_ids": list(ordered_binding_ids),
    }
    return FrozenResourcePackage(
        resource_rows=tuple(dict(row) for row in rows),
        sidecar_document=sidecar,
        trace_declaration=declaration,
        request_refs=tuple(row["request_ref"] for row in bindings),
    )


__all__ = [
    "FrozenResourceError",
    "FrozenResourcePackage",
    "package_frozen_resources",
]
