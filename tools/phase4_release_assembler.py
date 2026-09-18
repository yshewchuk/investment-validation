"""Assemble hash-bound Phase 4 comparison traces from captured sources."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from engine.v2.contracts import ScoreRequest
from engine.v2.foundation import content_hash, from_document
from engine.v2.scoring.identity import request_hash
from engine.v2.scoring.source_inputs import _ANSWER_FIELDS

TRACE_SCHEMA = "phase4_input_trace.v1.0"
TRANSLATION_SCHEMA = "phase4_input_translation.v1.0"
REQUIRED_STAGES = (
    "resolve_context", "features", "forecast", "geometry", "pricing",
    "analogs", "simulation", "gate", "chooser", "serialization",
)
NATIVE_INPUT_KEYS = frozenset({
    "context", "features", "forecast", "geometry", "pricing", "analogs",
    "simulation", "gate", "chooser", "diagnostics", "source_ref",
})
_FORECAST_ROLE_CONTAINERS = frozenset({"models", "model_artifact_refs"})


class ReleaseAssemblyError(ValueError):
    """Captured material cannot form a strict comparison trace."""


def _leaves(value: Any, path: tuple[Any, ...] = ()) -> dict[tuple[Any, ...], Any]:
    if isinstance(value, Mapping):
        if not value:
            return {path: {}}
        result = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise ReleaseAssemblyError(
                    "input translation object keys must be strings"
                )
            result.update(_leaves(value[key], path + (key,)))
        return result
    if isinstance(value, (list, tuple)):
        if not value:
            return {path: []}
        result = {}
        for index, item in enumerate(value):
            result.update(_leaves(item, path + (index,)))
        return result
    return {path: value}


def _request_hash(request: Mapping[str, Any]) -> str:
    try:
        typed = from_document(ScoreRequest, dict(request))
    except (TypeError, ValueError) as exc:
        raise ReleaseAssemblyError("request is not canonical v2") from exc
    return request_hash(typed)


def _reject_answers(
    value: Any,
    path: tuple[str, ...] = ("native_inputs",),
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key)
            role_declaration = (
                len(path) >= 2
                and path[-2] == "forecast"
                and path[-1] in _FORECAST_ROLE_CONTAINERS
            )
            if name in _ANSWER_FIELDS and not role_declaration:
                location = ".".join((*path, name))
                raise ReleaseAssemblyError(f"{location}: calculated answer")
            _reject_answers(child, (*path, name))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_answers(child, (*path, str(index)))


def _path(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ReleaseAssemblyError(f"{label}: expected nonempty path")
    result: list[Any] = []
    for segment in value:
        if isinstance(segment, str):
            if not segment:
                raise ReleaseAssemblyError(f"{label}: empty path segment")
        elif type(segment) is int:
            if segment < 0:
                raise ReleaseAssemblyError(f"{label}: negative list index")
        else:
            raise ReleaseAssemblyError(f"{label}: invalid path segment")
        result.append(segment)
    return tuple(result)


def _translation(
    shared: Mapping[str, Any],
    native: Mapping[str, Any],
    mappings: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    native_doc = {
        "request": shared["request"],
        "native_inputs": {
            key: value for key, value in native.items()
            if key != "source_ref"
        },
    }
    shared_leaves = _leaves(shared)
    native_leaves = _leaves(native_doc)
    if mappings is None:
        if set(shared_leaves) != set(native_leaves):
            raise ReleaseAssemblyError("translation leaf coverage differs")
        mapping_paths = tuple((path, path) for path in sorted(
            shared_leaves, key=repr,
        ))
    else:
        paths = []
        for index, row in enumerate(mappings):
            label = f"mappings[{index}]"
            if not isinstance(row, Mapping) or set(row) - {
                "shared_path", "native_path", "value_hash",
            }:
                raise ReleaseAssemblyError(f"{label}: malformed")
            if "shared_path" not in row or "native_path" not in row:
                raise ReleaseAssemblyError(f"{label}: malformed")
            paths.append((
                _path(row["shared_path"], f"{label}.shared_path"),
                _path(row["native_path"], f"{label}.native_path"),
            ))
        mapping_paths = tuple(paths)

    rows = []
    shared_paths: set[tuple[Any, ...]] = set()
    native_paths: set[tuple[Any, ...]] = set()
    for index, (shared_path, native_path) in enumerate(mapping_paths):
        label = f"mappings[{index}]"
        if shared_path in shared_paths or native_path in native_paths:
            raise ReleaseAssemblyError(f"{label}: duplicate path")
        if shared_path not in shared_leaves or native_path not in native_leaves:
            raise ReleaseAssemblyError(f"{label}: path is not a document leaf")
        shared_value = shared_leaves[shared_path]
        if content_hash(shared_value) != content_hash(native_leaves[native_path]):
            raise ReleaseAssemblyError(f"{label}: translated values differ")
        rows.append({
            "shared_path": list(shared_path),
            "native_path": list(native_path),
            "value_hash": content_hash(shared_value),
        })
        shared_paths.add(shared_path)
        native_paths.add(native_path)
    if shared_paths != set(shared_leaves) or native_paths != set(native_leaves):
        raise ReleaseAssemblyError("translation leaf coverage mismatch")

    body = {
        "schema_version": TRANSLATION_SCHEMA,
        "shared_input_hash": content_hash(shared),
        "native_input_hash": content_hash(native),
        "mappings": rows,
        "derived": [{
            "native_path": ["native_inputs", "source_ref"],
            "operation": "shared_input_hash",
            "value_hash": content_hash(content_hash(shared)),
        }],
    }
    body["translation_hash"] = content_hash(body)
    return body


def assemble_input_trace(*, request: Mapping[str, Any], shared_inputs: Mapping[str, Any],
                         native_inputs: Mapping[str, Any], observations: Sequence[Any],
                         resources: Sequence[Mapping[str, Any]],
                         metadata: Mapping[str, Any] | None = None,
                         mappings: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Build the strict trace consumed by ``checks.phase4_real``."""
    request_hash = _request_hash(request)
    if shared_inputs.get("request") != request:
        raise ReleaseAssemblyError("shared_inputs.request differs from request")
    unknown_inputs = set(native_inputs) - NATIVE_INPUT_KEYS
    missing_inputs = NATIVE_INPUT_KEYS - set(native_inputs)
    if unknown_inputs or missing_inputs:
        raise ReleaseAssemblyError(
            "native_inputs keys differ: "
            f"unknown={sorted(unknown_inputs)}, missing={sorted(missing_inputs)}"
        )
    _reject_answers(native_inputs)
    shared_hash = content_hash(shared_inputs)
    if native_inputs.get("source_ref") != shared_hash:
        raise ReleaseAssemblyError(
            "native_inputs.source_ref must equal shared input hash"
        )

    by_stage = {}
    allowed_stages = set(REQUIRED_STAGES) | {"diagnostics"}
    for item in observations:
        try:
            stage = item.receipt.stage
            input_hash = item.receipt.input_hash
            output_hash = item.receipt.output_hash
            owner = item.receipt.owner
            input_document = item.input_document
            output_document = item.output_document
        except AttributeError as exc:
            raise ReleaseAssemblyError("observation is malformed") from exc
        if stage not in allowed_stages:
            raise ReleaseAssemblyError(f"unknown native observation stage: {stage}")
        if stage in by_stage:
            raise ReleaseAssemblyError(f"duplicate native observation stage: {stage}")
        if not isinstance(owner, str) or not owner.strip():
            raise ReleaseAssemblyError(f"native observation {stage}: owner missing")
        if input_hash != content_hash(input_document):
            raise ReleaseAssemblyError(
                f"native observation {stage}: input hash mismatch"
            )
        if output_hash != content_hash(output_document):
            raise ReleaseAssemblyError(
                f"native observation {stage}: output hash mismatch"
            )
        by_stage[stage] = item
    missing = sorted(set(REQUIRED_STAGES) - set(by_stage))
    if missing:
        raise ReleaseAssemblyError(f"native observations missing stages: {missing}")
    stages = {}
    for stage in REQUIRED_STAGES:
        item = by_stage[stage]
        stages[stage] = {
            "input": item.input_document,
            "output": item.output_document,
            "input_hash": item.receipt.input_hash,
            "output_hash": item.receipt.output_hash,
            "owner": item.receipt.owner,
        }
    native_hash = content_hash(native_inputs)
    body = {
        "schema_version": TRACE_SCHEMA,
        "request": deepcopy(dict(request)),
        "request_hash": request_hash,
        "shared_inputs": deepcopy(dict(shared_inputs)),
        "shared_input_hash": shared_hash,
        "native_input_hash": native_hash,
        "native_inputs": deepcopy(dict(native_inputs)),
        "native_inputs_hash": native_hash,
        "input_translation": _translation(shared_inputs, native_inputs, mappings),
        "stages": stages,
        "resources": deepcopy(list(resources)),
        "metadata": dict(metadata or {}),
    }
    body["trace_hash"] = content_hash(body)
    return body


__all__ = [
    "NATIVE_INPUT_KEYS", "REQUIRED_STAGES", "ReleaseAssemblyError",
    "assemble_input_trace",
]
