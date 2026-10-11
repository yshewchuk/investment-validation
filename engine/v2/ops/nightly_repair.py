"""Generation-aware repair planning only; no stage or state writes (#564 slice 6)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from engine.v2.ops.errors import fail
from engine.v2.ops.nightly_receipts import Effect
from engine.v2.ops.nightly_session import Generation

__all__ = ["OutputBinding", "RepairPlan", "StageSpec", "build_repair_plan"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_ARTIFACT_BYTES, _HASH_CHUNK_BYTES = 512 * 1024 * 1024, 1024 * 1024


@dataclass(frozen=True)
class StageSpec:
    name: str
    dependencies: tuple[str, ...]
    request_digest: str
    implementation_digest: str


@dataclass(frozen=True)
class OutputBinding:
    """A prior succeeded output, retained by reference to its immutable source run."""

    stage: str
    source_run_id: str
    dependencies: tuple[str, ...]
    request_digest: str
    implementation_digest: str
    effect: Effect
    succeeded: bool = True


@dataclass(frozen=True)
class RepairPlan:
    plan_id: str
    run_id: str
    execute: tuple[str, ...]
    inherited: tuple[OutputBinding, ...]


def _valid_effect(effect: Effect) -> bool:
    if (not isinstance(effect, Effect) or not isinstance(effect.kind, str)
            or effect.kind not in {"catalog_job", "artifact", "external"}):
        return False
    if not isinstance(effect.ref, str) or not isinstance(effect.sha256, str) or "\0" in effect.ref:
        return False
    if effect.kind == "artifact":
        return bool(effect.ref and Path(effect.ref).is_absolute() and _SHA256.fullmatch(effect.sha256))
    return bool(effect.ref and not effect.sha256)


def _valid_stage(stage: StageSpec, graph: dict[str, StageSpec]) -> bool:
    return (isinstance(stage, StageSpec) and isinstance(stage.name, str)
            and bool(stage.name.strip()) and stage.name not in graph
            and isinstance(stage.dependencies, tuple)
            and all(isinstance(dep, str) and dep.strip() for dep in stage.dependencies)
            and len(set(stage.dependencies)) == len(stage.dependencies)
            and all(isinstance(value, str) and _SHA256.fullmatch(value)
                    for value in (stage.request_digest, stage.implementation_digest)))


def _validate_stages(stages: Sequence[StageSpec],
                     invalidation: Sequence[str]) -> dict[str, StageSpec]:
    if not isinstance(stages, (tuple, list)) or not stages:
        raise fail("INVALID_REQUEST", "repair planning needs a non-empty stage graph")
    if not isinstance(invalidation, (tuple, list)) or not all(
            isinstance(name, str) and name.strip() for name in invalidation):
        raise fail("INVALID_REQUEST", "invalidation roots must be stage names")
    graph: dict[str, StageSpec] = {}
    for stage in stages:
        if not _valid_stage(stage, graph):
            raise fail("INVALID_REQUEST", "stage graph contains a malformed or duplicate stage")
        if any(dep not in graph for dep in stage.dependencies):
            raise fail("INVALID_REQUEST", "stages must be ordered after their dependencies",
                       details={"stage": stage.name})
        graph[stage.name] = stage
    if any(name not in graph for name in invalidation):
        raise fail("INVALID_REQUEST", "invalidation names an unknown stage")
    return graph


def _valid_binding_identity(binding: OutputBinding, bindings: dict[str, OutputBinding],
                            active_run_id: str) -> bool:
    return (isinstance(binding, OutputBinding) and isinstance(binding.stage, str)
            and bool(binding.stage.strip()) and binding.stage not in bindings
            and isinstance(binding.source_run_id, str) and bool(binding.source_run_id.strip())
            and binding.source_run_id != active_run_id)


def _valid_binding_content(binding: OutputBinding) -> bool:
    return (isinstance(binding.dependencies, tuple)
            and all(isinstance(dep, str) and dep.strip() for dep in binding.dependencies)
            and len(set(binding.dependencies)) == len(binding.dependencies)
            and isinstance(binding.succeeded, bool)
            and all(isinstance(value, str) and _SHA256.fullmatch(value)
                    for value in (binding.request_digest, binding.implementation_digest))
            and _valid_effect(binding.effect))


def _validate_prior(previous: Sequence[OutputBinding], previous_run_id: str | None,
                    active_run_id: str) -> dict[str, OutputBinding]:
    if not isinstance(previous, (tuple, list)):
        raise fail("INVALID_REQUEST", "prior bindings must be a sequence")
    if previous and (not isinstance(previous_run_id, str) or not previous_run_id.strip()
                     or previous_run_id == active_run_id):
        raise fail("INVALID_REQUEST", "prior bindings need a distinct source generation ID")
    bindings: dict[str, OutputBinding] = {}
    for binding in previous:
        if (not _valid_binding_identity(binding, bindings, active_run_id)
                or not _valid_binding_content(binding)):
            raise fail("INVALID_REQUEST", "prior output binding is malformed or from another run")
        bindings[binding.stage] = binding
    return bindings


def _validate(stages: Sequence[StageSpec], generation: Generation,
              previous: Sequence[OutputBinding], previous_run_id: str | None,
              invalidation: Sequence[str]) -> tuple[dict[str, StageSpec], dict[str, OutputBinding]]:
    if (not isinstance(generation, Generation) or type(generation.generation) is not int
            or generation.generation < 1 or not isinstance(generation.run_id, str)
            or not generation.run_id.strip() or generation.status != "started"
            or not isinstance(generation.invalidation, tuple)
            or not all(isinstance(name, str) and name.strip()
                       for name in generation.invalidation)
            or not isinstance(invalidation, (tuple, list))):
        raise fail("INVALID_REQUEST", "repair planning needs a started generation and valid invalidation roots")
    graph = _validate_stages(stages, invalidation)
    bindings = _validate_prior(previous, previous_run_id, generation.run_id)
    return graph, bindings


def _artifact_holds(effect: Effect) -> bool:
    try:
        with os.fdopen(os.open(effect.ref, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_ARTIFACT_BYTES:
                return False
            digest = hashlib.sha256()
            total = 0
            while chunk := stream.read(_HASH_CHUNK_BYTES):
                total += len(chunk)
                if total > _MAX_ARTIFACT_BYTES:
                    return False
                digest.update(chunk)
            return digest.hexdigest() == effect.sha256
    except OSError:
        return False


def _effect_holds(effect: Effect, conn: sqlite3.Connection | None) -> bool:
    if effect.kind == "artifact":
        return _artifact_holds(effect)
    if effect.kind == "catalog_job" and conn is not None:
        try:
            row = conn.execute("SELECT state FROM jobs WHERE job_id = ?", (effect.ref,)).fetchone()
        except sqlite3.Error:
            return False
        return row is not None and row[0] == "succeeded"
    return False


def _plan_identity(generation: Generation, graph: dict[str, StageSpec]) -> str:
    canonical = [{"name": stage.name, "dependencies": sorted(stage.dependencies),
                  "request_digest": stage.request_digest,
                  "implementation_digest": stage.implementation_digest}
                 for stage in sorted(graph.values(), key=lambda item: item.name)]
    identity = json.dumps({"run_id": generation.run_id, "stages": canonical},
                          sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(identity.encode()).hexdigest()


def build_repair_plan(generation: Generation, stages: Sequence[StageSpec], *,
                      previous: Sequence[OutputBinding] = (),
                      previous_run_id: str | None = None,
                      invalidation: Sequence[str] = (),
                      conn: sqlite3.Connection | None = None) -> RepairPlan:
    """Build a generation-specific repair plan from verified prior outputs."""
    graph, prior = _validate(stages, generation, previous, previous_run_id, invalidation)
    roots = {name for name in generation.invalidation if name in graph} | set(invalidation)
    for name, stage in graph.items():
        old = prior.get(name)
        if (old is None or not old.succeeded
                or sorted(old.dependencies) != sorted(stage.dependencies)
                or old.request_digest != stage.request_digest
                or old.implementation_digest != stage.implementation_digest
                or not _effect_holds(old.effect, conn)):
            roots.add(name)
    invalid = set(roots)
    for name, stage in graph.items():
        if any(dep in invalid for dep in stage.dependencies):
            invalid.add(name)
    execute = tuple(name for name in graph if name in invalid)
    inherited = tuple(prior[name] for name in graph if name not in invalid)
    plan_id = _plan_identity(generation, graph)
    return RepairPlan(plan_id, generation.run_id, execute, inherited)
