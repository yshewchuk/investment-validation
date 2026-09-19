"""P5-6 staged-release layout and the required-member catalog.

Shared by ``tools/phase5_prepare_release.py`` (writes a staged release) and
``checks/phase5_acceptance.py`` (verifies one). Checks never import tools, so
the schema lives here.

A staged release root holds two things:

``deployment/``
    An ``engine.v2.models.deployment`` store: ``releases/<id>/manifest.json``,
    the content-addressed ``objects/<sha256>``, and (only if an incumbent was
    copied in) ``DEPLOYED`` plus ``history/``. The model bindings live here and
    are staged through ``stage_release``, so its completeness checks ran.
``phase5_release.json``
    The P5-6 manifest for everything that is NOT a model binding: the frozen
    non-model states (payoff line/surface, recalibration, residual pools, the
    admissible table, the analog matcher) and the Tier-4 serving folds. Their
    objects sit in the same ``deployment/objects/`` store, named by hash.

Every catalog row has a status. ``STAGED`` rows carry objects. ``MISSING``
means the artifact type exists in this tree but the preparer produced no
member. ``PENDING`` means the artifact type or builder has not landed in this
tree yet. The gate fails on both, so a release cannot go green while a member
is still waiting on other work.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from engine.v2.foundation import content_hash

PHASE5_RELEASE_SCHEMA = "phase5_staged_release.v1.0"
MANIFEST_NAME = "phase5_release.json"
DEPLOYMENT_DIR = "deployment"
OBJECTS_DIR = "objects"

STAGED = "STAGED"
MISSING = "MISSING"
PENDING = "PENDING"
MEMBER_STATUSES = (STAGED, MISSING, PENDING)

#: The seven champion bindings P5-1 found in the real registry
#: (guides/rearchitecture_phase5_models.md, coverage table). The gate requires
#: a staged, hash-verified model binding for each.
EXPECTED_MODEL_BINDINGS: tuple[tuple[str, str], ...] = (
    ("size", "*"),
    ("implied_t1", "*"),
    ("runup_move", "*"),
    ("iv_crush", "*"),
    ("gate", "STR-THRU"),
    ("gate", "STR-RUNUP"),
    ("chooser", "DYN-SV"),
)

TIER4_ROLES = ("size", "implied_t1", "runup_move", "iv_crush")


@dataclass(frozen=True)
class StateSpec:
    """One required non-model member of a release.

    ``modules`` are the modules its artifact type needs; when any fails to
    import, the member is PENDING in this tree. ``consumer`` names the v2
    score consumer that reads it (see ``checks.phase5_acceptance.CONSUMERS``);
    ``None`` means no v2 score consumer reads it.
    """

    member_id: str
    kind: str
    strategies: tuple[str, ...]
    modules: tuple[str, ...]
    consumer: str | None
    source: str


_PAYOFF = ("engine.v2.models.payoff_artifact",)
_RESIDUAL = ("engine.v2.models.residual_artifact", "engine.v2.models.frozen_state")
_ADMISSIBLE = ("engine.v2.models.admissible_table", "engine.v2.models.frozen_state")
_CHOOSER_POOL = ("engine.v2.models.chooser_analog_pool", "engine.v2.models.frozen_state")
#: The frozen recalibration-map artifact (P5-4, merged at 3dea05e).
RECALIBRATION_MODULES = ("engine.v2.models.recalibration_artifact",)
#: The frozen board analog-matcher population (P5-4): one
#: ``BoardAnalogPoolArtifact`` per (strategy, alpha, cutoff), loaded through
#: ``FrozenStateLoader``.
ANALOG_MODULES = ("engine.v2.models.analog_artifact", "engine.v2.models.frozen_state")
#: The entry-rule gate's frozen trailing ``pnl_sim`` cutoff (P5-4): one
#: ``TrailingCutoffArtifact`` per event month, loaded through
#: ``FrozenStateLoader``.
TRAILING_CUTOFF_MODULES = ("engine.v2.models.trailing_cutoff_artifact",
                           "engine.v2.models.frozen_state")

STATE_SPECS: tuple[StateSpec, ...] = (
    StateSpec("payoff_line:STR-THRU", "calibration", ("STR-THRU",), _PAYOFF,
              "model_stage.payoff_line", "P5-3 calibration job payoff_artifact.json"),
    StateSpec("payoff_line:STR-RUNUP", "calibration", ("STR-RUNUP",), _PAYOFF,
              None, "P5-3 calibration job payoff_artifact.json"),
    StateSpec("payoff_surface:STR-RUNUP", "calibration", ("STR-RUNUP",), _PAYOFF,
              "model_stage.payoff_surface", "P5-3 calibration job payoff_artifact.json"),
    StateSpec("recalibration_map:STR-THRU", "calibration", ("STR-THRU",),
              RECALIBRATION_MODULES, "model_stage.recalibration", "recalibration artifact"),
    # Legacy never recalibrates STR-RUNUP (engine/score.py), and the v2 model
    # stage refuses a declared map there (UNSUPPORTED_RECALIBRATION): the P5-3
    # recipe still builds it, so it is staged for replay but has no consumer.
    StateSpec("recalibration_map:STR-RUNUP", "calibration", ("STR-RUNUP",),
              RECALIBRATION_MODULES, None, "recalibration artifact"),
    StateSpec("driver_residual_pool:size", "residual_bucket", ("STR-THRU",), _RESIDUAL,
              "model_stage.driver_residual_pool", "P5-4 driver residual pool"),
    StateSpec("driver_residual_pool:implied_t1", "residual_bucket", ("STR-RUNUP",), _RESIDUAL,
              "model_stage.driver_residual_pool", "P5-4 driver residual pool"),
    StateSpec("driver_residual_pool:runup_move", "residual_bucket", ("STR-RUNUP",), _RESIDUAL,
              "model_stage.driver_residual_pool", "P5-4 driver residual pool"),
    StateSpec("paired_residual_pool", "paired_simulation", ("*",), _RESIDUAL,
              "simulation.paired_residual_pool", "P5-4 paired residual pool"),
    StateSpec("admissible_table:dyn_sv", "calibration", ("DYN-SV",), _ADMISSIBLE,
              "chooser.admissible_table", "P5-4 n_admissible table"),
    StateSpec("trailing_pnl_cutoff", "threshold", ("*",), TRAILING_CUTOFF_MODULES,
              "gate.trailing_cutoff",
              "training job --state trailing_pnl_cutoff (one object per event month)"),
    StateSpec("chooser_analog_pool", "residual", ("DYN-SV",), _CHOOSER_POOL,
              "chooser.analog_pool",
              "P5-4 chooser analog pool, built from data/features/chooser_analog_pool.parquet"),
    StateSpec("board_analog_matcher", "residual", ("*",), ANALOG_MODULES,
              "analogs.board_analog_matcher",
              "training job --state board_analog_matcher (one object per causal key)"),
    *(StateSpec(f"tier4_folds:{role}", "estimator", ("*",), (),
                "features.tier4_serving_folds", "data/models/tier4 serving folds")
      for role in TIER4_ROLES),
)


class ReleaseLayoutError(ValueError):
    """The staged release root is malformed."""


def modules_available(modules: tuple[str, ...]) -> tuple[bool, str]:
    """``(True, "")`` if every module imports, else ``(False, first missing)``."""
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError:
            return False, name
    return True, ""


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def object_relpath(member_hash: str) -> str:
    """The deployment store's own content-addressed object path."""
    return f"{OBJECTS_DIR}/" + member_hash.removeprefix("sha256:")


def deployment_root(release_root: Path) -> Path:
    return Path(release_root) / DEPLOYMENT_DIR


def write_object(release_root: Path, payload: bytes) -> tuple[str, str]:
    """Write ``payload`` content-addressed; return ``(content_hash, relpath)``."""
    digest = sha256_bytes(payload)
    rel = object_relpath(digest)
    dest = deployment_root(release_root) / rel
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(dest)
    return digest, rel


def manifest_body(release_id: str, deployment_id: str, members: list[dict],
                  sources: Mapping[str, Any]) -> dict:
    body = {
        "schema_version": PHASE5_RELEASE_SCHEMA,
        "release_id": release_id,
        "deployment_id": deployment_id,
        "members": sorted(members, key=lambda row: row["member_id"]),
        "sources": dict(sources),
    }
    body["manifest_hash"] = content_hash(body)
    return body


def write_manifest(release_root: Path, body: Mapping[str, Any]) -> Path:
    path = Path(release_root) / MANIFEST_NAME
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return path


def read_manifest(release_root: Path) -> dict:
    path = Path(release_root) / MANIFEST_NAME
    if not path.is_file():
        raise ReleaseLayoutError(f"no {MANIFEST_NAME} under {release_root}")
    body = json.loads(path.read_text())
    if body.get("schema_version") != PHASE5_RELEASE_SCHEMA:
        raise ReleaseLayoutError("unknown phase5 release schema")
    claimed = body.get("manifest_hash")
    unhashed = {key: value for key, value in body.items() if key != "manifest_hash"}
    if claimed != content_hash(unhashed):
        raise ReleaseLayoutError("phase5_release.json does not match its manifest_hash")
    return body


def member_row(spec: StateSpec, status: str, objects: list[dict], detail: str) -> dict:
    if status not in MEMBER_STATUSES:
        raise ReleaseLayoutError(f"unknown member status {status!r}")
    return {
        "member_id": spec.member_id, "kind": spec.kind,
        "strategies": list(spec.strategies), "status": status,
        "objects": sorted(objects, key=lambda row: row["name"]), "detail": detail,
    }


__all__ = [
    "ANALOG_MODULES", "DEPLOYMENT_DIR", "EXPECTED_MODEL_BINDINGS", "MANIFEST_NAME",
    "MEMBER_STATUSES", "MISSING", "OBJECTS_DIR", "PENDING", "PHASE5_RELEASE_SCHEMA",
    "RECALIBRATION_MODULES", "STAGED", "STATE_SPECS", "TIER4_ROLES",
    "TRAILING_CUTOFF_MODULES", "ReleaseLayoutError", "StateSpec", "deployment_root",
    "manifest_body", "member_row", "modules_available", "object_relpath",
    "read_manifest", "sha256_bytes", "write_manifest", "write_object",
]
