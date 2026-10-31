"""P5-6 staged-release layout and the required-member catalog.

Shared, among others, by ``tools/phase5_prepare_release.py`` (writes a staged release),
``checks/phase5_acceptance.py`` (verifies one) and
``checks/phase5_release_assembly.py``; it lives in
``engine.v2.models`` so checks and tools import it directly.

A staged release root holds two things:

``deployment/``
    An ``engine.v2.models.deployment`` store: ``releases/<id>/manifest.json``,
    the content-addressed ``objects/<sha256>``, and (only if an incumbent was
    copied in) ``DEPLOYED`` plus ``history/``. The model bindings live here and
    are staged through ``stage_release``, so its completeness checks ran.
``deployment/releases/<id>/phase5_release.json``
    The P5-6 catalog for everything that is NOT a model binding: the frozen
    non-model states (payoff line/surface, recalibration, residual pools, the
    admissible table, the analog matcher) and the Tier-4 serving folds. Their
    objects sit in the same ``deployment/objects/`` store, named by hash.
    A root ``phase5_release.json`` copy selects the last staged candidate.

Every catalog row has a status. ``STAGED`` rows carry objects. ``MISSING``
means the artifact type exists in this tree but the preparer produced no
member. ``PENDING`` means the artifact type or builder has not landed in this
tree yet. The gate fails on both, so a release cannot go green while a member
is still waiting on other work.

The public helpers for a staged release are ``catalog_path``, ``validate_catalog``,
``read_catalog``, ``verify_staged``, ``write_catalog`` and ``staged_manifest_path``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from engine.v2.foundation import content_hash
from engine.v2.models import deployment
from engine.v2.models.release_objects import (  # noqa: F401  (re-exported)
    DEPLOYMENT_DIR,
    OBJECTS_DIR,
    ObjectVerificationError,
    ReleaseLayoutError,
    deployment_root,
    object_relpath,
    read_regular_file,
    sha256_bytes,
    write_object,
)

PHASE5_RELEASE_SCHEMA = "phase5_staged_release.v1.0"
MANIFEST_NAME = "phase5_release.json"


class CatalogConflict(ReleaseLayoutError):
    """A candidate catalog already exists with different content."""

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


def _catalog_path(release_root: Path, release_id: str) -> Path:
    if not isinstance(release_id, str):
        raise ReleaseLayoutError("release_id must be a string")
    try:
        return deployment.manifest_path(deployment_root(release_root), release_id).with_name(
            MANIFEST_NAME)
    except deployment.DeploymentError as exc:
        raise ReleaseLayoutError("unsafe release_id") from exc


def _validate_catalog(body: Any, expected_id: str | None = None) -> dict:
    if not isinstance(body, dict) or body.get("schema_version") != PHASE5_RELEASE_SCHEMA:
        raise ReleaseLayoutError("unknown phase5 release schema")
    _catalog_path(Path(), body.get("release_id"))
    if expected_id is not None and body["release_id"] != expected_id:
        raise ReleaseLayoutError("phase5 release_id does not match selected release")
    sources = body.get("sources")
    if isinstance(sources, Mapping) and "retrain_cutoff" in sources:
        try:
            retrain_cutoff = sources["retrain_cutoff"]
            retrain_date = date.fromisoformat(retrain_cutoff)
        except (TypeError, ValueError):
            raise ReleaseLayoutError("invalid phase5 release sources cutoff") from None
        if retrain_date.isoformat() != retrain_cutoff:
            raise ReleaseLayoutError("invalid phase5 release sources cutoff")
        pool_cutoff = sources.get("pool_cutoff")
        if pool_cutoff is not None:
            try:
                pool_date = date.fromisoformat(pool_cutoff)
            except (TypeError, ValueError):
                raise ReleaseLayoutError("invalid phase5 release sources cutoff") from None
            if pool_date.isoformat() != pool_cutoff or retrain_date > pool_date:
                raise ReleaseLayoutError("invalid phase5 release sources cutoff")
    claimed = body.get("manifest_hash")
    unhashed = {key: value for key, value in body.items() if key != "manifest_hash"}
    if claimed != content_hash(unhashed):
        raise ReleaseLayoutError("phase5_release.json does not match its manifest_hash")
    return body


MAX_CATALOG_BYTES = 64 * 1024 * 1024


def _read_catalog(path: Path, expected_id: str | None = None) -> dict | None:
    try:
        data = read_regular_file(path, MAX_CATALOG_BYTES)
        if data is None:
            return None
        return _validate_catalog(json.loads(data), expected_id)
    except (OSError, ValueError, RecursionError) as exc:
        raise ReleaseLayoutError("phase5 catalog could not be read or validated") from exc


def _verify_staged(store: Path, release_id: str) -> None:
    try:
        manifest = deployment.read_manifest(store, release_id)
        if (manifest is None or manifest.release.release_id != release_id
                or not deployment.manifest_hash_matches(manifest)):
            raise ReleaseLayoutError("catalog requires a matching verified staged release")
    except (OSError, ValueError, deployment.DeploymentError) as exc:
        raise ReleaseLayoutError("catalog staged release could not be verified") from exc


def _write_catalog(path: Path, body: Mapping[str, Any]) -> None:
    deployment.atomic_write_bytes(
        path, (json.dumps(body, indent=2, sort_keys=True) + "\n").encode())


def validate_legacy_manifest(store: Path, legacy_path: Path) -> None:
    """Read-only preflight against a deployment store, including custom names."""
    body = _read_catalog(legacy_path)
    if body is not None:
        release_id = body["release_id"]
        _verify_staged(store, release_id)
        path = deployment.manifest_path(store, release_id).with_name(MANIFEST_NAME)
        _read_catalog(path, release_id)


def preserve_legacy_manifest(release_root: Path, legacy_path: Path | None = None) -> None:
    """Preserve a verified legacy catalog without replacing local authority."""
    body = _read_catalog(legacy_path if legacy_path is not None
                         else Path(release_root) / MANIFEST_NAME)
    if body is None:
        return
    release_id = body["release_id"]
    _verify_staged(deployment_root(release_root), release_id)
    path = _catalog_path(release_root, release_id)
    if _read_catalog(path, release_id) is None:
        _write_catalog(path, body)


def write_manifest(release_root: Path, body: Mapping[str, Any], *,
                   candidate_only: bool = False) -> Path:
    body = _validate_catalog(dict(body))
    path = _catalog_path(release_root, body["release_id"])
    if candidate_only:
        try:
            existing = _read_catalog(path)
        except ReleaseLayoutError as exc:
            raise CatalogConflict("candidate catalog exists but could not be read") from exc
        if existing is not None:
            existing_unhashed = {key: value for key, value in existing.items()
                                 if key != "manifest_hash"}
            new_unhashed = {key: value for key, value in body.items()
                            if key != "manifest_hash"}
            if existing_unhashed != new_unhashed:
                raise CatalogConflict("candidate catalog exists with different content")
    else:
        preserve_legacy_manifest(release_root)
    _verify_staged(deployment_root(release_root), body["release_id"])
    _write_catalog(path, body)
    if not candidate_only:
        _write_catalog(Path(release_root) / MANIFEST_NAME, body)
    return path


def read_manifest(release_root: Path) -> dict:
    """Read the root-selected candidate, preferring its release-local catalog."""
    body = _read_catalog(Path(release_root) / MANIFEST_NAME)
    if body is None:
        raise ReleaseLayoutError(f"no {MANIFEST_NAME} under {release_root}")
    local = _read_catalog(_catalog_path(release_root, body["release_id"]), body["release_id"])
    return body if local is None else local


def member_row(spec: StateSpec, status: str, objects: list[dict], detail: str) -> dict:
    if status not in MEMBER_STATUSES:
        raise ReleaseLayoutError(f"unknown member status {status!r}")
    return {
        "member_id": spec.member_id, "kind": spec.kind,
        "strategies": list(spec.strategies), "status": status,
        "objects": sorted(objects, key=lambda row: row["name"]), "detail": detail,
    }


catalog_path = _catalog_path
validate_catalog = _validate_catalog
read_catalog = _read_catalog
verify_staged = _verify_staged
write_catalog = _write_catalog


def staged_manifest_path(store, release_id):
    return deployment.manifest_path(store, release_id)


__all__ = [
    "ANALOG_MODULES", "CatalogConflict", "DEPLOYMENT_DIR", "EXPECTED_MODEL_BINDINGS",
    "MANIFEST_NAME",
    "MAX_CATALOG_BYTES", "MEMBER_STATUSES", "MISSING", "OBJECTS_DIR",
    "ObjectVerificationError", "PENDING",
    "PHASE5_RELEASE_SCHEMA",
    "RECALIBRATION_MODULES", "STAGED", "STATE_SPECS", "TIER4_ROLES",
    "TRAILING_CUTOFF_MODULES", "ReleaseLayoutError", "StateSpec", "catalog_path",
    "deployment_root", "manifest_body", "member_row", "object_relpath",
    "preserve_legacy_manifest", "read_catalog", "read_manifest", "read_regular_file",
    "sha256_bytes",
    "staged_manifest_path", "validate_catalog", "validate_legacy_manifest", "verify_staged",
    "write_catalog", "write_manifest", "write_object",
]
