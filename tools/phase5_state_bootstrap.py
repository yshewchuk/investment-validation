"""Bootstrap assembly for the ``phase5_state_stage`` job (#1102 slice B).

Contract: ``engine/v2/models/ARCHITECTURE.md`` section 2.2.2. Assembles a full
candidate release when the prior is ``NONE`` or ``NO_POOL_CATALOG``; the carry
path lives in ``tools/phase5_state_stage_job.py``. Never touches ``DEPLOYED``.
"""
from __future__ import annotations

import dataclasses
import json
import resource
import sys
import time
from pathlib import Path

MAX_STATE_BYTES = 512 * 1024 * 1024

_FROZEN_SCHEMA_PREFIXES = (
    "driver_residual_pool", "paired_residual_pool", "board_analog_pool",
    "chooser_analog_pool", "trailing_pnl_cutoff", "admissible_table",
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _classify_bound_files(root: Path, names):
    """Split bound files into frozen states and payoff/recalibration artifacts."""
    from engine.v2.models.payoff_artifact import (
        PAYOFF_LINE_ARTIFACT_V1,
        PAYOFF_SURFACE_ARTIFACT_V1,
    )
    from engine.v2.models.recalibration_artifact import RECALIBRATION_MAP_ARTIFACT_V1
    from tools.phase5_state_stage_job import StageRefused, _require_regular

    frozen_files = []
    artifact_files = []
    for name in names:
        path = Path(root) / name
        _require_regular(path, MAX_STATE_BYTES, "a bound state file")
        try:
            document = json.loads(path.read_bytes())
        except (ValueError, TypeError) as exc:
            raise StageRefused("a bound state file could not be loaded") from exc
        schema = document.get("schema_version") if isinstance(document, dict) else None
        if not isinstance(schema, str):
            raise StageRefused("a bound state file could not be loaded")
        if schema.startswith(_FROZEN_SCHEMA_PREFIXES):
            frozen_files.append(path)
        elif schema in (PAYOFF_LINE_ARTIFACT_V1, PAYOFF_SURFACE_ARTIFACT_V1,
                        RECALIBRATION_MAP_ARTIFACT_V1):
            artifact_files.append(path)
        else:
            raise StageRefused("a bound state file could not be loaded")
    return frozen_files, artifact_files


def _binding_view(binding) -> tuple:
    return (
        binding.binding_id, binding.model_id, binding.role, binding.strategy_id,
        binding.decision_clock_id, binding.adapter, tuple(binding.feature_order),
        tuple(binding.output_names),
        tuple(sorted((member.name, member.content_hash) for member in binding.members)),
    )


def _compare_incumbent_bindings(release_root: Path, prior_release_id, inventory, release):
    from engine.v2.models import deployment
    from engine.v2.models.release_layout import deployment_root
    from tools.phase5_state_stage_job import StageRefused

    try:
        incumbent = deployment.resolve_release(deployment_root(release_root),
                                               prior_release_id)
    except (deployment.DeploymentError, OSError) as exc:
        raise StageRefused("candidate bindings differ from the incumbent's",
                           {"code": "BINDINGS_DIFFER"}) from exc
    if inventory.deployment_id != incumbent.deployment_id:
        raise StageRefused("candidate bindings differ from the incumbent's",
                           {"code": "BINDINGS_DIFFER"})
    if {_binding_view(binding) for binding in release.bindings} != {
            _binding_view(binding) for binding in incumbent.bindings}:
        raise StageRefused("candidate bindings differ from the incumbent's",
                           {"code": "BINDINGS_DIFFER"})


def bootstrap_candidate(parameters, root: Path, prior_class) -> dict:
    """Assemble and stage a bootstrap candidate; returns a JSON-safe summary."""
    from checks.phase5_release_assembly import (
        PRIOR_NO_POOL_CATALOG,
        PRIOR_USABLE,
        ReleaseAssemblyRefusal,
        classify_prior,
    )
    from engine import paths
    from engine.v2.foundation import ArtifactError
    from engine.v2.models import deployment
    from engine.v2.models.inventory import current_release_inventory
    from engine.v2.models.pinned_panel import PinnedPanelContext, verify_panel_copy
    from engine.v2.models.release_layout import CatalogConflict, ObjectVerificationError
    from engine.v2.models.release_objects import sha256_bytes
    from tools.phase5_prepare_release import (
        PrepareRefused,
        _fold_policies,
        _merge,
        artifact_file_payloads,
        build_states,
        default_admissible_table,
        frozen_state_payloads,
        model_release,
        tier4_fold_payloads,
        write_release,
    )
    from tools.phase5_state_stage_job import (
        StageRefused,
        _require_regular,
        check_trailing_months,
        light_checks,
    )

    start = time.time()
    release_root = Path(parameters["release_root"])
    cutoff = parameters["cutoff"]
    bootstrap = parameters["bootstrap"]
    _log(f"[bootstrap] stage start prior={prior_class.kind} reason={prior_class.reason}")

    try:
        inventory, drift = current_release_inventory()
    except (OSError, ValueError, KeyError, TypeError, ArtifactError,
            deployment.DeploymentError, ReleaseAssemblyRefusal) as exc:
        raise StageRefused("release inventory could not be built") from exc
    if drift:
        raise StageRefused("release inventory drifted", {"issues": len(drift)})
    inventory = dataclasses.replace(inventory, release_id=parameters["release_id"])
    release, payloads = model_release(
        inventory, resolve=lambda ref: Path(ref) if Path(ref).is_absolute() else paths.ROOT / ref)

    if prior_class.kind == PRIOR_NO_POOL_CATALOG:
        _compare_incumbent_bindings(release_root, parameters["prior_release_id"],
                                    inventory, release)

    frozen_files, artifact_files = _classify_bound_files(root, parameters["state_files"])
    found: dict[str, dict[str, bytes]] = {}
    try:
        for path in frozen_files:
            _require_regular(path, MAX_STATE_BYTES, "a bound state file")
            for member, objects in frozen_state_payloads(
                    [path], max_bytes=MAX_STATE_BYTES).items():
                for object_name, data in objects.items():
                    if object_name in found.setdefault(member, {}):
                        raise StageRefused("a state object appears twice",
                                           {"member": member})
                    found[member][object_name] = data
        _merge(found, artifact_file_payloads(artifact_files))
    except StageRefused:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StageRefused("a bound state file could not be loaded") from exc

    ctx = PinnedPanelContext(bootstrap["panel_catalog"], bootstrap["panel_snapshot_id"],
                             bootstrap["panel_artifact_root"])
    try:
        snapshot = verify_panel_copy(ctx)["panel_copy_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StageRefused("candidate release could not be published") from exc

    _merge(found, default_admissible_table())
    _merge(found, tier4_fold_payloads(
        Path(bootstrap["tier4_dir"]),
        {binding.role: binding.artifact_id for binding in inventory.bindings
         if binding.strategy_id == "*"},
        snapshot, None))

    try:
        size_policy, chooser_policies = _fold_policies(found, snapshot)
        states = build_states(found, size_fold_policy=size_policy,
                              chooser_fold_policies=chooser_policies)
    except (PrepareRefused, ValueError) as exc:
        raise StageRefused("candidate release could not be published") from exc

    empty = sorted(build.spec.member_id for build in states
                   if build.status != "STAGED")
    if empty:
        raise StageRefused("candidate release has an empty member",
                           {"code": "EMPTY_MEMBER", "members": empty})

    cutoff_members = {"paired_residual_pool", "board_analog_matcher",
                      "chooser_analog_pool"}
    bad = []
    for build in states:
        if build.spec.member_id in cutoff_members:
            for name in build.payloads:
                if name.rsplit("|", 1)[-1] != cutoff:
                    bad.append(build.spec.member_id)
                    break
    if bad:
        raise StageRefused("a pool object cutoff differs from the session cutoff",
                           {"members": sorted(bad)})
    trailing = {build.spec.member_id: build.payloads for build in states
                if build.spec.member_id == "trailing_pnl_cutoff"}
    if trailing:
        check_trailing_months(trailing, cutoff)

    try:
        write_release(release_root, release, inventory, payloads, states,
                      sources={"pool_cutoff": cutoff, "carried_from": None},
                      pinned_panel=ctx, candidate_only=True)
    except CatalogConflict as exc:
        raise StageRefused("candidate release could not be published",
                           {"code": "ID_CONFLICT"}) from exc
    except (PrepareRefused, deployment.DeploymentError, ObjectVerificationError,
            OSError, ValueError) as exc:
        raise StageRefused("candidate release could not be published") from exc

    readback = classify_prior(release_root, parameters["release_id"])
    if readback.kind != PRIOR_USABLE:
        raise StageRefused("candidate release failed the light checks")
    new_hashes = {sha256_bytes(data) for build in states
                  for data in build.payloads.values()}
    light_checks(release_root, parameters, new_hashes)
    try:
        deployment.mark_staging_succeeded(Path(release_root) / "deployment",
                                          parameters["release_id"])
    except (deployment.DeploymentError, OSError) as exc:
        raise StageRefused("staging success could not be recorded") from exc

    n_objects = sum(len(build.payloads) for build in states)
    _log(f"[bootstrap] stage end elapsed={time.time() - start:.1f}s "
         f"members={len(states)} objects={n_objects} "
         f"rss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
    return {"release_id": parameters["release_id"],
            "prior_release_id": parameters["prior_release_id"], "cutoff": cutoff,
            "n_new": n_objects, "n_carried": 0}
