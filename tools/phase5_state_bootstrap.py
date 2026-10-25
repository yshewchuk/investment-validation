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
from typing import Mapping

from tools.phase5_prepare_release import MAX_STATE_BYTES

_FROZEN_SCHEMA_PREFIXES = (
    "driver_residual_pool", "paired_residual_pool", "board_analog_pool",
    "chooser_analog_pool", "trailing_pnl_cutoff", "admissible_depth_table",
)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def classify_bound_files(root: Path, names):
    """Split bound files into frozen states and payoff/recalibration artifacts."""
    from engine.v2.models.frozen_state_io import BoundedStateReadError, read_bounded_state
    from engine.v2.models.payoff_artifact import (
        PAYOFF_LINE_ARTIFACT_V1,
        PAYOFF_SURFACE_ARTIFACT_V1,
    )
    from engine.v2.models.recalibration_artifact import RECALIBRATION_MAP_ARTIFACT_V1
    from tools.phase5_state_stage_job import StageRefused

    frozen_files = []
    artifact_files = []
    for name in names:
        path = Path(root) / name
        try:
            data = read_bounded_state(path, MAX_STATE_BYTES, no_follow=True)
        except (BoundedStateReadError, OSError) as exc:
            raise StageRefused("a bound state file could not be loaded") from exc
        try:
            document = json.loads(data)
        except (ValueError, TypeError, RecursionError) as exc:
            raise StageRefused("a bound state file could not be loaded") from exc
        schema = document.get("schema_version") if isinstance(document, dict) else None
        if not isinstance(schema, str):
            raise StageRefused("a bound state file could not be loaded")
        if schema.startswith(_FROZEN_SCHEMA_PREFIXES):
            frozen_files.append((path, data))
        elif schema in (PAYOFF_LINE_ARTIFACT_V1, PAYOFF_SURFACE_ARTIFACT_V1,
                        RECALIBRATION_MAP_ARTIFACT_V1):
            artifact_files.append((path, data))
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


def _merge_checked(found: dict[str, dict[str, bytes]],
                   more: Mapping[str, Mapping[str, bytes]]) -> None:
    from tools.phase5_state_stage_job import StageRefused

    for member, objects in more.items():
        for name in objects:
            if name in found.get(member, {}):
                raise StageRefused("a state object appears twice", {"member": member})
    for member, objects in more.items():
        found.setdefault(member, {}).update(objects)


def _compare_incumbent_bindings(release_root: Path, prior_release_id, inventory, release):
    from engine.v2.models import deployment
    from engine.v2.models.release_layout import deployment_root
    from tools.phase5_state_stage_job import StageRefused

    try:
        incumbent = deployment.resolve_release(deployment_root(release_root),
                                               prior_release_id)
    except (deployment.DeploymentError, OSError) as exc:
        raise StageRefused("prior release could not be read",
                           {"code": "PRIOR_MISMATCH"}) from exc
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
    from engine.v2.models.frozen_state import FrozenStateError
    from engine.v2.models.frozen_state_io import BoundedStateReadError
    from engine.v2.models.inventory import current_release_inventory
    from engine.v2.models.payoff_artifact import PayoffArtifactError
    from engine.v2.models.pinned_panel import PinnedPanelContext, verify_panel_copy
    from engine.v2.models.recalibration_artifact import RecalibrationArtifactError
    from engine.v2.models.release_layout import (
        CatalogConflict,
        ObjectVerificationError,
        ReleaseLayoutError,
        catalog_path,
        read_catalog,
    )
    from engine.v2.models.release_objects import sha256_bytes
    from tools.phase5_prepare_release import (
        PrepareRefused,
        artifact_file_payloads,
        build_states,
        default_admissible_table,
        fold_policies,
        frozen_state_payloads,
        merge_payloads,
        model_release,
        tier4_fold_payloads,
        write_release,
    )
    from tools.phase5_state_stage_job import (
        StageRefused,
        check_trailing_months,
        light_checks,
        require_regular,
    )

    start = time.time()
    release_root = Path(parameters["release_root"])
    cutoff = parameters["cutoff"]
    bootstrap = parameters["bootstrap"]
    _log(f"[bootstrap] stage start prior={prior_class.kind} reason={prior_class.reason} "
         f"rss_kb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")

    try:
        try:
            inventory, drift = current_release_inventory()
        except (OSError, ValueError, KeyError, TypeError, ArtifactError,
                deployment.DeploymentError, ReleaseAssemblyRefusal) as exc:
            raise StageRefused("release inventory could not be built") from exc
        if drift:
            raise StageRefused("release inventory drifted", {"issues": len(drift)})
        inventory = dataclasses.replace(inventory, release_id=parameters["release_id"])

        try:
            release, payloads = model_release(
                inventory, resolve=lambda ref: Path(ref) if Path(ref).is_absolute() else paths.ROOT / ref)
        except (PrepareRefused, BoundedStateReadError, OSError) as exc:
            raise StageRefused("candidate release could not be prepared") from exc

        if prior_class.kind == PRIOR_NO_POOL_CATALOG:
            _compare_incumbent_bindings(release_root, parameters["prior_release_id"],
                                        inventory, release)

        frozen_files, artifact_files = classify_bound_files(root, parameters["state_files"])

        found: dict[str, dict[str, bytes]] = {}
        try:
            for path, _data in frozen_files:
                require_regular(path, MAX_STATE_BYTES, "a bound state file")
                loaded = frozen_state_payloads([path], max_bytes=MAX_STATE_BYTES)
                returned = [data for objects in loaded.values()
                            for data in objects.values()]
                if (len(returned) != 1
                        or {sha256_bytes(data) for data in returned} != {sha256_bytes(_data)}):
                    raise StageRefused("a bound state file could not be loaded")
                for member, objects in loaded.items():
                    for object_name, data in objects.items():
                        if object_name in found.setdefault(member, {}):
                            raise StageRefused("a state object appears twice",
                                               {"member": member})
                        found[member][object_name] = data
            merge_payloads(found, artifact_file_payloads(artifact_files))
        except StageRefused:
            raise
        except (FrozenStateError, BoundedStateReadError, OSError, ValueError, KeyError,
                TypeError, ArtifactError, RecursionError) as exc:
            raise StageRefused("a bound state file could not be loaded") from exc

        _log(f"[bootstrap] stage inputs loaded members={len(found)} "
             f"objects={sum(len(o) for o in found.values())}")

        ctx = PinnedPanelContext(bootstrap["panel_catalog"], bootstrap["panel_snapshot_id"],
                                 bootstrap["panel_artifact_root"])
        try:
            snapshot = verify_panel_copy(ctx)["panel_copy_sha256"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise StageRefused("candidate release could not be published") from exc

        try:
            _merge_checked(found, default_admissible_table())
            _merge_checked(found, tier4_fold_payloads(
                Path(bootstrap["tier4_dir"]),
                {binding.role: binding.artifact_id for binding in inventory.bindings
                 if binding.strategy_id == "*"},
                snapshot, None))

            size_policy, chooser_policies = fold_policies(found, snapshot)
            states = build_states(found, size_fold_policy=size_policy,
                                  chooser_fold_policies=chooser_policies)
        except StageRefused:
            raise
        except (PrepareRefused, PayoffArtifactError, RecalibrationArtifactError,
                FrozenStateError, BoundedStateReadError, OSError, ValueError, KeyError,
                TypeError, deployment.DeploymentError, ArtifactError,
                ReleaseLayoutError) as exc:
            raise StageRefused("candidate release could not be prepared") from exc

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

        existing_catalog = catalog_path(release_root, parameters["release_id"])
        if existing_catalog.exists():
            try:
                body = read_catalog(existing_catalog, parameters["release_id"])
                if body is not None:
                    existing_hashes = {
                        obj["content_hash"]
                        for row in body.get("members", [])
                        for obj in row.get("objects", [])
                    }
                    candidate_hashes = {
                        sha256_bytes(data)
                        for build in states if build.status == "STAGED"
                        for data in build.payloads.values()
                    }
                    if (body.get("sources", {}).get("pool_cutoff") != cutoff
                            or existing_hashes != candidate_hashes):
                        raise StageRefused("candidate release could not be published",
                                           {"code": "ID_CONFLICT"})
            except StageRefused:
                raise
            except ReleaseLayoutError as exc:
                raise StageRefused("candidate release could not be published",
                                   {"code": "ID_CONFLICT"}) from exc
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise StageRefused("candidate release could not be published") from exc

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
        _log(f"[bootstrap] stage published elapsed={time.time() - start:.1f}s "
             f"members={len(states)} objects={n_objects} "
             f"rss_kb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}")
        return {"release_id": parameters["release_id"],
                "prior_release_id": parameters["prior_release_id"], "cutoff": cutoff,
                "n_new": n_objects, "n_carried": 0}

    except StageRefused as exc:
        code = f" code={exc.details['code']}" if "code" in exc.details else ""
        _log(f"[bootstrap] stage failed elapsed={time.time() - start:.1f}s "
             f"rss_kb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}{code}")
        raise
