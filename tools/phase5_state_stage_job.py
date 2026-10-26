"""Stage the nightly-advanced pools or a monthly retrain candidate release (#192 slice 3).

The worker logic of the ``phase5_state_stage`` ops job (``engine/v2/ops/state_stage.py`` is
the thin job wrapper, as ``engine/v2/ops/training.py`` wraps ``tools/phase5_training_job.py``).
Contract: ``engine/v2/models/ARCHITECTURE.md`` section 2.2 (``phase5_state_stage``).
Never touches ``DEPLOYED``.
"""
from __future__ import annotations

import dataclasses
import stat
from datetime import date
from pathlib import Path

from tools.phase5_prepare_release import MAX_STATE_BYTES

ADVANCING_MEMBERS = ("paired_residual_pool", "board_analog_matcher", "trailing_pnl_cutoff")
MONTHLY_INPUT_MEMBERS = ("driver_residual_pool:size", "driver_residual_pool:implied_t1",
                         "driver_residual_pool:runup_move", "chooser_analog_pool")


class StageRefused(ValueError):
    """A refusal with a fixed message (never path text) and small JSON-safe details."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def require_regular(path: Path, limit: int, what: str) -> None:
    """Refuse a missing, symlinked, non-regular or oversize file before it is read."""
    try:
        info = path.lstat()
    except OSError as exc:
        raise StageRefused(f"{what} could not be read") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise StageRefused(f"{what} is not a bounded regular file")


def object_cutoff(member: str, name: str, cutoff: str) -> str:
    """Paired and board objects record their cutoff as the last name segment; trailing ones are stamped."""
    return cutoff if member == "trailing_pnl_cutoff" else name.rsplit("|", 1)[-1]


def _validate_monthly_object_cutoffs(new: dict[str, dict[str, bytes]],
                                    retrain_cutoff: str) -> None:
    """Every monthly object name must record exactly the retrain cutoff."""
    for member, objects in new.items():
        for name in objects:
            if object_cutoff(member, name, retrain_cutoff) != retrain_cutoff:
                raise StageRefused(
                    "a monthly object name cutoff disagrees with retrain_cutoff",
                    {"code": "CUTOFF_MISMATCH"})


def classify(root: Path, names, *, monthly: bool = False) -> dict[str, dict[str, bytes]]:
    """Load the bound state files; they must cover exactly the required members."""
    from engine.v2.models.frozen_state import FrozenStateError
    from engine.v2.models.frozen_state_io import BoundedStateReadError
    from tools.phase5_prepare_release import frozen_state_payloads

    new: dict[str, dict[str, bytes]] = {}
    try:
        for name in names:
            require_regular(root / name, MAX_STATE_BYTES, "a bound state file")
            for member, objects in frozen_state_payloads(
                    [root / name], max_bytes=MAX_STATE_BYTES).items():
                for object_name, data in objects.items():
                    if object_name in new.setdefault(member, {}):
                        raise StageRefused("a state object appears twice", {"member": member})
                    new[member][object_name] = data
    except StageRefused:
        raise
    except (FrozenStateError, BoundedStateReadError, OSError, ValueError, KeyError, TypeError) as exc:
        raise StageRefused("a bound state file could not be loaded") from exc
    expected = MONTHLY_INPUT_MEMBERS if monthly else ADVANCING_MEMBERS
    if set(new) != set(expected):
        message = ("state files are not exactly the monthly driver and chooser members"
                   if monthly else "state files are not exactly the advancing members")
        raise StageRefused(message, {"members": sorted(new)})
    return new


def check_trailing_months(new: dict[str, dict[str, bytes]], cutoff: str) -> None:
    """A trailing object's month (last name segment, ``YYYY-MM...``) may not start after ``cutoff``."""
    try:
        cutoff_day = date.fromisoformat(cutoff)
    except ValueError as exc:
        raise StageRefused("cutoff is not an ISO date") from exc
    late = []
    for name in new["trailing_pnl_cutoff"]:
        try:
            month_start = date.fromisoformat(name.rsplit("|", 1)[-1][:7] + "-01")
        except ValueError as exc:
            raise StageRefused("a trailing object month is not a date", {"months": [name]}) from exc
        if month_start > cutoff_day:
            late.append(name)
    if late:
        raise StageRefused("a trailing object month starts after cutoff", {"months": sorted(late)})


def light_checks(release_root: Path, parameters, new_hashes: set[str]) -> None:
    """Re-read the candidate catalog and staged manifest; refuse unless it matches the request."""
    from engine.v2.models import release_layout as rel

    release_id = parameters["release_id"]
    try:
        require_regular(rel.catalog_path(release_root, release_id), rel.MAX_CATALOG_BYTES,
                         "the candidate catalog")
        body = rel.read_catalog(rel.catalog_path(release_root, release_id), release_id)
        if body is None:
            raise StageRefused("candidate release failed the light checks")
        rel.verify_staged(rel.deployment_root(release_root), release_id)
        sources = body["sources"]
        present = {item["content_hash"] for row in body["members"]
                   for item in row.get("objects", ())}
        expected_carried_from = (None if parameters.get("mode", "carry") == "bootstrap"
                                 else parameters["prior_release_id"])
        ok = (sources.get("pool_cutoff") == parameters["cutoff"]
              and sources.get("carried_from") == expected_carried_from
              and (parameters.get("mode") != "monthly"
                   or sources.get("retrain_cutoff") == parameters["retrain_cutoff"])
              and new_hashes <= present)
    except StageRefused:
        raise
    except (rel.ReleaseLayoutError, OSError, ValueError, KeyError, TypeError) as exc:
        raise StageRefused("candidate release failed the light checks") from exc
    if not ok:
        raise StageRefused("candidate release failed the light checks")


def _stage_monthly_candidate(parameters, root: Path) -> dict:
    """Assemble and stage the monthly candidate release against the deployed prior."""
    from engine.v2.models import release_layout as rel
    from checks.phase5_release_assembly import (
        CHOOSER_MEMBER_ID,
        PRIOR_USABLE,
        PoolOutput,
        ReleaseAssemblyRefusal,
        carry_forward_release,
        classify_prior,
        prior_carry_set,
    )
    from engine.v2.foundation import ArtifactError
    from engine.v2.models import deployment
    from engine.v2.models.deployment import DeploymentError
    from engine.v2.models.inventory import current_release_inventory

    release_root = Path(parameters["release_root"])
    retrain_cutoff = parameters["cutoff"]
    try:
        pointer = deployment.current_pointer(Path(release_root) / "deployment")
    except (DeploymentError, OSError, ValueError) as exc:
        raise StageRefused("prior release could not be read",
                           {"code": "PRIOR_MISMATCH"}) from exc
    if isinstance(pointer, str):
        base_id = pointer
    else:
        base_id = getattr(pointer, "release_id", None)
        if not isinstance(base_id, str) and isinstance(pointer, tuple) and pointer:
            base_id = pointer[0]
    if not isinstance(base_id, str) or not base_id:
        raise StageRefused("prior release could not be read", {"code": "PRIOR_MISMATCH"})
    try:
        prior_catalog_path = rel.catalog_path(release_root, base_id)
        require_regular(prior_catalog_path, rel.MAX_CATALOG_BYTES, "the prior pool catalog")
        base_catalog = rel.read_catalog(prior_catalog_path, base_id)
    except StageRefused as exc:
        raise StageRefused("prior release could not be read",
                           {"code": "PRIOR_MISMATCH"}) from exc
    except (rel.ReleaseLayoutError, OSError, ValueError, KeyError, TypeError) as exc:
        raise StageRefused("prior release could not be read",
                           {"code": "PRIOR_MISMATCH"}) from exc
    if base_catalog is None:
        raise StageRefused("prior release could not be read", {"code": "PRIOR_MISMATCH"})
    try:
        prior_pool_cutoff = base_catalog["sources"]["pool_cutoff"]
    except (KeyError, TypeError) as exc:
        raise StageRefused("prior release could not be read",
                           {"code": "PRIOR_MISMATCH"}) from exc
    if not isinstance(prior_pool_cutoff, str):
        raise StageRefused("prior release could not be read", {"code": "PRIOR_MISMATCH"})
    try:
        prior_pool_date = date.fromisoformat(prior_pool_cutoff)
    except ValueError as exc:
        raise StageRefused("prior release could not be read",
                           {"code": "PRIOR_MISMATCH"}) from exc
    if prior_pool_date.isoformat() != prior_pool_cutoff:
        raise StageRefused("prior release could not be read", {"code": "PRIOR_MISMATCH"})
    effective_cutoff = max(prior_pool_date, date.fromisoformat(retrain_cutoff)).isoformat()
    prior_class = classify_prior(release_root, base_id)
    if prior_class.kind != PRIOR_USABLE:
        raise StageRefused("prior release could not be read", {"code": "PRIOR_MISMATCH"})
    new = classify(Path(root), parameters["state_files"], monthly=True)
    _validate_monthly_object_cutoffs(new, retrain_cutoff)
    try:
        _, carried = prior_carry_set(release_root, base_id, monthly=True)
    except ReleaseAssemblyRefusal as exc:
        raise StageRefused("prior release could not be read", {"code": exc.code}) from exc
    try:
        inventory, drift = current_release_inventory()
    except StageRefused:
        raise
    except (OSError, ValueError, KeyError, TypeError, ArtifactError, DeploymentError,
            rel.ReleaseLayoutError) as exc:
        raise StageRefused("release inventory could not be built") from exc
    if drift:
        raise StageRefused("release inventory drifted", {"issues": len(drift)})
    inventory = dataclasses.replace(inventory, release_id=parameters["release_id"])
    try:
        outputs = []
        for member, objects in new.items():
            for name, data in sorted(objects.items()):
                digest, _relpath = rel.write_object(release_root, data)
                outputs.append(PoolOutput(member, name, digest, retrain_cutoff,
                                          len(data)))
        n_new = len(outputs)
        new_hashes = {output.content_hash for output in outputs}
        chooser_outputs = [item for item in outputs if item.member_id == CHOOSER_MEMBER_ID]
        driver_outputs = [item for item in outputs if item.member_id != CHOOSER_MEMBER_ID]
    except StageRefused:
        raise
    except (OSError, ValueError, KeyError, TypeError, ArtifactError, DeploymentError,
            rel.ReleaseLayoutError) as exc:
        raise StageRefused("candidate objects could not be written") from exc
    try:
        carry_forward_release(
            release_root, prior_release_id=base_id,
            release_id=parameters["release_id"], inventory=inventory,
            outputs=driver_outputs, chooser=chooser_outputs,
            cutoff=effective_cutoff, monthly=True, retrain_cutoff=retrain_cutoff)
    except ReleaseAssemblyRefusal as exc:
        raise StageRefused("release assembly refused", {"code": exc.code}) from exc
    except (rel.ReleaseLayoutError, OSError) as exc:
        raise StageRefused("candidate release could not be published") from exc
    check_parameters = dict(parameters)
    check_parameters.update(mode="monthly", prior_release_id=base_id,
                            cutoff=effective_cutoff, retrain_cutoff=retrain_cutoff)
    light_checks(release_root, check_parameters, new_hashes)
    try:
        deployment.mark_staging_succeeded(Path(release_root) / "deployment",
                                          parameters["release_id"])
    except (deployment.DeploymentError, OSError) as exc:
        raise StageRefused("staging success could not be recorded") from exc
    return {"release_id": parameters["release_id"],
            "prior_release_id": base_id, "cutoff": retrain_cutoff,
            "n_new": n_new, "n_carried": len(carried)}


#: Re-exported for the pool sidecar (``engine/v2/ops/pool_refresh.py``, #1102
#: slice C): ``engine/v2`` may not import ``checks`` at all (``import_layers``
#: tier 0), so the classification reaches the sidecar through this sanctioned
#: tools layer -- the same layer ``run_state_stage_worker`` already crosses for
#: the worker path. ``stage_candidate`` keeps its own lazy import for the rest.
from checks.phase5_release_assembly import (
    PRIOR_DAMAGED,
    PRIOR_NO_POOL_CATALOG,
    PRIOR_NONE,
    PRIOR_USABLE,
    classify_prior,
)


def stage_candidate(parameters, root: Path) -> dict:
    """Assemble and stage the candidate release; returns a JSON-safe summary."""
    from engine.v2.models import release_layout as rel
    from checks.phase5_release_assembly import (
        CHOOSER_MEMBER_ID,
        PRIOR_DAMAGED,
        PRIOR_USABLE,
        PoolOutput,
        ReleaseAssemblyRefusal,
        carry_forward_release,
        classify_prior,
        prior_carry_set,
    )
    from engine.v2.foundation import ArtifactError
    from engine.v2.models import deployment
    from engine.v2.models.deployment import DeploymentError
    from engine.v2.models.inventory import current_release_inventory

    release_root = Path(parameters["release_root"])
    cutoff = parameters["cutoff"]
    if parameters.get("mode", "carry") == "bootstrap":
        prior_class = classify_prior(release_root, parameters["prior_release_id"])
        if prior_class.kind == PRIOR_DAMAGED:
            raise StageRefused("prior release could not be read", {"code": "PRIOR_MISMATCH"})
        if prior_class.kind == PRIOR_USABLE:
            raise StageRefused("prior release already has a pool catalog",
                               {"code": "PRIOR_USABLE"})
        from tools.phase5_state_bootstrap import bootstrap_candidate

        return bootstrap_candidate(parameters, Path(root), prior_class)
    if parameters.get("mode", "carry") == "monthly":
        return _stage_monthly_candidate(parameters, root)
    new = classify(Path(root), parameters["state_files"])
    check_trailing_months(new, cutoff)
    try:
        _, carried = prior_carry_set(release_root, parameters["prior_release_id"])
    except ReleaseAssemblyRefusal as exc:
        raise StageRefused("prior release could not be read", {"code": exc.code}) from exc
    try:
        inventory, drift = current_release_inventory()
    except StageRefused:
        raise
    except (OSError, ValueError, KeyError, TypeError, ArtifactError, DeploymentError,
            rel.ReleaseLayoutError) as exc:
        raise StageRefused("release inventory could not be built") from exc
    if drift:
        raise StageRefused("release inventory drifted", {"issues": len(drift)})
    inventory = dataclasses.replace(inventory, release_id=parameters["release_id"])
    try:
        outputs = []
        for member, objects in new.items():
            for name, data in sorted(objects.items()):
                digest, _relpath = rel.write_object(release_root, data)
                outputs.append(PoolOutput(member, name, digest,
                                          object_cutoff(member, name, cutoff), len(data)))
        n_new = len(outputs)
        new_hashes = {output.content_hash for output in outputs}
        supplied = {(output.member_id, output.name) for output in outputs}
        outputs += [item for item in carried if item.member_id != CHOOSER_MEMBER_ID
                    and (item.member_id, item.name) not in supplied]
        chooser = [item for item in carried if item.member_id == CHOOSER_MEMBER_ID]
    except StageRefused:
        raise
    except (OSError, ValueError, KeyError, TypeError, ArtifactError, DeploymentError,
            rel.ReleaseLayoutError) as exc:
        raise StageRefused("candidate objects could not be written") from exc
    try:
        carry_forward_release(
            release_root, prior_release_id=parameters["prior_release_id"],
            release_id=parameters["release_id"], inventory=inventory, outputs=outputs,
            chooser=chooser, cutoff=cutoff)
    except ReleaseAssemblyRefusal as exc:
        raise StageRefused("release assembly refused", {"code": exc.code}) from exc
    except (rel.ReleaseLayoutError, OSError) as exc:
        raise StageRefused("candidate release could not be published") from exc
    light_checks(release_root, parameters, new_hashes)
    try:
        deployment.mark_staging_succeeded(Path(release_root) / "deployment",
                                          parameters["release_id"])
    except (deployment.DeploymentError, OSError) as exc:
        raise StageRefused("staging success could not be recorded") from exc
    return {"release_id": parameters["release_id"],
            "prior_release_id": parameters["prior_release_id"], "cutoff": cutoff,
            "n_new": n_new, "n_carried": len(outputs) - n_new + len(chooser)}
