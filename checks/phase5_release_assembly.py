"""Pool release assembly: carry a staged release's pool objects forward (#192 slice 2).

Implements the contract in ``engine/v2/models/ARCHITECTURE.md`` section 2.1.
The phase5 catalog and its layout helpers live in ``checks/phase5_release.py``;
this module only derives an in-memory candidate catalog and stages it under a
new ``release_id``. ``derive_catalog`` is pure; nothing here ever touches
``DEPLOYED`` or promotes a release.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from checks.phase5_release import (
    STAGED,
    STATE_SPECS,
    ReleaseLayoutError,
    _catalog_path,
    _read_catalog,
    _validate_catalog,
    _verify_staged,
    _write_catalog,
    deployment_root,
    manifest_body,
    member_row,
    object_relpath,
    sha256_bytes,
)
from engine.v2.models import deployment

POOL_MEMBER_IDS: tuple[str, ...] = (
    "driver_residual_pool:size",
    "driver_residual_pool:implied_t1",
    "driver_residual_pool:runup_move",
    "paired_residual_pool",
    "trailing_pnl_cutoff",
    "board_analog_matcher",
)
CHOOSER_MEMBER_ID = "chooser_analog_pool"
_REFUSAL_CODES = (
    "PRIOR_MISMATCH", "POOL_SET_MISMATCH", "CUTOFF_MISMATCH", "CHOOSER_MISMATCH",
    "CUTOFF_NOT_ADVANCED", "OBJECT_UNVERIFIED", "ID_CONFLICT",
)
_REQUIRED_MEMBER_IDS = (*POOL_MEMBER_IDS, CHOOSER_MEMBER_ID)
_SPEC_BY_ID = {spec.member_id: spec for spec in STATE_SPECS}
_HEX = frozenset("0123456789abcdef")


class ReleaseAssemblyRefusal(ValueError):
    """A carry-forward refused. ``code`` is one of the stable refusal codes."""

    def __init__(self, code: str, detail: str) -> None:
        if code not in _REFUSAL_CODES:
            raise ValueError(f"unknown refusal code {code!r}")
        self.code = code
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class PoolOutput:
    member_id: str
    name: str
    content_hash: str
    cutoff: str
    payload_bytes: int


def _content_hash_valid(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value[len("sha256:"):]
    return len(digest) == 64 and all(char in _HEX for char in digest)


def _parse_day(value: object, code: str, detail: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ReleaseAssemblyRefusal(code, detail) from exc


def _outputs_by_member(outputs: object) -> dict[str, PoolOutput]:
    if not isinstance(outputs, (list, tuple)):
        raise ReleaseAssemblyRefusal("POOL_SET_MISMATCH", "outputs are not a sequence")
    by_member: dict[str, PoolOutput] = {}
    for output in outputs:
        if not isinstance(output, PoolOutput):
            raise ReleaseAssemblyRefusal("POOL_SET_MISMATCH", "an output is not a PoolOutput")
        if output.member_id in by_member:
            raise ReleaseAssemblyRefusal("POOL_SET_MISMATCH", "a member id appears twice")
        by_member[output.member_id] = output
    if set(by_member) != set(POOL_MEMBER_IDS):
        raise ReleaseAssemblyRefusal("POOL_SET_MISMATCH", "outputs are not exactly the six pool members")
    for output in by_member.values():
        if not _content_hash_valid(output.content_hash):
            raise ReleaseAssemblyRefusal("POOL_SET_MISMATCH", "an output hash is malformed")
    return by_member


def _prior_pool_rows(members: object) -> None:
    if not isinstance(members, list):
        raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "prior members are not a list")
    counts: dict[str, int] = {}
    rows: dict[str, object] = {}
    for row in members:
        if not isinstance(row, dict):
            raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "a prior member row is malformed")
        member_id = row.get("member_id")
        if member_id in _REQUIRED_MEMBER_IDS:
            counts[member_id] = counts.get(member_id, 0) + 1
            rows[member_id] = row
    for member_id in _REQUIRED_MEMBER_IDS:
        if counts.get(member_id) != 1 or rows[member_id].get("status") != STAGED:
            raise ReleaseAssemblyRefusal(
                "PRIOR_MISMATCH", "a pool/chooser row is missing, duplicated or not STAGED")


def derive_catalog(prior_catalog: object, *, release_id: str, expected_prior_release_id: str,
                   outputs: object, chooser: PoolOutput, cutoff: str) -> dict:
    """Return the pure candidate catalog body for carrying ``prior_catalog`` forward."""
    try:
        prior = _validate_catalog(prior_catalog, expected_prior_release_id)
    except ReleaseLayoutError as exc:
        raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "prior catalog is malformed or names another release") from exc
    sources = prior.get("sources")
    prior_cutoff = sources.get("pool_cutoff") if isinstance(sources, dict) else None
    if not isinstance(prior_cutoff, str):
        raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "prior catalog has no string pool_cutoff")
    _prior_pool_rows(prior.get("members"))
    prior_day = _parse_day(prior_cutoff, "PRIOR_MISMATCH", "prior pool_cutoff is not an ISO date")

    if not isinstance(release_id, str) or release_id == prior["release_id"]:
        raise ReleaseAssemblyRefusal("ID_CONFLICT", "candidate release id is not a new string")

    by_member = _outputs_by_member(outputs)
    if not _content_hash_valid(chooser.content_hash):
        raise ReleaseAssemblyRefusal("POOL_SET_MISMATCH", "the chooser hash is malformed")
    if chooser.member_id != CHOOSER_MEMBER_ID:
        raise ReleaseAssemblyRefusal("CHOOSER_MISMATCH", "chooser member id is not chooser_analog_pool")

    cutoff_day = _parse_day(cutoff, "CUTOFF_MISMATCH", "cutoff is not an ISO date")
    for output in (*by_member.values(), chooser):
        if output.cutoff != cutoff:
            raise ReleaseAssemblyRefusal("CUTOFF_MISMATCH", "an object cutoff disagrees with cutoff")
    if cutoff_day <= prior_day:
        raise ReleaseAssemblyRefusal("CUTOFF_NOT_ADVANCED", "cutoff is not after the prior pool_cutoff")

    chosen = {output.member_id: output for output in (*by_member.values(), chooser)}
    rows = copy.deepcopy(prior["members"])
    position = {row["member_id"]: index for index, row in enumerate(rows)}
    for member_id in _REQUIRED_MEMBER_IDS:
        output = chosen[member_id]
        objects = [{
            "name": output.name,
            "path": object_relpath(output.content_hash),
            "content_hash": output.content_hash,
            "bytes": output.payload_bytes,
        }]
        rows[position[member_id]] = member_row(
            _SPEC_BY_ID[member_id], STAGED, objects, f"carried forward {cutoff}")
    carried = {**sources, "pool_cutoff": cutoff, "carried_from": expected_prior_release_id}
    return manifest_body(release_id, prior["deployment_id"], rows, carried)


def _verify_objects(store: Path, supplied: tuple[PoolOutput, ...]) -> None:
    for output in supplied:
        path = store / object_relpath(output.content_hash)
        try:
            if not path.is_file():
                raise ReleaseAssemblyRefusal("OBJECT_UNVERIFIED", "a supplied object is missing")
            payload = path.read_bytes()
        except OSError as exc:
            raise ReleaseAssemblyRefusal("OBJECT_UNVERIFIED", "a supplied object is unreadable") from exc
        if sha256_bytes(payload) != output.content_hash or len(payload) != output.payload_bytes:
            raise ReleaseAssemblyRefusal("OBJECT_UNVERIFIED", "an object disagrees with its claim")


def carry_forward_release(release_root: Path, *, prior_release_id: str, release_id: str,
                          inventory, outputs, chooser: PoolOutput, cutoff: str) -> Path:
    """Verify, re-stage the prior model bindings and write the candidate catalog."""
    root = Path(release_root)
    store = deployment_root(root)
    try:
        prior_catalog = _read_catalog(_catalog_path(root, prior_release_id), prior_release_id)
    except ReleaseLayoutError as exc:
        raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "prior catalog could not be read or validated") from exc
    if prior_catalog is None:
        raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "no prior catalog exists for that release id")
    try:
        _verify_staged(store, prior_release_id)
    except ReleaseLayoutError as exc:
        raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "prior release is not a verified staged release") from exc

    body = derive_catalog(prior_catalog, release_id=release_id,
                          expected_prior_release_id=prior_release_id,
                          outputs=outputs, chooser=chooser, cutoff=cutoff)
    if inventory.release_id != release_id:
        raise ReleaseAssemblyRefusal("ID_CONFLICT", "inventory release id does not match the candidate")

    _verify_objects(store, (*outputs, chooser))

    candidate_path = _catalog_path(root, release_id)
    try:
        existing = _read_catalog(candidate_path, release_id)
    except ReleaseLayoutError as exc:
        raise ReleaseAssemblyRefusal(
            "ID_CONFLICT", "candidate catalog exists but cannot be read or validated") from exc
    if existing is not None:
        if existing == body:
            return candidate_path
        raise ReleaseAssemblyRefusal("ID_CONFLICT", "candidate catalog already exists with different content")

    prior = deployment.resolve_release(store, prior_release_id)
    candidate = dataclasses.replace(prior, release_id=release_id)
    payloads = {}
    for binding in prior.bindings:
        for member in binding.members:
            payloads[member.content_hash] = (store / member.path).read_bytes()
    try:
        deployment.stage_release(store, candidate, inventory, payloads)
    except deployment.StagingRefused as exc:
        if any(issue.code == "RELEASE_ID_REUSED" for issue in exc.issues):
            raise ReleaseAssemblyRefusal(
                "ID_CONFLICT", "candidate release id is already staged with different content") from exc
        raise

    _verify_staged(store, release_id)
    _write_catalog(candidate_path, body)
    return candidate_path


__all__ = [
    "CHOOSER_MEMBER_ID", "POOL_MEMBER_IDS", "PoolOutput", "ReleaseAssemblyRefusal",
    "carry_forward_release", "derive_catalog",
]
