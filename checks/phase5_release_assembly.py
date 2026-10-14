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
import fcntl
from contextlib import contextmanager
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
from engine.v2.models import deployment, releases

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
    "CUTOFF_NOT_ADVANCED", "OBJECT_UNVERIFIED", "ID_CONFLICT", "INVENTORY_MISMATCH",
    "PUBLICATION_FAILED",
)
_REQUIRED_MEMBER_IDS = (*POOL_MEMBER_IDS, CHOOSER_MEMBER_ID)
_SPEC_BY_ID = {spec.member_id: spec for spec in STATE_SPECS}
_HEX = frozenset("0123456789abcdef")
_UNSAFE_ID_CHARS = ("/", "\\", "\0")


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


def _admit(output: object, code: str) -> PoolOutput:
    """Refuse a malformed object with ``code`` before it is used as a mapping key."""
    if not isinstance(output, PoolOutput):
        raise ReleaseAssemblyRefusal(code, "an object is not a PoolOutput")
    if not isinstance(output.member_id, str):
        raise ReleaseAssemblyRefusal(code, "an object member id is not a string")
    if not isinstance(output.name, str) or not output.name:
        raise ReleaseAssemblyRefusal(code, "an object name is not a non-empty string")
    if (not isinstance(output.payload_bytes, int) or isinstance(output.payload_bytes, bool)
            or output.payload_bytes < 0):
        raise ReleaseAssemblyRefusal(code, "an object payload_bytes is not a non-negative int")
    return output


def _outputs_by_member(outputs: object) -> dict[str, PoolOutput]:
    if not isinstance(outputs, (list, tuple)):
        raise ReleaseAssemblyRefusal("POOL_SET_MISMATCH", "outputs are not a sequence")
    by_member: dict[str, PoolOutput] = {}
    for output in outputs:
        _admit(output, "POOL_SET_MISMATCH")
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
        return _derive_catalog(prior_catalog, release_id=release_id,
                               expected_prior_release_id=expected_prior_release_id,
                               outputs=outputs, chooser=chooser, cutoff=cutoff)
    except ReleaseAssemblyRefusal:
        raise
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ReleaseAssemblyRefusal(
            "PRIOR_MISMATCH", "prior catalog rows or identity are malformed") from exc


def _derive_catalog(prior_catalog: object, *, release_id: str, expected_prior_release_id: str,
                    outputs: object, chooser: PoolOutput, cutoff: str) -> dict:
    if not isinstance(expected_prior_release_id, str) or not expected_prior_release_id:
        raise ReleaseAssemblyRefusal(
            "PRIOR_MISMATCH", "expected prior release id must be a non-empty string")
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

    if (not isinstance(release_id, str) or not release_id or release_id.startswith(".")
            or any(char in release_id for char in _UNSAFE_ID_CHARS)):
        raise ReleaseAssemblyRefusal("ID_CONFLICT", "candidate release id is not a safe new string")
    if release_id == prior["release_id"]:
        raise ReleaseAssemblyRefusal("ID_CONFLICT", "candidate release id is not a new string")

    _admit(chooser, "CHOOSER_MISMATCH")
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
            if path.is_symlink() or not path.is_file():
                raise ReleaseAssemblyRefusal("OBJECT_UNVERIFIED", "a supplied object is missing")
            payload = path.read_bytes()
        except OSError as exc:
            raise ReleaseAssemblyRefusal("OBJECT_UNVERIFIED", "a supplied object is unreadable") from exc
        if sha256_bytes(payload) != output.content_hash or len(payload) != output.payload_bytes:
            raise ReleaseAssemblyRefusal("OBJECT_UNVERIFIED", "an object disagrees with its claim")


def _read_prior_payloads(store: Path, prior) -> dict[str, bytes]:
    """Read every prior model binding's payload, refusing symlinks and non-regular files."""
    payloads: dict[str, bytes] = {}
    for binding in prior.bindings:
        for member in binding.members:
            path = store / member.path
            try:
                if path.is_symlink() or not path.is_file():
                    raise ReleaseAssemblyRefusal(
                        "PRIOR_MISMATCH", "a prior model object is missing or unreadable")
                data = path.read_bytes()
            except OSError as exc:
                raise ReleaseAssemblyRefusal(
                    "PRIOR_MISMATCH", "a prior model object is missing or unreadable") from exc
            if sha256_bytes(data) != member.content_hash:
                raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "a prior model object is corrupt")
            payloads[member.content_hash] = data
    return payloads


@contextmanager
def _publication_lock(store: Path):
    """Serialise the stage/verify/write publication block for one release store."""
    store.mkdir(parents=True, exist_ok=True)
    handle = open(store / ".carry_forward.lock", "a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def carry_forward_release(release_root: Path, *, prior_release_id: str, release_id: str,
                          inventory, outputs, chooser: PoolOutput, cutoff: str) -> Path:
    """Verify, re-stage the prior model bindings and write the candidate catalog."""
    if not isinstance(getattr(inventory, "release_id", None), str):
        raise ReleaseAssemblyRefusal("INVENTORY_MISMATCH", "inventory is missing or has no release id")
    try:
        return _carry_forward_release(release_root, prior_release_id=prior_release_id,
                                      release_id=release_id, inventory=inventory, outputs=outputs,
                                      chooser=chooser, cutoff=cutoff)
    except ReleaseAssemblyRefusal:
        raise
    except OSError as exc:
        raise ReleaseAssemblyRefusal(
            "PUBLICATION_FAILED", "a filesystem error occurred while assembling the release") from exc
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ReleaseAssemblyRefusal(
            "PRIOR_MISMATCH", "prior catalog rows or identity are malformed") from exc


def _carry_forward_release(release_root: Path, *, prior_release_id: str, release_id: str,
                           inventory, outputs, chooser: PoolOutput, cutoff: str) -> Path:
    if not isinstance(prior_release_id, str) or not prior_release_id:
        raise ReleaseAssemblyRefusal(
            "PRIOR_MISMATCH", "expected prior release id must be a non-empty string")
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

    body = _derive_catalog(prior_catalog, release_id=release_id,
                           expected_prior_release_id=prior_release_id,
                           outputs=outputs, chooser=chooser, cutoff=cutoff)
    if inventory.release_id != release_id:
        raise ReleaseAssemblyRefusal("ID_CONFLICT", "inventory release id does not match the candidate")

    _verify_objects(store, (*outputs, chooser))

    try:
        prior = deployment.resolve_release(store, prior_release_id)
    except (deployment.DeploymentError, OSError, ValueError) as exc:
        raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "prior release could not be resolved") from exc
    if prior_catalog.get("deployment_id") != prior.deployment_id:
        raise ReleaseAssemblyRefusal(
            "PRIOR_MISMATCH", "prior catalog deployment id disagrees with its staged release")
    if getattr(inventory, "deployment_id", None) != prior.deployment_id:
        raise ReleaseAssemblyRefusal(
            "INVENTORY_MISMATCH", "inventory deployment id does not match the prior release")
    payloads = _read_prior_payloads(store, prior)

    candidate_path = _catalog_path(root, release_id)
    candidate = dataclasses.replace(prior, release_id=release_id)
    with _publication_lock(store):
        try:
            deployment.stage_release(store, candidate, inventory, payloads)
        except deployment.StagingRefused as exc:
            codes = {issue.code for issue in exc.issues}
            if codes & {"RELEASE_ID_REUSED", "MANIFEST_UNREADABLE"}:
                raise ReleaseAssemblyRefusal(
                    "ID_CONFLICT", "candidate release id is already staged with different content") from exc
            if codes & {"MISSING_MEMBER_PAYLOAD", "PAYLOAD_HASH_MISMATCH"}:
                raise ReleaseAssemblyRefusal(
                    "PRIOR_MISMATCH", "a prior model payload is missing or corrupt") from exc
            raise ReleaseAssemblyRefusal(
                "INVENTORY_MISMATCH",
                "inventory is incomplete or incompatible with the prior release") from exc
        except releases.ModelReleaseRefusal as exc:
            raise ReleaseAssemblyRefusal(
                "INVENTORY_MISMATCH",
                "inventory is incomplete or incompatible with the prior release") from exc
        except deployment.DeploymentError as exc:
            raise ReleaseAssemblyRefusal(
                "ID_CONFLICT", "candidate staged manifest is corrupt or conflicting") from exc

        try:
            _verify_staged(store, release_id)
        except ReleaseLayoutError as exc:
            raise ReleaseAssemblyRefusal(
                "ID_CONFLICT", "candidate staged release could not be verified") from exc

        try:
            existing = _read_catalog(candidate_path, release_id)
        except ReleaseLayoutError as exc:
            raise ReleaseAssemblyRefusal(
                "ID_CONFLICT", "candidate catalog exists but cannot be read or validated") from exc
        if existing is not None:
            if existing == body:
                return candidate_path
            raise ReleaseAssemblyRefusal(
                "ID_CONFLICT", "candidate catalog already exists with different content")

        try:
            _write_catalog(candidate_path, body)
        except OSError as exc:
            raise ReleaseAssemblyRefusal(
                "PUBLICATION_FAILED", "candidate catalog could not be written") from exc
    return candidate_path


__all__ = [
    "CHOOSER_MEMBER_ID", "POOL_MEMBER_IDS", "PoolOutput", "ReleaseAssemblyRefusal",
    "carry_forward_release", "derive_catalog",
]
