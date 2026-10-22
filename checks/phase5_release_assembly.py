"""Pool release assembly: carry a staged release's pool objects forward (#192 slice 2).

Implements the contract in ``engine/v2/models/ARCHITECTURE.md`` section 2.1.
The phase5 catalog and its layout helpers live in ``engine/v2/models/release_layout.py``;
this module only derives an in-memory candidate catalog and stages it under a
new ``release_id``. ``derive_catalog`` is pure; nothing here ever touches
``DEPLOYED`` or promotes a release.
"""
from __future__ import annotations

import copy
import dataclasses
import fcntl
import json
import os
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from engine.v2.models import deployment, releases
from engine.v2.models.deployment import MAX_MANIFEST_BYTES
from engine.v2.models.release_layout import (
    MAX_CATALOG_BYTES,
    STAGED,
    STATE_SPECS,
    ReleaseLayoutError,
    catalog_path,
    deployment_root,
    manifest_body,
    member_row,
    object_relpath,
    sha256_bytes,
    staged_manifest_path,
    validate_catalog,
    verify_staged,
    write_catalog,
)

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
_STRAY_ERRORS = (KeyError, TypeError, AttributeError, ValueError, RecursionError)
MAX_OBJECT_BYTES = 512 * 1024 * 1024
LOCK_WAIT_S = 30.0
PRIOR_NONE = "NONE"
PRIOR_NO_POOL_CATALOG = "NO_POOL_CATALOG"
PRIOR_USABLE = "USABLE"
PRIOR_DAMAGED = "DAMAGED"


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
    carried: bool = False


@dataclass(frozen=True)
class PriorClass:
    kind: str
    reason: str


def _content_hash_valid(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value[len("sha256:"):]
    return len(digest) == 64 and all(char in _HEX for char in digest)


def _parse_day(value: object, code: str, detail: str) -> date:
    try:
        if not isinstance(value, str) or value != date.fromisoformat(value).isoformat():
            raise ValueError("not an extended ISO date")
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
    if not isinstance(output.carried, bool):
        raise ReleaseAssemblyRefusal(code, "an object carried flag is not a bool")
    return output


def _require_safe_id(value: object, code: str, detail: str) -> str:
    """Refuse an unsafe release id; return it once admitted."""
    if (not isinstance(value, str) or not value or value.startswith(".")
            or any(char in value for char in _UNSAFE_ID_CHARS)):
        raise ReleaseAssemblyRefusal(code, detail)
    return value


def _normalize_objects(value: object, code: str, detail: str) -> tuple[PoolOutput, ...]:
    """Normalize a single PoolOutput or a sequence of them to a tuple."""
    if isinstance(value, PoolOutput):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(value)
    raise ReleaseAssemblyRefusal(code, detail)


def _all_outputs(by_member: dict[str, tuple[PoolOutput, ...]]) -> tuple[PoolOutput, ...]:
    return tuple(output for member_outputs in by_member.values() for output in member_outputs)


def _outputs_by_member(outputs: object) -> dict[str, tuple[PoolOutput, ...]]:
    if not isinstance(outputs, (list, tuple)):
        raise ReleaseAssemblyRefusal("POOL_SET_MISMATCH", "outputs are not a sequence")
    by_member: dict[str, list[PoolOutput]] = {}
    names_by_member: dict[str, set[str]] = {}
    for output in outputs:
        _admit(output, "POOL_SET_MISMATCH")
        names = names_by_member.setdefault(output.member_id, set())
        if output.name in names:
            raise ReleaseAssemblyRefusal(
                "POOL_SET_MISMATCH", "an object name appears twice within a member")
        names.add(output.name)
        by_member.setdefault(output.member_id, []).append(output)
    if set(by_member) != set(POOL_MEMBER_IDS):
        raise ReleaseAssemblyRefusal("POOL_SET_MISMATCH", "outputs are not exactly the six pool members")
    for member_outputs in by_member.values():
        for output in member_outputs:
            if not _content_hash_valid(output.content_hash):
                raise ReleaseAssemblyRefusal("POOL_SET_MISMATCH", "an output hash is malformed")
    return {member_id: tuple(sorted(member_outputs, key=lambda output: output.name))
            for member_id, member_outputs in by_member.items()}


def _check_carried(output: PoolOutput, cutoff_day: date, prior_rows: dict) -> None:
    """A carried object's cutoff must be an ISO date not after ``cutoff`` and the object must equal
    a prior-row object; it is recorded at the prior ``pool_cutoff``."""
    if _parse_day(output.cutoff, "CUTOFF_MISMATCH", "a carried object cutoff is not an ISO date") > cutoff_day:
        raise ReleaseAssemblyRefusal("CUTOFF_MISMATCH", "a carried object cutoff is after cutoff")
    code = "CHOOSER_MISMATCH" if output.member_id == CHOOSER_MEMBER_ID else "POOL_SET_MISMATCH"
    row = prior_rows.get(output.member_id)
    objects = row.get("objects") if isinstance(row, dict) else None
    if not isinstance(objects, list) or not any(
            isinstance(item, dict) and item.get("name") == output.name
            and item.get("content_hash") == output.content_hash
            and item.get("bytes") == output.payload_bytes for item in objects):
        raise ReleaseAssemblyRefusal(code, "a carried object is not in the prior row")


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
                   outputs: object, chooser: object, cutoff: str) -> dict:
    """Return the pure candidate catalog body for carrying ``prior_catalog`` forward."""
    try:
        return _derive_catalog(prior_catalog, release_id=release_id,
                               expected_prior_release_id=expected_prior_release_id,
                               outputs=outputs, chooser=chooser, cutoff=cutoff)
    except ReleaseAssemblyRefusal:
        raise
    except _STRAY_ERRORS as exc:
        raise ReleaseAssemblyRefusal(
            "PRIOR_MISMATCH", "prior catalog rows or identity are malformed") from exc


def _derive_catalog(prior_catalog: object, *, release_id: str, expected_prior_release_id: str,
                    outputs: object, chooser: object, cutoff: str) -> dict:
    _require_safe_id(
        expected_prior_release_id, "PRIOR_MISMATCH", "prior release id is not a safe string")
    try:
        prior = validate_catalog(prior_catalog, expected_prior_release_id)
    except ReleaseLayoutError as exc:
        raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "prior catalog is malformed or names another release") from exc
    sources = prior.get("sources")
    prior_cutoff = sources.get("pool_cutoff") if isinstance(sources, dict) else None
    if not isinstance(prior_cutoff, str):
        raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "prior catalog has no string pool_cutoff")
    _prior_pool_rows(prior.get("members"))
    prior_day = _parse_day(prior_cutoff, "PRIOR_MISMATCH", "prior pool_cutoff is not an ISO date")

    _require_safe_id(
        release_id, "ID_CONFLICT", "candidate release id is not a safe new string")
    if release_id == prior["release_id"]:
        raise ReleaseAssemblyRefusal("ID_CONFLICT", "candidate release id is not a new string")

    chooser_objects = _normalize_objects(
        chooser, "CHOOSER_MISMATCH", "chooser is not a PoolOutput or a sequence")
    if not chooser_objects:
        raise ReleaseAssemblyRefusal("CHOOSER_MISMATCH", "chooser is empty")
    for output in chooser_objects:
        _admit(output, "CHOOSER_MISMATCH")
    chooser_objects = tuple(sorted(chooser_objects, key=lambda output: output.name))
    by_member = _outputs_by_member(outputs)
    chooser_names: set[str] = set()
    for output in chooser_objects:
        if not _content_hash_valid(output.content_hash):
            raise ReleaseAssemblyRefusal("CHOOSER_MISMATCH", "the chooser hash is malformed")
        if output.member_id != CHOOSER_MEMBER_ID:
            raise ReleaseAssemblyRefusal("CHOOSER_MISMATCH", "chooser member id is not chooser_analog_pool")
        if output.name in chooser_names:
            raise ReleaseAssemblyRefusal("CHOOSER_MISMATCH", "a chooser object name appears twice")
        chooser_names.add(output.name)

    cutoff_day = _parse_day(cutoff, "CUTOFF_MISMATCH", "cutoff is not an ISO date")
    prior_rows = {row["member_id"]: row for row in prior["members"]}
    if cutoff_day <= prior_day:
        raise ReleaseAssemblyRefusal("CUTOFF_NOT_ADVANCED", "cutoff is not after the prior pool_cutoff")
    for output in (*_all_outputs(by_member), *chooser_objects):
        if output.carried:
            _check_carried(output, cutoff_day, prior_rows)
        elif output.cutoff != cutoff:
            raise ReleaseAssemblyRefusal("CUTOFF_MISMATCH", "an object cutoff disagrees with cutoff")

    all_by_member = {**by_member, CHOOSER_MEMBER_ID: chooser_objects}
    rows = copy.deepcopy(prior["members"])
    position = {row["member_id"]: index for index, row in enumerate(rows)}
    for member_id in _REQUIRED_MEMBER_IDS:
        objects = [{
            "name": output.name,
            "path": object_relpath(output.content_hash),
            "content_hash": output.content_hash,
            "bytes": output.payload_bytes,
        } for output in all_by_member[member_id]]
        rows[position[member_id]] = member_row(
            _SPEC_BY_ID[member_id], STAGED, objects, f"carried forward {cutoff}")
    carried = {**sources, "pool_cutoff": cutoff, "carried_from": expected_prior_release_id}
    return manifest_body(release_id, prior["deployment_id"], rows, carried)


def _read_regular(path: Path, limit: int, code: str, detail: str) -> bytes:
    """Read a bounded regular file, refusing symlinks, non-regular files and oversize input."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ReleaseAssemblyRefusal(code, detail) from exc
    try:
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise ReleaseAssemblyRefusal(code, detail)
            with os.fdopen(fd, "rb") as handle:
                fd = -1
                data = handle.read(min(st.st_size, limit) + 1)
        except OSError as exc:
            raise ReleaseAssemblyRefusal(code, detail) from exc
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError as exc:
                raise ReleaseAssemblyRefusal(code, detail) from exc
    if len(data) > limit:
        raise ReleaseAssemblyRefusal(code, detail)
    return data


def _load_catalog(path: Path, expected_id: str, code: str, detail: str) -> dict | None:
    """Load and validate a catalog; the no-follow open below is what guards the read (the lstat only detects absence)."""
    try:
        os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ReleaseAssemblyRefusal(code, detail) from exc
    data = _read_regular(path, MAX_CATALOG_BYTES, code, detail)
    try:
        body = json.loads(data)
    except (ValueError, RecursionError) as exc:
        raise ReleaseAssemblyRefusal(code, detail) from exc
    try:
        validate_catalog(body, expected_id)
    except ReleaseLayoutError as exc:
        raise ReleaseAssemblyRefusal(code, detail) from exc
    return body


def _probe_regular(path: Path, limit: int, code: str, detail: str) -> None:
    """Refuse a symlink, non-regular or oversize metadata file; skip a missing one."""
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ReleaseAssemblyRefusal(code, detail) from exc
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ReleaseAssemblyRefusal(code, detail) from exc
    try:
        try:
            stat_result = os.fstat(fd)
        except OSError as exc:
            raise ReleaseAssemblyRefusal(code, detail) from exc
        if not stat.S_ISREG(stat_result.st_mode) or stat_result.st_size > limit:
            raise ReleaseAssemblyRefusal(code, detail)
    finally:
        try:
            os.close(fd)
        except OSError as exc:
            raise ReleaseAssemblyRefusal(code, detail) from exc


def _verify_objects(store: Path, supplied: tuple[PoolOutput, ...]) -> None:
    for output in supplied:
        payload_bytes = output.payload_bytes
        if payload_bytes > MAX_OBJECT_BYTES:
            raise ReleaseAssemblyRefusal(
                "OBJECT_UNVERIFIED", "a supplied object payload is not a bounded size")
        path = store / object_relpath(output.content_hash)
        payload = _read_regular(
            path, payload_bytes, "OBJECT_UNVERIFIED",
            "a supplied object is missing or unreadable")
        if sha256_bytes(payload) != output.content_hash or len(payload) != payload_bytes:
            raise ReleaseAssemblyRefusal("OBJECT_UNVERIFIED", "an object disagrees with its claim")


def _read_prior_payloads(store: Path, prior) -> dict[str, bytes]:
    """Read every prior model binding's payload, refusing symlinks and non-regular files."""
    payloads: dict[str, bytes] = {}
    for binding in prior.bindings:
        for member in binding.members:
            data = _read_regular(
                store / member.path, MAX_OBJECT_BYTES, "PRIOR_MISMATCH",
                "a prior model object is missing or unreadable")
            if sha256_bytes(data) != member.content_hash:
                raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "a prior model object is corrupt")
            payloads[member.content_hash] = data
    return payloads


def _verify_carried_rows(store: Path, prior_catalog: dict) -> None:
    """Every untouched prior row that is STAGED must still have intact objects."""
    replaced = set(_REQUIRED_MEMBER_IDS)
    for row in prior_catalog["members"]:
        if row.get("member_id") in replaced or row.get("status") != STAGED:
            continue
        for obj in row.get("objects", ()):
            if not isinstance(obj, dict) or not _content_hash_valid(obj.get("content_hash")):
                raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "a carried row object is malformed")
            content_hash = obj["content_hash"]
            data = _read_regular(
                store / object_relpath(content_hash), MAX_OBJECT_BYTES, "PRIOR_MISMATCH",
                "a carried row object is missing or unreadable")
            if sha256_bytes(data) != content_hash or len(data) != obj.get("bytes"):
                raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "a carried row object is corrupt")


@contextmanager
def _publication_lock(store: Path):
    """Serialise the stage/verify/write publication block for one release store."""
    store.mkdir(parents=True, exist_ok=True)
    path = store / ".carry_forward.lock"
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    except OSError as exc:
        raise ReleaseAssemblyRefusal(
            "PUBLICATION_FAILED", "publication lock could not be opened") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ReleaseAssemblyRefusal(
                "PUBLICATION_FAILED", "publication lock is not a regular file")
        deadline = time.monotonic() + LOCK_WAIT_S
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise ReleaseAssemblyRefusal(
                        "PUBLICATION_FAILED",
                        "timed out waiting for the publication lock") from exc
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def classify_prior(release_root: Path, prior_release_id) -> PriorClass:
    """Read-only prior classification for the pool-release bootstrap (#1102); never writes."""
    if prior_release_id is None:
        return PriorClass(PRIOR_NONE, "no prior release was requested")
    try:
        _require_safe_id(prior_release_id, "PRIOR_MISMATCH", "prior release id is not a safe string")
        catalog = _load_catalog(catalog_path(Path(release_root), prior_release_id),
                                prior_release_id, "PRIOR_MISMATCH",
                                "prior catalog could not be read or validated")
    except ReleaseAssemblyRefusal:
        return PriorClass(PRIOR_DAMAGED, "the prior catalog could not be read or validated")
    except _STRAY_ERRORS:
        return PriorClass(PRIOR_DAMAGED, "the prior catalog could not be read or validated")
    if catalog is None:
        if prior_release_id.startswith("pools-"):
            return PriorClass(PRIOR_DAMAGED, "a pools-* release always has a catalog")
        return PriorClass(PRIOR_NO_POOL_CATALOG, "the prior release has no catalog")
    sources = catalog.get("sources")
    if isinstance(sources, dict) and "pool_cutoff" not in sources:
        return PriorClass(PRIOR_NO_POOL_CATALOG, "the prior catalog has no pool_cutoff")
    try:
        _parse_day(sources["pool_cutoff"], "PRIOR_MISMATCH",
                   "prior pool_cutoff is not an ISO date")
        _prior_pool_rows(catalog.get("members"))
    except ReleaseAssemblyRefusal:
        return PriorClass(PRIOR_DAMAGED, "the prior catalog is malformed")
    except _STRAY_ERRORS:
        return PriorClass(PRIOR_DAMAGED, "the prior catalog is malformed")
    return PriorClass(PRIOR_USABLE, "the prior catalog has a pool_cutoff")


def prior_carry_set(release_root: Path, prior_release_id: str) -> tuple[str, tuple[PoolOutput, ...]]:
    """The prior ``sources.pool_cutoff`` and every prior pool/chooser object as ``carried`` outputs.

    Each object keeps the prior release's pool cutoff as its recorded cutoff.
    Malformed or missing prior input refuses ``PRIOR_MISMATCH``.
    """
    try:
        _require_safe_id(prior_release_id, "PRIOR_MISMATCH", "prior release id is not a safe string")
        prior = _load_catalog(catalog_path(Path(release_root), prior_release_id), prior_release_id,
                              "PRIOR_MISMATCH", "prior catalog could not be read or validated")
        if prior is None:
            raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "no prior catalog exists for that release id")
        _prior_pool_rows(prior.get("members"))
        pool_cutoff = prior["sources"]["pool_cutoff"]
        _parse_day(pool_cutoff, "PRIOR_MISMATCH", "prior pool_cutoff is not an ISO date")
        carried = []
        for row in prior["members"]:
            if row.get("member_id") not in _REQUIRED_MEMBER_IDS:
                continue
            for item in row.get("objects", ()):
                carried.append(PoolOutput(row["member_id"], item["name"], item["content_hash"],
                                          pool_cutoff, item["bytes"], carried=True))
        return pool_cutoff, tuple(carried)
    except ReleaseAssemblyRefusal:
        raise
    except _STRAY_ERRORS as exc:
        raise ReleaseAssemblyRefusal(
            "PRIOR_MISMATCH", "prior catalog rows or identity are malformed") from exc


def carry_forward_release(release_root: Path, *, prior_release_id: str, release_id: str,
                          inventory, outputs, chooser: object, cutoff: str) -> Path:
    """Verify, re-stage the prior model bindings and write the candidate catalog.

    Lock or filesystem failures are ``PUBLICATION_FAILED``; a conflicting or
    corrupt candidate is ``ID_CONFLICT``.
    """
    try:
        return _carry_forward_release(release_root, prior_release_id=prior_release_id,
                                      release_id=release_id, inventory=inventory, outputs=outputs,
                                      chooser=chooser, cutoff=cutoff)
    except ReleaseAssemblyRefusal:
        raise
    except OSError as exc:
        raise ReleaseAssemblyRefusal(
            "PUBLICATION_FAILED", "a filesystem error occurred while assembling the release") from exc
    except _STRAY_ERRORS as exc:
        raise ReleaseAssemblyRefusal(
            "PRIOR_MISMATCH", "prior catalog rows or identity are malformed") from exc


def _carry_forward_release(release_root: Path, *, prior_release_id: str, release_id: str,
                           inventory, outputs, chooser: object, cutoff: str) -> Path:
    _require_safe_id(
        prior_release_id, "PRIOR_MISMATCH", "prior release id is not a safe string")
    root = Path(release_root)
    store = deployment_root(root)
    prior_path = catalog_path(root, prior_release_id)
    prior_catalog = _load_catalog(
        prior_path, prior_release_id, "PRIOR_MISMATCH",
        "prior catalog could not be read or validated")
    if prior_catalog is None:
        raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "no prior catalog exists for that release id")
    _probe_regular(staged_manifest_path(store, prior_release_id), MAX_MANIFEST_BYTES,
                   "PRIOR_MISMATCH", "prior manifest is missing or unreadable")
    try:
        verify_staged(store, prior_release_id)
    except ReleaseLayoutError as exc:
        raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "prior release is not a verified staged release") from exc

    if not (isinstance(getattr(inventory, "release_id", None), str)
            and isinstance(getattr(inventory, "artifacts", None), tuple)
            and isinstance(getattr(inventory, "bindings", None), tuple)):
        raise ReleaseAssemblyRefusal("INVENTORY_MISMATCH", "inventory is missing or malformed")

    body = _derive_catalog(prior_catalog, release_id=release_id,
                           expected_prior_release_id=prior_release_id,
                           outputs=outputs, chooser=chooser, cutoff=cutoff)
    if inventory.release_id != release_id:
        raise ReleaseAssemblyRefusal("ID_CONFLICT", "inventory release id does not match the candidate")

    _verify_objects(store, (*_normalize_objects(
        outputs, "POOL_SET_MISMATCH", "outputs are not a sequence"), *_normalize_objects(
        chooser, "CHOOSER_MISMATCH", "chooser is not a PoolOutput or a sequence")))

    try:
        prior = deployment.resolve_release(store, prior_release_id)
    except (deployment.DeploymentError, OSError, ValueError) as exc:
        raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "prior release could not be resolved") from exc
    except (TypeError, AttributeError, KeyError) as exc:
        raise ReleaseAssemblyRefusal("PRIOR_MISMATCH", "prior release could not be resolved") from exc
    if prior_catalog.get("deployment_id") != prior.deployment_id:
        raise ReleaseAssemblyRefusal(
            "PRIOR_MISMATCH", "prior catalog deployment id disagrees with its staged release")
    if getattr(inventory, "deployment_id", None) != prior.deployment_id:
        raise ReleaseAssemblyRefusal(
            "INVENTORY_MISMATCH", "inventory deployment id does not match the prior release")
    payloads = _read_prior_payloads(store, prior)
    _verify_carried_rows(store, prior_catalog)
    try:
        inventory_hashes = {member.content_hash for artifact in inventory.artifacts
                            for member in artifact.members}
    except (TypeError, AttributeError) as exc:
        raise ReleaseAssemblyRefusal("INVENTORY_MISMATCH", "inventory members are malformed") from exc
    if inventory_hashes != set(payloads):
        raise ReleaseAssemblyRefusal(
            "INVENTORY_MISMATCH", "inventory member hashes differ from the prior payloads")

    candidate_path = catalog_path(root, release_id)
    candidate = dataclasses.replace(prior, release_id=release_id)
    with _publication_lock(store):
        existing = _load_catalog(
            candidate_path, release_id, "ID_CONFLICT",
            "candidate catalog exists but cannot be read or validated")
        if existing is not None and existing != body:
            raise ReleaseAssemblyRefusal(
                "ID_CONFLICT", "candidate catalog already exists with different content")

        try:
            deployment.stage_release(store, candidate, inventory, payloads)
        except deployment.StagingRefused as exc:
            codes = {issue.code for issue in exc.issues}
            if codes & {"RELEASE_ID_REUSED", "MANIFEST_UNREADABLE"}:
                raise ReleaseAssemblyRefusal(
                    "ID_CONFLICT", "candidate release id is already staged with different content") from exc
            raise ReleaseAssemblyRefusal(
                "INVENTORY_MISMATCH",
                "inventory is incomplete or incompatible with the prior release") from exc
        except releases.ModelReleaseRefusal as exc:
            raise ReleaseAssemblyRefusal(
                "INVENTORY_MISMATCH",
                "inventory is incomplete or incompatible with the prior release") from exc
        except (TypeError, AttributeError, KeyError) as exc:
            raise ReleaseAssemblyRefusal("INVENTORY_MISMATCH", "inventory is malformed") from exc

        _probe_regular(staged_manifest_path(store, release_id), MAX_MANIFEST_BYTES,
                       "ID_CONFLICT", "candidate manifest is missing or unreadable")
        try:
            verify_staged(store, release_id)
        except ReleaseLayoutError as exc:
            raise ReleaseAssemblyRefusal(
                "ID_CONFLICT", "candidate staged release could not be verified") from exc

        if existing is not None:
            return candidate_path

        try:
            write_catalog(candidate_path, body)
        except OSError as exc:
            raise ReleaseAssemblyRefusal(
                "PUBLICATION_FAILED", "candidate catalog could not be written") from exc
    return candidate_path


__all__ = [
    "CHOOSER_MEMBER_ID", "LOCK_WAIT_S", "MAX_OBJECT_BYTES",
    "POOL_MEMBER_IDS", "PRIOR_DAMAGED", "PRIOR_NONE", "PRIOR_NO_POOL_CATALOG",
    "PRIOR_USABLE", "PoolOutput", "PriorClass", "ReleaseAssemblyRefusal",
    "carry_forward_release", "classify_prior", "derive_catalog", "prior_carry_set",
]
