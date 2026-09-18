"""Content-addressed release staging and atomic deployment pointer — P5-5.

A :class:`~.contracts.ModelRelease` (the shape :class:`~.loader.FrozenInference`
loads) is staged into a durable store — content-addressed member blobs plus an
immutable manifest keyed by ``release_id`` — only after it passes both
completeness checks this package already owns: :func:`~.releases.
require_complete_release` over the caller's :class:`~.releases.
ModelReleaseInventory` (internal consistency), and this module's own
cross-check that the *inference* release actually carries a member for every
role/strategy/clock binding the inventory declares, with the exact feature
order the inventory recorded (``_compatibility_issues``). A partial or
feature-order-incompatible release never finishes staging, so it can never be
promoted — there is no separate "promote-time" completeness gate to forget.

Promotion is one pointer swap: a ``DEPLOYED`` file naming the live
``release_id``. The write follows the pattern in
``tools/capture_tier0_corpus.py``'s ``_publish_current`` (temp file + rename)
made crash-proof per this task's contract — write the temp file, fsync it,
``os.replace`` it into place, fsync the containing directory — so a crash at
any point leaves either the old pointer or the new one, never a partial file.
Staging never opens or writes ``DEPLOYED``; only :func:`promote` and
:func:`rollback` do.

Every promotion and rollback also appends one immutable, sequence-numbered
file under ``history/`` — a record nothing ever rewrites — so a deployment's
past pointer states are auditable independent of what ``DEPLOYED`` shows now.

Replay does not read ``DEPLOYED`` at all. :func:`resolve_release` looks a
``release_id`` up directly in the staged-release store, so a score recorded
against release ``R`` keeps resolving ``R``'s exact, hash-verified members
after any number of later promotions or rollbacks — staged manifests are
never mutated or deleted by a pointer change.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

from engine.v2.foundation import (
    Clock,
    SystemClock,
    content_hash,
    format_timestamp,
    from_document,
    fsync_directory,
    to_document,
)

from .contracts import ArtifactMember, ModelRelease
from .releases import ModelReleaseInventory, ReleaseIssue, require_complete_release

__all__ = [
    "DEPLOYMENT_REFUSAL",
    "POINTER_STATE_V1",
    "STAGED_MANIFEST_V1",
    "DeploymentError",
    "NoPriorRelease",
    "PointerState",
    "ReleaseNotStaged",
    "StagedManifest",
    "StagingRefused",
    "current_pointer",
    "current_release",
    "pointer_history",
    "promote",
    "resolve_release",
    "rollback",
    "stage_release",
]

STAGED_MANIFEST_V1 = "staged_release_manifest.v1.0"
POINTER_STATE_V1 = "deployment_pointer_state.v1.0"
DEPLOYMENT_REFUSAL = "MODEL_DEPLOYMENT_REFUSED"


class DeploymentError(Exception):
    """Base for every refusal this module raises."""


class StagingRefused(DeploymentError):
    code = DEPLOYMENT_REFUSAL

    def __init__(self, issues: tuple[ReleaseIssue, ...]) -> None:
        self.issues = issues
        detail = "; ".join(f"{item.code} at {item.path}" for item in issues)
        super().__init__(f"{self.code}: {detail}")


class ReleaseNotStaged(DeploymentError):
    """A promote/rollback/replay named a ``release_id`` with no staged manifest."""


class NoPriorRelease(DeploymentError):
    """A rollback was requested with no earlier pointer state to return to."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class StagedManifest:
    """A staged release: content-addressed members, plus the release hash."""

    release: ModelRelease
    release_hash: str
    staged_at: str
    schema_version: str = STAGED_MANIFEST_V1


@dataclasses.dataclass(frozen=True, kw_only=True)
class PointerState:
    """One deployment pointer, live or historical."""

    sequence: int
    release_id: str
    previous_release_id: str | None
    action: str
    at: str
    schema_version: str = POINTER_STATE_V1


# --------------------------------------------------------------------------
# paths and atomic I/O
# --------------------------------------------------------------------------


def _release_dir(root: Path, release_id: str) -> Path:
    if not release_id or "/" in release_id or release_id in (".", ".."):
        raise DeploymentError(f"unsafe release id: {release_id!r}")
    return root / "releases" / release_id


def _manifest_path(root: Path, release_id: str) -> Path:
    return _release_dir(root, release_id) / "manifest.json"


def _object_relpath(member_hash: str) -> str:
    return "objects/" + member_hash.removeprefix("sha256:")


def _pointer_path(root: Path) -> Path:
    return root / "DEPLOYED"


def _history_dir(root: Path) -> Path:
    return root / "history"


def _encode(value: object) -> bytes:
    return json.dumps(to_document(value), indent=2, sort_keys=True).encode("utf-8")


def _decode(cls: type, data: bytes):
    return from_document(cls, json.loads(data.decode("utf-8")))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``temp -> fsync -> rename -> fsync(dir)``; a crash leaves ``path`` untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{os.urandom(4).hex()}")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fsync_directory(path.parent)


def _read_manifest(root: Path, release_id: str) -> StagedManifest | None:
    path = _manifest_path(root, release_id)
    if not path.is_file():
        return None
    return _decode(StagedManifest, path.read_bytes())


# --------------------------------------------------------------------------
# staging
# --------------------------------------------------------------------------


def _compatibility_issues(
    release: ModelRelease, inventory: ModelReleaseInventory,
) -> tuple[ReleaseIssue, ...]:
    """Cross-check the inference release against the completeness inventory.

    ``require_complete_release`` only ever sees ``inventory`` — it has no idea
    an inference-shaped :class:`ModelRelease` exists. This is the other half:
    every (role, strategy, clock) the inventory binds must have a matching
    inference binding, with the SAME feature order and at least one member
    named for each of the inventory's required member kinds.
    """
    issues: list[ReleaseIssue] = []
    if release.release_id != inventory.release_id:
        issues.append(ReleaseIssue(
            path="$.release_id", code="RELEASE_ID_MISMATCH",
            detail=f"{release.release_id} != {inventory.release_id}",
        ))
    by_key = {binding.key: binding for binding in inventory.bindings}
    seen = set()
    for binding in release.bindings:
        key = (binding.role, binding.strategy_id, binding.decision_clock_id)
        seen.add(key)
        path = f"$.bindings[{binding.binding_id}]"
        counterpart = by_key.get(key)
        if counterpart is None:
            issues.append(ReleaseIssue(path=path, code="UNBOUND_IN_INVENTORY", detail=repr(key)))
            continue
        if binding.feature_order != counterpart.ordered_features:
            issues.append(ReleaseIssue(
                path=f"{path}.feature_order", code="INCOMPATIBLE_FEATURE_ORDER", detail="order",
            ))
        member_names = {member.name for member in binding.members}
        for kind in counterpart.required_member_kinds:
            if kind not in member_names:
                issues.append(ReleaseIssue(
                    path=f"{path}.members", code=f"MISSING_{kind.upper()}_MEMBER",
                    detail=binding.binding_id,
                ))
    for key in sorted(set(by_key) - seen):
        issues.append(ReleaseIssue(path="$.bindings", code="MISSING_INFERENCE_BINDING", detail=repr(key)))
    return tuple(sorted(issues))


def _release_hash(release: ModelRelease) -> str:
    members = [
        {"binding_id": binding.binding_id, "member": member.name, "content_hash": member.content_hash}
        for binding in sorted(release.bindings, key=lambda b: b.binding_id)
        for member in sorted(binding.members, key=lambda m: m.name)
    ]
    return content_hash({"release_id": release.release_id, "members": members})


def _write_object(root: Path, member: ArtifactMember, payload: bytes) -> None:
    actual = "sha256:" + hashlib.sha256(payload).hexdigest()
    if actual != member.content_hash:
        raise StagingRefused((ReleaseIssue(
            path=f"$.members[{member.name}]", code="PAYLOAD_HASH_MISMATCH",
            detail=f"expected {member.content_hash} got {actual}",
        ),))
    dest = root / _object_relpath(member.content_hash)
    if not dest.is_file():
        _atomic_write_bytes(dest, payload)


def stage_release(
    root: Path,
    release: ModelRelease,
    inventory: ModelReleaseInventory,
    payloads: dict[str, bytes],
    *,
    clock: Clock = SystemClock(),
) -> StagedManifest:
    """Validate, then durably stage ``release``. Never touches ``DEPLOYED``.

    ``payloads`` maps every member's ``content_hash`` (across every binding)
    to its raw bytes. ``inventory`` must independently satisfy
    :func:`~.releases.require_complete_release`; a partial or
    feature-order-incompatible ``release`` refuses with :class:`StagingRefused`
    before anything is written. Staging the same ``release_id`` twice with the
    same content is a no-op; staging it twice with different content refuses.
    """
    require_complete_release(inventory)
    issues = _compatibility_issues(release, inventory)
    if issues:
        raise StagingRefused(issues)
    root = Path(root)
    staged_bindings = []
    for binding in release.bindings:
        staged_members = []
        for member in binding.members:
            payload = payloads.get(member.content_hash)
            if payload is None:
                raise StagingRefused((ReleaseIssue(
                    path=f"$.bindings[{binding.binding_id}].members[{member.name}]",
                    code="MISSING_MEMBER_PAYLOAD", detail=member.content_hash,
                ),))
            _write_object(root, member, payload)
            staged_members.append(dataclasses.replace(member, path=_object_relpath(member.content_hash)))
        staged_bindings.append(dataclasses.replace(binding, members=tuple(staged_members)))
    staged_release = dataclasses.replace(release, bindings=tuple(staged_bindings))
    manifest = StagedManifest(
        release=staged_release, release_hash=_release_hash(staged_release),
        staged_at=format_timestamp(clock.now()),
    )
    existing = _read_manifest(root, release.release_id)
    if existing is not None:
        if existing.release_hash != manifest.release_hash:
            raise StagingRefused((ReleaseIssue(
                path=f"$.releases[{release.release_id}]", code="RELEASE_ID_REUSED",
                detail="a different release is already staged under this id",
            ),))
        return existing  # identical content already staged: a no-op, not a re-timestamp
    _atomic_write_bytes(_manifest_path(root, release.release_id), _encode(manifest))
    return manifest


def resolve_release(root: Path, release_id: str) -> ModelRelease:
    """The exact staged ``ModelRelease`` for ``release_id`` — by id, not by pointer.

    Independent of ``DEPLOYED``: a later promotion or rollback never changes
    what this returns for an already-staged id, which is what makes a
    recorded score replayable after the deployment moves on.
    """
    manifest = _read_manifest(Path(root), release_id)
    if manifest is None:
        raise ReleaseNotStaged(release_id)
    return manifest.release


# --------------------------------------------------------------------------
# the deployment pointer
# --------------------------------------------------------------------------


def current_pointer(root: Path) -> PointerState | None:
    """The live pointer state, or ``None`` if nothing has ever been promoted."""
    path = _pointer_path(Path(root))
    if not path.is_file():
        return None
    return _decode(PointerState, path.read_bytes())


def pointer_history(root: Path) -> tuple[PointerState, ...]:
    """Every pointer state ever recorded, oldest first. Append-only: no entry is ever rewritten."""
    directory = _history_dir(Path(root))
    if not directory.is_dir():
        return ()
    entries = [_decode(PointerState, path.read_bytes()) for path in sorted(directory.glob("*.json"))]
    return tuple(sorted(entries, key=lambda item: item.sequence))


def _next_sequence(root: Path) -> int:
    directory = _history_dir(root)
    return len(list(directory.glob("*.json"))) if directory.is_dir() else 0


def _append_history(root: Path, state: PointerState) -> None:
    path = _history_dir(root) / f"{state.sequence:06d}.json"
    if path.exists():
        raise DeploymentError(f"history entry {state.sequence} already recorded")
    _atomic_write_bytes(path, _encode(state))


def _swap_pointer(root: Path, release_id: str, action: str, clock: Clock) -> PointerState:
    if _read_manifest(root, release_id) is None:
        raise ReleaseNotStaged(release_id)
    previous = current_pointer(root)
    state = PointerState(
        sequence=_next_sequence(root),
        release_id=release_id,
        previous_release_id=None if previous is None else previous.release_id,
        action=action,
        at=format_timestamp(clock.now()),
    )
    # The critical atomic step: a fault here leaves DEPLOYED exactly as it
    # was (old release, or absent). History is only appended after it lands.
    _atomic_write_bytes(_pointer_path(root), _encode(state))
    _append_history(root, state)
    return state


def promote(root: Path, release_id: str, *, clock: Clock = SystemClock()) -> PointerState:
    """Atomically point ``DEPLOYED`` at ``release_id``. Refuses an unstaged release."""
    return _swap_pointer(Path(root), release_id, "promote", clock)


def rollback(root: Path, *, clock: Clock = SystemClock()) -> PointerState:
    """Point ``DEPLOYED`` back at the release the current one was promoted from."""
    root = Path(root)
    current = current_pointer(root)
    if current is None or current.previous_release_id is None:
        raise NoPriorRelease("no prior release to roll back to")
    return _swap_pointer(root, current.previous_release_id, "rollback", clock)


def current_release(root: Path) -> ModelRelease | None:
    """The ``ModelRelease`` the live pointer resolves to, or ``None`` if unset."""
    root = Path(root)
    pointer = current_pointer(root)
    if pointer is None:
        return None
    return resolve_release(root, pointer.release_id)
