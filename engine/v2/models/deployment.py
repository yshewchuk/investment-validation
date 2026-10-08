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

A staged manifest is not by itself promotable. :func:`mark_staging_succeeded`
— called by the staging workflow only once every post-stage check has passed —
publishes the ``releases/<release_id>/staging-status.json`` record that binds
success to one exact manifest identity, and :func:`promote`/:func:`rollback`
refuse :class:`StagingNotSuccessful` before any pointer or history write when
that record is missing, unreadable, not a success, or bound to a different
``release_id`` or ``release_hash`` than the manifest they are about to deploy.

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
    "MODEL_RELEASE_ROOT_ENV",
    "POINTER_STATE_V1",
    "STAGED_MANIFEST_V1",
    "ConcurrentPromote",
    "CorruptManifest",
    "DeploymentError",
    "MissingReleaseRoot",
    "NoPriorRelease",
    "PointerState",
    "ReleaseNotStaged",
    "StagedManifest",
    "StagingNotSuccessful",
    "StagingRefused",
    "StaleReleaseHash",
    "current_pointer",
    "current_release",
    "invalidate_staging_success",
    "mark_staging_succeeded",
    "pointer_history",
    "production_deployment_root",
    "production_release_root",
    "promote",
    "resolve_release",
    "restage_semantic_hash",
    "rollback",
    "stage_release",
]

STAGED_MANIFEST_V1 = "staged_release_manifest.v1.0"
POINTER_STATE_V1 = "deployment_pointer_state.v1.0"
DEPLOYMENT_REFUSAL = "MODEL_DEPLOYMENT_REFUSED"
RELEASE_HASH_MEMBER_V1 = "member_only.v1"
RELEASE_HASH_SEMANTIC_V2 = "semantic_manifest.v2"
MODEL_RELEASE_ROOT_ENV = "MODEL_RELEASE_ROOT"
STAGING_STATUS_FILENAME = "staging-status.json"
STAGING_STATE_SUCCEEDED = "succeeded"


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


class MissingReleaseRoot(DeploymentError):
    """No production release root is configured -- see production_release_root()."""

    code = "MISSING_RELEASE_ROOT"

    def __init__(self) -> None:
        """Build the MISSING_RELEASE_ROOT refusal message."""
        super().__init__(f"{self.code}: set {MODEL_RELEASE_ROOT_ENV} to the "
                          f"production release root")


class StaleReleaseHash(DeploymentError):
    """A promote/rollback target's staged manifest predates the current
    release_hash_version. restage_semantic_hash() rewrites it in place, from
    already-staged content -- no retraining, no payload re-read -- but this
    is never applied automatically; a promote/rollback call only refuses."""

    code = "STALE_RELEASE_HASH"

    def __init__(self, release_id: str, hash_version: str) -> None:
        """Build the STALE_RELEASE_HASH refusal message, naming the release
        and its stale hash version."""
        self.release_id = release_id
        self.hash_version = hash_version
        super().__init__(f"{self.code}: {release_id} is staged under hash version "
                          f"{hash_version}, not {RELEASE_HASH_SEMANTIC_V2}")


class CorruptManifest(DeploymentError):
    """A staged manifest's declared release_hash no longer matches its own
    recomputed content hash -- the manifest was tampered or corrupted on
    disk after staging. Refused before promote/rollback ever move the
    pointer."""

    code = "CORRUPT_MANIFEST"

    def __init__(self, release_id: str) -> None:
        """Build the CORRUPT_MANIFEST refusal message, naming the release."""
        self.release_id = release_id
        super().__init__(f"{self.code}: {release_id}'s staged manifest content "
                          f"hash does not match its declared release_hash")


class StagingNotSuccessful(DeploymentError):
    """A promote/rollback target has no successful staging completion record.

    Typed, non-retryable refusal: a release that never finished staging
    cannot be promoted, and re-asking without re-staging changes nothing.
    """

    code = "STAGING_NOT_SUCCESSFUL"
    retryable = False

    def __init__(self, release_id: str) -> None:
        """Build the STAGING_NOT_SUCCESSFUL refusal message, naming the release."""
        self.release_id = release_id
        super().__init__(f"{self.code}: {release_id} has no successful staging "
                          f"completion record")


class ConcurrentPromote(DeploymentError):
    """A promote's optional expected-incumbent guard refused.

    Raised only when ``promote`` is given ``expected_previous_release_id``
    and the live pointer read for that swap is absent or names a different
    release -- another promotion already moved ``DEPLOYED`` (or nothing was
    ever promoted). Typed and non-retryable until the operator re-reads the
    pointer and re-issues with the observed incumbent; the refusal itself
    never retries, never picks another target, and never writes the pointer
    or ``history/``. The same-target no-op still wins over this guard.
    """

    code = "CONCURRENT_PROMOTE"

    def __init__(self, release_id: str, expected_previous_release_id: str) -> None:
        """Build the CONCURRENT_PROMOTE refusal, naming both ids."""
        self.release_id = release_id
        self.expected_previous_release_id = expected_previous_release_id
        super().__init__(f"{self.code}: {release_id} expected incumbent "
                          f"{expected_previous_release_id!r} deployed, but the "
                          f"deployment pointer is absent or names another release")


# --------------------------------------------------------------------------
# production configuration
# --------------------------------------------------------------------------


def production_release_root() -> Path:
    """The one configured production STORE root -- read fresh from
    ``MODEL_RELEASE_ROOT`` on every call, never cached. This is the store
    root, NOT this module's own ``root`` parameter:
    ``release_bindings.resolve_release_binding`` and
    ``checks/phase5_release.py`` both navigate from a value at this level
    by appending their own ``deployment/`` subdirectory
    (``<release_root>/deployment/DEPLOYED``); this module's own
    :func:`promote`/:func:`rollback`/:func:`resolve_release`/
    :func:`current_release`/:func:`stage_release`/
    :func:`restage_semantic_hash` all take THAT ``deployment/`` directory
    itself as their ``root`` -- see :func:`production_deployment_root`,
    which is what a caller of any of those wants, not this function
    directly. Raises :class:`MissingReleaseRoot` when the variable is
    unset or blank -- there is no fallback default, because a silent
    default here would let an operator promote or resolve against the
    wrong store without any signal. Always returns an absolute,
    ``~``-expanded path (``Path(...).expanduser().resolve()``).
    """
    value = os.environ.get(MODEL_RELEASE_ROOT_ENV, "").strip()
    if not value:
        raise MissingReleaseRoot()
    return Path(value).expanduser().resolve()


def production_deployment_root() -> Path:
    """``production_release_root() / "deployment"`` -- the directory this
    module's own root-taking functions (:func:`promote`, :func:`rollback`,
    :func:`resolve_release`, :func:`current_release`, :func:`stage_release`,
    :func:`restage_semantic_hash`) expect as their ``root`` argument when
    operating against the ONE configured production store, matching the
    same ``<release_root>/deployment/`` layout
    ``release_bindings.resolve_release_binding`` and
    ``checks/phase5_release.py`` already use for the identical value.
    ``training.promote_plan`` resolves an omitted ``--release-root``
    through THIS function, never through :func:`production_release_root`
    directly -- the two disagreed on which directory ``MODEL_RELEASE_ROOT``
    named until this function existed (2026-09-27 Opus gate finding:
    scoring's resolution and promote's resolution pointed at directories
    one level apart for the same configured value). Raises
    :class:`MissingReleaseRoot` exactly as :func:`production_release_root`
    does, for the same reason.
    """
    return production_release_root() / "deployment"


@dataclasses.dataclass(frozen=True, kw_only=True)
class StagedManifest:
    """A staged release: content-addressed members, plus the release hash."""

    release: ModelRelease
    release_hash: str
    staged_at: str
    schema_version: str = STAGED_MANIFEST_V1
    # Missing in historical manifests; from_document supplies this legacy default.
    release_hash_version: str = RELEASE_HASH_MEMBER_V1


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


def _staging_status_path(root: Path, release_id: str) -> Path:
    return _release_dir(root, release_id) / STAGING_STATUS_FILENAME


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


def _duplicate_binding_issues(release: ModelRelease) -> tuple[ReleaseIssue, ...]:
    """Every ``(role, strategy_id)`` pair bound at most once.

    Deliberately NOT clock-qualified: ``scoring.release_bindings.
    _resolve_model_bindings`` resolves its runtime catalog by
    ``f"{role}:{strategy_id}"`` alone, and nothing anywhere filters
    candidate bindings by ``decision_clock_id`` before that lookup runs. A
    release with two bindings sharing a ``(role, strategy_id)`` but
    different ``decision_clock_id`` values would stage/promote cleanly
    under a clock-qualified check, then make every score for that role/
    strategy fail with ``ModelNotReady("ambiguous binding")`` at read time
    -- so this check uses the SAME coarser key scoring does. Self-contained
    on ``release`` alone -- no ``inventory`` needed -- so both
    ``_compatibility_issues`` (staging, cross-checked against the inventory
    too) and ``_swap_pointer`` (promote/rollback, re-verified independent of
    whatever staged the manifest) share this ONE definition of "no
    ambiguous inference binding".
    """
    issues: list[ReleaseIssue] = []
    seen = set()
    for binding in release.bindings:
        key = (binding.role, binding.strategy_id)
        path = f"$.bindings[{binding.binding_id}]"
        if key in seen:
            issues.append(ReleaseIssue(path=path, code="DUPLICATE_BINDING", detail=repr(key)))
        else:
            seen.add(key)
    return tuple(issues)


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
    issues.extend(_duplicate_binding_issues(release))
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
    """Hash the complete inference meaning of a release, independent of storage paths.

    Member paths are rewritten when staged, so they describe storage layout
    rather than inference semantics. Every other release, binding and member
    field is part of the identity: the same bytes wired to a different adapter,
    feature order, role, output or clock are different releases.
    """
    bindings = []
    for binding in sorted(release.bindings, key=lambda item: item.binding_id):
        bindings.append({
            "binding_id": binding.binding_id,
            "model_id": binding.model_id,
            "role": binding.role,
            "strategy_id": binding.strategy_id,
            "decision_clock_id": binding.decision_clock_id,
            "adapter": binding.adapter,
            "feature_order": binding.feature_order,
            "output_names": binding.output_names,
            "schema_version": binding.schema_version,
            "members": [
                {
                    "name": member.name,
                    "content_hash": member.content_hash,
                    "schema_version": member.schema_version,
                }
                for member in sorted(binding.members, key=lambda item: item.name)
            ],
        })
    return content_hash({
        "release_id": release.release_id,
        "deployment_id": release.deployment_id,
        "schema_version": release.schema_version,
        "bindings": bindings,
    })


def _legacy_release_hash(release: ModelRelease) -> str:
    """The member-only hash used by manifests staged before semantic v2."""
    members = [
        {"binding_id": binding.binding_id, "member": member.name,
         "content_hash": member.content_hash}
        for binding in sorted(release.bindings, key=lambda item: item.binding_id)
        for member in sorted(binding.members, key=lambda item: item.name)
    ]
    return content_hash({"release_id": release.release_id, "members": members})


def _manifest_hash_matches(manifest: StagedManifest) -> bool:
    if manifest.release_hash_version == RELEASE_HASH_MEMBER_V1:
        return _legacy_release_hash(manifest.release) == manifest.release_hash
    if manifest.release_hash_version == RELEASE_HASH_SEMANTIC_V2:
        return _release_hash(manifest.release) == manifest.release_hash
    return False


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
    Refuses :class:`StagingRefused` (``MANIFEST_UNREADABLE``) if an existing
    manifest for this ``release_id`` cannot be read or parsed.
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
        release_hash_version=RELEASE_HASH_SEMANTIC_V2,
    )
    try:
        existing = _read_manifest(root, release.release_id)
    except (OSError, ValueError) as exc:
        raise StagingRefused((ReleaseIssue(
            path=f"$.releases[{release.release_id}]", code="MANIFEST_UNREADABLE",
            detail="the existing staged manifest could not be read",
        ),)) from exc
    if existing is not None:
        if not _manifest_hash_matches(existing):
            raise StagingRefused((ReleaseIssue(
                path=f"$.releases[{release.release_id}]", code="RELEASE_ID_REUSED",
                detail="the existing staged manifest has an invalid release hash",
            ),))
        # Compare semantic content even for a verified legacy manifest. This
        # keeps old ids idempotent without allowing their weaker byte-only hash
        # to mask a changed inference binding.
        if _release_hash(existing.release) != manifest.release_hash:
            raise StagingRefused((ReleaseIssue(
                path=f"$.releases[{release.release_id}]", code="RELEASE_ID_REUSED",
                detail="a different release is already staged under this id",
            ),))
        return existing  # identical content already staged: a no-op, not a re-timestamp
    _atomic_write_bytes(_manifest_path(root, release.release_id), _encode(manifest))
    return manifest


def restage_semantic_hash(root: Path, release_id: str) -> StagedManifest:
    """Rewrite ``release_id``'s staged manifest to the current semantic hash
    version, in place, from its already-staged :class:`~.contracts.ModelRelease`
    -- no payload re-read, no re-staging of objects, no touch to ``DEPLOYED``
    or ``history/``.

    An explicit, operator-run upgrade step, never automatic --
    :func:`stage_release`'s own idempotent re-stage of identical content
    deliberately KEEPS a legacy manifest's original bytes and version (see
    ``tests/test_checks_phase5_acceptance.py::
    test_model_release_loader_and_restage_support_versioned_hashes``).
    :func:`promote`/:func:`rollback` refuse any ``release_hash_version``
    other than ``RELEASE_HASH_SEMANTIC_V2`` (:class:`StaleReleaseHash`);
    this is how an already-staged legacy release becomes eligible again,
    without retraining or re-uploading a single byte.

    Refuses :class:`ReleaseNotStaged` if nothing is staged under
    ``release_id``. Refuses :class:`StagingRefused` (``RELEASE_ID_REUSED``,
    the same code :func:`stage_release` uses for the same condition) if the
    EXISTING manifest does not verify under its OWN declared
    ``release_hash_version`` first -- a corrupt or tampered manifest is
    never a starting point for a rewrite. A manifest already at
    ``RELEASE_HASH_SEMANTIC_V2`` is returned unchanged (idempotent no-op).
    """
    root = Path(root)
    try:
        existing = _read_manifest(root, release_id)
    except (OSError, ValueError) as exc:
        # _read_manifest's decode can raise a bare json.JSONDecodeError or
        # engine.v2.foundation.typed.DocumentError (both ValueError
        # subclasses), or a bare OSError racing the is_file() check --
        # exactly the same failure family engine/v2/scoring/
        # release_bindings.py's _read_and_verify_manifest already catches
        # for the identical read. Never left to escape uncaught.
        raise StagingRefused((ReleaseIssue(
            path=f"$.releases[{release_id}]", code="MANIFEST_UNREADABLE",
            detail="the existing staged manifest could not be read",
        ),)) from exc
    if existing is None:
        raise ReleaseNotStaged(release_id)
    if existing.release.release_id != release_id:
        raise StagingRefused((ReleaseIssue(
            path=f"$.releases[{release_id}]", code="RELEASE_ID_MISMATCH",
            detail=f"staged manifest at this path declares release_id "
                    f"{existing.release.release_id!r}, not {release_id!r}",
        ),))
    if not _manifest_hash_matches(existing):
        raise StagingRefused((ReleaseIssue(
            path=f"$.releases[{release_id}]", code="RELEASE_ID_REUSED",
            detail="the existing staged manifest has an invalid release hash",
        ),))
    if existing.release_hash_version == RELEASE_HASH_SEMANTIC_V2:
        return existing
    rewritten = dataclasses.replace(
        existing, release_hash=_release_hash(existing.release),
        release_hash_version=RELEASE_HASH_SEMANTIC_V2,
    )
    _atomic_write_bytes(_manifest_path(root, release_id), _encode(rewritten))
    return rewritten


def mark_staging_succeeded(root: Path, release_id: str) -> None:
    """Publish ``release_id``'s durable staging-completion record.

    The manifest ``stage_release`` writes only says a release landed in the
    store; it is this sidecar — the success state bound to that manifest's
    actual identity — that makes the release promotable at all. Call it from
    the staging workflow AFTER every post-stage check has passed (Phase 5
    verification and anything else gating the release), never from
    :func:`stage_release` itself, whose manifest necessarily precedes them.

    Writes ``releases/<release_id>/staging-status.json`` atomically — the same
    temp/fsync/rename/fsync-directory path as every other durable artifact
    here — containing exactly ``release_id``, that staged manifest's
    ``release_hash`` and ``state`` set to ``"succeeded"``. Re-marking an
    unchanged release rewrites an identical record; marking one whose manifest
    was since restaged rebinds to the new hash. Reads only: a manifest or any
    staged object is never rewritten, so re-running it is safe.

    Refuses :class:`ReleaseNotStaged` when nothing is staged under
    ``release_id``, :class:`StagingRefused` (``MANIFEST_UNREADABLE``) when the
    staged manifest cannot be read or parsed, :class:`StagingRefused`
    (``RELEASE_ID_MISMATCH``) when that manifest declares a different release
    than the path it sits on, :class:`CorruptManifest` when its declared
    hash no longer matches its own content, and :class:`StagingRefused`
    (``STATUS_UNWRITABLE``) when the completion record itself cannot be
    written. A success record may only bind a
    manifest this module would itself deploy -- never a hash invented from
    whatever bytes happen to be on disk.
    """
    root = Path(root)
    try:
        manifest = _read_manifest(root, release_id)
    except (OSError, ValueError) as exc:
        raise StagingRefused((ReleaseIssue(
            path=f"$.releases[{release_id}]", code="MANIFEST_UNREADABLE",
            detail="the staged manifest could not be read",
        ),)) from exc
    if manifest is None:
        raise ReleaseNotStaged(release_id)
    if manifest.release.release_id != release_id:
        raise StagingRefused((ReleaseIssue(
            path=f"$.releases[{release_id}]", code="RELEASE_ID_MISMATCH",
            detail=f"staged manifest at this path declares release_id "
                   f"{manifest.release.release_id!r}, not {release_id!r}",
        ),))
    if not _manifest_hash_matches(manifest):
        raise CorruptManifest(release_id)
    status = {
        "release_id": release_id,
        "release_hash": manifest.release_hash,
        "state": STAGING_STATE_SUCCEEDED,
    }
    try:
        _atomic_write_bytes(_staging_status_path(root, release_id), _encode(status))
    except OSError as exc:
        raise StagingRefused((ReleaseIssue(
            path=f"$.releases[{release_id}].staging-status", code="STATUS_UNWRITABLE",
            detail="the staging completion record could not be written",
        ),)) from exc


def invalidate_staging_success(root: Path, release_id: str) -> None:
    """Withdraw ``release_id``'s staging-completion record, if one exists.

    The inverse of :func:`mark_staging_succeeded`: removes
    ``releases/<release_id>/staging-status.json`` so a later
    :func:`promote`/:func:`rollback` refuses :class:`StagingNotSuccessful`
    again. A missing sidecar is an idempotent no-op. The staged manifest,
    every content-addressed object, ``DEPLOYED`` and ``history/`` are never
    touched.

    Raises:
        StagingRefused: STATUS_INVALIDATION_FAILED -- an ``OSError`` from the
            unlink; the previous staging success record could not be removed.
    """
    root = Path(root)
    path = _staging_status_path(root, release_id)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        raise StagingRefused((ReleaseIssue(
            path=f"$.releases[{release_id}].staging-status",
            code="STATUS_INVALIDATION_FAILED",
            detail="the previous staging success record could not be invalidated",
        ),)) from exc


def resolve_release(root: Path, release_id: str) -> ModelRelease:
    """The exact staged ``ModelRelease`` for ``release_id`` — by id, not by pointer.

    Independent of ``DEPLOYED``: a later promotion or rollback never changes
    what this returns for an already-staged id, which is what makes a
    recorded score replayable after the deployment moves on. Refuses
    :class:`StagingRefused` (``MANIFEST_UNREADABLE``) if the staged manifest
    cannot be read or parsed.
    """
    try:
        manifest = _read_manifest(Path(root), release_id)
    except (OSError, ValueError) as exc:
        raise StagingRefused((ReleaseIssue(
            path=f"$.releases[{release_id}]", code="MANIFEST_UNREADABLE",
            detail="the staged manifest could not be read",
        ),)) from exc
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


def _repair_history(root: Path) -> None:
    """Record the live pointer's history entry if a crash skipped it."""
    pointer = current_pointer(root)
    if pointer is None:
        return
    path = _history_dir(root) / f"{pointer.sequence:06d}.json"
    if not path.exists():
        _append_history(root, pointer)


def _status_field(document: object, key: str) -> str:
    """One text field of a staging-status document.

    ``""`` for an absent key, a non-string value, or a document that is not an
    object at all -- every one of which simply fails to equal what a real
    success record must carry, so no separate malformed-shape branch is needed.
    """
    if not isinstance(document, dict):
        return ""
    value = document.get(key)
    return value if isinstance(value, str) else ""


def _require_staging_success(
    root: Path, release_id: str, manifest: StagedManifest,
) -> None:
    """Refuse a staged release whose staging never reported success.

    The staged manifest alone does not make a release promotable: staging is
    the durable store AND its completion record, and
    :func:`mark_staging_succeeded` is what publishes that record -- bound to
    this exact ``release_id`` and ``release_hash``. Anything else refuses
    :class:`StagingNotSuccessful`: no sidecar at all (a staging job that
    crashed, was abandoned, or failed a post-stage check), one that cannot be
    read or parsed, one whose ``state`` is not ``"succeeded"``, one written for
    a different release id, or one left behind by an earlier generation of this
    id whose manifest has since been restaged under a different hash. A stale
    record is never trusted across a hash change.

    Raises:
        StagingNotSuccessful: No success record bound to this exact
            ``release_id`` and ``release_hash`` could be read.
    """
    path = _staging_status_path(root, release_id)
    try:
        document = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise StagingNotSuccessful(release_id) from exc
    if (_status_field(document, "release_id") != release_id
            or _status_field(document, "release_hash") != manifest.release_hash
            or _status_field(document, "state") != STAGING_STATE_SUCCEEDED):
        raise StagingNotSuccessful(release_id)


def _validate_staged_target(root: Path, release_id: str) -> StagedManifest:
    try:
        manifest = _read_manifest(root, release_id)
    except (OSError, ValueError) as exc:
        raise StagingRefused((ReleaseIssue(
            path=f"$.releases[{release_id}]", code="MANIFEST_UNREADABLE",
            detail="the target staged manifest could not be read",
        ),)) from exc
    if manifest is None:
        raise ReleaseNotStaged(release_id)
    if manifest.release_hash_version != RELEASE_HASH_SEMANTIC_V2:
        raise StaleReleaseHash(release_id, manifest.release_hash_version)
    if not _manifest_hash_matches(manifest):
        raise CorruptManifest(release_id)
    duplicate_issues = _duplicate_binding_issues(manifest.release)
    if duplicate_issues:
        raise StagingRefused(duplicate_issues)
    # Last gate before anything is written: a refusal here touches no pointer,
    # no history entry and no temp file, and leaves the staged store intact.
    _require_staging_success(root, release_id, manifest)
    return manifest


def _swap_pointer(
    root: Path, release_id: str, action: str, clock: Clock,
    *, validated_manifest: StagedManifest | None = None,
    expected_previous_release_id: str | None = None,
) -> PointerState:
    """Validate a staged release, then move ``DEPLOYED`` to it.

    Shared by both :func:`promote` and :func:`rollback`. Refuses
    :class:`ReleaseNotStaged`, :class:`StaleReleaseHash`,
    :class:`CorruptManifest`, :class:`StagingRefused` (``MANIFEST_UNREADABLE``
    when the target manifest cannot be read or parsed, duplicate bindings) and
    :class:`StagingNotSuccessful` (no staging-completion record bound to this
    exact ``release_id`` and ``release_hash``) before the pointer ever moves;
    a no-op if ``release_id`` is already live.

    ``expected_previous_release_id`` is promote's optional expected-incumbent
    guard (``None`` from rollback, which never takes it). When supplied, the
    live pointer is read for this swap BEFORE ``_repair_history`` or any
    write: an already-live target keeps the existing no-op (the guard loses
    to idempotency even when its expected id has since gone stale), and
    otherwise an absent pointer or one naming another release refuses
    :class:`ConcurrentPromote` -- no retry, no alternate target, and no
    pointer/history change, so the refusal cannot heal a crash window.

    ``validated_manifest`` lets a caller that already ran
    :func:`_validate_staged_target` on this exact target (rollback validates
    before repairing history) supply the verified result and skip only the
    re-validation; the repair/no-op/write order is unchanged either way.
    """
    if validated_manifest is None:
        _validate_staged_target(root, release_id)
    observed = current_pointer(root)
    already_live = observed is not None and observed.release_id == release_id
    if (expected_previous_release_id is not None and not already_live
            and (observed is None
                 or observed.release_id != expected_previous_release_id)):
        raise ConcurrentPromote(release_id, expected_previous_release_id)
    _repair_history(root)
    previous = current_pointer(root)
    if previous is not None and previous.release_id == release_id:
        # Already deployed: a repeated promote is a no-op, never its own predecessor.
        return previous
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


def promote(
    root: Path, release_id: str, *, clock: Clock = SystemClock(),
    expected_previous_release_id: str | None = None,
) -> PointerState:
    """Atomically point ``DEPLOYED`` at ``release_id``.

    Refuses an unstaged release (:class:`ReleaseNotStaged`) or one staged
    under a superseded ``release_hash_version`` (:class:`StaleReleaseHash`)
    -- see :func:`restage_semantic_hash` -- and one whose staging workflow
    never published a success record for this exact release and hash
    (:class:`StagingNotSuccessful`), see :func:`mark_staging_succeeded`.

    ``expected_previous_release_id`` is the optional expected-incumbent
    guard for callers promoting from a known incumbent: when supplied, the
    pointer read for this swap must name that release, or the swap refuses
    :class:`ConcurrentPromote` before any pointer/history write (an omitted
    guard keeps the existing behavior exactly; a request for the already
    live ``release_id`` stays the existing no-op even with a stale expected
    id). This is an optimistic check on one read, not a cross-process lock.
    """
    return _swap_pointer(Path(root), release_id, "promote", clock,
                         expected_previous_release_id=expected_previous_release_id)


def _rollback_target(root: Path) -> str:
    """The release id a rollback should land on.

    Replays ``pointer_history`` as an undo stack: push the release id on
    every ``"promote"`` action (including a promote back to an older
    release id), pop on every ``"rollback"`` action. The target is the
    second-from-top id after the replay -- the release that was live
    immediately before the most recent forward move -- so N chained
    ``rollback()`` calls undo N chained promotions and never revisit a
    release a prior rollback already left. Resolves exactly what a rollback
    after :func:`_repair_history` would resolve, without writing: when the
    live pointer's own sequence file is absent (the same condition
    :func:`_repair_history` repairs on -- a crash between the pointer write
    and the history append left DEPLOYED one ahead of history), that
    pointer is appended to the in-memory history tuple before the checks
    below run, so the crash window is replayed, not refused around.
    Refuses :class:`NoPriorRelease`
    when fewer than two ids remain on the replayed stack,
    :class:`StagingRefused` (``HISTORY_UNREADABLE``) when a recorded
    history entry can't be read, :class:`StagingRefused`
    (``HISTORY_SEQUENCE_GAP``) when the read sequences aren't exactly the
    contiguous range ``0..len(history) - 1`` (a hole in the middle is
    corruption :func:`_repair_history` can never restore, and replaying a
    gapped history as consecutive undo steps would target the wrong
    release), and :class:`StagingRefused` (``HISTORY_INCONSISTENT``) when
    the top of the replayed stack disagrees with the release ``DEPLOYED``
    actually names.
    """
    try:
        history = pointer_history(root)
    except (OSError, ValueError) as exc:
        raise StagingRefused((ReleaseIssue(
            path="$.history", code="HISTORY_UNREADABLE",
            detail="a recorded pointer-history entry could not be read",
        ),)) from exc
    pointer = current_pointer(root)
    if pointer is not None and not (
            _history_dir(root) / f"{pointer.sequence:06d}.json").exists():
        history = history + (pointer,)
    sequences = [state.sequence for state in history]
    if sequences != list(range(len(history))):
        raise StagingRefused((ReleaseIssue(
            path="$.history", code="HISTORY_SEQUENCE_GAP",
            detail=f"expected contiguous sequences 0..{len(history) - 1}, got {sequences}",
        ),))
    stack: list[str] = []
    for state in history:
        if state.action == "rollback":
            if stack:
                stack.pop()
        else:
            stack.append(state.release_id)
    if len(stack) < 2:
        raise NoPriorRelease("no prior release to roll back to")
    live_release_id = None if pointer is None else pointer.release_id
    if stack[-1] != live_release_id:
        raise StagingRefused((ReleaseIssue(
            path="$.history", code="HISTORY_INCONSISTENT",
            detail=f"replayed history believes {stack[-1]!r} is live, "
                   f"DEPLOYED says {live_release_id!r}",
        ),))
    return stack[-2]


def rollback(root: Path, *, clock: Clock = SystemClock()) -> PointerState:
    """Point ``DEPLOYED`` back at the release the current one was promoted
    from. Refuses :class:`NoPriorRelease` with nothing to roll back to, or
    :class:`StaleReleaseHash`/:class:`CorruptManifest` if THAT prior release
    is itself staged under a superseded hash version or a tampered manifest,
    or :class:`StagingNotSuccessful` if THAT prior release's staging never
    published a success record for its current manifest.
    Also refuses :class:`StagingRefused` when a recorded pointer-history
    entry can't be read (``HISTORY_UNREADABLE``), the replayed sequences
    aren't exactly contiguous (``HISTORY_SEQUENCE_GAP``), or the top of the
    replayed history disagrees with the release ``DEPLOYED`` actually names
    (``HISTORY_INCONSISTENT``).
    Target resolution and target validation both precede the crash-history
    repair, so a refusal above leaves ``DEPLOYED`` and ``history/`` exactly
    as they are -- even when a crash had left ``DEPLOYED`` one entry ahead
    of history -- while a successful rollback still repairs that crash
    window before appending its own entry.
    """
    root = Path(root)
    target = _rollback_target(root)
    validated_manifest = _validate_staged_target(root, target)
    _repair_history(root)
    return _swap_pointer(
        root, target, "rollback", clock, validated_manifest=validated_manifest,
    )


def current_release(root: Path) -> ModelRelease | None:
    """The ``ModelRelease`` the live pointer resolves to, or ``None`` if unset."""
    root = Path(root)
    pointer = current_pointer(root)
    if pointer is None:
        return None
    return resolve_release(root, pointer.release_id)
