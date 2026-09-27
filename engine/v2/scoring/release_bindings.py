"""Production reader of one live release's scoring catalog.

Resolves model identity, model artifact refs, the verified ``ModelRelease``, a
constructed ``FrozenInference``, and the analog/payoff/recalibration artifacts
from a live ``deployment.current_pointer()`` -- the failure-semantics contract
is ``engine/v2/scoring/ARCHITECTURE.md``'s ``release_bindings.py`` section (the
"4c R1-R6 template"). Nothing calls this yet; a later PR wires it into the
per-night ``SourceBundle`` assembler.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

from engine.v2.foundation import content_hash
from engine.v2.models import deployment
from engine.v2.models.analog_artifact import BoardAnalogPoolArtifact
from engine.v2.models.contracts import MODEL_NOT_READY, ModelRelease
from engine.v2.models.frozen_state import FrozenStateError, FrozenStateLoader, FrozenStateRef
from engine.v2.models.loader import FrozenInference
from engine.v2.models.payoff_artifact import (
    PayoffArtifactError,
    PayoffArtifactLoader,
    PayoffArtifactRef,
    PayoffLineArtifact,
    PayoffSurfaceArtifact,
)
from engine.v2.models.recalibration_artifact import (
    RecalibrationArtifactError,
    RecalibrationArtifactLoader,
    RecalibrationArtifactRef,
    RecalibrationMapArtifact,
)

__all__ = [
    "ModelIdentity",
    "ModelNotReady",
    "NoCurrentRelease",
    "ScoringReleaseBinding",
    "ReleaseBindingError",
    "resolve_release_binding",
]

_DEPLOYMENT_DIR = "deployment"
_STATE_CATALOG_NAME = "phase5_release.json"
_PHASE5_RELEASE_SCHEMA = "phase5_staged_release.v1.0"
_PAYOFF_PREFIXES = ("payoff_line:", "payoff_surface:")
_RECALIBRATION_PREFIX = "recalibration_map:"
_ANALOG_MEMBER_ID = "board_analog_matcher"


class ReleaseBindingError(ValueError):
    """Base for every refusal this module raises."""


class NoCurrentRelease(ReleaseBindingError):
    """R1(a): nothing has ever been promoted at this release root."""


class ModelNotReady(ReleaseBindingError):
    """One named member -- the DEPLOYED pointer, the staged manifest, a model
    binding member, or a payoff/recalibration/analog state object -- is
    missing, wrong status, hash-mismatched, or rejected by its own typed
    loader. Never a fallback: this is always the only outcome for a bad
    member (R1)."""

    code = MODEL_NOT_READY

    def __init__(self, member_id: str, detail: str) -> None:
        self.member_id = member_id
        self.detail = detail
        super().__init__(f"{self.code}: {member_id}: {detail}")


@dataclass(frozen=True, kw_only=True)
class ModelIdentity:
    """One resolved, hash-verified model binding's identity."""

    binding_id: str
    model_id: str
    role: str
    strategy_id: str
    decision_clock_id: str
    adapter: str
    feature_order: tuple[str, ...]
    output_names: tuple[str, ...]
    artifact_hash: str


@dataclass(frozen=True, kw_only=True)
class ScoringReleaseBinding:
    """One live release's resolved, hash-verified scoring catalog.

    Every field is a genuinely immutable value (frozen dataclass,
    ``MappingProxyType`` mappings) EXCEPT ``frozen_inference``: that field's
    referenced ``FrozenInference`` instance owns its own mutable, growing
    member cache that outlives this call (see ``loader.py``), by design --
    it is meant to be reused across every ``.infer()`` call the caller makes
    for as long as it holds this binding alive. ``frozen_inference`` is
    declared ``compare=False`` because ``FrozenInference`` defines no
    ``__eq__`` of its own. No caching beyond this one object otherwise: a
    second call to ``resolve_release_binding`` against the same release root
    builds a brand-new, independently re-verified ``ScoringReleaseBinding``
    with a brand-new, empty-cache ``FrozenInference``.
    """

    release_id: str
    #: Keyed ``"{role}:{strategy_id}"``.
    model_identity: Mapping[str, ModelIdentity]
    #: Keyed by ``binding_id`` -> that binding's composite artifact hash
    #: (the same value as ``model_identity[...].artifact_hash``).
    model_artifact_refs: Mapping[str, str]
    #: The verified staged release, exactly what PR-3 assigns to
    #: ``SourceBundle.model_release``.
    model_release: ModelRelease
    #: Exactly what PR-3 assigns to ``SourceBundle.frozen_inference``. See
    #: the class docstring for its cache/equality semantics.
    frozen_inference: FrozenInference = field(compare=False)
    #: Keyed by the loaded artifact's own ``.strategy`` field (never by a
    #: manifest row's declared ``strategies`` list).
    payoff_artifacts: Mapping[str, tuple[PayoffLineArtifact | PayoffSurfaceArtifact, ...]]
    #: Keyed by strategy, same rule as ``payoff_artifacts``.
    recalibration_artifacts: Mapping[str, tuple[RecalibrationMapArtifact, ...]]
    #: Keyed by strategy, same rule as ``payoff_artifacts``.
    analog_artifacts: Mapping[str, tuple[BoardAnalogPoolArtifact, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_identity", MappingProxyType(dict(self.model_identity)))
        object.__setattr__(self, "model_artifact_refs",
                           MappingProxyType(dict(self.model_artifact_refs)))
        object.__setattr__(self, "payoff_artifacts",
                           MappingProxyType({k: tuple(v) for k, v in self.payoff_artifacts.items()}))
        object.__setattr__(self, "recalibration_artifacts",
                           MappingProxyType({k: tuple(v) for k, v in self.recalibration_artifacts.items()}))
        object.__setattr__(self, "analog_artifacts",
                           MappingProxyType({k: tuple(v) for k, v in self.analog_artifacts.items()}))


def resolve_release_binding(release_root: Path | str) -> ScoringReleaseBinding:
    """Resolve the live release at ``release_root``. See
    ``engine/v2/scoring/ARCHITECTURE.md``'s ``release_bindings.py`` section
    (the "4c R1-R6 template") for the full failure-semantics contract.
    Read-only: performs no writes, and constructs fresh artifact loaders and
    a fresh ``FrozenInference`` on every call (no caching beyond this one
    call's returned object).
    """
    if not str(release_root).strip():
        raise ValueError("release_root must be a non-empty path")
    root = Path(release_root)
    dep_root = root / _DEPLOYMENT_DIR

    try:
        pointer = deployment.current_pointer(dep_root)
    except ValueError as exc:
        # R1(b): a corrupt/unparseable DEPLOYED file. json.JSONDecodeError,
        # engine.v2.foundation.typed.DocumentError and UnicodeDecodeError
        # (invalid UTF-8) are all ValueError subclasses; none is a
        # DeploymentError, so this module catches them here explicitly.
        raise ModelNotReady("DEPLOYED", str(exc)) from exc
    if pointer is None:
        raise NoCurrentRelease(f"no release has ever been promoted at {dep_root}")

    manifest = _read_and_verify_manifest(dep_root, pointer.release_id)
    release = manifest.release

    model_identity, model_artifact_refs = _resolve_model_bindings(dep_root, release)
    catalog = _read_state_catalog(root, pointer.release_id)
    payoff = _resolve_state_group(
        dep_root, catalog, lambda member_id: member_id.startswith(_PAYOFF_PREFIXES), _load_payoff)
    recalibration = _resolve_state_group(
        dep_root, catalog, lambda member_id: member_id.startswith(_RECALIBRATION_PREFIX),
        _load_recalibration)
    analog = _resolve_state_group(
        dep_root, catalog, lambda member_id: member_id == _ANALOG_MEMBER_ID, _load_analog)

    return ScoringReleaseBinding(
        release_id=release.release_id, model_identity=model_identity,
        model_artifact_refs=model_artifact_refs, model_release=release,
        frozen_inference=FrozenInference(dep_root),
        payoff_artifacts=payoff, recalibration_artifacts=recalibration, analog_artifacts=analog,
    )


def _read_and_verify_manifest(dep_root: Path, release_id: str) -> "deployment.StagedManifest":
    """R1(c)-(f): the one staged-manifest read, via ``deployment._read_manifest``
    -- never through the public ``resolve_release()`` wrapper, and never
    re-read later. Every way this can fail is its own typed refusal; only
    once all of them pass is the result trusted."""
    try:
        manifest = deployment._read_manifest(dep_root, release_id)
    except deployment.DeploymentError as exc:
        # R1(c): an unsafe release_id (contains "/" or is "."/"..").
        raise ModelNotReady("model_release", str(exc)) from exc
    except ValueError as exc:
        # R1(e): manifest.json exists but will not parse (same ValueError
        # family as R1(b) above).
        raise ModelNotReady("manifest.json", str(exc)) from exc
    if manifest is None:
        # R1(d): no staged manifest for this release_id.
        raise ModelNotReady("model_release", "release not staged")
    if not deployment._manifest_hash_matches(manifest):
        # R1(f): the staged manifest's own release_hash disagrees.
        raise ModelNotReady("model_release", "release_hash disagrees with manifest")
    return manifest


def _verify_object_bytes(dep_root: Path, member_id: str, name: str, path: str,
                         expected_hash: str) -> str:
    """Read and hash-verify one model-binding member object. Never falls
    back: a missing file or a hash mismatch is always ``ModelNotReady``
    naming ``member_id``, and nothing else is ever substituted for it.
    (The payoff/recalibration/analog loaders verify their own objects
    internally -- this helper is only for model-binding members, which have
    no dedicated loader.)"""
    base = dep_root.resolve()
    target = (base / path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise ModelNotReady(member_id, f"{name}: object path escapes the deployment root")
    if not target.is_file():
        raise ModelNotReady(member_id, f"{name}: missing object at {path}")
    try:
        payload = target.read_bytes()
    except OSError as exc:
        raise ModelNotReady(member_id, f"{name}: cannot read object at {path}: {exc}") from exc
    actual = "sha256:" + hashlib.sha256(payload).hexdigest()
    if actual != expected_hash:
        raise ModelNotReady(member_id, f"{name}: object hash disagrees with the pointer")
    return actual


def _resolve_model_bindings(
    dep_root: Path, release: ModelRelease,
) -> tuple[dict[str, ModelIdentity], dict[str, str]]:
    identity: dict[str, ModelIdentity] = {}
    refs: dict[str, str] = {}
    seen_keys: dict[str, int] = {}
    for binding in release.bindings:
        key = f"{binding.role}:{binding.strategy_id}"
        seen_keys[key] = seen_keys.get(key, 0) + 1
        member_id = f"model:{key}"
        member_hashes = [
            _verify_object_bytes(dep_root, member_id, member.name, member.path, member.content_hash)
            for member in binding.members
        ]
        artifact_hash = content_hash(tuple(sorted(member_hashes)))
        identity[key] = ModelIdentity(
            binding_id=binding.binding_id, model_id=binding.model_id, role=binding.role,
            strategy_id=binding.strategy_id, decision_clock_id=binding.decision_clock_id,
            adapter=binding.adapter, feature_order=binding.feature_order,
            output_names=binding.output_names, artifact_hash=artifact_hash,
        )
        refs[binding.binding_id] = artifact_hash
    duplicates = sorted(key for key, count in seen_keys.items() if count > 1)
    if duplicates:
        raise ModelNotReady(f"model:{duplicates[0]}",
                            f"ambiguous binding: {seen_keys[duplicates[0]]} declared")
    return identity, refs


def _read_state_catalog(release_root: Path, expected_release_id: str) -> Mapping:
    """The ``phase5_release.json`` catalog: parsed, schema-checked,
    self-hash-verified, and checked against the live pointer's release_id --
    all before any of its rows are trusted."""
    path = release_root / _STATE_CATALOG_NAME
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ModelNotReady(_STATE_CATALOG_NAME, f"cannot read {_STATE_CATALOG_NAME}: {exc}") from exc
    try:
        body = json.loads(raw)
    except ValueError as exc:
        raise ModelNotReady(_STATE_CATALOG_NAME, f"{_STATE_CATALOG_NAME} is not valid JSON") from exc
    if not isinstance(body, dict):
        raise ModelNotReady(_STATE_CATALOG_NAME, f"{_STATE_CATALOG_NAME} must be a JSON object")
    if body.get("schema_version") != _PHASE5_RELEASE_SCHEMA:
        raise ModelNotReady(_STATE_CATALOG_NAME,
                            f"unsupported schema_version {body.get('schema_version')!r}")
    claimed = body.get("manifest_hash")
    unhashed = {k: v for k, v in body.items() if k != "manifest_hash"}
    if claimed != content_hash(unhashed):
        raise ModelNotReady(_STATE_CATALOG_NAME, "manifest_hash does not match its own body")
    if body.get("release_id") != expected_release_id:
        raise ModelNotReady(
            _STATE_CATALOG_NAME,
            f"release_id {body.get('release_id')!r} disagrees with the deployed pointer "
            f"{expected_release_id!r}")
    return body


def _resolve_state_group(
    dep_root: Path, catalog: Mapping, matches: Callable[[str], bool], loader,
) -> dict[str, tuple]:
    grouped: dict[str, list] = {}
    for row in catalog.get("members", ()):
        if not isinstance(row, Mapping):
            raise ModelNotReady("phase5_release.json", "a members row is not a JSON object")
        member_id = row.get("member_id", "")
        if not matches(member_id):
            continue
        objects = row.get("objects")
        if row.get("status") != "STAGED" or not objects:
            raise ModelNotReady(member_id, f"status={row.get('status')!r}, no staged objects")
        if not isinstance(objects, list):
            raise ModelNotReady(member_id, "objects must be a JSON array")
        for obj in objects:
            if (not isinstance(obj, Mapping)
                    or not isinstance(obj.get("path"), str)
                    or not isinstance(obj.get("content_hash"), str)):
                raise ModelNotReady(member_id, f"malformed object reference: {obj!r}")
            artifact = loader(dep_root, member_id, obj)
            grouped.setdefault(artifact.strategy, []).append(artifact)
    return {strategy: tuple(items) for strategy, items in grouped.items()}


def _load_payoff(dep_root: Path, member_id: str, obj: Mapping):
    try:
        return PayoffArtifactLoader(dep_root).load(
            PayoffArtifactRef(path=obj["path"], content_hash=obj["content_hash"]))
    except PayoffArtifactError as exc:
        raise ModelNotReady(member_id, str(exc)) from exc


def _load_recalibration(dep_root: Path, member_id: str, obj: Mapping):
    try:
        return RecalibrationArtifactLoader(dep_root).load(
            RecalibrationArtifactRef(path=obj["path"], content_hash=obj["content_hash"]))
    except RecalibrationArtifactError as exc:
        raise ModelNotReady(member_id, str(exc)) from exc


def _load_analog(dep_root: Path, member_id: str, obj: Mapping):
    try:
        state = FrozenStateLoader(dep_root).load(
            FrozenStateRef(path=obj["path"], content_hash=obj["content_hash"]))
    except FrozenStateError as exc:
        raise ModelNotReady(member_id, str(exc)) from exc
    if not isinstance(state, BoardAnalogPoolArtifact):
        raise ModelNotReady(member_id, f"expected a board analog pool, got {type(state).__name__}")
    return state
