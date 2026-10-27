"""Production reader of one live release's scoring catalog.

Resolves model identity, model artifact refs, the verified ``ModelRelease``, a
constructed ``FrozenInference``, and the frozen scoring state artifacts
from a live ``deployment.current_pointer()`` -- the failure-semantics contract
is ``engine/v2/scoring/ARCHITECTURE.md``'s ``release_bindings.py`` section (the
"4c R1-R6 template"). Native batch and the supervisor's release-readiness
check consume this reader; nightly selects exact-cutoff chooser state and
causal producer folds for registered menu-member rows.

Every refusal below is path-free by construction: a fixed message plus the
exact ``member_id``/field name (and, where useful, a type name), never a raw
filesystem path, an exception's own text, or ``repr()`` of untrusted content.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, TypeVar

from engine.v2.foundation import content_hash, from_document
from engine.v2.models import deployment
from engine.v2.models.admissible_table import N_ADMISSIBLE_TABLE_ID, AdmissibleDepthTable
from engine.v2.models.analog_artifact import BoardAnalogPoolArtifact
from engine.v2.models.chooser_analog_pool import ChooserAnalogPoolArtifact
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
from engine.v2.models.residual_artifact import (
    DriverResidualPoolArtifact,
    PairedResidualPoolArtifact,
    ResidualArtifactError,
    residual_artifact_from_document,
)
from engine.v2.models.serving_fold_loader import load_chooser_serving_fold
from engine.v2.models.serving_folds import ChooserServingFoldDescriptor
from engine.v2.models.trailing_cutoff_artifact import TrailingCutoffArtifact
from engine.v2.scoring.gate_fold_selection import (
    SizeFoldCatalog,
    index_size_folds,
    select_gate_forecast_source,
)

__all__ = [
    "ChooserState",
    "ModelIdentity",
    "ModelNotReady",
    "NoCurrentRelease",
    "ScoringReleaseBinding",
    "ReleaseBindingError",
    "resolve_gate_policy",
    "resolve_production_release_binding",
    "resolve_release_binding",
    "select_chooser_state",
]

_DEPLOYMENT_DIR = "deployment"
_STATE_CATALOG_NAME = "phase5_release.json"
_MAX_CATALOG_BYTES = 64 * 1024 * 1024
_PHASE5_RELEASE_SCHEMA = "phase5_staged_release.v1.0"
_PAYOFF_PREFIXES = ("payoff_line:", "payoff_surface:")
_RECALIBRATION_PREFIX = "recalibration_map:"
_ANALOG_MEMBER_ID = "board_analog_matcher"
_DRIVER_RESIDUAL_PREFIX = "driver_residual_pool:"
_CHOOSER_POOL_MEMBER_ID = "chooser_analog_pool"
_CHOOSER_TABLE_MEMBER_ID = "admissible_table:dyn_sv"
_CHOOSER_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
_PAIRED_POOL_MEMBER_ID = "paired_residual_pool"
_TRAILING_CUTOFF_MEMBER_ID = "trailing_pnl_cutoff"
_ChooserState = TypeVar(
    "_ChooserState", ChooserAnalogPoolArtifact, AdmissibleDepthTable,
    PairedResidualPoolArtifact, TrailingCutoffArtifact)


class ReleaseBindingError(ValueError):
    """Base for every refusal this module raises."""


class NoCurrentRelease(ReleaseBindingError):
    """R1(a): nothing has ever been promoted at this release root."""


class ModelNotReady(ReleaseBindingError):
    """One named member -- the DEPLOYED pointer, the staged manifest, a model
    binding member, or a frozen scoring state object -- is
    missing, wrong status, hash-mismatched, or rejected by its own typed
    loader. Never a fallback: this is always the only outcome for a bad
    member (R1). ``detail`` is always a fixed, path-free message."""

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
    #: Driver residual pools, keyed by the loaded artifact's own ``.role``
    #: field. Empty for a release that stages none.
    driver_residual_artifacts: Mapping[str, DriverResidualPoolArtifact] = field(default_factory=dict)
    #: All verified objects, preserving catalog order; request-specific
    #: selection remains the assembler's responsibility. Absent -> empty.
    chooser_analog_artifacts: tuple[ChooserAnalogPoolArtifact, ...] = ()
    chooser_admissible_artifacts: tuple[AdmissibleDepthTable, ...] = ()
    paired_residual_artifacts: tuple[PairedResidualPoolArtifact, ...] = ()
    trailing_cutoff_artifacts: tuple[TrailingCutoffArtifact, ...] = ()
    #: Indexed descriptors of the release's ``tier4_folds:size`` member; absent
    #: or not STAGED -> empty. See ARCHITECTURE.md ("Forward forecast assembly").
    size_folds: "SizeFoldCatalog | None" = None
    chooser_fold_catalog_json: str = field(default="", repr=False, compare=False)
    chooser_parent_manifest: Any = field(default=None, repr=False, compare=False)

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
        object.__setattr__(self, "driver_residual_artifacts",
                           MappingProxyType(dict(self.driver_residual_artifacts)))
        object.__setattr__(self, "chooser_analog_artifacts", tuple(self.chooser_analog_artifacts))
        object.__setattr__(self, "chooser_admissible_artifacts",
                           tuple(self.chooser_admissible_artifacts))
        object.__setattr__(self, "paired_residual_artifacts",
                           tuple(self.paired_residual_artifacts))
        object.__setattr__(self, "trailing_cutoff_artifacts",
                           tuple(self.trailing_cutoff_artifacts))


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
    except (OSError, ValueError) as exc:
        # R1(b): a corrupt/unparseable/unreadable DEPLOYED file. json.
        # JSONDecodeError, engine.v2.foundation.typed.DocumentError and
        # UnicodeDecodeError (invalid UTF-8) are all ValueError subclasses;
        # a read racing the is_file() check inside current_pointer raises a
        # bare OSError. None is a DeploymentError, so this module catches
        # them here explicitly, with a fixed, path-free message.
        raise ModelNotReady("DEPLOYED", "the deployment pointer could not be read") from exc
    if pointer is None:
        raise NoCurrentRelease("no release has ever been promoted at this release root")

    manifest = _read_and_verify_manifest(dep_root, pointer.release_id)
    release = manifest.release

    model_identity, model_artifact_refs = _resolve_model_bindings(dep_root, release)
    catalog = _read_state_catalog(root, pointer.release_id)
    size_folds = index_size_folds(catalog, manifest)
    payoff = _resolve_state_group(
        dep_root, catalog, lambda member_id: member_id.startswith(_PAYOFF_PREFIXES), _load_payoff)
    recalibration = _resolve_state_group(
        dep_root, catalog, lambda member_id: member_id.startswith(_RECALIBRATION_PREFIX),
        _load_recalibration)
    analog = _resolve_state_group(
        dep_root, catalog, lambda member_id: member_id == _ANALOG_MEMBER_ID, _load_analog)
    driver_residual = _resolve_driver_residual_pools(dep_root, catalog)
    chooser_analog = _resolve_chooser_state_group(
        dep_root, catalog, _CHOOSER_POOL_MEMBER_ID, ChooserAnalogPoolArtifact)
    chooser_admissible = _resolve_chooser_state_group(
        dep_root, catalog, _CHOOSER_TABLE_MEMBER_ID, AdmissibleDepthTable)
    paired_residual = _resolve_chooser_state_group(
        dep_root, catalog, _PAIRED_POOL_MEMBER_ID, PairedResidualPoolArtifact,
        label="paired residual pool")
    trailing_cutoff = _resolve_chooser_state_group(
        dep_root, catalog, _TRAILING_CUTOFF_MEMBER_ID, TrailingCutoffArtifact,
        label="trailing cutoff")

    return ScoringReleaseBinding(
        release_id=release.release_id, model_identity=model_identity,
        model_artifact_refs=model_artifact_refs, model_release=release,
        frozen_inference=FrozenInference(dep_root),
        payoff_artifacts=payoff, recalibration_artifacts=recalibration, analog_artifacts=analog,
        driver_residual_artifacts=driver_residual,
        chooser_analog_artifacts=chooser_analog,
        chooser_admissible_artifacts=chooser_admissible,
        paired_residual_artifacts=paired_residual,
        trailing_cutoff_artifacts=trailing_cutoff, size_folds=size_folds,
        chooser_fold_catalog_json=json.dumps(catalog, sort_keys=True, separators=(",", ":")),
        chooser_parent_manifest=manifest,
    )


@dataclass(frozen=True, kw_only=True)
class ChooserState:
    analog_pool: ChooserAnalogPoolArtifact
    admissible_table: AdmissibleDepthTable
    fold_pools: Mapping[str, Mapping[str, Any]]
    producer_bindings: Mapping[str, Any]


def _chooser_day(value: Any) -> str:
    try:
        if isinstance(value, datetime):
            day = value.date()
        elif isinstance(value, date):
            day = value
        elif isinstance(value, str):
            if len(value) > 10 and value[10] in ("T", " "):
                day = datetime.fromisoformat(value).date()
            else:
                day = date.fromisoformat(value)
        else:
            raise ValueError("unsupported chooser day type")
        return day.isoformat()
    except (TypeError, ValueError, OverflowError):
        raise ModelNotReady("chooser state", "chooser state is missing or unusable") from None


def _chooser_fold_descriptor(
    binding: ScoringReleaseBinding, catalog: Mapping[str, Any],
    role: str, fold_start: str,
) -> ChooserServingFoldDescriptor:
    members = catalog.get("members")
    if not isinstance(members, list):
        raise ValueError("chooser fold catalog malformed")
    rows = [row for row in members if isinstance(row, dict)
            and row.get("member_id") == f"tier4_folds:{role}"]
    if len(rows) != 1 or rows[0].get("status") != "STAGED":
        raise ValueError("chooser fold member missing")
    objects = rows[0].get("objects")
    if not isinstance(objects, list):
        raise ValueError("chooser fold objects malformed")
    matches = []
    for obj in objects:
        descriptor = obj.get("serving_fold") if isinstance(obj, dict) else None
        policy = descriptor.get("policy") if isinstance(descriptor, dict) else None
        if (isinstance(descriptor, dict) and isinstance(policy, dict)
                and policy.get("role") == role
                and descriptor.get("fold_start") == fold_start):
            matches.append(descriptor)
    if len(matches) != 1:
        raise ValueError("chooser fold for cutoff missing or ambiguous")
    return from_document(ChooserServingFoldDescriptor, matches[0])


def _load_chooser_fold_state(
    binding: ScoringReleaseBinding, event_date: Any, as_of: Any,
    fold_start: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    size = select_gate_forecast_source(binding, event_date, as_of)
    pools = {"pred_abs_move": {
        "predictions": size.pool_pred, "residuals": size.pool_res,
        "interval_floor": size.interval_floor,
    }}
    producers = {}
    catalog_text = binding.chooser_fold_catalog_json
    parent = binding.chooser_parent_manifest
    if not catalog_text or parent is None:
        raise ValueError("chooser fold catalog missing")
    catalog = json.loads(catalog_text)
    for role in ("implied_t1", "runup_move"):
        descriptor = _chooser_fold_descriptor(binding, catalog, role, fold_start)
        ref = load_chooser_serving_fold(
            binding.frozen_inference, descriptor,
            verified_parent_manifest=parent, verified_catalog=catalog)
        if not ref.release.bindings:
            raise ValueError("chooser producer binding missing")
        pools[descriptor.producer_output] = {
            "predictions": ref.pool_pred, "residuals": ref.pool_res,
            "interval_floor": descriptor.policy.interval_floor,
        }
        producers[descriptor.producer_output] = ref.release.bindings[0]
    return pools, producers


def select_chooser_state(
    binding: ScoringReleaseBinding, event_date: Any, as_of: Any,
    *, fold_cache: dict[tuple[int, str], Any] | None = None,
) -> ChooserState:
    """Select exact cutoff chooser artifacts and causal Tier-4 pools."""
    cutoff = _chooser_day(as_of)
    analog = [item for item in binding.chooser_analog_artifacts if item.cutoff == cutoff]
    if len(analog) != 1:
        raise ModelNotReady("chooser_analog_pool", "chooser state is missing or unusable")
    tables = [item for item in binding.chooser_admissible_artifacts
              if item.key == (N_ADMISSIBLE_TABLE_ID, "v1")]
    if len(tables) != 1:
        raise ModelNotReady("admissible_table:dyn_sv", "chooser state is missing or unusable")
    try:
        event_month = date.fromisoformat(_chooser_day(event_date)).replace(day=1)
        decision_month = date.fromisoformat(cutoff).replace(day=1)
    except (TypeError, ValueError, OverflowError):
        raise ModelNotReady("chooser state", "chooser state is missing or unusable") from None
    fold_start = min(event_month, decision_month).isoformat()
    if date.fromisoformat(fold_start) >= date.fromisoformat(cutoff):
        raise ModelNotReady("chooser state", "chooser state is missing or unusable")
    cache_key = (id(binding), f"chooser:{fold_start}")
    fold_state = None if fold_cache is None else fold_cache.get(cache_key)
    if fold_state is None:
        try:
            fold_state = _load_chooser_fold_state(
                binding, event_date, as_of, fold_start)
        except ModelNotReady:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
            raise ModelNotReady("chooser state", "chooser state is missing or unusable") from None
        if fold_cache is not None:
            fold_cache[cache_key] = fold_state
    pools, producers = fold_state
    return ChooserState(
        analog_pool=analog[0], admissible_table=tables[0], fold_pools=pools,
        producer_bindings=producers)


def resolve_production_release_binding() -> ScoringReleaseBinding:
    """``resolve_release_binding`` against the ONE configured production
    release root (``engine.v2.models.deployment.production_release_root``).

    R1(g): a missing ``MODEL_RELEASE_ROOT`` is ``deployment.
    MissingReleaseRoot``, wrapped here as ``ModelNotReady("release_root",
    ...)`` -- this module's one refusal shape covers a missing config key
    exactly like a missing file. See ``engine/v2/scoring/ARCHITECTURE.md``'s
    ``release_bindings.py`` section.
    """
    try:
        root = deployment.production_release_root()
    except deployment.MissingReleaseRoot as exc:
        raise ModelNotReady("release_root", "no production release root is configured") from exc
    return resolve_release_binding(root)


def resolve_gate_policy(
    binding: ScoringReleaseBinding, release_root: Path | str,
) -> dict[str, dict[str, float]]:
    """``{strategy: {"threshold": float}}`` from each gate binding's staged
    ``threshold`` member (a content-addressed copy of the model registry; the
    threshold is the registry entry whose ``id`` is the binding's
    ``model_id``). A gate binding with no ``threshold`` member is omitted. No
    unique matching release binding for the gate identity, a binding with more
    than one threshold member, a
    missing/unreadable/hash-mismatched object, an unparseable registry, no
    unique entry for the model, or a non-finite/non-numeric threshold raises
    ``ModelNotReady`` with no fallback. Read-only, no cache: see
    ``engine/v2/scoring/ARCHITECTURE.md``'s ``resolve_gate_policy`` section."""
    dep_root = Path(release_root) / _DEPLOYMENT_DIR
    policy: dict[str, dict[str, float]] = {}
    for key, identity in binding.model_identity.items():
        if identity.role != "gate":
            continue
        matches = [b for b in binding.model_release.bindings
                   if b.binding_id == identity.binding_id and b.role == identity.role
                   and b.strategy_id == identity.strategy_id]
        if len(matches) != 1:
            raise ModelNotReady(
                f"model:{key}", "threshold: gate identity does not match exactly one release binding")
        threshold_members = [m for m in matches[0].members
                             if m.name == "threshold"]
        if len(threshold_members) > 1:
            raise ModelNotReady(
                f"model:{key}", "threshold: binding has more than one threshold member")
        if not threshold_members:
            continue
        member = threshold_members[0]
        member_id = f"model:{key}"
        _, payload = _read_verified_object(
            dep_root, member_id, member.name, member.path, member.content_hash)
        policy[identity.strategy_id] = {
            "threshold": _registry_threshold(member_id, payload, identity.model_id)}
    return policy


def _registry_threshold(member_id: str, payload: bytes, model_id: str) -> float:
    try:
        doc = json.loads(payload)
    except (ValueError, RecursionError) as exc:
        raise ModelNotReady(member_id, "threshold: object is not valid JSON") from exc
    models = doc.get("models") if isinstance(doc, dict) else None
    entries = ([e for e in models if isinstance(e, dict) and e.get("id") == model_id]
               if isinstance(models, list) else [])
    if len(entries) != 1:
        raise ModelNotReady(member_id, "threshold: registry has no unique entry for the gate model")
    value = entries[0].get("threshold")
    try:
        number = None if isinstance(value, bool) or not isinstance(value, (int, float)) else float(value)
    except OverflowError:
        number = None
    if number is None or not math.isfinite(number):
        raise ModelNotReady(member_id, "threshold: entry threshold is not a finite number")
    return number


def _read_and_verify_manifest(dep_root: Path, release_id: str) -> "deployment.StagedManifest":
    """R1(c)-(f): the one staged-manifest read, via ``deployment._read_manifest``
    -- never through the public ``resolve_release()`` wrapper, and never
    re-read later. Every way this can fail is its own typed refusal; only
    once all of them pass is the result trusted."""
    try:
        manifest = deployment._read_manifest(dep_root, release_id)
    except deployment.DeploymentError as exc:
        # R1(c): an unsafe release_id (contains "/" or is "."/".."). The
        # DeploymentError's own message would echo the release_id verbatim,
        # so it is never passed through -- fixed message only.
        raise ModelNotReady("model_release", "unsafe or invalid release id") from exc
    except (OSError, ValueError) as exc:
        # R1(e): manifest.json exists but will not parse (same ValueError
        # family as R1(b) above), or deployment._read_manifest's lstat plus
        # bounded no-follow read raises a bare OSError.
        raise ModelNotReady("manifest.json", "the staged manifest could not be read") from exc
    if manifest is None:
        # R1(d): no staged manifest for this release_id.
        raise ModelNotReady("model_release", "release not staged")
    if not deployment._manifest_hash_matches(manifest):
        # R1(f): the staged manifest's own release_hash disagrees.
        raise ModelNotReady("model_release", "release_hash disagrees with manifest")
    return manifest


def _read_verified_object(dep_root: Path, member_id: str, name: str, path: str,
                          expected_hash: str) -> tuple[str, bytes]:
    """Read and hash-verify one model-binding member object. Never falls
    back: a missing file or a hash mismatch is always ``ModelNotReady``
    naming ``member_id``, and nothing else is ever substituted for it.
    (The payoff/recalibration/analog loaders verify their own objects
    internally -- this helper is only for model-binding members, which have
    no dedicated loader.)"""
    base = dep_root.resolve()
    try:
        target = (base / path).resolve()
    except ValueError as exc:
        # A path with an embedded NUL (or another path resolve() rejects)
        # raises a bare ValueError, never an OSError; translate it to the
        # same fixed, path-free refusal instead of echoing the supplied path.
        raise ModelNotReady(member_id, f"{name}: object path is invalid") from exc
    try:
        target.relative_to(base)
    except ValueError:
        raise ModelNotReady(member_id, f"{name}: object path escapes the deployment root")
    if not target.is_file():
        raise ModelNotReady(member_id, f"{name}: missing staged object")
    try:
        payload = target.read_bytes()
    except OSError as exc:
        raise ModelNotReady(member_id, f"{name}: cannot read the staged object") from exc
    actual = "sha256:" + hashlib.sha256(payload).hexdigest()
    if actual != expected_hash:
        raise ModelNotReady(member_id, f"{name}: object hash disagrees with the pointer")
    return actual, payload


def _verify_object_bytes(dep_root: Path, member_id: str, name: str, path: str,
                         expected_hash: str) -> str:
    """The verified content hash of one model-binding member object (see
    :func:`_read_verified_object`)."""
    return _read_verified_object(dep_root, member_id, name, path, expected_hash)[0]


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


def _read_catalog_bytes(local: Path, root_copy: Path) -> bytes:
    """No-follow, non-blocking, regular-file, size-capped read; local first, then the legacy root copy when local is absent."""
    unreadable = ModelNotReady(_STATE_CATALOG_NAME, "catalog is not a readable file")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        try:
            fd = os.open(local, flags)
        except FileNotFoundError:
            fd = os.open(root_copy, flags)
    except OSError as exc:
        raise ModelNotReady(_STATE_CATALOG_NAME, f"cannot read {_STATE_CATALOG_NAME}") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise unreadable
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            raw = handle.read(_MAX_CATALOG_BYTES + 1)
    except OSError as exc:
        raise ModelNotReady(_STATE_CATALOG_NAME, f"cannot read {_STATE_CATALOG_NAME}") from exc
    finally:
        if fd >= 0:
            os.close(fd)
    if len(raw) > _MAX_CATALOG_BYTES:
        raise unreadable
    return raw


def _read_state_catalog(release_root: Path, expected_release_id: str) -> Mapping:
    """The ``phase5_release.json`` catalog: parsed, schema-checked,
    self-hash-verified, and checked against the live pointer's release_id --
    all before any of its rows are trusted."""
    path = deployment.manifest_path(
        release_root / _DEPLOYMENT_DIR, expected_release_id).with_name(_STATE_CATALOG_NAME)
    raw = _read_catalog_bytes(path, release_root / _STATE_CATALOG_NAME)
    try:
        body = json.loads(raw)
    except ValueError as exc:
        raise ModelNotReady(_STATE_CATALOG_NAME, f"{_STATE_CATALOG_NAME} is not valid JSON") from exc
    if not isinstance(body, dict):
        raise ModelNotReady(_STATE_CATALOG_NAME, f"{_STATE_CATALOG_NAME} must be a JSON object")
    if body.get("schema_version") != _PHASE5_RELEASE_SCHEMA:
        raise ModelNotReady(_STATE_CATALOG_NAME, "unsupported schema_version")
    claimed = body.get("manifest_hash")
    unhashed = {k: v for k, v in body.items() if k != "manifest_hash"}
    if claimed != content_hash(unhashed):
        raise ModelNotReady(_STATE_CATALOG_NAME, "manifest_hash does not match its own body")
    if body.get("release_id") != expected_release_id:
        raise ModelNotReady(_STATE_CATALOG_NAME, "release_id disagrees with the deployed pointer")
    from engine.v2.models.release_layout import ReleaseLayoutError, validate_catalog

    try:
        validate_catalog(body, expected_release_id)
    except ReleaseLayoutError as exc:
        raise ModelNotReady(
            _STATE_CATALOG_NAME, "cutoff metadata is invalid") from exc
    return body


def _artifact_strategy(artifact) -> str:
    return artifact.strategy


def _resolve_state_group(
    dep_root: Path, catalog: Mapping, matches: Callable[[str], bool], loader,
    key: Callable[[object], str] = _artifact_strategy,
) -> dict[str, tuple]:
    """Rows whose ``member_id`` satisfies ``matches`` are loaded through
    ``loader`` and grouped by ``key`` (default ``_artifact_strategy``: the
    loaded artifact's own ``.strategy``) into ``{group: (artifact, ...)}``
    -- one shared catalog contract for the payoff/recalibration/analog
    groups and the driver residual pools."""
    members = catalog.get("members")
    if members is None or not isinstance(members, list):
        raise ModelNotReady("phase5_release.json", "members must be a JSON array")
    grouped: dict[str, list] = {}
    for row in members:
        if not isinstance(row, Mapping):
            raise ModelNotReady("phase5_release.json", "a members row is not a JSON object")
        member_id = row.get("member_id", "")
        if not isinstance(member_id, str):
            raise ModelNotReady("phase5_release.json", "a members row has a non-string member_id")
        if not matches(member_id):
            continue
        objects = row.get("objects")
        if row.get("status") != "STAGED" or not objects:
            raise ModelNotReady(member_id, "status is not STAGED, or no staged objects")
        if not isinstance(objects, list):
            raise ModelNotReady(member_id, "objects must be a JSON array")
        for obj in objects:
            if (not isinstance(obj, Mapping)
                    or not isinstance(obj.get("path"), str)
                    or not isinstance(obj.get("content_hash"), str)):
                raise ModelNotReady(member_id, "malformed object reference in phase5_release.json")
            artifact = loader(dep_root, member_id, obj)
            grouped.setdefault(key(artifact), []).append(artifact)
    return {group: tuple(items) for group, items in grouped.items()}


def _load_payoff(dep_root: Path, member_id: str, obj: Mapping):
    try:
        return PayoffArtifactLoader(dep_root).load(
            PayoffArtifactRef(path=obj["path"], content_hash=obj["content_hash"]))
    except (PayoffArtifactError, KeyError, TypeError, ValueError) as exc:
        # A hash-valid document missing a required field raises a bare
        # KeyError out of PayoffArtifactLoader.load() (it does not guard its
        # own document-to-artifact conversion) -- caught here, not left to
        # escape uncaught.
        raise ModelNotReady(member_id, "payoff artifact could not be verified or loaded") from exc


def _load_recalibration(dep_root: Path, member_id: str, obj: Mapping):
    try:
        return RecalibrationArtifactLoader(dep_root).load(
            RecalibrationArtifactRef(path=obj["path"], content_hash=obj["content_hash"]))
    except (RecalibrationArtifactError, KeyError, TypeError, ValueError) as exc:
        raise ModelNotReady(member_id, "recalibration artifact could not be verified or loaded") from exc


def _load_analog(dep_root: Path, member_id: str, obj: Mapping):
    try:
        state = FrozenStateLoader(dep_root).load(
            FrozenStateRef(path=obj["path"], content_hash=obj["content_hash"]))
    except (FrozenStateError, KeyError, TypeError) as exc:
        raise ModelNotReady(member_id, "analog state could not be verified or loaded") from exc
    if not isinstance(state, BoardAnalogPoolArtifact):
        raise ModelNotReady(member_id, f"expected a board analog pool, got {type(state).__name__}")
    return state


def _resolve_chooser_state_group(
    dep_root: Path, catalog: Mapping, member_id: str, kind: type[_ChooserState],
    label: str = "chooser state",
) -> tuple[_ChooserState, ...]:
    """Load a whole optional family without selecting a request's state.

    Several distinct semantic keys are valid; duplicate keys are ambiguous
    even if their bytes agree. The typed loader verifies both byte and
    decoded hashes. A fresh loader per object prevents cache reuse from
    concealing a corrupt second reference to the same content hash.
    """
    def load(root: Path, member: str, obj: Mapping) -> _ChooserState:
        try:
            state = FrozenStateLoader(root, max_bytes=_CHOOSER_MAX_ARTIFACT_BYTES).load(
                FrozenStateRef(path=obj["path"], content_hash=obj["content_hash"]))
        except (OSError, ValueError, KeyError, TypeError, AttributeError,
                OverflowError, RuntimeError) as exc:
            # Decoder numeric/shape/depth errors and path resolution failures
            # must not leak raw parser or filesystem details.
            raise ModelNotReady(member, f"{label} could not be verified or loaded") from exc
        if not isinstance(state, kind):
            raise ModelNotReady(member, f"{label} has the wrong artifact type")
        return state

    grouped = _resolve_state_group(
        dep_root, catalog, lambda candidate: candidate == member_id, load,
        key=lambda state: member_id)
    artifacts = grouped.get(member_id, ())
    if len({artifact.key for artifact in artifacts}) != len(artifacts):
        raise ModelNotReady(member_id, f"ambiguous {label} key")
    return artifacts


def _resolve_driver_residual_pools(
    dep_root: Path, catalog: Mapping,
) -> dict[str, DriverResidualPoolArtifact]:
    """Every staged ``driver_residual_pool:<role>`` object, keyed by the
    artifact's own ``.role`` (one object per role; a duplicate role is an
    ambiguous member and refuses). No declared member yields ``{}``."""
    grouped = _resolve_state_group(
        dep_root, catalog, lambda member_id: member_id.startswith(_DRIVER_RESIDUAL_PREFIX),
        _load_driver_residual, key=lambda artifact: artifact.role)
    pools: dict[str, DriverResidualPoolArtifact] = {}
    for role, artifacts in grouped.items():
        if len(artifacts) != 1:
            raise ModelNotReady(
                _DRIVER_RESIDUAL_PREFIX + role, "ambiguous driver residual pool member")
        pools[role] = artifacts[0]
    return pools


def _load_driver_residual(dep_root: Path, member_id: str, obj: Mapping) -> DriverResidualPoolArtifact:
    """Verify one driver residual pool's bytes, then decode and validate it
    through the residual artifact's own typed contract. Every failure is a
    fixed, path-free ``ModelNotReady`` naming ``member_id``; no fallback."""
    verified_hash, payload = _read_verified_object(
        dep_root, member_id, "driver residual pool", obj["path"], obj["content_hash"])
    try:
        document = json.loads(payload)
    except (ValueError, RecursionError) as exc:
        # ValueError covers UnicodeDecodeError, json.JSONDecodeError, and the
        # integer string digit-limit error (a ValueError from json's number
        # parser); RecursionError covers excessive JSON nesting. All are
        # translated to the same fixed, path-free refusal.
        raise ModelNotReady(member_id, "driver residual pool is not valid JSON") from exc
    try:
        artifact = residual_artifact_from_document(document)
    except (ResidualArtifactError, KeyError, TypeError, ValueError,
            AttributeError, OverflowError, RecursionError) as exc:
        # RecursionError covers the document-to-artifact decoder hitting the
        # interpreter recursion limit on an accepted JSON document; it is
        # translated to the same fixed, path-free refusal. The JSON parsing
        # boundary above keeps its own distinct refusal unchanged.
        raise ModelNotReady(member_id, "driver residual pool could not be verified or loaded") from exc
    if not isinstance(artifact, DriverResidualPoolArtifact):
        raise ModelNotReady(member_id, "driver residual pool is not a driver-slot artifact")
    if artifact.content_hash != verified_hash:
        raise ModelNotReady(member_id, "driver residual pool disagrees with its own hash")
    if artifact.role != member_id[len(_DRIVER_RESIDUAL_PREFIX):]:
        raise ModelNotReady(member_id, "driver residual pool role disagrees with its member")
    return artifact
