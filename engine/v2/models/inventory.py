"""P5-1: the current model release, read off real files, never hand-typed.

``current_release_inventory`` maps every current registry key
(:mod:`engine.models.registry`) and role to a :class:`~.releases.ModelReleaseInventory`
built from the exact champion artifacts under ``data/models`` — content hashes,
not values, and never a rewrite of ``registry.json`` itself. It reuses the
release/inventory contracts in :mod:`engine.v2.models.releases` exactly as
written; :func:`~.releases.release_issues` and
:func:`~.releases.require_complete_release` need no changes to check this
release for completeness.

**What this covers, and how.** Six roles exist
(:data:`engine.models.registry.ROLES`). Four are feature-tier champions
(``size``, ``implied_t1``, ``runup_move``, ``iv_crush``) served two ways: the
full-refit artifact under ``data/models/<id>.joblib`` (what the registry
points at) and the monthly Tier-4 serving folds under
``data/models/tier4/<id>_<YYYYMM>_<snapshot12>.joblib`` (what
``engine.data.features.tier4.serving_model`` actually loads for a live score —
see system design §7: "Preserve the distinction between a monthly Tier-4 fold
forecast and the full refit champion forecast"). This module's
:class:`~.releases.ModelReleaseInventory` binds each role to the full-refit
champion, which is what a completeness/promotion check must see; fold coverage
is a parallel, non-binding report from :func:`tier4_fold_coverage` because a
static (role, strategy, clock) binding cannot name "whichever month is
current" without becoming stale the next month — that is P5-3/P5-4 territory
(fold selection and correction propagation), not artifact inventory.

**What is NOT in the release, and why that is not a bug in this module.**
Six pieces of serving-time state the legacy scorer builds live, every process,
from a ledger or feature table rather than from a versioned, hash-checked
artifact: the STR-THRU/STR-RUNUP payoff line, the STR-RUNUP payoff surface,
the win-rate recalibration map, the paired move/crush residual pool
(``engine.pnl_sim.ResidualPool``), the board's bucket-analog population
(``engine.analogs.AnalogMatcher``), and the DYN-SV chooser's k-NN analog pool.
None of them has a registry entry, a content hash, or a completeness check
today — see :func:`non_model_state_inventory`. That gap is exactly what
P5-4 ("Residuals and correction propagation") exists to close; recording it
here, rather than inventing a fake artifact to paper over it, is the honest
answer to "what's missing per role."

**Clocks.** Every current strategy binds to one decision clock,
``"legacy.entry_close.v1"`` (``engine.v2.registry.strategies.default_registry``,
both ``_spec`` and ``_dynamic_spec``). That id is duplicated here rather than
imported: ``engine.v2.registry`` and ``engine.v2.models`` are sibling layer-3
packages (system rearchitecture §4.1/§4.2) and neither may import the other.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from engine import paths
from engine.models import registry as legacy_registry

from .releases import (
    ArtifactInventoryMember,
    ModelArtifactInventory,
    ModelReleaseInventory,
    ReleaseBinding,
    ReleaseIssue,
    ReleaseRequirement,
)

__all__ = [
    "DEPLOYMENT_ID",
    "RELEASE_ID",
    "KNOWN_CLOCK_IDS",
    "FEATURE_ROLES",
    "NON_MODEL_STATE_ITEMS",
    "NonModelStateEntry",
    "FoldCoverageEntry",
    "current_release_inventory",
    "registry_drift_issues",
    "non_model_state_inventory",
    "tier4_fold_coverage",
    "served_roles",
]

#: The one decision clock every current strategy is bound to. See module
#: docstring for why it is a literal here rather than an import.
KNOWN_CLOCK_IDS = ("legacy.entry_close.v1",)

#: Matches ``engine.v2.registry.strategies.default_registry()``'s
#: ``deployment_id`` exactly (checked by ``tests/test_v2_models_inventory.py``
#: against a parsed copy of that module's source, so the two cannot drift
#: silently).
DEPLOYMENT_ID = "legacy-phase4-deployment.v1"

RELEASE_ID = "legacy-model-registry-current"

#: Tier-4 feature-model roles — the ones with a monthly serving fold under
#: ``data/models/tier4/``. Matches ``engine.data.features.tier4.FEATURE_MODELS``
#: (checked, not duplicated, by the module's own test).
FEATURE_ROLES = ("size", "implied_t1", "runup_move", "iv_crush")

_TIER4_DIR = paths.DATA / "models" / "tier4"
_FOLD_NAME = re.compile(r"^(?P<model_id>.+)_(?P<yyyymm>\d{6})_(?P<snapshot>[0-9a-f]+)\.joblib$")


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(paths.ROOT))
    except ValueError:
        return str(path)


def _hash_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return _hash_bytes(path.read_bytes())


def _required_kinds(role: str) -> tuple[str, ...]:
    """Which member kinds a champion of this role must carry today.

    ``transform`` is deliberately absent from every role: no current artifact
    separates a preprocessing/target transform from its estimator (a
    ``LogTargetRegressor`` or ``BlendModel`` embeds it in the pickled Python
    class instead — see the P5-1 guide addendum). Requiring it here would make
    every real release permanently, unfixably incomplete instead of naming a
    real P5-3 gap once, in prose. ``residual_bucket`` is likewise not required:
    :class:`engine.models.registry.ModelArtifact` falls back to its flat pool
    by design when a bucket is absent, so a missing bucket is a refinement
    forgone, not a broken release.
    """
    if role in FEATURE_ROLES:
        return ("estimator", "residual")
    if role == "gate":
        return ("estimator", "threshold")
    return ("estimator",)  # chooser


@dataclass(frozen=True)
class _MemberBuild:
    members: tuple[ArtifactInventoryMember, ...]
    issues: tuple[ReleaseIssue, ...]


def _champion_members(entry: "legacy_registry.RegistryEntry") -> _MemberBuild:
    """Real, hash-verified members for one champion — present/missing/incompatible.

    Never loads the estimator to predict anything; ``load_artifact`` only
    unpickles the :class:`ModelArtifact` wrapper to read its metadata (feature
    list, residual sizes), the same "single joblib file, a few MB" access the
    legacy loader itself performs.
    """
    path = entry.path
    issues: list[ReleaseIssue] = []
    members: list[ArtifactInventoryMember] = []
    file_hash = _hash_file(path)
    if file_hash is None:
        issues.append(ReleaseIssue(
            path=f"$.artifacts[{entry.id}]", code="MISSING_ARTIFACT_FILE", detail=_relative(path),
        ))
        return _MemberBuild((), tuple(issues))
    if entry.artifact_sha256 and file_hash[len("sha256:"):] != entry.artifact_sha256:
        issues.append(ReleaseIssue(
            path=f"$.artifacts[{entry.id}]", code="ARTIFACT_HASH_DRIFT",
            detail=f"registry={entry.artifact_sha256[:12]} file={file_hash[len('sha256:'):][:12]}",
        ))
    members.append(ArtifactInventoryMember(
        member_id=f"{entry.id}:estimator", kind="estimator",
        artifact_ref=_relative(path), content_hash=file_hash,
    ))
    try:
        artifact = legacy_registry.load_artifact(path)
    except Exception as exc:  # a corrupt or wrong-type pickle is INCOMPATIBLE, not absent
        issues.append(ReleaseIssue(
            path=f"$.artifacts[{entry.id}]", code="ARTIFACT_UNREADABLE", detail=type(exc).__name__,
        ))
        return _MemberBuild(tuple(members), tuple(issues))
    if list(artifact.features) != list(entry.features):
        issues.append(ReleaseIssue(
            path=f"$.artifacts[{entry.id}].ordered_features", code="FEATURE_LIST_DRIFT",
            detail="registry.json features disagree with the artifact's own",
        ))
    if artifact.residuals.size > 0:
        members.append(ArtifactInventoryMember(
            member_id=f"{entry.id}:residual", kind="residual",
            artifact_ref=_relative(path), content_hash=file_hash,
        ))
    if artifact.residual_buckets:
        members.append(ArtifactInventoryMember(
            member_id=f"{entry.id}:residual_bucket", kind="residual_bucket",
            artifact_ref=_relative(path), content_hash=file_hash,
        ))
    if entry.role == "gate" and entry.threshold is not None:
        manifest_hash = _hash_file(legacy_registry.REGISTRY_PATH)
        if manifest_hash is not None:
            members.append(ArtifactInventoryMember(
                member_id=f"{entry.id}:threshold", kind="threshold",
                artifact_ref=_relative(legacy_registry.REGISTRY_PATH), content_hash=manifest_hash,
            ))
    return _MemberBuild(tuple(members), tuple(issues))


def _upstream_ids(reg: "legacy_registry.Registry", entry: "legacy_registry.RegistryEntry") -> tuple[str, ...]:
    ids: set[str] = set()
    for column in entry.consumes:
        for producer in reg.producers(column):
            ids.add(producer.id)
    return tuple(sorted(ids))


def _evidence_refs(entry: "legacy_registry.RegistryEntry") -> tuple[str, ...]:
    refs = []
    oos = legacy_registry.ARTIFACT_DIR / f"{entry.id}_oos_predictions.parquet"
    if oos.is_file():
        refs.append(_relative(oos))
    evidence = paths.FEATURES / "model_evidence.json"
    if evidence.is_file():
        refs.append(_relative(evidence))
    return tuple(refs)


def current_release_inventory(
    reg: "legacy_registry.Registry | None" = None,
) -> tuple[ModelReleaseInventory, tuple[ReleaseIssue, ...]]:
    """The current release, plus real-file drift issues ``release_issues`` cannot see.

    ``release_issues`` (this package's existing, frozen checker) validates a
    release's INTERNAL consistency without touching disk — that is its
    documented contract. This function is the layer that DOES read disk: it
    verifies every member's hash against the real file and every artifact's
    feature list against the real pickle, and returns what it found as
    :class:`~.releases.ReleaseIssue` tuples using the same shape, on top of
    (never instead of) the deterministic checks ``release_issues`` still runs
    on the object this returns.
    """
    reg = reg if reg is not None else legacy_registry.load_registry()
    artifacts: list[ModelArtifactInventory] = []
    bindings: list[ReleaseBinding] = []
    requirements: list[ReleaseRequirement] = []
    drift_issues: list[ReleaseIssue] = []

    for entry in sorted((e for e in reg.entries if e.champion), key=lambda e: e.id):
        built = _champion_members(entry)
        drift_issues.extend(built.issues)
        artifacts.append(ModelArtifactInventory(
            artifact_id=entry.id,
            role=entry.role,
            strategy_ids=(entry.strategy,),
            compatible_clock_ids=KNOWN_CLOCK_IDS,
            target_contract_ref=entry.target,
            ordered_features=tuple(entry.features),
            members=built.members,
            upstream_artifact_ids=_upstream_ids(reg, entry),
            evidence_refs=_evidence_refs(entry),
        ))
        bindings.append(ReleaseBinding(
            role=entry.role,
            strategy_id=entry.strategy,
            clock_id=KNOWN_CLOCK_IDS[0],
            artifact_id=entry.id,
            ordered_features=tuple(entry.features),
            required_member_kinds=_required_kinds(entry.role),
        ))
        requirements.append(ReleaseRequirement(
            role=entry.role, strategy_id=entry.strategy, clock_id=KNOWN_CLOCK_IDS[0],
        ))

    manifest_hash = _hash_file(legacy_registry.REGISTRY_PATH)
    evidence = paths.FEATURES / "model_evidence.json"
    release = ModelReleaseInventory(
        release_id=RELEASE_ID,
        deployment_id=DEPLOYMENT_ID,
        known_clock_ids=KNOWN_CLOCK_IDS,
        artifacts=tuple(artifacts),
        bindings=tuple(bindings),
        requirements=tuple(requirements),
        artifact_manifest_ref=_relative(legacy_registry.REGISTRY_PATH),
        evidence_refs=(_relative(evidence),) if evidence.is_file() else ("MISSING",),
    )
    if manifest_hash is None:
        drift_issues.append(ReleaseIssue(
            path="$.artifact_manifest_ref", code="MISSING_ARTIFACT_FILE",
            detail=_relative(legacy_registry.REGISTRY_PATH),
        ))
    return release, tuple(sorted(drift_issues))


def registry_drift_issues(reg: "legacy_registry.Registry | None" = None) -> tuple[ReleaseIssue, ...]:
    """Convenience: just the real-file drift issues, without the release object."""
    return current_release_inventory(reg)[1]


# --------------------------------------------------------------------------
# Tier-4 monthly fold coverage — reported, not bound (see module docstring)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldCoverageEntry:
    model_id: str
    fold_start: str  # YYYYMM
    snapshot12: str
    content_hash: str
    size_bytes: int
    status: str  # "present" or "incompatible"
    reason: str | None = None


def tier4_fold_coverage(
    model_id: str | None = None, *, directory: Path | None = None,
) -> tuple[FoldCoverageEntry, ...]:
    """Every monthly serving-fold file on disk for ``model_id`` (or all four roles).

    Reads each fold file exactly once, one at a time (files here run under
    2 MB — see ``ls -la data/models/tier4``), to confirm it actually unpickles
    to the ``{estimator, model_id, fold_start, tier3_snapshot, features,
    pool_pred, pool_res}`` shape ``engine.data.features.tier4.serving_model``
    writes. A file that fails to load is reported ``incompatible``, never
    silently skipped. ``directory`` is a test seam; production callers omit it.
    """
    root = directory if directory is not None else _TIER4_DIR
    if not root.is_dir():
        return ()
    entries = []
    for path in sorted(root.glob("*.joblib")):
        match = _FOLD_NAME.match(path.name)
        if match is None:
            continue
        if model_id is not None and match.group("model_id") != model_id:
            continue
        content_hash = _hash_file(path)
        size = path.stat().st_size
        status, reason = "present", None
        try:
            import joblib

            stored = joblib.load(path)
            missing_keys = sorted(
                {"estimator", "model_id", "fold_start", "tier3_snapshot", "features"}
                - set(stored)
            )
            if missing_keys:
                status, reason = "incompatible", f"missing keys: {missing_keys}"
            elif "pool_pred" not in stored or "pool_res" not in stored:
                status, reason = "incompatible", "no embedded pool_pred/pool_res"
        except Exception as exc:  # pragma: no cover - defensive, exercised by corruption tests
            status, reason = "incompatible", type(exc).__name__
        entries.append(FoldCoverageEntry(
            model_id=match.group("model_id"),
            fold_start=match.group("yyyymm"),
            snapshot12=match.group("snapshot")[:12],
            content_hash=content_hash or "sha256:missing",
            size_bytes=size,
            status=status,
            reason=reason,
        ))
    return tuple(entries)


# --------------------------------------------------------------------------
# Non-model serving state: no registry entry, no content hash, today
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NonModelStateEntry:
    name: str
    kind: str  # release MemberKind this state most resembles
    strategies: tuple[str, ...]
    backing_file: str | None
    content_hash: str | None
    status: str  # "not_persisted" | "present_unversioned" | "source_present"
    detail: str


#: One row per serving-time state the legacy scorer builds live rather than
#: loading from a registered, hash-checked artifact. See module docstring for
#: why these are reported here instead of forced into a fake ModelRole.
NON_MODEL_STATE_ITEMS: tuple[tuple[str, str, tuple[str, ...], str | None, str], ...] = (
    ("payoff_line:STR-THRU", "calibration", ("STR-THRU",), None,
     "engine.payoff.fit_payoff — refit per (strategy, alpha, cutoff) from the trades ledger every Scorer"),
    ("payoff_line:STR-RUNUP", "calibration", ("STR-RUNUP",), None,
     "engine.payoff.fit_payoff — same mechanism, driver=im_t1"),
    ("payoff_surface:STR-RUNUP", "calibration", ("STR-RUNUP",), None,
     "engine.payoff.fit_runup_payoff — EXP-149 two-driver surface, refit live"),
    ("recalibration_map", "calibration", ("STR-THRU", "STR-RUNUP"),
     "data/features/recalibration_pairs.parquet",
     "engine.recalibrate.fit_recalibration — the (raw_win, outcome) pairs are persisted; "
     "the fitted isotonic map itself is not, and is refit per (strategy, alpha, cutoff)"),
    ("paired_residual_pool", "paired_simulation", ("*",),
     "data/features/tier4_forecasts.parquet",
     "engine.pnl_sim.ResidualPool via Scorer._residual_pool — built from the stored Tier-4 "
     "forecasts joined against the live panel and a live-recomputed crush table; not itself "
     "persisted, and documented as scorer-context-dependent (AGENTS.md: residual pool depends "
     "on loaded tickers)"),
    ("trailing_pnl_cutoff", "threshold", ("*",),
     "data/features/pnl_sim_history.parquet",
     "engine.pnl_sim.trailing_cutoff — a trailing-quantile bar recomputed per as_of date "
     "from the stored history; the bar itself is never written back"),
    ("chooser_analog_pool", "residual", ("DYN-SV",),
     "data/features/chooser_analog_pool.parquet",
     "Scorer._chooser_analog_pool — the k-NN population dyn_sv_chooser_v1_1's 5 analog "
     "features are drawn from; persisted, but bound to no release or registry entry"),
    ("board_analog_matcher", "residual", ("*",), None,
     "engine.analogs.AnalogMatcher — the bucket-analog population every board row's "
     "analog_mean/win_analog reads; rebuilt from the trades ledger every Scorer construction"),
)


def non_model_state_inventory(*, root: Path | None = None) -> tuple[NonModelStateEntry, ...]:
    """Present/missing status for the six serving-time states with no artifact.

    ``present_unversioned`` means the backing file exists but nothing checks
    its hash before use; ``source_present`` means only an UPSTREAM input is
    persisted and the state itself is refit live; ``not_persisted`` means
    there is no file at all — the state exists only inside a live Scorer.
    ``root`` is a test seam; production callers omit it.
    """
    base = root if root is not None else paths.ROOT
    entries = []
    for name, kind, strategies, rel_path, detail in NON_MODEL_STATE_ITEMS:
        if rel_path is None:
            entries.append(NonModelStateEntry(
                name=name, kind=kind, strategies=strategies,
                backing_file=None, content_hash=None, status="not_persisted", detail=detail,
            ))
            continue
        path = base / rel_path
        content_hash = _hash_file(path)
        if content_hash is None:
            status = "missing_source"
        elif name in ("chooser_analog_pool",):
            status = "present_unversioned"
        else:
            status = "source_present"
        entries.append(NonModelStateEntry(
            name=name, kind=kind, strategies=strategies,
            backing_file=rel_path, content_hash=content_hash, status=status, detail=detail,
        ))
    return tuple(entries)


# --------------------------------------------------------------------------
# Served-role coverage, derived from the scorer's own source
# --------------------------------------------------------------------------


def served_roles(score_source: str | None = None) -> tuple[str, ...]:
    """Every :data:`engine.models.registry.ROLES` value that appears in ``engine/score.py``.

    Derived from the scorer's OWN source text rather than hand-typed: a role
    string can reach the registry through more than one call shape
    (``self.model("gate", ...)`` for gate/chooser/size/runup_move/implied_t1,
    but ``iv_crush`` only ever resolves its champion inside
    ``engine.data.features.tier4.iv_crush_feature_model`` — never through
    ``Scorer.model``). Checking for the literal role string anywhere in the
    module catches both call shapes without hard-coding which function does
    the resolving; a role's disappearance from the source (dead code) makes it
    disappear from this list too, which is what a code-derived list is for.
    """
    if score_source is None:
        score_source = (paths.ENGINE / "score.py").read_text()
    return tuple(role for role in legacy_registry.ROLES if f'"{role}"' in score_source)
