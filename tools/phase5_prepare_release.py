#!/usr/bin/env python3
"""P5-6: assemble a staged model release from real artifacts.

Reads (never writes) the champion registry and ``data/models/*.joblib``, the
Tier-4 serving folds under ``data/models/tier4``, P5-3 training-job outputs
(``payoff_artifact.json`` and ``recalibration_artifact.json`` per calibration
fold, under ``--training-root``) and the frozen states the training job's
``--state`` writes (driver and paired residual pools, via ``--frozen-state``).
The admissible-depth table is built here from ``legacy_n_admissible_table()``,
and the frozen chooser analog pool from ``data/features/chooser_analog_pool.parquet``.
Which calibration folds to train: ``tools/phase5_calibration_keys.py``. Writes one release root (layout: ``engine/v2/models/release_layout.py``) under
``--out``, which may not be inside ``data/``:

* the seven champion bindings are staged through
  ``engine.v2.models.deployment.stage_release``, so its completeness and
  feature-order checks run before anything is written;
* every catalog state is written content-addressed next to them and listed in
  ``phase5_release.json`` as STAGED, MISSING (the artifact type exists but no
  member was produced) or PENDING (the artifact type has not landed here).

``--incumbent`` copies an existing deployment store (manifests, objects,
``DEPLOYED``, history) in first, so the acceptance gate can run its
promote -> rollback round trip against the release that is live today.

Prints ids, counts, statuses and byte sizes only. Heavy only through
``--tier3-snapshot auto`` (streams a hash of ``panel.parquet``); every
real-data run goes under ``tools/bounded_run.py``, run by the supervisor.

Usage::

    INVESTING_PLAN_ROOT=/root/investing-plan python3 tools/phase5_prepare_release.py \\
        --release-id p5-6-2026-09-18a --out /root/p5-6/release-A \\
        --training-root /root/p5-3-runs --plan-only
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from checks.phase5_modules import modules_available  # noqa: E402
from engine.v2.foundation import CONTENT_HASH_PREFIX, to_document  # noqa: E402
from engine.v2.models import deployment  # noqa: E402
from engine.v2.models.contracts import ArtifactMember, ModelBinding, ModelRelease  # noqa: E402
from engine.v2.models.frozen_state_io import BoundedStateReadError, read_bounded_state  # noqa: E402
from engine.v2.models.pinned_panel import (  # noqa: E402
    COPY_PREFIX,
    PinnedPanelContext,
    verify_panel_copy,
)
from engine.v2.models.release_layout import (  # noqa: E402
    MANIFEST_NAME,
    MISSING,
    PENDING,
    STAGED,
    STATE_SPECS,
    CatalogConflict,
    ObjectVerificationError,
    StateSpec,
    deployment_root,
    manifest_body,
    member_row,
    object_relpath,
    preserve_legacy_manifest,
    sha256_bytes,
    validate_legacy_manifest,
    write_manifest,
    write_object,
)
from engine.v2.models.releases import ModelReleaseInventory  # noqa: E402
from engine.v2.models.serving_fold_preflight import (  # noqa: E402
    MAX_FOLD_BYTES,
    descriptor_from_preflight,
    preflight_chooser_fold,
    preflight_size_fold,
)
from engine.v2.models.serving_folds import (  # noqa: E402
    ChooserFoldPolicy,
    SizeFoldPolicy,
    chooser_fold_outputs,
)

# The chooser pool source admission lives in the runtime training package (no
# ``checks`` imports) so ``tools/phase5_training_job.py`` can read it directly;
# these names stay importable from here for existing callers.
from engine.v2.models.training.chooser_pool_source import (  # noqa: E402
    CHOOSER_POOL_ID,
    read_chooser_pool_frame,
    require_chooser_pool_columns,
)

MAX_MODEL_BYTES = 512 * 1024 * 1024
MAX_STATE_BYTES = 512 * 1024 * 1024

JOBLIB_ADAPTER = "joblib-estimator.v1"
#: Output name per role for the frozen binding, following the Phase 4 native
#: field names where one exists. ``runup_move`` is the one role whose native
#: field (``runup_move_prediction``) names a FINAL, day-scaled value while
#: the joblib artifact itself (``LogTargetRegressor``) only ever returns the
#: raw T-14 magnitude -- ``engine.v2.scoring.application`` scales it by
#: ``days_before_print / 14`` exactly once, at scoring time, and refuses
#: (``PRETRANSFORMED_FROZEN_OUTPUT:runup_move``) any binding that instead
#: claims to already be final. This must stay a raw name from
#: ``engine.v2.scoring.application._RUNUP_RAW_NAMES``; see
#: ``engine.score._RUNUP_MODEL_OUTPUTS`` for the matching Phase 4 capture
#: fix (2026-09-21).
OUTPUT_NAMES = {
    "size": ("forecast_abs_move",), "implied_t1": ("driver_prediction",),
    "runup_move": ("pred_runup_abs_move_d14",), "iv_crush": ("pred_iv_crush",),
    "gate": ("gate_score",), "chooser": ("chooser_score",),
}
_FOLD = re.compile(r"^(?P<model>.+)_(?P<month>\d{6})_(?P<snap>[0-9a-f]{12})\.joblib$")


class PrepareRefused(RuntimeError):
    """The inputs cannot form a staged release."""


@dataclasses.dataclass
class StateBuild:
    spec: StateSpec
    status: str
    payloads: dict[str, bytes]  # object name -> bytes
    detail: str
    size_fold_policy: SizeFoldPolicy | None = None
    chooser_fold_policy: ChooserFoldPolicy | None = None


# --------------------------------------------------------------------------
# model bindings
# --------------------------------------------------------------------------


def _bounded_model_read(path: Path) -> bytes:
    try:
        return read_bounded_state(path, MAX_MODEL_BYTES, no_follow=True)
    except (BoundedStateReadError, OSError) as exc:
        raise PrepareRefused("model artifact cannot be read") from exc


def model_release(inventory: ModelReleaseInventory, *, read=_bounded_model_read,
                  resolve=lambda ref: Path(ref)) -> tuple[ModelRelease, dict[str, bytes]]:
    """One joblib binding per inventory binding, members named by required kind.

    ``residual``/``residual_bucket`` live inside the estimator pickle and the
    gate threshold inside ``registry.json`` (P5-1), so those members share
    the bytes of their inventory ``artifact_ref``.
    """
    artifacts = {item.artifact_id: item for item in inventory.artifacts}
    bindings, payloads = [], {}
    for binding in inventory.bindings:
        artifact = artifacts[binding.artifact_id]
        members = []
        for inv_member in artifact.members:
            if inv_member.kind not in (*binding.required_member_kinds, "estimator"):
                continue
            data = read(resolve(inv_member.artifact_ref))
            if sha256_bytes(data) != inv_member.content_hash:
                raise PrepareRefused(f"{inv_member.member_id}: file changed since inventory")
            payloads[inv_member.content_hash] = data
            members.append(ArtifactMember(name=inv_member.kind, path=inv_member.artifact_ref,
                                          content_hash=inv_member.content_hash))
        bindings.append(ModelBinding(
            binding_id=f"{binding.role}:{binding.strategy_id}", model_id=binding.artifact_id,
            role=binding.role, strategy_id=binding.strategy_id,
            decision_clock_id=binding.clock_id, adapter=JOBLIB_ADAPTER,
            feature_order=binding.ordered_features,
            output_names=OUTPUT_NAMES.get(binding.role, (f"{binding.role}_output",)),
            members=tuple(members),
        ))
    release = ModelRelease(release_id=inventory.release_id,
                           deployment_id=inventory.deployment_id, bindings=tuple(bindings))
    return release, payloads


# --------------------------------------------------------------------------
# frozen states
# --------------------------------------------------------------------------


def payoff_member_id(artifact) -> str:
    from engine.v2.models.payoff_artifact import PayoffLineArtifact

    kind = "payoff_line" if isinstance(artifact, PayoffLineArtifact) else "payoff_surface"
    return f"{kind}:{artifact.strategy}"


def payoff_payloads(training_roots: Iterable[Path]) -> dict[str, dict[str, bytes]]:
    """``payoff_artifact.json`` files from P5-3 job outputs, verified, by member."""
    from engine.v2.models.payoff_artifact import PayoffArtifactLoader, PayoffArtifactRef
    from engine.v2.models.training.calibration import PAYOFF_ARTIFACT_FILE

    found: dict[str, dict[str, bytes]] = {}
    for base in training_roots:
        for path in sorted(Path(base).rglob(PAYOFF_ARTIFACT_FILE)):
            data = read_bounded_state(path, MAX_STATE_BYTES, no_follow=True)
            artifact = PayoffArtifactLoader(path.parent).load_bytes(
                data, sha256_bytes(data))
            name = f"{artifact.strategy}|{artifact.alpha}|{artifact.cutoff}"
            found.setdefault(payoff_member_id(artifact), {})[name] = data
    return found


def recalibration_payloads(training_roots: Iterable[Path]) -> dict[str, dict[str, bytes]]:
    """``recalibration_artifact.json`` files from P5-3 job outputs, verified."""
    from engine.v2.models.recalibration_artifact import (
        RecalibrationArtifactLoader,
        RecalibrationArtifactRef,
    )
    from engine.v2.models.training.calibration import RECALIBRATION_ARTIFACT_FILE

    found: dict[str, dict[str, bytes]] = {}
    for base in training_roots:
        for path in sorted(Path(base).rglob(RECALIBRATION_ARTIFACT_FILE)):
            data = read_bounded_state(path, MAX_STATE_BYTES, no_follow=True)
            artifact = RecalibrationArtifactLoader(path.parent).load_bytes(
                data, sha256_bytes(data))
            name = f"{artifact.strategy}|{artifact.alpha}|{artifact.cutoff}"
            found.setdefault(f"recalibration_map:{artifact.strategy}", {})[name] = data
    return found


def artifact_file_payloads(files: Iterable[Path | tuple[Path, bytes]]) -> dict[str, dict[str, bytes]]:
    """Load payoff and recalibration artifact files, classified by JSON ``schema_version``."""
    from engine.v2.models.payoff_artifact import (
        PAYOFF_LINE_ARTIFACT_V1,
        PAYOFF_SURFACE_ARTIFACT_V1,
        PayoffArtifactLoader,
        PayoffArtifactRef,
    )
    from engine.v2.models.recalibration_artifact import (
        RECALIBRATION_MAP_ARTIFACT_V1,
        RecalibrationArtifactLoader,
        RecalibrationArtifactRef,
    )
    from tools.phase5_state_stage_job import StageRefused

    found: dict[str, dict[str, bytes]] = {}
    for item in files:
        if isinstance(item, tuple):
            path, data = item
        else:
            path = Path(item)
            try:
                data = read_bounded_state(path, MAX_STATE_BYTES, no_follow=True)
            except BoundedStateReadError as exc:
                raise StageRefused("a bound state file could not be loaded") from exc
        try:
            document = json.loads(data)
        except (ValueError, TypeError) as exc:
            raise StageRefused("a bound state file could not be loaded") from exc
        schema = document.get("schema_version") if isinstance(document, dict) else None
        if not isinstance(schema, str):
            raise StageRefused("a bound state file could not be loaded")
        if schema in (PAYOFF_LINE_ARTIFACT_V1, PAYOFF_SURFACE_ARTIFACT_V1):
            artifact = PayoffArtifactLoader(path.parent).load_bytes(
                data, sha256_bytes(data))
            name = f"{artifact.strategy}|{artifact.alpha}|{artifact.cutoff}"
            member = payoff_member_id(artifact)
            if name in found.setdefault(member, {}):
                raise StageRefused("a state object appears twice", {"member": member})
            found[member][name] = data
        elif schema == RECALIBRATION_MAP_ARTIFACT_V1:
            artifact = RecalibrationArtifactLoader(path.parent).load_bytes(
                data, sha256_bytes(data))
            name = f"{artifact.strategy}|{artifact.alpha}|{artifact.cutoff}"
            member = f"recalibration_map:{artifact.strategy}"
            if name in found.setdefault(member, {}):
                raise StageRefused("a state object appears twice", {"member": member})
            found[member][name] = data
        else:
            raise StageRefused("a bound state file could not be loaded")
    return found


#: ``tools/phase5_training_job.py --state`` writes a ``<member>.summary.json``
#: (counts, hash, status) beside each state file; it is not a state.
SUMMARY_SUFFIX = ".summary.json"


def frozen_state_files(paths: Iterable[Path]) -> list[Path]:
    """Expand ``--frozen-state`` arguments: a file as given, a directory to its
    ``*.json`` state files. ``*.summary.json`` is skipped either way."""
    files: list[Path] = []
    for path in map(Path, paths):
        candidates = sorted(path.glob("*.json")) if path.is_dir() else [path]
        files += [item for item in candidates if not item.name.endswith(SUMMARY_SUFFIX)]
    return files


def frozen_state_payloads(files: Iterable[Path],
                          max_bytes: int | None = MAX_STATE_BYTES) -> dict[str, dict[str, bytes]]:
    """Pre-built P5-4 frozen states (residual pools, tables), by member.

    The training job's ``--state`` outputs
    (``driver_residual_pool__<role>.json``, ``paired_residual_pool.json``,
    ``board_analog_matcher__<strategy>__<alpha>__<cutoff>.json``,
    ``trailing_pnl_cutoff__<month>.json``) are
    classified by their own schema and role, not by file name.
    ``max_bytes`` bounds each read and the loader; only the first read is no-follow,
    the loader re-reads the resolved path and its bytes must hash to that first read.
    """
    from engine.v2.models.frozen_state import FrozenStateLoader, FrozenStateRef
    from engine.v2.models.frozen_state_io import read_bounded_state

    limit = MAX_STATE_BYTES if max_bytes is None else max_bytes
    found: dict[str, dict[str, bytes]] = {}
    for path in frozen_state_files(files):
        data = read_bounded_state(path, limit, no_follow=True)
        state = FrozenStateLoader(path.parent, max_bytes=limit).load(
            FrozenStateRef(path=path.name, content_hash=sha256_bytes(data)))
        schema = state.schema_version
        if schema.startswith("driver_residual_pool"):
            member, name = f"driver_residual_pool:{state.role}", f"{state.model_id}|{state.fold}"
        elif schema.startswith("paired_residual_pool"):
            member = "paired_residual_pool"
            name = f"{state.move_model_id}|{state.crush_model_id}|{state.cutoff}"
        elif schema.startswith("board_analog_pool"):
            member = "board_analog_matcher"
            name = f"{state.strategy}|{state.alpha:.4f}|{state.cutoff}"
        elif schema.startswith("chooser_analog_pool"):
            member, name = "chooser_analog_pool", f"{state.pool_id}|{state.cutoff}"
        elif schema.startswith("trailing_pnl_cutoff"):
            member, name = "trailing_pnl_cutoff", f"{state.history_id}|{state.month}"
        else:
            member, name = "admissible_table:dyn_sv", f"{state.table_id}|{state.version}"
        found.setdefault(member, {})[name] = data
    return found


def default_admissible_table() -> dict[str, dict[str, bytes]]:
    from engine.v2.models.admissible_table import legacy_n_admissible_table
    from engine.v2.models.frozen_state import serialize_frozen_state

    table = legacy_n_admissible_table()
    return {"admissible_table:dyn_sv": {
        f"{table.table_id}|{table.version}": serialize_frozen_state(table)}}


def chooser_pool_payloads(path: Path) -> dict[str, dict[str, bytes]]:
    """The frozen chooser analog pool built from ``tools/build_chooser_pool.py``'s
    parquet, in its stored order. The pool is static, so its causal cutoff is
    the day after its last exit: every row it holds is inside the key.
    A plainly missing source is ``{}``; a symlink, non-regular or oversized
    source refuses through ``read_chooser_pool_frame``."""
    from engine.v2.models.frozen_state import serialize_frozen_state
    from engine.v2.models.lineage import DataDependency, Lineage
    from engine.v2.models.training.chooser_pool import build_chooser_analog_pool_artifact

    try:
        frame = read_chooser_pool_frame(path)
    except FileNotFoundError:
        return {}
    require_chooser_pool_columns(frame.columns)
    exits = pd.to_datetime(frame["exit_date"]).dropna()
    if exits.empty:
        return {}
    cutoff = str((exits.max().normalize() + pd.Timedelta(days=1)).date())
    pool = build_chooser_analog_pool_artifact(
        frame.to_dict("records"), pool_id=CHOOSER_POOL_ID, cutoff=cutoff,
        lineage=Lineage(data=(DataDependency(table=CHOOSER_POOL_ID, end_exclusive=cutoff),)))
    return {"chooser_analog_pool": {f"{pool.pool_id}|{pool.cutoff}":
                                    serialize_frozen_state(pool)}}


def tier4_fold_payloads(tier4_dir: Path, model_ids: Mapping[str, str], snapshot: str | None,
                        month: str | None) -> dict[str, dict[str, bytes]]:
    """Serving folds for the current Tier-3 snapshot (the only servable ones)."""
    found: dict[str, dict[str, bytes]] = {}
    if not Path(tier4_dir).is_dir():
        return found
    by_model = {model_id: role for role, model_id in model_ids.items()}
    for path in sorted(Path(tier4_dir).glob("*.joblib")):
        match = _FOLD.match(path.name)
        if not match or match["model"] not in by_model:
            continue
        if snapshot and match["snap"] != snapshot[:12]:
            continue
        if month and match["month"] != month:
            continue
        found.setdefault(f"tier4_folds:{by_model[match['model']]}", {})[path.name] = (
            _read_fold_payload(path))
    return found


def _read_fold_payload(path: Path) -> bytes:
    try:
        return read_bounded_state(path, MAX_FOLD_BYTES, no_follow=True)
    except BoundedStateReadError as exc:
        message = ("serving fold is not a regular file" if exc.code == "NOT_REGULAR_FILE"
                   else "serving fold exceeds byte limit")
        raise PrepareRefused(message) from None
    except OSError as exc:
        raise PrepareRefused("serving fold cannot be read") from exc


def build_states(available_payloads: Mapping[str, Mapping[str, bytes]],
                 notes: Mapping[str, str] | None = None, *,
                 size_fold_policy: SizeFoldPolicy | None = None,
                 chooser_fold_policies: Mapping[str, ChooserFoldPolicy] | None = None) -> list[StateBuild]:
    """One build row per catalog state: STAGED, MISSING or PENDING, never skipped."""
    notes = notes or {}
    chooser_fold_policies = dict(chooser_fold_policies or {})
    for role, policy in chooser_fold_policies.items():
        try:
            chooser_fold_outputs(role)
        except ValueError as exc:
            raise PrepareRefused("unsupported chooser fold role") from exc
        if not isinstance(policy, ChooserFoldPolicy):
            raise PrepareRefused("invalid chooser fold policy type")
        if policy.role != role:
            raise PrepareRefused("chooser fold policy role disagrees with member")
    builds = []
    for spec in STATE_SPECS:
        payloads = dict(available_payloads.get(spec.member_id, {}))
        ok, missing = modules_available(spec.modules)
        if payloads:
            builds.append(StateBuild(spec, STAGED, payloads, spec.source))
        elif not ok:
            builds.append(StateBuild(spec, PENDING, {}, f"{missing} not importable"))
        else:
            builds.append(StateBuild(
                spec, MISSING, {}, notes.get(spec.member_id, f"no member from {spec.source}")))
    for build in builds:
        if build.spec.member_id == "tier4_folds:size":
            build.size_fold_policy = size_fold_policy
        for role, policy in chooser_fold_policies.items():
            if build.spec.member_id == f"tier4_folds:{role}":
                build.chooser_fold_policy = policy
    return builds


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def _copy_incumbent(source: Path, dest: Path) -> None:
    if (dest / "releases").exists():
        raise PrepareRefused(f"{dest} already holds a deployment store")
    legacy = Path(source).parent / MANIFEST_NAME
    validate_legacy_manifest(Path(source), legacy)
    for name in ("releases", "objects", "history"):
        if (Path(source) / name).is_dir():
            shutil.copytree(Path(source) / name, dest / name)
    if (Path(source) / "DEPLOYED").is_file():
        shutil.copy2(Path(source) / "DEPLOYED", dest / "DEPLOYED")
    preserve_legacy_manifest(dest.parent, legacy)


def _pinned_preflight(release: ModelRelease, states: list[StateBuild],
                      pinned_panel: PinnedPanelContext) -> tuple[dict[str, str], dict]:
    identity, validated = verify_panel_copy(pinned_panel), {}
    for build in (b for b in states if b.spec.member_id == "tier4_folds:size"):
        policy, payloads = build.size_fold_policy, build.payloads
        if policy is None and payloads:
            raise PrepareRefused(f"{COPY_PREFIX}refused: size folds carry no pinned policy")
        if policy is not None and policy.panel_sha256 != identity["panel_copy_sha256"]:
            raise PrepareRefused(f"{COPY_PREFIX}refused: size policy does not match the pinned panel")
        for name, data in sorted(payloads.items() if policy else ()):
            if not isinstance(data, bytes):
                raise PrepareRefused(f"{COPY_PREFIX}refused: size fold is not bytes")
            if name in validated:
                raise PrepareRefused(f"{COPY_PREFIX}refused: duplicate size fold filename")
            try:
                validated[name] = preflight_size_fold(data, name, policy, release)
            except ValueError as exc:
                raise PrepareRefused(f"{COPY_PREFIX}refused: {exc}") from exc
    return identity, validated


def _destination_digest(path: Path) -> str:
    return hashlib.sha256(_read_fold_payload(path)).hexdigest()


def _size_preflight(release, states):
    validated = {}
    for build in states:
        if build.spec.member_id != "tier4_folds:size" or build.size_fold_policy is None:
            continue
        for name, data in sorted(build.payloads.items()):
            if name in validated or not isinstance(data, bytes):
                raise PrepareRefused("duplicate or non-byte size fold object")
            try:
                validated[name] = preflight_size_fold(data, name, build.size_fold_policy, release)
            except ValueError as exc:
                raise PrepareRefused(f"size fold refused: {exc}") from exc
    return validated


def _chooser_preflight(release, states, panel_identity=None):
    validated = {}
    for build in states:
        policy = build.chooser_fold_policy
        if policy is None:
            continue
        if not isinstance(policy, ChooserFoldPolicy):
            raise PrepareRefused("invalid chooser fold policy type")
        if build.spec.member_id != f"tier4_folds:{policy.role}":
            raise PrepareRefused("chooser fold policy role disagrees with member")
        if panel_identity is not None and policy.panel_sha256 != panel_identity["panel_copy_sha256"]:
            raise PrepareRefused("chooser fold policy does not match the pinned panel")
        for name, data in sorted(build.payloads.items()):
            key = build.spec.member_id, name
            if key in validated or not isinstance(data, bytes):
                raise PrepareRefused("duplicate or non-byte chooser fold object")
            try:
                validated[key] = preflight_chooser_fold(data, name, policy, release)
            except ValueError as exc:
                raise PrepareRefused(f"chooser fold refused: {exc}") from exc
    return validated


def _rebase_member_paths(release: ModelRelease, states: list[StateBuild],
                         pinned_panel: PinnedPanelContext | None) -> tuple[ModelRelease, list[StateBuild]]:
    """Fold-carrying or pinned releases name binding members by object path and copy their payloads."""
    if pinned_panel is None and not any(build.chooser_fold_policy is not None
                                        or build.size_fold_policy is not None for build in states):
        return release, states
    states = [dataclasses.replace(build, payloads=dict(build.payloads)) for build in states]
    release = dataclasses.replace(release, bindings=tuple(
        dataclasses.replace(binding, members=tuple(dataclasses.replace(
            member, path=object_relpath(member.content_hash)) for member in binding.members))
        for binding in release.bindings))
    return release, states


def _run_preflights(release: ModelRelease, states: list[StateBuild],
                    pinned_panel: PinnedPanelContext | None, sources):
    """Size and chooser fold preflights; the pinned identity (if any) joins ``sources``."""
    if pinned_panel is not None:
        identity, validated = _pinned_preflight(release, states, pinned_panel)
        sources = {**dict(sources or {}), **identity}
    else:
        identity, validated = None, _size_preflight(release, states)
    return sources, validated, _chooser_preflight(release, states, identity)


def _store_object(out: Path, build: StateBuild, data: bytes, pinned_panel: PinnedPanelContext | None):
    try:
        return write_object(out, data)
    except ObjectVerificationError as exc:
        if build.chooser_fold_policy is None and build.size_fold_policy is None:
            raise
        if isinstance(exc.__cause__, OSError):
            raise PrepareRefused("serving fold cannot be read") from exc.__cause__
        prefix = f"{COPY_PREFIX}refused: " if pinned_panel is not None else ""
        message = ("chooser fold object bytes differ from preflight" if build.chooser_fold_policy is not None
                   else f"{prefix}size fold object bytes differ")
        raise PrepareRefused(message) from None


def _object_entry(out: Path, build: StateBuild, name: str, data: bytes, validated, chooser_validated,
                  staged, pinned_panel: PinnedPanelContext | None) -> dict:
    digest, rel = _store_object(out, build, data, pinned_panel)
    is_size = build.spec.member_id == "tier4_folds:size"
    if pinned_panel is not None and is_size:
        if _destination_digest(deployment_root(out) / rel) != validated[name].digest:
            raise PrepareRefused(f"{COPY_PREFIX}refused: size fold object bytes differ")
    entry = {"name": name, "path": rel, "content_hash": digest, "bytes": len(data)}
    if is_size and build.size_fold_policy is not None:
        descriptor = descriptor_from_preflight(validated[name], build.size_fold_policy, staged)
        entry["serving_fold"] = to_document(descriptor)
    if build.chooser_fold_policy is not None:
        preflight = chooser_validated[build.spec.member_id, name]
        if _destination_digest(deployment_root(out) / rel) != preflight.digest:
            raise PrepareRefused("chooser fold object bytes differ from preflight")
        entry["serving_fold"] = to_document(descriptor_from_preflight(
            preflight, build.chooser_fold_policy, staged))
    return entry


def _staging_id_conflict(exc) -> bool:
    """True when ``exc`` is a conflict with already-staged candidate content."""
    if isinstance(exc, CatalogConflict):
        return True
    if isinstance(exc, deployment.StagingRefused):
        codes = {issue.code for issue in exc.issues}
        return bool(codes & {"RELEASE_ID_REUSED", "MANIFEST_UNREADABLE"})
    return False


def write_release(out: Path, release: ModelRelease, inventory: ModelReleaseInventory,
                  payloads: Mapping[str, bytes], states: list[StateBuild], *,
                  sources: Mapping[str, str] | None = None,
                  incumbent: Path | None = None, pinned_panel: PinnedPanelContext | None = None,
                  candidate_only: bool = False) -> Path:
    """Stage the model bindings, write the state objects, write the manifest.

    In ``candidate_only`` mode the existing candidate's staging-success record
    is snapshotted before it is withdrawn and restored when this attempt loses
    an id conflict with that candidate (``CatalogConflict`` or a
    ``StagingRefused`` carrying ``RELEASE_ID_REUSED``/``MANIFEST_UNREADABLE``).
    Any other failure leaves the record withdrawn as before.
    """
    if sources and any(str(key).startswith(COPY_PREFIX) for key in sources):
        raise PrepareRefused(f"{COPY_PREFIX}refused: caller-supplied pinned sources")
    release, states = _rebase_member_paths(release, states, pinned_panel)
    sources, validated, chooser_validated = _run_preflights(release, states, pinned_panel, sources)
    out = Path(out)
    root = deployment_root(out)
    root.mkdir(parents=True, exist_ok=True)
    if incumbent is not None and not candidate_only:
        _copy_incumbent(incumbent, root)
    snapshot = (deployment.read_staging_success_record(root, release.release_id)
                if candidate_only else None)
    try:
        deployment.invalidate_staging_success(root, release.release_id)
        staged = deployment.stage_release(root, release, inventory, dict(payloads))
        rows = []
        for build in states:
            objects = [_object_entry(out, build, name, data, validated, chooser_validated, staged, pinned_panel)
                       for name, data in sorted(build.payloads.items())]
            rows.append(member_row(build.spec, build.status, objects, build.detail))
        body = manifest_body(release.release_id, release.deployment_id, rows, sources or {})
        return write_manifest(out, body, candidate_only=candidate_only)
    except (CatalogConflict, deployment.StagingRefused) as exc:
        if candidate_only and snapshot is not None and _staging_id_conflict(exc):
            try:
                deployment.restore_staging_success_record(root, release.release_id, snapshot)
            except (OSError, deployment.DeploymentError) as restore_exc:
                raise PrepareRefused(
                    "staging success record could not be restored") from restore_exc
        raise


def _plan(release: ModelRelease, states: list[StateBuild]) -> dict:
    return {
        "release_id": release.release_id,
        "models": [{"binding_id": b.binding_id, "members": [m.name for m in b.members]}
                   for b in release.bindings],
        "states": [{"member_id": s.spec.member_id, "status": s.status,
                    "objects": len(s.payloads),
                    "bytes": sum(len(v) for v in s.payloads.values()), "detail": s.detail}
                   for s in states],
    }


def merge_payloads(found: dict[str, dict[str, bytes]], more: Mapping[str, Mapping[str, bytes]]) -> None:
    """Add objects per member; a later source adds to, never drops, a member."""
    for member_id, objects in more.items():
        found.setdefault(member_id, {}).update(objects)


def _refuse_data_dir(out: Path) -> None:
    from engine import paths

    resolved, data = Path(out).resolve(), paths.DATA.resolve()
    if resolved == data or data in resolved.parents:
        raise PrepareRefused(f"--out may not be inside {data}")


def _pinned_context(args) -> PinnedPanelContext | None:
    values = (args.panel_catalog, args.panel_snapshot_id, args.panel_artifact_root)
    if any(value is None for value in values) and not all(value is None for value in values):
        raise PrepareRefused(f"{COPY_PREFIX}refused: pinned panel flags are all-or-none")
    return None if all(value is None for value in values) else PinnedPanelContext(*values)


def _resolve_snapshot(args, pinned_panel: PinnedPanelContext | None) -> str:
    from engine import paths

    snapshot = args.tier3_snapshot
    if pinned_panel is not None:
        digest = verify_panel_copy(pinned_panel)["panel_copy_sha256"]
        if snapshot != "auto" and snapshot.removeprefix(CONTENT_HASH_PREFIX) != digest:
            raise PrepareRefused(f"{COPY_PREFIX}refused: --tier3-snapshot does not match the pinned panel")
        return digest
    if snapshot == "auto":
        from engine.data import store

        return store.file_sha256(paths.PANEL)
    return snapshot


def _collect_payloads(args, snapshot: str, feature_ids: Mapping[str, str]) -> dict[str, dict[str, bytes]]:
    from engine import paths
    from engine.data.features import tier4

    found: dict[str, dict[str, bytes]] = {}
    merge_payloads(found, tier4_fold_payloads(Path(args.tier4_dir or tier4.SERVING_DIR),
                                     feature_ids, snapshot, args.fold_month))
    if modules_available(("engine.v2.models.payoff_artifact",))[0]:
        merge_payloads(found, payoff_payloads(args.training_root or ()))
    if modules_available(("engine.v2.models.recalibration_artifact",))[0]:
        merge_payloads(found, recalibration_payloads(args.training_root or ()))
    if modules_available(("engine.v2.models.frozen_state",))[0]:
        merge_payloads(found, default_admissible_table())
        merge_payloads(found, frozen_state_payloads(args.frozen_state or ()))
    if modules_available(("engine.v2.models.chooser_analog_pool",))[0]:
        merge_payloads(found, chooser_pool_payloads(paths.FEATURES / "chooser_analog_pool.parquet"))
    return found


def fold_policies(found: Mapping[str, Mapping[str, bytes]], snapshot: str):
    from engine.data.features import tier4

    producer = tier4.size_feature_model()
    if producer.produces != "pred_abs_move":
        raise PrepareRefused("unsupported size serving output")
    policy = SizeFoldPolicy(model_id=producer.model_id, feature_order=producer.features,
                            panel_sha256=snapshot, interval_floor=producer.interval_floor)
    chooser_policies = {}
    for role, factory in (("implied_t1", tier4.im_t1_feature_model),
                          ("runup_move", tier4.runup_move_feature_model)):
        if found.get(f"tier4_folds:{role}"):
            try:
                chooser_producer = factory()
            except tier4.Tier4Error as exc:
                raise PrepareRefused(f"chooser {role} producer disagrees with registered recipe") from exc
            if chooser_producer.produces != chooser_fold_outputs(role)[1]:
                raise PrepareRefused("unsupported chooser serving output")
            chooser_policies[role] = ChooserFoldPolicy(
                role=role, model_id=chooser_producer.model_id, feature_order=chooser_producer.features,
                panel_sha256=snapshot, interval_floor=chooser_producer.interval_floor)
    return policy, chooser_policies


def _real_inputs(args, pinned_panel: PinnedPanelContext | None = None
                 ) -> tuple[ModelRelease, ModelReleaseInventory, dict, list[StateBuild]]:
    from engine import paths
    from engine.v2.models.inventory import current_release_inventory

    inventory, drift = current_release_inventory()
    if drift:
        raise PrepareRefused("registry drift: " + ", ".join(sorted({i.code for i in drift})))
    inventory = dataclasses.replace(inventory, release_id=args.release_id)
    release, payloads = model_release(
        inventory, resolve=lambda ref: Path(ref) if Path(ref).is_absolute() else paths.ROOT / ref)
    snapshot = _resolve_snapshot(args, pinned_panel)
    feature_ids = {b.role: b.artifact_id for b in inventory.bindings if b.strategy_id == "*"}
    found = _collect_payloads(args, snapshot, feature_ids)
    policy, chooser_policies = fold_policies(found, snapshot)
    return release, inventory, payloads, build_states(
        found, size_fold_policy=policy, chooser_fold_policies=chooser_policies)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, action="append",
                        help="P5-3 training-job output dir; every payoff_artifact.json "
                             "and recalibration_artifact.json under it is collected")
    parser.add_argument("--frozen-state", type=Path, action="append",
                        help="a frozen state JSON, or a dir of them (training job --state "
                             "output; *.summary.json is ignored)")
    parser.add_argument("--tier4-dir", type=Path)
    parser.add_argument("--tier3-snapshot", default="auto",
                        help="Tier-3 snapshot hash selecting servable folds; 'auto' hashes "
                             "panel.parquet")
    parser.add_argument("--fold-month", help="restrict Tier-4 folds to one YYYYMM")
    parser.add_argument("--incumbent", type=Path,
                        help="an existing deployment store to copy in before staging")
    parser.add_argument("--panel-catalog", type=Path)
    parser.add_argument("--panel-snapshot-id")
    parser.add_argument("--panel-artifact-root", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        pinned = _pinned_context(args)
        _refuse_data_dir(args.out)
        release, inventory, payloads, states = _real_inputs(args, pinned)
        plan = _plan(release, states)
        if args.plan_only:
            identity = None
            if pinned is not None:
                identity, _validated = _pinned_preflight(release, states, pinned)
            else:
                _size_preflight(release, states)
            _chooser_preflight(release, states, identity)
        manifest = (None if args.plan_only else write_release(
            args.out, release, inventory, payloads, states, incumbent=args.incumbent,
            pinned_panel=pinned))
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        for row in plan["states"]:
            print(f"{row['member_id']:34s} {row['status']:8s} objects={row['objects']} "
                  f"bytes={row['bytes']}")
        print(f"models: {len(plan['models'])} bindings")
        if args.plan_only:
            print(f"plan only: {args.out / 'plan.json'}")
            return 0
        print(f"staged: {manifest}")
    except (PrepareRefused, deployment.DeploymentError, BoundedStateReadError,
            OSError, ValueError) as exc:
        print(f"refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
