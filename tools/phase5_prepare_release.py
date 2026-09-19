#!/usr/bin/env python3
"""P5-6: assemble a staged model release from real artifacts.

Reads (never writes) the champion registry and ``data/models/*.joblib``, the
Tier-4 serving folds under ``data/models/tier4``, P5-3 training-job outputs
(``payoff_artifact.json`` per calibration fold) and any pre-built P5-4 frozen
states. Writes one release root (layout: ``checks/phase5_release.py``) under
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
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.phase5_release import (  # noqa: E402
    MISSING,
    PENDING,
    STAGED,
    STATE_SPECS,
    StateSpec,
    deployment_root,
    manifest_body,
    member_row,
    modules_available,
    sha256_bytes,
    write_manifest,
    write_object,
)
from engine.v2.models import deployment  # noqa: E402
from engine.v2.models.contracts import ArtifactMember, ModelBinding, ModelRelease  # noqa: E402
from engine.v2.models.releases import ModelReleaseInventory  # noqa: E402

JOBLIB_ADAPTER = "joblib-estimator.v1"
#: Output name per role for the frozen binding. Resolution-only today: no
#: consumer reads these names yet, so they follow the Phase 4 native field
#: names where one exists.
OUTPUT_NAMES = {
    "size": ("forecast_abs_move",), "implied_t1": ("driver_prediction",),
    "runup_move": ("runup_move_prediction",), "iv_crush": ("pred_iv_crush",),
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


# --------------------------------------------------------------------------
# model bindings
# --------------------------------------------------------------------------


def model_release(inventory: ModelReleaseInventory, *, read=lambda p: Path(p).read_bytes(),
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


def _payoff_member_id(artifact) -> str:
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
            data = path.read_bytes()
            artifact = PayoffArtifactLoader(path.parent).load(
                PayoffArtifactRef(path=path.name, content_hash=sha256_bytes(data)))
            name = f"{artifact.strategy}|{artifact.alpha}|{artifact.cutoff}"
            found.setdefault(_payoff_member_id(artifact), {})[name] = data
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
            data = path.read_bytes()
            artifact = RecalibrationArtifactLoader(path.parent).load(
                RecalibrationArtifactRef(path=path.name, content_hash=sha256_bytes(data)))
            name = f"{artifact.strategy}|{artifact.alpha}|{artifact.cutoff}"
            found.setdefault(f"recalibration_map:{artifact.strategy}", {})[name] = data
    return found


def frozen_state_payloads(files: Iterable[Path]) -> dict[str, dict[str, bytes]]:
    """Pre-built P5-4 frozen states (residual pools, tables), by member."""
    from engine.v2.models.frozen_state import FrozenStateLoader, FrozenStateRef

    found: dict[str, dict[str, bytes]] = {}
    for path in files:
        path = Path(path)
        data = path.read_bytes()
        state = FrozenStateLoader(path.parent).load(
            FrozenStateRef(path=path.name, content_hash=sha256_bytes(data)))
        schema = state.schema_version
        if schema.startswith("driver_residual_pool"):
            member, name = f"driver_residual_pool:{state.role}", f"{state.model_id}|{state.fold}"
        elif schema.startswith("paired_residual_pool"):
            member = "paired_residual_pool"
            name = f"{state.move_model_id}|{state.crush_model_id}|{state.cutoff}"
        else:
            member, name = "admissible_table:dyn_sv", f"{state.table_id}|{state.version}"
        found.setdefault(member, {})[name] = data
    return found


#: Model-stage driver pools: STR-THRU reads ``size`` (driver abs_move),
#: STR-RUNUP reads ``implied_t1`` and ``runup_move`` (engine/score.py).
DRIVER_POOL_ROLES = ("size", "implied_t1", "runup_move")


def champion_driver_pools(inventory: ModelReleaseInventory, *, load,
                          resolve=lambda ref: Path(ref)) -> dict[str, dict[str, bytes]]:
    """Freeze each driver champion's own embedded residual pool, unchanged.

    Legacy serves the model stage's draws from the full-refit champion's
    ``ModelArtifact.residuals`` and ``residual_buckets`` (flat pool plus
    prediction-decile buckets). Those arrays are wrapped as-is with the
    layer-3 constructor -- no re-bucketing, no fitting -- keyed
    ``(role, champion id, fold=None)``, the key the residual artifact module
    documents for a full-refit champion's own pool. The champion's training
    cutoff is not recorded anywhere, so the lineage declares the Tier-3 panel
    with no end bound: any panel correction invalidates it (conservative).
    """
    from engine.v2.models.frozen_state import serialize_frozen_state
    from engine.v2.models.lineage import DataDependency, Lineage
    from engine.v2.models.residual_artifact import make_driver_residual_pool_artifact
    from engine.v2.scoring import native_payoff

    lineage = Lineage(data=(DataDependency(table="tier3.panel"),))
    estimators = {item.artifact_id: next(m for m in item.members if m.kind == "estimator")
                  for item in inventory.artifacts}
    found: dict[str, dict[str, bytes]] = {}
    for binding in inventory.bindings:
        if binding.role not in DRIVER_POOL_ROLES:
            continue
        artifact = load(resolve(estimators[binding.artifact_id].artifact_ref))
        buckets = artifact.residual_buckets or None
        pool = make_driver_residual_pool_artifact(
            role=binding.role, model_id=binding.artifact_id, fold=None,
            flat_residuals=artifact.residuals,
            buckets=None if buckets is None else {"edges": buckets["edges"],
                                                  "pools": buckets["pools"]},
            deciles=native_payoff.DECILES,
            min_pool=int((buckets or {}).get("min_pool", native_payoff.MIN_POOL)),
            lineage=lineage,
        )
        found[f"driver_residual_pool:{binding.role}"] = {
            f"{binding.artifact_id}|champion": serialize_frozen_state(pool)}
    return found


def default_admissible_table() -> dict[str, dict[str, bytes]]:
    from engine.v2.models.admissible_table import legacy_n_admissible_table
    from engine.v2.models.frozen_state import serialize_frozen_state

    table = legacy_n_admissible_table()
    return {"admissible_table:dyn_sv": {
        f"{table.table_id}|{table.version}": serialize_frozen_state(table)}}


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
            path.read_bytes())
    return found


def build_states(available_payloads: Mapping[str, Mapping[str, bytes]],
                 notes: Mapping[str, str] | None = None) -> list[StateBuild]:
    """One build row per catalog state: STAGED, MISSING or PENDING, never skipped."""
    notes = notes or {}
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
    return builds


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


def _copy_incumbent(source: Path, dest: Path) -> None:
    if (dest / "releases").exists():
        raise PrepareRefused(f"{dest} already holds a deployment store")
    for name in ("releases", "objects", "history"):
        if (Path(source) / name).is_dir():
            shutil.copytree(Path(source) / name, dest / name)
    if (Path(source) / "DEPLOYED").is_file():
        shutil.copy2(Path(source) / "DEPLOYED", dest / "DEPLOYED")


def write_release(out: Path, release: ModelRelease, inventory: ModelReleaseInventory,
                  payloads: Mapping[str, bytes], states: list[StateBuild], *,
                  sources: Mapping[str, str] | None = None,
                  incumbent: Path | None = None) -> Path:
    """Stage the model bindings, write the state objects, write the manifest."""
    out = Path(out)
    root = deployment_root(out)
    root.mkdir(parents=True, exist_ok=True)
    if incumbent is not None:
        _copy_incumbent(incumbent, root)
    deployment.stage_release(root, release, inventory, dict(payloads))
    rows = []
    for build in states:
        objects = []
        for name, data in sorted(build.payloads.items()):
            digest, rel = write_object(out, data)
            objects.append({"name": name, "path": rel, "content_hash": digest,
                            "bytes": len(data)})
        rows.append(member_row(build.spec, build.status, objects, build.detail))
    body = manifest_body(release.release_id, release.deployment_id, rows, sources or {})
    return write_manifest(out, body)


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


def _merge(found: dict[str, dict[str, bytes]], more: Mapping[str, Mapping[str, bytes]]) -> None:
    """Add objects per member; a later source adds to, never drops, a member."""
    for member_id, objects in more.items():
        found.setdefault(member_id, {}).update(objects)


def _refuse_data_dir(out: Path) -> None:
    from engine import paths

    resolved, data = Path(out).resolve(), paths.DATA.resolve()
    if resolved == data or data in resolved.parents:
        raise PrepareRefused(f"--out may not be inside {data}")


def _real_inputs(args) -> tuple[ModelRelease, ModelReleaseInventory, dict, list[StateBuild]]:
    from engine import paths
    from engine.data.features import tier4
    from engine.v2.models.inventory import current_release_inventory

    inventory, drift = current_release_inventory()
    if drift:
        raise PrepareRefused("registry drift: " + ", ".join(sorted({i.code for i in drift})))
    inventory = dataclasses.replace(inventory, release_id=args.release_id)
    release, payloads = model_release(
        inventory, resolve=lambda ref: Path(ref) if Path(ref).is_absolute() else paths.ROOT / ref)
    snapshot = args.tier3_snapshot
    if snapshot == "auto":
        from engine.data import store

        snapshot = store.file_sha256(paths.PANEL)
    feature_ids = {b.role: b.artifact_id for b in inventory.bindings if b.strategy_id == "*"}
    found: dict[str, dict[str, bytes]] = {}
    _merge(found, tier4_fold_payloads(Path(args.tier4_dir or tier4.SERVING_DIR),
                                     feature_ids, snapshot, args.fold_month))
    if modules_available(("engine.v2.models.payoff_artifact",))[0]:
        _merge(found, payoff_payloads(args.training_root or ()))
    if modules_available(("engine.v2.models.recalibration_artifact",))[0]:
        _merge(found, recalibration_payloads(args.training_root or ()))
    if modules_available(("engine.v2.models.frozen_state",))[0]:
        from engine.models import registry as legacy_registry

        _merge(found, default_admissible_table())
        _merge(found, champion_driver_pools(
            inventory, load=legacy_registry.load_artifact,
            resolve=lambda ref: Path(ref) if Path(ref).is_absolute() else paths.ROOT / ref))
        _merge(found, frozen_state_payloads(args.frozen_state or ()))
    pool = paths.FEATURES / "chooser_analog_pool.parquet"
    if pool.is_file():
        found["chooser_analog_pool"] = {pool.name: pool.read_bytes()}
    return release, inventory, payloads, build_states(found)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, action="append",
                        help="P5-3 training-job output dir holding payoff_artifact.json files")
    parser.add_argument("--frozen-state", type=Path, action="append",
                        help="a pre-built P5-4 frozen state JSON (residual pool, table)")
    parser.add_argument("--tier4-dir", type=Path)
    parser.add_argument("--tier3-snapshot", default="auto",
                        help="Tier-3 snapshot hash selecting servable folds; 'auto' hashes "
                             "panel.parquet")
    parser.add_argument("--fold-month", help="restrict Tier-4 folds to one YYYYMM")
    parser.add_argument("--incumbent", type=Path,
                        help="an existing deployment store to copy in before staging")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        _refuse_data_dir(args.out)
        release, inventory, payloads, states = _real_inputs(args)
        plan = _plan(release, states)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        for row in plan["states"]:
            print(f"{row['member_id']:34s} {row['status']:8s} objects={row['objects']} "
                  f"bytes={row['bytes']}")
        print(f"models: {len(plan['models'])} bindings")
        if args.plan_only:
            print(f"plan only: {args.out / 'plan.json'}")
            return 0
        manifest = write_release(args.out, release, inventory, payloads, states,
                                 incumbent=args.incumbent)
        print(f"staged: {manifest}")
    except (PrepareRefused, deployment.DeploymentError, ValueError) as exc:
        print(f"refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
