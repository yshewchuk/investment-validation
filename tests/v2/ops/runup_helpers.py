"""Tiny, genuinely staged RUNUP model-state fixtures; never fit or read data."""
import json

from engine.v2.foundation import content_hash
from engine.v2.models.frozen_state import serialize_frozen_state
from engine.v2.models.lineage import Lineage
from engine.v2.models.payoff_artifact import (
    make_payoff_line_artifact, make_payoff_surface_artifact, serialize_payoff_artifact)
from engine.v2.models.recalibration_artifact import (
    make_recalibration_map_artifact, serialize_recalibration_artifact)
from engine.v2.models.residual_artifact import make_driver_residual_pool_artifact
from tests.v2.models.serving_fold_support import write_catalog
from tests.v2.scoring.test_v2_scoring_release_bindings import (
    _obj, _row, _write_catalog, _write_object)
from tests.v2.ops.test_native_score_batch_driver_alias import _stage


def pool(role, *, model_id=None, residuals=(0.0,), fold=None, buckets=None, min_pool=1):
    return make_driver_residual_pool_artifact(
        role=role, model_id=model_id or f"m-{role}-*", fold=fold,
        flat_residuals=residuals, buckets=buckets, deciles=10,
        min_pool=min_pool, lineage=Lineage())


def surface(*, cutoff="2026-01-09", alpha=0.5, strategy="STR-RUNUP",
            coefficients=(0.02, 0.004, 0.002, 0.0, 0.0, 0.0), residuals=(0.0,)):
    return make_payoff_surface_artifact(
        {"n": 2, "coefficients": coefficients, "resid_sd": 0.0, "r": 1.0, "residuals": residuals},
        strategy=strategy, alpha=alpha, cutoff=cutoff)


def recalibration(*, cutoff="2026-01-01", alpha=0.5, strategy="STR-RUNUP"):
    return make_recalibration_map_artifact(
        {"n": 10, "base_rate": 0.5, "x_thresholds": (0.0, 1.0),
         "y_thresholds": (0.2, 0.8)},
        strategy=strategy, alpha=alpha, cutoff=cutoff, min_pairs=10)


def payoff_line(*, cutoff="2026-01-09", alpha=0.5, strategy="STR-THRU"):
    return make_payoff_line_artifact(
        {"n": 2, "resid_sd": 0.0, "r": 1.0, "residuals": (0.0,),
         "intercept": 0.05, "slope": 0.5},
        strategy=strategy, driver="implied_t1", alpha=alpha, cutoff=cutoff)


def state_row(root, member_id, artifacts):
    objects = []
    for artifact in artifacts:
        if member_id.startswith("driver_residual_pool:"):
            payload = serialize_frozen_state(artifact)
        elif member_id.startswith("recalibration_map:"):
            payload = serialize_recalibration_artifact(artifact)
        else:
            payload = serialize_payoff_artifact(artifact)
        path = _write_object(root / "deployment", payload)
        objects.append(_obj(path, artifact.content_hash))
    return _row(member_id, objects)


def native_artifact_rows(root, strategies):
    rows = []
    for strategy in sorted(strategies):
        if strategy == "STR-THRU":
            rows.append(state_row(root, f"payoff_line:{strategy}", [payoff_line(strategy=strategy)]))
        if strategy == "STR-RUNUP":
            rows.append(state_row(root, f"payoff_surface:{strategy}", [surface(strategy=strategy)]))
        if strategy in ("STR-THRU", "STR-RUNUP"):
            rows.append(state_row(
                root, f"recalibration_map:{strategy}", [recalibration(strategy=strategy)]))
    return rows


def write_catalog_with_rows(root, release_id, rows, fold_object=None):
    if not rows:
        write_catalog(root, release_id, fold_object)
        return
    _write_catalog(root, release_id=release_id, rows=rows)
    if fold_object is None:
        return
    path = root / "phase5_release.json"
    body = json.loads(path.read_text())
    body["members"].append({
        "member_id": "tier4_folds:size", "status": "STAGED", "objects": [fold_object],
    })
    body["manifest_hash"] = content_hash(
        {k: v for k, v in body.items() if k != "manifest_hash"})
    path.write_text(json.dumps(body, sort_keys=True))


def stage_model_release(root, specs, release_id="r1", rows=None):
    """Supply real default model state unless the test stages that member itself."""
    rows = list(rows or ())
    supplied = {row["member_id"] for row in rows}
    for role, strategy, _ in specs:
        if role not in ("implied_t1", "runup_move") and (role, strategy) != ("driver", "STR-RUNUP"):
            continue
        member = f"driver_residual_pool:{role}"
        if member not in supplied:
            artifact = pool(role, model_id=f"m-{role}-{strategy}")
            rows.append(state_row(root, member, [artifact]))
    if "payoff_surface:STR-RUNUP" not in supplied:
        rows.append(state_row(root, "payoff_surface:STR-RUNUP", [surface()]))
    if "recalibration_map:STR-RUNUP" not in supplied:
        rows.append(state_row(root, "recalibration_map:STR-RUNUP", [recalibration()]))
    return _stage(root, specs, release_id=release_id, rows=rows)
