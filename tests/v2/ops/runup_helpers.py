"""Tiny, genuinely staged RUNUP model-state fixtures; never fit or read data."""
from engine.v2.models.frozen_state import serialize_frozen_state
from engine.v2.models.lineage import Lineage
from engine.v2.models.payoff_artifact import make_payoff_surface_artifact, serialize_payoff_artifact
from engine.v2.models.residual_artifact import make_driver_residual_pool_artifact
from tests.v2.scoring.test_v2_scoring_release_bindings import _obj, _row, _write_object
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


def state_row(root, member_id, artifacts):
    objects = []
    for artifact in artifacts:
        payload = (serialize_frozen_state(artifact) if member_id.startswith("driver_residual_pool:")
                   else serialize_payoff_artifact(artifact))
        path = _write_object(root / "deployment", payload)
        objects.append(_obj(path, artifact.content_hash))
    return _row(member_id, objects)


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
    return _stage(root, specs, release_id=release_id, rows=rows)
