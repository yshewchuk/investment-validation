"""Pure, fail-closed declarations for RUNUP's verified frozen model inputs."""
from __future__ import annotations

from math import isfinite
from typing import Any, Mapping, Sequence

from engine.v2.models.payoff_artifact import PayoffSurfaceArtifact
from engine.v2.models.residual_artifact import DriverResidualPoolArtifact
from engine.v2.scoring.native_payoff import RUNUP_TERMS
from engine.v2.scoring.nightly_source_bundle import NightlySourceBundleRefusal
from engine.v2.scoring.release_bindings import ModelIdentity, ScoringReleaseBinding


def _pool_is_usable(pool: DriverResidualPoolArtifact) -> bool:
    """Validate selectable populations, including the flat thin-bucket fallback."""
    if not pool.flat_residuals or not all(map(isfinite, pool.flat_residuals)):
        return False
    if pool.bucket_edges is None:
        return pool.bucket_pools is None
    edges, pools = pool.bucket_edges, pool.bucket_pools
    return (len(edges) >= 2 and pools is not None and len(pools) == len(edges) - 1
            and all(bucket or pool.min_pool > 0 for bucket in pools)
            and all(left < right for left, right in zip(edges, edges[1:]))
            and all(isfinite(value) for bucket in pools for value in bucket))


def _champion_pool(binding: ScoringReleaseBinding, identity: ModelIdentity) -> DriverResidualPoolArtifact:
    pool = binding.driver_residual_artifacts.get(identity.role)
    if (not isinstance(pool, DriverResidualPoolArtifact)
            or pool.role != identity.role or pool.model_id != identity.model_id
            or pool.fold is not None or not _pool_is_usable(pool)):
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY", "RUNUP champion residual pool is missing or unusable")
    return pool


def latest_artifacts_at_or_before(
    artifacts: Sequence[Any], *, cutoff: str, fill_alpha: float,
) -> tuple[Any, ...]:
    """Return all artifacts at the newest DATA cutoff no later than cutoff.

    Artifact ``computed_at`` is deliberately irrelevant: the embedded data
    cutoff alone establishes causal eligibility and ordering.
    """
    eligible = tuple(
        artifact for artifact in artifacts
        if isinstance(getattr(artifact, "cutoff", None), str)
        and artifact.cutoff <= cutoff
        and isinstance(getattr(artifact, "alpha", None), (int, float))
        and round(float(artifact.alpha), 4) == round(float(fill_alpha), 4)
    )
    if not eligible:
        return ()
    latest_cutoff = max(artifact.cutoff for artifact in eligible)
    return tuple(artifact for artifact in eligible if artifact.cutoff == latest_cutoff)


def _surface(binding: ScoringReleaseBinding, cutoff: str, fill_alpha: float) -> PayoffSurfaceArtifact:
    candidates = tuple(
        artifact for artifact in binding.payoff_artifacts.get("STR-RUNUP", ())
        if isinstance(artifact, PayoffSurfaceArtifact)
    )
    matches = latest_artifacts_at_or_before(
        candidates, cutoff=cutoff, fill_alpha=fill_alpha)
    if len(matches) == 0:
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY", "RUNUP has no eligible payoff surface")
    if len(matches) > 1:
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY", "RUNUP has duplicate latest payoff surfaces")
    surface = matches[0]
    if (surface.strategy != "STR-RUNUP"
            or len(surface.coefficients) != len(RUNUP_TERMS)
            or not all(map(isfinite, surface.coefficients))
            or not surface.residuals or not all(map(isfinite, surface.residuals))):
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY", "RUNUP payoff surface is numerically unusable")
    return surface


def runup_model_declarations(
    binding: ScoringReleaseBinding,
    identities: Mapping[str, ModelIdentity],
    *,
    cutoff: str,
    fill_alpha: float,
) -> dict[str, Any]:
    """Both champion slots and a unique causal surface, with no I/O or fallback.

    Slot identity follows forecast meaning, not declaration or catalog order.
    RUNUP's model gate does not request the separate entry-rule simulation;
    the payoff model's own Monte Carlo remains required and executes normally.
    """
    outputs = {"driver_prediction": "driver", "runup_move_prediction": "runup_move"}
    if set(identities) != set(outputs):
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY", "RUNUP requires both forecast identities")
    pools = {slot: _champion_pool(binding, identities[output]) for output, slot in outputs.items()}
    surface = _surface(binding, cutoff, fill_alpha)
    return {
        "model_residual_artifacts": pools,
        "model_residual_artifact_recipe": {slot: {
            "role": pool.role, "model_id": pool.model_id, "fold": None,
            "content_hash": pool.content_hash} for slot, pool in pools.items()},
        "payoff_artifact": surface,
        "payoff_artifact_recipe": {"before": cutoff, "bound_cutoff": surface.cutoff},
        "residual_recipe": {"mode": "not_applicable"},
    }
