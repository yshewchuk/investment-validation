"""Pure, fail-closed declarations for the entry-rule gate's frozen inputs."""
from __future__ import annotations

from math import isfinite
from typing import Any

from engine.v2.models.trailing_cutoff_artifact import (
    PNL_HISTORY_ID,
    TrailingCutoffError,
    cutoff_month,
    trailing_cutoff_key,
)
from engine.v2.scoring.native_entry_rule import ENTRY_RULE_STRATEGIES
from engine.v2.scoring.nightly_source_bundle import NightlySourceBundleRefusal
from engine.v2.scoring.release_bindings import ModelIdentity, ScoringReleaseBinding

__all__ = ["entry_rule_declarations"]


def entry_rule_declarations(
    binding: ScoringReleaseBinding,
    *,
    strategy: str,
    size_identity: ModelIdentity,
    crush_identity: ModelIdentity,
    cutoff: str,
    event_date: Any,
    mcap_usd: Any,
) -> dict[str, Any]:
    """The unique paired pool, the event month's cutoff, and the gate recipe,
    with no I/O, cache or fallback: equal inputs give equal output."""
    if strategy not in ENTRY_RULE_STRATEGIES:
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY", "entry rule strategy is not defined")
    pools = [artifact for artifact in binding.paired_residual_artifacts
             if artifact.move_model_id == size_identity.model_id
             and artifact.crush_model_id == crush_identity.model_id
             and artifact.cutoff == cutoff]
    if len(pools) == 0:
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY", "one exact paired residual pool is required")
    if len(pools) > 1:
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY", "more than one exact paired residual pool")
    pool = pools[0]
    try:
        month = cutoff_month(event_date)
    except (TrailingCutoffError, ValueError, TypeError):
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY",
            "event date is not a valid trailing cutoff month") from None
    key = trailing_cutoff_key(PNL_HISTORY_ID, month)
    cutoffs = [artifact for artifact in binding.trailing_cutoff_artifacts
               if artifact.key == key]
    if len(cutoffs) == 0:
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY", "one exact trailing cutoff artifact is required")
    if len(cutoffs) > 1:
        raise NightlySourceBundleRefusal(
            "MODEL_NOT_READY", "more than one exact trailing cutoff artifact")
    if mcap_usd is not None:
        try:
            usable = not isinstance(mcap_usd, bool) and isfinite(mcap_usd)
        except (TypeError, OverflowError):
            usable = False
        if not usable:
            raise NightlySourceBundleRefusal(
                "MODEL_NOT_READY",
                "entry rule mcap_usd must be a finite number or None")
    return {
        "paired_residual_artifact": pool,
        "paired_residual_recipe": {
            "move_model_id": pool.move_model_id,
            "crush_model_id": pool.crush_model_id,
            "cutoff": cutoff,
            "content_hash": pool.content_hash,
        },
        "trailing_cutoff_artifact": cutoffs[0],
        "gate_recipe": {"mode": "entry_rule", "mcap_usd": mcap_usd},
    }