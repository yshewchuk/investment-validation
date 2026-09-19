"""Release members for P5-4 frozen state (residual/analog pools, calibration tables).

A frozen state's serialized bytes hash to its own ``content_hash``
(``frozen_state.serialize_frozen_state``), so a release pins it exactly like
a model member: :func:`release_member` gives the ``ArtifactMember`` that
``deployment.stage_release`` verifies against the payload bytes, and
:func:`inventory_member` the ``ArtifactInventoryMember`` a
``ModelArtifactInventory`` lists under the matching ``MemberKind``.
"""
from __future__ import annotations

from typing import Any

from engine.v2.models.admissible_table import ADMISSIBLE_DEPTH_TABLE_V1
from engine.v2.models.analog_artifact import BOARD_ANALOG_POOL_ARTIFACT_V1
from engine.v2.models.chooser_analog_pool import CHOOSER_ANALOG_POOL_ARTIFACT_V1
from engine.v2.models.contracts import ArtifactMember
from engine.v2.models.releases import ArtifactInventoryMember
from engine.v2.models.residual_artifact import (
    DRIVER_RESIDUAL_POOL_ARTIFACT_V1,
    PAIRED_RESIDUAL_POOL_ARTIFACT_V1,
)
from engine.v2.models.trailing_cutoff_artifact import TRAILING_CUTOFF_ARTIFACT_V1

__all__ = ["inventory_member", "member_kind", "release_member"]

#: Which ``releases.MemberKind`` each frozen state is pinned as.
_MEMBER_KIND = {
    DRIVER_RESIDUAL_POOL_ARTIFACT_V1: "residual_bucket",
    PAIRED_RESIDUAL_POOL_ARTIFACT_V1: "paired_simulation",
    ADMISSIBLE_DEPTH_TABLE_V1: "calibration",
    BOARD_ANALOG_POOL_ARTIFACT_V1: "residual",
    # checks/phase5_release.py ``chooser_analog_pool`` (kind "residual").
    CHOOSER_ANALOG_POOL_ARTIFACT_V1: "residual",
    # checks/phase5_release.py ``trailing_pnl_cutoff`` (kind "threshold").
    TRAILING_CUTOFF_ARTIFACT_V1: "threshold",
}


def member_kind(state: Any) -> str:
    """The release member kind for a frozen state; refuses anything else."""
    try:
        return _MEMBER_KIND[state.schema_version]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"not a frozen state: {type(state).__name__}") from exc


def release_member(state: Any, name: str, path: str = "") -> ArtifactMember:
    member_kind(state)
    return ArtifactMember(name=name, path=path, content_hash=state.content_hash)


def inventory_member(state: Any, member_id: str, artifact_ref: str) -> ArtifactInventoryMember:
    return ArtifactInventoryMember(
        member_id=member_id, kind=member_kind(state),
        artifact_ref=artifact_ref, content_hash=state.content_hash,
    )
