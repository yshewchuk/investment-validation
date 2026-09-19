"""Schema dispatch for the P5-4 frozen states ``frozen_state`` loads.

Kept apart from the loader so each module stays within its import budget:
this one knows every frozen-state type, the loader knows only verification.
Each reader recomputes the content hash from the document; nothing here
trusts a stored hash.
"""
from __future__ import annotations

from typing import Any, Mapping, Union

from engine.v2.models.admissible_table import (
    ADMISSIBLE_DEPTH_TABLE_V1,
    AdmissibleDepthTable,
    admissible_table_from_document,
)
from engine.v2.models.analog_artifact import (
    BOARD_ANALOG_POOL_ARTIFACT_V1,
    BoardAnalogPoolArtifact,
    analog_artifact_from_document,
)
from engine.v2.models.chooser_analog_pool import (
    CHOOSER_ANALOG_POOL_ARTIFACT_V1,
    ChooserAnalogPoolArtifact,
    chooser_analog_pool_from_document,
)
from engine.v2.models.residual_artifact import (
    DriverResidualPoolArtifact,
    PairedResidualPoolArtifact,
    residual_artifact_from_document,
)

__all__ = ["FrozenState", "frozen_state_from_document"]

FrozenState = Union[
    DriverResidualPoolArtifact, PairedResidualPoolArtifact, AdmissibleDepthTable,
    BoardAnalogPoolArtifact, ChooserAnalogPoolArtifact,
]

#: ``schema_version`` -> reader; anything else is a residual pool (whose own
#: reader refuses an unknown schema).
_READERS = {
    ADMISSIBLE_DEPTH_TABLE_V1: admissible_table_from_document,
    BOARD_ANALOG_POOL_ARTIFACT_V1: analog_artifact_from_document,
    CHOOSER_ANALOG_POOL_ARTIFACT_V1: chooser_analog_pool_from_document,
}


def frozen_state_from_document(document: Mapping[str, Any]) -> FrozenState:
    """Rebuild a frozen state from its JSON document, by ``schema_version``."""
    reader = _READERS.get(document.get("schema_version"), residual_artifact_from_document)
    return reader(document)
