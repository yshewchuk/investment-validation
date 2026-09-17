"""Frozen model contracts, verified loading, and read-only inference."""

from .adapters import (
    AdapterError,
    InferenceAdapter,
    JoblibEstimatorAdapter,
    JsonLinearAdapter,
    RuntimeFitForbidden,
    default_adapters,
)
from .contracts import (
    MODEL_NOT_READY,
    MODEL_READY,
    ArtifactMember,
    InferenceRequest,
    InferenceResult,
    ModelBinding,
    ModelRelease,
    PredictionFrame,
)
from .loader import FrozenInference

__all__ = [
    "AdapterError",
    "ArtifactMember",
    "FrozenInference",
    "InferenceAdapter",
    "InferenceRequest",
    "InferenceResult",
    "JoblibEstimatorAdapter",
    "JsonLinearAdapter",
    "MODEL_NOT_READY",
    "MODEL_READY",
    "ModelBinding",
    "ModelRelease",
    "PredictionFrame",
    "RuntimeFitForbidden",
    "default_adapters",
]
from engine.v2.models.releases import (
    ARTIFACT_INVENTORY_MEMBER_V1,
    MODEL_ARTIFACT_INVENTORY_V1,
    MODEL_RELEASE_REFUSAL,
    MODEL_RELEASE_V1,
    RELEASE_BINDING_V1,
    RELEASE_REQUIREMENT_V1,
    ArtifactInventoryMember,
    ModelArtifactInventory,
    ModelReleaseInventory,
    ModelReleaseRefusal,
    ReleaseBinding,
    ReleaseIssue,
    ReleaseRequirement,
    release_issues,
    require_complete_release,
)

__all__ += [
    "ARTIFACT_INVENTORY_MEMBER_V1", "MODEL_ARTIFACT_INVENTORY_V1",
    "MODEL_RELEASE_REFUSAL", "MODEL_RELEASE_V1", "RELEASE_BINDING_V1",
    "RELEASE_REQUIREMENT_V1", "ArtifactInventoryMember", "ModelArtifactInventory",
    "ModelReleaseInventory", "ModelReleaseRefusal", "ReleaseBinding", "ReleaseIssue",
    "ReleaseRequirement", "release_issues", "require_complete_release",
]
