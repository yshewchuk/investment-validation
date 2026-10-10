"""Final-read metadata only; these records confer no read authority."""
from dataclasses import dataclass

from engine.v2.contracts.jobs import ArtifactRef


@dataclass(frozen=True, kw_only=True)
class FinalHoldoutDeclaration:
    evaluation_recipe_ref: ArtifactRef
    completeness_bar_ref: ArtifactRef
    frozen_model_refs: tuple[ArtifactRef, ...]
    schema_version: str = "final_holdout_declaration.v1.0"


@dataclass(frozen=True, kw_only=True)
class FinalHoldoutRequest:
    experiment_id: str
    registration_run_id: str
    variant_id: str
    registration_ref: ArtifactRef
    snapshot_id: str
    holdout_as_of_month: str
    random_membership_version: str
    rolling_membership_version: str
    execution_plan_hash: str
    source_closure_hash: str
    final_read: FinalHoldoutDeclaration
    schema_version: str = "final_holdout_request.v1.0"
