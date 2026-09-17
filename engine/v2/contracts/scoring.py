"""Canonical Phase 4 scoring contracts.

These are data shapes only.  Identity, validation and execution live in the
foundation, registry, features and scoring packages respectively.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "DEPLOYMENT_SPEC_V1", "FEATURE_FRAME_V1", "FEATURE_RECIPE_V1",
    "FEATURE_REQUEST_V1", "SCORE_RECORD_V1", "SCORE_REQUEST_V1",
    "STRATEGY_SPEC_V1", "DeploymentSpec", "FeatureFrame", "FeatureRecipe",
    "FeatureRequest", "EventScoreRequest", "ScoreBatch", "ReplayReceipt",
    "FeatureRequest", "ScoreRecord", "ScoreRequest", "StrategySpec",
]

SCORE_REQUEST_V1 = "score_request.v1.0"
SCORE_RECORD_V1 = "score_record.v1.0"
STRATEGY_SPEC_V1 = "strategy_spec.v1.0"
DEPLOYMENT_SPEC_V1 = "deployment_spec.v1.0"
FEATURE_RECIPE_V1 = "feature_recipe.v1.0"
FEATURE_REQUEST_V1 = "feature_request.v1.0"
FEATURE_FRAME_V1 = "feature_frame.v1.0"

ScoreMode = Literal["research", "replay", "shadow", "serving"]
ValidationStatus = Literal["promoted", "tracked", "disabled", "historical"]


@dataclass(frozen=True, kw_only=True)
class ScoreRequest:
    """A complete, replayable score command with no operational timestamp."""

    event_id: str
    calendar_revision: str
    strategy_version: str
    deployment_id: str
    decision_clock_id: str
    requested_decision_at: str
    snapshot_id: str
    mode: ScoreMode
    fill_model: dict[str, Any]
    event_revision: str | None = None
    contract_override: dict[str, Any] | None = None
    geometry_override: dict[str, Any] | None = None
    dependency_refs: tuple[str, ...] = ()
    model_artifact_refs: tuple[str, ...] = ()
    residual_state_ref: str | None = None
    analog_state_ref: str | None = None
    calibration_state_ref: str | None = None
    schema_version: str = SCORE_REQUEST_V1


@dataclass(frozen=True, kw_only=True)
class EventScoreRequest:
    event_id: str
    request_refs: tuple[str, ...]
    snapshot_id: str
    schema_version: str = "event_score_request.v1.0"


@dataclass(frozen=True, kw_only=True)
class ScoreBatch:
    batch_id: str
    requests: tuple[ScoreRequest, ...]
    population_ref: str
    schema_version: str = "score_batch.v1.0"


@dataclass(frozen=True, kw_only=True)
class ReplayReceipt:
    score_id: str
    request_hash: str
    replay_score_id: str
    status: Literal["replayed", "refused"]
    validation_receipt_ref: str
    schema_version: str = "replay_receipt.v1.0"


@dataclass(frozen=True, kw_only=True)
class ScoreRecord:
    """Immutable canonical score payload plus explicit readiness lineage."""

    score_id: str
    canonical_request: dict[str, Any]
    resolved_request: dict[str, Any]
    event_ref: dict[str, Any]
    clock_id: str
    snapshot_ref: str
    dependency_hash: str
    model_artifact_ids: tuple[str, ...]
    selected_contracts: tuple[dict[str, Any], ...]
    legs: tuple[dict[str, Any], ...]
    entry_exit_plan: dict[str, Any]
    quote_provenance: dict[str, Any]
    forecasts: dict[str, Any]
    uncertainty: dict[str, Any]
    residual_state_ref: str | None
    analog_state_ref: str | None
    payoff_state_ref: str | None
    feature_values: dict[str, Any]
    null_masks: dict[str, bool]
    feature_lineage_refs: tuple[str, ...]
    gate_terms: dict[str, Any]
    chooser_candidates: tuple[dict[str, Any], ...]
    chooser_selection: dict[str, Any] | None
    financial_diagnostics: dict[str, Any]
    requested_payoff_views: tuple[dict[str, Any], ...]
    validation_status: str
    reason_codes: tuple[str, ...]
    warnings: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    payload_hash: str = ""
    request_hash: str = ""
    dependency_manifest_ref: str | None = None
    readiness: str = "ready"
    valid_until: str | None = None
    validation_receipt_refs: tuple[str, ...] = ()
    operational_envelope: dict[str, Any] | None = None
    schema_version: str = SCORE_RECORD_V1


@dataclass(frozen=True, kw_only=True)
class StrategySpec:
    """Versioned strategy definition independent of model weights."""

    strategy_id: str
    strategy_version: str
    definition_hash: str
    validation_status: ValidationStatus
    structure_recipe: str
    structure_parameters: dict[str, Any]
    component_graph_ref: str
    decision_clock: str
    entry_policy: str
    exit_policy: str
    quote_policy: str
    fill_policy: str
    universe_policy: str
    domain_policy: str
    feature_recipe_ids: tuple[str, ...]
    model_role_bindings: dict[str, str]
    forecast_sizing_recipe: str | None
    analog_recipe: str | None
    payoff_recipe: str
    gate_recipe: str | None
    chooser_recipe: str | None
    fallback_policy: str
    refusal_codes: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    schema_version: str = STRATEGY_SPEC_V1


@dataclass(frozen=True, kw_only=True)
class DeploymentSpec:
    """Exact model and policy bindings used by one scoring deployment."""

    deployment_id: str
    strategy_spec_ref: str
    clock_contract_ref: str
    model_role_bindings: dict[str, str]
    feature_contract_bindings: dict[str, str]
    evidence_state_refs: tuple[str, ...]
    validation_receipt_refs: tuple[str, ...]
    effective_from: str
    mode: Literal["candidate", "shadow", "serving", "historical"]
    promotion_receipt_ref: str | None = None
    schema_version: str = DEPLOYMENT_SPEC_V1


@dataclass(frozen=True, kw_only=True)
class FeatureRecipe:
    """Namespaced causal feature recipe and its explicit population policy."""

    recipe_id: str
    version: str
    implementation_hash: str
    input_contracts: tuple[str, ...]
    dependency_recipe_ids: tuple[str, ...]
    output_columns: tuple[dict[str, Any], ...]
    source_scope: str
    history_scope: str
    lookback_rule: str
    observation_cutoff_rule: str
    label_availability_rule: str | None
    supported_clock_contracts: tuple[str, ...]
    fallback_policy: str
    determinism_policy: str
    schema_version: str = FEATURE_RECIPE_V1


@dataclass(frozen=True, kw_only=True)
class FeatureRequest:
    event_refs: tuple[dict[str, Any], ...]
    decision_contexts: tuple[dict[str, Any], ...]
    snapshot_ref: str
    feature_recipe_refs: tuple[str, ...]
    upstream_prediction_refs: tuple[str, ...] = ()
    schema_version: str = FEATURE_REQUEST_V1


@dataclass(frozen=True, kw_only=True)
class FeatureFrame:
    frame_ref: str
    schema_ref: str
    row_keys_ref: str
    ordered_columns: tuple[str, ...]
    recipe_refs: tuple[str, ...]
    dependency_refs: tuple[str, ...]
    values_hash: str
    null_mask_hash: str
    lineage_refs: tuple[str, ...]
    coverage_receipt_ref: str
    causal_audit_receipt_ref: str
    values: tuple[dict[str, Any], ...] = ()
    schema_version: str = FEATURE_FRAME_V1
