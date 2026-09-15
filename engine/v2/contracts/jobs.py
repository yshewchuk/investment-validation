"""Job, stage, attempt, checkpoint and artifact contracts.

Phase-1 guide §5.1–§5.2 and §6.3; component contracts §11.

Schemas only: frozen dataclasses and closed vocabularies. Nothing here computes
a hash, reads a clock or touches a file — a ``spec_hash`` is a field somebody
else filled in, never a value derived during construction.

The kinds of contracts §2.5 are kept straight because they decide lifecycle:

* ``JobSpec``, ``StageSpec`` are Definitions — immutable once submitted.
* ``SubmitRequest`` is a Command, hashed into the request digest that makes
  submission idempotent.
* ``ArtifactRef`` and ``LegacyInputManifest`` are Handles/provenance.
* ``*Receipt`` types are evidence, retained even when the operation failed.

``priority``, ``deadline_at`` and every field of ``ResolvedResources`` are
scheduling envelope. They are not economic inputs: changing them never makes a
new experiment hypothesis, and changing a fill rule, seed, model or fold plan
is never "merely a retry" (§5.1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from engine.v2.contracts.operations import (
    Problem,
    ProcessIdentity,
    ProgressEvent,
    QueueReason,
    ResolvedResources,
)

__all__ = [
    "ARTIFACT_REF_V1",
    "ATTEMPT_RECEIPT_V1",
    "CANCELLATION_RECEIPT_V1",
    "CHECKPOINT_RECEIPT_V1",
    "JOB_RECEIPT_V1",
    "JOB_SPEC_V1",
    "LEGACY_INPUT_MANIFEST_V1",
    "STAGE_RESULT_V1",
    "STAGE_SPEC_V1",
    "SUBMIT_REQUEST_V1",
    "ArtifactRef",
    "AttemptReceipt",
    "AttemptState",
    "CancellationReceipt",
    "CheckpointCandidate",
    "CheckpointReceipt",
    "EffectClass",
    "JobReceipt",
    "JobSpec",
    "JobState",
    "LegacyFileRef",
    "LegacyInputManifest",
    "OutputCandidate",
    "ProcessState",
    "StageResult",
    "StageSpec",
    "SubmitRequest",
]

ARTIFACT_REF_V1 = "artifact_ref.v1.0"
JOB_SPEC_V1 = "job_spec.v1.0"
SUBMIT_REQUEST_V1 = "submit_request.v1.0"
JOB_RECEIPT_V1 = "job_receipt.v1.0"
CANCELLATION_RECEIPT_V1 = "cancellation_receipt.v1.0"
STAGE_SPEC_V1 = "stage_spec.v1.0"
LEGACY_INPUT_MANIFEST_V1 = "legacy_input_manifest.v1.0"
ATTEMPT_RECEIPT_V1 = "attempt_receipt.v1.0"
CHECKPOINT_RECEIPT_V1 = "checkpoint_receipt.v1.0"
STAGE_RESULT_V1 = "stage_result.v1.0"

#: §6.3. ``blocked`` is a failed dependency or operator action needed.
JobState = Literal["queued", "running", "succeeded", "retry_wait", "failed",
                   "cancelling", "cancelled", "blocked"]
#: The logical outcome of one attempt. ``recovery_pending`` persists the
#: recovery condition even though it is not a public job state (§6.3).
AttemptState = Literal["starting", "running", "succeeded", "failed", "cancelling",
                       "cancelled", "recovery_pending"]
#: Whether the attempt's processes are known to be gone — separate from the
#: logical outcome, because reservations are released on THIS, not on that.
ProcessState = Literal["unlaunched", "alive", "exited", "verified_dead",
                       "unknown", "quarantined"]
EffectClass = Literal["pure", "staged", "catalog_commit", "external_delivery"]


@dataclass(frozen=True, kw_only=True)
class ArtifactRef:
    """An immutable, content-addressed object. Location is internal (contracts §2.1)."""

    artifact_id: str
    content_hash: str
    schema_ref: str
    byte_size: int
    storage_key: str
    schema_version: str = ARTIFACT_REF_V1


@dataclass(frozen=True, kw_only=True)
class JobSpec:
    """contracts §11.1 JobSpec — a specification, never an argv string.

    ``parameters`` are validated against the server-owned schema of ``kind``;
    a caller cannot name an executable, a module or a path.
    """

    kind: str
    implementation_ref: str
    spec_hash: str | None
    environment_ref: str
    parameters: dict[str, Any] = field(default_factory=dict)
    input_refs: tuple[str, ...] = ()
    dependency_job_ids: tuple[str, ...] = ()
    output_namespace: str
    resource_class: str
    priority: int = 0
    deadline_at: str | None = None
    provider_budget_ref: str | None = None
    retry_policy_ref: str
    checkpoint_contract_ref: str
    schema_version: str = JOB_SPEC_V1


@dataclass(frozen=True, kw_only=True)
class SubmitRequest:
    """The submission command. Its canonical digest is the idempotency payload."""

    namespace: str
    idempotency_key: str
    principal: str
    job: JobSpec
    schema_version: str = SUBMIT_REQUEST_V1


@dataclass(frozen=True, kw_only=True)
class JobReceipt:
    """contracts §11.1 JobReceipt — the one answer to "what happened to this job"."""

    job_id: str
    namespace: str
    idempotency_key: str
    request_digest: str
    kind: str
    spec_hash: str | None
    state: JobState
    priority: int
    created_at: str
    fence: int
    attempt_count: int
    active_attempt_id: str | None = None
    next_eligible_at: str | None = None
    queue_reason: QueueReason | None = None
    resolved_resources: ResolvedResources | None = None
    checkpoint_refs: tuple[str, ...] = ()
    output_refs: tuple[str, ...] = ()
    latest_progress: ProgressEvent | None = None
    failure: Problem | None = None
    schema_version: str = JOB_RECEIPT_V1


@dataclass(frozen=True, kw_only=True)
class CancellationReceipt:
    """Cancel is complete only when every owned process is dead (§6.3)."""

    job_id: str
    expected_attempt_id: str | None
    state: Literal["cancelling", "cancelled", "conflict", "already_terminal"]
    fence: int
    requested_at: str
    completed_at: str | None = None
    failure: Problem | None = None
    schema_version: str = CANCELLATION_RECEIPT_V1


@dataclass(frozen=True, kw_only=True)
class StageSpec:
    """Phase-1 specialization of a job for one registered stage (§5.1)."""

    stage_id: str
    job_kind: str
    implementation_ref: str
    parameter_ref: str
    input_contract_ref: str
    output_contract_ref: str
    dependency_stage_ids: tuple[str, ...]
    resource_class: str
    retry_policy_ref: str
    checkpoint_contract_ref: str
    effect_class: EffectClass
    required_validation_kinds: tuple[str, ...]
    determinism_policy_ref: str
    schema_version: str = STAGE_SPEC_V1


@dataclass(frozen=True, kw_only=True)
class LegacyFileRef:
    """One file a legacy stage read, pinned by content while its lease was held."""

    path: str
    content_hash: str
    byte_size: int


@dataclass(frozen=True, kw_only=True)
class LegacyInputManifest:
    """Transitional provenance for legacy inputs (§5.1).

    Not a SnapshotRef: it records what was read under a cooperative lease, and
    promises no repository isolation. Phase 2 replaces its resolution.
    Availability evidence is carried as references only — neither a revision
    identifier nor a hash proves information was available at decision time.
    """

    manifest_id: str
    file_refs: tuple[LegacyFileRef, ...]
    table_contract_refs: tuple[str, ...]
    registry_and_model_refs: tuple[str, ...]
    calendar_ref: str | None
    selected_session: str
    finality_receipt_refs: tuple[str, ...]
    knowledge_mode_by_table: dict[str, str]
    availability_evidence_refs: tuple[str, ...]
    read_set_complete: bool
    capture_implementation_ref: str
    schema_version: str = LEGACY_INPUT_MANIFEST_V1


@dataclass(frozen=True, kw_only=True)
class AttemptReceipt:
    """contracts §5.2 AttemptReceipt. Previous attempts are never overwritten."""

    job_id: str
    attempt_id: str
    attempt_number: int
    fence: int
    supervisor_epoch: str
    host_boot_id: str
    state: AttemptState
    process_state: ProcessState
    process_identity: ProcessIdentity | None = None
    started_at: str | None = None
    heartbeat_at: str | None = None
    lease_expires_at: str | None = None
    ended_at: str | None = None
    exit_code: int | None = None
    memory_current_bytes: int | None = None
    memory_peak_bytes: int | None = None
    resolved_resources: ResolvedResources | None = None
    input_manifest_ref: str | None = None
    checkpoint_refs: tuple[str, ...] = ()
    output_refs: tuple[str, ...] = ()
    validation_refs: tuple[str, ...] = ()
    failure: Problem | None = None
    schema_version: str = ATTEMPT_RECEIPT_V1


@dataclass(frozen=True, kw_only=True)
class OutputCandidate:
    """A file a worker staged, named relative to its own staging directory."""

    name: str
    staged_path: str
    schema_ref: str


@dataclass(frozen=True, kw_only=True)
class CheckpointCandidate:
    """A proposed checkpoint. Reusable only after the coordinator commits it."""

    shard_key: str
    cache_key: str
    input_hash: str
    implementation_hash: str
    parameter_hash: str
    environment_hash: str
    output_schema_ref: str
    outputs: tuple[OutputCandidate, ...]
    coverage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class CheckpointReceipt:
    """A committed checkpoint: the identity it was produced under, and by whom."""

    stage_id: str
    shard_key: str
    cache_key: str
    input_hash: str
    implementation_hash: str
    parameter_hash: str
    environment_hash: str
    output_schema_ref: str
    artifact_refs: tuple[ArtifactRef, ...]
    validation_refs: tuple[str, ...]
    producer_attempt_id: str
    producer_fence: int
    committed_at: str
    schema_version: str = CHECKPOINT_RECEIPT_V1


@dataclass(frozen=True, kw_only=True)
class StageResult:
    """What a worker sends back over its result channel (§5.2).

    Exit code zero is necessary and insufficient: a result missing its manifest,
    coverage or validators is not success, whatever the process returned.
    """

    job_id: str
    attempt_id: str
    fence: int
    stage_id: str
    input_manifest_ref: str | None
    checkpoint_candidates: tuple[CheckpointCandidate, ...] = ()
    output_candidates: tuple[OutputCandidate, ...] = ()
    validation_refs: tuple[str, ...] = ()
    completion_counts: dict[str, int] = field(default_factory=dict)
    failure: Problem | None = None
    schema_version: str = STAGE_RESULT_V1
