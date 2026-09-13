"""schemas and types only, no logic, no I/O

Layer 0 of `system_rearchitecture.md` §4.1. Replaces `the dataclasses currently
declared inside score.py`.

Rearchitecture phase 1 declares the operations contracts here: jobs, stages,
attempts, checkpoints and artifacts (``jobs``), and the failure, progress and
resource envelopes (``operations``). Frozen dataclasses and closed
vocabularies only — the strict decoder is ``engine.v2.foundation.typed``.
"""
from __future__ import annotations

from engine.v2.contracts.jobs import (
    ARTIFACT_REF_V1,
    ATTEMPT_RECEIPT_V1,
    CANCELLATION_RECEIPT_V1,
    CHECKPOINT_RECEIPT_V1,
    JOB_RECEIPT_V1,
    JOB_SPEC_V1,
    LEGACY_INPUT_MANIFEST_V1,
    STAGE_RESULT_V1,
    STAGE_SPEC_V1,
    SUBMIT_REQUEST_V1,
    ArtifactRef,
    AttemptReceipt,
    AttemptState,
    CancellationReceipt,
    CheckpointCandidate,
    CheckpointReceipt,
    EffectClass,
    JobReceipt,
    JobSpec,
    JobState,
    LegacyFileRef,
    LegacyInputManifest,
    OutputCandidate,
    ProcessState,
    StageResult,
    StageSpec,
    SubmitRequest,
)
from engine.v2.contracts.operations import (
    CAPACITY_SAMPLE_V1,
    FAILURE_CODES,
    PROBLEM_V1,
    PROGRESS_EVENT_V1,
    RESOLVED_RESOURCES_V1,
    RESOURCE_POLICY_V1,
    CapacitySample,
    Containment,
    ExecutorMode,
    LiveWindow,
    Problem,
    ProblemCategory,
    ProcessIdentity,
    ProgressEvent,
    ProgressKind,
    QueueReason,
    ResolvedResources,
    ResourcePolicy,
    ResourceProfile,
)

__all__ = [
    "ARTIFACT_REF_V1", "ATTEMPT_RECEIPT_V1", "CANCELLATION_RECEIPT_V1",
    "CAPACITY_SAMPLE_V1", "CHECKPOINT_RECEIPT_V1", "FAILURE_CODES", "JOB_RECEIPT_V1",
    "JOB_SPEC_V1", "LEGACY_INPUT_MANIFEST_V1", "PROBLEM_V1", "PROGRESS_EVENT_V1",
    "RESOLVED_RESOURCES_V1", "RESOURCE_POLICY_V1", "STAGE_RESULT_V1", "STAGE_SPEC_V1",
    "SUBMIT_REQUEST_V1",
    "ArtifactRef", "AttemptReceipt", "AttemptState", "CancellationReceipt",
    "CapacitySample", "CheckpointCandidate", "CheckpointReceipt", "Containment",
    "EffectClass", "ExecutorMode", "JobReceipt", "JobSpec", "JobState", "LegacyFileRef",
    "LegacyInputManifest", "LiveWindow", "OutputCandidate", "Problem", "ProblemCategory",
    "ProcessIdentity", "ProcessState", "ProgressEvent", "ProgressKind", "QueueReason",
    "ResolvedResources", "ResourcePolicy", "ResourceProfile", "StageResult", "StageSpec",
    "SubmitRequest",
]
