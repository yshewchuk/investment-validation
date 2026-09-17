"""Phase 3B incremental-ingestion contracts: schemas only, no I/O or logic.

The records here extend the Phase 2 immutable repository contracts without
introducing a second store. Producers retain raw acquisition evidence,
describe completed coverage against an explicit denominator, and publish the
logical changes and conservative dependency impact of a candidate version.

Cross-field rules that annotations cannot express are exercised by the
dedicated Phase 3B acceptance checker. The production data owner will adopt
those rules when it implements ingestion in later Phase 3B slices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from engine.v2.contracts.data import (
    DatasetVersionRef,
    ObjectRef,
    TableContractRef,
    TimeInterval,
)
from engine.v2.contracts.operations import Problem

__all__ = [
    "ACQUISITION_RECEIPT_V1",
    "CHANGESET_V1",
    "COMPLETED_COVERAGE_V1",
    "COVERAGE_KEY_V1",
    "COVERAGE_OUTCOME_V1",
    "DEPENDENCY_IMPACT_V1",
    "REVISION_CANDIDATE_V1",
    "ROW_CHANGE_V1",
    "AcquisitionReceipt",
    "ChangeSet",
    "CompletedCoverage",
    "CoverageKey",
    "CoverageOutcome",
    "DependencyImpact",
    "RevisionCandidate",
    "RowChange",
]

REVISION_CANDIDATE_V1 = "revision_candidate.v1.0"
ACQUISITION_RECEIPT_V1 = "acquisition_receipt.v1.0"
COVERAGE_KEY_V1 = "coverage_key.v1.0"
COVERAGE_OUTCOME_V1 = "coverage_outcome.v1.0"
COMPLETED_COVERAGE_V1 = "completed_coverage.v1.0"
ROW_CHANGE_V1 = "row_change.v1.0"
DEPENDENCY_IMPACT_V1 = "dependency_impact.v1.0"
CHANGESET_V1 = "changeset.v1.0"


@dataclass(frozen=True, kw_only=True)
class RevisionCandidate:
    """One retained provider revision for one requested logical item.

    A lower source_priority wins. Within a source priority, final wins over
    provisional and a larger provider revision_ordinal wins. received_at and
    revision_id only break ties whose content hashes agree; equal-ranked
    differing content is a conflict, never a silent pick.
    """

    revision_id: str
    logical_key: str
    source: str
    source_priority: int
    finality: Literal["provisional", "final"]
    revision_ordinal: int
    received_at: str
    content_hash: str
    supersedes_revision_id: str | None = None
    schema_version: str = REVISION_CANDIDATE_V1


@dataclass(frozen=True, kw_only=True)
class AcquisitionReceipt:
    """Redacted evidence for one cache-first provider acquisition attempt."""

    receipt_id: str
    request_hash: str
    source: str
    endpoint: str
    requested_at: str
    completed_at: str | None
    redacted_request_ref: str
    raw_object_ref: ObjectRef | None
    response_status: int | None
    outcome: Literal[
        "complete", "partial", "failed", "auth_failed", "rate_limited", "delayed"
    ]
    requested_items: tuple[str, ...]
    returned_items: tuple[str, ...]
    legitimate_empty_items: tuple[str, ...]
    unavailable_items: tuple[str, ...]
    revisions: tuple[RevisionCandidate, ...]
    selected_revision_ids: dict[str, str]
    quota_headers: dict[str, str]
    problem: Problem | None = None
    schema_version: str = ACQUISITION_RECEIPT_V1


@dataclass(frozen=True, kw_only=True)
class CoverageKey:
    """One member of an EOD coverage denominator."""

    item_key: str
    session_date: str
    ticker: str | None = None
    contract_id: str | None = None
    schema_version: str = COVERAGE_KEY_V1


@dataclass(frozen=True, kw_only=True)
class CoverageOutcome:
    """The classified result for one expected coverage member."""

    key: CoverageKey
    status: Literal["present", "legitimate_empty", "unsupported"]
    receipt_id: str
    revision_id: str | None
    finality: Literal["provisional", "final"]
    schema_version: str = COVERAGE_OUTCOME_V1


@dataclass(frozen=True, kw_only=True)
class CompletedCoverage:
    """Coverage over an explicit set, never an inferred maximum date."""

    coverage_id: str
    table_contract_ref: TableContractRef
    source: str
    endpoint: str
    interval: TimeInterval
    expected: tuple[CoverageKey, ...]
    outcomes: tuple[CoverageOutcome, ...]
    covered_tickers: tuple[str, ...]
    acquisition_receipt_refs: tuple[str, ...]
    state: Literal["incomplete", "complete"]
    completed_at: str | None
    prior_coverage_id: str | None = None
    schema_version: str = COMPLETED_COVERAGE_V1


@dataclass(frozen=True, kw_only=True)
class RowChange:
    """One logical key change emitted by an incremental merge."""

    logical_key: str
    partition_key: str
    columns: tuple[str, ...]
    time_range: TimeInterval | None
    old_hash: str | None
    new_hash: str | None
    revision_kind: Literal["append", "correction", "tombstone", "schema_change"]
    revision_id: str
    schema_version: str = ROW_CHANGE_V1


@dataclass(frozen=True, kw_only=True)
class DependencyImpact:
    """Conservative invalidation for one registered downstream dependency."""

    dependency_id: str
    scope: Literal["keys", "suffix", "full"]
    affected_keys: tuple[str, ...]
    time_range: TimeInterval | None = None
    schema_version: str = DEPENDENCY_IMPACT_V1


@dataclass(frozen=True, kw_only=True)
class ChangeSet:
    """The immutable delta between two Phase 2 dataset versions."""

    changeset_id: str
    table_contract_ref: TableContractRef
    base_dataset_version_ref: DatasetVersionRef
    result_dataset_version_ref: DatasetVersionRef
    acquisition_receipt_refs: tuple[str, ...]
    coverage_receipt_refs: tuple[str, ...]
    changes: tuple[RowChange, ...]
    changed_partitions: tuple[str, ...]
    dependency_impacts: tuple[DependencyImpact, ...]
    unknown_dependencies: tuple[str, ...]
    dependency_disposition: Literal["exact", "conservative_full", "refused"]
    outcome: Literal["noop", "changed", "refused"]
    normalized_payloads: int
    rewritten_partitions: int
    schema_version: str = CHANGESET_V1
