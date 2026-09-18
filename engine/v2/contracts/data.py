"""Data-boundary contracts: table definitions, immutable handles, bounded reads.

Phase-2 guide §5 and §11; component contracts §§2.1-2.5, 5.1-5.3.

Schemas only, exactly like ``contracts/jobs.py`` and ``contracts/operations.py``:
frozen dataclasses and closed vocabularies. Nothing here computes a hash, reads
a clock, opens a file or imports ``engine`` (legacy) code. ``definition_hash``,
``*_id`` and every other identity field is a value somebody else filled in,
never one derived during construction (phase-2 guide §5.1, §5.2).

The kinds of contracts §2.5 apply here too:

* ``TableContract``, ``ColumnContract`` are Definitions — registered once,
  versioned, immutable thereafter.
* ``TableContractRef``, ``ObjectRef``, ``FragmentRef``, ``DatasetVersionRef``,
  ``SnapshotRef``, ``EventRef`` are Handles — cheap pinned pointers, never
  carrying row payloads.
* ``FragmentRecord``, ``DatasetManifest``, ``EarningsEvent``, ``ChainSnapshot``,
  ``DependencyPlan`` are Records — durable, content-hashed, replayable.
* ``KeyPredicate``, ``TimeInterval``, ``DataQuery``, ``ChainQuery``,
  ``SnapshotImportRequest``, ``LegacyMaterializationRequest`` are Commands.
* ``SnapshotImportReceipt`` is a Receipt, retained even when import failed.

``LegacyFileRef`` (source capture handle) and ``Problem`` (the shared failure
envelope) are reused from ``contracts.jobs``/``contracts.operations`` rather
than redefined here (phase-2 guide §5.6). ``ArtifactRef`` is untouched — it
remains the Phase 1 artifact handle; ``ObjectRef`` is Phase 2's separate,
deliberately path-less immutable-object handle (phase-2 guide §5.2).

``engine/v2/data/documents.py`` is the only place that performs the extra
document checks (hash/timestamp/date format, table-contract and query
structural rules) that ``engine.v2.foundation.typed`` cannot express from
annotations alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from engine.v2.contracts.jobs import LegacyFileRef
from engine.v2.contracts.operations import Problem

__all__ = [
    "CHAIN_MEMBER_V1",
    "CHAIN_QUERY_V1",
    "CHAIN_SNAPSHOT_V1",
    "COLUMN_CONTRACT_V1",
    "CONTRACT_ID_V1",
    "DATA_FAILURE_CODES",
    "DATA_QUERY_V1",
    "DATASET_MANIFEST_V1",
    "DATASET_VERSION_REF_V1",
    "DEPENDENCY_PLAN_V1",
    "EARNINGS_EVENT_V1",
    "EVENT_REF_V1",
    "FRAGMENT_RECORD_V1",
    "FRAGMENT_REF_V1",
    "KEY_PREDICATE_V1",
    "LEGACY_MATERIALIZATION_REQUEST_V1",
    "OBJECT_REF_V1",
    "PRICE_QUERY_V1",
    "PRICE_SERIES_ROW_V1",
    "ROLLBACK_RECEIPT_V1",
    "SNAPSHOT_IMPORT_RECEIPT_V1",
    "SNAPSHOT_IMPORT_REQUEST_V1",
    "SNAPSHOT_REF_V1",
    "TABLE_CONTRACT_REF_V1",
    "TABLE_CONTRACT_V1",
    "TIME_INTERVAL_V1",
    "ChainMember",
    "ChainQuery",
    "ChainSnapshot",
    "ColumnContract",
    "ContractId",
    "DataQuery",
    "DatasetManifest",
    "DatasetVersionRef",
    "DependencyEntry",
    "DependencyPlan",
    "EarningsEvent",
    "EventRef",
    "FragmentRecord",
    "FragmentRef",
    "KeyPredicate",
    "KnowledgeMode",
    "LegacyMaterializationRequest",
    "ObjectRef",
    "PriceQuery",
    "PriceSeriesRow",
    "RollbackReceipt",
    "SnapshotImportReceipt",
    "SnapshotImportRequest",
    "SnapshotRef",
    "TableContract",
    "TableContractRef",
    "TimeInterval",
]

COLUMN_CONTRACT_V1 = "column_contract.v1.0"
TABLE_CONTRACT_V1 = "table_contract.v1.0"
TABLE_CONTRACT_REF_V1 = "table_contract_ref.v1.0"
OBJECT_REF_V1 = "object_ref.v1.0"
FRAGMENT_REF_V1 = "fragment_ref.v1.0"
FRAGMENT_RECORD_V1 = "fragment_record.v1.0"
DATASET_VERSION_REF_V1 = "dataset_version_ref.v1.0"
DATASET_MANIFEST_V1 = "dataset_manifest.v1.1"
SNAPSHOT_REF_V1 = "snapshot_ref.v1.0"
KEY_PREDICATE_V1 = "key_predicate.v1.0"
TIME_INTERVAL_V1 = "time_interval.v1.0"
DATA_QUERY_V1 = "data_query.v1.0"
EVENT_REF_V1 = "event_ref.v1.0"
#: v1.1 (task 2 review fix): session/session_source became nullable — a
#: legacy row whose session was never determined maps to None, never "" — a
#: nullable addition, so the minor version bumps rather than the major one.
EARNINGS_EVENT_V1 = "earnings_event.v1.1"
CONTRACT_ID_V1 = "contract_id.v1.0"
CHAIN_QUERY_V1 = "chain_query.v1.0"
CHAIN_MEMBER_V1 = "chain_member.v1.0"
CHAIN_SNAPSHOT_V1 = "chain_snapshot.v1.0"
DEPENDENCY_PLAN_V1 = "dependency_plan.v1.0"
SNAPSHOT_IMPORT_REQUEST_V1 = "snapshot_import_request.v1.0"
SNAPSHOT_IMPORT_RECEIPT_V1 = "snapshot_import_receipt.v1.0"
LEGACY_MATERIALIZATION_REQUEST_V1 = "legacy_materialization_request.v1.0"
#: task P2-C01 (Phase 2 review closeout, decision 5): the strict, generation-
#: bearing rollback document the evidence validator requires for D16. No
#: durable rollback receipt CONTRACT existed before this -- see
#: ``RollbackReceipt``'s docstring for the producer gap this leaves open.
ROLLBACK_RECEIPT_V1 = "rollback_receipt.v1.0"
#: task brief 2026-09-14: a bounded, decision-time-eligible price-history
#: request for one ticker, modeled on ``ChainQuery``.
PRICE_QUERY_V1 = "price_query.v1.0"
PRICE_SERIES_ROW_V1 = "price_series_row.v1.0"

#: component contracts §5.1: recorded per table, never per snapshot, because
#: the risk it describes (availability vs. vintage) is a property of the field.
KnowledgeMode = Literal["observed", "attested_stable", "reconstructed"]

#: phase-2 guide §11 -> (contracts §2.4 category, retryable by default). A
#: default only, exactly like ``contracts.operations.FAILURE_CODES``: a code
#: absent here is refused at construction rather than guessed.
DATA_FAILURE_CODES: dict[str, tuple[str, bool]] = {
    "SNAPSHOT_NOT_FOUND": ("dependency", False),
    "SNAPSHOT_NOT_READY": ("dependency", True),
    "SNAPSHOT_CONFLICT": ("dependency", True),
    "CONTRACT_MISMATCH": ("validation", False),
    "QUERY_NOT_BOUNDED": ("validation", False),
    "RESULT_LIMIT_EXCEEDED": ("resource", False),
    "INPUT_CHANGED": ("integrity", True),
    "OBJECT_CORRUPT": ("integrity", False),
    "MANIFEST_CORRUPT": ("integrity", False),
    "IDENTITY_CONFLICT": ("validation", False),
    "UNSUPPORTED_CONTRACT": ("validation", False),
    # Task 2 review fix (P2-4): stable codes for three refusals that
    # previously reused a more generic one.
    "EVENT_NOT_FOUND": ("dependency", False),
    "DEADLINE_EXCEEDED": ("resource", True),
    "POPULATION_COLLAPSED": ("validation", False),
    # P2-6: stable codes for the two structural refusals a legacy
    # materialization's dest_root can trigger before any byte is written.
    "DEST_ROOT_NOT_EMPTY": ("validation", False),
    "DEST_ROOT_UNSAFE": ("validation", False),
    # P2-6 review fix: trades is scanned whole, so its real (ticker, year)
    # span can escape a too-narrow evidence_scope; this is the stable code
    # for that refusal, distinct from the generic validation codes above.
    "EVIDENCE_SCOPE_INCOMPLETE": ("validation", False),
    # P2-6 review round 4: a pinned Tier-4 serving-model cache ref whose
    # filename's own embedded panel-hash prefix does not match this
    # materialization's actual panel object -- a stale ref from a different
    # snapshot, caught before it is ever copied in.
    "TIER4_CACHE_STALE": ("validation", False),
    # engine/v2/data/repository.py:585
    "STALE_EXPECTATION": ("validation", False),
}


# --------------------------------------------------------------------------
# §5.1 table definitions
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ColumnContract:
    """One column of a ``TableContract`` (phase-2 guide §5.1).

    ``unit``/``scale``/``adjustment_basis``/``timezone`` and the policy fields
    are column-type-conditional: a text column carries none of them. The guide
    does not enumerate a closed physical-type or policy vocabulary, so those
    stay open strings here rather than an invented ``Literal`` (see this
    package's task report for that judgement call).
    """

    name: str
    physical_type: str
    nullable: bool
    unit: str | None = None
    scale: str | None = None
    adjustment_basis: str | None = None
    timezone: str | None = None
    null_policy: str | None = None
    sentinel_policy: str | None = None
    #: ``(minimum, maximum)`` by convention when present; kept as the
    #: decoder's variadic ``tuple[float, ...]`` rather than a fixed-length
    #: pair, since ``engine.v2.foundation.typed`` only supports the former.
    allowed_range: tuple[float, ...] | None = None
    observation_time_semantics: str | None = None
    schema_version: str = COLUMN_CONTRACT_V1


@dataclass(frozen=True, kw_only=True)
class TableContract:
    """A registered, versioned table shape (phase-2 guide §5.1).

    ``definition_hash`` is computed by the registration function over the
    canonical payload excluding this field; it is never derived here.
    """

    contract_id: str
    definition_hash: str
    table_name: str
    semantic_version: str
    columns: tuple[ColumnContract, ...]
    primary_key: tuple[str, ...]
    duplicate_policy: str
    foreign_keys: tuple[str, ...]
    partition_columns: tuple[str, ...]
    filterable_columns: tuple[str, ...]
    orderable_columns: tuple[str, ...]
    observation_time_column: str | None = None
    publication_time_column: str | None = None
    receipt_time_column: str | None = None
    finality_semantics: str
    provenance_semantics: str
    coverage_semantics: str
    schema_evolution_policy: str
    maximum_batch_rows: int
    maximum_result_rows: int
    legacy_mapping_ref: str | None = None
    schema_version: str = TABLE_CONTRACT_V1


@dataclass(frozen=True, kw_only=True)
class TableContractRef:
    """The cheap pinned handle to one ``TableContract`` (phase-2 guide §5.1)."""

    contract_id: str
    definition_hash: str
    schema_version: str = TABLE_CONTRACT_REF_V1


# --------------------------------------------------------------------------
# §5.2 object, fragment, version, and snapshot handles
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ObjectRef:
    """An immutable, content-addressed object; storage location is internal.

    Deliberately separate from ``contracts.jobs.ArtifactRef`` (phase-2 guide
    §5.2): that handle keeps its Phase 1 identity and is not rewritten here.
    """

    kind: str
    object_id: str
    content_hash: str
    byte_size: int
    schema_version: str = OBJECT_REF_V1


@dataclass(frozen=True, kw_only=True)
class FragmentRef:
    """A cheap pinned pointer to one ``FragmentRecord`` (phase-2 guide §5.2)."""

    fragment_id: str
    manifest_hash: str
    schema_version: str = FRAGMENT_REF_V1


@dataclass(frozen=True, kw_only=True)
class FragmentRecord:
    """Row metadata and provenance for one immutable object (phase-2 guide §5.2).

    ``primary_key_min``/``primary_key_max`` carry the canonical scalar tuple in
    contract column order; no floats, matching ``KeyPredicate.values`` (exact
    strikes and decimals are exact strings, never binary floats).
    """

    fragment_id: str
    manifest_hash: str
    object_ref: ObjectRef
    table_contract_ref: TableContractRef
    partition_key: str
    row_count: int
    byte_hash: str
    logical_content_hash: str
    primary_key_min: tuple[str | int | bool, ...]
    primary_key_max: tuple[str | int | bool, ...]
    time_min: str | None = None
    time_max: str | None = None
    input_receipt_refs: tuple[str, ...]
    import_request_hash: str
    schema_version: str = FRAGMENT_RECORD_V1


@dataclass(frozen=True, kw_only=True)
class DatasetVersionRef:
    """The cheap pinned handle to one ``DatasetManifest`` (phase-2 guide §5.2)."""

    dataset_version_id: str
    table_contract_ref: TableContractRef
    manifest_hash: str
    schema_version: str = DATASET_VERSION_REF_V1


@dataclass(frozen=True, kw_only=True)
class DatasetManifest:
    """A complete, ordered fragment membership for one dataset version.

    Readers never follow a parent chain to reconstruct membership (phase-2
    guide §6 invariant 4): every dataset version is a complete logical view.

    ``partition_logical_hashes`` (v1.1, task 7a): a *multi*-fragment
    partition's key to its streamed ``logical_rows.v1`` hash — nullable-style,
    default ``{}``, since a single-fragment partition's entry is implicit (its
    own ``logical_content_hash``). In ``manifest_hash`` but not
    ``dataset_version_id``, whose identity stays fragment membership alone.
    """

    dataset_version_ref: DatasetVersionRef
    logical_content_hash: str
    row_count: int
    parent_dataset_version_id: str | None = None
    fragment_refs: tuple[FragmentRef, ...]
    coverage_receipt_refs: tuple[str, ...]
    knowledge_mode: KnowledgeMode
    availability_evidence_refs: tuple[str, ...]
    partition_logical_hashes: dict[str, str] = field(default_factory=dict)
    schema_version: str = DATASET_MANIFEST_V1


@dataclass(frozen=True, kw_only=True)
class SnapshotRef:
    """One immutable, exactly resolvable snapshot (phase-2 guide §5.2, §3.3).

    ``knowledge_mode_by_table`` — not a single ``knowledge_mode`` — per the
    §3.3 correction to the abbreviated component-contracts §5.1 example.
    """

    snapshot_id: str
    manifest_hash: str
    parent_snapshot_id: str | None = None
    table_versions: dict[str, DatasetVersionRef]
    calendar_version: str
    source_priority_version: str
    finality_receipt_refs: tuple[str, ...]
    knowledge_mode_by_table: dict[str, KnowledgeMode]
    schema_version: str = SNAPSHOT_REF_V1


# --------------------------------------------------------------------------
# §5.3 bounded query contracts
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class KeyPredicate:
    """One equality or membership filter over one declared column.

    ``values`` excludes ``float``: decimals and strikes are exact strings at
    the interface (component contracts §2.1), and a binary float would give
    the same membership two different command identities depending on
    encoding. Recorded as this package's own judgement call — the phase-2
    guide gives ``ordered tuple[canonical scalar]`` without naming the exact
    Python types.
    """

    column: str
    operator: Literal["eq", "in"]
    values: tuple[str | int | bool, ...]
    schema_version: str = KEY_PREDICATE_V1


@dataclass(frozen=True, kw_only=True)
class TimeInterval:
    """A half-open bound over one declared time column."""

    column: str
    start_inclusive: str | None = None
    end_exclusive: str | None = None
    schema_version: str = TIME_INTERVAL_V1


@dataclass(frozen=True, kw_only=True)
class DataQuery:
    """One bounded scan request (phase-2 guide §5.3).

    Both row limits are required finite positive integers in Phase 2; there is
    no unbounded-streaming mode yet. ``deadline`` is execution metadata and
    does not enter query identity (it is still part of this Command's payload
    for a worker to honor, but callers must not fold it into a cache key).
    """

    snapshot_id: str
    table_contract_ref: TableContractRef
    columns: tuple[str, ...]
    key_filter: tuple[KeyPredicate, ...]
    time_interval: TimeInterval | None = None
    order_by: tuple[str, ...]
    max_batch_rows: int
    max_result_rows: int
    deadline: str | None = None
    schema_version: str = DATA_QUERY_V1


# --------------------------------------------------------------------------
# §5.4 event, contract, and chain contracts
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class EventRef:
    """A stable event ID plus calendar revision; ticker/date are attributes."""

    event_id: str
    calendar_revision: str
    schema_version: str = EVENT_REF_V1


@dataclass(frozen=True, kw_only=True)
class EarningsEvent:
    """One earnings event as known under one calendar revision (§5.4).

    ``session``/``conflict_status`` stay open strings rather than an invented
    closed vocabulary (see this package's task report). A date move updates
    ``event_ref.calendar_revision`` and ``supersedes_revision`` rather than
    renaming the event. ``session``/``session_source`` are nullable as of
    v1.1 (task 2 review fix): a legacy row whose session was never
    determined maps to ``None``, never the empty string.
    """

    event_ref: EventRef
    security_id: str
    ticker_at_event: str
    scheduled_event_date: str
    session: str | None = None
    actual_announcement_at: str | None = None
    session_source: str | None = None
    confidence: float
    conflict_status: str
    known_from: str | None = None
    supersedes_revision: str | None = None
    schema_version: str = EARNINGS_EVENT_V1


@dataclass(frozen=True, kw_only=True)
class ContractId:
    """A stable option identity: internal ID plus vendor symbology (§5.4).

    ``exact_strike`` and ``multiplier`` are exact decimal strings, never
    display-rounded (component contracts §2.1, phase-2 guide §5.4).
    """

    contract_id: str
    security_id: str
    vendor_mappings: dict[str, str]
    expiry: str
    right: str
    exact_strike: str
    multiplier: str
    adjustment_identity: str
    schema_version: str = CONTRACT_ID_V1


@dataclass(frozen=True, kw_only=True)
class ChainQuery:
    """A bounded, decision-time-eligible option-chain request (§5.4, §8.3)."""

    event_ref: EventRef | None = None
    security_id: str
    observation_ceiling: str
    session_date: str
    expiry_interval: TimeInterval | None = None
    quote_policy_ref: str
    max_contracts: int
    schema_version: str = CHAIN_QUERY_V1


@dataclass(frozen=True, kw_only=True)
class PriceQuery:
    """A bounded, decision-time-eligible ``price_history`` request for one
    ticker (task brief 2026-09-14, modeled on :class:`ChainQuery`).

    ``session_date`` is the last date the caller may see; ``observation_ceiling``
    bounds which retrieved *versions* of a date's price may be used (the same
    per-ticker rule ``price_history.as_of_view`` implements: the latest
    version with ``retrieved_at <= observation_ceiling``, or, if none exists,
    the ticker's earliest retrieval). ``lookback_sessions`` bounds how many
    trading sessions strictly before (and including) ``session_date`` a
    caller may request — a query planner's cap, not a guarantee that many
    rows exist.
    """

    ticker: str
    session_date: str
    observation_ceiling: str
    lookback_sessions: int
    schema_version: str = PRICE_QUERY_V1


@dataclass(frozen=True, kw_only=True)
class PriceSeriesRow:
    """One resolved ``(ticker, date)`` price, with the provenance it came
    from (task brief 2026-09-14)."""

    date: str
    close_adj: float | None
    close_raw: float | None
    high_raw: float | None
    retrieved_at: str
    source_hash: str
    schema_version: str = PRICE_SERIES_ROW_V1


@dataclass(frozen=True, kw_only=True)
class ChainMember:
    """One contract row of a resolved chain, with null-with-reason liquidity.

    Never-collected size is null with ``missing_reason``, never zero
    (component contracts §5.2).
    """

    contract_id: ContractId
    quote_observation_id: str | None = None
    bid: str | None = None
    ask: str | None = None
    mid: str | None = None
    iv: float | None = None
    delta: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    source: str
    source_row_ref: str
    availability_status: str
    missing_reason: str | None = None
    quality_flags: tuple[str, ...]
    schema_version: str = CHAIN_MEMBER_V1


@dataclass(frozen=True, kw_only=True)
class ChainSnapshot:
    """A resolved chain: expected/supported/returned populations kept apart.

    ``spot_unadjusted``/``spot_adjusted`` are the guide's abbreviated "spot
    fields" (phase-2 guide §5.4), kept distinct per component contracts §2.1:
    adjusted and unadjusted spot are never interchangeable.
    """

    chain_id: str
    source_snapshot_ref: str
    security_id: str
    observed_at: str
    available_at: str | None = None
    received_at: str | None = None
    session_date: str
    quote_policy_ref: str
    spot_unadjusted: str | None = None
    spot_adjusted: str | None = None
    rows: tuple[ChainMember, ...]
    expected_contracts: int
    supported_contracts: int
    returned_contracts: int
    coverage_ref: str
    knowledge_mode: KnowledgeMode
    schema_version: str = CHAIN_SNAPSHOT_V1


@dataclass(frozen=True, kw_only=True)
class DependencyEntry:
    """One resolved dependency of a ``DataQuery``/``ChainQuery`` (§5.4).

    Not separately versioned: the phase-2 guide's §5.4 constant list names
    only ``DEPENDENCY_PLAN_V1``, so this member carries no ``schema_version``
    of its own, the same way ``ColumnContract`` is versioned only because
    ``COLUMN_CONTRACT_V1`` is explicitly given.
    """

    table_name: str
    dataset_version_ref: DatasetVersionRef
    fragment_ref: FragmentRef
    columns: tuple[str, ...]
    predicates: tuple[KeyPredicate, ...]
    estimated_rows: int
    maximum_rows: int


@dataclass(frozen=True, kw_only=True)
class DependencyPlan:
    """What a query would touch: exact snapshot, dataset versions, fragments.

    Query-only in Phase 2 (phase-2 guide §5.5): recipe dependency graphs and
    change invalidation are later phases and must return
    ``UNSUPPORTED_CONTRACT`` instead of a guessed plan.
    """

    request_hash: str
    snapshot_ref: SnapshotRef
    dependencies: tuple[DependencyEntry, ...]
    schema_version: str = DEPENDENCY_PLAN_V1


# --------------------------------------------------------------------------
# §5.6 import and compatibility contracts
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SnapshotImportRequest:
    """The frozen import plan submitted through the Phase 1 supervisor (§7.1)."""

    scope: str
    source_manifest_ref: str
    source_manifest_hash: str
    table_sources: dict[str, tuple[LegacyFileRef, ...]]
    table_contract_refs: dict[str, TableContractRef]
    legacy_snapshot_source_ref: LegacyFileRef
    calendar_version: str
    source_priority_version: str
    finality_receipt_refs: tuple[str, ...]
    knowledge_mode_by_table: dict[str, KnowledgeMode]
    expected_head_snapshot_id: str | None = None
    expected_head_generation: int
    schema_version: str = SNAPSHOT_IMPORT_REQUEST_V1


@dataclass(frozen=True, kw_only=True)
class SnapshotImportReceipt:
    """Evidence of one import attempt, retained even when it failed (§7.3)."""

    receipt_id: str
    request_hash: str
    attempt_id: str
    fence: int
    snapshot_ref: SnapshotRef | None = None
    legacy_snapshot_object_ref: ObjectRef | None = None
    prior_head_snapshot_id: str | None = None
    resulting_head_snapshot_id: str | None = None
    resulting_head_generation: int | None = None
    status: Literal["committed", "failed", "conflict"]
    problem: Problem | None = None
    envelope: dict[str, Any]
    schema_version: str = SNAPSHOT_IMPORT_RECEIPT_V1


@dataclass(frozen=True, kw_only=True)
class RollbackReceipt:
    """Evidence of one real head rollback (phase-2 guide §10 point 5).

    A real rollback is a compare-and-swap to a PREVIOUSLY COMMITTED snapshot
    (``catalog.move_head``), never a delete or a manifest rewrite: this
    receipt exists to make that provable to a reader who only has the
    receipt, not the catalog DB -- ``prior_generation``/``resulting_generation``
    are the catalog head's own generation counter immediately before and
    after the compare-and-swap, and ``resulting_generation`` is always
    STRICTLY GREATER (every head move, including a rollback to older data,
    is itself a new catalog event). ``prior_snapshot_id``/
    ``resulting_snapshot_id`` are cross-checked by the evidence validator
    against ``import_receipt_refs`` (or lineage), never trusted alone.

    Defined for task P2-C01 (Phase 2 review closeout, decision 5): no strict
    rollback contract existed before this. ``engine/v2/ops/snapshot_promotion.py``
    ``rollback()`` (out of this task's edit scope) currently publishes an
    UNTYPED ``snapshot_update_receipt.v1.0`` dict -- ``action``, ``scope``,
    ``from_snapshot_id``, ``to_snapshot_id``, ``at`` -- with no generation
    numbers at all. That producer must be updated to additionally emit (or
    extend its own document into) this shape before D16 evidence can pass
    the validator's strict decode; noted for whoever owns that file.
    """

    receipt_id: str
    scope: str
    prior_snapshot_id: str
    resulting_snapshot_id: str
    prior_generation: int
    resulting_generation: int
    at: str
    schema_version: str = ROLLBACK_RECEIPT_V1


@dataclass(frozen=True, kw_only=True)
class LegacyMaterializationRequest:
    """The request behind one private, bounded legacy compatibility root (§9.1).

    ``direct_scope``/``evidence_scope`` stay open ``dict[str, Any]`` bags
    (e.g. ``{"tickers": [...], "years": [...]}``) rather than a new structured
    type: the phase-2 guide names their contents in prose only. Recorded as a
    judgement call.

    ``observation_ceiling`` (SEND-BACK 2026-09-14 item 2) is the job's own
    decision cutoff -- the nightly plan's session, end-of-day -- pinned into
    the plan itself so materialization can never see a retrieval made after
    this job's cutoff, even once a later capture sits in the same pinned
    snapshot version. Replaces the earlier
    ``PRICE_SERIES_MATERIALIZATION_CEILING = "9999-12-31T23:59:59Z"``
    constant, which let materialization see every retrieval ever captured.
    """

    request_hash: str
    snapshot_ref: SnapshotRef
    legacy_snapshot_object_ref: ObjectRef
    direct_scope: dict[str, Any]
    evidence_scope: dict[str, Any]
    table_queries: dict[str, DataQuery]
    registry_and_model_refs: tuple[str, ...]
    calendar_refs: tuple[str, ...]
    legacy_layout_version: str
    expected_population: dict[str, int]
    observation_ceiling: str
    schema_version: str = LEGACY_MATERIALIZATION_REQUEST_V1
