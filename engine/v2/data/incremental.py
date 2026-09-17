"""Incremental daily_market ingestion over the Phase 2 repository.

The pure merge is independent from storage order. It resolves retained and
incoming revisions by source priority, finality, and provider revision number,
refuses equal-rank conflicting content, and compares the resulting logical
rows with the parent snapshot.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal, Mapping, Sequence

import pyarrow as pa
from pyarrow import parquet as pq

from engine.v2.contracts import (
    ArtifactRef,
    ChangeSet,
    CompletedCoverage,
    CoverageKey,
    CoverageOutcome,
    ObjectRef,
    RevisionCandidate,
    RowChange,
    TableContract,
    TableContractRef,
    TimeInterval,
)
from engine.v2.data import (
    catalog,
    errors,
    generic_incremental,
    incremental_tables,
    manifests,
    objects,
    repository,
)
from engine.v2.foundation import (
    CONTENT_HASH_PREFIX,
    ArtifactStore,
    SystemClock,
    artifact_reference,
    canonical_json,
    content_hash,
    format_timestamp,
    from_document,
    to_document,
)

__all__ = [
    "DailyMarketCandidate",
    "DailyMarketMerge",
    "DailyMarketRevision",
    "NormalizationRecord",
    "RawPayload",
    "RawReceiptRecord",
    "build_completed_coverage",
    "build_daily_market_candidate",
    "cache_normalization",
    "cache_raw_receipt",
    "commit_daily_market_candidate",
    "run_incremental_refresh",
    "daily_market_logical_key",
    "load_daily_market_rows",
    "merge_daily_market",
    "revision_content_hash",
    "select_revision_winners",
]

RAW_SCHEMA_REF = "incremental_raw.v1"
NORMALIZED_SCHEMA_REF = "daily_market_revisions.v1"
TABLE_NAME = "daily_market"
MAX_STAGED_INPUT_BYTES = 64 << 20
MAX_STAGED_PAYLOADS = 4096
MAX_RAW_PAYLOAD_BYTES = 32 << 20
_CATALOG_FAULT_POINTS = frozenset({
    "before_transaction", "after_contracts", "after_objects", "after_fragments",
    "after_dataset_versions", "after_memberships", "after_snapshot",
    "after_snapshot_tables", "before_head_update", "before_commit",
})


@dataclass(frozen=True, kw_only=True)
class RawPayload:
    payload: bytes
    response_kind: Literal[
        "complete", "legitimate_empty", "partial", "failed", "auth_failed",
        "rate_limited", "delayed", "unavailable",
    ]
    response_meta: Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class RawReceiptRecord:
    raw_receipt_id: str
    source: str
    endpoint: str
    request_hash: str
    raw_hash: str
    object_ref: ObjectRef
    response_kind: str
    request: Mapping[str, Any]
    response_meta: Mapping[str, Any]
    received_at: str
    cache_hit: bool = False


@dataclass(frozen=True, kw_only=True)
class DailyMarketRevision:
    candidate: RevisionCandidate
    ticker: str
    session_date: str
    row: Mapping[str, Any] | None
    deleted: bool
    raw_receipt_id: str
    normalization_id: str


@dataclass(frozen=True, kw_only=True)
class NormalizationRecord:
    normalization_id: str
    raw_hash: str
    normalizer_id: str
    contract_id: str
    normalized_hash: str
    object_ref: ObjectRef
    revisions: tuple[DailyMarketRevision, ...]
    cache_hit: bool


@dataclass(frozen=True, kw_only=True)
class DailyMarketMerge:
    prior_rows: tuple[dict[str, Any], ...]
    rows: tuple[dict[str, Any], ...]
    retained_revisions: tuple[DailyMarketRevision, ...]
    incoming_revisions: tuple[DailyMarketRevision, ...]
    winners: tuple[DailyMarketRevision, ...]
    changes: tuple[RowChange, ...]
    changed_partitions: tuple[str, ...]
    partition_hashes: dict[str, tuple[str, str]]


@dataclass(frozen=True, kw_only=True)
class DailyMarketCandidate:
    parent: manifests.ResolvedSnapshot
    contract: TableContract
    merge: DailyMarketMerge
    coverage: CompletedCoverage
    table_manifest: Any
    snapshot: Any
    contracts: tuple[TableContract, ...]
    objects: tuple[ObjectRef, ...]
    records: tuple[Any, ...]
    changeset: ChangeSet
    changeset_hash: str
    rewritten_partitions: int


def daily_market_logical_key(ticker: str, session_date: str) -> str:
    return canonical_json([ticker, _session_date(session_date)])


def revision_content_hash(*, ticker: str, session_date: str,
                          row: Mapping[str, Any] | None, deleted: bool) -> str:
    payload = {
        "kind": "daily_market_revision.v1",
        "ticker": ticker,
        "session_date": _session_date(session_date),
        "deleted": deleted,
        "row": _jsonable(dict(row)) if row is not None else None,
    }
    return content_hash(payload)


def _rank_revision_group(logical_key: str,
                         group: Sequence[DailyMarketRevision]) -> DailyMarketRevision:
    priority = min(item.candidate.source_priority for item in group)
    ranked = [item for item in group if item.candidate.source_priority == priority]
    finality = max(1 if item.candidate.finality == "final" else 0 for item in ranked)
    ranked = [item for item in ranked
              if (1 if item.candidate.finality == "final" else 0) == finality]
    ordinal = max(item.candidate.revision_ordinal for item in ranked)
    ranked = [item for item in ranked if item.candidate.revision_ordinal == ordinal]
    hashes = {item.candidate.content_hash for item in ranked}
    if len(hashes) != 1:
        raise errors.fail(
            "IDENTITY_CONFLICT",
            "equal-ranked daily_market revisions have conflicting content",
            details={"logical_key": logical_key},
        )
    return max(ranked, key=lambda item: (item.candidate.received_at,
                                         item.candidate.revision_id))


def select_revision_winners(
    revisions: Sequence[DailyMarketRevision],
) -> dict[str, DailyMarketRevision]:
    unique: dict[str, DailyMarketRevision] = {}
    for revision in revisions:
        prior = unique.get(revision.candidate.revision_id)
        if prior is not None and _revision_document(prior) != _revision_document(revision):
            raise errors.fail("IDENTITY_CONFLICT", "one revision id has conflicting payloads")
        unique[revision.candidate.revision_id] = revision
    grouped: dict[str, list[DailyMarketRevision]] = {}
    for revision in unique.values():
        grouped.setdefault(revision.candidate.logical_key, []).append(revision)

    return {logical_key: _rank_revision_group(logical_key, group)
            for logical_key, group in grouped.items()}


def _apply_revision_winners(
    prior: Mapping[tuple[str, str], dict[str, Any]],
    winners: Mapping[str, DailyMarketRevision],
) -> tuple[dict[tuple[str, str], dict[str, Any]],
           dict[str, DailyMarketRevision]]:
    result = dict(prior)
    winner_by_key = {}
    for logical_key, revision in winners.items():
        key = (revision.ticker, revision.session_date)
        winner_by_key[logical_key] = revision
        if revision.deleted:
            result.pop(key, None)
        else:
            result[key] = dict(revision.row or {})
    return result, winner_by_key


def _merge_changes(
    contract: TableContract,
    prior: Mapping[tuple[str, str], dict[str, Any]],
    result: Mapping[tuple[str, str], dict[str, Any]],
    winner_by_key: Mapping[str, DailyMarketRevision],
) -> tuple[RowChange, ...]:
    changes = []
    for key in sorted(set(prior) | set(result)):
        before = prior.get(key)
        after = result.get(key)
        old_hash = _row_hash(before) if before is not None else None
        new_hash = _row_hash(after) if after is not None else None
        if old_hash == new_hash:
            continue
        logical_key = daily_market_logical_key(*key)
        revision = winner_by_key.get(logical_key)
        if revision is None:
            raise errors.fail("MANIFEST_CORRUPT",
                              "a logical change has no retained revision winner")
        kind = "append" if before is None else ("tombstone" if after is None else "correction")
        start = key[1]
        end = (date.fromisoformat(start) + timedelta(days=1)).isoformat()
        changes.append(RowChange(
            logical_key=logical_key,
            partition_key=str(date.fromisoformat(start).year),
            columns=_changed_columns(contract, before, after),
            time_range=TimeInterval(column="date", start_inclusive=start, end_exclusive=end),
            old_hash=old_hash,
            new_hash=new_hash,
            revision_kind=kind,
            revision_id=revision.candidate.revision_id,
        ))
    return tuple(changes)


def merge_daily_market(
    contract: TableContract,
    prior_rows: Sequence[Mapping[str, Any]],
    retained_revisions: Sequence[DailyMarketRevision],
    incoming_revisions: Sequence[DailyMarketRevision],
) -> DailyMarketMerge:
    _daily_contract(contract)
    prior = _index_rows(contract, prior_rows)
    retained = tuple(_validate_revision(contract, item) for item in retained_revisions)
    incoming = tuple(_validate_revision(contract, item) for item in incoming_revisions)
    winners = select_revision_winners((*retained, *incoming))
    result, winner_by_key = _apply_revision_winners(prior, winners)
    changes = _merge_changes(contract, prior, result, winner_by_key)

    changed_partitions = tuple(sorted({change.partition_key for change in changes}))
    partition_hashes = {
        partition: (
            _partition_hash(contract, partition, prior.values()),
            _partition_hash(contract, partition, result.values()),
        )
        for partition in changed_partitions
    }
    return DailyMarketMerge(
        prior_rows=tuple(prior[key] for key in sorted(prior)),
        rows=tuple(result[key] for key in sorted(result)),
        retained_revisions=retained,
        incoming_revisions=incoming,
        winners=tuple(winners[key] for key in sorted(winners)),
        changes=tuple(changes),
        changed_partitions=changed_partitions,
        partition_hashes=partition_hashes,
    )


def _coverage_outcome_order(
    expected_ids: Sequence[str],
    expected: Sequence[CoverageKey],
    outcomes: Sequence[CoverageOutcome],
) -> tuple[str, tuple[CoverageOutcome, ...]]:
    by_key: dict[str, list[CoverageOutcome]] = {}
    for outcome in outcomes:
        by_key.setdefault(_coverage_key_identity(outcome.key), []).append(outcome)
    exact = (len(set(expected_ids)) == len(expected_ids)
             and set(by_key) == set(expected_ids)
             and all(len(by_key[key]) == 1 for key in expected_ids))
    ordered = (tuple(by_key[key][0] for key in expected_ids) if exact
               else tuple(sorted(outcomes, key=lambda item: _coverage_key_sort(item.key))))
    state = "complete" if exact else "incomplete"
    if state == "complete" and any(
            outcome.status == "present" and outcome.revision_id is None
            for outcome in ordered):
        state = "incomplete"
    return state, ordered


def _covered_tickers(outcomes: Sequence[CoverageOutcome]) -> tuple[str, ...]:
    return tuple(sorted({
        outcome.key.ticker for outcome in outcomes
        if outcome.key.ticker is not None and outcome.status in (
            "present", "legitimate_empty", "unsupported")
    }))


def build_completed_coverage(
    contract_ref: TableContractRef,
    *,
    source: str,
    endpoint: str,
    interval: TimeInterval,
    expected: Sequence[CoverageKey],
    outcomes: Sequence[CoverageOutcome],
    acquisition_receipt_refs: Sequence[str],
    completed_at: str | None,
    prior_coverage_id: str | None = None,
) -> CompletedCoverage:
    expected_ordered = tuple(sorted(expected, key=_coverage_key_sort))
    expected_ids = [_coverage_key_identity(item) for item in expected_ordered]
    state, ordered_outcomes = _coverage_outcome_order(
        expected_ids, expected_ordered, outcomes)
    covered_tickers = _covered_tickers(ordered_outcomes)
    identity = {
        "kind": "completed_coverage_id.v1",
        "table_contract_ref": to_document(contract_ref),
        "source": source,
        "endpoint": endpoint,
        "interval": to_document(interval),
        "expected": [to_document(item) for item in expected_ordered],
        "outcomes": [to_document(item) for item in ordered_outcomes],
        "acquisition_receipt_refs": sorted(set(acquisition_receipt_refs)),
        "state": state,
        "prior_coverage_id": prior_coverage_id,
    }
    coverage_id = "cov_" + content_hash(identity).removeprefix(CONTENT_HASH_PREFIX)[:32]
    return CompletedCoverage(
        coverage_id=coverage_id,
        table_contract_ref=contract_ref,
        source=source,
        endpoint=endpoint,
        interval=interval,
        expected=expected_ordered,
        outcomes=ordered_outcomes,
        covered_tickers=covered_tickers,
        acquisition_receipt_refs=tuple(sorted(set(acquisition_receipt_refs))),
        state=state,
        completed_at=completed_at if state == "complete" else None,
        prior_coverage_id=prior_coverage_id,
    )


def _validate_revision(contract, revision):
    session = _session_date(revision.session_date)
    logical_key = daily_market_logical_key(revision.ticker, session)
    if revision.candidate.logical_key != logical_key:
        raise errors.fail("CONTRACT_MISMATCH", "revision logical key does not match its row key")
    if revision.deleted:
        if revision.row is not None:
            raise errors.fail("CONTRACT_MISMATCH", "a tombstone cannot carry a row")
        row = None
    else:
        if revision.row is None:
            raise errors.fail("CONTRACT_MISMATCH", "a live revision requires a row")
        row = _canonical_row(contract, revision.row)
        if (row["ticker"], _date_value(row["date"])) != (revision.ticker, session):
            raise errors.fail("CONTRACT_MISMATCH", "revision row key does not match its metadata")
    expected = revision_content_hash(
        ticker=revision.ticker, session_date=session, row=row, deleted=revision.deleted)
    if revision.candidate.content_hash != expected:
        raise errors.fail("IDENTITY_CONFLICT", "revision content hash does not match its payload")
    return dataclasses.replace(revision, session_date=session, row=row)


def _index_rows(contract, rows):
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        row = _canonical_row(contract, raw)
        key = (row["ticker"], _date_value(row["date"]))
        if key in indexed:
            raise errors.fail("IDENTITY_CONFLICT", "daily_market parent contains duplicate keys")
        indexed[key] = row
    return indexed


def _canonical_column_value(column, raw):
    if column.name not in raw:
        if not column.nullable:
            raise errors.fail("CONTRACT_MISMATCH",
                              f"daily_market row is missing {column.name}")
        value = None
    else:
        value = raw[column.name]
    if isinstance(value, float) and value != value:
        value = None
    if isinstance(value, float) and (
            value == float("inf") or value == float("-inf")):
        raise errors.fail("CONTRACT_MISMATCH", "daily_market row has a non-finite value")
    if value is None and not column.nullable:
        raise errors.fail("CONTRACT_MISMATCH",
                          f"daily_market row has null {column.name}")
    if value is not None and column.physical_type.startswith("timestamp["):
        value = _datetime_value(value)
    return value


def _canonical_row(contract, raw):
    names = {column.name for column in contract.columns}
    if set(raw) - names:
        raise errors.fail("CONTRACT_MISMATCH", "daily_market row has undeclared columns")
    result = {column.name: _canonical_column_value(column, raw)
              for column in contract.columns}
    session = _date_value(result["date"])
    if int(result["year"]) != date.fromisoformat(session).year:
        raise errors.fail("CONTRACT_MISMATCH", "daily_market year disagrees with date")
    return result


def _changed_columns(contract, before, after):
    if before is None or after is None:
        return tuple(column.name for column in contract.columns)
    return tuple(column.name for column in contract.columns
                 if _jsonable(before[column.name]) != _jsonable(after[column.name]))


def _partition_hash(contract, partition, rows):
    selected = [row for row in rows if str(row["year"]) == partition]
    selected.sort(key=lambda row: tuple(_sort_value(row[key]) for key in contract.primary_key))
    contract_ref = TableContractRef(
        contract_id=contract.contract_id, definition_hash=contract.definition_hash)
    tuples = [tuple(row[column.name] for column in contract.columns) for row in selected]
    return objects.logical_partition_hash(contract, contract_ref, partition, tuples)


def _row_hash(row):
    return content_hash({"kind": "daily_market_row.v1", "row": _jsonable(dict(row))})


def _daily_contract(contract):
    if contract.table_name != TABLE_NAME:
        raise errors.fail("UNSUPPORTED_CONTRACT", "incremental merge supports daily_market only")
    if tuple(contract.primary_key) != ("ticker", "date") or "year" not in contract.partition_columns:
        raise errors.fail("UNSUPPORTED_CONTRACT",
                          "daily_market key and year partition contract are required")


def _revision_document(revision):
    return {
        "candidate": to_document(revision.candidate),
        "ticker": revision.ticker,
        "session_date": revision.session_date,
        "row": _jsonable(dict(revision.row)) if revision.row is not None else None,
        "deleted": revision.deleted,
        "raw_receipt_id": revision.raw_receipt_id,
        "normalization_id": revision.normalization_id,
    }


def _coverage_key_identity(key):
    return content_hash(to_document(key))


def _coverage_key_sort(key):
    return (key.session_date, key.ticker or "", key.contract_id or "", key.item_key)


def _session_date(value):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(str(value)[:10]).isoformat()


def _date_value(value):
    return _session_date(value)


def _datetime_value(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed


def _sort_value(value):
    return _jsonable(value)


def _jsonable(value):
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def cache_raw_receipt(conn: Any, store: ArtifactStore, payload: RawPayload,
                      *, source: str, endpoint: str, request: Mapping[str, Any],
                      received_at: str) -> RawReceiptRecord:
    request_hash = content_hash(_jsonable(dict(request)))
    predicted = artifact_reference(payload.payload, RAW_SCHEMA_REF)
    receipt_id = "raw_" + content_hash({
        "source": source, "endpoint": endpoint, "request_hash": request_hash,
        "raw_hash": predicted.content_hash,
    }).removeprefix(CONTENT_HASH_PREFIX)[:32]
    existing = conn.execute(
        "SELECT * FROM data_raw_receipts WHERE raw_receipt_id = ?",
        (receipt_id,)).fetchone()
    if existing is not None:
        if (existing["source"], existing["endpoint"], existing["request_hash"],
                existing["raw_hash"], existing["response_kind"]) != (
                source, endpoint, request_hash, predicted.content_hash,
                payload.response_kind):
            raise errors.fail("IDENTITY_CONFLICT",
                              "raw receipt identity has conflicting content")
        object_ref = _object_ref_document(existing["artifact_ref_json"])
        store.verify(_artifact_ref(object_ref, RAW_SCHEMA_REF))
        return RawReceiptRecord(
            raw_receipt_id=existing["raw_receipt_id"], source=existing["source"],
            endpoint=existing["endpoint"], request_hash=existing["request_hash"],
            raw_hash=existing["raw_hash"], object_ref=object_ref,
            response_kind=existing["response_kind"],
            request=json.loads(existing["request_json"]),
            response_meta=json.loads(existing["response_meta_json"]),
            received_at=existing["received_at"], cache_hit=True)

    ref = store.publish_bytes(payload.payload, schema_ref=RAW_SCHEMA_REF)
    if ref != predicted:
        raise errors.fail("OBJECT_CORRUPT", "raw artifact identity changed during publication")
    object_ref = ObjectRef(kind="raw_response", object_id=ref.artifact_id,
                           content_hash=ref.content_hash, byte_size=ref.byte_size)
    record = RawReceiptRecord(
        raw_receipt_id=receipt_id, source=source, endpoint=endpoint,
        request_hash=request_hash, raw_hash=ref.content_hash, object_ref=object_ref,
        response_kind=payload.response_kind, request=dict(request),
        response_meta=dict(payload.response_meta), received_at=received_at,
        cache_hit=False)
    with _write_transaction(conn):
        conn.execute(
            "INSERT INTO data_raw_receipts (raw_receipt_id, source, endpoint, request_hash, "
            "raw_hash, artifact_ref_json, response_kind, request_json, response_meta_json, "
            "received_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (record.raw_receipt_id, record.source, record.endpoint, record.request_hash,
             record.raw_hash, canonical_json(to_document(record.object_ref)),
             record.response_kind, canonical_json(_jsonable(record.request)),
             canonical_json(_jsonable(record.response_meta)), record.received_at))
    return record


def cache_normalization(conn: Any, store: ArtifactStore,
                        raw: RawReceiptRecord, revisions: Sequence[DailyMarketRevision], *,
                        normalizer_id: str, contract_id: str, created_at: str) -> NormalizationRecord:
    ordered = tuple(sorted(revisions, key=lambda item: item.candidate.revision_id))
    document = {"schema_version": NORMALIZED_SCHEMA_REF, "raw_hash": raw.raw_hash,
                "normalizer_id": normalizer_id,
                "revisions": [_revision_document(item) for item in ordered]}
    normalized_hash = content_hash(document)
    normalization_id = "norm_" + content_hash({
        "raw_hash": raw.raw_hash, "normalizer_id": normalizer_id, "contract_id": contract_id,
    }).removeprefix(CONTENT_HASH_PREFIX)[:32]
    existing = conn.execute(
        "SELECT * FROM data_normalizations WHERE normalization_id = ?",
        (normalization_id,)).fetchone()
    if existing is not None:
        if existing["normalized_hash"] != normalized_hash:
            raise errors.fail("IDENTITY_CONFLICT",
                              "normalization identity has conflicting content")
        object_ref = _object_ref_document(existing["artifact_ref_json"])
        stored = store.read_verified(_artifact_ref(object_ref, NORMALIZED_SCHEMA_REF))
        stored_revisions = _normalized_revisions(stored)
        return NormalizationRecord(
            normalization_id=normalization_id, raw_hash=raw.raw_hash,
            normalizer_id=normalizer_id, contract_id=contract_id,
            normalized_hash=normalized_hash, object_ref=object_ref,
            revisions=stored_revisions, cache_hit=True)

    encoded = canonical_json(document).encode("utf-8")
    ref = store.publish_bytes(encoded, schema_ref=NORMALIZED_SCHEMA_REF)
    object_ref = ObjectRef(kind="normalized_daily_market", object_id=ref.artifact_id,
                           content_hash=ref.content_hash, byte_size=ref.byte_size)
    record = NormalizationRecord(
        normalization_id=normalization_id, raw_hash=raw.raw_hash,
        normalizer_id=normalizer_id, contract_id=contract_id,
        normalized_hash=normalized_hash, object_ref=object_ref,
        revisions=ordered, cache_hit=False)
    with _write_transaction(conn):
        conn.execute(
            "INSERT INTO data_normalizations (normalization_id, raw_hash, normalizer_id, "
            "contract_id, normalized_hash, artifact_ref_json, row_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (normalization_id, raw.raw_hash, normalizer_id, contract_id, normalized_hash,
             canonical_json(to_document(object_ref)), len(ordered), created_at))
    return record


def _artifact_ref(object_ref: ObjectRef, schema_ref: str) -> ArtifactRef:
    digest = object_ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
    return ArtifactRef(
        artifact_id=object_ref.object_id, content_hash=object_ref.content_hash,
        schema_ref=schema_ref, byte_size=object_ref.byte_size,
        storage_key=f"objects/{digest[:2]}/{digest}")


def _object_ref_document(text: str) -> ObjectRef:
    try:
        return from_document(ObjectRef, json.loads(text))
    except Exception as exc:
        raise errors.fail("MANIFEST_CORRUPT",
                          "cached artifact reference is malformed") from exc


def _revision_from_document(document: Mapping[str, Any], *,
                            raw_receipt_id: str | None = None,
                            normalization_id: str | None = None) -> DailyMarketRevision:
    try:
        candidate = from_document(RevisionCandidate, document["candidate"])
        return DailyMarketRevision(
            candidate=candidate, ticker=str(document["ticker"]),
            session_date=_session_date(document["session_date"]),
            row=document.get("row"), deleted=bool(document["deleted"]),
            raw_receipt_id=(raw_receipt_id if raw_receipt_id is not None
                            else str(document["raw_receipt_id"])),
            normalization_id=(normalization_id if normalization_id is not None
                              else str(document["normalization_id"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise errors.fail("CONTRACT_MISMATCH",
                          "serialized daily_market revision is malformed") from exc


def _normalized_revisions(encoded: bytes) -> tuple[DailyMarketRevision, ...]:
    try:
        document = json.loads(encoded)
        if document.get("schema_version") != NORMALIZED_SCHEMA_REF:
            raise ValueError("normalization schema")
        return tuple(_revision_from_document(item)
                     for item in document["revisions"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise errors.fail("MANIFEST_CORRUPT",
                          "cached normalization artifact is malformed") from exc


def _normalization_identity(raw_hash: str, normalizer_id: str,
                            contract_id: str) -> str:
    return "norm_" + content_hash({
        "raw_hash": raw_hash, "normalizer_id": normalizer_id,
        "contract_id": contract_id,
    }).removeprefix(CONTENT_HASH_PREFIX)[:32]


def _load_retained_revisions(conn: Any, store: ArtifactStore) \
        -> tuple[DailyMarketRevision, ...]:
    rows = conn.execute(
        "SELECT r.*, n.artifact_ref_json FROM data_daily_market_revisions r "
        "JOIN data_normalizations n ON n.normalization_id = r.normalization_id "
        "ORDER BY r.revision_id").fetchall()
    artifacts: dict[str, dict[str, DailyMarketRevision]] = {}
    retained = []
    for row in rows:
        normalization_id = row["normalization_id"]
        if normalization_id not in artifacts:
            object_ref = _object_ref_document(row["artifact_ref_json"])
            encoded = store.read_verified(
                _artifact_ref(object_ref, NORMALIZED_SCHEMA_REF))
            artifacts[normalization_id] = {
                item.candidate.revision_id: item
                for item in _normalized_revisions(encoded)
            }
        revision = artifacts[normalization_id].get(row["revision_id"])
        if revision is None:
            raise errors.fail("MANIFEST_CORRUPT",
                              "revision audit row is absent from normalization artifact")
        revision = dataclasses.replace(
            revision, raw_receipt_id=row["raw_receipt_id"],
            normalization_id=normalization_id)
        expected = (
            revision.ticker, revision.session_date, revision.candidate.source,
            revision.candidate.source_priority,
            1 if revision.candidate.finality == "final" else 0,
            revision.candidate.revision_ordinal, int(revision.deleted),
            revision.candidate.content_hash,
        )
        actual = (
            row["ticker"], row["session_date"], row["source"],
            row["source_priority"], row["finality_rank"],
            row["revision_number"], row["deleted"], row["row_hash"],
        )
        if actual != expected:
            raise errors.fail("MANIFEST_CORRUPT",
                              "revision audit row disagrees with normalization artifact")
        retained.append(revision)
    return tuple(retained)


def load_daily_market_rows(store: ArtifactStore, records: Sequence[Any],
                           contract: TableContract, *, partitions=None) -> tuple[dict[str, Any], ...]:
    rows = []
    for record in records:
        if partitions is not None and record.partition_key not in partitions:
            continue
        path = objects.verify_object_path(store, record.object_ref)
        rows.extend(_canonical_row(contract, row) for row in pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: tuple(_sort_value(row[key]) for key in contract.primary_key))
    return tuple(rows)


def _affected_daily_partitions(contract: TableContract,
                               revisions: Sequence[DailyMarketRevision]):
    if not revisions:
        return frozenset()
    partitions = set()
    for revision in revisions:
        if revision.row is not None:
            partitions.add(_row_partition(contract, revision.row))
        elif contract.partition_columns:
            partitions.add("/".join(
                revision.session_date[:4] if name == "year" else revision.session_date
                for name in contract.partition_columns))
        else:
            partitions.add("__whole__")
    return frozenset(partitions)


def _row_partition(contract: TableContract, row: Mapping[str, Any]) -> str:
    if not contract.partition_columns:
        return "__whole__"
    return "/".join(str(row[name]) for name in contract.partition_columns)


def _write_daily_partitions(store: ArtifactStore, contract: TableContract,
                            contract_ref: TableContractRef, merge: DailyMarketMerge,
                            coverage: CompletedCoverage) -> tuple[tuple[Any, ...], tuple[ObjectRef, ...]]:
    new_records, new_objects = [], []
    for partition in merge.changed_partitions:
        rows = [row for row in merge.rows if str(row["year"]) == partition]
        if not rows:
            continue
        ref = store.publish_bytes(_parquet_bytes(contract, rows),
                                  schema_ref="parquet_fragment.v1.0")
        obj = ObjectRef(kind="parquet_fragment", object_id=ref.artifact_id,
                        content_hash=ref.content_hash, byte_size=ref.byte_size)
        inspection = objects.inspect_fragment(store, obj, contract, contract_ref, partition)
        new_records.append(manifests.fragment_record(
            inspection, contract_ref,
            input_receipt_refs=tuple(coverage.acquisition_receipt_refs),
            import_request_hash=content_hash({
                "table": TABLE_NAME, "partition": partition,
                "coverage": coverage.coverage_id,
            })))
        new_objects.append(obj)
    return tuple(new_records), tuple(new_objects)


def _daily_market_manifest(prior_manifest: Any, contract_ref: TableContractRef,
                           prior_records: Sequence[Any], new_records: Sequence[Any],
                           merge: DailyMarketMerge, coverage: CompletedCoverage) -> tuple[Any, tuple[Any, ...]]:
    changed = set(merge.changed_partitions)
    kept_records = [record for record in prior_records if record.partition_key not in changed]
    records = tuple(sorted((*kept_records, *new_records),
                           key=lambda item: (item.partition_key, item.primary_key_min)))
    partition_hashes = {
        partition: digest
        for partition, digest in prior_manifest.partition_logical_hashes.items()
        if partition not in changed
    }
    manifest = manifests.dataset_manifest(
        contract_ref, records, knowledge_mode=prior_manifest.knowledge_mode,
        coverage_receipt_refs=tuple(dict.fromkeys(
            (*prior_manifest.coverage_receipt_refs, coverage.coverage_id))),
        availability_evidence_refs=prior_manifest.availability_evidence_refs,
        parent_dataset_version_id=prior_manifest.dataset_version_ref.dataset_version_id,
        partition_logical_hashes=partition_hashes)
    return manifest, records


def _candidate_changeset(snapshot: Any, prior_manifest: Any,
                         table_manifest: Any, contract_ref: TableContractRef,
                         coverage: CompletedCoverage, merge: DailyMarketMerge,
                         incoming_count: int) -> ChangeSet:
    changeset_id = "changeset_" + content_hash({
        "snapshot": snapshot.snapshot_id,
        "changes": [to_document(item) for item in merge.changes],
    }).removeprefix(CONTENT_HASH_PREFIX)[:32]
    return ChangeSet(
        changeset_id=changeset_id, table_contract_ref=contract_ref,
        base_dataset_version_ref=prior_manifest.dataset_version_ref,
        result_dataset_version_ref=table_manifest.dataset_version_ref,
        acquisition_receipt_refs=coverage.acquisition_receipt_refs,
        coverage_receipt_refs=(coverage.coverage_id,), changes=merge.changes,
        changed_partitions=merge.changed_partitions, dependency_impacts=(),
        unknown_dependencies=(), dependency_disposition="exact",
        outcome="changed" if merge.changes else "noop",
        normalized_payloads=incoming_count,
        rewritten_partitions=len(merge.changed_partitions))


def build_daily_market_candidate(parent: manifests.ResolvedSnapshot, store: ArtifactStore,
                                 incoming_revisions: Sequence[DailyMarketRevision], *,
                                 coverage: CompletedCoverage,
                                 retained_revisions: Sequence[DailyMarketRevision] = (),
                                 parent_snapshot_id: str | None = None,
                                 normalized_payloads: int | None = None) -> DailyMarketCandidate:
    if coverage.state != "complete":
        raise errors.fail("INPUT_CHANGED", "incomplete coverage cannot build a candidate")
    contract = next((item for item in parent.contracts if item.table_name == TABLE_NAME), None)
    if contract is None:
        raise errors.fail("CONTRACT_MISMATCH", "parent snapshot has no daily_market contract")
    prior_manifest = parent.table_manifests.get(TABLE_NAME)
    if prior_manifest is None:
        raise errors.fail("CONTRACT_MISMATCH", "parent snapshot has no daily_market table")
    incoming_keys = {item.candidate.logical_key for item in incoming_revisions}
    retained_revisions = tuple(item for item in retained_revisions
                               if item.candidate.logical_key in incoming_keys)
    prior_ids = {ref.fragment_id for ref in prior_manifest.fragment_refs}
    prior_records = tuple(record for record in parent.records if record.fragment_id in prior_ids)
    affected = _affected_daily_partitions(contract, (*retained_revisions, *incoming_revisions))
    prior_rows = load_daily_market_rows(store, prior_records, contract, partitions=affected)
    merge = merge_daily_market(contract, prior_rows, retained_revisions, incoming_revisions)
    contract_ref = prior_manifest.dataset_version_ref.table_contract_ref
    new_records, new_objects = _write_daily_partitions(
        store, contract, contract_ref, merge, coverage)
    table_manifest, daily_records = _daily_market_manifest(
        prior_manifest, contract_ref, prior_records, new_records, merge, coverage)
    table_manifests = dict(parent.table_manifests)
    table_manifests[TABLE_NAME] = table_manifest
    snapshot = manifests.snapshot_ref(
        table_manifests, calendar_version=parent.snapshot.calendar_version,
        source_priority_version=parent.snapshot.source_priority_version,
        finality_receipt_refs=parent.snapshot.finality_receipt_refs,
        parent_snapshot_id=parent_snapshot_id or parent.snapshot.snapshot_id)
    changed = set(merge.changed_partitions)
    replaced_ids = {record.fragment_id for record in prior_records
                    if record.partition_key in changed}
    all_records = tuple(record for record in parent.records
                        if record.fragment_id not in replaced_ids)
    all_records = tuple(sorted((*all_records, *daily_records),
                               key=lambda item: (item.table_contract_ref.contract_id,
                                                 item.partition_key, item.primary_key_min)))
    all_objects = {item.object_id: item for item in parent.objects}
    all_objects.update({item.object_id: item for item in new_objects})
    changeset = _candidate_changeset(
        snapshot, prior_manifest, table_manifest, contract_ref, coverage,
        merge, (len(incoming_revisions) if normalized_payloads is None
                else normalized_payloads))
    return DailyMarketCandidate(
        parent=parent, contract=contract, merge=merge, coverage=coverage,
        table_manifest=table_manifest, snapshot=snapshot, contracts=parent.contracts,
        objects=tuple(all_objects.values()), records=all_records, changeset=changeset,
        changeset_hash=content_hash(to_document(changeset)),
        rewritten_partitions=len(merge.changed_partitions))


def _record_candidate_references(c: Any, committed_receipt_id: str,
                                 candidate: DailyMarketCandidate, clock) -> None:
    coverage = candidate.coverage
    coverage_json = canonical_json(to_document(coverage))
    coverage_hash = content_hash(to_document(coverage))
    existing_coverage = c.execute(
        "SELECT coverage_hash, coverage_json FROM data_snapshot_coverage "
        "WHERE snapshot_id = ? AND table_name = ? AND coverage_id = ?",
        (candidate.snapshot.snapshot_id, TABLE_NAME, coverage.coverage_id)).fetchone()
    if existing_coverage is None:
        c.execute(
            "INSERT INTO data_snapshot_coverage (snapshot_id, table_name, coverage_id, "
            "import_receipt_id, coverage_hash, coverage_json) VALUES (?, ?, ?, ?, ?, ?)",
            (candidate.snapshot.snapshot_id, TABLE_NAME, coverage.coverage_id,
             committed_receipt_id, coverage_hash, coverage_json))
    elif tuple(existing_coverage) != (coverage_hash, coverage_json):
        raise errors.fail("IDENTITY_CONFLICT", "coverage identity has conflicting content")

    changeset_json = canonical_json(to_document(candidate.changeset))
    existing_changeset = c.execute(
        "SELECT changeset_hash, changeset_json FROM data_changesets "
        "WHERE changeset_id = ?", (candidate.changeset.changeset_id,)).fetchone()
    if existing_changeset is None:
        c.execute(
            "INSERT INTO data_changesets (changeset_id, snapshot_id, import_receipt_id, "
            "table_name, old_dataset_version_id, new_dataset_version_id, changeset_hash, "
            "changeset_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (candidate.changeset.changeset_id, candidate.snapshot.snapshot_id,
             committed_receipt_id, TABLE_NAME,
             candidate.changeset.base_dataset_version_ref.dataset_version_id,
             candidate.changeset.result_dataset_version_ref.dataset_version_id,
             candidate.changeset_hash, changeset_json, format_timestamp(clock.now())))
    elif tuple(existing_changeset) != (candidate.changeset_hash, changeset_json):
        raise errors.fail("IDENTITY_CONFLICT", "changeset identity has conflicting content")

    for revision in candidate.merge.incoming_revisions:
        values = (
            revision.raw_receipt_id, revision.normalization_id, revision.ticker,
            revision.session_date, revision.candidate.source,
            revision.candidate.source_priority,
            1 if revision.candidate.finality == "final" else 0,
            revision.candidate.revision_ordinal, int(revision.deleted),
            revision.candidate.content_hash)
        existing = c.execute(
            "SELECT raw_receipt_id, normalization_id, ticker, session_date, source, "
            "source_priority, finality_rank, revision_number, deleted, row_hash "
            "FROM data_daily_market_revisions WHERE revision_id = ?",
            (revision.candidate.revision_id,)).fetchone()
        if existing is None:
            c.execute(
                "INSERT INTO data_daily_market_revisions "
                "(revision_id, import_receipt_id, raw_receipt_id, normalization_id, ticker, "
                "session_date, source, source_priority, finality_rank, revision_number, "
                "deleted, row_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (revision.candidate.revision_id, committed_receipt_id, *values,
                 format_timestamp(clock.now())))
        elif tuple(existing) != values:
            raise errors.fail("IDENTITY_CONFLICT", "revision identity has conflicting content")


def _candidate_head_fence(c: Any, scope: str, expected_snapshot: str | None,
                          expected_generation: int) -> None:
    row = c.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?",
        (scope,)).fetchone()
    actual = None if row is None else (row["snapshot_id"], row["generation"])
    if actual != (expected_snapshot, expected_generation):
        raise errors.fail(
            "SNAPSHOT_CONFLICT", "incremental refresh lost its parent head",
            details={"scope": scope, "expected_generation": expected_generation})


def commit_daily_market_candidate(conn: Any, store: ArtifactStore,
                                  candidate: DailyMarketCandidate, *, scope: str,
                                  expected_head_snapshot_id: str | None,
                                  expected_head_generation: int, clock,
                                  request_hash: str | None = None,
                                  receipt_id: str | None = None, attempt_id: str | None = None,
                                  fence: int = 1,
                                  fault: Callable[[str], None] | None = None,
                                  fence_check: Callable[[Any], None] | None = None):
    request_hash = request_hash or content_hash({"changeset": candidate.changeset_hash})
    receipt_id = receipt_id or "receipt_" + request_hash.removeprefix(CONTENT_HASH_PREFIX)[:32]
    attempt_id = attempt_id or "attempt_" + request_hash.removeprefix(CONTENT_HASH_PREFIX)[:32]

    manifests_to_commit = tuple(
        candidate.parent.table_manifests[name] if name != TABLE_NAME
        else candidate.table_manifest for name in candidate.parent.table_manifests)
    return catalog.commit_snapshot(
        conn, scope=scope, request_hash=request_hash, contracts=candidate.contracts,
        objects=candidate.objects, records=candidate.records, manifests=manifests_to_commit,
        snapshot=candidate.snapshot, expected_head_snapshot_id=expected_head_snapshot_id,
        expected_head_generation=expected_head_generation, receipt_id=receipt_id,
        attempt_id=attempt_id, fence=fence,
        fence_check=fence_check or (
            lambda c: _candidate_head_fence(
                c, scope, expected_head_snapshot_id, expected_head_generation)),
        clock=clock, fault=fault, store=store,
        record_references=lambda c, rid: _record_candidate_references(
            c, rid, candidate, clock),
        audit_partitions=True)


def run_incremental_refresh(parameters, root):
    path = root / "incremental_refresh_input.json"
    base = {
        "parent_snapshot_id": parameters.parent_snapshot_id,
        "refresh_plan_hash": parameters.refresh_plan_hash,
    }
    if not path.is_file():
        result = {
            **base, "completed_ids": [], "status": "failed",
            "coverage_advanced": False, "candidate_snapshot_id": None,
        }
        return _write_refresh_result(root, result)

    document = json.loads(path.read_text())
    binding = document.get("catalog_path")
    if not binding:
        result = {
            **base, "completed_ids": list(parameters.expected_ids),
            "status": "noop", "coverage_advanced": False,
            "candidate_snapshot_id": None,
        }
        return _write_refresh_result(root, result)

    catalog_path = binding
    object_root = document["objects_root"]
    scope = str(document["scope"])
    expected_generation = int(document["expected_head_generation"])
    expected_head = document.get("expected_head_snapshot_id", parameters.parent_snapshot_id)
    conn = catalog.sqlite3.connect(catalog_path)
    conn.row_factory = catalog.sqlite3.Row
    store = ArtifactStore(object_root)
    clock = SystemClock()
    try:
        parent = repository.Repository(conn).resolve_full(parameters.parent_snapshot_id)
        table_name = str(document.get("table_name", TABLE_NAME))
        if table_name != TABLE_NAME:
            return _run_generic_refresh(
                conn, store, parent, document, parameters, table_name, clock, root)
        contract = next(item for item in parent.contracts if item.table_name == TABLE_NAME)
        raw_records = _stage_raw_payloads(conn, store, document)
        incoming = tuple(_revision_from_document(item)
                         for item in document.get("incoming_revisions", ()))
        incoming, normalized_count = _stage_normalizations(
            conn, store, raw_records, incoming, contract.contract_id, clock)
        retained = _load_retained_revisions(conn, store)
        coverage = from_document(CompletedCoverage, document["coverage"])
        candidate = build_daily_market_candidate(
            parent, store, incoming, coverage=coverage,
            retained_revisions=retained, parent_snapshot_id=parameters.parent_snapshot_id,
            normalized_payloads=normalized_count)
        fault_point = document.get("fault_point")

        def fault(point):
            if point == fault_point:
                raise RuntimeError("injected incremental fault: " + point)

        receipt = commit_daily_market_candidate(
            conn, store, candidate, scope=scope,
            expected_head_snapshot_id=expected_head,
            expected_head_generation=expected_generation,
            clock=clock, request_hash=parameters.refresh_plan_hash,
            receipt_id=document.get("receipt_id"),
            attempt_id=document.get("attempt_id"),
            fence=int(document.get("fence", 1)), fault=fault)
        result = {
            **base, "completed_ids": list(parameters.expected_ids),
            "status": "complete", "coverage_advanced": True,
            "candidate_snapshot_id": receipt.resulting_head_snapshot_id,
        }
        return _write_refresh_result(root, result)
    finally:
        conn.close()


def _run_generic_refresh(conn, store, parent, document, parameters, table_name, clock, root):
    revisions = tuple(_generic_revision_from_document(item)
                      for item in document.get("generic_revisions", ()))
    coverage = from_document(CompletedCoverage, document["coverage"])
    candidate = generic_incremental.build_generic_table_candidate(
        parent, store, table_name, revisions, coverage=coverage,
        retained=generic_incremental.load_generic_revisions(
            conn, table_name, next(item for item in parent.contracts
                                   if item.table_name == table_name)),
        parent_snapshot_id=parameters.parent_snapshot_id)
    fault_point = document.get("fault_point")

    def fault(point):
        if point == fault_point:
            raise RuntimeError("injected incremental fault: " + point)

    receipt = generic_incremental.commit_generic_table_candidate(
        conn, store, candidate, scope=str(document["scope"]),
        expected_head_snapshot_id=document.get(
            "expected_head_snapshot_id", parameters.parent_snapshot_id),
        expected_head_generation=int(document["expected_head_generation"]),
        clock=clock, request_hash=parameters.refresh_plan_hash,
        receipt_id=document.get("receipt_id"), attempt_id=document.get("attempt_id"),
        fence=int(document.get("fence", 1)), fault=fault)
    result = {
        "parent_snapshot_id": parameters.parent_snapshot_id,
        "refresh_plan_hash": parameters.refresh_plan_hash,
        "completed_ids": list(parameters.expected_ids), "status": "complete",
        "coverage_advanced": True,
        "candidate_snapshot_id": receipt.resulting_head_snapshot_id,
    }
    return _write_refresh_result(root, result)


def _generic_revision_from_document(document):
    try:
        candidate = from_document(RevisionCandidate, document["candidate"])
        return incremental_tables.GenericRevision(
            candidate=candidate, row=document.get("row"),
            deleted=bool(document.get("deleted", False)),
            partition_key=document.get("partition_key"))
    except (KeyError, TypeError, ValueError) as exc:
        raise errors.fail("CONTRACT_MISMATCH", "generic revision document is malformed") from exc


def _stage_raw_payloads(conn, store, document):
    records = {}
    for item in document.get("raw_payloads", ()):
        payload = str(item.get("payload", "")).encode()
        if len(payload) > MAX_RAW_PAYLOAD_BYTES:
            raise errors.fail("INPUT_CHANGED", "staged raw payload exceeds size limit")
        record = cache_raw_receipt(
            conn, store, RawPayload(
                payload=payload, response_kind=item["response_kind"],
                response_meta=item.get("response_meta", {})),
            source=item["source"], endpoint=item["endpoint"],
            request=item["request"], received_at=item["received_at"])
        records[record.raw_receipt_id] = record
    return records


def _stage_normalizations(conn, store, raw_records, revisions, contract_id, clock):
    grouped = {}
    for revision in revisions:
        grouped.setdefault(revision.raw_receipt_id, []).append(revision)
    normalized = []
    cache_hits = 0
    for raw_id, group in grouped.items():
        raw = raw_records.get(raw_id)
        if raw is None:
            raise errors.fail("INPUT_CHANGED", "revision references an uncached raw receipt")
        record = cache_normalization(
            conn, store, raw, group, normalizer_id="daily_market.v1",
            contract_id=contract_id, created_at=format_timestamp(clock.now()))
        cache_hits += int(record.cache_hit)
        normalized.extend(dataclasses.replace(
            item, normalization_id=record.normalization_id) for item in group)
    return tuple(normalized), len(grouped) - cache_hits


def _write_refresh_result(root, result):
    document = {"schema_version": "incremental_refresh_result.v1.0", **result}
    (root / "incremental_refresh_result.json").write_text(canonical_json(document))
    return document


class _WriteTransaction:
    def __init__(self, conn: Any):
        self.conn = conn

    def __enter__(self):
        if self.conn.in_transaction:
            raise RuntimeError("nested incremental data transaction")
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type, _exc, _traceback):
        if exc_type is None:
            self.conn.execute("COMMIT")
        elif self.conn.in_transaction:
            self.conn.execute("ROLLBACK")
        return False


def _write_transaction(conn: Any):
    return _WriteTransaction(conn)


def _parquet_bytes(contract: TableContract, rows: Sequence[Mapping[str, Any]]) -> bytes:
    types = {
        "string": pa.string(), "float64": pa.float64(), "int64": pa.int64(),
        "bool": pa.bool_(), "timestamp[ns]": pa.timestamp("ns"),
        "timestamp[us]": pa.timestamp("us"),
    }
    arrays = {}
    for column in contract.columns:
        if column.physical_type not in types:
            raise errors.fail("UNSUPPORTED_CONTRACT", "daily_market physical type is unsupported")
        arrays[column.name] = pa.array([row.get(column.name) for row in rows],
                                       type=types[column.physical_type])
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table(arrays), sink)
    return sink.getvalue().to_pybytes()
