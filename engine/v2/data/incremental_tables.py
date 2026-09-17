"""Pure incremental merge primitives shared by the EOD table families.

``daily_market`` has a durable adapter in :mod:`incremental`; these generic
primitives make the same revision and partition rules available to chains,
calendar, reference, price-history, and derived table producers before each
producer is wired to its own provider adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from engine.v2.contracts import RevisionCandidate, RowChange, TableContract
from engine.v2.data import errors
from engine.v2.foundation import canonical_json, content_hash, to_document

__all__ = [
    "GenericRevision",
    "GenericMerge",
    "logical_key_for_row",
    "revision_hash",
    "merge_table_rows",
]


@dataclass(frozen=True, kw_only=True)
class GenericRevision:
    candidate: RevisionCandidate
    row: Mapping[str, Any] | None
    deleted: bool = False
    partition_key: str | None = None


@dataclass(frozen=True, kw_only=True)
class GenericMerge:
    rows: tuple[dict[str, Any], ...]
    retained_revisions: tuple[GenericRevision, ...]
    incoming_revisions: tuple[GenericRevision, ...]
    winners: tuple[GenericRevision, ...]
    changes: tuple[RowChange, ...]
    changed_partitions: tuple[str, ...]


def logical_key_for_row(contract: TableContract, row: Mapping[str, Any]) -> str:
    _validate_columns(contract, row)
    return canonical_json([row.get(name) for name in contract.primary_key])


def revision_hash(*, logical_key: str, row: Mapping[str, Any] | None,
                  deleted: bool) -> str:
    return content_hash({"kind": "table_revision.v1", "logical_key": logical_key,
                         "deleted": deleted, "row": dict(row) if row is not None else None})


def merge_table_rows(contract: TableContract, prior_rows: Sequence[Mapping[str, Any]],
                     retained: Sequence[GenericRevision],
                     incoming: Sequence[GenericRevision]) -> GenericMerge:
    """Merge any contract with deterministic winner selection.

    Equal-ranked revisions with different content are refused. A tombstone
    keeps its partition explicitly because a deleted row has no payload from
    which to infer that partition.
    """
    prior = _index_prior(contract, prior_rows)
    revisions = tuple(_validate_revision(contract, item) for item in (*retained, *incoming))
    winners = _select_winners(revisions)
    result = dict(prior)
    for key, revision in winners.items():
        if revision.deleted:
            result.pop(key, None)
        else:
            result[key] = dict(revision.row or {})
    changes = _build_changes(contract, prior, result, winners)
    return GenericMerge(
        rows=tuple(result[key] for key in sorted(result)),
        retained_revisions=tuple(retained), incoming_revisions=tuple(incoming),
        winners=tuple(winners[key] for key in sorted(winners)),
        changes=tuple(changes),
        changed_partitions=tuple(sorted({item.partition_key for item in changes})),
    )


def _index_prior(contract, rows):
    prior = {}
    for raw in rows:
        row = _canonical_row(contract, raw)
        key = logical_key_for_row(contract, row)
        if key in prior:
            raise errors.fail("IDENTITY_CONFLICT", "parent table contains duplicate logical keys")
        prior[key] = row
    return prior


def _build_changes(contract, prior, result, winners):
    changes = []
    for key in sorted(set(prior) | set(result)):
        before, after = prior.get(key), result.get(key)
        old_hash = _row_hash(before) if before is not None else None
        new_hash = _row_hash(after) if after is not None else None
        if old_hash != new_hash:
            changes.append(_change(contract, key, before, after, winners))
    return changes


def _change(contract, key, before, after, winners):
    revision = winners.get(key)
    if revision is None:
        raise errors.fail("MANIFEST_CORRUPT", "logical change has no revision winner")
    partition = _partition(contract, after or before, revision)
    if partition is None:
        raise errors.fail("CONTRACT_MISMATCH", "revision has no partition key")
    return RowChange(
        logical_key=key, partition_key=partition, columns=_changed_columns(contract, before, after),
        time_range=None, old_hash=_row_hash(before) if before is not None else None,
        new_hash=_row_hash(after) if after is not None else None,
        revision_kind=("append" if before is None else
                       "tombstone" if after is None else "correction"),
        revision_id=revision.candidate.revision_id)


def _select_winners(revisions: Sequence[GenericRevision]) -> dict[str, GenericRevision]:
    by_id = {}
    for revision in revisions:
        prior = by_id.get(revision.candidate.revision_id)
        if prior is not None and not _same_revision_identity(prior, revision):
            raise errors.fail("IDENTITY_CONFLICT", "revision id has conflicting payloads")
        by_id[revision.candidate.revision_id] = revision
    groups: dict[str, list[GenericRevision]] = {}
    for revision in by_id.values():
        groups.setdefault(revision.candidate.logical_key, []).append(revision)
    selected = {}
    for key, group in groups.items():
        priority = min(item.candidate.source_priority for item in group)
        ranked = [item for item in group if item.candidate.source_priority == priority]
        finality = max(item.candidate.finality == "final" for item in ranked)
        ranked = [item for item in ranked if (item.candidate.finality == "final") == finality]
        ordinal = max(item.candidate.revision_ordinal for item in ranked)
        ranked = [item for item in ranked if item.candidate.revision_ordinal == ordinal]
        if len({item.candidate.content_hash for item in ranked}) != 1:
            raise errors.fail("IDENTITY_CONFLICT", "equal-ranked revisions conflict")
        selected[key] = max(ranked, key=lambda item: (
            item.candidate.received_at, item.candidate.revision_id))
    return selected


def _same_revision_identity(left, right):
    a, b = left.candidate, right.candidate
    return (a.logical_key, a.source, a.source_priority, a.finality,
            a.revision_ordinal, a.content_hash) == (
                b.logical_key, b.source, b.source_priority, b.finality,
                b.revision_ordinal, b.content_hash)


def _validate_revision(contract, revision):
    if revision.deleted:
        if revision.row is not None:
            raise errors.fail("CONTRACT_MISMATCH", "tombstone cannot carry a row")
        if not revision.partition_key:
            raise errors.fail("CONTRACT_MISMATCH", "tombstone requires partition_key")
        expected = revision_hash(
            logical_key=revision.candidate.logical_key, row=None, deleted=True)
        if expected != revision.candidate.content_hash:
            raise errors.fail("IDENTITY_CONFLICT", "tombstone content hash is invalid")
        return revision
    if revision.row is None:
        raise errors.fail("CONTRACT_MISMATCH", "live revision requires a row")
    row = _canonical_row(contract, revision.row)
    logical_key = logical_key_for_row(contract, row)
    if revision.candidate.logical_key != logical_key:
        raise errors.fail("CONTRACT_MISMATCH", "revision logical key does not match row")
    expected = revision_hash(logical_key=logical_key, row=row, deleted=False)
    if expected != revision.candidate.content_hash:
        raise errors.fail("IDENTITY_CONFLICT", "revision content hash does not match row")
    return GenericRevision(candidate=revision.candidate, row=row, deleted=False,
                           partition_key=revision.partition_key or _partition(contract, row, revision))


def _canonical_row(contract, row):
    _validate_columns(contract, row)
    return {column.name: row.get(column.name) for column in contract.columns}


def _validate_columns(contract, row):
    names = {column.name for column in contract.columns}
    if set(row) - names or any(column.name not in row and not column.nullable
                               for column in contract.columns):
        raise errors.fail("CONTRACT_MISMATCH", "table row does not satisfy contract columns")


def _partition(contract, row, revision):
    if not contract.partition_columns:
        return revision.partition_key or "__whole__"
    if row is None:
        return revision.partition_key
    values = [row.get(name) for name in contract.partition_columns]
    if any(value is None for value in values):
        return revision.partition_key
    return "/".join(str(value) for value in values)


def _changed_columns(contract, before, after):
    if before is None or after is None:
        return tuple(column.name for column in contract.columns)
    return tuple(column.name for column in contract.columns
                 if before.get(column.name) != after.get(column.name))


def _row_hash(row):
    return None if row is None else content_hash({"kind": "table_row.v1", "row": dict(row)})


def _revision_document(revision):
    return {"candidate": to_document(revision.candidate), "row": dict(revision.row)
            if revision.row is not None else None, "deleted": revision.deleted,
            "partition_key": revision.partition_key}
