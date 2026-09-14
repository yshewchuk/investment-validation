"""Atomic snapshot commit, idempotent insert, and head compare-and-swap — §7.3.

One function commits a fully-built snapshot: :func:`commit_snapshot` opens one
short ``BEGIN IMMEDIATE`` coordinator transaction, verifies the caller's fence
and the expected current head, inserts every immutable row idempotently (an
existing ID is accepted only if its full canonical payload matches — an
IDENTITY_CONFLICT otherwise, with nothing written), inserts the success
receipt, and compare-and-swaps the scope's head — never last-writer-wins.
:func:`record_failed_import` persists a failed/conflict receipt in its own
transaction, without ever touching a head. :func:`move_head` is the one
rollback primitive: a bare CAS to a previously committed snapshot.

No file copy, Arrow scan, or hash calculation runs inside the transaction:
:func:`commit_snapshot` re-verifies every manifest, plus the row-sum and
key-overlap invariants SQL cannot express, *before* opening it.

This module never imports ``engine.v2.ops`` (layer 1, ``system_rearchitecture.md``
§4.1): it cannot call ``engine.v2.ops.lifecycle.verify_fence`` or
``engine.v2.ops.catalog.transaction`` directly. Two judgement calls close that
gap (task brief, P2-3):

* ``commit_snapshot`` takes a ``fence_check: Callable[[sqlite3.Connection], None]``,
  invoked first inside the transaction, before any insert. The ops wrapper
  (``engine/v2/ops/snapshots.py::commit_snapshot_for_attempt``) passes
  ``lambda c: verify_fence(c, attempt_id, fence, clock.now())``.
* the transaction itself is a private, local ``BEGIN IMMEDIATE`` context
  manager (:func:`_immediate_transaction`) rather than a reuse of
  ``engine.v2.ops.catalog.transaction`` — the same shape, kept local so this
  package's import fan-out never reaches ``ops``.

Resolve failures use ``MANIFEST_CORRUPT`` (§11 has no ``INTEGRITY_FAILED``). A
same-``*_id`` row with a different canonical payload raises ``IDENTITY_CONFLICT``;
a lost head compare-and-swap, including a scope's first head, ``SNAPSHOT_CONFLICT``.

**Review fix (task 1 follow-up):** a dataset version's/snapshot's ``*_id``
deliberately excludes ``parent_*_id`` — only ``manifest_hash`` covers it — so a
candidate identical to an already-stored version except for its declared
parent is a *reuse*, not a conflict: ``_insert_dataset_version``/
``_insert_snapshot`` compare only the identity-covered columns and, on reuse,
return the record reconciled to whatever parent/``manifest_hash`` is truly
stored (never the candidate's own). ``commit_snapshot`` threads that
reconciliation up through the snapshot before deciding whether the snapshot
itself is a reuse, so a freshly inserted one never stores a ``manifest_hash``
computed over an unreconciled child; an already-at-head result leaves the
head untouched rather than CAS'd to itself. Contracts, objects and fragments
have no parent field and keep the original same-ID-different-payload rule.

Required fault points (task brief decision 2), fired through one ``fault``
hook in commit order: ``before_transaction``, ``after_contracts``,
``after_objects``, ``after_fragments``, ``after_dataset_versions``,
``after_memberships``, ``after_snapshot``, ``after_snapshot_tables``,
``before_head_update``, ``before_commit`` — all before ``COMMIT``, so an
injected failure rolls back to nothing new; object-side fault points belong to
``objects.py`` and ``foundation.ArtifactStore``, which run before this module.
"""
from __future__ import annotations

import dataclasses
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import contextmanager

from engine.v2.contracts import (
    DatasetManifest,
    FragmentRecord,
    ObjectRef,
    Problem,
    SnapshotImportReceipt,
    SnapshotRef,
    TableContract,
)
from engine.v2.data.errors import fail
from engine.v2.data.manifests import snapshot_ref as build_snapshot_ref
from engine.v2.data.manifests import (
    table_contract_hash,
    verify_dataset_manifest,
    verify_fragment_record,
    verify_partition_hashes,
    verify_snapshot_ref,
)
from engine.v2.foundation import (
    CONTENT_HASH_PREFIX,
    ArtifactStore,
    Clock,
    canonical_json,
    format_timestamp,
    to_document,
)

__all__ = ["commit_snapshot", "move_head", "record_failed_import"]

FaultHook = Callable[[str], None]


# --------------------------------------------------------------------------
# local transaction — never engine.v2.ops.catalog.transaction (module docstring)
# --------------------------------------------------------------------------


@contextmanager
def _immediate_transaction(conn: sqlite3.Connection):
    if conn.in_transaction:
        raise RuntimeError("nested data catalog transaction: effects would commit in the caller's")
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _dumps(value) -> str:
    return canonical_json(to_document(value))


# --------------------------------------------------------------------------
# pre-transaction verification: manifests, row-sum, key-overlap (§6 invariant 6)
# --------------------------------------------------------------------------


def _records_for(manifest: DatasetManifest, records: Sequence[FragmentRecord]) -> list[FragmentRecord]:
    by_ref = {(r.fragment_id, r.manifest_hash): r for r in records}
    matched = []
    for ref in manifest.fragment_refs:
        record = by_ref.get((ref.fragment_id, ref.manifest_hash))
        if record is None:
            raise fail("MANIFEST_CORRUPT", "a dataset manifest references a fragment record "
                      "that was not provided", details={"fragment_id": ref.fragment_id})
        matched.append(record)
    return matched


def _table_name_for(manifest: DatasetManifest, snapshot: SnapshotRef) -> str:
    for name, ref in snapshot.table_versions.items():
        if ref == manifest.dataset_version_ref:
            return name
    raise fail("MANIFEST_CORRUPT", "a dataset manifest is not referenced by the snapshot",
              details={"dataset_version_id": manifest.dataset_version_ref.dataset_version_id})


def _check_no_key_overlap(records: Sequence[FragmentRecord]) -> None:
    """Fragment key ranges must not overlap within one logical partition.

    ``manifests.dataset_manifest``/``verify_dataset_manifest`` already refuse
    this via ``_check_membership_order`` — this is defense in depth for the
    schema-docstring invariant (§6 invariant 6) over reconciled records.
    """
    by_partition: dict[str, list[FragmentRecord]] = {}
    for record in records:
        by_partition.setdefault(record.partition_key, []).append(record)
    for partition_key, group in by_partition.items():
        ordered = sorted(group, key=lambda r: r.primary_key_min)
        for prev, cur in zip(ordered, ordered[1:]):
            if prev.primary_key_max >= cur.primary_key_min:
                raise fail("MANIFEST_CORRUPT", "fragment key ranges overlap within one partition",
                          details={"partition_key": partition_key})


def _verify_everything(contracts: Sequence[TableContract], records: Sequence[FragmentRecord],
                       manifests: Sequence[DatasetManifest], snapshot: SnapshotRef, store) -> None:
    for contract in contracts:
        if table_contract_hash(contract) != contract.definition_hash:
            raise fail("MANIFEST_CORRUPT", "contract definition_hash does not match its content",
                      details={"contract_id": contract.contract_id})
    for record in records:
        verify_fragment_record(record)
    contract_by_id = {c.contract_id: c for c in contracts}
    by_table: dict[str, DatasetManifest] = {}
    for manifest in manifests:
        matched = _records_for(manifest, records)
        verify_dataset_manifest(manifest, matched)
        _check_no_key_overlap(matched)
        contract = contract_by_id[manifest.dataset_version_ref.table_contract_ref.contract_id]
        verify_partition_hashes(store, manifest, matched, contract)
        by_table[_table_name_for(manifest, snapshot)] = manifest
    verify_snapshot_ref(snapshot, by_table)


# --------------------------------------------------------------------------
# idempotent inserts: existing ID accepted only if the full payload matches
# --------------------------------------------------------------------------


def _object_storage_key(ref: ObjectRef) -> str:
    digest = ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
    return f"objects/{digest[:2]}/{digest}"


def _insert_contract(conn: sqlite3.Connection, contract: TableContract, now: str) -> None:
    existing = conn.execute(
        "SELECT schema_version, table_name, definition_hash, definition_json FROM data_contracts "
        "WHERE contract_id = ?", (contract.contract_id,)).fetchone()
    payload = (contract.schema_version, contract.table_name, contract.definition_hash, _dumps(contract))
    if existing is not None:
        if tuple(existing) != payload:
            raise fail("IDENTITY_CONFLICT", "contract_id already exists with different content",
                      details={"contract_id": contract.contract_id})
        return
    conn.execute(
        "INSERT INTO data_contracts (contract_id, schema_version, table_name, definition_hash, "
        "definition_json, registered_at) VALUES (?, ?, ?, ?, ?, ?)",
        (contract.contract_id, *payload, now))


def _insert_object(conn: sqlite3.Connection, ref: ObjectRef, now: str) -> None:
    payload = (ref.kind, ref.content_hash, ref.byte_size, _object_storage_key(ref))
    existing = conn.execute(
        "SELECT kind, content_hash, byte_size, storage_key FROM data_objects WHERE object_id = ?",
        (ref.object_id,)).fetchone()
    if existing is not None:
        if tuple(existing) != payload:
            raise fail("IDENTITY_CONFLICT", "object_id already exists with different content",
                      details={"object_id": ref.object_id})
        return
    conn.execute(
        "INSERT INTO data_objects (object_id, kind, content_hash, byte_size, storage_key, "
        "registered_at) VALUES (?, ?, ?, ?, ?, ?)", (ref.object_id, *payload, now))


def _insert_fragment(conn: sqlite3.Connection, record: FragmentRecord, now: str) -> None:
    key_bounds = canonical_json({"primary_key_min": list(record.primary_key_min),
                                 "primary_key_max": list(record.primary_key_max)})
    time_bounds = (None if record.time_min is None else
                  canonical_json({"time_min": record.time_min, "time_max": record.time_max}))
    receipts = canonical_json(list(record.input_receipt_refs))
    payload = (record.object_ref.object_id, record.table_contract_ref.contract_id,
              record.partition_key, record.row_count, record.byte_hash,
              record.logical_content_hash, key_bounds, time_bounds,
              record.import_request_hash, receipts)
    existing = conn.execute(
        "SELECT object_id, contract_id, partition_key, row_count, byte_hash, logical_content_hash, "
        "key_bounds_json, time_bounds_json, import_request_hash, input_receipt_refs_json "
        "FROM data_fragments WHERE fragment_id = ?", (record.fragment_id,)).fetchone()
    if existing is not None:
        if tuple(existing) != payload:
            raise fail("IDENTITY_CONFLICT", "fragment_id already exists with different content",
                      details={"fragment_id": record.fragment_id})
        return
    conn.execute(
        "INSERT INTO data_fragments (fragment_id, object_id, contract_id, partition_key, row_count, "
        "byte_hash, logical_content_hash, key_bounds_json, time_bounds_json, import_request_hash, "
        "input_receipt_refs_json, registered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (record.fragment_id, *payload, now))


def _reconciled_manifest(manifest: DatasetManifest, true_parent: str | None,
                         true_manifest_hash: str) -> DatasetManifest:
    """``manifest`` with its parent/``manifest_hash`` overridden to whatever the
    catalog actually stores for this ``dataset_version_id`` — a no-op when
    they already agree (the ordinary fresh-insert and exact-repeat cases).
    """
    ref = manifest.dataset_version_ref
    if true_parent == manifest.parent_dataset_version_id and true_manifest_hash == ref.manifest_hash:
        return manifest
    new_ref = dataclasses.replace(ref, manifest_hash=true_manifest_hash)
    return dataclasses.replace(manifest, parent_dataset_version_id=true_parent, dataset_version_ref=new_ref)


def _insert_dataset_version(conn: sqlite3.Connection, manifest: DatasetManifest,
                            now: str) -> DatasetManifest:
    """Insert, or reuse an existing row whose identity-covered fields match.

    ``dataset_version_id`` excludes ``parent_dataset_version_id`` (only
    ``manifest_hash`` covers it — task brief review fix), so a candidate
    differing only in its declared parent reuses the stored row instead of
    conflicting. Returns ``manifest`` reconciled to the truly stored
    parent/``manifest_hash`` either way (see :func:`_reconciled_manifest`).
    """
    ref = manifest.dataset_version_ref
    evidence = canonical_json({"coverage_receipt_refs": list(manifest.coverage_receipt_refs),
                               "availability_evidence_refs": list(manifest.availability_evidence_refs)})
    identity = (ref.table_contract_ref.contract_id, manifest.logical_content_hash, manifest.row_count,
               manifest.knowledge_mode, evidence)
    partition_hashes_json = canonical_json(dict(manifest.partition_logical_hashes))
    existing = conn.execute(
        "SELECT contract_id, logical_content_hash, row_count, knowledge_mode, evidence_json, "
        "parent_dataset_version_id, manifest_hash FROM data_dataset_versions "
        "WHERE dataset_version_id = ?", (ref.dataset_version_id,)).fetchone()
    if existing is not None:
        if tuple(existing[:5]) != identity:
            raise fail("IDENTITY_CONFLICT", "dataset_version_id already exists with different content",
                      details={"dataset_version_id": ref.dataset_version_id})
        return _reconciled_manifest(manifest, existing["parent_dataset_version_id"],
                                    existing["manifest_hash"])
    conn.execute(
        "INSERT INTO data_dataset_versions (dataset_version_id, contract_id, "
        "parent_dataset_version_id, manifest_hash, logical_content_hash, row_count, "
        "knowledge_mode, evidence_json, partition_logical_hashes_json, registered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (ref.dataset_version_id, identity[0], manifest.parent_dataset_version_id, ref.manifest_hash,
         identity[1], identity[2], identity[3], identity[4], partition_hashes_json, now))
    return manifest


def _insert_memberships(conn: sqlite3.Connection, manifest: DatasetManifest) -> None:
    dsv_id = manifest.dataset_version_ref.dataset_version_id
    for ordinal, ref in enumerate(manifest.fragment_refs):
        existing = conn.execute(
            "SELECT fragment_id FROM data_version_fragments WHERE dataset_version_id = ? "
            "AND ordinal = ?", (dsv_id, ordinal)).fetchone()
        if existing is not None:
            if existing["fragment_id"] != ref.fragment_id:
                raise fail("IDENTITY_CONFLICT", "dataset version membership changed at this ordinal",
                          details={"dataset_version_id": dsv_id, "ordinal": ordinal})
            continue
        conn.execute(
            "INSERT INTO data_version_fragments (dataset_version_id, ordinal, fragment_id) "
            "VALUES (?, ?, ?)", (dsv_id, ordinal, ref.fragment_id))


def _reconcile_snapshot(snapshot: SnapshotRef, table_versions: dict[str, DatasetManifest]) -> SnapshotRef:
    """Rebuild ``snapshot`` over dataset versions already reconciled to their
    truly stored parent/``manifest_hash`` (:func:`_insert_dataset_version`), so
    a snapshot that turns out to be a fresh insert never stores a
    ``manifest_hash`` computed over an unreconciled child (task brief review
    fix). ``dataset_version_id`` never changes under reconciliation, so the
    rebuilt ``snapshot_id`` always equals the candidate's own.
    """
    return build_snapshot_ref(
        table_versions, calendar_version=snapshot.calendar_version,
        source_priority_version=snapshot.source_priority_version,
        finality_receipt_refs=snapshot.finality_receipt_refs,
        parent_snapshot_id=snapshot.parent_snapshot_id)


def _insert_snapshot(conn: sqlite3.Connection, snapshot: SnapshotRef, receipt_id: str,
                     now: str) -> SnapshotRef:
    """Insert, or reuse an existing row whose identity-covered fields match.

    ``snapshot_id`` excludes ``parent_snapshot_id`` (only ``manifest_hash``
    covers it — task brief review fix), so a candidate differing only in its
    declared parent reuses the stored row instead of conflicting. Returns
    ``snapshot`` reconciled to the truly stored parent/``manifest_hash``
    either way — never the candidate's own on reuse.
    """
    identity = (snapshot.calendar_version, snapshot.source_priority_version,
               canonical_json(list(snapshot.finality_receipt_refs)),
               canonical_json(dict(snapshot.knowledge_mode_by_table)))
    existing = conn.execute(
        "SELECT calendar_version, source_priority_version, finality_receipt_refs_json, "
        "knowledge_mode_by_table_json, parent_snapshot_id, manifest_hash FROM data_snapshots "
        "WHERE snapshot_id = ?", (snapshot.snapshot_id,)).fetchone()
    if existing is not None:
        if tuple(existing[:4]) != identity:
            raise fail("IDENTITY_CONFLICT", "snapshot_id already exists with different content",
                      details={"snapshot_id": snapshot.snapshot_id})
        if (existing["parent_snapshot_id"] == snapshot.parent_snapshot_id
                and existing["manifest_hash"] == snapshot.manifest_hash):
            return snapshot
        return dataclasses.replace(snapshot, parent_snapshot_id=existing["parent_snapshot_id"],
                                   manifest_hash=existing["manifest_hash"])
    conn.execute(
        "INSERT INTO data_snapshots (snapshot_id, parent_snapshot_id, manifest_hash, "
        "calendar_version, source_priority_version, finality_receipt_refs_json, "
        "knowledge_mode_by_table_json, commit_receipt_ref, registered_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (snapshot.snapshot_id, snapshot.parent_snapshot_id, snapshot.manifest_hash, *identity,
         receipt_id, now))
    return snapshot


def _insert_snapshot_tables(conn: sqlite3.Connection, snapshot: SnapshotRef) -> None:
    for table_name, ref in snapshot.table_versions.items():
        existing = conn.execute(
            "SELECT dataset_version_id FROM data_snapshot_tables WHERE snapshot_id = ? "
            "AND table_name = ?", (snapshot.snapshot_id, table_name)).fetchone()
        if existing is not None:
            if existing["dataset_version_id"] != ref.dataset_version_id:
                raise fail("IDENTITY_CONFLICT", "snapshot table binding changed",
                          details={"snapshot_id": snapshot.snapshot_id, "table_name": table_name})
            continue
        conn.execute(
            "INSERT INTO data_snapshot_tables (snapshot_id, table_name, dataset_version_id) "
            "VALUES (?, ?, ?)", (snapshot.snapshot_id, table_name, ref.dataset_version_id))


def _insert_receipt(conn: sqlite3.Connection, receipt: SnapshotImportReceipt, scope: str,
                    now: str) -> None:
    conn.execute(
        "INSERT INTO data_import_receipts (receipt_id, attempt_id, fence, source_manifest_hash, "
        "result_snapshot_id, status, problem_json, registered_at, scope) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (receipt.receipt_id, receipt.attempt_id, receipt.fence, receipt.request_hash,
         receipt.resulting_head_snapshot_id, receipt.status, None, now, scope))


# --------------------------------------------------------------------------
# head: precheck (fail fast) plus the real compare-and-swap (never trust the precheck alone)
# --------------------------------------------------------------------------


def _current_head(conn: sqlite3.Connection, scope: str) -> sqlite3.Row | None:
    return conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?",
                        (scope,)).fetchone()


def _check_head_expectation(current: sqlite3.Row | None, expected_snapshot_id: str | None,
                            expected_generation: int) -> None:
    if current is None:
        if expected_snapshot_id is not None or expected_generation != 0:
            raise fail("SNAPSHOT_CONFLICT", "expected head does not match the current catalog state")
        return
    if current["snapshot_id"] != expected_snapshot_id or current["generation"] != expected_generation:
        raise fail("SNAPSHOT_CONFLICT", "expected head does not match the current catalog state")


def _update_head(conn: sqlite3.Connection, scope: str, new_snapshot_id: str,
                 expected_snapshot_id: str | None, expected_generation: int, receipt_id: str,
                 now: str) -> None:
    if expected_snapshot_id is None:
        conn.execute(
            "INSERT INTO data_snapshot_heads (scope, snapshot_id, generation, updated_at, "
            "update_receipt_ref) VALUES (?, ?, 1, ?, ?)", (scope, new_snapshot_id, now, receipt_id))
        return
    cursor = conn.execute(
        "UPDATE data_snapshot_heads SET snapshot_id = ?, generation = ?, updated_at = ?, "
        "update_receipt_ref = ? WHERE scope = ? AND snapshot_id = ? AND generation = ?",
        (new_snapshot_id, expected_generation + 1, now, receipt_id, scope, expected_snapshot_id,
         expected_generation))
    if cursor.rowcount == 0:
        raise fail("SNAPSHOT_CONFLICT", "head compare-and-swap changed zero rows")


# --------------------------------------------------------------------------
# receipt-level idempotency: a repeated call under the same receipt_id is a no-op
# --------------------------------------------------------------------------


def _existing_receipt(conn: sqlite3.Connection, receipt_id: str, request_hash: str, attempt_id: str,
                      fence: int, scope: str, snapshot: SnapshotRef,
                      resulting_generation: int) -> SnapshotImportReceipt | None:
    """A prior committed receipt under ``receipt_id``, only if it is genuinely
    the same call replayed: same ``request_hash``, ``attempt_id``, ``scope``
    and resulting ``snapshot_id`` (task brief review fix — the short-circuit
    must verify what it short-circuits, not just trust the id). Anything else
    stored under this ``receipt_id`` is ``IDENTITY_CONFLICT``, nothing written.
    """
    row = conn.execute("SELECT * FROM data_import_receipts WHERE receipt_id = ?",
                       (receipt_id,)).fetchone()
    if row is None:
        return None
    expected = (attempt_id, fence, request_hash, scope, "committed", snapshot.snapshot_id)
    found = (row["attempt_id"], row["fence"], row["source_manifest_hash"], row["scope"], row["status"],
            row["result_snapshot_id"])
    if found != expected:
        raise fail("IDENTITY_CONFLICT", "receipt_id already exists with a different outcome",
                  details={"receipt_id": receipt_id})
    return SnapshotImportReceipt(
        receipt_id=receipt_id, request_hash=request_hash, attempt_id=attempt_id, fence=fence,
        snapshot_ref=snapshot, legacy_snapshot_object_ref=None, prior_head_snapshot_id=None,
        resulting_head_snapshot_id=snapshot.snapshot_id, resulting_head_generation=resulting_generation,
        status="committed", problem=None, envelope={})


# --------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------


def commit_snapshot(conn: sqlite3.Connection, *, scope: str, request_hash: str,
                    contracts: Sequence[TableContract], objects: Sequence[ObjectRef],
                    records: Sequence[FragmentRecord], manifests: Sequence[DatasetManifest],
                    snapshot: SnapshotRef, expected_head_snapshot_id: str | None,
                    expected_head_generation: int, receipt_id: str, attempt_id: str, fence: int,
                    fence_check: Callable[[sqlite3.Connection], None], clock: Clock,
                    fault: FaultHook | None = None, store: ArtifactStore | None = None,
                    record_references: Callable[[sqlite3.Connection, str], None] | None = None,
                    ) -> SnapshotImportReceipt:
    """§7.3 steps 1-7, over already-built, already-durable inputs.

    Re-verifies every manifest (identity, row-sum, key-overlap) before opening
    the transaction, then inserts everything idempotently, inserts the success
    receipt, and compare-and-swaps ``scope``'s head. Raises on any failure —
    it never returns a ``failed``/``conflict`` receipt; ``record_failed_import``
    is the caller's separate, later step for persisting evidence of that.
    ``record_references(conn, receipt_id)`` runs right after the receipt row,
    in the same transaction (``reference_catalog.insert_reference_inputs``).
    """
    fault = fault or (lambda point: None)
    _verify_everything(contracts, records, manifests, snapshot, store)
    table_names = [_table_name_for(manifest, snapshot) for manifest in manifests]
    now = format_timestamp(clock.now())
    fault("before_transaction")
    with _immediate_transaction(conn):
        fence_check(conn)
        shortcut = _existing_receipt(conn, receipt_id, request_hash, attempt_id, fence, scope,
                                     snapshot, expected_head_generation + 1)
        if shortcut is not None:
            return shortcut
        _check_head_expectation(_current_head(conn, scope), expected_head_snapshot_id,
                                expected_head_generation)
        for contract in contracts:
            _insert_contract(conn, contract, now)
        fault("after_contracts")
        for obj in objects:
            _insert_object(conn, obj, now)
        fault("after_objects")
        for record in records:
            _insert_fragment(conn, record, now)
        fault("after_fragments")
        reconciled = [_insert_dataset_version(conn, manifest, now) for manifest in manifests]
        fault("after_dataset_versions")
        for manifest in reconciled:
            _insert_memberships(conn, manifest)
        fault("after_memberships")
        # Rebuild the snapshot over reconciled children before deciding
        # whether the snapshot itself is a fresh insert or a reuse (task
        # brief review fix): a fresh insert must never store a manifest_hash
        # computed over an unreconciled child.
        snapshot = _reconcile_snapshot(snapshot, dict(zip(table_names, reconciled)))
        snapshot = _insert_snapshot(conn, snapshot, receipt_id, now)
        fault("after_snapshot")
        _insert_snapshot_tables(conn, snapshot)
        fault("after_snapshot_tables")
        already_at_head = snapshot.snapshot_id == expected_head_snapshot_id
        resulting_generation = (expected_head_generation if already_at_head
                                else expected_head_generation + 1)
        receipt = SnapshotImportReceipt(
            receipt_id=receipt_id, request_hash=request_hash, attempt_id=attempt_id, fence=fence,
            snapshot_ref=snapshot, legacy_snapshot_object_ref=None,
            prior_head_snapshot_id=expected_head_snapshot_id,
            resulting_head_snapshot_id=snapshot.snapshot_id,
            resulting_head_generation=resulting_generation, status="committed",
            problem=None, envelope={})
        _insert_receipt(conn, receipt, scope, now)
        if record_references is not None:
            record_references(conn, receipt_id)
        fault("before_head_update")
        if not already_at_head:
            # The candidate resolved (possibly by reuse) to a snapshot the
            # head already points at: no CAS, no generation bump (task brief
            # review fix — "unless the head is already that snapshot, which
            # is a no-op").
            _update_head(conn, scope, snapshot.snapshot_id, expected_head_snapshot_id,
                        expected_head_generation, receipt_id, now)
        fault("before_commit")
    return receipt


def record_failed_import(conn: sqlite3.Connection, *, receipt_id: str, request_hash: str,
                         attempt_id: str, fence: int, problem: Problem,
                         clock: Clock) -> SnapshotImportReceipt:
    """Persist a failed/conflict receipt in its own short transaction.

    Never advances a head (§7.3: "failure-receipt persistence never advances
    a snapshot head"). ``status`` is ``conflict`` for a lost head
    compare-and-swap and ``failed`` for everything else — this module's
    judgement call, since the receipt contract's ``status`` vocabulary names
    both but ``problem`` alone tells them apart.
    """
    status = "conflict" if problem.code == "SNAPSHOT_CONFLICT" else "failed"
    now = format_timestamp(clock.now())
    with _immediate_transaction(conn):
        existing = conn.execute("SELECT * FROM data_import_receipts WHERE receipt_id = ?",
                                (receipt_id,)).fetchone()
        expected = (attempt_id, fence, request_hash, status, None)
        if existing is not None:
            found = (existing["attempt_id"], existing["fence"], existing["source_manifest_hash"],
                     existing["status"], existing["result_snapshot_id"])
            if found != expected:
                raise fail("IDENTITY_CONFLICT", "receipt_id already exists with a different outcome",
                          details={"receipt_id": receipt_id})
        else:
            conn.execute(
                "INSERT INTO data_import_receipts (receipt_id, attempt_id, fence, "
                "source_manifest_hash, result_snapshot_id, status, problem_json, registered_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (receipt_id, attempt_id, fence, request_hash, None, status, _dumps(problem), now))
    return SnapshotImportReceipt(
        receipt_id=receipt_id, request_hash=request_hash, attempt_id=attempt_id, fence=fence,
        snapshot_ref=None, legacy_snapshot_object_ref=None, prior_head_snapshot_id=None,
        resulting_head_snapshot_id=None, resulting_head_generation=None, status=status,
        problem=problem, envelope={})


def move_head(conn: sqlite3.Connection, *, scope: str, to_snapshot_id: str, expected_snapshot_id: str,
             expected_generation: int, receipt_ref: str, clock: Clock) -> None:
    """Rollback's one primitive: compare-and-swap the head to a previously
    committed snapshot under a new update receipt (§10).

    Never deletes, rewrites, or relabels the snapshot it moves away from or
    to; both stay resolvable. ``to_snapshot_id`` must already be a committed
    ``data_snapshots`` row — this never mints a new one.
    """
    now = format_timestamp(clock.now())
    with _immediate_transaction(conn):
        target = conn.execute("SELECT 1 FROM data_snapshots WHERE snapshot_id = ?",
                              (to_snapshot_id,)).fetchone()
        if target is None:
            raise fail("SNAPSHOT_NOT_FOUND", "rollback target snapshot is not in the catalog",
                      details={"snapshot_id": to_snapshot_id})
        _check_head_expectation(_current_head(conn, scope), expected_snapshot_id, expected_generation)
        cursor = conn.execute(
            "UPDATE data_snapshot_heads SET snapshot_id = ?, generation = ?, updated_at = ?, "
            "update_receipt_ref = ? WHERE scope = ? AND snapshot_id = ? AND generation = ?",
            (to_snapshot_id, expected_generation + 1, now, receipt_ref, scope, expected_snapshot_id,
             expected_generation))
        if cursor.rowcount == 0:
            raise fail("SNAPSHOT_CONFLICT", "head compare-and-swap changed zero rows")
