"""Shared fixtures for the P2-4 test files (test_v2_data_query.py,
test_v2_data_events_chains.py, test_v2_data_roundtrip.py): a real catalog +
ArtifactStore, real contracts from ``build_legacy_mapping()``, and helpers to
publish real synthetic Parquet fragments and commit them into a multi-table
snapshot — reusing the same techniques ``tests/test_v2_data_manifests.py``
and ``tests/test_v2_data_commit.py`` already established, generalized to more
than one table (those two files' own helpers are hardcoded to ``securities``
alone). No market data, no network.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from engine.v2.contracts.data import (
    DatasetManifest,
    FragmentRecord,
    ObjectRef,
    SnapshotRef,
    TableContract,
    TableContractRef,
)
from engine.v2.data import catalog, manifests
from engine.v2.data.documents import decode_document
from engine.v2.data.legacy_mapping import build_legacy_mapping
from engine.v2.data.objects import FragmentInspection, inspect_fragment
from engine.v2.data.query import ARROW_TYPES
from engine.v2.foundation import ArtifactStore, content_hash
from engine.v2.ops.bootstrap import open_catalog
from tests.ops_support import FakeClock

_MAPPING = build_legacy_mapping()


def fake_hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


RECEIPT = fake_hash("scan-receipt")
IMPORT_REQUEST_HASH = fake_hash("scan-import-request")


def contract_for(name: str) -> TableContract:
    return decode_document(TableContract, _MAPPING["tables"][name])


def contract_ref_for(contract: TableContract) -> TableContractRef:
    return TableContractRef(contract_id=contract.contract_id, definition_hash=contract.definition_hash)


def catalog_and_store(tmp_path: Path):
    clock = FakeClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    store = ArtifactStore(tmp_path / "store")
    return conn, clock, store


def table_from_rows(contract: TableContract, rows: list[dict], *, omit: frozenset = frozenset()) -> pa.Table:
    """A pyarrow table in contract column order, optionally dropping some
    columns entirely (a legacy-gap fixture: a nullable column absent from an
    old fragment's physical file, or — to reach a scan-time-only failure a
    normal ``inspect_fragment`` publish would already refuse — a required
    one)."""
    arrays = {}
    for column in contract.columns:
        if column.name in omit:
            continue
        physical = ARROW_TYPES.get(column.physical_type, pa.string())
        arrays[column.name] = pa.array([row.get(column.name) for row in rows], type=physical)
    return pa.table(arrays)


def to_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def publish_bytes(store: ArtifactStore, data: bytes) -> ObjectRef:
    ref = store.publish_bytes(data, schema_ref="parquet_fragment.v1")
    return ObjectRef(kind="parquet_fragment", object_id=ref.artifact_id,
                     content_hash=ref.content_hash, byte_size=ref.byte_size)


def publish_and_inspect(store: ArtifactStore, contract: TableContract, contract_ref: TableContractRef,
                        rows: list[dict], partition_key: str, *,
                        input_receipt_refs=(RECEIPT,), import_request_hash=IMPORT_REQUEST_HASH,
                        omit: frozenset = frozenset()) -> FragmentRecord:
    """The normal, fully-validated path: publish a real Parquet file for
    ``rows`` and run it through ``inspect_fragment`` (as
    ``objects.publish_legacy_file``/``inspect_fragment`` would at import
    time)."""
    table = table_from_rows(contract, rows, omit=omit)
    obj = publish_bytes(store, to_bytes(table))
    inspection = inspect_fragment(store, obj, contract, contract_ref, partition_key)
    return manifests.fragment_record(inspection, contract_ref, input_receipt_refs=input_receipt_refs,
                                     import_request_hash=import_request_hash)


def hand_built_record(store: ArtifactStore, contract: TableContract, contract_ref: TableContractRef,
                      table: pa.Table, *, partition_key: str, row_count: int,
                      primary_key_min: tuple, primary_key_max: tuple,
                      time_min: str | None = None, time_max: str | None = None,
                      logical_label: str = "hand-built") -> FragmentRecord:
    """A real published object (whatever ``table`` actually contains, which
    may deliberately violate ``contract`` — e.g. a missing required column),
    paired with a *hand-built* ``FragmentInspection`` (real byte hash, fixed
    everything else) rather than one that has passed ``inspect_fragment``'s
    own contract check. Only for a scan-time-only failure a normal publish
    would already refuse before this repository package ever sees it.
    """
    obj = publish_bytes(store, to_bytes(table))
    inspection = FragmentInspection(
        object_ref=obj, partition_key=partition_key, row_count=row_count, byte_hash=obj.content_hash,
        logical_content_hash=content_hash({"label": logical_label}),
        primary_key_min=primary_key_min, primary_key_max=primary_key_max,
        time_min=time_min, time_max=time_max)
    return manifests.fragment_record(inspection, contract_ref, input_receipt_refs=(RECEIPT,),
                                     import_request_hash=IMPORT_REQUEST_HASH)


def commit_tables(conn, clock, tables: dict[str, list[FragmentRecord]],
                  contracts: dict[str, TableContract], *, scope: str = "shadow",
                  receipt_id: str = "r1", attempt_id: str = "att-1", fence: int = 1,
                  knowledge_mode: str = "reconstructed",
                  partition_logical_hashes: dict[str, dict[str, str]] | None = None,
                  store: ArtifactStore | None = None) -> SnapshotRef:
    """One fresh snapshot over ``tables`` (``{table_name: [FragmentRecord, ...]}``,
    already in ascending ``partition_key`` order per table).

    ``partition_logical_hashes``, when given, is ``{table_name: {partition_key:
    hash}}`` — required for any table whose ``records`` include a
    multi-fragment partition (``manifests.dataset_manifest``'s own
    requirement); every other table's ``None`` default is unchanged.
    """
    table_manifests: dict[str, DatasetManifest] = {}
    all_records: list[FragmentRecord] = []
    all_objects = []
    hashes_by_table = partition_logical_hashes or {}
    for table_name, records in tables.items():
        contract_ref = contract_ref_for(contracts[table_name])
        table_manifests[table_name] = manifests.dataset_manifest(
            contract_ref, records, knowledge_mode=knowledge_mode, coverage_receipt_refs=(RECEIPT,),
            availability_evidence_refs=(), partition_logical_hashes=hashes_by_table.get(table_name))
        all_records.extend(records)
        all_objects.extend(r.object_ref for r in records)
    snap = manifests.snapshot_ref(
        table_manifests, calendar_version="cal.v1", source_priority_version="prio.v1",
        finality_receipt_refs=(RECEIPT,))

    def _noop_fence(_conn) -> None:
        return None

    receipt = catalog.commit_snapshot(
        conn, scope=scope, request_hash=fake_hash(f"{receipt_id}-request"),
        contracts=list(contracts.values()), objects=all_objects, records=all_records,
        manifests=list(table_manifests.values()), snapshot=snap, expected_head_snapshot_id=None,
        expected_head_generation=0, receipt_id=receipt_id, attempt_id=attempt_id, fence=fence,
        fence_check=_noop_fence, clock=clock, store=store)
    return receipt.snapshot_ref
