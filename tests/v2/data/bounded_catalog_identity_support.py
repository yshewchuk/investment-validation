"""Synthetic native-catalog fixtures without materialized objects."""
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

from engine.v2.contracts.data import ObjectRef, TableContractRef
from engine.v2.data import catalog, manifests, schema
from engine.v2.data._bounded_catalog_transfer import transfer_catalog
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT
from engine.v2.data.objects import FragmentInspection
from engine.v2.data.repository import Repository
from engine.v2.foundation import content_hash

LIMITS = dict(max_catalog_tables=3, max_catalog_fragments=4, max_catalog_bytes=100_000,
              max_fragments=2, max_partitions=2)


def contract(name):
    value = replace(IV_CRUSH_CONTRACT, table_name=name, contract_id=f"synthetic.{name}.v2")
    return replace(value, definition_hash=manifests.table_contract_hash(value))


def record(definition, partition, *, first="2020-01-01", last="2020-01-02", rows=2):
    digest = content_hash({"synthetic-object": definition.table_name})
    obj = ObjectRef(kind="parquet_fragment", object_id=f"synthetic-{definition.table_name}",
                    content_hash=digest, byte_size=128)
    inspection = FragmentInspection(object_ref=obj, partition_key=partition, row_count=rows,
        byte_hash=digest, logical_content_hash=content_hash({"synthetic-partition": partition}),
        primary_key_min=("FIX", first), primary_key_max=("FIX", last),
        time_min=None, time_max=None)
    ref = TableContractRef(contract_id=definition.contract_id,
                           definition_hash=definition.definition_hash)
    return manifests.fragment_record(inspection, ref, input_receipt_refs=("synthetic-input",),
                                     import_request_hash=content_hash({"synthetic-import": 1}))


def manifest(definition, records, **changes):
    return manifests.dataset_manifest(
        TableContractRef(contract_id=definition.contract_id,
                         definition_hash=definition.definition_hash), records,
        **{"knowledge_mode": "reconstructed", "coverage_receipt_refs": ("synthetic-coverage",),
           "availability_evidence_refs": ("synthetic-availability",), **changes})


def valid_catalog():
    """Native writer/oracle, production SQL schema, metadata only; caller closes."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    for migration in schema.MIGRATIONS:
        for statement in migration[2]:
            conn.execute(statement)
    conn.commit()
    contracts = tuple(contract(name) for name in ("zeta", "empty", "alpha"))
    by_table = {definition.table_name: (() if definition.table_name == "empty" else
                tuple(record(definition, part) for part in ("2020", "2021")))
                for definition in contracts}
    table_manifests = {definition.table_name: manifest(definition, by_table[definition.table_name])
                       for definition in contracts}
    snapshot = manifests.snapshot_ref(table_manifests, calendar_version="synthetic-calendar",
        source_priority_version="synthetic-priority", finality_receipt_refs=("synthetic-finality",))
    records = tuple(record for group in by_table.values() for record in group)
    objects = tuple({record.object_ref.object_id: record.object_ref for record in records}.values())
    catalog.commit_snapshot(conn, scope="synthetic", request_hash=content_hash({"request": 1}),
        contracts=contracts, objects=objects, records=records, manifests=tuple(table_manifests.values()),
        snapshot=snapshot, expected_head_snapshot_id=None, expected_head_generation=0,
        receipt_id="synthetic-receipt", attempt_id="synthetic-attempt", fence=1,
        fence_check=lambda _: None, clock=SimpleNamespace(now=lambda: datetime(2024, 1, 1, tzinfo=UTC)),
        audit_partitions=False)
    oracle = Repository(conn).resolve_full(snapshot.snapshot_id)
    conn.execute("BEGIN")
    payload = transfer_catalog(conn, snapshot.snapshot_id, "alpha", LIMITS)
    conn.rollback()
    return conn, snapshot, oracle, payload, by_table


def normalized_snapshot_impostor(snapshot, kind):
    """Malformed native holder that roundtrips to the unchanged valid reference."""
    from engine.v2.foundation import to_document
    if kind == "receipt_list":
        return replace(snapshot, finality_receipt_refs=list(snapshot.finality_receipt_refs))
    versions = dict(snapshot.table_versions)
    value = versions["alpha"]
    if kind == "dataset_dict":
        versions["alpha"] = to_document(value)
    elif kind == "contract_dict":
        versions["alpha"] = replace(value, table_contract_ref=to_document(value.table_contract_ref))
    elif kind == "nested_subclass":
        class AlternateRef(type(value)):
            pass
        versions["alpha"] = AlternateRef(**vars(value))
    else:
        class AliasKey:
            def __str__(self):
                return "alpha"
        versions[AliasKey()] = versions.pop("alpha")
    return replace(snapshot, table_versions=versions)
