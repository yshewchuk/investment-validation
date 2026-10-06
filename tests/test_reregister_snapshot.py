"""tools.reregister_snapshot: the neutral, metadata-only catalog inventory.

Most tests build a real SQLite catalog by raw SQL (the same hand-built-chain
style as ``tests/test_v2_data_catalog.py``); one commits a genuine synthetic
catalog through ``tests/data_scan_support.py``'s manifest builders, and one
drives a real second SQLite writer under WAL. There are no mocks of the
module's data source; the only substitutions are tampered rows, a bare
database, an authorizer and a patched ``decode_document``, all to prove
refusals and the "never reads ``data_contracts``, never decodes a contract
document" boundary.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pytest

from engine.v2.data import documents
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import CONTENT_HASH_PREFIX, content_hash, format_timestamp
from engine.v2.ops.bootstrap import open_catalog
from tests.data_scan_support import (
    RECEIPT,
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
    publish_bytes,
)
from tests.ops_support import FakeClock
from tools import reregister_snapshot


def H(label: str) -> str:
    return content_hash({"label": label})


def _ts(clock: FakeClock) -> str:
    return format_timestamp(clock.now())


def insert_contract(conn, clock, *, contract_id, table_name):
    conn.execute(
        "INSERT INTO data_contracts (contract_id, schema_version, table_name,"
        " definition_hash, definition_json, registered_at) VALUES (?, ?, ?, ?, ?, ?)",
        (contract_id, "table_contract.v1.0", table_name, H(contract_id),
         json.dumps({"table_name": table_name}), _ts(clock)),
    )


def insert_object(conn, clock, *, object_id, object_hash, byte_size):
    conn.execute(
        "INSERT INTO data_objects (object_id, kind, content_hash, byte_size,"
        " storage_key, registered_at) VALUES (?, 'parquet', ?, ?, ?, ?)",
        (object_id, object_hash, byte_size, f"objects/{object_id}", _ts(clock)),
    )


def insert_fragment(conn, clock, *, fragment_id, object_id, contract_id, byte_hash,
                    partition_key, row_count, key_bounds=None, time_bounds=None):
    bounds = key_bounds or {"primary_key_min": ["a"], "primary_key_max": ["z"]}
    conn.execute(
        "INSERT INTO data_fragments (fragment_id, object_id, contract_id, partition_key,"
        " row_count, byte_hash, logical_content_hash, key_bounds_json, time_bounds_json,"
        " import_request_hash, registered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (fragment_id, object_id, contract_id, partition_key, row_count, byte_hash,
         H(fragment_id + "-logical"), json.dumps(bounds),
         None if time_bounds is None else json.dumps(time_bounds),
         H(fragment_id + "-request"), _ts(clock)),
    )


def insert_dataset_version(conn, clock, *, dataset_version_id, contract_id, row_count,
                           knowledge_mode="reconstructed", evidence=None):
    evidence = evidence if evidence is not None else {
        "coverage_receipt_refs": [], "availability_evidence_refs": [],
    }
    conn.execute(
        "INSERT INTO data_dataset_versions (dataset_version_id, contract_id,"
        " parent_dataset_version_id, manifest_hash, logical_content_hash, row_count,"
        " knowledge_mode, evidence_json, registered_at)"
        " VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?)",
        (dataset_version_id, contract_id, H(dataset_version_id + "-manifest"),
         H(dataset_version_id + "-logical"), row_count, knowledge_mode,
         json.dumps(evidence), _ts(clock)),
    )


def insert_membership(conn, *, dataset_version_id, fragment_id, ordinal):
    conn.execute(
        "INSERT INTO data_version_fragments (dataset_version_id, ordinal, fragment_id)"
        " VALUES (?, ?, ?)",
        (dataset_version_id, ordinal, fragment_id),
    )


def insert_snapshot(conn, clock, *, snapshot_id, knowledge_mode_by_table, finality=(),
                    calendar_version="calendar.v1",
                    source_priority_version="priority.v1"):
    conn.execute(
        "INSERT INTO data_snapshots (snapshot_id, parent_snapshot_id, manifest_hash,"
        " calendar_version, source_priority_version, finality_receipt_refs_json,"
        " knowledge_mode_by_table_json, commit_receipt_ref, registered_at)"
        " VALUES (?, NULL, ?, ?, ?, ?, ?, 'receipt-pending', ?)",
        (snapshot_id, H(snapshot_id + "-manifest"), calendar_version,
         source_priority_version, json.dumps(list(finality)),
         json.dumps(knowledge_mode_by_table), _ts(clock)),
    )


def insert_snapshot_table(conn, *, snapshot_id, table_name, dataset_version_id):
    conn.execute(
        "INSERT INTO data_snapshot_tables (snapshot_id, table_name, dataset_version_id)"
        " VALUES (?, ?, ?)",
        (snapshot_id, table_name, dataset_version_id),
    )


def insert_receipt(conn, clock, *, receipt_id, scope, status="committed",
                   result_snapshot_id=None):
    conn.execute(
        "INSERT INTO data_import_receipts (receipt_id, attempt_id, fence,"
        " source_manifest_hash, result_snapshot_id, status, problem_json, registered_at,"
        " scope) VALUES (?, ?, 1, ?, ?, ?, NULL, ?, ?)",
        (receipt_id, receipt_id + "-attempt", H(receipt_id + "-source"),
         result_snapshot_id, status, _ts(clock), scope),
    )


def insert_head(conn, clock, *, scope, snapshot_id, generation, receipt_ref):
    conn.execute(
        "INSERT INTO data_snapshot_heads (scope, snapshot_id, generation, updated_at,"
        " update_receipt_ref) VALUES (?, ?, ?, ?, ?)",
        (scope, snapshot_id, generation, _ts(clock), receipt_ref),
    )


def insert_reference(conn, *, receipt_id, legacy_path, kind, object_id, object_hash,
                     byte_size, fold=""):
    conn.execute(
        "INSERT INTO data_import_reference_inputs (receipt_id, legacy_path, kind,"
        " object_id, content_hash, byte_size, fold) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (receipt_id, legacy_path, kind, object_id, object_hash, byte_size, fold),
    )


def insert_lineage(conn, *, receipt_id, base_receipt_id):
    conn.execute(
        "INSERT INTO data_receipt_lineage (receipt_id, base_receipt_id, kind)"
        " VALUES (?, ?, 'price_history_capture')",
        (receipt_id, base_receipt_id),
    )


def insert_price_capture(conn, *, capture_id, receipt_id, contract_id, ticker="SPY",
                         source_kind="csv", outcome="added", rows_added=0,
                         rows_tombstoned=0, stamp="2026-09-12T00:00:00.000000Z"):
    conn.execute(
        "INSERT INTO data_price_captures (capture_id, receipt_id, ticker, source_kind,"
        " source_hash, retrieved_at, outcome, rows_added, rows_tombstoned, created_at,"
        " contract_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (capture_id, receipt_id, ticker, source_kind, H(capture_id + "-src"), stamp,
         outcome, rows_added, rows_tombstoned, stamp, contract_id),
    )


def insert_computed_capture(conn, *, capture_id, contract_id, ticker="SPY", outcome="added"):
    conn.execute(
        "INSERT INTO data_computed_moves_captures (capture_id, ticker, created_at,"
        " contract_id, outcome) VALUES (?, ?, ?, ?, ?)",
        (capture_id, ticker, "2026-09-12T00:00:00+00:00", contract_id, outcome),
    )


def build_chain(tmp_path, *, tables, scope="shadow", snapshot_id="snap-1",
                receipt_id="receipt-1", generation=1, status="committed", finality=(),
                references=(), head=True, calendar_version="calendar.v1",
                source_priority_version="priority.v1", name="catalog.sqlite"):
    """One real snapshot chain (contracts through receipt/head) from raw SQL."""
    clock = FakeClock()
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = open_catalog(path, clock=clock)
    chain = {"conn": conn, "clock": clock, "path": path, "scope": scope,
             "snapshot_id": snapshot_id, "receipt_id": receipt_id,
             "generation": generation}
    modes, snapshot_tables = {}, []
    for spec in tables:
        table_name = spec["name"]
        contract_id = spec.get("contract", table_name + ".v1")
        fragments = spec["fragments"] if "fragments" in spec else [{}]
        mode = spec.get("mode", "reconstructed")
        insert_contract(conn, clock, contract_id=contract_id, table_name=table_name)
        row_count = sum(frag.get("row_count", 10) for frag in fragments)
        dsv_id = table_name + "-dsv"
        insert_dataset_version(conn, clock, dataset_version_id=dsv_id,
                               contract_id=contract_id, row_count=row_count,
                               knowledge_mode=mode, evidence=spec.get("evidence"))
        for ordinal, frag in enumerate(fragments):
            object_id = f"{table_name}-obj-{ordinal}"
            object_hash = H(object_id)
            insert_object(conn, clock, object_id=object_id, object_hash=object_hash,
                          byte_size=frag.get("byte_size", 100))
            insert_fragment(conn, clock, fragment_id=f"{table_name}-frag-{ordinal}",
                            object_id=object_id, contract_id=contract_id,
                            byte_hash=object_hash,
                            partition_key=frag.get("partition_key", "2026"),
                            row_count=frag.get("row_count", 10),
                            key_bounds=frag.get("key_bounds"),
                            time_bounds=frag.get("time_bounds"))
            insert_membership(conn, dataset_version_id=dsv_id,
                              fragment_id=f"{table_name}-frag-{ordinal}", ordinal=ordinal)
        modes[table_name] = mode
        snapshot_tables.append((table_name, dsv_id))
    insert_snapshot(conn, clock, snapshot_id=snapshot_id,
                    knowledge_mode_by_table=modes, finality=finality,
                    calendar_version=calendar_version,
                    source_priority_version=source_priority_version)
    for table_name, dsv_id in snapshot_tables:
        insert_snapshot_table(conn, snapshot_id=snapshot_id, table_name=table_name,
                              dataset_version_id=dsv_id)
    insert_receipt(conn, clock, receipt_id=receipt_id, scope=scope, status=status,
                   result_snapshot_id=snapshot_id if status == "committed" else None)
    if head:
        insert_head(conn, clock, scope=scope, snapshot_id=snapshot_id,
                    generation=generation, receipt_ref=receipt_id)
    for reference in references:
        insert_reference(conn, receipt_id=receipt_id, **reference)
    return chain


def inventory(chain, **changed):
    kwargs = {"scope": chain["scope"], "snapshot_id": chain["snapshot_id"],
              "receipt_id": chain["receipt_id"], "generation": chain["generation"]}
    kwargs.update(changed)
    return reregister_snapshot.neutral_inventory(chain["conn"], **kwargs)


def expect_refusal(chain, code, **changed):
    with pytest.raises(DataError) as excinfo:
        inventory(chain, **changed)
    assert excinfo.value.code == code
    assert excinfo.value.problem.details == {}
    return excinfo.value.problem


def test_exact_neutral_payload_projection_and_determinism(tmp_path):
    chain = build_chain(
        tmp_path,
        tables=[{
            "name": "price_history", "contract": "price_history.v1",
            "evidence": {"coverage_receipt_refs": ["cov-1"],
                         "availability_evidence_refs": ["avail-1"]},
            "fragments": [
                {"partition_key": "2026", "row_count": 7, "byte_size": 700,
                 "key_bounds": {"primary_key_min": ["SPY", "2026-01-01"],
                                "primary_key_max": ["SPY", "2026-12-31"]},
                 "time_bounds": {"time_min": "2026-01-01", "time_max": "2026-12-31"}},
                {"partition_key": "2025", "row_count": 3, "byte_size": 300,
                 "key_bounds": {"primary_key_min": ["SPY", "2025-01-01"],
                                "primary_key_max": ["SPY", "2025-12-31"]},
                 "time_bounds": None},
            ]}],
        finality=["receipt-final"],
        references=[
            {"legacy_path": "b/second.csv", "kind": "calendar_csv", "object_id": "ref-b",
             "object_hash": H("ref-b"), "byte_size": 20, "fold": "202609"},
            {"legacy_path": "a/first.csv", "kind": "model_registry", "object_id": "ref-a",
             "object_hash": H("ref-a"), "byte_size": 10, "fold": ""},
        ],
    )
    conn, clock = chain["conn"], chain["clock"]
    insert_receipt(conn, clock, receipt_id="receipt-base", scope="shadow",
                   result_snapshot_id="snap-1")
    insert_lineage(conn, receipt_id="receipt-1", base_receipt_id="receipt-base")
    stamp = _ts(clock)
    insert_price_capture(conn, capture_id="cap-2", receipt_id="receipt-base",
                         contract_id="price_history.v1", rows_added=2, stamp=stamp)
    insert_price_capture(conn, capture_id="cap-1", receipt_id="receipt-base",
                         contract_id="price_history.v1", rows_added=5,
                         rows_tombstoned=1, stamp=stamp)
    payload = inventory(chain)
    assert payload == {
        "schema_version": "reregister_snapshot.v1",
        "scope": "shadow",
        "snapshot_id": "snap-1",
        "generation": 1,
        "receipt_id": "receipt-1",
        "calendar_version": "calendar.v1",
        "source_priority_version": "priority.v1",
        "finality_receipt_refs": ["receipt-final"],
        "tables": [{
            "table_name": "price_history",
            "dataset_version_id": "price_history-dsv",
            "contract_id": "price_history.v1",
            "knowledge_mode": "reconstructed",
            "coverage_receipt_refs": ["cov-1"],
            "availability_evidence_refs": ["avail-1"],
            "fragments": [
                {"fragment_id": "price_history-frag-0",
                 "object": {"kind": "parquet", "object_id": "price_history-obj-0",
                            "content_hash": H("price_history-obj-0"),
                            "byte_size": 700},
                 "partition_key": "2026", "row_count": 7,
                 "primary_key_min": ["SPY", "2026-01-01"],
                 "primary_key_max": ["SPY", "2026-12-31"],
                 "time_min": "2026-01-01", "time_max": "2026-12-31"},
                {"fragment_id": "price_history-frag-1",
                 "object": {"kind": "parquet", "object_id": "price_history-obj-1",
                            "content_hash": H("price_history-obj-1"),
                            "byte_size": 300},
                 "partition_key": "2025", "row_count": 3,
                 "primary_key_min": ["SPY", "2025-01-01"],
                 "primary_key_max": ["SPY", "2025-12-31"],
                 "time_min": None, "time_max": None},
            ]}],
        "references": [
            {"kind": "model_registry", "legacy_path": "a/first.csv",
             "object_id": "ref-a", "content_hash": H("ref-a"), "byte_size": 10,
             "fold": ""},
            {"kind": "calendar_csv", "legacy_path": "b/second.csv",
             "object_id": "ref-b", "content_hash": H("ref-b"), "byte_size": 20,
             "fold": "202609"},
        ],
        "lineage": {"receipt_ids": ["receipt-1", "receipt-base"],
                    "edges": [["receipt-1", "receipt-base"]]},
        "captures": {"price_history": [
            {"capture_id": "cap-1", "receipt_id": "receipt-base", "ticker": "SPY",
             "source_kind": "csv", "source_hash": H("cap-1-src"),
             "retrieved_at": stamp, "outcome": "added", "rows_added": 5,
             "rows_tombstoned": 1, "created_at": stamp,
             "contract_id": "price_history.v1"},
            {"capture_id": "cap-2", "receipt_id": "receipt-base", "ticker": "SPY",
             "source_kind": "csv", "source_hash": H("cap-2-src"),
             "retrieved_at": stamp, "outcome": "added", "rows_added": 2,
             "rows_tombstoned": 0, "created_at": stamp,
             "contract_id": "price_history.v1"},
        ]},
    }
    again = inventory(chain)
    assert again == payload
    assert content_hash(again) == content_hash(payload)


def test_planted_membership_ordinal_drift_changes_the_inventory(tmp_path):
    chain = build_chain(tmp_path, tables=[{
        "name": "price_history", "contract": "price_history.v1",
        "fragments": [
            {"partition_key": "2026", "row_count": 4, "byte_size": 40,
             "key_bounds": {"primary_key_min": ["A"], "primary_key_max": ["A"]}},
            {"partition_key": "2025", "row_count": 6, "byte_size": 60,
             "key_bounds": {"primary_key_min": ["B"], "primary_key_max": ["B"]}},
        ]}])
    baseline = inventory(chain)
    fragments = baseline["tables"][0]["fragments"]
    assert [fragment["fragment_id"] for fragment in fragments] == [
        "price_history-frag-0", "price_history-frag-1"]
    conn = chain["conn"]
    conn.execute("DROP TRIGGER data_version_fragments_no_delete")
    conn.execute("DELETE FROM data_version_fragments WHERE dataset_version_id = ?",
                 ("price_history-dsv",))
    insert_membership(conn, dataset_version_id="price_history-dsv",
                      fragment_id="price_history-frag-1", ordinal=0)
    insert_membership(conn, dataset_version_id="price_history-dsv",
                      fragment_id="price_history-frag-0", ordinal=1)
    drifted = inventory(chain)
    drifted_fragments = drifted["tables"][0]["fragments"]
    assert sorted(fragment["fragment_id"] for fragment in drifted_fragments) == sorted(
        fragment["fragment_id"] for fragment in fragments)
    assert [fragment["fragment_id"] for fragment in drifted_fragments] == [
        "price_history-frag-1", "price_history-frag-0"]
    assert content_hash(drifted) != content_hash(baseline)


def test_empty_references_and_captures_only_for_present_native_tables(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    insert_price_capture(chain["conn"], capture_id="cap-1", receipt_id="receipt-1",
                         contract_id="price_history.v1")
    insert_computed_capture(chain["conn"], capture_id="cm-1",
                            contract_id="daily_market.v1")
    payload = inventory(chain)
    assert payload["references"] == []
    assert payload["captures"] == {}
    assert payload["lineage"] == {"receipt_ids": ["receipt-1"], "edges": []}


def test_explicit_empty_table_stays_zero_fragment_and_zero_row(tmp_path):
    chain = build_chain(tmp_path, tables=[
        {"name": "daily_market", "fragments": []},
        {"name": "price_history", "contract": "price_history.v1",
         "fragments": [{"partition_key": "2026", "row_count": 2, "byte_size": 20,
                        "key_bounds": {"primary_key_min": ["SPY"],
                                       "primary_key_max": ["SPY"]}}]},
    ])
    payload = inventory(chain)
    assert payload == {
        "schema_version": "reregister_snapshot.v1",
        "scope": "shadow",
        "snapshot_id": "snap-1",
        "generation": 1,
        "receipt_id": "receipt-1",
        "calendar_version": "calendar.v1",
        "source_priority_version": "priority.v1",
        "finality_receipt_refs": [],
        "tables": [
            {"table_name": "daily_market", "dataset_version_id": "daily_market-dsv",
             "contract_id": "daily_market.v1", "knowledge_mode": "reconstructed",
             "coverage_receipt_refs": [], "availability_evidence_refs": [],
             "fragments": []},
            {"table_name": "price_history", "dataset_version_id": "price_history-dsv",
             "contract_id": "price_history.v1", "knowledge_mode": "reconstructed",
             "coverage_receipt_refs": [], "availability_evidence_refs": [],
             "fragments": [{
                 "fragment_id": "price_history-frag-0",
                 "object": {"kind": "parquet", "object_id": "price_history-obj-0",
                            "content_hash": H("price_history-obj-0"), "byte_size": 20},
                 "partition_key": "2026", "row_count": 2,
                 "primary_key_min": ["SPY"], "primary_key_max": ["SPY"],
                 "time_min": None, "time_max": None}]},
        ],
        "references": [],
        "lineage": {"receipt_ids": ["receipt-1"], "edges": []},
        "captures": {"price_history": []},
    }
    again = inventory(chain)
    assert again == payload
    assert content_hash(again) == content_hash(payload)


def test_capture_scopes_and_descriptors(tmp_path):
    chain = build_chain(tmp_path, tables=[
        {"name": "price_history", "contract": "price_history.v1"},
        {"name": "computed_moves", "contract": "computed_moves.v1"}])
    conn, clock = chain["conn"], chain["clock"]
    insert_receipt(conn, clock, receipt_id="receipt-base", scope="shadow",
                   result_snapshot_id="snap-1")
    insert_receipt(conn, clock, receipt_id="receipt-other", scope="shadow",
                   result_snapshot_id="snap-1")
    insert_lineage(conn, receipt_id="receipt-1", base_receipt_id="receipt-base")
    insert_contract(conn, clock, contract_id="computed_moves.v2",
                    table_name="computed_moves")
    insert_price_capture(conn, capture_id="price-in-lineage", receipt_id="receipt-base",
                         contract_id="price_history.v1")
    insert_price_capture(conn, capture_id="price-pinned", receipt_id="receipt-1",
                         contract_id="price_history.v1")
    insert_price_capture(conn, capture_id="price-outside", receipt_id="receipt-other",
                         contract_id="price_history.v1")
    insert_price_capture(conn, capture_id="price-other-contract", receipt_id="receipt-base",
                         contract_id="price_history.v2")
    insert_computed_capture(conn, capture_id="cm-in", contract_id="computed_moves.v1")
    insert_computed_capture(conn, capture_id="cm-other-contract",
                            contract_id="computed_moves.v2")
    payload = inventory(chain)
    assert [row["capture_id"] for row in payload["captures"]["price_history"]] == [
        "price-in-lineage", "price-pinned"]
    assert payload["captures"]["price_history"][0]["contract_id"] == "price_history.v1"
    assert [row["capture_id"] for row in payload["captures"]["computed_moves"]] == ["cm-in"]
    assert payload["lineage"] == {"receipt_ids": ["receipt-1", "receipt-base"],
                                  "edges": [["receipt-1", "receipt-base"]]}


def test_head_advance_does_not_move_the_pinned_payload(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    before = inventory(chain)
    conn, clock = chain["conn"], chain["clock"]
    insert_snapshot(conn, clock, snapshot_id="snap-2",
                    knowledge_mode_by_table={"daily_market": "reconstructed"})
    insert_snapshot_table(conn, snapshot_id="snap-2", table_name="daily_market",
                          dataset_version_id="daily_market-dsv")
    insert_receipt(conn, clock, receipt_id="receipt-2", scope="shadow",
                   result_snapshot_id="snap-2")
    conn.execute(
        "UPDATE data_snapshot_heads SET snapshot_id = 'snap-2', generation = 2,"
        " updated_at = ?, update_receipt_ref = 'receipt-2' WHERE scope = 'shadow'",
        (_ts(clock),))
    assert tuple(conn.execute(
        "SELECT generation, snapshot_id FROM data_snapshot_heads").fetchone()) == (2, "snap-2")
    after = inventory(chain)
    assert after == before
    assert content_hash(after) == content_hash(before)


def test_every_pinned_metadata_family_changes_the_hash(tmp_path):
    def spec():
        return {
            "tables": [{
                "name": "price_history", "contract": "price_history.v1",
                "evidence": {"coverage_receipt_refs": [],
                             "availability_evidence_refs": ["a"]},
                "fragments": [{"partition_key": "2026", "row_count": 10,
                               "byte_size": 100}],
            }, {
                "name": "daily_market",
                "fragments": [{"partition_key": "2026", "row_count": 4,
                               "byte_size": 40}],
            }],
            "finality": ["receipt-final"],
            "references": [{"legacy_path": "ref.csv", "kind": "calendar_csv",
                            "object_id": "ref-1", "object_hash": H("ref-1"),
                            "byte_size": 10, "fold": ""}],
        }

    def digest(chain):
        return content_hash(inventory(chain))

    def build(name, mutate=None):
        kwargs = spec()
        if mutate is not None:
            mutate(kwargs)
        return build_chain(tmp_path, name=name, **kwargs)

    base = digest(build("base/catalog.sqlite"))

    def variant(label, mutate):
        assert digest(build(f"{label}/catalog.sqlite", mutate)) != base, label

    variant("finality", lambda k: k.update(finality=["receipt-other"]))
    variant("calendar", lambda k: k.update(calendar_version="calendar.v2"))
    variant("priority", lambda k: k.update(source_priority_version="priority.v2"))
    variant("reference", lambda k: k["references"][0].update(byte_size=11))
    variant("reference_fold", lambda k: k["references"][0].update(fold="202609"))
    variant("evidence",
            lambda k: k["tables"][0]["evidence"].update(coverage_receipt_refs=["cov"]))
    variant("partition", lambda k: k["tables"][0]["fragments"][0].update(
        partition_key="2025"))
    variant("object_size", lambda k: k["tables"][0]["fragments"][0].update(byte_size=101))
    variant("time_bounds", lambda k: k["tables"][0]["fragments"][0].update(
        time_bounds={"time_min": "2026-01-01", "time_max": "2026-01-02"}))
    variant("knowledge_mode", lambda k: k["tables"][0].update(
        mode="observed",
        evidence={"coverage_receipt_refs": [],
                  "availability_evidence_refs": ["a"]}))
    variant("table_removed", lambda k: k["tables"].pop())

    chained = build("lineage/catalog.sqlite")
    insert_receipt(chained["conn"], chained["clock"], receipt_id="receipt-base",
                   scope="shadow", result_snapshot_id="snap-1")
    insert_lineage(chained["conn"], receipt_id="receipt-1",
                   base_receipt_id="receipt-base")
    assert digest(chained) != base, "lineage"

    capped = build("capture/catalog.sqlite")
    insert_price_capture(capped["conn"], capture_id="cap-1", receipt_id="receipt-1",
                         contract_id="price_history.v1")
    assert digest(capped) != base, "capture"


def test_pin_validation_and_membership_refusals(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    for bad in (0, -1, True, "1", 1.0, None):
        expect_refusal(chain, "INPUT_CHANGED", generation=bad)
    for field in ("scope", "snapshot_id", "receipt_id"):
        expect_refusal(chain, "INPUT_CHANGED", **{field: ""})
        expect_refusal(chain, "INPUT_CHANGED", **{field: None})
    expect_refusal(chain, "INPUT_CHANGED", scope="other")
    expect_refusal(chain, "INPUT_CHANGED", snapshot_id="snap-other")
    expect_refusal(chain, "INPUT_CHANGED", receipt_id="receipt-unknown")


def test_failed_pinned_receipt_is_refused(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}], status="failed",
                        head=False)
    expect_refusal(chain, "INPUT_CHANGED")


def test_missing_dataset_version_row_is_input_changed(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    conn = chain["conn"]
    conn.execute("DROP TRIGGER data_dataset_versions_no_delete")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DELETE FROM data_dataset_versions WHERE dataset_version_id = 'daily_market-dsv'")
    conn.execute("PRAGMA foreign_keys = ON")
    expect_refusal(chain, "INPUT_CHANGED")


def test_missing_fragment_row_is_input_changed(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    conn = chain["conn"]
    conn.execute("DROP TRIGGER data_fragments_no_delete")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DELETE FROM data_fragments WHERE fragment_id = 'daily_market-frag-0'")
    conn.execute("PRAGMA foreign_keys = ON")
    expect_refusal(chain, "INPUT_CHANGED")


def test_missing_object_row_is_input_changed(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    conn = chain["conn"]
    conn.execute("DROP TRIGGER data_objects_no_delete")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DELETE FROM data_objects WHERE object_id = 'daily_market-obj-0'")
    conn.execute("PRAGMA foreign_keys = ON")
    expect_refusal(chain, "INPUT_CHANGED")


def test_membership_removal_with_surviving_row_count_is_manifest_corrupt(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    chain["conn"].execute("DROP TRIGGER data_version_fragments_no_delete")
    chain["conn"].execute(
        "DELETE FROM data_version_fragments WHERE dataset_version_id = 'daily_market-dsv'")
    problem = expect_refusal(chain, "MANIFEST_CORRUPT")
    assert problem.message == "stored dataset row count is inconsistent"


def test_fragment_identity_mismatch_is_manifest_corrupt(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    conn = chain["conn"]
    conn.execute("DROP TRIGGER data_fragments_no_update")
    conn.execute(
        "UPDATE data_fragments SET byte_hash = ? WHERE fragment_id = 'daily_market-frag-0'",
        (H("other-bytes"),))
    problem = expect_refusal(chain, "MANIFEST_CORRUPT")
    assert problem.message == "stored fragment identity is inconsistent"


def test_fragment_contract_mismatch_is_manifest_corrupt(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    conn, clock = chain["conn"], chain["clock"]
    insert_contract(conn, clock, contract_id="daily_market.v2", table_name="daily_market")
    conn.execute("DROP TRIGGER data_fragments_no_update")
    conn.execute(
        "UPDATE data_fragments SET contract_id = 'daily_market.v2'"
        " WHERE fragment_id = 'daily_market-frag-0'")
    expect_refusal(chain, "MANIFEST_CORRUPT")


_TAMPER = {
    "fragment_key": ("data_fragments_no_update", "data_fragments", "key_bounds_json"),
    "fragment_time": ("data_fragments_no_update", "data_fragments", "time_bounds_json"),
    "evidence": ("data_dataset_versions_no_update", "data_dataset_versions",
                 "evidence_json"),
    "finality": ("data_snapshots_no_update", "data_snapshots",
                 "finality_receipt_refs_json"),
}


@pytest.mark.parametrize("target,value,message", [
    ("fragment_key", "not json", "stored metadata is not valid JSON"),
    ("fragment_key", "{}", "stored metadata is missing a required field"),
    ("fragment_time", "[]", "stored metadata has an unexpected shape"),
    ("evidence", "not json", "stored metadata is not valid JSON"),
    ("evidence", "{}", "stored metadata is missing a required field"),
    ("finality", "not json", "stored metadata is not valid JSON"),
    ("finality", "{}", "stored metadata has an unexpected shape"),
])
def test_malformed_stored_metadata_is_manifest_corrupt(tmp_path, target, value, message):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    trigger, table, column = _TAMPER[target]
    conn = chain["conn"]
    conn.execute(f"DROP TRIGGER {trigger}")
    conn.execute("PRAGMA ignore_check_constraints = ON")
    conn.execute(f"UPDATE {table} SET {column} = ?", (value,))
    problem = expect_refusal(chain, "MANIFEST_CORRUPT")
    assert problem.message == message
    assert value not in problem.message
    assert str(tmp_path) not in problem.message


@pytest.mark.parametrize("target,value", [
    ("fragment_key", {"primary_key_min": [{"contract_id": "x"}],
                      "primary_key_max": ["z"]}),
    ("fragment_key", {"primary_key_min": [["a"]], "primary_key_max": ["z"]}),
    ("fragment_key", {"primary_key_min": [None], "primary_key_max": ["z"]}),
    ("fragment_time", {"time_min": {"contract_id": "x"}, "time_max": None}),
    ("fragment_time", {"time_min": None, "time_max": ["2026-01-01"]}),
    ("evidence", {"coverage_receipt_refs": [{"receipt_id": "r"}],
                  "availability_evidence_refs": []}),
    ("evidence", {"coverage_receipt_refs": [],
                  "availability_evidence_refs": [{"receipt_id": "r"}]}),
    ("finality", [{"receipt_id": "r"}]),
])
def test_embedded_contract_shapes_are_refused_without_writes(tmp_path, target, value):
    chain = build_chain(tmp_path, tables=[{"name": "price_history",
                                           "contract": "price_history.v1"}])
    conn = chain["conn"]
    trigger, table, column = _TAMPER[target]
    conn.execute(f"DROP TRIGGER {trigger}")
    conn.execute("PRAGMA ignore_check_constraints = ON")
    conn.execute(f"UPDATE {table} SET {column} = ?", (json.dumps(value),))
    before_changes = conn.total_changes
    before_files = sorted(path.name for path in chain["path"].parent.iterdir())
    problem = expect_refusal(chain, "MANIFEST_CORRUPT")
    assert problem.message == "stored metadata has an unexpected shape"
    assert str(tmp_path) not in problem.message
    assert conn.total_changes == before_changes
    assert sorted(path.name for path in chain["path"].parent.iterdir()) == before_files


@pytest.mark.parametrize("value", [
    {},
    {"time_max": "2026-01-01"},
    {"time_min": "2026-01-01"},
])
def test_time_bounds_object_missing_a_required_bound_is_manifest_corrupt(tmp_path, value):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    trigger, table, column = _TAMPER["fragment_time"]
    conn = chain["conn"]
    conn.execute(f"DROP TRIGGER {trigger}")
    conn.execute("PRAGMA ignore_check_constraints = ON")
    conn.execute(f"UPDATE {table} SET {column} = ?", (json.dumps(value),))
    before_changes = conn.total_changes
    before_files = sorted(path.name for path in chain["path"].parent.iterdir())
    problem = expect_refusal(chain, "MANIFEST_CORRUPT")
    assert problem.message == "stored metadata is missing a required field"
    assert str(tmp_path) not in problem.message
    assert conn.total_changes == before_changes
    assert sorted(path.name for path in chain["path"].parent.iterdir()) == before_files


def test_explicit_null_time_bounds_are_accepted(tmp_path):
    chain = build_chain(tmp_path, tables=[{
        "name": "price_history", "contract": "price_history.v1",
        "fragments": [
            {"partition_key": "2026", "row_count": 2, "byte_size": 20,
             "time_bounds": {"time_min": None, "time_max": None}},
            {"partition_key": "2025", "row_count": 3, "byte_size": 30,
             "time_bounds": {"time_min": None, "time_max": "2026-01-02"}},
            {"partition_key": "2024", "row_count": 4, "byte_size": 40,
             "time_bounds": {"time_min": "2026-01-01", "time_max": None}},
        ]}])
    payload = inventory(chain)
    fragments = payload["tables"][0]["fragments"]
    assert [(fragment["time_min"], fragment["time_max"]) for fragment in fragments] == [
        (None, None), (None, "2026-01-02"), ("2026-01-01", None)]
    assert fragments[0]["time_min"] is None and fragments[0]["time_max"] is None


def test_cyclic_lineage_is_refused(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    conn, clock = chain["conn"], chain["clock"]
    insert_receipt(conn, clock, receipt_id="anc-committed", scope="shadow",
                   result_snapshot_id="snap-1")
    insert_lineage(conn, receipt_id="receipt-1", base_receipt_id="anc-committed")
    insert_lineage(conn, receipt_id="anc-committed", base_receipt_id="receipt-1")
    expect_refusal(chain, "INPUT_CHANGED")


def test_uncommitted_lineage_ancestor_is_refused(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    conn, clock = chain["conn"], chain["clock"]
    insert_receipt(conn, clock, receipt_id="anc-failed", scope="", status="failed")
    insert_lineage(conn, receipt_id="receipt-1", base_receipt_id="anc-failed")
    expect_refusal(chain, "INPUT_CHANGED")


def test_missing_lineage_ancestor_is_refused(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    conn = chain["conn"]
    conn.execute("PRAGMA foreign_keys = OFF")
    insert_lineage(conn, receipt_id="receipt-1", base_receipt_id="anc-missing")
    conn.execute("PRAGMA foreign_keys = ON")
    expect_refusal(chain, "INPUT_CHANGED")


def test_lineage_depth_boundary(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    conn, clock = chain["conn"], chain["clock"]
    parent = "receipt-1"
    for index in range(63):
        ancestor = f"anc-{index}"
        insert_receipt(conn, clock, receipt_id=ancestor, scope="shadow",
                       result_snapshot_id="snap-1")
        insert_lineage(conn, receipt_id=parent, base_receipt_id=ancestor)
        parent = ancestor
    assert inventory(chain)["lineage"]["receipt_ids"][0] == "anc-0"
    ancestor = "anc-63"
    insert_receipt(conn, clock, receipt_id=ancestor, scope="shadow",
                   result_snapshot_id="snap-1")
    insert_lineage(conn, receipt_id=parent, base_receipt_id=ancestor)
    expect_refusal(chain, "INPUT_CHANGED")


def test_schema_problem_is_input_changed_without_details(tmp_path):
    path = tmp_path / "bare.sqlite"
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        with pytest.raises(DataError) as excinfo:
            reregister_snapshot.neutral_inventory(
                conn, scope="shadow", snapshot_id="snap-1", receipt_id="receipt-1",
                generation=1)
    finally:
        conn.close()
    assert excinfo.value.code == "INPUT_CHANGED"
    assert excinfo.value.problem.message == "catalog metadata is not readable"
    assert str(path) not in excinfo.value.problem.message


def test_active_caller_transaction_is_preserved(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    conn = chain["conn"]
    conn.execute("BEGIN")
    try:
        expect_refusal(chain, "INPUT_CHANGED")
        assert conn.in_transaction
    finally:
        conn.execute("ROLLBACK")
    assert not conn.in_transaction
    assert inventory(chain)["tables"]


def test_success_and_failure_leave_the_catalog_unchanged(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "price_history"}])
    conn = chain["conn"]
    before_changes = conn.total_changes
    before_files = sorted(path.name for path in chain["path"].parent.iterdir())
    assert inventory(chain)
    expect_refusal(chain, "INPUT_CHANGED", receipt_id="receipt-missing")
    assert conn.total_changes == before_changes
    assert sorted(path.name for path in chain["path"].parent.iterdir()) == before_files


def test_never_reads_contracts_or_decodes_documents(tmp_path, monkeypatch):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    conn = chain["conn"]
    reads = []

    def authorizer(action, arg1, arg2, dbname, source):
        if action == sqlite3.SQLITE_READ:
            reads.append(arg1)
            if arg1 == "data_contracts":
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def boom(*args, **kwargs):
        raise AssertionError("decode_document must never run")

    conn.set_authorizer(authorizer)
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("SELECT definition_json FROM data_contracts LIMIT 1")
    reads.clear()
    monkeypatch.setattr(documents, "decode_document", boom)
    payload = inventory(chain)
    assert payload["tables"]
    assert reads
    assert "data_contracts" not in reads


_SECURITIES_ROW = {
    "ticker": "AAA", "year": 2024, "first_date": datetime(2024, 1, 2),
    "last_date": datetime(2024, 12, 30), "mcap_usd": 1.5e9, "mcap_log": 21.1,
    "mcap_raw": 1.5, "mcap_unit_era": "billions", "mcap_quantized": False,
    "n_obs": 250, "src": "orats",
}


def test_committed_synthetic_catalog_matches_neutral_inventory(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    securities, empty = contract_for("securities"), contract_for("option_chains")
    record = publish_and_inspect(store, securities, contract_ref_for(securities),
                                [_SECURITIES_ROW], "2024")
    snap = commit_tables(
        conn, clock, {"option_chains": [], "securities": [record]},
        {"option_chains": empty, "securities": securities},
        scope="shadow", receipt_id="receipt-1", store=store)
    head = conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads"
                        " WHERE scope = 'shadow'").fetchone()
    assert (head["snapshot_id"], head["generation"]) == (snap.snapshot_id, 1)

    resolved = Repository(conn, store).resolve_full(snap.snapshot_id)
    assert set(resolved.table_manifests) == {"option_chains", "securities"}
    assert resolved.table_manifests["option_chains"].row_count == 0
    assert resolved.table_manifests["option_chains"].fragment_refs == ()
    assert resolved.records == (record,)

    reads = []

    def authorizer(action, arg1, arg2, dbname, source):
        if action == sqlite3.SQLITE_READ:
            reads.append(arg1)
            if arg1 == "data_contracts":
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def boom(*args, **kwargs):
        raise AssertionError("decode_document must never run")

    conn.set_authorizer(authorizer)
    monkeypatch.setattr(documents, "decode_document", boom)
    payload = reregister_snapshot.neutral_inventory(
        conn, scope="shadow", snapshot_id=snap.snapshot_id, receipt_id="receipt-1",
        generation=head["generation"])
    empty_table, populated_table = payload["tables"]
    assert empty_table == {
        "table_name": "option_chains",
        "dataset_version_id": (resolved.table_manifests["option_chains"]
                               .dataset_version_ref.dataset_version_id),
        "contract_id": empty.contract_id, "knowledge_mode": "reconstructed",
        "coverage_receipt_refs": [RECEIPT], "availability_evidence_refs": [],
        "fragments": [],
    }
    assert populated_table["table_name"] == "securities"
    assert populated_table["dataset_version_id"] == (
        resolved.table_manifests["securities"].dataset_version_ref.dataset_version_id)
    assert populated_table["contract_id"] == securities.contract_id
    assert populated_table["knowledge_mode"] == "reconstructed"
    assert populated_table["coverage_receipt_refs"] == [RECEIPT]
    assert populated_table["availability_evidence_refs"] == []
    assert populated_table["fragments"] == [{
        "fragment_id": record.fragment_id,
        "object": {"kind": record.object_ref.kind,
                   "object_id": record.object_ref.object_id,
                   "content_hash": record.object_ref.content_hash,
                   "byte_size": record.object_ref.byte_size},
        "partition_key": record.partition_key, "row_count": record.row_count,
        "primary_key_min": list(record.primary_key_min),
        "primary_key_max": list(record.primary_key_max),
        "time_min": record.time_min, "time_max": record.time_max,
    }]
    assert payload["references"] == []
    assert payload["captures"] == {}
    assert payload["lineage"] == {"receipt_ids": ["receipt-1"], "edges": []}
    assert reads and "data_contracts" not in reads


def test_wal_reader_pins_one_snapshot_across_a_concurrent_capture(tmp_path, monkeypatch):
    chain = build_chain(tmp_path, tables=[{
        "name": "price_history", "contract": "price_history.v1",
        "fragments": [{"row_count": 3}]}])
    baseline = inventory(chain)
    real_execute = reregister_snapshot._execute
    fired = []

    def hooked_execute(conn, statement, parameters=()):
        cursor = real_execute(conn, statement, parameters)
        if not fired and "FROM data_import_receipts" in statement:
            writer = open_catalog(chain["path"], clock=FakeClock())
            try:
                insert_price_capture(writer, capture_id="cap-during-read",
                                     receipt_id="receipt-1",
                                     contract_id="price_history.v1", rows_added=1)
            finally:
                writer.close()
            fired.append(True)
        return cursor

    monkeypatch.setattr(reregister_snapshot, "_execute", hooked_execute)
    payload = inventory(chain)
    assert fired
    assert payload == baseline
    assert payload["captures"]["price_history"] == []
    fresh = inventory(chain)
    assert [row["capture_id"] for row in fresh["captures"]["price_history"]] == [
        "cap-during-read"]


_SENTINEL = b"pre-existing export\n"


def _securities_fixture(tmp_path):
    """One real committed ``securities`` snapshot: conn, store, head pins, object."""
    conn, clock, store = catalog_and_store(tmp_path)
    securities = contract_for("securities")
    record = publish_and_inspect(store, securities, contract_ref_for(securities),
                                 [_SECURITIES_ROW], "2024")
    commit_tables(conn, clock, {"securities": [record]}, {"securities": securities},
                  scope="shadow", receipt_id="receipt-1", store=store)
    head = conn.execute("SELECT snapshot_id, generation FROM data_snapshot_heads"
                        " WHERE scope = 'shadow'").fetchone()
    digest = record.object_ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
    return {"conn": conn, "store": store, "record": record,
            "pins": {"scope": "shadow", "snapshot_id": head["snapshot_id"],
                     "receipt_id": "receipt-1", "generation": head["generation"]},
            "catalog_path": tmp_path / "catalog.sqlite",
            "object_path": store.root / "objects" / digest[:2] / digest}


def _cli_argv(fixture, out):
    pins = fixture["pins"]
    return ["export", "--scope", pins["scope"], "--snapshot-id", pins["snapshot_id"],
            "--receipt-id", pins["receipt_id"], "--generation", str(pins["generation"]),
            "--catalog", str(fixture["catalog_path"]), "--objects", str(fixture["store"].root),
            "--out", str(out)]


def _export(fixture, out):
    return reregister_snapshot.export_inventory(fixture["conn"], **fixture["pins"],
                                                objects_root=fixture["store"].root, out=out)


def _chain_export_refused(chain, tmp_path, code, *, sentinel=None):
    pins = {key: chain[key] for key in ("scope", "snapshot_id", "receipt_id", "generation")}
    out = tmp_path / "export.json"
    if sentinel is not None:
        out.write_bytes(sentinel)
    with pytest.raises(DataError) as excinfo:
        reregister_snapshot.export_inventory(chain["conn"], **pins,
                                             objects_root=tmp_path / "store", out=out)
    assert excinfo.value.code == code
    if sentinel is None:
        assert not out.exists()
    else:
        assert out.read_bytes() == sentinel
    assert not list(tmp_path.glob(".export.json.*.part"))


def test_cli_export_wrapper_matches_inventory_and_reruns_byte_identically(tmp_path):
    fixture = _securities_fixture(tmp_path)
    ref = publish_bytes(fixture["store"], b"synthetic reference bytes\n")
    insert_reference(fixture["conn"], receipt_id=fixture["pins"]["receipt_id"],
                     legacy_path="refs/model_registry.json", kind="model_registry",
                     object_id=ref.object_id, object_hash=ref.content_hash,
                     byte_size=ref.byte_size, fold="")
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    assert reregister_snapshot.main(_cli_argv(fixture, first)) == 0
    assert reregister_snapshot.main(_cli_argv(fixture, second)) == 0
    wrapper = json.loads(first.read_text())
    expected = reregister_snapshot.neutral_inventory(fixture["conn"], **fixture["pins"])
    assert wrapper["schema_version"] == "reregister_snapshot_export.v1"
    assert wrapper["inventory"] == expected
    assert wrapper["content_hash"] == content_hash(expected)
    table = wrapper["inventory"]["tables"][0]
    assert table["table_name"] == "securities" and table["dataset_version_id"]
    assert table["coverage_receipt_refs"] == [RECEIPT]
    assert [frag["fragment_id"] for frag in table["fragments"]] == [
        fixture["record"].fragment_id]
    assert table["fragments"][0]["primary_key_min"]
    assert table["fragments"][0]["primary_key_max"]
    assert wrapper["inventory"]["references"] == [
        {"kind": "model_registry", "legacy_path": "refs/model_registry.json",
         "object_id": ref.object_id, "content_hash": ref.content_hash,
         "byte_size": ref.byte_size, "fold": ""}]
    assert wrapper["inventory"]["lineage"]["receipt_ids"] == ["receipt-1"]
    assert second.read_bytes() == first.read_bytes()
    assert json.loads(second.read_text())["content_hash"] == wrapper["content_hash"]


@pytest.mark.parametrize("drift", [
    lambda data: data[:-1] + bytes([data[-1] ^ 0xFF]),
    lambda data: data + b"\x00",
], ids=["same-length-bytes", "changed-length"])
def test_object_drift_refuses_nonzero_and_leaves_output_untouched(tmp_path, drift):
    fixture = _securities_fixture(tmp_path)
    out = tmp_path / "export.json"
    out.write_bytes(_SENTINEL)
    fixture["object_path"].chmod(0o644)
    fixture["object_path"].write_bytes(drift(fixture["object_path"].read_bytes()))
    assert reregister_snapshot.main(_cli_argv(fixture, out)) == 2
    assert out.read_bytes() == _SENTINEL


def test_removed_object_is_a_typed_corrupt_refusal(tmp_path):
    fixture = _securities_fixture(tmp_path)
    out = tmp_path / "export.json"
    out.write_bytes(_SENTINEL)
    fixture["object_path"].unlink()
    with pytest.raises(DataError) as excinfo:
        _export(fixture, out)
    assert excinfo.value.code == "OBJECT_CORRUPT"
    assert out.read_bytes() == _SENTINEL


def test_incomplete_membership_refuses_before_any_write(tmp_path):
    chain = build_chain(tmp_path, tables=[{
        "name": "price_history", "contract": "price_history.v1",
        "fragments": [{"partition_key": "2026"}, {"partition_key": "2025"}]}])
    conn = chain["conn"]
    conn.execute("DROP TRIGGER data_version_fragments_no_delete")
    conn.execute("DELETE FROM data_version_fragments WHERE dataset_version_id = ?"
                 " AND ordinal = 0", ("price_history-dsv",))
    _chain_export_refused(chain, tmp_path, "INPUT_CHANGED", sentinel=_SENTINEL)


def test_absent_catalog_is_a_typed_cli_refusal_with_no_output(tmp_path, capsys):
    out = tmp_path / "export.json"
    catalog_path = tmp_path / "absent.sqlite"
    assert reregister_snapshot.main([
        "export", "--scope", "shadow", "--snapshot-id", "snap-1",
        "--receipt-id", "receipt-1", "--generation", "1",
        "--catalog", str(catalog_path),
        "--objects", str(tmp_path / "store"), "--out", str(out)]) == 2
    stderr = capsys.readouterr().err
    assert str(catalog_path) not in stderr
    assert json.loads(stderr)["refused"] == "INPUT_CHANGED"
    assert not out.exists()


def test_absent_pinned_receipt_is_refused_with_no_output(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    conn = chain["conn"]
    conn.execute("DROP TRIGGER data_import_receipts_no_delete")
    conn.execute("DELETE FROM data_import_receipts WHERE receipt_id = 'receipt-1'")
    _chain_export_refused(chain, tmp_path, "INPUT_CHANGED")


def test_absent_snapshot_row_is_refused_with_no_output(tmp_path):
    chain = build_chain(tmp_path, tables=[{"name": "daily_market"}])
    conn = chain["conn"]
    conn.execute("DROP TRIGGER data_snapshots_no_delete")
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("DELETE FROM data_snapshots WHERE snapshot_id = 'snap-1'")
    conn.execute("PRAGMA foreign_keys = ON")
    _chain_export_refused(chain, tmp_path, "INPUT_CHANGED")


def test_replace_failure_keeps_old_bytes_and_leaves_no_temp(tmp_path, monkeypatch):
    fixture = _securities_fixture(tmp_path)
    out = tmp_path / "export.json"
    out.write_bytes(_SENTINEL)

    def boom(src, dst):
        raise OSError("injected crash before the rename")

    monkeypatch.setattr(reregister_snapshot.os, "replace", boom)
    with pytest.raises(DataError) as excinfo:
        _export(fixture, out)
    assert excinfo.value.code == "INPUT_CHANGED"
    assert excinfo.value.problem.message == "the export file could not be written"
    assert out.read_bytes() == _SENTINEL
    assert list(tmp_path.glob(".export.json.*.part")) == []


def test_head_advance_after_fsync_is_a_source_drift_refusal(tmp_path, monkeypatch):
    fixture = _securities_fixture(tmp_path)
    conn = fixture["conn"]
    out = tmp_path / "export.json"
    out.write_bytes(_SENTINEL)
    real_fsync = reregister_snapshot.os.fsync
    fired = []

    def hooked_fsync(fd):
        if not fired:
            fired.append(True)
            conn.execute("DROP TRIGGER data_snapshot_heads_update_generation")
            conn.execute("UPDATE data_snapshot_heads SET generation = 2"
                         " WHERE scope = 'shadow'")
        return real_fsync(fd)

    monkeypatch.setattr(reregister_snapshot.os, "fsync", hooked_fsync)
    with pytest.raises(DataError) as excinfo:
        _export(fixture, out)
    assert excinfo.value.code == "INPUT_CHANGED"
    assert excinfo.value.problem.message == (
        "the mutable head does not match the explicit pins")
    assert fired
    assert out.read_bytes() == _SENTINEL
    assert list(tmp_path.glob(".export.json.*.part")) == []
