"""D01: every data contract round-trips exactly; malformed documents are refused.

Phase-2 guide §12 (D01) and §5 (every type in §5.1-§5.6). ``engine/v2/contracts``
declares the shapes; ``engine/v2/data/documents.py`` is the only place doing the
extra checks (hash/timestamp/date format, table-contract and query structural
rules) that ``engine.v2.foundation.typed`` cannot express from annotations
alone. This file proves both halves together, the same way
``tests/test_v2_ops_contracts.py`` proves the Phase 1 contracts.
"""
from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2 import contracts  # noqa: E402
from engine.v2.contracts import (  # noqa: E402
    ChainMember,
    ChainQuery,
    ChainSnapshot,
    ColumnContract,
    ContractId,
    DataQuery,
    DatasetManifest,
    DatasetVersionRef,
    DependencyEntry,
    DependencyPlan,
    EarningsEvent,
    EventRef,
    FragmentRecord,
    FragmentRef,
    KeyPredicate,
    LegacyFileRef,
    LegacyMaterializationRequest,
    ObjectRef,
    SnapshotImportReceipt,
    SnapshotImportRequest,
    SnapshotRef,
    TableContract,
    TableContractRef,
    TimeInterval,
)
from engine.v2.data import decode_document, loads_document
from engine.v2.foundation import DocumentError, canonical_json, to_document

H = "sha256:" + "0" * 64
TIMESTAMP = "2026-01-15T20:00:00.000000Z"


# --------------------------------------------------------------------------
# one valid example instance per contract type
# --------------------------------------------------------------------------


def _build_samples() -> dict[str, object]:
    tcr = TableContractRef(contract_id="tc_securities", definition_hash=H)
    columns = (
        ColumnContract(name="ticker", physical_type="string", nullable=False),
        ColumnContract(name="year", physical_type="int64", nullable=False,
                       unit=None, allowed_range=(1990.0, 2100.0)),
    )
    tc = TableContract(
        contract_id="tc_securities", definition_hash=H, table_name="securities",
        semantic_version="1.0", columns=columns, primary_key=("ticker", "year"),
        duplicate_policy="reject", foreign_keys=(), partition_columns=("year",),
        filterable_columns=("ticker", "year"), orderable_columns=("ticker", "year"),
        finality_semantics="legacy_daily_close.v1", provenance_semantics="legacy_import.v1",
        coverage_semantics="legacy_full.v1", schema_evolution_policy="major_on_meaning_change.v1",
        maximum_batch_rows=10_000, maximum_result_rows=1_000_000)
    obj = ObjectRef(kind="parquet_fragment", object_id="obj_1", content_hash=H, byte_size=1024)
    frag_ref = FragmentRef(fragment_id="frag_1", manifest_hash=H)
    frag = FragmentRecord(
        fragment_id="frag_1", manifest_hash=H, object_ref=obj, table_contract_ref=tcr,
        partition_key="2026", row_count=100, byte_hash=H, logical_content_hash=H,
        primary_key_min=("AAPL", 2026), primary_key_max=("ZZZZ", 2026),
        input_receipt_refs=("receipt_1",), import_request_hash=H)
    dvr = DatasetVersionRef(dataset_version_id="dv_1", table_contract_ref=tcr, manifest_hash=H)
    dm = DatasetManifest(
        dataset_version_ref=dvr, logical_content_hash=H, row_count=100,
        fragment_refs=(frag_ref,), coverage_receipt_refs=("cov_1",),
        knowledge_mode="reconstructed", availability_evidence_refs=())
    sref = SnapshotRef(
        snapshot_id="snap_1", manifest_hash=H, table_versions={"securities": dvr},
        calendar_version="cal.v1", source_priority_version="prio.v1",
        finality_receipt_refs=("fin_1",), knowledge_mode_by_table={"securities": "reconstructed"})
    kp = KeyPredicate(column="ticker", operator="in", values=("AAPL", "MSFT"))
    ti = TimeInterval(column="date", start_inclusive="2026-01-01", end_exclusive="2026-06-01")
    dq = DataQuery(
        snapshot_id="snap_1", table_contract_ref=tcr, columns=("ticker", "year"),
        key_filter=(kp,), order_by=("ticker", "year"), max_batch_rows=1000,
        max_result_rows=5000)
    eref = EventRef(event_id="evt_1", calendar_revision="cal.v1")
    event = EarningsEvent(
        event_ref=eref, security_id="sec_1", ticker_at_event="AAPL",
        scheduled_event_date="2026-01-15", session="amc", session_source="legacy_calendar.v1",
        confidence=0.9, conflict_status="clear")
    cid = ContractId(
        contract_id="contract_1", security_id="sec_1",
        vendor_mappings={"orats": "AAPL260116C00150000"}, expiry="2026-01-16",
        right="call", exact_strike="150.00", multiplier="100",
        adjustment_identity="legacy_standard.v1")
    cq = ChainQuery(
        event_ref=eref, security_id="sec_1", observation_ceiling=TIMESTAMP,
        session_date="2026-01-15", quote_policy_ref="policy.v1", max_contracts=500)
    cm = ChainMember(
        contract_id=cid, quote_observation_id="q_1", bid="1.20", ask="1.30", mid="1.25",
        iv=0.35, delta=0.5, volume=10, open_interest=100, bid_size=5, ask_size=5,
        source="legacy_orats", source_row_ref="row_1", availability_status="available",
        quality_flags=())
    cs = ChainSnapshot(
        chain_id="chain_1", source_snapshot_ref="snap_1", security_id="sec_1",
        observed_at=TIMESTAMP, session_date="2026-01-15", quote_policy_ref="policy.v1",
        spot_unadjusted="150.00", spot_adjusted="150.00", rows=(cm,), expected_contracts=1,
        supported_contracts=1, returned_contracts=1, coverage_ref="cov_chain_1",
        knowledge_mode="reconstructed")
    de = DependencyEntry(
        table_name="securities", dataset_version_ref=dvr, fragment_ref=frag_ref,
        columns=("ticker", "year"), predicates=(kp,), estimated_rows=100, maximum_rows=1000)
    dp = DependencyPlan(request_hash=H, snapshot_ref=sref, dependencies=(de,))
    legacy_file = LegacyFileRef(path="data/curated/securities/2026.parquet",
                                content_hash=H, byte_size=2048)
    sir = SnapshotImportRequest(
        scope="initial_migration", source_manifest_ref="legacy_snapshot_2026",
        source_manifest_hash=H, table_sources={"securities": (legacy_file,)},
        table_contract_refs={"securities": tcr}, legacy_snapshot_source_ref=legacy_file,
        calendar_version="cal.v1", source_priority_version="prio.v1",
        finality_receipt_refs=("fin_1",), knowledge_mode_by_table={"securities": "reconstructed"},
        expected_head_generation=0)
    sirpt = SnapshotImportReceipt(
        receipt_id="receipt_import_1", request_hash=H, attempt_id="att_1", fence=1,
        snapshot_ref=sref, legacy_snapshot_object_ref=obj,
        resulting_head_snapshot_id="snap_1", resulting_head_generation=1,
        status="committed", envelope={"worker_id": "w1"})
    lmr = LegacyMaterializationRequest(
        request_hash=H, snapshot_ref=sref, legacy_snapshot_object_ref=obj,
        direct_scope={"tickers": ["AAPL"], "years": [2026]},
        evidence_scope={"tickers": ["AAPL", "MSFT"], "years": [2024, 2025, 2026]},
        table_queries={"data/curated/securities/2026.parquet": dq},
        registry_and_model_refs=("registry_1",), calendar_refs=("cal.v1",),
        legacy_layout_version="legacy_layout.v1", expected_population={"securities": 100})
    return {
        "ColumnContract": columns[0], "TableContract": tc, "TableContractRef": tcr,
        "ObjectRef": obj, "FragmentRef": frag_ref, "FragmentRecord": frag,
        "DatasetVersionRef": dvr, "DatasetManifest": dm, "SnapshotRef": sref,
        "KeyPredicate": kp, "TimeInterval": ti, "DataQuery": dq, "EventRef": eref,
        "EarningsEvent": event, "ContractId": cid, "ChainQuery": cq, "ChainMember": cm,
        "ChainSnapshot": cs, "DependencyEntry": de, "DependencyPlan": dp,
        "SnapshotImportRequest": sir, "SnapshotImportReceipt": sirpt,
        "LegacyMaterializationRequest": lmr,
    }


SAMPLES = _build_samples()


@pytest.mark.parametrize("name", sorted(SAMPLES), ids=lambda n: n)
def test_every_contract_round_trips_exactly(name):
    sample = SAMPLES[name]
    doc = to_document(sample)
    assert decode_document(type(sample), doc) == sample
    text = canonical_json(doc)
    assert loads_document(type(sample), text) == sample


# --------------------------------------------------------------------------
# every _hash/_at/deadline/observation_ceiling/known_from field is covered
# --------------------------------------------------------------------------


def _formats_declared_in_documents_py() -> set[tuple[str, str]]:
    tree = ast.parse((ROOT / "engine/v2/data/documents.py").read_text())
    for node in ast.walk(tree):
        is_formats_assign = (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_FORMATS" for t in node.targets)
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name) and node.target.id == "_FORMATS"
        )
        if is_formats_assign:
            return {(k.elts[0].value, k.elts[1].value) for k in node.value.keys}
    raise AssertionError("_FORMATS not found in engine/v2/data/documents.py")


def test_every_hash_and_time_field_is_in_the_format_table():
    declared = _formats_declared_in_documents_py()
    checked_names = ("_hash", "_at")
    checked_exact = {"deadline", "observation_ceiling", "known_from"}
    data_types = [obj for obj in vars(contracts).values()
                  if isinstance(obj, type) and dataclasses.is_dataclass(obj)
                  and obj.__module__ == "engine.v2.contracts.data"]
    assert data_types, "expected engine.v2.contracts.data types on engine.v2.contracts"
    for cls in data_types:
        for f in dataclasses.fields(cls):
            if f.name.endswith(checked_names) or f.name in checked_exact:
                assert (cls.__name__, f.name) in declared, (cls.__name__, f.name)


# --------------------------------------------------------------------------
# negative controls
# --------------------------------------------------------------------------


def test_unknown_field_is_refused():
    doc = to_document(SAMPLES["EventRef"])
    doc["unexpected"] = "x"
    with pytest.raises(DocumentError) as err:
        decode_document(EventRef, doc)
    assert err.value.code == "UNKNOWN_FIELD"


def test_unknown_knowledge_mode_is_refused():
    doc = to_document(SAMPLES["DatasetManifest"])
    doc["knowledge_mode"] = "guessed"
    with pytest.raises(DocumentError) as err:
        decode_document(DatasetManifest, doc)
    assert err.value.code == "BAD_ENUM"


def test_unknown_operator_is_refused():
    doc = to_document(SAMPLES["KeyPredicate"])
    doc["operator"] = "gt"
    with pytest.raises(DocumentError) as err:
        decode_document(KeyPredicate, doc)
    assert err.value.code == "BAD_ENUM"


def test_unknown_import_status_is_refused():
    doc = to_document(SAMPLES["SnapshotImportReceipt"])
    doc["status"] = "pending"
    with pytest.raises(DocumentError) as err:
        decode_document(SnapshotImportReceipt, doc)
    assert err.value.code == "BAD_ENUM"


def test_naive_timestamp_is_refused():
    doc = to_document(SAMPLES["DataQuery"])
    doc["deadline"] = "2026-01-15T20:00:00"
    with pytest.raises(DocumentError) as err:
        decode_document(DataQuery, doc)
    assert err.value.code == "BAD_TIMESTAMP_FORMAT"


@pytest.mark.parametrize("bad_hash", [
    "sha256:" + "A" * 64,  # uppercase
    "sha256:" + "0" * 63,  # too short
    "0" * 64,              # missing prefix
])
def test_malformed_hash_is_refused(bad_hash):
    doc = to_document(SAMPLES["TableContractRef"])
    doc["definition_hash"] = bad_hash
    with pytest.raises(DocumentError) as err:
        decode_document(TableContractRef, doc)
    assert err.value.code == "BAD_HASH_FORMAT"


@pytest.mark.parametrize("bad_date", ["2026/01/15", "2026-13-01"])
def test_bad_date_is_refused(bad_date):
    doc = to_document(SAMPLES["EarningsEvent"])
    doc["scheduled_event_date"] = bad_date
    with pytest.raises(DocumentError) as err:
        decode_document(EarningsEvent, doc)
    assert err.value.code == "BAD_DATE_FORMAT"


def test_duplicate_column_name_is_refused():
    doc = to_document(SAMPLES["TableContract"])
    doc["columns"][1]["name"] = doc["columns"][0]["name"]
    with pytest.raises(DocumentError) as err:
        decode_document(TableContract, doc)
    assert err.value.code == "DUPLICATE_COLUMN_NAME"


def test_primary_key_naming_an_undeclared_column_is_refused():
    doc = to_document(SAMPLES["TableContract"])
    doc["primary_key"] = ("ticker", "not_a_column")
    with pytest.raises(DocumentError) as err:
        decode_document(TableContract, doc)
    assert err.value.code == "UNDECLARED_COLUMN"


def test_duplicate_json_key_is_refused():
    text = '{"event_id": "e1", "calendar_revision": "c1", "event_id": "e2"}'
    with pytest.raises(DocumentError) as err:
        loads_document(EventRef, text)
    assert err.value.code == "DUPLICATE_KEY"


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nan_and_infinity_in_a_float_field_are_refused(bad_value):
    doc = to_document(SAMPLES["ChainMember"])
    doc["iv"] = bad_value
    with pytest.raises(DocumentError) as err:
        decode_document(ChainMember, doc)
    assert err.value.code == "BAD_TYPE"


def test_wrong_major_version_is_refused():
    doc = to_document(SAMPLES["EventRef"])
    doc["schema_version"] = "event_ref.v2.0"
    with pytest.raises(DocumentError) as err:
        decode_document(EventRef, doc)
    assert err.value.code == "UNSUPPORTED_VERSION"


def test_newer_minor_version_is_refused():
    doc = to_document(SAMPLES["EventRef"])
    doc["schema_version"] = "event_ref.v1.1"
    with pytest.raises(DocumentError) as err:
        decode_document(EventRef, doc)
    assert err.value.code == "UNSUPPORTED_VERSION"


def test_in_values_unsorted_is_refused():
    doc = to_document(SAMPLES["KeyPredicate"])
    doc["values"] = ["MSFT", "AAPL"]
    with pytest.raises(DocumentError) as err:
        decode_document(KeyPredicate, doc)
    assert err.value.code == "UNSORTED_IN_VALUES"


def test_in_values_duplicate_is_refused():
    doc = to_document(SAMPLES["KeyPredicate"])
    doc["values"] = ["AAPL", "AAPL"]
    with pytest.raises(DocumentError) as err:
        decode_document(KeyPredicate, doc)
    assert err.value.code == "DUPLICATE_IN_VALUE"


def test_in_values_empty_is_refused():
    doc = to_document(SAMPLES["KeyPredicate"])
    doc["values"] = []
    with pytest.raises(DocumentError) as err:
        decode_document(KeyPredicate, doc)
    assert err.value.code == "EMPTY_IN_VALUES"


def test_in_values_mixed_types_is_refused():
    doc = to_document(SAMPLES["KeyPredicate"])
    doc["values"] = ["AAPL", 5]
    with pytest.raises(DocumentError) as err:
        decode_document(KeyPredicate, doc)
    assert err.value.code == "MIXED_IN_VALUE_TYPES"


def test_eq_with_two_values_is_refused():
    doc = to_document(SAMPLES["KeyPredicate"])
    doc["operator"] = "eq"
    doc["values"] = ["AAPL", "MSFT"]
    with pytest.raises(DocumentError) as err:
        decode_document(KeyPredicate, doc)
    assert err.value.code == "EQ_REQUIRES_ONE_VALUE"


@pytest.mark.parametrize("field,value", [("max_batch_rows", 0), ("max_result_rows", -5)])
def test_nonpositive_limits_are_refused(field, value):
    doc = to_document(SAMPLES["DataQuery"])
    doc[field] = value
    with pytest.raises(DocumentError) as err:
        decode_document(DataQuery, doc)
    assert err.value.code == "INVALID_LIMIT"


def test_batch_exceeding_result_is_refused():
    doc = to_document(SAMPLES["DataQuery"])
    doc["max_batch_rows"] = 100
    doc["max_result_rows"] = 50
    with pytest.raises(DocumentError) as err:
        decode_document(DataQuery, doc)
    assert err.value.code == "BATCH_EXCEEDS_RESULT"


def test_query_with_neither_predicate_nor_time_bound_is_refused():
    doc = to_document(SAMPLES["DataQuery"])
    doc["key_filter"] = []
    doc["time_interval"] = None
    with pytest.raises(DocumentError) as err:
        decode_document(DataQuery, doc)
    assert err.value.code == "QUERY_NOT_BOUNDED"
