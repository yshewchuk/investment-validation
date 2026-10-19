"""IV-crush contract remains typed, deterministic and honest about its clocks."""
import hashlib
import json
from dataclasses import replace

import pyarrow as pa

from engine.v2.contracts import TableContract
from engine.v2.data.documents import decode_document
from engine.v2.data.iv_crush import build_rows
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT, IV_CRUSH_TABLE_NAME
from engine.v2.data.manifests import table_contract_hash
from engine.v2.data.query import arrow_type_for
from engine.v2.foundation import to_document

EXPECTED_FIELDS = (
    ("ticker", "string", False), ("event_date", "string", False), ("session", "string", True),
    ("pre_date", "string", True), ("post_date", "string", True),
    ("pre_iv30", "float64", True), ("post_iv30", "float64", True),
    ("crush_pct_iv30", "float64", True), ("available_as_of_date", "string", True),
    ("skipped", "bool", False), ("skip_reason", "string", True),
    ("source_id", "string", False), ("input_hash", "string", False),
)


def test_contract_roundtrip_hash_and_identity():
    contract = IV_CRUSH_CONTRACT
    assert contract.table_name == IV_CRUSH_TABLE_NAME == "iv_crush_targets"
    assert contract.contract_id == "iv_crush_targets.v1"
    assert decode_document(TableContract, to_document(contract)) == contract
    assert table_contract_hash(contract) == contract.definition_hash
    assert table_contract_hash(replace(contract, maximum_batch_rows=1)) != contract.definition_hash
    names = {c.name for c in contract.columns}
    assert set((*contract.primary_key, *contract.partition_columns, *contract.orderable_columns)) <= names
    assert tuple((c.name, c.physical_type, c.nullable) for c in contract.columns) == EXPECTED_FIELDS
    assert contract.semantic_version == "1.0.0"
    assert contract.primary_key == contract.orderable_columns == ("ticker", "event_date")
    assert contract.partition_columns == contract.filterable_columns == ("ticker",)
    assert contract.observation_time_column == "event_date"
    assert contract.maximum_batch_rows == 10000
    assert contract.foreign_keys == () and contract.legacy_mapping_ref is None
    assert {c.name: c.unit for c in contract.columns} == {
        name: "percent" if name in ("pre_iv30", "post_iv30", "crush_pct_iv30") else None
        for name, _, _ in EXPECTED_FIELDS}


def test_native_arrow_schema_accepts_signed_and_skipped_rows():
    contract = IV_CRUSH_CONTRACT
    rows = build_rows("TEST", [{"event_date": "2024-01-05", "session": "AMC"},
                               {"event_date": "2024-01-08", "session": None}],
                      [{"date": "2024-01-05", "iv30": 50}, {"date": "2024-01-08", "iv30": 30}],
                      source_id="synthetic")
    expected_types = {"string": pa.string(), "float64": pa.float64(), "bool": pa.bool_()}
    schema = pa.schema([pa.field(name, expected_types[kind], nullable=nullable)
                        for name, kind, nullable in EXPECTED_FIELDS])
    actual_schema = pa.schema([pa.field(c.name, arrow_type_for(c.physical_type), nullable=c.nullable)
                               for c in contract.columns])
    assert actual_schema == schema
    assert len(rows) == 2
    # This ASCII/integer fixture has identical compact sorted JSON and canonical
    # bytes, so the expected digest needs no production hash/projection helper.
    expected_projection = {
        "schema_version": "iv_crush_inputs.v1",
        "ticker": "TEST",
        "events": [{"event_date": "2024-01-05", "session": "AMC"},
                   {"event_date": "2024-01-08", "session": None}],
        "daily": [{"date": "2024-01-05", "iv30": 50},
                  {"date": "2024-01-08", "iv30": 30}],
    }
    expected_hash = "sha256:" + hashlib.sha256(
        json.dumps(expected_projection, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    expected_rows = [
        {"ticker": "TEST", "event_date": "2024-01-05", "session": "AMC",
         "pre_date": "2024-01-05", "post_date": "2024-01-08",
         "pre_iv30": 50.0, "post_iv30": 30.0, "crush_pct_iv30": -40.0,
         "available_as_of_date": "2024-01-09", "skipped": False,
         "skip_reason": None, "source_id": "synthetic", "input_hash": expected_hash},
        {"ticker": "TEST", "event_date": "2024-01-08", "session": None,
         "pre_date": None, "post_date": None, "pre_iv30": None, "post_iv30": None,
         "crush_pct_iv30": None, "available_as_of_date": None, "skipped": True,
         "skip_reason": "missing_session", "source_id": "synthetic", "input_hash": expected_hash},
    ]
    assert rows == expected_rows
    assert rows[0]["skipped"] is False
    assert rows[1]["skipped"] is True
    persisted = pa.Table.from_pylist(rows, schema=schema).to_pylist()
    assert persisted == expected_rows
    assert persisted[0]["skipped"] is False
    assert persisted[1]["skipped"] is True
    assert persisted == rows


def test_clock_and_provenance_claims_remain_distinct():
    contract = IV_CRUSH_CONTRACT
    assert {c.name: (c.observation_time_semantics, c.null_policy) for c in contract.columns} == {
        "ticker": (None, None),
        "event_date": (None, None),
        "session": (None, None),
        "pre_date": ("selected quoted close date, YYYY-MM-DD; no publication claim", None),
        "post_date": ("selected quoted close date, YYYY-MM-DD; no publication claim", None),
        "pre_iv30": (
            None, "missing/nonfinite anchor; finite values retained even on skipped pairs"),
        "post_iv30": (
            None, "missing/nonfinite anchor; finite values retained even on skipped pairs"),
        "crush_pct_iv30": (
            "signed relative IV30 change: 100*(post_iv30/pre_iv30-1); "
            "adjacent daily rows span at most five calendar days; "
            "finite positive pre-IV and finite post-IV, no post-positive filter",
            "null exactly when skipped; never imputed or floored"),
        "available_as_of_date": (
            "post_date plus one calendar day; modeled day-clock floor, "
            "not evidence of source publication or finality",
            "null exactly when skipped; unavailable labels cannot enter training"),
        "skipped": (None, None),
        "skip_reason": (None, "null exactly when not skipped"),
        "source_id": (None, "caller-declared identity, not verified by the pure builder"),
        "input_hash": (None, "canonical hash of the supplied typed event/daily projections"),
    }
    assert contract.publication_time_column is None
    assert contract.receipt_time_column is None
    assert contract.duplicate_policy == "duplicate event dates refuse"
    assert contract.finality_semantics == (
        "no finality or actual publication evidence; reconstructed targets only")
    assert contract.provenance_semantics == (
        "input_hash binds supplied projections; source_id is unverified caller metadata")
    assert contract.coverage_semantics == (
        "one row per supplied event, skipped rows included; no completeness inference")
    assert contract.schema_evolution_policy == "new contract identity for any semantic change"
