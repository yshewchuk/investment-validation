"""IV-crush contract remains typed, deterministic and honest about its clocks."""
from dataclasses import replace

import pyarrow as pa

from engine.v2.contracts import TableContract
from engine.v2.data.documents import decode_document
from engine.v2.data.iv_crush import build_rows
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT, IV_CRUSH_TABLE_NAME
from engine.v2.data.manifests import table_contract_hash
from engine.v2.data.query import arrow_type_for
from engine.v2.foundation import to_document


def test_contract_roundtrip_hash_and_identity():
    contract = IV_CRUSH_CONTRACT
    assert contract.table_name == IV_CRUSH_TABLE_NAME == "iv_crush_targets"
    assert contract.contract_id == "iv_crush_targets.v1"
    assert decode_document(TableContract, to_document(contract)) == contract
    assert table_contract_hash(contract) == contract.definition_hash
    assert table_contract_hash(replace(contract, maximum_batch_rows=1)) != contract.definition_hash
    names = {c.name for c in contract.columns}
    assert set((*contract.primary_key, *contract.partition_columns, *contract.orderable_columns)) <= names


def test_native_arrow_schema_accepts_signed_and_skipped_rows():
    contract = IV_CRUSH_CONTRACT
    rows = build_rows("TEST", [{"event_date": "2024-01-05", "session": "AMC"},
                               {"event_date": "2024-01-08", "session": None}],
                      [{"date": "2024-01-05", "iv30": 50}, {"date": "2024-01-08", "iv30": 30}],
                      source_id="synthetic")
    schema = pa.schema([pa.field(c.name, arrow_type_for(c.physical_type), nullable=c.nullable)
                        for c in contract.columns])
    assert pa.Table.from_pylist(rows, schema=schema).to_pylist() == rows


def test_clock_and_provenance_claims_remain_distinct():
    columns = {c.name: c for c in IV_CRUSH_CONTRACT.columns}
    assert "relative" in columns["crush_pct_iv30"].observation_time_semantics
    assert "one calendar day" in columns["available_as_of_date"].observation_time_semantics
    assert "not evidence" in columns["available_as_of_date"].observation_time_semantics
    assert IV_CRUSH_CONTRACT.publication_time_column is None
    assert IV_CRUSH_CONTRACT.receipt_time_column is None
    assert "unverified" in IV_CRUSH_CONTRACT.provenance_semantics
