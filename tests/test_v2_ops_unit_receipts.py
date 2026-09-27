"""S4C: the shared provider-receipt cache and failure classification.

Extracted from tests/test_v2_ops_calendar_moves_jobs.py (P6 slice-4c split,
Part 0) alongside the module it tests.
"""
from __future__ import annotations

import types

from engine.v2.foundation import ArtifactStore
from engine.v2.ops.incremental_data import RefreshUnit
from engine.v2.ops.unit_receipts import (
    cached_unit_outcomes, cached_unit_payloads, provider_failure_code, record_unit_receipt,
)
from tests.ops_support import catalog


def test_provider_failure_code_orders_mixed_kinds():
    assert provider_failure_code(("complete", "legitimate_empty")) is None
    assert provider_failure_code(("complete", "transient")) == "TRANSIENT_SOURCE"
    assert provider_failure_code(("transient", "not_final")) == "SOURCE_NOT_FINAL"
    # An unparseable body (refused) is bad source data, never a credential
    # problem; only the provider's own 401/403 kind is CREDENTIAL_INVALID.
    assert provider_failure_code(("transient", "refused")) == "SOURCE_INVALID"
    assert provider_failure_code(("refused", "not_final")) == "SOURCE_INVALID"
    assert provider_failure_code(("transient", "credential_invalid")) == "CREDENTIAL_INVALID"


def test_record_unit_receipt_stores_expected_fields(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    unit = RefreshUnit(
        request_id="req-1",
        table_name="daily_market",
        partition_key="AAPL",
        expected_keys=("2026-09-01", "2026-09-02"),
    )

    record = record_unit_receipt(
        conn,
        store,
        unit,
        b"unit-payload-bytes",
        source="fixture",
        endpoint="daily_market",
        received_at=clock.now().isoformat(),
    )

    assert record.source == "fixture"
    assert record.endpoint == "daily_market"
    assert record.response_kind == "complete"
    assert record.request == {
        "request_id": "req-1",
        "table_name": "daily_market",
        "partition_key": "AAPL",
        "keys": ["2026-09-01", "2026-09-02"],
    }

    row = conn.execute(
        "SELECT source, endpoint, response_kind FROM data_raw_receipts "
        "WHERE raw_receipt_id = ?",
        (record.raw_receipt_id,),
    ).fetchone()
    assert row is not None
    assert row["source"] == "fixture"
    assert row["endpoint"] == "daily_market"
    assert row["response_kind"] == "complete"
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 1


def test_cached_unit_payloads_restores_stored_bytes(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    unit = RefreshUnit(
        request_id="req-1",
        table_name="daily_market",
        partition_key="AAPL",
        expected_keys=("2026-09-01", "2026-09-02"),
    )
    original_bytes = b"unit-payload-bytes-for-cache-hit"

    record_unit_receipt(
        conn,
        store,
        unit,
        original_bytes,
        source="fixture",
        endpoint="daily_market",
        received_at=clock.now().isoformat(),
    )

    outcomes = cached_unit_outcomes(conn, [unit], source="fixture", endpoint="daily_market")
    assert unit.request_id in outcomes

    plan = types.SimpleNamespace(cached=tuple(outcomes.values()))
    payloads = cached_unit_payloads(conn, store, plan)

    assert payloads[unit.request_id] == original_bytes
