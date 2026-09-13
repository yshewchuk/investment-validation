"""Negative controls for the Phase 1 ledger/effect authority boundary."""
from __future__ import annotations

import json

import pytest

from engine.v2.ledger.decisions import DecisionConflict, import_lines, insert, set_authority
from engine.v2.ledger.export import export_generation
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError
from engine.v2.ops.outbox import claim, complete, enqueue, fail_effect
from tests.ops_support import catalog


def test_import_preserves_bytes_and_rejects_conflicting_duplicate(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    raw = b'{"row_id":"r1","event_id":"e1"}\n'
    with transaction(conn):
        set_authority(conn, None, "catalog", "2026-09-12T00:00:00Z")
        import_lines(conn, "sha256:source", [raw], kind="prediction", created_at="2026-09-12T00:00:00Z")
    assert conn.execute("SELECT original_bytes FROM decision_imports").fetchone()[0] == raw
    with pytest.raises(DecisionConflict, match="conflicting legacy duplicate"):
        with transaction(conn):
            import_lines(conn, "sha256:other", [b'{"row_id":"r1","event_id":"changed"}\n'],
                         kind="prediction", created_at="2026-09-12T00:00:00Z")


def test_export_is_complete_and_outbox_retry_is_durable(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    payload = {"row_id": "r1", "event_id": "e1", "as_of": "2026-09-12"}
    with transaction(conn):
        set_authority(conn, None, "catalog", "2026-09-12T00:00:00Z")
        insert(conn, logical_key="k", decision_id="prediction:r1", payload=payload,
               purpose="shadow", kind="prediction", validations={}, created_at="2026-09-12T00:00:00Z")
        enqueue(conn, "export", "generation-1", {"generation": "generation-1"})
    exported = export_generation(conn, tmp_path / "compat", generation="generation-1")
    assert json.loads((exported / "predictions" / "2026-09-12.jsonl").read_text()) == payload
    effect = claim(conn, "export", owner="worker-1", clock=clock, logical_key="generation-1",
                   lease_seconds=10)
    assert effect["attempts"] == 1
    fail_effect(conn, effect["effect_id"], {"code": "DELIVERY_FAILED"}, owner="worker-1",
                claim_token=effect["claim_token"], clock=clock)
    retry = claim(conn, "export", owner="worker-2", clock=clock, logical_key="generation-1")
    assert retry["effect_id"] == effect["effect_id"]
    assert retry["attempts"] == 2


def test_abandoned_claim_is_recovered_and_stale_worker_is_fenced(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    with transaction(conn):
        enqueue(conn, "publication", "release-1", {"release": "r1"})
    other = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    try:
        first = claim(conn, "publication", owner="old", logical_key="release-1",
                      clock=clock, lease_seconds=5)
        assert claim(other, "publication", owner="new", logical_key="release-1", clock=clock) is None
        clock.advance(6)
        second = claim(other, "publication", owner="new", logical_key="release-1", clock=clock)
        assert second["claim_token"] != first["claim_token"]
        with pytest.raises(OpsError, match="STALE_EXPECTATION"):
            complete(conn, first["effect_id"], {"old": True}, owner="old",
                     claim_token=first["claim_token"], clock=clock)
        with pytest.raises(OpsError, match="STALE_EXPECTATION"):
            fail_effect(conn, second["effect_id"], {"bad": True}, owner="old",
                        claim_token=first["claim_token"], clock=clock)
    finally:
        other.close()
