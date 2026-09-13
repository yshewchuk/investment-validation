"""Independent review controls for the copy-only ledger migration."""
from __future__ import annotations

import json
import base64

import pytest

from engine.v2.ledger.decisions import DecisionConflict, import_lines, rows, set_authority
from engine.v2.ledger.export import export_generation
from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore
from engine.v2.ops.catalog import transaction
from engine.v2.ops.decision_commit import import_settlement_candidates_in_transaction
from engine.v2.ops.lifecycle import request_cancel
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit
from tests.ops_support import catalog, sample

STAMP = "2026-09-13T12:00:00.000000Z"


def _open(tmp_path):
    conn, _, _ = catalog(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    return conn


def _outcome(status, resolved_at):
    return {"row_id": "prediction-1", "event_date": "2026-09-09",
            "ticker": "FAKE", "strategy": "TWIN-P", "status": status,
            "resolved_at": resolved_at,
            "realized_pnl": 0.125 if status == "resolved" else None}


def _raw(row):
    return (json.dumps(row, sort_keys=True) + "\n").encode()


def test_review_outcome_history_roundtrips_resolution_dates_and_reimports(tmp_path):
    conn = _open(tmp_path)
    first = _outcome("unresolvable", "2026-09-10T23:00:00+00:00")
    last = _outcome("resolved", "2026-09-12T00:30:00+00:00")
    original = [_raw(first), _raw(last)]
    try:
        with transaction(conn):
            imported = import_lines(conn, "original", original, kind="outcome", created_at=STAMP)
        assert len({item["decision_id"] for item in imported}) == 2
        with transaction(conn):
            repeated = import_lines(conn, "original", original, kind="outcome", created_at=STAMP)
            copied = import_lines(conn, "copied", original, kind="outcome", created_at=STAMP)
        assert [item["decision_id"] for item in imported] == [item["decision_id"] for item in repeated]
        assert [item["decision_id"] for item in imported] == [item["decision_id"] for item in copied]
        assert [json.loads(item["payload_json"]) for item in rows(conn, kind="outcome")] == [first, last]
        stored_bytes = [bytes(item[0]) for item in conn.execute(
            "SELECT original_bytes FROM decision_imports WHERE source_hash=? ORDER BY line_number",
            ("original",))]
        assert stored_bytes == original
        destination = export_generation(conn, tmp_path / "compat", generation="review")
        paths = sorted((destination / "outcomes").glob("*.jsonl"))
        assert [path.name for path in paths] == ["2026-09-10.jsonl", "2026-09-12.jsonl"]
        assert [json.loads(path.read_text()) for path in paths] == [first, last]
        assert export_generation(conn, tmp_path / "compat", generation="review") == destination
    finally:
        conn.close()


def test_review_changed_outcome_at_same_observation_rolls_back_entire_import(tmp_path):
    conn = _open(tmp_path)
    existing = _outcome("resolved", "2026-09-12T00:30:00+00:00")
    fresh = dict(existing, row_id="prediction-2")
    changed = dict(existing, status="unresolvable", realized_pnl=None)
    try:
        with transaction(conn):
            import_lines(conn, "original", [_raw(existing)], kind="outcome", created_at=STAMP)
        before = rows(conn)
        with pytest.raises(DecisionConflict):
            with transaction(conn):
                import_lines(conn, "conflict", [_raw(fresh), _raw(changed)],
                             kind="outcome", created_at=STAMP)
        assert rows(conn) == before
        assert conn.execute("SELECT COUNT(*) FROM decision_imports WHERE source_hash=?",
                            ("conflict",)).fetchone()[0] == 0
    finally:
        conn.close()


def test_review_prediction_duplicate_policy_is_unchanged(tmp_path):
    conn = _open(tmp_path)
    prediction = {"row_id": "prediction-1", "as_of": "2026-09-09", "score": {"win": 0.6}}
    try:
        with transaction(conn):
            import_lines(conn, "first", [_raw(prediction)], kind="prediction", created_at=STAMP)
        with transaction(conn):
            import_lines(conn, "copy", [_raw(prediction)], kind="prediction", created_at=STAMP)
        assert len(rows(conn, kind="prediction")) == 1
        with pytest.raises(DecisionConflict):
            with transaction(conn):
                import_lines(conn, "changed", [_raw(dict(prediction, score={"win": 0.7}))],
                             kind="prediction", created_at=STAMP)
        assert len(rows(conn, kind="prediction")) == 1
    finally:
        conn.close()


def _settlement_claim(conn, clock, supervisor):
    job = JobSpec(kind="legacy_settlement", implementation_ref="code", spec_hash=None,
                  environment_ref="env", parameters={
                      "expected_ids": ("legacy_settlement",), "session": "2026-09-13"},
                  output_namespace="shadow", resource_class="legacy_rebuild",
                  retry_policy_ref="bounded", checkpoint_contract_ref="legacy_action.v1.0")
    policy = NamespacePolicy({"operator": frozenset({"shadow"})})
    submit(conn, registry(), policy, SubmitRequest(namespace="shadow", idempotency_key="settle",
                                                   principal="operator", job=job), clock=clock)
    return claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor,
                      clock=clock, registry=registry())


def test_review_settlement_coordinator_retries_once_and_stale_fence_writes_nothing(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
        from engine.v2.ledger.decisions import insert
        insert(conn, logical_key="prediction-1", decision_id="prediction:prediction-1",
               payload={"row_id": "prediction-1", "ticker": "FAKE", "strategy": "TWIN-P",
                        "event_date": "2026-09-09", "settlement": {"policy": "fixed"}},
               purpose="shadow", kind="prediction", validations={}, created_at=STAMP)
    claim = _settlement_claim(conn, clock, supervisor)
    unresolved = _outcome("unresolvable", "2026-09-10T23:00:00+00:00") | {
        "settlement": {"policy": "fixed"}}
    resolved = _outcome("resolved", "2026-09-12T00:30:00+00:00") | {
        "settlement": {"policy": "fixed"}, "settlement_source": "orats_quote_simulation",
        "exit_source": "chain", "exit_finality": {"is_final": True}}
    captured = [{"row": row, "original_b64": base64.b64encode(_raw(row)).decode("ascii")}
                for row in (unresolved, resolved)]
    ref = store.publish_bytes(json.dumps({"rows": captured}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    with transaction(conn):
        first = import_settlement_candidates_in_transaction(
            conn, claim, ref, captured, clock=clock)
    with transaction(conn):
        repeated = import_settlement_candidates_in_transaction(
            conn, claim, ref, captured, clock=clock)
    assert [item["decision_id"] for item in first] == [item["decision_id"] for item in repeated]
    assert len(rows(conn, kind="outcome")) == 2
    assert conn.execute("SELECT COUNT(*) FROM outbox WHERE kind='export'").fetchone()[0] == 1
    assert tuple(conn.execute("SELECT stage,occurrence FROM watermarks").fetchone()) == (
        "settlement", "2026-09-13")

    request_cancel(conn, claim.job_id, claim.attempt_id, clock=clock)
    changed = dict(resolved, resolved_at="2026-09-13T00:30:00+00:00")
    changed_capture = [{"row": changed,
                        "original_b64": base64.b64encode(_raw(changed)).decode("ascii")}]
    changed_ref = store.publish_bytes(json.dumps({"rows": changed_capture}).encode(),
                                      schema_ref="legacy_action.v1.0")
    with pytest.raises(Exception, match="CANCELLED|LEASE_LOST"):
        with transaction(conn):
            import_settlement_candidates_in_transaction(
                conn, claim, changed_ref, changed_capture, clock=clock)
    assert len(rows(conn, kind="outcome")) == 2


def test_review_settlement_worker_captures_only_new_exact_bytes(tmp_path, monkeypatch):
    from engine import ledger as legacy_ledger
    from engine.v2.ops.legacy_adapter import _action_settlement

    directory = tmp_path / "legacy" / "ledger" / "outcomes"
    directory.mkdir(parents=True)
    old = _raw(_outcome("unresolvable", "2026-09-10T00:00:00+00:00"))
    new = _raw(_outcome("resolved", "2026-09-12T00:00:00+00:00"))
    path = directory / "2026-09-12.jsonl"
    path.write_bytes(old)

    def fake_score_outcomes(*, through):
        assert through == "2026-09-13"
        with path.open("ab") as stream:
            stream.write(new)
        return {"resolved": 1, "path": str(path)}

    monkeypatch.setattr(legacy_ledger, "score_outcomes", fake_score_outcomes)
    result = _action_settlement({"session": "2026-09-13"}, tmp_path)
    document = json.loads((tmp_path / result["path"]).read_text())
    assert document["result"]["resolved"] == 1
    assert len(document["rows"]) == 1
    assert base64.b64decode(document["rows"][0]["original_b64"]) == new
    assert document["rows"][0]["row"] == json.loads(new)
