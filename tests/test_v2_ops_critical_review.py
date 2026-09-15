"""Independent review controls for the copy-only ledger migration."""
from __future__ import annotations

import json
import base64

import pytest

from engine.ledger import SCHEMA_VERSION as _CURRENT_SCHEMA_VERSION
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
                        "event_date": "2026-09-09", "settlement": {"policy": "fixed"},
                        # current-schema row: this test exercises idempotent
                        # commit/retry, not the grandfathered finality-proof
                        # rule, so it keeps legacy's own exit_finality path.
                        "schema_version": _CURRENT_SCHEMA_VERSION},
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
    (tmp_path / "finality.json").write_text(json.dumps({
        "date": "2026-09-13", "is_final": True, "market_wide": True,
        "daily_share": 1.0, "chain_share": 1.0, "covered": 1, "detail": "final"}))

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
    assert document["session"] == "2026-09-13"
    assert document["requested_session"] == "2026-09-13"


# --------------------------------------------------------------------------
# 2026-09-15: legacy ledger duplicates (DLNG-shape) diverge instead of
# refusing the whole settlement stage -- guide §5.5 item 1 applied to
# settlement contract mismatches. See engine/v2/ops/decision_commit.py
# _settlement_line / _contract_mismatch_field / _record_settlement_divergence.
# --------------------------------------------------------------------------


def _seed_prediction(conn, row_id, event_date, *, ticker="DLNG", strategy="STR-THRU"):
    from engine.v2.ledger.decisions import insert
    with transaction(conn):
        insert(conn, logical_key=row_id, decision_id="prediction:" + row_id,
              payload={"row_id": row_id, "ticker": ticker, "strategy": strategy,
                       "event_date": event_date, "settlement": {"policy": "fixed"}},
              purpose="shadow", kind="prediction", validations={}, created_at=STAMP)


def _settlement_candidate(row_id, event_date, *, ticker="DLNG", strategy="STR-THRU",
                          status="unresolvable", resolved_at="2026-09-10T23:00:00+00:00"):
    row = {"row_id": row_id, "ticker": ticker, "strategy": strategy, "event_date": event_date,
           "settlement": {"policy": "fixed"}, "status": status, "resolved_at": resolved_at}
    return {"row": row, "original_b64": base64.b64encode(_raw(row)).decode("ascii")}


def test_settlement_event_date_mismatch_diverges_and_stage_succeeds(tmp_path):
    """The DLNG shape from the real shadow nightly (attempt 10): the recorded
    prediction has the event's ORIGINAL date (first-imported, first-wins),
    and the legacy ledger's settlement line names the date it moved to after
    an AMC->BMO shift. The mismatch must record a divergence, not refuse the
    stage."""
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "2026-09-04|DLNG|STR-THRU|2.5000|2026-09-18"
    _seed_prediction(conn, row_id, "2026-09-07")
    claim = _settlement_claim(conn, clock, supervisor)
    captured = [_settlement_candidate(row_id, "2026-09-08")]
    ref = store.publish_bytes(json.dumps({"rows": captured}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    diverged = []
    with transaction(conn):
        receipts = import_settlement_candidates_in_transaction(
            conn, claim, ref, captured, clock=clock, on_divergence=diverged.append)
    assert receipts == []
    assert len(rows(conn, kind="outcome")) == 0
    # The coordinator's job-output summary (engine/v2/ops/supervisor.py's
    # legacy_settlement effect) reports settlement_divergences: <count> plus
    # row_ids straight from this hook.
    assert diverged == [row_id]
    divergences = [dict(row) for row in conn.execute(
        "SELECT * FROM decision_divergences WHERE scope='legacy_settlement'")]
    assert len(divergences) == 1
    assert divergences[0]["decision_id"] == "prediction:" + row_id
    assert divergences[0]["occurrence"] == row_id
    assert "event_date" in divergences[0]["reason"]


def test_settlement_mixed_batch_commits_matches_and_diverges_the_mismatch(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    for row_id in ("row-a", "row-b", "row-c"):
        _seed_prediction(conn, row_id, "2026-09-07", ticker="FAKE", strategy="TWIN-P")
    claim = _settlement_claim(conn, clock, supervisor)
    captured = [
        _settlement_candidate("row-a", "2026-09-08", ticker="FAKE", strategy="TWIN-P"),  # mismatch
        _settlement_candidate("row-b", "2026-09-07", ticker="FAKE", strategy="TWIN-P"),  # matches
        _settlement_candidate("row-c", "2026-09-07", ticker="FAKE", strategy="TWIN-P"),  # matches
    ]
    ref = store.publish_bytes(json.dumps({"rows": captured}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    with transaction(conn):
        receipts = import_settlement_candidates_in_transaction(conn, claim, ref, captured, clock=clock)
    assert len(receipts) == 2
    assert len(rows(conn, kind="outcome")) == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM decision_divergences WHERE scope='legacy_settlement'").fetchone()[0] == 1


def test_settlement_divergence_retry_is_idempotent(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "2026-09-04|DLNG|STR-THRU|2.5000|2026-09-18"
    _seed_prediction(conn, row_id, "2026-09-07")
    claim = _settlement_claim(conn, clock, supervisor)
    captured = [_settlement_candidate(row_id, "2026-09-08")]
    ref = store.publish_bytes(json.dumps({"rows": captured}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    with transaction(conn):
        import_settlement_candidates_in_transaction(conn, claim, ref, captured, clock=clock)
    with transaction(conn):
        repeated = import_settlement_candidates_in_transaction(conn, claim, ref, captured, clock=clock)
    assert repeated == []
    assert conn.execute("SELECT COUNT(*) FROM decision_divergences").fetchone()[0] == 1
    assert len(rows(conn, kind="outcome")) == 0


def test_settlement_missing_prediction_still_refuses(tmp_path):
    """Not a duplicate -- a missing contract. Must still hard-refuse."""
    from engine.v2.ops.errors import OpsError

    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    claim = _settlement_claim(conn, clock, supervisor)
    captured = [_settlement_candidate("ghost-1", "2026-09-07")]
    ref = store.publish_bytes(json.dumps({"rows": captured}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    with pytest.raises(OpsError, match="settlement names no committed prediction"):
        with transaction(conn):
            import_settlement_candidates_in_transaction(conn, claim, ref, captured, clock=clock)
    assert conn.execute("SELECT COUNT(*) FROM decision_divergences").fetchone()[0] == 0


# --------------------------------------------------------------------------
# 2026-09-15: a grandfathered resolved settlement (recorded prediction
# schema_version < SCHEMA_VERSION, exactly engine.ledger.score_outcomes's
# own rule) settles on v2's OWN finality proof -- the recorded exit date at
# or before the settlement's finality-resolved session -- instead of
# legacy's exit_finality, which score_outcomes never computes for these
# rows (real shadow nightly attempt 12: all 207 resolved rows had
# exit_finality: None with schema_version 2 < SCHEMA_VERSION 3). A
# current-schema row keeps the original, unweakened check. See
# engine/v2/ops/decision_commit.py _validate_settlement_state.
# --------------------------------------------------------------------------

# _settlement_claim's job parameters pin session="2026-09-13" (no explicit
# session= is passed to import_settlement_candidates_in_transaction below,
# so that is the effective finality-resolved session throughout).


def _seed_prediction_ex(conn, row_id, event_date, *, ticker="FAKE", strategy="TWIN-P",
                        schema_version=None, exit_date=None):
    from engine.v2.ledger.decisions import insert
    payload = {"row_id": row_id, "ticker": ticker, "strategy": strategy,
              "event_date": event_date, "settlement": {"policy": "fixed"}}
    if schema_version is not None:
        payload["schema_version"] = schema_version
    if exit_date is not None:
        payload["structure"] = {"exit_date": exit_date}
    with transaction(conn):
        insert(conn, logical_key=row_id, decision_id="prediction:" + row_id, payload=payload,
              purpose="shadow", kind="prediction", validations={}, created_at=STAMP)


def _resolved_candidate(row_id, event_date, *, ticker="FAKE", strategy="TWIN-P",
                        resolved_at="2026-09-12T00:30:00+00:00", exit_finality=None):
    row = {"row_id": row_id, "ticker": ticker, "strategy": strategy, "event_date": event_date,
          "settlement": {"policy": "fixed"}, "status": "resolved", "resolved_at": resolved_at,
          "settlement_source": "orats_quote_simulation", "exit_source": "chain"}
    if exit_finality is not None:
        row["exit_finality"] = exit_finality
    return {"row": row, "original_b64": base64.b64encode(_raw(row)).decode("ascii")}


def test_settlement_grandfathered_resolved_row_admits_via_v2_finality_session(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "prediction-gf-1"
    _seed_prediction_ex(conn, row_id, "2026-09-09", schema_version=2, exit_date="2026-09-09")
    claim = _settlement_claim(conn, clock, supervisor)
    captured = [_resolved_candidate(row_id, "2026-09-09")]
    ref = store.publish_bytes(json.dumps({"rows": captured}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    admitted = []
    with transaction(conn):
        receipts = import_settlement_candidates_in_transaction(
            conn, claim, ref, captured, clock=clock, on_admitted=lambda row_id, proof: admitted.append((row_id, proof)))
    assert len(receipts) == 1
    assert admitted == [(row_id, "v2_finality_session")]
    assert len(rows(conn, kind="outcome")) == 1


def test_settlement_grandfathered_resolved_row_exit_after_session_refuses(tmp_path):
    from engine.v2.ops.errors import OpsError

    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "prediction-gf-2"
    # exit date is AFTER the settlement's finality-resolved session (2026-09-13).
    _seed_prediction_ex(conn, row_id, "2026-09-09", schema_version=2, exit_date="2026-09-20")
    claim = _settlement_claim(conn, clock, supervisor)
    captured = [_resolved_candidate(row_id, "2026-09-09")]
    ref = store.publish_bytes(json.dumps({"rows": captured}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    with pytest.raises(OpsError, match="resolved settlement lacks recorded exit evidence"):
        with transaction(conn):
            import_settlement_candidates_in_transaction(conn, claim, ref, captured, clock=clock)
    assert len(rows(conn, kind="outcome")) == 0


def test_settlement_grandfathered_resolved_row_missing_exit_date_refuses(tmp_path):
    from engine.v2.ops.errors import OpsError

    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "prediction-gf-3"
    _seed_prediction_ex(conn, row_id, "2026-09-09", schema_version=2)  # no exit_date at all
    claim = _settlement_claim(conn, clock, supervisor)
    captured = [_resolved_candidate(row_id, "2026-09-09")]
    ref = store.publish_bytes(json.dumps({"rows": captured}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    with pytest.raises(OpsError, match="resolved settlement lacks recorded exit evidence"):
        with transaction(conn):
            import_settlement_candidates_in_transaction(conn, claim, ref, captured, clock=clock)
    assert len(rows(conn, kind="outcome")) == 0


def test_settlement_current_schema_resolved_row_without_exit_finality_still_refuses(tmp_path):
    from engine.v2.ops.errors import OpsError

    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "prediction-cur-1"
    _seed_prediction_ex(conn, row_id, "2026-09-09", schema_version=_CURRENT_SCHEMA_VERSION,
                        exit_date="2026-09-09")
    claim = _settlement_claim(conn, clock, supervisor)
    captured = [_resolved_candidate(row_id, "2026-09-09")]  # no exit_finality
    ref = store.publish_bytes(json.dumps({"rows": captured}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    with pytest.raises(OpsError, match="resolved settlement lacks recorded exit evidence"):
        with transaction(conn):
            import_settlement_candidates_in_transaction(conn, claim, ref, captured, clock=clock)
    assert len(rows(conn, kind="outcome")) == 0


def test_settlement_current_schema_resolved_row_with_is_final_true_commits(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "prediction-cur-2"
    _seed_prediction_ex(conn, row_id, "2026-09-09", schema_version=_CURRENT_SCHEMA_VERSION,
                        exit_date="2026-09-09")
    claim = _settlement_claim(conn, clock, supervisor)
    captured = [_resolved_candidate(row_id, "2026-09-09", exit_finality={"is_final": True})]
    ref = store.publish_bytes(json.dumps({"rows": captured}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    admitted = []
    with transaction(conn):
        receipts = import_settlement_candidates_in_transaction(
            conn, claim, ref, captured, clock=clock, on_admitted=lambda row_id, proof: admitted.append((row_id, proof)))
    assert len(receipts) == 1
    assert admitted == [(row_id, "legacy_exit_finality")]


def test_settlement_unresolvable_rows_are_unaffected(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "prediction-unres-1"
    _seed_prediction_ex(conn, row_id, "2026-09-09")  # no schema_version, no exit_date at all
    claim = _settlement_claim(conn, clock, supervisor)
    captured = [_settlement_candidate(row_id, "2026-09-09", ticker="FAKE", strategy="TWIN-P")]
    ref = store.publish_bytes(json.dumps({"rows": captured}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    admitted = []
    with transaction(conn):
        receipts = import_settlement_candidates_in_transaction(
            conn, claim, ref, captured, clock=clock, on_admitted=lambda row_id, proof: admitted.append((row_id, proof)))
    assert len(receipts) == 1
    assert admitted == [(row_id, "unresolvable")]


def test_settlement_grandfathered_admission_retry_is_idempotent(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "prediction-gf-retry"
    _seed_prediction_ex(conn, row_id, "2026-09-09", schema_version=2, exit_date="2026-09-09")
    claim = _settlement_claim(conn, clock, supervisor)
    captured = [_resolved_candidate(row_id, "2026-09-09")]
    ref = store.publish_bytes(json.dumps({"rows": captured}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    with transaction(conn):
        first = import_settlement_candidates_in_transaction(conn, claim, ref, captured, clock=clock)
    with transaction(conn):
        repeated = import_settlement_candidates_in_transaction(conn, claim, ref, captured, clock=clock)
    assert [item["decision_id"] for item in first] == [item["decision_id"] for item in repeated]
    assert len(rows(conn, kind="outcome")) == 1


# --------------------------------------------------------------------------
# 2026-09-15: same-session settlement rerun dedupe (task brief rules 1/2/4).
# A rerun of ``score_outcomes`` stamps a NEW wall-clock ``resolved_at`` on
# the SAME underlying determination -- new candidate bytes, a new
# ``decisions.decision_id`` (``decisions._import_decision_id``), and by
# itself a duplicate commit. See ``decision_commit._settlement_dedupe_skip``.
# --------------------------------------------------------------------------


def test_settlement_same_session_rerun_with_new_wall_clock_commits_nothing(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "prediction-rerun-1"
    _seed_prediction_ex(conn, row_id, "2026-09-09", schema_version=2, exit_date="2026-09-09")
    claim = _settlement_claim(conn, clock, supervisor)  # session "2026-09-13"
    first_candidate = [_resolved_candidate(row_id, "2026-09-09", resolved_at="2026-09-13T01:00:00+00:00")]
    ref = store.publish_bytes(json.dumps({"rows": first_candidate}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    with transaction(conn):
        import_settlement_candidates_in_transaction(conn, claim, ref, first_candidate, clock=clock)
    assert len(rows(conn, kind="outcome")) == 1

    second_candidate = [_resolved_candidate(row_id, "2026-09-09", resolved_at="2026-09-13T09:00:00+00:00")]
    second_ref = store.publish_bytes(json.dumps({"rows": second_candidate}, sort_keys=True).encode(),
                                     schema_ref="legacy_action.v1.0")
    skipped = []
    with transaction(conn):
        repeated = import_settlement_candidates_in_transaction(
            conn, claim, second_ref, second_candidate, clock=clock, on_skip=lambda row_id, reason: skipped.append((row_id, reason)))
    assert repeated == []
    assert skipped == [(row_id, "already_observed_this_session")]
    assert len(rows(conn, kind="outcome")) == 1
    assert conn.execute("SELECT COUNT(*) FROM decision_divergences").fetchone()[0] == 0


def test_settlement_resolved_prediction_settled_again_on_later_session_commits_nothing(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "prediction-terminal-1"
    _seed_prediction_ex(conn, row_id, "2026-09-09", schema_version=2, exit_date="2026-09-09")
    claim = _settlement_claim(conn, clock, supervisor)  # session "2026-09-13"
    first_candidate = [_resolved_candidate(row_id, "2026-09-09", resolved_at="2026-09-13T01:00:00+00:00")]
    ref = store.publish_bytes(json.dumps({"rows": first_candidate}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    with transaction(conn):
        import_settlement_candidates_in_transaction(conn, claim, ref, first_candidate, clock=clock)
    assert len(rows(conn, kind="outcome")) == 1

    later_candidate = [_resolved_candidate(row_id, "2026-09-09", resolved_at="2026-09-20T01:00:00+00:00")]
    later_ref = store.publish_bytes(json.dumps({"rows": later_candidate}, sort_keys=True).encode(),
                                    schema_ref="legacy_action.v1.0")
    skipped = []
    with transaction(conn):
        repeated = import_settlement_candidates_in_transaction(
            conn, claim, later_ref, later_candidate, clock=clock, session="2026-09-20",
            on_skip=lambda row_id, reason: skipped.append((row_id, reason)))
    assert repeated == []
    assert skipped == [(row_id, "already_resolved")]
    assert len(rows(conn, kind="outcome")) == 1


def test_settlement_unresolvable_reobserved_on_later_session_commits_new_observation(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "prediction-retry-later-1"
    _seed_prediction_ex(conn, row_id, "2026-09-09", ticker="FAKE", strategy="TWIN-P")
    claim = _settlement_claim(conn, clock, supervisor)  # session "2026-09-13"
    first_candidate = [_settlement_candidate(row_id, "2026-09-09", ticker="FAKE", strategy="TWIN-P",
                                             status="unresolvable", resolved_at="2026-09-13T01:00:00+00:00")]
    ref = store.publish_bytes(json.dumps({"rows": first_candidate}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    with transaction(conn):
        import_settlement_candidates_in_transaction(conn, claim, ref, first_candidate, clock=clock)
    assert len(rows(conn, kind="outcome")) == 1

    later_candidate = [_settlement_candidate(row_id, "2026-09-09", ticker="FAKE", strategy="TWIN-P",
                                             status="unresolvable", resolved_at="2026-09-20T01:00:00+00:00")]
    later_ref = store.publish_bytes(json.dumps({"rows": later_candidate}, sort_keys=True).encode(),
                                    schema_ref="legacy_action.v1.0")
    with transaction(conn):
        repeated = import_settlement_candidates_in_transaction(
            conn, claim, later_ref, later_candidate, clock=clock, session="2026-09-20")
    assert len(repeated) == 1
    assert len(rows(conn, kind="outcome")) == 2


def test_settlement_same_session_status_change_records_one_divergence(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "prediction-status-change-1"
    _seed_prediction_ex(conn, row_id, "2026-09-09", ticker="FAKE", strategy="TWIN-P")
    claim = _settlement_claim(conn, clock, supervisor)  # session "2026-09-13"
    first_candidate = [_settlement_candidate(row_id, "2026-09-09", ticker="FAKE", strategy="TWIN-P",
                                             status="unresolvable", resolved_at="2026-09-13T01:00:00+00:00")]
    ref = store.publish_bytes(json.dumps({"rows": first_candidate}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    with transaction(conn):
        import_settlement_candidates_in_transaction(conn, claim, ref, first_candidate, clock=clock)
    assert len(rows(conn, kind="outcome")) == 1

    flipped = [_resolved_candidate(row_id, "2026-09-09", ticker="FAKE", strategy="TWIN-P",
                                   resolved_at="2026-09-13T09:00:00+00:00",
                                   exit_finality={"is_final": True})]
    flipped_ref = store.publish_bytes(json.dumps({"rows": flipped}, sort_keys=True).encode(),
                                      schema_ref="legacy_action.v1.0")
    diverged = []
    with transaction(conn):
        repeated = import_settlement_candidates_in_transaction(
            conn, claim, flipped_ref, flipped, clock=clock, on_divergence=diverged.append)
    assert repeated == []
    assert diverged == [row_id]
    assert len(rows(conn, kind="outcome")) == 1  # first committed observation stays
    divergences = [dict(row) for row in conn.execute(
        "SELECT * FROM decision_divergences WHERE scope='legacy_settlement' AND occurrence=?",
        (row_id,))]
    assert len(divergences) == 1
    assert "status" in divergences[0]["reason"]

    # A rerun of the same flip must not add a second divergence row.
    with transaction(conn):
        again = import_settlement_candidates_in_transaction(conn, claim, flipped_ref, flipped, clock=clock)
    assert again == []
    assert conn.execute(
        "SELECT COUNT(*) FROM decision_divergences WHERE occurrence=?", (row_id,)).fetchone()[0] == 1


# --------------------------------------------------------------------------
# 2026-09-15: engine.v2.ops.session_backfill -- the durable-session backfill
# that replaces _match_same_session's removed content-signature fallback,
# and the same-session/later-session behaviour that fallback made unsafe.
# --------------------------------------------------------------------------


def _register_nightly_settlement_artifact(conn, store, clock, claim, document):
    """Simulate what a real nightly ``legacy_settlement`` attempt leaves
    behind for ``session_backfill._nightly_session_map`` to find: an
    ``attempt_outputs`` row named ``legacy_settlement`` whose artifact IS the
    settlement candidate document (supervisor.py's own ``_settlement_effect``
    reads ``document.get("session")`` off exactly this artifact). Returns the
    artifact's ``content_hash`` -- the value to pass as ``import_lines``'
    ``source_hash`` so a directly-``import_lines``-inserted row (standing in
    for a pre-fix nightly commit, which never stamped ``generation_ref``)
    carries the same ``decision_imports.source_hash`` a real commit would.
    """
    from engine.v2.ops.checkpoints import register_artifact

    ref = store.publish_bytes(json.dumps(document, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    with transaction(conn):
        register_artifact(conn, ref, claim.attempt_id, clock)
        conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                     (claim.attempt_id, "legacy_settlement", ref.artifact_id))
    return ref.content_hash


def test_unresolvable_reobservation_on_later_session_commits_despite_matching_backfilled_content(
        tmp_path):
    """The defect this whole fix targets: a re-observation whose content is
    byte-identical (minus the wall-clock fields) to an OLDER, now-backfilled
    row must still commit on a later session -- it must never be skipped as
    ``already_observed_this_session`` just because the content matches."""
    from engine.v2.ops.session_backfill import backfill_outcome_sessions

    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "prediction-later-1"
    _seed_prediction_ex(conn, row_id, "2026-09-09", ticker="FAKE", strategy="TWIN-P")

    # A legacy-import-shaped outcome row (no generation_ref), as
    # ``ops ledger import-history`` would have left it, observed unresolvable
    # on 2026-09-01.
    old_row = {"row_id": row_id, "ticker": "FAKE", "strategy": "TWIN-P", "event_date": "2026-09-09",
              "settlement": {"policy": "fixed"}, "status": "unresolvable",
              "resolved_at": "2026-09-01T23:00:00+00:00"}
    with transaction(conn):
        import_lines(conn, "legacy_file_2026-09-01", [_raw(old_row)], kind="outcome", created_at=STAMP)
    assert len(rows(conn, kind="outcome")) == 1

    summary = backfill_outcome_sessions(conn, store, clock=clock)
    assert summary["import_history_resolved_at"] == 1
    [backfilled] = rows(conn, kind="outcome")
    assert backfilled["generation_ref"] == "2026-09-01"

    # A rerun on a LATER session (2026-09-13, the claim's default) re-observes
    # the same unresolvable determination -- identical content, new wall
    # clock. It must commit as a new observation, not be skipped.
    claim = _settlement_claim(conn, clock, supervisor)
    new_row = dict(old_row, resolved_at="2026-09-13T09:00:00+00:00")
    candidate = [{"row": new_row, "original_b64": base64.b64encode(_raw(new_row)).decode("ascii")}]
    ref = store.publish_bytes(json.dumps({"rows": candidate}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    skipped = []
    with transaction(conn):
        receipts = import_settlement_candidates_in_transaction(
            conn, claim, ref, candidate, clock=clock, on_skip=lambda r, reason: skipped.append((r, reason)))
    assert len(receipts) == 1
    assert skipped == []
    assert len(rows(conn, kind="outcome")) == 2


def test_same_session_rerun_against_backfilled_nightly_rows_commits_nothing(tmp_path):
    """A same-session rerun against a row the migration backfilled from a
    nightly ``legacy_settlement`` attempt's own provenance (attempt-13's real
    shape: generation_ref NULL before the fix) must dedupe exactly as a
    freshly-stamped row would -- 0 committed, 0 new divergences."""
    from engine.v2.ops.session_backfill import backfill_outcome_sessions

    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    row_id = "prediction-attempt13-1"
    _seed_prediction_ex(conn, row_id, "2026-09-09", ticker="FAKE", strategy="TWIN-P")
    claim = _settlement_claim(conn, clock, supervisor)  # job requested session "2026-09-13"

    old_row = {"row_id": row_id, "ticker": "FAKE", "strategy": "TWIN-P", "event_date": "2026-09-09",
              "settlement": {"policy": "fixed"}, "status": "unresolvable",
              "resolved_at": "2026-09-10T01:00:00+00:00"}
    # The settlement's own finality-resolved session (2026-09-10) differs
    # from the job's requested session (2026-09-13) -- a walk-back night,
    # exactly the shape the docstring's P2-C03 note describes.
    document = {"session": "2026-09-10", "requested_session": "2026-09-13", "rows": [old_row]}
    source_hash = _register_nightly_settlement_artifact(conn, store, clock, claim, document)
    with transaction(conn):
        import_lines(conn, source_hash, [_raw(old_row)], kind="outcome", created_at=STAMP)
    assert len(rows(conn, kind="outcome")) == 1
    assert rows(conn, kind="outcome")[0]["generation_ref"] is None

    summary = backfill_outcome_sessions(conn, store, clock=clock)
    assert summary["nightly_settlement"] == 1
    assert rows(conn, kind="outcome")[0]["generation_ref"] == "2026-09-10"

    # Rerun for the SAME settlement session (2026-09-10), new wall clock.
    new_row = dict(old_row, resolved_at="2026-09-10T09:00:00+00:00")
    candidate = [{"row": new_row, "original_b64": base64.b64encode(_raw(new_row)).decode("ascii")}]
    ref = store.publish_bytes(json.dumps({"rows": candidate}, sort_keys=True).encode(),
                              schema_ref="legacy_action.v1.0")
    skipped = []
    with transaction(conn):
        repeated = import_settlement_candidates_in_transaction(
            conn, claim, ref, candidate, clock=clock, session="2026-09-10",
            on_skip=lambda r, reason: skipped.append((r, reason)))
    assert repeated == []
    assert skipped == [(row_id, "already_observed_this_session")]
    assert len(rows(conn, kind="outcome")) == 1
    assert conn.execute("SELECT COUNT(*) FROM decision_divergences").fetchone()[0] == 0

    # A repeat backfill call is a no-op (idempotency marker).
    again = backfill_outcome_sessions(conn, store, clock=clock)
    assert again["already_applied"] is True


def test_backfill_derives_both_sources_and_leaves_underivable_rows_null(tmp_path):
    """One row from each of the two real sources, plus one the catalog
    cannot derive a session for at all -- the three counts must partition
    exactly, and the undetermined row must stay NULL."""
    from engine.v2.ops.session_backfill import backfill_outcome_sessions

    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
    claim = _settlement_claim(conn, clock, supervisor)

    _seed_prediction_ex(conn, "prediction-nightly-1", "2026-09-09", ticker="FAKE", strategy="TWIN-P")
    nightly_row = {"row_id": "prediction-nightly-1", "ticker": "FAKE", "strategy": "TWIN-P",
                   "event_date": "2026-09-09", "settlement": {"policy": "fixed"},
                   "status": "unresolvable", "resolved_at": "2026-09-10T01:00:00+00:00"}
    document = {"session": "2026-09-10", "rows": [nightly_row]}
    source_hash = _register_nightly_settlement_artifact(conn, store, clock, claim, document)
    with transaction(conn):
        import_lines(conn, source_hash, [_raw(nightly_row)], kind="outcome", created_at=STAMP)

    _seed_prediction_ex(conn, "prediction-import-1", "2026-09-05", ticker="FAKE", strategy="TWIN-P")
    import_row = {"row_id": "prediction-import-1", "ticker": "FAKE", "strategy": "TWIN-P",
                 "event_date": "2026-09-05", "settlement": {"policy": "fixed"},
                 "status": "resolved", "realized_pnl": 0.1,
                 "resolved_at": "2026-09-06T23:30:00+00:00"}
    with transaction(conn):
        import_lines(conn, "legacy_file_2026-09-06", [_raw(import_row)], kind="outcome", created_at=STAMP)

    _seed_prediction_ex(conn, "prediction-undetermined-1", "2026-09-01", ticker="FAKE",
                        strategy="TWIN-P")
    undetermined_row = {"row_id": "prediction-undetermined-1", "ticker": "FAKE", "strategy": "TWIN-P",
                        "event_date": "2026-09-01", "settlement": {"policy": "fixed"},
                        "status": "unresolvable", "resolved_at": "not-a-real-timestamp"}
    with transaction(conn):
        import_lines(conn, "legacy_file_corrupt", [_raw(undetermined_row)], kind="outcome",
                     created_at=STAMP)

    assert len(rows(conn, kind="outcome")) == 3
    summary = backfill_outcome_sessions(conn, store, clock=clock)
    assert summary == {"already_applied": False, "nightly_settlement": 1,
                       "import_history_resolved_at": 1, "undetermined": 1}

    by_row = {}
    for row in rows(conn, kind="outcome"):
        by_row[json.loads(row["payload_json"])["row_id"]] = row["generation_ref"]
    assert by_row["prediction-nightly-1"] == "2026-09-10"
    assert by_row["prediction-import-1"] == "2026-09-06"
    assert by_row["prediction-undetermined-1"] is None
