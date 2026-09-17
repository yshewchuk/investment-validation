from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from engine.v2.contracts import (
    CoverageKey,
    CoverageOutcome,
    RevisionCandidate,
    TimeInterval,
)
from engine.v2.data import incremental as data_incremental
from engine.v2.data.catalog import commit_snapshot
from engine.v2.data.manifests import dataset_manifest, snapshot_ref
from engine.v2.data.objects import inspect_fragment
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, content_hash, to_document
from engine.v2.ops.incremental_data import RefreshParameters
from tests.ops_support import catalog
from tests.test_v2_data_manifests import _DAILY_MARKET_CONTRACT, _DAILY_MARKET_REF


def test_raw_receipt_is_cache_first_and_idempotent(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    payload = data_incremental.RawPayload(
        payload=b"{\"ticker\":\"AAA\"}",
        response_kind="complete",
        response_meta={"status": 200},
    )
    first = data_incremental.cache_raw_receipt(
        conn,
        store,
        payload,
        source="fixture",
        endpoint="daily_market",
        request={"ticker": "AAA", "session": "2026-09-15"},
        received_at=clock.now().isoformat(),
    )
    second = data_incremental.cache_raw_receipt(
        conn,
        store,
        payload,
        source="fixture",
        endpoint="daily_market",
        request={"ticker": "AAA", "session": "2026-09-15"},
        received_at=clock.now().isoformat(),
    )
    assert first.raw_receipt_id == second.raw_receipt_id
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM data_objects").fetchone()[0] == 0


def test_supervised_refresh_without_staged_input_is_truthful_failure(tmp_path):
    result = data_incremental.run_incremental_refresh(
        RefreshParameters(
            expected_ids=("request-1",),
            parent_snapshot_id="nonexistent-snapshot",
            refresh_plan_hash="sha256:" + "a" * 64,
            provider_calls=0,
        ),
        tmp_path,
    )
    assert result["status"] == "failed"
    assert result["completed_ids"] == []
    assert result["coverage_advanced"] is False
    assert result["candidate_snapshot_id"] is None
    document = json.loads((tmp_path / "incremental_refresh_result.json").read_text())
    assert document["status"] == "failed"
    assert document["completed_ids"] == []
    assert document["coverage_advanced"] is False


def test_supervised_refresh_with_explicit_empty_input_remains_noop(tmp_path):
    (tmp_path / "incremental_refresh_input.json").write_text("{}")
    result = data_incremental.run_incremental_refresh(
        RefreshParameters(
            expected_ids=("request-1",),
            parent_snapshot_id="snapshot-1",
            refresh_plan_hash="sha256:" + "a" * 64,
            provider_calls=0,
        ),
        tmp_path,
    )
    assert result["status"] == "noop"
    assert result["completed_ids"] == ["request-1"]
    assert result["coverage_advanced"] is False


def _frozen_revision(row, raw_id, revision_id, *, deleted=False, spot=None):
    session = str(row["date"])[:10]
    body = None if deleted else dict(row, spot=row["spot"] + 0.01 if spot is None else spot)
    content = data_incremental.revision_content_hash(
        ticker=row["ticker"], session_date=session, row=body, deleted=deleted)
    candidate = RevisionCandidate(
        revision_id=revision_id,
        logical_key=data_incremental.daily_market_logical_key(row["ticker"], session),
        source="frozen-curated", source_priority=0, finality="final",
        revision_ordinal=1, received_at="2026-09-17T12:00:00Z",
        content_hash=content)
    return data_incremental.DailyMarketRevision(
        candidate=candidate, ticker=row["ticker"], session_date=session,
        row=body, deleted=deleted, raw_receipt_id=raw_id,
        normalization_id="pending")


def _coverage(revision, raw_id):
    key = CoverageKey(
        item_key=revision.candidate.logical_key, session_date=revision.session_date,
        ticker=revision.ticker)
    outcome = CoverageOutcome(
        key=key, status="present", receipt_id=raw_id,
        revision_id=revision.candidate.revision_id, finality="final")
    start = revision.session_date
    interval = TimeInterval(column="date", start_inclusive=start,
                            end_exclusive=f"{start}T23:59:59")
    return data_incremental.build_completed_coverage(
        _DAILY_MARKET_REF, source="frozen-curated", endpoint="parquet",
        interval=interval, expected=(key,), outcomes=(outcome,),
        acquisition_receipt_refs=(raw_id,), completed_at="2026-09-17T12:00:00Z")


def _refresh_input(tmp_path, parent_id, generation, plan_hash, revisions, raws,
                   coverage, *, fault_point=None):
    root = tmp_path / ("attempt-" + str(generation))
    root.mkdir(exist_ok=True)
    document = {
        "catalog_path": str(tmp_path / "ops.sqlite"),
        "objects_root": str(tmp_path / "objects"),
        "scope": "shadow",
        "expected_head_snapshot_id": parent_id,
        "expected_head_generation": generation,
        "receipt_id": "refresh-" + str(generation),
        "attempt_id": "attempt-" + str(generation),
        "fence": generation + 1,
        "coverage": to_document(coverage),
        "raw_payloads": raws,
        "incoming_revisions": [
            data_incremental._revision_document(item) for item in revisions],
    }
    if fault_point:
        document["fault_point"] = fault_point
    (root / "incremental_refresh_input.json").write_text(json.dumps(document))
    params = RefreshParameters(
        expected_ids=(f"daily-market-{generation}",),
        parent_snapshot_id=parent_id, refresh_plan_hash=plan_hash,
        provider_calls=0)
    return root, params


def test_frozen_curated_append_correction_tombstone_and_noop_replay(tmp_path):
    base_path = Path("data/curated/daily_market/year=2026/part-0018.parquet")
    append_path = Path("data/curated/daily_market/year=2026/part-0017.parquet")
    base_bytes = base_path.read_bytes()
    base_rows = pq.read_table(base_path).slice(0, 2).to_pylist()
    append_row = pq.read_table(append_path).slice(0, 1).to_pylist()[0]
    clock_conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(base_bytes, schema_ref="parquet_fragment.v1")
    object_ref = data_incremental.ObjectRef(
        kind="parquet_fragment", object_id=ref.artifact_id,
        content_hash=ref.content_hash, byte_size=ref.byte_size)
    inspection = inspect_fragment(
        store, object_ref, _DAILY_MARKET_CONTRACT, _DAILY_MARKET_REF, "2026")
    receipt_ref = content_hash({"frozen": str(base_path)})
    record = data_incremental.manifests.fragment_record(
        inspection, _DAILY_MARKET_REF, input_receipt_refs=(receipt_ref,),
        import_request_hash=receipt_ref)
    manifest = dataset_manifest(
        _DAILY_MARKET_REF, (record,), knowledge_mode="reconstructed",
        coverage_receipt_refs=(receipt_ref,), availability_evidence_refs=())
    parent = snapshot_ref(
        {"daily_market": manifest}, calendar_version="cal.v1",
        source_priority_version="frozen-curated",
        finality_receipt_refs=(receipt_ref,))
    commit_snapshot(
        clock_conn, scope="shadow", request_hash=content_hash({"r": 1}),
        contracts=(_DAILY_MARKET_CONTRACT,), objects=(object_ref,),
        records=(record,), manifests=(manifest,), snapshot=parent,
        expected_head_snapshot_id=None, expected_head_generation=0,
        receipt_id="base-receipt", attempt_id="base-attempt", fence=1,
        fence_check=lambda _conn: None, clock=clock, store=store)

    append_payload = {"kind": "append", "source": str(append_path)}
    append_raw = data_incremental.cache_raw_receipt(
        clock_conn, store, data_incremental.RawPayload(
            payload=json.dumps(append_payload).encode(), response_kind="complete",
            response_meta={"frozen_path": str(append_path)}),
        source="frozen-curated", endpoint="daily_market",
        request={"path": str(append_path), "keys": [append_row["ticker"]]},
        received_at=clock.now().isoformat())
    append_revision = _frozen_revision(
        append_row, append_raw.raw_receipt_id, "revision-append", spot=append_row["spot"])
    append_coverage = _coverage(append_revision, append_raw.raw_receipt_id)
    raw_doc = {
        "payload": json.dumps(append_payload).encode().decode(),
        "response_kind": "complete", "response_meta": {"frozen_path": str(append_path)},
        "source": "frozen-curated", "endpoint": "daily_market",
        "request": {"path": str(append_path), "keys": [append_row["ticker"]]},
        "received_at": clock.now().isoformat(),
    }
    plan_hash = "sha256:" + "1" * 64
    failed_root, failed_params = _refresh_input(
        tmp_path, parent.snapshot_id, 1, plan_hash, (append_revision,),
        (raw_doc,), append_coverage, fault_point="before_commit")
    try:
        data_incremental.run_incremental_refresh(failed_params, failed_root)
    except RuntimeError as exc:
        assert "before_commit" in str(exc)
    assert Repository(clock_conn).resolve(parent.snapshot_id) == parent

    retry_root, retry_params = _refresh_input(
        tmp_path, parent.snapshot_id, 1, plan_hash, (append_revision,),
        (raw_doc,), append_coverage)
    r2 = data_incremental.run_incremental_refresh(retry_params, retry_root)
    assert r2["status"] == "complete"
    snapshot_r2 = Repository(clock_conn).resolve(r2["candidate_snapshot_id"])
    assert snapshot_r2.snapshot_id != parent.snapshot_id

    correction_payload = {"kind": "correction", "source": str(base_path)}
    correction_raw = data_incremental.cache_raw_receipt(
        clock_conn, store, data_incremental.RawPayload(
            payload=json.dumps(correction_payload).encode(), response_kind="complete",
            response_meta={"frozen_path": str(base_path)}),
        source="frozen-curated", endpoint="daily_market",
        request={"path": str(base_path), "keys": [base_rows[0]["ticker"]]},
        received_at=clock.now().isoformat())
    correction_row = dict(base_rows[0])
    correction = _frozen_revision(
        correction_row, correction_raw.raw_receipt_id, "revision-correction",
        spot=correction_row["spot"] + 0.02)
    tombstone = _frozen_revision(
        base_rows[1], correction_raw.raw_receipt_id, "revision-tombstone", deleted=True)
    r3_coverage = _coverage(correction, correction_raw.raw_receipt_id)
    correction_doc = {
        "payload": json.dumps(correction_payload).encode().decode(),
        "response_kind": "complete", "response_meta": {"frozen_path": str(base_path)},
        "source": "frozen-curated", "endpoint": "daily_market",
        "request": {"path": str(base_path), "keys": [base_rows[0]["ticker"]]},
        "received_at": clock.now().isoformat(),
    }
    r3_root, r3_params = _refresh_input(
        tmp_path, snapshot_r2.snapshot_id, 2, "sha256:" + "2" * 64,
        (correction, tombstone), (correction_doc,), r3_coverage)
    r3 = data_incremental.run_incremental_refresh(r3_params, r3_root)
    assert r3["status"] == "complete"
    snapshot_r3 = Repository(clock_conn).resolve(r3["candidate_snapshot_id"])

    noop_root, noop_params = _refresh_input(
        tmp_path, snapshot_r3.snapshot_id, 3, "sha256:" + "3" * 64,
        (correction, tombstone), (correction_doc,), r3_coverage)
    noop = data_incremental.run_incremental_refresh(noop_params, noop_root)
    assert noop["candidate_snapshot_id"] == snapshot_r3.snapshot_id
    conn_count = clock_conn.execute(
        "SELECT COUNT(*) FROM data_changesets").fetchone()[0]
    assert conn_count == 3
