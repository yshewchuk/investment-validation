from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

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
from engine.v2.data.errors import DataError
from tests.test_v2_data_objects import _contract as _obj_contract
from tests.test_v2_data_objects import _daily_market_rows as _obj_daily_market_rows
from tests.test_v2_data_objects import _table_from_rows, _to_bytes


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
            provider_calls=0, catalog_path=str(tmp_path / "ops.sqlite"),
            objects_root=str(tmp_path / "objects"), scope="shadow",
            expected_head_generation=0,
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


def test_supervised_refresh_with_explicit_empty_input_fails_closed(tmp_path):
    (tmp_path / "incremental_refresh_input.json").write_text("{}")
    result = data_incremental.run_incremental_refresh(
        RefreshParameters(
            expected_ids=("request-1",),
            parent_snapshot_id="snapshot-1",
            refresh_plan_hash="sha256:" + "a" * 64,
            provider_calls=0, catalog_path=str(tmp_path / "ops.sqlite"),
            objects_root=str(tmp_path / "objects"), scope="shadow",
            expected_head_generation=0,
        ),
        tmp_path,
    )
    assert result["status"] == "failed"
    assert result["completed_ids"] == []
    assert result["coverage_advanced"] is False
    assert result["candidate_snapshot_id"] is None


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
        provider_calls=0, catalog_path=str(tmp_path / "ops.sqlite"),
        objects_root=str(tmp_path / "objects"), scope="shadow",
        expected_head_generation=generation,
        expected_head_snapshot_id=parent_id)
    return root, params


@pytest.mark.needs_data  # reads the real data/ root (gitignored, absent in CI and worktrees)
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
    raw_doc = {"receipt_id": append_raw.raw_receipt_id}
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
    correction_doc = {"receipt_id": correction_raw.raw_receipt_id}
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


def test_supplied_fence_check_composes_with_head_fence_not_replaces_it(tmp_path):
    """#81: a caller-supplied ``fence_check`` must be COMPOSED with
    ``commit_daily_market_candidate``'s own head fence, never replace it. A
    genuinely DIFFERENT candidate (a different writer racing against the same
    stale parent) must still be refused with ``SNAPSHOT_CONFLICT`` even when
    the caller's own ``fence_check`` is a no-op that would, by itself, allow
    the call (mirrors an attempt-lease check on an attempt that is still live)
    -- this is the exact scenario the bug let through silently. (#98's fix
    only exempts a same-effect retry, whose resulting snapshot does match what
    is actually at head; this different candidate's resulting snapshot does
    not, so the conflict stands.)"""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    contract = _obj_contract("daily_market")
    rows = _obj_daily_market_rows()
    table = _table_from_rows(contract, rows)
    ref = store.publish_bytes(_to_bytes(table), schema_ref="parquet_fragment.v1")
    object_ref = data_incremental.ObjectRef(
        kind="parquet_fragment", object_id=ref.artifact_id,
        content_hash=ref.content_hash, byte_size=ref.byte_size)
    inspection = inspect_fragment(store, object_ref, contract, _DAILY_MARKET_REF, "2024")
    receipt_ref = content_hash({"synthetic-base": True})
    record = data_incremental.manifests.fragment_record(
        inspection, _DAILY_MARKET_REF, input_receipt_refs=(receipt_ref,),
        import_request_hash=receipt_ref)
    manifest = dataset_manifest(
        _DAILY_MARKET_REF, (record,), knowledge_mode="reconstructed",
        coverage_receipt_refs=(receipt_ref,), availability_evidence_refs=())
    parent_ref = snapshot_ref(
        {"daily_market": manifest}, calendar_version="cal.v1",
        source_priority_version="synthetic", finality_receipt_refs=(receipt_ref,))
    commit_snapshot(
        conn, scope="shadow", request_hash=content_hash({"base": "r0"}),
        contracts=(contract,), objects=(object_ref,), records=(record,), manifests=(manifest,),
        snapshot=parent_ref, expected_head_snapshot_id=None, expected_head_generation=0,
        receipt_id="base-receipt", attempt_id="base-attempt", fence=1,
        fence_check=lambda _c: None, clock=clock, store=store)
    parent = Repository(conn).resolve_full(parent_ref.snapshot_id)

    append_row = dict(rows[0], ticker="ZZZ")
    append_raw = data_incremental.cache_raw_receipt(
        conn, store, data_incremental.RawPayload(
            payload=b'{"kind":"synthetic-append"}', response_kind="complete",
            response_meta={}),
        source="synthetic", endpoint="daily_market",
        request={"ticker": "ZZZ"}, received_at=clock.now().isoformat())
    revision = _frozen_revision(append_row, append_raw.raw_receipt_id, "revision-1")
    (revision,), _ = data_incremental._stage_normalizations(
        conn, store, {append_raw.raw_receipt_id: append_raw}, (revision,),
        contract.contract_id, clock)
    coverage = _coverage(revision, append_raw.raw_receipt_id)
    candidate = data_incremental.build_daily_market_candidate(
        parent, store, (revision,), coverage=coverage,
        parent_snapshot_id=parent_ref.snapshot_id)

    request_hash = "sha256:" + "1" * 64
    receipt1 = data_incremental.commit_daily_market_candidate(
        conn, store, candidate, scope="shadow",
        expected_head_snapshot_id=parent_ref.snapshot_id, expected_head_generation=1,
        clock=clock, request_hash=request_hash, receipt_id="receipt-1",
        attempt_id="attempt-1", fence=1)
    assert receipt1.resulting_head_snapshot_id != parent_ref.snapshot_id

    # A genuinely DIFFERENT candidate -- different ticker/raw/revision, built
    # against the SAME pre-receipt-1 ``parent`` -- races against the stale
    # head expectation with a caller-supplied fence_check that is itself a
    # permissive no-op. Its own resulting snapshot does not match what is
    # actually at head, so the composed head fence must still refuse
    # SNAPSHOT_CONFLICT: the no-op fence_check must not replace the head
    # fence outright and silently return a prior receipt.
    append_row2 = dict(rows[0], ticker="YYY")
    append_raw2 = data_incremental.cache_raw_receipt(
        conn, store, data_incremental.RawPayload(
            payload=b'{"kind":"synthetic-append-2"}', response_kind="complete",
            response_meta={}),
        source="synthetic", endpoint="daily_market",
        request={"ticker": "YYY"}, received_at=clock.now().isoformat())
    revision2 = _frozen_revision(append_row2, append_raw2.raw_receipt_id, "revision-2")
    (revision2,), _ = data_incremental._stage_normalizations(
        conn, store, {append_raw2.raw_receipt_id: append_raw2}, (revision2,),
        contract.contract_id, clock)
    coverage2 = _coverage(revision2, append_raw2.raw_receipt_id)
    candidate2 = data_incremental.build_daily_market_candidate(
        parent, store, (revision2,), coverage=coverage2,
        parent_snapshot_id=parent_ref.snapshot_id)

    with pytest.raises(DataError) as err:
        data_incremental.commit_daily_market_candidate(
            conn, store, candidate2, scope="shadow",
            expected_head_snapshot_id=parent_ref.snapshot_id, expected_head_generation=1,
            clock=clock, request_hash="sha256:" + "2" * 64, receipt_id="receipt-2",
            attempt_id="attempt-2", fence=2, fence_check=lambda _c: None)
    assert err.value.code == "SNAPSHOT_CONFLICT"


def test_retry_of_the_same_effect_after_a_crash_before_attempt_commit_succeeds(tmp_path):
    """#98: a retry using the SAME candidate/receipt/attempt (simulating a crash between
    commit_daily_market_candidate's own commit and the caller's later attempt-commit step)
    must succeed as an idempotent replay, not fail permanently with SNAPSHOT_CONFLICT,
    even though the head has since moved past the retry's stale expected_head_snapshot_id."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    contract = _obj_contract("daily_market")
    rows = _obj_daily_market_rows()
    table = _table_from_rows(contract, rows)
    ref = store.publish_bytes(_to_bytes(table), schema_ref="parquet_fragment.v1")
    object_ref = data_incremental.ObjectRef(
        kind="parquet_fragment", object_id=ref.artifact_id,
        content_hash=ref.content_hash, byte_size=ref.byte_size)
    inspection = inspect_fragment(store, object_ref, contract, _DAILY_MARKET_REF, "2024")
    receipt_ref = content_hash({"synthetic-base": True})
    record = data_incremental.manifests.fragment_record(
        inspection, _DAILY_MARKET_REF, input_receipt_refs=(receipt_ref,),
        import_request_hash=receipt_ref)
    manifest = dataset_manifest(
        _DAILY_MARKET_REF, (record,), knowledge_mode="reconstructed",
        coverage_receipt_refs=(receipt_ref,), availability_evidence_refs=())
    parent_ref = snapshot_ref(
        {"daily_market": manifest}, calendar_version="cal.v1",
        source_priority_version="synthetic", finality_receipt_refs=(receipt_ref,))
    commit_snapshot(
        conn, scope="shadow", request_hash=content_hash({"base": "r0"}),
        contracts=(contract,), objects=(object_ref,), records=(record,), manifests=(manifest,),
        snapshot=parent_ref, expected_head_snapshot_id=None, expected_head_generation=0,
        receipt_id="base-receipt", attempt_id="base-attempt", fence=1,
        fence_check=lambda _c: None, clock=clock, store=store)
    parent = Repository(conn).resolve_full(parent_ref.snapshot_id)

    append_row = dict(rows[0], ticker="ZZZ")
    append_raw = data_incremental.cache_raw_receipt(
        conn, store, data_incremental.RawPayload(
            payload=b'{"kind":"synthetic-append"}', response_kind="complete",
            response_meta={}),
        source="synthetic", endpoint="daily_market",
        request={"ticker": "ZZZ"}, received_at=clock.now().isoformat())
    revision = _frozen_revision(append_row, append_raw.raw_receipt_id, "revision-1")
    (revision,), _ = data_incremental._stage_normalizations(
        conn, store, {append_raw.raw_receipt_id: append_raw}, (revision,),
        contract.contract_id, clock)
    coverage = _coverage(revision, append_raw.raw_receipt_id)
    candidate = data_incremental.build_daily_market_candidate(
        parent, store, (revision,), coverage=coverage,
        parent_snapshot_id=parent_ref.snapshot_id)

    request_hash = "sha256:" + "1" * 64
    receipt1 = data_incremental.commit_daily_market_candidate(
        conn, store, candidate, scope="shadow",
        expected_head_snapshot_id=parent_ref.snapshot_id, expected_head_generation=1,
        clock=clock, request_hash=request_hash, receipt_id="receipt-1",
        attempt_id="attempt-1", fence=1)
    assert receipt1.resulting_head_snapshot_id != parent_ref.snapshot_id

    # Crash here: the effect's own commit landed (the head advanced to
    # candidate.snapshot.snapshot_id) but the caller's later attempt-commit
    # step never ran. The retry re-presents the SAME stale expected head.
    receipt2 = data_incremental.commit_daily_market_candidate(
        conn, store, candidate, scope="shadow",
        expected_head_snapshot_id=parent_ref.snapshot_id, expected_head_generation=1,
        clock=clock, request_hash=request_hash, receipt_id="receipt-1",
        attempt_id="attempt-1", fence=1)

    assert receipt2.receipt_id == receipt1.receipt_id
    assert receipt2.resulting_head_snapshot_id == receipt1.resulting_head_snapshot_id
    assert receipt2.resulting_head_snapshot_id == candidate.snapshot.snapshot_id
    head = conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope='shadow'"
    ).fetchone()
    assert (head["snapshot_id"], head["generation"]) == (
        candidate.snapshot.snapshot_id, 2)
    resolved = Repository(conn).resolve_full(receipt2.resulting_head_snapshot_id)
    manifest_ids = {item.fragment_id for item in
                    resolved.table_manifests["daily_market"].fragment_refs}
    records = tuple(item for item in resolved.records if item.fragment_id in manifest_ids)
    zzz_rows = [row for row in data_incremental.load_daily_market_rows(
        store, records, contract) if row["ticker"] == "ZZZ"]
    assert len(zzz_rows) == 1


def test_normalizer_id_bump_changes_cache_identity(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    data_incremental.catalog._insert_contract(
        conn, _DAILY_MARKET_CONTRACT, clock.now().isoformat())
    contract_id = _DAILY_MARKET_CONTRACT.contract_id
    payload = data_incremental.RawPayload(
        payload=b"{}", response_kind="complete", response_meta={})
    raw = data_incremental.cache_raw_receipt(
        conn, store, payload, source="orats", endpoint="daily_market",
        request={"keys": ["AAA"]}, received_at=clock.now().isoformat())
    record_v1 = data_incremental.cache_normalization(
        conn, store, raw, (), normalizer_id="daily_market.v1", contract_id=contract_id,
        created_at=clock.now().isoformat())
    record_v2 = data_incremental.cache_normalization(
        conn, store, raw, (), normalizer_id="daily_market.v2", contract_id=contract_id,
        created_at=clock.now().isoformat())
    assert record_v1.normalization_id != record_v2.normalization_id


def _cache_scope_plan(receipt_id, expected_keys=("AAA",)):
    return {
        "units": [{"request_id": "req-1", "table_name": "daily_market",
                   "partition_key": "2026-09-15",
                   "expected_keys": list(expected_keys)}],
        "cached": [{"request_id": "req-1", "receipt_ref": receipt_id}],
        "fetch_units": [],
    }


def test_cached_fetched_units_refuses_scope_mismatch_before_store_verify(tmp_path,
                                                                         monkeypatch):
    """#142: a cached receipt whose request keys do not match the unit's
    expected keys must refuse ``INPUT_CHANGED`` from the catalog row alone --
    before any ``store.verify``/read and before the provider merge, so an
    out-of-scope receipt is never trusted or rebuilt into evidence."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    raw = data_incremental.cache_raw_receipt(
        conn, store,
        data_incremental.RawPayload(
            payload=json.dumps({"summaries": {"data": []},
                                "cores": {"data": []}}).encode(),
            response_kind="complete", response_meta={}),
        source=data_incremental.FETCH_SOURCE, endpoint="daily_market",
        request={"request_id": "req-1", "table_name": "daily_market",
                 "partition_key": "2026-09-15", "keys": ["OLD"]},
        received_at=clock.now().isoformat())
    verified = []

    def _watch_verify(ref):
        verified.append(ref)
        raise AssertionError("out-of-scope cached receipt must not be verified")

    monkeypatch.setattr(store, "verify", _watch_verify)

    def fetcher(unit):
        raise AssertionError("cached rebuild must not call the provider")

    def merge_ticker_rows(summaries, cores, expected_keys=None):
        raise AssertionError("out-of-scope cached receipt must not be merged")

    fetcher.merge_ticker_rows = merge_ticker_rows
    with pytest.raises(DataError) as err:
        data_incremental._cached_fetched_units(
            conn, store, _DAILY_MARKET_CONTRACT,
            _cache_scope_plan(raw.raw_receipt_id), fetcher)
    assert err.value.code == "INPUT_CHANGED"
    assert verified == []
    assert conn.execute(
        "SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM data_snapshot_heads").fetchone()[0] == 0


def test_cached_fetched_units_reacquires_when_complete_receipt_loses_a_key(tmp_path):
    """#142: a cached receipt that claims ``complete`` and carries the unit's own
    key set but no longer reconstructs every expected key is refused with
    ``INPUT_CHANGED`` in the cache-only branch -- before any provider request
    and before any replacement receipt. Reacquisition belongs to planning,
    which demotes the unit to a budgeted fetch where call reservations are
    made, so no ``_FetchedUnit`` is returned and no ``missing`` outcome is
    staged against a receipt this branch was never allowed to replay."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    aaa_row = dict(_obj_daily_market_rows()[0], date="2026-09-15", year=2026)
    raw = data_incremental.cache_raw_receipt(
        conn, store,
        data_incremental.RawPayload(
            payload=json.dumps({"summaries": {"data": [
                {"ticker": "AAA", "tradeDate": "2026-09-15", "stockPrice": 100.0}]},
                "cores": {"data": []}}).encode(),
            response_kind="complete", response_meta={}),
        source=data_incremental.FETCH_SOURCE, endpoint="daily_market",
        request={"request_id": "req-1", "table_name": "daily_market",
                 "partition_key": "2026-09-15", "keys": ["AAA", "BBB"]},
        received_at=clock.now().isoformat())

    def merge_ticker_rows(summaries, cores, expected_keys=None):
        del cores, expected_keys
        return [dict(aaa_row) for row in summaries
                if str(row.get("ticker")) == "AAA"]

    calls = []

    def fetcher(unit):
        calls.append(unit)
        raise AssertionError("cache-only replay must not call an unreserved provider")

    fetcher.merge_ticker_rows = merge_ticker_rows

    with pytest.raises(DataError) as err:
        data_incremental._cached_fetched_units(
            conn, store, _DAILY_MARKET_CONTRACT,
            _cache_scope_plan(raw.raw_receipt_id, expected_keys=("AAA", "BBB")), fetcher)

    assert err.value.code == "INPUT_CHANGED"
    message = err.value.problem.message
    assert "does not reconstruct every expected key" in message
    assert "demote this receipt to a budgeted fetch" in message
    assert calls == []
    rows = conn.execute(
        "SELECT raw_receipt_id, response_kind FROM data_raw_receipts").fetchall()
    assert [row["raw_receipt_id"] for row in rows] == [raw.raw_receipt_id]
    assert rows[0]["response_kind"] == "complete"


def _empty_keyed_unit(*, expected_keys):
    return {"request_id": "req-1", "table_name": "daily_market",
            "partition_key": "2026-09-15", "expected_keys": list(expected_keys)}


def _legitimate_empty_fetcher(unit):
    return b"{}", "legitimate_empty", {}, ()


def test_keyed_legitimate_empty_refuses_source_not_final_before_caching(tmp_path):
    """#142: a ``legitimate_empty`` response for a unit WITH expected keys is
    not a final answer for those keys -- refuse ``SOURCE_NOT_FINAL`` before
    ``cache_raw_receipt``, so the empty response leaves no receipt behind."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    with pytest.raises(DataError) as err:
        data_incremental._fetch_unit(
            conn, store, _DAILY_MARKET_CONTRACT,
            _empty_keyed_unit(expected_keys=("AAA",)), _legitimate_empty_fetcher)
    assert err.value.code == "SOURCE_NOT_FINAL"
    assert conn.execute(
        "SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 0


def test_unkeyed_legitimate_empty_still_caches_one_receipt(tmp_path):
    """#142 preserved behavior: with NO expected keys a ``legitimate_empty``
    response is a complete final answer -- it succeeds and caches exactly one
    raw receipt, as before the guard."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    fetched = data_incremental._fetch_unit(
        conn, store, _DAILY_MARKET_CONTRACT,
        _empty_keyed_unit(expected_keys=()), _legitimate_empty_fetcher)
    assert fetched.raw_payload["response_kind"] == "legitimate_empty"
    assert fetched.expected == ()
    assert fetched.outcomes == ()
    rows = conn.execute(
        "SELECT raw_receipt_id, response_kind FROM data_raw_receipts").fetchall()
    assert len(rows) == 1
    assert rows[0]["response_kind"] == "legitimate_empty"
    assert fetched.receipt_id == rows[0]["raw_receipt_id"]
