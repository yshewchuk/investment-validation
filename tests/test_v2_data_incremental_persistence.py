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
from engine.v2.foundation import ArtifactStore, canonical_json, content_hash, to_document
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
        contract.contract_id, clock, {append_raw.raw_receipt_id: ("ZZZ",)})
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
        contract.contract_id, clock, {append_raw2.raw_receipt_id: ("YYY",)})
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
        contract.contract_id, clock, {append_raw.raw_receipt_id: ("ZZZ",)})
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
        created_at=clock.now().isoformat(), expected_keys=("AAA",))
    record_v2 = data_incremental.cache_normalization(
        conn, store, raw, (), normalizer_id="daily_market.v2", contract_id=contract_id,
        created_at=clock.now().isoformat(), expected_keys=("AAA",))
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


def test_cached_fetched_units_refuses_malformed_saved_keys_before_store_verify(tmp_path,
                                                                               monkeypatch):
    """#142: a cached receipt whose request carries malformed saved keys
    (``keys: None``) refuses ``INPUT_CHANGED`` from the saved-key shape guard
    alone -- the malformed set is never normalized, so no raw ``TypeError``
    leaks from ``_normalized_expected_keys``, and ``store.verify``, the
    provider fetch and the row merge all remain untouched."""
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
                 "partition_key": "2026-09-15", "keys": None},
        received_at=clock.now().isoformat())
    verified = []

    def _watch_verify(ref):
        verified.append(ref)
        raise AssertionError("malformed saved keys must not reach store.verify")

    monkeypatch.setattr(store, "verify", _watch_verify)

    def fetcher(unit):
        raise AssertionError("cache-only replay must not call the provider")

    def merge_ticker_rows(summaries, cores, expected_keys=None):
        raise AssertionError("malformed saved keys must not reach the merge")

    fetcher.merge_ticker_rows = merge_ticker_rows
    with pytest.raises(DataError) as err:
        data_incremental._cached_fetched_units(
            conn, store, _DAILY_MARKET_CONTRACT,
            _cache_scope_plan(raw.raw_receipt_id), fetcher)
    assert err.value.code == "INPUT_CHANGED"
    assert verified == []
    rows = conn.execute(
        "SELECT raw_receipt_id FROM data_raw_receipts").fetchall()
    assert [row["raw_receipt_id"] for row in rows] == [raw.raw_receipt_id]
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


def test_empty_planned_keys_refuse_before_the_fetcher(tmp_path):
    """#142 planned-key contract supersedes the old unkeyed behavior: an
    empty planned ``expected_keys`` is malformed at the ``_fetch_unit``
    boundary, so it refuses with the registered retryable ``INPUT_CHANGED``
    BEFORE the fetcher is invoked -- even a fetcher that would answer
    ``legitimate_empty``. No unkeyed planned response is ever cached."""
    conn, _clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    calls = []

    def fetcher(unit):
        calls.append(unit)
        return _legitimate_empty_fetcher(unit)

    with pytest.raises(DataError) as err:
        data_incremental._fetch_unit(
            conn, store, _DAILY_MARKET_CONTRACT,
            _empty_keyed_unit(expected_keys=()), fetcher)
    assert err.value.code == "INPUT_CHANGED"
    assert calls == []
    assert conn.execute(
        "SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 0


def _normalize_cache_setup(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    data_incremental.catalog._insert_contract(
        conn, _DAILY_MARKET_CONTRACT, clock.now().isoformat())
    raw = data_incremental.cache_raw_receipt(
        conn, store, data_incremental.RawPayload(
            payload=b'{"ticker":"AAA"}', response_kind="complete", response_meta={}),
        source="fixture", endpoint="daily_market",
        request={"keys": ["AAA", "BBB"]}, received_at=clock.now().isoformat())
    return conn, clock, store, raw, _DAILY_MARKET_CONTRACT.contract_id


def test_expected_key_sets_are_folded_into_normalization_identity(tmp_path):
    """#133 slice 3: the same raw payload/normalizer/contract/revisions under
    two distinct expected-key sets produce distinct normalization ids and two
    persisted rows (never a raw sqlite3.IntegrityError -- v13 dropped the
    tuple UNIQUE), while reordering or duplicating keys within an otherwise
    identical set stays the same identity."""
    conn, clock, store, raw, contract_id = _normalize_cache_setup(tmp_path)
    revision = _frozen_revision(_obj_daily_market_rows()[0], raw.raw_receipt_id,
                                "revision-1")

    def cache(keys):
        return data_incremental.cache_normalization(
            conn, store, raw, (revision,), normalizer_id="daily_market.v3",
            contract_id=contract_id, created_at=clock.now().isoformat(),
            expected_keys=keys)

    aaa = cache(("AAA",))
    assert aaa.cache_hit is False
    aaa_dupe = cache(("AAA", "AAA"))
    assert aaa_dupe.normalization_id == aaa.normalization_id
    assert aaa_dupe.cache_hit is True
    bbb = cache(("BBB",))
    assert bbb.normalization_id != aaa.normalization_id
    assert bbb.cache_hit is False
    bbb_dupe = cache(("BBB", "BBB", "BBB"))
    assert bbb_dupe.normalization_id == bbb.normalization_id
    assert bbb_dupe.cache_hit is True
    both = cache(("AAA", "BBB"))
    assert both.normalization_id not in {aaa.normalization_id, bbb.normalization_id}
    assert both.cache_hit is False
    reordered = cache(("BBB", "AAA"))
    assert reordered.normalization_id == both.normalization_id
    assert reordered.cache_hit is True
    grouped = conn.execute(
        "SELECT normalization_id, COUNT(*) FROM data_normalizations"
        " GROUP BY normalization_id").fetchall()
    assert len(grouped) == 3
    assert all(row[1] == 1 for row in grouped)


def test_changed_payload_under_same_expected_key_set_refuses_identity_conflict(tmp_path):
    """#133 slice 3: a changed normalized payload under the same raw hash,
    normalizer, contract and SAME canonical expected-key set is a genuine
    conflict -- typed IDENTITY_CONFLICT, no row overwrite."""
    conn, clock, store, raw, contract_id = _normalize_cache_setup(tmp_path)
    row = _obj_daily_market_rows()[0]
    first = data_incremental.cache_normalization(
        conn, store, raw,
        (_frozen_revision(row, raw.raw_receipt_id, "revision-1"),),
        normalizer_id="daily_market.v3", contract_id=contract_id,
        created_at=clock.now().isoformat(), expected_keys=("AAA",))
    assert first.cache_hit is False
    changed = _frozen_revision(row, raw.raw_receipt_id, "revision-1",
                               spot=row["spot"] + 0.05)
    with pytest.raises(DataError) as err:
        data_incremental.cache_normalization(
            conn, store, raw, (changed,), normalizer_id="daily_market.v3",
            contract_id=contract_id, created_at=clock.now().isoformat(),
            expected_keys=("AAA", "AAA"))
    assert err.value.code == "IDENTITY_CONFLICT"
    assert conn.execute(
        "SELECT COUNT(*) FROM data_normalizations").fetchone()[0] == 1


def test_legacy_triple_only_row_stays_addressable_but_is_never_a_cache_hit(tmp_path):
    """#133 slice 3 legacy policy (engine/v2/data/ARCHITECTURE.md): a row
    keyed by the old triple-only formula carries no expected-key metadata, so
    it stays unchanged and addressable by its legacy id but is never reused as
    a cache hit; the next request writes/uses the expected-set-scoped id and
    both rows coexist. The legacy row is seeded through the real catalog,
    store and canonical hashes at the old formula's id (incremental.py's
    pre-#133 ``norm_`` identity), not through a mock."""
    conn, clock, store, raw, contract_id = _normalize_cache_setup(tmp_path)
    normalizer_id = "daily_market.v3"
    legacy_document = {"schema_version": data_incremental.NORMALIZED_SCHEMA_REF,
                       "raw_hash": raw.raw_hash, "normalizer_id": normalizer_id,
                       "revisions": []}
    published = store.publish_bytes(canonical_json(legacy_document).encode("utf-8"),
                                    schema_ref=data_incremental.NORMALIZED_SCHEMA_REF)
    object_ref = data_incremental.ObjectRef(
        kind="normalized_daily_market", object_id=published.artifact_id,
        content_hash=published.content_hash, byte_size=published.byte_size)
    legacy_id = "norm_" + content_hash({
        "raw_hash": raw.raw_hash, "normalizer_id": normalizer_id,
        "contract_id": contract_id}).removeprefix("sha256:")[:32]
    conn.execute(
        "INSERT INTO data_normalizations (normalization_id, raw_hash, normalizer_id, "
        "contract_id, normalized_hash, artifact_ref_json, row_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (legacy_id, raw.raw_hash, normalizer_id, contract_id,
         content_hash(legacy_document), canonical_json(to_document(object_ref)),
         0, clock.now().isoformat()))
    legacy_before = conn.execute(
        "SELECT * FROM data_normalizations WHERE normalization_id = ?",
        (legacy_id,)).fetchone()
    assert legacy_before is not None
    assert "expected_keys" not in json.loads(store.read_verified(published))

    fresh = data_incremental.cache_normalization(
        conn, store, raw, (), normalizer_id=normalizer_id, contract_id=contract_id,
        created_at=clock.now().isoformat(), expected_keys=("AAA",))
    assert fresh.normalization_id != legacy_id
    assert fresh.cache_hit is False
    replay = data_incremental.cache_normalization(
        conn, store, raw, (), normalizer_id=normalizer_id, contract_id=contract_id,
        created_at=clock.now().isoformat(), expected_keys=("AAA",))
    assert replay.normalization_id == fresh.normalization_id
    assert replay.cache_hit is True

    legacy_after = conn.execute(
        "SELECT * FROM data_normalizations WHERE normalization_id = ?",
        (legacy_id,)).fetchone()
    assert tuple(legacy_after) == tuple(legacy_before)
    ids = sorted(row[0] for row in conn.execute(
        "SELECT normalization_id FROM data_normalizations"))
    assert ids == sorted([legacy_id, fresh.normalization_id])


def test_expected_keys_derive_only_for_referenced_receipts(tmp_path):
    """#133 slice 3 edge case (contract: engine/v2/data/ARCHITECTURE.md): the
    refresh derives its expected-key map only for receipts incoming revisions
    reference. A staged raw receipt with no saved ``request["keys"]`` that
    nothing references is never inspected and cannot abort the derivation; a
    referenced receipt still returns its saved keys and still fails closed
    with typed ``INPUT_CHANGED`` when they are missing; an unknown reference
    stays unmapped and keeps failing through ``_stage_normalizations``."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")

    def cache(payload, request):
        return data_incremental.cache_raw_receipt(
            conn, store, data_incremental.RawPayload(
                payload=payload, response_kind="complete", response_meta={}),
            source="synthetic", endpoint="daily_market",
            request=request, received_at=clock.now().isoformat())

    rows = _obj_daily_market_rows()
    referenced = cache(b'{"kind":"referenced"}', {"ticker": "AAA", "keys": ["AAA", "BBB"]})
    unused = cache(b'{"kind":"unused"}', {"ticker": "CCC"})
    broken = cache(b'{"kind":"broken"}', {"ticker": "DDD"})
    raw_records = {item.raw_receipt_id: item for item in (referenced, unused, broken)}
    revision = _frozen_revision(rows[0], referenced.raw_receipt_id, "revision-1")
    broken_revision = _frozen_revision(rows[1], broken.raw_receipt_id, "revision-2")
    orphan = _frozen_revision(rows[2], "raw-unknown", "revision-3")

    assert data_incremental._incoming_expected_keys(raw_records, ()) == {}
    assert data_incremental._incoming_expected_keys(raw_records, (orphan,)) == {}
    assert data_incremental._incoming_expected_keys(raw_records, (revision,)) == {
        referenced.raw_receipt_id: ("AAA", "BBB")}

    with pytest.raises(DataError) as err:
        data_incremental._incoming_expected_keys(raw_records, (broken_revision,))
    assert err.value.code == "INPUT_CHANGED"

    with pytest.raises(DataError) as err:
        data_incremental._stage_normalizations(
            conn, store, raw_records, (orphan,),
            _DAILY_MARKET_CONTRACT.contract_id, clock,
            data_incremental._incoming_expected_keys(raw_records, (orphan,)))
    assert err.value.code == "INPUT_CHANGED"
    assert conn.execute(
        "SELECT COUNT(*) FROM data_normalizations").fetchone()[0] == 0


def test_unreferenced_keyless_raw_receipt_does_not_abort_refresh(tmp_path):
    """#133 slice 3 edge case through ``run_incremental_refresh``: a staged
    raw receipt no incoming revision references, whose saved request carries
    no expected keys, must not abort a refresh it will never be normalized
    for; the referenced receipt's saved keys still drive the normalization
    identity and the refresh commits."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    receipt_ref = content_hash({"keyless-sidecar": "parent"})
    manifest = dataset_manifest(
        _DAILY_MARKET_REF, (), knowledge_mode="reconstructed",
        coverage_receipt_refs=(receipt_ref,), availability_evidence_refs=())
    parent = snapshot_ref(
        {"daily_market": manifest}, calendar_version="cal.v1",
        source_priority_version="synthetic", finality_receipt_refs=(receipt_ref,))
    commit_snapshot(
        conn, scope="shadow", request_hash=content_hash({"keyless-sidecar": "base"}),
        contracts=(_DAILY_MARKET_CONTRACT,), objects=(), records=(), manifests=(manifest,),
        snapshot=parent, expected_head_snapshot_id=None, expected_head_generation=0,
        receipt_id="base-receipt", attempt_id="base-attempt", fence=1,
        fence_check=lambda _c: None, clock=clock, store=store)

    append_row = dict(_obj_daily_market_rows()[0], ticker="ZZZ")
    referenced = data_incremental.cache_raw_receipt(
        conn, store, data_incremental.RawPayload(
            payload=b'{"kind":"keyless-sidecar-append"}', response_kind="complete",
            response_meta={}),
        source="synthetic", endpoint="daily_market",
        request={"ticker": "ZZZ", "keys": ["ZZZ"]}, received_at=clock.now().isoformat())
    unused = data_incremental.cache_raw_receipt(
        conn, store, data_incremental.RawPayload(
            payload=b'{"kind":"keyless-sidecar-unused"}', response_kind="complete",
            response_meta={}),
        source="synthetic", endpoint="daily_market",
        request={"ticker": "YYY"}, received_at=clock.now().isoformat())
    revision = _frozen_revision(append_row, referenced.raw_receipt_id,
                               "revision-sidecar")
    root, params = _refresh_input(
        tmp_path, parent.snapshot_id, 1, "sha256:" + "4" * 64, (revision,),
        ({"receipt_id": referenced.raw_receipt_id},
         {"receipt_id": unused.raw_receipt_id}),
        _coverage(revision, referenced.raw_receipt_id))

    result = data_incremental.run_incremental_refresh(params, root)

    assert result["status"] == "complete"
    assert conn.execute(
        "SELECT COUNT(*) FROM data_normalizations").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM data_daily_market_revisions").fetchone()[0] == 1


def test_malformed_saved_key_shapes_fail_closed_with_input_changed(tmp_path):
    """PR #400 CodeRabbit finding (contract: engine/v2/data/ARCHITECTURE.md):
    a raw receipt needed for normalization must carry ``request["keys"]`` as a
    nonempty list of nonempty strings. Missing, null, empty, a non-list value
    or any non-string/empty-string member refuses with the registered
    retryable ``INPUT_CHANGED`` -- never coercion, never a silent empty set --
    while a valid list is returned unchanged."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")

    def receipt(name, request):
        return data_incremental.cache_raw_receipt(
            conn, store, data_incremental.RawPayload(
                payload=json.dumps({"kind": name}).encode(),
                response_kind="complete", response_meta={}),
            source="synthetic", endpoint="daily_market",
            request=request, received_at=clock.now().isoformat())

    malformed = (
        ("missing", {"ticker": "AAA"}),
        ("null", {"keys": None}),
        ("empty", {"keys": []}),
        ("non-list-string", {"keys": "AAA"}),
        ("non-list-dict", {"keys": {"ticker": "AAA"}}),
        ("non-list-int", {"keys": 7}),
        ("non-list-tuple", {"keys": ("AAA", "BBB")}),
        ("empty-member", {"keys": ["AAA", ""]}),
        ("non-string-member", {"keys": ["AAA", 42]}),
        ("null-member", {"keys": ["AAA", None]}),
    )
    revision_row = _obj_daily_market_rows()[0]
    for name, request in malformed:
        raw = receipt(name, request)
        with pytest.raises(DataError) as err:
            data_incremental._receipt_expected_keys(raw)
        assert err.value.code == "INPUT_CHANGED", name
        assert err.value.problem.retryable is True, name
        referenced = _frozen_revision(revision_row, raw.raw_receipt_id,
                                      "revision-" + name)
        with pytest.raises(DataError) as err:
            data_incremental._incoming_expected_keys(
                {raw.raw_receipt_id: raw}, (referenced,))
        assert err.value.code == "INPUT_CHANGED", name

    valid = receipt("valid", {"keys": ["AAA", "BBB"]})
    assert data_incremental._receipt_expected_keys(valid) == ("AAA", "BBB")


def test_empty_expected_key_set_refuses_before_any_normalization_row(tmp_path):
    """PR #400 CodeRabbit finding: an empty canonical expected-key set refuses
    with the registered retryable ``INPUT_CHANGED`` before it is hashed into a
    normalization identity, so no ``data_normalizations`` row is ever inserted
    under a silently empty set."""
    conn, clock, store, raw, contract_id = _normalize_cache_setup(tmp_path)
    with pytest.raises(DataError) as err:
        data_incremental._normalization_identity(
            raw.raw_hash, "daily_market.v3", contract_id, ())
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.retryable is True
    with pytest.raises(DataError) as err:
        data_incremental.cache_normalization(
            conn, store, raw, (), normalizer_id="daily_market.v3",
            contract_id=contract_id, created_at=clock.now().isoformat(),
            expected_keys=())
    assert err.value.code == "INPUT_CHANGED"
    assert conn.execute(
        "SELECT COUNT(*) FROM data_normalizations").fetchone()[0] == 0


def test_valid_string_key_lists_keep_the_canonical_identity(tmp_path):
    """PR #400 CodeRabbit finding guard: nonempty string key lists keep the
    existing set-based identity formula -- order and duplicates never change
    it, a genuinely different set (or raw hash) always does."""
    conn, clock, store, raw, contract_id = _normalize_cache_setup(tmp_path)
    normalizer = "daily_market.v3"
    base = data_incremental._normalization_identity(
        raw.raw_hash, normalizer, contract_id, ("AAA", "BBB"))
    assert base.startswith("norm_")
    assert base == data_incremental._normalization_identity(
        raw.raw_hash, normalizer, contract_id, ["BBB", "AAA", "AAA"])
    assert base != data_incremental._normalization_identity(
        raw.raw_hash, normalizer, contract_id, ("AAA",))
    assert base != data_incremental._normalization_identity(
        raw.raw_hash, normalizer, contract_id, ("AAA", "CCC"))
    assert base != data_incremental._normalization_identity(
        content_hash({"other": "raw"}), normalizer, contract_id, ("AAA", "BBB"))


def test_malformed_directly_passed_expected_keys_refuse_before_identity(tmp_path):
    """PR #400 gate finding (contract: engine/v2/data/ARCHITECTURE.md): the
    normalization boundary validates, it never coerces. An empty sequence, a
    scalar string/bytes, a mapping/set, or any non-string/empty-string member
    refuses with the registered retryable ``INPUT_CHANGED`` through both
    ``cache_normalization`` and ``_normalization_identity`` before any identity
    is derived, so no ``data_normalizations`` row exists afterwards -- while a
    valid tuple/list keeps the set-based identity, order- and duplicate-wise."""
    conn, clock, store, raw, contract_id = _normalize_cache_setup(tmp_path)
    normalizer = "daily_market.v3"

    def cache(keys):
        return data_incremental.cache_normalization(
            conn, store, raw, (), normalizer_id=normalizer,
            contract_id=contract_id, created_at=clock.now().isoformat(),
            expected_keys=keys)

    malformed = (
        ("empty-tuple", ()),
        ("empty-list", []),
        ("scalar-string", "AAA"),
        ("scalar-bytes", b"AAA"),
        ("mapping", {"AAA": "ignored"}),
        ("set", {"AAA"}),
        ("non-string-member", (7,)),
        ("mixed-member", ["AAA", 42]),
        ("null-member", ("AAA", None)),
        ("empty-string-member", ("AAA", "")),
        ("only-empty-string-member", ("",)),
    )
    for name, keys in malformed:
        with pytest.raises(DataError) as err:
            cache(keys)
        assert err.value.code == "INPUT_CHANGED", name
        assert err.value.problem.retryable is True, name
        with pytest.raises(DataError) as err:
            data_incremental._normalization_identity(
                raw.raw_hash, normalizer, contract_id, keys)
        assert err.value.code == "INPUT_CHANGED", name
        assert err.value.problem.retryable is True, name
    assert conn.execute(
        "SELECT COUNT(*) FROM data_normalizations").fetchone()[0] == 0

    tuple_id = cache(("AAA", "BBB")).normalization_id
    assert tuple_id.startswith("norm_")
    assert cache(["BBB", "AAA", "AAA"]).normalization_id == tuple_id
    assert cache(("AAA", "BBB")).cache_hit is True
    assert conn.execute("SELECT COUNT(*) FROM data_normalizations").fetchone()[0] == 1


_ABSENT_PLANNED_KEYS = object()

PLANNED_KEYS_MALFORMED = (
    pytest.param([], id="empty-list"),
    pytest.param([42], id="int-member-only"),
    pytest.param(["AAA", 42], id="int-member"),
    pytest.param(["AAA", None], id="null-member"),
    pytest.param(["", "AAA"], id="empty-string-member"),
    pytest.param("AAA", id="scalar-string"),
    pytest.param({"AAA": "ignored"}, id="mapping"),
    pytest.param(None, id="null"),
    pytest.param(_ABSENT_PLANNED_KEYS, id="absent-field"),
)


def _planned_keys_acquisition(tmp_path):
    """A real catalog and store with a committed empty-manifest
    ``daily_market`` parent, plus the staged identity document and job
    parameters a planned refresh acquisition receives."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    receipt_ref = content_hash({"planned-keys": "parent"})
    manifest = dataset_manifest(
        _DAILY_MARKET_REF, (), knowledge_mode="reconstructed",
        coverage_receipt_refs=(receipt_ref,), availability_evidence_refs=())
    parent = snapshot_ref(
        {"daily_market": manifest}, calendar_version="cal.v1",
        source_priority_version="synthetic", finality_receipt_refs=(receipt_ref,))
    commit_snapshot(
        conn, scope="shadow", request_hash=content_hash({"planned-keys": "base"}),
        contracts=(_DAILY_MARKET_CONTRACT,), objects=(), records=(),
        manifests=(manifest,), snapshot=parent, expected_head_snapshot_id=None,
        expected_head_generation=0, receipt_id="planned-keys-base-receipt",
        attempt_id="planned-keys-base-attempt", fence=1,
        fence_check=lambda _c: None, clock=clock, store=store)
    document = {"catalog_path": str(tmp_path / "ops.sqlite"),
                "objects_root": str(tmp_path / "objects"), "table_name": "daily_market"}
    parameters = RefreshParameters(
        expected_ids=("daily-market-planned-keys",),
        parent_snapshot_id=parent.snapshot_id, refresh_plan_hash="sha256:" + "9" * 64,
        provider_calls=0, catalog_path=str(tmp_path / "ops.sqlite"),
        objects_root=str(tmp_path / "objects"), scope="shadow",
        expected_head_generation=1, expected_head_snapshot_id=parent.snapshot_id)
    return conn, clock, store, document, parameters


def _spy_acquisition_store(monkeypatch):
    """Records every ``ArtifactStore`` the data-layer acquisition builds,
    proving no cache work starts before planned-key validation."""
    built = []
    real = data_incremental.ArtifactStore

    def recording(root):
        built.append(str(root))
        return real(root)

    monkeypatch.setattr(data_incremental, "ArtifactStore", recording)
    return built


@pytest.mark.parametrize("keys", PLANNED_KEYS_MALFORMED)
def test_malformed_planned_keys_refuse_before_provider_call_or_cache_write(
        tmp_path, monkeypatch, keys):
    """Planned-key contract (engine/v2/data/ARCHITECTURE.md): a
    malformed planned fetch-unit ``expected_keys`` -- an integer member, an
    empty or null set, a scalar string, a mapping -- and an absent
    ``expected_keys`` field alike refuse at refresh acquisition with the
    registered retryable ``INPUT_CHANGED``, before the provider is invoked,
    before any store or catalog work and before any ``str()`` coercion. The
    old ``_acquire_refresh_units`` called the fetcher first and persisted
    coerced keys, and its earlier presence-only guard raised a non-retryable
    ``CONTRACT_MISMATCH`` for the absent field, so every guard assertion here
    fails on it."""
    conn, _clock, _store, document, parameters = _planned_keys_acquisition(tmp_path)
    unit = {"request_id": "u1", "table_name": "daily_market",
            "partition_key": "2026-05-01"}
    if keys is not _ABSENT_PLANNED_KEYS:
        unit["expected_keys"] = keys
    (tmp_path / "refresh_plan.json").write_text(
        canonical_json({"fetch_units": [unit]}))
    built = _spy_acquisition_store(monkeypatch)
    provider_calls = []

    def fetcher(unit):
        provider_calls.append(unit)
        return b'{"kind": "fixture"}', "complete", {"status": 200}, []

    with pytest.raises(DataError) as err:
        data_incremental._acquire_refresh_units(parameters, tmp_path, document, fetcher)
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.retryable is True
    assert provider_calls == []
    assert built == []
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM data_normalizations").fetchone()[0] == 0
    assert "raw_payloads" not in document and "coverage" not in document


@pytest.mark.parametrize("keys", (
    pytest.param(["AAA", 42], id="int-member"),
    pytest.param([], id="empty-list"),
    pytest.param(_ABSENT_PLANNED_KEYS, id="absent-field"),
))
def test_malformed_planned_keys_in_cached_plan_refuse_before_receipt_read(
        tmp_path, monkeypatch, keys):
    """The cached/caller-supplied-plan half of the planned-key contract: the
    malformed or absent set reaches the data layer only through the raw plan
    document (never through plan decoding), so acquisition must refuse with
    the registered retryable ``INPUT_CHANGED`` before any receipt is opened
    or read. The old code constructed the store, opened and verified the
    receipt, then ``str``-coerced each key into the provider merge -- the
    merge spy, the store spy and the unchanged receipt row count prove it no
    longer does."""
    conn, clock, store, document, parameters = _planned_keys_acquisition(tmp_path)
    cached_raw = data_incremental.cache_raw_receipt(
        conn, store, data_incremental.RawPayload(
            payload=json.dumps(
                {"summaries": {"data": []}, "cores": {"data": []}}).encode(),
            response_kind="complete", response_meta={}),
        source="daily_market", endpoint="daily_market",
        request={"request_id": "u1", "keys": ["AAA"]},
        received_at=clock.now().isoformat())
    unit = {"request_id": "u1", "table_name": "daily_market",
            "partition_key": "2026-05-01"}
    if keys is not _ABSENT_PLANNED_KEYS:
        unit["expected_keys"] = keys
    (tmp_path / "refresh_plan.json").write_text(canonical_json({
        "units": [unit],
        "cached": [{"request_id": "u1", "receipt_ref": cached_raw.raw_receipt_id}]}))
    built = _spy_acquisition_store(monkeypatch)
    merge_calls = []

    def merge(summaries, cores, expected_keys=None):
        merge_calls.append(list(expected_keys or ()))
        return []

    def fetcher(unit):
        raise AssertionError("a cached replay must never call the provider")

    fetcher.merge_ticker_rows = merge

    with pytest.raises(DataError) as err:
        data_incremental._acquire_refresh_units(parameters, tmp_path, document, fetcher)
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.retryable is True
    assert merge_calls == []
    assert built == []
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM data_normalizations").fetchone()[0] == 0


@pytest.mark.parametrize("keys", (
    pytest.param(("AAA", 42), id="int-member-tuple"),
    pytest.param((), id="empty-tuple"),
    pytest.param("AAA", id="scalar-string"),
    pytest.param({"AAA"}, id="set"),
    pytest.param(None, id="null"),
))
def test_fetch_unit_validates_planned_keys_before_the_injected_fetcher(tmp_path, keys):
    """The guard sits inside ``_fetch_unit`` itself: the helper can be reached
    without ``_acquire_refresh_units``, so a direct call with a malformed
    planned set must refuse with the registered retryable ``INPUT_CHANGED``
    before the injected fetcher runs and before any receipt is cached. The
    old ``_fetch_unit`` called the fetcher first and wrote ``str``-coerced
    keys into the receipt request."""
    conn, _clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    provider_calls = []

    def fetcher(unit):
        provider_calls.append(unit)
        return b'{"kind": "fixture"}', "complete", {"status": 200}, []

    unit = {"request_id": "u1", "table_name": "daily_market",
            "partition_key": "2026-05-01", "expected_keys": keys}
    with pytest.raises(DataError) as err:
        data_incremental._fetch_unit(conn, store, _DAILY_MARKET_CONTRACT, unit, fetcher)
    assert err.value.code == "INPUT_CHANGED"
    assert err.value.problem.retryable is True
    assert provider_calls == []
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 0


def test_valid_planned_keys_are_preserved_on_the_receipt_and_coverage(tmp_path):
    """The positive half of the planned-key contract: a valid set reaches the
    receipt request and the staged coverage as strings, order and duplicates
    included -- no coercion, no re-sorting -- and the saved receipt still
    carries them unchanged for the normalization boundary, which
    canonicalizes downstream (the identity tests above pin that)."""
    conn, _clock, store, _document, _parameters = _planned_keys_acquisition(tmp_path)
    unit = {"request_id": "u1", "table_name": "daily_market",
            "partition_key": "2026-05-01", "expected_keys": ["BBB", "AAA", "AAA"]}
    base_row = _obj_daily_market_rows()[0]
    ticker_rows = [dict(base_row, ticker="BBB", date="2026-05-01", year=2026),
                   dict(base_row, ticker="AAA", date="2026-05-01", year=2026)]

    def fetcher(_unit):
        return b'{"kind": "fixture"}', "complete", {"status": 200}, ticker_rows

    fetched = data_incremental._fetch_unit(
        conn, store, _DAILY_MARKET_CONTRACT, unit, fetcher)
    assert fetched.raw_payload["request"]["keys"] == ["BBB", "AAA", "AAA"]
    assert tuple(key.ticker for key in fetched.expected) == ("BBB", "AAA", "AAA")
    stored = json.loads(conn.execute(
        "SELECT request_json FROM data_raw_receipts").fetchone()["request_json"])
    assert stored["keys"] == ["BBB", "AAA", "AAA"]
    replayed = data_incremental._staged_raw_receipt(
        conn, store, {"receipt_id": fetched.receipt_id})
    assert data_incremental._receipt_expected_keys(replayed) == ("BBB", "AAA", "AAA")
