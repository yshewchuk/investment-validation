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
