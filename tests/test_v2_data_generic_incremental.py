from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from engine.v2.contracts import (
    CoverageKey,
    CoverageOutcome,
    RevisionCandidate,
    EventRef,
    TableContract,
    TableContractRef,
    TimeInterval,
)
from engine.v2.data import generic_incremental, incremental_tables
from engine.v2.data import incremental as daily_incremental
from engine.v2.data.catalog import commit_snapshot
from engine.v2.data.legacy_mapping import build_legacy_mapping
from engine.v2.data.manifests import dataset_manifest, snapshot_ref
from engine.v2.data.objects import inspect_fragment
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, content_hash, from_document
from tests.ops_support import catalog


def test_generic_earnings_events_candidate_commits_atomically(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    source = Path("data/curated/earnings_events/year=1996/part-0000.parquet")
    contract = from_document(TableContract, build_legacy_mapping()["tables"]["earnings_events"])
    contract_ref = TableContractRef(contract_id=contract.contract_id,
                                    definition_hash=contract.definition_hash)
    base_bytes = source.read_bytes()
    base_rows = tuple(pq.read_table(source).slice(0, 2).to_pylist())
    published = store.publish_bytes(base_bytes, schema_ref="parquet_fragment.v1.0")
    obj = generic_incremental.ObjectRef(
        kind="parquet_fragment", object_id=published.artifact_id,
        content_hash=published.content_hash, byte_size=published.byte_size)
    inspection = inspect_fragment(store, obj, contract, contract_ref, "1996")
    receipt_ref = content_hash({"source": str(source)})
    record = generic_incremental.manifests.fragment_record(
        inspection, contract_ref, input_receipt_refs=(receipt_ref,),
        import_request_hash=receipt_ref)
    manifest = dataset_manifest(
        contract_ref, (record,), knowledge_mode="reconstructed",
        coverage_receipt_refs=(receipt_ref,), availability_evidence_refs=())
    parent = snapshot_ref(
        {"earnings_events": manifest}, calendar_version="cal.v1",
        source_priority_version="frozen", finality_receipt_refs=(receipt_ref,))
    commit_snapshot(
        conn, scope="generic", request_hash=content_hash({"base": 1}),
        contracts=(contract,), objects=(obj,), records=(record,), manifests=(manifest,),
        snapshot=parent, expected_head_snapshot_id=None, expected_head_generation=0,
        receipt_id="base-receipt", attempt_id="base-attempt", fence=1,
        fence_check=lambda _conn: None, clock=clock, store=store)

    corrected = dict(base_rows[0])
    corrected["session"] = "BMO" if corrected["session"] != "BMO" else "AMC"
    corrected["event_date"] = datetime(
        corrected["event_date"].year + 1, 1, 2)
    corrected["year"] = corrected["event_date"].year
    key = incremental_tables.logical_key_for_row(contract, corrected)
    candidate = RevisionCandidate(
        revision_id="event-correction", logical_key=key, source="frozen",
        source_priority=0, finality="final", revision_ordinal=2,
        received_at="2026-09-17T00:00:00Z",
        content_hash=incremental_tables.revision_hash(
            logical_key=key, row=corrected, deleted=False))
    revision = incremental_tables.GenericRevision(
        candidate=candidate, row=corrected, partition_key="1996")
    coverage_key = CoverageKey(item_key=key, session_date="1997-01-02", ticker=corrected["ticker"])
    coverage = daily_incremental.build_completed_coverage(
        contract_ref, source="frozen", endpoint="parquet",
        interval=TimeInterval(column="event_date", start_inclusive="1997-01-02",
                               end_exclusive="1998-01-01"),
        expected=(coverage_key,), outcomes=(CoverageOutcome(
            key=coverage_key, status="present", receipt_id=receipt_ref,
            revision_id="event-correction", finality="final"),),
        acquisition_receipt_refs=(receipt_ref,), completed_at="2026-09-17T00:00:00Z")
    resolved = Repository(conn).resolve_full(parent.snapshot_id)
    candidate_table = generic_incremental.build_generic_table_candidate(
        resolved, store, "earnings_events", (revision,), coverage=coverage,
        parent_snapshot_id=parent.snapshot_id)
    with pytest.raises(RuntimeError):
        generic_incremental.commit_generic_table_candidate(
            conn, store, candidate_table, scope="generic",
            expected_head_snapshot_id=parent.snapshot_id, expected_head_generation=1,
            clock=clock, request_hash="sha256:" + "1" * 64,
            receipt_id="generic-fault", attempt_id="generic-fault-attempt", fence=2,
            fault=lambda point: (_ for _ in ()).throw(RuntimeError(point))
            if point == "before_commit" else None)
    assert Repository(conn).resolve(parent.snapshot_id) == parent
    committed = generic_incremental.commit_generic_table_candidate(
        conn, store, candidate_table, scope="generic",
        expected_head_snapshot_id=parent.snapshot_id, expected_head_generation=1,
        clock=clock, request_hash="sha256:" + "1" * 64,
        receipt_id="generic-retry", attempt_id="generic-retry-attempt", fence=2)
    assert committed.resulting_head_snapshot_id == candidate_table.snapshot.snapshot_id
    assert Repository(conn).resolve(committed.resulting_head_snapshot_id) == candidate_table.snapshot
    moved = Repository(conn, store).get_event(
        EventRef(
            event_id=corrected["event_id"],
            calendar_revision=candidate_table.snapshot.table_versions[
                "earnings_events"].dataset_version_id),
        candidate_table.snapshot)
    assert moved.scheduled_event_date == "1997-01-02"
    replay_parent = Repository(conn).resolve_full(committed.resulting_head_snapshot_id)
    retained = generic_incremental.load_generic_revisions(conn, "earnings_events", contract)
    replay = generic_incremental.build_generic_table_candidate(
        replay_parent, store, "earnings_events", (revision,), coverage=coverage,
        retained=retained, parent_snapshot_id=replay_parent.snapshot.snapshot_id)
    assert replay.rewritten_partitions == 0
    assert replay.merge.changes == ()
