from datetime import datetime
from io import BytesIO
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
from engine.v2.foundation import (
    ArtifactStore,
    canonical_json,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.ops.incremental_data import RefreshParameters
from tests.ops_support import catalog


@pytest.mark.needs_data  # reads the real data/ root (gitignored, absent in CI and worktrees)
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
        candidate=candidate, row=corrected)
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


@pytest.mark.parametrize(
    ("table_name", "row", "timestamp_columns"),
    (
        (
            "earnings_events",
            {
                "event_id": "AAA_2025-01-15",
                "ticker": "AAA",
                "event_date": datetime(2025, 1, 15),
                "year": 2025,
                "session": "AMC",
                "session_src": None,
                "annc_tod": None,
                "src_orats": True,
                "src_oquants": False,
                "src_nasdaq": False,
                "src_yfinance": False,
                "date_agree": True,
                "date_conflict": False,
                "updated_at": None,
                "event_cluster_id": None,
                "claim_count": None,
                "reconciliation": None,
            },
            ("event_date",),
        ),
        (
            "option_chains",
            {
                "ticker": "AAA",
                "obs_date": datetime(2025, 1, 2),
                "year": 2025,
                "expiry": datetime(2025, 1, 17),
                "dte": 15,
                "strike": 100.0,
                "right": "C",
                "bid": 2.0,
                "ask": 2.2,
                "mid": 2.1,
                "iv": 0.4,
                "delta": 0.5,
                "spot": 100.0,
                "src": "fixture",
                "src_file": None,
                "chain_kind": "entry",
                "volume": None,
                "open_interest": None,
                "bid_size": None,
                "ask_size": None,
                "quote_repaired": False,
            },
            ("obs_date", "expiry"),
        ),
    ),
)
def test_generic_worker_decodes_json_timestamps_before_parquet_write(
        table_name, row, timestamp_columns):
    contract = from_document(
        TableContract, build_legacy_mapping()["tables"][table_name])
    logical_key = incremental_tables.logical_key_for_row(contract, row)
    candidate = RevisionCandidate(
        revision_id="json-" + table_name,
        logical_key=logical_key,
        source="fixture",
        source_priority=0,
        finality="final",
        revision_ordinal=1,
        received_at="2026-09-17T00:00:00Z",
        content_hash=incremental_tables.revision_hash(
            logical_key=logical_key, row=row, deleted=False),
    )
    document = {
        "candidate": to_document(candidate),
        "row": to_document(row),
        "deleted": False,
        "partition_key": None,
    }

    revision = daily_incremental._generic_revision_from_document(
        document, contract)

    assert all(isinstance(revision.row[name], datetime)
               for name in timestamp_columns)
    payload = generic_incremental._parquet_bytes(contract, (revision.row,))
    restored = pq.read_table(BytesIO(payload)).to_pylist()[0]
    assert all(isinstance(restored[name], datetime)
               for name in timestamp_columns)


def _worker_row(table_name, year):
    if table_name == "earnings_events":
        return {
            "event_id": "event-1",
            "ticker": "AAA",
            "event_date": datetime(year, 1, 15),
            "year": year,
            "session": "AMC",
            "session_src": None,
            "annc_tod": None,
            "src_orats": True,
            "src_oquants": False,
            "src_nasdaq": False,
            "src_yfinance": False,
            "date_agree": True,
            "date_conflict": False,
            "updated_at": None,
            "event_cluster_id": None,
            "claim_count": None,
            "reconciliation": None,
        }
    return {
        "ticker": "AAA",
        "obs_date": datetime(year, 1, 2),
        "year": year,
        "expiry": datetime(year, 1, 17),
        "dte": 15,
        "strike": 100.0,
        "right": "C",
        "bid": 2.0,
        "ask": 2.2,
        "mid": 2.1,
        "iv": 0.4,
        "delta": 0.5,
        "spot": 100.0,
        "src": "fixture",
        "src_file": None,
        "chain_kind": "entry",
        "volume": None,
        "open_interest": None,
        "bid_size": None,
        "ask_size": None,
        "quote_repaired": False,
    }


@pytest.mark.parametrize("table_name", ("earnings_events", "option_chains"))
def test_generic_refresh_worker_commits_json_timestamp_rows(tmp_path, table_name):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    contract = from_document(
        TableContract, build_legacy_mapping()["tables"][table_name])
    contract_ref = TableContractRef(
        contract_id=contract.contract_id,
        definition_hash=contract.definition_hash,
    )
    base_row = _worker_row(table_name, 2024)
    base_bytes = generic_incremental._parquet_bytes(contract, (base_row,))
    published = store.publish_bytes(
        base_bytes, schema_ref="parquet_fragment.v1.0")
    obj = generic_incremental.ObjectRef(
        kind="parquet_fragment",
        object_id=published.artifact_id,
        content_hash=published.content_hash,
        byte_size=published.byte_size,
    )
    inspection = inspect_fragment(
        store, obj, contract, contract_ref, "2024")
    receipt_ref = content_hash({"fixture": table_name})
    record = generic_incremental.manifests.fragment_record(
        inspection,
        contract_ref,
        input_receipt_refs=(receipt_ref,),
        import_request_hash=receipt_ref,
    )
    manifest = dataset_manifest(
        contract_ref,
        (record,),
        knowledge_mode="reconstructed",
        coverage_receipt_refs=(receipt_ref,),
        availability_evidence_refs=(),
    )
    parent = snapshot_ref(
        {table_name: manifest},
        calendar_version="cal.v1",
        source_priority_version="fixture",
        finality_receipt_refs=(receipt_ref,),
    )
    commit_snapshot(
        conn,
        scope="generic-worker",
        request_hash=content_hash({"base": table_name}),
        contracts=(contract,),
        objects=(obj,),
        records=(record,),
        manifests=(manifest,),
        snapshot=parent,
        expected_head_snapshot_id=None,
        expected_head_generation=0,
        receipt_id="base-" + table_name,
        attempt_id="base-attempt-" + table_name,
        fence=1,
        fence_check=lambda _conn: None,
        clock=clock,
        store=store,
    )

    incoming = dict(base_row)
    if table_name == "earnings_events":
        incoming["event_date"] = datetime(2025, 1, 15)
        incoming["year"] = 2025
        incoming["session"] = "BMO"
    else:
        incoming["bid"] = 2.1
    logical_key = incremental_tables.logical_key_for_row(contract, incoming)
    candidate = RevisionCandidate(
        revision_id="worker-" + table_name,
        logical_key=logical_key,
        source="fixture",
        source_priority=0,
        finality="final",
        revision_ordinal=2,
        received_at="2026-09-17T00:00:00Z",
        content_hash=incremental_tables.revision_hash(
            logical_key=logical_key, row=incoming, deleted=False),
    )
    coverage_key = CoverageKey(
        item_key=logical_key,
        session_date=str(incoming[
            "event_date" if table_name == "earnings_events" else "obs_date"
        ])[:10],
        ticker="AAA",
    )
    coverage = daily_incremental.build_completed_coverage(
        contract_ref,
        source="fixture",
        endpoint="fixture",
        interval=TimeInterval(
            column=("event_date" if table_name == "earnings_events"
                    else "obs_date"),
            start_inclusive=coverage_key.session_date,
            end_exclusive=coverage_key.session_date + "T23:59:59",
        ),
        expected=(coverage_key,),
        outcomes=(CoverageOutcome(
            key=coverage_key,
            status="present",
            receipt_id=receipt_ref,
            revision_id=candidate.revision_id,
            finality="final",
        ),),
        acquisition_receipt_refs=(receipt_ref,),
        completed_at="2026-09-17T00:00:00Z",
    )
    root = tmp_path / ("worker-" + table_name)
    root.mkdir()
    plan_hash = content_hash({"plan": table_name})
    document = {
        "catalog_path": str(tmp_path / "ops.sqlite"),
        "objects_root": str(tmp_path / "objects"),
        "scope": "generic-worker",
        "expected_head_snapshot_id": parent.snapshot_id,
        "expected_head_generation": 1,
        "table_name": table_name,
        "coverage": to_document(coverage),
        "generic_revisions": [{
            "candidate": to_document(candidate),
            "row": to_document(incoming),
            "deleted": False,
            "partition_key": None,
        }],
    }
    (root / "incremental_refresh_input.json").write_text(
        canonical_json(document))
    result = daily_incremental.run_incremental_refresh(
        RefreshParameters(
            expected_ids=("request-" + table_name,),
            parent_snapshot_id=parent.snapshot_id,
            refresh_plan_hash=plan_hash,
            provider_calls=0, catalog_path=str(tmp_path / "ops.sqlite"),
            objects_root=str(tmp_path / "objects"), scope="generic-worker",
            expected_head_generation=1,
            expected_head_snapshot_id=parent.snapshot_id, table_name=table_name,
        ),
        root,
    )

    assert result["status"] == "complete"
    resolved = Repository(conn).resolve_full(result["candidate_snapshot_id"])
    result_manifest = resolved.table_manifests[table_name]
    fragment_ids = {item.fragment_id for item in result_manifest.fragment_refs}
    result_records = tuple(
        item for item in resolved.records if item.fragment_id in fragment_ids)
    rows = generic_incremental._load_rows(
        store, result_records, contract)
    matches = [
        row for row in rows
        if incremental_tables.logical_key_for_row(contract, row) == logical_key
    ]
    assert len(matches) == 1
    if table_name == "earnings_events":
        assert matches[0]["year"] == 2025
        assert {item.partition_key for item in result_records} == {"2025"}
    else:
        assert matches[0]["bid"] == 2.1
