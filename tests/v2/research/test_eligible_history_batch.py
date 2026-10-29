"""The eligible-history scan batch is bounded by one fragment, not by total rows."""
import os
import tracemalloc

import pyarrow as pa
import pyarrow.parquet as pq

from engine.v2.contracts.data import DataQuery, KeyPredicate
from engine.v2.data import manifests, objects
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT as CONTRACT
from engine.v2.data.repository import Repository
from experiments import eligible_history as history
from engine.v2.research.eligible_history_table import FEATURES, TABLE
from tests.data_scan_support import (
    IMPORT_REQUEST_HASH,
    RECEIPT,
    catalog_and_store,
    commit_tables,
    contract_ref_for,
    publish_bytes,
    table_from_rows,
)
from tests.v2.research.test_eligible_history import source


def test_stored_read_batch_cap_is_one_fragment_with_identical_rows(tmp_path, monkeypatch):
    data = source(tmp_path)
    try:
        request = {**data.request, "event_ids": [data.events[i]["event_id"] for i in (1, 5, -1)]}
        artifact = history.materialize(data.repository, data.snapshot, **request)
        caps, original = [], data.repository.scan

        def spy(query, **kwargs):
            if kwargs.get("table_name") == TABLE:
                caps.append(query.max_batch_rows)
            return original(query, **kwargs)

        monkeypatch.setattr(data.repository, "scan", spy)
        rows = history.read(data.repository, data.snapshot, artifact["feature_snapshot"],
            columns=list(FEATURES), **request)["rows"]
        assert len(rows) == 9
        assert caps == [3]
        assert [(r["event_id"], r["horizon_sessions"]) for r in rows] == sorted(
            (r["event_id"], r["horizon_sessions"]) for r in rows)
    finally:
        data.conn.close()


def _peak(tmp_path, fragments, batch_rows):
    """Peak Python allocation while scanning disjoint one-partition-per-fragment 3-row fragments."""
    tmp_path.mkdir()
    conn, clock, store = catalog_and_store(tmp_path)
    try:
        ref, records = contract_ref_for(CONTRACT), []
        for p in range(fragments):
            ticker = f"T{p:05d}"
            rows = [dict(ticker=ticker, event_date=f"2024-01-{i + 1:02}", session="AMC", skipped=False,
                source_id=os.urandom(50000).hex(), input_hash="sha256:" + "a" * 64) for i in range(3)]
            sink = pa.BufferOutputStream()
            pq.write_table(table_from_rows(CONTRACT, rows), sink)
            obj = publish_bytes(store, sink.getvalue().to_pybytes())
            records.append(manifests.fragment_record(objects.inspect_fragment(store, obj, CONTRACT, ref, ticker),
                ref, input_receipt_refs=(RECEIPT,), import_request_hash=IMPORT_REQUEST_HASH))
        snapshot = commit_tables(conn, clock, {CONTRACT.table_name: records}, {CONTRACT.table_name: CONTRACT})
        query = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=ref,
            columns=("ticker", "event_date", "source_id"), order_by=CONTRACT.primary_key,
            key_filter=(KeyPredicate(column="ticker", operator="in",
                values=tuple(f"T{p:05d}" for p in range(fragments))),),
            max_batch_rows=batch_rows or 3 * fragments, max_result_rows=3 * fragments)
        repository, seen = Repository(conn, store), []
        tracemalloc.start()
        try:
            for batch in repository.scan(query, table_name=CONTRACT.table_name, max_object_bytes=10**8):
                seen.extend(batch.column("ticker").to_pylist())
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert len(seen) == 3 * fragments and seen == sorted(seen)
        return peak
    finally:
        conn.close()


def test_scan_peak_is_flat_in_fragment_count_with_one_fragment_batches(tmp_path):
    small = _peak(tmp_path / "a", 10, 3)
    large = _peak(tmp_path / "b", 60, 3)
    assert large < small * 1.5
    # the all-rows cap the eligible-history read used to request grows with the row count
    assert _peak(tmp_path / "c", 60, None) > large * 3
