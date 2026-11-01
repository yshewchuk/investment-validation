"""Native target-object byte bounds before reads/decoding; no receipt changes."""
from contextlib import contextmanager
from random import Random
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import pytest

from engine.v2.data import objects
from engine.v2.data.errors import DataError
from engine.v2.research import magnitude_inputs
from engine.v2.research.event_history import MAX_OUTCOME_OBJECT_BYTES
from tests.v2.research.magnitude_trial_support import (
    commit_history_inputs,
    history_event,
    outcome_move,
)

LIMIT = 1 << 20


@contextmanager
def source(tmp_path, kind):
    # Real, correctly hashed Parquet: random hex resists compression; repeated
    # text compresses below the cap while its actual row group expands above it.
    payload = Random(0).randbytes(LIMIT).hex() if kind == "stored" else "x" * (2 * LIMIT) if kind == "expanded" else "synthetic"
    event = history_event("2024-05-06")
    conn, repository, snapshot = commit_history_inputs(tmp_path, [event],
        [outcome_move(event, -3., source_hash=payload)], include_calendar=False)
    try:
        record, = repository.fragment_records(snapshot, "computed_moves")
        store = repository._store
        metadata = pq.read_metadata(objects.verify_object_path(store, record.object_ref))
        expanded = sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups))
        yield SimpleNamespace(conn=conn, repository=repository, snapshot=snapshot, record=record,
            store=store, expanded=expanded, event=event)
    finally:
        conn.close()


def load(case):
    return magnitude_inputs.load_magnitude_targets(case.repository, case.snapshot,
        as_of_month="2025-01", labels_before="2024-07-01", event_ids=[case.event["event_id"]])


def observe(monkeypatch, case):
    calls = {"caps": [], "bounded": [], "stored": [], "path": [], "decoded": 0}
    target = case.record.object_ref.object_id
    original_scan, original_batches = case.repository.scan, case.repository._iter_batches
    original_bytes, original_path = objects.read_verified_bytes, objects.verify_object_path
    original_store = case.store.read_verified
    def scan(query, *, table_name, max_object_bytes=None):
        if table_name == "computed_moves":
            calls["caps"].append(max_object_bytes)
        return original_scan(query, table_name=table_name, max_object_bytes=max_object_bytes)
    def bounded(store, ref, *, max_bytes):
        if ref.object_id == target:
            calls["bounded"].append(max_bytes)
        return original_bytes(store, ref, max_bytes=max_bytes)
    def path(store, ref):
        if ref.object_id == target:
            calls["path"].append(ref.object_id)
        return original_path(store, ref)
    def stored(ref):
        payload = original_store(ref)
        if ref.artifact_id == target:
            calls["stored"].append(len(payload))
        return payload
    def batches(parquet_file, present, batch_cap):
        if "realized_move_pct" in parquet_file.schema_arrow.names:
            calls["decoded"] += 1
        return original_batches(parquet_file, present, batch_cap)
    monkeypatch.setattr(case.repository, "scan", scan)
    monkeypatch.setattr(case.repository, "_iter_batches", batches)
    monkeypatch.setattr(objects, "read_verified_bytes", bounded)
    monkeypatch.setattr(objects, "verify_object_path", path)
    monkeypatch.setattr(case.store, "read_verified", stored)
    return calls


@pytest.mark.parametrize("kind,message", [
    ("stored", "object exceeds max_object_bytes"),
    ("expanded", "parquet exceeds max_object_bytes when uncompressed"),
])
def test_target_byte_limit_refuses_real_oversized_objects_before_decode(tmp_path, monkeypatch, kind, message):
    assert MAX_OUTCOME_OBJECT_BYTES == LIMIT
    with source(tmp_path, kind) as case:
        size = case.record.object_ref.byte_size
        assert case.expanded > LIMIT
        assert size > LIMIT if kind == "stored" else size < LIMIT
        calls, changes = observe(monkeypatch, case), case.conn.total_changes
        with pytest.raises(DataError) as caught:
            load(case)
        problem = caught.value.problem
        assert (problem.code, problem.message, problem.category, problem.retryable, problem.stage, problem.details) == (
            "RESULT_LIMIT_EXCEEDED", message, "resource", False, None, {})
        assert calls == {"caps": [LIMIT], "bounded": [] if kind == "stored" else [LIMIT],
            "stored": [] if kind == "stored" else [size], "path": [], "decoded": 0}
        counters = case.repository.read_counters.snapshot()["computed_moves"]
        assert counters["scans"] == 1 and counters["rows_decoded"] == counters["rows_returned"] == 0
        assert case.conn.total_changes == changes


def test_under_cap_target_values_and_receipt_match_uncapped_transport(tmp_path, monkeypatch):
    with source(tmp_path, "small") as case:
        assert max(case.record.object_ref.byte_size, case.expanded) < LIMIT
        scan = case.repository.scan
        # Reference the prior transport behavior on the identical pinned bytes,
        # not a replacement target reader or manufactured receipt.
        with monkeypatch.context() as prior:
            def uncapped(query, *, table_name, max_object_bytes=None):
                return scan(query, table_name=table_name, max_object_bytes=None if table_name == "computed_moves" else max_object_bytes)
            prior.setattr(case.repository, "scan", uncapped)
            expected = load(case)
        calls, changes = observe(monkeypatch, case), case.conn.total_changes
        actual = load(case)
        pd.testing.assert_frame_equal(actual, expected)
        assert actual.attrs == expected.attrs and actual.attrs["inputs_id"] == expected.attrs["inputs_id"]
        assert actual.event_id.tolist() == [case.event["event_id"]]
        assert actual.realized_move_pct.tolist() == [-3.] and actual.absolute_move_pct.tolist() == [3.]
        assert actual.target_available_on.tolist() == ["2024-05-08"]
        ref = case.record.object_ref
        assert actual.attrs["consumed_inputs"] == [{"event_id": case.event["event_id"],
            "fragment_id": case.record.fragment_id, "object_id": ref.object_id, "content_hash": ref.content_hash}]
        assert calls == {"caps": [LIMIT], "bounded": [LIMIT], "stored": [ref.byte_size], "path": [], "decoded": 1}
        assert case.conn.total_changes == changes
