"""Independent byte/expansion bounds over real immutable synthetic Parquet."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from engine.v2.contracts.data import DataQuery, KeyPredicate
from engine.v2.data import manifests, objects
from engine.v2.data.errors import DataError
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT as CONTRACT
from engine.v2.data.repository import Repository
from engine.v2.foundation import artifacts
from tests.data_scan_support import (
    IMPORT_REQUEST_HASH,
    RECEIPT,
    catalog_and_store,
    commit_tables,
    contract_ref_for,
    publish_bytes,
    table_from_rows,
)


@contextmanager
def fixture(tmp_path, *, groups=1, payload=20, partitions=1):
    conn, clock, store = catalog_and_store(tmp_path)
    ref, records, expected, expanded = contract_ref_for(CONTRACT), [], [], []
    for partition in range(partitions):
        rows = [dict(ticker=f"S{partition}", event_date=f"2024-01-{i + 1:02}", session="AMC",
            skipped=False, source_id="x" * payload * (partition + 1), input_hash="sha256:" + "a" * 64)
            for i in range(groups)]
        sink = pa.BufferOutputStream()
        pq.write_table(table_from_rows(CONTRACT, rows), sink, row_group_size=1,
                       compression="zstd", use_dictionary=False, write_statistics=False)
        data = sink.getvalue().to_pybytes()
        metadata = pq.read_metadata(pa.BufferReader(data))
        expanded.append([metadata.row_group(i).total_byte_size for i in range(groups)])
        obj = publish_bytes(store, data)
        inspected = objects.inspect_fragment(store, obj, CONTRACT, ref, f"S{partition}")
        records.append(manifests.fragment_record(inspected, ref, input_receipt_refs=(RECEIPT,),
                                                 import_request_hash=IMPORT_REQUEST_HASH))
        expected.extend({"ticker": row["ticker"], "event_date": row["event_date"]} for row in rows)
    snapshot = commit_tables(conn, clock, {CONTRACT.table_name: records}, {CONTRACT.table_name: CONTRACT})
    query = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=ref,
        columns=("ticker", "event_date"), order_by=CONTRACT.primary_key,
        key_filter=(KeyPredicate(column="ticker", operator="in", values=tuple(f"S{i}" for i in range(partitions))),),
        max_batch_rows=1, max_result_rows=groups * partitions)
    try:
        yield Repository(conn, store), query, records, expected, expanded, store
    finally:
        conn.close()


def scan(repository, query, cap):
    return repository.scan(query, table_name=CONTRACT.table_name, max_object_bytes=cap)


def problem(caught, code, message, category):
    p = caught.value.problem
    assert (p.code, p.message, p.category, p.retryable, p.stage, p.details) == (
        code, message, category, False, None, {})


@pytest.mark.parametrize("cap", [True, False, 0, -1, 1.5, "1024"])
def test_invalid_bound_refuses_before_store_or_catalog_access(cap):
    with pytest.raises(DataError) as caught:
        list(scan(Repository(None), None, cap))
    problem(caught, "QUERY_NOT_BOUNDED", "max_object_bytes must be a positive integer", "validation")


@pytest.mark.parametrize("cap", [None, True, False, 0, -1, 1.5, "1024"])
def test_public_object_reader_rejects_invalid_caps_before_any_object_access(cap):
    with pytest.raises(DataError) as caught:
        objects.read_verified_bytes(None, None, max_bytes=cap)
    problem(caught, "QUERY_NOT_BOUNDED", "max_object_bytes must be a positive integer", "validation")


def test_public_object_reader_rejects_oversize_without_relying_on_repository_guard():
    with pytest.raises(DataError) as caught:
        objects.read_verified_bytes(None, SimpleNamespace(byte_size=1025), max_bytes=1024)
    problem(caught, "RESULT_LIMIT_EXCEEDED", "object exceeds max_object_bytes", "resource")


@pytest.mark.parametrize("cap", [2**31 - 1, 2**31, 2**63, 10**100])
def test_huge_positive_caps_keep_valid_scans_without_arrow_integer_overflow(tmp_path, monkeypatch, cap):
    with fixture(tmp_path) as (repo, query, _, expected, _, _):
        calls, original = [], repo._open_parquet
        def opened(source, **limits):
            calls.append(limits)
            return original(source, **limits)
        monkeypatch.setattr(repo, "_open_parquet", opened)
        assert [r for b in scan(repo, query, cap) for r in b.to_pylist()] == expected
        assert calls == [{"thrift_string_size_limit": 2147483647, "thrift_container_size_limit": 2147483647}]


def test_exact_file_bound_and_default_keep_identical_rows(tmp_path, monkeypatch):
    with fixture(tmp_path) as (repo, query, records, expected, _, store):
        assert [r for b in scan(repo, query, None) for r in b.to_pylist()] == expected
        def forbidden(*args, **kwargs):
            pytest.fail("bounded scan used path verification/cache")
        monkeypatch.setattr(objects, "verify_object_path", forbidden)
        monkeypatch.setattr(store, "verify", forbidden)
        cap = records[0].object_ref.byte_size
        assert [r for b in scan(repo, query, cap) for r in b.to_pylist()] == expected
        with pytest.raises(DataError) as caught:
            list(scan(repo, query, cap - 1))
        problem(caught, "RESULT_LIMIT_EXCEEDED", "object exceeds max_object_bytes", "resource")


def test_late_oversized_candidate_refuses_before_first_read(tmp_path, monkeypatch):
    with fixture(tmp_path, partitions=2) as (repo, query, records, _, _, store):
        large = replace(records[1], object_ref=replace(records[1].object_ref, byte_size=100001))
        monkeypatch.setattr(repo, "_table_records", lambda *args: (CONTRACT, [records[0], large]))
        monkeypatch.setattr(store, "read_verified", lambda *args: pytest.fail("early object opened"))
        with pytest.raises(DataError) as caught:
            list(scan(repo, query, 100000))
        problem(caught, "RESULT_LIMIT_EXCEEDED", "object exceeds max_object_bytes", "resource")


def test_total_uncompressed_bound_includes_all_groups_before_decode(tmp_path, monkeypatch):
    with fixture(tmp_path, groups=3, payload=40000) as (repo, query, records, expected, expanded, _):
        cap = sum(expanded[0])
        assert records[0].object_ref.byte_size < cap and max(expanded[0]) < cap - 1
        assert [r for b in scan(repo, query, cap) for r in b.to_pylist()] == expected
        monkeypatch.setattr(repo, "_iter_batches", lambda *args: pytest.fail("oversized data decoded"))
        with pytest.raises(DataError) as caught:
            list(scan(repo, query, cap - 1))
        problem(caught, "RESULT_LIMIT_EXCEEDED", "parquet exceeds max_object_bytes when uncompressed", "resource")


@pytest.mark.parametrize("growth", [False, True])
def test_actual_bytes_are_bounded_and_rehashed_even_after_cached_verification(tmp_path, monkeypatch, growth):
    with fixture(tmp_path) as (repo, query, records, _, _, _):
        ref = records[0].object_ref
        path = objects.verify_object_path(repo._store, ref)
        original = path.read_bytes()
        path.chmod(0o644)
        path.write_bytes(original + b"x" * 100000 if growth else b"x" + original[1:])
        requests, real_read = [], artifacts.os.read
        def read(fd, size):
            requests.append(size)
            return real_read(fd, size)
        monkeypatch.setattr(artifacts.os, "read", read)
        monkeypatch.setattr(repo, "_open_parquet", lambda *args, **kwargs: pytest.fail("corrupt bytes parsed"))
        with pytest.raises(DataError) as caught:
            list(scan(repo, query, ref.byte_size))
        problem(caught, "OBJECT_CORRUPT", "published object bytes do not match its recorded hash", "integrity")
        expected = [ref.byte_size + 1] if growth else [ref.byte_size + 1, 1]
        assert requests == expected


@pytest.mark.parametrize("failure", [False, True])
def test_buffers_close_on_early_cancellation_and_decode_failure(tmp_path, monkeypatch, failure):
    with fixture(tmp_path, groups=2) as (repo, query, records, _, _, _):
        sources, files, original = [], [], repo._open_parquet
        def opened(source, **limits):
            sources.append(source)
            files.append(original(source, **limits))
            return files[-1]
        monkeypatch.setattr(repo, "_open_parquet", opened)
        if failure:
            def broken(*args):
                raise RuntimeError("decode sentinel")
            monkeypatch.setattr(repo, "_iter_batches", broken)
        stream = scan(repo, query, records[0].object_ref.byte_size)
        if failure:
            with pytest.raises(RuntimeError, match="^decode sentinel$"):
                next(stream)
        else:
            next(stream)
            stream.close()
        assert len(sources) == len(files) == 1
        assert sources[0].closed and files[0].closed
