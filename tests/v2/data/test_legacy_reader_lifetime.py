"""Real legacy reads must not retain Python file handles in Arrow workers."""
from __future__ import annotations

import gzip

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from engine.data import store as legacy_store
from engine.v2.data import legacy_adapter
from engine.v2.data.repository import Repository
from tests.test_v2_data_legacy_materialization import (
    _build_request,
    _build_snapshot,
    _snapshot_object_ref,
)


def _frame(empty=False):
    frame = pd.DataFrame({
        "text": pd.Series(["a", None, "c"], dtype="str"),
        "value": pd.Series([1, None, 3], dtype="Int64"),
        "year": [2024, 2024, 2024],
    })
    frame.index = pd.Index([10, 20, 30], name="row")
    frame.attrs = {"source": "synthetic"}
    return frame.iloc[:0] if empty else frame


def _native_reads(monkeypatch):
    """Reject the unsafe boundary, while forwarding every read to real Arrow."""
    actual = pq.read_table
    handles = []

    def read(source, *args, **kwargs):
        assert isinstance(source, pa.OSFile), "Python-owned Parquet handle"
        assert not source.closed
        handles.append(source)
        return actual(source, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", read)
    return handles


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("columns", [None, [], ["year", "missing", "text"], ["missing"]])
def test_reader_keeps_legacy_values_metadata_and_projection(tmp_path, monkeypatch, columns, empty):
    path = tmp_path / "year=2024" / "part.parquet"
    path.parent.mkdir()
    expected = _frame(empty)
    expected.to_parquet(path)
    if columns:
        expected = expected.reindex(columns=columns)
    before = path.read_bytes()
    handles = _native_reads(monkeypatch)

    actual = legacy_adapter.read_legacy_part(path, columns)

    pd.testing.assert_frame_equal(actual, expected)
    assert actual.attrs == expected.attrs
    assert path.read_bytes() == before
    assert handles and all(source.closed for source in handles)


def test_materialization_uses_native_handles_for_actual_legacy_validation(tmp_path, monkeypatch):
    conn, store, snapshot = _build_snapshot(tmp_path)
    try:
        repository = Repository(conn, store)
        request = _build_request(repository, snapshot, _snapshot_object_ref(store), store)
        handles = _native_reads(monkeypatch)

        manifest = legacy_adapter.materialize(repository, store, request, tmp_path / "materialized")

        assert manifest
        assert handles and all(source.closed for source in handles)
    finally:
        conn.close()


def test_reader_closes_native_handle_on_parquet_error(tmp_path, monkeypatch):
    path = tmp_path / "broken.parquet"
    path.write_bytes(b"not parquet")
    handles = _native_reads(monkeypatch)
    with pytest.raises(pa.ArrowInvalid):
        legacy_adapter.read_legacy_part(path, None)
    assert len(handles) == 1 and handles[0].closed
    assert path.read_bytes() == b"not parquet"


def test_reader_missing_part_remains_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        legacy_adapter.read_legacy_part(tmp_path / "missing.parquet", None)


def test_reader_keeps_legacy_gzip_fallback(tmp_path, monkeypatch):
    path = tmp_path / "part.csv.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("value,text\n1,a\n2,b\n")
    monkeypatch.setattr(legacy_store, "HAVE_PARQUET", False)
    expected = legacy_store._read_part(path, ["text", "missing", "value"])
    actual = legacy_adapter.read_legacy_part(path, ["text", "missing", "value"])
    pd.testing.assert_frame_equal(actual, expected)
