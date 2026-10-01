"""Tier-0 tests for ``experiments.common_v2.load_v2_chain_quotes``: the read
forwards the RESOLVED snapshot ref (never the raw ``snapshot_id`` string) and
a real ``set`` of keys into ``load_chain_index``, and closes its catalog
connection even when the read raises. Fixture machinery is reused from the
existing ``common_v2`` tests (``tests/data_scan_support.catalog_and_store``
plus ``tests/test_v2_research_replay``'s ``_commit``/``_chain_rows``/
``_event_rows``); the two read-path edges are spied in ``common_v2``'s own
namespace, where the loader calls them."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.research._chains import ChainIndex, load_chain_index  # noqa: E402
from experiments import common_v2  # noqa: E402
from tests.data_scan_support import catalog_and_store  # noqa: E402
from tests.test_v2_research_replay import (  # noqa: E402
    _chain_rows,
    _commit,
    _event_rows,
)

KEYS = [("TEST", pd.Timestamp("2024-05-02")), ("TEST", pd.Timestamp("2024-05-03"))]


def _fixture(tmp_path, monkeypatch):
    """A real committed snapshot plus spies on resolve/load/open_catalog.

    Returns ``(conn, snapshot_id, calls)`` with ``calls`` recording every
    ``Repository.resolve`` argument/return, every ``load_chain_index``
    ``(snapshot_ref, keys, result)`` pass-through, and every connection
    ``open_catalog`` handed to the loader.
    """
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit(conn, clock, store, chain_rows=_chain_rows(),
                       event_rows=_event_rows(), receipt_id="r1")
    calls: dict = {"resolved": [], "loaded": [], "opened": []}

    class SpyRepository(common_v2.Repository):
        def resolve(self, snapshot_id):
            ref = super().resolve(snapshot_id)
            calls["resolved"].append((snapshot_id, ref))
            return ref

    def spy_open(path, **kwargs):
        opened = real_open(path, **kwargs)
        calls["opened"].append(opened)
        return opened

    def spy_load(repository, snapshot_ref, keys):
        result = load_chain_index(repository, snapshot_ref, keys)
        calls["loaded"].append((snapshot_ref, keys, result))
        return result

    real_open = common_v2.open_catalog
    monkeypatch.setattr(common_v2, "Repository", SpyRepository)
    monkeypatch.setattr(common_v2, "load_chain_index", spy_load)
    monkeypatch.setattr(common_v2, "open_catalog", spy_open)
    return conn, snapshot.snapshot_id, calls


def _read(snapshot_id, tmp_path):
    return common_v2.load_v2_chain_quotes(
        list(KEYS), catalog=tmp_path / "catalog.sqlite",
        store_root=tmp_path / "store", snapshot_id=snapshot_id)


def test_load_v2_chain_quotes_forwards_resolved_ref_and_key_set(tmp_path, monkeypatch):
    conn, snapshot_id, calls = _fixture(tmp_path, monkeypatch)
    index = _read(snapshot_id, tmp_path)
    conn.close()

    assert len(calls["loaded"]) == 1
    snapshot_ref, keys, result = calls["loaded"][0]
    assert not isinstance(snapshot_ref, str)
    assert isinstance(keys, set) and keys == set(KEYS)
    assert index is result
    assert isinstance(index, ChainIndex) and len(index) == 2


def test_load_v2_chain_quotes_closes_connection_when_read_raises(tmp_path, monkeypatch):
    conn, snapshot_id, calls = _fixture(tmp_path, monkeypatch)

    def boom(repository, snapshot_ref, keys):
        raise RuntimeError("chain read failed")

    monkeypatch.setattr(common_v2, "load_chain_index", boom)
    with pytest.raises(RuntimeError, match="chain read failed"):
        _read(snapshot_id, tmp_path)
    conn.close()

    assert len(calls["opened"]) == 1
    loader_conn = calls["opened"][0]
    with pytest.raises(sqlite3.ProgrammingError):
        loader_conn.execute("SELECT 1")


def test_load_v2_chain_quotes_forwards_resolve_return_not_the_raw_id(tmp_path, monkeypatch):
    conn, snapshot_id, calls = _fixture(tmp_path, monkeypatch)
    _read(snapshot_id, tmp_path)
    conn.close()

    assert len(calls["loaded"]) == 1
    # ``read_table`` re-resolves internally, so the helper's own pin is the
    # FIRST recorded call; every resolve must see the pinned id, never a scope.
    assert calls["resolved"]
    resolved_arg, resolved_ref = calls["resolved"][0]
    assert resolved_arg == snapshot_id
    assert all(arg == snapshot_id for arg, _ref in calls["resolved"])
    snapshot_ref, _keys, _result = calls["loaded"][0]
    assert snapshot_ref is resolved_ref
    assert snapshot_ref != snapshot_id
    assert snapshot_ref.snapshot_id == snapshot_id
