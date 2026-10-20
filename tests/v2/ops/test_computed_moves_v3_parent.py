"""A computed_moves generation over a v3 parent: refuse incremental, rebuild fully.

Rebuild over migration (#651): v3 partitions are never carried or migrated. The v3 parent is
produced by the real refresh path under a v3-shaped contract; the v4 runs are unpatched.
"""
from __future__ import annotations

import dataclasses

import pandas as pd
import pyarrow as pa
import pytest

from engine.v2.contracts.data import ColumnContract, TableContractRef
from engine.v2.data import manifests
from engine.v2.ops.errors import OpsError
from engine.v2.data.errors import fail
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore
from engine.v2.ops import computed_moves_store as store_module
from tests.ops_support import catalog
from tests.v2.ops.test_v2_ops_computed_moves_store import (
    _AS_OF, _EVENT_DAYS, _CountingFetcher, _build_parent, _closes_csv, _event_row, _head_row,
    _parameters, _write_input)

_NEW = ("event_ordinal", "admitted", "mean_prior_or_implied", "n_prior_implied")
_TICKERS = ("AAAA", "BBBB", "CCCC")


@pytest.fixture(autouse=True)
def _small_fixture_bar(monkeypatch):
    """The fixtures carry 8-ish events per ticker: lower the store's one bar to the old
    effective admission bar (5) so they classify as computed outcomes."""
    from engine.v2.ops import computed_moves_store as _store
    monkeypatch.setattr(_store, "MIN_SCOREABLE", 5)


def _v3_contract():
    v4 = store_module.COMPUTED_MOVES_CONTRACT
    columns = [c for c in v4.columns if c.name not in _NEW]
    at = [c.name for c in columns].index("implied_move_pct") + 1
    columns.insert(at, ColumnContract(name="quarter_ordinal", physical_type="int64",
                                      nullable=False))
    draft = dataclasses.replace(v4, contract_id="computed_moves.v3", semantic_version="3.0.0",
                                columns=tuple(columns), definition_hash="sha256:" + "0" * 64)
    return dataclasses.replace(draft, definition_hash=manifests.table_contract_hash(draft))


def _patch_v3(monkeypatch):
    v3 = _v3_contract()
    fields = [f for f in store_module._ARROW_SCHEMA if f.name not in _NEW]
    fields.insert([f.name for f in fields].index("implied_move_pct") + 1,
                  pa.field("quarter_ordinal", pa.int64()))
    names = [c for c in store_module.FRAGMENT_COLUMNS if c not in _NEW]
    names.insert(names.index("implied_move_pct") + 1, "quarter_ordinal")
    real_build_rows = store_module.build_event_rows
    monkeypatch.setattr(store_module, "COMPUTED_MOVES_CONTRACT", v3)
    monkeypatch.setattr(store_module, "_CONTRACT_REF", TableContractRef(
        contract_id=v3.contract_id, definition_hash=v3.definition_hash))
    monkeypatch.setattr(store_module, "_ARROW_SCHEMA", pa.schema(fields))
    monkeypatch.setattr(store_module, "FRAGMENT_COLUMNS", tuple(names))
    monkeypatch.setattr(store_module, "build_event_rows", lambda *a, **k: [
        {**row, "quarter_ordinal": 0} for row in real_build_rows(*a, **k)])


def _run(conn, tmp_path, name, head, ids, monkeypatch, fetcher=None):
    monkeypatch.setattr(store_module, "target_tickers_from_snapshot",
                        lambda *a, **k: (list(ids), {}))
    root = tmp_path / name
    _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
    parameters = _parameters(head, expected_ids=tuple(ids), catalog_path=tmp_path / "ops.sqlite",
                             objects_root=tmp_path)
    return store_module.run_computed_moves_refresh(
        parameters, root, as_of=_AS_OF, fetcher=fetcher or _CountingFetcher(_closes_csv()))


def _records(conn, store, snapshot_id):
    resolved = Repository(conn, store).resolve_full(snapshot_id)
    return sorted((r.table_contract_ref.contract_id, r.partition_key, r.object_ref.object_id)
                  for r in resolved.records
                  if r.table_contract_ref.contract_id.startswith("computed_moves."))


def _write_counts(conn):
    """Rows a generation would write: snapshots, heads, captures (raw receipts may remain)."""
    return tuple(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                 for table in ("data_snapshots", "data_snapshot_heads",
                               "data_computed_moves_captures"))


def _v3_head(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    events = [_event_row(t, d) for t in _TICKERS for d in _EVENT_DAYS]
    head = _build_parent(conn, clock, store, events_rows=events)
    with monkeypatch.context() as patch:
        _patch_v3(patch)
        assert _run(conn, tmp_path, "v3run", head, _TICKERS, patch).status == "complete"
    v3_head = _head_row(conn)
    records = _records(conn, store, v3_head["snapshot_id"])
    assert {contract for contract, _, _ in records} == {"computed_moves.v3"}
    assert {ticker for _, ticker, _ in records} == set(_TICKERS)
    return conn, store, v3_head


def test_incremental_generation_over_a_v3_parent_refuses_without_writing(tmp_path, monkeypatch):
    conn, store, v3_head = _v3_head(tmp_path, monkeypatch)
    written = _write_counts(conn)
    with pytest.raises(OpsError) as err:
        _run(conn, tmp_path, "v4partial", v3_head, ("AAAA",), monkeypatch)
    assert _write_counts(conn) == written
    assert "VALIDATION_FAILED" in str(err.value)
    assert "full rebuild required" in str(err.value)
    after = _head_row(conn)
    assert (after["snapshot_id"], after["generation"]) == (
        v3_head["snapshot_id"], v3_head["generation"])


def test_full_rebuild_over_a_v3_head_writes_complete_v5(tmp_path, monkeypatch):
    conn, store, v3_head = _v3_head(tmp_path, monkeypatch)
    assert _run(conn, tmp_path, "v4full", v3_head, _TICKERS, monkeypatch).status == "complete"
    new_head = _head_row(conn)
    assert new_head["snapshot_id"] != v3_head["snapshot_id"]
    records = _records(conn, store, new_head["snapshot_id"])
    assert {contract for contract, _, _ in records} == {"computed_moves.v5"}
    assert {ticker for _, ticker, _ in records} == set(_TICKERS)


def test_all_v4_incremental_generation_carries_unselected_tickers(tmp_path, monkeypatch):
    conn, store, v3_head = _v3_head(tmp_path, monkeypatch)
    _run(conn, tmp_path, "v4full", v3_head, _TICKERS, monkeypatch)
    v4_head = _head_row(conn)
    before = [r for r in _records(conn, store, v4_head["snapshot_id"]) if r[1] != "AAAA"]
    _run(conn, tmp_path, "v4one", v4_head, ("AAAA",), monkeypatch)
    after_head = _head_row(conn)
    after = _records(conn, store, after_head["snapshot_id"])
    assert [r for r in after if r[1] != "AAAA"] == before
    assert {ticker for _, ticker, _ in after} == set(_TICKERS)



def test_full_rebuild_where_a_v3_ticker_has_no_v5_rows_drops_its_manifest_hash(
        tmp_path, monkeypatch):
    conn, store, v3_head = _v3_head(tmp_path, monkeypatch)
    before = Repository(conn, store).resolve_full(v3_head["snapshot_id"])
    assert "CCCC" in before.table_manifests["computed_moves"].partition_logical_hashes
    real_build_rows = store_module.build_event_rows
    # CCCC yields no v4 rows (the refresh records it as too_few); the rest is the real builder.
    monkeypatch.setattr(store_module, "build_event_rows", lambda ticker, *a, **k: (
        [] if ticker == "CCCC" else real_build_rows(ticker, *a, **k)))
    _run(conn, tmp_path, "v4full", v3_head, _TICKERS, monkeypatch)
    new_head = _head_row(conn)
    assert new_head["snapshot_id"] != v3_head["snapshot_id"]
    records = _records(conn, store, new_head["snapshot_id"])
    assert {contract for contract, _, _ in records} == {"computed_moves.v5"}
    assert {ticker for _, ticker, _ in records} == {"AAAA", "BBBB"}
    after = Repository(conn, store).resolve_full(new_head["snapshot_id"])
    hashes = after.table_manifests["computed_moves"].partition_logical_hashes
    assert set(hashes) == {"AAAA", "BBBB"}


def test_full_rebuild_producing_no_rows_over_a_v3_head_refuses_instead_of_leaving_v3_live(
        tmp_path, monkeypatch):
    conn, store, v3_head = _v3_head(tmp_path, monkeypatch)
    monkeypatch.setattr(store_module, "build_event_rows", lambda *a, **k: [])
    written = _write_counts(conn)
    with pytest.raises(OpsError) as err:
        _run(conn, tmp_path, "v4none", v3_head, _TICKERS, monkeypatch)
    assert "VALIDATION_FAILED" in str(err.value)
    assert "full rebuild required" in str(err.value)
    assert _write_counts(conn) == written
    after = _head_row(conn)
    assert (after["snapshot_id"], after["generation"]) == (
        v3_head["snapshot_id"], v3_head["generation"])


def test_build_rows_refusal_reaches_the_ops_layer_typed_without_writing(tmp_path, monkeypatch):
    """A DataError from build_event_rows (duplicate event dates, a daily frame without src_iv: those
    refusals are unit-tested on build_rows itself, and the fixture publisher cannot carry such
    a parent) must leave the refresh as an OpsError (INPUT_CHANGED, data code kept in the message and details): the CLI catches only
    OpsError. The raise is injected at build_event_rows; the conversion under test is the store's."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    events = [_event_row(t, d) for t in _TICKERS for d in _EVENT_DAYS]
    head = _build_parent(conn, clock, store, events_rows=events)

    def refuse(*args, **kwargs):
        raise fail("CONTRACT_MISMATCH", "event dates must be strictly ascending")

    monkeypatch.setattr(store_module, "build_event_rows", refuse)
    written = _write_counts(conn)
    with pytest.raises(OpsError) as err:
        _run(conn, tmp_path, "dup", head, _TICKERS, monkeypatch)
    assert "INPUT_CHANGED" in str(err.value)
    assert "CONTRACT_MISMATCH" in str(err.value)
    assert err.value.problem.details["data_code"] == "CONTRACT_MISMATCH"
    assert _write_counts(conn) == written


class _LateHistoryFetcher:
    """The normal closes, but the history starts at business day 6 (earlier events predate it)."""

    def __call__(self, ticker):
        lines = _closes_csv().decode().splitlines()
        return "\n".join(lines[:1] + lines[1 + 6:]).encode(), "complete", {}, None


def test_store_keeps_an_event_before_the_first_close_as_a_skipped_row_in_the_ordinal(
        tmp_path, monkeypatch):
    from engine.v2.contracts import DataQuery, KeyPredicate
    from engine.v2.data.computed_moves_table import COMPUTED_MOVES_TABLE_NAME

    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    bdays = pd.bdate_range("2024-01-02", "2024-01-31")
    events = [_event_row("AAAA", bdays[i]) for i in (1, 8, 10, 12, 14, 18, 20)]
    head = _build_parent(conn, clock, store, events_rows=events)
    result = _run(conn, tmp_path, "late", head, ("AAAA",), monkeypatch,
                  fetcher=_LateHistoryFetcher())
    assert result.status == "complete"

    repository = Repository(conn, store)
    snapshot = repository.resolve(_head_row(conn)["snapshot_id"])
    dvr = snapshot.table_versions[COMPUTED_MOVES_TABLE_NAME]
    key_filter = (KeyPredicate(column="ticker", operator="eq", values=("AAAA",)),)
    bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=COMPUTED_MOVES_TABLE_NAME,
        table_contract_ref=dvr.table_contract_ref, key_filter=key_filter, time_interval=None)
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=dvr.table_contract_ref,
        columns=("event_date", "event_ordinal", "admitted", "skipped", "realized_move_pct"),
        key_filter=key_filter, order_by=("ticker", "event_date"),
        max_batch_rows=min(100, bound), max_result_rows=min(100, bound))
    rows = [r for batch in repository.scan(query, table_name=COMPUTED_MOVES_TABLE_NAME)
            for r in batch.to_pylist()]
    assert [r["event_ordinal"] for r in rows] == list(range(7))
    assert [r["admitted"] for r in rows] == [False] * 4 + [True] * 3
    assert [r["skipped"] for r in rows] == [True] + [False] * 6
    assert rows[0]["realized_move_pct"] is None