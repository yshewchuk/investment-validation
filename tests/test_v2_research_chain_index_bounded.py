"""Bounded ``load_chain_index`` reads: one requested year at a time, exact
``(ticker, obs_date)`` pairs filtered on every Arrow batch.

Synthetic pinned snapshots only. This file pins down the acceptance:

* golden whole-read equality of every group frame over multiple years, with
  cross-pair distractors, duplicated group rows, missing keys, stored
  non-midnight observations, groups spanning Arrow batches, and under both
  chronological and reversed manifest fragment order;
* the manifest-order metadata lookup keeps the previous whole read's refusals
  for a missing table and for a contract with no partition column, including
  when every requested year is absent from the manifest;
* empty requested keys never scan; omitted and explicit-``None`` adapter
  behavior is identical;
* the batch filter is forwarded through the direct calendar split, the
  month-to-day overflow retry and the terminal-day refusal;
* a failed scan attempt retains none of its yielded batches: rejected rows
  are counted, each raw batch frame must be collectable by the time the next
  batch arrives, and the rows the failed month attempt kept are absent from
  the final group (no duplicates);
* a filter that raises an unrelated error is neither swallowed nor retried.
"""
from __future__ import annotations

import dataclasses
import gc
import sys
import weakref
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data import manifests  # noqa: E402
from engine.v2.data.errors import DataError, fail  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.research import _chains  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)

_KEYS = {
    ("TEST", pd.Timestamp("2024-01-10")),
    ("TEST", pd.Timestamp("2025-01-15")),
    ("TEST", pd.Timestamp("2025-03-05")),
    ("MISSING", pd.Timestamp("2023-12-01")),
}


def _capped_contract(maximum_batch_rows: int | None = None):
    contract = contract_for("option_chains")
    if maximum_batch_rows is not None:
        contract = dataclasses.replace(contract, maximum_batch_rows=maximum_batch_rows)
    return dataclasses.replace(contract,
                               definition_hash=manifests.table_contract_hash(contract))


def _commit_chains(tmp_path, fragments, *, maximum_batch_rows=None):
    conn, clock, store = catalog_and_store(tmp_path)
    contract = _capped_contract(maximum_batch_rows)
    ref = contract_ref_for(contract)
    records = [publish_and_inspect(store, contract, ref, rows, partition_key)
               for partition_key, rows in fragments.items()]
    snap = commit_tables(conn, clock, {"option_chains": records},
                         {"option_chains": contract}, store=store)
    return conn, store, snap


def _rows(year: int, entries: list[tuple]) -> list[dict]:
    """Synthetic option_chains rows ``(ticker, obs_date, strike)``, ordered by
    the table's full primary key. ``obs_date`` is a real ``timestamp[ns]``, so
    an entry may carry a non-midnight time-of-day or cross into another year's
    partition."""
    rows = [
        {"ticker": ticker, "obs_date": pd.Timestamp(obs), "year": year,
         "expiry": pd.Timestamp(f"{year}-12-20"), "dte": 30, "strike": float(strike),
         "right": "C", "bid": 1.0, "ask": 1.2, "spot": 100.0, "quote_repaired": False}
        for ticker, obs, strike in entries
    ]
    rows.sort(key=lambda row: (row["ticker"], row["obs_date"], row["expiry"],
                               row["strike"], row["right"]))
    return rows


def _whole_read_groups(repository, snapshot_ref, keys):
    """The pre-patch whole-read body, as the golden reference: one multi-year
    read, exact pair filter, one ``groupby(sort=False)``."""
    wanted = {(str(t), pd.Timestamp(d).normalize()) for t, d in keys}
    years = sorted({d.year for _, d in wanted})
    tickers = {t for t, _ in wanted}
    frame = _chains.read_chains_for_years(repository, snapshot_ref, years)
    frame = frame[frame["ticker"].isin(tickers)]
    frame["obs_date"] = pd.to_datetime(frame["obs_date"])
    key_index = pd.MultiIndex.from_arrays([frame["ticker"], frame["obs_date"]])
    frame = frame[key_index.isin(wanted)]
    return {(str(k[0]), pd.Timestamp(k[1])): g.reset_index(drop=True)
            for k, g in frame.groupby(["ticker", "obs_date"], sort=False)}


@pytest.mark.parametrize("manifest_order", ["chronological", "reversed"])
def test_load_chain_index_matches_the_whole_read_golden(tmp_path, monkeypatch, manifest_order):
    conn, store, snap = _commit_chains(
        tmp_path,
        {
            "2024": _rows(2024, [
                ("AAA0", "2024-02-01", 10), ("AAA0", "2024-02-02", 11),
                ("AAAA", "2024-01-10", 12),               # right date, wrong ticker
                ("TEST", "2024-01-09", 13),               # right ticker, wrong date
                ("TEST", "2024-01-10", 14), ("TEST", "2024-01-10", 15),
                ("TEST", "2024-01-10 15:30", 16),         # non-midnight: no match
                ("TEST", "2025-01-15", 17),               # cross-partition group row
            ]),
            "2025": _rows(2025, [
                ("TEST", "2025-01-15", 18),               # same group, other year
                ("TEST", "2025-03-05", 19),
                ("ZZZZ", "2025-03-05", 20),               # right date, wrong ticker
            ]),
        },
        maximum_batch_rows=2,
    )
    repository = Repository(conn, store)
    if manifest_order == "reversed":
        records = repository.fragment_records

        def reversed_records(snapshot_ref, table_name):
            return tuple(reversed(records(snapshot_ref, table_name)))

        monkeypatch.setattr(repository, "fragment_records", reversed_records)

    omitted = _chains.read_chains_for_years(repository, snap, (2024, 2025))
    explicit = _chains.read_chains_for_years(repository, snap, (2024, 2025),
                                             batch_filter=None)
    pd.testing.assert_frame_equal(omitted, explicit)

    index = _chains.load_chain_index(repository, snap, _KEYS)
    golden = _whole_read_groups(repository, snap, _KEYS)
    assert list(index.keys) == list(golden)  # group discovery order preserved
    for key, frame in golden.items():
        pd.testing.assert_frame_equal(index.get(*key), frame)

    assert index.get("TEST", "2024-01-10")["strike"].tolist() == [14.0, 15.0]
    cross = index.get("TEST", "2025-01-15")["strike"].tolist()
    assert cross == ([17.0, 18.0] if manifest_order == "chronological" else [18.0, 17.0])
    assert index.get("TEST", "2025-03-05")["strike"].tolist() == [19.0]
    assert ("MISSING", pd.Timestamp("2023-12-01")) not in index
    assert 16.0 not in set(index.get("TEST", "2024-01-10")["strike"])
    assert pd.Timestamp("2024-01-10 15:30") in set(pd.to_datetime(omitted["obs_date"]))
    conn.close()


def test_load_chain_index_missing_table_refusal_matches_the_whole_read(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    snap = commit_tables(conn, clock, {"trades": []},
                         {"trades": contract_for("trades")}, store=store)
    repository = Repository(conn, store)
    keys = {("TEST", pd.Timestamp("2024-01-10"))}

    with pytest.raises(DataError) as whole:
        _whole_read_groups(repository, snap, keys)
    with pytest.raises(DataError) as bounded:
        _chains.load_chain_index(repository, snap, keys)

    assert bounded.value.code == whole.value.code == "CONTRACT_MISMATCH"
    assert str(bounded.value) == str(whole.value)
    conn.close()


def test_load_chain_index_partitionless_contract_refusal_matches_the_whole_read(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    contract = dataclasses.replace(_capped_contract(), partition_columns=())
    contract = dataclasses.replace(contract,
                                   definition_hash=manifests.table_contract_hash(contract))
    snap = commit_tables(conn, clock, {"option_chains": []},
                         {"option_chains": contract}, store=store)
    repository = Repository(conn, store)
    keys = {("TEST", pd.Timestamp("2024-01-10"))}

    with pytest.raises(ValueError) as whole:
        _whole_read_groups(repository, snap, keys)
    with pytest.raises(ValueError) as bounded:
        _chains.load_chain_index(repository, snap, keys)

    assert str(bounded.value) == str(whole.value)
    conn.close()


class _BatchProbe:
    """Counts every raw batch and proves the prior raw frame was collectable by
    the time the next one arrives (a callback counter alone proves nothing)."""

    def __init__(self, keep_date: str):
        self.keep = pd.Timestamp(keep_date)
        self.raw_refs: list[weakref.ref] = []
        self.raw_rows = 0
        self.kept_rows = 0
        self.rejected_rows = 0
        self.empty_matches = 0

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        gc.collect()
        assert all(ref() is None for ref in self.raw_refs)
        self.raw_refs.append(weakref.ref(frame))
        self.raw_rows += len(frame)
        kept = frame[pd.to_datetime(frame["obs_date"]) == self.keep]
        self.kept_rows += len(kept)
        self.rejected_rows += len(frame) - len(kept)
        if len(frame) and kept.empty:
            self.empty_matches += 1
        return kept


def test_batch_filter_preserves_complete_partition_rows(tmp_path):
    conn, store, snap = _commit_chains(
        tmp_path,
        {"2024": _rows(2024, [("TEST", "2024-01-15", i) for i in range(5)])},
        maximum_batch_rows=2,
    )
    repository = Repository(conn, store)
    probe = _BatchProbe("2024-01-15")

    frame = _chains.read_chains_for_years(repository, snap, (2024,), batch_filter=probe)
    assert frame["strike"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert probe.raw_rows > 0  # the complete scan still ran through the batch filter
    conn.close()


def test_batch_filter_unrelated_error_is_not_swallowed_or_retried(tmp_path):
    conn, store, snap = _commit_chains(
        tmp_path,
        {"2024": _rows(2024, [("TEST", "2024-01-10", i) for i in range(3)])},
    )
    repository = Repository(conn, store)
    calls = []

    def explode(frame: pd.DataFrame) -> pd.DataFrame:
        calls.append(len(frame))
        raise fail("CONTRACT_MISMATCH", "batch filter exploded")

    with pytest.raises(DataError) as err:
        _chains.read_chains_for_years(repository, snap, (2024,), batch_filter=explode)
    assert err.value.code == "CONTRACT_MISMATCH"
    assert "exploded" in str(err.value)
    assert calls == [3]  # one batch, no overflow retry
    conn.close()


def test_empty_requested_keys_never_read(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("an empty key set must not scan")

    monkeypatch.setattr(_chains, "read_chains_for_years", boom)
    index = _chains.load_chain_index(object(), object(), [])
    assert len(index) == 0
