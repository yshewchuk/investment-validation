"""Bounded ``load_chain_index`` reads push ticker/obs_date predicates into
shared manifest-bounded scans (issue #271). The exact-pair ``batch_filter``
continues to decide membership after decoding, so key-filter distractors never
appear in results.

Synthetic pinned snapshots exercise:
* sorted, unique ``in`` predicates and empty-input behavior;
* one scan per selected year, with results matching a complete-read golden
  population and excluding cross-pair distractors;
* strict primary-key ordering and complete selected-population reads.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts.data import KeyPredicate  # noqa: E402
from engine.v2.data import manifests  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.research import _chains  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)


def _batch_limited_contract(maximum_batch_rows: int | None = None):
    contract = contract_for("option_chains")
    if maximum_batch_rows is not None:
        contract = dataclasses.replace(contract, maximum_batch_rows=maximum_batch_rows)
    return dataclasses.replace(contract,
                               definition_hash=manifests.table_contract_hash(contract))


def _commit_chains(tmp_path, fragments, *, maximum_batch_rows=None):
    conn, clock, store = catalog_and_store(tmp_path)
    contract = _batch_limited_contract(maximum_batch_rows)
    ref = contract_ref_for(contract)
    records = [publish_and_inspect(store, contract, ref, rows, partition_key)
               for partition_key, rows in fragments.items()]
    snap = commit_tables(conn, clock, {"option_chains": records},
                         {"option_chains": contract}, store=store)
    return conn, store, snap


def _rows(year: int, entries: list[tuple]) -> list[dict]:
    """Synthetic option_chains rows ``(ticker, obs_date, strike)``, ordered by
    the table's full primary key."""
    rows = [
        {"ticker": ticker, "obs_date": pd.Timestamp(obs), "year": year,
         "expiry": pd.Timestamp(f"{year}-12-20"), "dte": 30, "strike": float(strike),
         "right": "C", "bid": 1.0, "ask": 1.2, "spot": 100.0, "quote_repaired": False}
        for ticker, obs, strike in entries
    ]
    rows.sort(key=lambda row: (row["ticker"], row["obs_date"], row["expiry"],
                               row["strike"], row["right"]))
    return rows


def test_chain_key_filter_builds_sorted_unique_predicates():
    tickers = {"ZEBRA", "TEST"}
    wanted = {("ZEBRA", pd.Timestamp("2024-09-01")),
              ("TEST", pd.Timestamp("2024-06-15"))}
    predicates = _chains._chain_key_filter(tickers, wanted)
    assert len(predicates) == 2
    by_column = {p.column: p for p in predicates}
    assert set(by_column) == {"ticker", "obs_date"}
    ticker_pred = by_column["ticker"]
    assert isinstance(ticker_pred, KeyPredicate)
    assert ticker_pred.operator == "in"
    assert ticker_pred.values == ("TEST", "ZEBRA")
    date_pred = by_column["obs_date"]
    assert isinstance(date_pred, KeyPredicate)
    assert date_pred.operator == "in"
    assert date_pred.values == ("2024-06-15", "2024-09-01")
    assert _chains._chain_key_filter(set(), set()) == ()
    assert _chains._chain_key_filter({"TEST"}, set()) == ()


def test_load_chain_index_narrow_filter_scans_once_and_matches_complete_read(
        tmp_path, monkeypatch):
    wanted = {("TEST", pd.Timestamp("2024-06-15")),
              ("ZEBRA", pd.Timestamp("2024-09-01"))}
    entries = [
        ("TEST", "2024-06-15", 10),   # wanted pair 1
        ("TEST", "2024-09-01", 11),   # cross: right ticker, the OTHER wanted date
        ("ZEBRA", "2024-06-15", 12),  # cross: right ticker, the OTHER wanted date
        ("ZEBRA", "2024-09-01", 13),  # wanted pair 2
        ("TEST", "2024-01-05", 14),   # right ticker, unwanted date
    ] + [(f"AAA{i}", f"2024-{i:02d}-10", 100 + i) for i in range(1, 13)]
    conn, store, snap = _commit_chains(
        tmp_path, {"2024": _rows(2024, entries)},
        maximum_batch_rows=2,
    )
    repository = Repository(conn, store)

    calls: list = []
    original_scan = repository.scan

    def counting_scan(query, *, table_name):
        calls.append((query, table_name))
        yield from original_scan(query, table_name=table_name)

    monkeypatch.setattr(repository, "scan", counting_scan)

    index = _chains.load_chain_index(repository, snap, wanted)
    narrow_calls = len(calls)
    assert narrow_calls == 1  # one partition, scanned once -- the acceptance criterion

    # The complete no-key_filter reference read reproduces _whole_read_groups'
    # exact-pair filter, so both sides must agree.
    calls.clear()
    tickers = {t for t, _ in wanted}
    ref = _chains.read_chains_for_years(
        repository, snap, (2024,),
        batch_filter=_chains._chain_batch_filter(tickers, wanted))
    ref = ref[ref["ticker"].isin(tickers)]
    ref["obs_date"] = pd.to_datetime(ref["obs_date"])
    key_index = pd.MultiIndex.from_arrays([ref["ticker"], ref["obs_date"]])
    ref = ref[key_index.isin(wanted)]
    golden = {(str(k[0]), pd.Timestamp(k[1])): g.reset_index(drop=True)
              for k, g in ref.groupby(["ticker", "obs_date"], sort=False)}
    assert len(calls) == 1  # the complete pinned partition is read once

    test_group = index.get("TEST", pd.Timestamp("2024-06-15"))
    zebra_group = index.get("ZEBRA", pd.Timestamp("2024-09-01"))
    assert test_group["strike"].tolist() \
        == golden[("TEST", pd.Timestamp("2024-06-15"))]["strike"].tolist() == [10.0]
    assert zebra_group["strike"].tolist() \
        == golden[("ZEBRA", pd.Timestamp("2024-09-01"))]["strike"].tolist() == [13.0]
    for group in (test_group, zebra_group):
        strikes = set(group["strike"])
        assert 11.0 not in strikes  # cross row excluded by the exact-pair batch filter
        assert 12.0 not in strikes
    conn.close()


def test_load_chain_index_returns_requested_keys_from_complete_population(tmp_path):
    wanted = {("TEST", pd.Timestamp("2024-06-15")),
              ("ZEBRA", pd.Timestamp("2024-06-15"))}
    entries = [
        ("TEST", "2024-06-15", 10),
        ("ZEBRA", "2024-06-15", 11),
    ] + [(f"AAA{i}", f"2024-{i:02d}-10", 100 + i) for i in range(1, 13)]
    conn, store, snap = _commit_chains(
        tmp_path, {"2024": _rows(2024, entries)}, maximum_batch_rows=1)
    repository = Repository(conn, store)

    index = _chains.load_chain_index(repository, snap, wanted)
    assert index.get("TEST", pd.Timestamp("2024-06-15"))["strike"].tolist() == [10.0]
    assert index.get("ZEBRA", pd.Timestamp("2024-06-15"))["strike"].tolist() == [11.0]
    conn.close()
