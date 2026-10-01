"""Tier-0 tests for ``experiments.common_v2.make_v2_repricer``: pricing over
one pinned v2 snapshot, never the legacy mutable store. Fixture machinery is
reused from ``tests/test_v2_research_replay.py``."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import common_v2  # noqa: E402
from tests.data_scan_support import catalog_and_store  # noqa: E402
from tests.test_v2_research_replay import (  # noqa: E402
    _chain_rows,
    _commit,
    _event_rows,
)


def _trades(entry: str, exit_: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "event_id": "TEST_2024-05-02", "ticker": "TEST",
        "event_date": pd.Timestamp("2024-05-02"), "session": "AMC",
        "entry_date": pd.Timestamp(entry), "exit_date": pd.Timestamp(exit_),
    }])


def _repricer(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit(conn, clock, store, chain_rows=_chain_rows(),
                       event_rows=_event_rows(), receipt_id="r1")
    repricer = common_v2.make_v2_repricer(
        "STR-THRU", catalog=tmp_path / "catalog.sqlite",
        store_root=tmp_path / "store", snapshot_id=snapshot.snapshot_id)
    return conn, repricer


def test_make_v2_repricer_prices_a_trade_covered_by_the_pinned_chains(tmp_path):
    conn, repricer = _repricer(tmp_path)
    assert callable(repricer)
    # entry 2024-05-01 / exit 2024-05-02 shifted +1 trading day lands on the
    # fixture's covered 2024-05-02 / 2024-05-03 chains. Those chains price the
    # STR-THRU ATM straddle (spot 100, strike 100, first post-event expiry,
    # scale 1.0 leg) at alpha 0.5: entry mid 2.2 + 1.2 = 3.4, exit mid
    # 3.2 + 2.2 = 5.4, ret (5.4 - 3.4) / 3.4.
    result = repricer(_trades("2024-05-01", "2024-05-02"), shift_days=1)
    assert isinstance(result, pd.DataFrame)
    assert result.attrs["coverage"] == 1.0
    assert len(result) == 1
    row = result.iloc[0]
    assert row["entry_cost"] == pytest.approx(3.4)
    assert row["exit_value"] == pytest.approx(5.4)
    assert row["ret"] == pytest.approx(2.0 / 3.4)
    conn.close()


def test_make_v2_repricer_skips_a_trade_whose_shifted_chains_are_missing(tmp_path):
    conn, repricer = _repricer(tmp_path)
    result = repricer(_trades("2024-05-02", "2024-05-03"), shift_days=1)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert result.attrs["coverage"] == 0.0
    conn.close()


def test_make_v2_repricer_reuses_cached_chains_across_shift_calls(tmp_path, monkeypatch):
    conn, repricer = _repricer(tmp_path)
    original = common_v2.load_chain_index
    calls: list[set] = []

    def counting_load_chain_index(repository, snapshot, keys):
        calls.append(set(keys))
        return original(repository, snapshot, keys)

    monkeypatch.setattr(common_v2, "load_chain_index", counting_load_chain_index)

    first = repricer(_trades("2024-05-01", "2024-05-02"), shift_days=1)
    assert first.attrs["coverage"] == 1.0
    assert first.iloc[0]["entry_cost"] == pytest.approx(3.4)
    assert first.iloc[0]["exit_value"] == pytest.approx(5.4)

    # A shift this repricer has never been called with directly: its keys
    # were already folded into the cache by the first call above, so this
    # call makes no new chain scan at all (the fixture has no chain data for
    # these shifted dates, so coverage is 0.0 -- same as calling a fresh,
    # uncached repricer with this shift would give).
    second = repricer(_trades("2024-05-01", "2024-05-02"), shift_days=-1)
    assert second.attrs["coverage"] == 0.0

    # Repeating the FIRST call's exact shift gives byte-identical output.
    third = repricer(_trades("2024-05-01", "2024-05-02"), shift_days=1)
    pd.testing.assert_frame_equal(third.reset_index(drop=True), first.reset_index(drop=True))

    assert len(calls) == 1, f"expected exactly one chain scan, got {len(calls)}: {calls}"
    conn.close()


def test_make_v2_repricer_second_call_scans_only_newly_missing_keys(tmp_path, monkeypatch):
    conn, repricer = _repricer(tmp_path)
    original = common_v2.load_chain_index
    requested: list[set] = []

    def counting_load_chain_index(repository, snapshot, keys):
        requested.append(set(keys))
        return original(repository, snapshot, keys)

    monkeypatch.setattr(common_v2, "load_chain_index", counting_load_chain_index)

    repricer(_trades("2024-05-01", "2024-05-02"), shift_days=1)
    assert len(requested) == 1
    first_request = requested[0]

    # A different trade population introduces a genuinely new key
    # (2024-05-06); the already-cached keys from the first call are not
    # requested again.
    repricer(_trades("2024-05-02", "2024-05-03"), shift_days=1)
    assert len(requested) == 2
    assert requested[1].isdisjoint(first_request)
    conn.close()
