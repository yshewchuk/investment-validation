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