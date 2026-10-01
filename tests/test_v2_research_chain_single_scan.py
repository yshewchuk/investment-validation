"""Issue #252 — one option_chains scan per strategy, not two.

``replay()`` derives availability from the loaded ``ChainIndex``'s own keys
instead of a separate ``read_chain_keys`` scan of the whole table. This test
pins that: a single-strategy replay never touches ``read_chain_keys`` and
still prices the available event while skipping the one missing its exit
chain.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.research import _chains, replay  # noqa: E402
from tests.data_scan_support import catalog_and_store  # noqa: E402
from tests.test_v2_research_replay import (  # noqa: E402
    _calendar,
    _chain,
    _chain_rows,
    _commit,
    _event_rows,
)


def _miss_entry_chain_rows() -> list[dict]:
    """``MISS`` has a chain at the entry obs_date and none at the exit."""
    frame = _chain("MISS", "2024-05-02")
    frame["year"] = 2024
    return frame.to_dict("records")


def _two_event_rows() -> list[dict]:
    base = _event_rows()[0]
    miss = dict(base, event_id="MISS_2024-05-02", ticker="MISS")
    return [miss, base]


def _two_events() -> pd.DataFrame:
    row = {"event_date": pd.Timestamp("2024-05-02"), "session": "AMC"}
    return pd.DataFrame(
        [{"event_id": "TEST_2024-05-02", "ticker": "TEST", **row},
         {"event_id": "MISS_2024-05-02", "ticker": "MISS", **row}]
    )


def test_replay_availability_comes_from_the_chain_index_not_read_chain_keys(
    tmp_path, monkeypatch
):
    conn, clock, store = catalog_and_store(tmp_path)
    rows = _chain_rows() + _miss_entry_chain_rows()
    rows.sort(key=lambda row: (row["ticker"], row["obs_date"], row["expiry"],
                               row["strike"], row["right"]))
    snap = _commit(conn, clock, store, chain_rows=rows,
                   event_rows=_two_event_rows(), receipt_id="r1")
    repository = Repository(conn, store)

    def _boom(*args, **kwargs):
        raise AssertionError("read_chain_keys should not be called")

    monkeypatch.setattr(_chains, "read_chain_keys", _boom)
    monkeypatch.setattr(replay, "read_chain_keys", _boom, raising=False)

    result = replay.replay(repository, snap, "STR-THRU", _two_events(),
                           calendar=_calendar())

    assert len(result.trades) >= 1
    tickers = set(result.trades["ticker"].astype(str))
    assert "TEST" in tickers
    assert "MISS" not in tickers
    assert result.skipped.get("no_exit_chain", 0) >= 1
    conn.close()
