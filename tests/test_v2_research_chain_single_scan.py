"""Issue #252 — one option_chains scan per strategy, not two.

``replay()`` now derives availability from a single ``load_chain_index``
call over the plan's own (unfiltered) keys, then prunes that same
``ChainIndex`` down to the filtered plan's surviving keys with
``ChainIndex.restrict`` before pricing — one scan, retaining exactly the
same rows the old two-scan (``read_chain_keys`` then a filtered
``load_chain_index``) code kept. This test pins that: a single-strategy
replay never touches the whole-table ``read_chain_keys``, calls
``load_chain_index`` exactly once (with the unfiltered keys — the MISS
entry included), and the restricted index handed to pricing holds no MISS
data, while the available event still prices and the one missing its exit
chain is skipped.
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

    real_restrict = _chains.ChainIndex.restrict
    restricted: list = []

    def _restrict_spy(self, keys):
        result = real_restrict(self, keys)
        restricted.append(result)
        return result

    monkeypatch.setattr(_chains.ChainIndex, "restrict", _restrict_spy)

    load_calls: list = []
    real_load = replay.load_chain_index

    def _load_spy(repository, snapshot_ref, keys):
        keys = set(keys)
        load_calls.append(keys)
        return real_load(repository, snapshot_ref, keys)

    monkeypatch.setattr(replay, "load_chain_index", _load_spy)

    result = replay.replay(repository, snap, "STR-THRU", _two_events(),
                           calendar=_calendar())

    assert len(load_calls) == 1
    assert len(restricted) == 1
    assert ("MISS", pd.Timestamp("2024-05-02")) not in set(restricted[0].keys)
    assert len(result.trades) >= 1
    tickers = set(result.trades["ticker"].astype(str))
    assert "TEST" in tickers
    assert "MISS" not in tickers
    assert result.skipped.get("no_exit_chain", 0) == 1
    conn.close()
