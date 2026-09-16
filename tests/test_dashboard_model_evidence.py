"""``engine.dashboard.model_evidence``'s store-facing and process-isolation
helpers.

Regression coverage for the 2026-09-15 memory fixes:

* ``_replay_trades`` used to be ``store.read_table("trades")`` followed by a
  boolean filter on ``provenance``, which held the full unfiltered table and
  its filtered copy in memory at once (measured peak: 4115.5 MiB for that
  one step, on real data, isolated). It now streams and filters one year at
  a time, the same shape ``_daily_subset`` already used for
  ``daily_market``. These tests are the parity check: the streamed result
  must be byte-identical to what the old read-then-filter shape produced,
  across more than one partition, or the fix is not safe.
* ``_run_isolated`` runs each champion's dataset build in its own spawned
  subprocess (see ``_champion_block``/``_champion_block_impl`` in the
  module) so one champion's peak memory cannot compound with another's --
  the trades fix and ``malloc_trim`` alone were not enough: five real
  forced rebuilds in one process still peaked 4.85-5.55 GiB PSS, against
  ~3.5-3.6 GiB for any one champion measured alone.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.data import store
from engine.dashboard.model_evidence import _release_free_pages, _replay_trades, _run_isolated


def _trades_row(trade_id, year, provenance, ticker="AAPL"):
    return {
        "trade_id": trade_id,
        "kind": "sim",
        "strategy": "STR-RUNUP",
        "ticker": ticker,
        "year": year,
        "provenance": provenance,
    }


@pytest.fixture
def replay_and_other_trades(tmp_root):
    """Two partitions (years), each with a mix of provenance values."""
    rows_2023 = [
        _trades_row("t2023-1", 2023, "engine.replay"),
        _trades_row("t2023-2", 2023, "engine.replay"),
        _trades_row("t2023-3", 2023, "legacy_import"),
    ]
    rows_2024 = [
        _trades_row("t2024-1", 2024, "engine.replay"),
        _trades_row("t2024-2", 2024, "manual_paper"),
    ]
    store.write_partition(pd.DataFrame(rows_2023), "trades", 2023)
    store.write_partition(pd.DataFrame(rows_2024), "trades", 2024)
    return {"t2023-1", "t2023-2", "t2024-1"}


def test_replay_trades_keeps_only_engine_replay_rows(replay_and_other_trades):
    out = _replay_trades()
    assert set(out["trade_id"]) == replay_and_other_trades
    assert (out["provenance"] == "engine.replay").all()


def test_replay_trades_matches_the_old_read_then_filter_shape(replay_and_other_trades):
    """The streamed-per-year result must equal ``store.read_table`` then a
    plain boolean filter -- the shape this function replaced -- on every
    column, not just row count, or the memory fix silently changed the data
    every downstream model trains on."""
    streamed = _replay_trades().reset_index(drop=True)

    whole = store.read_table("trades")
    old_shape = whole[whole["provenance"].astype(str) == "engine.replay"].reset_index(drop=True)

    streamed = streamed.sort_values("trade_id").reset_index(drop=True)
    old_shape = old_shape.sort_values("trade_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(streamed, old_shape)


def test_replay_trades_empty_store_returns_empty_frame(tmp_root):
    out = _replay_trades()
    assert len(out) == 0
    assert "provenance" in out.columns


def test_release_free_pages_does_not_raise():
    """Best-effort ``malloc_trim`` call -- must never break a caller even on a
    platform where the libc symbol is unavailable (see the function's own
    docstring)."""
    _release_free_pages()


# --------------------------------------------------------------------------
# _run_isolated: the per-champion subprocess isolation (2026-09-15 fix,
# revised) -- runs each champion's dataset build in its own spawned
# process so one champion's peak cannot compound with another's the way
# five real forced rebuilds in one process measured (4.85-5.55 GiB, even
# after the trades-streaming and malloc_trim fixes above).
# --------------------------------------------------------------------------

#: Module-level (not a closure) so ``spawn`` can import it by reference in
#: the child process.
def _isolated_add(a, b):
    import os

    return {"result": a + b, "pid": os.getpid()}


def _isolated_boom():
    raise ValueError("kaboom")


def test_run_isolated_executes_in_a_different_process():
    import os

    ok, value = _run_isolated(_isolated_add, 2, 3)
    assert ok is True
    assert value == {"result": 5, "pid": value["pid"]}
    assert value["pid"] != os.getpid()


def test_run_isolated_reports_a_raised_exception_without_raising_here():
    """The parent must learn about a child's exception through the return
    value, not have it propagate -- a crashing champion must not take the
    whole rebuild down with it (see _champion_block)."""
    ok, description = _run_isolated(_isolated_boom)
    assert ok is False
    assert "ValueError" in description
    assert "kaboom" in description


def test_run_isolated_passes_kwargs_through():
    ok, value = _run_isolated(_isolated_add, a=10, b=32)
    assert ok is True
    assert value["result"] == 42
