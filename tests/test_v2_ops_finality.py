"""Focused regression tests for native v2 session finality."""
from __future__ import annotations

import pandas as pd

from engine.data import finality as legacy_finality
from engine.v2.ops import finality


def test_disjoint_daily_and_chain_tickers_are_not_final():
    """A ticker carried by one table must count as missing from the other."""
    day = "2026-09-10"
    frames = {
        "daily_market": pd.DataFrame({"ticker": ["A"], "date": [day]}),
        "option_chains": pd.DataFrame({"ticker": ["B"], "obs_date": [day]}),
    }

    result = finality.session_finality(
        day, ["A", "B"], frames=frames, market_wide=True)

    assert not result.is_final
    assert result.covered == 2
    assert result.daily_share == 0.5
    assert result.chain_share == 0.5
    assert "daily 50%" in result.detail
    assert "chains 50%" in result.detail


def test_tickers_absent_from_both_tables_do_not_veto_finality():
    """The shared denominator still excludes names no source carries."""
    day = "2026-09-10"
    frames = {
        "daily_market": pd.DataFrame({"ticker": ["A"], "date": [day]}),
        "option_chains": pd.DataFrame({"ticker": ["A"], "obs_date": [day]}),
    }

    result = finality.session_finality(
        day, ["A", "GHOST"], frames=frames, market_wide=True)

    assert result.is_final
    assert result.covered == 1
    assert result.daily_share == 1.0
    assert result.chain_share == 1.0


def test_covered_tickers_honors_the_legacy_seam_and_is_not_an_echo(monkeypatch):
    """R3B-7 regression: ``engine.v2.ops.finality.covered_tickers`` reads its
    frames and market-wide flag through ``engine.v2.ops.finality._coverage_frame``
    / ``_market_wide_complete``. Both of those must honor an explicit
    monkeypatch of the corresponding private helpers on ``engine.data.finality``
    -- the shape a fixture patches to fake data without also faking the public
    ``covered_tickers`` name itself. Before this fix, the native path bypassed
    the patched helpers entirely (it read real ORATS cache / curated store
    data), so it silently returned an empty list for every ticker instead of a
    genuine per-ticker result -- indistinguishable, in isolation, from a code
    path that just echoed (or emptied) the request.
    """
    day = "2026-09-10"
    daily = pd.DataFrame({"ticker": ["A", "B"], "date": [day, day]})
    # B's chain observation is stale; C is never carried at all.
    chains = pd.DataFrame({"ticker": ["A", "B"], "obs_date": [day, "2026-01-01"]})

    monkeypatch.setattr(legacy_finality, "_market_wide_complete", lambda stamp: True)
    monkeypatch.setattr(
        legacy_finality, "_coverage_frame",
        lambda table, column, stamp: daily if table == "daily_market" else chains)

    result = finality.covered_tickers(day, ["A", "B", "C"])

    assert result == ["A"]
    # The whole point: a genuine per-ticker computation differs from the
    # request it was asked about, in both directions -- it must not merely
    # echo the request back, and it must not empty out to nothing either.
    assert result != sorted(["A", "B", "C"])
    assert result != []
