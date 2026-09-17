"""Focused regression tests for native v2 session finality."""
from __future__ import annotations

import pandas as pd

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
