"""Shared fixtures.

Tests must not touch the real data store, the real raw cache, or the network.
Anything that writes goes to a ``tmp_path`` root; anything that would fetch uses
a fake adapter.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    """Point ``engine.paths`` at a throwaway tree for the duration of a test."""
    monkeypatch.setenv("INVESTING_PLAN_ROOT", str(tmp_path))
    from engine import paths

    importlib.reload(paths)
    yield tmp_path
    monkeypatch.delenv("INVESTING_PLAN_ROOT", raising=False)
    importlib.reload(paths)


@pytest.fixture
def chain_rows():
    """A small, hand-checkable two-expiry chain for one ticker.

    Spot is 100. Strikes bracket it so ATM selection has a unique answer, and
    every bid/ask is a round number so leg arithmetic can be verified by hand.
    """
    obs = pd.Timestamp("2024-05-01")
    rows = []
    for expiry, dte in ((pd.Timestamp("2024-05-03"), 2), (pd.Timestamp("2024-05-24"), 23)):
        for strike in (95.0, 100.0, 105.0):
            for right, bid, ask in (("C", 2.0, 2.4), ("P", 1.0, 1.4)):
                scale = 1.0 if dte < 10 else 2.0
                rows.append(
                    {
                        "ticker": "TEST",
                        "obs_date": obs,
                        "expiry": expiry,
                        "dte": dte,
                        "strike": strike,
                        "right": right,
                        "bid": round(bid * scale, 4),
                        "ask": round(ask * scale, 4),
                        "mid": round((bid + ask) / 2 * scale, 4),
                        "iv": 0.4,
                        "delta": 0.5 if right == "C" else -0.5,
                        "spot": 100.0,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def chain_snapshot(chain_rows):
    from engine.structures import ChainSnapshot

    return ChainSnapshot(
        ticker="TEST",
        obs_date=pd.Timestamp("2024-05-01"),
        event_date=pd.Timestamp("2024-05-02"),
        rows=chain_rows,
        spot=100.0,
    )
