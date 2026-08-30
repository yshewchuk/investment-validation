"""The fill-quality join: ORATS quotes against Polygon real trades.

The numbers this tool prints are what the mid-fill assumption gets measured
against, so the arithmetic — alpha direction, bucketing, the OCC-id join key —
is tested on hand-checkable rows rather than trusted to the join.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.fill_quality import contract_ticker_column, summarize


def make_chains(rows):
    return pd.DataFrame(rows)


class TestContractKey:
    def test_the_key_matches_the_pulls_occ_builder(self):
        from engine.data.sources.polygon import option_ticker

        chains = make_chains(
            [
                {
                    "ticker": "TSLA",
                    "expiry": pd.Timestamp("2024-09-06"),
                    "strike": 210.0,
                    "right": "C",
                }
            ]
        )
        assert contract_ticker_column(chains).iloc[0] == option_ticker(
            "TSLA", "2024-09-06", "C", 210.0
        )

    def test_fractional_strikes_pad_to_eight_digits(self):
        chains = make_chains(
            [
                {
                    "ticker": "F",
                    "expiry": pd.Timestamp("2024-01-19"),
                    "strike": 12.5,
                    "right": "P",
                }
            ]
        )
        assert contract_ticker_column(chains).iloc[0] == "O:F240119P00012500"


class TestSummarize:
    def joined(self, n_trades, close, bid, ask):
        """One matched contract-day with the fields summarize() reads."""
        width = ask - bid
        return {
            "contract_ticker": "O:X",
            "obs_date": pd.Timestamp("2025-05-05"),
            "close": close,
            "vwap": close,
            "volume": 10.0,
            "n_trades": n_trades,
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
            "rel_spread": width / ((bid + ask) / 2) if bid + ask > 0 else np.nan,
            "alpha_close": (ask - close) / width if width > 0 else np.nan,
            "alpha_vwap": (ask - close) / width if width > 0 else np.nan,
        }

    def test_alpha_direction(self):
        # A close at the ask is alpha 0 (worst fill for a buyer); at the bid
        # it is alpha 1; at mid it is 0.5.
        df = pd.DataFrame(
            [
                self.joined(6, 1.2, 1.0, 1.4),  # at ask
                self.joined(6, 1.0, 1.0, 1.4),  # at bid
                self.joined(6, 1.2, 1.0, 1.4),
            ]
        )
        out = summarize(df)
        assert out["median_alpha_close"].iloc[0] == pytest.approx(0.5)

    def test_trades_through_the_spread_are_counted(self):
        df = pd.DataFrame(
            [
                self.joined(6, 1.6, 1.0, 1.4),  # above the ask: alpha -0.5
                self.joined(6, 1.2, 1.0, 1.4),
            ]
        )
        out = summarize(df)
        assert out["pct_close_beyond_worst"].iloc[0] == pytest.approx(0.5)

    def test_liquidity_buckets(self):
        df = pd.DataFrame(
            [self.joined(n, 1.2, 1.0, 1.4) for n in (1, 3, 20, 100)]
        )
        out = summarize(df)
        assert list(out["n_trades_bucket"]) == ["1", "2-5", "6-50", "50+"]
        assert list(out["days"]) == [1, 1, 1, 1]
