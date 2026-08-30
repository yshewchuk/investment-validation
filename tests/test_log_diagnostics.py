"""tools.log_diagnostics — where a reported curve's return actually came from.

The arithmetic here is decision-relevant: it is what says the ungated STR-THRU
edge is inside the spread. So the re-pricing is tested against hand-computed
values rather than against the engine, which is the point of the tool — it must
be able to disagree with the engine that produced the log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools import log_diagnostics as ld


def _log(n: int = 4, side1: str = "buy", side2: str = "buy") -> pd.DataFrame:
    """A long straddle: two bought legs, sold at exit."""
    return pd.DataFrame({
        "row": range(1, n + 1),
        "ticker": ["T"] * n,
        "event_date": pd.date_range("2024-01-02", periods=n, freq="30D"),
        "entry_cost": [8.4] * n,
        "exit_value": [9.4, 6.4, 8.4, 12.4][:n],
        "ret": [(9.4 - 8.4) / 8.4, (6.4 - 8.4) / 8.4, 0.0, (12.4 - 8.4) / 8.4][:n],
        "entry_leg1_bid": [4.0] * n, "entry_leg1_ask": [4.8] * n,
        "entry_leg1_side": [side1] * n, "entry_leg1_qty": [1.0] * n,
        "entry_leg2_bid": [4.0] * n, "entry_leg2_ask": [4.8] * n,
        "entry_leg2_side": [side2] * n, "entry_leg2_qty": [1.0] * n,
        "exit_leg1_bid": [4.5] * n, "exit_leg1_ask": [5.3] * n,
        "exit_leg1_side": ["sell"] * n, "exit_leg1_qty": [1.0] * n,
        "exit_leg2_bid": [4.5] * n, "exit_leg2_ask": [5.3] * n,
        "exit_leg2_side": ["sell"] * n, "exit_leg2_qty": [1.0] * n,
    })


class TestRepricing:
    def test_mid_reproduces_the_quoted_mid(self):
        cost, value, ret = ld.repriced(_log(), 0.5)
        assert cost.iloc[0] == pytest.approx(8.8)      # 2 legs x mid 4.4
        assert value.iloc[0] == pytest.approx(9.8)     # 2 legs x mid 4.9

    def test_worst_and_best_bracket_it(self):
        worst_cost, worst_value, _ = ld.repriced(_log(), 0.0)
        best_cost, best_value, _ = ld.repriced(_log(), 1.0)
        assert worst_cost.iloc[0] == pytest.approx(9.6)   # pay both asks
        assert worst_value.iloc[0] == pytest.approx(9.0)  # sell both bids
        assert best_cost.iloc[0] == pytest.approx(8.0)    # pay both bids
        assert best_value.iloc[0] == pytest.approx(10.6)  # sell both asks
        assert worst_value.iloc[0] - worst_cost.iloc[0] < best_value.iloc[0] - best_cost.iloc[0]

    def test_a_sold_entry_leg_is_a_credit(self):
        """A calendar's short front leg reduces the debit, not increases it."""
        cost_long, _v, _r = ld.repriced(_log(side2="buy"), 0.5)
        cost_calendar, _v2, _r2 = ld.repriced(_log(side2="sell"), 0.5)
        assert cost_calendar.iloc[0] == pytest.approx(0.0)
        assert cost_calendar.iloc[0] < cost_long.iloc[0]


class TestCapitalWeighted:
    def test_it_weights_by_premium_not_by_trade(self):
        log = pd.DataFrame({
            "entry_cost": [1.0, 100.0],
            "exit_value": [2.0, 90.0],       # +100% on $1, -10% on $100
        })
        equal = np.mean([1.0, -0.1])
        capital = ld.capital_weighted(log.entry_cost, log.exit_value)
        assert equal == pytest.approx(0.45)
        assert capital == pytest.approx((92.0 - 101.0) / 101.0)
        assert capital < 0 < equal, "the divergence this column exists to show"


class TestBreakevenAlpha:
    def test_it_finds_the_crossing(self):
        alpha = ld.breakeven_alpha(_log())
        assert alpha is not None and 0.0 < alpha < 1.0
        cost, value, _ = ld.repriced(_log(), alpha)
        assert (value - cost).sum() == pytest.approx(0.0, abs=1e-6)

    def test_none_when_the_book_never_crosses(self):
        log = _log()
        for c in ("exit_leg1_bid", "exit_leg1_ask", "exit_leg2_bid", "exit_leg2_ask"):
            log[c] = 100.0                     # profitable at every alpha
        assert ld.breakeven_alpha(log) is None


class TestSections:
    def test_build_sections_covers_the_four_cuts(self):
        log = _log(4)
        titles = [s["title"] for s in ld.build_sections(log, split_year=None)]
        assert "Headline, two ways" in titles
        assert "The same book at every fill assumption" in titles
        assert any("cost" in t for t in titles) and any("wide" in t for t in titles)
        assert "How many trades make the result" in titles

    def test_the_gated_split_appears_only_when_asked(self):
        log = _log(4)
        without = [s["title"] for s in ld.build_sections(log, split_year=None)]
        with_split = [s["title"] for s in ld.build_sections(log, split_year=2024)]
        assert not any("Gated vs ungated" in t for t in without)
        assert any("Gated vs ungated" in t for t in with_split)

    def test_small_samples_do_not_fabricate_quintiles(self):
        section = ld.bucket_table(_log(4), pd.Series([1, 2, 3, 4]), "t", "n", "x", "f")
        assert "Too few distinct values" in " ".join(section.get("body", []))
