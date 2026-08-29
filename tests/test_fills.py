"""FillModel — the interpolation, the guards, and the breakeven solve."""
from __future__ import annotations

import numpy as np
import pytest

from engine.fills import (
    BEST,
    MID,
    WORST,
    ALPHA_SWEEP,
    FillModel,
    breakeven_alpha,
)


class TestEndpoints:
    def test_worst_buys_the_ask_and_sells_the_bid(self):
        assert WORST.buy(1.0, 1.4) == 1.4
        assert WORST.sell(1.0, 1.4) == 1.0

    def test_best_buys_the_bid_and_sells_the_ask(self):
        assert BEST.buy(1.0, 1.4) == 1.0
        assert BEST.sell(1.0, 1.4) == 1.4

    def test_mid_is_the_midpoint_on_both_sides(self):
        assert MID.buy(1.0, 1.4) == pytest.approx(1.2)
        assert MID.sell(1.0, 1.4) == pytest.approx(1.2)
        assert MID.buy(1.0, 1.4) == MID.sell(1.0, 1.4)

    @pytest.mark.parametrize("alpha", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_alpha_is_the_fraction_of_spread_captured(self, alpha):
        bid, ask = 2.0, 3.0
        fill = FillModel(alpha)
        # Buying: you save `alpha` of the spread relative to paying the ask.
        assert fill.buy(bid, ask) == pytest.approx(ask - alpha * (ask - bid))
        # Selling: you gain `alpha` of the spread relative to hitting the bid.
        assert fill.sell(bid, ask) == pytest.approx(bid + alpha * (ask - bid))

    def test_buy_and_sell_are_symmetric_about_the_mid(self):
        for alpha in (0.0, 0.3, 0.5, 1.0):
            fill = FillModel(alpha)
            buy, sell = fill.buy(4.0, 5.0), fill.sell(4.0, 5.0)
            assert (buy + sell) / 2 == pytest.approx(4.5)

    def test_zero_spread_prices_identically_at_every_alpha(self):
        for alpha in (0.0, 0.5, 1.0):
            assert FillModel(alpha).buy(2.5, 2.5) == 2.5
            assert FillModel(alpha).sell(2.5, 2.5) == 2.5


class TestValidation:
    @pytest.mark.parametrize("alpha", [-0.01, 1.01, 2.0, -1.0, float("nan"), float("inf")])
    def test_alpha_outside_the_unit_interval_is_rejected(self, alpha):
        with pytest.raises(ValueError):
            FillModel(alpha)

    def test_crossed_quote_raises_rather_than_pricing(self):
        # A crossed quote means a bad chain row escaped ingestion validation.
        # Pricing it would yield a plausible-looking number, so it must raise.
        with pytest.raises(ValueError, match="crossed"):
            MID.buy(1.5, 1.0)

    def test_negative_prices_raise(self):
        with pytest.raises(ValueError, match="negative bid"):
            MID.buy(-0.1, 1.0)
        with pytest.raises(ValueError, match="negative ask"):
            MID.sell(0.0, -1.0)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="NaN"):
            MID.buy(float("nan"), 1.0)

    def test_zero_bid_against_positive_ask_is_a_real_market_and_prices(self):
        # Common and legitimate for deep-OTM and near-expiry options.
        assert MID.buy(0.0, 0.10) == pytest.approx(0.05)
        assert WORST.sell(0.0, 0.10) == 0.0

    def test_unknown_side_raises(self):
        with pytest.raises(ValueError, match="unknown side"):
            MID.price("hold", 1.0, 2.0)


class TestWideMarkets:
    def test_wide_market_flagged_above_half_the_mid(self):
        assert FillModel.is_wide(0.5, 1.5) is True  # spread 1.0 on a 1.0 mid
        assert FillModel.is_wide(1.0, 1.1) is False

    def test_no_market_at_all_counts_as_wide(self):
        assert FillModel.is_wide(0.0, 0.0) is True

    def test_zero_bid_is_wide_by_construction(self):
        # (ask - 0) / (ask/2) == 2.0 > 0.5 for any positive ask.
        assert FillModel.is_wide(0.0, 3.0) is True


class TestCashFlow:
    def test_buying_is_cash_out_and_selling_is_cash_in(self):
        assert MID.cash_flow("buy", 1.0, 1.4) == pytest.approx(-1.2)
        assert MID.cash_flow("sell", 1.0, 1.4) == pytest.approx(1.2)

    def test_quantity_scales_linearly(self):
        assert MID.cash_flow("buy", 1.0, 1.4, qty=3) == pytest.approx(-3.6)


class TestVectorized:
    def test_arrays_price_elementwise(self):
        bid = np.array([1.0, 2.0, 0.0])
        ask = np.array([1.4, 2.0, 0.5])
        out = MID.buy(bid, ask)
        assert isinstance(out, np.ndarray)
        np.testing.assert_allclose(out, [1.2, 2.0, 0.25])

    def test_a_single_crossed_row_fails_the_whole_batch(self):
        with pytest.raises(ValueError, match="crossed"):
            MID.buy(np.array([1.0, 9.0]), np.array([1.4, 2.0]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            MID.buy(np.array([1.0, 2.0]), np.array([1.4]))


class TestBreakevenAlpha:
    def test_solves_the_linear_crossing(self):
        # -1 at worst, +1 at best → crosses zero exactly at the mid.
        assert breakeven_alpha(-1.0, 1.0) == pytest.approx(0.5)
        # -3 at worst, +1 at best → needs 75% of the spread to break even.
        assert breakeven_alpha(-3.0, 1.0) == pytest.approx(0.75)

    def test_never_crossing_returns_none(self):
        assert breakeven_alpha(1.0, 2.0) is None  # profitable everywhere
        assert breakeven_alpha(-2.0, -1.0) is None  # losing everywhere

    def test_flat_pnl_returns_none(self):
        assert breakeven_alpha(0.5, 0.5) is None

    def test_matches_a_direct_alpha_sweep(self):
        # P&L is linear in alpha for fixed legs and quotes, so the two-endpoint
        # solve must agree with an explicit sweep.
        bid_in, ask_in, bid_out, ask_out = 2.0, 3.0, 2.2, 3.2

        def pnl(alpha):
            fill = FillModel(alpha)
            return fill.sell(bid_out, ask_out) - fill.buy(bid_in, ask_in)

        root = breakeven_alpha(pnl(0.0), pnl(1.0))
        assert root is not None
        assert pnl(root) == pytest.approx(0.0, abs=1e-12)

        swept = [a for a in ALPHA_SWEEP if pnl(a) >= 0]
        assert min(swept) >= root - 0.05
