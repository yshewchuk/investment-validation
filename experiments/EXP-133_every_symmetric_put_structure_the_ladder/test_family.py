#!/usr/bin/env python3
"""Tests for EXP-133's enumeration, its width rules, and its fast pricer.

    python3 -m pytest experiments/EXP-133_every_symmetric_put_structure_the_ladder

Three things are pinned here, and each of them is a claim the spec makes in
prose that would otherwise be unfalsifiable:

1. **the enumeration is complete and correct** — eight families, both
   incumbents recovered, and no twin peak on four strikes or three;
2. **the defined-risk property survives** — contracts sum to zero, the
   deep-ITM tail is exactly flat, and the floor is zero, on real geometry;
3. **the fast pricer is `engine.structures.price_structure`** — the same
   quotes, the same fill arithmetic, to floating point, on a synthetic ladder
   where the answer can be worked out by hand.

The build's own ``check_equivalence`` proves (3) again on real chains and
aborts the run if it fails. This file proves it where a failure is debuggable.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.fills import FillModel  # noqa: E402
from engine.structures import (  # noqa: E402
    ChainSnapshot, price_structure, twin_peak, twin_peak_5,
)

import build  # noqa: E402
import family as fam  # noqa: E402

MID = FillModel(0.5)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def uniform_rows():
    """A one-expiry put ladder on a uniform $2.50 grid, spot 101.

    Deliberately the same geometry ``tests/test_structures.py`` uses, so a
    disagreement between this file and that one is a disagreement about the
    structures and not about the fixture.
    """
    obs, expiry, dte = pd.Timestamp("2024-05-01"), pd.Timestamp("2024-05-17"), 16
    rows = []
    strike = 60.0
    while strike <= 140.0 + 1e-9:
        for right in ("C", "P"):
            intrinsic = (max(strike - 101.0, 0.0) if right == "P"
                         else max(101.0 - strike, 0.0))
            mid = intrinsic + 2.0
            rows.append({"ticker": "TEST", "obs_date": obs, "expiry": expiry,
                         "dte": dte, "strike": round(strike, 4), "right": right,
                         "bid": round(mid - 0.2, 4), "ask": round(mid + 0.2, 4),
                         "iv": 0.5, "delta": -0.5, "spot": 101.0})
        strike += 2.5
    return pd.DataFrame(rows)


@pytest.fixture
def uniform_snapshot(uniform_rows):
    return ChainSnapshot(ticker="TEST", obs_date=pd.Timestamp("2024-05-01"),
                         event_date=pd.Timestamp("2024-05-02"), rows=uniform_rows,
                         spot=101.0, session="AMC")


def _terminal(strikes, qty, spot):
    """Payoff of a signed put position at expiry, long-positive."""
    return float(sum(q * max(k - spot, 0.0) for k, q in zip(strikes, qty)))


# --------------------------------------------------------------------------
# 1. the enumeration
# --------------------------------------------------------------------------


class TestEnumeration:
    def test_exactly_eight_families_on_three_four_five_and_seven_strikes(self):
        assert len(fam.FAMILIES) == 8
        assert sorted(f.n_strikes for f in fam.FAMILIES) == [3, 4, 5, 5, 5, 7, 7, 7]

    def test_both_incumbents_fall_out_of_the_enumeration(self):
        """The check that the four properties are the right four.

        TWIN-P and TWIN-P5 were written by hand in `engine.structures` months
        before this enumeration existed. If the properties were wrong — too
        loose, too tight, or the wrong ones — the shapes the program already
        trades would not appear in the list.
        """
        keys = {f.key for f in fam.FAMILIES}
        assert "N5q2_-2_1" in keys      # TWIN-P5: +2 at A, -2 at A±a, +1 at A±ma
        assert "N7q2_-1_-1_1" in keys   # TWIN-P:  +2 at A, -1 at A±w, ±2w, +1 at A±4w
        assert "N4q0_-1_1" in keys      # CND-P's shape: long the wings, short inside

    def test_only_two_families_are_twin_peaked(self):
        twin = {f.key for f in fam.FAMILIES if f.twin_peaked}
        assert twin == {"N5q2_-2_1", "N7q2_-1_-1_1"}

    def test_no_twin_peak_exists_on_four_strikes_or_three(self):
        """The registered fact that cuts against the motivating intuition.

        Dropping from five strikes to four does not buy a cheaper twin peak.
        Every symmetric, tail-cancelling, zero-floor debit structure on four
        strikes or three has its maximum at the axis — it is a tent that wants
        a quiet print, which is a different thesis, not a cheaper version of
        the same one.
        """
        small = [f for f in fam.FAMILIES if f.n_strikes in (3, 4)]
        assert small, "the enumeration produced no small families at all"
        assert not any(f.twin_peaked for f in small)

    def test_every_family_cancels_its_tails(self):
        for f in fam.FAMILIES:
            assert f.q0 + 2 * sum(f.tail) == 0

    def test_no_family_exceeds_twin_p_s_eight_contracts(self):
        for f in fam.FAMILIES:
            assert 0 < f.contracts <= fam.MAX_CONTRACTS

    def test_scaled_duplicates_are_not_enumerated_twice(self):
        """Doubling every leg is the same trade: same `ret`, twice the debit."""
        import math

        for f in fam.FAMILIES:
            assert math.gcd(abs(f.q0), *[abs(q) for q in f.tail]) == 1

    @pytest.mark.parametrize("f", fam.FAMILIES, ids=lambda f: f.key)
    def test_the_deep_itm_tail_is_exactly_zero(self, f):
        """Not approximately: the whole defined-risk claim rests on it.

        Below the lowest strike every put is intrinsic, so the position is
        worth ``sum(q_i) * (K_i - S)``. The ``S`` term dies on the contracts
        summing to zero and the constant dies on the mirror symmetry — and
        both have to hold, which is why the mirrored strike must be LISTED
        rather than snapped to the nearest one.
        """
        offsets = next(ds for ds in itertools.combinations(range(1, 13), f.tiers)
                       if fam.offsets_admissible(f.q0, f.tail, ds))
        strikes = [100.0] + [100.0 + d for d in offsets] + [100.0 - d for d in offsets]
        qty = [f.q0] + list(f.tail) + list(f.tail)
        for spot in (0.0, 1.0, 100.0 - max(offsets) - 5.0):
            assert _terminal(strikes, qty, spot) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("f", fam.FAMILIES, ids=lambda f: f.key)
    def test_the_floor_is_zero_everywhere_not_only_at_the_strikes(self, f):
        """The payoff is piecewise linear, so checking the strikes is enough —
        this sweeps between them anyway, because "enough" is the claim."""
        offsets = next(ds for ds in itertools.combinations(range(1, 13), f.tiers)
                       if fam.offsets_admissible(f.q0, f.tail, ds))
        strikes = [100.0] + [100.0 + d for d in offsets] + [100.0 - d for d in offsets]
        qty = [f.q0] + list(f.tail) + list(f.tail)
        grid = np.arange(100.0 - max(offsets) - 3, 100.0 + max(offsets) + 3, 0.05)
        assert min(_terminal(strikes, qty, s) for s in grid) >= -1e-9

    def test_twin_p5_needs_its_wing_at_twice_the_peak_spacing_or_wider(self):
        """`engine.structures.twin_peak_5` allows wings 2 and 3 and no others.
        The enumeration says why: below 2 the floor goes negative."""
        f = fam.FAMILY_BY_KEY["N5q2_-2_1"]
        assert not fam.offsets_admissible(f.q0, f.tail, (2, 3))    # 3 < 2x2
        assert fam.offsets_admissible(f.q0, f.tail, (1, 2))        # wing 2
        assert fam.offsets_admissible(f.q0, f.tail, (1, 3))        # wing 3
        assert fam.offsets_admissible(f.q0, f.tail, (1, 7))        # and wider still

    def test_reduce_offsets_collapses_a_shape_to_its_ratio(self):
        assert fam.reduce_offsets((2, 4, 8)) == (1, 2, 4)
        assert fam.reduce_offsets((3, 9)) == (1, 3)
        assert fam.reduce_offsets((2, 5)) == (2, 5)
        assert fam.reduce_offsets((4,)) == (1,)


# --------------------------------------------------------------------------
# 2. the bridge to the structures the program already trades
# --------------------------------------------------------------------------


class TestMatchesTheIncumbents:
    def test_twin_p5_wing_three_resolves_to_the_same_contracts(self, uniform_snapshot):
        """Same strikes, same sides, same quantities as `twin_peak_5(3)`.

        On a uniform $2.50 ladder one grid step IS the spacing, so the
        enumerated pattern ``N5q2_-2_1`` at offsets (1, 3) and the hand-written
        structure must resolve identically. If they do not, one of them is not
        the shape the program thinks it is.
        """
        pattern = fam.Pattern(fam.FAMILY_BY_KEY["N5q2_-2_1"], (1, 3), 0)
        mine = price_structure(fam.to_structure(pattern), uniform_snapshot, MID)
        theirs = price_structure(twin_peak_5(wing_multiple=3, steps=1),
                                 uniform_snapshot, MID)
        assert _signed(mine) == _signed(theirs)
        assert mine.cost == pytest.approx(theirs.cost)

    def test_twin_p5_wing_two_resolves_to_the_same_contracts(self, uniform_snapshot):
        pattern = fam.Pattern(fam.FAMILY_BY_KEY["N5q2_-2_1"], (1, 2), 0)
        mine = price_structure(fam.to_structure(pattern), uniform_snapshot, MID)
        theirs = price_structure(twin_peak_5(wing_multiple=2, steps=1),
                                 uniform_snapshot, MID)
        assert _signed(mine) == _signed(theirs)
        assert mine.cost == pytest.approx(theirs.cost)

    def test_twin_p_resolves_to_the_same_contracts(self, uniform_snapshot):
        """TWIN-P is ``N7q2_-1_-1_1`` at 1:2:4, and nothing else in the family."""
        pattern = fam.Pattern(fam.FAMILY_BY_KEY["N7q2_-1_-1_1"], (1, 2, 4), 0)
        mine = price_structure(fam.to_structure(pattern), uniform_snapshot, MID)
        theirs = price_structure(twin_peak(steps=1), uniform_snapshot, MID)
        assert _signed(mine) == _signed(theirs)
        assert mine.cost == pytest.approx(theirs.cost)

    def test_the_four_strike_family_has_no_engine_structure(self):
        """Stated as a test because the equivalence receipt reports it as a gap.

        `LegSpec` requires a positive qty and the mirror selector needs a leg
        at the axis to mirror about, so a family carrying no contract at its
        own axis cannot be expressed. The receipt counts these rather than
        quietly narrowing what it checked.
        """
        pattern = fam.Pattern(fam.FAMILY_BY_KEY["N4q0_-1_1"], (1, 3), 0)
        assert fam.to_structure(pattern) is None


def _signed(price):
    """``{strike: signed contracts}`` — the only description of a structure
    that two different constructions can be compared on."""
    out: dict[float, float] = {}
    for leg in price.legs:
        q = leg.qty * (1.0 if leg.side == "buy" else -1.0)
        out[round(leg.strike, 6)] = out.get(round(leg.strike, 6), 0.0) + q
    return {k: v for k, v in out.items() if v != 0}


# --------------------------------------------------------------------------
# 3. the fast pricer and the per-event rules
# --------------------------------------------------------------------------


class TestFastPricer:
    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_net_matches_fillmodel_leg_by_leg(self, alpha):
        """`_net` is `FillModel.cash_flow` summed over legs, sign for sign."""
        bid = np.array([1.0, 2.0, 3.0, 4.0])
        ask = np.array([1.2, 2.4, 3.3, 4.5])
        qty = np.array([[2.0, -2.0, 1.0, 0.0]])
        idx = np.array([[0, 1, 2, 3]])
        fill = FillModel(alpha)
        buy, sell = fill.buy(bid, ask), fill.sell(bid, ask)

        want_cost = -sum(
            fill.cash_flow("buy" if q > 0 else "sell", bid[i], ask[i], abs(q))
            for i, q in zip(idx[0], qty[0]) if q != 0)
        assert build._net(qty, idx, buy, sell)[0] == pytest.approx(want_cost)

        want_exit = sum(
            fill.cash_flow("sell" if q > 0 else "buy", bid[i], ask[i], abs(q))
            for i, q in zip(idx[0], qty[0]) if q != 0)
        assert build._net(qty, idx, sell, buy)[0] == pytest.approx(want_exit)

    def test_a_mirror_that_is_not_listed_is_refused_not_snapped(self):
        """The ladder-position trap, as a test.

        A ladder listing $1 strikes near the money and $5 strikes outside it
        has plenty of positions above the anchor whose exact dollar mirror
        below simply does not exist. Snapping to the nearest one would break
        the symmetry that cancels the tail, so it must return −1.
        """
        strikes = np.array([80.0, 85.0, 90.0, 95.0, 98.0, 99.0, 100.0,
                            101.0, 102.0, 103.0, 105.0])
        anchor = 6                                        # the 100.0 strike
        got = build._mirror_index(strikes, anchor, np.arange(7, 11))
        assert strikes[got[0]] == 99.0                    # 101 -> 99, listed
        assert strikes[got[1]] == 98.0                    # 102 -> 98, listed
        assert got[2] == -1                               # 103 -> 97, NOT listed
        assert strikes[got[3]] == 95.0                    # 105 -> 95, listed

    def test_a_mirror_off_the_bottom_of_the_ladder_is_refused(self):
        strikes = np.array([98.0, 99.0, 100.0, 101.0, 102.0, 120.0])
        got = build._mirror_index(strikes, 2, np.array([5]))   # 120 -> 80, absent
        assert got[0] == -1

    def test_the_floor_check_reads_dollars_not_ladder_positions(self):
        """The reason `patterns()` does not prune on the offset ratio.

        ``N5q2_-2_1`` at ladder positions (1, 2) has a negative floor on a
        uniform ladder — the wing must be at least twice the peak spacing —
        and a fine floor on a ladder whose second step is much larger. Pruning
        on positions would delete the second case, which lives exactly on the
        tickers with coarse outer ladders that this experiment exists to reach.
        """
        # TWIN-P5's contract vector: +2 at the anchor, -2 at A+-a, +1 at A+-wing.
        qty = np.array([[2.0, -2.0, -2.0, 1.0, 1.0, 0.0, 0.0]])
        idx = np.array([[2, 3, 1, 4, 0, 2, 2]])
        # Same ladder POSITIONS both times. Uniform: the wing sits at exactly
        # twice the spacing, floor 0.
        uniform = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        assert build._floor_ok(uniform, idx, qty).all()
        # Ladder that TIGHTENS outward: the wing is only 1.5x the spacing in
        # dollars, so the floor at the anchor goes negative.
        tight = np.array([92.5, 95.0, 100.0, 105.0, 107.5])
        assert not build._floor_ok(tight, idx, qty).any()
        # Ladder that WIDENS outward: the same positions are a fine structure,
        # and a position-based prune would have deleted it.
        wide = np.array([80.0, 95.0, 100.0, 105.0, 120.0])
        assert build._floor_ok(wide, idx, qty).all()

    def test_expected_pnl_is_linear_in_the_contract_vector(self):
        """The claim that makes 12,600 candidates affordable, in isolation.

        Simulating a structure is averaging a sum over draws; the shortcut sums
        an average over strikes. They are the same number because expectation
        is linear, and if that ever stops being true the whole build is wrong.
        """
        rng = np.random.default_rng(7)
        spot_exit = 100.0 * (1.0 + rng.normal(0, 0.08, 4000))
        vol = np.full(4000, 0.35)
        strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        values = build.pnl_sim.black_scholes_put(
            spot_exit[:, None], strikes[None, :], 9 / 365.0, vol[:, None])
        qty = np.array([1.0, -2.0, 2.0, -2.0, 1.0])
        per_draw = (values * qty[None, :]).sum(axis=1).mean()
        via_means = float(qty @ values.mean(axis=0))
        assert per_draw == pytest.approx(via_means, rel=1e-12)


class TestWidthRules:
    """The two rules, applied to the OUTER strikes rather than to a half-width.

    Writing them on both outer strikes is what makes an off-centre anchor pay
    for itself: shifting the axis up by delta tightens the band by delta at
    both ends, so a shifted structure has to be a narrower one.
    """

    @staticmethod
    def _rule(anchor, half, spot, m, s):
        lo, hi = anchor - half, anchor + half
        spans = lo < spot * (1 - m / 100) and hi > spot * (1 + m / 100)
        within = (lo >= spot * (1 - (m + 3 * s) / 100)
                  and hi <= spot * (1 + (m + 3 * s) / 100))
        return spans, within

    def test_a_structure_narrower_than_the_forecast_is_refused(self):
        spans, _ = self._rule(anchor=100.0, half=4.0, spot=100.0, m=6.0, s=4.0)
        assert not spans

    def test_a_structure_that_spans_the_forecast_passes_the_first_rule(self):
        spans, within = self._rule(anchor=100.0, half=7.0, spot=100.0, m=6.0, s=4.0)
        assert spans and within

    def test_a_structure_wider_than_three_sds_is_refused(self):
        _, within = self._rule(anchor=100.0, half=20.0, spot=100.0, m=6.0, s=4.0)
        assert not within

    def test_shifting_the_anchor_up_tightens_the_band_at_both_ends(self):
        """A width that is fine centred can fail both rules once shifted."""
        centred_spans, centred_within = self._rule(100.0, 6.2, 100.0, 6.0, 4.0)
        assert centred_spans and centred_within
        shifted_spans, _ = self._rule(102.0, 6.2, 100.0, 6.0, 4.0)
        assert not shifted_spans          # its LOWER strike no longer reaches
