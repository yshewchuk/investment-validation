"""Structures — resolution, pricing, and the open/close side discipline."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.fills import BEST, MID, WORST, FillModel
from engine.structures import (
    ChainSnapshot,
    ExpirySelector,
    LegSpec,
    ResolvedLeg,
    StrikeSelector,
    Structure,
    StructureError,
    StructurePrice,
    STRUCTURES,
    price_structure,
    put_calendar,
    put_condor,
    straddle_runup,
    straddle_through,
    structure_return,
)


class TestSelectors:
    def test_first_post_event_skips_expiries_that_die_before_the_print(self, chain_rows):
        # 430 events in the existing S2 trade set have an expiry at or before
        # the earnings date; picking "the front expiry" without this filter
        # books an option that expired before the event.
        rows = chain_rows.copy()
        event = pd.Timestamp("2024-05-10")
        chosen = ExpirySelector(kind="first_post_event").select(rows, event)
        assert chosen == pd.Timestamp("2024-05-24")

    def test_first_post_event_raises_when_nothing_survives(self, chain_rows):
        with pytest.raises(StructureError, match="on or after"):
            ExpirySelector(kind="first_post_event").select(
                chain_rows, pd.Timestamp("2025-01-01")
            )

    def test_fixed_expiry_selects_exactly_what_was_asked_for(self, chain_rows):
        """The Phase 1 `score(..., expiry=)` argument resolves through this."""
        chosen = ExpirySelector(
            kind="fixed", expiry=pd.Timestamp("2024-05-24")
        ).select(chain_rows, pd.Timestamp("2024-05-02"))
        assert chosen == pd.Timestamp("2024-05-24")

    def test_fixed_expiry_ignores_the_dte_filters(self):
        """Those exist to *choose* an expiry; a named one leaves nothing to choose."""
        rows = pd.DataFrame(
            {"expiry": [pd.Timestamp("2024-05-03")], "dte": [2]}
        )
        chosen = ExpirySelector(
            kind="fixed", expiry=pd.Timestamp("2024-05-03"), min_dte=30
        ).select(rows, pd.Timestamp("2024-05-01"))
        assert chosen == pd.Timestamp("2024-05-03")

    def test_fixed_expiry_absent_from_the_chain_raises(self, chain_rows):
        with pytest.raises(StructureError, match="absent from this chain"):
            ExpirySelector(kind="fixed", expiry=pd.Timestamp("2030-01-18")).select(
                chain_rows, pd.Timestamp("2024-05-02")
            )

    def test_fixed_expiry_requires_an_expiry(self):
        with pytest.raises(ValueError, match="fixed requires expiry"):
            ExpirySelector(kind="fixed")

    def test_nearest_dte_breaks_ties_toward_the_longer_expiry(self):
        rows = pd.DataFrame(
            {"expiry": [pd.Timestamp("2024-05-05"), pd.Timestamp("2024-05-15")],
             "dte": [5, 15]}
        )
        chosen = ExpirySelector(kind="nearest_dte", target_dte=10).select(
            rows, pd.Timestamp("2024-05-01")
        )
        assert chosen == pd.Timestamp("2024-05-15")

    def test_first_dte_at_least_picks_the_earliest_qualifying(self, chain_rows):
        chosen = ExpirySelector(kind="first_dte_at_least", target_dte=20).select(
            chain_rows, pd.Timestamp("2024-05-02")
        )
        assert chosen == pd.Timestamp("2024-05-24")

    def test_atm_picks_the_strike_nearest_spot(self, chain_rows):
        rows = chain_rows[chain_rows["expiry"] == pd.Timestamp("2024-05-03")]
        assert StrikeSelector("atm").select(rows, 100.0, {}) == 100.0
        assert StrikeSelector("atm").select(rows, 104.0, {}) == 105.0

    def test_moneyness_scales_the_target(self, chain_rows):
        rows = chain_rows[chain_rows["expiry"] == pd.Timestamp("2024-05-03")]
        got = StrikeSelector(kind="moneyness", moneyness=0.95).select(rows, 100.0, {})
        assert got == 95.0

    def test_same_as_reuses_a_previously_resolved_strike(self, chain_rows):
        rows = chain_rows[chain_rows["expiry"] == pd.Timestamp("2024-05-24")]
        got = StrikeSelector(kind="same_as", ref="front").select(rows, 100.0, {"front": 95.0})
        assert got == 95.0

    def test_same_as_raises_when_the_reference_strike_is_absent(self, chain_rows):
        rows = chain_rows[chain_rows["expiry"] == pd.Timestamp("2024-05-24")]
        with pytest.raises(StructureError, match="absent"):
            StrikeSelector(kind="same_as", ref="front").select(rows, 100.0, {"front": 42.0})

    def test_unknown_selector_kinds_are_rejected_at_construction(self):
        with pytest.raises(ValueError):
            ExpirySelector(kind="whenever")
        with pytest.raises(ValueError):
            StrikeSelector(kind="vibes")

    def test_selectors_requiring_a_parameter_demand_it(self):
        with pytest.raises(ValueError, match="target_dte"):
            ExpirySelector(kind="nearest_dte")
        with pytest.raises(ValueError, match="moneyness"):
            StrikeSelector(kind="moneyness")


class TestStructureSpecs:
    def test_put_calendar_is_short_front_and_long_back(self):
        spec = put_calendar(back_dte=20)
        assert [leg.side for leg in spec.legs] == ["sell", "buy"]
        assert all(leg.right == "P" for leg in spec.legs)

    def test_put_calendar_opens_and_closes_both_legs_together(self):
        # The structural claim in the plan: at no point is the short put naked.
        spec = put_calendar()
        assert spec.entry_offset == 0 and spec.exit_offset == 1
        assert spec.holds_through_print

    def test_put_calendar_is_same_strike_unless_made_a_diagonal(self):
        assert put_calendar().legs[1].strike.kind == "same_as"
        assert put_calendar(back_moneyness=0.95).legs[1].strike.kind == "moneyness"

    def test_straddle_runup_never_sees_the_print(self):
        spec = straddle_runup()
        assert spec.exit_offset == 0
        assert not spec.holds_through_print

    def test_straddle_through_holds_across_the_event(self):
        assert straddle_through().holds_through_print

    def test_only_the_calendar_carries_a_short_leg(self):
        assert put_calendar().has_short_leg
        assert not straddle_through().has_short_leg
        assert not straddle_runup().has_short_leg

    def test_exit_must_follow_entry(self):
        with pytest.raises(ValueError, match="must be after"):
            straddle_through(entry_offset=1, exit_offset=0)

    def test_duplicate_leg_names_are_rejected(self):
        expiry = ExpirySelector(kind="first_post_event")
        with pytest.raises(ValueError, match="duplicate leg names"):
            Structure(
                name="X",
                legs=(
                    LegSpec("a", "C", "buy", expiry),
                    LegSpec("a", "P", "buy", expiry),
                ),
                entry_offset=0,
                exit_offset=1,
            )

    def test_forward_same_as_reference_is_rejected(self):
        expiry = ExpirySelector(kind="first_post_event")
        with pytest.raises(ValueError, match="resolves later"):
            Structure(
                name="X",
                legs=(
                    LegSpec("first", "P", "buy", expiry, StrikeSelector("same_as", ref="second")),
                    LegSpec("second", "P", "buy", expiry),
                ),
                entry_offset=0,
                exit_offset=1,
            )

    def test_a_structure_decides_at_its_entry_close_by_default(self):
        """The status quo, made explicit: every shipped structure decides on
        the same close it enters on, which is why STR-THRU's prediction cannot
        be acted on — the chain it prices against is only published once that
        close has already happened."""
        for spec in (put_calendar(), straddle_through(), straddle_runup()):
            assert spec.decision_offset is None
            assert spec.decided_at == spec.entry_offset
            assert not spec.decided_early

    def test_a_decision_offset_moves_the_decision_earlier(self):
        spec = straddle_through(decision_offset=-1)
        assert spec.decided_at == -1
        assert spec.decided_early
        assert spec.entry_offset == 0  # the trade itself is unchanged

    def test_the_decision_may_not_come_after_the_entry(self):
        with pytest.raises(ValueError, match="cannot be after entry_offset"):
            straddle_through(decision_offset=1)

    def test_deciding_at_the_entry_close_is_allowed_explicitly(self):
        """`decision_offset == entry_offset` is legal — it just says out loud
        what `None` says implicitly — but it is not `decided_early`, so it must
        not change a variant label or pull a second chain."""
        spec = straddle_through(decision_offset=0)
        assert spec.decided_at == 0
        assert not spec.decided_early

    def test_every_structure_can_take_one(self):
        assert put_calendar(decision_offset=-1).decided_at == -1
        assert straddle_runup(decision_offset=-16).decided_at == -16

    def test_spec_round_trips_to_a_dict(self):
        spec = put_calendar(back_dte=30)
        blob = spec.to_dict()
        assert blob["name"] == "CAL-P"
        assert blob["params"]["back_dte"] == 30
        assert len(blob["legs"]) == 2
        # Serialized even when unset: a spec that omits it is indistinguishable
        # from one written before decision offsets existed, and the two do not
        # mean the same thing once any structure sets it.
        assert blob["decision_offset"] is None
        assert straddle_through(decision_offset=-1).to_dict()["decision_offset"] == -1


class TestPricing:
    def test_long_straddle_costs_the_two_asks_at_worst_fills(self, chain_snapshot):
        priced = price_structure(straddle_through(), chain_snapshot, WORST)
        # 2024-05-03 expiry is the first on or after the 2024-05-02 event.
        assert {leg.expiry for leg in priced.legs} == {pd.Timestamp("2024-05-03")}
        assert priced.cost == pytest.approx(2.4 + 1.4)

    def test_mid_fill_costs_the_two_mids(self, chain_snapshot):
        priced = price_structure(straddle_through(), chain_snapshot, MID)
        assert priced.cost == pytest.approx(2.2 + 1.2)

    def test_best_fill_costs_the_two_bids(self, chain_snapshot):
        priced = price_structure(straddle_through(), chain_snapshot, BEST)
        assert priced.cost == pytest.approx(2.0 + 1.0)

    def test_cost_is_monotone_decreasing_in_alpha(self, chain_snapshot):
        costs = [
            price_structure(straddle_through(), chain_snapshot, FillModel(a)).cost
            for a in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        assert costs == sorted(costs, reverse=True)

    def test_put_calendar_debit_is_back_minus_front(self, chain_snapshot):
        priced = price_structure(put_calendar(back_dte=20), chain_snapshot, MID)
        front, back = priced.leg("front_put"), priced.leg("back_put")
        assert front.side == "sell" and back.side == "buy"
        assert front.expiry == pd.Timestamp("2024-05-03")
        assert back.expiry == pd.Timestamp("2024-05-24")
        # Back put mid is 2.4 (2x scale), front put mid is 1.2.
        assert priced.cost == pytest.approx(2.4 - 1.2)

    def test_calendar_legs_share_a_strike_by_default(self, chain_snapshot):
        priced = price_structure(put_calendar(), chain_snapshot, MID)
        assert priced.leg("front_put").strike == priced.leg("back_put").strike

    def test_missing_contract_raises_rather_than_substituting(self, chain_rows):
        snapshot = ChainSnapshot(
            ticker="TEST",
            obs_date=pd.Timestamp("2024-05-01"),
            event_date=pd.Timestamp("2024-05-02"),
            rows=chain_rows[chain_rows["right"] == "C"],
            spot=100.0,
        )
        with pytest.raises(StructureError, match="no P at strike"):
            price_structure(straddle_through(), snapshot, MID)

    def test_snapshot_missing_columns_is_caught_at_construction(self):
        with pytest.raises(StructureError, match="missing columns"):
            ChainSnapshot(
                ticker="T",
                obs_date=pd.Timestamp("2024-05-01"),
                event_date=pd.Timestamp("2024-05-02"),
                rows=pd.DataFrame({"strike": [100.0]}),
            )

    def test_wide_market_is_flagged_through_to_the_structure(self, chain_rows):
        rows = chain_rows.copy()
        rows.loc[rows["right"] == "P", "bid"] = 0.0
        snapshot = ChainSnapshot(
            "TEST", pd.Timestamp("2024-05-01"), pd.Timestamp("2024-05-02"), rows, 100.0
        )
        priced = price_structure(straddle_through(), snapshot, MID)
        assert priced.any_wide_market


class TestOpenCloseDiscipline:
    """Closing reverses every leg — the single easiest place to overstate P&L."""

    def test_closing_flips_each_leg_side(self, chain_snapshot):
        opened = price_structure(straddle_through(), chain_snapshot, MID)
        closed = price_structure(straddle_through(), chain_snapshot, MID, closing=True)
        assert [leg.side for leg in opened.legs] == ["buy", "buy"]
        assert [leg.side for leg in closed.legs] == ["sell", "sell"]

    def test_worst_case_round_trip_pays_the_full_spread_twice(self, chain_snapshot):
        spec = straddle_through()
        opened = price_structure(spec, chain_snapshot, WORST)
        closed = price_structure(spec, chain_snapshot, WORST, pin=opened.legs, closing=True)
        result = structure_return(opened, closed)
        # Bought at both asks (3.8), sold at both bids (3.0) with no move.
        assert result["cost"] == pytest.approx(3.8)
        assert result["exit_value"] == pytest.approx(3.0)
        assert result["pnl"] == pytest.approx(-0.8)

    def test_mid_round_trip_with_no_move_is_flat(self, chain_snapshot):
        spec = straddle_through()
        opened = price_structure(spec, chain_snapshot, MID)
        closed = price_structure(spec, chain_snapshot, MID, pin=opened.legs, closing=True)
        assert structure_return(opened, closed)["pnl"] == pytest.approx(0.0)

    def test_cost_is_rejected_on_a_closing_price(self, chain_snapshot):
        closed = price_structure(straddle_through(), chain_snapshot, MID, closing=True)
        with pytest.raises(StructureError, match="opening concept"):
            _ = closed.cost

    def test_exit_value_is_rejected_on_an_opening_price(self, chain_snapshot):
        opened = price_structure(straddle_through(), chain_snapshot, MID)
        with pytest.raises(StructureError, match="closing concept"):
            _ = opened.exit_value

    def test_structure_return_rejects_two_opens(self, chain_snapshot):
        opened = price_structure(straddle_through(), chain_snapshot, MID)
        with pytest.raises(StructureError, match="closing=True"):
            structure_return(opened, opened)


class TestPinning:
    def test_pinning_holds_the_contract_when_spot_moves(self, chain_rows):
        spec = straddle_through()
        entry = ChainSnapshot(
            "TEST", pd.Timestamp("2024-05-01"), pd.Timestamp("2024-05-02"), chain_rows, 100.0
        )
        opened = price_structure(spec, entry, MID)
        assert opened.leg("call").strike == 100.0

        # Spot gaps to 105 post-print. Without pinning, ATM selection follows.
        moved = ChainSnapshot(
            "TEST", pd.Timestamp("2024-05-03"), pd.Timestamp("2024-05-02"), chain_rows, 105.0
        )
        unpinned = price_structure(spec, moved, MID, closing=True)
        assert unpinned.leg("call").strike == 105.0

        pinned = price_structure(spec, moved, MID, pin=opened.legs, closing=True)
        assert pinned.leg("call").strike == 100.0

    def test_unpinned_close_on_a_moved_spot_is_rejected_by_structure_return(self, chain_rows):
        spec = straddle_through()
        entry = ChainSnapshot(
            "TEST", pd.Timestamp("2024-05-01"), pd.Timestamp("2024-05-02"), chain_rows, 100.0
        )
        moved = ChainSnapshot(
            "TEST", pd.Timestamp("2024-05-03"), pd.Timestamp("2024-05-02"), chain_rows, 105.0
        )
        opened = price_structure(spec, entry, MID)
        closed = price_structure(spec, moved, MID, closing=True)
        with pytest.raises(StructureError, match="pin="):
            structure_return(opened, closed)

    def test_pinning_to_a_missing_expiry_raises(self, chain_rows):
        spec = straddle_through()
        entry = ChainSnapshot(
            "TEST", pd.Timestamp("2024-05-01"), pd.Timestamp("2024-05-02"), chain_rows, 100.0
        )
        opened = price_structure(spec, entry, MID)
        thin = chain_rows[chain_rows["expiry"] == pd.Timestamp("2024-05-24")]
        later = ChainSnapshot(
            "TEST", pd.Timestamp("2024-05-06"), pd.Timestamp("2024-05-02"), thin, 100.0
        )
        with pytest.raises(StructureError, match="pinned expiry"):
            price_structure(spec, later, MID, pin=opened.legs, closing=True)


class TestReturns:
    def test_return_is_quoted_on_the_net_debit(self, chain_snapshot):
        spec = straddle_through()
        opened = price_structure(spec, chain_snapshot, MID)

        richer = chain_snapshot.rows.copy()
        richer.loc[:, ["bid", "ask"]] *= 2.0
        exit_snapshot = ChainSnapshot(
            "TEST", pd.Timestamp("2024-05-03"), pd.Timestamp("2024-05-02"), richer, 100.0
        )
        closed = price_structure(spec, exit_snapshot, MID, pin=opened.legs, closing=True)
        result = structure_return(opened, closed)
        assert result["ret"] == pytest.approx(1.0)  # doubled in value

    @staticmethod
    def _pair(entry_cash_flows, exit_cash_flows, exit_value=1.0):
        """Two StructurePrice objects with hand-set leg cash flows.

        Bypasses chain resolution entirely so the entry cost can be pinned to
        an exact floating-point value — the only way to reproduce the
        cancellation EXP-121 found without needing a real chain that happens
        to cancel to 1e-16.
        """
        def legs(cash_flows):
            return tuple(
                ResolvedLeg(
                    name=f"l{i}", right="P", side="buy" if cf < 0 else "sell",
                    qty=1.0, expiry=pd.Timestamp("2024-05-17"), strike=100.0,
                    dte=15, bid=1.0, ask=1.0, price=abs(cf), cash_flow=cf,
                    wide_market=False,
                )
                for i, cf in enumerate(cash_flows)
            )
        entry = StructurePrice("CND-P", "TEST", pd.Timestamp("2024-05-01"),
                               pd.Timestamp("2024-05-02"), 100.0, 1.0, legs(entry_cash_flows))
        exit_ = StructurePrice("CND-P", "TEST", pd.Timestamp("2024-05-03"),
                               pd.Timestamp("2024-05-02"), 100.0, 1.0,
                               legs(exit_cash_flows), closing=True)
        return entry, exit_

    def test_a_floating_point_noise_cost_returns_nan_not_a_huge_number(self):
        """Two legs whose cash flows nearly cancel (net cost ~1e-16) is the
        exact mechanism EXP-121 found on CND-P's best-fill column: 65 of
        18,388 trades costing between 1e-17 and 2e-15, producing a mean return
        in the quadrillions of percent. `ret` must be NaN, not astronomical —
        the true cost here is unknowable to be positive at all, so it is the
        same "no meaningful denominator" case as an exact zero.
        """
        # -(0.1+0.2) + 0.3 = -5.551115123125783e-17, the textbook float64
        # residual — the same class of artifact as EXP-121's 1e-17..2e-15 band.
        entry, exit_ = self._pair([-(0.1 + 0.2), 0.3], [10.0, -8.0])
        result = structure_return(entry, exit_)
        assert 0 < result["cost"] < 1e-9  # the float-noise cost, not exactly 0
        assert result["ret"] != result["ret"]  # NaN

    def test_a_real_penny_cost_still_returns_a_real_number(self):
        """The floor must not swallow a legitimately cheap structure — the
        cleanest real value above the noise band, from the same data, is
        exactly $0.01."""
        entry, exit_ = self._pair([-5.0, 4.99], [10.0, -7.99])
        result = structure_return(entry, exit_)
        assert result["cost"] == pytest.approx(0.01)
        assert result["ret"] == pytest.approx(200.0)  # pnl 2.00 / cost 0.01


class TestSessionAwareExpiry:
    """An expiry on the event date survives a BMO print but not an AMC one."""

    @pytest.fixture
    def rows(self):
        obs = pd.Timestamp("2024-05-06")
        out = []
        for expiry, dte in ((pd.Timestamp("2024-05-08"), 2), (pd.Timestamp("2024-05-10"), 4)):
            for right in ("C", "P"):
                out.append({
                    "ticker": "TEST", "obs_date": obs, "expiry": expiry, "dte": dte,
                    "strike": 100.0, "right": right, "bid": 1.0, "ask": 1.2,
                    "mid": 1.1, "iv": 0.4, "delta": 0.5, "spot": 100.0,
                })
        return pd.DataFrame(out)

    def test_bmo_may_use_an_expiry_on_the_event_date(self, rows):
        # The announcement lands before the open, so the expiry outlives it.
        chosen = ExpirySelector(kind="first_post_event").select(
            rows, pd.Timestamp("2024-05-08"), "BMO"
        )
        assert chosen == pd.Timestamp("2024-05-08")

    def test_amc_must_skip_an_expiry_on_the_event_date(self, rows):
        # The announcement lands after the close; that option is already dead.
        chosen = ExpirySelector(kind="first_post_event").select(
            rows, pd.Timestamp("2024-05-08"), "AMC"
        )
        assert chosen == pd.Timestamp("2024-05-10")

    def test_an_unknown_session_keeps_the_permissive_legacy_rule(self, rows):
        chosen = ExpirySelector(kind="first_post_event").select(
            rows, pd.Timestamp("2024-05-08"), None
        )
        assert chosen == pd.Timestamp("2024-05-08")

    def test_amc_with_nothing_left_raises_rather_than_booking_a_dead_option(self, rows):
        only_same_day = rows[rows["expiry"] == pd.Timestamp("2024-05-08")]
        with pytest.raises(StructureError, match="dies at the close"):
            ExpirySelector(kind="first_post_event").select(
                only_same_day, pd.Timestamp("2024-05-08"), "AMC"
            )

    def test_the_snapshot_threads_session_into_pricing(self, rows):
        snapshot = ChainSnapshot(
            ticker="TEST",
            obs_date=pd.Timestamp("2024-05-06"),
            event_date=pd.Timestamp("2024-05-08"),
            rows=rows,
            spot=100.0,
            session="AMC",
        )
        priced = price_structure(straddle_through(), snapshot, MID)
        assert priced.leg("call").expiry == pd.Timestamp("2024-05-10")


# --------------------------------------------------------------------------
# CND-P — the long put condor
# --------------------------------------------------------------------------


@pytest.fixture
def condor_rows():
    """A one-expiry put/call ladder on a uniform $2.50 grid, spot 101.

    Spot sits between two listed strikes on purpose: the two shorts have to
    straddle it, and a spot that lands exactly on a grid point would hide a
    boundary error in `bracket`.
    """
    obs = pd.Timestamp("2024-05-01")
    expiry, dte = pd.Timestamp("2024-05-17"), 16
    rows = []
    strike = 80.0
    while strike <= 120.0 + 1e-9:
        for right in ("C", "P"):
            intrinsic = max(strike - 101.0, 0.0) if right == "P" else max(101.0 - strike, 0.0)
            mid = intrinsic + 2.0
            rows.append(
                {
                    "ticker": "TEST", "obs_date": obs, "expiry": expiry, "dte": dte,
                    "strike": round(strike, 4), "right": right,
                    "bid": round(mid - 0.2, 4), "ask": round(mid + 0.2, 4),
                    "iv": 0.5, "delta": -0.5 if right == "P" else 0.5, "spot": 101.0,
                }
            )
        strike += 2.5
    return pd.DataFrame(rows)


@pytest.fixture
def condor_snapshot(condor_rows):
    return ChainSnapshot(
        ticker="TEST", obs_date=pd.Timestamp("2024-05-01"),
        event_date=pd.Timestamp("2024-05-02"), rows=condor_rows,
        spot=101.0, session="AMC",
    )


def _condor_strikes(price):
    return {leg.name: leg.strike for leg in price.legs}


def _terminal_payoff(price, spot_at_expiry: float) -> float:
    """Intrinsic value of the whole structure at expiry, long-positive."""
    total = 0.0
    for leg in price.legs:
        intrinsic = max(leg.strike - spot_at_expiry, 0.0)
        total += intrinsic * leg.qty * (1.0 if leg.side == "buy" else -1.0)
    return total


class TestPutCondor:
    def test_registered_under_its_strategy_code(self):
        assert STRUCTURES["CND-P"] is put_condor
        assert put_condor().name == "CND-P"

    def test_four_put_legs_one_expiry_two_short_two_long(self, condor_snapshot):
        price = price_structure(put_condor(), condor_snapshot, MID)
        assert [leg.right for leg in price.legs] == ["P"] * 4
        assert len({leg.expiry for leg in price.legs}) == 1
        assert sorted(leg.side for leg in price.legs) == ["buy", "buy", "sell", "sell"]

    def test_the_two_shorts_straddle_spot(self, condor_snapshot):
        k = _condor_strikes(price_structure(put_condor(), condor_snapshot, MID))
        assert k["short_lo"] <= condor_snapshot.spot_price < k["short_hi"]

    def test_strikes_are_evenly_spaced(self, condor_snapshot):
        k = _condor_strikes(price_structure(put_condor(width=0.05), condor_snapshot, MID))
        ordered = [k["long_lo"], k["short_lo"], k["short_hi"], k["long_hi"]]
        assert ordered == sorted(ordered)
        gaps = [b - a for a, b in zip(ordered, ordered[1:])]
        assert gaps[0] == pytest.approx(gaps[1])
        assert gaps[1] == pytest.approx(gaps[2])

    def test_width_snaps_to_the_listed_grid(self, condor_snapshot):
        """0.05 x 101 = 5.05, and the grid step is 2.50, so the spacing is 5.00."""
        k = _condor_strikes(price_structure(put_condor(width=0.05), condor_snapshot, MID))
        assert k["short_lo"] == pytest.approx(100.0)
        assert k["short_hi"] == pytest.approx(105.0)
        assert k["long_lo"] == pytest.approx(95.0)
        assert k["long_hi"] == pytest.approx(110.0)

    def test_a_wider_width_widens_every_gap(self, condor_snapshot):
        narrow = _condor_strikes(price_structure(put_condor(width=0.025), condor_snapshot, MID))
        wide = _condor_strikes(price_structure(put_condor(width=0.10), condor_snapshot, MID))
        assert narrow["short_hi"] - narrow["short_lo"] < wide["short_hi"] - wide["short_lo"]

    def test_width_below_one_grid_step_still_moves_a_strike(self, condor_snapshot):
        """Snapping a sub-step spacing back onto the anchor would collapse the body."""
        k = _condor_strikes(price_structure(put_condor(width=0.001), condor_snapshot, MID))
        assert k["short_hi"] > k["short_lo"]
        assert len({round(v, 6) for v in k.values()}) == 4

    def test_terminal_payoff_is_never_negative(self, condor_snapshot):
        """The defined-risk claim, as a property of the geometry rather than a hope.

        Even spacing is what makes the settlement below the bottom strike
        `(K4-K3) - (K2-K1) = 0`; an uneven condor pays a negative constant there
        and loses more than the debit no matter how it was filled.
        """
        price = price_structure(put_condor(), condor_snapshot, MID)
        grid = [0.0, 50.0, 90.0, 94.9, 95.0, 97.5, 100.0, 102.5, 105.0, 110.0, 130.0, 1000.0]
        assert min(_terminal_payoff(price, s) for s in grid) >= -1e-9

    def test_max_terminal_payoff_is_the_spacing(self, condor_snapshot):
        price = price_structure(put_condor(), condor_snapshot, MID)
        k = _condor_strikes(price)
        spacing = k["short_hi"] - k["short_lo"]
        for s in (k["short_lo"], k["short_hi"], 0.5 * (k["short_lo"] + k["short_hi"])):
            assert _terminal_payoff(price, s) == pytest.approx(spacing)

    def test_it_costs_a_debit(self, condor_snapshot):
        assert price_structure(put_condor(), condor_snapshot, MID).cost > 0

    def test_an_unlisted_mirror_strike_refuses_rather_than_approximating(self, condor_rows):
        """No 110 put ⇒ no symmetric condor. Refused, not snapped to 107.50."""
        rows = condor_rows[
            ~((condor_rows["strike"] == 110.0) & (condor_rows["right"] == "P"))
        ]
        snap = ChainSnapshot(
            ticker="TEST", obs_date=pd.Timestamp("2024-05-01"),
            event_date=pd.Timestamp("2024-05-02"), rows=rows, spot=101.0, session="AMC",
        )
        with pytest.raises(StructureError, match="not listed"):
            price_structure(put_condor(width=0.05), snap, MID)

    def test_strike_selection_stays_within_the_legs_own_right(self, condor_rows):
        """A 102.50 call with no 102.50 put must not become a put leg's strike."""
        rows = condor_rows[
            ~((condor_rows["strike"] == 102.5) & (condor_rows["right"] == "P"))
        ]
        snap = ChainSnapshot(
            ticker="TEST", obs_date=pd.Timestamp("2024-05-01"),
            event_date=pd.Timestamp("2024-05-02"), rows=rows, spot=101.0, session="AMC",
        )
        k = _condor_strikes(price_structure(put_condor(width=0.015), snap, MID))
        assert 102.5 not in {round(v, 6) for v in k.values()}

    def test_pinning_closes_on_the_same_four_contracts(self, condor_snapshot):
        entry = price_structure(put_condor(), condor_snapshot, MID)
        exit_ = price_structure(put_condor(), condor_snapshot, MID,
                                pin=entry.legs, closing=True)
        assert _condor_strikes(entry) == _condor_strikes(exit_)
        assert [leg.side for leg in exit_.legs] == ["buy", "buy", "sell", "sell"]

    def test_width_must_be_positive(self):
        with pytest.raises(ValueError, match="width must be positive"):
            put_condor(width=0.0)


class TestCrossLegStrikeReferences:
    def test_a_forward_reference_is_rejected_at_construction(self):
        expiry = ExpirySelector(kind="first_post_event")
        with pytest.raises(ValueError, match="resolves later"):
            Structure(
                name="BAD",
                legs=(
                    LegSpec("a", "P", "buy", expiry,
                            StrikeSelector("mirror", ref="b", about="b")),
                    LegSpec("b", "P", "sell", expiry, StrikeSelector("atm")),
                ),
                entry_offset=0, exit_offset=1,
            )

    def test_a_reference_to_a_leg_that_does_not_exist_is_rejected(self):
        expiry = ExpirySelector(kind="first_post_event")
        with pytest.raises(ValueError, match="is not a leg"):
            Structure(
                name="BAD",
                legs=(
                    LegSpec("a", "P", "buy", expiry, StrikeSelector("atm")),
                    LegSpec("b", "P", "sell", expiry,
                            StrikeSelector("offset_from", ref="ghost", moneyness=0.05)),
                ),
                entry_offset=0, exit_offset=1,
            )

    def test_bracket_needs_a_side(self):
        with pytest.raises(ValueError, match="bracket side must be"):
            StrikeSelector("bracket", side="sideways")

    def test_bracket_below_includes_a_strike_exactly_at_spot(self, condor_rows):
        below = StrikeSelector("bracket", side="below").select(
            condor_rows, 100.0, {}, right="P")
        above = StrikeSelector("bracket", side="above").select(
            condor_rows, 100.0, {}, right="P")
        assert below == pytest.approx(100.0)
        assert above == pytest.approx(102.5)

    def test_bracket_raises_when_spot_is_off_the_ladder(self, condor_rows):
        with pytest.raises(StructureError, match="no listed strike below"):
            StrikeSelector("bracket", side="below").select(condor_rows, 10.0, {}, right="P")

    def test_offset_from_stays_on_the_side_its_sign_points(self, condor_rows):
        """A grid that is sparse below the anchor must not pull the offset down."""
        rows = condor_rows[
            (condor_rows["strike"] >= 100.0) | (condor_rows["strike"] <= 85.0)
        ]
        chosen = StrikeSelector("offset_from", ref="a", moneyness=0.02).select(
            rows, 101.0, {"a": 100.0}, right="P")
        assert chosen > 100.0

    def test_offset_from_raises_when_the_grid_ends(self, condor_rows):
        with pytest.raises(StructureError, match="no listed strike above"):
            StrikeSelector("offset_from", ref="a", moneyness=0.05).select(
                condor_rows, 101.0, {"a": 120.0}, right="P")
