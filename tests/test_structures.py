"""Structures — resolution, pricing, and the open/close side discipline."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.fills import BEST, MID, WORST, FillModel
from engine.structures import (
    ChainSnapshot,
    ExpirySelector,
    LegSpec,
    StrikeSelector,
    Structure,
    StructureError,
    price_structure,
    put_calendar,
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

    def test_spec_round_trips_to_a_dict(self):
        spec = put_calendar(back_dte=30)
        blob = spec.to_dict()
        assert blob["name"] == "CAL-P"
        assert blob["params"]["back_dte"] == 30
        assert len(blob["legs"]) == 2


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
