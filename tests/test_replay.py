"""The replay path — the one place a structure meets real quotes.

Everything downstream (the analog layer, the payoff calibration, the gate's
training target, Phase 2's backtests) inherits whatever this produces, so the
arithmetic is checked against hand-computed numbers rather than against itself.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from engine import replay
from engine.calendar import TradingCalendar
from engine.structures import put_calendar, straddle_runup, straddle_through


@pytest.fixture
def calendar():
    """Consecutive weekdays around the test event.

    Long enough before it for STR-RUNUP's fourteen-trading-day entry to land
    inside the calendar rather than off the front of it.
    """
    return TradingCalendar(pd.bdate_range("2024-03-01", periods=60))


@pytest.fixture
def events():
    return pd.DataFrame(
        [
            {"event_id": "TEST_2024-05-02", "ticker": "TEST",
             "event_date": pd.Timestamp("2024-05-02"), "session": "AMC"},
            {"event_id": "BMO_2024-05-02", "ticker": "BMO",
             "event_date": pd.Timestamp("2024-05-02"), "session": "BMO"},
        ]
    )


def chain(ticker, obs_date, *, call=(2.0, 2.4), put=(1.0, 1.4), spot=100.0):
    """Two expiries × three strikes, round numbers so P&L is checkable by hand."""
    rows = []
    obs = pd.Timestamp(obs_date)
    for expiry, dte in ((pd.Timestamp("2024-05-03"), 2), (pd.Timestamp("2024-05-24"), 23)):
        for strike in (95.0, 100.0, 105.0):
            for right, (bid, ask) in (("C", call), ("P", put)):
                scale = 1.0 if dte < 10 else 2.0
                rows.append(
                    {
                        "ticker": ticker, "obs_date": obs, "expiry": expiry, "dte": dte,
                        "strike": strike, "right": right,
                        "bid": bid * scale, "ask": ask * scale,
                        "spot": spot, "quote_repaired": False,
                    }
                )
    return pd.DataFrame(rows)


class TestPlanEvents:
    def test_resolves_both_sessions(self, events, calendar):
        plan = replay.plan_events(straddle_through(), events, calendar=calendar)
        rows = plan.frame.set_index("ticker")
        # AMC: announcement lands after the 05-02 close, so that close is the
        # last information-free one and the exit is the next day.
        assert rows.loc["TEST", "entry_date"] == pd.Timestamp("2024-05-02")
        assert rows.loc["TEST", "exit_date"] == pd.Timestamp("2024-05-03")
        # BMO: announcement lands before the 05-02 open, so entry is 05-01.
        assert rows.loc["BMO", "entry_date"] == pd.Timestamp("2024-05-01")
        assert rows.loc["BMO", "exit_date"] == pd.Timestamp("2024-05-02")

    def test_skips_events_without_a_session_rather_than_guessing(self, calendar):
        events = pd.DataFrame(
            [{"event_id": "X_2024-05-02", "ticker": "X",
              "event_date": pd.Timestamp("2024-05-02"), "session": None}]
        )
        plan = replay.plan_events(straddle_through(), events, calendar=calendar)
        assert plan.frame.empty
        assert plan.skipped["no_session"] == 1

    def test_runup_enters_fourteen_trading_days_early(self, events, calendar):
        plan = replay.plan_events(straddle_runup(), events, calendar=calendar)
        row = plan.frame.set_index("ticker").loc["TEST"]
        assert row["exit_date"] == pd.Timestamp("2024-05-02")
        gap = calendar.days.get_loc(row["exit_date"]) - calendar.days.get_loc(row["entry_date"])
        assert gap == 14

    def test_out_of_range_events_are_counted(self, calendar):
        events = pd.DataFrame(
            [{"event_id": "X_2030-01-02", "ticker": "X",
              "event_date": pd.Timestamp("2030-01-02"), "session": "AMC"}]
        )
        plan = replay.plan_events(straddle_through(), events, calendar=calendar)
        assert plan.skipped["calendar_out_of_range"] == 1

    def test_chain_keys_cover_entry_and_exit(self, events, calendar):
        plan = replay.plan_events(straddle_through(), events, calendar=calendar)
        assert ("TEST", pd.Timestamp("2024-05-02")) in plan.chain_keys
        assert ("TEST", pd.Timestamp("2024-05-03")) in plan.chain_keys


class TestReplayOne:
    def build_index(self, *, exit_call=(2.0, 2.4), exit_put=(1.0, 1.4)):
        return replay.ChainIndex(
            {
                ("TEST", pd.Timestamp("2024-05-02")): chain("TEST", "2024-05-02"),
                ("TEST", pd.Timestamp("2024-05-03")): chain(
                    "TEST", "2024-05-03", call=exit_call, put=exit_put
                ),
            }
        )

    def plan_row(self):
        return {
            "event_id": "TEST_2024-05-02", "ticker": "TEST",
            "event_date": pd.Timestamp("2024-05-02"), "session": "AMC",
            "entry_date": pd.Timestamp("2024-05-02"),
            "exit_date": pd.Timestamp("2024-05-03"),
        }

    def test_prices_the_known_arithmetic(self):
        """Hand-checked: STR-THRU ATM at the 2-DTE expiry, worst fills.

        Entry buys both legs at the ask: 2.4 + 1.4 = 3.8.
        Exit sells both at the bid:      2.0 + 1.0 = 3.0.
        """
        rows, reason = replay.replay_one(
            straddle_through(), self.plan_row(), self.build_index(), alphas=[0.0]
        )
        assert reason is None
        row = rows[0]
        assert row["entry_cost"] == pytest.approx(3.8)
        assert row["exit_value"] == pytest.approx(3.0)
        assert row["ret"] == pytest.approx(3.0 / 3.8 - 1)
        assert row["strike"] == 100.0
        assert row["dte_entry"] == 2

    def test_mid_fills_beat_worst_fills(self):
        rows, _ = replay.replay_one(
            straddle_through(), self.plan_row(), self.build_index(),
            alphas=[0.0, 0.5, 1.0],
        )
        returns = [r["ret"] for r in rows]
        assert returns[0] < returns[1] < returns[2]
        # mid: entry (2.2 + 1.2) = 3.4, exit the same → flat.
        assert rows[1]["entry_cost"] == pytest.approx(3.4)
        assert rows[1]["ret"] == pytest.approx(0.0)

    def test_every_alpha_prices_the_same_contracts(self):
        """The alpha sweep must compare one trade, not five different ones."""
        rows, _ = replay.replay_one(
            straddle_through(), self.plan_row(), self.build_index(), alphas=[0.0, 1.0]
        )
        assert len({r["strike"] for r in rows}) == 1
        assert len({r["expiry"] for r in rows}) == 1

    def test_a_missing_exit_chain_is_reported_not_invented(self):
        index = replay.ChainIndex(
            {("TEST", pd.Timestamp("2024-05-02")): chain("TEST", "2024-05-02")}
        )
        rows, reason = replay.replay_one(straddle_through(), self.plan_row(), index)
        assert rows == []
        assert reason == "no_exit_chain"

    def test_a_missing_entry_chain_is_reported(self):
        index = replay.ChainIndex(
            {("TEST", pd.Timestamp("2024-05-03")): chain("TEST", "2024-05-03")}
        )
        rows, reason = replay.replay_one(straddle_through(), self.plan_row(), index)
        assert reason == "no_entry_chain"

    def test_unquoted_strikes_are_dropped_not_fatal(self):
        entry = chain("TEST", "2024-05-02")
        entry.loc[entry["strike"] == 105.0, ["bid", "ask"]] = np.nan
        index = replay.ChainIndex(
            {
                ("TEST", pd.Timestamp("2024-05-02")): entry,
                ("TEST", pd.Timestamp("2024-05-03")): chain("TEST", "2024-05-03"),
            }
        )
        rows, reason = replay.replay_one(straddle_through(), self.plan_row(), index)
        assert reason is None and rows

    def test_records_spot_and_dte_in_the_legs_blob(self):
        rows, _ = replay.replay_one(
            straddle_through(), self.plan_row(), self.build_index(), alphas=[0.5]
        )
        doc = json.loads(rows[0]["legs"])
        assert doc["spot_entry"] == 100.0
        assert doc["dte_entry"] == 2
        assert len(doc["entry"]) == 2 and len(doc["exit"]) == 2

    def test_exit_legs_transact_the_opposite_side(self):
        rows, _ = replay.replay_one(
            straddle_through(), self.plan_row(), self.build_index(), alphas=[0.0]
        )
        doc = json.loads(rows[0]["legs"])
        assert {leg["side"] for leg in doc["entry"]} == {"buy"}
        assert {leg["side"] for leg in doc["exit"]} == {"sell"}

    def test_a_credit_structure_is_skipped_not_booked(self):
        """Return-on-debit is meaningless without a debit.

        A put calendar whose near-dated short leg fetches more than its back leg
        costs opens for a credit. It is a legitimate trade — it is simply not one
        whose P&L can be quoted as a return on the debit every other metric in
        the program uses, so it is counted rather than booked with a nonsense
        denominator.
        """
        entry = chain("TEST", "2024-05-02")
        near, far = entry["dte"] == 2, entry["dte"] == 23
        puts = entry["right"] == "P"
        entry.loc[near & puts, ["bid", "ask"]] = [5.0, 5.5]
        entry.loc[far & puts, ["bid", "ask"]] = [0.01, 0.02]
        index = replay.ChainIndex(
            {
                ("TEST", pd.Timestamp("2024-05-02")): entry,
                ("TEST", pd.Timestamp("2024-05-03")): chain("TEST", "2024-05-03"),
            }
        )
        rows, reason = replay.replay_one(put_calendar(back_dte=20), self.plan_row(), index)
        assert rows == []
        assert reason == "zero_cost"

    def test_a_floating_point_noise_cost_is_also_skipped_as_zero_cost(self, monkeypatch):
        """The orchestration side of the MIN_MEANINGFUL_COST fix (EXP-121):
        `structure_return`'s own arithmetic is covered in test_structures.py —
        this checks that replay_one treats a near-zero-but-POSITIVE cost the
        same way it treats an exact zero or a credit, rather than booking it
        with a denominator that produces a quadrillion-percent return.
        """
        import engine.replay as replay_mod

        real = replay_mod.structure_return
        monkeypatch.setattr(
            replay_mod, "structure_return",
            lambda entry, exit_: real(entry, exit_) | {"cost": 5.55e-17},
        )
        entry = chain("TEST", "2024-05-02")
        index = replay.ChainIndex(
            {
                ("TEST", pd.Timestamp("2024-05-02")): entry,
                ("TEST", pd.Timestamp("2024-05-03")): chain("TEST", "2024-05-03"),
            }
        )
        rows, reason = replay.replay_one(put_calendar(back_dte=20), self.plan_row(), index)
        assert rows == []
        assert reason == "zero_cost"

    def test_wide_markets_are_flagged(self):
        entry = chain("TEST", "2024-05-02", call=(0.1, 5.0), put=(0.1, 5.0))
        index = replay.ChainIndex(
            {
                ("TEST", pd.Timestamp("2024-05-02")): entry,
                ("TEST", pd.Timestamp("2024-05-03")): chain("TEST", "2024-05-03"),
            }
        )
        rows, _ = replay.replay_one(
            straddle_through(), self.plan_row(), index, alphas=[0.5]
        )
        assert rows[0]["wide_market"] is True


class TestDecisionDates:
    """The plan has to carry the decision close, and carry it inertly.

    Nothing in the shipped structures sets a decision offset yet, so the whole
    point of these tests is that the new column changes no counts, no keys and
    no labels until a structure actually asks for it.
    """

    def test_the_plan_carries_a_decision_date(self, events, calendar):
        plan = replay.plan_events(straddle_through(), events, calendar=calendar)
        assert "decision_date" in plan.frame.columns
        assert (plan.frame["decision_date"] == plan.frame["entry_date"]).all()

    def test_an_early_decision_moves_only_that_column(self, events, calendar):
        base = replay.plan_events(straddle_through(), events, calendar=calendar)
        early = replay.plan_events(
            straddle_through(decision_offset=-1), events, calendar=calendar
        )
        assert list(early.frame["entry_date"]) == list(base.frame["entry_date"])
        assert list(early.frame["exit_date"]) == list(base.frame["exit_date"])
        assert (early.frame["decision_date"] < early.frame["entry_date"]).all()

    def test_no_extra_chains_are_loaded_when_the_decision_is_the_entry(
        self, events, calendar
    ):
        plan = replay.plan_events(straddle_through(), events, calendar=calendar)
        # 2 events × (entry, exit), with no third date to fetch.
        assert len(plan.chain_keys) == 4

    def test_an_early_decision_asks_for_a_third_chain(self, events, calendar):
        plan = replay.plan_events(
            straddle_through(decision_offset=-1), events, calendar=calendar
        )
        assert len(plan.chain_keys) == 6
        for row in plan.frame.itertuples():
            assert (row.ticker, row.decision_date) in plan.chain_keys

    def test_a_missing_decision_chain_is_counted_separately(self, events, calendar):
        plan = replay.plan_events(
            straddle_through(decision_offset=-1), events, calendar=calendar
        )
        row = plan.frame[plan.frame["ticker"] == "TEST"].iloc[0]
        available = {("TEST", row["entry_date"]), ("TEST", row["exit_date"])}
        filtered = replay.filter_plan_by_availability(plan, available)
        assert filtered.frame.empty
        assert filtered.skipped["no_decision_chain"] == 1
        # Not double-counted: TEST had both of the chains those reasons name.
        assert filtered.skipped["no_entry_chain"] == 1  # the BMO event only
        assert filtered.skipped["no_exit_chain"] == 0

    def test_no_decision_chain_stays_zero_for_a_same_close_decision(
        self, events, calendar
    ):
        plan = replay.plan_events(straddle_through(), events, calendar=calendar)
        filtered = replay.filter_plan_by_availability(plan, set())
        assert filtered.skipped["no_decision_chain"] == 0

    def test_the_variant_label_distinguishes_the_two_books(self):
        # `e+0_x+1` already names a book in the trades table; it must not come
        # to mean two different trade sets.
        assert replay._variant_label(straddle_through()) == "e+0_x+1"
        assert replay._variant_label(straddle_through(decision_offset=0)) == "e+0_x+1"
        assert replay._variant_label(straddle_through(decision_offset=-1)) == "e+0_x+1_d-1"


class TestFilterByAvailability:
    def test_counts_missing_chains_under_the_pricing_reasons(self, events, calendar):
        plan = replay.plan_events(straddle_through(), events, calendar=calendar)
        available = {("TEST", pd.Timestamp("2024-05-02"))}  # entry only
        filtered = replay.filter_plan_by_availability(plan, available)
        assert filtered.frame.empty
        assert filtered.skipped["no_entry_chain"] == 1  # the BMO event
        assert filtered.skipped["no_exit_chain"] == 1  # TEST has entry, no exit

    def test_keeps_events_with_both(self, events, calendar):
        plan = replay.plan_events(straddle_through(), events, calendar=calendar)
        available = {
            ("TEST", pd.Timestamp("2024-05-02")),
            ("TEST", pd.Timestamp("2024-05-03")),
        }
        filtered = replay.filter_plan_by_availability(plan, available)
        assert list(filtered.frame["ticker"]) == ["TEST"]


class TestReplay:
    def test_end_to_end_with_a_supplied_index(self, events, calendar):
        index = replay.ChainIndex(
            {
                ("TEST", pd.Timestamp("2024-05-02")): chain("TEST", "2024-05-02"),
                ("TEST", pd.Timestamp("2024-05-03")): chain("TEST", "2024-05-03"),
                ("BMO", pd.Timestamp("2024-05-01")): chain("BMO", "2024-05-01"),
                ("BMO", pd.Timestamp("2024-05-02")): chain("BMO", "2024-05-02"),
            }
        )
        result = replay.replay(
            "STR-THRU", events, calendar=calendar, index=index, alphas=[0.0, 0.5, 1.0]
        )
        assert result.n_trades == 2
        assert len(result.trades) == 6  # 2 events × 3 alphas
        assert result.planned == 2
        assert set(result.trades["strategy"]) == {"STR-THRU"}

    def test_variant_label_is_stable_and_descriptive(self):
        assert replay._variant_label(straddle_through()) == "e+0_x+1"
        assert "target_dte=30" in replay._variant_label(straddle_runup())

    def test_unknown_strategy_raises(self, events):
        with pytest.raises(KeyError, match="unknown strategy"):
            replay.replay("NOPE", events)


class TestToTradesTable:
    def test_conforms_to_the_tier2_schema(self, events, calendar):
        from engine.data.schemas import assert_schema, coerce

        index = replay.ChainIndex(
            {
                ("TEST", pd.Timestamp("2024-05-02")): chain("TEST", "2024-05-02"),
                ("TEST", pd.Timestamp("2024-05-03")): chain("TEST", "2024-05-03"),
            }
        )
        result = replay.replay(
            "STR-THRU", events, calendar=calendar, index=index, alphas=[0.0, 0.5]
        )
        table = replay.to_trades_table([result])
        assert_schema(coerce(table, "trades"), "trades")

    def test_alpha_is_part_of_the_trade_identity(self, events, calendar):
        index = replay.ChainIndex(
            {
                ("TEST", pd.Timestamp("2024-05-02")): chain("TEST", "2024-05-02"),
                ("TEST", pd.Timestamp("2024-05-03")): chain("TEST", "2024-05-03"),
            }
        )
        result = replay.replay(
            "STR-THRU", events, calendar=calendar, index=index, alphas=[0.0, 0.5, 1.0]
        )
        table = replay.to_trades_table([result])
        assert table["trade_id"].nunique() == len(table)
        assert table["trade_id"].str.endswith("a50").sum() == 1

    def test_empty_results_produce_an_empty_typed_frame(self):
        table = replay.to_trades_table([])
        assert table.empty
        assert "trade_id" in table.columns
