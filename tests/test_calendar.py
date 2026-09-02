"""Calendar — session mapping, trading-day math, and drift detection."""
from __future__ import annotations

import pandas as pd
import pytest

from engine import calendar as cal
from engine import paths
from engine.calendar import (
    AMC,
    BMO,
    DateChange,
    PrintWindow,
    TradingCalendar,
    build_calendar,
    detect_date_changes,
    projected_trading_days,
    session_from_annc_tod,
    us_market_holidays,
)


class TestSessionMapping:
    @pytest.mark.parametrize("value", ["0900", "0830", "0700", "1159", 900, "930"])
    def test_before_noon_is_before_the_open(self, value):
        assert session_from_annc_tod(value) == BMO

    @pytest.mark.parametrize("value", ["1600", "1630", "1200", "2000", 1630])
    def test_noon_or_later_is_after_the_close(self, value):
        assert session_from_annc_tod(value) == AMC

    @pytest.mark.parametrize("value", [None, "", "  ", "none", float("nan"), "abc"])
    def test_unusable_values_yield_no_session(self, value):
        assert session_from_annc_tod(value) is None

    def test_out_of_range_times_are_rejected(self):
        assert session_from_annc_tod("9999") is None


class TestHolidayRules:
    """The generator has to reproduce real NYSE history to be trustworthy."""

    @pytest.mark.skipif(
        not paths.GSPC_DAILY.exists(),
        reason="needs the cached S&P daily series; absent in a bare clone",
    )
    def test_reproduces_the_index_series_except_unscheduled_closures(self):
        tc = cal.trading_calendar()
        observed = set(tc.days[tc.days <= tc.observed_through])
        generated = set(projected_trading_days("2005-12-31", tc.observed_through))

        closed_but_generated = {str(d.date()) for d in generated - observed}
        open_but_not_generated = {str(d.date()) for d in observed - generated}

        # The ONLY days the rules get wrong are one-off closures no rule covers.
        assert closed_but_generated == set(cal.UNSCHEDULED_CLOSURES)
        assert open_but_not_generated == set()

    def test_new_years_on_a_saturday_does_not_close_the_prior_friday(self):
        # The NYSE exception: unlike July 4 and Christmas, New Year's Day is
        # simply not observed when it lands on a Saturday. Getting this wrong
        # closes 2010-12-31 and 2021-12-31, both full trading days.
        for year in (2011, 2022):
            assert pd.Timestamp(year=year - 1, month=12, day=31) not in us_market_holidays(year)

    def test_july_fourth_on_a_saturday_moves_to_the_friday(self):
        assert pd.Timestamp("2020-07-03") in us_market_holidays(2020)

    def test_christmas_on_a_sunday_moves_to_the_monday(self):
        assert pd.Timestamp("2022-12-26") in us_market_holidays(2022)

    def test_good_friday_is_two_days_before_easter(self):
        assert pd.Timestamp("2024-03-29") in us_market_holidays(2024)

    def test_juneteenth_only_from_2022(self):
        assert pd.Timestamp("2021-06-18") not in us_market_holidays(2021)
        assert pd.Timestamp("2022-06-20") in us_market_holidays(2022)


class TestTradingCalendar:
    @pytest.fixture
    def tc(self):
        # Mon–Fri for two weeks, with Wednesday 2024-05-08 closed.
        days = [d for d in pd.date_range("2024-05-01", "2024-05-17", freq="B")
                if d != pd.Timestamp("2024-05-08")]
        return TradingCalendar(days)

    def test_shift_steps_over_holidays_and_weekends(self, tc):
        assert tc.shift("2024-05-07", 1) == pd.Timestamp("2024-05-09")
        assert tc.shift("2024-05-03", 1) == pd.Timestamp("2024-05-06")

    def test_shift_backwards(self, tc):
        assert tc.shift("2024-05-09", -1) == pd.Timestamp("2024-05-07")

    def test_shift_beyond_the_calendar_raises(self, tc):
        with pytest.raises(KeyError, match="out of range"):
            tc.shift("2024-05-17", 5)

    def test_non_trading_days_are_recognized(self, tc):
        assert not tc.is_trading_day("2024-05-08")
        assert tc.is_trading_day("2024-05-09")

    def test_index_of_a_closed_day_requires_a_side(self, tc):
        with pytest.raises(KeyError, match="not a trading day"):
            tc.index_of("2024-05-08")
        assert tc.days[tc.index_of("2024-05-08", side="prev")] == pd.Timestamp("2024-05-07")
        assert tc.days[tc.index_of("2024-05-08", side="next")] == pd.Timestamp("2024-05-09")

    # -- the session-aware anchors, which decide every entry and exit date --

    def test_bmo_prints_before_its_own_open(self, tc):
        # Announcement lands before the open of 2024-05-07, so the last
        # information-free close is 2024-05-06 and the first informed close is
        # 2024-05-07 itself.
        assert tc.last_pre_print("2024-05-07", BMO) == pd.Timestamp("2024-05-06")
        assert tc.first_post_print("2024-05-07", BMO) == pd.Timestamp("2024-05-07")

    def test_amc_prints_after_its_own_close(self, tc):
        # Announcement lands after the close of 2024-05-07, so that close is
        # still pre-print and the first informed close is the next session.
        assert tc.last_pre_print("2024-05-07", AMC) == pd.Timestamp("2024-05-07")
        assert tc.first_post_print("2024-05-07", AMC) == pd.Timestamp("2024-05-09")

    def test_anchors_step_over_a_closed_event_date(self, tc):
        assert tc.last_pre_print("2024-05-08", AMC) == pd.Timestamp("2024-05-07")
        assert tc.first_post_print("2024-05-08", BMO) == pd.Timestamp("2024-05-09")

    def test_unknown_session_raises(self, tc):
        with pytest.raises(ValueError, match="unknown session"):
            tc.last_pre_print("2024-05-07", "OVERNIGHT")

    def test_offsets_resolve_relative_to_the_print(self, tc):
        window = tc.resolve_offsets("2024-05-09", AMC, entry_offset=0, exit_offset=1)
        assert window.entry_date == pd.Timestamp("2024-05-09")  # last pre-print
        assert window.exit_date == pd.Timestamp("2024-05-10")  # first post-print

    def test_negative_offsets_step_back_from_the_pre_print_close(self, tc):
        # AMC on the 9th → pre-print close is the 9th. Two *trading* days back
        # is the 6th, not the 7th: the 8th is closed in this fixture, which is
        # exactly the arithmetic a calendar-day offset would get wrong.
        window = tc.resolve_offsets("2024-05-09", AMC, entry_offset=-2, exit_offset=1)
        assert window.entry_date == pd.Timestamp("2024-05-06")
        assert window.last_pre_print == pd.Timestamp("2024-05-09")

    def test_the_same_offsets_mean_different_dates_per_session(self, tc):
        bmo = tc.resolve_offsets("2024-05-09", BMO, 0, 1)
        amc = tc.resolve_offsets("2024-05-09", AMC, 0, 1)
        assert bmo.entry_date == pd.Timestamp("2024-05-07")
        assert amc.entry_date == pd.Timestamp("2024-05-09")
        assert bmo.exit_date == pd.Timestamp("2024-05-09")
        assert amc.exit_date == pd.Timestamp("2024-05-10")

    def test_runup_exit_lands_before_the_print(self, tc):
        window = tc.resolve_offsets("2024-05-16", AMC, entry_offset=-5, exit_offset=0)
        assert window.exit_date == window.last_pre_print
        assert window.exit_date < window.first_post_print

    def test_projected_days_are_marked(self):
        tc = TradingCalendar(
            pd.date_range("2024-05-01", "2024-05-31", freq="B"),
            observed_through="2024-05-15",
        )
        assert not tc.is_projected("2024-05-15")
        assert tc.is_projected("2024-05-16")

    def test_empty_calendar_is_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            TradingCalendar([])


class TestDecisionOffsets:
    """The decision close, which is what makes a prediction actionable.

    Every structure that shipped before this decided at its entry close, which
    for STR-THRU is the last pre-print close — a chain that does not exist until
    that session is over. `decision_offset` is how a structure buys lead time.
    """

    @pytest.fixture
    def tc(self):
        # Same calendar as TestTradingCalendar: Wednesday 2024-05-08 is closed,
        # so a one-session step back is not a one-calendar-day step back.
        days = [d for d in pd.date_range("2024-05-01", "2024-05-17", freq="B")
                if d != pd.Timestamp("2024-05-08")]
        return TradingCalendar(days)

    def test_no_decision_offset_means_decide_at_the_entry_close(self, tc):
        window = tc.resolve_offsets("2024-05-09", AMC, entry_offset=0, exit_offset=1)
        assert window.decision_date == window.entry_date

    def test_passing_none_is_the_same_as_not_passing_it(self, tc):
        implicit = tc.resolve_offsets("2024-05-09", AMC, 0, 1)
        explicit = tc.resolve_offsets("2024-05-09", AMC, 0, 1, decision_offset=None)
        assert implicit == explicit

    def test_a_decision_offset_steps_back_on_the_same_anchor(self, tc):
        window = tc.resolve_offsets("2024-05-09", AMC, 0, 1, decision_offset=-1)
        assert window.entry_date == pd.Timestamp("2024-05-09")
        assert window.decision_date == pd.Timestamp("2024-05-07")  # the 8th is closed
        assert window.exit_date == pd.Timestamp("2024-05-10")

    def test_the_decision_lands_a_session_earlier_for_both_sessions(self, tc):
        bmo = tc.resolve_offsets("2024-05-09", BMO, 0, 1, decision_offset=-1)
        amc = tc.resolve_offsets("2024-05-09", AMC, 0, 1, decision_offset=-1)
        # BMO's entry is already the day before the print, so deciding a session
        # earlier puts the decision two sessions ahead of the announcement.
        assert bmo.entry_date == pd.Timestamp("2024-05-07")
        assert bmo.decision_date == pd.Timestamp("2024-05-06")
        assert amc.entry_date == pd.Timestamp("2024-05-09")
        assert amc.decision_date == pd.Timestamp("2024-05-07")

    def test_the_decision_is_never_after_the_last_information_free_close(self, tc):
        for session in (BMO, AMC):
            window = tc.resolve_offsets("2024-05-09", session, 0, 1, decision_offset=-1)
            assert window.decision_date < window.last_pre_print

    def test_a_directly_built_window_still_has_a_decision_date(self):
        """Nothing constructs PrintWindow by hand today, but if it does the
        decision date must not silently be None and break date arithmetic."""
        day = pd.Timestamp("2024-05-09")
        window = PrintWindow(
            event_date=day, session=AMC, entry_date=day,
            exit_date=day + pd.Timedelta(days=1),
            last_pre_print=day, first_post_print=day + pd.Timedelta(days=1),
        )
        assert window.decision_date == day


class TestBuildCalendar:
    def test_agreement_flag_marks_doubly_confirmed_events(self):
        orats = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB"],
                "event_date": pd.to_datetime(["2024-01-15", "2024-04-15", "2024-02-01"]),
                "annc_tod": ["1630", "0900", "1630"],
                "session": [AMC, BMO, AMC],
                "updated_at": [None, None, None],
            }
        )
        oquants = pd.DataFrame(
            {
                "ticker": ["AAA", "CCC"],
                "event_date": pd.to_datetime(["2024-01-15", "2024-03-01"]),
            }
        )
        out = build_calendar(orats=orats, oquants=oquants)

        assert len(out) == 4
        both = out[out["date_agree"]]
        assert len(both) == 1
        assert both.iloc[0]["ticker"] == "AAA"

        # ORATS-only events are kept (its calendar reaches back to the 1980s)…
        assert (out["src_orats"] & ~out["src_oquants"]).sum() == 2
        # …and so are oquants-only ones, flagged rather than dropped.
        assert (~out["src_orats"] & out["src_oquants"]).sum() == 1

    def test_event_id_is_ticker_and_date(self):
        orats = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "event_date": pd.to_datetime(["2024-01-15"]),
                "annc_tod": ["1630"],
                "session": [AMC],
                "updated_at": [None],
            }
        )
        out = build_calendar(orats=orats, oquants=pd.DataFrame(columns=["ticker", "event_date"]))
        assert out.iloc[0]["event_id"] == "AAA_2024-01-15"


class TestDateChangeDetection:
    """Stale earnings dates are a known loss source, so drift must be visible."""

    def _frame(self, rows):
        return pd.DataFrame(rows, columns=["ticker", "event_date", "session"])

    def test_a_moved_date_is_flagged(self):
        soon = pd.Timestamp.today().normalize() + pd.Timedelta(days=10)
        previous = self._frame([["AAA", soon, AMC]])
        current = self._frame([["AAA", soon + pd.Timedelta(days=3), AMC]])
        changes = detect_date_changes(previous, current)
        assert [c.kind for c in changes] == ["moved"]
        assert changes[0].ticker == "AAA"

    def test_added_and_removed_events_are_flagged(self):
        soon = pd.Timestamp.today().normalize() + pd.Timedelta(days=5)
        previous = self._frame([["OLD", soon, AMC]])
        current = self._frame([["NEW", soon, AMC]])
        kinds = {(c.ticker, c.kind) for c in detect_date_changes(previous, current)}
        assert kinds == {("OLD", "removed"), ("NEW", "added")}

    def test_a_session_flip_is_flagged(self):
        soon = pd.Timestamp.today().normalize() + pd.Timedelta(days=7)
        previous = self._frame([["AAA", soon, AMC]])
        current = self._frame([["AAA", soon, BMO]])
        changes = detect_date_changes(previous, current)
        assert [c.kind for c in changes] == ["session_changed"]

    def test_unchanged_calendars_produce_no_noise(self):
        soon = pd.Timestamp.today().normalize() + pd.Timedelta(days=7)
        frame = self._frame([["AAA", soon, AMC]])
        assert detect_date_changes(frame, frame.copy()) == []

    def test_events_outside_the_horizon_are_ignored(self):
        far = pd.Timestamp.today().normalize() + pd.Timedelta(days=300)
        previous = self._frame([["AAA", far, AMC]])
        current = self._frame([["AAA", far + pd.Timedelta(days=5), AMC]])
        assert detect_date_changes(previous, current) == []

    def test_past_events_are_ignored(self):
        past = pd.Timestamp.today().normalize() - pd.Timedelta(days=10)
        previous = self._frame([["AAA", past, AMC]])
        current = self._frame([["AAA", past + pd.Timedelta(days=2), AMC]])
        assert detect_date_changes(previous, current) == []

    def test_empty_inputs_are_handled(self):
        empty = self._frame([])
        assert detect_date_changes(empty, empty) == []


class TestEventsNormalizer:
    """The Tier-2 adapter over `build_calendar` (engine/data/normalize/n_events)."""

    def test_it_shapes_the_calendar_to_the_tier_2_schema(self, monkeypatch):
        from engine.data.normalize import n_events
        from engine.data.schemas import SCHEMAS, assert_schema, coerce

        cal = pd.DataFrame(
            {
                "event_id": ["AAA_2024-01-15", "BBB_2024-02-01"],
                "ticker": ["AAA", "BBB"],
                "event_date": pd.to_datetime(["2024-01-15", "2024-02-01"]),
                "session": [AMC, BMO],
                "annc_tod": ["1630", "0900"],
                "session_src": ["orats", "orats"],
                "src_orats": [True, True],
                "src_oquants": [True, False],
                "src_nasdaq": [False, False],
                "src_yfinance": [False, False],
                "date_agree": [True, False],
                "date_conflict": [False, False],
                "updated_at": [None, None],
            }
        )
        monkeypatch.setattr(n_events, "build_calendar", lambda **kw: cal)
        out, report = n_events.normalize()

        assert_schema(coerce(out, "earnings_events"), "earnings_events")
        assert report["rows"] == 2
        assert report["both_sources"] == 1
        assert report["orats_only"] == 1
        assert set(out["year"]) == {2024}

    def test_an_empty_calendar_reports_zero_rather_than_raising(self, monkeypatch):
        from engine.data.normalize import n_events

        monkeypatch.setattr(
            n_events, "build_calendar", lambda **kw: pd.DataFrame(columns=["ticker"])
        )
        out, report = n_events.normalize()
        assert out.empty and report["rows"] == 0

    def test_session_coverage_counts_both_sessions_and_gaps(self, monkeypatch):
        from engine.data.normalize import n_events

        events = pd.DataFrame({"session": [AMC, AMC, BMO, None]})
        counts = n_events.session_coverage(events)
        assert counts["AMC"] == 2
        assert counts["BMO"] == 1


class TestForwardCalendar:
    """The sources that see ahead. ORATS /hist/earnings does not: its payloads
    stop at the last print that already happened, so without these the
    monitoring board has nothing to score."""

    def _orats(self):
        return pd.DataFrame(
            {
                "ticker": ["AAA"],
                "event_date": pd.to_datetime(["2024-01-15"]),
                "annc_tod": ["1630"],
                "session": [AMC],
                "updated_at": [None],
            }
        )

    def _forward(self, ticker="AAA", date="2026-09-10", session=None):
        return pd.DataFrame(
            {
                "ticker": [ticker],
                "event_date": pd.to_datetime([date]),
                "annc_tod": [None],
                "session": [session],
                "updated_at": ["2026-08-30T00:00:00Z"],
            }
        )

    def test_a_forward_source_adds_events_orats_cannot_have(self):
        out = build_calendar(
            orats=self._orats(),
            oquants=pd.DataFrame(columns=["ticker", "event_date"]),
            nasdaq=self._forward(session="AMC"),
        )
        assert len(out) == 2
        row = out[out["event_date"] == pd.Timestamp("2026-09-10")].iloc[0]
        assert row["session"] == "AMC"
        assert row["session_src"] == "nasdaq"
        assert bool(row["src_nasdaq"]) and not bool(row["src_orats"])

    def test_orats_outranks_the_forward_sources_on_session(self):
        """Once ORATS carries the event, its anncTod is the answer — a forward
        guess must not survive alongside the authority."""
        contested = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "event_date": pd.to_datetime(["2024-01-15"]),
                "annc_tod": [None],
                "session": [BMO],  # disagrees with the ORATS AMC above
                "updated_at": [None],
            }
        )
        out = build_calendar(
            orats=self._orats(),
            oquants=pd.DataFrame(columns=["ticker", "event_date"]),
            nasdaq=contested,
            yfinance=contested,
        )
        row = out.iloc[0]
        assert row["session"] == AMC
        assert row["session_src"] == "orats"
        assert row["annc_tod"] == "1630"

    def test_yfinance_outranks_nasdaq(self):
        """yfinance keeps the announcement TIME on historical rows, so its
        session is gradeable after the fact; Nasdaq's never is."""
        out = build_calendar(
            orats=pd.DataFrame(columns=["ticker", "event_date", "annc_tod", "session", "updated_at"]),
            nasdaq=self._forward(session="BMO"),
            yfinance=self._forward(session="AMC"),
        )
        assert out.iloc[0]["session"] == "AMC"
        assert out.iloc[0]["session_src"] == "yfinance"

    def test_a_session_nobody_supplies_stays_null(self):
        """`time-not-supplied` must not become a guess: the scorer skips events
        with no session, and a wrong one shifts entry and exit by a day."""
        out = build_calendar(
            orats=pd.DataFrame(columns=["ticker", "event_date", "annc_tod", "session", "updated_at"]),
            nasdaq=self._forward(session=None),
        )
        assert pd.isna(out.iloc[0]["session"])
        assert pd.isna(out.iloc[0]["session_src"])

    def test_rival_forward_dates_are_flagged_and_both_kept(self):
        out = build_calendar(
            orats=pd.DataFrame(columns=["ticker", "event_date", "annc_tod", "session", "updated_at"]),
            nasdaq=self._forward(date="2026-09-02"),
            yfinance=self._forward(date="2026-09-08", session="AMC"),
        )
        assert len(out) == 2, "a disagreement must never be resolved by dropping a row"
        assert out["date_conflict"].all()

    def test_a_normal_quarterly_cadence_is_not_a_conflict(self):
        out = build_calendar(
            orats=pd.DataFrame(columns=["ticker", "event_date", "annc_tod", "session", "updated_at"]),
            nasdaq=pd.concat([self._forward(date="2026-09-02"), self._forward(date="2026-12-02")]),
        )
        assert not out["date_conflict"].any()

    def test_an_orats_confirmed_event_is_never_a_conflict(self):
        """History is left alone: a fiscal-calendar change really can put two
        prints close together, and ORATS is the authority on what happened."""
        orats = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "event_date": pd.to_datetime(["2024-01-15", "2024-02-05"]),
                "annc_tod": ["1630", "1630"],
                "session": [AMC, AMC],
                "updated_at": [None, None],
            }
        )
        out = build_calendar(orats=orats)
        assert not out["date_conflict"].any()

    def test_explicit_frames_do_not_pull_the_local_cache(self):
        """A caller handing over frames wants a closed, reproducible world."""
        out = build_calendar(orats=self._orats())
        assert len(out) == 1
