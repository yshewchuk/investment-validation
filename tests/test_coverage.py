"""Coverage analysis — the measurement the Sep-1 pull budget is spent on."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.calendar import AMC, BMO, TradingCalendar
from engine.data import coverage
from engine.data.coverage import MCAP_BUCKETS, bucket_mcap, side_coverage


class TestMcapBuckets:
    def test_the_plan_slices_are_the_declared_ones(self):
        assert [label for label, _, _ in MCAP_BUCKETS] == ["<1B", "1-10B", ">10B"]

    def test_values_land_in_the_right_slice(self):
        got = bucket_mcap([5e8, 5e9, 5e10])
        assert list(got) == ["<1B", "1-10B", ">10B"]

    def test_boundaries_are_left_inclusive(self):
        assert list(bucket_mcap([1e9, 1e10])) == ["1-10B", ">10B"]

    def test_missing_size_is_unknown_rather_than_guessed(self):
        assert list(bucket_mcap([np.nan, None])) == ["unknown", "unknown"]


class TestEventChainCoverage:
    @pytest.fixture
    def calendar(self, monkeypatch):
        days = pd.bdate_range("2024-01-01", "2024-12-31")
        cal = TradingCalendar(days)
        monkeypatch.setattr(coverage, "trading_calendar", lambda *a, **k: cal)
        return cal

    @pytest.fixture
    def events(self):
        return pd.DataFrame(
            {
                "event_id": ["AAA_2024-05-08", "BBB_2024-05-08"],
                "ticker": ["AAA", "BBB"],
                "event_date": pd.to_datetime(["2024-05-08", "2024-05-08"]),
                "year": [2024, 2024],
                "session": [AMC, BMO],
                "date_agree": [True, True],
            }
        )

    def _index(self, rows):
        return pd.DataFrame(rows, columns=["ticker", "obs_date", "right", "dte"])

    def test_through_print_needs_both_ends_and_both_sides(self, events, calendar):
        # AAA (AMC): pre-print 05-08, post-print 05-09. Give it all four.
        index = self._index(
            [
                ("AAA", pd.Timestamp("2024-05-08"), "C", 10),
                ("AAA", pd.Timestamp("2024-05-08"), "P", 10),
                ("AAA", pd.Timestamp("2024-05-09"), "C", 9),
                ("AAA", pd.Timestamp("2024-05-09"), "P", 9),
                # BBB gets only the entry side — not tradeable through the print.
                ("BBB", pd.Timestamp("2024-05-07"), "C", 10),
                ("BBB", pd.Timestamp("2024-05-07"), "P", 10),
            ]
        )
        out = coverage.event_chain_coverage(events, index, min_year=2024)
        by_ticker = dict(zip(out["ticker"], out["through_print_ready"]))
        assert by_ticker["AAA"]
        assert not by_ticker["BBB"]

    def test_session_decides_which_dates_are_checked(self, events, calendar):
        # BBB is BMO on 05-08, so its pre-print close is 05-07 and its
        # post-print close is 05-08 — different dates from the AMC name.
        index = self._index(
            [
                ("BBB", pd.Timestamp("2024-05-07"), "C", 10),
                ("BBB", pd.Timestamp("2024-05-07"), "P", 10),
                ("BBB", pd.Timestamp("2024-05-08"), "C", 9),
                ("BBB", pd.Timestamp("2024-05-08"), "P", 9),
            ]
        )
        out = coverage.event_chain_coverage(events, index, min_year=2024)
        by_ticker = dict(zip(out["ticker"], out["through_print_ready"]))
        assert by_ticker["BBB"]
        assert not by_ticker["AAA"]

    def test_one_side_only_is_not_counted_as_covered(self, events, calendar):
        index = self._index(
            [
                ("AAA", pd.Timestamp("2024-05-08"), "C", 10),
                ("AAA", pd.Timestamp("2024-05-09"), "C", 9),
            ]
        )
        out = coverage.event_chain_coverage(events, index, min_year=2024)
        row = out[out["ticker"] == "AAA"].iloc[0]
        assert row["entry_call"] and not row["entry_put"]
        assert row["entry_any"] and not row["entry_both"]
        assert not row["through_print_ready"]

    def test_events_without_a_session_are_excluded(self, events, calendar):
        events = events.copy()
        events["session"] = [None, None]
        out = coverage.event_chain_coverage(events, self._index([]), min_year=2024)
        assert out.empty

    def test_events_before_the_min_year_are_excluded(self, events, calendar):
        out = coverage.event_chain_coverage(events, self._index([]), min_year=2025)
        assert out.empty


class TestSideCoverage:
    def test_reports_all_three_observation_points(self):
        events = pd.DataFrame(
            {
                f"{point}_{side}": [True, False]
                for point in ("entry", "exit", "t14")
                for side in ("call", "put", "both", "any")
            }
        )
        out = side_coverage(events)
        assert list(out["point"]) == ["entry", "exit", "t14"]
        assert out["call"].iloc[0] == pytest.approx(0.5)

    def test_empty_input_reports_zero_rather_than_failing(self):
        columns = {
            f"{point}_{side}": pd.Series(dtype=bool)
            for point in ("entry", "exit", "t14")
            for side in ("call", "put", "both", "any")
        }
        out = side_coverage(pd.DataFrame(columns))
        assert (out[["call", "put", "both", "either"]] == 0).all().all()
