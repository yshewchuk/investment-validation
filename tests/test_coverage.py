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

    def test_no_decision_offsets_leaves_the_frame_alone(self, events, calendar):
        """Every existing caller passes nothing and must see what it always saw."""
        index = self._index([])
        out = coverage.event_chain_coverage(events, index, min_year=2024)
        assert not [c for c in out.columns if c.startswith("d1_") or c.startswith("d2_")]

    def test_a_decision_offset_steps_back_from_the_entry_close(self, events, calendar):
        index = self._index([])
        out = coverage.event_chain_coverage(
            events, index, min_year=2024, decision_offsets=(-1,)
        )
        by_ticker = {
            t: (e, d) for t, e, d in zip(out["ticker"], out["entry_date"], out["d1_date"])
        }
        # AAA is AMC → entry 05-08, decision 05-07.
        assert by_ticker["AAA"] == (pd.Timestamp("2024-05-08"), pd.Timestamp("2024-05-07"))
        # BBB is BMO → entry 05-07, decision 05-06. The decision date is
        # session-dependent because the close it steps back from already is.
        assert by_ticker["BBB"] == (pd.Timestamp("2024-05-07"), pd.Timestamp("2024-05-06"))

    def test_the_decision_chain_is_measured_on_its_own_date(self, events, calendar):
        index = self._index(
            [
                # AAA has the entry chain but nothing a session earlier.
                ("AAA", pd.Timestamp("2024-05-08"), "C", 10),
                ("AAA", pd.Timestamp("2024-05-08"), "P", 10),
                # BBB has the decision chain but only one side of it.
                ("BBB", pd.Timestamp("2024-05-06"), "C", 11),
            ]
        )
        out = coverage.event_chain_coverage(
            events, index, min_year=2024, decision_offsets=(-1,)
        )
        rows = out.set_index("ticker")
        assert rows.loc["AAA", "entry_both"] and not rows.loc["AAA", "d1_both"]
        assert rows.loc["BBB", "d1_any"] and not rows.loc["BBB", "d1_both"]

    def test_several_offsets_each_get_their_own_block(self, events, calendar):
        out = coverage.event_chain_coverage(
            events, self._index([]), min_year=2024, decision_offsets=(-1, -2)
        )
        # 05-04 is the Saturday; two business days back from 05-08 is 05-06.
        assert out.set_index("ticker").loc["AAA", "d2_date"] == pd.Timestamp("2024-05-06")
        for label in ("d1", "d2"):
            for suffix in ("date", "call", "put", "any", "both"):
                assert f"{label}_{suffix}" in out.columns

    def test_a_forward_decision_offset_is_refused(self):
        with pytest.raises(ValueError, match="steps back"):
            coverage.decision_label(1)

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


class TestRendering:
    def test_a_table_renders_with_a_header_and_rows(self):
        frame = pd.DataFrame(
            {"1-10B": [0.5, 0.75], ">10B": [0.9, 0.95]},
            index=pd.Index([2023, 2024], name="year"),
        )
        out = coverage._table(frame)
        assert "| year | 1-10B | >10B |" in out
        assert "0.500" in out and "0.950" in out

    def test_an_empty_table_says_so_instead_of_rendering_nothing(self):
        assert "no data" in coverage._table(pd.DataFrame())

    def test_non_finite_cells_render_as_a_dash(self):
        frame = pd.DataFrame({"a": [np.nan]}, index=pd.Index([2024], name="year"))
        assert "—" in coverage._table(frame)

    def test_integer_formatting_is_respected(self):
        frame = pd.DataFrame({"n": [1234.0]}, index=pd.Index([2024], name="year"))
        assert "1234" in coverage._table(frame, "{:.0f}")

    def test_the_audit_carries_the_sections_the_plan_asks_for(self, monkeypatch):
        events = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "year": [2024, 2024],
                "mcap_bucket": ["1-10B", ">10B"],
                "through_print_ready": [True, False],
                "entry_both": [True, True],
                **{
                    f"{point}_{side}": [True, False]
                    for point in ("entry", "exit", "t14")
                    for side in ("call", "put", "both", "any")
                },
            }
        )
        monkeypatch.setattr(
            coverage,
            "manifest",
            None,
            raising=False,
        )
        body = coverage.render_audit(events, [])
        for heading in (
            "Phase 0 — Data Audit",
            "Store inventory",
            "Chain coverage",
            "Call vs put coverage",
        ):
            assert heading in body

    def test_sanity_results_are_tabulated_when_supplied(self):
        from engine.data.validate import CheckResult

        events = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "year": [2024],
                "mcap_bucket": ["1-10B"],
                "through_print_ready": [True],
                "entry_both": [True],
                **{
                    f"{point}_{side}": [True]
                    for point in ("entry", "exit", "t14")
                    for side in ("call", "put", "both", "any")
                },
            }
        )
        checks = [CheckResult("spot_vs_yfinance", True, 100, 2, "median 0.3%")]
        body = coverage.render_audit(events, checks)
        assert "Price-sanity battery" in body
        assert "spot_vs_yfinance" in body
        assert "PASS" in body


class TestCoverageMatrix:
    def test_it_reports_rates_and_the_counts_behind_them(self):
        events = pd.DataFrame(
            {
                "year": [2024, 2024, 2024],
                "mcap_bucket": ["1-10B", "1-10B", ">10B"],
                "through_print_ready": [True, False, True],
            }
        )
        rates, counts = coverage.coverage_matrix(events)
        assert rates.loc[2024, "1-10B"] == pytest.approx(0.5)
        assert counts.loc[2024, "1-10B"] == 2

    def test_an_empty_frame_yields_an_empty_matrix(self):
        assert coverage.coverage_matrix(pd.DataFrame()).empty
