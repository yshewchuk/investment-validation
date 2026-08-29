"""The ingestion gate: what is excluded, what is quarantined, what is neither."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from engine.data import validate
from engine.data.validate import (
    QUARANTINE_EXCLUSION_RATE,
    ValidationReport,
    quarantine,
    should_quarantine,
    spot_vs_yfinance,
    validate_chains,
    validate_daily,
)


def chain_frame(n=100, **overrides) -> pd.DataFrame:
    obs = pd.Timestamp("2024-05-01")
    expiry = pd.Timestamp("2024-05-17")
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * n,
            "obs_date": [obs] * n,
            "expiry": [expiry] * n,
            "dte": [16] * n,
            "strike": [100.0 + i for i in range(n)],
            "right": ["C"] * n,
            "bid": [1.0] * n,
            "ask": [1.4] * n,
            "mid": [1.2] * n,
        }
    )
    for key, value in overrides.items():
        frame[key] = value
    return frame


class TestChainIntegrity:
    def test_a_clean_frame_passes_untouched(self):
        frame = chain_frame()
        clean, report = validate_chains(frame)
        assert report.ok
        assert len(clean) == len(frame)
        assert report.rows_excluded == 0

    def test_crossed_quotes_are_excluded(self):
        frame = chain_frame()
        frame.loc[0, "bid"] = 5.0  # bid > ask
        clean, report = validate_chains(frame)
        assert len(clean) == len(frame) - 1
        assert not report.ok

    def test_negative_prices_are_excluded(self):
        frame = chain_frame()
        frame.loc[0, "bid"] = -1.0
        clean, _ = validate_chains(frame)
        assert len(clean) == len(frame) - 1

    def test_dte_disagreeing_with_the_dates_is_excluded(self):
        frame = chain_frame()
        frame.loc[0, "dte"] = 99
        clean, report = validate_chains(frame)
        assert len(clean) == len(frame) - 1
        assert any(c.name == "dte_matches_dates" and not c.passed for c in report.checks)

    def test_an_expiry_before_the_observation_is_excluded(self):
        frame = chain_frame()
        frame.loc[0, "expiry"] = pd.Timestamp("2024-04-01")
        frame.loc[0, "dte"] = -30
        clean, _ = validate_chains(frame)
        assert len(clean) == len(frame) - 1

    def test_an_invalid_right_is_excluded(self):
        frame = chain_frame()
        frame.loc[0, "right"] = "X"
        clean, _ = validate_chains(frame)
        assert len(clean) == len(frame) - 1

    def test_duplicate_keys_keep_only_the_first(self):
        frame = chain_frame(n=4)
        frame.loc[3, "strike"] = frame.loc[0, "strike"]
        clean, report = validate_chains(frame)
        assert len(clean) == 3
        assert any(c.name == "primary_key_unique" and not c.passed for c in report.checks)

    def test_missing_quotes_are_kept_not_excluded(self):
        # An option with no quote is a real state; it is not a corrupt row.
        frame = chain_frame(n=4)
        frame.loc[0, ["bid", "ask"]] = np.nan
        clean, report = validate_chains(frame)
        assert len(clean) == 4
        assert report.ok

    def test_a_zero_bid_against_a_positive_ask_is_kept(self):
        frame = chain_frame(n=4)
        frame.loc[0, "bid"] = 0.0
        clean, report = validate_chains(frame)
        assert len(clean) == 4
        assert report.ok

    def test_an_empty_frame_is_handled(self):
        clean, report = validate_chains(chain_frame(n=0))
        assert clean.empty and report.rows_in == 0


class TestQuarantinePolicy:
    """Routine row exclusions are counted; structural failures are flagged."""

    def test_a_few_crossed_penny_quotes_do_not_flag_the_file(self, tmp_path):
        # Deep-OTM 0.04/0.03 markets appear in most chain files. Flagging 19,000
        # files for them would bury the cases that need a human.
        frame = chain_frame(n=1000)
        frame.loc[0, "bid"] = 5.0
        _, report = validate_chains(
            frame, source_file="x.json.gz", quarantine_root=tmp_path
        )
        assert report.rows_excluded == 1
        assert report.quarantined_files == []
        assert list(tmp_path.glob("*.flag.json")) == []

    def test_a_high_exclusion_rate_flags_the_file(self, tmp_path):
        frame = chain_frame(n=100)
        frame.loc[0:10, "bid"] = 5.0
        _, report = validate_chains(
            frame, source_file="bad.json.gz", quarantine_root=tmp_path
        )
        assert report.quarantined_files == ["bad.json.gz"]
        assert len(list(tmp_path.glob("*.flag.json"))) == 1

    def test_a_structural_failure_flags_regardless_of_rate(self, tmp_path):
        # One bad DTE means the parser disagrees with the file about what it is.
        frame = chain_frame(n=1000)
        frame.loc[0, "dte"] = 99
        _, report = validate_chains(
            frame, source_file="odd.json.gz", quarantine_root=tmp_path
        )
        assert report.quarantined_files == ["odd.json.gz"]

    def test_the_threshold_is_where_it_says_it_is(self):
        report = ValidationReport(table="t", rows_in=1000, rows_out=1000 - 5)
        assert not should_quarantine(report)
        report = ValidationReport(table="t", rows_in=1000, rows_out=1000 - 20)
        assert should_quarantine(report)
        assert QUARANTINE_EXCLUSION_RATE == pytest.approx(0.01)

    def test_a_flag_file_names_the_raw_file_and_the_reason(self, tmp_path):
        path = quarantine("2024-05-01_b0.json.gz", "because", {"n": 3}, root=tmp_path)
        payload = json.loads(path.read_text())
        assert payload[-1]["source_file"] == "2024-05-01_b0.json.gz"
        assert payload[-1]["reason"] == "because"
        assert payload[-1]["details"] == {"n": 3}

    def test_repeated_flags_append_rather_than_overwrite(self, tmp_path):
        quarantine("f.json.gz", "first", root=tmp_path)
        path = quarantine("f.json.gz", "second", root=tmp_path)
        assert len(json.loads(path.read_text())) == 2

    def test_quarantine_never_touches_the_raw_bytes(self, tmp_path):
        # Tier 1 is append-only: every byte ever fetched is kept, parseable or
        # not. A flag is a pointer, not a move.
        raw = tmp_path / "raw" / "2024-05-01_b0.json.gz"
        raw.parent.mkdir()
        raw.write_bytes(b"original")
        quarantine(str(raw), "reason", root=tmp_path / "q")
        assert raw.read_bytes() == b"original"


class TestDailyIntegrity:
    def _frame(self, n=50):
        return pd.DataFrame(
            {
                "ticker": ["AAA"] * n,
                "date": pd.date_range("2024-01-01", periods=n, freq="D"),
                "spot": [100.0] * n,
                "iv30": [30.0] * n,
                "implied_move": [5.0] * n,
                "rvol30": [25.0] * n,
            }
        )

    def test_a_clean_frame_passes(self):
        clean, report = validate_daily(self._frame())
        assert report.ok and len(clean) == 50

    def test_non_positive_spot_is_excluded(self):
        frame = self._frame()
        frame.loc[0, "spot"] = 0.0
        clean, _ = validate_daily(frame)
        assert len(clean) == 49

    def test_negative_vol_is_excluded(self):
        frame = self._frame()
        frame.loc[0, "iv30"] = -1.0
        clean, _ = validate_daily(frame)
        assert len(clean) == 49

    def test_duplicate_dates_collapse_to_the_last(self):
        frame = self._frame(n=3)
        frame.loc[2, "date"] = frame.loc[1, "date"]
        clean, _ = validate_daily(frame)
        assert len(clean) == 2

    def test_monotonicity_is_checked_per_ticker(self):
        clean, report = validate_daily(self._frame())
        assert any(c.name == "dates_monotone_per_ticker" and c.passed for c in report.checks)


class TestReportAggregation:
    def test_merging_sums_counts_and_keeps_the_verdict(self):
        total = ValidationReport(table="option_chains")
        for i in range(3):
            frame = chain_frame(n=10)
            if i == 2:
                frame.loc[0, "bid"] = 9.0
            _, part = validate_chains(frame)
            total.merge(part)
        assert total.rows_in == 30
        assert total.rows_out == 29
        assert not total.ok
        assert "bid_le_ask" in total.summary()["failed_checks"]

    def test_summary_is_serializable(self):
        _, report = validate_chains(chain_frame(n=5))
        assert json.dumps(report.summary())


class TestCrossSourceChecks:
    def test_spot_agrees_with_the_cached_yfinance_close(self, tmp_path):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        closes = np.linspace(100, 120, 50)
        pd.DataFrame({"date": dates, "close_raw": closes}).to_csv(
            tmp_path / "px_AAA.csv", index=False
        )
        daily = pd.DataFrame({"ticker": "AAA", "date": dates, "spot": closes * 1.001})
        result = spot_vs_yfinance(daily, sample_frac=1.0, yf_dir=tmp_path)
        assert result.passed
        assert result.n_checked > 0

    def test_a_systematic_spot_break_fails_the_check(self, tmp_path):
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        closes = np.linspace(100, 120, 50)
        pd.DataFrame({"date": dates, "close_raw": closes}).to_csv(
            tmp_path / "px_AAA.csv", index=False
        )
        daily = pd.DataFrame({"ticker": "AAA", "date": dates, "spot": closes * 1.10})
        assert not spot_vs_yfinance(daily, sample_frac=1.0, yf_dir=tmp_path).passed

    def test_a_handful_of_stale_rows_does_not_fail_an_otherwise_sound_ingest(self, tmp_path):
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        closes = np.full(100, 100.0)
        pd.DataFrame({"date": dates, "close_raw": closes}).to_csv(
            tmp_path / "px_AAA.csv", index=False
        )
        spot = closes.copy()
        spot[:3] = 150.0  # three bad rows out of a hundred
        daily = pd.DataFrame({"ticker": "AAA", "date": dates, "spot": spot})
        result = spot_vs_yfinance(daily, sample_frac=1.0, yf_dir=tmp_path)
        assert result.passed  # the median is what the tolerance is about
        assert result.n_failed == 3

    def test_no_overlap_is_reported_rather_than_passed_silently(self, tmp_path):
        daily = pd.DataFrame(
            {"ticker": ["AAA"], "date": pd.to_datetime(["2024-01-01"]), "spot": [100.0]}
        )
        result = spot_vs_yfinance(daily, yf_dir=tmp_path)
        assert result.n_checked == 0
