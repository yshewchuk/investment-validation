"""Trade-set normalization.

Two things here are correctness guards rather than plumbing, and both would fail
silently if wrong:

* ``exit_mode == "chain"`` filtering. The S2 set mixes trades priced from a real
  exit chain with trades priced at intrinsic value at expiry. The intrinsic
  fallback peeks at the settlement price, so including it is look-ahead bias.
* ``fill_alpha`` tagging. These sets were priced buy-ask / sell-bid. A P&L number
  in Tier 2 that does not record the fill convention that produced it is not
  interpretable, and mixing conventions is exactly how the program's original
  NOT-VIABLE verdicts got made.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.data.normalize.n_trades import LEGACY_SPECS, normalize_legacy_set


def write_trades(path, rows, columns=None):
    frame = pd.DataFrame(rows)
    if columns:
        frame = frame[columns]
    frame.to_csv(path, index=False)
    return path


def base_row(**over):
    row = {
        "ticker": "AAA",
        "date": "2024-05-08",
        "entry_date": "2024-05-07",
        "exit_date": "2024-05-09",
        "expiry": "2024-05-17",
        "strike": 100.0,
        "cost": 4.0,
        "exit_val": 5.0,
        "ret": 0.25,
        "year": 2024,
    }
    row.update(over)
    return row


@pytest.fixture
def spec_factory(tmp_path):
    def make(rows, variant="v"):
        path = write_trades(tmp_path / "trades.csv", rows)
        return {"path": path, "entry": "entry_date", "exit": "exit_date", "variant": variant}

    return make


class TestLookAheadGuard:
    def test_intrinsic_exits_are_dropped(self, spec_factory):
        rows = [
            base_row(exit_mode="chain"),
            base_row(ticker="BBB", exit_mode="intrinsic"),
        ]
        out, report = normalize_legacy_set("S2_short_dte", spec_factory(rows))
        assert len(out) == 1
        assert out.iloc[0]["ticker"] == "AAA"
        assert report["dropped_non_chain_exit"] == 1

    def test_the_drop_is_counted_not_silent(self, spec_factory):
        rows = [base_row(ticker=f"T{i}", exit_mode="intrinsic") for i in range(3)]
        out, report = normalize_legacy_set("S2_short_dte", spec_factory(rows))
        assert out.empty
        assert report["dropped_non_chain_exit"] == 3

    def test_sets_without_an_exit_mode_column_are_kept_whole(self, spec_factory):
        rows = [base_row(), base_row(ticker="BBB")]
        out, report = normalize_legacy_set("S1_calendar", spec_factory(rows))
        assert len(out) == 2
        assert report["dropped_non_chain_exit"] == 0


class TestFillConvention:
    def test_every_row_records_the_worst_case_alpha(self, spec_factory):
        out, _ = normalize_legacy_set("S1_calendar", spec_factory([base_row()]))
        assert (out["fill_alpha"] == 0.0).all()

    def test_the_reported_mean_is_labelled_worst_fill(self, spec_factory):
        rows = [base_row(ret=0.2), base_row(ticker="BBB", ret=-0.4)]
        _, report = normalize_legacy_set("S1_calendar", spec_factory(rows))
        assert report["mean_ret_worst_fill"] == pytest.approx(-0.1)


class TestIdentityAndProvenance:
    def test_trade_ids_are_stable_and_unique(self, spec_factory):
        rows = [base_row(), base_row(ticker="BBB", strike=50.0)]
        out, _ = normalize_legacy_set("S1_calendar", spec_factory(rows))
        assert out["trade_id"].is_unique
        assert out.iloc[0]["trade_id"] == "S1_calendar:AAA:20240508:100"

    def test_duplicate_ids_are_collapsed_and_counted(self, spec_factory):
        rows = [base_row(), base_row()]
        out, report = normalize_legacy_set("S1_calendar", spec_factory(rows))
        assert len(out) == 1
        assert report["dropped_duplicate_ids"] == 1

    def test_event_id_matches_the_calendar_convention(self, spec_factory):
        out, _ = normalize_legacy_set("S1_calendar", spec_factory([base_row()]))
        assert out.iloc[0]["event_id"] == "AAA_2024-05-08"

    def test_provenance_names_the_source_file(self, spec_factory):
        out, _ = normalize_legacy_set("S1_calendar", spec_factory([base_row()]))
        assert out.iloc[0]["provenance"].startswith("legacy:")

    def test_rows_are_marked_as_simulations(self, spec_factory):
        out, _ = normalize_legacy_set("S1_calendar", spec_factory([base_row()]))
        assert (out["kind"] == "sim").all()


class TestRobustness:
    def test_a_missing_trade_set_reports_rather_than_raising(self, tmp_path):
        spec = {"path": tmp_path / "nope.csv", "entry": "entry_date",
                "exit": "exit_date", "variant": "v"}
        out, report = normalize_legacy_set("S1_calendar", spec)
        assert out.empty
        assert "missing" in report["reason"]

    def test_rows_with_an_unparseable_date_are_dropped(self, spec_factory):
        rows = [base_row(), base_row(ticker="BBB", date="not-a-date")]
        out, _ = normalize_legacy_set("S1_calendar", spec_factory(rows))
        assert len(out) == 1


class TestSpecRegistry:
    def test_the_legacy_codes_are_not_the_program_strategy_codes(self):
        # S2 and STR-THRU are structurally similar but were not specified
        # identically. Conflating them would let an old trade set masquerade as
        # evidence for a spec it never tested — the exact mistake the plan
        # warns about for CAL-P.
        assert set(LEGACY_SPECS) == {"S1_calendar", "S2_short_dte", "S3_runup"}
        for code in ("CAL-P", "STR-THRU", "STR-RUNUP"):
            assert code not in LEGACY_SPECS

    def test_every_spec_names_its_entry_column(self):
        # S3 opens at T-14, recorded as `t10_date`; using `entry_date` would
        # silently price the wrong day.
        assert LEGACY_SPECS["S3_runup"]["entry"] == "t10_date"
        for spec in LEGACY_SPECS.values():
            assert spec["entry"] and spec["exit"] and spec["variant"]
