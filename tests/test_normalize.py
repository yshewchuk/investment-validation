"""Normalizers — every unit and convention trap, asserted."""
from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from engine.data.normalize.common import (
    FLT_MAX_THRESHOLD,
    MCAP_ERAS,
    clip_implausible,
    mask_sentinels,
    mcap_era,
    mcap_to_usd,
)
from engine.data.normalize import n_chains, n_daily, n_securities


class TestMarketCapEras:
    """ORATS mktCap changes units TWICE, and the legacy panel knew about one."""

    def test_three_eras_are_declared_in_order(self):
        labels = [label for _, _, label in MCAP_ERAS]
        assert labels == ["billions", "millions", "thousands"]

    def test_the_2017_boundary_is_the_undocumented_one(self):
        # AAPL 2017-06-27 reads 824 and 2017-06-28 reads 761,377 — the same
        # company, a factor of ~1000 apart. Verified on 148 tickers whose
        # history spans the date.
        before = mcap_to_usd([824], [pd.Timestamp("2017-06-27")])[0]
        after = mcap_to_usd([761_377], [pd.Timestamp("2017-06-28")])[0]
        assert before == pytest.approx(824e9)
        assert after == pytest.approx(761_377e6)
        assert 0.5 < after / before < 2.0  # continuous across the boundary

    def test_the_2026_boundary_is_the_documented_one(self):
        before = mcap_to_usd([3_833_686], [pd.Timestamp("2026-03-10")])[0]
        after = mcap_to_usd([3_820_766_685], [pd.Timestamp("2026-03-11")])[0]
        assert before == pytest.approx(3.833686e12)
        assert after == pytest.approx(3.820766685e12)
        assert 0.9 < after / before < 1.1

    def test_known_anchor_values_convert_to_reality(self):
        # AAPL 2007-01-03 at an $83.80 share price was ~$72B, not $72M.
        assert mcap_to_usd([72], [pd.Timestamp("2007-01-03")])[0] == pytest.approx(72e9)
        # AAPL 2026-08-27 at $313.90 was ~$4.6T.
        assert mcap_to_usd([4_608_409_846], [pd.Timestamp("2026-08-27")])[0] == pytest.approx(
            4.608409846e12
        )

    def test_the_legacy_rule_understates_the_billions_era_by_a_thousand(self):
        # This is the documented delta the migration test carries: the legacy
        # panel applied x1e6 to everything before 2026-03-11.
        raw, date = 72, pd.Timestamp("2007-01-03")
        legacy = np.log(raw * 1e6)
        corrected = np.log(mcap_to_usd([raw], [date])[0])
        assert corrected - legacy == pytest.approx(np.log(1000))

    def test_era_labels_track_the_boundaries(self):
        eras = mcap_era(pd.to_datetime(["2010-01-01", "2020-01-01", "2026-06-01"]))
        assert list(eras) == ["billions", "millions", "thousands"]

    def test_non_positive_market_caps_become_nan_not_negative_infinity(self):
        out = mcap_to_usd([0, -5, None], pd.to_datetime(["2020-01-01"] * 3))
        assert np.isnan(out).all()


class TestSentinels:
    def test_flt_max_is_masked(self):
        # ORATS encodes missing as FLT_MAX (~3.4e38) rather than null; left in
        # place it poisons every mean, z-score, and fit downstream.
        out = mask_sentinels([1.0, 3.4e38, 2.0])
        assert np.isnan(out[1])
        assert out[0] == 1.0 and out[2] == 2.0

    def test_negative_sentinels_are_masked_too(self):
        assert np.isnan(mask_sentinels([-3.4e38])[0])

    def test_ordinary_large_values_survive(self):
        assert mask_sentinels([1e20])[0] == 1e20
        assert FLT_MAX_THRESHOLD > 1e20

    def test_unparseable_values_become_nan(self):
        assert np.isnan(mask_sentinels(["abc"])[0])


class TestPlausibleRanges:
    def test_out_of_range_values_are_nulled_and_counted(self):
        frame = pd.DataFrame({"implied_move": [5.0, 500.0, -2.0], "iv30": [30.0, 30.0, 30.0]})
        out, counts = clip_implausible(frame)
        assert counts["implied_move"] == 2
        assert out["implied_move"].tolist()[0] == 5.0
        assert out["implied_move"].isna().sum() == 2

    def test_clean_frames_report_nothing(self):
        frame = pd.DataFrame({"implied_move": [5.0, 8.0]})
        _, counts = clip_implausible(frame)
        assert counts == {}


class TestChainNormalizer:
    def _write(self, tmp_path, name, rows, tickers=("AAA",)):
        path = tmp_path / name
        doc = {"entry_date": name[:10], "tickers": list(tickers), "rows": rows}
        with gzip.open(path, "wt") as fh:
            json.dump(doc, fh)
        return path

    def _row(self, **over):
        base = {
            "ticker": "AAA",
            "tradeDate": "2024-05-01",
            "expirDate": "2024-05-17",
            "dte": 16,
            "strike": 100.0,
            "stockPrice": 99.5,
            "spotPrice": 99.5,
            "callBidPrice": 2.0,
            "callAskPrice": 2.4,
            "callMidIv": 0.31,
            "putBidPrice": 1.0,
            "putAskPrice": 1.4,
            "putMidIv": 0.33,
            "delta": 0.55,
            "smvVol": 0.32,
        }
        base.update(over)
        return base

    @pytest.mark.parametrize(
        "name,kind",
        [
            ("2024-05-01_b0.json.gz", "eod"),
            ("2024-05-01_t14_b3.json.gz", "t14"),
            ("2024-05-01_c2_b1.json.gz", "c2"),
        ],
    )
    def test_all_four_pull_generations_parse(self, name, kind):
        assert n_chains.parse_filename(name)["chain_kind"] == kind

    def test_done_markers_are_not_payload(self):
        assert n_chains.parse_filename("2024-05-01.done") is None
        assert n_chains.parse_filename("2024-05-01.done_t14") is None

    def test_each_strike_yields_a_call_and_a_put_row(self, tmp_path):
        path = self._write(tmp_path, "2024-05-01_b0.json.gz", [self._row()])
        out, report = n_chains.normalize_file(path)
        assert len(out) == 2
        assert set(out["right"]) == {"C", "P"}
        assert report["rows_in"] == 1 and report["rows_out"] == 2

    def test_put_delta_is_derived_from_the_reported_call_delta(self, tmp_path):
        # ORATS reports the CALL delta; the put at the same strike is delta-1.
        path = self._write(tmp_path, "2024-05-01_b0.json.gz", [self._row(delta=0.55)])
        out, _ = n_chains.normalize_file(path)
        assert out.loc[out["right"] == "C", "delta"].iloc[0] == pytest.approx(0.55)
        assert out.loc[out["right"] == "P", "delta"].iloc[0] == pytest.approx(-0.45)

    def test_dte_is_recomputed_from_the_dates(self, tmp_path):
        # The cached `dte` is whatever the API said at pull time; the schema
        # contract is dte == expiry - obs_date, so it is derived, not trusted.
        path = self._write(tmp_path, "2024-05-01_b0.json.gz", [self._row(dte=999)])
        out, _ = n_chains.normalize_file(path)
        assert (out["dte"] == 16).all()

    def test_mid_is_the_average_of_bid_and_ask(self, tmp_path):
        path = self._write(tmp_path, "2024-05-01_b0.json.gz", [self._row()])
        out, _ = n_chains.normalize_file(path)
        assert out.loc[out["right"] == "C", "mid"].iloc[0] == pytest.approx(2.2)

    def test_sentinels_in_chain_prices_are_masked(self, tmp_path):
        path = self._write(tmp_path, "2024-05-01_b0.json.gz", [self._row(callAskPrice=3.4e38)])
        out, _ = n_chains.normalize_file(path)
        assert pd.isna(out.loc[out["right"] == "C", "ask"].iloc[0])

    def test_provenance_records_the_raw_file(self, tmp_path):
        path = self._write(tmp_path, "2024-05-01_t14_b0.json.gz", [self._row()])
        out, _ = n_chains.normalize_file(path)
        assert (out["src_file"] == "2024-05-01_t14_b0.json.gz").all()
        assert (out["src"] == "orats.hist.strikes").all()
        assert (out["chain_kind"] == "t14").all()

    def test_an_empty_document_yields_nothing_and_says_so(self, tmp_path):
        path = self._write(tmp_path, "2024-05-01_b0.json.gz", [])
        out, report = n_chains.normalize_file(path)
        assert out.empty and report["rows_out"] == 0

    def test_a_document_missing_price_columns_is_reported(self, tmp_path):
        path = self._write(
            tmp_path,
            "2024-05-01_b0.json.gz",
            [{"ticker": "AAA", "tradeDate": "2024-05-01", "expirDate": "2024-05-17",
              "strike": 100.0}],
        )
        out, report = n_chains.normalize_file(path)
        assert out.empty
        assert "no price columns" in report["reason"]


class TestDailyNormalizer:
    def _write(self, root, ticker, rows):
        root.mkdir(parents=True, exist_ok=True)
        with gzip.open(root / f"{ticker}.json.gz", "wt") as fh:
            json.dump(rows, fh)

    def test_iv_fields_are_converted_to_vol_points(self, tmp_path):
        # ORATS delivers decimals; the panel convention every model was fit on
        # is vol points, so the x100 happens here and nowhere else.
        summaries = tmp_path / "summaries"
        self._write(summaries, "AAA", [
            {"ticker": "AAA", "tradeDate": "2024-05-01", "stockPrice": 100.0,
             "iv30d": 0.4074, "impliedMove": 0.1003, "rVol30": 0.409,
             "skewing": 0.9, "contango": -0.33}
        ])
        out, _ = n_daily.normalize_ticker("AAA", summaries_dir=summaries, cores_dir=tmp_path / "none")
        row = out.iloc[0]
        assert row["iv30"] == pytest.approx(40.74)
        assert row["implied_move"] == pytest.approx(10.03)
        assert row["rvol30"] == pytest.approx(40.9)
        # Unitless fields are NOT scaled.
        assert row["skew"] == pytest.approx(0.9)
        assert row["contango"] == pytest.approx(-0.33)
        # Spot is a price, not a rate.
        assert row["spot"] == pytest.approx(100.0)

    def test_market_cap_is_joined_and_converted(self, tmp_path):
        summaries, cores = tmp_path / "summaries", tmp_path / "cores"
        self._write(summaries, "AAA", [
            {"ticker": "AAA", "tradeDate": "2007-01-03", "stockPrice": 83.8},
        ])
        self._write(cores, "AAA", [
            {"ticker": "AAA", "tradeDate": "2007-01-03", "mktCap": 72},
        ])
        out, report = n_daily.normalize_ticker("AAA", summaries_dir=summaries, cores_dir=cores)
        assert out.iloc[0]["mcap_usd"] == pytest.approx(72e9)
        assert out.iloc[0]["mcap_log"] == pytest.approx(np.log(72e9))
        assert report["mcap_rows"] == 1

    def test_a_missing_cores_file_leaves_market_cap_null_not_zero(self, tmp_path):
        summaries = tmp_path / "summaries"
        self._write(summaries, "AAA", [
            {"ticker": "AAA", "tradeDate": "2024-05-01", "stockPrice": 10.0}
        ])
        out, _ = n_daily.normalize_ticker("AAA", summaries_dir=summaries, cores_dir=tmp_path / "none")
        assert out["mcap_usd"].isna().all()

    def test_duplicate_trade_dates_collapse(self, tmp_path):
        summaries = tmp_path / "summaries"
        self._write(summaries, "AAA", [
            {"ticker": "AAA", "tradeDate": "2024-05-01", "stockPrice": 10.0},
            {"ticker": "AAA", "tradeDate": "2024-05-01", "stockPrice": 11.0},
        ])
        out, _ = n_daily.normalize_ticker("AAA", summaries_dir=summaries, cores_dir=tmp_path / "n")
        assert len(out) == 1
        assert out.iloc[0]["spot"] == 11.0  # last wins

    def test_a_missing_ticker_reports_rather_than_raising(self, tmp_path):
        out, report = n_daily.normalize_ticker(
            "NOPE", summaries_dir=tmp_path, cores_dir=tmp_path
        )
        assert out.empty
        assert "no summaries" in report["reason"]


class TestSecurities:
    def test_year_end_size_and_era_are_recorded(self):
        daily = pd.DataFrame(
            {
                "ticker": ["AAA"] * 3,
                "date": pd.to_datetime(["2007-01-03", "2007-06-01", "2007-12-31"]),
                "year": [2007] * 3,
                "mcap_usd": [72e9, 80e9, 90e9],
            }
        )
        out, report = n_securities.normalize_from_daily(daily)
        row = out.iloc[0]
        assert row["mcap_usd"] == pytest.approx(90e9)  # last observation of the year
        assert row["mcap_unit_era"] == "billions"
        assert row["n_obs"] == 3
        assert report["rows"] == 1

    def test_the_quantized_era_is_flagged(self):
        # Pre-2017 caps are INTEGER billions: a $1.4B name reads as 1, so the
        # 1-10B slice in that era has ~7 distinguishable buckets.
        daily = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "date": pd.to_datetime(["2010-05-01", "2024-05-01"]),
                "year": [2010, 2024],
                "mcap_usd": [3e9, 3e9],
            }
        )
        out, _ = n_securities.normalize_from_daily(daily)
        flags = dict(zip(out["ticker"], out["mcap_quantized"]))
        assert flags["AAA"] is True or flags["AAA"] == True  # noqa: E712
        assert not flags["BBB"]

    def test_the_raw_figure_is_reconstructible_from_tier_2(self):
        daily = pd.DataFrame(
            {"ticker": ["AAA"], "date": pd.to_datetime(["2024-05-01"]),
             "year": [2024], "mcap_usd": [4.0e10]}
        )
        out, _ = n_securities.normalize_from_daily(daily)
        assert out.iloc[0]["mcap_raw"] == pytest.approx(40_000.0)  # millions era

    def test_a_year_with_no_size_still_produces_a_row(self):
        daily = pd.DataFrame(
            {"ticker": ["AAA"], "date": pd.to_datetime(["2024-05-01"]),
             "year": [2024], "mcap_usd": [np.nan]}
        )
        out, _ = n_securities.normalize_from_daily(daily)
        assert len(out) == 1
        assert pd.isna(out.iloc[0]["mcap_usd"])


class TestChainLiquidityFields:
    """The 2026-09 pull asks for size; the older cache never did."""

    def _raw(self, extra: dict | None = None) -> dict:
        row = {
            "ticker": "AMD", "tradeDate": "2026-09-02", "expirDate": "2026-09-19",
            "dte": 17, "strike": 150.0, "stockPrice": 151.0, "spotPrice": 151.0,
            "callBidPrice": 4.3, "callAskPrice": 4.5, "putBidPrice": 3.9,
            "putAskPrice": 4.1, "callMidIv": 0.42, "putMidIv": 0.41, "delta": 0.52,
        }
        row.update(extra or {})
        return {"entry_date": "2026-09-02", "tickers": ["AMD"], "rows": [row]}

    def test_liquidity_lands_in_tier_2_when_present(self):
        from engine.data.normalize.n_chains import rows_to_frame

        raw = self._raw({"callVolume": 1200, "callOpenInterest": 8400,
                         "callBidSize": 30, "callAskSize": 2,
                         "putVolume": 300, "putOpenInterest": 5100,
                         "putBidSize": 0, "putAskSize": 16})
        frame, _meta = rows_to_frame(raw["rows"], source_id="probe", chain_kind="entry")
        call = frame[frame["right"] == "C"].iloc[0]
        put = frame[frame["right"] == "P"].iloc[0]
        assert call["volume"] == 1200 and call["open_interest"] == 8400
        assert call["bid_size"] == 30 and call["ask_size"] == 2
        assert put["volume"] == 300 and put["bid_size"] == 0

    def test_absent_liquidity_is_null_not_zero(self):
        """The distinction the whole fill question turns on."""
        import numpy as np

        from engine.data.normalize.n_chains import rows_to_frame

        frame, _meta = rows_to_frame(self._raw()["rows"], source_id="legacy", chain_kind="entry")
        for column in ("volume", "open_interest", "bid_size", "ask_size"):
            assert frame[column].isna().all(), (
                f"{column} must be NaN when the pull never requested it — a zero "
                "would assert there was no size")
            assert not (frame[column] == 0).any()

    def test_a_true_zero_survives(self):
        from engine.data.normalize.n_chains import rows_to_frame

        raw = self._raw({"callVolume": 0, "callOpenInterest": 0,
                         "callBidSize": 0, "callAskSize": 0})
        frame, _meta = rows_to_frame(raw["rows"], source_id="probe", chain_kind="entry")
        call = frame[frame["right"] == "C"].iloc[0]
        assert call["volume"] == 0 and call["bid_size"] == 0


class TestPullRequestsLiquidity:
    def test_the_sep_pull_asks_for_size_and_open_interest(self):
        from engine.data.pulls.sep2026_plan import FIELDS, LIQUIDITY_FIELDS

        for field in LIQUIDITY_FIELDS:
            assert field in FIELDS, f"{field} missing from the pull's field list"
        # ORATS bills per call, not per field: this costs nothing to ask for.
        assert len(FIELDS.split(",")) == 23
