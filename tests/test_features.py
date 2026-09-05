"""Feature builders.

The load-bearing property here is that the live path and the panel path produce
the same numbers, because the models are trained on the panel and served on the
live path. :func:`advance_history` is where that property is won or lost, so it
is checked against hand-computed recursions rather than against the panel it is
meant to reproduce.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine import features
from engine.data.features import panel as panel_mod


class TestAdvanceHistory:
    """The one-step recursion that lets the live path resume the panel's state."""

    def row(self, **kwargs):
        base = {
            "n_prior": 10,
            "move": 6.0,
            "abs_move": 6.0,
            "or_implied": 5.0,
            "mean_prior_move": 1.0,
            "mean_prior_abs_move": 4.0,
            "mean_prior_or_implied": 5.5,
        }
        for span in panel_mod.SPANS:
            base[f"ema{span}_prior_move"] = 2.0
            base[f"ema{span}_prior_abs_move"] = 3.0
        base.update(kwargs)
        return pd.Series(base)

    def test_counter_advances(self):
        assert features.advance_history(self.row())["n_prior"] == 11

    def test_mean_is_exact(self):
        """(1.0 × 10 + 6.0) / 11."""
        out = features.advance_history(self.row())
        assert out["mean_prior_move"] == pytest.approx((1.0 * 10 + 6.0) / 11)
        assert out["mean_prior_abs_move"] == pytest.approx((4.0 * 10 + 6.0) / 11)

    def test_implied_mean_is_exact(self):
        out = features.advance_history(self.row())
        assert out["mean_prior_or_implied"] == pytest.approx((5.5 * 10 + 5.0) / 11)

    def test_ema_resumes_the_panel_recursion(self):
        out = features.advance_history(self.row())
        for span in panel_mod.SPANS:
            a = 2.0 / (span + 1.0)
            assert out[f"ema{span}_prior_move"] == pytest.approx(a * 6.0 + (1 - a) * 2.0)
            assert out[f"ema{span}_prior_abs_move"] == pytest.approx(a * 6.0 + (1 - a) * 3.0)

    def test_an_unavailable_ema_stays_unavailable(self):
        """You cannot resume a recursion you have no value to resume from."""
        out = features.advance_history(self.row(ema12_prior_move=np.nan))
        assert np.isnan(out["ema12_prior_move"])
        assert not np.isnan(out["ema2_prior_move"])

    def test_a_missing_implied_carries_the_mean_forward(self):
        out = features.advance_history(self.row(or_implied=np.nan))
        assert out["mean_prior_or_implied"] == pytest.approx(5.5)

    def test_matches_a_full_recomputation_over_a_known_history(self):
        """Advancing must agree with recomputing from scratch, given full history."""
        rng = np.random.default_rng(0)
        moves = rng.normal(0, 5, 30).tolist()
        abs_moves = [abs(m) for m in moves]

        # `history_features` no longer takes an implied series: the running
        # implied mean is built by `add_implied_history` from the ORATS quote,
        # not from the oquants column the panel used to carry.
        at_20 = panel_mod.history_features(moves[:20], abs_moves[:20])
        row = pd.Series({**at_20, "move": moves[20], "abs_move": abs_moves[20]})
        advanced = features.advance_history(row)

        direct = panel_mod.history_features(moves[:21], abs_moves[:21])
        for key, value in direct.items():
            if value is None:
                continue
            assert advanced[key] == pytest.approx(value, rel=1e-12), key


class TestLiveUnavailable:
    def test_flags_an_unservable_feature(self, monkeypatch):
        """The guard still fires — on a stand-in, since nothing real trips it.

        This named `implied_move` until 2026-09-05, when the column left the
        panel entirely and LIVE_UNAVAILABLE became empty. The machinery is kept
        for the next realized-only column (see its docstring), so it keeps a
        test; an untested guard is one nobody learns has broken.
        """
        monkeypatch.setattr(features, "LIVE_UNAVAILABLE", ("a_realized_only_column",))
        with pytest.raises(features.UnservableFeature, match="or_implied"):
            features.assert_live_available(["ema12r_abs", "a_realized_only_column"])

    def test_passes_a_servable_list(self):
        features.assert_live_available(["ema12r_abs", "or_implied", "mcap_log"])

    def test_the_legacy_implied_move_is_gone_from_the_panel(self):
        """Not quarantined — removed. The trap stopped existing.

        The realized move is computed from prices and the implied move comes
        from ORATS `daily_market`, and both exist for an event that has not
        happened yet. There is no longer a column to bar from a model.
        """
        assert "implied_move" not in features.PANEL_FEATURE_COLUMNS
        assert "mean_prior_implied_move" not in features.PANEL_FEATURE_COLUMNS
        assert features.LIVE_UNAVAILABLE == ()

    def test_outcomes_are_never_features(self):
        for column in features.OUTCOME_COLUMNS:
            assert column not in features.PANEL_FEATURE_COLUMNS


class TestDailyStateFrame:
    @pytest.fixture
    def daily(self):
        dates = pd.bdate_range("2024-01-01", periods=30)
        return pd.DataFrame(
            {
                "ticker": "TEST",
                "date": dates,
                "src_iv": "orats",
                "implied_move": np.arange(30, dtype=float),
                "iv10": np.arange(30, dtype=float) * 2,
                "iv30": np.arange(30, dtype=float) * 3,
                "exern_iv10": 1.0,
                "exern_iv30": np.arange(30, dtype=float),
                "iee": 1.0,
                "skew": 1.0,
                "contango": 1.0,
                "fwd90_30": 1.0,
                "fexern90_30": 1.0,
                "rvol30": 1.0,
                "spot": 100.0,
                "mcap_log": 25.0,
            }
        )

    def test_reads_the_row_on_or_before_as_of(self, daily):
        """as_of is a close we would trade at, so that close's quotes are ours."""
        request = pd.DataFrame({"ticker": ["TEST"], "as_of": [daily["date"].iloc[10]]})
        out = features.daily_state_frame(request, daily=daily)
        assert out["im"].iloc[0] == 10.0

    def test_falls_back_to_the_previous_row_on_a_non_trading_date(self, daily):
        friday = daily["date"].iloc[4]
        assert friday.weekday() == 4
        saturday = friday + pd.Timedelta(days=1)
        request = pd.DataFrame({"ticker": ["TEST"], "as_of": [saturday]})
        out = features.daily_state_frame(request, daily=daily)
        assert out["im"].iloc[0] == 4.0

    def test_lags_count_rows_not_calendar_days(self, daily):
        request = pd.DataFrame({"ticker": ["TEST"], "as_of": [daily["date"].iloc[10]]})
        out = features.daily_state_frame(request, daily=daily)
        assert out["im_d1"].iloc[0] == pytest.approx(1.0)
        assert out["im_d5"].iloc[0] == pytest.approx(5.0)
        assert out["iv30_d10"].iloc[0] == pytest.approx(30.0)

    def test_no_history_yields_nan_not_a_borrowed_value(self, daily):
        request = pd.DataFrame(
            {"ticker": ["TEST"], "as_of": [daily["date"].iloc[0] - pd.Timedelta(days=5)]}
        )
        out = features.daily_state_frame(request, daily=daily)
        assert np.isnan(out["im"].iloc[0])

    def test_a_lag_reaching_before_the_series_is_nan(self, daily):
        request = pd.DataFrame({"ticker": ["TEST"], "as_of": [daily["date"].iloc[2]]})
        out = features.daily_state_frame(request, daily=daily)
        assert out["im"].iloc[0] == 2.0
        assert np.isnan(out["im_d5"].iloc[0])

    def test_unknown_ticker_yields_nan(self, daily):
        request = pd.DataFrame({"ticker": ["OTHER"], "as_of": [daily["date"].iloc[10]]})
        out = features.daily_state_frame(request, daily=daily)
        assert np.isnan(out["im"].iloc[0])

    def test_rows_without_an_iv_source_are_not_the_answer(self, daily):
        """`daily_market` also carries cap-only rows, which have no surface."""
        extra = daily.iloc[[11]].copy()
        extra["src_iv"] = None
        for column in ("implied_move", "iv10", "iv30"):
            extra[column] = np.nan
        seeded = pd.concat([daily.iloc[:11], extra], ignore_index=True)
        request = pd.DataFrame({"ticker": ["TEST"], "as_of": [daily["date"].iloc[11]]})
        out = features.daily_state_frame(request, daily=seeded)
        assert out["im"].iloc[0] == 10.0

    def test_preserves_request_order(self, daily):
        request = pd.DataFrame(
            {
                "ticker": ["TEST", "TEST"],
                "as_of": [daily["date"].iloc[20], daily["date"].iloc[5]],
            }
        )
        out = features.daily_state_frame(request, daily=daily)
        assert out["im"].tolist() == [20.0, 5.0]


class TestContextColumns:
    def test_the_context_loads_every_column_its_consumers_read(self):
        """A shared loader must satisfy every consumer, not just the first one.

        `add_orats_features` reads ORATS_FEATURES and `daily_state_frame` reads
        DAILY_STATE_FIELDS; the two overlap but neither contains the other.
        Loading only one set produces a KeyError deep inside a scoring call, on
        a column nobody was thinking about.
        """
        import inspect

        source = inspect.getsource(features.FeatureContext.load)
        assert "DAILY_STATE_FIELDS" in source
        assert "ORATS_FEATURES" in source

    def test_the_two_field_sets_genuinely_differ(self):
        """If they ever coincide, the test above is guarding nothing."""
        only_daily = set(features.DAILY_STATE_FIELDS) - set(panel_mod.ORATS_FEATURES)
        assert only_daily, "DAILY_STATE_FIELDS is now a subset — revisit the loader test"
        assert {"iv10", "exern_iv10", "spot"} <= only_daily


class TestFeatureLists:
    def test_history_features_are_all_panel_columns(self):
        for name in features.EVENT_HISTORY_FEATURES:
            assert name in panel_mod.PANEL_COLUMNS, name

    def test_market_state_is_excluded_from_the_history_block(self):
        """The distinction that keeps STR-RUNUP's early entry leak-free."""
        for leaky in ("or_iv30", "dist_high", "spy_vol20", "or_implied"):
            assert leaky not in features.EVENT_HISTORY_FEATURES

    def test_daily_state_columns_cover_levels_and_lags(self):
        assert "im" in features.DAILY_STATE_COLUMNS
        assert "im_d10" in features.DAILY_STATE_COLUMNS
        # Size and price levels are not differenced.
        assert "mcap_log_d1" not in features.DAILY_STATE_COLUMNS
        assert "spot_d1" not in features.DAILY_STATE_COLUMNS


# --------------------------------------------------------------------------
# market-state anchoring
# --------------------------------------------------------------------------


@pytest.fixture
def market_days():
    """400 business days of strictly-distinct closes.

    Strictly rising, so every anchor date maps to a different value and an
    off-by-one-session anchor cannot coincidentally agree.
    """
    days = pd.bdate_range("2023-01-02", periods=400)
    return days, 100.0 + np.arange(len(days), dtype=float)


@pytest.fixture
def gspc_file(tmp_path, market_days):
    """A GSPC daily CSV in the shape `_gspc_series` parses: three junk header rows."""
    days, closes = market_days
    lines = ["header", "header", "header"] + [
        f"{d.date()},{c},{c},{c},{c},{c},0" for d, c in zip(days, closes)
    ]
    path = tmp_path / "gspc.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def px_dir(tmp_path, market_days):
    days, closes = market_days
    pd.DataFrame({"date": days, "close_adj": closes}).to_csv(
        tmp_path / "px_T.csv", index=False
    )
    return tmp_path


@pytest.fixture
def orats_daily(market_days):
    days, closes = market_days
    frame = pd.DataFrame(
        {
            "ticker": "T",
            "date": days,
            "src_iv": 1.0,
            "mcap_usd": closes * 1e7,
            "mcap_log": np.log(closes * 1e7),
        }
    )
    for source in panel_mod.ORATS_FEATURES:
        frame[source] = closes
    return frame


@pytest.fixture
def event_frame(market_days):
    """One event, with a decision column three sessions earlier."""
    days, _ = market_days
    return pd.DataFrame(
        {
            "ticker": ["T"],
            "date": [days[300]],
            "dec": [days[297]],
            "move": [5.0],
            "n_prior": [20],
            "ema12_prior_abs_move": [4.0],
            "mean_prior_abs_move": [3.0],
        }
    )


class TestMarketBlockAnchoring:
    """The three market-state builders must honour the decision date.

    All three anchored on the EVENT date regardless of the caller's `as_of`.
    For a decision taken at the last pre-print close of a BMO name the two
    coincide, which is why this was invisible — and why moving the decision one
    session earlier would have handed every model a session of hindsight.
    """

    def test_regime_block_moves_with_the_anchor(self, gspc_file, event_frame, market_days):
        days, _ = market_days
        at_event = panel_mod.add_regime_features(event_frame.copy(), gspc_path=gspc_file)
        at_decision = panel_mod.add_regime_features(
            event_frame.copy(), gspc_path=gspc_file, as_of_column="dec"
        )
        # Event date → the close strictly before it. Decision date → that close
        # itself, which is a close we would trade at.
        assert at_event["regime_asof"].iloc[0] == days[299]
        assert at_decision["regime_asof"].iloc[0] == days[297]
        assert at_event["spy_ret21"].iloc[0] != at_decision["spy_ret21"].iloc[0]

    def test_runup_block_moves_with_the_anchor(self, px_dir, event_frame, market_days):
        days, _ = market_days
        at_event = panel_mod.add_runup_features(event_frame.copy(), px_dir=px_dir)
        at_decision = panel_mod.add_runup_features(
            event_frame.copy(), px_dir=px_dir, as_of_column="dec"
        )
        assert at_event["runup_asof"].iloc[0] == days[299]
        assert at_decision["runup_asof"].iloc[0] == days[297]
        assert at_event["ret5"].iloc[0] != at_decision["ret5"].iloc[0]

    def test_orats_block_moves_with_the_anchor(self, orats_daily, event_frame, market_days):
        days, _ = market_days
        at_event = panel_mod.add_orats_features(event_frame.copy(), daily=orats_daily)
        at_decision = panel_mod.add_orats_features(
            event_frame.copy(), daily=orats_daily, as_of_column="dec"
        )
        assert at_event["orats_asof"].iloc[0] == days[299]
        assert at_decision["orats_asof"].iloc[0] == days[297]
        assert at_event["or_implied"].iloc[0] != at_decision["or_implied"].iloc[0]
        # The cap has its own series and its own anchor, and must move too.
        assert at_event["mcap_asof"].iloc[0] != at_decision["mcap_asof"].iloc[0]

    def test_the_event_date_stays_a_hard_ceiling(self, orats_daily, event_frame, market_days):
        """A decision LATER than the panel's anchor must not move the value.

        This is what preserves the AMC convention: `last_pre_print` is the event
        date itself there, and the panel reads the row before it. If the
        decision's ceiling applied alone, every AMC name would silently gain a
        session — a modelling change smuggled in as a refactor.
        """
        days, _ = market_days
        event_frame = event_frame.copy()
        event_frame["late"] = days[305]
        at_event = panel_mod.add_orats_features(event_frame.copy(), daily=orats_daily)
        at_late = panel_mod.add_orats_features(
            event_frame.copy(), daily=orats_daily, as_of_column="late"
        )
        assert at_late["orats_asof"].iloc[0] == days[299]
        assert at_late["or_implied"].iloc[0] == at_event["or_implied"].iloc[0]

    def test_the_default_is_the_event_date(
        self, gspc_file, px_dir, orats_daily, event_frame
    ):
        """Omitting `as_of_column` must reproduce the pre-refactor panel exactly."""
        for call in (
            lambda f, **kw: panel_mod.add_regime_features(f, gspc_path=gspc_file, **kw),
            lambda f, **kw: panel_mod.add_runup_features(f, px_dir=px_dir, **kw),
            lambda f, **kw: panel_mod.add_orats_features(f, daily=orats_daily, **kw),
        ):
            implicit = call(event_frame.copy())
            explicit = call(event_frame.copy(), as_of_column="date")
            pd.testing.assert_frame_equal(implicit, explicit)

    def test_the_anchor_is_the_row_actually_read(self, orats_daily, event_frame, market_days):
        """The reported anchor must be the daily row's own date, not the request's.

        This is the property the stamp rests on. A stamp recomputed from `as_of`
        would say `dec`; the value came from the last row strictly before it.
        """
        days, closes = market_days
        out = panel_mod.add_orats_features(
            event_frame.copy(), daily=orats_daily, as_of_column="dec"
        )
        anchor = out["orats_asof"].iloc[0]
        assert anchor == days[297]
        assert anchor <= event_frame["dec"].iloc[0]
        assert out["or_implied"].iloc[0] == pytest.approx(closes[297])

    def test_build_panel_drops_the_anchor_columns(self):
        """Tier 3 stays byte-identical: provenance for a caller's as-of is not a feature."""
        for column in panel_mod.ANCHOR_COLUMNS:
            assert column not in panel_mod.PANEL_COLUMNS
            assert column not in features.PANEL_FEATURE_COLUMNS


class TestStamps:
    """A stamp is provenance, not a restatement of the request."""

    def test_each_block_is_stamped_at_its_own_anchor(self):
        as_of = pd.Timestamp("2024-05-07")
        stamps = features._stamps(
            ["spy_vol20", "dist_high", "or_iv30", "dte_entry"],
            as_of,
            market_dates={
                "regime_asof": pd.Timestamp("2024-05-06"),
                "runup_asof": pd.Timestamp("2024-05-03"),
                "orats_asof": pd.Timestamp("2024-05-02"),
            },
        )
        assert stamps["spy_vol20"] == pd.Timestamp("2024-05-06")
        assert stamps["dist_high"] == pd.Timestamp("2024-05-03")
        assert stamps["or_iv30"] == pd.Timestamp("2024-05-02")
        # Nothing else knows better than the decision close.
        assert stamps["dte_entry"] == as_of

    def test_blocks_resolve_to_different_dates(self):
        """If the three ever collapsed to one stamp, the split would guard nothing."""
        assert features._BLOCK_ANCHOR["spy_vol20"] == "regime_asof"
        assert features._BLOCK_ANCHOR["abs_dist_ema"] == "runup_asof"
        assert features._BLOCK_ANCHOR["mcap_log"] == "orats_asof"
        assert set(features._BLOCK_ANCHOR) == set(features._MARKET_BLOCK)

    def test_a_missing_anchor_falls_back_to_the_decision_close(self):
        as_of = pd.Timestamp("2024-05-07")
        stamps = features._stamps(["or_iv30"], as_of, market_dates={})
        assert stamps["or_iv30"] == as_of

    def test_an_anchor_after_the_decision_is_caught(self):
        """The regression this whole change exists for.

        A stamp derived from `as_of` can never exceed `as_of`, so `assert_causal`
        could not fail however stale the value was. A stamp taken from the
        value's own anchor can — and must.
        """
        from engine.audit import FeatureVector, LeakError, assert_causal

        as_of = pd.Timestamp("2024-05-07")
        leaked = FeatureVector(
            ticker="T",
            as_of=as_of,
            values={"or_iv30": 1.0},
            feature_as_of=features._stamps(
                ["or_iv30"], as_of,
                market_dates={"orats_asof": pd.Timestamp("2024-05-08")},
            ),
        )
        with pytest.raises(LeakError, match="or_iv30"):
            assert_causal(leaked)


class TestQuarantinedFeatures:
    """`or_exern_z252` is computed, kept for the Phase 0 reconciliation, and read
    by nothing.

    The guard in `add_orats_features` stops NEW panels carrying the leaked
    values; this stops anything consuming the ones already stored. Both halves
    are needed — a rebuild is not free, so old panels stay on disk.
    """

    def test_it_is_quarantined(self):
        assert "or_exern_z252" in features.QUARANTINED_FEATURES

    def test_it_stays_in_the_panel(self):
        """Quarantine is a READ rule. The column, its order, and the Phase 0
        reconciliation that depends on it are all untouched."""
        assert "or_exern_z252" in panel_mod.PANEL_COLUMNS

    def test_no_model_can_read_it(self):
        assert "or_exern_z252" not in features.PANEL_FEATURE_COLUMNS
        assert "or_exern_z252" not in features._MARKET_BLOCK
        assert "or_exern_z252" not in features._BLOCK_ANCHOR

    def test_the_scorer_does_not_copy_it_into_the_feature_frame(self):
        from engine.score import _PANEL_MARKET_BLOCK

        assert "or_exern_z252" not in _PANEL_MARKET_BLOCK

    def test_a_model_listing_it_is_rejected(self):
        with pytest.raises(features.QuarantinedFeature, match="or_exern_z252"):
            features.assert_not_quarantined(["mcap_log", "or_exern_z252"])

    def test_a_clean_list_passes(self):
        features.assert_not_quarantined(["mcap_log", "or_iv30"])

    def test_no_registry_entry_lists_it(self):
        """Champions AND retired entries — a retired model can be re-promoted."""
        import json

        from engine import paths

        registry = json.loads((paths.ROOT / "engine/models/registry.json").read_text())
        for model in registry["models"]:
            assert "or_exern_z252" not in (model.get("features") or []), model["id"]


class TestZ252Guard:
    """The leak itself: a z-score for an event with no prior daily row."""

    def test_no_prior_daily_row_yields_nan(self, event_frame, orats_daily, market_days):
        """Before the fix this read `exern[-1]` — the LAST row of the series,
        years after the print — against the whole history."""
        days, _ = market_days
        early = event_frame.copy()
        early["date"] = days[0]          # before every daily row
        early["dec"] = days[0]
        out = panel_mod.add_orats_features(early, daily=orats_daily)
        assert pd.isna(out["or_exern_z252"].iloc[0])
        # And the block it belongs to is absent too, which is the tell that no
        # daily row was found at all.
        assert pd.isna(out["or_implied"].iloc[0])
        assert pd.isna(out["orats_asof"].iloc[0])

    def test_a_prior_row_still_produces_a_score(self, event_frame, orats_daily):
        out = panel_mod.add_orats_features(event_frame.copy(), daily=orats_daily)
        assert np.isfinite(out["or_exern_z252"].iloc[0])


class TestPanelFeaturesStaleMarketBlock:
    """The panel's market block is baked at the event date and cannot be re-aimed."""

    @pytest.fixture
    def context(self):
        from engine.calendar import trading_calendar

        rows = []
        for i, date in enumerate(pd.to_datetime(["2024-02-08", "2024-05-09"])):
            row = {name: 1.0 for name in features.PANEL_FEATURE_COLUMNS}
            row.update({"ticker": "T", "k": i, "date": date, "quarter": "Q1",
                        "move": 2.0, "abs_move": 2.0, "year": date.year,
                        "mcap_asof": date})
            rows.append(row)
        return features.FeatureContext(
            panel=pd.DataFrame(rows), daily=None, calendar=trading_calendar()
        )

    def test_served_at_the_last_pre_print_close(self, context):
        vector = features.panel_features(
            "T", "2024-05-09", as_of=pd.Timestamp("2024-05-08"),
            session="BMO", context=context,
        )
        assert vector.values["or_iv30"] == 1.0
        assert vector.meta["market_block_withheld"] is False

    def test_withheld_one_session_earlier(self, context):
        """A D−1 decision cannot read a block anchored at D0. NaN, not a stale number."""
        vector = features.panel_features(
            "T", "2024-05-09", as_of=pd.Timestamp("2024-05-07"),
            session="BMO", context=context,
        )
        assert vector.meta["market_block_withheld"] is True
        for name in features._MARKET_BLOCK:
            if name in vector.values:
                assert np.isnan(vector.values[name]), name

    def test_event_history_survives_the_withholding(self, context):
        """Only the market block is anchored at the event; the recursions are not."""
        vector = features.panel_features(
            "T", "2024-05-09", as_of=pd.Timestamp("2024-05-07"),
            session="BMO", context=context,
        )
        assert vector.values["mean_prior_move"] == 1.0
