"""Tier 4 — the forecast table, and the causality its own audit cannot see.

Tier 3's leak discipline is enforced by dates and checked by ``assert_causal``,
which compares feature stamps against ``as_of``. That machinery is structurally
blind to the failure this layer can have: a leak inside a model's *training
set*. A row built from a model fit on everything would carry perfectly ordered
feature stamps and be worthless. So the first test here refits from scratch on
``< fold_start`` and demands the stored number back, exactly.

The second is the one whose failure would be silent and cumulative: if a
``--since`` rebuild and a full rebuild can disagree, every backfill quietly
corrupts the table and nothing downstream would notice.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from engine.data.features import tier4
from engine.data.features.tier4 import (
    COLUMNS,
    FIRST_FOLD,
    FeatureModel,
    Tier4Error,
    build_forecasts,
    fit_fold,
    fold_start_of,
    load_forecasts,
    normalize,
    training_frames,
    write_forecasts,
)


# --------------------------------------------------------------------------
# a cheap, exactly reproducible feature model
# --------------------------------------------------------------------------


class _Ridgeless:
    """Least squares with an intercept. Deterministic to the last bit.

    The champion is an OLS+MLP blend that takes seconds to fit; these tests need
    hundreds of fits and care about *reproducibility*, not architecture. A model
    whose refit is bitwise identical is what lets the leak test assert equality
    rather than a tolerance — and a tolerance is exactly how a subtle leak would
    hide.
    """

    def __init__(self, coef: np.ndarray) -> None:
        self.coef = coef

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray) -> "_Ridgeless":
        design = np.column_stack([np.ones(len(X)), X])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        return cls(coef)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), X]) @ self.coef


def _fit(X, y, seed):  # noqa: ARG001 - the signature the harness calls
    return _Ridgeless.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))


def _prepare(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["n_prior"] >= 4].reset_index(drop=True)


MODEL = FeatureModel(
    model_id="test_size",
    produces="pred_abs_move",
    features=("f1", "f2"),
    target="abs_move",
    fit=_fit,
    prepare=_prepare,
    seed=1,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    """Four years of monthly events across enough tickers to clear MIN_TRAIN_ROWS."""
    rng = np.random.default_rng(20260904)
    dates = pd.date_range("2012-01-15", "2015-12-15", freq="MS") + pd.Timedelta(days=14)
    tickers = [f"T{i:03d}" for i in range(40)]
    rows = []
    for ticker in tickers:
        for k, date in enumerate(dates):
            f1, f2 = rng.normal(size=2)
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "k": k,
                    "n_prior": k,
                    "f1": f1,
                    "f2": f2,
                    "abs_move": 5.0 + 2.0 * f1 - 1.5 * f2 + rng.normal(scale=0.5),
                }
            )
    frame = pd.DataFrame(rows)
    frame["date"] = frame["date"].astype("datetime64[us]")
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


@pytest.fixture
def built(panel, monkeypatch):
    monkeypatch.setattr(tier4, "FIRST_FOLD", pd.Timestamp("2013-01-01"))
    return build_forecasts(panel, model=MODEL, tier3_snapshot="snap", log=lambda _m: None)


# --------------------------------------------------------------------------
# §9 — the acceptance tests from the design note
# --------------------------------------------------------------------------


class TestARowNeverSeesItsOwnPeriod:
    def test_refitting_on_before_fold_start_reproduces_the_value_exactly(self, panel, built):
        _, trainable = training_frames(panel, MODEL)
        scored = built[built["pred_abs_move"].notna()]
        assert len(scored) > 0

        for fold, group in list(scored.groupby("fold_start"))[::7]:
            model = fit_fold(trainable, MODEL, fold)
            rows = panel.merge(
                group[["ticker", "event_date", "pred_abs_move"]],
                left_on=["ticker", "date"],
                right_on=["ticker", "event_date"],
            )
            expected = model.predict(rows[list(MODEL.features)].to_numpy(dtype=float))
            assert np.array_equal(expected, rows["pred_abs_move"].to_numpy())

    def test_a_model_fit_on_everything_would_fail_that_test(self, panel, built):
        # The leak this layer can have, made concrete: refit on the full sample
        # and the stored values no longer reproduce. If this ever passes, the
        # test above has stopped being able to detect anything.
        _, trainable = training_frames(panel, MODEL)
        leaky = _fit(
            trainable[list(MODEL.features)].to_numpy(dtype=float),
            trainable[MODEL.target].to_numpy(dtype=float),
            MODEL.seed,
        )
        scored = built[built["pred_abs_move"].notna()]
        rows = panel.merge(
            scored[["ticker", "event_date", "pred_abs_move"]],
            left_on=["ticker", "date"],
            right_on=["ticker", "event_date"],
        )
        leaked = leaky.predict(rows[list(MODEL.features)].to_numpy(dtype=float))
        assert not np.allclose(leaked, rows["pred_abs_move"].to_numpy())

    def test_the_training_pool_stops_strictly_before_the_fold(self, panel):
        _, trainable = training_frames(panel, MODEL)
        fold = pd.Timestamp("2014-06-01")
        inside = trainable[
            (trainable["date"] >= fold) & (trainable["date"] < fold + pd.offsets.MonthBegin(1))
        ]
        assert len(inside) > 0, "fixture must have events inside the fold to be a real test"

        model = fit_fold(trainable, MODEL, fold)
        without = _fit(
            trainable[trainable["date"] < fold][list(MODEL.features)].to_numpy(dtype=float),
            trainable[trainable["date"] < fold][MODEL.target].to_numpy(dtype=float),
            MODEL.seed,
        )
        assert np.array_equal(model.coef, without.coef)


class TestSinceEquivalence:
    """Non-negotiable: if incremental and full disagree, every backfill corrupts."""

    @pytest.mark.parametrize("since", ["2014-01-01", "2014-07-14", "2015-03-01"])
    def test_incremental_matches_a_full_rebuild(self, panel, built, since):
        incremental = build_forecasts(
            panel,
            model=MODEL,
            since=since,
            existing=built,
            tier3_snapshot="snap",
            log=lambda _m: None,
        )
        pd.testing.assert_frame_equal(incremental, built)

    def test_since_is_rounded_down_to_its_fold(self, panel, built):
        mid = build_forecasts(
            panel, model=MODEL, since="2014-07-14", existing=built,
            tier3_snapshot="snap", log=lambda _m: None,
        )
        start = build_forecasts(
            panel, model=MODEL, since="2014-07-01", existing=built,
            tier3_snapshot="snap", log=lambda _m: None,
        )
        pd.testing.assert_frame_equal(mid, start)

    def test_a_partial_build_is_visible_in_the_provenance(self, panel, built):
        # The point of storing tier3_snapshot per row rather than per file: a
        # table stitched from two Tier-3 states says so.
        rebuilt = build_forecasts(
            panel, model=MODEL, since="2015-01-01", existing=built,
            tier3_snapshot="different", log=lambda _m: None,
        )
        assert set(rebuilt["tier3_snapshot"].unique()) == {"snap", "different"}
        assert rebuilt.loc[rebuilt["event_date"] < "2015-01-01", "tier3_snapshot"].eq("snap").all()

    def test_a_round_trip_through_parquet_survives_the_comparison(self, panel, built, tmp_path):
        path = tmp_path / "tier4.parquet"
        write_forecasts(built, path)
        pd.testing.assert_frame_equal(load_forecasts(path), built)


class TestCarryOverGuards:
    def test_a_cadence_change_refuses_the_carry_over(self, panel, built):
        tampered = built.copy()
        stamped = tampered["fold_start"].notna()
        tampered.loc[stamped, "fold_start"] = tampered.loc[stamped, "fold_start"] + pd.Timedelta(
            days=3
        )
        with pytest.raises(Tier4Error, match="cadence"):
            build_forecasts(
                panel, model=MODEL, since="2015-01-01", existing=tampered,
                tier3_snapshot="snap", log=lambda _m: None,
            )

    def test_a_different_model_refuses_the_carry_over(self, panel, built):
        other = replace(MODEL, model_id="test_size_v2")
        with pytest.raises(Tier4Error, match="promotion invalidates Tier 4"):
            build_forecasts(
                panel, model=other, since="2015-01-01", existing=built,
                tier3_snapshot="snap", log=lambda _m: None,
            )

    def test_events_added_inside_the_carried_prefix_refuse_the_carry_over(self, panel, built):
        # A backfill at date D that added events; carrying the prefix over would
        # leave holes in a table consumers rely on being total.
        thinned = built[built["event_date"] != built["event_date"].min()]
        with pytest.raises(Tier4Error, match="permanent holes"):
            build_forecasts(
                panel, model=MODEL, since="2015-01-01", existing=thinned,
                tier3_snapshot="snap", log=lambda _m: None,
            )


class TestTotalityAndNulls:
    def test_every_tier3_event_gets_a_row(self, panel, built):
        assert len(built) == len(panel)
        keys = set(map(tuple, built[["ticker", "event_date"]].to_numpy()))
        assert keys == set(map(tuple, panel[["ticker", "date"]].to_numpy()))

    def test_rows_before_the_first_fold_carry_a_null_forecast(self, built):
        early = built[built["event_date"] < FIRST_FOLD]
        assert len(early) > 0
        assert early["pred_abs_move"].isna().all()
        assert early["model_id"].isna().all()
        assert early["fold_start"].isna().all()

    def test_null_is_not_zero(self, built):
        # A consumer that reads a NULL forecast as 0.0 would size a structure to
        # nothing rather than declining to size it. Nothing in the pipeline may
        # fill these.
        nulls = built[built["pred_abs_move"].isna()]
        assert len(nulls) > 0
        assert not (nulls["pred_abs_move"] == 0).any()

    def test_a_row_with_a_forecast_always_names_its_model_and_fold(self, built):
        scored = built[built["pred_abs_move"].notna()]
        assert scored["model_id"].notna().all()
        assert scored["fold_start"].notna().all()
        assert (scored["fold_start"] <= scored["event_date"]).all()

    def test_no_row_carries_provenance_without_a_forecast(self, built):
        blank = built[built["pred_abs_move"].isna()]
        assert blank["model_id"].isna().all()
        assert blank["fold_start"].isna().all()

    def test_an_unrealized_event_still_gets_a_forecast(self, panel, monkeypatch):
        # The distinction that makes Tier 4 usable live: a prediction needs
        # complete features, not a realized target. Conflating the two would
        # leave every upcoming event — the only ones that can still be traded —
        # without a forecast.
        monkeypatch.setattr(tier4, "FIRST_FOLD", pd.Timestamp("2013-01-01"))
        upcoming = panel.copy()
        last = upcoming["date"] == upcoming["date"].max()
        upcoming.loc[last, "abs_move"] = np.nan
        frame = build_forecasts(
            upcoming, model=MODEL, tier3_snapshot="snap", log=lambda _m: None
        )
        tail = frame[frame["event_date"] == upcoming["date"].max()]
        assert tail["pred_abs_move"].notna().all()


class TestFolds:
    def test_a_fold_start_is_the_first_of_its_month(self):
        stamps = pd.to_datetime(["2024-01-01", "2024-01-31", "2024-02-29", "2013-12-15"])
        assert list(fold_start_of(stamps).dt.strftime("%Y-%m-%d")) == [
            "2024-01-01",
            "2024-01-01",
            "2024-02-01",
            "2013-12-01",
        ]

    def test_every_stored_fold_start_is_a_legal_boundary(self, built):
        stamped = built["fold_start"].dropna()
        assert len(stamped) > 0
        assert (fold_start_of(stamped).to_numpy() == stamped.to_numpy()).all()

    def test_a_thin_training_pool_is_skipped_rather_than_fit(self, panel, monkeypatch):
        monkeypatch.setattr(tier4, "MIN_TRAIN_ROWS", 10**9)
        frame = build_forecasts(panel, model=MODEL, tier3_snapshot="snap", log=lambda _m: None)
        assert frame["pred_abs_move"].isna().all()

    def test_fit_fold_refuses_a_pool_under_the_floor(self, panel):
        _, trainable = training_frames(panel, MODEL)
        with pytest.raises(Tier4Error, match="trainable rows"):
            fit_fold(trainable, MODEL, "2012-02-01")


class TestSchema:
    def test_the_table_is_narrow(self, built):
        assert tuple(built.columns) == COLUMNS

    def test_normalize_is_idempotent(self, built):
        pd.testing.assert_frame_equal(normalize(built), built)

    def test_a_duplicate_tier3_key_is_refused(self, panel):
        doubled = pd.concat([panel, panel.head(1)], ignore_index=True)
        with pytest.raises(Tier4Error, match="duplicate"):
            build_forecasts(doubled, model=MODEL, tier3_snapshot="snap", log=lambda _m: None)


class TestLiveAndHistoricalAgree:
    def test_the_current_folds_model_is_the_one_the_table_used(self, panel, built):
        # The property monthly cadence buys: the model the live scorer needs for
        # an upcoming event IS the current fold's model, so the board and the
        # table agree by construction rather than by a test that hopes they do.
        # This asserts the construction, on the fold that is currently "live".
        _, trainable = training_frames(panel, MODEL)
        live_fold = built["fold_start"].max()
        served = fit_fold(trainable, MODEL, live_fold)

        rows = panel[fold_start_of(panel["date"]).to_numpy() == live_fold]
        rows = _prepare(rows)
        stored = built[built["fold_start"] == live_fold]
        merged = rows.merge(
            stored[["ticker", "event_date", "pred_abs_move"]],
            left_on=["ticker", "date"],
            right_on=["ticker", "event_date"],
        )
        assert len(merged) > 0
        live = served.predict(merged[list(MODEL.features)].to_numpy(dtype=float))
        assert np.array_equal(live, merged["pred_abs_move"].to_numpy())


class TestTier3IsUnchanged:
    def test_building_tier4_does_not_touch_the_panel(self, panel, tmp_path, monkeypatch):
        from engine import paths
        from engine.data import store

        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path, index=False)
        before = store.file_sha256(panel_path)

        monkeypatch.setattr(tier4, "FIRST_FOLD", pd.Timestamp("2013-01-01"))
        monkeypatch.setattr(paths, "TIER4", tmp_path / "tier4_forecasts.parquet")
        frame = build_forecasts(
            panel, model=MODEL, tier3_snapshot=before, log=lambda _m: None
        )
        write_forecasts(frame)

        assert paths.TIER4.exists()
        assert store.file_sha256(panel_path) == before

    def test_the_tier4_hash_is_not_folded_into_the_snapshot(self, tmp_root, monkeypatch):
        # An experiment that never reads a forecast must not be invalidated by a
        # Tier-4 rebuild. That is the whole reason Tier 4 is not inside Tier 3.
        import importlib

        from engine import paths
        from engine.data import manifest

        importlib.reload(manifest)
        monkeypatch.setattr(manifest, "collect_stats", lambda: {})
        paths.TIER4.parent.mkdir(parents=True, exist_ok=True)

        before = manifest.snapshot_hash({})
        paths.TIER4.write_bytes(b"a materially different tier 4")
        assert manifest.snapshot_hash({}) == before
        assert manifest._tier4_digest() is not None


class TestRegistryGraph:
    def test_the_vocabulary_matches_what_tier4_actually_builds(self):
        from engine.models import registry as reg

        assert set(tier4.FORECAST_COLUMNS) == set(reg.TIER4_COLUMNS)

    def test_a_role_lands_on_its_tier_without_anyone_editing_the_entry(self):
        from engine.models import registry as reg

        base = dict(
            id="x", strategy="*", artifact="a", artifact_sha256="h",
            features=["f"], target="t", train_window="w",
        )
        assert reg.RegistryEntry(role="size", **base).tier == "feature"
        assert reg.RegistryEntry(role="gate", **base).tier == "decision"

    def test_a_feature_model_may_not_consume(self):
        from engine.models import registry as reg

        with pytest.raises(reg.RegistryError, match="may not declare `consumes`"):
            reg.RegistryEntry(
                id="x", role="size", strategy="*", artifact="a", artifact_sha256="h",
                features=["f"], target="t", train_window="w",
                consumes=["pred_abs_move"],
            )

    def test_a_decision_model_may_not_produce(self):
        from engine.models import registry as reg

        with pytest.raises(reg.RegistryError, match="produces no Tier-4 column"):
            reg.RegistryEntry(
                id="x", role="gate", strategy="S", artifact="a", artifact_sha256="h",
                features=["f"], target="t", train_window="w",
                produces="pred_abs_move",
            )

    def test_an_unknown_tier4_column_is_refused(self):
        from engine.models import registry as reg

        with pytest.raises(reg.RegistryError, match="not a Tier-4 column"):
            reg.RegistryEntry(
                id="x", role="size", strategy="*", artifact="a", artifact_sha256="h",
                features=["f"], target="t", train_window="w",
                produces="pred_something_else",
            )

    def test_two_champion_producers_are_a_problem(self):
        from engine.models import registry as reg

        def entry(model_id, strategy):
            return reg.RegistryEntry(
                id=model_id, role="size", strategy=strategy, artifact="a",
                artifact_sha256="h", features=["f"], target="t", train_window="w",
                produces="pred_abs_move", champion=True,
            )

        registry = reg.Registry(entries=[entry("a", "*"), entry("b", "STR-THRU")])
        problems = registry.validate(check_artifacts=False)
        assert any("more than one champion" in p for p in problems)

    def test_a_consumer_with_no_producer_is_a_problem(self):
        from engine.models import registry as reg

        registry = reg.Registry(
            entries=[
                reg.RegistryEntry(
                    id="g", role="gate", strategy="TWIN-P", artifact="a",
                    artifact_sha256="h", features=["f"], target="t", train_window="w",
                    consumes=["pred_abs_move"], champion=True, threshold=0.0,
                )
            ]
        )
        problems = registry.validate(check_artifacts=False)
        assert any("no champion produces it" in p for p in problems)

    def test_the_shipped_registry_declares_one_producer_for_the_forecast(self):
        from engine.models import registry as reg

        graph = reg.load_registry().tier4_graph()
        assert graph["pred_abs_move"]["produced_by"] == ["size_v1_4"]

    def test_the_shipped_registry_is_clean(self):
        from engine.models import registry as reg

        assert reg.load_registry().validate(check_artifacts=False) == []


class TestTheReadTimeJoin:
    """``load_panel(with_forecasts=True)`` — opt-in, left, and total."""

    @pytest.fixture
    def wired(self, panel, built, tmp_path, monkeypatch):
        from engine import paths

        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path, index=False)
        monkeypatch.setattr(paths, "TIER4", tmp_path / "tier4_forecasts.parquet")
        write_forecasts(built)
        return panel_path, built

    def test_the_join_adds_the_forecast_without_adding_or_losing_rows(self, wired):
        from engine.features import FORECAST_COLUMNS, load_panel

        panel_path, built = wired
        plain = load_panel(panel_path)
        joined = load_panel(panel_path, with_forecasts=True)

        assert len(joined) == len(plain)
        assert set(joined.columns) == set(plain.columns) | set(FORECAST_COLUMNS)
        assert "event_date" not in joined.columns
        assert joined["pred_abs_move"].notna().sum() == built["pred_abs_move"].notna().sum()

    def test_the_forecast_lands_on_the_right_event(self, wired):
        from engine.features import load_panel

        panel_path, built = wired
        joined = load_panel(panel_path, with_forecasts=True)
        check = joined.merge(
            built, left_on=["ticker", "date"], right_on=["ticker", "event_date"],
            suffixes=("", "_t4"),
        )
        assert len(check) == len(joined)
        pd.testing.assert_series_equal(
            check["pred_abs_move"], check["pred_abs_move_t4"], check_names=False
        )

    def test_the_default_stays_tier3_only(self, wired):
        from engine.features import FORECAST_COLUMNS, load_panel

        panel_path, _ = wired
        plain = load_panel(panel_path)
        assert not set(FORECAST_COLUMNS) & set(plain.columns)

    def test_asking_for_forecasts_that_were_never_built_is_an_error(
        self, panel, tmp_path, monkeypatch
    ):
        # Silently returning a NULL column would be indistinguishable from a
        # model that predicts nothing, and a consumer would decline every trade
        # for a reason it could not diagnose.
        from engine import paths
        from engine.features import load_panel

        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path, index=False)
        monkeypatch.setattr(paths, "TIER4", tmp_path / "absent.parquet")
        with pytest.raises(FileNotFoundError, match="engine.data.features.tier4"):
            load_panel(panel_path, with_forecasts=True)
