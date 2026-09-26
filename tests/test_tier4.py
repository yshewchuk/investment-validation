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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.data.features import tier4
from engine.data.features.tier4 import (
    COLUMNS,
    FIRST_FOLD,
    FeatureModel,
    ServingModel,
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

#: These tests exercise ONE producer with a bitwise-reproducible fit. The other
#: registered producer is a real champion whose artifact and event window are
#: not available to a unit test, so every build here names the group it means.
_ONLY = ("pred_abs_move",)
_MODELS = {"pred_abs_move": MODEL}


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
    return build_forecasts(panel, produces=_ONLY, models=_MODELS, tier3_snapshot="snap", log=lambda _m: None)


# --------------------------------------------------------------------------
# §9 — the acceptance tests from the design note
# --------------------------------------------------------------------------


class TestARowNeverSeesItsOwnPeriod:
    def test_refitting_on_before_fold_start_reproduces_the_value_exactly(self, panel, built):
        _, trainable = training_frames(panel, MODEL)
        scored = built[built["pred_abs_move"].notna()]
        assert len(scored) > 0

        for fold, group in list(scored.groupby("pred_abs_move_fold_start"))[::7]:
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
            produces=_ONLY,
            models=_MODELS,
            since=since,
            existing=built,
            tier3_snapshot="snap",
            log=lambda _m: None,
        )
        pd.testing.assert_frame_equal(incremental, built)

    def test_since_is_rounded_down_to_its_fold(self, panel, built):
        mid = build_forecasts(
            panel, produces=_ONLY, models=_MODELS, since="2014-07-14", existing=built,
            tier3_snapshot="snap", log=lambda _m: None,
        )
        start = build_forecasts(
            panel, produces=_ONLY, models=_MODELS, since="2014-07-01", existing=built,
            tier3_snapshot="snap", log=lambda _m: None,
        )
        pd.testing.assert_frame_equal(mid, start)

    def test_a_partial_build_is_visible_in_the_provenance(self, panel, built):
        # The point of storing tier3_snapshot per row rather than per file: a
        # table stitched from two Tier-3 states says so.
        rebuilt = build_forecasts(
            panel, produces=_ONLY, models=_MODELS, since="2015-01-01", existing=built,
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
        stamped = tampered["pred_abs_move_fold_start"].notna()
        tampered.loc[stamped, "pred_abs_move_fold_start"] = tampered.loc[stamped, "pred_abs_move_fold_start"] + pd.Timedelta(
            days=3
        )
        with pytest.raises(Tier4Error, match="cadence"):
            build_forecasts(
                panel, produces=_ONLY, models=_MODELS, since="2015-01-01", existing=tampered,
                tier3_snapshot="snap", log=lambda _m: None,
            )

    def test_a_different_model_refuses_the_carry_over(self, panel, built):
        other = replace(MODEL, model_id="test_size_v2")
        with pytest.raises(Tier4Error, match="promotion invalidates Tier 4"):
            build_forecasts(
                panel, produces=_ONLY, models={"pred_abs_move": other},
                since="2015-01-01", existing=built,
                tier3_snapshot="snap", log=lambda _m: None,
            )

    def test_events_added_inside_the_carried_prefix_refuse_the_carry_over(self, panel, built):
        # A backfill at date D that added events; carrying the prefix over would
        # leave holes in a table consumers rely on being total.
        thinned = built[built["event_date"] != built["event_date"].min()]
        with pytest.raises(Tier4Error, match="permanent holes"):
            build_forecasts(
                panel, produces=_ONLY, models=_MODELS, since="2015-01-01", existing=thinned,
                tier3_snapshot="snap", log=lambda _m: None,
            )


class TestPrefixGapBackfill:
    """Regression test for the 2026-09 catch-up nightlies (logs 09-11, 09-14
    through 09-17, and 09-25 all show the SAME error shape:
    ``Tier4Error: Tier 3 has N events before <fold> that the existing Tier-4
    table does not cover``) and for the identical gap still open on disk today
    in ``data/features/tier4_forecasts.parquet`` (3 rows, one ticker, three
    2021 dates).

    The fixture is deliberately identical to
    ``TestCarryOverGuards.test_events_added_inside_the_carried_prefix_refuse_the_carry_over``
    above — same ``thinned`` construction, same ``since`` — because it is the
    same event, given a different requirement. That test locks in today's
    behaviour (refuse and raise). This one states what an incremental build
    must do about a gap instead of refusing outright: recompute the one fold
    the gap sits in and return a table that is total over Tier 3 again,
    the same as a full rebuild would.

    THIS TEST IS EXPECTED TO FAIL until the gap-fill fix lands — it exists to
    let a reviewer see the exact acceptance bar the fix has to clear before
    any fix code is written.
    """

    def test_a_gap_in_the_carried_prefix_is_backfilled_not_refused(self, panel, built):
        # Must be a SCORED event (fold_start >= FIRST_FOLD), not the panel's
        # earliest date overall — an unscored prefix row carries a null
        # forecast and would let this pass without exercising recomputation.
        gap_date = built.loc[built["pred_abs_move"].notna(), "event_date"].min()
        thinned = built[built["event_date"] != gap_date]

        # Today's incremental build raises Tier4Error here (see the sibling
        # test in TestCarryOverGuards) and the caller — the legacy nightly's
        # step 2b — treats that as "Tier 3/Tier 4 not rebuilt" and moves on,
        # leaving the gap unfilled for every night after. The fix must make
        # this call succeed instead of raising.
        backfilled = build_forecasts(
            panel, produces=_ONLY, models=_MODELS, since="2015-01-01",
            existing=thinned, tier3_snapshot="snap", log=lambda _m: None,
        )

        # Total over Tier 3 again, exactly like a full rebuild.
        assert len(backfilled) == len(panel)
        gap_rows = backfilled[backfilled["event_date"] == gap_date]
        assert len(gap_rows) == (panel["date"] == gap_date).sum()

        # The backfilled row(s) must match a full rebuild bit-for-bit — a
        # gap-fill that disagrees with a full rebuild is the same silent
        # corruption TestSinceEquivalence exists to catch for the ordinary
        # incremental path.
        pd.testing.assert_frame_equal(
            gap_rows.sort_values("ticker").reset_index(drop=True),
            built[built["event_date"] == gap_date].sort_values("ticker").reset_index(drop=True),
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
        assert early["pred_abs_move_model_id"].isna().all()
        assert early["pred_abs_move_fold_start"].isna().all()

    def test_null_is_not_zero(self, built):
        # A consumer that reads a NULL forecast as 0.0 would size a structure to
        # nothing rather than declining to size it. Nothing in the pipeline may
        # fill these.
        nulls = built[built["pred_abs_move"].isna()]
        assert len(nulls) > 0
        assert not (nulls["pred_abs_move"] == 0).any()

    def test_a_row_with_a_forecast_always_names_its_model_and_fold(self, built):
        scored = built[built["pred_abs_move"].notna()]
        assert scored["pred_abs_move_model_id"].notna().all()
        assert scored["pred_abs_move_fold_start"].notna().all()
        assert (scored["pred_abs_move_fold_start"] <= scored["event_date"]).all()

    def test_no_row_carries_provenance_without_a_forecast(self, built):
        blank = built[built["pred_abs_move"].isna()]
        assert blank["pred_abs_move_model_id"].isna().all()
        assert blank["pred_abs_move_fold_start"].isna().all()

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
            upcoming, produces=_ONLY, models=_MODELS, tier3_snapshot="snap", log=lambda _m: None
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
        stamped = built["pred_abs_move_fold_start"].dropna()
        assert len(stamped) > 0
        assert (fold_start_of(stamped).to_numpy() == stamped.to_numpy()).all()

    def test_a_thin_training_pool_is_skipped_rather_than_fit(self, panel, monkeypatch):
        monkeypatch.setattr(tier4, "MIN_TRAIN_ROWS", 10**9)
        frame = build_forecasts(panel, produces=_ONLY, models=_MODELS, tier3_snapshot="snap", log=lambda _m: None)
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
            build_forecasts(doubled, produces=_ONLY, models=_MODELS, tier3_snapshot="snap", log=lambda _m: None)


class TestLiveAndHistoricalAgree:
    def test_the_current_folds_model_is_the_one_the_table_used(self, panel, built):
        # The property monthly cadence buys: the model the live scorer needs for
        # an upcoming event IS the current fold's model, so the board and the
        # table agree by construction rather than by a test that hopes they do.
        # This asserts the construction, on the fold that is currently "live".
        _, trainable = training_frames(panel, MODEL)
        live_fold = built["pred_abs_move_fold_start"].max()
        served = fit_fold(trainable, MODEL, live_fold)

        rows = panel[fold_start_of(panel["date"]).to_numpy() == live_fold]
        rows = _prepare(rows)
        stored = built[built["pred_abs_move_fold_start"] == live_fold]
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
            panel, produces=_ONLY, models=_MODELS, tier3_snapshot=before, log=lambda _m: None
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


class TestTheInterval:
    """The 80% band and the residual SD, and the causality they inherit.

    An interval computed from errors the model had not yet made is the same
    leak as a forecast computed from data it had not yet seen — wearing a
    different hat, and not visible to the point-forecast tests at all.
    """

    def test_the_band_brackets_the_forecast(self, built):
        # Restricted to non-negative forecasts, and that is not a dodge. The
        # floor at zero is unconditional, so a NEGATIVE prediction — which this
        # fixture's linear target can produce and the real size model does not
        # (0 of 85,618, min +0.39) — ends up below its own p10. That is the
        # floor working: a magnitude cannot be negative, and the row is refused
        # by `forecast_params` before it can size anything.
        banded = built.dropna(subset=["pred_abs_move_p10", "pred_abs_move_p90"])
        banded = banded[banded["pred_abs_move"] >= 0]
        assert len(banded) > 0
        assert (banded["pred_abs_move_p10"] <= banded["pred_abs_move"]).all()
        assert (banded["pred_abs_move"] <= banded["pred_abs_move_p90"]).all()

    def test_the_band_is_ordered_everywhere(self, built):
        banded = built.dropna(subset=["pred_abs_move_p10", "pred_abs_move_p90"])
        assert (banded["pred_abs_move_p10"] <= banded["pred_abs_move_p90"]).all()

    def test_a_negative_forecast_is_floored_and_never_sized(self, built):
        from engine.forecast_sizing import forecast_params

        negative = built[built["pred_abs_move"] < 0].dropna(subset=["pred_abs_move_p10"])
        assert len(negative) > 0, "fixture must produce one to be a real test"
        assert (negative["pred_abs_move_p10"] == 0.0).all()
        for value in negative["pred_abs_move"]:
            assert forecast_params("TWIN-P", value) is None

    def test_the_lower_bound_is_never_negative(self, built):
        # The target is a MAGNITUDE. An interval whose floor is below zero is
        # reporting an outcome that cannot happen.
        banded = built["pred_abs_move_p10"].dropna()
        assert len(banded) > 0
        assert (banded >= 0).all()

    def test_the_sd_is_positive_where_it_exists(self, built):
        sd = built["pred_abs_move_sd"].dropna()
        assert len(sd) > 0
        assert (sd > 0).all()

    def test_a_band_always_says_how_many_errors_it_came_from(self, built):
        banded = built["pred_abs_move_sd"].notna()
        assert built.loc[banded, "pred_abs_move_resid_n"].notna().all()
        assert (built.loc[banded, "pred_abs_move_resid_n"] >= tier4.MIN_RESIDUALS).all()
        assert built.loc[~banded, "pred_abs_move_resid_n"].isna().all()

    def test_the_earliest_folds_carry_no_band(self, built):
        # Nothing has been predicted yet, so there are no held-out errors to
        # build one from. A band from four residuals would be a number that
        # reads as a confidence interval and is not one.
        scored = built[built["pred_abs_move"].notna()]
        first = scored["pred_abs_move_fold_start"].min()
        assert scored.loc[scored["pred_abs_move_fold_start"] == first, "pred_abs_move_sd"].isna().all()
        assert scored["pred_abs_move_sd"].notna().any(), "no fold ever gets a band"

    def test_the_band_appears_once_and_stays(self, built):
        # Monotone in fold order: the pool only grows, so a fold that has a band
        # is never followed by one that lost it.
        scored = built[built["pred_abs_move"].notna()]
        by_fold = scored.groupby("pred_abs_move_fold_start")["pred_abs_move_sd"].apply(
            lambda s: bool(s.notna().any())
        )
        seen = False
        for has_band in by_fold.sort_index():
            if has_band:
                seen = True
            assert not (seen and not has_band), "a fold lost a band it already had"

    def test_a_row_with_no_forecast_has_no_band(self, built):
        blank = built[built["pred_abs_move"].isna()]
        assert blank["pred_abs_move_p10"].isna().all()
        assert blank["pred_abs_move_p90"].isna().all()
        assert blank["pred_abs_move_sd"].isna().all()
        assert blank["pred_abs_move_resid_n"].isna().all()

    def test_the_band_uses_only_earlier_folds(self, panel, built):
        # Reproduce one fold's band by hand from the residuals of everything
        # before it, and nothing else.
        _, trainable = training_frames(panel, MODEL)
        realized = trainable.set_index(["ticker", "date"])[MODEL.target]
        scored = built[built["pred_abs_move"].notna() & built["pred_abs_move_sd"].notna()]
        fold = scored["pred_abs_move_fold_start"].min()  # the FIRST fold that got a band

        earlier = built[
            built["pred_abs_move"].notna() & (built["pred_abs_move_fold_start"] < fold)
        ]
        keys = pd.MultiIndex.from_arrays([earlier["ticker"], earlier["event_date"]])
        truth = realized.reindex(keys).to_numpy(dtype=float)
        made = earlier["pred_abs_move"].to_numpy(dtype=float)
        ok = np.isfinite(truth) & np.isfinite(made)

        rows = scored[scored["pred_abs_move_fold_start"] == fold]
        p10, p90, sd, n = tier4.interval_for(
            rows["pred_abs_move"].to_numpy(), made[ok], (truth - made)[ok]
        )
        assert np.allclose(p10, rows["pred_abs_move_p10"].to_numpy())
        assert np.allclose(p90, rows["pred_abs_move_p90"].to_numpy())
        assert np.allclose(sd, rows["pred_abs_move_sd"].to_numpy())
        assert np.allclose(n, rows["pred_abs_move_resid_n"].to_numpy(dtype=float))


class TestIntervalMechanics:
    """`interval_for` on its own, including the conditioned branch the
    synthetic panel is too small to reach."""

    def test_a_thin_pool_produces_nothing(self):
        rng = np.random.default_rng(3)
        pred = rng.uniform(1, 10, 40)
        p10, p90, sd, n = tier4.interval_for(np.array([5.0]), pred, rng.normal(size=40))
        assert np.isnan(p10).all() and np.isnan(p90).all() and np.isnan(sd).all()
        assert np.isnan(n).all()

    def test_a_flat_pool_gives_every_row_the_same_width(self):
        rng = np.random.default_rng(4)
        pool_pred = rng.uniform(1, 10, 1000)
        pool_res = rng.normal(0, 2, 1000)
        preds = np.array([3.0, 8.0])
        p10, p90, sd, n = tier4.interval_for(preds, pool_pred, pool_res)
        assert np.allclose(p90 - p10, (p90 - p10)[0])
        assert sd[0] == pytest.approx(sd[1])
        assert n[0] == 1000

    def test_a_large_pool_conditions_the_band_on_the_prediction(self):
        # EXP-115's finding: error scales with the prediction, so one pool for
        # every row understates the band at the top of the range. With enough
        # residuals the deciles kick in and the widths must differ.
        rng = np.random.default_rng(5)
        pool_pred = rng.uniform(1, 20, 40_000)
        pool_res = rng.normal(0, 1, 40_000) * pool_pred  # error grows with size
        p10, p90, sd, _ = tier4.interval_for(np.array([2.0, 18.0]), pool_pred, pool_res)
        assert (p90 - p10)[1] > (p90 - p10)[0] * 2
        assert sd[1] > sd[0]

    def test_a_nan_prediction_gets_no_band(self):
        rng = np.random.default_rng(6)
        p10, p90, sd, n = tier4.interval_for(
            np.array([np.nan]), rng.uniform(1, 10, 1000), rng.normal(size=1000)
        )
        assert np.isnan(p10).all() and np.isnan(sd).all() and np.isnan(n).all()

    def test_a_table_from_an_older_layout_is_refused_not_null_filled(self):
        """The failure mode is silence, so the guard has to be total.

        `normalize` reindexes onto COLUMNS. A file missing a column comes back
        with that column NULL, and "no forecast for any event" is exactly what
        an unbuilt table looks like — so a stale file would read as a valid
        empty one. Checking that every declared column is PRESENT catches every
        older layout, including ones nobody enumerated.
        """
        import pandas as _pd

        frame = _pd.DataFrame({c: [] for c in COLUMNS})
        stale = frame.drop(columns=["pred_im_t1_d14_fold_start"])
        with pytest.raises(Tier4Error, match="older layout"):
            tier4._assert_current_schema(stale, Path("t.parquet"))
        # The current layout passes untouched.
        assert tier4._assert_current_schema(frame, Path("t.parquet")) is frame

    def test_a_signed_target_keeps_its_negative_band(self):
        """`floor=None` is what makes a producer with a signed target possible.

        Both producers so far predict a magnitude, so zero is the right clip
        for them and was hard-coded. An IV crush is negative at 83% of prints:
        a hard zero would collapse every one of its bands to [0, 0] — and
        [0, 0] is not inverted, so the interval check would pass while every
        band said nothing.
        """
        rng = np.random.default_rng(11)
        pool_pred = rng.uniform(-30, 5, 2000)
        pool_res = rng.normal(0, 6, 2000)
        preds = np.array([-20.0, -5.0])
        floored = tier4.interval_for(preds, pool_pred, pool_res)
        signed = tier4.interval_for(preds, pool_pred, pool_res, floor=None)

        # Every lower bound is clipped away, and the deep row loses its band
        # entirely: [0, 0] on a forecast of -20.
        assert (floored[0] == 0).all()
        assert floored[0][0] == 0 and floored[1][0] == 0
        assert (signed[1] > signed[0]).all()
        assert (signed[0] < 0).all()
        # The centre still sits inside its own band.
        assert (signed[0] <= preds).all() and (preds <= signed[1]).all()

    def test_the_bucket_min_pool_is_not_the_interval_floor(self):
        """A regression: the two were both called `floor` and one shadowed the other.

        `bucket_residuals` carries a MIN_POOL count — 250 — and conditioning
        used a local named `floor` for it. Adding a `floor` PARAMETER made the
        count clip the band, so every row came back [250, 250]: a plausible
        pair of numbers in the target's units, produced by a row count.
        """
        rng = np.random.default_rng(12)
        pool_pred = rng.uniform(1, 20, 40_000)
        pool_res = rng.normal(0, 1, 40_000) * pool_pred
        p10, p90, _, n = tier4.interval_for(np.array([2.0, 18.0]), pool_pred, pool_res)
        assert not (p10 == tier4.MIN_RESIDUALS).any()
        assert not (p90 == tier4.MIN_RESIDUALS).any()
        assert (p90 > p10).all()

    def test_each_producer_declares_the_floor_its_target_needs(self):
        """A magnitude floors at zero; a signed target must not.

        `pred_iv_crush_30` is negative at 83% of prints. Floored, every one of
        its bands would be [0, 0] — which no check would flag, because [0, 0]
        is not inverted.
        """
        expected = {
            "pred_abs_move": 0.0,
            "pred_im_t1_d14": 0.0,
            "pred_runup_abs_move_d14": 0.0,
            "pred_iv_crush_30": None,
        }
        for produces in tier4.PRODUCES:
            model = tier4.feature_model(produces)
            assert model.interval_floor == expected[produces], (
                f"{produces} declares floor {model.interval_floor!r}"
            )

    def test_the_floor_clips_a_band_that_runs_below_zero(self):
        rng = np.random.default_rng(7)
        pool_pred = rng.uniform(1, 10, 1000)
        pool_res = rng.normal(-5, 1, 1000)  # residuals push the band well negative
        p10, _, _, _ = tier4.interval_for(np.array([0.5]), pool_pred, pool_res)
        assert p10[0] == 0.0


class TestTheServingBand:
    """A live band and a stored one come from the same pool, or they are two
    answers to one question."""

    @pytest.fixture
    def wired(self, panel, built, tmp_path, monkeypatch):
        from engine import paths

        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path, index=False)
        monkeypatch.setattr(paths, "PANEL", panel_path)
        monkeypatch.setattr(paths, "TIER4", tmp_path / "tier4_forecasts.parquet")
        write_forecasts(built)
        return built

    def test_the_served_band_reproduces_the_stored_one(self, panel, wired):
        stored = wired
        fold = stored.loc[stored["pred_abs_move_sd"].notna(), "pred_abs_move_fold_start"].max()
        served = tier4.serving_model(fold, panel=panel, model=MODEL, cache=False)
        rows = stored[stored["pred_abs_move_fold_start"] == fold]

        p10, p90, sd, n = served.interval(rows["pred_abs_move"].to_numpy(dtype=float))
        assert np.allclose(p10, rows["pred_abs_move_p10"].to_numpy(dtype=float))
        assert np.allclose(p90, rows["pred_abs_move_p90"].to_numpy(dtype=float))
        assert np.allclose(sd, rows["pred_abs_move_sd"].to_numpy(dtype=float))
        assert np.allclose(n, rows["pred_abs_move_resid_n"].to_numpy(dtype=float))

    def test_the_pool_stops_at_the_fold(self, panel, wired):
        # The served pool must not contain the fold's own errors, or a live
        # band would be narrower than the one the backtest recorded for exactly
        # the reason that makes it wrong.
        stored = wired
        fold = stored.loc[stored["pred_abs_move_sd"].notna(), "pred_abs_move_fold_start"].max()
        served = tier4.serving_model(fold, panel=panel, model=MODEL, cache=False)
        expected = int(
            (stored["pred_abs_move"].notna() & (stored["pred_abs_move_fold_start"] < fold)).sum()
        )
        assert served.pool_pred.size == expected
        assert served.pool_res.size == expected

    def test_no_table_means_no_band_rather_than_a_made_up_one(
        self, panel, tmp_path, monkeypatch
    ):
        from engine import paths

        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path, index=False)
        monkeypatch.setattr(paths, "PANEL", panel_path)
        monkeypatch.setattr(paths, "TIER4", tmp_path / "absent.parquet")

        served = tier4.serving_model(
            pd.Timestamp("2015-06-01"), panel=panel, model=MODEL, cache=False
        )
        assert served.pool_pred.size == 0
        _, _, sd, _ = served.interval([5.0])
        assert np.isnan(sd).all()
        # ...and the point forecast still works. A missing band must not take
        # the forecast down with it.
        assert np.isfinite(served.predict(_prepare(panel).head(1))).all()


class TestTheServingCacheCarriesItsPool:
    """2026-09-15 memory fix: a serving-cache HIT must not re-derive the pool.

    `_pool_before` was previously called unconditionally on every
    `serving_model()` invocation, cache hit or miss -- `model.prepare(panel)`
    re-reads `daily_market` and rebuilds the whole trainable frame every time
    (~2-2.4 GiB measured on a real shadow-nightly attempt). Since the pool is
    a pure function of (fold, model, panel) -- already the exact key a cache
    hit verifies -- a cache file that also stores the pool can serve it
    verbatim.
    """

    @pytest.fixture
    def wired(self, panel, built, tmp_path, monkeypatch):
        from engine import paths

        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path, index=False)
        monkeypatch.setattr(paths, "PANEL", panel_path)
        monkeypatch.setattr(paths, "TIER4", tmp_path / "tier4_forecasts.parquet")
        monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path / "serving")
        write_forecasts(built)
        return built

    def _fold(self, stored):
        return stored.loc[stored["pred_abs_move_sd"].notna(), "pred_abs_move_fold_start"].max()

    def test_a_cache_hit_never_calls_pool_before(self, panel, wired, monkeypatch):
        fold = self._fold(wired)
        reference = tier4._pool_before(fold, MODEL, panel)  # ground truth, pre-fix path

        # Cold: writes the cache file, embedding the pool this fix adds.
        first = tier4.serving_model(fold, panel=panel, model=MODEL, cache=True)
        assert np.array_equal(first.pool_pred, reference[0])
        assert np.array_equal(first.pool_res, reference[1])

        def _forbidden(*_a, **_k):
            raise AssertionError("_pool_before must not run on a cache hit")

        monkeypatch.setattr(tier4, "_pool_before", _forbidden)

        # Warm: a second process/call would hit the same cache file. Must
        # return the identical pool without calling the (now forbidden)
        # _pool_before.
        second = tier4.serving_model(fold, panel=panel, model=MODEL, cache=True)
        assert np.array_equal(second.pool_pred, reference[0])
        assert np.array_equal(second.pool_res, reference[1])

    def test_an_old_format_cache_file_without_pool_arrays_still_works(
        self, panel, wired, monkeypatch
    ):
        import joblib

        fold = self._fold(wired)
        reference = tier4._pool_before(fold, MODEL, panel)

        served = tier4.serving_model(fold, panel=panel, model=MODEL, cache=True)
        path = tier4._serving_path(MODEL.model_id, fold, served.tier3_snapshot)
        stored = joblib.load(path)
        # Simulate a cache file written before this fix: no pool arrays.
        stored.pop("pool_pred", None)
        stored.pop("pool_res", None)
        joblib.dump(stored, path)

        again = tier4.serving_model(fold, panel=panel, model=MODEL, cache=True)
        assert np.array_equal(again.pool_pred, reference[0])
        assert np.array_equal(again.pool_res, reference[1])


class TestAFreshFitBuildsTheTrainableFrameOnce:
    """2026-09-18 memory fix: a FRESH fit must not rebuild `trainable` twice.

    `TestTheServingCacheCarriesItsPool` above fixed the cache-HIT path: a
    served fold no longer re-derives its pool via `_pool_before`, which used
    to call `training_frames(panel, model)` (`model.prepare(panel)` re-reads
    `daily_market` and rebuilds the whole trainable frame, ~2-2.4 GiB
    measured) on every call regardless of hit or miss. The FRESH-FIT branch
    (a cache MISS -- the exact path a boundary/pinned/coarse rescore touching
    a fold never served before takes) was left calling `training_frames`
    directly AND THEN calling `_pool_before(fold, model, panel)`, which
    called `training_frames` AGAIN internally: `model.prepare(panel)` itself
    is deduplicated by `training_frames`'s own `_PREPARED` cache (keyed on
    `model.model_id` and `panel`'s identity), but each call still returns a
    FRESH `.copy()` of the cached frames regardless -- so a fresh fit paid
    for two copies of the same trainable frame, back to back, one of them
    for a value identical to the one already sitting in a local variable.

    Real-world trigger measured 2026-09-18: a standalone boundary-pass
    diagnostic (skipping the forward pass, so every fold it touches is a
    genuine first-ever fresh fit) showed transient ~1.2 GB spikes, released
    before the next request, at STR-THRU/STR-RUNUP boundary requests --
    never at CAL-P ones, which do not reach forecast sizing. `load_chain_index`
    was ruled out directly (10 calls, 0.4-1.4s each, all resolved) before
    this path was read.
    """

    @pytest.fixture
    def wired(self, panel, built, tmp_path, monkeypatch):
        from engine import paths

        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path, index=False)
        monkeypatch.setattr(paths, "PANEL", panel_path)
        monkeypatch.setattr(paths, "TIER4", tmp_path / "tier4_forecasts.parquet")
        monkeypatch.setattr(tier4, "SERVING_DIR", tmp_path / "serving")
        write_forecasts(built)
        return built

    def test_a_fresh_fit_calls_training_frames_only_once(self, panel, wired, monkeypatch):
        stored = wired
        fold = stored.loc[stored["pred_abs_move_sd"].notna(), "pred_abs_move_fold_start"].max()
        # `wired` points SERVING_DIR at a fresh, empty tmp_path directory, so
        # every fold -- this one included -- is necessarily a first-ever,
        # never-cached fresh fit: `serving_model` cannot take the cache-hit
        # branch, which is what makes this test exercise the fresh-fit path
        # at all.
        calls = {"n": 0}
        original = tier4.training_frames

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(tier4, "training_frames", counting)
        tier4.serving_model(fold, panel=panel, model=MODEL, cache=True)
        assert calls["n"] == 1, (
            f"expected exactly one training_frames() call on a fresh fit, got {calls['n']}"
        )

    def test_the_reused_trainable_produces_the_identical_pool(self, panel, wired, monkeypatch):
        """Passing the already-built frame through must not change the
        answer -- `training_frames` is a pure function of (panel, model), so
        the caller's copy and a freshly rebuilt one are byte-for-byte the
        same input to `_pool_before`'s own computation.
        """
        fold = wired.loc[wired["pred_abs_move_sd"].notna(), "pred_abs_move_fold_start"].max()
        _, trainable = tier4.training_frames(panel, MODEL)

        reused = tier4._pool_before(fold, MODEL, panel, trainable=trainable)
        rebuilt = tier4._pool_before(fold, MODEL, panel)  # trainable=None -> old path

        assert np.array_equal(reused[0], rebuilt[0])
        assert np.array_equal(reused[1], rebuilt[1])

    def test_serving_model_output_is_unchanged_by_the_reuse(self, panel, wired, monkeypatch):
        """End to end: the fix touches only HOW the pool is computed, never
        the fold's fitted estimator, its pool arrays, or anything else on
        the returned `ServingModel`.
        """
        fold = wired.loc[wired["pred_abs_move_sd"].notna(), "pred_abs_move_fold_start"].max()
        reference_pool = tier4._pool_before(fold, MODEL, panel)

        served = tier4.serving_model(fold, panel=panel, model=MODEL, cache=True)
        assert np.array_equal(served.pool_pred, reference_pool[0])
        assert np.array_equal(served.pool_res, reference_pool[1])


class TestThePnLGateSimulator:
    """`engine.pnl_sim` — the machinery the TWIN-P5 gate turns on."""

    def test_a_move_draw_past_minus_100_percent_cannot_produce_a_negative_spot(self):
        """The bug that silently poisoned means before it was caught.

        The move residual pool is heavy-tailed. Untruncated, a down draw beyond
        -100% put spot below zero, `log(S/K)` returned NaN, and the mean of
        every affected event was quietly destroyed — no error, no flag.
        """
        from engine import pnl_sim

        floor = pnl_sim.MIN_SPOT_FRACTION
        assert 0 < floor < 0.01
        value = float(pnl_sim.black_scholes_put(100 * floor, 100.0, 0.03, 0.4))
        assert np.isfinite(value)
        assert value == pytest.approx(100.0, abs=0.05), "a stock at zero makes the put worth its strike"

    def test_the_pricer_is_intrinsic_at_expiry(self):
        from engine import pnl_sim

        assert float(pnl_sim.black_scholes_put(90, 100, 0.0, 0.4)) == pytest.approx(10.0)
        assert float(pnl_sim.black_scholes_put(110, 100, 0.0, 0.4)) == pytest.approx(0.0)

    def test_the_cutoff_window_ends_strictly_before_the_month_it_gates(self):
        """An event must never be ranked against itself or anything later."""
        from engine import pnl_sim

        history = pd.DataFrame({
            "event_date": pd.date_range("2024-01-01", periods=400, freq="D"),
            "exp_pnl_sim": np.linspace(-0.5, 0.5, 400),
        })
        bar = pnl_sim.trailing_cutoff(history, "2024-07-15", window_months=6, quantile=0.20)
        assert bar is not None
        inside = history[(history.event_date >= "2024-01-01") & (history.event_date < "2024-07-01")]
        assert bar == pytest.approx(float(np.quantile(inside.exp_pnl_sim, 0.80)))

    def test_a_thin_window_yields_no_bar_rather_than_a_fabricated_one(self):
        from engine import pnl_sim

        history = pd.DataFrame({
            "event_date": pd.to_datetime(["2024-06-01"] * 20),
            "exp_pnl_sim": np.linspace(0, 1, 20),
        })
        assert pnl_sim.trailing_cutoff(history, "2024-07-15") is None

    def test_the_registered_window_and_quantile_are_what_exp131_confirmed(self):
        from engine import pnl_sim

        assert pnl_sim.WINDOW_MONTHS == 6
        assert pnl_sim.QUANTILE == 0.20


class TestServingIsProcessDeterministic:
    """2026-09-15 investigation: v2's legacy_render/legacy_selfcheck build a
    FRESH ``Scorer``/``tier4.serving_model`` in separate worker processes,
    unlike legacy nightly.py's single shared ``Scorer``. A real reproduction
    (attempt-14 inputs) found 8-10 digest mismatches between a rendered
    board and a fresh selfcheck re-score.

    Root-caused: NOT process nondeterminism in ``tier4``/``Scorer``. The
    board's frozen ``score.json`` was a real production artifact computed
    under a different code version (``implementation_ref`` a content hash of
    the worker source tree) than the reproduction worktree's HEAD --
    confirmed by comparing ``content_hash(worker_source_manifest(...))``
    directly against the real job's recorded ``implementation_ref``: they
    differ. A controlled, single-code-version reproduction --
    ``legacy_score`` -> ``legacy_render`` -> ``legacy_selfcheck`` run as
    three SEPARATE PROCESSES against one pinned legacy read-set -- came back
    clean: 20/88 rows re-scored, 0 mismatches; a second independent full
    ``score_calendar`` (88 rows, every producer/fold the board touches) also
    reproduced every row's digest bit-for-bit against the first. A synthetic
    MLPRegressor determinism check (the champion size model's real
    architecture) also found predictions bit-identical whether BLAS was
    pinned to 1 thread or 8.

    This test is therefore a REGRESSION GUARD, not a fix demonstration: it
    already passes on the code that produced these findings. It exists so a
    future change to ``serving_model``/``fit_fold``/``_pool_before`` that
    reintroduces process-order sensitivity (unsorted iteration, a hash-order
    dependent join, etc.) is caught here instead of by a 3am selfcheck run.
    """

    @pytest.fixture
    def wired_dir(self, panel, built, tmp_path):
        panel_path = tmp_path / "panel.parquet"
        panel.to_parquet(panel_path, index=False)
        tier4_path = tmp_path / "tier4_forecasts.parquet"
        # Explicit `path=` so this writes ONLY the tmp-path file, never the
        # real `paths.TIER4` — this fixture is not monkeypatching the main
        # test process's globals (the subprocess script below sets its OWN
        # `paths.TIER4`/`tier4.SERVING_DIR` before ever reading them).
        write_forecasts(built, path=tier4_path)
        serving_dir = tmp_path / "serving"
        serving_dir.mkdir()
        return {"panel": panel_path, "tier4": tier4_path, "serving": serving_dir}

    def _fold(self, built):
        return built.loc[built["pred_abs_move_sd"].notna(), "pred_abs_move_fold_start"].max()

    def _run_subprocess(self, wired_dir, script_path, panel_path, serving_dir, out_path, fold):
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, str(script_path), str(panel_path), str(serving_dir), str(out_path),
             fold.isoformat()],
            check=True, capture_output=True, text=True, timeout=120,
            cwd=str(Path(__file__).resolve().parents[1]),
        )

    def _write_probe_script(self, wired_dir):
        import textwrap

        script = textwrap.dedent(f"""
            import sys, json, hashlib
            sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})
            sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
            import numpy as np
            import pandas as pd
            from engine import paths
            from engine.data.features import tier4
            from test_tier4 import MODEL

            from pathlib import Path as _Path
            panel_path, serving_dir, out_path, fold_iso = sys.argv[1:5]
            paths.PANEL = _Path(panel_path)
            paths.TIER4 = _Path({str(wired_dir["tier4"])!r})
            tier4.SERVING_DIR = _Path(serving_dir)

            panel = pd.read_parquet(panel_path)
            served = tier4.serving_model(fold_iso, panel=panel, model=MODEL, cache=True)
            # Select the SAME five (ticker, date) rows regardless of the
            # panel's own on-disk order -- a positional .head(5) would pick
            # different underlying events once the panel is shuffled, which
            # is a test bug, not evidence of a row-order dependency.
            probe_rows = panel[panel["ticker"].isin(["T000", "T001"]) & (panel["k"] < 3)]
            probe_rows = probe_rows.sort_values(["ticker", "date"]).reset_index(drop=True)
            preds = np.asarray(served.predict(probe_rows), dtype=float)
            payload = {{
                "pred_rounded": [None if not np.isfinite(v) else round(float(v), 6) for v in preds],
                "pred_hash": hashlib.sha256(preds.tobytes()).hexdigest(),
                "pool_pred_hash": hashlib.sha256(np.asarray(served.pool_pred, dtype=float).tobytes()).hexdigest(),
                "pool_res_hash": hashlib.sha256(np.asarray(served.pool_res, dtype=float).tobytes()).hexdigest(),
                "pool_n": int(served.pool_pred.size),
            }}
            with open(out_path, "w") as fh:
                json.dump(payload, fh)
        """)
        script_path = wired_dir["serving"].parent / "probe.py"
        script_path.write_text(script)
        return script_path

    def test_two_subprocesses_with_no_pre_existing_cache_fit_bit_identically(
        self, panel, built, wired_dir
    ):
        """Two SEPARATE python processes, each given its OWN empty cache
        directory (so BOTH independently reach a cache MISS -> ``fit_fold``
        -> write, the exact scenario legacy_render and legacy_selfcheck are
        in when a fold's model was not already pinned), on the SAME panel
        bytes. This is the direct analogue of legacy nightly.py's single
        shared ``Scorer`` (one fit, reused) versus v2's per-process fresh
        ``Scorer`` (independent fits) -- proven here to agree bit-for-bit.
        """
        import json as _json

        fold = self._fold(built)
        script_path = self._write_probe_script(wired_dir)

        serving_a = wired_dir["serving"].parent / "serving_a"
        serving_b = wired_dir["serving"].parent / "serving_b"
        serving_a.mkdir()
        serving_b.mkdir()
        out_a = wired_dir["serving"].parent / "out_a.json"
        out_b = wired_dir["serving"].parent / "out_b.json"

        self._run_subprocess(wired_dir, script_path, wired_dir["panel"], serving_a, out_a, fold)
        self._run_subprocess(wired_dir, script_path, wired_dir["panel"], serving_b, out_b, fold)

        result_a = _json.loads(out_a.read_text())
        result_b = _json.loads(out_b.read_text())
        assert result_a == result_b, (
            "serving_model()'s independently-fit estimator/pool diverged "
            "between two separate processes on byte-identical inputs -- "
            "the render-vs-selfcheck symptom"
        )

    def test_serving_is_insensitive_to_the_panel_rows_own_order(self, panel, built, wired_dir):
        """A determinism test that feeds two processes the SAME frame in the
        SAME order proves nothing about order (AGENTS.md's self-check
        section: 'shuffle the input'). Compared at the board's own six-place
        precision (``_norm`` in ``engine/dashboard/selfcheck.py``), not by
        exact bytes -- a linear fit's normal equations are allowed a few ULP
        of summation-order noise; a real order BUG moves the sixth decimal,
        not the sixteenth.
        """
        import json as _json

        fold = self._fold(built)
        script_path = self._write_probe_script(wired_dir)

        shuffled = panel.sample(frac=1.0, random_state=7).reset_index(drop=True)
        shuffled_path = wired_dir["panel"].with_name("panel_shuffled.parquet")
        shuffled.to_parquet(shuffled_path, index=False)

        serving_a = wired_dir["serving"].parent / "serving_order_a"
        serving_b = wired_dir["serving"].parent / "serving_order_b"
        serving_a.mkdir()
        serving_b.mkdir()
        out_a = wired_dir["serving"].parent / "out_order_a.json"
        out_b = wired_dir["serving"].parent / "out_order_b.json"

        self._run_subprocess(wired_dir, script_path, wired_dir["panel"], serving_a, out_a, fold)
        self._run_subprocess(wired_dir, script_path, shuffled_path, serving_b, out_b, fold)

        result_a = _json.loads(out_a.read_text())
        result_b = _json.loads(out_b.read_text())
        assert result_a["pred_rounded"] == result_b["pred_rounded"], (
            "serving_model()'s prediction moved past six decimal places "
            "when the panel's own row order changed -- a row-order "
            "dependency the board's own digest precision would catch"
        )


# --------------------------------------------------------------------------
# ServingModel.artifact_ref -- what a Phase 4 strict trace binds to
#
# Fix context: `_size_from_forecast` (engine/score.py) captures the served
# fold model's FEATURES into a Phase 4 trace but, before this fix, never
# recorded a `model_bindings` entry for it -- unlike every other scored
# role. `capture_source_bundle` only derives `native_recipes["forecast"]`
# from a captured binding (see test_phase4_trace_collector.py), so a
# FORECAST_SIZED strategy (TWIN-P and friends) hit
# "source_inputs.native_recipes missing ['forecast']" in strict assembly
# even though the served model's artifact genuinely exists on disk --
# `serving_model()` always persists it before returning. `artifact_ref()`
# is the accessor the fix now calls to get a real path + sha256 instead of
# leaving the binding out.
# --------------------------------------------------------------------------


def test_serving_model_artifact_ref_points_at_the_persisted_cache_file(tmp_path, monkeypatch):
    from engine.data.features import tier4 as tier4_module

    monkeypatch.setattr(tier4_module, "SERVING_DIR", tmp_path)
    fold = pd.Timestamp("2026-01-01")
    served = ServingModel(
        estimator=object(),
        model_id="size_v1_4",
        fold_start=fold,
        tier3_snapshot="deadbeef" * 8,
        features=("has_implied_quote", "mcap_log"),
    )
    expected_path = tier4_module._serving_path(
        served.model_id, served.fold_start, served.tier3_snapshot
    )
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path.write_bytes(b"not a real joblib file, just needs bytes to hash")

    path, digest = served.artifact_ref()

    assert path == expected_path
    import hashlib

    assert digest == hashlib.sha256(expected_path.read_bytes()).hexdigest()


def test_serving_model_artifact_ref_is_missing_when_never_served(tmp_path, monkeypatch):
    """No cache file, no answer -- `artifact_ref` must not fabricate a hash."""
    from engine.data.features import tier4 as tier4_module

    monkeypatch.setattr(tier4_module, "SERVING_DIR", tmp_path)
    served = ServingModel(
        estimator=object(),
        model_id="size_v1_4",
        fold_start=pd.Timestamp("2026-01-01"),
        tier3_snapshot="deadbeef" * 8,
        features=("has_implied_quote",),
    )

    with pytest.raises(FileNotFoundError):
        served.artifact_ref()
