"""Real-data dataset builders for the Phase 5 frozen states, on synthetic data.

Two claims per builder (``tools/phase5_datasets.py``):

1. its rows are exactly the rows legacy selects -- proved against a legacy
   ``Scorer`` built over the same synthetic world (``Scorer.trades`` for the
   payoff fits, ``recalibrate.load_pairs`` for the maps,
   ``Scorer._residual_pool`` for the paired pool, ``ModelArtifact`` pools for
   the driver pools);
2. ``tools/phase5_training_job.py`` then writes artifacts byte-identical to
   calling the P5-4 builder directly on those legacy rows.

Everything lives under ``tmp_path``; the legacy store readers are
monkeypatched onto an in-memory synthetic store.
"""
from __future__ import annotations

import json
import types

import numpy as np
import pandas as pd
import pytest

from engine.calendar import TradingCalendar
from engine.features import FeatureContext
from engine.models.no_fit import RuntimeFitForbidden, no_fit_guard
from engine.models.registry import ModelArtifact, Registry, RegistryEntry, bucket_residuals
from engine.v2.models.frozen_state import serialize_frozen_state
from engine.v2.models.no_fit import RuntimeFitForbidden as V2Forbidden
from engine.v2.models.no_fit import no_fit_guard as v2_no_fit_guard
from engine.v2.models.payoff_artifact import serialize_payoff_artifact
from engine.v2.models.recalibration_artifact import serialize_recalibration_artifact
from tools import phase5_datasets as data
from tools import phase5_training_job as job

TICKERS = [f"T{i}" for i in range(20)]
EVENTS = pd.date_range("2018-01-15", periods=24, freq="61D")
CUTOFF = "2021-06-01"


# --------------------------------------------------------------------------
# the synthetic world
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(3)
    rows = []
    for ticker in TICKERS:
        for k, date in enumerate(EVENTS):
            rows.append({"ticker": ticker, "date": date, "year": date.year, "n_prior": k + 4,
                         "abs_move": float(rng.uniform(0.5, 12.0)),
                         "or_implied": float(rng.uniform(2.0, 10.0)),
                         "mean_prior_or_implied": 6.0, "mcap_usd": 5e9})
    frame = pd.DataFrame(rows)
    frame.loc[3, "abs_move"] = np.nan          # a driver the finite guard drops
    return frame


@pytest.fixture(scope="module")
def daily():
    dates = pd.bdate_range("2017-12-01", "2022-06-30")
    rng = np.random.default_rng(4)
    frames = []
    for ticker in TICKERS[:-1]:                # the last ticker has no daily rows at all
        n = len(dates)
        frames.append(pd.DataFrame({
            "ticker": ticker, "date": dates, "src_iv": "orats", "implied_move": 6.0,
            "iv10": rng.uniform(20, 60, n), "iv30": rng.uniform(20, 60, n),
            "exern_iv10": 30.0, "exern_iv30": 28.0, "iee": 1.2, "skew": 0.3,
            "contango": 0.1, "fwd90_30": 32.0, "fexern90_30": 30.0, "rvol30": 38.0,
            "spot": 100.0, "mcap_log": np.log(5e9)}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def trades(panel):
    """Tier-2-shaped trades with legs, both strategies, two alphas, junk rows."""
    rng = np.random.default_rng(5)
    rows = []
    for i, event in enumerate(panel.itertuples(index=False)):
        for strategy in ("STR-THRU", "STR-RUNUP", "CAL-P"):
            for alpha in (0.5, 0.25):
                spot = float(rng.uniform(20, 200))
                legs = {"spot_entry": spot, "dte_entry": 2,
                        "spot_exit": spot * float(rng.uniform(0.8, 1.2)), "entry": [], "exit": []}
                exit_days = int(rng.integers(1, 20))
                rows.append({
                    "trade_id": f"{event.ticker}:{event.date.date()}:{strategy}:{alpha}",
                    "kind": "sim", "strategy": strategy, "variant": "v",
                    "ticker": event.ticker, "event_id": f"{event.ticker}_{event.date.date()}",
                    "event_date": event.date, "year": event.date.year,
                    "legs": json.dumps(legs), "entry_date": event.date,
                    "exit_date": event.date + pd.Timedelta(days=exit_days),
                    "strike": spot * float(rng.uniform(0.9, 1.1)),
                    "expiry": event.date + pd.Timedelta(days=exit_days),
                    "fill_alpha": alpha, "entry_cost": 3.0,
                    "exit_value": float(rng.uniform(0, 30)), "ret": 0.1,
                    "provenance": "engine.replay" if i % 17 else "paper.log",
                })
    frame = pd.DataFrame(rows)
    frame.loc[7, "legs"] = "not json"          # no spot: the finite guard drops it
    frame.loc[11, "legs"] = json.dumps({"spot_entry": 0.0, "spot_exit": 1.0})  # spot > 0 guard
    return frame


@pytest.fixture(scope="module")
def calendar():
    return TradingCalendar(pd.bdate_range("2017-01-01", "2023-12-31"))


@pytest.fixture(scope="module")
def legacy_trades(trades, panel, daily, calendar):
    """``Scorer.trades`` exactly as the legacy scorer builds it."""
    from engine.score import Scorer

    context = FeatureContext(panel=panel, daily=daily, calendar=calendar)
    scorer = Scorer(registry=Registry(entries=[]), trades=trades, context=context,
                    snapshot="snap", analog_daily=daily)
    return scorer.trades


def _partitions(trades):
    """Two contiguous 'year' partitions, as ``store.iter_table`` yields them."""
    half = len(trades) // 2
    return [(2019, trades.iloc[:half].reset_index(drop=True)),
            (2020, trades.iloc[half:].reset_index(drop=True))]


# --------------------------------------------------------------------------
# payoff: rows == Scorer.trades selection; job == direct builder
# --------------------------------------------------------------------------


def _legacy_payoff_rows(scorer_trades, strategy, driver, alpha, before, surface=False):
    """``fit_payoff``/``fit_runup_payoff``'s kept rows, in order."""
    rows = scorer_trades[(scorer_trades["strategy"] == strategy)
                         & np.isclose(scorer_trades["fill_alpha"].astype(float), alpha)]
    rows = rows[pd.to_datetime(rows["exit_date"]) < pd.Timestamp(before)]
    cols = [driver, "spot_entry", "exit_value"] + (["spot_exit", "strike"] if surface else [])
    values = rows[cols].apply(pd.to_numeric, errors="coerce")
    ok = np.isfinite(values.to_numpy(float)).all(axis=1) & (values["spot_entry"] > 0).to_numpy()
    if surface:
        ok &= (values["spot_exit"] > 0).to_numpy() & (values["strike"] > 0).to_numpy()
    return rows[ok]


def _recipe(label):
    from engine.v2.models.training import current_recipes

    return current_recipes()[job._key(label)]


@pytest.mark.parametrize("label,driver,surface", [
    ("payoff_line:STR-THRU:calibration", "abs_move", False),
    ("payoff_line:STR-RUNUP:calibration", "im_t1", False),
    ("payoff_surface:STR-RUNUP:calibration", "im_t1", True),
])
def test_payoff_dataset_members_are_the_rows_legacy_fits(label, driver, surface, trades, panel,
                                                        legacy_trades):
    from engine.v2.models.training import RowFilter, plan_folds, prepare_dataset

    dataset = data.payoff_trades(_partitions(trades), panel=panel)
    recipe = _recipe(label)
    strategy = recipe.key.strategy
    for alpha in (0.5, 0.25):
        prepared = prepare_dataset(recipe, dataset,
                                   extra_filters=(RowFilter("fill_alpha", "isclose", alpha),))
        (fold,) = plan_folds(recipe, prepared, cutoff=CUTOFF)
        mine = prepared.frame.iloc[fold.train]
        legacy = _legacy_payoff_rows(legacy_trades, strategy, driver, alpha, CUTOFF, surface)
        assert len(mine) == len(legacy) > 200
        for col in ("event_id", driver, "spot_entry", "exit_value", "spot_exit", "strike"):
            np.testing.assert_array_equal(mine[col].to_numpy(), legacy[col].to_numpy(), col)


def test_payoff_dataset_keeps_legacy_order_and_drops_non_replay_rows(trades, panel, legacy_trades):
    dataset = data.payoff_trades(_partitions(trades), panel=panel)
    legacy = legacy_trades[legacy_trades["strategy"].isin(data.PAYOFF_STRATEGIES)]
    assert list(dataset["event_id"]) == list(legacy["event_id"])
    assert list(dataset["fill_alpha"]) == list(legacy["fill_alpha"])
    assert (dataset["provenance"].astype(str) == "engine.replay").all()


def _run_payoff_job(monkeypatch, tmp_path, trades, panel, label, *extra):
    monkeypatch.setattr(data, "_trade_partitions", lambda: iter(_partitions(trades)))
    monkeypatch.setattr("engine.features.load_panel", lambda *a, **k: panel)
    out = tmp_path / label.replace(":", "_")
    code = job.main(["--recipe", label, "--alpha", "0.5", "--cutoff", CUTOFF,
                     "--cutoff", "2020-06-01", "--out", str(out), *extra])
    assert code == 0
    return out


def test_payoff_job_artifacts_equal_the_direct_builder_on_legacy_rows(monkeypatch, tmp_path,
                                                                     trades, panel, legacy_trades):
    from engine import payoff
    from engine.v2.models.training.payoff import (
        build_payoff_line_artifact,
        build_payoff_surface_artifact,
    )

    def records(strategy, driver):
        rows = legacy_trades[(legacy_trades["strategy"] == strategy)
                             & np.isclose(legacy_trades["fill_alpha"], 0.5)]
        return [{"driver": r[driver], "spot_entry": r["spot_entry"], "exit_value": r["exit_value"],
                 "spot_exit": r["spot_exit"], "strike": r["strike"],
                 "exit_date": pd.Timestamp(r["exit_date"]).strftime("%Y-%m-%d")}
                for _, r in rows.iterrows()]

    for label, strategy, driver in (("payoff_line:STR-THRU:calibration", "STR-THRU", "abs_move"),
                                    ("payoff_line:STR-RUNUP:calibration", "STR-RUNUP", "im_t1"),
                                    ("payoff_surface:STR-RUNUP:calibration", "STR-RUNUP", "im_t1")):
        out = _run_payoff_job(monkeypatch, tmp_path, trades, panel, label)
        for cutoff in (CUTOFF, "2020-06-01"):
            if label.startswith("payoff_line"):
                expected = build_payoff_line_artifact(records(strategy, driver), strategy=strategy,
                                                      driver=driver, alpha=0.5, before=cutoff)
                legacy = payoff.fit_payoff(legacy_trades, strategy, alpha=0.5, before=cutoff)
                assert (expected.n, expected.slope, expected.intercept) == (
                    legacy.n, legacy.slope, legacy.intercept)
            else:
                expected = build_payoff_surface_artifact(records(strategy, driver), alpha=0.5,
                                                         before=cutoff)
                assert expected.n == payoff.fit_runup_payoff(legacy_trades, alpha=0.5,
                                                             before=cutoff).n
            written = (out / f"folds/cut-{cutoff}/payoff_artifact.json").read_bytes()
            assert written == serialize_payoff_artifact(expected)


def test_payoff_plan_only_writes_receipts_and_no_artifact(monkeypatch, tmp_path, trades, panel):
    out = _run_payoff_job(monkeypatch, tmp_path, trades, panel,
                          "payoff_line:STR-THRU:calibration", "--plan-only")
    fold = out / f"folds/cut-{CUTOFF}"
    assert (fold / "membership_receipt.json").is_file()
    assert not (fold / "payoff_artifact.json").exists()


def test_calibration_recipe_needs_alpha_and_cutoff_and_out_outside_data(tmp_path):
    from engine import paths

    with pytest.raises(SystemExit):
        job.main(["--recipe", "payoff_line:STR-THRU:calibration", "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit):
        job.main(["--recipe", "payoff_line:STR-THRU:calibration", "--alpha", "0.5",
                  "--cutoff", CUTOFF, "--out", str(paths.DATA / "p5")])
    assert not (paths.DATA / "p5").exists()


# --------------------------------------------------------------------------
# recalibration: the pairs table legacy reads
# --------------------------------------------------------------------------


def _pairs(n=900, seed=8):
    rng = np.random.default_rng(seed)
    exits = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    frame = pd.DataFrame({
        "strategy": np.where(rng.random(n) < 0.6, "STR-THRU", "STR-RUNUP"),
        "fill_alpha": 0.5, "event_id": [f"E{i}" for i in range(n)],
        "ticker": "T", "event_date": exits - pd.Timedelta(days=3), "exit_date": exits,
        "raw_win": rng.uniform(0, 1, n), "outcome": (rng.random(n) < 0.5).astype(float)})
    frame.loc[:3, "raw_win"] = np.nan
    return frame


def test_recalibration_job_fits_legacy_pairs_and_equals_the_builder(tmp_path):
    from engine import recalibrate
    from engine.v2.models.training.recalibration import build_recalibration_map_artifact

    path = tmp_path / "pairs.parquet"
    _pairs().to_parquet(path, index=False)
    for strategy in ("STR-THRU", "STR-RUNUP"):
        out = tmp_path / strategy
        assert job.main(["--recipe", f"recalibration_map:{strategy}:calibration", "--alpha", "0.5",
                         "--cutoff", CUTOFF, "--pairs", str(path), "--out", str(out)]) == 0
        legacy_pairs = recalibrate.load_pairs(path)
        expected = build_recalibration_map_artifact(legacy_pairs, strategy=strategy, alpha=0.5,
                                                    before=CUTOFF)
        written = (out / f"folds/cut-{CUTOFF}/recalibration_artifact.json").read_bytes()
        assert written == serialize_recalibration_artifact(expected)
        legacy = recalibrate.fit_recalibration(strategy, 0.5, before=CUTOFF, pairs=legacy_pairs)
        assert expected.fitted and expected.n == legacy.n
        assert np.array_equal(expected.x_thresholds, legacy.x_thresholds)
        assert np.array_equal(expected.y_thresholds, legacy.y_thresholds)


def test_missing_pairs_table_refuses(tmp_path):
    with pytest.raises(SystemExit, match="missing or empty"):
        data.recalibration_pairs(tmp_path / "absent.parquet")


# --------------------------------------------------------------------------
# paired residual pool: Scorer._residual_pool over the FULL universe
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def events():
    rng = np.random.default_rng(9)
    frame = pd.DataFrame([{"ticker": t, "event_date": d,
                           "session": "AMC" if rng.random() < 0.5 else "BMO"}
                          for t in TICKERS for d in EVENTS])
    frame.loc[5, "session"] = None             # crush_frame drops sessionless events
    return frame


@pytest.fixture(scope="module")
def forecasts(panel):
    rng = np.random.default_rng(10)
    frame = panel[["ticker", "date"]].rename(columns={"date": "event_date"}).copy()
    frame["pred_abs_move"] = rng.uniform(1, 9, len(frame))
    frame["pred_iv_crush_30"] = rng.uniform(-40, 5, len(frame))
    frame.loc[::13, "pred_iv_crush_30"] = np.nan
    frame["pred_abs_move_model_id"] = "size_test"
    frame["pred_iv_crush_30_model_id"] = "crush_test"
    return frame


@pytest.fixture
def fake_store(monkeypatch, events, daily):
    """``store.read_table``/``iter_table`` over an in-memory Tier-2 store."""
    from engine.data import store

    tables = {"earnings_events": (events, "event_date"), "daily_market": (daily, "date")}

    def iter_table(name, *, years=None, columns=None):
        frame, date = tables[name]
        year = pd.to_datetime(frame[date]).dt.year
        for y in sorted(year.unique()):
            if years is None or y in set(years):
                part = frame[year == y].reset_index(drop=True)
                yield y, part[list(columns)] if columns else part

    def read_table(name, *, years=None, columns=None):
        return pd.concat([p for _, p in iter_table(name, years=years, columns=columns)],
                         ignore_index=True)

    monkeypatch.setattr(store, "iter_table", iter_table)
    monkeypatch.setattr(store, "read_table", read_table)
    return tables


def _legacy_pool(monkeypatch, panel, forecasts, context_daily=None):
    """``Scorer._residual_pool`` on a bare scorer whose context holds ``context_daily``."""
    from engine import score as score_mod

    monkeypatch.setattr("engine.features.load_panel", lambda *a, **k: panel)
    monkeypatch.setattr("engine.data.features.tier4.load_forecasts", lambda *a, **k: forecasts)
    scorer = score_mod.Scorer.__new__(score_mod.Scorer)
    scorer._pool = score_mod._UNSET
    scorer._crush_frame = score_mod._UNSET
    scorer.context = types.SimpleNamespace(daily=context_daily)
    pool = scorer._residual_pool()
    assert pool is not None
    return sorted(zip(pd.to_datetime(pool._dates).strftime("%Y-%m-%d"), pool._pred.tolist(),
                      pool._move.tolist(), pool._crush.tolist()))


def _artifact_rows(artifact):
    return sorted((r[0], r[2], r[3], r[4]) for r in artifact.rows)


def test_chunked_crush_table_equals_legacy_full_crush_frame(fake_store, events, daily):
    from engine.models.training import iv_crush

    legacy = iv_crush.crush_frame()
    for chunk in (1, 5, 1000):
        mine = data.crush_table(ticker_chunk=chunk)
        pd.testing.assert_frame_equal(mine.reset_index(drop=True), legacy.reset_index(drop=True))


def test_paired_pool_rows_equal_the_legacy_full_universe_pool(monkeypatch, tmp_path, fake_store,
                                                             panel, forecasts, daily):
    legacy = _legacy_pool(monkeypatch, panel, forecasts)
    inputs = data.paired_pool_inputs(forecasts=forecasts, panel=panel, ticker_chunk=4)
    artifact, summary = job.build_state("paired_residual_pool", paired_inputs=inputs)
    assert _artifact_rows(artifact) == legacy and len(legacy) > 100
    assert (artifact.move_model_id, artifact.crush_model_id) == ("size_test", "crush_test")

    # Difference from legacy, measured: a bounded scorer scopes the crush
    # table to its loaded tickers and gets a DIFFERENT pool; the builder
    # takes no context, so the same inputs always give the full pool.
    scoped = _legacy_pool(monkeypatch, panel, forecasts,
                          context_daily=daily[daily["ticker"].isin(TICKERS[:3])])
    assert set(scoped) < set(legacy)


def test_paired_pool_job_writes_the_direct_builder_bytes(monkeypatch, tmp_path, fake_store,
                                                        panel, forecasts):
    from engine.v2.models.lineage import DataDependency, Lineage
    from engine.v2.models.training.residuals import build_paired_residual_pool_artifact

    inputs = data.paired_pool_inputs(forecasts=forecasts, panel=panel, ticker_chunk=3)
    monkeypatch.setattr(data, "paired_pool_inputs", lambda **_: inputs)
    out = tmp_path / "paired"
    assert job.main(["--state", "paired_residual_pool", "--cutoff", "2021-01-01",
                     "--out", str(out), "--plan-only"]) == 0
    assert not (out / "paired_residual_pool.json").exists()
    assert job.main(["--state", "paired_residual_pool", "--cutoff", "2021-01-01",
                     "--out", str(out)]) == 0

    lineage = Lineage(data=tuple(DataDependency(table=t, end_exclusive="2021-01-01") for t in (
        "tier4.forecasts", "tier3.panel", "tier2.earnings_events", "tier2.daily_market")))
    direct = build_paired_residual_pool_artifact(
        inputs["forecasts"].to_dict("records"), inputs["outcomes"].to_dict("records"),
        inputs["crush"].to_dict("records"), move_model_id="size_test",
        crush_model_id="crush_test", cutoff="2021-01-01", lineage=lineage)
    written = (out / "paired_residual_pool.json").read_bytes()
    assert written == serialize_frozen_state(direct)
    assert all(row[0] < "2021-01-01" for row in direct.rows)
    summary = json.loads((out / "paired_residual_pool.summary.json").read_text())
    assert summary["status"] == "written" and summary["content_hash"] == direct.content_hash
    # A rerun keeps the identical file.
    assert job.main(["--state", "paired_residual_pool", "--cutoff", "2021-01-01",
                     "--out", str(out)]) == 0
    assert json.loads((out / "paired_residual_pool.summary.json").read_text())["status"] == "resumed"


# --------------------------------------------------------------------------
# driver residual pools: the champion's embedded pool, unchanged
# --------------------------------------------------------------------------


class _Const:
    def predict(self, X):
        return np.zeros(len(X))


def _registry(tmp_path, *, buckets: bool):
    rng = np.random.default_rng(12)
    pred = rng.uniform(0, 10, 3000)
    res = rng.normal(0, 1 + pred / 5, 3000)
    entries = []
    for role in ("size", "implied_t1"):
        art = ModelArtifact(model=_Const(), role=role, features=("x",), residuals=res,
                            target="y",
                            residual_buckets=bucket_residuals(pred, res) if buckets else None)
        path = tmp_path / f"{role}.joblib"
        digest = art.save(path)
        entries.append(RegistryEntry(id=f"{role}_test", role=role, strategy="*",
                                     artifact=str(path), artifact_sha256=digest, features=["x"],
                                     target="y", train_window="t", champion=True))
    return Registry(entries=entries)


@pytest.mark.parametrize("buckets", [True, False])
def test_driver_pool_serves_exactly_the_legacy_artifact_pool(tmp_path, buckets):
    from engine.v2.scoring import native_residuals

    registry = _registry(tmp_path, buckets=buckets)
    _, legacy = registry.load_champion("size", "*")
    artifact, summary = job.build_state("driver_residual_pool:size", registry=registry)
    assert artifact.key == ("size", "size_test", None)
    assert (artifact.bucket_edges is not None) is buckets
    block = {native_residuals.RESIDUAL_ARTIFACTS_FIELD: {"s": artifact},
             native_residuals.RESIDUAL_KEYS_FIELD: {"s": {"role": "size", "model_id": "size_test",
                                                          "fold": None}}}
    for prediction in (-5.0, 0.1, 2.5, 5.0, 7.7, 9.9, 50.0, float("nan")):
        pool, flag = native_residuals.driver_pool_from_artifact(block, "s", prediction)
        assert flag is None
        assert np.array_equal(pool, legacy.residual_pool(prediction)[0])


def test_driver_pool_job_writes_the_direct_builder_bytes(monkeypatch, tmp_path):
    from engine.v2.models.lineage import DataDependency, Lineage
    from engine.v2.models.training.residuals import freeze_stored_driver_residual_pool

    registry = _registry(tmp_path, buckets=True)
    monkeypatch.setattr("engine.models.registry.load_registry", lambda *a, **k: registry)
    out = tmp_path / "driver"
    assert job.main(["--state", "driver_residual_pool:implied_t1", "--out", str(out)]) == 0
    entry, legacy = registry.load_champion("implied_t1", "*")
    direct = freeze_stored_driver_residual_pool(
        legacy.residuals, legacy.residual_buckets, role="implied_t1", model_id=entry.id,
        lineage=Lineage(data=(DataDependency(table=f"model_artifact:{entry.id}",
                                             keys=(entry.artifact_sha256,)),)))
    written = (out / "driver_residual_pool__implied_t1.json").read_bytes()
    assert written == serialize_frozen_state(direct)
    with pytest.raises(SystemExit):
        job.main(["--state", "driver_residual_pool:size", "--cutoff", CUTOFF, "--out", str(out)])


# --------------------------------------------------------------------------
# no-fit guards: the builders run only in the explicit job
# --------------------------------------------------------------------------


@pytest.mark.parametrize("guard,error", [(no_fit_guard, RuntimeFitForbidden),
                                         (v2_no_fit_guard, V2Forbidden)])
def test_state_and_calibration_jobs_trip_either_no_fit_guard_before_reading(
        monkeypatch, tmp_path, guard, error):
    def boom(*_a, **_k):
        raise AssertionError("read data under a no-fit guard")

    monkeypatch.setattr(data, "payoff_trades", boom)
    monkeypatch.setattr(data, "paired_pool_inputs", boom)
    monkeypatch.setattr(data, "champion_driver_pool", boom)
    with guard():
        with pytest.raises(error):
            job.main(["--recipe", "payoff_line:STR-THRU:calibration", "--alpha", "0.5",
                      "--cutoff", CUTOFF, "--out", str(tmp_path / "a")])
        for state in job.STATES:
            with pytest.raises(error):
                job.build_state(state)
    assert not (tmp_path / "a").exists()
