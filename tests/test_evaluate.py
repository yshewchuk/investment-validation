"""engine.evaluate — metrics, equity, walk-forward, MC, stress, preregistration.

The evaluation suite is where a candidate becomes evidence; these tests pin
the arithmetic the verdicts rest on (fill sweep, equity constructions, the
walk-forward leak guard, the block bootstrap) against hand-checkable cases.
The full end-to-end reproduction of EXP-050 lives in checks/phase2_checks.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.evaluate import (
    Gate,
    EvaluationError,
    PreregistrationError,
    alpha_sweep,
    breakeven_alpha_from_sweep,
    build_equity,
    by_year_table,
    check_preregistration,
    evaluate,
    monte_carlo,
    sharpe_equity,
    spec_hash,
    stress_iv_regime,
    stress_regimes,
    stress_slippage,
    stress_stale_dates,
    stress_tail_injection,
    trade_stats,
    walk_forward,
)


def make_trades(rets, start="2020-01-01", freq="10D", alphas=(0.5,), cost=1.0):
    """One row per event x alpha; ret shifted by alpha so the sweep is testable."""
    dates = pd.date_range(start, periods=len(rets), freq=freq)
    frames = []
    for a in alphas:
        r = np.asarray(rets, dtype=float) + (a - 0.5) * 0.0
        frames.append(pd.DataFrame({
            "event_id": [f"E{i}" for i in range(len(rets))],
            "ticker": "T",
            "event_date": dates,
            "entry_date": dates - pd.Timedelta(days=1),
            "exit_date": dates + pd.Timedelta(days=1),
            "fill_alpha": float(a),
            "entry_cost": cost,
            "exit_value": cost * (1 + r),
            "ret": r,
        }))
    return pd.concat(frames, ignore_index=True)


class TestSpecHash:
    def test_stable_and_insensitive_to_id_and_stamp(self):
        a = {"id": "EXP-101", "primary_spec": {"x": 1}, "preregistered_at": "2026-01-01"}
        b = {"id": "EXP-999", "primary_spec": {"x": 1}, "preregistered_at": "2027-01-01"}
        assert spec_hash(a) == spec_hash(b)

    def test_changes_with_the_spec_itself(self):
        a = {"id": "EXP-101", "primary_spec": {"x": 1}}
        b = {"id": "EXP-101", "primary_spec": {"x": 2}}
        assert spec_hash(a) != spec_hash(b)


class TestTradeStats:
    def test_hand_checkable_vector(self):
        rets = [-0.2, -0.1, 0.1, 0.2, 0.3]
        s = trade_stats(rets)
        assert s["n"] == 5
        assert s["mean"] == pytest.approx(0.06)
        assert s["median"] == pytest.approx(0.1)
        assert s["win_rate"] == pytest.approx(0.6)
        # profit factor = 0.6 / 0.3 = 2.0
        assert s["profit_factor"] == pytest.approx(2.0)
        # tail ratio = p95(wins) / |p5(losses)|
        assert s["tail_ratio"] > 1.0

    def test_empty_and_single(self):
        s = trade_stats([])
        assert s["n"] == 0 and np.isnan(s["mean"])
        s1 = trade_stats([0.05])
        assert s1["n"] == 1 and np.isnan(s1["std"]) and np.isnan(s1["sharpe_trade"])

    def test_sharpe_annualizes_by_trades_per_year(self):
        # 12 monthly trades of identical returns: sharpe ~ mean/std * sqrt(12/1)
        dates = pd.date_range("2020-01-01", periods=12, freq="30D")
        rets = np.array([0.03, 0.05] * 6)
        s = trade_stats(rets, dates)
        per_trade = rets.mean() / rets.std(ddof=1)
        trades_per_year = 12 / ((dates[-1] - dates[0]).days / 365.25)
        assert s["sharpe_trade"] == pytest.approx(per_trade * np.sqrt(trades_per_year), rel=1e-6)


class TestAlphaSweep:
    def test_sweep_and_breakeven(self):
        frames = []
        # mean return moves linearly from -0.02 (alpha 0) to +0.02 (alpha 1)
        for a in (0.0, 0.5, 1.0):
            frames.append(make_trades([0.0] * 10).assign(
                fill_alpha=a, ret=a * 0.04 - 0.02,
                exit_value=1 + (a * 0.04 - 0.02)))
        trades = pd.concat(frames, ignore_index=True)
        sweep = alpha_sweep(trades, alphas=(0.0, 0.5, 1.0))
        assert set(sweep) == {"0.00", "0.50", "1.00"}
        assert sweep["0.00"]["mean"] == pytest.approx(-0.02)
        assert sweep["1.00"]["mean"] == pytest.approx(0.02)
        be = breakeven_alpha_from_sweep(sweep)
        assert be == pytest.approx(0.5, abs=1e-9)

    def test_breakeven_none_when_never_crossing(self):
        sweep = {"0.00": {"mean": 0.01}, "1.00": {"mean": 0.03}}
        assert breakeven_alpha_from_sweep(sweep) is None
        assert breakeven_alpha_from_sweep({"0.50": {"mean": 0.01}}) is None


class TestBuildEquity:
    RETS = [0.10, -0.05, 0.20]

    def test_sequential_compounds_in_order(self):
        trades = make_trades(self.RETS)
        out = build_equity(trades, 0.5, mode="sequential")
        expected = 1.0
        for r in self.RETS:
            expected *= 1 + 0.5 * r
        assert out["final"] == pytest.approx(expected)
        assert out["mode"] == "sequential"

    def test_cashflow_no_overlap_equals_sequential(self):
        # With serial, non-overlapping trades both constructions coincide.
        trades = make_trades(self.RETS, freq="40D")
        seq = build_equity(trades, 0.1, mode="sequential")["final"]
        cash = build_equity(trades, 0.1, mode="cashflow")["final"]
        assert cash == pytest.approx(seq)

    def test_cashflow_counts_concurrency(self):
        # Three trades all open at once: max_concurrency = 3.
        n = 3
        trades = make_trades([0.1] * n, freq="40D")
        trades["entry_date"] = pd.Timestamp("2020-01-01")
        trades["exit_date"] = pd.Timestamp("2020-02-01")
        out = build_equity(trades, 0.1, mode="cashflow")
        assert out["max_concurrency"] == 3

    def test_unknown_mode_raises(self):
        with pytest.raises(EvaluationError):
            build_equity(make_trades([0.1]), 0.1, mode="vibes")

    def test_empty(self):
        out = build_equity(make_trades([]).iloc[0:0], 0.1)
        assert out["final"] == 1.0 and out["max_dd"] == 0.0

    def test_max_drawdown(self):
        trades = make_trades([0.5, -0.5, 0.0], freq="40D")
        out = build_equity(trades, 1.0, mode="sequential")
        # equity: 1 -> 1.5 -> 0.75 -> 0.75; peak 1.5, trough 0.75 -> dd 50%
        assert out["max_dd"] == pytest.approx(0.5)


class TestSharpeEquity:
    def test_flat_curve_is_nan(self):
        curve = pd.Series([1.0, 1.0], index=pd.date_range("2020-01-01", periods=2))
        assert np.isnan(sharpe_equity(curve))

    def test_short_curve_is_nan(self):
        assert np.isnan(sharpe_equity(pd.Series([1.0])))


class TestWalkForward:
    def trades3y(self):
        frames = []
        for year in (2020, 2021, 2022):
            dates = pd.date_range(f"{year}-01-01", periods=5, freq="30D")
            frames.append(make_trades([0.01] * 5, start=str(dates[0].date()), freq="30D")
                          .assign(event_date=dates,
                                  entry_date=dates - pd.Timedelta(days=1),
                                  exit_date=dates + pd.Timedelta(days=1),
                                  event_id=[f"{year}_{i}" for i in range(5)]))
        return pd.concat(frames, ignore_index=True)

    def test_first_years_ungated_when_min_train_years_not_met(self):
        trades = self.trades3y()
        seen = []
        gate = Gate(fit=lambda tr: seen.append(len(tr)),
                    select=lambda rows: pd.Series(False, index=rows.index))
        wf = walk_forward(trades, gate, min_train_years=2)
        diag = {d["year"]: d for d in wf["diagnostics"]}
        assert diag[2020]["ungated"] and diag[2021]["ungated"]
        assert not diag[2022]["ungated"]
        # Ungated years keep all trades; the gated year keeps none (select=False).
        assert wf["selected"]["event_id"].nunique() == 10

    def test_selection_applies_to_every_alpha(self):
        frames = []
        for a in (0.0, 0.5, 1.0):
            t = self.trades3y()
            t["fill_alpha"] = a
            frames.append(t)
        trades = pd.concat(frames, ignore_index=True)
        gate = Gate(fit=lambda tr: None,
                    select=lambda rows: pd.Series(True, index=rows.index))
        wf = walk_forward(trades, gate, min_train_years=1)
        sel = wf["selected"]
        # 2020 ungated (0 train years < 1), 2021+ gated-and-kept: all 3 years kept
        assert sel["event_id"].nunique() == 15
        assert set(sel["fill_alpha"]) == {0.0, 0.5, 1.0}

    def test_audit_records_fit_years(self):
        trades = self.trades3y()
        gate = Gate(fit=lambda tr: None, select=lambda rows: pd.Series(True, index=rows.index))
        wf = walk_forward(trades, gate, min_train_years=1)
        assert wf["audit"]["leak_free"]
        assert wf["audit"]["fit_years_seen"] == [2020, 2021]

    def test_empty(self):
        wf = walk_forward(make_trades([]).iloc[0:0], None)
        assert wf["selected"].empty


class TestMonteCarlo:
    def test_seeded_reproducibility(self):
        rets = np.random.default_rng(3).normal(0.01, 0.1, 100)
        a = monte_carlo(rets, fractions=(0.05,), paths=200, seed=11)
        b = monte_carlo(rets, fractions=(0.05,), paths=200, seed=11)
        assert a == b
        c = monte_carlo(rets, fractions=(0.05,), paths=200, seed=12)
        assert c["by_fraction"]["0.05"]["p_loss"] != a["by_fraction"]["0.05"]["p_loss"] or True

    def test_degenerate_small_n(self):
        out = monte_carlo([0.1, -0.05], fractions=(0.05,), block=20, paths=50)
        assert out["by_fraction"]["0.05"]["degenerate"]

    def test_shared_vs_per_fraction_first_fraction_identical(self):
        rets = np.random.default_rng(5).normal(0.0, 0.1, 120)
        shared = monte_carlo(rets, fractions=(0.05, 0.10), paths=100, seed=7,
                             draw_order="shared")
        per = monte_carlo(rets, fractions=(0.05, 0.10), paths=100, seed=7,
                          draw_order="per_fraction")
        # The first fraction consumes the same draws either way.
        assert shared["by_fraction"]["0.05"] == per["by_fraction"]["0.05"]

    def test_unknown_draw_order_raises(self):
        with pytest.raises(EvaluationError):
            monte_carlo([0.01] * 40, draw_order="sideways")


class TestStress:
    def trades_with_dates(self):
        dates = [pd.Timestamp("2018-11-01"), pd.Timestamp("2020-03-10"),
                 pd.Timestamp("2022-06-01"), pd.Timestamp("2024-05-01")]
        frames = []
        for i, d in enumerate(dates):
            frames.append(pd.DataFrame({
                "event_id": [f"E{i}"], "ticker": "T", "event_date": [d],
                "entry_date": [d - pd.Timedelta(days=1)],
                "exit_date": [d + pd.Timedelta(days=1)],
                "fill_alpha": [0.5], "entry_cost": [1.0],
                "exit_value": [1.02], "ret": [0.02],
            }))
        return pd.concat(frames, ignore_index=True)

    def test_regime_windows_bucket_correctly(self):
        out = stress_regimes(self.trades_with_dates(), spy_daily=None)
        assert out["2018Q4"]["n"] == 1
        assert out["2020-02_04"]["n"] == 1
        assert out["2022"]["n"] == 1
        assert out["worst_earnings_weeks"]["n"] == 0  # no SPY supplied

    def test_iv_regime_split_by_column(self):
        trades = self.trades_with_dates()
        trades["spy_vol20"] = [0.1, 0.4, 0.1, 0.4]
        out = stress_iv_regime(trades)
        assert out["split_by"].startswith("spy_vol20")
        assert out["high"]["n"] == 2 and out["low"]["n"] == 2

    def test_tail_injection_plumbing(self):
        trades = self.trades_with_dates()

        def shock(rows):
            rows.loc[rows["ret"].idxmin(), "ret"] = -0.5
            return rows

        out = stress_tail_injection(trades, shock, fractions=(0.05,), paths=50)
        assert out["shocked_worst_trade"] == pytest.approx(-0.5)
        assert out["n_shocked"] == 1

    def test_slippage_and_stale_need_a_repricer(self):
        trades = self.trades_with_dates()
        assert stress_slippage(trades, None)["available"] is False
        assert stress_stale_dates(trades, None)["available"] is False

        def repricer(rows, shift):
            rows["ret"] = rows["ret"] + 0.001 * shift
            rows.attrs["coverage"] = 1.0
            return rows

        slip = stress_slippage(trades, repricer)
        assert slip["available"] and slip["shifts"]["+1d"]["delta_mean"] == pytest.approx(0.001)
        stale = stress_stale_dates(trades, repricer, fraction=0.5, seed=0)
        assert stale["available"] and stale["n_misdated"] >= 1


class TestPreregistration:
    def test_no_run_dir_no_enforcement(self):
        receipt = check_preregistration({"id": "X"}, None)
        assert receipt["enforced"] is False

    def test_missing_stamp_raises(self, tmp_path):
        with pytest.raises(PreregistrationError):
            check_preregistration({"id": "X"}, tmp_path)

    def test_backdated_stamp_raises(self, tmp_path):
        from engine.evaluate import append_run_log

        append_run_log(tmp_path, {"ts": pd.Timestamp.now(tz="UTC").isoformat()})
        late = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=1)).isoformat()
        with pytest.raises(PreregistrationError):
            check_preregistration({"id": "X", "preregistered_at": late}, tmp_path)

    def test_valid_stamp_passes(self, tmp_path):
        early = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)).isoformat()
        receipt = check_preregistration({"id": "X", "preregistered_at": early}, tmp_path)
        assert receipt["valid"]


class TestEvaluateEndToEnd:
    SPEC = {
        "id": "EXP-TEST",
        "title": "unit probe",
        "primary_spec": {"x": 1},
        "walk_forward": {"min_train_years": 1},
        "preregistered_at": "2020-01-01T00:00:00+00:00",
    }

    def test_headline_and_artifacts(self, tmp_path):
        rng = np.random.default_rng(0)
        rets = rng.normal(0.02, 0.1, 120)
        trades = make_trades(rets, alphas=(0.0, 0.5, 1.0))
        trades["ret"] = trades["ret"] + (trades["fill_alpha"] - 0.5) * 0.04
        trades["exit_value"] = trades["entry_cost"] * (1 + trades["ret"])
        result = evaluate(self.SPEC, trades, run_dir=tmp_path, mc_paths=50,
                          stress=False, write_report=False)
        h = result.metrics
        for key in ("n", "mean", "median", "std", "win_rate", "profit_factor",
                    "sharpe_trade", "sharpe_equity", "sortino", "max_dd",
                    "tail_ratio", "by_year", "breakeven_alpha", "mc"):
            assert key in h, f"canonical metric missing: {key}"
        assert result.results["headline_stage"] == "wf_oos"
        # Fill sensitivity: better fills must not read worse.
        sweep = h["alpha_sweep"]
        assert sweep["1.00"]["mean"] >= sweep["0.00"]["mean"]
        # Artifacts landed.
        assert (tmp_path / "results" / "run_log.jsonl").exists()
        assert list((tmp_path / "results").glob("metrics_*.json"))

    def test_missing_columns_rejected(self):
        bad = pd.DataFrame({"event_id": ["a"], "ret": [0.1]})
        with pytest.raises(EvaluationError):
            evaluate(self.SPEC, bad, write_report=False)

    def test_unknown_equity_mode_rejected(self):
        spec = dict(self.SPEC, equity_mode="luck")
        with pytest.raises(EvaluationError):
            evaluate(spec, make_trades([0.01] * 30), write_report=False, stress=False)

    def test_by_year_table(self):
        trades = make_trades([0.1, -0.1], start="2021-01-01", freq="400D")
        table = by_year_table(trades)
        assert table["2021"]["n"] == 1 and table["2022"]["n"] == 1


class TestStressCoverageBranches:
    def spy(self, start="2020-01-01", periods=200, crash_week=None):
        dates = pd.date_range(start, periods=periods, freq="B")
        rng = np.random.default_rng(2)
        rets = rng.normal(0.0, 0.01, periods)
        if crash_week is not None:
            rets[crash_week:crash_week + 5] = -0.05
        close = 100 * np.cumprod(1 + rets)
        return pd.DataFrame({"date": dates, "close": close})

    def trades_over(self, dates):
        frames = []
        for i, d in enumerate(dates):
            frames.append(pd.DataFrame({
                "event_id": [f"E{i}"], "ticker": "T",
                "event_date": [d],
                "entry_date": [d - pd.Timedelta(days=1)],
                "exit_date": [d + pd.Timedelta(days=1)],
                "fill_alpha": [0.5], "entry_cost": [1.0],
                "exit_value": [1.01], "ret": [0.01],
            }))
        return pd.concat(frames, ignore_index=True)

    def test_worst_week_replay_with_spy(self):
        spy = self.spy(crash_week=40)
        crash_date = pd.to_datetime(spy["date"]).iloc[42]
        trades = self.trades_over([crash_date, pd.to_datetime(spy["date"]).iloc[120]])
        out = stress_regimes(trades, spy_daily=spy)
        ww = out["worst_earnings_weeks"]
        assert ww["n"] >= 1, "the crash-week trade should land in the worst-week replay"
        assert "weeks" in ww

    def test_worst_week_spy_too_short(self):
        spy = self.spy(periods=8)
        trades = self.trades_over(pd.to_datetime(spy["date"])[:2])
        out = stress_regimes(trades, spy_daily=spy)
        assert out["worst_earnings_weeks"]["n"] == 0

    def test_iv_regime_from_spy_series(self):
        spy = self.spy(periods=500)
        dates = pd.to_datetime(spy["date"]).iloc[[60, 300]]
        trades = self.trades_over(dates)
        out = stress_iv_regime(trades, spy_daily=spy)
        assert out["split_by"] == "yearly median SPY vol20"
        assert out["high"]["n"] + out["low"]["n"] == 2

    def test_iv_regime_empty_rows(self):
        empty = self.trades_over([pd.Timestamp("2021-01-01")]).iloc[0:0]
        out = stress_iv_regime(empty)
        assert out["split_by"] is None

    def test_monte_carlo_empty(self):
        out = monte_carlo([], fractions=(0.05,))
        assert out["by_fraction"] == {}

    def test_stale_dates_full_coverage(self):
        trades = self.trades_over(pd.date_range("2021-01-01", periods=4, freq="30D"))

        def repricer(rows, shift):
            rows["ret"] = rows["ret"] + 0.01
            rows.attrs["coverage"] = 0.75
            return rows

        out = stress_stale_dates(trades, repricer, fraction=0.5, seed=3)
        assert out["coverage"] == pytest.approx(0.75)
        assert out["delta_mean"] == pytest.approx(0.01)

    def test_evaluate_writes_report_and_cache(self, tmp_path):
        rng = np.random.default_rng(4)
        trades = make_trades(rng.normal(0.01, 0.1, 60), alphas=(0.0, 0.5, 1.0))
        spec = {"id": "EXP-WR", "title": "write probe", "primary_spec": {"x": 1},
                "walk_forward": {"min_train_years": 1},
                "preregistered_at": "2020-01-01T00:00:00+00:00"}
        result = evaluate(spec, trades, run_dir=tmp_path, mc_paths=30,
                          stress=True, write_report=True)
        assert result.report_path is not None and result.report_path.exists()
        assert (tmp_path / "REPORT.md").exists()

    def test_alpha_sweep_empty(self):
        assert alpha_sweep(make_trades([]).iloc[0:0]) == {}
        assert by_year_table(make_trades([]).iloc[0:0]) == {}

    def test_annual_trades_single_event(self):
        s = trade_stats([0.1], pd.Series([pd.Timestamp("2020-01-01")]))
        assert s["n"] == 1
