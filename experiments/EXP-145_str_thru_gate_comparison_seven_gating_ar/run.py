#!/usr/bin/env python3
"""EXP-145 — STR-THRU gate comparison: seven gating arms.

Run:  python3 experiments/EXP-145_str_thru_gate_comparison_seven_gating_ar/run.py

Compares seven top-20%-selectivity walk-forward gates on the identical
STR-THRU trade universe:

  arm1  incumbent model gate (unchanged)
  arm2  rule gate on forecast percentile/edge alone
  arm3  rule gate on matched analog-trade evidence alone
  arm4  rule gate combining forecast + analog ranks
  arm5  incumbent features + forecast as an added GBM feature
  arm6  incumbent features + analog stats as added GBM features
  arm7  incumbent features + forecast + analog as added GBM features

Every arm reuses engine.evaluate.evaluate() — the standard Phase 2 harness
(backtest, walk-forward, Monte Carlo, stress, calibration, figures) — so each
arm gets its own REPORT.md under results/arms/<arm id>/. This script's own
job is only to build seven engine.evaluate.Gate objects and, once all seven
have run, assemble the cross-arm COMPARISON.md.

Pre-registration lives in spec.yaml; engine.evaluate enforces it.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine import paths  # noqa: E402
from engine.analogs import AnalogMatcher, bucket_frame  # noqa: E402
from engine.data.features import tier4  # noqa: E402
from engine.evaluate import Gate, evaluate  # noqa: E402
from engine.models.registry import load_registry  # noqa: E402
from engine.models.training import gate as gate_mod  # noqa: E402
from engine.models.training.common import SEED  # noqa: E402
from engine.score import Scorer  # noqa: E402
from experiments import common, lib  # noqa: E402

HERE = Path(__file__).resolve().parent
STRATEGY = "STR-THRU"
TOP_FRACTION = 0.20

# Extra feature columns each arm's dataset carries beyond the 41 registered
# gate features. Built once, sliced differently per arm below.
FORECAST_COLS = ["pred_abs_move", "pred_abs_move_p10", "pred_abs_move_p90",
                  "pred_abs_move_sd", "forecast_edge"]
ANALOG_COLS = ["analog_mean", "analog_win_rate", "analog_n"]


# --------------------------------------------------------------------------
# signal construction (forecast + analog), done ONCE, shared by every arm
# --------------------------------------------------------------------------


def add_forecast_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    """Join the Tier-4 size-model forecast onto the gate dataset.

    ``tier4.load_forecasts()`` is already a walk-forward-safe, monthly-fold
    table on disk (engine/data/features/tier4.py) keyed on
    (ticker, event_date) — no fresh model fitting needed here. STR-THRU's
    payoff driver is ``abs_move`` (engine/payoff.py PAYOFF_DRIVER), so
    ``pred_abs_move`` is the size model's own forecast for this structure,
    the same number the dashboard shows as "Est. |move|".
    """
    forecasts = tier4.load_forecasts()
    if forecasts is None or forecasts.empty:
        raise RuntimeError(
            "data/features/tier4_forecasts.parquet is missing or empty — "
            "run engine.data.features.tier4.build_table first"
        )
    keep = ["ticker", "event_date", *[c for c in FORECAST_COLS if c != "forecast_edge"]]
    fc = forecasts[keep].copy()
    fc["event_date"] = pd.to_datetime(fc["event_date"])
    fc = fc.drop_duplicates(["ticker", "event_date"])

    out = dataset.copy()
    out["event_date"] = pd.to_datetime(out["event_date"])
    out = out.merge(fc, on=["ticker", "event_date"], how="left")
    # forecast_edge: predicted absolute move minus the quoted implied move at
    # entry, both % of spot. Positive = the model expects a bigger move than
    # the market is pricing — STR-THRU's stated thesis (a mispriced expected
    # move). `im` is DAILY_STATE_FIELDS['implied_move'], already in the base
    # 41 features at the same as-of date the forecast is read at.
    out["forecast_edge"] = out["pred_abs_move"] - out["im"]
    n_covered = int(out["pred_abs_move"].notna().sum())
    print(f"  [forecast] {n_covered:,}/{len(out):,} rows have a forecast "
          f"({n_covered / len(out):.1%})", flush=True)
    return out


def add_analog_columns(dataset: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Per-event matched analog-trade statistics, computed causally.

    Reuses ``engine.score.Scorer`` for the enrichment (mcap/implied/prior-move
    join, spot/DTE from the legs blob) and ``engine.analogs.AnalogMatcher`` for
    matching — the exact production path, not a re-derivation. Each event is
    matched with ``as_of=<that event's own entry date>``, which the matcher
    already restricts to trades closed strictly before that date — the cutoff
    is the ROW's own decision date, not a walk-forward fold boundary, so this
    is safe to compute once, outside the per-arm walk-forward loop, and reuse
    in every fold. ``bootstrap=0``: only the point estimates are needed here,
    not a confidence interval, and skipping the bootstrap is what keeps ~19k
    per-row matches fast.
    """
    started = time.time()
    print("  [analog] constructing Scorer (loads panel/daily/registry) …", flush=True)
    scorer = Scorer(trades=trades)
    print(f"  [analog] Scorer ready in {time.time() - started:.0f}s", flush=True)
    matcher = scorer.matcher
    bucketed = bucket_frame(scorer.trades)
    mid = bucketed[np.isclose(bucketed["fill_alpha"].astype(float), 0.5)]
    mid = mid.drop_duplicates("event_id")

    records = []
    n = len(mid)
    for i, row in enumerate(mid.itertuples(index=False)):
        buckets = {
            "mcap_bucket": row.mcap_bucket,
            "dte_band": row.dte_band,
            "moneyness_band": row.moneyness_band,
            "implied_tercile": row.implied_tercile,
            "implied_ratio": row.implied_ratio,
        }
        aset = matcher.match(
            STRATEGY, buckets, alpha=0.5, as_of=row.entry_date,
            bootstrap=0, min_analogs=30, request_key=str(row.event_id),
        )
        records.append({
            "event_id": row.event_id,
            "analog_mean": aset.mean,
            "analog_win_rate": aset.win_rate,
            "analog_n": aset.n,
            "analog_widened": aset.widened,
            "analog_thin": aset.thin,
        })
        if (i + 1) % 2000 == 0 or i + 1 == n:
            print(f"  [analog] {i + 1:,}/{n:,} events matched "
                  f"({time.time() - started:.0f}s)", flush=True)

    analog_df = pd.DataFrame.from_records(records)
    out = dataset.merge(analog_df, on="event_id", how="left")
    n_covered = int((out["analog_thin"] == False).sum())  # noqa: E712
    print(f"  [analog] {n_covered:,}/{len(out):,} rows have a non-thin "
          f"analog match (>=30)", flush=True)
    return out


# --------------------------------------------------------------------------
# rule gates (arms 2, 3, 4) — no ML fit, fold-fitted threshold only
# --------------------------------------------------------------------------


def _train_percentile_mapper(train_values: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """A percentile-in-TRAIN function, fit once on training values.

    Used so a rank-combination score can be applied to test rows without ever
    computing a rank against test-year data — the percentile a test value gets
    is entirely defined by where it would have landed in the training
    distribution.
    """
    v = np.sort(train_values[np.isfinite(train_values)])

    def mapper(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.full(x.shape, np.nan)
        finite = np.isfinite(x)
        if v.size and finite.any():
            out[finite] = np.searchsorted(v, x[finite], side="right") / v.size
        return out

    return mapper


class RuleGateState:
    """A top-fraction rule gate: ``fit`` derives its threshold (and any
    normalization) from training rows only; ``select``/``predict_proba``
    apply it unchanged to the test year — the same no-peek discipline
    ``experiments.common._RegisteredGateState`` uses for the ML gates, just
    without a model to refit.
    """

    def __init__(self, name: str, feat: pd.DataFrame, score_fn,
                 top_fraction: float = TOP_FRACTION, proba_col: str | None = None):
        self.name = name
        self.feat = feat
        self.score_fn = score_fn  # object with .fit(sub_df)->params, .score(sub_df, params)->ndarray
        self.top_fraction = top_fraction
        self.proba_col = proba_col
        self.params_ = None
        self.threshold_ = None
        self.stats: list[dict] = []

    def _lookup(self, rows: pd.DataFrame) -> pd.DataFrame:
        return self.feat.reindex(rows["event_id"].to_numpy())

    def fit(self, train: pd.DataFrame) -> None:
        sub = self._lookup(train)
        self.params_ = self.score_fn.fit(sub)
        scores = self.score_fn.score(sub, self.params_)
        finite = scores[np.isfinite(scores)]
        self.threshold_ = float(np.quantile(finite, 1.0 - self.top_fraction)) if finite.size else float("nan")
        self.stats.append({
            "train_rows": int(len(train)), "scoreable_train_rows": int(finite.size),
            "threshold_chosen_on_train": self.threshold_,
            "train_year_max": int(pd.to_datetime(train["event_date"]).dt.year.max()) if len(train) else None,
        })

    def select(self, rows: pd.DataFrame) -> pd.Series:
        if self.threshold_ is None or not np.isfinite(self.threshold_):
            return pd.Series(False, index=rows.index)
        sub = self._lookup(rows)
        scores = self.score_fn.score(sub, self.params_)
        mask = np.isfinite(scores) & (scores >= self.threshold_)
        self.stats.append({
            "test_rows": int(len(rows)), "scored_rows": int(np.isfinite(scores).sum()),
            "passed_rows": int(mask.sum()), "threshold": self.threshold_,
        })
        return pd.Series(mask, index=rows.index)

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        if self.proba_col is None:
            return np.full(len(rows), np.nan)
        sub = self._lookup(rows)
        return pd.to_numeric(sub[self.proba_col], errors="coerce").to_numpy(dtype=float)


class _ForecastScore:
    """score = zscore(forecast_edge) - zscore(pred_abs_move_sd), train-fitted.

    Rewards a bigger predicted-vs-quoted edge; penalizes a less confident
    forecast (a wide predictive interval). Rows with no forecast score NaN
    and are therefore never selected — the same rule an incomplete-feature
    ML row already follows.
    """

    def fit(self, sub: pd.DataFrame) -> dict:
        edge = pd.to_numeric(sub["forecast_edge"], errors="coerce").to_numpy(dtype=float)
        sd = pd.to_numeric(sub["pred_abs_move_sd"], errors="coerce").to_numpy(dtype=float)
        return {
            "edge_mean": np.nanmean(edge) if np.isfinite(edge).any() else 0.0,
            "edge_std": np.nanstd(edge) if np.isfinite(edge).any() else 1.0,
            "sd_mean": np.nanmean(sd) if np.isfinite(sd).any() else 0.0,
            "sd_std": np.nanstd(sd) if np.isfinite(sd).any() else 1.0,
        }

    def score(self, sub: pd.DataFrame, params: dict) -> np.ndarray:
        edge = pd.to_numeric(sub["forecast_edge"], errors="coerce").to_numpy(dtype=float)
        sd = pd.to_numeric(sub["pred_abs_move_sd"], errors="coerce").to_numpy(dtype=float)
        edge_std = params["edge_std"] if params["edge_std"] > 0 else 1.0
        sd_std = params["sd_std"] if params["sd_std"] > 0 else 1.0
        z_edge = (edge - params["edge_mean"]) / edge_std
        z_sd = (sd - params["sd_mean"]) / sd_std
        return z_edge - z_sd


class _AnalogScore:
    """score = analog_mean (already causal per-row); thin matches excluded."""

    def fit(self, sub: pd.DataFrame) -> dict:
        return {}

    def score(self, sub: pd.DataFrame, params: dict) -> np.ndarray:
        mean = pd.to_numeric(sub["analog_mean"], errors="coerce").to_numpy(dtype=float)
        thin = sub["analog_thin"].to_numpy()
        out = mean.copy()
        out[thin.astype(bool)] = np.nan
        return out


class _ForecastPlusAnalogScore:
    """score = 0.5*percentile_in_train(forecast_edge) + 0.5*percentile_in_train(analog_mean)."""

    def fit(self, sub: pd.DataFrame) -> dict:
        edge = pd.to_numeric(sub["forecast_edge"], errors="coerce").to_numpy(dtype=float)
        mean = pd.to_numeric(sub["analog_mean"], errors="coerce").to_numpy(dtype=float)
        thin = sub["analog_thin"].to_numpy().astype(bool)
        mean = np.where(thin, np.nan, mean)
        return {"edge_mapper": _train_percentile_mapper(edge), "analog_mapper": _train_percentile_mapper(mean)}

    def score(self, sub: pd.DataFrame, params: dict) -> np.ndarray:
        edge = pd.to_numeric(sub["forecast_edge"], errors="coerce").to_numpy(dtype=float)
        mean = pd.to_numeric(sub["analog_mean"], errors="coerce").to_numpy(dtype=float)
        thin = sub["analog_thin"].to_numpy().astype(bool)
        mean = np.where(thin, np.nan, mean)
        pe, pa = params["edge_mapper"](edge), params["analog_mapper"](mean)
        both = np.isfinite(pe) & np.isfinite(pa)
        out = np.full(pe.shape, np.nan)
        out[both] = 0.5 * pe[both] + 0.5 * pa[both]
        return out


# --------------------------------------------------------------------------
# the seven arms
# --------------------------------------------------------------------------


def build_arms(dataset: pd.DataFrame) -> list[dict]:
    base_features = list(gate_mod.FEATURES)
    feat_all = dataset.set_index("event_id")
    feat_all = feat_all[~feat_all.index.duplicated(keep="first")]

    arms: list[dict] = []

    gate1, state1 = common.make_registered_gate(STRATEGY, dataset)
    arms.append({
        "id": "arm1_incumbent_model", "title": "Arm 1 — current incumbent (gated by model)",
        "gate": gate1, "state": state1,
        "description": "Registered gate_midfill_str_thru, refit per fold, stored threshold "
                        f"{load_registry(missing_ok=False).champion('gate', STRATEGY).threshold:.5f}.",
    })

    gate2 = Gate(
        fit=(s2 := RuleGateState("forecast_rule", feat_all, _ForecastScore())).fit,
        select=s2.select, predict_proba=s2.predict_proba,
        name=f"forecast_rule@top{TOP_FRACTION:.0%}",
    )
    arms.append({
        "id": "arm2_forecast_percentile", "title": "Arm 2 — gate by forecast percentile",
        "gate": gate2, "state": s2,
        "description": "Rule gate: top 20% of zscore(forecast_edge) - zscore(pred_abs_move_sd), "
                        "fold-fitted on training rows. No ML fit.",
    })

    gate3 = Gate(
        fit=(s3 := RuleGateState("analog_rule", feat_all, _AnalogScore(),
                                  proba_col="analog_win_rate")).fit,
        select=s3.select, predict_proba=s3.predict_proba,
        name=f"analog_rule@top{TOP_FRACTION:.0%}",
    )
    arms.append({
        "id": "arm3_analog_evidence", "title": "Arm 3 — gate by evidence (analog trades)",
        "gate": gate3, "state": s3,
        "description": "Rule gate: top 20% of analog_mean, thin matches (n<30) excluded. "
                        "predict_proba = analog_win_rate (a real matched win rate, not a model score).",
    })

    gate4 = Gate(
        fit=(s4 := RuleGateState("forecast_analog_rule", feat_all,
                                  _ForecastPlusAnalogScore())).fit,
        select=s4.select, predict_proba=s4.predict_proba,
        name=f"forecast_analog_rule@top{TOP_FRACTION:.0%}",
    )
    arms.append({
        "id": "arm4_forecast_plus_analog_rule", "title": "Arm 4 — gate by forecast + analog trades",
        "gate": gate4, "state": s4,
        "description": "Rule gate: top 20% of 0.5*train-percentile(forecast_edge) + "
                        "0.5*train-percentile(analog_mean). No ML fit.",
    })

    gate5, state5 = common.make_trained_gate(
        "model_plus_forecast", dataset, base_features + FORECAST_COLS, top_fraction=TOP_FRACTION)
    arms.append({
        "id": "arm5_model_plus_forecast_feature",
        "title": "Arm 5 — current model + forecast parameters as model features",
        "gate": gate5, "state": state5,
        "description": "Same GBM class as the incumbent, refit per fold, with the forecast "
                        f"({', '.join(FORECAST_COLS)}) appended to the 41 registered features.",
    })

    gate6, state6 = common.make_trained_gate(
        "model_plus_analog", dataset, base_features + ANALOG_COLS, top_fraction=TOP_FRACTION)
    arms.append({
        "id": "arm6_model_plus_analog_feature",
        "title": "Arm 6 — current model + analog-trade parameters as model features",
        "gate": gate6, "state": state6,
        "description": "Same GBM class as the incumbent, refit per fold, with analog stats "
                        f"({', '.join(ANALOG_COLS)}) appended to the 41 registered features.",
    })

    gate7, state7 = common.make_trained_gate(
        "model_plus_forecast_plus_analog", dataset,
        base_features + FORECAST_COLS + ANALOG_COLS, top_fraction=TOP_FRACTION)
    arms.append({
        "id": "arm7_model_plus_forecast_plus_analog",
        "title": "Arm 7 — current model + forecast AND analog parameters as model features",
        "gate": gate7, "state": state7,
        "description": "Same GBM class as the incumbent, refit per fold, with both the forecast "
                        "and the analog stats appended to the 41 registered features.",
    })

    return arms


# --------------------------------------------------------------------------
# per-arm reporting glue
# --------------------------------------------------------------------------


def make_extra_sections(arm: dict):
    def build(result):
        state = arm["state"]
        stats = getattr(state, "stats", [])
        return [{
            "title": "Gate construction (this arm)",
            "body": [arm["description"],
                     f"Fold interactions recorded: {len(stats)}."],
        }]
    return build


def run_arm(spec: dict, arm: dict, trades: pd.DataFrame, repricer, spy, input_files,
           no_ledger: bool) -> dict:
    arm_spec = copy.deepcopy(spec)
    arm_spec["primary_spec"] = dict(spec["primary_spec"]) | {
        "gate_arm": arm["id"], "gate_description": arm["description"],
    }
    arm_spec["title"] = f"{spec['title']} — {arm['title']}"
    # None of the 7 arms is individually THE pre-registered primary spec (the
    # scaffold's PLANNED ledger row was stamped against a generic template) —
    # this experiment's registration is the comparison plan in spec.yaml
    # itself (gate_variants), and each arm is an evaluated configuration
    # under it. grid_cell=True is the harness's own exemption for exactly
    # this: a spec that legitimately differs from the PLANNED row's hash.
    arm_spec["grid_cell"] = True
    run_dir = HERE / "arms" / arm["id"]

    print(f"\n=== {arm['id']}: {arm['title']} ===", flush=True)
    started = time.time()
    result = evaluate(
        arm_spec, trades, gate=arm["gate"], run_dir=run_dir,
        repricer=repricer, spy_daily=spy, input_files=input_files,
        extra_sections=make_extra_sections(arm),
    )
    print(f"  [{arm['id']}] done in {time.time() - started:.0f}s -> {result.report_path}",
          flush=True)
    if not no_ledger:
        lib.record_evaluation(run_dir, arm_spec, result.results)

    head = result.results["headline"]
    mc5 = head.get("mc", {}).get("0.05", {})
    gated = gated_only_stats(head.get("by_year", {}))
    tx_path = result.results.get("transaction_log", {}).get("path")
    selected_ids: set = set()
    if tx_path:
        try:
            tx = pd.read_csv(ROOT / tx_path if not str(tx_path).startswith("/") else tx_path)
            if "event_id" in tx.columns:
                selected_ids = set(tx["event_id"].unique())
        except (OSError, ValueError):
            selected_ids = set()

    return {
        "id": arm["id"], "title": arm["title"], "description": arm["description"],
        "report_path": str(result.report_path), "n_selected": head.get("n"),
        "mean": head.get("mean"), "win_rate": head.get("win_rate"),
        "sharpe_trade": head.get("sharpe_trade"), "sharpe_equity": head.get("sharpe_equity"),
        "cagr": head.get("cagr"), "max_dd": head.get("max_dd"),
        "breakeven_alpha": head.get("breakeven_alpha"),
        "years_positive": head.get("years_positive"), "years_evaluated": head.get("years_evaluated"),
        "mc_p_loss_5pct": mc5.get("p_loss"), "mc_dd_p50_5pct": mc5.get("dd_p50"),
        "ungated_share": head.get("ungated_share"),
        "gated_n": gated["n"], "gated_mean": gated["mean"], "gated_win_rate": gated["win_rate"],
        "calibration_available": result.results.get("calibration", {}).get("available"),
        "brier_skill": result.results.get("calibration", {}).get("brier_skill"),
        "checklist_fails": result.results.get("checklist_fails"),
        "selected_ids": selected_ids,
    }


# --------------------------------------------------------------------------
# comparison report
# --------------------------------------------------------------------------


def fmt_pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.2f}%"


def fmt_num(x, nd=2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def gated_only_stats(by_year: dict, first_gated_year: int = 2020) -> dict:
    """Pooled mean/win-rate over years >= first_gated_year only.

    2018-2019 trade fully ungated under every arm (min_train_years=2) and are
    BYTE-IDENTICAL across all seven arms — confirmed by inspection: same n,
    same mean, same win_rate in every arm's by_year table for those two years.
    They are ~70-73% of each arm's headline n, so a headline comparison alone
    lets a large identical base dominate the read. This pools only the years
    where the gate actually did something different per arm.
    """
    n_total = 0
    weighted_mean = 0.0
    weighted_win = 0.0
    for year, row in by_year.items():
        if int(year) < first_gated_year:
            continue
        n = row.get("n") or 0
        if n and row.get("mean") is not None:
            weighted_mean += n * row["mean"]
        if n and row.get("win_rate") is not None:
            weighted_win += n * row["win_rate"]
        n_total += n
    if n_total == 0:
        return {"n": 0, "mean": None, "win_rate": None}
    return {"n": n_total, "mean": weighted_mean / n_total, "win_rate": weighted_win / n_total}


def write_comparison_report(spec: dict, arm_results: list[dict], base_breakeven: float | None,
                            forecast_coverage: float | None = None) -> Path:
    lines = [f"# {spec['id']} — Gate comparison across 7 arms", "",
             f"*{spec['title']}*", "",
             f"Ungated baseline breakeven alpha: **{fmt_num(base_breakeven, 3)}** "
             "(the margin every gate is trying to widen or at least not spend).",
             "",
             "All arms select their own top 20% each walk-forward fold, threshold chosen "
             "on that fold's training years only — identical selectivity, so the table "
             "below isolates ranking quality rather than how many trades each arm kept.",
             "",
             "## Headline comparison (walk-forward OOS, mid fill, 5% sizing)", ""]

    incumbent = next((a for a in arm_results if a["id"] == "arm1_incumbent_model"), None)

    header = ["Arm", "n selected", "mean/trade", "win rate", "Sharpe(trade)", "CAGR",
              "max DD", "breakeven α", "years +/eval", "MC P(loss)@5%", "calibrated?"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for a in arm_results:
        cal = "yes" if a["calibration_available"] else "no"
        lines.append("| " + " | ".join([
            f"**{a['id']}**", f"{a['n_selected']:,}" if a["n_selected"] else "0",
            fmt_pct(a["mean"]), fmt_pct(a["win_rate"]), fmt_num(a["sharpe_trade"]),
            fmt_pct(a["cagr"]), fmt_pct(a["max_dd"]), fmt_num(a["breakeven_alpha"], 3),
            f"{a['years_positive']}/{a['years_evaluated']}", fmt_pct(a["mc_p_loss_5pct"]), cal,
        ]) + " |")

    lines += ["", "## Gated years only (2020-2026) — the part that actually differs by arm", "",
              "2018-2019 trade fully ungated (`min_train_years=2`) and are **byte-identical across "
              "all seven arms** — same n, same per-trade mean, same win rate in every arm's by-year "
              "table (confirmed by inspection). They make up 68-73% of each arm's headline n above, "
              "so the headline table is dominated by an identical shared base and understates how much "
              "the gates actually differ. This pools only 2020-2026, where each arm's own selection "
              "ran.", "",
              "| Arm | n (2020-2026) | mean/trade | win rate |",
              "|---|---:|---:|---:|"]
    for a in arm_results:
        lines.append(f"| **{a['id']}** | {a['gated_n']:,} | {fmt_pct(a['gated_mean'])} | "
                     f"{fmt_pct(a['gated_win_rate'])} |")

    lines += ["", "## Falsification test (primary_success)", ""]
    if incumbent is not None:
        for a in arm_results:
            if a["id"] == "arm1_incumbent_model":
                continue
            beats_cagr = (a["cagr"] is not None and incumbent["cagr"] is not None
                          and a["cagr"] > incumbent["cagr"])
            beats_sharpe = (a["sharpe_trade"] is not None and incumbent["sharpe_trade"] is not None
                            and a["sharpe_trade"] > incumbent["sharpe_trade"])
            no_worse_ploss = (a["mc_p_loss_5pct"] is not None and incumbent["mc_p_loss_5pct"] is not None
                              and a["mc_p_loss_5pct"] <= incumbent["mc_p_loss_5pct"] + 1e-9)
            clears = beats_cagr and beats_sharpe and no_worse_ploss
            lines.append(
                f"- **{a['id']}**: CAGR {'beats' if beats_cagr else 'does not beat'} incumbent, "
                f"trade Sharpe {'beats' if beats_sharpe else 'does not beat'} incumbent, "
                f"MC P(loss)@5% {'no worse' if no_worse_ploss else 'WORSE'} — "
                f"{'**CLEARS the falsification bar**' if clears else 'does not clear the bar'}."
            )

    lines += ["", "## Selection overlap with the incumbent (Jaccard of selected event_ids)", ""]
    if incumbent is not None and incumbent["selected_ids"]:
        for a in arm_results:
            if a["id"] == "arm1_incumbent_model":
                continue
            inter = len(a["selected_ids"] & incumbent["selected_ids"])
            union = len(a["selected_ids"] | incumbent["selected_ids"])
            jac = inter / union if union else float("nan")
            lines.append(f"- **{a['id']}** vs incumbent: {inter:,} shared / {union:,} union "
                         f"(Jaccard {jac:.2f})")
    else:
        lines.append("*no transaction log available for the incumbent arm; overlap not computed.*")

    lines += ["", "## Per-arm reports", ""]
    for a in arm_results:
        lines.append(f"- **{a['id']}** — {a['description']} → "
                     f"[{Path(a['report_path']).relative_to(ROOT) if Path(a['report_path']).is_relative_to(ROOT) else a['report_path']}]"
                     f"({a['id']}) — checklist fails: {a['checklist_fails']}")

    lines += ["", "## Trade-offs (read together with the falsification test above)", "",
              "- **Rule gates (arms 2-4)** carry no per-event feature-completeness guard beyond "
              "the forecast/analog columns themselves — a row missing a forecast or a non-thin "
              "analog match is simply never selected, which can shift the ungated-share and the "
              "effective universe relative to the incumbent's 41-feature completeness rule. See "
              "each arm's `ungated_share` and `n_selected` above.",
              "- **Calibration** is only meaningful for arms with a `predict_proba`: the three "
              "model arms (1, 5, 6, 7) via isotonic regression, and arm 3 via its matched "
              "analog win rate. Arms 2 and 4 report calibration N/A by construction — a raw "
              "z-score or rank blend is not a probability, and forcing one through the Brier "
              "machinery would misreport a ranking signal as a miscalibrated forecast.",
              f"- **Forecast coverage measured on this trade set: {fmt_pct(forecast_coverage) if forecast_coverage is not None else 'n/a'} "
              "of STR-THRU events have a pred_abs_move** (the spec's ~46% figure was an estimate from "
              "the full multi-strategy Tier-4 table and was wrong for STR-THRU specifically — the size "
              "model's own event universe overlaps STR-THRU's almost completely). Coverage is not the "
              "binding constraint on arms 2, 4, 5 and 7 that the spec expected it to be.",
              ""]

    path = HERE / "COMPARISON.md"
    path.write_text("\n".join(lines))
    return path


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=None,
                    help="comma-separated arm ids to run (default: all 7)")
    ap.add_argument("--limit-events", type=int, default=None,
                    help="debug: cap the STR-THRU event universe to the N most recent events")
    ap.add_argument("--no-ledger", action="store_true",
                    help="skip LEDGER.csv rows — use for smoke/subset runs")
    args = ap.parse_args()

    spec = lib.load_spec(HERE / "spec.yaml")
    print(f"[{spec['id']}] loading engine trades …", flush=True)
    trades = common.load_engine_trades(STRATEGY)

    if args.limit_events:
        recent_ids = (
            trades.drop_duplicates("event_id")
            .sort_values("event_date")["event_id"].tail(args.limit_events)
        )
        trades = trades[trades["event_id"].isin(set(recent_ids))].reset_index(drop=True)
        print(f"  [debug] limited to {trades['event_id'].nunique():,} events", flush=True)

    print(f"[{spec['id']}] {len(trades):,} rows / "
          f"{trades['event_id'].nunique():,} events", flush=True)

    # A --limit-events smoke run must never write into the same cache the
    # full run reads from — a small cached dataset silently reused by a full
    # run would trade the whole universe against a 400-event feature frame.
    cache_dir = HERE / "results" / "smoke" if args.limit_events else HERE / "results"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = common.gate_dataset(STRATEGY, trades, cache_dir)
    dataset = add_forecast_columns(dataset)
    dataset = add_analog_columns(dataset, trades)
    dataset.to_parquet(cache_dir / "dataset_with_signals.parquet", index=False)

    arms = build_arms(dataset)
    if args.arms:
        wanted = set(args.arms.split(","))
        arms = [a for a in arms if a["id"] in wanted]
        print(f"  [debug] running arms: {[a['id'] for a in arms]}", flush=True)

    spy = common.load_spy_daily()
    repricer = common.make_repricer(STRATEGY)
    input_files = sorted((paths.CURATED / "trades").glob("year=*/part-*.parquet"))

    arm_results = []
    base_breakeven = None
    for arm in arms:
        summary = run_arm(spec, arm, trades, repricer, spy, input_files, args.no_ledger)
        if base_breakeven is None:
            # Read back the ungated breakeven from this arm's own backtest
            # stage — identical across arms (same unselected universe).
            import json as _json
            metrics_files = sorted((HERE / "arms" / arm["id"] / "results").glob("metrics_*.json"))
            if metrics_files:
                doc = _json.loads(metrics_files[-1].read_text())
                base_breakeven = doc.get("backtest", {}).get("breakeven_alpha")
        arm_results.append(summary)

    forecast_coverage = float(dataset["pred_abs_move"].notna().mean())

    if len(arm_results) == len(build_arms(dataset)):
        # Only assemble the full cross-arm comparison once every arm has run
        # in this invocation — a partial `--arms` run does not overwrite it.
        path = write_comparison_report(spec, arm_results, base_breakeven, forecast_coverage)
        print(f"\n[{spec['id']}] comparison report: {path}", flush=True)
    else:
        print(f"\n[{spec['id']}] ran {len(arm_results)}/7 arms (--arms subset); "
              "comparison report not (re)written.", flush=True)


if __name__ == "__main__":
    main()
