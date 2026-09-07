#!/usr/bin/env python3
"""EXP-148: add causal forecast-PnL and analog evidence to STR-RUNUP gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BASE_CACHE = (
    ROOT
    / "experiments/EXP-144_str_runup_t14_corrected_calendar_gate_rebaseline"
    / "results/factor_dataset.parquet"
)
SIGNAL_CACHE = RESULTS / "dataset_with_signals.parquet"
SCORE_CACHE = RESULTS / "oos_scores.parquet"
STRATEGY = "STR-RUNUP"
VARIANT = "e-14_x+0_target_dte=30"
PRIMARY = "model_plus_forecast_pnl_plus_analogs"
STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.analogs import match_frame  # noqa: E402
from engine.evaluate import Gate, evaluate  # noqa: E402
from engine.models.registry import bucket_residuals, load_registry  # noqa: E402
from engine.models.training import gate as gate_mod  # noqa: E402
from engine.models.training import implied_t1  # noqa: E402
from engine.payoff import simulate_returns  # noqa: E402
from engine.score import Scorer  # noqa: E402
from experiments import common, lib  # noqa: E402


BASE_FEATURES = tuple(gate_mod.FEATURES)
RAW_FORECAST = (
    "pred_im_t1",
    "pred_im_t1_p10",
    "pred_im_t1_p90",
    "pred_im_t1_sd",
    "predicted_runup",
)
FORECAST_PNL = (
    "forecast_pnl_mean",
    "forecast_pnl_win",
    "forecast_pnl_p10",
    "forecast_pnl_p90",
    "forecast_pnl_sd",
)
ANALOG = ("analog_mean", "analog_win_rate", "analog_n")
ARM_FEATURES = {
    "incumbent_registered_threshold": BASE_FEATURES,
    "base_top20": BASE_FEATURES,
    "model_plus_raw_forecast": BASE_FEATURES + RAW_FORECAST,
    "model_plus_forecast_pnl": BASE_FEATURES + FORECAST_PNL,
    "model_plus_analogs": BASE_FEATURES + ANALOG,
    "model_plus_raw_forecast_plus_analogs": BASE_FEATURES + RAW_FORECAST + ANALOG,
    PRIMARY: BASE_FEATURES + FORECAST_PNL + ANALOG,
}
ALL_ARMS = tuple(ARM_FEATURES)
LEARNED_ARMS = tuple(a for a in ALL_ARMS if a != "incumbent_registered_threshold")
MODEL_DRAWS = 4000
MIN_MODEL_ROWS = 500
MIN_VALID_QUOTE = 1.0


def log(message: str) -> None:
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-148 {elapsed:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def load_trades() -> pd.DataFrame:
    trades = common.load_engine_trades(STRATEGY)
    trades = trades[trades["variant"] == VARIANT].copy()
    log(
        f"Loaded {len(trades):,} trade rows, "
        f"{trades['event_id'].nunique():,} exact T-14 events"
    )
    return trades


def load_base(spec: dict) -> pd.DataFrame:
    if not BASE_CACHE.exists():
        raise FileNotFoundError(f"missing corrected feature cache: {BASE_CACHE}")
    snapshot = json.loads(paths.SNAPSHOT_FILE.read_text()).get("snapshot")
    expected = spec["data"]["data_snapshot"]
    if snapshot != expected:
        raise RuntimeError(f"Tier-3 snapshot changed: {snapshot} versus {expected}")
    frame = pd.read_parquet(BASE_CACHE)
    frame["event_id"] = frame["event_id"].astype(str)
    for column in ("event_date", "entry_date", "exit_date", "expiry"):
        frame[column] = pd.to_datetime(frame[column])
    frame["days_before_print"] = 14.0
    frame["mcap_usd"] = np.exp(pd.to_numeric(frame["mcap_log"], errors="coerce"))
    log(f"Loaded {len(frame):,} corrected midpoint feature rows")
    return frame


def _target_im_t1(frame: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    panel = scorer.context.panel
    target = panel[["ticker", "date", "or_implied"]].copy()
    target = target.rename(columns={"date": "event_date", "or_implied": "im_t1_actual"})
    target["event_date"] = pd.to_datetime(target["event_date"])
    target = target.drop_duplicates(["ticker", "event_date"], keep="last")
    out = frame.merge(target, on=["ticker", "event_date"], how="left")
    actual = pd.to_numeric(out["im_t1_actual"], errors="coerce")
    out.loc[actual < MIN_VALID_QUOTE, "im_t1_actual"] = np.nan
    return out


def _crossfit_residuals(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features = list(implied_t1.FEATURES)
    complete = np.isfinite(
        train[features + ["im_t1_actual"]].to_numpy(dtype=float)
    ).all(axis=1)
    use = train.loc[complete].reset_index(drop=True)
    if len(use) < MIN_MODEL_ROWS or use["ticker"].nunique() < 2:
        return np.empty(0), np.empty(0)
    folds = min(5, int(use["ticker"].nunique()))
    splitter = GroupKFold(n_splits=folds)
    pred = np.full(len(use), np.nan)
    X = use[features].to_numpy(dtype=float)
    y = use["im_t1_actual"].to_numpy(dtype=float)
    groups = use["ticker"].astype(str).to_numpy()
    for fit_index, val_index in splitter.split(X, y, groups):
        model = implied_t1.fit(X[fit_index], y[fit_index])
        pred[val_index] = model.predict(X[val_index])
    ok = np.isfinite(pred) & np.isfinite(y)
    return pred[ok], (y[ok] - pred[ok])


def _residual_pool(prediction: float, pool_pred: np.ndarray, pool_res: np.ndarray) -> np.ndarray:
    flat = np.asarray(pool_res, dtype=float)
    flat = flat[np.isfinite(flat)]
    if flat.size < 250:
        return np.empty(0)
    buckets = bucket_residuals(pool_pred, pool_res)
    if not buckets:
        return flat
    index = int(
        np.clip(
            np.searchsorted(buckets["edges"], prediction, side="right") - 1,
            0,
            len(buckets["pools"]) - 1,
        )
    )
    chosen = np.asarray(buckets["pools"][index], dtype=float)
    if chosen.size < int(buckets.get("min_pool", 0)):
        return flat
    return chosen[np.isfinite(chosen)]


def add_forecast_signals(frame: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    out = _target_im_t1(frame, scorer)
    for column in RAW_FORECAST + FORECAST_PNL:
        out[column] = np.nan
    forecast_features = list(implied_t1.FEATURES)

    for year in range(2019, int(out["year"].max()) + 1):
        train = out[out["year"] < year].copy()
        test_index = out.index[out["year"] == year]
        if len(test_index) == 0:
            continue
        train_complete = np.isfinite(
            train[forecast_features + ["im_t1_actual"]].to_numpy(dtype=float)
        ).all(axis=1)
        train_fit = train.loc[train_complete]
        test_complete = np.isfinite(
            out.loc[test_index, forecast_features].to_numpy(dtype=float)
        ).all(axis=1)
        scored_index = test_index[test_complete]
        if len(train_fit) < MIN_MODEL_ROWS or len(scored_index) == 0:
            log(
                f"T-1 forecast {year}: skipped, train={len(train_fit):,}, "
                f"scoreable={len(scored_index):,}"
            )
            continue

        log(
            f"T-1 forecast {year}: fitting {len(train_fit):,}, "
            f"scoring {len(scored_index):,}, cross-fitting residuals"
        )
        pool_pred, pool_res = _crossfit_residuals(train_fit)
        model = implied_t1.fit(
            train_fit[forecast_features].to_numpy(dtype=float),
            train_fit["im_t1_actual"].to_numpy(dtype=float),
        )
        points = np.asarray(
            model.predict(out.loc[scored_index, forecast_features].to_numpy(dtype=float)),
            dtype=float,
        )
        out.loc[scored_index, "pred_im_t1"] = points

        for position, (idx, point) in enumerate(zip(scored_index, points), start=1):
            residuals = _residual_pool(float(point), pool_pred, pool_res)
            if residuals.size == 0:
                continue
            q10, q90 = np.quantile(residuals, [0.10, 0.90])
            out.at[idx, "pred_im_t1_p10"] = max(0.0, float(point + q10))
            out.at[idx, "pred_im_t1_p90"] = max(0.0, float(point + q90))
            out.at[idx, "pred_im_t1_sd"] = float(residuals.std(ddof=1))
            entry_im = float(out.at[idx, "im"]) if pd.notna(out.at[idx, "im"]) else np.nan
            if np.isfinite(entry_im) and entry_im >= MIN_VALID_QUOTE:
                out.at[idx, "predicted_runup"] = float(point - entry_im)

            cost = float(out.at[idx, "entry_cost"])
            spot = float(out.at[idx, "spot_entry"])
            entry_date = pd.Timestamp(out.at[idx, "entry_date"])
            if not np.isfinite(cost) or cost <= 0 or not np.isfinite(spot) or spot <= 0:
                continue
            payoff = scorer.payoff(STRATEGY, 0.5, entry_date)
            if payoff is None:
                continue
            event_id = str(out.at[idx, "event_id"])
            seed_payload = (
                f"{scorer.snapshot}|EXP-148|{event_id}|{year}"
            ).encode()
            seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            driver_draws = float(point) + rng.choice(
                residuals, size=MODEL_DRAWS, replace=True
            )
            payoff_noise = payoff.residual_draws(MODEL_DRAWS, rng)
            returns = simulate_returns(
                driver_draws, payoff, spot, cost, payoff_noise
            )
            returns = returns[np.isfinite(returns)]
            if returns.size == 0:
                continue
            out.at[idx, "forecast_pnl_mean"] = float(returns.mean())
            out.at[idx, "forecast_pnl_win"] = float((returns > 0).mean())
            out.at[idx, "forecast_pnl_p10"] = float(np.quantile(returns, 0.10))
            out.at[idx, "forecast_pnl_p90"] = float(np.quantile(returns, 0.90))
            out.at[idx, "forecast_pnl_sd"] = float(returns.std(ddof=1))
            if position % 500 == 0 or position == len(scored_index):
                log(
                    f"T-1 forecast {year}: simulated "
                    f"{position:,}/{len(scored_index):,}"
                )
        log(
            f"T-1 forecast {year}: PnL coverage "
            f"{out.loc[test_index, 'forecast_pnl_mean'].notna().sum():,}/"
            f"{len(test_index):,}"
        )
    return out


def prepare_signals(spec: dict, trades: pd.DataFrame, force: bool) -> pd.DataFrame:
    if SIGNAL_CACHE.exists() and not force:
        frame = pd.read_parquet(SIGNAL_CACHE)
        for column in ("event_date", "entry_date", "exit_date", "expiry"):
            frame[column] = pd.to_datetime(frame[column])
        log(f"Loaded signal dataset cache: {len(frame):,} events")
        return frame

    frame = load_base(spec)
    log("Constructing scorer for causal payoff maps and analog matches")
    scorer = Scorer(trades=trades)
    log("Matching causal analog sets")
    analogs = match_frame(
        scorer.trades,
        scorer.matcher,
        strategy=STRATEGY,
        alpha=0.5,
        progress_every=500,
    )
    frame = frame.merge(analogs, on="event_id", how="left")
    frame = add_forecast_signals(frame, scorer)
    frame.to_parquet(SIGNAL_CACHE, index=False)
    coverage = {
        column: int(frame[column].notna().sum())
        for column in (*RAW_FORECAST, *FORECAST_PNL, *ANALOG)
    }
    write_json(RESULTS / "signal_coverage.json", coverage)
    log(f"Signal dataset written: {SIGNAL_CACHE}")
    return frame


def _fit_gate(train: pd.DataFrame, test: pd.DataFrame, features: tuple[str, ...]):
    columns = list(features)
    train_ok = np.isfinite(
        train[columns + ["ret"]].to_numpy(dtype=float)
    ).all(axis=1)
    test_ok = np.isfinite(test[columns].to_numpy(dtype=float)).all(axis=1)
    fit_rows = train.loc[train_ok]
    score = np.full(len(test), np.nan)
    pwin = np.full(len(test), np.nan)
    if len(fit_rows) < MIN_MODEL_ROWS or not test_ok.any():
        return score, pwin, None, int(len(fit_rows)), int(test_ok.sum())
    X_train = fit_rows[columns].to_numpy(dtype=float)
    y_train = fit_rows["ret"].to_numpy(dtype=float)
    model = gate_mod.fit(X_train, y_train)
    train_score = np.asarray(model.predict(X_train), dtype=float)
    score[test_ok] = model.predict(test.loc[test_ok, columns].to_numpy(dtype=float))
    cutoff = gate_mod.choose_threshold(train_score, 0.20)
    base_rate = float((y_train > 0).mean())
    try:
        iso = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(train_score, (y_train > 0).astype(float))
        pwin[test_ok] = iso.predict(score[test_ok])
    except ValueError:
        pwin[test_ok] = base_rate
    return score, pwin, cutoff, int(len(fit_rows)), int(test_ok.sum())


def generate_scores(
    dataset: pd.DataFrame, registered_threshold: float, force: bool
) -> tuple[pd.DataFrame, list[dict]]:
    diagnostics_path = RESULTS / "fold_diagnostics.json"
    if SCORE_CACHE.exists() and diagnostics_path.exists() and not force:
        scores = pd.read_parquet(SCORE_CACHE)
        scores["event_date"] = pd.to_datetime(scores["event_date"])
        diagnostics = json.loads(diagnostics_path.read_text())
        log(f"Loaded score cache: {len(scores):,} OOS events")
        return scores, diagnostics

    base_complete = np.isfinite(
        dataset[list(BASE_FEATURES) + ["ret", "mcap_usd"]].to_numpy(dtype=float)
    ).all(axis=1)
    live_domain = dataset["mcap_usd"] >= 1e9
    pieces = []
    diagnostics = []
    for year in range(2020, int(dataset["year"].max()) + 1):
        train = dataset[dataset["year"] < year].copy()
        test = dataset[
            (dataset["year"] == year) & base_complete & live_domain
        ].copy()
        if test.empty:
            continue
        piece = test[
            [
                "event_id",
                "ticker",
                "event_date",
                "year",
                "ret",
                "mcap_usd",
                "relative_spread",
            ]
        ].rename(columns={"ret": "realized_ret"}).reset_index(drop=True)
        fold_diag = {
            "year": year,
            "n_test_live_domain": int(len(test)),
            "training_rows": {},
            "scoreable_test": {},
            "cutoffs": {},
        }
        cache = {}
        for arm in LEARNED_ARMS:
            features = ARM_FEATURES[arm]
            score, pwin, cutoff, n_train, n_test = _fit_gate(
                train, test, features
            )
            piece[arm] = score
            piece[f"{arm}_pwin"] = pwin
            piece[f"selected_{arm}"] = (
                np.isfinite(score) & (score >= cutoff)
                if cutoff is not None
                else False
            )
            fold_diag["training_rows"][arm] = n_train
            fold_diag["scoreable_test"][arm] = n_test
            fold_diag["cutoffs"][arm] = cutoff
            if arm == "base_top20":
                cache["base_score"] = score
                cache["base_pwin"] = pwin
        base_score = cache["base_score"]
        piece["incumbent_registered_threshold"] = base_score
        piece["incumbent_registered_threshold_pwin"] = cache["base_pwin"]
        piece["selected_incumbent_registered_threshold"] = (
            np.isfinite(base_score) & (base_score >= registered_threshold)
        )
        fold_diag["training_rows"]["incumbent_registered_threshold"] = (
            fold_diag["training_rows"]["base_top20"]
        )
        fold_diag["scoreable_test"]["incumbent_registered_threshold"] = (
            fold_diag["scoreable_test"]["base_top20"]
        )
        fold_diag["cutoffs"]["incumbent_registered_threshold"] = registered_threshold
        diagnostics.append(fold_diag)
        pieces.append(piece)
        selected_text = ", ".join(
            f"{arm}={int(piece[f'selected_{arm}'].sum())}"
            for arm in ALL_ARMS
        )
        log(f"Gate fold {year}: {len(test):,} candidates; {selected_text}")

    scores = pd.concat(pieces, ignore_index=True)
    scores["event_date"] = pd.to_datetime(scores["event_date"])
    scores = scores.sort_values(["event_date", "event_id"]).reset_index(drop=True)
    scores.to_parquet(SCORE_CACHE, index=False)
    write_json(diagnostics_path, diagnostics)
    log(f"OOS scores written: {len(scores):,} events")
    return scores, diagnostics


class PrecomputedGate:
    def __init__(self, scores: pd.DataFrame, arm: str):
        indexed = scores.drop_duplicates("event_id").set_index("event_id")
        self.selected = indexed[f"selected_{arm}"].astype(bool).to_dict()
        self.pwin = indexed[f"{arm}_pwin"].astype(float).to_dict()
        self.arm = arm

    def fit(self, train: pd.DataFrame) -> None:
        return None

    def select(self, rows: pd.DataFrame) -> pd.Series:
        return rows["event_id"].map(self.selected).fillna(False).astype(bool)

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        return rows["event_id"].map(self.pwin).to_numpy(dtype=float)

    def gate(self) -> Gate:
        return Gate(
            fit=self.fit,
            select=self.select,
            predict_proba=self.predict_proba,
            name=self.arm,
        )


def rank_metrics(scores: pd.DataFrame) -> dict:
    output = {}
    common_mask = np.isfinite(scores[list(ALL_ARMS)].to_numpy(dtype=float)).all(axis=1)
    for arm in ALL_ARMS:
        own = scores[[arm, "realized_ret"]].dropna()
        common = scores.loc[common_mask, [arm, "realized_ret"]]
        def measure(frame):
            if len(frame) < 20 or frame[arm].nunique() < 10:
                return {"n": int(len(frame)), "spearman": None, "top_bottom": None}
            corr = spearmanr(frame[arm], frame["realized_ret"], nan_policy="omit")
            ranked = frame.copy()
            ranked["decile"] = pd.qcut(
                ranked[arm], 10, labels=False, duplicates="drop"
            ) + 1
            means = ranked.groupby("decile")["realized_ret"].mean()
            return {
                "n": int(len(ranked)),
                "spearman": float(corr.statistic),
                "top_bottom": float(means.iloc[-1] - means.iloc[0]),
            }
        output[arm] = {"own": measure(own), "common": measure(common)}
    return output


def matched_selectivity(scores: pd.DataFrame) -> list[dict]:
    common = np.isfinite(scores[list(ALL_ARMS)].to_numpy(dtype=float)).all(axis=1)
    rows = []
    for year, group in scores.loc[common].groupby("year"):
        n = int(np.floor(0.20 * len(group)))
        for arm in ALL_ARMS:
            selected = group.nlargest(n, arm)
            rows.append({
                "year": int(year),
                "arm": arm,
                "candidates": int(len(group)),
                "selected": n,
                "mean": float(selected["realized_ret"].mean()) if n else None,
            })
    return rows


def weekly_bootstrap(
    scores: pd.DataFrame,
    challenger: str = PRIMARY,
    incumbent: str = "base_top20",
    draws: int = 10000,
) -> dict:
    frame = scores.copy()
    frame["week"] = frame["event_date"].dt.to_period("W").astype(str)
    blocks = []
    for _, group in frame.groupby("week"):
        left = group.loc[group[f"selected_{challenger}"], "realized_ret"]
        right = group.loc[group[f"selected_{incumbent}"], "realized_ret"]
        blocks.append((left.sum(), len(left), right.sum(), len(right)))
    blocks = np.asarray(blocks, dtype=float)
    rng = np.random.default_rng(148)
    estimates = []
    for start in range(0, draws, 1000):
        count = min(1000, draws - start)
        index = rng.integers(0, len(blocks), size=(count, len(blocks)))
        sampled = blocks[index].sum(axis=1)
        ok = (sampled[:, 1] > 0) & (sampled[:, 3] > 0)
        estimates.extend(
            (
                sampled[ok, 0] / sampled[ok, 1]
                - sampled[ok, 2] / sampled[ok, 3]
            ).tolist()
        )
    arr = np.asarray(estimates)
    left = frame.loc[frame[f"selected_{challenger}"], "realized_ret"]
    right = frame.loc[frame[f"selected_{incumbent}"], "realized_ret"]
    return {
        "challenger": challenger,
        "incumbent": incumbent,
        "observed": float(left.mean() - right.mean()),
        "ci90": np.quantile(arr, [0.05, 0.95]).tolist(),
        "ci95": np.quantile(arr, [0.025, 0.975]).tolist(),
        "p_gt_zero": float((arr > 0).mean()),
        "draws": int(len(arr)),
    }


def slice_metrics(scores: pd.DataFrame) -> list[dict]:
    cap = scores["mcap_usd"].to_numpy(dtype=float)
    spread = scores["relative_spread"].to_numpy(dtype=float)
    masks = {
        "mcap_1b_10b": (cap >= 1e9) & (cap < 1e10),
        "mcap_ge_10b": cap >= 1e10,
        "spread_le_15pct": spread <= 0.15,
        "spread_15_50pct": (spread > 0.15) & (spread <= 0.50),
        "spread_gt_50pct": spread > 0.50,
        "spread_le_50pct": spread <= 0.50,
    }
    rows = []
    for arm in ALL_ARMS:
        selected = scores[f"selected_{arm}"].astype(bool).to_numpy()
        for cohort, mask in masks.items():
            part = scores.loc[selected & mask]
            rows.append({
                "arm": arm,
                "cohort": cohort,
                "candidates": int(mask.sum()),
                "selected": int(len(part)),
                "mean": float(part["realized_ret"].mean()) if len(part) else None,
                "win_rate": float((part["realized_ret"] > 0).mean())
                if len(part) else None,
            })
    return rows


def arm_spec(spec: dict, arm: str) -> dict:
    if arm == PRIMARY:
        return spec
    out = dict(spec)
    out["primary_spec"] = dict(spec["primary_spec"])
    out["primary_spec"]["challenger"] = arm
    out["promotion_target"] = None
    out["grid_cell"] = True
    return out


def fmt_pct(value) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{100 * value:+.2f}%"


def report_sections(
    result,
    evaluations: dict,
    ranks: dict,
    matched: list[dict],
    bootstrap: dict,
    slices: list[dict],
    diagnostics: list[dict],
    coverage: dict,
):
    primary = result.results["headline"]
    base = evaluations["base_top20"].results["headline"]
    registered = evaluations["incumbent_registered_threshold"].results["headline"]
    slice_index = {(r["arm"], r["cohort"]): r for r in slices}
    primary_liquid = slice_index[(PRIMARY, "spread_le_50pct")]
    base_liquid = slice_index[("base_top20", "spread_le_50pct")]
    checks = {
        "mean": primary["mean"] > base["mean"],
        "cagr": primary["cagr"] > base["cagr"],
        "sharpe": primary["sharpe_trade"] > base["sharpe_trade"],
        "positive_year_share": (
            primary["years_positive"] / max(primary["years_evaluated"], 1)
            >= base["years_positive"] / max(base["years_evaluated"], 1)
        ),
        "breakeven_alpha": (
            primary.get("breakeven_alpha") is not None
            and base.get("breakeven_alpha") is not None
            and primary["breakeven_alpha"] <= base["breakeven_alpha"]
        ),
        "spread_le_50pct": (
            primary_liquid["mean"] is not None
            and base_liquid["mean"] is not None
            and primary_liquid["mean"] > base_liquid["mean"]
        ),
        "policy_ci": bootstrap["ci90"][0] > 0,
    }
    clears = all(checks.values())
    arm_rows = []
    for arm in ALL_ARMS:
        metrics = (
            primary if arm == PRIMARY
            else evaluations[arm].results["headline"]
        )
        liquid = slice_index[(arm, "spread_le_50pct")]
        arm_rows.append([
            arm,
            f"{metrics['n']:,}",
            fmt_pct(metrics["mean"]),
            fmt_pct(metrics.get("dollar_weighted")),
            fmt_pct(metrics.get("cagr")),
            f"{metrics['sharpe_trade']:.2f}",
            f"{metrics['years_positive']}/{metrics['years_evaluated']}",
            f"{metrics['breakeven_alpha']:.3f}"
            if metrics.get("breakeven_alpha") is not None else "n/a",
            f"{liquid['selected']:,}",
            fmt_pct(liquid["mean"]),
        ])
    rank_rows = []
    for arm in ALL_ARMS:
        own = ranks[arm]["own"]
        common = ranks[arm]["common"]
        rank_rows.append([
            arm,
            f"{own['n']:,}",
            f"{own['spearman']:+.3f}" if own["spearman"] is not None else "n/a",
            fmt_pct(own["top_bottom"]),
            f"{common['n']:,}",
            f"{common['spearman']:+.3f}" if common["spearman"] is not None else "n/a",
            fmt_pct(common["top_bottom"]),
        ])
    matched_rows = [
        [str(r["year"]), r["arm"], f"{r['selected']:,}", fmt_pct(r["mean"])]
        for r in matched
    ]
    slice_rows = [
        [
            r["arm"],
            r["cohort"],
            f"{r['candidates']:,}",
            f"{r['selected']:,}",
            fmt_pct(r["mean"]),
            fmt_pct(r["win_rate"]),
        ]
        for r in slices
    ]
    training_rows = []
    for row in diagnostics:
        training_rows.append([
            str(row["year"]),
            f"{row['n_test_live_domain']:,}",
            f"{row['training_rows']['base_top20']:,}",
            f"{row['training_rows'][PRIMARY]:,}",
            f"{row['scoreable_test'][PRIMARY]:,}",
            f"{row['cutoffs']['base_top20']:.4f}",
            (
                f"{row['cutoffs'][PRIMARY]:.4f}"
                if row["cutoffs"][PRIMARY] is not None
                else "n/a"
            ),
        ])
    lo, hi = bootstrap["ci90"]
    return [
        {
            "title": "Forecast-plus-analog gate decision",
            "body": [
                f"**{'PRIMARY SUCCESS CRITERIA MET' if clears else 'PRIMARY DOES NOT CLEAR'}**.",
                f"Primary minus equal-threshold base policy value: "
                f"{fmt_pct(bootstrap['observed'])}; 90% earnings-week interval "
                f"[{fmt_pct(lo)}, {fmt_pct(hi)}], "
                f"P(greater than zero) {bootstrap['p_gt_zero']:.1%}.",
                f"After excluding entry spreads above 50%: primary "
                f"{fmt_pct(primary_liquid['mean'])} on {primary_liquid['selected']:,} "
                f"trades versus base {fmt_pct(base_liquid['mean'])} on "
                f"{base_liquid['selected']:,}.",
                f"The registered-threshold operational benchmark returned "
                f"{fmt_pct(registered['mean'])} over {registered['n']:,} trades.",
                "This is exploratory. A passing result requires a separate "
                "confirmatory experiment before any registry change.",
            ],
        },
        {
            "title": "One T-14 book, seven gate variants",
            "columns": [
                "gate", "selected", "mean", "capital weighted", "CAGR",
                "Sharpe", "years+", "breakeven alpha", "n spread<=50%", "mean spread<=50%",
            ],
            "align": ["---"] + ["---:"] * 9,
            "rows": arm_rows,
        },
        {
            "title": "Pre-registered primary checks",
            "columns": ["check", "status"],
            "align": ["---", "---"],
            "rows": [[name, "PASS" if passed else "FAIL"] for name, passed in checks.items()],
        },
        {
            "title": "Ranking quality",
            "note": "Own coverage and the all-arm common cohort are both shown.",
            "columns": [
                "gate", "own n", "own Spearman", "own top-bottom",
                "common n", "common Spearman", "common top-bottom",
            ],
            "align": ["---", "---:", "---:", "---:", "---:", "---:", "---:"],
            "rows": rank_rows,
        },
        {
            "title": "Exact annual top-20-percent common-cohort diagnostic",
            "note": "Uses each full test-year score distribution and cannot trigger promotion.",
            "columns": ["year", "gate", "selected", "mean"],
            "align": ["---:", "---", "---:", "---:"],
            "rows": matched_rows,
        },
        {
            "title": "Market-cap and quoted-spread slices",
            "columns": ["gate", "cohort", "candidates", "selected", "mean", "win rate"],
            "align": ["---", "---", "---:", "---:", "---:", "---:"],
            "rows": slice_rows,
        },
        {
            "title": "Fold populations and thresholds",
            "columns": [
                "test year", "live-domain test", "base train", "primary train",
                "primary scoreable", "base cutoff", "primary cutoff",
            ],
            "align": ["---:"] * 7,
            "rows": training_rows,
        },
        {
            "title": "Signal construction and coverage",
            "body": [
                f"- Corrected T-14 events: {coverage['events']:,}.",
                f"- Causal T-1 point forecasts: {coverage['pred_im_t1']:,}.",
                f"- Forecast PnL distributions: {coverage['forecast_pnl_mean']:,}.",
                f"- Analog means: {coverage['analog_mean']:,}.",
                f"- Non-thin analog matches: {coverage['analog_non_thin']:,}.",
                "- Stored event-month Tier-4 forecasts were not used: 2,266 of "
                "4,219 covered rows entered before their stored model fold. "
                "Each T-1 forecast here is refit inside its outer annual training set.",
                "- Forecast errors are five-fold ticker-group cross-fitted inside "
                "that training set. Payoff maps and analog pools contain only "
                "trades closed before the candidate entry date.",
            ],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-signals", action="store_true")
    parser.add_argument("--force-scores", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    spec = lib.load_spec(HERE / "spec.yaml")
    registry = load_registry(missing_ok=False)
    incumbent = registry.champion("gate", STRATEGY)
    if incumbent.id != "gate_midfill_str_runup":
        raise RuntimeError(f"registered STR-RUNUP gate changed: {incumbent.id}")
    if tuple(incumbent.features) != BASE_FEATURES:
        raise RuntimeError("registered STR-RUNUP feature contract changed")
    if incumbent.threshold is None:
        raise RuntimeError("registered STR-RUNUP threshold is missing")

    trades_all = load_trades()
    dataset = prepare_signals(spec, trades_all, args.force_signals)
    scores, diagnostics = generate_scores(
        dataset, float(incumbent.threshold), args.force_scores
    )
    ranks = rank_metrics(scores)
    matched = matched_selectivity(scores)
    bootstrap = weekly_bootstrap(scores)
    slices = slice_metrics(scores)
    coverage = {
        "events": int(len(dataset)),
        "pred_im_t1": int(dataset["pred_im_t1"].notna().sum()),
        "forecast_pnl_mean": int(dataset["forecast_pnl_mean"].notna().sum()),
        "analog_mean": int(dataset["analog_mean"].notna().sum()),
        "analog_non_thin": int(
            (
                dataset["analog_mean"].notna()
                & ~dataset["analog_thin"].fillna(True).astype(bool)
            ).sum()
        ),
    }
    write_json(RESULTS / "ranking.json", ranks)
    write_json(RESULTS / "matched_selectivity.json", matched)
    write_json(RESULTS / "policy_bootstrap.json", bootstrap)
    write_json(RESULTS / "slice_metrics.json", slices)
    write_json(RESULTS / "signal_coverage.json", coverage)

    event_ids = set(scores["event_id"].astype(str))
    trades = trades_all[trades_all["event_id"].astype(str).isin(event_ids)].copy()
    if trades["event_id"].nunique() != len(scores):
        raise RuntimeError("OOS scores do not reconcile to the priced trade set")
    spy = common.load_spy_daily()
    repricer = common.make_repricer(STRATEGY)
    input_files = [
        BASE_CACHE,
        SIGNAL_CACHE,
        SCORE_CACHE,
        RESULTS / "fold_diagnostics.json",
    ]
    input_files += sorted((paths.CURATED / "trades").glob("year=*/part-*.parquet"))

    evaluations = {}
    run_order = [arm for arm in ALL_ARMS if arm != PRIMARY] + [PRIMARY]
    for arm in run_order:
        this_spec = arm_spec(spec, arm)
        run_dir = HERE if arm == PRIMARY else HERE / "arms" / arm
        log(f"Evaluating {arm}")
        gate = PrecomputedGate(scores, arm).gate()
        if arm == PRIMARY:
            extra = lambda result: report_sections(
                result,
                evaluations,
                ranks,
                matched,
                bootstrap,
                slices,
                diagnostics,
                coverage,
            )
        else:
            extra = lambda result, arm=arm: [{
                "title": "Gate construction",
                "body": [
                    f"Arm: {arm}.",
                    f"Features: {len(ARM_FEATURES[arm])}.",
                    "Scores were fit in strict annual outer folds and the "
                    "threshold came only from each folds training predictions.",
                ],
            }]
        result = evaluate(
            this_spec,
            trades,
            gate=gate,
            run_dir=run_dir,
            repricer=repricer,
            spy_daily=spy,
            input_files=input_files,
            extra_sections=extra,
        )
        evaluations[arm] = result
        if not args.no_ledger:
            lib.record_evaluation(run_dir, this_spec, result.results)
        headline = result.results["headline"]
        log(
            f"{arm}: n={headline['n']:,}, mean={headline['mean']:+.4f}, "
            f"CAGR={headline['cagr']:+.4f}, Sharpe={headline['sharpe_trade']:.3f}"
        )

    comparison = {
        arm: evaluations[arm].results["headline"]
        for arm in ALL_ARMS
    }
    comparison["policy_bootstrap"] = bootstrap
    comparison["ranking"] = ranks
    write_json(RESULTS / "comparison.json", comparison)
    log(f"Generated report: {evaluations[PRIMARY].report_path}")


if __name__ == "__main__":
    main()
