#!/usr/bin/env python3
"""EXP-149: drift-aware STR-RUNUP forecast-PnL calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SIGNAL_CACHE = (
    ROOT
    / "experiments/EXP-148_str_runup_forecast_pnl_analog_gate"
    / "results/dataset_with_signals.parquet"
)
SIM_DIR = ROOT / "experiments/EXP-142_str_runup_t14_factor_simulation_pnl_gate"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SIM_DIR))

from engine import paths  # noqa: E402
from engine.evaluate import Gate, evaluate  # noqa: E402
from engine.models.training import gate as gate_mod  # noqa: E402
from engine.models.training import implied_t1  # noqa: E402
from experiments import common, lib  # noqa: E402
import simulation as sim  # noqa: E402


STRATEGY = "STR-RUNUP"
VARIANT = "e-14_x+0_target_dte=30"
PRIMARY = "drift_empirical_sign"
ARMS = (
    "current_payoff_map",
    "surface_no_drift",
    "drift_symmetric_sign",
    PRIMARY,
)
NEW_ARMS = ARMS[1:]
MIN_TRAIN = 500
MIN_POOL = 250
STARTED = time.monotonic()


def log(message):
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-149 {elapsed:,.0f}s] {message}", flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def safe_spearman(x, y):
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    ok = np.isfinite(left) & np.isfinite(right)
    if ok.sum() < 3 or np.std(left[ok]) == 0 or np.std(right[ok]) == 0:
        return None
    return float(spearmanr(left[ok], right[ok]).statistic)


def load_trades():
    trades = common.load_engine_trades(STRATEGY)
    trades = trades[trades["variant"] == VARIANT].copy()
    log(
        f"Loaded {len(trades):,} trade rows and "
        f"{trades['event_id'].nunique():,} exact T-14 events"
    )
    return trades


def load_dataset(spec):
    snapshot = json.loads(paths.SNAPSHOT_FILE.read_text()).get("snapshot")
    expected = spec["data"]["data_snapshot"]
    if snapshot != expected:
        raise RuntimeError(f"Tier-3 snapshot changed: {snapshot} versus {expected}")
    frame = pd.read_parquet(SIGNAL_CACHE)
    for column in ("event_date", "entry_date", "exit_date", "expiry"):
        frame[column] = pd.to_datetime(frame[column])
    frame["signed_log_move_pct"] = 100.0 * np.log(
        frame["spot_exit_actual"] / frame["spot_entry_leg"]
    )
    required = list(dict.fromkeys(
        list(gate_mod.FEATURES)
        + list(implied_t1.FEATURES)
        + [
            "im_t1_actual", "target_log1p_abs_spot", "spot_sign",
            "spot_entry_leg", "spot_exit_actual", "strike_leg", "entry_cost",
            "exit_value", "ret", "mcap_log", "mcap_usd", "im",
            "relative_spread", "signed_log_move_pct",
        ]
    ))
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    ok = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    ok &= numeric["entry_cost"].to_numpy() > 0
    ok &= numeric["spot_entry_leg"].to_numpy() > 0
    ok &= numeric["spot_exit_actual"].to_numpy() > 0
    ok &= numeric["strike_leg"].to_numpy() > 0
    out = frame.loc[ok].copy().sort_values(["event_date", "event_id"])
    log(
        f"Model-ready dataset: {len(out):,}/{len(frame):,} events; "
        "old-forecast coverage is enforced only on the comparison cohort"
    )
    return out


def distribution_stats(values):
    x = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    quantiles = x.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "sd": float(x.std(ddof=1)),
        "positive_share": float((x > 0).mean()),
        "mean_absolute": float(x.abs().mean()),
        "quantiles": {f"p{int(q * 100):02d}": float(v) for q, v in quantiles.items()},
    }


def drift_diagnostics(full_frame, dataset):
    valid = full_frame.copy()
    valid["signed_log_move_pct"] = 100.0 * np.log(
        valid["spot_exit_actual"] / valid["spot_entry_leg"]
    )
    valid = valid[np.isfinite(valid["signed_log_move_pct"])]
    live = valid[(valid["year"] >= 2020) & (valid["mcap_usd"] >= 1e9)].copy()
    slices = {
        "all_corrected": distribution_stats(valid["signed_log_move_pct"]),
        "live_domain": distribution_stats(live["signed_log_move_pct"]),
        "live_spread_le_50pct": distribution_stats(
            live.loc[live["relative_spread"] <= 0.50, "signed_log_move_pct"]
        ),
        "live_spread_le_15pct": distribution_stats(
            live.loc[live["relative_spread"] <= 0.15, "signed_log_move_pct"]
        ),
    }
    by_year = []
    for year, rows in live.groupby("year"):
        row = distribution_stats(rows["signed_log_move_pct"])
        row["year"] = int(year)
        by_year.append(row)

    ranked = live.copy()
    ranked["move_decile"] = pd.qcut(
        ranked["signed_log_move_pct"], 10, labels=False, duplicates="drop"
    )
    deciles = []
    for decile, rows in ranked.groupby("move_decile"):
        deciles.append({
            "decile": int(decile) + 1,
            "n": int(len(rows)),
            "mean_move_pct": float(rows["signed_log_move_pct"].mean()),
            "mean_option_return": float(rows["ret"].mean()),
        })

    ordered = valid.sort_values(["ticker", "event_date"]).copy()
    ordered["prior_move"] = ordered.groupby("ticker")["signed_log_move_pct"].shift(1)
    ordered["prior4_mean"] = ordered.groupby("ticker")["signed_log_move_pct"].transform(
        lambda values: values.shift(1).rolling(4, min_periods=2).mean()
    )
    persistence = {
        "previous_event_spearman": safe_spearman(
            ordered["signed_log_move_pct"], ordered["prior_move"]
        ),
        "prior_four_mean_spearman": safe_spearman(
            ordered["signed_log_move_pct"], ordered["prior4_mean"]
        ),
    }

    direction = []
    features = list(gate_mod.FEATURES)
    for year in range(2020, 2027):
        train = dataset[dataset["year"] < year]
        test = dataset[(dataset["year"] == year) & (dataset["mcap_usd"] >= 1e9)].copy()
        if len(train) < MIN_TRAIN or test.empty:
            continue
        model = gate_mod.fit(
            train[features].to_numpy(float),
            train["signed_log_move_pct"].to_numpy(float),
            seed=149,
        )
        pred = model.predict(test[features].to_numpy(float))
        actual = test["signed_log_move_pct"].to_numpy(float)
        direction.append({
            "year": int(year),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "spearman": safe_spearman(pred, actual),
            "mae": float(np.mean(np.abs(pred - actual))),
            "zero_mae": float(np.mean(np.abs(actual))),
            "sign_accuracy": float(np.mean(np.sign(pred) == np.sign(actual))),
            "predicted_mean": float(np.mean(pred)),
            "actual_mean": float(np.mean(actual)),
        })
    return {
        "slices": slices,
        "by_year": by_year,
        "move_deciles": deciles,
        "persistence": persistence,
        "direction_walk_forward": direction,
    }


def payoff_design(implied, moneyness):
    im = np.asarray(implied, dtype=float)
    money = np.asarray(moneyness, dtype=float)
    absolute = np.abs(money)
    return np.column_stack([
        np.ones(len(im)),
        im,
        absolute,
        np.square(money) / 10.0,
        money,
        im * absolute / 10.0,
    ])


def crossfit_models(train):
    im_features = list(implied_t1.FEATURES)
    move_features = list(gate_mod.FEATURES)
    groups = train["ticker"].astype(str).to_numpy()
    folds = min(5, int(train["ticker"].nunique()))
    splitter = GroupKFold(n_splits=folds)
    im_oof = np.full(len(train), np.nan)
    move_oof = np.full(len(train), np.nan)
    for fold, (fit_index, val_index) in enumerate(
        splitter.split(train, groups=groups), start=1
    ):
        log(
            f"cross-fit {fold}/{folds}: fit {len(fit_index):,}, "
            f"validate {len(val_index):,}"
        )
        im_model = implied_t1.fit(
            train.iloc[fit_index][im_features].to_numpy(float),
            train.iloc[fit_index]["im_t1_actual"].to_numpy(float),
            seed=149 + fold,
        )
        move_model = gate_mod.fit(
            train.iloc[fit_index][move_features].to_numpy(float),
            train.iloc[fit_index]["target_log1p_abs_spot"].to_numpy(float),
            seed=249 + fold,
        )
        im_oof[val_index] = im_model.predict(
            train.iloc[val_index][im_features].to_numpy(float)
        )
        move_oof[val_index] = move_model.predict(
            train.iloc[val_index][move_features].to_numpy(float)
        )
    return im_oof, move_oof


def bin_edges(values, quantiles):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return np.unique(np.quantile(x, quantiles)) if len(x) else np.array([])


def value_bin(value, edges):
    if not np.isfinite(value) or len(edges) < 2:
        return -1
    return int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, len(edges) - 2))


def residual_pool(row, residuals, mcap_edges, im_edges):
    mcap_bin = value_bin(float(row["mcap_log"]), mcap_edges)
    im_bin = value_bin(float(row["im"]), im_edges)
    residual_mcap = np.array([value_bin(v, mcap_edges) for v in residuals["mcap_log"]])
    residual_im = np.array([value_bin(v, im_edges) for v in residuals["im"]])
    both = np.flatnonzero((residual_mcap == mcap_bin) & (residual_im == im_bin))
    if len(both) >= MIN_POOL:
        return both, "mcap_x_im"
    mcap = np.flatnonzero(residual_mcap == mcap_bin)
    if len(mcap) >= MIN_POOL:
        return mcap, "mcap"
    return np.arange(len(residuals)), "global"


def summarize_draws(draws):
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(values)),
        "pwin": float(np.mean(values > 0)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
        "sd": float(np.std(values, ddof=1)),
    }


def seed_for(event_id):
    payload = f"EXP-149|{event_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def simulate_year(train, test, draws):
    im_features = list(implied_t1.FEATURES)
    move_features = list(gate_mod.FEATURES)
    im_oof, move_oof = crossfit_models(train)

    im_model = implied_t1.fit(
        train[im_features].to_numpy(float),
        train["im_t1_actual"].to_numpy(float),
        seed=149,
    )
    move_model = gate_mod.fit(
        train[move_features].to_numpy(float),
        train["target_log1p_abs_spot"].to_numpy(float),
        seed=249,
    )
    pred_im = im_model.predict(test[im_features].to_numpy(float))
    pred_move = move_model.predict(test[move_features].to_numpy(float))

    train_money = 100.0 * np.log(
        train["spot_exit_actual"].to_numpy(float)
        / train["strike_leg"].to_numpy(float)
    )
    payoff_y = (
        train["exit_value"].to_numpy(float)
        / train["spot_entry_leg"].to_numpy(float)
    )
    payoff_x = payoff_design(train["im_t1_actual"], train_money)
    coefficient = np.linalg.lstsq(payoff_x, payoff_y, rcond=None)[0]
    payoff_residual = payoff_y - payoff_x @ coefficient

    residuals = pd.DataFrame({
        "err_im": train["im_t1_actual"].to_numpy(float) - im_oof,
        "err_move": train["target_log1p_abs_spot"].to_numpy(float) - move_oof,
        "spot_sign": train["spot_sign"].to_numpy(float),
        "payoff_error": payoff_residual,
        "mcap_log": train["mcap_log"].to_numpy(float),
        "im": train["im"].to_numpy(float),
    }).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    mcap_edges = bin_edges(residuals["mcap_log"], [0.0, 1 / 3, 2 / 3, 1.0])
    im_edges = bin_edges(residuals["im"], np.linspace(0.0, 1.0, 6))
    fallback = {"mcap_x_im": 0, "mcap": 0, "global": 0}
    rows = []

    for position, ((_, row), point_im, point_move) in enumerate(
        zip(test.iterrows(), pred_im, pred_move), start=1
    ):
        pool, pool_kind = residual_pool(row, residuals, mcap_edges, im_edges)
        fallback[pool_kind] += 1
        rng = np.random.default_rng(seed_for(row["event_id"]))
        chosen = rng.choice(pool, size=int(draws), replace=True)
        im_draw = np.maximum(
            float(point_im) + residuals["err_im"].to_numpy()[chosen], 0.0
        )
        move_draw = np.maximum(
            np.expm1(
                float(point_move) + residuals["err_move"].to_numpy()[chosen]
            ),
            0.0,
        )
        payoff_error = residuals["payoff_error"].to_numpy()[chosen]
        empirical_sign = residuals["spot_sign"].to_numpy()[chosen]
        symmetric_sign = rng.choice((-1.0, 1.0), size=int(draws))
        spot_entry = float(row["spot_entry_leg"])
        strike = float(row["strike_leg"])
        entry_cost = float(row["entry_cost"])

        record = {
            "event_id": str(row["event_id"]),
            "ticker": str(row["ticker"]),
            "event_date": pd.Timestamp(row["event_date"]),
            "year": int(row["year"]),
            "mcap_usd": float(row["mcap_usd"]),
            "relative_spread": float(row["relative_spread"]),
            "realized_ret": float(row["ret"]),
            "signed_log_move_pct": float(row["signed_log_move_pct"]),
            "pred_im_t1_new": float(point_im),
            "pred_abs_move_new": float(np.expm1(point_move)),
            "pool_n": int(len(pool)),
            "pool_kind": pool_kind,
            "current_payoff_map": float(row["forecast_pnl_mean"]),
            "current_payoff_map_pwin": float(row["forecast_pnl_win"]),
            "current_payoff_map_p10": float(row["forecast_pnl_p10"]),
            "current_payoff_map_p90": float(row["forecast_pnl_p90"]),
            "current_payoff_map_sd": float(row["forecast_pnl_sd"]),
        }

        directions = {
            "surface_no_drift": np.zeros(int(draws)),
            "drift_symmetric_sign": symmetric_sign * move_draw,
            PRIMARY: empirical_sign * move_draw,
        }
        for arm, signed_move in directions.items():
            exit_spot = spot_entry * np.exp(signed_move / 100.0)
            exit_money = 100.0 * np.log(exit_spot / strike)
            value_spot = np.maximum(
                payoff_design(im_draw, exit_money) @ coefficient + payoff_error,
                0.0,
            )
            simulated_return = (
                value_spot * spot_entry - entry_cost
            ) / entry_cost
            stats = summarize_draws(simulated_return)
            record[arm] = stats["mean"]
            record[f"{arm}_pwin"] = stats["pwin"]
            record[f"{arm}_p10"] = stats["p10"]
            record[f"{arm}_p90"] = stats["p90"]
            record[f"{arm}_sd"] = stats["sd"]
        rows.append(record)
        if position % 500 == 0 or position == len(test):
            log(f"simulated {position:,}/{len(test):,} test events")

    actual_im = test["im_t1_actual"].to_numpy(float)
    actual_move = np.expm1(test["target_log1p_abs_spot"].to_numpy(float))
    diagnostics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "residual_pool": int(len(residuals)),
        "payoff_coefficients": coefficient.tolist(),
        "payoff_oracle_mae_value_over_spot": float(np.mean(np.abs(payoff_residual))),
        "implied_t1": {
            "mae_pp": float(np.mean(np.abs(pred_im - actual_im))),
            "bias_pp": float(np.mean(pred_im - actual_im)),
            "pearson": float(np.corrcoef(pred_im, actual_im)[0, 1]),
        },
        "absolute_move": {
            "mae_pp": float(np.mean(np.abs(np.expm1(pred_move) - actual_move))),
            "bias_pp": float(np.mean(np.expm1(pred_move) - actual_move)),
            "pearson": float(np.corrcoef(np.expm1(pred_move), actual_move)[0, 1]),
        },
        "pool_fallback": fallback,
        "prior_positive_sign_share": float((train["spot_sign"] > 0).mean()),
    }
    return pd.DataFrame(rows), diagnostics


def generate_scores(dataset, draws, force):
    score_dir = RESULTS / "score_folds"
    score_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    diagnostics = []
    for year in range(2020, 2027):
        score_path = score_dir / f"scores_{year}.parquet"
        diagnostic_path = score_dir / f"diagnostics_{year}.json"
        if score_path.exists() and diagnostic_path.exists() and not force:
            piece = pd.read_parquet(score_path)
            piece["event_date"] = pd.to_datetime(piece["event_date"])
            pieces.append(piece)
            diagnostics.append(json.loads(diagnostic_path.read_text()))
            log(f"Year {year}: loaded {len(piece):,} cached forecasts")
            continue
        train = dataset[dataset["year"] < year].copy().reset_index(drop=True)
        test = dataset[dataset["year"] == year].copy().reset_index(drop=True)
        if len(train) < MIN_TRAIN or test.empty:
            log(f"Year {year}: skipped, train={len(train):,}, test={len(test):,}")
            continue
        log(f"Year {year}: fitting {len(train):,}, forecasting {len(test):,}")
        piece, diagnostic = simulate_year(train, test, draws)
        diagnostic["year"] = int(year)
        piece.to_parquet(score_path, index=False)
        write_json(diagnostic_path, diagnostic)
        pieces.append(piece)
        diagnostics.append(diagnostic)
        log(f"Year {year}: forecast complete")
    scores = pd.concat(pieces, ignore_index=True)
    scores["event_date"] = pd.to_datetime(scores["event_date"])
    return scores.sort_values(["event_date", "event_id"]).reset_index(drop=True), diagnostics


def accuracy_metrics(frame, arm):
    actual = frame["realized_ret"].to_numpy(float)
    pred = frame[arm].to_numpy(float)
    error = pred - actual
    p10 = frame[f"{arm}_p10"].to_numpy(float)
    p90 = frame[f"{arm}_p90"].to_numpy(float)
    return {
        "n": int(len(frame)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "pearson": float(np.corrcoef(pred, actual)[0, 1]),
        "spearman": safe_spearman(pred, actual),
        "interval_80_coverage": float(np.mean((actual >= p10) & (actual <= p90))),
        "interval_80_width": float(np.mean(p90 - p10)),
        "predicted_mean": float(np.mean(pred)),
        "realized_mean": float(np.mean(actual)),
    }


def accuracy_bundle(scores):
    output = {"overall": {}, "spread_le_50pct": {}, "by_year": {}, "quintiles": {}}
    liquid = scores[scores["relative_spread"] <= 0.50]
    for arm in ARMS:
        output["overall"][arm] = accuracy_metrics(scores, arm)
        output["spread_le_50pct"][arm] = accuracy_metrics(liquid, arm)
        output["by_year"][arm] = {
            str(int(year)): accuracy_metrics(rows, arm)
            for year, rows in scores.groupby("year")
        }
        ranked = scores[[arm, "realized_ret"]].copy()
        ranked["quintile"] = pd.qcut(ranked[arm], 5, labels=False, duplicates="drop")
        output["quintiles"][arm] = [
            {
                "quintile": int(quintile) + 1,
                "n": int(len(rows)),
                "predicted_mean": float(rows[arm].mean()),
                "realized_mean": float(rows["realized_ret"].mean()),
            }
            for quintile, rows in ranked.groupby("quintile")
        ]
    return output


def paired_error_bootstrap(scores, draws=10000):
    frame = scores[[
        "event_date", "realized_ret", PRIMARY, "current_payoff_map"
    ]].copy()
    frame["week"] = frame["event_date"].dt.to_period("W").astype(str)
    frame["difference"] = (
        (frame[PRIMARY] - frame["realized_ret"]).abs()
        - (frame["current_payoff_map"] - frame["realized_ret"]).abs()
    )
    groups = [rows["difference"].to_numpy(float) for _, rows in frame.groupby("week")]
    rng = np.random.default_rng(149)
    estimates = np.empty(int(draws))
    for index in range(int(draws)):
        chosen = rng.integers(0, len(groups), len(groups))
        estimates[index] = np.mean(np.concatenate([groups[i] for i in chosen]))
    return {
        "observed_mae_difference": float(frame["difference"].mean()),
        "ci90": np.quantile(estimates, [0.05, 0.95]).tolist(),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "p_below_zero": float(np.mean(estimates < 0)),
        "draws": int(draws),
        "weeks": int(len(groups)),
    }


class ForecastGate:
    def __init__(self, scores, arm):
        indexed = scores.drop_duplicates("event_id").set_index("event_id")
        self.selected = (indexed[arm] > 0).to_dict()
        self.probability = indexed[f"{arm}_pwin"].to_dict()

    def fit(self, train):
        return None

    def select(self, rows):
        return rows["event_id"].map(self.selected).fillna(False).astype(bool)

    def predict_proba(self, rows):
        return rows["event_id"].map(self.probability).to_numpy(float)

    def gate(self, arm):
        return Gate(
            fit=self.fit,
            select=self.select,
            predict_proba=self.predict_proba,
            name=arm,
        )


def cell_spec(spec, arm):
    if arm == PRIMARY:
        return spec
    out = dict(spec)
    out["primary_spec"] = dict(spec["primary_spec"])
    out["primary_spec"]["challenger"] = arm
    out["grid_cell"] = True
    return out


def fmt_pct(value):
    return "n/a" if value is None or not np.isfinite(value) else f"{100 * value:+.2f}%"


def report_sections(result, evaluations, accuracy, bootstrap, drift, model_diagnostics, counts):
    overall_rows = []
    for arm in ARMS:
        row = accuracy["overall"][arm]
        liquid = accuracy["spread_le_50pct"][arm]
        overall_rows.append([
            arm,
            f"{row['n']:,}",
            fmt_pct(row["mae"]),
            fmt_pct(row["rmse"]),
            fmt_pct(row["bias"]),
            f"{row['spearman']:+.3f}",
            f"{row['pearson']:+.3f}",
            f"{row['interval_80_coverage']:.1%}",
            fmt_pct(liquid["mae"]),
            f"{liquid['spearman']:+.3f}",
        ])

    year_rows = []
    for year in sorted(accuracy["by_year"][PRIMARY]):
        for arm in ARMS:
            row = accuracy["by_year"][arm][year]
            year_rows.append([
                year, arm, f"{row['n']:,}", fmt_pct(row["mae"]),
                fmt_pct(row["bias"]), f"{row['spearman']:+.3f}",
            ])

    book_rows = []
    for arm in ARMS:
        headline = (
            result.results["headline"]
            if arm == PRIMARY
            else evaluations[arm].results["headline"]
        )
        book_rows.append([
            arm,
            f"{headline['n']:,}",
            fmt_pct(headline["mean"]),
            fmt_pct(headline.get("dollar_weighted")),
            fmt_pct(headline.get("cagr")),
            f"{headline['sharpe_trade']:.2f}",
            f"{headline['years_positive']}/{headline['years_evaluated']}",
            f"{headline['breakeven_alpha']:.3f}"
            if headline.get("breakeven_alpha") is not None else "n/a",
        ])

    drift_rows = []
    for name, row in drift["slices"].items():
        drift_rows.append([
            name, f"{row['n']:,}", f"{row['mean']:+.3f}%",
            f"{row['median']:+.3f}%", f"{row['sd']:.3f}%",
            f"{row['positive_share']:.1%}", f"{row['mean_absolute']:.3f}%",
        ])

    drift_year_rows = [
        [
            str(row["year"]), f"{row['n']:,}", f"{row['mean']:+.3f}%",
            f"{row['median']:+.3f}%", f"{row['positive_share']:.1%}",
            f"{row['mean_absolute']:.3f}%",
        ]
        for row in drift["by_year"]
    ]
    direction_rows = [
        [
            str(row["year"]), f"{row['n_test']:,}",
            f"{row['spearman']:+.3f}", f"{row['mae']:.3f}%",
            f"{row['zero_mae']:.3f}%", f"{row['sign_accuracy']:.1%}",
        ]
        for row in drift["direction_walk_forward"]
    ]
    lo, hi = bootstrap["ci90"]
    current = accuracy["overall"]["current_payoff_map"]
    primary = accuracy["overall"][PRIMARY]
    liquid_current = accuracy["spread_le_50pct"]["current_payoff_map"]
    liquid_primary = accuracy["spread_le_50pct"][PRIMARY]
    success = (
        primary["mae"] < current["mae"]
        and primary["rmse"] < current["rmse"]
        and abs(primary["bias"]) < abs(current["bias"])
        and primary["spearman"] > current["spearman"]
        and liquid_primary["mae"] < liquid_current["mae"]
        and liquid_primary["rmse"] < liquid_current["rmse"]
        and abs(liquid_primary["bias"]) < abs(liquid_current["bias"])
        and liquid_primary["spearman"] > liquid_current["spearman"]
        and hi < 0
    )
    sign_delta = (
        accuracy["overall"][PRIMARY]["mae"]
        - accuracy["overall"]["drift_symmetric_sign"]["mae"]
    )
    return [
        {
            "title": "Decision",
            "body": [
                f"**{'DRIFT-AWARE FORECAST SUPPORTED' if success else 'DRIFT-AWARE FORECAST NOT SUPPORTED'}**.",
                f"Primary minus current MAE is {fmt_pct(bootstrap['observed_mae_difference'])}; "
                f"90% earnings-week interval [{fmt_pct(lo)}, {fmt_pct(hi)}], "
                f"P(improvement) {bootstrap['p_below_zero']:.1%}.",
                f"Empirical-sign minus symmetric-sign MAE is {fmt_pct(sign_delta)}. "
                "A negative value means the measured upward drift bias adds accuracy beyond movement magnitude.",
            ],
        },
        {
            "title": "Forecast accuracy",
            "columns": [
                "forecast", "n", "MAE", "RMSE", "bias", "Spearman",
                "Pearson", "80% coverage", "MAE spread <=50%", "rho spread <=50%",
            ],
            "align": ["---"] + ["---:"] * 9,
            "rows": overall_rows,
        },
        {
            "title": "Forecast accuracy by year",
            "columns": ["year", "forecast", "n", "MAE", "bias", "Spearman"],
            "align": ["---", "---", "---:", "---:", "---:", "---:"],
            "rows": year_rows,
        },
        {
            "title": "Books implied by forecast PnL above zero",
            "note": (
                "Every row trades the exact T-14 straddle with zero commissions. "
                "Only the forecast used to approve the trade changes."
            ),
            "columns": [
                "forecast", "selected", "mean", "capital weighted", "CAGR",
                "Sharpe", "years+", "breakeven alpha",
            ],
            "align": ["---"] + ["---:"] * 7,
            "rows": book_rows,
        },
        {
            "title": "Observed T-14 to T-1 stock drift",
            "columns": [
                "slice", "n", "mean", "median", "SD", "positive", "mean absolute",
            ],
            "align": ["---"] + ["---:"] * 6,
            "rows": drift_rows,
        },
        {
            "title": "Stock drift by year",
            "columns": ["year", "n", "mean", "median", "positive", "mean absolute"],
            "align": ["---"] + ["---:"] * 5,
            "rows": drift_year_rows,
        },
        {
            "title": "Can direction be predicted at T-14?",
            "note": (
                "The same registered feature set is fit only on prior years. "
                "Zero MAE is the error from forecasting no signed move."
            ),
            "columns": ["year", "n", "Spearman", "model MAE", "zero MAE", "sign accuracy"],
            "align": ["---"] + ["---:"] * 5,
            "rows": direction_rows,
        },
        {
            "title": "Mechanism and sample funnel",
            "body": [
                f"- Corrected T-14 events: {counts['priced']:,}.",
                f"- Common model-ready events: {counts['model_ready']:,}.",
                f"- Common live-domain OOS events: {counts['evaluated']:,}.",
                f"- Current-move versus previous-event drift Spearman: {drift['persistence']['previous_event_spearman']:+.3f}.",
                f"- Current-move versus prior-four-event mean Spearman: {drift['persistence']['prior_four_mean_spearman']:+.3f}.",
                "- All factor residual vectors are ticker-group cross-fitted inside strictly prior years.",
                "- This experiment evaluates forecast construction. It does not promote a production gate.",
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    spec = lib.load_spec(HERE / "spec.yaml")
    if int(args.draws) != int(spec["primary_spec"]["draws_per_event"]):
        raise ValueError("draw count differs from the preregistered primary")

    trades = load_trades()
    full_frame = pd.read_parquet(SIGNAL_CACHE)
    for column in ("event_date", "entry_date", "exit_date", "expiry"):
        full_frame[column] = pd.to_datetime(full_frame[column])
    dataset = load_dataset(spec)
    drift = drift_diagnostics(full_frame, dataset)
    write_json(RESULTS / "drift_distribution.json", drift)
    log(
        f"Live-domain drift: n={drift['slices']['live_domain']['n']:,}, "
        f"mean={drift['slices']['live_domain']['mean']:+.3f}%, "
        f"positive={drift['slices']['live_domain']['positive_share']:.1%}"
    )

    scores, model_diagnostics = generate_scores(dataset, args.draws, args.force)
    needed = []
    for arm in ARMS:
        needed.extend([arm, f"{arm}_pwin", f"{arm}_p10", f"{arm}_p90", f"{arm}_sd"])
    complete = np.isfinite(scores[needed].to_numpy(float)).all(axis=1)
    scores = scores[
        complete & (scores["year"] >= 2020) & (scores["mcap_usd"] >= 1e9)
    ].copy()
    if scores.empty:
        raise RuntimeError("No common live-domain forecast rows")
    scores.to_parquet(RESULTS / "oos_scores.parquet", index=False)
    write_json(RESULTS / "model_diagnostics.json", model_diagnostics)

    accuracy = accuracy_bundle(scores)
    bootstrap = paired_error_bootstrap(scores)
    write_json(RESULTS / "accuracy.json", accuracy)
    write_json(RESULTS / "paired_error_bootstrap.json", bootstrap)
    log(
        f"Accuracy: current MAE {accuracy['overall']['current_payoff_map']['mae']:.2%}, "
        f"primary MAE {accuracy['overall'][PRIMARY]['mae']:.2%}, "
        f"primary rho {accuracy['overall'][PRIMARY]['spearman']:+.3f}"
    )

    common_ids = set(scores["event_id"])
    eval_trades = trades[trades["event_id"].isin(common_ids)].copy()
    if eval_trades["event_id"].nunique() != len(common_ids):
        raise RuntimeError("Forecast universe does not reconcile to replay trades")
    spy = common.load_spy_daily()
    evaluations = {}
    for arm in ARMS:
        if arm == PRIMARY:
            continue
        run_dir = HERE / "arms" / arm
        state = ForecastGate(scores, arm)
        result = evaluate(
            cell_spec(spec, arm),
            eval_trades,
            gate=state.gate(arm),
            run_dir=run_dir,
            spy_daily=spy,
            fractions=(0.02, 0.05),
            mc_paths=500,
            seed=149,
            write_report=True,
            input_files=[RESULTS / "oos_scores.parquet"],
        )
        evaluations[arm] = result
        if not args.no_ledger:
            lib.record_evaluation(run_dir, cell_spec(spec, arm), result.results)
        log(
            f"{arm}: selected {result.metrics['n']:,}, "
            f"mean {result.metrics['mean']:+.2%}, CAGR {result.metrics['cagr']:+.2%}"
        )

    counts = {
        "priced": int(trades["event_id"].nunique()),
        "model_ready": int(len(dataset)),
        "evaluated": int(len(scores)),
    }
    primary_state = ForecastGate(scores, PRIMARY)
    primary_result = evaluate(
        spec,
        eval_trades,
        gate=primary_state.gate(PRIMARY),
        run_dir=HERE,
        spy_daily=spy,
        fractions=(0.02, 0.05),
        mc_paths=1000,
        seed=149,
        write_report=True,
        input_files=[
            RESULTS / "oos_scores.parquet",
            RESULTS / "accuracy.json",
            RESULTS / "drift_distribution.json",
        ],
        extra_sections=lambda result: report_sections(
            result, evaluations, accuracy, bootstrap, drift, model_diagnostics, counts
        ),
    )
    evaluations[PRIMARY] = primary_result
    if not args.no_ledger:
        lib.record_evaluation(HERE, spec, primary_result.results)

    summary = {
        "spec_hash": lib.spec_hash(spec),
        "counts": counts,
        "accuracy": accuracy,
        "paired_error_bootstrap": bootstrap,
        "drift": drift,
        "model_diagnostics": model_diagnostics,
        "arms": {arm: evaluations[arm].results["headline"] for arm in ARMS},
        "quota_calls": 0,
    }
    write_json(RESULTS / "summary.json", summary)
    log(f"Report: {primary_result.report_path}")


if __name__ == "__main__":
    main()
