#!/usr/bin/env python3
"""EXP-150: rerun the EXP-148 gate with move-aware forecast PnL."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE_SIGNALS = (
    ROOT
    / "experiments/EXP-148_str_runup_forecast_pnl_analog_gate"
    / "results/dataset_with_signals.parquet"
)
SIGNAL_CACHE = RESULTS / "dataset_with_move_aware_signals.parquet"
SCORE_CACHE = RESULTS / "oos_scores.parquet"
MODEL_DRAWS = 4000
MIN_MODEL_ROWS = 500
MIN_POOL = 250
STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.models.training import implied_t1, runup_move  # noqa: E402
from engine.payoff import runup_payoff_design  # noqa: E402


def load_runner(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load runner {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_runner(
    "exp148_runner",
    ROOT / "experiments/EXP-148_str_runup_forecast_pnl_analog_gate/run.py",
)


def log(message: str) -> None:
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-150 {elapsed:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def seed_for(event_id: str, year: int) -> int:
    payload = f"EXP-150|{event_id}|{year}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def bin_edges(values, quantiles) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return np.unique(np.quantile(array, quantiles)) if array.size else np.array([])


def value_bin(value: float, edges: np.ndarray) -> int:
    if not np.isfinite(value) or len(edges) < 2:
        return -1
    return int(
        np.clip(np.searchsorted(edges, value, side="right") - 1, 0, len(edges) - 2)
    )


def residual_pool(
    row: pd.Series,
    residuals: pd.DataFrame,
    mcap_edges: np.ndarray,
    im_edges: np.ndarray,
) -> tuple[np.ndarray, str]:
    mcap_bin = value_bin(float(row["mcap_log"]), mcap_edges)
    im_bin = value_bin(float(row["im"]), im_edges)
    residual_mcap = np.array(
        [value_bin(value, mcap_edges) for value in residuals["mcap_log"]]
    )
    residual_im = np.array(
        [value_bin(value, im_edges) for value in residuals["im"]]
    )
    both = np.flatnonzero((residual_mcap == mcap_bin) & (residual_im == im_bin))
    if len(both) >= MIN_POOL:
        return both, "mcap_x_im"
    mcap = np.flatnonzero(residual_mcap == mcap_bin)
    if len(mcap) >= MIN_POOL:
        return mcap, "mcap"
    return np.arange(len(residuals)), "global"


def model_ready(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["runup_abs_move_d14"] = np.expm1(
        pd.to_numeric(out["target_log1p_abs_spot"], errors="coerce")
    )
    required = list(
        dict.fromkeys(
            [
                *implied_t1.FEATURES,
                *runup_move.FEATURES,
                "im_t1_actual",
                "runup_abs_move_d14",
                "spot_entry_leg",
                "spot_exit_actual",
                "strike_leg",
                "entry_cost",
                "exit_value",
                "mcap_log",
                "im",
            ]
        )
    )
    numeric = out[required].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(float)).all(axis=1)
    valid &= numeric["spot_entry_leg"].to_numpy(float) > 0
    valid &= numeric["spot_exit_actual"].to_numpy(float) > 0
    valid &= numeric["strike_leg"].to_numpy(float) > 0
    valid &= numeric["entry_cost"].to_numpy(float) > 0
    return out.loc[valid].copy().sort_values(["event_date", "event_id"])


def crossfit(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    groups = train["ticker"].astype(str).to_numpy()
    folds = min(5, int(train["ticker"].nunique()))
    if folds < 2:
        return np.full(len(train), np.nan), np.full(len(train), np.nan)
    splitter = GroupKFold(n_splits=folds)
    im_oof = np.full(len(train), np.nan)
    move_oof = np.full(len(train), np.nan)
    im_features = list(implied_t1.FEATURES)
    move_features = list(runup_move.FEATURES)
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
            seed=150 + fold,
        )
        move_model = runup_move.fit(
            train.iloc[fit_index][move_features].to_numpy(float),
            train.iloc[fit_index]["runup_abs_move_d14"].to_numpy(float),
            seed=250 + fold,
        )
        im_oof[val_index] = im_model.predict(
            train.iloc[val_index][im_features].to_numpy(float)
        )
        move_oof[val_index] = move_model.predict(
            train.iloc[val_index][move_features].to_numpy(float)
        )
    return im_oof, move_oof


def simulate_year(
    train: pd.DataFrame,
    test: pd.DataFrame,
    year: int,
) -> tuple[pd.DataFrame, dict]:
    im_oof, move_oof = crossfit(train)
    im_features = list(implied_t1.FEATURES)
    move_features = list(runup_move.FEATURES)
    im_model = implied_t1.fit(
        train[im_features].to_numpy(float),
        train["im_t1_actual"].to_numpy(float),
        seed=150,
    )
    move_model = runup_move.fit(
        train[move_features].to_numpy(float),
        train["runup_abs_move_d14"].to_numpy(float),
        seed=250,
    )
    pred_im = np.asarray(
        im_model.predict(test[im_features].to_numpy(float)), dtype=float
    )
    pred_move = np.asarray(
        move_model.predict(test[move_features].to_numpy(float)), dtype=float
    )

    train_money = 100.0 * np.log(
        train["spot_exit_actual"].to_numpy(float)
        / train["strike_leg"].to_numpy(float)
    )
    payoff_y = (
        train["exit_value"].to_numpy(float)
        / train["spot_entry_leg"].to_numpy(float)
    )
    payoff_x = runup_payoff_design(train["im_t1_actual"], train_money)
    coefficients = np.linalg.lstsq(payoff_x, payoff_y, rcond=None)[0]
    payoff_error = payoff_y - payoff_x @ coefficients

    residuals = pd.DataFrame(
        {
            "err_im": train["im_t1_actual"].to_numpy(float) - im_oof,
            "err_move": train["runup_abs_move_d14"].to_numpy(float) - move_oof,
            "payoff_error": payoff_error,
            "mcap_log": train["mcap_log"].to_numpy(float),
            "im": train["im"].to_numpy(float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    mcap_edges = bin_edges(residuals["mcap_log"], [0.0, 1 / 3, 2 / 3, 1.0])
    im_edges = bin_edges(residuals["im"], np.linspace(0.0, 1.0, 6))
    fallback = {"mcap_x_im": 0, "mcap": 0, "global": 0}
    records = []

    for position, ((_, row), point_im, point_move) in enumerate(
        zip(test.iterrows(), pred_im, pred_move), start=1
    ):
        pool, pool_kind = residual_pool(row, residuals, mcap_edges, im_edges)
        fallback[pool_kind] += 1
        rng = np.random.default_rng(seed_for(str(row["event_id"]), year))
        chosen = rng.choice(pool, size=MODEL_DRAWS, replace=True)
        im_draw = np.maximum(
            float(point_im) + residuals["err_im"].to_numpy()[chosen], 0.0
        )
        move_draw = np.maximum(
            float(point_move) + residuals["err_move"].to_numpy()[chosen], 0.0
        )
        signed_move = rng.choice((-1.0, 1.0), size=MODEL_DRAWS) * move_draw
        exit_spot = float(row["spot_entry_leg"]) * np.exp(signed_move / 100.0)
        exit_money = 100.0 * np.log(exit_spot / float(row["strike_leg"]))
        value_per_spot = np.maximum(
            runup_payoff_design(im_draw, exit_money) @ coefficients
            + residuals["payoff_error"].to_numpy()[chosen],
            0.0,
        )
        returns = (
            value_per_spot * float(row["spot_entry_leg"])
            - float(row["entry_cost"])
        ) / float(row["entry_cost"])
        records.append(
            {
                "event_id": str(row["event_id"]),
                "pred_runup_abs_move_d14": float(point_move),
                "pred_runup_move_p10": float(np.quantile(move_draw, 0.10)),
                "pred_runup_move_p90": float(np.quantile(move_draw, 0.90)),
                "pred_runup_move_sd": float(np.std(move_draw, ddof=1)),
                "forecast_pnl_mean": float(np.mean(returns)),
                "forecast_pnl_win": float(np.mean(returns > 0)),
                "forecast_pnl_p10": float(np.quantile(returns, 0.10)),
                "forecast_pnl_p90": float(np.quantile(returns, 0.90)),
                "forecast_pnl_sd": float(np.std(returns, ddof=1)),
            }
        )
        if position % 500 == 0 or position == len(test):
            log(f"forecast {year}: simulated {position:,}/{len(test):,}")

    diagnostics = {
        "year": int(year),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "residual_pool": int(len(residuals)),
        "pool_fallback": fallback,
        "implied_t1_mae_pp": float(
            np.mean(np.abs(pred_im - test["im_t1_actual"].to_numpy(float)))
        ),
        "runup_move_mae_pp": float(
            np.mean(
                np.abs(pred_move - test["runup_abs_move_d14"].to_numpy(float))
            )
        ),
        "payoff_coefficients": coefficients.tolist(),
    }
    return pd.DataFrame(records), diagnostics


def prepare_signals(spec: dict, trades: pd.DataFrame, force: bool) -> pd.DataFrame:
    del trades
    if SIGNAL_CACHE.exists() and not force:
        frame = pd.read_parquet(SIGNAL_CACHE)
        for column in ("event_date", "entry_date", "exit_date", "expiry"):
            frame[column] = pd.to_datetime(frame[column])
        log(f"Loaded move-aware signal cache: {len(frame):,} events")
        return frame

    snapshot = json.loads(paths.SNAPSHOT_FILE.read_text()).get("snapshot")
    expected = spec["data"]["data_snapshot"]
    if snapshot != expected:
        raise RuntimeError(f"Tier-3 snapshot changed: {snapshot} versus {expected}")
    source = pd.read_parquet(SOURCE_SIGNALS)
    for column in ("event_date", "entry_date", "exit_date", "expiry"):
        source[column] = pd.to_datetime(source[column])
    dataset = model_ready(source).reset_index(drop=True)
    output = source.copy()
    for column in base.FORECAST_PNL:
        output[column] = np.nan
    for column in (
        "pred_runup_abs_move_d14",
        "pred_runup_move_p10",
        "pred_runup_move_p90",
        "pred_runup_move_sd",
    ):
        output[column] = np.nan

    diagnostics = []
    for year in range(2019, int(dataset["year"].max()) + 1):
        train = dataset[dataset["year"] < year].copy().reset_index(drop=True)
        test = dataset[dataset["year"] == year].copy().reset_index(drop=True)
        if len(train) < MIN_MODEL_ROWS or test.empty:
            log(f"forecast {year}: skipped, train={len(train):,}, test={len(test):,}")
            continue
        log(f"forecast {year}: fitting {len(train):,}, scoring {len(test):,}")
        signals, diagnostic = simulate_year(train, test, year)
        diagnostics.append(diagnostic)
        indexed = signals.set_index("event_id")
        event_ids = output["event_id"].astype(str)
        for column in signals.columns:
            if column == "event_id":
                continue
            output.loc[event_ids.isin(indexed.index), column] = (
                event_ids[event_ids.isin(indexed.index)].map(indexed[column]).to_numpy()
            )
        log(
            f"forecast {year}: PnL coverage "
            f"{signals['forecast_pnl_mean'].notna().sum():,}/{len(test):,}"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    output.to_parquet(SIGNAL_CACHE, index=False)
    write_json(RESULTS / "forecast_diagnostics.json", diagnostics)
    log(f"Move-aware signal dataset written: {SIGNAL_CACHE}")
    return output


original_report_sections = base.report_sections


def report_sections(*args, **kwargs):
    sections = original_report_sections(*args, **kwargs)
    sections.insert(
        1,
        {
            "title": "Controlled EXP-148 rerun",
            "body": [
                "Only the five forecast-PnL features changed from EXP-148.",
                "The new simulator jointly draws causal implied-move error, "
                "absolute stock-move error and payoff-surface error, then gives "
                "the stock move an independent symmetric sign.",
                "The T-14 trade, raw-forecast controls, analog evidence, gate "
                "features, annual thresholds, fills and evaluation remain fixed.",
                "No ORATS or Polygon requests are made by this experiment.",
            ],
        },
    )
    return sections


base.HERE = HERE
base.RESULTS = RESULTS
base.BASE_CACHE = SOURCE_SIGNALS
base.SIGNAL_CACHE = SIGNAL_CACHE
base.SCORE_CACHE = SCORE_CACHE
base.STARTED = STARTED
base.MODEL_DRAWS = MODEL_DRAWS
base.log = log
base.prepare_signals = prepare_signals
base.report_sections = report_sections


if __name__ == "__main__":
    base.main()
