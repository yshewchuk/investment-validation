#!/usr/bin/env python3
"""EXP-158: neural selection over fixed real-price short-vol candidates."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CANDIDATES = (
    ROOT
    / "experiments/EXP-133_every_symmetric_put_structure_the_ladder"
    / "results/candidates.parquet"
)
SCORE_CACHE = RESULTS / "oos_scores.parquet"
DIAGNOSTICS = RESULTS / "fold_diagnostics.json"
SEEDS = (20260908, 20260909, 20260910)
TOP_FRACTION = 0.20
FIRST_TEST_YEAR = 2020
ANALOG_K = 25
STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))

from engine.evaluate import Gate, evaluate
from engine.features import load_panel
from experiments import common, lib

BASE = (
    "exp_pnl_sim",
    "exp_pnl_sim_select",
    "analog_mean",
    "analog_win_rate",
    "analog_p10",
    "analog_p90",
    "analog_n",
)
CATEGORIES = {
    "geometry": (
        "pred_abs_move", "pred_abs_move_sd", "width_over_forecast",
        "half_width_pct_spot", "anchor_over_spot", "n_legs", "n_admissible",
        "dte_entry",
    ),
    "history": (
        "mean_prior_abs_move", "ema12r_abs", "signed_streak",
        "mean_prior_or_implied", "mcap_log", "or_implied", "or_rvol30",
    ),
    "market": (
        "spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol5",
        "spy_vol20", "spy_vol60", "spy_vol252", "spy_vol20_rel252",
    ),
    "execution": (
        "rel_spread", "entry_cost_pct", "quote_repaired", "wide_market",
    ),
}
ARM_FEATURES = {
    "nn_pnl_analogs_top20": BASE,
    "nn_plus_geometry_top20": BASE + CATEGORIES["geometry"],
    "nn_plus_history_top20": BASE + CATEGORIES["history"],
    "nn_plus_market_top20": BASE + CATEGORIES["market"],
    "nn_plus_execution_top20": BASE + CATEGORIES["execution"],
    "nn_all_categories_top20": BASE + tuple(
        column for category in CATEGORIES.values() for column in category
    ),
}
PRIMARY = "nn_all_categories_top20"
CONTROL = "nn_pnl_analogs_top20"
ALL_ARMS = tuple(ARM_FEATURES)


def log(message: str) -> None:
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-158 {elapsed:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "arm", "event_id", "ticker", "event_date", "entry_date", "exit_date",
        "fill_alpha", "entry_cost", "exit_value", "spot_entry", "ret", "exp_pnl_sim",
        "exp_pnl_sim_select", "pred_abs_move", "pred_abs_move_sd",
        "width_over_forecast", "half_width_pct_spot", "anchor_over_spot",
        "n_legs", "n_admissible", "dte_entry", "rel_spread",
        "quote_repaired", "wide_market",
    ]
    raw = pd.read_parquet(CANDIDATES, columns=columns)
    raw = raw[raw["arm"].eq("best_all")].copy()
    for column in ("event_date", "entry_date", "exit_date"):
        raw[column] = pd.to_datetime(raw[column]).dt.normalize()
    numeric = [column for column in columns if column not in {
        "arm", "event_id", "ticker", "event_date", "entry_date", "exit_date",
    }]
    for column in numeric:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["entry_cost_pct"] = 100.0 * raw["entry_cost"] / raw["spot_entry"]
    raw["year"] = raw["event_date"].dt.year
    mid = raw[np.isclose(raw["fill_alpha"], 0.5)].copy()
    if mid["event_id"].duplicated().any():
        raise RuntimeError("best_all midpoint candidates are not one row per event")

    panel_columns = [
        "ticker", "date", *CATEGORIES["history"], *CATEGORIES["market"],
    ]
    panel = load_panel()[panel_columns].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.drop_duplicates(["ticker", "date"], keep="last")
    mid = mid.merge(
        panel, left_on=["ticker", "event_date"], right_on=["ticker", "date"],
        how="left", validate="one_to_one",
    ).drop(columns="date")
    for column in CATEGORIES["history"] + CATEGORIES["market"]:
        mid[column] = pd.to_numeric(mid[column], errors="coerce")
    for column in ("quote_repaired", "wide_market"):
        mid[column] = mid[column].fillna(False).astype(float)
    coverage = {
        column: int(mid[column].notna().sum())
        for column in tuple(BASE[:2]) + tuple(
            column for category in CATEGORIES.values() for column in category
        )
    }
    log(
        f"Loaded {len(mid):,} midpoint events and {len(raw):,} stored price rows; "
        f"feature coverage={coverage}"
    )
    return mid.sort_values(["entry_date", "event_id"]).reset_index(drop=True), raw


def add_causal_analogs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dims = (
        "exp_pnl_sim", "width_over_forecast", "n_legs", "anchor_over_spot",
        "rel_spread",
    )
    for column in ("analog_mean", "analog_win_rate", "analog_p10", "analog_p90", "analog_n"):
        out[column] = np.nan
    values = out[list(dims)].to_numpy(dtype=float)
    rets = out["ret"].to_numpy(dtype=float)
    entry = pd.to_datetime(out["entry_date"]).to_numpy()
    exits = pd.to_datetime(out["exit_date"]).to_numpy()
    for i in range(len(out)):
        usable = (
            (exits[:i] < entry[i])
            & np.isfinite(rets[:i])
            & np.isfinite(values[:i]).all(axis=1)
            & np.isfinite(values[i]).all()
        )
        pool = values[:i][usable]
        pool_rets = rets[:i][usable]
        if len(pool) < ANALOG_K:
            continue
        scale = pool.std(axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
        center = pool.mean(axis=0)
        dist = (((pool - values[i]) / scale) ** 2).mean(axis=1)
        take = np.argpartition(dist, ANALOG_K - 1)[:ANALOG_K]
        analog = pool_rets[take]
        out.at[i, "analog_mean"] = float(analog.mean())
        out.at[i, "analog_win_rate"] = float((analog > 0).mean())
        out.at[i, "analog_p10"] = float(np.quantile(analog, 0.10))
        out.at[i, "analog_p90"] = float(np.quantile(analog, 0.90))
        out.at[i, "analog_n"] = float(len(analog))
        if i and i % 2000 == 0:
            log(f"Causal analogs: {i:,}/{len(out):,} events")
    counts = out.groupby("year")["analog_mean"].count().to_dict()
    log(f"Causal analog coverage by year: {counts}")
    return out


def fit_ensemble(X: np.ndarray, y: np.ndarray):
    models = []
    for seed in SEEDS:
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=seed,
            ),
        ).fit(X, y)
        models.append(model)
    return models


def predict(models, X: np.ndarray) -> np.ndarray:
    return np.mean([model.predict(X) for model in models], axis=0)


def generate_scores(dataset: pd.DataFrame, force: bool) -> tuple[pd.DataFrame, list[dict]]:
    if SCORE_CACHE.exists() and DIAGNOSTICS.exists() and not force:
        scores = pd.read_parquet(SCORE_CACHE)
        scores["event_date"] = pd.to_datetime(scores["event_date"])
        return scores, json.loads(DIAGNOSTICS.read_text())
    pieces = []
    diagnostics = []
    for year in range(FIRST_TEST_YEAR, int(dataset["year"].max()) + 1):
        train = dataset[dataset["year"] < year].copy()
        test = dataset[dataset["year"] == year].copy()
        piece = test[["event_id", "ticker", "event_date", "entry_date", "exit_date", "ret"]].copy()
        piece = piece.rename(columns={"ret": "realized_ret"})
        fold = {"year": year, "families": {}}
        for arm, features in ARM_FEATURES.items():
            train_ok = np.isfinite(train[list(features) + ["ret"]].to_numpy(dtype=float)).all(axis=1)
            test_ok = np.isfinite(test[list(features)].to_numpy(dtype=float)).all(axis=1)
            score = np.full(len(test), np.nan)
            fit_rows = train.loc[train_ok]
            if len(fit_rows) >= 500 and test_ok.any():
                X_train = fit_rows[list(features)].to_numpy(dtype=float)
                y_train = fit_rows["ret"].to_numpy(dtype=float)
                models = fit_ensemble(X_train, y_train)
                train_score = predict(models, X_train)
                score[test_ok] = predict(models, test.loc[test_ok, list(features)].to_numpy(dtype=float))
                cutoff = float(np.quantile(train_score, 1.0 - TOP_FRACTION))
            else:
                cutoff = float("nan")
            piece[arm] = score
            piece[f"selected_{arm}"] = np.isfinite(score) & np.isfinite(cutoff) & (score >= cutoff)
            fold["families"][arm] = {
                "features": len(features),
                "train_rows": int(len(fit_rows)),
                "scoreable_test": int(test_ok.sum()),
                "cutoff": cutoff,
                "selected": int(piece[f"selected_{arm}"].sum()),
            }
        pieces.append(piece)
        diagnostics.append(fold)
        counts = ", ".join(
            f"{arm}={fold['families'][arm]['selected']}" for arm in ALL_ARMS
        )
        log(f"Fold {year}: train={len(train):,} test={len(test):,}; {counts}")
    scores = pd.concat(pieces, ignore_index=True).sort_values(["event_date", "event_id"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SCORE_CACHE, index=False)
    write_json(DIAGNOSTICS, diagnostics)
    return scores, diagnostics


def rank_metrics(scores: pd.DataFrame) -> dict:
    result = {}
    for arm in ALL_ARMS:
        rows = scores[[arm, "realized_ret"]].dropna()
        if len(rows) < 20 or rows[arm].nunique() < 10:
            result[arm] = {"n": int(len(rows)), "spearman": None, "top_bottom": None}
            continue
        rho = spearmanr(rows[arm], rows["realized_ret"]).statistic
        decile = pd.qcut(rows[arm], 10, labels=False, duplicates="drop")
        means = rows.groupby(decile)["realized_ret"].mean()
        result[arm] = {
            "n": int(len(rows)),
            "spearman": float(rho),
            "top_bottom": float(means.iloc[-1] - means.iloc[0]),
        }
    return result


def policy_bootstrap(scores: pd.DataFrame, draws: int = 10000) -> dict:
    data = scores.copy()
    primary = data[f"selected_{PRIMARY}"].to_numpy(dtype=bool)
    control = data[f"selected_{CONTROL}"].to_numpy(dtype=bool)
    ret = data["realized_ret"].to_numpy(dtype=float)
    data["policy_delta"] = primary * ret - control * ret
    weekly = data.groupby(data["event_date"].dt.to_period("W"))["policy_delta"].mean().to_numpy(dtype=float)
    rng = np.random.default_rng(20260908)
    indices = rng.integers(0, len(weekly), size=(draws, len(weekly)))
    simulated = weekly[indices].mean(axis=1)
    return {
        "unit": "equal-weighted per-opportunity return difference, grouped by earnings week",
        "weeks": int(len(weekly)),
        "observed": float(weekly.mean()),
        "ci90": [float(v) for v in np.quantile(simulated, [0.05, 0.95])],
        "p_gt_zero": float((simulated > 0).mean()),
    }


class PrecomputedGate:
    def __init__(self, scores: pd.DataFrame, arm: str):
        indexed = scores.drop_duplicates("event_id").set_index("event_id")
        self.selected = indexed[f"selected_{arm}"].astype(bool).to_dict()
        self.arm = arm

    def fit(self, train: pd.DataFrame) -> None:
        return None

    def select(self, rows: pd.DataFrame) -> pd.Series:
        return rows["event_id"].map(self.selected).fillna(False).astype(bool)

    def gate(self) -> Gate:
        return Gate(fit=self.fit, select=self.select, name=self.arm)


def arm_spec(spec: dict, arm: str) -> dict:
    out = deepcopy(spec)
    out["primary_spec"]["primary"] = arm
    out["primary_spec"]["features"] = list(ARM_FEATURES[arm])
    if arm != PRIMARY:
        out["grid_cell"] = True
        out["promotion_target"] = None
    return out


def tail_to_debit(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["exit_value"] = 0.0
    out["ret"] = -1.0
    return out


def sections(result, evaluations: dict, ranks: dict, bootstrap: dict, diagnostics: list[dict]) -> list[dict]:
    control = evaluations[CONTROL].results["headline"]
    primary = result.results["headline"]
    ci = bootstrap["ci90"]
    checks = {
        "mean": primary["mean"] > control["mean"],
        "sharpe": primary["sharpe_trade"] > control["sharpe_trade"],
        "positive_year_share": (
            primary["years_positive"] / max(primary["years_evaluated"], 1)
            >= control["years_positive"] / max(control["years_evaluated"], 1)
        ),
        "breakeven_alpha": (
            primary["breakeven_alpha"] is not None
            and control["breakeven_alpha"] is not None
            and primary["breakeven_alpha"] <= control["breakeven_alpha"]
        ),
        "policy_ci": ci[0] > 0,
    }
    rows = []
    for arm in ALL_ARMS:
        headline = evaluations[arm].results["headline"] if arm in evaluations else primary
        rank = ranks[arm]
        rows.append([
            arm,
            str(len(ARM_FEATURES[arm])),
            f"{headline['n']:,}",
            f"{100 * headline['mean']:+.2f}%",
            f"{headline['sharpe_trade']:.2f}",
            f"{100 * headline['cagr']:+.2f}%",
            f"{headline['years_positive']}/{headline['years_evaluated']}",
            f"{headline['breakeven_alpha']:.3f}" if headline["breakeven_alpha"] is not None else "n/a",
            f"{rank['spearman']:+.3f}" if rank["spearman"] is not None else "n/a",
            f"{100 * rank['top_bottom']:+.2f}%" if rank["top_bottom"] is not None else "n/a",
        ])
    fold_rows = []
    for fold in diagnostics:
        for arm in ALL_ARMS:
            row = fold["families"][arm]
            fold_rows.append([
                str(fold["year"]), arm, str(row["train_rows"]), str(row["scoreable_test"]),
                str(row["selected"]), f"{row['cutoff']:.4f}" if np.isfinite(row["cutoff"]) else "n/a",
            ])
    return [
        {
            "title": "Neural selector decision",
            "body": [
                f"**{'PRIMARY SUCCESS CRITERIA MET' if all(checks.values()) else 'PRIMARY DOES NOT CLEAR'}**.",
                f"Full-category policy minus PnL-plus-analogs policy: {100 * bootstrap['observed']:+.2f}% per opportunity; 90% earnings-week interval [{100 * ci[0]:+.2f}%, {100 * ci[1]:+.2f}%].",
                "The primary comparison is matched MLP against matched MLP. Every arm uses the same fixed real-quote candidates and annual training-only top-20% cutoffs.",
                "This exploratory result does not modify the dynamic short-vol entry rule or registry.",
            ],
        },
        {
            "title": "Feature-category comparison",
            "columns": ["arm", "features", "selected", "mean", "Sharpe", "CAGR", "years+", "breakeven alpha", "rho", "top-bottom"],
            "align": ["---", "---:"] + ["---:"] * 8,
            "rows": rows,
        },
        {
            "title": "Pre-registered primary checks",
            "columns": ["check", "status"],
            "align": ["---", "---"],
            "rows": [[name, "PASS" if value else "FAIL"] for name, value in checks.items()],
        },
        {
            "title": "Feature categories",
            "body": [
                "- Base: simulated expected PnL plus strictly prior-outcome nearest-neighbor analog statistics.",
                "- Geometry/forecast: move forecast, uncertainty, payoff width, anchor, legs and candidate count.",
                "- History: prior earnings moves and current quoted-volatility levels.",
                "- Market: causal SPY return, drawdown and multi-horizon volatility state.",
                "- Execution: relative spread, premium burden and quote-quality flags.",
            ],
        },
        {
            "title": "Annual neural fits",
            "columns": ["test year", "arm", "train", "scoreable", "selected", "training cutoff"],
            "align": ["---:","---"] + ["---:"] * 4,
            "rows": fold_rows,
        },
    ]


def plan(specs: dict[str, dict]) -> None:
    ledger = lib.ledger_read()
    rows = []
    for arm, spec in specs.items():
        digest = lib.spec_hash(spec)
        if not (ledger["spec_hash"] == digest).any():
            rows.append({
                "id": spec["id"], "spec_hash": digest, "date": "2026-09-08",
                "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "",
                "promoted": "False",
            })
    if rows:
        lib.ledger_append(rows)
        log(f"Registered {len(rows)} planned neural specifications")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-scores", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    spec = lib.load_spec(HERE / "spec.yaml")
    specs = {arm: arm_spec(spec, arm) for arm in ALL_ARMS}
    if not args.no_ledger:
        plan(specs)
    dataset, priced = load_dataset()
    dataset = add_causal_analogs(dataset)
    scores, diagnostics = generate_scores(dataset, args.force_scores)
    ranks = rank_metrics(scores)
    bootstrap = policy_bootstrap(scores)
    write_json(RESULTS / "ranking.json", ranks)
    write_json(RESULTS / "policy_bootstrap.json", bootstrap)

    oos_ids = set(scores["event_id"].astype(str))
    priced["event_id"] = priced["event_id"].astype(str)
    trades = priced[priced["event_id"].isin(oos_ids)].copy()
    if trades["event_id"].nunique() != len(scores):
        raise RuntimeError("priced candidates do not reconcile to OOS score events")
    spy = common.load_spy_daily()
    inputs = [CANDIDATES, ROOT / "data/features/panel.parquet", SCORE_CACHE, DIAGNOSTICS]
    evaluations = {}
    order = [arm for arm in ALL_ARMS if arm != PRIMARY] + [PRIMARY]
    for arm in order:
        run_dir = HERE if arm == PRIMARY else HERE / "arms" / arm
        log(f"Evaluating {arm}")
        if arm == PRIMARY:
            extra = lambda result: sections(result, evaluations, ranks, bootstrap, diagnostics)
        else:
            extra = lambda result, arm=arm: [{
                "title": "Neural arm construction",
                "body": [
                    f"Features ({len(ARM_FEATURES[arm])}): {', '.join(ARM_FEATURES[arm])}.",
                    "Scores are a mean across three fixed-seed scaled MLP fits, trained only on earlier annual folds.",
                ],
            }]
        result = evaluate(
            specs[arm], trades, gate=PrecomputedGate(scores, arm).gate(),
            run_dir=run_dir, spy_daily=spy, tail_shock=tail_to_debit,
            input_files=inputs, extra_sections=extra,
        )
        evaluations[arm] = result
        if not args.no_ledger:
            lib.record_evaluation(run_dir, specs[arm], result.results)
        headline = result.results["headline"]
        log(
            f"{arm}: n={headline['n']:,}, mean={headline['mean']:+.4f}, "
            f"Sharpe={headline['sharpe_trade']:.3f}, CAGR={headline['cagr']:+.4f}"
        )
    comparison = {
        "primary": PRIMARY,
        "control": CONTROL,
        "bootstrap": bootstrap,
        "ranking": ranks,
        "headline": {arm: evaluations[arm].results["headline"] for arm in ALL_ARMS},
    }
    write_json(RESULTS / "comparison.json", comparison)
    log(f"Primary report: {evaluations[PRIMARY].report_path}")


if __name__ == "__main__":
    main()
