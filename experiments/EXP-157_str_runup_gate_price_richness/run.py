#!/usr/bin/env python3
"""EXP-157: do price-richness measures improve the market-augmented gate?"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SCORE_CACHE = RESULTS / "oos_scores.parquet"
DIAGNOSTICS_PATH = RESULTS / "fold_diagnostics.json"
TOP_FRACTIONS = (0.10, 0.20)
OPERATIONAL = "incumbent_registered_threshold"
CONTROL_FAMILY = "market5"
PRIMARY_FAMILY = "market5_both"
PRIMARY = "market5_both_top20"
MARKET_FEATURES = (
    "spy_ret21", "spy_ret63", "spy_ret252", "spy_vol20", "spy_dd252",
)
RESID = "rich_resid"
SURFACE_INPUTS = ("im", "exern_iv30", "dte_entry_leg", "entry_cost_pct")
NEEDS_RESID = ("market5_rich", "market5_both")
MIN_SURFACE_ROWS = 300
STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.evaluate import evaluate  # noqa: E402
from engine.models.registry import load_registry  # noqa: E402
from experiments import common, lib  # noqa: E402


def load_runner(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load runner {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


previous = load_runner(
    "exp151_runner",
    ROOT / "experiments/EXP-151_str_runup_move_aware_gate_corrected_harness/run.py",
)
base = previous.base
FULL_FEATURES = tuple(base.BASE_FEATURES)
CONTROL_FEATURES = FULL_FEATURES + MARKET_FEATURES
FAMILY_FEATURES = {
    CONTROL_FAMILY: CONTROL_FEATURES,
    "market5_vrp": CONTROL_FEATURES + ("vrp",),
    "market5_rich": CONTROL_FEATURES + (RESID,),
    PRIMARY_FAMILY: CONTROL_FEATURES + ("vrp", RESID),
}
FAMILIES = tuple(FAMILY_FEATURES)
ALL_ARMS = (OPERATIONAL,) + tuple(
    f"{family}_top{int(fraction * 100)}"
    for family in FAMILIES
    for fraction in TOP_FRACTIONS
)


def log(message: str) -> None:
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-157 {elapsed:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def arm_name(family: str, fraction: float) -> str:
    return f"{family}_top{int(fraction * 100)}"


def arm_details(arm: str) -> tuple[str, float | None]:
    if arm == OPERATIONAL:
        return CONTROL_FAMILY, None
    family, fraction = arm.rsplit("_top", 1)
    return family, float(fraction) / 100.0


def build_market_features(dataset: pd.DataFrame) -> pd.DataFrame:
    spy = common.load_spy_daily().copy()
    spy["date"] = pd.to_datetime(spy["date"]).dt.normalize().astype("datetime64[ns]")
    spy = spy.drop_duplicates("date", keep="last").sort_values("date")
    spy["close"] = pd.to_numeric(spy["close"], errors="coerce")
    ret = spy["close"].pct_change()
    spy["spy_ret21"] = ret.rolling(21).sum()
    spy["spy_ret63"] = ret.rolling(63).sum()
    spy["spy_ret252"] = ret.rolling(252).sum()
    spy["spy_vol20"] = ret.rolling(20).std() * np.sqrt(252.0)
    spy["spy_dd252"] = spy["close"] / spy["close"].rolling(252).max() - 1.0
    features = spy[["date", *MARKET_FEATURES]].dropna(subset=MARKET_FEATURES)

    left = dataset[["event_id", "entry_date"]].drop_duplicates("event_id").copy()
    left["event_id"] = left["event_id"].astype(str)
    left["entry_date"] = pd.to_datetime(left["entry_date"]).dt.normalize().astype("datetime64[ns]")
    left = left.sort_values("entry_date").reset_index(drop=True)
    merged = pd.merge_asof(
        left, features, left_on="entry_date", right_on="date",
        direction="backward",
    )
    out = merged[["event_id", *MARKET_FEATURES]].copy()
    coverage = {
        feature: int(out[feature].notna().sum()) for feature in MARKET_FEATURES
    }
    log(f"Market state merged: {coverage}")
    return out


def surface_residuals(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list | None]:
    ok = np.isfinite(
        train[list(SURFACE_INPUTS)].to_numpy(dtype=float)
    ).all(axis=1)
    use = train.loc[ok]
    if len(use) < MIN_SURFACE_ROWS:
        log(
            f"surface fit: only {len(use):,} usable training rows; residual NaN"
        )
        return (
            np.full(len(train), np.nan),
            np.full(len(test), np.nan),
            None,
        )
    design = np.column_stack([
        np.ones(len(use)),
        use["im"].to_numpy(dtype=float),
        np.sqrt(np.maximum(use["dte_entry_leg"].to_numpy(dtype=float), 0.0))
        * use["exern_iv30"].to_numpy(dtype=float),
    ])
    beta = np.linalg.lstsq(
        design, use["entry_cost_pct"].to_numpy(dtype=float), rcond=None
    )[0]

    def residual(frame: pd.DataFrame) -> np.ndarray:
        d = np.column_stack([
            np.ones(len(frame)),
            frame["im"].to_numpy(dtype=float),
            np.sqrt(np.maximum(frame["dte_entry_leg"].to_numpy(dtype=float), 0.0))
            * frame["exern_iv30"].to_numpy(dtype=float),
        ])
        okf = np.isfinite(d).all(axis=1) & np.isfinite(
            frame["entry_cost_pct"].to_numpy(dtype=float)
        )
        out = np.full(len(frame), np.nan)
        out[okf] = frame["entry_cost_pct"].to_numpy(dtype=float)[okf] - d[okf] @ beta
        return out

    return residual(train), residual(test), beta.tolist()


def fit_family(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[float, float], int, int]:
    columns = list(features)
    train_ok = np.isfinite(
        train[columns + ["ret"]].to_numpy(dtype=float)
    ).all(axis=1)
    test_ok = np.isfinite(test[columns].to_numpy(dtype=float)).all(axis=1)
    fit_rows = train.loc[train_ok]
    score = np.full(len(test), np.nan)
    pwin = np.full(len(test), np.nan)
    cutoffs = {fraction: float("nan") for fraction in TOP_FRACTIONS}
    if len(fit_rows) < base.MIN_MODEL_ROWS or not test_ok.any():
        return score, pwin, cutoffs, int(len(fit_rows)), int(test_ok.sum())
    X_train = fit_rows[columns].to_numpy(dtype=float)
    y_train = fit_rows["ret"].to_numpy(dtype=float)
    model = base.gate_mod.fit(X_train, y_train)
    train_score = np.asarray(model.predict(X_train), dtype=float)
    score[test_ok] = model.predict(test.loc[test_ok, columns].to_numpy(dtype=float))
    cutoffs = {
        fraction: float(base.gate_mod.choose_threshold(train_score, fraction))
        for fraction in TOP_FRACTIONS
    }
    base_rate = float((y_train > 0).mean())
    try:
        isotonic = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(train_score, (y_train > 0).astype(float))
        pwin[test_ok] = isotonic.predict(score[test_ok])
    except ValueError:
        pwin[test_ok] = base_rate
    return score, pwin, cutoffs, int(len(fit_rows)), int(test_ok.sum())


def generate_scores(
    dataset: pd.DataFrame, registered_threshold: float, force: bool
) -> tuple[pd.DataFrame, list[dict]]:
    if SCORE_CACHE.exists() and DIAGNOSTICS_PATH.exists() and not force:
        scores = pd.read_parquet(SCORE_CACHE)
        scores["event_date"] = pd.to_datetime(scores["event_date"])
        diagnostics = json.loads(DIAGNOSTICS_PATH.read_text())
        log(f"Loaded score cache: {len(scores):,} OOS events")
        return scores, diagnostics

    control_complete = np.isfinite(
        dataset[list(CONTROL_FEATURES) + ["ret", "mcap_usd"]].to_numpy(dtype=float)
    ).all(axis=1)
    live_domain = dataset["mcap_usd"] >= 1e9
    pieces = []
    diagnostics = []
    for year in range(2020, int(dataset["year"].max()) + 1):
        train = dataset[dataset["year"] < year].copy()
        test = dataset[
            (dataset["year"] == year) & control_complete & live_domain
        ].copy()
        if test.empty:
            continue
        train_resid, test_resid, beta = surface_residuals(train, test)
        train_aug = train.copy()
        test_aug = test.copy()
        train_aug[RESID] = train_resid
        test_aug[RESID] = test_resid
        piece = test[
            [
                "event_id", "ticker", "event_date", "year", "ret", "mcap_usd",
                "relative_spread", "spy_vol20", "entry_cost_pct",
            ]
        ].rename(columns={"ret": "realized_ret"}).reset_index(drop=True)
        fold = {
            "year": int(year),
            "n_test_live_domain": int(len(test)),
            "training_rows": {},
            "scoreable_test": {},
            "cutoffs": {},
            "surface_beta": beta,
        }
        # The registered threshold is calibrated on the full-42 score scale, so
        # the operational arm must use that score even though the matched
        # control family for this experiment is market5.
        op_score, op_pwin, op_cutoffs, op_train, op_testable = fit_family(
            train, test, FULL_FEATURES
        )
        piece[OPERATIONAL] = op_score
        piece[f"{OPERATIONAL}_pwin"] = op_pwin
        piece[f"selected_{OPERATIONAL}"] = (
            np.isfinite(op_score) & (op_score >= registered_threshold)
        )
        fold["training_rows"][OPERATIONAL] = op_train
        fold["scoreable_test"][OPERATIONAL] = op_testable
        fold["cutoffs"][OPERATIONAL] = registered_threshold
        cache = {}
        for family, features in FAMILY_FEATURES.items():
            tr, te = (train_aug, test_aug) if family in NEEDS_RESID else (train, test)
            score, pwin, cutoffs, n_train, n_test = fit_family(tr, te, features)
            cache[family] = (score, pwin)
            for fraction in TOP_FRACTIONS:
                arm = arm_name(family, fraction)
                cutoff = cutoffs[fraction]
                piece[arm] = score
                piece[f"{arm}_pwin"] = pwin
                piece[f"selected_{arm}"] = (
                    np.isfinite(score) & np.isfinite(cutoff) & (score >= cutoff)
                )
                fold["training_rows"][arm] = n_train
                fold["scoreable_test"][arm] = n_test
                fold["cutoffs"][arm] = cutoff
        control_score, control_pwin = cache[CONTROL_FAMILY]
        diagnostics.append(fold)
        pieces.append(piece)
        counts = ", ".join(
            f"{arm}={int(piece[f'selected_{arm}'].sum())}"
            for arm in ALL_ARMS
        )
        log(f"Gate fold {year}: {len(test):,} candidates; {counts}")

    scores = pd.concat(pieces, ignore_index=True)
    scores["event_date"] = pd.to_datetime(scores["event_date"])
    scores = scores.sort_values(["event_date", "event_id"]).reset_index(drop=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SCORE_CACHE, index=False)
    write_json(DIAGNOSTICS_PATH, diagnostics)
    log(f"OOS scores written: {len(scores):,} events")
    return scores, diagnostics


def selected_slice(scores: pd.DataFrame, arm: str, mask=None) -> dict:
    selected = scores[f"selected_{arm}"].astype(bool)
    if mask is not None:
        selected &= mask
    rows = scores.loc[selected]
    return {
        "n": int(len(rows)),
        "mean": float(rows["realized_ret"].mean()) if len(rows) else None,
        "win_rate": float((rows["realized_ret"] > 0).mean()) if len(rows) else None,
    }


def rank_metrics(scores: pd.DataFrame) -> dict:
    output = {}
    common_mask = np.isfinite(scores[list(ALL_ARMS)].to_numpy(dtype=float)).all(axis=1)
    for arm in ALL_ARMS:
        own = scores[[arm, "realized_ret"]].dropna()
        common = scores.loc[common_mask, [arm, "realized_ret"]]

        def measure(frame, key=arm):
            if len(frame) < 20 or frame[key].nunique() < 10:
                return {"n": int(len(frame)), "spearman": None, "top_bottom": None}
            corr = spearmanr(frame[key], frame["realized_ret"], nan_policy="omit")
            ranked = frame.copy()
            ranked["decile"] = pd.qcut(
                ranked[key], 10, labels=False, duplicates="drop"
            ) + 1
            means = ranked.groupby("decile")["realized_ret"].mean()
            return {
                "n": int(len(ranked)),
                "spearman": float(corr.statistic),
                "top_bottom": float(means.iloc[-1] - means.iloc[0]),
            }

        output[arm] = {"own": measure(own), "common": measure(common)}
    return output


def arm_spec(spec: dict, arm: str) -> dict:
    family, fraction = arm_details(arm)
    out = deepcopy(spec)
    out["primary_spec"]["challenger"] = arm
    out["primary_spec"]["score_family"] = family
    out["primary_spec"]["top_fraction"] = fraction
    if fraction is None:
        out["primary_spec"]["gate_rule"] = (
            "The registered operational STR-RUNUP threshold is applied to the "
            "full-42-feature gate score unchanged in every annual test fold."
        )
    if arm != PRIMARY:
        out["promotion_target"] = None
        out["grid_cell"] = True
    return out


def fmt_pct(value) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{100 * value:+.2f}%"


def completed_result(run_dir: Path, spec: dict):
    stem = lib.spec_hash(spec)[:12]
    path = run_dir / "results" / f"metrics_{stem}.json"
    if not path.exists():
        return None
    results = json.loads(path.read_text())
    if "headline" not in results:
        raise RuntimeError(f"incomplete cached evaluation: {path}")
    return SimpleNamespace(results=results, report_path=run_dir / "REPORT.md")


def release_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def sweep_sections(
    result,
    evaluations: dict,
    scores: pd.DataFrame,
    ranks: dict,
    bootstraps: dict,
    diagnostics: list[dict],
) -> list[dict]:
    all_results = dict(evaluations)
    all_results[PRIMARY] = result
    liquid_mask = scores["relative_spread"].to_numpy(dtype=float) <= 0.50
    primary = result.results["headline"]
    control = all_results["market5_top20"].results["headline"]
    incumbent = all_results[OPERATIONAL].results["headline"]
    primary_liquid = selected_slice(scores, PRIMARY, liquid_mask)
    control_liquid = selected_slice(scores, "market5_top20", liquid_mask)
    bootstrap = bootstraps[0.20]
    checks = {
        "mean": primary["mean"] > control["mean"],
        "cagr": primary["cagr"] > control["cagr"],
        "sharpe": primary["sharpe_trade"] > control["sharpe_trade"],
        "positive_year_share": (
            primary["years_positive"] / max(primary["years_evaluated"], 1)
            >= control["years_positive"] / max(control["years_evaluated"], 1)
        ),
        "breakeven_alpha": (
            primary.get("breakeven_alpha") is not None
            and control.get("breakeven_alpha") is not None
            and primary["breakeven_alpha"] <= control["breakeven_alpha"]
        ),
        "spread_le_50pct": (
            primary_liquid["mean"] is not None
            and control_liquid["mean"] is not None
            and primary_liquid["mean"] > control_liquid["mean"]
        ),
        "policy_ci": bootstrap["ci90"][0] > 0,
    }
    matrix = []
    for arm in ALL_ARMS:
        family, fraction = arm_details(arm)
        headline = all_results[arm].results["headline"]
        rank = ranks[arm]["own"]
        liquid = selected_slice(scores, arm, liquid_mask)
        matrix.append([
            "registered" if fraction is None else f"top {fraction:.0%}",
            family,
            f"{len(FAMILY_FEATURES[family])}",
            f"{headline['n']:,}",
            fmt_pct(headline["mean"]),
            fmt_pct(headline.get("cagr")),
            f"{headline['sharpe_trade']:.2f}",
            f"{headline['years_positive']}/{headline['years_evaluated']}",
            (
                f"{headline['breakeven_alpha']:.3f}"
                if headline.get("breakeven_alpha") is not None
                else "n/a"
            ),
            f"{rank['spearman']:+.3f}" if rank["spearman"] is not None else "n/a",
            fmt_pct(rank["top_bottom"]),
            f"{liquid['n']:,}",
            fmt_pct(liquid["mean"]),
        ])
    bootstrap_rows = []
    for fraction in TOP_FRACTIONS:
        row = bootstraps[fraction]
        lo, hi = row["ci90"]
        challenger = arm_name(PRIMARY_FAMILY, fraction)
        control_arm = arm_name(CONTROL_FAMILY, fraction)
        bootstrap_rows.append([
            f"top {fraction:.0%}",
            f"{selected_slice(scores, challenger)['n']:,}",
            f"{selected_slice(scores, control_arm)['n']:,}",
            fmt_pct(row["observed"]),
            f"[{fmt_pct(lo)}, {fmt_pct(hi)}]",
            f"{row['p_gt_zero']:.1%}",
        ])
    premium = scores["entry_cost_pct"].to_numpy(dtype=float)
    premium_cut = float(np.nanmedian(premium))
    premium_ok = np.isfinite(premium)
    cheap_rows = []
    for arm in (PRIMARY, "market5_top20", OPERATIONAL):
        sel = scores[f"selected_{arm}"].astype(bool).to_numpy() & premium_ok
        cheap = sel & (premium <= premium_cut)
        rich = sel & (premium > premium_cut)
        cheap_rows.append([
            arm,
            f"{premium_cut:.1f}",
            f"{int(cheap.sum()):,}",
            fmt_pct(scores.loc[cheap, "realized_ret"].mean()),
            f"{int(rich.sum()):,}",
            fmt_pct(scores.loc[rich, "realized_ret"].mean()),
        ])
    fold_rows = []
    for row in diagnostics:
        beta = row["surface_beta"]
        fold_rows.append([
            str(row["year"]),
            f"{row['n_test_live_domain']:,}",
            f"{row['training_rows']['market5_top20']:,}",
            f"{row['training_rows'][PRIMARY]:,}",
            f"{row['scoreable_test'][PRIMARY]:,}",
            f"{row['cutoffs']['market5_top20']:.4f}",
            f"{row['cutoffs'][PRIMARY]:.4f}",
            f"{beta[1]:.2f}/{beta[2]:.2f}" if beta else "n/a",
        ])
    lo, hi = bootstrap["ci90"]
    return [
        {
            "title": "Price-richness gate decision",
            "body": [
                f"**{'PRIMARY SUCCESS CRITERIA MET' if all(checks.values()) else 'PRIMARY DOES NOT CLEAR'}**.",
                f"The market5_both_top20 policy minus the market5_top20 policy "
                f"value: {fmt_pct(bootstrap['observed'])}; 90% earnings-week "
                f"interval [{fmt_pct(lo)}, {fmt_pct(hi)}], "
                f"P(greater than zero) {bootstrap['p_gt_zero']:.1%}.",
                f"After excluding entry spreads above 50%: primary "
                f"{fmt_pct(primary_liquid['mean'])} on {primary_liquid['n']:,} "
                f"trades versus market5 {fmt_pct(control_liquid['mean'])} on "
                f"{control_liquid['n']:,}.",
                f"The registered operational incumbent returned "
                f"{fmt_pct(incumbent['mean'])} on {incumbent['n']:,} trades "
                f"versus primary {fmt_pct(primary['mean'])} on {primary['n']:,}.",
                "This is exploratory. A passing result requires a separate "
                "confirmatory experiment before any registry change.",
            ],
        },
        {
            "title": "All families and annual top-percent selections",
            "note": "feat = feature count; rho = own-coverage OOS score Spearman.",
            "columns": [
                "selection", "family", "feat", "selected", "mean", "CAGR",
                "Sharpe", "years+", "breakeven alpha", "rho", "top-bottom",
                "n spread<=50%", "mean spread<=50%",
            ],
            "align": ["---", "---"] + ["---:"] * 11,
            "rows": matrix,
        },
        {
            "title": "Pre-registered primary checks",
            "note": "market5_both_top20 against the matched market5_top20 policy.",
            "columns": ["check", "status"],
            "align": ["---", "---"],
            "rows": [[name, "PASS" if passed else "FAIL"] for name, passed in checks.items()],
        },
        {
            "title": "Price-augmented family versus market5 control",
            "note": "Earnings-week bootstrap; both sides use the same annual top fraction.",
            "columns": [
                "selection", "primary n", "market5 n", "mean difference", "90% interval",
                "P(greater than zero)",
            ],
            "align": ["---", "---:", "---:", "---:", "---:", "---:"],
            "rows": bootstrap_rows,
        },
        {
            "title": "Selected-trade mean by entry-premium half",
            "note": f"Split at the candidate-set median premium ({premium_cut:.1f}% of spot).",
            "columns": ["gate", "median premium", "cheap n", "cheap mean", "rich n", "rich mean"],
            "align": ["---", "---:", "---:", "---:", "---:", "---:"],
            "rows": cheap_rows,
        },
        {
            "title": "Fold populations, thresholds and surface coefficients",
            "note": "w_im/w_surf = per-fold richness-surface fit (training years only).",
            "columns": [
                "test year", "live-domain test", "market5 train", "primary train",
                "primary scoreable", "market5 top-20 cutoff", "primary top-20 cutoff",
                "w_im/w_surf",
            ],
            "align": ["---:"] * 8,
            "rows": fold_rows,
        },
        {
            "title": "Price features",
            "body": [
                "- vrp = iv30 / rvol30, both already gate inputs in level form; "
                "the ratio is new (trees cannot divide).",
                "- rich_resid = entry premium minus the premium implied by the "
                "name's own surface (intercept + w1*im + w2*sqrt(DTE)*exIV), "
                "re-fit each fold on earlier years only; test residuals use "
                "training coefficients.",
                "- The control is EXP-155's market5 gate (42 + 5 market-state "
                "features), with market state computed identically.",
                "- All trades, fills and evaluation are otherwise the "
                "registered harness.",
            ],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-scores", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    spec = lib.load_spec(HERE / "spec.yaml")
    registry = load_registry(missing_ok=False)
    incumbent = registry.champion("gate", base.STRATEGY)
    if incumbent.id != "gate_midfill_str_runup":
        raise RuntimeError(f"registered STR-RUNUP gate changed: {incumbent.id}")
    if tuple(incumbent.features) != FULL_FEATURES or incumbent.threshold is None:
        raise RuntimeError("registered STR-RUNUP gate contract changed")

    previous.RESULTS = RESULTS
    trades_all = base.load_trades()
    dataset = previous.load_frozen_signals(spec, trades_all, False)
    market = build_market_features(dataset)
    dataset["event_id"] = dataset["event_id"].astype(str)
    market["event_id"] = market["event_id"].astype(str)
    dataset = dataset.merge(market, on="event_id", how="left")
    for column in (*MARKET_FEATURES, "vrp", *SURFACE_INPUTS):
        if column in dataset.columns:
            dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    rvol = dataset["rvol30"].to_numpy(dtype=float)
    iv = dataset["iv30"].to_numpy(dtype=float)
    dataset["vrp"] = np.where(np.isfinite(iv) & np.isfinite(rvol) & (rvol > 0), iv / rvol, np.nan)
    vrp_ok = int(dataset["vrp"].notna().sum())
    log(f"vrp computed: {vrp_ok:,} of {len(dataset):,} rows")
    scores, diagnostics = generate_scores(
        dataset, float(incumbent.threshold), args.force_scores
    )
    ranks = rank_metrics(scores)
    bootstraps = {}
    for fraction in TOP_FRACTIONS:
        bootstraps[fraction] = base.weekly_bootstrap(
            scores,
            challenger=arm_name(PRIMARY_FAMILY, fraction),
            incumbent=arm_name(CONTROL_FAMILY, fraction),
        )
    write_json(RESULTS / "ranking.json", ranks)
    write_json(RESULTS / "policy_bootstrap.json", bootstraps)

    event_ids = set(scores["event_id"].astype(str))
    trades = trades_all[trades_all["event_id"].astype(str).isin(event_ids)].copy()
    if trades["event_id"].nunique() != len(scores):
        raise RuntimeError("OOS scores do not reconcile to the priced trade set")
    spy = common.load_spy_daily()
    input_files = [
        SCORE_CACHE,
        DIAGNOSTICS_PATH,
        previous.MOVE_AWARE_SIGNALS,
    ]
    input_files += sorted((paths.CURATED / "trades").glob("year=*/part-*.parquet"))

    evaluations = {}
    run_order = [arm for arm in ALL_ARMS if arm != PRIMARY] + [PRIMARY]
    for arm in run_order:
        this_spec = arm_spec(spec, arm)
        run_dir = HERE if arm == PRIMARY else HERE / "arms" / arm
        prior = completed_result(run_dir, this_spec)
        if prior is not None:
            evaluations[arm] = prior
            headline = prior.results["headline"]
            log(
                f"Resuming {arm}: n={headline['n']:,}, "
                f"mean={headline['mean']:+.4f}, Sharpe={headline['sharpe_trade']:.3f}"
            )
            continue
        log(f"Evaluating {arm}")
        gate = base.PrecomputedGate(scores, arm).gate()
        family, fraction = arm_details(arm)
        if arm == PRIMARY:
            extra = lambda result: sweep_sections(
                result, evaluations, scores, ranks, bootstraps, diagnostics
            )
        else:
            label = "registered threshold" if fraction is None else f"top {fraction:.0%}"
            extra = lambda result, family=family, label=label: [{
                "title": "Gate construction",
                "body": [
                    f"Score family: {family} ({len(FAMILY_FEATURES[family])} features).",
                    f"Selection policy: {label}.",
                    "The score fit used only earlier annual folds and the cutoff "
                    "was computed from that folds training predictions.",
                ],
            }]
        repricer = common.make_repricer(base.STRATEGY)
        try:
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
        finally:
            del repricer
            release_memory()
        evaluations[arm] = result
        if not args.no_ledger:
            lib.record_evaluation(run_dir, this_spec, result.results)
        headline = result.results["headline"]
        log(
            f"{arm}: n={headline['n']:,}, mean={headline['mean']:+.4f}, "
            f"CAGR={headline['cagr']:+.4f}, Sharpe={headline['sharpe_trade']:.3f}"
        )

    comparison = {
        arm: evaluations[arm].results["headline"] for arm in ALL_ARMS
    }
    comparison["policy_bootstrap"] = bootstraps
    comparison["ranking"] = ranks
    write_json(RESULTS / "comparison.json", comparison)
    log(f"Generated report: {evaluations[PRIMARY].report_path}")


if __name__ == "__main__":
    main()
