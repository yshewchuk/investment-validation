#!/usr/bin/env python3
"""EXP-154: does pruning noisy features improve the STR-RUNUP gate?"""
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
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SCORE_CACHE = RESULTS / "oos_scores.parquet"
DIAGNOSTICS_PATH = RESULTS / "fold_diagnostics.json"
TOP_FRACTIONS = (0.10, 0.20)
OPERATIONAL = "incumbent_registered_threshold"
CONTROL_FAMILY = "full42"
PRIMARY_FAMILY = "nosigned_nolags"
PRIMARY = "nosigned_nolags_top20"
MIN_IMPORTANCE_ROWS = 500
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

SIGNED_HISTORY = (
    "mean_prior_move",
    "ema2_prior_move",
    "ema4_prior_move",
    "ema8_prior_move",
    "ema12_prior_move",
    "signed_streak",
)
LAG_COLUMNS = tuple(
    f"{name}_d{lag}"
    for name in ("im", "iv10", "iv30", "exern_iv30")
    for lag in (1, 5, 10)
)
FAMILY_FEATURES = {
    CONTROL_FAMILY: FULL_FEATURES,
    "nosigned": tuple(f for f in FULL_FEATURES if f not in SIGNED_HISTORY),
    "nolags": tuple(f for f in FULL_FEATURES if f not in LAG_COLUMNS),
    PRIMARY_FAMILY: tuple(
        f for f in FULL_FEATURES if f not in SIGNED_HISTORY and f not in LAG_COLUMNS
    ),
    "perm_drop10": FULL_FEATURES,
    "perm_top15": FULL_FEATURES,
}
FAMILIES = tuple(FAMILY_FEATURES)
STATIC_FAMILIES = tuple(
    f for f in FAMILIES if f not in ("perm_drop10", "perm_top15")
)
ALL_ARMS = (OPERATIONAL,) + tuple(
    f"{family}_top{int(fraction * 100)}"
    for family in FAMILIES
    for fraction in TOP_FRACTIONS
)


def log(message: str) -> None:
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-154 {elapsed:,.0f}s] {message}", flush=True)


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


def permutation_prune(
    train: pd.DataFrame, columns: list[str], year: int, kind: str
) -> list[str]:
    """Causal feature pruning: importance measured only inside the training years.

    Fits the standard gate on 80 percent of the training rows (ticker-group
    split), scores permutation importance on the held-out 20 percent, then
    returns the pruned column list. The gate itself is refit afterwards on all
    training rows with those columns, exactly like the static families.
    """
    wanted = columns + ["ret", "ticker"]
    ok = np.isfinite(
        train[[c for c in columns if c != "ticker"] + ["ret"]].to_numpy(float)
    ).all(axis=1) & train["ticker"].notna()
    use = train.loc[ok, wanted]
    if len(use) < MIN_IMPORTANCE_ROWS or use["ticker"].nunique() < 10:
        log(
            f"perm {kind} {year}: training rows {len(use):,} below "
            f"{MIN_IMPORTANCE_ROWS}; keeping all features"
        )
        return columns
    X = use[columns].to_numpy(dtype=float)
    y = use["ret"].to_numpy(dtype=float)
    groups = use["ticker"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1000 + year)
    fit_idx, val_idx = next(splitter.split(X, y, groups))
    model = base.gate_mod.fit(X[fit_idx], y[fit_idx])
    result = permutation_importance(
        model,
        X[val_idx],
        y[val_idx],
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=2000 + year,
    )
    importance = result.importances_mean
    order = np.argsort(-importance, kind="stable")
    if kind == "perm_drop10":
        keep = {columns[i] for i in order[:-10]}
    else:
        keep = {columns[i] for i in order[:15]}
    return [c for c in columns if c in keep]


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

    base_complete = np.isfinite(
        dataset[list(FULL_FEATURES) + ["ret", "mcap_usd"]].to_numpy(dtype=float)
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
        fold = {
            "year": int(year),
            "n_test_live_domain": int(len(test)),
            "training_rows": {},
            "scoreable_test": {},
            "cutoffs": {},
            "perm_features": {},
        }
        cache = {}
        for family in FAMILIES:
            if family in STATIC_FAMILIES:
                features = FAMILY_FEATURES[family]
            else:
                features = tuple(
                    permutation_prune(
                        train, list(FULL_FEATURES), year, family
                    )
                )
                fold["perm_features"][family] = list(features)
                log(
                    f"perm {family} {year}: {len(features)} features kept"
                )
            score, pwin, cutoffs, n_train, n_test = fit_family(
                train, test, features
            )
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
        piece[OPERATIONAL] = control_score
        piece[f"{OPERATIONAL}_pwin"] = control_pwin
        piece[f"selected_{OPERATIONAL}"] = (
            np.isfinite(control_score) & (control_score >= registered_threshold)
        )
        fold["training_rows"][OPERATIONAL] = fold["training_rows"][
            "full42_top20"
        ]
        fold["scoreable_test"][OPERATIONAL] = fold["scoreable_test"]["full42_top20"]
        fold["cutoffs"][OPERATIONAL] = registered_threshold
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
            "full-feature gate score unchanged in every annual test fold."
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
    control = all_results["full42_top20"].results["headline"]
    incumbent = all_results[OPERATIONAL].results["headline"]
    primary_liquid = selected_slice(scores, PRIMARY, liquid_mask)
    control_liquid = selected_slice(scores, "full42_top20", liquid_mask)
    bootstrap = bootstraps[0.20]
    primary_rank = ranks[PRIMARY]["own"]
    control_rank = ranks["full42_top20"]["own"]
    checks = {
        "score_spearman": (
            primary_rank["spearman"] is not None
            and control_rank["spearman"] is not None
            and primary_rank["spearman"] > control_rank["spearman"]
        ),
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
    fold_rows = []
    for row in diagnostics:
        fold_rows.append([
            str(row["year"]),
            f"{row['n_test_live_domain']:,}",
            f"{row['training_rows']['full42_top20']:,}",
            f"{row['training_rows'][PRIMARY]:,}",
            f"{row['scoreable_test'][PRIMARY]:,}",
            f"{row['cutoffs']['full42_top20']:.4f}",
            f"{row['cutoffs'][PRIMARY]:.4f}",
        ])
    perm_rows = []
    for row in diagnostics:
        for family in ("perm_drop10", "perm_top15"):
            kept = row["perm_features"].get(family)
            if kept:
                perm_rows.append([
                    str(row["year"]),
                    family,
                    str(len(kept)),
                    ", ".join(sorted(set(FULL_FEATURES) - set(kept)))[:80] + "…",
                ])
    lo, hi = bootstrap["ci90"]
    return [
        {
            "title": "Feature-pruning gate decision",
            "body": [
                f"**{'PRIMARY SUCCESS CRITERIA MET' if all(checks.values()) else 'PRIMARY DOES NOT CLEAR'}**.",
                f"Ranking accuracy: pruned own-coverage OOS Spearman "
                f"{primary_rank['spearman']:+.4f} versus full-42 "
                f"{control_rank['spearman']:+.4f}.",
                f"The pruned nosigned_nolags_top20 policy minus the full42_top20 "
                f"policy value: {fmt_pct(bootstrap['observed'])}; 90% "
                f"earnings-week interval [{fmt_pct(lo)}, {fmt_pct(hi)}], "
                f"P(greater than zero) {bootstrap['p_gt_zero']:.1%}.",
                f"After excluding entry spreads above 50%: pruned "
                f"{fmt_pct(primary_liquid['mean'])} on {primary_liquid['n']:,} "
                f"trades versus full-42 {fmt_pct(control_liquid['mean'])} on "
                f"{control_liquid['n']:,}.",
                f"The registered operational incumbent returned "
                f"{fmt_pct(incumbent['mean'])} on {incumbent['n']:,} trades "
                f"versus pruned {fmt_pct(primary['mean'])} on {primary['n']:,}.",
                "This is exploratory. A passing result requires a separate "
                "confirmatory experiment before any registry change.",
            ],
        },
        {
            "title": "All feature families and annual top-percent selections",
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
            "note": "nosigned_nolags_top20 against the matched full42_top20 policy.",
            "columns": ["check", "status"],
            "align": ["---", "---"],
            "rows": [[name, "PASS" if passed else "FAIL"] for name, passed in checks.items()],
        },
        {
            "title": "Pruned family versus full-42 control",
            "note": "Earnings-week bootstrap; both sides use the same annual top fraction.",
            "columns": [
                "selection", "pruned n", "full n", "mean difference", "90% interval",
                "P(greater than zero)",
            ],
            "align": ["---", "---:", "---:", "---:", "---:", "---:"],
            "rows": bootstrap_rows,
        },
        {
            "title": "Fold populations and thresholds",
            "columns": [
                "test year", "live-domain test", "full42 train", "pruned train",
                "pruned scoreable", "full42 top-20 cutoff", "pruned top-20 cutoff",
            ],
            "align": ["---:"] * 7,
            "rows": fold_rows,
        },
        {
            "title": "Permutation-pruning choices by fold",
            "note": "Dropped-from-42 columns (perm_drop10) / complement (perm_top15).",
            "columns": ["test year", "family", "kept", "dropped (truncated)"],
            "align": ["---:", "---", "---:", "---"],
            "rows": perm_rows,
        },
        {
            "title": "What each family drops",
            "body": [
                f"- nosigned drops {len(SIGNED_HISTORY)} signed-direction history "
                f"terms: {', '.join(SIGNED_HISTORY)}.",
                f"- nolags drops {len(LAG_COLUMNS)} lag-difference columns: "
                f"{', '.join(LAG_COLUMNS)}.",
                f"- nosigned_nolags drops both ({len(FAMILY_FEATURES[PRIMARY_FAMILY])} "
                f"features remain of {len(FULL_FEATURES)}).",
                "- perm families re-select features every fold from within-training "
                "permutation importance (ticker-group holdout, five repeats), so "
                "no test-year information enters the choice.",
                "- All gate fits, thresholds, trades, fills and evaluation are "
                "otherwise the registered harness.",
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
