#!/usr/bin/env python3
"""EXP-153: four implied-move-versus-history features as the STR-RUNUP gate."""
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
FEATURE_CACHE = RESULTS / "im_gap_features.parquet"
SCORE_CACHE = RESULTS / "oos_scores.parquet"
DIAGNOSTICS_PATH = RESULTS / "fold_diagnostics.json"
TOP_FRACTIONS = (0.10, 0.20, 0.30, 0.40)
OPERATIONAL = "incumbent_registered_threshold"
CHALLENGER_FAMILY = "im_gap"
BASE_FAMILY = "base"
PRIMARY = "im_gap_top20"
MIN_VALID_QUOTE = 1.0
IM_GAP_FEATURES = (
    "im_ratio_last_implied",
    "im_gap_last_realized",
    "im_ratio_mean_prior_implied",
    "im_gap_mean_prior_realized",
)
STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.evaluate import evaluate  # noqa: E402
from engine.features import load_panel  # noqa: E402
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
BASE_FEATURES = base.BASE_FEATURES
FAMILY_FEATURES = {
    CHALLENGER_FAMILY: IM_GAP_FEATURES,
    BASE_FAMILY: BASE_FEATURES,
}
FAMILIES = tuple(FAMILY_FEATURES)
ALL_ARMS = (OPERATIONAL,) + tuple(
    f"{family}_top{int(fraction * 100)}"
    for family in FAMILIES
    for fraction in TOP_FRACTIONS
)


def log(message: str) -> None:
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-153 {elapsed:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def arm_name(family: str, fraction: float) -> str:
    return f"{family}_top{int(fraction * 100)}"


def arm_details(arm: str) -> tuple[str, float | None]:
    if arm == OPERATIONAL:
        return BASE_FAMILY, None
    family, fraction = arm.rsplit("_top", 1)
    return family, float(fraction) / 100.0


def build_im_gap_features(spec: dict, dataset: pd.DataFrame) -> pd.DataFrame:
    snapshot = json.loads(paths.SNAPSHOT_FILE.read_text()).get("snapshot")
    expected = spec["data"]["data_snapshot"]
    if snapshot != expected:
        raise RuntimeError(f"Tier-3 snapshot changed: {snapshot} versus {expected}")
    panel = load_panel()
    panel["date"] = pd.to_datetime(panel["date"]).astype("datetime64[ns]")
    quotes = (
        panel[["ticker", "date", "or_implied", "abs_move"]]
        .drop_duplicates(["ticker", "date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    del panel
    left = dataset[
        [
            "event_id",
            "ticker",
            "entry_date",
            "im",
            "mean_prior_or_implied",
            "mean_prior_abs_move",
        ]
    ].copy()
    left["event_id"] = left["event_id"].astype(str)
    left["ticker"] = left["ticker"].astype(str)
    for column in ("im", "mean_prior_or_implied", "mean_prior_abs_move"):
        left[column] = pd.to_numeric(left[column], errors="coerce")
    left = left.sort_values("entry_date").reset_index(drop=True)
    merged = pd.merge_asof(
        left,
        quotes,
        left_on="entry_date",
        right_on="date",
        by="ticker",
        direction="backward",
        allow_exact_matches=False,
    )
    im = merged["im"]
    mean_prior_implied = merged["mean_prior_or_implied"]
    mean_prior_realized = merged["mean_prior_abs_move"]
    im_last = pd.to_numeric(merged["or_implied"], errors="coerce")
    last_realized = pd.to_numeric(merged["abs_move"], errors="coerce")
    quote_ok = im >= MIN_VALID_QUOTE
    safe_last = im_last.where(im_last >= MIN_VALID_QUOTE)
    safe_mean_implied = mean_prior_implied.where(mean_prior_implied >= MIN_VALID_QUOTE)
    out = pd.DataFrame(
        {
            "event_id": left["event_id"],
            "im_ratio_last_implied": im / safe_last,
            "im_gap_last_realized": np.where(quote_ok, im - last_realized, np.nan),
            "im_ratio_mean_prior_implied": im / safe_mean_implied,
            "im_gap_mean_prior_realized": np.where(
                quote_ok, im - mean_prior_realized, np.nan
            ),
            "prior_event_found": merged["date"].notna(),
            "im_last": im_last,
            "last_realized": last_realized,
        }
    )
    coverage = {
        "events": int(len(out)),
        "im_ge_floor": int(quote_ok.sum()),
        "prior_event_found": int(out["prior_event_found"].sum()),
        "im_last_valid": int((im_last >= MIN_VALID_QUOTE).sum()),
        "mean_prior_implied_valid": int(
            (mean_prior_implied >= MIN_VALID_QUOTE).sum()
        ),
        "all_four_finite": int(
            np.isfinite(out[list(IM_GAP_FEATURES)].to_numpy(dtype=float)).all(axis=1).sum()
        ),
    }
    log(f"Im-gap features: {coverage}")
    return out, coverage


def prepare_features(
    spec: dict, dataset: pd.DataFrame, force: bool
) -> tuple[pd.DataFrame, dict]:
    if FEATURE_CACHE.exists() and not force:
        out = pd.read_parquet(FEATURE_CACHE)
        coverage = json.loads((RESULTS / "feature_coverage.json").read_text())
        log(f"Loaded im-gap feature cache: {len(out):,} events")
        return out, coverage
    out, coverage = build_im_gap_features(spec, dataset)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FEATURE_CACHE, index=False)
    write_json(RESULTS / "feature_coverage.json", coverage)
    log(f"Im-gap feature dataset written: {FEATURE_CACHE}")
    return out, coverage


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
        fold = {
            "year": int(year),
            "n_test_live_domain": int(len(test)),
            "training_rows": {},
            "scoreable_test": {},
            "cutoffs": {},
        }
        cache = {}
        for family, features in FAMILY_FEATURES.items():
            score, pwin, cutoffs, n_train, n_test = fit_family(train, test, features)
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
        base_score, base_pwin = cache[BASE_FAMILY]
        piece[OPERATIONAL] = base_score
        piece[f"{OPERATIONAL}_pwin"] = base_pwin
        piece[f"selected_{OPERATIONAL}"] = (
            np.isfinite(base_score) & (base_score >= registered_threshold)
        )
        fold["training_rows"][OPERATIONAL] = fold["training_rows"]["base_top20"]
        fold["scoreable_test"][OPERATIONAL] = fold["scoreable_test"]["base_top20"]
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
            "base score unchanged in every annual test fold."
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
    coverage: dict,
    diagnostics: list[dict],
) -> list[dict]:
    all_results = dict(evaluations)
    all_results[PRIMARY] = result
    liquid_mask = scores["relative_spread"].to_numpy(dtype=float) <= 0.50
    primary = result.results["headline"]
    base = all_results["base_top20"].results["headline"]
    incumbent = all_results[OPERATIONAL].results["headline"]
    primary_liquid = selected_slice(scores, PRIMARY, liquid_mask)
    base_liquid = selected_slice(scores, "base_top20", liquid_mask)
    bootstrap = bootstraps[0.20]
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
    matrix = []
    for arm in ALL_ARMS:
        family, fraction = arm_details(arm)
        headline = all_results[arm].results["headline"]
        liquid = selected_slice(scores, arm, liquid_mask)
        matrix.append([
            "registered" if fraction is None else f"top {fraction:.0%}",
            family,
            f"{headline['n']:,}",
            fmt_pct(headline["mean"]),
            fmt_pct(headline.get("dollar_weighted")),
            fmt_pct(headline.get("cagr")),
            f"{headline['sharpe_trade']:.2f}",
            f"{headline['years_positive']}/{headline['years_evaluated']}",
            (
                f"{headline['breakeven_alpha']:.3f}"
                if headline.get("breakeven_alpha") is not None
                else "n/a"
            ),
            f"{liquid['n']:,}",
            fmt_pct(liquid["mean"]),
        ])
    bootstrap_rows = []
    for fraction in TOP_FRACTIONS:
        row = bootstraps[fraction]
        lo, hi = row["ci90"]
        challenger = arm_name(CHALLENGER_FAMILY, fraction)
        incumbent_name = arm_name(BASE_FAMILY, fraction)
        bootstrap_rows.append([
            f"top {fraction:.0%}",
            f"{selected_slice(scores, challenger)['n']:,}",
            f"{selected_slice(scores, incumbent_name)['n']:,}",
            fmt_pct(row["observed"]),
            f"[{fmt_pct(lo)}, {fmt_pct(hi)}]",
            f"{row['p_gt_zero']:.1%}",
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
    fold_rows = []
    for row in diagnostics:
        fold_rows.append([
            str(row["year"]),
            f"{row['n_test_live_domain']:,}",
            f"{row['training_rows']['base_top20']:,}",
            f"{row['training_rows']['im_gap_top20']:,}",
            f"{row['scoreable_test']['im_gap_top20']:,}",
            f"{row['cutoffs']['base_top20']:.4f}",
            f"{row['cutoffs']['im_gap_top20']:.4f}",
        ])
    lo, hi = bootstrap["ci90"]
    return [
        {
            "title": "Four-feature gate decision",
            "body": [
                f"**{'PRIMARY SUCCESS CRITERIA MET' if all(checks.values()) else 'PRIMARY DOES NOT CLEAR'}**.",
                f"The four-feature im_gap_top20 policy minus the matched "
                f"base_top20 policy value: {fmt_pct(bootstrap['observed'])}; "
                f"90% earnings-week interval [{fmt_pct(lo)}, {fmt_pct(hi)}], "
                f"P(greater than zero) {bootstrap['p_gt_zero']:.1%}.",
                f"After excluding entry spreads above 50%: im_gap "
                f"{fmt_pct(primary_liquid['mean'])} on {primary_liquid['n']:,} "
                f"trades versus base {fmt_pct(base_liquid['mean'])} on "
                f"{base_liquid['n']:,}.",
                f"The registered operational incumbent returned "
                f"{fmt_pct(incumbent['mean'])} on {incumbent['n']:,} trades "
                f"versus im_gap_top20 {fmt_pct(primary['mean'])} on "
                f"{primary['n']:,}.",
                "This is exploratory. A passing result requires a separate "
                "confirmatory experiment before any registry change.",
            ],
        },
        {
            "title": "All score families and annual top-percent selections",
            "columns": [
                "selection", "score family", "selected", "mean", "capital weighted",
                "CAGR", "Sharpe", "years+", "breakeven alpha", "n spread<=50%", "mean spread<=50%",
            ],
            "align": ["---", "---"] + ["---:"] * 9,
            "rows": matrix,
        },
        {
            "title": "Pre-registered primary checks",
            "note": "im_gap_top20 against the matched base_top20 policy.",
            "columns": ["check", "status"],
            "align": ["---", "---"],
            "rows": [[name, "PASS" if passed else "FAIL"] for name, passed in checks.items()],
        },
        {
            "title": "Four-feature family versus matching base family",
            "note": "Earnings-week bootstrap; both sides use the same annual top fraction.",
            "columns": [
                "selection", "im_gap n", "base n", "mean difference", "90% interval", "P(greater than zero)",
            ],
            "align": ["---", "---:", "---:", "---:", "---:", "---:"],
            "rows": bootstrap_rows,
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
            "title": "Fold populations and thresholds",
            "columns": [
                "test year", "live-domain test", "base train", "im_gap train",
                "im_gap scoreable", "base top-20 cutoff", "im_gap top-20 cutoff",
            ],
            "align": ["---:"] * 7,
            "rows": fold_rows,
        },
        {
            "title": "Feature construction and coverage",
            "body": [
                f"- Frozen T-14 events: {coverage['events']:,}.",
                f"- Events with a valid entry implied quote: {coverage['im_ge_floor']:,}.",
                f"- Events matching a strictly prior panel event: {coverage['prior_event_found']:,}.",
                f"- Events with a valid last implied quote: {coverage['im_last_valid']:,}.",
                f"- Events with a valid average past implied quote: {coverage['mean_prior_implied_valid']:,}.",
                f"- Events with all four features finite: {coverage['all_four_finite']:,}.",
                "- The four features are the entry implied move over the last "
                "implied quote, the entry implied move minus the last realized "
                "move, the entry implied move over the average past implied "
                "quote, and the entry implied move minus the average past "
                "realized move.",
                "- Implied quotes below 1.0 percent are invalid; realized moves "
                "are never floored. The prior-event merge is backward as-of "
                "with exact matches disallowed.",
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
    if tuple(incumbent.features) != BASE_FEATURES or incumbent.threshold is None:
        raise RuntimeError("registered STR-RUNUP gate contract changed")

    previous.RESULTS = RESULTS
    trades_all = base.load_trades()
    dataset = previous.load_frozen_signals(spec, trades_all, False)
    features, feature_coverage = prepare_features(
        spec, dataset, args.force_scores
    )
    features["event_id"] = features["event_id"].astype(str)
    dataset["event_id"] = dataset["event_id"].astype(str)
    dataset = dataset.merge(
        features, on="event_id", how="left", suffixes=("", "_im_gap")
    )
    for column in IM_GAP_FEATURES:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    scores, diagnostics = generate_scores(
        dataset, float(incumbent.threshold), args.force_scores
    )
    ranks = rank_metrics(scores)
    bootstraps = {}
    for fraction in TOP_FRACTIONS:
        bootstraps[fraction] = base.weekly_bootstrap(
            scores,
            challenger=arm_name(CHALLENGER_FAMILY, fraction),
            incumbent=arm_name(BASE_FAMILY, fraction),
        )
    write_json(RESULTS / "ranking.json", ranks)
    write_json(RESULTS / "policy_bootstrap.json", bootstraps)

    event_ids = set(scores["event_id"].astype(str))
    trades = trades_all[trades_all["event_id"].astype(str).isin(event_ids)].copy()
    if trades["event_id"].nunique() != len(scores):
        raise RuntimeError("OOS scores do not reconcile to the priced trade set")
    spy = common.load_spy_daily()
    input_files = [
        FEATURE_CACHE,
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
                result,
                evaluations,
                scores,
                ranks,
                bootstraps,
                feature_coverage,
                diagnostics,
            )
        else:
            label = "registered threshold" if fraction is None else f"top {fraction:.0%}"
            extra = lambda result, family=family, label=label: [{
                "title": "Gate construction",
                "body": [
                    f"Score family: {family}.",
                    f"Selection policy: {label}.",
                    f"Features: {len(FAMILY_FEATURES[family])}.",
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
    comparison["feature_coverage"] = feature_coverage
    comparison["policy_bootstrap"] = bootstraps
    comparison["ranking"] = ranks
    write_json(RESULTS / "comparison.json", comparison)
    log(f"Generated report: {evaluations[PRIMARY].report_path}")


if __name__ == "__main__":
    main()
