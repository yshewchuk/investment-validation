#!/usr/bin/env python3
"""EXP-143: native missing-value STR-RUNUP gates on the fixed T-14 trade."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FEATURE_CACHE = ROOT / "experiments/EXP-142_str_runup_t14_factor_simulation_pnl_gate/results/factor_dataset.parquet"
sys.path.insert(0, str(ROOT))

from engine.data import store  # noqa: E402
from engine.evaluate import Gate, evaluate  # noqa: E402
from engine.features import FeatureContext  # noqa: E402
from engine.models.registry import load_registry  # noqa: E402
from engine.models.training import gate as gate_mod  # noqa: E402
from experiments import common, lib  # noqa: E402


STRATEGY = "STR-RUNUP"
VARIANT = "e-14_x+0_target_dte=30"
THRESHOLD = 0.07394797616848554
PRIMARY = "native_nan_availability_proxy"
ARMS = (
    "incumbent_complete_case",
    "native_nan",
    "native_nan_availability",
    "native_nan_availability_proxy",
    "split_quote_cohort",
)
BASE_FEATURES = tuple(gate_mod.FEATURES)
IM_COLUMNS = ("im", "im_d1", "im_d5", "im_d10")
AVAILABILITY = tuple(f"has_{name}" for name in IM_COLUMNS)
PROXY_FEATURES = ("iv_implied_move_proxy", "im_filled_proxy", "im_minus_proxy")
FEATURES_BY_ARM = {
    "incumbent_complete_case": BASE_FEATURES,
    "native_nan": BASE_FEATURES,
    "native_nan_availability": BASE_FEATURES + AVAILABILITY,
    "native_nan_availability_proxy": BASE_FEATURES + AVAILABILITY + PROXY_FEATURES,
    "split_quote_cohort": BASE_FEATURES + AVAILABILITY + PROXY_FEATURES,
}
STARTED = time.monotonic()


def log(message):
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-143 {elapsed:,.0f}s] {message}", flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def load_trades():
    columns = [
        "trade_id", "kind", "strategy", "variant", "ticker", "event_id",
        "event_date", "legs", "entry_date", "exit_date", "strike", "expiry",
        "fill_alpha", "entry_cost", "exit_value", "ret", "provenance",
    ]
    frame = store.read_table("trades", columns=columns)
    frame = frame[
        (frame["strategy"] == STRATEGY)
        & (frame["variant"] == VARIANT)
        & (frame["provenance"].astype(str) == "engine.replay")
    ].copy()
    for column in ("event_date", "entry_date", "exit_date", "expiry"):
        frame[column] = pd.to_datetime(frame[column])
    log(f"Loaded {frame.event_id.nunique():,} exact T-14 events and {len(frame):,} fill rows")
    return frame


def load_gate_dataset(trades):
    if FEATURE_CACHE.exists():
        frame = pd.read_parquet(FEATURE_CACHE)
        log(f"Loaded current gate dataset from EXP-142 cache: {len(frame):,} events")
    else:
        tickers = sorted(trades["ticker"].astype(str).unique())
        log("EXP-142 cache absent; rebuilding the current gate dataset from Tier-2")
        context = FeatureContext.load(tickers=tickers, years=range(2007, 2027), with_daily=True)
        frame = gate_mod.build_dataset(trades, panel=context.panel, daily=context.daily)
    for column in ("event_date", "entry_date", "exit_date", "expiry"):
        frame[column] = pd.to_datetime(frame[column])
    missing = [name for name in BASE_FEATURES if name not in frame.columns]
    if missing:
        raise RuntimeError(f"gate dataset missing current features: {missing}")
    if frame["event_id"].duplicated().any():
        raise RuntimeError("gate dataset has duplicate event IDs")
    for name in BASE_FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for name in IM_COLUMNS:
        frame[f"has_{name}"] = np.isfinite(frame[name].to_numpy(dtype=float)).astype(float)
    iv = pd.to_numeric(frame["iv30"], errors="coerce") / 100.0
    ex = pd.to_numeric(frame["exern_iv30"], errors="coerce") / 100.0
    event_variance = np.maximum((iv**2 - ex**2) * (30.0 / 365.0), 0.0)
    frame["iv_implied_move_proxy"] = 100.0 * np.sqrt(event_variance)
    frame["im_filled_proxy"] = frame["im"].where(
        np.isfinite(frame["im"]), frame["iv_implied_move_proxy"]
    )
    frame["im_minus_proxy"] = frame["im"] - frame["iv_implied_move_proxy"]
    frame["quote_present"] = frame["has_im"].astype(bool)
    frame["im_block_complete"] = np.isfinite(
        frame[list(IM_COLUMNS)].to_numpy(dtype=float)
    ).all(axis=1)
    return frame


def calibrated_probability(model, train, features, eligible):
    y = train.loc[eligible, "ret"].to_numpy(dtype=float)
    score = model.predict(train.loc[eligible, list(features)].to_numpy(dtype=float))
    base = float((y > 0).mean())
    try:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(score, (y > 0).astype(float))
        return iso, base
    except ValueError:
        return None, base


def fit_predict(train, test, features, complete_case=False):
    y_train = pd.to_numeric(train["ret"], errors="coerce").to_numpy(dtype=float)
    train_ok = np.isfinite(y_train)
    test_ok = np.ones(len(test), dtype=bool)
    if complete_case:
        train_ok &= np.isfinite(train[list(features)].to_numpy(dtype=float)).all(axis=1)
        test_ok &= np.isfinite(test[list(features)].to_numpy(dtype=float)).all(axis=1)
    if train_ok.sum() < 250:
        return np.full(len(test), np.nan), np.full(len(test), np.nan), int(train_ok.sum())
    active_features = []
    for feature in features:
        values = train.loc[train_ok, feature].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and np.unique(finite).size >= 2:
            active_features.append(feature)
    dropped = [feature for feature in features if feature not in active_features]
    if dropped:
        log(f"Removing non-informative fit columns: {dropped}")
    if not active_features:
        score = np.full(len(test), np.nan)
        proba = np.full(len(test), np.nan)
        score[test_ok] = float(np.mean(y_train[train_ok]))
        proba[test_ok] = float((y_train[train_ok] > 0).mean())
        return score, proba, int(train_ok.sum())
    model = gate_mod.fit(
        train.loc[train_ok, active_features].to_numpy(dtype=float), y_train[train_ok]
    )
    score = np.full(len(test), np.nan)
    proba = np.full(len(test), np.nan)
    if test_ok.any():
        score[test_ok] = model.predict(test.loc[test_ok, active_features].to_numpy(dtype=float))
        iso, base = calibrated_probability(model, train, active_features, train_ok)
        proba[test_ok] = iso.predict(score[test_ok]) if iso is not None else base
    return score, proba, int(train_ok.sum())


def split_predict(train, test, fallback_score, fallback_proba):
    features = FEATURES_BY_ARM["split_quote_cohort"]
    score = np.full(len(test), np.nan)
    proba = np.full(len(test), np.nan)
    counts = {}
    for present in (False, True):
        train_group = train["quote_present"].to_numpy(dtype=bool) == present
        test_group = test["quote_present"].to_numpy(dtype=bool) == present
        label = "quote_present" if present else "quote_absent"
        counts[label] = int(train_group.sum())
        if train_group.sum() < 250:
            score[test_group] = fallback_score[test_group]
            proba[test_group] = fallback_proba[test_group]
            counts[f"{label}_fallback"] = True
            continue
        part_score, part_proba, _ = fit_predict(
            train.loc[train_group], test.loc[test_group], features, complete_case=False
        )
        score[test_group] = part_score
        proba[test_group] = part_proba
        counts[f"{label}_fallback"] = False
    return score, proba, counts


def generate_scores(dataset):
    score_dir = RESULTS / "score_folds_v2"
    score_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    diagnostics = []
    for year in range(2020, int(dataset["year"].max()) + 1):
        path = score_dir / f"scores_{year}.parquet"
        diag_path = score_dir / f"diagnostics_{year}.json"
        if path.exists() and diag_path.exists():
            piece = pd.read_parquet(path)
            piece["event_date"] = pd.to_datetime(piece["event_date"])
            pieces.append(piece)
            diagnostics.append(json.loads(diag_path.read_text()))
            log(f"Year {year}: loaded {len(piece):,} cached scores")
            continue
        train = dataset[dataset["year"] < year].copy()
        test = dataset[dataset["year"] == year].copy()
        if len(train) < 500 or test.empty:
            continue
        log(f"Year {year}: fitting {len(train):,}, scoring {len(test):,}")
        piece = test[
            ["event_id", "ticker", "event_date", "year", "ret",
             "quote_present", "im_block_complete"]
        ].rename(columns={"ret": "realized_ret"}).reset_index(drop=True)
        train_counts = {}
        for arm in ARMS[:-1]:
            complete_case = arm == "incumbent_complete_case"
            score, proba, n_train = fit_predict(
                train, test, FEATURES_BY_ARM[arm], complete_case=complete_case
            )
            piece[arm] = score
            piece[f"{arm}_pwin"] = proba
            train_counts[arm] = n_train
        fallback_score = piece[PRIMARY].to_numpy(dtype=float)
        fallback_proba = piece[f"{PRIMARY}_pwin"].to_numpy(dtype=float)
        score, proba, split_counts = split_predict(
            train, test, fallback_score, fallback_proba
        )
        piece["split_quote_cohort"] = score
        piece["split_quote_cohort_pwin"] = proba
        for arm in ARMS:
            piece[f"selected_{arm}"] = piece[arm] >= THRESHOLD
        diag = {
            "year": year,
            "max_train_year": int(train["year"].max()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_rows_by_arm": train_counts,
            "split_train": split_counts,
            "scoreable_by_arm": {
                arm: int(np.isfinite(piece[arm]).sum()) for arm in ARMS
            },
            "selected_by_arm": {
                arm: int(piece[f"selected_{arm}"].sum()) for arm in ARMS
            },
        }
        piece.to_parquet(path, index=False)
        write_json(diag_path, diag)
        pieces.append(piece)
        diagnostics.append(diag)
        log(f"Year {year}: complete")
    scores = pd.concat(pieces, ignore_index=True).sort_values(
        ["event_date", "event_id"]
    ).reset_index(drop=True)
    return scores, diagnostics


def verify_incumbent(scores, dataset, spec):
    required = spec["incumbent"]["reproduction_required"]
    base_numeric = dataset[list(BASE_FEATURES) + ["ret"]].apply(
        pd.to_numeric, errors="coerce"
    )
    gate_complete = np.isfinite(base_numeric.to_numpy(dtype=float)).all(axis=1)
    complete_rows = int(gate_complete.sum())
    oos_score = scores["incumbent_complete_case"]
    learned = gate_mod.choose_threshold(oos_score.dropna().to_numpy(dtype=float))
    selected = int(scores["selected_incumbent_complete_case"].sum())
    checks = {
        "gate_feature_complete_rows": complete_rows,
        "oos_rows": int(oos_score.notna().sum()),
        "selected": selected,
        "learned_threshold": float(learned),
        "stored_threshold": THRESHOLD,
    }
    if checks["oos_rows"] != int(required["oos_rows"]):
        raise RuntimeError(f"incumbent OOS count mismatch: {checks}")
    if selected != int(required["selected"]):
        raise RuntimeError(f"incumbent selected count mismatch: {checks}")
    if abs(float(learned) - float(required["threshold"])) > 1e-12:
        raise RuntimeError(f"incumbent threshold mismatch: {checks}")
    log(f"Incumbent reproduced exactly: {checks['oos_rows']:,} scores, {selected:,} passes")
    return checks


class PrecomputedGate:
    def __init__(self, scores, arm):
        indexed = scores.drop_duplicates("event_id").set_index("event_id")
        self.selected = indexed[f"selected_{arm}"].astype(bool).to_dict()
        self.proba = indexed[f"{arm}_pwin"].astype(float).to_dict()
        self.arm = arm

    def fit(self, train):
        return None

    def select(self, rows):
        return rows["event_id"].map(self.selected).fillna(False).astype(bool)

    def predict_proba(self, rows):
        return rows["event_id"].map(self.proba).to_numpy(dtype=float)

    def gate(self):
        return Gate(
            fit=self.fit, select=self.select, predict_proba=self.predict_proba,
            name=self.arm,
        )


def score_ranking(scores, arm):
    frame = scores[[arm, "realized_ret"]].dropna()
    corr = spearmanr(frame[arm], frame["realized_ret"], nan_policy="omit")
    frame = frame.copy()
    frame["decile"] = pd.qcut(frame[arm], 10, labels=False, duplicates="drop") + 1
    means = frame.groupby("decile")["realized_ret"].mean()
    return {
        "n": int(len(frame)),
        "spearman": float(corr.statistic),
        "top_bottom_decile": float(means.iloc[-1] - means.iloc[0]),
    }


def cohort_metrics(scores):
    output = {}
    cohorts = {
        "quote_present": scores["quote_present"].astype(bool),
        "quote_absent": ~scores["quote_present"].astype(bool),
        "im_block_complete": scores["im_block_complete"].astype(bool),
        "im_block_incomplete": ~scores["im_block_complete"].astype(bool),
    }
    for arm in ARMS:
        output[arm] = {}
        for name, cohort in cohorts.items():
            rows = scores[cohort & scores[f"selected_{arm}"]]
            output[arm][name] = {
                "candidates": int(cohort.sum()),
                "scoreable": int(np.isfinite(scores.loc[cohort, arm]).sum()),
                "selected": int(len(rows)),
                "mean": float(rows["realized_ret"].mean()) if len(rows) else None,
                "win_rate": float((rows["realized_ret"] > 0).mean()) if len(rows) else None,
            }
    return output


def matched_selectivity(scores):
    rows = []
    for year, group in scores.groupby("year"):
        n = int(group["selected_incumbent_complete_case"].sum())
        picked = group.dropna(subset=[PRIMARY]).nlargest(n, PRIMARY) if n else group.iloc[0:0]
        rows.append({
            "year": int(year), "n": n,
            "primary_mean": float(picked["realized_ret"].mean()) if n else None,
        })
    return rows


def policy_bootstrap(scores, draws=10000):
    frame = scores.copy()
    frame["week"] = pd.to_datetime(frame["event_date"]).dt.to_period("W").astype(str)
    rows = []
    for _, group in frame.groupby("week"):
        primary = group.loc[group[f"selected_{PRIMARY}"], "realized_ret"]
        incumbent = group.loc[group["selected_incumbent_complete_case"], "realized_ret"]
        rows.append((primary.sum(), len(primary), incumbent.sum(), len(incumbent)))
    weekly = np.asarray(rows, dtype=float)
    rng = np.random.default_rng(143)
    estimates = []
    for start in range(0, int(draws), 1000):
        n = min(1000, int(draws) - start)
        index = rng.integers(0, len(weekly), size=(n, len(weekly)))
        sampled = weekly[index].sum(axis=1)
        ok = (sampled[:, 1] > 0) & (sampled[:, 3] > 0)
        estimates.extend(
            (sampled[ok, 0] / sampled[ok, 1] - sampled[ok, 2] / sampled[ok, 3]).tolist()
        )
    arr = np.asarray(estimates)
    primary = frame.loc[frame[f"selected_{PRIMARY}"], "realized_ret"]
    incumbent = frame.loc[frame["selected_incumbent_complete_case"], "realized_ret"]
    return {
        "observed": float(primary.mean() - incumbent.mean()),
        "ci90": np.quantile(arr, [0.05, 0.95]).tolist(),
        "ci95": np.quantile(arr, [0.025, 0.975]).tolist(),
        "p_gt_zero": float((arr > 0).mean()),
        "draws": int(len(arr)),
    }


def cell_spec(spec, arm):
    if arm == PRIMARY:
        return spec
    out = dict(spec)
    out["primary_spec"] = dict(spec["primary_spec"])
    out["primary_spec"]["challenger"] = arm
    out["promotion_target"] = None
    out["grid_cell"] = True
    return out


def fmt_pct(value):
    return "n/a" if value is None or not np.isfinite(value) else f"{100 * value:+.2f}%"


def report_sections(result, evaluations, rankings, cohorts, matched, policy, reproduction, counts):
    primary = result.results["headline"]
    incumbent = evaluations["incumbent_complete_case"].results["headline"]
    no_quote = cohorts[PRIMARY]["quote_absent"]
    lo90, hi90 = policy["ci90"]
    promoted = (
        primary["cagr"] > incumbent["cagr"]
        and primary["sharpe_trade"] > incumbent["sharpe_trade"]
        and primary["years_positive"] / max(primary["years_evaluated"], 1)
        >= incumbent["years_positive"] / max(incumbent["years_evaluated"], 1)
        and primary["mc"].get("0.05", {}).get("p_loss", 1.0)
        <= incumbent["mc"].get("0.05", {}).get("p_loss", 1.0)
        and lo90 > 0
        and no_quote["mean"] is not None and no_quote["mean"] > 0
        and result.results.get("calibration", {}).get("brier_skill", -np.inf) >= -0.05
    )
    arm_rows = []
    for arm in ARMS:
        metrics = primary if arm == PRIMARY else evaluations[arm].results["headline"]
        arm_rows.append([
            arm, f"{metrics['n']:,}", fmt_pct(metrics["mean"]),
            fmt_pct(metrics.get("dollar_weighted")), fmt_pct(metrics.get("cagr")),
            f"{metrics['sharpe_trade']:.2f}", f"{metrics['years_positive']}/{metrics['years_evaluated']}",
            f"{metrics['breakeven_alpha']:.3f}" if metrics.get("breakeven_alpha") is not None else "n/a",
            f"{metrics['mc'].get('0.05', {}).get('p_loss', float('nan')):.3f}",
        ])
    cohort_rows = []
    for arm in ARMS:
        for name in ("quote_present", "quote_absent"):
            row = cohorts[arm][name]
            cohort_rows.append([
                arm, name, f"{row['candidates']:,}", f"{row['scoreable']:,}",
                f"{row['selected']:,}", fmt_pct(row["mean"]), fmt_pct(row["win_rate"]),
            ])
    rank_rows = [[
        arm, f"{row['n']:,}", f"{row['spearman']:+.3f}",
        fmt_pct(row["top_bottom_decile"]),
    ] for arm, row in rankings.items()]
    matched_rows = [[
        str(row["year"]), f"{row['n']:,}", fmt_pct(row["primary_mean"])
    ] for row in matched]
    return [
        {
            "title": "Decision",
            "body": [
                f"**{'PROMOTION CRITERIA MET' if promoted else 'NO PROMOTION'}**.",
                f"Primary minus incumbent mean return: {fmt_pct(policy['observed'])}; "
                f"90% weekly interval [{fmt_pct(lo90)}, {fmt_pct(hi90)}], "
                f"P(greater than zero) {policy['p_gt_zero']:.1%}.",
                f"Primary quote-absent selections: {no_quote['selected']:,}, mean {fmt_pct(no_quote['mean'])}.",
                f"Incumbent reproduced exactly: {reproduction['oos_rows']:,} scores, "
                f"{reproduction['selected']:,} passes, threshold {reproduction['learned_threshold']:.8f}.",
            ],
        },
        {
            "title": "One T-14 trade, five missing-data policies",
            "note": "Every row uses the same option trade, zero commissions and fixed score threshold.",
            "columns": ["gate", "selected", "mean", "capital weighted", "CAGR", "Sharpe", "years+", "breakeven alpha", "MC P(loss) 5%"],
            "align": ["---"] + ["---:"] * 8,
            "rows": arm_rows,
        },
        {
            "title": "Quote-availability cohorts",
            "columns": ["gate", "cohort", "candidates", "scoreable", "selected", "mean", "win rate"],
            "align": ["---", "---"] + ["---:"] * 5,
            "rows": cohort_rows,
        },
        {
            "title": "Ranking quality",
            "columns": ["gate", "scoreable", "Spearman", "top-bottom decile"],
            "align": ["---", "---:", "---:", "---:"],
            "rows": rank_rows,
        },
        {
            "title": "Primary at incumbent-matched selectivity",
            "columns": ["year", "incumbent n", "primary mean"],
            "align": ["---", "---:", "---:"],
            "rows": matched_rows,
        },
        {
            "title": "Sample funnel",
            "body": [
                f"- Exact T-14 priced events: {counts['priced']:,}.",
                f"- OOS 2020-2026 candidates: {counts['oos_candidates']:,}.",
                f"- Candidates with a finite entry implied move: {counts['quote_present']:,}.",
                f"- Candidates without a finite entry implied move: {counts['quote_absent']:,}.",
                "- ORATS zero sentinels remain missing; no value is fabricated. The proxy is a separate feature.",
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    spec = lib.load_spec(HERE / "spec.yaml")
    registry = load_registry(missing_ok=False)
    incumbent = registry.champion("gate", STRATEGY)
    if incumbent.id != "gate_midfill_str_runup" or abs(float(incumbent.threshold) - THRESHOLD) > 1e-12:
        raise RuntimeError("registered STR-RUNUP gate changed after preregistration")
    if tuple(incumbent.features) != BASE_FEATURES:
        raise RuntimeError("registered feature contract changed after preregistration")

    trades = load_trades()
    dataset = load_gate_dataset(trades)
    scores, diagnostics = generate_scores(dataset)
    reproduction = verify_incumbent(scores, dataset, spec)
    scores.to_parquet(RESULTS / "oos_scores.parquet", index=False)
    write_json(RESULTS / "fold_diagnostics.json", diagnostics)
    write_json(RESULTS / "incumbent_reproduction.json", reproduction)

    rankings = {arm: score_ranking(scores, arm) for arm in ARMS}
    cohorts = cohort_metrics(scores)
    matched = matched_selectivity(scores)
    policy = policy_bootstrap(scores, draws=10000)
    write_json(RESULTS / "ranking.json", rankings)
    write_json(RESULTS / "cohort_metrics.json", cohorts)
    write_json(RESULTS / "matched_selectivity.json", matched)
    write_json(RESULTS / "policy_bootstrap.json", policy)

    event_ids = set(scores["event_id"])
    eval_trades = trades[trades["event_id"].isin(event_ids)].copy()
    if eval_trades["event_id"].nunique() != len(scores):
        raise RuntimeError("OOS score universe does not reconcile to priced trades")
    counts = {
        "priced": int(trades["event_id"].nunique()),
        "oos_candidates": int(len(scores)),
        "quote_present": int(scores["quote_present"].sum()),
        "quote_absent": int((~scores["quote_present"].astype(bool)).sum()),
    }
    log(f"Evaluating {len(scores):,} OOS candidates and {len(eval_trades):,} fill rows")
    spy = common.load_spy_daily()
    evaluations = {}
    for arm in ARMS:
        if arm == PRIMARY:
            continue
        run_dir = HERE / "arms" / arm
        result = evaluate(
            cell_spec(spec, arm), eval_trades,
            gate=PrecomputedGate(scores, arm).gate(), run_dir=run_dir,
            spy_daily=spy, fractions=(0.02, 0.05), mc_paths=500, seed=143,
            write_report=True, input_files=[RESULTS / "oos_scores.parquet"],
        )
        evaluations[arm] = result
        if not args.no_ledger:
            lib.record_evaluation(run_dir, cell_spec(spec, arm), result.results)
        log(f"{arm}: n={result.metrics['n']:,}, mean={result.metrics['mean']:+.2%}, CAGR={result.metrics['cagr']:+.2%}")

    primary = evaluate(
        spec, eval_trades, gate=PrecomputedGate(scores, PRIMARY).gate(), run_dir=HERE,
        spy_daily=spy, fractions=(0.02, 0.05), mc_paths=1000, seed=143,
        write_report=True,
        input_files=[RESULTS / "oos_scores.parquet", RESULTS / "incumbent_reproduction.json"],
        extra_sections=lambda result: report_sections(
            result, evaluations, rankings, cohorts, matched, policy, reproduction, counts
        ),
    )
    evaluations[PRIMARY] = primary
    if not args.no_ledger:
        lib.record_evaluation(HERE, spec, primary.results)
    summary = {
        "spec_hash": lib.spec_hash(spec),
        "counts": counts,
        "incumbent_reproduction": reproduction,
        "policy_bootstrap": policy,
        "ranking": rankings,
        "cohorts": cohorts,
        "arms": {arm: evaluations[arm].results["headline"] for arm in ARMS},
        "quota_calls": 0,
    }
    write_json(RESULTS / "summary.json", summary)
    log(f"Report: {primary.report_path}")


if __name__ == "__main__":
    main()
