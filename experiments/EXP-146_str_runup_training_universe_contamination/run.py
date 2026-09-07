#!/usr/bin/env python3
"""EXP-146: isolate STR-RUNUP gate training-population contamination."""
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
CURRENT_CACHE = (
    ROOT
    / "experiments/EXP-144_str_runup_t14_corrected_calendar_gate_rebaseline"
    / "results/factor_dataset.parquet"
)
LEGACY_CACHE = (
    ROOT
    / "experiments/EXP-107_str_runup_validation_registered_mid_fill"
    / "results/gate_dataset_str_runup.parquet"
)
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.data import store  # noqa: E402
from engine.evaluate import Gate, evaluate  # noqa: E402
from engine.models.registry import load_registry  # noqa: E402
from engine.models.training import gate as gate_mod  # noqa: E402
from experiments import common, lib  # noqa: E402


STRATEGY = "STR-RUNUP"
VARIANT = "e-14_x+0_target_dte=30"
PRIMARY = "above_1b_pool"
MODEL_ARMS = (
    "full_pool",
    "above_1b_pool",
    "legacy_pool",
    "mcap_experts",
    "spread_experts",
)
ALL_ARMS = ("ungated",) + MODEL_ARMS
FEATURES = tuple(gate_mod.FEATURES)
STARTED = time.monotonic()


def log(message):
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-146 {elapsed:,.0f}s] {message}", flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def load_dataset(spec):
    if not CURRENT_CACHE.exists() or not LEGACY_CACHE.exists():
        raise FileNotFoundError("EXP-107 or EXP-144 cache is missing")
    current_snapshot = json.loads(paths.SNAPSHOT_FILE.read_text()).get("snapshot")
    expected_snapshot = spec["data"]["data_snapshot"]
    if current_snapshot != expected_snapshot:
        raise RuntimeError(
            f"Tier-3 snapshot changed: {current_snapshot} versus {expected_snapshot}"
        )
    frame = pd.read_parquet(CURRENT_CACHE)
    legacy = pd.read_parquet(LEGACY_CACHE, columns=["event_id"])
    legacy_ids = set(legacy["event_id"].astype(str))
    frame["event_id"] = frame["event_id"].astype(str)
    frame["legacy_member"] = frame["event_id"].isin(legacy_ids)
    for column in ("event_date", "entry_date", "exit_date", "expiry"):
        frame[column] = pd.to_datetime(frame[column])
    numeric_columns = list(dict.fromkeys(
        list(FEATURES) + ["ret", "mcap_log", "relative_spread"]
    ))
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame[list(numeric.columns)] = numeric
    frame["mcap_usd"] = np.exp(frame["mcap_log"])
    frame["feature_complete"] = np.isfinite(
        frame[list(FEATURES) + ["ret"]].to_numpy(dtype=float)
    ).all(axis=1)
    frame["core_member"] = (
        frame["legacy_member"]
        & frame["feature_complete"]
        & (frame["mcap_usd"] >= 1e9)
        & np.isfinite(frame["relative_spread"])
    )
    log(
        f"Loaded {len(frame):,} corrected events; "
        f"{frame['legacy_member'].sum():,} legacy overlap; "
        f"{frame['core_member'].sum():,} fixed core events"
    )
    return frame


def load_trades(event_ids):
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
        & frame["event_id"].astype(str).isin(event_ids)
    ].copy()
    for column in ("event_date", "entry_date", "exit_date", "expiry"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def fit_score(train, test):
    if len(train) < 250:
        return (
            np.full(len(test), np.nan),
            np.full(len(test), np.nan),
        )
    x_train = train[list(FEATURES)].to_numpy(dtype=float)
    x_test = test[list(FEATURES)].to_numpy(dtype=float)
    y = train["ret"].to_numpy(dtype=float)
    model = gate_mod.fit(x_train, y)
    score = model.predict(x_test)
    train_score = model.predict(x_train)
    base = float((y > 0).mean())
    try:
        iso = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(train_score, (y > 0).astype(float))
        pwin = iso.predict(score)
    except ValueError:
        pwin = np.full(len(test), base)
    return np.asarray(score, dtype=float), np.asarray(pwin, dtype=float)


def mcap_bucket(frame):
    cap = frame["mcap_usd"].to_numpy(dtype=float)
    return np.where(cap < 1e10, "1b_10b", "ge_10b")


def spread_bucket(frame):
    spread = frame["relative_spread"].to_numpy(dtype=float)
    return np.where(
        spread <= 0.15, "le_15pct",
        np.where(spread <= 0.50, "15_50pct", "gt_50pct"),
    )


def expert_score(train, test, buckets, fallback_score, fallback_pwin):
    train_labels = buckets(train)
    test_labels = buckets(test)
    score = np.asarray(fallback_score, dtype=float).copy()
    pwin = np.asarray(fallback_pwin, dtype=float).copy()
    counts = {}
    for label in sorted(set(test_labels)):
        train_group = train_labels == label
        test_group = test_labels == label
        counts[label] = {
            "train": int(train_group.sum()),
            "test": int(test_group.sum()),
            "fallback": bool(train_group.sum() < 250),
        }
        if train_group.sum() < 250:
            continue
        part_score, part_pwin = fit_score(
            train.loc[train_group], test.loc[test_group]
        )
        score[test_group] = part_score
        pwin[test_group] = part_pwin
    return score, pwin, counts


def generate_scores(dataset, force=False):
    score_dir = RESULTS / "score_folds"
    score_dir.mkdir(parents=True, exist_ok=True)
    complete = dataset[dataset["feature_complete"]].copy()
    pieces = []
    diagnostics = []
    for year in range(2019, int(dataset["year"].max()) + 1):
        score_path = score_dir / f"scores_{year}.parquet"
        diag_path = score_dir / f"diagnostics_{year}.json"
        if score_path.exists() and diag_path.exists() and not force:
            piece = pd.read_parquet(score_path)
            piece["event_date"] = pd.to_datetime(piece["event_date"])
            pieces.append(piece)
            diagnostics.append(json.loads(diag_path.read_text()))
            log(f"Year {year}: loaded {len(piece):,} cached core scores")
            continue

        train_full = complete[complete["year"] < year].copy()
        train_above = train_full[train_full["mcap_usd"] >= 1e9].copy()
        train_legacy = train_above[train_above["legacy_member"]].copy()
        test = complete[
            (complete["year"] == year) & complete["core_member"]
        ].copy()
        if len(train_full) < 500 or test.empty:
            log(
                f"Year {year}: skipped, train={len(train_full):,}, "
                f"test={len(test):,}"
            )
            continue
        log(
            f"Year {year}: test {len(test):,}; training full "
            f"{len(train_full):,}, above1b {len(train_above):,}, "
            f"legacy {len(train_legacy):,}"
        )
        piece = test[[
            "event_id", "ticker", "event_date", "year", "ret",
            "mcap_usd", "relative_spread",
        ]].rename(columns={"ret": "realized_ret"}).reset_index(drop=True)
        full_score, full_pwin = fit_score(train_full, test)
        above_score, above_pwin = fit_score(train_above, test)
        legacy_score, legacy_pwin = fit_score(train_legacy, test)
        mcap_score, mcap_pwin, mcap_counts = expert_score(
            train_above, test, mcap_bucket, above_score, above_pwin
        )
        spread_train = train_above[
            np.isfinite(train_above["relative_spread"])
        ].copy()
        spread_score, spread_pwin, spread_counts = expert_score(
            spread_train, test, spread_bucket, above_score, above_pwin
        )
        values = {
            "full_pool": (full_score, full_pwin),
            "above_1b_pool": (above_score, above_pwin),
            "legacy_pool": (legacy_score, legacy_pwin),
            "mcap_experts": (mcap_score, mcap_pwin),
            "spread_experts": (spread_score, spread_pwin),
        }
        for arm, (score, pwin) in values.items():
            piece[arm] = score
            piece[f"{arm}_pwin"] = pwin
        diag = {
            "year": year,
            "max_train_year": int(train_full["year"].max()),
            "n_test_fixed_core": int(len(test)),
            "training_rows": {
                "full_pool": int(len(train_full)),
                "above_1b_pool": int(len(train_above)),
                "legacy_pool": int(len(train_legacy)),
                "mcap_experts": mcap_counts,
                "spread_experts": spread_counts,
            },
        }
        piece.to_parquet(score_path, index=False)
        write_json(diag_path, diag)
        pieces.append(piece)
        diagnostics.append(diag)
        log(f"Year {year}: complete")
    scores = pd.concat(pieces, ignore_index=True)
    scores["event_date"] = pd.to_datetime(scores["event_date"])
    return scores.sort_values(
        ["event_date", "event_id"]
    ).reset_index(drop=True), diagnostics


def apply_trailing_gate(scores, arm):
    selected = pd.Series(False, index=scores.index)
    rows = []
    dates = pd.to_datetime(scores["event_date"])
    months = dates.dt.to_period("M").dt.to_timestamp()
    for current in sorted(months.unique()):
        current = pd.Timestamp(current)
        start = current - pd.DateOffset(months=6)
        history = scores[
            (dates >= start) & (dates < current)
        ][arm].dropna()
        here = months == current
        cutoff = float(history.quantile(0.80)) if len(history) >= 100 else None
        if cutoff is not None:
            selected.loc[here] = scores.loc[here, arm] >= cutoff
        rows.append({
            "month": str(current.date()),
            "arm": arm,
            "history_n": int(len(history)),
            "cutoff": cutoff,
            "candidates": int(here.sum()),
            "selected": int(selected.loc[here].sum()),
        })
    return selected, rows


def add_decisions(scores):
    out = scores.copy()
    cutoffs = []
    for arm in MODEL_ARMS:
        selected, rows = apply_trailing_gate(out, arm)
        out[f"selected_{arm}"] = selected
        cutoffs.extend(rows)
    out["selected_ungated"] = True
    return out, cutoffs


class PrecomputedGate:
    def __init__(self, scores, arm):
        indexed = scores.drop_duplicates("event_id").set_index("event_id")
        self.selected = indexed[f"selected_{arm}"].astype(bool).to_dict()
        self.pwin = indexed[f"{arm}_pwin"].astype(float).to_dict()
        self.arm = arm

    def fit(self, train):
        return None

    def select(self, rows):
        return rows["event_id"].map(
            self.selected
        ).fillna(False).astype(bool)

    def predict_proba(self, rows):
        return rows["event_id"].map(
            self.pwin
        ).to_numpy(dtype=float)

    def gate(self):
        return Gate(
            fit=self.fit,
            select=self.select,
            predict_proba=self.predict_proba,
            name=self.arm,
        )


def rank_metrics(scores):
    output = {}
    for arm in MODEL_ARMS:
        frame = scores[[arm, "realized_ret"]].dropna().copy()
        corr = spearmanr(
            frame[arm], frame["realized_ret"], nan_policy="omit"
        )
        frame["decile"] = pd.qcut(
            frame[arm], 10, labels=False, duplicates="drop"
        ) + 1
        means = frame.groupby("decile")["realized_ret"].mean()
        output[arm] = {
            "n": int(len(frame)),
            "spearman": float(corr.statistic),
            "top_bottom_decile": float(means.iloc[-1] - means.iloc[0]),
        }
    return output


def matched_selectivity(scores):
    rows = []
    for year, group in scores.groupby("year"):
        n = int(np.floor(0.20 * len(group)))
        for arm in MODEL_ARMS:
            selected = group.nlargest(n, arm)
            rows.append({
                "year": int(year),
                "arm": arm,
                "candidates": int(len(group)),
                "selected": n,
                "mean": float(selected["realized_ret"].mean()) if n else None,
            })
    return rows


def weekly_bootstrap(scores, challenger, incumbent="full_pool", draws=10000):
    frame = scores.copy()
    frame["week"] = pd.to_datetime(
        frame["event_date"]
    ).dt.to_period("W").astype(str)
    rows = []
    for _, group in frame.groupby("week"):
        left = group.loc[
            group[f"selected_{challenger}"], "realized_ret"
        ]
        right = group.loc[
            group[f"selected_{incumbent}"], "realized_ret"
        ]
        rows.append((left.sum(), len(left), right.sum(), len(right)))
    weekly = np.asarray(rows, dtype=float)
    rng = np.random.default_rng(146)
    estimates = []
    for start in range(0, draws, 1000):
        n_draws = min(1000, draws - start)
        index = rng.integers(
            0, len(weekly), size=(n_draws, len(weekly))
        )
        sampled = weekly[index].sum(axis=1)
        ok = (sampled[:, 1] > 0) & (sampled[:, 3] > 0)
        estimates.extend((
            sampled[ok, 0] / sampled[ok, 1]
            - sampled[ok, 2] / sampled[ok, 3]
        ).tolist())
    arr = np.asarray(estimates)
    left = frame.loc[
        frame[f"selected_{challenger}"], "realized_ret"
    ]
    right = frame.loc[
        frame[f"selected_{incumbent}"], "realized_ret"
    ]
    return {
        "challenger": challenger,
        "incumbent": incumbent,
        "observed": float(left.mean() - right.mean()),
        "ci90": np.quantile(arr, [0.05, 0.95]).tolist(),
        "ci95": np.quantile(arr, [0.025, 0.975]).tolist(),
        "p_gt_zero": float((arr > 0).mean()),
        "draws": int(len(arr)),
    }


def slice_metrics(scores):
    cap = scores["mcap_usd"].to_numpy(dtype=float)
    spread = scores["relative_spread"].to_numpy(dtype=float)
    masks = {
        "mcap_1b_10b": (cap >= 1e9) & (cap < 1e10),
        "mcap_ge_10b": cap >= 1e10,
        "spread_le_15pct": spread <= 0.15,
        "spread_15_50pct": (spread > 0.15) & (spread <= 0.50),
        "spread_gt_50pct": spread > 0.50,
    }
    output = []
    for arm in ALL_ARMS:
        selected = scores[f"selected_{arm}"].astype(bool).to_numpy()
        for cohort, mask in masks.items():
            rows = scores.loc[selected & mask]
            output.append({
                "arm": arm,
                "cohort": cohort,
                "candidates": int(mask.sum()),
                "selected": int(len(rows)),
                "mean": float(rows["realized_ret"].mean()) if len(rows) else None,
                "win_rate": float((rows["realized_ret"] > 0).mean())
                if len(rows) else None,
            })
    return output


def arm_spec(spec, arm):
    if arm == PRIMARY:
        return spec
    out = dict(spec)
    out["primary_spec"] = dict(spec["primary_spec"])
    out["primary_spec"]["challenger"] = arm
    out["promotion_target"] = None
    out["grid_cell"] = True
    return out


def fmt_pct(value):
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{100 * value:+.2f}%"


def report_sections(result, evaluations, ranks, matched, policies, slices, diagnostics, counts):
    primary = result.results["headline"]
    incumbent = evaluations["full_pool"].results["headline"]
    primary_policy = policies["above_1b_pool"]
    legacy_policy = policies["legacy_pool"]
    checks = {
        "mean": primary["mean"] > incumbent["mean"],
        "capital_weighted": primary.get("dollar_weighted", -np.inf)
        > incumbent.get("dollar_weighted", -np.inf),
        "sharpe": primary["sharpe_trade"] > incumbent["sharpe_trade"],
        "positive_year_share": (
            primary["years_positive"] / max(primary["years_evaluated"], 1)
            >= incumbent["years_positive"] / max(incumbent["years_evaluated"], 1)
        ),
        "breakeven_alpha": (
            primary.get("breakeven_alpha") is not None
            and incumbent.get("breakeven_alpha") is not None
            and primary["breakeven_alpha"] <= incumbent["breakeven_alpha"]
        ),
        "policy_ci": primary_policy["ci90"][0] > 0,
        "rank_correlation": (
            ranks[PRIMARY]["spearman"] > ranks["full_pool"]["spearman"]
        ),
        "decile_spread": (
            ranks[PRIMARY]["top_bottom_decile"]
            > ranks["full_pool"]["top_bottom_decile"]
        ),
    }
    promoted = all(checks.values())
    arm_rows = []
    for arm in ALL_ARMS:
        metrics = (
            primary if arm == PRIMARY
            else evaluations[arm].results["headline"]
        )
        arm_rows.append([
            arm, f"{metrics['n']:,}", fmt_pct(metrics["mean"]),
            fmt_pct(metrics.get("dollar_weighted")),
            fmt_pct(metrics.get("cagr")),
            f"{metrics['sharpe_trade']:.2f}",
            f"{metrics['years_positive']}/{metrics['years_evaluated']}",
            f"{metrics['breakeven_alpha']:.3f}"
            if metrics.get("breakeven_alpha") is not None else "n/a",
        ])
    rank_rows = [[
        arm, f"{row['n']:,}", f"{row['spearman']:+.3f}",
        fmt_pct(row["top_bottom_decile"]),
    ] for arm, row in ranks.items()]
    matched_rows = [[
        str(row["year"]), row["arm"], f"{row['selected']:,}",
        fmt_pct(row["mean"]),
    ] for row in matched]
    slice_rows = [[
        row["arm"], row["cohort"], f"{row['candidates']:,}",
        f"{row['selected']:,}", fmt_pct(row["mean"]),
        fmt_pct(row["win_rate"]),
    ] for row in slices]
    training_rows = []
    for diag in diagnostics:
        training = diag["training_rows"]
        training_rows.append([
            str(diag["year"]), f"{diag['n_test_fixed_core']:,}",
            f"{training['full_pool']:,}",
            f"{training['above_1b_pool']:,}",
            f"{training['legacy_pool']:,}",
        ])
    lo90, hi90 = primary_policy["ci90"]
    legacy_lo, legacy_hi = legacy_policy["ci90"]
    return [
        {
            "title": "Training-universe decision",
            "body": [
                f"**{'PRIMARY SUCCESS CRITERIA MET' if promoted else 'PRIMARY DOES NOT CLEAR'}**.",
                f"Removing sub-$1B training rows changes policy value by "
                f"{fmt_pct(primary_policy['observed'])}; 90% earnings-week "
                f"interval [{fmt_pct(lo90)}, {fmt_pct(hi90)}], "
                f"P(greater than zero) {primary_policy['p_gt_zero']:.1%}.",
                f"Returning to legacy-cohort training changes policy value by "
                f"{fmt_pct(legacy_policy['observed'])}; 90% interval "
                f"[{fmt_pct(legacy_lo)}, {fmt_pct(legacy_hi)}], "
                f"P(greater than zero) {legacy_policy['p_gt_zero']:.1%}.",
            ],
        },
        {
            "title": "One fixed test cohort, six policies",
            "note": "Only the training population or expert partition changes.",
            "columns": [
                "policy", "selected", "mean", "capital weighted",
                "CAGR", "Sharpe", "years+", "breakeven alpha",
            ],
            "align": ["---"] + ["---:"] * 7,
            "rows": arm_rows,
        },
        {
            "title": "Pre-registered primary checks",
            "columns": ["check", "status"],
            "align": ["---", "---"],
            "rows": [
                [name, "PASS" if passed else "FAIL"]
                for name, passed in checks.items()
            ],
        },
        {
            "title": "Ranking quality on the identical cohort",
            "columns": ["model", "scoreable", "Spearman", "top-bottom decile"],
            "align": ["---", "---:", "---:", "---:"],
            "rows": rank_rows,
        },
        {
            "title": "Exact annual top-20-percent diagnostic",
            "note": "Every model selects the same count from the same test rows. This full-year diagnostic cannot trigger promotion.",
            "columns": ["year", "model", "selected", "mean"],
            "align": ["---", "---", "---:", "---:"],
            "rows": matched_rows,
        },
        {
            "title": "Training populations by fold",
            "columns": [
                "test year", "fixed test", "full train",
                "above-$1B train", "legacy train",
            ],
            "align": ["---:", "---:", "---:", "---:", "---:"],
            "rows": training_rows,
        },
        {
            "title": "Market-cap and spread slices",
            "columns": [
                "policy", "cohort", "candidates", "selected", "mean", "win rate",
            ],
            "align": ["---", "---", "---:", "---:", "---:", "---:"],
            "rows": slice_rows,
        },
        {
            "title": "Sample funnel",
            "body": [
                f"- Corrected T-14 events: {counts['corrected']:,}.",
                f"- Events overlapping EXP-107: {counts['legacy_overlap']:,}.",
                f"- Fixed core events across all years: {counts['core_all']:,}.",
                f"- Fixed OOS 2020-2026 test events: {counts['core_oos']:,}.",
                "- Every modeled arm scores every fixed test event.",
                "- The 2019 fold supplies causal threshold history and is excluded from headline returns.",
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    spec = lib.load_spec(HERE / "spec.yaml")
    registry = load_registry(missing_ok=False)
    incumbent = registry.champion("gate", STRATEGY)
    if incumbent.id != "gate_midfill_str_runup":
        raise RuntimeError("registered STR-RUNUP gate changed")
    if tuple(incumbent.features) != FEATURES:
        raise RuntimeError("registered feature contract changed")

    dataset = load_dataset(spec)
    scores, diagnostics = generate_scores(dataset, force=args.force)
    scores, cutoffs = add_decisions(scores)
    oos = scores[scores["year"] >= 2020].copy()
    if not np.isfinite(
        oos[list(MODEL_ARMS)].to_numpy(dtype=float)
    ).all():
        raise RuntimeError("a model failed to score the fixed test cohort")
    oos.to_parquet(RESULTS / "oos_scores.parquet", index=False)
    pd.DataFrame(cutoffs).to_csv(
        RESULTS / "trailing_cutoffs.csv", index=False
    )
    write_json(RESULTS / "fold_diagnostics.json", diagnostics)

    ranks = rank_metrics(oos)
    matched = matched_selectivity(oos)
    policies = {
        arm: weekly_bootstrap(oos, arm)
        for arm in ("above_1b_pool", "legacy_pool")
    }
    slices = slice_metrics(oos)
    write_json(RESULTS / "ranking.json", ranks)
    write_json(RESULTS / "matched_selectivity.json", matched)
    write_json(RESULTS / "policy_bootstrap.json", policies)
    write_json(RESULTS / "slice_metrics.json", slices)

    event_ids = set(oos["event_id"])
    trades = load_trades(event_ids)
    if trades["event_id"].nunique() != len(oos):
        raise RuntimeError("fixed test scores do not reconcile to priced trades")
    counts = {
        "corrected": int(len(dataset)),
        "legacy_overlap": int(dataset["legacy_member"].sum()),
        "core_all": int(dataset["core_member"].sum()),
        "core_oos": int(len(oos)),
    }
    log(f"Evaluating {len(oos):,} fixed OOS events and {len(trades):,} fill rows")
    spy = common.load_spy_daily()
    evaluations = {}
    ungated = evaluate(
        arm_spec(spec, "ungated"), trades,
        run_dir=HERE / "arms/ungated", spy_daily=spy,
        fractions=(0.02, 0.05), mc_paths=500, seed=146,
        write_report=True, input_files=[RESULTS / "oos_scores.parquet"],
    )
    evaluations["ungated"] = ungated
    if not args.no_ledger:
        lib.record_evaluation(
            HERE / "arms/ungated", arm_spec(spec, "ungated"),
            ungated.results,
        )
    log(f"ungated: n={ungated.metrics['n']:,}, mean={ungated.metrics['mean']:+.2%}")

    for arm in MODEL_ARMS:
        if arm == PRIMARY:
            continue
        run_dir = HERE / "arms" / arm
        result = evaluate(
            arm_spec(spec, arm), trades,
            gate=PrecomputedGate(oos, arm).gate(), run_dir=run_dir,
            spy_daily=spy, fractions=(0.02, 0.05), mc_paths=500,
            seed=146, write_report=True,
            input_files=[RESULTS / "oos_scores.parquet"],
        )
        evaluations[arm] = result
        if not args.no_ledger:
            lib.record_evaluation(
                run_dir, arm_spec(spec, arm), result.results
            )
        log(f"{arm}: n={result.metrics['n']:,}, mean={result.metrics['mean']:+.2%}")

    primary = evaluate(
        spec, trades,
        gate=PrecomputedGate(oos, PRIMARY).gate(), run_dir=HERE,
        spy_daily=spy, fractions=(0.02, 0.05), mc_paths=1000,
        seed=146, write_report=True,
        input_files=[
            RESULTS / "oos_scores.parquet",
            RESULTS / "policy_bootstrap.json",
        ],
        extra_sections=lambda result: report_sections(
            result, evaluations, ranks, matched, policies,
            slices, diagnostics, counts,
        ),
    )
    evaluations[PRIMARY] = primary
    if not args.no_ledger:
        lib.record_evaluation(HERE, spec, primary.results)
    summary = {
        "spec_hash": lib.spec_hash(spec),
        "counts": counts,
        "policy_bootstrap": policies,
        "ranking": ranks,
        "matched_selectivity": matched,
        "slices": slices,
        "arms": {
            arm: evaluations[arm].results["headline"]
            for arm in ALL_ARMS
        },
        "quota_calls": 0,
    }
    write_json(RESULTS / "summary.json", summary)
    log(f"Report: {primary.report_path}")


if __name__ == "__main__":
    main()
