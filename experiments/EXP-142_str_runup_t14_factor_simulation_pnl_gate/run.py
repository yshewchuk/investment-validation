#!/usr/bin/env python3
"""EXP-142: T-14 factor-simulation PnL gate versus the incumbent gate."""
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
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from engine.data import store  # noqa: E402
from engine.evaluate import Gate, evaluate  # noqa: E402
from engine.features import FeatureContext  # noqa: E402
from engine.models.registry import load_registry  # noqa: E402
from engine.models.training import gate as gate_mod  # noqa: E402
from experiments import common, lib  # noqa: E402
import simulation as sim  # noqa: E402


STRATEGY = "STR-RUNUP"
VARIANT = "e-14_x+0_target_dte=30"
PRIMARY = "sim_joint"
ARMS = (
    "incumbent_direct_return",
    "sim_surface_point",
    "sim_surface_swarm",
    "sim_joint_independent",
    "sim_joint",
)
STARTED = time.monotonic()


def log(message):
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-142 {elapsed:,.0f}s] {message}", flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    log(
        f"Loaded {frame.event_id.nunique():,} exact incumbent events and "
        f"{len(frame):,} fill rows")
    return frame


def build_factor_dataset(trades):
    cache = RESULTS / "factor_dataset.parquet"
    if cache.exists():
        frame = pd.read_parquet(cache)
        for column in ("event_date", "entry_date", "exit_date", "expiry"):
            frame[column] = pd.to_datetime(frame[column])
        log(f"Factor dataset from cache: {len(frame):,} events")
        return frame

    tickers = sorted(trades["ticker"].astype(str).unique())
    log("Loading causal panel and daily market history")
    context = FeatureContext.load(
        tickers=tickers, years=range(2007, 2027), with_daily=True
    )
    log(
        f"Feature context: {len(context.panel):,} event rows and "
        f"{len(context.daily):,} daily rows")
    frame = gate_mod.build_dataset(
        trades, panel=context.panel, daily=context.daily
    )
    log(f"Current registered feature contract built: {len(frame):,} events")

    parsed = frame["legs"].apply(sim.parse_leg_state).apply(pd.Series)
    frame = pd.concat([frame.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1)
    daily = context.daily[
        ["ticker", "date", "iv30", "exern_iv30"]
    ].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.drop_duplicates(["ticker", "date"], keep="last")
    daily = daily.rename(columns={
        "date": "exit_state_date",
        "iv30": "exit_iv30",
        "exern_iv30": "exit_exern_iv30",
    })
    frame = frame.merge(
        daily,
        left_on=["ticker", "exit_date"],
        right_on=["ticker", "exit_state_date"],
        how="left",
        validate="many_to_one",
    )
    log("Inferring entry contract IV and constructing factor targets")
    frame = sim.add_targets(frame)
    frame.to_parquet(cache, index=False)
    log(f"Factor dataset written: {len(frame):,} rows")
    return frame


def complete_dataset(frame):
    required = list(gate_mod.FEATURES) + list(sim.TARGETS) + [
        "entry_cost", "exit_value", "ret", "spot_entry_leg",
        "spot_exit_actual", "strike_leg", "dte_entry_leg", "dte_exit",
        "entry_iv_basis", "exit_exern_iv30", "exit_event_var", "spot_sign",
        "mcap_log", "im",
    ]
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    ok = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    ok &= numeric["entry_cost"].to_numpy() > 0
    ok &= numeric["dte_exit"].to_numpy() > 0
    out = frame.loc[ok].copy().sort_values(["event_date", "event_id"])
    log(
        f"Common model-ready state: {len(out):,}/{len(frame):,} events "
        f"({len(out) / max(len(frame), 1):.1%})")
    return out


def direct_gate_scores(train, test):
    features = list(gate_mod.FEATURES)
    x_train = train[features].to_numpy(dtype=float)
    x_test = test[features].to_numpy(dtype=float)
    y = train["ret"].to_numpy(dtype=float)
    model = gate_mod.fit(x_train, y)
    train_score = model.predict(x_train)
    test_score = model.predict(x_test)
    base = float((y > 0).mean())
    try:
        iso = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(train_score, (y > 0).astype(float))
        proba = iso.predict(test_score)
    except ValueError:
        proba = np.full(len(test), base)
    return test_score, np.asarray(proba, dtype=float)


def registered_gate_scores(frame, expected_threshold):
    features = list(gate_mod.FEATURES)
    numeric = frame[features + ["ret"]].apply(pd.to_numeric, errors="coerce")
    complete = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    data = frame.loc[complete].copy()
    pieces = []
    for year in range(2020, int(data["year"].max()) + 1):
        train = data[data["year"] < year]
        test = data[data["year"] == year]
        if len(train) < 500 or test.empty:
            continue
        score, pwin = direct_gate_scores(train, test)
        pieces.append(pd.DataFrame({
            "event_id": test["event_id"].to_numpy(),
            "year": int(year),
            "incumbent_direct_return": score,
            "incumbent_direct_return_pwin": pwin,
        }))
    if not pieces:
        raise RuntimeError("No registered incumbent scores were produced")
    scores = pd.concat(pieces, ignore_index=True)
    learned_threshold = gate_mod.choose_threshold(
        scores["incumbent_direct_return"].to_numpy(dtype=float)
    )
    if abs(float(learned_threshold) - float(expected_threshold)) > 1e-12:
        raise RuntimeError(
            "Incumbent walk-forward does not reproduce the registered threshold: "
            f"{learned_threshold} versus {expected_threshold}"
        )
    scores["selected"] = (
        scores["incumbent_direct_return"] >= float(expected_threshold)
    )
    diagnostic = {
        "gate_feature_complete_rows": int(len(data)),
        "oos_rows": int(len(scores)),
        "learned_threshold": float(learned_threshold),
        "stored_threshold": float(expected_threshold),
        "passes_full_gate_universe": int(scores["selected"].sum()),
        "by_year": (
            scores.groupby("year")
            .agg(n=("event_id", "size"), n_passed=("selected", "sum"))
            .reset_index()
            .to_dict("records")
        ),
    }
    log(
        f"Registered incumbent reproduced: {len(scores):,} OOS rows, "
        f"threshold {learned_threshold:.8f}, {scores['selected'].sum():,} passes"
    )
    return scores.drop(columns=["selected"]), diagnostic


def generate_scores(dataset, draws):
    score_dir = RESULTS / "score_folds"
    score_dir.mkdir(parents=True, exist_ok=True)
    features = list(gate_mod.FEATURES)
    pieces = []
    diagnostics = []
    years = range(2019, int(dataset["year"].max()) + 1)
    for year in years:
        score_path = score_dir / f"scores_{year}.parquet"
        diag_path = score_dir / f"diagnostics_{year}.json"
        if score_path.exists() and diag_path.exists():
            piece = pd.read_parquet(score_path)
            piece["event_date"] = pd.to_datetime(piece["event_date"])
            pieces.append(piece)
            diagnostics.append(json.loads(diag_path.read_text()))
            log(f"Year {year}: scores loaded from cache, n={len(piece):,}")
            continue

        train = dataset[dataset["year"] < year].copy()
        test = dataset[dataset["year"] == year].copy()
        if len(train) < 500 or test.empty:
            log(f"Year {year}: skipped, train={len(train):,}, test={len(test):,}")
            continue
        log(f"Year {year}: fitting on {len(train):,}, scoring {len(test):,}")
        predictions, residuals, factor_metrics = sim.fit_outer_models(
            train, test, features, log=log
        )
        piece, fallback = sim.simulate_test_rows(
            test, predictions, residuals, draws=draws
        )
        direct, direct_pwin = direct_gate_scores(train, test)
        piece["incumbent_direct_return"] = direct
        piece["incumbent_direct_return_pwin"] = direct_pwin
        piece.to_parquet(score_path, index=False)
        diag = {
            "year": year,
            "max_train_year": int(train["year"].max()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "residual_pool": int(len(residuals)),
            "factor_metrics": factor_metrics,
            "pool_fallback": fallback,
        }
        write_json(diag_path, diag)
        diagnostics.append(diag)
        pieces.append(piece)
        log(f"Year {year}: simulation complete")
    if not pieces:
        raise RuntimeError("No outer-fold simulator scores were produced")
    scores = pd.concat(pieces, ignore_index=True)
    scores["event_date"] = pd.to_datetime(scores["event_date"])
    scores = scores.sort_values(["event_date", "event_id"]).reset_index(drop=True)
    return scores, diagnostics


def apply_trailing_gate(scores, arm):
    result = pd.Series(False, index=scores.index)
    cutoffs = []
    dates = pd.to_datetime(scores["event_date"])
    month = dates.dt.to_period("M").dt.to_timestamp()
    for current in sorted(month.unique()):
        start = pd.Timestamp(current) - pd.DateOffset(months=6)
        history = scores[
            (dates >= start) & (dates < pd.Timestamp(current))
        ][arm].dropna()
        here = month == current
        cutoff = float(history.quantile(0.80)) if len(history) >= 100 else None
        if cutoff is not None:
            result.loc[here] = scores.loc[here, arm] >= cutoff
        cutoffs.append({
            "month": str(pd.Timestamp(current).date()),
            "arm": arm,
            "history_n": int(len(history)),
            "cutoff": cutoff,
            "selected": int(result.loc[here].sum()),
            "candidates": int(here.sum()),
        })
    return result, cutoffs


def add_gate_decisions(scores, threshold):
    out = scores.copy()
    out["selected_incumbent_direct_return"] = (
        out["incumbent_direct_return"] >= float(threshold)
    )
    cutoffs = []
    for arm in sim.SCORE_ARMS:
        selected, rows = apply_trailing_gate(out, arm)
        out[f"selected_{arm}"] = selected
        cutoffs.extend(rows)
    return out, cutoffs


class PrecomputedGate:
    def __init__(self, scores, arm):
        indexed = scores.drop_duplicates("event_id").set_index("event_id")
        self.selected = indexed[f"selected_{arm}"].astype(bool).to_dict()
        self.proba = indexed[f"{arm}_pwin"].astype(float).to_dict()

    def fit(self, train):
        return None

    def select(self, rows):
        return rows["event_id"].map(self.selected).fillna(False).astype(bool)

    def predict_proba(self, rows):
        return rows["event_id"].map(self.proba).to_numpy(dtype=float)

    def gate(self, arm):
        return Gate(
            fit=self.fit,
            select=self.select,
            predict_proba=self.predict_proba,
            name=arm,
        )


def score_deciles(scores, arm):
    frame = scores[["event_id", arm, "realized_ret"]].dropna().copy()
    if len(frame) < 20 or frame[arm].nunique() < 10:
        return [], None, None
    frame["decile"] = pd.qcut(
        frame[arm], 10, labels=False, duplicates="drop"
    ) + 1
    rows = (
        frame.groupby("decile")
        .agg(n=("event_id", "size"), score_mean=(arm, "mean"),
             realized_mean=("realized_ret", "mean"))
        .reset_index()
        .to_dict("records")
    )
    corr = spearmanr(frame[arm], frame["realized_ret"], nan_policy="omit")
    spread = float(rows[-1]["realized_mean"] - rows[0]["realized_mean"])
    return rows, float(corr.statistic), spread


def policy_bootstrap(scores, challenger, incumbent, draws=10000):
    frame = scores[
        ["event_id", "event_date", "realized_ret",
         f"selected_{challenger}", f"selected_{incumbent}"]
    ].copy()
    frame["week"] = pd.to_datetime(frame["event_date"]).dt.to_period("W").astype(str)
    groups = [group for _, group in frame.groupby("week")]
    rng = np.random.default_rng(142)
    estimates = []
    for _ in range(int(draws)):
        picked = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        sample = pd.concat(picked, ignore_index=True)
        c = sample.loc[sample[f"selected_{challenger}"], "realized_ret"]
        h = sample.loc[sample[f"selected_{incumbent}"], "realized_ret"]
        if len(c) and len(h):
            estimates.append(float(c.mean() - h.mean()))
    arr = np.asarray(estimates)
    observed = (
        frame.loc[frame[f"selected_{challenger}"], "realized_ret"].mean()
        - frame.loc[frame[f"selected_{incumbent}"], "realized_ret"].mean()
    )
    return {
        "observed": float(observed),
        "ci90": np.quantile(arr, [0.05, 0.95]).tolist(),
        "ci95": np.quantile(arr, [0.025, 0.975]).tolist(),
        "p_gt_zero": float((arr > 0).mean()),
        "draws": int(len(arr)),
    }


def matched_selectivity(scores, arm):
    rows = []
    for year, group in scores.groupby("year"):
        n = int(group["selected_incumbent_direct_return"].sum())
        ranked = group.nlargest(n, arm) if n else group.iloc[0:0]
        rows.append({
            "year": int(year),
            "n": n,
            "mean": float(ranked["realized_ret"].mean()) if n else None,
        })
    return rows


def oracle_guard(dataset):
    rows = dataset[dataset["year"] >= 2020].copy()
    oracle = sim.oracle_reprice(rows)
    median_error = float(oracle["value_abs_error_fraction"].median())
    mean_bias = float((oracle["oracle_ret"] - oracle["ret"]).mean())
    return {
        "n": int(len(oracle)),
        "median_absolute_value_error_fraction": median_error,
        "mean_return_bias": mean_bias,
        "passes": bool(median_error <= 0.10 and abs(mean_bias) <= 0.02),
    }, oracle


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


def report_sections(
    result,
    evaluations,
    ranking,
    factor_diagnostics,
    guard,
    policy,
    matched,
    counts,
    incumbent_reproduction,
):
    primary = result.results["headline"]
    arm_rows = []
    for arm in ARMS:
        metrics = (
            primary if arm == PRIMARY
            else evaluations[arm].results["headline"]
        )
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
            f"{metrics['mc'].get('0.05', {}).get('p_loss', float('nan')):.3f}",
        ])

    rank_rows = []
    for arm, values in ranking.items():
        rank_rows.append([
            arm,
            f"{values['spearman']:+.3f}" if values["spearman"] is not None else "n/a",
            fmt_pct(values["decile_spread"]),
        ])

    factor_rows = []
    for diag in factor_diagnostics:
        if int(diag["year"]) < 2020:
            continue
        for target, values in diag["factor_metrics"].items():
            factor_rows.append([
                str(diag["year"]), target, f"{values['n']:,}",
                f"{values['r']:+.3f}" if values["r"] is not None else "n/a",
                f"{values['mae']:.4f}", f"{values['bias']:+.4f}",
            ])

    matched_rows = [
        [str(row["year"]), f"{row['n']:,}", fmt_pct(row["mean"])]
        for row in matched
    ]
    lo90, hi90 = policy["ci90"]
    promotion = (
        primary["cagr"] > evaluations["incumbent_direct_return"].results["headline"]["cagr"]
        and primary["sharpe_trade"] > evaluations["incumbent_direct_return"].results["headline"]["sharpe_trade"]
        and primary["years_positive"] / max(primary["years_evaluated"], 1)
        >= evaluations["incumbent_direct_return"].results["headline"]["years_positive"]
        / max(evaluations["incumbent_direct_return"].results["headline"]["years_evaluated"], 1)
        and primary["mc"].get("0.05", {}).get("p_loss", 1.0)
        <= evaluations["incumbent_direct_return"].results["headline"]["mc"].get("0.05", {}).get("p_loss", 1.0)
        and lo90 > 0
        and result.results.get("calibration", {}).get("brier_skill", -np.inf) >= -0.05
        and guard["passes"]
    )
    return [
        {
            "title": "Decision",
            "body": [
                f"**{'PROMOTION CRITERIA MET' if promotion else 'NO PROMOTION'}**.",
                f"The primary policy-value difference is {fmt_pct(policy['observed'])}; "
                f"90% earnings-week interval [{fmt_pct(lo90)}, {fmt_pct(hi90)}], "
                f"P(greater than zero) {policy['p_gt_zero']:.1%}.",
                f"Oracle-state repricing guard: {'PASS' if guard['passes'] else 'FAIL'} "
                f"on {guard['n']:,} events; median value error "
                f"{guard['median_absolute_value_error_fraction']:.1%}, mean return bias "
                f"{fmt_pct(guard['mean_return_bias'])}.",
                f"Registered incumbent reproduction: threshold "
                f"{incumbent_reproduction['learned_threshold']:.8f}, "
                f"{incumbent_reproduction['passes_full_gate_universe']:,} passes on its "
                f"full OOS gate universe before the common-universe intersection.",
            ],
        },
        {
            "title": "One trade, five gates",
            "note": (
                "Every row trades the exact incumbent T-14, minimum-30-DTE "
                "straddle with zero commissions. Only the selection rule changes."
            ),
            "columns": [
                "gate", "selected", "mean", "capital weighted", "CAGR",
                "Sharpe", "years+", "breakeven alpha", "MC P(loss) at 5%",
            ],
            "align": ["---"] + ["---:"] * 8,
            "rows": arm_rows,
        },
        {
            "title": "Ranking quality",
            "columns": ["gate score", "Spearman versus realized return", "top-bottom decile"],
            "align": ["---", "---:", "---:"],
            "rows": rank_rows,
        },
        {
            "title": "Factor forecasts by outer year",
            "columns": ["year", "target", "n", "r", "MAE", "bias"],
            "align": ["---", "---", "---:", "---:", "---:", "---:"],
            "rows": factor_rows,
        },
        {
            "title": "Primary at incumbent-matched selectivity",
            "note": "Secondary diagnostic: sim_joint takes exactly the incumbent count inside each year.",
            "columns": ["year", "matched n", "sim_joint mean"],
            "align": ["---", "---:", "---:"],
            "rows": matched_rows,
        },
        {
            "title": "Sample funnel",
            "body": [
                f"- Exact engine-replayed T-14 events: {counts['priced']:,}.",
                f"- Complete factor state and registered features: {counts['model_ready']:,}.",
                f"- Common scored 2020-2026 evaluation events: {counts['evaluated']:,}.",
                "- Simulator thresholds use only score distributions from the strictly prior six months.",
                "- Residual vectors are cross-fitted inside prior years and paired by historical event.",
            ],
        },
        {
            "title": "Interpretation boundary",
            "body": [
                "- ORATS impliedMove is a conditioning feature, not the variance used to price options.",
                "- The state-to-price mapping must pass the oracle guard before any gate result is promotable.",
                "- The independent-residual and point-forecast arms explain the mechanism; they cannot trigger promotion.",
                "- The experiment predicts midpoint value. Realized returns remain reported across the full bid-ask fill grid.",
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    spec = lib.load_spec(HERE / "spec.yaml")
    if int(args.draws) != int(spec["primary_spec"]["draws_per_event"]):
        raise ValueError("draw count differs from the preregistered primary")

    registry = load_registry(missing_ok=False)
    incumbent = registry.champion("gate", STRATEGY)
    expected = float(spec["incumbent"]["threshold"])
    if incumbent.id != "gate_midfill_str_runup" or abs(float(incumbent.threshold) - expected) > 1e-12:
        raise RuntimeError("registered STR-RUNUP gate changed after preregistration")
    if tuple(incumbent.features) != tuple(gate_mod.FEATURES):
        raise RuntimeError("registered feature contract changed after preregistration")

    trades = load_trades()
    gate_dataset = build_factor_dataset(trades)
    dataset = complete_dataset(gate_dataset)
    guard, oracle = oracle_guard(dataset)
    oracle.to_parquet(RESULTS / "oracle_repricing.parquet", index=False)
    write_json(RESULTS / "repricing_guard.json", guard)
    log(
        f"Oracle repricing guard {'PASS' if guard['passes'] else 'FAIL'}: "
        f"median value error {guard['median_absolute_value_error_fraction']:.1%}, "
        f"return bias {guard['mean_return_bias']:+.2%}")

    scores, factor_diagnostics = generate_scores(dataset, args.draws)
    incumbent_scores, incumbent_reproduction = registered_gate_scores(
        gate_dataset, expected
    )
    scores = scores.drop(columns=[
        "incumbent_direct_return", "incumbent_direct_return_pwin"
    ]).merge(incumbent_scores, on=["event_id", "year"], how="left")
    scores, cutoffs = add_gate_decisions(scores, expected)
    needed = [
        arm for arm in ARMS
    ] + [f"{arm}_pwin" for arm in ARMS]
    common_ok = np.isfinite(scores[needed].to_numpy(dtype=float)).all(axis=1)
    common_scores = scores[
        common_ok & (scores["year"] >= int(spec["primary_spec"]["first_test_year"]))
    ].copy()
    common_ids = set(common_scores["event_id"])
    eval_trades = trades[
        trades["event_id"].isin(common_ids)
        & (trades["event_date"].dt.year >= int(spec["primary_spec"]["first_test_year"]))
    ].copy()
    if eval_trades["event_id"].nunique() != len(common_ids):
        raise RuntimeError("common score universe does not reconcile to priced trades")
    common_scores.to_parquet(RESULTS / "oos_scores.parquet", index=False)
    pd.DataFrame(cutoffs).to_csv(RESULTS / "trailing_cutoffs.csv", index=False)
    write_json(RESULTS / "factor_diagnostics.json", factor_diagnostics)
    write_json(RESULTS / "incumbent_reproduction.json", incumbent_reproduction)
    log(
        f"Evaluation universe: {len(common_scores):,} common events, "
        f"{len(eval_trades):,} fill rows")

    ranking = {}
    for arm in ARMS:
        deciles, corr, spread = score_deciles(common_scores, arm)
        ranking[arm] = {
            "spearman": corr,
            "decile_spread": spread,
            "deciles": deciles,
        }
    write_json(RESULTS / "ranking.json", ranking)
    policy = policy_bootstrap(
        common_scores, PRIMARY, "incumbent_direct_return", draws=10000
    )
    write_json(RESULTS / "policy_bootstrap.json", policy)
    matched = matched_selectivity(common_scores, PRIMARY)
    write_json(RESULTS / "matched_selectivity.json", matched)

    spy = common.load_spy_daily()
    evaluations = {}
    for arm in ARMS:
        if arm == PRIMARY:
            continue
        run_dir = HERE / "arms" / arm
        state = PrecomputedGate(common_scores, arm)
        result = evaluate(
            cell_spec(spec, arm),
            eval_trades,
            gate=state.gate(arm),
            run_dir=run_dir,
            spy_daily=spy,
            fractions=(0.02, 0.05),
            mc_paths=500,
            seed=142,
            write_report=True,
            input_files=[RESULTS / "oos_scores.parquet"],
        )
        evaluations[arm] = result
        if not args.no_ledger:
            lib.record_evaluation(run_dir, cell_spec(spec, arm), result.results)
        log(
            f"{arm}: selected {result.metrics['n']:,}, "
            f"mean {result.metrics['mean']:+.2%}, CAGR {result.metrics['cagr']:+.2%}")

    counts = {
        "priced": int(trades["event_id"].nunique()),
        "model_ready": int(len(dataset)),
        "evaluated": int(len(common_scores)),
    }
    primary_state = PrecomputedGate(common_scores, PRIMARY)
    primary_result = evaluate(
        spec,
        eval_trades,
        gate=primary_state.gate(PRIMARY),
        run_dir=HERE,
        spy_daily=spy,
        fractions=(0.02, 0.05),
        mc_paths=1000,
        seed=142,
        write_report=True,
        input_files=[RESULTS / "oos_scores.parquet", RESULTS / "repricing_guard.json"],
        extra_sections=lambda result: report_sections(
            result, evaluations, ranking, factor_diagnostics,
            guard, policy, matched, counts, incumbent_reproduction,
        ),
    )
    evaluations[PRIMARY] = primary_result
    if not args.no_ledger:
        lib.record_evaluation(HERE, spec, primary_result.results)

    summary = {
        "spec_hash": lib.spec_hash(spec),
        "counts": counts,
        "repricing_guard": guard,
        "policy_bootstrap": policy,
        "incumbent_reproduction": incumbent_reproduction,
        "ranking": ranking,
        "arms": {
            arm: evaluations[arm].results["headline"] for arm in ARMS
        },
        "quota_calls": 0,
    }
    write_json(RESULTS / "summary.json", summary)
    log(f"Report: {primary_result.report_path}")


if __name__ == "__main__":
    main()
