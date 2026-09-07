#!/usr/bin/env python3
"""EXP-144: corrected-calendar STR-RUNUP T-14 gate rebaseline."""
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
SIM_DIR = ROOT / "experiments/EXP-142_str_runup_t14_factor_simulation_pnl_gate"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SIM_DIR))

from engine.data import store  # noqa: E402
from engine.evaluate import Gate, evaluate  # noqa: E402
from engine.features import FeatureContext  # noqa: E402
from engine.models.registry import load_registry  # noqa: E402
from engine.models.training import gate as gate_mod  # noqa: E402
from experiments import common, lib  # noqa: E402
import simulation as sim  # noqa: E402


STRATEGY = "STR-RUNUP"
VARIANT = "e-14_x+0_target_dte=30"
PRIMARY = "native_nan"
MODELED_ARMS = (
    "incumbent_complete_case",
    "native_nan",
    "native_nan_availability",
    "sim_joint",
)
ALL_ARMS = ("ungated",) + MODELED_ARMS
BASE_FEATURES = tuple(gate_mod.FEATURES)
IM_COLUMNS = ("im", "im_d1", "im_d5", "im_d10")
AVAILABILITY = tuple(f"has_{name}" for name in IM_COLUMNS)
FEATURES_BY_ARM = {
    "incumbent_complete_case": BASE_FEATURES,
    "native_nan": BASE_FEATURES,
    "native_nan_availability": BASE_FEATURES + AVAILABILITY,
}
STARTED = time.monotonic()


def log(message):
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-144 {elapsed:,.0f}s] {message}", flush=True)


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


def entry_relative_spread(raw):
    try:
        doc = json.loads(raw) if isinstance(raw, str) else raw
        legs = (doc or {}).get("entry") or []
        mids = [0.5 * (float(x["bid"]) + float(x["ask"])) for x in legs]
        spreads = [float(x["ask"]) - float(x["bid"]) for x in legs]
        total = sum(mids)
        return sum(spreads) / total if total > 0 else np.nan
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return np.nan


def build_dataset(trades, force=False):
    cache = RESULTS / "factor_dataset.parquet"
    if cache.exists() and not force:
        frame = pd.read_parquet(cache)
        for column in ("event_date", "entry_date", "exit_date", "expiry"):
            frame[column] = pd.to_datetime(frame[column])
        log(f"Loaded corrected factor dataset cache: {len(frame):,} events")
        return frame

    tickers = sorted(trades["ticker"].astype(str).unique())
    log("Loading corrected causal panel and daily market history")
    context = FeatureContext.load(
        tickers=tickers, years=range(2007, 2027), with_daily=True
    )
    log(f"Feature context: {len(context.panel):,} event rows, {len(context.daily):,} daily rows")
    frame = gate_mod.build_dataset(
        trades, panel=context.panel, daily=context.daily
    )
    for column in ("event_date", "entry_date", "exit_date", "expiry"):
        frame[column] = pd.to_datetime(frame[column])
    for name in BASE_FEATURES:
        frame[name] = pd.to_numeric(
            frame[name], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
    for name in IM_COLUMNS:
        frame[f"has_{name}"] = np.isfinite(
            frame[name].to_numpy(dtype=float)
        ).astype(float)
    frame["quote_present"] = frame["has_im"].astype(bool)
    frame["relative_spread"] = frame["legs"].map(entry_relative_spread)

    parsed = frame["legs"].apply(sim.parse_leg_state).apply(pd.Series)
    frame = pd.concat(
        [frame.reset_index(drop=True), parsed.reset_index(drop=True)], axis=1
    )
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
    log("Inferring contract IV and constructing simulation targets")
    frame = sim.add_targets(frame)
    if frame["event_id"].duplicated().any():
        raise RuntimeError("factor dataset has duplicate event IDs")
    frame.to_parquet(cache, index=False)
    log(f"Corrected factor dataset written: {len(frame):,} events")
    return frame


def simulation_ready(frame):
    required = list(BASE_FEATURES) + list(sim.TARGETS) + [
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
    log(f"Simulation-ready state: {len(out):,}/{len(frame):,} events")
    return out


def calibrated_probability(model, train, features, eligible):
    y = train.loc[eligible, "ret"].to_numpy(dtype=float)
    score = model.predict(
        train.loc[eligible, list(features)].to_numpy(dtype=float)
    )
    base = float((y > 0).mean())
    try:
        iso = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        iso.fit(score, (y > 0).astype(float))
        return iso, base
    except ValueError:
        return None, base


def fit_direct(train, test, features, complete_case):
    y_train = pd.to_numeric(
        train["ret"], errors="coerce"
    ).to_numpy(dtype=float)
    train_ok = np.isfinite(y_train)
    test_ok = np.ones(len(test), dtype=bool)
    if complete_case:
        train_ok &= np.isfinite(
            train[list(features)].to_numpy(dtype=float)
        ).all(axis=1)
        test_ok &= np.isfinite(
            test[list(features)].to_numpy(dtype=float)
        ).all(axis=1)
    if train_ok.sum() < 250:
        return np.full(len(test), np.nan), np.full(len(test), np.nan), int(train_ok.sum())

    active = []
    for feature in features:
        values = train.loc[train_ok, feature].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and np.unique(finite).size >= 2:
            active.append(feature)
    model = gate_mod.fit(
        train.loc[train_ok, active].to_numpy(dtype=float),
        y_train[train_ok],
    )
    score = np.full(len(test), np.nan)
    pwin = np.full(len(test), np.nan)
    if test_ok.any():
        score[test_ok] = model.predict(
            test.loc[test_ok, active].to_numpy(dtype=float)
        )
        iso, base = calibrated_probability(model, train, active, train_ok)
        pwin[test_ok] = (
            iso.predict(score[test_ok]) if iso is not None else base
        )
    return score, pwin, int(train_ok.sum())


def generate_scores(dataset, sim_data, draws, force=False):
    score_dir = RESULTS / "score_folds"
    score_dir.mkdir(parents=True, exist_ok=True)
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
            log(f"Year {year}: loaded {len(piece):,} cached scores")
            continue

        train = dataset[dataset["year"] < year].copy()
        test = dataset[dataset["year"] == year].copy()
        if len(train) < 500 or test.empty:
            log(f"Year {year}: skipped, train={len(train):,}, test={len(test):,}")
            continue
        log(f"Year {year}: direct gates fit {len(train):,}, score {len(test):,}")
        keep = [
            "event_id", "ticker", "event_date", "year", "ret",
            "quote_present", "mcap_log", "relative_spread",
        ]
        piece = test[keep].rename(
            columns={"ret": "realized_ret"}
        ).reset_index(drop=True)
        train_counts = {}
        for arm in MODELED_ARMS[:3]:
            features = FEATURES_BY_ARM[arm]
            score, pwin, n_train = fit_direct(
                train, test, features,
                complete_case=arm == "incumbent_complete_case",
            )
            piece[arm] = score
            piece[f"{arm}_pwin"] = pwin
            train_counts[arm] = n_train

        sim_train = sim_data[sim_data["year"] < year].copy()
        sim_test = sim_data[sim_data["year"] == year].copy()
        factor_metrics = {}
        pool_fallback = {}
        if len(sim_train) >= 500 and len(sim_test):
            log(f"Year {year}: simulation fit {len(sim_train):,}, score {len(sim_test):,}")
            predictions, residuals, factor_metrics = sim.fit_outer_models(
                sim_train, sim_test, BASE_FEATURES, log=log
            )
            sim_scores, pool_fallback = sim.simulate_test_rows(
                sim_test, predictions, residuals, draws=draws
            )
            piece = piece.merge(
                sim_scores[
                    ["event_id", "sim_joint", "sim_joint_pwin"]
                ],
                on="event_id", how="left", validate="one_to_one",
            )
        else:
            piece["sim_joint"] = np.nan
            piece["sim_joint_pwin"] = np.nan

        diag = {
            "year": year,
            "max_train_year": int(train["year"].max()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_rows_by_arm": train_counts,
            "simulation_train": int(len(sim_train)),
            "simulation_test": int(len(sim_test)),
            "factor_metrics": factor_metrics,
            "pool_fallback": pool_fallback,
            "scoreable_by_arm": {
                arm: int(np.isfinite(piece[arm]).sum())
                for arm in MODELED_ARMS
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
    cutoffs = []
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
        cutoffs.append({
            "month": str(current.date()),
            "arm": arm,
            "history_n": int(len(history)),
            "cutoff": cutoff,
            "candidates": int(np.isfinite(scores.loc[here, arm]).sum()),
            "selected": int(selected.loc[here].sum()),
        })
    return selected, cutoffs


def add_decisions(scores):
    out = scores.copy()
    cutoffs = []
    for arm in MODELED_ARMS:
        selected, rows = apply_trailing_gate(out, arm)
        out[f"selected_{arm}"] = selected
        cutoffs.extend(rows)
    out["selected_ungated"] = True
    return out, cutoffs


def incumbent_reproduction(scores, spec):
    oos = scores[scores["year"] >= 2020]["incumbent_complete_case"].dropna()
    learned = float(gate_mod.choose_threshold(oos.to_numpy(dtype=float)))
    stored = float(spec["incumbent"]["stored_threshold"])
    selected = int((oos >= stored).sum())
    result = {
        "oos_rows": int(len(oos)),
        "learned_threshold": learned,
        "stored_threshold": stored,
        "selected_at_stored_threshold": selected,
        "expected_oos_rows": int(spec["incumbent"]["expected_oos_rows"]),
        "expected_selected": int(
            spec["incumbent"]["expected_selected_at_stored_threshold"]
        ),
    }
    if result["oos_rows"] != result["expected_oos_rows"]:
        raise RuntimeError(f"incumbent OOS row mismatch: {result}")
    if selected != result["expected_selected"]:
        raise RuntimeError(f"incumbent selected-count mismatch: {result}")
    if abs(learned - stored) > 1e-12:
        raise RuntimeError(f"incumbent threshold mismatch: {result}")
    log(f"Incumbent reproduced: {len(oos):,} scores, {selected:,} stored-threshold passes")
    return result


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


def ranking(scores):
    output = {}
    for arm in MODELED_ARMS:
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


def exact_matched(scores):
    rows = []
    common = np.isfinite(
        scores[list(MODELED_ARMS)].to_numpy(dtype=float)
    ).all(axis=1)
    for year, group in scores[common].groupby("year"):
        n = int(np.floor(0.20 * len(group)))
        for arm in MODELED_ARMS:
            picked = group.nlargest(n, arm)
            rows.append({
                "year": int(year),
                "arm": arm,
                "common_candidates": int(len(group)),
                "selected": n,
                "mean": float(picked["realized_ret"].mean()) if n else None,
            })
    return rows


def policy_bootstrap(scores, draws=10000):
    frame = scores.copy()
    frame["week"] = pd.to_datetime(
        frame["event_date"]
    ).dt.to_period("W").astype(str)
    rows = []
    for _, group in frame.groupby("week"):
        challenger = group.loc[
            group[f"selected_{PRIMARY}"], "realized_ret"
        ]
        incumbent = group.loc[
            group["selected_incumbent_complete_case"], "realized_ret"
        ]
        rows.append((
            challenger.sum(), len(challenger),
            incumbent.sum(), len(incumbent),
        ))
    weekly = np.asarray(rows, dtype=float)
    rng = np.random.default_rng(144)
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
    challenger = frame.loc[
        frame[f"selected_{PRIMARY}"], "realized_ret"
    ]
    incumbent = frame.loc[
        frame["selected_incumbent_complete_case"], "realized_ret"
    ]
    return {
        "observed": float(challenger.mean() - incumbent.mean()),
        "ci90": np.quantile(arr, [0.05, 0.95]).tolist(),
        "ci95": np.quantile(arr, [0.025, 0.975]).tolist(),
        "p_gt_zero": float((arr > 0).mean()),
        "draws": int(len(arr)),
    }


def cohort_metrics(scores):
    output = []
    quote_masks = {
        "quote_present": scores["quote_present"].astype(bool),
        "quote_absent": ~scores["quote_present"].astype(bool),
    }
    mcap = np.exp(pd.to_numeric(scores["mcap_log"], errors="coerce"))
    mcap_masks = {
        "mcap_lt_1b": mcap < 1e9,
        "mcap_1b_10b": (mcap >= 1e9) & (mcap < 1e10),
        "mcap_ge_10b": mcap >= 1e10,
    }
    spread = pd.to_numeric(scores["relative_spread"], errors="coerce")
    try:
        spread_q = pd.qcut(spread, 5, labels=False, duplicates="drop")
    except ValueError:
        spread_q = pd.Series(np.nan, index=scores.index)
    spread_masks = {
        f"spread_q{q + 1}": spread_q == q
        for q in sorted(spread_q.dropna().astype(int).unique())
    }
    masks = {**quote_masks, **mcap_masks, **spread_masks}
    for arm in ALL_ARMS:
        selected = scores[f"selected_{arm}"].astype(bool)
        for cohort, mask in masks.items():
            rows = scores[selected & mask]
            output.append({
                "arm": arm,
                "cohort": cohort,
                "candidates": int(mask.sum()),
                "selected": int(len(rows)),
                "mean": float(rows["realized_ret"].mean()) if len(rows) else None,
                "win_rate": float((rows["realized_ret"] > 0).mean()) if len(rows) else None,
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


def report_sections(result, evaluations, ranks, matched, policy, cohorts, reproduction, counts):
    primary = result.results["headline"]
    incumbent = evaluations[
        "incumbent_complete_case"
    ].results["headline"]
    checks = {
        "mean": primary["mean"] > incumbent["mean"],
        "capital_weighted": primary.get("dollar_weighted", -np.inf)
        > incumbent.get("dollar_weighted", -np.inf),
        "cagr": primary["cagr"] > incumbent["cagr"],
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
        "policy_ci": policy["ci90"][0] > 0,
        "rank_correlation": (
            ranks[PRIMARY]["spearman"]
            > ranks["incumbent_complete_case"]["spearman"]
        ),
        "decile_spread": (
            ranks[PRIMARY]["top_bottom_decile"]
            > ranks["incumbent_complete_case"]["top_bottom_decile"]
        ),
    }
    promotion = all(checks.values())
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
        arm, f"{values['n']:,}", f"{values['spearman']:+.3f}",
        fmt_pct(values["top_bottom_decile"]),
    ] for arm, values in ranks.items()]
    matched_rows = [[
        str(row["year"]), row["arm"], f"{row['common_candidates']:,}",
        f"{row['selected']:,}", fmt_pct(row["mean"]),
    ] for row in matched]
    cohort_rows = [[
        row["arm"], row["cohort"], f"{row['candidates']:,}",
        f"{row['selected']:,}", fmt_pct(row["mean"]),
        fmt_pct(row["win_rate"]),
    ] for row in cohorts]
    check_rows = [[
        name, "PASS" if passed else "FAIL"
    ] for name, passed in checks.items()]
    lo90, hi90 = policy["ci90"]
    return [
        {
            "title": "Rebaseline decision",
            "body": [
                f"**{'PROMOTION CRITERIA MET' if promotion else 'NO PROMOTION'}**.",
                f"Native-missing minus complete-case policy value: "
                f"{fmt_pct(policy['observed'])}; 90% earnings-week interval "
                f"[{fmt_pct(lo90)}, {fmt_pct(hi90)}], "
                f"P(greater than zero) {policy['p_gt_zero']:.1%}.",
                f"Incumbent reproduction: {reproduction['oos_rows']:,} OOS scores, "
                f"{reproduction['selected_at_stored_threshold']:,} stored-threshold passes, "
                f"threshold {reproduction['learned_threshold']:.8f}.",
            ],
        },
        {
            "title": "One corrected T-14 book, five policies",
            "note": "All policies use zero commissions and identical ORATS-priced option trades.",
            "columns": [
                "policy", "selected", "mean", "capital weighted",
                "CAGR", "Sharpe", "years+", "breakeven alpha",
            ],
            "align": ["---"] + ["---:"] * 7,
            "rows": arm_rows,
        },
        {
            "title": "Pre-registered promotion checks",
            "columns": ["check", "status"],
            "align": ["---", "---"],
            "rows": check_rows,
        },
        {
            "title": "Ranking quality",
            "columns": ["gate", "scoreable", "Spearman", "top-bottom decile"],
            "align": ["---", "---:", "---:", "---:"],
            "rows": rank_rows,
        },
        {
            "title": "Exact annual top-20-percent matched diagnostic",
            "note": "All gates rank the same common scoreable events. This full-year diagnostic is not deployable and cannot trigger promotion.",
            "columns": ["year", "gate", "common candidates", "selected", "mean"],
            "align": ["---", "---", "---:", "---:", "---:"],
            "rows": matched_rows,
        },
        {
            "title": "Availability, market-cap and spread slices",
            "columns": [
                "policy", "cohort", "candidates", "selected", "mean", "win rate",
            ],
            "align": ["---", "---", "---:", "---:", "---:", "---:"],
            "rows": cohort_rows,
        },
        {
            "title": "Sample funnel",
            "body": [
                f"- Exact corrected-calendar T-14 priced events: {counts['priced']:,}.",
                f"- OOS 2020-2026 candidates: {counts['oos']:,}.",
                f"- Complete-case incumbent scores: {counts['incumbent_scoreable']:,}.",
                f"- Native-missing scores: {counts['native_scoreable']:,}.",
                f"- Joint-simulation scores: {counts['simulation_scoreable']:,}.",
                f"- Entry implied move present: {counts['quote_present']:,}; absent: {counts['quote_absent']:,}.",
                "- The 2019 fold supplies causal score history only and is excluded from every headline return.",
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
        raise ValueError("draw count differs from preregistration")

    registry = load_registry(missing_ok=False)
    incumbent = registry.champion("gate", STRATEGY)
    expected = float(spec["incumbent"]["stored_threshold"])
    if incumbent.id != spec["incumbent"]["id"]:
        raise RuntimeError("registered STR-RUNUP gate ID changed")
    if abs(float(incumbent.threshold) - expected) > 1e-12:
        raise RuntimeError("registered STR-RUNUP threshold changed")
    if tuple(incumbent.features) != BASE_FEATURES:
        raise RuntimeError("registered STR-RUNUP feature contract changed")

    trades = load_trades()
    dataset = build_dataset(trades, force=args.force)
    sim_data = simulation_ready(dataset)
    scores, diagnostics = generate_scores(
        dataset, sim_data, args.draws, force=args.force
    )
    scores, cutoffs = add_decisions(scores)
    oos = scores[scores["year"] >= 2020].copy()
    reproduction = incumbent_reproduction(scores, spec)
    oos.to_parquet(RESULTS / "oos_scores.parquet", index=False)
    pd.DataFrame(cutoffs).to_csv(
        RESULTS / "trailing_cutoffs.csv", index=False
    )
    write_json(RESULTS / "fold_diagnostics.json", diagnostics)
    write_json(RESULTS / "incumbent_reproduction.json", reproduction)

    ranks = ranking(oos)
    matched = exact_matched(oos)
    policy = policy_bootstrap(oos)
    cohorts = cohort_metrics(oos)
    write_json(RESULTS / "ranking.json", ranks)
    write_json(RESULTS / "matched_selectivity.json", matched)
    write_json(RESULTS / "policy_bootstrap.json", policy)
    write_json(RESULTS / "cohort_metrics.json", cohorts)

    event_ids = set(oos["event_id"])
    eval_trades = trades[trades["event_id"].isin(event_ids)].copy()
    if eval_trades["event_id"].nunique() != len(oos):
        raise RuntimeError("OOS scores do not reconcile to priced trades")
    counts = {
        "priced": int(trades["event_id"].nunique()),
        "oos": int(len(oos)),
        "incumbent_scoreable": int(
            np.isfinite(oos["incumbent_complete_case"]).sum()
        ),
        "native_scoreable": int(np.isfinite(oos["native_nan"]).sum()),
        "simulation_scoreable": int(np.isfinite(oos["sim_joint"]).sum()),
        "quote_present": int(oos["quote_present"].sum()),
        "quote_absent": int((~oos["quote_present"].astype(bool)).sum()),
    }
    log(f"Evaluating {len(oos):,} OOS events and {len(eval_trades):,} fill rows")
    spy = common.load_spy_daily()
    evaluations = {}

    ungated = evaluate(
        arm_spec(spec, "ungated"), eval_trades,
        run_dir=HERE / "arms/ungated", spy_daily=spy,
        fractions=(0.02, 0.05), mc_paths=500, seed=144,
        write_report=True, input_files=[RESULTS / "oos_scores.parquet"],
    )
    evaluations["ungated"] = ungated
    if not args.no_ledger:
        lib.record_evaluation(
            HERE / "arms/ungated", arm_spec(spec, "ungated"),
            ungated.results,
        )
    log(f"ungated: n={ungated.metrics['n']:,}, mean={ungated.metrics['mean']:+.2%}")

    for arm in MODELED_ARMS:
        if arm == PRIMARY:
            continue
        run_dir = HERE / "arms" / arm
        result = evaluate(
            arm_spec(spec, arm), eval_trades,
            gate=PrecomputedGate(oos, arm).gate(), run_dir=run_dir,
            spy_daily=spy, fractions=(0.02, 0.05), mc_paths=500,
            seed=144, write_report=True,
            input_files=[RESULTS / "oos_scores.parquet"],
        )
        evaluations[arm] = result
        if not args.no_ledger:
            lib.record_evaluation(
                run_dir, arm_spec(spec, arm), result.results
            )
        log(f"{arm}: n={result.metrics['n']:,}, mean={result.metrics['mean']:+.2%}")

    primary = evaluate(
        spec, eval_trades,
        gate=PrecomputedGate(oos, PRIMARY).gate(), run_dir=HERE,
        spy_daily=spy, fractions=(0.02, 0.05), mc_paths=1000,
        seed=144, write_report=True,
        input_files=[
            RESULTS / "oos_scores.parquet",
            RESULTS / "incumbent_reproduction.json",
        ],
        extra_sections=lambda result: report_sections(
            result, evaluations, ranks, matched, policy,
            cohorts, reproduction, counts,
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
        "ranking": ranks,
        "matched_selectivity": matched,
        "cohorts": cohorts,
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
