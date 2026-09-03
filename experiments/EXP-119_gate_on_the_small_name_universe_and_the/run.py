#!/usr/bin/env python3
"""EXP-119 — gate on the small-name universe and the threshold ladder.

Run:  python3 experiments/EXP-119_gate_on_the_small_name_universe_and_the/run.py

Can the champion gate machinery make money on the small-name slice by itself,
and does selecting fewer trades raise per-trade returns? Four pre-registered
arms (spec.yaml, plan guides/exp119_small_universe_gate_thresholds.md):

  a_thru   PRIMARY    STR-THRU,  train slice / test slice 2024-26, min_train_rows=100
  a_runup  secondary  STR-RUNUP, train slice / test slice 2025-26, min_train_rows=100
  b_thru   secondary  STR-THRU,  train full universe / test slice rows 2020-26
  b_runup  secondary  STR-RUNUP, train full universe / test slice rows 2020-26

Everything about the model is held to the champion spec — gate_mod.FEATURES,
gate_mod.fit (HistGBM, unchanged hyperparameters), walk-forward by calendar
year, threshold = choose_threshold quantile of scored predictions — varying
only the train/test universe, the pre-registered min_train_rows relaxation for
arm A, and the top fraction. The threshold ladder {10%, 20%, 30%} is read off
ONE walk-forward run per arm (same pooled OOS scores, three cutoffs), judged
at the headline 20% cutoff with 10,000 seeded bootstrap resamples.

The data path mirrors EXP-118 exactly (same trades filter, same
gate_mod.build_dataset call). The primary arm additionally renders through the
standard engine.report harness so this report carries the same structure and
figures as EXP-102/EXP-105: equity curve and drawdown, fill-quality
(breakeven alpha) curve, by-year, Monte Carlo fan, stress grid, reliability.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.data.coverage import bucket_mcap  # noqa: E402
from engine.evaluate import (  # noqa: E402
    MC_BLOCK,
    MC_PATHS,
    SIZING_FRACTIONS,
    EvalResult,
    alpha_sweep,
    append_run_log,
    breakeven_alpha_from_sweep,
    build_equity,
    by_year_table,
    calibration_block,
    capacity_note,
    check_preregistration,
    dollar_weighted_return,
    monte_carlo,
    sharpe_equity,
    spec_hash,
    stress_iv_regime,
    stress_regimes,
    trade_stats,
    transaction_log,
    reconcile_transaction_log,
)
from engine.features import load_panel  # noqa: E402
from engine.models.registry import load_registry  # noqa: E402
from engine.models.training import gate as gate_mod  # noqa: E402
from engine.models.training.common import SEED  # noqa: E402
from engine.report import Report, fig_equity  # noqa: E402
from experiments import common, lib  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

CUTOFFS = (0.10, 0.20, 0.30)
HEADLINE = 0.20
N_BOOT = 10_000

ARMS = [
    # name, strategy, train universe ("slice" | "full"), first_test_year, min_train_rows, status
    ("a_thru", "STR-THRU", "slice", 2024, 100, "primary"),
    ("a_runup", "STR-RUNUP", "slice", 2025, 100, "secondary (exploratory power)"),
    ("b_thru", "STR-THRU", "full", 2020, 500, "secondary"),
    ("b_runup", "STR-RUNUP", "full", 2020, 500, "secondary"),
]


def log(msg: str) -> None:
    print(f"[EXP-119 {datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# universe
# ---------------------------------------------------------------------------


def computed_moves_names() -> frozenset:
    """The EXP-117 universe, same derivation as engine.score."""
    return frozenset(
        p.stem[len("moves_"):] for p in paths.COMPUTED_MOVES.glob("moves_*.json")
    )


def mcap_bucket_of(dataset: pd.DataFrame) -> pd.Series:
    """bucket_mcap(exp(mcap_log)) — the plan's spelling; NaN stays unknown."""
    mcap_log = pd.to_numeric(dataset.get("mcap_log"), errors="coerce")
    return bucket_mcap(np.exp(mcap_log.to_numpy(dtype=float)))


def slice_mask(dataset: pd.DataFrame, computed_names: frozenset) -> pd.Series:
    """Plan section 2: mcap below $1B OR a computed-moves name."""
    small = (mcap_bucket_of(dataset) == "<1B").to_numpy()
    computed = dataset["ticker"].isin(computed_names).to_numpy()
    return pd.Series(small | computed, index=dataset.index)


def build_gate_dataset(strategy: str, trades: pd.DataFrame, panel) -> pd.DataFrame:
    """gate_mod.build_dataset with a per-strategy parquet cache (delete the
    cache file to force a rebuild after a store change — same convention as
    experiments.common.gate_dataset)."""
    RESULTS.mkdir(exist_ok=True)
    cache = RESULTS / f"gate_dataset_{strategy.lower().replace('-', '_')}.parquet"
    if cache.exists():
        frame = pd.read_parquet(cache)
        log(f"{strategy}: gate dataset {len(frame):,} rows from cache")
        return frame
    started = time.time()
    frame = gate_mod.build_dataset(trades, panel=panel)
    frame.to_parquet(cache, index=False)
    log(f"{strategy}: gate dataset {len(frame):,} rows built in {time.time() - started:.0f}s")
    return frame


# ---------------------------------------------------------------------------
# walk-forward (champion machinery, test rows may be a subset of the universe)
# ---------------------------------------------------------------------------


def walk_forward_oos(dataset: pd.DataFrame, test_mask: pd.Series, *,
                     first_test_year: int, min_train_rows: int):
    """Mirrors engine.models.training.common.walk_forward — same feature
    completeness rule, same gate_mod.fit, same expanding window — with one
    difference the arms need: the TEST rows can be restricted (arm B scores
    only slice rows while training on the full universe). For arm A the
    dataset IS the slice and test_mask is all-True, which reproduces the
    champion harness exactly at the pre-registered min_train_rows.

    Also collects an in-fold isotonic P(win) map (EXP-105 convention) on each
    fold's train predictions — reporting-only; selection uses the raw
    regression predictions and the quantile threshold, nothing else.
    """
    from sklearn.isotonic import IsotonicRegression

    features = gate_mod.FEATURES
    usable = dataset[list(features) + ["ret", "year"]]
    complete = np.isfinite(usable[list(features)].to_numpy(dtype=float)).all(axis=1)
    complete &= np.isfinite(usable["ret"].to_numpy(dtype=float))
    data = dataset[complete].copy()
    n_complete = int(len(data))
    log(f"  walk-forward: {n_complete:,} of {len(dataset):,} rows complete on "
        f"{len(features)} features")

    data["pred"] = np.nan
    data["proba"] = np.nan
    test_mask = test_mask.reindex(data.index, fill_value=False)

    years = sorted(int(y) for y in data["year"].dropna().unique())
    years = [y for y in years if y >= first_test_year]
    year_values = data["year"].to_numpy()

    diagnostics: list[dict] = []
    fit_years_seen: list[int] = []
    for year in years:
        train = data[year_values < year]
        test = data[(year_values == year) & test_mask.to_numpy()]
        n_train = int(len(train))
        if n_train < min_train_rows or test.empty:
            diagnostics.append({
                "year": int(year), "n_train": n_train, "n_test": int(len(test)),
                "n_selected": 0, "ungated": False, "fitted": False,
                "reason": (f"n_train {n_train} < min_train_rows {min_train_rows}"
                           if n_train < min_train_rows else "no test rows"),
            })
            continue
        X_train = train[list(features)].to_numpy(dtype=float)
        y_train = train["ret"].to_numpy(dtype=float)
        X_test = test[list(features)].to_numpy(dtype=float)
        model = gate_mod.fit(X_train, y_train, SEED)
        pred = np.asarray(model.predict(X_test), dtype=float).ravel()
        data.loc[test.index, "pred"] = pred
        # Diagnostic P(win): isotonic fit on the fold's train predictions,
        # exactly the EXP-105 registered-gate convention. Never touches the
        # selection rule.
        train_pred = np.asarray(model.predict(X_train), dtype=float).ravel()
        win_train = (y_train > 0).astype(float)
        try:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(train_pred, win_train)
            data.loc[test.index, "proba"] = np.asarray(iso.predict(pred), dtype=float)
        except ValueError:
            pass  # degenerate predictions: proba stays NaN
        fit_years_seen.append(int(train["year"].max()))
        diagnostics.append({
            "year": int(year), "n_train": n_train, "n_test": int(len(test)),
            "n_selected": None,  # filled once the pooled threshold exists
            "ungated": False, "fitted": True,
        })
        log(f"  fold {year}: train {n_train:,} -> test {len(test):,}")

    scored = data[test_mask.to_numpy() & np.isfinite(data["pred"].to_numpy())].copy()
    audit = {
        "years": sorted(int(y) for y in scored["year"].unique()) if len(scored) else [],
        "fit_years_seen": fit_years_seen,
        "leak_free": True,  # train is year < Y by construction (asserted below)
        "receipt": ({
            "n_rows_checked": int(len(scored)),
            "n_folds_checked": len(fit_years_seen),
            "max_fit_year": int(max(fit_years_seen)),
            "min_margin_years": 1,
            "paths": ["exp119.walk_forward_oos"],
        } if fit_years_seen else None),
    }
    for seen, year in zip(fit_years_seen, [d["year"] for d in diagnostics if d["fitted"]]):
        assert seen < year, "walk-forward leak: a fit saw its own test year"
    return scored, diagnostics, audit, n_complete


# ---------------------------------------------------------------------------
# threshold ladder + bootstrap
# ---------------------------------------------------------------------------


def max_trade_contribution(gated: pd.DataFrame) -> dict:
    """The single largest trade's contribution to the gated mean (plan §3):
    a gated mean carried by one trade is not a strategy."""
    if gated.empty:
        return {"n": 0}
    rets = gated["ret"].to_numpy(dtype=float)
    i = int(np.argmax(np.abs(rets)))
    n = len(rets)
    mean = float(rets.mean())
    largest = float(rets[i])
    row = gated.iloc[i]
    without = float(np.delete(rets, i).mean()) if n > 1 else float("nan")
    return {
        "n": n,
        "largest_ret": largest,
        "largest_trade": f"{row.get('ticker', '?')} {pd.Timestamp(row['event_date']):%Y-%m-%d}",
        "contribution_to_mean": largest / n,
        "share_of_mean": (largest / n) / mean if abs(mean) > 1e-12 else None,
        "mean_without_it": without,
    }


def ladder_stats(scored: pd.DataFrame) -> dict:
    """Per-cutoff stats on one arm's pooled OOS scores. The cutoff is the
    choose_threshold quantile of the pooled OOS predictions (plan §4: the
    production rule with the top fraction as the only variable), so exactly
    the top fraction of the pooled scores passes, up to ties."""
    preds = scored["pred"].to_numpy(dtype=float)
    out = {}
    for f in CUTOFFS:
        threshold = gate_mod.choose_threshold(preds, top_fraction=f)
        gated = scored[preds >= threshold]
        rets = gated["ret"].to_numpy(dtype=float)
        by_year = []
        for year, grp in gated.groupby("year", sort=True):
            g = grp["ret"].to_numpy(dtype=float)
            by_year.append({
                "year": int(year), "n_passed": int(len(g)),
                "gated_mean": float(g.mean()) if len(g) else None,
                "gated_win": float((g > 0).mean()) if len(g) else None,
            })
        out[f"{f:.2f}"] = {
            "cutoff": f,
            "threshold": float(threshold),
            "n_passed": int(len(gated)),
            "gated_mean": float(rets.mean()) if len(rets) else None,
            "gated_win": float((rets > 0).mean()) if len(rets) else None,
            "by_year": by_year,
            "max_trade": max_trade_contribution(gated),
            "gated_event_ids": gated["event_id"].tolist(),
        }
    return out


def bootstrap_cis(scored: pd.DataFrame, rng: np.random.Generator) -> dict:
    """10,000 seeded bootstrap resamples (plan §4).

    Per cutoff: resample the pooled OOS GATED trades with replacement, CI on
    the gated mean. Shape test: resample the pooled OOS rows and re-apply all
    three cutoffs to the same resample (paired), giving CIs on the adjacent
    differences without a second look at the data.
    """
    preds = scored["pred"].to_numpy(dtype=float)
    rets = scored["ret"].to_numpy(dtype=float)
    n = len(rets)
    out: dict = {"n_boot": N_BOOT, "seed": int(SEED), "by_cutoff": {}, "shape": {}}
    if n == 0:
        return out

    boot_means = {}
    for f in CUTOFFS:
        thr = gate_mod.choose_threshold(preds, top_fraction=f)
        gated = rets[preds >= thr]
        ng = len(gated)
        if ng == 0:
            out["by_cutoff"][f"{f:.2f}"] = {"ci_lo": None, "ci_hi": None, "half_width": None}
            continue
        idx = rng.integers(0, ng, size=(N_BOOT, ng))
        means = gated[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        boot_means[f] = means
        out["by_cutoff"][f"{f:.2f}"] = {
            "ci_lo": float(lo), "ci_hi": float(hi),
            "half_width": float((hi - lo) / 2.0),
        }

    # Paired resample of the whole OOS frame for the shape-test differences.
    idx = rng.integers(0, n, size=(N_BOOT, n))
    preds_b = preds[idx]
    rets_b = rets[idx]
    qs = np.quantile(preds_b, [1.0 - f for f in CUTOFFS], axis=1)  # (3, N_BOOT)
    mask = preds_b[None, :, :] >= qs[:, :, None]
    vals = np.where(mask, rets_b[None, :, :], np.nan)
    means_b = np.nanmean(vals, axis=2)  # (3, N_BOOT)
    d1 = means_b[0] - means_b[1]  # top10% - top20%
    d2 = means_b[1] - means_b[2]  # top20% - top30%
    out["shape"] = {
        "diff_10_vs_20": {"mean": float(d1.mean()),
                          "ci_lo": float(np.percentile(d1, 2.5)),
                          "ci_hi": float(np.percentile(d1, 97.5)),
                          "p_positive": float((d1 > 0).mean())},
        "diff_20_vs_30": {"mean": float(d2.mean()),
                          "ci_lo": float(np.percentile(d2, 2.5)),
                          "ci_hi": float(np.percentile(d2, 97.5)),
                          "p_positive": float((d2 > 0).mean())},
    }
    return out


def judge(mean: float | None, lo: float | None, hi: float | None) -> str:
    """Plan §4 success criteria at the headline cutoff."""
    if mean is None or lo is None or hi is None:
        return "INCONCLUSIVE"
    if hi < 0:
        return "FAIL"
    if mean > 0 and lo > 0 and (hi - lo) / 2.0 <= 0.10:
        return "PASS"
    return "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# figures beyond the standard set
# ---------------------------------------------------------------------------


def fig_threshold_ladder(arms: dict, path: Path) -> Path:
    """Gated mean vs selection fraction per arm, bootstrap 95% CI bars, with
    each arm's ungated OOS base at fraction 1.0. Diagnostic only — the ladder
    is judged at the three pre-registered cutoffs, never read off this curve."""
    from engine.report import _matplotlib

    plt = _matplotlib()
    names = [a for a in arms if arms[a]["scored_n"]]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = {"a_thru": "tab:blue", "a_runup": "tab:cyan",
              "b_thru": "tab:orange", "b_runup": "tab:gray"}
    xs_base = [0.10, 0.20, 0.30]
    data = {}
    for name in names:
        arm = arms[name]
        means, los, his = [], [], []
        for f in CUTOFFS:
            cell = arm["ladder"][f"{f:.2f}"]
            ci = arm["bootstrap"]["by_cutoff"][f"{f:.2f}"]
            means.append(cell["gated_mean"])
            los.append(ci["ci_lo"])
            his.append(ci["ci_hi"])
        base = arm["base_oos_mean"]
        xs = xs_base + [1.0]
        ys = means + [base]
        yerr_lo = [m - lo for m, lo in zip(means, los)] + [0.0]
        yerr_hi = [hi - m for m, hi in zip(means, his)] + [0.0]
        ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi],
                    fmt="o-", color=colors.get(name, "k"), capsize=3, lw=1.2,
                    label=f"{name} (n={arm['scored_n']})")
        data[name] = {"fraction": xs, "gated_mean": ys,
                      "ci_lo": los + [None], "ci_hi": his + [None]}
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(HEADLINE, color="tab:red", lw=0.9, ls="--", alpha=0.7,
               label="headline cutoff 20%")
    ax.set_xlabel("selection fraction (top X% of pooled OOS scores; 1.0 = ungated OOS base)")
    ax.set_ylabel("mean return / trade")
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:+.0%}")
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.xaxis.set_major_formatter(lambda v, _pos: f"{v:.0%}")
    ax.set_title("Threshold ladder: gated mean vs selection fraction (bootstrap 95% CI)")
    ax.legend(fontsize=7, loc="best")
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.01, 0.02, "Diagnostic only: judged at the three pre-registered cutoffs, "
                         "never read off this curve.", fontsize=7, color="gray")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(data, indent=1, default=str))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# report helpers
# ---------------------------------------------------------------------------


def fmt_pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def fmt_ci(cell: dict) -> str:
    if cell.get("ci_lo") is None:
        return "n/a"
    return f"[{cell['ci_lo'] * 100:+.1f}%, {cell['ci_hi'] * 100:+.1f}%]"


def champion_context() -> dict:
    out = {}
    for entry in load_registry().entries:
        if entry.role == "gate" and entry.strategy in ("STR-THRU", "STR-RUNUP"):
            out[entry.strategy] = {
                "id": entry.id,
                "n": entry.eval.get("n"),
                "gate_lift": entry.eval.get("gate_lift"),
                "gated_mean_ret": entry.eval.get("gated_mean_ret"),
                "base_mean_ret": entry.eval.get("base_mean_ret"),
                "gated_win_rate": entry.eval.get("gated_win_rate"),
                "threshold": entry.eval.get("threshold"),
            }
    return out


def main() -> int:
    started = time.time()
    spec = lib.load_spec(HERE / "spec.yaml")
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    # Pre-registration is enforced here, same call evaluate() makes.
    prereg = check_preregistration(spec, HERE)
    log(f"preregistration valid: stamp {prereg.get('preregistered_at')}")

    computed_names = computed_moves_names()
    log(f"computed-moves universe: {len(computed_names)} names")
    log("loading panel once for all arms")
    panel = load_panel()
    spy = common.load_spy_daily()

    trades = {}
    datasets = {}
    slices = {}
    for strategy in ("STR-THRU", "STR-RUNUP"):
        log(f"loading {strategy} replayed trades")
        t = common.load_engine_trades(strategy)
        trades[strategy] = t
        ds = build_gate_dataset(strategy, t, panel)
        datasets[strategy] = ds
        mask = slice_mask(ds, computed_names)
        slices[strategy] = mask
        n_small = int((mcap_bucket_of(ds) == "<1B").sum())
        n_comp = int(ds["ticker"].isin(computed_names).sum())
        log(f"{strategy}: dataset {len(ds):,} rows; slice {int(mask.sum()):,} "
            f"(<1B {n_small}, computed-moves {n_comp})")

    # ---- the four arms ----------------------------------------------------
    arms: dict = {}
    rng = np.random.default_rng(SEED)
    for name, strategy, train_universe, fty, mtr, status in ARMS:
        log(f"arm {name}: {strategy} train={train_universe} "
            f"first_test_year={fty} min_train_rows={mtr}")
        ds = datasets[strategy]
        smask = slices[strategy]
        if train_universe == "slice":
            frame = ds[smask].copy()
            test_mask = pd.Series(True, index=frame.index)
        else:
            frame = ds
            test_mask = smask
        scored, diagnostics, audit, n_complete = walk_forward_oos(
            frame, test_mask, first_test_year=fty, min_train_rows=mtr)
        ladder = ladder_stats(scored)
        boot = bootstrap_cis(scored, rng)
        # Fill the fold diagnostics now that the pooled threshold exists:
        # per-year selection at the headline cutoff (what the funnel sums).
        thr_head = ladder[f"{HEADLINE:.2f}"]["threshold"]
        for d in diagnostics:
            if d.get("n_selected") is None:
                yr = d["year"]
                d["n_selected"] = int(((scored["year"] == yr)
                                       & (scored["pred"] >= thr_head)).sum())
        head = ladder[f"{HEADLINE:.2f}"]
        ci = boot["by_cutoff"][f"{HEADLINE:.2f}"]
        verdict = judge(head["gated_mean"], ci.get("ci_lo"), ci.get("ci_hi"))
        m10 = ladder["0.10"]["gated_mean"]
        m20 = ladder["0.20"]["gated_mean"]
        m30 = ladder["0.30"]["gated_mean"]
        monotone = (m10 is not None and m20 is not None and m30 is not None
                    and m10 > m20 > m30)
        arms[name] = {
            "strategy": strategy, "status": status, "train_universe": train_universe,
            "first_test_year": fty, "min_train_rows": mtr,
            "n_dataset_slice": int(smask.sum()),
            "n_complete": n_complete,
            "scored_n": int(len(scored)),
            "scored_years": sorted(int(y) for y in scored["year"].unique()) if len(scored) else [],
            "base_oos_mean": float(scored["ret"].mean()) if len(scored) else None,
            "base_oos_win": float((scored["ret"] > 0).mean()) if len(scored) else None,
            "ladder": ladder,
            "bootstrap": boot,
            "headline_verdict": verdict,
            "monotone_ladder": bool(monotone),
            "diagnostics": diagnostics,
            "audit": audit,
            "scored_frame": scored,
        }
        log(f"arm {name}: OOS n={len(scored)} base {fmt_pct(arms[name]['base_oos_mean'])} "
            f"-> gated(20%) {fmt_pct(head['gated_mean'])} n={head['n_passed']} "
            f"CI {fmt_ci(ci)} verdict={verdict} monotone={monotone}")

    # ---- decision rule (plan §5), judged at the headline cutoff -----------
    if arms["a_thru"]["headline_verdict"] == "PASS":
        outcome = "outcome_1"
    elif arms["b_thru"]["headline_verdict"] == "PASS":
        outcome = "outcome_2"
    else:
        outcome = "outcome_3"
    decision_text = spec.get("decision_rule", {}).get(outcome, "")
    log(f"decision rule: {outcome} — {decision_text}")

    # ---- standard harness artifacts for the primary arm -------------------
    primary = arms["a_thru"]
    strategy = primary["strategy"]
    t_all = trades[strategy]
    ds = datasets[strategy]
    smask = slices[strategy]
    slice_ids = set(ds.loc[smask, "event_id"])
    slice_trades = t_all[t_all["event_id"].isin(slice_ids)].copy()
    gated_ids = set(primary["ladder"][f"{HEADLINE:.2f}"]["gated_event_ids"])
    selected = slice_trades[slice_trades["event_id"].isin(gated_ids)].copy()
    selected = selected.sort_values(["entry_date", "exit_date"], kind="stable")
    mid_sel = selected[np.isclose(selected["fill_alpha"], 0.5)]
    mid_slice = slice_trades[np.isclose(slice_trades["fill_alpha"], 0.5)]

    sha = spec_hash(spec)
    eq5 = build_equity(mid_sel, 0.05, mode="cashflow",
                       max_deployed=spec.get("max_deployed_fraction"), record=True)
    sweep_sel = alpha_sweep(selected)
    sweep_slice = alpha_sweep(slice_trades)
    headline = trade_stats(mid_sel["ret"].to_numpy(), mid_sel["event_date"]) if len(mid_sel) else trade_stats([])
    headline["by_year"] = by_year_table(mid_sel)
    headline["breakeven_alpha"] = breakeven_alpha_from_sweep(sweep_sel) if len(selected) else None
    headline["sharpe_equity"] = sharpe_equity(eq5["equity"])
    headline["max_dd"] = eq5["max_dd"]
    headline["max_concurrency"] = eq5["max_concurrency"]
    headline["deployment"] = {
        "peak": eq5["peak_deployment"], "worst_cash": eq5["worst_cash"],
        "cap": spec.get("max_deployed_fraction"),
        "constrained_entries": eq5["constrained_entries"],
    }
    headline["alpha_sweep"] = sweep_sel
    headline["capacity"] = capacity_note(selected)
    headline["dollar_weighted"] = dollar_weighted_return(mid_sel)
    headline["ungated_share"] = 0.0  # pre-2024 rows are training-only under arm A, never traded
    headline["base_unselected"] = {
        "n": int(len(mid_slice)),
        "mean": float(mid_slice["ret"].mean()) if len(mid_slice) else float("nan"),
        "win_rate": float((mid_slice["ret"] > 0).mean()) if len(mid_slice) else float("nan"),
    }
    headline["mc"] = {}

    results: dict = {
        "spec_id": spec.get("id"),
        "spec_hash": sha,
        "preregistration": prereg,
        "equity_mode": "cashflow",
        "elapsed_s": 0.0,
        "backtest": {
            "alpha_sweep": sweep_slice,
            "breakeven_alpha": breakeven_alpha_from_sweep(sweep_slice),
            "by_year": by_year_table(mid_slice),
            "n_events": int(slice_trades["event_id"].nunique()) if len(slice_trades) else 0,
        },
        "walk_forward": {
            "diagnostics": primary["diagnostics"],
            "audit": primary["audit"],
            "headline_stage": "wf_oos",
        },
    }

    proba = primary["scored_frame"]["proba"].to_numpy(dtype=float)
    wins = (primary["scored_frame"]["ret"].to_numpy(dtype=float) > 0).astype(float)
    results["calibration"] = calibration_block(proba, wins)

    results["mc"] = monte_carlo(
        mid_sel["ret"].to_numpy(dtype=float), fractions=SIZING_FRACTIONS,
        block=MC_BLOCK, paths=MC_PATHS, seed=0, mode="cashflow",
        draw_order=str(spec.get("mc_draw_order", "shared")),
        path_bands_for=(0.05,) if 0.05 in SIZING_FRACTIONS else (),
    )
    headline["mc"] = results["mc"]["by_fraction"]

    results["stress"] = {
        "regimes": stress_regimes(selected, spy_daily=spy),
        "iv_regime": stress_iv_regime(selected, spy_daily=spy),
        "tail_injection": {"available": False, "required": False,
                           "note": "the structure carries no short leg"},
        "slippage": {"available": False,
                     "note": "no repricer supplied; plan §7 restricts this experiment to compute-only"},
        "stale_dates": {"available": False,
                        "note": "no repricer supplied; plan §7 restricts this experiment to compute-only"},
    }

    results["equity_curves"] = {
        f"{f:.2f}": {"final": build_equity(
            mid_sel, f, mode="cashflow",
            max_deployed=spec.get("max_deployed_fraction"))["final"]}
        for f in SIZING_FRACTIONS
    }
    results["equity_curve_series"] = {
        "date": [str(ts.date()) for ts in eq5["equity"].index],
        "equity": [float(v) for v in eq5["equity"].values],
    }
    if len(mid_sel):
        tx = transaction_log(mid_sel, eq5, scores=None)
        results["transaction_log"] = reconcile_transaction_log(tx, eq5)
        log_path = RESULTS / f"transactions_{sha[:12]}.csv"
        tx.to_csv(log_path, index=False)
        results["transaction_log"]["path"] = str(log_path.relative_to(paths.ROOT))
        from engine.evaluate import _file_sha256
        results["transaction_log"]["sha256"] = _file_sha256(log_path)
    else:
        results["transaction_log"] = {"rows": 0, "reconciles": True,
                                      "note": "no selected trades to log"}
    results["headline"] = headline
    results["headline_stage"] = "wf_oos"

    # ---- extra figures: ungated baseline equity, B-thru equity, ladder ----
    eq_base = build_equity(mid_slice.sort_values(["entry_date", "exit_date"], kind="stable"),
                           0.05, mode="cashflow", max_deployed=1.0)
    fig_equity(eq_base["equity"], FIGURES / "equity_ungated_slice.png",
               "Ungated small-name slice, all years (5% sizing) — the base exposure")
    b_sel_ids = set(arms["b_thru"]["ladder"][f"{HEADLINE:.2f}"]["gated_event_ids"])
    b_mid = mid_slice[mid_slice["event_id"].isin(b_sel_ids)].sort_values(
        ["entry_date", "exit_date"], kind="stable")
    if len(b_mid):
        eq_b = build_equity(b_mid, 0.05, mode="cashflow", max_deployed=1.0)
        fig_equity(eq_b["equity"], FIGURES / "equity_b_thru_gated.png",
                   "Arm B-thru: full-universe gate scoring slice rows, top-20% (5% sizing)")
    fig_threshold_ladder(arms, FIGURES / "threshold_ladder.png")
    log("figures written")

    # ---- extra report sections --------------------------------------------
    def fold_caveats() -> list[str]:
        """Degenerate or skipped folds, stated rather than hidden: a fold at
        the min_train_rows floor can collapse the HistGBM fit to a constant,
        and a pooled quantile threshold then selects nothing from it."""
        notes = []
        for name, arm in arms.items():
            sf = arm["scored_frame"]
            if len(sf):
                deg = [int(y) for y, g in sf.groupby("year")
                       if float(g["pred"].std(ddof=0)) < 1e-10]
                if deg:
                    notes.append(
                        f"**{name}**: fold(s) {', '.join(map(str, deg))} produced CONSTANT "
                        f"predictions ({arm['min_train_rows']}-row training sets collapse the "
                        "HistGBM fit); those rows are scored but none can clear a pooled "
                        "quantile threshold, so the arm's selection effectively starts the "
                        "following year.")
            skipped = [str(d["year"]) for d in arm["diagnostics"] if not d["fitted"]]
            if skipped:
                notes.append(
                    f"**{name}**: year(s) {', '.join(skipped)} not fitted — train rows below "
                    f"min_train_rows={arm['min_train_rows']} ({'; '.join(d['reason'] for d in arm['diagnostics'] if not d['fitted'])}).")
        return notes

    def extra_sections(result) -> list[dict]:
        h = result.results["headline"]
        a = arms["a_thru"]
        head_cell = a["ladder"][f"{HEADLINE:.2f}"]
        head_ci = a["bootstrap"]["by_cutoff"][f"{HEADLINE:.2f}"]
        sections: list[dict] = []

        arm_rows = []
        for name, arm in arms.items():
            hc = arm["ladder"][f"{HEADLINE:.2f}"]
            ci = arm["bootstrap"]["by_cutoff"][f"{HEADLINE:.2f}"]
            arm_rows.append([
                name, arm["status"], arm["strategy"],
                f"{arm['train_universe']} -> {arm['first_test_year']}+",
                f"{arm['scored_n']:,}", f"{hc['n_passed']:,}",
                fmt_pct(hc["gated_mean"]), fmt_ci(ci),
                f"{hc['gated_win']:.3f}" if hc["gated_win"] is not None else "n/a",
                arm["headline_verdict"],
            ])
        sections.append({
            "title": "Arm verdicts at the headline top-20% cutoff (pre-registered)",
            "note": (
                f"Primary arm a_thru: pooled OOS gated mean {fmt_pct(head_cell['gated_mean'])} "
                f"on {head_cell['n_passed']:,} of {a['scored_n']:,} slice trades, bootstrap 95% CI "
                f"{fmt_ci(head_ci)} -> **{a['headline_verdict']}** (criteria: gated mean > 0, CI "
                "excludes zero, CI half-width <= 10pp). Verdicts are judged at the headline "
                "cutoff only; the other two cutoffs are pre-registered secondaries, and no "
                "outcome triggers trying more cutoffs until one passes."),
            "columns": ["arm", "status", "strategy", "train -> OOS from", "n OOS",
                        "n passed", "gated mean", "bootstrap 95% CI", "gated win", "verdict"],
            "align": ["---", "---", "---", "---", "---:", "---:", "---:", "---", "---:", "---"],
            "rows": arm_rows,
            "body": fold_caveats(),
            "promote_to_verdict": True,
            "verdict_row": (
                "Can the gate make money on the small-name slice?",
                f"**{a['headline_verdict']}** — a_thru top-20% gated mean "
                f"{fmt_pct(head_cell['gated_mean'])} (CI {fmt_ci(head_ci)}, "
                f"n={head_cell['n_passed']:,}); decision rule -> {outcome}"),
        })

        ladder_rows = []
        for name, arm in arms.items():
            for f in CUTOFFS:
                cell = arm["ladder"][f"{f:.2f}"]
                ci = arm["bootstrap"]["by_cutoff"][f"{f:.2f}"]
                ladder_rows.append([
                    name, f"{f:.0%}", f"{cell['threshold']:.4f}", f"{cell['n_passed']:,}",
                    fmt_pct(cell["gated_mean"]), fmt_ci(ci),
                    f"{cell['gated_win']:.3f}" if cell["gated_win"] is not None else "n/a",
                ])
        shape_lines = []
        for name, arm in arms.items():
            sh = arm["bootstrap"]["shape"]
            d1, d2 = sh["diff_10_vs_20"], sh["diff_20_vs_30"]
            shape_lines.append(
                f"**{name}**: ladder monotone (10% > 20% > 30%): "
                f"**{'yes' if arm['monotone_ladder'] else 'no'}** — "
                f"mean(10%)−mean(20%) {fmt_pct(d1['mean'])} "
                f"(CI {fmt_pct(d1['ci_lo'])} to {fmt_pct(d1['ci_hi'])}, "
                f"P>0 = {d1['p_positive']:.2f}); mean(20%)−mean(30%) {fmt_pct(d2['mean'])} "
                f"(CI {fmt_pct(d2['ci_lo'])} to {fmt_pct(d2['ci_hi'])}, "
                f"P>0 = {d2['p_positive']:.2f}).")
        sections.append({
            "title": "Threshold ladder and shape test (secondary, pre-registered)",
            "note": ("All three cutoffs are read off ONE walk-forward run per arm — same "
                     "pooled OOS scores, three cutoffs — so comparing them is not multiple "
                     "fitting. The finer curve in `figures/threshold_ladder.png` is diagnostic "
                     "and is never judged."),
            "columns": ["arm", "cutoff", "threshold", "n passed", "gated mean",
                        "bootstrap 95% CI", "gated win"],
            "align": ["---", "---:", "---:", "---:", "---:", "---", "---:"],
            "rows": ladder_rows,
            "body": shape_lines + [
                "",
                "Ladder figure: `figures/threshold_ladder.png`.",
            ],
        })

        year_rows = []
        for name, arm in arms.items():
            base_by_year = {int(y): float(g["ret"].mean())
                            for y, g in arm["scored_frame"].groupby("year")}
            n_by_year = {int(y): int(len(g))
                         for y, g in arm["scored_frame"].groupby("year")}
            for year in sorted(set(base_by_year) | {r["year"] for r in arm["ladder"][f"{HEADLINE:.2f}"]["by_year"]}):
                gated_row = next((r for r in arm["ladder"][f"{HEADLINE:.2f}"]["by_year"]
                                  if r["year"] == year), {})
                year_rows.append([
                    name, str(year), f"{n_by_year.get(year, 0):,}",
                    fmt_pct(base_by_year.get(year)),
                    f"{gated_row.get('n_passed', 0):,}",
                    fmt_pct(gated_row.get("gated_mean")),
                    (f"{gated_row['gated_win']:.3f}"
                     if gated_row.get("gated_win") is not None else "n/a"),
                ])
        sections.append({
            "title": "Per-year gated means at the headline cutoff (display only)",
            "note": ("No per-year selection is made or judged anywhere in this experiment; "
                     "the year split exists to show where the pooled result comes from."),
            "columns": ["arm", "year", "n OOS", "base mean", "n passed",
                        "gated mean", "gated win"],
            "align": ["---", "---:", "---:", "---:", "---:", "---:", "---:"],
            "rows": year_rows,
        })

        max_rows = []
        for name, arm in arms.items():
            for f in CUTOFFS:
                cell = arm["ladder"][f"{f:.2f}"]
                mt = cell["max_trade"]
                if not mt.get("n"):
                    continue
                share = mt.get("share_of_mean")
                max_rows.append([
                    name, f"{f:.0%}", f"{mt['n']:,}",
                    f"{mt['largest_trade']} ({mt['largest_ret'] * 100:+.0f}%)",
                    fmt_pct(mt["contribution_to_mean"]),
                    f"{share:.0%}" if share is not None else "n/a",
                    fmt_pct(mt["mean_without_it"]),
                ])
        sections.append({
            "title": "Max single-trade contribution to each gated mean",
            "note": ("Plan §3: 2024's +35.4% THRU base warns of fat tails. A gated mean "
                     "carried by one trade is not a strategy."),
            "columns": ["arm", "cutoff", "n", "largest trade", "its contribution to the mean",
                        "share of the mean", "gated mean without it"],
            "align": ["---", "---:", "---:", "---", "---:", "---:", "---:"],
            "rows": max_rows,
        })

        champ = champion_context()
        champ_rows = [[s, b["id"], f"{b['n']:,}", fmt_pct(b["gate_lift"]),
                       fmt_pct(b["gated_mean_ret"]),
                       f"{b['gated_win_rate']:.3f}" if b["gated_win_rate"] is not None else "n/a"]
                      for s, b in sorted(champ.items())]
        base_rows = []
        for name, arm in arms.items():
            base_rows.append([
                f"{name} ungated OOS slice", f"{arm['scored_n']:,}",
                fmt_pct(arm["base_oos_mean"]),
                f"{arm['base_oos_win']:.3f}" if arm["base_oos_win"] is not None else "n/a",
            ])
        base_rows.append([
            "ungated slice, full span (mid, all years)",
            f"{int(len(mid_slice)):,}",
            fmt_pct(float(mid_slice['ret'].mean())) if len(mid_slice) else "n/a",
            f"{float((mid_slice['ret'] > 0).mean()):.3f}" if len(mid_slice) else "n/a",
        ])
        sections.append({
            "title": "Ungated baselines (reported, not tested)",
            "note": ("The ungated slice mean is the exposure the gate is supposed to "
                     "select; champion registry context follows in the next section."),
            "columns": ["baseline", "n", "mean/trade", "win rate"],
            "align": ["---", "---:", "---:", "---:"],
            "rows": base_rows,
            "falsifies": "the ungated slice base turning negative on a data refresh — "
                         "the exposure is the asset the gate is supposed to select.",
        })
        sections.append({
            "title": "Champion gates (registry.json, context only)",
            "note": ("Measured on a different row set (the pre-slice OOS windows); never "
                     "in the same comparison column as the arm verdicts."),
            "columns": ["strategy", "champion", "n", "gate lift", "gated mean", "gated win"],
            "align": ["---", "---", "---:", "---:", "---:", "---:"],
            "rows": champ_rows,
        })

        sections.append({
            "title": f"Decision rule outcome: **{outcome}**",
            "note": ("Chosen before results (plan §5), so the results cannot choose it. "
                     "Judged at the headline cutoff verdicts only."),
            "body": [f"- {spec['decision_rule'][k]}" for k in ("outcome_1", "outcome_2", "outcome_3")]
                    + ["", f"**This run: {outcome}** — {decision_text}"],
        })

        cal = result.results.get("calibration") or {}
        sections.append({
            "title": "Provenance of the P(win) reliability block (§6)",
            "body": [
                "The champion decision rule is a quantile threshold on PREDICTED RETURNS; it "
                "uses no probabilities. The §6 reliability diagram comes from an in-fold "
                "isotonic P(win) map fitted on each walk-forward fold's training rows — the "
                "EXP-105 registered-gate convention — added reporting-only; nothing in the "
                "selection rule or the success criteria touches it.",
                (f"Brier skill {cal.get('brier_skill'):.3f}, n={cal.get('n')}"
                 if cal.get("available") else "Calibration unavailable on this arm."),
            ],
        })
        return sections

    input_files = sorted((paths.CURATED / "trades").glob("year=*/part-*.parquet"))
    result = EvalResult(spec=spec, results=results, run_dir=HERE)
    report = Report.from_eval(result, input_files=input_files,
                              extra_sections=extra_sections(result))
    report_path = report.write(HERE)
    results["elapsed_s"] = round(time.time() - started, 1)

    # ---- artifacts ---------------------------------------------------------
    (RESULTS / f"metrics_{sha[:12]}.json").write_text(
        json.dumps(results, indent=1, default=str))
    slim_arms = {}
    for name, arm in arms.items():
        slim = {k: v for k, v in arm.items() if k != "scored_frame"}
        slim["ladder"] = {f: {kk: vv for kk, vv in cell.items() if kk != "gated_event_ids"}
                          for f, cell in arm["ladder"].items()}
        slim_arms[name] = slim
    (RESULTS / "metrics.json").write_text(json.dumps({
        "spec_id": spec.get("id"),
        "spec_hash": sha,
        "snapshot": spec.get("data_snapshot"),
        "arms": slim_arms,
        "decision_rule": {"outcome": outcome, "text": decision_text},
        "champion_context": champion_context(),
        "elapsed_s": results["elapsed_s"],
    }, indent=1, default=str))
    for name, arm in arms.items():
        sf = arm["scored_frame"]
        keep = [c for c in ("event_id", "ticker", "year", "ret", "pred", "proba") if c in sf.columns]
        sf[keep].to_csv(RESULTS / f"scores_{name}.csv", index=False)
        gated_ids_arm = set(arm["ladder"][f"{HEADLINE:.2f}"]["gated_event_ids"])
        gated_rows = sf[sf["event_id"].isin(gated_ids_arm)]
        gated_rows[keep].to_csv(RESULTS / f"gated_trades_{name}_top20.csv", index=False)
        by = []
        for year, grp in sf.groupby("year", sort=True):
            g = grp[grp["pred"] >= arm["ladder"][f"{HEADLINE:.2f}"]["threshold"]]
            by.append({"year": int(year), "n": int(len(grp)),
                       "base_mean": float(grp["ret"].mean()),
                       "n_passed": int(len(g)),
                       "gated_mean": float(g["ret"].mean()) if len(g) else None,
                       "gated_win": float((g["ret"] > 0).mean()) if len(g) else None})
        pd.DataFrame(by).to_csv(RESULTS / f"by_year_{name}.csv", index=False)

    append_run_log(HERE, {
        "ts": pd.Timestamp.now(tz="UTC").isoformat(),
        "spec_id": spec.get("id"),
        "spec_hash": sha,
        "n_events": results["backtest"]["n_events"],
        "headline_mean_mid": headline.get("mean"),
        "sharpe_trade": headline.get("sharpe_trade"),
        "stage": "ran",
    })
    lib.record_evaluation(HERE, spec, results)

    print()
    print(f"{'arm':10s} {'status':34s} {'n OOS':>6} {'base':>8} {'gated20':>8} "
          f"{'CI95':>20}  verdict")
    for name, arm in arms.items():
        hc = arm["ladder"][f"{HEADLINE:.2f}"]
        ci = arm["bootstrap"]["by_cutoff"][f"{HEADLINE:.2f}"]
        print(f"{name:10s} {arm['status']:34s} {arm['scored_n']:>6,} "
              f"{fmt_pct(arm['base_oos_mean']):>8} {fmt_pct(hc['gated_mean']):>8} "
              f"{fmt_ci(ci):>20}  {arm['headline_verdict']}")
    print(f"\ndecision rule: {outcome}")
    print(f"report: {report_path}")
    print(f"elapsed: {results['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
