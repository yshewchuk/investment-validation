#!/usr/bin/env python3
"""Phase 2 acceptance tests.

    python3 checks/phase2_checks.py               # everything
    python3 checks/phase2_checks.py --list
    python3 checks/phase2_checks.py --only synthetic_known
    python3 checks/phase2_checks.py --no-data     # skip checks needing the store

The guide's six acceptance tests:

1. ``synthetic_known`` — trades from a known distribution reproduce the
   analytic metrics and MC P(loss) within tolerance.
2. ``harness_regression`` — LOAD-BEARING: the EXP-050 configuration re-run
   through ``engine.evaluate``; trade count, per-year returns, equity, and MC
   must reproduce. Any divergence is root-caused and written to
   ``reports/phase2_exp050_regression.md`` before new experiments run.
3. ``preregistration`` — missing or back-dated stamps refuse the OOS stage.
4. ``wf_leak_poison`` — a spy gate proves fit never sees the test year.
5. ``ledger_append_only`` — the ledger grows by prefix only; no rewrite path.
6. ``promotion_dry_run`` — synthetic challenger better/worse → promote/refuse.

Unit-level behaviour lives in ``tests/`` and runs here as check 0.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.evaluate import (  # noqa: E402
    Gate,
    PreregistrationError,
    append_run_log,
    build_equity,
    check_preregistration,
    evaluate,
    monte_carlo,
    trade_stats,
    walk_forward,
)
from experiments import lib  # noqa: E402
from experiments import promote as promote_mod  # noqa: E402

#: EXP-050 reference artifacts (grandfathered, read-only).
OPF = paths.EP_OPF
EXP050_RESULT = OPF / "results" / "exp050_equity.json"
EXP045C_TRADES = OPF / "results" / "exp045c_short_trades.csv"
MODEL_ROWS_J14 = OPF / "data" / "derived" / "model_rows_j14.csv"

#: The feature list the EXP-049/050 gate trained on, verbatim.
EXP049_FEATS = [
    "im_j", "im_d1", "im_d5", "im_d10", "iv10d", "iv30d", "iv30d_z",
    "exErnIv10d", "exErnIv30d", "ieeEarnEffect", "skewing", "contango",
    "fwd90_30", "fexErn90_30", "rVol30", "ret5", "ret10", "ret20",
    "dist_high", "log_mcap", "spy_ret21", "spy_vol20", "spy_dd252",
    "month", "dow", "pe_ema_runup10", "pe_runner_rate",
    "pe_ema_im10", "pe_ema_im5", "pe_ema_im1",
]


@dataclass
class CheckOutcome:
    name: str
    passed: bool
    detail: str = ""
    elapsed_s: float = 0.0
    skipped: bool = False


REGISTRY: dict[str, dict] = {}


def check(name: str, *, needs_data: bool = True, description: str = ""):
    def wrap(fn):
        REGISTRY[name] = {"fn": fn, "needs_data": needs_data, "description": description}
        return fn

    return wrap


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# --------------------------------------------------------------------------
# 0. unit suite
# --------------------------------------------------------------------------


@check("unittests", needs_data=False, description="the pytest suite (pure logic)")
def check_unittests() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "--no-header"],
        capture_output=True, text=True, cwd=ROOT,
    )
    tail = (result.stdout or result.stderr).strip().splitlines()[-1:]
    _require(result.returncode == 0, f"pytest failed: {' '.join(tail)}")
    return " ".join(tail)


# --------------------------------------------------------------------------
# 1. synthetic-known test
# --------------------------------------------------------------------------


def _synthetic_trades(n_years: int = 50, per_year: int = 6, mean: float = 0.02,
                      std: float = 0.10, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    rng = np.random.default_rng(seed)
    n = n_years * per_year
    rets = rng.normal(mean, std, n)
    dates = pd.date_range("2020-01-01", periods=n, freq=f"{365 // per_year}D")
    frame = pd.DataFrame({
        "event_id": [f"S{i}" for i in range(n)],
        "ticker": "SYN",
        "event_date": dates,
        "entry_date": dates - pd.Timedelta(days=1),
        "exit_date": dates + pd.Timedelta(days=1),
        "fill_alpha": 0.5,
        "entry_cost": 1.0,
        "exit_value": 1.0 + rets,
        "ret": rets,
    })
    return frame, rets, dates


@check("synthetic_known", needs_data=False,
       description="metrics and MC reproduce a known distribution analytically")
def check_synthetic_known() -> str:
    mean, std, per_year = 0.02, 0.10, 6
    trades, rets, dates = _synthetic_trades()
    n = len(rets)

    stats = trade_stats(rets, dates)
    se_mean = std / math.sqrt(n)
    _require(abs(stats["mean"] - mean) < 3.5 * se_mean,
             f"mean {stats['mean']:.4f} off the known {mean} by >3.5 SE ({se_mean:.4f})")

    # Analytic win rate for N(mean, std): Phi(mean/std).
    win_true = 0.5 * (1 + math.erf(mean / std / math.sqrt(2)))
    se_win = math.sqrt(win_true * (1 - win_true) / n)
    _require(abs(stats["win_rate"] - win_true) < 3.5 * se_win,
             f"win rate {stats['win_rate']:.3f} off analytic {win_true:.3f}")

    # Analytic annualized Sharpe: (mean/std) * sqrt(trades/year).
    sharpe_true = mean / std * math.sqrt(per_year)
    se_sharpe = math.sqrt((1 + 0.5 * sharpe_true ** 2) / n)
    _require(abs(stats["sharpe_trade"] - sharpe_true) < 3.5 * se_sharpe,
             f"sharpe_trade {stats['sharpe_trade']:.3f} off analytic {sharpe_true:.3f}")

    # MC correctness, three ways:
    # (i) with block=2 the bootstrap is near-iid, so it must match a direct
    #     iid simulation of the same statistic;
    # (ii) with block=20 it must react to earnings-week clustering, which an
    #      iid resample misses (the whole point of the block bootstrap);
    # (iii) block=20 must match an independent block simulator of the same
    #      definition within MC noise.
    # The reference distribution is zero-mean: P(loss) sits near 0.5 there, so
    # the comparisons are actually sensitive.
    _, rets0, _ = _synthetic_trades(mean=0.0, seed=43)
    # The iid reference draws from the EMPIRICAL measure (sample mean/std),
    # because that is what the bootstrap resamples — comparing against the
    # true N(0, std) would conflate sampling noise of the mean with a bug.
    mc_small = monte_carlo(rets0, fractions=(0.05,), block=2, paths=1000, seed=0)
    p_loss_b2 = mc_small["by_fraction"]["0.05"]["p_loss"]
    rng = np.random.default_rng(1)
    finals = []
    for _ in range(2000):
        path = rng.normal(float(rets0.mean()), float(rets0.std(ddof=1)), len(rets0))
        finals.append(float(np.prod(1 + 0.05 * path)))
    p_loss_iid = float(np.mean(np.array(finals) < 1.0))
    _require(abs(p_loss_b2 - p_loss_iid) < 0.05,
             f"block-2 P(loss) {p_loss_b2:.3f} vs iid simulation {p_loss_iid:.3f} (>0.05 apart)")

    clustered = rets0.copy()
    for start in range(30, len(clustered) - 4, 60):
        clustered[start:start + 3] = -0.10  # a bad earnings week, repeated
    mc_clust = monte_carlo(clustered, fractions=(0.05,), block=20, paths=1500, seed=0)
    mc_unclust = monte_carlo(clustered, fractions=(0.05,), block=2, paths=1500, seed=0)
    s_clust = mc_clust["by_fraction"]["0.05"]
    s_unclust = mc_unclust["by_fraction"]["0.05"]
    # Clustering fattens the lower tail: block=20 must read a worse p05 (and no
    # better P(loss)) than the near-iid block=2 on the same series.
    _require(s_clust["terminal_p05"] < s_unclust["terminal_p05"],
             f"block=20 lower tail not heavier: p05 {s_clust['terminal_p05']:.3f} "
             f"vs block-2 {s_unclust['terminal_p05']:.3f}")
    _require(s_clust["p_loss"] >= s_unclust["p_loss"] - 0.01,
             f"block=20 P(loss) {s_clust['p_loss']:.3f} implausibly better than "
             f"block-2 {s_unclust['p_loss']:.3f}")

    def independent_block_ploss(rets_seq: np.ndarray, block: int, paths: int,
                                seed: int, f: float) -> float:
        r = np.random.default_rng(seed)
        n = len(rets_seq)
        losses = 0
        for _ in range(paths):
            idx: list[int] = []
            while len(idx) < n:
                s = int(r.integers(0, n - block))
                idx.extend(range(s, s + block))
            if float(np.prod(1 + f * rets_seq[np.array(idx[:n])])) < 1.0:
                losses += 1
        return losses / paths

    p_ref = independent_block_ploss(clustered, 20, 1000, 7, 0.05)
    p_harness = monte_carlo(clustered, fractions=(0.05,), block=20, paths=1000,
                            seed=7)["by_fraction"]["0.05"]["p_loss"]
    _require(abs(p_ref - p_harness) < 0.04,
             f"block-20 P(loss) {p_harness:.3f} vs independent simulator {p_ref:.3f}")

    # Deterministic equity sanity: median MC terminal brackets the analytic
    # compounded growth exp(n * E[ln(1 + f r)]) on the positive-mean set.
    mc_pos = monte_carlo(rets, fractions=(0.05,), block=20, paths=500, seed=0)
    log_growth = float(np.mean(np.log(1 + 0.05 * rets)))
    analytic_terminal = math.exp(len(rets) * log_growth)
    med = mc_pos["by_fraction"]["0.05"]["terminal_p50"]
    _require(abs(med - analytic_terminal) / analytic_terminal < 0.15,
             f"MC median terminal {med:.2f} vs analytic {analytic_terminal:.2f}")
    return (f"mean {stats['mean']:+.4f}/{mean}, win {stats['win_rate']:.3f}/{win_true:.3f}, "
            f"sharpe {stats['sharpe_trade']:.3f}/{sharpe_true:.3f}, "
            f"P(loss) b2 {p_loss_b2:.3f} vs iid {p_loss_iid:.3f}, "
            f"cluster p05 b20 {s_clust['terminal_p05']:.3f} < b2 {s_unclust['terminal_p05']:.3f}")


# --------------------------------------------------------------------------
# 2. harness regression — EXP-050 (load-bearing)
# --------------------------------------------------------------------------


def _exp050_inputs() -> tuple[pd.DataFrame, list[str]]:
    """Rebuild the EXP-050 model matrix exactly as the reference script did."""
    res = pd.read_csv(EXP045C_TRADES)
    res["date"] = pd.to_datetime(res["date"])
    res["ret_mid"] = ((res["s3_exit"] + res["ask_exit"]) / 2) / \
        ((res["bid_entry"] + res["s3_cost"]) / 2) - 1
    feats = pd.read_csv(MODEL_ROWS_J14, dtype={"ticker": str})
    feats["event_date"] = pd.to_datetime(feats["event_date"])
    feats = feats.drop(columns=[c for c in ["year", "j", "im1", "runup"] if c in feats.columns])
    m = res.merge(feats, left_on=["ticker", "date"], right_on=["ticker", "event_date"], how="inner")
    fc = [c for c in EXP049_FEATS if c in m.columns]
    fc = [c for c in fc if m[c].nunique(dropna=True) >= 2 and m[c].notna().sum() >= 500]
    return m, fc


def _exp050_walk_forward_predictions(m: pd.DataFrame, fc: list[str]) -> pd.DataFrame:
    """The reference WF prediction pass, verbatim (same hyper-parameters)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    preds = []
    for y in sorted(m["year"].unique()):
        trn = m[m["year"] < y]
        tst = m[m["year"] == y]
        if len(trn) < 800 or len(tst) < 100:
            continue
        mdl = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4,
            min_samples_leaf=50, l2_regularization=1.0, random_state=0)
        mdl.fit(trn[fc], trn["ret_mid"].to_numpy())
        tst = tst.copy()
        tst["gpred"] = mdl.predict(tst[fc])
        preds.append(tst)
    return pd.concat(preds)


def _exp050_trades_frame(p: pd.DataFrame) -> pd.DataFrame:
    """The gated candidate as an evaluate() trade frame (mid fills only)."""
    entry_cost = (p["bid_entry"] + p["s3_cost"]) / 2
    exit_value = (p["s3_exit"] + p["ask_exit"]) / 2
    frame = pd.DataFrame({
        "event_id": p["ticker"].astype(str) + "_" + p["date"].dt.strftime("%Y%m%d"),
        "ticker": p["ticker"].astype(str),
        "event_date": pd.to_datetime(p["date"]),
        "entry_date": pd.to_datetime(p["date"]),
        "exit_date": pd.to_datetime(p["date"]) + pd.Timedelta(days=1),
        "fill_alpha": 0.5,
        "entry_cost": entry_cost.to_numpy(),
        "exit_value": exit_value.to_numpy(),
        "ret": p["ret_mid"].to_numpy(),
        "gpred": p["gpred"].to_numpy(),
    })
    return frame


@check("harness_regression",
       description="EXP-050 reproduces through evaluate() (load-bearing)")
def check_harness_regression() -> str:
    for p in (EXP050_RESULT, EXP045C_TRADES, MODEL_ROWS_J14):
        _require(p.exists(), f"reference artifact missing: {p}")
    reference = json.loads(EXP050_RESULT.read_text())

    m, fc = _exp050_inputs()
    p = _exp050_walk_forward_predictions(m, fc).sort_values("date", kind="stable")
    q80 = float(p["gpred"].quantile(0.8))
    trades = _exp050_trades_frame(p)

    # The reference gate: pooled OOS quantile threshold (mild look-ahead on
    # the threshold itself — reproduced and documented, not repeated).
    class PooledQuantileGate:
        def __init__(self):
            self.seen = []

        def fit(self, train):
            self.seen.append(int(train["year"].max()) if len(train) else None)

        def select(self, rows):
            return rows["gpred"] >= q80

    # The reference gate: pooled OOS quantile threshold (mild look-ahead on
    # the threshold itself — reproduced and documented, not repeated).
    gate_state = PooledQuantileGate()
    gate = Gate(fit=gate_state.fit, select=gate_state.select, name="exp050_pooled_q80")

    with tempfile.TemporaryDirectory(prefix="exp050_regression_") as tmp:
        run_dir = Path(tmp)
        spec = {
            "id": "EXP-050-REGRESSION",
            "title": "Harness regression: EXP-050 GBM top-20% gate",
            "hypothesis": "The harness reproduces the evidence the plan rests on.",
            "strategy": "STR-RUNUP",
            "price_source": "ORATS chains via earnings_predictions/opf (engine.replay lineage)",
            "primary_spec": {"gate": "gbm_top20_pooled_oos", "fill": "mid"},
            # min_train_years=0: the gate's model training happened upstream on
            # the 2018+ feature matrix (the gpred column embodies it), so the
            # harness must apply the pooled threshold even to the first year.
            "walk_forward": {"unit": "year", "min_train_years": 0},
            "equity_mode": "sequential",   # the EXP-050 reference construction
            "mc_draw_order": "per_fraction",  # the EXP-050 reference rng order
            "has_short_leg": False,
            "preregistered_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        result = evaluate(
            spec, trades, gate=gate, run_dir=run_dir,
            alphas=(0.5,), fractions=(0.05, 0.10, 0.20),
            mc_paths=1000, mc_block=20, seed=0,
            stress=False, write_report=True,
            input_files=[EXP045C_TRADES, MODEL_ROWS_J14, EXP050_RESULT],
        )

    headline = result.metrics
    n = headline["n"]
    _require(n == 531, f"gated trade count {n} != 531 (the reference number)")

    ref_year = reference["f5"]["per_year"]
    for year, ref_mean in ref_year.items():
        got = headline["by_year"].get(year, {}).get("mean")
        _require(got is not None and abs(got - ref_mean) < 5e-4,
                 f"year {year}: mean {got} vs reference {ref_mean}")

    # Equity and MC vs the saved reference artifact (reproducible today).
    tol_eq = {0.05: 0.02, 0.10: 0.06, 0.20: 0.15}
    tol_mc = 0.02
    details = []
    for f in (0.05, 0.10, 0.20):
        key = f"f{int(f * 100)}"
        eq_final = result.results["equity_curves"][f"{f:.2f}"]["final"]
        ref_final = reference[key]["final_equity"]
        _require(abs(eq_final - ref_final) < tol_eq[f] * max(ref_final, 1.0),
                 f"sizing {f:.0%}: terminal equity {eq_final:.2f} vs artifact {ref_final}")
        p_loss = headline["mc"][f"{f:.2f}"]["p_loss"]
        ref_ploss = reference[key]["mc_p_loss"]
        _require(abs(p_loss - ref_ploss) < tol_mc,
                 f"sizing {f:.0%}: MC P(loss) {p_loss:.3f} vs artifact {ref_ploss}")
        details.append(f"{f:.0%}: {eq_final:.2f}x/{p_loss:.2f} (artifact {ref_final}x/{ref_ploss})")

    # The report-of-record comparison: f10/f20 match the experiment markdown;
    # the f5 row (2.4x / P(loss) 0.06) does NOT reproduce from the reference's
    # own saved artifacts — the divergence is documented, not hidden.
    report_numbers = {"f5": (2.4, 0.06), "f10": (5.8, 0.20), "f20": (10.4, 0.32)}
    mismatches = []
    for key, (rep_eq, rep_ploss) in report_numbers.items():
        artifact_eq = reference[key]["final_equity"]
        artifact_ploss = reference[key]["mc_p_loss"]
        if abs(artifact_eq - rep_eq) > 0.1 or abs(artifact_ploss - rep_ploss) > 0.02:
            mismatches.append(
                f"{key}: report {rep_eq}x/P(loss){rep_ploss} vs artifact "
                f"{artifact_eq}x/P(loss){artifact_ploss}")
    _write_regression_report(reference, result, mismatches)
    _require(len(mismatches) == 1 and mismatches[0].startswith("f5"),
             f"unexpected report/artifact mismatches: {mismatches}")
    return (f"n={n}, per-year means match, " + "; ".join(details) +
            "; KNOWN DIVERGENCE documented: report f5 row is stale "
            "(reports/phase2_exp050_regression.md)")


def _write_regression_report(reference: dict, result, mismatches: list[str]) -> None:
    headline = result.metrics
    lines = [
        "# Phase 2 harness regression — EXP-050 reproduction & divergence record",
        "",
        f"*{pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds')} — generated by "
        "checks/phase2_checks.py harness_regression*",
        "",
        "## What reproduced",
        "",
        "The EXP-050 configuration (GBM top-20% gate, walk-forward, mid fills, S3",
        "scoped universe) re-run through `engine.evaluate` reproduces the saved",
        "reference artifact `earnings_predictions/opf/results/exp050_equity.json`:",
        "",
        f"- gated trades: {headline['n']} (reference 531, 2019-2026)",
        "- per-year mean returns: match within 5e-4 for all eight years",
    ]
    for f in (0.05, 0.10, 0.20):
        key = f"f{int(f * 100)}"
        eq_final = result.results["equity_curves"][f"{f:.2f}"]["final"]
        p_loss = headline["mc"][f"{f:.2f}"]["p_loss"]
        lines.append(
            f"- sizing {f:.0%}: terminal {eq_final:.2f}x / MC P(loss) {p_loss:.3f} "
            f"vs artifact {reference[key]['final_equity']}x / {reference[key]['mc_p_loss']}")
    lines += [
        "",
        "Equity mode `sequential` and `mc_draw_order=per_fraction` reproduce the",
        "reference script's constructions exactly; with the harness defaults",
        "(cashflow equity, shared bootstrap draws) the numbers differ by design.",
        "",
        "## The divergence",
        "",
        "The experiment report of record (opf/experiments/EXP-048_050_midfill_program.md)",
        "and the plan cite **2.4x terminal / MC P(loss) 6% at 5% sizing**. Those",
        "numbers do NOT reproduce from the reference's own saved artifacts — today,",
        "or with the reference script itself:",
        "",
    ]
    lines += [f"- {m}" for m in mismatches]
    lines += [
        "",
        "Root cause: the f10 and f20 rows of the report match the artifact exactly",
        "(5.8 vs 5.75, 10.4 vs 10.4; P(loss) 0.20/0.32 vs 0.197/0.316), so the report",
        "was written from this artifact, but the f5 row matches nothing in it. The",
        "deterministic 5% terminal equity is a function of the 531 trade returns",
        "alone (which the report's per-year narrative matches: 2022 +46%, 2024 +42%),",
        "and it is 2.83x — so the 2.4x/6% row is a stale number from an earlier",
        "variant that was overwritten, not a reproducible statistic. The plan's",
        "'2.4x at 5% sizing with MC P(loss)=6%' should read **2.83x / P(loss) 15%**",
        "for this exact trade set, or be re-derived from a fresh pre-registered run.",
        "",
        "## Consequence",
        "",
        "The harness asserts against the REPRODUCIBLE artifact numbers and carries",
        "this divergence record, per the guide's 'root-cause and document ANY",
        "divergence before running new experiments'. The 5%-sizing conclusion",
        "(conservative configuration) survives — P(loss) at 5% is still the lowest",
        "of the three sizings — but the margin is thinner than the plan quotes.",
        "",
    ]
    paths.REPORTS.mkdir(parents=True, exist_ok=True)
    (paths.REPORTS / "phase2_exp050_regression.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------
# 3. preregistration enforcement
# --------------------------------------------------------------------------


@check("preregistration", needs_data=False,
       description="missing or back-dated stamps refuse the OOS stage")
def check_prereg_enforcement() -> str:
    trades, _, _ = _synthetic_trades(n_years=6, per_year=6)

    with tempfile.TemporaryDirectory(prefix="prereg_") as tmp:
        run_dir = Path(tmp)
        base = {"id": "EXP-901", "primary_spec": {"x": 1}}

        # (a) no stamp -> refuse
        try:
            check_preregistration(base, run_dir)
            raise AssertionError("missing preregistered_at was accepted")
        except PreregistrationError:
            pass

        # (b) scaffolded spec carries a stamp; scaffolder numbering starts at 101
        folder = _scaffold_into(Path(tmp) / "experiments", "Scaffold probe",
                                "A probe hypothesis.", number=101)
        spec = lib.load_spec(folder / "spec.yaml")
        _require(spec.get("preregistered_at"), "scaffolder did not stamp preregistered_at")
        try:
            lib.parse_experiment_id("EXP-050")
            raise AssertionError("EXP-050 (0-50 range) was accepted as a new experiment id")
        except ValueError:
            pass

        # (c) back-dated stamp (after the first recorded run) -> refuse
        append_run_log(run_dir, {"ts": pd.Timestamp.now(tz="UTC").isoformat(),
                                 "spec_id": "EXP-901", "stage": "ran"})
        late = dict(base)
        late["preregistered_at"] = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=2)).isoformat()
        try:
            check_preregistration(late, run_dir)
            raise AssertionError("back-dated preregistered_at was accepted")
        except PreregistrationError:
            pass

        # (d) valid stamp -> the evaluation runs
        ok = dict(base)
        ok["preregistered_at"] = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)).isoformat()
        result = evaluate(ok, trades, run_dir=run_dir, mc_paths=50,
                          stress=False, write_report=False)
        _require(result.results["preregistration"]["valid"], "valid stamp rejected")
        _require((run_dir / "results" / "run_log.jsonl").exists(), "run log not written")

    return "missing refused, back-dated refused, valid accepted; scaffold stamps + EXP-101 floor"


def _scaffold_into(root: Path, title: str, hypothesis: str, number: int) -> Path:
    from experiments.new_experiment import scaffold

    return scaffold(title, hypothesis, exp_id=f"EXP-{number}", root=root,
                    ledger_path=root / "LEDGER.csv")


# --------------------------------------------------------------------------
# 4. walk-forward leak poison
# --------------------------------------------------------------------------


@check("wf_leak_poison", needs_data=False,
       description="a spy gate proves fit never receives the test year")
def check_wf_leak_poison() -> str:
    trades, _, _ = _synthetic_trades(n_years=8, per_year=10)
    trades["year"] = pd.to_datetime(trades["event_date"]).dt.year
    # The marker: 1.0 exactly on the rows of the year being traded. A gate that
    # ever saw a test-year row during fit would have seen marker == test year.
    trades["poison_marker"] = trades["year"].astype(float)

    seen: list[pd.DataFrame] = []

    class SpyGate:
        def fit(self, train):
            seen.append(train.copy())

        def select(self, rows):
            return pd.Series(True, index=rows.index)

    spy = SpyGate()
    gate = Gate(fit=spy.fit, select=spy.select, name="spy")
    wf = walk_forward(trades, gate, min_train_years=2)

    gated = [d for d in wf["diagnostics"] if d["n_train"] and not d.get("ungated")]
    _require(len(seen) == len(gated), f"fit called {len(seen)}x, expected {len(gated)}")
    for frame, diag in zip(seen, gated):
        test_year = diag["year"]
        poison_hit = frame[frame["poison_marker"] == float(test_year)]
        _require(poison_hit.empty,
                 f"year {test_year}: fit received {len(poison_hit)} rows of the test year")
        _require(int(frame["year"].max()) < test_year,
                 f"year {test_year}: fit saw year {frame['year'].max()}")
    _require(wf["audit"]["leak_free"], "audit receipt does not certify leak-free")
    return f"{len(seen)} folds; no fit ever saw its test year (marker-proven)"


# --------------------------------------------------------------------------
# 5. ledger append-only
# --------------------------------------------------------------------------


@check("ledger_append_only", needs_data=False,
       description="the ledger grows by prefix only; there is no rewrite path")
def check_ledger_append_only() -> str:
    with tempfile.TemporaryDirectory(prefix="ledger_") as tmp:
        path = Path(tmp) / "LEDGER.csv"
        row = {"id": "EXP-101", "spec_hash": "abc123", "date": "2026-08-30",
               "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "",
               "promoted": "False"}
        lib.ledger_append([row], path=path)
        before = path.read_bytes()

        # A superseding row APPENDS; the planned row stays (history preserved).
        lib.ledger_append([{**row, "stage": "ran", "oos_mean_mid": "0.02"}], path=path)
        after = path.read_bytes()
        _require(lib.verify_append(before, after), "append did not preserve the prefix")
        frame = lib.ledger_read(path)
        _require(len(frame) == 2, f"expected 2 rows (history kept), got {len(frame)}")
        _require(frame.iloc[0]["stage"] == "planned" and frame.iloc[1]["stage"] == "ran",
                 "supersede rewrote history instead of appending")

        # The rewrite predicate catches a tampered history.
        tampered = after.replace(b"planned", b"ran___")
        _require(not lib.verify_append(before, tampered) or tampered == after,
                 "verify_append accepted a rewrite")
        rewritten = before[:-3] + b"XYZ"
        _require(not lib.verify_append(before, rewritten),
                 "verify_append accepted a trimmed/edited history")

        # There is no rewrite or delete API to call.
        for forbidden in ("ledger_rewrite", "ledger_delete", "ledger_replace", "ledger_write"):
            _require(not hasattr(lib, forbidden), f"lib exposes a forbidden {forbidden}()")

        # A row missing columns is refused.
        try:
            lib.ledger_append([{"id": "EXP-102"}], path=path)
            raise AssertionError("incomplete ledger row was accepted")
        except lib.LedgerError:
            pass
    return "prefix-only growth verified; supersede keeps history; no rewrite API"


# --------------------------------------------------------------------------
# 6. promotion dry-run
# --------------------------------------------------------------------------


def _metrics(mean: float, sharpe: float, p_loss: float, regimes: dict | None = None) -> dict:
    return {
        "mean": mean,
        "sharpe_trade": sharpe,
        "mc": {"p_loss": p_loss},
        "stress": {"regimes": regimes or {"2022": {"n": 40, "mean": 0.05}},
                   "tail_injection": {"available": True}},
    }


@check("promotion_dry_run", needs_data=False,
       description="synthetic challenger better/worse -> promote/refuse")
def check_promotion_dry_run() -> str:
    champion = _metrics(0.03, 1.2, 0.10)

    better = _metrics(0.04, 1.5, 0.08)
    promote, reasons = promote_mod.decide(better, champion, prereg_valid=True)
    _require(promote, f"better challenger refused: {reasons}")

    worse_mean = _metrics(0.02, 1.5, 0.08)
    promote, reasons = promote_mod.decide(worse_mean, champion, prereg_valid=True)
    _require(not promote, "worse-mean challenger was promoted")

    worse_ploss = _metrics(0.04, 1.5, 0.20)
    promote, _ = promote_mod.decide(worse_ploss, champion, prereg_valid=True)
    _require(not promote, "worsened MC P(loss) was promoted")

    no_prereg = _metrics(0.04, 1.5, 0.08)
    promote, _ = promote_mod.decide(no_prereg, champion, prereg_valid=False)
    _require(not promote, "unregistered challenger was promoted")

    short_no_tail = _metrics(0.04, 1.5, 0.08)
    short_no_tail["stress"]["tail_injection"] = {}
    promote, _ = promote_mod.decide(short_no_tail, champion, prereg_valid=True, short_leg=True)
    _require(not promote, "short-leg challenger without tail injection was promoted")

    red_regime = _metrics(0.04, 1.5, 0.08,
                          regimes={"2022": {"n": 40, "mean": -0.03}})
    promote, _ = promote_mod.decide(red_regime, champion, prereg_valid=True)
    _require(not promote, "challenger with a new red stress cell was promoted")

    # End-to-end CLI dry-runs on a scaffolded experiment.
    with tempfile.TemporaryDirectory(prefix="promote_") as tmp:
        tmp_root = Path(tmp)
        folder = _scaffold_into(tmp_root / "experiments", "Promotion probe",
                                "A probe.", number=101)
        spec = lib.load_spec(folder / "spec.yaml")
        results = {"headline": better, "preregistration": {"valid": True},
                   "spec_hash": lib.spec_hash(spec)}
        (folder / "results").mkdir(exist_ok=True)
        (folder / "results" / f"metrics_{lib.spec_hash(spec)[:12]}.json").write_text(
            json.dumps(results))
        champ_path = tmp_root / "champion.json"
        champ_path.write_text(json.dumps(champion))

        good = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / "promote.py"), "EXP-101",
             "--champion-metrics", str(champ_path), "--dry-run"],
            capture_output=True, text=True, cwd=tmp_root,
            env={**__import__("os").environ, "INVESTING_PLAN_ROOT": str(tmp_root)},
        )
        _require(good.returncode == 0, f"dry-run of a better challenger failed: {good.stdout}{good.stderr}")

        results["headline"] = worse_mean
        (folder / "results" / f"metrics_{lib.spec_hash(spec)[:12]}.json").write_text(
            json.dumps(results))
        bad = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / "promote.py"), "EXP-101",
             "--champion-metrics", str(champ_path), "--dry-run"],
            capture_output=True, text=True, cwd=tmp_root,
            env={**__import__("os").environ, "INVESTING_PLAN_ROOT": str(tmp_root)},
        )
        _require(bad.returncode == 1, "dry-run of a worse challenger did not refuse")

    return "6 rule cases + CLI dry-runs: better promoted, worse/worse-P(loss)/unregistered/red refused"


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

ORDER = [
    "unittests",
    "synthetic_known",
    "preregistration",
    "wf_leak_poison",
    "ledger_append_only",
    "promotion_dry_run",
    "harness_regression",
]


def run(names: list[str], skip_data: bool = False) -> list[CheckOutcome]:
    outcomes: list[CheckOutcome] = []
    for name in names:
        spec = REGISTRY[name]
        if skip_data and spec["needs_data"]:
            outcomes.append(CheckOutcome(name, True, "skipped (--no-data)", skipped=True))
            print(f"  SKIP  {name}", flush=True)
            continue
        started = time.time()
        print(f"  ...   {name}", flush=True)
        try:
            detail = spec["fn"]() or ""
            passed = True
        except Exception as exc:  # noqa: BLE001 - a failing check must not end the run
            detail = f"{type(exc).__name__}: {exc}"
            passed = False
        elapsed = time.time() - started
        skipped = isinstance(detail, str) and detail.startswith("SKIP")
        outcomes.append(CheckOutcome(name, passed, detail, elapsed, skipped))
        status = "SKIP" if skipped else ("PASS" if passed else "FAIL")
        print(f"  {status:5s} {name}  ({elapsed:.1f}s)  {detail}", flush=True)
    return outcomes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", nargs="*", choices=ORDER, default=None)
    ap.add_argument("--no-data", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.list:
        for name in ORDER:
            spec = REGISTRY[name]
            flag = "data" if spec["needs_data"] else "pure"
            print(f"  {name:20s} [{flag}]  {spec['description']}")
        return 0

    names = args.only or ORDER
    print(f"Phase 2 acceptance checks ({len(names)} checks)\n", flush=True)
    started = time.time()
    outcomes = run(names, skip_data=args.no_data)

    failed = [o for o in outcomes if not o.passed]
    skipped = [o for o in outcomes if o.skipped]
    print(
        f"\n{len(outcomes) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} skipped in {time.time()-started:.0f}s"
    )
    if args.json:
        Path(args.json).write_text(
            json.dumps([o.__dict__ for o in outcomes], indent=1, default=str)
        )
    if failed:
        print("\nFAILED:", file=sys.stderr)
        for outcome in failed:
            print(f"  {outcome.name}: {outcome.detail}", file=sys.stderr)
        return 1
    print("\nPHASE 2 CHECKS: GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
