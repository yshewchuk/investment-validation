#!/usr/bin/env python3
"""EXP-115 — condition the residual pool on the prediction, in the scoring path.

    python3 experiments/EXP-115_condition_the_residual_pool_on_the_predi/run.py
    python3 .../run.py --stage 1

EXP-114 established the effect offline, in its own code. This experiment asks
whether it survives the trip into ``ModelArtifact`` and the scorer, and whether
it is safe there — conditioning the pool changes the draws that feed the payoff
simulation, so it moves ``exp_pnl_model`` and ``win_model`` too.

Stage 1 therefore does NOT reimplement the bucketing. It drives the shipping
``ModelArtifact.residual_pool`` and checks the numbers land where EXP-114 said
they would. A reimplementation agreeing with itself would prove nothing; the
failure this guards against is the shipped path differing from the studied one.

Causality is the same as EXP-114's: for test year Y the buckets are built only
from out-of-sample residuals of years before Y, and the decile edges come from
those years' predictions alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.models.registry import ModelArtifact, bucket_residuals  # noqa: E402
from experiments import lib  # noqa: E402

HERE = Path(__file__).resolve().parent
DECILES = 10
MIN_POOL = 250

#: EXP-114's out-of-sample walk-forward predictions for the size champion —
#: the same rows its own arms were scored on, so a difference here is this
#: experiment's doing rather than a different sample.
EXP114_BASE = (ROOT / "experiments" / "EXP-114_size_v2_0_step_5_learn_the_error_scale_i"
               / "results" / "base_size_v1_4.parquet")

#: What EXP-114 reported for the same two arms. Stage 1 reproduces these or the
#: shipping path does not match the studied one.
EXP114_REPORTED = {
    "flat": {"coverage_80": 0.7276, "pit_ks": 0.0525, "crps": 2.9523},
    "by_prediction_decile": {"coverage_80": 0.7934, "pit_ks": 0.0056, "crps": 2.7894},
}


class _Dummy:
    """The artifact needs a model to exist; stage 1 never predicts through it."""

    def predict(self, X):
        return np.zeros(len(X))


def _artifact(residuals, buckets=None) -> ModelArtifact:
    return ModelArtifact(
        model=_Dummy(), role="size", features=("unused",),
        residuals=residuals, target="abs_move", residual_buckets=buckets,
    )


def _pool_stats(pool: np.ndarray, residual: float) -> dict:
    """Coverage, PIT and CRPS of one residual against one pool."""
    ordered = np.sort(pool)
    lo80, hi80, lo50, hi50 = np.quantile(ordered, [0.10, 0.90, 0.25, 0.75])
    sub = (ordered if ordered.size <= 300
           else np.sort(np.random.default_rng(0).choice(pool, 300, replace=False)))
    crps = float(np.abs(sub - residual).mean()) - 0.5 * float(
        np.abs(sub[:, None] - sub[None, :]).mean()
    )
    return {
        "in80": bool(lo80 <= residual <= hi80),
        "in50": bool(lo50 <= residual <= hi50),
        "pit": float(np.searchsorted(ordered, residual, side="right") / ordered.size),
        "crps": crps,
        "width80": float(hi80 - lo80),
    }


def stage_1() -> dict:
    """Does the shipping code path reproduce EXP-114's calibration result?"""
    from scipy import stats as st

    if not EXP114_BASE.exists():
        raise SystemExit(
            f"missing {EXP114_BASE}\n"
            "Stage 1 reuses EXP-114's walk-forward predictions so both "
            "experiments describe the same rows. Re-run EXP-114 to rebuild it."
        )
    base = pd.read_parquet(EXP114_BASE)
    base["residual"] = base["actual"] - base["pred"]
    print(f"[EXP-115] {len(base):,} OOS rows from EXP-114's walk-forward", flush=True)

    rows, fallbacks, thin_years = [], 0, {}
    for year in sorted(base["year"].unique()):
        past = base[base["year"] < year]
        now = base[base["year"] == year]
        if len(past) < MIN_POOL or now.empty:
            continue
        residuals = past["residual"].to_numpy(dtype=float)
        buckets = bucket_residuals(past["pred"].to_numpy(dtype=float), residuals,
                                   deciles=DECILES, min_pool=MIN_POOL)
        conditioned = _artifact(residuals, buckets)
        flat = _artifact(residuals)
        if buckets and buckets["thin"]:
            thin_years[int(year)] = buckets["thin"]

        for pred, residual in zip(now["pred"].to_numpy(dtype=float),
                                  now["residual"].to_numpy(dtype=float)):
            pool_c, label = conditioned.residual_pool(pred)
            pool_f, _ = flat.residual_pool(pred)
            if label.startswith("flat"):
                fallbacks += 1
            rows.append({"year": int(year), "pred": float(pred), "arm": "by_prediction_decile",
                         **_pool_stats(pool_c, residual)})
            rows.append({"year": int(year), "pred": float(pred), "arm": "flat",
                         **_pool_stats(pool_f, residual)})

    scored = pd.DataFrame(rows)
    out = {"rows": int(len(scored) // 2), "fallback_draws": fallbacks,
           "thin_buckets_by_year": thin_years, "arms": {}}

    for arm, d in scored.groupby("arm"):
        w = d["width80"]
        out["arms"][arm] = {
            "n": int(len(d)),
            "coverage_80": float(d["in80"].mean()),
            "coverage_50": float(d["in50"].mean()),
            "coverage_80_error": float(abs(d["in80"].mean() - 0.80)),
            "pit_ks": float(st.kstest(d["pit"].to_numpy(dtype=float), "uniform").statistic),
            "crps": float(d["crps"].mean()),
            "width80_p10": float(w.quantile(0.10)),
            "width80_p50": float(w.quantile(0.50)),
            "width80_p90": float(w.quantile(0.90)),
        }

    # Does the shipped path agree with the experiment that motivated it?
    out["reproduces_exp114"] = {}
    for arm, want in EXP114_REPORTED.items():
        got = out["arms"][arm]
        deltas = {k: round(got[k] - v, 5) for k, v in want.items()}
        out["reproduces_exp114"][arm] = {
            "expected": want,
            "delta": deltas,
            "agrees": all(abs(v) < 0.005 for v in deltas.values()),
        }

    # The programme's consistency standard, not a magnitude cutoff.
    err = scored.pivot_table(index="year", columns="arm", values="in80", aggfunc="mean")
    err = (err - 0.80).abs()
    gain = err["flat"] - err["by_prediction_decile"]
    best = gain.idxmax()
    out["per_year"] = {
        "years_closer": int((gain > 0).sum()),
        "years_total": int(len(gain)),
        "mean_gain": round(float(gain.mean()), 5),
        "wilcoxon_p": round(float(st.wilcoxon(gain.values).pvalue), 5),
        "best_year": int(best),
        "mean_excluding_best": round(float(gain.drop(index=best).mean()), 5),
        "by_year": {int(y): round(float(v), 5) for y, v in gain.items()},
    }
    out["clears"] = bool(
        out["arms"]["by_prediction_decile"]["coverage_80_error"]
        < out["arms"]["flat"]["coverage_80_error"]
        and out["per_year"]["wilcoxon_p"] <= 0.05
        and out["per_year"]["mean_excluding_best"] > 0
    )
    return out, scored



# --------------------------------------------------------------------------
# stage 2 — gate invariance
# --------------------------------------------------------------------------


def _historical_context(as_of: pd.Timestamp):
    """A FeatureContext that knows nothing after ``as_of``.

    ``FeatureContext.load()`` reads the whole panel, which is correct for
    scoring today and a leak for scoring a past date: the causal audit refuses
    the run outright, naming the prior-move features it can see at a later event
    than the decision. Truncating the panel and the daily frame to rows on or
    before the decision date is what makes a historical window legitimate rather
    than merely quiet — the audit still runs, and now it passes on the merits.
    """
    from engine.features import FeatureContext

    ctx = FeatureContext.load()
    panel = ctx.panel[pd.to_datetime(ctx.panel["date"]) <= as_of]
    daily = ctx.daily
    if daily is not None:
        daily = daily[pd.to_datetime(daily["date"]) <= as_of]
    return FeatureContext(panel=panel, daily=daily, calendar=ctx.calendar)


def stage_2(tickers=None, horizon_days: int = 21, as_of=None,
            n_tickers: int = 300) -> dict:
    """Prove the gate does not move when the size model's pool is conditioned.

    Structurally it cannot: the gate scores from its own artifact and its own
    features, and never reads the size model. That argument was also available
    for EXP-105/109, where it was made and was wrong in practice — a size-model
    change silently replaced the registered gate with a size-driven selector and
    both arms of a stage came out counterfactual. So it is measured.

    The same Scorer scores the same board twice, with the cached size artifact
    swapped in between, and the gate columns are compared row for row.

    The CONTROL is the half that makes this a test at all: the size-derived
    numbers must actually move. An invariance that holds because nothing changed
    anywhere would pass while proving nothing.
    """
    from engine.data import store
    from engine.score import Scorer, score_calendar

    # A HISTORICAL as-of, deliberately. The forward board has no option chains
    # until the next ORATS pull, and without a chain the gate is never scored:
    # the first version of this stage ran on the forward board, compared
    # gate_score null-against-null on all 444 rows, found "0 differing", and
    # reported the invariance as proven. It proved nothing. The window is
    # therefore chosen inside the replayed history, and the guard below now
    # requires the gate to have actually scored and actually passed rows.
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else None
    if as_of is None:
        trades = store.read_table("trades", columns=["event_date", "provenance"])
        trades = trades[trades["provenance"].astype(str) == "engine.replay"]
        last = pd.Timestamp(trades["event_date"].max()).normalize()
        as_of = last - pd.Timedelta(days=horizon_days + 7)

    base = pd.read_parquet(EXP114_BASE)
    residual = (base["actual"] - base["pred"]).to_numpy(dtype=float)
    buckets = bucket_residuals(base["pred"].to_numpy(dtype=float), residual,
                               deciles=DECILES, min_pool=MIN_POOL)

    # Trades are limited the same way, so the payoff maps and analogs behind
    # the two runs are the ones that existed on the night being replayed.
    trades = store.read_table("trades")
    trades = trades[pd.to_datetime(trades["event_date"]) <= as_of]
    scorer = Scorer(context=_historical_context(as_of), trades=trades)

    # Scoped, and scoped to where the evidence is. The unrestricted historical
    # window is 2,297 events — 15x the forward board — and scoring it twice runs
    # over an hour. Gate invariance is a structural property: it holds for every
    # row or it holds for none, so a few hundred rows that REALLY pass the gate
    # demonstrate it as well as thousands -- but 40 tickers gave only 30 rows
    # where the gate actually ran, which is a real result on too small a
    # sample to lean on. The tickers are the ones with the most
    # replayed trades in this window, so chain coverage (and therefore a scored
    # gate) is where the sample is, rather than hand-picked.
    if not tickers:
        window = trades[
            (pd.to_datetime(trades["event_date"]) > as_of)
            & (pd.to_datetime(trades["event_date"]) <= as_of + pd.Timedelta(days=horizon_days))
        ]
        if window.empty:
            window = trades
        tickers = (window["ticker"].value_counts().head(n_tickers).index.tolist())
    tickers = sorted(tickers)

    # progress_every left at its default (50). It was set to 0 here, copied from
    # a helper that scores a board small enough not to need it; on a run this
    # size that turned an hour of work into an hour of silence.
    kw = dict(horizon_days=horizon_days, scorer=scorer, alt_strikes=0,
              tickers=tickers)

    before = score_calendar(as_of, **kw)

    # Swap the pool INSIDE the loaded champion, so everything else about the
    # scoring run — data, calendar, payoff maps, seeds — is held fixed.
    swapped = []
    for key, loaded in list(scorer._models.items()):
        if loaded is None or key[1] != "size":
            continue
        entry, artifact = loaded
        artifact.residual_buckets = dict(buckets)
        artifact.__post_init__()
        swapped.append(f"{key[0]}/{key[1]}")
    if not swapped:
        return {"available": False,
                "reason": "no size champion was loaded, so nothing was conditioned"}

    after = score_calendar(as_of, **kw)

    key_cols = ["ticker", "strategy", "event_date"]
    merged = before.merge(after, on=key_cols, suffixes=("_before", "_after"))

    def _same(column: str) -> dict:
        a, b = merged[f"{column}_before"], merged[f"{column}_after"]
        both_null = a.isna() & b.isna()
        equal = both_null | (a.astype(object) == b.astype(object))
        return {"identical": bool(equal.all()),
                "differing_rows": int((~equal).sum()),
                "compared": int(len(merged))}

    gate = {c: _same(c) for c in ("gate_score", "gate_threshold", "gate_pass")}
    control = {c: _same(c) for c in ("driver_p10", "driver_p90")}

    passers_before = before[before["gate_pass"] == True]  # noqa: E712
    passers_after = after[after["gate_pass"] == True]  # noqa: E712
    selection_identical = (
        set(map(tuple, passers_before[key_cols].astype(str).to_numpy()))
        == set(map(tuple, passers_after[key_cols].astype(str).to_numpy()))
    )

    moved = sum(1 for c in control.values() if not c["identical"])
    # Vacuity guards. Each of these was a way the first version passed while
    # measuring nothing, so each is checked explicitly rather than assumed.
    gate_scored = int(before["gate_score"].notna().sum())
    gate_identical = all(g["identical"] for g in gate.values())
    vacuous = None
    if gate_scored == 0:
        vacuous = "the gate never scored on this window, so its columns are null-vs-null"
    elif len(passers_before) == 0:
        vacuous = "no row passed the gate, so the selection comparison is empty-vs-empty"
    elif moved == 0:
        vacuous = "the size outputs did not move, so the gate could not have"

    return {
        "tickers": tickers,
        "gate_scored_rows": gate_scored,
        "vacuous": vacuous,
        "available": True,
        "as_of": str(as_of.date()),
        "rows_compared": int(len(merged)),
        "conditioned_champions": swapped,
        "gate": gate,
        "control_size_outputs": control,
        "gate_passers_before": int(len(passers_before)),
        "gate_passers_after": int(len(passers_after)),
        "selection_identical": bool(selection_identical),
        "control_moved": bool(moved > 0),
        "clears": bool(gate_identical and selection_identical and vacuous is None),
        "why": (
            f"VACUOUS: {vacuous}" if vacuous is not None
            else "gate moved — conditioning is NOT gate-invariant" if not gate_identical
            else "gate and selection identical over rows where the gate really ran, "
                 "while the size outputs moved"
        ),
    }


# --------------------------------------------------------------------------
# stages 3 and 4 — what conditioning costs downstream
# --------------------------------------------------------------------------

GATE_ALPHA = 0.5
TOP_QUINTILE = 0.2


def _simulated(arm_pools, trades, payoffs, seed_salt: str) -> pd.DataFrame:
    """Push each trade's driver draws through its causal payoff map.

    This is the scorer's own arithmetic — ``simulate_returns`` over draws from
    the artifact's pool plus the payoff map's own residual noise — run over
    replayed trades whose outcome is known, which is the only way to ask whether
    a probability was any good.
    """
    from engine.payoff import simulate_returns
    from engine.score import MODEL_DRAWS

    rows = []
    for record in trades.itertuples(index=False):
        payoff = payoffs.get(record.year)
        pool = arm_pools.get(record.year)
        if payoff is None or pool is None:
            continue
        artifact, _ = pool
        # sha256, not hash(). Python salts string hashing per process, so the
        # first version of this drew different residuals on every run: the
        # reliability slope moved 0.6138 -> 0.5512 between two runs of identical
        # code, which is larger than the effect stage 3 was trying to measure.
        # engine.score seeds the same way for the same reason.
        rng = np.random.default_rng(
            int.from_bytes(
                hashlib.sha256(
                    f"{seed_salt}|{record.ticker}|{record.event_date}".encode()
                ).digest()[:8],
                "big",
            )
        )
        draws = record.pred + artifact.residual_draws(
            MODEL_DRAWS, rng, prediction=record.pred
        )
        noise = payoff.residual_draws(MODEL_DRAWS, rng)
        returns = simulate_returns(draws, payoff, record.spot_entry,
                                   record.entry_cost, noise)
        rows.append({
            "ticker": record.ticker, "event_date": record.event_date,
            "exit_date": record.exit_date, "year": record.year,
            "raw_win": float((returns > 0).mean()),
            "exp_pnl": float(returns.mean()),
            "ret": float(record.ret),
            "outcome": float(record.ret > 0),
        })
    return pd.DataFrame(rows)


def _refit_recalibration(scored: pd.DataFrame) -> pd.DataFrame:
    """Isotonic recalibration refitted per year on that arm's OWN pairs.

    The shipped map was fitted against the flat pool's probabilities. Scoring a
    conditioned arm through it would measure the mismatch between two arms'
    distributions rather than either arm's calibration, so each arm is refitted
    on its own (raw_win, outcome) pairs from years strictly before the test
    year — the same causal rule ``fit_recalibration`` applies.
    """
    from sklearn.isotonic import IsotonicRegression

    out = []
    for year in sorted(scored["year"].unique()):
        past = scored[scored["year"] < year]
        now = scored[scored["year"] == year].copy()
        if len(past) < 200 or now.empty:
            continue
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(past["raw_win"].to_numpy(dtype=float),
                past["outcome"].to_numpy(dtype=float))
        now["win"] = iso.predict(now["raw_win"].to_numpy(dtype=float))
        out.append(now)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _calibration_metrics(d: pd.DataFrame, column: str) -> dict:
    from sklearn.linear_model import LinearRegression

    p = np.clip(d[column].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    y = d["outcome"].to_numpy(dtype=float)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    # Reliability slope: regress outcome on forecast. 1.0 is perfect.
    slope = float(LinearRegression().fit(p.reshape(-1, 1), y).coef_[0])
    return {"n": int(len(d)), "brier": brier, "log_loss": logloss,
            "reliability_slope": slope, "mean_forecast": float(p.mean()),
            "base_rate": float(y.mean())}


def stages_3_and_4() -> dict:
    """Does conditioning cost win-rate calibration, or trade quality?"""
    from engine.data import store
    from engine.payoff import PayoffError, fit_payoff
    from engine.replay import legs_spot_dte

    base = pd.read_parquet(EXP114_BASE)
    base["residual"] = base["actual"] - base["pred"]

    trades = store.read_table("trades")
    trades = trades[
        (trades["provenance"].astype(str) == "engine.replay")
        & (trades["strategy"] == "STR-THRU")
        & np.isclose(trades["fill_alpha"].astype(float), GATE_ALPHA)
    ].copy()
    trades["event_date"] = pd.to_datetime(trades["event_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    trades["year"] = trades["event_date"].dt.year
    if "spot_entry" not in trades.columns:
        trades["spot_entry"], trades["dte_entry"] = legs_spot_dte(trades)

    preds = base[["ticker", "date", "pred"]].rename(columns={"date": "event_date"})
    preds["event_date"] = pd.to_datetime(preds["event_date"])
    trades = trades.merge(preds, on=["ticker", "event_date"], how="inner")

    # fit_payoff needs its DRIVER, and the Tier-2 trades schema deliberately
    # does not carry it — abs_move lives in the panel. Exactly the join
    # Scorer._enrich does, for the same reason, and exactly what EXP-109 had to
    # fix when the payoff fit failed for every year.
    from engine.features import load_panel

    driver = load_panel()[["ticker", "date", "abs_move"]].rename(
        columns={"date": "event_date"})
    driver["event_date"] = pd.to_datetime(driver["event_date"])
    trades = trades.merge(driver, on=["ticker", "event_date"], how="left")
    trades = trades.dropna(subset=["spot_entry", "entry_cost", "ret", "pred"])
    trades = trades[trades["entry_cost"] > 0]
    print(f"[EXP-115] stages 3-4: {len(trades):,} replayed STR-THRU trades "
          f"carrying an OOS prediction", flush=True)

    # Causal per-year payoff maps and residual pools, both fitted on the past.
    payoffs, pools = {}, {"flat": {}, "by_prediction_decile": {}}
    for year in sorted(trades["year"].unique()):
        try:
            payoffs[year] = fit_payoff(trades, "STR-THRU", alpha=GATE_ALPHA,
                                       before=pd.Timestamp(f"{year}-01-01"))
        except PayoffError as exc:
            # NOT a bare skip. A thin early year genuinely has too few closed
            # trades and is a legitimate skip; anything else is a bug, and
            # swallowing it reports "0 years" instead of failing. That is how
            # the missing driver column above hid for a whole run.
            if "trades" not in str(exc).lower():
                raise
            continue
        past = base[base["year"] < year]
        if len(past) < MIN_POOL:
            payoffs.pop(year, None)
            continue
        residuals = past["residual"].to_numpy(dtype=float)
        buckets = bucket_residuals(past["pred"].to_numpy(dtype=float), residuals,
                                   deciles=DECILES, min_pool=MIN_POOL)
        pools["flat"][year] = (_artifact(residuals), None)
        pools["by_prediction_decile"][year] = (_artifact(residuals, buckets), None)
    print(f"  {len(payoffs)} years with a causal payoff map and a pool", flush=True)

    out = {"trades": int(len(trades)), "years": sorted(int(y) for y in payoffs)}
    scored_by_arm, stage3, stage4 = {}, {}, {}
    for arm in ("flat", "by_prediction_decile"):
        sim = _simulated(pools[arm], trades, payoffs, seed_salt=arm)
        if sim.empty:
            raise RuntimeError(
                f"{arm}: no trade produced a simulated return. "
                f"{len(payoffs)} payoff maps, {len(trades):,} trades — a stage "
                "that cannot run must say so rather than report an empty result."
            )
        recal = _refit_recalibration(sim)
        if recal.empty:
            stage3[arm] = {"available": False, "reason": "no year had enough prior pairs"}
            continue
        scored_by_arm[arm] = recal
        # Persisted: "why did the ranking change" cannot be answered from
        # aggregates, and re-running the simulation to ask is wasteful.
        (HERE / "results").mkdir(exist_ok=True)
        recal.to_parquet(HERE / "results" / f"stage34_{arm}.parquet", index=False)
        stage3[arm] = {
            "raw": _calibration_metrics(recal, "raw_win"),
            "recalibrated": _calibration_metrics(recal, "win"),
        }
        print(f"  {arm:22s} brier {stage3[arm]['recalibrated']['brier']:.5f} "
              f"log-loss {stage3[arm]['recalibrated']['log_loss']:.5f} "
              f"slope {stage3[arm]['recalibrated']['reliability_slope']:.4f}", flush=True)

        # Stage 4 rides on the same simulation: rank by expected PnL per year.
        rows = []
        for year, group in recal.groupby("year"):
            k = max(1, int(round(len(group) * TOP_QUINTILE)))
            top = group.nlargest(k, "exp_pnl")
            rows.append({"year": int(year), "n": int(len(group)), "k": k,
                         "top_ret": float(top["ret"].mean()),
                         "all_ret": float(group["ret"].mean())})
        by_year = pd.DataFrame(rows)
        weights = by_year["k"]
        stage4[arm] = {
            "by_year": rows,
            "top_quintile_return": float(np.average(by_year["top_ret"], weights=weights)),
            "all_trade_return": float(by_year["all_ret"].mean()),
            "oos_mean": float(recal["ret"].mean()),
            "sharpe_trade": float(recal["ret"].mean() / recal["ret"].std(ddof=1)),
        }

    if len(scored_by_arm) == 2:
        a, b = stage3["flat"]["recalibrated"], stage3["by_prediction_decile"]["recalibrated"]
        stage3["delta"] = {k: round(b[k] - a[k], 6)
                           for k in ("brier", "log_loss", "reliability_slope")}
        # Lower Brier and log-loss are better; slope nearer 1 is better.
        stage3["clears"] = bool(
            b["brier"] <= a["brier"] + 1e-4
            and b["log_loss"] <= a["log_loss"] + 1e-3
            and abs(b["reliability_slope"] - 1) <= abs(a["reliability_slope"] - 1) + 0.05
        )
        fa, fb = stage4["flat"], stage4["by_prediction_decile"]
        stage4["delta"] = {k: round(fb[k] - fa[k], 6)
                           for k in ("top_quintile_return", "oos_mean", "sharpe_trade")}
        stage4["clears"] = bool(
            fb["top_quintile_return"] >= fa["top_quintile_return"] - 1e-4
        )
    out["stage_3"] = stage3
    out["stage_4"] = stage4
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", type=int, default=0, choices=(0, 1, 2, 3),
                    help="0 runs every stage built so far")
    args = ap.parse_args()

    spec = lib.load_spec(HERE / "spec.yaml")
    results = {"spec_hash": lib.spec_hash(spec), "arm": spec["primary_spec"]["arm"]}
    scored = None
    if args.stage in (0, 1):
        s1, scored = stage_1()
        results["stage_1"] = s1
    if args.stage in (0, 2):
        print("\n[EXP-115] stage 2: scoring the board twice", flush=True)
        results["stage_2"] = stage_2()
    if args.stage in (0, 3):
        both = stages_3_and_4()
        results["stage_3"] = both["stage_3"]
        results["stage_4"] = both["stage_4"]
        results["stage_34_context"] = {k: both[k] for k in ("trades", "years")}

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    if scored is not None:
        scored.to_parquet(out / "stage1_scored.parquet", index=False)
    existing = {}
    if (out / "metrics.json").exists():
        existing = json.loads((out / "metrics.json").read_text())
    existing.update(results)
    (out / "metrics.json").write_text(json.dumps(existing, indent=1, default=str))
    results = existing

    if "stage_2" in results:
        s2 = results["stage_2"]
        if s2.get("available"):
            print(f"  rows compared      {s2['rows_compared']:,}")
            print(f"  gate identical     {all(g['identical'] for g in s2['gate'].values())}")
            print(f"  selection identical {s2['selection_identical']} "
                  f"({s2['gate_passers_before']} passers)")
            print(f"  gate actually scored {s2['gate_scored_rows']} rows")
            print(f"  control moved      {s2['control_moved']} "
                  f"(size outputs must change, or the test is vacuous)")
            print(f"  stage 2 {'CLEARS' if s2['clears'] else 'does NOT clear'}: {s2['why']}")
        else:
            print(f"  stage 2 unavailable: {s2.get('reason')}")
    if "stage_1" not in results:
        return 0
    s1 = results["stage_1"]

    for arm in ("flat", "by_prediction_decile"):
        a = s1["arms"][arm]
        print(f"  {arm:22s} cov80 {a['coverage_80']:.4f} (err {a['coverage_80_error']:.4f}) "
              f"PIT-KS {a['pit_ks']:.4f} CRPS {a['crps']:.4f} "
              f"width p10/p90 {a['width80_p10']:.2f}/{a['width80_p90']:.2f}", flush=True)
    for arm, rep in s1["reproduces_exp114"].items():
        print(f"  reproduces EXP-114 [{arm}]: {rep['agrees']}  delta={rep['delta']}")
    p = s1["per_year"]
    print(f"\n  per year: {p['years_closer']}/{p['years_total']} closer, "
          f"mean {p['mean_gain']:+.4f}, p={p['wilcoxon_p']}, "
          f"excl-best {p['mean_excluding_best']:+.4f}")
    print(f"  fallback draws: {s1['fallback_draws']:,} of {s1['rows']:,}")
    print(f"\nstage 1 {'CLEARS' if s1['clears'] else 'does NOT clear'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
