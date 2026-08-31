#!/usr/bin/env python3
"""EXP-109 — Absolute prior-move features, registered primary, through to trade selection.

    python3 experiments/EXP-109_absolute_prior_move_features_registered/run.py

One arm, registered before this ran, because EXP-108 already paid the
multiple-comparison cost of finding it.

**Stage 1 asks whether the accuracy gain is consistent**, not whether it is
large. EXP-108 used magnitude thresholds chosen by judgement; they reached the
right verdict on its primary for the wrong reason and would have reached the
wrong verdict on this arm. Per-year win count, a signed-rank p, and an explicit
check that no single year is carrying the result.

**Stage 2 asks whether it picks better trades.** A driver model that predicts
|move| more accurately has not thereby made anyone money: the prediction is
turned into an expected PnL through a payoff map and compared against the
premium actually quoted. Stage 2 joins each model's out-of-sample prediction to
the replayed STR-THRU trades, ranks them, and reports what the top quintile
really returned.

Both stages are causal by construction: the predictions come from walk-forward
(trained on prior years only), and the payoff map for a test year is fitted only
on trades that had CLOSED before that year began.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.data import store  # noqa: E402
from engine.features import load_panel  # noqa: E402
from engine.models.training import size_model  # noqa: E402
from engine.models.training.common import walk_forward  # noqa: E402
from engine.payoff import PayoffError, fit_payoff  # noqa: E402
from experiments import lib  # noqa: E402

HERE = Path(__file__).resolve().parent
GATE_ALPHA = 0.5
TOP_QUINTILE = 0.2


def derive(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    out = frame
    for name, rule in (spec.get("derived_features") or {}).items():
        source = rule[len("abs("):-1]
        out = out.assign(**{name: pd.to_numeric(out[source], errors="coerce").abs()})
    return out


def features_for(spec: dict) -> list[str]:
    base = list(size_model.FEATURES)
    for name in spec["primary_spec"]["features_added"]:
        if name not in base:
            base.append(name)
    return [f for f in base if f not in spec["primary_spec"]["features_removed"]]


def common_rows(frame, feature_sets, target):
    needed = sorted({f for fs in feature_sets for f in fs} | {target})
    values = frame[needed].apply(pd.to_numeric, errors="coerce")
    return frame.index[np.isfinite(values.to_numpy(dtype=float)).all(axis=1)]


# --------------------------------------------------------------------------
# stage 1 — is the gain consistent
# --------------------------------------------------------------------------


def stage_1(spec: dict) -> tuple[dict, dict]:
    from scipy import stats

    panel = derive(load_panel(), spec)
    data = size_model.prepare(panel)
    base, cand = list(size_model.FEATURES), features_for(spec)
    data = data.loc[common_rows(data, [tuple(base), tuple(cand)], "abs_move")]
    first_year = int(spec["evaluation"]["stage_1"]["first_test_year"])
    print(f"[EXP-109] stage 1: {len(data):,} rows usable by both", flush=True)

    runs = {}
    for name, feats in (("incumbent", base), ("candidate", cand)):
        started = time.time()
        runs[name] = walk_forward(
            data, feats, "abs_move", size_model.fit, first_test_year=first_year
        )
        print(f"  {name:10s} r={runs[name].metrics['r']:.4f} "
              f"mae={runs[name].metrics['mae']:.4f} ({time.time()-started:.0f}s)", flush=True)

    inc = runs["incumbent"].by_year.set_index("year")
    can = runs["candidate"].by_year.set_index("year")
    years = [y for y in inc.index if y in can.index]
    deltas = np.array([float(inc.loc[y, "mae"] - can.loc[y, "mae"]) for y in years])

    improved = int((deltas > 0).sum())
    p = float(stats.wilcoxon(deltas).pvalue) if len(deltas) > 5 else float("nan")
    # The 2020 failure mode from EXP-108, made an explicit criterion: drop the
    # single best year and see whether anything is left.
    without_best = np.delete(deltas, int(np.argmax(deltas)))
    criteria = spec["evaluation"]["stage_1"]["criteria"]

    out = {
        "rows": int(len(data)),
        "years": [int(y) for y in years],
        "per_year_mae_gain": {int(y): round(float(d), 5) for y, d in zip(years, deltas)},
        "years_improved": improved,
        "years_total": len(years),
        "mean_gain_pp": round(float(deltas.mean()), 5),
        "mean_gain_excluding_best_year": round(float(without_best.mean()), 5),
        "best_year": int(years[int(np.argmax(deltas))]),
        "wilcoxon_p": round(p, 5),
        "incumbent": dict(runs["incumbent"].metrics),
        "candidate": dict(runs["candidate"].metrics),
    }
    out["clears"] = bool(
        improved >= criteria["min_years_improved"]
        and p <= criteria["max_wilcoxon_p"]
        and without_best.mean() > 0
    )
    print(f"  improved {improved}/{len(years)} years, mean {out['mean_gain_pp']:+.4f}pp, "
          f"p={p:.4f}, without best year {out['mean_gain_excluding_best_year']:+.4f}pp "
          f"→ {'CLEARS' if out['clears'] else 'does not clear'}", flush=True)
    return out, runs


# --------------------------------------------------------------------------
# stage 2 — does it pick better trades
# --------------------------------------------------------------------------


def stage_2(spec: dict, runs: dict) -> dict:
    """Rank the replayed trades under each model and compare what they returned.

    The gate is untouched by a size-model change (it predicts ``ret`` from its
    own features), so what a better size model can move is the ORDERING a
    reader acts on. That is what this measures.
    """
    trades = store.read_table("trades")
    trades = trades[
        (trades["provenance"].astype(str) == "engine.replay")
        & (trades["strategy"] == "STR-THRU")
        & np.isclose(trades["fill_alpha"].astype(float), GATE_ALPHA)
    ].copy()
    trades["event_date"] = pd.to_datetime(trades["event_date"])
    trades["year"] = trades["event_date"].dt.year
    from engine.replay import legs_spot_dte

    if "spot_entry" not in trades.columns:
        trades["spot_entry"], trades["dte_entry"] = legs_spot_dte(trades)

    # The payoff fit needs its driver, and the Tier-2 trades schema deliberately
    # does not carry it — `abs_move` lives in the panel. This is the same join
    # `Scorer._enrich` does for the same reason.
    panel = load_panel()[["ticker", "date", "abs_move"]].rename(columns={"date": "event_date"})
    panel["event_date"] = pd.to_datetime(panel["event_date"])
    trades = trades.merge(panel, on=["ticker", "event_date"], how="left")
    print(f"[EXP-109] stage 2: {len(trades):,} STR-THRU trades at alpha={GATE_ALPHA}, "
          f"{int(trades['abs_move'].notna().sum()):,} with the payoff driver", flush=True)

    preds = {}
    for name, run in runs.items():
        frame = run.frame.copy()
        frame["event_date"] = pd.to_datetime(frame["date"])
        preds[name] = frame[["ticker", "event_date", "pred"]].rename(
            columns={"pred": f"pred_{name}"}
        )
    joined = trades
    for name in runs:
        joined = joined.merge(preds[name], on=["ticker", "event_date"], how="inner")
    print(f"  {len(joined):,} trades carry an OOS prediction from both models", flush=True)

    rows, skipped = [], []
    for year, group in joined.groupby("year"):
        # Causal payoff map: only trades that had CLOSED before this year began.
        # NOT wrapped in a bare except. An earlier version swallowed the
        # PayoffError raised by a missing driver column and reported "the model
        # did not select better trades" — a finding, from a test that never
        # ran. A stage that cannot run must say so, loudly.
        try:
            payoff = fit_payoff(
                trades, "STR-THRU", alpha=GATE_ALPHA,
                before=pd.Timestamp(f"{year}-01-01"),
            )
        except PayoffError as exc:
            # Genuinely insufficient history for an early year is expected and
            # is a skip; anything else is a bug and must surface.
            if "trades" not in str(exc).lower():
                raise
            skipped.append({"year": int(year), "reason": str(exc)[:120]})
            continue
        for name in runs:
            exit_value = (payoff.intercept + payoff.slope * group[f"pred_{name}"]) * group["spot_entry"]
            group = group.assign(**{f"exp_{name}": exit_value / group["entry_cost"] - 1.0})
        k = max(1, int(round(len(group) * TOP_QUINTILE)))
        row = {"year": int(year), "n": int(len(group)), "k": k,
               "all_mean_ret": round(float(group["ret"].mean()), 5)}
        for name in runs:
            top = group.nlargest(k, f"exp_{name}")
            row[f"top_{name}"] = round(float(top["ret"].mean()), 5)
        row["delta"] = round(row["top_candidate"] - row["top_incumbent"], 5)
        rows.append(row)

    by_year = pd.DataFrame(rows)
    if by_year.empty:
        return {"available": False,
                "reason": "no year produced a causal payoff map",
                "skipped": skipped}

    weights = by_year["k"]
    out = {
        "available": True,
        "trades": int(len(joined)),
        "years": int(len(by_year)),
        "by_year": rows,
        "top_quintile_incumbent": round(float(np.average(by_year["top_incumbent"], weights=weights)), 5),
        "top_quintile_candidate": round(float(np.average(by_year["top_candidate"], weights=weights)), 5),
        "all_trade_mean": round(float(by_year["all_mean_ret"].mean()), 5),
        "years_improved": int((by_year["delta"] > 0).sum()),
        "skipped_years": skipped,
    }
    out["delta"] = round(out["top_quintile_candidate"] - out["top_quintile_incumbent"], 5)
    out["clears"] = bool(out["delta"] >= 0)
    print(f"  top quintile: incumbent {out['top_quintile_incumbent']:+.4f} vs "
          f"candidate {out['top_quintile_candidate']:+.4f} "
          f"({out['delta']:+.4f}), better in {out['years_improved']}/{out['years']} years", flush=True)
    return out


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    results = {"spec_hash": lib.spec_hash(spec), "arm": spec["primary_spec"]["arm"]}

    s1, runs = stage_1(spec)
    results["stage_1"] = s1
    if not s1["clears"]:
        results["verdict"] = {
            "promote": False,
            "why": "stage 1 did not clear; stage 2 is gated on it and did not run",
        }
    else:
        s2 = stage_2(spec, runs)
        results["stage_2"] = s2
        results["verdict"] = {
            "promote": bool(s2.get("clears")),
            "why": (
                "both stages clear — write the promotion report"
                if s2.get("clears")
                else "stage 1 cleared but the better model did not select better trades"
            ),
        }

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1, default=str))
    print("\nverdict:", json.dumps(results["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
