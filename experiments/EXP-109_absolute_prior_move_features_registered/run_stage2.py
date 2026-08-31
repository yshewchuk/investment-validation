#!/usr/bin/env python3
"""EXP-109 stage 2 — the promoted model through the full evaluation suite.

    python3 experiments/EXP-109_absolute_prior_move_features_registered/run_stage2.py

Stage 1 answered "does it predict |move| better". This answers the question the
promotion protocol is actually written on: **what happens to the backtest, the
Monte Carlo and the stress battery** when the board is ranked by the new model
instead of the old one.

A size-model change cannot alter which trades EXIST or what they returned — the
trades are priced from real ORATS chains and their P&L is fixed. What it alters
is SELECTION: the ordering a reader acts on. So the model is wrapped as a
walk-forward :class:`~engine.evaluate.Gate` that keeps the top quintile by
predicted expected PnL, and the same suite EXP-105 runs is run twice — once
selecting with size_v1_3, once with size_v1_4. Two standard reports, same
format, so every line is comparable.

Causality is the harness's to enforce and it does: ``fit`` only ever sees years
strictly before the year being traded, and the payoff map that turns a
prediction into an expected PnL is fitted only on trades that had CLOSED before
that year began.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.evaluate import Gate, evaluate  # noqa: E402
from engine.features import add_absolute_features, load_panel  # noqa: E402
from engine.models.training import size_model  # noqa: E402
from engine.payoff import PayoffError, fit_payoff  # noqa: E402
from experiments import common, lib  # noqa: E402

HERE = Path(__file__).resolve().parent
STRATEGY = "STR-THRU"
TOP_FRACTION = 0.2

#: The incumbent's feature list, kept literal. `size_model.FEATURES` is now the
#: promoted set, so reading the old one from the module would silently compare
#: the challenger against itself.
INCUMBENT_FEATURES = (
    "ema12r_abs", "mean_prior_move", "signed_streak", "dist_high", "dist_ema",
    "spy_vol20", "spy_dd252", "mean_prior_implied_move", "or_implied",
    "or_rvol30", "mcap_log",
)


class SizeModelSelector:
    """Rank a year's trades by predicted expected PnL; keep the top quintile.

    ``fit`` retrains the size model on panel events from the years it is given —
    the harness guarantees those are strictly before the year being selected —
    and fits the payoff line on trades that had closed by then. ``select`` then
    prices every candidate: an expected PnL is the payoff line evaluated at the
    prediction, against the premium that trade actually paid.
    """

    def __init__(self, features, panel, trades, name):
        self.features = list(features)
        self.panel = panel
        self.trades = trades
        self.name = name
        self.model = None
        self.payoff = None
        self.folds = []

    def fit(self, train: pd.DataFrame) -> None:
        cutoff = int(pd.to_datetime(train["event_date"]).dt.year.max()) + 1
        rows = self.panel[self.panel["year"] < cutoff]
        usable = rows[self.features + ["abs_move"]].apply(pd.to_numeric, errors="coerce")
        keep = np.isfinite(usable.to_numpy(dtype=float)).all(axis=1)
        rows = rows[keep]
        X = rows[self.features].to_numpy(dtype=float)
        y = rows["abs_move"].to_numpy(dtype=float)
        self.model = size_model.fit(X, y)
        try:
            self.payoff = fit_payoff(
                self.trades, STRATEGY, alpha=0.5, before=pd.Timestamp(f"{cutoff}-01-01")
            )
        except PayoffError:
            self.payoff = None
        self.folds.append({"cutoff": cutoff, "train_rows": int(len(rows)),
                           "payoff": self.payoff is not None})

    def _expected(self, rows: pd.DataFrame) -> pd.Series:
        feat = rows[self.features].apply(pd.to_numeric, errors="coerce")
        ok = np.isfinite(feat.to_numpy(dtype=float)).all(axis=1)
        out = pd.Series(np.nan, index=rows.index)
        if not ok.any() or self.model is None or self.payoff is None:
            return out
        pred = self.model.predict(feat[ok].to_numpy(dtype=float))
        exit_value = (self.payoff.intercept + self.payoff.slope * pred) * rows.loc[ok, "spot_entry"]
        out.loc[ok] = exit_value / rows.loc[ok, "entry_cost"] - 1.0
        return out

    def select(self, rows: pd.DataFrame) -> pd.Series:
        expected = self._expected(rows)
        scored = expected.dropna()
        if scored.empty:
            # No usable prediction is not a reason to take every trade.
            return pd.Series(False, index=rows.index)
        k = max(1, int(round(len(scored) * TOP_FRACTION)))
        keep = set(scored.nlargest(k).index)
        return pd.Series([i in keep for i in rows.index], index=rows.index)


def build(features, name, panel, trades):
    state = SizeModelSelector(features, panel, trades, name)
    return Gate(fit=state.fit, select=state.select, name=name), state


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    panel = add_absolute_features(load_panel())
    panel = size_model.prepare(panel)

    trades = common.load_engine_trades(STRATEGY)
    # The payoff driver and the entry spot, joined the way Scorer._enrich does.
    keys = panel[["ticker", "date", "abs_move"]].rename(columns={"date": "event_date"})
    keys["event_date"] = pd.to_datetime(keys["event_date"])
    trades["event_date"] = pd.to_datetime(trades["event_date"])
    trades = trades.merge(keys, on=["ticker", "event_date"], how="left", suffixes=("", "_panel"))
    if "spot_entry" not in trades.columns:
        from engine.replay import legs_spot_dte

        trades["spot_entry"], trades["dte_entry"] = legs_spot_dte(trades)

    # Every size-model feature, per trade, so `select` can price a candidate.
    feature_cols = sorted(set(INCUMBENT_FEATURES) | set(size_model.FEATURES))
    have = [c for c in feature_cols if c in panel.columns]
    joined = panel[["ticker", "date"] + have].rename(columns={"date": "event_date"})
    joined["event_date"] = pd.to_datetime(joined["event_date"])
    trades = trades.merge(joined, on=["ticker", "event_date"], how="left", suffixes=("", "_p"))
    print(f"[EXP-109] {len(trades):,} rows / {trades['event_id'].nunique():,} events", flush=True)

    spy = common.load_spy_daily()
    repricer = common.make_repricer(STRATEGY)
    input_files = sorted((paths.CURATED / "trades").glob("year=*/part-*.parquet"))

    out = {}
    for label, features in (
        ("size_v1_3", INCUMBENT_FEATURES),
        ("size_v1_4", tuple(size_model.FEATURES)),
    ):
        print(f"\n[EXP-109] evaluating selection by {label} …", flush=True)
        gate, state = build(features, f"top{int(TOP_FRACTION*100)}pct@{label}", panel, trades)
        run_dir = HERE / f"eval_{label}"
        run_dir.mkdir(exist_ok=True)
        # The spec's id is stamped per arm so the two runs are distinguishable
        # in the ledger and neither is mistaken for the registered primary.
        arm_spec = dict(spec, id=f"{spec['id']}-{label}")
        result = evaluate(
            arm_spec, trades, gate=gate, run_dir=run_dir,
            repricer=repricer, spy_daily=spy, input_files=input_files,
        )
        out[label] = {
            "report": str(result.report_path),
            "results": result.results,
            "folds": state.folds,
        }
        print(f"[EXP-109] {label} report: {result.report_path}", flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "stage2_evaluations.json").write_text(
        json.dumps(out, indent=1, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
