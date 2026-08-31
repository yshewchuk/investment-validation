#!/usr/bin/env python3
"""EXP-109 stage 2 — what the promoted model does, and does not, change.

    python3 experiments/EXP-109_absolute_prior_move_features_registered/run_stage2.py

**Part A is the real stage 2.** In this system the size model does not select
trades: the registered GATE does, and the gate reads panel/daily features and
the premium — never a prediction. So the question "does promoting the size model
change the backtest" has an exact answer, and Part A proves it by re-running
EXP-105's own registered-gate evaluation under the current champion and checking
it reproduces EXP-105's published numbers to the digit. Same trades, same P&L,
same Monte Carlo.

**Part B is an explicitly counterfactual side-question**, kept because it is
worth knowing and clearly labelled because it is not the system: *what if
selection were driven by predicted expected PnL instead of by the gate?* The
model is wrapped as a walk-forward Gate keeping the top quintile by predicted
expected PnL, and the suite is run once per size model.

Part B's reports must never be read as evaluations of STR-THRU as it is actually
traded. An earlier version of this file ran only Part B, and its two arms were
compared as though one of them were the live system — which produced a
confident, entirely counterfactual recommendation to roll the promotion back.
The tell was the trade count: 7,853 against EXP-105's 7,620. Same strategy, same
fills, a different n can only mean a different selector.

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


def stamp_counterfactual(run_dir: Path) -> int:
    """Watermark every figure in a counterfactual run, on the image itself.

    The report title says COUNTERFACTUAL and the verdict section says it again,
    and neither travels with the picture. An equity curve opened straight out of
    `figures/` looks exactly like STR-THRU's — same shape, same axes, same
    plausible drawdown — and that is how a chart of a system nobody trades ends
    up in front of someone as though it were the system. The marker has to be on
    the artifact, not just around it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    stamped = 0
    for png in sorted((run_dir / "figures").glob("*.png")):
        image = mpimg.imread(png)
        height, width = image.shape[0], image.shape[1]
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.5, 0.5, "COUNTERFACTUAL\nnot STR-THRU as traded",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=max(14, width / 34), color="#d0342c", alpha=0.28,
            rotation=24, fontweight="bold", linespacing=1.4,
        )
        # Banded across the TOP: the generator already writes a falsification
        # caption along the bottom edge, and two lines of red text on top of
        # each other is how a warning becomes unreadable.
        # Short enough to fit the narrowest figure the generator emits. An
        # earlier version ran off both edges, which turns a warning into noise.
        ax.text(
            0.5, 0.988,
            "COUNTERFACTUAL — not STR-THRU as traded",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=max(9, width / 78), color="#ffffff", fontweight="bold",
            bbox={"facecolor": "#d0342c", "edgecolor": "none", "pad": 3.5, "alpha": 0.93},
        )
        fig.savefig(png, dpi=dpi)
        plt.close(fig)
        stamped += 1
    return stamped


def build(features, name, panel, trades):
    state = SizeModelSelector(features, panel, trades, name)
    return Gate(fit=state.fit, select=state.select, name=name), state


#: EXP-105's published registered-gate numbers, the invariance target.
EXP105_PUBLISHED = {
    "n": 7620,
    "mean": 0.040393728522348626,
    "sharpe_trade": 2.233590012932244,
}


def part_a_gate_invariance() -> dict:
    """The real stage 2: prove the gate-selected backtest is untouched.

    If this ever stops reproducing, a size-model change HAS reached trade
    selection and the promotion protocol's OOS-mean-and-Sharpe rule becomes
    live. As long as it reproduces, that rule is not applicable — the traded set
    is identical by construction — and the promotion rests on stage 1 alone.
    """
    d105 = ROOT / "experiments/EXP-105_str_thru_validation_registered_mid_fill"
    spec = lib.load_spec(d105 / "spec.yaml")
    trades = common.load_engine_trades(STRATEGY)
    dataset = common.gate_dataset(STRATEGY, trades, d105 / "results")
    gate, _ = common.make_registered_gate(STRATEGY, dataset)
    out = HERE / "results" / "gate_invariance"
    out.mkdir(parents=True, exist_ok=True)
    result = evaluate(
        dict(spec, id="EXP-109-invariance"), trades, gate=gate, run_dir=out,
        spy_daily=common.load_spy_daily(), stress=False, write_report=False,
    )
    h = result.results["headline"]
    got = {k: h[k] for k in EXP105_PUBLISHED}
    matches = all(
        (isinstance(v, int) and got[k] == v) or
        (not isinstance(v, int) and abs(got[k] - v) < 1e-12)
        for k, v in EXP105_PUBLISHED.items()
    )
    from engine.models.registry import load_registry

    champion = load_registry().champion("size").id
    print(f"[EXP-109] part A: registered gate under champion {champion} → "
          f"n={got['n']}, mean={got['mean']:.6f}, sharpe={got['sharpe_trade']:.6f} "
          f"→ {'REPRODUCES EXP-105' if matches else 'DIVERGES FROM EXP-105'}", flush=True)
    return {"champion": champion, "published": EXP105_PUBLISHED, "observed": got,
            "reproduces": bool(matches)}


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    invariance = part_a_gate_invariance()
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
        arm_spec = dict(
            spec,
            id=f"{spec['id']}-counterfactual-{label}",
            title=(
                f"COUNTERFACTUAL — STR-THRU selected by {label} predicted expected "
                "PnL, NOT by the registered gate"
            ),
            hypothesis=(
                "This is not the system as traded. STR-THRU's selector is the "
                "registered gate gate_midfill_str_thru, which never reads the size "
                "model. This run replaces that gate with a top-quintile ranking on "
                f"{label}'s predicted expected PnL, to answer a side question: how "
                "would selection by predicted PnL compare? Its numbers must not be "
                "read as STR-THRU's, and its trade count differs from EXP-105's for "
                "exactly that reason."
            ),
        )
        result = evaluate(
            arm_spec, trades, gate=gate, run_dir=run_dir,
            repricer=repricer, spy_daily=spy, input_files=input_files,
        )
        stamped = stamp_counterfactual(run_dir)
        out[label] = {
            "report": str(result.report_path),
            "results": result.results,
            "folds": state.folds,
            "figures_stamped": stamped,
        }
        print(f"[EXP-109] {label}: {stamped} figure(s) watermarked", flush=True)
        print(f"[EXP-109] {label} report: {result.report_path}", flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "stage2_evaluations.json").write_text(
        json.dumps({"gate_invariance": invariance, "counterfactual": out},
                   indent=1, default=str)
    )
    return 0 if invariance["reproduces"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
