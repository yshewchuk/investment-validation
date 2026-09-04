#!/usr/bin/env python3
"""EXP-126 — pricing three twin-peak shapes over one event universe.

EXP-125 built one shape at a time. Here three shapes have to be priced on the
SAME events so a per-event choice between them means anything, and the chain
index is what dominates the wall clock — so the loop is year-outermost: load a
year of chains once, price all three shapes through it, drop it.

Per-event widths are handled the way EXP-125 handled them, by bucketing: each
event's target spacing is rounded to 0.1% of spot and every bucket is replayed
with the structure carrying that spacing. The snap to the ticker's own ladder
then happens inside the selector exactly as it always has.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine import replay as replay_mod  # noqa: E402
from engine.build_trades import event_universe  # noqa: E402
from engine.data import store  # noqa: E402
from engine.structures import twin_peak, twin_peak_5  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PREDICTIONS = ROOT / "data" / "models" / "size_v1_4_oos_predictions.parquet"

#: Registered in spec.yaml before any of this ran.
MAX_REL_SPREAD = 0.25
MCAP_FLOOR = 10e9
#: Target spacings are rounded to this share of spot before bucketing.
BUCKET = 0.001
#: Same bounds EXP-125 used, applied to the SPACING rather than the forecast.
WIDTH_MIN, WIDTH_MAX = 0.005, 0.15


class Shape:
    """One twin-peak geometry: how to build it, size it, and read its peak.

    ``peak_multiple`` is the terminal payoff at the peak in units of the
    spacing, and it is the only thing that differs in the reward term. The
    registered rule is ``cost < peak / 2`` — max profit beats max loss — which
    on the seven-strike shape (peak ``2w``) is exactly EXP-123's ``cost < w``.
    """

    def __init__(self, key, strategy, factory, peak_multiple, plateau_centre,
                 dead_multiple, label):
        self.key = key
        self.strategy = strategy
        self.factory = factory
        self.peak_multiple = float(peak_multiple)
        #: Where the payoff is centred, in units of spacing — what the forecast
        #: is placed on. The seven-strike shape has a plateau from 1a to 2a so
        #: its centre is 1.5a; both five-strike shapes peak at exactly 1a.
        self.plateau_centre = float(plateau_centre)
        #: |move| / spacing at and beyond which the shape pays nothing.
        self.dead_multiple = float(dead_multiple)
        self.label = label

    def spacing_target(self, forecast_pct: pd.Series) -> pd.Series:
        """Spacing as a share of spot that puts the payoff centre on the forecast."""
        return (forecast_pct / 100.0) / self.plateau_centre

    def build(self, spacing: float):
        return self.factory(spacing)

    def trades_path(self) -> Path:
        return RESULTS / f"trades_{self.key}.parquet"


SHAPES = {
    s.key: s
    for s in (
        Shape("seven", "TWIN-P",
              lambda w: twin_peak(width_moneyness=w),
              peak_multiple=2.0, plateau_centre=1.5, dead_multiple=4.0,
              label="seven strikes, plateau 1a-2a, wings +/-4a"),
        Shape("five_wide", "TWIN-P5",
              lambda a: twin_peak_5(wing_multiple=3, width_moneyness=a),
              peak_multiple=2.0, plateau_centre=1.0, dead_multiple=3.0,
              label="five strikes, peak at +/-1a, wings +/-3a"),
        Shape("five_tight", "TWIN-P5",
              lambda a: twin_peak_5(wing_multiple=2, width_moneyness=a),
              peak_multiple=1.0, plateau_centre=1.0, dead_multiple=2.0,
              label="five strikes, peak at +/-1a, wings +/-2a"),
    )
}


def forecasts() -> pd.DataFrame:
    """Walk-forward OOS predicted absolute move, per (ticker, date)."""
    pred = pd.read_parquet(PREDICTIONS)
    pred["date"] = pd.to_datetime(pred["date"])
    pred = pred.rename(columns={"pred_abs_move": "forecast"})
    pred = pred[pred["forecast"].notna() & (pred["forecast"] > 0)]
    return pred[["ticker", "date", "forecast"]]


def build_all(*, force: bool = False) -> dict[str, pd.DataFrame]:
    """Price every shape on one event universe, one year of chains at a time."""
    cached = {k: s.trades_path() for k, s in SHAPES.items()}
    if all(p.exists() for p in cached.values()) and not force:
        out = {}
        for key, path in cached.items():
            out[key] = pd.read_parquet(path)
            print(f"[shapes] {key}: {out[key]['event_id'].nunique():,} events from cache",
                  flush=True)
        return out

    started = time.time()
    events = event_universe()
    events["event_date"] = pd.to_datetime(events["event_date"])
    events = events.merge(forecasts(), left_on=["ticker", "event_date"],
                          right_on=["ticker", "date"], how="inner")
    print(f"[shapes] {len(events):,} events carry an OOS forecast", flush=True)

    # Entry and exit dates depend on the offsets and on chain availability, not
    # on the geometry, so ONE plan serves all three shapes. Built off the
    # seven-strike factory purely so this matches EXP-125's universe exactly.
    plan_all = replay_mod.filter_plan_by_availability(
        replay_mod.plan_events(twin_peak(width_moneyness=0.05), events))
    keyframe = plan_all.frame.set_index("event_id")
    events = events[events["event_id"].isin(set(plan_all.frame["event_id"]))].copy()
    events["_year"] = events["event_date"].dt.year
    print(f"[shapes] {len(events):,} of those have both chains, "
          f"{events['_year'].nunique()} years", flush=True)

    for key, shape in SHAPES.items():
        target = shape.spacing_target(events["forecast"])
        ok = (target >= WIDTH_MIN) & (target <= WIDTH_MAX)
        events[f"bucket_{key}"] = np.where(
            ok, (target / BUCKET).round(), np.nan)
        print(f"[shapes] {key}: {int(ok.sum()):,} events sizeable "
              f"({events[f'bucket_{key}'].nunique()} buckets)", flush=True)

    parts: dict[str, list] = {k: [] for k in SHAPES}
    skipped: dict[str, dict] = {k: {} for k in SHAPES}

    for year, block in events.groupby("_year", sort=True):
        rows = keyframe.reindex(block["event_id"]).dropna(subset=["entry_date"])
        keys = set()
        for col in ("entry_date", "exit_date"):
            keys |= {(t, d) for t, d in zip(rows["ticker"], rows[col])}
        index = replay_mod.load_chain_index(keys, progress_every=0)
        for key, shape in SHAPES.items():
            col = f"bucket_{key}"
            sizeable = block[block[col].notna()]
            for bucket, sub in sizeable.groupby(col, sort=True):
                res = replay_mod.replay(
                    shape.strategy, sub, structure=shape.build(bucket * BUCKET),
                    index=index, progress_every=0)
                if len(res.trades):
                    parts[key].append(res.trades)
                for k, v in res.skipped.items():
                    skipped[key][k] = skipped[key].get(k, 0) + int(v)
            print(f"[shapes] {year} {key}: {sum(len(p) for p in parts[key]):,} rows "
                  f"cumulative, {time.time() - started:.0f}s", flush=True)
        del index

    out = {}
    for key, shape in SHAPES.items():
        trades = (pd.concat(parts[key], ignore_index=True) if parts[key]
                  else replay_mod._empty_trades())
        trades = attach(trades, shape)
        trades.to_parquet(shape.trades_path(), index=False)
        (RESULTS / f"skips_{key}.json").write_text(json.dumps(skipped[key], indent=1))
        print(f"[shapes] {key}: {trades['event_id'].nunique():,} events priced, "
              f"skips {skipped[key]}", flush=True)
        out[key] = trades
    return out


def attach(trades: pd.DataFrame, shape: Shape) -> pd.DataFrame:
    """Add the geometry, the three registered filter terms, and reward:risk.

    Everything here is entry-close arithmetic on mid quotes — nothing fitted,
    nothing that needs the outcome — which is what lets the choice between
    shapes be made once, at the decision, rather than in hindsight.
    """
    if trades.empty:
        return trades
    t = trades.copy()
    t["shape"] = shape.key
    docs = t["legs"].apply(lambda b: json.loads(b) if isinstance(b, str) else {})
    strikes = docs.apply(lambda d: {l["name"]: float(l["strike"])
                                    for l in (d.get("entry") or [])})
    t["anchor"] = strikes.apply(lambda d: d.get("atm", np.nan))
    t["w"] = strikes.apply(lambda d: d.get("up1", np.nan)) - t["anchor"]
    t["spot_entry"] = docs.apply(lambda d: float(d.get("spot_entry", np.nan)))

    def rel_spread(d):
        vals = []
        for leg in (d.get("entry") or []):
            bid, ask = float(leg["bid"]), float(leg["ask"])
            mid = 0.5 * (bid + ask)
            if mid > 0:
                vals.append((ask - bid) / mid)
        return float(np.mean(vals)) if vals else np.nan

    t["rel_spread"] = docs.apply(rel_spread)
    t["cost"] = pd.to_numeric(t["entry_cost"], errors="coerce")
    t["peak"] = shape.peak_multiple * t["w"]
    t["max_profit"] = t["peak"] - t["cost"]
    # reward:risk, the one number the chooser ranks on. > 1 is the registered
    # reward term written so it reduces to EXP-123's `cost < w` on the
    # seven-strike shape, whose peak is 2w.
    t["rr"] = t["max_profit"] / t["cost"]
    t["year"] = pd.to_datetime(t["event_date"]).dt.year

    sec = store.read_table("securities", years=range(2017, 2027),
                           columns=["ticker", "year", "mcap_usd"])
    t = t.merge(sec, on=["ticker", "year"], how="left")

    t["f_reward"] = t["cost"] < (t["peak"] / 2.0)
    t["f_spread"] = t["rel_spread"] <= MAX_REL_SPREAD
    t["f_mcap"] = t["mcap_usd"] >= MCAP_FLOOR
    t["passes"] = t["f_reward"] & t["f_spread"] & t["f_mcap"]

    # Where the print landed, in units of the shape's own spacing, and how far
    # the snapped geometry drifted from what the forecast asked for. Both are
    # required outputs; both are computed here so every arm reads them the same.
    t["landed"] = ((pd.to_numeric(t["spot_exit"], errors="coerce") - t["anchor"])
                   / t["w"]).abs()
    t["dead"] = t["landed"] >= shape.dead_multiple
    t["centre"] = t["landed"] <= 0.25
    #: Where the payoff centre actually ended up, as a share of spot, once the
    #: ladder had its say — and how far that is from the move the forecast
    #: asked us to put it on. This is the second of the two criteria the
    #: experiment registers a chooser for.
    t["peak_pct_spot"] = 100.0 * shape.plateau_centre * t["w"] / t["spot_entry"]
    t["event_date"] = pd.to_datetime(t["event_date"])
    t = t.merge(forecasts().rename(columns={"date": "event_date"}),
                on=["ticker", "event_date"], how="left")
    t["fit_err"] = (t["peak_pct_spot"] / t["forecast"] - 1.0).abs()
    return t


if __name__ == "__main__":
    build_all(force="--force" in sys.argv)
