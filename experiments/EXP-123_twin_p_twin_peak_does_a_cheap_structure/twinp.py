#!/usr/bin/env python3
"""EXP-123 — building the TWIN-P trade set and applying the registered filters.

Pricing goes through engine.replay like every other structure. The four entry
filters are arithmetic on the entry close — no model, nothing fitted — so they
are expressed as a :class:`engine.evaluate.Gate` whose ``fit`` is a no-op. That
is not a formality: it routes the rule through the harness's own walk-forward,
so the anti-selection baseline and the per-year gated/ungated split come from
the same code path every other strategy uses, rather than from a comparison
this experiment writes for itself.
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
from engine.evaluate import Gate  # noqa: E402
from engine.structures import twin_peak  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
STRATEGY = "TWIN-P"

#: Registered in spec.yaml before any of this ran.
MAX_REL_SPREAD = 0.25
MCAP_FLOOR = 10e9
MEGA_CAP = 100e9


def trades_path(steps: int) -> Path:
    return RESULTS / f"trades_twin_p_s{steps}.parquet"


def build(steps: int, *, force: bool = False) -> pd.DataFrame:
    """Replay TWIN-P at ``steps``, one year of chains in memory at a time."""
    path = trades_path(steps)
    if path.exists() and not force:
        out = pd.read_parquet(path)
        print(f"[twinp] steps={steps}: {out['event_id'].nunique():,} events from cache",
              flush=True)
        return out

    started = time.time()
    events = event_universe()
    plan = replay_mod.filter_plan_by_availability(
        replay_mod.plan_events(twin_peak(steps=steps), events))
    frame = plan.frame
    print(f"[twinp] steps={steps}: {len(frame):,} events have both chains", flush=True)

    parts, skipped = [], {}
    for year, block in frame.groupby(frame["entry_date"].dt.year, sort=True):
        keys = set()
        for col in ("entry_date", "exit_date"):
            keys |= {(t, d) for t, d in zip(block["ticker"], block[col])}
        index = replay_mod.load_chain_index(keys, progress_every=0)
        sub = events[events["event_id"].isin(set(block["event_id"]))]
        res = replay_mod.replay(STRATEGY, sub, structure=twin_peak(steps=steps),
                                index=index, progress_every=0)
        if len(res.trades):
            parts.append(res.trades)
        for k, v in res.skipped.items():
            skipped[k] = skipped.get(k, 0) + int(v)
        del index
        print(f"[twinp] steps={steps} {year}: "
              f"{sum(len(p) for p in parts):,} rows, {time.time()-started:.0f}s", flush=True)

    trades = (pd.concat(parts, ignore_index=True) if parts
              else replay_mod._empty_trades())
    trades = attach_filters(trades)
    trades.to_parquet(path, index=False)
    (RESULTS / f"skips_s{steps}.json").write_text(json.dumps(skipped, indent=1))
    print(f"[twinp] steps={steps}: {trades['event_id'].nunique():,} events priced, "
          f"skips {skipped}", flush=True)
    return trades


def attach_filters(trades: pd.DataFrame) -> pd.DataFrame:
    """Add w, spread, mcap and the registered pass/fail — all entry-close data."""
    if trades.empty:
        return trades
    t = trades.copy()
    docs = t["legs"].apply(lambda b: json.loads(b) if isinstance(b, str) else {})
    strikes = docs.apply(lambda d: {l["name"]: float(l["strike"])
                                    for l in (d.get("entry") or [])})
    t["w"] = strikes.apply(lambda d: d.get("up1", np.nan) - d.get("atm", np.nan))
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
    t["max_profit"] = 2.0 * t["w"] - t["cost"]
    t["year"] = pd.to_datetime(t["event_date"]).dt.year

    sec = store.read_table("securities", years=range(2017, 2027),
                           columns=["ticker", "year", "mcap_usd"])
    t = t.merge(sec, on=["ticker", "year"], how="left")

    # The registered rule, term by term, so each one's cost is countable.
    t["f_reward"] = t["cost"] < t["w"]
    t["f_spread"] = t["rel_spread"] <= MAX_REL_SPREAD
    t["f_mcap"] = t["mcap_usd"] >= MCAP_FLOOR
    t["passes"] = t["f_reward"] & t["f_spread"] & t["f_mcap"]
    t["mega"] = t["mcap_usd"] >= MEGA_CAP
    return t


def make_filter_gate(column: str = "passes") -> Gate:
    """The registered entry rule as a Gate. ``fit`` learns nothing, by design."""
    def fit(_train: pd.DataFrame) -> None:
        return None

    def select(rows: pd.DataFrame) -> pd.Series:
        return rows[column].fillna(False).astype(bool)

    return Gate(fit=fit, select=select, name=f"entry-rule[{column}]")
