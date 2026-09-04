#!/usr/bin/env python3
"""EXP-125 — replaying TWIN-P with a per-event tent width.

`engine.replay` prices ONE structure across a set of events, so a per-event
width is done by bucketing: every event's target width is rounded to 0.1% of
spot, and each bucket is replayed with the structure carrying that width. The
snap to the ticker's own ladder then happens inside the selector exactly as it
always has, so nothing about the geometry is special-cased here.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "EXP-123_twin_p_twin_peak_does_a_cheap_structure"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine import replay as replay_mod  # noqa: E402
from engine.build_trades import event_universe  # noqa: E402
from engine.features import load_panel  # noqa: E402
from engine.structures import twin_peak  # noqa: E402

import twinp  # noqa: E402  — EXP-123's filters, imported not copied

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
STRATEGY = "TWIN-P"
PREDICTIONS = ROOT / "data" / "models" / "size_v1_4_oos_predictions.parquet"

#: plateau centre is 1.5w, so w = forecast / 1.5. Registered in spec.yaml.
PLATEAU_CENTRE = 1.5
#: target widths are rounded to this share of spot before bucketing
BUCKET = 0.001
WIDTH_MIN, WIDTH_MAX = 0.005, 0.15


def targets(arm: str) -> pd.DataFrame:
    """Per-event target width, from the OOS forecast or from implied."""
    panel = load_panel()[["ticker", "date", "implied_move"]].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    if arm == "predicted":
        pred = pd.read_parquet(PREDICTIONS)
        pred["date"] = pd.to_datetime(pred["date"])
        src = pred.rename(columns={"pred_abs_move": "forecast"})[["ticker", "date", "forecast"]]
    elif arm == "implied":
        src = panel.rename(columns={"implied_move": "forecast"})[["ticker", "date", "forecast"]]
    else:
        raise ValueError(arm)
    src = src[src["forecast"].notna() & (src["forecast"] > 0)].copy()
    src["width_moneyness"] = (src["forecast"] / 100.0) / PLATEAU_CENTRE
    src = src[(src["width_moneyness"] >= WIDTH_MIN) & (src["width_moneyness"] <= WIDTH_MAX)]
    src["bucket"] = (src["width_moneyness"] / BUCKET).round().astype(int)
    return src


def trades_path(arm: str) -> Path:
    return RESULTS / f"trades_twin_p_{arm}.parquet"


def build(arm: str, *, force: bool = False) -> pd.DataFrame:
    path = trades_path(arm)
    if path.exists() and not force:
        out = pd.read_parquet(path)
        print(f"[sized] {arm}: {out['event_id'].nunique():,} events from cache", flush=True)
        return out

    started = time.time()
    events = event_universe()
    events["event_date"] = pd.to_datetime(events["event_date"])
    tgt = targets(arm)
    events = events.merge(tgt, left_on=["ticker", "event_date"],
                          right_on=["ticker", "date"], how="inner")
    print(f"[sized] {arm}: {len(events):,} events carry a {arm} width "
          f"({events['bucket'].nunique()} distinct buckets)", flush=True)

    # One chain index per YEAR, shared by every width bucket in it. Letting
    # each bucket load its own would re-scan the year partitions a hundred
    # times over for the same rows.
    parts, skipped = [], {}
    plan_all = replay_mod.filter_plan_by_availability(
        replay_mod.plan_events(twin_peak(width_moneyness=0.05), events))
    keyframe = plan_all.frame.set_index("event_id")
    events = events[events["event_id"].isin(set(plan_all.frame["event_id"]))].copy()
    events["_year"] = events["event_date"].dt.year

    for year, block in events.groupby("_year", sort=True):
        rows = keyframe.reindex(block["event_id"]).dropna(subset=["entry_date"])
        keys = set()
        for col in ("entry_date", "exit_date"):
            keys |= {(t, d) for t, d in zip(rows["ticker"], rows[col])}
        index = replay_mod.load_chain_index(keys, progress_every=0)
        for bucket, sub in block.groupby("bucket", sort=True):
            res = replay_mod.replay(STRATEGY, sub,
                                    structure=twin_peak(width_moneyness=bucket * BUCKET),
                                    index=index, progress_every=0)
            if len(res.trades):
                parts.append(res.trades)
            for k, v in res.skipped.items():
                skipped[k] = skipped.get(k, 0) + int(v)
        del index
        print(f"[sized] {arm} {year}: {sum(len(p) for p in parts):,} rows, "
              f"{block['bucket'].nunique()} buckets, {time.time()-started:.0f}s", flush=True)

    trades = (pd.concat(parts, ignore_index=True) if parts else replay_mod._empty_trades())
    trades = twinp.attach_filters(trades)
    trades.to_parquet(path, index=False)
    (RESULTS / f"skips_{arm}.json").write_text(json.dumps(skipped, indent=1))
    print(f"[sized] {arm}: {trades['event_id'].nunique():,} events priced, skips {skipped}",
          flush=True)
    return trades
