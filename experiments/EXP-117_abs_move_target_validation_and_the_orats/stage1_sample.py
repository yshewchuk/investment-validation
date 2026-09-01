#!/usr/bin/env python3
"""EXP-117 Stage 1 — stratified validation sample of tickers, fetched from
Polygon one range call per ticker.

    python3 stage1_sample.py            # writes results/stage1_sample.json
    python3 stage1_pull.py              # resumable pull, Tier-1 cached

Sample design (registered in spec.yaml):
  * forced: every ticker with a Stage-0 >5pp disagreement (the adjudication
    targets), deduplicated;
  * stratified: ~18 tickers per (decade x mcap bucket) cell, drawn from the
    oquants panel universe with >=1 event in that decade and last-observed
    mcap in the bucket; random_state=117, deterministic;
  * one Polygon call per ticker buys its whole daily history (adjusted=false),
    which validates every event of the ticker at once.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/investing-plan")
sys.path.insert(0, str(ROOT))

from engine.data import store  # noqa: E402

HERE = Path(__file__).resolve().parent
EV = pd.read_parquet(HERE / "results" / "stage0_events.parquet")


def last_mcap() -> pd.Series:
    dm = store.read_table("daily_market", columns=["ticker", "date", "mcap_usd"])
    dm = dm[dm["mcap_usd"].notna()]
    dm = dm.sort_values("date").groupby("ticker").tail(1)
    return dm.set_index("ticker")["mcap_usd"]


def main() -> None:
    events = EV[["ticker", "date", "session"]].copy()
    events["year"] = events["date"].dt.year

    def decade(y):
        if y <= 2009:
            return "2007-2009"
        if y <= 2014:
            return "2010-2014"
        if y <= 2019:
            return "2015-2019"
        return "2020-2026"

    events["decade"] = events["year"].map(decade)

    big = EV[(EV.move_orats - EV.oq_move).abs() > 5]
    forced = sorted(set(big["ticker"]))
    print(f"forced (stage0 >5pp) tickers: {len(forced)}")

    mcap = last_mcap()

    def bucket(v):
        if not np.isfinite(v):
            return "unknown"
        if v < 1e9:
            return "<1B"
        if v < 10e9:
            return "1-10B"
        return ">10B"

    events["mcap_bucket"] = events["ticker"].map(mcap).map(bucket)

    rng = np.random.default_rng(117)
    picked: set[str] = set(forced)
    cells = {}
    for (dec, bkt), g in events.groupby(["decade", "mcap_bucket"]):
        if bkt == "unknown":
            continue
        tickers = sorted(set(g["ticker"]) - picked)
        n_take = min(18, len(tickers))
        if n_take:
            idx = rng.choice(len(tickers), size=n_take, replace=False)
            take = [tickers[i] for i in sorted(idx)]
        else:
            take = []
        cells[f"{dec} x {bkt}"] = {
            "eligible": len(tickers),
            "picked": take,
        }
        picked |= set(take)

    sample = sorted(picked)
    n_events = events[events["ticker"].isin(sample)]
    out = {
        "n_tickers": len(sample),
        "n_forced": len(forced),
        "n_stratified": len(sample) - len([t for t in sample if t in set(forced)]),
        "tickers": sample,
        "forced": forced,
        "cells": {k: {"eligible": v["eligible"], "n_picked": len(v["picked"]),
                      "picked": v["picked"]} for k, v in cells.items()},
        "events_covered": int(len(n_events)),
        "events_total": int(len(events)),
        "events_by_decade": n_events.groupby("decade").size().to_dict(),
        "events_by_bucket": n_events.groupby("mcap_bucket").size().to_dict(),
    }
    path = HERE / "results" / "stage1_sample.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}: {len(sample)} tickers, {len(n_events):,} events")
    print(json.dumps({k: v for k, v in out.items() if k != "tickers"}, indent=1)[:1500])


if __name__ == "__main__":
    main()
