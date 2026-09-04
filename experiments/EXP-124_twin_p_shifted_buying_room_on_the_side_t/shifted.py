#!/usr/bin/env python3
"""EXP-124 — building TWIN-P at several anchor offsets.

Same entry filters and same attach_filters as EXP-123, imported rather than
re-implemented, so the comparison against its +2.61% baseline is like for like.
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
from engine.structures import twin_peak  # noqa: E402

import twinp  # noqa: E402  — EXP-123's filters, imported not copied

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
STRATEGY = "TWIN-P"


def trades_path(offset: int) -> Path:
    return RESULTS / f"trades_twin_p_off{offset}.parquet"


def build(offset: int, *, force: bool = False) -> pd.DataFrame:
    path = trades_path(offset)
    if path.exists() and not force:
        out = pd.read_parquet(path)
        print(f"[shift] offset={offset:+d}: {out['event_id'].nunique():,} events from cache",
              flush=True)
        return out
    started = time.time()
    events = event_universe()
    struct = twin_peak(steps=1, anchor_offset=offset)
    plan = replay_mod.filter_plan_by_availability(replay_mod.plan_events(struct, events))
    frame = plan.frame
    parts, skipped = [], {}
    for year, block in frame.groupby(frame["entry_date"].dt.year, sort=True):
        keys = set()
        for col in ("entry_date", "exit_date"):
            keys |= {(t, d) for t, d in zip(block["ticker"], block[col])}
        index = replay_mod.load_chain_index(keys, progress_every=0)
        sub = events[events["event_id"].isin(set(block["event_id"]))]
        res = replay_mod.replay(STRATEGY, sub, structure=struct, index=index,
                                progress_every=0)
        if len(res.trades):
            parts.append(res.trades)
        for k, v in res.skipped.items():
            skipped[k] = skipped.get(k, 0) + int(v)
        del index
        print(f"[shift] offset={offset:+d} {year}: {sum(len(p) for p in parts):,} rows, "
              f"{time.time()-started:.0f}s", flush=True)
    trades = (pd.concat(parts, ignore_index=True) if parts else replay_mod._empty_trades())
    trades = twinp.attach_filters(trades)
    trades.to_parquet(path, index=False)
    (RESULTS / f"skips_off{offset}.json").write_text(json.dumps(skipped, indent=1))
    print(f"[shift] offset={offset:+d}: {trades['event_id'].nunique():,} events priced",
          flush=True)
    return trades


def segments(sel: pd.DataFrame) -> list[dict]:
    """The band table EXP-123 used, so the two are directly comparable."""
    d = sel["legs"].apply(json.loads)
    spot_exit = d.apply(lambda x: float(x["spot_exit"]))
    move_w = (spot_exit - sel["spot_entry"]) / sel["w"]
    eoc = sel["exit_value"] / sel["cost"]
    bands = [(-99, -4, "deep DOWN"), (-4, -2, "down ramp"), (-2, -1, "down plateau"),
             (-1, 1, "ATM dip"), (1, 2, "up plateau"), (2, 4, "up ramp"), (4, 99, "deep UP")]
    out = []
    for lo, hi, lbl in bands:
        g = (move_w >= lo) & (move_w < hi)
        if not g.any():
            continue
        out.append({"band": lbl, "n": int(g.sum()), "share": float(g.mean()),
                    "mean_ret": float(sel.loc[g, "ret"].mean()),
                    "win": float((sel.loc[g, "ret"] > 0).mean()),
                    "exit_over_debit": float(eoc[g].median())})
    return out
