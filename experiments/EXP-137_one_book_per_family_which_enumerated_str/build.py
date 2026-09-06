#!/usr/bin/env python3
"""EXP-137 — one complete book per enumerated family.

EXP-134 asked which family WINS an argmax against seven rivals. This asks a
different question that its numbers cannot answer: which families can carry a
book at all. A family that is second-best on every single event has an argmax
share near zero and may still be a perfectly good strategy; a family that wins
often may be winning only where the estimate is noisiest.

So the chooser is not removed, it is *restricted*: on every event, each family
independently picks its own best width and anchor and trades. Eight arms, eight
books, one universe.

Everything else is EXP-134's and is imported from it rather than
reimplemented — the enumeration, the width rules, the three no-arbitrage
conditions, the spread and market-cap filters, the linear expected-P&L
shortcut, the equivalence receipt, and the two exit branches carried on every
emitted row. The only new thing here is the arm masks and the ``common``
flag.

**Ungated is emitted too, and it is the point.** A family positive without the
gate and negative with it has a gate problem, not a structure problem, and
nothing but the unselected population tells the two apart. The gate is applied
downstream in ``run.py``, so both readings come out of one build.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
E133 = ROOT / "experiments" / "EXP-133_every_symmetric_put_structure_the_ladder"
E134 = ROOT / "experiments" / "EXP-134_priced_right_funded_and_held_structure_s"
for p in (ROOT, E133, E134, HERE):
    sys.path.insert(0, str(p))

from engine import pnl_sim, replay as replay_mod  # noqa: E402
from engine.structures import twin_peak_5  # noqa: E402


def _load(name: str, path: Path):
    """Import by PATH — three experiments here have a `build.py`."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e133 = _load("e137_e133build", E133 / "build.py")
e134 = _load("e137_e134build", E134 / "build.py")

import family as _f  # noqa: E402  (resolves to EXP-133's, which is the one)

RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
GRID = e133.GRID
MID = e133.MID

#: One arm per enumerated family, in the spec's order.
FAMILIES = tuple(f.key for f in _f.FAMILIES)
#: Pattern masks, precomputed once: which grid rows belong to which family.
FAMILY_MASK = {k: (GRID.family_key == k) for k in FAMILIES}


def build_all(*, force: bool = False, limit_years=None) -> dict:
    out = RESULTS / "candidates.parquet"
    if out.exists() and not force:
        print("[e137] cached", flush=True)
        return {"trades": pd.read_parquet(out),
                "tallies": pd.read_parquet(RESULTS / "tallies.parquet"),
                "events": pd.read_parquet(RESULTS / "event_summary.parquet"),
                "meta": json.loads((RESULTS / "build_meta.json").read_text())}

    started = time.time()
    events = e133.event_universe()
    events["event_date"] = pd.to_datetime(events["event_date"])
    events = events.merge(e133.forecasts(), on=["ticker", "event_date"], how="inner")
    plan = replay_mod.filter_plan_by_availability(
        replay_mod.plan_events(twin_peak_5(wing_multiple=3, width_moneyness=0.05), events))
    keyframe = plan.frame.set_index("event_id")
    events = events[events["event_id"].isin(set(plan.frame["event_id"]))].copy()
    events["_year"] = events["event_date"].dt.year
    if limit_years is not None:
        events = events[events["_year"].isin(set(limit_years))]
    print(f"[e137] {len(events):,} events with chains and forecasts", flush=True)

    pool = pnl_sim.ResidualPool(e133.residual_history())
    spot_at = e134.expiry_spot()

    n = len(GRID)
    tally = {k: np.zeros(n, dtype=np.int64) for k in ("listed", "admissible", "tradeable")}
    for fam in FAMILIES:
        tally[f"argmax_{fam}"] = np.zeros(n, dtype=np.int64)
    skips: dict[str, int] = {}
    parts, ev_rows, equivalence = [], [], []

    for year, block in events.groupby("_year", sort=True):
        rows = keyframe.reindex(block["event_id"]).dropna(subset=["entry_date"])
        keys = set()
        for col in ("entry_date", "exit_date"):
            keys |= {(t, d) for t, d in zip(rows["ticker"], rows[col])}
        index = replay_mod.load_chain_index(keys, progress_every=0)
        merged = block.merge(plan.frame[["event_id", "entry_date", "exit_date"]],
                             on="event_id", how="inner")
        for row in merged.to_dict("records"):
            er = index.get(row["ticker"], row["entry_date"])
            xr = index.get(row["ticker"], row["exit_date"])
            if er is None or xr is None or er.empty or xr.empty:
                skips["no_chain"] = skips.get("no_chain", 0) + 1
                continue
            rng = np.random.default_rng(int.from_bytes(
                hashlib.sha256(f"pick|{row['event_id']}".encode()).digest()[:8], "big"))
            try:
                ev = e133.price_event(row, er, xr, pool, rng_pick=rng)
            except e133.EventSkip as exc:
                skips[str(exc)] = skips.get(str(exc), 0) + 1
                continue
            except Exception as exc:                               # noqa: BLE001
                skips[f"error:{type(exc).__name__}"] = skips.get(
                    f"error:{type(exc).__name__}", 0) + 1
                continue

            mids = 0.5 * (ev["bid_e"] + ev["ask_e"])
            arb = e134.no_arb_ok(ev["strikes"], mids, ev["idx"], GRID.slot_used,
                                 subset=ev["admissible"])
            tradeable = ev["tradeable"] & arb
            sim = ev["sim_sel"]
            usable = tradeable & np.isfinite(sim)
            tally["listed"] += ev["listed"]
            tally["admissible"] += ev["admissible"]
            tally["tradeable"] += tradeable

            # Which families this event can carry AT ALL — the denominator that
            # separates "never wins" from "never even offered".
            offers = {fam: bool((usable & FAMILY_MASK[fam]).any()) for fam in FAMILIES}
            common = all(offers.values())

            if tradeable.any() and len(equivalence) < e133.EQUIV_EVENTS:
                if rng.random() < e133.EQUIV_RATE or not equivalence:
                    ev2 = dict(ev); ev2["tradeable"] = tradeable
                    equivalence.append(e133.check_equivalence(ev2, row, er, xr, pool))

            emitted = []
            for fam in FAMILIES:
                ok = np.flatnonzero(usable & FAMILY_MASK[fam])
                if ok.size == 0:
                    continue
                pick = int(ok[np.argmax(sim[ok])])
                tally[f"argmax_{fam}"][pick] += 1
                got = e133._emit(ev, row, fam, ev["idx"][pick], GRID.qty[pick],
                                 pick, float(sim[pick]), float(ev["sim_gate"][pick]),
                                 str(GRID.key[pick]))
                for r in got:
                    r["common_universe"] = common
                emitted += got

            for r in emitted:
                r["exit_arb_ok"] = e134.legs_no_arb_ok(r["legs"])
                s = spot_at.get((row["ticker"], pd.Timestamp(r["expiry"])), np.nan)
                r["spot_expiry"] = float(s) if np.isfinite(s) else np.nan
                r["exit_value_expiry"] = (e134.terminal_payoff(r["legs"], s)
                                          if np.isfinite(s) else np.nan)
            parts += emitted

            ev_rows.append({
                "event_id": row["event_id"], "ticker": row["ticker"],
                "event_date": row["event_date"], "year": int(year),
                "n_tradeable": int(tradeable.sum()),
                "families_offered": int(sum(offers.values())),
                "common_universe": common,
                **{f"offers_{fam}": offers[fam] for fam in FAMILIES},
            })
        del index
        print(f"[e137] {year}: {len(ev_rows):,} events, {len(parts):,} rows, "
              f"{time.time() - started:.0f}s", flush=True)

    trades = pd.DataFrame(parts)
    tallies = e133._pattern_frame()
    for k, v in tally.items():
        tallies[k] = v
    ev_frame = pd.DataFrame(ev_rows)
    meta = {"events_priced": int(len(ev_frame)), "skips": skips,
            "equivalence": e133._equiv_summary(equivalence),
            "common_universe_events": int(ev_frame["common_universe"].sum()) if len(ev_frame) else 0,
            "elapsed_s": round(time.time() - started, 1)}
    trades.to_parquet(out, index=False)
    tallies.to_parquet(RESULTS / "tallies.parquet", index=False)
    ev_frame.to_parquet(RESULTS / "event_summary.parquet", index=False)
    (RESULTS / "build_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[e137] done: {len(ev_frame):,} events, {len(trades):,} rows, "
          f"common universe {meta['common_universe_events']:,}, skips {skips}", flush=True)
    return {"trades": trades, "tallies": tallies, "events": ev_frame, "meta": meta}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--years", type=int, nargs="*", default=None)
    a = ap.parse_args()
    build_all(force=a.force, limit_years=a.years)
