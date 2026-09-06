#!/usr/bin/env python3
"""EXP-139 — the same candidates, simulated from a size-matched residual pool.

EXP-137's build, with one substitution: the residual pool conditions on the
event's market-cap bucket as well as its predicted-move decile. Everything
else — the eight families, the widths, the anchors, the no-arbitrage filter,
the spread and cap floors, the pricing, the seeds — is imported from EXP-133
and EXP-137 unchanged, so any difference between this run's expectations and
EXP-137's is attributable to the pool and to nothing else.

The ``uncorrected`` arm is not recomputed here. It already exists on disk as
EXP-137's candidates, drawn under identical seeds, and re-simulating it would
risk a difference that came from the rerun rather than the treatment.

**Throttled deliberately.** This runs alongside EXP-135's ORATS pull, which is
network-bound and rate-limited against a finite API quota. Starving it wastes
the one resource here that does not come back, so the launcher pins BLAS to a
single thread and runs at the lowest scheduling priority.
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
E137 = ROOT / "experiments" / "EXP-137_one_book_per_family_which_enumerated_str"
for p in (ROOT, E133, E134, E137, HERE):
    sys.path.insert(0, str(p))

from engine import replay as replay_mod  # noqa: E402
from engine.data import store  # noqa: E402
from engine.structures import twin_peak_5  # noqa: E402


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


e133 = _load("e139_e133build", E133 / "build.py")
e134 = _load("e139_e134build", E134 / "build.py")
import cap_pool as cp  # noqa: E402
import family as _f  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
GRID = e133.GRID
MID = e133.MID
FAMILIES = tuple(f.key for f in _f.FAMILIES)
FAMILY_MASK = {k: (GRID.family_key == k) for k in FAMILIES}


def build_all(*, force: bool = False, limit_years=None) -> dict:
    meta_path = RESULTS / "build_meta.json"
    if meta_path.exists() and not force:
        print("[e139] cached", flush=True)
        return {"tallies": pd.read_parquet(RESULTS / "tallies.parquet"),
                "events": pd.read_parquet(RESULTS / "event_summary.parquet"),
                "meta": json.loads(meta_path.read_text())}

    started = time.time()
    events = e133.event_universe()
    events["event_date"] = pd.to_datetime(events["event_date"])
    events = events.merge(e133.forecasts(), on=["ticker", "event_date"], how="inner")
    # The event's OWN market cap, which is what the pool conditions on.
    events["year"] = events["event_date"].dt.year
    sec = store.read_table("securities", years=range(2013, 2027),
                           columns=["ticker", "year", "mcap_usd"])
    events = events.merge(sec, on=["ticker", "year"], how="left")

    plan = replay_mod.filter_plan_by_availability(
        replay_mod.plan_events(twin_peak_5(wing_multiple=3, width_moneyness=0.05), events))
    keyframe = plan.frame.set_index("event_id")
    events = events[events["event_id"].isin(set(plan.frame["event_id"]))].copy()
    events["_year"] = events["event_date"].dt.year
    if limit_years is not None:
        events = events[events["_year"].isin(set(limit_years))]
    print(f"[e139] {len(events):,} events; cap known for "
          f"{int(events['mcap_usd'].notna().sum()):,}", flush=True)

    pool = cp.CapResidualPool(cp.history_with_caps(e133.residual_history(), store))
    spot_at = e134.expiry_spot()

    shards = RESULTS / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    for old in shards.glob("*.parquet"):
        old.unlink()
    skips: dict[str, int] = {}
    parts, ev_rows, n_rows = [], [], 0

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
            except Exception as exc:                              # noqa: BLE001
                skips[f"error:{type(exc).__name__}"] = skips.get(
                    f"error:{type(exc).__name__}", 0) + 1
                continue

            mids = 0.5 * (ev["bid_e"] + ev["ask_e"])
            arb = e134.no_arb_ok(ev["strikes"], mids, ev["idx"], GRID.slot_used,
                                 subset=ev["admissible"])
            tradeable = ev["tradeable"] & arb
            sim = ev["sim_sel"]
            usable = tradeable & np.isfinite(sim)
            if not usable.any():
                continue
            offers = {fam: bool((usable & FAMILY_MASK[fam]).any()) for fam in FAMILIES}
            common = all(offers.values())
            emitted = []
            for fam in FAMILIES:
                ok = np.flatnonzero(usable & FAMILY_MASK[fam])
                if ok.size == 0:
                    continue
                pick = int(ok[np.argmax(sim[ok])])
                got = e133._emit(ev, row, fam, ev["idx"][pick], GRID.qty[pick], pick,
                                 float(sim[pick]), float(ev["sim_gate"][pick]),
                                 str(GRID.key[pick]))
                for r in got:
                    r["common_universe"] = common
                    r["cap_bucket"] = int(cp.bucket_of(row.get("mcap_usd", np.nan)))
                    r["mcap_usd"] = row.get("mcap_usd", np.nan)
                emitted += got
            for r in emitted:
                r["exit_arb_ok"] = e134.legs_no_arb_ok(r["legs"])
                s = spot_at.get((row["ticker"], pd.Timestamp(r["expiry"])), np.nan)
                r["spot_expiry"] = float(s) if np.isfinite(s) else np.nan
                r["exit_value_expiry"] = (e134.terminal_payoff(r["legs"], s)
                                          if np.isfinite(s) else np.nan)
            parts += emitted
            ev_rows.append({"event_id": row["event_id"], "ticker": row["ticker"],
                            "event_date": row["event_date"], "year": int(year),
                            "common_universe": common,
                            "cap_bucket": int(cp.bucket_of(row.get("mcap_usd", np.nan)))})
        del index
        if parts:
            pd.DataFrame(parts).to_parquet(shards / f"year={year}.parquet", index=False)
            n_rows += len(parts)
            parts = []
        print(f"[e139] {year}: {len(ev_rows):,} events, {n_rows:,} rows, "
              f"fallbacks {pool.fallbacks}, {time.time() - started:.0f}s", flush=True)

    tallies = e133._pattern_frame()
    ev_frame = pd.DataFrame(ev_rows)
    total = sum(pool.fallbacks.values()) or 1
    meta = {
        "events_priced": int(len(ev_frame)), "rows": int(n_rows), "skips": skips,
        "common_universe_events": int(ev_frame["common_universe"].sum()) if len(ev_frame) else 0,
        # The registered acceptance criterion: a treatment applied to a minority
        # of events is not a treatment.
        "pool_conditioning": {k: v for k, v in pool.fallbacks.items()},
        "cap_conditioned_share": pool.fallbacks["cap"] / total,
        "median_pool_size": float(np.median(pool.pool_sizes)) if pool.pool_sizes else None,
        "elapsed_s": round(time.time() - started, 1),
    }
    tallies.to_parquet(RESULTS / "tallies.parquet", index=False)
    ev_frame.to_parquet(RESULTS / "event_summary.parquet", index=False)
    meta_path.write_text(json.dumps(meta, indent=1, default=str))
    print(f"[e139] done: {len(ev_frame):,} events, {n_rows:,} rows, "
          f"cap-conditioned {100*meta['cap_conditioned_share']:.1f}% of draws, "
          f"skips {skips}", flush=True)
    return {"tallies": tallies, "events": ev_frame, "meta": meta}


def load_all() -> pd.DataFrame:
    """Every row, read shard by shard."""
    qs = sorted((RESULTS / "shards").glob("*.parquet"))
    if not qs:
        raise FileNotFoundError("no shards; run the build")
    return pd.concat([pd.read_parquet(q) for q in qs], ignore_index=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--years", type=int, nargs="*", default=None)
    a = ap.parse_args()
    build_all(force=a.force, limit_years=a.years)
