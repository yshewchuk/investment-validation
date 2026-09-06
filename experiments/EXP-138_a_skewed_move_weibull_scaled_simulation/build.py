#!/usr/bin/env python3
"""EXP-138 — the same universe, four ways of drawing the move.

One pass over the chains. Pricing, the no-arbitrage filter, the width rules and
the leg geometry are computed ONCE per event and shared by every arm — only the
simulated expected P&L is recomputed per move model, because only the move's
marginal differs. That matters for more than speed: it means the four arms
differ in exactly one thing, and any difference between their books is
attributable to the move distribution and nothing else.

Per event and per move model, each of the eight families picks its own best
width and anchor, exactly as EXP-137 does. So the output is 4 x 8 books over
one universe, plus the ``common_universe`` flag that makes them comparable.
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
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


e133 = _load("e138_e133build", E133 / "build.py")
e134 = _load("e138_e134build", E134 / "build.py")
import move_models as mm  # noqa: E402
import family as _f  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
GRID = e133.GRID
MID = e133.MID
FAMILIES = tuple(f.key for f in _f.FAMILIES)
FAMILY_MASK = {k: (GRID.family_key == k) for k in FAMILIES}
#: Registered order; `weibull` is the primary and is listed first.
MODELS = ("weibull", "additive", "ratio", "gamma")


def build_all(*, force: bool = False, limit_years=None) -> dict:
    out = RESULTS / "build_meta.json"
    if out.exists() and not force:
        print("[e138] cached", flush=True)
        return {"tallies": pd.read_parquet(RESULTS / "tallies.parquet"),
                "events": pd.read_parquet(RESULTS / "event_summary.parquet"),
                "meta": json.loads(out.read_text())}

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
    print(f"[e138] {len(events):,} events; {len(MODELS)} move models x "
          f"{len(FAMILIES)} families", flush=True)

    pool = pnl_sim.ResidualPool(e133.residual_history())
    models = mm.MOVE_MODELS(pool)
    spot_at = e134.expiry_spot()

    # Written per YEAR rather than accumulated. Four move models over this
    # universe is ~1.45M rows, each carrying a ~1-2KB `legs` blob — about 3GB
    # of strings, doubled while `pd.DataFrame` builds from a list of dicts. The
    # first attempt at this build reached the last year and was killed by the
    # OOM killer with nothing written. Sharding costs one directory and makes
    # the peak memory one year's worth.
    shards = RESULTS / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    for old in shards.glob("*.parquet"):
        old.unlink()
    skips: dict[str, int] = {}
    parts, ev_rows, equivalence = [], [], []
    n_rows = 0

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
                # Priced once. Everything except the simulation is shared.
                ev = e133.price_event(row, er, xr, pool, rng_pick=rng)
            except e133.EventSkip as exc:
                skips[str(exc)] = skips.get(str(exc), 0) + 1
                continue
            except Exception as exc:                                # noqa: BLE001
                skips[f"error:{type(exc).__name__}"] = skips.get(
                    f"error:{type(exc).__name__}", 0) + 1
                continue

            mids = 0.5 * (ev["bid_e"] + ev["ask_e"])
            arb = e134.no_arb_ok(ev["strikes"], mids, ev["idx"], GRID.slot_used,
                                 subset=ev["admissible"])
            tradeable = ev["tradeable"] & arb
            if not tradeable.any():
                continue

            if len(equivalence) < e133.EQUIV_EVENTS and (
                    rng.random() < e133.EQUIV_RATE or not equivalence):
                ev2 = dict(ev); ev2["tradeable"] = tradeable
                equivalence.append(e133.check_equivalence(ev2, row, er, xr, pool))

            cost_mid = ev["priced"][MID][0]
            offers_by_model = {}
            emitted = []
            for model in MODELS:
                # ONLY this changes between arms.
                m_sel, m_gate, _ = e133._sim_means(
                    ev["strikes"], ev["spot_entry"], ev["dte_exit"],
                    row["pred_abs_move"], row["pred_iv_crush_30"], row["pre_iv30"],
                    row["event_date"], pool, key=f"EXP-133|{row['ticker']}",
                    move_model=models[model])
                if m_sel is None:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    sim = ((GRID.qty * m_sel[ev["idx"]]).sum(axis=1) - cost_mid) / cost_mid
                    sim_g = ((GRID.qty * m_gate[ev["idx"]]).sum(axis=1) - cost_mid) / cost_mid
                usable = tradeable & np.isfinite(sim)
                offers_by_model[model] = {
                    fam: bool((usable & FAMILY_MASK[fam]).any()) for fam in FAMILIES}
                for fam in FAMILIES:
                    ok = np.flatnonzero(usable & FAMILY_MASK[fam])
                    if ok.size == 0:
                        continue
                    pick = int(ok[np.argmax(sim[ok])])
                    got = e133._emit(ev, row, fam, ev["idx"][pick], GRID.qty[pick],
                                     pick, float(sim[pick]), float(sim_g[pick]),
                                     str(GRID.key[pick]))
                    for r in got:
                        r["move_model"] = model
                    emitted += got

            common = all(all(o.values()) for o in offers_by_model.values()) and \
                len(offers_by_model) == len(MODELS)
            for r in emitted:
                r["common_universe"] = common
                r["exit_arb_ok"] = e134.legs_no_arb_ok(r["legs"])
                s = spot_at.get((row["ticker"], pd.Timestamp(r["expiry"])), np.nan)
                r["spot_expiry"] = float(s) if np.isfinite(s) else np.nan
                r["exit_value_expiry"] = (e134.terminal_payoff(r["legs"], s)
                                          if np.isfinite(s) else np.nan)
            parts += emitted
            ev_rows.append({"event_id": row["event_id"], "ticker": row["ticker"],
                            "event_date": row["event_date"], "year": int(year),
                            "common_universe": common,
                            "n_tradeable": int(tradeable.sum())})
        del index
        if parts:
            pd.DataFrame(parts).to_parquet(shards / f"year={year}.parquet", index=False)
            n_rows += len(parts)
            parts = []
        print(f"[e138] {year}: {len(ev_rows):,} events, {n_rows:,} rows written, "
              f"{time.time() - started:.0f}s", flush=True)

    trades = None
    tallies = e133._pattern_frame()
    ev_frame = pd.DataFrame(ev_rows)
    meta = {"events_priced": int(len(ev_frame)), "skips": skips,
            "models": list(MODELS),
            "common_universe_events": int(ev_frame["common_universe"].sum()) if len(ev_frame) else 0,
            "equivalence": e133._equiv_summary(equivalence),
            "elapsed_s": round(time.time() - started, 1)}
    meta["rows"] = int(n_rows)
    meta["shards"] = sorted(q.name for q in shards.glob("*.parquet"))
    tallies.to_parquet(RESULTS / "tallies.parquet", index=False)
    ev_frame.to_parquet(RESULTS / "event_summary.parquet", index=False)
    (RESULTS / "build_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[e138] done: {len(ev_frame):,} events, {n_rows:,} rows across "
          f"{len(meta['shards'])} shards, common "
          f"{meta['common_universe_events']:,}, skips {skips}", flush=True)
    return {"tallies": tallies, "events": ev_frame, "meta": meta}


def load_model(model: str) -> pd.DataFrame:
    """Every row for ONE move model, read shard by shard.

    Never holds more than one model's rows — about a quarter of the run — which
    is what keeps the analysis inside memory that the build already proved is
    not generous.
    """
    shards = sorted((RESULTS / "shards").glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no shards under {RESULTS / 'shards'}; run the build")
    frames = []
    for q in shards:
        f = pd.read_parquet(q)
        frames.append(f[f["move_model"] == model])
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--years", type=int, nargs="*", default=None)
    a = ap.parse_args()
    build_all(force=a.force, limit_years=a.years)
