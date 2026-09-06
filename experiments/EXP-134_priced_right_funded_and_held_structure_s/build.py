#!/usr/bin/env python3
"""EXP-134's candidate build: EXP-133's, with the price checked and the exit real.

Everything about the candidate set — the eight enumerated families, the width
rules, the anchors, the ladder bound, the linear expected-P&L shortcut, the
equivalence receipt — is IMPORTED from EXP-133 rather than reimplemented. Two
experiments that differ in three registered constraints must not also differ in
a thousand lines of pricing code, or the comparison measures the code.

What this module adds:

``no_arb_ok``
    A one-expiry put curve satisfies three static conditions, none of which
    needs a model: ``P(K)`` non-decreasing in strike, every vertical worth no
    more in value than in strike width, every butterfly non-negative. A vendor
    MID surface is a smoothed estimate and violates them routinely in thin
    strikes — harmless until something searches over combinations of those
    strikes, at which point each violation is free money the search will find.
    EXP-133 measured 44.6% of its traded entries sitting on a violation
    against the fixed-shape incumbent's 1.0%. Here a candidate carrying one is
    not priced.

``settle_at_expiry``
    The terminal payoff against the Tier-2 close on the expiry date. Needed
    because the registered exit holds the position when the post-print curve
    is inconsistent, and a held position has to settle somewhere real.

The margin constraint is NOT here. It depends on the whole book in date order —
what is already open, what equity is now — so it lives in ``margin.py`` and is
applied in ``run.py`` after the gate.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
# EXP-133's build is the shared implementation; its directory has to be on the
# path for its own `import family` to resolve.
E133 = ROOT / "experiments" / "EXP-133_every_symmetric_put_structure_the_ladder"
sys.path.insert(0, str(E133))

import build as e133                                              # noqa: E402
from engine import pnl_sim, replay as replay_mod                  # noqa: E402
from engine.data import store                                     # noqa: E402
from engine.structures import twin_peak_5                         # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
MID = e133.MID
GRID = e133.GRID

#: Dollar tolerance on the no-arbitrage conditions. Quotes live on a one-cent
#: grid, so this only forgives float64 noise, never a real violation.
ARB_TOL = 1e-9

ARMS = ("best_all", "best_twin_only", "best_twin_p5", "random_pick", "incumbent")


# --------------------------------------------------------------------------
# no-arbitrage on the strikes a candidate actually uses
# --------------------------------------------------------------------------


def curve_violations(strikes: np.ndarray, mids: np.ndarray) -> np.ndarray:
    """``(n_strikes-1,)`` and ``(n_strikes-2,)`` violation flags, as one mask.

    Returned per ADJACENT PAIR and TRIPLE on the full ladder, so a candidate's
    verdict is a lookup over the pairs and triples its own strikes span rather
    than a recomputation per candidate. That is what keeps a check over
    12,600 candidates cheaper than the pricing it guards.
    """
    dK = np.diff(strikes)
    dP = np.diff(mids)
    pair_bad = (dP < -ARB_TOL) | (dP / np.where(dK > 0, dK, 1.0) > 1.0 + ARB_TOL)
    lam = (strikes[2:] - strikes[1:-1]) / (strikes[2:] - strikes[:-2])
    triple_bad = mids[1:-1] > lam * mids[:-2] + (1 - lam) * mids[2:] + ARB_TOL
    return pair_bad, triple_bad


def no_arb_ok(strikes: np.ndarray, mids: np.ndarray, idx: np.ndarray,
              used: np.ndarray, subset: np.ndarray | None = None) -> np.ndarray:
    """True where a candidate's own strikes carry no violation.

    A candidate is judged on the sub-curve it trades, not on the whole ladder:
    a violation four strikes away is somebody else's problem and refusing the
    event for it would throw away most of the universe. The sub-curve is the
    candidate's strikes in ascending order, and every adjacent pair and triple
    OF THAT SUB-CURVE must satisfy all three conditions.
    """
    n = idx.shape[0]
    out = np.ones(n, dtype=bool)
    # Only candidates that got as far as being admissible are judged. An
    # unresolved candidate's leg indices are CLIPPED into range rather than
    # left out of bounds (see `_leg_indices`), so several of its slots can
    # point at the same strike — which is a zero denominator in the convexity
    # ratio and a warning about a row whose verdict is discarded anyway.
    live = np.ones(n, dtype=bool) if subset is None else np.asarray(subset, dtype=bool)
    order = np.argsort(np.where(used, strikes[idx], np.inf), axis=1)
    ks = np.take_along_axis(strikes[idx], order, axis=1)
    ms = np.take_along_axis(mids[idx], order, axis=1)
    cnt = used.sum(axis=1)
    for width in np.unique(cnt):
        rows = np.flatnonzero((cnt == width) & live)
        if width < 2 or rows.size == 0:
            continue
        K = ks[rows, :width]
        P = ms[rows, :width]
        dK, dP = np.diff(K, axis=1), np.diff(P, axis=1)
        bad = ((dP < -ARB_TOL) | (dP / np.where(dK > 0, dK, 1.0) > 1.0 + ARB_TOL)).any(axis=1)
        if width >= 3:
            lam = (K[:, 2:] - K[:, 1:-1]) / (K[:, 2:] - K[:, :-2])
            bad |= (P[:, 1:-1] > lam * P[:, :-2] + (1 - lam) * P[:, 2:] + ARB_TOL).any(axis=1)
        out[rows] = ~bad
    return out


def legs_no_arb_ok(blob) -> bool:
    """The same three conditions on a stored trade's own legs, either side.

    Used at the exit, where the decision is not whether to price the candidate
    but whether to close it. Same arithmetic, same tolerance.
    """
    doc = json.loads(blob) if isinstance(blob, str) else blob
    curve = sorted({l["strike"]: 0.5 * (l["bid"] + l["ask"]) for l in doc["exit"]}.items())
    K = np.array([k for k, _ in curve], dtype=float)
    P = np.array([p for _, p in curve], dtype=float)
    if K.size < 2:
        return True
    dK, dP = np.diff(K), np.diff(P)
    if ((dP < -ARB_TOL) | (dP / dK > 1.0 + ARB_TOL)).any():
        return False
    if K.size >= 3:
        lam = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
        if (P[1:-1] > lam * P[:-2] + (1 - lam) * P[2:] + ARB_TOL).any():
            return False
    return True


def terminal_payoff(blob, spot_at_expiry: float) -> float:
    """What the structure is worth if it is simply allowed to expire.

    Non-negative by construction for every enumerated family — the contracts
    sum to zero over exactly mirrored strikes — which is the whole reason the
    hold branch makes defined risk exact rather than approximate.
    """
    doc = json.loads(blob) if isinstance(blob, str) else blob
    K = np.array([l["strike"] for l in doc["entry"]], dtype=float)
    q = np.array([l["qty"] * (1.0 if l["side"] == "buy" else -1.0)
                  for l in doc["entry"]], dtype=float)
    return float((q * np.maximum(K - float(spot_at_expiry), 0.0)).sum())


def expiry_spot() -> pd.Series:
    """Tier-2 close per ``(ticker, date)``, for settling a held position."""
    dm = store.read_table("daily_market", years=range(2017, 2027),
                          columns=["ticker", "date", "spot"])
    dm["date"] = pd.to_datetime(dm["date"])
    return dm.set_index(["ticker", "date"])["spot"]


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------


def build_all(*, force: bool = False, limit_years=None) -> dict:
    """EXP-133's loop with the no-arbitrage filter inserted before pricing."""
    out_trades = RESULTS / "candidates.parquet"
    if out_trades.exists() and not force:
        print("[e134] cached", flush=True)
        return {"trades": pd.read_parquet(out_trades),
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
    print(f"[e134] {len(events):,} events with chains and forecasts", flush=True)

    pool = pnl_sim.ResidualPool(e133.residual_history())
    spot_at = expiry_spot()

    n = len(GRID)
    tally = {k: np.zeros(n, dtype=np.int64)
             for k in ("listed", "admissible", "arb_ok", "tradeable")}
    for arm in ARMS:
        tally[f"argmax_{arm}"] = np.zeros(n, dtype=np.int64)
    skips: dict[str, int] = {}
    parts, ev_rows, equivalence = [], [], []

    import hashlib

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
            rng_pick = np.random.default_rng(int.from_bytes(
                hashlib.sha256(f"pick|{row['event_id']}".encode()).digest()[:8], "big"))
            try:
                ev = e133.price_event(row, er, xr, pool, rng_pick=rng_pick)
            except e133.EventSkip as exc:
                skips[str(exc)] = skips.get(str(exc), 0) + 1
                continue
            except Exception as exc:                                # noqa: BLE001
                skips[f"error:{type(exc).__name__}"] = skips.get(
                    f"error:{type(exc).__name__}", 0) + 1
                continue

            # THE NEW FILTER. Applied to `admissible` rather than to the final
            # mask so the funnel can report what it alone costs.
            mids = 0.5 * (ev["bid_e"] + ev["ask_e"])
            arb = no_arb_ok(ev["strikes"], mids, ev["idx"], GRID.slot_used,
                            subset=ev["admissible"])
            tally["listed"] += ev["listed"]
            tally["admissible"] += ev["admissible"]
            tally["arb_ok"] += ev["admissible"] & arb
            tradeable = ev["tradeable"] & arb
            tally["tradeable"] += tradeable

            if tradeable.any() and len(equivalence) < e133.EQUIV_EVENTS:
                if rng_pick.random() < e133.EQUIV_RATE or not equivalence:
                    ev2 = dict(ev); ev2["tradeable"] = tradeable
                    equivalence.append(
                        e133.check_equivalence(ev2, row, er, xr, pool))

            sim = ev["sim_sel"]
            picks = {}
            for arm in ARMS:
                if arm == "incumbent":
                    continue
                mask = tradeable & np.isfinite(sim) & e133.ARM_PATTERN_MASK[arm]
                ok = np.flatnonzero(mask)
                if ok.size == 0:
                    picks[arm] = None
                    continue
                picks[arm] = int(ok[rng_pick.integers(0, ok.size)]) if arm == "random_pick" \
                    else int(ok[np.argmax(sim[ok])])

            emitted = []
            for arm, pick in picks.items():
                if pick is None:
                    continue
                tally[f"argmax_{arm}"][pick] += 1
                emitted += e133._emit(ev, row, arm, ev["idx"][pick], GRID.qty[pick],
                                      int(pick), float(sim[pick]),
                                      float(ev["sim_gate"][pick]), str(GRID.key[pick]))
            inc = ev["incumbent"]
            if inc is not None:
                ii, iq = inc
                used = iq != 0
                arb_inc = bool(no_arb_ok(ev["strikes"], mids, ii[None, :],
                                         used[None, :])[0])
                sc = e133._score_one(ev, ii, iq)
                if (arb_inc and sc["cost"][MID] > e133.MIN_MEANINGFUL_COST
                        and sc["rel_spread"] <= e133.MAX_REL_SPREAD
                        and all(sc["cost"][a] > e133.MIN_MEANINGFUL_COST
                                for a in e133.ALPHA_GRID)):
                    s1, s2 = e133._incumbent_sim(ev, ii, iq, sc["cost"][MID])
                    emitted += e133._emit(ev, row, "incumbent", ii, iq, -1, s1, s2,
                                          "TWIN-P5-w3-pred100")

            # Attach both exit branches to every emitted row while the chain is
            # still in hand: whether the post-print curve was consistent, and
            # what the structure settles at if held instead.
            for r in emitted:
                r["exit_arb_ok"] = legs_no_arb_ok(r["legs"])
                s = spot_at.get((row["ticker"], pd.Timestamp(r["expiry"])), np.nan)
                r["spot_expiry"] = float(s) if np.isfinite(s) else np.nan
                r["exit_value_expiry"] = (terminal_payoff(r["legs"], s)
                                          if np.isfinite(s) else np.nan)
            parts += emitted

            ev_rows.append({
                "event_id": row["event_id"], "ticker": row["ticker"],
                "event_date": row["event_date"], "year": int(year),
                "n_listed": int(ev["listed"].sum()),
                "n_admissible": int(ev["admissible"].sum()),
                "n_arb_ok": int((ev["admissible"] & arb).sum()),
                "n_arb_rejected": int((ev["admissible"] & ~arb).sum()),
                "n_tradeable": int(tradeable.sum()),
                "spot_entry": ev["spot_entry"],
            })
        del index
        print(f"[e134] {year}: {len(ev_rows):,} events, {len(parts):,} rows, "
              f"{time.time() - started:.0f}s", flush=True)

    trades = pd.DataFrame(parts)
    tallies = e133._pattern_frame()
    for k, v in tally.items():
        tallies[k] = v
    ev_frame = pd.DataFrame(ev_rows)
    meta = {"events_priced": int(len(ev_frame)), "skips": skips,
            "equivalence": e133._equiv_summary(equivalence),
            "elapsed_s": round(time.time() - started, 1)}
    trades.to_parquet(out_trades, index=False)
    tallies.to_parquet(RESULTS / "tallies.parquet", index=False)
    ev_frame.to_parquet(RESULTS / "event_summary.parquet", index=False)
    (RESULTS / "build_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[e134] done: {len(ev_frame):,} events, {len(trades):,} rows, skips {skips}",
          flush=True)
    return {"trades": trades, "tallies": tallies, "events": ev_frame, "meta": meta}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--years", type=int, nargs="*", default=None)
    a = ap.parse_args()
    build_all(force=a.force, limit_years=a.years)
