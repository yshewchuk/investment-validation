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
emitted row. Monthly mode selects candidates by effective exit month and repaired
expiry availability; it returns rows in memory and uses typed refusals.

**Ungated is emitted too, and it is the point.** A family positive without the
gate and negative with it has a gate problem, not a structure problem, and
nothing but the unselected population tells the two apart. The gate is applied
downstream in ``run.py``, so both readings come out of one build.

"""
from __future__ import annotations

import hashlib
import json
import resource
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
from engine.v2.models.training.fold_store import TrainingRefused  # noqa: E402
from engine.v2.models.training.receipts import ReceiptIssue  # noqa: E402


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


def _monthly_refusal(code: str, detail: str) -> TrainingRefused:
    return TrainingRefused((ReceiptIssue("candidate_month", code, detail),))


_MONTHLY_BENIGN_SKIPS = {"structure_unresolved", "exit_structure_unresolved",
                         "expiry_gone_at_exit", "bad_quote", "zero_cost"}


def _month_window(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    try:
        period = pd.Period(month, freq="M")
        if str(period) != month:
            raise ValueError(f"invalid_exit_month: {month}")
        start, end = period.start_time, (period + 1).start_time
    except (OverflowError, TypeError, ValueError) as exc:
        raise _monthly_refusal("INVALID_EXIT_MONTH", f"invalid_exit_month: {month}") from exc
    dates = pd.to_datetime(frame["exit_date"], errors="coerce")
    return frame.loc[dates.ge(start) & dates.lt(end)].copy()


def _effective_month_plan(frame, month: str, replay, e133):
    current = _month_window(frame, month)
    previous = _month_window(frame, str(pd.Period(month, freq="M") - 1))
    if previous.empty:
        return current
    keys = {(row["ticker"], row["entry_date"])
            for row in previous[["ticker", "entry_date"]].to_dict("records")}
    started = time.monotonic()
    print(f"[e137] availability scan start month={month} prior_events={len(previous)}",
          file=sys.stderr, flush=True)
    index = replay.load_chain_index(keys, progress_every=100)
    selected = []
    for done, row in enumerate(previous.to_dict("records"), 1):
        entry = index.get(row["ticker"], row["entry_date"])
        if entry is None or entry.empty:
            raise _monthly_refusal(
                "PRIOR_MONTH_ENTRY_CHAIN_MISSING",
                f"required_prior_month_entry_chain_missing: {row['event_id']}")
        entry = replay_mod._clean(entry)
        if entry.empty:
            continue
        try:
            expiry = pd.Timestamp(e133.ExpirySelector(kind="first_post_event").select(
                entry, row["event_date"], row["session"]))
            if pd.isna(expiry):
                raise ValueError("expiry selector returned no date")
        except Exception as exc:  # noqa: BLE001 — required to classify expiry availability.
            raise _monthly_refusal(
                "PRIOR_MONTH_EXPIRY_SELECTION_FAILED",
                f"prior_month_expiry_selection_failed: {row['event_id']}: {exc}") from exc
        if expiry.strftime("%Y-%m") == month:
            selected.append(row["event_id"])
        if done == 1 or done % 100 == 0 or done == len(previous):
            elapsed = max(time.monotonic() - started, 0.001)
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"[e137] availability scan {done}/{len(previous)}, "
                  f"{done / elapsed:.2f}/s, ETA {(len(previous)-done)*elapsed/done:.0f}s, "
                  f"selected={len(selected)}, RSS {rss_mb:.0f}MB", file=sys.stderr, flush=True)
    earlier = previous.loc[previous["event_id"].isin(selected)]
    return pd.concat([current, earlier], ignore_index=True)


def _refuse_monthly_event_skip(exc: Exception, event_id: str) -> None:
    if str(exc) not in _MONTHLY_BENIGN_SKIPS:
        raise _monthly_refusal(
            "MONTHLY_CANDIDATE_INPUT_FAILED",
            f"required_monthly_candidate_input_failed: {event_id}: {exc}") from exc


def _effective_exit_trades(trades: pd.DataFrame, month: str) -> pd.DataFrame:
    out = trades.copy()
    repaired = ~out["exit_arb_ok"].fillna(True).astype(bool) | out.exit_value.lt(0)
    effective_exit = pd.to_datetime(out["exit_date"], errors="coerce")
    expiry = pd.to_datetime(out["expiry"], errors="coerce")
    effective_exit.loc[repaired] = expiry.loc[repaired]
    selected = effective_exit.dt.strftime("%Y-%m").eq(month)
    if (effective_exit.isna().any() or expiry.loc[repaired].isna().any()
            or out.loc[repaired & selected, "exit_value_expiry"].isna().any()):
        raise _monthly_refusal(
            "MONTHLY_EFFECTIVE_EXIT_DATA_MISSING",
            "monthly_candidate_effective_exit_date_or_settlement_value_missing")
    out["exit_date"] = effective_exit
    out.loc[repaired, "exit_value"] = out.loc[repaired, "exit_value_expiry"]
    out["pnl"] = out["exit_value"] - out["entry_cost"]
    out["ret"] = (out["pnl"] / out["entry_cost"]).where(
        out["entry_cost"] > e133.MIN_MEANINGFUL_COST)
    return out.loc[selected].copy()


def build_all(*, force: bool = False, limit_years=None,
              limit_exit_month: str | None = None) -> dict:
    if limit_exit_month is not None:
        _month_window(pd.DataFrame({"exit_date": []}), limit_exit_month)
    out = RESULTS / "candidates.parquet"
    if limit_exit_month is None and out.exists() and not force:
        print("[e137] cached", flush=True)
        return {"trades": pd.read_parquet(out),
                "tallies": pd.read_parquet(RESULTS / "tallies.parquet"),
                "events": pd.read_parquet(RESULTS / "event_summary.parquet"),
                "meta": json.loads((RESULTS / "build_meta.json").read_text())}

    started = time.time()
    planning_started = time.monotonic()
    if limit_exit_month is not None:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[e137] month {limit_exit_month} plan start RSS {rss_mb:.0f}MB",
              file=sys.stderr, flush=True)
    try:
        events = e133.event_universe()
        events["event_date"] = pd.to_datetime(events["event_date"])
        structure = twin_peak_5(wing_multiple=3, width_moneyness=0.05)
        if limit_exit_month is not None:
            forecasts = e133.forecasts()
            events = events.merge(forecasts, on=["ticker", "event_date"], how="inner")
            planned = replay_mod.plan_events(structure, events)
            available_plan = replay_mod.filter_plan_by_availability(planned)
            month_plan = _effective_month_plan(
                available_plan.frame, limit_exit_month, replay_mod, e133)
            month_ids = set(month_plan["event_id"])
            month_universe = events[events["event_id"].isin(month_ids)].copy()
            month_ids = set(month_universe["event_id"])
            month_plan = month_plan.loc[month_plan["event_id"].isin(month_ids)].copy()
            plan = replay_mod.ReplayPlan(
                frame=month_plan.reset_index(drop=True), skipped=available_plan.skipped)
            events = month_universe[month_universe["event_id"].isin(set(plan.frame["event_id"]))].copy()
            elapsed = max(time.monotonic() - planning_started, 0.001)
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"[e137] month {limit_exit_month} plan end events={len(events)} "
                  f"elapsed={elapsed:.1f}s RSS {rss_mb:.0f}MB", file=sys.stderr, flush=True)
        else:
            events = events.merge(e133.forecasts(), on=["ticker", "event_date"], how="inner")
            plan = replay_mod.filter_plan_by_availability(replay_mod.plan_events(structure, events))
            events = events[events["event_id"].isin(set(plan.frame["event_id"]))].copy()
        keyframe = plan.frame.set_index("event_id")
        events["_year"] = events["event_date"].dt.year
        if limit_years is not None:
            events = events[events["_year"].isin(set(limit_years))]
        print(f"[e137] {len(events):,} events with chains and forecasts", flush=True)
        if limit_exit_month is not None:
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"[e137] month {limit_exit_month} build start events={len(events)} "
                  f"RSS {rss_mb:.0f}MB", file=sys.stderr, flush=True)
        pool = pnl_sim.ResidualPool(e133.residual_history())
        spot_at = e134.expiry_spot()
    except Exception as exc:  # noqa: BLE001 — required monthly source failed.
        if limit_exit_month is not None and not isinstance(exc, TrainingRefused):
            raise _monthly_refusal("MONTHLY_SOURCE_LOAD_FAILED", str(exc)) from exc
        raise
    n = len(GRID)
    if limit_exit_month is None:
        tally = {k: np.zeros(n, dtype=np.int64) for k in ("listed", "admissible", "tradeable")}
        for fam in FAMILIES:
            tally[f"argmax_{fam}"] = np.zeros(n, dtype=np.int64)
    skips: dict[str, int] = {}
    progress_done = 0
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
            progress_done += 1
            if limit_exit_month is not None and (
                    progress_done == 1 or progress_done % 50 == 0 or progress_done == len(events)):
                elapsed = max(time.time() - started, 0.001)
                rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                print(f"[e137] month {limit_exit_month}: {progress_done}/{len(events)} events, "
                      f"{progress_done / elapsed:.2f}/s, "
                      f"ETA {(len(events) - progress_done) * elapsed / progress_done:.0f}s, "
                      f"RSS {rss_mb:.0f}MB", file=sys.stderr, flush=True)
            er = index.get(row["ticker"], row["entry_date"])
            xr = index.get(row["ticker"], row["exit_date"])
            if er is None or xr is None or er.empty or xr.empty:
                if limit_exit_month is None:
                    skips["no_chain"] = skips.get("no_chain", 0) + 1
                continue
            rng = np.random.default_rng(int.from_bytes(
                hashlib.sha256(f"pick|{row['event_id']}".encode()).digest()[:8], "big"))
            try:
                ev = e133.price_event(row, er, xr, pool, rng_pick=rng)
            except e133.EventSkip as exc:
                if limit_exit_month is not None:
                    _refuse_monthly_event_skip(exc, row["event_id"])
                if limit_exit_month is None:
                    skips[str(exc)] = skips.get(str(exc), 0) + 1
                continue
            except Exception as exc:                               # noqa: BLE001
                if limit_exit_month is not None:
                    raise _monthly_refusal(
                        "MONTHLY_CANDIDATE_FAILED",
                        f"required_monthly_candidate_failed: {row['event_id']}: {exc}") from exc
                skips[f"error:{type(exc).__name__}"] = skips.get(
                    f"error:{type(exc).__name__}", 0) + 1
                continue

            mids = 0.5 * (ev["bid_e"] + ev["ask_e"])
            arb = e134.no_arb_ok(ev["strikes"], mids, ev["idx"], GRID.slot_used,
                                 subset=ev["admissible"])
            tradeable = ev["tradeable"] & arb
            sim = ev["sim_sel"]
            usable = tradeable & np.isfinite(sim)
            if limit_exit_month is None:
                tally["listed"] += ev["listed"]
                tally["admissible"] += ev["admissible"]
                tally["tradeable"] += tradeable
            # Which families this event can carry AT ALL — the denominator that
            # separates "never wins" from "never even offered".
            offers = {fam: bool((usable & FAMILY_MASK[fam]).any()) for fam in FAMILIES}
            common = all(offers.values())

            if (limit_exit_month is None and tradeable.any()
                    and len(equivalence) < e133.EQUIV_EVENTS):
                if rng.random() < e133.EQUIV_RATE or not equivalence:
                    ev2 = dict(ev); ev2["tradeable"] = tradeable
                    equivalence.append(e133.check_equivalence(ev2, row, er, xr, pool))

            emitted = []
            for fam in FAMILIES:
                ok = np.flatnonzero(usable & FAMILY_MASK[fam])
                if ok.size == 0:
                    continue
                pick = int(ok[np.argmax(sim[ok])])
                if limit_exit_month is None:
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
                if (limit_exit_month == pd.Timestamp(r["expiry"]).strftime("%Y-%m")
                        and (not r["exit_arb_ok"] or r["exit_value"] < 0) and not np.isfinite(s)):
                    raise _monthly_refusal(
                        "MONTHLY_SETTLEMENT_SPOT_MISSING",
                        f"required_monthly_settlement_spot_missing: {row['event_id']} {r['expiry']}")
                r["spot_expiry"] = float(s) if np.isfinite(s) else np.nan
                r["exit_value_expiry"] = (e134.terminal_payoff(r["legs"], s)
                                          if np.isfinite(s) else np.nan)
            parts += emitted
            if limit_exit_month is None:
                ev_rows.append({
                    "event_id": row["event_id"], "ticker": row["ticker"],
                    "event_date": row["event_date"], "year": int(year),
                    "n_tradeable": int(tradeable.sum()),
                    "families_offered": int(sum(offers.values())),
                    "common_universe": common,
                    **{f"offers_{fam}": offers[fam] for fam in FAMILIES},
                })
        del index
        if limit_exit_month is None:
            print(f"[e137] {year}: {len(ev_rows):,} events, {len(parts):,} rows, "
                  f"{time.time() - started:.0f}s", flush=True)

    if limit_exit_month is not None:
        elapsed = max(time.time() - started, 0.001)
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"[e137] month {limit_exit_month} build end events={progress_done} "
              f"elapsed={elapsed:.1f}s RSS {rss_mb:.0f}MB", file=sys.stderr, flush=True)
    trades = pd.DataFrame(parts)
    if limit_exit_month is not None and not trades.empty:
        trades = _effective_exit_trades(trades, limit_exit_month)
    if limit_exit_month is not None:
        return {"trades": trades}
    tallies = e133._pattern_frame()
    for k, v in tally.items():
        tallies[k] = v
    ev_frame = pd.DataFrame(ev_rows)
    meta = {"events_priced": int(len(ev_frame)), "skips": skips,
            "equivalence": e133._equiv_summary(equivalence),
            "common_universe_events": int(ev_frame["common_universe"].sum()) if len(ev_frame) else 0,
            "elapsed_s": round(time.time() - started, 1)}
    if limit_exit_month is None:
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
