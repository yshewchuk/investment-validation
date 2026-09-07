"""Compact cached entries, pin contracts, plan exact exit gaps, and price EXP-135."""
from __future__ import annotations
import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_job import HERE, RESULTS, OFFSETS, log, write_json, get_events, request_history, batches
from engine import paths, replay
from engine.data import store, validate
from engine.data.normalize.n_chains import rows_to_frame
from engine.fills import FillModel, MIN_MEANINGFUL_COST
from engine.structures import straddle_runup, ExpirySelector

KEY = ["ticker", "obs_date", "expiry", "strike", "right"]
RULES = ("first_post_event", "second_post_event", "exit_plus_14_calendar_days")

def read_raw(record):
    key = record["key"]
    path = paths.RAW_FETCH / "orats" / key[:2] / (key + ".body.gz")
    with gzip.open(path, "rt") as fh:
        doc = json.load(fh)
    rows = doc["data"]
    wanted = record["params"]["ticker"].split(",")
    if len(wanted) == 1:
        for row in rows:
            row.setdefault("ticker", wanted[0])
    frame, _ = rows_to_frame(rows, source_id="fetch:orats/hist/strikes/" + key[:12], chain_kind="runup_grid")
    if frame.empty:
        return frame
    frame, _ = validate.validate_chains(frame)
    return replay._clean(frame)

def compact(frame):
    if frame.empty:
        return frame
    group = ["ticker", "obs_date", "expiry"]
    calls = frame[frame.right == "C"].copy()
    calls["gap"] = (calls.strike - calls.spot).abs()
    calls = calls.dropna(subset=["gap"])
    if calls.empty:
        return frame.iloc[:0]
    chosen = calls.loc[calls.groupby(group, sort=False).gap.idxmin(), group + ["strike"]]
    return frame.merge(chosen, on=group + ["strike"], how="inner")

def entry_compacts():
    cache = RESULTS / "compact"
    cache.mkdir(exist_ok=True)
    records = [r for r in request_history() if r["status"] == "ok" and r.get("key")]
    seen = set()
    frames = []
    for i, record in enumerate(records, 1):
        key = record["key"]
        if key in seen:
            continue
        seen.add(key)
        out = cache / (key + ".parquet")
        if not out.exists():
            c = compact(read_raw(record))
            c.to_parquet(out, index=False)
        else:
            c = pd.read_parquet(out)
        if not c.empty:
            frames.append(c)
        if i % 100 == 0:
            log(f"Compact entries {i}/{len(records)} raw responses")
    if not frames:
        raise RuntimeError("No entry data cached yet")
    result = pd.concat(frames, ignore_index=True).drop_duplicates(KEY, keep="last")
    log(f"Compact entries: {len(result):,} rows")
    return result

def candidate_entries(events, entries):
    grouped = {k: g for k,g in entries.groupby(["ticker", "obs_date"], sort=False)}
    rows, omissions = [], []
    for i, event in enumerate(events.itertuples(), 1):
        earliest = grouped.get((event.ticker, event.d21))
        if earliest is None:
            omissions.append(dict(event_id=event.event_id, reason="no_t21_chain"))
            continue
        # Expiries must include the event; selecting from the earliest entry
        # is causal for every later entry too.
        expiries = sorted(pd.Timestamp(x) for x in earliest.expiry.unique() if pd.Timestamp(x) > event.exit_date)
        longer = [e for e in expiries if e >= event.exit_date + pd.Timedelta(days=14)]
        menu = [expiries[0] if expiries else None,
                expiries[1] if len(expiries)>1 else None,
                longer[0] if longer else None]
        for rule, expiry in zip(RULES, menu):
            for offset in OFFSETS:
                obs = getattr(event, f"d{offset}")
                entry = grouped.get((event.ticker, obs))
                if entry is None or expiry is None:
                    omissions.append(dict(event_id=event.event_id, rule=rule, offset=offset, reason="entry_or_expiry_unavailable"))
                    continue
                at = entry[entry.expiry == expiry]
                c, p = at[at.right == "C"], at[at.right == "P"]
                if c.empty or p.empty:
                    omissions.append(dict(event_id=event.event_id, rule=rule, offset=offset, reason="entry_atm_leg_missing"))
                    continue
                c, p = c.iloc[0], p.iloc[0]
                assert c.strike == p.strike
                rows.append(dict(event_id=event.event_id, ticker=event.ticker,
                     event_date=event.event_date, session=event.session, entry_date=obs,
                     exit_date=event.exit_date, offset=offset, rule=rule, expiry=expiry,
                     strike=float(c.strike), spot_entry=float(c.spot),
                     call_bid_entry=float(c.bid), call_ask_entry=float(c.ask),
                     put_bid_entry=float(p.bid), put_ask_entry=float(p.ask),
                     call_iv_entry=float(c.iv), put_iv_entry=float(p.iv),
                     entry_repaired=bool(c.quote_repaired or p.quote_repaired),
                     entry_src=str(c.src_file)))
        if i % 500 == 0:
            log(f"Resolve candidates {i}/{len(events)} events; {len(rows):,} cells")
    return pd.DataFrame(rows), omissions

def exit_quotes(candidates):
    wanted = candidates[["ticker","exit_date","expiry","strike"]].drop_duplicates().rename(columns={"exit_date":"obs_date"})
    need = pd.MultiIndex.from_frame(wanted)
    keys4 = ["ticker", "obs_date", "expiry", "strike"]
    frames = []
    columns = KEY + ["bid","ask","spot","iv","quote_repaired","src_file"]
    for year, c in store.iter_table("option_chains", columns=columns):
        c = c[pd.MultiIndex.from_frame(c[keys4]).isin(need)]
        if len(c):
            frames.append(c)
        log(f"Exit cache scan {year}: {len(c):,} relevant rows")
    # Exit repair responses use the same raw store. Only parse records on
    # needed ticker/dates, and retain exact pinned strikes.
    pairset = set(zip(wanted.ticker, wanted.obs_date.dt.strftime("%Y-%m-%d")))
    for i, record in enumerate(request_history(),1):
        if record["status"] != "ok" or not record.get("key"):
            continue
        params = record["params"]
        date = params["tradeDate"]
        if not any((t,date) in pairset for t in params["ticker"].split(",")):
            continue
        c = read_raw(record)
        if not c.empty:
            c = c[pd.MultiIndex.from_frame(c[keys4]).isin(need)]
            if len(c):
                frames.append(c[columns])
        if i % 100 == 0:
            log(f"Exit raw cache scan {i} records")
    if not frames:
        return pd.DataFrame(columns=columns)
    return replay._clean(pd.concat(frames,ignore_index=True).drop_duplicates(KEY,keep="last"))

def join_exits(candidates, quotes):
    keys = ["ticker", "exit_date", "expiry", "strike"]
    t = candidates.copy()
    for right, prefix in (("C","call"),("P","put")):
        q = quotes[quotes.right == right].rename(columns={
             "obs_date":"exit_date", "bid":prefix+"_bid_exit", "ask":prefix+"_ask_exit",
             "iv":prefix+"_iv_exit", "spot":prefix+"_spot_exit",
             "quote_repaired":prefix+"_repaired_exit", "src_file":prefix+"_src_exit"})
        cols = keys + [prefix+s for s in ("_bid_exit","_ask_exit","_iv_exit","_spot_exit","_repaired_exit","_src_exit")]
        t = t.merge(q[cols],on=keys,how="left",validate="many_to_one")
    return t

def price_grid(t):
    needed = ["call_bid_exit","call_ask_exit","put_bid_exit","put_ask_exit"]
    out = t.dropna(subset=needed).copy()
    if out.empty:
        return out
    cb, ca = out.call_bid_entry.to_numpy(), out.call_ask_entry.to_numpy()
    pb, pa = out.put_bid_entry.to_numpy(), out.put_ask_entry.to_numpy()
    xb, xa = out.call_bid_exit.to_numpy(), out.call_ask_exit.to_numpy()
    yb, ya = out.put_bid_exit.to_numpy(), out.put_ask_exit.to_numpy()
    valid = np.ones(len(out),dtype=bool)
    slices = []
    for alpha in replay.ALPHA_GRID:
        fill = FillModel(alpha)
        cost = fill.price("BUY", cb, ca) + fill.price("BUY", pb, pa)
        value = fill.price("SELL", xb, xa) + fill.price("SELL", yb, ya)
        valid &= cost > MIN_MEANINGFUL_COST
        s = out.copy()
        s["fill_alpha"] = alpha
        s["entry_cost"] = cost
        s["exit_value"] = value
        s["pnl"] = value-cost
        s["ret"] = (value-cost)/np.maximum(cost, MIN_MEANINGFUL_COST)
        s["entry_cost_net"] = cost+.013
        s["exit_value_net"] = value-.013
        s["ret_net"] = (value-cost-.026)/(cost+.013)
        s["dte_entry"] = (s.expiry-s.entry_date).dt.days
        s["dte_exit"] = (s.expiry-s.exit_date).dt.days
        s["entry_spread_pct"] = ((ca-cb)+(pa-pb))/((ca+cb+pa+pb)/2)
        s["quote_repaired"] = s.entry_repaired | s.call_repaired_exit | s.put_repaired_exit
        slices.append(s)
    result = pd.concat([s[valid] for s in slices],ignore_index=True)
    log(f"Priced {len(out)} candidates; excluded {(~valid).sum()} with zero debit at any alpha")
    return result

def verify_engine(trades, n=120):
    if trades.empty:
        return
    mid = trades[np.isclose(trades.fill_alpha,.5)]
    # Cover years, offsets and expiry rules deterministically.
    sample = mid.groupby([mid.event_date.dt.year,"offset","rule"], group_keys=False).head(1)
    sample = sample.head(max(n,len(sample)))
    checks = 0
    max_error = 0.
    for row in sample.itertuples():
        frames = {}
        for suffix, date in (("entry",row.entry_date),("exit",row.exit_date)):
            rr = []
            for right,prefix in (("C","call"),("P","put")):
                spot = row.spot_entry if suffix == "entry" else getattr(row,prefix+"_spot_exit")
                rr.append(dict(ticker=row.ticker,obs_date=date,expiry=row.expiry,
                     dte=(row.expiry-date).days,strike=row.strike,right=right,
                     bid=getattr(row,prefix+"_bid_"+suffix),ask=getattr(row,prefix+"_ask_"+suffix),
                     spot=spot,quote_repaired=False))
            frames[(row.ticker,date)] = pd.DataFrame(rr)
        structure = straddle_runup(entry_offset=-row.offset)
        selector = ExpirySelector(kind="fixed",expiry=row.expiry)
        structure = replace(structure,legs=tuple(replace(leg,expiry=selector) for leg in structure.legs))
        planrow = {k:getattr(row,k) for k in ("event_id","ticker","event_date","session","entry_date","exit_date")}
        priced, reason = replay.replay_one(structure,planrow,replay.ChainIndex(frames))
        assert reason is None, reason
        for actual in priced:
            reference = trades[(trades.event_id == row.event_id)&(trades.offset == row.offset)&
                               (trades.rule == row.rule)&(trades.fill_alpha == actual["fill_alpha"])].iloc[0]
            for key in ("entry_cost","exit_value","ret"):
                err = abs(actual[key]-reference[key])
                max_error = max(max_error,err)
                assert err < 1e-10, (key,err)
            checks += 1
    write_json(RESULTS / "engine_equivalence.json",dict(cells=len(sample),alpha_checks=checks,max_error=max_error))
    log(f"Engine equality: {checks} alpha checks; max error={max_error}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    events = get_events()
    entries = entry_compacts()
    candidates, omissions = candidate_entries(events,entries)
    if candidates.empty:
        raise RuntimeError("No complete candidates yet")
    candidates.to_parquet(RESULTS / "candidates.parquet", index=False)
    exits = exit_quotes(candidates)
    joined = join_exits(candidates,exits)
    missing = joined[joined[["call_bid_exit","put_bid_exit"]].isna().any(axis=1)]
    repair_pairs = set(zip(missing.ticker,missing.exit_date.dt.strftime("%Y-%m-%d")))
    # Permanent negative answers and successful broad exit requests cannot be
    # repaired by repeatedly asking the same date again.
    asked = set()
    for r in request_history():
        if r["status"] in ("ok","absent"):
            p = r["params"]
            asked.update((t,p["tradeDate"]) for t in p["ticker"].split(","))
    jobs = batches(repair_pairs-asked)
    write_json(RESULTS / "exit_repair_plan.json",jobs)
    write_json(RESULTS / "coverage.json",dict(events=len(events),candidates=len(candidates),
              entry_omissions=omissions,missing_exit_cells=len(missing),
              repair_pairs=len(repair_pairs),new_exit_calls=len(jobs)))
    trades = price_grid(joined)
    trades.to_parquet(RESULTS / "grid_trades.parquet",index=False)
    if not args.skip_verify:
        verify_engine(trades)
    log(f"Build complete: {len(trades):,} alpha rows; {len(jobs)} exit repair calls required")
if __name__ == "__main__":
    main()
