#!/usr/bin/env python3
"""EXP-121 — building the CND-P trade set, and the risk mechanics it registers.

Pricing goes through :func:`engine.replay.replay` like every other structure in
the programme; nothing here prices a leg. The one thing this module does that
``engine.build_trades`` does not is keep the result in the experiment's own
results/ directory instead of the Tier-2 ``trades`` table: ``build_trades``
rewrites every ``engine.replay`` row it did not just rebuild, so landing a
fourth strategy there means rebuilding the other three, and a descriptive
experiment has no business doing that.
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
from engine.features import load_panel  # noqa: E402
from engine.structures import put_condor  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
STRATEGY = "CND-P"

#: A condor cannot settle below zero, so a return below -100% of the debit is
#: never the structure losing money — it is the EXIT quotes.
DEBIT_EXCEEDED = -1.0


def trades_path(width: float) -> Path:
    return RESULTS / f"trades_cnd_p_w{int(round(width * 1000)):04d}.parquet"


def build_all_widths(widths, *, force: bool = False) -> dict[float, pd.DataFrame]:
    """Replay every candidate width, one year of chains in memory at a time.

    The chain index is the expensive part and it does not depend on the width,
    so all four condors are priced off each year's index before it is dropped.
    """
    widths = [float(w) for w in widths]
    wanted = [w for w in widths if force or not trades_path(w).exists()]
    out: dict[float, pd.DataFrame] = {
        w: pd.read_parquet(trades_path(w)) for w in widths if w not in wanted
    }
    for w in out:
        print(f"[condor] width {w}: {out[w]['event_id'].nunique():,} events from cache",
              flush=True)
    if not wanted:
        return {w: out[w] for w in widths}

    started = time.time()
    events = event_universe()
    plan = replay_mod.plan_events(put_condor(), events)
    plan = replay_mod.filter_plan_by_availability(plan)
    frame = plan.frame
    print(f"[condor] {len(frame):,} events have both chains", flush=True)

    rows: dict[float, list[pd.DataFrame]] = {w: [] for w in wanted}
    skipped: dict[float, dict] = {w: {} for w in wanted}
    for year, block in frame.groupby(frame["entry_date"].dt.year, sort=True):
        keys = set()
        for col in ("entry_date", "exit_date"):
            keys |= {(t, d) for t, d in zip(block["ticker"], block[col])}
        index = replay_mod.load_chain_index(keys, progress_every=0)
        year_events = events[events["event_id"].isin(set(block["event_id"]))]
        for w in wanted:
            result = replay_mod.replay(
                STRATEGY, year_events, structure=put_condor(width=w),
                index=index, progress_every=0)
            if len(result.trades):
                rows[w].append(result.trades)
            for reason, n in result.skipped.items():
                skipped[w][reason] = skipped[w].get(reason, 0) + int(n)
        del index
        print(f"[condor] {year}: "
              + ", ".join(f"w={w} {sum(len(f) for f in rows[w]):,} rows" for w in wanted)
              + f", {time.time() - started:.0f}s", flush=True)

    for w in wanted:
        frames = rows[w]
        trades = (pd.concat(frames, ignore_index=True) if frames
                  else replay_mod._empty_trades())
        trades.to_parquet(trades_path(w), index=False)
        (RESULTS / f"skips_w{int(round(w * 1000)):04d}.json").write_text(
            json.dumps(skipped[w], indent=1))
        out[w] = trades
        print(f"[condor] width {w}: {trades['event_id'].nunique():,} events priced, "
              f"skips {skipped[w]}", flush=True)
    return {w: out[w] for w in widths}


# --------------------------------------------------------------------------
# the pre-registered risk mechanics
# --------------------------------------------------------------------------


def _f(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def parse_legs(mid: pd.DataFrame) -> pd.DataFrame:
    """Four-leg contract details out of the stored legs blob."""
    out_rows = []
    for t in mid.itertuples(index=False):
        doc = json.loads(t.legs) if isinstance(t.legs, str) else {}
        row: dict = {
            "event_id": t.event_id, "ticker": t.ticker, "event_date": t.event_date,
            "exit_date": t.exit_date, "entry_cost": t.entry_cost,
            "exit_value": t.exit_value, "ret": t.ret,
            "wide_market": getattr(t, "wide_market", None),
            "quote_repaired": getattr(t, "quote_repaired", None),
            "spot_entry": _f(doc.get("spot_entry")),
            "spot_exit": _f(doc.get("spot_exit")),
            "dte_entry": doc.get("dte_entry"),
        }
        entry = {leg["name"]: leg for leg in (doc.get("entry") or [])}
        exit_ = {leg["name"]: leg for leg in (doc.get("exit") or [])}
        for name in ("long_lo", "short_lo", "short_hi", "long_hi"):
            row[f"k_{name}"] = _f(entry.get(name, {}).get("strike"))
            row[f"xbid_{name}"] = _f(exit_.get(name, {}).get("bid"))
            row[f"xask_{name}"] = _f(exit_.get(name, {}).get("ask"))
        out_rows.append(row)
    legs = pd.DataFrame(out_rows)
    legs["year"] = pd.to_datetime(legs["event_date"]).dt.year
    legs["spacing"] = legs["k_short_hi"] - legs["k_short_lo"]
    legs["spacing_pct"] = 100.0 * legs["spacing"] / legs["spot_entry"]
    legs["realized_move"] = legs["spot_exit"] / legs["spot_entry"] - 1.0
    legs["abs_move_pct"] = 100.0 * legs["realized_move"].abs()
    legs["cost_over_spacing"] = legs["entry_cost"] / legs["spacing"]
    return legs


def classify_exceedances(legs: pd.DataFrame) -> pd.DataFrame:
    """Each debit-exceeding trade: real loss, or which exit-quote artifact.

    A long condor's terminal payoff is bounded below by zero — that is what the
    even spacing buys — so a close that costs more than the debit is a statement
    about the EXIT quotes, not about the trade. Three signatures are checkable
    from what the row already carries, with no chain re-read: a repaired crossed
    quote anywhere in the event's chains, a leg quoted through a market wider
    than half its mid, and a long leg bought back below its own intrinsic (or a
    short sold above it), which no rational market fills.
    """
    exceeded = legs[legs["ret"] < DEBIT_EXCEEDED].copy()
    if exceeded.empty:
        exceeded["classification"] = pd.Series(dtype=str)
        return exceeded

    classes = []
    for row in exceeded.itertuples(index=False):
        reasons = []
        if bool(row.quote_repaired):
            reasons.append("crossed_quote_repaired")
        if bool(row.wide_market):
            reasons.append("wide_market_flagged")
        stale = []
        for name in ("long_lo", "short_lo", "short_hi", "long_hi"):
            bid = getattr(row, f"xbid_{name}")
            ask = getattr(row, f"xask_{name}")
            strike = getattr(row, f"k_{name}")
            if not (np.isfinite(bid) and np.isfinite(ask) and np.isfinite(strike)):
                continue
            mid = 0.5 * (bid + ask)
            intrinsic = max(strike - row.spot_exit, 0.0)
            if intrinsic > 0 and mid < 0.95 * intrinsic:
                stale.append(name)
        if stale:
            reasons.append("exit_below_intrinsic:" + "+".join(stale))
        classes.append("data_artifact: " + "+".join(reasons) if reasons else "real_loss")
    exceeded["classification"] = classes
    return exceeded


def risk_mechanics(mid: pd.DataFrame, geometry: dict | None = None) -> dict:
    """The four analyses spec.yaml registers as required outputs."""
    from engine.data import store

    legs = parse_legs(mid)
    sec = store.read_table("securities", years=range(2017, 2027),
                           columns=["ticker", "year", "mcap_usd"])
    legs = legs.merge(sec, on=["ticker", "year"], how="left")
    legs["mcap_bucket"] = pd.cut(legs["mcap_usd"], bins=[-1, 1e9, 10e9, np.inf],
                                 labels=["<1B", "1-10B", ">10B"])

    # -- 1. defined risk ---------------------------------------------------
    exceeded = classify_exceedances(legs)
    by_bucket = {}
    for bucket, g in legs.groupby("mcap_bucket", observed=True):
        by_bucket[str(bucket)] = {
            "n": int(len(g)),
            "n_exceeded": int((g["ret"] < DEBIT_EXCEEDED).sum()),
            "worst_ret": float(g["ret"].min()) if len(g) else None,
            "p01_ret": float(np.percentile(g["ret"], 1)) if len(g) else None,
        }
    worst = legs.loc[legs["ret"].idxmin()] if len(legs) else None

    # -- 2. assignment exposure -------------------------------------------
    depth_hi = (legs["k_short_hi"] - legs["spot_exit"]) / legs["k_short_hi"]
    depth_lo = (legs["k_short_lo"] - legs["spot_exit"]) / legs["k_short_lo"]
    beyond = (legs["k_short_hi"] - legs["spot_exit"]) > legs["spacing"]
    assignment = {
        "n": int(len(legs)),
        "short_hi_itm_at_entry": float(
            (legs["k_short_hi"] > legs["spot_entry"]).mean()),
        "short_hi_itm_at_post_print_close": float((depth_hi > 0).mean()),
        "short_lo_itm_at_post_print_close": float((depth_lo > 0).mean()),
        "both_shorts_itm": float(((depth_hi > 0) & (depth_lo > 0)).mean()),
        "short_hi_itm_by_more_than_one_spacing": float(beyond.mean()),
        "median_short_hi_depth_pct": float(100.0 * depth_hi.median()),
        "dte_entry": {
            "median": float(pd.to_numeric(legs["dte_entry"], errors="coerce").median()),
            "min": float(pd.to_numeric(legs["dte_entry"], errors="coerce").min()),
            "max": float(pd.to_numeric(legs["dte_entry"], errors="coerce").max()),
        },
    }

    # -- 3. geometry realized ----------------------------------------------
    realized = {
        "wide_market_share": float(legs["wide_market"].astype(bool).mean()),
        "quote_repaired_share": float(legs["quote_repaired"].astype(bool).mean()),
        "spacing_pct_median": float(legs["spacing_pct"].median()),
        "spacing_pct_p25": float(legs["spacing_pct"].quantile(0.25)),
        "spacing_pct_p75": float(legs["spacing_pct"].quantile(0.75)),
        "cost_over_spacing_median": float(legs["cost_over_spacing"].median()),
        "max_payoff_is_the_spacing": (
            "terminal payoff peaks at one spacing, so cost/spacing is the share "
            "of the maximum that is paid up front"),
    }
    if geometry is not None:
        realized["stage0_resolvability"] = geometry

    # -- 4. the oracle ceiling --------------------------------------------
    panel = load_panel()[["ticker", "date", "implied_move"]].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    legs = legs.merge(panel, left_on=["ticker", "event_date"],
                      right_on=["ticker", "date"], how="left")
    oracle = {
        "hindsight_quintile_by_realized_abs_move": _quintile_block(
            legs, "abs_move_pct"),
        "tradeable_quintile_by_implied_move": _quintile_block(
            legs, "implied_move"),
        "note": ("the first cut uses the realized move and is HINDSIGHT — it is "
                 "the ceiling a gate must fit under, never a strategy; the second "
                 "is a real cut on a quote known before the print"),
    }

    return {
        "defined_risk": {
            "n": int(len(exceeded)),
            "share": float(len(exceeded) / len(legs)) if len(legs) else None,
            "worst_ret": float(legs["ret"].min()) if len(legs) else None,
            "worst_trade": ({
                "ticker": worst["ticker"],
                "event_date": str(worst["event_date"])[:10],
                "entry_cost": float(worst["entry_cost"]),
                "exit_value": float(worst["exit_value"]),
                "ret": float(worst["ret"]),
                "realized_move": float(worst["realized_move"]),
            } if worst is not None else {}),
            "by_year": {str(int(y)): int(n) for y, n in
                        exceeded.groupby("year").size().items()},
            "by_mcap_bucket": by_bucket,
            "classification": {k: int(v) for k, v in
                               exceeded["classification"].value_counts().items()},
        },
        "assignment_exposure": assignment,
        "realized_geometry": realized,
        "oracle_ceiling": oracle,
    }


def _quintile_block(legs: pd.DataFrame, column: str) -> dict:
    """Mean return per quintile of ``column``, quietest first."""
    values = pd.to_numeric(legs[column], errors="coerce")
    ok = values.notna() & legs["ret"].notna()
    if ok.sum() < 100:
        return {"n": int(ok.sum()), "note": "too few rows to quintile"}
    q = pd.qcut(values[ok], 5, labels=False, duplicates="drop")
    out = {"n": int(ok.sum()), "column": column, "quintiles": []}
    for bucket in sorted(pd.unique(q.dropna())):
        rows = legs[ok][q == bucket]
        out["quintiles"].append({
            "quintile": int(bucket) + 1,
            "n": int(len(rows)),
            f"{column}_median": float(pd.to_numeric(rows[column]).median()),
            "mean_ret": float(rows["ret"].mean()),
            "win_rate": float((rows["ret"] > 0).mean()),
        })
    quiet = out["quintiles"][0]
    out["quietest_quintile_mean_ret"] = quiet["mean_ret"]
    out["quietest_quintile_win_rate"] = quiet["win_rate"]
    return out
