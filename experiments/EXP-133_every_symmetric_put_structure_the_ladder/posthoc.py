#!/usr/bin/env python3
"""POST-HOC — the same chooser, forbidden from picking a net debit it cannot fill.

    python3 experiments/EXP-133.../posthoc.py            (after run.py)

**This is not a registered arm and nothing here is promotable.** It was written
after EXP-133's primary had run, in response to what the primary did, and it is
reported as a curve with the registered rule marked on it rather than as a
result. A successor experiment has to register the floor before seeing the
answer for any of this to be evidence.

**What the primary did.** The registered chooser maximises
``(E[exit value] - cost) / cost`` over every admissible candidate. That ratio
is unbounded as the debit goes to zero, and an eight-leg structure whose long
and short legs nearly cancel has a net mid debit that can land anywhere,
including at half a cent. So the argmax walks straight into the cheapest net
debit the ladder can produce — which is the same failure EXP-126's `choose_rr`
had, arriving through a different criterion, and it is why that risk is
registered in this spec's `known_before_registration`.

**Why a floor is a pricing fact and not a strategy rule.** US listed options
quote on a one-cent grid below $3.00 and a five-cent grid above it. The MID of
a one-tick market is half a tick, which is not a price anyone fills at on a
single leg; net across eight legs, a debit of a few cents is the residue of
eight mid-quotes rather than a number a broker would take. The program already
carries this idea as ``engine.fills.MIN_MEANINGFUL_COST``, set at 1e-6 because
EXP-121 measured CND-P's four-leg noise band topping out at 2e-15 and the next
real price at $0.01. Eight legs move that band up by orders of magnitude, and
nobody had a reason to look until a search started optimising into it.

**The better knob, found by looking at the worst trade.** MDT 2024-05-23: the
seven-strike twin peak at $1 spacing, max payoff $2.00 at expiry, entry mid
debit **half a cent**. Six of its seven legs quote inside 5%; the seventh, the
$90 wing, quotes 2.85 / 4.85 — two dollars wide on a $3.85 mid — and that one
leg's mid is what produced the half-cent net. Its MEAN relative leg spread is
11.4%, comfortably inside the registered 25% filter, because a mean over legs
cannot see that the NET is a small difference of large numbers. The quantity
that can see it is the total half-spread across the legs divided by the net
debit: how uncertain the price is, in units of the price. On that trade it is
**251x**. So the sweep below has two axes, and the second is the one that
matters:

    debit floor      a dollar minimum on the net debit
    spread-to-debit  total entry half-spread / net debit, capped

Both are swept, never chosen: every level is reported, so the reader sees the
whole curve and can see where — or whether — an edge survives.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import pnl_sim, replay as replay_mod  # noqa: E402
from engine.fills import MIN_MEANINGFUL_COST  # noqa: E402
from engine.structures import twin_peak_5  # noqa: E402

import build as build_mod  # noqa: E402
import run as run_mod  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

#: Sweep A. 0.00 reproduces the registered primary exactly and is the reference
#: point every other level is read against.
FLOORS = (0.00, 0.05, 0.10, 0.25, 0.50, 1.00)

#: Sweep B, and the one the worst trade points at: the total entry half-spread
#: as a multiple of the net debit. `inf` reproduces the registered primary; the
#: incumbent TWIN-P5 book's own median on this universe is 0.24, so the bottom
#: of this sweep is where the shape the program already trades lives.
S2D_CAPS = (float("inf"), 4.0, 2.0, 1.0, 0.5, 0.25)

#: Only the primary's rule is swept. The point is what the FLOORS do to the
#: chooser, and adding arms would turn a diagnostic into a leaderboard.
BASE_ARM = "best_all"


def _half_spread(ev) -> np.ndarray:
    """Total entry half-spread of every candidate, in dollars.

    Half the bid-ask on each leg, times its contract count, summed. It is how
    far the entry price could move against you before any adverse selection —
    the width of what "mid" is guessing at, for the structure as a whole.
    """
    half = 0.5 * (ev["ask_e"] - ev["bid_e"])
    return (np.abs(build_mod.GRID.qty) * half[ev["idx"]]).sum(axis=1)


def build_floors(*, force: bool = False) -> pd.DataFrame:
    """Re-select under each floor, on the same events, draws and quotes.

    A second pass over the chains rather than a filter on the primary's output,
    and the difference matters: filtering would report the primary's expensive
    PICKS, which answers "is the edge only in the pennies". Re-selecting asks
    the question that is actually interesting — "what would this rule have
    bought if the pennies were not on the menu" — and they are not the same
    set, because the second-best candidate at a penny is rarely the best
    candidate at a dime.
    """
    out = RESULTS / "posthoc_candidates.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)

    started = time.time()
    events = build_mod.event_universe()
    events["event_date"] = pd.to_datetime(events["event_date"])
    events = events.merge(build_mod.forecasts(), on=["ticker", "event_date"], how="inner")
    plan = replay_mod.filter_plan_by_availability(
        replay_mod.plan_events(twin_peak_5(wing_multiple=3, width_moneyness=0.05), events))
    keyframe = plan.frame.set_index("event_id")
    events = events[events["event_id"].isin(set(plan.frame["event_id"]))].copy()
    events["_year"] = events["event_date"].dt.year
    pool = pnl_sim.ResidualPool(build_mod.residual_history())

    parts = []
    for year, block in events.groupby("_year", sort=True):
        rows = keyframe.reindex(block["event_id"]).dropna(subset=["entry_date"])
        keys = set()
        for col in ("entry_date", "exit_date"):
            keys |= {(t, d) for t, d in zip(rows["ticker"], rows[col])}
        index = replay_mod.load_chain_index(keys, progress_every=0)
        merged = block.merge(plan.frame[["event_id", "entry_date", "exit_date"]],
                             on="event_id", how="inner")
        for row in merged.to_dict("records"):
            entry_rows = index.get(row["ticker"], row["entry_date"])
            exit_rows = index.get(row["ticker"], row["exit_date"])
            if entry_rows is None or exit_rows is None or entry_rows.empty or exit_rows.empty:
                continue
            rng_pick = np.random.default_rng(
                int.from_bytes(hashlib.sha256(
                    f"pick|{row['event_id']}".encode()).digest()[:8], "big"))
            try:
                ev = build_mod.price_event(row, entry_rows, exit_rows, pool, rng_pick=rng_pick)
            except Exception:                                   # noqa: BLE001
                continue
            cost_mid = ev["priced"][build_mod.MID][0]
            base = ev["tradeable"] & np.isfinite(ev["sim_sel"])
            with np.errstate(divide="ignore", invalid="ignore"):
                s2d = _half_spread(ev) / cost_mid
            arms = [(f"debit_{f:.2f}", base & (cost_mid >= max(f, MIN_MEANINGFUL_COST)))
                    for f in FLOORS]
            arms += [(f"s2d_{c:g}", base & (cost_mid > MIN_MEANINGFUL_COST) & (s2d <= c))
                     for c in S2D_CAPS]
            for name, mask in arms:
                ok = np.flatnonzero(mask)
                if ok.size == 0:
                    continue
                pick = int(ok[np.argmax(ev["sim_sel"][ok])])
                parts.extend(build_mod._emit(
                    ev, row, name, ev["idx"][pick], build_mod.GRID.qty[pick], pick,
                    float(ev["sim_sel"][pick]), float(ev["sim_gate"][pick]),
                    str(build_mod.GRID.key[pick])))
        del index
        print(f"[posthoc] {year}: {len(parts):,} rows, {time.time()-started:.0f}s",
              flush=True)

    frame = pd.DataFrame(parts)
    frame.to_parquet(out, index=False)
    return frame


def main() -> None:
    built = build_mod.build_all()
    trades = run_mod.attach(build_floors(), built["tallies"])

    arm_names = [(f"debit_{f:.2f}", "debit floor", f"${f:.2f}") for f in FLOORS]
    arm_names += [(f"s2d_{c:g}", "spread-to-debit",
                   "no cap" if c == float("inf") else f"<= {c:g}x") for c in S2D_CAPS]

    table, summary = [], {}
    for arm, axis, label in arm_names:
        kept, marked = run_mod.arm_rows(trades, arm)
        mid = kept[np.isclose(kept["fill_alpha"].astype(float), run_mod.MID)]
        stats = run_mod.book_stats(mid)
        if not stats:
            continue
        stats["breakeven_alpha"] = run_mod.breakeven(kept)
        # The alpha grid is where a debit that is not a real price shows up:
        # a structure whose mid net is pennies costs a multiple of that at any
        # fill worse than mid, so its return collapses long before alpha 0.
        stats["mean_at_alpha_25"] = float(
            kept.loc[np.isclose(kept["fill_alpha"], 0.25), "ret"].mean())
        stats["median_ret"] = float(mid["ret"].median())
        stats["cost_median"] = float(mid["entry_cost"].median())
        stats["cost_p05"] = float(mid["entry_cost"].quantile(0.05))
        summary[arm] = stats
        registered = arm in ("debit_0.00", "s2d_inf")
        table.append([
            axis, label + (" — the registered rule" if registered else ""),
            f"{stats['n']:,}", f"{stats['tickers']:,}",
            f"${stats['cost_median']:.2f}", f"${stats['cost_p05']:.2f}",
            f"{100*stats['mean']:+.1f}%", f"{100*stats['median_ret']:+.1f}%",
            f"{100*stats['return_on_capital']:+.1f}%",
            f"{100*stats['mean_at_alpha_25']:+.1f}%",
            f"{stats['breakeven_alpha']:.3f}" if stats["breakeven_alpha"] is not None else "never",
            f"{100*stats['cagr']:+.1f}%" if stats["cagr"] == stats["cagr"] else "n/a",
            f"{stats['sharpe_trade']:.2f}", f"{stats['years_positive']}/{stats['years']}",
            f"{100*stats['centre_share']:.0f}%",
        ])
        print(f"[posthoc] {arm}: {stats['n']:,} trades, median cost "
              f"${stats['cost_median']:.2f}, mean {100*stats['mean']:+.1f}%, "
              f"at a=0.25 {100*stats['mean_at_alpha_25']:+.1f}%, CAGR "
              f"{100*stats['cagr']:+.1f}%", flush=True)

    (RESULTS / "posthoc_debit_floor.json").write_text(
        json.dumps({"floors": list(FLOORS), "s2d_caps": [str(c) for c in S2D_CAPS],
                    "arms": summary}, indent=1, default=str))

    header = ["sweep", "level", "n", "tickers", "median debit", "5th pct debit",
              "mean/trade", "median/trade", "on capital", "mean at a=0.25",
              "breakeven a", "CAGR", "Sharpe", "years+", "centre"]
    lines = [
        "",
        "## Appendix P — POST-HOC: the chooser with a floor under the net debit",
        "",
        "*Written after the primary ran, in response to what it did. Not a "
        "registered arm; nothing here is promotable, and a successor experiment "
        "has to register the floor before seeing the answer for any of it to be "
        "evidence. Reported as a sweep rather than a chosen level, so the reader "
        "sees the whole curve.*",
        "",
        "The registered chooser maximises `(E[exit value] − cost) / cost`, which "
        "is unbounded as the debit goes to zero. An eight-leg structure whose "
        "longs and shorts nearly cancel can net to a few cents at mid, so the "
        "argmax walks into the cheapest net debit the ladder can produce. US "
        "options quote on a one-cent grid below $3.00; the mid of a one-tick "
        "market is half a tick, and a net of a few cents across eight legs is "
        "the residue of eight mid-quotes rather than a price a broker fills. "
        "`mean at a=0.25` is the column that settles it: a debit that is not "
        "real collapses the moment the fill is anything but perfect mid.",
        "",
        "Two knobs are swept. A dollar floor under the debit is the blunt one. "
        "The sharp one is **spread-to-debit** — the total entry half-spread over "
        "the net debit, i.e. how uncertain the price is in units of the price — "
        "because the registered 25% MEAN-leg-spread filter cannot see this "
        "failure at all: the chooser's picks and the incumbent's have the same "
        "median mean-leg spread (0.147 against 0.148), while their "
        "spread-to-debit medians are 1.31 and 0.24. A mean over legs cannot "
        "tell that the NET is a small difference of large numbers.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in table]
    (RESULTS / "posthoc_debit_floor.md").write_text("\n".join(lines) + "\n")
    print(f"[posthoc] wrote {RESULTS / 'posthoc_debit_floor.md'}", flush=True)


if __name__ == "__main__":
    main()
