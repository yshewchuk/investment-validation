#!/usr/bin/env python3
"""§6.3 — what a session of lead time costs, measured on real quotes.

    python3 checks/t2_drift.py --json reports/t2_drift.json

The T−2 design quotes a premium at the `D−1` close and fills at the `D0` close.
That gap is not a modelling choice, it is a price, and this measures it:

    drift = entry_cost@D0 / quoted_cost@D-1 - 1

**This is a go/no-go, and it runs before any retrain.** The ungated STR-THRU
base return is +2.92% per trade. If the median drift eats a meaningful share of
that, then committing a session early costs more than the timing buys, and a
retrained T−2 gate would be answering the wrong question — no model can recover
premium the entry never had.

Two strike rules are priced, because the choice is real and the second is
nearly free once the first is loaded (guide §5.4):

``A``   name the strike and expiry at `D−1` and buy **that contract** at `D0`.
        The board's number and the trade are the same thing. Recommended.
``A'``  re-resolve ATM at `D0`. Cleaner moneyness, but the premium the board
        showed refers to a contract you did not buy.

No quota is spent and no model is involved: both legs of the comparison are
quotes already in the store.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import paths, replay  # noqa: E402
from engine.data import coverage, store  # noqa: E402
from engine.fills import MID  # noqa: E402
from engine.structures import (  # noqa: E402
    ExpirySelector,
    StrikeSelector,
    StructureError,
    price_structure,
    straddle_through,
)

__all__ = ["measure_drift", "summarize", "render"]

#: The base the drift is judged against — ungated STR-THRU mean return.
BASE_RETURN = 0.0292


def _pin_contract(structure, strike: float, expiry: pd.Timestamp):
    """The same structure, bound to one already-chosen contract.

    Arm A's whole claim is that the board names a contract and the trade buys
    *that* contract. Re-resolving ATM at the entry would quietly compare two
    different trades and report the difference as drift.
    """
    legs = []
    for leg in structure.legs:
        legs.append(
            type(leg)(
                name=leg.name,
                right=leg.right,
                side=leg.side,
                expiry=ExpirySelector(kind="fixed", expiry=pd.Timestamp(expiry)),
                strike=StrikeSelector(kind="fixed", strike=float(strike)),
                qty=leg.qty,
            )
        )
    return type(structure)(
        name=structure.name,
        legs=tuple(legs),
        entry_offset=structure.entry_offset,
        exit_offset=structure.exit_offset,
        decision_offset=structure.decision_offset,
        description=structure.description,
        params=dict(structure.params),
    )


def measure_drift(decision_offset: int = -1, min_year: int = 2018, limit: int | None = None):
    """One row per event: what was quoted at the decision, what it cost at entry."""
    structure = straddle_through(decision_offset=decision_offset)
    events = store.read_table("earnings_events")
    events = events[(events["year"] >= min_year) & events["session"].notna()]

    plan = replay.plan_events(structure, events)
    print(f"  planned {len(plan.frame):,} events", flush=True)
    plan = replay.filter_plan_by_availability(plan)
    print(f"  {len(plan.frame):,} have decision, entry and exit chains", flush=True)
    if limit:
        plan.frame = plan.frame.head(limit)

    keys = set(zip(plan.frame["ticker"], plan.frame["decision_date"]))
    keys |= set(zip(plan.frame["ticker"], plan.frame["entry_date"]))
    index = replay.load_chain_index(keys)

    rows: list[dict] = []
    skipped: dict[str, int] = {}
    started = time.time()
    for i, row in enumerate(plan.frame.to_dict("records"), 1):
        ticker = row["ticker"]
        dec_rows = index.get(ticker, row["decision_date"])
        ent_rows = index.get(ticker, row["entry_date"])
        if dec_rows is None or ent_rows is None:
            skipped["missing_chain"] = skipped.get("missing_chain", 0) + 1
            continue
        dec_rows, ent_rows = replay._clean(dec_rows), replay._clean(ent_rows)
        if dec_rows.empty or ent_rows.empty:
            skipped["bad_quote"] = skipped.get("bad_quote", 0) + 1
            continue

        from engine.structures import ChainSnapshot

        dec_snap = ChainSnapshot(
            ticker=ticker, obs_date=row["decision_date"], event_date=row["event_date"],
            rows=dec_rows, session=row["session"],
        )
        ent_snap = ChainSnapshot(
            ticker=ticker, obs_date=row["entry_date"], event_date=row["event_date"],
            rows=ent_rows, session=row["session"],
        )
        try:
            quoted = price_structure(structure, dec_snap, MID)
        except StructureError:
            skipped["decision_unresolved"] = skipped.get("decision_unresolved", 0) + 1
            continue

        strike = float(quoted.legs[0].strike)
        expiry = pd.Timestamp(quoted.legs[0].expiry)
        # Arm A: the contract the board named.
        try:
            same = price_structure(_pin_contract(structure, strike, expiry), ent_snap, MID)
            cost_a = same.cost
        except StructureError:
            cost_a = float("nan")
        # Arm A': ATM re-resolved at the entry close.
        try:
            cost_ap = price_structure(structure, ent_snap, MID).cost
        except StructureError:
            cost_ap = float("nan")

        rows.append(
            {
                "event_id": row["event_id"], "ticker": ticker,
                "event_date": row["event_date"], "session": row["session"],
                "decision_date": row["decision_date"], "entry_date": row["entry_date"],
                "strike": strike, "expiry": expiry,
                "spot_decision": quoted.spot, "spot_entry": ent_snap.spot_price,
                "quoted_cost": quoted.cost, "entry_cost_same": cost_a,
                "entry_cost_atm": cost_ap,
                "dte_decision": int(quoted.legs[0].dte),
            }
        )
        if i % 5000 == 0:
            print(f"  priced {i:,}/{len(plan.frame):,}  {time.time()-started:.0f}s", flush=True)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, skipped
    frame["drift_same"] = frame["entry_cost_same"] / frame["quoted_cost"] - 1.0
    frame["drift_atm"] = frame["entry_cost_atm"] / frame["quoted_cost"] - 1.0
    frame["spot_move"] = frame["spot_entry"] / frame["spot_decision"] - 1.0
    frame = coverage.attach_mcap(frame.rename(columns={"event_date": "event_date"}))
    return frame, skipped


def _stats(series: pd.Series) -> dict:
    s = series.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p10": float(s.quantile(0.10)),
        "p90": float(s.quantile(0.90)),
        "share_positive": float((s > 0).mean()),
    }


def _cost_weighted(frame: pd.DataFrame, column: str) -> float:
    """What a BOOK of these trades pays against what it was quoted.

    The mean of a per-trade ratio is not this number and is much worse: 3.7% of
    the events are straddles under $0.50, where a two-cent tick is a 4% move,
    and they drag the mean far above anything a portfolio would experience.
    Summing the dollars first is the aggregate that has a P&L meaning.
    """
    ok = frame[[column, "quoted_cost"]].replace([np.inf, -np.inf], np.nan).dropna()
    if ok.empty or ok["quoted_cost"].sum() == 0:
        return float("nan")
    return float(ok[column].sum() / ok["quoted_cost"].sum() - 1.0)


def summarize(frame: pd.DataFrame) -> dict:
    within = {}
    d = frame["drift_same"].replace([np.inf, -np.inf], np.nan).dropna()
    for band in (0.02, 0.05, 0.10):
        within[f"within_{int(band*100)}pct"] = float((d.abs() <= band).mean())
    out = {
        "n_events": int(len(frame)),
        "base_return": BASE_RETURN,
        "arm_a_same_contract": _stats(frame["drift_same"]),
        "arm_a_prime_atm": _stats(frame["drift_atm"]),
        "spot_move": _stats(frame["spot_move"]),
        "cost_weighted_same": _cost_weighted(frame, "entry_cost_same"),
        "cost_weighted_atm": _cost_weighted(frame, "entry_cost_atm"),
        "abs_drift_median": float(d.abs().median()),
        "abs_drift_p90": float(d.abs().quantile(0.90)),
        "quote_lands_within": within,
        "by_bucket": {}, "by_year": {}, "by_session": {},
    }
    for bucket, chunk in frame.groupby("mcap_bucket"):
        out["by_bucket"][str(bucket)] = _stats(chunk["drift_same"])
    for year, chunk in frame.groupby(frame["event_date"].dt.year):
        out["by_year"][int(year)] = _stats(chunk["drift_same"])
    for session, chunk in frame.groupby("session"):
        out["by_session"][str(session)] = _stats(chunk["drift_same"])
    return out


def render(summary: dict, skipped: dict) -> str:
    a = summary["arm_a_same_contract"]
    ap = summary["arm_a_prime_atm"]
    med = a.get("median", float("nan"))
    lines = [
        "", "=" * 68,
        "T-2 QUOTE/FILL DRIFT  (guide §6.3 — the decision point)",
        "=" * 68,
        f"  events priced on both closes : {summary['n_events']:,}",
        f"  skipped                      : {skipped or '{}'}",
        "",
        "  drift = entry_cost@D0 / quoted_cost@D-1 - 1",
        "",
        f"  {'':<22s} {'n':>8s} {'median':>9s} {'mean':>9s} {'p10':>9s} {'p90':>9s} {'>0':>7s}",
    ]
    for label, st in (("A  same contract", a), ("A' ATM re-resolved", ap),
                      ("   spot move", summary["spot_move"])):
        lines.append(
            f"  {label:<22s} {st['n']:>8,} {st['median']:>8.2%} {st['mean']:>9.2%} "
            f"{st['p10']:>9.2%} {st['p90']:>9.2%} {st['share_positive']:>7.1%}"
        )
    cw, cwa = summary["cost_weighted_same"], summary["cost_weighted_atm"]
    lines += [
        "",
        "  cost-weighted (what a BOOK pays vs what it was quoted):",
        f"    A  same contract   : {cw:+.2%}",
        f"    A' ATM re-resolved : {cwa:+.2%}",
        f"    A costs {cw - cwa:+.2%} more than A'",
        "",
        f"  base STR-THRU return per trade : {summary['base_return']:.2%}",
        f"  median drift as a share of it  : {med / summary['base_return']:.0%}",
        f"  cost-weighted as a share of it : {cw / summary['base_return']:.0%}",
        "",
        "  how well the board's quote predicts the fill:",
        f"    median |drift| {summary['abs_drift_median']:.2%}, "
        f"p90 {summary['abs_drift_p90']:.2%}",
        *[f"    within +/-{k.split('_')[1]:<5s} {v:.1%}"
          for k, v in summary["quote_lands_within"].items()],
        "",
    ]
    for name, key in (("by mcap bucket", "by_bucket"), ("by session", "by_session")):
        lines.append(f"  {name} (Arm A median):")
        for k, st in sorted(summary[key].items()):
            lines.append(f"    {k:<10s} {st.get('median', float('nan')):>8.2%}  ({st.get('n', 0):,} events)")
        lines.append("")
    lines.append("  by year (Arm A median):")
    for k, st in sorted(summary["by_year"].items()):
        lines.append(f"    {k}  {st.get('median', float('nan')):>8.2%}  ({st.get('n', 0):,} events)")
    lines += ["", "=" * 68, ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decision-offset", type=int, default=-1)
    ap.add_argument("--min-year", type=int, default=2018)
    ap.add_argument("--limit", type=int, default=None, help="first N events, for a smoke run")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    frame, skipped = measure_drift(args.decision_offset, args.min_year, args.limit)
    if frame.empty:
        print("no events priced")
        return 1
    summary = summarize(frame)
    print(render(summary, skipped))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"summary": summary, "skipped": skipped}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
