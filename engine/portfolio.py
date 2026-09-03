#!/usr/bin/env python3
"""The hypothetical book: what you'd hold if you took every recommendation.

    python3 -m engine.portfolio                      # the book, as it stands
    python3 -m engine.portfolio --contracts 1 --json reports/portfolio.json

Reads only the frozen ledger — the predictions the board actually published on
the night it published them, paired with the outcomes the replay later settled.
Nothing here re-scores anything, which is the point: a backtest can be re-run
until it agrees with you, and this cannot.

**What counts as a recommendation.** `gate_pass is True`, and nothing else. A
row where the gate was withheld — out-of-domain name, missing features, no
chain — is not a recommendation you could have acted on, and counting those
would be reading the board's silence as a buy.

**One purchase per trade.** The same upcoming event is proposed again every
night until it enters, so a book that bought each row would hold the same
straddle five times. The buy is taken on the FIRST night the trade was
recommended, because that is the night you would have acted.

**Open, unresolved and unresolvable are three different states**, and none of
them is zero:

``settled``        the event happened and the replay priced it — a real P&L
``open``           the event has not happened yet; capital is committed
``awaiting_exit``  the print has happened but its exit chain is not published.
                   ORATS posts a session's chains around midnight, so a trade
                   that printed today cannot settle until tomorrow. Calling
                   these unresolvable would write off every fresh trade.
``unresolvable``   the exit chain should exist by now and the replay still
                   cannot price it. Reported, never dropped: silently removing
                   the trades a book could not follow is how it flatters itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engine import ledger

__all__ = ["build_book", "summarize", "render", "CONTRACT_MULTIPLIER"]

#: Shares per option contract. One straddle contract costs 100x the quoted
#: per-share premium, and P&L scales with it.
CONTRACT_MULTIPLIER = 100

#: Calendar days a print gets to settle before "cannot price this" becomes a
#: real verdict rather than "the exit chain has not been published yet".
#: Three covers a Friday print whose exit chain lands Monday night.
SETTLE_LAG_DAYS = 3


def build_book(contracts: int = 1) -> pd.DataFrame:
    """One row per distinct recommended trade, with its state and P&L."""
    preds = pd.DataFrame(ledger.read_predictions())
    if preds.empty:
        return pd.DataFrame()
    score = preds["score"].apply(lambda s: s or {})
    preds["gate_pass"] = score.apply(lambda s: s.get("gate_pass"))
    preds["exp_pnl_model"] = score.apply(lambda s: s.get("exp_pnl_model"))
    preds["win_model"] = score.apply(lambda s: s.get("win_model"))
    prices = preds["intended_prices"].apply(lambda p: p or {})
    preds["entry_cost"] = pd.to_numeric(prices.apply(lambda p: p.get("entry_cost")),
                                        errors="coerce")
    preds["as_of"] = pd.to_datetime(preds["as_of"])
    preds["event_date"] = pd.to_datetime(preds["event_date"])

    rec = preds[preds["gate_pass"] == True].copy()  # noqa: E712 — None must not pass
    if rec.empty:
        return pd.DataFrame()

    # The night you would have acted: the first time the board said buy.
    rec = rec.sort_values("as_of")
    book = rec.groupby(["ticker", "strategy", "event_date"], as_index=False).first()

    outcomes = pd.DataFrame(ledger.read_outcomes())
    if not outcomes.empty:
        outcomes = outcomes.drop_duplicates("row_id", keep="last")
        book = book.merge(
            outcomes[["row_id", "status", "realized_pnl", "realized_entry_cost",
                      "realized_exit_value", "reason"]],
            on="row_id", how="left",
        )
    else:
        for c in ("status", "realized_pnl", "realized_entry_cost",
                  "realized_exit_value", "reason"):
            book[c] = None

    # A print settles on the FIRST session after it, and that session's chains
    # are published later still. Anything inside that window has not had its
    # chance to settle yet, and must not be counted as a failure.
    today = pd.Timestamp.today().normalize()
    settle_window = today - pd.Timedelta(days=SETTLE_LAG_DAYS)
    book["state"] = np.where(
        book["status"] == "resolved", "settled",
        np.where(
            book["event_date"] >= today, "open",
            np.where(book["event_date"] >= settle_window, "awaiting_exit", "unresolvable"),
        ),
    )
    # Capital is the quoted premium at the decision, which is what you would
    # have committed. `realized_entry_cost` is what the replay actually paid;
    # the two differ by the quote/fill drift and both are reported.
    size = contracts * CONTRACT_MULTIPLIER
    book["capital"] = book["entry_cost"] * size
    book["pnl"] = pd.to_numeric(book["realized_pnl"], errors="coerce") * pd.to_numeric(
        book["realized_entry_cost"], errors="coerce") * size
    book["contracts"] = contracts
    return book.sort_values(["event_date", "ticker"]).reset_index(drop=True)


def summarize(book: pd.DataFrame) -> dict[str, Any]:
    if book.empty:
        return {"n_recommended": 0}
    settled = book[book["state"] == "settled"]
    out: dict[str, Any] = {
        "n_recommended": int(len(book)),
        "by_state": {k: int(v) for k, v in book["state"].value_counts().items()},
        "capital_committed": float(book["capital"].sum(skipna=True)),
        "n_settled": int(len(settled)),
        "pnl": float(settled["pnl"].sum(skipna=True)) if len(settled) else 0.0,
        "capital_settled": float(settled["capital"].sum(skipna=True)) if len(settled) else 0.0,
    }
    if len(settled):
        out["return_on_capital"] = (
            out["pnl"] / out["capital_settled"] if out["capital_settled"] else float("nan")
        )
        out["win_rate"] = float((settled["pnl"] > 0).mean())
        out["mean_trade_return"] = float(
            pd.to_numeric(settled["realized_pnl"], errors="coerce").mean()
        )
        out["best"] = float(settled["pnl"].max())
        out["worst"] = float(settled["pnl"].min())
        out["by_strategy"] = {
            str(k): {"n": int(len(g)), "pnl": float(g["pnl"].sum(skipna=True)),
                     "mean_ret": float(pd.to_numeric(g["realized_pnl"], errors="coerce").mean())}
            for k, g in settled.groupby("strategy")
        }
    return out


def render(book: pd.DataFrame, summary: dict) -> str:
    if book.empty:
        return ("\nNo recommendations in the ledger yet — no row has "
                "gate_pass=True.\n")
    L = ["", "=" * 72, "HYPOTHETICAL BOOK — every recommendation, taken at the recommended time",
         "=" * 72,
         f"  recommendations      : {summary['n_recommended']:,} distinct trades",
         f"  by state             : {summary.get('by_state')}",
         f"  contracts per trade  : {int(book['contracts'].iloc[0])}",
         f"  capital committed    : ${summary['capital_committed']:,.0f}", ""]
    if summary.get("n_settled"):
        L += [f"  SETTLED ({summary['n_settled']}):",
              f"    P&L                : ${summary['pnl']:+,.0f}",
              f"    capital at risk    : ${summary['capital_settled']:,.0f}",
              f"    return on capital  : {summary['return_on_capital']:+.2%}",
              f"    win rate           : {summary['win_rate']:.1%}",
              f"    mean trade return  : {summary['mean_trade_return']:+.2%}",
              f"    best / worst       : ${summary['best']:+,.0f} / ${summary['worst']:+,.0f}", ""]
        for k, v in (summary.get("by_strategy") or {}).items():
            L.append(f"    {k:<12s} n={v['n']:<4d} P&L ${v['pnl']:+,.0f}  mean {v['mean_ret']:+.2%}")
        L.append("")
    cols = ["as_of", "ticker", "strategy", "event_date", "state",
            "entry_cost", "realized_pnl", "pnl"]
    view = book[cols].copy()
    view["as_of"] = view["as_of"].dt.date
    view["event_date"] = view["event_date"].dt.date
    L += ["  the book:", "    " + view.to_string(index=False).replace("\n", "\n    "), ""]
    n_wait = int((book["state"] == "awaiting_exit").sum())
    if n_wait:
        L += [f"  {n_wait} trade(s) have printed but their exit chain is not published yet;",
              "  they settle once ORATS posts that session. Not failures.", ""]
    n_unres = int((book["state"] == "unresolvable").sum())
    if n_unres:
        L += [f"  {n_unres} recommended trade(s) could not be priced and are shown, not dropped —",
              "  removing them would flatter the book by exactly the trades it could not follow.", ""]
    L += ["=" * 72, ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    book = build_book(contracts=args.contracts)
    summary = summarize(book)
    print(render(book, summary))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"summary": summary, "book": json.loads(book.to_json(orient="records"))
             if not book.empty else []}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
