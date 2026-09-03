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

#: Target dollars per position. One contract each is not equal sizing: premiums
#: in a single week ran from $2.40 (AEO) to $69.00 (CASY), so a one-contract
#: book puts 29x more capital behind CASY and lets a few expensive names decide
#: the P&L. Sizing to a budget makes every recommendation count roughly the
#: same, which is what a book measuring SELECTION should do.
#:
#: $10,000 because contracts are WHOLE. A $69 premium is $6,900 a contract, so
#: a smaller budget would round it to one lot and leave the cheap names at 4x
#: the weight; at $10,000 even a $100 premium still gets a position, and the
#: cheap names land close to the target instead of far above it.
CAPITAL_PER_TRADE = 10_000.0


def build_book(contracts: int | None = None,
               capital_per_trade: float = CAPITAL_PER_TRADE) -> pd.DataFrame:
    """One row per distinct recommended trade, with its state and P&L.

    Sized by equal DOLLARS per trade by default. Pass ``contracts`` to size by
    a fixed contract count instead — useful for reading the book as a literal
    order list, misleading for judging the strategy, because it weights the
    expensive premiums far above the cheap ones.

    Contracts are WHOLE, because that is what you can actually buy — the book
    is meant to be a thing you could have executed, not a fractional
    abstraction. It rounds DOWN to stay inside the budget, with a floor of one:
    a premium larger than the whole budget still takes a position rather than
    vanishing, because a book that silently omits its most expensive
    recommendations is not measuring the board it claims to.

    Rounding means sizes are near-equal, not equal, and `capital` records what
    each position actually costs so the spread is visible rather than assumed.
    """
    # Canonical, not raw: the ledger restates the same trade every night it is
    # on the board (5.25x), and as chains arrive the gate verdict CHANGES. The
    # canonical row is the last view at or before the entry — the verdict you
    # would have acted on. Grouping the raw rows instead can pick a night the
    # gate was still withholding and drop a real recommendation.
    preds = pd.DataFrame(ledger.canonical_predictions())
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
    book = rec.sort_values("as_of").reset_index(drop=True)

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
    cost_per_contract = book["entry_cost"] * CONTRACT_MULTIPLIER
    if contracts is not None:
        book["contracts"] = float(contracts)
        book["sizing"] = f"{contracts} contract(s)"
    else:
        # Round DOWN to stay inside the budget, floor of one lot so an
        # expensive name is still taken rather than dropped.
        lots = np.floor(capital_per_trade / cost_per_contract.replace(0, np.nan))
        book["contracts"] = lots.clip(lower=1).fillna(1)
        book["sizing"] = f"~${capital_per_trade:,.0f}/trade, whole lots"
    book["capital"] = cost_per_contract * book["contracts"]
    # P&L on the price actually paid, not the quoted premium: the two differ by
    # the quote/fill drift, and the realized one is what a fill would have cost.
    book["pnl"] = (
        pd.to_numeric(book["realized_pnl"], errors="coerce")
        * pd.to_numeric(book["realized_entry_cost"], errors="coerce")
        * CONTRACT_MULTIPLIER * book["contracts"]
    )
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
         f"  sizing               : {book['sizing'].iloc[0]}",
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
    ap.add_argument("--contracts", type=int, default=None,
                    help="size by a fixed contract count instead of equal dollars")
    ap.add_argument("--capital-per-trade", type=float, default=CAPITAL_PER_TRADE)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    book = build_book(contracts=args.contracts,
                      capital_per_trade=args.capital_per_trade)
    summary = summarize(book)
    print(render(book, summary))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"summary": summary, "book": json.loads(book.to_json(orient="records"))
             if not book.empty else []}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
