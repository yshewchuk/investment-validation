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

**The contrarian book.** `build_book(include_declined=True)` adds every row the
gate looked at and REJECTED (`gate_pass is False`) — never the withheld ones —
tagged `recommended=False`, priced and sized exactly like a real trade would
have been. It answers "what did we pass on", the mirror question to the book
itself: a gate earning its keep should show a positive book and a flat-to-
negative contrarian one, on the SAME pricing path, so the comparison is not
two reports that might quietly disagree on method.

Splitting by strategy or by recommended/declined is one table, filtered —
`book["strategy"]` and `book["recommended"]` carry both dimensions already;
nothing about the underlying trade population changes and there is no reason
to build three separate books when a reader can filter one.

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


#: Columns pulled from `ledger.read_outcomes()` into the book. `row_id` is the
#: join key; the rest are populated only on a resolved outcome and stay NaN/None
#: on an unresolvable or not-yet-settled one — see the reindex note in
#: :func:`build_book`.
OUTCOME_MERGE_COLUMNS = (
    "row_id", "status", "realized_pnl", "realized_entry_cost",
    "realized_exit_value", "reason",
)


def build_book(contracts: int | None = None,
               capital_per_trade: float = CAPITAL_PER_TRADE,
               *, include_declined: bool = False) -> pd.DataFrame:
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

    # `gate_pass` is True (recommended), False (the gate looked and declined)
    # or None (the gate never got to decide — out-of-domain, missing features,
    # a disabled structure). None is neither a recommendation nor a rejection
    # and belongs in neither book: including it in the contrarian side would
    # answer "what did withheld rows do", not "what did the gate say no to".
    wanted = [True, False] if include_declined else [True]
    rec = preds[preds["gate_pass"].isin(wanted)].copy()
    if rec.empty:
        return pd.DataFrame()
    book = rec.sort_values("as_of").reset_index(drop=True)
    book["recommended"] = book["gate_pass"] == True  # noqa: E712 — explicit, not falsy-None

    outcomes = pd.DataFrame(ledger.read_outcomes())
    # `realized_entry_cost` / `realized_exit_value` are written only on a
    # RESOLVED outcome row (engine.ledger.score_outcomes) — an unresolvable one
    # has nothing to price yet. Every outcome on file can legitimately be
    # unresolvable at once (a freshly reset ledger, or simply no trade has had
    # its exit session settle yet), and a DataFrame built from JSON records
    # never gets a column no row supplied. Reindexing onto the full column set
    # before selecting keeps the merge from raising in exactly that gap.
    outcomes = outcomes.reindex(columns=list(OUTCOME_MERGE_COLUMNS))
    if not outcomes.empty:
        outcomes = outcomes.drop_duplicates("row_id", keep="last")
        book = book.merge(outcomes[list(OUTCOME_MERGE_COLUMNS)],
                          on="row_id", how="left")
    else:
        for c in OUTCOME_MERGE_COLUMNS[1:]:
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


def _summarize_settled(settled: pd.DataFrame) -> dict[str, Any]:
    """P&L stats for one settled population — the shared arithmetic behind
    both the top-level summary and each `recommended`/`declined` split."""
    out: dict[str, Any] = {
        "n_settled": int(len(settled)),
        "pnl": float(settled["pnl"].sum(skipna=True)) if len(settled) else 0.0,
        "capital_settled": float(settled["capital"].sum(skipna=True)) if len(settled) else 0.0,
    }
    if not len(settled):
        return out
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


def summarize(book: pd.DataFrame) -> dict[str, Any]:
    """Stats over `book`. Top-level numbers are the RECOMMENDED population —
    unchanged shape for every existing caller.

    When `book` also carries declined rows (`build_book(include_declined=True)`
    was used), `by_recommended` gives the same breakdown split by
    `recommended` — the book and its contrarian counterpart side by side, so
    "the gate adds value" is a comparison you can see in one summary rather
    than two separately-run reports that might drift apart in method.
    """
    if book.empty:
        return {"n_recommended": 0}
    settled = book[book["state"] == "settled"]
    out: dict[str, Any] = {
        "n_recommended": int(len(book)),
        "by_state": {k: int(v) for k, v in book["state"].value_counts().items()},
        "capital_committed": float(book["capital"].sum(skipna=True)),
    }
    out.update(_summarize_settled(settled))
    if "recommended" in book.columns and book["recommended"].nunique() > 1:
        out["by_recommended"] = {
            ("recommended" if flag else "declined"): {
                "n_recommended": int(len(pop)),
                "capital_committed": float(pop["capital"].sum(skipna=True)),
                **_summarize_settled(pop[pop["state"] == "settled"]),
            }
            for flag, pop in book.groupby("recommended")
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
    by_rec = summary.get("by_recommended")
    if by_rec:
        L += ["  RECOMMENDED vs DECLINED — the gate's own contrarian check:", ""]
        for label in ("recommended", "declined"):
            block = by_rec.get(label)
            if not block:
                continue
            L.append(f"    {label.upper()} (n={block['n_recommended']}, "
                     f"{block.get('n_settled', 0)} settled):")
            if block.get("n_settled"):
                L.append(f"      P&L ${block['pnl']:+,.0f}  "
                         f"mean {block['mean_trade_return']:+.2%}  "
                         f"win {block['win_rate']:.1%}")
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
    ap.add_argument("--include-declined", action="store_true",
                    help="also book the gate's rejections, tagged recommended=False — "
                         "the contrarian check")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    book = build_book(contracts=args.contracts,
                      capital_per_trade=args.capital_per_trade,
                      include_declined=args.include_declined)
    summary = summarize(book)
    print(render(book, summary))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"summary": summary, "book": json.loads(book.to_json(orient="records"))
             if not book.empty else []}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
