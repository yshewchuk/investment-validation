#!/usr/bin/env python3
"""Diagnose a reported equity curve from its transaction log.

    python3 tools/log_diagnostics.py --exp EXP-105
    python3 tools/log_diagnostics.py --log path/to/transactions_x.csv --out reports/x.md

The standard report answers "what did this strategy return". This answers the
question a sceptical reader asks next: **where did that return actually come
from, and does it survive being priced honestly?** Four cuts, all computed from
the log's own quote columns so nothing here depends on the engine agreeing with
itself:

1. **Equal- vs capital-weighted.** The mean counts a $1.10 contract and a $9.60
   contract alike; fixed-fraction sizing buys nine times as many of the former.
2. **Fill sensitivity, re-derived per trade.** Every leg's bid and ask travel in
   the log, so the whole book can be re-priced at any alpha — including the
   capital-weighted breakeven alpha, which is the number a fill assumption
   lives or dies on.
3. **By cost and by quoted width.** An edge concentrated in the cheapest, widest
   markets is an edge in the mid-fill assumption, because mid is only a real
   price where the market is tight enough for mid to mean something.
4. **Concentration.** How many trades make the result.

Emits through ``engine.report`` like everything else, so the answer is a record
with a provenance block rather than a console session.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.report import Report, build_provenance  # noqa: E402

#: Alphas the report re-prices at. 0.5 is mid — the assumption every headline
#: in this program rests on; the rest measure how much of it is slack.
ALPHAS = (1.0, 0.6, 0.5, 0.45, 0.4, 0.3, 0.0)


def leg_columns(log: pd.DataFrame, phase: str) -> list[int]:
    return sorted({int(c.split("_leg")[1].split("_")[0])
                   for c in log.columns if c.startswith(f"{phase}_leg") and c.endswith("_bid")})


def price_at(log: pd.DataFrame, alpha: float, phase: str) -> pd.Series:
    """Re-price one side of every trade at ``alpha``, from the logged quotes.

    alpha interpolates worst -> best: a buy pays the ask at 0 and the bid at 1.
    The side column is honoured per leg, so this is correct for calendars and
    spreads, not only for long straddles.
    """
    total = pd.Series(0.0, index=log.index)
    for leg in leg_columns(log, phase):
        bid = pd.to_numeric(log[f"{phase}_leg{leg}_bid"], errors="coerce")
        ask = pd.to_numeric(log[f"{phase}_leg{leg}_ask"], errors="coerce")
        qty = pd.to_numeric(log.get(f"{phase}_leg{leg}_qty", 1.0), errors="coerce").fillna(1.0)
        side = log.get(f"{phase}_leg{leg}_side", "buy").astype(str)
        buy = side.eq("buy")
        price = np.where(buy, ask - alpha * (ask - bid), bid + alpha * (ask - bid))
        # An entry pays for what it buys; an exit receives for what it sells.
        sign = np.where(buy, 1.0, -1.0) if phase == "entry" else np.where(buy, -1.0, 1.0)
        total = total + pd.Series(sign * qty.to_numpy() * price, index=log.index).fillna(0.0)
    return total


def repriced(log: pd.DataFrame, alpha: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    cost = price_at(log, alpha, "entry")
    value = price_at(log, alpha, "exit")
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = value / cost - 1.0
    return cost, value, ret.replace([np.inf, -np.inf], np.nan)


def capital_weighted(cost: pd.Series, value: pd.Series) -> float:
    ok = cost.notna() & value.notna() & (cost > 0)
    return float((value[ok] - cost[ok]).sum() / cost[ok].sum()) if ok.any() else float("nan")


def breakeven_alpha(log: pd.DataFrame, lo: float = 0.0, hi: float = 1.0) -> float | None:
    """The alpha at which the book breaks even on capital, by bisection."""
    def net(a: float) -> float:
        cost, value, _ = repriced(log, a)
        ok = cost.notna() & value.notna()
        return float((value[ok] - cost[ok]).sum())

    if net(lo) > 0 or net(hi) < 0:
        return None
    for _ in range(50):
        mid = (lo + hi) / 2
        if net(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _pct(x, nd=2, signed=True) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:+.{nd}f}%" if signed else f"{x * 100:.{nd}f}%"


def fill_table(log: pd.DataFrame) -> dict:
    rows = []
    labels = {1.0: "best — buy bid, sell ask", 0.6: "better than mid", 0.5: "MID (the headline)",
              0.45: "give up 5% of the spread", 0.4: "capture 40% of the spread",
              0.3: "capture 30%", 0.0: "worst — buy ask, sell bid"}
    for alpha in ALPHAS:
        cost, value, ret = repriced(log, alpha)
        rows.append([f"{alpha:.2f}", labels.get(alpha, ""), f"{len(ret.dropna()):,}",
                     _pct(ret.mean()), _pct(capital_weighted(cost, value)),
                     _pct(ret.median()), _pct(float((ret > 0).mean()), nd=1, signed=False)])
    return {"title": "The same book at every fill assumption",
            "note": "Re-priced per trade from the log's own leg quotes. The headline "
                    "sits on the MID row; everything below it is what a worse fill costs.",
            "columns": ["alpha", "meaning", "n", "equal-weighted", "capital-weighted",
                        "median", "win rate"],
            "align": ["---", "---", "---:", "---:", "---:", "---:", "---:"],
            "rows": rows,
            "falsifies": "measured live fill quality (Phase 5 alpha-hat) landing below "
                         "the capital-weighted breakeven alpha."}


def bucket_table(log: pd.DataFrame, by: pd.Series, title: str, note: str,
                 label: str, falsifies: str) -> dict:
    cost, value, ret = repriced(log, 0.5)
    frame = pd.DataFrame({"by": by, "cost": cost, "value": value, "ret": ret}).dropna()
    if len(frame) < 25 or frame["by"].nunique() < 5:
        return {"title": title, "body": ["Too few distinct values to bucket."]}
    frame["q"] = pd.qcut(frame["by"], 5, labels=False, duplicates="drop")
    net_dollars = (frame["value"] - frame["cost"]).sum()
    net_returns = frame["ret"].sum()
    rows = []
    for q, g in frame.groupby("q"):
        rows.append([
            f"Q{int(q) + 1}", f"{len(g):,}", f"{g['by'].median():.3f}",
            _pct(g["ret"].mean()), _pct(capital_weighted(g["cost"], g["value"])),
            _pct(g["ret"].median()),
            _pct(float(g["ret"].sum() / net_returns), nd=1) if net_returns else "n/a",
            _pct(float((g["value"] - g["cost"]).sum() / net_dollars), nd=1) if net_dollars else "n/a",
        ])
    return {"title": title, "note": note + (
        "\n\nTwo shares are given because they answer different questions. **Share of "
        "Σ returns** is what drives the plotted curve: fixed-fraction sizing gives every "
        "trade the same weight in percentage terms, so the curve compounds this column. "
        "**Share of net $** is what a fixed-notional book would have earned. When they "
        "disagree, the curve is being carried by trades that contribute little capital."),
            "columns": ["quintile", "n", f"median {label}", "equal-weighted",
                        "capital-weighted", "median return", "share of Σ returns",
                        "share of net $"],
            "align": ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
            "rows": rows, "falsifies": falsifies}


def concentration_table(log: pd.DataFrame) -> dict:
    cost, value, ret = repriced(log, 0.5)
    pnl = (value - cost).dropna()
    net = pnl.sum()
    ordered = pnl.reindex(pnl.abs().sort_values(ascending=False).index)
    rows = [[f"top {k}", f"{ordered.head(k).sum() / net:.1%}" if net else "n/a"]
            for k in (1, 5, 10, 25, 50, 100) if k <= len(ordered)]
    return {"title": "How many trades make the result",
            "note": "Net dollar P&L contributed by the largest positions, at mid.",
            "columns": ["trades", "share of net $ P&L"], "align": ["---", "---:"],
            "rows": rows,
            "falsifies": "the result surviving the removal of its largest few trades."}


def build_sections(log: pd.DataFrame, split_year: int | None) -> list[dict]:
    cost, value, ret = repriced(log, 0.5)
    be = breakeven_alpha(log)
    rel = pd.Series(0.0, index=log.index)
    for leg in leg_columns(log, "entry"):
        rel = rel + (pd.to_numeric(log[f"entry_leg{leg}_ask"], errors="coerce")
                     - pd.to_numeric(log[f"entry_leg{leg}_bid"], errors="coerce"))
    rel = rel / cost.replace(0, np.nan)

    headline = {
        "title": "Headline, two ways",
        "columns": ["reading", "value", "what it means"],
        "align": ["---", "---:", "---"],
        "rows": [
            ["equal-weighted mean", _pct(ret.mean()), "what the average TRADE returned"],
            ["capital-weighted", _pct(capital_weighted(cost, value)),
             "what the average DOLLAR returned"],
            ["median trade", _pct(ret.median()),
             "what a typical trade did — the premium showing through"],
            ["win rate", _pct(float((ret > 0).mean()), nd=1, signed=False), ""],
            ["capital-weighted breakeven alpha",
             f"{be:.3f}" if be is not None else "n/a",
             "mid is 0.500; the gap is the whole margin of safety"],
        ],
        "promote_to_verdict": True,
        "verdict_row": (
            "Does the average dollar earn what the average trade does?",
            f"**{'Yes' if abs(ret.mean() - capital_weighted(cost, value)) <= 0.01 else 'No'}** "
            f"— equal-weighted {_pct(ret.mean())} against capital-weighted "
            f"{_pct(capital_weighted(cost, value))}", ""),
    }

    sections = [headline, fill_table(log)]

    if split_year is not None and "event_date" in log.columns:
        years = pd.to_datetime(log["event_date"]).dt.year
        rows = []
        for name, mask in (("ungated (before the gate trains)", years < split_year),
                           ("gated", years >= split_year)):
            part = log[mask]
            if not len(part):
                continue
            c, v, r = repriced(part, 0.5)
            part_be = breakeven_alpha(part)
            rows.append([name, f"{len(part):,}", _pct(r.mean()),
                         _pct(capital_weighted(c, v)), _pct(r.median()),
                         f"{part_be:.3f}" if part_be is not None else "n/a"])
        sections.append({
            "title": f"Gated vs ungated (gate trains from {split_year})",
            "note": "In an ungated fold every row is kept, so those years are the base "
                    "exposure rather than a gate result. Reported separately because a "
                    "headline that blends them describes neither.",
            "columns": ["segment", "n", "equal-weighted", "capital-weighted", "median",
                        "breakeven alpha"],
            "align": ["---", "---:", "---:", "---:", "---:", "---:"],
            "rows": rows,
            "falsifies": "the gated segment failing to beat the ungated one on "
                         "capital-weighted return AND breakeven alpha."})

    sections.append(bucket_table(
        log, cost, "By what the trade cost",
        "Fixed-fraction sizing buys the most contracts of the cheapest structures, so an "
        "edge that lives in Q1 is bought in size precisely where size is hardest to get.",
        "cost", "the cheapest quintile carrying the result while the dearest does not."))
    sections.append(bucket_table(
        log, rel, "By how wide the market was quoted",
        "Relative spread = summed entry leg spreads / entry cost. Mid is a real price "
        "only where this is small.",
        "rel. spread", "the widest quintile carrying the result — that is an edge in the "
        "fill assumption, not in the trade."))
    sections.append(concentration_table(log))
    return sections


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", help="experiment id prefix, e.g. EXP-105")
    ap.add_argument("--log", help="path to a transactions_*.csv")
    ap.add_argument("--split-year", type=int, default=None,
                    help="first gated year; splits the report into gated/ungated")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.log:
        log_path = Path(args.log)
        exp_id = log_path.parent.parent.name.split("_")[0]
    else:
        if not args.exp:
            ap.error("pass --exp or --log")
        matches = sorted((ROOT / "experiments").glob(f"{args.exp}*/results/transactions_*.csv"))
        if not matches:
            ap.error(f"no transaction log for {args.exp}; run the experiment first")
        log_path, exp_id = matches[0], args.exp

    log = pd.read_csv(log_path)
    split_year = args.split_year
    if split_year is None:
        metrics = sorted(log_path.parent.glob("metrics_*.json"))
        if metrics:
            diagnostics = json.loads(metrics[0].read_text()).get("walk_forward", {}).get("diagnostics", [])
            gated = [int(d["year"]) for d in diagnostics if not d.get("ungated")]
            split_year = min(gated) if gated else None

    out = Path(args.out) if args.out else paths.REPORTS / f"{exp_id.lower()}_log_diagnostics.md"
    out = paths.assert_writable(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    context = {
        "kind": "audit",
        "spec": {"id": f"{exp_id}-DIAG", "type": "descriptive",
                 "title": f"Where {exp_id}'s reported return comes from",
                 "hypothesis": "The reported curve is a trade edge rather than an "
                               "artifact of equal-weighting, cheap contracts, or the "
                               "mid-fill assumption on wide markets."},
        "results": {"headline": {}, "stress": {}, "mc": {}},
        "headline": {}, "backtest": {}, "checklist": [],
        "provenance": build_provenance(seeds={}, input_files=[log_path]),
        "survivorship_note": "", "calibration": None,
        "funnel": [{"stage": "trades in the plotted curve", "events": int(len(log)),
                    "note": f"from {log_path.name}", "headline": True}],
        "extra_sections": build_sections(log, split_year),
    }
    Report(context).write(out.parent, filename=out.name)
    print(f"wrote {out} ({out.stat().st_size:,} bytes) from {log_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
