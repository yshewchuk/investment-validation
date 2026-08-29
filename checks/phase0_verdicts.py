#!/usr/bin/env python3
"""Reproduce a published verdict number per strategy through the new engine.

    python3 checks/phase0_verdicts.py [--report reports/phase0_verdicts.md]

The Phase 0 exit criteria require that "the engine imports cleanly and
reproduces one known backtest number per strategy from the verdict docs". This
recomputes the EXP-048 mid-fill base-exposure means — the numbers that flipped
all three strategies from NOT-VIABLE to positive, and therefore the numbers the
whole program is predicated on — using only Tier-2 chains and
:class:`engine.fills.FillModel`.

If these three numbers do not come back, either the chain normalization changed
a price or the fill model disagrees with the research convention. Both are
program-stopping, which is why this is a check and not a note.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import paths  # noqa: E402
from engine.data import store  # noqa: E402
from engine.fills import MID, WORST  # noqa: E402

#: ``label -> (trade set, entry-date column)``. The entry column differs per
#: set: S3 opens at T−14, recorded as ``t10_date``.
TRADE_SETS = {
    "S1_calendar": (
        paths.EP_STRATEGIES / "s1_vrp_calendar_straddle" / "data" / "trades_real.csv",
        "entry_date",
    ),
    "S2_short": (
        paths.EP_STRATEGIES / "s2_underpriced_vol" / "data" / "trades_real.csv",
        "entry_date",
    ),
    "S3_long": (
        paths.EP_STRATEGIES / "s3_pre_earnings_long_vol" / "data" / "trades_real_t14.csv",
        "t10_date",
    ),
}

PUBLISHED = paths.EP_OPF / "results" / "exp048_midfill_rerun.json"

#: Tolerance on the reproduced mean, in return units. The published figures are
#: rounded to 4dp and the new pipeline excludes a small number of rows that
#: failed ingestion validation, so exact equality is not the right bar; 0.5pp on
#: a +2.7% to +3.9% effect is.
TOLERANCE = 0.005


def _straddle_legs(trades: pd.DataFrame, date_column: str, label: str) -> pd.DataFrame:
    """The (ticker, obs_date, expiry, strike) rows a trade set needs priced."""
    out = trades[["ticker", date_column, "exit_date", "expiry", "strike"]].copy()
    out = out.rename(columns={date_column: "entry_date"})
    out["trade_ix"] = np.arange(len(out))
    return out


def _price_side(needs: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Merge the needed contracts against the chain store, partition by partition."""
    wanted = needs[["trade_ix", "ticker", date_col, "expiry", "strike"]].rename(
        columns={date_col: "obs_date"}
    )
    years = sorted({int(y) for y in pd.to_datetime(wanted["obs_date"]).dt.year.dropna()})
    collected = []
    for _, chunk in store.iter_table(
        "option_chains",
        years=years,
        columns=["ticker", "obs_date", "expiry", "strike", "right", "bid", "ask"],
    ):
        merged = chunk.merge(wanted, on=["ticker", "obs_date", "expiry", "strike"], how="inner")
        if len(merged):
            collected.append(merged)
    if not collected:
        return pd.DataFrame(columns=["trade_ix", "right", "bid", "ask"])
    return pd.concat(collected, ignore_index=True)


def _straddle_price(priced: pd.DataFrame, fill, opening: bool) -> pd.Series:
    """Straddle price per trade at ``fill``; NaN where a leg is missing."""
    wide = priced.pivot_table(
        index="trade_ix", columns="right", values=["bid", "ask"], aggfunc="first"
    )
    if ("bid", "C") not in wide.columns or ("bid", "P") not in wide.columns:
        return pd.Series(dtype=float)
    total = pd.Series(0.0, index=wide.index)
    for side in ("C", "P"):
        bid = wide[("bid", side)]
        ask = wide[("ask", side)]
        ok = bid.notna() & ask.notna() & (bid <= ask) & (bid >= 0)
        leg = pd.Series(np.nan, index=wide.index)
        if opening:
            leg[ok] = ask[ok] - fill.alpha * (ask[ok] - bid[ok])
        else:
            leg[ok] = bid[ok] + fill.alpha * (ask[ok] - bid[ok])
        total = total + leg
    return total


def reproduce(label: str, path: Path, date_column: str) -> dict:
    if not path.exists():
        return {"strategy": label, "error": f"missing trade set {path}"}
    trades = pd.read_csv(path, dtype={"ticker": str})
    if "exit_mode" in trades.columns:
        # Intrinsic-value exits peek at the settlement price; only chain exits
        # are admissible (a standing rule from the research).
        trades = trades[trades["exit_mode"] == "chain"].reset_index(drop=True)
    for column in (date_column, "exit_date", "expiry"):
        trades[column] = pd.to_datetime(trades[column])

    needs = _straddle_legs(trades, date_column, label)
    entry_rows = _price_side(needs, "entry_date")
    exit_rows = _price_side(needs, "exit_date")

    results = {}
    for fill_label, fill in (("mid", MID), ("worst", WORST)):
        cost = _straddle_price(entry_rows, fill, opening=True)
        value = _straddle_price(exit_rows, fill, opening=False)
        both = cost.index.intersection(value.index)
        cost, value = cost.loc[both], value.loc[both]
        usable = cost.notna() & value.notna() & (cost > 0)
        rets = (value[usable] / cost[usable]) - 1.0
        results[fill_label] = {"n": int(len(rets)), "mean": float(rets.mean()) if len(rets) else float("nan")}
    return {"strategy": label, "trades_in_set": int(len(trades)), **results}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", default=str(paths.REPORTS / "phase0_verdicts.md"))
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args(argv)

    if not PUBLISHED.exists():
        print(f"published results missing: {PUBLISHED}", file=sys.stderr)
        return 1
    published = json.loads(PUBLISHED.read_text())

    rows = []
    ok = True
    print("Reproducing EXP-048 base-exposure means through the engine\n", flush=True)
    for label, (path, date_column) in TRADE_SETS.items():
        result = reproduce(label, path, date_column)
        if "error" in result:
            print(f"  {label}: {result['error']}", flush=True)
            ok = False
            continue
        target = published.get(label, {})
        expected = target.get("mid_mean")
        got = result["mid"]["mean"]
        delta = got - expected if expected is not None else float("nan")
        passed = expected is not None and abs(delta) <= args.tolerance
        ok = ok and passed
        rows.append(
            {
                "strategy": label,
                "published_mid": expected,
                "engine_mid": got,
                "delta": delta,
                "published_n": target.get("n"),
                "engine_n": result["mid"]["n"],
                "engine_worst": result["worst"]["mean"],
                "published_worst": target.get("wc_mean"),
                "passed": passed,
            }
        )
        print(
            f"  {'PASS' if passed else 'FAIL'}  {label:12s} "
            f"engine mid {got:+.4f} vs published {expected:+.4f} "
            f"(delta {delta:+.4f}, n {result['mid']['n']:,} vs {target.get('n'):,})",
            flush=True,
        )

    body = [
        "# Phase 0 — Verdict Reproduction",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "Recomputes the EXP-048 base-exposure means using only Tier-2 chains and "
        "`engine.fills.FillModel`. These are the numbers that flipped all three "
        "strategies from NOT-VIABLE to positive, so reproducing them is what "
        "establishes that the new pipeline did not change a price.",
        "",
        f"Tolerance: {args.tolerance:.4f} on the mean per-trade return.",
        "",
        "| Strategy | Published mid | Engine mid | Delta | Published n | Engine n | Engine worst | Published worst | |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        body.append(
            f"| {row['strategy']} | {row['published_mid']:+.4f} | {row['engine_mid']:+.4f} | "
            f"{row['delta']:+.4f} | {row['published_n']:,} | {row['engine_n']:,} | "
            f"{row['engine_worst']:+.4f} | {row['published_worst']:+.4f} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    body += [
        "",
        "The worst-case column is shown alongside because the gap between the two "
        "is the entire thesis: the same trades, priced at the two ends of the "
        "spread, differ by roughly 20 percentage points per trade.",
        "",
    ]

    report_path = paths.assert_writable(Path(args.report))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(body))
    print(f"\n  report → {report_path}")
    print("\nVERDICT REPRODUCTION: " + ("GREEN" if ok else "RED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
