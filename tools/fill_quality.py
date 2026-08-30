#!/usr/bin/env python3
"""Measured fill quality: real Polygon trades vs ORATS quoted bid/ask.

The mid-fill assumption is the program's verdict flipper, and until this table
existed there was no evidence for it — only the assumption. This tool joins
``option_chains`` (ORATS EOD quotes) with ``option_daily`` (Polygon real traded
bars) on (contract, date) and measures, wherever both exist:

* how far the last real trade and the day's VWAP sit from the quoted mid,
  expressed as an implied fill alpha in [0, 1] (0 = filled at the touch against
  you, 0.5 = mid, 1 = filled at the touch in your favour, outside = through it);
* how that gap behaves as a function of liquidity — trade count, volume, and
  the quoted relative spread — which is the slicing any fill model must
  condition on.

Read-only on Tier 2; writes the joined table and a summary to ``reports/``.

    python3 -m tools.fill_quality                # full window
    python3 -m tools.fill_quality --csv out.csv  # also dump the joined rows

Caveat, stated in the output: the Polygon close is the day's LAST trade and the
VWAP is over the whole day, while the ORATS quote is the EOD quote; on thin
names the two did not necessarily happen at the same instant. That is why the
liquidity slices are printed next to the averages — an alpha measured on one
trade a day is not an alpha you can execute.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from engine import paths
from engine.data import store

__all__ = ["contract_ticker_column", "join_quotes_and_trades", "summarize"]


def contract_ticker_column(chains: pd.DataFrame) -> pd.Series:
    """The OCC id for each chain row, built the same way the pull builds jobs."""
    strike_int = (chains["strike"] * 1000).round().astype("int64")
    return (
        "O:"
        + chains["ticker"].astype(str)
        + chains["expiry"].dt.strftime("%y%m%d")
        + chains["right"].astype(str)
        + strike_int.astype(str).str.zfill(8)
    )


def join_quotes_and_trades() -> pd.DataFrame:
    """Inner join of ORATS quotes and Polygon traded bars on (contract, date)."""
    trades = store.read_table("option_daily")
    if trades.empty:
        return trades
    years = sorted({int(y) for y in trades["obs_date"].dt.year})
    frames = [f for _, f in store.iter_table("option_chains", years=years)]
    quotes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if quotes.empty:
        return pd.DataFrame()

    quotes = quotes.copy()
    quotes["contract_ticker"] = contract_ticker_column(quotes)
    out = trades.merge(
        quotes[["contract_ticker", "obs_date", "bid", "ask", "mid", "spot", "chain_kind"]],
        on=["contract_ticker", "obs_date"],
        how="inner",
        validate="one_to_one",
    )
    out["rel_spread"] = np.where(
        out["mid"] > 0, (out["ask"] - out["bid"]) / out["mid"], np.nan
    )
    width = out["ask"] - out["bid"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["alpha_close"] = np.where(
            width > 0, (out["ask"] - out["close"]) / width, np.nan
        )
        out["alpha_vwap"] = np.where(
            width > 0, (out["ask"] - out["vwap"]) / width, np.nan
        )
    return out


def _bucket(n: pd.Series) -> pd.Series:
    return pd.cut(
        n.fillna(0),
        bins=[-1, 1, 5, 50, np.inf],
        labels=["1", "2-5", "6-50", "50+"],
    )


def summarize(joined: pd.DataFrame) -> pd.DataFrame:
    """Per-liquidity-bucket stats of the measured fill gaps."""
    df = joined.copy()
    df["trades_bucket"] = _bucket(df["n_trades"])
    rows = []
    for bucket, chunk in df.groupby("trades_bucket", observed=True):
        rows.append(
            {
                "n_trades_bucket": str(bucket),
                "days": len(chunk),
                "contracts": int(chunk["contract_ticker"].nunique()),
                "median_alpha_close": round(float(chunk["alpha_close"].median()), 3),
                "median_alpha_vwap": round(float(chunk["alpha_vwap"].median()), 3),
                "pct_close_beyond_worst": round(
                    float(
                        (
                            (chunk["alpha_close"] < 0) | (chunk["alpha_close"] > 1)
                        ).mean()
                    ),
                    3,
                ),
                "median_rel_spread": round(float(chunk["rel_spread"].median()), 3),
                "median_volume": float(chunk["volume"].median()),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=None, help="also write the joined rows here")
    ap.add_argument(
        "--since",
        default=None,
        help="restrict to contract-days on/after this date (e.g. 2026-07-30)",
    )
    args = ap.parse_args(argv)

    joined = join_quotes_and_trades()
    if joined.empty:
        print("no overlap rows — is option_daily populated?", flush=True)
        return 1
    if args.since:
        joined = joined[joined["obs_date"] >= pd.Timestamp(args.since)]
        if joined.empty:
            print(f"no overlap rows on/after {args.since}", flush=True)
            return 1

    table = summarize(joined)
    print(
        f"\nFill quality, measured: {len(joined):,} contract-days with BOTH an "
        f"ORATS quote and a Polygon trade ({joined['contract_ticker'].nunique():,} "
        f"contracts), {int(joined['obs_date'].min().year)}–{int(joined['obs_date'].max().year)}.\n",
        flush=True,
    )
    print(table.to_string(index=False), flush=True)
    print(
        "\nalpha = 0 is a fill at the touch against you, 0.5 is mid, 1 is the "
        "touch in your favour; beyond [0,1] the trade happened THROUGH the "
        "quoted spread. Median over days, bucketed by how many real trades "
        "the contract saw that day.",
        flush=True,
    )

    stamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    out = paths.assert_writable(paths.REPORTS / f"fill_quality_{stamp}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(out, engine="pyarrow", index=False)
    print(f"\njoined rows → {out}", flush=True)
    if args.csv:
        joined.to_csv(args.csv, index=False)
        print(f"csv → {args.csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
