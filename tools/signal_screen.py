#!/usr/bin/env python3
"""Hypothesis-generation screen for cross-sectional long-put signals.

Screens a family of candidate entry rules on what a long put profits from:
forward spot returns AND forward IV change (a cheap-IV entry can win even on a
flat stock if IV mean-reverts up). Everything comes from Tier-2
``daily_market`` — zero API calls.

This is discovery, not evidence: any candidate taken forward must be
pre-registered as an experiment (spec.yaml + ledger) and pass walk-forward OOS
before it means anything. The screen exists to pick WHICH hypotheses are worth
spending that discipline on, and to size them (signal frequency, base rates).

    python3 -m tools.signal_screen

Outputs: printed table + ``reports/signal_screen_YYYY-MM-DD.{md,parquet}``.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from engine import paths
from engine.data import store

#: Rolling window for history percentiles and the 52-week extremes.
WINDOW = 252
#: Minimum history before a ticker-day can be scored.
MIN_HIST = 300
#: Forward horizons measured after the signal day (close of t).
HORIZONS = (5, 20)

#: (name, description, feature, threshold, direction)
#: direction 'ge'/'le' applies the threshold to the feature column.
SIGNALS = {
    "A_near_high_low_iv": "user hypothesis: spot>=97% of 52w high AND iv30 <= 20th trailing pct",
    "A2_at_high_deep_low_iv": "sharpened: spot within 0.5% of 52w high AND iv30 <= 10th pct",
    "B_high_rvol_extended": "overheated: spot>=97% of high AND realized vol >= 80th pct",
    "C_deep_low_iv_alone": "complacency alone: iv30 <= 5th trailing pct, no price condition",
    "D_near_low_high_iv": "falling knife: spot<=103% of 52w low AND iv30 >= 80th pct",
    "E_inverted_term_extended": "extended + term structure inverted (fwd90_30 < 100)",
    "F_low_skew": "protection cheap: skew <= 10th trailing pct (convention checked in output)",
    "F2_high_skew": "protection rich: skew >= 90th trailing pct (control for F)",
}


def build_features() -> pd.DataFrame:
    """One row per (ticker, date) with the screen's features and forward outcomes."""
    cols = ["ticker", "date", "spot", "iv30", "rvol30", "skew", "fwd90_30", "mcap_usd"]
    frames = [f for _, f in store.iter_table("daily_market", columns=cols)]
    dm = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    print(f"  daily_market: {len(dm):,} rows, {dm['ticker'].nunique():,} tickers", flush=True)

    out = []
    t0 = time.time()
    groups = dm.groupby("ticker", sort=True)
    n_groups = len(groups)
    for i, (t, g) in enumerate(groups, 1):
        g = g.dropna(subset=["spot"]).reset_index(drop=True)
        if len(g) < MIN_HIST:
            continue
        spot = g["spot"]
        g["dist_hi"] = spot / spot.rolling(WINDOW, min_periods=WINDOW).max()
        g["dist_lo"] = spot / spot.rolling(WINDOW, min_periods=WINDOW).min()
        for src, dst in (("iv30", "iv30_pct"), ("rvol30", "rvol_pct"), ("skew", "skew_pct")):
            s = g[src]
            g[dst] = s.rolling(WINDOW, min_periods=WINDOW).rank(pct=True)
        for h in HORIZONS:
            g[f"fwd_ret_{h}"] = spot.shift(-h) / spot - 1.0
            g[f"fwd_iv_chg_{h}"] = g["iv30"].shift(-h) - g["iv30"]
        g["year"] = g["date"].dt.year
        out.append(g)
        if i % 300 == 0 or i == n_groups:
            print(
                f"  [features] {i}/{n_groups} tickers, "
                f"{time.time()-t0:.0f}s elapsed",
                flush=True,
            )
    feats = pd.concat(out, ignore_index=True)
    feats["mcap_bucket"] = pd.cut(
        feats["mcap_usd"],
        bins=[0, 1e9, 10e9, np.inf],
        labels=["<1B", "1-10B", ">10B"],
    )
    return feats


def signal_masks(feats: pd.DataFrame) -> dict[str, pd.Series]:
    ok = feats["iv30_pct"].notna()  # implies the rolling window is full
    return {
        "A_near_high_low_iv": ok & (feats["dist_hi"] >= 0.97) & (feats["iv30_pct"] <= 0.20),
        "A2_at_high_deep_low_iv": ok & (feats["dist_hi"] >= 0.995) & (feats["iv30_pct"] <= 0.10),
        "B_high_rvol_extended": ok & feats["rvol_pct"].notna()
        & (feats["dist_hi"] >= 0.97) & (feats["rvol_pct"] >= 0.80),
        "C_deep_low_iv_alone": ok & (feats["iv30_pct"] <= 0.05),
        "D_near_low_high_iv": ok & (feats["dist_lo"] <= 1.03) & (feats["iv30_pct"] >= 0.80),
        "E_inverted_term_extended": ok & (feats["dist_hi"] >= 0.97)
        & feats["fwd90_30"].notna() & (feats["fwd90_30"] < 100),
        "F_low_skew": ok & feats["skew_pct"].notna() & (feats["skew_pct"] <= 0.10),
        "F2_high_skew": ok & feats["skew_pct"].notna() & (feats["skew_pct"] >= 0.90),
    }


def screen(feats: pd.DataFrame) -> pd.DataFrame:
    base = feats[feats["fwd_ret_20"].notna()]
    rows = [
        {
            "signal": "BASE_RATE (unconditional)",
            "n": len(base),
            "mean_fwd_ret_20": base["fwd_ret_20"].mean(),
            "p_down_20d": (base["fwd_ret_20"] < 0).mean(),
            "p_down_5pct_20d": (base["fwd_ret_20"] <= -0.05).mean(),
            "mean_fwd_iv_chg_20": base["fwd_iv_chg_20"].mean(),
            "p_iv_up_3pts_20d": (base["fwd_iv_chg_20"] >= 3.0).mean(),
        }
    ]
    masks = signal_masks(feats)
    for name, mask in masks.items():
        sub = feats[mask & feats["fwd_ret_20"].notna()]
        if len(sub) < 30:
            continue
        yearly = sub.groupby("year")["fwd_ret_20"].mean()
        rows.append(
            {
                "signal": name,
                "n": len(sub),
                "mean_fwd_ret_20": sub["fwd_ret_20"].mean(),
                "p_down_20d": (sub["fwd_ret_20"] < 0).mean(),
                "p_down_5pct_20d": (sub["fwd_ret_20"] <= -0.05).mean(),
                "mean_fwd_iv_chg_20": sub["fwd_iv_chg_20"].mean(),
                "p_iv_up_3pts_20d": (sub["fwd_iv_chg_20"] >= 3.0).mean(),
                "years_neg_ret": int((yearly < 0).sum()),
                "years_total": int(len(yearly)),
            }
        )
    table = pd.DataFrame(rows)
    for col in ("mean_fwd_ret_20", "p_down_20d", "p_down_5pct_20d", "mean_fwd_iv_chg_20", "p_iv_up_3pts_20d"):
        table[col] = table[col].astype(float).round(4)
    return table


def mcap_slice(feats: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    masks = signal_masks(feats)
    rows = []
    for name in names:
        for bucket in ("<1B", "1-10B", ">10B"):
            sub = feats[masks[name] & feats["fwd_ret_20"].notna() & (feats["mcap_bucket"] == bucket)]
            if len(sub) < 30:
                continue
            rows.append(
                {
                    "signal": name,
                    "mcap": bucket,
                    "n": len(sub),
                    "mean_fwd_ret_20": round(float(sub["fwd_ret_20"].mean()), 4),
                    "p_down_20d": round(float((sub["fwd_ret_20"] < 0).mean()), 4),
                    "mean_fwd_iv_chg_20": round(float(sub["fwd_iv_chg_20"].mean()), 4),
                }
            )
    return pd.DataFrame(rows)


def _md(table: pd.DataFrame) -> str:
    try:
        return table.to_markdown(index=False)
    except ImportError:
        return "```\n" + table.to_string(index=False) + "\n```"


def main() -> int:
    print("signal screen: building features from daily_market …", flush=True)
    feats = build_features()
    table = screen(feats)
    slices = mcap_slice(feats, ("A_near_high_low_iv", "A2_at_high_deep_low_iv"))

    skew = feats["skew"].describe().round(3)
    print("\n== candidate signals vs base rate (fwd 20 trading days) ==\n", flush=True)
    print(table.to_string(index=False), flush=True)
    print("\n== mcap slices (A / A2) ==\n", flush=True)
    print(slices.to_string(index=False), flush=True)
    print(f"\nskew field distribution (for reading F/F2):\n{skew}", flush=True)

    stamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    reports = paths.assert_writable(paths.REPORTS)
    reports.mkdir(parents=True, exist_ok=True)
    feats_tables = table.copy()
    feats_tables.to_parquet(reports / f"signal_screen_{stamp}.parquet", index=False)
    md = [
        f"# Signal screen — {stamp}",
        "",
        "Hypothesis generation for cross-sectional long-put entries. Discovery",
        "only: nothing here is pre-registered or walk-forward; any candidate",
        "taken forward goes through the experiment scaffold first.",
        "",
        "Features from Tier-2 daily_market only; forward windows are 5/20",
        f"trading days; history window {WINDOW}. Survivorship caveat: the",
        "universe is current-listed names.",
        "",
        "## Candidates vs base rate",
        "",
        _md(table),
        "",
        "## Mcap slices",
        "",
        _md(slices),
        "",
        "## Signal definitions",
        "",
        *[f"- **{k}** — {v}" for k, v in SIGNALS.items()],
        "",
    ]
    (reports / f"signal_screen_{stamp}.md").write_text("\n".join(md))
    print(f"\nreport → reports/signal_screen_{stamp}.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
