"""Normalize the pre-engine simulated trade sets into Tier-2 ``trades``.

Three trade sets exist from the research that produced the current verdicts
(S1 calendar straddle, S2 short-DTE through the print, S3 T-14 run-up). They
carry different column names for the same concepts and different entry-date
conventions, which is exactly the per-source divergence Tier 2 exists to end.

Two things are preserved rather than smoothed over:

* **``exit_mode``.** The S2 set contains rows priced from a real exit chain and
  rows priced from intrinsic value at expiry. Only ``exit_mode == "chain"`` is
  admissible — the intrinsic fallback peeks at the settlement price and is
  look-ahead biased. Non-chain rows are dropped and counted, never silently
  mixed in.
* **The fill convention.** These sets were priced worst-case (buy the ask, sell
  the bid), so every row lands with ``fill_alpha = 0.0``. Nothing in Tier 2 is
  allowed to carry a P&L number without saying what fill produced it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import paths

__all__ = ["LEGACY_SPECS", "normalize_legacy_set", "normalize_all"]

#: ``strategy -> (path, entry_column, exit_column, legacy_label)``.
#:
#: The strategy codes are the *legacy* research labels (S1/S2/S3), not the three
#: program strategies (CAL-P / STR-THRU / STR-RUNUP). S2 and STR-THRU are
#: structurally similar but were not specified identically, and conflating them
#: here would let an old trade set masquerade as evidence for a spec it never
#: tested — the exact mistake the CAL-P entry in the plan warns about.
LEGACY_SPECS = {
    "S1_calendar": {
        "path": paths.EP_STRATEGIES / "s1_vrp_calendar_straddle" / "data" / "trades_real.csv",
        "entry": "entry_date",
        "exit": "exit_date",
        "variant": "calendar_straddle_20_45dte",
    },
    "S2_short_dte": {
        "path": paths.EP_STRATEGIES / "s2_underpriced_vol" / "data" / "trades_real.csv",
        "entry": "entry_date",
        "exit": "exit_date",
        "variant": "straddle_through_2_10dte",
    },
    "S3_runup": {
        "path": paths.EP_STRATEGIES / "s3_pre_earnings_long_vol" / "data" / "trades_real_t14.csv",
        "entry": "t10_date",
        "exit": "exit_date",
        "variant": "straddle_runup_t14",
    },
}


def normalize_legacy_set(strategy: str, spec: dict | None = None) -> tuple[pd.DataFrame, dict]:
    spec = spec or LEGACY_SPECS[strategy]
    path = Path(spec["path"])
    if not path.exists():
        return pd.DataFrame(), {"strategy": strategy, "reason": f"missing {path}"}

    src = pd.read_csv(path, dtype={"ticker": str})
    rows_in = len(src)

    dropped_non_chain = 0
    if "exit_mode" in src.columns:
        keep = src["exit_mode"] == "chain"
        dropped_non_chain = int((~keep).sum())
        src = src[keep].copy()

    event_date = pd.to_datetime(src["date"], errors="coerce")
    entry_date = pd.to_datetime(src[spec["entry"]], errors="coerce")
    exit_date = pd.to_datetime(src[spec["exit"]], errors="coerce")

    out = pd.DataFrame(
        {
            "kind": "sim",
            "strategy": strategy,
            "variant": spec["variant"],
            "ticker": src["ticker"].astype(str),
            "event_date": event_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "strike": pd.to_numeric(src.get("strike"), errors="coerce"),
            "expiry": pd.to_datetime(src.get("expiry"), errors="coerce"),
            "entry_cost": pd.to_numeric(src.get("cost"), errors="coerce"),
            "exit_value": pd.to_numeric(src.get("exit_val"), errors="coerce"),
            "ret": pd.to_numeric(src.get("ret"), errors="coerce"),
        }
    )
    out["event_id"] = out["ticker"] + "_" + event_date.dt.strftime("%Y-%m-%d")
    out["year"] = event_date.dt.year
    # These sets were built buying the ask and selling the bid.
    out["fill_alpha"] = 0.0
    out["legs"] = None
    out["provenance"] = f"legacy:{path.relative_to(paths.ROOT)}"
    out["trade_id"] = (
        strategy
        + ":"
        + out["ticker"]
        + ":"
        + event_date.dt.strftime("%Y%m%d")
        + ":"
        + out["strike"].map(lambda v: "na" if pd.isna(v) else f"{v:g}")
    )

    out = out.dropna(subset=["year"])
    dupes = int(out.duplicated("trade_id").sum())
    if dupes:
        out = out.drop_duplicates("trade_id", keep="first")

    report = {
        "strategy": strategy,
        "rows_in": rows_in,
        "rows_out": int(len(out)),
        "dropped_non_chain_exit": dropped_non_chain,
        "dropped_duplicate_ids": dupes,
        "mean_ret_worst_fill": (
            round(float(out["ret"].mean()), 4) if out["ret"].notna().any() else None
        ),
    }
    return out, report


def normalize_all() -> tuple[pd.DataFrame, list[dict]]:
    frames, reports = [], []
    for strategy in sorted(LEGACY_SPECS):
        frame, report = normalize_legacy_set(strategy)
        reports.append(report)
        if len(frame):
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), reports
    return pd.concat(frames, ignore_index=True), reports
