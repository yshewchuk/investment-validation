"""Replay output shaping — the per-trade record and the Tier-2 trades table.

Split out of ``engine.v2.research.replay`` (review blocker: module fan-out)
with bodies unchanged. ``_trade_record`` is the record one priced (event,
alpha) produces; ``to_trades_table`` is the Tier-2 handoff that turns replay
results into the ``trades`` schema.
"""
from __future__ import annotations

import json
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["legs_exit_spot", "legs_spot_dte", "to_trades_table"]


def _trade_record(
    plan_row: Mapping,
    ticker: str,
    decision_date,
    alpha: float,
    quoted_cost: float,
    spot_decision: float,
    dte_decision: int | None,
    result: Mapping,
    entry,
    exit_,
    entry_rows: pd.DataFrame,
    exit_rows: pd.DataFrame,
    include_legs: bool,
) -> dict:
    """Stage 4: the result record for one priced (event, alpha)."""
    return {
        "event_id": plan_row["event_id"],
        "ticker": ticker,
        "event_date": plan_row["event_date"],
        "session": plan_row["session"],
        "decision_date": decision_date,
        "entry_date": plan_row["entry_date"],
        "exit_date": plan_row["exit_date"],
        "fill_alpha": float(alpha),
        #: What the board would have QUOTED at the decision close, mid.
        #: NaN when the decision is the entry, where the two are the
        #: same number by construction.
        "quoted_cost": quoted_cost,
        "spot_decision": spot_decision,
        "dte_decision": dte_decision,
        "entry_cost": result["cost"],
        "exit_value": result["exit_value"],
        "pnl": result["pnl"],
        "ret": result["ret"],
        "spot_entry": entry.spot,
        "spot_exit": exit_.spot,
        **({"entry_legs": [
            {"name": leg.name, "right": leg.right, "side": leg.side,
             "qty": leg.qty, "strike": leg.strike, "expiry": leg.expiry,
             "bid": leg.bid, "ask": leg.ask, "price": leg.price}
            for leg in entry.legs
        ]} if include_legs else {}),
        "strike": entry.legs[0].strike,
        "expiry": entry.legs[0].expiry,
        "dte_entry": int(entry.legs[0].dte),
        "n_legs": len(entry.legs),
        "wide_market": entry.any_wide_market or exit_.any_wide_market,
        "quote_repaired": bool(
            entry_rows.get("quote_repaired", pd.Series(dtype=bool)).any()
            or exit_rows.get("quote_repaired", pd.Series(dtype=bool)).any()
        ),
        # The Tier-2 schema has no column for spot, and every consumer
        # that quotes a value per unit of spot (the payoff fit, the
        # moneyness bucket) needs the one the trade was actually priced
        # against — not a spot re-read later from a different table.
        "legs": json.dumps(
            {
                "spot_entry": entry.spot,
                "spot_exit": exit_.spot,
                "dte_entry": int(entry.legs[0].dte),
                "entry": entry.to_dict()["legs"],
                "exit": exit_.to_dict()["legs"],
            },
            default=str,
        ),
    }


def legs_spot_dte(trades: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Recover entry spot and DTE from the stored ``legs`` blob."""
    spots = np.full(len(trades), np.nan)
    dtes = np.full(len(trades), np.nan)
    if "legs" not in trades.columns:
        return pd.Series(spots, index=trades.index), pd.Series(dtes, index=trades.index)

    for i, blob in enumerate(trades["legs"].to_numpy()):
        if not isinstance(blob, str):
            continue
        try:
            doc = json.loads(blob)
        except ValueError:
            continue
        if not isinstance(doc, dict):
            continue
        spots[i] = _as_float(doc.get("spot_entry"))
        dte = doc.get("dte_entry")
        if dte is None:
            legs = doc.get("entry") or []
            dte = legs[0].get("dte") if legs else None
        dtes[i] = _as_float(dte)
    return pd.Series(spots, index=trades.index), pd.Series(dtes, index=trades.index)


def legs_exit_spot(trades: pd.DataFrame) -> pd.Series:
    """Recover the exit spot stored beside the pinned exit legs."""
    spots = np.full(len(trades), np.nan)
    if "legs" not in trades.columns:
        return pd.Series(spots, index=trades.index)
    for index, blob in enumerate(trades["legs"].to_numpy()):
        if not isinstance(blob, str):
            continue
        try:
            document = json.loads(blob)
        except ValueError:
            continue
        if isinstance(document, dict):
            spots[index] = _as_float(document.get("spot_exit"))
    return pd.Series(spots, index=trades.index)


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def to_trades_table(results: Sequence) -> pd.DataFrame:
    """Shape replay output into the Tier-2 ``trades`` schema.

    ``trade_id`` carries the alpha, because the schema's primary key is the
    trade id and one event priced at five alphas is five rows.
    """
    frames = []
    for result in results:
        if not len(result.trades):
            continue
        trades = result.trades
        out = pd.DataFrame(
            {
                "trade_id": (
                    trades["strategy"] + ":" + trades["variant"] + ":"
                    + trades["ticker"] + ":"
                    + pd.to_datetime(trades["event_date"]).dt.strftime("%Y%m%d") + ":a"
                    + (trades["fill_alpha"].astype(float) * 100).round().astype(int).astype(str)
                ),
                "kind": "sim",
                "strategy": trades["strategy"],
                "variant": trades["variant"],
                "ticker": trades["ticker"],
                "event_id": trades["event_id"],
                "event_date": pd.to_datetime(trades["event_date"]),
                "year": pd.to_datetime(trades["event_date"]).dt.year,
                "legs": trades["legs"],
                "entry_date": pd.to_datetime(trades["entry_date"]),
                "exit_date": pd.to_datetime(trades["exit_date"]),
                "strike": trades["strike"].astype(float),
                "expiry": pd.to_datetime(trades["expiry"]),
                "fill_alpha": trades["fill_alpha"].astype(float),
                "entry_cost": trades["entry_cost"].astype(float),
                "exit_value": trades["exit_value"].astype(float),
                "ret": trades["ret"].astype(float),
                "provenance": "engine.replay",
            }
        )
        frames.append(out)
    if not frames:
        columns = [
            "trade_id", "kind", "strategy", "variant", "ticker", "event_id",
            "event_date", "year", "legs", "entry_date", "exit_date", "strike",
            "expiry", "fill_alpha", "entry_cost", "exit_value", "ret", "provenance",
        ]
        return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})
    return pd.concat(frames, ignore_index=True)
