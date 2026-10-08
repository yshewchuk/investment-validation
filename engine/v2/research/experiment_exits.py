"""Fixed-day exits over held contracts and one pinned snapshot; no publication."""
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from math import isfinite

import pandas as pd

from engine.v2.data.errors import fail
from engine.v2.research._chains import load_chain_index
from engine.v2.research._pricing import FillModel, trading_calendar_from_snapshot


@dataclass(frozen=True)
class PositionLeg:
    expiry: str
    strike: float
    right: str
    side: str
    qty: float


@dataclass(frozen=True)
class EnteredPosition:
    trade_id: str
    ticker: str
    entry_date: str
    legs: tuple[PositionLeg, ...]


@dataclass(frozen=True)
class ExitDecision:
    trade_id: str
    exit_date: str
    entry_cost: float
    exit_value: float
    pnl: float
    visited_dates: tuple[str, ...]
    snapshot_id: str
    fill_alpha: float
    reason: str = "fixed_day"
    pnl_basis: str = "mark_based"
    mark_source: str = "option_chains"
    fill_convention: str = "alpha_ladder"


def _session(value):
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("noncanonical session")
    return pd.Timestamp(parsed)


def _validate_position(position):
    if (not isinstance(position.trade_id, str) or not position.trade_id.strip()
            or not isinstance(position.ticker, str) or not position.ticker.strip()
            or not isinstance(position.legs, tuple) or not position.legs):
        raise ValueError("invalid entered position")
    contracts = set()
    for leg in position.legs:
        key = (_session(leg.expiry), leg.strike, leg.right)
        if (leg.side not in ("buy", "sell") or leg.right not in ("C", "P")
                or not isfinite(leg.qty) or leg.qty <= 0
                or not isfinite(leg.strike) or leg.strike <= 0 or key in contracts):
            raise ValueError("invalid held leg")
        contracts.add(key)


def _cash_flow(position, rows, fill, *, closing):
    if rows is None:
        raise ValueError("missing daily chain")
    total = 0.0
    for leg in position.legs:
        hits = rows[(pd.to_datetime(rows["expiry"]) == _session(leg.expiry))
                    & (rows["strike"] == leg.strike) & (rows["right"] == leg.right)]
        if len(hits) != 1:
            raise ValueError("missing or ambiguous held contract")
        bid, ask = float(hits.iloc[0]["bid"]), float(hits.iloc[0]["ask"])
        if not isfinite(bid) or not isfinite(ask):
            raise ValueError("nonfinite required mark")
        side = {"buy": "sell", "sell": "buy"}[leg.side] if closing else leg.side
        total += float(fill.cash_flow(side, bid, ask, leg.qty))
    if not isfinite(total):
        raise ValueError("nonfinite position cash flow")
    return total


def _walk_position(repository, snapshot, position, calendar, days, fill):
    session = position.entry_date
    try:
        _validate_position(position)
        start = calendar.index_of(_session(session))
        dates = calendar.days[start:start + days + 1]
        if len(dates) != days + 1:
            raise ValueError("insufficient observed calendar")
        index = load_chain_index(repository, snapshot, [(position.ticker, d) for d in dates])
        flows = []
        for number, day in enumerate(dates):
            session = day.date().isoformat()
            if any(day > _session(leg.expiry) for leg in position.legs):
                raise ValueError("required mark past expiry")
            flows.append(_cash_flow(position, index.get(position.ticker, day), fill,
                                    closing=number > 0))
        pnl = flows[0] + flows[-1]
        if not isfinite(pnl):
            raise ValueError("nonfinite P&L")
    except (ValueError, TypeError, KeyError, OverflowError):
        raise fail("EXPERIMENT_VARIANT_FAILED", "required daily position mark unavailable",
                   details={"trade_id": position.trade_id, "session": session}) from None
    return ExitDecision(position.trade_id, session, -flows[0], flows[-1], pnl,
                        tuple(d.date().isoformat() for d in dates), snapshot.snapshot_id,
                        float(fill.alpha))


def walk_fixed_day(repository, snapshot, positions, *, economic_params):
    """Return all decisions or raise; never skip a trade or return partial results.

    Pass ``resolved_plan.economic_params`` directly. The caller supplies already
    selected held contracts, never a priced trade table or replacement selector.
    """
    try:
        recipe = economic_params["exit"]
        alpha = economic_params["fill"]
        if (not isinstance(recipe, Mapping) or set(recipe) != {"kind", "trading_days"}
                or recipe["kind"] != "fixed_day" or type(recipe["trading_days"]) is not int
                or recipe["trading_days"] <= 0 or type(alpha) not in (int, float)):
            raise ValueError("invalid fixed-day economics")
        fill = FillModel(alpha)
    except (KeyError, TypeError, ValueError, OverflowError):
        raise fail("EXPERIMENT_VARIANT_FAILED", "invalid fixed-day exit economics") from None
    calendar = trading_calendar_from_snapshot(repository, snapshot, extend_days=0)
    return tuple(_walk_position(repository, snapshot, p, calendar, recipe["trading_days"], fill)
                 for p in positions)
