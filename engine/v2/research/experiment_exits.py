"""Declared fixed-day and target/stop exits over held contracts; no publication."""
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
    exit_fill_alpha: float
    reason: str = "fixed_day"
    pnl_basis: str = "mark_based"
    mark_source: str = "option_chains"
    fill_convention: str = "alpha_ladder"
    ambiguous_exit: bool = False


def _session(value):
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("noncanonical session")
    return pd.Timestamp(parsed)


def _validated_leg_key(leg):
    if not isinstance(leg, PositionLeg):
        raise ValueError("invalid held leg record")
    if (leg.side not in ("buy", "sell") or leg.right not in ("C", "P")
            or not isfinite(leg.qty) or leg.qty <= 0
            or not isfinite(leg.strike) or leg.strike <= 0):
        raise ValueError("invalid held leg")
    return _session(leg.expiry), leg.strike, leg.right


def _validate_position(position):
    if (not isinstance(position.trade_id, str) or not position.trade_id.strip()
            or not isinstance(position.ticker, str) or not position.ticker.strip()
            or not isinstance(position.legs, tuple) or not position.legs):
        raise ValueError("invalid entered position")
    contracts = set()
    for leg in position.legs:
        key = _validated_leg_key(leg)
        if key in contracts:
            raise ValueError("duplicate held contract")
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
        if not isfinite(bid) or not isfinite(ask) or ask <= 0:
            raise ValueError("unusable required mark")
        side = {"buy": "sell", "sell": "buy"}[leg.side] if closing else leg.side
        total += float(fill.cash_flow(side, bid, ask, leg.qty))
    if not isfinite(total):
        raise ValueError("nonfinite position cash flow")
    return total


def _walk_position(repository, snapshot, position, calendar, days, fill):
    if not isinstance(position, EnteredPosition):
        raise fail("EXPERIMENT_VARIANT_FAILED", "entered position record is malformed")
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
                        float(fill.alpha), float(fill.alpha))


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


def _exit_recipe(economic_params):
    """The one declared exit recipe and its already-validated alpha."""
    recipe = economic_params["exit"]
    alpha = economic_params["fill"]
    if type(alpha) not in (int, float) or not isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("invalid fill economics")
    if not isinstance(recipe, Mapping):
        raise ValueError("invalid exit recipe")
    days = recipe.get("trading_days")
    if type(days) is not int or days <= 0:
        raise ValueError("invalid trading horizon")
    kind = recipe.get("kind")
    if kind == "fixed_day":
        if set(recipe) != {"kind", "trading_days"}:
            raise ValueError("invalid fixed-day recipe")
    elif kind == "target_stop":
        if set(recipe) != {"kind", "trading_days", "target_pnl", "stop_pnl"}:
            raise ValueError("invalid target/stop recipe")
        for name, sign in (("target_pnl", 1), ("stop_pnl", -1)):
            value = recipe[name]
            if (type(value) not in (int, float) or not isfinite(value)
                    or sign * value <= 0):
                raise ValueError("invalid target/stop threshold")
    else:
        raise ValueError("unknown exit kind")
    return recipe


def _decision(position, session, opening, held, dates, snapshot_id, alpha,
              reason, ambiguous=False, exit_alpha=None):
    if exit_alpha is None:
        exit_alpha = alpha
    return ExitDecision(position.trade_id, session, -opening, held, opening + held,
                        tuple(d.date().isoformat() for d in dates), snapshot_id,
                        alpha, exit_alpha, reason=reason, ambiguous_exit=ambiguous)


def _held_decision(position, index, dates, opening, fill, target, stop, snapshot_id):
    """First target/stop crossing, or the 7a fixed-day fallback at day N.

    Each observed day's declared-alpha mark is compared after its alpha 0/1
    quote bounds: a range spanning both a positive target and a negative stop
    cannot order the two crossings, so the conservative stop wins, the decision
    is labelled ambiguous, and its exit value and P&L use the worst existing
    alpha (0.0) fill.
    """
    worst, best = FillModel(0.0), FillModel(1.0)
    for number, day in enumerate(dates[1:], start=1):
        session = day.date().isoformat()
        if any(day > _session(leg.expiry) for leg in position.legs):
            raise ValueError("required mark past expiry")
        rows = index.get(position.ticker, day)
        held = _cash_flow(position, rows, fill, closing=True)
        worst_held = _cash_flow(position, rows, worst, closing=True)
        pnl = opening + held
        low = opening + worst_held
        high = opening + _cash_flow(position, rows, best, closing=True)
        visited = dates[:number + 1]
        if low <= stop and high >= target:
            return _decision(position, session, opening, worst_held, visited, snapshot_id,
                             float(fill.alpha), "stop", True, exit_alpha=0.0)
        if pnl >= target:
            return _decision(position, session, opening, held, visited, snapshot_id,
                             float(fill.alpha), "target")
        if pnl <= stop:
            return _decision(position, session, opening, held, visited, snapshot_id,
                             float(fill.alpha), "stop")
    last = dates[-1]
    held = _cash_flow(position, index.get(position.ticker, last), fill, closing=True)
    return _decision(position, last.date().isoformat(), opening, held, dates,
                     snapshot_id, float(fill.alpha), "fixed_day")


def _walk_target_stop(repository, snapshot, position, calendar, days, fill, target, stop):
    if not isinstance(position, EnteredPosition):
        raise fail("EXPERIMENT_VARIANT_FAILED", "entered position record is malformed")
    session = position.entry_date
    try:
        _validate_position(position)
        start = calendar.index_of(_session(session))
        dates = calendar.days[start:start + days + 1]
        if len(dates) != days + 1:
            raise ValueError("insufficient observed calendar")
        index = load_chain_index(repository, snapshot, [(position.ticker, d) for d in dates])
        opening = _cash_flow(position, index.get(position.ticker, dates[0]), fill,
                             closing=False)
        decision = _held_decision(position, index, dates, opening, fill, target, stop,
                                  snapshot.snapshot_id)
        if not isfinite(decision.pnl):
            raise ValueError("nonfinite P&L")
    except (ValueError, TypeError, KeyError, OverflowError):
        raise fail("EXPERIMENT_VARIANT_FAILED", "required daily position mark unavailable",
                   details={"trade_id": position.trade_id, "session": session}) from None
    return decision


def walk_exit(repository, snapshot, positions, *, economic_params):
    """Walk the declared fixed-day or target/stop exit over one pinned snapshot.

    Both recipes aggregate each observed day's net held-position P&L with the
    pinned ``option_chains`` quotes and the existing alpha ladder. A target or
    stop exits on its first crossing at that day's declared-alpha mark; when a
    day's alpha 0/1 quote bounds span both thresholds the stop is selected and
    ``ambiguous_exit`` is set, priced at the worst existing alpha (0.0) so the
    ambiguous P&L can only be as adverse as, never better than, the stop. No
    crossing by N sessions uses the fixed-day fallback at day N. Refusal is
    all-or-nothing: the whole call fails, never a partial or skipped trade.
    """
    try:
        recipe = _exit_recipe(economic_params)
        fill = FillModel(economic_params["fill"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise fail("EXPERIMENT_VARIANT_FAILED", "invalid exit economics") from None
    if recipe["kind"] == "fixed_day":
        return walk_fixed_day(repository, snapshot, positions,
                              economic_params=economic_params)
    calendar = trading_calendar_from_snapshot(repository, snapshot, extend_days=0)
    return tuple(_walk_target_stop(repository, snapshot, p, calendar,
                                   recipe["trading_days"], fill,
                                   float(recipe["target_pnl"]), float(recipe["stop_pnl"]))
                 for p in positions)


#: One report row per decision, in a stable column order for empty inputs too.
REPORT_COLUMNS = ("trade_id", "exit_reason", "exit_day", "mark_based_pnl", "pnl_basis",
                  "mark_source", "fill_convention", "fill_alpha", "exit_fill_alpha",
                  "snapshot_id", "ambiguous_exit")


def exit_report_frame(decisions):
    """Project exit decisions into per-trade report columns; write nothing.

    Mark-based P&L stays in its own ``mark_based_pnl`` column with its basis,
    source, fill convention/alpha, the alpha actually applied at exit, snapshot
    provenance and ambiguity, never blended with fill-validated performance. An
    empty input still returns the full set of :data:`REPORT_COLUMNS`.
    """
    rows = [{"trade_id": d.trade_id, "exit_reason": d.reason, "exit_day": d.exit_date,
             "mark_based_pnl": d.pnl, "pnl_basis": d.pnl_basis,
             "mark_source": d.mark_source, "fill_convention": d.fill_convention,
             "fill_alpha": d.fill_alpha, "exit_fill_alpha": d.exit_fill_alpha,
             "snapshot_id": d.snapshot_id,
             "ambiguous_exit": d.ambiguous_exit} for d in decisions]
    return pd.DataFrame(rows, columns=list(REPORT_COLUMNS))
