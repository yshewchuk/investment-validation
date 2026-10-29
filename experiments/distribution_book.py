"""Pure premium-funded h1 research book; supplied cash totals include all costs.

The native adapter owns pricing, quantity/multiplier economics and quote clocks.
No future exit availability filters a decision or its entry affordability.
"""
from collections import defaultdict
from datetime import date, timedelta
from fractions import Fraction
from itertools import groupby
from math import isfinite

from engine.v2.ops.errors import fail
from engine.v2.research.progress import Progress, stage

__all__ = ["validate_book_config", "funded_book"]


def _require(condition):
    if not condition:
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution book inputs or arithmetic")


def _number(value):
    _require(type(value) in (int, float))
    try:
        _require(isfinite(value))
    except OverflowError:
        _require(False)
    return value


def _date(value):
    try:
        result = date.fromisoformat(value)
        _require(result.isoformat() == value)
        return result
    except (TypeError, ValueError):
        _require(False)


def _cash(value):
    try:
        return _number(float(value))
    except OverflowError:
        _require(False)


def validate_book_config(config):
    """Validate every explicit limit; settlement delays are calendar days."""
    _require(type(config) is dict and set(config) == {"initial_cash", "contracts_per_trade",
        "max_trade_debit", "max_ticker_debit", "max_concurrent", "settlement_delay_days", "same_time_order"})
    for key in ("initial_cash", "max_trade_debit", "max_ticker_debit"):
        _require(_number(config[key]) > 0)
    for key in ("contracts_per_trade", "max_concurrent", "settlement_delay_days"):
        _require(type(config[key]) is int and config[key] >= (0 if key == "settlement_delay_days" else 1))
    _require(config["same_time_order"] in ("settlements_before_entries", "entries_before_settlements"))
    return dict(config)


def funded_book(intents, quotes, *, config, report_end):
    """Run immutable decisions against total-USD quotes without changing inputs.

    Premium and capacity stay reserved through settlement, not just exit. Same
    day competition is explicit, then lexicographic event_id. Missing exit cash
    stays funded/unresolved through report_end. Equity is settled P&L plus initial
    cash (open positions at debit cost), NOT marked portfolio value. Monetary
    comparisons use exact decimal spellings, with no epsilon/rounding. O(n log n).
    """
    config, end = validate_book_config(config), _date(report_end)
    _require(type(intents) is list and type(quotes) is dict)
    rows, actions = {}, []
    entry_phase = int(config["same_time_order"] == "settlements_before_entries")
    for intent in intents:
        _require(type(intent) is dict and set(intent) ==
            {"event_id", "ticker", "event_date", "decision_at", "entry_at", "selected"})
        event = intent["event_id"]
        _require(all(type(intent[k]) is str and intent[k] for k in ("event_id", "ticker")))
        _require(event not in rows and type(intent["selected"]) is bool)
        _require(_date(intent["event_date"]) <= end)
        rows[event] = {**intent, "quote": None, "status": "no_trade", "net_profit": 0.0, "debit": 0.0, "capital_days": 0.0}
        if intent["decision_at"] is None and intent["entry_at"] is None:
            _require(not intent["selected"] and quotes.get(event) is None)
            rows[event]["status"] = "abstained"
            continue
        entry = _date(intent["entry_at"])
        _require(_date(intent["decision_at"]) < entry <= end)
        quote = quotes.get(event)
        if quote is not None:
            _require(type(quote) is dict and set(quote) == {"entry_cash", "exit_cash", "exit_at", "contracts", "multiplier"})
            _require(type(quote["contracts"]) is int and quote["contracts"] == config["contracts_per_trade"])
            _require(type(quote["multiplier"]) is int and quote["multiplier"] > 0)
            _require(quote["entry_cash"] is None or _number(quote["entry_cash"]) > 0)
            if quote["exit_cash"] is not None:
                _number(quote["exit_cash"])  # Net exit cash can be negative after fees.
            try:
                settlement = _date(quote["exit_at"]) + timedelta(days=config["settlement_delay_days"])
            except OverflowError:
                _require(False)
            _require(entry < _date(quote["exit_at"]) <= settlement <= end)
            actions.append((settlement, 1 - entry_phase, event, "settle"))
        rows[event]["quote"] = dict(quote) if quote is not None else None
        actions.append((entry, entry_phase, event, "entry"))
    _require(set(quotes) <= set(rows))
    money = {key: Fraction(str(config[key])) for key in ("initial_cash", "max_trade_debit", "max_ticker_debit")}
    cash = peak = equity = money["initial_cash"]
    locked, live, drawdown, capital_days = defaultdict(Fraction), set(), Fraction(), Fraction()
    curve, progress = [], Progress("distribution book", len(actions))
    with stage("distribution book"):
        for day, batch in groupby(sorted(actions), key=lambda action: action[0]):
            for _, _, event, action in batch:
                row, quote = rows[event], rows[event]["quote"]
                ticker = row["ticker"]
                if action == "settle" and event in live and quote["exit_cash"] is not None:
                    credit, debit = Fraction(str(quote["exit_cash"])), Fraction(str(quote["entry_cash"]))
                    cash += credit
                    row.update(status="settled", net_profit=_cash(credit - debit))
                    row["capital_days"] = _cash(debit * (day - _date(row["entry_at"])).days)
                    equity += credit - debit
                    locked[ticker] -= debit
                    live.remove(event)
                elif action == "entry" and row["selected"]:
                    debit = Fraction(str(quote["entry_cash"])) if quote is not None and quote["entry_cash"] is not None else None
                    reason = ("missing_entry" if debit is None else "trade_cap" if debit > money["max_trade_debit"]
                        else "ticker_cap" if locked[ticker] + debit > money["max_ticker_debit"]
                        else "concurrent_cap" if len(live) >= config["max_concurrent"]
                        else "cash" if debit > cash else "unresolved")
                    row["status"] = reason
                    if reason == "unresolved":
                        cash -= debit
                        locked[ticker] += debit
                        live.add(event)
                        row.update(debit=_cash(debit), net_profit=None,
                            capital_days=_cash(debit * (end - _date(row["entry_at"])).days))
                progress.tick()
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
            curve.append({"date": day.isoformat(), "cash": _cash(cash), "settlement_equity": _cash(equity)})
    if not curve or curve[-1]["date"] != report_end:
        curve.append({"date": report_end, "cash": _cash(cash), "settlement_equity": _cash(equity)})
    ordered = [rows[event] for event in sorted(rows)]
    for row in ordered:
        capital_days += Fraction(str(row["capital_days"]))
    funded = sum(row["debit"] > 0 for row in ordered)
    return {"schema_version": "distribution_book.v1", "config": config, "report_end": report_end,
        "per_event": ordered, "curve": curve, "summary": {"opportunities": len(rows), "funded": funded,
            "unresolved": len(live), "funded_coverage": funded / len(rows) if rows else None,
            "settled_net_profit": _cash(equity - money["initial_cash"]), "cash": _cash(cash),
            "settlement_equity": _cash(equity), "capital_days": _cash(capital_days), "settlement_drawdown": _cash(drawdown),
            "net_profit_per_opportunity": _cash((equity - money["initial_cash"]) / len(rows)) if rows and not live else None},
        "equity_policy": "settled_pnl_open_debit_cost_not_mtm", "capacity_policy": "reserved_until_settlement"}
