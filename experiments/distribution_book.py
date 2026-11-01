"""Pure premium-funded h1 research book; supplied cash totals include all costs.

The native adapter owns pricing, quantity/multiplier economics and quote clocks.
No future exit availability filters a decision or its entry affordability.
"""
from datetime import date, timedelta
from math import isfinite

from engine.v2.data.errors import DataError
from engine.v2.evaluation.premium_settlement import settle_premiums
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
    rows = {}
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
        rows[event]["quote"] = dict(quote) if quote is not None else None
    _require(set(quotes) <= set(rows))
    records = []
    for event, row in rows.items():
        quote = row["quote"]
        records.append(dict(id=event, bucket=row["ticker"], selected=row["selected"],
            entry_date=_date(row["entry_at"]) if row["entry_at"] is not None else None,
            settlement_date=_date(quote["exit_at"]) + timedelta(days=config["settlement_delay_days"]) if quote else None,
            debit=quote["entry_cash"] if quote else None, credit=quote["exit_cash"] if quote else None))
    progress = Progress("distribution book", sum((row["entry_date"] is not None) +
                        (row["settlement_date"] is not None) for row in records))
    order = "settlements_before_entries" if config["same_time_order"] == "settlements_before_entries" else "entries_before_settlements"
    with stage("distribution book"):
        try:
            result = settle_premiums(records, initial_cash=config["initial_cash"],
                max_trade_debit=config["max_trade_debit"], max_bucket_debit=config["max_ticker_debit"],
                max_concurrent=config["max_concurrent"], report_end=end, same_time_order=order, on_action=progress.tick)
        except DataError as exc:
            if exc.code != "CONTRACT_MISMATCH":
                raise
            raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution book inputs or arithmetic") from None
    for state in result.pop("per_opportunity"):
        event = state.pop("id")
        if state["status"] == "bucket_cap":
            state["status"] = "ticker_cap"
        rows[event].update(state)
    return {"schema_version": "distribution_book.v1", "config": config, "report_end": report_end,
            "per_event": [rows[event] for event in sorted(rows)], **result}
