"""Pure premium-funded settlement accounting, not selection or mark-to-market."""
from collections import defaultdict
from datetime import date
from fractions import Fraction
from itertools import groupby
from math import isfinite

from engine.v2.data.errors import fail

__all__ = ["settle_premiums"]


def _require(condition):
    if not condition:
        raise fail("CONTRACT_MISMATCH", "invalid premium settlement inputs or arithmetic")


def _finite(value):
    _require(type(value) in (int, float))
    try:
        _require(isfinite(value))
    except OverflowError:
        _require(False)
    return value


def _cash(value):
    try:
        return _finite(float(value))
    except OverflowError:
        _require(False)


def _record_fields(record, seen):
    _require(type(record) is dict and set(record) ==
             {"id", "bucket", "selected", "entry_date", "settlement_date", "debit", "credit"})
    _require(all(type(record[key]) is str and record[key] for key in ("id", "bucket")))
    event, entry, settlement = record["id"], record["entry_date"], record["settlement_date"]
    _require(event not in seen and type(record["selected"]) is bool)
    _require(all(value is None or type(value) is date for value in (entry, settlement)))
    _require(record["debit"] is None or _finite(record["debit"]) > 0)
    if record["credit"] is not None:
        _finite(record["credit"])
    return event, entry, settlement


def _prepare(records, report_end, entry_phase):
    prepared, rows, actions = {}, {}, []
    for record in records:
        event, entry, settlement = _record_fields(record, prepared)
        if entry is None:
            _require(not record["selected"] and settlement is None and record["debit"] is None and record["credit"] is None)
        else:
            _require(entry <= report_end)
            if settlement is None:
                _require(record["debit"] is None and record["credit"] is None)
            else:
                _require(entry < settlement <= report_end)
                actions.append((settlement, 1 - entry_phase, event, "settle"))
            actions.append((entry, entry_phase, event, "entry"))
        prepared[event] = dict(record)
        rows[event] = {"id": event, "status": "abstained" if entry is None else "no_trade",
                       "net_profit": 0.0, "debit": 0.0, "capital_days": 0.0}
    return prepared, rows, actions


def _entry_reason(debit, money, locked, concurrent, max_concurrent, cash):
    return ("missing_entry" if debit is None else "trade_cap" if debit > money[1]
            else "bucket_cap" if locked + debit > money[2]
            else "concurrent_cap" if concurrent >= max_concurrent
            else "cash" if debit > cash else "unresolved")


def _settle(prepared, rows, actions, money, max_concurrent, report_end, on_action):
    cash = peak = equity = money[0]
    locked, live, drawdown = defaultdict(Fraction), set(), Fraction()
    curve = []
    for day, batch in groupby(sorted(actions), key=lambda action: action[0]):
        for _, _, event, action in batch:
            row, record = rows[event], prepared[event]
            bucket = record["bucket"]
            if action == "settle" and event in live and record["credit"] is not None:
                credit, debit = Fraction(str(record["credit"])), Fraction(str(record["debit"]))
                cash += credit
                row.update(status="settled", net_profit=_cash(credit - debit))
                row["capital_days"] = _cash(debit * (day - record["entry_date"]).days)
                equity += credit - debit
                locked[bucket] -= debit
                live.remove(event)
            elif action == "entry" and record["selected"]:
                debit = Fraction(str(record["debit"])) if record["debit"] is not None else None
                reason = _entry_reason(debit, money, locked[bucket], len(live), max_concurrent, cash)
                row["status"] = reason
                if reason == "unresolved":
                    cash -= debit
                    locked[bucket] += debit
                    live.add(event)
                    row.update(debit=_cash(debit), net_profit=None,
                               capital_days=_cash(debit * (report_end - record["entry_date"]).days))
            if on_action is not None:
                on_action()
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
        curve.append({"date": day.isoformat(), "cash": _cash(cash), "settlement_equity": _cash(equity)})
    if not curve or curve[-1]["date"] != report_end.isoformat():
        curve.append({"date": report_end.isoformat(), "cash": _cash(cash), "settlement_equity": _cash(equity)})
    return cash, equity, drawdown, len(live), curve


def settle_premiums(records, *, initial_cash, max_trade_debit, max_bucket_debit,
                    max_concurrent, report_end, same_time_order, on_action=None):
    """Settle every normalized opportunity; supplied cash already includes costs.

    Dates are plain date objects. Funding and capacity stay reserved until
    settlement; missing credit remains unresolved. Decimal-spelling arithmetic
    has no epsilon. The caller owns stage logging and an optional per-action
    progress callback, whose exceptions propagate. O(n log n) time, O(n) space.
    """
    _require(type(records) is list and type(report_end) is date)
    _require(type(max_concurrent) is int and max_concurrent > 0)
    _require(type(same_time_order) is str and same_time_order in
             ("settlements_before_entries", "entries_before_settlements"))
    _require(on_action is None or callable(on_action))
    for value in (initial_cash, max_trade_debit, max_bucket_debit):
        _require(_finite(value) > 0)
    money = [Fraction(str(value)) for value in (initial_cash, max_trade_debit, max_bucket_debit)]
    prepared, rows, actions = _prepare(records, report_end, int(same_time_order == "settlements_before_entries"))
    cash, equity, drawdown, unresolved, curve = _settle(prepared, rows, actions, money, max_concurrent, report_end, on_action)
    ordered = [rows[event] for event in sorted(rows)]
    capital_days = sum((Fraction(str(row["capital_days"])) for row in ordered), Fraction())
    funded = sum(row["debit"] > 0 for row in ordered)
    return {"per_opportunity": ordered, "curve": curve, "summary": {"opportunities": len(rows), "funded": funded,
        "unresolved": unresolved, "funded_coverage": funded / len(rows) if rows else None,
        "settled_net_profit": _cash(equity - money[0]), "cash": _cash(cash), "settlement_equity": _cash(equity),
        "capital_days": _cash(capital_days), "settlement_drawdown": _cash(drawdown),
        "net_profit_per_opportunity": _cash((equity - money[0]) / len(rows)) if rows and not unresolved else None},
        "equity_policy": "settled_pnl_open_debit_cost_not_mtm", "capacity_policy": "reserved_until_settlement"}
