"""Pure signed IV30 target rows; supplied projections confer no read authority."""
from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from datetime import date, timedelta

from engine.v2.foundation import content_hash

from . import errors

MAX_GAP_DAYS = 5


def _day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError
    return value


def _text(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip() or len(value) > 256:
        raise ValueError
    return value


def _projection(rows, *, day, field, cap):
    result = []
    for row in rows:
        if len(result) >= cap:
            raise errors.fail("RESULT_LIMIT_EXCEEDED", "IV-crush input exceeds its row bound")
        value = row[field]
        if value is not None and type(value) not in (str, bool, int, float):
            raise ValueError
        if isinstance(value, str) and len(value) > 256:
            raise ValueError
        result.append({day: _day(row[day]), field: value})
    result.sort(key=lambda row: row[day])
    if len({row[day] for row in result}) != len(result):
        raise ValueError
    return result


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _pair(event, daily, days):
    """Retain the actual adjacent rows; never skip over an invalid quote."""
    row = {"session": event["session"] if event["session"] in ("AMC", "BMO") else None,
           "pre_date": None, "post_date": None, "pre_iv30": None, "post_iv30": None,
           "crush_pct_iv30": None, "available_as_of_date": None}
    if row["session"] is None:
        return row, "missing_session" if event["session"] is None else "unknown_session"
    search = bisect_right if row["session"] == "AMC" else bisect_left
    post = search(days, event["event_date"])
    for prefix, index in (("pre", post - 1), ("post", post)):
        if 0 <= index < len(daily):
            quote = daily[index]
            row[prefix + "_date"] = quote["date"]
            row[prefix + "_iv30"] = float(quote["iv30"]) if _finite(quote["iv30"]) else None
    if post == 0:
        return row, "missing_pre"
    if post == len(daily):
        return row, "missing_post"
    pre_day, post_day = date.fromisoformat(row["pre_date"]), date.fromisoformat(row["post_date"])
    if (post_day - pre_day).days > MAX_GAP_DAYS:
        return row, "gap_exceeds_limit"
    pre, after = row["pre_iv30"], row["post_iv30"]
    if pre is None or pre <= 0:
        return row, "invalid_pre_iv30"
    if after is None:
        return row, "invalid_post_iv30"
    value = 100.0 * (after / pre - 1.0)
    if not math.isfinite(value):
        return row, "nonfinite_target"
    # Finite nonpositive post-IV deliberately retains the existing recipe's rule.
    row.update(crush_pct_iv30=value, available_as_of_date=(post_day + timedelta(days=1)).isoformat())
    return row, None


def build_rows(ticker, events, daily, *, source_id, max_rows=10000):
    """One target-or-skip row per supplied event, sorted by canonical event date.

    ``events`` contains event_date/session; ``daily`` contains date/iv30 for
    this ticker only. Both are bounded independently by max_rows. Other fields
    are ignored, except that an optional ticker must agree. ``source_id`` is a
    caller assertion. The input hash binds projections, not original objects,
    source publication, completeness or finality. No I/O or holdout admission.
    """
    try:
        _text(ticker)
        _text(source_id)
        if type(max_rows) is not int or max_rows <= 0:
            raise ValueError

        def checked(rows):
            for row in rows:
                if row.get("ticker", ticker) != ticker:
                    raise ValueError
                yield row

        events = _projection(checked(events), day="event_date", field="session", cap=max_rows)
        daily = _projection(checked(daily), day="date", field="iv30", cap=max_rows)
        input_hash = content_hash({"schema_version": "iv_crush_inputs.v1", "ticker": ticker,
                                   "events": events, "daily": daily})
        days = [row["date"] for row in daily]
        result = []
        for event in events:
            row, reason = _pair(event, daily, days)
            result.append({"ticker": ticker, "event_date": event["event_date"], **row,
                           "skipped": reason is not None, "skip_reason": reason,
                           "source_id": source_id, "input_hash": input_hash})
        return result
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise errors.fail("CONTRACT_MISMATCH", "IV-crush input is malformed or ambiguous") from None
