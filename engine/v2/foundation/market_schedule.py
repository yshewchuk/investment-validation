"""Latest completed NYSE session at a serving clock's moment."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from engine.v2.foundation.market_calendar import is_scheduled_session

__all__ = ["latest_completed_session"]

_EASTERN = ZoneInfo("America/New_York")
_REGULAR_CLOSE_HOUR = 16
_EARLY_CLOSE_HOUR = 13


def _scheduled_close_hour(day) -> int:
    """The America/New_York close hour of a session day under this calendar.

    Returns the early ``_EARLY_CLOSE_HOUR`` for the three recurring early-close
    rules this dependency-free calendar models -- July 3 when it is a scheduled
    session (a Friday July 4 observance makes July 3 a full holiday, so it is
    not a session), the Friday after Thanksgiving, and Christmas Eve when it is
    a scheduled session -- and the regular ``_REGULAR_CLOSE_HOUR`` otherwise.
    Exceptional exchange-announced half-days beyond these three modeled rules
    are not covered: they keep the regular close, a bounded conservative lag.
    """
    if ((day.month, day.day) == (7, 3) or (day.month, day.day) == (12, 24)) \
            and is_scheduled_session(day):
        return _EARLY_CLOSE_HOUR
    if day.weekday() == 4 and day.month == 11 and 23 <= day.day <= 29:
        return _EARLY_CLOSE_HOUR
    return _REGULAR_CLOSE_HOUR


def _snapshot_clock(frozen_at: str | None) -> datetime | None:
    """The snapshot clock, or ``None`` for the live wall clock.

    Only an absent value or the exact string ``"unknown"`` means live. Every
    other value must be a parseable ISO timestamp; a date-only string is
    refused, never read as midnight. Empty, malformed and non-string evidence
    is refused rather than silently replaced by live time.
    """
    if frozen_at is None:
        return None
    if not isinstance(frozen_at, str) or not frozen_at:
        raise ValueError("frozen_at must be a parseable ISO timestamp or 'unknown'")
    if frozen_at == "unknown":
        return None
    try:
        parsed = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as err:
        raise ValueError(f"frozen_at {frozen_at!r} is not a parseable ISO timestamp") from err
    if len(frozen_at) <= 10:
        raise ValueError(f"frozen_at {frozen_at!r} is a date, not an ISO timestamp")
    return parsed


def latest_completed_session(now: datetime | None = None,
                             frozen_at: str | None = None) -> str:
    """The latest completed NYSE session at ``now`` or the snapshot clock.

    Today counts once its scheduled America/New_York close has passed -- 16:00
    as usual, 13:00 on the three early-close days this module models: July 3
    when it is a scheduled session (an observed July 4 that falls on July 3 is
    a full holiday, not a session), the Friday after Thanksgiving, and
    Christmas Eve when it is a scheduled session. Exceptional
    exchange-announced half-days beyond these three modeled rules are not
    covered and keep the 16:00 boundary. Otherwise the most recent earlier
    scheduled session. ``now`` wins when supplied. Otherwise ``frozen_at``
    supplies the serving snapshot's clock; ``None`` and the exact string
    ``"unknown"`` mean live, and any other empty, malformed or non-string
    value raises ``ValueError`` instead of falling back to live time.
    """
    snapshot = _snapshot_clock(frozen_at)
    if now is None:
        now = snapshot
    moment = datetime.now(_EASTERN) if now is None else now
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_EASTERN)
    moment = moment.astimezone(_EASTERN)
    day = moment.date()
    closed_at = datetime.combine(day, time(_scheduled_close_hour(day)), tzinfo=_EASTERN)
    if moment < closed_at or not is_scheduled_session(day):
        day -= timedelta(days=1)
    while not is_scheduled_session(day):
        day -= timedelta(days=1)
    return day.isoformat()
