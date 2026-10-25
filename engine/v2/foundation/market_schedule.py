"""Latest completed NYSE session at a serving clock's moment."""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from engine.v2.foundation.market_calendar import _holidays

__all__ = ["latest_completed_session"]

_EASTERN = ZoneInfo("America/New_York")


def latest_completed_session(now: datetime | None = None,
                             frozen_at: str | None = None) -> str:
    """The latest completed NYSE session at ``now`` or the snapshot clock.

    Today counts once its regular 16:00 America/New_York close has passed;
    otherwise the most recent earlier weekday that is not a scheduled holiday.
    A parseable ``frozen_at`` is the serving snapshot's clock, overriding the
    wall clock only when ``now`` is omitted; ``"unknown"`` means live.
    """
    if now is None and frozen_at and frozen_at != "unknown":
        try:
            parsed = datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None:
            now = parsed
    moment = datetime.now(_EASTERN) if now is None else now
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_EASTERN)
    moment = moment.astimezone(_EASTERN)
    day = moment.date()
    if moment.hour < 16 or day.weekday() >= 5 or day in _holidays(day.year):
        day -= timedelta(days=1)
    while day.weekday() >= 5 or day in _holidays(day.year):
        day -= timedelta(days=1)
    return day.isoformat()
