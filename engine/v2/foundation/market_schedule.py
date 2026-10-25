"""Latest completed NYSE session at a serving clock's moment."""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from engine.v2.foundation.market_calendar import is_scheduled_session

__all__ = ["latest_completed_session"]

_EASTERN = ZoneInfo("America/New_York")


def _snapshot_clock(frozen_at: str | None) -> datetime | None:
    """The snapshot clock, or ``None`` for the live wall clock.

    Only an absent value or the exact string ``"unknown"`` means live. Every
    other value must be a parseable ISO timestamp; empty, malformed and
    non-string evidence is refused rather than silently replaced by live time.
    """
    if frozen_at is None:
        return None
    if not isinstance(frozen_at, str) or not frozen_at:
        raise ValueError("frozen_at must be a parseable ISO timestamp or 'unknown'")
    if frozen_at == "unknown":
        return None
    try:
        return datetime.fromisoformat(frozen_at.replace("Z", "+00:00"))
    except ValueError as err:
        raise ValueError(f"frozen_at {frozen_at!r} is not a parseable ISO timestamp") from err


def latest_completed_session(now: datetime | None = None,
                             frozen_at: str | None = None) -> str:
    """The latest completed NYSE session at ``now`` or the snapshot clock.

    Today counts once its regular 16:00 America/New_York close has passed;
    otherwise the most recent earlier scheduled session. ``now`` wins when
    supplied. Otherwise ``frozen_at`` supplies the serving snapshot's clock;
    ``None`` and the exact string ``"unknown"`` mean live, and any other
    empty, malformed or non-string value raises ``ValueError`` instead of
    falling back to live time.
    """
    snapshot = _snapshot_clock(frozen_at)
    if now is None:
        now = snapshot
    moment = datetime.now(_EASTERN) if now is None else now
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_EASTERN)
    moment = moment.astimezone(_EASTERN)
    day = moment.date()
    if moment.hour < 16 or not is_scheduled_session(day):
        day -= timedelta(days=1)
    while not is_scheduled_session(day):
        day -= timedelta(days=1)
    return day.isoformat()
