"""Clocks and the one wire form of a timestamp — contracts §2.1.

Two clocks, never confused:

* ``now()`` is UTC wall time, for anything persisted. It can jump — NTP, a
  suspended laptop, a WSL host resuming — so it is never used to measure a
  duration.
* ``monotonic()`` is for in-process durations and deadlines. It cannot be
  persisted meaningfully: a new process, or a reboot, starts it elsewhere.

Everything that reads time takes a :class:`Clock`, so tests inject a fake one
rather than sleeping for the minutes a lease expiry would otherwise take.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Protocol

__all__ = ["Clock", "SystemClock", "format_timestamp", "parse_timestamp"]

_WIRE = "%Y-%m-%dT%H:%M:%S.%fZ"


class Clock(Protocol):
    """What anything that reads time depends on."""

    def now(self) -> datetime:
        """Timezone-aware UTC wall time."""

    def monotonic(self) -> float:
        """Seconds on a clock that never goes backwards within one process."""


class SystemClock:
    """The real clocks."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return time.monotonic()


def format_timestamp(value: datetime) -> str:
    """RFC 3339 UTC with an explicit ``Z`` and microseconds, per contracts §2.1.

    A naive datetime is refused rather than assumed to be UTC: "never a naive
    datetime" is the contract, and guessing is how a local-time value becomes
    a timestamp that is wrong by the host's offset.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("a naive datetime has no instant; attach a timezone")
    return value.astimezone(timezone.utc).strftime(_WIRE)


def parse_timestamp(text: str) -> datetime:
    """Inverse of :func:`format_timestamp`. Anything else is refused."""
    if not isinstance(text, str) or not text.endswith("Z"):
        raise ValueError(f"{text!r} is not an RFC 3339 UTC timestamp ending in Z")
    return datetime.strptime(text, _WIRE).replace(tzinfo=timezone.utc)
