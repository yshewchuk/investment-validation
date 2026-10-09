"""Shared experiment membership definitions; no read/authorization capability."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import ClassVar


@dataclass(frozen=True)
class ExperimentHoldouts:
    as_of_month: str
    random_version: ClassVar[str] = "canonical-event-sha256.v1"
    rolling_version: ClassVar[str] = "calendar-months.v1"
    _month: int = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.as_of_month, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}", self.as_of_month):
            raise ValueError("as-of month must be YYYY-MM")
        month = date.fromisoformat(self.as_of_month + "-01")
        object.__setattr__(self, "_month", month.year * 12 + month.month - 1)

    @staticmethod
    def random_membership(event_id: str) -> bool:
        if not isinstance(event_id, str) or not event_id.strip() or event_id != event_id.strip():
            raise ValueError("canonical event identity must be a nonblank, unpadded string")
        digest = hashlib.sha256(b"experiment-holdouts/random/v1\0" + event_id.encode("utf-8")).digest()
        return 100 * int.from_bytes(digest, "big") < 3 * (1 << 256)

    def classify(self, event_id: str, event_date) -> frozenset[str]:
        labels = set()
        try:
            if self.random_membership(event_id):
                labels.add("random")
        except ValueError:
            labels.add("ambiguous")
        try:
            day = _calendar_date(event_date)
            month = day.year * 12 + day.month - 1
            if month > self._month:
                labels.add("ambiguous")
            elif month >= self._month - 5:
                labels.add("rolling")
        except (TypeError, ValueError, OverflowError):
            labels.add("ambiguous")
        return frozenset(labels)


def _calendar_date(value) -> date:
    if isinstance(value, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        value = date.fromisoformat(value)
    if isinstance(value, datetime):
        day = value.date()
        if value.tzinfo is not None or value != datetime.combine(day, time.min):
            raise ValueError("canonical date must be an unambiguous calendar day")
        value = day
    if type(value) is not date:
        raise ValueError("canonical date must be an unambiguous calendar day")
    return value
