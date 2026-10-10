"""Shared value types for native score batch inputs and refusals."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

from engine.v2.ops.native_board_universe import BoardRequest

__all__ = ["NativeScoreBatchRowRefusal"]

_EVENT_DATE_IDENTITY_ERROR = (
    "event_date must be a naive calendar date/day value or a naive "
    "datetime, encoded as YYYY-MM-DD (midnight) or a canonical ISO datetime")


def _event_date_identity(value: Any) -> str:
    if isinstance(value, str):
        canonical_input: str | None = value
    else:
        if not isinstance(value, (date, datetime)) \
                or getattr(value, "tzinfo", None) is not None:
            raise ValueError(_EVENT_DATE_IDENTITY_ERROR)
        canonical_input = None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise ValueError(_EVENT_DATE_IDENTITY_ERROR) from None
    if pd.isna(timestamp) or getattr(timestamp, "tz", None) is not None:
        raise ValueError(_EVENT_DATE_IDENTITY_ERROR)
    encoded = (timestamp.date().isoformat()
               if timestamp.normalize() == timestamp else timestamp.isoformat())
    if canonical_input is not None and encoded != canonical_input:
        raise ValueError(_EVENT_DATE_IDENTITY_ERROR)
    return encoded


class NativeScoreBatchRowRefusal(ValueError):
    """One row's typed refusal, collected rather than raised by batch assembly."""

    def __init__(self, key: BoardRequest, code: str, detail: str) -> None:
        self.key = key
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {key!r}: {detail}")

    def as_document(self) -> dict[str, Any]:
        return {
            "key": {
                "ticker": self.key.ticker,
                "strategy": self.key.strategy,
                "event_date": _event_date_identity(self.key.event_date),
                "session": self.key.session,
            },
            "code": self.code,
            "detail": self.detail,
        }
