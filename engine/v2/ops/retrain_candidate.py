"""Pure candidate-ID and cutoff helpers for the monthly retrain chain (S3a).

The three driver pools are S1 re-frozen and champion-carried; the chooser pool is dated.
Candidate identity and cutoff derive deterministically, never inspecting state.
"""
import re
from datetime import date, timedelta

from engine.v2.ops.errors import fail

_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_SESSION = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")


def _safe_id(value: object) -> bool:
    return (isinstance(value, str) and bool(value) and not value.startswith(".")
            and not any(char in value for char in ("/", "\\", "\0")))


def _need(condition, field):
    if not condition:
        raise fail("INVALID_REQUEST", "invalid retrain builder input", details={"field": field})


def _date_ok(value: object, pattern: re.Pattern[str], suffix: str = "") -> bool:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        return False
    text = value + suffix
    try:
        return date.fromisoformat(text).isoformat() == text
    except ValueError:
        return False


def candidate_id(month: str, session: str) -> str:
    _need(_date_ok(month, _MONTH, "-01"), "month")
    _need(_date_ok(session, _SESSION), "session")
    result = f"pools-retrain-{month}-{session}"
    _need(_safe_id(result), "candidate_id")
    return result


def retrain_cutoff(month: str) -> str:
    _need(_date_ok(month, _MONTH, "-01"), "month")
    start = date(int(month[:4]), int(month[5:]), 1)
    try:
        return (start + timedelta(days=32)).replace(day=1).isoformat()
    except (ValueError, OverflowError):
        _need(False, "month")

