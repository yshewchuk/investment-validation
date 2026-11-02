"""Pure date helpers for chooser cutoff and fold-start rules."""

from __future__ import annotations

from datetime import date, datetime

_ERROR = "invalid chooser cutoff"


def _iso_date(value: object) -> date:
    if not isinstance(value, str):
        raise ValueError(_ERROR)
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(_ERROR) from None
    if parsed.isoformat() != value:
        raise ValueError(_ERROR)
    return parsed


def canonical_day(value: object) -> str:
    """Return a canonical ISO day from a date, naive datetime or canonical string."""
    try:
        if isinstance(value, datetime):
            if value.utcoffset() is not None:
                raise ValueError(_ERROR)
            day = value.date()
        elif isinstance(value, date):
            day = value
        elif isinstance(value, str) and len(value) == 10:
            day = date.fromisoformat(value)
            if day.isoformat() != value:
                raise ValueError(_ERROR)
        elif (isinstance(value, str) and len(value) == 19
              and value[10] in ("T", " ")):
            parsed = datetime.fromisoformat(value)
            if parsed.utcoffset() is not None:
                raise ValueError(_ERROR)
            canonical = (parsed.isoformat(timespec="seconds") if value[10] == "T"
                         else parsed.isoformat(sep=" ", timespec="seconds"))
            if canonical != value:
                raise ValueError(_ERROR)
            day = parsed.date()
        else:
            raise ValueError(_ERROR)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(_ERROR) from None
    return day.isoformat()


def require_causal_cutoff(
    training_cutoff: object, fold_start: object, decision_cutoff: object,
) -> None:
    """Require canonical date strings with training strictly before both cutoffs."""
    training = _iso_date(training_cutoff)
    fold = _iso_date(fold_start)
    decision = _iso_date(decision_cutoff)
    if training >= fold or training >= decision:
        raise ValueError(_ERROR)


def chooser_fold_start(event_date: object, decision_cutoff: object) -> str:
    """Return the earlier month start for an event date and decision cutoff."""
    event = date.fromisoformat(canonical_day(event_date)).replace(day=1)
    decision = date.fromisoformat(canonical_day(decision_cutoff)).replace(day=1)
    return min(event, decision).isoformat()
