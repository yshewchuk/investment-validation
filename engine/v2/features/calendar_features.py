"""Weekday/month/session come from the target's own stored event date and session, days-since-prior from the last H11 contributor's event date (design guides/size_feature_groups_design.md section 2); no production caller yet."""

from __future__ import annotations

from datetime import date

from engine.v2.data.errors import fail


def _contract(message):
    """Raise a CONTRACT_MISMATCH failure with the given message."""
    return fail("CONTRACT_MISMATCH", message)


def _iso(value, message):
    """Return value parsed as a canonical ISO date, otherwise raise a contract failure."""
    try:
        if not isinstance(value, str):
            raise _contract(message)
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise _contract(message)
    except (TypeError, ValueError):
        raise _contract(message) from None
    return parsed


def calendar_features(session, event_date, prior_event_date=None) -> dict:
    """Build the calendar columns for one target from its session and event dates."""
    if not isinstance(session, str) or session not in ("AMC", "BMO"):
        raise _contract("session must be AMC or BMO")
    event = _iso(event_date, "event_date must be a canonical ISO date")
    if prior_event_date is None:
        days_since_prior = None
    else:
        prior = _iso(prior_event_date, "prior_event_date must be None or a canonical ISO date")
        if prior > event:
            raise _contract("prior_event_date is after event_date")
        days_since_prior = (event - prior).days
    return {
        "cal_amc": int(session == "AMC"),
        "cal_weekday": event.weekday(),
        "cal_month": event.month,
        "cal_days_since_prior": days_since_prior,
    }
