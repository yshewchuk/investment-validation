"""Season heat population roster: bounds metadata only, never outcome payloads.

plan_roster reads the canonical population, the native bounds table and the
whole-table computed-move fragment index, then binds anchors to fragments with
the shared outcome_binding helpers, so the roster digest binds label values
without opening a single outcome object.
"""
from datetime import date, timedelta

from engine.v2.data.errors import fail
from engine.v2.foundation import content_hash
from engine.v2.foundation.market_calendar import CalendarSessions
from engine.v2.research import outcome_binding
from engine.v2.research.experiment_population import load_population
from engine.v2.research.progress import decision, stage

WINDOW_SESSIONS = 252
SAMPLE_IDS = 10
SCHEMA = "season_heat_population.v1"
_REASONS = ("missing_native_bounds", "null_source_bounds", "off_session_event_date")


def _limits(limits):
    if type(limits) is not dict or type(limits.get("max_object_bytes")) is not int \
            or limits["max_object_bytes"] <= 0:
        raise fail("CONTRACT_MISMATCH", "explicit positive max_object_bytes required")
    return limits


def _validate(calendar, decision_sessions):
    if not isinstance(calendar, CalendarSessions):
        raise fail("CONTRACT_MISMATCH", "market calendar sessions required")
    sessions = list(decision_sessions)
    if not sessions or any(type(day) is not str or day not in calendar.days for day in sessions):
        raise fail("CONTRACT_MISMATCH", "decision sessions must be non-empty calendar sessions")
    return sessions


def _window_start(calendar, first):
    return calendar.days[max(0, calendar.days.index(first) - (WINDOW_SESSIONS - 1))]


def _roster(population, anchors, records, calendar, start, last):
    """Classify every population event once; no outcome object is opened here."""
    days = set(calendar.days)
    excluded = {reason: [] for reason in _REASONS}
    retained, in_window = [], 0
    for event in population.sort_values(["event_date", "event_id"]).itertuples(index=False):
        day = event.event_date.date().isoformat()
        if day < start or day > last:
            continue
        in_window += 1
        if (event.ticker, day) not in anchors:
            excluded["missing_native_bounds"].append(event.event_id)
            continue
        row = outcome_binding.bound(event, anchors, records)
        if row is None:
            excluded["null_source_bounds"].append(event.event_id)
            continue
        if day not in days:
            excluded["off_session_event_date"].append(event.event_id)
            continue
        available = (date.fromisoformat(row["post_close_date"]) + timedelta(days=1)).isoformat()
        if available > last:
            continue
        retained.append({"event_id": event.event_id, "outcome_object_id": row["outcome_object_id"],
            "outcome_content_hash": row["outcome_content_hash"], "available_as_of_date": available})
    retained.sort(key=lambda item: item["event_id"])
    return in_window, retained, excluded


def _receipt(*, as_of_month, purpose, start, first, last, calendar_source, population_size,
             in_window, retained, excluded):
    pairs = sorted([reason, event_id] for reason, event_ids in excluded.items()
        for event_id in event_ids)
    roster = [[item["event_id"], item["outcome_object_id"], item["outcome_content_hash"],
        item["available_as_of_date"]] for item in retained]
    excluded_ids = sorted({event_id for event_ids in excluded.values() for event_id in event_ids})
    excluded_total = sum(len(event_ids) for event_ids in excluded.values())
    receipt = {"schema_version": SCHEMA, "as_of_month": as_of_month, "purpose": purpose,
        "window": {"start": start, "end": last, "earliest_decision": first, "latest_decision": last},
        "calendar_source": calendar_source,
        "counts": {"population": population_size, "in_window": in_window, "retained": len(retained),
            "excluded": excluded_total,
            "not_yet_available": in_window - len(retained) - excluded_total},
        "exclusions": {reason: len(excluded[reason]) for reason in sorted(excluded)},
        "exclusions_digest": content_hash(pairs), "roster_digest": content_hash(roster),
        "sample_ids": excluded_ids[:SAMPLE_IDS]}
    receipt["receipt_id"] = content_hash(receipt)
    return receipt


def plan_roster(repository, snapshot, *, as_of_month, decision_sessions, calendar, calendar_source,
                limits, purpose="selection"):
    """Metadata-only roster over the population window; opens no outcome.
    Returns {"receipt": {...}, "retained": [...]}."""
    _limits(limits)
    sessions = _validate(calendar, decision_sessions)
    first, last = min(sessions), max(sessions)
    start = _window_start(calendar, first)
    with stage("season_roster"):
        population = load_population(repository, snapshot, as_of_month=as_of_month, purpose=purpose,
            max_object_bytes=limits["max_object_bytes"])
        anchors = outcome_binding.anchors(repository, snapshot, limits)
        records = outcome_binding.fragment_index(repository, snapshot)
        in_window, retained, excluded = _roster(population, anchors, records, calendar, start, last)
        receipt = _receipt(as_of_month=as_of_month, purpose=purpose, start=start, first=first,
            last=last, calendar_source=calendar_source, population_size=len(population),
            in_window=in_window, retained=retained, excluded=excluded)
        decision("season_roster", f"population {receipt['counts']['population']}; in window "
            f"{in_window}; retained {len(retained)}; excluded {receipt['counts']['excluded']}; "
            f"not yet available {receipt['counts']['not_yet_available']}; window {start}..{last}")
    return {"receipt": receipt, "retained": retained}
