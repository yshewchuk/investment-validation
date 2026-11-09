"""Season heat population roster: bounds metadata only, never outcome payloads.

plan_roster reads the canonical population, the native bounds table and the
whole-table computed-move fragment index, then binds anchors to fragments with
the shared outcome_binding helpers, so the roster digest binds label values
without opening a single outcome object. read_labels is the value build: it
admits the source, resolves the roster ids through the population, then reads
exactly those outcome objects: each retained label's object is read once per
call; reads go ticker by ticker so a ticker's run stream finishes before the
next opens (adjacent same-ticker labels share one run scan), refusing on any
contradiction.
"""
import math
from datetime import date, timedelta

from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_V5_CONTRACT
from engine.v2.data.errors import fail
from engine.v2.foundation import content_hash
from engine.v2.foundation.market_calendar import CalendarSessions
from engine.v2.research import event_history, outcome_binding
from engine.v2.research.eligible_history import admit_source
from engine.v2.research.experiment_population import load_population
from engine.v2.research.progress import Progress, decision, stage

WINDOW_SESSIONS = 252
SAMPLE_IDS = 10
SCHEMA = "season_heat_population.v1"
_REASONS = ("missing_native_bounds", "null_source_bounds", "off_session_event_date")


def _validate(calendar, decision_sessions):
    if not isinstance(calendar, CalendarSessions):
        raise fail("CONTRACT_MISMATCH", "market calendar sessions required")
    try:
        sessions = list(decision_sessions)
    except TypeError:
        raise fail("CONTRACT_MISMATCH", "decision sessions must be an iterable") from None
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
    limits is the eligible-history limits dict (max_source_rows, max_fragments,
    max_object_bytes, max_total_bytes, max_output_rows); the H11 admission bounds
    every whole-table read before it happens, and the retained count is capped by
    max_output_rows.
    Returns {"receipt": {...}, "retained": [...]}."""
    admit_source(repository, snapshot, limits)
    sessions = _validate(calendar, decision_sessions)
    first, last = min(sessions), max(sessions)
    start = _window_start(calendar, first)
    with stage("season_roster"):
        population = load_population(repository, snapshot, as_of_month=as_of_month, purpose=purpose,
            max_object_bytes=limits["max_object_bytes"])
        anchors = outcome_binding.anchors(repository, snapshot, limits)
        records = outcome_binding.fragment_index(repository, snapshot)
        in_window, retained, excluded = _roster(population, anchors, records, calendar, start, last)
        if len(retained) > limits["max_output_rows"]:
            raise fail("RESULT_LIMIT_EXCEEDED", "season roster exceeds max_output_rows")
        receipt = _receipt(as_of_month=as_of_month, purpose=purpose, start=start, first=first,
            last=last, calendar_source=calendar_source, population_size=len(population),
            in_window=in_window, retained=retained, excluded=excluded)
        decision("season_roster", f"population {receipt['counts']['population']}; in window "
            f"{in_window}; retained {len(retained)}; excluded {receipt['counts']['excluded']}; "
            f"not yet available {receipt['counts']['not_yet_available']}; window {start}..{last}")
    return {"receipt": receipt, "retained": retained}


def _abs_move(value):
    """The label move as a finite float magnitude, or a variant refusal."""
    try:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError
        return float(abs(value))
    except (TypeError, ValueError, OverflowError):
        raise fail("EXPERIMENT_VARIANT_FAILED", "label move is invalid") from None


def _label_move(rows, event, item):
    """Validate one decoded outcome row against its roster binding; returns abs_move."""
    day = event.event_date.date().isoformat()
    if len(rows) != 1 or rows[0]["event_date"] != day:
        raise fail("CONTRACT_MISMATCH", "label outcome identity is invalid")
    if rows[0]["skipped"] is not False:
        raise fail("CONTRACT_MISMATCH", "source bounds contradict a skipped label outcome")
    if rows[0]["available_as_of_date"] != item["available_as_of_date"]:
        raise fail("CONTRACT_MISMATCH", "label availability differs from the roster anchor")
    return _abs_move(rows[0]["realized_move_pct"])


def read_labels(repository, snapshot, retained, *, as_of_month, limits, purpose="selection"):
    """Value-build read of the planned roster's realized labels.

    Each retained label's object is read once per call; reads go ticker by
    ticker so a ticker's run stream finishes before the next opens (adjacent
    same-ticker labels share one run scan; event_history.plan_with_reader).
    limits is the eligible-history limits dict (same as plan_roster); the H11
    admission validates it and bounds every whole-table read before they
    happen, and its max_object_bytes bounds the population read.

    Returns season_heat label dicts (event_id, ticker, event_date, available_as_of_date,
    abs_move) in retained order. Source admission runs before any outcome opens, so a
    held-out label refuses the whole read; nothing is cached across calls, nothing is
    retried, and any refusal propagates without a partial result. The roster already dropped
    labels available after the latest decision, so no decision-date comparison happens here.
    """
    if not retained:
        return []
    admit_source(repository, snapshot, limits)
    ids = [item["event_id"] for item in retained]
    if len(set(ids)) != len(ids):
        raise fail("HOLDOUT_ACCESS_DENIED", "roster repeats an event id")
    population = load_population(repository, snapshot, as_of_month=as_of_month, purpose=purpose,
        event_ids=ids, max_object_bytes=limits["max_object_bytes"])
    if set(population.event_id) != set(ids) or len(population) != len(ids):
        raise fail("HOLDOUT_ACCESS_DENIED", "population differs from the planned roster")
    contract = repository.table_contract(snapshot, "computed_moves")
    if contract not in (COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_V5_CONTRACT):
        raise fail("CONTRACT_MISMATCH", "labels require the native computed-moves contract")
    anchors = {item["event_id"]: item for item in retained}
    candidates = population.sort_values(["event_date", "event_id"])
    plans, read_rows = event_history.plan_with_reader(repository, snapshot, candidates, contract)
    for event, _, record in plans:
        item = anchors[event.event_id]
        if (record.object_ref.object_id != item["outcome_object_id"]
                or record.object_ref.content_hash != item["outcome_content_hash"]):
            raise fail("HOLDOUT_ACCESS_DENIED", "planned outcome differs from the roster binding")
    labels = {}
    with stage("season_labels"):
        meter = Progress("season labels", len(plans))
        for event, request, _ in sorted(plans, key=lambda plan: (plan[0].ticker, plan[0].event_date,
                plan[0].event_id)):
            item = anchors[event.event_id]
            abs_move = _label_move(read_rows(event, request), event, item)
            labels[event.event_id] = {"event_id": event.event_id, "ticker": event.ticker,
                "event_date": event.event_date.date().isoformat(),
                "available_as_of_date": item["available_as_of_date"], "abs_move": abs_move}
            meter.tick()
        decision("season_labels", f"{len(labels)} labels from "
            f"{len({event.ticker for event, _, _ in plans})} tickers")
    return [labels[event_id] for event_id in ids]
