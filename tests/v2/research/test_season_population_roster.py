"""Plain-fake coverage for season_population.plan_roster: metadata-only, window edges, digests."""
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data.errors import DataError
from engine.v2.foundation.market_calendar import CalendarSessions
from engine.v2.research import season_population

LIMITS = {"max_object_bytes": 1 << 20}
SNAPSHOT = SimpleNamespace(snapshot_id="snap_test", table_versions={})
SOURCE = "calendar-source-v1"
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64


def _weekdays(count=300):
    days, day = [], date(2024, 1, 1)
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day.isoformat())
        day += timedelta(days=1)
    return days


DAYS = _weekdays()
START, FIRST, LAST = DAYS[9], DAYS[260], DAYS[299]
CALENDAR = CalendarSessions(tuple(DAYS), DAYS[-1])
SESSIONS = [FIRST, LAST]


def event(event_id, ticker, day, session="AMC"):
    return {"event_id": event_id, "ticker": ticker, "event_date": pd.Timestamp(day), "session": session}


def bounds_row(ticker, day, **changes):
    base = {"ticker": ticker, "event_date": day, "session": "AMC",
        "pre_close_date": (date.fromisoformat(day) - timedelta(days=1)).isoformat(),
        "post_close_date": day, "outcome_object_id": "obj-1", "outcome_content_hash": HASH_A}
    return {**base, **changes}


def record(ticker, day, *, object_id="obj-1", content_hash=HASH_A):
    key = (ticker, day)
    return SimpleNamespace(partition_key=ticker, row_count=1, primary_key_min=key, primary_key_max=key,
        time_min=key[1], time_max=key[1],
        object_ref=SimpleNamespace(object_id=object_id, content_hash=content_hash, byte_size=8))


class Repository:
    def __init__(self, moves):
        self.moves, self.tables, self.scans = moves, [], []

    def fragment_records(self, snapshot, table):
        self.tables.append(table)
        return self.moves

    def scan(self, request, *, table_name, max_object_bytes):
        self.scans.append(table_name)
        if table_name == "computed_moves":
            raise AssertionError("roster opened an outcome payload")
        return iter(())


def plan(monkeypatch, events, anchors, moves, sessions=None, calendar=CALENDAR, purpose="selection",
        limits=LIMITS):
    repository = Repository(moves)
    reads = []

    def load_population(repository, snapshot, **kwargs):
        reads.append(kwargs)
        return pd.DataFrame(events)

    monkeypatch.setattr(season_population, "load_population", load_population)
    monkeypatch.setattr(season_population.outcome_binding, "anchors",
        lambda repository, snapshot, limits, **kwargs: anchors)
    result = season_population.plan_roster(repository, SNAPSHOT, as_of_month="2024-01",
        decision_sessions=SESSIONS if sessions is None else sessions, calendar=calendar,
        calendar_source=SOURCE, limits=limits, purpose=purpose)
    return repository, result, reads


def _mixed():
    saturday = (date.fromisoformat(DAYS[274]) + timedelta(days=1)).isoformat()
    events = [event("ev-ret", "AAA", DAYS[270]), event("ev-miss", "BBB", DAYS[272]),
        event("ev-null", "CCC", DAYS[273]), event("ev-sat", "DDD", saturday)]
    anchors = {("AAA", DAYS[270]): bounds_row("AAA", DAYS[270]),
        ("CCC", DAYS[273]): bounds_row("CCC", DAYS[273], pre_close_date=None, post_close_date=None),
        ("DDD", saturday): bounds_row("DDD", saturday, pre_close_date=DAYS[274], post_close_date=DAYS[275])}
    moves = [record("AAA", DAYS[270]), record("CCC", DAYS[273]), record("DDD", saturday)]
    return events, anchors, moves


def test_retained_exclusions_and_metadata_only_reads(monkeypatch):
    events, anchors, moves = _mixed()
    repository, result, reads = plan(monkeypatch, events, anchors, moves)
    receipt, retained = result["receipt"], result["retained"]
    assert reads == [{"as_of_month": "2024-01", "purpose": "selection",
        "max_object_bytes": LIMITS["max_object_bytes"]}]
    assert (repository.tables, repository.scans) == (["computed_moves"], [])
    assert receipt["schema_version"] == "season_heat_population.v1"
    assert receipt["window"] == {"start": START, "end": LAST, "earliest_decision": FIRST,
        "latest_decision": LAST}
    assert receipt["calendar_source"] == SOURCE and receipt["as_of_month"] == "2024-01"
    assert receipt["counts"] == {"population": 4, "in_window": 4, "retained": 1, "excluded": 3,
        "not_yet_available": 0}
    assert receipt["exclusions"] == {"missing_native_bounds": 1, "null_source_bounds": 1,
        "off_session_event_date": 1}
    assert receipt["sample_ids"] == ["ev-miss", "ev-null", "ev-sat"]
    assert retained == [{"event_id": "ev-ret", "outcome_object_id": "obj-1",
        "outcome_content_hash": HASH_A,
        "available_as_of_date": (date.fromisoformat(DAYS[270]) + timedelta(days=1)).isoformat()}]


def test_window_edges_and_start_clamp(monkeypatch):
    before, on_start, at_last, avail_last = DAYS[8], START, LAST, DAYS[298]
    events = [event("ev-before", "AAA", before), event("ev-start", "BBB", on_start),
        event("ev-last", "CCC", at_last), event("ev-avail", "DDD", avail_last)]
    anchors = {("BBB", on_start): bounds_row("BBB", on_start),
        ("CCC", at_last): bounds_row("CCC", at_last),
        ("DDD", avail_last): bounds_row("DDD", avail_last)}
    moves = [record(t, d) for t, d in anchors]
    _, result, _ = plan(monkeypatch, events, anchors, moves)
    receipt = result["receipt"]
    assert receipt["counts"] == {"population": 4, "in_window": 3, "retained": 2, "excluded": 0,
        "not_yet_available": 1}
    counts = receipt["counts"]
    assert counts["in_window"] == (counts["retained"] + counts["excluded"]
        + counts["not_yet_available"])
    assert [item["event_id"] for item in result["retained"]] == ["ev-avail", "ev-start"]
    assert receipt["exclusions"] == dict.fromkeys(
        ["missing_native_bounds", "null_source_bounds", "off_session_event_date"], 0)
    _, clamped, _ = plan(monkeypatch, events, {}, [], sessions=[DAYS[100], LAST])
    assert clamped["receipt"]["window"]["start"] == DAYS[0]


@pytest.mark.parametrize("index, code", [
    pytest.param("dup", "HOLDOUT_ACCESS_DENIED", id="table-wide-duplicate"),
    pytest.param("object", "HOLDOUT_ACCESS_DENIED", id="object-id-mismatch"),
    pytest.param("null-one", "CONTRACT_MISMATCH", id="one-null-bounds-date"),
])
def test_binding_refusals_propagate(monkeypatch, index, code):
    day = DAYS[270]
    events, anchors = [event("ev-ret", "AAA", day)], {("AAA", day): bounds_row("AAA", day)}
    moves = [record("AAA", day)]
    if index == "dup":
        distant = (date.fromisoformat(DAYS[2]) - timedelta(days=1)).isoformat()
        moves = moves + [record("ZZZ", distant), record("ZZZ", distant)]
    elif index == "object":
        moves = [record("AAA", day, object_id="other")]
    else:
        anchors = {("AAA", day): bounds_row("AAA", day, pre_close_date=None)}
    with pytest.raises(DataError) as caught:
        plan(monkeypatch, events, anchors, moves)
    assert caught.value.code == code


@pytest.mark.parametrize("calendar, sessions", [
    pytest.param(SimpleNamespace(days=tuple(DAYS)), SESSIONS, id="wrong-calendar-type"),
    pytest.param(CALENDAR, [], id="empty-sessions"),
    pytest.param(CALENDAR, [FIRST, DAYS[274] + "Sat"], id="non-date-session"),
    pytest.param(CALENDAR, [FIRST, (date.fromisoformat(DAYS[274]) + timedelta(days=1)).isoformat()],
        id="session-off-calendar"),
])
def test_contract_validation_refusals(monkeypatch, calendar, sessions):
    with pytest.raises(DataError) as caught:
        plan(monkeypatch, [event("ev", "AAA", DAYS[270])], {}, [], calendar=calendar,
            sessions=sessions)
    assert caught.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("limits", [
    pytest.param(None, id="none"),
    pytest.param({}, id="empty-dict"),
    pytest.param({"max_object_bytes": None}, id="null-bytes"),
    pytest.param({"max_object_bytes": 0}, id="zero-bytes"),
    pytest.param({"max_object_bytes": True}, id="bool-bytes"),
    pytest.param({"max_object_bytes": "9"}, id="string-bytes"),
])
def test_limits_refusals_before_any_reads(monkeypatch, limits):
    calls = []
    monkeypatch.setattr(season_population, "load_population",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(season_population.outcome_binding, "anchors",
        lambda *args, **kwargs: pytest.fail("anchors opened before limits check"))
    with pytest.raises(DataError) as caught:
        season_population.plan_roster(Repository([]), SNAPSHOT, as_of_month="2024-01",
            decision_sessions=SESSIONS, calendar=CALENDAR, calendar_source=SOURCE,
            limits=limits, purpose="selection")
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert calls == []


def test_purpose_forwards_to_load_and_receipt(monkeypatch):
    events, anchors, moves = _mixed()
    _, result, reads = plan(monkeypatch, events, anchors, moves, purpose="final")
    assert reads[0]["purpose"] == "final"
    assert result["receipt"]["purpose"] == "final"


def _receipt_digests(monkeypatch, events, anchors, moves):
    _, result, _ = plan(monkeypatch, events, anchors, moves)
    receipt = result["receipt"]
    return receipt["exclusions_digest"], receipt["roster_digest"], receipt["receipt_id"]


def test_digest_stability_and_sensitivity(monkeypatch):
    events, anchors, moves = _mixed()
    baseline = _receipt_digests(monkeypatch, events, anchors, moves)
    again = _receipt_digests(monkeypatch, list(reversed(events)), anchors, moves)
    assert baseline == again
    day = DAYS[270]
    changed = {**anchors, ("AAA", day): {**anchors[("AAA", day)], "outcome_content_hash": HASH_B}}
    changed_moves = [record("AAA", day, content_hash=HASH_B)] + [m for m in moves if m.partition_key != "AAA"]
    shifted = _receipt_digests(monkeypatch, events, changed, changed_moves)
    assert shifted[0] == baseline[0] and shifted[1:] != baseline[1:]
    extra = events + [event("ev-extra", "EEE", DAYS[271])]
    grown = _receipt_digests(monkeypatch, extra, anchors, moves)
    assert grown[0] != baseline[0]


def test_sample_ids_capped_at_ten(monkeypatch):
    events = [event(f"ev-{i:02}", f"T{i:02}", DAYS[265 + i]) for i in range(12)]
    _, result, _ = plan(monkeypatch, events, {}, [])
    receipt = result["receipt"]
    assert receipt["counts"]["excluded"] == 12
    assert receipt["sample_ids"] == [f"ev-{i:02}" for i in range(10)]


def test_purpose_and_identity_reproducible(monkeypatch):
    events, anchors, moves = _mixed()
    _, first, _ = plan(monkeypatch, events, anchors, moves, purpose="holdout")
    _, second, _ = plan(monkeypatch, events, anchors, moves, purpose="holdout")
    assert first["receipt"] == second["receipt"]
    assert first["receipt"]["purpose"] == "holdout"
