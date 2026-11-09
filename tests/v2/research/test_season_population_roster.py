"""Plain-fake coverage for season_population.plan_roster: metadata-only, window edges, digests."""
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data.errors import DataError, fail
from engine.v2.foundation import content_hash
from engine.v2.foundation.market_calendar import CalendarSessions
from engine.v2.research import eligible_history, season_population

LIMITS = {"max_source_rows": 1 << 20, "max_fragments": 1 << 20, "max_object_bytes": 1 << 20,
    "max_total_bytes": 1 << 20, "max_output_rows": 1 << 20}
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


def _available(day):
    return (date.fromisoformat(day) + timedelta(days=1)).isoformat()


class Repository:
    def __init__(self, moves):
        self.moves, self.tables, self.scans, self.anchor_limits = moves, [], [], []

    def fragment_records(self, snapshot, table):
        self.tables.append(table)
        return self.moves

    def scan(self, request, *, table_name, max_object_bytes):
        self.scans.append(table_name)
        if table_name == "computed_moves":
            raise AssertionError("roster opened an outcome payload")
        return iter(())


_DEFAULT = object()


def plan(monkeypatch, events, anchors, moves, sessions=_DEFAULT, calendar=CALENDAR, purpose="selection",
        limits=LIMITS):
    repository = Repository(moves)
    reads, calls = [], []
    monkeypatch.setattr(season_population, "admit_source",
        lambda repository, snapshot, limits: calls.append(limits))

    def load_population(repository, snapshot, **kwargs):
        reads.append(kwargs)
        return pd.DataFrame(events)

    monkeypatch.setattr(season_population, "load_population", load_population)

    def anchors_stub(repository, snapshot, limits, **kwargs):
        repository.anchor_limits.append(limits)
        return anchors

    monkeypatch.setattr(season_population.outcome_binding, "anchors", anchors_stub)
    result = season_population.plan_roster(repository, SNAPSHOT, as_of_month="2024-01",
        decision_sessions=SESSIONS if sessions is _DEFAULT else sessions, calendar=calendar,
        calendar_source=SOURCE, limits=limits, purpose=purpose)
    return repository, result, reads, calls


def _mixed():
    saturday = (date.fromisoformat(DAYS[274]) + timedelta(days=1)).isoformat()
    events = [event("ev-ret", "AAA", DAYS[270]), event("ev-miss", "BBB", DAYS[272]),
        event("ev-null", "CCC", DAYS[273]), event("ev-sat", "DDD", saturday)]
    anchors = {("AAA", DAYS[270]): bounds_row("AAA", DAYS[270]),
        ("CCC", DAYS[273]): bounds_row("CCC", DAYS[273], pre_close_date=None, post_close_date=None),
        ("DDD", saturday): bounds_row("DDD", saturday, pre_close_date=DAYS[274], post_close_date=DAYS[275])}
    moves = [record("AAA", DAYS[270]), record("CCC", DAYS[273]), record("DDD", saturday)]
    return events, anchors, moves


def _retained(count):
    """Count population events that all survive into the roster (none excluded)."""
    events, anchors, moves = [], {}, []
    for i in range(count):
        day, ticker = DAYS[270 + i], f"T{i:02}"
        events.append(event(f"ev-{i:02}", ticker, day))
        anchors[(ticker, day)] = bounds_row(ticker, day)
        moves.append(record(ticker, day))
    return events, anchors, moves


def _receipt_id_oracle(receipt):
    assert receipt["receipt_id"] == content_hash(
        {key: value for key, value in receipt.items() if key != "receipt_id"})


def _digest_oracles(receipt, roster, exclusion_pairs):
    assert receipt["roster_digest"] == content_hash(roster)
    assert receipt["exclusions_digest"] == content_hash(exclusion_pairs)
    _receipt_id_oracle(receipt)


def test_retained_exclusions_and_metadata_only_reads(monkeypatch):
    events, anchors, moves = _mixed()
    repository, result, reads, calls = plan(monkeypatch, events, anchors, moves)
    receipt, retained = result["receipt"], result["retained"]
    assert calls == [LIMITS]
    assert repository.anchor_limits == [LIMITS]
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
    available = _available(DAYS[270])
    assert retained == [{"event_id": "ev-ret", "outcome_object_id": "obj-1",
        "outcome_content_hash": HASH_A, "available_as_of_date": available}]
    _digest_oracles(receipt, [["ev-ret", "obj-1", HASH_A, available]],
        sorted([["missing_native_bounds", "ev-miss"], ["null_source_bounds", "ev-null"],
            ["off_session_event_date", "ev-sat"]]))


def test_window_edges_and_start_clamp(monkeypatch):
    before, on_start, at_last, avail_last = DAYS[8], START, LAST, DAYS[298]
    events = [event("ev-before", "AAA", before), event("ev-start", "BBB", on_start),
        event("ev-last", "CCC", at_last), event("ev-avail", "DDD", avail_last)]
    anchors = {("BBB", on_start): bounds_row("BBB", on_start),
        ("CCC", at_last): bounds_row("CCC", at_last),
        ("DDD", avail_last): bounds_row("DDD", avail_last)}
    moves = [record(t, d) for t, d in anchors]
    _, result, _, _ = plan(monkeypatch, events, anchors, moves)
    receipt = result["receipt"]
    assert receipt["counts"] == {"population": 4, "in_window": 3, "retained": 2, "excluded": 0,
        "not_yet_available": 1}
    counts = receipt["counts"]
    assert counts["in_window"] == 3
    assert counts["retained"] == 2
    assert counts["excluded"] == 0
    assert counts["not_yet_available"] == 1
    assert [item["event_id"] for item in result["retained"]] == ["ev-avail", "ev-start"]
    assert receipt["exclusions"] == dict.fromkeys(
        ["missing_native_bounds", "null_source_bounds", "off_session_event_date"], 0)
    _digest_oracles(receipt, [["ev-avail", "obj-1", HASH_A, _available(avail_last)],
        ["ev-start", "obj-1", HASH_A, _available(on_start)]], [])
    _, clamped, _, _ = plan(monkeypatch, events, {}, [], sessions=[DAYS[100], LAST])
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
    pytest.param(CALENDAR, None, id="none-sessions"),
    pytest.param(CALENDAR, 5, id="non-iterable-sessions"),
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
    pytest.param({k: v for k, v in LIMITS.items() if k != "max_fragments"}, id="missing-key"),
    pytest.param({**LIMITS, "max_object_bytes": None}, id="null-bytes"),
    pytest.param({**LIMITS, "max_object_bytes": 0}, id="zero-bytes"),
    pytest.param({**LIMITS, "max_object_bytes": True}, id="bool-bytes"),
    pytest.param({**LIMITS, "max_object_bytes": "9"}, id="string-bytes"),
    pytest.param({**LIMITS, "extra": 1}, id="extra-key"),
])
def test_limits_refusals_before_any_reads(monkeypatch, limits):
    calls = []
    monkeypatch.setattr(season_population, "admit_source", eligible_history.admit_source)
    monkeypatch.setattr(season_population, "load_population",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(season_population.outcome_binding, "anchors",
        lambda *args, **kwargs: pytest.fail("anchors opened before limits check"))
    with pytest.raises(DataError) as caught:
        season_population.plan_roster(SimpleNamespace(), SNAPSHOT, as_of_month="2024-01",
            decision_sessions=SESSIONS, calendar=CALENDAR, calendar_source=SOURCE,
            limits=limits, purpose="selection")
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert calls == []


def test_result_limit_exceeded_propagates_before_reads(monkeypatch):
    def refuse(repository, snapshot, limits):
        raise fail("RESULT_LIMIT_EXCEEDED", "history source exceeds explicit bounds")

    calls = []
    monkeypatch.setattr(season_population, "admit_source", refuse)
    monkeypatch.setattr(season_population, "load_population",
        lambda *args, **kwargs: calls.append("population"))
    monkeypatch.setattr(season_population.outcome_binding, "anchors",
        lambda *args, **kwargs: calls.append("anchors"))
    monkeypatch.setattr(season_population.outcome_binding, "fragment_index",
        lambda *args, **kwargs: calls.append("fragments"))
    with pytest.raises(DataError) as caught:
        season_population.plan_roster(SimpleNamespace(), SNAPSHOT, as_of_month="2024-01",
            decision_sessions=SESSIONS, calendar=CALENDAR, calendar_source=SOURCE,
            limits=LIMITS, purpose="selection")
    assert caught.value.code == "RESULT_LIMIT_EXCEEDED"
    assert calls == []


def test_purpose_forwards_to_load_and_receipt(monkeypatch):
    events, anchors, moves = _mixed()
    _, result, reads, _ = plan(monkeypatch, events, anchors, moves, purpose="training")
    assert reads[0]["purpose"] == "training"
    assert result["receipt"]["purpose"] == "training"


def _receipt_digests(monkeypatch, events, anchors, moves):
    _, result, _, _ = plan(monkeypatch, events, anchors, moves)
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
    _, retrained, _, _ = plan(monkeypatch, events, anchors, moves, purpose="training")
    assert retrained["receipt"]["purpose"] == "training"
    assert retrained["receipt"]["exclusions_digest"] == baseline[0]
    assert retrained["receipt"]["roster_digest"] == baseline[1]
    assert retrained["receipt"]["receipt_id"] != baseline[2]


def test_sample_ids_capped_at_ten(monkeypatch):
    events = [event(f"ev-{i:02}", f"T{i:02}", DAYS[265 + i]) for i in range(12)]
    _, result, _, _ = plan(monkeypatch, events, {}, [])
    receipt = result["receipt"]
    assert receipt["counts"]["excluded"] == 12
    assert receipt["sample_ids"] == [f"ev-{i:02}" for i in range(10)]


def test_purpose_and_identity_reproducible(monkeypatch):
    events, anchors, moves = _mixed()
    _, first, _, _ = plan(monkeypatch, events, anchors, moves, purpose="training")
    _, second, _, _ = plan(monkeypatch, events, anchors, moves, purpose="training")
    assert first["receipt"] == second["receipt"]
    assert first["receipt"]["purpose"] == "training"
    _receipt_id_oracle(first["receipt"])


@pytest.mark.parametrize("retained, cap", [
    pytest.param(2, 2, id="at-cap"),
    pytest.param(3, 2, id="over-cap"),
])
def test_max_output_rows_bounds_retained(monkeypatch, retained, cap):
    events, anchors, moves = _retained(retained)
    limits = {**LIMITS, "max_output_rows": cap}
    if retained <= cap:
        repository, result, _, _ = plan(monkeypatch, events, anchors, moves, limits=limits)
        assert result["receipt"]["counts"]["retained"] == retained
        assert len(result["retained"]) == retained
        assert repository.anchor_limits == [limits]
    else:
        with pytest.raises(DataError) as caught:
            plan(monkeypatch, events, anchors, moves, limits=limits)
        assert caught.value.code == "RESULT_LIMIT_EXCEEDED"
