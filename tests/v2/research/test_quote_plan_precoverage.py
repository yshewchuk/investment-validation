"""quote_plan: events before a ticker's first price date are excluded; other bracket misses refuse."""
import pytest

from engine.v2.data.errors import DataError
from engine.v2.research import _magnitude_panel_sources as sources

REQUEST = {"history_start": "2018-01-01", "as_of_month": "2026-01", "policy": {"policy_id": "p"}}
CONTEXT = {"targets": [{"ticker": "FOX", "forecasts": [{"source_session_ceiling": "2030-01-01"}]}]}


def _event(event_id, day, ordinal=10):
    return {"event_id": event_id, "ticker": "FOX", "event_date": day, "session": "BMO", "ordinal": ordinal}


def _run(monkeypatch, days, events):
    seen = {}
    monkeypatch.setattr(sources, "calendar_input", lambda repository, source: (None, None))
    monkeypatch.setattr(sources, "forecast_cutoff", lambda ts, session, h, cal: {
        "horizon_sessions": h, "source_session_ceiling": "2029-01-01"})
    monkeypatch.setattr(sources.eligibility_policy, "plan_eligibility",
        lambda repository, source, **kw: seen.setdefault("metadata", kw["metadata"]) and "plan")
    index = {"sequences": [{"ticker": "FOX", "events": events}]}
    meta = {"FOX": [{"date": d} for d in days], "^GSPC": []}
    before = repr(index)
    plan, excluded = sources.quote_plan(None, {}, index, CONTEXT, REQUEST, meta)
    assert plan == "plan"  # quote_plan returns the eligibility plan unchanged as its first element
    assert repr(index) == before  # the confirmed-metadata index is never mutated
    return plan, excluded, seen["metadata"]


DAYS = ["2019-03-13", "2019-03-14", "2019-03-15", "2019-03-18", "2019-03-19"]


def test_pre_coverage_event_is_excluded_and_counted(monkeypatch, capsys):
    events = [_event("fox-early", "2019-02-06"), _event("fox-ok", "2019-03-14")]
    _, excluded, metadata = _run(monkeypatch, DAYS, events)
    assert excluded == ["fox-early"]
    assert [g["event_id"] for g in metadata["groups"]] == ["fox-ok"]
    assert "before_price_coverage" in capsys.readouterr().err


def test_event_on_first_price_date_is_kept(monkeypatch):
    try:
        # AMC brackets the event day and the next session, so a first-date event has a bracket
        _, excluded, metadata = _run(monkeypatch, DAYS, [{**_event("fox-first", "2019-03-13"), "session": "AMC"}])
    except DataError:
        pytest.fail("an event on the first price date must not be excluded as pre-coverage")
    assert excluded == []
    assert [g["event_id"] for g in metadata["groups"]] == ["fox-first"]


def test_gap_inside_range_still_refuses(monkeypatch):
    days = ["2019-03-13", "2019-03-14", "2019-04-30", "2019-05-01"]
    with pytest.raises(DataError) as err:
        _run(monkeypatch, days, [_event("fox-gap", "2019-04-01")])
    assert err.value.code == "CONTRACT_MISMATCH"


def test_event_after_last_date_still_refuses(monkeypatch):
    with pytest.raises(DataError) as err:
        _run(monkeypatch, DAYS, [_event("fox-late", "2019-06-03")])
    assert err.value.code == "CONTRACT_MISMATCH"


def test_context_document_shape_and_absence_when_empty():
    assert sources.precoverage_document([]) == {}
    assert sources.precoverage_document(["a", "b"]) == {"quote_precoverage_excluded": {
        "reason": "before_price_coverage", "count": 2, "event_ids": ["a", "b"]}}


def test_multiple_pre_coverage_events_are_returned_sorted(monkeypatch):
    events = [_event("z-early", "2019-01-02"), _event("a-early", "2019-02-06")]
    _, excluded, metadata = _run(monkeypatch, DAYS, events)
    assert excluded == ["a-early", "z-early"] and metadata["groups"] == []


def test_bmo_event_on_first_price_date_still_refuses(monkeypatch):
    with pytest.raises(DataError) as err:
        _run(monkeypatch, DAYS, [_event("fox-bmo-first", "2019-03-13")])
    assert err.value.code == "CONTRACT_MISMATCH"
