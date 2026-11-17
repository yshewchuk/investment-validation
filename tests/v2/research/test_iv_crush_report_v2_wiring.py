"""Unit tests for report-only implied input collection in the IV crush search report."""
from experiments.iv_crush_search_report import _implied_inputs


def _fold():
    return {"evaluation_event_ids": ["e1", "e2"]}


def _inputs():
    return {"features": [
        {"event_id": "e1", "horizon_sessions": 1, "iv30": 50., "exern_iv30": 40., "other": 9},
        {"event_id": "e1", "horizon_sessions": 3, "iv30": None, "exern_iv30": 41.},
        {"event_id": "e1", "horizon_sessions": 5, "iv30": 52., "exern_iv30": 42.},
        {"event_id": "e2", "horizon_sessions": 1, "iv30": 60., "exern_iv30": 0.},
        {"event_id": "t1", "horizon_sessions": 1, "iv30": 70., "exern_iv30": 30.},
    ]}


def test_implied_inputs_selects_evaluation_events_and_horizons_only():
    sink = {}
    _implied_inputs(_fold(), _inputs(), (1, 3), sink)
    assert set(sink) == {("e1", 1), ("e1", 3), ("e2", 1)}


def test_implied_inputs_ignores_training_rows():
    sink = {}
    _implied_inputs(_fold(), _inputs(), (1, 3), sink)
    assert not any(event == "t1" for event, _ in sink)


def test_implied_inputs_copies_only_the_two_fields_and_preserves_none():
    sink = {}
    _implied_inputs(_fold(), _inputs(), (1, 3), sink)
    assert sink[("e1", 1)] == {"iv30": 50., "exern_iv30": 40.}
    assert sink[("e1", 3)] == {"iv30": None, "exern_iv30": 41.}
    assert sink[("e2", 1)] == {"iv30": 60., "exern_iv30": 0.}


def test_implied_inputs_accumulates_across_calls_without_clearing():
    sink = {("prior", 1): {"iv30": 1., "exern_iv30": 2.}}
    _implied_inputs(_fold(), _inputs(), (5,), sink)
    assert set(sink) == {("prior", 1), ("e1", 5)}
    assert sink[("e1", 5)] == {"iv30": 52., "exern_iv30": 42.}
