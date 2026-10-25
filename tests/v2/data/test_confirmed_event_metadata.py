"""Synthetic metadata-only confirmation, independent of numerical eligibility."""
import numpy as np
import pandas as pd
import pytest

from engine.v2.contracts import Problem
from engine.v2.data import computed_moves
from engine.v2.data.confirmed_event_metadata import classify_metadata
from engine.v2.data.errors import DataError


def frame():
    return pd.DataFrame({"event_date": pd.to_datetime([
        "2024-03-01", "2024-01-01", "2024-01-01", "2024-02-01", "2024-04-01"]),
        "session": ["AMC", "BMO", None, pd.NA, "BMO"],
        "src_orats": [True, False, False, True, np.bool_(True)]}, index=[7, 2, 2, 9, 1])


def assert_refusal(problem, message):
    assert problem == Problem(code="CONTRACT_MISMATCH", category="validation",
                              retryable=False, message=message)


def test_preserves_every_row_order_duplicates_and_input():
    events = frame().assign(unused_payload=[object() for _ in range(5)])
    before = events.copy(deep=True)
    result = classify_metadata(events)
    assert result == (None, "not_orats", "not_orats", "no_session", None)
    assert type(result) is tuple
    pd.testing.assert_frame_equal(events, before)
    events.loc[7, "src_orats"] = False
    assert result[0] is None  # output detached from later input changes


def test_distinct_reasons_share_an_index_and_exact_duplicate_rows_survive():
    events = frame().iloc[[0, 3, 0, 1]].copy()
    events.index = [7, 7, 7, 1]
    assert classify_metadata(events) == (None, "no_session", None, "not_orats")


def test_unused_payload_values_are_not_consumed():
    class Poison:
        def forbidden(self, *args, **kwargs):
            pytest.fail("unused payload was consumed")
        __bool__ = __float__ = __str__ = __iter__ = __eq__ = forbidden
    events = frame().assign(unused_payload=[Poison() for _ in range(5)])
    assert classify_metadata(events) == (None, "not_orats", "not_orats", "no_session", None)


@pytest.mark.parametrize("session", [None, pd.NA, np.nan, "BMO", "AMC"])
@pytest.mark.parametrize("source", [False, True, np.bool_(False), np.bool_(True)])
def test_native_predicate_parity_and_explicit_truth_table(source, session):
    events = frame().iloc[:1].copy()
    events["src_orats"], events["session"] = [source], [session]
    computed_moves._validate_inputs(events, (), (), min_scoreable=0)
    expected = "not_orats" if not source else "no_session" if pd.isna(session) else None
    assert classify_metadata(events) == (expected,) == tuple(computed_moves._event_reasons(events))


def test_delegates_guards_before_predicate_without_population_or_price_work(monkeypatch):
    calls = []
    validate, reasons = computed_moves._validate_inputs, computed_moves._event_reasons
    def guard(events, sd, sc, min_scoreable):
        assert sd == sc == () and min_scoreable == 0
        calls.append("validate")
        return validate(events, sd, sc, min_scoreable)
    def predicate(events):
        calls.append("reasons")
        return reasons(events)
    def forbidden(*args, **kwargs):
        pytest.fail("metadata facade reached population, price or construction work")
    monkeypatch.setattr(computed_moves, "_validate_inputs", guard)
    monkeypatch.setattr(computed_moves, "_event_reasons", predicate)
    for name in ("_collapse_and_order", "_collapse_same_date", "_classify",
                 "_session_move_indexed", "session_move", "_implied_by_event", "_row",
                 "classify_event_reasons", "build_rows", "build_event_rows", "build_ticker"):
        monkeypatch.setattr(computed_moves, name, forbidden)
    assert classify_metadata(frame()) == (None, "not_orats", "not_orats", "no_session", None)
    assert calls == ["validate", "reasons"]


@pytest.mark.parametrize("empty", [False, True])
def test_typed_empty_and_native_timezone_semantics(empty):
    events = frame()
    events["event_date"] = events.event_date.dt.tz_localize("UTC")
    if empty:
        events = events.iloc[:0]
    before = events.copy(deep=True)
    expected = () if empty else (None, "not_orats", "not_orats", "no_session", None)
    assert classify_metadata(events) == expected == tuple(computed_moves._event_reasons(events))
    pd.testing.assert_frame_equal(events, before)


@pytest.mark.parametrize("bad", [None, [], {}, pd.Series([], dtype=object),
    frame().to_dict("list"), frame().to_dict("records")])
def test_non_dataframe_refusal_envelope(bad):
    with pytest.raises(DataError) as caught:
        classify_metadata(bad)
    assert_refusal(caught.value.problem, "event metadata requires a DataFrame")


@pytest.mark.parametrize("bad,message", [
    (pd.DataFrame(), "event metadata columns must be unique and complete"),
    (pd.DataFrame(columns=["event_date", "session", "src_orats"]),
     "events event_date must be a datetime column"),
    (pd.concat([frame(), frame()[["session"]]], axis=1),
     "event metadata columns must be unique and complete"),
])
def test_bad_frame_shape_refusal_envelope(bad, message):
    with pytest.raises(DataError) as caught:
        classify_metadata(bad)
    assert_refusal(caught.value.problem, message)


@pytest.mark.parametrize("column", ["event_date", "session", "src_orats"])
def test_missing_column_refuses_before_native_guard(monkeypatch, column):
    monkeypatch.setattr(computed_moves, "_validate_inputs", lambda *a, **k: pytest.fail("shape bypass"))
    with pytest.raises(DataError) as caught:
        classify_metadata(frame().drop(columns=column))
    assert_refusal(caught.value.problem, "event metadata columns must be unique and complete")


@pytest.mark.parametrize("confirmed", [True, False])
@pytest.mark.parametrize("column,bad", [
    ("event_date", ["2024-01-01"]), ("event_date", [pd.NaT]),
    ("event_date", [pd.Timestamp("2024-01-01 01:00")]),
    ("src_orats", [None]), ("src_orats", [1]), ("src_orats", ["true"]),
    ("session", ["bmo"]), ("session", [""]), ("session", [42]),
])
def test_native_refusal_envelope_is_preserved(column, bad, confirmed):
    events = frame().iloc[:1].copy()
    events["src_orats"] = confirmed
    events[column] = bad
    with pytest.raises(DataError) as native:
        computed_moves._validate_inputs(events, (), (), min_scoreable=0)
    with pytest.raises(DataError) as facade:
        classify_metadata(events)
    assert facade.value.problem == native.value.problem


def test_nonscalar_session_builtin_failure_is_closed_and_redacted():
    events = frame().iloc[:1].copy()
    events["session"] = [["private value", "other"]]
    with pytest.raises(DataError) as caught:
        classify_metadata(events)
    assert_refusal(caught.value.problem, "event metadata has unsupported shape or values")
    assert caught.value.__suppress_context__ and caught.value.__cause__ is None


def test_native_null_like_acceptance_is_not_stricter_source_admission():
    events = frame().iloc[:1].copy()
    events["session"] = [[None]]
    computed_moves._validate_inputs(events, (), (), min_scoreable=0)
    assert classify_metadata(events) == tuple(computed_moves._event_reasons(events)) == ("no_session",)


def test_unhashable_column_label_is_a_closed_shape_refusal():
    events = frame().assign(extra=None)
    events.columns = pd.Index(["event_date", "session", "src_orats", []], tupleize_cols=False)
    with pytest.raises(DataError) as caught:
        classify_metadata(events)
    assert_refusal(caught.value.problem, "event metadata has unsupported shape or values")
