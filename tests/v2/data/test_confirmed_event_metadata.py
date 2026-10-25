"""Synthetic metadata-only confirmation, independent of numerical eligibility."""
import numpy as np
import pandas as pd
import pytest

from engine.v2.data import computed_moves
from engine.v2.data.confirmed_event_metadata import classify_metadata
from engine.v2.data.errors import DataError


def frame():
    return pd.DataFrame({"event_date": pd.to_datetime([
        "2024-03-01", "2024-01-01", "2024-01-01", "2024-02-01", "2024-04-01"]),
        "session": ["AMC", "BMO", None, pd.NA, "BMO"],
        "src_orats": [True, False, False, True, np.bool_(True)]}, index=[7, 2, 2, 9, 1])


def test_preserves_every_row_order_duplicates_and_input():
    events = frame().assign(unused_payload=[object() for _ in range(5)])
    before = events.copy(deep=True)
    result = classify_metadata(events)
    assert result == (None, "not_orats", "not_orats", "no_session", None)
    assert type(result) is tuple
    pd.testing.assert_frame_equal(events, before)
    events.loc[7, "src_orats"] = False
    assert result[0] is None  # output detached from later input changes


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
                 "_session_move_indexed", "classify_event_reasons", "build_rows", "build_event_rows"):
        monkeypatch.setattr(computed_moves, name, forbidden)
    assert classify_metadata(frame()) == (None, "not_orats", "not_orats", "no_session", None)
    assert calls == ["validate", "reasons"]


@pytest.mark.parametrize("empty", [False, True])
def test_typed_empty_and_native_timezone_semantics(empty):
    events = frame()
    events["event_date"] = events.event_date.dt.tz_localize("UTC")
    if empty:
        events = events.iloc[:0]
    assert classify_metadata(events) == tuple(computed_moves._event_reasons(events))
    assert len(classify_metadata(events)) == len(events)


@pytest.mark.parametrize("bad", [None, [], {}, pd.Series([], dtype=object),
    pd.DataFrame(), pd.DataFrame(columns=["event_date", "session", "src_orats"]),
    pd.concat([frame(), frame()[["session"]]], axis=1)])
def test_bad_frame_shape_refuses_without_partial_result(bad):
    with pytest.raises(DataError) as caught:
        classify_metadata(bad)
    assert (caught.value.code, caught.value.problem.retryable) == ("CONTRACT_MISMATCH", False)


@pytest.mark.parametrize("column", ["event_date", "session", "src_orats"])
def test_missing_column_refuses_before_native_guard(monkeypatch, column):
    monkeypatch.setattr(computed_moves, "_validate_inputs", lambda *a, **k: pytest.fail("shape bypass"))
    with pytest.raises(DataError, match="columns must be unique and complete"):
        classify_metadata(frame().drop(columns=column))


@pytest.mark.parametrize("column,bad", [
    ("event_date", ["2024-01-01"]), ("event_date", [pd.NaT]),
    ("event_date", [pd.Timestamp("2024-01-01 01:00")]),
    ("src_orats", [None]), ("src_orats", [1]), ("src_orats", ["true"]),
    ("session", ["bmo"]), ("session", [""]), ("session", [42]),
])
def test_native_refusal_envelope_is_preserved(column, bad):
    events = frame().iloc[:1].copy()
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
    assert caught.value.problem.message == "event metadata has unsupported shape or values"
    assert caught.value.problem.details == {} and caught.value.code == "CONTRACT_MISMATCH"


def test_native_null_like_acceptance_is_not_stricter_source_admission():
    events = frame().iloc[:1].copy()
    events["session"] = [[None]]
    computed_moves._validate_inputs(events, (), (), min_scoreable=0)
    assert classify_metadata(events) == tuple(computed_moves._event_reasons(events)) == ("no_session",)


def test_unhashable_column_label_is_a_closed_shape_refusal():
    events = frame().assign(extra=None)
    events.columns = pd.Index(["event_date", "session", "src_orats", []], tupleize_cols=False)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        classify_metadata(events)
