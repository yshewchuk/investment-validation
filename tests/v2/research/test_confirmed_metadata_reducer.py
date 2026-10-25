"""Synthetic complete-view reduction; no source or numerical admission."""
import json

import numpy as np
import pandas as pd
import pytest

from engine.v2.contracts import Problem
from engine.v2.data import computed_moves
from engine.v2.data.errors import DataError, fail
from engine.v2.research import _confirmed_metadata as reducer


def frame(rows=None):
    rows = rows if rows is not None else [
        (f"opaque/{i}", "A", f"2001-{i + 1:02d}-01", "BMO", False, None, True)
        for i in range(6)]
    events = pd.DataFrame(rows, columns=reducer._COLUMNS)
    events["event_date"] = pd.to_datetime(events["event_date"]).astype("datetime64[ns]")
    return events


def refused(events, code="CONTRACT_MISMATCH", tickers=None):
    with pytest.raises(DataError) as caught:
        reducer.reduce_sequences(events, tickers=["A"] if tickers is None else tickers)
    assert caught.value.code == code
    assert caught.value.problem.retryable is False


def test_exact_states_evidence_warmup_order_and_detachment():
    events = pd.concat([frame(), frame([
        ("old", "A", "1990-01-01", None, False, None, True),
        ("b", "B", "2000-01-01", "AMC", False, "same", False),
        ("c", "C", "2000-01-01", "AMC", False, "same", True)])])
    before = events.copy(deep=True)
    result = reducer.reduce_sequences(events.iloc[::-1], tickers=["Z", "C", "B", "A"])
    expected = []
    for i, row in enumerate(frame().to_dict("records")):
        row["event_date"] = row["event_date"].date().isoformat()
        expected.append(dict(row, ordinal=i, warmup_complete=i >= 4))
    assert result[0] == {"ticker": "A", "status": "confirmed_events", "selected_rows": 7,
        "confirmed_rows": 6, "first_confirmed_date": "2001-01-01", "events": expected}
    assert result[1] == {"ticker": "B", "status": "no_confirmed_events", "selected_rows": 1,
        "confirmed_rows": 0, "first_confirmed_date": None, "events": []}
    assert result[2] == {"ticker": "C", "status": "confirmed_events", "selected_rows": 1,
        "confirmed_rows": 1, "first_confirmed_date": "2000-01-01", "events": [dict(
            zip((*reducer._COLUMNS, "ordinal", "warmup_complete"),
                ("c", "C", "2000-01-01", "AMC", False, "same", True, 0, False)))]}
    assert result[3] == {"ticker": "Z", "status": "absent_in_snapshot", "selected_rows": 0,
        "confirmed_rows": 0, "first_confirmed_date": None, "events": []}
    assert result == json.loads(json.dumps(result, allow_nan=False))
    for sequence in result:
        assert all(type(sequence[key]) is int for key in ("selected_rows", "confirmed_rows"))
        assert all(type(sequence[key]) is str for key in ("ticker", "status"))
        assert sequence["first_confirmed_date"] is None or type(sequence["first_confirmed_date"]) is str
        for event in sequence["events"]:
            assert type(event["ordinal"]) is int
            assert all(type(event[key]) is bool for key in ("src_orats", "date_conflict", "warmup_complete"))
            assert all(type(event[key]) is str for key in ("event_id", "ticker", "event_date", "session"))
            assert event["event_cluster_id"] is None or type(event["event_cluster_id"]) is str
    assert result == reducer.reduce_sequences(events, tickers=["A", "B", "C", "Z"])
    pd.testing.assert_frame_equal(events, before)
    result_before = json.loads(json.dumps(result))
    events.iloc[0, 0] = "different"
    assert result == result_before
    result[0]["events"][0]["event_id"] = "changed"
    assert events.iloc[0, 0] == "different"


def test_older_confirmed_rows_count_without_a_date_cutoff():
    events = pd.concat([frame(), frame([("older", "A", "1980-01-01", "AMC", False, None, True)])])
    result = reducer.reduce_sequences(events, tickers=["A"])[0]
    assert result["first_confirmed_date"] == "1980-01-01"
    assert result["selected_rows"] == result["confirmed_rows"] == 7
    assert [row["event_id"] for row in result["events"]] == ["older", *(f"opaque/{i}" for i in range(6))]
    assert [row["ordinal"] for row in result["events"]] == list(range(7))
    assert [row["warmup_complete"] for row in result["events"]] == [False] * 4 + [True] * 3


def test_public_primitives_once_per_call_and_ticker(monkeypatch):
    calls = []
    native_classify, native_sequence = reducer.classify_metadata, reducer.event_sequence
    def classify(events):
        calls.append(("classify", len(events)))
        return native_classify(events)
    def sequence(days, *, warmup):
        calls.append(("sequence", list(days), warmup))
        return native_sequence(days, warmup=warmup)
    monkeypatch.setattr(reducer, "classify_metadata", classify)
    monkeypatch.setattr(reducer, "event_sequence", sequence)
    assert reducer.reduce_sequences(frame([]), tickers=["Z", "A"]) == [
        {"ticker": t, "status": "absent_in_snapshot", "selected_rows": 0,
         "confirmed_rows": 0, "first_confirmed_date": None, "events": []} for t in ["A", "Z"]]
    assert calls == [("classify", 0), ("sequence", [], 4), ("sequence", [], 4)]
    calls.clear()
    reducer.reduce_sequences(frame().iloc[::-1], tickers=["B", "A"])
    assert calls == [("classify", 6),
        ("sequence", [f"2001-{i + 1:02d}-01" for i in range(6)], 4), ("sequence", [], 4)]
    monkeypatch.setattr(reducer, "classify_metadata", lambda events: ("not_orats",) * len(events))
    assert reducer.reduce_sequences(frame(), tickers=["A"])[0]["confirmed_rows"] == 0


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("change", [
    {"event_id": "opaque/0"}, {"event_date": pd.Timestamp("2001-01-01")},
    {"event_cluster_id": "cluster"}, {"date_conflict": True},
    {"event_id": "opaque/0", "ticker": "B"},
])
def test_conflicts_include_unconfirmed_siblings_in_either_order(change, reverse):
    events = frame().iloc[:2].copy()
    events["event_cluster_id"] = ["cluster", None]
    events.loc[1, "src_orats"] = False
    for column, value in change.items():
        events.loc[1, column] = value
    refused(events.iloc[::-1] if reverse else events, "IDENTITY_CONFLICT", ["A", "B"])


def test_exact_repeats_refuse_and_cross_ticker_clusters_are_independent():
    events = frame().iloc[:1]
    refused(pd.concat([events, events]), "IDENTITY_CONFLICT")
    events = frame().iloc[:2].copy()
    events["ticker"], events["event_cluster_id"] = ["A", "B"], ["opaque", "opaque"]
    assert [s["confirmed_rows"] for s in reducer.reduce_sequences(events, tickers=["B", "A"])] == [1, 1]


@pytest.mark.parametrize("tickers", [[], (), ("A",), "A", {"A"}, iter(["A"]), ["A", "A"],
    [None], [1], [""], [" A"], ["A "], ["A\0"], ["\ud800"], [np.str_("A")]])
def test_bad_requests_refuse_before_frame_access(tickers, monkeypatch):
    monkeypatch.setattr(reducer, "_rows", lambda *args: pytest.fail("invalid request reached metadata"))
    refused(frame(), tickers=tickers)


@pytest.mark.parametrize("bad", [None, [], {}, pd.DataFrame(),
    frame().drop(columns="ticker"), frame().assign(extra=1),
    pd.concat([frame(), frame()[["session"]]], axis=1)])
def test_bad_frame_shapes(bad):
    refused(bad)


@pytest.mark.parametrize("column,value", [
    ("event_id", ""), ("event_id", None), ("event_id", 1), ("event_id", "\ud800"),
    ("ticker", "B"), ("ticker", None), ("ticker", " A"),
    ("event_cluster_id", ""), ("event_cluster_id", [None]), ("event_cluster_id", 1),
    ("event_cluster_id", "\ud800"), ("date_conflict", 1), ("date_conflict", None),
    ("event_cluster_id", 0.0), ("event_cluster_id", np.inf), ("event_cluster_id", -np.inf),
    ("date_conflict", "false"), ("src_orats", 1), ("src_orats", "true"),
    ("src_orats", None), ("session", "bmo"), ("session", ""),
    ("session", [None]), ("session", 1), ("session", pd.NaT),
    ("event_date", "2001-01-01"), ("event_date", pd.NaT),
    ("event_date", pd.Timestamp("2001-01-01 00:00:00.000000001")),
    ("event_date", pd.Timestamp("2001-01-01", tz="UTC")),
])
def test_malformed_rows_refuse_even_when_unconfirmed(column, value):
    events = frame().iloc[:1].copy()
    events["src_orats"] = False
    events[column] = pd.Series([value], dtype=object) if column != "event_date" else [value]
    refused(events)


@pytest.mark.parametrize("dtype", ["datetime64[ms]", "datetime64[ns, UTC]", "object"])
@pytest.mark.parametrize("empty", [False, True])
def test_source_date_types_never_parse_or_convert(dtype, empty):
    events = frame().iloc[:0 if empty else 1].copy()
    events["event_date"] = pd.Series([pd.Timestamp("2001-01-01")] * len(events), dtype=dtype)
    refused(events)


@pytest.mark.parametrize("null", [None, pd.NA, np.nan])
def test_nulls_real_booleans_column_order_and_opaque_ids(null):
    events = frame([(" arbitrary ID ", "A", "2000-01-01", "AMC", np.bool_(False), null, np.bool_(True)),
                    ("other", "A", "2001-01-01", null, False, null, True)])
    result = reducer.reduce_sequences(events[events.columns[::-1]], tickers=["A"])[0]
    assert (result["selected_rows"], result["confirmed_rows"]) == (2, 1)
    assert result["events"][0]["event_id"] == " arbitrary ID "
    assert result["events"][0]["event_cluster_id"] is None


@pytest.mark.parametrize("helper", ["classify_metadata", "event_sequence"])
@pytest.mark.parametrize("code,category,retryable", [
    ("MANIFEST_CORRUPT", "integrity", False), ("TRANSIENT_SOURCE", "source", True)])
def test_native_dataerror_propagates_unchanged_without_partial_output(monkeypatch, helper, code, category, retryable):
    error = fail(code, "native refusal", stage="native", details={"probe": "retained"})
    def stop(*args, **kwargs):
        raise error
    monkeypatch.setattr(reducer, helper, stop)
    events = frame()
    before = events.copy(deep=True)
    with pytest.raises(DataError) as caught:
        reducer.reduce_sequences(events, tickers=["A", "B"])
    assert caught.value is error
    assert caught.value.problem == Problem(code=code, category=category, retryable=retryable,
        message="native refusal", stage="native", details={"probe": "retained"})
    pd.testing.assert_frame_equal(events, before)


@pytest.mark.parametrize("column,value", [("event_date", pd.NaT),
    ("event_date", pd.Timestamp("2001-01-01 00:00:00.000000001")), ("src_orats", 1), ("session", "private")])
def test_native_validation_envelopes_match_the_public_facade(column, value):
    events = frame().iloc[:1].copy()
    events[column] = pd.Series([value], dtype="datetime64[ns]" if column == "event_date" else object)
    with pytest.raises(DataError) as native:
        reducer.classify_metadata(events)
    with pytest.raises(DataError) as caught:
        reducer.reduce_sequences(events, tickers=["A"])
    assert caught.value.problem == native.value.problem


@pytest.mark.parametrize("events,code,message", [
    (None, "CONTRACT_MISMATCH", "metadata requires a DataFrame"),
    (frame().assign(ticker="private row value"), "CONTRACT_MISMATCH", "metadata ticker is outside the request"),
    (pd.concat([frame(), frame()]), "IDENTITY_CONFLICT", "metadata view has conflicting event identities"),
])
def test_complete_refusal_envelopes(events, code, message):
    with pytest.raises(DataError) as caught:
        reducer.reduce_sequences(events, tickers=["A"])
    assert caught.value.problem == Problem(code=code, category="validation", retryable=False,
                                          message=message, stage=None, details={})


@pytest.mark.parametrize("tickers,change,message", [
    ([], {}, "tickers must be a nonempty plain list"),
    ([None], {}, "ticker must be a nonempty string"),
    (["\ud800"], {}, "ticker must be valid Unicode"),
    (["private "], {}, "ticker names must be canonical"),
    (["A", "A"], {}, "ticker names must be distinct"),
    (["A"], {"extra": 1}, "metadata requires exactly seven unique columns"),
    (["A"], {"event_date": "private"}, "metadata dates require naive datetime64[ns]"),
    (["A"], {"session": ["private"]}, "metadata session must be scalar text or null"),
    (["A"], {"event_id": 1}, "metadata text must be a nonempty string"),
    (["A"], {"event_id": "\ud800"}, "metadata text must be valid Unicode"),
    (["A"], {"date_conflict": "private"}, "metadata date_conflict must be a bool"),
])
def test_remaining_local_refusal_envelopes(tickers, change, message):
    events = frame().iloc[:1].copy()
    for column, value in change.items():
        events[column] = pd.Series([value], dtype=object)
    with pytest.raises(DataError) as caught:
        reducer.reduce_sequences(events, tickers=tickers)
    assert caught.value.problem == Problem(code="CONTRACT_MISMATCH", category="validation",
        retryable=False, message=message, stage=None, details={})


def test_no_price_population_or_io_hooks(monkeypatch):
    from engine.v2.data.repository import Repository
    def forbidden(*args, **kwargs):
        pytest.fail("pure metadata reducer reached I/O, population or price work")
    for name in ("_collapse_and_order", "_classify", "_session_move_indexed", "session_move",
                 "_implied_by_event", "classify_event_reasons", "build_rows", "build_event_rows", "build_ticker"):
        monkeypatch.setattr(computed_moves, name, forbidden)
    for name in ("scan", "resolve", "resolve_full"):
        monkeypatch.setattr(Repository, name, forbidden)
    assert reducer.reduce_sequences(frame(), tickers=["A"])[0]["confirmed_rows"] == 6
