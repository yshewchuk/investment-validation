"""Value-build label reads: roster binding, contradictions, isolation and run-batched ticker-major reads."""
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data.computed_move_bounds import CONTRACT as BOUNDS_CONTRACT
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT
from engine.v2.data.errors import DataError
from engine.v2.research import eligible_history, event_history, season_population
from tests.v2.research.magnitude_trial_support import history_event, outcome_move, poison_outcomes
from tests.v2.research.test_event_history import MONTH, commit_event_history
from tests.v2.research.test_season_population_roster import (
    CALENDAR,
    DAYS as ROSTER_DAYS,
    FIRST,
    LAST,
    LIMITS,
    bounds_row,
    plan,
    record,
)

PROTECTED = "2023-04-04"
DAYS = ("2023-01-03", "2023-02-07", "2023-03-07", "2023-05-02", "2023-06-06")
TICKERS = ("AAA", "BBB", "CCC", "DDD", "EEE")
# Native bounds rows in place of a bounds table the fixtures never commit: the source
# fixture's make fills (ticker, event day) -> {"post_close_date": availability - 1 day}
# and the autouse fixture serves this dict to read_labels through outcome_binding.anchors.
BOUNDS = {}


def _spec(row):
    day, ticker, value = row[:3]
    return day, ticker, value, (row[3] if len(row) > 3 else {}), (row[4] if len(row) > 4 else False)


def _items(repository, snapshot, events, moves):
    records = {record_.primary_key_min: record_
               for record_ in repository.fragment_records(snapshot, "computed_moves")}
    by_key = {(move["ticker"], move["event_date"]): move for move in moves}
    items = []
    for event in events:
        key = (event["ticker"], event["event_date"].date().isoformat())
        items.append({"event_id": event["event_id"], "outcome_object_id": records[key].object_ref.object_id,
            "outcome_content_hash": records[key].object_ref.content_hash,
            "available_as_of_date": by_key[key]["available_as_of_date"]})
    return items


@pytest.fixture
def source(tmp_path):
    """Real tracked snapshot plus the metadata roster read_labels is given."""
    conns = []

    def make(specs, *, order=None):
        events, moves, protected_ids = [], [], set()
        for day, ticker, value, overrides, protected in map(_spec, specs):
            event = history_event(day, ticker=ticker, protected=protected)
            events.append(event)
            moves.append(outcome_move(event, value, **overrides))
            if protected:
                protected_ids.add(event["event_id"])
        conn, repository, snapshot = commit_event_history(tmp_path, events, moves)
        conns.append(conn)
        items = _items(repository, snapshot, events, moves)
        for event, item in zip(events, items):
            available = item["available_as_of_date"]
            BOUNDS[(event["ticker"], event["event_date"].date().isoformat())] = {
                "post_close_date": (date.fromisoformat(available) - timedelta(days=1)).isoformat()
                if available else None}
        held = [item for item in items if item["event_id"] in protected_ids]
        retained = [item for item in items if item["event_id"] not in protected_ids]
        if order is not None:
            retained = [retained[index] for index in order]
        return SimpleNamespace(repository=repository, snapshot=snapshot, events=events, moves=moves,
            retained=retained, held=held)

    yield make
    for conn in conns:
        conn.close()


@pytest.fixture(autouse=True)
def admit_recorder(monkeypatch):
    """Record admissions and serve native bounds: the real-repo fixtures have no bounds table
    for the real admit_source or for the roster-versus-bounds check in read_labels."""
    admitted = []
    monkeypatch.setattr(season_population, "admit_source",
        lambda repository, snapshot, limits: admitted.append(limits))
    BOUNDS.clear()
    monkeypatch.setattr(season_population.outcome_binding, "anchors",
        lambda repository, snapshot, limits, **kwargs: BOUNDS)
    return admitted


def _read(case, **kwargs):
    return season_population.read_labels(case.repository, case.snapshot, case.retained,
        as_of_month=MONTH, limits=LIMITS, **kwargs)


def _moves_scans(repository, start=0):
    return [query for name, query in repository.scans[start:] if name == "computed_moves"]


def _outcome_ids(case):
    return {record_.object_ref.object_id for record_
            in case.repository.fragment_records(case.snapshot, "computed_moves")}


def _stub_rows(monkeypatch, rows):
    """Plan from real records but serve caller-chosen decoded rows to the reader."""

    def fake(repository, snapshot, candidates, contract):
        records = {record_.primary_key_min: record_
                   for record_ in repository.fragment_records(snapshot, "computed_moves")}
        plans = [(event, object(), records[(event.ticker, event.event_date.date().isoformat())])
                 for event in candidates.itertuples(index=False)]
        return plans, lambda event, request: rows[event.event_id]

    monkeypatch.setattr(event_history, "plan_with_reader", fake)


def _valid_row(case, index=0):
    item = case.retained[index]
    event = next(each for each in case.events if each["event_id"] == item["event_id"])
    return {"event_date": event["event_date"].date().isoformat(), "skipped": False,
        "available_as_of_date": item["available_as_of_date"], "realized_move_pct": -1.5}


def test_labels_match_the_roster_in_retained_order(source):
    case = source([(DAYS[0], "AAA", -2.5), (DAYS[1], "BBB", 4.0), (DAYS[2], "CCC", -0.5)], order=[2, 0, 1])
    labels = _read(case)
    expected = [(DAYS[2], "CCC", 0.5), (DAYS[0], "AAA", 2.5), (DAYS[1], "BBB", 4.0)]
    for label, (day, ticker, move), item in zip(labels, expected, case.retained):
        assert label == {"event_id": item["event_id"], "ticker": ticker, "event_date": day,
            "available_as_of_date": item["available_as_of_date"], "abs_move": move}
        assert set(label) == {"event_id", "ticker", "event_date", "available_as_of_date", "abs_move"}
        assert label["available_as_of_date"] == (date.fromisoformat(day) + timedelta(days=2)).isoformat()


def test_empty_retained_returns_nothing_and_touches_no_repository():
    class Nothing:
        def __getattr__(self, name):
            raise AssertionError(f"empty roster touched {name}")

    assert season_population.read_labels(Nothing(), None, [], as_of_month=MONTH, limits=LIMITS) == []


def test_skipped_outcome_refuses_contract_mismatch(source):
    case = source([(DAYS[0], "AAA", None, {"skipped": True, "available_as_of_date": None})])
    with pytest.raises(DataError) as caught:
        _read(case)
    assert caught.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("shift", [pytest.param(1, id="one-day-later"), pytest.param(None, id="null")])
def test_availability_differing_from_the_roster_refuses(source, shift):
    case = source([(DAYS[0], "AAA", 2.0), (DAYS[1], "BBB", 3.0)])
    roster_day = case.retained[0]["available_as_of_date"]
    case.retained[0]["available_as_of_date"] = (
        (date.fromisoformat(roster_day) + timedelta(days=shift)).isoformat() if shift else None)
    with pytest.raises(DataError) as caught:
        _read(case)
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_bounds_post_close_one_day_earlier_than_the_roster_implies_refuses_before_any_scan(source):
    case = source([(DAYS[0], "AAA", 2.0), (DAYS[1], "BBB", 3.0)])
    row = BOUNDS[("AAA", DAYS[0])]
    row["post_close_date"] = (date.fromisoformat(row["post_close_date"]) - timedelta(days=1)).isoformat()
    with pytest.raises(DataError) as caught:
        _read(case)
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert _moves_scans(case.repository) == []


def test_missing_bounds_row_for_a_retained_event_refuses_before_any_scan(source):
    case = source([(DAYS[0], "AAA", 2.0), (DAYS[1], "BBB", 3.0)])
    del BOUNDS[("AAA", DAYS[0])]
    with pytest.raises(DataError) as caught:
        _read(case)
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert _moves_scans(case.repository) == []


@pytest.mark.parametrize("value", [pytest.param(float("nan"), id="nan"), pytest.param(True, id="bool"),
                                   pytest.param("2.0", id="string"), pytest.param(10 ** 400, id="huge-int"),
                                   pytest.param(float("inf"), id="inf"),
                                   pytest.param(float("-inf"), id="-inf")])
def test_invalid_move_refuses_variant(source, monkeypatch, value):
    case = source([(DAYS[0], "AAA", 2.0)])
    row = _valid_row(case)
    row["realized_move_pct"] = value
    _stub_rows(monkeypatch, {case.retained[0]["event_id"]: [row]})
    with pytest.raises(DataError) as caught:
        _read(case)
    assert caught.value.code == "EXPERIMENT_VARIANT_FAILED"


@pytest.mark.parametrize("column", [pytest.param("outcome_object_id", id="object-id"),
                                    pytest.param("outcome_content_hash", id="content-hash")])
def test_roster_binding_mismatch_refuses_before_any_outcome_scan(source, column):
    case = source([(DAYS[0], "AAA", 2.0), (DAYS[1], "BBB", 3.0)])
    case.retained[0][column] = "mismatched"
    with pytest.raises(DataError) as caught:
        _read(case)
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
    assert _moves_scans(case.repository) == []


def test_foreign_computed_moves_contract_refuses(source, monkeypatch):
    case = source([(DAYS[0], "AAA", 2.0)])
    monkeypatch.setattr(case.repository, "table_contract", lambda snapshot, name: BOUNDS_CONTRACT)
    with pytest.raises(DataError) as caught:
        _read(case)
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert _moves_scans(case.repository) == []


def test_duplicate_roster_ids_refuse_before_any_outcome_scan(source):
    case = source([(DAYS[0], "AAA", 2.0)])
    case.retained.append(dict(case.retained[0]))
    with pytest.raises(DataError) as caught:
        _read(case)
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
    assert _moves_scans(case.repository) == []


@pytest.mark.parametrize("shape", ["none", "two", "wrong-day"])
def test_outcome_rows_that_are_not_the_exact_singleton_refuse(source, monkeypatch, shape):
    case = source([(DAYS[0], "AAA", 2.0), (DAYS[1], "BBB", 3.0)])
    row = _valid_row(case)
    event_id = case.retained[0]["event_id"]
    if shape == "none":
        rows = {event_id: []}
    elif shape == "two":
        rows = {event_id: [row, dict(row)]}
    else:
        row["event_date"] = DAYS[1]
        rows = {event_id: [row]}
    _stub_rows(monkeypatch, rows)
    with pytest.raises(DataError) as caught:
        _read(case)
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_heldout_label_refuses_the_whole_read_before_any_outcome_opens(source, monkeypatch):
    case = source([(DAYS[0], "AAA", 2.0), (DAYS[1], "BBB", 3.0), (PROTECTED, "CCC", None, {}, True)])
    case.retained.extend(case.held)  # a roster that still names a held-out label
    opened = poison_outcomes(monkeypatch, case.repository, case.snapshot, [PROTECTED])
    with pytest.raises(DataError) as caught:
        _read(case)
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
    assert set(opened) & _outcome_ids(case) == set()
    assert _moves_scans(case.repository) == []


def test_only_retained_objects_are_opened(source, monkeypatch):
    case = source([(day, ticker, 1.0) for day, ticker in zip(DAYS, TICKERS)]
        + [(PROTECTED, "ZZZ", None, {}, True)])
    opened = poison_outcomes(monkeypatch, case.repository, case.snapshot, [PROTECTED])
    labels = _read(case)
    assert len(labels) == len(DAYS)
    assert set(opened) & _outcome_ids(case) == {item["outcome_object_id"] for item in case.retained}


def test_one_run_scan_per_ticker_and_a_second_call_is_a_new_build(source):
    case = source([(day, ticker, 1.0) for day, ticker in zip(DAYS, TICKERS)])
    start = len(case.repository.scans)
    labels = _read(case)
    days = _moves_scans(case.repository, start)
    assert len(days) == len(DAYS)
    assert [query.time_interval.start_inclusive for query in days] == sorted(DAYS)
    assert [label["event_date"] for label in labels] == sorted(DAYS)
    again = _read(case)
    assert again == labels
    assert len(_moves_scans(case.repository, start)) == 2 * len(DAYS)


def test_admission_is_called_once_with_the_limits_before_the_population_read(source, monkeypatch,
        admit_recorder):
    case = source([(day, ticker, 1.0) for day, ticker in zip(DAYS, TICKERS)])
    real_load = season_population.load_population

    def spy(repository, snapshot, **kwargs):
        admit_recorder.append("load_population")
        return real_load(repository, snapshot, **kwargs)

    monkeypatch.setattr(season_population, "load_population", spy)
    labels = _read(case)
    assert len(labels) == len(DAYS)
    assert admit_recorder == [LIMITS, "load_population"]


def test_invalid_limits_refuse_before_any_scan(source, monkeypatch):
    monkeypatch.setattr(season_population, "admit_source", eligible_history.admit_source)
    case = source([(DAYS[0], "AAA", 2.0)])
    limits = {key: value for key, value in LIMITS.items() if key != "max_output_rows"}
    with pytest.raises(DataError) as caught:
        season_population.read_labels(case.repository, case.snapshot, case.retained,
            as_of_month=MONTH, limits=limits)
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert _moves_scans(case.repository) == []


def test_same_ticker_labels_share_one_run_scan_across_over_cap_tickers(source):
    specs = []
    for index in range(40):
        ticker = f"T{index:02}"
        specs.append((DAYS[0], ticker, -(index + 1)))
        specs.append((DAYS[1], ticker, (index + 1) + 0.5))
    case = source(specs)
    start = len(case.repository.scans)
    labels = _read(case)
    assert len(_moves_scans(case.repository, start)) == 40
    assert len(labels) == len(specs)
    for index, label in enumerate(labels):
        day, ticker, value = specs[index]
        assert label["ticker"] == ticker
        assert label["event_date"] == day
        assert label["abs_move"] == abs(float(value))


def test_population_read_is_bound_by_the_limit_max_object_bytes(source, monkeypatch):
    case = source([(day, ticker, 1.0) for day, ticker in zip(DAYS, TICKERS)])
    calls = []
    real_load = season_population.load_population

    def spy(repository, snapshot, **kwargs):
        calls.append(kwargs)
        return real_load(repository, snapshot, **kwargs)

    monkeypatch.setattr(season_population, "load_population", spy)
    labels = season_population.read_labels(case.repository, case.snapshot, case.retained,
        as_of_month=MONTH, limits={**LIMITS, "max_object_bytes": 12345})
    assert len(labels) == len(DAYS)
    assert [call["max_object_bytes"] for call in calls] == [12345]


# (event_day, pre_close, post_close): post_close + 1 calendar day is the availability.
# AMC labels close on their own session (decision E-1 close for the target session after it);
# BMO labels carry the prior-close bracket and close one session later (decision E-2 close).
BOUNDARY = {"AMC": ((ROSTER_DAYS[298], ROSTER_DAYS[297], ROSTER_DAYS[298]),
    (LAST, ROSTER_DAYS[298], LAST)),
    "BMO": ((ROSTER_DAYS[297], ROSTER_DAYS[296], ROSTER_DAYS[298]),
        (ROSTER_DAYS[298], ROSTER_DAYS[297], LAST))}


def _bounds(ticker, day, session, pre, post):
    return bounds_row(ticker, day, session=session, pre_close_date=pre, post_close_date=post)


@pytest.mark.parametrize("session", ["AMC", "BMO"])
def test_availability_boundary_and_the_read_of_a_retained_label(monkeypatch, session):
    """post_close + 1 calendar day == the latest decision is retained; one day later is not."""
    (in_day, in_pre, in_post), (out_day, out_pre, out_post) = BOUNDARY[session]
    assert (date.fromisoformat(in_post) + timedelta(days=1)).isoformat() == LAST
    assert (date.fromisoformat(out_post) + timedelta(days=1)).isoformat() > LAST
    events = [{"event_id": "ev-in", "ticker": "IN", "event_date": pd.Timestamp(in_day), "session": session},
        {"event_id": "ev-out", "ticker": "OUT", "event_date": pd.Timestamp(out_day), "session": session}]
    anchors = {("IN", in_day): _bounds("IN", in_day, session, in_pre, in_post),
        ("OUT", out_day): _bounds("OUT", out_day, session, out_pre, out_post)}
    moves = [record("IN", in_day), record("OUT", out_day)]
    _, result, _, _ = plan(monkeypatch, events, anchors, moves, sessions=[FIRST, LAST], calendar=CALENDAR)
    assert [item["event_id"] for item in result["retained"]] == ["ev-in"]
    assert result["retained"][0]["available_as_of_date"] == LAST
    assert result["receipt"]["counts"]["not_yet_available"] == 1

    item = result["retained"][0]
    frame = pd.DataFrame([{"event_id": "ev-in", "ticker": "IN", "event_date": pd.Timestamp(in_day)}])
    monkeypatch.setattr(season_population, "load_population", lambda *args, **kwargs: frame)
    planned = record("IN", in_day)
    rows = [{"event_date": in_day, "skipped": False, "available_as_of_date": LAST,
        "realized_move_pct": -0.75}]
    monkeypatch.setattr(event_history, "plan_with_reader", lambda repository, snapshot, candidates, contract:
        ([(event, object(), planned) for event in candidates.itertuples(index=False)],
         lambda event, request: rows))
    repository = SimpleNamespace(table_contract=lambda snapshot, name: COMPUTED_MOVES_CONTRACT,
        anchor_limits=[])
    assert season_population.read_labels(repository, None, [item], as_of_month=MONTH, limits=LIMITS) == [
        {"event_id": "ev-in", "ticker": "IN", "event_date": in_day, "available_as_of_date": LAST,
            "abs_move": 0.75}]
