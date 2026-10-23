"""Run-batched candidate reads: identical rows, exact isolation, lazy streaming, retry."""
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data import errors
from engine.v2.data import query as query_planner
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT
from engine.v2.data.errors import DataError
from engine.v2.research import event_history
from engine.v2.research.experiment_population import load_population
from tests.v2.research.magnitude_trial_support import history_event, outcome_move, poison_outcomes
from tests.v2.research.test_event_history import MONTH, commit_event_history

DAYS = ("2023-01-03", "2023-02-07", "2023-03-07", "2023-05-02", "2023-06-06")
PROTECTED = "2023-04-04"


@pytest.fixture
def source(tmp_path):
    events = [history_event(day) for day in DAYS]
    protected = history_event(PROTECTED, protected=True)
    moves = [outcome_move(event, float(index + 1)) for index, event in enumerate(events)]
    moves.append(outcome_move(protected, 9.0))
    conn, repository, snapshot = commit_event_history(tmp_path, [*events, protected], moves)
    try:
        yield repository, snapshot
    finally:
        conn.close()


def _plan(repository, snapshot):
    population = load_population(repository, snapshot, as_of_month=MONTH, purpose="selection")
    candidates = population.sort_values(["event_date", "event_id"])
    return event_history.plan_with_reader(repository, snapshot, candidates, COMPUTED_MOVES_CONTRACT)


def _outcome_scans(repository, start):
    return [query for name, query in repository.scans[start:] if name == "computed_moves"]


def _per_candidate(repository, request):
    return [row for batch in repository.scan(request, table_name="computed_moves") for row in batch.to_pylist()]


def test_rows_match_per_candidate_scans_with_one_scan_per_run_and_protected_unopened(source, monkeypatch):
    repository, snapshot = source
    poison_outcomes(monkeypatch, repository, snapshot, [PROTECTED])
    plans, read_rows = _plan(repository, snapshot)
    assert [e.event_date.date().isoformat() for e, _, _ in plans] == list(DAYS)
    start = len(repository.scans)
    batched = [read_rows(event, request) for event, request, _ in plans]
    assert len(_outcome_scans(repository, start)) == 2  # cut at the protected day
    per_candidate = [[row for batch in repository.scan(request, table_name="computed_moves")
                      for row in batch.to_pylist()] for _, request, _ in plans]
    assert batched == per_candidate
    assert all(len(rows) == 1 for rows in batched)


def test_stream_opens_one_object_per_requested_candidate_and_caches(source, monkeypatch):
    repository, snapshot = source
    opened = poison_outcomes(monkeypatch, repository, snapshot, [PROTECTED])
    plans, read_rows = _plan(repository, snapshot)
    base = len(opened)
    first = read_rows(plans[0][0], plans[0][1])
    assert len(opened) == base + 1  # later candidates of the run are not opened yet
    read_rows(plans[1][0], plans[1][1])
    assert len(opened) == base + 2
    start = len(repository.scans)
    assert read_rows(plans[0][0], plans[0][1]) is first
    assert len(repository.scans) == start and len(opened) == base + 2
    assert first == _per_candidate(repository, plans[0][1])  # after the opened-count assertions


def test_membership_mismatch_refuses_before_any_object_opens(source, monkeypatch):
    repository, snapshot = source
    opened = poison_outcomes(monkeypatch, repository, snapshot, [], all_outcomes=True)
    plans, read_rows = _plan(repository, snapshot)
    base = len(opened)
    original = query_planner.plan_scan_population

    def short(contract, records, **kwargs):
        population = original(contract, records, **kwargs)
        if len(population.records) > 1:
            return type(population)(records=population.records[:-1], row_count=population.row_count - 1)
        return population

    monkeypatch.setattr(query_planner, "plan_scan_population", short)
    with pytest.raises(DataError, match="(?s)HOLDOUT_ACCESS_DENIED.*research outcome isolation cannot be proven"):
        read_rows(plans[0][0], plans[0][1])
    assert len(opened) == base


@pytest.mark.parametrize("clause", ["records", "row_count"])
def test_run_membership_refuses_on_each_clause(source, monkeypatch, clause):
    repository, snapshot = source
    opened = poison_outcomes(monkeypatch, repository, snapshot, [], all_outcomes=True)
    plans, read_rows = _plan(repository, snapshot)
    base = len(opened)
    protected = next(r for r in repository.fragment_records(snapshot, "computed_moves")
                     if r.time_min == PROTECTED)
    original = query_planner.plan_scan_population

    def mismatched(contract, records, **kwargs):
        population = original(contract, records, **kwargs)
        if len(population.records) > 1:
            if clause == "records":  # equal row_count with a different fragment set
                return type(population)(records=(*population.records[:-1], protected),
                                        row_count=population.row_count)
            return type(population)(records=population.records,
                                    row_count=population.row_count + 1)
        return population

    monkeypatch.setattr(query_planner, "plan_scan_population", mismatched)
    with pytest.raises(DataError, match="(?s)HOLDOUT_ACCESS_DENIED.*research outcome isolation cannot be proven"):
        read_rows(plans[0][0], plans[0][1])
    assert len(opened) == base


def test_scan_failure_drops_the_stream_and_a_retry_rescans(source, monkeypatch):
    repository, snapshot = source
    plans, read_rows = _plan(repository, snapshot)
    original, calls, closed = repository.scan, [], []

    def flaky(query, *, table_name):
        calls.append(query)
        try:
            if len(calls) == 1:
                raise errors.fail("MANIFEST_CORRUPT", "synthetic scan failure")
            yield from original(query, table_name=table_name)
        finally:
            closed.append(query)

    monkeypatch.setattr(repository, "scan", flaky)
    with pytest.raises(DataError, match="(?s)MANIFEST_CORRUPT.*synthetic scan failure"):
        read_rows(plans[0][0], plans[0][1])
    assert closed  # the failed generator was closed
    rows = read_rows(plans[0][0], plans[0][1])
    assert len(calls) == 2
    assert rows == [row for batch in original(plans[0][1], table_name="computed_moves")
                    for row in batch.to_pylist()]


def test_retry_after_rows_accumulated_returns_full_rows(source, monkeypatch):
    repository, snapshot = source
    plans, read_rows = _plan(repository, snapshot)
    original, calls = repository.scan, []

    def flaky(query, *, table_name):
        calls.append(query)
        if len(calls) > 1:  # only the first call of the stream fails
            yield from original(query, table_name=table_name)
            return
        batches = original(query, table_name=table_name)
        yield next(batches)  # one batch reaches the reader, then the stream dies
        raise errors.fail("MANIFEST_CORRUPT", "synthetic scan failure")

    monkeypatch.setattr(repository, "scan", flaky)
    first = read_rows(plans[0][0], plans[0][1])  # the first batch succeeds
    assert first == [row for batch in original(plans[0][1], table_name="computed_moves")
                     for row in batch.to_pylist()]
    with pytest.raises(DataError, match="(?s)MANIFEST_CORRUPT.*synthetic scan failure"):
        read_rows(plans[1][0], plans[1][1])
    rows = read_rows(plans[1][0], plans[1][1])  # a retry re-scans the run from its start
    assert rows == [row for batch in original(plans[1][1], table_name="computed_moves")
                    for row in batch.to_pylist()]
    assert read_rows(plans[2][0], plans[2][1]) == [
        row for batch in original(plans[2][1], table_name="computed_moves")
        for row in batch.to_pylist()]  # no duplicated rows from the restarted stream


def test_extra_fragment_beyond_the_run_refuses_before_any_object_opens(source, monkeypatch):
    repository, snapshot = source
    opened = poison_outcomes(monkeypatch, repository, snapshot, [], all_outcomes=True)
    plans, read_rows = _plan(repository, snapshot)
    base = len(opened)
    protected = next(r for r in repository.fragment_records(snapshot, "computed_moves") if r.time_min == PROTECTED)
    original = query_planner.plan_scan_population

    def widened(contract, records, **kwargs):
        population = original(contract, records, **kwargs)
        if len(population.records) > 1:
            return type(population)(records=(*population.records, protected), row_count=population.row_count + 1)
        return population

    monkeypatch.setattr(query_planner, "plan_scan_population", widened)
    with pytest.raises(DataError, match="(?s)HOLDOUT_ACCESS_DENIED.*research outcome isolation cannot be proven"):
        read_rows(plans[0][0], plans[0][1])
    assert len(opened) == base


def test_out_of_order_requests_match_in_order_rows(source):
    repository, snapshot = source
    plans, read_rows = _plan(repository, snapshot)
    for index in (2, 0, 1, 4, 3):
        event, request, _ = plans[index]
        expected = [row for batch in repository.scan(request, table_name="computed_moves")
                    for row in batch.to_pylist()]
        assert read_rows(event, request) == expected


def test_read_rows_refuses_a_mismatched_date_ticker_or_request(source, monkeypatch):
    repository, snapshot = source
    opened = poison_outcomes(monkeypatch, repository, snapshot, [], all_outcomes=True)
    plans, read_rows = _plan(repository, snapshot)
    base = len(opened)
    event = plans[0][0]
    wrong_date = SimpleNamespace(event_id=event.event_id, ticker=event.ticker,
                                 event_date=plans[1][0].event_date)
    wrong_ticker = SimpleNamespace(event_id=event.event_id, ticker="OTHER",
                                   event_date=event.event_date)
    for candidate, request in ((wrong_date, plans[0][1]), (wrong_ticker, plans[0][1]),
                               (event, plans[1][1])):
        with pytest.raises(DataError, match="(?s)HOLDOUT_ACCESS_DENIED.*event or request was not preflighted for this reader"):
            read_rows(candidate, request)
    assert len(opened) == base


def test_event_without_identity_refuses_not_attribute_error(source):
    repository, snapshot = source
    plans, read_rows = _plan(repository, snapshot)
    with pytest.raises(DataError, match="(?s)HOLDOUT_ACCESS_DENIED.*event or request was not preflighted for this reader"):
        read_rows(SimpleNamespace(), plans[0][1])


def test_unplanned_event_refuses_and_malformed_candidates_refuse_contract_mismatch(source, monkeypatch):
    repository, snapshot = source
    opened = poison_outcomes(monkeypatch, repository, snapshot, [], all_outcomes=True)
    plans, read_rows = _plan(repository, snapshot)
    base = len(opened)
    stranger = SimpleNamespace(event_id="not-planned", event_date=pd.Timestamp(DAYS[0]))
    with pytest.raises(DataError, match="(?s)HOLDOUT_ACCESS_DENIED.*event or request was not preflighted for this reader"):
        read_rows(stranger, plans[0][1])
    assert len(opened) == base
    population = load_population(repository, snapshot, as_of_month=MONTH, purpose="selection")
    for bad in (population.drop(columns=["ticker"]), population.assign(ticker=" " + population.ticker),
                population.assign(event_date=population.event_date + pd.Timedelta(hours=1)),
                population.drop(columns=["event_id"]), population.assign(event_id=population.event_id.iloc[0]),
                population.assign(event_id=None)):
        with pytest.raises(DataError, match="(?s)CONTRACT_MISMATCH.*event target candidates are malformed"):
            event_history.plan_with_reader(repository, snapshot, bad, COMPUTED_MOVES_CONTRACT)


def test_run_stream_is_closed_after_its_last_candidate(source, monkeypatch):
    repository, snapshot = source
    plans, read_rows = _plan(repository, snapshot)
    original, closed = repository.scan, []
    held = []

    def tracked(query, *, table_name):
        def body():
            try:
                yield from original(query, table_name=table_name)
            finally:
                closed.append(query.time_interval.start_inclusive)
        generator = body()
        held.append(generator)  # a strong reference: only an explicit close() can run the finally
        return generator

    monkeypatch.setattr(repository, "scan", tracked)
    read_rows(plans[0][0], plans[0][1])
    assert closed == []
    read_rows(plans[1][0], plans[1][1])
    assert closed == []  # only an intermediate candidate: the stream stays open
    read_rows(plans[2][0], plans[2][1])
    assert closed == [DAYS[0]]  # the run's last candidate closes it immediately
    assert read_rows(plans[1][0], plans[1][1])[0]["event_date"] == DAYS[1]
    assert read_rows(plans[1][0], plans[1][1]) == _per_candidate(repository, plans[1][1])


def test_streams_beyond_the_cap_close_the_oldest(source, monkeypatch):
    monkeypatch.setattr(event_history, "_MAX_OPEN_RUN_STREAMS", 1)
    repository, snapshot = source
    plans, read_rows = _plan(repository, snapshot)
    original, closed = repository.scan, []
    held = []

    def tracked(query, *, table_name):
        def body():
            try:
                yield from original(query, table_name=table_name)
            finally:
                closed.append(query.time_interval.start_inclusive)
        generator = body()
        held.append(generator)  # a strong reference: only an explicit close() can run the finally
        return generator

    monkeypatch.setattr(repository, "scan", tracked)
    read_rows(plans[0][0], plans[0][1])  # first run, not its last candidate
    assert closed == []
    read_rows(plans[3][0], plans[3][1])  # second run: the first stream is released
    assert closed == [DAYS[0]]
    assert read_rows(plans[1][0], plans[1][1]) == [
        row for batch in original(plans[1][1], table_name="computed_moves")
        for row in batch.to_pylist()]  # re-scan after close gives correct rows


def test_interleaved_runs_keep_their_streams_and_scan_once(source, monkeypatch):
    repository, snapshot = source
    plans, read_rows = _plan(repository, snapshot)
    original, calls = repository.scan, []

    def counting(query, *, table_name):
        calls.append(query)
        yield from original(query, table_name=table_name)

    monkeypatch.setattr(repository, "scan", counting)
    order = (0, 3, 1, 4, 2)  # two runs interleaved, under the default cap
    rows = [read_rows(plans[index][0], plans[index][1]) for index in order]
    assert len(calls) == 2
    assert rows == [[row for batch in original(plans[index][1], table_name="computed_moves")
                     for row in batch.to_pylist()] for index in order]


def _plan_at(day):
    return (None, None, SimpleNamespace(time_min=day, time_max=day))


def test_runs_are_cut_only_where_another_fragment_may_lie_between_neighbours():
    group = [_plan_at(day) for day in ("2023-01-01", "2023-02-01", "2023-03-01")]
    lengths = lambda others: [len(run) for run in event_history._runs(group, others)]  # noqa: E731
    assert lengths([]) == [3]
    assert lengths([("2022-01-01", "2022-12-31"), ("2023-04-01", "2023-05-01")]) == [3]  # outside the range
    assert lengths([("2023-02-10", "2023-02-10")]) == [2, 1]  # strictly between the last two
    assert lengths([("2023-01-01", "2023-01-01")]) == [1, 2]  # touching a candidate's day also cuts
    assert lengths([("2023-02-01", "2023-02-01")]) == [1, 1, 1]  # the middle day cuts both gaps
    assert lengths([("2023-03-01", "2023-03-01")]) == [2, 1]  # o[0] <= hi kills the later gap only
    assert lengths([None]) == [1, 1, 1]  # an unreadable interval cuts everywhere


def test_cached_event_still_refuses_a_mismatched_day_or_request(source, monkeypatch):
    repository, snapshot = source
    opened = poison_outcomes(monkeypatch, repository, snapshot, [PROTECTED])
    plans, read_rows = _plan(repository, snapshot)
    event, request, _ = plans[0]
    first = read_rows(event, request)  # now cached
    assert first == _per_candidate(repository, request)
    base = len(opened)  # after the comparison scan: the refusals below must open nothing more
    wrong_day = SimpleNamespace(event_id=event.event_id, ticker=event.ticker, event_date=plans[1][0].event_date)
    with pytest.raises(DataError, match="(?s)HOLDOUT_ACCESS_DENIED.*event or request was not preflighted for this reader"):
        read_rows(wrong_day, request)
    with pytest.raises(DataError, match="(?s)HOLDOUT_ACCESS_DENIED.*event or request was not preflighted for this reader"):
        read_rows(event, plans[1][1])
    assert len(opened) == base


def test_planned_id_without_ticker_or_date_refuses_typed(source):
    repository, snapshot = source
    plans, read_rows = _plan(repository, snapshot)
    event = plans[0][0]
    for partial in (SimpleNamespace(event_id=event.event_id),
                    SimpleNamespace(event_id=event.event_id, ticker=event.ticker),
                    SimpleNamespace(event_id=event.event_id, event_date=event.event_date)):
        with pytest.raises(DataError, match="(?s)HOLDOUT_ACCESS_DENIED.*event or request was not preflighted for this reader"):
            read_rows(partial, plans[0][1])
