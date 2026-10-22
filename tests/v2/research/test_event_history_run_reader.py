"""Run-batched candidate reads: identical rows, exact isolation, lazy streaming, retry."""
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
    with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
        read_rows(plans[0][0], plans[0][1])
    assert len(opened) == base


def test_scan_failure_drops_the_stream_and_a_retry_rescans(source, monkeypatch):
    repository, snapshot = source
    plans, read_rows = _plan(repository, snapshot)
    original, calls = repository.scan, []

    def flaky(query, *, table_name):
        calls.append(query)
        if len(calls) == 1:
            raise errors.fail("MANIFEST_CORRUPT", "synthetic scan failure")
        yield from original(query, table_name=table_name)

    monkeypatch.setattr(repository, "scan", flaky)
    with pytest.raises(DataError, match="MANIFEST_CORRUPT"):
        read_rows(plans[0][0], plans[0][1])
    rows = read_rows(plans[0][0], plans[0][1])
    assert len(calls) == 2 and len(rows) == 1 and rows[0]["event_date"] == DAYS[0]