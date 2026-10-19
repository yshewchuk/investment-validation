"""Metadata-only index equivalence and native synthetic preflight poison tests."""
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.contracts.data import DataQuery, KeyPredicate, TimeInterval
from engine.v2.data import query as query_planner
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT
from engine.v2.data.errors import DataError, fail
from engine.v2.research import event_history
from engine.v2.research.experiment_population import load_population
from engine.v2.research.prediction_inputs import _target_filters
from tests.v2.research.magnitude_trial_support import (
    history_event,
    outcome_move,
    poison_outcomes,
)
from tests.v2.research.test_event_history import commit_event_history, read_event_history

DAY = "2023-03-02"
TICKER = "SYNTH"


@pytest.fixture
def committed(tmp_path):
    event = history_event(DAY)
    conn, repository, snapshot = commit_event_history(tmp_path, [event], [outcome_move(event)])
    try:
        record, = repository.fragment_records(snapshot, "computed_moves")
        yield repository, snapshot, record
    finally:
        conn.close()


def _record(base, label, lo=DAY, hi=DAY, **changes):
    """Clone inspected synthetic metadata only; these objects are never scanned."""
    return replace(base, fragment_id=label, time_min=lo, time_max=hi, **changes)


def _event_query(day=DAY, ticker=TICKER):
    return _target_filters(SimpleNamespace(ticker=ticker, event_date=pd.Timestamp(day)))


def _population(records, *, day=DAY, ticker=TICKER):
    keys, interval = _event_query(day, ticker)
    try:
        return query_planner.plan_scan_population(COMPUTED_MOVES_CONTRACT, records,
            key_filter=keys, time_interval=interval)
    except DataError as exc:
        return exc.problem


def _equivalent(records, *, day=DAY, ticker=TICKER):
    index = event_history._record_index(records)
    indexed = event_history._indexed_records(index, ticker, day)
    full = _population(records, day=day, ticker=ticker)
    assert _population(indexed, day=day, ticker=ticker) == full
    return full


def _ids(population):
    return [record.fragment_id for record in population.records]


def _assert_refusal_paths(committed, monkeypatch, records, expected, *, key_filter=None):
    """Literal error oracle across full/indexed planning and public readers."""
    repository, snapshot, _ = committed
    keys, interval = _event_query()
    if key_filter is not None:
        keys = key_filter
    # Index construction must not raise before query validation or an earlier
    # record's membership error. Poison every outcome open on all scan paths.
    opened = poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
    index = event_history._record_index(records)
    indexed = event_history._indexed_records(index, TICKER, DAY)
    monkeypatch.setattr(repository, "_table_records",
                        lambda *_args: (COMPUTED_MOVES_CONTRACT, records))
    monkeypatch.setattr(repository, "fragment_records", lambda *_args: records)
    request = DataQuery(snapshot_id=snapshot.snapshot_id,
        table_contract_ref=snapshot.table_versions["computed_moves"].table_contract_ref,
        columns=("event_date",), key_filter=keys, time_interval=interval,
        order_by=("ticker", "event_date"), max_batch_rows=1, max_result_rows=1)
    actions = {
        "full": lambda: query_planner.plan_scan_population(COMPUTED_MOVES_CONTRACT,
            records, key_filter=keys, time_interval=interval),
        "indexed": lambda: query_planner.plan_scan_population(COMPUTED_MOVES_CONTRACT,
            indexed, key_filter=keys, time_interval=interval),
        "bound": lambda: repository.scan_population_bound(snapshot.snapshot_id,
            table_name="computed_moves", table_contract_ref=request.table_contract_ref,
            key_filter=keys, time_interval=interval),
        "explain": lambda: repository.explain_dependencies(request, table_name="computed_moves"),
        "scan": lambda: list(repository.scan(request, table_name="computed_moves")),
    }
    if key_filter is None:
        candidates = pd.DataFrame([history_event(DAY)])
        actions["history"] = lambda: event_history._plans(
            repository, snapshot, candidates, COMPUTED_MOVES_CONTRACT)
    for name, action in actions.items():
        with pytest.raises(DataError) as failure:
            action()
        assert failure.value.problem == expected, name
    assert opened == []


def test_native_two_issuer_index_preserves_exact_admitted_membership_without_outcome_reads(
        tmp_path, monkeypatch):
    first, second = "2023-01-02", DAY
    events = [history_event(day, ticker=ticker) for ticker in (TICKER, "SYNTH_B")
              for day in (first, second)]
    conn, repository, snapshot = commit_event_history(tmp_path, events[::-1], [outcome_move(e) for e in events])
    try:
        records = tuple(repository.fragment_records(snapshot, "computed_moves"))
        by_key = {record.primary_key_min: record for record in records}
        assert set(by_key) == {(e["ticker"], e["event_date"].date().isoformat()) for e in events}
        assert len(records) == len(by_key) == 4
        assert len({record.object_ref.object_id for record in records}) == 4
        opened = poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
        before = conn.total_changes
        admitted = load_population(repository, snapshot, as_of_month="2025-01")
        assert set(admitted.event_id) == {e["event_id"] for e in events}
        assert admitted.attrs["holdout_exclusions"] == []
        # Overlapping dates cannot substitute for issuer membership. Candidate order
        # interleaves issuers and differs from both native manifest and date order.
        expected_events = [events[i] for i in (3, 0, 2, 1)]
        candidates = admitted.set_index("event_id").loc[
            [e["event_id"] for e in expected_events]].reset_index()
        expected_records = [by_key[(e["ticker"], e["event_date"].date().isoformat())]
                            for e in expected_events]
        assert expected_records != list(records)
        enumerations, planner_records = [], []
        original_records = repository.fragment_records
        original_plan = query_planner.plan_scan_population

        def metadata(received_snapshot, table_name):
            assert received_snapshot is snapshot and table_name == "computed_moves"
            enumerations.append(table_name)
            assert len(enumerations) == 1
            return original_records(received_snapshot, table_name)

        def plan(contract, selected, **kwargs):
            selected = tuple(selected)
            planner_records.append(selected)
            return original_plan(contract, selected, **kwargs)

        def no_second_lookup(*args, **kwargs):
            pytest.fail("a supplied population bound must not enumerate metadata again")

        monkeypatch.setattr(repository, "fragment_records", metadata)
        monkeypatch.setattr(repository, "scan_population_bound", no_second_lookup)
        monkeypatch.setattr(query_planner, "plan_scan_population", plan)
        scans_before = list(repository.scans)
        plans = event_history._plans(repository, snapshot, candidates, COMPUTED_MOVES_CONTRACT)
        assert enumerations == ["computed_moves"]
        assert planner_records == [(record,) for record in expected_records]
        assert [(e.event_id, e.ticker, e.event_date, r.fragment_id, r.object_ref.object_id)
                for e, _, r in plans] == [
            (e["event_id"], e["ticker"], e["event_date"], r.fragment_id, r.object_ref.object_id)
            for e, r in zip(expected_events, expected_records)]
        for (_, request, record), event, expected in zip(plans, expected_events, expected_records):
            day = event["event_date"].date().isoformat()
            assert record == expected
            assert record.primary_key_min == record.primary_key_max == (event["ticker"], day)
            assert record.partition_key == event["ticker"]
            assert record.time_min == record.time_max == day and record.row_count == 1
            assert request == DataQuery(snapshot_id=snapshot.snapshot_id,
                table_contract_ref=snapshot.table_versions["computed_moves"].table_contract_ref,
                columns=("event_date", "realized_move_pct", "available_as_of_date", "skipped"),
                key_filter=(KeyPredicate(column="ticker", operator="eq", values=(event["ticker"],)),),
                time_interval=TimeInterval(column="event_date", start_inclusive=day,
                    end_exclusive=(event["event_date"] + pd.Timedelta(days=1)).date().isoformat()),
                order_by=("ticker", "event_date"), max_batch_rows=1, max_result_rows=1)
        assert repository.scans == scans_before
        assert scans_before and all(name == "earnings_events" for name, _ in scans_before)
        assert not ({record.object_ref.object_id for record in records} & set(opened))
        assert conn.total_changes == before
    finally:
        conn.close()


def test_large_point_corpus_is_enumerated_once_and_planner_sees_one_record_per_target(
        committed, monkeypatch):
    repository, snapshot, base = committed
    days = pd.date_range("2000-01-01", periods=4096)
    records = [_record(base, f"point-{index}", day.date().isoformat(), day.date().isoformat(),
        primary_key_min=(TICKER, day.date().isoformat()),
        primary_key_max=(TICKER, day.date().isoformat())) for index, day in enumerate(days)]
    # Neither manifest order nor candidate order agrees with tree traversal order.
    records = records[::2][::-1] + records[1::2]
    candidates = pd.DataFrame({"event_id": [f"candidate-{i}" for i in range(0, len(days), 17)],
        "ticker": TICKER, "event_date": days[::17]}).iloc[::-1]
    enumerations, yielded, calls, admitted = [], [], [], []

    def metadata(received_snapshot, table_name):
        assert received_snapshot is snapshot and table_name == "computed_moves"
        enumerations.append(table_name)
        assert len(enumerations) == 1
        for record in records:
            yielded.append(record.fragment_id)
            yield record

    original = query_planner.plan_scan_population
    original_bounds = query_planner.fragment_key_bounds

    def admit(record, contract):
        admitted.append(record.fragment_id)
        return original_bounds(record, contract)

    def plan(contract, selected, **kwargs):
        selected = tuple(selected)
        assert len(selected) == 1
        calls.append(selected[0].fragment_id)
        return original(contract, selected, **kwargs)

    def no_second_lookup(*args, **kwargs):
        pytest.fail("a supplied population bound must not enumerate metadata again")

    monkeypatch.setattr(repository, "fragment_records", metadata)
    monkeypatch.setattr(repository, "scan_population_bound", no_second_lookup)
    monkeypatch.setattr(query_planner, "plan_scan_population", plan)
    monkeypatch.setattr(query_planner, "fragment_key_bounds", admit)
    plans = event_history._plans(repository, snapshot, candidates, COMPUTED_MOVES_CONTRACT)
    expected = [f"point-{i}" for i in reversed(range(0, len(days), 17))]
    assert enumerations == ["computed_moves"]
    assert yielded == [record.fragment_id for record in records]
    assert calls == expected == [record.fragment_id for _, _, record in plans]
    assert admitted == [record.fragment_id for record in records] + expected
    assert [event.event_id for event, _, _ in plans] == candidates.event_id.tolist()
    for event, request, _ in plans:
        keys, interval = _target_filters(event)
        assert request.key_filter == keys and request.time_interval == interval
        assert request.snapshot_id == snapshot.snapshot_id
        assert request.max_result_rows == request.max_batch_rows == 1
    assert repository.scans == []


@pytest.mark.parametrize("day, expected", [
    ("2022-12-31", []),
    ("2023-01-01", ["outer", "left-edge"]),
    ("2023-01-02", ["outer"]),
    ("2023-03-01", ["nested", "outer", "overlap"]),
    (DAY, ["duplicate", "nested", "outer", "point", "overlap", "duplicate"]),
    ("2023-03-03", ["nested", "outer", "overlap"]),
    ("2023-03-04", ["outer"]),
    ("2023-12-31", ["outer", "right-edge"]),
    ("2024-01-01", []),
])
def test_nested_overlapping_duplicate_intervals_and_inclusive_edges_keep_manifest_order(
        committed, day, expected):
    _, _, base = committed
    duplicate = _record(base, "duplicate")
    records = [duplicate, _record(base, "nested", "2023-02-01", "2023-03-03"),
        _record(base, "outer", "2023-01-01", "2023-12-31"), _record(base, "point"),
        _record(base, "right-edge", "2023-12-31", "2023-12-31"),
        _record(base, "overlap", "2023-03-01", "2023-03-03"), duplicate,
        _record(base, "left-edge", "2023-01-01", "2023-01-01")]
    population = _equivalent(records, day=day)
    assert _ids(population) == expected
    assert population.row_count == len(expected)
    assert _ids(_equivalent(records, day=day, ticker="ABSENT")) == []


@pytest.mark.parametrize("lo, hi, selected", [
    (None, None, True), (None, "invalid", True), ("invalid", None, True),
    ("20230302", "20230302", True),
    ("2023-03-02T12:00:00", "2023-03-02T12:00:00", True),
    ("2023-03-02T00:00:00Z", "2023-03-02T00:00:00Z", True),
    ("2023-03-01T23:30:00-01:00", "2023-03-01T23:30:00-01:00", True),
    ("2023-03-03T00:30:00+01:00", "2023-03-03T00:30:00+01:00", True),
    ("2023-03-02T00:30:00+01:00", "2023-03-02T00:30:00+01:00", False),
    ("2023-03-03", "2023-03-01", False),
    (DAY, "2023-03-01", False),
])
def test_uncertain_missing_noncanonical_timezone_and_reversed_bounds_use_real_planner(
        committed, lo, hi, selected):
    _, _, base = committed
    record = _record(base, "uncertain", lo, hi)
    assert event_history._canonical_interval(record) is None
    population = _equivalent([record])
    assert _ids(population) == (["uncertain"] if selected else [])
    assert population.row_count == int(selected)


@pytest.mark.parametrize("lo, hi", [
    ("bad", DAY), ("2023-02-30", DAY), ("2023-3-2", DAY), (DAY, "bad"),
    (7, DAY), (DAY, False),
])
def test_invalid_time_bounds_preserve_full_typed_failure(committed, lo, hi):
    _, _, base = committed
    problem = _equivalent([_record(base, "invalid", lo, hi)])
    assert problem == fail("CONTRACT_MISMATCH",
        "a time bound is not a parseable date or timestamp").problem


@pytest.mark.parametrize("day", ["0001-01-01", "0999-03-02", DAY])
def test_years_below_1000_follow_planner_timestamp_normalization(committed, day):
    _, _, base = committed
    records = [_record(base, "ancient", "0001-01-01", "0999-12-31"),
        _record(base, "ancient-point", "0999-03-02", "0999-03-02"),
        _record(base, "canonical-wide", "1000-01-01", "9999-12-31"),
        _record(base, "modern")]
    for record in records[:2]:
        assert event_history._canonical_interval(record) is None
    index = event_history._record_index(records)
    if day[:4] < "1000":
        # strftime may omit year padding; canonical ranges must also reach the planner.
        assert event_history._indexed_records(index, TICKER, day) == records
    _equivalent(records, day=day)


@pytest.mark.parametrize("field, value", [
    ("primary_key_min", (7, DAY)), ("primary_key_min", (False, DAY)),
    ("primary_key_min", ({}, DAY)),
    ("primary_key_max", (7, DAY)), ("primary_key_max", (False, DAY)),
])
def test_unsafe_leading_key_errors_are_not_hidden_by_nonoverlapping_time_bounds(
        committed, monkeypatch, field, value):
    _, _, base = committed
    record = _record(base, "unsafe-key", "2024-01-01", "2024-01-01", **{field: value})
    _assert_refusal_paths(committed, monkeypatch, [record], fail("MANIFEST_CORRUPT",
        "fragment primary-key bounds are invalid", details={"fragment_id": "unsafe-key"}).problem)


@pytest.mark.parametrize("lo, hi", [
    ((), (TICKER, DAY)), (("AAA",), ("AAA",)),
    ((TICKER, DAY), (TICKER,)), ((TICKER, DAY, "extra"), (TICKER, DAY)),
    (None, (TICKER, DAY)), ((TICKER, DAY), None),
    (TICKER, (TICKER, DAY)), ((TICKER, DAY), TICKER),
    ((TICKER, 7), (TICKER, DAY)), ((TICKER, DAY), (TICKER, False)),
    ((TICKER, {}), (TICKER, DAY)), ((None, 7), (TICKER, DAY)),
    (("ZZZ", DAY), ("AAA", DAY)), ((TICKER, "2024-01-01"), (TICKER, DAY)),
])
@pytest.mark.parametrize("placement", ["matching", "off-day", "off-partition"])
def test_full_key_admission_precedes_every_history_index_prune(
        committed, monkeypatch, lo, hi, placement):
    _, _, base = committed
    day = "2024-01-01" if placement == "off-day" else DAY
    partition = "OTHER" if placement == "off-partition" else TICKER
    bad = _record(base, "bad-bound", day, day, partition_key=partition,
                  primary_key_min=lo, primary_key_max=hi)
    _assert_refusal_paths(committed, monkeypatch, [_record(base, "valid"), bad],
        fail("MANIFEST_CORRUPT", "fragment primary-key bounds are invalid",
             details={"fragment_id": "bad-bound"}).problem)


@pytest.mark.parametrize("lo, hi", [
    (None, None), ((None, DAY), (TICKER, DAY)),
    ((TICKER, None), (TICKER, DAY)), ((TICKER, DAY), (None, DAY)),
    ((TICKER, DAY), (TICKER, None)),
])
@pytest.mark.parametrize("day, expected", [(DAY, ["null-key"]), ("2024-01-01", [])])
def test_null_key_bounds_remain_conservative_without_defeating_valid_time_pruning(
        committed, lo, hi, day, expected):
    _, _, base = committed
    record = _record(base, "null-key", primary_key_min=lo, primary_key_max=hi)
    population = _equivalent([record], day=day)
    assert _ids(population) == expected and population.row_count == len(expected)


def test_wrong_primary_key_day_does_not_prune_overlapping_time_bounds(committed):
    _, _, base = committed
    wrong_day = _record(base, "wrong-key-day", primary_key_min=(TICKER, "1999-01-01"),
        primary_key_max=(TICKER, "1999-01-01"))
    missing_key = _record(base, "missing-key", primary_key_min=None, primary_key_max=None)
    list_key = _record(base, "list-key", primary_key_min=[TICKER, DAY],
        primary_key_max=[TICKER, DAY])
    population = _equivalent([wrong_day, missing_key, list_key])
    assert _ids(population) == ["wrong-key-day", "missing-key", "list-key"]
    assert population.row_count == 3


@pytest.mark.parametrize("count", [True, -1, 1.5, "1", None])
@pytest.mark.parametrize("fallback_first", [False, True])
def test_bad_row_counts_preserve_first_manifest_failure_and_details(
        committed, count, fallback_first):
    _, _, base = committed
    canonical = _record(base, "bad-canonical", row_count=count)
    fallback = _record(base, "bad-fallback", None, None, row_count=count)
    records = [fallback, canonical] if fallback_first else [canonical, fallback]
    # Invalid counts in pruned records never supersede a surviving failure.
    records.insert(0, _record(base, "pruned-count", "2024-01-01", "2024-01-01", row_count=None))
    assert _equivalent(records) == fail("MANIFEST_CORRUPT",
        "a surviving fragment's recorded row_count is not valid metadata",
        details={"fragment_id": "bad-fallback" if fallback_first else "bad-canonical"}).problem


@pytest.mark.parametrize("key_first", [False, True])
@pytest.mark.parametrize("partition", [TICKER, "OTHER"])
def test_membership_failures_precede_row_count_validation_in_original_order(
        committed, monkeypatch, key_first, partition):
    _, _, base = committed
    bad_time = _record(base, "bad-time", "bad", DAY)
    bad_key = _record(base, "bad-key", "2024-01-01", "2024-01-01",
                      partition_key=partition, primary_key_min=(7, DAY))
    failures = [bad_key, bad_time] if key_first else [bad_time, bad_key]
    records = [_record(base, "bad-count", row_count=-1), *failures]
    expected = (fail("MANIFEST_CORRUPT", "fragment primary-key bounds are invalid",
                     details={"fragment_id": "bad-key"}) if key_first else
                fail("CONTRACT_MISMATCH", "a time bound is not a parseable date or timestamp"))
    _assert_refusal_paths(committed, monkeypatch, records, expected.problem)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("day", [DAY, "0001-01-01"])
def test_global_invalid_key_fallback_keeps_first_manifest_error_details(committed, reverse, day):
    _, _, base = committed
    records = [_record(base, "other-partition", partition_key="OTHER", primary_key_min=()),
        _record(base, "off-day", "2024-01-01", "2024-01-01", primary_key_max=(TICKER,))]
    if reverse:
        records.reverse()
    index = event_history._record_index(records)
    assert event_history._indexed_records(index, TICKER, day) == records
    assert _equivalent(records, day=day) == fail("MANIFEST_CORRUPT",
        "fragment primary-key bounds are invalid",
        details={"fragment_id": "off-day" if reverse else "other-partition"}).problem


def test_query_validation_precedes_deferred_global_key_failure(committed, monkeypatch):
    _, _, base = committed
    records = [_record(base, "bad-key", partition_key="OTHER", primary_key_min=())]
    keys = (KeyPredicate(column="unknown", operator="eq", values=(TICKER,)),)
    _assert_refusal_paths(committed, monkeypatch, records,
        fail("CONTRACT_MISMATCH", "'unknown' is not a filterable column").problem,
        key_filter=keys)


def test_irrelevant_malformed_metadata_and_future_min_preserve_short_circuit(committed, monkeypatch):
    repository, snapshot, base = committed
    records = [_record(base, "other-partition", "bad", "bad", partition_key="OTHER",
            primary_key_min=("OTHER", DAY), primary_key_max=("OTHER", DAY), row_count=None),
        _record(base, "future-min", "2024-01-01", "bad", row_count=None),
        _record(base, "other-leading-key", "bad", "bad", primary_key_min=("ZZZ", DAY),
            primary_key_max=("ZZZ", DAY), row_count=None),
        _record(base, "matching")]
    population = _equivalent(records)
    assert _ids(population) == ["matching"]
    assert population.row_count == 1
    monkeypatch.setattr(repository, "_table_records",
                        lambda *_args: (COMPUTED_MOVES_CONTRACT, records))
    monkeypatch.setattr(repository, "fragment_records", lambda *_args: records)
    plans = event_history._plans(repository, snapshot,
                                pd.DataFrame([history_event(DAY)]), COMPUTED_MOVES_CONTRACT)
    (_, request, selected), = plans
    assert selected == records[-1]
    assert repository.scan_population_bound(snapshot.snapshot_id, table_name="computed_moves",
        table_contract_ref=request.table_contract_ref, key_filter=request.key_filter,
        time_interval=request.time_interval) == 1
    explained = repository.explain_dependencies(request, table_name="computed_moves")
    assert [(item.fragment_ref.fragment_id, item.estimated_rows)
            for item in explained.dependencies] == [("matching", 1)]
    assert [row for batch in repository.scan(request, table_name="computed_moves")
            for row in batch.to_pylist()] == [{"event_date": DAY, "realized_move_pct": 2.0,
                "available_as_of_date": "2023-03-04", "skipped": False}]
    assert repository.read_counters.snapshot()["computed_moves"]["fragments_opened"] == 1


def test_late_competing_native_range_refuses_before_opening_earlier_safe_object(tmp_path, monkeypatch):
    first, second, target = history_event("2023-01-02"), history_event(DAY), history_event("2024-06-20")
    competing_days = (f"{DAY}T12:00:00", f"{DAY}T15:00:00")
    moves = [outcome_move(first), outcome_move(second), *[outcome_move(second, event_date=day) for day in competing_days]]
    conn, repository, snapshot = commit_event_history(tmp_path, [first, second, target], moves,
        mixed_days=competing_days)
    try:
        records = repository.fragment_records(snapshot, "computed_moves")
        first_population = _equivalent(records, day="2023-01-02")
        late_population = _equivalent(records)
        assert first_population.row_count == len(first_population.records) == 1
        assert late_population.row_count == 3 and len(late_population.records) == 2
        opened = poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
        with pytest.raises(DataError) as failure:
            read_event_history(repository, snapshot, target)
        assert failure.value.problem == fail("HOLDOUT_ACCESS_DENIED",
            "research requires event-isolated outcome fragments").problem
        assert all(name == "earnings_events" for name, _ in repository.scans)
        assert not ({record.object_ref.object_id for record in records} & set(opened))
    finally:
        conn.close()
