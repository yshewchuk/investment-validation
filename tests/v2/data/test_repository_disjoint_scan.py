"""Real Arrow scans retain only the current manifest-overlap group's streams."""
from dataclasses import replace
from datetime import datetime

import pytest

from engine.v2.contracts.data import ColumnContract, DataQuery, KeyPredicate, TableContract
from engine.v2.data import manifests, objects
from engine.v2.data import query as data_query
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_ref_for,
    hand_built_record,
    publish_and_inspect,
    table_from_rows,
)


def _contract(physical="int64", *, nullable_sequence=False):
    contract = TableContract(
        contract_id="disjoint_scan.v1", definition_hash="sha256:" + "0" * 64,
        table_name="disjoint_scan", semantic_version="1.0",
        columns=(ColumnContract(name="key", physical_type=physical, nullable=False),
                 ColumnContract(name="sequence", physical_type="int64", nullable=nullable_sequence),
                 ColumnContract(name="partition", physical_type="string", nullable=False),
                 ColumnContract(name="payload", physical_type="string", nullable=False)),
        primary_key=("key", "sequence"), duplicate_policy="reject", foreign_keys=(),
        partition_columns=("partition",), filterable_columns=("partition", "payload", "key"),
        orderable_columns=("key", "sequence"), finality_semantics="synthetic.v1",
        provenance_semantics="synthetic.v1", coverage_semantics="synthetic.v1",
        schema_evolution_policy="synthetic.v1", maximum_batch_rows=100)
    return replace(contract, definition_hash=manifests.table_contract_hash(contract))


@pytest.fixture
def storage(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    yield conn, clock, store
    conn.close()


def _rows(partition, keys):
    return [dict(key=key, sequence=sequence, partition=partition,
                 payload=f"{partition}/{index}")
            for index, (key, sequence) in enumerate(keys)]


def _build(storage, groups, *, physical="int64", hand_built=False, nullable_sequence=False):
    conn, clock, store = storage
    contract = _contract(physical, nullable_sequence=nullable_sequence)
    ref = contract_ref_for(contract)
    records, expected = [], []
    for index, keys in enumerate(groups):
        partition = f"p{index:03d}"
        rows = _rows(partition, keys)
        if hand_built:
            record = hand_built_record(
                store, contract, ref, table_from_rows(contract, rows),
                partition_key=partition, row_count=len(rows),
                primary_key_min=min(keys), primary_key_max=max(keys))
        else:
            record = publish_and_inspect(store, contract, ref, rows, partition)
        records.append(record)
        expected.extend(rows)
    snap = commit_tables(conn, clock, {contract.table_name: records},
                         {contract.table_name: contract})
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=ref,
        columns=("key", "sequence", "payload"),
        key_filter=(KeyPredicate(column="partition", operator="in",
                                 values=tuple(record.partition_key for record in records)),),
        order_by=contract.primary_key, max_batch_rows=2, max_result_rows=len(expected))
    expected = [{name: row[name] for name in query.columns}
                for row in sorted(expected, key=lambda row: (row["key"], row["sequence"]))]
    return Repository(conn, store), query, records, expected


def _scan(repository, query):
    return repository.scan(query, table_name="disjoint_scan")


def _track_streams(monkeypatch):
    state = {"active": 0, "maximum": 0, "opened": []}
    original = Repository._fragment_rows

    def tracked(self, record, *args):
        state["active"] += 1
        state["maximum"] = max(state["maximum"], state["active"])
        state["opened"].append(record.fragment_id)
        try:
            yield from original(self, record, *args)
        finally:
            state["active"] -= 1

    monkeypatch.setattr(Repository, "_fragment_rows", tracked)
    return state


def _replace_records(monkeypatch, records):
    original = Repository._table_records

    def replaced(self, *args):
        contract, _records = original(self, *args)
        return contract, records

    monkeypatch.setattr(Repository, "_table_records", replaced)


def test_disjoint_ranges_bound_streams_and_preserve_full_key_order(storage, monkeypatch):
    # The leading key is identical; full-PK ranges, not partition or leading-
    # key order, prove these 64 fragments can be opened one at a time.
    groups = [[(7, sequence), (7, sequence + 1)] for sequence in reversed(range(0, 128, 2))]
    repository, query, records, expected = _build(storage, groups)
    state = _track_streams(monkeypatch)
    scan = _scan(repository, query)
    first = next(scan)
    assert first.to_pylist() == expected[:2]
    assert state == {"active": 1, "maximum": 1, "opened": [records[-1].fragment_id]}
    batches = [first, *scan]
    assert [row for batch in batches for row in batch.to_pylist()] == expected
    assert all(batch.num_rows == 2 for batch in batches)
    assert state["maximum"] == 1 and state["active"] == 0
    assert state["opened"] == [record.fragment_id for record in reversed(records)]


def test_overlapping_and_nested_ranges_merge_only_current_group(storage, monkeypatch):
    groups = [[(20, 0), (21, 0)], [(1, 0), (10, 0)], [(2, 0), (3, 0)], [(4, 0), (11, 0)]]
    repository, query, records, expected = _build(storage, groups)
    state = _track_streams(monkeypatch)
    scan = _scan(repository, query)
    first = next(scan)
    assert first.to_pylist() == expected[:2]
    assert state["opened"] == [record.fragment_id for record in records[1:]]
    assert state["maximum"] == 3
    assert [row for batch in [first, *scan] for row in batch.to_pylist()] == expected
    assert state["maximum"] == 3 and state["active"] == 0
    assert state["opened"][-1] == records[0].fragment_id


def test_unknown_bounds_fall_back_to_global_merge_with_identical_rows(storage, monkeypatch):
    groups = [[(8, 0), (9, 0)], [(1, 0), (5, 0)], [(2, 0), (6, 0)]]
    repository, query, records, expected = _build(storage, groups)
    known = [row for batch in _scan(repository, query) for row in batch.to_pylist()]
    records[0] = replace(records[0], primary_key_min=None, primary_key_max=None)
    _replace_records(monkeypatch, records)
    state = _track_streams(monkeypatch)
    unknown = [row for batch in _scan(repository, query) for row in batch.to_pylist()]
    assert unknown == known == expected
    assert state["maximum"] == 3 and state["active"] == 0
    assert state["opened"] == [record.fragment_id for record in records]


@pytest.mark.parametrize("unknown", [False, True])
def test_duplicate_boundary_key_is_refused_globally(storage, monkeypatch, unknown):
    repository, query, records, _expected = _build(
        storage, [[(1, 0), (3, 0)], [(3, 0), (4, 0)]])
    if unknown:
        _replace_records(monkeypatch, [replace(record, primary_key_min=None, primary_key_max=None)
                                      for record in records])
    state = _track_streams(monkeypatch)
    with pytest.raises(DataError, match="scan result is not in strict primary-key order") as error:
        list(_scan(repository, query))
    assert error.value.code == "MANIFEST_CORRUPT"
    assert error.value.problem.message == "scan result is not in strict primary-key order"
    assert state["active"] == 0


@pytest.mark.parametrize("keys", [[(1, 0), (1, 0)], [(2, 0), (1, 0)]])
def test_fragment_duplicate_or_regression_is_still_refused(storage, keys):
    repository, query, _records, _expected = _build(storage, [keys], hand_built=True)
    with pytest.raises(DataError, match="scan result is not in strict primary-key order") as error:
        list(_scan(repository, query))
    assert error.value.code == "MANIFEST_CORRUPT"
    assert error.value.problem.message == "scan result is not in strict primary-key order"


@pytest.mark.parametrize("lo, hi", [
    ((), (2, 0)), ((1,), (2, 0)), ((1, 0), (2,)),
    ((1, 0), None), (None, (2, 0)), ((3, 0), (2, 0)),
    (("private-value", 0), (2, 0)), ((True, 0), (2, 0)),
    ((1.5, 0), (2, 0)),
])
@pytest.mark.parametrize("entry", ["scan", "bound", "explain", "plan", "prune"])
@pytest.mark.parametrize("selection", ["leading-key", "excluded-partition"])
def test_malformed_bounds_refuse_before_any_object_opens(
        storage, monkeypatch, lo, hi, entry, selection):
    repository, query, records, _expected = _build(storage, [[(1, 0), (2, 0)]])
    predicate = (KeyPredicate(column="key", operator="eq", values=(1,))
                 if selection == "leading-key" else
                 KeyPredicate(column="partition", operator="eq", values=("excluded",)))
    query = replace(query, key_filter=(predicate,))
    record = replace(records[0], primary_key_min=lo, primary_key_max=hi)
    _replace_records(monkeypatch, [record])
    state = _track_streams(monkeypatch)
    with pytest.raises(DataError) as error:
        if entry == "scan":
            list(_scan(repository, query))
        elif entry == "bound":
            repository.scan_population_bound(
                query.snapshot_id, table_name="disjoint_scan",
                table_contract_ref=query.table_contract_ref, key_filter=query.key_filter)
        elif entry == "explain":
            repository.explain_dependencies(query, table_name="disjoint_scan")
        elif entry == "plan":
            data_query.plan_scan_population(_contract(), [record], key_filter=query.key_filter)
        else:
            data_query.fragment_may_match(record, _contract(), query)
    assert error.value.code == "MANIFEST_CORRUPT"
    assert error.value.problem.message == "fragment primary-key bounds are invalid"
    assert error.value.problem.details == {"fragment_id": record.fragment_id}
    assert state == {"active": 0, "maximum": 0, "opened": []}


def test_false_disjoint_bound_refuses_out_of_range_row(storage, monkeypatch):
    repository, query, records, _expected = _build(storage, [[(1, 0), (10, 0)], [(5, 0), (6, 0)]])
    records[0] = replace(records[0], primary_key_max=(2, 0))
    _replace_records(monkeypatch, records)
    state = _track_streams(monkeypatch)
    with pytest.raises(DataError, match="scan key is outside fragment bounds") as error:
        list(_scan(repository, query))
    assert error.value.code == "MANIFEST_CORRUPT"
    assert error.value.problem.message == "scan key is outside fragment bounds"
    assert state["active"] == 0 and state["maximum"] == 1
    assert state["opened"] == [records[0].fragment_id]


@pytest.mark.parametrize("stop", ["close", "limit", "footer", "object"])
def test_disjoint_scan_cleanup_and_integrity_checks(storage, monkeypatch, stop):
    repository, query, records, expected = _build(storage, [[(1, 0), (2, 0)], [(3, 0), (4, 0)]])
    if stop == "limit":
        query = replace(query, max_result_rows=2)
    elif stop == "footer":
        records[1] = replace(records[1], row_count=3)
        _replace_records(monkeypatch, records)
    elif stop == "object":
        path = objects.verify_object_path(storage[2], records[1].object_ref)
        path.chmod(0o644)
        path.write_bytes(b"corrupt synthetic parquet")
    state = _track_streams(monkeypatch)
    scan = _scan(repository, query)
    assert next(scan).to_pylist() == expected[:2]
    assert state["opened"] == [records[0].fragment_id]
    if stop == "close":
        scan.close()
        assert state["opened"] == [records[0].fragment_id]
    else:
        expected_code = {"limit": "RESULT_LIMIT_EXCEEDED", "footer": "MANIFEST_CORRUPT",
                         "object": "OBJECT_CORRUPT"}[stop]
        expected_message = {
            "limit": "scan result exceeds max_result_rows",
            "footer": "parquet footer row count does not match the manifest",
            "object": "published object bytes do not match its recorded hash",
        }[stop]
        with pytest.raises(DataError) as error:
            list(scan)
        assert error.value.code == expected_code
        assert error.value.problem.message == expected_message
    assert state["active"] == 0 and state["maximum"] == 1


def test_timestamp_bounds_share_row_key_normalization(storage, monkeypatch):
    groups = [[(datetime(2024, 2, 1), 0), (datetime(2024, 2, 2), 0)],
              [(datetime(2024, 1, 1), 0), (datetime(2024, 1, 2), 0)]]
    repository, query, _records, expected = _build(storage, groups, physical="timestamp[us]")
    state = _track_streams(monkeypatch)
    assert [row for batch in _scan(repository, query) for row in batch.to_pylist()] == expected
    assert state["maximum"] == 1 and state["active"] == 0


@pytest.mark.parametrize("physical, keys", [
    ("string", ["A", "B", "C", "D"]),
    ("float64", [-2.5, -0.5, 1.5, 2.5]),
    ("bool", [False, False, True, True]),
])
def test_other_supported_key_types_keep_exact_order(storage, monkeypatch, physical, keys):
    groups = [[(keys[2], 2), (keys[3], 3)], [(keys[0], 0), (keys[1], 1)]]
    repository, query, _records, expected = _build(storage, groups, physical=physical)
    state = _track_streams(monkeypatch)
    assert [row for batch in _scan(repository, query) for row in batch.to_pylist()] == expected
    assert state["maximum"] == 1 and state["active"] == 0


@pytest.mark.parametrize("physical, bad", [
    ("float64", float("nan")), ("float64", float("inf")), ("float64", -float("inf")),
    ("timestamp[us]", "invalid-private-timestamp"), ("timestamp[us]", 1),
    ("timestamp[us]", "0001-01-01T00:00:00+01:00"),
    ("timestamp[us]", "9999-12-31T23:59:59-01:00"),
])
def test_invalid_float_and_timestamp_metadata_is_redacted(storage, monkeypatch, physical, bad):
    keys = [(1.0, 0), (2.0, 0)] if physical == "float64" else [
        (datetime(2024, 1, 1), 0), (datetime(2024, 1, 2), 0)]
    repository, query, records, _expected = _build(storage, [keys], physical=physical)
    _replace_records(monkeypatch, [replace(records[0], primary_key_min=(bad, 0))])
    with pytest.raises(DataError) as error:
        list(_scan(repository, query))
    assert error.value.code == "MANIFEST_CORRUPT"
    assert error.value.problem.message == "fragment primary-key bounds are invalid"
    assert error.value.problem.details == {"fragment_id": records[0].fragment_id}
    assert repository.read_counters.snapshot()["disjoint_scan"]["fragments_opened"] == 0


def test_hidden_keys_and_empty_filtered_groups_do_not_prime_later_fragments(storage, monkeypatch):
    repository, query, records, _expected = _build(
        storage, [[(1, 0), (2, 0)], [(3, 0), (4, 0)], [(5, 0), (6, 0)]])
    query = replace(query, columns=("payload",), max_batch_rows=1,
                    key_filter=(KeyPredicate(column="payload", operator="eq", values=("p001/0",)),))
    state = _track_streams(monkeypatch)
    scan = _scan(repository, query)
    assert next(scan).to_pylist() == [{"payload": "p001/0"}]
    assert state["opened"] == [record.fragment_id for record in records[:2]]
    assert list(scan) == []
    assert state["opened"] == [record.fragment_id for record in records]
    assert state["maximum"] == 1 and state["active"] == 0


@pytest.mark.parametrize("nullable", [False, True])
def test_null_key_bounds_are_conservative_on_every_planning_path(storage, monkeypatch, nullable):
    repository, query, records, expected = _build(
        storage, [[(8, None), (9, None)], [(1, None), (2, None)]], nullable_sequence=nullable)
    contract = repository.table_contract(repository.resolve(query.snapshot_id), "disjoint_scan")
    for record in records:
        assert data_query.fragment_key_bounds(record, contract) is None
    query = replace(query, key_filter=(KeyPredicate(column="key", operator="eq", values=(1,)),))
    assert repository.scan_population_bound(
        query.snapshot_id, table_name="disjoint_scan", table_contract_ref=query.table_contract_ref,
        key_filter=query.key_filter) == 4
    plan = repository.explain_dependencies(query, table_name="disjoint_scan")
    assert [(item.fragment_ref.fragment_id, item.estimated_rows) for item in plan.dependencies] == [
        (record.fragment_id, 2) for record in records]
    assert repository.read_counters.snapshot() == {}
    state = _track_streams(monkeypatch)
    assert [row for batch in _scan(repository, query) for row in batch.to_pylist()] == expected[:1]
    assert state["opened"] == [record.fragment_id for record in records]
    assert state["active"] == 0


@pytest.mark.parametrize("value, expected", [
    ("2024-01-01", datetime(2024, 1, 1)),
    ("2024-01-01T01:02:03.000004", datetime(2024, 1, 1, 1, 2, 3, 4)),
    ("2024-01-01T01:00:00+01:00", datetime(2024, 1, 1)),
    ("0001-01-01", datetime(1, 1, 1)),
])
def test_public_time_bound_parser_has_literal_utc_oracles(value, expected):
    assert data_query.parse_time_bound(value) == expected


@pytest.mark.parametrize("value", ["private-invalid", "2024-02-30", None,
                                   "0001-01-01T00:00:00+01:00",
                                   "9999-12-31T23:59:59-01:00"])
def test_public_time_bound_parser_refuses_with_redacted_problem(value):
    with pytest.raises(DataError) as error:
        data_query.parse_time_bound(value)
    assert error.value.code == "CONTRACT_MISMATCH"
    assert error.value.problem.message == "a time bound is not a parseable date or timestamp"
    assert error.value.problem.details == {}


def test_closing_overlap_group_closes_every_primed_stream(storage, monkeypatch):
    repository, query, records, expected = _build(
        storage, [[(1, 0), (10, 0)], [(2, 0), (9, 0)], [(20, 0), (21, 0)]])
    state = _track_streams(monkeypatch)
    scan = _scan(repository, query)
    assert next(scan).to_pylist() == expected[:2]
    assert state["active"] == 2
    scan.close()
    assert state == {"active": 0, "maximum": 2,
                     "opened": [record.fragment_id for record in records[:2]]}
