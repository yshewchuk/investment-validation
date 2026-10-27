"""Synthetic factory-owned admission and detached metadata lifetime tests."""
import sqlite3
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.data import _bounded_catalog_identity as identity
from engine.v2.data import bounded_snapshot, errors, objects
from engine.v2.data.repository import Repository
from engine.v2.research.snapshot_inputs import admit_snapshot
from tests.v2.data.bounded_catalog_identity_support import LIMITS, valid_catalog


@pytest.fixture
def fixture():
    conn, snapshot, oracle, _, records = valid_catalog()
    yield conn, snapshot, oracle, records
    conn.set_authorizer(None)
    conn.close()


def opened(fixture, *, repository=None, snapshot=None, **kwargs):
    conn, expected, _, _ = fixture
    return bounded_snapshot.open_bounded_snapshot(
        Repository(conn) if repository is None else repository,
        expected if snapshot is None else snapshot,
        table_name=kwargs.pop("table_name", "alpha"),
        limits=kwargs.pop("limits", dict(LIMITS)), **kwargs)


def assert_code(code, call):
    with pytest.raises(errors.DataError) as caught:
        call()
    assert caught.value.code == code
    return caught.value


def enter(context):
    with context:
        pytest.fail("invalid admission yielded a session")


def test_exact_native_oracle_after_rollback_no_sql_or_holder_cache(fixture, monkeypatch):
    conn, expected, oracle, records = fixture
    holder = Repository(conn, pin_snapshots=True)
    caches = ({"poison": object()}, {"poison": object()})
    holder._pinned_snapshots, holder._pinned_tables = caches
    def forbidden(*_, **__):
        pytest.fail("ordinary resolver or object access")
    for name in ("resolve", "resolve_full", "table_contract", "fragment_records", "scan"):
        monkeypatch.setattr(Repository, name, forbidden)
    monkeypatch.setattr(objects, "read_verified_bytes", forbidden)
    statements = []
    conn.set_trace_callback(statements.append)
    with opened(fixture, repository=holder) as session:
        assert not conn.in_transaction
        assert statements[0] == "BEGIN" and statements[-1] == "ROLLBACK"
        conn.set_authorizer(lambda *_: sqlite3.SQLITE_DENY)
        assert session.resolve(expected.snapshot_id) == expected
        assert admit_snapshot(session, expected) == expected
        actual = session.resolve_full(expected.snapshot_id)
        assert actual.snapshot == oracle.snapshot
        assert actual.table_manifests == oracle.table_manifests
        assert actual.contracts == tuple(sorted(oracle.contracts, key=lambda c: c.table_name))
        assert actual.records == tuple(r for name in sorted(records) for r in records[name])
        for name in ("alpha", "empty", "zeta"):
            contract = next(c for c in oracle.contracts if c.table_name == name)
            assert session.table_contract(expected, name) == contract
            assert session.fragment_records(expected, name) == records[name]
    assert holder._pinned_snapshots is caches[0] and holder._pinned_tables is caches[1]
    assert holder.pins_snapshots and holder._conn is conn
    assert conn.row_factory is sqlite3.Row and conn.text_factory is str
    assert statements.count("BEGIN") == statements.count("ROLLBACK") == 1


def test_returned_nested_values_and_input_ref_are_detached(fixture):
    _, expected, oracle, _ = fixture
    expected_copy = deepcopy(expected)
    with opened(fixture) as session:
        first = session.resolve_full(expected.snapshot_id)
        first.snapshot.table_versions.clear()
        first.snapshot.knowledge_mode_by_table.clear()
        first.table_manifests["alpha"].partition_logical_hashes["changed"] = "changed"
        first.table_manifests.clear()
        returned = session.resolve(expected.snapshot_id)
        returned.table_versions.clear()
        contract = session.table_contract(expected, "alpha")
        assert contract is not session.table_contract(expected, "alpha")
        records = session.fragment_records(expected, "alpha")
        assert records[0] is not session.fragment_records(expected, "alpha")[0]
        assert records[0].object_ref is not session.fragment_records(expected, "alpha")[0].object_ref
        expected.table_versions.clear()
        assert session.resolve(expected.snapshot_id) == expected_copy
        assert session.resolve_full(expected.snapshot_id).table_manifests == oracle.table_manifests
        assert session.table_contract(expected_copy, "alpha") == next(
            c for c in oracle.contracts if c.table_name == "alpha")


def test_active_and_closed_guards_precede_binding_checks(fixture):
    _, snapshot, _, _ = fixture
    with opened(fixture) as session:
        for method in (session.resolve, session.resolve_full):
            for value in ("unknown", None, [], 1):
                assert_code("SNAPSHOT_NOT_FOUND", lambda: method(value))
        for method in (session.table_contract, session.fragment_records):
            for ref, name in ((None, "alpha"), (replace(snapshot, calendar_version="other"), "alpha"),
                              (snapshot, "unknown"), (snapshot, [])):
                assert_code("CONTRACT_MISMATCH", lambda: method(ref, name))
    for method, args in ((session.resolve, (snapshot.snapshot_id,)),
                         (session.resolve_full, (None,)),
                         (session.table_contract, (None, [])),
                         (session.fragment_records, (snapshot, "alpha"))):
        assert_code("CONTRACT_MISMATCH", lambda: method(*args))


def test_consumer_failure_closes_session_without_rolling_back_caller_work(fixture):
    conn, expected, _, _ = fixture
    primary = RuntimeError("synthetic consumer failure")
    with pytest.raises(RuntimeError) as caught:
        with opened(fixture) as session:
            conn.execute("BEGIN")  # Admission has already released its transaction.
            raise primary
    assert caught.value is primary and conn.in_transaction
    assert_code("CONTRACT_MISMATCH", lambda: session.resolve(expected.snapshot_id))
    conn.rollback()


@pytest.mark.parametrize("kind", ["proxy", "repository_subclass", "connection_subclass",
                                  "row_factory", "text_factory", "active_transaction"])
def test_invalid_holder_configuration_is_rejected_without_rollback(fixture, kind):
    conn, expected, _, _ = fixture
    holder = Repository(conn)
    other = None
    if kind == "proxy":
        holder = SimpleNamespace(_conn=conn, _store=None)
    elif kind == "repository_subclass":
        class CustomRepository(Repository):
            pass
        holder = CustomRepository(conn)
    elif kind == "connection_subclass":
        class CustomConnection(sqlite3.Connection):
            pass
        other = sqlite3.connect(":memory:", factory=CustomConnection)
        other.row_factory = sqlite3.Row
        holder = Repository(other)
    elif kind == "row_factory":
        conn.row_factory = lambda _, row: row
    elif kind == "text_factory":
        conn.text_factory = bytes
    else:
        conn.execute("BEGIN")
    statements = []
    conn.set_trace_callback(statements.append)
    assert_code("CONTRACT_MISMATCH", lambda: enter(opened(fixture, repository=holder)))
    assert statements == []
    assert conn.in_transaction == (kind == "active_transaction")
    if other:
        other.close()


@pytest.mark.parametrize("snapshot", [None, "snapshot-id", {}, 1])
def test_non_native_snapshot_ref_fails_before_sql(fixture, snapshot):
    conn = fixture[0]
    statements = []
    conn.set_trace_callback(statements.append)
    assert_code("SNAPSHOT_UNRESOLVED", lambda: enter(
        bounded_snapshot.open_bounded_snapshot(Repository(conn), snapshot,
            table_name="alpha", limits=LIMITS)))
    assert statements == []


@pytest.mark.parametrize("changes", [{"snapshot_id": ""}, {"snapshot_id": "\ud800"},
                                     {"manifest_hash": "bad"}, {"table_versions": []}])
def test_malformed_native_snapshot_ref_fails_before_sql(fixture, changes):
    conn, expected, _, _ = fixture
    statements = []
    conn.set_trace_callback(statements.append)
    assert_code("SNAPSHOT_UNRESOLVED", lambda: enter(opened(
        fixture, snapshot=replace(expected, **changes))))
    assert statements == []


@pytest.mark.parametrize("kwargs", [{"table_name": ""}, {"table_name": None},
    {"table_name": 1}, {"limits": {}}, {"limits": {**LIMITS, "max_fragments": True}},
    {"limits": {**LIMITS, "max_catalog_bytes": 0}}])
def test_invalid_request_fails_before_transaction(fixture, kwargs):
    conn = fixture[0]
    statements = []
    conn.set_trace_callback(statements.append)
    assert_code("CONTRACT_MISMATCH", lambda: enter(opened(fixture, **kwargs)))
    assert statements == []


@pytest.mark.parametrize("case,code", [("missing_table", "CONTRACT_MISMATCH"),
    ("missing_snapshot", "SNAPSHOT_UNRESOLVED"), ("nonexact", "SNAPSHOT_UNRESOLVED"),
    ("cap", "RESULT_LIMIT_EXCEEDED"), ("schema", "MANIFEST_CORRUPT"),
    ("json", "MANIFEST_CORRUPT"), ("unrelated_identity", "MANIFEST_CORRUPT")])
def test_admission_failures_cleanup_owned_transaction(fixture, case, code):
    conn, snapshot, _, _ = fixture
    kwargs = {}
    conn.execute("DROP TRIGGER data_dataset_versions_no_update")
    if case == "missing_table":
        kwargs["table_name"] = "missing"
    elif case == "missing_snapshot":
        kwargs["snapshot"] = replace(snapshot, snapshot_id="missing")
    elif case == "nonexact":
        kwargs["snapshot"] = replace(snapshot, calendar_version="different")
    elif case == "cap":
        kwargs["limits"] = {**LIMITS, "max_catalog_tables": 2}
    elif case == "schema":
        conn.execute("ALTER TABLE data_contracts RENAME TO absent_contracts")
    elif case == "json":
        conn.execute("UPDATE data_dataset_versions SET evidence_json='{}' ")
    else:
        conn.execute("UPDATE data_dataset_versions SET row_count=99 WHERE contract_id='synthetic.zeta.v2'")
    conn.commit()
    assert_code(code, lambda: enter(opened(fixture, **kwargs)))
    assert not conn.in_transaction
    assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_reopen_reauthenticates_catalog_not_old_session(fixture):
    conn, snapshot, _, _ = fixture
    holder = Repository(conn, pin_snapshots=True)
    conn.execute("DROP TRIGGER data_dataset_versions_no_update")
    with opened(fixture, repository=holder) as first:
        assert first.resolve(snapshot.snapshot_id) == snapshot
    conn.execute("UPDATE data_dataset_versions SET row_count=99 WHERE contract_id='synthetic.zeta.v2'")
    conn.commit()
    assert_code("MANIFEST_CORRUPT", lambda: enter(opened(fixture, repository=holder)))
    assert not conn.in_transaction


@pytest.mark.parametrize("primary_failure", [False, True])
def test_rollback_failure_never_hides_primary_and_never_yields(fixture, primary_failure):
    conn, _, _, _ = fixture
    if primary_failure:
        conn.execute("DROP TRIGGER data_dataset_versions_no_update")
        conn.execute("UPDATE data_dataset_versions SET evidence_json='{}' ")
        conn.commit()
    def deny_rollback(action, arg1, *_):
        return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_TRANSACTION and arg1 == "ROLLBACK" else sqlite3.SQLITE_OK
    conn.set_authorizer(deny_rollback)
    caught = assert_code("MANIFEST_CORRUPT", lambda: enter(opened(fixture)))
    if primary_failure:
        assert caught.problem.message == "catalog transfer malformed"
    assert conn.in_transaction
    conn.set_authorizer(None)
    conn.rollback()


def test_session_cannot_be_constructed_directly():
    assert_code("CONTRACT_MISMATCH", bounded_snapshot.BoundedSnapshot)


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_native_failure_identity_envelope_and_no_logs(fixture, monkeypatch, capsys, caplog,
                                                      cleanup_failure):
    conn = fixture[0]
    primary = errors.fail("IDENTITY_CONFLICT", "synthetic native detail", stage="identity",
                          details={"synthetic": ["full diagnostic", 1]})
    def fail(*_, **__):
        raise primary
    monkeypatch.setattr(identity, "reconstruct_catalog", fail)
    if cleanup_failure:
        conn.set_authorizer(lambda action, arg1, *_: sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_TRANSACTION and arg1 == "ROLLBACK" else sqlite3.SQLITE_OK)
    caught = assert_code("IDENTITY_CONFLICT", lambda: enter(opened(fixture)))
    assert caught is primary and caught.problem == primary.problem
    assert conn.in_transaction == cleanup_failure
    assert capsys.readouterr() == ("", "") and not caplog.records


@pytest.mark.parametrize("primary_code", [sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED,
    sqlite3.SQLITE_IOERR, sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_FULL, sqlite3.SQLITE_NOMEM,
    sqlite3.SQLITE_CORRUPT, None])
def test_new_sql_errors_use_numeric_code_and_cleanup(fixture, monkeypatch, primary_code):
    error = sqlite3.OperationalError("ignored synthetic diagnostic")
    if primary_code is not None:
        error.sqlite_errorcode = primary_code | 256  # Exercise extended result codes.
    def fail(*_, **__):
        raise error
    monkeypatch.setattr(identity, "reconstruct_catalog", fail)
    resource = primary_code not in (None, sqlite3.SQLITE_CORRUPT)
    caught = assert_code("RESOURCE_UNAVAILABLE" if resource else "MANIFEST_CORRUPT",
                         lambda: enter(opened(fixture)))
    assert caught.problem.retryable == resource
    assert "ignored" not in caught.problem.message
    assert caught.__suppress_context__
    assert not fixture[0].in_transaction


def test_failed_begin_does_not_rollback_or_yield(fixture):
    conn = fixture[0]
    transactions = []
    def deny_begin(action, arg1, *_):
        if action == sqlite3.SQLITE_TRANSACTION:
            transactions.append(arg1)
            if arg1 == "BEGIN":
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    conn.set_authorizer(deny_begin)
    assert_code("MANIFEST_CORRUPT", lambda: enter(opened(fixture)))
    assert transactions == ["BEGIN"] and not conn.in_transaction


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("mode", ["autocommit", "legacy_none"])
def test_owned_sql_transaction_is_closed_in_autocommit_modes(fixture, failure, mode):
    conn, snapshot, _, _ = fixture
    if mode == "autocommit":
        if not hasattr(conn, "autocommit"):
            pytest.skip("sqlite3 autocommit requires Python 3.12+")
        conn.autocommit = True
    else:
        conn.isolation_level = None
    kwargs = {"limits": {**LIMITS, "max_catalog_tables": 2}} if failure else {}
    if failure:
        assert_code("RESULT_LIMIT_EXCEEDED", lambda: enter(opened(fixture, **kwargs)))
    else:
        with opened(fixture) as session:
            assert not conn.in_transaction
            assert session.resolve(snapshot.snapshot_id) == snapshot
    assert not conn.in_transaction


def test_pep249_autocommit_false_is_existing_caller_transaction(fixture):
    conn = fixture[0]
    if not hasattr(conn, "autocommit"):
        pytest.skip("sqlite3 autocommit requires Python 3.12+")
    conn.autocommit = False
    assert conn.in_transaction
    statements = []
    conn.set_trace_callback(statements.append)
    assert_code("CONTRACT_MISMATCH", lambda: enter(opened(fixture)))
    assert conn.in_transaction and statements == []


@pytest.mark.parametrize("kind", ["receipt_list", "dataset_dict", "contract_dict",
                                  "nested_subclass", "coerced_key"])
def test_normalized_reference_impostors_refuse_before_any_sql(fixture, kind):
    from engine.v2.data import documents
    from engine.v2.foundation import to_document
    from tests.v2.data.bounded_catalog_identity_support import normalized_snapshot_impostor
    conn, expected, _, _ = fixture
    supplied = normalized_snapshot_impostor(expected, kind)
    assert supplied != expected
    assert documents.decode_document(type(expected), to_document(supplied)) == expected
    statements = []
    conn.set_trace_callback(statements.append)
    caught = assert_code("SNAPSHOT_UNRESOLVED", lambda: enter(opened(fixture, snapshot=supplied)))
    problem = caught.problem
    assert (problem.category, problem.retryable, problem.stage, problem.message, problem.details) == (
        "dependency", False, None, "invalid expected snapshot", {})
    assert statements == [] and not conn.in_transaction
