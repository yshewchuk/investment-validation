"""Isolated journal limits and typed refusals over real SQLite and artifact storage."""
import errno
import json
import sqlite3
import traceback
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    SystemClock,
    artifact_reference,
    canonical_json,
    from_document,
    to_document,
)
from engine.v2.foundation import artifacts as artifact_store
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError
from engine.v2.ops.migrations import migrate
from experiments import magnitude_search_records as records
from experiments.magnitude_search_errors import SearchArtifactError, artifact_failure

RUN = "run-search"
SLOT = "start:trial"


@pytest.fixture
def journal(tmp_path):
    conn = connect(tmp_path / "catalog.sqlite")
    conn.execute("CREATE TABLE experiment_runs (run_id TEXT PRIMARY KEY) STRICT")
    conn.execute("INSERT INTO experiment_runs VALUES (?)", (RUN,))
    migrate(conn, records.OWNER, records.MIGRATIONS, clock=SystemClock())
    store = ArtifactStore(tmp_path / "store")
    store.root.mkdir()
    try:
        yield SimpleNamespace(conn=conn, store=store)
    finally:
        conn.close()


def _wrapper():
    return {"schema_version": records.SCHEMA, "run_id": RUN, "slot": SLOT,
            "body": {"forecasts": [1, 2, 3]}}


def _insert(journal, ref_json):
    journal.conn.execute("INSERT INTO magnitude_search_records VALUES (?,?,?)",
                         (RUN, SLOT, ref_json))


def _publish(journal, payload):
    ref = journal.store.publish_bytes(payload, schema_ref=records.SCHEMA)
    _insert(journal, canonical_json(to_document(ref)))
    return ref


@contextmanager
def _problem(code, message):
    with pytest.raises(OpsError) as caught:
        yield caught
    category, retryable = {"RESOURCE_LIMIT_EXCEEDED": ("resource", False),
        "RESOURCE_UNAVAILABLE": ("resource", True), "EXPERIMENT_IDENTITY_CONFLICT": ("validation", False)}[code]
    assert to_document(caught.value.problem) == {
        "code": code,
        "category": category,
        "retryable": retryable,
        "message": message,
        "stage": None,
        "trace_id": None,
        "dependency_refs": [],
        "retry_after_seconds": None,
        "diagnostic_ref": None,
        "details": {},
        "schema_version": "problem.v1.0",
    }


def _journal_state(journal):
    return (journal.conn.execute("SELECT * FROM magnitude_search_records ORDER BY slot").fetchall(),
            journal.conn.total_changes)


def _assert_redacted_artifact_error(error):
    assert type(error) is SearchArtifactError
    assert error.__cause__ is None
    assert error.__suppress_context__
    diagnostic = "".join(traceback.format_exception(error))
    assert "/private/credential" not in diagnostic
    assert "token=secret" not in diagnostic


def test_journal_default_limit_is_sixteen_mebibytes():
    assert records.MAX_BYTES == 16 * 1024 * 1024


@pytest.mark.parametrize("artifact_code, unsafe_is_conflict, code, category, retryable, message", [
    ("MISSING", False, "RESOURCE_UNAVAILABLE", "resource", True,
     "caller resource failure"),
    ("MISSING", True, "RESOURCE_UNAVAILABLE", "resource", True,
     "caller resource failure"),
    ("IO_ERROR", False, "RESOURCE_UNAVAILABLE", "resource", True,
     "caller resource failure"),
    ("IO_ERROR", True, "RESOURCE_UNAVAILABLE", "resource", True,
     "caller resource failure"),
    ("INTEGRITY_FAILED", False, "EXPERIMENT_IDENTITY_CONFLICT", "validation", False,
     "caller integrity failure"),
    ("INTEGRITY_FAILED", True, "EXPERIMENT_IDENTITY_CONFLICT", "validation", False,
     "caller integrity failure"),
    ("UNSAFE_PATH", False, "RESOURCE_UNAVAILABLE", "resource", True,
     "caller resource failure"),
    ("UNSAFE_PATH", True, "EXPERIMENT_IDENTITY_CONFLICT", "validation", False,
     "caller integrity failure"),
])
def test_artifact_failure_policy_has_exact_sanitized_problem(
    artifact_code, unsafe_is_conflict, code, category, retryable, message,
):
    error = ArtifactError(artifact_code, "/private/credential: token=secret")
    failure = artifact_failure(
        error,
        integrity_message="caller integrity failure",
        resource_message="caller resource failure",
        unsafe_is_conflict=unsafe_is_conflict,
    )

    assert type(failure) is SearchArtifactError
    assert isinstance(failure, OpsError)
    assert str(failure) == f"{code}: {message}"
    assert to_document(failure.problem) == {
        "code": code,
        "category": category,
        "retryable": retryable,
        "message": message,
        "stage": None,
        "trace_id": None,
        "dependency_refs": [],
        "retry_after_seconds": None,
        "diagnostic_ref": None,
        "details": {},
        "schema_version": "problem.v1.0",
    }


@pytest.mark.parametrize("size", [255, 256, 257])
def test_write_cap_counts_utf8_bytes_and_accepts_exact_limit(journal, monkeypatch, size):
    monkeypatch.setattr(records, "MAX_BYTES", 256)
    wrapper = {**_wrapper(), "body": {"text": ""}}
    overhead = len(canonical_json(wrapper).encode())
    wrapper["body"]["text"] = "é" + "x" * (size - overhead - 2)
    payload = canonical_json(wrapper).encode()
    assert len(payload) == size
    assert len(payload.decode()) < size
    if size > 256:
        with _problem("RESOURCE_LIMIT_EXCEEDED", "search journal record exceeds its byte limit"):
            records.append_record(journal.conn, journal.store, RUN, SLOT, wrapper["body"])
        assert journal.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0] == 0
        assert not (journal.store.root / "objects").exists()
    else:
        assert records.append_record(journal.conn, journal.store, RUN, SLOT, wrapper["body"]) == wrapper["body"]
        assert records.read_record(journal.conn, journal.store, RUN, SLOT) == wrapper["body"]


@pytest.mark.parametrize("allowance", [-1, 0, 1])
def test_reference_byte_cap_is_independent_of_artifact_integrity(journal, monkeypatch, allowance):
    payload = canonical_json(_wrapper()).encode()
    ref = _publish(journal, payload)
    # Lower only the admission cap: altered reference sizes would fail store
    # integrity even if the journal's own size guard disappeared.
    assert journal.store.read_verified(ref) == payload
    monkeypatch.setattr(records, "MAX_BYTES", len(payload) + allowance)
    if allowance < 0:
        with _problem("EXPERIMENT_IDENTITY_CONFLICT", "search journal reference is invalid"):
            records.read_record(journal.conn, journal.store, RUN, SLOT)
    else:
        assert records.read_record(journal.conn, journal.store, RUN, SLOT) == _wrapper()["body"]


@pytest.mark.parametrize("size", [4095, 4096, 4097])
def test_reference_json_cap_accepts_exact_limit_with_otherwise_valid_reference(journal, size):
    payload = canonical_json(_wrapper()).encode()
    ref = journal.store.publish_bytes(payload, schema_ref=records.SCHEMA)
    document = canonical_json(to_document(ref))
    # JSON whitespace crosses the SQL limit without creating malformed JSON,
    # changing the reference, or requiring an oversized backing artifact.
    document += " " * (size - len(document.encode()))
    assert len(document.encode()) == size
    assert from_document(ArtifactRef, json.loads(document)) == ref
    assert journal.store.read_verified(ref) == payload
    _insert(journal, document)
    if size > 4096:
        with _problem("EXPERIMENT_IDENTITY_CONFLICT", "search journal reference is invalid"):
            records.read_record(journal.conn, journal.store, RUN, SLOT)
    else:
        assert records.read_record(journal.conn, journal.store, RUN, SLOT) == _wrapper()["body"]


@pytest.mark.parametrize("mutation", [
    "malformed_json", "null", "array", "scalar", "missing_field", "extra_field",
    "wrong_version", "wrong_schema", "zero_size", "negative_size", "boolean_size",
    "string_size", "float_size", "null_size", "wrong_key_type",
])
def test_invalid_references_have_exact_redacted_problem(journal, monkeypatch, mutation):
    payload = canonical_json(_wrapper()).encode()
    ref = journal.store.publish_bytes(payload, schema_ref=records.SCHEMA)
    document = to_document(ref)
    if mutation == "missing_field":
        del document["content_hash"]
    elif mutation == "extra_field":
        document["private"] = "/private/credential"
    elif mutation in ("wrong_version", "wrong_schema", "wrong_key_type"):
        field, value = {"wrong_version": ("schema_version", "artifact_ref.v2.0"),
                        "wrong_schema": ("schema_ref", "private_schema.v1"),
                        "wrong_key_type": ("storage_key", None)}[mutation]
        document[field] = value
    elif mutation.endswith("_size"):
        document["byte_size"] = {"zero_size": 0, "negative_size": -1, "boolean_size": True,
                                 "string_size": "1", "float_size": 1.5, "null_size": None}[mutation]
    encoded = {"malformed_json": "{private", "null": "null", "array": "[]",
               "scalar": '"/private/credential"'}.get(mutation, canonical_json(document))
    _insert(journal, encoded)
    monkeypatch.setattr(journal.store, "read_verified", lambda *a: pytest.fail("invalid reference opened"))
    with _problem("EXPERIMENT_IDENTITY_CONFLICT", "search journal reference is invalid"):
        records.read_record(journal.conn, journal.store, RUN, SLOT)


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("storage_key", [
    "", "/private/credential", "../private/credential", "objects//credential",
    "./private/credential", "objects/./credential", "objects/../credential",
    "objects\\private\\credential", "objects/\x00credential", "objects/credential/",
])
def test_unsafe_reference_syntax_is_rejected_before_artifact_read(
    journal, monkeypatch, replay, storage_key,
):
    body = _wrapper()["body"]
    ref = artifact_reference(canonical_json(_wrapper()).encode(), records.SCHEMA)
    _insert(journal, canonical_json(to_document(replace(ref, storage_key=storage_key))))
    before = _journal_state(journal)
    with monkeypatch.context() as patch:
        patch.setattr(journal.store, "read_verified", lambda *a: pytest.fail("invalid reference read"))
        patch.setattr(artifact_store.os, "open", lambda *a, **kw: pytest.fail("invalid reference opened"))
        with _problem("EXPERIMENT_IDENTITY_CONFLICT", "search journal reference is invalid"):
            if replay:
                records.append_record(journal.conn, journal.store, RUN, SLOT, body)
            else:
                records.read_record(journal.conn, journal.store, RUN, SLOT)
    assert _journal_state(journal) == before
    assert not journal.conn.in_transaction


@pytest.mark.parametrize("mutation", [
    "schema_version", "run_id", "slot", "extra_field", "missing_schema_version",
    "missing_run_id", "missing_slot", "missing_body", "malformed_json", "null",
    "array", "scalar", "field_array",
])
def test_wrapper_requires_exact_schema_run_slot_and_fields(journal, mutation):
    wrapper = _wrapper()
    if mutation in ("schema_version", "run_id", "slot"):
        wrapper[mutation] = "/private/credential"
    elif mutation == "extra_field":
        wrapper["private"] = "/private/credential"
    elif mutation.startswith("missing_"):
        del wrapper[mutation.removeprefix("missing_")]
    payload = {"malformed_json": b"{private", "null": b"null", "array": b"[]",
               "scalar": b"42", "field_array": b'["schema_version","run_id","slot","body"]'}.get(
                   mutation, canonical_json(wrapper).encode())
    ref = _publish(journal, payload)
    assert journal.store.read_verified(ref) == payload
    with _problem("EXPERIMENT_IDENTITY_CONFLICT", "search journal binding is invalid"):
        records.read_record(journal.conn, journal.store, RUN, SLOT)


@pytest.mark.parametrize("failure, code, message", [
    ("missing", "RESOURCE_UNAVAILABLE", "search journal artifact could not be verified"),
    ("wrong_size", "EXPERIMENT_IDENTITY_CONFLICT", "search journal artifact could not be verified"),
    ("corrupt_bytes", "EXPERIMENT_IDENTITY_CONFLICT", "search journal artifact could not be verified"),
    ("unsafe_path", "EXPERIMENT_IDENTITY_CONFLICT", "search journal reference is invalid"),
    ("nonobject_key", "EXPERIMENT_IDENTITY_CONFLICT", "search journal artifact could not be verified"),
    ("wrong_shard", "EXPERIMENT_IDENTITY_CONFLICT", "search journal artifact could not be verified"),
    ("nonregular_object", "EXPERIMENT_IDENTITY_CONFLICT", "search journal artifact could not be verified"),
])
def test_artifact_read_failures_preserve_classification_and_redact_paths(journal, monkeypatch, failure, code, message):
    payload = canonical_json(_wrapper()).encode()
    ref = artifact_reference(payload, records.SCHEMA)
    if failure != "missing":
        assert journal.store.publish_bytes(payload, schema_ref=records.SCHEMA) == ref
    if failure == "wrong_size":
        ref = replace(ref, byte_size=ref.byte_size - 1)
    elif failure == "corrupt_bytes":
        path = journal.store.root / ref.storage_key
        path.chmod(0o644)
        path.write_bytes(b"x" * len(payload))
    elif failure == "unsafe_path":
        ref = replace(ref, storage_key="../private/credential")
    elif failure == "nonobject_key":
        ref = replace(ref, storage_key="attempts/private/credential")
    elif failure == "wrong_shard":
        parts = ref.storage_key.split("/")
        ref = replace(ref, storage_key=f"objects/{parts[1]}x/{parts[2]}")
    elif failure == "nonregular_object":
        path = journal.store.root / ref.storage_key
        path.rename(path.with_suffix(".original"))
        path.mkdir()
    _insert(journal, canonical_json(to_document(ref)))
    before = _journal_state(journal)
    with monkeypatch.context() as guard:
        if failure == "unsafe_path":
            guard.setattr(artifact_store.os, "open", lambda *a, **k: pytest.fail("unsafe reference attempted a filesystem open"))
        with _problem(code, message):
            records.read_record(journal.conn, journal.store, RUN, SLOT)
    assert _journal_state(journal) == before


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("number", [errno.EACCES, errno.EMFILE, errno.EIO])
@pytest.mark.parametrize("point", ["base", "objects", "shard", "object"])
def test_read_open_failures_are_retryable_after_reference_admission(
    journal, monkeypatch, replay, number, point,
):
    body = _wrapper()["body"]
    ref = _publish(journal, canonical_json(_wrapper()).encode())
    before = _journal_state(journal)
    parts = ref.storage_key.split("/")
    paths = [journal.store.root, *parts]
    index = ["base", "objects", "shard", "object"].index(point)
    real_open = artifact_store.os.open
    calls = []

    def fault(path, flags, *args, **kwargs):
        assert not journal.conn.in_transaction
        calls.append(path)
        if path == paths[index]:
            raise OSError(number, "token=secret", "/private/credential")
        return real_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(artifact_store.os, "open", fault)
        patch.setattr(artifact_store.os, "read", lambda *a: pytest.fail("failed open read bytes"))
        patch.setattr(journal.store, "publish_bytes", lambda *a, **kw: pytest.fail("failed replay published"))
        with _problem("RESOURCE_UNAVAILABLE", "search journal artifact could not be verified") as caught:
            if replay:
                records.append_record(journal.conn, journal.store, RUN, SLOT, body)
            else:
                records.read_record(journal.conn, journal.store, RUN, SLOT)
    _assert_redacted_artifact_error(caught.value)
    assert calls == paths[:index + 1]
    assert _journal_state(journal) == before
    assert records.read_record(journal.conn, journal.store, RUN, SLOT) == body
    assert records.append_record(journal.conn, journal.store, RUN, SLOT, body) == body
    assert _journal_state(journal) == before
    assert not journal.conn.in_transaction


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("point", ["base", "intermediate", "object"])
def test_real_symlink_is_retryable_without_reading_its_target(
    journal, monkeypatch, tmp_path, replay, point,
):
    body = _wrapper()["body"]
    ref = _publish(journal, canonical_json(_wrapper()).encode())
    before = _journal_state(journal)
    path = {"base": journal.store.root, "intermediate": journal.store.root / "objects",
            "object": journal.store.root / ref.storage_key}[point]
    target = tmp_path / "private-credential-token=secret"
    path.rename(target)
    path.symlink_to(target, target_is_directory=point != "object")
    with monkeypatch.context() as patch:
        patch.setattr(artifact_store.os, "read", lambda *a: pytest.fail("symlink target payload read"))
        patch.setattr(journal.store, "publish_bytes", lambda *a, **kw: pytest.fail("failed replay published"))
        with pytest.raises(ArtifactError) as foundation_error:
            journal.store.read_verified(ref)
        assert foundation_error.value.code == "UNSAFE_PATH"
        with _problem("RESOURCE_UNAVAILABLE", "search journal artifact could not be verified") as caught:
            if replay:
                records.append_record(journal.conn, journal.store, RUN, SLOT, body)
            else:
                records.read_record(journal.conn, journal.store, RUN, SLOT)
    _assert_redacted_artifact_error(caught.value)
    assert _journal_state(journal) == before
    path.unlink()
    target.rename(path)
    assert records.read_record(journal.conn, journal.store, RUN, SLOT) == body
    assert records.append_record(journal.conn, journal.store, RUN, SLOT, body) == body
    assert _journal_state(journal) == before
    assert not journal.conn.in_transaction


@pytest.mark.parametrize("artifact_code, code", [
    ("MISSING", "RESOURCE_UNAVAILABLE"),
    ("IO_ERROR", "RESOURCE_UNAVAILABLE"),
    ("INTEGRITY_FAILED", "EXPERIMENT_IDENTITY_CONFLICT"),
    ("UNSAFE_PATH", "EXPERIMENT_IDENTITY_CONFLICT"),
])
def test_publication_failures_leave_no_journal_association(journal, artifact_code, code):
    def fault(point):
        assert not journal.conn.in_transaction
        assert point == "copied"
        raise ArtifactError(artifact_code, "/private/credential")

    store = ArtifactStore(journal.store.root, fault=fault)
    with _problem(code, "search journal artifact could not be published"):
        records.append_record(journal.conn, store, RUN, SLOT, _wrapper()["body"])
    assert journal.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0] == 0
    assert records.read_record(journal.conn, journal.store, RUN, SLOT) is None
    assert not journal.conn.in_transaction


@pytest.mark.parametrize("point", ["copied", "linked"])
@pytest.mark.parametrize("number", [errno.ENOSPC, errno.EIO, errno.EACCES])
def test_publication_oserror_has_exact_redacted_problem_and_no_journal_mutation(journal, point, number):
    records.append_record(journal.conn, journal.store, RUN, "seal", {"existing": True})
    before = _journal_state(journal)
    visited = []

    def fault(actual):
        assert not journal.conn.in_transaction
        visited.append(actual)
        if actual == point:
            raise OSError(number, "token=secret", "/private/credential")

    store = ArtifactStore(journal.store.root, fault=fault)
    with _problem("RESOURCE_UNAVAILABLE", "search journal artifact could not be published") as caught:
        records.append_record(journal.conn, store, RUN, SLOT, _wrapper()["body"])
    _assert_redacted_artifact_error(caught.value)
    assert visited == (["copied"] if point == "copied" else ["copied", "linked"])
    assert _journal_state(journal) == before
    assert records.read_record(journal.conn, journal.store, RUN, SLOT) is None
    assert not journal.conn.in_transaction


@pytest.mark.parametrize("operation,number", [
    ("write", errno.ENOSPC), ("fsync", errno.EIO),
    ("chmod", errno.EACCES), ("link", errno.EIO),
])
def test_publication_syscall_oserror_preserves_journal(journal, monkeypatch, operation, number):
    records.append_record(journal.conn, journal.store, RUN, "seal", {"existing": True})
    before = _journal_state(journal)
    calls = []

    def fault(*args, **kwargs):
        assert not journal.conn.in_transaction
        calls.append(operation)
        raise OSError(number, "token=secret", "/private/credential")

    with monkeypatch.context() as patch:
        patch.setattr(artifact_store.os, operation, fault)
        with _problem("RESOURCE_UNAVAILABLE", "search journal artifact could not be published") as caught:
            records.append_record(journal.conn, journal.store, RUN, SLOT, _wrapper()["body"])
    _assert_redacted_artifact_error(caught.value)
    assert calls == [operation]
    assert _journal_state(journal) == before
    assert records.read_record(journal.conn, journal.store, RUN, SLOT) is None
    assert not journal.conn.in_transaction


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("number", [errno.EIO, errno.EACCES])
def test_read_syscall_oserror_has_exact_redacted_problem_and_no_journal_mutation(
    journal, monkeypatch, replay, number,
):
    body = _wrapper()["body"]
    records.append_record(journal.conn, journal.store, RUN, SLOT, body)
    before = _journal_state(journal)
    calls = []

    def fault(fd, size):
        assert not journal.conn.in_transaction
        calls.append((fd, size))
        raise OSError(number, "token=secret", "/private/credential")

    with monkeypatch.context() as patch:
        patch.setattr(artifact_store.os, "read", fault)
        patch.setattr(journal.store, "publish_bytes", lambda *a, **kw: pytest.fail("read failure published"))
        with _problem("RESOURCE_UNAVAILABLE", "search journal artifact could not be verified") as caught:
            if replay:
                records.append_record(journal.conn, journal.store, RUN, SLOT, body)
            else:
                records.read_record(journal.conn, journal.store, RUN, SLOT)
    _assert_redacted_artifact_error(caught.value)
    assert len(calls) == 1
    assert _journal_state(journal) == before
    assert records.read_record(journal.conn, journal.store, RUN, SLOT) == body
    assert not journal.conn.in_transaction


@pytest.mark.parametrize("operation", ["read", "append"])
def test_catalog_errors_still_propagate_without_journal_mutation(journal, operation):
    records.append_record(journal.conn, journal.store, RUN, "seal", {"existing": True})
    before = _journal_state(journal)
    denied = sqlite3.SQLITE_READ if operation == "read" else sqlite3.SQLITE_INSERT

    def authorize(action, table, *args):
        return (sqlite3.SQLITE_DENY if action == denied and table == "magnitude_search_records"
                else sqlite3.SQLITE_OK)

    journal.conn.set_authorizer(authorize)
    try:
        with pytest.raises(sqlite3.DatabaseError) as caught:
            if operation == "read":
                records.read_record(journal.conn, journal.store, RUN, "seal")
            else:
                records.append_record(journal.conn, journal.store, RUN, SLOT, _wrapper()["body"])
    finally:
        journal.conn.set_authorizer(None)
    assert type(caught.value) is sqlite3.DatabaseError
    assert caught.value.sqlite_errorcode == sqlite3.SQLITE_AUTH
    assert caught.value.sqlite_errorname == "SQLITE_AUTH"
    assert _journal_state(journal) == before
    assert not journal.conn.in_transaction


@pytest.mark.parametrize("replay,same_body", [(True, False), (False, True), (False, False), (True, True)])
def test_transaction_rechecks_existing_slot_after_an_empty_optimistic_read(journal, monkeypatch, replay, same_body):
    original = _wrapper()["body"]
    records.append_record(journal.conn, journal.store, RUN, SLOT, original)
    before = journal.conn.execute("SELECT * FROM magnitude_search_records ORDER BY slot").fetchall()
    read = records.read_record
    monkeypatch.setattr(records, "read_record", lambda *args: None)
    body = original if same_body else {"forecasts": [9, 8, 7]}
    if replay and same_body:
        assert records.append_record(journal.conn, journal.store, RUN, SLOT, body, replay=replay) == original
    else:
        with _problem("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed"):
            records.append_record(journal.conn, journal.store, RUN, SLOT, body, replay=replay)
    assert journal.conn.execute("SELECT * FROM magnitude_search_records ORDER BY slot").fetchall() == before
    assert read(journal.conn, journal.store, RUN, SLOT) == original
    assert not journal.conn.in_transaction
