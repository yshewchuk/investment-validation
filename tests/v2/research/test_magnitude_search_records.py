"""Isolated journal limits and typed refusals over real SQLite and artifact storage."""
import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.contracts import FAILURE_CODES, ArtifactRef
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    SystemClock,
    artifact_reference,
    canonical_json,
    from_document,
    to_document,
)
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError
from engine.v2.ops.migrations import migrate
from experiments import magnitude_search_records as records

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
        yield
    assert to_document(caught.value.problem) == {
        "code": code,
        "category": FAILURE_CODES[code][0],
        "retryable": FAILURE_CODES[code][1],
        "message": message,
        "stage": None,
        "trace_id": None,
        "dependency_refs": [],
        "retry_after_seconds": None,
        "diagnostic_ref": None,
        "details": {},
        "schema_version": "problem.v1.0",
    }


def test_journal_default_limit_is_sixteen_mebibytes():
    assert records.MAX_BYTES == 16 * 1024 * 1024


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


@pytest.mark.parametrize("failure, code", [
    ("missing", "RESOURCE_UNAVAILABLE"),
    ("wrong_size", "EXPERIMENT_IDENTITY_CONFLICT"),
    ("corrupt_bytes", "EXPERIMENT_IDENTITY_CONFLICT"),
    ("unsafe_path", "EXPERIMENT_IDENTITY_CONFLICT"),
])
def test_artifact_read_failures_preserve_classification_and_redact_paths(journal, failure, code):
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
    _insert(journal, canonical_json(to_document(ref)))
    with _problem(code, "search journal artifact could not be verified or published"):
        records.read_record(journal.conn, journal.store, RUN, SLOT)


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
    with _problem(code, "search journal artifact could not be verified or published"):
        records.append_record(journal.conn, store, RUN, SLOT, _wrapper()["body"])
    assert journal.conn.execute("SELECT COUNT(*) FROM magnitude_search_records").fetchone()[0] == 0
    assert records.read_record(journal.conn, journal.store, RUN, SLOT) is None
    assert not journal.conn.in_transaction
