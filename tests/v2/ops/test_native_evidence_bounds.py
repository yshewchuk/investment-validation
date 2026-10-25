"""Opt-in native evidence bounds preserve storage errors and exact metadata."""
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.foundation import artifact_reference, canonical_json, content_hash, to_document
from engine.v2.ops import native_registration as native
from engine.v2.ops.errors import OpsError
from tests.v2.ops.native_bounds_support import BoundedSqlRows

CAP = 16 * 1024 * 1024


@contextmanager
def evidence_catalog(*, encoding="UTF-8"):
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA encoding='{encoding}'")
    conn.executescript("""
        CREATE TABLE experiment_runs (run_id TEXT, spec_hash TEXT, mode TEXT,
            input_hash TEXT, evidence_json TEXT);
        CREATE TABLE hypotheses (run_id TEXT, spec_hash TEXT, input_hash TEXT,
            payload_hash TEXT, payload_json TEXT);
        INSERT INTO experiment_runs VALUES ('native-test', 'spec', 'primary', 'input', '{}');
        INSERT INTO hypotheses VALUES ('native-test', 'spec', 'input', 'payload', '{}');
    """)
    try:
        yield conn
    finally:
        conn.close()


def sized_object(size, character):
    available = size - len('{"padding":""}'.encode())
    repeats, extra = divmod(available, len(character.encode()))
    document = {"padding": character * repeats + "x" * extra}
    payload = canonical_json(document)
    assert len(payload.encode()) == size
    return document, payload


def forbid_decode(*args, **kwargs):
    pytest.fail("oversized evidence crossed its pre-decode bound")


def registered_binding(conn, *, padding=0):
    document = {"schema_version": "native_experiment_registration.v1.0", "spec_hash": "spec"}
    if padding:
        document["padding"] = "x" * padding
    binding = canonical_json(document).encode()
    ref = artifact_reference(binding, document["schema_version"])
    variant = content_hash(document)
    evidence = {"native_registration": to_document(ref), "variant_id": variant, "planned_variants": 1}
    conn.execute("UPDATE experiment_runs SET input_hash=?, evidence_json=?", (variant, canonical_json(evidence)))
    conn.execute("UPDATE hypotheses SET input_hash=?, payload_hash=?",
                 (variant, content_hash({"spec": "spec", "input": variant})))
    return binding, ref, evidence


def registration_store(binding, expected_ref):
    def read_verified(ref):
        assert ref == expected_ref
        return binding
    return SimpleNamespace(read_verified=read_verified)


def test_native_evidence_cap_is_sixteen_mib():
    assert native.MAX_NATIVE_EVIDENCE_BYTES == CAP


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
@pytest.mark.parametrize("storage", ["text", "blob"])
@pytest.mark.parametrize("reader", ["fetchone", "fetchmany", "fetchall", "iterate"])
def test_sql_row_oracle_preserves_rows_and_checks_all_fetch_methods(encoding, storage, reader):
    with evidence_catalog(encoding=encoding) as conn:
        seen = []
        def record_row(cursor, values):
            row = sqlite3.Row(cursor, values)
            seen.append(row)
            return row
        conn.row_factory = record_row
        payload = "漢"
        value = payload if storage == "text" else payload.encode(encoding)
        size = len(payload.encode(encoding))
        bounded = BoundedSqlRows(conn, max_bytes=size, column="evidence_json", sql_fragment="SELECT ?")
        assert bounded.row_factory is record_row and bounded.in_transaction is conn.in_transaction
        cursor = bounded.execute("SELECT ? AS evidence_json", (value,))
        assert cursor.description[0][0] == "evidence_json"
        rows = (list(cursor) if reader == "iterate" else [cursor.fetchone()] if reader == "fetchone"
                else getattr(cursor, reader)())
        assert rows == seen[-1:] and rows[0] is seen[-1]
        assert rows[0]["evidence_json"] == value and tuple(rows[0]) == (value,)
        assert bounded.sizes == [size]
        assert cursor.fetchone() is None and bounded.sizes == [size]
        oversized = bounded.execute("SELECT ? AS evidence_json", (value + value,))
        with pytest.raises(AssertionError, match="SQL row-transfer bound"):
            if reader == "iterate":
                list(oversized)
            else:
                getattr(oversized, reader)()


@pytest.mark.parametrize("stage", ["execute", "fetchone", "fetchmany", "fetchall", "iterate"])
def test_sql_row_oracle_preserves_original_sql_errors(stage):
    original = sqlite3.OperationalError("synthetic SQL interruption")
    def interrupted(*args, **kwargs):
        raise original
    with evidence_catalog() as conn:
        bounded = BoundedSqlRows(conn, max_bytes=CAP, column="evidence_json", sql_fragment="SELECT")
        if stage == "execute":
            bounded.conn = SimpleNamespace(execute=interrupted)
            def action():
                return bounded.execute("SELECT evidence_json FROM experiment_runs")
        else:
            cursor = bounded.execute("SELECT evidence_json FROM experiment_runs")
            class InterruptedCursor:
                fetchone = fetchmany = fetchall = __next__ = interrupted
            cursor.cursor = InterruptedCursor()
            action = (lambda: next(cursor)) if stage == "iterate" else getattr(cursor, stage)
        with pytest.raises(sqlite3.OperationalError) as error:
            action()
        assert error.value is original


def test_native_evidence_reads_metadata_and_object_in_one_sql_snapshot():
    with evidence_catalog() as conn:
        statements = []
        conn.set_trace_callback(statements.append)
        row, evidence = native.read_native_evidence(conn, "native-test")
        assert evidence == {}
        assert {name: row[name] for name in ("run_id", "spec_hash", "mode", "input_hash",
                "registered_input", "registered_payload", "registered_document")} == {
            "run_id": "native-test", "spec_hash": "spec", "mode": "primary", "input_hash": "input",
            "registered_input": "input", "registered_payload": "payload", "registered_document": "{}"}
        selects = [sql for sql in statements if sql.startswith("SELECT")]
        assert len(selects) == 1 and "LEFT JOIN hypotheses" in selects[0] and "e.*" not in selects[0]
        assert native.read_native_evidence(conn, "absent") is None


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
@pytest.mark.parametrize("character", ["x", "漢"])
@pytest.mark.parametrize("extra", [0, 1])
def test_native_evidence_exact_and_over_utf8_bound(encoding, character, extra, monkeypatch):
    document, payload = sized_object(CAP + extra, character)
    with evidence_catalog(encoding=encoding) as conn:
        conn.execute("UPDATE experiment_runs SET evidence_json=?", (payload,))
        limit = CAP * (2 if encoding.startswith("UTF-16") else 1)
        bounded = BoundedSqlRows(conn, max_bytes=limit, column="evidence_json",
                                 sql_fragment="FROM experiment_runs")
        if extra:
            monkeypatch.setattr(native.json, "loads", forbid_decode)
            with pytest.raises(OpsError) as error:
                native.read_native_evidence(bounded, "native-test")
            assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and not error.value.problem.retryable
        else:
            row, decoded = native.read_native_evidence(bounded, "native-test")
            assert decoded == document
            assert row["evidence_json"].decode(encoding) == payload
        sql_size = len(payload.encode(encoding))
        assert bounded.sizes == [None if sql_size > limit else sql_size]


@pytest.mark.parametrize("payload", ["{", "[]", "null", "true", "1", '"text"', "[" * 2000 + "0" + "]" * 2000],
                         ids=["malformed", "list", "null", "bool", "number", "string", "recursive"])
def test_native_evidence_refuses_malformed_or_nonobject_json(payload):
    with evidence_catalog() as conn:
        conn.execute("UPDATE experiment_runs SET evidence_json=?", (payload,))
        with pytest.raises(OpsError) as error:
            native.read_native_evidence(conn, "native-test")
        assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and not error.value.problem.retryable


@pytest.mark.parametrize("payload", [None, b"{}"])
def test_native_evidence_refuses_null_or_nontext_storage_before_json(payload, monkeypatch):
    with evidence_catalog() as conn:
        conn.execute("UPDATE experiment_runs SET evidence_json=?", (payload,))
        monkeypatch.setattr(native.json, "loads", forbid_decode)
        with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
            native.read_native_evidence(conn, "native-test")


@pytest.mark.parametrize("oversized", [False, True])
def test_native_evidence_sql_projection_handles_invalid_utf8_without_text_transfer(oversized, monkeypatch):
    with evidence_catalog() as conn:
        payload = b"\xff" * (CAP + 1 if oversized else 1)
        conn.execute("UPDATE experiment_runs SET evidence_json=CAST(? AS TEXT)", (payload,))
        # A raw TEXT projection would fail inside sqlite's UTF-8 conversion.
        with pytest.raises(sqlite3.OperationalError, match="Could not decode to UTF-8"):
            conn.execute("SELECT evidence_json FROM experiment_runs").fetchone()
        bounded = BoundedSqlRows(conn, max_bytes=CAP, column="evidence_json",
                                 sql_fragment="FROM experiment_runs")
        monkeypatch.setattr(native.json, "loads", forbid_decode)
        with pytest.raises(OpsError) as error:
            native.read_native_evidence(bounded, "native-test")
        assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and not error.value.problem.retryable
        assert bounded.sizes == [None if oversized else 1]


@pytest.mark.parametrize("exception", [OSError, sqlite3.OperationalError, sqlite3.IntegrityError])
def test_native_evidence_preserves_storage_exception_identity(exception):
    original = exception("storage interruption")
    def execute(*args):
        raise original
    with pytest.raises(exception) as error:
        native.read_native_evidence(SimpleNamespace(execute=execute), "native-test")
    assert error.value is original


@pytest.mark.parametrize("character", ["x", "漢"])
@pytest.mark.parametrize("extra", [0, 1])
def test_native_evidence_writer_uses_the_same_utf8_bound(character, extra):
    document, payload = sized_object(CAP + extra, character)
    if extra:
        with pytest.raises(OpsError) as error:
            native.native_evidence_json(document)
        assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and not error.value.problem.retryable
    else:
        assert native.native_evidence_json(document) == payload


@pytest.mark.parametrize("damage", ["nonobject", "recursive", "invalid-unicode"])
def test_native_evidence_writer_refuses_invalid_object_encoding(damage):
    document = [] if damage == "nonobject" else {"text": "\ud800"} if damage == "invalid-unicode" else {}
    if damage == "recursive":
        document["cycle"] = document
    with pytest.raises(OpsError) as error:
        native.native_evidence_json(document)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and not error.value.problem.retryable


def test_native_evidence_writer_is_canonical():
    assert native.native_evidence_json({"z": 1.0, "a": "漢"}) == '{"a":"漢","z":1}'
    assert json.loads(native.native_evidence_json({})) == {}


def test_native_evidence_reader_and_writer_accept_explicit_bound():
    with evidence_catalog() as conn:
        assert native.read_native_evidence(conn, "native-test", max_bytes=2)[1] == {}
        assert native.native_evidence_json({}, max_bytes=2) == "{}"
        with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
            native.read_native_evidence(conn, "native-test", max_bytes=1)
        with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
            native.native_evidence_json({}, max_bytes=1)


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
def test_registration_opt_in_verifier_uses_same_snapshot_and_expected_ref(encoding):
    with evidence_catalog(encoding=encoding) as conn:
        binding, ref, _ = registered_binding(conn)
        store = registration_store(binding, ref)
        assert native.verified_registration_bytes(conn, store, "native-test", expected_ref=ref,
                                                  max_evidence_bytes=CAP) == binding
        assert native.verified_registration_bytes(conn, store, "absent", max_evidence_bytes=CAP) is None
        with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
            native.verified_registration_bytes(conn, store, "native-test", expected_ref=replace(ref, byte_size=1),
                                               max_evidence_bytes=CAP)


def test_registration_evidence_bound_is_opt_in_and_precedes_json_and_store_read(monkeypatch):
    with evidence_catalog() as conn:
        binding, ref, evidence = registered_binding(conn)
        evidence["padding"] = "x" * CAP
        payload = canonical_json(evidence)
        assert len(payload.encode()) > CAP
        conn.execute("UPDATE experiment_runs SET evidence_json=?", (payload,))
        assert native.verified_registration_bytes(conn, registration_store(binding, ref), "native-test") == binding
        bounded = BoundedSqlRows(conn, max_bytes=CAP, column="evidence_json",
                                 sql_fragment="FROM experiment_runs")
        monkeypatch.setattr(native.json, "loads", forbid_decode)
        with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
            native.verified_registration_bytes(bounded, SimpleNamespace(read_verified=forbid_decode), "native-test",
                                               max_evidence_bytes=CAP)
        assert bounded.sizes == [None]


def test_registration_artifact_bound_is_opt_in_and_precedes_store_read():
    with evidence_catalog() as conn:
        binding, ref, _ = registered_binding(conn, padding=CAP)
        assert ref.byte_size > CAP
        assert native.verified_registration_bytes(conn, registration_store(binding, ref), "native-test") == binding
        with pytest.raises(OpsError) as error:
            native.verified_registration_bytes(conn, SimpleNamespace(read_verified=forbid_decode), "native-test",
                                               max_evidence_bytes=CAP)
        assert error.value.code == "RESOURCE_LIMIT_EXCEEDED" and not error.value.problem.retryable


@pytest.mark.parametrize("extra", [0, 1])
def test_registration_opt_in_artifact_metadata_honors_exact_custom_bound(extra):
    with evidence_catalog() as conn:
        binding, ref, evidence = registered_binding(conn, padding=1024)
        limit = len(binding)
        if extra:
            evidence["native_registration"] = to_document(replace(ref, byte_size=limit + extra))
            conn.execute("UPDATE experiment_runs SET evidence_json=?", (canonical_json(evidence),))
            with pytest.raises(OpsError, match="RESOURCE_LIMIT_EXCEEDED"):
                native.verified_registration_bytes(conn, SimpleNamespace(read_verified=forbid_decode), "native-test",
                                                   max_evidence_bytes=limit)
        else:
            assert native.verified_registration_bytes(conn, registration_store(binding, ref), "native-test",
                                                      max_evidence_bytes=limit) == binding


@pytest.mark.parametrize("binding", [b"{", b"[]", b"null", b"[" * 2000 + b"0" + b"]" * 2000],
                         ids=["malformed", "list", "null", "recursive"])
def test_registration_opt_in_refuses_malformed_or_nonobject_artifact(binding):
    with evidence_catalog() as conn:
        _, original_ref, evidence = registered_binding(conn)
        ref = artifact_reference(binding, original_ref.schema_ref)
        evidence["native_registration"] = to_document(ref)
        conn.execute("UPDATE experiment_runs SET evidence_json=?", (canonical_json(evidence),))
        with pytest.raises(OpsError) as error:
            native.verified_registration_bytes(conn, registration_store(binding, ref), "native-test",
                                               max_evidence_bytes=CAP)
        assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and not error.value.problem.retryable


@pytest.mark.parametrize("location", ["evidence", "artifact"])
def test_registration_default_preserves_existing_recursion_error(location, monkeypatch):
    with evidence_catalog() as conn:
        binding, ref, _ = registered_binding(conn)
        original = RecursionError("JSON decoder recursion bound")
        loads = native.json.loads
        def limited_decode(payload):
            if (location == "artifact") == isinstance(payload, bytes):
                raise original
            return loads(payload)
        monkeypatch.setattr(native.json, "loads", limited_decode)
        with pytest.raises(RecursionError) as error:
            native.verified_registration_bytes(conn, registration_store(binding, ref), "native-test")
        assert error.value is original
        with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
            native.verified_registration_bytes(conn, registration_store(binding, ref), "native-test",
                                               max_evidence_bytes=CAP)


@pytest.mark.parametrize("max_bytes", [None, CAP])
@pytest.mark.parametrize("exception", [OSError, sqlite3.OperationalError, sqlite3.IntegrityError])
def test_registration_verifier_preserves_storage_exceptions(max_bytes, exception):
    original = exception("storage interruption")
    def interrupted(*args):
        raise original
    with pytest.raises(exception) as error:
        native.verified_registration_bytes(SimpleNamespace(execute=interrupted), None, "native-test",
                                           max_evidence_bytes=max_bytes)
    assert error.value is original
    with evidence_catalog() as conn:
        registered_binding(conn)
        with pytest.raises(exception) as error:
            native.verified_registration_bytes(conn, SimpleNamespace(read_verified=interrupted), "native-test",
                                               max_evidence_bytes=max_bytes)
        assert error.value is original


@pytest.mark.parametrize("invalid_utf8", [False, True])
def test_native_evidence_projects_only_the_expected_empty_registered_document(invalid_utf8):
    with evidence_catalog() as conn:
        binding, ref, _ = registered_binding(conn)
        payload = (b"\xff" if invalid_utf8 else b"x") * (CAP + 1)
        conn.execute("UPDATE hypotheses SET payload_json=CAST(? AS TEXT)", (payload,))
        row, _ = native.read_native_evidence(conn, "native-test")
        assert row["registered_document"] is None
        with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
            native.verified_registration_bytes(conn, registration_store(binding, ref), "native-test",
                                               max_evidence_bytes=CAP)


@pytest.mark.parametrize("payload, limit", [
    ('{"extra":' + "[" * 2000 + "0" + "]" * 2000 + "}", CAP),
    ('{"extra":NaN}', len('{"extra":NaN}')),
], ids=["deep-unrelated-field", "canonical-expansion"])
def test_native_evidence_reader_refuses_objects_that_cannot_be_rewritten(payload, limit):
    # Python 3.14's JSON decoder accepts this nesting. Canonical serialization
    # has its own recursion limit and can expand compact nonfinite literals.
    with evidence_catalog() as conn:
        conn.execute("UPDATE experiment_runs SET evidence_json=?", (payload,))
        with pytest.raises(OpsError) as error:
            native.read_native_evidence(conn, "native-test", max_bytes=limit)
        assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and not error.value.problem.retryable


def test_native_evidence_reader_preserves_noncanonical_but_rewritable_input():
    payload = ' { "extra" : 1.0 } '
    with evidence_catalog() as conn:
        conn.execute("UPDATE experiment_runs SET evidence_json=?", (payload,))
        row, evidence = native.read_native_evidence(conn, "native-test", max_bytes=len(payload))
        assert row["evidence_json"] == payload.encode()
        assert evidence == {"extra": 1.0}
        assert native.native_evidence_json(evidence, max_bytes=len(payload)) == '{"extra":1}'


def test_registration_rewritable_evidence_contract_is_opt_in():
    with evidence_catalog() as conn:
        binding, ref, evidence = registered_binding(conn)
        payload = canonical_json(evidence)[:-1] + ',"ignored":' + "[" * 2000 + "0" + "]" * 2000 + "}"
        conn.execute("UPDATE experiment_runs SET evidence_json=?", (payload,))
        assert native.verified_registration_bytes(conn, registration_store(binding, ref), "native-test") == binding
        with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
            native.verified_registration_bytes(conn, SimpleNamespace(read_verified=forbid_decode), "native-test",
                                               max_evidence_bytes=CAP)
