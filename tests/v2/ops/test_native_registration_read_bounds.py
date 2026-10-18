"""The public registration loader admits bytes before catalog/object materialization."""
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.foundation import ArtifactError, ArtifactStore, canonical_json, content_hash, to_document
from engine.v2.ops.errors import OpsError, fail
from experiments import native_registration as native
from tests.v2.ops.native_bounds_support import BoundedSqlRows, catalog_rows, object_bytes
from tests.v2.ops.test_native_evidence_bounds import evidence_catalog

CAP = 16 * 1024 * 1024
SPEC = SimpleNamespace(experiment_id="bounded-loader", primary_arm_id="candidate", spec_hash="spec")
RUN_ID = "native_" + content_hash({"experiment_id": SPEC.experiment_id,
                                  "arm": SPEC.primary_arm_id}).removeprefix("sha256:")


def forbid(*args, **kwargs):
    pytest.fail("registration load crossed its protected read/effect boundary")


@contextmanager
def refuses(message, code="EXPERIMENT_IDENTITY_CONFLICT"):
    with pytest.raises(OpsError) as captured:
        yield
    assert type(captured.value) is OpsError
    assert captured.value.code == code
    assert captured.value.problem.retryable is False
    assert captured.value.problem.message == message


@contextmanager
def registered_catalog(tmp_path, *, encoding="UTF-8", binding_size=None):
    with evidence_catalog(encoding=encoding) as conn:
        document = {"schema_version": "native_experiment_registration.v1.0", "spec_hash": SPEC.spec_hash}
        if binding_size is not None:
            document = pad(document, binding_size, "漢")
        binding = canonical_json(document).encode()
        store = ArtifactStore(tmp_path / "store")
        ref = store.publish_bytes(binding, schema_ref=document["schema_version"])
        variant = content_hash(document)
        evidence = {"native_registration": to_document(ref), "variant_id": variant, "planned_variants": 1}
        conn.execute("UPDATE experiment_runs SET run_id=?, input_hash=?, evidence_json=?",
                     (RUN_ID, variant, canonical_json(evidence)))
        conn.execute("UPDATE hypotheses SET run_id=?, input_hash=?, payload_hash=?",
                     (RUN_ID, variant, content_hash({"spec": SPEC.spec_hash, "input": variant})))
        yield conn, store, native.NativeRegistration(RUN_ID, binding), ref, evidence


def pad(document, size, character):
    document = {**document, "padding": ""}
    available = size - len(canonical_json(document).encode())
    count, extra = divmod(available, len(character.encode()))
    assert count >= 0
    document["padding"] = character * count + "x" * extra
    assert len(canonical_json(document).encode()) == size
    return document


def save_evidence(conn, evidence):
    conn.execute("UPDATE experiment_runs SET evidence_json=?", (canonical_json(evidence),))


@pytest.mark.parametrize("value, admitted", [
    (1, True), (CAP, True), (2**62 - 1, True), (0, False), (-1, False),
    (2**62, False), (None, False), (True, False), (False, False), (1.0, False), ("1", False),
])
def test_public_budget_predicate_returns_exact_bool_without_coercion(value, admitted):
    assert native.valid_registration_read_budget(value) is admitted


def test_public_budget_predicate_rejects_integer_subclasses():
    class IntegerSubclass(int):
        pass
    assert native.valid_registration_read_budget(IntegerSubclass(1)) is False


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
@pytest.mark.parametrize("character", ["x", "漢"])
@pytest.mark.parametrize("extra", [0, 1])
def test_default_loader_caps_sql_transfer_and_utf8_before_json(tmp_path, monkeypatch, encoding, character, extra):
    with registered_catalog(tmp_path, encoding=encoding) as (conn, store, registration, _, evidence):
        payload = canonical_json(pad(evidence, CAP + extra, character))
        conn.execute("UPDATE experiment_runs SET evidence_json=?", (payload,))
        before = catalog_rows(conn)
        sql_limit = CAP * (2 if encoding.startswith("UTF-16") else 1)
        bounded = BoundedSqlRows(conn, max_bytes=sql_limit, column="evidence_json",
                                 sql_fragment="FROM experiment_runs")
        conn.execute("PRAGMA query_only=ON")
        monkeypatch.setattr(store, "publish_bytes", forbid)
        if extra:
            with monkeypatch.context() as patch:
                patch.setattr(native.json, "loads", forbid)
                patch.setattr(store, "read_verified", forbid)
                with refuses("native evidence requires bounded JSON object text"):
                    native.read_native_registration(bounded, store, SPEC)
        else:
            assert native.read_native_registration(bounded, store, SPEC) == registration
        sql_size = len(payload.encode(encoding))
        assert bounded.sizes == [None if sql_size > sql_limit else sql_size]
        assert catalog_rows(conn) == before


def test_loader_uses_one_catalog_snapshot_and_one_verified_object_read(tmp_path, monkeypatch):
    with registered_catalog(tmp_path) as (conn, store, registration, ref, _):
        statements, reads = [], []
        original = store.read_verified

        def read_once(reference):
            reads.append(reference)
            assert reads == [ref]
            return original(reference)

        before, objects = catalog_rows(conn), object_bytes(store)
        monkeypatch.setattr(store, "read_verified", read_once)
        conn.execute("PRAGMA query_only=ON")
        conn.set_trace_callback(statements.append)
        assert native.read_native_registration(conn, store, SPEC) == registration
        selects = [sql for sql in statements if sql.startswith("SELECT")]
        assert len(selects) == 1 and "LEFT JOIN hypotheses" in selects[0]
        assert reads == [ref]
        assert catalog_rows(conn) == before and object_bytes(store) == objects


@pytest.mark.parametrize("extra", [0, 1])
def test_default_loader_admits_reference_size_before_artifact_read(tmp_path, monkeypatch, extra):
    with registered_catalog(tmp_path, binding_size=CAP + extra) as (conn, store, registration, ref, _):
        assert ref.byte_size == CAP + extra
        before, objects = catalog_rows(conn), object_bytes(store)
        conn.execute("PRAGMA query_only=ON")
        monkeypatch.setattr(store, "publish_bytes", forbid)
        if extra:
            monkeypatch.setattr(store, "read_verified", forbid)
            with refuses("native registration differs from its immutable binding"):
                native.read_native_registration(conn, store, SPEC)
        else:
            assert native.read_native_registration(conn, store, SPEC) == registration
        assert catalog_rows(conn) == before and object_bytes(store) == objects


@pytest.mark.parametrize("location", ["catalog", "object"])
def test_explicit_larger_finite_budget_applies_to_both_read_boundaries(tmp_path, location):
    size = CAP + 1
    with registered_catalog(tmp_path, binding_size=size if location == "object" else None) as values:
        conn, store, registration, _, evidence = values
        if location == "catalog":
            save_evidence(conn, pad(evidence, size, "漢"))
        message = ("native evidence requires bounded JSON object text" if location == "catalog"
                   else "native registration differs from its immutable binding")
        with refuses(message):
            native.read_native_registration(conn, store, SPEC)
        assert native.read_native_registration(conn, store, SPEC, max_evidence_bytes=size) == registration
        with refuses(message):
            native.read_native_registration(conn, store, SPEC, max_evidence_bytes=size - 1)


@pytest.mark.parametrize("budget", [None, True, False, 0, -1, 1.0, float("inf"), float("nan"), "1024", 2**62, 2**100])
def test_invalid_budget_refuses_before_any_catalog_or_object_access(budget):
    with pytest.raises(OpsError) as captured:
        native.read_native_registration(SimpleNamespace(execute=forbid), SimpleNamespace(read_verified=forbid),
                                        None, max_evidence_bytes=budget)
    assert type(captured.value) is OpsError
    assert captured.value.code == "INVALID_EXPERIMENT_SPEC" and captured.value.problem.retryable is False
    assert captured.value.problem.message == "registration read requires a positive SQLite-safe byte budget"


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
def test_smallest_budget_reaches_sql_admission_instead_of_configuration_refusal(tmp_path, monkeypatch, encoding):
    with registered_catalog(tmp_path, encoding=encoding) as (conn, store, _, _, _):
        bounded = BoundedSqlRows(conn, max_bytes=2 if encoding.startswith("UTF-16") else 1,
                                 column="evidence_json", sql_fragment="FROM experiment_runs")
        monkeypatch.setattr(store, "read_verified", forbid)
        with refuses("native evidence requires bounded JSON object text"):
            native.read_native_registration(bounded, store, SPEC, max_evidence_bytes=1)
        assert bounded.sizes == [None]


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
def test_largest_safe_finite_budget_can_be_bound_in_sql(tmp_path, encoding):
    with registered_catalog(tmp_path, encoding=encoding) as (conn, store, registration, _, _):
        assert native.read_native_registration(conn, store, SPEC, max_evidence_bytes=2**62 - 1) == registration


@pytest.mark.parametrize("payload", [None, b"{}", "{", "[]", "null", "true", "1", '"text"',
                                     "[" * 2000 + "0" + "]" * 2000])
def test_invalid_catalog_evidence_is_typed_before_object_access(tmp_path, monkeypatch, payload):
    with registered_catalog(tmp_path) as (conn, store, _, _, _):
        conn.execute("UPDATE experiment_runs SET evidence_json=?", (payload,))
        before = catalog_rows(conn)
        monkeypatch.setattr(store, "read_verified", forbid)
        with refuses("native evidence requires bounded JSON object text"):
            native.read_native_registration(conn, store, SPEC)
        assert catalog_rows(conn) == before


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_canonical_catalog_expansion_refuses_before_object_access(tmp_path, monkeypatch, value):
    with registered_catalog(tmp_path) as (conn, store, _, _, evidence):
        payload = canonical_json(evidence)[:-1] + ',"padding":' + value + '}'
        conn.execute("UPDATE experiment_runs SET evidence_json=?", (payload,))
        before = catalog_rows(conn)
        limit = len(payload.encode())
        bounded = BoundedSqlRows(conn, max_bytes=limit, column="evidence_json", sql_fragment="FROM experiment_runs")
        monkeypatch.setattr(store, "read_verified", forbid)
        with refuses("native evidence exceeds its canonical object contract"):
            native.read_native_registration(bounded, store, SPEC, max_evidence_bytes=limit)
        assert bounded.sizes == [limit]
        assert catalog_rows(conn) == before


def test_invalid_utf8_is_transferred_as_bounded_bytes_and_refuses_before_decode(tmp_path, monkeypatch):
    with registered_catalog(tmp_path) as (conn, store, _, _, _):
        conn.execute("UPDATE experiment_runs SET evidence_json=CAST(? AS TEXT)", (b"\xff",))
        bounded = BoundedSqlRows(conn, max_bytes=CAP, column="evidence_json", sql_fragment="FROM experiment_runs")
        monkeypatch.setattr(native.json, "loads", forbid)
        monkeypatch.setattr(store, "read_verified", forbid)
        with refuses("native evidence requires bounded JSON object text"):
            native.read_native_registration(bounded, store, SPEC)
        assert bounded.sizes == [1]


@pytest.mark.parametrize("binding", [b"{", b"null", b"[]", b"[" * 2000 + b"0" + b"]" * 2000])
def test_malformed_registration_objects_refuse_without_rewrites(tmp_path, monkeypatch, binding):
    with registered_catalog(tmp_path) as (conn, store, _, ref, evidence):
        evidence["native_registration"] = to_document(store.publish_bytes(binding, schema_ref=ref.schema_ref))
        save_evidence(conn, evidence)
        before, objects = catalog_rows(conn), object_bytes(store)
        monkeypatch.setattr(store, "publish_bytes", forbid)
        with refuses("native registration differs from its immutable binding"):
            native.read_native_registration(conn, store, SPEC)
        assert catalog_rows(conn) == before and object_bytes(store) == objects


@pytest.mark.parametrize("damage", ["missing-reference", "bad-reference", "schema", "variant", "count",
                                   "mode", "input", "hypothesis", "payload", "registered-document", "spec"])
def test_loader_preserves_identity_checks_and_exact_refusal(tmp_path, monkeypatch, damage):
    with registered_catalog(tmp_path) as (conn, store, _, ref, evidence):
        spec = SPEC
        if damage == "missing-reference":
            evidence.pop("native_registration")
        elif damage == "bad-reference":
            evidence["native_registration"] = {}
        elif damage == "schema":
            evidence["native_registration"] = to_document(replace(ref, schema_ref="unsupported.v1"))
        elif damage == "variant":
            evidence["variant_id"] = "wrong"
        elif damage == "count":
            evidence["planned_variants"] = True
        elif damage == "mode":
            conn.execute("UPDATE experiment_runs SET mode='secondary'")
        elif damage == "input":
            conn.execute("UPDATE experiment_runs SET input_hash='changed'")
        elif damage == "hypothesis":
            conn.execute("DELETE FROM hypotheses")
        elif damage == "payload":
            conn.execute("UPDATE hypotheses SET payload_hash='changed'")
        elif damage == "registered-document":
            conn.execute("UPDATE hypotheses SET payload_json='[]'")
        else:
            spec = SimpleNamespace(**(vars(SPEC) | {"spec_hash": "changed"}))
        save_evidence(conn, evidence)
        before, objects = catalog_rows(conn), object_bytes(store)
        monkeypatch.setattr(store, "publish_bytes", forbid)
        if damage in ("missing-reference", "bad-reference", "schema"):
            monkeypatch.setattr(store, "read_verified", forbid)
        message = ("requested specification differs from registration" if damage == "spec"
                   else "native registration differs from its immutable binding")
        with refuses(message):
            native.read_native_registration(conn, store, spec)
        assert catalog_rows(conn) == before and object_bytes(store) == objects


def test_absent_reservation_stays_r1_and_never_opens_objects(tmp_path, monkeypatch):
    with registered_catalog(tmp_path) as (conn, store, _, _, _):
        conn.execute("DELETE FROM experiment_runs")
        before = catalog_rows(conn)
        monkeypatch.setattr(store, "read_verified", forbid)
        with refuses("native experiment has not been preregistered", "INVALID_EXPERIMENT_SPEC"):
            native.read_native_registration(conn, store, SPEC)
        assert catalog_rows(conn) == before


@pytest.mark.parametrize("stage", ["catalog", "object"])
@pytest.mark.parametrize("exception", [OSError, sqlite3.OperationalError, sqlite3.IntegrityError])
def test_storage_exceptions_propagate_unchanged(tmp_path, monkeypatch, stage, exception):
    original = exception("synthetic storage interruption")

    def interrupted(*args, **kwargs):
        raise original

    with registered_catalog(tmp_path) as (conn, store, _, _, _):
        if stage == "catalog":
            conn = SimpleNamespace(execute=interrupted)
        else:
            monkeypatch.setattr(store, "read_verified", interrupted)
        with pytest.raises(exception) as captured:
            native.read_native_registration(conn, store, SPEC)
        assert captured.value is original


def test_foundation_artifact_failure_remains_identity_conflict(tmp_path, monkeypatch):
    with registered_catalog(tmp_path) as (conn, store, _, _, _):
        def corrupt(*args):
            raise ArtifactError("INTEGRITY_FAILED", "synthetic corrupt object")
        monkeypatch.setattr(store, "read_verified", corrupt)
        with refuses("native registration differs from its immutable binding"):
            native.read_native_registration(conn, store, SPEC)


@pytest.mark.parametrize("damage", ["code", "retryability", "empty-message", "constant-message", "subclass"])
def test_refusal_oracle_rejects_changed_type_envelope_or_diagnostic(damage):
    class DifferentOpsError(OpsError):
        pass
    message = "native registration differs from its immutable binding"
    error = fail("INVALID_EXPERIMENT_SPEC" if damage == "code" else "EXPERIMENT_IDENTITY_CONFLICT",
                 message)
    if damage == "retryability":
        error = OpsError(replace(error.problem, retryable=True))
    elif damage in ("empty-message", "constant-message"):
        error = OpsError(replace(error.problem, message="" if damage == "empty-message" else "refused"))
    elif damage == "subclass":
        error = DifferentOpsError(error.problem)
    with pytest.raises(AssertionError):
        with refuses(message):
            raise error
