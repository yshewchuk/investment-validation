"""Opt-in collection bounds reach real native publication and replay boundaries."""
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    DocumentError,
    artifact_reference,
    canonical_json,
    from_document,
    to_document,
)
from engine.v2.ops import checkpoints
from engine.v2.ops.errors import OpsError, make_problem
from experiments import native_outcomes as outcomes
from tests.v2.ops.native_bounds_support import (
    BoundedSqlRows,
    catalog_rows,
    native_evidence,
    object_bytes,
    register_native_case,
    registration_ref,
)
from tests.v2.ops.test_native_registration_read_bounds import size_refusal
from tests.v2.research.test_native_registration import source as source

CAP = 8192
REPORT = "# Synthetic result\nObserved result: 0.25.\n"
ACTIONS = ("publish_native_outcome", "replay_native_outcome", "export_native_report")


def invoke(action, source, registration, ledger, **kwargs):
    options = {"report": REPORT, "attempted": True} if action == ACTIONS[0] else {}
    return getattr(outcomes, action)(source.conn, source.store, registration,
                                    ledger_path=ledger, **(options | kwargs))


def save_evidence(source, registration, evidence):
    source.conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                        (canonical_json(evidence), registration.run_id))


def pad_evidence(evidence, size, character="x"):
    document = {**evidence, "padding": ""}
    available = size - len(canonical_json(document).encode())
    repeats, extra = divmod(available, len(character.encode()))
    assert repeats >= 0
    document["padding"] = character * repeats + "x" * extra
    assert len(canonical_json(document).encode()) == size
    return document


def forbid(*args, **kwargs):
    pytest.fail("bounded refusal crossed its protected read/effect boundary")


def assert_conflict(call):
    with pytest.raises(OpsError) as error:
        call()
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert error.value.problem.retryable is False


@pytest.mark.parametrize("no_ledger", [False, True])
def test_bounded_publication_replay_and_export_preserve_exact_results(source, tmp_path, no_ledger):
    registration = register_native_case(source)
    ledger = tmp_path / "ledger.csv"
    options = {"no_ledger": no_ledger, "max_evidence_bytes": CAP}
    # Reports are not JSON evidence: the opt-in metadata limit must not cap them.
    report = REPORT + "漢" * CAP
    first = invoke(ACTIONS[0], source, registration, ledger, report=report, **options)
    assert type(first["outcome"]["attempted_variants"]) is int
    assert first["outcome"]["attempted_variants"] == 1
    assert invoke(ACTIONS[1], source, registration, ledger, **options) == first
    assert invoke(ACTIONS[0], source, registration, ledger, report=report, **options) == first
    path = invoke(ACTIONS[2], source, registration, ledger, **options)
    assert path.read_text() == report
    assert ledger.exists() is not no_ledger
    key = "native_smoke" if no_ledger else "native_outcome"
    evidence = native_evidence(source, registration.run_id)
    for name in (key, key + "_receipt"):
        ref = from_document(ArtifactRef, evidence[name])
        assert ref.byte_size <= CAP
        assert source.store.read_verified(ref) == canonical_json(
            first["outcome" if name == key else "receipt"]).encode()


@pytest.mark.parametrize("options", [{}, {"max_evidence_bytes": None}], ids=["omitted", "explicit-none"])
def test_bounds_are_opt_in_for_all_public_outcome_apis(source, tmp_path, monkeypatch, options):
    registration = register_native_case(source)
    evidence = pad_evidence(native_evidence(source, registration.run_id), CAP + 1, "漢")
    save_evidence(source, registration, evidence)
    ledger = tmp_path / "ledger.csv"
    first = invoke(ACTIONS[0], source, registration, ledger, **options)
    assert invoke(ACTIONS[1], source, registration, ledger, **options) == first
    assert invoke(ACTIONS[2], source, registration, ledger, **options).read_text() == REPORT
    before = ledger.read_bytes()
    monkeypatch.setattr(source.store, "read_verified", forbid)
    monkeypatch.setattr(outcomes, "ledger_append", forbid)
    for action in ACTIONS:
        assert_conflict(lambda: invoke(action, source, registration, ledger, max_evidence_bytes=CAP))
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("character", ["x", "漢"])
def test_exact_catalog_evidence_bound_replays_and_exports(source, tmp_path, character):
    registration = register_native_case(source)
    ledger = tmp_path / "ledger.csv"
    first = invoke(ACTIONS[0], source, registration, ledger)
    evidence = pad_evidence(native_evidence(source, registration.run_id), CAP, character)
    save_evidence(source, registration, evidence)
    assert invoke(ACTIONS[1], source, registration, ledger, max_evidence_bytes=CAP) == first
    assert invoke(ACTIONS[2], source, registration, ledger, max_evidence_bytes=CAP).read_text() == REPORT
    assert native_evidence(source, registration.run_id) == evidence


@pytest.mark.parametrize("action", ACTIONS)
def test_registration_size_is_checked_before_verified_read(source, tmp_path, monkeypatch, action):
    registration = register_native_case(source)
    cap = registration_ref(source, registration).byte_size - 1
    assert len(canonical_json(native_evidence(source, registration.run_id)).encode()) <= cap
    before_rows, before_objects = catalog_rows(source.conn), object_bytes(source.store)
    monkeypatch.setattr(source.store, "read_verified", forbid)
    monkeypatch.setattr(outcomes, "ledger_append", forbid)
    with size_refusal(cap + 1, cap):
        invoke(action, source, registration, tmp_path / "ledger.csv", max_evidence_bytes=cap)
    assert catalog_rows(source.conn) == before_rows and object_bytes(source.store) == before_objects
    assert not (tmp_path / "ledger.csv").exists()


def test_exact_registration_artifact_bound_allows_empty_replay(source, tmp_path):
    registration = register_native_case(source)
    assert invoke(ACTIONS[1], source, registration, tmp_path / "ledger.csv",
                  max_evidence_bytes=registration_ref(source, registration).byte_size) is None


@pytest.mark.parametrize("slot", ["native_outcome", "native_outcome_receipt"])
@pytest.mark.parametrize("action", ACTIONS)
def test_oversized_outcome_and_completion_references_are_not_read(
        source, tmp_path, monkeypatch, slot, action):
    registration = register_native_case(source)
    ledger = tmp_path / "ledger.csv"
    invoke(ACTIONS[0], source, registration, ledger)
    evidence = native_evidence(source, registration.run_id)
    schema = evidence[slot]["schema_ref"]
    oversized = source.store.publish_bytes(canonical_json({"padding": "x" * CAP}).encode(), schema_ref=schema)
    evidence[slot] = to_document(oversized)
    save_evidence(source, registration, evidence)
    before = ledger.read_bytes()
    real_read, reads = source.store.read_verified, []

    def bounded_read(ref):
        assert ref != oversized, "oversized JSON reached the artifact store"
        reads.append(ref)
        return real_read(ref)

    monkeypatch.setattr(source.store, "read_verified", bounded_read)
    monkeypatch.setattr(outcomes, "ledger_append", forbid)
    assert_conflict(lambda: invoke(action, source, registration, ledger, max_evidence_bytes=CAP))
    assert registration_ref(source, registration) in reads
    assert native_evidence(source, registration.run_id) == evidence
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("payload", ["{", "[]", "null", "[" * 2000 + "0" + "]" * 2000],
                         ids=["malformed", "list", "null", "recursive"])
@pytest.mark.parametrize("action", ACTIONS)
def test_malformed_catalog_evidence_is_typed_before_artifact_reads(
        source, tmp_path, monkeypatch, payload, action):
    registration = register_native_case(source)
    source.conn.execute("UPDATE experiment_runs SET evidence_json=?", (payload,))
    monkeypatch.setattr(source.store, "read_verified", forbid)
    monkeypatch.setattr(outcomes, "ledger_append", forbid)
    assert_conflict(lambda: invoke(action, source, registration, tmp_path / "ledger.csv",
                                  max_evidence_bytes=CAP))


@pytest.mark.parametrize("payload", [b"{", b"null", b"[" * 2000 + b"0" + b"]" * 2000],
                         ids=["malformed", "null", "recursive"])
@pytest.mark.parametrize("slot", ["native_outcome", "native_outcome_receipt"])
def test_bounded_replay_refuses_malformed_json_artifacts(source, tmp_path, monkeypatch, payload, slot):
    registration = register_native_case(source)
    ledger = tmp_path / "ledger.csv"
    invoke(ACTIONS[0], source, registration, ledger)
    evidence = native_evidence(source, registration.run_id)
    evidence[slot] = to_document(source.store.publish_bytes(payload, schema_ref=evidence[slot]["schema_ref"]))
    save_evidence(source, registration, evidence)
    monkeypatch.setattr(outcomes, "ledger_append", forbid)
    assert_conflict(lambda: invoke(ACTIONS[1], source, registration, ledger, max_evidence_bytes=CAP))


@pytest.mark.parametrize("slot", ["native_registration", "native_outcome", "native_outcome_receipt"])
def test_bounded_replay_does_not_follow_json_artifact_symlinks(source, tmp_path, monkeypatch, slot):
    registration = register_native_case(source)
    ledger = tmp_path / "ledger.csv"
    invoke(ACTIONS[0], source, registration, ledger)
    ref = from_document(ArtifactRef, native_evidence(source, registration.run_id)[slot])
    path, outside = source.store.root / ref.storage_key, tmp_path / "foreign.json"
    path.rename(outside)
    expected = outside.read_bytes()
    path.symlink_to(outside)
    monkeypatch.setattr(outcomes, "ledger_append", forbid)
    assert_conflict(lambda: invoke(ACTIONS[1], source, registration, ledger, max_evidence_bytes=CAP))
    assert path.is_symlink()
    assert outside.read_bytes() == expected


def test_oversized_outcome_write_refuses_before_publication_or_ledger(source, tmp_path, monkeypatch):
    registration = register_native_case(source)
    before = native_evidence(source, registration.run_id)
    problem = make_problem("EXPERIMENT_VARIANT_FAILED", "synthetic", details={"padding": "漢" * CAP})
    real_publish, schemas = source.store.publish_bytes, []

    def refusal_only(data, *, schema_ref):
        assert schema_ref == "native_experiment_refusal.v1.0"
        schemas.append(schema_ref)
        return real_publish(data, schema_ref=schema_ref)

    monkeypatch.setattr(source.store, "publish_bytes", refusal_only)
    monkeypatch.setattr(outcomes, "ledger_append", forbid)
    assert_conflict(lambda: invoke(ACTIONS[0], source, registration, tmp_path / "ledger.csv",
                                  report=None, problem=problem, max_evidence_bytes=CAP))
    # The outcome writer refuses before publication; the public R6 boundary then publishes its private refusal.
    assert schemas == ["native_experiment_refusal.v1.0"]
    assert native_evidence(source, registration.run_id) == before
    assert not (tmp_path / "ledger.csv").exists()


@pytest.mark.parametrize("options", [{}, {"max_evidence_bytes": None}], ids=["omitted", "explicit-none"])
def test_default_publication_keeps_unbounded_failure_details(source, tmp_path, options):
    registration = register_native_case(source)
    problem = make_problem("EXPERIMENT_VARIANT_FAILED", "synthetic", details={"padding": "漢" * CAP})
    ledger = tmp_path / "ledger.csv"
    first = invoke(ACTIONS[0], source, registration, ledger, report=None, problem=problem, **options)
    assert first["outcome"]["failure_details"] == problem.details
    assert first["receipt"]["outcome_ref"]["byte_size"] > CAP
    assert invoke(ACTIONS[1], source, registration, ledger, **options) == first
    # The exact artifact size remains inclusive, even for non-ASCII JSON.
    assert invoke(ACTIONS[1], source, registration, ledger,
                  max_evidence_bytes=first["receipt"]["outcome_ref"]["byte_size"]) == first


def test_catalog_evidence_growth_refuses_without_reserving_partial_intent(source, tmp_path, monkeypatch):
    registration = register_native_case(source)
    before = pad_evidence(native_evidence(source, registration.run_id), CAP, "漢")
    save_evidence(source, registration, before)
    monkeypatch.setattr(outcomes, "ledger_append", forbid)
    assert_conflict(lambda: invoke(ACTIONS[0], source, registration, tmp_path / "ledger.csv",
                                  max_evidence_bytes=CAP))
    assert native_evidence(source, registration.run_id) == before
    assert not source.conn.in_transaction


@pytest.mark.parametrize("schema", [outcomes.SCHEMA, "native_experiment_completion.v1.0"])
def test_outcome_and_completion_writer_share_utf8_byte_bound(source, monkeypatch, schema):
    document = {"schema_version": schema, "text": "漢" * 100}
    payload = canonical_json(document).encode()
    ref = outcomes._publish(source.store, document, schema, max_evidence_bytes=len(payload))
    assert source.store.read_verified(ref) == payload
    monkeypatch.setattr(source.store, "publish_bytes", forbid)
    assert_conflict(lambda: outcomes._publish(source.store, document, schema,
                                              max_evidence_bytes=len(payload) - 1))


@contextmanager
def artifact_catalog(payload, *, encoding="UTF-8"):
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.execute(f"PRAGMA encoding='{encoding}'")
    conn.execute("CREATE TABLE artifacts (artifact_id TEXT, ref_json TEXT)")
    conn.execute("INSERT INTO artifacts VALUES ('artifact', ?)", (payload,))
    try:
        yield conn
    finally:
        conn.close()


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
@pytest.mark.parametrize("character", ["x", "漢"])
def test_registered_artifact_exact_utf8_bound_and_opt_in_default(encoding, character, monkeypatch):
    ref = replace(artifact_reference(b"result", "result.v1.0"), schema_ref=character * 100)
    payload = canonical_json(to_document(ref))
    size = len(payload.encode())
    with artifact_catalog(payload, encoding=encoding) as conn:
        assert checkpoints.registered_artifact(conn, "artifact") == ref
        assert checkpoints.registered_artifact(conn, "artifact", max_metadata_bytes=None) == ref
        multiplier = 2 if encoding.startswith("UTF-16") else 1
        bounded = BoundedSqlRows(conn, max_bytes=size * multiplier, column=0, sql_fragment="FROM artifacts")
        assert checkpoints.registered_artifact(bounded, "artifact", max_metadata_bytes=size) == ref
        sql_size = len(payload.encode(encoding))
        assert bounded.sizes == [sql_size]
        bounded = BoundedSqlRows(conn, max_bytes=(size - 1) * multiplier, column=0,
                                 sql_fragment="FROM artifacts")
        monkeypatch.setattr(checkpoints, "load_json", forbid)
        with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
            checkpoints.registered_artifact(bounded, "artifact", max_metadata_bytes=size - 1)
        assert bounded.sizes == [None if sql_size > (size - 1) * multiplier else sql_size]


@pytest.mark.parametrize("payload", [None, b"{}"])
def test_registered_artifact_nontext_refusal_precedes_decoder(payload, monkeypatch):
    with artifact_catalog(payload) as conn:
        monkeypatch.setattr(checkpoints, "load_json", forbid)
        with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
            checkpoints.registered_artifact(conn, "artifact", max_metadata_bytes=CAP)


@pytest.mark.parametrize("oversized", [False, True])
def test_registered_artifact_invalid_utf8_is_bounded_before_sql_text_decoding(oversized, monkeypatch):
    with artifact_catalog("{}") as conn:
        conn.execute("UPDATE artifacts SET ref_json=CAST(? AS TEXT)",
                     (b"\xff" * (CAP + 1 if oversized else 1),))
        with pytest.raises(sqlite3.OperationalError, match="Could not decode to UTF-8"):
            conn.execute("SELECT ref_json FROM artifacts").fetchone()
        bounded = BoundedSqlRows(conn, max_bytes=CAP, column=0, sql_fragment="FROM artifacts")
        monkeypatch.setattr(checkpoints, "load_json", forbid)
        with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
            checkpoints.registered_artifact(bounded, "artifact", max_metadata_bytes=CAP)
        assert bounded.sizes == [None if oversized else 1]


@pytest.mark.parametrize("payload, exception, code", [
    ("{", json.JSONDecodeError, None), ("[]", DocumentError, "BAD_TYPE"),
    ("null", DocumentError, "BAD_TYPE"), ("{}", DocumentError, "MISSING_FIELD"),
])
def test_registered_artifact_keeps_strict_metadata_decoder(payload, exception, code):
    with artifact_catalog(payload) as conn:
        with pytest.raises(exception) as error:
            checkpoints.registered_artifact(conn, "artifact", max_metadata_bytes=CAP)
        assert type(error.value) is exception
        if code is not None:
            assert error.value.code == code
        with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
            checkpoints.registered_artifact(conn, "missing", max_metadata_bytes=CAP)
