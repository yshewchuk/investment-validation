"""Final-read schemas and authority-free preparation over synthetic metadata only."""
import json
import sqlite3
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.contracts import FinalHoldoutDeclaration, FinalHoldoutRequest, SnapshotRef
from engine.v2.foundation import ArtifactStore, artifact_reference, canonical_json, content_hash, to_document
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.foundation.final_holdout import (
    declaration_from_document, request_from_document, request_from_registration,
)
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiments import ExperimentSpec
from experiments import native_registration as native


def _metadata_ref(store, label, schema):
    return store.publish_bytes(canonical_json({"synthetic_metadata": label}).encode(), schema_ref=schema)


@pytest.fixture
def source(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=lambda: datetime(2026, 10, 10, tzinfo=timezone.utc))
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    store = ArtifactStore(tmp_path / "store")
    snapshot = SnapshotRef(snapshot_id="synthetic-snapshot", manifest_hash=content_hash("manifest"),
                           table_versions={}, calendar_version="synthetic-calendar",
                           source_priority_version="synthetic-priority", finality_receipt_refs=(),
                           knowledge_mode_by_table={})
    repository = SimpleNamespace(resolve_pinned=lambda scope: snapshot)
    (tmp_path / "runner.py").write_text("raise AssertionError('runner must never execute')\n")
    spec = ExperimentSpec(experiment_id="synthetic-experiment", hypothesis="synthetic metadata",
                          primary_arm_id="winner", arms=("winner",), seed=1, folds=("fold",),
                          economic_params={"fill": 0.5}, price_source="option_chains", runner="runner.py")
    context = {"holdout_as_of_month": "2025-01",
               "random_membership_version": ExperimentHoldouts.random_version,
               "rolling_membership_version": ExperimentHoldouts.rolling_version}
    monkeypatch.setattr(native, "load_population", lambda *a, **k: pd.DataFrame([
        {"event_id": "synthetic-admitted-metadata", **context}]))
    declaration = FinalHoldoutDeclaration(
        evaluation_recipe_ref=_metadata_ref(store, "recipe", "final_holdout_recipe.v1.0"),
        completeness_bar_ref=_metadata_ref(store, "completeness", "final_holdout_completeness.v1.0"),
        frozen_model_refs=(_metadata_ref(store, "model identity", "synthetic_model_metadata.v1.0"),))
    value = SimpleNamespace(conn=conn, store=store, repository=repository, spec=spec,
                            root=tmp_path, declaration=declaration, snapshot=snapshot)
    yield value
    conn.close()


def _register(source, **kwargs):
    return native.register_native(source.conn, source.store, source.repository, source.spec,
                                  code_root=source.root, as_of_month="2025-01", **kwargs)


def _prepared(source):
    registration = _register(source, final_read=source.declaration)
    return registration, native.prepare_final_holdout_request(source.conn, source.store, registration)


def test_request_is_exact_immutable_projection_not_a_grant(source):
    registration, request = _prepared(source)
    document = registration.document
    assert document["schema_version"] == "native_experiment_registration.v1.1"
    assert document["final_read"] == to_document(source.declaration)
    assert request == FinalHoldoutRequest(
        experiment_id=source.spec.experiment_id, registration_run_id=registration.run_id,
        variant_id=registration.variant_id,
        registration_ref=artifact_reference(registration.binding_json, document["schema_version"]),
        snapshot_id=source.snapshot.snapshot_id,
        execution_plan_hash=content_hash(document["execution_plan"]),
        source_closure_hash=content_hash(document["source_closure"]),
        final_read=source.declaration, **document["holdouts"])
    assert request_from_document(to_document(request)) == request
    assert request_from_registration(document, registration.run_id,
                                     to_document(request.registration_ref)) == request
    assert declaration_from_document(to_document(source.declaration)) == source.declaration
    assert native.prepare_final_holdout_request(source.conn, source.store, registration) == request
    with pytest.raises(FrozenInstanceError):
        request.experiment_id = "changed"
    with pytest.raises(FrozenInstanceError):
        request.final_read.completeness_bar_ref = None
    detached = to_document(request)
    detached["final_read"]["frozen_model_refs"].clear()
    assert request.final_read.frozen_model_refs
    assert not {"decision_id", "authorization", "actor", "expires_at"} & set(to_document(request))


def test_preparation_reads_only_the_original_registration_and_never_writes(source, monkeypatch):
    registration, expected = _prepared(source)
    before = source.conn.total_changes
    read = source.store.read_verified
    seen = []

    def registration_only(ref):
        assert ref == expected.registration_ref
        seen.append(ref)
        return read(ref)

    def forbidden(*args, **kwargs):
        pytest.fail("preparation attempted a prohibited effect or data access")

    monkeypatch.setattr(source.store, "read_verified", registration_only)
    monkeypatch.setattr(source.store, "publish_bytes", forbidden)
    monkeypatch.setattr(source.repository, "resolve_pinned", forbidden)
    monkeypatch.setattr(native, "load_population", forbidden)
    monkeypatch.setattr(native, "_binding", forbidden)
    source.conn.set_authorizer(lambda action, *args: sqlite3.SQLITE_OK if action in
                              (sqlite3.SQLITE_READ, sqlite3.SQLITE_SELECT) else sqlite3.SQLITE_DENY)
    try:
        assert native.prepare_final_holdout_request(source.conn, source.store, registration) == expected
    finally:
        source.conn.set_authorizer(None)
    assert seen == [expected.registration_ref]
    assert source.conn.total_changes == before


def test_legacy_registration_stays_readable_but_cannot_prepare_final_request(source):
    registration = _register(source)
    assert registration.document["schema_version"] == "native_experiment_registration.v1.0"
    assert "final_read" not in registration.document
    native.verify_native_registration(source.conn, source.store, registration)
    assert _register(source) == registration
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        native.prepare_final_holdout_request(source.conn, source.store, registration)


def test_declaration_is_preregistered_and_cannot_be_replaced_afterward(source):
    registration, original = _prepared(source)
    changed = replace(source.declaration, frozen_model_refs=())
    before = source.conn.total_changes
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _register(source, final_read=changed)
    with pytest.raises(TypeError):
        native.prepare_final_holdout_request(source.conn, source.store, registration, final_read=changed)
    assert native.prepare_final_holdout_request(source.conn, source.store, registration) == original
    assert source.conn.total_changes == before
    assert native.require_native_registration(
        source.conn, source.store, source.repository, source.spec, code_root=source.root,
        as_of_month="2025-01", final_read=source.declaration) == registration


@pytest.mark.parametrize("declared", [False, True])
def test_stored_read_and_expected_admission_preserve_both_registration_versions(source, declared):
    declaration = source.declaration if declared else None
    registration = _register(source, final_read=declaration)
    assert native.read_native_registration(source.conn, source.store, source.spec) == registration
    before = source.conn.total_changes
    assert native.require_native_registration(
        source.conn, source.store, source.repository, source.spec, code_root=source.root,
        as_of_month="2025-01", final_read=declaration, expected=registration) == registration
    assert source.conn.total_changes == before


@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("changed", ["spec", "source", "snapshot", "scope", "environment", "declaration"])
def test_expected_identity_refuses_drift_before_population(source, monkeypatch, declared, changed):
    declaration = source.declaration if declared else None
    registration = _register(source, final_read=declaration)
    arguments = {"code_root": source.root, "as_of_month": "2025-01",
                 "final_read": declaration, "expected": registration}
    spec = source.spec
    if changed == "spec":
        spec = replace(spec, seed=2)
    elif changed == "source":
        monkeypatch.setattr(native, "source_closure", lambda *a: {"runner.py": content_hash("changed")})
    elif changed == "snapshot":
        monkeypatch.setattr(source.repository, "resolve_pinned",
                            lambda scope: replace(source.snapshot, snapshot_id="changed"))
    elif changed == "scope":
        arguments["scope"] = "changed"
    elif changed == "environment":
        monkeypatch.setattr(native, "environment_identity", lambda *a: {"changed": True})
    else:
        arguments["final_read"] = replace(source.declaration, frozen_model_refs=())

    def forbidden(*args, **kwargs):
        pytest.fail("changed identity reached population or publication")

    monkeypatch.setattr(native, "load_population", forbidden)
    monkeypatch.setattr(source.store, "publish_bytes", forbidden)
    before = source.conn.total_changes
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        native.require_native_registration(source.conn, source.store, source.repository, spec, **arguments)
    assert source.conn.total_changes == before


@pytest.mark.parametrize("field", ["evaluation_recipe_ref", "completeness_bar_ref", "frozen_model_refs"])
def test_every_final_read_commitment_changes_request_identity(source, field):
    _, request = _prepared(source)
    changed = to_document(request)
    replacement = _metadata_ref(source.store, "changed", "synthetic_model_metadata.v1.0")
    if field == "frozen_model_refs":
        changed["final_read"][field] = [to_document(replacement)]
    else:
        schema = changed["final_read"][field]["schema_ref"]
        changed["final_read"][field] = to_document(_metadata_ref(source.store, "changed", schema))
    altered = request_from_document(changed)
    assert altered.experiment_id == request.experiment_id
    assert content_hash(to_document(altered)) != content_hash(to_document(request))


@pytest.mark.parametrize("field,value", [
    ("schema_version", "final_holdout_request.v2.0"), ("schema_version", "final_holdout_request.v1.1"),
    ("unexpected", True), ("experiment_id", ""), ("experiment_id", " padded"),
    ("experiment_id", None), ("registration_run_id", ""), ("snapshot_id", " "),
    ("variant_id", "sha256:" + "A" * 64), ("variant_id", content_hash("different")),
    ("execution_plan_hash", "bad"), ("source_closure_hash", 12),
    ("holdout_as_of_month", "2025-13"), ("holdout_as_of_month", "2025-1"),
    ("random_membership_version", "unknown"), ("rolling_membership_version", "unknown"),
    ("final_read", None), ("registration_ref", None),
])
def test_request_refuses_invalid_closed_shape(source, field, value):
    _, request = _prepared(source)
    document = {**to_document(request), field: value}
    with pytest.raises(ValueError):
        request_from_document(document)


@pytest.mark.parametrize("field", list(FinalHoldoutRequest.__dataclass_fields__))
def test_request_requires_each_field(source, field):
    _, request = _prepared(source)
    document = to_document(request)
    del document[field]
    with pytest.raises(ValueError):
        request_from_document(document)


@pytest.mark.parametrize("field,value", [
    ("byte_size", True), ("byte_size", 0), ("byte_size", -1), ("content_hash", "bad"),
    ("artifact_id", "other"), ("storage_key", "../other"), ("schema_ref", "wrong.v1.0"),
    ("schema_version", "artifact_ref.v2.0"), ("unknown", 1),
])
def test_declaration_rejects_malformed_references_before_registration(source, field, value):
    document = to_document(source.declaration)
    document["completeness_bar_ref"][field] = value
    before = source.conn.total_changes
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        _register(source, final_read=document)
    assert source.conn.total_changes == before


@pytest.mark.parametrize("kind", ["missing", "variant", "ref", "bytes"])
def test_original_registration_corruption_refuses_without_effects(source, kind, monkeypatch):
    registration, request = _prepared(source)
    row = source.conn.execute("SELECT evidence_json FROM experiment_runs WHERE run_id=?",
                              (registration.run_id,)).fetchone()
    evidence = json.loads(row[0])
    if kind == "missing":
        del evidence["native_registration"]
    elif kind == "variant":
        evidence["variant_id"] = content_hash("changed")
    elif kind == "ref":
        evidence["native_registration"]["schema_ref"] = "other.v1.0"
    else:
        monkeypatch.setattr(source.store, "read_verified", lambda ref: b"corrupt metadata")
    source.conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                        (json.dumps(evidence), registration.run_id))
    before = source.conn.total_changes
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        native.prepare_final_holdout_request(source.conn, source.store, registration)
    assert source.conn.total_changes == before


def test_request_tests_are_registered_in_both_affected_mutation_owners():
    from tools import mutation_pilot

    if not (mutation_pilot.REPO / ".git").exists():
        pytest.skip("mutation work copy has no git index")
    config = mutation_pilot.load_config()
    for name in ("foundation", "ops_cli"):
        assert "tests/v2/research/test_final_holdout_requests.py" in mutation_pilot.test_files(config, name)


@pytest.mark.parametrize("path,value", [
    (("execution_plan",), None), (("execution_plan", "arms"), []),
    (("execution_plan", "arms"), ["first", "second"]),
    (("execution_plan", "schema_version"), "other.v1.0"),
    (("snapshot",), {}), (("snapshot", "manifest_hash"), "bad"),
    (("source_closure",), None), (("source_closure",), {}),
    (("source_closure",), {"runner.py": "bad"}),
    (("holdouts", "extra"), "unknown"), (("final_read",), None),
])
def test_projection_refuses_malformed_registration_metadata(source, path, value):
    registration, request = _prepared(source)
    document = registration.document
    target = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    ref = artifact_reference(canonical_json(document).encode(), document["schema_version"])
    with pytest.raises(ValueError):
        request_from_registration(document, registration.run_id, ref)
    assert native.prepare_final_holdout_request(source.conn, source.store, registration) == request
