"""The driver admits a large stored-shape binding only with a finite override.

Real synthetic metadata and feature snapshots supply the base registration.
Its expanded event records form valid closed-schema stored evidence, not a
freshly admitted or executable expanded population. Search admission/dispatch
after loading is synthetic; catalog, artifact verification and loading stay real.
"""
import json
import sqlite3
from dataclasses import asdict

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import ArtifactStore, canonical_json, to_document
from engine.v2.ops.errors import fail
from engine.v2.ops.native_registration import MAX_NATIVE_EVIDENCE_BYTES
from experiments import native_registration as native
from experiments import v2_magnitude_search_run as cli
from tests.v2.ops.native_bounds_support import object_bytes, registration_ref
from tests.v2.ops.test_native_registration_large_binding import scaled_history_binding
from tests.v2.research import magnitude_trial_support as support

ACTION_NAMES = {"seal": "seal_search_run", "run": "run_search", "collect": "collect_outer_results"}
LIMITS = {"memory_bytes": 1024, "cpu_seconds": 2, "wall_seconds": 3}


@pytest.fixture(scope="module")
def large_driver_source(tmp_path_factory):
    source = support.magnitude_source(tmp_path_factory.mktemp("large-driver-binding"), register=False)
    try:
        base, spec = native.prepare_native_registration(
            source.repository, source.spec, scope="shadow", event_ids=None, **support.history_binding(source))
        source.registration = scaled_history_binding(base)
        source.size = len(source.registration.binding_json)
        source.event_count = len(source.registration.document["event_ids"])
        assert source.event_count == 7000 and source.size > MAX_NATIVE_EVIDENCE_BYTES
        # Persist legal stored shape, without claiming fresh semantic admission
        # of the expanded population against the original committed snapshots.
        native.reserve_native_registration(source.conn, source.store, source.registration, spec)
        source.registration_ref = registration_ref(source, source.registration)
        # Search metadata stops at the post-loader stubs; no search plan or fit
        # is admitted for the uncommitted expanded population.
        source.reference = ArtifactRef(artifact_id="synthetic-plan", content_hash="sha256:" + "1" * 64,
            schema_ref="search_plan.v1", byte_size=2, storage_key="objects/synthetic-plan")
        source.documents = {"spec": asdict(spec), "request": {"synthetic_dispatch": True},
                            "plan-ref": to_document(source.reference)}
        source.paths = {}
        for name, document in source.documents.items():
            source.paths[name] = source.root / (name + ".json")
            source.paths[name].write_text(canonical_json(document), encoding="utf-8")
        source.catalog = source.root / "catalog.sqlite"
        yield source
    finally:
        source.conn.close()


@pytest.mark.parametrize("action", ACTION_NAMES)
def test_large_binding_reaches_driver_dispatch_only_with_exact_finite_override(
        large_driver_source, monkeypatch, capsys, record_property, action):
    source = large_driver_source
    record_property("synthetic_event_count", source.event_count)
    record_property("registration_bytes", source.size)
    arguments = [action, "--catalog", str(source.catalog), "--store-root", str(source.store.root)]
    for name, path in source.paths.items():
        arguments += ["--" + name, str(path)]
    if action == "run":
        for name, value in LIMITS.items():
            arguments += ["--" + name.replace("_", "-"), str(value)]
    before_rows, before_objects = support.catalog_rows(source.conn), object_bytes(source.store)
    before_catalog = source.catalog.read_bytes()
    reads, calls, admissions, connections = [], [], [], []
    read_verified = ArtifactStore.read_verified

    def verified_read(store, reference):
        assert store.root == source.store.root
        assert reference == source.registration_ref and reference.byte_size == source.size
        reads.append(reference)
        assert len(reads) == 1
        payload = read_verified(store, reference)
        assert payload == source.registration.binding_json
        return payload

    def admitted(registration):
        assert reads == [source.registration_ref]
        assert registration == source.registration
        assert registration.binding_json == source.registration.binding_json

    def admit_driver(connection, store, registration, request, reference):
        admitted(registration)
        admissions.append(registration.run_id)

    result = {"action": action, "registration_bytes": source.size, "synthetic_dispatch": True}

    def dispatch(connection, store, repository, registration, request, reference, **kwargs):
        admitted(registration)
        assert store.root == source.store.root and isinstance(repository, cli.Repository)
        assert request == source.documents["request"] and reference == source.reference
        assert support.catalog_rows(connection) == before_rows
        if action == "run":
            assert kwargs == {"limits": LIMITS}
        elif action == "collect":
            assert kwargs == {} and admissions == [registration.run_id]
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                connection.execute("UPDATE experiment_runs SET run_id=run_id")
        else:
            assert set(kwargs) == {"clock"}
        connections.append(connection)
        calls.append(action)
        return result

    monkeypatch.setattr(ArtifactStore, "read_verified", verified_read)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", support.forbid_access)
    monkeypatch.setattr(cli.Repository, "scan", support.forbid_access)
    monkeypatch.setattr(cli.Repository, "resolve_pinned", support.forbid_access)
    monkeypatch.setattr(native, "prepare_native_registration", support.forbid_access)
    monkeypatch.setattr(native, "reserve_native_registration", support.forbid_access)
    support.forbid_history_rebuild(monkeypatch)
    monkeypatch.setattr(cli, "_admit_driver", admit_driver)
    for name in ACTION_NAMES.values():
        monkeypatch.setattr(cli, name, support.forbid_access)
    monkeypatch.setattr(cli, ACTION_NAMES[action], dispatch)
    if action == "collect":
        monkeypatch.setattr(cli, "connect", support.forbid_access)

    refusal = {"refused": to_document(fail("EXPERIMENT_IDENTITY_CONFLICT",
        "native registration differs from its immutable binding").problem)}
    for budget in (None, source.size - 1, source.size):
        override = [] if budget is None else ["--max-registration-bytes", str(budget)]
        accepted = budget == source.size
        assert cli.main([*arguments, *override]) == (0 if accepted else 2)
        captured = capsys.readouterr()
        assert not captured.err
        assert json.loads(captured.out) == (result if accepted else refusal)
        assert captured.out == canonical_json(result if accepted else refusal) + "\n"
        assert calls == ([action] if accepted else [])
        assert reads == ([source.registration_ref] if accepted else [])
        assert admissions == ([source.registration.run_id] if accepted and action == "collect" else [])
        assert support.catalog_rows(source.conn) == before_rows
        assert object_bytes(source.store) == before_objects
        if action == "collect":
            assert source.catalog.read_bytes() == before_catalog
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connections[0].execute("SELECT 1")
