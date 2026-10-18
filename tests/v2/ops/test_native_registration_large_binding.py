"""A finite override admits a large, closed-schema stored history binding.

The base registration comes from real synthetic metadata and feature snapshots.
Scaling its event records exercises stored-shape integrity only: the expanded
population is not committed to those snapshots, so this is not fresh semantic
admission or an executable research population.
"""
from dataclasses import replace

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.ops.errors import OpsError
from engine.v2.ops.native_registration import MAX_NATIVE_EVIDENCE_BYTES
from engine.v2.research.magnitude_folds import annual_fold
from experiments import native_registration as native
from tests.v2.ops.native_bounds_support import object_bytes
from tests.v2.research import magnitude_trial_support as support


def scaled_history_binding(registration):
    """Add legal event records and repair every duplicated plan/hash cross-link."""
    document = registration.document
    assert document["schema_version"] == "native_experiment_registration.v1.4"
    plan = document["magnitude_plan"]
    originals = plan["events"]
    plan["events"] = sorted([
        {**event, "event_id": f"{event['event_id']}:replica-{replica:04d}",
         "ticker": f"{event['ticker']}_{replica:04d}"}
        for replica in range(700) for event in originals
    ], key=lambda event: (event["event_date"], event["event_id"]))
    plan["folds"] = [annual_fold(year, plan["events"]) for year in plan["evaluation_years"]]
    plan["plan_id"] = content_hash({key: value for key, value in plan.items() if key != "plan_id"})
    document["event_ids"] = sorted(event["event_id"] for event in plan["events"])
    context = document["magnitude_history"]["context"]
    context["temporal_plan"] = plan
    context["context_id"] = content_hash({key: value for key, value in context.items() if key != "context_id"})
    return replace(registration, binding_json=canonical_json(document).encode())


def test_large_history_binding_requires_finite_override_and_loads_unchanged(tmp_path, monkeypatch, record_property):
    source = support.magnitude_source(tmp_path, register=False)
    try:
        base, spec = native.prepare_native_registration(
            source.repository, source.spec, scope="shadow", event_ids=None, **support.history_binding(source))
        registration = scaled_history_binding(base)
        size = len(registration.binding_json)
        count = len(registration.document["event_ids"])
        assert count == 7000 and size > MAX_NATIVE_EVIDENCE_BYTES
        record_property("synthetic_event_count", count)
        record_property("registration_bytes", size)
        # Persist a stored-shape fixture through the existing reservation writer.
        # This does not claim the expanded metadata passed fresh preparation.
        native.reserve_native_registration(source.conn, source.store, registration, spec)
        before_rows = support.catalog_rows(source.conn)
        before_objects = object_bytes(source.store)
        source.conn.execute("PRAGMA query_only=ON")
        monkeypatch.setattr(source.store, "publish_bytes", support.forbid_access)
        monkeypatch.setattr(source.repository, "scan", support.forbid_access)
        monkeypatch.setattr(source.repository, "resolve_pinned", support.forbid_access)
        read_verified, reads = source.store.read_verified, []

        def read_once(ref):
            reads.append(ref)
            assert len(reads) == 1 and ref.byte_size == size
            return read_verified(ref)

        monkeypatch.setattr(source.store, "read_verified", read_once)
        with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
            native.read_native_registration(source.conn, source.store, spec)
        assert reads == []
        with support.refuses(OpsError, "EXPERIMENT_IDENTITY_CONFLICT"):
            native.read_native_registration(source.conn, source.store, spec, max_evidence_bytes=size - 1)
        assert reads == []
        loaded = native.read_native_registration(source.conn, source.store, spec, max_evidence_bytes=size)
        assert loaded == registration and loaded.binding_json == registration.binding_json
        assert len(reads) == 1
        assert support.catalog_rows(source.conn) == before_rows
        assert object_bytes(source.store) == before_objects
    finally:
        source.conn.close()
