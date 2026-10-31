"""Native IV registration reserve/read/replay using real synthetic catalogs and CAS."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT
from engine.v2.foundation import SystemClock, canonical_json, content_hash, to_document
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import source_closure
from engine.v2.ops.native_registration import validate_regression_registration
from experiments import iv_crush_search, size_panel
from experiments import native_registration as native
from tests.data_scan_support import commit_tables, contract_ref_for, publish_and_inspect
from tests.v2.research.magnitude_trial_support import catalog_rows, poison_outcomes
from tests.v2.research.test_iv_crush_inputs import poison, target
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401
from tests.v2.research.test_native_registration import artifact_object_bytes, native_spec


def test_native_policy_dispatch_does_not_seal_search_orchestration():
    sources = source_closure(Path(native.__file__).resolve().parents[1],
                             ["experiments/native_registration.py"])
    assert "experiments/iv_crush_search.py" not in sources
    assert "experiments/magnitude_search_plan.py" not in sources
    assert "experiments/iv_crush_registration.py" in sources


@pytest.fixture
def prepared(panel_data, monkeypatch):  # noqa: F811 - imported pytest fixture
    data = panel_data
    repository, store = data.repository, data.repository._store
    features = size_panel.materialize(repository, data.panel_source, **data.panel_request)["feature_snapshot"]
    contract = IV_CRUSH_CONTRACT
    records = [publish_and_inspect(store, contract, contract_ref_for(contract), [target(event)], event["ticker"])
               for event in data.events]
    records.sort(key=lambda record: (record.partition_key, record.primary_key_min))
    from engine.v2.data import objects
    hashes = {partition: objects.partition_logical_hash(store,
        [r.object_ref for r in records if r.partition_key == partition], contract, contract_ref_for(contract), partition)
        for partition in {r.partition_key for r in records}}
    targets = commit_tables(data.conn, SystemClock(), {
        "earnings_events": repository.fragment_records(data.panel_source, "earnings_events"),
        contract.table_name: records}, {"earnings_events": repository.table_contract(data.panel_source, "earnings_events"),
        contract.table_name: contract}, scope="iv-target", receipt_id="iv-target", attempt_id="iv-target",
        partition_logical_hashes={contract.table_name: hashes}, store=store)
    poison_outcomes(monkeypatch, repository, data.panel_source, (), all_outcomes=True)
    poison(monkeypatch, repository, targets)
    spec = native_spec(experiment_id="iv-native", runner="experiments/iv_crush_search.py",
                       seed=0, economic_params={}, price_source="iv_crush_targets")
    options = dict(source=data.panel_source, feature_snapshot=features, target_snapshot=targets,
        scope="panel-source", as_of_month=data.panel_request["as_of_month"], training_start="2022-06-01",
        labels_before="2025-01-01")
    registration, spec = iv_crush_search.prepare(repository, spec, **options)
    return SimpleNamespace(**vars(data), store=store, features=features, targets=targets,
                           spec=spec, registration=registration, options=options)


def test_native_reserve_read_and_current_request_replay(prepared):
    data = prepared
    before = catalog_rows(data.conn)
    registered = native.reserve_native_registration(data.conn, data.store, data.registration, data.spec,
                                                     max_evidence_bytes=native.MAX_NATIVE_EVIDENCE_BYTES)
    assert catalog_rows(data.conn) != before
    assert registered.document["schema_version"] == "native_experiment_registration.v1.8"
    assert registered.document["feature_snapshot"] == to_document(data.features)
    assert registered.document["target_snapshot"] == to_document(data.targets)
    assert data.features.snapshot_id != data.targets.snapshot_id != data.panel_source.snapshot_id
    assert registered.document["regression_plan"]["target_recipe"] == {
        "schema_version": "regression_target.v1", "column": "crush_pct_iv30",
        "prediction_column": "predicted_crush_pct_iv30", "domain": "signed",
        "definition_id": "sha256:f0e35be0f33f06ba24450dd12c87fea1d70671610761628b9b3ff188d6e6f393",
        "target_id": "sha256:8ec392a560559a7b8c75d67ff73b6fcdfc4aa02ff1bc79bfe6f13c8fcb92c23a"}
    assert registered.document["regression_plan"]["horizons"] == [5, 3, 1]
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    loaded = native.read_native_registration(data.conn, data.store, data.spec)
    rebuilt, spec = iv_crush_search.prepare(data.repository, data.spec, **data.options)
    assert loaded.binding_json == rebuilt.binding_json == registered.binding_json
    assert native.reserve_native_registration(data.conn, data.store, rebuilt, spec) == registered
    native.verify_native_registration(data.conn, data.store, loaded)
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects


@pytest.mark.parametrize("change", ["target", "source", "features", "clock", "unknown", "plan", "seed"])
def test_malformed_binding_refuses_before_any_publication(prepared, change):
    data = prepared
    document = data.registration.document
    if change == "target":
        document["regression_plan"]["target_recipe"]["domain"] = "nonnegative"
    elif change in ("source", "features"):
        key = "snapshot" if change == "source" else "feature_snapshot"
        document[key]["unrecognized"] = True
    elif change == "clock":
        document["regression_execution"]["evaluation_labels_before"] = "2024-01-01"
    elif change == "unknown":
        document["other"] = 1
    elif change == "plan":
        document["regression_plan"]["folds"][0]["evaluation_event_ids"] = []
        plan = document["regression_plan"]
        plan["plan_id"] = content_hash({k: v for k, v in plan.items() if k != "plan_id"})
    else:
        document["execution_plan"]["seed"] = True
    altered = replace(data.registration, binding_json=canonical_json(document).encode())
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    with pytest.raises(OpsError) as caught:
        native.reserve_native_registration(data.conn, data.store, altered, data.spec)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects


@pytest.mark.parametrize("field", ["snapshot", "feature_snapshot", "target_snapshot"])
def test_changed_valid_snapshot_pin_cannot_replay_reserved_identity(prepared, field):
    data = prepared
    native.reserve_native_registration(data.conn, data.store, data.registration, data.spec)
    document = data.registration.document
    document[field] = to_document(data.targets if field != "target_snapshot" else data.features)
    altered = replace(data.registration, binding_json=canonical_json(document).encode())
    with pytest.raises(OpsError) as caught:
        native.verify_native_registration(data.conn, data.store, altered)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_changed_cutoff_creates_different_binding_but_cannot_reregister(prepared):
    data = prepared
    native.reserve_native_registration(data.conn, data.store, data.registration, data.spec)
    changed, spec = iv_crush_search.prepare(data.repository, data.spec,
                                          **{**data.options, "labels_before": "2025-02-01"})
    assert changed.variant_id != data.registration.variant_id
    with pytest.raises(OpsError) as caught:
        native.reserve_native_registration(data.conn, data.store, changed, spec)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_invalid_committed_reference_and_wrong_runner_refuse(prepared):
    data = prepared
    with pytest.raises(DataError):
        iv_crush_search.prepare(data.repository, data.spec,
            **{**data.options, "target_snapshot": replace(data.targets, manifest_hash="sha256:" + "0" * 64)})
    with pytest.raises(OpsError) as caught:
        iv_crush_search.prepare(data.repository, replace(data.spec, runner="experiments/size_search.py"), **data.options)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    validate_regression_registration(data.registration.document)


@pytest.mark.parametrize(("section", "field", "value"), [
    ("target", "domain", "nonnegative"),
    ("target", "column", "other_target"),
    ("target", "prediction_column", "other_prediction"),
    ("target", "definition_id", "sha256:" + "0" * 64),
    ("execution", "price_source", "computed_moves"),
    ("execution", "runner", "experiments/size_search.py"),
])
@pytest.mark.parametrize("operation", ["validate", "reserve", "read"])
def test_rehashed_non_iv_identity_refuses_at_each_boundary(prepared, section, field, value, operation):
    data = prepared
    document = data.registration.document
    plan = document["regression_plan"]
    target = plan["target_recipe"]
    (target if section == "target" else document["execution_plan"])[field] = value
    target["target_id"] = content_hash({k: v for k, v in target.items() if k != "target_id"})
    plan["plan_id"] = content_hash({k: v for k, v in plan.items() if k != "plan_id"})
    validate_regression_registration(document)  # Generic wire policy remains target-agnostic.
    changed = replace(data.registration, binding_json=canonical_json(document).encode())
    if operation == "read":
        native.reserve_native_registration(data.conn, data.store, data.registration, data.spec)
        ref = data.store.publish_bytes(changed.binding_json, schema_ref=document["schema_version"])
        evidence = {"native_registration": to_document(ref), "variant_id": changed.variant_id, "planned_variants": 1}
        data.conn.execute("UPDATE experiment_runs SET input_hash=?, evidence_json=? WHERE run_id=?",
                          (changed.variant_id, canonical_json(evidence), changed.run_id))
        data.conn.execute("UPDATE hypotheses SET input_hash=?, payload_hash=? WHERE run_id=?",
            (changed.variant_id, content_hash({"spec": data.spec.spec_hash, "input": changed.variant_id}), changed.run_id))
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    with pytest.raises(OpsError) as caught:
        if operation == "validate":
            iv_crush_search.validate_registration(document)
        elif operation == "reserve":
            native.reserve_native_registration(data.conn, data.store, changed, data.spec)
        else:
            native.read_native_registration(data.conn, data.store, data.spec)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects


def test_non_iv_generic_regression_can_still_reserve_and_read(prepared):
    data = prepared
    spec = replace(data.spec, runner="experiments/v2_magnitude_search.py", price_source="computed_moves")
    document = data.registration.document
    document["spec_hash"] = spec.spec_hash
    document["execution_plan"].update(runner=spec.runner, price_source=spec.price_source)
    plan = document["regression_plan"]
    target = plan["target_recipe"]
    target["domain"] = "nonnegative"
    target["target_id"] = content_hash({k: v for k, v in target.items() if k != "target_id"})
    plan["plan_id"] = content_hash({k: v for k, v in plan.items() if k != "plan_id"})
    registration = replace(data.registration, binding_json=canonical_json(document).encode())
    native.reserve_native_registration(data.conn, data.store, registration, spec)
    assert native.read_native_registration(data.conn, data.store, spec) == registration
    native.verify_native_registration(data.conn, data.store, registration)


def test_inline_native_byte_cap_refuses_without_publication(prepared, monkeypatch):
    data = prepared
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    monkeypatch.setattr(native, "MAX_NATIVE_EVIDENCE_BYTES", 100)
    with pytest.raises(OpsError) as caught:
        iv_crush_search.prepare(data.repository, data.spec, **data.options)
    assert caught.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects


@pytest.mark.parametrize("change", ["spec", "run_id", "runner", "price_source"])
@pytest.mark.parametrize("bounded", [False, True])
def test_requested_spec_binding_refuses_before_reservation(prepared, change, bounded):
    data = prepared
    spec, document = data.spec, data.registration.document
    if change == "spec":
        spec = replace(spec, hypothesis="different declared hypothesis")
    elif change != "run_id":
        # Keep the requested hash self-consistent while forging only its execution policy.
        spec = replace(spec, runner="experiments/v2_magnitude_search.py", price_source="computed_moves")
        document["spec_hash"] = spec.spec_hash
        document["execution_plan"].update(runner=spec.runner, price_source=spec.price_source)
        document["execution_plan"][change] = "other"
    changed = replace(data.registration, binding_json=canonical_json(document).encode())
    if change == "run_id":
        changed = replace(changed, run_id="native_" + "0" * 64)
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    options = {"max_evidence_bytes": native.MAX_NATIVE_EVIDENCE_BYTES} if bounded else {}
    with pytest.raises(OpsError) as caught:
        native.reserve_native_registration(data.conn, data.store, changed, spec, **options)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.message == "requested specification differs from registration"
    assert not caught.value.problem.retryable
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects


@pytest.mark.parametrize("field", ["runner", "price_source"])
def test_rehashed_generic_execution_cannot_change_requested_read(prepared, field):
    data = prepared
    spec = replace(data.spec, runner="experiments/v2_magnitude_search.py", price_source="computed_moves")
    document = data.registration.document
    document["spec_hash"] = spec.spec_hash
    document["execution_plan"].update(runner=spec.runner, price_source=spec.price_source)
    registration = replace(data.registration, binding_json=canonical_json(document).encode())
    native.reserve_native_registration(data.conn, data.store, registration, spec)
    document["execution_plan"][field] = "other"
    changed = replace(registration, binding_json=canonical_json(document).encode())
    ref = data.store.publish_bytes(changed.binding_json, schema_ref=document["schema_version"])
    evidence = {"native_registration": to_document(ref), "variant_id": changed.variant_id, "planned_variants": 1}
    data.conn.execute("UPDATE experiment_runs SET input_hash=?, evidence_json=? WHERE run_id=?",
                      (changed.variant_id, canonical_json(evidence), changed.run_id))
    data.conn.execute("UPDATE hypotheses SET input_hash=?, payload_hash=? WHERE run_id=?",
        (changed.variant_id, content_hash({"spec": spec.spec_hash, "input": changed.variant_id}), changed.run_id))
    native.verify_native_registration(data.conn, data.store, changed)
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    with pytest.raises(OpsError) as caught:
        native.read_native_registration(data.conn, data.store, spec)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.message == "requested specification differs from registration"
    assert not caught.value.problem.retryable
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects


@pytest.mark.parametrize("operation", ["reserve", "read"])
def test_iv_requested_runner_rejects_downgraded_legacy_wire(prepared, operation):
    data = prepared
    document = data.registration.document
    document["schema_version"] = "native_experiment_registration.v1.0"
    changed = replace(data.registration, binding_json=canonical_json(document).encode())
    if operation == "read":
        native.reserve_native_registration(data.conn, data.store, data.registration, data.spec)
        ref = data.store.publish_bytes(changed.binding_json, schema_ref=document["schema_version"])
        evidence = {"native_registration": to_document(ref), "variant_id": changed.variant_id, "planned_variants": 1}
        data.conn.execute("UPDATE experiment_runs SET input_hash=?, evidence_json=? WHERE run_id=?",
            (changed.variant_id, canonical_json(evidence), changed.run_id))
        data.conn.execute("UPDATE hypotheses SET input_hash=?, payload_hash=? WHERE run_id=?",
            (changed.variant_id, content_hash({"spec": data.spec.spec_hash, "input": changed.variant_id}), changed.run_id))
        native.verify_native_registration(data.conn, data.store, changed)
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    with pytest.raises(OpsError) as caught:
        if operation == "reserve":
            native.reserve_native_registration(data.conn, data.store, changed, data.spec)
        else:
            native.read_native_registration(data.conn, data.store, data.spec)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.message == "regression registration binding differs"
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects
