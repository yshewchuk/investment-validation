"""Prepared regression wire does not authorize external inputs on requested admission."""
from dataclasses import replace
from pathlib import Path

import pytest

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiments import resolve_experiment_plan
from experiments import iv_crush_search
from experiments import native_registration as native
from tests.v2.research.magnitude_trial_support import catalog_rows
from tests.v2.research.test_iv_crush_registration import prepared  # noqa: F401
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401
from tests.v2.research.test_native_registration import artifact_object_bytes


@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize("operation", ["prepare", "reserve", "reserve_existing", "read"])
def test_external_files_refuse_consistently_without_mutation(prepared, monkeypatch, generic, operation):  # noqa: F811
    data = prepared
    spec, registration = data.spec, data.registration
    if generic:
        spec = replace(spec, runner="experiments/v2_magnitude_search.py", price_source="computed_moves")
        document = registration.document
        document["spec_hash"] = spec.spec_hash
        document["execution_plan"].update(runner=spec.runner, price_source=spec.price_source)
        registration = replace(registration, binding_json=canonical_json(document).encode())
    if operation in ("reserve_existing", "read"):
        native.reserve_native_registration(data.conn, data.store, registration, spec)
    invalid = replace(spec, input_files=("external-must-not-be-opened.csv",))
    assert invalid.spec_hash == spec.spec_hash
    assert resolve_experiment_plan(invalid) == resolve_experiment_plan(spec)
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    monkeypatch.setattr(data.store, "publish_bytes", lambda *a, **k: pytest.fail("invalid request published"))
    with pytest.raises(OpsError) as caught:
        if operation == "prepare":
            if generic:
                native.prepare_regression_registration(data.repository, invalid,
                    code_root=Path(native.__file__).resolve().parents[1],
                    target=registration.document["regression_plan"]["target_recipe"], **data.options)
            else:
                iv_crush_search.prepare(data.repository, invalid, **data.options)
        elif operation.startswith("reserve"):
            native.reserve_native_registration(data.conn, data.store, registration, invalid)
        else:
            native.read_native_registration(data.conn, data.store, invalid)
    assert caught.value.problem.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.message == "native registration requires a Python runner and no file inputs"
    assert caught.value.problem.retryable is False
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects


@pytest.mark.parametrize("change", [
    {"runner": "../escape.py"}, {"runner": "experiments/not_python.sh"},
    {"hypothesis": " "}, {"experiment_id": " "}, {"primary_arm_id": " ", "arms": (" ",)},
])
def test_self_consistent_regression_wire_cannot_bypass_existing_supported_spec_rule(prepared, monkeypatch, change):  # noqa: F811
    data = prepared
    base = replace(data.spec, runner="experiments/v2_magnitude_search.py", price_source="computed_moves")
    invalid = replace(base, **change)
    document = data.registration.document
    document.update(spec_hash=invalid.spec_hash, execution_plan=resolve_experiment_plan(invalid).as_document())
    run_id = "native_" + content_hash({"experiment_id": invalid.experiment_id,
                                    "arm": invalid.primary_arm_id}).removeprefix("sha256:")
    registration = native.NativeRegistration(run_id, canonical_json(document).encode())
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    monkeypatch.setattr(data.store, "publish_bytes", lambda *a, **k: pytest.fail("unsupported spec published"))
    with pytest.raises(OpsError) as caught:
        native.reserve_native_registration(data.conn, data.store, registration, invalid)
    assert caught.value.problem.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.message == "native registration requires a Python runner and no file inputs"
    assert caught.value.problem.retryable is False
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects
