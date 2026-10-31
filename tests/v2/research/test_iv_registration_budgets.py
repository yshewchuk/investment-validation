"""New regression defaults are bounded without tightening legacy native callers."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.foundation import canonical_json, to_document
from engine.v2.ops import native_registration as evidence
from experiments import native_registration as native
from tests.v2.ops.native_bounds_support import size_refusal
from tests.v2.ops.test_native_evidence_bounds import evidence_catalog, registered_binding
from tests.v2.research.magnitude_trial_support import catalog_rows
from tests.v2.research.test_iv_crush_registration import prepared  # noqa: F401
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401
from tests.v2.research.test_native_registration import artifact_object_bytes


def forbidden(*args, **kwargs):
    pytest.fail("oversized regression evidence reached storage or identity inspection")


def test_low_level_regression_default_cap_precedes_artifact_read():
    with evidence_catalog() as conn:
        _, ref, document = registered_binding(conn)
        ref = replace(ref, schema_ref="native_experiment_registration.v1.8",
                      byte_size=evidence.MAX_NATIVE_EVIDENCE_BYTES + 1)
        document["native_registration"] = to_document(ref)
        conn.execute("UPDATE experiment_runs SET evidence_json=?", (canonical_json(document),))
        before = list(conn.iterdump())
        with size_refusal(ref.byte_size, evidence.MAX_NATIVE_EVIDENCE_BYTES):
            evidence.verified_registration_bytes(conn, SimpleNamespace(read_verified=forbidden), "native-test")
        assert list(conn.iterdump()) == before


@pytest.mark.parametrize("verify", [native.verify_native_registration, native.native_registration_exists])
def test_regression_verify_default_cap_precedes_object_read(prepared, monkeypatch, verify):  # noqa: F811
    data = prepared
    native.reserve_native_registration(data.conn, data.store, data.registration, data.spec)
    size = len(data.registration.binding_json)
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    monkeypatch.setattr(evidence, "MAX_NATIVE_EVIDENCE_BYTES", size - 1)
    monkeypatch.setattr(data.store, "read_verified", forbidden)
    with size_refusal(size, size - 1):
        verify(data.conn, data.store, data.registration)
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects


def test_regression_reserve_default_cap_precedes_identity_and_publication(prepared, monkeypatch):  # noqa: F811
    data = prepared
    size = len(data.registration.binding_json)
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    monkeypatch.setattr(native, "MAX_NATIVE_EVIDENCE_BYTES", size - 1)
    monkeypatch.setattr(data.store, "publish_bytes", forbidden)
    with size_refusal(size, size - 1):
        native.reserve_native_registration(data.conn, data.store, data.registration, None)
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects
