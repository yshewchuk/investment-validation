"""Synthetic catalog growth cannot bypass current-request registration bounds."""
from contextlib import nullcontext

import pytest

from engine.v2.foundation import canonical_json, to_document
from engine.v2.ops.errors import OpsError
from experiments import magnitude_inputs
from experiments import native_registration as native
from tests.v2.ops.native_bounds_support import (
    BoundedSqlRows,
    catalog_rows,
    native_evidence,
    object_bytes,
    size_refusal,
)
from tests.v2.research.test_native_registration import MONTH, call_native_registration, native_spec
from tests.v2.research.test_native_registration import source as source


def forbid(*args, **kwargs):
    pytest.fail("revalidation crossed its protected read/effect boundary")


def assert_refusal(error, code, message):
    assert type(error) is OpsError and error.problem.retryable is False
    assert to_document(error.problem) == {
        "schema_version": "problem.v1.0", "code": code, "category": "validation",
        "retryable": False, "message": message, "stage": None, "trace_id": None,
        "dependency_refs": [], "retry_after_seconds": None, "diagnostic_ref": None, "details": {}}


@pytest.mark.parametrize("stage", ["admission", "exists"])
@pytest.mark.parametrize("growth", ["catalog", "reference"])
def test_both_revalidation_reads_refuse_catalog_growth_before_object_access(source, monkeypatch, stage, growth):
    registration = call_native_registration(source)
    cap = len(registration.binding_json)
    assert native.read_native_registration(source.conn, source.store, native_spec(),
                                           max_evidence_bytes=cap) == registration
    before = []

    def grow():
        evidence = native_evidence(source, registration.run_id)
        if growth == "catalog":
            evidence["padding"] = "漢" * cap
        else:
            evidence["native_registration"]["byte_size"] = cap + 1
        source.conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                            (canonical_json(evidence), registration.run_id))
        before.append((catalog_rows(source.conn), object_bytes(source.store)))
        monkeypatch.setattr(source.store, "read_verified", forbid)
        monkeypatch.setattr(source.store, "publish_bytes", forbid)
        monkeypatch.setattr(native, "load_population", forbid)

    if stage == "admission":
        grow()
    else:
        prepare = native.prepare_native_registration
        def prepare_then_grow(*args, **kwargs):
            assert kwargs["max_evidence_bytes"] == cap
            result = prepare(*args, **kwargs)
            grow()
            return result
        monkeypatch.setattr(native, "prepare_native_registration", prepare_then_grow)
    bounded = BoundedSqlRows(source.conn, max_bytes=cap, column="evidence_json", sql_fragment="FROM experiment_runs")
    resource = stage == "admission" and growth == "reference"
    with size_refusal(cap + 1, cap) if resource else pytest.raises(OpsError) as caught:
        native.require_native_registration(bounded, source.store, source.repository, native_spec(),
            code_root=source.code_root, as_of_month=MONTH, expected=registration, max_evidence_bytes=cap)
    if not resource:
        message = ("native evidence requires bounded JSON object text" if growth == "catalog" else
                   "native registration differs from its immutable binding")
        assert_refusal(caught.value, "EXPERIMENT_IDENTITY_CONFLICT", message)
    assert len(before) == 1
    assert (catalog_rows(source.conn), object_bytes(source.store)) == before[0]
    if growth == "catalog":
        assert bounded.sizes[-1] is None


@pytest.mark.parametrize("limit", [None, "exact", "short"])
def test_finite_revalidation_default_and_exact_boundary_preserve_bytes(source, limit):
    registration = call_native_registration(source)
    cap = len(registration.binding_json)
    before = catalog_rows(source.conn), object_bytes(source.store)
    kwargs = {} if limit is None else {"max_evidence_bytes": cap - (limit == "short")}
    with size_refusal(cap, cap - 1) if limit == "short" else nullcontext():
        assert call_native_registration(source, admit=True, **kwargs) == registration
    assert (catalog_rows(source.conn), object_bytes(source.store)) == before


@pytest.mark.parametrize("budget", [None, True, 0, -1, 1.0, "1024", 2**62])
def test_require_rejects_unbounded_or_invalid_budget_before_preparation(monkeypatch, budget):
    monkeypatch.setattr(native, "prepare_native_registration", forbid)
    with pytest.raises(OpsError) as caught:
        native.require_native_registration(None, None, None, None, code_root=None, as_of_month=None,
                                           max_evidence_bytes=budget)
    assert_refusal(caught.value, "INVALID_EXPERIMENT_SPEC",
                   "registration read requires a positive SQLite-safe byte budget")


@pytest.mark.parametrize("budget", [None, 8192])
def test_magnitude_admission_forwards_budget_before_label_preflight(monkeypatch, budget):
    calls = []
    def require(*args, **kwargs):
        calls.append(kwargs["max_evidence_bytes"])
        raise RuntimeError("stop before labels")
    monkeypatch.setattr(magnitude_inputs, "require_native_registration", require)
    monkeypatch.setattr(magnitude_inputs, "preflight_magnitude_targets", forbid)
    kwargs = {} if budget is None else {"max_evidence_bytes": budget}
    with pytest.raises(RuntimeError, match="^stop before labels$"):
        magnitude_inputs.admit_magnitude_inputs(None, None, None, None, code_root=None,
            as_of_month=MONTH, magnitude_training_start="2020-01-01", magnitude_labels_before="2025-01-01", **kwargs)
    assert calls == [native.MAX_NATIVE_EVIDENCE_BYTES if budget is None else budget]
