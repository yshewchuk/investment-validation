"""Production registry and worker-path tests for Phase 3B incremental refresh."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import content_hash
from engine.v2.ops import incremental_data, provider_budget, worker
from engine.v2.ops.errors import OpsError
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit, validate_request
from tests.ops_support import catalog, sample

POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


def _spec(*, calls=2, account="orats-account"):
    return JobSpec(
        kind="incremental_refresh",
        implementation_ref="code",
        spec_hash=None,
        environment_ref="env",
        parameters={
            "expected_ids": ["daily-market-2026-09-15"],
            "parent_snapshot_id": "snap-parent",
            "refresh_plan_hash": content_hash({"plan": 1}),
            "provider_calls": calls,
            "catalog_path": "catalog.sqlite",
            "objects_root": "objects",
            "scope": "shadow",
            "expected_head_generation": 1,
            "expected_head_snapshot_id": "snap-parent",
            "table_name": "daily_market",
            "input_bindings": None,
        },
        input_refs=(),
        output_namespace="shadow",
        resource_class="io_fetch",
        provider_budget_ref=account,
        retry_policy_ref="bounded",
        checkpoint_contract_ref=incremental_data.REFRESH_RESULT_SCHEMA,
    )


def _request(spec=None):
    return SubmitRequest(
        namespace="shadow",
        idempotency_key="incremental-2026-09-15",
        principal="operator",
        job=spec or _spec(),
    )


def _parameters():
    return _spec().parameters


def _evidence(params, *, status="complete", coverage_advanced=True,
              candidate_snapshot_id="snap-candidate", **overrides):
    document = {
        "schema_version": incremental_data.REFRESH_RESULT_SCHEMA,
        "status": status,
        "completed_ids": list(params.expected_ids),
        "coverage_advanced": coverage_advanced,
        "parent_snapshot_id": params.parent_snapshot_id,
        "refresh_plan_hash": params.refresh_plan_hash,
        "candidate_snapshot_id": candidate_snapshot_id,
    }
    document.update(overrides)
    return document


def _write_evidence(root, document):
    (root / incremental_data.REFRESH_RESULT_PATH).write_text(
        json.dumps(document, sort_keys=True))


def test_production_registry_owns_incremental_refresh_contract():
    kind = registry().get("incremental_refresh")
    assert kind.worker == "incremental_refresh"
    assert kind.parameters is incremental_data.RefreshParameters
    assert kind.resource_classes == frozenset({"io_fetch"})
    assert kind.retry.max_attempts == 3
    assert kind.retry.backoff_seconds == (5, 65)
    assert kind.checkpoint_contract == incremental_data.REFRESH_RESULT_SCHEMA


def test_real_registry_strictly_validates_parameters_and_provider_pairing():
    validate_request(registry(), POLICY, _request())

    malformed = dict(_spec().parameters)
    malformed["unexpected"] = True
    with pytest.raises(OpsError) as caught:
        validate_request(
            registry(), POLICY, _request(replace(_spec(), parameters=malformed)))
    assert caught.value.code == "INVALID_REQUEST"

    without_account = replace(_spec(), provider_budget_ref=None)
    with pytest.raises(OpsError) as caught:
        validate_request(registry(), POLICY, _request(without_account))
    assert caught.value.code == "INVALID_REQUEST"

    zero_with_account = replace(
        _spec(), parameters={**_spec().parameters, "provider_calls": 0})
    with pytest.raises(OpsError) as caught:
        validate_request(registry(), POLICY, _request(zero_with_account))
    assert caught.value.code == "INVALID_REQUEST"


def test_real_registry_claim_reserves_exact_provider_budget(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    provider_budget.configure_account(
        conn, "orats-account", "generation-1", remaining=20, live_reserve=5)
    receipt = submit(conn, registry(), POLICY, _request(), clock=clock)
    claim = claim_next(
        conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor,
        clock=clock, registry=registry())
    assert claim is not None
    assert claim.job_id == receipt.job_id
    row = conn.execute(
        "SELECT reserved_calls, used_calls FROM provider_reservations "
        "WHERE account = ? AND released_at IS NULL",
        ("orats-account",),
    ).fetchone()
    assert tuple(row) == (2, 0)


def test_worker_dispatch_invokes_data_callback_and_returns_bounded_manifest(
        tmp_path, monkeypatch):
    observed = {}

    def callback(params, root):
        observed["params"] = params
        document = _evidence(params)
        _write_evidence(root, document)
        return document

    monkeypatch.setattr(
        incremental_data, "_load_data_refresh_callback", lambda: callback)
    result = worker.dispatch("incremental_refresh", _parameters(), tmp_path)

    assert isinstance(observed["params"], incremental_data.RefreshParameters)
    assert result == {
        "outputs": [{
            "name": "incremental_refresh",
            "path": incremental_data.REFRESH_RESULT_PATH,
            "schema": incremental_data.REFRESH_RESULT_SCHEMA,
        }],
        "completed_ids": ["daily-market-2026-09-15"],
        "no_work": False,
    }
    assert set(result) == {"outputs", "completed_ids", "no_work"}


def test_actual_worker_refuses_missing_refresh_input_without_completing_ids(tmp_path):
    with pytest.raises(OpsError) as caught:
        worker.dispatch("incremental_refresh", _parameters(), tmp_path)
    assert caught.value.code == "WORKER_FAILED"

    document = json.loads(
        (tmp_path / incremental_data.REFRESH_RESULT_PATH).read_text())
    assert document["status"] == "failed"
    assert document["completed_ids"] == []
    assert document["coverage_advanced"] is False
    assert document["candidate_snapshot_id"] is None


@pytest.mark.parametrize(
    ("status", "code"),
    [
        ("partial", "TRANSIENT_SOURCE"),
        ("rate_limited", "RATE_LIMITED"),
        ("not_final", "SOURCE_NOT_FINAL"),
        ("credential_invalid", "CREDENTIAL_INVALID"),
        ("transient", "TRANSIENT_SOURCE"),
    ],
)
def test_worker_maps_incomplete_callback_status_to_existing_typed_failure(
        tmp_path, monkeypatch, status, code):
    def callback(params, root):
        document = _evidence(
            params, status=status, coverage_advanced=False,
            candidate_snapshot_id=None, completed_ids=[])
        _write_evidence(root, document)
        return document

    monkeypatch.setattr(
        incremental_data, "_load_data_refresh_callback", lambda: callback)
    with pytest.raises(OpsError) as caught:
        worker.dispatch("incremental_refresh", _parameters(), tmp_path)
    assert caught.value.code == code


def test_partial_callback_cannot_report_a_watermark_advance(tmp_path, monkeypatch):
    def callback(params, root):
        document = _evidence(
            params, status="partial", coverage_advanced=True,
            candidate_snapshot_id=None)
        _write_evidence(root, document)
        return document

    monkeypatch.setattr(
        incremental_data, "_load_data_refresh_callback", lambda: callback)
    with pytest.raises(OpsError) as caught:
        worker.dispatch("incremental_refresh", _parameters(), tmp_path)
    assert caught.value.code == "INTEGRITY_FAILED"


def test_success_receipt_is_bounded(tmp_path, monkeypatch):
    def callback(params, root):
        (root / incremental_data.REFRESH_RESULT_PATH).write_bytes(
            b"x" * (incremental_data.MAX_REFRESH_RESULT_BYTES + 1))
        return _evidence(params)

    monkeypatch.setattr(
        incremental_data, "_load_data_refresh_callback", lambda: callback)
    with pytest.raises(OpsError) as caught:
        worker.dispatch("incremental_refresh", _parameters(), tmp_path)
    assert caught.value.code == "VALIDATION_FAILED"


def test_complete_requires_committed_candidate_snapshot(tmp_path, monkeypatch):
    def callback(params, root):
        document = _evidence(params, candidate_snapshot_id=None)
        _write_evidence(root, document)
        return document

    monkeypatch.setattr(
        incremental_data, "_load_data_refresh_callback", lambda: callback)
    with pytest.raises(OpsError) as caught:
        worker.dispatch("incremental_refresh", _parameters(), tmp_path)
    assert caught.value.code == "VALIDATION_FAILED"


@pytest.mark.parametrize("field", ["parent_snapshot_id", "refresh_plan_hash"])
def test_staged_result_must_bind_to_submitted_parent_and_plan(
        tmp_path, monkeypatch, field):
    def callback(params, root):
        wrong = "snap-other" if field == "parent_snapshot_id" else "sha256:" + "f" * 64
        document = _evidence(params, **{field: wrong})
        _write_evidence(root, document)
        return document

    monkeypatch.setattr(
        incremental_data, "_load_data_refresh_callback", lambda: callback)
    with pytest.raises(OpsError) as caught:
        worker.dispatch("incremental_refresh", _parameters(), tmp_path)
    assert caught.value.code == "STALE_EXPECTATION"


def test_callback_return_must_match_staged_evidence(tmp_path, monkeypatch):
    def callback(params, root):
        staged = _evidence(params)
        _write_evidence(root, staged)
        return {**staged, "candidate_snapshot_id": "snap-different"}

    monkeypatch.setattr(
        incremental_data, "_load_data_refresh_callback", lambda: callback)
    with pytest.raises(OpsError) as caught:
        worker.dispatch("incremental_refresh", _parameters(), tmp_path)
    assert caught.value.code == "INTEGRITY_FAILED"


def test_noop_evidence_cannot_carry_candidate_or_watermark(tmp_path, monkeypatch):
    def callback(params, root):
        document = _evidence(
            params, status="noop", coverage_advanced=False,
            candidate_snapshot_id="snap-illegal")
        _write_evidence(root, document)
        return document

    monkeypatch.setattr(
        incremental_data, "_load_data_refresh_callback", lambda: callback)
    with pytest.raises(OpsError) as caught:
        worker.dispatch("incremental_refresh", _parameters(), tmp_path)
    assert caught.value.code == "INTEGRITY_FAILED"


def test_bound_noop_evidence_returns_no_work_without_advancement(tmp_path, monkeypatch):
    def callback(params, root):
        document = _evidence(
            params, status="noop", coverage_advanced=False,
            candidate_snapshot_id=None)
        _write_evidence(root, document)
        return document

    monkeypatch.setattr(
        incremental_data, "_load_data_refresh_callback", lambda: callback)
    result = worker.dispatch("incremental_refresh", _parameters(), tmp_path)
    assert result["no_work"] is True
    assert result["completed_ids"] == ["daily-market-2026-09-15"]


def test_complete_evidence_must_report_candidate_coverage_advance(tmp_path, monkeypatch):
    def callback(params, root):
        document = _evidence(params, coverage_advanced=False)
        _write_evidence(root, document)
        return document

    monkeypatch.setattr(
        incremental_data, "_load_data_refresh_callback", lambda: callback)
    with pytest.raises(OpsError) as caught:
        worker.dispatch("incremental_refresh", _parameters(), tmp_path)
    assert caught.value.code == "INTEGRITY_FAILED"


@pytest.mark.parametrize("defect", ["unknown_field", "missing_version"])
def test_callback_evidence_document_is_strict(tmp_path, monkeypatch, defect):
    def callback(params, root):
        document = _evidence(params)
        if defect == "unknown_field":
            document["data_candidate"] = {"private": "shape"}
        else:
            del document["schema_version"]
        _write_evidence(root, document)
        return document

    monkeypatch.setattr(
        incremental_data, "_load_data_refresh_callback", lambda: callback)
    with pytest.raises(OpsError) as caught:
        worker.dispatch("incremental_refresh", _parameters(), tmp_path)
    assert caught.value.code == "VALIDATION_FAILED"
