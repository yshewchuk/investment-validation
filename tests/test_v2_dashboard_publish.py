"""Retained-input publication submission stays inside the normal job lifecycle."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from engine.v2.foundation import ArtifactStore
from engine.v2.ops.errors import OpsError
from engine.v2.ops.input_bindings import ResolvedBinding, record_resolved_bindings
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.catalog import transaction
from engine.v2.ops.publication_submit import submit_retained_publication
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, get_job
from tests.test_v2_ops_effects_graph import _params as _ops_params, _submit_and_claim
from tests.ops_support import TEST_POLICY, catalog, sample
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.supervisor import Service
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.foundation import content_hash
from engine.v2.ops.publication import current as release_current


def _source(conn, store, clock, supervisor):
    refs = {}
    for name, doc in {
            "bundle.tar": b"bundle",
            "finality.json": json.dumps({"date": "2026-09-14", "is_final": True}).encode(),
            "selfcheck.json": json.dumps({"ok": True}).encode(),
            "engineering_gate.json": json.dumps({"ok": True}).encode(),
    }.items():
        ref = store.publish_bytes(doc, schema_ref="test.v1")
        from engine.v2.ops.checkpoints import register_artifact
        register_artifact(conn, ref, None, clock)
        refs[name] = ref.artifact_id
    claim = _submit_and_claim(conn, clock, supervisor, kind="publication", key="source",
                              parameters=_ops_params("publication", "2026-09-14", "shadow",
                                                     input_bindings=refs))
    with transaction(conn):
        record_resolved_bindings(conn, claim.attempt_id, {
            name: ResolvedBinding(name=name, binding=ref, artifact_id=ref, content_hash="sha256:test")
            for name, ref in refs.items()})
    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead"), clock=clock)
    return claim.job_id, refs


def test_retained_submit_is_fresh_idempotent_and_does_not_mutate_source(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    try:
        source, source_refs = _source(conn, store, clock, supervisor)
        projection = store.publish_bytes(b"projection-a", schema_ref="projection_binding.v1.0")
        from engine.v2.ops.checkpoints import register_artifact
        register_artifact(conn, projection, None, clock)
        policy = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})
        first = submit_retained_publication(conn, store, registry=registry(), policy=policy, clock=clock,
                                            source_job_id=source, projection_binding_ref=projection.artifact_id,
                                            operation_id="publish-a")
        retry = submit_retained_publication(conn, store, registry=registry(), policy=policy, clock=clock,
                                            source_job_id=source, projection_binding_ref=projection.artifact_id,
                                            operation_id="publish-a")
        assert first.job_id == retry.job_id and first.job_id != source
        assert get_job(conn, source).state == "succeeded"
        row = conn.execute("SELECT COUNT(*) FROM attempts WHERE job_id=?", (source,)).fetchone()
        assert row[0] == 1
        fresh = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor,
                           clock=clock, registry=registry())
        assert fresh.job_id == first.job_id and fresh.fence == 1
        from engine.v2.ops.input_bindings import resolve_bindings
        resolved = resolve_bindings(conn, store, fresh.spec)
        assert resolved["projection_binding.json"].artifact_id == projection.artifact_id
        assert resolved["bundle.tar"].artifact_id == source_refs["bundle.tar"]
        assert "publication_operation.json" in resolved
    finally:
        conn.close()


def test_retained_submit_refuses_unsucceeded_source(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    try:
        claim = _submit_and_claim(conn, clock, supervisor, kind="publication", key="unfinished",
                                  parameters=_ops_params("publication", "2026-09-14", "shadow"))
        projection = store.publish_bytes(b"projection", schema_ref="projection_binding.v1.0")
        from engine.v2.ops.checkpoints import register_artifact
        register_artifact(conn, projection, None, clock)
        with pytest.raises(OpsError) as raised:
            submit_retained_publication(conn, store, registry=registry(),
                                        policy=NamespacePolicy({"operator": frozenset({"shadow", "smoke"})}),
                                        clock=clock, source_job_id=claim.job_id,
                                        projection_binding_ref=projection.artifact_id, operation_id="bad")
        assert raised.value.problem.code == "INPUT_CHANGED"
    finally:
        conn.close()


def test_retained_publication_runs_through_real_service(tmp_path):
    from tests.test_v2_ops_effects_graph import FAKE_STORE_ROOT, REPO, _bundle_tar, _seed_decisions
    from engine.v2.contracts import JobSpec, SubmitRequest
    from engine.v2.ops.checkpoints import register_artifact
    from engine.v2.ops.lifecycle import Outcome, commit_attempt

    conn, clock, supervisor = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    try:
        scope, session = "shadow", "2026-09-14"
        _seed_decisions(conn, clock, scope, session, predictions=[])
        docs = {"bundle.tar": _bundle_tar(),
                "finality.json": json.dumps({"date": session, "is_final": True, "market_wide": True,
                                              "daily_share": 1.0, "chain_share": 1.0}).encode(),
                "selfcheck.json": json.dumps({"ok": True}).encode(),
                "engineering_gate.json": json.dumps({"ok": True}).encode()}
        refs = {}
        for name, payload in docs.items():
            ref = store.publish_bytes(payload, schema_ref="test.v1")
            register_artifact(conn, ref, None, clock)
            refs[name] = ref.artifact_id
        implementation = content_hash(worker_source_manifest(REPO))
        environment = content_hash(environment_identity(1))
        source_job = JobSpec(kind="publication", implementation_ref=implementation, spec_hash=None,
            environment_ref=environment, parameters=_ops_params("publication", session, scope,
            input_bindings=refs), input_refs=tuple(refs.values()), output_namespace="shadow",
            resource_class="delivery", retry_policy_ref="bounded", checkpoint_contract_ref="effect_receipt.v1.0")
        source = __import__("engine.v2.ops.submission", fromlist=["submit"]).submit(
            conn, registry(), NamespacePolicy({"operator": frozenset({"shadow", "smoke"})}),
            SubmitRequest(namespace="shadow", idempotency_key="service-source", principal="operator",
                          job=source_job), clock=clock)
        source_claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor,
                                  clock=clock, registry=registry())
        from engine.v2.ops.input_bindings import resolve_and_record
        resolve_and_record(conn, store, source_claim)
        commit_attempt(conn, source_claim.attempt_id, source_claim.fence, Outcome(True, "verified_dead"), clock=clock)
        projection = store.publish_bytes(b"projection-service", schema_ref="projection_binding.v1.0")
        register_artifact(conn, projection, None, clock)
        receipt = submit_retained_publication(conn, store, registry=registry(),
            policy=NamespacePolicy({"operator": frozenset({"shadow", "smoke"})}), clock=clock,
            source_job_id=source.job_id, projection_binding_ref=projection.artifact_id,
            operation_id="service-a")
        service = Service(conn, tmp_path, registry(), TEST_POLICY, clock=clock, code_source=REPO,
                          store_root=FAKE_STORE_ROOT)
        service.start()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            service.tick()
            state = get_job(conn, receipt.job_id).state
            if state in ("succeeded", "failed"):
                break
            time.sleep(0.05)
        assert get_job(conn, receipt.job_id).state == "succeeded"
        attempt = conn.execute("SELECT fence,state FROM attempts WHERE job_id=?", (receipt.job_id,)).fetchone()
        assert attempt["fence"] == 1 and attempt["state"] == "succeeded"
        assert release_current(tmp_path / "releases" / scope) is not None
    finally:
        conn.close()
