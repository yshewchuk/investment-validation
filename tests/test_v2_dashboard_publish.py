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
        projection = store.publish_bytes(json.dumps({"projection_release_id": "projection-a"}).encode(), schema_ref="projection_binding.v1.0")
        register_artifact(conn, projection, None, clock)
        projection_b = store.publish_bytes(json.dumps({"projection_release_id": "projection-b"}).encode(), schema_ref="projection_binding.v1.0")
        register_artifact(conn, projection_b, None, clock)
        from tools.v2_dashboard_publish import main as publish_main
        log = tmp_path / "publication-log.json"
        assert publish_main(["--root", str(tmp_path), "--catalog", str(tmp_path / "ops.sqlite"),
                             "--store-root", str(FAKE_STORE_ROOT), "--source-publication-job", source.job_id,
                             "--projection-binding", projection.artifact_id, "--operation-id", "service-a",
                             "--verify-sequence", "--b-source-publication-job", source.job_id,
                             "--b-projection-binding", projection_b.artifact_id, "--sequence-log", str(log)]) == 0
        from checks.rearchitecture_phase3_publish import build_fenced_rollback
        rollback, negative = build_fenced_rollback(log, tmp_path / "releases" / scope, tmp_path / "failure",
            catalog_path=tmp_path / "ops.sqlite", store_root=tmp_path, code_hash="test", environment_hash="test")
        assert rollback.resulting_snapshot_id == "projection-a" and negative.verdict == "differ"
        document = json.loads(log.read_text())
        original_log = log.read_text()
        document["update"][0]["publication_job_id"] = source.job_id
        log.write_text(json.dumps(document))
        with pytest.raises(RuntimeError, match="does not deterministically produce release"):
            build_fenced_rollback(log, tmp_path / "releases" / scope, tmp_path / "failure",
                catalog_path=tmp_path / "ops.sqlite", store_root=tmp_path, code_hash="test", environment_hash="test")
        log.write_text(original_log)
        first_id, second_id = document["update"][0]["ops_release_id"], document["update"][1]["ops_release_id"]
        original_manifest = conn.execute("SELECT manifest_json FROM releases WHERE release_id=?", (first_id,)).fetchone()[0]
        second_manifest = json.loads(conn.execute("SELECT manifest_json FROM releases WHERE release_id=?", (second_id,)).fetchone()[0])
        tampered = json.loads(original_manifest)
        tampered["gates"]["decision"] = second_manifest["gates"]["decision"]
        conn.execute("UPDATE releases SET manifest_json=? WHERE release_id=?", (json.dumps(tampered), first_id))
        with pytest.raises(RuntimeError, match="gate input hash"):
            build_fenced_rollback(log, tmp_path / "releases" / scope, tmp_path / "failure",
                catalog_path=tmp_path / "ops.sqlite", store_root=tmp_path, code_hash="test", environment_hash="test")
        conn.execute("UPDATE releases SET manifest_json=? WHERE release_id=?", (original_manifest, first_id))
        stale = submit_retained_publication(conn, store, registry=registry(),
            policy=NamespacePolicy({"operator": frozenset({"shadow", "smoke"})}), clock=clock,
            source_job_id=source.job_id, projection_binding_ref=projection.artifact_id,
            operation_id="cancelled-fence")
        stale_claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor,
                                 clock=clock, registry=registry())
        from engine.v2.ops.input_bindings import resolve_and_record
        resolve_and_record(conn, store, stale_claim)
        from engine.v2.ops.lifecycle import request_cancel
        request_cancel(conn, stale.job_id, stale_claim.attempt_id, clock=clock)
        from engine.v2.ops.effects_graph import publication_effect
        from engine.v2.ops.errors import OpsError
        current_before = release_current(tmp_path / "releases" / scope)
        with pytest.raises(OpsError, match="LEASE_LOST"):
            publication_effect(conn, store, stale_claim, tmp_path, REPO, clock=clock, store_root=FAKE_STORE_ROOT)
        assert release_current(tmp_path / "releases" / scope) == current_before
    finally:
        conn.close()
