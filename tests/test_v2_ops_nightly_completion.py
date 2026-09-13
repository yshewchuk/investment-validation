"""P2-5/B1a: launch-time resolution of job-ID input bindings.

A worker's ``input_bindings`` may name a parent job's committed output as
``job_<id>#<output_name>``. Resolution happens once, at launch, against the
parent's committed state; the result is recorded in ``attempt_input_bindings``
and the coordinator (``decision_commit.py``) trusts that durable record
instead of re-deriving a ``job_`` binding at commit time. This file is the
real-Service/real-subprocess proof for that path (mirroring
``test_v2_ops_supervised_legacy.py``), plus the negative controls and the
checkpoint cache-identity guarantee.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest

from engine.ledger import build_prediction_rows
from engine.v2.contracts import (
    ArtifactRef,
    JobSpec,
    LegacyFileRef,
    LegacyInputManifest,
    SubmitRequest,
)
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash, to_document
from engine.v2.ledger.decisions import set_authority
from engine.v2.ops import schema
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import cache_identity, register_artifact
from engine.v2.ops.decision_commit import commit_decisions_in_transaction, validated_decision_candidate
from engine.v2.ops.decision_replay import compare_rows, decision_population, population_key
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import environment_identity, file_hash, worker_source_manifest
from engine.v2.ops.input_bindings import record_resolved_bindings, resolve_bindings, resolved_inputs_hash
from engine.v2.ops.legacy_actions import ACTION_NAMES
from engine.v2.ops.legacy_adapter import _action_decision_replay
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.migrations import applied_versions, checksum
from engine.v2.ops.nightly import _legacy_resource, build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, job_id_for, submit, submit_graph
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, sample

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})
SESSION = "2026-09-12"


# --------------------------------------------------------------------------
# shared fixtures (mirrors tests/test_v2_ops_supervised_legacy.py)
# --------------------------------------------------------------------------


def _publish(store, conn, clock, value, schema_ref):
    ref = store.publish_bytes(json.dumps(value, sort_keys=True).encode(), schema_ref=schema_ref)
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


def _manifest_ref(store, conn, clock, file_refs):
    document = to_document(LegacyInputManifest(
        manifest_id="m1", file_refs=tuple(file_refs), table_contract_refs=(),
        registry_and_model_refs=(), calendar_ref=None, selected_session=SESSION,
        finality_receipt_refs=(), knowledge_mode_by_table={}, availability_evidence_refs=(),
        read_set_complete=True, capture_implementation_ref="test.v1"))
    return _publish(store, conn, clock, document, "legacy_input_manifest.v1.0")


def _decision_evidence(score_ref, finality_ref, plan_ref, score, finality, plan):
    expected = plan["expected_population"]
    common = {"schema_version": "decision_receipt.v1.0",
              "score_artifact_id": score_ref.artifact_id,
              "score_content_hash": score_ref.content_hash,
              "finality_artifact_id": finality_ref.artifact_id,
              "finality_content_hash": finality_ref.content_hash,
              "plan_artifact_id": plan_ref.artifact_id,
              "plan_content_hash": plan_ref.content_hash,
              "session": SESSION, "deployment": plan["deployment"],
              "decision_clock": plan["decision_clock"], "expected_population": expected}
    receipts = {kind: dict(common, kind=kind) for kind in (
        "causality", "coverage", "finality", "selection", "replay")}
    receipts["causality"]["observed_cutoffs"] = {expected[0]: score["evidence_cutoff"]}
    receipts["coverage"]["observed_population"] = expected
    receipts["finality"].update(observed_finality_hash=content_hash(finality),
                                 covered_tickers=[score["ticker"]])
    receipts["selection"]["eligible_candidate_keys"] = expected
    receipts["replay"].update(source_rows=[score], replayed_rows=[score],
                               source_rows_hash=content_hash([score]),
                               replayed_rows_hash=content_hash([score]), findings=[])
    return {"schema_version": "decision_evidence.v1.0", "receipts": receipts}


def _score_and_finality():
    score = {"ticker": "FAKE", "event_id": "event-1", "event_date": SESSION,
             "as_of": SESSION, "entry_date": SESSION, "evidence_cutoff": SESSION,
             "strategy": "TWIN-P", "strike": 100.0, "expiry": "2026-10-16",
             "session": "AMC", "snapshot_hash": "sha256:" + "a" * 64}
    finality = {"date": SESSION, "is_final": True, "market_wide": True,
                "daily_share": 1.0, "chain_share": 1.0, "covered": 1}
    return score, finality


def _run_until_terminal(service, conn, job_id, timeout=18):
    deadline = time.monotonic() + timeout
    state = "queued"
    while time.monotonic() < deadline:
        service.tick()
        state = conn.execute("SELECT state FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]
        if state in ("succeeded", "failed", "blocked", "cancelled"):
            return state
        time.sleep(0.05)
    return state


def _succeed_parent(conn, clock, supervisor, *, key, output_name, ref):
    """A genuinely succeeded parent job/attempt, without a live subprocess.

    Drives the real submission + claim + attempt-completion machinery — the
    same ``commit_attempt`` path ``Service._finish`` uses — so the child job
    under test sees a real succeeded parent and a real ``attempt_outputs``
    row. Never mocks the resolver or the validator.
    """
    job = JobSpec(kind="artifact_check", implementation_ref="parent-setup", spec_hash=None,
                  environment_ref="parent-setup", parameters={"expected_ids": ()},
                  output_namespace="shadow", resource_class="delivery",
                  retry_policy_ref="bounded", checkpoint_contract_ref="receipt.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key=key, principal="operator", job=job), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                       supervisor=supervisor, clock=clock, registry=registry())

    def effects(inner_conn):
        register_artifact(inner_conn, ref, claim.attempt_id, clock)
        inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                           (claim.attempt_id, output_name, ref.artifact_id))

    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                   clock=clock, effects=effects)
    return job_id_for("shadow", key)


def _minimal_decisions_job(*, key, manifest_id, score_binding, dependency_job_ids=()):
    """A ``legacy_decisions`` submission whose only live binding is ``score.json``,
    for exercising launch-time resolution failures cheaply (no subprocess runs)."""
    job = JobSpec(
        kind="legacy_decisions",
        implementation_ref=content_hash(worker_source_manifest(REPO)),
        spec_hash=None,
        environment_ref=content_hash(environment_identity(
            profile_named(DEFAULT_POLICY, "validation").thread_count or
            profile_named(DEFAULT_POLICY, "validation").cpu_count)),
        parameters={"expected_ids": ("legacy_decisions",), "session": SESSION,
                    "tickers": (), "year_start": 2024, "year_end": 2026,
                    "input_bindings": {"legacy_manifest.json": manifest_id,
                                       "score.json": score_binding}},
        input_refs=(manifest_id,), dependency_job_ids=dependency_job_ids,
        output_namespace="shadow", resource_class="validation", retry_policy_ref="bounded",
        checkpoint_contract_ref="legacy_action.v1.0")
    return SubmitRequest(namespace="shadow", idempotency_key=key, principal="operator", job=job)


# --------------------------------------------------------------------------
# positive path: real Service, real subprocess, job_-bound parents
# --------------------------------------------------------------------------


def test_legacy_decisions_commits_through_job_id_bindings(tmp_path):
    root = tmp_path / "case1"
    root.mkdir()
    store_root = root / "prod"
    store_root.mkdir()
    fixture = store_root / "unused.txt"
    fixture.write_bytes(b"a legacy read-set member decisions never opens")

    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    store = ArtifactStore(root)
    try:
        setup_epoch = begin_epoch(conn, clock=clock, boot_id="setup", pid=1)
        setup = Supervisor(setup_epoch, "setup")

        manifest_ref = _manifest_ref(store, conn, clock, (LegacyFileRef(
            path="unused.txt", content_hash=file_hash(fixture),
            byte_size=fixture.stat().st_size),))
        score, finality = _score_and_finality()
        plan = {"schema_version": "decision_plan.v1.0", "session": SESSION,
                "deployment": "shadow-deployment", "decision_clock": SESSION + "T21:00:00+00:00",
                "expected_population": ["FAKE|TWIN-P|" + SESSION]}
        score_ref = _publish(store, conn, clock, {"rows": [score]}, "legacy_action.v1.0")
        finality_ref = _publish(store, conn, clock, finality, "legacy_action.v1.0")
        plan_ref = _publish(store, conn, clock, plan, "decision_plan.v1.0")
        evidence = _decision_evidence(score_ref, finality_ref, plan_ref, score, finality, plan)
        evidence_ref = _publish(store, conn, clock, evidence, "decision_evidence.v1.0")

        score_job = _succeed_parent(conn, clock, setup, key="score-parent",
                                    output_name="legacy_score", ref=score_ref)
        finality_job = _succeed_parent(conn, clock, setup, key="finality-parent",
                                       output_name="legacy_finality", ref=finality_ref)
        plan_job = _succeed_parent(conn, clock, setup, key="plan-parent",
                                   output_name="decision_plan", ref=plan_ref)
        evidence_job = _succeed_parent(conn, clock, setup, key="evidence-parent",
                                       output_name="decision_evidence", ref=evidence_ref)

        with transaction(conn):
            set_authority(conn, None, "catalog", SESSION + "T20:00:00.000000Z")

        bindings = {
            "legacy_manifest.json": manifest_ref.artifact_id,
            "score.json": score_job + "#legacy_score",
            "finality.json": finality_job + "#legacy_finality",
            "decision_plan.json": plan_job + "#decision_plan",
            "decision_evidence.json": evidence_job + "#decision_evidence",
        }
        profile = profile_named(DEFAULT_POLICY, "validation")
        job = JobSpec(
            kind="legacy_decisions",
            implementation_ref=content_hash(worker_source_manifest(REPO)),
            spec_hash=None,
            environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
            parameters={"expected_ids": ("legacy_decisions",), "session": SESSION,
                        "tickers": (), "year_start": 2024, "year_end": 2026,
                        "input_bindings": bindings},
            input_refs=(manifest_ref.artifact_id,),
            dependency_job_ids=(score_job, finality_job, plan_job, evidence_job),
            output_namespace="shadow", resource_class="validation", retry_policy_ref="bounded",
            checkpoint_contract_ref="legacy_action.v1.0")
        receipt = submit(conn, registry(), POLICY, SubmitRequest(
            namespace="shadow", idempotency_key="dec-job-bound", principal="operator", job=job),
            clock=clock)

        service = Service(conn, root, registry(), TEST_POLICY, clock=clock,
                          code_source=REPO, store_root=store_root)
        try:
            service.start()
            state = _run_until_terminal(service, conn, receipt.job_id)
        finally:
            service.close()

        row = conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?",
                           (receipt.job_id,)).fetchone()
        assert state == "succeeded", row["failure_json"]
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1

        attempt_id = conn.execute("SELECT attempt_id FROM attempts WHERE job_id=?",
                                  (receipt.job_id,)).fetchone()[0]
        recorded = {r["name"]: r for r in conn.execute(
            "SELECT name, binding, artifact_id, content_hash FROM attempt_input_bindings "
            "WHERE attempt_id=? ORDER BY name", (attempt_id,))}
        assert recorded["score.json"]["artifact_id"] == score_ref.artifact_id
        assert recorded["score.json"]["binding"] == bindings["score.json"]
        assert recorded["finality.json"]["artifact_id"] == finality_ref.artifact_id
        assert recorded["decision_plan.json"]["artifact_id"] == plan_ref.artifact_id
        assert recorded["decision_evidence.json"]["artifact_id"] == evidence_ref.artifact_id
        assert recorded["legacy_manifest.json"]["artifact_id"] == manifest_ref.artifact_id
        assert recorded["legacy_manifest.json"]["binding"] == manifest_ref.artifact_id
    finally:
        conn.close()


# --------------------------------------------------------------------------
# negative controls: launch-time resolution failures (no subprocess runs)
# --------------------------------------------------------------------------


def _run_minimal_and_assert_failed(tmp_path, request, expected_code):
    root = tmp_path
    root.mkdir(exist_ok=True)
    store_root = root / "prod"
    store_root.mkdir(exist_ok=True)
    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    try:
        receipt = submit(conn, registry(), POLICY, request, clock=clock)
        service = Service(conn, root, registry(), TEST_POLICY, clock=clock,
                          code_source=REPO, store_root=store_root)
        try:
            service.start()
            state = _run_until_terminal(service, conn, receipt.job_id, timeout=10)
        finally:
            service.close()
        row = conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?",
                           (receipt.job_id,)).fetchone()
        assert state == "failed", row["failure_json"] if row else state
        assert expected_code in row["failure_json"]
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM attempt_outputs").fetchone()[0] == 0
    finally:
        conn.close()


def test_job_binding_to_undeclared_dependency_fails_input_changed(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    store = ArtifactStore(tmp_path)
    manifest_ref = _manifest_ref(store, conn, clock, ())
    conn.close()
    request = _minimal_decisions_job(
        key="undeclared", manifest_id=manifest_ref.artifact_id,
        score_binding="job_00000000000000000000000000000000#legacy_score",
        dependency_job_ids=())
    _run_minimal_and_assert_failed(tmp_path, request, "INPUT_CHANGED")


def test_job_binding_without_output_name_fails_validation(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    store = ArtifactStore(tmp_path)
    manifest_ref = _manifest_ref(store, conn, clock, ())
    conn.close()
    request = _minimal_decisions_job(
        key="malformed", manifest_id=manifest_ref.artifact_id,
        score_binding="job_00000000000000000000000000000000",
        dependency_job_ids=())
    _run_minimal_and_assert_failed(tmp_path, request, "VALIDATION_FAILED")


def test_direct_artifact_binding_outside_input_refs_fails_input_changed(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    store = ArtifactStore(tmp_path)
    manifest_ref = _manifest_ref(store, conn, clock, ())
    stray_ref = _publish(store, conn, clock, {"rows": []}, "legacy_action.v1.0")
    conn.close()
    request = _minimal_decisions_job(
        key="stray-direct", manifest_id=manifest_ref.artifact_id,
        score_binding=stray_ref.artifact_id, dependency_job_ids=())
    _run_minimal_and_assert_failed(tmp_path, request, "INPUT_CHANGED")


def test_job_binding_parent_not_succeeded_fails_resolution(tmp_path):
    """Unreachable through the live scheduler (a child cannot be claimed while
    a declared dependency has not succeeded — see ``scheduler.claim_next``'s
    readiness query), so this exercises ``resolve_bindings`` directly."""
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    try:
        parent = JobSpec(kind="artifact_check", implementation_ref="x", spec_hash=None,
                         environment_ref="x", parameters={"expected_ids": ()},
                         output_namespace="shadow", resource_class="delivery",
                         retry_policy_ref="bounded", checkpoint_contract_ref="receipt.v1.0")
        submit(conn, registry(), POLICY, SubmitRequest(
            namespace="shadow", idempotency_key="stuck-parent", principal="operator", job=parent),
            clock=clock)
        parent_id = job_id_for("shadow", "stuck-parent")
        assert conn.execute("SELECT state FROM jobs WHERE job_id=?",
                            (parent_id,)).fetchone()[0] == "queued"

        store = ArtifactStore(tmp_path)
        child = JobSpec(kind="legacy_decisions", implementation_ref="x", spec_hash=None,
                        environment_ref="x",
                        parameters={"expected_ids": ("legacy_decisions",), "session": SESSION,
                                    "input_bindings": {"score.json": parent_id + "#legacy_score"}},
                        dependency_job_ids=(parent_id,), output_namespace="shadow",
                        resource_class="validation", retry_policy_ref="bounded",
                        checkpoint_contract_ref="legacy_action.v1.0")
        with pytest.raises(OpsError) as excinfo:
            resolve_bindings(conn, store, child)
        assert excinfo.value.code == "INPUT_CHANGED"
    finally:
        conn.close()


# --------------------------------------------------------------------------
# coordinator tamper defense: recorded binding disagrees with validation
# --------------------------------------------------------------------------


def test_coordinator_refuses_when_recorded_binding_disagrees_with_validation(tmp_path):
    """No DB seam exists to change a recorded row after the fact — update and
    delete are refused by trigger — so this drives the real resolver to
    record real bindings, then asks the coordinator to commit a *different*
    (corrupted) validated context than what got recorded, exactly the
    "validation used X, launch recorded Y" case commit_decisions_in_transaction
    must catch. Neither the resolver nor the validator is mocked."""
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    store = ArtifactStore(tmp_path)
    try:
        score, finality = _score_and_finality()
        plan = {"schema_version": "decision_plan.v1.0", "session": SESSION,
                "deployment": "shadow-deployment", "decision_clock": SESSION + "T21:00:00+00:00",
                "expected_population": ["FAKE|TWIN-P|" + SESSION]}
        score_ref = _publish(store, conn, clock, {"rows": [score]}, "legacy_action.v1.0")
        finality_ref = _publish(store, conn, clock, finality, "legacy_action.v1.0")
        plan_ref = _publish(store, conn, clock, plan, "decision_plan.v1.0")
        evidence = _decision_evidence(score_ref, finality_ref, plan_ref, score, finality, plan)
        evidence_ref = _publish(store, conn, clock, evidence, "decision_evidence.v1.0")
        # Content-addressed, so it must differ from score_ref's bytes to get
        # a genuinely different artifact_id to tamper with.
        decoy_ref = _publish(store, conn, clock, {"rows": [score], "decoy": True}, "legacy_action.v1.0")
        with transaction(conn):
            set_authority(conn, None, "catalog", SESSION + "T20:00:00.000000Z")

        bindings = {"score.json": score_ref.artifact_id, "finality.json": finality_ref.artifact_id,
                    "decision_plan.json": plan_ref.artifact_id,
                    "decision_evidence.json": evidence_ref.artifact_id}
        refs = (score_ref.artifact_id, finality_ref.artifact_id, plan_ref.artifact_id,
                evidence_ref.artifact_id, decoy_ref.artifact_id)
        job = JobSpec(kind="legacy_decisions", implementation_ref="x", spec_hash=None,
                     environment_ref="x",
                     parameters={"expected_ids": ("legacy_decisions",), "session": SESSION,
                                 "input_bindings": bindings},
                     input_refs=refs, output_namespace="shadow", resource_class="validation",
                     retry_policy_ref="bounded", checkpoint_contract_ref="legacy_action.v1.0")
        submit(conn, registry(), POLICY, SubmitRequest(
            namespace="shadow", idempotency_key="tamper", principal="operator", job=job), clock=clock)
        setup_epoch = begin_epoch(conn, clock=clock, boot_id="setup", pid=1)
        claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                           supervisor=Supervisor(setup_epoch, "setup"), clock=clock,
                           registry=registry())

        resolved = resolve_bindings(conn, store, claim.spec)
        with transaction(conn):
            record_resolved_bindings(conn, claim.attempt_id, resolved)

        decision = {"row_id": "FAKE|TWIN-P|" + SESSION, "event_id": "event-1",
                    "ticker": "FAKE", "strategy": "TWIN-P", "event_date": SESSION,
                    "as_of": SESSION, "written_at": plan["decision_clock"],
                    "decision_ts": plan["decision_clock"], "snapshot_hash": score["snapshot_hash"],
                    "score": score, "finality": finality}
        candidate_ref = _publish(store, conn, clock, {"rows": [decision]}, "legacy_action.v1.0")
        candidates, context = validated_decision_candidate(conn, store, claim, candidate_ref)
        assert context["bindings"]["score"]["artifact_id"] == score_ref.artifact_id

        # Corrupt only the coordinator's own snapshot: a validated context
        # that no longer agrees with what launch actually recorded.
        tampered = dict(context)
        tampered["bindings"] = dict(context["bindings"])
        tampered["bindings"]["score"] = dict(context["bindings"]["score"])
        tampered["bindings"]["score"]["artifact_id"] = decoy_ref.artifact_id

        with pytest.raises(OpsError) as excinfo, transaction(conn):
            commit_decisions_in_transaction(conn, claim, candidates, tampered, clock=clock)
        assert excinfo.value.code == "INPUT_CHANGED"
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0

        # The untampered context still commits: the defense is precise, not
        # a blanket refusal of every commit against this attempt.
        with transaction(conn):
            receipts = commit_decisions_in_transaction(conn, claim, candidates, context, clock=clock)
        assert len(receipts) == 1
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    finally:
        conn.close()


# --------------------------------------------------------------------------
# immutability
# --------------------------------------------------------------------------


def test_attempt_input_bindings_are_immutable(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    store = ArtifactStore(tmp_path)
    try:
        parent = JobSpec(kind="artifact_check", implementation_ref="x", spec_hash=None,
                         environment_ref="x", parameters={"expected_ids": ()},
                         output_namespace="shadow", resource_class="delivery",
                         retry_policy_ref="bounded", checkpoint_contract_ref="receipt.v1.0")
        submit(conn, registry(), POLICY, SubmitRequest(
            namespace="shadow", idempotency_key="k", principal="operator", job=parent), clock=clock)
        setup_epoch = begin_epoch(conn, clock=clock, boot_id="setup", pid=1)
        claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                           supervisor=Supervisor(setup_epoch, "setup"), clock=clock,
                           registry=registry())
        ref = store.publish_bytes(b"immutable-fixture", schema_ref="receipt.v1.0")
        with transaction(conn):
            register_artifact(conn, ref, claim.attempt_id, clock)
            conn.execute("INSERT INTO attempt_input_bindings VALUES (?,?,?,?,?)",
                        (claim.attempt_id, "score.json", ref.artifact_id, ref.artifact_id,
                         ref.content_hash))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE attempt_input_bindings SET content_hash=? WHERE attempt_id=?",
                        ("sha256:" + "0" * 64, claim.attempt_id))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM attempt_input_bindings WHERE attempt_id=?",
                        (claim.attempt_id,))
        assert conn.execute("SELECT COUNT(*) FROM attempt_input_bindings").fetchone()[0] == 1
    finally:
        conn.close()


# --------------------------------------------------------------------------
# checkpoint cache identity
# --------------------------------------------------------------------------


def test_cache_identity_changes_when_resolved_parent_output_changes(tmp_path):
    root = tmp_path
    store_root = root / "prod"
    store_root.mkdir()
    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    store = ArtifactStore(root)
    try:
        setup_epoch = begin_epoch(conn, clock=clock, boot_id="setup", pid=1)
        setup = Supervisor(setup_epoch, "setup")
        ref_a = _publish(store, conn, clock, {"variant": "a"}, "receipt.v1.0")
        ref_b = _publish(store, conn, clock, {"variant": "b"}, "receipt.v1.0")
        parent_a = _succeed_parent(conn, clock, setup, key="leaf-a",
                                   output_name="receipt", ref=ref_a)
        parent_b = _succeed_parent(conn, clock, setup, key="leaf-b",
                                   output_name="receipt", ref=ref_b)

        def child_request(key, parent_job_id):
            job = JobSpec(kind="artifact_check",
                         implementation_ref=content_hash(worker_source_manifest(REPO)),
                         spec_hash=None, environment_ref=content_hash(environment_identity(1)),
                         parameters={"expected_ids": (),
                                     "input_bindings": {"parent.json": parent_job_id + "#receipt"}},
                         dependency_job_ids=(parent_job_id,), output_namespace="shadow",
                         resource_class="delivery", retry_policy_ref="bounded",
                         checkpoint_contract_ref="receipt.v1.0")
            return SubmitRequest(namespace="shadow", idempotency_key=key,
                                 principal="operator", job=job)

        req_a = child_request("child-a", parent_a)
        req_b = child_request("child-b", parent_b)
        job_a, job_b = req_a.job, req_b.job
        receipt_a = submit(conn, registry(), POLICY, req_a, clock=clock)
        receipt_b = submit(conn, registry(), POLICY, req_b, clock=clock)

        service = Service(conn, root, registry(), TEST_POLICY, clock=clock,
                          code_source=REPO, store_root=store_root)
        try:
            service.start()
            state_a = _run_until_terminal(service, conn, receipt_a.job_id)
            state_b = _run_until_terminal(service, conn, receipt_b.job_id)
        finally:
            service.close()
        assert state_a == "succeeded" and state_b == "succeeded"

        cache_keys = [row[0] for row in conn.execute(
            "SELECT cache_key FROM checkpoints ORDER BY rowid")]
        assert len(cache_keys) == 2
        assert len(set(cache_keys)) == 2, "distinct parent outputs must not share a checkpoint"

        # Independently recompute each key the same way the supervisor does
        # (resolve the real recorded bindings, then the same cache_identity
        # formula) and confirm it matches what was actually stored.
        key_a = cache_identity(
            kind=job_a.kind, inputs=resolved_inputs_hash(job_a, resolve_bindings(conn, store, job_a)),
            implementation=job_a.implementation_ref, parameters=content_hash(job_a.parameters),
            environment=job_a.environment_ref, schema="receipt.v1.0", shard="default")
        key_b = cache_identity(
            kind=job_b.kind, inputs=resolved_inputs_hash(job_b, resolve_bindings(conn, store, job_b)),
            implementation=job_b.implementation_ref, parameters=content_hash(job_b.parameters),
            environment=job_b.environment_ref, schema="receipt.v1.0", shard="default")
        assert key_a != key_b
        assert {key_a, key_b} == set(cache_keys)

        # The recorded binding for each attempt names its own parent's output
        # artifact — confirming which resolved artifact actually drove each
        # cache key above.
        for job_id, expected_parent_ref in ((receipt_a.job_id, ref_a), (receipt_b.job_id, ref_b)):
            attempt_id = conn.execute("SELECT attempt_id FROM attempts WHERE job_id=?",
                                      (job_id,)).fetchone()[0]
            row = conn.execute(
                "SELECT artifact_id FROM attempt_input_bindings WHERE attempt_id=? AND name=?",
                (attempt_id, "parent.json")).fetchone()
            assert row[0] == expected_parent_ref.artifact_id

        # Each checkpoint was produced by its own attempt: the second run
        # never reused the first's checkpoint.
        producers = [row[0] for row in conn.execute(
            "SELECT DISTINCT producer_attempt_id FROM checkpoints")]
        assert len(producers) == 2
    finally:
        conn.close()


# --------------------------------------------------------------------------
# migration hygiene
# --------------------------------------------------------------------------


def test_attempt_input_bindings_migration_applies_once_and_checksums_hold(tmp_path):
    path = tmp_path / "ops.sqlite"
    conn = open_catalog(path, clock=SystemClock())
    applied = applied_versions(conn, schema.OWNER)
    assert applied.get(7) == checksum(schema.MIGRATIONS[-1])
    assert schema.MIGRATIONS[-1].version == 7
    conn.close()
    # Re-opening the same catalog re-applies no migration and does not raise.
    conn2 = open_catalog(path, clock=SystemClock())
    assert applied_versions(conn2, schema.OWNER) == applied
    conn2.close()


# --------------------------------------------------------------------------
# P2-5/B1b: the supervised decision-replay stage (pure helpers + DAG wiring)
# --------------------------------------------------------------------------
#
# Tier 0 -- no private data, no real scoring. decision_population is checked
# against engine.ledger.build_prediction_rows on a synthetic frame the same
# way tests/test_v2_ops_supervised_legacy.py's own docstring notes:
# build_prediction_rows degrades to an empty earnings_events join when no
# store data is present, so the comparison needs nothing private.
#
# _action_decision_replay's ticker-scoping is exercised in-process with a
# monkeypatched FeatureContext.load/Scorer/score_calendar (never real
# scoring); the DYN-SV replay-through-score_calendar behavior itself needs
# real data and is not re-verified here.

_REPLAY_SYNTHETIC_ROWS = [
    # entry-today: eligible.
    {"ticker": "AAA", "strategy": "TWIN-P", "event_date": "2026-09-12",
     "as_of": SESSION, "session": "AMC", "fill": 0.5, "strike_offset": None},
    # a ladder row on the SAME event/strategy -- build_prediction_rows would
    # record it too (it does not know about ladders); decision_population
    # excludes it, but the two population-KEY sets still agree because the
    # ladder row shares its ATM sibling's (ticker, strategy, event_date).
    {"ticker": "AAA", "strategy": "TWIN-P", "event_date": "2026-09-12",
     "as_of": SESSION, "session": "AMC", "fill": 0.5, "strike_offset": 0.025},
    # forward (future entry): not eligible for SESSION.
    {"ticker": "BBB", "strategy": "TWIN-P", "event_date": "2026-09-15",
     "as_of": "2026-09-14", "session": "BMO", "fill": 0.5, "strike_offset": None},
    # null entry/decision date: not eligible.
    {"ticker": "CCC", "strategy": "TWIN-P", "event_date": "2026-09-16",
     "as_of": None, "entry_date": None, "session": "BMO", "fill": 0.5,
     "strike_offset": None},
]


def test_decision_population_agrees_with_ledger_eligibility_as_a_key_set():
    population = decision_population({"rows": _REPLAY_SYNTHETIC_ROWS}, SESSION)
    observed = {(row["ticker"], row["strategy"], row["event_date"]) for row in population}

    frame = pd.DataFrame(_REPLAY_SYNTHETIC_ROWS)
    ledger_rows = build_prediction_rows(frame, as_of=SESSION, entry_dated_only=True)
    expected = {(row["ticker"], row["strategy"], row["event_date"]) for row in ledger_rows}

    assert observed == expected == {("AAA", "TWIN-P", "2026-09-12")}
    # decision_population additionally drops the ladder row the ledger keeps
    # duplicated under the same key; the ledger's list is longer, the key
    # sets are not.
    assert len(ledger_rows) == 2
    assert len(population) == 1


def test_decision_population_rejects_duplicate_keys():
    rows = [
        {"ticker": "AAA", "strategy": "TWIN-P", "event_date": "2026-09-12",
         "as_of": SESSION, "strike_offset": None},
        {"ticker": "AAA", "strategy": "TWIN-P", "event_date": "2026-09-12",
         "as_of": SESSION, "strike_offset": None},
    ]
    with pytest.raises(OpsError, match="duplicate"):
        decision_population({"rows": rows}, SESSION)


def test_decision_population_is_sorted_by_population_key():
    rows = [
        {"ticker": "ZZZ", "strategy": "TWIN-P", "event_date": "2026-09-12",
         "as_of": SESSION, "strike_offset": None},
        {"ticker": "AAA", "strategy": "TWIN-P", "event_date": "2026-09-12",
         "as_of": SESSION, "strike_offset": None},
    ]
    population = decision_population({"rows": rows}, SESSION)
    assert [row["ticker"] for row in population] == ["AAA", "ZZZ"]


# --------------------------------------------------------------------------
# compare_rows: only row_id/strike_offset are ignored -- replay now runs the
# real score_calendar, so chosen_strategy/chosen_margin/menu_size are
# genuinely recomputed and stay compared like any other field.
# --------------------------------------------------------------------------

_BASE_REPLAY_ROW = {"ticker": "AAA", "strategy": "TWIN-P", "event_date": "2026-09-12",
                    "exp_pnl_model": 0.123456, "gate_pass": True, "row_id": "AAA|TWIN-P|...",
                    "strike_offset": None}


def test_compare_rows_identical_rows_has_no_findings():
    assert compare_rows([dict(_BASE_REPLAY_ROW)], [dict(_BASE_REPLAY_ROW)]) == []


def test_compare_rows_names_a_planted_float_change_and_a_planted_flag_change():
    replayed = dict(_BASE_REPLAY_ROW, exp_pnl_model=0.123457, gate_pass=False)
    findings = compare_rows([_BASE_REPLAY_ROW], [replayed])
    key = "AAA|TWIN-P|2026-09-12"
    assert {"key": key, "field": "exp_pnl_model", "reason": "value_mismatch"} in findings
    assert {"key": key, "field": "gate_pass", "reason": "value_mismatch"} in findings
    assert len(findings) == 2


def test_compare_rows_ignores_only_row_id_and_strike_offset():
    replayed = dict(_BASE_REPLAY_ROW, row_id="a-completely-different-id", strike_offset=0.0)
    assert compare_rows([_BASE_REPLAY_ROW], [replayed]) == []


def test_compare_rows_now_compares_chosen_strategy_and_menu_fields():
    source = dict(_BASE_REPLAY_ROW, strategy="DYN-SV", chosen_strategy="TWIN-P",
                  chosen_margin=0.4, menu_size=3, detail="chose TWIN-P of 3 (...)")
    assert compare_rows([source], [dict(source)]) == []
    replayed = dict(source, chosen_strategy="CTR5", chosen_margin=0.5, menu_size=4)
    findings = compare_rows([source], [replayed])
    key = "AAA|DYN-SV|2026-09-12"
    assert {"key": key, "field": "chosen_strategy", "reason": "value_mismatch"} in findings
    assert {"key": key, "field": "chosen_margin", "reason": "value_mismatch"} in findings
    assert {"key": key, "field": "menu_size", "reason": "value_mismatch"} in findings


def test_compare_rows_names_missing_row_when_an_eligible_key_never_replays():
    findings = compare_rows([_BASE_REPLAY_ROW], [])
    assert findings == [{"key": "AAA|TWIN-P|2026-09-12", "field": "<row>", "reason": "row_missing"}]


# --------------------------------------------------------------------------
# _action_decision_replay: FeatureContext gets the score job's FULL ticker
# set (analog pools / registered champions read off that context); the real
# score_calendar call is scoped to just the eligible rows' own tickers.
# --------------------------------------------------------------------------


def test_decision_replay_action_scopes_context_full_and_scoring_eligible(monkeypatch, tmp_path):
    row_aaa = {"ticker": "AAA", "strategy": "TWIN-P", "event_date": "2026-09-12",
              "as_of": SESSION, "session": "AMC", "fill": 0.5, "strike_offset": None,
              "exp_pnl_model": 0.1}
    row_zzz_forward = {"ticker": "ZZZ", "strategy": "TWIN-P", "event_date": "2026-09-15",
                       "as_of": "2026-09-14", "session": "BMO", "fill": 0.5,
                       "strike_offset": None}
    (tmp_path / "score.json").write_text(json.dumps({"rows": [row_aaa, row_zzz_forward]}))

    calls = {}

    def fake_load(tickers, years):
        calls["context_tickers"] = sorted(tickers)
        return object()

    class _FakeReplayScorer:
        def __init__(self, context):
            self.context = context

    def fake_score_calendar(as_of, *, horizon_days, alt_strikes, scorer, tickers, **kwargs):
        calls["score_tickers"] = sorted(tickers)
        calls["alt_strikes"] = alt_strikes
        return pd.DataFrame([row_aaa])

    import engine.features as features_module
    import engine.score as score_module
    monkeypatch.setattr(features_module.FeatureContext, "load", staticmethod(fake_load))
    monkeypatch.setattr(score_module, "Scorer", _FakeReplayScorer)
    monkeypatch.setattr(score_module, "score_calendar", fake_score_calendar)

    result = _action_decision_replay(
        {"session": SESSION, "tickers": ("AAA", "ZZZ"), "year_start": 2024, "year_end": 2026},
        tmp_path)
    assert result["hash"]

    # context sees every ticker the score job scored; score_calendar itself
    # is only asked for the eligible rows' own tickers.
    assert calls["context_tickers"] == ["AAA", "ZZZ"]
    assert calls["score_tickers"] == ["AAA"]
    assert calls["alt_strikes"] == 0

    document = json.loads((tmp_path / "replay.json").read_text())
    assert document["population"] == ["AAA|TWIN-P|2026-09-12"]
    assert document["findings"] == []
    assert document["source_rows"] == [row_aaa]
    [replayed_row] = document["replayed_rows"]
    assert replayed_row["ticker"] == "AAA" and "row_id" in replayed_row


def test_decision_replay_action_empty_population_skips_scoring(monkeypatch, tmp_path):
    row = {"ticker": "AAA", "strategy": "TWIN-P", "event_date": "2026-09-15",
          "as_of": "2026-09-14", "session": "BMO", "fill": 0.5, "strike_offset": None}
    (tmp_path / "score.json").write_text(json.dumps({"rows": [row]}))

    def boom(*args, **kwargs):
        raise AssertionError("score_calendar must not run for an empty population")

    import engine.features as features_module
    import engine.score as score_module
    monkeypatch.setattr(features_module.FeatureContext, "load", staticmethod(boom))
    monkeypatch.setattr(score_module, "score_calendar", boom)

    _action_decision_replay(
        {"session": SESSION, "tickers": ("AAA",), "year_start": 2024, "year_end": 2026}, tmp_path)
    document = json.loads((tmp_path / "replay.json").read_text())
    assert document["population"] == []
    assert document["source_rows"] == document["replayed_rows"] == []
    assert document["findings"] == []


# --------------------------------------------------------------------------
# DAG: build_legacy_job_requests wires legacy_decision_replay off score
# --------------------------------------------------------------------------


def test_decision_replay_job_depends_on_score_with_named_output_binding():
    plan = build_nightly_plan(str(REPO), SESSION)
    requests = build_legacy_job_requests(plan, tickers=("FAKE",),
                                         year_start=2025, year_end=2026)
    by_kind = {r.job.kind: r for r in requests}
    assert "legacy_decision_replay" in by_kind
    replay = by_kind["legacy_decision_replay"]
    score = by_kind["legacy_score"]
    score_job_id = job_id_for("shadow", score.idempotency_key)
    assert score_job_id in replay.job.dependency_job_ids
    assert replay.job.parameters["input_bindings"]["score.json"] == (
        score_job_id + "#legacy_score")
    assert replay.job.resource_class == "legacy_score"

    conn = open_catalog(Path(tempfile.mkdtemp()) / "ops.sqlite", clock=SystemClock())
    try:
        policy = NamespacePolicy({"operator": frozenset({"shadow"})})
        receipts = submit_graph(conn, registry(), policy, requests, clock=SystemClock())
        assert len(receipts) == len(requests)
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == len(requests)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Registry / environment ref
# --------------------------------------------------------------------------


def test_legacy_decision_replay_kind_is_allowlisted_and_registered():
    assert "legacy_decision_replay" in ACTION_NAMES
    assert "legacy_decision_replay" in registry().names()


def test_legacy_decision_replay_environment_ref_matches_launch_formula():
    """A3-style check (see test_v2_ops_legacy_defects.py): the env ref the DAG
    request carries for this kind must equal what ``_launch`` computes from
    the SAME resource class -- ``legacy_score``, per ``_legacy_resource``."""
    assert _legacy_resource("legacy_decision_replay") == "legacy_score"
    plan = build_nightly_plan(str(REPO), SESSION)
    requests = build_legacy_job_requests(plan, tickers=("FAKE",),
                                         year_start=2025, year_end=2026)
    replay = next(r for r in requests if r.job.kind == "legacy_decision_replay")
    profile = profile_named(DEFAULT_POLICY, "legacy_score")
    thread_count = profile.thread_count or profile.cpu_count
    assert replay.job.environment_ref == content_hash(environment_identity(thread_count))
