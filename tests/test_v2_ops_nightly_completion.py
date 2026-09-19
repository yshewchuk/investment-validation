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

import base64
import json
import sqlite3
import tempfile
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
from engine.v2.foundation import ArtifactStore, SystemClock, artifact_reference, content_hash, to_document
from engine.v2.ledger import decisions as ledger_decisions
from engine.v2.ledger.decisions import DecisionConflict, set_authority
from engine.v2.ops import schema
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import cache_identity, register_artifact
from engine.v2.ops.decision_commit import (
    _contract_mismatch_field,
    _match_same_session,
    _require_legacy_exit_finality,
    _require_v2_finality_session,
    _settlement_line,
    _stamp_date,
    _validate_settlement_state,
    commit_decisions,
    commit_decisions_in_transaction,
    validate_candidates,
    validated_decision_candidate,
)
from engine.v2.ops.decision_evidence import derive
from engine.v2.ops.decision_replay import compare_rows, decision_population, population_key
from engine.v2.ops.decision_validation import _validate_causality, validate
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import environment_identity, file_hash, worker_source_manifest
from engine.v2.ops.input_bindings import (
    record_resolved_bindings,
    resolve_and_record,
    resolve_bindings,
    resolved_inputs_hash,
)
from engine.v2.ops.legacy_actions import ACTION_NAMES
from engine.v2.ops.legacy_adapter import (
    _action_decision_replay,
    _json_stdout,
    _load_action_frame,
    _load_finality,
    _load_score_document,
    copy_read_set,
    invoke_nightly_helper,
    legacy_action,
    legacy_ledger_schema_version,
    manifest_files,
    overlay_read_set,
    run_legacy_rebuild,
    run_legacy_script,
)
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.migrations import applied_versions, checksum
from engine.v2.ops.nightly import _legacy_resource, build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, job_id_for, submit, submit_graph
from engine.v2.ops.supervisor import Service, _verify_decision_evidence
from tests.ops_support import TEST_POLICY, run_until, sample

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


def _finality_coverage(covered_tickers=("FAKE",), *, date=SESSION):
    return {"schema_version": "finality_coverage.v1.0", "date": date,
            "covered_tickers": list(covered_tickers)}


def _run_until_terminal(service, conn, job_id, timeout=18):
    return run_until(service, conn, job_id, timeout=timeout)


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
    # guide §5.5 item 2 (engineering observation retry counts) added
    # migration 9; this tracks whichever migration is actually latest rather
    # than a hardcoded version, so the NEXT schema change updates only the
    # migration table.
    assert applied.get(schema.MIGRATIONS[-1].version) == checksum(schema.MIGRATIONS[-1])
    assert schema.MIGRATIONS[-1].version == 9
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
    (tmp_path / "finality.json").write_text(json.dumps(_score_and_finality()[1]))

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
    (tmp_path / "finality.json").write_text(json.dumps(_score_and_finality()[1]))

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


# --------------------------------------------------------------------------
# P2-5/B1c: decision_evidence — a coordinator-validated stage that derives
# decision_plan.v1.0/decision_evidence.v1.0 from committed artifacts, wired
# into the nightly job DAG so legacy_decisions finally has something to bind.
# Everything below is synthetic (D18); no real legacy score/finality/replay
# run happens anywhere in this section.
# --------------------------------------------------------------------------


# -- pure: derive() and validate() agree on a synthetic score document -----


def test_derive_and_validator_agree_on_eligible_forward_and_ladder_rows():
    """One eligible row, one forward row (future as_of), one ladder row
    (strike_offset set). ``derive()``'s population must equal the ONE key
    both ``decision_population`` (strike_offset-aware) and
    ``build_prediction_rows`` (date-aware, ladder-blind) agree is eligible —
    the forward and ladder rows are excluded from ELIGIBILITY by two
    different mechanisms, but land on the same net answer, so the two
    independent computations never disagree about what got decided.
    """
    eligible = {"ticker": "AAA", "strategy": "TWIN-P", "event_date": SESSION, "as_of": SESSION,
                "entry_date": SESSION, "evidence_cutoff": SESSION, "strike": 100.0,
                "expiry": "2026-10-16", "session": "AMC", "snapshot_hash": "sha256:" + "a" * 64,
                "strike_offset": None, "event_id": "evt-aaa"}
    forward = {"ticker": "BBB", "strategy": "TWIN-P", "event_date": "2026-09-20",
              "as_of": "2026-09-19", "entry_date": "2026-09-19", "evidence_cutoff": "2026-09-19",
              "strike": 50.0, "expiry": "2026-10-17", "session": "BMO",
              "snapshot_hash": "sha256:" + "b" * 64, "strike_offset": None, "event_id": "evt-bbb"}
    ladder = {"ticker": "CCC", "strategy": "TWIN-P", "event_date": "2026-09-22",
             "as_of": "2026-09-21", "entry_date": "2026-09-21", "evidence_cutoff": "2026-09-21",
             "strike": 75.0, "expiry": "2026-10-18", "session": "BMO",
             "snapshot_hash": "sha256:" + "c" * 64, "strike_offset": 0.025, "event_id": "evt-ccc"}
    score_doc = {"rows": [eligible, forward, ladder]}
    finality = {"date": SESSION, "is_final": True, "market_wide": True,
                "daily_share": 1.0, "chain_share": 1.0, "covered": 3}
    key = "AAA|TWIN-P|" + SESSION
    replay_doc = {"schema_version": "decision_replay.v1.0", "session": SESSION,
                  "population": [key], "source_rows": [eligible], "replayed_rows": [eligible],
                  "source_rows_hash": content_hash([eligible]),
                  "replayed_rows_hash": content_hash([eligible]), "findings": []}
    coverage_doc = {"schema_version": "finality_coverage.v1.0", "date": SESSION,
                    "covered_tickers": ["AAA", "BBB", "CCC"]}
    score_ref = artifact_reference(b"synthetic-score-bytes", "legacy_action.v1.0")
    finality_ref = artifact_reference(b"synthetic-finality-bytes", "legacy_action.v1.0")
    deployment, decision_clock = "shadow:synthetic-impl", SESSION + "T21:00:00+00:00"

    plan_bytes, evidence_bytes = derive(score_doc, score_ref, finality, finality_ref, replay_doc,
                                        coverage_doc, requested_session=SESSION, deployment=deployment,
                                        decision_clock=decision_clock)
    plan = json.loads(plan_bytes)
    evidence = json.loads(evidence_bytes)
    assert plan["expected_population"] == [key]

    frame = pd.DataFrame([eligible, forward, ladder])
    rows = build_prediction_rows(frame, as_of=SESSION, decision_ts=plan["decision_clock"],
                                 finality=finality, entry_dated_only=True)
    # legacy_adapter._action_decisions's own post-processing: the clock is
    # patched onto every candidate after build_prediction_rows returns it.
    for row in rows:
        row["written_at"] = plan["decision_clock"]
        row["decision_ts"] = plan["decision_clock"]
        if not row.get("event_id"):
            row["event_id"] = (row.get("score") or {}).get("event_id")
    assert len(rows) == 1

    plan_ref = artifact_reference(plan_bytes, "decision_plan.v1.0")
    evidence_ref = artifact_reference(evidence_bytes, "decision_evidence.v1.0")
    context = validate(rows, score=score_doc, finality=finality, plan=plan, evidence=evidence,
                       bindings={"score": {"artifact_id": score_ref.artifact_id,
                                          "content_hash": score_ref.content_hash},
                                "finality": {"artifact_id": finality_ref.artifact_id,
                                            "content_hash": finality_ref.content_hash},
                                "plan": {"artifact_id": plan_ref.artifact_id,
                                        "content_hash": plan_ref.content_hash},
                                "evidence": {"artifact_id": evidence_ref.artifact_id,
                                            "content_hash": evidence_ref.content_hash}})
    assert context["session"] == SESSION
    assert context["deployment"] == deployment


# -- shared scaffolding for the supervised (real Service) tests below ------


def _succeed_finality_parent(conn, clock, supervisor, *, key, finality_ref, coverage_ref):
    """A succeeded ``legacy_finality`` parent job/attempt with BOTH outputs
    registered on the SAME attempt -- matching ``_action_finality``'s real
    shape (finality.json plus the finality_coverage.json review fix)."""
    job = JobSpec(kind="artifact_check", implementation_ref="parent-setup", spec_hash=None,
                  environment_ref="parent-setup", parameters={"expected_ids": ()},
                  output_namespace="shadow", resource_class="delivery",
                  retry_policy_ref="bounded", checkpoint_contract_ref="receipt.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key=key, principal="operator", job=job), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                       supervisor=supervisor, clock=clock, registry=registry())

    def effects(inner_conn):
        register_artifact(inner_conn, finality_ref, claim.attempt_id, clock)
        register_artifact(inner_conn, coverage_ref, claim.attempt_id, clock)
        inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                           (claim.attempt_id, "legacy_finality", finality_ref.artifact_id))
        inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                           (claim.attempt_id, "legacy_finality_coverage", coverage_ref.artifact_id))

    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                   clock=clock, effects=effects)
    return job_id_for("shadow", key)


def test_finality_checkpoint_records_real_names_through_the_real_supervisor_path(tmp_path):
    """Gap-closer (2026-09-14 checkpoint-naming defect): every other test in
    this file that needs a succeeded ``legacy_finality`` parent goes through
    ``_succeed_finality_parent`` above, which hand-inserts ``attempt_outputs``
    rows already named ``legacy_finality``/``legacy_finality_coverage`` --
    it never calls ``Service._checkpoint_refs``, so this whole file passed
    the entire time the real method enumerated names as '0'/'1'. This test
    drives a real ``Service`` (real catalog, real ``ArtifactStore``, real
    ``_checkpoint_refs``) instead, staging both outputs itself and letting
    the supervisor name them -- the same real method the production
    ``legacy_finality`` worker result flows through in ``_commit_success``.
    """
    root = tmp_path / "finality-checkpoint-real"
    root.mkdir()
    store_root = root / "prod"
    store_root.mkdir()
    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    service = Service(conn, root, registry(), TEST_POLICY, clock=clock,
                      code_source=REPO, store_root=store_root)
    service.start()

    job = JobSpec(kind="artifact_check", implementation_ref="parent-real", spec_hash=None,
                  environment_ref="parent-real", parameters={"expected_ids": ()},
                  output_namespace="shadow", resource_class="delivery",
                  retry_policy_ref="bounded", checkpoint_contract_ref="receipt.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key="finality-real", principal="operator", job=job),
        clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                       supervisor=service.identity, clock=clock, registry=registry())
    finality_job_id = job_id_for("shadow", "finality-real")

    staging = service.store.staging_dir(claim.attempt_id)
    finality_bytes = json.dumps({"date": SESSION, "is_final": True}, sort_keys=True).encode()
    coverage_bytes = json.dumps(_finality_coverage(), sort_keys=True).encode()
    (staging / "finality.json").write_bytes(finality_bytes)
    (staging / "finality_coverage.json").write_bytes(coverage_bytes)
    outputs = [{"name": "legacy_finality", "path": "finality.json", "schema": "legacy_action.v1.0"},
              {"name": "legacy_finality_coverage", "path": "finality_coverage.json",
               "schema": "finality_coverage.v1.0"}]

    refs = service._checkpoint_refs(claim, outputs, None)
    assert [name for name, _ in refs] == ["legacy_finality", "legacy_finality_coverage"]

    def effects(inner_conn):
        for name, ref in refs:
            register_artifact(inner_conn, ref, claim.attempt_id, clock)
            inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                               (claim.attempt_id, name, ref.artifact_id))
    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                   clock=clock, effects=effects)

    child_job = JobSpec(kind="artifact_check", implementation_ref="child-real", spec_hash=None,
                        environment_ref="child-real",
                        parameters={"expected_ids": (),
                                    "input_bindings": {
                                        "finality.json": finality_job_id + "#legacy_finality",
                                        "finality_coverage.json":
                                            finality_job_id + "#legacy_finality_coverage"}},
                        dependency_job_ids=(finality_job_id,),
                        output_namespace="shadow", resource_class="delivery",
                        retry_policy_ref="bounded", checkpoint_contract_ref="receipt.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key="score-real", principal="operator", job=child_job),
        clock=clock)
    child_claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                             supervisor=service.identity, clock=clock, registry=registry())

    # Before the fix: OpsError(INPUT_CHANGED, "input binding names an output
    # its parent did not produce") -- the parent's rows were '0'/'1'.
    resolved = resolve_and_record(conn, service.store, child_claim)
    assert resolved["finality.json"].content_hash == refs[0][1].content_hash
    assert resolved["finality_coverage.json"].content_hash == refs[1][1].content_hash


def _decision_evidence_request(*, key, score_job, finality_job, replay_job,
                               deployment, decision_clock):
    bindings = {"score.json": score_job + "#legacy_score",
                "finality.json": finality_job + "#legacy_finality",
                "replay.json": replay_job + "#legacy_decision_replay",
                "finality_coverage.json": finality_job + "#legacy_finality_coverage"}
    profile = profile_named(DEFAULT_POLICY, "validation")
    job = JobSpec(
        kind="decision_evidence",
        implementation_ref=content_hash(worker_source_manifest(REPO)),
        spec_hash=None,
        environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
        parameters={"expected_ids": ("decision_evidence",), "session": SESSION,
                    "tickers": (), "year_start": 2024, "year_end": 2026,
                    "deployment": deployment, "decision_clock": decision_clock,
                    "input_bindings": bindings},
        input_refs=(), dependency_job_ids=(score_job, finality_job, replay_job),
        output_namespace="shadow", resource_class="validation", retry_policy_ref="bounded",
        checkpoint_contract_ref="decision_evidence_pair.v1.0")
    return SubmitRequest(namespace="shadow", idempotency_key=key, principal="operator", job=job)


def _legacy_decisions_request(*, key, manifest_ref, score_job, finality_job, evidence_job):
    bindings = {"legacy_manifest.json": manifest_ref.artifact_id,
                "score.json": score_job + "#legacy_score",
                "finality.json": finality_job + "#legacy_finality",
                "decision_plan.json": evidence_job + "#decision_plan",
                "decision_evidence.json": evidence_job + "#decision_evidence"}
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
        dependency_job_ids=(evidence_job, score_job, finality_job),
        output_namespace="shadow", resource_class="validation", retry_policy_ref="bounded",
        checkpoint_contract_ref="legacy_action.v1.0")
    return SubmitRequest(namespace="shadow", idempotency_key=key, principal="operator", job=job)


# -- supervised positive path: real Service, real subprocess workers -------


def test_decision_evidence_and_decisions_commit_through_submit_graph(tmp_path):
    root = tmp_path / "evidence-positive"
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
        key = "FAKE|TWIN-P|" + SESSION
        score_ref = _publish(store, conn, clock, {"rows": [score]}, "legacy_action.v1.0")
        finality_ref = _publish(store, conn, clock, finality, "legacy_action.v1.0")
        coverage_ref = _publish(store, conn, clock, _finality_coverage(), "finality_coverage.v1.0")
        replay_doc = {"schema_version": "decision_replay.v1.0", "session": SESSION,
                      "population": [key], "source_rows": [score], "replayed_rows": [score],
                      "source_rows_hash": content_hash([score]),
                      "replayed_rows_hash": content_hash([score]), "findings": []}
        replay_ref = _publish(store, conn, clock, replay_doc, "legacy_action.v1.0")

        score_job = _succeed_parent(conn, clock, setup, key="pos-score",
                                    output_name="legacy_score", ref=score_ref)
        finality_job = _succeed_finality_parent(conn, clock, setup, key="pos-finality",
                                                finality_ref=finality_ref, coverage_ref=coverage_ref)
        replay_job = _succeed_parent(conn, clock, setup, key="pos-replay",
                                     output_name="legacy_decision_replay", ref=replay_ref)

        with transaction(conn):
            set_authority(conn, None, "catalog", SESSION + "T20:00:00.000000Z")

        deployment, decision_clock = "shadow:test-impl", SESSION + "T21:00:00+00:00"
        evidence_request = _decision_evidence_request(
            key="pos-evidence", score_job=score_job, finality_job=finality_job,
            replay_job=replay_job, deployment=deployment, decision_clock=decision_clock)
        evidence_job_id = job_id_for("shadow", "pos-evidence")
        decisions_request = _legacy_decisions_request(
            key="pos-decisions", manifest_ref=manifest_ref, score_job=score_job,
            finality_job=finality_job, evidence_job=evidence_job_id)

        receipts = submit_graph(conn, registry(), POLICY,
                                [evidence_request, decisions_request], clock=clock)
        assert len(receipts) == 2

        service = Service(conn, root, registry(), TEST_POLICY, clock=clock,
                          code_source=REPO, store_root=store_root)
        try:
            service.start()
            evidence_state = _run_until_terminal(service, conn, evidence_job_id)
            decisions_job_id = job_id_for("shadow", "pos-decisions")
            decisions_state = _run_until_terminal(service, conn, decisions_job_id)
        finally:
            service.close()

        evidence_row = conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?",
                                    (evidence_job_id,)).fetchone()
        assert evidence_state == "succeeded", evidence_row["failure_json"]
        decisions_row = conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?",
                                     (decisions_job_id,)).fetchone()
        assert decisions_state == "succeeded", decisions_row["failure_json"]

        evidence_attempt = conn.execute("SELECT attempt_id FROM attempts WHERE job_id=?",
                                        (evidence_job_id,)).fetchone()[0]
        evidence_outputs = {r[0] for r in conn.execute(
            "SELECT name FROM attempt_outputs WHERE attempt_id=?", (evidence_attempt,))}
        assert evidence_outputs == {"decision_plan", "decision_evidence"}

        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
        assert sorted(r[0] for r in conn.execute("SELECT kind FROM outbox")) == [
            "export", "release_intent"]

        # Resubmitting the identical graph is idempotent: the same job
        # receipts come back and no new decision is recorded.
        resubmitted = submit_graph(conn, registry(), POLICY,
                                   [evidence_request, decisions_request], clock=clock)
        assert [r.job_id for r in resubmitted] == [r.job_id for r in receipts]
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    finally:
        conn.close()


# -- supervised negative paths: planted defects, each refused ---------------


def _succeed_evidence_parent(conn, clock, supervisor, *, key, plan_ref, evidence_ref):
    """A succeeded ``decision_evidence`` parent job/attempt with BOTH outputs
    registered on the SAME attempt -- matching the real worker's shape
    (one job, two outputs), unlike ``_succeed_parent`` above which seeds one
    output per job."""
    job = JobSpec(kind="artifact_check", implementation_ref="parent-setup", spec_hash=None,
                  environment_ref="parent-setup", parameters={"expected_ids": ()},
                  output_namespace="shadow", resource_class="delivery",
                  retry_policy_ref="bounded", checkpoint_contract_ref="receipt.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key=key, principal="operator", job=job), clock=clock)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                       supervisor=supervisor, clock=clock, registry=registry())

    def effects(inner_conn):
        register_artifact(inner_conn, plan_ref, claim.attempt_id, clock)
        register_artifact(inner_conn, evidence_ref, claim.attempt_id, clock)
        inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                           (claim.attempt_id, "decision_plan", plan_ref.artifact_id))
        inner_conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                           (claim.attempt_id, "decision_evidence", evidence_ref.artifact_id))

    commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                   clock=clock, effects=effects)
    return job_id_for("shadow", key)


def _run_decisions_with_derived_evidence(tmp_path, tag, score_doc, finality, replay_doc, *,
                                         coverage_doc=None, deployment="shadow:test-impl",
                                         decision_clock=None):
    """Derive plan/evidence with :func:`derive` (possibly over a tampered
    score/replay/coverage input), seed succeeded score/finality/plan/evidence
    parents, submit ``legacy_decisions`` for real, run it to completion, and
    return ``(state, failure_json, decisions_count, outbox_count)``.
    """
    decision_clock = decision_clock or (SESSION + "T21:00:00+00:00")
    coverage_doc = coverage_doc if coverage_doc is not None else _finality_coverage()
    root = tmp_path / tag
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
        score_ref = _publish(store, conn, clock, score_doc, "legacy_action.v1.0")
        finality_ref = _publish(store, conn, clock, finality, "legacy_action.v1.0")
        coverage_ref = _publish(store, conn, clock, coverage_doc, "finality_coverage.v1.0")

        plan_bytes, evidence_bytes = derive(score_doc, score_ref, finality, finality_ref, replay_doc,
                                            coverage_doc, requested_session=SESSION, deployment=deployment,
                                            decision_clock=decision_clock)
        plan_ref = store.publish_bytes(plan_bytes, schema_ref="decision_plan.v1.0")
        evidence_ref = store.publish_bytes(evidence_bytes, schema_ref="decision_evidence.v1.0")

        score_job = _succeed_parent(conn, clock, setup, key=tag + "-score",
                                    output_name="legacy_score", ref=score_ref)
        finality_job = _succeed_finality_parent(conn, clock, setup, key=tag + "-finality",
                                                finality_ref=finality_ref, coverage_ref=coverage_ref)
        evidence_job = _succeed_evidence_parent(conn, clock, setup, key=tag + "-evidence",
                                                plan_ref=plan_ref, evidence_ref=evidence_ref)

        with transaction(conn):
            set_authority(conn, None, "catalog", SESSION + "T20:00:00.000000Z")

        decisions_request = _legacy_decisions_request(
            key=tag + "-decisions", manifest_ref=manifest_ref, score_job=score_job,
            finality_job=finality_job, evidence_job=evidence_job)
        receipt = submit(conn, registry(), POLICY, decisions_request, clock=clock)

        service = Service(conn, root, registry(), TEST_POLICY, clock=clock,
                          code_source=REPO, store_root=store_root)
        try:
            service.start()
            state = _run_until_terminal(service, conn, receipt.job_id)
        finally:
            service.close()

        row = conn.execute("SELECT failure_json FROM jobs WHERE job_id=?",
                           (receipt.job_id,)).fetchone()
        decisions = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
        outbox = conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]
        return state, row["failure_json"], decisions, outbox
    finally:
        conn.close()


def test_planted_defect_replayed_float_change_is_refused(tmp_path):
    score, finality = _score_and_finality()
    score = dict(score, exp_pnl_model=0.1)
    score_doc = {"rows": [score]}
    tampered = dict(score, exp_pnl_model=0.2)
    key = "FAKE|TWIN-P|" + SESSION
    replay_doc = {"schema_version": "decision_replay.v1.0", "session": SESSION,
                  "population": [key], "source_rows": [score], "replayed_rows": [tampered],
                  "source_rows_hash": content_hash([score]),
                  "replayed_rows_hash": content_hash([tampered]), "findings": []}
    state, failure, decisions, outbox = _run_decisions_with_derived_evidence(
        tmp_path, "float", score_doc, finality, replay_doc)
    assert state == "failed", failure
    assert "VALIDATION_FAILED" in failure
    assert decisions == 0
    assert outbox == 0


def test_planted_defect_late_evidence_cutoff_is_refused(tmp_path):
    score, finality = _score_and_finality()
    # evidence_cutoff after as_of (SESSION, midnight UTC): decision_validation
    # refuses this regardless of what the receipt claims -- see
    # decision_validation._validate_causality.
    score = dict(score, evidence_cutoff=SESSION + "T23:59:59+00:00")
    score_doc = {"rows": [score]}
    key = "FAKE|TWIN-P|" + SESSION
    replay_doc = {"schema_version": "decision_replay.v1.0", "session": SESSION,
                  "population": [key], "source_rows": [score], "replayed_rows": [score],
                  "source_rows_hash": content_hash([score]),
                  "replayed_rows_hash": content_hash([score]), "findings": []}
    state, failure, decisions, outbox = _run_decisions_with_derived_evidence(
        tmp_path, "cutoff", score_doc, finality, replay_doc)
    assert state == "failed", failure
    assert "VALIDATION_FAILED" in failure
    assert decisions == 0
    assert outbox == 0


def test_planted_defect_replay_missing_an_eligible_row_is_refused(tmp_path):
    score, finality = _score_and_finality()
    score_doc = {"rows": [score]}
    key = "FAKE|TWIN-P|" + SESSION
    replay_doc = {"schema_version": "decision_replay.v1.0", "session": SESSION,
                  "population": [key], "source_rows": [score], "replayed_rows": [],
                  "source_rows_hash": content_hash([score]),
                  "replayed_rows_hash": content_hash([]), "findings": []}
    state, failure, decisions, outbox = _run_decisions_with_derived_evidence(
        tmp_path, "missing", score_doc, finality, replay_doc)
    assert state == "failed", failure
    assert "VALIDATION_FAILED" in failure
    assert decisions == 0
    assert outbox == 0


def test_planted_defect_finality_coverage_missing_a_ticker_is_refused(tmp_path):
    """A finality_coverage.json that never names FAKE -- the ``covered_tickers``
    subset check in ``decision_validation._validate_finality_receipt`` refuses
    the whole candidate set, same as before, but now fed from the real
    coverage document rather than the score rows."""
    score, finality = _score_and_finality()
    score_doc = {"rows": [score]}
    key = "FAKE|TWIN-P|" + SESSION
    replay_doc = {"schema_version": "decision_replay.v1.0", "session": SESSION,
                  "population": [key], "source_rows": [score], "replayed_rows": [score],
                  "source_rows_hash": content_hash([score]),
                  "replayed_rows_hash": content_hash([score]), "findings": []}
    coverage_doc = _finality_coverage(covered_tickers=())
    state, failure, decisions, outbox = _run_decisions_with_derived_evidence(
        tmp_path, "coverage-missing", score_doc, finality, replay_doc, coverage_doc=coverage_doc)
    assert state == "failed", failure
    assert "VALIDATION_FAILED" in failure
    assert decisions == 0
    assert outbox == 0


def test_finality_coverage_document_for_a_different_session_is_refused():
    """``derive()`` itself refuses a coverage document dated for another
    session -- a pure, in-process check, not something that needs a real
    subprocess to observe."""
    score, finality = _score_and_finality()
    score_doc = {"rows": [score]}
    key = "FAKE|TWIN-P|" + SESSION
    replay_doc = {"schema_version": "decision_replay.v1.0", "session": SESSION,
                  "population": [key], "source_rows": [score], "replayed_rows": [score],
                  "source_rows_hash": content_hash([score]),
                  "replayed_rows_hash": content_hash([score]), "findings": []}
    coverage_doc = _finality_coverage(date="2026-09-11")
    score_ref = artifact_reference(b"synthetic-score-bytes", "legacy_action.v1.0")
    finality_ref = artifact_reference(b"synthetic-finality-bytes", "legacy_action.v1.0")
    with pytest.raises(OpsError, match="VALIDATION_FAILED"):
        derive(score_doc, score_ref, finality, finality_ref, replay_doc, coverage_doc,
              requested_session=SESSION, deployment="shadow:test-impl",
              decision_clock=SESSION + "T21:00:00+00:00")


def test_action_finality_writes_a_coverage_output_from_monkeypatched_frames(monkeypatch, tmp_path):
    """In-process, no real market data: ``_action_finality`` must write a
    SECOND output, finality_coverage.json, whose per-ticker list is computed
    for real (through ``engine.data.finality.covered_tickers``) and differs
    from the requested tickers when the underlying frames say so -- proving
    this is a genuine per-ticker test, not an echo of the request.
    """
    import pandas as pd

    from engine.data import finality as finality_module
    from engine.v2.ops.legacy_adapter import _action_finality

    class _FixedResult:
        date = SESSION

        def as_dict(self):
            return {"date": SESSION, "is_final": True, "market_wide": True,
                    "daily_share": 1.0, "chain_share": 1.0, "covered": 2, "tickers": 3,
                    "detail": "final"}

    daily = pd.DataFrame({"ticker": ["AAA", "BBB"], "date": [SESSION, SESSION]})
    # BBB's chain observation is stale (not at SESSION); CCC is never carried.
    chains = pd.DataFrame({"ticker": ["AAA", "BBB"], "obs_date": [SESSION, "2026-09-01"]})

    monkeypatch.setattr(finality_module, "resolve_final_session", lambda *a, **k: _FixedResult())
    monkeypatch.setattr(finality_module, "_market_wide_complete", lambda stamp: True)
    monkeypatch.setattr(finality_module, "_coverage_frame",
                        lambda table, column, stamp: daily if table == "daily_market" else chains)
    import engine.calendar as calendar_module
    monkeypatch.setattr(calendar_module, "trading_calendar", lambda extend_days=400: object())

    result = _action_finality({"session": SESSION, "tickers": ("AAA", "BBB", "CCC")}, tmp_path)
    assert result["path"] == "finality.json"
    assert (tmp_path / "finality.json").is_file()
    coverage = json.loads((tmp_path / "finality_coverage.json").read_text())
    assert coverage == {"schema_version": "finality_coverage.v1.0", "date": SESSION,
                        "covered_tickers": ["AAA"]}
    assert coverage["covered_tickers"] != sorted(("AAA", "BBB", "CCC"))


def test_coordinator_rederivation_catches_tampered_worker_evidence_bytes(tmp_path):
    """Even if a worker's own output looks plausible, the coordinator
    independently re-derives from the recorded bindings and refuses a
    ``decision_evidence`` attempt whose published bytes disagree."""
    root = tmp_path / "tamper"
    root.mkdir()
    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    store = ArtifactStore(root)
    try:
        setup_epoch = begin_epoch(conn, clock=clock, boot_id="setup", pid=1)
        setup = Supervisor(setup_epoch, "setup")

        score, finality = _score_and_finality()
        key = "FAKE|TWIN-P|" + SESSION
        score_ref = _publish(store, conn, clock, {"rows": [score]}, "legacy_action.v1.0")
        finality_ref = _publish(store, conn, clock, finality, "legacy_action.v1.0")
        coverage_ref = _publish(store, conn, clock, _finality_coverage(), "finality_coverage.v1.0")
        replay_doc = {"schema_version": "decision_replay.v1.0", "session": SESSION,
                      "population": [key], "source_rows": [score], "replayed_rows": [score],
                      "source_rows_hash": content_hash([score]),
                      "replayed_rows_hash": content_hash([score]), "findings": []}
        replay_ref = _publish(store, conn, clock, replay_doc, "legacy_action.v1.0")

        score_job = _succeed_parent(conn, clock, setup, key="tp-score",
                                    output_name="legacy_score", ref=score_ref)
        finality_job = _succeed_finality_parent(conn, clock, setup, key="tp-finality",
                                                finality_ref=finality_ref, coverage_ref=coverage_ref)
        replay_job = _succeed_parent(conn, clock, setup, key="tp-replay",
                                     output_name="legacy_decision_replay", ref=replay_ref)

        request = _decision_evidence_request(
            key="tp-evidence", score_job=score_job, finality_job=finality_job,
            replay_job=replay_job, deployment="shadow:test-impl",
            decision_clock=SESSION + "T21:00:00+00:00")
        submit(conn, registry(), POLICY, request, clock=clock)
        claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                           supervisor=setup, clock=clock, registry=registry())
        assert claim is not None
        resolve_and_record(conn, store, claim)

        # A worker that published something other than what re-derivation
        # from the SAME recorded bindings would produce.
        wrong_plan = store.publish_bytes(
            json.dumps({"schema_version": "decision_plan.v1.0", "tampered": True}).encode(),
            schema_ref="decision_plan.v1.0")
        wrong_evidence = store.publish_bytes(
            json.dumps({"schema_version": "decision_evidence.v1.0", "receipts": {}}).encode(),
            schema_ref="decision_evidence.v1.0")
        refs = [("decision_plan", wrong_plan), ("decision_evidence", wrong_evidence)]

        with pytest.raises(OpsError, match="VALIDATION_FAILED"):
            _verify_decision_evidence(conn, store, claim, refs)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# review fix #2: a no-entry night must succeed with zero decisions -- v1
# renders every night, and projection depends on decision_commit.
# --------------------------------------------------------------------------


def test_no_entry_night_commits_zero_decisions_and_still_enqueues_release(tmp_path):
    """No score row is entry-dated for SESSION; the plan's expected_population
    is genuinely empty, and legacy_decisions must SUCCEED with zero
    decisions while still enqueueing export/release_intent -- the board
    still renders on a no-entry night."""
    forward = {"ticker": "FAKE", "strategy": "TWIN-P", "event_date": "2026-09-15",
              "as_of": "2026-09-14", "entry_date": "2026-09-14", "evidence_cutoff": "2026-09-14",
              "strike": 100.0, "expiry": "2026-10-16", "session": "AMC",
              "snapshot_hash": "sha256:" + "a" * 64}
    score_doc = {"rows": [forward]}
    finality = {"date": SESSION, "is_final": True, "market_wide": True,
                "daily_share": 1.0, "chain_share": 1.0, "covered": 1}
    replay_doc = {"schema_version": "decision_replay.v1.0", "session": SESSION,
                  "population": [], "source_rows": [], "replayed_rows": [],
                  "source_rows_hash": content_hash([]), "replayed_rows_hash": content_hash([]),
                  "findings": []}
    coverage_doc = _finality_coverage(covered_tickers=("FAKE",))
    state, failure, decisions, outbox = _run_decisions_with_derived_evidence(
        tmp_path, "no-entry", score_doc, finality, replay_doc, coverage_doc=coverage_doc)
    assert state == "succeeded", failure
    assert decisions == 0
    assert outbox == 2


def test_unscored_row_with_no_evidence_window_does_not_block_decisions(tmp_path):
    """Real shadow nightly attempt 14: 16 of 93 score rows (tickers CBRL/FDS/
    LEN/SCHL) were ones ``engine/score.py`` never scored at all --
    ``as_of``/``entry_date``/``exit_date``/``evidence_cutoff``/``quote_date``
    all ``None``, flagged ``UNVALIDATED_STRUCTURE``. ``build_prediction_rows``
    (``entry_dated_only=True``) already excludes any row with no ``as_of`` /
    ``decision_date`` / ``entry_date`` from the recorded population -- legacy
    never turns one into a prediction. ``decision_validation._validate_causality``
    used to scan every row in ``score.json`` unconditionally, including these,
    and refuse each one for having no cutoff to bind (there being no evidence
    window at all) -- 16 spurious ``unbound_or_late_cutoff`` findings that
    blocked a night with real, valid decisions alongside them. A row with no
    ``as_of`` must be skipped, not refused; the genuinely eligible row must
    still commit.
    """
    score, finality = _score_and_finality()
    # Same column set as `score` (real score.json rows all share one uniform
    # schema, None-filled where a row was never scored) -- a differing key
    # set would make pandas pad `score`'s OWN reconstructed record with
    # spurious NaN columns when legacy_decisions rebuilds it from the frame,
    # which is an artifact of this test's DataFrame round trip, not
    # anything the real bug is about.
    unscored = dict.fromkeys(score, None)
    unscored.update(ticker="FAKE", event_id="event-2", event_date="2026-09-16",
                    strategy="CAL-P")
    score_doc = {"rows": [score, unscored]}
    key = "FAKE|TWIN-P|" + SESSION
    replay_doc = {"schema_version": "decision_replay.v1.0", "session": SESSION,
                  "population": [key], "source_rows": [score], "replayed_rows": [score],
                  "source_rows_hash": content_hash([score]),
                  "replayed_rows_hash": content_hash([score]), "findings": []}
    state, failure, decisions, outbox = _run_decisions_with_derived_evidence(
        tmp_path, "unscored-row", score_doc, finality, replay_doc)
    assert state == "succeeded", failure
    assert decisions == 1
    assert outbox == 2


def test_validate_causality_unit_cases():
    """Direct coverage of ``_validate_causality``'s per-row rule, isolated
    from plan/finality/receipt-binding plumbing: an unscored row (no
    ``as_of``, real shadow nightly attempt 14) is skipped entirely, while a
    scored row (``as_of`` set) is still refused for a missing, late or
    mismatched cutoff -- the fix must not weaken that half of the check."""
    scored = {"as_of": SESSION, "evidence_cutoff": SESSION}
    unscored = {"as_of": None, "evidence_cutoff": None}
    good_causal = {"observed_cutoffs": {"scored": SESSION, "unscored": None}}

    findings = []
    _validate_causality(good_causal, {"scored": scored, "unscored": unscored}, findings)
    assert findings == []

    missing = []
    _validate_causality({"observed_cutoffs": {"unscored": None}},
                        {"scored": scored, "unscored": unscored}, missing)
    assert [f["field"] for f in missing] == ["evidence.causality.scored"]

    late = []
    late_row = {"as_of": SESSION, "evidence_cutoff": SESSION + "T23:59:59+00:00"}
    _validate_causality({"observed_cutoffs": {"scored": SESSION + "T23:59:59+00:00"}},
                        {"scored": late_row}, late)
    assert [f["field"] for f in late] == ["evidence.causality.scored"]

    mismatched = []
    _validate_causality({"observed_cutoffs": {"scored": "2026-01-01"}},
                        {"scored": scored}, mismatched)
    assert [f["field"] for f in mismatched] == ["evidence.causality.scored"]


def _empty_plan_and_evidence(score_doc, finality, coverage_doc):
    """A hand-built (never through ``derive()``) plan/evidence pair claiming
    an empty population, for the two adversarial pure tests below -- neither
    scenario is something an honest ``derive()`` call would ever produce."""
    plan = {"schema_version": "decision_plan.v1.0", "session": SESSION,
            "deployment": "shadow:test-impl", "decision_clock": SESSION + "T21:00:00+00:00",
            "expected_population": []}
    score_ref = artifact_reference(json.dumps(score_doc, sort_keys=True).encode(), "legacy_action.v1.0")
    finality_ref = artifact_reference(json.dumps(finality, sort_keys=True).encode(), "legacy_action.v1.0")
    plan_ref = artifact_reference(json.dumps(plan, sort_keys=True).encode(), "decision_plan.v1.0")
    common = {"schema_version": "decision_receipt.v1.0",
              "score_artifact_id": score_ref.artifact_id, "score_content_hash": score_ref.content_hash,
              "finality_artifact_id": finality_ref.artifact_id,
              "finality_content_hash": finality_ref.content_hash,
              "plan_artifact_id": plan_ref.artifact_id, "plan_content_hash": plan_ref.content_hash,
              "session": SESSION, "deployment": plan["deployment"],
              "decision_clock": plan["decision_clock"], "expected_population": []}
    receipts = {kind: dict(common, kind=kind) for kind in
               ("causality", "coverage", "finality", "selection", "replay")}
    receipts["causality"]["observed_cutoffs"] = {population_key(row): row.get("evidence_cutoff")
                                                 for row in score_doc["rows"]}
    receipts["coverage"]["observed_population"] = []
    receipts["finality"].update(observed_finality_hash=content_hash(finality),
                                covered_tickers=coverage_doc["covered_tickers"])
    receipts["selection"]["eligible_candidate_keys"] = []
    receipts["replay"].update(source_rows=[], replayed_rows=[], source_rows_hash=content_hash([]),
                              replayed_rows_hash=content_hash([]), findings=[])
    evidence = {"schema_version": "decision_evidence.v1.0", "receipts": receipts}
    bindings = {"score": {"artifact_id": score_ref.artifact_id, "content_hash": score_ref.content_hash},
               "finality": {"artifact_id": finality_ref.artifact_id,
                           "content_hash": finality_ref.content_hash},
               "plan": {"artifact_id": plan_ref.artifact_id, "content_hash": plan_ref.content_hash},
               "evidence": {"artifact_id": "art_synthetic_evidence", "content_hash": "sha256:" + "0" * 64}}
    return plan, evidence, bindings


def test_empty_plan_population_with_eligible_score_rows_is_refused():
    """The plan and its (hand-built) evidence agree on zero eligible rows,
    but the bound score document actually has one for SESSION -- the
    validator's own recomputation (not a trust of the evidence receipt)
    catches the understated plan."""
    score, finality = _score_and_finality()
    score_doc = {"rows": [score]}
    coverage_doc = _finality_coverage()
    plan, evidence, bindings = _empty_plan_and_evidence(score_doc, finality, coverage_doc)
    with pytest.raises(OpsError, match="VALIDATION_FAILED") as excinfo:
        validate([], score=score_doc, finality=finality, plan=plan, evidence=evidence, bindings=bindings)
    findings = excinfo.value.problem.details["findings"]
    assert any(item["reason"] == "eligible_rows_exist" for item in findings)


def test_empty_population_with_one_candidate_is_refused():
    """The plan and score genuinely agree on zero eligible rows, but a
    candidate is presented anyway -- refused by the existing population-
    mismatch check, unchanged by review fix #2."""
    forward = {"ticker": "FAKE", "strategy": "TWIN-P", "event_date": "2026-09-20",
              "as_of": "2026-09-19", "entry_date": "2026-09-19", "evidence_cutoff": "2026-09-19",
              "strike": 100.0, "expiry": "2026-10-16", "session": "AMC",
              "snapshot_hash": "sha256:" + "a" * 64}
    score_doc = {"rows": [forward]}
    finality = {"date": SESSION, "is_final": True, "market_wide": True,
                "daily_share": 1.0, "chain_share": 1.0, "covered": 1}
    coverage_doc = _finality_coverage()
    plan, evidence, bindings = _empty_plan_and_evidence(score_doc, finality, coverage_doc)
    stray = {"row_id": "stray-1", "event_id": "evt-1", "ticker": "FAKE", "strategy": "TWIN-P",
            "event_date": SESSION, "as_of": SESSION, "written_at": plan["decision_clock"],
            "decision_ts": plan["decision_clock"], "score": {}, "snapshot_hash": "sha256:" + "b" * 64,
            "finality": finality}
    with pytest.raises(OpsError, match="VALIDATION_FAILED") as excinfo:
        validate([stray], score=score_doc, finality=finality, plan=plan, evidence=evidence,
                bindings=bindings)
    findings = excinfo.value.problem.details["findings"]
    assert any(item["reason"] == "expected_population_mismatch" for item in findings)


# --------------------------------------------------------------------------
# DAG shape: decision_evidence wired between decision_replay and
# decision_commit; settlement stays independent of both.
# --------------------------------------------------------------------------


def test_decision_evidence_stage_wired_with_parents_and_bindings():
    plan = build_nightly_plan(str(REPO), SESSION)
    requests = build_legacy_job_requests(plan, tickers=("FAKE",),
                                         year_start=2025, year_end=2026)
    assert [r.job.kind for r in requests] == [
        "legacy_finality", "legacy_score", "legacy_decision_replay", "decision_evidence",
        "legacy_decisions", "legacy_settlement", "legacy_model_evidence", "ledger_export",
        "engineering_gate", "legacy_render", "legacy_selfcheck", "publication", "backup"]
    by_kind = {r.job.kind: r for r in requests}

    evidence = by_kind["decision_evidence"]
    score_id = job_id_for("shadow", by_kind["legacy_score"].idempotency_key)
    finality_id = job_id_for("shadow", by_kind["legacy_finality"].idempotency_key)
    replay_id = job_id_for("shadow", by_kind["legacy_decision_replay"].idempotency_key)
    assert set(evidence.job.dependency_job_ids) == {score_id, finality_id, replay_id}
    assert evidence.job.parameters["input_bindings"] == {
        "score.json": score_id + "#legacy_score", "finality.json": finality_id + "#legacy_finality",
        "replay.json": replay_id + "#legacy_decision_replay",
        "finality_coverage.json": finality_id + "#legacy_finality_coverage"}
    assert evidence.job.parameters["deployment"] == "shadow:" + plan["implementation_ref"]
    assert evidence.job.parameters["decision_clock"] == plan["decision_clock"]

    decisions = by_kind["legacy_decisions"]
    evidence_id = job_id_for("shadow", evidence.idempotency_key)
    assert set(decisions.job.dependency_job_ids) == {evidence_id, score_id, finality_id}
    assert decisions.job.parameters["input_bindings"]["decision_plan.json"] == (
        evidence_id + "#decision_plan")
    assert decisions.job.parameters["input_bindings"]["decision_evidence.json"] == (
        evidence_id + "#decision_evidence")

    settlement = by_kind["legacy_settlement"]
    assert settlement.job.dependency_job_ids == (finality_id,)
    assert evidence_id not in settlement.job.dependency_job_ids
    assert job_id_for("shadow", decisions.idempotency_key) not in settlement.job.dependency_job_ids

    conn = open_catalog(Path(tempfile.mkdtemp()) / "ops.sqlite", clock=SystemClock())
    try:
        receipts = submit_graph(conn, registry(), NamespacePolicy({"operator": frozenset({"shadow"})}),
                                requests, clock=SystemClock())
        assert len(receipts) == len(requests)
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == len(requests)
    finally:
        conn.close()


def test_plan_nightly_pins_decision_clock_and_resubmission_reuses_it(tmp_path, capsys):
    from engine.v2.ops import cli

    root = tmp_path / "ops"
    population_file = tmp_path / "population.json"
    population_file.write_text(json.dumps(["FAKE|TWIN-P|" + SESSION]))
    manifest_file = tmp_path / "manifest.json"
    from engine.v2.data.legacy_nightly_read_plan import NIGHTLY_CAPTURE_IMPLEMENTATION_REF
    manifest_file.write_text(json.dumps({
        "manifest_id": "m1",
        "file_refs": [{"path": "data/curated/daily_market/year=2024/part-0000.parquet",
                      "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
                     {"path": "data/curated/option_chains/year=2024/part-0000.parquet",
                      "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
                     {"path": "data/curated/earnings_events/year=2024/part-0000.parquet",
                      "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
                     {"path": "data/curated/trades/year=2024/part-0000.parquet",
                      "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
                     {"path": "data/raw/fetch/orats/ab/placeholder.meta.json",
                      "content_hash": "sha256:" + "0" * 64, "byte_size": 1}],
        "registry_and_model_refs": ["placeholder::sha256:" + "0" * 64],
        "calendar_ref": "placeholder::sha256:" + "0" * 64,
        "capture_implementation_ref": NIGHTLY_CAPTURE_IMPLEMENTATION_REF}))
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(root), "plan", "nightly", "--as-of", SESSION,
                     "--tickers", "FAKE", "--input-manifest", str(manifest_file),
                     "--expected-population", str(population_file)]) == 0
    plan_doc = json.loads(capsys.readouterr().out)
    decision_clock = plan_doc["plan"]["decision_clock"]
    assert decision_clock
    plan_ref = plan_doc["plan_ref"]

    assert cli.main(["--root", str(root), "submit", "--plan", plan_ref,
                     "--idempotency-key", "s1"]) == 0
    capsys.readouterr()
    # A second submission of the SAME plan artifact is the "retry" case: the
    # decision_evidence job it names is unchanged (job ids are keyed off the
    # plan's own session/scope, not the CLI's --idempotency-key), so its
    # decision_clock parameter is read back off the ORIGINAL submission.
    assert cli.main(["--root", str(root), "submit", "--plan", plan_ref,
                     "--idempotency-key", "s2"]) == 0
    capsys.readouterr()

    raw = sqlite3.connect(root / "catalog.sqlite")
    try:
        rows = raw.execute("SELECT spec_json FROM jobs WHERE kind='decision_evidence'").fetchall()
        assert len(rows) == 1
        assert json.loads(rows[0][0])["parameters"]["decision_clock"] == decision_clock
    finally:
        raw.close()


# ==========================================================================
# Coverage ratchet fixes (2026-09-15): engine.v2.ledger.decisions is D18's
# own decision ledger (this file already imports set_authority from it);
# engine.v2.ops.decision_commit's settlement-import path
# (import_settlement_candidates_in_transaction and everything it calls) and
# commit_decisions_in_transaction's own validation guards were entirely
# untested under the Phase 2 fixed suite even though the module is already
# in scope here. engine.v2.ops.legacy_adapter's filesystem-safety helpers
# are exercised the same way. Direct/white-box calls into private helpers
# are the established style in this file already (``_validate_causality``,
# ``_action_decision_replay``, ``_legacy_resource`` above).
# ==========================================================================


# --------------------------------------------------------------------------
# engine.v2.ledger.decisions: schema install, authority, insert, imports
# --------------------------------------------------------------------------


def test_ledger_decisions_install_creates_tables_standalone():
    """``install`` -- unlike every other test in this file, which reaches the
    same three tables through ``open_catalog``'s migration path -- is never
    itself called anywhere else."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        ledger_decisions.install(conn)
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"decisions", "decision_imports", "decision_authority"} <= names
    finally:
        conn.close()


def test_set_authority_requires_transaction_and_rejects_wrong_owner(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    try:
        with pytest.raises(ValueError):
            set_authority(conn, None, "catalog", "2026-09-12T20:00:00Z")
        with transaction(conn):
            set_authority(conn, None, "catalog", "2026-09-12T20:00:00Z")
        with pytest.raises(DecisionConflict):
            with transaction(conn):
                set_authority(conn, "someone-else", "catalog-2", "2026-09-12T20:01:00Z")
        # Real owner switch: current owner "catalog" hands off to "catalog-2".
        with transaction(conn):
            set_authority(conn, "catalog", "catalog-2", "2026-09-12T20:02:00Z")
        row = conn.execute("SELECT owner, generation FROM decision_authority WHERE singleton=1").fetchone()
        assert row["owner"] == "catalog-2"
        assert row["generation"] == 2
    finally:
        conn.close()


def test_ledger_decisions_insert_conflict_paths(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    try:
        with pytest.raises(ValueError):
            ledger_decisions.insert(conn, logical_key="k1", decision_id="prediction:r1",
                                    payload={"row_id": "r1"}, purpose="shadow", kind="prediction",
                                    validations=[], created_at="2026-09-12T20:00:00Z")
        with transaction(conn):
            set_authority(conn, None, "catalog", "2026-09-12T20:00:00Z")
        # Authority mismatch: no caller currently holds "wrong-owner".
        with pytest.raises(DecisionConflict):
            with transaction(conn):
                ledger_decisions.insert(conn, logical_key="k1", decision_id="prediction:r1",
                                        payload={"row_id": "r1"}, purpose="shadow", kind="prediction",
                                        validations=[], created_at="2026-09-12T20:00:00Z",
                                        owner="wrong-owner")
        # Supersession without a reason.
        with pytest.raises(DecisionConflict):
            with transaction(conn):
                ledger_decisions.insert(conn, logical_key="k1", decision_id="prediction:r1",
                                        payload={"row_id": "r1"}, purpose="shadow", kind="prediction",
                                        validations=[], created_at="2026-09-12T20:00:00Z",
                                        supersedes="prediction:r0")
        with transaction(conn):
            first = ledger_decisions.insert(conn, logical_key="k1", decision_id="prediction:r1",
                                            payload={"row_id": "r1", "v": 1}, purpose="shadow",
                                            kind="prediction", validations=[],
                                            created_at="2026-09-12T20:00:00Z")
        assert first["decision_id"] == "prediction:r1"
        # Byte-identical retry (same logical_key, decision_id, payload_hash,
        # purpose, kind, supersedes) is a no-op that returns the same row.
        with transaction(conn):
            again = ledger_decisions.insert(conn, logical_key="k1", decision_id="prediction:r1",
                                            payload={"row_id": "r1", "v": 1}, purpose="shadow",
                                            kind="prediction", validations=[],
                                            created_at="2026-09-12T20:05:00Z")
        assert again["sequence"] == first["sequence"]
        # Same identity, different content: IDEMPOTENCY_CONFLICT.
        with pytest.raises(DecisionConflict):
            with transaction(conn):
                ledger_decisions.insert(conn, logical_key="k1", decision_id="prediction:r1",
                                        payload={"row_id": "r1", "v": 2}, purpose="shadow",
                                        kind="prediction", validations=[],
                                        created_at="2026-09-12T20:06:00Z")
    finally:
        conn.close()


def test_ledger_decisions_rows_filters_by_kind_and_through(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    try:
        with transaction(conn):
            set_authority(conn, None, "catalog", "2026-09-12T20:00:00Z")
            ledger_decisions.insert(conn, logical_key="k1", decision_id="prediction:r1",
                                    payload={"row_id": "r1"}, purpose="shadow", kind="prediction",
                                    validations=[], created_at="2026-09-12T20:00:00Z")
            ledger_decisions.insert(conn, logical_key="k2", decision_id="outcome:r1:a",
                                    payload={"row_id": "r1"}, purpose="shadow", kind="outcome",
                                    validations=[], created_at="2026-09-12T20:01:00Z")
        all_rows = ledger_decisions.rows(conn)
        assert [r["kind"] for r in all_rows] == ["prediction", "outcome"]
        only_predictions = ledger_decisions.rows(conn, kind="prediction")
        assert len(only_predictions) == 1
        first_sequence = all_rows[0]["sequence"]
        bounded = ledger_decisions.rows(conn, through=first_sequence)
        assert [r["decision_id"] for r in bounded] == ["prediction:r1"]
    finally:
        conn.close()


def test_row_conflicts_pure():
    original = b'{"a": 1}'
    payload = {"a": 1}
    assert ledger_decisions._row_conflicts(None, [], original, payload) is False
    # A prior import with DIFFERENT bytes under the same identity conflicts.
    assert ledger_decisions._row_conflicts(None, [(b'{"a": 2}',)], original, payload) is True
    # A committed row whose canonical content differs from this payload.
    existing_row = {"payload_json": ledger_decisions.canonical_json({"a": 2})}
    assert ledger_decisions._row_conflicts(existing_row, [], original, payload) is True
    matching_row = {"payload_json": ledger_decisions.canonical_json(payload)}
    assert ledger_decisions._row_conflicts(matching_row, [], original, payload) is False


def test_outcome_generation_ref_parses_and_rejects():
    assert ledger_decisions.outcome_generation_ref({}) is None
    assert ledger_decisions.outcome_generation_ref({"resolved_at": "not-a-timestamp"}) is None
    assert ledger_decisions.outcome_generation_ref(
        {"resolved_at": "2026-09-12T21:00:00Z"}) == "2026-09-12"
    assert ledger_decisions.outcome_generation_ref(
        {"settled_at": "2026-09-12T21:00:00+00:00"}) == "2026-09-12"


def test_import_lines_conflict_and_diverge_paths(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    try:
        with transaction(conn):
            set_authority(conn, None, "catalog", "2026-09-12T20:00:00Z")
        with pytest.raises(ValueError):
            with transaction(conn):
                ledger_decisions.import_lines(conn, "src-hash", [], kind="outcome",
                                              created_at="2026-09-12T20:00:00Z",
                                              on_conflict="not-a-mode")
        line_a = json.dumps({"row_id": "r1", "v": 1}, sort_keys=True).encode()
        with transaction(conn):
            receipts = ledger_decisions.import_lines(
                conn, "src-hash", [line_a], kind="prediction", created_at="2026-09-12T20:00:00Z")
        assert len(receipts) == 1
        # Byte-identical re-import of the same (source_hash, line_number):
        # the "prior" shortcut, a no-op that returns the same receipt.
        with transaction(conn):
            again = ledger_decisions.import_lines(
                conn, "src-hash", [line_a], kind="prediction", created_at="2026-09-12T20:09:00Z")
        assert again[0]["decision_id"] == receipts[0]["decision_id"]
        # A changed byte at an ALREADY-imported (source_hash, line_number):
        # a provenance conflict, refused in both modes.
        line_a_changed = json.dumps({"row_id": "r1", "v": 999}, sort_keys=True).encode()
        with pytest.raises(DecisionConflict):
            with transaction(conn):
                ledger_decisions.import_lines(conn, "src-hash", [line_a_changed], kind="prediction",
                                              created_at="2026-09-12T20:10:00Z")
        # No row_id at all.
        with pytest.raises(DecisionConflict):
            with transaction(conn):
                ledger_decisions.import_lines(
                    conn, "src-hash-2", [json.dumps({}).encode()], kind="prediction",
                    created_at="2026-09-12T20:11:00Z")
        # Same decision_id, DIFFERENT content, imported under a NEW
        # (source_hash, line) pair: on_conflict="raise" (default) refuses.
        line_b_conflict = json.dumps({"row_id": "r1", "v": 2}, sort_keys=True).encode()
        with pytest.raises(DecisionConflict):
            with transaction(conn):
                ledger_decisions.import_lines(conn, "src-hash-3", [line_b_conflict], kind="prediction",
                                              created_at="2026-09-12T20:12:00Z")
        # Same conflict under on_conflict="diverge": records a
        # decision_divergences row instead of failing, and is idempotent.
        with transaction(conn):
            diverged = ledger_decisions.import_lines(
                conn, "src-hash-4", [line_b_conflict], kind="prediction",
                created_at="2026-09-12T20:13:00Z", on_conflict="diverge",
                provenance_label="legacy-ledger.jsonl")
        assert diverged[0]["decision_id"] == receipts[0]["decision_id"]
        count = conn.execute("SELECT COUNT(*) FROM decision_divergences").fetchone()[0]
        assert count == 1
        with transaction(conn):
            ledger_decisions.import_lines(
                conn, "src-hash-4", [line_b_conflict], kind="prediction",
                created_at="2026-09-12T20:14:00Z", on_conflict="diverge")
        assert conn.execute("SELECT COUNT(*) FROM decision_divergences").fetchone()[0] == 1
        # Fresh outcome import with derive_generation_ref: a per-row
        # generation_ref stamped from the payload's own resolved_at.
        outcome_line = json.dumps(
            {"row_id": "r9", "status": "resolved", "resolved_at": "2026-09-12T22:00:00Z"},
            sort_keys=True).encode()
        with transaction(conn):
            outcome_receipts = ledger_decisions.import_lines(
                conn, "src-hash-5", [outcome_line], kind="outcome",
                created_at="2026-09-12T20:15:00Z", derive_generation_ref=True)
        assert outcome_receipts[0]["generation_ref"] == "2026-09-12"
    finally:
        conn.close()


def test_import_decision_id_outcome_requires_observation_identity():
    with pytest.raises(DecisionConflict):
        ledger_decisions._import_decision_id("outcome", "r1", {"status": "resolved"})
    with pytest.raises(DecisionConflict):
        ledger_decisions._import_decision_id(
            "outcome", "r1", {"status": "pending", "resolved_at": "2026-09-12T20:00:00Z"})
    first = ledger_decisions._import_decision_id(
        "outcome", "r1", {"status": "resolved", "resolved_at": "2026-09-12T20:00:00Z"})
    assert first.startswith("outcome:r1:")
    assert ledger_decisions._import_decision_id("prediction", "r1", {}) == "prediction:r1"




# --------------------------------------------------------------------------
# engine.v2.ops.decision_commit: pure helpers
# --------------------------------------------------------------------------


def test_stamp_date_parses_and_rejects():
    assert _stamp_date(None) is None
    assert _stamp_date("") is None
    assert _stamp_date("not-a-date") is None
    assert _stamp_date("2026-09-12").isoformat() == "2026-09-12"
    assert _stamp_date("2026-09-12T21:00:00Z").isoformat() == "2026-09-12"


def test_contract_mismatch_field_detects_each_field():
    recorded = {"ticker": "FAKE", "strategy": "TWIN-P", "event_date": SESSION, "settlement": "closed"}
    assert _contract_mismatch_field(dict(recorded), recorded) is None
    for field in ("ticker", "strategy", "event_date", "settlement"):
        payload = dict(recorded)
        payload[field] = "different"
        assert _contract_mismatch_field(payload, recorded) == field


def test_match_same_session_only_matches_recorded_generation_ref():
    existing = [{"status": "unresolvable", "generation_ref": None},
               {"status": "resolved", "generation_ref": "2026-09-10"},
               {"status": "resolved", "generation_ref": SESSION}]
    match = _match_same_session(existing, SESSION)
    assert match["generation_ref"] == SESSION
    assert _match_same_session(existing, "2026-09-20") is None
    # A NULL generation_ref (legacy/never-backfilled row) can never match,
    # even when the caller asks for "no session" (None).
    assert _match_same_session(existing, None) is None


def test_validate_settlement_state_paths():
    with pytest.raises(OpsError) as err:
        _validate_settlement_state({"status": "pending"}, recorded={}, session=SESSION)
    assert err.value.code == "VALIDATION_FAILED"

    assert _validate_settlement_state(
        {"status": "unresolvable", "resolved_at": SESSION + "T21:00:00Z"},
        recorded={}, session=SESSION) == "unresolvable"

    with pytest.raises(OpsError):
        _validate_settlement_state(
            {"status": "resolved", "resolved_at": SESSION + "T21:00:00Z"}, recorded={}, session=SESSION)

    # Current schema: legacy's own exit_finality must say is_final True.
    current = {"schema_version": legacy_ledger_schema_version()}
    with pytest.raises(OpsError):
        _validate_settlement_state(
            {"status": "resolved", "resolved_at": SESSION + "T21:00:00Z",
             "settlement_source": "polygon", "exit_source": "polygon"},
            recorded=current, session=SESSION)
    with pytest.raises(OpsError):
        _require_legacy_exit_finality({"exit_finality": {"is_final": False}})
    assert _require_legacy_exit_finality({"exit_finality": {"is_final": True}}) == "legacy_exit_finality"
    assert _validate_settlement_state(
        {"status": "resolved", "resolved_at": SESSION + "T21:00:00Z",
         "settlement_source": "polygon", "exit_source": "polygon",
         "exit_finality": {"is_final": True}},
        recorded=current, session=SESSION) == "legacy_exit_finality"

    # Grandfathered (schema_version below current): v2's own exit-date proof.
    grandfathered = {"structure": {"exit_date": SESSION}}
    assert _require_v2_finality_session(grandfathered, SESSION) == "v2_finality_session"
    with pytest.raises(OpsError):
        _require_v2_finality_session({"structure": {"exit_date": "2026-09-20"}}, SESSION)
    with pytest.raises(OpsError):
        _require_v2_finality_session({}, SESSION)
    assert _validate_settlement_state(
        {"status": "resolved", "resolved_at": SESSION + "T21:00:00Z",
         "settlement_source": "polygon", "exit_source": "polygon"},
        recorded=grandfathered, session=SESSION) == "v2_finality_session"


# --------------------------------------------------------------------------
# engine.v2.ops.decision_commit: the settlement-import line validator,
# driven directly against a real ledger connection (no job scheduler
# needed -- _settlement_line/_settlement_dedupe_skip take only ``conn``).
# --------------------------------------------------------------------------


def _seed_prediction(conn, clock, *, row_id, generation_ref, extra=None):
    payload = {"row_id": row_id, "ticker": "FAKE", "strategy": "TWIN-P", "event_date": SESSION,
              "settlement": "closed"}
    if extra:
        payload.update(extra)
    with transaction(conn):
        return ledger_decisions.insert(
            conn, logical_key="pred-" + row_id, decision_id="prediction:" + row_id, payload=payload,
            purpose="shadow", kind="prediction", validations=[], created_at=SESSION + "T20:00:00Z",
            generation_ref=generation_ref)


def _settlement_item(row_id, **fields):
    payload = {"row_id": row_id, "ticker": "FAKE", "strategy": "TWIN-P", "event_date": SESSION,
              "settlement": "closed"}
    payload.update(fields)
    original = json.dumps(payload, sort_keys=True).encode()
    return {"original_b64": base64.b64encode(original).decode("ascii"), "row": payload}


def test_settlement_line_malformed_and_missing_prediction(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    try:
        with transaction(conn):
            set_authority(conn, None, "catalog", SESSION + "T20:00:00Z")
        with pytest.raises(OpsError) as err:
            with transaction(conn):
                _settlement_line(conn, "not-a-dict", session=SESSION, clock=clock, existing_index={})
        assert err.value.code == "VALIDATION_FAILED"

        with pytest.raises(OpsError):
            with transaction(conn):
                _settlement_line(conn, {"original_b64": "###not-b64###", "row": {}}, session=SESSION,
                                 clock=clock, existing_index={})

        item = _settlement_item("r-nomatch", status="resolved", resolved_at=SESSION + "T21:00:00Z")
        item["row"] = dict(item["row"], ticker="DIFFERENT")
        with pytest.raises(OpsError):
            with transaction(conn):
                _settlement_line(conn, item, session=SESSION, clock=clock, existing_index={})

        unknown = _settlement_item("r-unknown", status="resolved", resolved_at=SESSION + "T21:00:00Z")
        with pytest.raises(OpsError) as err2:
            with transaction(conn):
                _settlement_line(conn, unknown, session=SESSION, clock=clock, existing_index={})
        assert err2.value.code == "VALIDATION_FAILED"
    finally:
        conn.close()


def test_settlement_line_contract_mismatch_records_divergence(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    try:
        with transaction(conn):
            set_authority(conn, None, "catalog", SESSION + "T20:00:00Z")
        _seed_prediction(conn, clock, row_id="r1", generation_ref="gen-a")
        item = _settlement_item("r1", status="resolved", resolved_at=SESSION + "T21:00:00Z",
                                settlement_source="polygon", exit_source="polygon",
                                ticker="OTHER")
        diverged = []
        with transaction(conn):
            result = _settlement_line(conn, item, session=SESSION, clock=clock, existing_index={},
                                      on_divergence=lambda row_id, *a: diverged.append(row_id))
        assert result is None
        assert diverged == ["r1"]
        assert conn.execute("SELECT COUNT(*) FROM decision_divergences").fetchone()[0] == 1
    finally:
        conn.close()


def test_settlement_line_dedupe_and_happy_paths(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    try:
        with transaction(conn):
            set_authority(conn, None, "catalog", SESSION + "T20:00:00Z")
        _seed_prediction(conn, clock, row_id="r1", generation_ref="gen-a",
                         extra={"structure": {"exit_date": SESSION}})

        # already_observed_this_session: same generation_ref, same status.
        existing_index = {"r1": [{"status": "resolved", "generation_ref": SESSION}]}
        item = _settlement_item("r1", status="resolved", resolved_at=SESSION + "T21:00:00Z",
                                settlement_source="polygon", exit_source="polygon")
        skipped = []
        with transaction(conn):
            result = _settlement_line(conn, item, session=SESSION, clock=clock,
                                      existing_index=existing_index,
                                      on_skip=lambda row_id, reason: skipped.append((row_id, reason)))
        assert result is None
        assert skipped == [("r1", "already_observed_this_session")]

        # Same session, DIFFERENT status: a status-change divergence.
        diverged = []
        with transaction(conn):
            result = _settlement_line(conn, item, session=SESSION, clock=clock,
                                      existing_index={"r1": [{"status": "unresolvable",
                                                              "generation_ref": SESSION}]},
                                      on_divergence=lambda row_id, *a: diverged.append(row_id))
        assert result is None
        assert diverged == ["r1"]

        # already_resolved: a DIFFERENT session already recorded "resolved".
        skipped2 = []
        with transaction(conn):
            result = _settlement_line(
                conn, item, session=SESSION, clock=clock,
                existing_index={"r1": [{"status": "resolved", "generation_ref": "2026-09-01"}]},
                on_skip=lambda row_id, reason: skipped2.append((row_id, reason)))
        assert result is None
        assert skipped2 == [("r1", "already_resolved")]

        # Happy path: no existing observation at all -- returns the original
        # bytes and reports the grandfathered proof kind.
        admitted = []
        with transaction(conn):
            result = _settlement_line(
                conn, item, session=SESSION, clock=clock, existing_index={},
                on_admitted=lambda row_id, proof: admitted.append((row_id, proof)))
        assert result == base64.b64decode(item["original_b64"])
        assert json.loads(result) == item["row"]
        assert admitted == [("r1", "v2_finality_session")]
    finally:
        conn.close()


def test_validate_candidates_requires_bound_evidence():
    with pytest.raises(OpsError) as err:
        validate_candidates([], {})
    assert err.value.code == "VALIDATION_FAILED"




# --------------------------------------------------------------------------
# engine.v2.ops.decision_commit: commit_decisions_in_transaction's own
# validation guards, and the commit_decisions/validate_candidates
# "compatibility test seam" pair -- reusing the real claim/context build
# from test_coordinator_refuses_when_recorded_binding_disagrees_with_validation
# above, since ``resolve_bindings``/``validated_decision_candidate`` are not
# worth re-deriving by hand.
# --------------------------------------------------------------------------


def _build_valid_claim_and_context(conn, store, clock):
    score, finality = _score_and_finality()
    plan = {"schema_version": "decision_plan.v1.0", "session": SESSION,
            "deployment": "shadow-deployment", "decision_clock": SESSION + "T21:00:00+00:00",
            "expected_population": ["FAKE|TWIN-P|" + SESSION]}
    score_ref = _publish(store, conn, clock, {"rows": [score]}, "legacy_action.v1.0")
    finality_ref = _publish(store, conn, clock, finality, "legacy_action.v1.0")
    plan_ref = _publish(store, conn, clock, plan, "decision_plan.v1.0")
    evidence = _decision_evidence(score_ref, finality_ref, plan_ref, score, finality, plan)
    evidence_ref = _publish(store, conn, clock, evidence, "decision_evidence.v1.0")
    with transaction(conn):
        set_authority(conn, None, "catalog", SESSION + "T20:00:00.000000Z")
    bindings = {"score.json": score_ref.artifact_id, "finality.json": finality_ref.artifact_id,
                "decision_plan.json": plan_ref.artifact_id,
                "decision_evidence.json": evidence_ref.artifact_id}
    refs = (score_ref.artifact_id, finality_ref.artifact_id, plan_ref.artifact_id,
            evidence_ref.artifact_id)
    job = JobSpec(kind="legacy_decisions", implementation_ref="x", spec_hash=None,
                 environment_ref="x",
                 parameters={"expected_ids": ("legacy_decisions",), "session": SESSION,
                             "input_bindings": bindings},
                 input_refs=refs, output_namespace="shadow", resource_class="validation",
                 retry_policy_ref="bounded", checkpoint_contract_ref="legacy_action.v1.0")
    submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key="guard-fixture", principal="operator", job=job),
        clock=clock)
    setup_epoch = begin_epoch(conn, clock=clock, boot_id="setup", pid=1)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                       supervisor=Supervisor(setup_epoch, "setup"), clock=clock, registry=registry())
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
    return claim, candidates, context, score, finality, plan, evidence


def test_commit_decisions_in_transaction_guard_branches(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    store = ArtifactStore(tmp_path)
    try:
        claim, candidates, context, *_ = _build_valid_claim_and_context(conn, store, clock)

        # Requires the caller's own open transaction.
        with pytest.raises(ValueError):
            commit_decisions_in_transaction(conn, claim, candidates, context, clock=clock)

        bad_purpose = dict(context, purpose="not-shadow")
        with pytest.raises(OpsError) as err:
            with transaction(conn):
                commit_decisions_in_transaction(conn, claim, candidates, bad_purpose, clock=clock)
        assert err.value.code == "VALIDATION_FAILED"

        bad_requested = dict(context, requested_session="2020-01-01")
        with pytest.raises(OpsError) as err:
            with transaction(conn):
                commit_decisions_in_transaction(conn, claim, candidates, bad_requested, clock=clock)
        assert err.value.code == "INPUT_CHANGED"

        bad_finality_date = dict(context, finality_date="2020-01-01")
        with pytest.raises(OpsError) as err:
            with transaction(conn):
                commit_decisions_in_transaction(conn, claim, candidates, bad_finality_date, clock=clock)
        assert err.value.code == "INPUT_CHANGED"

        bad_rows_hash = dict(context, candidate_rows_hash="sha256:" + "0" * 64)
        with pytest.raises(OpsError) as err:
            with transaction(conn):
                commit_decisions_in_transaction(conn, claim, candidates, bad_rows_hash, clock=clock)
        assert err.value.code == "INPUT_CHANGED"

        # No decisions committed by any of the refused attempts above.
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0

        # The untampered context still commits cleanly.
        with transaction(conn):
            receipts = commit_decisions_in_transaction(conn, claim, candidates, context, clock=clock)
        assert len(receipts) == 1
    finally:
        conn.close()


def test_commit_decisions_wrapper_and_validate_candidates_seam(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    store = ArtifactStore(tmp_path)
    try:
        claim, candidates, context, score, finality, plan, evidence = (
            _build_valid_claim_and_context(conn, store, clock))
        strict = {"score": {"rows": [score]}, "finality": finality, "plan": plan, "evidence": evidence,
                  "bindings": context["bindings"]}
        wrapper_context = {"candidate_validation": strict}
        validated = validate_candidates(candidates, wrapper_context)
        assert validated["candidate_rows_hash"] == context["candidate_rows_hash"]

        # A mismatched validated_context refuses before ever opening a
        # transaction or touching the catalog.
        with pytest.raises(OpsError) as err:
            commit_decisions(conn, claim, candidates, wrapper_context, {"tampered": True}, clock=clock)
        assert err.value.code == "INPUT_CHANGED"
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0

        # The matched validated_context commits through the real
        # in-transaction path.
        receipts = commit_decisions(conn, claim, candidates, wrapper_context, validated, clock=clock)
        assert len(receipts) == 1
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    finally:
        conn.close()




# --------------------------------------------------------------------------
# engine.v2.ops.legacy_adapter: filesystem-safety helpers and small pure
# refusals -- real files/symlinks under tmp_path, no legacy subprocess.
# --------------------------------------------------------------------------


def test_manifest_files_refuses_missing_or_symlinked_member(tmp_path):
    real = tmp_path / "a.txt"
    real.write_bytes(b"data")
    manifest = manifest_files(tmp_path, ("a.txt",))
    assert manifest["a.txt"]["byte_size"] == 4

    with pytest.raises(OpsError) as err:
        manifest_files(tmp_path, ("missing.txt",))
    assert err.value.code == "INPUT_CHANGED"

    (tmp_path / "link.txt").symlink_to(real)
    with pytest.raises(OpsError):
        manifest_files(tmp_path, ("link.txt",))


def test_copy_read_set_refuses_symlinked_roots_and_verifies_copies(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_bytes(b"data")
    target = tmp_path / "private"

    copied = copy_read_set(source, target, ("a.txt",))
    assert copied["a.txt"]["byte_size"] == 4
    assert (target / "a.txt").read_bytes() == b"data"

    symlinked_source = tmp_path / "source-link"
    symlinked_source.symlink_to(source)
    with pytest.raises(OpsError) as err:
        copy_read_set(symlinked_source, tmp_path / "private2", ("a.txt",))
    assert err.value.code == "INTEGRITY_FAILED"

    symlinked_target = tmp_path / "target-link"
    symlinked_target.symlink_to(target)
    with pytest.raises(OpsError):
        copy_read_set(source, symlinked_target, ("a.txt",))


def test_overlay_read_set_refuses_symlinked_source_and_existing_destination(tmp_path):
    materialization = tmp_path / "materialized"
    materialization.mkdir()
    (materialization / "curated").mkdir()
    (materialization / "curated" / "f.txt").write_bytes(b"x")

    private = tmp_path / "overlay"
    overlay_read_set(materialization, private)
    assert (private / "curated" / "f.txt").is_symlink()
    assert (private / "curated" / "f.txt").read_bytes() == b"x"

    with pytest.raises(OpsError) as err:
        overlay_read_set(materialization, private)
    assert err.value.code == "INTEGRITY_FAILED"

    symlinked = tmp_path / "materialized-link"
    symlinked.symlink_to(materialization)
    with pytest.raises(OpsError):
        overlay_read_set(symlinked, tmp_path / "overlay-2")


def test_legacy_action_refuses_unallowlisted_action(tmp_path):
    with pytest.raises(OpsError) as err:
        legacy_action("legacy_not_a_real_action", {}, tmp_path)
    assert err.value.code == "INVALID_REQUEST"


def test_invoke_nightly_helper_refuses_unaudited_helper():
    with pytest.raises(OpsError) as err:
        invoke_nightly_helper(REPO, "not_an_audited_helper")
    assert err.value.code == "INVALID_REQUEST"


def test_run_legacy_script_refuses_unregistered_and_args(tmp_path):
    with pytest.raises(OpsError) as err:
        run_legacy_script(REPO, "experiments/not_registered/run.py")
    assert err.value.code == "INVALID_REQUEST"

    with pytest.raises(OpsError) as err:
        run_legacy_script(REPO, "experiments/EXP-182_d_1_gated_execution_parity_registered/run.py",
                          args=("--extra",))
    assert err.value.code == "INVALID_REQUEST"


def test_run_legacy_rebuild_surfaces_subprocess_failure(tmp_path):
    """A candidate root with no ``engine.data.rebuild`` module fails fast
    (``ModuleNotFoundError``, non-zero exit) -- a real subprocess, never a
    successful rebuild, so this stays cheap."""
    empty_repo = tmp_path / "empty-repo"
    empty_repo.mkdir()
    with pytest.raises(OpsError) as err:
        run_legacy_rebuild(tmp_path / "candidate", empty_repo, timeout=30)
    assert err.value.code == "VALIDATION_FAILED"


class _FakeResult:
    def __init__(self, stdout, stderr=""):
        self.stdout = stdout
        self.stderr = stderr


def test_json_stdout_refuses_non_json_output():
    with pytest.raises(OpsError) as err:
        _json_stdout(_FakeResult("not json", "boom"), "could not parse")
    assert err.value.code == "VALIDATION_FAILED"
    assert _json_stdout(_FakeResult(json.dumps({"a": 1})), "x") == {"a": 1}


def test_legacy_adapter_load_helpers_require_their_artifact(tmp_path):
    with pytest.raises(OpsError) as err:
        _load_action_frame(tmp_path)
    assert err.value.code == "INPUT_CHANGED"
    with pytest.raises(OpsError):
        _load_finality(tmp_path)
    with pytest.raises(OpsError):
        _load_score_document(tmp_path)


def test_legacy_ledger_schema_version_is_a_real_constant():
    assert legacy_ledger_schema_version() == 3




# --------------------------------------------------------------------------
# engine.v2.foundation: small pure edges the Phase 1/2 suites never happen
# to exercise (RFC 8785 exponential number form, a naive-datetime refusal,
# and _serialize's own defensive type guard).
# --------------------------------------------------------------------------


def test_format_timestamp_refuses_a_naive_datetime():
    import datetime as _dt

    from engine.v2.foundation import format_timestamp
    with pytest.raises(ValueError):
        format_timestamp(_dt.datetime(2026, 9, 12, 12, 0, 0))


def test_canonical_json_renders_large_magnitudes_in_exponential_form():
    from engine.v2.foundation import canonical_json
    encoded = canonical_json(1e21)
    assert "e+" in encoded


def test_canonical_serialize_refuses_an_unnormalized_type_directly():
    from engine.v2.foundation import canonical as canonical_module
    with pytest.raises(TypeError):
        canonical_module._serialize(object())




# --------------------------------------------------------------------------
# engine.v2.ops.provider_budget: account-wide leases, call accounting and
# durable source backoff. Not part of any D0x row's own dedicated test file,
# but wholly untested under the Phase 2 fixed suite before this addition
# (only D4's admission-blocking case, in tests/test_v2_ops_provider_admission.py,
# is covered, and that file is not in the fixed suite) -- real sqlite
# transactions via ops_support.catalog/enqueue_claim, no mocking.
# --------------------------------------------------------------------------

from engine.v2.ops import provider_budget as _provider_budget  # noqa: E402
from tests import ops_support as _ops_support  # noqa: E402


def test_configure_account_refuses_a_negative_budget(tmp_path):
    conn, clock, _supervisor = _ops_support.catalog(tmp_path)
    with pytest.raises(OpsError) as err:
        _provider_budget.configure_account(conn, "acct-neg", "gen-1", remaining=-1, live_reserve=0)
    assert err.value.code == "INVALID_REQUEST"
    with pytest.raises(OpsError):
        _provider_budget.configure_account(conn, "acct-neg", "gen-1", remaining=1, live_reserve=-1)


def test_reserve_refuses_a_non_positive_call_count(tmp_path):
    conn, clock, supervisor = _ops_support.catalog(tmp_path)
    claim = _ops_support.enqueue_claim(conn, clock, supervisor)
    with pytest.raises(OpsError) as err:
        _provider_budget.reserve(conn, claim, "acct-1", 0, clock=clock)
    assert err.value.code == "INVALID_REQUEST"


def test_reserve_refuses_an_unconfigured_or_blocked_account(tmp_path):
    conn, clock, supervisor = _ops_support.catalog(tmp_path)
    claim = _ops_support.enqueue_claim(conn, clock, supervisor)
    with pytest.raises(OpsError) as err:
        _provider_budget.reserve(conn, claim, "no-such-account", 1, clock=clock)
    assert err.value.code == "CREDENTIAL_INVALID"

    _provider_budget.configure_account(conn, "acct-blocked", "gen-1", remaining=10, live_reserve=0)
    _provider_budget.record_response(conn, "acct-blocked", 401, clock=clock)
    with pytest.raises(OpsError) as err:
        _provider_budget.reserve(conn, claim, "acct-blocked", 1, clock=clock)
    assert err.value.code == "CREDENTIAL_INVALID"


def test_reserve_refuses_during_an_unexpired_backoff(tmp_path):
    conn, clock, supervisor = _ops_support.catalog(tmp_path)
    claim = _ops_support.enqueue_claim(conn, clock, supervisor)
    _provider_budget.configure_account(conn, "acct-rl", "gen-1", remaining=10, live_reserve=0)
    _provider_budget.record_response(conn, "acct-rl", 429, clock=clock)
    with pytest.raises(OpsError) as err:
        _provider_budget.reserve(conn, claim, "acct-rl", 1, clock=clock)
    assert err.value.code == "RATE_LIMITED"
    clock.advance(66)
    _provider_budget.reserve(conn, claim, "acct-rl", 1, clock=clock)


def test_reserve_refuses_a_second_active_lease_and_an_over_budget_request(tmp_path):
    conn, clock, supervisor = _ops_support.catalog(tmp_path)
    claim_a = _ops_support.enqueue_claim(conn, clock, supervisor, key="one")
    claim_b = _ops_support.enqueue_claim(conn, clock, supervisor, key="two")
    _provider_budget.configure_account(conn, "acct-lease", "gen-1", remaining=5, live_reserve=0)
    _provider_budget.reserve(conn, claim_a, "acct-lease", 2, clock=clock)
    with pytest.raises(OpsError) as err:
        _provider_budget.reserve(conn, claim_b, "acct-lease", 1, clock=clock)
    assert err.value.code == "RESOURCE_UNAVAILABLE"

    _provider_budget.configure_account(conn, "acct-tight", "gen-1", remaining=2, live_reserve=1)
    with pytest.raises(OpsError) as err:
        _provider_budget.reserve(conn, claim_a, "acct-tight", 5, clock=clock)
    assert err.value.code == "RESOURCE_UNAVAILABLE"


def test_before_request_refuses_without_an_active_reservation(tmp_path):
    conn, clock, supervisor = _ops_support.catalog(tmp_path)
    claim = _ops_support.enqueue_claim(conn, clock, supervisor)
    _provider_budget.configure_account(conn, "acct-noreserve", "gen-1", remaining=10, live_reserve=0)
    with pytest.raises(OpsError) as err:
        _provider_budget.before_request(conn, claim, "acct-noreserve", clock=clock)
    assert err.value.code == "CREDENTIAL_INVALID"


def test_before_request_refuses_rate_limit_and_exhausted_retries_then_admits(tmp_path):
    conn, clock, supervisor = _ops_support.catalog(tmp_path)
    claim = _ops_support.enqueue_claim(conn, clock, supervisor)
    _provider_budget.configure_account(conn, "acct-br", "gen-1", remaining=10, live_reserve=0)
    _provider_budget.reserve(conn, claim, "acct-br", 1, clock=clock)

    with transaction(conn):
        conn.execute("UPDATE provider_accounts SET next_eligible_at = ? WHERE account = ?",
                     ("2099-01-01T00:00:00.000000Z", "acct-br"))
    with pytest.raises(OpsError) as err:
        _provider_budget.before_request(conn, claim, "acct-br", clock=clock)
    assert err.value.code == "RATE_LIMITED"
    with transaction(conn):
        conn.execute("UPDATE provider_accounts SET next_eligible_at = NULL WHERE account = ?",
                     ("acct-br",))

    before = conn.execute("SELECT remaining FROM provider_accounts WHERE account=?",
                          ("acct-br",)).fetchone()["remaining"]
    _provider_budget.before_request(conn, claim, "acct-br", clock=clock)
    after = conn.execute("SELECT remaining, uncertain FROM provider_accounts WHERE account=?",
                         ("acct-br",)).fetchone()
    assert after["remaining"] == before - 1
    assert after["uncertain"] == 1

    with pytest.raises(OpsError) as err:
        _provider_budget.before_request(conn, claim, "acct-br", clock=clock)
    assert err.value.code == "RESOURCE_UNAVAILABLE"


def test_record_response_maps_every_status_family(tmp_path):
    conn, clock, supervisor = _ops_support.catalog(tmp_path)
    _provider_budget.configure_account(conn, "acct-resp", "gen-1", remaining=10, live_reserve=0)

    with pytest.raises(OpsError) as err:
        _provider_budget.record_response(conn, "acct-resp", 200, clock=clock, remaining=-1)
    assert err.value.code == "INVALID_REQUEST"

    code = _provider_budget.record_response(conn, "acct-resp", 403, clock=clock)
    assert code == "CREDENTIAL_INVALID"
    blocked = conn.execute("SELECT blocked_code FROM provider_accounts WHERE account=?",
                           ("acct-resp",)).fetchone()["blocked_code"]
    assert blocked == "CREDENTIAL_INVALID"

    assert _provider_budget.record_response(conn, "acct-resp", 404, clock=clock) == "SOURCE_NOT_FOUND"
    assert _provider_budget.record_response(conn, "acct-resp", 503, clock=clock) == "TRANSIENT_SOURCE"

    code = _provider_budget.record_response(conn, "acct-resp", 429, clock=clock)
    assert code == "RATE_LIMITED"
    eligible = conn.execute("SELECT next_eligible_at FROM provider_accounts WHERE account=?",
                            ("acct-resp",)).fetchone()["next_eligible_at"]
    assert eligible is not None

    assert _provider_budget.record_response(conn, "acct-resp", 200, clock=clock, final=False) == "SOURCE_NOT_FINAL"
    assert _provider_budget.record_response(conn, "acct-resp", 200, clock=clock, empty=True) == "SOURCE_EMPTY"
    assert _provider_budget.record_response(conn, "acct-resp", 200, clock=clock, remaining=42) is None
    remaining = conn.execute("SELECT remaining, uncertain FROM provider_accounts WHERE account=?",
                             ("acct-resp",)).fetchone()
    assert remaining["remaining"] == 42
    assert remaining["uncertain"] == 0




# --------------------------------------------------------------------------
# engine.v2.ops.worker: the fixed subprocess entrypoint's own pure/file-local
# helpers (exception classification, diagnostics staging, the non-subprocess
# ``dispatch`` branches) -- real files under tmp_path, no mocking. ``main()``
# itself (stdin pipe + sched_setaffinity) stays untested here; every worker
# already exercises it end-to-end through a real subprocess elsewhere.
# --------------------------------------------------------------------------

from engine.v2.ops import worker as _worker  # noqa: E402


def test_classify_maps_each_reviewed_exception_family(tmp_path):
    ops_err = OpsError(_worker.make_problem("VALIDATION_FAILED", "x"))
    assert _worker._classify(ops_err) is ops_err.problem

    problem = _worker._classify(ModuleNotFoundError("no module named 'bogus_thing'"))
    assert problem.code == "INPUT_CHANGED"
    assert "bogus_thing" in problem.message or problem.details.get("module") is not None

    problem = _worker._classify(ValueError("boom"))
    assert problem.code == "VALIDATION_FAILED"
    assert "boom" in problem.message

    problem = _worker._classify(KeyError("missing"))
    assert problem.code == "WORKER_FAILED"
    assert problem.details["exception_type"] == "KeyError"
    assert "missing" not in problem.message


def test_last_frame_reads_the_deepest_traceback_entry_or_none():
    assert _worker._last_frame(ValueError("no traceback")) is None
    try:
        raise ValueError("has one")
    except ValueError as exc:
        frame = _worker._last_frame(exc)
        assert frame is not None
        assert __file__.split("/")[-1] in frame or "test_v2_ops_nightly_completion.py" in frame


def test_write_diagnostics_and_failure_details_land_in_private_files(tmp_path):
    try:
        raise RuntimeError("diagnostic content")
    except RuntimeError:
        _worker._write_diagnostics(tmp_path)
    stderr_path = tmp_path / "diagnostics" / "worker.stderr"
    assert "diagnostic content" in stderr_path.read_text()

    _worker._write_failure_details(tmp_path, {"missing": ["a"], "unplanned": ["b"]})
    details_path = tmp_path / "diagnostics" / "failure_details.json"
    assert json.loads(details_path.read_text()) == {"missing": ["a"], "unplanned": ["b"]}


def test_failure_result_assembles_a_typed_worker_result(tmp_path):
    try:
        raise ValueError("bad candidate")
    except ValueError as exc:
        result = _worker._failure_result(tmp_path, exc)
    assert result["failure"] == "VALIDATION_FAILED"
    assert result["problem"]["code"] == "VALIDATION_FAILED"
    assert (tmp_path / "diagnostics" / "worker.stderr").exists()


def test_dispatch_refuses_an_unsupported_worker(tmp_path):
    with pytest.raises(ValueError):
        _worker.dispatch("no-such-worker", {}, tmp_path)


def test_dispatch_artifact_check_reports_affinity_and_threads(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    result = _worker.dispatch("artifact_check", {"expected_ids": ["a", "b"]}, tmp_path)
    assert result["completed_ids"] == ["a", "b"]
    assert result["no_work"] is False
    assert (tmp_path / "receipt.json").exists()
    assert result["observed"]["threads"] == "1"


def test_dispatch_artifact_check_reports_no_work_for_an_empty_expected_set(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    result = _worker.dispatch("artifact_check", {"expected_ids": []}, tmp_path)
    assert result["no_work"] is True
    assert result["completed_ids"] == []


def test_dispatch_effect_receipt_names_its_output_with_a_receipt_suffix(tmp_path):
    for kind in ("ledger_export", "engineering_gate", "publication", "backup"):
        result = _worker.dispatch(kind, {"expected_ids": ["x"]}, tmp_path)
        assert result["outputs"][0]["name"] == kind + "_receipt"
        assert result["completed_ids"] == ["x"]
        assert result["no_work"] is False




# --------------------------------------------------------------------------
# engine.v2.foundation: SystemClock's real (non-fake) clock methods, and
# untag_nonfinite's malformed-tag fallback -- neither exercised by any
# existing suite file (every test injects a FakeClock; every ``__nonfinite__``
# round-trip test uses a real repr(nan)/repr(inf) string).
# --------------------------------------------------------------------------


def test_system_clock_now_and_monotonic_return_real_values():
    import datetime as _dt

    clock = SystemClock()
    now = clock.now()
    assert now.tzinfo is not None
    assert now.tzinfo.utcoffset(now) == _dt.timedelta(0)
    a = clock.monotonic()
    b = clock.monotonic()
    assert isinstance(a, float) and isinstance(b, float)
    assert b >= a


def test_untag_nonfinite_leaves_a_malformed_tag_as_an_ordinary_dict():
    from engine.v2.foundation.canonical import untag_nonfinite

    malformed = {"__nonfinite__": "not-a-number"}
    assert untag_nonfinite(malformed) == {"__nonfinite__": "not-a-number"}
    real_nan_tag = {"__nonfinite__": repr(float("nan"))}
    import math as _math
    assert _math.isnan(untag_nonfinite(real_nan_tag))


