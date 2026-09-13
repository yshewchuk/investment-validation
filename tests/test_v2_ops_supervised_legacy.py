"""A1/A4 end-to-end: one real legacy kind through the real Service, using a
real subprocess worker — the gap that hid every defect in this file's siblings.

``legacy_decisions`` is the cheapest legacy kind to run for real: it needs no
market data, no scorer, no FeatureContext — ``engine.ledger.build_prediction_rows``
only touches ``store.read_table("earnings_events", ...)``, which degrades to
an empty frame when the table is absent (see ``engine/ledger.py:_event_ids``),
so the whole legacy fixture root can be a single unrelated file the manifest
declares as its (otherwise-unused) read set.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from engine.v2.contracts import JobSpec, LegacyFileRef, LegacyInputManifest, SubmitRequest
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash, to_document
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import artifact as load_artifact, register_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import environment_identity, file_hash, worker_source_manifest
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit
from engine.v2.ops.supervisor import Service
from engine.v2.ledger.decisions import set_authority

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})
SESSION = "2026-09-12"


def _publish(store, conn, clock, value, schema_ref):
    ref = store.publish_bytes(json.dumps(value, sort_keys=True).encode(), schema_ref=schema_ref)
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


def _manifest_ref(store, conn, clock, file_refs, *, complete=True):
    document = to_document(LegacyInputManifest(
        manifest_id="m1", file_refs=tuple(file_refs), table_contract_refs=(),
        registry_and_model_refs=(), calendar_ref=None, selected_session=SESSION,
        finality_receipt_refs=(), knowledge_mode_by_table={}, availability_evidence_refs=(),
        read_set_complete=complete, capture_implementation_ref="test.v1"))
    return _publish(store, conn, clock, document, "legacy_input_manifest.v1.0")


def _submit_decisions(conn, *, manifest_ref, score_ref, finality_ref,
                      plan_ref=None, evidence_ref=None, key="dec1"):
    profile = profile_named(DEFAULT_POLICY, "validation")
    bindings = {"legacy_manifest.json": manifest_ref.artifact_id,
                "score.json": score_ref.artifact_id,
                "finality.json": finality_ref.artifact_id}
    refs = [manifest_ref.artifact_id, score_ref.artifact_id, finality_ref.artifact_id]
    if plan_ref is not None:
        bindings["decision_plan.json"] = plan_ref.artifact_id
        refs.append(plan_ref.artifact_id)
    if evidence_ref is not None:
        bindings["decision_evidence.json"] = evidence_ref.artifact_id
        refs.append(evidence_ref.artifact_id)
    job = JobSpec(
        kind="legacy_decisions",
        implementation_ref=content_hash(worker_source_manifest(REPO)),
        spec_hash=None,
        environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
        parameters={"expected_ids": ("legacy_decisions",), "session": SESSION,
                    "tickers": (), "year_start": 2024, "year_end": 2026,
                    "input_bindings": bindings},
        input_refs=tuple(refs),
        output_namespace="shadow", resource_class="validation", retry_policy_ref="bounded",
        checkpoint_contract_ref="legacy_action.v1.0")
    return submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key=key, principal="operator", job=job), clock=SystemClock())


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


def test_legacy_decisions_runs_supervised_with_real_subprocess(tmp_path):
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
        manifest_ref = _manifest_ref(store, conn, clock, (LegacyFileRef(
            path="unused.txt", content_hash=file_hash(fixture),
            byte_size=fixture.stat().st_size),))
        score = {"ticker": "FAKE", "event_id": "event-1", "event_date": SESSION,
                 "as_of": SESSION, "entry_date": SESSION, "evidence_cutoff": SESSION,
                 "strategy": "TWIN-P", "strike": 100.0, "expiry": "2026-10-16",
                 "session": "AMC", "snapshot_hash": "sha256:" + "a" * 64}
        score_ref = _publish(store, conn, clock, {"rows": [score]}, "legacy_action.v1.0")
        finality = {"date": SESSION, "is_final": True, "market_wide": True,
                    "daily_share": 1.0, "chain_share": 1.0, "covered": 1}
        finality_ref = _publish(store, conn, clock, finality, "legacy_action.v1.0")
        plan = {"schema_version": "decision_plan.v1.0", "session": SESSION,
                "deployment": "shadow-deployment", "decision_clock": SESSION + "T21:00:00+00:00",
                "expected_population": ["FAKE|TWIN-P|" + SESSION]}
        plan_ref = _publish(store, conn, clock, plan, "decision_plan.v1.0")
        evidence = _decision_evidence(score_ref, finality_ref, plan_ref, score, finality, plan)
        evidence_ref = _publish(store, conn, clock, evidence, "decision_evidence.v1.0")
        with transaction(conn):
            set_authority(conn, None, "catalog", SESSION + "T20:00:00.000000Z")
        receipt = _submit_decisions(conn, manifest_ref=manifest_ref, score_ref=score_ref,
                                    finality_ref=finality_ref, plan_ref=plan_ref,
                                    evidence_ref=evidence_ref)
        service = Service(conn, root, registry(), DEFAULT_POLICY, clock=clock,
                          code_source=REPO, store_root=store_root)
        try:
            service.start()
            state = _run_until_terminal(service, conn, receipt.job_id)
        finally:
            service.close()

        row = conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?",
                           (receipt.job_id,)).fetchone()
        assert state == "succeeded", row["failure_json"]

        attempt_id = conn.execute("SELECT attempt_id FROM attempts WHERE job_id=?",
                                  (receipt.job_id,)).fetchone()[0]
        outputs = conn.execute("SELECT artifact_id FROM attempt_outputs WHERE attempt_id=?",
                               (attempt_id,)).fetchall()
        assert len(outputs) == 1

        ref = load_artifact(conn, store, outputs[0][0])
        content = json.loads(store.read_verified(ref))
        assert content["expected_rows"] == 1
        [decision] = content["rows"]
        assert decision["ticker"] == "FAKE"
        assert decision["finality"] == finality
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
        assert sorted(row[0] for row in conn.execute("SELECT kind FROM outbox")) == [
            "export", "release_intent"]
        mark = conn.execute("SELECT stage,occurrence FROM watermarks").fetchone()
        assert tuple(mark) == ("decisions", SESSION)

        # A1: the declared legacy read set is a real, read-only, non-linked
        # copy under staging — never a symlink or hard link to production.
        copied = root / "attempts" / attempt_id / "staging" / "legacy" / "unused.txt"
        assert copied.is_file() and not copied.is_symlink()
        assert copied.stat().st_nlink == 1
        assert oct(copied.stat().st_mode)[-3:] == "444"
        assert copied.read_bytes() == fixture.read_bytes()
        assert fixture.read_bytes() == b"a legacy read-set member decisions never opens"

        # A worker may prepare a valid-looking candidate, but missing immutable
        # replay evidence must roll back its output registration and every
        # catalog effect when the attempt completes.
        refused = _submit_decisions(conn, manifest_ref=manifest_ref, score_ref=score_ref,
                                    finality_ref=finality_ref, plan_ref=plan_ref,
                                    evidence_ref=None, key="dec-missing-evidence")
        service = Service(conn, root, registry(), DEFAULT_POLICY, clock=clock,
                          code_source=REPO, store_root=store_root)
        try:
            service.start()
            refused_state = _run_until_terminal(service, conn, refused.job_id)
        finally:
            service.close()
        assert refused_state == "failed"
        failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id=?",
                               (refused.job_id,)).fetchone()[0]
        assert "VALIDATION_FAILED" in failure
        refused_outputs = conn.execute(
            "SELECT COUNT(*) FROM attempt_outputs ao JOIN attempts a "
            "ON a.attempt_id=ao.attempt_id WHERE a.job_id=?", (refused.job_id,)).fetchone()[0]
        assert refused_outputs == 0
        assert conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM watermarks").fetchone()[0] == 1
    finally:
        conn.close()


def test_missing_legacy_read_set_member_fails_input_changed_and_releases(tmp_path):
    root = tmp_path / "case2"
    root.mkdir()
    store_root = root / "prod"
    store_root.mkdir()

    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    store = ArtifactStore(root)
    try:
        # The manifest declares a member that was never written to store_root.
        manifest_ref = _manifest_ref(store, conn, clock, (LegacyFileRef(
            path="missing.bin", content_hash="sha256:" + "0" * 64, byte_size=1),))
        score_ref = _publish(store, conn, clock, {"rows": []}, "legacy_action.v1.0")
        finality_ref = _publish(store, conn, clock, {}, "legacy_action.v1.0")
        receipt = _submit_decisions(conn, manifest_ref=manifest_ref, score_ref=score_ref,
                                    finality_ref=finality_ref, key="dec2")
        service = Service(conn, root, registry(), DEFAULT_POLICY, clock=clock,
                          code_source=REPO, store_root=store_root)
        try:
            service.start()
            state = _run_until_terminal(service, conn, receipt.job_id)
        finally:
            service.close()

        row = conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?",
                           (receipt.job_id,)).fetchone()
        assert state == "failed"
        assert "INPUT_CHANGED" in row["failure_json"]
        assert conn.execute("SELECT COUNT(*) FROM store_leases WHERE released_at IS NULL"
                            ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM attempt_outputs").fetchone()[0] == 0
    finally:
        conn.close()


def test_worker_crash_writes_private_diagnostics_and_leaks_no_exception_text(tmp_path):
    """A7: the subprocess crashes for real (no ``score.json`` binding), and its
    traceback must land only in the private per-attempt diagnostics file."""
    root = tmp_path / "case3"
    root.mkdir()
    store_root = root / "prod"
    store_root.mkdir()

    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    store = ArtifactStore(root)
    try:
        manifest_ref = _manifest_ref(store, conn, clock, ())
        finality_ref = _publish(store, conn, clock, {}, "legacy_action.v1.0")
        profile = profile_named(DEFAULT_POLICY, "validation")
        job = JobSpec(
            kind="legacy_decisions",
            implementation_ref=content_hash(worker_source_manifest(REPO)),
            spec_hash=None,
            environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
            parameters={"expected_ids": ("legacy_decisions",), "session": SESSION,
                        "tickers": (), "year_start": 2024, "year_end": 2026,
                        # No "score.json" binding: _load_action_frame must raise.
                        "input_bindings": {"legacy_manifest.json": manifest_ref.artifact_id,
                                           "finality.json": finality_ref.artifact_id}},
            input_refs=(manifest_ref.artifact_id, finality_ref.artifact_id),
            output_namespace="shadow", resource_class="validation", retry_policy_ref="bounded",
            checkpoint_contract_ref="legacy_action.v1.0")
        receipt = submit(conn, registry(), POLICY, SubmitRequest(
            namespace="shadow", idempotency_key="dec3", principal="operator", job=job), clock=clock)
        service = Service(conn, root, registry(), DEFAULT_POLICY, clock=clock,
                          code_source=REPO, store_root=store_root)
        try:
            service.start()
            state = _run_until_terminal(service, conn, receipt.job_id)
        finally:
            service.close()

        row = conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?",
                           (receipt.job_id,)).fetchone()
        assert state == "failed"
        assert "WORKER_FAILED" in row["failure_json"]
        assert "score artifact is missing" not in row["failure_json"]

        attempt_id = conn.execute("SELECT attempt_id FROM attempts WHERE job_id=?",
                                  (receipt.job_id,)).fetchone()[0]
        diagnostics = root / "attempts" / attempt_id / "staging" / "diagnostics" / "worker.stderr"
        assert diagnostics.is_file() and not diagnostics.is_symlink()
        assert oct(diagnostics.stat().st_mode)[-3:] == "600"
        assert "score artifact is missing" in diagnostics.read_text()
    finally:
        conn.close()
