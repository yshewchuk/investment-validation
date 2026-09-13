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


def _submit_decisions(conn, *, manifest_ref, score_ref, finality_ref, key="dec1"):
    profile = profile_named(DEFAULT_POLICY, "validation")
    job = JobSpec(
        kind="legacy_decisions",
        implementation_ref=content_hash(worker_source_manifest(REPO)),
        spec_hash=None,
        environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
        parameters={"expected_ids": ("legacy_decisions",), "session": SESSION,
                    "tickers": (), "year_start": 2024, "year_end": 2026,
                    "input_bindings": {"legacy_manifest.json": manifest_ref.artifact_id,
                                       "score.json": score_ref.artifact_id,
                                       "finality.json": finality_ref.artifact_id}},
        input_refs=(manifest_ref.artifact_id, score_ref.artifact_id, finality_ref.artifact_id),
        output_namespace="shadow", resource_class="validation", retry_policy_ref="bounded",
        checkpoint_contract_ref="legacy_action.v1.0")
    return submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key=key, principal="operator", job=job), clock=SystemClock())


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
        score_ref = _publish(store, conn, clock, {"rows": [
            {"ticker": "FAKE", "event_date": "2026-09-12", "as_of": "2026-09-12",
             "strategy": "TWIN-P", "strike": 100.0, "expiry": "2026-10-16",
             "session": "AMC"}]}, "legacy_action.v1.0")
        finality_ref = _publish(store, conn, clock,
                                {"date": "2026-09-12", "is_final": True}, "legacy_action.v1.0")
        receipt = _submit_decisions(conn, manifest_ref=manifest_ref, score_ref=score_ref,
                                    finality_ref=finality_ref)
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
        assert decision["finality"] == {"date": "2026-09-12", "is_final": True}

        # A1: the declared legacy read set is a real, read-only, non-linked
        # copy under staging — never a symlink or hard link to production.
        copied = root / "attempts" / attempt_id / "staging" / "legacy" / "unused.txt"
        assert copied.is_file() and not copied.is_symlink()
        assert copied.stat().st_nlink == 1
        assert oct(copied.stat().st_mode)[-3:] == "444"
        assert copied.read_bytes() == fixture.read_bytes()
        assert fixture.read_bytes() == b"a legacy read-set member decisions never opens"
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
