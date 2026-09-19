"""P2-5 follow-up: the effect-receipt output-name collision, reproduced
through a REAL ``Service`` with REAL subprocess workers.

Every existing test of ``ledger_export``/``engineering_gate``/``publication``/
``backup`` (``test_v2_ops_effects_graph.py``, ``test_v2_ops_resolved_session.py``)
calls the coordinator effect function directly against a hand-built claim --
it never launches ``worker.py``'s own subprocess, so ``refs`` (the worker's
outputs) never actually lands in ``attempt_outputs`` the way
``supervisor.Service._commit_success`` really builds it: from the LAUNCHED
worker's own ``outputs``, not a stub. That is the gap this file closes.

Before the fix, ``worker.py::_dispatch_effect_receipt`` named its receipt
output after the bare kind (``"ledger_export"``, ``"engineering_gate"``,
``"publication"``, ``"backup"``) -- the exact same name
``effects_graph.ledger_export_effect``/``engineering_gate_effect`` use for
their own coordinator-published ``extra_refs`` artifact. Both land in
``attempt_outputs``, whose primary key is ``(attempt_id, name)``
(``engine/v2/ops/schema_runtime.py``), so the second ``INSERT`` collided:
a raw ``sqlite3.IntegrityError`` surfaced out of ``commit_attempt`` for
``ledger_export`` and ``engineering_gate`` specifically (the two kinds whose
coordinator effect actually returns a non-empty ``extra_refs``;
``publication``/``backup`` return ``(None, ())`` and never collided, but get
the same rename for consistency -- see the module docstring in
``worker.py::_dispatch_effect_receipt``).
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash, format_timestamp
from engine.v2.ledger.decisions import insert, set_authority
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import artifact as load_artifact
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.input_bindings import resolve_bindings
from engine.v2.ops.outbox import enqueue, watermark
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, run_until

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})
SESSION = "2026-09-12"


def _run_until_terminal(service, conn, job_id, timeout=60):
    return run_until(service, conn, job_id, timeout=timeout)


def _submit_effect_kind(conn, clock, *, kind, key, scope, input_bindings=None):
    """A real ``JobSpec`` for one of the four effect-receipt kinds, launched
    through a real subprocess worker exactly the way ``ops submit`` does --
    never a hand-built claim."""
    profile = profile_named(DEFAULT_POLICY, "delivery")
    job = JobSpec(
        kind=kind, implementation_ref=content_hash(worker_source_manifest(REPO)), spec_hash=None,
        environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
        parameters={"expected_ids": (kind,), "session": SESSION, "effect_scope": scope,
                    "input_bindings": input_bindings or {}},
        output_namespace="shadow", resource_class="delivery", retry_policy_ref="bounded",
        checkpoint_contract_ref="effect_receipt.v1.0")
    return submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key=key, principal="operator", job=job), clock=clock)


def _seed_decisions(conn, clock, scope, session, predictions):
    """The committed rows plus the ``decisions`` watermark/outbox rows
    ``ledger_export_effect`` reads -- seeded directly, the same shape
    ``test_v2_ops_effects_graph.py::_seed_decisions`` leaves them in."""
    ts = format_timestamp(clock.now())
    release_key = content_hash([scope, session, "collision-seed", predictions])
    with transaction(conn):
        if conn.execute("SELECT 1 FROM decision_authority WHERE singleton=1").fetchone() is None:
            set_authority(conn, None, "catalog", ts)
        for row in predictions:
            insert(conn, logical_key="pred:" + row["row_id"], decision_id="prediction:" + row["row_id"],
                  payload=row, purpose="shadow", kind="prediction", validations={}, created_at=ts)
        enqueue(conn, "export", release_key, {"validation": "seed"})
        enqueue(conn, "release_intent", release_key, {"validation": "seed"})
        watermark(conn, "nightly", scope, "decisions", session, release_key, clock=clock)


def _open(tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    store_root = root / "prod"
    store_root.mkdir()
    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    store = ArtifactStore(root)
    return root, store_root, clock, conn, store


def _run(conn, root, store_root, clock, job_id, timeout=60):
    service = Service(conn, root, registry(), TEST_POLICY, clock=clock, code_source=REPO,
                      store_root=store_root)
    try:
        service.start()
        return _run_until_terminal(service, conn, job_id, timeout=timeout)
    finally:
        service.close()


def test_ledger_export_runs_supervised_and_binds_the_tar_not_the_receipt(tmp_path):
    """The actual collision: before the fix this raised a raw
    ``sqlite3.IntegrityError`` (UNIQUE constraint failed on
    ``attempt_outputs.attempt_id, attempt_outputs.name``) inside
    ``commit_attempt``'s ``effects`` closure, because the worker's receipt
    and the coordinator's tar both claimed the name ``"ledger_export"``.
    """
    root, store_root, clock, conn, store = _open(tmp_path, "export")
    try:
        scope = "shadow"
        _seed_decisions(conn, clock, scope, SESSION, [
            {"row_id": "evt-1-pred", "event_id": "evt-1", "ticker": "FAKE", "strategy": "TWIN-P",
             "event_date": SESSION, "status": "resolved", "resolved_at": SESSION + "T21:00:00+00:00"}])
        receipt = _submit_effect_kind(conn, clock, kind="ledger_export", key="export-1", scope=scope)
        state = _run(conn, root, store_root, clock, receipt.job_id)

        row = conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?",
                           (receipt.job_id,)).fetchone()
        assert state == "succeeded", row["failure_json"]

        attempt_id = conn.execute("SELECT attempt_id FROM attempts WHERE job_id=?",
                                  (receipt.job_id,)).fetchone()[0]
        outputs = {r[0]: r[1] for r in conn.execute(
            "SELECT name, artifact_id FROM attempt_outputs WHERE attempt_id=?", (attempt_id,))}
        # Two distinct names, never a collision: the worker's own trivial
        # receipt, and the coordinator's real generation tar under the bare
        # kind name downstream bindings expect.
        assert set(outputs) == {"ledger_export_receipt", "ledger_export"}

        render_spec = JobSpec(kind="legacy_render", implementation_ref="x", spec_hash=None,
                              environment_ref="x",
                              parameters={"input_bindings": {
                                  "ledger_generation.tar": receipt.job_id + "#ledger_export"}},
                              dependency_job_ids=(receipt.job_id,), output_namespace="shadow",
                              resource_class="projection", retry_policy_ref="bounded",
                              checkpoint_contract_ref="legacy_action.v1.0")
        resolved = resolve_bindings(conn, store, render_spec)
        tar_ref = load_artifact(conn, store, resolved["ledger_generation.tar"].artifact_id)
        assert tar_ref.artifact_id == outputs["ledger_export"]
        with tarfile.open(store.verify(tar_ref)) as archive:
            members = {m.name for m in archive.getmembers() if m.isfile()}
        assert "predictions/" + SESSION + ".jsonl" in members
    finally:
        conn.close()


def test_engineering_gate_runs_supervised_and_binds_the_gate_document_not_the_receipt(tmp_path):
    """The second actual collision (worker receipt vs. coordinator
    ``extra_refs``, both named ``"engineering_gate"`` before the fix)."""
    root, store_root, clock, conn, store = _open(tmp_path, "gate")
    try:
        scope = "shadow"
        receipt = _submit_effect_kind(conn, clock, kind="engineering_gate", key="gate-1", scope=scope)
        state = _run(conn, root, store_root, clock, receipt.job_id, timeout=120)

        row = conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?",
                           (receipt.job_id,)).fetchone()
        assert state == "succeeded", row["failure_json"]

        attempt_id = conn.execute("SELECT attempt_id FROM attempts WHERE job_id=?",
                                  (receipt.job_id,)).fetchone()[0]
        outputs = {r[0]: r[1] for r in conn.execute(
            "SELECT name, artifact_id FROM attempt_outputs WHERE attempt_id=?", (attempt_id,))}
        assert set(outputs) == {"engineering_gate_receipt", "engineering_gate"}

        gate_ref = load_artifact(conn, store, outputs["engineering_gate"])
        document = json.loads(store.read_verified(gate_ref))
        assert document["schema_version"] == "engineering_gate.v1.0"
        assert "rows" in document  # the real gate document, never the receipt
    finally:
        conn.close()


def test_backup_runs_supervised_with_no_collision(tmp_path):
    """``backup``'s coordinator effect returns no ``extra_refs`` (never
    collided even before the fix), but gets the same ``<kind>_receipt``
    rename for consistency -- proved end to end here rather than assumed."""
    root, store_root, clock, conn, store = _open(tmp_path, "backup")
    try:
        scope = "shadow"
        receipt = _submit_effect_kind(conn, clock, kind="backup", key="backup-1", scope=scope)
        state = _run(conn, root, store_root, clock, receipt.job_id)

        row = conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?",
                           (receipt.job_id,)).fetchone()
        assert state == "succeeded", row["failure_json"]

        attempt_id = conn.execute("SELECT attempt_id FROM attempts WHERE job_id=?",
                                  (receipt.job_id,)).fetchone()[0]
        outputs = [r[0] for r in conn.execute(
            "SELECT name FROM attempt_outputs WHERE attempt_id=?", (attempt_id,))]
        assert outputs == ["backup_receipt"]
        wm = conn.execute("SELECT occurrence FROM watermarks WHERE pipeline='nightly' "
                          "AND scope=? AND stage='backup'", (scope,)).fetchone()
        assert wm["occurrence"] == SESSION
    finally:
        conn.close()


def test_publication_runs_supervised_and_fails_clean_not_with_an_integrity_error(tmp_path):
    """``publication``'s coordinator effect also returns no ``extra_refs``
    (never collided), gets the same rename, and -- with no finality/bundle
    bound, deliberately minimal here since full publication setup is already
    covered directly in ``test_v2_ops_effects_graph.py`` -- fails with a
    clean ``VALIDATION_FAILED`` refusal, proving the real subprocess worker
    and the rename never surface a raw ``sqlite3.IntegrityError`` regardless
    of how the coordinator effect concludes."""
    root, store_root, clock, conn, store = _open(tmp_path, "publication")
    try:
        scope = "shadow"
        receipt = _submit_effect_kind(conn, clock, kind="publication", key="pub-1", scope=scope)
        state = _run(conn, root, store_root, clock, receipt.job_id)

        row = conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?",
                           (receipt.job_id,)).fetchone()
        assert state == "failed"
        assert "VALIDATION_FAILED" in row["failure_json"]
        assert "IntegrityError" not in row["failure_json"]
        attempt_id = conn.execute("SELECT attempt_id FROM attempts WHERE job_id=?",
                                  (receipt.job_id,)).fetchone()[0]
        # The whole commit is atomic: a refused coordinator effect leaves no
        # partial attempt_outputs row behind either, worker receipt included.
        assert conn.execute("SELECT COUNT(*) FROM attempt_outputs WHERE attempt_id=?",
                            (attempt_id,)).fetchone()[0] == 0
    finally:
        conn.close()
