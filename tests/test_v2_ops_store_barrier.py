"""O17: the cooperative legacy store barrier — leases, dirty domains, pinned inputs.

Claim-level controls use the fake clock; the supervisor controls launch the
real fixed worker against a synthetic production root and mutate a pinned
input between the launching tick and the finishing tick.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from engine.v2.contracts import LegacyFileRef, LegacyInputManifest
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash, to_document
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.errors import OpsError, make_problem
from engine.v2.ops.fingerprints import environment_identity, file_hash, worker_source_manifest
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.stages import CheckParameters
from engine.v2.ops.store_barrier import (
    confirm_read_set,
    pin_read_set,
    read_set_complete,
    verified_write_in,
)
from engine.v2.ops.submission import JobKind, KindRegistry, RetryPolicy, submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import POLICY, TEST_POLICY, AdmissionWatch, catalog, request, run_until, sample


def _kind(name, mode):
    return JobKind(name=name, worker="artifact_check", parameters=CheckParameters,
                   resource_classes=frozenset({"delivery"}), effects=("staged",),
                   retry=RetryPolicy("bounded", 3, (1,)), checkpoint_contract="receipt.v1.0",
                   namespaces=frozenset({"shadow"}), store_domains=(("legacy_store", mode),))


BARRIER_REGISTRY = KindRegistry([_kind("reader", "read"), _kind("writer", "write")])


def _submit(conn, clock, key, kind):
    return submit(conn, BARRIER_REGISTRY, POLICY,
                  request(key, kind=kind, checkpoint_contract_ref="receipt.v1.0",
                          parameters={"expected_ids": []}),
                  clock=clock)


def _claim(conn, clock, supervisor):
    return claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                      supervisor=supervisor, clock=clock, registry=BARRIER_REGISTRY)


def _job(conn, key):
    return conn.execute("SELECT state, attempt_count, queue_reason_json FROM jobs "
                        "WHERE idempotency_key=?", (key,)).fetchone()


def _dirty(conn):
    row = conn.execute("SELECT dirty FROM store_domains WHERE domain='legacy_store'").fetchone()
    return None if row is None else row[0]


def _open_leases(conn):
    return conn.execute("SELECT COUNT(*) FROM store_leases WHERE released_at IS NULL").fetchone()[0]


def test_o17_writer_excluded_while_readers_hold_shared_leases(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    _submit(conn, clock, "r1", "reader")
    clock.advance(1)
    _submit(conn, clock, "w1", "writer")
    clock.advance(1)
    _submit(conn, clock, "r2", "reader")

    first = _claim(conn, clock, supervisor)
    assert first.spec.kind == "reader"
    second = _claim(conn, clock, supervisor)
    assert second is not None and second.spec.kind == "reader"
    held = conn.execute("SELECT mode FROM store_leases WHERE released_at IS NULL").fetchall()
    assert [row[0] for row in held] == ["read", "read"]

    assert _claim(conn, clock, supervisor) is None
    waiting = _job(conn, "w1")
    assert waiting["state"] == "queued" and waiting["attempt_count"] == 0
    assert "STORE_LEASE_HELD" in waiting["queue_reason_json"]

    for claim in (first, second):
        commit_attempt(conn, claim.attempt_id, claim.fence,
                       Outcome(True, "verified_dead", 0), clock=clock)
    writer = _claim(conn, clock, supervisor)
    assert writer is not None and writer.spec.kind == "writer"
    assert _dirty(conn) == 1

    # While the write lease is held the domain is dirty: readers stay queued
    # without an attempt, and the write lease excludes a second writer.
    clock.advance(1)
    _submit(conn, clock, "r3", "reader")
    assert _claim(conn, clock, supervisor) is None
    blocked = _job(conn, "r3")
    assert blocked["state"] == "queued" and blocked["attempt_count"] == 0
    assert "STORE_RECOVERY" in blocked["queue_reason_json"]

    def effects(connection):
        verified_write_in(connection, writer.attempt_id, "legacy_store")

    commit_attempt(conn, writer.attempt_id, writer.fence,
                   Outcome(True, "verified_dead", 0), clock=clock, effects=effects)
    assert _dirty(conn) == 0
    assert _open_leases(conn) == 0
    reader_again = _claim(conn, clock, supervisor)
    assert reader_again is not None and reader_again.spec.kind == "reader"
    assert reader_again.job_id == _job_id(conn, "r3")


def _job_id(conn, key):
    return conn.execute("SELECT job_id FROM jobs WHERE idempotency_key=?", (key,)).fetchone()[0]


def test_o17_failed_write_blocks_readers_until_verified_recovery(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    _submit(conn, clock, "w1", "writer")
    clock.advance(1)
    _submit(conn, clock, "r1", "reader")

    attempt = _claim(conn, clock, supervisor)
    assert attempt.spec.kind == "writer"
    commit_attempt(conn, attempt.attempt_id, attempt.fence,
                   Outcome(False, "verified_dead", 1,
                           make_problem("WORKER_FAILED", "worker did not complete its contract")),
                   clock=clock)
    # The lease is released with the attempt, but the domain stays dirty: a
    # partially failed rebuild is not a readable input domain.
    assert _dirty(conn) == 1
    assert _open_leases(conn) == 0

    # The writer's retry is not eligible yet, so this pass reaches the reader
    # and records why it waits: the dirty domain refuses reads.
    assert _claim(conn, clock, supervisor) is None
    blocked = _job(conn, "r1")
    assert blocked["state"] == "queued" and blocked["attempt_count"] == 0
    assert "STORE_RECOVERY" in blocked["queue_reason_json"]

    # The recovery is the writer's own bounded retry; only its verified
    # success clears the domain.
    clock.advance(2)
    recovery = _claim(conn, clock, supervisor)
    assert recovery is not None and recovery.spec.kind == "writer"
    assert recovery.attempt_number == 2

    def effects(connection):
        verified_write_in(connection, recovery.attempt_id, "legacy_store")

    commit_attempt(conn, recovery.attempt_id, recovery.fence,
                   Outcome(True, "verified_dead", 0), clock=clock, effects=effects)
    assert _dirty(conn) == 0
    reader = _claim(conn, clock, supervisor)
    assert reader is not None and reader.spec.kind == "reader"


def _manifest(file_path, relative, *, complete=True, content_hash_override=None):
    return to_document(LegacyInputManifest(
        manifest_id="manifest-1",
        file_refs=(LegacyFileRef(path=relative,
                                 content_hash=content_hash_override or file_hash(file_path),
                                 byte_size=file_path.stat().st_size),),
        table_contract_refs=(), registry_and_model_refs=(), calendar_ref=None,
        selected_session="2026-09-12", finality_receipt_refs=(),
        knowledge_mode_by_table={}, availability_evidence_refs=(),
        read_set_complete=complete, capture_implementation_ref="capture.v1"))


def test_o17_pinned_read_set_refuses_mutated_and_undeclared_inputs(tmp_path):
    prod = tmp_path / "prod"
    (prod / "data").mkdir(parents=True)
    pinned = prod / "data" / "x.bin"
    pinned.write_bytes(b"pinned")
    conn, clock, supervisor = catalog(tmp_path)
    _submit(conn, clock, "r1", "reader")
    claim = _claim(conn, clock, supervisor)

    pin_read_set(conn, claim.attempt_id, _manifest(pinned, "data/x.bin"), prod)
    assert read_set_complete(conn, claim.attempt_id) is True
    confirm_read_set(conn, claim.attempt_id, prod)

    pinned.write_bytes(b"mutated behind the lease")
    with pytest.raises(OpsError, match="INPUT_CHANGED"):
        confirm_read_set(conn, claim.attempt_id, prod)
    pinned.write_bytes(b"pinned")
    confirm_read_set(conn, claim.attempt_id, prod)

    # A manifest whose hashes do not match the bytes on disk is refused at pin
    # time, a missing member is refused, and an incomplete read set is
    # recorded as such.
    clock.advance(1)
    _submit(conn, clock, "r2", "reader")
    second = _claim(conn, clock, supervisor)
    with pytest.raises(OpsError, match="INPUT_CHANGED"):
        pin_read_set(conn, second.attempt_id,
                     _manifest(pinned, "data/x.bin", content_hash_override="sha256:" + "0" * 64),
                     prod)
    with pytest.raises(OpsError, match="INPUT_CHANGED"):
        pin_read_set(conn, second.attempt_id, _manifest(pinned, "data/missing.bin"), prod)
    pin_read_set(conn, second.attempt_id, _manifest(pinned, "data/x.bin", complete=False), prod)
    assert read_set_complete(conn, second.attempt_id) is False


@dataclass(frozen=True)
class BoundParameters:
    expected_ids: tuple[str, ...]
    input_bindings: dict[str, str] | None = None


BOUND_KIND = JobKind(name="bounded_reader", worker="artifact_check", parameters=BoundParameters,
                     resource_classes=frozenset({"delivery"}), effects=("staged",),
                     retry=RetryPolicy("bounded", 1, ()), checkpoint_contract="receipt.v1.0",
                     namespaces=frozenset({"shadow"}), store_domains=(("legacy_store", "read"),))
BOUND_REGISTRY = KindRegistry([BOUND_KIND])


def _run_supervisor(root, *, mutate, complete):
    """One supervised job against a synthetic production root; tick boundaries
    separate launch (which pins) from finish (which confirms)."""
    root.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    prod = root / "prod"
    (prod / "data").mkdir(parents=True)
    pinned = prod / "data" / "x.bin"
    pinned.write_bytes(b"pinned")

    clock = SystemClock()
    conn = open_catalog(root / "ops.sqlite", clock=clock)
    store = ArtifactStore(root)
    ref = store.publish_bytes(
        json.dumps(_manifest(pinned, "data/x.bin", complete=complete), sort_keys=True).encode(),
        schema_ref="legacy_input_manifest.v1.0")
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    job = submit(conn, BOUND_REGISTRY, POLICY, request(
        "bounded", kind="bounded_reader", checkpoint_contract_ref="receipt.v1.0",
        parameters={"expected_ids": ["a"],
                    "input_bindings": {"legacy_manifest.json": ref.artifact_id}},
        input_refs=(ref.artifact_id,),
        implementation_ref=content_hash(worker_source_manifest(repo)),
        environment_ref=content_hash(environment_identity(1))), clock=clock)

    service = Service(conn, root, BOUND_REGISTRY, TEST_POLICY, clock=clock,
                      code_source=repo, store_root=prod)
    try:
        service.start()
        if service.tick() is not True:  # claimed and launched, read set pinned
            AdmissionWatch(conn, job.job_id).check(final=True)  # RESOURCE WAIT, if that is why
            pytest.fail("first tick did not claim the job")
        attempt_id = conn.execute("SELECT attempt_id FROM attempts").fetchone()[0]
        assert conn.execute("SELECT read_set_complete FROM store_read_pins WHERE attempt_id=?",
                            (attempt_id,)).fetchone()[0] == int(complete)
        if mutate:
            pinned.write_bytes(b"mutated while the worker ran")
        run_until(service, conn, job.job_id, timeout=90)
        row = conn.execute("SELECT state, failure_json FROM jobs WHERE job_id=?",
                           (job.job_id,)).fetchone()
        return {"state": row[0], "failure": row[1],
                "checkpoints": conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0],
                "outputs": conn.execute("SELECT COUNT(*) FROM attempt_outputs").fetchone()[0],
                "leases_open": conn.execute("SELECT COUNT(*) FROM store_leases "
                                            "WHERE released_at IS NULL").fetchone()[0]}
    finally:
        service.close()
        conn.close()


def test_o17_supervisor_blocks_commit_when_pinned_input_changed(tmp_path):
    result = _run_supervisor(tmp_path / "case1", mutate=True, complete=True)
    assert result["state"] == "failed"
    assert "INPUT_CHANGED" in result["failure"]
    assert result["checkpoints"] == 0
    assert result["outputs"] == 0
    assert result["leases_open"] == 0


def test_o17_supervisor_commits_when_pinned_input_holds(tmp_path):
    result = _run_supervisor(tmp_path / "case2", mutate=False, complete=True)
    assert result["state"] == "succeeded"
    assert result["checkpoints"] == 1
    assert result["outputs"] == 1


def test_o17_incomplete_read_set_disables_cache_reuse(tmp_path):
    result = _run_supervisor(tmp_path / "case3", mutate=False, complete=False)
    assert result["state"] == "succeeded"
    assert result["checkpoints"] == 0
    assert result["outputs"] == 1
