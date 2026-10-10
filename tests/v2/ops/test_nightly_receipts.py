"""Real-file, real-catalog tests for step receipts and reconciliation (slice 2 of #564).

Jobs are submitted through the real catalog; the one seeded layer is the terminal job
state, set with SQL because driving a worker to completion is not what is under test.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from engine.v2.ops import nightly_receipts as nr
from engine.v2.ops import nightly_session as ns
from engine.v2.ops.errors import OpsError
from engine.v2.ops.submission import job_id_for, submit
from tests.ops_support import POLICY, REGISTRY, catalog, request

IDENT = ns.SessionIdentity("2026-10-08", "shadow", "sel-1", "cat-1")
DIGEST = "digest-a"


def _code(excinfo) -> str:
    return excinfo.value.code


@pytest.fixture
def root(tmp_path):
    ns.ensure_session(tmp_path, IDENT)
    ns.mark_started(tmp_path, IDENT, 1)
    return tmp_path


def _receipts_file(root, generation=1):
    run_id = ns.load_session(root, IDENT).generations[generation - 1].run_id
    return root.joinpath(*ns.STATE_DIR, f"{run_id}.receipts.json")


def _artifact(tmp_path, data=b"payload"):
    path = tmp_path / "out.bin"
    return path, nr.Effect("artifact", str(path), hashlib.sha256(data).hexdigest())


@pytest.fixture
def ops(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    yield conn, clock
    conn.close()


def test_begin_is_idempotent_and_refuses_a_changed_request(root, tmp_path):
    _, effect = _artifact(tmp_path)
    first = nr.begin_step(root, IDENT, 1, "capture", DIGEST, effect)
    assert first.state == "intent"
    assert nr.begin_step(root, IDENT, 1, "capture", DIGEST, effect) == first
    with pytest.raises(OpsError) as exc:
        nr.begin_step(root, IDENT, 1, "capture", "digest-b", effect)
    assert _code(exc) == "IDEMPOTENCY_CONFLICT"
    with pytest.raises(OpsError) as exc:
        nr.reconcile_step(root, IDENT, 1, "capture", "digest-b")
    assert _code(exc) == "IDEMPOTENCY_CONFLICT"


@pytest.mark.parametrize("step,digest,effect", [
    ("", DIGEST, nr.Effect("external")), ("s", " ", nr.Effect("external")),
    ("s", DIGEST, nr.Effect("bogus", "x")), ("s", DIGEST, nr.Effect("catalog_job")),
    ("s", DIGEST, nr.Effect("artifact", "/x")), ("s", DIGEST, "not-an-effect"),
     ("s", DIGEST, nr.Effect("external", "x", "abc")),
    ("s", DIGEST, nr.Effect("artifact", "relative/out.bin", "abc"))])
def test_malformed_step_or_effect_is_invalid_request(root, step, digest, effect):
    with pytest.raises(OpsError) as exc:
        nr.begin_step(root, IDENT, 1, step, digest, effect)
    assert _code(exc) == "INVALID_REQUEST"
    assert not _receipts_file(root).exists()


def test_only_the_active_started_generation_may_record(tmp_path):
    ns.ensure_session(tmp_path, IDENT)
    with pytest.raises(OpsError) as exc:  # allocated, not started
        nr.begin_step(tmp_path, IDENT, 1, "s", DIGEST, nr.Effect("external"))
    assert _code(exc) == "CHECKPOINT_INCOMPATIBLE"
    ns.mark_started(tmp_path, IDENT, 1)
    ns.request_rerun(tmp_path, IDENT, ("s",))
    ns.mark_started(tmp_path, IDENT, 2)
    with pytest.raises(OpsError) as exc:  # superseded generation
        nr.begin_step(tmp_path, IDENT, 1, "s", DIGEST, nr.Effect("external"))
    assert _code(exc) == "CHECKPOINT_INCOMPATIBLE"
    with pytest.raises(OpsError) as exc:
        nr.reconcile_step(tmp_path, IDENT, True, "s", DIGEST)
    assert _code(exc) == "CHECKPOINT_INCOMPATIBLE"


def test_artifact_crash_between_publish_and_receipt_is_adopted(root, tmp_path):
    path, effect = _artifact(tmp_path)
    assert nr.reconcile_step(root, IDENT, 1, "capture", DIGEST).outcome == "not_started"
    nr.begin_step(root, IDENT, 1, "capture", DIGEST, effect)
    assert nr.reconcile_step(root, IDENT, 1, "capture", DIGEST).outcome == "not_started"
    path.write_bytes(b"payload")  # the effect happened; the process died before complete_step
    adopted = nr.reconcile_step(root, IDENT, 1, "capture", DIGEST)
    assert adopted.outcome == "completed" and adopted.receipt.state == "succeeded"
    assert json.loads(_receipts_file(root).read_text())["steps"]["capture"]["state"] == "succeeded"
    assert nr.complete_step(root, IDENT, 1, "capture") == adopted.receipt
    assert nr.reconcile_step(root, IDENT, 1, "capture", DIGEST).outcome == "completed"


def test_artifact_with_other_bytes_is_uncertain_and_writes_nothing(root, tmp_path):
    path, effect = _artifact(tmp_path)
    nr.begin_step(root, IDENT, 1, "capture", DIGEST, effect)
    path.write_bytes(b"someone else")
    before = _receipts_file(root).read_bytes()
    for _ in range(2):  # still uncertain on every call: never auto-cleared or repeated
        with pytest.raises(OpsError) as exc:
            nr.reconcile_step(root, IDENT, 1, "capture", DIGEST)
        assert _code(exc) == "CHECKPOINT_INCOMPATIBLE"
    with pytest.raises(OpsError) as exc:
        nr.complete_step(root, IDENT, 1, "capture")
    assert _code(exc) == "INVALID_REQUEST"
    assert _receipts_file(root).read_bytes() == before


def test_succeeded_artifact_that_changes_is_an_integrity_failure(root, tmp_path):
    path, effect = _artifact(tmp_path)
    path.write_bytes(b"payload")
    nr.begin_step(root, IDENT, 1, "capture", DIGEST, effect)
    nr.complete_step(root, IDENT, 1, "capture")
    path.write_bytes(b"tampered")
    with pytest.raises(OpsError) as exc:
        nr.reconcile_step(root, IDENT, 1, "capture", DIGEST)
    assert _code(exc) == "INTEGRITY_FAILED"
    assert nr.begin_step(root, IDENT, 1, "capture", DIGEST, effect).state == "succeeded"


def test_catalog_job_reconciles_from_the_real_catalog(root, ops):
    conn, clock = ops
    job_id = job_id_for("shadow", "import-1")
    effect = nr.Effect("catalog_job", job_id)
    nr.begin_step(root, IDENT, 1, "import", DIGEST, effect)
    assert nr.reconcile_step(root, IDENT, 1, "import", DIGEST, conn=conn).outcome == "not_started"
    submit(conn, REGISTRY, POLICY, request("import-1"), clock=clock)  # crash before the receipt
    for state in ("queued", "failed"):  # live or failed: uncertain, never resubmitted
        conn.execute("UPDATE jobs SET state = ? WHERE job_id = ?", (state, job_id))
        with pytest.raises(OpsError) as exc:
            nr.reconcile_step(root, IDENT, 1, "import", DIGEST, conn=conn)
        assert _code(exc) == "CHECKPOINT_INCOMPATIBLE"
        with pytest.raises(OpsError) as exc:
            nr.complete_step(root, IDENT, 1, "import", conn=conn)
        assert _code(exc) == "INVALID_REQUEST"
    conn.execute("UPDATE jobs SET state = 'succeeded' WHERE job_id = ?", (job_id,))
    done = nr.reconcile_step(root, IDENT, 1, "import", DIGEST, conn=conn)
    assert done.outcome == "completed" and done.receipt.effect == effect
    with pytest.raises(OpsError) as exc:  # the catalog is required to prove a job
        nr.reconcile_step(root, IDENT, 1, "import", DIGEST)
    assert _code(exc) == "INVALID_REQUEST"


def test_succeeded_job_missing_from_catalog_is_an_integrity_failure(root, ops):
    conn, clock = ops
    submit(conn, REGISTRY, POLICY, request("import-2"), clock=clock)
    job_id = job_id_for("shadow", "import-2")
    conn.execute("UPDATE jobs SET state = 'succeeded' WHERE job_id = ?", (job_id,))
    nr.begin_step(root, IDENT, 1, "import", DIGEST, nr.Effect("catalog_job", job_id))
    nr.complete_step(root, IDENT, 1, "import", conn=conn)
    conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
    with pytest.raises(OpsError) as exc:
        nr.reconcile_step(root, IDENT, 1, "import", DIGEST, conn=conn)
    assert _code(exc) == "INTEGRITY_FAILED"


def test_external_effect_is_uncertain_until_the_caller_binds_it(root):
    nr.begin_step(root, IDENT, 1, "refresh", DIGEST, nr.Effect("external", "price-refresh"))
    with pytest.raises(OpsError) as exc:
        nr.reconcile_step(root, IDENT, 1, "refresh", DIGEST)
    assert _code(exc) == "CHECKPOINT_INCOMPATIBLE"
    nr.complete_step(root, IDENT, 1, "refresh")
    assert nr.reconcile_step(root, IDENT, 1, "refresh", DIGEST).outcome == "completed"


def test_complete_without_intent_is_invalid_request(root):
    with pytest.raises(OpsError) as exc:
        nr.complete_step(root, IDENT, 1, "never-begun")
    assert _code(exc) == "INVALID_REQUEST"


def test_a_rerun_generation_starts_with_no_receipts(root):
    nr.begin_step(root, IDENT, 1, "refresh", DIGEST, nr.Effect("external"))
    nr.complete_step(root, IDENT, 1, "refresh")
    ns.request_rerun(root, IDENT, ("refresh",))
    ns.mark_started(root, IDENT, 2)
    assert nr.reconcile_step(root, IDENT, 2, "refresh", DIGEST).outcome == "not_started"
    assert _receipts_file(root, 1).exists() and not _receipts_file(root, 2).exists()


def test_unreadable_or_foreign_receipt_files_stop(root):
    nr.begin_step(root, IDENT, 1, "refresh", DIGEST, nr.Effect("external"))
    path = _receipts_file(root)
    good = json.loads(path.read_text())
    cases = [("{not json", "INTEGRITY_FAILED"),
             (json.dumps({**good, "run_id": "other"}), "INTEGRITY_FAILED"),
             (json.dumps({**good, "steps": {"x": {"state": "weird", "request_digest": "d",
                                                  "effect": {"kind": "external", "ref": "",
                                                             "sha256": ""}}}}),
              "INTEGRITY_FAILED"),
             (json.dumps({**good, "schema": "nightly_receipts.v0"}), "CHECKPOINT_INCOMPATIBLE")]
    for text, code in cases:
        path.write_text(text)
        with pytest.raises(OpsError) as exc:
            nr.reconcile_step(root, IDENT, 1, "refresh", DIGEST)
        assert _code(exc) == code
        with pytest.raises(OpsError):  # never treated as absent: begin does not overwrite it
            nr.begin_step(root, IDENT, 1, "other", DIGEST, nr.Effect("external"))
        assert path.read_text() == text


def test_no_session_state_refuses_every_call(tmp_path):
    with pytest.raises(OpsError) as exc:
        nr.reconcile_step(tmp_path, IDENT, 1, "s", DIGEST)
    assert _code(exc) == "CHECKPOINT_INCOMPATIBLE"


def test_begin_refuses_a_changed_effect_under_the_same_digest(root, tmp_path):
    _, effect = _artifact(tmp_path)
    nr.begin_step(root, IDENT, 1, "capture", DIGEST, effect)
    _, other = _artifact(tmp_path, b"other bytes")
    with pytest.raises(OpsError) as exc:
        nr.begin_step(root, IDENT, 1, "capture", DIGEST, other)
    assert _code(exc) == "IDEMPOTENCY_CONFLICT"


@pytest.mark.parametrize("step", ["", " ", None, ["x"]])
def test_malformed_step_is_refused_by_complete_and_reconcile_without_a_receipt(root, step):
    for call in (lambda: nr.complete_step(root, IDENT, 1, step),
                 lambda: nr.reconcile_step(root, IDENT, 1, step, DIGEST)):
        with pytest.raises(OpsError) as exc:
            call()
        assert _code(exc) == "INVALID_REQUEST"
    assert not _receipts_file(root).exists()