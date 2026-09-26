"""P6-3 ``decisions-supersede`` through the real supervisor path.

Submits the CLI's own ``decisions_supersede`` job into a tmp catalog, runs a
real ``Service`` (the trivial worker subprocess plus the real coordinator
effect) with a fixed capacity sample, and asserts the committed catalog row:
a ``kind="prediction"`` decision that ``catalog_reader.read_predictions``
resolves, that a lost fence before finish commits nothing, and that a second
identical supersede stays one row. No real ledger is ever opened.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import content_hash
from engine.v2.ledger.catalog_reader import read_predictions
from engine.v2.ledger.decisions import insert, rows, set_authority
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.cli import decisions_command, parser
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import submit
from engine.v2.ops.supervisor import LEASE_SECONDS, Service, serve
from tests.ops_support import POLICY, TEST_POLICY, FakeClock, sample

ROOT = Path(__file__).resolve().parents[1]
STAMP = "2026-09-12T00:00:00.000000Z"
OLD = "prediction:r1"
OLD_PAYLOAD = {"row_id": "r1", "event_id": "e1", "ticker": "r1"}
NEW = "prediction:r1b"
NEW_PAYLOAD = {"row_id": "r1b", "restated": True}
REASON = "restated by operator"

pytestmark = pytest.mark.xdist_group("serial")


@pytest.fixture(autouse=True)
def _fixed_capacity(monkeypatch):
    """Admission must not depend on the live host's free memory (the same fix
    ``tests/test_v2_ops_coordinator_lease.py`` applies)."""
    import engine.v2.ops.supervisor as supervisor_module
    monkeypatch.setattr(supervisor_module, "sample_capacity",
                        lambda root, *, clock: sample(clock))


def _service(tmp_path, clock):
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    service = Service(conn, tmp_path, registry(), TEST_POLICY, clock=clock, code_source=ROOT)
    return conn, service


def _seed(conn):
    with transaction(conn):
        set_authority(conn, None, "catalog", STAMP)
        insert(conn, logical_key=OLD, decision_id=OLD, payload=OLD_PAYLOAD,
               purpose="shadow", kind="prediction", validations={}, created_at=STAMP)


def _submit(tmp_path, conn, clock, *, row_id=OLD, reason=REASON):
    path = tmp_path / "new_payload.json"
    path.write_text(json.dumps(NEW_PAYLOAD, sort_keys=True))
    argv = ["decisions", "supersede", "--row-id", row_id, "--reason", reason,
            "--from-json", str(path)]
    return decisions_command(parser().parse_args(argv), tmp_path, conn, clock)


def _resubmit_identical(conn, clock):
    """The same supersession again under a fresh idempotency key, exactly as
    ``ops decisions supersede`` would build it."""
    profile = profile_named(DEFAULT_POLICY, "io_fetch")
    job = JobSpec(
        kind="decisions_supersede",
        implementation_ref=content_hash(worker_source_manifest(ROOT)),
        spec_hash=None,
        environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
        parameters={"expected_ids": ["decisions_supersede"], "old_decision_id": OLD,
                    "reason": REASON, "new_payload": dict(NEW_PAYLOAD)},
        input_refs=(), output_namespace="shadow", resource_class="io_fetch",
        retry_policy_ref="bounded", checkpoint_contract_ref="decisions_supersede_receipt.v1.0")
    return submit(conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key="retry", principal="operator", job=job), clock=clock)


def _job_state(conn, job_id):
    return conn.execute("SELECT state FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]


def test_supervisor_commits_and_resolves_the_superseding_prediction(tmp_path):
    clock = FakeClock()
    conn, service = _service(tmp_path, clock)
    _seed(conn)
    receipt = _submit(tmp_path, conn, clock)

    serve(service, once=True)

    assert _job_state(conn, receipt["job_id"]) == "succeeded"
    committed = {row["decision_id"]: row for row in rows(conn)}
    assert set(committed) == {OLD, NEW}
    new = committed[NEW]
    assert new["kind"] == "prediction"
    assert new["supersedes"] == OLD
    payload = json.loads(new["payload_json"])
    assert payload["supersedes"] == "r1"
    assert payload["supersede_reason"] == REASON
    assert [row["row_id"] for row in read_predictions(conn)] == ["r1b"]


def test_lost_fence_before_finish_leaves_no_row(tmp_path, monkeypatch):
    clock = FakeClock()
    conn, service = _service(tmp_path, clock)
    _seed(conn)
    _submit(tmp_path, conn, clock)
    real = Service._coordinator_effect

    def effect(self, claim, refs, launch=None, keepalive=None):
        result = real(self, claim, refs, launch, keepalive)
        clock.advance(LEASE_SECONDS * 4)
        return result

    monkeypatch.setattr(Service, "_coordinator_effect", effect)
    serve(service, once=True)

    assert [row["decision_id"] for row in rows(conn)] == [OLD]
    attempt = conn.execute("SELECT state FROM attempts").fetchone()
    assert attempt["state"] in ("recovery_pending", "failed")


def test_a_second_identical_supersede_commits_one_row(tmp_path):
    clock = FakeClock()
    conn, service = _service(tmp_path, clock)
    _seed(conn)
    first = _submit(tmp_path, conn, clock)
    serve(service, once=True)
    assert _job_state(conn, first["job_id"]) == "succeeded"

    _resubmit_identical(conn, clock)
    clock.advance(60)  # a fresh supervisor epoch id, distinct from the first serve's
    serve(Service(conn, tmp_path, registry(), TEST_POLICY, clock=clock, code_source=ROOT),
          once=True)

    superseding = [row for row in rows(conn) if row["supersedes"]]
    assert [row["decision_id"] for row in superseding] == [NEW]
    assert [row["decision_id"] for row in rows(conn)] == [OLD, NEW]
    assert [row["row_id"] for row in read_predictions(conn)] == ["r1b"]
