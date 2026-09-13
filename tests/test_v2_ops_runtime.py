"""Small subprocess and checkpoint fault tests."""
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from engine.v2.contracts import CheckpointCandidate, OutputCandidate, ProcessIdentity
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.ops.checkpoints import assemble, cache_identity, commit_checkpoint, reuse
from engine.v2.ops.errors import OpsError
from engine.v2.ops.executor_watchdog import observe
from engine.v2.ops.fingerprints import (
    environment_identity,
    rerun_plan,
    source_closure,
    worker_source_manifest,
)
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.provider_budget import before_request, configure_account, record_response, reserve
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import submit
from engine.v2.ops.supervisor import Service, serve
from tests.ops_support import POLICY, catalog, enqueue_claim, request


def test_o06_actual_child_affinity_threads_and_outputs(tmp_path):
    conn, _, _ = catalog(tmp_path)
    clock = SystemClock()
    spec = request(kind="artifact_check", checkpoint_contract_ref="receipt.v1.0",
                   parameters={"expected_ids": ["a", "b"]},
                   implementation_ref=content_hash(worker_source_manifest(
                       Path(__file__).resolve().parents[1])),
                   environment_ref=content_hash(environment_identity(1)))
    job = submit(conn, registry(), POLICY, spec, clock=clock)
    service = Service(conn, tmp_path, registry(), DEFAULT_POLICY, clock=clock,
                      code_source=Path(__file__).resolve().parents[1])
    serve(service, once=True)
    row = conn.execute("SELECT state FROM jobs WHERE job_id = ?", (job.job_id,)).fetchone()
    assert row[0] == "succeeded"
    assert conn.execute("SELECT COUNT(*) FROM attempt_outputs").fetchone()[0] == 1
    assert conn.execute("SELECT memory_peak_bytes FROM attempts").fetchone()[0] > 0
    assert conn.execute("SELECT released_at FROM resource_reservations").fetchone()[0]


def candidate(claim):
    fields = dict(kind=claim.spec.kind, inputs=content_hash(list(claim.spec.input_refs)),
                  implementation=claim.spec.implementation_ref,
                  parameters=content_hash(claim.spec.parameters), environment=claim.spec.environment_ref,
                  schema="rows.v1.0", shard="a")
    return CheckpointCandidate(shard_key="a", cache_key=cache_identity(**fields),
                               input_hash=fields["inputs"], implementation_hash=fields["implementation"],
                               parameter_hash=fields["parameters"], environment_hash=fields["environment"],
                               output_schema_ref=fields["schema"],
                               outputs=(OutputCandidate(name="rows", staged_path="rows", schema_ref="rows.v1.0"),))


@pytest.mark.parametrize("point,committed", [("before_catalog", False), ("after_catalog", True)])
def test_o11_checkpoint_crash_and_reuse(tmp_path, point, committed):
    conn, clock, supervisor = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)
    store = ArtifactStore(tmp_path)
    (store.staging_dir(claim.attempt_id) / "rows").write_bytes(b"immutable")
    proposed = candidate(claim)
    def fault(observed):
        if observed == point:
            raise RuntimeError(point)
    with pytest.raises(RuntimeError):
        commit_checkpoint(conn, store, claim, proposed, clock=clock,
                          inputs_hash=proposed.input_hash, fault=fault)
    assert (reuse(conn, store, proposed.cache_key) is not None) is committed
    receipt = commit_checkpoint(conn, store, claim, proposed, clock=clock,
                                inputs_hash=proposed.input_hash)
    assert store.read_verified(receipt.artifact_refs[0]) == b"immutable"


def test_o12_transitive_source_and_diamond(tmp_path):
    (tmp_path / "a.py").write_text("import b\n")
    (tmp_path / "b.py").write_text("VALUE = 1\n")
    before = source_closure(tmp_path, ["a.py"])
    (tmp_path / "b.py").write_text("VALUE = 2\n")
    assert source_closure(tmp_path, ["a.py"]) != before
    (tmp_path / "docs.md").write_text("unrelated")
    after = source_closure(tmp_path, ["a.py"])
    (tmp_path / "docs.md").write_text("changed")
    assert source_closure(tmp_path, ["a.py"]) == after
    graph = {name: dict(inputs="i", implementation="v1", parameters="p", environment="e", schema="s",
                        dependencies=parents) for name, parents in
             [("A", []), ("B", ["A"]), ("C", ["A"]), ("D", ["B", "C"])]}
    previous = {key: dict(value) for key, value in graph.items()}
    graph["B"]["implementation"] = "v2"
    plan = rerun_plan(graph, previous)
    assert {key for key, row in plan.items() if row["action"] == "rerun"} == {"B", "D"}


def test_o13_missing_duplicate_and_empty_coverage():
    for batches in ([[{"row_id": "a"}]], [[{"row_id": "a"}, {"row_id": "a"}]]):
        with pytest.raises(OpsError, match="coverage"):
            assemble(["a", "b"], batches)
    with pytest.raises(OpsError, match="no-work"):
        assemble([], [])
    assert assemble([], [], no_work_receipt={"expected_count": 0, "reason": "no_eligible_inputs"}) == []


def test_o15_provider_lease_retry_headers_and_backoff(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    a = enqueue_claim(conn, clock, supervisor)
    b = enqueue_claim(conn, clock, supervisor, "two")
    configure_account(conn, "polygon", "generation", 10, 2)
    reserve(conn, a, "polygon", 3, clock=clock)
    with pytest.raises(OpsError, match="RESOURCE_UNAVAILABLE"):
        reserve(conn, b, "polygon", 3, clock=clock)
    before_request(conn, a, "polygon", clock=clock)
    assert record_response(conn, "polygon", 429, clock=clock, remaining=8) == "RATE_LIMITED"
    with pytest.raises(OpsError, match="RATE_LIMITED"):
        before_request(conn, a, "polygon", clock=clock)
    clock.advance(66)
    before_request(conn, a, "polygon", clock=clock)
    assert conn.execute("SELECT remaining,uncertain FROM provider_accounts").fetchone()[:] == (7, 1)
    assert record_response(conn, "polygon", 404, clock=clock) == "SOURCE_NOT_FOUND"
    assert record_response(conn, "polygon", 200, clock=clock, empty=True) == "SOURCE_EMPTY"
    assert record_response(conn, "polygon", 200, clock=clock, final=False) == "SOURCE_NOT_FINAL"
    assert record_response(conn, "polygon", 401, clock=clock) == "CREDENTIAL_INVALID"
    with pytest.raises(OpsError, match="CREDENTIAL_INVALID"):
        before_request(conn, a, "polygon", clock=clock)


def test_o07_pid_reuse_and_reparenting():
    parent = ProcessIdentity(boot_id="boot", pid=1, start_ticks=10, process_group=1)
    child = ProcessIdentity(boot_id="boot", pid=2, start_ticks=20, process_group=2)
    known, alive, memory = observe((parent,), "boot", table={1: (parent, 0, "S", 10), 2: (child, 1, "S", 20)})
    assert memory == 30 and len(alive) == 2
    reused = replace(parent, start_ticks=100)
    _, alive, memory = observe(known, "boot", table={1: (reused, 0, "S", 1000), 2: (child, 0, "S", 20)})
    assert alive == (child,) and memory == 20


def test_live_window_admission_is_conservative_about_unknown_durations():
    from dataclasses import replace
    from datetime import datetime, timezone

    from engine.v2.contracts import LiveWindow
    from engine.v2.ops.resources import live_window_reason

    window = LiveWindow(name="live_session", weekdays=(1,), start_utc="13:30", end_utc="20:00")
    policy = replace(DEFAULT_POLICY, live_windows=(window,))
    utc = timezone.utc
    before = datetime(2026, 9, 14, 13, 0, tzinfo=utc)   # Monday, 30 min before the window
    inside = datetime(2026, 9, 14, 15, 0, tzinfo=utc)
    after = datetime(2026, 9, 14, 21, 0, tzinfo=utc)

    short = replace(DEFAULT_POLICY.profiles[0], estimated_seconds=600)     # finishes 13:10
    long = replace(DEFAULT_POLICY.profiles[0], estimated_seconds=7200)     # finishes 15:00
    unknown_heavy = replace(DEFAULT_POLICY.profiles[0], heavy=True, estimated_seconds=None)
    unknown_light = replace(DEFAULT_POLICY.profiles[0], heavy=False, estimated_seconds=None)

    # A job that conservatively completes before the window is admitted.
    assert live_window_reason(policy, short, before) is None
    # A job whose conservative completion overlaps the window is refused with
    # the numbers, and never started on an assumption it will finish early.
    reason = live_window_reason(policy, long, before)
    assert reason is not None and reason.code == "LIVE_WINDOW"
    assert reason.needed["completion_seconds"] == 7200
    assert reason.available["seconds_until_window"] == 1800
    # Unknown-duration HEAVY work cannot be assumed safe before or inside the
    # window; unknown-duration light work may proceed before it.
    assert live_window_reason(policy, unknown_heavy, before) is not None
    assert live_window_reason(policy, unknown_heavy, inside) is not None
    assert live_window_reason(policy, unknown_light, before) is None
    # After the window everything is admitted again, and a malformed window
    # fails closed rather than open.
    assert live_window_reason(policy, unknown_heavy, after) is None
    broken = replace(policy, live_windows=(replace(window, start_utc="99:99"),))
    assert live_window_reason(broken, short, before).code == "INVALID_LIVE_WINDOW"
