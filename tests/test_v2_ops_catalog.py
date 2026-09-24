"""O01-O05, O08-O10, O19 transaction kernel and O32 migrations."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from engine.v2.contracts import CapacitySample
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import backup_to, integrity_errors, transaction
from engine.v2.ops.errors import OpsError, make_problem
from engine.v2.ops.lifecycle import Outcome, commit_attempt, complete_cancel, request_cancel
from engine.v2.ops.migrations import Migration, migrate
from engine.v2.ops.profiles import DEFAULT_POLICY, GIB, profile_named
from engine.v2.ops.recovery import expire_leases, reconcile_attempt
from engine.v2.ops.resources import headroom_bytes
from engine.v2.ops.scheduler import HEADROOM_CEILING_WINDOW_SECONDS, claim_next
from engine.v2.ops.submission import get_job, job_id_for, submit, submit_graph
from tests.ops_support import POLICY, REGISTRY, FakeClock, catalog, enqueue_claim, request, sample


def test_o01_concurrent_idempotency(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    def insert():
        other = open_catalog(tmp_path / "ops.sqlite", clock=clock)
        try:
            return submit(other, REGISTRY, POLICY, request(), clock=clock).job_id
        finally:
            other.close()
    with ThreadPoolExecutor(2) as pool:
        assert len(set(pool.map(lambda _: insert(), range(2)))) == 1
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    with pytest.raises(OpsError, match="IDEMPOTENCY_CONFLICT"):
        submit(conn, REGISTRY, POLICY, request(parameters={"value": 2}), clock=clock)


def test_o02_graph_and_authority(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    bad = [request(kind="arbitrary"), request(output_namespace="../escape"),
           request(dependency_job_ids=("unknown",)), replace(request(), namespace="production"),
           request(parameters={"argv": ["echo"]})]
    for item in bad:
        with pytest.raises(OpsError):
            submit(conn, REGISTRY, POLICY, item, clock=clock)
    a = request("a", dependency_job_ids=(job_id_for("shadow", "b"),))
    b = request("b", dependency_job_ids=(job_id_for("shadow", "a"),))
    with pytest.raises(OpsError, match="cycle"):
        submit_graph(conn, REGISTRY, POLICY, [a, b], clock=clock)
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_o03_concurrent_claims(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    for key in ("a", "b"):
        submit(conn, REGISTRY, POLICY, request(key), clock=clock)
    def claim():
        other = open_catalog(tmp_path / "ops.sqlite", clock=clock)
        try:
            return claim_next(other, policy=DEFAULT_POLICY, sample=sample(clock),
                              supervisor=supervisor, clock=clock)
        finally:
            other.close()
    with ThreadPoolExecutor(2) as pool:
        a, b = list(pool.map(lambda _: claim(), range(2)))
    assert a.job_id != b.job_id
    assert not set(a.resources.assigned_cpu_ids) & set(b.resources.assigned_cpu_ids)


def test_o04_o05_admission_and_no_undersized_retry(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    one = enqueue_claim(conn, clock, supervisor, resource_class="legacy_score")
    assert one is not None
    submit(conn, REGISTRY, POLICY, request("two", resource_class="legacy_score"), clock=clock)
    assert claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                      supervisor=supervisor, clock=clock) is None
    assert conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
    commit_attempt(conn, one.attempt_id, one.fence, Outcome(True, "verified_dead"), clock=clock)
    limited = replace(sample(clock), container_limit_bytes=3 << 30, container_current_bytes=1 << 30)
    assert claim_next(conn, policy=DEFAULT_POLICY, sample=limited,
                      supervisor=supervisor, clock=clock) is None
    queued = get_job(conn, job_id_for("shadow", "two"))
    assert queued.queue_reason.code == "PROFILE_EXCEEDS_CAPACITY"
    assert queued.attempt_count == 0


#: A sample where legacy_score's 5.25 GiB profile does NOT fit headroom
#: (host_available=5 GiB -> headroom ~4.5 GiB), but comfortably fits
#: capacity_bytes (host_total=8 GiB -> capacity 7 GiB) -- so only
#: MEMORY_HEADROOM can fire, never PROFILE_EXCEEDS_CAPACITY/RESERVATION_BUDGET.
def _low_sample(clock):
    return CapacitySample(sampled_at=clock.now().strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
                          allowed_cpu_ids=(1, 3, 5, 7, 9, 11), host_total_bytes=8 * GIB,
                          host_available_bytes=5 * GIB, container_limit_bytes=None,
                          container_current_bytes=None, swap_total_bytes=0, swap_free_bytes=0,
                          disk_free_bytes=100 * GIB, executor_mode="fake", containment="none")


#: Plenty of headroom for legacy_score (host_available=7 GiB -> headroom ~6.5 GiB).
def _enough_sample(clock):
    return replace(_low_sample(clock), host_available_bytes=7 * GIB)


def test_non_ops_headroom_dip_queues_then_admits(tmp_path):
    """§8.1 (2026-09-15, legacy_score v5 6 GiB incident follow-up): a single
    low sample with nothing active must NOT fail the job -- ``active`` only
    tracks ops-managed reservations, so one bad sample is indistinguishable
    from a non-ops process (another agent's test sweep) transiently holding
    memory. It stays a normal MEMORY_HEADROOM queue and admits as soon as a
    later sample shows enough headroom."""
    conn, clock, supervisor = catalog(tmp_path)
    submit(conn, REGISTRY, POLICY, request(resource_class="legacy_score"), clock=clock)
    assert claim_next(conn, policy=DEFAULT_POLICY, sample=_low_sample(clock),
                      supervisor=supervisor, clock=clock) is None
    job = get_job(conn, job_id_for("shadow", "one"))
    assert job.state == "queued"
    assert job.queue_reason.code == "MEMORY_HEADROOM"
    assert job.queue_reason.available["ceiling_window_samples"] == 1
    claimed = claim_next(conn, policy=DEFAULT_POLICY, sample=_enough_sample(clock),
                         supervisor=supervisor, clock=clock)
    assert claimed is not None
    assert get_job(conn, job_id_for("shadow", "one")).state == "running"


def test_sustained_headroom_shortage_fails_with_details(tmp_path):
    """The same low sample, continuously, with zero active reservations for
    the whole ``HEADROOM_CEILING_WINDOW_SECONDS`` window: nothing ops-tracked
    can be blamed, so this now fails outright (typed
    ``RESOURCE_PROFILE_UNSATISFIABLE``, details needed/max_headroom_observed/
    window_s/samples) instead of queuing forever."""
    conn, clock, supervisor = catalog(tmp_path)
    submit(conn, REGISTRY, POLICY, request(resource_class="legacy_score"), clock=clock)
    assert claim_next(conn, policy=DEFAULT_POLICY, sample=_low_sample(clock),
                      supervisor=supervisor, clock=clock) is None
    assert get_job(conn, job_id_for("shadow", "one")).queue_reason.code == "MEMORY_HEADROOM"
    clock.advance(HEADROOM_CEILING_WINDOW_SECONDS + 5)
    assert claim_next(conn, policy=DEFAULT_POLICY, sample=_low_sample(clock),
                      supervisor=supervisor, clock=clock) is None
    job = get_job(conn, job_id_for("shadow", "one"))
    assert job.state == "failed"
    assert job.attempt_count == 0
    assert job.queue_reason is None
    assert job.failure.code == "RESOURCE_PROFILE_UNSATISFIABLE"
    profile = profile_named(DEFAULT_POLICY, "legacy_score")
    assert job.failure.details["needed"]["memory_bytes"] == profile.memory_bytes
    assert job.failure.details["max_headroom_observed"] == headroom_bytes(DEFAULT_POLICY,
                                                                          _low_sample(clock))
    assert job.failure.details["window_s"] == HEADROOM_CEILING_WINDOW_SECONDS
    assert job.failure.details["samples"] == 2


def test_active_reservation_during_window_never_fails_and_resets_it(tmp_path):
    """An active reservation on ANY sample during the window -- even one that
    cannot itself explain the whole shortfall -- means nothing ops-tracked is
    ruled out, so that sample must not advance the window, and the window
    must not silently keep counting from before it once the active job is
    gone (§8.1: the failure must come from sustained, unexplained badness,
    never from wall-clock time that happened to elapse while something else
    was legitimately running)."""
    conn, clock, supervisor = catalog(tmp_path)
    # A lease long enough to survive advancing the clock past the window --
    # this test is about the memory window, not the unrelated lease timeout.
    supervisor = replace(supervisor, lease_seconds=10 * HEADROOM_CEILING_WINDOW_SECONDS)
    submit(conn, REGISTRY, POLICY, request("one", resource_class="legacy_score"), clock=clock)
    assert claim_next(conn, policy=DEFAULT_POLICY, sample=_low_sample(clock),
                      supervisor=supervisor, clock=clock) is None
    assert get_job(conn, job_id_for("shadow", "one")).queue_reason.available[
        "ceiling_window_samples"] == 1

    submit(conn, REGISTRY, POLICY, request("helper", resource_class="delivery"), clock=clock)
    helper = claim_next(conn, policy=DEFAULT_POLICY, sample=_low_sample(clock),
                        supervisor=supervisor, clock=clock)
    assert helper is not None and helper.job_id == job_id_for("shadow", "helper")

    clock.advance(HEADROOM_CEILING_WINDOW_SECONDS + 5)
    # "one" still does not fit, and wall-clock time is past the window, but
    # "helper"'s reservation is active -- must not fail.
    assert claim_next(conn, policy=DEFAULT_POLICY, sample=_low_sample(clock),
                      supervisor=supervisor, clock=clock) is None
    job = get_job(conn, job_id_for("shadow", "one"))
    assert job.state == "queued"
    assert job.queue_reason.code == "MEMORY_HEADROOM"
    assert "ceiling_window_started_epoch" not in job.queue_reason.available

    commit_attempt(conn, helper.attempt_id, helper.fence, Outcome(True, "verified_dead"),
                   clock=clock)
    # active is empty again; despite wall-clock time already far past the
    # window, this must start a FRESH window (reset, not resumed) and so
    # must not fail immediately.
    assert claim_next(conn, policy=DEFAULT_POLICY, sample=_low_sample(clock),
                      supervisor=supervisor, clock=clock) is None
    job = get_job(conn, job_id_for("shadow", "one"))
    assert job.state == "queued"
    assert job.queue_reason.available["ceiling_window_samples"] == 1


def test_legacy_score_fits_the_measured_ceiling_with_documented_margins():
    """A policy test, not a live probe: the ceiling comes from
    ``resources.headroom_bytes`` and ``DEFAULT_POLICY``'s own constants
    (``base_reserve_bytes``/``free_margin_bytes``), applied to two real
    capacity samples taken on this host 2026-09-15 (cited in profiles.py's
    v6 docstring) rather than any host number hard-coded into the
    assertion. The conservative (lower) sample is the attempt-17 queuing
    incident's ``free -b`` read (host_total=8162775040,
    host_available=6543368192) -- the moment legacy_score's old 6 GiB
    reservation queued forever with nothing else heavy running."""
    incident = CapacitySample(sampled_at="2026-09-15T00:00:00.000000Z",
                              allowed_cpu_ids=tuple(range(12)), host_total_bytes=8162775040,
                              host_available_bytes=6543368192, container_limit_bytes=None,
                              container_current_bytes=None, swap_total_bytes=0, swap_free_bytes=0,
                              disk_free_bytes=100 * GIB, executor_mode="watchdog",
                              containment="best_effort")
    ceiling = headroom_bytes(DEFAULT_POLICY, incident)
    profile = profile_named(DEFAULT_POLICY, "legacy_score")
    assert profile.memory_bytes < ceiling
    margin_under_ceiling = ceiling - profile.memory_bytes
    assert margin_under_ceiling > 100 * (1 << 20)  # documented ~199.5 MiB margin

    # Measured watchdog (tree-RSS) peaks, attempt-16 -- the same basis the
    # executor's own RESOURCE_LIMIT_EXCEEDED kill check uses.
    legacy_score_peak_bytes = 5447757824  # ~5.07 GiB
    legacy_score_requests_peak_bytes = int(4.44 * GIB)  # warm-cache, documented in profiles.py
    for peak in (legacy_score_peak_bytes, legacy_score_requests_peak_bytes):
        assert profile.memory_bytes > peak
    margin_over_score_peak = profile.memory_bytes - legacy_score_peak_bytes
    assert margin_over_score_peak > 100 * (1 << 20)  # documented ~180.6 MiB margin


def test_o08_o09_fence_and_recovery(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)
    clock.advance(121)
    assert expire_leases(conn, clock=clock) == [claim.attempt_id]
    with pytest.raises(OpsError, match="LEASE_LOST"):
        commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead"), clock=clock)
    assert reconcile_attempt(conn, claim.attempt_id, process_state="quarantined", clock=clock) == "recovery_pending"
    assert conn.execute("SELECT released_at FROM resource_reservations").fetchone()[0] is None
    reconcile_attempt(conn, claim.attempt_id, process_state="verified_dead", clock=clock)
    clock.advance(2)
    replacement = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                             supervisor=supervisor, clock=clock)
    assert replacement.attempt_number == 2 and replacement.fence > claim.fence


def test_o07_o10_cancellation_and_surviving_child(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)
    with pytest.raises(OpsError, match="reconciled"):
        commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "alive"), clock=clock)
    assert request_cancel(conn, claim.job_id, "stale", clock=clock).state == "conflict"
    assert request_cancel(conn, claim.job_id, claim.attempt_id, clock=clock).state == "cancelling"
    assert complete_cancel(conn, claim.job_id, process_state="alive", clock=clock).state == "cancelling"
    assert complete_cancel(conn, claim.job_id, process_state="verified_dead", clock=clock).state == "cancelled"


def test_o19_atomic_effects(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)
    def broken(db):
        db.execute("INSERT INTO health_observations VALUES ('night','test',1,'{}',0)")
        raise RuntimeError("crash")
    with pytest.raises(RuntimeError):
        commit_attempt(conn, claim.attempt_id, claim.fence, Outcome(True, "exited"),
                       clock=clock, effects=broken)
    assert conn.execute("SELECT COUNT(*) FROM health_observations").fetchone()[0] == 0
    assert get_job(conn, claim.job_id).state == "running"


def test_o32_migration_fault_and_backup(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    migration = Migration(1, "fault", ("CREATE TABLE planted(value INTEGER)",))
    def crash(point):
        raise RuntimeError(point)
    with pytest.raises(RuntimeError):
        migrate(conn, "test", [migration], clock=clock, fault=crash)
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='planted'").fetchone() is None
    migrate(conn, "test", [migration], clock=clock)
    with transaction(conn):
        conn.execute("INSERT INTO planted VALUES (17)")
    backup_to(conn, tmp_path / "backup.sqlite")
    with sqlite3.connect(tmp_path / "backup.sqlite") as restored:
        assert restored.execute("SELECT value FROM planted").fetchone()[0] == 17
    with pytest.raises(OpsError, match="newer"):
        migrate(conn, "test", [], clock=clock)


#: What ``integrity_errors`` must report for one orphaned
#: ``job_dependencies`` row: the finding names the CHILD table holding the
#: orphan (first element of the ``foreign_key_check`` row); the parent table
#: ``jobs`` is the third element, so the exact string pins that apart.
_ORPHAN_FINDING = "foreign key violation in job_dependencies"


def _plant_orphaned_dependency(conn, child_job_id):
    """A dependency row naming a parent that does not exist, written with
    enforcement off -- the state a catalog that lost a job row is found in."""
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("INSERT INTO job_dependencies (child_job_id, parent_job_id,"
                 " required_output_contract) VALUES (?, 'ghost', 'rows.v1.0')",
                 (child_job_id,))
    conn.execute("PRAGMA foreign_keys = ON")


def test_backup_to_carries_rows_committed_while_only_in_wal(tmp_path):
    """§6.1: the backup is made through the online backup API, so rows that
    were committed while still living only in the WAL are present in the
    copy, and a sound copy reports no integrity findings."""
    conn, clock, _ = catalog(tmp_path)
    ids = {submit(conn, REGISTRY, POLICY, request(key), clock=clock).job_id
           for key in ("one", "two")}
    backup_to(conn, tmp_path / "backup-copy.sqlite")
    with sqlite3.connect(tmp_path / "backup-copy.sqlite") as restored:
        assert integrity_errors(restored) == []
        assert {row[0] for row in restored.execute("SELECT job_id FROM jobs")} == ids


def test_integrity_errors_is_empty_when_sound_and_names_the_offending_table(tmp_path):
    """Empty when the catalog is sound; one finding per foreign-key
    violation, keyed by the child table that holds the orphaned row."""
    conn, clock, _ = catalog(tmp_path)
    assert integrity_errors(conn) == []
    child = submit(conn, REGISTRY, POLICY, request("one"), clock=clock).job_id
    _plant_orphaned_dependency(conn, child)
    assert integrity_errors(conn) == [_ORPHAN_FINDING]


def test_backup_to_refuses_a_catalog_with_an_orphaned_dependency(tmp_path):
    """Corruption is refused, not blessed: the refusal's check ran over the
    copy (which is left behind, orphan included, for an operator to
    inspect), and the failure carries INTEGRITY_FAILED with the first
    finding in structured details."""
    conn, clock, _ = catalog(tmp_path)
    child = submit(conn, REGISTRY, POLICY, request("one"), clock=clock).job_id
    _plant_orphaned_dependency(conn, child)
    with pytest.raises(OpsError) as err:
        backup_to(conn, tmp_path / "refused.sqlite")
    assert err.value.code == "INTEGRITY_FAILED"
    assert err.value.problem.details == {"first_problem": _ORPHAN_FINDING}
    with sqlite3.connect(tmp_path / "refused.sqlite") as refused:
        assert integrity_errors(refused) == [_ORPHAN_FINDING]
