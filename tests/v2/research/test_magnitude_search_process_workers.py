"""Prove serial and process dispatch leave the same durable scheduler record contract."""
import multiprocessing
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import pytest

from engine.v2.foundation import SystemClock
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.migrations import migrate
from experiments import v2_magnitude_search_run as driver
from experiments.magnitude_search_records import (
    MIGRATIONS,
    OWNER,
    append_record,
    read_record,
)

SELECTION = {"selections": [{"outer_year": 2024}]}
_failure = fail


def _append_same_slot(task):
    conn_path, store_root, barrier = task
    barrier.wait()
    conn = driver.connect(conn_path, must_exist=True)
    conn.execute("PRAGMA busy_timeout=30000")
    store = driver.ArtifactStore(store_root)
    try:
        return append_record(conn, store, "run", "same-slot", {"status": "committed"})
    finally:
        conn.close()


def _run(monkeypatch, tmp_path, workers, *, fail=False, crash=False, parent_memory=100,
         schema="magnitude_search_request.v2", include_workers=True):
    with multiprocessing.Manager() as manager:
        records = manager.dict()
        pids = manager.list()
        initializers = manager.list()
        parent_pid = os.getpid()
        captured_selection = {}
        plan = {
            "schema_version": "magnitude_search_plan.v2",
            "plan_id": "plan",
            "folds": [{"outer": {"year": 2024}, "inner": [{"year": 2023}]}],
            "candidates": [{"candidate_id": "candidate"}],
        }
        registration = SimpleNamespace(run_id="run")
        store = SimpleNamespace(root=tmp_path)
        conn = SimpleNamespace(execute=lambda sql: SimpleNamespace(
            fetchone=lambda: (0, "main", str(tmp_path / "catalog.sqlite"))))
        request = {
            "schema_version": schema,
            "limits": {
                "child": {"memory_bytes": 10, "cpu_seconds": 1, "wall_seconds": 1},
                "parent": {"memory_bytes": parent_memory},
            },
        }
        if include_workers:
            request["workers"] = workers

        def key(*args):
            return ":".join(map(str, args))

        def trial_key(_plan, *args):
            return key(*args)

        class Trials:
            def __init__(self, records, captured_selection, should_fail, crash, pids, initializers,
                         parent_pid):
                self.records = records
                self.captured_selection = captured_selection
                self.should_fail = should_fail
                self.crash = crash
                self.pids = pids
                self.initializers = initializers
                self.parent_pid = parent_pid

            def execute(self, *args, limits):
                trial = key(*args)
                self.records["start:" + trial] = "start:" + trial
                self.records["execution-start:" + trial] = "execution-start:" + trial
                pid = os.getpid()
                self.pids.append(pid)
                if self.should_fail:
                    if self.crash and pid != self.parent_pid:
                        os._exit(23)
                    raise _failure("RESOURCE_UNAVAILABLE", "synthetic worker failure")
                self.records["execution-result:" + trial] = "execution-result:" + trial
                self.records["execution:" + trial] = "execution:" + trial
                self.records["terminal:" + trial] = "terminal:" + trial
                return {"trial_id": trial, "status": "succeeded"}

            def _execute_borrowed(self, *args, limits):
                return self.execute(*args, limits=limits)

            def reconcile(self, *args, limits):
                return self.execute(*args, limits=limits)

            def freeze(self):
                self.captured_selection["value"] = SELECTION
                return SELECTION

        def initializer(*args):
            assert "workers" not in args[3]
            assert args[5] == 10
            initializers.append(os.getpid())
            driver._PARALLEL = (
                conn, store,
                Trials(records, captured_selection, fail, crash, pids, initializers, parent_pid),
                registration, plan)

        def admit_driver(*args):
            assert "workers" not in args[3]
            return plan

        monkeypatch.setattr(driver, "_admit_driver", admit_driver)
        monkeypatch.setattr(driver, "read_record",
                            lambda conn, store, run_id, slot: records.get(slot))
        monkeypatch.setattr(driver, "trial_key", trial_key)
        monkeypatch.setattr(driver, "execute_outer_fold", lambda *args, **kwargs: None)
        monkeypatch.setattr(driver, "collect_outer_results", lambda *args: {
            "records": sorted(records.items()),
            "selection": captured_selection["value"],
        })
        monkeypatch.setattr(driver.os, "sched_getaffinity", lambda pid: {0, 1, 2})
        monkeypatch.setattr(driver.os, "sched_setaffinity", lambda pid, cpus: None)
        monkeypatch.setattr(driver, "get_context",
                            lambda *args, **kwargs: multiprocessing.get_context("fork"))

        def make_trials(*args, **kwargs):
            assert "workers" not in args[4]
            return Trials(records, captured_selection, fail, crash, pids, initializers,
                          parent_pid)

        monkeypatch.setattr(driver, "SearchTrials", make_trials)
        monkeypatch.setattr(driver, "_init_trial_worker", initializer)

        result, codes = None, []
        for _ in range(2 if fail else 1):
            try:
                result = driver.run_search(conn, store, None, registration, request, object(),
                                           limits=request["limits"]["child"], collect_results=True)
            except OpsError as error:
                codes.append(error.code)
                if not fail:
                    raise
        return result, dict(records), codes, list(pids), list(initializers), parent_pid


@pytest.mark.parametrize("workers", [1, 3])
def test_scheduler_records_do_not_depend_on_worker_count(monkeypatch, tmp_path, workers):
    result, records, codes, pids, initializers, parent_pid = _run(
        monkeypatch, tmp_path, workers)
    trial = "2024:2023:candidate"
    assert codes == []
    assert records == {
        "start:" + trial: "start:" + trial,
        "execution-start:" + trial: "execution-start:" + trial,
        "execution-result:" + trial: "execution-result:" + trial,
        "execution:" + trial: "execution:" + trial,
        "terminal:" + trial: "terminal:" + trial,
    }
    assert result == {"records": sorted(records.items()), "selection": SELECTION}
    if workers == 1:
        assert pids == [parent_pid]
        assert initializers == []
    else:
        assert len(initializers) == 3
        assert pids and pids[0] != parent_pid


@pytest.mark.parametrize("workers", [1, 3])
def test_worker_typed_error_semantics_match_serial(monkeypatch, tmp_path, workers):
    _, records, codes, pids, _, parent_pid = _run(monkeypatch, tmp_path, workers, fail=True)
    trial = "2024:2023:candidate"
    assert codes == ["RESOURCE_UNAVAILABLE", "EXPERIMENT_IDENTITY_CONFLICT"]
    assert "start:" + trial in records
    assert "execution-start:" + trial in records
    assert "execution-result:" + trial not in records
    assert "terminal:" + trial not in records
    if workers == 1:
        assert set(pids) == {parent_pid}
    else:
        assert pids and pids[0] != parent_pid


def test_broken_worker_process_refuses_resource_unavailable(monkeypatch, tmp_path):
    _, records, codes, pids, _, parent_pid = _run(
        monkeypatch, tmp_path, 3, fail=True, crash=True)
    trial = "2024:2023:candidate"
    assert codes == ["RESOURCE_UNAVAILABLE", "EXPERIMENT_IDENTITY_CONFLICT"]
    assert records["start:" + trial] == "start:" + trial
    assert pids and pids[0] != parent_pid


@pytest.mark.parametrize("workers", [0, -1, 9, True, 1.5])
def test_invalid_worker_counts_are_rejected(monkeypatch, tmp_path, workers):
    with pytest.raises(OpsError) as error:
        _run(monkeypatch, tmp_path, workers)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert error.value.problem.retryable is False


@pytest.mark.parametrize("schema", [
    "magnitude_search_request.v1", "regression_search_request.v1"])
def test_legacy_declared_workers_refuse_before_admission(monkeypatch, tmp_path, schema):
    with pytest.raises(OpsError) as error:
        _run(monkeypatch, tmp_path, 3, schema=schema)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert error.value.problem.retryable is False
    assert error.value.problem.message == "stage-2 workers require a v2 or v3 search request"


@pytest.mark.parametrize("schema", [
    "magnitude_search_request.v1", "regression_search_request.v1"])
def test_legacy_omitted_workers_default_to_one(monkeypatch, tmp_path, schema):
    result, records, codes, pids, initializers, parent_pid = _run(
        monkeypatch, tmp_path, 3, schema=schema, include_workers=False)
    assert codes == []
    assert initializers == []
    assert pids == [parent_pid]
    assert result["selection"] == SELECTION
    assert "terminal:2024:2023:candidate" in records


def test_worker_cpu_bound_refuses_before_pool_start(monkeypatch, tmp_path):
    with pytest.raises(OpsError) as error:
        _run(monkeypatch, tmp_path, 4)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert error.value.problem.message == "search worker count exceeds CPU or memory bounds"


def test_worker_memory_bound_equality_is_admitted(monkeypatch, tmp_path):
    result, records, codes, _, initializers, _ = _run(
        monkeypatch, tmp_path, 3, parent_memory=30)
    assert codes == []
    assert len(initializers) == 3
    assert result["selection"] == SELECTION
    assert "terminal:2024:2023:candidate" in records


@pytest.mark.parametrize(("workers", "parent_memory"), [(3, 29), (3, 0)])
def test_invalid_parent_memory_is_rejected(monkeypatch, tmp_path, workers, parent_memory):
    with pytest.raises(OpsError) as error:
        _run(monkeypatch, tmp_path, workers, parent_memory=parent_memory)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert error.value.problem.retryable is False


def test_worker_initializer_reopens_file_backed_inputs(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.sqlite"
    sqlite3.connect(catalog).close()
    store_root = tmp_path / "store"
    store_root.mkdir()
    original = driver.connect(str(catalog), must_exist=True)
    plan = {"plan_id": "plan"}
    watchdogs = []
    monkeypatch.setattr(driver, "read_search_plan", lambda *args: plan)
    monkeypatch.setattr(driver, "start_rss_watchdog", lambda value: watchdogs.append(value))
    registration = SimpleNamespace(run_id="run", document={})
    driver._init_trial_worker(str(catalog), str(store_root), registration, {}, object(), 1024)
    worker_conn, worker_store, worker_trials, worker_registration, admitted_plan = driver._PARALLEL
    try:
        assert watchdogs == [1024]
        assert worker_conn is not original
        assert worker_conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        assert worker_store.root == store_root
        assert worker_registration is registration
        assert admitted_plan == plan
        assert worker_trials.conn is worker_conn
        assert worker_trials.store is worker_store
    finally:
        worker_conn.close()
        original.close()


def test_append_same_slot_across_spawn_workers(tmp_path):
    catalog = tmp_path / "catalog.sqlite"
    conn = sqlite3.connect(catalog)
    conn.execute("CREATE TABLE experiment_runs(run_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO experiment_runs(run_id) VALUES (?)", ("run",))
    conn.commit()
    conn.close()

    conn = driver.connect(str(catalog), must_exist=True)
    migrate(conn, OWNER, MIGRATIONS, clock=SystemClock())
    conn.close()

    store_root = tmp_path / "store"
    store_root.mkdir()
    with multiprocessing.Manager() as manager:
        barrier = manager.Barrier(2)
        with ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            first = executor.submit(_append_same_slot, (str(catalog), str(store_root), barrier))
            second = executor.submit(_append_same_slot, (str(catalog), str(store_root), barrier))
            bodies = [first.result(), second.result()]

    assert bodies == [{"status": "committed"}, {"status": "committed"}]
    parent = driver.connect(str(catalog), must_exist=True)
    try:
        store = driver.ArtifactStore(str(store_root))
        assert read_record(parent, store, "run", "same-slot") == {"status": "committed"}
    finally:
        parent.close()
