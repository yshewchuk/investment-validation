"""Prove serial and process dispatch leave the same durable scheduler record contract."""
import multiprocessing
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import pytest

from engine.v2.data.errors import DataError, fail as data_fail
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


class _FailureLog(list):
    def __init__(self):
        super().__init__()
        self.details = []
        self.controls = []

    def record(self, error):
        self.append(error.code)
        self.details.append((type(error), error.problem))


def _append_same_slot(task):
    conn_path, store_root, barrier = task
    conn = driver.connect(conn_path, must_exist=True)
    conn.execute("PRAGMA busy_timeout=30000")
    store = driver.ArtifactStore(store_root)
    try:
        assert read_record(conn, store, "run", "same-slot") is None
        barrier.wait()
        return append_record(conn, store, "run", "same-slot", {"status": "committed"})
    finally:
        conn.close()


def _run(monkeypatch, tmp_path, workers, *, fail=False, crash=False, parent_memory=100,
         schema="magnitude_search_request.v2", include_workers=True, data_failure=False,
         guard_refusal=False, inner_years=(2023,), on_error=None):
    with multiprocessing.Manager() as manager:
        records = manager.dict()
        pids = manager.list()
        initializers = manager.list()
        parent_pid = os.getpid()
        captured_selection = {}
        plan = {
            "schema_version": "magnitude_search_plan.v2",
            "plan_id": "plan",
            "folds": [{"outer": {"year": 2024}, "inner": [{"year": year} for year in inner_years]}],
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
                         parent_pid, data_failure):
                self.records = records
                self.captured_selection = captured_selection
                self.should_fail = should_fail
                self.crash = crash
                self.pids = pids
                self.initializers = initializers
                self.parent_pid = parent_pid
                self.data_failure = data_failure

            def execute(self, *args, limits):
                trial = key(*args)
                self.records["start:" + trial] = "start:" + trial
                self.records["execution-start:" + trial] = "execution-start:" + trial
                pid = os.getpid()
                self.pids.append(pid)
                if self.data_failure:
                    raise data_fail("POPULATION_COLLAPSED", "synthetic training qualification failure")
                if self.should_fail:
                    if self.crash and pid != self.parent_pid:
                        os._exit(23)
                    raise _failure("RESOURCE_UNAVAILABLE", "synthetic worker failure")
                self.records["execution-result:" + trial] = "execution-result:" + trial
                self.records["execution:" + trial] = "execution:" + trial
                self.records["terminal:" + trial] = "terminal:" + trial
                return {"trial_id": trial, "status": "succeeded"}

            def _execute_borrowed(self, *args, limits):
                self.initializers.append(("borrowed", os.getpid(), key(*args)))
                return self.execute(*args, limits=limits)

            def reconcile(self, *args, limits):
                return self.execute(*args, limits=limits)

            def freeze(self):
                self.captured_selection["value"] = SELECTION
                return SELECTION

        def initializer(*args):
            assert "workers" not in args[3]
            assert args[5] == 10
            assert args[2] == "run"
            initializers.append(os.getpid())
            driver._PARALLEL = (
                conn, store,
                Trials(records, captured_selection, fail, crash, pids, initializers, parent_pid,
                       data_failure),
                registration, plan)

        def admit_driver(*args):
            assert "workers" not in args[3]
            return plan

        def v2_guard(where):
            control_calls.append(("v2", where))
            if guard_refusal:
                raise _failure("INVALID_EXPERIMENT_SPEC", "synthetic no-fit refusal")

        def legacy_guard(where):
            control_calls.append(("legacy", where))

        monkeypatch.setattr(driver, "forbid_fitting", v2_guard)
        monkeypatch.setattr(driver, "forbid_legacy_fitting", legacy_guard)
        monkeypatch.setattr(driver, "remaining", lambda *args: control_calls.append("deadline") or 60)
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
                          parent_pid, data_failure)

        monkeypatch.setattr(driver, "SearchTrials", make_trials)
        monkeypatch.setattr(driver, "_init_trial_worker", initializer)

        result, codes = None, _FailureLog()
        control_calls = codes.controls
        for _ in range(2 if fail else 1):
            try:
                result = driver.run_search(conn, store, None, registration, request, object(),
                                           limits=request["limits"]["child"], collect_results=True)
            except (OpsError, DataError) as error:
                codes.record(error)
                if on_error is not None:
                    on_error(dict(records), list(control_calls))
                if not fail:
                    raise
        return result, dict(records), codes, list(pids), list(initializers), parent_pid


@pytest.mark.parametrize("workers", [1, 3])
def test_scheduler_records_do_not_depend_on_worker_count(monkeypatch, tmp_path, workers):
    years = (2023, 2022, 2021, 2020, 2019)
    result, records, codes, pids, initializers, parent_pid = _run(
        monkeypatch, tmp_path, workers, inner_years=years)
    expected = {}
    for year in years:
        trial = f"2024:{year}:candidate"
        expected.update({name + trial: name + trial for name in
                         ("start:", "execution-start:", "execution-result:", "execution:", "terminal:")})
    assert codes == []
    assert records == expected
    assert result == {"records": sorted(records.items()), "selection": SELECTION}
    if workers == 1:
        assert pids == [parent_pid] * len(years)
        assert initializers == []
        assert codes.controls == []
    else:
        worker_pids = [value for value in initializers if type(value) is int]
        borrowed = [value for value in initializers if isinstance(value, tuple) and value[0] == "borrowed"]
        assert len(worker_pids) == workers
        assert {value[2] for value in borrowed} == {f"2024:{year}:candidate" for year in years}
        assert len(pids) == len(years) and all(pid != parent_pid for pid in pids)
        assert codes.controls == [("v2", "run_search"), ("legacy", "run_search"), "deadline"]


@pytest.mark.parametrize("workers", [1, 3])
def test_worker_typed_error_semantics_match_serial(monkeypatch, tmp_path, workers):
    _, records, codes, pids, _, parent_pid = _run(monkeypatch, tmp_path, workers, fail=True)
    trial = "2024:2023:candidate"
    expected_problem = _failure("RESOURCE_UNAVAILABLE", "synthetic worker failure").problem
    assert codes == ["RESOURCE_UNAVAILABLE", "EXPERIMENT_IDENTITY_CONFLICT"]
    assert codes.details[0] == (OpsError, expected_problem)
    assert "start:" + trial in records
    assert "execution-start:" + trial in records
    assert "execution-result:" + trial not in records
    assert "terminal:" + trial not in records
    if workers == 1:
        assert set(pids) == {parent_pid}
    else:
        assert pids and pids[0] != parent_pid


def test_worker_data_error_preserves_type_and_problem_from_serial(monkeypatch, tmp_path):
    serial = _run(monkeypatch, tmp_path / "serial", 1, fail=True, data_failure=True)
    parallel = _run(monkeypatch, tmp_path / "parallel", 3, fail=True, data_failure=True)
    assert serial[2] == parallel[2] == ["POPULATION_COLLAPSED", "EXPERIMENT_IDENTITY_CONFLICT"]
    assert serial[2].details == parallel[2].details
    assert serial[2].details[0][0] is DataError
    assert serial[2].details[0][1].message == "synthetic training qualification failure"
    assert serial[1] == parallel[1]


def test_broken_worker_process_refuses_resource_unavailable(monkeypatch, tmp_path):
    _, records, codes, pids, _, parent_pid = _run(
        monkeypatch, tmp_path, 3, fail=True, crash=True,
        inner_years=(2023, 2022, 2021, 2020, 2019))
    trial = "2024:2023:candidate"
    assert codes == ["RESOURCE_UNAVAILABLE", "EXPERIMENT_IDENTITY_CONFLICT"]
    assert records["start:" + trial] == "start:" + trial
    assert "start:2024:2020:candidate" not in records
    assert "start:2024:2019:candidate" not in records
    assert pids and pids[0] != parent_pid


@pytest.mark.parametrize("workers", [0, -1, 9, True, 1.5])
def test_invalid_worker_counts_are_rejected(monkeypatch, tmp_path, workers):
    with pytest.raises(OpsError) as error:
        _run(monkeypatch, tmp_path, workers)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert error.value.problem.retryable is False
    assert error.value.problem.message == "stage-2 workers requires an integer from 1 through 8"


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
    snapshots = []
    with pytest.raises(OpsError) as error:
        _run(monkeypatch, tmp_path, 4,
             on_error=lambda records, controls: snapshots.append((records, controls)))
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert error.value.problem.message == "search worker count exceeds CPU or memory bounds"
    assert snapshots == [({}, [])]


def test_worker_memory_bound_equality_is_admitted(monkeypatch, tmp_path):
    result, records, codes, _, initializers, _ = _run(
        monkeypatch, tmp_path, 3, parent_memory=30)
    assert codes == []
    assert sum(isinstance(value, int) for value in initializers) == 3
    assert result["selection"] == SELECTION
    assert "terminal:2024:2023:candidate" in records


@pytest.mark.parametrize(("workers", "parent_memory"), [(3, 29), (3, 0)])
def test_invalid_parent_memory_is_rejected(monkeypatch, tmp_path, workers, parent_memory):
    snapshots = []
    with pytest.raises(OpsError) as error:
        _run(monkeypatch, tmp_path, workers, parent_memory=parent_memory,
             on_error=lambda records, controls: snapshots.append((records, controls)))
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert error.value.problem.retryable is False
    expected = ("search worker memory bound is invalid" if parent_memory == 0
                else "search worker count exceeds CPU or memory bounds")
    assert error.value.problem.message == expected
    assert snapshots == [({}, [])]


def test_worker_initializer_reopens_file_backed_inputs(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.sqlite"
    sqlite3.connect(catalog).close()
    store_root = tmp_path / "store"
    store_root.mkdir()
    original = driver.connect(str(catalog), must_exist=True)
    plan = {"plan_id": "plan"}
    events = []
    monkeypatch.setattr(driver, "read_search_plan", lambda *args: plan)
    def fake_verified_registration_bytes(conn, store, run_id, **kwargs):
        events.append(("verify", run_id))
        return b"binding"
    def fake_load_registration_document(store, binding, **kwargs):
        events.append(("hydrate", binding))
        return {"schema_version": "native_experiment_registration.v1.7"}
    monkeypatch.setattr(driver, "start_rss_watchdog", lambda value: events.append(("watchdog", value)))
    monkeypatch.setattr(driver, "verified_registration_bytes", fake_verified_registration_bytes)
    monkeypatch.setattr(driver, "load_registration_document", fake_load_registration_document)
    monkeypatch.setattr(driver.signal, "signal", lambda *args: signal.SIG_DFL)
    driver._init_trial_worker(str(catalog), str(store_root), "run", {}, object(), 1024)
    worker_conn, worker_store, worker_trials, worker_registration, admitted_plan = driver._PARALLEL
    try:
        assert events[0] == ("watchdog", 1024)
        assert events[1] == ("verify", "run")
        assert events[2] == ("hydrate", b"binding")
        assert worker_conn is not original
        assert worker_conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        assert worker_store.root == store_root
        assert worker_registration.run_id == "run"
        assert worker_registration.binding_json == b"binding"
        assert worker_registration.document_value == {"schema_version": "native_experiment_registration.v1.7"}
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


def test_parallel_no_fit_guard_refuses_before_deadline_or_claim(monkeypatch, tmp_path):
    observed = []
    with pytest.raises(OpsError) as error:
        _run(monkeypatch, tmp_path, 3, guard_refusal=True,
             on_error=lambda records, controls: observed.append((records, controls)))
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert observed == [({}, [("v2", "run_search")])]


def test_parallel_preflight_refuses_in_memory_catalog(tmp_path):
    conn = sqlite3.connect(":memory:")
    try:
        request = {"limits": {"parent": {"memory_bytes": 30}}}
        with pytest.raises(OpsError) as error:
            driver._parallel_preflight(conn, request, {"memory_bytes": 10}, 2)
        assert error.value.code == "INVALID_EXPERIMENT_SPEC"
        assert error.value.problem.retryable is False
    finally:
        conn.close()


def _install_test_worker_termination_handler():
    signal.signal(signal.SIGTERM, driver._terminate_worker)


def _wait_for_child_with_reaping(task):
    pid_path, status_path = task
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        Path(pid_path).write_text(str(child.pid))
        child.wait()
    finally:
        if child.poll() is None:
            child.kill()
        status = child.wait()
        Path(status_path).write_text(str(status))


def _cancellable_child_task(task):
    try:
        _wait_for_child_with_reaping(task)
    except driver._WorkerTerminating:
        os._exit(128 + signal.SIGTERM)


def test_worker_termination_handler_raises_only_first_signal(monkeypatch):
    monkeypatch.setattr(driver, "_WORKER_TERMINATING", False)
    with pytest.raises(driver._WorkerTerminating):
        driver._terminate_worker(signal.SIGTERM, None)
    assert driver._WORKER_TERMINATING is True
    assert driver._terminate_worker(signal.SIGTERM, None) is None


def test_pool_abort_waits_for_worker_to_reap_active_fit_child(tmp_path):
    pid_path = tmp_path / "fit-child.pid"
    status_path = tmp_path / "fit-child.status"
    executor = ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn"),
        initializer=_install_test_worker_termination_handler)
    stopped = False
    try:
        executor.submit(_cancellable_child_task, (str(pid_path), str(status_path)))
        deadline = time.monotonic() + 10
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_path.exists(), "worker did not start its fit child"
        queued = [executor.submit(_cancellable_child_task,
                                  (str(tmp_path / f"queued-{index}.pid"),
                                   str(tmp_path / f"queued-{index}.status")))
                  for index in range(8)]
        driver._abort_executor(executor)
        stopped = True
        assert status_path.exists(), "worker exited before reaping its fit child"
        assert int(status_path.read_text()) < 0
        assert any(future.cancelled() for future in queued)
        assert not any((tmp_path / f"queued-{index}.pid").exists() for index in range(8))
    finally:
        if not stopped:
            driver._abort_executor(executor)


def test_worker_initializer_transports_data_problem(monkeypatch):
    class Connection:
        def execute(self, statement):
            return self

    connection = Connection()
    problem = data_fail("POPULATION_COLLAPSED", "training qualification collapsed").problem

    def reject_plan(*args):
        raise DataError(problem)

    monkeypatch.setattr(driver, "_PARALLEL", None)
    monkeypatch.setattr(driver.signal, "signal", lambda *args: signal.SIG_DFL)
    monkeypatch.setattr(driver, "start_rss_watchdog", lambda limit: None)
    monkeypatch.setattr(driver, "connect", lambda *args, **kwargs: connection)
    monkeypatch.setattr(driver, "ArtifactStore", lambda root: SimpleNamespace(root=root))
    monkeypatch.setattr(driver, "Repository", lambda *args, **kwargs: object())
    monkeypatch.setattr(driver, "verified_registration_bytes", lambda *args, **kwargs: b"binding")
    monkeypatch.setattr(driver, "load_registration_document", lambda *args, **kwargs: {})
    monkeypatch.setattr(driver, "read_search_plan", reject_plan)
    monkeypatch.setattr(driver, "_WORKER_TERMINATING", True)
    driver._init_trial_worker("catalog", "store", "run", {}, object(), 1024)
    assert driver._WORKER_TERMINATING is False
    receipt = driver._parallel_trial_task(((2024, 2023, "candidate"), 0, {}))
    assert receipt["worker_error_type"] == "data"
    assert driver.from_document(driver.Problem, receipt["worker_problem"]) == problem


@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_worker_affinity_failure_is_a_tagged_resource_problem(monkeypatch, error_type):
    monkeypatch.setattr(driver, "_PARALLEL", None)

    def reject_affinity(*args):
        raise error_type("synthetic affinity failure")

    monkeypatch.setattr(driver.os, "sched_setaffinity", reject_affinity)
    receipt = driver._parallel_trial_task(((2024, 2023, "candidate"), 0, {}))
    problem = driver.from_document(driver.Problem, receipt["worker_problem"])
    assert receipt["worker_error_type"] == "ops"
    assert problem.code == "RESOURCE_UNAVAILABLE"
    assert problem.message == "search worker affinity unavailable"
    assert problem.retryable is True


@pytest.mark.parametrize("value", [None, [], "not-an-object"])
def test_worker_validator_rejects_non_object_runtime_requests(value):
    with pytest.raises(OpsError) as error:
        driver.validate_workers(value)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert error.value.problem.message == "stage-2 request requires an object"
