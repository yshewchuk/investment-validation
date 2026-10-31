"""Explicitly seal, run or read-only collect a registered magnitude search."""
import argparse
import io
import json
import os
import signal
import sqlite3
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import redirect_stdout
from multiprocessing import get_context
from pathlib import Path

from engine.v2.contracts import ArtifactRef, Problem
from engine.v2.data.documents import reject_duplicate_keys
from engine.v2.data.errors import DataError
from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    DocumentError,
    SystemClock,
    content_hash,
    from_document,
    iter_canonical_json,
    to_document,
)
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from engine.v2.ops.fingerprints import source_closure
from engine.v2.ops.migrations import require_current
from engine.v2.research.progress import Progress, decision, stage
from experiments.magnitude_search_execution import validate_execution_limits
from experiments.magnitude_search_outer import collect_outer_results, execute_outer_fold
from experiments.magnitude_search_plan import read_search_plan, trial_key
from experiments.magnitude_search_request import validate_workers
from experiments.size_job_budget import remaining
from experiments.magnitude_search_records import MIGRATIONS, OWNER, append_record, read_record
from experiments.magnitude_search_trials import SearchTrials, require_execution_enabled
from experiments.native_gate_worker import read_native_staged_input
from experiments.native_registration import (
    MAX_NATIVE_EVIDENCE_BYTES,
    read_native_registration,
    valid_registration_read_budget,
)
from experiments.memory_guard import start_rss_watchdog

_PARALLEL = None


class _WorkerTerminating(BaseException):
    """Pool termination arrives as SIGTERM; unwind the fit child's cleanup first."""


def _terminate_worker(signum, frame):
    raise _WorkerTerminating


def _init_trial_worker(conn_path, store_root, registration, request, plan_ref, memory_bytes):
    global _PARALLEL
    try:
        signal.signal(signal.SIGTERM, _terminate_worker)
        start_rss_watchdog(memory_bytes)
        conn = connect(conn_path, must_exist=True)
        conn.execute("PRAGMA busy_timeout=30000")
        store = ArtifactStore(store_root)
        repository = Repository(conn, store, pin_snapshots=True)
        plan = read_search_plan(conn, store, registration, request, plan_ref)
        _PARALLEL = (conn, store, SearchTrials(conn, store, repository, registration, request, plan_ref), registration, plan)
    except (OpsError, DataError) as error:
        _PARALLEL = _worker_problem(error)
    except MemoryError:
        _PARALLEL = _worker_problem(fail("RESOURCE_LIMIT_EXCEEDED", "search worker memory exhausted"))
    except (OSError, sqlite3.Error):
        _PARALLEL = _worker_problem(fail("RESOURCE_UNAVAILABLE", "search worker setup unavailable"))
    except Exception:
        _PARALLEL = _worker_problem(fail("RESOURCE_UNAVAILABLE", "search worker setup failed"))


def _complete_trial(trials, action, args, key, *, limits, read):
    try:
        return action(*args, limits=limits)
    except OpsError as error:
        terminal = read("terminal:" + key)
        if (error.code != "EXPERIMENT_VARIANT_FAILED" or error.__cause__ is not None
                or not isinstance(terminal, dict) or terminal.get("status") != "refused"):
            raise
        return trials.reconcile(*args, limits=limits)


def _worker_problem(error):
    kind = "data" if isinstance(error, DataError) else "ops"
    return {"worker_problem": to_document(error.problem), "worker_error_type": kind}


def _parallel_trial_task(task):
    args, cpu_id, limits = task
    try:
        if (isinstance(_PARALLEL, dict)
                and set(_PARALLEL) == {"worker_problem", "worker_error_type"}):
            return _PARALLEL
        try:
            os.sched_setaffinity(0, {cpu_id})
        except (OSError, ValueError):
            return _worker_problem(fail("RESOURCE_UNAVAILABLE", "search worker affinity unavailable"))
        conn, store, trials, registration, plan = _PARALLEL
        key = trial_key(plan, *args)
        _complete_trial(
            trials, trials._execute_borrowed, args, key, limits=limits,
            read=lambda slot: read_record(conn, store, registration.run_id, slot))
        return None
    except _WorkerTerminating:
        os._exit(128 + signal.SIGTERM)
    except (OpsError, DataError) as error:
        return _worker_problem(error)
    except MemoryError:
        return _worker_problem(fail("RESOURCE_LIMIT_EXCEEDED", "search worker memory exhausted"))
    except (OSError, sqlite3.Error):
        return _worker_problem(fail("RESOURCE_UNAVAILABLE", "search worker storage unavailable"))


def _parallel_preflight(conn, request, limits, workers):
    """Admit CPU, parent-memory and file-backed storage bounds before replay writes."""
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError, ValueError):
        raise fail("RESOURCE_UNAVAILABLE", "search workers require Linux CPU affinity") from None
    try:
        parent_memory = request["limits"]["parent"]["memory_bytes"]
    except (KeyError, TypeError):
        raise fail("INVALID_EXPERIMENT_SPEC", "search worker memory bound is missing") from None
    if type(parent_memory) is not int or parent_memory < 1:
        raise fail("INVALID_EXPERIMENT_SPEC", "search worker memory bound is invalid")
    if workers > min(8, len(cpus)) or workers * limits["memory_bytes"] > parent_memory:
        decision("search workers", f"refusing workers={workers}; cpu_bound={min(8, len(cpus))}; worker_memory_bytes={limits['memory_bytes']}; parent_memory_bytes={parent_memory}")
        raise fail("INVALID_EXPERIMENT_SPEC", "search worker count exceeds CPU or memory bounds")
    database = conn.execute("PRAGMA database_list").fetchone()
    if database is None or len(database) < 3 or not database[2]:
        raise fail("INVALID_EXPERIMENT_SPEC", "parallel search requires a file-backed catalog")
    decision("search workers", f"process pool workers={workers}; cpu_bound={min(8, len(cpus))}; worker_memory_bytes={limits['memory_bytes']}")
    return cpus, database[2]


def _abort_executor(executor):
    """Cancel queued work, terminate workers, and join them after child cleanup."""
    processes = list(getattr(executor, "_processes", {}).values())
    terminate = getattr(executor, "terminate_workers", None)
    if callable(terminate):
        terminate()
    else:
        for process in processes:
            if process.is_alive():
                process.terminate()
    for process in processes:
        process.join()
    executor.shutdown(wait=True, cancel_futures=True)

MAX_REPORT_BYTES = 64 * 1024 * 1024


def _binding(conn, store, registration, request, plan_ref):
    plan = read_search_plan(conn, store, registration, request, plan_ref)
    try:
        sources = source_closure(Path(__file__).resolve().parents[1], ["experiments/v2_magnitude_search_run.py"])
    except (OpsError, ArtifactError, OSError, SyntaxError, ValueError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search driver sources cannot be verified") from None
    return plan, {"plan_id": plan["plan_id"], "plan_ref": to_document(plan_ref), "driver_sources": sources}


def _roster(plan):
    roster, seen = [], set()
    for fold in plan["folds"]:
        for candidate in plan["candidates"]:
            for inner in fold["inner"]:
                args = (fold["outer"]["year"], inner["year"], candidate["candidate_id"])
                key = trial_key(plan, *args)
                if key not in seen:
                    seen.add(key)
                    roster.append(args)
    return roster


def seal_search_run(conn, store, repository, registration, request, plan_ref, *, clock):
    """Seal the driver before work; never retroactively bind started library journals."""
    plan, binding = _binding(conn, store, registration, request, plan_ref)
    SearchTrials(conn, store, repository, registration, request, plan_ref).seal(clock=clock)
    previous = read_record(conn, store, registration.run_id, "driver-seal")
    absent = tuple(["start:" + trial_key(plan, *args) for args in _roster(plan)]
        + ["outer-start:" + content_hash([plan["plan_id"], fold["outer"]["year"]]) for fold in plan["folds"]]
        + ["selection"])
    if previous is None and any(read_record(conn, store, registration.run_id, slot) is not None for slot in absent):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search driver cannot bind an already started journal")
    append_record(conn, store, registration.run_id, "driver-seal", binding, absent=absent)
    return {"sealed": plan["plan_id"], "run_id": registration.run_id, "driver_id": content_hash(binding)}


def _admit_driver(conn, store, registration, request, plan_ref):
    require_execution_enabled(registration)
    require_current(conn, OWNER, MIGRATIONS)
    plan, binding = _binding(conn, store, registration, request, plan_ref)
    if read_record(conn, store, registration.run_id, "driver-seal") != binding:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search execution requires its exact driver seal")
    return plan


def run_search(conn, store, repository, registration, request, plan_ref, *, limits, collect_results=True):
    """Validate every claimed inner result before any fresh serial fit; never retry; collect_results=False returns None before the final collect."""
    workers = validate_workers(request)
    request = {key: value for key, value in request.items() if key != "workers"}
    validate_execution_limits(limits)
    plan = _admit_driver(conn, store, registration, request, plan_ref)
    trials = SearchTrials(conn, store, repository, registration, request, plan_ref)
    def read(slot):
        return read_record(conn, store, registration.run_id, slot)
    claimed, fresh = [], []
    for args in _roster(plan):
        key = trial_key(plan, *args)
        started, result = read("start:" + key), read("execution-result:" + key)
        if ((started is not None and result is None) or (started is None and any(
                item is not None for item in (result, read("execution:" + key), read("terminal:" + key))))):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search contains an unfinished or orphan inner claim")
        (claimed if started is not None else fresh).append((args, key))
    if fresh and read("selection") is not None:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search selection precedes unfinished inner work")
    completed_outer = []
    for fold in plan["folds"]:
        key = content_hash([plan["plan_id"], fold["outer"]["year"]])
        if (read("outer-start:" + key) is None) != (read("outer-terminal:" + key) is None):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search contains an unfinished or orphan outer claim")
        if read("outer-start:" + key) is not None:
            completed_outer.append(fold["outer"]["year"])
    parallel = workers > 1 and bool(fresh)
    if parallel:
        cpus, conn_path = _parallel_preflight(conn, request, limits, workers)
        forbid_fitting("run_search")
        forbid_legacy_fitting("run_search")
        remaining(conn, store, registration, plan)
    meter = Progress("search trials", len(claimed) + len(fresh))
    decision("search trials", f"{len(claimed)} replay claims; {len(fresh)} fresh trials; zero retries")
    for args, key in claimed:
        _complete_trial(trials, trials.reconcile, args, key, limits=limits, read=read)
        meter.tick()
    for year in completed_outer:
        execute_outer_fold(conn, store, repository, registration, request, plan_ref, year, limits=limits)
    if not parallel:
        for args, key in fresh:
            _complete_trial(trials, trials.execute, args, key, limits=limits, read=read)
            meter.tick()
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_init_trial_worker,
            initargs=(conn_path, str(store.root), registration, request, plan_ref, limits["memory_bytes"]),
        )
        with stage("search fresh trials"):
            parallel_meter = Progress("search fresh trials", len(fresh))
            try:
                for offset in range(0, len(fresh), workers):
                    batch = [(fresh[offset + index][0], cpus[(offset + index) % len(cpus)], limits)
                             for index in range(min(workers, len(fresh) - offset))]
                    for receipt in executor.map(_parallel_trial_task, batch):
                        if receipt is not None:
                            if (isinstance(receipt, dict)
                                    and set(receipt) == {"worker_problem", "worker_error_type"}
                                    and receipt["worker_error_type"] in ("ops", "data")):
                                problem = from_document(Problem, receipt["worker_problem"])
                                raise DataError(problem) if receipt["worker_error_type"] == "data" else OpsError(problem)
                            raise fail("RESOURCE_UNAVAILABLE", "search worker returned an invalid result")
                        meter.tick()
                        parallel_meter.tick()
            except BrokenProcessPool:
                _abort_executor(executor)
                raise fail("RESOURCE_UNAVAILABLE", "search worker process exited unexpectedly") from None
            except BaseException:
                _abort_executor(executor)
                raise
            else:
                executor.shutdown()
    selected = trials.freeze()
    for selection in selected["selections"]:
        if selection["outer_year"] in completed_outer:
            continue
        execute_outer_fold(conn, store, repository, registration, request, plan_ref,
                           selection["outer_year"], limits=limits)
    if not collect_results:
        return None
    return collect_outer_results(conn, store, repository, registration, request, plan_ref)


def _read_document(path):
    data = read_native_staged_input(path)
    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant")
    try:
        document = json.loads(data, object_pairs_hook=reject_duplicate_keys, parse_constant=invalid_constant)
        if not isinstance(document, dict):
            raise ValueError
        return document
    except (ValueError, UnicodeError, RecursionError):
        raise fail("INVALID_EXPERIMENT_SPEC", "search requires valid JSON object metadata") from None


def _render(document, *, scratch=True):
    """Finish budget admission before stdout; refusals need no scratch allocation."""
    stream = tempfile.TemporaryFile(mode="w+t", encoding="utf-8", newline="") if scratch else io.StringIO()
    try:
        count = 1
        for chunk in iter_canonical_json(document):
            count += len(chunk.encode("utf-8"))
            if count > MAX_REPORT_BYTES:
                raise fail("RESOURCE_LIMIT_EXCEEDED", "search report exceeds its stdout byte budget")
            stream.write(chunk)
        stream.write("\n")
        stream.seek(0)
        return stream
    except BaseException:
        stream.close()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "run", "collect"))
    for name in ("spec", "request", "plan-ref", "catalog", "store-root"):
        parser.add_argument("--" + name, required=True)
    for name in ("memory-bytes", "cpu-seconds", "wall-seconds"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--max-registration-bytes", type=int, default=MAX_NATIVE_EVIDENCE_BYTES)
    parser.add_argument("--no-ledger", action="store_true", help="accepted no-op; no mode publishes a ledger")
    args = parser.parse_args(argv)
    if not valid_registration_read_budget(args.max_registration_bytes):
        parser.error("--max-registration-bytes must be a positive SQLite-safe byte budget")
    if any("\0" in getattr(args, name) for name in ("spec", "request", "plan_ref", "catalog", "store_root")):
        parser.error("search paths must not contain NUL characters")
    limits = {name: getattr(args, name) for name in ("memory_bytes", "cpu_seconds", "wall_seconds")}
    if (args.action == "run" and any(value is None for value in limits.values())) or (
            args.action != "run" and any(value is not None for value in limits.values())):
        parser.error("run requires all three resource limits; seal and collect accept none")
    if args.action == "run":
        try:
            validate_execution_limits(limits)
        except OpsError as error:
            parser.error(error.problem.message)
    conn, status = None, 0
    try:
        with redirect_stdout(sys.stderr):
            if not Path(args.store_root).is_dir():
                raise fail("RESOURCE_UNAVAILABLE", "search requires an existing artifact store")
            try:
                spec = experiment_spec_from_document(_read_document(args.spec))
                request = _read_document(args.request)
                reference = from_document(ArtifactRef, _read_document(args.plan_ref))
                if request.get("schema_version") in ("magnitude_search_request.v2", "magnitude_search_request.v3"):
                    raise fail("INVALID_EXPERIMENT_SPEC", "qualified SIZE requires experiments.size_search CLI containment")
            except (DocumentError, TypeError):
                raise fail("INVALID_EXPERIMENT_SPEC", "search metadata could not be admitted") from None
            if args.action == "collect":
                conn = sqlite3.connect(Path(args.catalog).resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None)
                conn.row_factory = sqlite3.Row
            else:
                conn = connect(args.catalog, must_exist=True)
            store = ArtifactStore(args.store_root)
            registration = read_native_registration(conn, store, spec, max_evidence_bytes=args.max_registration_bytes)
            repository = Repository(conn, store, pin_snapshots=True)
            context = (conn, store, repository, registration, request, reference)
            if args.action == "seal":
                result = seal_search_run(*context, clock=SystemClock())
            elif args.action == "run":
                result = run_search(*context, limits=limits)
            else:
                _admit_driver(conn, store, registration, request, reference)
                result = collect_outer_results(*context)
            output = _render(result)
    except (OpsError, DataError) as error:
        output, status = _render({"refused": to_document(error.problem)}, scratch=False), 2
    except (ArtifactError, OSError, sqlite3.Error):
        problem = fail("RESOURCE_UNAVAILABLE", "search request could not be completed").problem
        output, status = _render({"refused": to_document(problem)}, scratch=False), 2
    finally:
        if conn is not None:
            conn.close()
    with output:
        binary = getattr(sys.stdout, "buffer", None)
        destination = sys.stdout if binary is None else binary
        while chunk := output.read(65536):
            data = chunk if binary is None else chunk.encode("utf-8")
            offset = 0
            while offset < len(data):
                written = destination.write(data[offset:])
                if type(written) is not int or not 0 < written <= len(data) - offset:
                    raise OSError("stdout write did not report valid progress")
                offset += written
        destination.flush()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
