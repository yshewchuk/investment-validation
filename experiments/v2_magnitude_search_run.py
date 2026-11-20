"""Explicitly seal, run or read-only collect a registered magnitude search."""
import argparse
import ctypes
import gc
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

from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.contracts import ArtifactRef, Problem
from engine.v2.data.documents import reject_duplicate_keys
from engine.v2.data.errors import DataError
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
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from engine.v2.ops.fingerprints import source_closure
from engine.v2.ops.migrations import require_current
from engine.v2.research.progress import Progress, decision, stage
from experiments.magnitude_search_execution import validate_execution_limits
from experiments.magnitude_search_outer import collect_outer_results, execute_outer_fold, shared_freeze
from experiments.magnitude_search_plan import plan_memo, read_search_plan, trial_key
from experiments.magnitude_search_records import MIGRATIONS, OWNER, append_record, read_record
from experiments.magnitude_search_refits import per_candidate, possible_slots, refit_units
from experiments.magnitude_search_request import validate_workers
from experiments.magnitude_search_request_schema import qualified, regression_request_version, request_version
from experiments.magnitude_search_trials import SearchTrials
from experiments.memory_guard import start_rss_watchdog
from experiments.native_gate_worker import read_native_staged_input
from experiments.native_registration import (
    MAX_NATIVE_EVIDENCE_BYTES,
    NativeRegistration,
    load_registration_document,
    read_native_registration,
    valid_registration_read_budget,
    verified_registration_bytes,
)
from experiments.search_memory_log import log_memory, panel_memo_holder
from experiments.size_job_budget import remaining

_PARALLEL = None
_WORKER_WATCHDOG_STOP = None
_WORKER_TERMINATING = False
_WORKER_LOG = {"slices": 0, "fits": 0}


def _worker_due(kind):
    """Count one worker stage line of this kind; True for the first three and every twentieth."""
    _WORKER_LOG[kind] += 1
    return _WORKER_LOG[kind] <= 3 or _WORKER_LOG[kind] % 20 == 0


def _worker_stage(trials):
    """Log worker memory after a cohort slice loads and after trials fit, at a bounded cadence."""
    load = trials._cohort_inputs

    def loaded(cohort, *args, **kwargs):
        fresh = cohort not in trials._memo
        result = load(cohort, *args, **kwargs)
        if fresh and _worker_due("slices"):
            log_memory("worker slice loaded", panel=panel_memo_holder())
        return result
    trials._cohort_inputs = loaded


def _stop_worker_watchdog():
    global _WORKER_WATCHDOG_STOP
    stop, _WORKER_WATCHDOG_STOP = _WORKER_WATCHDOG_STOP, None
    if stop is not None:
        stop()


class _WorkerTerminating(BaseException):
    """Pool termination arrives as SIGTERM; unwind the fit child's cleanup first."""


def _terminate_worker(signum, frame):
    """Raise once so fit-child cleanup unwinds; repeated termination must not interrupt it."""
    global _WORKER_TERMINATING
    if _WORKER_TERMINATING:
        return
    _WORKER_TERMINATING = True
    raise _WorkerTerminating


def _stored_plan(store, plan_ref):
    """Read the verified plan artifact the parent sealed; the worker does not rebuild it from the registration."""
    if not isinstance(plan_ref, ArtifactRef):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search plan reference is invalid")
    try:
        plan = json.loads(store.read_verified(plan_ref))
    except (ArtifactError, ValueError, RecursionError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "stored search plan failed verification") from None
    if not isinstance(plan, dict) or not isinstance(plan.get("plan_id"), str):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "stored search plan failed verification")
    return plan


def _init_trial_worker(conn_path, store_root, run_id, request, plan_ref, memory_bytes, adopt=False):
    global _PARALLEL, _WORKER_WATCHDOG_STOP, _WORKER_TERMINATING
    _WORKER_WATCHDOG_STOP = None
    _WORKER_TERMINATING = False
    _WORKER_LOG["slices"] = _WORKER_LOG["fits"] = 0
    try:
        signal.signal(signal.SIGTERM, _terminate_worker)
        _WORKER_WATCHDOG_STOP = start_rss_watchdog(memory_bytes)
        conn = connect(conn_path, must_exist=True)
        conn.execute("PRAGMA busy_timeout=30000")
        store = ArtifactStore(store_root)
        verified_size_rows = {}
        binding = verified_registration_bytes(conn, store, run_id, max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES,
                                              verified_size_rows=verified_size_rows)
        if binding is None:
            raise fail("INVALID_EXPERIMENT_SPEC", "native experiment has not been preregistered")
        try:
            schema = json.loads(binding).get("schema_version")
            hydrate = ((lambda: load_registration_document(store, binding, verified_size_rows=verified_size_rows))
                       if schema == "native_experiment_registration.v1.7" else None)
            registration = NativeRegistration(run_id, binding, hydrate=hydrate)
            log_memory("worker registration loaded", registration=registration.lean)
        except (ArtifactError, ValueError, KeyError, TypeError, AttributeError):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native registration evidence is invalid") from None
        repository = Repository(conn, store, pin_snapshots=True)
        plan = _stored_plan(store, plan_ref)
        log_memory("worker plan read", plan=plan)
        trials = SearchTrials(conn, store, repository, registration, request, plan_ref)
        trials.trusted_plan = plan
        _worker_stage(trials)
        trials.adopt_published = adopt
        _PARALLEL = (conn, store, trials, registration, plan)
    except (OpsError, DataError) as error:
        _PARALLEL = _worker_problem(error)
    except MemoryError:
        _stop_worker_watchdog()
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


def _raise_receipt(receipt):
    if (isinstance(receipt, dict) and set(receipt) == {"worker_problem", "worker_error_type"}
            and receipt["worker_error_type"] in ("ops", "data")):
        problem = from_document(Problem, receipt["worker_problem"])
        raise DataError(problem) if receipt["worker_error_type"] == "data" else OpsError(problem)
    raise fail("RESOURCE_UNAVAILABLE", "search worker returned an invalid result")


def _guarded(cpu_id, work):
    try:
        if (isinstance(_PARALLEL, dict)
                and set(_PARALLEL) == {"worker_problem", "worker_error_type"}):
            return _PARALLEL
        try:
            os.sched_setaffinity(0, {cpu_id})
        except (OSError, ValueError):
            return _worker_problem(fail("RESOURCE_UNAVAILABLE", "search worker affinity unavailable"))
        work()
        return None
    except _WorkerTerminating:
        os._exit(128 + signal.SIGTERM)
    except (OpsError, DataError) as error:
        return _worker_problem(error)
    except MemoryError:
        _stop_worker_watchdog()
        return _worker_problem(fail("RESOURCE_LIMIT_EXCEEDED", "search worker memory exhausted"))
    except (OSError, sqlite3.Error):
        return _worker_problem(fail("RESOURCE_UNAVAILABLE", "search worker storage unavailable"))


def _parallel_trial_task(task):
    args, cpu_id, limits = task

    def work():
        conn, store, trials, registration, plan = _PARALLEL
        key = trial_key(plan, *args)
        _complete_trial(
            trials, trials._execute_borrowed, args, key, limits=limits,
            read=lambda slot: read_record(conn, store, registration.run_id, slot))
        if _worker_due("fits"):
            log_memory("worker after fit", panel=panel_memo_holder())
        if panel_memo_holder():
            release_parent_caches("worker trial")
    return _guarded(cpu_id, work)


def _prepublish_task(task):
    args_list, cpu_id = task
    return _guarded(cpu_id, lambda: _PARALLEL[2].prepublish(args_list))


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
        + ["outer-start:" + key for _, _, key in possible_slots(plan)]
        + ["selection"])
    if "target" in plan:
        absent += tuple("comparison-start:" + content_hash([plan["plan_id"], fold["outer"]["year"], "comparison"])
                        for fold in plan["folds"])
    if previous is None and any(read_record(conn, store, registration.run_id, slot) is not None for slot in absent):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search driver cannot bind an already started journal")
    append_record(conn, store, registration.run_id, "driver-seal", binding, absent=absent)
    return {"sealed": plan["plan_id"], "run_id": registration.run_id, "driver_id": content_hash(binding)}


def _admit_driver(conn, store, registration, request, plan_ref):
    require_current(conn, OWNER, MIGRATIONS)
    plan, binding = _binding(conn, store, registration, request, plan_ref)
    if read_record(conn, store, registration.run_id, "driver-seal") != binding:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search execution requires its exact driver seal")
    return plan


def release_parent_caches(label, *trials):
    """Drop the parent's re-derivable inputs: the shared panel memo and each SearchTrials._memo (never _published), then gc and malloc_trim. A failure logs and continues; MemoryError propagates."""
    try:
        panel = panel_memo_holder()
        if panel is not None:
            panel.clear()
        for item in trials:
            memo = getattr(item, "_memo", None)
            if isinstance(memo, dict):
                memo.clear()
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            decision("search release", f"{label}: malloc_trim unavailable")
    except MemoryError:
        raise
    except Exception as error:
        decision("search release", f"{label}: release failed ({type(error).__name__}); continuing")


def _run_search(conn, store, repository, registration, request, plan_ref, *, limits, collect_results=True, before_dispatch=None, isolate=False):
    """Body of run_search; the public wrapper installs the plan memo."""
    workers = validate_workers(request)
    request = {key: value for key, value in request.items() if key != "workers"}
    validate_execution_limits(limits)
    plan = _admit_driver(conn, store, registration, request, plan_ref)
    trials = SearchTrials(conn, store, repository, registration, request, plan_ref)
    log_memory("start", panel_memo=panel_memo_holder(), trials_memo=getattr(trials, "_memo", None))
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
    for year, candidate_id, key in possible_slots(plan):
        if (read("outer-start:" + key) is None) != (read("outer-terminal:" + key) is None):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search contains an unfinished or orphan outer claim")
        if read("outer-start:" + key) is not None:
            completed_outer.append((year, candidate_id))
    parallel = (workers > 1 or isolate) and bool(fresh)
    deadline_seconds = None
    if parallel:
        cpus, conn_path = _parallel_preflight(conn, request, limits, workers)
        if limits != request["limits"]["child"]:
            raise fail("INVALID_EXPERIMENT_SPEC", "v2 trial limits differ from sealed request")
        forbid_fitting("run_search")
        forbid_legacy_fitting("run_search")
        deadline_seconds = remaining(conn, store, registration, plan)
    elif before_dispatch is not None:
        deadline_seconds = remaining(conn, store, registration, plan)
    if before_dispatch is not None:
        before_dispatch(deadline_seconds)
    meter = Progress("search trials", len(claimed) + len(fresh))
    decision("search trials", f"{len(claimed)} replay claims; {len(fresh)} fresh trials; zero retries")
    for args, key in claimed:
        _complete_trial(trials, trials.reconcile, args, key, limits=limits, read=read)
        meter.tick()
    for year, candidate_id in completed_outer:
        execute_outer_fold(conn, store, repository, registration, request, plan_ref, year,
                           limits=limits, candidate_id=candidate_id)
    release_parent_caches("before pool", trials)
    log_memory("after claim replays", panel_memo=panel_memo_holder(), trials_memo=getattr(trials, "_memo", None))
    if not parallel:
        for args, key in fresh:
            _complete_trial(trials, trials.execute, args, key, limits=limits, read=read)
            meter.tick()
    else:
        with stage("search prepublish"):
            pool = ProcessPoolExecutor(
                max_workers=1, mp_context=get_context("spawn"), initializer=_init_trial_worker,
                initargs=(conn_path, str(store.root), registration.run_id, request, plan_ref,
                          request["limits"]["parent"]["memory_bytes"]))
            try:
                receipt = pool.submit(_prepublish_task, ([args for args, _ in fresh], cpus[0])).result()
                if receipt is not None:
                    _raise_receipt(receipt)
            except BrokenProcessPool:
                _abort_executor(pool)
                raise fail("RESOURCE_UNAVAILABLE", "search worker process exited unexpectedly") from None
            except BaseException:
                _abort_executor(pool)
                raise
            else:
                pool.shutdown()
        log_memory("after prepublish", panel_memo=panel_memo_holder(), trials_memo=getattr(trials, "_memo", None))
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_init_trial_worker,
            initargs=(conn_path, str(store.root), registration.run_id, request, plan_ref, limits["memory_bytes"], True),
        )
        with stage("search fresh trials"):
            parallel_meter = Progress("search fresh trials", len(fresh))
            try:
                for offset in range(0, len(fresh), workers):
                    batch = [(fresh[offset + index][0], cpus[(offset + index) % len(cpus)], limits)
                             for index in range(min(workers, len(fresh) - offset))]
                    for receipt in executor.map(_parallel_trial_task, batch):
                        if receipt is not None:
                            _raise_receipt(receipt)
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
    log_memory("after fresh trials", panel_memo=panel_memo_holder(), trials_memo=getattr(trials, "_memo", None))
    release_parent_caches("before freeze", trials)
    selected = trials.freeze()
    log_memory("after freeze", panel_memo=panel_memo_holder(), trials_memo=getattr(trials, "_memo", None))
    with shared_freeze(registration.run_id, plan_ref, selected):
        completed = set(completed_outer)
        if per_candidate(plan):
            try:
                units = [(item["outer_year"], unit) for item in selected["selections"]
                         for unit in refit_units(plan, item)]
            except (ValueError, TypeError, KeyError):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "stored selection is malformed") from None
            fresh_units = [pair for pair in units if pair not in completed]
            decision("search refits", f"{len(units)} units; {len(units) - len(fresh_units)} completed replays; "
                     f"{len(fresh_units)} fresh refits")
            refit_meter = Progress("search refits", len(fresh_units))
            with stage("search refits"):
                for year, unit in fresh_units:
                    outcome = execute_outer_fold(conn, store, repository, registration, request, plan_ref,
                                                 year, limits=limits, candidate_id=unit)
                    refit_meter.tick()
                    if isinstance(outcome, dict) and outcome.get("status") == "failed":
                        decision("search refit failed", f"{unit} year={year}")
        else:
            for item in selected["selections"]:
                try:
                    units = refit_units(plan, item)
                except (ValueError, TypeError, KeyError):
                    raise fail("EXPERIMENT_IDENTITY_CONFLICT", "stored selection is malformed") from None
                for unit in units:
                    if (item["outer_year"], unit) not in completed:
                        execute_outer_fold(conn, store, repository, registration, request, plan_ref,
                                           item["outer_year"], limits=limits, candidate_id=unit)
    release_parent_caches("after refits", trials)
    log_memory("after refits", panel_memo=panel_memo_holder(), trials_memo=getattr(trials, "_memo", None))
    if not collect_results:
        return None
    release_parent_caches("before collect", trials)
    return collect_outer_results(conn, store, repository, registration, request, plan_ref)


def run_search(conn, store, repository, registration, request, plan_ref, *, limits, collect_results=True, before_dispatch=None, isolate=False):
    """Validate claimed inner results before fresh work; never retry; collect_results=False returns None before the final collect. isolate=True sends fresh trials through the spawn pool even for workers=1 (requires a v2+ request with limits); library callers keep the in-process default."""
    with plan_memo():
        return _run_search(conn, store, repository, registration, request, plan_ref, limits=limits,
                           collect_results=collect_results, before_dispatch=before_dispatch, isolate=isolate)


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
                if regression_request_version(request.get("schema_version")) is not None:
                    raise fail("INVALID_EXPERIMENT_SPEC", "IV requires dedicated CLI containment; use the admitted library path")
                if qualified(request_version(request.get("schema_version"))):
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
