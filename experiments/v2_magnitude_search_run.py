"""Explicitly seal, serially run or read-only collect a registered magnitude search."""
import argparse
import io
import json
import sqlite3
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from engine.v2.contracts import ArtifactRef
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
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from engine.v2.ops.fingerprints import source_closure
from engine.v2.ops.migrations import require_current
from experiments.magnitude_search_execution import validate_execution_limits
from experiments.magnitude_search_outer import collect_outer_results, execute_outer_fold
from experiments.magnitude_search_plan import read_search_plan
from experiments.magnitude_search_records import MIGRATIONS, OWNER, append_record, read_record
from experiments.magnitude_search_trials import SearchTrials
from experiments.native_gate_worker import read_native_staged_input
from experiments.native_registration import (
    MAX_NATIVE_EVIDENCE_BYTES,
    read_native_registration,
    valid_registration_read_budget,
)

MAX_REPORT_BYTES = 64 * 1024 * 1024


def _binding(conn, store, registration, request, plan_ref):
    plan = read_search_plan(conn, store, registration, request, plan_ref)
    try:
        sources = source_closure(Path(__file__).resolve().parents[1], ["experiments/v2_magnitude_search_run.py"])
    except (OpsError, ArtifactError, OSError, SyntaxError, ValueError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search driver sources cannot be verified") from None
    return plan, {"plan_id": plan["plan_id"], "plan_ref": to_document(plan_ref), "driver_sources": sources}


def _roster(plan):
    return [(fold["outer"]["year"], inner["year"], candidate["candidate_id"])
            for fold in plan["folds"] for candidate in plan["candidates"] for inner in fold["inner"]]


def seal_search_run(conn, store, repository, registration, request, plan_ref, *, clock):
    """Seal the driver before work; never retroactively bind started library journals."""
    plan, binding = _binding(conn, store, registration, request, plan_ref)
    SearchTrials(conn, store, repository, registration, request, plan_ref).seal(clock=clock)
    previous = read_record(conn, store, registration.run_id, "driver-seal")
    absent = tuple(["start:" + content_hash([plan["plan_id"], *args]) for args in _roster(plan)]
        + ["outer-start:" + content_hash([plan["plan_id"], fold["outer"]["year"]]) for fold in plan["folds"]]
        + ["selection"])
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


def run_search(conn, store, repository, registration, request, plan_ref, *, limits):
    """Validate every claimed inner result before any fresh serial fit; never retry."""
    validate_execution_limits(limits)
    plan = _admit_driver(conn, store, registration, request, plan_ref)
    trials = SearchTrials(conn, store, repository, registration, request, plan_ref)
    def read(slot):
        return read_record(conn, store, registration.run_id, slot)
    claimed, fresh = [], []
    for args in _roster(plan):
        key = content_hash([plan["plan_id"], *args])
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
    def complete(action, args, key):
        try:
            action(*args, limits=limits)
        except OpsError as error:
            terminal = read("terminal:" + key)
            if (error.code != "EXPERIMENT_VARIANT_FAILED" or error.__cause__ is not None
                    or not isinstance(terminal, dict) or terminal.get("status") != "refused"):
                raise
            trials.reconcile(*args, limits=limits)
    for args, key in claimed:
        complete(trials.reconcile, args, key)
    for year in completed_outer:
        execute_outer_fold(conn, store, repository, registration, request, plan_ref, year, limits=limits)
    for args, key in fresh:
        complete(trials.execute, args, key)
    selected = trials.freeze()
    for selection in selected["selections"]:
        if selection["outer_year"] in completed_outer:
            continue
        execute_outer_fold(conn, store, repository, registration, request, plan_ref,
                           selection["outer_year"], limits=limits)
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
            except (DocumentError, TypeError):
                raise fail("INVALID_EXPERIMENT_SPEC", "search metadata could not be admitted") from None
            if args.action == "collect":
                conn = sqlite3.connect(Path(args.catalog).resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None)
                conn.row_factory = sqlite3.Row
            else:
                conn = connect(args.catalog, must_exist=True)
            store = ArtifactStore(args.store_root)
            registration = read_native_registration(conn, store, spec, max_evidence_bytes=args.max_registration_bytes)
            repository = Repository(conn, store)
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
