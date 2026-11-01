"""Explicit local qualified SIZE development lifecycle; never builds or reads holdouts."""
import argparse
import json
import resource
import signal
import sqlite3
import sys
import time
from contextlib import ExitStack, redirect_stdout
from functools import partial
from pathlib import Path

from engine.v2.contracts import ArtifactRef
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    DocumentError,
    SystemClock,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from experiments.magnitude_search_plan import seal_search_plan
from experiments.magnitude_search_records import read_record
from experiments.memory_guard import rss_watchdog
from experiments.native_registration import (
    MAX_NATIVE_EVIDENCE_BYTES,
    read_native_registration,
    reserve_native_registration,
)
from experiments.size_job_budget import remaining
from experiments.size_preflight_receipt import admit, input_bindings, issue
from experiments.size_search_binding import prepare, validate_panel_request
from experiments.size_search_report import collect, resource_report
from experiments.v2_magnitude_search_run import (
    _admit_driver,
    _read_document,
    run_search,
    seal_search_run,
)


def _contain(request, horizons=(5, 3, 1)):
    """Apply explicit parent/job ceilings in the CLI process, never silently increase."""
    from experiments.magnitude_search_request_versions import compile_request
    limits = compile_request(request, horizons=tuple(h for h in (5, 3, 1) if h in horizons))["request"]["limits"]
    memory = limits["parent"]["memory_bytes"]
    if limits["child"]["memory_bytes"] > memory:
        raise fail("INVALID_EXPERIMENT_SPEC", "child memory must not exceed inherited parent limit")
    def expired(signum, frame):
        raise fail("RESOURCE_LIMIT_EXCEEDED", "parent/job wall budget exhausted; no fit retry")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(min(limits["parent"]["wall_seconds"], limits["job"]["wall_seconds"]))
    return memory


def _run(context, *, collector=None, read_only=False):
    """Verify completed collection; a read-only incomplete probe returns None."""
    conn, store, _, registration, request, ref = context
    plan = _admit_driver(conn, store, registration, request, ref)
    complete = all(read_record(conn, store, registration.run_id, prefix + content_hash(parts)) is not None
        for group in plan["folds"] for prefix, parts in (
            ("outer-terminal:", [plan["plan_id"], group["outer"]["year"]]),
            ("comparison-terminal:", [plan["plan_id"], group["outer"]["year"], "comparison"])))
    if not complete:
        if read_only:
            return None
        seconds = remaining(conn, store, registration, plan)
        previous = signal.alarm(0)
        signal.alarm(min(previous or 1, seconds))
        run_search(*context, limits=request["limits"]["child"], collect_results=False)
    return (collect if collector is None else collector)(*context)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "register", "seal", "run", "report"))
    for name in ("spec", "request", "catalog", "store-root"):
        parser.add_argument("--" + name, required=True)
    for name in ("panel-request", "scope", "training-start", "labels-before", "plan-ref",
                 "receipt-ref"):
        parser.add_argument("--" + name)
    parser.add_argument("--no-ledger", action="store_true", required=True)
    args = parser.parse_args(argv)
    fresh = args.action in ("preflight", "register")
    if fresh and not all((args.panel_request, args.scope, args.training_start, args.labels_before)):
        parser.error("preflight/register require panel-request, scope, training-start and labels-before")
    if args.action == "register" and not args.receipt_ref:
        parser.error("register requires receipt-ref")
    if not fresh and not args.plan_ref:
        parser.error("seal/run/report require plan-ref")
    conn, started = None, time.monotonic()
    stack = ExitStack()
    try:
        request = _read_document(args.request)
        if isinstance(request, dict) and request.get("schema_version") == "regression_search_request.v1":
            raise fail("INVALID_EXPERIMENT_SPEC", "IV requires its dedicated CLI; use the admitted library path")
        panel_request = validate_panel_request(_read_document(args.panel_request)) if fresh else None
        # _contain only extracts resource ceilings here for every action; exact fit-count
        # feasibility is checked later by admitted plan validation, so non-fresh uses (1,).
        memory = _contain(request, panel_request["horizons"] if fresh else (1,))
        stack.enter_context(rss_watchdog(memory))
        print("[size] stage=" + args.action + " start rss_kib=" + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), file=sys.stderr)
        with redirect_stdout(sys.stderr):
            if not Path(args.store_root).is_dir():
                raise fail("RESOURCE_UNAVAILABLE", "existing artifact store required")
            try:
                spec = experiment_spec_from_document(_read_document(args.spec))
            except TypeError:
                raise fail("INVALID_EXPERIMENT_SPEC", "invalid SIZE specification metadata") from None
            conn = connect(args.catalog, must_exist=True)
            store, clock = ArtifactStore(args.store_root), SystemClock()
            repository = Repository(conn, store, pin_snapshots=True)
            if fresh:
                bind = partial(input_bindings, repository, spec, request, panel_request, scope=args.scope,
                    training_start=args.training_start, labels_before=args.labels_before)
                if args.action == "preflight":
                    registration, spec = prepare(repository, spec, request, panel_request,
                        scope=args.scope, training_start=args.training_start,
                        labels_before=args.labels_before)
                    from experiments.magnitude_search_request_versions import compile_request
                    counts = compile_request(request, horizons=tuple(h for h in (5, 3, 1)
                        if h in panel_request["horizons"]))["counts"]
                    result = resource_report({"counts": counts, "request": request})
                    result["receipt_ref"] = to_document(issue(store, bind(), registration))
                else:
                    try:
                        receipt_ref = from_document(ArtifactRef, _read_document(args.receipt_ref))
                    except FileNotFoundError:
                        raise fail("INVALID_EXPERIMENT_SPEC", "SIZE preflight receipt is missing", details={"reason": "receipt_missing"}) from None
                    registration, spec = admit(store, receipt_ref, bind(), spec)
                    reserve_native_registration(conn, store, registration, spec, max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES)
                    result = seal_search_plan(conn, store, registration, request)
                print("[size] receipt " + ("issued" if args.action == "preflight" else "admitted"), file=sys.stderr)
            else:
                registration = read_native_registration(conn, store, spec)
                ref = from_document(ArtifactRef, _read_document(args.plan_ref))
                context = (conn, store, repository, registration, request, ref)
                if args.action == "seal":
                    result = seal_search_run(*context, clock=clock)
                else:
                    result = _run(context) if args.action == "run" else collect(*context)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
    except (OpsError, DataError) as error:
        print(json.dumps({"refused": error.code, "message": str(error)}))
        return 2
    except MemoryError:
        print('{"refused": "RESOURCE_LIMIT_EXCEEDED", "message": "SIZE parent memory exhausted"}')
        return 2
    except DocumentError:
        print(json.dumps({"refused": "INVALID_EXPERIMENT_SPEC", "message": "invalid SIZE metadata"}))
        return 2
    except (ArtifactError, OSError, sqlite3.Error):
        print(json.dumps({"refused": "RESOURCE_UNAVAILABLE", "message": "SIZE storage unavailable"}))
        return 2
    finally:
        stack.close()
        signal.alarm(0)
        if conn is not None:
            conn.close()
        print(f"[size] stage={args.action} end elapsed={time.monotonic() - started:.3f}s rss_kib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
