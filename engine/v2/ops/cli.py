"""Local operator commands; no timers, provider pulls or production writes on init."""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from engine.v2.foundation import (
    ArtifactStore,
    SystemClock,
    content_hash,
    ensure_directory,
    to_document,
)
from engine.v2.ops import executor
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import integrity_errors, transaction
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.diagnostics import report as diagnostic_report
from engine.v2.ops.discovery import sample_capacity
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.executor_watchdog import signal_owned
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.health import health, write_health
from engine.v2.ops.lifecycle import attempt_receipts, request_cancel
from engine.v2.ops.nightly import build_legacy_job_requests
from engine.v2.ops.plans import nightly_plan, request_from_plan, save_plan
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.recovery import (
    SupervisorLock,
    prove_ownership_gone,
    read_boot_id,
    reconcile_attempt,
)
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, get_job, submit, submit_graph
from engine.v2.ops.supervisor import Service, serve


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", default="data/operations")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("init", "doctor", "health"):
        sub = commands.add_parser(name)
        sub.add_argument("--root", default=argparse.SUPPRESS)
        sub.add_argument("--json", action="store_true")
        if name == "health":
            sub.add_argument("--out", type=Path, default=None,
                             help="also write the authenticated sidecar health.json here")
    server = commands.add_parser("serve")
    server.add_argument("--root", default=argparse.SUPPRESS)
    server.add_argument("--once", action="store_true")
    plan = commands.add_parser("plan")
    plan.add_argument("kind", choices=("nightly", "experiment"))
    plan.add_argument("--as-of")
    plan.add_argument("--mode", default="shadow", choices=("shadow",))
    plan.add_argument("--spec", type=Path)
    plan.add_argument("--no-ledger", action="store_true")
    plan.add_argument("--input-manifest", type=Path)
    plan.add_argument("--expected-population", type=Path,
                      help="JSON file: a list of 'ticker|strategy|event_date' keys")
    plan.add_argument("--tickers", default="")
    plan.add_argument("--year-start", type=int, default=2024)
    plan.add_argument("--year-end", type=int, default=2026)
    submission = commands.add_parser("submit")
    submission.add_argument("--plan", required=True)
    submission.add_argument("--idempotency-key", required=True)
    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--root", default=argparse.SUPPRESS)
    reconcile.add_argument("job_id")
    reconcile.add_argument("--expected-attempt", required=True)
    for name in ("get", "logs", "cancel", "resume", "explain"):
        sub = commands.add_parser(name)
        sub.add_argument("job_id")
        sub.add_argument("--json", action="store_true")
        if name == "logs":
            sub.add_argument("--follow", action="store_true")
        if name == "cancel":
            sub.add_argument("--expected-attempt", required=True)
        if name == "resume":
            sub.add_argument("--dry-run", action="store_true")
    return result


def doctor(root, clock):
    nearest = root if root.exists() else root.parent
    while not nearest.exists():
        nearest = nearest.parent
    capacity = to_document(sample_capacity(nearest, clock=clock))
    catalog = root / "catalog.sqlite"
    result = {"schema_version": "operations_doctor.v1.0", "capacity": capacity,
              "catalog_exists": catalog.exists(), "activation": "shadow_only"}
    if catalog.exists():
        conn = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True)
        try:
            result["diagnostics"] = diagnostic_report(conn, root, boot_id=read_boot_id())
            result["integrity_errors"] = integrity_errors(conn)
            result["schema_versions"] = [list(row) for row in conn.execute(
                "SELECT owner,version,checksum FROM schema_versions")]
        finally:
            conn.close()
    return result


def _read_input_manifest_ref(args, root, conn, clock):
    if not args.input_manifest:
        return None
    if not args.input_manifest.is_file() or args.input_manifest.is_symlink():
        raise fail("INPUT_CHANGED", "input manifest is missing")
    manifest = ArtifactStore(root).publish_bytes(
        args.input_manifest.read_bytes(), schema_ref="legacy_input_manifest.v1.0")
    from engine.v2.ops.checkpoints import register_artifact
    with transaction(conn):
        register_artifact(conn, manifest, None, clock)
    return manifest.artifact_id


def _read_expected_population(args):
    if not args.expected_population:
        return ()
    if not args.expected_population.is_file() or args.expected_population.is_symlink():
        raise fail("INPUT_CHANGED", "expected population file is missing")
    payload = json.loads(args.expected_population.read_text())
    if not isinstance(payload, list) or not all(isinstance(v, str) for v in payload):
        raise fail("INVALID_REQUEST", "expected population must be a JSON list of strings")
    return tuple(payload)


def _plan_command(args, root, conn, clock):
    if args.kind == "nightly":
        plan = nightly_plan(Path(__file__).resolve().parents[3], args.as_of,
                            mode=args.mode, manifest_ref=_read_input_manifest_ref(args, root, conn, clock),
                            tickers=tuple(filter(None, args.tickers.split(","))),
                            year_start=args.year_start, year_end=args.year_end,
                            expected_population=_read_expected_population(args))
    else:
        from engine.v2.ops.experiments import experiment_plan
        plan = experiment_plan(args.spec, smoke=args.no_ledger)
    ref = save_plan(conn, root, plan, clock=clock)
    return {"plan_ref": ref.artifact_id, "plan": plan}


def dispatch(args, root, conn, clock):
    if args.command == "init":
        return {"initialized": True, "activation": "shadow_only"}
    if args.command == "health":
        document = health(conn, clock=clock)
        if getattr(args, "out", None) is not None:
            write_health(args.out, document)
        return document
    if args.command == "serve":
        service = Service(conn, root, registry(), DEFAULT_POLICY, clock=clock,
                          code_source=Path(__file__).resolve().parents[3])
        serve(service, once=args.once)
        return {"stopped": True}
    if args.command == "plan":
        return _plan_command(args, root, conn, clock)
    if args.command == "submit":
        store = ArtifactStore(root)
        ref = artifact(conn, store, args.plan)
        plan = json.loads(store.read_verified(ref))
        policy = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})
        if plan.get("kind") == "nightly":
            if plan.get("blocked_prerequisites"):
                raise fail("INVALID_REQUEST", "nightly plan has unresolved prerequisites",
                          details={"blocked_prerequisites": plan["blocked_prerequisites"]})
            requests = build_legacy_job_requests(
                plan, tickers=tuple(plan.get("tickers", ())),
                year_start=plan["year_start"], year_end=plan["year_end"],
                input_refs=(plan["input_manifest_ref"],),
                expected_population=tuple(plan.get("expected_population", ())),
                include_prerequisites=False)
            receipts = submit_graph(conn, registry(), policy, requests, clock=clock)
            return {"run_id": "run_" + plan["plan_hash"][:24],
                    "jobs": [to_document(item) for item in receipts]}
        return submit(conn, registry(), policy, request_from_plan(plan, args.idempotency_key), clock=clock)
    if args.command == "reconcile":
        return reconcile_command(args, root, conn, clock)
    return job_command(args, conn, clock)


def reconcile_command(args, root, conn, clock):
    """Settle one ``recovery_pending`` attempt by hand, when no supervisor is ticking.

    Refuses outright if a running supervisor holds the lock (it already
    reconciles every tick). Otherwise runs the identical B1 ownership proof
    and settles only if it passes; there is no force flag, so a tree that
    cannot be proven gone stays quarantined and this prints its blockers.
    """
    lock = SupervisorLock(root / "supervisor.lock")
    if not lock.acquire():
        raise fail("RESOURCE_UNAVAILABLE",
                   "a running supervisor already reconciles this catalog")
    try:
        job = get_job(conn, args.job_id)
        if job.active_attempt_id != args.expected_attempt:
            raise fail("STALE_EXPECTATION",
                       "the expected attempt is not the job's active attempt")
        attempt = conn.execute("SELECT state FROM attempts WHERE attempt_id = ?",
                               (args.expected_attempt,)).fetchone()
        if attempt is None or attempt["state"] != "recovery_pending":
            raise fail("STALE_EXPECTATION", "the attempt is not awaiting reconciliation")
        boot_id = read_boot_id()
        proof = prove_ownership_gone(conn, args.expected_attempt, boot_id=boot_id)
        if proof.known:
            executor.persist_members(conn, args.expected_attempt, proof.known)
        signal_owned(proof.alive, boot_id, hard=True)
        if not proof.proven:
            raise fail("RESOURCE_UNAVAILABLE", "the process tree is not verified gone",
                      details={"blocking": [{"pid": pid, "start_ticks": ticks}
                                             for pid, ticks in proof.blockers]})
        state = reconcile_attempt(conn, args.expected_attempt, process_state="verified_dead",
                                  clock=clock)
        return {"attempt_id": args.expected_attempt, "settled": True, "state": state}
    finally:
        lock.release()


def job_command(args, conn, clock):
    if args.command == "get":
        return {"job": to_document(get_job(conn, args.job_id)),
                "attempts": to_document(attempt_receipts(conn, args.job_id))}
    if args.command == "cancel":
        return request_cancel(conn, args.job_id, args.expected_attempt, clock=clock)
    if args.command == "resume":
        return resume_command(args, conn)
    if args.command == "explain":
        return explain_command(args, conn)
    return logs_command(args, conn)


def resume_command(args, conn):
    job = get_job(conn, args.job_id)
    if not args.dry_run:
        raise fail("INVALID_REQUEST", "use the saved immutable plan to submit a changed run")
    row = conn.execute("SELECT spec_json FROM jobs WHERE job_id = ?", (args.job_id,)).fetchone()
    spec = json.loads(row[0])
    impl = content_hash(worker_source_manifest(Path(__file__).resolve().parents[3]))
    env = content_hash(environment_identity(
        job.resolved_resources.thread_count if job.resolved_resources else 1))
    reasons = [name for name, expected, actual in (
        ("implementation", spec.get("implementation_ref"), impl),
        ("environment", spec.get("environment_ref"), env)) if expected != actual]
    return {"job": to_document(job),
            "action": "automatic_retry" if job.state == "retry_wait" else job.state,
            "invalidation": {"reasons": reasons,
                             "recompute_from": job.kind if reasons else None,
                             "checkpoint_reuse": not reasons},
            "new_effects_authorized": False}


def explain_command(args, conn):
    job = get_job(conn, args.job_id)
    return {"job_id": args.job_id, "state": job.state,
            "queue_reason": to_document(job.queue_reason),
            "failure": to_document(job.failure)}


def logs_command(args, conn):
    while True:
        rows = conn.execute("SELECT body_json FROM progress_events WHERE job_id=? "
                            "ORDER BY recorded_at,sequence", (args.job_id,)).fetchall()
        if not args.follow:
            return [json.loads(row[0]) for row in rows]
        print(json.dumps([json.loads(row[0]) for row in rows]), flush=True)
        if get_job(conn, args.job_id).state in ("succeeded", "failed", "cancelled", "blocked"):
            return {"complete": True}
        time.sleep(2)


def main(argv=None):
    args, clock = parser().parse_args(argv), SystemClock()
    root = Path(args.root).resolve()
    try:
        if args.command == "doctor":
            document = doctor(root, clock)
        else:
            if args.command == "init":
                ensure_directory(root)
                root.chmod(0o700)
            if not root.is_dir():
                raise fail("INVALID_REQUEST", "operations root must be initialized first")
            conn = open_catalog(root / "catalog.sqlite", clock=clock)
            try:
                document = dispatch(args, root, conn, clock)
            finally:
                conn.close()
        print(json.dumps(to_document(document), indent=2))
        return 0
    except OpsError as exc:
        print(json.dumps(to_document(exc.problem)))
        return 2
    except (OSError, ValueError, TypeError):
        print(json.dumps({"code": "INVALID_REQUEST", "message": "command could not read or validate its inputs"}))
        return 2
