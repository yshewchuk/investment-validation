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
from engine.v2.ops.nightly import build_legacy_job_requests, refuse_oversize_plan
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


def _add_ledger_commands(commands):
    """The ``ops ledger import-history`` sub-subparser -- bootstraps a fresh
    catalog's ``decisions``/``decision_imports`` tables from the legacy JSONL
    ledger so ``legacy_settlement`` has a committed prediction to settle
    against on a catalog's first shadow night (see
    ``engine.v2.ops.ledger_history_import``)."""
    ledger = commands.add_parser("ledger")
    ledger.add_argument("--root", default=argparse.SUPPRESS)
    ledger_sub = ledger.add_subparsers(dest="ledger_command", required=True)
    import_history_p = ledger_sub.add_parser("import-history")
    import_history_p.add_argument("--source-root", required=True, type=Path,
                                  help="the legacy checkout to read ledger/predictions and "
                                       "ledger/outcomes from, read-only")
    import_history_p.add_argument("--through", default=None,
                                  help="YYYY-MM-DD; excludes predictions/outcomes dated after "
                                       "this session (by their own as_of/resolved_at field)")
    import_history_p.add_argument("--dry-run", action="store_true",
                                  help="report counts without writing anything")


def _add_snapshot_commands(commands):
    """The ``ops snapshot plan-import|submit|promote|rollback`` sub-subparsers,
    split out of :func:`parser` to keep that function under the line budget."""
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--root", default=argparse.SUPPRESS)
    snapshot_sub = snapshot.add_subparsers(dest="snapshot_command", required=True)
    plan_import_p = snapshot_sub.add_parser("plan-import")
    plan_import_p.add_argument("--source-root", required=True, type=Path)
    plan_import_p.add_argument("--scope", required=True)
    plan_import_p.add_argument("--expected-head-snapshot-id", default=None)
    plan_import_p.add_argument("--expected-head-generation", type=int, default=0)
    submit_import_p = snapshot_sub.add_parser("submit")
    submit_import_p.add_argument("plan_ref")
    submit_import_p.add_argument("--idempotency-key", required=True)
    promote_p = snapshot_sub.add_parser("promote")
    promote_p.add_argument("--candidate-scope", required=True)
    promote_p.add_argument("--target-scope", required=True)
    promote_p.add_argument("--expected-snapshot-id", default=None)
    promote_p.add_argument("--expected-generation", type=int, required=True)
    promote_p.add_argument("--comparison-receipt", required=True)
    rollback_p = snapshot_sub.add_parser("rollback")
    rollback_p.add_argument("--scope", required=True)
    rollback_p.add_argument("--to-snapshot-id", required=True)
    rollback_p.add_argument("--expected-snapshot-id", default=None)
    rollback_p.add_argument("--expected-generation", type=int, required=True)


def _add_price_refresh_command(commands):
    """``ops price-refresh --session YYYY-MM-DD [--dry-run]`` — the scheduled
    full-history yfinance re-downloader (see
    ``engine.data.pulls.price_refresh``). Kept a single small function, in its
    own subparser, so it stays a minimal diff alongside the ``price-history``
    subcommands another agent edits this same file to add."""
    price_refresh = commands.add_parser("price-refresh")
    price_refresh.add_argument("--root", default=argparse.SUPPRESS)
    price_refresh.add_argument("--session", required=True,
                               help="YYYY-MM-DD; the trading session to plan/run for")
    price_refresh.add_argument("--dry-run", action="store_true",
                               help="print plan counts only; touches disk (Tier-2 events, "
                                    "the Tier-1 fetch store) but never the network")


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
    server.add_argument("--store-root", type=Path, default=None,
                        help="the legacy checkout a snapshot-backed job's pinned read set and "
                             "materialization roots resolve against; defaults to this code "
                             "checkout (Service's own default) when omitted. Never inferred from "
                             "a plan or manifest -- always exactly what was passed here.")
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
    plan.add_argument("--context-tickers", default="",
                      help="historical evidence ticker universe: comma list or @file "
                           "(comma- or newline-separated); defaults to --tickers")
    plan.add_argument("--full-run", action="store_true",
                      help="declare this run as scoring its whole context: writes the global "
                           "'shadow' effect scope instead of a subset hash. Refused unless "
                           "--tickers equals --context-tickers exactly. Omitted (the default), "
                           "the effect scope is always the subset scope, even when the "
                           "watchlist happens to equal the context.")
    plan.add_argument("--year-start", type=int, default=2024)
    plan.add_argument("--year-end", type=int, default=2026)
    plan.add_argument("--input-mode", default="legacy", choices=("legacy", "snapshot"),
                      help="snapshot: pin one data snapshot head at plan time (P2-6)")
    plan.add_argument("--snapshot-scope", default=None)
    submission = commands.add_parser("submit")
    submission.add_argument("--plan", required=True)
    submission.add_argument("--idempotency-key", required=True)
    capture = commands.add_parser("capture-inputs")
    capture.add_argument("--as-of", required=True)
    capture.add_argument("--tickers", default="")
    capture.add_argument("--context-tickers", default="",
                         help="historical evidence ticker universe: comma list or @file; "
                              "defaults to --tickers")
    capture.add_argument("--year-start", type=int, required=True)
    capture.add_argument("--year-end", type=int, required=True)
    capture.add_argument("--source-root", required=True, type=Path)
    capture.add_argument("--output", required=True, type=Path)
    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--root", default=argparse.SUPPRESS)
    reconcile.add_argument("job_id")
    reconcile.add_argument("--expected-attempt", required=True)
    _add_snapshot_commands(commands)
    _add_ledger_commands(commands)
    _add_price_refresh_command(commands)
    for name in ("get", "logs", "cancel", "resume", "explain"):
        sub = commands.add_parser(name)
        sub.add_argument("job_id")
        sub.add_argument("--json", action="store_true")
        if name == "logs":
            sub.add_argument("--follow", action="store_true")
        if name == "cancel":
            sub.add_argument(
                "--expected-attempt", default=None,
                help="the job's active attempt id. Omit it only for a job with no active "
                     "attempt (queued or retry_wait): omission means 'expect none', so a job "
                     "that does have one refuses with STALE_EXPECTATION")
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


def _check_nightly_manifest(raw_bytes: bytes) -> None:
    """Plan-time guard (task brief deliverable 3): refuse a manifest that is
    not a nightly capture, or that lacks a family a barrier kind requires --
    the exact operator mistake behind the real 2026-09-14 failure
    (a ``snapshot_import_plan.v1`` manifest passed as ``--input-manifest`` to
    a barrier-mode nightly, which failed ``legacy_finality`` with
    ``SOURCE_NOT_FINAL`` because it declared no ``data/raw/fetch/orats``).
    Every barrier-only kind (``legacy_finality``/``legacy_decisions``/
    ``legacy_settlement``/``legacy_model_evidence``/``legacy_render``/
    ``legacy_selfcheck``) always reads this same manifest, in EVERY nightly
    plan regardless of ``--input-mode`` (``nightly.py``'s
    ``SNAPSHOT_STAGES = {"score", "decision_replay"}`` -- "every other stage
    keeps the Phase 1 barrier"), so this check applies unconditionally
    whenever ``--input-manifest`` is given for a nightly plan.
    """
    from engine.v2.data.legacy_nightly_read_plan import manifest_problems

    try:
        document = json.loads(raw_bytes)
    except ValueError as exc:
        raise fail("INPUT_CHANGED", "input manifest is not valid JSON") from exc
    problems = manifest_problems(document)
    if problems:
        raise fail("INPUT_CHANGED",
                  "input manifest is not a complete nightly capture for the barrier kinds",
                  details={"problems": problems})


def _check_submitted_nightly_manifest(plan, conn, store):
    """Deliverable 3 defence in depth: re-check the bound manifest at submit
    time too, not only at plan time -- a plan artifact can be built and
    saved by a caller other than this CLI's own ``plan nightly`` (a test, a
    future planner), and submit is the last gate before jobs are created.
    """
    if not plan.get("input_manifest_ref"):
        return
    _check_nightly_manifest(store.read_verified(artifact(conn, store, plan["input_manifest_ref"])))


def _read_input_manifest_ref(args, root, conn, clock):
    if not args.input_manifest:
        return None
    if not args.input_manifest.is_file() or args.input_manifest.is_symlink():
        raise fail("INPUT_CHANGED", "input manifest is missing")
    raw_bytes = args.input_manifest.read_bytes()
    _check_nightly_manifest(raw_bytes)
    manifest = ArtifactStore(root).publish_bytes(
        raw_bytes, schema_ref="legacy_input_manifest.v1.0")
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


def _ticker_list(value):
    """Comma list or ``@file`` (comma- or newline-separated ticker file)."""
    if not value:
        return ()
    raw = value
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.is_file() or path.is_symlink():
            raise fail("INPUT_CHANGED", "ticker list file is missing")
        raw = path.read_text()
    return tuple(filter(None, (part.strip() for part in raw.replace(",", "\n").splitlines())))


def _snapshot_inputs(args, root, conn, clock, context_tickers, population):
    """``--input-mode snapshot``: resolve the scope's head exactly once, here.

    ``context_tickers`` (P2-C04) is the historical evidence universe — the
    scope :func:`engine.v2.ops.snapshot_planning.pin_snapshot_inputs` builds
    its evidence years/tickers from — never the narrower direct watchlist.
    """
    if args.input_mode != "snapshot":
        return None
    if not args.snapshot_scope:
        raise fail("INVALID_REQUEST", "snapshot input mode needs --snapshot-scope")
    from engine.v2.ops.snapshot_planning import pin_snapshot_inputs
    return pin_snapshot_inputs(conn, ArtifactStore(root), args.snapshot_scope,
                               tickers=context_tickers, year_start=args.year_start,
                               year_end=args.year_end, expected_population=population, clock=clock)


def _plan_command(args, root, conn, clock):
    if args.kind == "nightly":
        tickers = _ticker_list(args.tickers)
        context_tickers = _ticker_list(args.context_tickers) or tickers
        population = _read_expected_population(args)
        plan = nightly_plan(Path(__file__).resolve().parents[3], args.as_of,
                            mode=args.mode, manifest_ref=_read_input_manifest_ref(args, root, conn, clock),
                            tickers=tickers, context_tickers=context_tickers,
                            year_start=args.year_start, year_end=args.year_end,
                            expected_population=population, clock=clock,
                            input_mode=args.input_mode, full_run=args.full_run,
                            snapshot_inputs=_snapshot_inputs(args, root, conn, clock, context_tickers,
                                                             population))
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
        store_root = getattr(args, "store_root", None)
        if store_root is not None and not store_root.is_dir():
            raise fail("INVALID_REQUEST", "--store-root must be an existing directory",
                      details={"store_root": str(store_root)})
        service = Service(conn, root, registry(), DEFAULT_POLICY, clock=clock,
                          code_source=Path(__file__).resolve().parents[3], store_root=store_root)
        serve(service, once=args.once)
        return {"stopped": True}
    if args.command == "plan":
        return _plan_command(args, root, conn, clock)
    if args.command == "submit":
        return _submit_command(args, root, conn, clock)
    if args.command == "reconcile":
        return reconcile_command(args, root, conn, clock)
    if args.command == "snapshot":
        return snapshot_command(args, root, conn, clock)
    if args.command == "ledger":
        return ledger_command(args, root, conn, clock)
    return job_command(args, conn, clock)


def _submit_nightly(plan, conn, store, policy, clock):
    # guide §5.5 item 1 / rearchitecture_phase1_runbook.md
    # "--idempotency-key semantics for nightly submission": this flag
    # is REQUIRED by the CLI grammar (every ``submit`` needs one) but
    # is NEVER read for a nightly plan. Nightly stage identity is
    # fully and only determined by the plan document itself (session,
    # scope, and the pinned plan identity -- implementation, legacy
    # manifest, decision_clock -- nightly.py's ``_plan_identity``), so
    # two ``submit --plan <same-plan-ref>`` calls with DIFFERENT
    # ``--idempotency-key`` values are still the same retry and
    # resolve to the identical jobs; the key only distinguishes
    # submissions of a non-nightly (``artifact_check``/experiment)
    # plan below, where it IS the job identity.
    if plan.get("blocked_prerequisites"):
        raise fail("INVALID_REQUEST", "nightly plan has unresolved prerequisites",
                  details={"blocked_prerequisites": plan["blocked_prerequisites"]})
    _check_submitted_nightly_manifest(plan, conn, store)
    context_tickers = tuple(plan.get("context_tickers", ()))
    # P2-5 collision fix / effect-scope decision: only a plan built
    # with ``--full-run`` declares the global universe; every other
    # plan gets a subset effect scope even when the watchlist equals
    # the context (nightly.effect_scope_for).
    full_universe = context_tickers if plan.get("full_run") else None
    requests = build_legacy_job_requests(
        plan, tickers=tuple(plan.get("tickers", ())),
        context_tickers=context_tickers,
        year_start=plan["year_start"], year_end=plan["year_end"],
        input_refs=(plan["input_manifest_ref"],),
        expected_population=tuple(plan.get("expected_population", ())),
        include_prerequisites=False, input_mode=plan.get("input_mode", "legacy"),
        snapshot_inputs=plan.get("snapshot_inputs"), full_universe=full_universe)
    # 2026-09-14: refuse before submission a plan that would only
    # fail later at claim time (RESOURCE_LIMIT_EXCEEDED) because some
    # job's legacy read set exceeds its resource profile's scratch
    # budget -- see nightly.plan_scratch_problems.
    refuse_oversize_plan(conn, store, requests)
    receipts = submit_graph(conn, registry(), policy, requests, clock=clock)
    return {"run_id": "run_" + plan["plan_hash"][:24],
            "jobs": [to_document(item) for item in receipts]}


def _submit_command(args, root, conn, clock):
    store = ArtifactStore(root)
    ref = artifact(conn, store, args.plan)
    plan = json.loads(store.read_verified(ref))
    policy = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})
    if plan.get("kind") == "nightly":
        return _submit_nightly(plan, conn, store, policy, clock)
    return submit(conn, registry(), policy, request_from_plan(plan, args.idempotency_key), clock=clock)


# --------------------------------------------------------------------------
# snapshot import/promotion/rollback (P2-7/Task7b, §7/§10)
# --------------------------------------------------------------------------


def snapshot_command(args, root, conn, clock):
    """``ops snapshot plan-import|submit|promote|rollback`` — see
    ``engine.v2.data.import_snapshot``/``engine.v2.ops.snapshot_import``/
    ``engine.v2.ops.snapshot_promotion`` for what each step actually does.
    """
    from engine.v2.data.import_snapshot import plan_import
    from engine.v2.ops.snapshot_import import save_import_plan, submit_import
    from engine.v2.ops.snapshot_promotion import promote as promote_snapshot
    from engine.v2.ops.snapshot_promotion import rollback as rollback_snapshot

    store = ArtifactStore(root)
    repo_root = Path(__file__).resolve().parents[3]
    if args.snapshot_command == "plan-import":
        plan = plan_import(args.source_root, scope=args.scope,
                           expected_head_snapshot_id=args.expected_head_snapshot_id,
                           expected_head_generation=args.expected_head_generation)
        return {"plan_ref": save_import_plan(conn, store, plan, clock=clock).artifact_id}
    if args.snapshot_command == "submit":
        policy = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})
        receipt = submit_import(conn, store, args.plan_ref, registry=registry(), policy=policy,
                                clock=clock, idempotency_key=args.idempotency_key,
                                repo_root=repo_root)
        return to_document(receipt)
    if args.snapshot_command == "promote":
        ref = promote_snapshot(conn, store, candidate_scope=args.candidate_scope,
                               target_scope=args.target_scope,
                               expected_snapshot_id=args.expected_snapshot_id,
                               expected_generation=args.expected_generation,
                               comparison_receipt_id=args.comparison_receipt, clock=clock)
        return {"receipt_ref": ref.artifact_id}
    ref = rollback_snapshot(conn, store, scope=args.scope, to_snapshot_id=args.to_snapshot_id,
                            expected_snapshot_id=args.expected_snapshot_id,
                            expected_generation=args.expected_generation, clock=clock)
    return {"receipt_ref": ref.artifact_id}


def ledger_command(args, root, conn, clock):
    """``ops ledger import-history`` -- see ``engine.v2.ops.ledger_history_import``."""
    from datetime import date

    from engine.v2.ops.ledger_history_import import import_history

    through = date.fromisoformat(args.through) if args.through else None
    if not args.source_root.is_dir():
        raise fail("INVALID_REQUEST", "--source-root must be an existing directory",
                  details={"source_root": str(args.source_root)})
    return import_history(conn, root, args.source_root, through=through,
                          dry_run=args.dry_run, clock=clock)


def capture_command(args):
    """``ops capture-inputs`` (task brief deliverable 2): the supported
    capture for the shadow nightly's ``--input-manifest``. Pure filesystem
    work under ``--source-root`` -- no operations catalog, no clock, no
    network -- so it is dispatched before ``main()`` opens one, exactly like
    ``doctor``.
    """
    from engine.v2.ops.capture_inputs import capture, write_manifest

    tickers = _ticker_list(args.tickers)
    context_tickers = _ticker_list(args.context_tickers) or tickers
    manifest = capture(args.source_root, as_of=args.as_of, tickers=tickers,
                       context_tickers=context_tickers, year_start=args.year_start,
                       year_end=args.year_end)
    write_manifest(manifest, args.output)
    return {"schema_version": "capture_inputs_report.v1.0", "output": str(args.output),
            "manifest_id": manifest.manifest_id, "file_count": len(manifest.file_refs),
            "total_bytes": sum(ref.byte_size for ref in manifest.file_refs)}


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


def price_refresh_command(args, root):
    """``ops price-refresh``. No catalog/job dependency — a plain data pull —
    so it is dispatched in :func:`main` before the catalog is opened, the same
    way ``capture-inputs`` is. Crosses into ``engine.data.*`` only through
    ``engine.v2.ops.legacy_adapter.invoke_price_refresh`` — this module may
    not import legacy code directly (one adapter module per package,
    ``checks/import_layers.py`` §4.2). ``--dry-run`` returns after planning,
    never constructing a :class:`~engine.data.fetch.Fetcher`, so it makes no
    provider call by construction. A real run's report lands at
    ``<root>/price_refresh/<session>.json``."""
    from engine.v2.ops.legacy_adapter import invoke_price_refresh

    result = invoke_price_refresh(args.session, dry_run=args.dry_run)
    plan = result["plan"]
    if args.dry_run:
        return {"session": plan["session"], "dry_run": True,
                "counts": {"daily": len(plan["daily"]),
                          "monthly": len(plan["monthly"]),
                          "skipped_already_fetched": len(plan["skipped_already_fetched"])}}

    report = result["report"]
    out_dir = root / "price_refresh"
    ensure_directory(out_dir)
    out_path = out_dir / f"{plan['session']}.json"
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(json.dumps(to_document(report), indent=2))
    tmp_path.replace(out_path)
    report["report_path"] = str(out_path)
    return report


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
        elif args.command == "capture-inputs":
            document = capture_command(args)
        elif args.command == "price-refresh":
            document = price_refresh_command(args, root)
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
    except (OSError, ValueError, TypeError) as exc:
        # Redacted: never the exception's own text (it may carry a value or a
        # path), only its class name -- enough to tell a bad unpack from a
        # missing file from a bad type without ever printing what triggered it.
        print(json.dumps({"code": "INVALID_REQUEST",
                          "message": "command could not read or validate its inputs",
                          "details": {"exception_type": type(exc).__name__}}))
        return 2
