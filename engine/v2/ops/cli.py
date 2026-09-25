"""Local operator commands; no timers, provider pulls or production writes on init."""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from engine.v2.foundation import (
    ArtifactStore,
    DocumentError,
    SystemClock,
    content_hash,
    ensure_directory,
    from_document,
    parse_timestamp,
    to_document,
)
from engine.v2.ops import executor
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import integrity_errors, transaction
from engine.v2.ops.checkpoints import artifact, register_artifact
from engine.v2.ops.diagnostics import report as diagnostic_report
from engine.v2.ops.discovery import sample_capacity
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.executor_watchdog import signal_owned
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.health import health, write_health
from engine.v2.ops.lifecycle import attempt_receipts, request_cancel
from engine.v2.ops.nightly import (
    build_legacy_job_requests,
    refuse_oversize_plan,
    refuse_unfittable_memory_plan,
)
from engine.v2.ops.plans import nightly_plan, request_from_plan, save_plan
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.recovery import (
    SupervisorLock,
    prove_ownership_gone,
    read_boot_id,
    reconcile_attempt,
)
from engine.v2.ops.session_backfill import backfill_outcome_sessions
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, get_job, submit, submit_graph
from engine.v2.ops.supervisor import Service, serve


def _add_ledger_commands(commands):
    """``ops ledger import-history|status|calibrate|book`` -- see
    ``engine.v2.ops.ledger_history_import`` and, for the last three (P6-3),
    ``engine.v2.ledger.status``/``calibration``/``portfolio``: the native v2
    ledger-status summary, calibration/health recompute and hypothetical-book
    accounting over catalog decisions, matching legacy
    ``engine/ledger.py::status``/``calibrate`` and
    ``engine/portfolio.py::build_book``/``summarize``."""
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

    ledger_sub.add_parser("status", help="counts, duplicates and pending settlement "
                                         "over catalog decisions/outcomes")

    calibrate_p = ledger_sub.add_parser(
        "calibrate", help="regenerate the calibration report and health payload")
    calibrate_p.add_argument("--force", action="store_true")
    calibrate_p.add_argument("--trigger", type=int, default=None,
                             help="newly scored rows needed to trigger a recompute "
                                  "(default: engine.v2.ledger.calibration.CALIBRATION_TRIGGER)")

    book_p = ledger_sub.add_parser(
        "book", help="the hypothetical book, capital per trade and funding "
                     "over catalog decisions")
    book_p.add_argument("--contracts", type=int, default=None,
                        help="size by a fixed contract count instead of equal dollars")
    book_p.add_argument("--capital-per-trade", type=float, default=None)
    book_p.add_argument("--include-declined", action="store_true",
                        help="also book the gate's rejections, tagged recommended=False")


def _add_price_history_commands(commands):
    """The ``ops price-history capture`` sub-subparser (task brief
    2026-09-14; Tier-2 rework 2026-09-14 SEND-BACK): reads the two legacy
    yfinance sources read-only, captures them into the ``price_history``
    Tier-2 catalog table, and commits a new snapshot generation under
    ``--scope`` carrying every other table's dataset version forward
    unchanged alongside it (see ``engine.v2.ops.price_history_store``). Like
    ``snapshot plan-import``, this needs the shared operations catalog, so
    (unlike the ``capture-inputs``/``doctor`` pure-filesystem commands) it is
    dispatched from inside ``dispatch()``, after ``main()`` opens one.
    """
    price_history = commands.add_parser("price-history")
    price_history.add_argument("--root", default=argparse.SUPPRESS)
    price_history_sub = price_history.add_subparsers(dest="price_history_command", required=True)
    capture_p = price_history_sub.add_parser("capture")
    capture_p.add_argument("--source-root", required=True, type=Path,
                           help="the legacy checkout to read the px csv tree and the Tier-1 "
                                "yfinance fetch cache from, read-only")
    capture_p.add_argument("--scope", required=True,
                           help="the snapshot scope to add/advance price_history's dataset "
                                "version in (e.g. 'shadow'); must already have a head snapshot")
    capture_p.add_argument("--dry-run", action="store_true",
                           help="report counts; every check still runs, but nothing is written "
                                "and no snapshot is committed")


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


def _add_refresh_mode_arguments(plan):
    """R3B-3: ``--refresh-mode native [--refresh-plan <file>]`` submits one
    real ``incremental_refresh`` job as part of the nightly DAG (see
    ``nightly._resolve_refresh_plan``). Default ``legacy`` submits none --
    ``ops price-refresh`` remains the only refresh path, unchanged."""
    plan.add_argument("--refresh-mode", default="legacy", choices=("legacy", "native"))
    plan.add_argument("--refresh-plan", type=Path, default=None,
                      help="optional pinned RefreshPlan document (to_document()'d JSON); "
                           "when omitted the shadow head builds the plan at submission")


def _add_operator_plan_arguments(plan):
    """``ops plan training``/``promote``'s arguments (P6 slice 5). ``--mode``
    is already nightly's, hence ``--training-mode``."""
    plan.add_argument("--training-mode",
                      choices=("recipe", "state", "board_analog", "trailing_cutoff"))
    plan.add_argument("--recipe", default="")
    plan.add_argument("--state", default="")
    plan.add_argument("--alpha", type=float)
    plan.add_argument("--cutoff", action="append", default=[])
    plan.add_argument("--strategy", action="append", default=[])
    plan.add_argument("--pairs", default="")
    plan.add_argument("--ticker-chunk", type=int, default=1000)
    plan.add_argument("--release-root", default="")
    plan.add_argument("--release-id", default="")


def _add_reconcile_command(commands):
    """The ``ops reconcile`` subparser, split out of :func:`parser` to keep
    that function under the line budget."""
    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--root", default=argparse.SUPPRESS)
    reconcile.add_argument("job_id")
    reconcile.add_argument("--expected-attempt", required=True)


def _add_provider_account_command(commands):
    """S4B2: ``ops provider-account``, the one production writer of the
    ``provider_accounts`` row the scheduler reserves against
    (``scheduler._reserve_provider`` refuses with CREDENTIAL_INVALID when the
    row is absent). Only budget numbers live here -- never a credential."""
    provider = commands.add_parser("provider-account")
    provider.add_argument("--root", default=argparse.SUPPRESS)
    provider.add_argument("--account", required=True)
    provider.add_argument("--remaining", type=int, required=True,
                          help="total provider calls the account may spend")
    provider.add_argument("--live-reserve", type=int, required=True,
                          help="calls held back from ordinary admission")


def _add_rescore_command(commands):
    """The read-only ``ops rescore`` subparser (see :func:`rescore_command`)."""
    rescore = commands.add_parser("rescore")
    rescore.add_argument("--request", type=Path, required=True,
                         help="a ScoreRequest canonical JSON document (to_document output)")
    rescore.add_argument("--native-inputs", type=Path, required=True,
                         help="a NativeScoreInputs canonical JSON document (to_document output); "
                              "already-captured data only, no provider pulls, no fitting")


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
    plan.add_argument("kind", choices=("nightly", "experiment", "training", "promote"))
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
    _add_operator_plan_arguments(plan)
    _add_refresh_mode_arguments(plan)
    submission = commands.add_parser("submit")
    submission.add_argument("--plan", required=True)
    submission.add_argument("--idempotency-key", required=True)
    _add_rescore_command(commands)
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
    _add_reconcile_command(commands)
    _add_provider_account_command(commands)
    _add_snapshot_commands(commands)
    _add_ledger_commands(commands)
    _add_price_refresh_command(commands)
    _add_price_history_commands(commands)
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
            # 2026-09-15: a NULL generation_ref outcome row can never dedupe
            # a same-session legacy_settlement rerun (decision_commit.
            # _match_same_session matches only a recorded generation_ref).
            # backfill_outcome_sessions (engine.v2.ops.session_backfill)
            # runs automatically on every write-opened catalog (cli.main's
            # _ensure_outcome_sessions_backfilled), so a nonzero count here
            # means either the backfill has not yet had a chance to run
            # against this root (a fresh copy nobody has "ops serve"d or
            # "ops ledger import-history"d yet) or those specific rows were
            # underivable and are staying NULL by design -- surfaced,
            # never hidden, so an operator can tell the two apart.
            applied = any(row[0] == "outcome_session_backfill" for row in result["schema_versions"])
            result["outcome_sessions"] = {
                "backfill_applied": applied,
                "undetermined": conn.execute(
                    "SELECT COUNT(*) FROM decisions WHERE kind='outcome' "
                    "AND generation_ref IS NULL").fetchone()[0]}
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
                               year_end=args.year_end, expected_population=population, clock=clock,
                               session=args.as_of)


def _read_refresh_plan(args):
    """R3B-3/S4B2: ``--refresh-mode native``'s optional pinned ``RefreshPlan``.

    ``--refresh-plan`` is now purely an override: omitted, the caller is
    asking the submit path to build the production plan from the shadow head
    (``nightly.build_legacy_job_requests``/``_build_native_refresh_plan``).
    Resolving a real plan from live cache inventory is the data layer's job
    (``engine.v2.ops.incremental_data.plan_refresh``), not this CLI's -- like
    ``--input-manifest``, this reads a document the caller already resolved.
    """
    if args.refresh_mode != "native":
        return None
    if not args.refresh_plan:
        return None
    if not args.refresh_plan.is_file() or args.refresh_plan.is_symlink():
        raise fail("INPUT_CHANGED", "refresh plan file is missing")
    return json.loads(args.refresh_plan.read_text())


def _plan_command(args, root, conn, clock):
    if args.kind == "nightly":
        tickers = _ticker_list(args.tickers)
        context_tickers = _ticker_list(args.context_tickers) or tickers
        population = _read_expected_population(args)
        from engine.v2.ops.snapshot_stages import _catalog_path
        plan = nightly_plan(Path(__file__).resolve().parents[3], args.as_of,
                            mode=args.mode, manifest_ref=_read_input_manifest_ref(args, root, conn, clock),
                            tickers=tickers, context_tickers=context_tickers,
                            year_start=args.year_start, year_end=args.year_end,
                            expected_population=population, clock=clock,
                            input_mode=args.input_mode, full_run=args.full_run,
                            snapshot_inputs=_snapshot_inputs(args, root, conn, clock, context_tickers,
                                                             population),
                            refresh_mode=args.refresh_mode, refresh_plan=_read_refresh_plan(args),
                            catalog_path=_catalog_path(conn), objects_root=str(root))
    elif args.kind == "training":
        from engine.v2.ops.training import training_plan
        plan = training_plan(mode=args.training_mode, recipe=args.recipe or "",
                             state=args.state or "", alpha=args.alpha,
                             cutoffs=tuple(args.cutoff), strategies=tuple(args.strategy),
                             pairs_path=args.pairs or "", ticker_chunk=args.ticker_chunk,
                             manifest_ref=_read_input_manifest_ref(args, root, conn, clock))
    elif args.kind == "promote":
        from engine.v2.ops.training import promote_plan
        plan = promote_plan(release_root=args.release_root, release_id=args.release_id)
    else:
        from engine.v2.ops.experiments import experiment_plan
        plan = experiment_plan(args.spec, smoke=args.no_ledger)
    ref = save_plan(conn, root, plan, clock=clock)
    return {"plan_ref": ref.artifact_id, "plan": plan}


def _ensure_outcome_sessions_backfilled(conn, root, clock):
    """Run the outcome-session data migration exactly once per catalog,
    before the first ``legacy_settlement`` commit in this process can see a
    ``generation_ref IS NULL`` outcome row (``engine.v2.ops.session_backfill``;
    2026-09-15: without a recorded ``generation_ref``,
    ``decision_commit._match_same_session`` can never dedupe a same-session
    rerun against that row).

    Called from ``main()`` right after ``open_catalog`` returns -- the one
    place in the CLI a production catalog connection AND its artifact store
    are both already in hand (``ArtifactStore(root)``, byte-identical to the
    store every command below constructs and to ``supervisor.Service``'s own
    ``self.store``), and ahead of every real writer of a ``kind="outcome"``
    decision: ``ops serve`` (the nightly's ``legacy_settlement`` action) and
    ``ops ledger import-history`` both dispatch from here. ``ops init``
    reaches this too -- a no-op on the empty catalog it just created, but it
    marks the one-shot ``schema_versions`` bookkeeping applied immediately,
    so the very first real command afterwards costs only that marker's
    SELECT, never an artifact scan.

    Idempotent and check-first by construction (not by anything added here):
    ``backfill_outcome_sessions`` itself reads the marker before touching
    any artifact or row, so a catalog that already has it applied is one
    SELECT, never a rescan. It runs the whole recovery -- reading every
    nightly candidate artifact this catalog still holds, and updating every
    NULL row -- inside ONE transaction (``engine.v2.ops.catalog.transaction``);
    a crash or error partway through rolls the entire attempt back rather
    than leaving some rows stamped and others not, so the marker is never
    written on a partial pass and the next process attempts the complete
    backfill again from scratch (safe and cheap: it only ever touches
    ``generation_ref IS NULL`` rows and reads artifacts read-only).
    """
    return backfill_outcome_sessions(conn, ArtifactStore(root), clock=clock)


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
    if args.command == "price-history":
        return price_history_command(args, root, conn, clock)
    if args.command == "provider-account":
        return provider_account_command(args, conn)
    return job_command(args, conn, clock, root)


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
        snapshot_inputs=plan.get("snapshot_inputs"), full_universe=full_universe,
        refresh_mode=plan.get("refresh_mode", "legacy"), refresh_plan=plan.get("refresh_plan"),
        catalog_path=plan.get("catalog_path"), objects_root=plan.get("objects_root"),
        conn=conn, store=store, clock=clock)
    # 2026-09-14: refuse before submission a plan that would only
    # fail later at claim time (RESOURCE_LIMIT_EXCEEDED) because some
    # job's legacy read set exceeds its resource profile's scratch
    # budget -- see nightly.plan_scratch_problems.
    refuse_oversize_plan(conn, store, requests)
    # §8.1 (2026-09-15, legacy_score v5 6 GiB incident follow-up): refuse
    # before submission a plan naming a resource profile too big for this
    # host to EVER admit -- a structural check (host_total-based), never the
    # live host_available_bytes a claim-time sample reads, so it cannot trip
    # on another process's transient memory use.
    refuse_unfittable_memory_plan(requests, policy=DEFAULT_POLICY,
                                  sample=sample_capacity(store.root, clock=clock))
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
    if plan.get("kind") == "experiment":
        spec_ref = store.publish_bytes(json.dumps(plan["spec_document"], sort_keys=True).encode(),
                                       schema_ref="experiment_spec.v1.0")
        with transaction(conn):
            register_artifact(conn, spec_ref, None, clock)
        plan["input_refs"] = [spec_ref.artifact_id]
        plan["parameters"]["input_bindings"] = {"spec.json": spec_ref.artifact_id}
    return submit(conn, registry(), policy, request_from_plan(plan, args.idempotency_key), clock=clock)


_PROBLEM_STATUS = {"validation": 400, "dependency": 409, "resource": 503,
                   "source": 502, "integrity": 404, "internal": 500}


def _status_for(category: str) -> int:
    return _PROBLEM_STATUS.get(category, 500)


def refresh_action(root: Path, payload, *, clock=None) -> tuple[int, dict]:
    """Submit the nightly DAG named by an already-published plan artifact.

    ``payload`` is the parsed JSON body of a POST to the operations server's
    refresh action: ``{"plan_ref": "<artifact id from `ops plan nightly`>"}``.
    Never runs the nightly itself — only queues it via the same
    ``submit_graph`` path ``ops submit`` uses. Duplicate-submit protection is
    inherited for free: ``build_legacy_job_requests`` derives every job's
    identity from the plan's own content, so two calls with the SAME
    ``plan_ref`` produce identical ``SubmitRequest``s and
    ``submission._insert_or_match`` returns the existing jobs rather than
    inserting new rows (submission.py's idempotency-by-digest rule).
    """
    clock = clock or SystemClock()
    plan_ref = payload.get("plan_ref") if isinstance(payload, dict) else None
    if not isinstance(plan_ref, str) or not plan_ref:
        return 400, {"error": "plan_ref is required and must be a non-empty string"}
    store = ArtifactStore(root)
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    try:
        ref = artifact(conn, store, plan_ref)
        plan = json.loads(store.read_verified(ref))
        if plan.get("kind") != "nightly":
            return 400, {"error": "plan_ref must reference a nightly plan"}
        policy = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})
        result = _submit_nightly(plan, conn, store, policy, clock)
        return 202, result
    except OpsError as exc:
        return _status_for(exc.problem.category), {"problem": to_document(exc.problem)}
    finally:
        conn.close()


def _restored_model_block(model: dict) -> dict:
    """C3 (partial): rehydrate the typed payoff artifact inside a model block.

    ``to_document(NativeScoreInputs)`` flattens a nested
    ``PayoffLineArtifact``/``PayoffSurfaceArtifact`` into a plain JSON dict,
    but the native model stage requires the concrete type
    (``stages._artifact_key_mismatch``), so a valid serialized document would
    otherwise score MODEL_NOT_READY. The dict is restored with
    ``payoff_artifact._artifact_from_document`` (its ``schema_version`` picks
    line vs surface) and the declared ``content_hash`` must be a string equal
    to the restored artifact's recomputed hash. Refusals are ``DocumentError``
    at ``$.model.payoff_artifact`` (or a child path) and never echo input
    values. The caller's block is never mutated: a fresh dict is returned.
    """
    from engine.v2.models.payoff_artifact import (
        PayoffArtifactError,
        _artifact_from_document,
    )

    path = "$.model.payoff_artifact"
    artifact_doc = model["payoff_artifact"]
    if not isinstance(artifact_doc, dict):
        raise DocumentError("BAD_TYPE", path, "expected a payoff artifact object")
    if "content_hash" not in artifact_doc:
        raise DocumentError("MISSING_FIELD", f"{path}.content_hash", "required")
    if not isinstance(artifact_doc["content_hash"], str):
        raise DocumentError("BAD_TYPE", f"{path}.content_hash", "expected a string")
    try:
        artifact = _artifact_from_document(artifact_doc)
    except KeyError as exc:
        field = exc.args[0] if exc.args and isinstance(exc.args[0], str) else ""
        raise DocumentError("MISSING_FIELD",
                            f"{path}.{field}" if field else path, "required") from None
    except PayoffArtifactError:
        raise DocumentError("BAD_SCHEMA_VERSION", f"{path}.schema_version",
                            "not a supported payoff artifact schema_version") from None
    except (TypeError, ValueError, OverflowError):
        raise DocumentError("BAD_TYPE", path, "malformed payoff artifact document") from None
    if artifact_doc["content_hash"] != artifact.content_hash:
        raise DocumentError("CONTENT_HASH_MISMATCH", f"{path}.content_hash",
                            "declared hash does not match the artifact payload")
    block = dict(model)
    block["payoff_artifact"] = artifact
    return block


def _load_native_score_inputs(doc: dict):
    from engine.v2.domain.generation import Geometry, Pricing
    from engine.v2.scoring.stages import NativeScoreInputs, StageReceipt

    geometry = from_document(Geometry, doc["geometry"]) if doc.get("geometry") is not None else None
    pricing = from_document(Pricing, doc["pricing"]) if doc.get("pricing") is not None else None
    receipts = tuple(from_document(StageReceipt, row) for row in doc["stage_receipts"])
    kwargs = dict(
        context=doc["context"], features=doc["features"], forecast=doc["forecast"],
        geometry=geometry, pricing=pricing, analogs=doc["analogs"], simulation=doc["simulation"],
        gate=doc["gate"], chooser=doc["chooser"], diagnostics=doc.get("diagnostics", {}),
        source_ref=doc["source_ref"], stage_receipts=receipts,
    )
    if "model" in doc:
        model = doc["model"]
        if isinstance(model, dict) and model.get("payoff_artifact") is not None:
            model = _restored_model_block(model)
        kwargs["model"] = model
    return NativeScoreInputs(**kwargs)


def rescore_command(args):
    """Read-only ad-hoc rescore: no provider pulls, no fitting.

    ``args.request`` / ``args.native_inputs`` are paths to JSON files
    already produced elsewhere from already-captured data:
    ``args.request`` is ``to_document(a ScoreRequest)``; ``args.native_inputs``
    is ``to_document(a NativeScoreInputs)``. The one ticker/event this scores
    is identified by whatever ``event_id``/``context`` those documents
    already carry -- this command never resolves a ticker/event to data
    itself.
    """
    from engine.v2.contracts import ScoreRequest
    from engine.v2.models.no_fit import no_fit_guard
    from engine.v2.scoring.application import score_one

    request_doc = json.loads(args.request.read_text())
    native_doc = json.loads(args.native_inputs.read_text())
    request = from_document(ScoreRequest, request_doc)
    inputs = _load_native_score_inputs(native_doc)
    with no_fit_guard():
        return score_one(request, inputs)


_PENDING_JOB_STATES = ("queued", "running", "retry_wait")
_STOPPED_JOB_STATES = ("failed", "cancelling", "cancelled", "blocked")


def whatif_action(root: Path, payload, *, clock=None) -> tuple[int, dict]:
    """Submit a supervised ad-hoc rescore job; never scores inline.

    ``payload`` is ``{"request": to_document(ScoreRequest), "native_inputs":
    to_document(NativeScoreInputs)}``. Both documents are published as
    verified artifacts and bound into the job's ``input_bindings`` --
    ``adhoc_rescore`` (engine/v2/ops/worker.py) reads them back from its own
    staging directory; this function never calls score_one itself. The
    idempotency key is the content hash of the payload, so an identical
    repeat POST resolves to the SAME job (submission.py's
    idempotency-by-digest rule) instead of a duplicate.
    """
    from engine.v2.contracts import JobSpec, SubmitRequest
    from engine.v2.ops.checkpoints import register_artifact
    from engine.v2.ops.profiles import profile_named

    clock = clock or SystemClock()
    if not isinstance(payload, dict) or "request" not in payload or "native_inputs" not in payload:
        return 400, {"error": "request and native_inputs are both required"}
    store = ArtifactStore(root)
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    try:
        request_ref = store.publish_bytes(
            json.dumps(payload["request"], sort_keys=True).encode(), schema_ref="score_request.v1.0")
        native_ref = store.publish_bytes(
            json.dumps(payload["native_inputs"], sort_keys=True).encode(),
            schema_ref="native_score_inputs.v1.0")
        with transaction(conn):
            register_artifact(conn, request_ref, None, clock)
            register_artifact(conn, native_ref, None, clock)
        profile = profile_named(DEFAULT_POLICY, "io_fetch")
        repo_root = Path(__file__).resolve().parents[3]
        job = JobSpec(
            kind="adhoc_rescore",
            implementation_ref=content_hash(worker_source_manifest(repo_root)),
            spec_hash=None,
            environment_ref=content_hash(environment_identity(profile.thread_count or profile.cpu_count)),
            parameters={"expected_ids": ["adhoc_rescore"],
                       "input_bindings": {"request.json": request_ref.artifact_id,
                                          "native_inputs.json": native_ref.artifact_id}},
            input_refs=(request_ref.artifact_id, native_ref.artifact_id),
            output_namespace="shadow", resource_class="io_fetch", retry_policy_ref="bounded",
            checkpoint_contract_ref="adhoc_rescore_record.v1.0")
        idempotency_key = content_hash(payload)
        policy = NamespacePolicy({"operator": frozenset({"shadow", "smoke"})})
        receipt = submit(conn, registry(), policy, SubmitRequest(
            namespace="shadow", idempotency_key=idempotency_key, principal="operator", job=job),
            clock=clock)
        return 202, to_document(receipt)
    except OpsError as exc:
        return _status_for(exc.problem.category), {"problem": to_document(exc.problem)}
    finally:
        conn.close()


def whatif_result_action(root: Path, job_id: str, *, clock=None) -> tuple[int, dict]:
    """Fetch a finished ad-hoc rescore job's canonical record, or its state
    while it is not yet finished. Never runs any computation itself."""
    clock = clock or SystemClock()
    store = ArtifactStore(root)
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    try:
        receipt = get_job(conn, job_id)
        if receipt.state in _PENDING_JOB_STATES:
            return 202, {"state": receipt.state}
        if receipt.state in _STOPPED_JOB_STATES:
            return 409, {"state": receipt.state,
                         "failure": to_document(receipt.failure) if receipt.failure else None}
        if not receipt.output_refs:
            return 500, {"error": "job succeeded with no output artifact"}
        ref = artifact(conn, store, receipt.output_refs[0])
        return 200, json.loads(store.read_verified(ref))
    except OpsError as exc:
        return _status_for(exc.problem.category), {"problem": to_document(exc.problem)}
    finally:
        conn.close()


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
    """``ops ledger import-history|status|calibrate|book`` -- see
    ``engine.v2.ops.ledger_history_import`` (import-history) and
    ``engine.v2.ledger.status``/``calibration``/``portfolio`` (P6-3)."""
    if args.ledger_command == "status":
        from engine.v2.ledger.status import status

        return status(conn)
    if args.ledger_command == "calibrate":
        from engine.v2.ledger.calibration import calibrate

        kwargs = {"force": args.force}
        if args.trigger is not None:
            kwargs["trigger"] = args.trigger
        return calibrate(conn, ArtifactStore(root), clock=clock, **kwargs)
    if args.ledger_command == "book":
        from engine.v2.ledger.portfolio import build_book, summarize

        book = build_book(conn, contracts=args.contracts,
                          capital_per_trade=args.capital_per_trade,
                          include_declined=args.include_declined)
        return {"summary": summarize(book),
                "book": json.loads(book.to_json(orient="records")) if not book.empty else []}

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


def price_history_command(args, root, conn, clock):
    """``ops price-history capture`` -- see ``engine.v2.ops.price_history_store``.
    Needs the shared operations catalog (a Tier-2 table, SEND-BACK 2026-09-14),
    so this runs inside ``dispatch()``, not before it.
    """
    from engine.v2.ops.price_history_store import capture

    if not args.source_root.is_dir():
        raise fail("INVALID_REQUEST", "--source-root must be an existing directory",
                  details={"source_root": str(args.source_root)})
    return capture(conn, ArtifactStore(root), args.source_root, scope=args.scope, root=root,
                   dry_run=args.dry_run, clock=clock)


def _provider_account_row(conn, account):
    return conn.execute(
        "SELECT account, generation, remaining, live_reserve, uncertain, "
        "blocked_code, next_eligible_at FROM provider_accounts WHERE account = ?",
        (account,)).fetchone()


def provider_account_command(args, conn):
    """Create or update one provider account's durable budget row (S4B2).

    Absent: created with generation 1. Present: ``remaining`` and
    ``live_reserve`` are replaced and ``generation`` increments, in one
    transaction; ``blocked_code``/``next_eligible_at`` (operator and backoff
    state) are never touched by a budget edit. The resulting row is printed
    as JSON; this table holds no credentials.
    """
    from engine.v2.ops.provider_budget import configure_account

    if min(args.remaining, args.live_reserve) < 0:
        raise fail("INVALID_REQUEST", "negative provider budget")
    if _provider_account_row(conn, args.account) is None:
        configure_account(conn, args.account, 1, args.remaining, args.live_reserve)
    else:
        with transaction(conn):
            conn.execute(
                "UPDATE provider_accounts SET remaining = ?, live_reserve = ?, "
                "generation = CAST(CAST(generation AS INTEGER) + 1 AS TEXT) "
                "WHERE account = ?",
                (args.remaining, args.live_reserve, args.account))
    return dict(_provider_account_row(conn, args.account))


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


def job_command(args, conn, clock, root):
    if args.command == "get":
        return {"job": to_document(get_job(conn, args.job_id)),
                "attempts": to_document(attempt_receipts(conn, args.job_id))}
    if args.command == "cancel":
        return request_cancel(conn, args.job_id, args.expected_attempt, clock=clock)
    if args.command == "resume":
        return resume_command(args, conn)
    if args.command == "explain":
        return explain_command(args, conn, root)
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


#: Bound on the raw ``worker.stderr`` excerpt ``explain`` reads off local
#: disk. This is the one unredacted field in explain's output -- see
#: ``_stderr_tail``'s docstring for why that is safe here and nowhere else.
_STDERR_TAIL_BYTES = 4000


def explain_command(args, conn, root):
    """Everything an attempt did, from its own catalog rows and its private
    staging files, so nobody has to query the catalog, staging files or an
    external monitor by hand (task: "make a v2 ops job explain itself from
    its own logs")."""
    job = get_job(conn, args.job_id)
    store = ArtifactStore(root)
    attempts = [_explain_attempt(conn, store, receipt)
               for receipt in attempt_receipts(conn, args.job_id)]
    return {"job_id": args.job_id, "state": job.state,
            "queue_reason": to_document(job.queue_reason),
            "failure": to_document(job.failure), "attempts": attempts}


def _explain_attempt(conn, store, receipt):
    rows = conn.execute("SELECT body_json FROM progress_events WHERE attempt_id = ? "
                        "ORDER BY sequence", (receipt.attempt_id,)).fetchall()
    events = [json.loads(row[0]) for row in rows]
    reserved = (receipt.resolved_resources.reserved_memory_bytes
               if receipt.resolved_resources else None)
    return {"attempt_id": receipt.attempt_id, "attempt_number": receipt.attempt_number,
            "state": receipt.state, "process_state": receipt.process_state,
            "started_at": receipt.started_at, "ended_at": receipt.ended_at,
            "duration_seconds": _duration(receipt.started_at, receipt.ended_at),
            "exit_code": receipt.exit_code,
            "memory": {"peak_bytes": receipt.memory_peak_bytes, "reserved_bytes": reserved},
            "step_events_recorded": any(e.get("kind") == "progress" for e in events),
            "steps": _step_timeline(events),
            "failure": to_document(receipt.failure),
            "stderr_tail": _stderr_tail(store, receipt.attempt_id)}


def _duration(started_at, ended_at):
    if not started_at or not ended_at:
        return None
    return (parse_timestamp(ended_at) - parse_timestamp(started_at)).total_seconds()


def _step_timeline(events):
    """One ordered entry per completed step -- duration, RSS at its end and
    the memory peak sampled while it was running. A "step started" row (kept
    for ``ops logs --follow``) carries no duration/peak yet, so it is not a
    timeline entry on its own; an attempt with none of either (an older
    format, or a worker kind nothing instruments) yields an empty list,
    rendered as "no step events recorded" rather than an empty table."""
    return [{"step": event["step"], "duration_seconds": event.get("step_duration_seconds"),
            "rss_at_end_bytes": event.get("memory_current_bytes"),
            "peak_bytes": event.get("memory_peak_bytes"), "units": event.get("step_units")}
           for event in events
           if event.get("kind") == "progress" and event.get("message") == "step complete"]


def _stderr_tail(store, attempt_id):
    """The last ``_STDERR_TAIL_BYTES`` of this attempt's private
    ``worker.stderr``, unredacted -- unlike ``failure.details`` (published as
    a catalog artifact, so restricted to a reviewed, secret-free shape;
    ``worker.py``'s ``_classify``/``_generic_problem``), this file never
    leaves local disk and never crosses a namespace boundary: an operator
    running ``ops explain`` already has the same filesystem access to it
    directly. ``None`` when nothing was ever written (no crash, or an
    attempt that predates this field)."""
    path = store.staging_dir(attempt_id) / "diagnostics" / "worker.stderr"
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if not data:
        return None
    tail = data[-_STDERR_TAIL_BYTES:]
    return tail.decode("utf-8", errors="replace")


def logs_command(args, conn):
    """All recorded ``progress_events`` rows; ``--follow`` streams only the
    ones not yet printed, so a step boundary shows up live instead of
    waiting behind a reprinted, ever-growing blob."""
    if not args.follow:
        return _progress_rows(conn, args.job_id)
    printed = 0
    while True:
        rows = _progress_rows(conn, args.job_id)
        for row in rows[printed:]:
            print(json.dumps(row) if args.json else _render_progress_text(row), flush=True)
        printed = len(rows)
        if get_job(conn, args.job_id).state in ("succeeded", "failed", "cancelled", "blocked"):
            return {"complete": True}
        time.sleep(2)


def _progress_rows(conn, job_id):
    rows = conn.execute("SELECT body_json FROM progress_events WHERE job_id = ? "
                        "ORDER BY recorded_at, sequence", (job_id,)).fetchall()
    return [json.loads(row[0]) for row in rows]


def _print_result(args, document):
    """``explain``/``logs`` (non-``--follow``) render as text by default,
    ``--json`` unchanged; every other command prints JSON exactly as before.
    ``--follow`` already streams its own lines as it goes (``logs_command``);
    its final ``{"complete": True}`` still prints as JSON here."""
    if args.command == "explain" and not args.json:
        print(_render_explain_text(document))
    elif args.command == "logs" and not args.follow and not args.json:
        for row in document:
            print(_render_progress_text(row))
    else:
        print(json.dumps(to_document(document), indent=2))


def _human_bytes(value):
    if value is None:
        return "?"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024 or unit == "GiB":
            return f"{amount:.0f}{unit}" if unit == "B" else f"{amount:.1f}{unit}"
        amount /= 1024
    return f"{amount:.1f}GiB"


def _fmt_seconds(value):
    return f"{value:.1f}s" if value is not None else "?"


def _render_explain_text(document):
    lines = [f"job {document['job_id']}: {document['state']}"]
    failure = document.get("failure")
    if failure:
        lines.append(f"  job failure: {failure['code']}: {failure['message']}")
    if not document["attempts"]:
        lines.append("  (no attempts yet)")
    for attempt in document["attempts"]:
        lines.extend(_render_attempt_text(attempt))
    return "\n".join(lines)


def _render_attempt_text(attempt):
    mem = attempt["memory"]
    lines = [f"attempt {attempt['attempt_number']} ({attempt['attempt_id']}): "
            f"{attempt['state']}/{attempt['process_state']}, started {attempt['started_at']}, "
            f"duration {_fmt_seconds(attempt['duration_seconds'])}, exit {attempt['exit_code']}",
            f"  memory: peak {_human_bytes(mem['peak_bytes'])} / "
            f"reserved {_human_bytes(mem['reserved_bytes'])}"]
    lines.extend(_render_steps_text(attempt["steps"]) if attempt["steps"]
                else ["  steps: no step events recorded"])
    lines.extend(_render_failure_text(attempt["failure"], attempt["stderr_tail"]))
    return lines


def _render_steps_text(steps):
    lines = ["  steps:"]
    for step in steps:
        units = f"  units {step['units']}" if step["units"] is not None else ""
        lines.append(f"    {step['step']}: {_fmt_seconds(step['duration_seconds'])}  "
                    f"rss {_human_bytes(step['rss_at_end_bytes'])}  "
                    f"peak {_human_bytes(step['peak_bytes'])}{units}")
    return lines


def _render_failure_text(failure, stderr_tail):
    if not failure:
        return []
    lines = [f"  failure: {failure['code']}: {failure['message']}"]
    if failure.get("details"):
        lines.append(f"    details: {json.dumps(failure['details'], sort_keys=True)}")
    if failure.get("diagnostic_ref"):
        lines.append(f"    diagnostic_ref: {failure['diagnostic_ref']}")
    if stderr_tail:
        lines.append("  stderr tail:")
        lines.extend(f"    {line}" for line in stderr_tail.splitlines()[-20:])
    return lines


def _render_progress_text(event):
    step, kind = event.get("step"), event.get("kind")
    if kind == "progress" and step and event.get("message") == "step complete":
        units = f"  units {event['step_units']}" if event.get("step_units") is not None else ""
        return (f"step {step} done  {_fmt_seconds(event.get('step_duration_seconds'))}  "
               f"rss {_human_bytes(event.get('memory_current_bytes'))}  "
               f"peak {_human_bytes(event.get('memory_peak_bytes'))}{units}")
    if kind == "progress" and step:
        return f"step {step} started"
    bits = [event.get("message", "")]
    if event.get("memory_current_bytes") is not None:
        bits.append(f"mem {_human_bytes(event['memory_current_bytes'])}")
    if event.get("memory_peak_bytes") is not None:
        bits.append(f"peak {_human_bytes(event['memory_peak_bytes'])}")
    if step:
        bits.append(f"at {step}")
    return "  ".join(bits)


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
        elif args.command == "rescore":
            document = rescore_command(args)
        else:
            if args.command == "init":
                ensure_directory(root)
                root.chmod(0o700)
            if not root.is_dir():
                raise fail("INVALID_REQUEST", "operations root must be initialized first")
            conn = open_catalog(root / "catalog.sqlite", clock=clock)
            try:
                _ensure_outcome_sessions_backfilled(conn, root, clock)
                document = dispatch(args, root, conn, clock)
            finally:
                conn.close()
        _print_result(args, document)
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
