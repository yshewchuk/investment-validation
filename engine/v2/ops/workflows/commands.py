"""Plan and submit application services for the ops operator commands.

``cli`` keeps argparse and output formatting; ``nightly_trigger`` calls these
services in-process. This module must not import ``cli`` or any runtime module.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from engine.v2.foundation import ArtifactStore, to_document
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import artifact, register_artifact
from engine.v2.ops.discovery import sample_capacity
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.nightly import (
    build_legacy_job_requests,
    refuse_oversize_plan,
    refuse_unfittable_cpu_plan,
    refuse_unfittable_memory_plan,
)
from engine.v2.ops.plans import nightly_plan, request_from_plan, save_plan
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit, submit_graph


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


def _read_input_manifest_ref(args, root, conn, clock, *, nightly=True):
    if not args.input_manifest:
        return None
    if not args.input_manifest.is_file() or args.input_manifest.is_symlink():
        raise fail("INPUT_CHANGED", "input manifest is missing")
    raw_bytes = args.input_manifest.read_bytes()
    if nightly:
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


def _planned_population(args, root, conn, clock, tickers):
    """``(population, snapshot_id, candidate_exclusions)``: a supplied
    ``--expected-population`` file always wins, unchanged. Otherwise ``--input-mode snapshot``
    generates it (:func:`~engine.v2.ops.snapshot_planning.generated_population`) for the ``--tickers``
    watchlist and forwards the snapshot id it read plus the uncarried-ticker exclusions it built;
    any other mode has no snapshot and yields ``()`` and empty exclusions."""
    if args.expected_population or args.input_mode != "snapshot":
        return _read_expected_population(args), None, ()
    if not args.snapshot_scope:
        raise fail("INVALID_REQUEST", "snapshot input mode needs --snapshot-scope")
    from engine.v2.ops.snapshot_planning import generated_population
    return generated_population(conn, ArtifactStore(root), args.snapshot_scope, as_of=args.as_of,
                                tickers=tickers, clock=clock,
                                expected_snapshot_id=getattr(args, "expected_snapshot_id", None))


def _snapshot_inputs(args, root, conn, clock, context_tickers, population, snapshot_id=None):
    """``--input-mode snapshot``: pin the scope's head here, via ``pin_snapshot_inputs``.

    A generated population has already resolved and scanned the head once; pinning resolves it
    again and refuses ``INPUT_CHANGED`` if it moved (``snapshot_id`` is the scanned id).

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
                               session=args.as_of,
                               expected_snapshot_id=snapshot_id
                               or getattr(args, "expected_snapshot_id", None))


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


def _rollback_plan_from_args(args):
    """Build a rollback plan, rejecting the inapplicable ``--release-id`` first.

    Rollback resolves its target from the release store's own history, so a
    named release id is meaningless and refused as ``INVALID_REQUEST`` before
    any plan is saved -- never quietly ignored.  The shared ``--release-id``
    argument defaults to ``""`` but records whether it was supplied, so an
    explicit empty value (``--release-id ""``) is still a supplied value and
    is refused too, while promote's own missing id keeps failing
    ``promote_plan``'s validation.
    """
    from engine.v2.ops.training import rollback_plan
    if getattr(args, "release_id_supplied", False):
        raise fail("INVALID_REQUEST",
                   "rollback takes no --release-id; the target is resolved from "
                   "deployment history")
    return rollback_plan(release_root=args.release_root)


def _plan_command(args, root, conn, clock):
    if args.kind == "nightly":
        tickers = _ticker_list(args.tickers)
        context_tickers = _ticker_list(args.context_tickers) or tickers
        population, snapshot_id, candidate_exclusions = _planned_population(
            args, root, conn, clock, tickers)
        from engine.v2.ops.snapshot_stages import _catalog_path
        plan = nightly_plan(Path(__file__).resolve().parents[4], args.as_of,
                            mode=args.mode, manifest_ref=_read_input_manifest_ref(args, root, conn, clock),
                            tickers=tickers, context_tickers=context_tickers,
                            year_start=args.year_start, year_end=args.year_end,
                            expected_population=population, clock=clock,
                            input_mode=args.input_mode, full_run=args.full_run,
                            candidate_exclusions=candidate_exclusions,
                            snapshot_inputs=_snapshot_inputs(args, root, conn, clock, context_tickers,
                                                             population, snapshot_id),
                            refresh_mode=args.refresh_mode, refresh_plan=_read_refresh_plan(args),
                            catalog_path=_catalog_path(conn), objects_root=str(root))
    elif args.kind == "training":
        from engine.v2.ops.training import training_plan
        plan = training_plan(mode=args.training_mode, recipe=args.recipe or "",
                             state=args.state or "", alpha=args.alpha,
                             cutoffs=tuple(args.cutoff), strategies=tuple(args.strategy),
                             pairs_path=args.pairs or "", ticker_chunk=args.ticker_chunk,
                             manifest_ref=_read_input_manifest_ref(args, root, conn, clock, nightly=False))
    elif args.kind == "promote":
        from engine.v2.ops.training import promote_plan
        expected_previous = getattr(args, "expected_previous_release_id", None)
        if expected_previous is not None and not expected_previous.strip():
            raise fail("INVALID_REQUEST",
                       "--expected-previous-release-id must not be blank; omit it for the "
                       "unguarded behavior")
        plan = promote_plan(release_root=args.release_root, release_id=args.release_id,
                            expected_previous_release_id=expected_previous)
    elif args.kind == "rollback":
        plan = _rollback_plan_from_args(args)
    else:
        from engine.v2.ops.experiments import experiment_plan
        if args.no_ledger and args.activate_ledger:
            raise fail("INVALID_REQUEST",
                       "--no-ledger and --activate-ledger are mutually exclusive")
        if not args.no_ledger and not args.activate_ledger:
            raise fail("INVALID_REQUEST", "production experiment activation is disabled")
        plan = experiment_plan(args.spec, smoke=args.no_ledger)
    ref = save_plan(conn, root, plan, clock=clock)
    return {"plan_ref": ref.artifact_id, "plan": plan}


def _submit_nightly(plan, conn, store, policy, clock, host_policy):
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
    refuse_unfittable_memory_plan(requests, policy=host_policy,
                                  sample=sample_capacity(store.root, clock=clock))
    # issue #103: the CPU twin of the memory check above -- same requests, same typed
    # refusal, before any job row is inserted. Unlike the memory check, this canNOT
    # reuse sample_capacity()'s own allowed_cpu_ids directly: that field is THIS
    # process's own os.sched_getaffinity(0), which a narrower wrapper (a test run under
    # bounded_run --cores 4, a CI sandbox) can restrict well below what an ordinary
    # production host offers -- and, measured directly, doing so makes an ordinary
    # existing test (submitting a real DEFAULT_POLICY nightly plan under a 4-core test
    # wrapper) fail with a false RESOURCE_PROFILE_UNSATISFIABLE. os.cpu_count() is the
    # host-wide logical CPU count and, unlike allowed_cpu_ids, is NOT narrowed by this
    # process's own sched_setaffinity/taskset restriction -- the CPU analogue of the
    # memory check's own host-total (sample.host_total_bytes, also unaffected by this
    # process's own limits) rather than its live, per-process reading.
    capacity = sample_capacity(store.root, clock=clock)
    host_cpus = os.cpu_count() or len(capacity.allowed_cpu_ids)
    refuse_unfittable_cpu_plan(
        requests, policy=host_policy,
        sample=replace(capacity, allowed_cpu_ids=tuple(range(host_cpus))))
    receipts = submit_graph(conn, registry(), policy, requests, clock=clock)
    return {"run_id": "run_" + plan["plan_hash"][:24],
            "jobs": [to_document(item) for item in receipts]}


def _recheck_experiment_preregistration(plan):
    """Re-bind a primary experiment plan's spec at submit.

    The plan records the checkout root its pre-registration check read
    (``preregistration_root``); this recomputes the registered runner's
    legacy ``spec.yaml`` hash with the SAME
    ``experiments.legacy_spec_hash`` the PLANNED row used and refuses with
    ``SPEC_CHANGED`` when it no longer matches -- a spec edited after
    planning never reaches a job.
    """
    from engine.v2.ops.experiments import (
        default_checkout_root,
        experiment_spec_from_document,
        require_preregistration,
    )

    recorded = plan.get("preregistration_root")
    checkout_root = Path(recorded) if recorded else default_checkout_root()
    require_preregistration(checkout_root, experiment_spec_from_document(plan["spec_document"]))


def _registered_runner_manifest(plan):
    """Resolve the primary plan's registered runner and staged source set."""
    from engine.v2.ops.experiments import (
        RUNNER_INVENTORY,
        default_checkout_root,
        runner_manifest,
    )

    runner = plan["spec_document"].get("runner")
    entry = RUNNER_INVENTORY.get(runner) if isinstance(runner, str) else None
    declared = entry.get("declared_runtime_sources") if isinstance(entry, dict) else None
    if (not isinstance(declared, (list, tuple)) or not declared
            or not all(isinstance(item, str) and item for item in declared)):
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "experiment runner registration has no declared runtime sources",
                   details={"runner": runner})
    recorded = plan.get("preregistration_root")
    checkout_root = Path(recorded) if recorded else default_checkout_root()
    try:
        manifest = runner_manifest(checkout_root, runner)
    except OpsError as exc:
        if exc.code == "INPUT_CHANGED":
            raise fail("VALIDATION_FAILED",
                       "registered runner source is missing or indirect",
                       details={"runner": runner}) from exc
        raise fail("INVALID_EXPERIMENT_SPEC", "experiment runner is not registered",
                   details={"runner": runner}) from exc
    return declared, Path(checkout_root).resolve(), manifest


def _primary_runner_bindings(plan, store):
    """Publish the registered primary runner, spec, and runtime source closure."""
    declared, base, manifest = _registered_runner_manifest(plan)
    relative_paths = list(dict.fromkeys(
        [manifest["runner"], manifest["spec_source"], *declared,
         *manifest["source_closure"]]))
    def checkout_path(relative, message):
        candidate = base / relative
        try:
            resolved = candidate.resolve()
            resolved.relative_to(base)
        except (OSError, RuntimeError, ValueError) as exc:
            raise fail("VALIDATION_FAILED", message,
                       details={"path": relative}) from exc
        if not candidate.is_file() or candidate.is_symlink():
            raise fail("VALIDATION_FAILED", message,
                       details={"path": relative})
        return resolved

    bindings = []
    for relative in relative_paths:
        path = checkout_path(relative, "registered runner source is missing")
        bindings.append((relative, store.publish_bytes(
            path.read_bytes(), schema_ref="experiment_runner_source.v1.0")))
    for relative in manifest.get("declared_runtime_inputs", ()):
        path = checkout_path(relative, "registered runner input is missing")
        bindings.append((relative, store.publish_bytes(
            path.read_bytes(), schema_ref="experiment_runner_input.v1.0")))
    spec_path = checkout_path(
        manifest["spec_source"], "registered runner source is missing")
    bindings.append(("spec.yaml", store.publish_bytes(
        spec_path.read_bytes(), schema_ref="experiment_runner_source.v1.0")))
    return bindings


def _submit_command(args, root, conn, clock, host_policy):
    store = ArtifactStore(root)
    ref = artifact(conn, store, args.plan)
    plan = json.loads(store.read_verified(ref))
    policy = NamespacePolicy({"operator": frozenset({"shadow", "smoke", "primary"})})
    if plan.get("kind") == "nightly":
        return _submit_nightly(plan, conn, store, policy, clock, host_policy)
    if plan.get("kind") == "experiment":
        binding_refs = []
        if plan.get("parameters", {}).get("no_ledger", True) is False:
            _recheck_experiment_preregistration(plan)
            binding_refs = _primary_runner_bindings(plan, store)
        spec_ref = store.publish_bytes(json.dumps(plan["spec_document"], sort_keys=True).encode(),
                                       schema_ref="experiment_spec.v1.0")
        with transaction(conn):
            register_artifact(conn, spec_ref, None, clock)
            for _, binding_ref in binding_refs:
                register_artifact(conn, binding_ref, None, clock)
        plan["input_refs"] = [spec_ref.artifact_id,
                              *(binding_ref.artifact_id for _, binding_ref in binding_refs)]
        plan["parameters"]["input_bindings"] = {
            "spec.json": spec_ref.artifact_id,
            **{relative: binding_ref.artifact_id for relative, binding_ref in binding_refs}}
    return submit(conn, registry(), policy, request_from_plan(plan, args.idempotency_key), clock=clock)
