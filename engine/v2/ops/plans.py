"""Immutable, reviewable plans. Planning never authorizes an omitted effect."""
from __future__ import annotations

import json
from pathlib import Path

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore, content_hash
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest

NIGHTLY_GRAPH = {
    "refresh": (), "finality": ("refresh",), "features": ("finality",),
    "score": ("features",), "decision_validation": ("score",),
    "decision_commit": ("decision_validation",), "settlement": ("finality",),
    "model_evidence": ("features",), "export": ("decision_commit",),
    "projection": ("export", "model_evidence"), "selfcheck": ("projection",),
    "engineering": (), "publication": ("selfcheck", "engineering"),
    "delivery": ("publication",), "backup": ("decision_commit",),
}


def _pin_refresh_mode(identity, plan, refresh_mode, refresh_plan, catalog_path=None,
                      objects_root=None):
    """R3B-3: validate and pin ``refresh_mode``/``refresh_plan`` onto
    ``identity``/``plan`` in place. Mirrors the ``input_mode`` block in
    :func:`nightly_plan`: the default ``"legacy"`` touches neither dict, so
    a plan built before this stage existed hashes byte-identically.

    S4A: a native plan also pins the attempt's deployment identity
    (``catalog_path``/``objects_root``) so the refresh job's staged input is
    reproducible from the saved plan alone, never re-resolved from whatever
    catalog happens to be open at submit time.

    S4B2: a native plan may omit ``refresh_plan`` entirely. The plan document
    then records only the mode (and deployment identity) and
    ``build_legacy_job_requests`` builds the production plan from the shadow
    head at submission time; a caller-supplied plan remains a strict override.
    """
    if refresh_mode not in ("legacy", "native"):
        raise fail("INVALID_REQUEST", "refresh_mode must be legacy or native")
    if refresh_mode != "native":
        return
    identity.update(refresh_mode=refresh_mode, catalog_path=catalog_path,
                    objects_root=objects_root)
    plan.update(refresh_mode=refresh_mode, catalog_path=catalog_path,
                objects_root=objects_root)
    if refresh_plan is None:
        return
    from engine.v2.foundation import to_document
    refresh_document = refresh_plan if isinstance(refresh_plan, dict) else to_document(refresh_plan)
    identity.update(refresh_plan_hash=refresh_document.get("plan_hash", ""))
    plan.update(refresh_plan=refresh_document)


def check_plan(source_root, expected_ids):
    manifest = worker_source_manifest(Path(source_root))
    return {"schema_version": "operations_plan.v1.0", "kind": "artifact_check", "mode": "shadow",
            "source_manifest": manifest, "implementation_ref": content_hash(manifest),
            "environment_ref": content_hash(environment_identity()), "effects": ["private_artifacts"],
            "parameters": {"expected_ids": list(expected_ids)}, "resource_class": "delivery",
            "input_refs": [], "blocked_prerequisites": []}


def nightly_plan(source_root, session, *, mode="shadow", manifest_ref=None,
                 tickers=(), context_tickers=(), year_start=2024, year_end=2026,
                 expected_population=(), clock=None, input_mode="legacy", snapshot_inputs=None,
                 full_run=False, refresh_mode="legacy", refresh_plan=None,
                 catalog_path=None, objects_root=None):
    """``full_run`` (``--full-run``, decision: writing the global ``"shadow"``
    effect scope must be explicit) records this plan's universe declaration.
    Without it, ``build_legacy_job_requests`` always uses a subset effect
    scope, even when ``tickers`` equals ``context_tickers``. With it, planning
    refuses unless the watchlist equals the context exactly — a full run
    scores its whole context, never a slice of it.

    ``refresh_mode`` (R3B-3): ``"legacy"`` (default) leaves the DAG exactly
    as it is today -- no ``refresh`` job is ever submitted; the out-of-band
    ``ops price-refresh`` (``legacy_adapter.invoke_price_refresh``) stays the
    only refresh path, unchanged. ``"native"`` submits ONE real
    ``incremental_refresh`` job: an optional caller-resolved
    ``engine.v2.ops.incremental_data.RefreshPlan`` (``refresh_plan``, a
    ``RefreshPlan`` or its ``to_document()`` form) is pinned into this plan
    document as a strict override; when omitted,
    ``nightly.build_legacy_job_requests`` builds the production plan from the
    shadow snapshot head at submission time. The choice is recorded on the
    plan itself (``plan["refresh_mode"]``, present only when ``"native"`` --
    absent means legacy, mirroring ``input_mode`` below), so a reader of a
    saved plan or a completed run's jobs can tell which path executed
    without inferring it from job presence/absence.
    """
    from datetime import date

    from engine.v2.foundation import SystemClock, format_timestamp
    date.fromisoformat(session)
    if mode != "shadow":
        raise fail("INVALID_REQUEST", "production cutover has not been activated")
    # P2-C04: the historical evidence universe defaults to the direct
    # watchlist (today's full-universe plans are unchanged); the watchlist
    # must always be covered by it.
    context_tickers = tuple(context_tickers) or tuple(tickers)
    if not set(tickers) <= set(context_tickers):
        raise fail("INVALID_REQUEST", "watchlist tickers must be a subset of the context tickers")
    if full_run and sorted(tickers) != sorted(context_tickers):
        raise fail("INVALID_REQUEST", "a full run must score its whole context",
                  details={"tickers": sorted(tickers), "context_tickers": sorted(context_tickers)})
    plan = check_plan(source_root, [])
    population = tuple(expected_population)
    blocked = [] if manifest_ref else ["frozen_legacy_input_manifest", "adapter_parity_receipt"]
    if not population:
        blocked.append("planned_population")
    clock = clock or SystemClock()
    identity = {"session": session, "manifest": manifest_ref, "tickers": list(tickers),
                "context_tickers": list(context_tickers), "full_run": bool(full_run),
                "year_start": year_start, "year_end": year_end,
                "expected_population": list(population),
                "implementation": plan["implementation_ref"]}
    if input_mode != "legacy":
        # P2-6 §9.3/§8.1: the head was resolved once by the caller; the plan
        # carries those exact refs, so a new head needs a new plan.
        if input_mode != "snapshot" or not snapshot_inputs:
            raise fail("INVALID_REQUEST", "snapshot input mode needs pinned snapshot inputs")
        identity.update(input_mode=input_mode, snapshot_inputs=snapshot_inputs)
        plan.update(input_mode=input_mode, snapshot_inputs=dict(snapshot_inputs))
    _pin_refresh_mode(identity, plan, refresh_mode, refresh_plan,
                      catalog_path=catalog_path, objects_root=objects_root)
    plan.update(kind="nightly", session=session, graph=NIGHTLY_GRAPH,
                plan_hash=content_hash(identity),
                input_manifest_ref=manifest_ref, tickers=list(tickers),
                context_tickers=list(context_tickers), full_run=bool(full_run),
                year_start=year_start, year_end=year_end,
                expected_population=list(population),
                # P2-5/B1c: pinned once, here, at planning time. The saved
                # plan artifact carries this value forever; every job the DAG
                # later builds off this SAME plan (including a resubmission
                # of the identical plan, i.e. a retry) reads it back rather
                # than invoking the clock again.
                decision_clock=format_timestamp(clock.now()),
                effects=["private_shadow_artifacts", "copy_only_decision_authority"],
                blocked_prerequisites=blocked)
    return plan


def save_plan(conn, root, document, *, clock):
    store = ArtifactStore(root)
    ref = store.publish_bytes(json.dumps(document, sort_keys=True).encode(), schema_ref="operations_plan.v1.0")
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


#: Plan kinds enabled for submission, with the effect scope each must carry.
_ENABLED_PLAN_KINDS = {"artifact_check": ["private_artifacts"], "experiment": ["staged"],
                       "training": ["staged"], "promote": ["staged"]}

#: The checkpoint contract each enabled kind's registered ``JobKind`` declares
#: (``engine.v2.ops.stages.registry()``); a submission whose contract differs
#: from its kind is refused by ``submission.validate_request``.
_PLAN_CHECKPOINT_CONTRACTS = {"artifact_check": "receipt.v1.0",
                              "experiment": "experiment_receipt.v1.0",
                              "training": "training_job_result.v1.0",
                              "promote": "promote_pointer_state.v1.0"}

#: Plan kinds whose registered job kind differs from the plan kind itself.
_PLAN_JOB_KINDS = {"promote": "models_promote"}


def request_from_plan(plan, key):
    if plan.get("schema_version") != "operations_plan.v1.0" or plan.get("blocked_prerequisites"):
        raise fail("INVALID_REQUEST", "plan has unsupported schema or blocked prerequisites")
    if plan["kind"] not in _ENABLED_PLAN_KINDS or plan["effects"] != _ENABLED_PLAN_KINDS[plan["kind"]]:
        raise fail("INVALID_REQUEST", "plan kind is not enabled for submission")
    job = JobSpec(kind=_PLAN_JOB_KINDS.get(plan["kind"], plan["kind"]),
                  implementation_ref=plan["implementation_ref"],
                  spec_hash=plan.get("spec_hash"), environment_ref=plan["environment_ref"],
                  parameters=plan["parameters"], input_refs=tuple(plan["input_refs"]),
                  output_namespace=plan["mode"], resource_class=plan["resource_class"],
                  retry_policy_ref="bounded",
                  checkpoint_contract_ref=_PLAN_CHECKPOINT_CONTRACTS[plan["kind"]])
    return SubmitRequest(namespace=plan["mode"], idempotency_key=key, principal="operator", job=job)
