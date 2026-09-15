"""Bounded shadow nightly graph around audited legacy helpers.

This module is deliberately separate from ``engine.dashboard.nightly``.  The
legacy entrypoint combines compute and writes, so it cannot be a retryable
worker.  A shadow run accepts stage callables and records explicit degradation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from engine.v2.foundation import content_hash
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import source_closure
from engine.v2.ops.legacy_adapter import copy_read_set
from engine.v2.ops.profiles import DEFAULT_POLICY, profile_named

GRAPH = {
    "refresh": (), "finality": ("refresh",), "features": ("finality",),
    "score": ("features",), "decision_validation": ("score",),
    "decision_commit": ("decision_validation",), "settlement": ("finality",),
    "model_evidence": ("features",), "export": ("decision_commit",),
    "projection": ("export", "model_evidence"), "selfcheck": ("projection",),
    "engineering": (), "publication": ("selfcheck", "engineering"),
    "delivery": ("publication",), "backup": ("decision_commit",),
}
OPTIONAL = frozenset({"settlement", "model_evidence", "engineering", "backup"})


@dataclass(frozen=True)
class StageReceipt:
    stage_id: str
    status: str
    input_hash: str
    output_hash: str | None = None
    error_code: str | None = None
    detail: str | None = None


@dataclass
class NightlyReceipt:
    schema_version: str = "nightly_shadow.v1.0"
    session: str = ""
    status: str = "running"
    read_set: dict = field(default_factory=dict)
    stages: list[StageReceipt] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"schema_version": self.schema_version, "session": self.session,
                "status": self.status, "read_set": self.read_set,
                "stages": [item.__dict__ for item in self.stages]}


def graph_order() -> tuple[str, ...]:
    """Return a deterministic topological order for the bounded graph."""
    remaining = dict(GRAPH)
    result = []
    while remaining:
        ready = sorted(name for name, parents in remaining.items()
                       if all(parent in result for parent in parents))
        if not ready:
            raise fail("INVALID_REQUEST", "nightly graph contains a cycle")
        result.extend(ready)
        for name in ready:
            remaining.pop(name)
    return tuple(result)


def _implementation(root: Path) -> str:
    manifest = source_closure(root, ["engine/v2/ops/nightly.py",
                                     "engine/v2/ops/legacy_adapter.py",
                                     "engine/v2/ops/legacy_actions.py",
                                     "engine/v2/ops/worker.py"])
    return content_hash(manifest)


def build_nightly_plan(source_root: Path | str, session: str, *, mode="shadow",
                       read_set=(), clock=None) -> dict:
    if mode != "shadow":
        raise fail("INVALID_REQUEST", "production nightly activation is disabled")
    from engine.v2.foundation import SystemClock, format_timestamp
    clock = clock or SystemClock()
    return {"schema_version": "nightly_shadow_plan.v1.0", "mode": mode,
            "session": session, "graph": GRAPH, "order": graph_order(),
            "implementation_ref": _implementation(Path(source_root)),
            # B1c: pinned once here, the way ``plans.py::nightly_plan`` pins
            # it for the real CLI path; every stage built off this SAME plan
            # dict (including a retry) carries this one value.
            "decision_clock": format_timestamp(clock.now()),
            "read_set": list(read_set), "effects": ["private_shadow_artifacts"]}


def _legacy_action(stage):
    return "legacy_" + {"finality": "finality", "score": "score",
                         "decision_commit": "decisions", "settlement": "settlement",
                         "model_evidence": "model_evidence", "projection": "render",
                         "selfcheck": "selfcheck"}.get(stage, stage)


#: Stages whose worker kind IS the stage name (never a ``legacy_*`` action):
#: ``decision_evidence`` (P2-5/B1c) and the four effects-graph coordinator
#: stages wired in P2-5/Task5. Each has a trivial pure worker; the real
#: catalog/outbox/filesystem work runs in the supervisor's coordinator effect
#: (``engine.v2.ops.effects_graph``), never in the worker subprocess.
PURE_STAGES = frozenset({"decision_evidence", "ledger_export", "engineering_gate",
                         "publication", "backup"})


def _action_for(stage):
    return stage if stage in PURE_STAGES else _legacy_action(stage)


def _job_output(stage, keys):
    """The ``job_<id>#<output_name>`` binding for a parent stage's committed output.

    A worker's recorded output is always named for the action it ran (see
    ``dispatch`` in ``worker.py``), so the exact output name is the parent's
    own action name (``_action_for``), never the JSON filename it happens to
    write.
    """
    return keys[stage] + "#" + _action_for(stage)


def _generation_pin(plan):
    """The pinned plan identity every generation-aware stage carries (guide
    §5.5 item 1): the code state and decision clock a genuinely new ``ops
    plan nightly`` call always changes, and a retry of the same saved plan
    always reproduces unchanged. ``effects_graph._generation_ref`` folds
    this (plus the stage's own ``legacy_manifest.json`` binding) into a
    per-generation watermark/release identity.
    """
    return {"deployment": "shadow:" + plan["implementation_ref"],
            "decision_clock": plan["decision_clock"]}


def _publication_bindings(keys):
    return {"bundle.tar": _job_output("projection", keys),
            "selfcheck.json": _job_output("selfcheck", keys),
            "engineering_gate.json": _job_output("engineering_gate", keys),
            # P2-C03: the independent anchor for "which session does this
            # release speak for" — never trust the decisions watermark's own
            # occurrence alone (a stale one for the wrong session must still
            # refuse the decision gate; see effects_graph.publication_effect).
            "finality.json": _job_output("finality", keys)}


def _legacy_params(action, plan, tickers, year_start, year_end, keys, *, effect_scope="",
                   context_tickers=(), prior_selfcheck_ref=None):
    # P2-C04: ``context_tickers`` is the historical EVIDENCE universe a
    # scoring/replay action loads (``FeatureContext.load``); ``tickers`` stays
    # the direct watchlist actually scored. Defaults to ``tickers`` so a
    # caller that predates this parameter is unchanged.
    context = tuple(sorted(context_tickers)) if context_tickers else tuple(sorted(tickers))
    params = {"expected_ids": (action,), "session": plan["session"],
              "tickers": tuple(sorted(tickers)), "context_tickers": context,
              "year_start": year_start,
              "year_end": year_end, "input_bindings": {}, "effect_scope": effect_scope}
    if action in ("legacy_score", "legacy_settlement"):
        # P2-C03: score/settlement need the finality-RESOLVED session, never
        # the requested one, so both bind the finality job's own output.
        params["input_bindings"] = {"finality.json": _job_output("finality", keys)}
    if action == "legacy_decisions":
        params["input_bindings"] = {
            "score.json": _job_output("score", keys), "finality.json": _job_output("finality", keys),
            "decision_plan.json": keys["decision_evidence"] + "#decision_plan",
            "decision_evidence.json": keys["decision_evidence"] + "#decision_evidence"}
    if action == "legacy_render":
        # P2-5/Task5: the ledger generation is the verified output of the
        # ``ledger_export`` stage (never a staged mutable ledger copy) —
        # exactly the generation the export coordinator tarred and verified
        # by reading it back through the compatibility reader.
        params["input_bindings"] = {"score.json": _job_output("score", keys),
                                     "model_evidence.json": _job_output("model_evidence", keys),
                                     "finality.json": _job_output("finality", keys),
                                     "ledger_generation.tar": _job_output("ledger_export", keys)}
        if prior_selfcheck_ref:
            # P2-C08: optional -- a previous run's committed selfcheck
            # artifact, bound as a direct ref (no job in this plan produces
            # it: this run's own selfcheck stage runs strictly after render).
            # Absent by default, so render's binding set is unchanged unless
            # a caller opts in.
            params["input_bindings"]["prior_selfcheck.json"] = prior_selfcheck_ref
    if action == "legacy_selfcheck":
        params["input_bindings"] = {"bundle.tar": _job_output("projection", keys)}
    if action == "legacy_decision_replay":
        # P2-C03: replay must re-score at the same resolved session score used.
        params["input_bindings"] = {"score.json": _job_output("score", keys),
                                     "finality.json": _job_output("finality", keys)}
    if action == "decision_evidence":
        # B1c: pinned once by ``ops plan nightly`` (``plans.py::nightly_plan``)
        # and carried unchanged on every retry/resubmission of this same plan.
        params.update(_generation_pin(plan))
        params["input_bindings"] = {
            "score.json": _job_output("score", keys), "finality.json": _job_output("finality", keys),
            "replay.json": _job_output("decision_replay", keys),
            "finality_coverage.json": keys["finality"] + "#legacy_finality_coverage"}
    if action in ("ledger_export", "engineering_gate", "backup"):
        # guide §5.5 item 1: the same pinned plan identity ``decision_evidence``
        # and ``publication`` carry, so ``effects_graph._generation_ref`` can
        # scope these effects' own watermark receipts per generation -- the
        # real 2026-09-14 failure was exactly this field's absence on
        # ``engineering_gate``, colliding with an earlier generation's receipt.
        params.update(_generation_pin(plan))
    if action == "publication":
        # guide §5.5 item 1: the SAME pinned plan identity distinguishes a
        # genuinely new same-session generation's release from a retry of
        # this exact saved plan (see effects_graph.publication_effect).
        params.update(_generation_pin(plan))
        params["input_bindings"] = _publication_bindings(keys)
    return params


def effect_scope_for(tickers, full_universe=None):
    """The outbox/watermark scope for the effects-graph coordinator stages.

    Writing the global ``"shadow"`` scope must be explicit: it is returned
    ONLY when ``full_universe`` is given AND equals ``tickers`` exactly —
    the caller's declaration that this run is a ``--full-run`` scoring its
    whole context. ``full_universe=None`` (no full run declared) always
    returns the subset scope ``"shadow:" + <hash of the ticker subset>``,
    even when ``tickers`` happens to equal whatever context loaded alongside
    it — equality alone is never inferred as a full run (guide §9.4 item 3;
    tightened so a small debugging run can no longer silently advance the
    global watermark).
    """
    if full_universe is not None and sorted(tickers) == sorted(full_universe):
        return "shadow"
    return "shadow:" + content_hash(sorted(tickers)).split(":")[1][:16]


def _legacy_resource(kind):
    if kind in ("legacy_score", "legacy_decision_replay"):
        return "legacy_score"
    if kind == "legacy_model_evidence":
        return "model_evidence"
    if kind == "legacy_settlement":
        return "legacy_rebuild"
    # 2026-09-14 right-sizing: legacy_materialize no longer borrows
    # legacy_rebuild (5.5 GiB, sized for tier rebuilds) — it gets its own
    # small "materialize" profile (2 GiB, profiles.py has the measurement).
    if kind == "legacy_materialize":
        return "materialize"
    if kind == "legacy_render":
        return "projection"
    if kind in ("ledger_export", "engineering_gate", "publication", "backup"):
        return "delivery"
    return "validation"


def _thread_count(kind):
    """A3: the same formula ``_launch`` uses to verify ``environment_ref``."""
    profile = profile_named(DEFAULT_POLICY, _legacy_resource(kind))
    return profile.thread_count or profile.cpu_count


#: The legacy worker DAG's stages when prerequisites are not included.
_DAG_STAGES = ("finality", "score", "decision_replay", "decision_evidence", "decision_commit",
               "settlement", "model_evidence", "ledger_export", "engineering_gate",
               "projection", "selfcheck", "publication", "backup")
# P2-5/Task5: ``ledger_export`` and ``backup`` name ``decision_commit`` as
# their only hard scheduler dependency, never ``settlement`` — a failed
# settlement job would otherwise cascade through ``block_descendants``
# and permanently block both (guide: "settlement failure must not block
# export"). Each coordinator instead reads the settlement *watermark*
# (present or absent) directly, independent of scheduling.
_DAG_PARENTS = {"finality": (), "score": ("finality",),
                "decision_replay": ("score", "finality"),
                "decision_evidence": ("score", "finality", "decision_replay"),
                "decision_commit": ("decision_evidence", "score", "finality"),
                "settlement": ("finality",),
                "model_evidence": ("score",),
                "ledger_export": ("decision_commit",),
                "engineering_gate": (),
                "projection": ("ledger_export", "model_evidence", "finality", "score"),
                "selfcheck": ("projection",),
                "publication": ("selfcheck", "engineering_gate", "projection", "decision_commit",
                                "finality"),
                "backup": ("decision_commit",)}
#: P2-6 §9.3: DAG stages whose kinds read a verified snapshot materialization
#: in snapshot input mode. Every other stage keeps the Phase 1 barrier.
#: ``projection``/``selfcheck`` (attempt-19 fix, 2026-09-15): render and
#: selfcheck must bind the SAME materialization ``legacy_score`` used, never
#: a second independent ``legacy_manifest.json`` live-tree capture -- see
#: ``stages.SNAPSHOT_BACKED_KINDS``.
SNAPSHOT_STAGES = frozenset({"score", "decision_replay", "projection", "selfcheck"})
_SNAPSHOT_REQUIRED = ("snapshot_ref_artifact_id", "materialization_request_ref",
                      "scratch_estimate_bytes")


def _checkpoint_contract(kind):
    if kind == "decision_evidence":
        return "decision_evidence_pair.v1.0"
    if kind in ("ledger_export", "engineering_gate", "publication", "backup"):
        return "effect_receipt.v1.0"
    if kind == "legacy_materialize":
        return "legacy_materialization_manifest.v1.0"
    return "legacy_action.v1.0"


def _snapshot_inputs(input_mode, snapshot_inputs, include_prerequisites):
    if input_mode == "legacy":
        return None
    if input_mode != "snapshot":
        raise fail("INVALID_REQUEST", "input_mode must be legacy or snapshot")
    if include_prerequisites or not isinstance(snapshot_inputs, dict) or any(
            key not in snapshot_inputs for key in _SNAPSHOT_REQUIRED):
        raise fail("INVALID_REQUEST",
                   "snapshot input mode needs pinned snapshot inputs and the legacy worker DAG")
    return snapshot_inputs


def _plan_identity(plan, input_refs):
    """The saved plan's own identity: pinned implementation, decision clock,
    and legacy manifest -- exactly what the Sep-14 operator trace (guide
    §5.5 item 1) found ``_scope_hash`` omitted.

    These three values are written ONCE into the immutable plan document at
    planning time (``plans.py::nightly_plan``/``build_nightly_plan``) and the
    manifest artifact a caller resolves alongside it. Resubmitting the SAME
    saved plan artifact (a retry) reads all three back completely unchanged,
    so this identity -- and therefore every stage key derived from it --
    stays identical. A fresh ``ops plan nightly`` call always re-pins
    ``decision_clock`` from the current clock, so a genuinely new same-
    session plan always changes this identity even when
    tickers/years/population/snapshot happen to match exactly; a code or
    manifest change (the operator's actual Sep-14 scenario) changes it
    independently of the clock.
    """
    return {"implementation_ref": plan.get("implementation_ref") or "",
            "decision_clock": plan.get("decision_clock") or "",
            "legacy_manifest_ref": input_refs[0] if input_refs else ""}


def _scope_hash(tickers, year_start, year_end, expected_population, snapshot, context_tickers=(),
                plan_identity=None):
    scope = {"tickers": sorted(tickers), "context_tickers": sorted(context_tickers),
             "year_start": year_start, "year_end": year_end,
             "expected_population": list(expected_population),
             "plan_identity": plan_identity or {}}
    if snapshot is not None:
        scope.update(input_mode="snapshot", snapshot_ref=snapshot["snapshot_ref_artifact_id"],
                     materialization_request=snapshot["materialization_request_ref"])
    return content_hash(scope)[:24]


def _stage_parameters(stage, plan, tickers, year_start, year_end, keys, effect_scope, snapshot,
                      prior_selfcheck_ref=None, context_tickers=()):
    if stage == "materialize":
        return {"expected_ids": ("legacy_materialize",), "input_bindings": {},
                "scratch_estimate_bytes": int(snapshot["scratch_estimate_bytes"])}
    params = _legacy_params(_action_for(stage), plan, tickers, year_start, year_end, keys,
                            effect_scope=effect_scope, prior_selfcheck_ref=prior_selfcheck_ref,
                            context_tickers=context_tickers)
    if snapshot is not None:
        # P2-C02 review fix: every stage in a snapshot-mode plan graph learns
        # which committed snapshot the plan pinned -- a barrier-only kind
        # cannot bind to it as its own read plan (SNAPSHOT_STAGES), but the
        # generation-binding check still needs to know it is one, and which.
        params["snapshot_generation_id"] = snapshot.get("snapshot_id", "")
        params["snapshot_generation_scope"] = snapshot.get("scope", "")
        # External review #5: the exact import receipt pin_snapshot_inputs
        # resolved at plan time -- see LegacyParameters.snapshot_generation_receipt_id.
        params["snapshot_generation_receipt_id"] = snapshot.get("snapshot_generation_receipt_id", "")
    return params


def _stage_inputs(stage, parameters, keys, input_refs, snapshot):
    """Bind one stage's inputs for the plan's input mode; returns its ``input_refs``."""
    bindings = parameters["input_bindings"]
    if snapshot is None or stage not in SNAPSHOT_STAGES | {"materialize"}:
        if input_refs:
            bindings["legacy_manifest.json"] = input_refs[0]
        return tuple(input_refs)
    bindings["snapshot_ref.json"] = snapshot["snapshot_ref_artifact_id"]
    bindings["materialization_request.json"] = snapshot["materialization_request_ref"]
    if stage != "materialize":
        parameters["input_mode"] = "snapshot"
        bindings["materialization_manifest.json"] = keys["materialize"] + "#materialization_manifest"
    return (snapshot["snapshot_ref_artifact_id"], snapshot["materialization_request_ref"])


def _stage_parents(stage, include_prerequisites, snapshot):
    if include_prerequisites:
        return GRAPH[stage]
    if stage == "materialize":
        return ()
    if snapshot is not None and stage in SNAPSHOT_STAGES:
        return _DAG_PARENTS[stage] + ("materialize",)
    return _DAG_PARENTS[stage]


def _job_spec(kind, parameters, input_refs, dependency_job_ids, implementation_ref,
              environment_ref):
    from engine.v2.contracts import JobSpec
    from engine.v2.ops.fingerprints import environment_identity

    return JobSpec(kind=kind, implementation_ref=implementation_ref, spec_hash=None,
                   environment_ref=(environment_ref or content_hash(
                       environment_identity(_thread_count(kind)))),
                   parameters=parameters, input_refs=tuple(input_refs),
                   dependency_job_ids=dependency_job_ids, output_namespace="shadow",
                   resource_class=_legacy_resource(kind), retry_policy_ref="bounded",
                   checkpoint_contract_ref=_checkpoint_contract(kind))


def build_legacy_job_requests(plan, *, tickers, year_start, year_end,
                              environment_ref=None, include_prerequisites=False,
                              expected_population=(), alt_strikes=1, input_refs=(),
                              full_universe=None, context_tickers=(),
                              input_mode="legacy", snapshot_inputs=None,
                              prior_selfcheck_ref=None):
    """Build server-allowlisted JobSpecs for the actual legacy worker DAG.

    ``input_mode="snapshot"`` (P2-6 §9.3) adds one ``legacy_materialize``
    stage bound to the plan's pinned SnapshotRef and request artifacts, and
    makes ``score``/``decision_replay`` read its verified root instead of the
    barrier. Every other stage, and the default ``"legacy"`` graph, is unchanged.

    ``prior_selfcheck_ref`` (P2-C08): optional artifact ID of a previous
    run's committed selfcheck output. When given, it is bound to the render
    (``projection``) job as ``prior_selfcheck.json`` and admitted into that
    job's ``input_refs`` so it resolves as a direct artifact ref, not a
    ``job_<id>#output`` reference — no job in THIS plan produces it. Omitted
    by default, so the render job's binding set is unchanged unless a caller
    opts in.

    ``context_tickers`` (P2-C04) is the historical evidence universe; it
    defaults to ``tickers`` (today's full-universe plans are unchanged) and
    ``tickers`` (the direct watchlist) must be a subset of it.

    ``full_universe`` (only ever passed by a ``--full-run`` plan) is the
    caller's explicit declaration that this run scores its whole context —
    writing the global ``"shadow"`` effect scope requires it; see
    :func:`effect_scope_for`. Omitted (the default), the effect scope is
    always the subset hash, even when ``tickers`` happens to equal
    ``context_tickers`` — equality is never inferred as a full run. Given, a
    watchlist narrower than the context is refused: a full run must score
    its whole context, never a slice of it.
    """
    from engine.v2.contracts import SubmitRequest
    from engine.v2.ops.fingerprints import worker_source_manifest
    from engine.v2.ops.submission import job_id_for

    snapshot = _snapshot_inputs(input_mode, snapshot_inputs, include_prerequisites)
    implementation_ref = content_hash(worker_source_manifest(Path(__file__).resolve().parents[3]))
    requests = []
    keys = {}
    context_tickers = tuple(context_tickers) or tuple(tickers)
    if not set(tickers) <= set(context_tickers):
        raise fail("INVALID_REQUEST", "watchlist tickers must be a subset of the context tickers")
    if full_universe is not None and sorted(tickers) != sorted(context_tickers):
        raise fail("INVALID_REQUEST", "a full run must score its whole context",
                  details={"tickers": sorted(tickers), "context_tickers": sorted(context_tickers)})
    plan_identity = _plan_identity(plan, input_refs)
    scope_hash = _scope_hash(tickers, year_start, year_end, expected_population, snapshot,
                             context_tickers=context_tickers, plan_identity=plan_identity)
    effect_scope = effect_scope_for(tickers, full_universe)
    stages = tuple(plan["order"]) if include_prerequisites else _DAG_STAGES
    if snapshot is not None:
        stages = ("materialize",) + stages
    for stage in stages:
        key = "nightly:" + plan["session"] + ":" + scope_hash + ":" + stage
        keys[stage] = job_id_for("shadow", key)
        kind = _action_for(stage)
        parameters = _stage_parameters(stage, plan, tickers, year_start, year_end, keys,
                                       effect_scope, snapshot,
                                       prior_selfcheck_ref=prior_selfcheck_ref,
                                       context_tickers=context_tickers)
        refs = _stage_inputs(stage, parameters, keys, input_refs, snapshot)
        if stage == "projection" and prior_selfcheck_ref:
            refs = tuple(refs) + (prior_selfcheck_ref,)
        if stage != "materialize":
            parameters["expected_population"] = tuple(expected_population)
            parameters["alt_strikes"] = int(alt_strikes)
        parents = _stage_parents(stage, include_prerequisites, snapshot)
        requests.append(SubmitRequest(
            namespace="shadow", idempotency_key=key, principal="operator",
            job=_job_spec(kind, parameters, refs, tuple(keys[parent] for parent in parents),
                          implementation_ref, environment_ref)))
    return tuple(requests)


def _run_stage(stage, handler, value, input_hash):
    if handler is None:
        if stage in OPTIONAL:
            return StageReceipt(stage, "degraded", input_hash,
                                error_code="NOT_CONFIGURED",
                                detail="optional branch has no shadow adapter"), value
        raise fail("INVALID_REQUEST", "required nightly stage has no adapter",
                   details={"stage": stage})
    try:
        output = handler(value)
        encoded = json.dumps(output, sort_keys=True, default=str).encode()
        return StageReceipt(stage, "succeeded", input_hash, content_hash(encoded)), output
    except Exception as exc:
        if stage in OPTIONAL:
            return StageReceipt(stage, "degraded", input_hash,
                                error_code=type(exc).__name__,
                                detail="optional stage failed; inspect private diagnostic"), value
        raise fail("VALIDATION_FAILED", "required nightly stage failed",
                   details={"stage": stage, "error": type(exc).__name__}) from exc


def run_shadow_nightly(source_root: Path | str, private_root: Path | str,
                       session: str, *, handlers: dict[str, Callable] | None = None,
                       read_set=(), initial=None, receipt_path=None) -> dict:
    """Execute the real coarse graph using private copies and stage receipts."""
    handlers = handlers or {}
    source = Path(source_root).resolve()
    private = Path(private_root).resolve()
    receipt = NightlyReceipt(session=session)
    receipt.read_set = copy_read_set(source, private / "legacy", tuple(read_set))
    value = initial if initial is not None else {"session": session}
    for stage in graph_order():
        input_hash = content_hash({"stage": stage, "value": value})
        stage_receipt, value = _run_stage(stage, handlers.get(stage), value, input_hash)
        receipt.stages.append(stage_receipt)
    required = [item for item in receipt.stages if item.status == "succeeded"]
    receipt.status = "succeeded" if len(required) == len(receipt.stages) else "degraded"
    document = receipt.as_dict()
    if receipt_path:
        path = Path(receipt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2, sort_keys=True))
    return document


# --------------------------------------------------------------------------
# 2026-09-14: plan-time scratch admission (A1 follow-up). A job whose legacy
# read set exceeds its resource profile's scratch budget must be refused
# HERE, before ``submission.submit_graph`` ever creates it -- not only
# discovered later, after materialize/upstream jobs already ran, when
# ``supervisor.Service._populate_legacy_staging`` (barrier path) or
# ``snapshot_stages._check_scratch`` (``legacy_materialize``) claims it.
# Both of those still run too (defence in depth); this is the plan-time half.
# --------------------------------------------------------------------------


def plan_scratch_problems(conn, store, requests, *, policy=DEFAULT_POLICY, kind_registry=None):
    """Per-job scratch admission problems for an already-built request graph.

    Only a request whose kind actually stages something is checked, mirrored
    exactly from the two claim-time checks this runs ahead of:

    * a kind with a declared ``legacy_store`` read domain
      (``store_barrier.domains_of``, the same test
      ``supervisor.Service._store_domains``/``_pin_read_set`` use) is checked
      against its bound ``legacy_manifest.json`` manifest's ``file_refs`` byte
      total -- exactly what ``_populate_legacy_staging`` sums before it
      copies anything. A ``SNAPSHOT_BACKED_KINDS`` kind running IN snapshot
      mode declares no read domain (``domains_of``: snapshot ``input_mode``
      takes no legacy-store lease) and so is never checked here -- P2-6 §9.3,
      it reads the already-verified materialization root, staging nothing
      new of its own.
    * ``legacy_materialize`` is checked against its own
      ``scratch_estimate_bytes`` parameter -- exactly what
      ``snapshot_stages._check_scratch`` reads.
    * every other kind (``decision_evidence``, the effects-graph coordinator
      kinds) stages nothing and is skipped, even though it may carry the same
      ``legacy_manifest.json`` binding every stage in a legacy-mode plan does.

    Returns a list of problem dicts (empty means every job fits); never
    raises on its own.
    """
    from engine.v2.ops.stages import registry as default_registry
    from engine.v2.ops.store_barrier import domains_of

    reg = kind_registry or default_registry()
    manifest_totals: dict[str, int] = {}
    problems = []
    for request in requests:
        kind = request.job.kind
        parameters = request.job.parameters or {}
        if kind == "legacy_materialize":
            needed = int(parameters.get("scratch_estimate_bytes", 0))
        else:
            reads = any(mode == "read" for _, mode in domains_of(reg, kind, parameters))
            if not reads:
                continue
            manifest_id = (parameters.get("input_bindings") or {}).get("legacy_manifest.json")
            if not manifest_id or str(manifest_id).startswith("job_"):
                continue
            if manifest_id not in manifest_totals:
                manifest = json.loads(store.read_verified(artifact(conn, store, manifest_id)))
                manifest_totals[manifest_id] = sum(
                    int(item["byte_size"]) for item in manifest.get("file_refs", []))
            needed = manifest_totals[manifest_id]
        profile = profile_named(policy, _legacy_resource(kind))
        if needed > profile.scratch_bytes:
            problems.append({"kind": kind, "profile": profile.name,
                             "needed_bytes": needed, "scratch_bytes": profile.scratch_bytes})
    return problems


def refuse_oversize_plan(conn, store, requests, *, policy=DEFAULT_POLICY):
    """Raise before submission if any job in ``requests`` would be refused at
    claim time for exceeding its profile's scratch budget (A1 follow-up)."""
    problems = plan_scratch_problems(conn, store, requests, policy=policy)
    if problems:
        raise fail("RESOURCE_LIMIT_EXCEEDED",
                   "plan stages a legacy read set larger than its profile's scratch budget",
                   details={"jobs": problems})


def plan_memory_problems(requests, *, policy=DEFAULT_POLICY, sample):
    """Per-job STATIC memory admission problems for an already-built request
    graph (§8.1, the legacy_score v5 6 GiB incident's plan-time follow-up).

    Unlike ``plan_scratch_problems`` above, this never reads a manifest or a
    live ``host_available_bytes`` -- it only asks whether ``resource_class``'s
    profile could EVER be admitted under this policy at all
    (``resources.static_ceiling_bytes``, host-total-based). ``sample`` is
    still required (there is no other source for ``host_total_bytes``/
    ``container_limit_bytes``), but its live, fluctuating field
    (``host_available_bytes``) is never read here, so this never trips on
    another process's transient memory use the way a claim-time
    ``MEMORY_HEADROOM`` sample can.

    Returns a list of problem dicts (empty means every named profile could
    ever fit); never raises on its own.
    """
    from engine.v2.ops.resources import static_ceiling_bytes

    ceiling = static_ceiling_bytes(policy, sample)
    problems = []
    seen_classes: set[str] = set()
    for request in requests:
        resource_class = request.job.resource_class
        if resource_class in seen_classes:
            continue
        seen_classes.add(resource_class)
        profile = profile_named(policy, resource_class)
        if profile.memory_bytes > ceiling:
            problems.append({"kind": request.job.kind, "profile": profile.name,
                             "needed_bytes": profile.memory_bytes, "max_possible_bytes": ceiling})
    return problems


def refuse_unfittable_memory_plan(requests, *, policy=DEFAULT_POLICY, sample):
    """Raise before submission if any job in ``requests`` names a resource
    profile that could never be admitted on this host at all, no matter how
    idle it gets (§8.1). Complements the claim-time sustained-window check
    (``scheduler._advance_headroom_ceiling_window``): this one is static and
    immediate, that one is live and patient."""
    problems = plan_memory_problems(requests, policy=policy, sample=sample)
    if problems:
        raise fail("RESOURCE_PROFILE_UNSATISFIABLE",
                   "plan names a resource profile bigger than this host could ever admit",
                   details={"jobs": problems})
