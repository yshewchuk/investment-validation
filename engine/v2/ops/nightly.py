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


def _legacy_params(action, plan, tickers, year_start, year_end, keys, *, effect_scope=""):
    params = {"expected_ids": (action,), "session": plan["session"],
              "tickers": tuple(sorted(tickers)), "year_start": year_start,
              "year_end": year_end, "input_bindings": {}, "effect_scope": effect_scope}
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
    if action == "legacy_selfcheck":
        params["input_bindings"] = {"bundle.tar": _job_output("projection", keys)}
    if action == "legacy_decision_replay":
        params["input_bindings"] = {"score.json": _job_output("score", keys)}
    if action == "decision_evidence":
        # B1c: pinned once by ``ops plan nightly`` (``plans.py::nightly_plan``)
        # and carried unchanged on every retry/resubmission of this same plan.
        params["deployment"] = "shadow:" + plan["implementation_ref"]
        params["decision_clock"] = plan["decision_clock"]
        params["input_bindings"] = {
            "score.json": _job_output("score", keys), "finality.json": _job_output("finality", keys),
            "replay.json": _job_output("decision_replay", keys),
            "finality_coverage.json": keys["finality"] + "#legacy_finality_coverage"}
    if action == "publication":
        params["input_bindings"] = {
            "bundle.tar": _job_output("projection", keys),
            "selfcheck.json": _job_output("selfcheck", keys),
            "engineering_gate.json": _job_output("engineering_gate", keys)}
    return params


def effect_scope_for(tickers, full_universe=None):
    """The outbox/watermark scope for the effects-graph coordinator stages.

    ``"shadow"`` only when ``tickers`` is exactly the full planned universe;
    otherwise ``"shadow:" + <hash of the ticker subset>``, so a subset shadow
    run can never advance the same watermark row a full run does (guide
    §9.4 item 3). ``full_universe=None`` (no universe declared) is treated as
    "always the full run" — the pre-existing, single-scope behaviour.
    """
    if full_universe is None or sorted(tickers) == sorted(full_universe):
        return "shadow"
    return "shadow:" + content_hash(sorted(tickers)).split(":")[1][:16]


def _legacy_resource(kind):
    if kind in ("legacy_score", "legacy_decision_replay"):
        return "legacy_score"
    if kind == "legacy_model_evidence":
        return "model_evidence"
    if kind in ("legacy_settlement", "legacy_materialize"):
        return "legacy_rebuild"
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
                "decision_replay": ("score",),
                "decision_evidence": ("score", "finality", "decision_replay"),
                "decision_commit": ("decision_evidence", "score", "finality"),
                "settlement": ("finality",),
                "model_evidence": ("score",),
                "ledger_export": ("decision_commit",),
                "engineering_gate": (),
                "projection": ("ledger_export", "model_evidence", "finality", "score"),
                "selfcheck": ("projection",),
                "publication": ("selfcheck", "engineering_gate", "projection", "decision_commit"),
                "backup": ("decision_commit",)}
#: P2-6 §9.3: DAG stages whose kinds read a verified snapshot materialization
#: in snapshot input mode. Every other stage keeps the Phase 1 barrier.
SNAPSHOT_STAGES = frozenset({"score", "decision_replay"})
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


def _scope_hash(tickers, year_start, year_end, expected_population, snapshot):
    scope = {"tickers": sorted(tickers), "year_start": year_start, "year_end": year_end,
             "expected_population": list(expected_population)}
    if snapshot is not None:
        scope.update(input_mode="snapshot", snapshot_ref=snapshot["snapshot_ref_artifact_id"],
                     materialization_request=snapshot["materialization_request_ref"])
    return content_hash(scope)[:24]


def _stage_parameters(stage, plan, tickers, year_start, year_end, keys, effect_scope, snapshot):
    if stage == "materialize":
        return {"expected_ids": ("legacy_materialize",), "input_bindings": {},
                "scratch_estimate_bytes": int(snapshot["scratch_estimate_bytes"])}
    return _legacy_params(_action_for(stage), plan, tickers, year_start, year_end, keys,
                          effect_scope=effect_scope)


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
                              full_universe=None, input_mode="legacy", snapshot_inputs=None):
    """Build server-allowlisted JobSpecs for the actual legacy worker DAG.

    ``input_mode="snapshot"`` (P2-6 §9.3) adds one ``legacy_materialize``
    stage bound to the plan's pinned SnapshotRef and request artifacts, and
    makes ``score``/``decision_replay`` read its verified root instead of the
    barrier. Every other stage, and the default ``"legacy"`` graph, is unchanged.
    """
    from engine.v2.contracts import SubmitRequest
    from engine.v2.ops.fingerprints import worker_source_manifest
    from engine.v2.ops.submission import job_id_for

    snapshot = _snapshot_inputs(input_mode, snapshot_inputs, include_prerequisites)
    implementation_ref = content_hash(worker_source_manifest(Path(__file__).resolve().parents[3]))
    requests = []
    keys = {}
    scope_hash = _scope_hash(tickers, year_start, year_end, expected_population, snapshot)
    effect_scope = effect_scope_for(tickers, full_universe)
    stages = tuple(plan["order"]) if include_prerequisites else _DAG_STAGES
    if snapshot is not None:
        stages = ("materialize",) + stages
    for stage in stages:
        key = "nightly:" + plan["session"] + ":" + scope_hash + ":" + stage
        keys[stage] = job_id_for("shadow", key)
        kind = _action_for(stage)
        parameters = _stage_parameters(stage, plan, tickers, year_start, year_end, keys,
                                       effect_scope, snapshot)
        refs = _stage_inputs(stage, parameters, keys, input_refs, snapshot)
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
