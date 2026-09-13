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


def _job_output(stage, keys):
    """The ``job_<id>#<output_name>`` binding for a parent stage's committed output.

    A worker's recorded output is always named for the action it ran (see
    ``dispatch`` in ``worker.py``), so the exact output name is the parent's
    own ``_legacy_action`` name, never the JSON filename it happens to write.
    """
    return keys[stage] + "#" + _legacy_action(stage)


def _legacy_params(action, plan, tickers, year_start, year_end, keys):
    params = {"expected_ids": (action,), "session": plan["session"],
              "tickers": tuple(sorted(tickers)), "year_start": year_start,
              "year_end": year_end, "input_bindings": {}}
    if action == "legacy_decisions":
        params["input_bindings"] = {
            "score.json": _job_output("score", keys), "finality.json": _job_output("finality", keys),
            "decision_plan.json": keys["decision_evidence"] + "#decision_plan",
            "decision_evidence.json": keys["decision_evidence"] + "#decision_evidence"}
    if action == "legacy_render":
        # No "ledger_generation.tar" binding yet: the export stage (§9.4 item 3,
        # not yet wired into this graph) is what produces it. Until it exists,
        # legacy_render refuses at VALIDATION_FAILED("ledger generation not
        # bound") rather than falling back to a staged mutable ledger.
        params["input_bindings"] = {"score.json": _job_output("score", keys),
                                     "model_evidence.json": _job_output("model_evidence", keys),
                                     "finality.json": _job_output("finality", keys)}
    if action == "legacy_selfcheck":
        params["input_bindings"] = {"bundle.tar": _job_output("projection", keys)}
    if action == "legacy_decision_replay":
        params["input_bindings"] = {"score.json": _job_output("score", keys)}
    if action == "decision_evidence":
        # B1c: pinned once by ``ops plan nightly`` (``plans.py::nightly_plan``)
        # and carried unchanged on every retry/resubmission of this same plan.
        params["deployment"] = "shadow:" + plan["implementation_ref"]
        params["decision_clock"] = plan["decision_clock"]
        params["input_bindings"] = {"score.json": _job_output("score", keys),
                                     "finality.json": _job_output("finality", keys),
                                     "replay.json": _job_output("decision_replay", keys)}
    return params


def _legacy_resource(kind):
    if kind in ("legacy_score", "legacy_decision_replay"):
        return "legacy_score"
    if kind == "legacy_model_evidence":
        return "model_evidence"
    if kind == "legacy_settlement":
        return "legacy_rebuild"
    if kind == "legacy_render":
        return "projection"
    return "validation"


def _thread_count(kind):
    """A3: the same formula ``_launch`` uses to verify ``environment_ref``."""
    profile = profile_named(DEFAULT_POLICY, _legacy_resource(kind))
    return profile.thread_count or profile.cpu_count


def build_legacy_job_requests(plan, *, tickers, year_start, year_end,
                              environment_ref=None, include_prerequisites=False,
                              expected_population=(), alt_strikes=1, input_refs=()):
    """Build server-allowlisted JobSpecs for the actual legacy worker DAG."""
    from engine.v2.contracts import JobSpec, SubmitRequest
    from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
    from engine.v2.ops.submission import job_id_for

    provided_environment_ref = environment_ref

    implementation_ref = content_hash(worker_source_manifest(Path(__file__).resolve().parents[3]))
    requests = []
    keys = {}
    scope_hash = content_hash({"tickers": sorted(tickers), "year_start": year_start,
                               "year_end": year_end,
                               "expected_population": list(expected_population)})[:24]
    stages = (tuple(plan["order"]) if include_prerequisites else
              ("finality", "score", "decision_replay", "decision_evidence", "decision_commit",
               "settlement", "model_evidence", "projection", "selfcheck"))
    parent_map = {"finality": (), "score": ("finality",),
                  "decision_replay": ("score",),
                  "decision_evidence": ("score", "finality", "decision_replay"),
                  "decision_commit": ("decision_evidence", "score", "finality"),
                  "settlement": ("finality",),
                  "model_evidence": ("score",),
                  "projection": ("decision_commit", "model_evidence", "finality", "score"),
                  "selfcheck": ("projection",)}
    for stage in stages:
        key = "nightly:" + plan["session"] + ":" + scope_hash + ":" + stage
        keys[stage] = job_id_for("shadow", key)
        # "decision_evidence" is a pure, non-legacy worker (P2-5/B1c): its
        # kind IS the stage name, never run through ``_legacy_action``'s
        # ``legacy_`` prefixing.
        action = "decision_evidence" if stage == "decision_evidence" else _legacy_action(stage)
        kind = action
        parameters = _legacy_params(action, plan, tickers, year_start, year_end, keys)
        if input_refs:
            parameters["input_bindings"]["legacy_manifest.json"] = input_refs[0]
        parameters["expected_population"] = tuple(expected_population)
        parameters["alt_strikes"] = int(alt_strikes)
        parents = (GRAPH[stage] if include_prerequisites else parent_map[stage])
        requests.append(SubmitRequest(
            namespace="shadow", idempotency_key=key, principal="operator",
            job=JobSpec(kind=kind, implementation_ref=implementation_ref,
                        spec_hash=None, environment_ref=(
                            provided_environment_ref or content_hash(
                                environment_identity(_thread_count(kind)))),
                        parameters=parameters,
                        input_refs=tuple(input_refs),
                        dependency_job_ids=tuple(keys[parent] for parent in parents),
                        output_namespace="shadow",
                        resource_class=_legacy_resource(kind),
                        retry_policy_ref="bounded",
                        checkpoint_contract_ref=(
                            "decision_evidence_pair.v1.0" if kind == "decision_evidence"
                            else "legacy_action.v1.0"))))
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
