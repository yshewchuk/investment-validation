"""Allowlisted stage contracts and strict result validation."""
from __future__ import annotations

from dataclasses import dataclass

from engine.v2.foundation import ArtifactError, safe_relative_path
from engine.v2.ops.errors import fail
from engine.v2.ops.submission import JobKind, KindRegistry, RetryPolicy


@dataclass(frozen=True)
class CheckParameters:
    expected_ids: tuple[str, ...]
    #: Exercises the same launch-time input-binding resolution as the legacy
    #: kinds (P2-5/B1a); the ``artifact_check`` worker never reads a bound
    #: file, so this is only ever used to test resolution and cache identity.
    input_bindings: dict[str, str] | None = None


@dataclass(frozen=True)
class SnapshotImportParameters:
    """P2-7/Task7b: the ``snapshot_import`` worker needs nothing scalar at all
    — its full plan (table sources, contract refs, calendar/source-priority
    versions, expected head) lives in the bound
    ``snapshot_import_request.json``/``legacy_table_mapping.json`` documents,
    resolved through the same ``input_bindings`` machinery every other legacy
    kind already uses (§7.1 point 4: "the worker receives no mutable-current
    alias")."""

    expected_ids: tuple[str, ...]
    input_bindings: dict[str, str] | None = None


@dataclass(frozen=True)
class RebuildCandidateParameters:
    """P2-7/Task7b (§10): a private candidate root, the paths outside it that
    must stay untouched, and the pinned fingerprint of those paths taken
    before submission — ``snapshot_promotion.legacy_rebuild_candidate_effect``
    recomputes the same fingerprint after the attempt and refuses on any
    difference."""

    expected_ids: tuple[str, ...]
    candidate_root: str = ""
    protected_paths: tuple[str, ...] = ()
    protected_before_hash: str = ""
    tables: tuple[str, ...] = ()
    sample: int | None = None


@dataclass(frozen=True)
class LegacyParameters:
    expected_ids: tuple[str, ...]
    session: str = ""
    tickers: tuple[str, ...] = ()
    #: P2-C04: the historical EVIDENCE universe ``_action_score``/
    #: ``_action_decision_replay`` load ``FeatureContext`` with — separate
    #: from ``tickers``, the direct watchlist actually scored. Left ``()``
    #: (meaning: fall back to ``tickers``) by every kind that predates it.
    context_tickers: tuple[str, ...] = ()
    year_start: int = 0
    year_end: int = 0
    horizon_days: int = 35
    sample: int = 10
    force: bool = False
    alt_strikes: int = 1
    expected_population: tuple[str, ...] = ()
    input_bindings: dict[str, str] | None = None
    #: A6: where ``legacy_score_requests`` finds its request batch, relative
    #: to the staging root — under ``legacy/`` once the read set is copied in.
    requests_path: str = ""
    #: P2-5/B1c: the ``decision_evidence`` worker's pinned session identity.
    #: Unused (left "") by every ``legacy_*`` action.
    deployment: str = ""
    decision_clock: str = ""
    #: P2-5/Task5: the outbox/watermark scope the effects-graph coordinator
    #: effects (``ledger_export``, ``engineering_gate``, ``publication``,
    #: ``backup``) use. Left "" (meaning: fall back to ``output_namespace``)
    #: by every other kind.
    effect_scope: str = ""
    #: P2-6 §9.3: ``"legacy"`` (the Phase 1 read-set barrier, the default) or
    #: ``"snapshot"`` (a verified, read-only materialization root). Absent from
    #: every barrier-path job's parameters, so their identity is unchanged.
    input_mode: str = "legacy"
    #: P2-C02 review fix: non-empty only on a job built inside a snapshot-mode
    #: plan graph (``nightly.build_legacy_job_requests(input_mode="snapshot")``).
    #: A barrier-only kind (``legacy_finality``/``legacy_model_evidence``/
    #: ``legacy_selfcheck``) can never itself declare ``input_mode="snapshot"``
    #: (``input_mode_problems`` refuses it — no declared read plan), so this is
    #: the only way it learns which committed snapshot the REST of its own
    #: plan is scored against: the exact ``snapshot_id``/``scope`` the plan's
    #: ``pin_snapshot_inputs`` pinned, never "whatever is newest in scope right
    #: now" — a default legacy-mode nightly leaves both "", and the
    #: generation-binding check is skipped entirely for it.
    snapshot_generation_id: str = ""
    snapshot_generation_scope: str = ""
    #: External review #5 (2026-09-14): the EXACT ``data_import_receipts``
    #: row ``pin_snapshot_inputs`` resolved at plan time for
    #: ``snapshot_generation_id`` -- ``generation_binding.accepted_generation_refs``
    #: reads this receipt by id and never re-resolves "latest for this
    #: snapshot id", so a later reference-only reimport against the same
    #: snapshot id cannot invalidate (or silently rebind) an already-planned
    #: job. Empty on a job planned before this field existed; such a job is
    #: refused ``generation_not_pinned`` rather than falling back to "latest".
    snapshot_generation_receipt_id: str = ""


@dataclass(frozen=True)
class MaterializeParameters:
    """P2-6 §9.3: one ``legacy_materialize`` job writes (or re-verifies) the
    private legacy root for its bound ``materialization_request.json``;
    ``scratch_estimate_bytes`` is the request's pinned byte total."""

    expected_ids: tuple[str, ...]
    input_bindings: dict[str, str] | None = None
    scratch_estimate_bytes: int = 0


#: Kinds whose complete reads ``LEGACY_SCORE_READ_PLAN_V1`` declares
#: (``FeatureContext.load`` + ``Scorer`` + ``score_calendar(alt_strikes=0)``).
SNAPSHOT_BACKED_KINDS = frozenset({"legacy_score", "legacy_score_requests",
                                   "legacy_decision_replay"})
#: Read-only kinds guide §9.3 names that stay on the barrier: no declared read plan.
BARRIER_ONLY_REASONS = {
    "legacy_finality": "reads trading-calendar and finality coverage inputs that "
                       "LEGACY_SCORE_READ_PLAN_V1 does not declare",
    "legacy_model_evidence": "reads model-evidence and training artifacts outside "
                             "LEGACY_SCORE_READ_PLAN_V1",
    "legacy_selfcheck": "re-derives the rendered bundle from stores no read plan declares",
}
SNAPSHOT_BINDINGS = ("snapshot_ref.json", "materialization_request.json",
                     "materialization_manifest.json")


def input_mode_problems(job, params):
    """Kind validator: snapshot mode only on declared kinds, with all three bindings."""
    mode = getattr(params, "input_mode", "legacy")
    if mode not in ("legacy", "snapshot"):
        return ("input_mode must be legacy or snapshot",)
    if mode == "legacy":
        return ()
    if job.kind not in SNAPSHOT_BACKED_KINDS:
        return ("snapshot input mode has no declared read plan for this kind",)
    bindings = params.input_bindings or {}
    if "legacy_manifest.json" in bindings:
        return ("snapshot input mode may not bind a mutable legacy manifest",)
    missing = [name for name in SNAPSHOT_BINDINGS if name not in bindings]
    return ("snapshot input mode is missing bindings: " + ",".join(missing),) if missing else ()


def registry():
    kinds = [
        JobKind(
            name="artifact_check", worker="artifact_check", parameters=CheckParameters,
            resource_classes=frozenset({"delivery"}), effects=("staged",),
            retry=RetryPolicy("bounded", 3, (1, 5)), checkpoint_contract="receipt.v1.0",
            namespaces=frozenset({"shadow", "smoke"})),
        # P2-5/B1c: a pure, non-legacy worker (see worker.py) that derives
        # decision_plan.v1.0/decision_evidence.v1.0 from bound, already-
        # committed score/finality/replay artifacts. Never reads the legacy
        # tree, so it carries no ``store_domains`` read lease.
        JobKind(
            name="decision_evidence", worker="decision_evidence", parameters=LegacyParameters,
            resource_classes=frozenset({"validation"}), effects=("staged",),
            retry=RetryPolicy("bounded", 2, (5, 30)),
            checkpoint_contract="decision_evidence_pair.v1.0",
            namespaces=frozenset({"shadow", "smoke"})),
        # P2-7/Task7b (§7): streams the pinned legacy read set into per-file
        # fragment inspections. Coordinator-validated, like decision_evidence
        # above — see engine.v2.ops.snapshot_promotion.snapshot_import_effect.
        JobKind(
            name="snapshot_import", worker="snapshot_import", parameters=SnapshotImportParameters,
            resource_classes=frozenset({"legacy_rebuild"}), effects=("staged",),
            retry=RetryPolicy("bounded", 2, (5, 30)),
            checkpoint_contract="snapshot_import_inspections.v1.0",
            namespaces=frozenset({"shadow", "smoke"}),
            store_domains=(("legacy_store", "read"),)),
        # P2-7/Task7b (§10): runs the real legacy rebuild rooted at a private
        # candidate directory. No store_domains: it writes only beneath its
        # own candidate root, never the shared legacy_store domain, so it
        # never contends with a real legacy-store writer/reader (§9.2).
        JobKind(
            name="legacy_rebuild_candidate", worker="legacy_rebuild_candidate",
            parameters=RebuildCandidateParameters,
            resource_classes=frozenset({"legacy_rebuild"}), effects=("staged",),
            retry=RetryPolicy("bounded", 1, (30,)),
            checkpoint_contract="legacy_rebuild_candidate.v1.0",
            namespaces=frozenset({"shadow", "smoke"})),
        # P2-6 §9.3: writes/re-verifies one private read-only legacy root per
        # request. No store_domains: it reads only the immutable object store
        # and a read-only catalog connection, never the mutable legacy tree.
        JobKind(
            name="legacy_materialize", worker="legacy_materialize",
            parameters=MaterializeParameters,
            resource_classes=frozenset({"materialize"}), effects=("staged",),
            retry=RetryPolicy("bounded", 2, (5, 30)),
            checkpoint_contract="legacy_materialization_manifest.v1.0",
            namespaces=frozenset({"shadow", "smoke"})),
    ]
    # P2-5/Task5: the export/publication/backup outbox effects, wired into the
    # nightly job DAG (guide §9.4 item 3). Each worker is trivial (it emits a
    # small receipt); the supervisor's coordinator effect
    # (``engine.v2.ops.effects_graph``) does the real catalog/outbox/
    # filesystem work inside the fenced finish path, exactly the pattern
    # ``decision_evidence`` already uses for a pure, non-legacy stage.
    for effect_kind in ("ledger_export", "engineering_gate", "publication", "backup"):
        kinds.append(JobKind(
            name=effect_kind, worker=effect_kind, parameters=LegacyParameters,
            resource_classes=frozenset({"delivery"}), effects=("staged",),
            retry=RetryPolicy("bounded", 3, (5, 30)),
            checkpoint_contract="effect_receipt.v1.0",
            namespaces=frozenset({"shadow", "smoke"})))
    profiles = {"legacy_score": "legacy_score", "legacy_score_requests": "legacy_score",
                "legacy_decision_replay": "legacy_score",
                "legacy_finality": "validation",
                "legacy_decisions": "validation", "legacy_settlement": "legacy_rebuild",
                "legacy_model_evidence": "model_evidence", "legacy_render": "projection",
                "legacy_selfcheck": "validation"}
    for action in ("legacy_finality", "legacy_score", "legacy_decisions",
                   "legacy_settlement", "legacy_model_evidence", "legacy_render",
                   "legacy_selfcheck", "legacy_score_requests", "legacy_decision_replay"):
        kinds.append(JobKind(
            name=action, worker=action, parameters=LegacyParameters,
            resource_classes=frozenset({profiles[action]}),
            effects=("staged",), retry=RetryPolicy("bounded", 2, (5, 30)),
            checkpoint_contract="legacy_action.v1.0",
            namespaces=frozenset({"shadow", "smoke"}),
            validate=input_mode_problems,
            store_domains=(("legacy_store", "read"),)))
    return KindRegistry(kinds)


def validate_result(claim, result):
    if result.get("schema_version") != "worker_result.v1.0":
        raise fail("VALIDATION_FAILED", "missing or unsupported worker result")
    if (result.get("job_id"), result.get("attempt_id"), result.get("fence")) != (
            claim.job_id, claim.attempt_id, claim.fence):
        raise fail("LEASE_LOST", "worker result belongs to another attempt")
    expected = claim.spec.parameters["expected_ids"]
    actual = result.get("completed_ids", [])
    if actual != list(expected) or len(actual) != len(set(actual)):
        raise fail("VALIDATION_FAILED", "worker coverage differs", details={"field": "completed_ids"})
    if not actual and result.get("no_work") is not True:
        raise fail("VALIDATION_FAILED", "no-work receipt missing")
    outputs = result.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise fail("VALIDATION_FAILED", "worker output manifest missing")
    names = set()
    for output in outputs:
        if not isinstance(output, dict) or not all(
                isinstance(output.get(field), str) for field in ("name", "path", "schema")):
            raise fail("VALIDATION_FAILED", "worker output manifest is malformed")
        if output["name"] in names:
            raise fail("VALIDATION_FAILED", "worker output names are duplicated")
        names.add(output["name"])
        try:
            safe_relative_path(output["path"])
        except ArtifactError:
            raise fail("VALIDATION_FAILED", "worker output path escapes staging") from None
    return outputs
