"""One comparator over two ScoreRecord-shaped mappings.

It has exactly the two properties component contracts §15 requires, and the
rest of this module is the machinery that makes them true rather than claimed:

**Stage-localized.** Fields are grouped by the stage plan, each stage gets an
input and an output hash on both sides, and every finding names the earliest
stage — upstream of or at the stage owning its field — whose inputs agreed and
outputs did not. That localization is OBSERVED from record fields along the
declared graph. It says where the records first disagree, not which internal
computation diverged; execution-level stage hashes are rearchitecture phase 1.

**Complete, not first-wins.** Every field is compared in one pass. Stopping at
the first difference is what turned five independent causes into five nights on
2026-09-11.

The compared field set is **derived from the records**, never from a list kept
here. That is the `28cf8b1` fix restated as a structural property: the
explainer that compared 41 of the 70 fields the digest hashed did so because
the 41 were typed out somewhere. The record IS what the digest hashes, so every
leaf that can change the digest is a compared path; remove a field from the
record and it leaves the comparison, add one and it joins it, and neither costs
an edit to this file.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any, Mapping

from engine.v2.diagnosis.canonical import content_hash
from engine.v2.diagnosis.receipt import (
    AGREE,
    DIFFER,
    INCOMPARABLE,
    ComparisonReceipt,
    Envelope,
    Finding,
    Population,
    StageHashes,
    problem,
)
from engine.v2.diagnosis.stage_plan import SCORER_V1, StagePlan
from engine.v2.diagnosis.stage_plan import root_of as _root_of
from engine.v2.diagnosis.tolerance import SCORE_RECORD_V1, Tolerance, TolerancePolicy

__all__ = ["flatten", "root_of", "compare_records", "merge_receipts"]

#: Sentinel for "this path is not in this record at all", which is a different
#: fact from "this path is null here".
_ABSENT = object()


# --------------------------------------------------------------------------
# flattening
# --------------------------------------------------------------------------


def _escape_key(key: str) -> str:
    """Escape a mapping key for a path segment: ``\\`` then ``.``."""
    return key.replace("\\", "\\\\").replace(".", "\\.")


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a record to ``{field_path: leaf}`` over TYPED, injective paths.

    Mapping keys become ``a.b`` with ``.`` and ``\\`` escaped inside each
    segment; sequence positions become ``a[0]``. The two segment shapes are
    what make the mapping unambiguous: a leg LIST flattens to ``legs[0]``
    while a dict ``{0: leg}`` flattens to ``legs.0``, so containers of
    different types can never collide onto one path and compare equal, and a
    mapping key that itself contains a dot (``{"a.b": v}`` → ``a\\.b``) can
    never collide with the nested path ``{"a": {"b": v}}`` → ``a.b``. An
    earlier untyped scheme allowed both collisions, and a comparator that can
    call different records equal is worse than no comparator.

    An empty container is itself a leaf, so ``{}``, ``[]`` and a populated
    container are all distinguishable — the leaf carries the type.
    """
    if isinstance(value, Mapping):
        if not value:
            return {prefix: {}} if prefix else {}
        out: dict[str, Any] = {}
        for key in value:
            segment = _escape_key(str(key))
            path = f"{prefix}.{segment}" if prefix else segment
            out.update(flatten(value[key], path))
        return out
    if isinstance(value, (list, tuple)):
        if not value:
            return {prefix: []} if prefix else {}
        out = {}
        for index, item in enumerate(value):
            out.update(flatten(item, f"{prefix}[{index}]"))
        return out
    return {prefix: value}


def root_of(field_path: str) -> str:
    """Re-export of :func:`stage_plan.root_of` — the typed-path root key."""
    return _root_of(field_path)


# --------------------------------------------------------------------------
# one field
# --------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _kind(left: Any, right: Any) -> str | None:
    """What kind of disagreement, or None when the pair is not yet decided."""
    if left is _ABSENT or right is _ABSENT:
        return "missing_field"
    if (left is None) != (right is None):
        return "null_mask"
    if left is None:
        return None  # both null: agreement, and the mask agrees too
    if _is_int(left) and _is_int(right):
        # Exact, and NEVER through float: float(2**53) == float(2**53 + 1), so
        # a float conversion silently agrees adjacent integers above 2**53.
        return None if left == right else "value"
    if _is_number(left) and _is_number(right):
        # One int, one float — `2` against `2.0`. §7.3 compares integer
        # quantities exactly, and a quantity that changed type changed even
        # where the values coincide. Canonical JSON hashes the two identically,
        # which is precisely why the comparator must not.
        return "type" if _is_int(left) or _is_int(right) else None
    if type(left) is not type(right):
        return "type"
    return None if left == right else "value"


def _finding_id(stage: str, path: str, kind: str) -> str:
    return content_hash([stage, path, kind])[7:19]


def _int_finding(path: str, left: int, right: int, stage: str) -> Finding:
    """A differing integer pair, reported with an exact int delta."""
    delta = right - left  # a float delta would round above 2**53
    return Finding(
        finding_id=_finding_id(stage, path, "value"),
        first_differing_stage=stage,
        field_path=path,
        left_value=left,
        right_value=right,
        delta=delta,
        tolerance_applied="exact (integer)",
        exceeded_by=abs(delta),
        kind="value",
        owning_stage=stage,
    )


def _float_finding(path: str, left: float, right: float, tol: Tolerance,
                   stage: str) -> Finding | None:
    """A float pair under its declared tolerance. NaN matches only NaN.

    Non-finite values never enter the tolerance arithmetic: ``inf - inf`` is
    NaN, and a NaN "excess" is neither inside nor outside any tolerance.
    """
    if left == right or (math.isnan(left) and math.isnan(right)):
        return None
    finite = math.isfinite(left) and math.isfinite(right)
    exceeded = tol.exceeded_by(left, right) if finite else None
    if exceeded == 0.0:
        return None
    return Finding(
        finding_id=_finding_id(stage, path, "value"),
        first_differing_stage=stage,
        field_path=path,
        left_value=left,
        right_value=right,
        delta=right - left if finite else None,
        tolerance_applied=tol.reason,
        exceeded_by=exceeded,
        kind="value",
        owning_stage=stage,
    )


def _compare_field(path: str, left: Any, right: Any, policy: TolerancePolicy,
                   stage: str) -> Finding | None:
    """Compare one field path. Returns a finding, or None when they agree."""
    kind = _kind(left, right)
    if _is_int(left) and _is_int(right):
        return None if kind is None else _int_finding(path, left, right, stage)
    if kind is None and isinstance(left, float) and isinstance(right, float):
        return _float_finding(path, left, right, policy.for_field(path), stage)
    if kind is None:
        return None
    return Finding(
        finding_id=_finding_id(stage, path, kind),
        first_differing_stage=stage,
        field_path=path,
        left_value=None if left is _ABSENT else left,
        right_value=None if right is _ABSENT else right,
        tolerance_applied=policy.for_field(path).reason,
        null_mask_left=left is None or left is _ABSENT,
        null_mask_right=right is None or right is _ABSENT,
        kind=kind,
        owning_stage=stage,
    )


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


def _typed(value: Any) -> list[Any]:
    """A leaf as ``[type, value]``, for the stage hashes.

    Canonical JSON alone hashes an absent field and a null one identically,
    and ``2`` and ``2.0`` identically — each a finding to the comparator. A
    stage row that agrees while a finding in that stage exists is a receipt
    contradicting itself, so the hash carries every distinction the comparison
    makes.
    """
    if value is _ABSENT:
        return ["absent"]
    return [type(value).__name__, value]


def _hash_of(record: dict[str, Any], paths: list[str]) -> str:
    return content_hash({p: _typed(record.get(p, _ABSENT)) for p in paths})


def _stage_hashes(plan: StagePlan, left: dict[str, Any], right: dict[str, Any],
                  paths: list[str]) -> tuple[StageHashes, ...]:
    """Per-stage input/output hashes. A stage's inputs are its predecessors.

    PROVISIONAL by construction, and the receipt's readers must treat it so:
    the input hashes are RECONSTRUCTED from the predecessor stages' output
    FIELDS, not captured from the execution inputs the scorer actually read.
    A stage whose real inputs differed while every predecessor output field
    agrees will be localized as a root. The localization answers "where do the
    records first disagree along the declared graph", not "which computation
    diverged".
    """
    by_stage: dict[str, list[str]] = {sid: [] for sid in plan.stage_ids()}
    for path in paths:
        by_stage[plan.stage_of(path)].append(path)

    rows: list[StageHashes] = []
    for stage_id in plan.stage_ids():
        own = sorted(by_stage[stage_id])
        upstream = sorted(
            path
            for dep in plan.depends_on(stage_id)
            for path in by_stage.get(dep, ())
        )
        left_out, right_out = _hash_of(left, own), _hash_of(right, own)
        rows.append(StageHashes(
            stage_id=stage_id,
            left_input_hash=_hash_of(left, upstream),
            left_output_hash=left_out,
            right_input_hash=_hash_of(right, upstream),
            right_output_hash=right_out,
            agrees=left_out == right_out,
        ))
    return tuple(rows)


def _root_stages(rows: tuple[StageHashes, ...]) -> list[str]:
    """Stages whose inputs agree and outputs do not, in plan order."""
    return [r.stage_id for r in rows if r.inputs_agree and not r.agrees]


def _localized(plan: StagePlan, rows: tuple[StageHashes, ...]) -> dict[str, str]:
    """Owning stage -> the earliest root whose difference can reach it.

    A root reaches a stage only through stages that ALSO differ. A forecast
    that differs while geometry agrees cannot explain a serialization field
    that reads geometry, so the walk follows differing dependencies only, and
    a stage whose own inputs agree is its own origin.
    """
    by_id = {row.stage_id: row for row in rows}
    rank = {row.stage_id: i for i, row in enumerate(rows)}
    memo: dict[str, str] = {}

    def origin(stage_id: str) -> str:
        if stage_id not in memo:
            row = by_id[stage_id]
            upstream = [] if row.inputs_agree else [
                origin(dep) for dep in plan.depends_on(stage_id)
                if dep in by_id and not by_id[dep].agrees
            ]
            memo[stage_id] = min(upstream, key=rank.__getitem__) if upstream else stage_id
        return memo[stage_id]

    return {stage_id: origin(stage_id) for stage_id in by_id}


def _link_localization(findings: list[Finding], rows: tuple[StageHashes, ...],
                       plan: StagePlan) -> tuple[Finding, ...]:
    """Localize every finding, and link the ones in root stages to each other.

    A finding whose owning stage received differing inputs is reported at the
    earliest root upstream of it — a blanked forecast that empties the
    simulation names ``forecast`` on the simulation fields too — and carries no
    link, because it MAY be a consequence of that root. Findings in root
    stages cannot be consequences of one another THROUGH THE DECLARED GRAPH,
    so each names all the others: separate leads to investigate at once.

    This is a localization claim, not a causal-independence proof. Two
    findings in the SAME root stage (``ci_low`` and ``ci_high``, say) are
    linked even though one bootstrap defect produces both: the graph separates
    stages, not causes within a stage.
    """
    where = _localized(plan, rows)
    root = set(_root_stages(rows))
    localized = [
        replace(f, first_differing_stage=where.get(f.owning_stage, f.owning_stage))
        for f in findings
    ]
    root_ids = [f.finding_id for f in localized if f.owning_stage in root]
    out: list[Finding] = []
    for finding in localized:
        if finding.owning_stage in root:
            others = tuple(i for i in root_ids if i != finding.finding_id)
            out.append(replace(finding, not_downstream_of=others))
        else:
            out.append(finding)
    return tuple(out)


# --------------------------------------------------------------------------
# the comparator
# --------------------------------------------------------------------------


def _incomparable(kind: str, tier: int, left_ref: str, right_ref: str,
                  plan: StagePlan, policy: TolerancePolicy,
                  prob: dict[str, Any], population: Population,
                  started: float) -> ComparisonReceipt:
    return ComparisonReceipt(
        receipt_id=content_hash([kind, left_ref, right_ref, prob["code"]])[7:23],
        comparison_kind=kind,
        tier=tier,
        left_ref=left_ref,
        right_ref=right_ref,
        stage_plan_ref=plan.plan_id,
        tolerance_policy_ref=policy.policy_id,
        verdict=INCOMPARABLE,
        population=population,
        problems=(prob,),
        envelope=Envelope.since(started),
    )


def compare_records(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
    *,
    comparison_kind: str = "score_record_parity",
    tier: int = 0,
    left_ref: str = "left",
    right_ref: str = "right",
    stage_plan: StagePlan = SCORER_V1,
    tolerance_policy: TolerancePolicy = SCORE_RECORD_V1,
) -> ComparisonReceipt:
    """Compare two ScoreRecord-shaped mappings and return one receipt.

    Reports every finding in one pass. Returns ``incomparable`` — never
    ``agree`` — when an input is missing or the compared population is empty.
    """
    started = time.monotonic()
    for value, ref in ((left, left_ref), (right, right_ref)):
        if value is None:
            return _incomparable(
                comparison_kind, tier, left_ref, right_ref, stage_plan,
                tolerance_policy,
                problem("MISSING_INPUT", f"{ref} is missing",
                        category="dependency", stage="resolve_context"),
                Population(), started,
            )

    flat_left, flat_right = flatten(dict(left)), flatten(dict(right))
    paths = sorted(set(flat_left) | set(flat_right))
    population = Population(
        expected=len(paths), supported=len(paths), compared=len(paths),
    )
    if not paths:
        return _incomparable(
            comparison_kind, tier, left_ref, right_ref, stage_plan,
            tolerance_policy,
            problem("EMPTY_POPULATION",
                    "both records are empty; an empty comparison is not agreement",
                    category="validation", stage="resolve_context"),
            population, started,
        )

    findings = [
        f for f in (
            _compare_field(p, flat_left.get(p, _ABSENT), flat_right.get(p, _ABSENT),
                           tolerance_policy, stage_plan.stage_of(p))
            for p in paths
        ) if f is not None
    ]
    rows = _stage_hashes(stage_plan, flat_left, flat_right, paths)
    return ComparisonReceipt(
        receipt_id=content_hash([comparison_kind, left_ref, right_ref,
                                 [f.finding_id for f in findings]])[7:23],
        comparison_kind=comparison_kind,
        tier=tier,
        left_ref=left_ref,
        right_ref=right_ref,
        stage_plan_ref=stage_plan.plan_id,
        tolerance_policy_ref=tolerance_policy.policy_id,
        verdict=DIFFER if findings else AGREE,
        stage_hashes=rows,
        findings=_link_localization(findings, rows, stage_plan),
        population=population,
        envelope=Envelope.since(started),
    )


def _merged_population(receipts: list[ComparisonReceipt],
                       expected: int | None) -> Population:
    skipped: dict[str, int] = {}
    for receipt in receipts:
        for prob in receipt.problems:
            skipped[prob["code"]] = skipped.get(prob["code"], 0) + 1
    return Population(
        expected=len(receipts) if expected is None else expected,
        supported=len(receipts),
        compared=sum(1 for r in receipts if r.verdict != INCOMPARABLE),
        skipped_with_reasons=skipped,
    )


def _merged_verdict(findings: tuple[Finding, ...], population: Population,
                    expected: int | None) -> str:
    """A collapsed or short population is incomparable, never agreement."""
    if population.collapsed or (expected is not None and population.compared < expected):
        return INCOMPARABLE
    return DIFFER if findings else AGREE


def merge_receipts(
    receipts: list[ComparisonReceipt],
    *,
    comparison_kind: str,
    tier: int = 0,
    expected: int | None = None,
) -> ComparisonReceipt:
    """Fold per-pair receipts into one corpus-level receipt.

    ``expected`` is the population the caller *asked* for. Passing it is what
    makes a collapsed corpus visible: forty fixtures that silently became zero
    compare zero pairs, and this returns ``incomparable`` rather than ``agree``.
    """
    started = time.monotonic()
    findings = tuple(f for r in receipts for f in r.findings)
    population = _merged_population(receipts, expected)
    verdict = _merged_verdict(findings, population, expected)
    return ComparisonReceipt(
        receipt_id=content_hash([comparison_kind, [r.receipt_id for r in receipts]])[7:23],
        comparison_kind=comparison_kind,
        tier=tier,
        left_ref="corpus",
        right_ref="replay",
        stage_plan_ref=receipts[0].stage_plan_ref if receipts else SCORER_V1.plan_id,
        tolerance_policy_ref=(
            receipts[0].tolerance_policy_ref if receipts else SCORE_RECORD_V1.policy_id
        ),
        verdict=verdict,
        # No stage table: a merged receipt spans many records, and one
        # stage hash per corpus would be a number that means nothing. The
        # per-pair receipts keep theirs, and every finding still names its
        # stage, which is what the localization property asks for.
        stage_hashes=(),
        findings=findings,
        population=population,
        problems=tuple(p for r in receipts for p in r.problems),
        envelope=Envelope.since(started),
    )
