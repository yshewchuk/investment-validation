"""One comparator over two ScoreRecord-shaped mappings.

It has exactly the two properties component contracts §15 requires, and the
rest of this module is the machinery that makes them true rather than claimed:

**Stage-localized.** Fields are grouped by the stage plan, each stage gets an
input and an output hash on both sides, and the first stage whose inputs agreed
and outputs did not is a computed row of the receipt.

**Complete, not first-wins.** Every field is compared in one pass. Stopping at
the first difference is what turned five independent causes into five nights on
2026-09-11.

The compared field set is **derived from the records**, never from a list kept
here. That is the `28cf8b1` fix restated as a structural property: the
explainer that compared 41 of the 70 fields the digest hashed did so because
the 41 were typed out somewhere. Remove a field from the record and it leaves
the comparison; add one and it joins it; neither costs an edit to this file.
"""
from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timezone
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
from engine.v2.diagnosis.tolerance import SCORE_RECORD_V1, TolerancePolicy

__all__ = ["flatten", "compare_records", "merge_receipts"]

#: Sentinel for "this path is not in this record at all", which is a different
#: fact from "this path is null here".
_ABSENT = object()


# --------------------------------------------------------------------------
# flattening
# --------------------------------------------------------------------------


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a record to ``{field_path: leaf}``.

    Nested mappings become ``a.b``; sequences become ``a.0``, so a leg list of
    different lengths reports the exact missing positions rather than one
    opaque "legs differ". An empty container is itself a leaf, so ``{}`` and a
    populated dict are distinguishable.
    """
    if isinstance(value, Mapping):
        if not value:
            return {prefix: {}} if prefix else {}
        out: dict[str, Any] = {}
        for key in value:
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(value[key], path))
        return out
    if isinstance(value, (list, tuple)):
        if not value:
            return {prefix: []} if prefix else {}
        out = {}
        for index, item in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            out.update(flatten(item, path))
        return out
    return {prefix: value}


# --------------------------------------------------------------------------
# one field
# --------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _kind(left: Any, right: Any) -> str | None:
    """What kind of disagreement, or None when the pair is not yet decided."""
    if left is _ABSENT or right is _ABSENT:
        return "missing_field"
    if (left is None) != (right is None):
        return "null_mask"
    if left is None:
        return None  # both null: agreement, and the mask agrees too
    if _is_number(left) and _is_number(right):
        return None  # decided numerically below
    if type(left) is not type(right):
        return "type"
    return None if left == right else "value"


def _finding_id(stage: str, path: str, kind: str) -> str:
    return content_hash([stage, path, kind])[7:19]


def _compare_field(path: str, left: Any, right: Any, policy: TolerancePolicy,
                   stage: str) -> Finding | None:
    """Compare one field path. Returns a finding, or None when they agree."""
    kind = _kind(left, right)
    tol = policy.for_field(path)
    if kind is None and _is_number(left) and _is_number(right):
        exceeded = tol.exceeded_by(float(left), float(right))
        if exceeded == 0.0:
            return None
        return Finding(
            finding_id=_finding_id(stage, path, "value"),
            first_differing_stage=stage,
            field_path=path,
            left_value=left,
            right_value=right,
            delta=float(right) - float(left),
            tolerance_applied=tol.reason,
            exceeded_by=exceeded,
            kind="value",
        )
    if kind is None:
        return None
    return Finding(
        finding_id=_finding_id(stage, path, kind),
        first_differing_stage=stage,
        field_path=path,
        left_value=None if left is _ABSENT else left,
        right_value=None if right is _ABSENT else right,
        tolerance_applied=tol.reason,
        null_mask_left=left is None or left is _ABSENT,
        null_mask_right=right is None or right is _ABSENT,
        kind=kind,
    )


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


def _stage_hashes(plan: StagePlan, left: dict[str, Any], right: dict[str, Any],
                  paths: list[str]) -> tuple[StageHashes, ...]:
    """Per-stage input/output hashes. A stage's inputs are its predecessors."""
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
        left_out = content_hash({p: left.get(p, None) for p in own})
        right_out = content_hash({p: right.get(p, None) for p in own})
        rows.append(StageHashes(
            stage_id=stage_id,
            left_input_hash=content_hash({p: left.get(p, None) for p in upstream}),
            left_output_hash=left_out,
            right_input_hash=content_hash({p: right.get(p, None) for p in upstream}),
            right_output_hash=right_out,
            agrees=left_out == right_out,
        ))
    return tuple(rows)


def _link_independent(findings: list[Finding],
                      rows: tuple[StageHashes, ...]) -> tuple[Finding, ...]:
    """Record which findings were PROVED not to be consequences of each other.

    A finding in a stage whose inputs already differed may be downstream of an
    earlier one, so it gets no link. Findings in stages that received agreeing
    inputs cannot be consequences of one another, so each names all the others:
    an operator can fix that whole set at once, which is the difference between
    five nights and one.
    """
    root = {r.stage_id for r in rows if r.inputs_agree and not r.agrees}
    root_ids = [f.finding_id for f in findings if f.first_differing_stage in root]
    out: list[Finding] = []
    for finding in findings:
        if finding.first_differing_stage in root:
            others = tuple(i for i in root_ids if i != finding.finding_id)
            out.append(replace(finding, independent_of=others))
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
        envelope=_envelope(started),
    )


def _envelope(started: float) -> Envelope:
    return Envelope(
        started_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        duration_seconds=time.monotonic() - started,
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

    Reports every independent finding in one pass. Returns ``incomparable`` —
    never ``agree`` — when an input is missing or the compared population is
    empty.
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
        findings=_link_independent(findings, rows),
        population=population,
        envelope=_envelope(started),
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
        envelope=_envelope(started),
    )
