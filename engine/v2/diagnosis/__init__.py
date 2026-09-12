"""Validation/diagnosis — comparators, tolerance policies, stage plans, receipts.

Layer — of `system_rearchitecture.md` §4.1: a **sink**. It may read every
layer's artifacts and no layer may import it, so a comparator can never become
a dependency of the thing it compares. `checks/import_layers.py` enforces that.

This is the one v2 package phase 0 writes code into
(`guides/rearchitecture_phase0_baseline.md` §5), which means it is subject to
the §4.3 budgets from its first line — zero exemptions, no inherited backlog.

See ``README.md`` for the public interface and the negative control.
"""
from __future__ import annotations

from engine.v2.diagnosis.canonical import canonical_json, content_hash
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
from engine.v2.diagnosis.record_comparator import (
    compare_records,
    flatten,
    merge_receipts,
)
from engine.v2.diagnosis.stage_plan import SCORER_V1, StagePlan, load_stage_plan
from engine.v2.diagnosis.tolerance import (
    EXACT,
    SCORE_RECORD_V1,
    Tolerance,
    TolerancePolicy,
)

__all__ = [
    "AGREE",
    "DIFFER",
    "INCOMPARABLE",
    "ComparisonReceipt",
    "Envelope",
    "Finding",
    "Population",
    "StageHashes",
    "SCORER_V1",
    "SCORE_RECORD_V1",
    "StagePlan",
    "Tolerance",
    "TolerancePolicy",
    "EXACT",
    "canonical_json",
    "compare_records",
    "content_hash",
    "flatten",
    "load_stage_plan",
    "merge_receipts",
    "problem",
]
