#!/usr/bin/env python3
"""Review Phase 4 completion separately from its foundation gate.

This check is intentionally read-only.  It does not score rows or bless the
foundation evidence; it maps the observable repository state to the completion
requirements in ``guides/rearchitecture_phase4_scoring.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.phase4_native_audit import audit as native_audit  # noqa: E402
from engine.v2.contracts import ScoreRecord  # noqa: E402

__all__ = ["review", "main"]


@dataclass(frozen=True)
class Blocker:
    blocker_id: str
    assignment: str
    summary: str
    evidence: tuple[str, ...]
    completion_requires: tuple[str, ...]


def _source(root: Path, relative: str) -> str:
    path = root / relative
    return path.read_text() if path.is_file() else ""


def _blocker(blocker_id: str, assignment: str, summary: str,
             evidence: list[str], completion_requires: list[str]) -> Blocker:
    return Blocker(blocker_id, assignment, summary,
                   tuple(evidence), tuple(completion_requires))


def _phase5_blocker(root: Path, evidence: dict[str, Any]) -> Blocker | None:
    models = "\n".join(
        path.read_text() for path in (root / "engine/v2/models").glob("*.py")
    )
    has_interface = "PredictionFrame" in models and "def predict" in models
    declared = evidence.get("phase5_inference_integrated") is True
    if declared and has_interface:
        return None
    facts = [
        f"phase5_inference_integrated={evidence.get('phase5_inference_integrated')!r}",
        f"read_only_prediction_interface={has_interface}",
    ]
    return _blocker(
        "P4-B01", "P4-3 / Phase 5 dependency",
        "Frozen Phase 5 inference is not integrated.", facts,
        ["PredictionFrame plus verified read-only artifact inference",
         "native scoring consumes exact model, fold, residual, and calibration refs"],
    )


def _native_kernel_blocker(root: Path) -> Blocker | None:
    app = _source(root, "engine/v2/scoring/application.py")
    compatibility = _source(root, "engine/v2/scoring/compatibility.py")
    delegates = "score_legacy_request(request, legacy_fields)" in app
    constructs_legacy = delegates and "Scorer()" in compatibility
    if not delegates and not constructs_legacy:
        return None
    return _blocker(
        "P4-B02", "P4-3",
        "The application is a legacy-backed wrapper, not the native staged STR-THRU kernel.",
        [f"application_delegates_to_legacy={delegates}",
         f"compatibility_constructs_legacy_scorer={constructs_legacy}",
         "legacy Scorer._serving fits fold models and its payoff/recalibration paths fit on demand"],
        ["native context, feature, geometry, pricing, inference, analog, gate, and diagnostic stages",
         "request-time fitting and cache building rigged to fail"],
    )


def _contract_blocker() -> Blocker | None:
    actual = {field.name for field in fields(ScoreRecord)}
    required = {
        "payload_hash", "request_hash", "dependency_manifest_ref",
        "readiness", "valid_until", "validation_receipt_refs",
    }
    missing = sorted(required - actual)
    computed_inside_payload = "computed_at" in actual
    if not missing and not computed_inside_payload:
        return None
    return _blocker(
        "P4-B03", "P4-1",
        "The canonical score contract is still a foundation subset.",
        [f"missing_score_record_fields={missing}",
         f"computed_at_inside_score_record={computed_inside_payload}"],
        ["complete payload/request/dependency identities and validation/readiness fields",
         "operational timestamps persisted in a separate envelope"],
    )


def _context_blocker(root: Path, evidence: dict[str, Any]) -> Blocker | None:
    feature_modules = sorted(
        path.name for path in (root / "engine/v2/features").glob("*.py")
        if path.name != "__init__.py"
    )
    controls = evidence.get("completion_controls") or {}
    required = ("watchlist_scope_preserves_analog_population", "cutoff_leak_rejected",
                "changed_missing_mask_rejected")
    missing = [name for name in required if controls.get(name) is not True]
    executable = any(name not in {"recipes.py"} for name in feature_modules)
    if executable and not missing:
        return None
    return _blocker(
        "P4-B04", "P4-2",
        "Feature recipes name context scopes but do not execute or prove them.",
        [f"feature_modules={feature_modules}", f"missing_controls={missing}"],
        ["an executable context planner and FeatureFrame builder",
         "watchlist, cutoff-leak, and changed-null-mask negative controls"],
    )


def _strategy_blocker(evidence: dict[str, Any]) -> Blocker | None:
    controls = evidence.get("completion_controls") or {}
    required = (
        "str_thru_stage_parity", "all_factory_geometry_expiry_fill_parity",
        "irregular_ladder_rejected", "exact_mirrors_preserved",
        "zero_quantity_reference_legs_preserved", "chooser_tie_control",
        "chooser_missing_competitor_control", "chooser_fallback_control",
        "chooser_no_regating_control",
    )
    missing = [name for name in required if controls.get(name) is not True]
    if not missing:
        return None
    return _blocker(
        "P4-B05", "P4-3 / P4-4",
        "Strategy and DYN-SV parity is represented by inventory and synthetic rows only.",
        [f"missing_controls={missing}",
         "Tier-0 replay compares frozen legacy counterparts, not native-stage outputs"],
        ["stage-named STR-THRU comparison on identical real inputs",
         "all factory/refusal and complete chooser edge-case controls"],
    )


def _financial_blocker(root: Path, evidence: dict[str, Any]) -> Blocker | None:
    valuation = _source(root, "engine/v2/domain/valuation/__init__.py")
    controls = evidence.get("completion_controls") or {}
    required = ("full_precision_and_display_parity", "planned_exit_valuation_parity",
                "terminal_payoff_parity", "multi_expiry_refusal")
    missing = [name for name in required if controls.get(name) is not True]
    valuation_empty = "Empty by construction" in valuation
    if not valuation_empty and not missing:
        return None
    return _blocker(
        "P4-B06", "P4-5",
        "Financial ownership covers a few ratios but not the required payoff and display semantics.",
        [f"valuation_package_empty={valuation_empty}", f"missing_controls={missing}",
         "foundation evidence sets terminal_and_planned_exit_labels to a literal true"],
        ["engine-owned fair value, ratios, risk summaries, payoff and simulation outputs",
         "separate full-precision/display parity and terminal/planned/multi-expiry controls"],
    )


def _handoff_blocker(root: Path, evidence: dict[str, Any]) -> Blocker | None:
    contracts = _source(root, "engine/v2/contracts/scoring.py")
    scoring = _source(root, "engine/v2/scoring/__init__.py")
    serving = "\n".join(
        path.read_text() for path in (root / "engine/v2/serving").glob("*.py")
    )
    missing = []
    for name in ("EventScoreRequest", "ScoreBatch", "ReplayReceipt"):
        if f"class {name}" not in contracts:
            missing.append(name)
    exports_replay = (
        "def replay(" in scoring
        or "\"replay\"" in scoring
        or "import replay" in scoring
    )
    if not exports_replay:
        missing.append("replay(score_id)")
    if "ScoreRecord" not in serving:
        missing.append("ScoreRecord legacy projection")
    controls = evidence.get("completion_controls") or {}
    if controls.get("supervised_batch_resource_profile") is not True:
        missing.append("supervised batch resource profile")
    if not missing:
        return None
    return _blocker(
        "P4-B07", "P4-6",
        "The shared application and Phase 6 handoff are incomplete.",
        [f"missing_handoff_surfaces={missing}"],
        ["event/batch/replay contracts with per-request failures and manifests",
         "legacy-format ScoreRecord projection and measured supervised batch profile"],
    )


def _acceptance_blocker(root: Path, evidence: dict[str, Any]) -> Blocker | None:
    subjects = evidence.get("subjects") or {}
    foundation = sorted(
        name for name, row in subjects.items()
        if row.get("status") == "FOUNDATION_PASS"
    )
    controls = evidence.get("completion_controls") or {}
    required = ("native_stage_comparator_planted_defect", "full_saved_release_compared",
                "complete_report_written", "batch_resources_measured")
    missing = [name for name in required if controls.get(name) is not True]
    scope = evidence.get("evidence_scope")
    runner = _source(root, "checks/phase4_real.py")
    pair_count_used_as_population = (
        '"expected": len(corpus.pairs)' in runner
        and '"compared": len(corpus.pairs)' in runner
    )
    if not foundation and not missing and scope == "native_full_release":
        return None
    return _blocker(
        "P4-B08", "Final acceptance",
        "Current evidence is a Tier-0 foundation receipt, not final Phase 4 acceptance.",
        [f"evidence_scope={scope!r}", f"foundation_subjects={foundation}",
         f"missing_controls={missing}",
         f"fixture_pair_count_used_as_population={pair_count_used_as_population}",
         f"population={evidence.get('population')!r}"],
        ["native sequential comparisons over Tier-0 and one complete saved release",
         "planted defect, complete report, dependency inventory, and measured batch resources"],
    )


def _native_audit_blocker(root: Path, evidence: dict[str, Any]) -> Blocker | None:
    result = native_audit(root, evidence)
    if result["ok"]:
        return None
    return _blocker(
        "P4-B09", "Final acceptance",
        "Independent native-scoring audit rejected the acceptance package.",
        list(result["finding_ids"]),
        ["typed native stage execution", "population-complete same-input receipts"],
    )


def review(root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    """Return concrete completion blockers for the supplied foundation evidence."""
    candidates = (
        _phase5_blocker(root, evidence),
        _native_kernel_blocker(root),
        _contract_blocker(),
        _context_blocker(root, evidence),
        _strategy_blocker(evidence),
        _financial_blocker(root, evidence),
        _handoff_blocker(root, evidence),
        _acceptance_blocker(root, evidence),
        _native_audit_blocker(root, evidence),
    )
    blockers = [candidate for candidate in candidates if candidate is not None]
    return {
        "schema_version": "phase4_completion_review.v1.0",
        "status": "COMPLETE" if not blockers else "BLOCKED",
        "complete": not blockers,
        "foundation_status": evidence.get("status"),
        "blocker_count": len(blockers),
        "blockers": [blocker.__dict__ for blocker in blockers],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = review(ROOT, json.loads(args.evidence.read_text()))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{result['status']}: {result['blocker_count']} blocker(s)")
        for blocker in result["blockers"]:
            print(f"{blocker['blocker_id']} {blocker['summary']}")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
