#!/usr/bin/env python3
"""Read-only audit of Phase 4 native-scoring acceptance claims.

The Phase 4 acceptance producer is deliberately not trusted by this audit.
It checks the canonical application source and requires evidence receipts that
show a full saved-release comparison on real, identical inputs.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

__all__ = ["audit", "main"]

REQUIRED_DIMENSIONS = frozenset({
    "keys", "contracts", "verdicts", "flags", "null_masks",
    "forecasts", "simulation", "financial_diagnostics",
})
REQUIRED_STAGES = frozenset({
    "resolve_context", "features", "forecast", "geometry", "pricing",
    "analogs", "simulation", "gate", "chooser", "serialization",
})
REAL_INPUT_KINDS = frozenset({"saved_release", "sequential_real"})


def _finding(finding_id: str, summary: str, facts: list[str],
             requires: list[str]) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "summary": summary,
        "facts": facts,
        "requires": requires,
    }


def _annotation_is_mapping(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    rendered = ast.unparse(annotation)
    return any(name in rendered for name in ("Mapping", "dict", "Dict"))


def _names(node: ast.AST) -> set[str]:
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}


def _assigned_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
            names.add(item.id)
    return names


def _canonical_mapping_finding(root: Path) -> dict[str, Any] | None:
    relative = Path("engine/v2/scoring/application.py")
    path = root / relative
    if not path.is_file():
        return _finding(
            "P4N-001",
            "Canonical native scoring cannot be inspected.",
            [f"missing_source={relative}"],
            ["a canonical score_one entrypoint with typed native stage inputs"],
        )
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError) as exc:
        return _finding(
            "P4N-001",
            "Canonical native scoring cannot be inspected.",
            [f"source_parse_error={exc}"],
            ["parseable canonical application source"],
        )
    score_one = next(
        (node for node in tree.body
         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
         and node.name == "score_one"),
        None,
    )
    if score_one is None:
        return _finding(
            "P4N-001",
            "Canonical native scoring has no score_one entrypoint.",
            ["score_one_present=false"],
            ["a canonical score_one entrypoint with typed native stage inputs"],
        )

    mapping_parameters = {
        argument.arg for argument in score_one.args.args[1:]
        if _annotation_is_mapping(argument.annotation)
    }
    tainted = set(mapping_parameters)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(score_one):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if value is None or not (_names(value) & tainted):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            additions = set().union(*(_assigned_names(target) for target in targets))
            if not additions.issubset(tainted):
                tainted.update(additions)
                changed = True

    sinks: list[str] = []
    native_values_escape = False
    mapping_passthrough = False
    for node in ast.walk(score_one):
        input_names = {argument.arg for argument in score_one.args.args[1:]}
        if (isinstance(node, ast.Attribute) and node.attr == "values"
                and isinstance(node.value, ast.Name)
                and node.value.id in input_names):
            mapping_passthrough = True
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_legacy_fields"):
            mapping_passthrough = True
        if not isinstance(node, ast.Call):
            continue
        called = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else ""
        )
        if called in {"_record_values", "ScoreRecord"} and (_names(node) & tainted):
            sinks.append(called)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "native_values"):
            native_values_escape = True
    if (mapping_parameters and sinks) or mapping_passthrough:
        return _finding(
            "P4N-001",
            "Canonical scoring accepts arbitrary precomputed mappings.",
            [
                f"mapping_parameters={sorted(mapping_parameters)}",
                f"mapping_record_sinks={sorted(set(sinks))}",
                f"native_values_escape={native_values_escape}",
                f"typed_input_mapping_passthrough={mapping_passthrough}",
            ],
            [
                "typed native stage inputs at the canonical entrypoint",
                "context, feature, forecast, geometry, pricing, analog, gate, and diagnostic execution before record construction",
            ],
        )
    return None


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _receipt(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _saved_release_finding(evidence: dict[str, Any]) -> dict[str, Any] | None:
    comparison = evidence.get("saved_release_comparison")
    if not isinstance(comparison, dict):
        claimed = (evidence.get("completion_controls") or {}).get(
            "full_saved_release_compared"
        )
        return _finding(
            "P4N-002",
            "Saved-release acceptance is not backed by an executed comparison.",
            [
                f"full_saved_release_compared={claimed!r}",
                "saved_release_comparison_present=false",
            ],
            [
                "full expected and compared population counts",
                "distinct native and legacy execution receipts plus a comparison receipt",
                "key, contract, verdict, flag, and null-mask comparison results",
            ],
        )

    population = comparison.get("population") or {}
    expected = population.get("expected")
    compared = population.get("compared")
    dimensions = set(comparison.get("comparison_dimensions") or ())
    native_receipt = comparison.get("native_execution_receipt")
    legacy_receipt = comparison.get("legacy_execution_receipt")
    comparison_receipt = comparison.get("comparison_receipt")
    source = comparison.get("source_release") or {}
    problems = []
    if not (_positive_int(expected) and expected == compared):
        problems.append(f"population_expected={expected!r},compared={compared!r}")
    if comparison.get("complete") is not True:
        problems.append(f"complete={comparison.get('complete')!r}")
    if not REQUIRED_DIMENSIONS.issubset(dimensions):
        problems.append(
            f"missing_dimensions={sorted(REQUIRED_DIMENSIONS - dimensions)}"
        )
    if not (_receipt(native_receipt) and _receipt(legacy_receipt)
            and native_receipt != legacy_receipt):
        problems.append("distinct_execution_receipts=false")
    if not _receipt(comparison_receipt):
        problems.append("comparison_receipt_present=false")
    if not (_receipt(source.get("release_id"))
            and _receipt(source.get("manifest_hash"))):
        problems.append("source_release_identity_complete=false")
    if problems:
        return _finding(
            "P4N-002",
            "Saved-release evidence proves artifact presence, not full execution parity.",
            problems,
            [
                "a population-complete native-versus-legacy saved-release comparison",
                "independent execution and comparison receipts tied to one source release",
            ],
        )
    return None


def _parity_finding(evidence: dict[str, Any]) -> dict[str, Any] | None:
    parity = evidence.get("native_parity")
    if not isinstance(parity, dict):
        return _finding(
            "P4N-003",
            "Native parity evidence has no real-input provenance.",
            ["native_parity_present=false"],
            [
                "same-input native and legacy comparison over a saved release or sequential real corpus",
                "stage-named results and a detected planted defect",
            ],
        )

    provenance = parity.get("input_provenance") or {}
    population = parity.get("population") or {}
    expected = population.get("expected")
    compared = population.get("compared")
    stages = set(parity.get("stages") or ())
    dimensions = set(parity.get("comparison_dimensions") or ())
    planted = parity.get("planted_defect") or {}
    problems = []
    if parity.get("synthetic") is not False:
        problems.append(f"synthetic={parity.get('synthetic')!r}")
    if provenance.get("kind") not in REAL_INPUT_KINDS:
        problems.append(f"input_kind={provenance.get('kind')!r}")
    if not (_receipt(provenance.get("release_id"))
            and _receipt(provenance.get("manifest_hash"))):
        problems.append("input_release_identity_complete=false")
    if parity.get("same_input_hashes") is not True:
        problems.append(f"same_input_hashes={parity.get('same_input_hashes')!r}")
    if not (_positive_int(expected) and expected == compared):
        problems.append(f"population_expected={expected!r},compared={compared!r}")
    if not REQUIRED_STAGES.issubset(stages):
        problems.append(f"missing_stages={sorted(REQUIRED_STAGES - stages)}")
    if not REQUIRED_DIMENSIONS.issubset(dimensions):
        problems.append(
            f"missing_dimensions={sorted(REQUIRED_DIMENSIONS - dimensions)}"
        )
    if planted.get("detected") is not True or not _receipt(planted.get("receipt")):
        problems.append("planted_defect_detection_proven=false")
    if problems:
        return _finding(
            "P4N-003",
            "Parity evidence is synthetic or lacks full same-input stage proof.",
            problems,
            [
                "real saved-release input provenance and complete population counts",
                "all required stages and comparison dimensions",
                "a planted defect that the native comparator detects",
            ],
        )
    return None


def audit(root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    """Return a strict, read-only verdict for Phase 4 native acceptance."""
    findings = [
        finding for finding in (
            _canonical_mapping_finding(root),
            _saved_release_finding(evidence),
            _parity_finding(evidence),
        )
        if finding is not None
    ]
    return {
        "schema_version": "phase4_native_audit.v1.0",
        "status": "PASS" if not findings else "FAIL",
        "ok": not findings,
        "finding_ids": [finding["finding_id"] for finding in findings],
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    evidence = json.loads(Path(args.evidence).read_text())
    result = audit(Path(args.root), evidence)
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result["status"])
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
