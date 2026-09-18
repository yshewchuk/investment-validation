#!/usr/bin/env python3
"""P5-1: write the current model-release inventory to a JSON file.

Reads the legacy registry (``engine/models/registry.json``) and the real
files it points at under ``data/models`` — nothing is hand-typed, nothing is
loaded as a Scorer, no panel or Tier-3/4 table is touched. Every value in the
output is a name, a role, a count, a byte size, or a content hash; no model
weight, residual, coefficient, price or PnL figure is ever printed or written.

Usage::

    python3 tools/phase5_inventory.py --out reports/phase5_model_inventory.json
    INVESTING_PLAN_ROOT=/root/investing-plan python3 tools/phase5_inventory.py --out /tmp/inv.json

Exit status is always 0 — this is a report, not a gate. Read the printed
coverage table (or ``release_issues`` in the JSON) to see what is missing.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.models import registry as legacy_registry  # noqa: E402
from engine.v2.models import release_issues  # noqa: E402
from engine.v2.models.inventory import (  # noqa: E402
    FEATURE_ROLES,
    current_release_inventory,
    non_model_state_inventory,
    served_roles,
    tier4_fold_coverage,
)


def _jsonable(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def build_document(reg: "legacy_registry.Registry | None" = None) -> dict:
    reg = reg if reg is not None else legacy_registry.load_registry()
    release, drift_issues = current_release_inventory(reg)
    declared_issues = release_issues(release)
    # Folds are named by MODEL id, not role, and more than one id can have
    # served a role historically (a superseded champion's old folds are still
    # on disk). Walk every registry entry once and ask the filesystem, per id.
    fold_coverage = {}
    for entry in reg.entries:
        if entry.role in FEATURE_ROLES:
            fold_coverage[entry.id] = [_jsonable(item) for item in tier4_fold_coverage(entry.id)]
    return {
        "schema_version": "phase5_model_inventory.v1.0",
        "release": _jsonable(release),
        "release_issues": [_jsonable(item) for item in declared_issues],
        "real_file_issues": [_jsonable(item) for item in drift_issues],
        "tier4_fold_coverage": fold_coverage,
        "non_model_state": [_jsonable(item) for item in non_model_state_inventory()],
        "served_roles": list(served_roles()),
        "registry_roles": list(legacy_registry.ROLES),
    }


def _coverage_table(document: dict) -> str:
    """role -> complete/missing, from ``release.bindings`` and its issues.

    Never a raw value: only role/strategy names, member-kind names and issue
    counts.
    """
    by_role_strategy: dict[tuple[str, str], list[str]] = {}
    for binding in document["release"]["bindings"]:
        by_role_strategy[(binding["role"], binding["strategy_id"])] = []
    for issue in document["release_issues"] + document["real_file_issues"]:
        path = issue.get("path", "")
        for key in by_role_strategy:
            role, strategy = key
            if f"[{role}" in path or role in path:
                by_role_strategy[key].append(issue["code"])
    lines = ["role                 strategy        status     issue_codes"]
    for (role, strategy), codes in sorted(by_role_strategy.items()):
        status = "complete" if not codes else "missing"
        lines.append(f"{role:<20} {strategy:<15} {status:<10} {','.join(sorted(set(codes))) or '-'}")
    missing_roles = set(document["registry_roles"]) - {b["role"] for b in document["release"]["bindings"]}
    for role in sorted(missing_roles):
        lines.append(f"{role:<20} {'(none)':<15} {'missing':<10} NO_CHAMPION")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="path to write the inventory JSON to")
    parser.add_argument("--registry", type=Path, default=None,
                        help="alternate registry.json (default: engine/models/registry.json)")
    args = parser.parse_args(argv)

    reg = legacy_registry.load_registry(args.registry) if args.registry else None
    document = build_document(reg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")

    print(f"wrote {args.out} ({args.out.stat().st_size:,} bytes)")
    print(f"artifacts: {len(document['release']['artifacts'])}  "
          f"bindings: {len(document['release']['bindings'])}  "
          f"release_issues: {len(document['release_issues'])}  "
          f"real_file_issues: {len(document['real_file_issues'])}")
    print()
    print(_coverage_table(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
