#!/usr/bin/env python3
"""P6-6 session evidence completeness: every Phase 6 capability row expected
to have evidence is covered by at least one receipt under
``reports/phase6_evidence/``.

The row list comes from the same document ``tools/phase6_inventory.py``
builds from ``tools/phase6_capabilities.toml``; the evidence side is every
JSON file found recursively under the evidence root, unioned by their
``capabilities_covered`` lists. A row is exempt when its disposition is
``missing`` or ``dormant-historical``, because those name capabilities with
no v2 producer to measure yet; every other row's id must appear at least
once.

Deliberately tolerant in one direction only: a file that is not valid JSON
is reported under ``unreadable_evidence_files`` and the scan carries on --
one broken receipt must not hide whether the rest of the session is
covered. A row whose declaration has no ``disposition`` key at all is NOT
exempt (fails closed): an incomplete declaration needs evidence, not a pass.
``[[user_decision]]`` entries are not rows and are ignored here.

Exit 0 iff every non-exempt row is covered, else 1.

Usage::

    python3 tools/v2_session_evidence_check.py            # human summary
    python3 tools/v2_session_evidence_check.py --json     # JSON only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.phase6_inventory import build_document  # noqa: E402

__all__ = [
    "EXEMPT_DISPOSITIONS",
    "DEFAULT_EVIDENCE_DIR",
    "covered_capability_ids",
    "check",
    "main",
]

EXEMPT_DISPOSITIONS = frozenset({"missing", "dormant-historical"})
DEFAULT_EVIDENCE_DIR = Path("reports/phase6_evidence")


def covered_capability_ids(evidence_dir: Path) -> tuple[set[str], list[str]]:
    """Union of every ``capabilities_covered`` list under ``evidence_dir``.

    Returns ``(covered_ids, unreadable_files)``; an unparsable file is named
    rather than fatal, and a file that parses to something other than an
    object simply carries no capabilities.
    """
    covered: set[str] = set()
    unreadable: list[str] = []
    if not evidence_dir.is_dir():
        return covered, unreadable
    for path in sorted(evidence_dir.rglob("*.json")):
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            unreadable.append(str(path))
            continue
        ids = document.get("capabilities_covered") if isinstance(document, dict) else None
        if isinstance(ids, list):
            covered.update(item for item in ids if isinstance(item, str))
    return covered, unreadable


def check(*, declarations=None, evidence_dir=DEFAULT_EVIDENCE_DIR, root=ROOT) -> dict:
    root = Path(root)
    evidence_path = Path(evidence_dir)
    if not evidence_path.is_absolute():
        evidence_path = root / evidence_path
    rows = build_document(root=root, declarations=declarations)["rows"]
    covered, unreadable = covered_capability_ids(evidence_path)
    exempt = sum(1 for row in rows if row.get("disposition") in EXEMPT_DISPOSITIONS)
    uncovered = [row["id"] for row in rows
                 if row.get("disposition") not in EXEMPT_DISPOSITIONS
                 and row["id"] not in covered]
    return {
        "rows_total": len(rows),
        "rows_exempt": exempt,
        "rows_covered": len(rows) - exempt - len(uncovered),
        "rows_uncovered": uncovered,
        "unreadable_evidence_files": unreadable,
    }


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="print only the JSON document")
    parser.add_argument("--declarations", type=Path, default=None,
                        help="override tools/phase6_capabilities.toml")
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    result = check(declarations=args.declarations, evidence_dir=args.evidence_dir)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"phase6 session evidence: {result['rows_total']} rows, "
              f"{result['rows_exempt']} exempt, {result['rows_covered']} covered, "
              f"{len(result['rows_uncovered'])} uncovered")
        for row_id in result["rows_uncovered"]:
            print(f"  UNCOVERED {row_id}")
        for path in result["unreadable_evidence_files"]:
            print(f"  UNREADABLE {path}")
    return 0 if not result["rows_uncovered"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
