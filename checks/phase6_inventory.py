#!/usr/bin/env python3
"""P6-1 check: regenerate the Phase 6 capability matrix and fail on gaps.

Tier 0, stdlib only, no ``engine.*`` import. It rebuilds the document with
``tools/phase6_inventory.py`` and reports independent findings:

* ``UNOWNED_ENTRYPOINT``   a discovered consumer entrypoint has no row and no exclusion;
* ``NO_DISPOSITION`` / ``BAD_DISPOSITION``  a row lacks a valid disposition;
* ``NO_OWNER`` / ``BAD_OWNER``  a row lacks a valid owning phase;
* ``NEW_ENTRYPOINT_MISSING``  a row claims a v2 entrypoint that does not exist;
* ``MISSING_MISMATCH``  ``MISSING`` in ``new`` iff the disposition is ``missing``;
* ``NO_NEW_ENTRYPOINT``  a non-dormant, non-missing row names no v2 entrypoint;
* ``STALE_OLD_ENTRYPOINT``  a row's old entrypoint is no longer in the source;
* ``TEST_MISSING``  a listed test file/symbol does not exist;
* ``DORMANT_ACTIVE``  a dormant-historical row is marked active (or vice versa);
* ``INTERACTIVE_CLI_UNACCEPTED``  an interactive action replaced by a CLI
  without the CLI-with-user-acceptance-needed disposition and a decision;
* ``REQUIRED_AREA_ABSENT``  a required inventory entry has no row;
* ``DUPLICATE_ROW`` / ``UNKNOWN_AREA`` / ``STALE_EXCLUSION`` / ``UNKNOWN_DECISION``;
* ``ADAPTER_EDGE_UNOWNED`` / ``ADAPTER_MODULE_UNDECLARED`` / ``ADAPTER_MODULE_MISSING``;
* ``GUIDE_STALE``  the guide's generated section differs from a fresh render.

Usage::

    python3 checks/phase6_inventory.py          # findings, exit 1 on any
    python3 checks/phase6_inventory.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import phase6_inventory as inv  # noqa: E402

_OLD_PREFIXES = ("py:", "doc:")


def _finding(code: str, subject: str, detail: str) -> dict:
    return {"code": code, "subject": subject, "detail": detail}


def _covered(document: dict) -> set[str]:
    covered = set()
    for row in document["rows"]:
        covered |= set(row["old"]) | set(row["new"])
    for excluded in document["excluded"]:
        covered |= set(excluded["ids"])
    return covered


def entrypoint_findings(document: dict) -> list[dict]:
    covered = _covered(document)
    findings = [_finding("UNOWNED_ENTRYPOINT", ident, f"{meta['kind']} in {meta['source']} has no row")
                for ident, meta in document["discovered"].items() if ident not in covered]
    for excluded in document["excluded"]:
        findings += [_finding("STALE_EXCLUSION", ident, "excluded id is not discovered")
                     for ident in excluded["ids"] if ident not in document["discovered"]]
    return findings


def _old_resolves(root: Path, ref: str, discovered: dict) -> bool:
    if ref.startswith("doc:"):
        return (root / ref.removeprefix("doc:")).is_file()
    return inv.resolve_entrypoint(root, ref, discovered)


def _disposition_findings(row: dict) -> list[dict]:
    rid, disposition = row["id"], row["disposition"]
    if not disposition:
        return [_finding("NO_DISPOSITION", rid, "row has no disposition")]
    if disposition not in inv.DISPOSITIONS:
        return [_finding("BAD_DISPOSITION", rid, f"{disposition!r} not in {inv.DISPOSITIONS}")]
    findings = []
    if ("MISSING" in row["new"]) != (disposition == "missing"):
        findings.append(_finding("MISSING_MISMATCH", rid, "MISSING in new iff disposition is missing"))
    real_new = [ref for ref in row["new"] if ref != "MISSING"]
    if disposition in inv.RESOLVING and not real_new:
        findings.append(_finding("NO_NEW_ENTRYPOINT", rid, f"{disposition} row names no v2 entrypoint"))
    if (disposition == "dormant-historical") == row["active"]:
        findings.append(_finding("DORMANT_ACTIVE", rid, "dormant-historical rows, and only they, are inactive"))
    return findings


def _interactive_findings(row: dict, decisions: set[str]) -> list[dict]:
    if not row["interactive"]:
        return []
    real_new = [ref for ref in row["new"] if ref != "MISSING"]
    only_cli = bool(real_new) and all(ref.startswith("cli:") for ref in real_new)
    if only_cli and (row["disposition"] != "CLI-with-user-acceptance-needed" or row["decision"] not in decisions):
        return [_finding("INTERACTIVE_CLI_UNACCEPTED", row["id"],
                         "an interactive action replaced by a CLI needs the acceptance disposition and a decision")]
    return []


def row_findings(document: dict, root: Path = inv.ROOT) -> list[dict]:
    findings, seen = [], set()
    decisions = {d["id"] for d in document["user_decisions"]}
    for row in document["rows"]:
        rid = row["id"]
        if rid in seen:
            findings.append(_finding("DUPLICATE_ROW", rid, "row id repeated"))
        seen.add(rid)
        if row["area"] not in inv.REQUIRED_AREAS:
            findings.append(_finding("UNKNOWN_AREA", rid, f"area {row['area']!r}"))
        findings += _disposition_findings(row)
        if not row["owner"]:
            findings.append(_finding("NO_OWNER", rid, "row has no owning phase"))
        elif row["owner"] not in inv.ROW_OWNERS:
            findings.append(_finding("BAD_OWNER", rid, f"{row['owner']!r} not in {inv.ROW_OWNERS}"))
        findings += [_finding("NEW_ENTRYPOINT_MISSING", rid, ref)
                     for ref, ok in row["new_resolved"].items() if not ok]
        findings += [_finding("STALE_OLD_ENTRYPOINT", rid, ref)
                     for ref in row["old"] if not _old_resolves(root, ref, document["discovered"])]
        findings += [_finding("TEST_MISSING", rid, test) for test, ok in row["tests_resolved"].items() if not ok]
        findings += _interactive_findings(row, decisions)
        if row["decision"] and row["decision"] not in decisions:
            findings.append(_finding("UNKNOWN_DECISION", rid, row["decision"]))
    present = {row["area"] for row in document["rows"]}
    findings += [_finding("REQUIRED_AREA_ABSENT", area, label)
                 for area, label in inv.REQUIRED_AREAS.items() if area not in present]
    return findings


def _dotted(rel: str) -> str:
    return rel.removesuffix(".py").replace("/", ".")


def adapter_findings(document: dict) -> list[dict]:
    findings = []
    for edge in document["adapter_edges"]:
        if edge["owner"] not in inv.EDGE_OWNERS:
            findings.append(_finding("ADAPTER_EDGE_UNOWNED", edge["id"],
                                     f"owner {edge['owner']!r} (ledger label {edge['ledger_removal_phase']!r})"))
        if not edge.get("module_exists", True):
            findings.append(_finding("ADAPTER_MODULE_MISSING", edge["id"], edge["module"]))
    declared = {_dotted(e["module"]) if "/" in e["module"] else e["module"] for e in document["adapter_edges"]}
    findings += [_finding("ADAPTER_MODULE_UNDECLARED", rel, "adapter module has no edge")
                 for rel in document["adapter_modules"] if _dotted(rel) not in declared]
    return findings


def guide_findings(document: dict, root: Path = inv.ROOT) -> list[dict]:
    path = root / inv.GUIDE
    block = inv.guide_block(path.read_text()) if path.is_file() else None
    if block != inv.render_markdown(document):
        return [_finding("GUIDE_STALE", str(inv.GUIDE),
                         "generated section differs; run python3 tools/phase6_inventory.py --write-guide")]
    return []


def run(root: Path = inv.ROOT, declarations: Path | None = None, discovered: dict | None = None,
        check_guide: bool = True) -> tuple[dict, list[dict]]:
    document = inv.build_document(root, declarations, discovered)
    findings = entrypoint_findings(document) + row_findings(document, root) + adapter_findings(document)
    if check_guide:
        findings += guide_findings(document, root)
    return document, findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P6-1 capability matrix check")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    document, findings = run()
    if args.json:
        print(json.dumps({"ok": not findings, "counts": inv.counts(document), "findings": findings}, indent=2))
    else:
        for item in findings:
            print(f"{item['code']}: {item['subject']}: {item['detail']}")
        print(f"phase6 inventory: {len(document['rows'])} rows, {len(document['adapter_edges'])} edges, "
              f"{len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
