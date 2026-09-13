#!/usr/bin/env python3
"""Phase 1 engineering evidence over a specified checkout, including new files.

This check consumes source/artifact evidence. Production never imports it.
Decision correctness, projection security and engineering budgets remain
separate gates; passing this command does not authorize a prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks import code_budgets, import_layers, install_hooks, package_readmes, repo_hygiene
from checks.layer_map import PACKAGES
from checks.rearchitecture_phase1_coverage import BASELINE, compare, validate_measurement
from checks.rearchitecture_phase1_lint import run as lint


def source_files(root):
    command = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others",
                              "--exclude-standard", "-z"], capture_output=True, check=True)
    rels = set(command.stdout.decode().split("\0")) - {""}
    return {rel: (root / rel).read_bytes() for rel in sorted(rels) if (root / rel).is_file()}


def source_hash(files):
    digest = hashlib.sha256()
    for name, blob in sorted(files.items()):
        digest.update(name.encode() + b"\0" + hashlib.sha256(blob).digest())
    return "sha256:" + digest.hexdigest()


COVERAGE_COMMAND = ("python3 checks/rearchitecture_phase1_coverage.py --measure "
                    "--output <file>; then rerun this gate with --coverage <file>")


def coverage_check(path, baseline, root=ROOT):
    if path is None or not path.is_file() or not baseline.is_file():
        # Missing evidence is a failing check, never a green one. The bare
        # invocation is therefore red by design: measure first, then pass
        # the measurement with --coverage.
        return {"ok": False, "findings": [{
            "code": "COVERAGE_EVIDENCE_MISSING",
            "reason": ("no coverage measurement supplied; run without --coverage "
                       "is red by design"),
            "produce_with": COVERAGE_COMMAND}]}
    try:
        measured = json.loads(path.read_text())
        findings = compare(measured, json.loads(baseline.read_text()))
        findings.extend(validate_measurement(measured, root))
        return {"ok": not findings, "findings": findings}
    except (ValueError, KeyError, TypeError):
        return {"ok": False, "findings": [{"code": "COVERAGE_EVIDENCE_INVALID"}]}


def gate(root=ROOT, *, coverage_path=None):
    started = time.monotonic()
    print("[phase1-gate] reading source inventory", file=sys.stderr, flush=True)
    files = source_files(root)
    python = {rel: blob for rel, blob in files.items() if rel.endswith(".py")}
    readmes = {p.dotted: (root / p.path / "README.md").read_text()
               if (root / p.path / "README.md").exists() else None for p in PACKAGES}
    budget = code_budgets.check_files(python)
    layers = import_layers.check_files(python, import_layers.load_adapters(root / "checks/legacy_adapters.json"))
    readme = package_readmes.check(python, readmes)
    hygiene = repo_hygiene.check_files(files, repo_hygiene.load_secrets(root / ".env"))
    hook = install_hooks.state(root)
    print("[phase1-gate] structural checks complete; lint and coverage evidence", file=sys.stderr, flush=True)
    structural = {
        "imports": {"ok": layers.ok, "findings": [str(v) for v in layers.violations]},
        "readmes": {"ok": readme.ok, "findings": [str(v) for v in readme.violations]},
        "hygiene": {"ok": hygiene.ok, "findings": [str(v) for v in hygiene.violations]},
    }
    engineering = {
        "budgets": {"ok": budget.ok, "findings": [str(v) for v in budget.violations]},
        "lint": lint(root, worktree=True),
        "coverage": coverage_check(coverage_path, root / BASELINE.relative_to(ROOT), root),
        "hook": {"ok": hook["installed"] and hook["current"] and hook["executable"], **hook},
    }
    return {"schema_version": "phase1_engineering_gate.v1.0", "code_hash": source_hash(files),
            "ok": all(r["ok"] for r in [*structural.values(), *engineering.values()]),
            "structural": structural, "engineering": engineering,
            "decision_correctness": {"ok": None, "reason": "requires per-candidate receipts"},
            "projection_security": {"ok": None, "reason": "requires serialized release and access receipts"},
            "seconds": time.monotonic() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--coverage", type=Path,
                        help="coverage measurement JSON from: " + COVERAGE_COMMAND
                        + " (required for a green gate; omitting it fails the "
                        "coverage row with COVERAGE_EVIDENCE_MISSING)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = gate(args.repo_root, coverage_path=args.coverage)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)
    return int(not result["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
