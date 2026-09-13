#!/usr/bin/env python3
"""Per-package v2 line coverage ratchet. Measurement never changes the baseline.

Fixed command: coverage run --source=engine/v2 -m pytest
tests/test_v2_ops_*.py tests/test_v2_data_*.py tests/test_diagnosis_comparator.py.
The exact sorted test-file list is recorded in every measurement.

``SUITE_VERSION`` bumped to v2 (P2-5/B1c review fix #3): the fixed suite
covered only ``tests/test_v2_ops_*.py`` plus one diagnosis file, so every
``engine.v2.data`` module merged since Phase 2 opened lowered that package's
measured percentage without a single line of it actually going untested --
its own ``tests/test_v2_data_*.py`` files simply were never run here. Widening
the suite is a real behavior change (the measured numbers move), which is
exactly why the version is part of ``compare``'s drift check: a v1 baseline
can never be compared against a v2 measurement by accident.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.layer_map import PACKAGES, package_of

BASELINE = ROOT / "checks/rearchitecture_phase1_coverage_baseline.json"
SUITE_VERSION = "phase1_coverage_suite.v2"


def suite(root=ROOT):
    return sorted([p.relative_to(root).as_posix() for p in (root / "tests").glob("test_v2_ops_*.py")]
                  + [p.relative_to(root).as_posix() for p in (root / "tests").glob("test_v2_data_*.py")]
                  + ["tests/test_diagnosis_comparator.py"])


def measurement_identity(root=ROOT):
    paths = sorted([p.relative_to(root).as_posix() for p in (root / "engine/v2").rglob("*.py")]
                   + suite(root))
    digest = hashlib.sha256()
    for rel in paths:
        digest.update(rel.encode() + b"\0" + hashlib.sha256((root / rel).read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


def empty_package(files):
    for path in files:
        tree = ast.parse(path.read_text())
        body = [node for node in tree.body if not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str))]
        if body:
            return False
    return True


def package_counts(document, root=ROOT):
    groups = {p.dotted: {"executed": 0, "executable": 0, "files": []} for p in PACKAGES}
    for rel, item in document["files"].items():
        path = Path(rel)
        if path.is_absolute():
            path = path.relative_to(root)
        module = path.with_suffix("").as_posix().replace("/", ".")
        package = package_of(module)
        if package is None:
            continue
        group = groups[package.dotted]
        group["executed"] += item["summary"]["covered_lines"]
        group["executable"] += item["summary"]["num_statements"]
        group["files"].append(path.as_posix())
    for package in PACKAGES:
        group = groups[package.dotted]
        paths = [p for p in (root / package.path).rglob("*.py")
                 if package_of(p.relative_to(root).with_suffix("").as_posix().replace("/", ".")) == package]
        observed = set(group["files"])
        group["missing_files"] = [p.relative_to(root).as_posix() for p in paths
                                  if p.relative_to(root).as_posix() not in observed]
        group["empty"] = empty_package(paths)
        group["percentage"] = (100.0 * group["executed"] / group["executable"]
                               if group["executable"] else None)
    return groups


def compare(measured, baseline):
    failures = []
    if any(document.get("suite_version") != SUITE_VERSION for document in (measured, baseline)):
        failures.append({"code": "COVERAGE_SUITE_DRIFT"})
    if set(measured["packages"]) != set(baseline.get("packages", {})):
        failures.append({"code": "COVERAGE_PACKAGE_INVENTORY_DRIFT"})
    for name, current in measured["packages"].items():
        previous = baseline.get("packages", {}).get(name)
        counts = (current.get("executed"), current.get("executable"))
        if (any(type(value) is not int for value in counts)
                or not 0 <= counts[0] <= counts[1]):
            failures.append({"code": "COVERAGE_COUNTS_INVALID", "package": name})
            continue
        if current["missing_files"]:
            failures.append({"code": "COVERAGE_FILES_MISSING", "package": name})
        if previous is None:
            failures.append({"code": "COVERAGE_BASELINE_MISSING", "package": name})
            continue
        if current["empty"]:
            continue
        if not current["executable"] or not current["executed"]:
            failures.append({"code": "COVERAGE_UNMEASURED_PACKAGE", "package": name})
        elif previous["empty"] or not previous["executable"]:
            failures.append({"code": "COVERAGE_NEW_PACKAGE_BASELINE_REQUIRED", "package": name})
        elif current["executed"] * previous["executable"] < previous["executed"] * current["executable"]:
            failures.append({"code": "COVERAGE_REGRESSION", "package": name,
                             "previous": [previous["executed"], previous["executable"]],
                             "current": [current["executed"], current["executable"]]})
    return failures


def validate_measurement(measured, root=ROOT):
    """Evidence must cover the current complete source and fixed test command."""
    findings = []
    if measured.get("schema_version") != "phase1_coverage.v1.0":
        findings.append({"code": "COVERAGE_SCHEMA_INVALID"})
    if measured.get("test_files") != suite(root):
        findings.append({"code": "COVERAGE_TEST_INVENTORY_DRIFT"})
    if set(measured.get("packages", {})) != {p.dotted for p in PACKAGES}:
        findings.append({"code": "COVERAGE_PACKAGE_INVENTORY_DRIFT"})
    if measured.get("source_hash") != measurement_identity(root):
        findings.append({"code": "COVERAGE_SOURCE_DRIFT"})
    return findings


def measure(root=ROOT):
    tests = suite(root)
    source_hash = measurement_identity(root)
    with tempfile.TemporaryDirectory(prefix="phase1-coverage-") as scratch:
        env = dict(os.environ, COVERAGE_FILE=str(Path(scratch) / "coverage"))
        command = [sys.executable, "-m", "coverage", "run", "--source=engine/v2",
                   "-m", "pytest", "-q", *tests]
        print(f"[coverage] running fixed suite ({len(tests)} files)", flush=True)
        run = subprocess.run(command, cwd=root, env=env, check=False)
        if run.returncode:
            raise RuntimeError("coverage test suite failed")
        output = Path(scratch) / "coverage.json"
        subprocess.run([sys.executable, "-m", "coverage", "json", "-o", str(output)],
                       cwd=root, env=env, check=True)
        document = json.loads(output.read_text())
    if measurement_identity(root) != source_hash:
        raise RuntimeError("source changed during coverage measurement; rerun against stable code")
    return {"schema_version": "phase1_coverage.v1.0", "suite_version": SUITE_VERSION,
            "test_files": tests, "source_hash": source_hash, "packages": package_counts(document, root)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args(argv)
    result = measure() if args.measure else json.loads(args.input.read_text())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    failures = compare(result, json.loads(args.baseline.read_text())) if args.baseline.exists() else [
        {"code": "COVERAGE_BASELINE_MISSING"}]
    print(json.dumps({"ok": not failures, "findings": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
