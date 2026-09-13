#!/usr/bin/env python3
"""Per-package v2 line coverage ratchet for the Phase 2 suite.

The fixed suite is every test file named in ``checks/phase2_acceptance.json``
that currently exists on disk, plus ``tests/test_v2_ops_engineering.py``
(phase-2 guide §12.1). ``suite_version`` is a hash of that exact file list, so
a registry change that adds or removes a covered file invalidates the
committed baseline (``SUITE_DRIFT``) rather than silently comparing two
different suites.

Unlike the Phase 1 script, ``source_hash`` here is the SAME whole-tree hash
``rearchitecture_phase1_gate.source_hash`` computes and
``rearchitecture_phase2_gate.py``/the Phase2Evidence document use as
``code_hash`` -- phase-2 guide §12.1: "Both Phase 2 inputs record the same
code hash." Package percentages still come only from ``engine/v2``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks.layer_map import PACKAGES
from checks.rearchitecture_phase1_coverage import package_counts
from checks.rearchitecture_phase1_gate import source_files, source_hash

BASELINE = ROOT / "checks/rearchitecture_phase2_coverage_baseline.json"
REGISTRY = ROOT / "checks/phase2_acceptance.json"
SCHEMA_VERSION = "phase2_coverage.v1.0"


def load_registry(path=REGISTRY):
    return json.loads(path.read_text())["rows"]


def suite(root=ROOT, registry=None):
    """The fixed, existing test files: every registry test that exists, plus engineering."""
    registry = load_registry() if registry is None else registry
    declared = {t for row in registry.values() for t in row.get("tests", [])}
    declared.add("tests/test_v2_ops_engineering.py")
    return sorted(rel for rel in declared if (root / rel).is_file())


def suite_version(files):
    digest = hashlib.sha256("\0".join(sorted(files)).encode()).hexdigest()
    return "phase2_coverage_suite:" + digest


def _outcome(testcase):
    for tag in ("failure", "error"):
        if testcase.find(tag) is not None:
            return "failed"
    if testcase.find("skipped") is not None:
        return "skipped"
    return "passed"


def _nodeid(classname, name, root):
    """Reconstruct a pytest node id from junit xunit2's dotted classname.

    junit does not record the ``::``-separated node id directly, only a
    dotted ``classname`` (module, then any enclosing test classes) and the
    bare test ``name``. The longest dotted prefix that names a real file on
    disk is the module; anything left over is enclosing class names.
    """
    parts = classname.split(".")
    for split in range(len(parts), 0, -1):
        candidate = "/".join(parts[:split]) + ".py"
        if (root / candidate).is_file():
            remainder = parts[split:]
            return candidate + "::" + "::".join([*remainder, name])
    return classname.replace(".", "/") + ".py::" + name


def parse_junit(path, root=ROOT):
    tree = ElementTree.parse(path)
    outcomes = []
    for testcase in tree.getroot().iter("testcase"):
        classname = testcase.get("classname", "")
        name = testcase.get("name", "")
        outcomes.append({"nodeid": _nodeid(classname, name, root), "outcome": _outcome(testcase)})
    return outcomes


def measurement_identity(root=ROOT):
    """The same whole-tree hash the Phase 2 gate and Phase2Evidence use as code_hash."""
    return source_hash(source_files(root))


def measure(root=ROOT, *, parallel=False):
    """``parallel=True`` runs the fixed suite under ``-n auto --dist
    loadgroup``. Coverage under xdist needs coverage.py's own multi-process
    combine, not pytest-cov: every worker is a fresh interpreter, so
    ``COVERAGE_PROCESS_START`` plus the system ``coverage.process_startup``
    ``.pth`` hook starts a coverage instance in each one automatically
    (config ``parallel = true``), and ``coverage combine`` merges their
    suffixed data files before ``coverage json`` reads the total. Junit
    outcomes still come from one ``--junitxml`` on the controller process;
    xdist forwards every worker's own test reports into that same junit
    plugin, so per-test outcomes should stay complete regardless of which
    worker ran which test, and junit's classname/name shape is pytest's own
    (unaffected by which worker collected/ran a test), not xdist's -- so
    ``_nodeid`` here should not need to change. NOT yet run under real
    xdist workers to confirm; verify with a live ``--parallel`` measurement
    before relying on this comment."""
    registry = load_registry()
    tests = suite(root, registry)
    version = suite_version(tests)
    identity = measurement_identity(root)
    with tempfile.TemporaryDirectory(prefix="phase2-coverage-") as scratch:
        data_file = str(Path(scratch) / "coverage")
        env = dict(os.environ, COVERAGE_FILE=data_file)
        junit = Path(scratch) / "junit.xml"
        if parallel:
            rcfile = Path(scratch) / "parallel.coveragerc"
            rcfile.write_text("[run]\nparallel = true\nsource = engine/v2\n")
            env["COVERAGE_PROCESS_START"] = str(rcfile)
            command = [sys.executable, "-m", "coverage", "run", f"--rcfile={rcfile}",
                       "-m", "pytest", "-q", "-n", "auto", "--dist", "loadgroup",
                       f"--junitxml={junit}", *tests]
        else:
            command = [sys.executable, "-m", "coverage", "run", "--source=engine/v2",
                       "-m", "pytest", "-q", f"--junitxml={junit}", *tests]
        print(f"[phase2-coverage] running fixed suite ({len(tests)} files)"
              + (" under -n auto --dist loadgroup" if parallel else ""), flush=True)
        run = subprocess.run(command, cwd=root, env=env, check=False)
        outcomes = parse_junit(junit, root) if junit.is_file() else []
        if run.returncode and not outcomes:
            raise RuntimeError("phase 2 coverage suite produced no junit report")
        if parallel:
            subprocess.run([sys.executable, "-m", "coverage", "combine"],
                           cwd=root, env=env, check=True)
        output = Path(scratch) / "coverage.json"
        subprocess.run([sys.executable, "-m", "coverage", "json", "-o", str(output)],
                       cwd=root, env=env, check=True)
        document = json.loads(output.read_text())
    if measurement_identity(root) != identity:
        raise RuntimeError("source changed during coverage measurement; rerun against stable code")
    return {"schema_version": SCHEMA_VERSION, "suite_version": version, "test_files": tests,
            "source_hash": identity, "packages": package_counts(document, root),
            "test_outcomes": outcomes}


def compare(measured, baseline):
    """Package-level ratchet findings, using the Phase 2 gate's own code vocabulary."""
    findings = []
    if measured.get("suite_version") != baseline.get("suite_version"):
        findings.append({"code": "SUITE_DRIFT"})
    if set(measured.get("packages", {})) != set(baseline.get("packages", {})):
        findings.append({"code": "SUITE_DRIFT", "reason": "package inventory"})
    for name, current in measured.get("packages", {}).items():
        previous = baseline.get("packages", {}).get(name)
        counts = (current.get("executed"), current.get("executable"))
        if any(type(v) is not int for v in counts) or not 0 <= counts[0] <= counts[1]:
            findings.append({"code": "STALE_COVERAGE", "package": name, "reason": "invalid counts"})
            continue
        if current.get("missing_files"):
            findings.append({"code": "STALE_COVERAGE", "package": name, "reason": "missing files"})
        if previous is None:
            findings.append({"code": "STALE_COVERAGE", "package": name, "reason": "no baseline"})
            continue
        if current.get("empty"):
            continue
        # The Phase 2 suite is eight files, not the whole v2 tree: many packages
        # are legitimately untouched by it (they belong to Phase 1's suite or a
        # later phase). Only a RATIO drop against this package's own baseline
        # counts as a regression -- "still zero, as before" is not one, and
        # "was covered, now zero" is caught by the ratio itself.
        if not current.get("executable") or previous.get("empty") or not previous.get("executable"):
            continue
        if current["executed"] * previous["executable"] < previous["executed"] * current["executable"]:
            findings.append({"code": "COVERAGE_REGRESSION", "package": name,
                             "previous": [previous["executed"], previous["executable"]],
                             "current": [current["executed"], current["executable"]]})
    return findings


def validate_measurement(measured, root=ROOT, registry=None):
    """Evidence must cover the fixed suite as it currently exists, on the current tree."""
    findings = []
    registry = load_registry() if registry is None else registry
    if measured.get("schema_version") != SCHEMA_VERSION:
        findings.append({"code": "STALE_COVERAGE", "reason": "schema"})
    if measured.get("test_files") != suite(root, registry):
        findings.append({"code": "SUITE_DRIFT", "reason": "test file list"})
    if measured.get("suite_version") != suite_version(measured.get("test_files") or []):
        findings.append({"code": "SUITE_DRIFT", "reason": "suite version"})
    if set(measured.get("packages", {})) != {p.dotted for p in PACKAGES}:
        findings.append({"code": "SUITE_DRIFT", "reason": "package inventory"})
    if measured.get("source_hash") != measurement_identity(root):
        findings.append({"code": "STALE_COVERAGE", "reason": "source hash"})
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--parallel", action="store_true",
                        help="run the fixed suite under -n auto --dist loadgroup "
                             "(requires pytest-xdist); default is serial")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args(argv)
    result = measure(parallel=args.parallel) if args.measure else json.loads(args.input.read_text())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    failures = compare(result, json.loads(args.baseline.read_text())) if args.baseline.exists() else [
        {"code": "STALE_COVERAGE", "reason": "no baseline"}]
    print(json.dumps({"ok": not failures, "findings": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
