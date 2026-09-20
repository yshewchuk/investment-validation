#!/usr/bin/env python3
"""Per-package v2 line coverage ratchet, in three pinned-suite profiles.

Not phase-scoped: ``engine/v2`` keeps growing after every rearchitecture phase
closes, so this measure/compare implementation is permanent infrastructure,
not an artifact of one migration phase -- only the three PROFILES below
(phase1, phase2, phase3) stay pinned to the fixed test suite and committed
baseline each phase's gate checks against. Measurement never changes a
baseline.

phase1 profile
--------------
Fixed command: coverage run --source=engine/v2 -m pytest
tests/test_v2_ops_*.py tests/test_v2_data_*.py tests/test_v2_dashboard_*.py
tests/test_v2_serving_*.py tests/test_diagnosis_comparator.py. The exact
sorted test-file list is recorded in every measurement.

``PHASE1_SUITE_VERSION`` bumped to v2 (P2-5/B1c review fix #3): the fixed
suite covered only ``tests/test_v2_ops_*.py`` plus one diagnosis file, so
every ``engine.v2.data`` module merged since Phase 2 opened lowered that
package's measured percentage without a single line of it actually going
untested -- its own ``tests/test_v2_data_*.py`` files simply were never run
here. Widening the suite is a real behavior change (the measured numbers
move), which is exactly why the version is part of ``phase1_compare``'s drift
check: a v1 baseline can never be compared against a v2 measurement by
accident.

Bumped again to v3 by two independent same-day changes, merged together:

* (P3-0) ``engine/v2/dashboard`` gained its first production code (the
  compatibility preview launcher), so it stopped being the ``empty: true``
  package the v2 baseline recorded and needed its own fixed test file counted
  here for the same reason ``data`` did above -- ``tests/test_v2_dashboard_*.py``
  was added to the glob for exactly this.
* (P3-1a) the first real production module under ``engine/v2/serving``
  (``bridge.py``, plus its ``contracts/serving.py`` schemas) lands with its
  own ``tests/test_v2_serving_bridge.py`` -- named for the *layer*
  (``serving``), not the ops adapter, so it matched neither existing glob and
  would otherwise merge as permanently-uncovered ``engine.v2.serving`` source.
  ``phase1_suite()`` now also picks up ``tests/test_v2_serving_*.py``.

phase2 profile
--------------
The fixed suite is every test file named in ``checks/phase2_acceptance.json``
that currently exists on disk, plus ``tests/test_v2_ops_engineering.py``
(phase-2 guide §12.1). ``suite_version`` is a hash of that exact file list, so
a registry change that adds or removes a covered file invalidates the
committed baseline (``SUITE_DRIFT``) rather than silently comparing two
different suites.

Unlike the phase1 profile, ``phase2_measurement_identity`` here is the SAME
whole-tree hash ``rearchitecture_phase1_gate.source_hash`` computes and
``rearchitecture_phase2_gate.py``/the Phase2Evidence document use as
``code_hash`` -- phase-2 guide §12.1: "Both Phase 2 inputs record the same
code hash." Package percentages still come only from ``engine/v2``.

phase3 profile
--------------
The fixed suite is ``PHASE3_FIXED_SUITE``, a hand-picked list of every real
test file that exercises the Phase 3 read API / bridge / projections /
UI-facing server code (no ``phase3_acceptance.json`` "tests" registry exists
to derive this from, unlike phase1/phase2), kept only if it exists on disk.
``rearchitecture_phase3_quality.py`` calls ``phase3_measure_coverage`` to
produce the ``coverage_receipt_ref`` evidence document and
``rearchitecture_phase3_evidence.py`` calls ``phase3_coverage_findings`` to
ratchet it, reusing this module's ``package_counts`` rather than
reimplementing it.
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

# --------------------------------------------------------------------------
# shared: reused verbatim by every profile
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# phase1 profile
# --------------------------------------------------------------------------

PHASE1_BASELINE = ROOT / "checks/v2_coverage_ratchet_phase1_baseline.json"
PHASE1_SUITE_VERSION = "phase1_coverage_suite.v3"


def phase1_suite(root=ROOT):
    return sorted([p.relative_to(root).as_posix() for p in (root / "tests").glob("test_v2_ops_*.py")]
                  + [p.relative_to(root).as_posix() for p in (root / "tests").glob("test_v2_data_*.py")]
                  + [p.relative_to(root).as_posix() for p in (root / "tests").glob("test_v2_dashboard_*.py")]
                  + [p.relative_to(root).as_posix() for p in (root / "tests").glob("test_v2_serving_*.py")]
                  + ["tests/test_diagnosis_comparator.py"])


def phase1_measurement_identity(root=ROOT):
    paths = sorted([p.relative_to(root).as_posix() for p in (root / "engine/v2").rglob("*.py")]
                   + phase1_suite(root))
    digest = hashlib.sha256()
    for rel in paths:
        digest.update(rel.encode() + b"\0" + hashlib.sha256((root / rel).read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


def phase1_compare(measured, baseline):
    failures = []
    # A baseline captured under `--parallel` can carry the executor_watchdog.py
    # /proc-race noise `phase1_measure()` documents above (execution can only
    # ADD lines under contention) baked in as if it were the truth -- comparing
    # a later serial (contention-free) measurement against it would then
    # read as a regression that never happened. Refuse the baseline itself,
    # not the measurement: a parallel MEASUREMENT against a serial baseline
    # is still fine (only decreases trip COVERAGE_REGRESSION, and this bug
    # only adds). A baseline missing "mode" entirely, for today's own suite
    # version, predates this field and cannot be trusted either -- once
    # every committed baseline carries "mode": "serial" this never fires
    # for real ones again.
    baseline_mode = baseline.get("mode")
    if baseline_mode == "parallel" or (
            baseline_mode is None and baseline.get("suite_version") == measured.get("suite_version")):
        failures.append({"code": "BASELINE_NOT_SERIAL"})
    if any(document.get("suite_version") != PHASE1_SUITE_VERSION for document in (measured, baseline)):
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


def phase1_validate_measurement(measured, root=ROOT):
    """Evidence must cover the current complete source and fixed test command."""
    findings = []
    if measured.get("schema_version") != "phase1_coverage.v1.0":
        findings.append({"code": "COVERAGE_SCHEMA_INVALID"})
    if measured.get("test_files") != phase1_suite(root):
        findings.append({"code": "COVERAGE_TEST_INVENTORY_DRIFT"})
    if set(measured.get("packages", {})) != {p.dotted for p in PACKAGES}:
        findings.append({"code": "COVERAGE_PACKAGE_INVENTORY_DRIFT"})
    if measured.get("source_hash") != phase1_measurement_identity(root):
        findings.append({"code": "COVERAGE_SOURCE_DRIFT"})
    return findings


def phase1_measure(root=ROOT, *, parallel=False):
    """``parallel=True`` runs the same fixed suite under ``-n auto --dist
    loadgroup`` instead of one process. Coverage still counts every worker:
    each xdist worker is a fresh interpreter (execnet, not a fork), so a
    single ``coverage run`` around the controller process alone would only
    see the controller's own near-zero execution. Instead every subprocess
    gets ``COVERAGE_PROCESS_START`` pointing at a ``parallel = true``
    coveragerc; the system-wide ``coverage.process_startup`` sitecustomize
    hook (installed as a ``.pth`` file) starts a coverage instance in each
    new interpreter automatically, writing its own suffixed data file next
    to the controller's. ``coverage combine`` then merges all of them before
    ``coverage json`` reads the total -- no pytest-cov needed.

    Verified against two consecutive serial and two consecutive parallel
    measurements: identical per-package executed/executable counts every
    time (engine.v2.ops included). ``coverage combine``'s "skipped N" line
    is its own content-hash dedup of data files whose recorded lines are
    byte-identical to one already combined (``coverage/data.py``'s
    ``DataFileClassifier``) -- real, correct behavior, not lost data; most
    of engine.v2.ops's real short-lived subprocess tests execute the same
    handful of lines, so many of their data files hash identically.

    One known, small, execution-order-dependent noise source in
    engine.v2.ops specifically (not caused by this flag, and not always
    triggered): ``process_table()`` in ``executor_watchdog.py`` scans every
    PID under ``/proc`` and can hit a real TOCTOU race when a process exits
    mid-scan (caught by its own ``except (OSError, ValueError, IndexError):
    continue``) -- more real concurrent process churn (more xdist workers,
    more real subprocesses alive at once) makes that except branch more
    likely to execute, never less. Seen once, isolated to exactly those two
    lines, in a phase2-profile --parallel run (see ``phase2_measure``).
    It can only ADD executed lines under contention, so it can never trip
    ``phase1_compare``'s regression check (which only fires on a DECREASE) --
    but if a --parallel measurement here ever looks unexpectedly higher by
    a line or two, check this function first before suspecting the combine
    step."""
    tests = phase1_suite(root)
    source_hash = phase1_measurement_identity(root)
    with tempfile.TemporaryDirectory(prefix="phase1-coverage-") as scratch:
        data_file = str(Path(scratch) / "coverage")
        env = dict(os.environ, COVERAGE_FILE=data_file)
        if parallel:
            rcfile = Path(scratch) / "parallel.coveragerc"
            rcfile.write_text("[run]\nparallel = true\nsource = engine/v2\n")
            env["COVERAGE_PROCESS_START"] = str(rcfile)
            command = [sys.executable, "-m", "coverage", "run", f"--rcfile={rcfile}",
                       "-m", "pytest", "-q", "-n", "auto", "--dist", "loadgroup", *tests]
        else:
            command = [sys.executable, "-m", "coverage", "run", "--source=engine/v2",
                       "-m", "pytest", "-q", *tests]
        print(f"[coverage] running fixed suite ({len(tests)} files)"
              + (" under -n auto --dist loadgroup" if parallel else ""), flush=True)
        run = subprocess.run(command, cwd=root, env=env, check=False)
        if run.returncode:
            raise RuntimeError("coverage test suite failed")
        if parallel:
            subprocess.run([sys.executable, "-m", "coverage", "combine"],
                           cwd=root, env=env, check=True)
        output = Path(scratch) / "coverage.json"
        subprocess.run([sys.executable, "-m", "coverage", "json", "-o", str(output)],
                       cwd=root, env=env, check=True)
        document = json.loads(output.read_text())
    if phase1_measurement_identity(root) != source_hash:
        raise RuntimeError("source changed during coverage measurement; rerun against stable code")
    return {"schema_version": "phase1_coverage.v1.0", "suite_version": PHASE1_SUITE_VERSION,
            "test_files": tests, "source_hash": source_hash, "packages": package_counts(document, root),
            "mode": "parallel" if parallel else "serial"}


# --------------------------------------------------------------------------
# phase2 profile
# --------------------------------------------------------------------------

PHASE2_BASELINE = ROOT / "checks/v2_coverage_ratchet_phase2_baseline.json"
PHASE2_REGISTRY = ROOT / "checks/phase2_acceptance.json"
PHASE2_SCHEMA_VERSION = "phase2_coverage.v1.0"


def phase2_load_registry(path=PHASE2_REGISTRY):
    return json.loads(path.read_text())["rows"]


def phase2_suite(root=ROOT, registry=None):
    """The fixed, existing test files: every registry test that exists, plus engineering."""
    registry = phase2_load_registry() if registry is None else registry
    declared = {t for row in registry.values() for t in row.get("tests", [])}
    declared.add("tests/test_v2_ops_engineering.py")
    return sorted(rel for rel in declared if (root / rel).is_file())


def phase2_suite_version(files):
    digest = hashlib.sha256("\0".join(sorted(files)).encode()).hexdigest()
    return "phase2_coverage_suite:" + digest


def _phase2_outcome(testcase):
    for tag in ("failure", "error"):
        if testcase.find(tag) is not None:
            return "failed"
    if testcase.find("skipped") is not None:
        return "skipped"
    return "passed"


def _phase2_nodeid(classname, name, root):
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


def phase2_parse_junit(path, root=ROOT):
    from xml.etree import ElementTree
    tree = ElementTree.parse(path)
    outcomes = []
    for testcase in tree.getroot().iter("testcase"):
        classname = testcase.get("classname", "")
        name = testcase.get("name", "")
        outcomes.append({"nodeid": _phase2_nodeid(classname, name, root), "outcome": _phase2_outcome(testcase)})
    return outcomes


def phase2_measurement_identity(root=ROOT):
    """The same whole-tree hash the Phase 2 gate and Phase2Evidence use as code_hash."""
    from checks.rearchitecture_phase1_gate import source_files, source_hash
    return source_hash(source_files(root))


def phase2_measure(root=ROOT, *, parallel=False):
    """``parallel=True`` runs the fixed suite under ``-n auto --dist
    loadgroup``. Coverage under xdist needs coverage.py's own multi-process
    combine, not pytest-cov: every worker is a fresh interpreter, so
    ``COVERAGE_PROCESS_START`` plus the system ``coverage.process_startup``
    ``.pth`` hook starts a coverage instance in each one automatically
    (config ``parallel = true``), and ``coverage combine`` merges their
    suffixed data files before ``coverage json`` reads the total. Junit
    outcomes still come from one ``--junitxml`` on the controller process;
    xdist forwards every worker's own test reports into that same junit
    plugin, so per-test outcomes stay complete regardless of which worker
    ran which test, and junit's classname/name shape is pytest's own
    (unaffected by which worker collected/ran a test), not xdist's, so
    ``_phase2_nodeid`` needed no change. Verified with a live measurement: 361
    outcomes both serial and parallel, identical nodeids, identical
    per-test outcomes.

    Package counts, however, are NOT guaranteed identical here the way
    ``phase1_measure``'s are: one parallel run measured engine.v2.ops 2 lines
    HIGHER than two consecutive serial runs (2709 vs 2707 executed, out of
    4083). Traced to exact line numbers -- only executor_watchdog.py:37-38,
    the ``except (OSError, ValueError, IndexError): continue`` in
    ``process_table()``'s real ``/proc`` scan -- a genuine TOCTOU race (a
    process exiting mid-scan) that more real concurrent process churn makes
    more likely to hit, not a combine bug (junit above and phase1's 4-for-4
    identical packages both confirm the combine mechanism itself is sound).
    Left available rather than refused: this can only ADD executed lines
    under contention, never remove them, so it can never trip
    ``phase2_compare``'s regression check (DECREASE-only) or the Phase 2
    gate's coverage row -- if a --parallel measurement here looks
    unexpectedly higher by a line or two, check executor_watchdog.py first
    before suspecting data loss."""
    registry = phase2_load_registry()
    tests = phase2_suite(root, registry)
    version = phase2_suite_version(tests)
    identity = phase2_measurement_identity(root)
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
        outcomes = phase2_parse_junit(junit, root) if junit.is_file() else []
        if run.returncode and not outcomes:
            raise RuntimeError("phase 2 coverage suite produced no junit report")
        if parallel:
            subprocess.run([sys.executable, "-m", "coverage", "combine"],
                           cwd=root, env=env, check=True)
        output = Path(scratch) / "coverage.json"
        subprocess.run([sys.executable, "-m", "coverage", "json", "-o", str(output)],
                       cwd=root, env=env, check=True)
        document = json.loads(output.read_text())
    if phase2_measurement_identity(root) != identity:
        raise RuntimeError("source changed during coverage measurement; rerun against stable code")
    return {"schema_version": PHASE2_SCHEMA_VERSION, "suite_version": version, "test_files": tests,
            "source_hash": identity, "packages": package_counts(document, root),
            "test_outcomes": outcomes, "mode": "parallel" if parallel else "serial"}


def phase2_compare(measured, baseline):
    """Package-level ratchet findings, using the Phase 2 gate's own code vocabulary."""
    findings = []
    # See phase1_compare()'s full reason: a baseline captured under --parallel
    # can bake in the executor_watchdog.py /proc-race noise (measured here
    # once: engine.v2.ops 2709 vs 2707/4083) as if it were the truth, so a
    # later serial measurement would read as a false regression. Refuse the
    # baseline, not the measurement -- a parallel measurement against a
    # serial baseline is still fine.
    baseline_mode = baseline.get("mode")
    if baseline_mode == "parallel" or (
            baseline_mode is None and baseline.get("suite_version") == measured.get("suite_version")):
        findings.append({"code": "BASELINE_NOT_SERIAL"})
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


def phase2_validate_measurement(measured, root=ROOT, registry=None):
    """Evidence must cover the fixed suite as it currently exists, on the current tree."""
    findings = []
    registry = phase2_load_registry() if registry is None else registry
    if measured.get("schema_version") != PHASE2_SCHEMA_VERSION:
        findings.append({"code": "STALE_COVERAGE", "reason": "schema"})
    if measured.get("test_files") != phase2_suite(root, registry):
        findings.append({"code": "SUITE_DRIFT", "reason": "test file list"})
    if measured.get("suite_version") != phase2_suite_version(measured.get("test_files") or []):
        findings.append({"code": "SUITE_DRIFT", "reason": "suite version"})
    if set(measured.get("packages", {})) != {p.dotted for p in PACKAGES}:
        findings.append({"code": "SUITE_DRIFT", "reason": "package inventory"})
    if measured.get("source_hash") != phase2_measurement_identity(root):
        findings.append({"code": "STALE_COVERAGE", "reason": "source hash"})
    return findings


# --------------------------------------------------------------------------
# phase3 profile
# --------------------------------------------------------------------------

PHASE3_BASELINE = ROOT / "checks/v2_coverage_ratchet_phase3_baseline.json"
PHASE3_COVERAGE_SCHEMA = "phase3_coverage.v1.0"
PHASE3_BASELINE_SCHEMA = "phase3_coverage_baseline.v1.0"

#: A scope choice (no phase3_acceptance.json "tests" registry exists to
#: derive this from, unlike phase1/phase2) -- every real test file that
#: exercises the Phase 3 read API / bridge / projections / UI-facing server
#: code, kept only if it exists on disk.
PHASE3_FIXED_SUITE = (
    "tests/test_v2_serving_api.py",
    "tests/test_v2_serving_bridge.py",
    "tests/test_v2_serving_legacy_bundle.py",
    "tests/test_v2_serving_projections.py",
    "tests/test_v2_serving_publication_binding.py",
    "tests/test_v2_dashboard_preview.py",
    "tests/test_v2_dashboard_browser.py",
    "tests/test_v2_dashboard_integration.py",
    "tests/test_checks_phase3_gate.py",
    "tests/test_v2_dashboard_publish.py",
)


def phase3_measure_coverage(root: Path, code_hash: str) -> dict:
    suite = [f for f in PHASE3_FIXED_SUITE if (root / f).is_file()]
    missing = [f for f in PHASE3_FIXED_SUITE if f not in suite]
    with tempfile.TemporaryDirectory(prefix="phase3-cov-") as tmp:
        cov_json = Path(tmp) / "coverage.json"
        cov_data = Path(tmp) / ".coverage"
        run_cmd = [sys.executable, "-m", "coverage", "run", f"--data-file={cov_data}",
                  "--source=engine/v2", "-m", "pytest", "-q", "-p", "no:cacheprovider", *suite]
        result = subprocess.run(run_cmd, cwd=root, capture_output=True, text=True, timeout=1200)
        subprocess.run([sys.executable, "-m", "coverage", "json", f"--data-file={cov_data}",
                        "-o", str(cov_json)], cwd=root, capture_output=True, text=True)
        document = json.loads(cov_json.read_text()) if cov_json.is_file() else {"files": {}}
    packages = package_counts(document, root=root)
    packages_out = {name: {"percentage": group["percentage"], "executed": group["executed"],
                           "executable": group["executable"]} for name, group in packages.items()}
    return {"schema_version": PHASE3_COVERAGE_SCHEMA, "source_hash": code_hash, "suite": suite,
           "suite_missing": missing, "pytest_returncode": result.returncode,
           "pytest_tail": "\n".join(result.stdout.splitlines()[-15:]), "packages": packages_out,
           "baseline_ref": PHASE3_BASELINE.name}


def phase3_coverage_findings(document: dict, root: Path = ROOT) -> list[dict]:
    """Reject a stale, partial, or lower-coverage Phase 3 measurement."""
    findings = []
    baseline_path = root / PHASE3_BASELINE.relative_to(ROOT)
    if not baseline_path.is_file():
        return [{"code": "COVERAGE_BASELINE_MISSING"}]
    try:
        baseline = json.loads(baseline_path.read_text())
    except ValueError:
        return [{"code": "COVERAGE_BASELINE_INVALID"}]
    if baseline.get("schema_version") != PHASE3_BASELINE_SCHEMA:
        return [{"code": "COVERAGE_BASELINE_INVALID"}]
    expected_suite = [f for f in PHASE3_FIXED_SUITE if (root / f).is_file()]
    if baseline.get("suite") != list(PHASE3_FIXED_SUITE):
        return [{"code": "COVERAGE_BASELINE_INVALID"}]
    if document.get("suite") != expected_suite or document.get("suite_missing"):
        findings.append({"code": "COVERAGE_SUITE_DRIFT"})
    if document.get("pytest_returncode") != 0:
        findings.append({"code": "COVERAGE_TEST_FAILURE"})
    packages, previous = document.get("packages"), baseline.get("packages")
    if not isinstance(packages, dict) or set(packages) != set(previous or {}):
        return findings + [{"code": "COVERAGE_PACKAGE_INVENTORY_DRIFT"}]
    for name, prior in previous.items():
        current = packages[name]
        executed, executable = current.get("executed"), current.get("executable")
        old_executed, old_executable = prior.get("executed"), prior.get("executable")
        valid_counts = all(
            type(v) is int for v in (executed, executable, old_executed, old_executable))
        if not valid_counts or not 0 <= executed <= executable or not 0 <= old_executed <= old_executable:
            findings.append({"code": "COVERAGE_COUNTS_INVALID", "package": name})
        elif executable and old_executable and executed * old_executable < old_executed * executable:
            findings.append({"code": "COVERAGE_REGRESSION", "package": name,
                             "previous": [old_executed, old_executable],
                             "current": [executed, executable]})
    return findings


# --------------------------------------------------------------------------
# CLI: python3 checks/v2_coverage_ratchet.py --profile {phase1,phase2} ...
# --------------------------------------------------------------------------

_PROFILES = {
    "phase1": {"measure": phase1_measure, "compare": phase1_compare, "baseline": PHASE1_BASELINE,
              "no_baseline": {"code": "COVERAGE_BASELINE_MISSING"}},
    "phase2": {"measure": phase2_measure, "compare": phase2_compare, "baseline": PHASE2_BASELINE,
              "no_baseline": {"code": "STALE_COVERAGE", "reason": "no baseline"}},
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(_PROFILES), required=True)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--parallel", action="store_true",
                        help="run the fixed suite under -n auto --dist loadgroup "
                             "(requires pytest-xdist); default is serial")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)
    profile = _PROFILES[args.profile]
    baseline_path = args.baseline or profile["baseline"]
    result = profile["measure"](parallel=args.parallel) if args.measure else json.loads(args.input.read_text())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    failures = profile["compare"](result, json.loads(baseline_path.read_text())) if baseline_path.exists() else [
        profile["no_baseline"]]
    print(json.dumps({"ok": not failures, "findings": failures}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
