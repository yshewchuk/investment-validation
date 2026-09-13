#!/usr/bin/env python3
"""Phase 2 acceptance gate: fresh coverage plus evidence for D01-D20.

Phase-2 guide §12.1: "checks/rearchitecture_phase2_gate.py -- validates fresh
evidence for D01-D20 and reruns Phase 0/1 prerequisites." This script never
writes repository, import or ops behavior and never produces evidence itself
-- it only validates a coverage measurement and (optionally) a private
Phase2Evidence document someone else produced.

Prerequisites are Phase 0 and Phase 1 as a WHOLE gate each, except Phase 1's
own coverage row: that ratchet is tracked separately (this gate has its own,
stricter one over the Phase 2 suite), and today's structural/engineering rows
can be green while the Phase 1 baseline is mid-update by someone else.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase1_gate import source_files, source_hash
from checks.rearchitecture_phase2_coverage import (
    BASELINE as COVERAGE_BASELINE,
    REGISTRY,
    compare as coverage_compare,
    load_registry,
    validate_measurement as coverage_validate,
)
from checks.rearchitecture_phase2_evidence import authority_mode_ok, validate_evidence

PHASE0_SCRIPT = "checks/rearchitecture_phase0_gate.py"
PHASE1_COVERAGE_SCRIPT = "checks/rearchitecture_phase1_coverage.py"
PHASE1_GATE_SCRIPT = "checks/rearchitecture_phase1_gate.py"


def environment_hash(root=ROOT):
    """Reuse the ops environment fingerprint if importable side-effect-free;
    else hash the interpreter version plus requirements.txt."""
    try:
        from engine.v2.ops.fingerprints import environment_identity
        payload, source = environment_identity(), "engine.v2.ops.fingerprints.environment_identity"
    except Exception:
        lock = root / "requirements.txt"
        payload = {"python": platform.python_version(),
                   "requirements": lock.read_text() if lock.is_file() else None}
        source = "python_version+requirements.txt"
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    return "sha256:" + digest, source


def _run_json(argv, root):
    try:
        result = subprocess.run(argv, cwd=root, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        return json.loads(result.stdout)
    except ValueError:
        return None


def default_prerequisite_runner(root=ROOT):
    """Real subprocess execution of the Phase 0 and Phase 1 gates.

    Phase 1 needs a fresh coverage measurement to have any chance of a green
    ``ok``; that measurement is produced here and discarded, never committed
    -- Phase 1's own baseline belongs to whoever owns that script.
    """
    phase0_raw = _run_json([sys.executable, PHASE0_SCRIPT, "--json"], root)
    with tempfile.TemporaryDirectory(prefix="phase2-gate-prereq-") as scratch:
        coverage_path = Path(scratch) / "phase1-coverage.json"
        subprocess.run([sys.executable, PHASE1_COVERAGE_SCRIPT, "--measure",
                        "--output", str(coverage_path)],
                       cwd=root, capture_output=True, check=False, timeout=3600)
        phase1_raw = _run_json([sys.executable, PHASE1_GATE_SCRIPT,
                                "--coverage", str(coverage_path)], root)
    return {"phase0": {"ok": bool(phase0_raw and phase0_raw.get("ok")), "raw": phase0_raw},
            "phase1": {"ok": bool(phase1_raw and phase1_raw.get("ok")), "raw": phase1_raw}}


def _phase1_ok_excluding_coverage(raw):
    """Phase 1's structural + engineering rows, minus its coverage row.

    Coverage is a separately tracked, separately owned ratchet; folding it in
    here would make this gate red for a reason this task does not own.
    """
    if not isinstance(raw, dict):
        return False
    rows = {**raw.get("structural", {}), **raw.get("engineering", {})}
    rows.pop("coverage", None)
    return bool(rows) and all(isinstance(r, dict) and r.get("ok") for r in rows.values())


def coverage_check(path, baseline_path, root, registry=None):
    if path is None or not Path(path).is_file() or not baseline_path.is_file():
        return {"ok": False, "findings": [{"code": "STALE_COVERAGE", "reason": "no measurement"}],
                "test_outcomes": []}
    try:
        measured = json.loads(Path(path).read_text())
        baseline = json.loads(baseline_path.read_text())
    except (ValueError, OSError):
        return {"ok": False, "findings": [{"code": "STALE_COVERAGE", "reason": "invalid json"}],
                "test_outcomes": []}
    findings = coverage_compare(measured, baseline)
    findings.extend(coverage_validate(measured, root, registry))
    return {"ok": not findings, "findings": findings, "test_outcomes": measured.get("test_outcomes", [])}


def _prefix_status(prefix, test_outcomes):
    matches = [o for o in test_outcomes if o.get("nodeid", "").startswith(prefix)]
    if any(m.get("outcome") == "failed" for m in matches):
        return "failed"
    if any(m.get("outcome") == "passed" for m in matches):
        return "passed"
    return "missing"


def _check_row(d_id, row, test_outcomes, evidence, evidence_valid, phase1_raw):
    findings = []
    statuses = [_prefix_status(p, test_outcomes) for p in row.get("tests", [])]
    if "failed" in statuses:
        findings.append({"code": "TEST_FAILED", "d_id": d_id})
    if "missing" in statuses:
        findings.append({"code": "MISSING_EVIDENCE", "d_id": d_id})
    if row.get("reuse_phase1_structural_engineering") and not _phase1_ok_excluding_coverage(phase1_raw):
        findings.append({"code": "MISSING_EVIDENCE", "d_id": d_id, "reason": "phase1_structural_engineering"})
    if row.get("tier") == 2:
        if not evidence_valid:
            findings.append({"code": "MISSING_EVIDENCE", "d_id": d_id, "reason": "evidence_manifest"})
        else:
            for field in row.get("evidence_fields", []):
                if evidence.get(field) is None:
                    findings.append({"code": "MISSING_EVIDENCE", "d_id": d_id, "reason": field})
    return findings


def gate(root=ROOT, *, coverage_path=None, evidence_manifest_path=None,
         artifact_root=None, prerequisite_runner=None, registry_path=REGISTRY):
    started = time.monotonic()
    registry = load_registry(registry_path)
    files = source_files(root)
    code_hash = source_hash(files)
    env_hash, env_source = environment_hash(root)
    findings = []

    runner = prerequisite_runner or default_prerequisite_runner
    prereqs = runner(root)
    if not prereqs.get("phase0", {}).get("ok"):
        findings.append({"code": "PREREQUISITE_FAILED", "prerequisite": "phase0"})
    phase1_raw = prereqs.get("phase1", {}).get("raw")
    if not _phase1_ok_excluding_coverage(phase1_raw):
        findings.append({"code": "PREREQUISITE_FAILED", "prerequisite": "phase1"})

    coverage_baseline = root / COVERAGE_BASELINE.relative_to(ROOT)
    coverage = coverage_check(coverage_path, coverage_baseline, root, registry)
    findings.extend(coverage["findings"])
    test_outcomes = coverage["test_outcomes"]

    evidence, evidence_valid = None, False
    if evidence_manifest_path is not None and Path(evidence_manifest_path).is_file():
        try:
            evidence = json.loads(Path(evidence_manifest_path).read_text())
        except (ValueError, OSError):
            evidence = None
        if evidence is not None:
            resolved_root = Path(artifact_root) if artifact_root else root
            evidence_findings = validate_evidence(evidence, artifact_root=resolved_root,
                                                   code_hash=code_hash, environment_hash=env_hash)
            findings.extend(evidence_findings)
            evidence_valid = not evidence_findings and authority_mode_ok(evidence)

    for d_id in sorted(registry):
        findings.extend(_check_row(d_id, registry[d_id], test_outcomes, evidence,
                                   evidence_valid, phase1_raw))

    return {"schema_version": "phase2_gate.v1.0", "ok": not findings, "code_hash": code_hash,
            "environment_hash": env_hash, "environment_hash_source": env_source,
            "findings": findings, "seconds": round(time.monotonic() - started, 2)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = gate(coverage_path=args.coverage, evidence_manifest_path=args.evidence_manifest,
                  artifact_root=args.artifact_root)
    encoded = json.dumps(result, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if args.json:
        print(encoded)
    else:
        print(f"phase 2 gate: {'GREEN' if result['ok'] else 'RED'} "
              f"({len(result['findings'])} findings) in {result['seconds']:.1f}s")
        for finding in result["findings"]:
            print(f"  {finding['code']}: " +
                  ", ".join(f"{k}={v}" for k, v in finding.items() if k != "code"))
    return int(not result["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
