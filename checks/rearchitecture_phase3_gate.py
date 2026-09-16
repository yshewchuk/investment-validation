#!/usr/bin/env python3
"""Phase 3 launch gate: L01-L14 evidence for the first real-score dashboard.

Phase-3 guide §10: "Implement a strict private gate input" (``Phase3Evidence``,
validated by ``checks/rearchitecture_phase3_evidence.py``) and "Required new
gate command after evidence producers finish":

    /usr/bin/python3 -u checks/rearchitecture_phase3_gate.py \\
        --evidence-manifest /tmp/phase3-evidence.json --write-report

This script never writes repository, serving or ops behavior and never
produces evidence itself -- it only validates a (optionally absent) private
``Phase3Evidence`` document someone else produced, against the L01-L14
registry (``checks/phase3_acceptance.json``). Bare (no ``--evidence-manifest``,
or a manifest whose refs do not yet exist) is RED by design: every producer
this guide's tasks (P3-0..P3-4) have not yet finished shows up as its own
``MISSING_EVIDENCE`` finding, never a silent pass.

Prerequisite is Phase 2 as a WHOLE evidence document (guide §10: "the Phase 3
gate requires Phase 2 evidence validated against its producer commit"),
reusing ``rearchitecture_phase2_evidence.validate_evidence`` directly --
never reimplemented, never re-run as a live subprocess gate the way Phase 2
reruns Phase 0/1: Phase 2's OWN evidence document is what is validated here,
bound to ITS OWN declared ``code_hash``/``environment_hash`` (its producer
commit), not to this tree's current state. An invalid Phase 2 evidence
document collapses to one ``PREREQUISITE_FAILED`` finding
(``checks/rearchitecture_phase3_evidence.py::_check_phase2``).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase1_gate import source_files, source_hash  # noqa: E402
from checks.rearchitecture_phase2_gate import environment_hash  # noqa: E402
from checks.rearchitecture_phase3_evidence import validate_evidence  # noqa: E402

REGISTRY = ROOT / "checks/phase3_acceptance.json"


def load_registry(path=REGISTRY):
    return json.loads(Path(path).read_text())["rows"]


class ReportPathError(ValueError):
    """Raised when ``--write-report`` is asked to write inside the repo."""

    code = "REPORT_PATH_INSIDE_REPO"


def _row_findings(l_id: str, row: dict, field_ok: dict, kind_ok: dict) -> list[dict]:
    findings = []
    for requirement in row.get("requires", []):
        if "field" in requirement:
            name = requirement["field"]
            if not field_ok.get(name):
                findings.append({"code": "MISSING_EVIDENCE", "l_id": l_id, "reason": name})
        else:
            key = (requirement["list"], requirement["kind"])
            if not kind_ok.get(key):
                findings.append({"code": "MISSING_EVIDENCE", "l_id": l_id,
                                 "reason": f"{key[0]}:{key[1]}"})
    return findings


def gate(root=ROOT, *, evidence_manifest_path=None, artifact_root=None, registry_path=REGISTRY,
         corpus_root=None):
    started = time.monotonic()
    registry = load_registry(registry_path)
    files = source_files(root)
    code_hash = source_hash(files)
    env_hash, env_source = environment_hash(root)
    findings: list[dict] = []

    evidence = None
    field_ok: dict[str, bool] = {}
    kind_ok: dict = {}
    if evidence_manifest_path is not None and Path(evidence_manifest_path).is_file():
        try:
            evidence = json.loads(Path(evidence_manifest_path).read_text())
        except (ValueError, OSError):
            evidence = None
        if evidence is not None:
            resolved_root = Path(artifact_root) if artifact_root else root
            evidence_findings, field_ok, _document_ok, kind_ok = validate_evidence(
                evidence, artifact_root=resolved_root, corpus_root=corpus_root,
                implementation_code_hash=code_hash, environment_hash=env_hash)
            findings.extend(evidence_findings)
    if evidence is None:
        findings.append({"code": "MISSING_EVIDENCE", "reason": "no evidence manifest provided"})

    l_rows: dict[str, dict] = {}
    for l_id in sorted(registry):
        row_findings = _row_findings(l_id, registry[l_id], field_ok, kind_ok)
        findings.extend(row_findings)
        l_rows[l_id] = {"ok": not row_findings, "requires": registry[l_id].get("requires", [])}

    retained = [f for f in findings if f.get("code") == "PHASE2_STRICT_FINDINGS_RETAINED"]
    readiness_findings = [f for f in findings if f.get("code") != "PHASE2_STRICT_FINDINGS_RETAINED"]
    return {"schema_version": "phase3_gate.v1.0", "ok": not findings,
            "accepted_readiness_ok": not readiness_findings,
            "retained_prerequisite_findings": retained,
            "implementation_code_hash": code_hash, "environment_hash": env_hash,
            "environment_hash_source": env_source, "findings": findings, "l_rows": l_rows,
            "evidence": evidence, "seconds": round(time.monotonic() - started, 2)}


def write_report(root: Path, report_path: Path, result: dict) -> Path:
    """Render the private acceptance report. Refuses a path inside ``root``.

    Never writes to the repository -- guide §10: "The gate generates a
    private REPORT.md ... into a caller-given private path, never into the
    repo." Provenance, the L01-L14 matrix, limitations (every finding), and
    evidence refs/deferred owners (read straight from the evidence document
    already in ``result``, never re-derived).
    """
    resolved = report_path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        pass
    else:
        raise ReportPathError(f"refusing to write a report inside the repo: {resolved}")

    lines = ["# Phase 3 launch acceptance report", "",
             f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "",
             "## Provenance", "",
             f"- implementation_code_hash: `{result['implementation_code_hash']}`",
             f"- environment_hash: `{result['environment_hash']}` (source: "
             f"{result['environment_hash_source']})",
             f"- strict gate ok: **{result['ok']}**",
             f"- accepted readiness ok: **{result.get('accepted_readiness_ok', False)}**", "",
             "## L01-L14 matrix", "", "| L-ID | ok |", "|---|---|"]
    for l_id, row in sorted(result.get("l_rows", {}).items()):
        lines.append(f"| {l_id} | {'PASS' if row['ok'] else 'FAIL'} |")
    lines += ["", "## Limitations / findings", ""]
    if result["findings"]:
        for finding in result["findings"]:
            lines.append("- " + ", ".join(f"{k}={v}" for k, v in finding.items()))
    else:
        lines.append("- none")
    evidence = result.get("evidence") or {}
    lines += ["", "## Evidence refs", ""]
    for field in ("phase2_acceptance_ref", "population_manifest_ref", "browser_receipt_ref",
                 "refresh_rollback_receipt_ref", "engineering_receipt_ref", "coverage_receipt_ref",
                 "performance_receipt_ref", "view_field_inventory_ref", "deferred_work_ref"):
        ref = evidence.get(field)
        if isinstance(ref, dict) and "path" in ref:
            lines.append(f"- {field}: `{ref['path']}` ({ref.get('content_hash', '')})")
    lines += ["", "## Deferred owners", "",
             "See `deferred_work_ref` above; this report does not re-decode its content -- "
             "read the artifact directly for the itemized owner list."]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--report-path", type=Path, default=None,
                        help="private path for REPORT.md; must not resolve inside the repo. "
                             "Defaults to a path under the system temp directory.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = gate(evidence_manifest_path=args.evidence_manifest, artifact_root=args.artifact_root,
                  registry_path=args.registry)
    encoded_result = {k: v for k, v in result.items() if k != "evidence"}
    encoded = json.dumps(encoded_result, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if args.write_report:
        report_path = args.report_path or (Path(tempfile.gettempdir()) / "phase3-report" / "REPORT.md")
        try:
            written = write_report(ROOT, report_path, result)
            print(f"report: {written}", file=sys.stderr)
        except ReportPathError as exc:
            print(f"refusing to write report: {exc}", file=sys.stderr)
            return 2
    if args.json:
        print(encoded)
    else:
        print(f"phase 3 gate: {'GREEN' if result['ok'] else 'RED'} "
              f"({len(result['findings'])} findings) in {result['seconds']:.1f}s")
        for finding in result["findings"]:
            print(f"  {finding['code']}: " +
                  ", ".join(f"{k}={v}" for k, v in finding.items() if k != "code"))
    return int(not result["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
