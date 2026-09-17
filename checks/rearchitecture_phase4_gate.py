#!/usr/bin/env python3
"""Strict gate for the Phase 4 contract/application foundation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

__all__ = ["check", "main"]

REQUIRED = ("P4-01", "P4-02", "P4-03", "P4-04", "P4-05", "P4-06", "P4-07", "P4-08", "P4-09")


def check(evidence: dict) -> dict:
    findings = []
    if evidence.get("schema_version") != "phase4_acceptance.v1.0":
        findings.append("unsupported evidence schema")
    population = evidence.get("population") or {}
    if not population or not (population.get("expected") == population.get("supported") == population.get("compared")):
        findings.append("population is incomplete")
    if evidence.get("status") not in {"FOUNDATION_PASS", "PASS"}:
        findings.append("phase 4 status is not accepted")
    if evidence.get("phase5_inference_integrated") and evidence.get("evidence_scope") != "native_full_release":
        findings.append("Phase 5 integration lacks native full-release evidence")
    subjects = evidence.get("subjects") or {}
    for subject in REQUIRED:
        row = subjects.get(subject)
        if not row:
            findings.append(f"missing subject {subject}")
            continue
        if row.get("status") not in {"PASS", "FOUNDATION_PASS"}:
            findings.append(f"subject {subject} failed")
        for name, value in (row.get("controls") or {}).items():
            if value is not True:
                findings.append(f"{subject}.{name} is not proven")
    return {"status": "PASS" if not findings else "FAIL", "ok": not findings,
            "findings": findings, "subjects": len(subjects),
            "phase5_inference_integrated": bool(evidence.get("phase5_inference_integrated"))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(json.loads(Path(args.evidence).read_text()))
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else result["status"])
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
