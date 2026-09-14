#!/usr/bin/env python3
"""D15 CLI: legacy-vs-snapshot score parity, over an existing ops catalog.

Reads two already-committed ``legacy_score`` jobs from ``--root`` (the same
``data/operations``-shaped directory ``engine/v2/ops/cli.py`` uses: a
``catalog.sqlite`` plus a content-addressed object store at its side) via
``engine.v2.ops.parity.load_score_parity_inputs`` (loading only — that module
may not import ``engine.v2.diagnosis``, a declared sink layer), then runs the
real ``compare_records`` here, under its declared tolerance, to build a real
``ComparisonReceipt`` (v1.1). Writes it under ``--artifact-root``, ready to be
handed to ``checks/rearchitecture_phase2_evidence_build.py --score-receipt``.

Makes no provider calls and mutates nothing: strictly read-only over the
catalog and the store.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase1_gate import source_files, source_hash  # noqa: E402
from checks.rearchitecture_phase2_gate import environment_hash as _environment_hash  # noqa: E402
from engine.v2.diagnosis import (  # noqa: E402
    AGREE,
    DIFFER,
    INCOMPARABLE,
    SCORE_RECORD_V1,
    SCORER_V1,
    ComparisonReceipt,
    Envelope,
    Finding,
    Population,
    compare_records,
    content_hash,
)
from engine.v2.diagnosis.stage_plan import UNASSIGNED  # noqa: E402
from engine.v2.foundation import ArtifactStore, SystemClock, to_document  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402
from engine.v2.ops.parity import ScoreParityInputs, load_score_parity_inputs  # noqa: E402

#: ``compare_records``'s own default ``comparison_kind`` (task P2-C01 decision
#: 3) — the D15 evidence field (``checks/rearchitecture_phase2_evidence.py``'s
#: ``SCORE_PARITY_KIND``) expects exactly this string.
SCORE_PARITY_KIND = "score_record_parity"


def _redact(finding: Finding, row_id: str) -> Finding:
    """Findings name only ``<row_id>::<field_path>`` and a ``kind`` — never a
    value (task brief). ``not_downstream_of`` linked finding ids within one
    row's own receipt; once merged across rows those links are stale, so they
    are dropped rather than carried forward incorrectly."""
    return dataclasses.replace(finding, field_path=f"{row_id}::{finding.field_path}",
                               left_value=None, right_value=None, delta=None,
                               exceeded_by=None, not_downstream_of=())


def _row_findings(row_id: str, left_row: dict, right_row: dict) -> list[Finding]:
    receipt = compare_records(left_row, right_row, comparison_kind=SCORE_PARITY_KIND, tier=2,
                              left_ref="legacy", right_ref="snapshot",
                              tolerance_policy=SCORE_RECORD_V1)
    return [_redact(f, row_id) for f in receipt.findings]


def _missing_row_finding(row_id: str, *, in_legacy: bool) -> Finding:
    return Finding(finding_id=content_hash(["score_row_parity", row_id, "missing_field"])[7:19],
                   first_differing_stage=UNASSIGNED, field_path=row_id, kind="missing_field",
                   null_mask_left=not in_legacy, null_mask_right=in_legacy, owning_stage=UNASSIGNED)


def _verdict(legacy_keys: set, snapshot_keys: set, findings: list, population: Population) -> str:
    if population.expected <= 0 or population.supported <= 0 or population.compared <= 0:
        return INCOMPARABLE
    if legacy_keys != snapshot_keys or findings:
        return DIFFER
    return AGREE


def build_receipt(inputs: ScoreParityInputs, *, code_hash: str,
                  environment_hash: str) -> ComparisonReceipt:
    """Populations: ``expected`` = legacy rows; ``supported``/``compared`` =
    rows present on both sides (a row missing on either side is a
    disagreement, one ``missing_field`` finding, never a population
    exclusion). Verdict ``agree`` only when both key sets are identical,
    every field is within tolerance, and all populations are positive."""
    legacy_keys = set(inputs.legacy_rows)
    snapshot_keys = set(inputs.snapshot_rows)
    supported_keys = legacy_keys & snapshot_keys

    findings = [_missing_row_finding(row_id, in_legacy=row_id in legacy_keys)
               for row_id in sorted(legacy_keys ^ snapshot_keys)]
    for row_id in sorted(supported_keys):
        findings.extend(_row_findings(row_id, inputs.legacy_rows[row_id], inputs.snapshot_rows[row_id]))

    population = Population(expected=len(legacy_keys), supported=len(supported_keys),
                            compared=len(supported_keys))
    envelope = Envelope(code_hash=code_hash, environment_hash=environment_hash)
    if inputs.snapshot_ref is not None:
        envelope = dataclasses.replace(envelope, snapshot_id=inputs.snapshot_ref.snapshot_id,
                                       snapshot_manifest_hash=inputs.snapshot_ref.manifest_hash)
    receipt_id = content_hash(["score_record_parity", inputs.legacy_job_id, inputs.snapshot_job_id,
                               [f.finding_id for f in findings]])[7:23]
    return ComparisonReceipt(
        receipt_id=receipt_id, comparison_kind=SCORE_PARITY_KIND, tier=2,
        left_ref=inputs.legacy_job_id, right_ref=inputs.snapshot_job_id,
        stage_plan_ref=SCORER_V1.plan_id, tolerance_policy_ref=SCORE_RECORD_V1.policy_id,
        verdict=_verdict(legacy_keys, snapshot_keys, findings, population),
        findings=tuple(findings), population=population, envelope=envelope)


def build(root: Path, *, legacy_job: str, snapshot_job: str) -> ComparisonReceipt:
    """``code_hash``/``environment_hash`` are computed the SAME way
    ``rearchitecture_phase2_gate.py`` computes them, over THIS repo checkout
    (not the operations root) — "both Phase 2 inputs record the same code
    hash" (phase-2 guide §12.1)."""
    code_hash = source_hash(source_files(ROOT))
    environment_hash, _source = _environment_hash(ROOT)
    conn = open_catalog(root / "catalog.sqlite", clock=SystemClock())
    try:
        store = ArtifactStore(root)
        inputs = load_score_parity_inputs(conn, store, legacy_job_id=legacy_job,
                                          snapshot_job_id=snapshot_job)
    finally:
        conn.close()
    return build_receipt(inputs, code_hash=code_hash, environment_hash=environment_hash)


def publish(receipt: ComparisonReceipt, artifact_root: Path) -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    data = json.dumps(to_document(receipt), indent=2, sort_keys=True).encode()
    path = artifact_root / f"score_parity_{receipt.receipt_id}.json"
    path.write_bytes(data)
    return {"path": path.name, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--legacy-job", required=True)
    parser.add_argument("--snapshot-job", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = build(args.root, legacy_job=args.legacy_job, snapshot_job=args.snapshot_job)
    except OpsError as exc:
        print(json.dumps({"refused": exc.code, "message": str(exc)}, indent=2))
        return 1
    ref = publish(receipt, args.artifact_root)
    print(json.dumps({**ref, "verdict": receipt.verdict}, indent=2))
    return 0 if receipt.verdict == AGREE else 1


if __name__ == "__main__":
    raise SystemExit(main())
