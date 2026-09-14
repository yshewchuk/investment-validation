"""D15: legacy-vs-snapshot-adapted score parity, over committed catalog state.

``score_parity_receipt`` loads both jobs' committed ``score.json`` (via
``attempt_outputs``/``ArtifactStore``, never the filesystem directly), checks
they target the same resolved session/tickers/horizon, and compares every row
present on both sides field-by-field under the declared Phase 0 field
tolerances (``engine/v2/diagnosis/tolerance.py::SCORE_RECORD_V1`` — empty
rules, i.e. exact equality for every field; there is no phase-2 parity
tolerance declared yet).

``engine.v2.diagnosis`` is a declared *sink* layer (7.5) that no other v2
package may import (``checks/layer_map.py``, enforced by
``checks/import_layers.py`` inside the Phase 1 gate's "imports" row) — a
comparator must never become a dependency of the thing it compares. This
module (layer 7.0, ``engine.v2.ops``) therefore builds the
``comparison_receipt.v1.1`` document BY HAND, as a plain JSON-shaped dict,
rather than importing ``ComparisonReceipt``/``compare_records`` — the same
choice every other real receipt producer in this tree already makes
(``checks/rearchitecture_phase1_canary.py``, ``checks/tier0_corpus.py``: both
live in ``checks/``, which is not a layered v2 package and may import the
sink freely). The returned document strictly decodes as ``ComparisonReceipt``
(round-tripped by ``checks/rearchitecture_phase2_parity.py`` and by this
module's own tests) — recorded as this task's judgement call.
"""
from __future__ import annotations

import json
import math

from engine.v2.contracts import JobSpec
from engine.v2.contracts.data import SnapshotRef
from engine.v2.data.documents import decode_document
from engine.v2.foundation import content_hash
from engine.v2.ops.catalog import load_json
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.errors import fail
from engine.v2.ops.input_bindings import recorded_bindings

__all__ = ["score_parity_receipt"]

#: ``compare_records``'s own default ``comparison_kind`` (§P2-C01 decision 3):
#: the D15 evidence field expects exactly this string.
SCORE_PARITY_KIND = "score_record_parity"
#: The real ``StagePlan``/``TolerancePolicy`` identifiers this document
#: nominally refers to (``engine/v2/diagnosis/stage_plan.py::SCORER_V1``,
#: ``.../tolerance.py::SCORE_RECORD_V1``), copied as literal strings since
#: this module may not import diagnosis to fetch them.
_STAGE_PLAN_REF = "scorer.v1"
_TOLERANCE_POLICY_REF = "score_record.exact.v1"
#: Every finding is reported here: this comparator does not run the real
#: stage-localization graph (that machinery lives in diagnosis), so claiming
#: a specific stage per field would be a fabricated localization, not an
#: observed one.
_UNASSIGNED_STAGE = "unassigned"
_OUTPUT_NAME = "legacy_score"
_ABSENT = object()

_AGREE, _DIFFER, _INCOMPARABLE = "agree", "differ", "incomparable"


def _succeeded_attempt(conn, job_id: str) -> str:
    row = conn.execute(
        "SELECT attempt_id FROM attempts WHERE job_id=? AND state='succeeded' "
        "ORDER BY attempt_number DESC LIMIT 1", (job_id,)).fetchone()
    if row is None:
        raise fail("INPUT_CHANGED", "job has no succeeded attempt", details={"job_id": job_id})
    return row[0]


def _committed_score(conn, store, job_id: str) -> tuple[dict, str]:
    attempt_id = _succeeded_attempt(conn, job_id)
    row = conn.execute("SELECT artifact_id FROM attempt_outputs WHERE attempt_id=? AND name=?",
                       (attempt_id, _OUTPUT_NAME)).fetchone()
    if row is None:
        raise fail("INPUT_CHANGED", "job did not commit a score artifact", details={"job_id": job_id})
    ref = artifact(conn, store, row[0])
    return json.loads(store.read_verified(ref)), attempt_id


def _job_parameters(conn, job_id: str) -> dict:
    row = conn.execute("SELECT spec_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        raise fail("INPUT_CHANGED", "unknown job", details={"job_id": job_id})
    return load_json(JobSpec, row[0]).parameters


def _snapshot_binding(conn, store, attempt_id: str) -> SnapshotRef | None:
    """The ``snapshot_ref.json`` bound at launch, if this attempt is snapshot-backed."""
    row = recorded_bindings(conn, attempt_id).get("snapshot_ref.json")
    if row is None:
        return None
    ref = artifact(conn, store, row.artifact_id)
    return decode_document(SnapshotRef, json.loads(store.read_verified(ref)))


def _rows_by_id(doc: dict) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in doc.get("rows", []):
        row_id = row.get("row_id")
        if not row_id or row_id in rows:
            raise fail("VALIDATION_FAILED", "score document has a missing or duplicate row_id")
        rows[row_id] = row
    return rows


def _verify_same_target(legacy_doc, snapshot_doc, legacy_params, snapshot_params) -> None:
    legacy_target = (legacy_doc.get("session"), tuple(sorted(legacy_doc.get("tickers") or ())),
                     legacy_params.get("horizon_days", 35))
    snapshot_target = (snapshot_doc.get("session"), tuple(sorted(snapshot_doc.get("tickers") or ())),
                       snapshot_params.get("horizon_days", 35))
    if legacy_target != snapshot_target:
        raise fail("INPUT_CHANGED",
                   "legacy and snapshot score jobs do not target the same resolved "
                   "session, tickers and horizon",
                   details={"legacy": list(legacy_target), "snapshot": list(snapshot_target)})


# --------------------------------------------------------------------------
# field-level comparison — exact, per SCORE_RECORD_V1's (empty) declared rules
# --------------------------------------------------------------------------


def _escape(key: str) -> str:
    return key.replace("\\", "\\\\").replace(".", "\\.")


def _flatten(value, prefix: str = "") -> dict:
    """Typed-path flatten, mirroring ``diagnosis.record_comparator.flatten``
    (duplicated here rather than imported: see module docstring)."""
    if isinstance(value, dict):
        if not value:
            return {prefix: {}} if prefix else {}
        out: dict = {}
        for key, sub in value.items():
            path = f"{prefix}.{_escape(str(key))}" if prefix else _escape(str(key))
            out.update(_flatten(sub, path))
        return out
    if isinstance(value, (list, tuple)):
        if not value:
            return {prefix: []} if prefix else {}
        out = {}
        for index, item in enumerate(value):
            out.update(_flatten(item, f"{prefix}[{index}]"))
        return out
    return {prefix: value}


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return _is_int(value) or isinstance(value, float)


def _numeric_kind(left, right) -> str | None:
    """Both sides are ``int``/``float`` (excluding ``bool``): exact
    everywhere, since ``SCORE_RECORD_V1`` declares no per-field tolerance —
    a float pair is compared for bit equality (NaN matches NaN)."""
    if _is_int(left) != _is_int(right):
        return "type"  # 2 vs 2.0: same value, different quantity type
    if isinstance(left, float) and math.isnan(left) and math.isnan(right):
        return None
    return None if left == right else "value"


def _kind(left, right) -> str | None:
    """``None`` = agree."""
    if left is _ABSENT or right is _ABSENT:
        return "missing_field"
    if (left is None) != (right is None):
        return "null_mask"
    if left is None:
        return None
    if _is_number(left) and _is_number(right):
        return _numeric_kind(left, right)
    if type(left) is not type(right):
        return "type"
    return None if left == right else "value"


def _finding_id(*parts) -> str:
    return content_hash(list(parts))[7:19]


def _receipt_id(*parts) -> str:
    return content_hash(list(parts))[7:23]


def _row_findings(row_id: str, left_row: dict, right_row: dict) -> list[dict]:
    flat_left, flat_right = _flatten(left_row), _flatten(right_row)
    findings = []
    for path in sorted(set(flat_left) | set(flat_right)):
        left, right = flat_left.get(path, _ABSENT), flat_right.get(path, _ABSENT)
        kind = _kind(left, right)
        if kind is None:
            continue
        findings.append({
            "finding_id": _finding_id("score_row_parity", row_id, path, kind),
            "first_differing_stage": _UNASSIGNED_STAGE, "field_path": f"{row_id}::{path}",
            "tolerance_applied": "exact",
            "null_mask_left": left is None or left is _ABSENT,
            "null_mask_right": right is None or right is _ABSENT,
            "kind": kind, "owning_stage": _UNASSIGNED_STAGE, "affected_count": 1,
        })
    return findings


def _missing_row_finding(row_id: str, *, in_legacy: bool) -> dict:
    return {
        "finding_id": _finding_id("score_row_parity", row_id, "missing_field"),
        "first_differing_stage": _UNASSIGNED_STAGE, "field_path": row_id,
        "tolerance_applied": "exact", "null_mask_left": not in_legacy,
        "null_mask_right": in_legacy, "kind": "missing_field",
        "owning_stage": _UNASSIGNED_STAGE, "affected_count": 1,
    }


def _verdict(legacy_keys: set, snapshot_keys: set, findings: list, population: dict) -> str:
    if population["expected"] <= 0 or population["supported"] <= 0 or population["compared"] <= 0:
        return _INCOMPARABLE
    if legacy_keys != snapshot_keys or findings:
        return _DIFFER
    return _AGREE


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------


def score_parity_receipt(conn, store, *, legacy_job_id: str, snapshot_job_id: str,
                         code_hash: str, environment_hash: str) -> dict:
    """A ``comparison_receipt.v1.1``-shaped document comparing two committed
    ``legacy_score`` jobs: ``legacy_job_id`` normally run ``--input-mode
    legacy`` (the barrier path) and ``snapshot_job_id`` run ``--input-mode
    snapshot`` (the adapted path) — this function does not itself check
    ``input_mode``, only that both jobs committed a ``score.json`` targeting
    the same resolved session/tickers/horizon.

    Findings name only ``<row_id>::<field_path>`` and a ``kind`` — never a
    value. A row missing on either side is one ``missing_field`` finding
    (a disagreement, not a population exclusion); ``expected`` counts only
    legacy rows, ``supported``/``compared`` count rows present on both sides.
    Raises (``INPUT_CHANGED``/``VALIDATION_FAILED``) rather than returning a
    receipt when either job has no committed score, or the two do not target
    the same resolved session/tickers/horizon.
    """
    legacy_doc, _ = _committed_score(conn, store, legacy_job_id)
    snapshot_doc, snapshot_attempt_id = _committed_score(conn, store, snapshot_job_id)
    _verify_same_target(legacy_doc, snapshot_doc, _job_parameters(conn, legacy_job_id),
                        _job_parameters(conn, snapshot_job_id))
    snapshot_ref = _snapshot_binding(conn, store, snapshot_attempt_id)

    legacy_rows, snapshot_rows = _rows_by_id(legacy_doc), _rows_by_id(snapshot_doc)
    legacy_keys, snapshot_keys = set(legacy_rows), set(snapshot_rows)
    supported_keys = legacy_keys & snapshot_keys

    findings = [_missing_row_finding(row_id, in_legacy=row_id in legacy_keys)
               for row_id in sorted(legacy_keys ^ snapshot_keys)]
    for row_id in sorted(supported_keys):
        findings.extend(_row_findings(row_id, legacy_rows[row_id], snapshot_rows[row_id]))

    population = {"expected": len(legacy_keys), "supported": len(supported_keys),
                 "compared": len(supported_keys), "skipped_with_reasons": {}, "excluded": []}
    envelope = {"code_hash": code_hash, "environment_hash": environment_hash}
    if snapshot_ref is not None:
        envelope["snapshot_id"] = snapshot_ref.snapshot_id
        envelope["snapshot_manifest_hash"] = snapshot_ref.manifest_hash
    receipt_id = _receipt_id("score_record_parity", legacy_job_id, snapshot_job_id,
                             [f["finding_id"] for f in findings])
    return {
        "schema_version": "comparison_receipt.v1.1", "receipt_id": receipt_id,
        "comparison_kind": SCORE_PARITY_KIND, "tier": 2, "left_ref": legacy_job_id,
        "right_ref": snapshot_job_id, "stage_plan_ref": _STAGE_PLAN_REF,
        "tolerance_policy_ref": _TOLERANCE_POLICY_REF,
        "verdict": _verdict(legacy_keys, snapshot_keys, findings, population),
        "findings": findings, "population": population, "envelope": envelope,
    }
