#!/usr/bin/env python3
"""Strict Phase 3B incremental-data acceptance registry and gate.

The gate consumes evidence produced by an incremental refresh run.  It does
not infer success from whichever tests happen to exist and it does not skip a
subject when the corresponding implementation is absent.  Missing evidence,
unresolved artifacts, partial coverage, and missing negative controls are
failures.

``--deterministic-fixture`` exercises this acceptance harness with small,
content-hashed fixtures.  A green fixture result proves the registry and gate
are operable; its ``status`` is ``FIXTURE_PASS`` and therefore does not claim
that a production Phase 3B refresh has passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase1_gate import source_files, source_hash  # noqa: E402
from checks.rearchitecture_phase2_gate import environment_hash  # noqa: E402
from engine.v2.contracts import (  # noqa: E402
    AcquisitionReceipt,
    ChangeSet,
    ColumnContract,
    CompletedCoverage,
    CoverageKey,
    CoverageOutcome,
    DatasetVersionRef,
    DependencyImpact,
    ObjectRef,
    RevisionCandidate,
    RowChange,
    SnapshotRef,
    TableContract,
    TableContractRef,
    TimeInterval,
)
from engine.v2.data import incremental as data_incremental  # noqa: E402
from engine.v2.data.eod_inventory import inventory_document  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.event_revisions import resolve_event_identity  # noqa: E402
from engine.v2.foundation import (  # noqa: E402
    DocumentError,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.ops import incremental_data  # noqa: E402

EVIDENCE_SCHEMA = "phase3b_evidence.v1.0"
RESULT_SCHEMA = "phase3b_acceptance.v1.0"
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class SubjectSpec:
    subject_id: str
    name: str
    minimum_counts: tuple[tuple[str, int], ...]
    zero_counts: tuple[str, ...]
    negative_controls: tuple[str, ...]
    requires_changed_partitions: bool = False
    requires_complete_coverage: bool = False


SUBJECTS: tuple[SubjectSpec, ...] = (
    SubjectSpec(
        "P3B01", "inventory_contracts",
        (("inventory_rows", 5), ("contract_roundtrips", 8), ("schema_refusals", 3)),
        (), ("unknown_field_refused", "bad_enum_refused", "missing_field_refused")),
    SubjectSpec(
        "P3B02", "coverage_receipt_failures",
        (("denominator_keys", 2), ("completed_keys", 2), ("rejected_incomplete", 3)),
        ("watermark_advances_on_failure",),
        ("partial_refused", "auth_failure_refused", "truncation_refused"),
        requires_complete_coverage=True),
    SubjectSpec(
        "P3B03", "cache_first_acquisition",
        (("planned_requests", 2), ("cache_hits", 1), ("provider_calls_saved", 1)),
        ("unnecessary_refetches",),
        ("same_input_uses_cache", "empty_distinguished", "quota_shared")),
    SubjectSpec(
        "P3B04", "daily_market_merge_rebuild_equivalence",
        (("logical_rows", 3), ("retry_replays", 1)),
        ("rebuild_mismatches",),
        ("clean_rebuild_equal", "retry_idempotent", "noop_fragment_hash_stable"),
        requires_changed_partitions=True),
    SubjectSpec(
        "P3B05", "correction_tombstone_conflict",
        (("corrections", 1), ("tombstones", 1), ("conflicts_refused", 1),
         ("ambiguous_events_refused", 1)),
        (), ("filesystem_order_irrelevant", "conflict_refused", "event_identity_preserved"),
        requires_changed_partitions=True),
    SubjectSpec(
        "P3B06", "dependency_explanation_finality",
        (("explanations", 1), ("finality_receipts", 1),
         ("conservative_invalidations", 1)),
        ("moving_head_mismatches",),
        ("snapshot_pinned", "unknown_dependency_invalidates", "finality_native")),
    SubjectSpec(
        "P3B07", "atomic_fault_recovery",
        (("injected_faults", 1), ("recoveries", 1), ("old_snapshot_reads", 1)),
        ("partial_promotions",),
        ("old_head_retained", "retry_classified", "conflict_refused")),
    SubjectSpec(
        "P3B08", "resource_counters",
        (("runtime_ms", 1), ("peak_rss_bytes", 1)),
        ("concurrent_provider_processes",),
        ("runtime_recorded", "rss_recorded", "cache_counter_recorded")),
)

SUBJECT_BY_ID = {subject.subject_id: subject for subject in SUBJECTS}
SUBJECT_BY_NAME = {subject.name: subject for subject in SUBJECTS}

_TOP_LEVEL_FIELDS = {
    "schema_version", "run_id", "evidence_scope", "artifacts", "subjects",
    "retained_snapshot_refs",
}
_SUBJECT_FIELDS = {
    "status", "artifact_refs", "counts", "changed_partitions", "no_op_writes",
    "negative_controls", "coverage_state",
}


def registry() -> tuple[dict, ...]:
    """Return a machine-readable copy of the fixed acceptance inventory."""
    return tuple({
        "subject_id": item.subject_id,
        "name": item.name,
        "minimum_counts": dict(item.minimum_counts),
        "zero_counts": list(item.zero_counts),
        "negative_controls": list(item.negative_controls),
        "requires_changed_partitions": item.requires_changed_partitions,
        "requires_complete_coverage": item.requires_complete_coverage,
    } for item in SUBJECTS)


def _finding(code: str, *, subject_id: str | None = None, reason: str | None = None,
             **details) -> dict:
    result = {"code": code}
    if subject_id is not None:
        result["subject_id"] = subject_id
    if reason is not None:
        result["reason"] = reason
    result.update(details)
    return result


def _read_artifact_payload(ref: str, artifact: Mapping[str, object],
                           artifact_root: Path, evidence_scope: object) -> object:
    if "inline" in artifact:
        return artifact["inline"]
    raw_path = artifact.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = artifact_root / path
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None


def _validate_artifact(ref: str, artifacts: object, artifact_root: Path,
                       evidence_scope: object) -> tuple[list[dict], object]:
    if not isinstance(artifacts, dict) or ref not in artifacts:
        return [_finding("MISSING_ARTIFACT", reason=ref)], None
    artifact = artifacts[ref]
    if not isinstance(artifact, dict):
        return [_finding("INVALID_ARTIFACT", reason=ref)], None
    expected_hash = artifact.get("content_hash")
    if not isinstance(expected_hash, str) or not HASH_RE.fullmatch(expected_hash):
        return [_finding("INVALID_ARTIFACT", reason=ref + ":content_hash")], None
    has_inline = "inline" in artifact
    has_path = "path" in artifact
    if has_inline == has_path:
        return [_finding("INVALID_ARTIFACT", reason=ref + ":one of inline/path required")], None
    if has_inline:
        if evidence_scope == "frozen_real_data":
            return [_finding("FIXTURE_ARTIFACT_FOR_REAL_DATA", reason=ref)], None
        actual_hash = content_hash(artifact["inline"])
    else:
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            return [_finding("INVALID_ARTIFACT", reason=ref + ":path")], None
        path = Path(raw_path)
        if not path.is_absolute():
            path = artifact_root / path
        if not path.is_file():
            return [_finding("MISSING_ARTIFACT", reason=ref + ":" + raw_path)], None
        try:
            actual_hash = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return [_finding("MISSING_ARTIFACT", reason=ref + ":unreadable")], None
    if actual_hash != expected_hash:
        return [_finding("ARTIFACT_HASH_MISMATCH", reason=ref)], None
    return [], _read_artifact_payload(ref, artifact, artifact_root, evidence_scope)


def _subject_result(spec: SubjectSpec, record: object, artifacts: object,
                    artifact_root: Path, evidence_scope: object) -> dict:
    findings: list[dict] = []
    if not isinstance(record, dict):
        findings.append(_finding("MISSING_SUBJECT", subject_id=spec.subject_id,
                                 reason=spec.name))
        record = {}
    unknown = sorted(set(record) - _SUBJECT_FIELDS)
    if unknown:
        findings.append(_finding("UNKNOWN_FIELD", subject_id=spec.subject_id,
                                 reason=unknown[0]))
    if record.get("status") != "passed":
        findings.append(_finding("SUBJECT_NOT_PASSED", subject_id=spec.subject_id,
                                 reason=str(record.get("status", "missing"))))

    refs = record.get("artifact_refs")
    if not isinstance(refs, list) or not refs or not all(isinstance(ref, str) for ref in refs):
        findings.append(_finding("MISSING_ARTIFACT", subject_id=spec.subject_id,
                                 reason="artifact_refs"))
        refs = []
    artifact_payloads = {}
    for ref in refs:
        artifact_findings, payload = _validate_artifact(
            ref, artifacts, artifact_root, evidence_scope)
        artifact_payloads[ref] = payload
        for artifact_finding in artifact_findings:
            artifact_finding["subject_id"] = spec.subject_id
            findings.append(artifact_finding)

    counts = record.get("counts")
    if not isinstance(counts, dict):
        findings.append(_finding("INVALID_COUNT", subject_id=spec.subject_id,
                                 reason="counts"))
        counts = {}
    for name, minimum in spec.minimum_counts:
        value = counts.get(name)
        if type(value) is not int or value < minimum:
            findings.append(_finding("COUNT_BELOW_MINIMUM", subject_id=spec.subject_id,
                                     reason=name, expected=minimum, actual=value))
    for name in spec.zero_counts:
        value = counts.get(name)
        if type(value) is not int or value != 0:
            findings.append(_finding("NONZERO_FAILURE_COUNT", subject_id=spec.subject_id,
                                     reason=name, actual=value))
    if spec.requires_complete_coverage:
        if record.get("coverage_state") != "complete":
            findings.append(_finding("PARTIAL_COVERAGE", subject_id=spec.subject_id,
                                     reason=str(record.get("coverage_state", "missing"))))
        denominator = counts.get("denominator_keys")
        completed = counts.get("completed_keys")
        if type(denominator) is not int or type(completed) is not int or completed != denominator:
            findings.append(_finding("PARTIAL_COVERAGE", subject_id=spec.subject_id,
                                     reason="completed_keys must equal denominator_keys"))

    changed = record.get("changed_partitions")
    valid_changed = (isinstance(changed, list)
                     and all(isinstance(item, str) and item for item in changed)
                     and len(changed) == len(set(changed)))
    if not valid_changed:
        findings.append(_finding("INVALID_CHANGED_PARTITIONS", subject_id=spec.subject_id))
        changed = []
    elif spec.requires_changed_partitions and not changed:
        findings.append(_finding("MISSING_CHANGED_PARTITIONS", subject_id=spec.subject_id))

    no_op_writes = record.get("no_op_writes")
    if type(no_op_writes) is not int or no_op_writes != 0:
        findings.append(_finding("NOOP_REWROTE_DATA", subject_id=spec.subject_id,
                                 actual=no_op_writes))
        no_op_writes = no_op_writes if type(no_op_writes) is int else 0

    controls = record.get("negative_controls")
    if not isinstance(controls, dict):
        controls = {}
    normalized_controls = {}
    for name in spec.negative_controls:
        passed = controls.get(name) is True
        normalized_controls[name] = passed
        if not passed:
            findings.append(_finding("NEGATIVE_CONTROL_FAILED", subject_id=spec.subject_id,
                                     reason=name))

    if evidence_scope == "frozen_real_data":
        for ref, payload in artifact_payloads.items():
            if not isinstance(payload, dict) or payload.get(
                    "schema_version") != "phase3b_run_receipt.v1.0":
                findings.append(_finding(
                    "INVALID_REAL_RECEIPT", subject_id=spec.subject_id,
                    reason=ref + ":schema_version"))
                continue
            subject_receipts = payload.get("subject_results")
            source_record = (subject_receipts.get(spec.subject_id)
                             if isinstance(subject_receipts, dict) else None)
            if not isinstance(source_record, dict):
                findings.append(_finding(
                    "INVALID_REAL_RECEIPT", subject_id=spec.subject_id,
                    reason=ref + ":subject_results"))
                continue
            if source_record.get("counts") != counts:
                findings.append(_finding(
                    "RECEIPT_COUNT_MISMATCH", subject_id=spec.subject_id,
                    reason=ref))
            if source_record.get("negative_controls") != normalized_controls:
                findings.append(_finding(
                    "RECEIPT_CONTROL_MISMATCH", subject_id=spec.subject_id,
                    reason=ref))

    return {
        "name": spec.name,
        "status": "PASS" if not findings else "FAIL",
        "counts": counts,
        "changed_partitions": changed,
        "no_op_writes": no_op_writes,
        "negative_controls": normalized_controls,
        "findings": findings,
    }


def evaluate(evidence: object, *, root: Path = ROOT,
             artifact_root: Path | None = None) -> dict:
    """Validate one Phase 3B evidence document and return JSON-shaped results."""
    started = time.monotonic()
    findings: list[dict] = []
    artifact_root = Path(artifact_root) if artifact_root is not None else root
    if not isinstance(evidence, dict):
        findings.append(_finding("MISSING_EVIDENCE", reason="evidence document"))
        evidence = {}
    unknown = sorted(set(evidence) - _TOP_LEVEL_FIELDS)
    if unknown:
        findings.append(_finding("UNKNOWN_FIELD", reason=unknown[0]))
    if evidence.get("schema_version") != EVIDENCE_SCHEMA:
        findings.append(_finding("UNSUPPORTED_VERSION", reason="schema_version"))
    if not isinstance(evidence.get("run_id"), str) or not evidence.get("run_id"):
        findings.append(_finding("MISSING_EVIDENCE", reason="run_id"))
    scope = evidence.get("evidence_scope")
    if scope not in ("deterministic_fixture", "frozen_real_data"):
        findings.append(_finding("BAD_EVIDENCE_SCOPE", reason=str(scope)))

    records = evidence.get("subjects")
    if not isinstance(records, dict):
        records = {}
    unknown_subjects = sorted(set(records) - set(SUBJECT_BY_ID))
    for subject_id in unknown_subjects:
        findings.append(_finding("UNKNOWN_SUBJECT", subject_id=subject_id))
    artifacts = evidence.get("artifacts")
    subject_results = {}
    for spec in SUBJECTS:
        result = _subject_result(
            spec, records.get(spec.subject_id), artifacts, artifact_root, scope)
        subject_results[spec.subject_id] = result
        findings.extend(result["findings"])

    retained = evidence.get("retained_snapshot_refs")
    if not isinstance(retained, list) or len(retained) < 2 \
            or not all(isinstance(ref, str) and ref for ref in retained):
        findings.append(_finding("MISSING_EVIDENCE", reason="retained_snapshot_refs"))
        retained = []

    controls = {
        f"{subject_id}.{name}": passed
        for subject_id, result in subject_results.items()
        for name, passed in result["negative_controls"].items()
    }
    changed = sorted({partition for result in subject_results.values()
                      for partition in result["changed_partitions"]})
    no_op_writes = sum(result["no_op_writes"] for result in subject_results.values())
    passed = sum(result["status"] == "PASS" for result in subject_results.values())
    artifact_findings = sum(finding["code"] in {
        "MISSING_ARTIFACT", "INVALID_ARTIFACT", "ARTIFACT_HASH_MISMATCH"
    } for finding in findings)
    code_hash = source_hash(source_files(root))
    env_hash, env_source = environment_hash(root)
    ok = not findings
    status = "FIXTURE_PASS" if ok and scope == "deterministic_fixture" else (
        "PASS" if ok else "FAIL")
    return {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "ok": ok,
        "evidence_scope": scope,
        "run_id": evidence.get("run_id"),
        "implementation_code_hash": code_hash,
        "environment_hash": env_hash,
        "environment_hash_source": env_source,
        "counts": {
            "registered_subjects": len(SUBJECTS),
            "passed_subjects": passed,
            "failed_subjects": len(SUBJECTS) - passed,
            "artifact_findings": artifact_findings,
            "negative_controls_passed": sum(controls.values()),
            "negative_controls_total": len(controls),
        },
        "changed_partitions": changed,
        "no_op_writes": no_op_writes,
        "negative_controls": controls,
        "retained_snapshot_refs": retained,
        "subjects": subject_results,
        "findings": findings,
        "seconds": round(time.monotonic() - started, 3),
    }


def _artifact(payload: object) -> dict:
    return {"inline": payload, "content_hash": content_hash(payload)}


def _daily_market_contract() -> TableContract:
    h = "sha256:" + "4" * 64
    return TableContract(
        contract_id="daily_market.v1", definition_hash=h, table_name="daily_market",
        semantic_version="1.0", columns=(
            ColumnContract(name="ticker", physical_type="string", nullable=False),
            ColumnContract(name="date", physical_type="date32", nullable=False),
            ColumnContract(name="year", physical_type="int64", nullable=False),
            ColumnContract(name="close", physical_type="float64", nullable=False,
                           unit="USD"),
        ), primary_key=("ticker", "date"), duplicate_policy="reject",
        foreign_keys=(), partition_columns=("year",),
        filterable_columns=("ticker", "date", "year"),
        orderable_columns=("ticker", "date"),
        finality_semantics="eod_final.v1", provenance_semantics="raw_receipt.v1",
        coverage_semantics="ticker_session_denominator.v1",
        schema_evolution_policy="major_on_meaning_change.v1",
        maximum_batch_rows=1000, maximum_result_rows=10000)


def _daily_revision(*, ticker: str, session_date: str, row: dict | None,
                    revision_id: str, ordinal: int, deleted: bool = False,
                    received_at: str = "2026-09-16T01:00:00.000000Z"):
    revision_hash = data_incremental.revision_content_hash(
        ticker=ticker, session_date=session_date, row=row, deleted=deleted)
    candidate = RevisionCandidate(
        revision_id=revision_id,
        logical_key=data_incremental.daily_market_logical_key(ticker, session_date),
        source="fixture", source_priority=1, finality="final",
        revision_ordinal=ordinal, received_at=received_at, content_hash=revision_hash)
    return data_incremental.DailyMarketRevision(
        candidate=candidate, ticker=ticker, session_date=session_date, row=row,
        deleted=deleted, raw_receipt_id="raw-" + revision_id,
        normalization_id="norm-" + revision_id)


def _contract_fixture() -> tuple[dict, dict]:
    """Exercise every new contract through the Phase 2 typed-document codec."""
    h = "sha256:" + "1" * 64
    table = TableContractRef(contract_id="daily_market.v1", definition_hash=h)
    obj = ObjectRef(kind="raw_response", object_id="raw-1", content_hash=h, byte_size=42)
    interval = TimeInterval(column="date", start_inclusive="2026-09-14",
                            end_exclusive="2026-09-16")
    revision = RevisionCandidate(
        revision_id="rev-1", logical_key="AAPL:2026-09-15", source="fixture",
        source_priority=1, finality="final", revision_ordinal=1,
        received_at="2026-09-16T01:00:00.000000Z", content_hash=h)
    receipt = AcquisitionReceipt(
        receipt_id="receipt-1", request_hash=h, source="fixture", endpoint="daily",
        requested_at="2026-09-16T00:00:00.000000Z",
        completed_at="2026-09-16T01:00:00.000000Z", redacted_request_ref="request-1",
        raw_object_ref=obj, response_status=200, outcome="complete",
        requested_items=("AAPL",), returned_items=("AAPL",),
        legitimate_empty_items=(), unavailable_items=(), revisions=(revision,),
        selected_revision_ids={"AAPL:2026-09-15": "rev-1"}, quota_headers={})
    key = CoverageKey(item_key="AAPL:2026-09-15", session_date="2026-09-15",
                      ticker="AAPL")
    outcome = CoverageOutcome(key=key, status="present", receipt_id="receipt-1",
                              revision_id="rev-1", finality="final")
    coverage = CompletedCoverage(
        coverage_id="coverage-1", table_contract_ref=table, source="fixture",
        endpoint="daily", interval=interval, expected=(key,), outcomes=(outcome,),
        covered_tickers=("AAPL",), acquisition_receipt_refs=("receipt-1",),
        state="complete", completed_at="2026-09-16T01:00:00.000000Z")
    row_change = RowChange(
        logical_key="AAPL:2026-09-15", partition_key="2026-09", columns=("close",),
        time_range=interval, old_hash=None, new_hash=h, revision_kind="append",
        revision_id="rev-1")
    impact = DependencyImpact(dependency_id="rolling", scope="suffix",
                              affected_keys=("AAPL",), time_range=interval)
    base = DatasetVersionRef(dataset_version_id="dv-1", table_contract_ref=table,
                             manifest_hash=h)
    result = DatasetVersionRef(dataset_version_id="dv-2", table_contract_ref=table,
                               manifest_hash="sha256:" + "2" * 64)
    changeset = ChangeSet(
        changeset_id="change-1", table_contract_ref=table,
        base_dataset_version_ref=base, result_dataset_version_ref=result,
        acquisition_receipt_refs=("receipt-1",), coverage_receipt_refs=("coverage-1",),
        changes=(row_change,), changed_partitions=("2026-09",),
        dependency_impacts=(impact,), unknown_dependencies=(),
        dependency_disposition="exact", outcome="changed", normalized_payloads=1,
        rewritten_partitions=1)
    samples = (revision, receipt, key, outcome, coverage, row_change, impact, changeset)
    for sample in samples:
        assert from_document(type(sample), to_document(sample)) == sample
    failures = 0
    bad = to_document(receipt)
    bad["outcome"] = "invented"
    try:
        from_document(AcquisitionReceipt, bad)
    except DocumentError:
        failures += 1
    bad = to_document(receipt)
    bad["unknown"] = True
    try:
        from_document(AcquisitionReceipt, bad)
    except DocumentError:
        failures += 1
    bad = to_document(receipt)
    del bad["receipt_id"]
    try:
        from_document(AcquisitionReceipt, bad)
    except DocumentError:
        failures += 1
    return ({"roundtrips": len(samples), "schema_refusals": failures},
            {"table_ref": to_document(table), "sample_types": [type(x).__name__ for x in samples]})


def deterministic_fixture_evidence() -> dict:
    """Build deterministic, network-free evidence for testing the gate itself."""
    contract_metrics, contract_artifact = _contract_fixture()
    h = "sha256:" + "3" * 64
    snapshot = SnapshotRef(
        snapshot_id="snap-r1", manifest_hash=h, table_versions={}, calendar_version="cal-v1",
        source_priority_version="priority-v1", finality_receipt_refs=("finality-r1",),
        knowledge_mode_by_table={})
    unit_a = incremental_data.RefreshUnit(
        request_id="req-a", table_name="daily_market", partition_key="2026-09",
        expected_keys=("AAPL",))
    unit_b = incremental_data.RefreshUnit(
        request_id="req-b", table_name="daily_market", partition_key="2026-09",
        expected_keys=("MSFT",))
    cached = incremental_data.classify_response(
        200, ("AAPL",), returned_keys=("AAPL",), request_id="req-a",
        receipt_ref="receipt-a", raw_hash=h, cache_hit=True)
    plan = incremental_data.plan_refresh(
        snapshot, (unit_b, unit_a), cached_outcomes={"req-a": cached},
        provider_account="polygon", max_attempts=3)
    complete_b = incremental_data.classify_response(
        200, ("MSFT",), returned_keys=("MSFT",), request_id="req-b",
        receipt_ref="receipt-b", raw_hash=h)
    admission = incremental_data.admit_candidate_commit(plan, (complete_b,))
    partial = incremental_data.classify_response(
        200, ("AAPL", "MSFT"), returned_keys=("AAPL",), request_id="partial",
        receipt_ref="partial-receipt")
    auth = incremental_data.classify_response(
        401, ("AAPL",), request_id="auth", receipt_ref="auth-receipt")
    truncated = incremental_data.classify_response(
        200, ("AAPL",), returned_keys=("AAPL",), truncated=True,
        request_id="truncated", receipt_ref="truncated-receipt")

    daily_contract = _daily_market_contract()
    daily_ref = TableContractRef(
        contract_id=daily_contract.contract_id,
        definition_hash=daily_contract.definition_hash)
    original_rows = (
        {"ticker": "AAPL", "date": "2026-09-14", "year": 2026, "close": 230.0},
        {"ticker": "MSFT", "date": "2026-09-14", "year": 2026, "close": 510.0},
    )
    base_revisions = tuple(_daily_revision(
        ticker=row["ticker"], session_date=row["date"], row=row,
        revision_id="base-" + row["ticker"].lower(), ordinal=1)
        for row in original_rows)
    append_row = {"ticker": "AAPL", "date": "2026-09-15", "year": 2026,
                  "close": 232.0}
    append_revision = _daily_revision(
        ticker="AAPL", session_date="2026-09-15", row=append_row,
        revision_id="append-aapl", ordinal=1)
    incremental_merge = data_incremental.merge_daily_market(
        daily_contract, original_rows, base_revisions, (append_revision,))
    clean_merge = data_incremental.merge_daily_market(
        daily_contract, (), (), (*base_revisions, append_revision))
    retry_merge = data_incremental.merge_daily_market(
        daily_contract, incremental_merge.rows, incremental_merge.winners,
        (append_revision,))

    correction_row = {"ticker": "AAPL", "date": "2026-09-14", "year": 2026,
                      "close": 231.0}
    correction_revision = _daily_revision(
        ticker="AAPL", session_date="2026-09-14", row=correction_row,
        revision_id="correct-aapl", ordinal=2)
    tombstone_revision = _daily_revision(
        ticker="MSFT", session_date="2026-09-14", row=None,
        revision_id="delete-msft", ordinal=2, deleted=True)
    revision_merge = data_incremental.merge_daily_market(
        daily_contract, incremental_merge.rows, incremental_merge.winners,
        (correction_revision, tombstone_revision))
    conflict_a = _daily_revision(
        ticker="AAPL", session_date="2026-09-14", row=correction_row,
        revision_id="conflict-a", ordinal=3)
    conflict_b_row = dict(correction_row, close=229.0)
    conflict_b = _daily_revision(
        ticker="AAPL", session_date="2026-09-14", row=conflict_b_row,
        revision_id="conflict-b", ordinal=3)
    conflict_refused = False
    try:
        data_incremental.select_revision_winners((conflict_a, conflict_b))
    except DataError as exc:
        conflict_refused = exc.code == "IDENTITY_CONFLICT"
    ordered_winners = data_incremental.select_revision_winners(
        (*base_revisions, append_revision))
    reversed_winners = data_incremental.select_revision_winners(
        tuple(reversed((*base_revisions, append_revision))))
    order_irrelevant = {
        key: value.candidate.revision_id for key, value in ordered_winners.items()
    } == {
        key: value.candidate.revision_id for key, value in reversed_winners.items()
    }
    event_before = {"event_id": "AAPL_2026-09-16", "ticker": "AAPL",
                    "event_date": "2026-09-16", "session": "AMC",
                    "event_cluster_id": "AAPL-q3"}
    event_after = {"ticker": "AAPL", "event_date": "2026-09-17", "session": "BMO",
                   "event_cluster_id": "AAPL-q3"}
    event_identity_preserved = (
        resolve_event_identity(event_after, (event_before,)) == event_before["event_id"])
    ambiguous_events_refused = False
    try:
        resolve_event_identity(event_after, (
            event_before, dict(event_before, event_id="AAPL-duplicate")))
    except DataError as exc:
        ambiguous_events_refused = exc.code == "IDENTITY_CONFLICT"

    coverage_interval = TimeInterval(
        column="date", start_inclusive="2026-09-15", end_exclusive="2026-09-16")
    coverage_keys = (
        CoverageKey(item_key="AAPL:2026-09-15", session_date="2026-09-15",
                    ticker="AAPL"),
        CoverageKey(item_key="MSFT:2026-09-15", session_date="2026-09-15",
                    ticker="MSFT"),
    )
    coverage_outcomes = tuple(CoverageOutcome(
        key=key, status="present", receipt_id="receipt-" + key.ticker.lower(),
        revision_id="revision-" + key.ticker.lower(), finality="final")
        for key in coverage_keys)
    complete_coverage = data_incremental.build_completed_coverage(
        daily_ref, source="fixture", endpoint="daily", interval=coverage_interval,
        expected=coverage_keys, outcomes=coverage_outcomes,
        acquisition_receipt_refs=("receipt-aapl", "receipt-msft"),
        completed_at="2026-09-16T01:00:00.000000Z")
    partial_coverage = data_incremental.build_completed_coverage(
        daily_ref, source="fixture", endpoint="daily", interval=coverage_interval,
        expected=coverage_keys, outcomes=coverage_outcomes[:1],
        acquisition_receipt_refs=("receipt-aapl",),
        completed_at="2026-09-16T01:00:00.000000Z")

    change = incremental_data.DataChange(
        change_id="change-correction", table_name="daily_market",
        revision_kind="correction", columns=("close",), start="2026-09-15",
        end_exclusive="2026-09-16", keys=("AAPL",))
    unknown = incremental_data.DataChange(
        change_id="change-unknown", table_name="earnings_events",
        revision_kind="correction", columns_known=False, dependency_known=False,
        start="2026-09-01")
    impact = incremental_data.plan_dependency_impacts(
        snapshot, (change, unknown), (), (
            incremental_data.DependencyRule(
                consumer_id="rolling", table_names=("daily_market",), mode="suffix",
                columns=("close",)),
            incremental_data.DependencyRule(
                consumer_id="folds", table_names=("earnings_events",), mode="unknown"),
        ))

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE data_snapshot_heads "
                 "(scope TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, generation INTEGER NOT NULL)")
    conn.execute("INSERT INTO data_snapshot_heads VALUES (?, ?, ?)",
                 ("shadow", "snap-r1", 4))
    candidate = incremental_data.CandidatePromotion(
        candidate_scope="candidate", target_scope="shadow", candidate_snapshot_id="snap-r2",
        expected_target_snapshot_id="snap-r1", expected_target_generation=4,
        comparison_receipt_id="comparison-r1", commit_admission=admission)
    recovery = incremental_data.recovery_decision(conn, candidate)
    conn.close()

    inventory = list(inventory_document())
    payloads = {
        "artifact-inventory": {"inventory": inventory, **contract_artifact},
        "artifact-coverage": {"partial": partial.kind, "auth": auth.kind,
                              "truncated": truncated.kind, "admitted": admission.admitted,
                              "complete_state": complete_coverage.state,
                              "partial_state": partial_coverage.state},
        "artifact-cache": {"plan_hash": plan.plan_hash, "fetch_units": ["req-b"],
                           "provider_calls": plan.provider_calls},
        "artifact-merge": {"incremental": list(incremental_merge.rows),
                           "clean": list(clean_merge.rows),
                           "retry_changes": len(retry_merge.changes),
                           "partition_hashes": incremental_merge.partition_hashes},
        "artifact-revisions": {
            "selection_order": ["source", "finality", "ordinal"],
            "revision_kinds": [item.revision_kind for item in revision_merge.changes],
            "conflict": "refused" if conflict_refused else "accepted",
            "order_irrelevant": order_irrelevant,
        },
        "artifact-dependencies": {"snapshot": impact.snapshot_id,
                                  "scopes": [item.scope for item in impact.invalidations]},
        "artifact-atomic": {"recovery": recovery.action, "old_head": "snap-r1"},
        "artifact-resources": {"runtime_ms": 12, "peak_rss_bytes": 1_048_576,
                               "provider_calls": plan.provider_calls},
    }
    artifacts = {name: _artifact(payload) for name, payload in payloads.items()}

    def record(ref, counts, controls, *, partitions=(), coverage_state=None):
        value = {
            "status": "passed", "artifact_refs": [ref], "counts": counts,
            "changed_partitions": list(partitions), "no_op_writes": 0,
            "negative_controls": controls,
        }
        if coverage_state is not None:
            value["coverage_state"] = coverage_state
        return value

    return {
        "schema_version": EVIDENCE_SCHEMA,
        "run_id": "phase3b-deterministic-fixture-v1",
        "evidence_scope": "deterministic_fixture",
        "artifacts": artifacts,
        "retained_snapshot_refs": ["snap-r1", "snap-r2"],
        "subjects": {
            "P3B01": record("artifact-inventory", {
                "inventory_rows": len(inventory),
                "contract_roundtrips": contract_metrics["roundtrips"],
                "schema_refusals": contract_metrics["schema_refusals"],
            }, {"unknown_field_refused": True, "bad_enum_refused": True,
                "missing_field_refused": True}),
            "P3B02": record("artifact-coverage", {
                "denominator_keys": len(complete_coverage.expected),
                "completed_keys": len(complete_coverage.outcomes), "rejected_incomplete": 3,
                "watermark_advances_on_failure": 0,
            }, {"partial_refused": partial.kind == "partial",
                "auth_failure_refused": auth.kind == "credential_invalid",
                "truncation_refused": (truncated.kind == "partial"
                                       and partial_coverage.state == "incomplete")},
                coverage_state=complete_coverage.state),
            "P3B03": record("artifact-cache", {
                "planned_requests": 2, "cache_hits": len(plan.cached),
                "provider_calls_saved": 3, "unnecessary_refetches": 0,
            }, {"same_input_uses_cache": len(plan.cached) == 1,
                "empty_distinguished": True, "quota_shared": plan.provider_account == "polygon"}),
            "P3B04": record("artifact-merge", {
                "logical_rows": len(incremental_merge.rows), "retry_replays": 1,
                "rebuild_mismatches": int(incremental_merge.rows != clean_merge.rows),
            }, {"clean_rebuild_equal": incremental_merge.rows == clean_merge.rows,
                "retry_idempotent": not retry_merge.changes,
                "noop_fragment_hash_stable": not retry_merge.changed_partitions},
                partitions=tuple("daily_market/" + item
                                 for item in incremental_merge.changed_partitions)),
            "P3B05": record("artifact-revisions", {
                "corrections": sum(item.revision_kind == "correction"
                                   for item in revision_merge.changes),
                "tombstones": sum(item.revision_kind == "tombstone"
                                   for item in revision_merge.changes),
                "conflicts_refused": int(conflict_refused),
                "ambiguous_events_refused": int(ambiguous_events_refused),
            }, {"filesystem_order_irrelevant": order_irrelevant,
                "conflict_refused": conflict_refused,
                "event_identity_preserved": event_identity_preserved},
                partitions=("earnings_events/2026",)),
            "P3B06": record("artifact-dependencies", {
                "explanations": 1, "finality_receipts": len(snapshot.finality_receipt_refs),
                "conservative_invalidations": int(impact.conservative),
                "moving_head_mismatches": 0,
            }, {"snapshot_pinned": impact.snapshot_id == snapshot.snapshot_id,
                "unknown_dependency_invalidates": any(item.scope == "full"
                                                      for item in impact.invalidations),
                "finality_native": True}),
            "P3B07": record("artifact-atomic", {
                "injected_faults": 1, "recoveries": 1, "old_snapshot_reads": 1,
                "partial_promotions": 0,
            }, {"old_head_retained": True, "retry_classified": recovery.action == "retry_promotion",
                "conflict_refused": True}),
            "P3B08": record("artifact-resources", {
                "runtime_ms": 12, "peak_rss_bytes": 1_048_576,
                "provider_calls": plan.provider_calls, "cache_hits": len(plan.cached),
                "concurrent_provider_processes": 0,
            }, {"runtime_recorded": True, "rss_recorded": True,
                "cache_counter_recorded": True}),
        },
    }


def _load_evidence(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--evidence", type=Path)
    source.add_argument("--deterministic-fixture", action="store_true")
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list-subjects", action="store_true")
    args = parser.parse_args(argv)
    if args.list_subjects:
        print(json.dumps({"schema_version": RESULT_SCHEMA, "subjects": registry()}, indent=2))
        return 0
    evidence = (deterministic_fixture_evidence() if args.deterministic_fixture
                else _load_evidence(args.evidence) if args.evidence else None)
    result = evaluate(evidence, artifact_root=args.artifact_root)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    if args.json:
        print(encoded)
    else:
        print(f"phase 3B acceptance: {result['status']} "
              f"({len(result['findings'])} findings) in {result['seconds']:.3f}s")
    return int(not result["ok"])


if __name__ == "__main__":
    raise SystemExit(main())
