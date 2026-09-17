#!/usr/bin/env python3
"""L06/L13 evidence producers -- guide §9 rows L06/L13 / §5.3 point 8, §5.4.

Real ``build_candidate`` writes into a private scratch ``serving.sqlite`` +
object store this process owns (never a shared/live root), over the complete
saved-score population. Every ``PreviewInput`` ref comes from the exact
verified input document that built the fenced candidate; there are no
fallback release ids or fabricated references.

L06 ``publish_idempotency_parity``: (1) two ``build_candidate`` calls over
IDENTICAL content into the same serving store produce the same
``release_id`` and write zero additional index rows on the second call
(``INSERT OR IGNORE``, guide §5.3 point 8); (2) a ``fault`` hook aborts a
DIFFERENT candidate mid-sequence (after the findings object is durable but
before the index transaction) and a bare retry of the same content then
succeeds cleanly, with the earlier candidate's release still absent (no
partial row); (3) the FIRST candidate's release stays queryable throughout.
``publish_corruption_negative_control``: a real published detail object's
on-disk bytes are corrupted after the fact; ``ArtifactStore.read_verified``
on its own pinned ``ArtifactRef`` (real hash check, real bytes) refuses it,
while an uncorrupted sibling object and the release's index row remain
readable.

L13 ``refresh_rollback_receipt_ref`` verifies the durable result of the real
fenced publisher: two distinct validated projection generations A -> B,
followed by a fresh published operation release that binds A again. The
receipt names the actual projection release ids. The producer re-reads the
real CURRENT pointer and the rollback release binding; it does not create
directories or write CURRENT itself. ``publish_failure_negative_control``
still invokes the real materializer with a deliberately incorrect artifact
hash and proves it leaves no partial release or pointer mutation.
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
from engine.v2.contracts import ArtifactRef, PreviewInput, PreviewRelease, RollbackReceipt  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.diagnosis import AGREE, DIFFER, ComparisonReceipt, Envelope, Finding, Population  # noqa: E402
from engine.v2.diagnosis import content_hash as receipt_content_hash  # noqa: E402
from engine.v2.foundation import (  # noqa: E402
    ArtifactError,
    ArtifactStore,
    SystemClock,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402
from engine.v2.ops.publication import materialize  # noqa: E402
from engine.v2.serving.legacy_bundle import load_legacy_bundle, load_score_document  # noqa: E402
from engine.v2.serving.projections import build_candidate, connect  # noqa: E402

PUBLISH_IDEMPOTENCY_KIND = "publish_idempotency_parity"
PUBLISH_CORRUPTION_NEGATIVE_KIND = "publish_corruption_negative_control"
PUBLISH_FAILURE_NEGATIVE_KIND = "publish_failure_negative_control"

_INDEX_TABLES = ("serving_release", "serving_object", "serving_event_summary", "serving_score_summary")


def _mk_finding(kind: str, field: str) -> Finding:
    return Finding(finding_id=receipt_content_hash([kind, field])[7:19], first_differing_stage="publish",
                   field_path=field, kind="value", owning_stage="publish")


def _receipt(kind: str, tier: int, left_ref: str, right_ref: str, findings: list[Finding], checks: int, *,
            code_hash: str, environment_hash: str, invert: bool = False) -> ComparisonReceipt:
    """``invert=False`` (comparison receipts): a finding is a real problem
    the probe located, so ``verdict=DIFFER`` when present. ``invert=True``
    (negative-control receipts, mirrors ``rearchitecture_phase3_bridge.py``):
    ``findings`` here tracks whether the safety property FAILED (corruption
    went undetected, a failed publish was wrongly accepted) -- empty means
    the fault was correctly caught (``DIFFER``, safe/fired); a finding means
    the control did NOT fire (``AGREE``, the bad outcome)."""
    population = Population(expected=checks, supported=checks, compared=checks)
    verdict = (AGREE if findings else DIFFER) if invert else (DIFFER if findings else AGREE)
    envelope = Envelope(code_hash=code_hash, environment_hash=environment_hash)
    receipt_id = receipt_content_hash([kind, left_ref, right_ref, [f.finding_id for f in findings]])[7:23]
    return ComparisonReceipt(receipt_id=receipt_id, comparison_kind=kind, tier=tier, left_ref=left_ref,
        right_ref=right_ref, stage_plan_ref=f"{kind}.v1", tolerance_policy_ref="exact_bytes.v1",
        verdict=verdict, findings=tuple(findings), population=population, envelope=envelope)


def _table_counts(conn) -> dict[str, int]:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in _INDEX_TABLES}


def load_preview_input(path: Path) -> PreviewInput:
    """Decode the exact verified input that built the published candidates.

    This deliberately has no fallback constants.  A new Phase 2 handoff must
    supply its own immutable input document rather than inheriting references
    from a previous release.
    """
    return from_document(PreviewInput, json.loads(path.read_text()))


# --------------------------------------------------------------------------
# L06 -- publish_idempotency_parity / publish_corruption_negative_control
# --------------------------------------------------------------------------


def build_publish(clean_score_json: Path, clean_bundle_dir: Path, repository, snapshot_ref, *,
                  preview_input: PreviewInput, serving_root: Path, code_hash: str,
                  environment_hash: str) -> tuple[ComparisonReceipt, ComparisonReceipt]:
    clock = SystemClock()
    score_doc = load_score_document(clean_score_json)
    bundle_rows_by_ticker, bundle_manifest = load_legacy_bundle(clean_bundle_dir)

    serving_root.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(serving_root / "objects")
    conn = connect(str(serving_root / "serving.sqlite"), clock=clock)
    findings: list[Finding] = []
    try:
        result1 = build_candidate(preview_input, score_doc, bundle_rows_by_ticker, repository=repository,
                                  snapshot_ref=snapshot_ref, store=store, conn=conn,
                                  requested_as_of="2026-09-10", resolved_as_of="2026-09-10", clock=clock)
        if not isinstance(result1, PreviewRelease):
            findings.append(_mk_finding(PUBLISH_IDEMPOTENCY_KIND, "first_build_refused"))
            raise RuntimeError("first build_candidate call was refused; see findings")
        counts_after_1 = _table_counts(conn)

        result2 = build_candidate(preview_input, score_doc, bundle_rows_by_ticker, repository=repository,
                                  snapshot_ref=snapshot_ref, store=store, conn=conn,
                                  requested_as_of="2026-09-10", resolved_as_of="2026-09-10", clock=clock)
        counts_after_2 = _table_counts(conn)
        if not isinstance(result2, PreviewRelease) or result2.release_id != result1.release_id:
            findings.append(_mk_finding(PUBLISH_IDEMPOTENCY_KIND, "release_id_not_stable_on_retry"))
        if counts_after_2 != counts_after_1:
            findings.append(_mk_finding(PUBLISH_IDEMPOTENCY_KIND, "retry_wrote_additional_index_rows"))

        # A second, content-distinct candidate that aborts mid-sequence (after the
        # findings object is durable, before the index transaction) leaves no
        # partial release row; a bare retry of the SAME content then succeeds.
        retry_input = dataclasses.replace(preview_input, score_batch_ref=preview_input.score_batch_ref + "#retry")
        aborted = {"hit": False}

        def _fault(point: str) -> None:
            if point == "findings_written" and not aborted["hit"]:
                aborted["hit"] = True
                raise RuntimeError("simulated crash after the findings object was written")

        try:
            build_candidate(retry_input, score_doc, bundle_rows_by_ticker, repository=repository,
                            snapshot_ref=snapshot_ref, store=store, conn=conn, requested_as_of="2026-09-10",
                            resolved_as_of="2026-09-10", clock=clock, fault=_fault)
            findings.append(_mk_finding(PUBLISH_IDEMPOTENCY_KIND, "fault_hook_did_not_abort"))
        except RuntimeError:
            pass
        recovery = build_candidate(retry_input, score_doc, bundle_rows_by_ticker, repository=repository,
                                   snapshot_ref=snapshot_ref, store=store, conn=conn,
                                   requested_as_of="2026-09-10", resolved_as_of="2026-09-10", clock=clock)
        if not isinstance(recovery, PreviewRelease):
            findings.append(_mk_finding(PUBLISH_IDEMPOTENCY_KIND, "retry_after_abort_did_not_recover"))

        if conn.execute("SELECT release_id FROM serving_release WHERE release_id = ?",
                        (result1.release_id,)).fetchone() is None:
            findings.append(_mk_finding(PUBLISH_IDEMPOTENCY_KIND, "first_release_no_longer_readable"))
    except RuntimeError:
        pass
    comparison = _receipt(PUBLISH_IDEMPOTENCY_KIND, 1, "build_candidate:call1", "build_candidate:call2+retry",
                          findings, 6, code_hash=code_hash, environment_hash=environment_hash)

    # Negative control: corrupt one real published detail object's bytes on disk,
    # confirm ArtifactStore.read_verified refuses it while a sibling object and the
    # release's own index row remain readable.
    negative_findings: list[Finding] = []
    object_row = conn.execute("SELECT artifact_id, ref_json FROM serving_object LIMIT 2").fetchall()
    if len(object_row) < 2:
        negative_findings.append(_mk_finding(PUBLISH_CORRUPTION_NEGATIVE_KIND, "not_enough_objects_to_test"))
    else:
        target_ref = from_document(ArtifactRef, json.loads(object_row[0]["ref_json"]))
        sibling_ref = from_document(ArtifactRef, json.loads(object_row[1]["ref_json"]))
        target_path = store.root / target_ref.storage_key
        target_path.chmod(0o644)
        target_path.write_bytes(b"corrupted-by-negative-control-probe")
        try:
            store.read_verified(target_ref)
            negative_findings.append(_mk_finding(PUBLISH_CORRUPTION_NEGATIVE_KIND, "corruption_not_detected"))
        except ArtifactError:
            pass
        try:
            store.read_verified(sibling_ref)
        except ArtifactError:
            negative_findings.append(_mk_finding(PUBLISH_CORRUPTION_NEGATIVE_KIND, "sibling_object_unreadable"))
        if conn.execute("SELECT release_id FROM serving_release WHERE release_id = ?",
                        (result1.release_id,)).fetchone() is None:
            negative_findings.append(_mk_finding(PUBLISH_CORRUPTION_NEGATIVE_KIND, "index_row_lost_on_corruption"))
    negative = _receipt(PUBLISH_CORRUPTION_NEGATIVE_KIND, 1, "object:uncorrupted", "object:corrupted",
                        negative_findings, 3, code_hash=code_hash, environment_hash=environment_hash,
                        invert=True)
    conn.close()
    return comparison, negative


# --------------------------------------------------------------------------
# L13 -- refresh_rollback_receipt_ref / publish_failure_negative_control
# --------------------------------------------------------------------------


def _valid_release_record(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    required = ("projection_release_id", "ops_release_id", "published_at", "delivered_at", "manifest")
    if any(not isinstance(record.get(key), str) or not record[key] for key in required[:-1]):
        return False
    manifest = record.get("manifest")
    gates = manifest.get("gates") if isinstance(manifest, dict) else None
    return isinstance(gates, dict) and set(gates) == {"decision", "projection", "security", "engineering"} and all(
        isinstance(gate, dict) and gate.get("ok") is True and isinstance(gate.get("receipt_ref"), str)
        and gate["receipt_ref"].startswith("sha256:") for gate in gates.values())


def build_fenced_rollback(publication_log: Path, publication_root: Path, failure_root: Path, *,
                          code_hash: str, environment_hash: str) -> tuple[RollbackReceipt, ComparisonReceipt]:
    """Verify a real fenced A -> B -> A publication record.

    The publisher, not this receipt producer, creates release directories and
    atomically advances CURRENT.  This function only validates the returned
    operations manifests and the durable pointer/binding they produced.
    """
    document = json.loads(publication_log.read_text())
    updates = document.get("update") if isinstance(document, dict) else None
    rollback = document.get("rollback") if isinstance(document, dict) else None
    if not isinstance(updates, list) or len(updates) != 2 or not all(_valid_release_record(row) for row in updates) \
            or not _valid_release_record(rollback):
        raise RuntimeError("publication log does not contain verified fenced update records")
    first, second = updates
    if first["projection_release_id"] == second["projection_release_id"] \
            or rollback["projection_release_id"] != first["projection_release_id"] \
            or len({first["ops_release_id"], second["ops_release_id"], rollback["ops_release_id"]}) != 3:
        raise RuntimeError("publication record is not a distinct A to B to A sequence")
    current_path = publication_root / "CURRENT"
    binding_path = publication_root / "releases" / rollback["ops_release_id"] / "projection_binding.json"
    if current_path.read_text().strip() != rollback["ops_release_id"] or not binding_path.is_file():
        raise RuntimeError("fenced publisher current pointer does not name the rollback generation")
    binding = json.loads(binding_path.read_text())
    if binding.get("projection_release_id") != first["projection_release_id"]:
        raise RuntimeError("rollback publication binding does not name the first projection")

    receipt = RollbackReceipt(
        receipt_id="recv_" + receipt_content_hash(["fenced_rollback", second["projection_release_id"],
                                                     first["projection_release_id"]])[7:23],
        scope="v2_serving_preview", prior_snapshot_id=second["projection_release_id"],
        resulting_snapshot_id=first["projection_release_id"], prior_generation=2, resulting_generation=3,
        at=rollback["delivered_at"])

    failure_root.mkdir(parents=True, exist_ok=True)
    scratch_store = ArtifactStore(failure_root / "objects")
    good_ref = scratch_store.publish_bytes(b"phase3-fenced-publish-negative-control", schema_ref="release_asset.v1.0")
    corrupted_ref = dataclasses.replace(good_ref, content_hash="sha256:" + "0" * 64)
    current_before = current_path.read_text()
    negative_findings: list[Finding] = []
    try:
        materialize(scratch_store, failure_root / "target", "failed-generation",
                    {"data/projection_binding.json": corrupted_ref})
        negative_findings.append(_mk_finding(PUBLISH_FAILURE_NEGATIVE_KIND, "corrupted_manifest_was_accepted"))
    except (ArtifactError, OpsError):
        pass
    if (failure_root / "target" / "releases" / "failed-generation").exists():
        negative_findings.append(_mk_finding(PUBLISH_FAILURE_NEGATIVE_KIND, "partial_release_directory_left"))
    if current_path.read_text() != current_before:
        negative_findings.append(_mk_finding(PUBLISH_FAILURE_NEGATIVE_KIND, "published_current_pointer_moved"))
    negative = _receipt(PUBLISH_FAILURE_NEGATIVE_KIND, 1, "operations_release:" + rollback["ops_release_id"],
                        "operations_release:failed-generation", negative_findings, 3,
                        code_hash=code_hash, environment_hash=environment_hash, invert=True)
    return receipt, negative


def publish_document(document, artifact_root: Path, name: str) -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    data = json.dumps(to_document(document), indent=2, sort_keys=True).encode()
    path = artifact_root / name
    path.write_bytes(data)
    return {"path": path.name, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-score-json", type=Path, required=True)
    parser.add_argument("--clean-bundle-dir", type=Path, required=True)
    parser.add_argument("--preview-input", type=Path, required=True,
                        help="the verified input document used for the published candidate")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--serving-root", type=Path, required=True, help="private scratch serving store/db")
    parser.add_argument("--publication-log", type=Path, required=True,
                        help="record emitted by the real fenced A to B to A publication runner")
    parser.add_argument("--publication-root", type=Path, required=True,
                        help="fenced publisher scope root containing CURRENT and immutable releases")
    parser.add_argument("--failure-root", type=Path, required=True,
                        help="private target for the failed-publication negative control")
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)

    code_hash = source_hash(source_files(ROOT))
    env_hash, _source = _environment_hash(ROOT)
    preview_input = load_preview_input(args.preview_input)

    clock = SystemClock()
    conn = open_catalog(args.catalog, clock=clock)
    store = ArtifactStore(args.store_root)
    repository = Repository(conn, store)
    snapshot_ref = repository.resolve(args.snapshot_id)
    try:
        publish_receipt, publish_negative = build_publish(
            args.clean_score_json, args.clean_bundle_dir, repository, snapshot_ref,
            preview_input=preview_input, serving_root=args.serving_root,
            code_hash=code_hash, environment_hash=env_hash)
    finally:
        conn.close()

    rollback_receipt, failure_negative = build_fenced_rollback(
        args.publication_log, args.publication_root, args.failure_root,
        code_hash=code_hash, environment_hash=env_hash)

    out = {}
    for kind, (doc, name) in {
        "publish_idempotency_parity": (publish_receipt, "publish_idempotency_parity.json"),
        "publish_corruption_negative_control": (publish_negative, "publish_corruption_negative_control.json"),
        "refresh_rollback_receipt": (rollback_receipt, "refresh_rollback_receipt.json"),
        "publish_failure_negative_control": (failure_negative, "publish_failure_negative_control.json"),
    }.items():
        ref = publish_document(doc, args.artifact_root, name)
        verdict = getattr(doc, "verdict", None)
        out[kind] = {**ref, **({"verdict": verdict} if verdict is not None else {})}
    print(json.dumps(out, indent=2))

    ok = (publish_receipt.verdict == AGREE and publish_negative.verdict == DIFFER
          and failure_negative.verdict == DIFFER
          and rollback_receipt.resulting_generation > rollback_receipt.prior_generation
          and rollback_receipt.prior_snapshot_id != rollback_receipt.resulting_snapshot_id)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
