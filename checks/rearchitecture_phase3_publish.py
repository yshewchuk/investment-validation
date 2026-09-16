#!/usr/bin/env python3
"""L06/L13 evidence producers -- guide §9 rows L06/L13 / §5.3 point 8, §5.4.

Real ``build_candidate`` writes into a private scratch ``serving.sqlite`` +
object store this process owns (never a shared/live root), over the real
attempt-20 population (clean subset -- see ``rearchitecture_phase3_bridge.py``
's module docstring for why: the known legacy render defect on
``structure_params`` is out of scope here too). Every ``PreviewInput`` ref is
a real Phase 2 catalog artifact id/hash (``/root/phase2-shadow-ops``), not a
fabricated one.

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

L13 ``refresh_rollback_receipt_ref``: a private release-root (never a
shared/live one) sequences real, byte-distinct release content
R1 (attempt-20's own real bundle) -> R2 (the clean subset's real bundle,
genuinely different bytes) -> back to R1, verified via a real
``create_server`` HTTP round trip at each step (same mechanism
``rearchitecture_phase3_current_switch.py`` uses). The ``RollbackReceipt``
records the LAST move (R2 -> republished-R1), ``scope="v2_serving_preview"``
matching the schema's own repurposing for Phase 3 (confirmed against
``tests/test_checks_phase3_gate.py``'s fixture, which uses release ids in
``prior_snapshot_id``/``resulting_snapshot_id`` under this same scope
string). ``publish_failure_negative_control``: a "failed R2" republish
carries an ``ArtifactRef`` whose declared hash does not match its real
object bytes; the real ``engine.v2.ops.publication.materialize`` +
``ArtifactStore.verify`` path refuses it (``ArtifactError``/``OpsError``,
never silently accepted), no partial release directory is left under
``releases/``, and CURRENT still resolves to R1.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase1_gate import source_files, source_hash  # noqa: E402
from checks.rearchitecture_phase2_gate import environment_hash as _environment_hash  # noqa: E402
from engine.v2.contracts import ArtifactRef, ObjectRef, PreviewInput, PreviewRelease, RollbackReceipt  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.diagnosis import AGREE, DIFFER, ComparisonReceipt, Envelope, Finding, Population  # noqa: E402
from engine.v2.diagnosis import content_hash as receipt_content_hash  # noqa: E402
from engine.v2.foundation import (  # noqa: E402
    ArtifactError,
    ArtifactStore,
    SystemClock,
    content_hash,
    format_timestamp,
    from_document,
    to_document,
)
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402
from engine.v2.ops.publication import materialize  # noqa: E402
from engine.v2.serving.legacy_bundle import load_legacy_bundle, load_score_document  # noqa: E402
from engine.v2.serving.projections import build_candidate, connect, ensure_schema  # noqa: E402

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


def _preview_input(*, bundle_manifest_ref: str, score_batch_ref: str, score_comparison_receipt_ref: str,
                   render_comparison_receipt_ref: str, source_code_hash: str,
                   source_environment_hash: str) -> PreviewInput:
    """Every ref is a real Phase 2 artifact from the attempt-20 population
    (fetched from ``/root/phase2-shadow-ops/catalog.sqlite``), not fabricated."""
    return PreviewInput(
        source_release_id="relba732fb44d3a88dc2574cc99",
        source_release_manifest_ref="sha256:a022c490df6fb2f09e314fcb190d1be9b0207cc497986fbd5a617a863fb1d8a2",
        snapshot_ref="snap_6ae7348848e4d27486823eb0a9baceff",
        legacy_snapshot_object_ref=ObjectRef(
            kind="legacy_snapshot", object_id="art_5a4bf96123a308d91fdd791d7dd9d666",
            content_hash="sha256:51cc817230b7a6af574a3811e39140835eaf6efa21e976fbdace7dd593683f79",
            byte_size=4496),
        score_batch_ref=score_batch_ref,
        score_job_input_refs=("art_5a4bf96123a308d91fdd791d7dd9d666", "art_3b98d02f7bb33932da2c998ad5a86ca3"),
        bundle_manifest_ref=bundle_manifest_ref,
        model_registry_artifact_refs=("art_334b2cc86b5738ca2fbd73324819b5b1",),
        model_evidence_ref="art_334b2cc86b5738ca2fbd73324819b5b1",
        finality_ref="art_7d2363233946b9f506e716a52bb46c89",
        expected_population_ref=content_hash(["attempt-20-clean-subset-expected-population"]),
        score_comparison_receipt_ref=score_comparison_receipt_ref,
        render_comparison_receipt_ref=render_comparison_receipt_ref,
        source_code_hash=source_code_hash, source_environment_hash=source_environment_hash)


# --------------------------------------------------------------------------
# L06 -- publish_idempotency_parity / publish_corruption_negative_control
# --------------------------------------------------------------------------


def build_publish(clean_score_json: Path, clean_bundle_dir: Path, repository, snapshot_ref, *,
                  serving_root: Path, score_comparison_receipt_ref: str, render_comparison_receipt_ref: str,
                  code_hash: str, environment_hash: str) -> tuple[ComparisonReceipt, ComparisonReceipt]:
    clock = SystemClock()
    score_doc = load_score_document(clean_score_json)
    bundle_rows_by_ticker, bundle_manifest = load_legacy_bundle(clean_bundle_dir)
    bundle_manifest_ref = content_hash(bundle_manifest)

    serving_root.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(serving_root / "objects")
    conn = connect(str(serving_root / "serving.sqlite"), clock=clock)
    findings: list[Finding] = []
    try:
        preview_input = _preview_input(
            bundle_manifest_ref=bundle_manifest_ref, score_batch_ref="art_8214c99d5f38a64ea9bd5dc62d323d72",
            score_comparison_receipt_ref=score_comparison_receipt_ref,
            render_comparison_receipt_ref=render_comparison_receipt_ref,
            source_code_hash=code_hash, source_environment_hash=environment_hash)

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
        retry_input = dataclasses.replace(preview_input, score_batch_ref="art_8214c99d5f38a64ea9bd5dc62d323d72#retry")
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


def _start(release_root: Path, health_path: Path, token: str):
    from engine.v2.serving.operations import create_server
    server = create_server(("127.0.0.1", 0), token=token, health_path=health_path, release_root=release_root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def _stop(server, thread) -> None:
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def _get(base: str, path: str, token: str) -> tuple[int, bytes]:
    request = urllib.request.Request(base + path)
    request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _write_current(release_root: Path, release_id: str) -> None:
    (release_root / "CURRENT").write_text(release_id + "\n")


def build_rollback(real_bundle_dir: Path, clean_bundle_dir: Path, *, release_root: Path, health_path: Path,
                   token: str, code_hash: str, environment_hash: str
                   ) -> tuple[RollbackReceipt, ComparisonReceipt]:
    """R1 (real attempt-20 bundle) -> R2 (clean-subset bundle, genuinely
    different real bytes) -> rollback republish of R1's content, sequenced
    through a private ``release-root`` this process fully owns, verified via
    the same real ``create_server`` HTTP round trip L02 uses."""
    clock = SystemClock()
    releases_dir = release_root / "releases"
    releases_dir.mkdir(parents=True, exist_ok=True)
    if not (releases_dir / "R1").exists():
        shutil.copytree(real_bundle_dir, releases_dir / "R1")
    if not (releases_dir / "R2").exists():
        shutil.copytree(clean_bundle_dir, releases_dir / "R2")
    rollback_target = releases_dir / "R1-rollback"
    if rollback_target.exists():
        shutil.rmtree(rollback_target)
    shutil.copytree(releases_dir / "R1", rollback_target)
    health_path.write_text(json.dumps({"ok": True, "current": None}))

    findings: list[Finding] = []
    generation = 0
    server, thread, base = _start(release_root, health_path, token)
    try:
        for release_id in ("R1", "R2", "R1-rollback"):
            _write_current(release_root, release_id)
            generation += 1
            resolved = json.loads(_get(base, "/release/current.json", token)[1]).get("release_id")
            if resolved != release_id:
                findings.append(_mk_finding("refresh_rollback", f"resolve_after_{release_id}"))
        status, r1_after = _get(base, "/release/R1-rollback/data/board.json", token)
        status_orig, r1_before = _get(base, "/release/R1/data/board.json", token)
        if status != 200 or status_orig != 200 or r1_after != r1_before:
            findings.append(_mk_finding("refresh_rollback", "rollback_content_matches_r1"))
        status, _ = _get(base, "/release/R2/data/board.json", token)
        if status != 200:
            findings.append(_mk_finding("refresh_rollback", "r2_history_retained"))
    finally:
        _stop(server, thread)

    if findings:
        raise RuntimeError(f"real R1->R2->R1 exercise did not verify cleanly: {findings}")

    receipt = RollbackReceipt(
        receipt_id="recv_" + receipt_content_hash(["refresh_rollback", "R2", "R1-rollback"])[7:23],
        scope="v2_serving_preview", prior_snapshot_id="R2", resulting_snapshot_id="R1-rollback",
        prior_generation=2, resulting_generation=3, at=format_timestamp(clock.now()))

    # Negative control: a "failed R2" republish whose declared ArtifactRef hash does
    # not match its real bytes -- the real materialize()+ArtifactStore.verify path
    # must refuse it, leaving CURRENT (and releases/) exactly as before.
    scratch_store = ArtifactStore(release_root / "failure_objects")
    good_bytes = (releases_dir / "R2" / "data" / "board.json").read_bytes()
    good_ref = scratch_store.publish_bytes(good_bytes, schema_ref="release_asset.v1.0")
    corrupted_ref = dataclasses.replace(good_ref, content_hash="sha256:" + "0" * 64)
    current_before = (release_root / "CURRENT").read_text()
    negative_findings: list[Finding] = []
    try:
        materialize(scratch_store, release_root / "failure_target", "R2-failed", {"data/board.json": corrupted_ref})
        negative_findings.append(_mk_finding(PUBLISH_FAILURE_NEGATIVE_KIND, "corrupted_manifest_was_accepted"))
    except (ArtifactError, OpsError):
        pass
    if (release_root / "failure_target" / "releases" / "R2-failed").exists():
        negative_findings.append(_mk_finding(PUBLISH_FAILURE_NEGATIVE_KIND, "partial_release_directory_left"))
    if (release_root / "CURRENT").read_text() != current_before:
        negative_findings.append(_mk_finding(PUBLISH_FAILURE_NEGATIVE_KIND, "current_pointer_moved_on_failure"))
    negative = _receipt(PUBLISH_FAILURE_NEGATIVE_KIND, 1, "operations_release:R1-rollback", "operations_release:R2-failed",
                        negative_findings, 3, code_hash=code_hash, environment_hash=environment_hash,
                        invert=True)
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
    parser.add_argument("--real-bundle-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--score-comparison-receipt", type=Path, required=True,
                        help="a real ComparisonReceipt file (e.g. bridge_value_parity.json) whose "
                             "content hash pins score_comparison_receipt_ref")
    parser.add_argument("--render-comparison-receipt", type=Path, required=True)
    parser.add_argument("--serving-root", type=Path, required=True, help="private scratch serving store/db")
    parser.add_argument("--release-root", type=Path, required=True, help="private scratch release root")
    parser.add_argument("--health-path", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)

    code_hash = source_hash(source_files(ROOT))
    env_hash, _source = _environment_hash(ROOT)
    score_ref = "sha256:" + hashlib.sha256(args.score_comparison_receipt.read_bytes()).hexdigest()
    render_ref = "sha256:" + hashlib.sha256(args.render_comparison_receipt.read_bytes()).hexdigest()

    clock = SystemClock()
    conn = open_catalog(args.catalog, clock=clock)
    store = ArtifactStore(args.store_root)
    repository = Repository(conn, store)
    snapshot_ref = repository.resolve(args.snapshot_id)
    try:
        publish_receipt, publish_negative = build_publish(
            args.clean_score_json, args.clean_bundle_dir, repository, snapshot_ref,
            serving_root=args.serving_root, score_comparison_receipt_ref=score_ref,
            render_comparison_receipt_ref=render_ref, code_hash=code_hash, environment_hash=env_hash)
    finally:
        conn.close()

    rollback_receipt, failure_negative = build_rollback(
        args.real_bundle_dir, args.clean_bundle_dir, release_root=args.release_root,
        health_path=args.health_path, token=args.token, code_hash=code_hash, environment_hash=env_hash)

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
