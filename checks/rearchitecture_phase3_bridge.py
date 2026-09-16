#!/usr/bin/env python3
"""L03/L04/L05 evidence producers -- guide §9 rows L03-L05 / §5.3.

Real ``build_bridges`` runs over one retained Phase 2 score document and its
matching rendered bundle. No mocks: every input is either the complete real
population on disk, or a deliberate in-memory corruption of a copy for a
negative control. The legacy renderer now preserves replay-input precision in
ticker payloads, so every AGREE-side receipt covers the complete candidate
population. A release with a rounded display field is rejected, never reduced
to a passing subset.

L03 ``bridge_identity_parity``: two independent ``build_bridges`` runs over
identical full-population content produce the SAME ``score_id`` set (a pure
function of ``engine_record``/pinned refs -- ``bridge._build_bridge`` never
reads a clock); a run with a changed ``score_batch_ref`` (a real pinned
dependency) produces a DIFFERENT set; a real ``LegacyScoreBridge`` round-trips
through ``to_document``/``decode_document`` unchanged.
``bridge_malformed_ref_negative_control``: an unsupported schema version and a
missing required field each FAIL strict decode. **Judgement call (mirrors
``rearchitecture_phase3_preview.py``'s auth negative control):** a finding is
recorded only when a malformed document is WRONGLY accepted (a validation
bypass), so ``verdict=DIFFER`` (no findings) means "every malformed case was
correctly refused" (safe), and ``verdict=AGREE`` would mean a bypass fired.

L04 ``bridge_mapping_parity``: the full population's real ``ProjectionFindings``
funnel (planned/rendered main+ladder counted separately, matched==compared,
zero missing/unplanned/duplicate/identity/unresolved_event findings).
``bridge_mapping_negative_control``: a duplicated join key and an unmapped
event date, injected into a COPY of the clean bundle, each produce their own
independent :class:`~engine.v2.contracts.Finding` in one ``build_bridges``
call -- standard convention (a finding here directly means the corruption was
caught, so ``verdict=DIFFER`` when both are found).

L05 ``bridge_value_parity``: the complete source population agrees on every
full-precision field, with zero ``value``-category findings.
``bridge_value_negative_control`` changes one numeric ``structure_params``
field in a copied display row. It is ``DIFFER`` only when the bridge reports
that precision mutation.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase1_gate import source_files, source_hash  # noqa: E402
from checks.rearchitecture_phase2_gate import environment_hash as _environment_hash  # noqa: E402
from engine.v2.contracts import LegacyScoreBridge  # noqa: E402
from engine.v2.data.documents import decode_document  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.diagnosis import (  # noqa: E402
    AGREE,
    DIFFER,
    ComparisonReceipt,
    Envelope,
    Finding,
    Population,
    content_hash,
)
from engine.v2.foundation import ArtifactStore, DocumentError, SystemClock, to_document  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.serving.bridge import build_bridges  # noqa: E402
from engine.v2.serving.legacy_bundle import load_legacy_bundle, load_score_document  # noqa: E402
from engine.v2.serving.projections import _pairs, resolve_event_refs  # noqa: E402

BRIDGE_IDENTITY_KIND = "bridge_identity_parity"
BRIDGE_MALFORMED_REF_KIND = "bridge_malformed_ref_negative_control"
BRIDGE_MAPPING_KIND = "bridge_mapping_parity"
BRIDGE_MAPPING_NEGATIVE_KIND = "bridge_mapping_negative_control"
BRIDGE_VALUE_KIND = "bridge_value_parity"
BRIDGE_VALUE_NEGATIVE_KIND = "bridge_value_negative_control"


def _open_repository(catalog_path: Path, store_root: Path, snapshot_id: str):
    clock = SystemClock()
    conn = open_catalog(catalog_path, clock=clock)
    store = ArtifactStore(store_root)
    repository = Repository(conn, store)
    snapshot_ref = repository.resolve(snapshot_id)
    return conn, repository, snapshot_ref


def _run_bridges(score_doc, bundle_rows_by_ticker, event_refs, snapshot_ref, *,
                 score_batch_ref: str, model_registry_artifact_refs: tuple[str, ...],
                 request_provenance_refs: tuple[str, ...]):
    return build_bridges(
        score_doc, bundle_rows_by_ticker, event_refs, score_batch_ref=score_batch_ref,
        snapshot_ref=snapshot_ref.snapshot_id, model_registry_artifact_refs=model_registry_artifact_refs,
        request_provenance_refs=request_provenance_refs)


def _mk_finding(kind: str, field: str) -> Finding:
    return Finding(finding_id=content_hash([kind, field])[7:19], first_differing_stage="bridge",
                   field_path=field, kind="value", owning_stage="bridge")


def _receipt(kind: str, tier: int, left_ref: str, right_ref: str, findings: list[Finding], checks: int, *,
            code_hash: str, environment_hash: str, invert: bool = False) -> ComparisonReceipt:
    """``invert=False`` (comparison receipts): a finding IS a real difference,
    so ``verdict=DIFFER`` when findings are present, ``AGREE`` when clean.

    ``invert=True`` (negative-control receipts): ``findings`` here tracks
    whether the safety property FAILED (corruption went undetected / a
    malformed document was wrongly accepted) -- the mirror of
    ``rearchitecture_phase3_preview.py``'s auth negative control. Empty
    findings means the fault was correctly caught, i.e. the two sides
    genuinely differ (``DIFFER``, safe/fired); a finding here means the
    control did NOT fire (``AGREE``, the bad outcome the phase-3 evidence
    validator's ``NEGATIVE_CONTROL_NOT_TRIGGERED`` exists to catch).
    """
    population = Population(expected=checks, supported=checks, compared=checks)
    if invert:
        verdict = AGREE if findings else DIFFER
    else:
        verdict = DIFFER if findings else AGREE
    envelope = Envelope(code_hash=code_hash, environment_hash=environment_hash)
    receipt_id = content_hash([kind, left_ref, right_ref, [f.finding_id for f in findings]])[7:23]
    return ComparisonReceipt(receipt_id=receipt_id, comparison_kind=kind, tier=tier, left_ref=left_ref,
        right_ref=right_ref, stage_plan_ref=f"{kind}.v1", tolerance_policy_ref="exact_bytes.v1",
        verdict=verdict, findings=tuple(findings), population=population, envelope=envelope)


# --------------------------------------------------------------------------
# L03 -- bridge_identity_parity / bridge_malformed_ref_negative_control
# --------------------------------------------------------------------------


def build_identity(clean_score_json: Path, clean_bundle_dir: Path, repository, snapshot_ref, *,
                   model_registry_artifact_refs: tuple[str, ...],
                   request_provenance_refs: tuple[str, ...],
                   code_hash: str, environment_hash: str) -> tuple[ComparisonReceipt, ComparisonReceipt]:
    score_doc = load_score_document(clean_score_json)
    bundle_rows_by_ticker, _manifest = load_legacy_bundle(clean_bundle_dir)
    event_refs = resolve_event_refs(repository, snapshot_ref, _pairs(score_doc, bundle_rows_by_ticker))

    run1, findings1 = _run_bridges(score_doc, bundle_rows_by_ticker, event_refs, snapshot_ref,
                                   score_batch_ref="identity-probe-batch-1",
                                   model_registry_artifact_refs=model_registry_artifact_refs,
                                   request_provenance_refs=request_provenance_refs)
    run2, _findings2 = _run_bridges(score_doc, bundle_rows_by_ticker, event_refs, snapshot_ref,
                                    score_batch_ref="identity-probe-batch-1",
                                    model_registry_artifact_refs=model_registry_artifact_refs,
                                    request_provenance_refs=request_provenance_refs)
    run3, _findings3 = _run_bridges(score_doc, bundle_rows_by_ticker, event_refs, snapshot_ref,
                                    score_batch_ref="identity-probe-batch-2",
                                    model_registry_artifact_refs=model_registry_artifact_refs,
                                    request_provenance_refs=request_provenance_refs)
    assert findings1.ok and not findings1.findings, "clean subset must be defect-free (see module docstring)"

    ids1 = sorted(b.score_id for b in run1)
    ids2 = sorted(b.score_id for b in run2)
    ids3 = sorted(b.score_id for b in run3)

    findings: list[Finding] = []
    if ids1 != ids2:
        findings.append(_mk_finding(BRIDGE_IDENTITY_KIND, "score_id_stable_under_identical_rebuild"))
    if ids1 == ids3:
        findings.append(_mk_finding(BRIDGE_IDENTITY_KIND, "score_id_sensitive_to_pinned_score_batch_ref"))

    sample = run1[0]
    round_tripped = decode_document(LegacyScoreBridge, to_document(sample))
    if round_tripped != sample:
        findings.append(_mk_finding(BRIDGE_IDENTITY_KIND, "legacy_score_bridge_round_trip"))
    # score_id never reads a clock (bridge._build_bridge's content_hash input set has no
    # operational timestamp) -- ids1==ids2 above, built from two independently-timed calls,
    # is the real evidence for "operational timings do not change identity".

    comparison = _receipt(BRIDGE_IDENTITY_KIND, 0, "build:run1", "build:run2/run3", findings, 3,
                          code_hash=code_hash, environment_hash=environment_hash)

    negative_findings: list[Finding] = []
    valid_doc = to_document(sample)
    for label, mutate in (
        ("unsupported_schema_version", lambda d: {**d, "schema_version": "legacy_score_bridge.v99.0"}),
        ("missing_required_field", lambda d: {k: v for k, v in d.items() if k != "score_id"}),
    ):
        malformed = mutate(copy.deepcopy(valid_doc))
        try:
            decode_document(LegacyScoreBridge, malformed)
        except DocumentError:
            continue  # correctly refused -- the safe/expected outcome
        negative_findings.append(_mk_finding(BRIDGE_MALFORMED_REF_KIND, label))
    negative = _receipt(BRIDGE_MALFORMED_REF_KIND, 0, "malformed:legacy_score_bridge", "strict_decode",
                        negative_findings, 2, code_hash=code_hash, environment_hash=environment_hash,
                        invert=True)
    return comparison, negative


# --------------------------------------------------------------------------
# L04 -- bridge_mapping_parity / bridge_mapping_negative_control
# --------------------------------------------------------------------------


def build_mapping(clean_score_json: Path, clean_bundle_dir: Path, repository, snapshot_ref, *,
                  model_registry_artifact_refs: tuple[str, ...], request_provenance_refs: tuple[str, ...],
                  code_hash: str, environment_hash: str) -> tuple[ComparisonReceipt, ComparisonReceipt]:
    score_doc = load_score_document(clean_score_json)
    bundle_rows_by_ticker, _manifest = load_legacy_bundle(clean_bundle_dir)
    event_refs = resolve_event_refs(repository, snapshot_ref, _pairs(score_doc, bundle_rows_by_ticker))
    _bridges, findings = _run_bridges(score_doc, bundle_rows_by_ticker, event_refs, snapshot_ref,
                                      score_batch_ref="mapping-probe",
                                      model_registry_artifact_refs=model_registry_artifact_refs,
                                      request_provenance_refs=request_provenance_refs)

    mapping_findings: list[Finding] = []
    if findings.planned_population <= 0:
        mapping_findings.append(_mk_finding(BRIDGE_MAPPING_KIND, "planned_population_empty"))
    if findings.rendered_main_population <= 0 or findings.rendered_ladder_population <= 0:
        mapping_findings.append(_mk_finding(BRIDGE_MAPPING_KIND, "main_ladder_not_counted_separately"))
    if findings.matched_population != findings.compared_population or findings.compared_population <= 0:
        mapping_findings.append(_mk_finding(BRIDGE_MAPPING_KIND, "matched_ne_compared"))
    mapping_categories = {"missing", "unplanned", "duplicate", "identity", "unresolved_event"}
    if any(f.category in mapping_categories for f in findings.findings):
        mapping_findings.append(_mk_finding(BRIDGE_MAPPING_KIND, "unexpected_mapping_finding_on_population"))
    comparison = _receipt(BRIDGE_MAPPING_KIND, 1, "build:full_population", "score_doc.expected_population",
                          mapping_findings, len(score_doc["expected_population"]),
                          code_hash=code_hash, environment_hash=environment_hash)

    # Negative control: corrupt a COPY of the bundle -- a duplicated join key (a display row
    # cloned) and an unmapped event date -- and confirm build_bridges independently reports both.
    corrupted = {ticker: [dict(row) for row in rows] for ticker, rows in bundle_rows_by_ticker.items()}
    tickers_with_rows = [t for t, rows in corrupted.items() if rows]
    assert len(tickers_with_rows) >= 2, "population must span at least two tickers to corrupt independently"
    dup_ticker, unmapped_ticker = tickers_with_rows[0], tickers_with_rows[1]
    corrupted[dup_ticker].append(dict(corrupted[dup_ticker][0]))  # duplicate join key
    corrupted[unmapped_ticker][0] = {**corrupted[unmapped_ticker][0], "event_date": "1999-01-01"}
    _bridges2, corrupt_findings = _run_bridges(score_doc, corrupted, event_refs, snapshot_ref,
                                               score_batch_ref="mapping-probe-corrupt",
                                               model_registry_artifact_refs=model_registry_artifact_refs,
                                               request_provenance_refs=request_provenance_refs)
    categories_seen = {f.category for f in corrupt_findings.findings}
    negative_findings: list[Finding] = []
    if "duplicate" not in categories_seen:
        negative_findings.append(_mk_finding(BRIDGE_MAPPING_NEGATIVE_KIND, "duplicate_key_not_detected"))
    if "unresolved_event" not in categories_seen:
        negative_findings.append(_mk_finding(BRIDGE_MAPPING_NEGATIVE_KIND, "unmapped_event_not_detected"))
    negative = _receipt(BRIDGE_MAPPING_NEGATIVE_KIND, 1, "build:full_population", "build:corrupted_population",
                        negative_findings, 2, code_hash=code_hash, environment_hash=environment_hash,
                        invert=True)
    return comparison, negative


# --------------------------------------------------------------------------
# L05 -- bridge_value_parity / bridge_value_negative_control
# --------------------------------------------------------------------------


def build_value(score_json: Path, bundle_dir: Path, repository, snapshot_ref, *,
                model_registry_artifact_refs: tuple[str, ...],
                request_provenance_refs: tuple[str, ...], code_hash: str,
                environment_hash: str) -> tuple[ComparisonReceipt, ComparisonReceipt]:
    score_doc = load_score_document(score_json)
    bundle, _manifest = load_legacy_bundle(bundle_dir)
    event_refs = resolve_event_refs(repository, snapshot_ref, _pairs(score_doc, bundle))
    _bridges, findings = _run_bridges(score_doc, bundle, event_refs, snapshot_ref,
                                      score_batch_ref="value-probe",
                                            model_registry_artifact_refs=model_registry_artifact_refs,
                                            request_provenance_refs=request_provenance_refs)
    comparison_findings: list[Finding] = []
    if any(f.category == "value" for f in findings.findings) or not findings.ok:
        comparison_findings.append(_mk_finding(BRIDGE_VALUE_KIND, "full_population_value_mismatch"))
    compared = len(score_doc["expected_population"])
    comparison = _receipt(BRIDGE_VALUE_KIND, 1, "score.json:full_population", "bundle:full_population",
                          comparison_findings, compared, code_hash=code_hash, environment_hash=environment_hash)

    corrupted = copy.deepcopy(bundle)
    altered = False
    for rows in corrupted.values():
        for row in rows:
            params = row.get("structure_params")
            if isinstance(params, dict):
                for key, value in params.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        params[key] = value + 0.000001
                        altered = True
                        break
            if altered:
                break
        if altered:
            break
    if not altered:
        raise RuntimeError("full population contains no numeric structure_params field to corrupt")
    _corrupt_bridges, corrupt_findings = _run_bridges(
        score_doc, corrupted, event_refs, snapshot_ref,
        score_batch_ref="value-probe-corrupt",
        model_registry_artifact_refs=model_registry_artifact_refs,
        request_provenance_refs=request_provenance_refs)
    negative_findings: list[Finding] = []
    value_mismatches = [finding for finding in corrupt_findings.findings if finding.category == "value"]
    if not value_mismatches:
        negative_findings.append(_mk_finding(
            BRIDGE_VALUE_NEGATIVE_KIND, "precision_mutation_not_detected"))
    negative = _receipt(BRIDGE_VALUE_NEGATIVE_KIND, 1, "score.json:full_population",
                        "bundle:precision_mutation", negative_findings, 1,
                        code_hash=code_hash, environment_hash=environment_hash, invert=True)
    return comparison, negative


def publish(receipt: ComparisonReceipt, artifact_root: Path, name: str) -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    data = json.dumps(to_document(receipt), indent=2, sort_keys=True).encode()
    path = artifact_root / name
    path.write_bytes(data)
    return {"path": path.name, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-json", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--model-registry-artifact-ref", action="append", required=True, dest="model_refs")
    parser.add_argument("--request-provenance-ref", action="append", required=True, dest="provenance_refs")
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)

    code_hash = source_hash(source_files(ROOT))
    env_hash, _source = _environment_hash(ROOT)
    conn, repository, snapshot_ref = _open_repository(args.catalog, args.store_root, args.snapshot_id)
    try:
        identity, malformed = build_identity(
            args.score_json, args.bundle_dir, repository, snapshot_ref,
            model_registry_artifact_refs=tuple(args.model_refs),
            request_provenance_refs=tuple(args.provenance_refs), code_hash=code_hash, environment_hash=env_hash)
        mapping, mapping_negative = build_mapping(
            args.score_json, args.bundle_dir, repository, snapshot_ref,
            model_registry_artifact_refs=tuple(args.model_refs),
            request_provenance_refs=tuple(args.provenance_refs), code_hash=code_hash, environment_hash=env_hash)
        value, value_negative = build_value(
            args.score_json, args.bundle_dir,
            repository, snapshot_ref, model_registry_artifact_refs=tuple(args.model_refs),
            request_provenance_refs=tuple(args.provenance_refs), code_hash=code_hash, environment_hash=env_hash)
    finally:
        conn.close()

    results = {
        "bridge_identity_parity": (identity, "bridge_identity_parity.json"),
        "bridge_malformed_ref_negative_control": (malformed, "bridge_malformed_ref_negative_control.json"),
        "bridge_mapping_parity": (mapping, "bridge_mapping_parity.json"),
        "bridge_mapping_negative_control": (mapping_negative, "bridge_mapping_negative_control.json"),
        "bridge_value_parity": (value, "bridge_value_parity.json"),
        "bridge_value_negative_control": (value_negative, "bridge_value_negative_control.json"),
    }
    out = {}
    for kind, (receipt, name) in results.items():
        ref = publish(receipt, args.artifact_root, name)
        out[kind] = {**ref, "verdict": receipt.verdict}
    print(json.dumps(out, indent=2))
    want_agree = {"bridge_identity_parity", "bridge_mapping_parity", "bridge_value_parity"}
    ok = all((receipt.verdict == AGREE) == (kind in want_agree) for kind, (receipt, _n) in results.items())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
