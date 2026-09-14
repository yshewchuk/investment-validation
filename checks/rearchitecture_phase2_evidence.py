#!/usr/bin/env python3
"""Strict validator for the Phase 2 gate's private evidence document.

Phase-2 guide §12.1 quotes the shape::

    PHASE2_EVIDENCE_V1 = "phase2_evidence.v1.0"

    Phase2Evidence:
      schema_version, code_hash, environment_hash
      snapshot_ref, legacy_snapshot_object_ref
      table_contract_mapping_hash
      import_receipt_refs, fault_matrix_ref
      dependency_plan_refs, comparison_receipt_ref, render_comparison_receipt_ref,
      rollback_receipt_ref, corpus_comparison_receipt_ref (task P2-C01, D14)
      expected_population, supported_population, compared_population
      authority_mode: shadow

``comparison_receipt_ref`` and ``render_comparison_receipt_ref`` are two
DIFFERENT receipts, not one shared field: D15 is legacy-vs-adapter SCORE
parity and D19 is v2-vs-legacy RENDER BUNDLE parity, over different
populations and different stage graphs. One field would let D15's receipt
silently stand in for D19's. ``corpus_comparison_receipt_ref`` (task P2-C01
decision 7) is a THIRD, again distinct, receipt: D14 supervised scoring of
the Tier-0 corpus. Schema stays ``phase2_evidence.v1.0`` -- this is an
optional-field addition and no evidence document has ever been committed, so
there is no v1.0 consumer to break by adding an optional field to it.

"The evidence validator verifies every referenced artifact, requires all
D01-D20 rows, rejects a code/environment mismatch, rejects zero or collapsed
populations, and requires the comparison verdict agree. It does not accept a
summary boolean in place of the referenced receipts." -- and, task P2-C01
(Phase 2 review closeout): every ref DECODES STRICTLY into its real contract
(a plain ``{"verdict": "agree"}`` dict is refused, not just tolerated),
receipts are BOUND to this evidence's own code/environment/snapshot identity
rather than merely coexisting with matching top-level fields, score and
render receipts must be different artifacts of different declared kinds,
every population drop is explained by an itemized exclusion list, and the
rollback/fault evidence is checked for real content, not just presence.

This module never produces real evidence: it only checks a document someone
else hands it. Reference shape (documented here because ``ArtifactStore``'s
content-addressed ``objects/<xx>/<hash>`` layout does not fit a manifest whose
refs point at diverse artifact kinds a coordinator did not itself publish)::

    {"path": "<relative path under --artifact-root>", "content_hash": "sha256:<hex>"}

A list-valued field (``import_receipt_refs``, ``dependency_plan_refs``) is a
list of these. A bare boolean anywhere a ref belongs is refused outright: it
is exactly the "summary boolean in place of the referenced receipts" the
guide names.

``validate_evidence`` returns ``(findings, field_ok, document_ok)``.
``document_ok`` covers only whole-document identity (schema/code/environment
hash, authority_mode) -- a failure there means NOTHING in the manifest can be
trusted. ``field_ok`` is per declared field (a ref resolves, decodes, a
receipt's verdict/kind/bindings agree, the populations are in order): a
problem with ONE field (say ``fault_matrix_ref``) must not silently invalidate
a DIFFERENT field's row (D15's populations, D19's render receipt). The gate
combines ``document_ok`` and the specific fields a row's ``evidence_fields``
name to decide that row, never the other rows' fields.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from checks.tier0_corpus import DEFAULT_CORPUS
from engine.v2.contracts.data import (
    DependencyPlan,
    ObjectRef,
    RollbackReceipt,
    SnapshotImportReceipt,
    SnapshotRef,
)
from engine.v2.data.documents import decode_document
from engine.v2.diagnosis.receipt import AGREE, ComparisonReceipt
from engine.v2.foundation import DocumentError

PHASE2_EVIDENCE_V1 = "phase2_evidence.v1.0"

#: The two comparison kinds ``compare_records`` callers must declare (task
#: P2-C01 decision 3) -- ``score_record_parity`` is ``compare_records``' own
#: default ``comparison_kind`` already, so D15 costs its producer nothing new.
SCORE_PARITY_KIND = "score_record_parity"
RENDER_PARITY_KIND = "render_bundle_parity"
#: D14 (task P2-C01 decision 7): supervised scoring of the Tier-0 corpus,
#: a THIRD kind distinct from both D15 (adapter parity) and D19 (render).
CORPUS_PARITY_KIND = "corpus_score_parity"

#: D14 corpus-snapshot binding (task D14 review): the machine-checkable
#: artifact ``checks/rearchitecture_phase2_corpus_parity.py``'s ``compare``/
#: ``control-missing-analogs`` publish next to the corpus receipt and point
#: the receipt's own ``envelope.diagnostic_ref`` at, replacing the free-text
#: stuffing this schema version supersedes. No contract type in
#: ``engine/v2/contracts`` (bare hand-validated JSON, like ``fault_matrix_ref``
#: below) -- the receipt schema and ``engine/v2/diagnosis`` are unchanged.
CORPUS_SNAPSHOT_BINDING_V1 = "corpus_snapshot_binding.v1.0"

#: Single-artifact reference fields whose bytes decode to a known contract.
#: Excludes ``fault_matrix_ref`` (no contract type -- checked directly against
#: the §7.3 fault points below) and ``table_contract_mapping_hash`` (a bare
#: hash of reviewed content, not a pointer to stored bytes).
DECODE_CLASS: dict[str, type] = {
    "snapshot_ref": SnapshotRef,
    "legacy_snapshot_object_ref": ObjectRef,
    "comparison_receipt_ref": ComparisonReceipt,
    "render_comparison_receipt_ref": ComparisonReceipt,
    "corpus_comparison_receipt_ref": ComparisonReceipt,
    "rollback_receipt_ref": RollbackReceipt,
}
LIST_DECODE_CLASS: dict[str, type] = {
    "import_receipt_refs": SnapshotImportReceipt,
    "dependency_plan_refs": DependencyPlan,
}
REF_FIELDS = tuple(DECODE_CLASS) + ("fault_matrix_ref",)
LIST_REF_FIELDS = tuple(LIST_DECODE_CLASS)
#: Refs whose bytes must additionally decode to a comparison receipt with
#: verdict "agree" -- D15's score-parity receipt, D19's render-bundle-parity
#: receipt, and D14's corpus receipt, each checked independently.
VERDICT_REF_FIELDS = ("comparison_receipt_ref", "render_comparison_receipt_ref",
                      "corpus_comparison_receipt_ref")
COMPARISON_KIND_BY_FIELD = {
    "comparison_receipt_ref": SCORE_PARITY_KIND,
    "render_comparison_receipt_ref": RENDER_PARITY_KIND,
    "corpus_comparison_receipt_ref": CORPUS_PARITY_KIND,
}
POPULATION_FIELDS = ("expected_population", "supported_population", "compared_population")
AUTHORITY_MODE = "shadow"

#: The fifteen §7.3 commit-boundary fault points a real D11/D16 fault matrix
#: must cover (task P2-C01 decision 5): five object-side, ten catalog-side,
#: named verbatim in ``tests/test_v2_data_atomicity.py`` (``OBJECT_SIDE_POINTS``
#: / ``CATALOG_SIDE_POINTS``) and fired as string literals from
#: ``engine/v2/data/objects.py``, ``engine/v2/foundation/artifacts.py`` and
#: ``engine/v2/data/catalog.py``. Not re-imported from the test module (checks
#: importing tests would be the wrong direction); pinned here as the one
#: place this validator's own notion of "every fault point" lives.
FAULT_POINTS = (
    "before_copy", "during_copy", "copied", "linked", "during_inspection",
    "before_transaction", "after_contracts", "after_objects", "after_fragments",
    "after_dataset_versions", "after_memberships", "after_snapshot",
    "after_snapshot_tables", "before_head_update", "before_commit",
)
FAULT_OUTCOMES = ("old_head", "new_head")


def _resolve(ref: Any, artifact_root: Path, findings: list, field: str) -> bytes | None:
    if isinstance(ref, bool):
        findings.append({"code": "SUMMARY_BOOLEAN_REFUSED", "field": field})
        return None
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) \
            or not isinstance(ref.get("content_hash"), str):
        findings.append({"code": "ARTIFACT_MISSING", "field": field})
        return None
    path = (artifact_root / ref["path"]).resolve()
    try:
        path.relative_to(artifact_root.resolve())
    except ValueError:
        findings.append({"code": "ARTIFACT_MISSING", "field": field})
        return None
    if not path.is_file():
        findings.append({"code": "ARTIFACT_MISSING", "field": field})
        return None
    data = path.read_bytes()
    actual = "sha256:" + hashlib.sha256(data).hexdigest()
    if actual != ref["content_hash"]:
        findings.append({"code": "ARTIFACT_HASH_MISMATCH", "field": field})
        return None
    return data


def _check_refs(evidence: dict, artifact_root: Path, findings: list,
                field_ok: dict[str, bool]) -> tuple[dict[str, bytes], dict[str, list[bytes]]]:
    """Resolve every declared ref's bytes; strict decode happens separately."""
    resolved: dict[str, bytes] = {}
    for field in REF_FIELDS:
        if field not in evidence or evidence[field] is None:
            continue
        data = _resolve(evidence[field], artifact_root, findings, field)
        field_ok[field] = data is not None
        if data is not None:
            resolved[field] = data
    resolved_lists: dict[str, list[bytes]] = {}
    for field in LIST_REF_FIELDS:
        if field not in evidence or evidence[field] is None:
            continue
        items = evidence[field]
        if isinstance(items, bool) or not isinstance(items, list):
            findings.append({"code": "SUMMARY_BOOLEAN_REFUSED", "field": field})
            field_ok[field] = False
            continue
        pieces = [_resolve(item, artifact_root, findings, f"{field}[{i}]")
                 for i, item in enumerate(items)]
        field_ok[field] = bool(items) and all(p is not None for p in pieces)
        resolved_lists[field] = [p for p in pieces if p is not None]
    return resolved, resolved_lists


def _decode(cls: type, data: bytes, findings: list, field: str) -> Any:
    """Strict decode: real JSON, then ``decode_document`` -- unknown fields,
    bad enums, non-finite numbers and a newer-than-supported schema version
    are all refused (task P2-C01 decision 1). A plain dict missing required
    fields, or one carrying only a summary key like ``{"verdict": "agree"}``,
    fails here rather than silently passing as its shallow shape.
    """
    try:
        doc = json.loads(data)
    except ValueError:
        findings.append({"code": "ARTIFACT_SHAPE_INVALID", "field": field, "reason": "invalid_json"})
        return None
    try:
        return decode_document(cls, doc)
    except DocumentError as exc:
        findings.append({"code": "ARTIFACT_SHAPE_INVALID", "field": field, "reason": exc.code})
        return None


def _decode_typed_refs(resolved: dict[str, bytes], resolved_lists: dict[str, list[bytes]],
                       findings: list, field_ok: dict[str, bool]
                       ) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    decoded: dict[str, Any] = {}
    for field, cls in DECODE_CLASS.items():
        data = resolved.get(field)
        if data is None:
            continue
        obj = _decode(cls, data, findings, field)
        if obj is None:
            field_ok[field] = False
        else:
            decoded[field] = obj
    decoded_lists: dict[str, list[Any]] = {}
    for field, cls in LIST_DECODE_CLASS.items():
        items = resolved_lists.get(field)
        if items is None:
            continue
        objs = [_decode(cls, data, findings, f"{field}[{i}]") for i, data in enumerate(items)]
        if any(obj is None for obj in objs):
            field_ok[field] = False
        decoded_lists[field] = [obj for obj in objs if obj is not None]
    return decoded, decoded_lists


def _check_fault_matrix(data: bytes | None, findings: list, field_ok: dict[str, bool]) -> None:
    if data is None:
        return
    try:
        points = json.loads(data)
    except ValueError:
        points = None
    by_point: dict[str, Any] = {}
    shape_ok = isinstance(points, list)
    if shape_ok:
        for entry in points:
            valid_entry = (isinstance(entry, dict) and isinstance(entry.get("point"), str)
                          and entry.get("outcome") in FAULT_OUTCOMES
                          and isinstance(entry.get("verified_objects"), bool))
            if not valid_entry:
                shape_ok = False
                break
            by_point[entry["point"]] = entry
    missing = [p for p in FAULT_POINTS if p not in by_point]
    if not shape_ok or missing:
        findings.append({"code": "FAULT_MATRIX_INCOMPLETE", "missing": missing})
        field_ok["fault_matrix_ref"] = False


def _check_populations(evidence: dict, findings: list, field_ok: dict[str, bool]) -> None:
    values = {}
    for field in POPULATION_FIELDS:
        value = evidence.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            findings.append({"code": "POPULATION_EMPTY", "field": field})
            for f in POPULATION_FIELDS:
                field_ok[f] = False
            return
        values[field] = value
    ordered = (values["compared_population"] <= values["supported_population"]
              <= values["expected_population"])
    if not ordered:
        findings.append({"code": "POPULATION_COLLAPSED"})
    for f in POPULATION_FIELDS:
        field_ok[f] = ordered


def _check_verdicts(decoded: dict[str, Any], findings: list, field_ok: dict[str, bool]) -> None:
    for field in VERDICT_REF_FIELDS:
        receipt = decoded.get(field)
        if receipt is not None and receipt.verdict != AGREE:
            findings.append({"code": "VERDICT_NOT_AGREE", "field": field})
            field_ok[field] = False


def _check_receipt_kinds(evidence: dict, decoded: dict[str, Any], findings: list,
                         field_ok: dict[str, bool]) -> None:
    """D15/D19 (and D14) never share one artifact, and each declares its own
    kind (task P2-C01 decision 3): swapped or reused receipts are refused.
    """
    score_ref, render_ref = evidence.get("comparison_receipt_ref"), evidence.get(
        "render_comparison_receipt_ref")
    if (isinstance(score_ref, dict) and isinstance(render_ref, dict)
            and score_ref.get("content_hash") == render_ref.get("content_hash")):
        findings.append({"code": "RECEIPT_KIND_MISMATCH", "field": "render_comparison_receipt_ref",
                         "reason": "same_artifact_as_comparison_receipt_ref"})
        field_ok["render_comparison_receipt_ref"] = False
        field_ok["comparison_receipt_ref"] = False
    for field, expected_kind in COMPARISON_KIND_BY_FIELD.items():
        receipt = decoded.get(field)
        if receipt is not None and receipt.comparison_kind != expected_kind:
            findings.append({"code": "RECEIPT_KIND_MISMATCH", "field": field})
            field_ok[field] = False


def _load_corpus_binding(receipt: Any, artifact_root: Path, findings: list,
                         field_ok: dict[str, bool]) -> dict | None:
    """Strictly resolve and decode the ``corpus_snapshot_binding.v1.0``
    artifact D14's ``envelope.diagnostic_ref`` points at -- the structured
    replacement for the free-text ``legacy_snapshot_hash=...;matches_corpus_
    snapshot=...`` stuffing. ``diagnostic_ref`` carries the reference dict
    itself, JSON-encoded (the field is a plain ``str`` on ``Envelope``, which
    this module does not touch): ``{"path": ..., "content_hash": ...}``, the
    SAME reference shape every other evidence field uses.

    Any failure -- missing/unparseable ``diagnostic_ref``, an unresolvable or
    hash-mismatched artifact, or a malformed/legacy-free-text document --
    marks ``corpus_comparison_receipt_ref`` not-ok and returns ``None``.
    """
    field = "corpus_comparison_receipt_ref"
    diagnostic_ref = receipt.envelope.diagnostic_ref
    ref = None
    if isinstance(diagnostic_ref, str):
        try:
            candidate = json.loads(diagnostic_ref)
        except ValueError:
            candidate = None
        if (isinstance(candidate, dict) and isinstance(candidate.get("path"), str)
                and isinstance(candidate.get("content_hash"), str)):
            ref = candidate
    if ref is None:
        # Covers both an absent diagnostic_ref AND the pre-D14-review
        # free-text form (``legacy_snapshot_hash=...;...``), which is not
        # valid JSON and so never becomes a dict here.
        findings.append({"code": "CORPUS_BINDING_MISSING", "field": field})
        field_ok[field] = False
        return None
    path = (artifact_root / ref["path"]).resolve()
    try:
        path.relative_to(artifact_root.resolve())
        exists = path.is_file()
    except ValueError:
        exists = False
    if not exists:
        findings.append({"code": "CORPUS_BINDING_MISSING", "field": field})
        field_ok[field] = False
        return None
    data = path.read_bytes()
    actual = "sha256:" + hashlib.sha256(data).hexdigest()
    if actual != ref["content_hash"]:
        findings.append({"code": "CORPUS_BINDING_HASH_MISMATCH", "field": field})
        field_ok[field] = False
        return None
    try:
        doc = json.loads(data)
    except ValueError:
        doc = None
    valid = (
        isinstance(doc, dict) and doc.get("schema_version") == CORPUS_SNAPSHOT_BINDING_V1
        and isinstance(doc.get("corpus_version"), str)
        and (doc.get("corpus_snapshot_hash") is None or isinstance(doc.get("corpus_snapshot_hash"), str))
        and (doc.get("source_snapshot_hash") is None or isinstance(doc.get("source_snapshot_hash"), str))
        and isinstance(doc.get("control"), bool)
        and (doc.get("control_drop_ticker") is None or isinstance(doc.get("control_drop_ticker"), str))
    )
    if not valid:
        findings.append({"code": "CORPUS_BINDING_SHAPE_INVALID", "field": field})
        field_ok[field] = False
        return None
    return doc


def _corpus_index_snapshot(corpus_root: Path, corpus_version: str) -> str | None:
    """The named corpus version's OWN declared ``INDEX.json`` ``snapshot`` --
    re-derived here, never trusted from the binding, so a forged binding
    cannot claim agreement with a corpus it does not match. ``corpus_version
    == ""`` is the bare/unversioned layout (``corpus_root/INDEX.json`` IS the
    corpus -- ``checks.tier0_corpus.resolve_corpus``'s own other case)."""
    base = corpus_root if not corpus_version else corpus_root / corpus_version
    try:
        return json.loads((base / "INDEX.json").read_text()).get("snapshot")
    except (ValueError, OSError):
        return None


def _check_corpus_binding(decoded: dict[str, Any], artifact_root: Path, corpus_root: Path,
                          findings: list, field_ok: dict[str, bool]) -> None:
    """D14 (task D14 review): the corpus receipt's diagnostic_ref must bind a
    real, matching corpus-snapshot binding -- refusing a corpus parity
    receipt scored against a legacy store that had since moved from the
    frozen corpus, a receipt that never declares the binding at all, and a
    control receipt (a SEPARATE piece of evidence, D14's own control) offered
    in D14's place.
    """
    field = "corpus_comparison_receipt_ref"
    receipt = decoded.get(field)
    if receipt is None:
        return
    binding = _load_corpus_binding(receipt, artifact_root, findings, field_ok)
    if binding is None:
        return
    if binding.get("control"):
        findings.append({"code": "CORPUS_BINDING_IS_CONTROL", "field": field})
        field_ok[field] = False
        return
    corpus_hash = binding.get("corpus_snapshot_hash")
    source_hash = binding.get("source_snapshot_hash")
    if corpus_hash != source_hash:
        findings.append({"code": "CORPUS_BINDING_SOURCE_MISMATCH", "field": field})
        field_ok[field] = False
    index_snapshot = _corpus_index_snapshot(Path(corpus_root), binding.get("corpus_version") or "")
    if corpus_hash != index_snapshot:
        findings.append({"code": "CORPUS_BINDING_INDEX_MISMATCH", "field": field})
        field_ok[field] = False


def _check_bindings(decoded: dict[str, Any], snapshot_ref_obj: Any, findings: list,
                    field_ok: dict[str, bool], *, code_hash: str, environment_hash: str) -> None:
    """Every comparison receipt binds ITSELF to the evidence's own code,
    environment and snapshot identity (task P2-C01 decision 2) -- a receipt
    that merely coexists with a manifest claiming fresh values is not bound
    to them.
    """
    for field in VERDICT_REF_FIELDS:
        receipt = decoded.get(field)
        if receipt is None:
            continue
        env = receipt.envelope
        if env.code_hash != code_hash:
            findings.append({"code": "CODE_HASH_MISMATCH", "field": field})
            field_ok[field] = False
        if env.environment_hash != environment_hash:
            findings.append({"code": "ENVIRONMENT_MISMATCH", "field": field})
            field_ok[field] = False
        if snapshot_ref_obj is not None and (
                env.snapshot_id != snapshot_ref_obj.snapshot_id
                or env.snapshot_manifest_hash != snapshot_ref_obj.manifest_hash):
            findings.append({"code": "SNAPSHOT_BINDING_MISMATCH", "field": field})
            field_ok[field] = False


def _check_import_receipts(import_receipts: list[Any], snapshot_ref_obj: Any, findings: list,
                           field_ok: dict[str, bool]) -> None:
    """Import receipts must be committed and name the evidence's own
    snapshot (task P2-C01 decision 2, last line)."""
    if snapshot_ref_obj is None or not import_receipts:
        return
    resulting_ids = {r.resulting_head_snapshot_id for r in import_receipts}
    all_committed = all(r.status == "committed" for r in import_receipts)
    if not all_committed or snapshot_ref_obj.snapshot_id not in resulting_ids:
        findings.append({"code": "SNAPSHOT_BINDING_MISMATCH", "field": "import_receipt_refs"})
        field_ok["import_receipt_refs"] = False


def _check_population_binding(evidence: dict, receipt: Any, findings: list,
                              field_ok: dict[str, bool]) -> None:
    """D15's declared populations must equal the receipt's own counts, and
    every drop the receipt reports must be itemized (task P2-C01 decision 4).
    """
    if receipt is None:
        return
    pop = receipt.population
    for ev_field, value in (("expected_population", pop.expected),
                            ("supported_population", pop.supported),
                            ("compared_population", pop.compared)):
        # Only compare against the receipt when the evidence's OWN field
        # already passed `_check_populations` (present, positive, ordered):
        # a field that already failed that basic check must not also emit
        # this separate receipt-mismatch code -- one field, one code.
        if not field_ok.get(ev_field, False):
            continue
        if evidence.get(ev_field) != value:
            findings.append({"code": "POPULATION_COLLAPSED", "field": ev_field,
                             "reason": "receipt_mismatch"})
            field_ok[ev_field] = False
    for stage, drop in (("expected_to_supported", pop.expected - pop.supported),
                        ("supported_to_compared", pop.supported - pop.compared)):
        if drop <= 0:
            continue
        entries = [e for e in pop.excluded if isinstance(e, dict) and e.get("stage") == stage]
        keys = [e.get("key") for e in entries]
        explained = (len(entries) == drop and len(set(keys)) == len(keys)
                    and all(isinstance(k, str) and isinstance(e.get("reason"), str)
                            for k, e in zip(keys, entries)))
        if not explained:
            findings.append({"code": "POPULATION_UNEXPLAINED", "stage": stage})
            for f in POPULATION_FIELDS:
                field_ok[f] = False


def _check_rollback(receipt: Any, import_receipts: list[Any], findings: list,
                    field_ok: dict[str, bool]) -> None:
    """A real head move: a generation increase, both snapshots named in
    import receipts (task P2-C01 decision 5)."""
    if receipt is None:
        return
    named = {r.resulting_head_snapshot_id for r in import_receipts}
    valid = (receipt.resulting_generation > receipt.prior_generation
             and receipt.prior_snapshot_id in named
             and receipt.resulting_snapshot_id in named)
    if not valid:
        findings.append({"code": "ROLLBACK_EVIDENCE_INVALID", "field": "rollback_receipt_ref"})
        field_ok["rollback_receipt_ref"] = False


def validate_evidence(evidence: dict, *, artifact_root: Path,
                      code_hash: str, environment_hash: str,
                      corpus_root: Path | None = None,
                      ) -> tuple[list[dict], dict[str, bool], bool]:
    """Every finding the strict ``phase2_evidence.v1.0`` document can produce.

    Returns ``(findings, field_ok, document_ok)`` -- see the module docstring
    for what each covers. ``code_hash``/``environment_hash`` are the CURRENT
    tree's values, computed the same way the gate computes them; a mismatch
    means the evidence was produced against a different working tree or
    interpreter/library set. ``corpus_root``: the base directory D14's
    ``corpus_comparison_receipt_ref`` binding is checked against (default
    ``checks.tier0_corpus.DEFAULT_CORPUS``) -- a caller validating evidence
    built against a different (e.g. synthetic test) corpus passes its own.
    """
    corpus_root = Path(corpus_root) if corpus_root is not None else DEFAULT_CORPUS
    findings: list[dict] = []
    if not isinstance(evidence, dict) or evidence.get("schema_version") != PHASE2_EVIDENCE_V1:
        findings.append({"code": "CODE_HASH_MISMATCH", "field": "schema_version"})
        return findings, {}, False
    document_ok = True
    if evidence.get("code_hash") != code_hash:
        findings.append({"code": "CODE_HASH_MISMATCH", "field": "code_hash"})
        document_ok = False
    if evidence.get("environment_hash") != environment_hash:
        findings.append({"code": "CODE_HASH_MISMATCH", "field": "environment_hash"})
        document_ok = False
    if evidence.get("authority_mode") != AUTHORITY_MODE:
        findings.append({"code": "AUTHORITY_NOT_SHADOW"})
        document_ok = False

    field_ok: dict[str, bool] = {}
    resolved, resolved_lists = _check_refs(evidence, artifact_root, findings, field_ok)
    decoded, decoded_lists = _decode_typed_refs(resolved, resolved_lists, findings, field_ok)
    imports = decoded_lists.get("import_receipt_refs", [])
    snapshot_ref_obj = decoded.get("snapshot_ref")

    _check_fault_matrix(resolved.get("fault_matrix_ref"), findings, field_ok)
    _check_populations(evidence, findings, field_ok)
    _check_verdicts(decoded, findings, field_ok)
    _check_receipt_kinds(evidence, decoded, findings, field_ok)
    _check_corpus_binding(decoded, artifact_root, corpus_root, findings, field_ok)
    _check_bindings(decoded, snapshot_ref_obj, findings, field_ok,
                    code_hash=code_hash, environment_hash=environment_hash)
    _check_import_receipts(imports, snapshot_ref_obj, findings, field_ok)
    _check_population_binding(evidence, decoded.get("comparison_receipt_ref"), findings, field_ok)
    _check_rollback(decoded.get("rollback_receipt_ref"), imports, findings, field_ok)
    return findings, field_ok, document_ok
