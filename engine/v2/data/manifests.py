"""Deterministic identity: ``TableContract``, fragment, dataset version, snapshot.

Phase-2 guide §5.1-§5.2: every ``*_id``/``manifest_hash`` on a Handle or
Record is a value the *builder* computes from a canonical payload, never one
the dataclass derives for itself. ``table_contract_hash`` (P2-1b) is that one
place for ``TableContract``; this slice (P2-2c) adds the three that chain
below it: fragment, dataset version, snapshot.

Each of the three follows the same shape, stated once here rather than per
function:

* an ``*_id`` covers only the payload the guide names for it (contract,
  logical position, content — never an operational fact: no timestamp,
  attempt ID or log ref);
* a ``manifest_hash`` covers the *entire* built document, including
  provenance/parent/evidence the ``*_id`` deliberately excludes, computed with
  that document's own ``manifest_hash`` field removed;
* a payload includes an explicit ``"id"`` family tag (``"fragment_id.v1"`` etc)
  so a fragment id, dataset version id and snapshot id can never collide by
  accident of identical field values across the three kinds.

Judgement calls recorded here (this package's own, not the guide's):

* the fragment payload's object fields are ``object_ref.content_hash`` and
  ``.byte_size`` only — never ``object_id``/``kind`` — since those two are the
  ones a byte-identical republish under a different attempt reproduces
  exactly (``objects.publish_legacy_file``'s own retry test);
* the dataset logical hash is its own streaming-style payload
  (``dataset_logical_rows.v1``: contract, ordered ``(partition_key,
  logical_content_hash)`` pairs) — a sibling to ``objects.logical_partition_hash``
  one level up, not a reuse of its algorithm name;
* ``coverage_receipt_refs``/``availability_evidence_refs``/
  ``finality_receipt_refs`` are sorted *only inside the identity payload*: the
  guide's "reordering meaningful columns or fragments changes identity" is
  about fragment membership order (meaningful: partition sequence), not these
  unordered evidence sets, so two callers citing the same evidence in a
  different order must not mint two identities. The dataclass field itself
  keeps the caller's order — a display/audit concern, not an identity one.
* the three ``*_identity_payload`` helpers are public and exported, mirroring
  ``objects.logical_partition_hash``: a payload/hash helper worth pinning by
  its own permutation test is worth naming.

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts``, ``engine.v2.foundation``, and this package's own
``errors``/``objects`` (for the ``FragmentInspection`` type built one slice
down) — never ``engine.v2.ops`` or legacy ``engine.*``.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from engine.v2.contracts.data import (
    DatasetManifest,
    DatasetVersionRef,
    FragmentRecord,
    FragmentRef,
    KnowledgeMode,
    SnapshotRef,
    TableContract,
    TableContractRef,
)
from engine.v2.data.errors import fail
from engine.v2.data.objects import FragmentInspection
from engine.v2.foundation import CONTENT_HASH_PREFIX, content_hash, to_document

__all__ = [
    "DATASET_LOGICAL_ALGORITHM",
    "dataset_manifest",
    "dataset_version_identity_payload",
    "fragment_identity_payload",
    "fragment_record",
    "fragment_ref",
    "snapshot_identity_payload",
    "snapshot_ref",
    "table_contract_hash",
    "verify_dataset_manifest",
    "verify_fragment_record",
    "verify_snapshot_ref",
]

DATASET_LOGICAL_ALGORITHM = "dataset_logical_rows.v1"
_PLACEHOLDER_HASH = CONTENT_HASH_PREFIX + "0" * 64


def table_contract_hash(contract: TableContract) -> str:
    """``sha256:...`` over ``contract``'s canonical payload, excluding its own hash.

    Column order, key order, and every other field are meaningful (phase-2
    guide §5.1): this hashes ``to_document(contract)`` verbatim except for the
    ``definition_hash`` field itself, so reordering two columns or changing one
    column's unit changes the result, and two calls over the same content are
    byte-identical.
    """
    payload = to_document(contract)
    del payload["definition_hash"]
    return content_hash(payload)


def _derive_id(prefix: str, payload: dict) -> str:
    return prefix + content_hash(payload).removeprefix(CONTENT_HASH_PREFIX)[:32]


# --------------------------------------------------------------------------
# fragment identity (phase-2 guide §5.2)
# --------------------------------------------------------------------------


def fragment_identity_payload(contract_ref: TableContractRef, partition_key: str, *,
                              logical_content_hash: str, object_content_hash: str,
                              byte_size: int, row_count: int,
                              primary_key_min: Sequence, primary_key_max: Sequence,
                              time_min: str | None, time_max: str | None) -> dict:
    """The ``fragment_id.v1`` payload: contract, logical partition, content,
    row count, and bounds — never provenance or an operational fact.
    """
    return {
        "id": "fragment_id.v1",
        "table_contract_ref": to_document(contract_ref),
        "partition_key": partition_key,
        "logical_content_hash": logical_content_hash,
        "object_content_hash": object_content_hash,
        "byte_size": byte_size,
        "row_count": row_count,
        "primary_key_min": list(primary_key_min),
        "primary_key_max": list(primary_key_max),
        "time_min": time_min,
        "time_max": time_max,
    }


def _fragment_payload_from(contract_ref: TableContractRef, *, partition_key: str,
                           logical_content_hash: str, object_content_hash: str, byte_size: int,
                           row_count: int, primary_key_min, primary_key_max,
                           time_min: str | None, time_max: str | None) -> dict:
    return fragment_identity_payload(
        contract_ref, partition_key, logical_content_hash=logical_content_hash,
        object_content_hash=object_content_hash, byte_size=byte_size, row_count=row_count,
        primary_key_min=primary_key_min, primary_key_max=primary_key_max,
        time_min=time_min, time_max=time_max,
    )


def fragment_record(inspection: FragmentInspection, contract_ref: TableContractRef, *,
                    input_receipt_refs: Sequence[str], import_request_hash: str) -> FragmentRecord:
    """Build one ``FragmentRecord`` from a Phase-2c ``FragmentInspection``.

    ``fragment_id`` excludes ``input_receipt_refs``/``import_request_hash``;
    ``manifest_hash`` — computed second, over the built record with its own
    ``manifest_hash`` removed — covers them (this module's docstring).
    """
    if inspection.primary_key_min is None or inspection.primary_key_max is None:
        raise fail("MANIFEST_CORRUPT", "a fragment record requires a non-empty partition")
    payload = _fragment_payload_from(
        contract_ref, partition_key=inspection.partition_key,
        logical_content_hash=inspection.logical_content_hash,
        object_content_hash=inspection.object_ref.content_hash,
        byte_size=inspection.object_ref.byte_size, row_count=inspection.row_count,
        primary_key_min=inspection.primary_key_min, primary_key_max=inspection.primary_key_max,
        time_min=inspection.time_min, time_max=inspection.time_max,
    )
    record = FragmentRecord(
        fragment_id=_derive_id("frag_", payload), manifest_hash=_PLACEHOLDER_HASH,
        object_ref=inspection.object_ref, table_contract_ref=contract_ref,
        partition_key=inspection.partition_key, row_count=inspection.row_count,
        byte_hash=inspection.byte_hash, logical_content_hash=inspection.logical_content_hash,
        primary_key_min=inspection.primary_key_min, primary_key_max=inspection.primary_key_max,
        time_min=inspection.time_min, time_max=inspection.time_max,
        input_receipt_refs=tuple(input_receipt_refs), import_request_hash=import_request_hash,
    )
    return dataclasses.replace(record, manifest_hash=_record_manifest_hash(record))


def _record_manifest_hash(record: FragmentRecord) -> str:
    doc = to_document(record)
    del doc["manifest_hash"]
    return content_hash(doc)


def fragment_ref(record: FragmentRecord) -> FragmentRef:
    """The cheap pinned handle to ``record`` — both its id fields, nothing else."""
    return FragmentRef(fragment_id=record.fragment_id, manifest_hash=record.manifest_hash)


def verify_fragment_record(record: FragmentRecord) -> None:
    """Recompute both of ``record``'s identity fields; raise on any mismatch."""
    payload = _fragment_payload_from(
        record.table_contract_ref, partition_key=record.partition_key,
        logical_content_hash=record.logical_content_hash,
        object_content_hash=record.object_ref.content_hash, byte_size=record.object_ref.byte_size,
        row_count=record.row_count, primary_key_min=record.primary_key_min,
        primary_key_max=record.primary_key_max, time_min=record.time_min, time_max=record.time_max,
    )
    if record.fragment_id != _derive_id("frag_", payload):
        raise fail("MANIFEST_CORRUPT", "fragment_id does not match its record's identity payload")
    if record.manifest_hash != _record_manifest_hash(record):
        raise fail("MANIFEST_CORRUPT", "manifest_hash does not match the fragment record")


# --------------------------------------------------------------------------
# fragment membership — shared by the dataset builder and its verifier
# --------------------------------------------------------------------------


def _check_same_contract(contract_ref: TableContractRef, records: Sequence[FragmentRecord]) -> None:
    for record in records:
        if record.table_contract_ref != contract_ref:
            raise fail("CONTRACT_MISMATCH",
                      "a fragment record's table_contract_ref does not match the dataset's")


def _check_membership_order(records: Sequence[FragmentRecord]) -> None:
    previous: str | None = None
    for record in records:
        if previous is not None:
            if record.partition_key == previous:
                raise fail("MANIFEST_CORRUPT",
                          f"duplicate partition key {record.partition_key!r} in fragment membership")
            if record.partition_key < previous:
                raise fail("MANIFEST_CORRUPT",
                          f"fragment membership is out of partition_key order at {record.partition_key!r}")
        previous = record.partition_key


def _dataset_logical_hash(contract_ref: TableContractRef,
                          partitions: Sequence[tuple[str, str]]) -> str:
    payload = {
        "algorithm": DATASET_LOGICAL_ALGORITHM,
        "table_contract_ref": to_document(contract_ref),
        "partitions": [list(pair) for pair in partitions],
    }
    return content_hash(payload)


# --------------------------------------------------------------------------
# dataset version identity (phase-2 guide §5.2)
# --------------------------------------------------------------------------


def dataset_version_identity_payload(contract_ref: TableContractRef, fragment_ids: Sequence[str], *,
                                     knowledge_mode: KnowledgeMode,
                                     coverage_receipt_refs: Sequence[str],
                                     availability_evidence_refs: Sequence[str]) -> dict:
    """The ``dataset_version_id.v1`` payload: contract, *ordered* fragment
    membership, knowledge mode, and evidence (sorted — see module docstring).
    """
    return {
        "id": "dataset_version_id.v1",
        "table_contract_ref": to_document(contract_ref),
        "fragment_ids": list(fragment_ids),
        "knowledge_mode": knowledge_mode,
        "coverage_receipt_refs": sorted(coverage_receipt_refs),
        "availability_evidence_refs": sorted(availability_evidence_refs),
    }


def dataset_manifest(contract_ref: TableContractRef, records: Sequence[FragmentRecord], *,
                     knowledge_mode: KnowledgeMode, coverage_receipt_refs: Sequence[str],
                     availability_evidence_refs: Sequence[str],
                     parent_dataset_version_id: str | None = None) -> DatasetManifest:
    """Build a complete, ordered ``DatasetManifest`` from ``records``.

    ``records`` must already be in strict ascending ``partition_key`` order
    with no duplicate — this builder never sorts silently (§6 invariant 4:
    every dataset version is a complete logical view).
    """
    _check_same_contract(contract_ref, records)
    _check_membership_order(records)
    if knowledge_mode in ("observed", "attested_stable") and not availability_evidence_refs:
        raise fail("CONTRACT_MISMATCH",
                  f"{knowledge_mode} dataset version requires non-empty availability_evidence_refs")
    refs = tuple(fragment_ref(r) for r in records)
    row_count = sum(r.row_count for r in records)
    logical_hash = _dataset_logical_hash(
        contract_ref, [(r.partition_key, r.logical_content_hash) for r in records])
    payload = dataset_version_identity_payload(
        contract_ref, [r.fragment_id for r in records], knowledge_mode=knowledge_mode,
        coverage_receipt_refs=coverage_receipt_refs,
        availability_evidence_refs=availability_evidence_refs,
    )
    version_ref = DatasetVersionRef(dataset_version_id=_derive_id("dsv_", payload),
                                    table_contract_ref=contract_ref, manifest_hash=_PLACEHOLDER_HASH)
    manifest = DatasetManifest(
        dataset_version_ref=version_ref, logical_content_hash=logical_hash, row_count=row_count,
        parent_dataset_version_id=parent_dataset_version_id, fragment_refs=refs,
        coverage_receipt_refs=tuple(coverage_receipt_refs), knowledge_mode=knowledge_mode,
        availability_evidence_refs=tuple(availability_evidence_refs),
    )
    final_ref = dataclasses.replace(version_ref, manifest_hash=_manifest_hash(manifest))
    return dataclasses.replace(manifest, dataset_version_ref=final_ref)


def _manifest_hash(manifest: DatasetManifest) -> str:
    doc = to_document(manifest)
    del doc["dataset_version_ref"]["manifest_hash"]
    return content_hash(doc)


def verify_dataset_manifest(manifest: DatasetManifest, records: Sequence[FragmentRecord]) -> None:
    """Recompute ``manifest`` end to end against ``records``; raise on any mismatch."""
    contract_ref = manifest.dataset_version_ref.table_contract_ref
    _check_same_contract(contract_ref, records)
    _check_membership_order(records)
    expected_refs = tuple(fragment_ref(r) for r in records)
    if manifest.fragment_refs != expected_refs:
        raise fail("MANIFEST_CORRUPT", "dataset fragment membership does not match its records")
    row_count = sum(r.row_count for r in records)
    if manifest.row_count != row_count:
        raise fail("MANIFEST_CORRUPT",
                  "dataset row_count does not equal the sum of its fragment row counts")
    logical_hash = _dataset_logical_hash(
        contract_ref, [(r.partition_key, r.logical_content_hash) for r in records])
    if manifest.logical_content_hash != logical_hash:
        raise fail("MANIFEST_CORRUPT", "dataset logical_content_hash does not match its fragments")
    payload = dataset_version_identity_payload(
        contract_ref, [r.fragment_id for r in records], knowledge_mode=manifest.knowledge_mode,
        coverage_receipt_refs=manifest.coverage_receipt_refs,
        availability_evidence_refs=manifest.availability_evidence_refs,
    )
    if manifest.dataset_version_ref.dataset_version_id != _derive_id("dsv_", payload):
        raise fail("MANIFEST_CORRUPT", "dataset_version_id does not match the dataset manifest payload")
    if manifest.dataset_version_ref.manifest_hash != _manifest_hash(manifest):
        raise fail("MANIFEST_CORRUPT", "manifest_hash does not match the dataset manifest")


# --------------------------------------------------------------------------
# snapshot identity (phase-2 guide §5.2, §3.3)
# --------------------------------------------------------------------------


def snapshot_identity_payload(table_version_ids: Mapping[str, str], *, calendar_version: str,
                              source_priority_version: str, finality_receipt_refs: Sequence[str],
                              knowledge_mode_by_table: Mapping[str, KnowledgeMode]) -> dict:
    """The ``snapshot_id.v1`` payload: table-to-version map, calendar and
    source-priority versions, finality evidence (sorted), and knowledge modes.
    """
    return {
        "id": "snapshot_id.v1",
        "table_versions": dict(table_version_ids),
        "calendar_version": calendar_version,
        "source_priority_version": source_priority_version,
        "finality_receipt_refs": sorted(finality_receipt_refs),
        "knowledge_mode_by_table": dict(knowledge_mode_by_table),
    }


def snapshot_ref(table_versions: Mapping[str, DatasetManifest], *, calendar_version: str,
                 source_priority_version: str, finality_receipt_refs: Sequence[str],
                 parent_snapshot_id: str | None = None) -> SnapshotRef:
    """Build a ``SnapshotRef`` over one ``DatasetManifest`` per table.

    Knowledge modes are read from the manifests, never passed separately: a
    caller cannot relabel a table's mode without also changing its manifest.
    """
    version_refs = {name: manifest.dataset_version_ref for name, manifest in table_versions.items()}
    modes: dict[str, KnowledgeMode] = {name: manifest.knowledge_mode
                                       for name, manifest in table_versions.items()}
    payload = snapshot_identity_payload(
        {name: ref.dataset_version_id for name, ref in version_refs.items()},
        calendar_version=calendar_version, source_priority_version=source_priority_version,
        finality_receipt_refs=finality_receipt_refs, knowledge_mode_by_table=modes,
    )
    ref = SnapshotRef(
        snapshot_id=_derive_id("snap_", payload), manifest_hash=_PLACEHOLDER_HASH,
        parent_snapshot_id=parent_snapshot_id, table_versions=version_refs,
        calendar_version=calendar_version, source_priority_version=source_priority_version,
        finality_receipt_refs=tuple(finality_receipt_refs), knowledge_mode_by_table=modes,
    )
    return dataclasses.replace(ref, manifest_hash=_snapshot_manifest_hash(ref))


def _snapshot_manifest_hash(ref: SnapshotRef) -> str:
    doc = to_document(ref)
    del doc["manifest_hash"]
    return content_hash(doc)


def verify_snapshot_ref(ref: SnapshotRef, manifests: Mapping[str, DatasetManifest]) -> None:
    """Recompute ``ref`` end to end against ``manifests``; raise on any mismatch."""
    if set(ref.table_versions) != set(manifests):
        raise fail("MANIFEST_CORRUPT", "snapshot table_versions keys do not match the given manifests")
    for name, manifest in manifests.items():
        if ref.table_versions[name] != manifest.dataset_version_ref:
            raise fail("MANIFEST_CORRUPT", f"snapshot table {name!r} does not point at its manifest")
        if ref.knowledge_mode_by_table[name] != manifest.knowledge_mode:
            raise fail("MANIFEST_CORRUPT",
                      f"snapshot knowledge mode for {name!r} does not match its manifest")
    payload = snapshot_identity_payload(
        {name: dvr.dataset_version_id for name, dvr in ref.table_versions.items()},
        calendar_version=ref.calendar_version, source_priority_version=ref.source_priority_version,
        finality_receipt_refs=ref.finality_receipt_refs,
        knowledge_mode_by_table=ref.knowledge_mode_by_table,
    )
    if ref.snapshot_id != _derive_id("snap_", payload):
        raise fail("MANIFEST_CORRUPT", "snapshot_id does not match its identity payload")
    if ref.manifest_hash != _snapshot_manifest_hash(ref):
        raise fail("MANIFEST_CORRUPT", "manifest_hash does not match the snapshot ref")
