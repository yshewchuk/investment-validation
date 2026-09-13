"""D03: content-derived identity for fragments, dataset versions, snapshots.

Phase-2 guide §5.2, §11, §12 (D03). ``engine.v2.data.manifests`` builds
``FragmentRecord``/``DatasetManifest``/``SnapshotRef`` from their inputs and
verifies them against tampering; this test drives that module only, plus the
real ``securities`` ``TableContract`` from ``build_legacy_mapping()`` for the
one end-to-end case.

Two kinds of fixture:

* the end-to-end case publishes real synthetic Parquet through
  ``objects.inspect_fragment``, exactly like ``tests/test_v2_data_objects.py``;
* every other test builds ``FragmentInspection``/``FragmentRecord`` objects
  directly from deterministic fake ``sha256:`` hashes (``_fake_hash``) — the
  identity mechanics under test do not care whether a hash came from a real
  Parquet file, and skipping the file I/O keeps this module fast and each
  fixture's one differing field obvious at the call site.
"""
from __future__ import annotations

import dataclasses
import hashlib
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts.data import (  # noqa: E402
    DatasetManifest,
    FragmentRecord,
    ObjectRef,
    SnapshotRef,
    TableContract,
    TableContractRef,
)
from engine.v2.data import manifests  # noqa: E402
from engine.v2.data.documents import decode_document  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.legacy_adapter import build_legacy_mapping  # noqa: E402
from engine.v2.data.objects import FragmentInspection, inspect_fragment  # noqa: E402
from engine.v2.foundation import ArtifactStore, content_hash, to_document  # noqa: E402

_ARROW_TYPES = {"string": pa.string(), "float64": pa.float64(), "int64": pa.int64(),
                "bool": pa.bool_(), "timestamp[ns]": pa.timestamp("ns")}

_MAPPING = build_legacy_mapping()


def _contract(name: str) -> TableContract:
    return decode_document(TableContract, _MAPPING["tables"][name])


def _contract_ref(contract: TableContract) -> TableContractRef:
    return TableContractRef(contract_id=contract.contract_id, definition_hash=contract.definition_hash)


_SEC_CONTRACT = _contract("securities")
_SEC_REF = _contract_ref(_SEC_CONTRACT)


def _fake_hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


RECEIPT_A = _fake_hash("receipt-a")
RECEIPT_B = _fake_hash("receipt-b")
IRH_A = _fake_hash("import-request-a")
IRH_B = _fake_hash("import-request-b")


# --------------------------------------------------------------------------
# synthetic FragmentInspection/FragmentRecord fixtures (no parquet I/O)
# --------------------------------------------------------------------------


def _inspection_for(year: str) -> FragmentInspection:
    """A plausible ``FragmentInspection`` for partition ``year``, all fake hashes.

    ``time_min``/``time_max`` are ``None`` throughout, matching the real
    ``securities`` contract (no ``observation_time_column``) rather than
    exercising the timestamp-format path — that belongs to ``objects``' own
    tests.
    """
    object_hash = _fake_hash(f"object-{year}")
    return FragmentInspection(
        object_ref=ObjectRef(kind="parquet_fragment", object_id="art_" + hashlib.sha256(year.encode()).hexdigest()[:32],
                             content_hash=object_hash, byte_size=100 + int(year)),
        partition_key=year, row_count=3, byte_hash=object_hash,
        logical_content_hash=_fake_hash(f"logical-{year}"),
        primary_key_min=("AAA", int(year)), primary_key_max=("CCC", int(year)),
        time_min=None, time_max=None,
    )


def _base_inspection() -> FragmentInspection:
    return _inspection_for("2024")


def _record_for(year: str, contract_ref: TableContractRef = _SEC_REF, **kwargs) -> FragmentRecord:
    inspection = kwargs.pop("inspection", None) or _inspection_for(year)
    return manifests.fragment_record(
        inspection, contract_ref,
        input_receipt_refs=kwargs.pop("input_receipt_refs", (RECEIPT_A,)),
        import_request_hash=kwargs.pop("import_request_hash", IRH_A),
    )


def _two_records() -> tuple[TableContractRef, list[FragmentRecord]]:
    return _SEC_REF, [_record_for("2024"), _record_for("2025")]


def _good_manifest() -> tuple[DatasetManifest, list[FragmentRecord]]:
    contract_ref, records = _two_records()
    manifest = manifests.dataset_manifest(
        contract_ref, records, knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT_A,), availability_evidence_refs=())
    return manifest, records


def _good_snapshot() -> tuple[SnapshotRef, DatasetManifest]:
    manifest, _ = _good_manifest()
    snap = manifests.snapshot_ref(
        {"securities": manifest}, calendar_version="cal.v1", source_priority_version="prio.v1",
        finality_receipt_refs=(RECEIPT_A,))
    return snap, manifest


# --------------------------------------------------------------------------
# real parquet -> publish -> inspect -> records -> manifest -> snapshot
# --------------------------------------------------------------------------


def _securities_rows(year: int) -> list[dict]:
    return [
        dict(ticker="AAA", year=year, first_date=None, last_date=None, mcap_usd=1.5e9, mcap_log=21.1,
             mcap_raw=1.5, mcap_unit_era="billions", mcap_quantized=False, n_obs=250, src="orats"),
        dict(ticker="BBB", year=year, first_date=None, last_date=None, mcap_usd=2.5e8, mcap_log=19.3,
             mcap_raw=250.0, mcap_unit_era="millions", mcap_quantized=False, n_obs=248, src="orats"),
    ]


def _table_from_rows(contract: TableContract, rows: list[dict]) -> pa.Table:
    arrays = {}
    for column in contract.columns:
        physical = _ARROW_TYPES.get(column.physical_type, pa.string())
        arrays[column.name] = pa.array([row.get(column.name) for row in rows], type=physical)
    return pa.table(arrays)


def _to_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def _publish_bytes(store: ArtifactStore, data: bytes) -> ObjectRef:
    ref = store.publish_bytes(data, schema_ref="parquet_fragment.v1")
    return ObjectRef(kind="parquet_fragment", object_id=ref.artifact_id,
                     content_hash=ref.content_hash, byte_size=ref.byte_size)


def _real_records(store: ArtifactStore) -> list[FragmentRecord]:
    records = []
    for year in (2024, 2025):
        table = _table_from_rows(_SEC_CONTRACT, _securities_rows(year))
        obj = _publish_bytes(store, _to_bytes(table))
        inspection = inspect_fragment(store, obj, _SEC_CONTRACT, _SEC_REF, str(year))
        records.append(manifests.fragment_record(
            inspection, _SEC_REF, input_receipt_refs=(RECEIPT_A,), import_request_hash=IRH_A))
    return records


def test_end_to_end_fragments_dataset_snapshot(tmp_path):
    store = ArtifactStore(tmp_path)
    records = _real_records(store)
    assert [r.partition_key for r in records] == ["2024", "2025"]

    for record in records:
        manifests.verify_fragment_record(record)
        assert decode_document(FragmentRecord, to_document(record)) == record

    manifest = manifests.dataset_manifest(
        _SEC_REF, records, knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT_A,), availability_evidence_refs=())
    manifests.verify_dataset_manifest(manifest, records)
    assert decode_document(DatasetManifest, to_document(manifest)) == manifest
    assert manifest.row_count == sum(r.row_count for r in records)

    snap = manifests.snapshot_ref(
        {"securities": manifest}, calendar_version="cal.v1", source_priority_version="prio.v1",
        finality_receipt_refs=(RECEIPT_A,))
    manifests.verify_snapshot_ref(snap, {"securities": manifest})
    assert decode_document(SnapshotRef, to_document(snap)) == snap
    assert snap.knowledge_mode_by_table == {"securities": "reconstructed"}


# --------------------------------------------------------------------------
# fragment_id excludes provenance; manifest_hash covers it
# --------------------------------------------------------------------------


def test_fragment_id_stable_but_manifest_hash_moves_with_provenance():
    base = _record_for("2024")
    other_receipts = _record_for("2024", input_receipt_refs=(RECEIPT_B,))
    other_irh = _record_for("2024", import_request_hash=IRH_B)

    assert base.fragment_id == other_receipts.fragment_id == other_irh.fragment_id
    assert base.manifest_hash != other_receipts.manifest_hash
    assert base.manifest_hash != other_irh.manifest_hash


def test_fragment_id_identical_across_attempts_publishing_the_same_bytes(tmp_path):
    """Operational identity (attempt id) is excluded: two publications of the
    same bytes under different attempts yield the same fragment_id."""
    store = ArtifactStore(tmp_path)
    table = _table_from_rows(_SEC_CONTRACT, _securities_rows(2024))
    data = _to_bytes(table)

    obj1 = _publish_bytes(store, data)
    obj2 = _publish_bytes(store, data)
    assert obj1 == obj2  # content-addressed: the store itself is attempt-independent

    insp1 = inspect_fragment(store, obj1, _SEC_CONTRACT, _SEC_REF, "2024")
    insp2 = inspect_fragment(store, obj2, _SEC_CONTRACT, _SEC_REF, "2024")
    rec1 = manifests.fragment_record(insp1, _SEC_REF, input_receipt_refs=(RECEIPT_A,), import_request_hash=IRH_A)
    rec2 = manifests.fragment_record(insp2, _SEC_REF, input_receipt_refs=(RECEIPT_A,), import_request_hash=IRH_A)
    assert rec1.fragment_id == rec2.fragment_id
    assert rec1.manifest_hash == rec2.manifest_hash


def test_reordering_contract_columns_changes_definition_hash_and_fragment_id():
    reordered = dataclasses.replace(_SEC_CONTRACT, columns=tuple(reversed(_SEC_CONTRACT.columns)))
    reordered_hash = manifests.table_contract_hash(reordered)
    assert reordered_hash != _SEC_CONTRACT.definition_hash

    reordered_ref = TableContractRef(contract_id=_SEC_CONTRACT.contract_id, definition_hash=reordered_hash)
    base = _record_for("2024")
    changed = _record_for("2024", contract_ref=reordered_ref)
    assert base.fragment_id != changed.fragment_id


# --------------------------------------------------------------------------
# fragment identity: every field in the payload changes the id
# --------------------------------------------------------------------------


def _mutate_object_hash(insp: FragmentInspection) -> FragmentInspection:
    return dataclasses.replace(insp, object_ref=dataclasses.replace(
        insp.object_ref, content_hash=_fake_hash("mutated-object")))


FRAGMENT_MUTATIONS = {
    "logical_content_hash": lambda insp: dataclasses.replace(
        insp, logical_content_hash=_fake_hash("mutated-logical")),
    "object_content_hash": _mutate_object_hash,
    "row_count": lambda insp: dataclasses.replace(insp, row_count=insp.row_count + 1),
    "primary_key_min": lambda insp: dataclasses.replace(insp, primary_key_min=("ZZZ", 2024)),
    "primary_key_max": lambda insp: dataclasses.replace(insp, primary_key_max=("ZZZ", 2024)),
    "partition_key": lambda insp: dataclasses.replace(insp, partition_key="2025"),
}


@pytest.mark.parametrize("field", sorted(FRAGMENT_MUTATIONS))
def test_fragment_id_changes_with_each_identity_field(field):
    base_insp = _base_inspection()
    base = manifests.fragment_record(base_insp, _SEC_REF, input_receipt_refs=(RECEIPT_A,),
                                     import_request_hash=IRH_A)
    mutated_insp = FRAGMENT_MUTATIONS[field](base_insp)
    mutated = manifests.fragment_record(mutated_insp, _SEC_REF, input_receipt_refs=(RECEIPT_A,),
                                        import_request_hash=IRH_A)
    assert mutated.fragment_id != base.fragment_id


# --------------------------------------------------------------------------
# dataset version identity: the payload/hash helper, permuted and mutated
# --------------------------------------------------------------------------


def _base_dsv_payload() -> dict:
    return manifests.dataset_version_identity_payload(
        _SEC_REF, ["frag_a", "frag_b"], knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT_A,), availability_evidence_refs=())


DATASET_MUTATIONS = {
    "fragment_ids_permuted": lambda p: {**p, "fragment_ids": list(reversed(p["fragment_ids"]))},
    "fragment_ids_changed": lambda p: {**p, "fragment_ids": ["frag_a", "frag_c"]},
    "knowledge_mode": lambda p: {**p, "knowledge_mode": "observed"},
    "coverage_receipt_refs": lambda p: {**p, "coverage_receipt_refs": [RECEIPT_B]},
    "availability_evidence_refs": lambda p: {**p, "availability_evidence_refs": [RECEIPT_B]},
}


@pytest.mark.parametrize("field", sorted(DATASET_MUTATIONS))
def test_dataset_version_hash_changes_with_each_identity_field(field):
    base_hash = content_hash(_base_dsv_payload())
    mutated_hash = content_hash(DATASET_MUTATIONS[field](_base_dsv_payload()))
    assert mutated_hash != base_hash


def test_dataset_version_hash_reordering_evidence_refs_does_not_change_it():
    """Only ``fragment_ids`` order is meaningful; evidence refs are sorted
    inside the payload, so citing the same evidence in a different order
    must not mint a new identity."""
    a = manifests.dataset_version_identity_payload(
        _SEC_REF, ["frag_a", "frag_b"], knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT_A, RECEIPT_B), availability_evidence_refs=())
    b = manifests.dataset_version_identity_payload(
        _SEC_REF, ["frag_a", "frag_b"], knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT_B, RECEIPT_A), availability_evidence_refs=())
    assert content_hash(a) == content_hash(b)


# --------------------------------------------------------------------------
# snapshot identity: the payload/hash helper, mutated
# --------------------------------------------------------------------------


def _base_snapshot_payload() -> dict:
    return manifests.snapshot_identity_payload(
        {"securities": "dsv_" + "1" * 32}, calendar_version="cal.v1", source_priority_version="prio.v1",
        finality_receipt_refs=(RECEIPT_A,), knowledge_mode_by_table={"securities": "reconstructed"})


SNAPSHOT_MUTATIONS = {
    "table_version": lambda p: {**p, "table_versions": {"securities": "dsv_" + "2" * 32}},
    "calendar_version": lambda p: {**p, "calendar_version": "cal.v2"},
    "source_priority_version": lambda p: {**p, "source_priority_version": "prio.v2"},
    "finality_receipt_refs": lambda p: {**p, "finality_receipt_refs": [RECEIPT_B]},
    "knowledge_mode": lambda p: {**p, "knowledge_mode_by_table": {"securities": "observed"}},
}


@pytest.mark.parametrize("field", sorted(SNAPSHOT_MUTATIONS))
def test_snapshot_id_changes_with_each_identity_field(field):
    base_hash = content_hash(_base_snapshot_payload())
    mutated_hash = content_hash(SNAPSHOT_MUTATIONS[field](_base_snapshot_payload()))
    assert mutated_hash != base_hash


# --------------------------------------------------------------------------
# one fragment per logical partition: the builder never sorts silently
# --------------------------------------------------------------------------


def test_dataset_manifest_refuses_out_of_order_records():
    contract_ref, records = _two_records()
    with pytest.raises(DataError) as err:
        manifests.dataset_manifest(contract_ref, list(reversed(records)), knowledge_mode="reconstructed",
                                   coverage_receipt_refs=(), availability_evidence_refs=())
    assert err.value.code == "MANIFEST_CORRUPT"


def test_dataset_manifest_refuses_duplicate_partition_keys():
    contract_ref, records = _two_records()
    with pytest.raises(DataError) as err:
        manifests.dataset_manifest(contract_ref, [records[0], records[0]], knowledge_mode="reconstructed",
                                   coverage_receipt_refs=(), availability_evidence_refs=())
    assert err.value.code == "MANIFEST_CORRUPT"


def test_dataset_manifest_refuses_mismatched_contract_refs():
    contract_ref, records = _two_records()
    other_ref = TableContractRef(contract_id=contract_ref.contract_id, definition_hash=_fake_hash("other"))
    mismatched = [dataclasses.replace(records[0], table_contract_ref=other_ref), records[1]]
    with pytest.raises(DataError) as err:
        manifests.dataset_manifest(contract_ref, mismatched, knowledge_mode="reconstructed",
                                   coverage_receipt_refs=(), availability_evidence_refs=())
    assert err.value.code == "CONTRACT_MISMATCH"


def test_dataset_manifest_refuses_observed_mode_without_evidence():
    contract_ref, records = _two_records()
    with pytest.raises(DataError) as err:
        manifests.dataset_manifest(contract_ref, records, knowledge_mode="observed",
                                   coverage_receipt_refs=(), availability_evidence_refs=())
    assert err.value.code == "CONTRACT_MISMATCH"


def test_dataset_manifest_accepts_attested_stable_with_evidence():
    contract_ref, records = _two_records()
    manifest = manifests.dataset_manifest(
        contract_ref, records, knowledge_mode="attested_stable",
        coverage_receipt_refs=(), availability_evidence_refs=(RECEIPT_A,))
    manifests.verify_dataset_manifest(manifest, records)


# --------------------------------------------------------------------------
# verifiers catch tampering
# --------------------------------------------------------------------------


def test_verify_fragment_record_catches_row_count_tamper():
    record = _record_for("2024")
    tampered = dataclasses.replace(record, row_count=record.row_count + 1)
    with pytest.raises(DataError) as err:
        manifests.verify_fragment_record(tampered)
    assert err.value.code == "MANIFEST_CORRUPT"


def test_verify_dataset_manifest_catches_row_count_tamper():
    manifest, records = _good_manifest()
    tampered = dataclasses.replace(manifest, row_count=manifest.row_count + 1)
    with pytest.raises(DataError) as err:
        manifests.verify_dataset_manifest(tampered, records)
    assert err.value.code == "MANIFEST_CORRUPT"


def test_verify_dataset_manifest_catches_membership_swap():
    manifest, records = _good_manifest()
    swapped = (manifest.fragment_refs[1], manifest.fragment_refs[0])
    tampered = dataclasses.replace(manifest, fragment_refs=swapped)
    with pytest.raises(DataError) as err:
        manifests.verify_dataset_manifest(tampered, records)
    assert err.value.code == "MANIFEST_CORRUPT"


def test_verify_snapshot_ref_catches_wrong_manifest_pointer():
    snap, manifest = _good_snapshot()
    other_manifest = manifests.dataset_manifest(
        _SEC_REF, [_record_for("2024")], knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT_B,), availability_evidence_refs=())
    with pytest.raises(DataError) as err:
        manifests.verify_snapshot_ref(snap, {"securities": other_manifest})
    assert err.value.code == "MANIFEST_CORRUPT"


def test_verify_snapshot_ref_catches_knowledge_mode_tamper():
    snap, manifest = _good_snapshot()
    tampered = dataclasses.replace(snap, knowledge_mode_by_table={"securities": "observed"})
    with pytest.raises(DataError) as err:
        manifests.verify_snapshot_ref(tampered, {"securities": manifest})
    assert err.value.code == "MANIFEST_CORRUPT"
