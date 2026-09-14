"""Task 7a: a logical partition may hold several ordered, non-overlapping
fragments, so legacy multi-part years import byte-for-byte without
compaction (phase-2 guide §3.3, §6 invariant 6).

The real ``daily_market`` contract's rows for one year are written three
ways — one file, 3 ordered non-overlapping parts, 19 tiny parts — and every
construction must give an identical ``objects.partition_logical_hash`` and
dataset ``logical_content_hash``, while minting a different
``dataset_version_id`` (fragment membership differs). Negative controls that
need only fragment metadata (overlap, boundary duplicate, out-of-order,
missing/wrong stored hash) reuse ``tests/test_v2_data_manifests.py``'s
synthetic fixtures rather than real Parquet I/O; only the ones that need real
streamed bytes (an internally unsorted part, a wrong stored hash caught by
re-streaming, the end-to-end commit/resolve) publish real objects.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts.data import DatasetManifest, ObjectRef  # noqa: E402
from engine.v2.data import manifests  # noqa: E402
from engine.v2.data.catalog import commit_snapshot  # noqa: E402
from engine.v2.data.documents import decode_document  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.objects import (  # noqa: E402
    FragmentInspection,
    inspect_fragment,
    inspect_staged_file,
    inspect_staged_partition,
    partition_logical_hash,
)
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import (  # noqa: E402
    ArtifactStore,
    DocumentError,
    content_hash,
    to_document,
)
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from tests.ops_support import FakeClock  # noqa: E402
from tests.test_v2_data_manifests import (  # noqa: E402
    _DAILY_MARKET_CONTRACT,
    _DAILY_MARKET_REF,
    _SEC_REF,
    RECEIPT_A,
    _fake_hash,
    _publish_bytes,
    _record_for,
    _table_from_rows,
    _to_bytes,
)

YEAR = 2024
_YEAR_KEY = str(YEAR)
_IRH = _fake_hash("partitions-irh")


def _hash(label: str) -> str:
    return content_hash({"label": label})


def _noop_fence(conn) -> None:
    return None


def _catalog(tmp_path):
    clock = FakeClock()
    return open_catalog(tmp_path / "catalog.sqlite", clock=clock), clock


# --------------------------------------------------------------------------
# real daily_market rows for one year: 19 tickers, one row each, presorted
# --------------------------------------------------------------------------


def _rows(n: int = 19) -> list[dict]:
    date = datetime(YEAR, 1, 2)
    common = dict(spot=100.0, iv10=30.0, iv30=32.0, exern_iv10=29.0, exern_iv30=31.0,
                 implied_move=5.0, implied_reconstructed=False, rvol30=28.0, skew=1.1,
                 contango=0.5, fwd90_30=33.0, fexern90_30=34.0, iee=0.2, mcap_usd=1e9,
                 mcap_log=20.7, mcap_asof=date, mcap_age_days=0.0,
                 src_spot="orats", src_iv="orats", src_mcap="orats")
    return [dict(ticker=f"T{i:02d}", date=date, year=YEAR, **common) for i in range(n)]


def _publish_rows(store: ArtifactStore, rows: list[dict]) -> ObjectRef:
    table = _table_from_rows(_DAILY_MARKET_CONTRACT, rows)
    return _publish_bytes(store, _to_bytes(table))


def _fragment(store: ArtifactStore, rows: list[dict]) -> FragmentInspection:
    obj = _publish_rows(store, rows)
    return inspect_fragment(store, obj, _DAILY_MARKET_CONTRACT, _DAILY_MARKET_REF, _YEAR_KEY)


def _chunks(rows: list[dict], sizes: list[int]) -> list[list[dict]]:
    out, i = [], 0
    for size in sizes:
        out.append(rows[i:i + size])
        i += size
    assert i == len(rows)
    return out


def _records(inspections: list[FragmentInspection]):
    return [manifests.fragment_record(i, _DAILY_MARKET_REF, input_receipt_refs=(RECEIPT_A,),
                                      import_request_hash=_IRH)
            for i in inspections]


# --------------------------------------------------------------------------
# three constructions of the same year give the same logical content
# --------------------------------------------------------------------------


def test_one_three_and_nineteen_parts_give_identical_hashes(tmp_path):
    store = ArtifactStore(tmp_path)
    rows = _rows()
    one = [_fragment(store, rows)]
    three = [_fragment(store, chunk) for chunk in _chunks(rows, [7, 6, 6])]
    nineteen = [_fragment(store, [row]) for row in rows]

    partition_hashes = {
        name: partition_logical_hash(store, [i.object_ref for i in inspections],
                                     _DAILY_MARKET_CONTRACT, _DAILY_MARKET_REF, _YEAR_KEY)
        for name, inspections in (("one", one), ("three", three), ("nineteen", nineteen))
    }
    assert partition_hashes["one"] == partition_hashes["three"] == partition_hashes["nineteen"]
    assert partition_hashes["one"] == one[0].logical_content_hash  # single fragment: its own hash

    manifest_one = manifests.dataset_manifest(
        _DAILY_MARKET_REF, _records(one), knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT_A,), availability_evidence_refs=())
    manifest_three = manifests.dataset_manifest(
        _DAILY_MARKET_REF, _records(three), knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT_A,), availability_evidence_refs=(),
        partition_logical_hashes={_YEAR_KEY: partition_hashes["three"]})
    manifest_nineteen = manifests.dataset_manifest(
        _DAILY_MARKET_REF, _records(nineteen), knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT_A,), availability_evidence_refs=(),
        partition_logical_hashes={_YEAR_KEY: partition_hashes["nineteen"]})

    assert (manifest_one.logical_content_hash == manifest_three.logical_content_hash
           == manifest_nineteen.logical_content_hash)
    ids = {m.dataset_version_ref.dataset_version_id for m in (manifest_one, manifest_three,
                                                              manifest_nineteen)}
    assert len(ids) == 3


def test_single_fragment_dataset_logical_hash_is_unchanged(tmp_path):
    """D08/D03: the new (partition_key, partition hash) formula reproduces the
    old (partition_key, fragment logical hash) one exactly for every dataset
    with no multi-fragment partition (task brief decision 3)."""
    store = ArtifactStore(tmp_path)
    record = _records([_fragment(store, _rows())])[0]
    manifest = manifests.dataset_manifest(
        _DAILY_MARKET_REF, [record], knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT_A,), availability_evidence_refs=())
    expected = content_hash({
        "algorithm": manifests.DATASET_LOGICAL_ALGORITHM,
        "table_contract_ref": to_document(_DAILY_MARKET_REF),
        "partitions": [[record.partition_key, record.logical_content_hash]],
    })
    assert manifest.logical_content_hash == expected
    assert manifest.partition_logical_hashes == {}


# --------------------------------------------------------------------------
# commit a multi-fragment snapshot; resolve reconstructs it, every verifier passes
# --------------------------------------------------------------------------


def test_commit_and_resolve_multi_fragment_snapshot(tmp_path):
    store = ArtifactStore(tmp_path / "objects")
    conn, clock = _catalog(tmp_path)
    rows = _rows()
    three = [_fragment(store, chunk) for chunk in _chunks(rows, [7, 6, 6])]
    object_refs = [i.object_ref for i in three]
    partition_hash = partition_logical_hash(store, object_refs, _DAILY_MARKET_CONTRACT,
                                            _DAILY_MARKET_REF, _YEAR_KEY)
    records = _records(three)
    manifest = manifests.dataset_manifest(
        _DAILY_MARKET_REF, records, knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT_A,), availability_evidence_refs=(),
        partition_logical_hashes={_YEAR_KEY: partition_hash})
    snap = manifests.snapshot_ref(
        {"daily_market": manifest}, calendar_version="cal.v1", source_priority_version="prio.v1",
        finality_receipt_refs=(RECEIPT_A,))

    for record in records:
        manifests.verify_fragment_record(record)
    manifests.verify_dataset_manifest(manifest, records)
    manifests.verify_partition_hashes(store, manifest, records, _DAILY_MARKET_CONTRACT)
    manifests.verify_snapshot_ref(snap, {"daily_market": manifest})

    receipt = commit_snapshot(
        conn, scope="shadow", request_hash=_hash("commit-1"), contracts=[_DAILY_MARKET_CONTRACT],
        objects=object_refs, records=records, manifests=[manifest], snapshot=snap,
        expected_head_snapshot_id=None, expected_head_generation=0, receipt_id="r1",
        attempt_id="att-1", fence=1, fence_check=_noop_fence, clock=clock, store=store)
    assert receipt.status == "committed"

    resolved = Repository(conn).resolve(snap.snapshot_id)
    assert resolved == snap


# --------------------------------------------------------------------------
# negative controls needing only fragment metadata (no real Parquet I/O)
# --------------------------------------------------------------------------


def _synthetic_record(partition_key: str, key_min: tuple, key_max: tuple, *, label: str):
    object_hash = _fake_hash(f"object-{label}")
    inspection = FragmentInspection(
        object_ref=ObjectRef(kind="parquet_fragment",
                             object_id="art_" + hashlib.sha256(label.encode()).hexdigest()[:32],
                             content_hash=object_hash, byte_size=100),
        partition_key=partition_key, row_count=3, byte_hash=object_hash,
        logical_content_hash=_fake_hash(f"logical-{label}"),
        primary_key_min=key_min, primary_key_max=key_max, time_min=None, time_max=None)
    return manifests.fragment_record(inspection, _SEC_REF, input_receipt_refs=(RECEIPT_A,),
                                     import_request_hash=_IRH)


def test_overlapping_ranges_are_refused():
    rec1 = _synthetic_record("2024", ("T00", 2024), ("T10", 2024), label="ov-1")
    rec2 = _synthetic_record("2024", ("T05", 2024), ("T18", 2024), label="ov-2")
    with pytest.raises(DataError) as err:
        manifests.dataset_manifest(_SEC_REF, [rec1, rec2], knowledge_mode="reconstructed",
                                   coverage_receipt_refs=(), availability_evidence_refs=(),
                                   partition_logical_hashes={"2024": _fake_hash("ph")})
    assert err.value.code == "MANIFEST_CORRUPT"


def test_boundary_duplicate_key_is_refused():
    rec1 = _synthetic_record("2024", ("T00", 2024), ("T09", 2024), label="bd-1")
    rec2 = _synthetic_record("2024", ("T09", 2024), ("T18", 2024), label="bd-2")
    with pytest.raises(DataError) as err:
        manifests.dataset_manifest(_SEC_REF, [rec1, rec2], knowledge_mode="reconstructed",
                                   coverage_receipt_refs=(), availability_evidence_refs=(),
                                   partition_logical_hashes={"2024": _fake_hash("ph")})
    assert err.value.code == "MANIFEST_CORRUPT"


def test_out_of_order_parts_are_refused():
    rec1 = _synthetic_record("2024", ("T00", 2024), ("T09", 2024), label="oo-1")
    rec2 = _synthetic_record("2024", ("T10", 2024), ("T18", 2024), label="oo-2")
    with pytest.raises(DataError) as err:
        manifests.dataset_manifest(_SEC_REF, [rec2, rec1], knowledge_mode="reconstructed",
                                   coverage_receipt_refs=(), availability_evidence_refs=(),
                                   partition_logical_hashes={"2024": _fake_hash("ph")})
    assert err.value.code == "MANIFEST_CORRUPT"


def test_missing_partition_hash_for_multi_fragment_is_refused():
    rec1 = _synthetic_record("2024", ("T00", 2024), ("T09", 2024), label="mp-1")
    rec2 = _synthetic_record("2024", ("T10", 2024), ("T18", 2024), label="mp-2")
    with pytest.raises(DataError) as err:
        manifests.dataset_manifest(_SEC_REF, [rec1, rec2], knowledge_mode="reconstructed",
                                   coverage_receipt_refs=(), availability_evidence_refs=())
    assert err.value.code == "MANIFEST_CORRUPT"


def test_supplied_single_fragment_hash_must_match_is_still_enforced():
    """Sanity check on the nullable-style default: a caller may still state a
    single-fragment partition's hash explicitly, but only if it agrees."""
    record = _record_for("2024")
    with pytest.raises(DataError) as err:
        manifests.dataset_manifest(_SEC_REF, [record], knowledge_mode="reconstructed",
                                   coverage_receipt_refs=(), availability_evidence_refs=(),
                                   partition_logical_hashes={"2024": _fake_hash("wrong")})
    assert err.value.code == "MANIFEST_CORRUPT"


# --------------------------------------------------------------------------
# negative controls needing real streamed bytes
# --------------------------------------------------------------------------


def test_partition_logical_hash_refuses_internally_unsorted_fragment(tmp_path):
    store = ArtifactStore(tmp_path)
    rows = _rows()
    chunk1, chunk2 = rows[:10], rows[10:]
    shuffled_chunk2 = [chunk2[-1], *chunk2[:-1]]
    frag1 = _fragment(store, chunk1)
    unsorted_obj = _publish_rows(store, shuffled_chunk2)
    with pytest.raises(DataError) as err:
        partition_logical_hash(store, [frag1.object_ref, unsorted_obj], _DAILY_MARKET_CONTRACT,
                               _DAILY_MARKET_REF, _YEAR_KEY)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_wrong_stored_partition_hash_caught_by_verify_and_commit(tmp_path):
    store = ArtifactStore(tmp_path / "objects")
    rows = _rows()
    three = [_fragment(store, chunk) for chunk in _chunks(rows, [7, 6, 6])]
    records = _records(three)
    wrong_hash = _fake_hash("wrong-partition-hash")
    manifest = manifests.dataset_manifest(
        _DAILY_MARKET_REF, records, knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT_A,), availability_evidence_refs=(),
        partition_logical_hashes={_YEAR_KEY: wrong_hash})

    with pytest.raises(DataError) as err:
        manifests.verify_partition_hashes(store, manifest, records, _DAILY_MARKET_CONTRACT)
    assert err.value.code == "MANIFEST_CORRUPT"

    conn, clock = _catalog(tmp_path)
    snap = manifests.snapshot_ref(
        {"daily_market": manifest}, calendar_version="cal.v1", source_priority_version="prio.v1",
        finality_receipt_refs=(RECEIPT_A,))
    with pytest.raises(DataError) as err:
        commit_snapshot(
            conn, scope="shadow", request_hash=_hash("commit-2"), contracts=[_DAILY_MARKET_CONTRACT],
            objects=[i.object_ref for i in three], records=records, manifests=[manifest],
            snapshot=snap, expected_head_snapshot_id=None, expected_head_generation=0,
            receipt_id="r2", attempt_id="att-1", fence=1, fence_check=_noop_fence, clock=clock,
            store=store)
    assert err.value.code == "MANIFEST_CORRUPT"


# --------------------------------------------------------------------------
# document format: v1.1 round-trips, v1.2 is refused, a bad hash value is refused
# --------------------------------------------------------------------------


def test_v1_1_manifest_round_trips_and_v1_2_is_refused():
    record = _record_for("2024")
    manifest = manifests.dataset_manifest(
        _SEC_REF, [record], knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT_A,), availability_evidence_refs=())
    doc = to_document(manifest)
    assert doc["schema_version"] == "dataset_manifest.v1.1"
    assert decode_document(DatasetManifest, doc) == manifest

    newer_minor = dict(doc, schema_version="dataset_manifest.v1.2")
    with pytest.raises(DocumentError) as err:
        decode_document(DatasetManifest, newer_minor)
    assert err.value.code == "UNSUPPORTED_VERSION"


def test_bad_partition_hash_format_is_refused():
    record = _record_for("2024")
    manifest = manifests.dataset_manifest(
        _SEC_REF, [record], knowledge_mode="reconstructed",
        coverage_receipt_refs=(RECEIPT_A,), availability_evidence_refs=(),
        partition_logical_hashes={"2024": record.logical_content_hash})
    doc = to_document(manifest)
    doc["partition_logical_hashes"]["2024"] = "not-a-hash"
    with pytest.raises(DocumentError) as err:
        decode_document(DatasetManifest, doc)
    assert err.value.code == "BAD_HASH_FORMAT"


# --------------------------------------------------------------------------
# worker side: one streaming pass gives fragment AND partition hashes
# --------------------------------------------------------------------------


def _staged_parts(tmp_path, store, sizes):
    staged = tmp_path / "staged"
    staged.mkdir()
    files, object_refs = [], []
    for index, chunk in enumerate(_chunks(_rows(), sizes)):
        data = _to_bytes(_table_from_rows(_DAILY_MARKET_CONTRACT, chunk))
        path = staged / f"part-{index:04d}.parquet"
        path.write_bytes(data)
        files.append((path, "sha256:" + hashlib.sha256(data).hexdigest(), len(data)))
        object_refs.append(_publish_bytes(store, data))
    return files, object_refs


def test_staged_partition_single_pass_matches_the_published_audit(tmp_path):
    """The worker's combined hash is what the coordinator's audit re-derives
    from the published objects, and each fragment matches ``inspect_staged_file``."""
    store = ArtifactStore(tmp_path / "store")
    files, object_refs = _staged_parts(tmp_path, store, [7, 6, 6])
    inspections, partition_hash = inspect_staged_partition(
        files, _DAILY_MARKET_CONTRACT, _DAILY_MARKET_REF, _YEAR_KEY)
    assert partition_hash == partition_logical_hash(
        store, object_refs, _DAILY_MARKET_CONTRACT, _DAILY_MARKET_REF, _YEAR_KEY)
    for (path, digest, size), inspection in zip(files, inspections):
        assert inspection == inspect_staged_file(
            path, _DAILY_MARKET_CONTRACT, _DAILY_MARKET_REF, _YEAR_KEY,
            expected_content_hash=digest, expected_byte_size=size)
    alone, alone_hash = inspect_staged_partition(
        files[:1], _DAILY_MARKET_CONTRACT, _DAILY_MARKET_REF, _YEAR_KEY)
    assert alone_hash is None and alone == inspections[:1]


def test_staged_partition_refuses_seam_disorder_and_changed_bytes(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    files, _ = _staged_parts(tmp_path, store, [7, 6, 6])
    with pytest.raises(DataError) as err:
        inspect_staged_partition(list(reversed(files)), _DAILY_MARKET_CONTRACT,
                                 _DAILY_MARKET_REF, _YEAR_KEY)
    assert err.value.code == "CONTRACT_MISMATCH"
    path, _, size = files[1]
    with pytest.raises(DataError) as err:
        inspect_staged_partition([files[0], (path, _fake_hash("other bytes"), size)],
                                 _DAILY_MARKET_CONTRACT, _DAILY_MARKET_REF, _YEAR_KEY)
    assert err.value.code == "INPUT_CHANGED"
