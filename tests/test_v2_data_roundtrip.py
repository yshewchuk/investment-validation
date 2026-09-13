"""D09: a real Parquet file copied, fsynced, registered, read through Arrow,
written to the private legacy layout, and read back with identical typed
rows and byte/logical refs — phase-2 guide §7.2, §8.2, §12; task brief
decision 6 (the round trip happens in this test, not in production code).

Corrupting one object byte on disk and re-scanning proves the scan refuses
it (OBJECT_CORRUPT) before yielding any row from that fragment — the
stat-tuple verification cache §8.2 step 6 describes is deferred (TD-1), so
every open re-hashes, unconditionally.
"""
from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts.data import DataQuery, KeyPredicate  # noqa: E402
from engine.v2.contracts.jobs import LegacyFileRef  # noqa: E402
from engine.v2.data import manifests  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.objects import inspect_fragment, publish_legacy_file  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore, CONTENT_HASH_PREFIX  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    commit_tables,
    contract_for,
    contract_ref_for,
    fake_hash,
    table_from_rows,
    to_bytes,
)
from tests.ops_support import FakeClock  # noqa: E402

from engine.data import store as legacy_store  # noqa: E402

_SEC = contract_for("securities")
_SEC_REF = contract_ref_for(_SEC)

_ROWS = [
    dict(ticker="AAA", year=2024, first_date=datetime(2024, 1, 2), last_date=datetime(2024, 12, 30),
        mcap_usd=1.5e9, mcap_log=21.1, mcap_raw=1.5, mcap_unit_era="billions",
        mcap_quantized=False, n_obs=250, src="orats"),
    dict(ticker="BBB", year=2024, first_date=datetime(2024, 1, 3), last_date=datetime(2024, 12, 29),
        mcap_usd=2.5e8, mcap_log=19.3, mcap_raw=250.0, mcap_unit_era="millions",
        mcap_quantized=False, n_obs=248, src="orats"),
]


def _write_source(root: Path, rel: str, data: bytes) -> LegacyFileRef:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    return LegacyFileRef(path=rel, content_hash=f"{CONTENT_HASH_PREFIX}{digest}", byte_size=len(data))


def _catalog(tmp_path):
    clock = FakeClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    return conn, clock


def _published(tmp_path):
    """Publish + inspect + commit one real securities fragment through the
    genuine ``publish_legacy_file`` copy path (never bytes handed directly to
    the store, unlike most of this package's other tests)."""
    source_root = tmp_path / "source"
    store = ArtifactStore(tmp_path / "objects")
    table = table_from_rows(_SEC, _ROWS)
    data = to_bytes(table)
    file_ref = _write_source(source_root, "securities/2024.parquet", data)

    obj = publish_legacy_file(store, "att_1", source_root, file_ref)
    inspection = inspect_fragment(store, obj, _SEC, _SEC_REF, "2024")
    record = manifests.fragment_record(inspection, _SEC_REF, input_receipt_refs=(fake_hash("r"),),
                                       import_request_hash=fake_hash("irh"))
    conn, clock = _catalog(tmp_path)
    snap = commit_tables(conn, clock, {"securities": [record]}, {"securities": _SEC})
    return conn, store, snap, table, record


def _scan_all(repo, snap, columns):
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_SEC_REF, columns=columns,
        key_filter=(KeyPredicate(column="year", operator="eq", values=(2024,)),),
        order_by=("ticker", "year"), max_batch_rows=10, max_result_rows=10)
    return [row for batch in repo.scan(query, table_name="securities") for row in batch.to_pylist()]


def test_full_round_trip_through_arrow_curated_layout_and_legacy_reader(tmp_path):
    conn, store, snap, table, record = _published(tmp_path)
    columns = tuple(c.name for c in _SEC.columns)
    repo = Repository(conn, store)
    scanned = _scan_all(repo, snap, columns)
    assert scanned == table.to_pylist()

    # Write the same logical content to the private legacy layout
    # (decision 6: <tmp>/curated/<table>/year=<y>/part.parquet).
    curated_dir = tmp_path / "curated" / "securities" / "year=2024"
    curated_dir.mkdir(parents=True)
    curated_path = curated_dir / "part.parquet"
    # A different physical encoding (compression) than the original publish,
    # to prove the logical hash below is content-derived, not byte-derived.
    pq.write_table(table, curated_path, compression="gzip")

    # Read back with pyarrow directly.
    reread = pq.read_table(curated_path).to_pylist()
    assert reread == table.to_pylist()

    # Read back with the legacy single-path reader (`engine.data.store`),
    # never through the global curated-store singleton.
    legacy_frame = legacy_store._read_part(curated_path, columns=None)
    assert legacy_frame["ticker"].tolist() == ["AAA", "BBB"]
    assert legacy_frame["mcap_usd"].tolist() == [1.5e9, 2.5e8]

    # Byte/logical refs: re-inspecting the curated copy (different physical
    # file, same logical content) reproduces the same logical_content_hash,
    # even though it is a fresh object with its own byte_hash.
    curated_store = ArtifactStore(tmp_path / "curated_objects")
    curated_obj = curated_store.publish_bytes(curated_path.read_bytes(), schema_ref="parquet_fragment.v1")
    from engine.v2.contracts.data import ObjectRef
    curated_ref = ObjectRef(kind="parquet_fragment", object_id=curated_obj.artifact_id,
                            content_hash=curated_obj.content_hash, byte_size=curated_obj.byte_size)
    curated_inspection = inspect_fragment(curated_store, curated_ref, _SEC, _SEC_REF, "2024")
    assert curated_inspection.logical_content_hash == record.logical_content_hash
    assert curated_inspection.byte_hash != record.byte_hash  # a different physical file


def test_corrupted_object_byte_refuses_the_scan_before_any_row(tmp_path):
    conn, store, snap, _table, record = _published(tmp_path)
    digest = record.object_ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
    object_path = store.root / "objects" / digest[:2] / digest
    os.chmod(object_path, 0o644)
    data = bytearray(object_path.read_bytes())
    data[-1] ^= 0xFF  # flip the last byte
    object_path.write_bytes(bytes(data))
    os.chmod(object_path, 0o444)

    repo = Repository(conn, store)
    columns = tuple(c.name for c in _SEC.columns)
    with pytest.raises(DataError) as err:
        list(_scan_all(repo, snap, columns))
    assert err.value.code == "OBJECT_CORRUPT"
