"""D08/D09: publish one legacy Parquet file as an object, inspect it as a
fragment candidate (phase-2 guide §7.2).

Every synthetic Parquet file is built with pyarrow under ``tmp_path`` against
the real ``securities``/``daily_market`` ``TableContract``s from
``build_legacy_mapping()``, with a few fake rows — never private data.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts.data import ObjectRef, TableContract, TableContractRef  # noqa: E402
from engine.v2.contracts.jobs import LegacyFileRef  # noqa: E402
from engine.v2.data.documents import decode_document  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.legacy_adapter import build_legacy_mapping  # noqa: E402
from engine.v2.data.objects import (  # noqa: E402
    PARQUET_FRAGMENT_SCHEMA_REF,
    inspect_fragment,
    logical_partition_hash,
    publish_legacy_file,
)
from engine.v2.foundation import ArtifactStore  # noqa: E402

_ARROW_TYPES = {
    "string": pa.string(),
    "float64": pa.float64(),
    "int64": pa.int64(),
    "bool": pa.bool_(),
    "timestamp[ns]": pa.timestamp("ns"),
    "timestamp[us]": pa.timestamp("us"),
}

_MAPPING = build_legacy_mapping()


def _contract(name: str) -> TableContract:
    return decode_document(TableContract, _MAPPING["tables"][name])


def _contract_ref(contract: TableContract) -> TableContractRef:
    return TableContractRef(contract_id=contract.contract_id, definition_hash=contract.definition_hash)


def _securities_rows() -> list[dict]:
    return [
        dict(ticker="AAA", year=2024, first_date=datetime(2024, 1, 2), last_date=datetime(2024, 12, 30),
             mcap_usd=1.5e9, mcap_log=21.1, mcap_raw=1.5, mcap_unit_era="billions",
             mcap_quantized=False, n_obs=250, src="orats"),
        dict(ticker="BBB", year=2024, first_date=datetime(2024, 1, 3), last_date=datetime(2024, 12, 29),
             mcap_usd=2.5e8, mcap_log=19.3, mcap_raw=250.0, mcap_unit_era="millions",
             mcap_quantized=False, n_obs=248, src="orats"),
        dict(ticker="CCC", year=2024, first_date=datetime(2024, 1, 4), last_date=datetime(2024, 12, 28),
             mcap_usd=None, mcap_log=None, mcap_raw=None, mcap_unit_era=None,
             mcap_quantized=None, n_obs=None, src=None),
    ]


def _daily_market_rows() -> list[dict]:
    common = dict(spot=100.0, iv10=30.0, iv30=32.0, exern_iv10=29.0, exern_iv30=31.0,
                  implied_move=5.0, implied_reconstructed=False, rvol30=28.0, skew=1.1,
                  contango=0.5, fwd90_30=33.0, fexern90_30=34.0, iee=0.2, mcap_usd=1e9,
                  mcap_log=20.7, mcap_asof=datetime(2024, 1, 2), mcap_age_days=0.0,
                  src_spot="orats", src_iv="orats", src_mcap="orats")
    return [
        dict(ticker="AAA", date=datetime(2024, 1, 2), year=2024, **common),
        dict(ticker="AAA", date=datetime(2024, 1, 3), year=2024, **common),
        dict(ticker="BBB", date=datetime(2024, 1, 2), year=2024, **common),
    ]


def _table_from_rows(contract: TableContract, rows: list[dict], *, drop: tuple = ()) -> pa.Table:
    arrays = {}
    for column in contract.columns:
        if column.name in drop:
            continue
        values = [row.get(column.name) for row in rows]
        arrays[column.name] = pa.array(values, type=_ARROW_TYPES[column.physical_type])
    return pa.table(arrays)


def _to_bytes(table: pa.Table, **write_kwargs) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, **write_kwargs)
    return sink.getvalue().to_pybytes()


def _publish_bytes(store: ArtifactStore, data: bytes) -> ObjectRef:
    """Publish raw bytes as an object directly — D08's subject is inspection, not copy mechanics."""
    ref = store.publish_bytes(data, schema_ref=PARQUET_FRAGMENT_SCHEMA_REF)
    return ObjectRef(kind="parquet_fragment", object_id=ref.artifact_id,
                     content_hash=ref.content_hash, byte_size=ref.byte_size)


def _inspect(store, obj, contract, contract_ref, partition_key="2024", **kwargs):
    return inspect_fragment(store, obj, contract, contract_ref, partition_key, **kwargs)


# --------------------------------------------------------------------------
# D08: logical_partition_hash — cell-content distinctness (no parquet I/O)
# --------------------------------------------------------------------------


def test_logical_hash_distinguishes_null_zero_empty_false():
    contract = _contract("securities")
    ref = _contract_ref(contract)
    # ticker, year, first_date, last_date, mcap_usd, mcap_log, mcap_raw,
    # mcap_unit_era, mcap_quantized, n_obs, src — n_obs varies below.
    base = ["AAA", 2024, "2024-01-02T00:00:00.000000", "2024-12-30T00:00:00.000000",
            100.0, 4.6, 100.0, "billions", False, 10, "orats"]
    hashes = set()
    for value in (None, 0, "", False):
        row = tuple(base[:9] + [value] + base[10:])
        hashes.add(logical_partition_hash(contract, ref, "2024", [row]))
    assert len(hashes) == 4


def test_logical_hash_distinguishes_float_precision():
    contract = _contract("securities")
    ref = _contract_ref(contract)
    base = ["AAA", 2024, "2024-01-02T00:00:00.000000", "2024-12-30T00:00:00.000000",
            100.0, 4.6, 100.0, "billions", False, 10, "orats"]
    row_a = tuple(base[:4] + [100.0] + base[5:])
    row_b = tuple(base[:4] + [100.00000000000001] + base[5:])
    assert (logical_partition_hash(contract, ref, "2024", [row_a])
            != logical_partition_hash(contract, ref, "2024", [row_b]))


# --------------------------------------------------------------------------
# D08: the logical hash is a function of contract-column values, not encoding
# --------------------------------------------------------------------------


def _cast_strings_large(table: pa.Table) -> pa.Table:
    return pa.table({name: (col.cast(pa.large_string()) if pa.types.is_string(col.type) else col)
                     for name, col in zip(table.column_names, table.columns)})


def test_logical_hash_identical_across_parquet_encodings(tmp_path):
    contract = _contract("securities")
    ref = _contract_ref(contract)
    rows = _securities_rows()
    rows[2] = dict(rows[2], mcap_usd=float("nan"))  # legacy_nan_is_null.v1 exercised too
    table = _table_from_rows(contract, rows)
    store = ArtifactStore(tmp_path)

    hashes = set()
    for kwargs in (dict(compression="snappy"), dict(compression="zstd"),
                   dict(row_group_size=1), dict(row_group_size=1000),
                   dict(use_dictionary=True), dict(use_dictionary=False)):
        obj = _publish_bytes(store, _to_bytes(table, **kwargs))
        hashes.add(_inspect(store, obj, contract, ref).logical_content_hash)

    obj = _publish_bytes(store, _to_bytes(_cast_strings_large(table)))
    hashes.add(_inspect(store, obj, contract, ref).logical_content_hash)

    data = _to_bytes(table)
    obj = _publish_bytes(store, data)
    for batch_rows in (1, 1000):
        hashes.add(_inspect(store, obj, contract, ref, batch_rows=batch_rows).logical_content_hash)

    assert len(hashes) == 1


def test_legacy_nan_hashes_identically_to_an_explicit_null(tmp_path):
    contract = _contract("securities")
    ref = _contract_ref(contract)
    store = ArtifactStore(tmp_path)

    rows_nan = _securities_rows()
    rows_nan[2] = dict(rows_nan[2], mcap_usd=float("nan"))
    obj_nan = _publish_bytes(store, _to_bytes(_table_from_rows(contract, rows_nan)))

    rows_null = _securities_rows()
    rows_null[2] = dict(rows_null[2], mcap_usd=None)
    obj_null = _publish_bytes(store, _to_bytes(_table_from_rows(contract, rows_null)))

    assert (_inspect(store, obj_nan, contract, ref).logical_content_hash
            == _inspect(store, obj_null, contract, ref).logical_content_hash)


# --------------------------------------------------------------------------
# D08: planted defects, each refused with its code
# --------------------------------------------------------------------------


def _run(store, contract, ref, rows, *, drop=(), extra=False, partition_key="2024"):
    table = _table_from_rows(contract, rows, drop=drop)
    if extra:
        table = table.append_column("bogus_extra_column",
                                    pa.array([0] * table.num_rows, type=pa.int64()))
    obj = _publish_bytes(store, _to_bytes(table))
    return _inspect(store, obj, contract, ref, partition_key=partition_key)


def test_swapped_rows_is_refused(tmp_path):
    contract = _contract("securities")
    ref = _contract_ref(contract)
    rows = _securities_rows()
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(DataError) as err:
        _run(ArtifactStore(tmp_path), contract, ref, rows)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_duplicate_key_is_refused(tmp_path):
    contract = _contract("securities")
    ref = _contract_ref(contract)
    rows = [dict(r) for r in _securities_rows()]
    rows[1]["ticker"], rows[1]["year"] = rows[0]["ticker"], rows[0]["year"]
    with pytest.raises(DataError) as err:
        _run(ArtifactStore(tmp_path), contract, ref, rows)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_row_from_another_year_is_refused(tmp_path):
    contract = _contract("securities")
    ref = _contract_ref(contract)
    rows = [dict(r) for r in _securities_rows()]
    rows[2]["year"] = 2023
    with pytest.raises(DataError) as err:
        _run(ArtifactStore(tmp_path), contract, ref, rows)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_nan_in_non_nullable_column_is_refused(tmp_path):
    contract = _contract("securities")
    # No column of either real contract this task's tests use is a
    # non-nullable float64; force one so the "NaN in a non-nullable column"
    # rule has a column to fire on, keeping every other field identical to
    # the registered contract (same contract_id/definition_hash).
    forced_columns = tuple(
        dataclasses.replace(c, nullable=False) if c.name == "mcap_usd" else c
        for c in contract.columns
    )
    forced = dataclasses.replace(contract, columns=forced_columns)
    ref = _contract_ref(forced)
    rows = [dict(r) for r in _securities_rows()]
    rows[0]["mcap_usd"] = float("nan")
    with pytest.raises(DataError) as err:
        _run(ArtifactStore(tmp_path), forced, ref, rows)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_infinite_value_is_always_refused(tmp_path):
    contract = _contract("securities")
    ref = _contract_ref(contract)
    rows = [dict(r) for r in _securities_rows()]
    rows[0]["mcap_usd"] = float("inf")
    with pytest.raises(DataError) as err:
        _run(ArtifactStore(tmp_path), contract, ref, rows)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_extra_undeclared_column_is_refused(tmp_path):
    contract = _contract("securities")
    ref = _contract_ref(contract)
    with pytest.raises(DataError) as err:
        _run(ArtifactStore(tmp_path), contract, ref, _securities_rows(), extra=True)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_missing_non_nullable_column_is_refused(tmp_path):
    contract = _contract("securities")
    ref = _contract_ref(contract)
    with pytest.raises(DataError) as err:
        _run(ArtifactStore(tmp_path), contract, ref, _securities_rows(), drop=("ticker",))
    assert err.value.code == "CONTRACT_MISMATCH"


def test_missing_nullable_column_passes(tmp_path):
    contract = _contract("securities")
    ref = _contract_ref(contract)
    inspection = _run(ArtifactStore(tmp_path), contract, ref, _securities_rows(), drop=("src",))
    assert inspection.row_count == 3


def test_daily_market_contract_also_inspects_cleanly(tmp_path):
    """The other real contract this task's tests use, with its timestamp[ns] observation column."""
    contract = _contract("daily_market")
    ref = _contract_ref(contract)
    inspection = _run(ArtifactStore(tmp_path), contract, ref, _daily_market_rows())
    assert inspection.row_count == 3
    assert inspection.primary_key_min == ("AAA", "2024-01-02T00:00:00.000000")
    assert inspection.time_min == "2024-01-02T00:00:00.000000"
    assert inspection.time_max == "2024-01-03T00:00:00.000000"


# --------------------------------------------------------------------------
# D09: publish_legacy_file
# --------------------------------------------------------------------------


def _write_source(root: Path, rel: str, data: bytes) -> LegacyFileRef:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    return LegacyFileRef(path=rel, content_hash=f"sha256:{digest}", byte_size=len(data))


def test_published_bytes_hash_equals_the_source(tmp_path):
    source_root = tmp_path / "source"
    store = ArtifactStore(tmp_path / "store")
    data = b"legacy parquet bytes, standing in for a real file"
    file_ref = _write_source(source_root, "securities/2024.parquet", data)

    obj = publish_legacy_file(store, "att_1", source_root, file_ref)

    assert obj.content_hash == f"sha256:{hashlib.sha256(data).hexdigest()}"
    assert obj.byte_size == len(data)
    assert obj.kind == "parquet_fragment"


def test_symlinked_source_is_refused(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    store = ArtifactStore(tmp_path / "store")
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside bytes")
    (source_root / "link.bin").symlink_to(target)
    file_ref = LegacyFileRef(path="link.bin",
                             content_hash=f"sha256:{hashlib.sha256(b'outside bytes').hexdigest()}",
                             byte_size=13)

    with pytest.raises(DataError) as err:
        publish_legacy_file(store, "att_1", source_root, file_ref)
    assert err.value.code == "INPUT_CHANGED"


def test_hard_linked_source_is_refused(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    store = ArtifactStore(tmp_path / "store")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"hardlinked bytes")
    os.link(outside, source_root / "hl.bin")
    file_ref = LegacyFileRef(path="hl.bin",
                             content_hash=f"sha256:{hashlib.sha256(b'hardlinked bytes').hexdigest()}",
                             byte_size=17)

    with pytest.raises(DataError) as err:
        publish_legacy_file(store, "att_1", source_root, file_ref)
    assert err.value.code == "INPUT_CHANGED"


def test_escaping_path_is_refused(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    store = ArtifactStore(tmp_path / "store")
    (tmp_path / "outside.bin").write_bytes(b"x")
    file_ref = LegacyFileRef(path="../outside.bin",
                             content_hash=f"sha256:{hashlib.sha256(b'x').hexdigest()}", byte_size=1)

    with pytest.raises(DataError) as err:
        publish_legacy_file(store, "att_1", source_root, file_ref)
    assert err.value.code == "INPUT_CHANGED"


def test_wrong_expected_hash_is_refused(tmp_path):
    source_root = tmp_path / "source"
    store = ArtifactStore(tmp_path / "store")
    data = b"real content here"
    (source_root).mkdir()
    (source_root / "real.bin").write_bytes(data)
    file_ref = LegacyFileRef(path="real.bin", content_hash="sha256:" + "0" * 64, byte_size=len(data))

    with pytest.raises(DataError) as err:
        publish_legacy_file(store, "att_1", source_root, file_ref)
    assert err.value.code == "INPUT_CHANGED"


def test_corrupted_published_object_is_refused_at_inspection(tmp_path):
    contract = _contract("securities")
    ref = _contract_ref(contract)
    source_root = tmp_path / "source"
    store = ArtifactStore(tmp_path / "store")
    data = _to_bytes(_table_from_rows(contract, _securities_rows()))
    file_ref = _write_source(source_root, "securities/2024.parquet", data)

    obj = publish_legacy_file(store, "att_1", source_root, file_ref)
    digest = obj.content_hash.removeprefix("sha256:")
    object_path = store.root / "objects" / digest[:2] / digest
    os.chmod(object_path, 0o644)
    flipped = bytearray(object_path.read_bytes())
    flipped[0] ^= 0xFF
    object_path.write_bytes(bytes(flipped))
    os.chmod(object_path, 0o444)

    with pytest.raises(DataError) as err:
        _inspect(store, obj, contract, ref)
    assert err.value.code == "OBJECT_CORRUPT"


class _Crash(Exception):
    pass


def _crash_at(point: str):
    def fault(name: str) -> None:
        if name == point:
            raise _Crash(name)
    return fault


def test_fault_during_copy_raises_and_a_retry_publishes_the_same_object(tmp_path):
    source_root = tmp_path / "source"
    store = ArtifactStore(tmp_path / "store")
    data = b"legacy parquet payload " * 50
    file_ref = _write_source(source_root, "trades/2024.parquet", data)

    with pytest.raises(_Crash):
        publish_legacy_file(store, "att_1", source_root, file_ref, fault=_crash_at("during_copy"))

    first = publish_legacy_file(store, "att_1", source_root, file_ref)
    second = publish_legacy_file(store, "att_2", source_root, file_ref)
    assert first == second
    assert first.content_hash == f"sha256:{hashlib.sha256(data).hexdigest()}"
