"""Publish one legacy file as an immutable object; inspect it into a fragment
candidate (phase-2 guide §7.2).

Two operations, in order:

* :func:`publish_legacy_file` copies a legacy Parquet file's bytes through
  :class:`engine.v2.foundation.artifacts.ArtifactStore` — never a rename, a
  symlink or a hard link into the object store — and returns the resulting
  ``ObjectRef``.
* :func:`inspect_fragment` re-verifies the published object's byte hash,
  streams it in bounded Arrow batches, validates it against a
  ``TableContract``, and returns a :class:`FragmentInspection`: row count, key
  and time bounds, byte hash, and a streaming ``logical_rows.v1`` content
  hash. It never inserts into a catalog and never derives a fragment/dataset
  ID — that is a later slice's job.

Judgement calls, recorded here rather than silently decided:

* Partition-column values are read directly from the row when the contract
  declares a same-named physical column (true for all eight legacy-mapped
  tables today — every Tier-2 table already carries an explicit ``year``
  column). The "derive year from the date column" fallback in
  :func:`_partition_value` exists only for a future table that omits one; it
  is unexercised by the current mapping and is kept short for that reason.

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts``, ``engine.v2.foundation``, this package's own
``errors``/``time_formats`` (``from . import errors, time_formats``, one
relative-import fan-out edge rather than two absolute ones — the same style
``legacy_adapter.py`` uses), and ``pyarrow`` — never ``engine.v2.ops`` or
legacy ``engine.*``. A handful of stdlib facts (``stat.S_ISREG``, ``uuid4``,
``math.isnan``) are reproduced with bit tests/``os`` primitives instead of
imported, to keep this module's import fan-out inside the §4.3 budget (8,
non-orchestrator).
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

from engine.v2.contracts import (
    ArtifactRef,
    LegacyFileRef,
    ObjectRef,
    TableContract,
    TableContractRef,
)
from engine.v2.foundation import (
    CONTENT_HASH_PREFIX,
    ArtifactError,
    ArtifactStore,
    canonical_json,
    safe_relative_path,
)

from . import errors, time_formats

__all__ = [
    "FragmentInspection",
    "LOGICAL_ROWS_ALGORITHM",
    "NAN_POLICY",
    "PARQUET_FRAGMENT_SCHEMA_REF",
    "inspect_fragment",
    "inspect_staged_file",
    "inspect_staged_partition",
    "logical_partition_hash",
    "normalize_physical_type",
    "partition_logical_hash",
    "publish_legacy_file",
    "verify_object_path",
]

PARQUET_FRAGMENT_SCHEMA_REF = "parquet_fragment.v1"
LOGICAL_ROWS_ALGORITHM = "logical_rows.v1"
#: This package's judgement call (task brief): legacy pandas NaN in a
#: nullable float64 column encodes as JSON null, identically to a Parquet
#: null; NaN in a non-nullable column and any ±Inf are always refused.
NAN_POLICY = "legacy_nan_is_null.v1"

_CHUNK = 1 << 20
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_EPOCH = datetime(1970, 1, 1)
#: ``stat.S_IFMT``/``stat.S_IFREG``, inlined rather than imported (see the
#: module docstring's fan-out note).
_S_IFMT = 0o170000
_S_IFREG = 0o100000

#: pyarrow's own spellings for two contract physical types. Every other
#: pyarrow type string — including a timestamp's unit — passes through
#: unchanged, so a genuine ``timestamp[ns]`` vs ``timestamp[us]`` mismatch is
#: still reported rather than aliased away.
_PHYSICAL_TYPE_ALIASES: dict[str, str] = {"large_string": "string", "double": "float64"}


def normalize_physical_type(actual: str) -> str:
    """Pyarrow's spelling of a physical type, aliased to the contract vocabulary.

    The single normalization rule shared by ``inspect_fragment``'s contract
    check and the P2-1b private-schema test against the real curated store.
    """
    return _PHYSICAL_TYPE_ALIASES.get(actual, actual)


# --------------------------------------------------------------------------
# publish_legacy_file
# --------------------------------------------------------------------------


def publish_legacy_file(store: ArtifactStore, attempt_id: str, source_root, file_ref: LegacyFileRef, *,
                        fault=None) -> ObjectRef:
    """Copy one legacy file's bytes into ``store`` and return its ``ObjectRef``.

    Never a rename, a symlink or a hard link: ``file_ref.path`` is opened
    beneath ``source_root`` (a path or ``os.PathLike``) refusing a symlink at
    any path component, then streamed into the store's staging area while
    hashing, so the published bytes are exactly the bytes that were hashed.
    ``fault``, when given, is a one-argument callable invoked by name at each
    named crash point (``before_copy``, ``during_copy``, ``after_publication``).
    """
    fault = fault or (lambda point: None)
    parts = _safe_source_parts(file_ref)
    fault("before_copy")
    src_fd = _open_legacy_source(os.path.realpath(os.fspath(source_root)), parts)
    try:
        _check_regular_private_copy(src_fd)
        os.set_blocking(src_fd, True)
        rel, hexdigest, size = _copy_into_staging(store, attempt_id, src_fd, fault)
    finally:
        os.close(src_fd)
    _check_matches_reference(store, attempt_id, rel, hexdigest, size, file_ref)
    ref = store.publish_candidate(attempt_id, rel, schema_ref=PARQUET_FRAGMENT_SCHEMA_REF)
    fault("after_publication")
    return ObjectRef(kind="parquet_fragment", object_id=ref.artifact_id,
                     content_hash=ref.content_hash, byte_size=ref.byte_size)


def _safe_source_parts(file_ref: LegacyFileRef) -> tuple[str, ...]:
    try:
        return safe_relative_path(file_ref.path)
    except ArtifactError as exc:
        raise errors.fail("INPUT_CHANGED", "legacy file reference path is not a safe relative path") from exc


def _open_legacy_source(source_root: str, parts: tuple[str, ...]) -> int:
    """Open ``source_root/parts...`` refusing a symlink at every component."""
    try:
        fd = os.open(source_root, _DIR_FLAGS)
    except OSError as exc:
        raise errors.fail("INPUT_CHANGED", "legacy source root is not a real directory") from exc
    try:
        for part in parts[:-1]:
            nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return os.open(parts[-1], _FILE_FLAGS, dir_fd=fd)
    except OSError as exc:
        raise errors.fail("INPUT_CHANGED", "legacy source file could not be opened safely") from exc
    finally:
        os.close(fd)


def _check_regular_private_copy(src_fd: int) -> None:
    info = os.fstat(src_fd)
    if (info.st_mode & _S_IFMT) != _S_IFREG:
        raise errors.fail("INPUT_CHANGED", "legacy source is not a regular file")
    if info.st_nlink != 1:
        raise errors.fail("INPUT_CHANGED", "legacy source has more than one hard link")


def _copy_into_staging(store: ArtifactStore, attempt_id: str, src_fd: int, fault) -> tuple[str, str, int]:
    staging_dir = store.staging_dir(attempt_id)
    rel = f"{os.urandom(16).hex()}.parquet"
    dst_fd = os.open(staging_dir / rel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        digest, size = hashlib.sha256(), 0
        while True:
            fault("during_copy")
            chunk = os.read(src_fd, _CHUNK)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
            _write_all(dst_fd, chunk)
        os.fsync(dst_fd)
    except BaseException:
        os.close(dst_fd)
        (staging_dir / rel).unlink(missing_ok=True)
        raise
    os.close(dst_fd)
    return rel, digest.hexdigest(), size


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def _check_matches_reference(store: ArtifactStore, attempt_id: str, rel: str, hexdigest: str,
                             size: int, file_ref: LegacyFileRef) -> None:
    actual_hash = CONTENT_HASH_PREFIX + hexdigest
    if size != file_ref.byte_size or actual_hash != file_ref.content_hash:
        (store.staging_dir(attempt_id) / rel).unlink(missing_ok=True)
        raise errors.fail("INPUT_CHANGED", "legacy source bytes do not match its recorded file reference")


# --------------------------------------------------------------------------
# inspect_fragment
# --------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class FragmentInspection:
    """What one immutable object measures out to as a fragment candidate.

    No ``fragment_id``, no ``manifest_hash``, no catalog row: those are a
    later slice's job (phase-2 guide §7.2 scope for this one).
    """

    object_ref: ObjectRef
    partition_key: str
    row_count: int
    byte_hash: str
    logical_content_hash: str
    primary_key_min: tuple[str | int | bool, ...] | None
    primary_key_max: tuple[str | int | bool, ...] | None
    time_min: str | None
    time_max: str | None


@dataclass
class _StreamState:
    row_count: int = 0
    previous_key: tuple | None = None
    key_min: tuple | None = None
    key_max: tuple | None = None
    time_min: str | None = None
    time_max: str | None = None


def inspect_fragment(store: ArtifactStore, object_ref: ObjectRef, contract: TableContract,
                     contract_ref: TableContractRef, partition_key: str, *,
                     batch_rows: int = 65536, fault=None) -> FragmentInspection:
    """Verify, stream and hash one published object as a fragment candidate.

    ``fault``, when given, is invoked with ``"during_inspection"`` once per
    Arrow batch. Carries no ``import_request_hash``: ``manifests.fragment_record``
    (P2-2c) takes that directly from its caller instead.
    """
    fault = fault or (lambda point: None)
    path = verify_object_path(store, object_ref)
    inspection = _inspect_path(path, contract, contract_ref, partition_key, batch_rows, fault)
    return dataclasses.replace(inspection, object_ref=object_ref, byte_hash=object_ref.content_hash)


def _inspect_path(path, contract: TableContract, contract_ref: TableContractRef,
                  partition_key: str, batch_rows: int, fault) -> FragmentInspection:
    """Shared streaming/decoding core of :func:`inspect_fragment` and
    :func:`inspect_staged_file`. ``object_ref``/``byte_hash`` are left as
    placeholders (``None``/``""``) here — each caller fills them from what it
    already knows (a published ``ObjectRef``, or a staged file's pinned
    ``LegacyFileRef``) rather than this function re-deriving them.
    """
    parquet_file = _open_parquet_file(path)
    present, missing = _match_contract_columns(contract, parquet_file.schema_arrow)
    state = _StreamState()
    rows = _stream_rows(parquet_file, contract, present, missing, partition_key, batch_rows,
                        state, fault)
    logical_hash = logical_partition_hash(contract, contract_ref, partition_key, rows)
    return FragmentInspection(
        object_ref=None, partition_key=partition_key, row_count=state.row_count,
        byte_hash="", logical_content_hash=logical_hash,
        primary_key_min=state.key_min, primary_key_max=state.key_max,
        time_min=state.time_min, time_max=state.time_max,
    )


def verify_object_path(store: ArtifactStore, object_ref: ObjectRef):
    """Re-hash ``object_ref`` against ``store`` (TD-1: no cache, every open) and
    return its path. Shared by :func:`inspect_fragment` and the repository's
    scan path (P2-4), so the two never diverge on how an object is opened.
    """
    digest = object_ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
    ref = ArtifactRef(artifact_id=object_ref.object_id, content_hash=object_ref.content_hash,
                      schema_ref=PARQUET_FRAGMENT_SCHEMA_REF, byte_size=object_ref.byte_size,
                      storage_key=f"objects/{digest[:2]}/{digest}")
    try:
        return store.verify(ref)
    except ArtifactError as exc:
        raise errors.fail("OBJECT_CORRUPT", "published object bytes do not match its recorded hash") from exc


def inspect_staged_file(path, contract: TableContract, contract_ref: TableContractRef,
                        partition_key: str, *, expected_content_hash: str,
                        expected_byte_size: int, batch_rows: int = 65536,
                        fault=None) -> FragmentInspection:
    """Verify, stream and hash one legacy file staged locally — the worker side
    of the §7 snapshot-import worker/coordinator split (task brief decision 1).

    Unlike :func:`inspect_fragment`, ``path`` is never published as an
    ``ArtifactStore`` object first: this is a fresh legacy file sitting in a
    worker's own staging directory, already re-verified once when the read set
    was pinned/copied (``ops.store_barrier``). This function re-verifies its
    bytes again, independently, against the same pinned
    ``expected_content_hash``/``expected_byte_size`` — before opening it as
    Parquet — then runs exactly ``inspect_fragment``'s own streaming/decoding
    path. The returned ``object_ref`` carries no real ``object_id`` (nothing is
    published yet); a coordinator that later publishes this same path replaces
    it with the real ``ObjectRef`` (``dataclasses.replace``) before building a
    ``FragmentRecord`` — the published bytes are guaranteed identical, since
    :func:`publish_legacy_file` independently re-hashes and refuses a mismatch.
    """
    fault = fault or (lambda point: None)
    _check_staged_bytes(path, expected_content_hash, expected_byte_size)
    inspection = _inspect_path(path, contract, contract_ref, partition_key, batch_rows, fault)
    placeholder = ObjectRef(kind="parquet_fragment", object_id="", content_hash=expected_content_hash,
                            byte_size=expected_byte_size)
    return dataclasses.replace(inspection, object_ref=placeholder, byte_hash=expected_content_hash)


def inspect_staged_partition(files, contract: TableContract, contract_ref: TableContractRef,
                             partition_key: str, *, batch_rows: int = 65536
                             ) -> tuple[list[FragmentInspection], str | None]:
    """One streaming pass over a partition's ordered staged files (the §7 worker).

    ``files`` is ``[(path, expected_content_hash, expected_byte_size), ...]`` in
    partition order. Each file is byte-verified as :func:`inspect_staged_file`
    does; every decoded row then feeds both its fragment's ``logical_rows.v1``
    digest and the partition's combined digest, so a multi-fragment partition
    costs no second pass. Key order is enforced across fragment seams too. The
    combined hash equals :func:`partition_logical_hash` over the same bytes once
    published, and is ``None`` for a single fragment (its own hash is the
    partition's).
    """
    combined = _logical_digest(contract, contract_ref, partition_key)
    seam, inspections = _StreamState(), []
    for path, expected_hash, expected_size in files:
        _check_staged_bytes(path, expected_hash, expected_size)
        parquet_file = _open_parquet_file(path)
        present, missing = _match_contract_columns(contract, parquet_file.schema_arrow)
        state, digest = _StreamState(), _logical_digest(contract, contract_ref, partition_key)
        for row in _stream_rows(parquet_file, contract, present, missing, partition_key,
                                batch_rows, state, lambda point: None):
            line = b"\n" + canonical_json(row).encode("utf-8")
            digest.update(line)
            combined.update(line)
        if state.key_min is not None:
            _check_key_order(seam, state.key_min)
            seam.previous_key = state.key_max
        inspections.append(FragmentInspection(
            object_ref=ObjectRef(kind="parquet_fragment", object_id="", content_hash=expected_hash,
                                 byte_size=expected_size),
            partition_key=partition_key, row_count=state.row_count, byte_hash=expected_hash,
            logical_content_hash=CONTENT_HASH_PREFIX + digest.hexdigest(),
            primary_key_min=state.key_min, primary_key_max=state.key_max,
            time_min=state.time_min, time_max=state.time_max))
    partition_hash = CONTENT_HASH_PREFIX + combined.hexdigest() if len(inspections) > 1 else None
    return inspections, partition_hash


def _check_staged_bytes(path, expected_content_hash: str, expected_byte_size: int) -> None:
    actual_hash, actual_size = _hash_local_file(path)
    if actual_size != expected_byte_size or actual_hash != expected_content_hash:
        raise errors.fail("INPUT_CHANGED",
                          "staged legacy file no longer matches its pinned reference")


def _hash_local_file(path) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return CONTENT_HASH_PREFIX + digest.hexdigest(), size


def _open_parquet_file(path) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(path)
    except (OSError, pa.ArrowException) as exc:
        raise errors.fail("OBJECT_CORRUPT", "parquet footer could not be read") from exc


def _match_contract_columns(contract: TableContract, schema: pa.Schema) -> tuple[list[str], list[str]]:
    declared = {c.name: c for c in contract.columns}
    file_names = set(schema.names)
    extra = sorted(file_names - set(declared))
    if extra:
        raise errors.fail("CONTRACT_MISMATCH", f"undeclared column(s) in the parquet file: {extra}")
    present, missing = [], []
    for column in contract.columns:
        if column.name in file_names:
            _check_column_type(column, schema)
            present.append(column.name)
        elif column.nullable:
            missing.append(column.name)
        else:
            raise errors.fail("CONTRACT_MISMATCH", f"missing non-nullable column {column.name!r}")
    return present, missing


def _check_column_type(column, schema: pa.Schema) -> None:
    actual = normalize_physical_type(str(schema.field(column.name).type))
    if actual != column.physical_type:
        raise errors.fail("CONTRACT_MISMATCH",
                   f"{column.name}: contract declares {column.physical_type!r}, parquet has {actual!r}")


def _stream_rows(parquet_file: pq.ParquetFile, contract: TableContract, present: list[str],
                 missing: list[str], partition_key: str, batch_rows: int, state: _StreamState, fault):
    for batch in _iter_batches(parquet_file, present, batch_rows):
        fault("during_inspection")
        decoded = _decode_batch(batch, contract, missing)
        for i in range(batch.num_rows):
            yield _process_row(contract, decoded, i, partition_key, state)


def _iter_batches(parquet_file: pq.ParquetFile, present: list[str], batch_rows: int):
    try:
        yield from parquet_file.iter_batches(batch_size=batch_rows, columns=present)
    except (OSError, pa.ArrowException) as exc:
        raise errors.fail("OBJECT_CORRUPT", "parquet batch could not be decoded") from exc


def _process_row(contract: TableContract, decoded: dict, i: int, partition_key: str,
                 state: _StreamState) -> tuple:
    key = tuple(decoded[name][i] for name in contract.primary_key)
    _check_key_order(state, key)
    _check_partition(contract, decoded, i, partition_key)
    _update_time_bounds(contract, decoded, i, state)
    state.row_count += 1
    return tuple(decoded[c.name][i] for c in contract.columns)


def _check_key_order(state: _StreamState, key: tuple) -> None:
    if state.previous_key is None:
        state.key_min = key
    elif key == state.previous_key:
        raise errors.fail("CONTRACT_MISMATCH", "duplicate primary key")
    elif key < state.previous_key:
        raise errors.fail("CONTRACT_MISMATCH", "primary key is out of order")
    state.key_max = key
    state.previous_key = key


def _check_partition(contract: TableContract, decoded: dict, i: int, partition_key: str) -> None:
    for name in contract.partition_columns:
        value = _partition_value(contract, decoded, name, i)
        if str(value) != partition_key:
            raise errors.fail("CONTRACT_MISMATCH", f"row does not belong to partition {name}={partition_key!r}")


def _partition_value(contract: TableContract, decoded: dict, name: str, i: int):
    if name in decoded:
        return decoded[name][i]
    # Unexercised by the current legacy mapping — see the module docstring.
    if name == "year" and contract.observation_time_column in decoded:
        raw = decoded[contract.observation_time_column][i]
        return None if raw is None else int(raw[:4])
    raise errors.fail("CONTRACT_MISMATCH", f"partition column {name!r} cannot be derived")


def _update_time_bounds(contract: TableContract, decoded: dict, i: int, state: _StreamState) -> None:
    column = contract.observation_time_column
    if not column:
        return
    value = decoded[column][i]
    if value is None:
        return
    if state.time_min is None or value < state.time_min:
        state.time_min = value
    if state.time_max is None or value > state.time_max:
        state.time_max = value


# --------------------------------------------------------------------------
# batch decoding — legacy NaN/naive-timestamp policy applied once, streaming
# --------------------------------------------------------------------------


def _decode_batch(batch: pa.RecordBatch, contract: TableContract, missing: list[str]) -> dict:
    n = batch.num_rows
    out: dict[str, list] = {}
    for column in contract.columns:
        if column.name in missing:
            out[column.name] = [None] * n
        else:
            out[column.name] = _decode_column(column, batch.column(column.name))
    return out


def _decode_column(column, array: pa.Array) -> list:
    physical_type = column.physical_type
    if physical_type == "float64":
        return [_decode_float(column, v) for v in array.to_pylist()]
    if physical_type.startswith("timestamp["):
        return _decode_timestamps(column, array, physical_type[len("timestamp["):-1])
    if physical_type == "date":
        return [None if v is None else v.isoformat() for v in array.to_pylist()]
    return array.to_pylist()


def _decode_float(column, value: float | None) -> float | None:
    if value is None:
        return None
    if value != value:  # NaN is the only float that compares unequal to itself
        if column.nullable:
            return None
        raise errors.fail("CONTRACT_MISMATCH", f"{column.name}: NaN in a non-nullable column")
    if value in (float("inf"), float("-inf")):
        raise errors.fail("CONTRACT_MISMATCH", f"{column.name}: an infinite value is never valid")
    return value


def _decode_timestamps(column, array: pa.Array, unit: str) -> list[str | None]:
    ticks = array.cast(pa.int64()).to_pylist()
    out: list[str | None] = []
    for tick in ticks:
        if tick is None:
            out.append(None)
            continue
        if unit == "ns":
            if tick % 1000 != 0:
                raise errors.fail("CONTRACT_MISMATCH",
                           f"{column.name}: sub-microsecond timestamp precision is refused")
            micros = tick // 1000
        else:
            micros = tick
        out.append(_format_naive_timestamp(micros))
    return out


def _format_naive_timestamp(micros: int) -> str:
    return (_EPOCH + timedelta(microseconds=micros)).strftime(time_formats.NAIVE_TIMESTAMP_FORMAT)


# --------------------------------------------------------------------------
# logical_rows.v1 — a separately testable streaming hash
# --------------------------------------------------------------------------


def logical_partition_hash(contract: TableContract, contract_ref: TableContractRef,
                           partition_key: str, rows) -> str:
    """Stream ``rows`` (contract-column-ordered, already NaN/timestamp-normalized
    scalars) into the ``logical_rows.v1`` hash. Never holds the partition in memory.
    """
    digest = _logical_digest(contract, contract_ref, partition_key)
    for row in rows:
        digest.update(b"\n")
        digest.update(canonical_json(row).encode("utf-8"))
    return CONTENT_HASH_PREFIX + digest.hexdigest()


def _logical_digest(contract: TableContract, contract_ref: TableContractRef, partition_key: str):
    """A ``logical_rows.v1`` sha256 with its header already written."""
    header = {
        "algorithm": LOGICAL_ROWS_ALGORITHM,
        "table_contract_ref": {"contract_id": contract_ref.contract_id,
                               "definition_hash": contract_ref.definition_hash},
        "partition_key": partition_key,
        "columns": [{"name": c.name, "physical_type": c.physical_type} for c in contract.columns],
        "nan_policy": NAN_POLICY,
    }
    digest = hashlib.sha256()
    digest.update(canonical_json(header).encode("utf-8"))
    return digest


# --------------------------------------------------------------------------
# partition_logical_hash — several ordered fragments as one logical partition
# --------------------------------------------------------------------------


def partition_logical_hash(store: ArtifactStore, object_refs_in_order: list[ObjectRef],
                           contract: TableContract, contract_ref: TableContractRef,
                           partition_key: str, *, batch_rows: int = 65536) -> str:
    """A logical partition may hold several ordered, non-overlapping fragments
    (task brief decision 2): stream every one of ``object_refs_in_order`` in
    order and hash them as if they were one file, never holding the partition
    in memory.

    Exactly one ``logical_rows.v1`` header is emitted, then every row of every
    fragment. Strict primary-key order and uniqueness are enforced *across*
    fragment boundaries too, by sharing one :class:`_StreamState` across every
    fragment: the same :func:`_check_key_order` that guards one fragment in
    :func:`inspect_fragment` also guards the seam between two fragments here,
    since it only ever compares a row to the previous row it saw — it does
    not know or care whether that previous row came from the same file. With
    bounded memory, this equals :func:`logical_partition_hash` over the same
    rows written as one file (D08); for a single-fragment partition it equals
    that fragment's own hash.
    """
    state = _StreamState()
    no_fault = lambda point: None  # noqa: E731 — streamed once per fragment, not worth a fault hook

    def rows():
        for object_ref in object_refs_in_order:
            path = verify_object_path(store, object_ref)
            parquet_file = _open_parquet_file(path)
            present, missing = _match_contract_columns(contract, parquet_file.schema_arrow)
            yield from _stream_rows(parquet_file, contract, present, missing, partition_key,
                                    batch_rows, state, no_fault)

    return logical_partition_hash(contract, contract_ref, partition_key, rows())
