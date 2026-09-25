"""Snapshot-pinned table reads for the research tools.

This is the slice's repository-backed replacement for the legacy
``engine.data.store`` helpers. On the sibling research branch the same
functionality lives in ``engine.v2.research._scan``; that module is not
created here (supervisor decision: do not create files whose names collide
with branch ``worktree-agent-aa49bb24e5d746918``), so the two functions this
package needs are defined here under a non-colliding module name.

``read_table`` is deliberately the *only* frame-shaped read: every caller
resolves one signed ``SnapshotRef`` once and this turns it into a pandas frame
over the table's pinned fragments. It never reads the legacy mutable store.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd
import pyarrow as pa

from engine.v2.contracts.data import DataQuery, KeyPredicate

__all__ = [
    "DEFAULT_SCOPE",
    "artifact_store",
    "catalog_connection",
    "head_generation",
    "read_table",
    "resolve_snapshot",
]

#: The scope the research/nightly v2 side writes and reads by default.
DEFAULT_SCOPE = "shadow"


def catalog_connection(repository):
    """The catalog connection behind a ``Repository`` (write-path callers)."""
    return repository._conn


def artifact_store(repository):
    """The ``ArtifactStore`` behind a ``Repository``; refuses a store-less one."""
    store = repository._store
    if store is None:
        raise RuntimeError("this read needs a Repository opened with an ArtifactStore")
    return store


def head_generation(repository, scope: str = DEFAULT_SCOPE) -> int:
    """The scope head's generation — the fence a generic commit must pass.

    ``SNAPSHOT_NOT_READY`` when the scope has no head, the same code
    ``Repository.resolve_pinned`` raises for the identical condition.
    """
    row = repository._conn.execute(
        "SELECT generation FROM data_snapshot_heads WHERE scope = ?", (scope,)
    ).fetchone()
    if row is None:
        from engine.v2.data import errors

        raise errors.fail("SNAPSHOT_NOT_READY", "scope has no committed head",
                          details={"scope": scope})
    return int(row["generation"])


def resolve_snapshot(repository, *, scope: str = DEFAULT_SCOPE, snapshot_id: str | None = None):
    """The pinned snapshot a run should read: explicit id, else the scope head.

    An explicit ``snapshot_id`` is resolved through ``Repository.resolve``
    exactly like a scope head, so a tampered manifest is refused
    (``MANIFEST_CORRUPT``) rather than silently used; a scope with no committed
    head refuses ``SNAPSHOT_NOT_READY``.
    """
    if snapshot_id is not None:
        return repository.resolve(snapshot_id)
    return repository.resolve_pinned(scope)


def _predicate_values(contract, column: str, values: Sequence) -> tuple:
    """Cast ``values`` to the partition column's declared physical type.

    ``operator="in"`` requires one shared scalar type and unique values; a
    caller passing year strings against an int64 partition would otherwise
    match nothing rather than fail loudly.
    """
    physical = next(c.physical_type for c in contract.columns if c.name == column)
    if physical == "int64":
        cast = tuple(dict.fromkeys(int(v) for v in values))
    elif physical == "bool":
        cast = tuple(dict.fromkeys(bool(v) for v in values))
    else:
        cast = tuple(dict.fromkeys(str(v) for v in values))
    return cast


def read_table(repository, snapshot_ref, table_name: str, columns: Sequence[str],
               partition_keys: Sequence[str] | None = None) -> pd.DataFrame:
    """One table's pinned rows, projected to ``columns``, as a frame.

    ``partition_keys``, when given, restricts the scan to the contract's
    declared partition column(s) — the replacement for the legacy
    ``iter_table(..., years=...)`` partition filter.
    """
    contract = repository.table_contract(snapshot_ref, table_name)
    predicates: list[KeyPredicate] = []
    for column in contract.partition_columns:
        values = partition_keys if partition_keys else sorted(
            {record.partition_key for record
             in repository.fragment_records(snapshot_ref, table_name)}
        )
        cast = _predicate_values(contract, column, values)
        if cast:
            predicates.append(KeyPredicate(column=column, operator="in", values=cast))
    if not predicates:
        # A scan must be bounded by at least one key predicate or a time bound.
        raise ValueError(
            f"{table_name}: a bounded read needs partition_keys over a declared "
            "partition column"
        )
    query = DataQuery(
        snapshot_id=snapshot_ref.snapshot_id,
        table_contract_ref=snapshot_ref.table_versions[table_name].table_contract_ref,
        columns=tuple(columns),
        key_filter=tuple(predicates),
        order_by=tuple(contract.primary_key),
        max_batch_rows=min(contract.maximum_batch_rows, contract.maximum_result_rows),
        max_result_rows=contract.maximum_result_rows,
    )
    batches = list(repository.scan(query, table_name=table_name))
    if not batches:
        return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})
    return pa.Table.from_batches(batches).to_pandas().reset_index(drop=True)
