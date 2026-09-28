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

from engine.v2.research import _scan

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


def read_table(repository, snapshot_ref, table_name: str, columns: Sequence[str],
               partition_keys: Sequence[str] | None = None) -> pd.DataFrame:
    """One table's pinned rows, projected to ``columns``, as a frame.

    ``partition_keys``, when given, restricts the scan to the contract's
    declared partition column(s) — the replacement for the legacy
    ``iter_table(..., years=...)`` partition filter.

    The actual scan is delegated to ``_scan.read_table``, which bounds each
    partition's read and splits it into calendar months, then days, on
    ``RESULT_LIMIT_EXCEEDED`` (issue #107) — this function keeps its own
    partition-key resolution, its own refusal when no partition column or
    partition value is available at all (issue #70, unchanged), and its own
    empty-result frame shape.
    """
    contract = repository.table_contract(snapshot_ref, table_name)
    keys = partition_keys if partition_keys else sorted(
        {record.partition_key for record
         in repository.fragment_records(snapshot_ref, table_name)}
    )
    if not contract.partition_columns or not keys:
        # A scan must be bounded by at least one key predicate or a time bound.
        raise ValueError(
            f"{table_name}: a bounded read needs partition_keys over a declared "
            "partition column"
        )
    frame = _scan.read_table(repository, snapshot_ref, table_name, columns,
                             partition_keys=keys)
    if frame.empty:
        return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})
    return frame
