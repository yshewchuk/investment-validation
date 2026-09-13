"""The data-owner catalog schema, as one numbered migration — phase-2 guide §6.

Colocated with the ops/ledger schemas in the same SQLite file, under its own
migration owner ``"data"`` (phase-2 guide §3.3 "One SQLite file, separate
schema owner"). This module declares only ``OWNER`` and ``MIGRATIONS`` as
plain, immutable ``(version, name, statements)`` tuples — never
``engine.v2.ops.migrations.Migration`` objects, because that class lives on a
higher layer. ``engine/v2/ops/bootstrap.py`` imports this module, wraps each
tuple into a ``Migration``, and applies it with the existing migration
machinery after the ops/ledger owners. This module never imports
``engine.v2.ops``.

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts`` and ``engine.v2.foundation``, matching
``engine/v2/data/manifests.py`` and ``legacy_adapter.py``.

Invariants live in the schema, not only in a later Python precheck (phase-2
guide §6):

1. Contract, object, fragment, dataset-version, snapshot, snapshot-table
   membership, and import-receipt rows are append-only: every ``data_*``
   table except ``data_snapshot_heads`` refuses ``UPDATE``/``DELETE`` with a
   ``BEFORE`` trigger that ``RAISE(ABORT)``s.
2. Only ``data_snapshot_heads`` is mutable, and only forward by exactly one
   generation without changing scope; it refuses ``DELETE`` outright. The
   schema enforces the *shape* of a valid update (same scope, ``generation =
   generation + 1``, and — via the ordinary foreign key — an existing
   snapshot); the *compare-and-swap* itself (``UPDATE ... WHERE scope=? AND
   snapshot_id=? AND generation=?``, zero rows changed is a conflict) is a
   caller discipline no schema constraint can express, because SQLite has no
   way to make an ``UPDATE`` fail merely for changing zero rows — that check
   belongs to the Python layer that reads ``cursor.rowcount`` (a later task).
3. A snapshot commit transaction inserts new immutable metadata, complete
   membership, and an import receipt together; that atomicity is a property
   of the caller's transaction boundary (``engine.v2.ops.catalog.transaction``
   is one ``BEGIN IMMEDIATE``), not something a single table's DDL can compel
   on its own — nothing here relies on deferred foreign keys to achieve it,
   because ordering the ``CREATE TABLE`` statements so every foreign key
   already has its referent avoids that complexity entirely (see the
   docstring note below on ``commit_receipt_ref``/``update_receipt_ref``).
4. Every dataset version is a complete logical view: ``data_version_fragments``
   carries full membership per version; nothing here reads
   ``parent_dataset_version_id`` to reconstruct it. That is a *reader*
   discipline (guide §8.1), not a schema constraint.
5. A fragment object is durable and re-verified before its catalog row is
   inserted. That ordering is a caller/process discipline; a catalog row
   cannot itself prove the bytes it names were fsynced first.
6. Dataset row count equals the sum of its unique fragment row counts, and
   fragment key ranges do not overlap within a logical partition. **Not
   expressed in SQL**: both checks are aggregates over a dataset version's
   full fragment membership, which is only complete once every
   ``data_version_fragments`` row has been inserted — SQLite has no
   deferred-trigger mechanism (only deferred foreign keys) to validate an
   aggregate once a multi-row insert sequence finishes. This is left as a
   Python-level check when the manifest-commit code is written (P2-3).
7. Snapshot table names are unique, known, and bound to the matching
   contract: ``(snapshot_id, table_name)`` is the primary key, and a
   ``BEFORE INSERT`` trigger on ``data_snapshot_tables`` rejects a row whose
   ``dataset_version_id`` resolves (through ``data_dataset_versions`` and
   ``data_contracts``) to a different ``table_name``.
8. ``knowledge_mode`` is ``observed``, ``attested_stable``, or
   ``reconstructed``; ``observed``/``attested_stable`` require a non-empty
   evidence-ref array. Guide §3.3 fixes knowledge mode as *per table
   version*, so this is enforced once, as a plain ``CHECK`` on
   ``data_dataset_versions`` (the only table that carries a scalar
   ``knowledge_mode``) — every table named in a snapshot's
   ``knowledge_mode_by_table`` map already resolved through a
   ``data_dataset_versions`` row that satisfied this same check, so the
   per-element values inside ``data_snapshots.knowledge_mode_by_table_json``
   only need a *closed-vocabulary* check (a ``BEFORE INSERT`` trigger over
   ``json_each``, since a plain ``CHECK`` cannot iterate a JSON array/object
   in SQLite), not invariant 8's evidence requirement a second time.

Judgement calls, recorded here rather than only in the task report:

* ``commit_receipt_ref`` (``data_snapshots``) and ``update_receipt_ref``
  (``data_snapshot_heads``) are plain ``TEXT`` columns, **not** declared
  foreign keys to ``data_import_receipts``. The guide's explicit foreign-key
  list (§6, "object, contract, parent version/snapshot, fragment, dataset
  version, head snapshot") does not name a receipt target, and
  ``data_import_receipts.result_snapshot_id`` already references
  ``data_snapshots`` the other way — making the receipt ref a real foreign
  key too would force a circular table dependency (deferred foreign keys,
  forward-declared ``CREATE TABLE``s) for a fact a join can otherwise recover
  cheaply. Referential integrity for these two columns is left to whatever
  Python code writes them.
* Hash columns use the exact check the task describes: length 71,
  ``sha256:`` prefix, and the 64-character suffix restricted to lowercase hex
  via ``NOT GLOB '*[^0-9a-f]*'``. The literal prefix comes from
  ``engine.v2.foundation.CONTENT_HASH_PREFIX`` rather than being retyped, so
  the two can never silently drift apart.
* JSON columns are validated with both ``json_valid`` and ``json_type`` (an
  ``object`` doc must not silently accept a JSON array, and vice versa).
* ``data_fragments.key_bounds_json`` holds
  ``{"primary_key_min": [...], "primary_key_max": [...]}``;
  ``time_bounds_json`` (nullable — many datasets have no time column) holds
  ``{"time_min": ..., "time_max": ...}``. ``data_dataset_versions.evidence_json``
  holds ``{"coverage_receipt_refs": [...], "availability_evidence_refs":
  [...]}`` — the two ``DatasetManifest`` evidence fields (component contracts
  §5.2), combined into the one "evidence JSON" column §6 names.
"""
from __future__ import annotations

from typing import get_args

from engine.v2.contracts.data import KnowledgeMode
from engine.v2.foundation import CONTENT_HASH_PREFIX

__all__ = ["MIGRATIONS", "OWNER"]

OWNER = "data"

#: Mirrors ``engine.v2.contracts.data.KnowledgeMode`` exactly, read off the
#: ``Literal`` itself so the schema can never silently diverge from it.
_KNOWLEDGE_MODES = get_args(KnowledgeMode)

#: Mirrors ``engine.v2.contracts.data.SnapshotImportReceipt.status``. That
#: field has no separately exported ``Literal`` alias to read via
#: ``get_args`` (unlike ``KnowledgeMode``), so it is transcribed once here,
#: the same way ``engine/v2/ops/schema.py`` hardcodes its own state tuples.
_RECEIPT_STATUSES = ("committed", "failed", "conflict")

_HASH_LEN = len(CONTENT_HASH_PREFIX) + 64


def _hash_check(column: str) -> str:
    """``column`` is ``sha256:`` + 64 lowercase hex characters, exactly."""
    prefix_len = len(CONTENT_HASH_PREFIX)
    return (
        f"length({column}) = {_HASH_LEN} AND "
        f"substr({column}, 1, {prefix_len}) = '{CONTENT_HASH_PREFIX}' AND "
        f"substr({column}, {prefix_len + 1}) NOT GLOB '*[^0-9a-f]*'"
    )


def _json_check(column: str, json_type: str = "object", *, nullable: bool = False) -> str:
    """``column`` is valid JSON of the given top-level ``json_type``."""
    valid = f"json_valid({column}) AND json_type({column}) = '{json_type}'"
    return f"{column} IS NULL OR ({valid})" if nullable else valid


def _immutable_triggers(table: str) -> tuple[str, str]:
    """``BEFORE UPDATE``/``BEFORE DELETE`` refusal pair (invariant 1)."""
    return (
        f"""CREATE TRIGGER {table}_no_update
        BEFORE UPDATE ON {table}
        BEGIN
            SELECT RAISE(ABORT, '{table} rows are immutable');
        END""",
        f"""CREATE TRIGGER {table}_no_delete
        BEFORE DELETE ON {table}
        BEGIN
            SELECT RAISE(ABORT, '{table} rows are immutable');
        END""",
    )


# --------------------------------------------------------------------------
# data_contracts — TableContract (Definition), phase-2 guide §5.1
# --------------------------------------------------------------------------

_CONTRACTS = (
    f"""CREATE TABLE data_contracts (
        contract_id TEXT PRIMARY KEY,
        schema_version TEXT NOT NULL,
        table_name TEXT NOT NULL,
        definition_hash TEXT NOT NULL UNIQUE,
        definition_json TEXT NOT NULL,
        registered_at TEXT NOT NULL,
        CHECK ({_hash_check('definition_hash')}),
        CHECK ({_json_check('definition_json')})
    ) STRICT""",
    *_immutable_triggers("data_contracts"),
)


# --------------------------------------------------------------------------
# data_objects — ObjectRef (Handle), phase-2 guide §5.2
# --------------------------------------------------------------------------

_OBJECTS = (
    f"""CREATE TABLE data_objects (
        object_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
        storage_key TEXT NOT NULL,
        registered_at TEXT NOT NULL,
        UNIQUE (kind, content_hash),
        CHECK ({_hash_check('content_hash')})
    ) STRICT""",
    *_immutable_triggers("data_objects"),
)


# --------------------------------------------------------------------------
# data_fragments — FragmentRecord, phase-2 guide §5.2
# --------------------------------------------------------------------------

_FRAGMENTS = (
    f"""CREATE TABLE data_fragments (
        fragment_id TEXT PRIMARY KEY,
        object_id TEXT NOT NULL REFERENCES data_objects(object_id),
        contract_id TEXT NOT NULL REFERENCES data_contracts(contract_id),
        partition_key TEXT NOT NULL,
        row_count INTEGER NOT NULL CHECK (row_count >= 0),
        byte_hash TEXT NOT NULL,
        logical_content_hash TEXT NOT NULL,
        key_bounds_json TEXT NOT NULL,
        time_bounds_json TEXT,
        import_request_hash TEXT NOT NULL,
        registered_at TEXT NOT NULL,
        CHECK ({_hash_check('byte_hash')}),
        CHECK ({_hash_check('logical_content_hash')}),
        CHECK ({_hash_check('import_request_hash')}),
        CHECK ({_json_check('key_bounds_json')}),
        CHECK ({_json_check('time_bounds_json', nullable=True)})
    ) STRICT""",
    "CREATE INDEX data_fragments_object ON data_fragments(object_id)",
    "CREATE INDEX data_fragments_contract ON data_fragments(contract_id)",
    *_immutable_triggers("data_fragments"),
)


# --------------------------------------------------------------------------
# data_dataset_versions — DatasetManifest/DatasetVersionRef, guide §5.2
# --------------------------------------------------------------------------

_DATASET_VERSIONS = (
    f"""CREATE TABLE data_dataset_versions (
        dataset_version_id TEXT PRIMARY KEY,
        contract_id TEXT NOT NULL REFERENCES data_contracts(contract_id),
        parent_dataset_version_id TEXT REFERENCES data_dataset_versions(dataset_version_id),
        manifest_hash TEXT NOT NULL UNIQUE,
        logical_content_hash TEXT NOT NULL,
        row_count INTEGER NOT NULL CHECK (row_count >= 0),
        knowledge_mode TEXT NOT NULL CHECK (knowledge_mode IN {_KNOWLEDGE_MODES!r}),
        evidence_json TEXT NOT NULL,
        registered_at TEXT NOT NULL,
        CHECK ({_hash_check('manifest_hash')}),
        CHECK ({_hash_check('logical_content_hash')}),
        CHECK ({_json_check('evidence_json')}),
        CHECK (
            knowledge_mode = 'reconstructed' OR (
                json_type(json_extract(evidence_json, '$.availability_evidence_refs')) = 'array'
                AND json_array_length(evidence_json, '$.availability_evidence_refs') > 0
            )
        )
    ) STRICT""",
    "CREATE INDEX data_dataset_versions_contract ON data_dataset_versions(contract_id)",
    *_immutable_triggers("data_dataset_versions"),
)


# --------------------------------------------------------------------------
# data_version_fragments — complete, ordered fragment membership, guide §5.2
# --------------------------------------------------------------------------

_VERSION_FRAGMENTS = (
    """CREATE TABLE data_version_fragments (
        dataset_version_id TEXT NOT NULL REFERENCES data_dataset_versions(dataset_version_id),
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        fragment_id TEXT NOT NULL REFERENCES data_fragments(fragment_id),
        PRIMARY KEY (dataset_version_id, ordinal),
        UNIQUE (dataset_version_id, fragment_id)
    ) STRICT""",
    "CREATE INDEX data_version_fragments_fragment ON data_version_fragments(fragment_id)",
    *_immutable_triggers("data_version_fragments"),
)


# --------------------------------------------------------------------------
# data_snapshots — SnapshotRef (Handle), phase-2 guide §5.2
# --------------------------------------------------------------------------

_SNAPSHOTS = (
    f"""CREATE TABLE data_snapshots (
        snapshot_id TEXT PRIMARY KEY,
        parent_snapshot_id TEXT REFERENCES data_snapshots(snapshot_id),
        manifest_hash TEXT NOT NULL UNIQUE,
        calendar_version TEXT NOT NULL,
        source_priority_version TEXT NOT NULL,
        finality_receipt_refs_json TEXT NOT NULL,
        knowledge_mode_by_table_json TEXT NOT NULL,
        commit_receipt_ref TEXT NOT NULL,
        registered_at TEXT NOT NULL,
        CHECK ({_hash_check('manifest_hash')}),
        CHECK ({_json_check('finality_receipt_refs_json', 'array')}),
        CHECK ({_json_check('knowledge_mode_by_table_json')})
    ) STRICT""",
    # Invariant 8's vocabulary, applied per map value: a plain CHECK cannot
    # iterate a JSON object's values, so this is a BEFORE INSERT trigger over
    # json_each instead (see module docstring point 8).
    f"""CREATE TRIGGER data_snapshots_knowledge_mode_valid
    BEFORE INSERT ON data_snapshots
    WHEN EXISTS (
        SELECT 1 FROM json_each(NEW.knowledge_mode_by_table_json)
        WHERE json_each.value NOT IN {_KNOWLEDGE_MODES!r}
    )
    BEGIN
        SELECT RAISE(ABORT, 'data_snapshots.knowledge_mode_by_table_json has an unknown knowledge mode');
    END""",
    *_immutable_triggers("data_snapshots"),
)


# --------------------------------------------------------------------------
# data_snapshot_tables — one dataset version per named table, guide §5.2/§6
# --------------------------------------------------------------------------

_SNAPSHOT_TABLES = (
    """CREATE TABLE data_snapshot_tables (
        snapshot_id TEXT NOT NULL REFERENCES data_snapshots(snapshot_id),
        table_name TEXT NOT NULL,
        dataset_version_id TEXT NOT NULL REFERENCES data_dataset_versions(dataset_version_id),
        PRIMARY KEY (snapshot_id, table_name)
    ) STRICT""",
    "CREATE INDEX data_snapshot_tables_dataset_version ON data_snapshot_tables(dataset_version_id)",
    # Invariant 7: the bound dataset version's contract table_name must equal
    # this row's own table_name.
    """CREATE TRIGGER data_snapshot_tables_contract_match
    BEFORE INSERT ON data_snapshot_tables
    WHEN NEW.table_name <> (
        SELECT c.table_name FROM data_dataset_versions v
        JOIN data_contracts c ON c.contract_id = v.contract_id
        WHERE v.dataset_version_id = NEW.dataset_version_id
    )
    BEGIN
        SELECT RAISE(ABORT, 'data_snapshot_tables.table_name does not match its dataset version contract');
    END""",
    *_immutable_triggers("data_snapshot_tables"),
)


# --------------------------------------------------------------------------
# data_snapshot_heads — the one mutable table, guide §6 invariant 2
# --------------------------------------------------------------------------

_SNAPSHOT_HEADS = (
    """CREATE TABLE data_snapshot_heads (
        scope TEXT PRIMARY KEY,
        snapshot_id TEXT NOT NULL REFERENCES data_snapshots(snapshot_id),
        generation INTEGER NOT NULL CHECK (generation > 0),
        updated_at TEXT NOT NULL,
        update_receipt_ref TEXT NOT NULL
    ) STRICT""",
    """CREATE TRIGGER data_snapshot_heads_insert_generation
    BEFORE INSERT ON data_snapshot_heads
    WHEN NEW.generation <> 1
    BEGIN
        SELECT RAISE(ABORT, 'data_snapshot_heads insert must start at generation 1');
    END""",
    # scope is the primary key, but SQLite still permits an UPDATE that
    # changes a row's own primary key value, so "scope unchanged" needs its
    # own guard rather than relying on the PK alone. A missing new snapshot
    # is already refused by the ordinary foreign key on snapshot_id.
    """CREATE TRIGGER data_snapshot_heads_update_generation
    BEFORE UPDATE ON data_snapshot_heads
    WHEN NEW.scope <> OLD.scope OR NEW.generation <> OLD.generation + 1
    BEGIN
        SELECT RAISE(ABORT, 'data_snapshot_heads update must keep scope and advance generation by exactly one');
    END""",
    """CREATE TRIGGER data_snapshot_heads_no_delete
    BEFORE DELETE ON data_snapshot_heads
    BEGIN
        SELECT RAISE(ABORT, 'data_snapshot_heads rows cannot be deleted');
    END""",
)


# --------------------------------------------------------------------------
# data_import_receipts — SnapshotImportReceipt, phase-2 guide §5.6/§7.3
# --------------------------------------------------------------------------

_IMPORT_RECEIPTS = (
    f"""CREATE TABLE data_import_receipts (
        receipt_id TEXT PRIMARY KEY,
        attempt_id TEXT NOT NULL,
        fence INTEGER NOT NULL CHECK (fence > 0),
        source_manifest_hash TEXT NOT NULL,
        result_snapshot_id TEXT REFERENCES data_snapshots(snapshot_id),
        status TEXT NOT NULL CHECK (status IN {_RECEIPT_STATUSES!r}),
        problem_json TEXT,
        registered_at TEXT NOT NULL,
        CHECK ({_hash_check('source_manifest_hash')}),
        CHECK ({_json_check('problem_json', nullable=True)}),
        CHECK (
            (status = 'committed' AND result_snapshot_id IS NOT NULL)
            OR (status IN ('failed', 'conflict') AND result_snapshot_id IS NULL)
        )
    ) STRICT""",
    "CREATE INDEX data_import_receipts_result_snapshot ON data_import_receipts(result_snapshot_id)",
    *_immutable_triggers("data_import_receipts"),
)


_V1 = (
    _CONTRACTS + _OBJECTS + _FRAGMENTS + _DATASET_VERSIONS + _VERSION_FRAGMENTS
    + _SNAPSHOTS + _SNAPSHOT_TABLES + _SNAPSHOT_HEADS + _IMPORT_RECEIPTS
)


# --------------------------------------------------------------------------
# v2 — P2-3: data_fragments.input_receipt_refs_json (never edit v1 above)
# --------------------------------------------------------------------------
#
# ``FragmentRecord.input_receipt_refs`` is part of a fragment's
# ``manifest_hash`` (``manifests.fragment_record``'s docstring) but v1's
# ``data_fragments`` has no column for it. Every other field
# ``engine/v2/data/repository.py::Repository.resolve`` needs to rebuild an
# exact ``FragmentRecord`` — object/contract refs (joins), partition_key,
# row_count, byte/logical hashes, key/time bounds, import_request_hash — is
# already a v1 column; this is the one addition P2-3 needs (task brief
# decision 3). ``DEFAULT '[]'`` lets the existing raw-SQL fixtures in
# ``tests/test_v2_data_catalog.py`` keep inserting rows that never mention
# this column; every row ``engine/v2/data/catalog.py::commit_snapshot``
# writes supplies its real value.
_V2 = (
    """ALTER TABLE data_fragments ADD COLUMN input_receipt_refs_json TEXT NOT NULL DEFAULT '[]'
    CHECK (json_valid(input_receipt_refs_json) AND json_type(input_receipt_refs_json) = 'array')""",
)

#: Plain ``(version, name, statements)`` tuples — never ``ops.migrations.Migration``
#: (module docstring). ``engine/v2/ops/bootstrap.py`` wraps these.
MIGRATIONS = ((1, "snapshot_catalog", _V1), (2, "fragment_input_receipt_refs", _V2))
