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


# --------------------------------------------------------------------------
# v3 — Task 1 review fix: data_import_receipts.scope (never edit v1/v2 above)
# --------------------------------------------------------------------------
#
# The receipt-id retry short-circuit (``catalog.py::_existing_receipt``) must
# refuse a reused ``receipt_id`` whose stored scope differs from the call, but
# v1/v2 never recorded which scope an import receipt was written for.
# ``DEFAULT ''`` is an explicit sentinel for the two rows this column cannot
# meaningfully backfill: a v1/v2-era row (none exist in production yet) and
# ``catalog.py::record_failed_import``, which is not itself scoped.
_V3 = (
    """ALTER TABLE data_import_receipts ADD COLUMN scope TEXT NOT NULL DEFAULT ''""",
)


# --------------------------------------------------------------------------
# v4 — task 7a: data_dataset_versions.partition_logical_hashes_json (never edit v1-v3 above)
# --------------------------------------------------------------------------
#
# A logical partition may now hold several ordered, non-overlapping fragments
# (``engine/v2/data/manifests.py``'s task 7a docstring note): ``DatasetManifest
# .partition_logical_hashes`` maps a multi-fragment partition_key to its
# streamed ``logical_rows.v1`` hash — required for such a partition, implicit
# (defaults to the lone fragment's own hash) for every other one. ``DEFAULT
# '{}'`` matches the dataclass's own nullable-style default, the same way
# v2's ``DEFAULT '[]'`` let existing fixtures keep inserting rows that never
# mention the column, and every pre-task-7a commit decodes identically to "no
# multi-fragment partitions".
_V4 = (
    """ALTER TABLE data_dataset_versions ADD COLUMN partition_logical_hashes_json TEXT NOT NULL
    DEFAULT '{}' CHECK (json_valid(partition_logical_hashes_json)
    AND json_type(partition_logical_hashes_json) = 'object')""",
)


# --------------------------------------------------------------------------
# v5 — reference inputs per import receipt (never edit v1-v4 above)
# --------------------------------------------------------------------------
#
# Guide §14: the legacy SNAPSHOT, the model registry, champion artifacts,
# Tier-4 serving caches, the chooser analog pool and the calendar CSV are
# separately pinned compatibility inputs, not snapshot identity. One row per
# pinned file, keyed by the import receipt that pinned it
# (``engine/v2/data/reference_catalog.py``), inserted in the same transaction
# as that receipt. A row may name only a committed receipt, and is immutable.
_V5 = (
    f"""CREATE TABLE data_import_reference_inputs (
        receipt_id TEXT NOT NULL REFERENCES data_import_receipts(receipt_id),
        legacy_path TEXT NOT NULL CHECK (length(legacy_path) > 0),
        kind TEXT NOT NULL CHECK (length(kind) > 0),
        object_id TEXT NOT NULL CHECK (length(object_id) > 0),
        content_hash TEXT NOT NULL,
        byte_size INTEGER NOT NULL CHECK (byte_size >= 0),
        CHECK ({_hash_check('content_hash')}),
        PRIMARY KEY (receipt_id, legacy_path)
    ) STRICT""",
    """CREATE TRIGGER data_import_reference_inputs_committed_receipt
    BEFORE INSERT ON data_import_reference_inputs
    WHEN (SELECT status FROM data_import_receipts WHERE receipt_id = NEW.receipt_id)
         IS NOT 'committed'
    BEGIN
        SELECT RAISE(ABORT, 'data_import_reference_inputs requires a committed import receipt');
    END""",
    *_immutable_triggers("data_import_reference_inputs"),
)


# --------------------------------------------------------------------------
# v6 — data_import_reference_inputs.fold (never edit v1-v5 above)
# --------------------------------------------------------------------------
#
# Task brief 2026-09-14: two of the pinned reference inputs
# (``pnl_sim_history``, ``recalibration_pairs`` — model OUTPUT downstream of
# Tier 4) are pinned per Tier-4 monthly fold, not per snapshot identity. Every
# other kind's fold is meaningless, so ``DEFAULT ''`` (matching v2/v3/v4's own
# convention for a column most existing rows do not use) is the explicit
# sentinel for "not fold-bound", and the check accepts only that sentinel or a
# 6-digit ``YYYYMM``.
_V6 = (
    """ALTER TABLE data_import_reference_inputs ADD COLUMN fold TEXT NOT NULL DEFAULT ''
    CHECK (fold = '' OR (length(fold) = 6 AND fold GLOB '[0-9][0-9][0-9][0-9][0-9][0-9]'))""",
)


# --------------------------------------------------------------------------
# v7 — price_history capture log (never edit v1-v6 above)
# --------------------------------------------------------------------------
#
# SEND-BACK 2026-09-14 ("it isn't a Tier-2 table... no separate ops-root
# SQLite registry"): ``engine.v2.ops.price_history_store`` used to keep its
# own per-attempt capture log in a private ``registry.sqlite3`` outside the
# shared catalog. That log's PURPOSE — one immutable row per per-ticker
# capture ATTEMPT, successful or refused, so a later capture can tell "have I
# already observed this exact source_hash" and "what is the latest
# retrieved_at I've ever seen for this ticker" without re-deriving either from
# the dataset's own rows (a no-change retrieval adds no row there, so it
# alone cannot tell a refused re-download apart from one that was never
# tried) — is unchanged by moving here; only its storage is. One row per
# capture attempt, keyed by the ``data_import_receipts`` row of the commit
# it happened inside of (``price_history_store.capture`` mints exactly one
# price_history-advancing receipt per call, via ``record_references``,
# mirroring v5's own pattern), immutable, and requiring a committed receipt
# the same way v5 does.
_PRICE_CAPTURE_OUTCOMES = ("added", "no_change", "duplicate_source_hash", "refused_backdate",
                          "refused_partial", "error")

_V7 = (
    f"""CREATE TABLE data_price_captures (
        capture_id TEXT PRIMARY KEY,
        receipt_id TEXT NOT NULL REFERENCES data_import_receipts(receipt_id),
        ticker TEXT NOT NULL CHECK (length(ticker) > 0),
        source_kind TEXT NOT NULL,
        source_hash TEXT NOT NULL,
        retrieved_at TEXT NOT NULL,
        outcome TEXT NOT NULL CHECK (outcome IN {_PRICE_CAPTURE_OUTCOMES!r}),
        rows_added INTEGER NOT NULL CHECK (rows_added >= 0),
        rows_tombstoned INTEGER NOT NULL CHECK (rows_tombstoned >= 0),
        created_at TEXT NOT NULL
    ) STRICT""",
    "CREATE INDEX data_price_captures_ticker ON data_price_captures(ticker, retrieved_at)",
    "CREATE INDEX data_price_captures_receipt ON data_price_captures(receipt_id)",
    """CREATE TRIGGER data_price_captures_committed_receipt
    BEFORE INSERT ON data_price_captures
    WHEN (SELECT status FROM data_import_receipts WHERE receipt_id = NEW.receipt_id)
         IS NOT 'committed'
    BEGIN
        SELECT RAISE(ABORT, 'data_price_captures requires a committed import receipt');
    END""",
    *_immutable_triggers("data_price_captures"),
)

# --------------------------------------------------------------------------
# v8 — receipt lineage: a capture generation's accepted legacy read-set is
# inherited from its base (never edit v1-v7 above)
# --------------------------------------------------------------------------
#
# The integration gap this closes: ``engine.v2.ops.price_history_store``
# commits a price_history-only capture generation under its OWN honest,
# non-scheduler ``attempt_id`` (v7's docstring, "Honest attempt identity"),
# which never carries an ``attempt_input_bindings`` row for
# ``legacy_manifest.json`` the way a real scheduler-issued import attempt
# does. A capture never changes the legacy input manifest — it only adds
# ``price_history`` and carries every other table's dataset version forward
# unchanged (``price_history_store``'s own "Cadence" docstring section) — so
# its accepted legacy read-set IS its base receipt's, by construction, not a
# fact this table re-derives or re-verifies.
#
# One row per capture receipt, naming the committed receipt (of the base
# head at capture time) it inherits from. Immutable, like every other
# catalog metadata table (invariant 1); unlike v5/v7 there is no "requires a
# committed receipt" BEFORE INSERT trigger on ``receipt_id`` itself, because
# the row is written inside that very receipt's own commit transaction
# (``price_history_store._commit_generation``'s ``record_references``
# callback, via ``engine.v2.ops.generation_binding.record_price_history_lineage``)
# — the receipt row that owns it always exists by the time this one is
# inserted, exactly as v5/v7's own committed-receipt check would already
# require, but there is nothing to check separately since both writes share
# one all-or-nothing transaction. ``base_receipt_id`` is a plain foreign key
# (not required to be 'committed' at the SQL layer): by construction the
# writer only ever passes an already-committed receipt
# (``reference_catalog.committed_receipt_for_snapshot``'s result), and a
# lineage row that names one no longer committed — impossible under normal
# operation, but not something a foreign key alone can rule out for all
# time — is a launch-time refusal in
# ``engine.v2.ops.generation_binding.accepted_generation_refs`` (its lineage
# walk re-queries each hop's own committed status), not a schema constraint.
# ``kind`` is a closed, single-value vocabulary today (only one thing ever
# produces a receipt with no manifest binding of its own); the ``CHECK``
# keeps it that way rather than silently accepting an unrelated future kind
# under the same walk semantics without a deliberate decision to extend it.
_V8 = (
    """CREATE TABLE data_receipt_lineage (
        receipt_id TEXT PRIMARY KEY REFERENCES data_import_receipts(receipt_id),
        base_receipt_id TEXT NOT NULL REFERENCES data_import_receipts(receipt_id),
        kind TEXT NOT NULL CHECK (kind = 'price_history_capture')
    ) STRICT""",
    "CREATE INDEX data_receipt_lineage_base ON data_receipt_lineage(base_receipt_id)",
    *_immutable_triggers("data_receipt_lineage"),
)

#: Plain ``(version, name, statements)`` tuples — never ``ops.migrations.Migration``
#: (module docstring). ``engine/v2/ops/bootstrap.py`` wraps these.
MIGRATIONS = ((1, "snapshot_catalog", _V1), (2, "fragment_input_receipt_refs", _V2),
             (3, "import_receipt_scope", _V3), (4, "dataset_version_partition_hashes", _V4),
             (5, "import_reference_inputs", _V5), (6, "import_reference_input_fold", _V6),
             (7, "price_captures", _V7), (8, "receipt_lineage", _V8))
