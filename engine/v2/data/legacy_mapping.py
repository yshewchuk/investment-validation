"""Legacy table/annotation -> v2 ``TableContract`` mapping.

Phase-2 guide §3.2 ("Existing implementation to preserve"), §4, §5.1 ("Do not
transcribe column definitions by hand in two places"), §12 D02.

Review round 3, item 1: split out of ``legacy_adapter.py`` so that module
holds only legacy-touching code (thin accessors and ``materialize()``'s
validation calls). This module builds every v2 object here — ``ColumnContract``,
``TableContract``, the mapping document — and touches legacy facts ONLY
through ``legacy_adapter``'s thin accessors (:func:`~engine.v2.data.
legacy_adapter.legacy_table_schemas`, ``.legacy_panel_columns``,
``.legacy_tier4_columns``, ``.legacy_tier4_key_columns``,
``.legacy_source_priority``) — never a legacy ``engine.*`` symbol directly.
``engine.v2.data.legacy_adapter`` stays the package's ONLY legacy-importing
module (phase-2 guide §4); calling it from here is an intra-package v2->v2
edge, never a new legacy one, so this module carries no
``checks/legacy_adapters.json`` entries of its own.

:func:`build_legacy_mapping` produces one ``legacy_table_mapping.v1.0``
document: one ``TableContract`` (as a strict document) per dataset in guide
§5.1's table, in order, a top-level ``knowledge_mode_by_table`` (every legacy
table is ``reconstructed`` — no accepted attestation or availability receipt
is referenced yet, phase-2 guide §3.3), and a separate
``legacy_snapshot_metadata`` entry describing the legacy ``SNAPSHOT``
compatibility object (``engine.paths.SNAPSHOT_FILE``), which is explicitly
**not** queryable and **not** a ``TableContract``.

Column *names* and, for the six Tier-2 tables, physical *dtype*/*nullable*
come only from ``legacy_adapter``'s accessors — never retyped by hand. Every
other fact a ``ColumnContract``/``TableContract`` needs (unit, scale,
sentinel and null policy, finality/provenance/coverage text, and — for
``feature_panel`` and ``tier4_forecasts``, which carry no legacy
``TableSchema`` — physical type and nullability too) comes from the
separately reviewed ``engine/v2/data/legacy_annotations.json``, sourced only
from ``engine.data.schemas.Column.doc``/``CONVENTIONS``/``SOURCE_PRIORITY``
and the ``engine.data.features`` docstrings. An unstated unit or time meaning
is recorded as the literal string ``"unknown"``, never guessed.

``PANEL_RELATIVE_PATH``/``TIER4_RELATIVE_PATH``/``SNAPSHOT_RELATIVE_PATH``
mirror ``engine/paths.py``'s ``PANEL``/``TIER4``/``SNAPSHOT_FILE`` (each
relative to ``engine.paths.DATA``) as reviewed literals rather than a legacy
import: they are exercised empirically by this package's private-schema test
against the real curated store, which would fail loudly if they ever
drifted. ``SOURCE_PRIORITY_VERSION`` is §7.1's
``SnapshotImportRequest.source_priority_version``: a deterministic
fingerprint of the reviewed source-priority text
``legacy_adapter.legacy_source_priority()`` documents, so a legacy
re-priority (say ORATS -> another vendor for spot) changes the version an
import request carries.

``build_legacy_mapping``, ``LegacyMappingError`` and the path/version
constants above used to live in ``legacy_adapter.py`` directly; every caller
was updated to import from here instead when the split landed (review round
3), since re-exporting them back through ``legacy_adapter.py`` would create
an import cycle (this module already imports ``legacy_adapter`` for its
accessors).

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts``, ``engine.v2.foundation``, and this package's own
``documents``/``manifests``/``legacy_adapter`` — never ``engine.v2.ops``.
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.v2.contracts.data import ColumnContract, TableContract
from engine.v2.foundation import content_hash, to_document

from . import documents, legacy_adapter, manifests

__all__ = [
    "ALLOWED_PHYSICAL_TYPES",
    "ANNOTATIONS_PATH",
    "DATASET_ORDER",
    "LEGACY_DTYPE_MAP",
    "LegacyMappingError",
    "PANEL_RELATIVE_PATH",
    "SNAPSHOT_RELATIVE_PATH",
    "SOURCE_PRIORITY_VERSION",
    "TIER2_DATASETS",
    "TIER4_RELATIVE_PATH",
    "build_legacy_mapping",
]

#: Legacy ``engine.data.schemas.Column.dtype`` -> contract ``physical_type``.
#: Closed and exhaustive over the five dtypes ``engine/data/schemas.py``
#: actually declares (its own ``_PANDAS_DTYPE`` table); an unmapped dtype is a
#: build-time failure, never a silent pass-through.
LEGACY_DTYPE_MAP: dict[str, str] = {
    "string": "string",
    "float64": "float64",
    "int64": "int64",
    "bool": "bool",
    "datetime64[ns]": "timestamp[ns]",
}

#: ``feature_panel``/``tier4_forecasts`` state their own ``physical_type``
#: directly in the annotations (no legacy dtype to map from); this is the
#: closed vocabulary those declarations must land in — the same targets
#: :data:`LEGACY_DTYPE_MAP` maps onto, plus ``timestamp[us]`` for the two
#: tables' microsecond-resolution pandas columns (confirmed against the real
#: curated store's Parquet schema by this package's private-schema test).
ALLOWED_PHYSICAL_TYPES = frozenset(LEGACY_DTYPE_MAP.values()) | {"timestamp[us]"}

#: Guide §5.1's table, in order. ``build_legacy_mapping`` never reorders it.
TIER2_DATASETS: tuple[str, ...] = (
    "securities", "earnings_events", "daily_market", "option_chains", "option_daily", "trades",
)
DATASET_ORDER: tuple[str, ...] = TIER2_DATASETS + ("feature_panel", "tier4_forecasts")

ANNOTATIONS_PATH = Path(__file__).resolve().parent / "legacy_annotations.json"

#: Mirror engine.paths.PANEL/TIER4/SNAPSHOT_FILE, each relative to
#: engine.paths.DATA — see the module docstring for why these are reviewed
#: literals rather than a legacy import.
PANEL_RELATIVE_PATH = "features/panel.parquet"
TIER4_RELATIVE_PATH = "features/tier4_forecasts.parquet"
SNAPSHOT_RELATIVE_PATH = "features/SNAPSHOT"

#: §7.1's ``SnapshotImportRequest.source_priority_version`` (task brief
#: decision 2): a deterministic fingerprint of the reviewed source-priority
#: text ``legacy_adapter.legacy_source_priority()`` documents, so a legacy
#: re-priority (say ORATS -> another vendor for spot) changes the version an
#: import request carries, without ``engine.v2.data.import_snapshot`` needing
#: its own legacy import (§4.2: one adapter module per package).
SOURCE_PRIORITY_VERSION = "legacy_source_priority:" + content_hash(legacy_adapter.legacy_source_priority())

MAPPING_SCHEMA_VERSION = "legacy_table_mapping.v1.0"
CONTRACT_SEMANTIC_VERSION = "1.0.0"
SCHEMA_EVOLUTION_POLICY = (
    "Never edit a registered definition under the same contract_id (phase-2 guide §5.1). Changed "
    "units, key meaning, time meaning, or null policy require a new major contract_id/semantic_version; "
    "nullable additions require a minor version only."
)


class LegacyMappingError(RuntimeError):
    """A legacy table/annotation mismatch this build refuses to paper over.

    ``code`` is stable across callers (tests match on it, not on message
    text); ``dataset`` names which of the eight datasets failed, so a failure
    never has to be traced back through the annotation file by hand.
    """

    def __init__(self, code: str, dataset: str, detail: str) -> None:
        super().__init__(f"{code} [{dataset}]: {detail}")
        self.code = code
        self.dataset = dataset
        self.detail = detail


def build_legacy_mapping(annotations: dict[str, object] | None = None) -> dict[str, object]:
    """The ``legacy_table_mapping.v1.0`` document for all eight datasets.

    ``annotations`` defaults to the reviewed ``legacy_annotations.json`` beside
    this module; a caller may pass a modified copy (tests do) to prove a
    missing/extra/unmapped annotation fails the build rather than silently
    passing through.
    """
    doc = annotations if annotations is not None else _load_annotations()
    tables_spec = doc["tables"]
    modes = doc["knowledge_mode_by_table"]
    if set(modes) != set(DATASET_ORDER):
        raise LegacyMappingError(
            "KNOWLEDGE_MODE_KEYS_MISMATCH", "*",
            f"knowledge_mode_by_table keys {sorted(modes)} != dataset set {sorted(DATASET_ORDER)}",
        )

    tables: dict[str, dict[str, object]] = {}
    for name in TIER2_DATASETS:
        tables[name] = _build_tier2_contract(name, _spec_for(tables_spec, name))
    tables["feature_panel"] = _build_feature_panel_contract(_spec_for(tables_spec, "feature_panel"))
    tables["tier4_forecasts"] = _build_tier4_contract(_spec_for(tables_spec, "tier4_forecasts"))
    return {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "tables": tables,
        "knowledge_mode_by_table": {name: modes[name] for name in DATASET_ORDER},
        "legacy_snapshot_metadata": _snapshot_metadata(doc["legacy_snapshot_metadata"]),
    }


def _load_annotations() -> dict[str, object]:
    return json.loads(ANNOTATIONS_PATH.read_text())


def _spec_for(tables_spec: dict[str, object], dataset: str) -> dict[str, object]:
    if dataset not in tables_spec:
        raise LegacyMappingError("MISSING_TABLE_ANNOTATION", dataset,
                                  "legacy_annotations.json has no entry for this dataset")
    return tables_spec[dataset]


# --------------------------------------------------------------------------
# column construction, shared by every dataset kind
# --------------------------------------------------------------------------


def _build_columns(
    dataset: str,
    source: list[tuple[str, str, bool]],
    ann_columns: dict[str, object],
    *,
    dtype_from_source: bool,
) -> tuple[ColumnContract, ...]:
    """Combine legacy ``(name, dtype, nullable)`` with reviewed per-column facts.

    ``dtype_from_source`` is True for the six Tier-2 tables (physical_type maps
    from the legacy dtype; nullable is the legacy declaration) and False for
    ``feature_panel``/``tier4_forecasts`` (both come from the annotation
    itself, since the legacy symbols name only columns there).
    """
    source_names = [name for name, _, _ in source]
    _check_column_coverage(dataset, source_names, ann_columns)
    columns = []
    for name, dtype, nullable in source:
        ann = ann_columns[name]
        if dtype_from_source:
            physical_type = _map_dtype(dataset, name, dtype)
        else:
            physical_type, nullable = _own_type(dataset, name, ann)
            _check_physical_type(dataset, name, physical_type)
        columns.append(_column_contract(name, physical_type, nullable, ann))
    return tuple(columns)


def _check_column_coverage(dataset: str, source_names: list[str], ann_columns: dict[str, object]) -> None:
    source_set = set(source_names)
    missing = sorted(n for n in source_names if n not in ann_columns)
    if missing:
        raise LegacyMappingError(
            "MISSING_ANNOTATION", dataset,
            f"column(s) {missing} have a legacy source but no reviewed annotation",
        )
    extra = sorted(set(ann_columns) - source_set)
    if extra:
        raise LegacyMappingError(
            "UNKNOWN_ANNOTATED_COLUMN", dataset,
            f"annotation names column(s) {extra} the legacy source does not have",
        )


def _own_type(dataset: str, name: str, ann: dict[str, object]) -> tuple[str, bool]:
    """``(physical_type, nullable)`` for a column whose legacy source declares neither.

    ``feature_panel``/``tier4_forecasts`` name columns only; this review's own
    annotation is the sole source, so a missing sub-field is the same
    ``MISSING_ANNOTATION`` failure as a wholly-absent column entry.
    """
    physical_type = ann.get("physical_type")
    nullable = ann.get("nullable")
    if physical_type is None or nullable is None:
        raise LegacyMappingError(
            "MISSING_ANNOTATION", dataset,
            f"column {name!r} annotation is missing physical_type and/or nullable",
        )
    return physical_type, nullable


def _column_contract(name: str, physical_type: str, nullable: bool, ann: dict[str, object]) -> ColumnContract:
    allowed_range = ann.get("allowed_range")
    return ColumnContract(
        name=name,
        physical_type=physical_type,
        nullable=nullable,
        unit=ann.get("unit"),
        scale=ann.get("scale"),
        adjustment_basis=ann.get("adjustment_basis"),
        timezone=ann.get("timezone"),
        null_policy=ann.get("null_policy"),
        sentinel_policy=ann.get("sentinel_policy"),
        allowed_range=tuple(allowed_range) if allowed_range is not None else None,
        observation_time_semantics=ann.get("observation_time_semantics"),
    )


def _map_dtype(dataset: str, name: str, dtype: str) -> str:
    physical = LEGACY_DTYPE_MAP.get(dtype)
    if physical is None:
        raise LegacyMappingError(
            "UNMAPPED_DTYPE", dataset,
            f"legacy dtype {dtype!r} for column {name!r} has no entry in LEGACY_DTYPE_MAP",
        )
    return physical


def _check_physical_type(dataset: str, name: str, physical_type: str) -> None:
    if physical_type not in ALLOWED_PHYSICAL_TYPES:
        raise LegacyMappingError(
            "UNMAPPED_DTYPE", dataset,
            f"declared physical_type {physical_type!r} for column {name!r} is not one of "
            f"{sorted(ALLOWED_PHYSICAL_TYPES)}",
        )


def _check_declared_subset(dataset: str, label: str, names: list[str], declared: set[str]) -> None:
    undeclared = sorted(set(names) - declared)
    if undeclared:
        raise LegacyMappingError(
            "UNDECLARED_COLUMN", dataset, f"{label} names undeclared column(s) {undeclared}",
        )


# --------------------------------------------------------------------------
# per-dataset-kind construction
# --------------------------------------------------------------------------


def _finalize(dataset: str, spec: dict[str, object], columns: tuple[ColumnContract, ...],
              partition_columns: tuple[str, ...], legacy_mapping_ref: str) -> dict[str, object]:
    declared = {c.name for c in columns}
    primary_key = tuple(spec["primary_key"])
    filterable = tuple(spec["filterable_columns"])
    orderable = tuple(spec["orderable_columns"])
    _check_declared_subset(dataset, "primary_key", list(primary_key), declared)
    _check_declared_subset(dataset, "filterable_columns", list(filterable), declared)
    _check_declared_subset(dataset, "orderable_columns", list(orderable), declared)

    fields = dict(
        contract_id=f"legacy.{dataset}.v1",
        table_name=dataset,
        semantic_version=CONTRACT_SEMANTIC_VERSION,
        columns=columns,
        primary_key=primary_key,
        duplicate_policy=spec["duplicate_policy"],
        foreign_keys=(),
        partition_columns=partition_columns,
        filterable_columns=filterable,
        orderable_columns=orderable,
        observation_time_column=spec["observation_time_column"],
        publication_time_column=spec["publication_time_column"],
        receipt_time_column=spec["receipt_time_column"],
        finality_semantics=spec["finality_semantics"],
        provenance_semantics=spec["provenance_semantics"],
        coverage_semantics=spec["coverage_semantics"],
        schema_evolution_policy=SCHEMA_EVOLUTION_POLICY,
        maximum_batch_rows=spec["maximum_batch_rows"],
        maximum_result_rows=spec["maximum_result_rows"],
        legacy_mapping_ref=legacy_mapping_ref,
    )
    # definition_hash is computed by this registration function, never by the
    # dataclass (phase-2 guide §5.1): build once with a placeholder to get a
    # hashable payload, then once more with the real hash — never mutated.
    placeholder = TableContract(definition_hash="sha256:" + "0" * 64, **fields)
    contract = TableContract(definition_hash=manifests.table_contract_hash(placeholder), **fields)
    doc = to_document(contract)
    documents.decode_document(TableContract, doc)  # validated as a document; returned as one (guide §5.1)
    return doc


def _build_tier2_contract(dataset: str, spec: dict[str, object]) -> dict[str, object]:
    schema = legacy_adapter.legacy_table_schemas()[dataset]
    source = [(c.name, c.dtype, c.nullable) for c in schema.columns]
    columns = _build_columns(dataset, source, spec["columns"], dtype_from_source=True)
    partition_columns = (schema.partition_by,) if schema.partition_by else ()
    ref = f"engine.data.schemas.{dataset.upper()}"
    return _finalize(dataset, spec, columns, partition_columns, ref)


def _build_feature_panel_contract(spec: dict[str, object]) -> dict[str, object]:
    source = [(name, "", False) for name in legacy_adapter.legacy_panel_columns()]
    columns = _build_columns("feature_panel", source, spec["columns"], dtype_from_source=False)
    # One declared logical partition (phase-2 guide §3.3, §5.1): partition_columns
    # is empty rather than a physical partitioning column. The partition-key
    # literal ("all") future manifest construction (P2-2/P2-3) will use is not a
    # TableContract field, so it is recorded only here and in the task report —
    # see phase-2 guide §3.3 "Logical partitions are stable".
    ref = f"engine.paths.PANEL({PANEL_RELATIVE_PATH});engine.data.features.panel.PANEL_COLUMNS"
    return _finalize("feature_panel", spec, columns, (), ref)


def _build_tier4_contract(spec: dict[str, object]) -> dict[str, object]:
    tier4_columns = legacy_adapter.legacy_tier4_columns()
    tier4_key_columns = legacy_adapter.legacy_tier4_key_columns()
    source = [(name, "", False) for name in tier4_columns]
    columns = _build_columns("tier4_forecasts", source, spec["columns"], dtype_from_source=False)
    if tuple(spec["primary_key"]) != tier4_key_columns:
        raise LegacyMappingError(
            "PRIMARY_KEY_MISMATCH", "tier4_forecasts",
            f"annotation primary_key {spec['primary_key']} != engine.data.features.tier4.KEY_COLUMNS "
            f"{tier4_key_columns}",
        )
    ref = (f"engine.paths.TIER4({TIER4_RELATIVE_PATH});"
           "engine.data.features.tier4.KEY_COLUMNS;engine.data.features.tier4.COLUMNS")
    return _finalize("tier4_forecasts", spec, columns, (), ref)


# --------------------------------------------------------------------------
# legacy snapshot compatibility metadata — not a TableContract, not queryable
# --------------------------------------------------------------------------


def _snapshot_metadata(spec: dict[str, object]) -> dict[str, object]:
    declared_path = spec["legacy_path"]
    if SNAPSHOT_RELATIVE_PATH != declared_path:
        raise LegacyMappingError(
            "SNAPSHOT_PATH_MISMATCH", "legacy_snapshot_metadata",
            f"SNAPSHOT_RELATIVE_PATH is {SNAPSHOT_RELATIVE_PATH!r}, annotation says {declared_path!r}",
        )
    if spec["queryable"] is not False:
        raise LegacyMappingError(
            "SNAPSHOT_MUST_NOT_BE_QUERYABLE", "legacy_snapshot_metadata",
            "the legacy SNAPSHOT compatibility object is never a queryable dataset (phase-2 guide §5.1)",
        )
    return {
        "queryable": False,
        "legacy_path": SNAPSHOT_RELATIVE_PATH,
        "expected_top_level_keys": tuple(spec["expected_top_level_keys"]),
        "description": spec["description"],
    }
