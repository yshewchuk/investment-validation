"""Snapshot-to-legacy materialization planning — phase-2 guide §9.1, §9.2, D13/D14.

Pure logic only: this module never opens a file, a database connection or a
legacy ``engine.*`` symbol (Layer 1, ``system_rearchitecture.md`` §4.1). It
imports only ``engine.v2.contracts``, ``engine.v2.foundation``, and this
package's own ``repository`` (for its ``table_contract``/``explain_dependencies``
public reads). ``engine/v2/data/legacy_adapter.py`` — the package's one
legacy-importing module — imports *from* this module, never the other way,
so there is no import cycle between the two.

:data:`LEGACY_SCORE_READ_PLAN_V1` is this task's central judgement call (task
brief decision 1): which curated tables, in what scope, plus which model/
calendar files, a legacy score batch (``engine.features.FeatureContext.load``
+ ``engine.score.Scorer``/``score_calendar``) actually reads. It was derived
by reading those loaders end to end, not guessed:

* ``earnings_events`` and ``trades`` are read UNCONDITIONALLY WHOLE — every
  ``store.read_table("earnings_events", ...)``/``store.iter_table("trades",
  ...)`` call site in ``engine/features.py``/``engine/score.py`` omits
  ``years``/``tickers`` entirely. ``trades`` in particular is exactly "the
  broader historical analog population required by current behavior" the
  guide's §9.2 names: ``Scorer.__init__`` unconditionally enriches the whole
  table (``_enrich`` -> ``_entry_implied_move``) before any board is scored.
* ``daily_market`` is read at the evidence ticker/year scope
  (``FeatureContext.load(tickers, years=years)``'s own bound) for the
  feature-vector path. Its SECOND read site (``Scorer._entry_implied_move``,
  chunked by the ``trades`` table's own ticker/year span, unconditionally
  exercised at construction time) needs a still-broader ticker/year span that
  this module cannot derive without reading ``trades`` first — so, per this
  package's judgement call, the caller's ``evidence_scope`` is trusted to
  already be the union of both needs (the memory note this task was given:
  "the evidence scope must be the score job's FULL ticker set and year
  range"). A caller whose ``evidence_scope`` is only the board's own ticker/
  year window under-covers the second read site; this is recorded as a
  caveat in this task's report, not silently patched here.
* ``feature_panel``/``tier4_forecasts`` are always read whole
  (``engine.features.load_panel()``, ``engine.data.features.tier4.
  load_forecasts()`` both take no scope argument).
* ``securities``/``option_chains``/``option_daily`` are EXCLUDED: no call
  site in ``engine/features.py``/``engine/score.py`` reads them (confirmed by
  source audit — ``score_calendar`` prices structures from panel/Tier-4
  features, never a live chain lookup, and ``alt_strikes=0`` needs no strike
  ladder beyond ATM).
* Registry/model/calendar inputs are plain pinned files, never
  Repository-scanned tables: ``engine/models/registry.json``
  (``engine.models.registry.REGISTRY_PATH``, read unconditionally by
  ``Scorer.__init__``'s ``load_registry()``), ``engine/models/structures.json``
  (``engine.structure_registry.CHAMPIONS_PATH`` — optional in legacy code,
  since an absent file falls back to a documented default, but pinned here
  anyway so a private materialization reproduces which structure shape was
  actually live), each champion's joblib artifact
  (``RegistryEntry.artifact``/``.artifact_sha256``, under
  ``data/models/...``), and ``data/raw/polygon/gspc_daily.csv``
  (``engine.paths.GSPC_DAILY``, ``engine.calendar.trading_calendar``'s sole
  input). None of these are expressible as a bounded ``DataQuery`` — they are
  not Repository-tracked datasets — so they travel as pinned refs instead
  (this module's judgement call 2 below), never approximated as a query.

No STOP finding: every read this audit found is either a boundable
``DataQuery`` or a nameable pinned file. The one caveat is the
``evidence_scope`` completeness note above, which is a scope-input
responsibility, not an inexpressible read.
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from engine.v2.contracts.data import (
    DataQuery,
    KeyPredicate,
    LegacyMaterializationRequest,
    SnapshotRef,
    TableContract,
    TimeInterval,
)
from engine.v2.data.repository import Repository
from engine.v2.foundation import CONTENT_HASH_PREFIX, content_hash, to_document

from . import errors
from . import query as query_mod

__all__ = [
    "LEGACY_LAYOUT_VERSION",
    "LEGACY_SCORE_READ_PLAN_V1",
    "SCORE_READ_PLAN_TABLES",
    "TABLE_OUTPUT_KIND",
    "MaterializedTree",
    "assert_rows_match",
    "build_materialization_request",
    "format_pinned_ref",
    "lock_down",
    "materialize_tree",
    "narrow_query_to_year",
    "parse_pinned_ref",
    "read_plan_complete",
    "scanned_rows",
]

LEGACY_LAYOUT_VERSION = "legacy_curated_layout.v1"

#: Tables a legacy score batch actually reads, in the order they are written.
#: ``"whole_table"`` means the loader reads the table unconditionally, with no
#: ticker/year bound; ``"evidence_scoped"`` means it is bounded by the
#: request's ``evidence_scope`` tickers/years.
LEGACY_SCORE_READ_PLAN_V1: dict[str, object] = {
    "schema_version": "legacy_score_read_plan.v1",
    "tables": {
        "earnings_events": {
            "scope": "whole_table", "columns": "full", "output": "curated",
            "reason": "engine.features._session_index/_next_event, engine.score.score_calendar "
                      "all call store.read_table('earnings_events', ...) with no years/tickers bound",
        },
        "daily_market": {
            "scope": "evidence_scoped", "columns": "full", "output": "curated",
            "reason": "engine.features.FeatureContext.load(tickers, years=years) bounds its own read "
                      "to the caller's tickers/years; engine.score.Scorer._entry_implied_move's own "
                      "second read site needs the broader trades-ticker span — trusted to already be "
                      "inside evidence_scope (see module docstring caveat)",
        },
        "trades": {
            "scope": "whole_table", "columns": "full", "output": "curated",
            "reason": "engine.score._load_trades_without_legs streams store.iter_table('trades', ...) "
                      "with no years/tickers bound; this IS the broader historical analog population "
                      "the guide's §9.2 names, read unconditionally at Scorer.__init__",
        },
        "feature_panel": {
            "scope": "whole_table", "columns": "full", "output": "single_file",
            "reason": "engine.features.load_panel() takes no scope argument",
        },
        "tier4_forecasts": {
            "scope": "whole_table", "columns": "full", "output": "single_file",
            "reason": "engine.data.features.tier4.load_forecasts() takes no scope argument; read by "
                      "serving_model/_pool_before, unconditionally reachable from score()",
        },
    },
    "excluded_tables": ("securities", "option_chains", "option_daily"),
    "excluded_reason": "no call site in engine/features.py or engine/score.py reads them for a "
                        "score_calendar(alt_strikes=0) batch",
    "registry_and_model_refs": {
        "engine/models/registry.json": "engine.models.registry.REGISTRY_PATH, load_registry() default",
        "engine/models/structures.json": "engine.structure_registry.CHAMPIONS_PATH (optional in legacy "
                                          "code; pinned here for exact structure-champion parity)",
        "data/models/...": "each champion's joblib artifact named by RegistryEntry.artifact/"
                            ".artifact_sha256 inside registry.json",
    },
    "calendar_refs": {
        "data/raw/polygon/gspc_daily.csv": "engine.paths.GSPC_DAILY, engine.calendar.trading_calendar's "
                                            "sole input",
    },
}

SCORE_READ_PLAN_TABLES: tuple[str, ...] = tuple(LEGACY_SCORE_READ_PLAN_V1["tables"])
_EVIDENCE_SCOPED_TABLES = frozenset(
    name for name, spec in LEGACY_SCORE_READ_PLAN_V1["tables"].items() if spec["scope"] == "evidence_scoped")
#: table_name -> legacy output shape: "curated" (year-partitioned) or
#: "single_file" (one file, no year partitioning) — ``legacy_adapter.materialize``'s
#: own routing, kept here so it is derived from the plan rather than restated.
TABLE_OUTPUT_KIND: dict[str, str] = {
    name: spec["output"] for name, spec in LEGACY_SCORE_READ_PLAN_V1["tables"].items()}

#: A "no real restriction" time bound for a whole-table read (documents.py's
#: ``QUERY_NOT_BOUNDED`` check refuses a ``DataQuery`` with neither a
#: key_filter nor a time_interval — §8.2/§5.3 do not offer an "unbounded but
#: still legal" mode). This package's judgement call: a deliberately wide,
#: clearly-named sentinel interval over the table's own observation_time_column,
#: rather than inventing a third TimeInterval shape.
_WHOLE_TABLE_START = "1900-01-01"
_WHOLE_TABLE_END = "2999-12-31"

_PLACEHOLDER_HASH = CONTENT_HASH_PREFIX + "0" * 64
#: Judgement call 2 (task brief): a "pinned ref" for a plain file that is not
#: a Repository-tracked dataset (a registry/model/calendar file) is a single
#: string combining its legacy-relative destination path and its content
#: hash — ``LegacyMaterializationRequest.registry_and_model_refs``/
#: ``.calendar_refs`` are ``tuple[str, ...]`` in the contract, so this is the
#: one self-contained encoding that lets ``materialize`` resolve bytes from
#: the object store using only ``(repository, store, request, dest_root)``.
_REF_SEP = "::"


def format_pinned_ref(relative_path: str, ref_content_hash: str) -> str:
    """Build one ``registry_and_model_refs``/``calendar_refs`` entry."""
    if _REF_SEP in relative_path:
        raise ValueError(f"a legacy relative path may not contain {_REF_SEP!r}: {relative_path!r}")
    return f"{relative_path}{_REF_SEP}{ref_content_hash}"


def parse_pinned_ref(ref: str) -> tuple[str, str]:
    """``(relative_path, content_hash)`` from one pinned-ref string."""
    if _REF_SEP not in ref:
        raise ValueError(f"not a pinned ref (missing {_REF_SEP!r}): {ref!r}")
    path, _, ref_content_hash = ref.partition(_REF_SEP)
    if not path or not ref_content_hash:
        raise ValueError(f"pinned ref has an empty path or hash: {ref!r}")
    return path, ref_content_hash


def _scope_bounds(table_name: str, contract: TableContract,
                  evidence_scope: dict) -> tuple[tuple[KeyPredicate, ...], TimeInterval]:
    column = contract.observation_time_column
    if table_name not in _EVIDENCE_SCOPED_TABLES:
        return (), TimeInterval(column=column, start_inclusive=_WHOLE_TABLE_START,
                                end_exclusive=_WHOLE_TABLE_END)
    tickers = sorted(set(evidence_scope["tickers"]))
    years = sorted(int(y) for y in evidence_scope["years"])
    key_filter = (KeyPredicate(column="ticker", operator="in", values=tuple(tickers)),)
    interval = TimeInterval(column=column, start_inclusive=f"{years[0]}-01-01",
                            end_exclusive=f"{years[-1] + 1}-01-01")
    return key_filter, interval


def _build_table_query(repository: Repository, snapshot_ref: SnapshotRef, table_name: str,
                       evidence_scope: dict) -> DataQuery:
    contract = repository.table_contract(snapshot_ref, table_name)
    contract_ref = snapshot_ref.table_versions[table_name].table_contract_ref
    key_filter, time_interval = _scope_bounds(table_name, contract, evidence_scope)
    columns = tuple(c.name for c in contract.columns)
    probe = DataQuery(
        snapshot_id=snapshot_ref.snapshot_id, table_contract_ref=contract_ref, columns=columns,
        key_filter=key_filter, time_interval=time_interval, order_by=contract.primary_key,
        max_batch_rows=contract.maximum_batch_rows, max_result_rows=contract.maximum_result_rows)
    plan = repository.explain_dependencies(probe, table_name=table_name)
    # Decision 2: max_result_rows is the pinned row count this exact scope
    # touches, derived from the resolved snapshot's own fragment records
    # (DependencyEntry.estimated_rows == FragmentRecord.row_count) — finite
    # and manifest-derived, never the table's generic cap. max_batch_rows is
    # capped to the same bound (never > max_result_rows, per §5.3's own rule).
    row_bound = max(1, sum(entry.estimated_rows for entry in plan.dependencies))
    max_result_rows = min(row_bound, contract.maximum_result_rows)
    max_batch_rows = min(contract.maximum_batch_rows, max_result_rows)
    return dataclasses.replace(probe, max_batch_rows=max_batch_rows, max_result_rows=max_result_rows)


def build_materialization_request(
    repository: Repository, snapshot_ref: SnapshotRef, legacy_snapshot_object_ref,
    *, direct_scope: dict, evidence_scope: dict, registry_and_model_refs, calendar_refs,
    expected_population: dict,
) -> LegacyMaterializationRequest:
    """One ``DataQuery`` per :data:`SCORE_READ_PLAN_TABLES` entry, hashed as a
    Command (decision 2): ``request_hash`` covers everything except itself.

    ``repository`` is not in the brief's own abbreviated signature but is
    required here (recorded as a deviation in this task's report): computing
    a manifest-derived ``max_result_rows`` per table needs
    ``Repository.explain_dependencies``, which only a live repository can
    answer.
    """
    table_queries = {name: _build_table_query(repository, snapshot_ref, name, evidence_scope)
                     for name in SCORE_READ_PLAN_TABLES}
    placeholder = LegacyMaterializationRequest(
        request_hash=_PLACEHOLDER_HASH, snapshot_ref=snapshot_ref,
        legacy_snapshot_object_ref=legacy_snapshot_object_ref, direct_scope=dict(direct_scope),
        evidence_scope=dict(evidence_scope), table_queries=table_queries,
        registry_and_model_refs=tuple(registry_and_model_refs), calendar_refs=tuple(calendar_refs),
        legacy_layout_version=LEGACY_LAYOUT_VERSION, expected_population=dict(expected_population))
    doc = to_document(placeholder)
    del doc["request_hash"]
    return dataclasses.replace(placeholder, request_hash=content_hash(doc))


def _scope_superset(evidence_scope: dict, direct_scope: dict) -> bool:
    for key in ("tickers", "years"):
        if key not in direct_scope:
            continue
        if key not in evidence_scope:
            return False
        if not set(direct_scope[key]) <= set(evidence_scope[key]):
            return False
    return True


def read_plan_complete(request: LegacyMaterializationRequest) -> bool:
    """Decision 4: True only if every plan entry has a query or ref and the
    evidence scope is a superset of the direct scope (guide §9.2's refusal:
    "if the adapter cannot prove its read plan complete, it refuses checkpoint
    reuse")."""
    if set(request.table_queries) != set(SCORE_READ_PLAN_TABLES):
        return False
    if not request.registry_and_model_refs or not request.calendar_refs:
        return False
    return _scope_superset(request.evidence_scope, request.direct_scope)


# ==========================================================================
# materialize_tree — everything that touches no legacy symbol (§9.1)
# ==========================================================================
#
# ``legacy_adapter.materialize()`` is the only place a legacy reader is
# actually invoked (fan-out budget note in that module's docstring); every
# other piece of the write — dest_root safety, Parquet writing, content
# hashing, row comparison, lock-down — lives here instead, since none of it
# touches a legacy ``engine.*`` symbol.


@dataclasses.dataclass(frozen=True)
class MaterializedTree:
    """What :func:`materialize_tree` wrote, before validation or lock-down.

    ``curated_files``/``single_files`` name exactly the paths
    ``legacy_adapter.materialize`` must re-read with the legacy loaders;
    ``manifest`` is the full ``{relative_path: content_hash}`` this task's
    contract returns.
    """

    manifest: dict[str, str]
    curated_files: dict[str, dict[int, Path]]
    single_files: dict[str, Path]


def materialize_tree(repository: Repository, store, request: LegacyMaterializationRequest,
                     dest_root) -> MaterializedTree:
    """Write every declared path under ``dest_root``. Never chmods (decision
    3: lock-down is a separate, later step) and never opens a legacy
    ``engine.*`` symbol."""
    dest_root = Path(dest_root)
    _check_dest_root(store, dest_root)
    manifest: dict[str, str] = {}
    curated_files: dict[str, dict[int, Path]] = {}
    single_files: dict[str, Path] = {}
    for table_name, query in request.table_queries.items():
        contract = repository.table_contract(request.snapshot_ref, table_name)
        if TABLE_OUTPUT_KIND[table_name] == "curated":
            year_paths = _write_curated_table(repository, query, table_name, dest_root)
            curated_files[table_name] = year_paths
            for path in year_paths.values():
                manifest[str(path.relative_to(dest_root))] = _file_content_hash(path)
        else:
            rel = _single_file_relative_path(table_name)
            path = dest_root / rel
            _write_single_file(repository, query, table_name, contract, path)
            single_files[table_name] = path
            manifest[rel] = _file_content_hash(path)
    snapshot_rel = "data/features/SNAPSHOT"
    snapshot_path = dest_root / snapshot_rel
    _write_pinned_bytes(store, request.legacy_snapshot_object_ref.content_hash, snapshot_path)
    manifest[snapshot_rel] = _file_content_hash(snapshot_path)
    for ref in (*request.registry_and_model_refs, *request.calendar_refs):
        rel, ref_hash = parse_pinned_ref(ref)
        path = dest_root / rel
        _write_pinned_bytes(store, ref_hash, path)
        manifest[rel] = _file_content_hash(path)
    return MaterializedTree(manifest=manifest, curated_files=curated_files, single_files=single_files)


def _single_file_relative_path(table_name: str) -> str:
    name = "panel.parquet" if table_name == "feature_panel" else "tier4_forecasts.parquet"
    return f"data/features/{name}"


# -- dest_root safety --------------------------------------------------------


def _check_dest_root(store, dest_root: Path) -> None:
    if dest_root.is_symlink():
        raise errors.fail("DEST_ROOT_UNSAFE", "materialization dest_root must not be a symlink")
    store_root = Path(store.root).resolve()
    resolved = dest_root.resolve() if dest_root.exists() else dest_root.absolute()
    if resolved == store_root or store_root in resolved.parents:
        raise errors.fail("DEST_ROOT_UNSAFE",
                  "materialization dest_root must not be inside the object store")
    if not dest_root.exists():
        dest_root.mkdir(parents=True)
        return
    if not dest_root.is_dir():
        raise errors.fail("DEST_ROOT_UNSAFE", "materialization dest_root must be a directory")
    if any(dest_root.iterdir()):
        raise errors.fail("DEST_ROOT_NOT_EMPTY", "materialization dest_root must be empty")


# -- curated (year-partitioned) tables ---------------------------------------


def _write_curated_table(repository: Repository, query: DataQuery, table_name: str,
                         dest_root: Path) -> dict[int, Path]:
    curated_root = dest_root / "data" / "curated" / table_name
    writers: dict[int, pq.ParquetWriter] = {}
    paths: dict[int, Path] = {}
    for batch in repository.scan(query, table_name=table_name):
        _split_batch_by_year(batch, curated_root, writers, paths)
    for writer in writers.values():
        writer.close()
    return paths


def _split_batch_by_year(batch: pa.RecordBatch, curated_root: Path,
                         writers: dict[int, pq.ParquetWriter], paths: dict[int, Path]) -> None:
    years = batch.column(batch.schema.get_field_index("year")).to_pylist()
    for year in sorted(set(years)):
        sub = batch.filter(pa.array([y == year for y in years]))
        writer = writers.get(year)
        if writer is None:
            part_dir = curated_root / f"year={year}"
            part_dir.mkdir(parents=True, exist_ok=True)
            path = part_dir / "part-0000.parquet"
            writer = pq.ParquetWriter(path, batch.schema)
            writers[year] = writer
            paths[year] = path
        writer.write_batch(sub)


# -- single-file tables (feature_panel / tier4_forecasts) --------------------


def _write_single_file(repository: Repository, query: DataQuery, table_name: str,
                       contract: TableContract, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for batch in repository.scan(query, table_name=table_name):
            if writer is None:
                writer = pq.ParquetWriter(dest_path, batch.schema)
            writer.write_batch(batch)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        _write_empty_table(query, contract, dest_path)


def _write_empty_table(query: DataQuery, contract: TableContract, dest_path: Path) -> None:
    physical = {c.name: c.physical_type for c in contract.columns}
    schema = pa.schema([(name, query_mod.arrow_type_for(physical[name])) for name in query.columns])
    empty = pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)
    pq.write_table(empty, dest_path)


# -- SNAPSHOT bytes and pinned registry/model/calendar refs ------------------


def _write_pinned_bytes(store, ref_hash: str, dest_path: Path) -> None:
    """Resolve ``ref_hash`` against ``store``'s own content-addressed object
    pool (the same ``objects/<hash[:2]>/<hash>`` layout
    ``engine.v2.foundation.ArtifactStore`` commits to) and copy it to
    ``dest_path``. Never a symlink/hard link — always a fresh byte copy."""
    digest = ref_hash.removeprefix(CONTENT_HASH_PREFIX)
    object_path = Path(store.root) / "objects" / digest[:2] / digest
    try:
        data = object_path.read_bytes()
    except OSError as exc:
        raise errors.fail("OBJECT_CORRUPT", "a pinned ref names no object in the store") from exc
    if CONTENT_HASH_PREFIX + hashlib.sha256(data).hexdigest() != ref_hash:
        raise errors.fail("OBJECT_CORRUPT", "a pinned ref's bytes do not match its recorded hash")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(data)


# -- row comparison, hashing, lock-down --------------------------------------


def narrow_query_to_year(query: DataQuery, contract: TableContract, year: int) -> DataQuery:
    """``query`` restricted to one partition year — how validation re-scans
    exactly one already-written curated file without holding the whole table."""
    return dataclasses.replace(query, time_interval=TimeInterval(
        column=contract.observation_time_column, start_inclusive=f"{year}-01-01",
        end_exclusive=f"{year + 1}-01-01"))


def scanned_rows(repository: Repository, query: DataQuery, table_name: str) -> list[dict]:
    return [row for batch in repository.scan(query, table_name=table_name) for row in batch.to_pylist()]


def _normalize_value(value):
    if value is None:
        return None
    try:
        if value != value:  # NaN / NaT, the only values unequal to themselves
            return None
    except (TypeError, ValueError):
        pass
    to_pydatetime = getattr(value, "to_pydatetime", None)
    return to_pydatetime() if to_pydatetime is not None else value


def assert_rows_match(scanned: list[dict], legacy_frame, columns: tuple[str, ...]) -> None:
    """Row-for-row equality between a fresh Repository scan and a legacy
    ``_read_part`` frame of the same materialized file (D14: full-precision
    values, not an approximation)."""
    legacy_rows = legacy_frame.to_dict("records") if len(legacy_frame) else []
    if len(scanned) != len(legacy_rows):
        raise errors.fail("CONTRACT_MISMATCH",
                  f"materialized row count {len(legacy_rows)} != scanned row count {len(scanned)}")
    for scanned_row, legacy_row in zip(scanned, legacy_rows):
        for name in columns:
            if _normalize_value(scanned_row.get(name)) != _normalize_value(legacy_row.get(name)):
                raise errors.fail("CONTRACT_MISMATCH",
                          f"materialized value for column {name!r} disagrees with the scanned row")


def _file_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return CONTENT_HASH_PREFIX + digest.hexdigest()


def lock_down(dest_root) -> None:
    """chmod every file 0444 and every directory 0555 (decision 3), including
    ``dest_root`` itself."""
    dest_root = Path(dest_root)
    for path in dest_root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((p for p in dest_root.rglob("*") if p.is_dir()), reverse=True):
        path.chmod(0o555)
    dest_root.chmod(0o555)
