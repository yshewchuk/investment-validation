"""Legacy ``engine.*`` symbols this package is allowed to touch.

Phase-2 guide §4: this is ``engine.v2.data``'s ONLY legacy-importing module.
Every legacy import lives here and nowhere else in the package; every name
below has a matching entry in ``checks/legacy_adapters.json``.

Review round 3, item 1: this module used to also build v2 objects
(``ColumnContract``/``TableContract``, the mapping document,
``build_legacy_mapping``) directly, which — combined with the legacy symbols
the exact-symbol ledger requires (review round 2's stop-and-report finding:
``[fan_out] engine/v2/data/legacy_adapter.py = 9, budget 8``) — spent this
module's whole §4.3 fan-out budget and one over. Fixed structurally, not with
an exemption or collapsed ledger entries: this module now builds NOTHING —
it only imports the legacy modules and exposes thin accessors/wrappers that
hand legacy values to v2 code (:func:`legacy_table_schemas`,
:func:`legacy_panel_columns`, :func:`legacy_tier4_columns`,
:func:`legacy_tier4_key_columns`, :func:`legacy_source_priority`,
:func:`read_legacy_part`, :func:`coerce_legacy`). Everything that builds a
v2 object — contracts, annotations handling, the mapping document,
materialization planning and writing — lives in
``engine/v2/data/legacy_mapping.py`` and ``.../legacy_materialization.py``,
both legacy-free, both calling this module's accessors instead of importing
legacy code directly. ``build_legacy_mapping`` moved to ``legacy_mapping.py``
with it; every caller and this package's ``README.md`` were updated in the
same commit rather than re-exported back through here, since that would
create an import cycle (``legacy_mapping.py`` already imports this module for
its accessors).

:func:`materialize` stays here: it is the one place D13/D14 actually invokes
a legacy reader (:func:`read_legacy_part`) and the legacy schema coercion
(:func:`coerce_legacy`) to prove a materialized file is readable by
unchanged legacy code, so it cannot move to a legacy-free module. Every
other piece of its machinery (dest_root safety, Parquet writing, hashing,
row comparison, lock-down) already lives in ``legacy_materialization.py``.

Layer 1 of ``system_rearchitecture.md`` §4.1: besides the declared legacy
edges, this module imports only its own package's ``errors`` and
``legacy_materialization`` (both reached through one relative-import edge)
— never ``engine.v2.contracts``, ``engine.v2.foundation`` or
``engine.v2.ops``. §4.3 fan-out: 5 distinct modules (``engine.data.schemas``,
``engine.data.features.panel``, ``engine.data.features.tier4``,
``engine.data.store``, ``"."``) — comfortably under the budget of 8 that
``build_legacy_mapping``'s old presence here used to exhaust.
"""
from __future__ import annotations

from engine.data.features.panel import PANEL_COLUMNS
from engine.data.features.tier4 import COLUMNS as TIER4_COLUMNS
from engine.data.features.tier4 import KEY_COLUMNS as TIER4_KEY_COLUMNS
from engine.data.schemas import SCHEMAS, SOURCE_PRIORITY
from engine.data.schemas import coerce as _legacy_coerce
from engine.data.store import _read_part as _legacy_read_part

from . import errors, legacy_materialization

__all__ = [
    "coerce_legacy",
    "legacy_panel_columns",
    "legacy_source_priority",
    "legacy_table_schemas",
    "legacy_tier4_columns",
    "legacy_tier4_key_columns",
    "materialize",
    "read_legacy_part",
]


# --------------------------------------------------------------------------
# thin accessors — the only way any other module reaches a legacy value
# --------------------------------------------------------------------------


def legacy_table_schemas() -> dict:
    """``engine.data.schemas.SCHEMAS`` — the six Tier-2 ``TableSchema`` objects."""
    return SCHEMAS


def legacy_panel_columns() -> tuple[str, ...]:
    """``engine.data.features.panel.PANEL_COLUMNS``, as an immutable tuple."""
    return tuple(PANEL_COLUMNS)


def legacy_tier4_columns() -> tuple[str, ...]:
    """``engine.data.features.tier4.COLUMNS``, as an immutable tuple."""
    return tuple(TIER4_COLUMNS)


def legacy_tier4_key_columns() -> tuple[str, ...]:
    """``engine.data.features.tier4.KEY_COLUMNS``, as an immutable tuple."""
    return tuple(TIER4_KEY_COLUMNS)


def legacy_source_priority():
    """``engine.data.schemas.SOURCE_PRIORITY`` — the reviewed source-priority text."""
    return SOURCE_PRIORITY


def read_legacy_part(path, columns):
    """``engine.data.store._read_part`` — the unchanged legacy single-path reader."""
    return _legacy_read_part(path, columns)


def coerce_legacy(frame, name: str):
    """``engine.data.schemas.coerce`` — the unchanged legacy schema coercion."""
    return _legacy_coerce(frame, name)


# --------------------------------------------------------------------------
# P2-6: materialize — the one legacy-touching validation step (§9.1, D13/D14)
# --------------------------------------------------------------------------


def materialize(repository, store, request, dest_root) -> dict[str, str]:
    """Write ``request``'s private legacy layout under ``dest_root``.

    ``dest_root`` must be a fresh, empty, non-symlink directory outside
    ``store``'s own tree — refused with a stable ``DEST_ROOT_*`` code
    otherwise. Every written file is re-read with the unchanged legacy
    readers before the tree is made read-only (chmod 0444 files / 0555
    dirs). Returns ``{relative_path: content_hash}``.

    A table in ``tree.copied_tables`` (review round 4, decision 1) was
    written by a verified byte-for-byte object copy, not a rewrite: its
    bytes are already proven correct by that copy's own hash check, so this
    only re-opens it with the unchanged legacy reader (never a fresh
    Repository scan to compare against — that would re-read the whole table
    a second time, exactly the cost decision 1 exists to avoid).
    """
    tree = legacy_materialization.materialize_tree(repository, store, request, dest_root)
    for table_name, year_paths in tree.curated_files.items():
        if table_name in tree.copied_tables:
            _validate_copied_curated_table(year_paths, table_name)
        else:
            contract = repository.table_contract(request.snapshot_ref, table_name)
            _validate_curated_table(repository, request.table_queries[table_name], table_name, contract,
                                    year_paths)
    for table_name, path in tree.single_files.items():
        if table_name in tree.copied_tables:
            read_legacy_part(path, columns=None)  # proves the unchanged reader opens it; no coerce()
        else:                                     # for feature_panel/tier4_forecasts (no legacy schema)
            _validate_single_file(repository, request.table_queries[table_name], table_name, path)
    legacy_materialization.lock_down(dest_root)
    return tree.manifest


def _validate_copied_curated_table(year_paths: dict, table_name: str) -> None:
    for paths_for_year in year_paths.values():
        for path in paths_for_year:
            _assert_legacy_coerce_accepts(read_legacy_part(path, columns=None), table_name)


def _validate_curated_table(repository, query, table_name: str, contract, year_paths: dict) -> None:
    for year, paths_for_year in year_paths.items():
        (path,) = paths_for_year  # the rewrite path always writes exactly one part-0000.parquet
        year_query = legacy_materialization.narrow_query_to_year(query, contract, year)
        scanned = legacy_materialization.scanned_rows(repository, year_query, table_name)
        legacy_frame = read_legacy_part(path, columns=None)
        legacy_materialization.assert_rows_match(scanned, legacy_frame, query.columns)
        _assert_legacy_coerce_accepts(legacy_frame, table_name)


def _validate_single_file(repository, query, table_name: str, path) -> None:
    scanned = legacy_materialization.scanned_rows(repository, query, table_name)
    legacy_frame = read_legacy_part(path, columns=None)
    legacy_materialization.assert_rows_match(scanned, legacy_frame, query.columns)


def _assert_legacy_coerce_accepts(legacy_frame, table_name: str) -> None:
    try:
        coerce_legacy(legacy_frame, table_name)
    except Exception as exc:
        raise errors.fail("CONTRACT_MISMATCH",
                  f"legacy coerce() refused the materialized {table_name!r} file") from exc
