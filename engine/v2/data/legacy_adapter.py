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
``engine.v2.ops``. §4.3 fan-out: 6 distinct modules (``engine.data.schemas``,
``engine.data.features.panel``, ``engine.data.features.tier4``,
``engine.data.store``, ``shutil`` — review round 4, the ``materialize()``
cleanup-on-failure below — and ``"."``), plus ``engine.paths`` and
``engine.models.registry`` for the reference-input path accessors: 8, exactly
the budget.

P2-C02: :func:`legacy_serving_fold` and :func:`legacy_tier4_serving_header`
add two exact legacy symbols (``engine.data.features.tier4.serving_fold``,
``.read_serving_header``) but reach them through the ``engine.data.features.
tier4`` module already on the list above, so the fan-out count above is
unchanged — still 8, still the budget, not 9.
"""
from __future__ import annotations

import shutil

from engine.data.features.panel import PANEL_COLUMNS
from engine.data.features.tier4 import COLUMNS as TIER4_COLUMNS
from engine.data.features.tier4 import KEY_COLUMNS as TIER4_KEY_COLUMNS
from engine.data.features.tier4 import SERVING_DIR as TIER4_SERVING_DIR
from engine.data.features.tier4 import read_serving_header as _legacy_read_serving_header
from engine.data.features.tier4 import serving_fold as _legacy_serving_fold
from engine.data.schemas import SCHEMAS, SOURCE_PRIORITY
from engine.data.schemas import coerce as _legacy_coerce
from engine.data.store import _read_part as _legacy_read_part
from engine.models.registry import ARTIFACT_DIR, REGISTRY_PATH
from engine.paths import DATA, FEATURES, GSPC_DAILY, ROOT, SNAPSHOT_FILE

from . import errors, legacy_materialization

__all__ = [
    "coerce_legacy",
    "legacy_calendar_path",
    "legacy_chooser_pool_path",
    "legacy_data_dir",
    "legacy_models_dir",
    "legacy_pnl_sim_history_path",
    "legacy_recalibration_pairs_path",
    "legacy_registry_path",
    "legacy_serving_fold",
    "legacy_snapshot_path",
    "legacy_structures_path",
    "legacy_tier4_serving_dir",
    "legacy_tier4_serving_header",
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


def legacy_serving_fold(event_date, as_of):
    """``engine.data.features.tier4.serving_fold`` — the scorer's own fold rule.

    P2-C02: the launch-time Tier-4 coverage check reaches this instead of
    reimplementing the ``min(event_fold, decision_fold)`` rule, so a v2 fold
    can never drift from what ``Scorer._serving`` actually asks for.
    """
    return _legacy_serving_fold(event_date, as_of)


def legacy_tier4_serving_header(path, *, expected_sha256_hex: str, max_bytes: int):
    """``engine.data.features.tier4.read_serving_header`` — bounded, hash-verified.

    P2-C02: reads a pinned Tier-4 serving-cache file's plain-data header
    (never its estimator) for the launch-time coverage check, refusing to
    unpickle a file larger than ``max_bytes`` or one whose bytes do not hash
    to ``expected_sha256_hex``.
    """
    return _legacy_read_serving_header(path, expected_sha256_hex=expected_sha256_hex,
                                       max_bytes=max_bytes)


# --------------------------------------------------------------------------
# reference-input paths — each relative to engine.paths.ROOT, POSIX form
# --------------------------------------------------------------------------
#
# ``engine/v2/data/reference_inputs.py`` builds ``LEGACY_REFERENCE_INPUTS_V1``
# from these and nothing else. Four leaf names are not exported by any module
# this adapter can import within its fan-out budget of 8: ``structures.json``
# (``engine.structure_registry.CHAMPIONS_PATH``),
# ``chooser_analog_pool.parquet`` (``engine.score.CHOOSER_ANALOG_POOL``),
# ``pnl_sim_history.parquet`` (``engine.pnl_sim.HISTORY_PATH`` — task brief
# 2026-09-14, pinning the two derived model artifacts the legacy scorer reads)
# and ``recalibration_pairs.parquet`` (``engine.recalibrate.PAIRS_PATH``, same
# task). All four sit beside a constant that is imported here (or, for
# ``pnl_sim_history.parquet``, are typed verbatim — ``HISTORY_PATH`` is
# already the exact root-relative POSIX string, not a ``Path`` built from an
# imported root), and a tier-0 test in ``tests/test_v2_data_reference_inputs.py``
# pins each one to its real legacy constant.


def _root_relative(path) -> str:
    return path.relative_to(ROOT).as_posix()


def legacy_calendar_path() -> str:
    """``engine.paths.GSPC_DAILY`` — ``engine.calendar.trading_calendar``'s input."""
    return _root_relative(GSPC_DAILY)


def legacy_registry_path() -> str:
    """``engine.models.registry.REGISTRY_PATH`` — ``Scorer.__init__``'s ``load_registry()``."""
    return _root_relative(REGISTRY_PATH)


def legacy_structures_path() -> str:
    """``engine.structure_registry.CHAMPIONS_PATH``: ``structures.json`` next to the registry."""
    return _root_relative(REGISTRY_PATH.parent / "structures.json")


def legacy_models_dir() -> str:
    """``engine.models.registry.ARTIFACT_DIR`` — where champion joblib artifacts live."""
    return _root_relative(ARTIFACT_DIR)


def legacy_tier4_serving_dir() -> str:
    """``engine.data.features.tier4.SERVING_DIR`` — the Tier-4 serving-model cache."""
    return _root_relative(TIER4_SERVING_DIR)


def legacy_chooser_pool_path() -> str:
    """``engine.paths.FEATURES / engine.score.CHOOSER_ANALOG_POOL``."""
    return _root_relative(FEATURES / "chooser_analog_pool.parquet")


def legacy_snapshot_path() -> str:
    """``engine.paths.SNAPSHOT_FILE`` — the legacy Tier-3 SNAPSHOT JSON."""
    return _root_relative(SNAPSHOT_FILE)


def legacy_data_dir() -> str:
    """``engine.paths.DATA`` — the root of the curated store and feature files."""
    return _root_relative(DATA)


def legacy_pnl_sim_history_path() -> str:
    """``engine.pnl_sim.HISTORY_PATH`` — the gate's trailing-cutoff history.

    Model OUTPUT downstream of Tier 4, sized beside the panel rather than
    inside it (``engine.pnl_sim``'s own docstring note above ``HISTORY_PATH``).
    Typed verbatim rather than imported: ``engine.pnl_sim`` would be a ninth
    distinct module on this adapter's reviewed 8-edge fan-out budget (see the
    comment above). ``HISTORY_PATH`` is already the exact root-relative POSIX
    string (``"data/features/pnl_sim_history.parquet"``), so no ``Path``
    join is needed the way ``legacy_chooser_pool_path`` needs one.
    """
    return "data/features/pnl_sim_history.parquet"


def legacy_recalibration_pairs_path() -> str:
    """``engine.recalibrate.PAIRS_PATH`` — the win-rate recalibration pairs cache.

    Model OUTPUT downstream of Tier 4, same reasoning as
    :func:`legacy_pnl_sim_history_path`. Built from ``FEATURES`` (already
    imported here) rather than importing ``engine.recalibrate``, for the same
    fan-out-budget reason.
    """
    return _root_relative(FEATURES / "recalibration_pairs.parquet")


# --------------------------------------------------------------------------
# P2-6: materialize — the one legacy-touching validation step (§9.1, D13/D14)
# --------------------------------------------------------------------------


#: ``materialize_tree``'s own dest_root safety refusals (raised before this
#: function's try/except below has written a single byte, from inside
#: ``legacy_materialization._check_dest_root``) must never trigger the
#: cleanup ``except`` clause: for ``DEST_ROOT_NOT_EMPTY`` that directory is
#: the CALLER's own pre-existing content, and for ``DEST_ROOT_UNSAFE`` it may
#: be a symlink to something else entirely or a path inside the object store
#: itself — deleting through either would destroy data this function never
#: touched. Every other failure code is only reachable after
#: ``_check_dest_root`` has already proven ``dest_root`` a safe, empty
#: directory this call itself created or is about to fill, so cleanup there
#: is always safe.
_DEST_ROOT_SAFETY_CODES = frozenset({"DEST_ROOT_NOT_EMPTY", "DEST_ROOT_UNSAFE"})


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

    Any failure past the dest_root safety checks (corrupt object bytes,
    a stale Tier-4 cache ref, a legacy coerce()/row-count mismatch, an
    oversized query, ...) removes every byte this call itself wrote before
    re-raising — there is no terminal state between "nothing written" and
    "the full tree, locked down": a caller can never observe a partial,
    writable materialization and mistake it for a valid one.
    """
    try:
        tree = legacy_materialization.materialize_tree(repository, store, request, dest_root)
        for table_name, year_paths in tree.curated_files.items():
            if table_name in tree.copied_tables:
                _validate_copied_curated_table(year_paths, table_name)
            else:
                contract = repository.table_contract(request.snapshot_ref, table_name)
                _validate_curated_table(repository, request.table_queries[table_name], table_name,
                                        contract, year_paths)
        for table_name, path in tree.single_files.items():
            if table_name in tree.copied_tables:
                read_legacy_part(path, columns=None)  # proves the unchanged reader opens it; no coerce()
            else:                                     # feature_panel/tier4_forecasts (no legacy schema)
                _validate_single_file(repository, request.table_queries[table_name], table_name, path)
        px_tickers = legacy_materialization.px_series_tickers(request)
        if px_tickers and legacy_materialization.PRICE_HISTORY_TABLE_NAME in request.snapshot_ref.table_versions:
            # Older snapshots (imported before price_history joined the
            # catalog, task brief 2026-09-14) have no price_history table at
            # all -- their materialized tree simply carries no px files,
            # exactly as it did before this task, rather than refusing every
            # replay of pre-existing snapshot generations.
            legacy_materialization.materialize_price_series(
                repository, request.snapshot_ref, dest_root, tickers=px_tickers,
                observation_ceiling=request.observation_ceiling)
        legacy_materialization.lock_down(dest_root)
    except errors.DataError as exc:
        if exc.code not in _DEST_ROOT_SAFETY_CODES:
            shutil.rmtree(dest_root, ignore_errors=True)
        raise
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
