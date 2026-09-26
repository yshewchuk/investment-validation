"""The ``computed_moves`` ``TableContract`` (spec s4b Change 1).

Not built by ``legacy_mapping.py``: ``computed_moves`` has no legacy Tier-2
table backing it -- the legacy pull writes one ``moves_<TICKER>.json`` file
per ticker under ``engine.paths.COMPUTED_MOVES`` and the panel globs that
directory directly. There is no legacy-scanned table to map, so this module
builds its one contract directly, the same *mechanism*
``price_history_table.py`` uses (``manifests.table_contract_hash`` over a
placeholder, never a dataclass deriving its own hash) and none of
``legacy_mapping``'s legacy-annotation plumbing.

The pull this table replaces is ``engine.data.pulls.computed_moves``; the pure
row-building math lives in :mod:`engine.v2.data.computed_moves` and the
capture/store path in :mod:`engine.v2.ops.computed_moves_store`.

Layout: ``ticker, event_date, realized_move_pct, implied_move_pct,
quarter_ordinal, skipped, computed_at, source_hash, capture_id``, primary key
``(ticker, event_date)``, one fragment per ticker (``partition_columns =
("ticker",)``) holding that ticker's whole event history -- the same
whole-partition-rewrite reasoning ``price_history_table.py``'s docstring
gives: a later capture can correct an already-past event, which the catalog's
non-overlapping-fragment-range invariant cannot accept as an append.

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts`` and this package's own ``manifests`` -- never
``engine.v2.ops`` or legacy ``engine.*``.
"""
from __future__ import annotations

from engine.v2.contracts.data import ColumnContract, TableContract

from . import manifests

__all__ = ["COMPUTED_MOVES_CONTRACT", "COMPUTED_MOVES_TABLE_NAME"]

COMPUTED_MOVES_TABLE_NAME = "computed_moves"

_SCHEMA_EVOLUTION_POLICY = (
    "Never edit a registered definition under the same contract_id (phase-2 guide §5.1). "
    "Changed units, key meaning, time meaning, or null policy require a new major "
    "contract_id/semantic_version; nullable additions require a minor version only."
)

_COLUMNS = (
    ColumnContract(name="ticker", physical_type="string", nullable=False),
    ColumnContract(name="event_date", physical_type="string", nullable=False,
                   observation_time_semantics="the trading session the earnings event is "
                                              "for, YYYY-MM-DD, never a timestamp"),
    ColumnContract(name="realized_move_pct", physical_type="float64", nullable=True,
                   unit="percent",
                   null_policy="null on a skipped event (skipped=true); a skip is recorded "
                               "as a row, never silently dropped"),
    ColumnContract(name="implied_move_pct", physical_type="float64", nullable=True,
                   unit="percent",
                   null_policy="null when daily_market has no implied_move row strictly "
                               "before event_date"),
    ColumnContract(name="quarter_ordinal", physical_type="int64", nullable=False,
                   observation_time_semantics="ordinal of the event within its calendar "
                                              "year, as the legacy computed-moves pull labels it"
                                              " (0 on a skipped row -- the ordinal is meaningful "
                                              "only when skipped is False)"),
    ColumnContract(name="skipped", physical_type="bool", nullable=False,
                   null_policy="the event could not be bracketed: a missing close on either "
                               "side, or a P→Q window wider than MAX_GAP_CALENDAR_DAYS"),
    ColumnContract(name="computed_at", physical_type="string", nullable=False,
                   timezone="UTC",
                   observation_time_semantics="when this row's realized/implied move was "
                                              "computed -- the bitemporal 'as of' dimension"),
    ColumnContract(name="source_hash", physical_type="string", nullable=False,
                   null_policy="sha256 of the exact yfinance closes array used; never null"),
    ColumnContract(name="capture_id", physical_type="string", nullable=False,
                   null_policy="the immutable capture identity of the retrieval the row "
                               "came from; never null"),
)


def _build() -> TableContract:
    fields = dict(
        contract_id="computed_moves.v1",
        table_name=COMPUTED_MOVES_TABLE_NAME,
        semantic_version="1.0.0",
        columns=_COLUMNS,
        primary_key=("ticker", "event_date"),
        duplicate_policy="none_by_construction -- one row per (ticker, event_date)",
        foreign_keys=(),
        partition_columns=("ticker",),
        filterable_columns=("ticker",),
        orderable_columns=("ticker", "event_date"),
        observation_time_column="event_date",
        publication_time_column=None,
        receipt_time_column=None,
        finality_semantics="no finality window: a later capture may rewrite any event's "
                           "computed move. Whole-partition rewrite is the correction "
                           "mechanism (see the module docstring).",
        provenance_semantics="source_hash names the exact yfinance closes array a row was "
                             "computed from; capture_id keys the append-only "
                             "data_computed_moves_captures log.",
        coverage_semantics="a (ticker, event_date) pair is covered once a row exists for "
                           "it, skipped rows included; an event absent from a ticker's "
                           "fragment was never observed by a capture.",
        schema_evolution_policy=_SCHEMA_EVOLUTION_POLICY,
        maximum_batch_rows=65536,
        maximum_result_rows=500_000,
        legacy_mapping_ref=None,
    )
    placeholder = TableContract(definition_hash="sha256:" + "0" * 64, **fields)
    return TableContract(definition_hash=manifests.table_contract_hash(placeholder), **fields)


#: Built once at import time -- pure, deterministic, no I/O.
COMPUTED_MOVES_CONTRACT: TableContract = _build()
