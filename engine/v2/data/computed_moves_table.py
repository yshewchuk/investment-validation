"""The ``computed_moves`` ``TableContract`` (spec s4b Change 1).

Not built by ``legacy_mapping.py``: ``computed_moves`` has no legacy Tier-2
table backing it -- the legacy pull writes one ``moves_<TICKER>.json`` file
per ticker under ``engine.paths.COMPUTED_MOVES`` and the panel globs that
directory directly. There is no legacy-scanned table to map, so this module
builds its two contracts (v4 and the additive v5) directly, the same *mechanism*
``price_history_table.py`` uses (``manifests.table_contract_hash`` over a
placeholder, never a dataclass deriving its own hash) and none of
``legacy_mapping``'s legacy-annotation plumbing.

The pull this table replaces is ``engine.data.pulls.computed_moves``; the pure
row-building math lives in :mod:`engine.v2.data.computed_moves` and the
capture/store path in :mod:`engine.v2.ops.computed_moves_store`.

``computed_moves.v5`` (issue #966) is the same table one major version on: its
columns are v4's plus ``not_computed_reason`` (see the module docstring in
:mod:`engine.v2.data.computed_moves` and ``ARCHITECTURE.md``). v4 stays pinned
and readable; the two contracts share every field but ``contract_id``,
``semantic_version`` and ``columns``.

Layout: ``ticker, event_date, realized_move_pct, available_as_of_date,
implied_move_pct, event_ordinal, admitted, mean_prior_or_implied,
n_prior_implied, skipped, computed_at, source_hash,
capture_id`` (v5: plus ``not_computed_reason``), primary key
``(ticker, event_date)``, one partition per ticker (``partition_columns =
("ticker",)``). New captures use one fragment per event; older snapshots may
hold a whole ticker in one fragment. Both layouts use the same
whole-partition-rewrite reasoning ``price_history_table.py``'s docstring
gives: a later capture can correct an already-past event, which the catalog's
non-overlapping-fragment-range invariant cannot accept as an append.

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts`` and this package's own ``manifests`` -- never
``engine.v2.ops`` or legacy ``engine.*``.
"""
from __future__ import annotations

import dataclasses

from engine.v2.contracts.data import ColumnContract, TableContract

from . import manifests

__all__ = [
    "COMPUTED_MOVES_CONTRACT",
    "COMPUTED_MOVES_TABLE_NAME",
    "COMPUTED_MOVES_V5_CONTRACT",
]

COMPUTED_MOVES_TABLE_NAME = "computed_moves"

_SCHEMA_EVOLUTION_POLICY = (
    "Never edit a registered definition under the same contract_id (phase-2 guide §5.1). "
    "Changed units, key meaning, time meaning, or null policy require a new major "
    "contract_id/semantic_version. A nullable addition needs only a minor version IF "
    "its contract_id has nothing committed under it yet; once anything has been "
    "committed, catalog.commit_snapshot's data_contracts table (keyed by contract_id "
    "alone) raises IDENTITY_CONFLICT on any second, differently-shaped definition "
    "under that id regardless of nullability, so a nullable addition to an "
    "already-committed contract_id needs a new major contract_id/semantic_version too "
    "(computed_moves.v4/4.0.0 is this table's own example)."
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
    ColumnContract(name="available_as_of_date", physical_type="string", nullable=True,
                   observation_time_semantics="the day FOLLOWING the close that allowed "
                                              "computing realized_move_pct -- the measured "
                                              "close date (session_move's sd[j_post]) plus "
                                              "one calendar day, YYYY-MM-DD, never a "
                                              "timestamp; the earliest decision_session at "
                                              "which this event's move was knowable",
                   null_policy="null exactly when realized_move_pct is null (a skipped event, "
                               "whatever the reason): null means not available to any decision "
                               "-- never treat a null available_as_of_date as always available"),
    ColumnContract(name="implied_move_pct", physical_type="float64", nullable=True,
                   unit="percent",
                   null_policy="implied_move on the last daily_market row strictly before "
                               "event_date whose src_iv is non-null, filled for skipped events "
                               "too; null when no such row precedes the event or that row's "
                               "implied_move is null or non-finite (never an older row)"),
    ColumnContract(name="event_ordinal", physical_type="int64", nullable=False,
                   observation_time_semantics="0-based count of every earlier event of the "
                                              "ticker in the builder's event list, realised "
                                              "or skipped"),
    ColumnContract(name="admitted", physical_type="bool", nullable=False,
                   null_policy="event_ordinal >= 4 (the legacy panel MIN_HISTORY warm-up)"),
    ColumnContract(name="mean_prior_or_implied", physical_type="float64", nullable=True,
                   unit="percent",
                   null_policy="mean of the stored implied_move_pct of earlier admitted "
                               "events, nulls skipped; null on a non-admitted row or when no "
                               "earlier admitted event has an implied value"),
    ColumnContract(name="n_prior_implied", physical_type="int64", nullable=False,
                   null_policy="count of values behind mean_prior_or_implied; 0 when it is null"),
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

#: The six v4 descriptors stating a v4 row rule v5 breaks (issue #966);
#: every other v4 descriptor stays byte-identical, so no v4 contract moves.
_V5_TEXT_OVERRIDES: dict[str, dict[str, str]] = {
    "implied_move_pct": dict(
        null_policy=(
            "implied_move on the last daily_market row strictly before "
            "event_date whose src_iv is non-null, filled for no_price_window "
            "rows; null when no such row precedes the event or that row's "
            "implied_move is null or non-finite (never an older row); always "
            "null on not_orats, no_session and ticker-level rows")),
    "event_ordinal": dict(
        observation_time_semantics=(
            "0-based count of earlier ORATS-confirmed sessioned events of the "
            "ticker; counts confirmed events only, so unconfirmed/no-outcome "
            "rows carry the count of earlier confirmed events")),
    "admitted": dict(
        null_policy=(
            "event_ordinal >= 4 and the row is a confirmed event on a ticker "
            "that clears the bars; false on every row whose "
            "not_computed_reason is not_orats, no_session or ticker-level")),
    "skipped": dict(
        null_policy=(
            "true exactly when not_computed_reason is non-null: no outcome "
            "was computed for this event")),
    "mean_prior_or_implied": dict(
        null_policy=(
            "mean of the stored implied_move_pct of earlier admitted events, "
            "nulls skipped; null on a non-admitted row, on a "
            "not_orats/no_session/ticker-level row, or when no earlier "
            "admitted event has an implied value")),
    "n_prior_implied": dict(
        null_policy=(
            "count of values behind mean_prior_or_implied; 0 when it is null, "
            "including every not_orats/no_session/ticker-level row")),
}

_NOT_COMPUTED_REASON_COLUMN = ColumnContract(
    name="not_computed_reason", physical_type="string", nullable=True,
    null_policy="null exactly when the row is a computed outcome (skipped false); non-null "
                "exactly when skipped is true and realized_move_pct and available_as_of_date "
                "are null. Closed reason set: not_orats (no ORATS confirmation, e.g. a "
                "forward_claim phantom), no_session (ORATS-confirmed but session null), "
                "no_price_window (the v4 skipped case: a missing close on either side or a "
                "P->Q window wider than MAX_GAP_CALENDAR_DAYS), no_price_history (the "
                "ticker's close series is empty), ticker_too_few_events (fewer than "
                "MIN_SCOREABLE ORATS-confirmed sessioned events or fewer than "
                "ADMISSION_EVENTS computable ones); the last two are ticker-level and apply "
                "to every row of that ticker.",
)

#: v4's columns with the contradicting descriptors rewritten, plus
#: ``not_computed_reason`` (issue #966); derived, never a copy-paste.
_V5_COLUMNS = tuple(
    dataclasses.replace(column, **_V5_TEXT_OVERRIDES[column.name])
    if column.name in _V5_TEXT_OVERRIDES else column
    for column in _COLUMNS
) + (_NOT_COMPUTED_REASON_COLUMN,)


def _fields(*, contract_id: str, semantic_version: str, columns) -> dict:
    """Every ``TableContract`` field but the identity and the column set."""
    return dict(
        contract_id=contract_id,
        table_name=COMPUTED_MOVES_TABLE_NAME,
        semantic_version=semantic_version,
        columns=columns,
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
        legacy_mapping_ref=None,
    )


def _build() -> TableContract:
    fields = _fields(contract_id="computed_moves.v4", semantic_version="4.0.0",
                     columns=_COLUMNS)
    placeholder = TableContract(definition_hash="sha256:" + "0" * 64, **fields)
    return TableContract(definition_hash=manifests.table_contract_hash(placeholder), **fields)


def _build_v5() -> TableContract:
    fields = _fields(contract_id="computed_moves.v5", semantic_version="5.0.0",
                     columns=_V5_COLUMNS)
    placeholder = TableContract(definition_hash="sha256:" + "0" * 64, **fields)
    return TableContract(definition_hash=manifests.table_contract_hash(placeholder), **fields)


#: Built once at import time -- pure, deterministic, no I/O.
COMPUTED_MOVES_CONTRACT: TableContract = _build()

#: The v5 contract (issue #966): v4's columns plus ``not_computed_reason``.
COMPUTED_MOVES_V5_CONTRACT: TableContract = _build_v5()
