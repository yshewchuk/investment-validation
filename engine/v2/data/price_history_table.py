"""The ``price_history`` ``TableContract`` (design confirmed 2026-09-14
SEND-BACK: "the user explicitly asked for Tier 2").

**Bumped to ``price_history.v2`` (2026-09-15).** The first real shadow
capture found that ``close_raw``/``high_raw`` parsed all-NaN for every
Tier-1-sourced ticker and ``close_adj`` silently carried a split-adjusted
(not dividend-adjusted) value -- ``engine.v2.data.price_download_sources``'s
module docstring has the defect. Fixing the parser changes ``close_raw``'s/
``high_raw``'s ``null_policy``/``adjustment_basis`` (Tier-1 now supplies real
values, not a source-kind-conditioned NaN) -- exactly the "changed... null
policy... requires a new major contract_id" case the schema evolution policy
below names, so ``contract_id``/``semantic_version`` both bumped rather than
editing ``price_history.v1`` in place. See
``engine.v2.ops.price_history_store``'s module docstring, "Contract bump on
the first real capture's parsing defect", for how a capture run recaptures
cleanly under the new id (a fresh per-contract dataset-version chain, plus a
schema v9 ``contract_id`` scope on the no-op/anti-backdating capture log) and
why the OLD, ``price_history.v1``-pinned snapshot is never touched.

Not built by ``legacy_mapping.py``: that module's ``build_legacy_mapping``
exists to map a *legacy-scanned* table's own ``engine.data.schemas`` symbol
plus a reviewed ``legacy_annotations.json`` entry onto a ``TableContract``
(phase-2 guide §5.1). ``price_history`` has no legacy source of that shape --
it is captured natively by ``engine.v2.ops.price_history_store`` straight
into the v2 catalog, never scanned from a legacy Parquet tree -- so this
module builds its one contract directly, the same *mechanism*
(``manifests.table_contract_hash`` computes ``definition_hash`` over a
placeholder, never the dataclass deriving its own hash) but none of
``legacy_mapping``'s legacy-annotation plumbing.

Layout (design doc): ``ticker, date, close_adj, close_raw, high_raw,
retrieved_at, deleted, source_kind, source_hash, capture_id``, primary key
``(ticker, date, retrieved_at)``, one fragment per ticker (``partition_columns
= ("ticker",)``) holding that ticker's ENTIRE version history -- every date,
every ``retrieved_at``, tombstones included -- rewritten whole each time a
capture changes that ticker (never a byte-level append to an existing
fragment). See ``engine.v2.ops.price_history_store``'s module docstring for
why: the catalog's own non-overlapping-fragment-range invariant
(``manifests._check_membership_order``) requires a partition's later fragment
to sit entirely ABOVE the earlier one in primary-key order, which a
value-correction to an ALREADY-PAST date can never satisfy under any
``(ticker, date, retrieved_at)``-keyed column order. Whole-partition rewrite
sidesteps the invariant instead of relaxing it (relaxing it would touch
``manifests.py``, shared by every other Tier-2 table): the OLD fragment stays
immutable in the object store and in ``data_fragments`` forever (an earlier
dataset version, and any snapshot that pinned it, still resolves it exactly),
it is simply no longer referenced by the NEWEST dataset version's manifest.

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts`` and this package's own ``manifests`` -- never
``engine.v2.ops`` or legacy ``engine.*``.
"""
from __future__ import annotations

from engine.v2.contracts.data import ColumnContract, TableContract

from . import manifests

__all__ = ["PRICE_HISTORY_CONTRACT", "PRICE_HISTORY_TABLE_NAME"]

PRICE_HISTORY_TABLE_NAME = "price_history"

_SCHEMA_EVOLUTION_POLICY = (
    "Never edit a registered definition under the same contract_id (phase-2 guide §5.1). "
    "Changed units, key meaning, time meaning, or null policy require a new major "
    "contract_id/semantic_version; nullable additions require a minor version only."
)

_COLUMNS = (
    ColumnContract(name="ticker", physical_type="string", nullable=False),
    ColumnContract(name="date", physical_type="string", nullable=False,
                   observation_time_semantics="the trading session this price row is for, "
                                              "YYYY-MM-DD, never a timestamp"),
    ColumnContract(name="close_adj", physical_type="float64", nullable=True,
                   unit="price_per_share", adjustment_basis="legacy_px_csv/tier1_fetch as read",
                   null_policy="null on a tombstone row (deleted=true)"),
    ColumnContract(name="close_raw", physical_type="float64", nullable=True,
                   unit="price_per_share", adjustment_basis="unadjusted (px: Close; Tier-1: the "
                                                             "yfinance history CSV's own Close, "
                                                             "split-adjusted only)",
                   null_policy="null on a tombstone row only -- both sources carry this column "
                              "as of price_history.v2"),
    ColumnContract(name="high_raw", physical_type="float64", nullable=True,
                   unit="price_per_share", adjustment_basis="unadjusted (px: High; Tier-1: the "
                                                             "yfinance history CSV's own High)",
                   null_policy="null on a tombstone row only -- both sources carry this column "
                              "as of price_history.v2"),
    ColumnContract(name="retrieved_at", physical_type="string", nullable=False,
                   timezone="UTC", observation_time_semantics="when this row's value was captured "
                                                               "-- the bitemporal 'as of' dimension"),
    ColumnContract(name="deleted", physical_type="bool", nullable=False,
                   null_policy="a tombstone: this date dropped out of a later full-history "
                              "retrieval and is not part of the ticker's current history"),
    ColumnContract(name="source_kind", physical_type="string", nullable=False),
    ColumnContract(name="source_hash", physical_type="string", nullable=False),
    ColumnContract(name="capture_id", physical_type="string", nullable=False),
)


def _build() -> TableContract:
    fields = dict(
        contract_id="price_history.v2",
        table_name=PRICE_HISTORY_TABLE_NAME,
        semantic_version="2.0.0",
        columns=_COLUMNS,
        primary_key=("ticker", "date", "retrieved_at"),
        duplicate_policy="none_by_construction -- diff_retrieval never emits two rows for the "
                         "same (ticker, date, retrieved_at)",
        foreign_keys=(),
        partition_columns=("ticker",),
        filterable_columns=("ticker",),
        orderable_columns=("ticker", "date", "retrieved_at"),
        observation_time_column="retrieved_at",
        publication_time_column=None,
        receipt_time_column=None,
        finality_semantics="no finality window: a later capture may append a changed-value or "
                           "tombstone row for any prior date. An as-of view at a stated "
                           "retrieved_at cutoff (price_history.as_of_view) is the only finality "
                           "boundary a reader gets.",
        provenance_semantics="source_kind/source_hash/capture_id name the exact retrieval a row "
                             "came from; capture_id keys the append-only data_price_captures log "
                             "(one row per per-ticker capture attempt, successful or refused).",
        coverage_semantics="a (ticker, date) pair is covered once any non-tombstoned version of "
                           "it has ever been captured; a ticker absent from this table has never "
                           "been captured at all.",
        schema_evolution_policy=_SCHEMA_EVOLUTION_POLICY,
        maximum_batch_rows=65536,
        maximum_result_rows=500_000,
        legacy_mapping_ref=None,
    )
    placeholder = TableContract(definition_hash="sha256:" + "0" * 64, **fields)
    return TableContract(definition_hash=manifests.table_contract_hash(placeholder), **fields)


#: Built once at import time -- pure, deterministic, no I/O (matches
#: ``legacy_mapping``'s own registered contracts, which are likewise fixed
#: for the life of the process).
PRICE_HISTORY_CONTRACT: TableContract = _build()
