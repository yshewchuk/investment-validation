"""Pure diff/as-of logic for the ``price_history`` bitemporal dataset
(design confirmed 2026-09-14, superseding the original whole-download-object
design -- ``engine.v2.data.price_downloads``, which held that original
design's resolution/pin/replay primitives, was removed by the 2026-09-14
SEND-BACK once :func:`as_of_view` below was confirmed to cover the same
resolution rule at the row-version grain; its stability property test now
lives in ``tests/test_v2_data_price_downloads.py``).

``price_history`` is a normalized table of per-``(ticker, date)`` price
versions: ``ticker, date, close_adj, close_raw, high_raw, retrieved_at,
deleted, source_kind, source_hash, capture_id``. A capture never rewrites a
row; it appends a new version only for a date that is new, that changed
value, or that dropped out of the retrieval (a tombstone, ``deleted=True``).
Everything in this module is pure pandas-in, pandas/dataclass-out -- no
filesystem, catalog or object-store I/O. ``engine.v2.ops.price_history_store``
is where captures actually get written, read back and materialized.

Two operations:

* :func:`diff_retrieval` -- one capture's delta against a ticker's current
  latest state (design doc §2's capture rules, including the
  full-history-only / no-partial-window refusal).
* :func:`as_of_view` -- the design doc §3 as-of reconstruction for one ticker
  at one cutoff, decided once for the whole ticker (never per row/date) so a
  cutoff before the ticker's first capture cannot accidentally splice a later
  capture's data in for one date and an earlier capture's for another. Built
  on :func:`resolve_pool`, the shared "pick one retrieval to view the ticker
  as of, then reconstruct every row up to it" implementation both
  ``as_of_view`` and ``engine.v2.data.price_history_query._provenance_by_date``
  use (SEND-BACK 2026-09-14 items 1 and 3 -- see :data:`_STORED_ROW_COLUMNS`'s
  own comment for the single read rule).
"""
from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from . import errors

__all__ = [
    "PRICE_HISTORY_VALUE_COLUMNS",
    "as_of_view",
    "check_not_backdated",
    "diff_retrieval",
    "latest_state",
    "resolve_pool",
]

#: The three float columns a version's identity is compared on ("any value
#: differs from that ticker/date's latest stored version").
PRICE_HISTORY_VALUE_COLUMNS = ("close_adj", "close_raw", "high_raw")

#: The one pinned read rule (user decision 2026-09-14, replacing an earlier
#: two-policy design -- "let's not have a special rule for parity runs, it's
#: potentially hiding issues... we need to run our real code"). Per ticker:
#: select the latest retrieval of ANY source with ``retrieved_at <= cutoff``
#: (else the ticker's globally earliest retrieval), then read the table as it
#: stood at that retrieval's ``retrieved_at`` -- every version with
#: ``retrieved_at`` at or before it, latest version per date, tombstones
#: dropped. Never filtered by ``source_kind``: rows are diff-only, so an
#: unchanged value across two different-source retrievals is stored only
#: once, under whichever retrieval first saw it; filtering by source_kind
#: would silently drop those shared rows and break after restatements.
#:
#: Known fact, not a special-cased behaviour: the legacy scorer read
#: ``px_<T>.csv`` whenever it existed and never a fresher Tier-1 body
#: (``panel.py:533``, falling back to Tier-1 only when px was missing --
#: ``panel.py:444-474``). Parity against the 2026-09-10 legacy run may
#: therefore differ on runup features for px tickers -- this module makes no
#: attempt to reproduce that px-first behaviour.
_STORED_ROW_COLUMNS = ("date", "close_adj", "close_raw", "high_raw", "retrieved_at", "deleted",
                       "source_kind", "source_hash", "capture_id")


def _float_equal(a: float, b: float) -> bool:
    """Exact float64 equality, NaN equal to NaN (design doc §2)."""
    if a != a and b != b:  # both NaN: the only float that compares unequal to itself
        return True
    return a == b


def _values_equal(left, right) -> bool:
    return all(_float_equal(getattr(left, column), getattr(right, column))
              for column in PRICE_HISTORY_VALUE_COLUMNS)


def check_not_backdated(stored_retrieved_ats: Sequence[str], candidate_retrieved_at: str) -> None:
    """Refuse typed (never floor-adjust) unless ``candidate_retrieved_at`` is
    later than every ``retrieved_at`` already recorded for this ticker --
    design doc §2: "retrieved_at must be later than every stored retrieved_at
    for that ticker, and no earlier than the last capture that observed the
    previous state." The second clause is implied by the first whenever
    ``stored_retrieved_ats`` includes every capture ATTEMPT for the ticker
    (not only the ones that produced a new row) -- the caller
    (``engine.v2.ops.price_history_store``) is responsible for passing the
    capture-log's full attempt history, not just this dataset's own stored
    rows, so a no-change retrieval still counts as having observed the
    current state.
    """
    if stored_retrieved_ats and candidate_retrieved_at <= max(stored_retrieved_ats):
        raise errors.fail("INPUT_CHANGED",
                          "capture retrieved_at is not later than an already-observed "
                          "retrieved_at for this ticker",
                          details={"candidate_retrieved_at": candidate_retrieved_at})


def latest_state(stored_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per date: the most recently retrieved version, tombstones
    included. Empty in, empty out (a ticker's first-ever capture).
    """
    if stored_rows.empty:
        return stored_rows.reindex(columns=_STORED_ROW_COLUMNS)
    idx = stored_rows.groupby("date")["retrieved_at"].idxmax()
    return stored_rows.loc[idx].reset_index(drop=True)


def diff_retrieval(stored_rows: pd.DataFrame, retrieval: pd.DataFrame, *, retrieved_at: str,
                   source_kind: str, source_hash: str, capture_id: str) -> pd.DataFrame:
    """The new version rows one capture contributes, or an empty frame for a
    no-change retrieval. Never mutates or returns any existing row.

    ``stored_rows``: every version row known for this ticker so far (any
    order; only ``latest_state`` of it is used). ``retrieval``: this
    capture's freshly parsed full history (columns ``date, close_adj,
    close_raw, high_raw``; sorted or not).

    Refuses ``CONTRACT_MISMATCH`` if ``retrieval`` starts strictly later than
    the ticker's currently-live (non-tombstoned) stored history -- design doc
    §2's "full history only" rule. A genuine period=max re-download only ever
    holds or extends its start date; a shorter window would tombstone real
    history as "missing from the retrieval" and silently splice two
    adjustment bases together.
    """
    current = latest_state(stored_rows)
    live = current[~current["deleted"].astype(bool)] if len(current) else current
    if len(live) and len(retrieval):
        stored_min = live["date"].min()
        retrieval_min = retrieval["date"].min()
        if retrieval_min > stored_min:
            raise errors.fail(
                "CONTRACT_MISMATCH",
                "retrieval starts later than the ticker's stored history; a partial "
                "window cannot be captured without stitching adjustment bases",
                details={"stored_min_date": str(stored_min), "retrieval_min_date": str(retrieval_min)})
    by_date = {row.date: row for row in current.itertuples(index=False)}
    seen = set()
    new_rows = []
    for row in retrieval.itertuples(index=False):
        seen.add(row.date)
        prior = by_date.get(row.date)
        if prior is None or prior.deleted or not _values_equal(prior, row):
            new_rows.append({"date": row.date, "close_adj": row.close_adj,
                             "close_raw": row.close_raw, "high_raw": row.high_raw,
                             "retrieved_at": retrieved_at, "deleted": False,
                             "source_kind": source_kind, "source_hash": source_hash,
                             "capture_id": capture_id})
    for date_, prior in by_date.items():
        if date_ not in seen and not prior.deleted:
            new_rows.append({"date": date_, "close_adj": float("nan"), "close_raw": float("nan"),
                             "high_raw": float("nan"), "retrieved_at": retrieved_at, "deleted": True,
                             "source_kind": source_kind, "source_hash": source_hash,
                             "capture_id": capture_id})
    return pd.DataFrame(new_rows, columns=list(_STORED_ROW_COLUMNS))


def _chosen_retrieved_at(stored_rows: pd.DataFrame, cutoff: str) -> str:
    """The single ``retrieved_at`` value to view the ticker as of, at
    ``cutoff``: the latest retrieval of ANY source at or before the cutoff,
    else the ticker's globally earliest retrieval (user decision 2026-09-14
    -- one rule, no source_kind filter, no policy choice)."""
    eligible = stored_rows.loc[stored_rows["retrieved_at"] <= cutoff, "retrieved_at"]
    if len(eligible):
        return eligible.max()
    return stored_rows["retrieved_at"].min()


def resolve_pool(stored_rows: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Design doc §3 as-of reconstruction: every stored column (tombstones,
    ``retrieved_at``, ``source_hash`` included) for the view ``cutoff``
    resolves to -- the ONE reconstruction both :func:`as_of_view` (drops
    tombstones and the extra columns) and
    ``price_history_query._provenance_by_date`` (keeps them, for per-date
    provenance) share.

    First, :func:`_chosen_retrieved_at` picks ONE ``retrieved_at`` value --
    the latest retrieval of any source at or before ``cutoff``, else the
    earliest. Then every row with ``retrieved_at`` at or before that chosen
    value is included, latest version per date winning -- a forward-fill
    reconstruction across ALL sources. Never source_kind-filtered: rows are
    diff-only, so an unchanged value across two different-source retrievals
    is stored only once, under whichever retrieval first saw it; filtering by
    source_kind would silently drop it.
    """
    if stored_rows.empty:
        return stored_rows.reindex(columns=_STORED_ROW_COLUMNS)
    chosen = _chosen_retrieved_at(stored_rows, cutoff)
    eligible = stored_rows[stored_rows["retrieved_at"] <= chosen]
    idx = eligible.groupby("date")["retrieved_at"].idxmax()
    return eligible.loc[idx].reset_index(drop=True)


def as_of_view(stored_rows: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """The reconstructed ``(date, close_adj, close_raw, high_raw)`` series
    for one ticker at cutoff ``cutoff`` (design doc §3), tombstones dropped.
    See :func:`resolve_pool` for the shared reconstruction; this just drops
    tombstones and the provenance columns.
    """
    pool = resolve_pool(stored_rows, cutoff)
    view = pool[~pool["deleted"].astype(bool)].sort_values("date").reset_index(drop=True)
    return view[["date", "close_adj", "close_raw", "high_raw"]]
