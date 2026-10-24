"""Pure diff/as-of logic for the ``price_history`` bitemporal dataset
(design confirmed 2026-09-14, superseding the original whole-download-object
design -- ``engine.v2.data.price_downloads``, which held that original
design's resolution/pin/replay primitives, was removed by the 2026-09-14
SEND-BACK once :func:`as_of_view` below was confirmed to cover the same
resolution rule at the row-version grain; its stability property test now
lives in ``tests/v2/data/test_price_downloads.py``).

``price_history`` is a normalized table of per-``(ticker, date)`` price
versions: ``ticker, date, close_adj, close_raw, high_raw, retrieved_at,
deleted, source_kind, source_hash, capture_id``. A capture never rewrites a
row; it appends a new version only for a date that is new or that changed
value. A retrieval omitting any currently live stored date refuses the whole
response instead (issue #719), so ``diff_retrieval`` never emits a tombstone
(``deleted=True``); tombstoned rows still exist from earlier captures and are
still honoured by the read side.
Everything in this module is pure pandas-in, pandas/dataclass-out -- no
filesystem, catalog or object-store I/O. ``engine.v2.ops.price_history_store``
is where captures actually get written, read back and materialized.

Two operations:

* :func:`diff_retrieval` -- one capture's delta against a ticker's current
  latest state (design doc §2's capture rules, including the
  full-history-only / no-partial-window refusal, extended by issue #719 to
  any omitted live date).
* :func:`as_of_view` -- the design doc §3 reconstruction for one ticker,
  decided once for the whole ticker, never per row/date. A string cutoff
  preserves retrieval-time replay; ``latest_snapshot=True`` selects the
  latest retrieval in the pinned snapshot. Built on :func:`resolve_pool`,
  the shared
  "pick one retrieval boundary, then reconstruct every row through it"
  implementation both ``as_of_view`` and
  ``engine.v2.data.price_history_query._provenance_by_date`` use. The price
  history reader uses the latest pinned retrieval and applies its decision
  session ceiling to bar dates after reconstruction.

**Vendor revision risk and leakage discipline for ``close_adj``.** A later
corporate action may retroactively rescale every close ON OR BEFORE it. The
price-history reader selects the latest retrieval in its pinned snapshot,
even when retrieval is after the observation ceiling, and filters by bar
date. This admits a session close fetched after midnight, but the data-date
filter cannot prevent a later vendor revision from changing an eligible
historical value; the reader accepts this residual split/dividend-adjustment
risk. Within one retrieval, a corporate action rescales its history
uniformly, not per-date. So ``close_adj`` is safe to use ONLY for a feature
that is scale-invariant within ONE retrieval's own view: a ratio of two
closes drawn from the SAME ``retrieved_at`` (``engine.data.features.panel.
add_runup_features``'s ``dist_high``/``dist_ema``/``ret5``/``ret10``/``ret20``,
panel.py:544-573, are exactly this -- each is `close[t] / f(close[<=t])`, so
the constant rescale factor cancels). Measured on real data (FDS, the ticker
the verified Tier-1 parsing defect flagged as a genuine dividend
restatement): the 08-27 px download and the 09-01 Tier-1 ``Adj Close`` differ
by up to 3.8e-3 relative, yet the resulting runup features differ by at most
1.4e-5 percentage points -- consistent with cancellation, not coincidence.
It is UNSAFE to read an absolute ``close_adj`` LEVEL (a raw price, a
dollar-denominated feature) or to STITCH ``close_adj`` values across
different retrievals as if they were one continuous series (:func:`resolve_pool`
already forward-fills across retrievals at the row-version grain by
construction -- see its own docstring -- but each row keeps ITS retrieval's
scale, so a feature spanning a version boundary mixes two different rescale
factors and no longer cancels): either leaks a later corporate action's
information into an earlier decision date. Under :data:`CLOSE_ADJ_RTOL` a
sub-tolerance ``close_adj`` difference between retrievals is accepted rather
than versioned, so an as-of view can mix values from retrievals that differ
by less than that tolerance. ``close_raw``/``high_raw``
(split-adjusted only, never dividend-adjusted, since 2026-09-15) do not have
this problem for splits handled the same way both sources apply them, but
still restate around actual stock splits, so the same "ratio within one
retrieval" discipline applies to them too.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from . import errors

__all__ = [
    "CLOSE_ADJ_RTOL",
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

#: Relative tolerance for ``close_adj`` equality (#1285): the vendor's
#: adjusted close jitters at ~1e-6 relative between downloads, and exact
#: equality turned that jitter into a full-history rewrite per ticker per
#: day. A genuine dividend or split restatement (>= ~1e-3 relative) still
#: exceeds this and versions every affected date.
CLOSE_ADJ_RTOL = 1e-5

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
    """Exact float64 equality, NaN equal to NaN (design doc §2): the rule for
    ``close_raw``/``high_raw`` and the first pass for ``close_adj`` before the
    :data:`CLOSE_ADJ_RTOL` relative tolerance in :func:`_values_equal`."""
    if a != a and b != b:  # both NaN: the only float that compares unequal to itself
        return True
    return a == b


def _values_equal(left, right) -> bool:
    """Whether a retrieval row matches a stored latest version's values.

    ``close_raw`` and ``high_raw`` compare exactly (:func:`_float_equal`, NaN
    equal to NaN). ``close_adj`` is equal when exactly equal OR both finite
    and within :data:`CLOSE_ADJ_RTOL` relative tolerance (#1285 vendor float
    jitter); one NaN against a finite value is NOT equal. A ``None`` (absent)
    ``close_adj`` is equal only to another ``None`` -- the tolerance branch is
    never reached with it, so no ``math.isfinite``/``None`` arithmetic can
    raise."""
    adj_left, adj_right = getattr(left, "close_adj"), getattr(right, "close_adj")
    if adj_left is None or adj_right is None:
        adj_close_equal = adj_left is None and adj_right is None
    else:
        adj_close_equal = _float_equal(adj_left, adj_right) or (
            math.isfinite(adj_left) and math.isfinite(adj_right)
            and abs(adj_left - adj_right) <= CLOSE_ADJ_RTOL * max(abs(adj_left), abs(adj_right)))
    return (adj_close_equal
            and _float_equal(getattr(left, "close_raw"), getattr(right, "close_raw"))
            and _float_equal(getattr(left, "high_raw"), getattr(right, "high_raw")))


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
    no-change retrieval. Never mutates or returns any existing row, and never
    emits a tombstone (``deleted=True``): it refuses instead (issue #719).
    A previously deleted row is still revived by a later retrieval that
    carries its date again (the ``prior.deleted`` branch below), so legacy
    tombstoned rows stay fixable. A date is "changed" on an exact
    ``close_raw``/``high_raw`` difference or a ``close_adj`` difference
    beyond the :data:`CLOSE_ADJ_RTOL` relative tolerance, so sub-tolerance
    vendor jitter on ``close_adj`` no longer rewrites history (#1285).

    ``stored_rows``: every version row known for this ticker so far (any
    order; only ``latest_state`` of it is used). ``retrieval``: this
    capture's freshly parsed full history (columns ``date, close_adj,
    close_raw, high_raw``; sorted or not).

    Refuses ``CONTRACT_MISMATCH`` -- the WHOLE response for this ticker is
    refused, nothing written -- when ``retrieval`` starts strictly later
    than the ticker's currently-live (non-tombstoned) stored history
    (design doc §2's "full history only" rule), or when it omits ANY
    currently live stored date (issue #719: a truncated end, a mid-history
    gap, or an empty retrieval against a non-empty live history). An
    omission is reported in ``details`` as ``missing_count`` plus either
    ``missing_dates`` (a sorted list, when there are at most 20) or
    ``first_missing_date``/``last_missing_date`` (past 20). A genuine
    period=max re-download only ever holds or extends its start date; a
    shorter window would silently drop real history and splice two
    adjustment bases together.
    """
    current = latest_state(stored_rows)
    live = current[~current["deleted"].astype(bool)] if len(current) else current
    retrieval_dates = set(retrieval["date"]) if len(retrieval) else set()
    if len(live):
        if retrieval_dates and min(retrieval["date"]) > live["date"].min():
            raise errors.fail(
                "CONTRACT_MISMATCH",
                "retrieval starts later than the ticker's stored history; a partial "
                "window cannot be captured without stitching adjustment bases",
                details={"stored_min_date": str(live["date"].min()),
                         "retrieval_min_date": str(min(retrieval["date"]))})
        missing = sorted({str(date_) for date_ in live["date"] if date_ not in retrieval_dates})
        if missing:
            details = {"missing_count": len(missing)}
            if len(missing) <= 20:
                details["missing_dates"] = missing
            else:
                details["first_missing_date"] = missing[0]
                details["last_missing_date"] = missing[-1]
            raise errors.fail(
                "CONTRACT_MISMATCH",
                "retrieval omits dates that are live in the ticker's stored history; "
                "the whole response is refused",
                details=details)
    by_date = {row.date: row for row in current.itertuples(index=False)}
    new_rows = []
    for row in retrieval.itertuples(index=False):
        prior = by_date.get(row.date)
        if prior is None or prior.deleted or not _values_equal(prior, row):
            new_rows.append({"date": row.date, "close_adj": row.close_adj,
                             "close_raw": row.close_raw, "high_raw": row.high_raw,
                             "retrieved_at": retrieved_at, "deleted": False,
                             "source_kind": source_kind, "source_hash": source_hash,
                             "capture_id": capture_id})
    return pd.DataFrame(new_rows, columns=list(_STORED_ROW_COLUMNS))


def _chosen_retrieved_at(stored_rows: pd.DataFrame, cutoff: str | None, *,
                         latest_snapshot: bool = False) -> str:
    """Choose latest pinned retrieval explicitly, or replay a required cutoff."""
    if latest_snapshot:
        if cutoff is not None:
            raise ValueError("cutoff must be omitted when latest_snapshot=True")
        return stored_rows["retrieved_at"].max()
    if cutoff is None:
        raise ValueError("cutoff is required unless latest_snapshot=True")
    eligible = stored_rows.loc[stored_rows["retrieved_at"] <= cutoff, "retrieved_at"]
    if len(eligible):
        return eligible.max()
    return stored_rows["retrieved_at"].min()


def resolve_pool(stored_rows: pd.DataFrame, cutoff: str | None = None, *,
                 latest_snapshot: bool = False) -> pd.DataFrame:
    """Design doc §3 as-of reconstruction: every stored column (tombstones,
    ``retrieved_at``, ``source_hash`` included) for the view ``cutoff``
    resolves to -- the ONE reconstruction both :func:`as_of_view` (drops
    tombstones and the extra columns) and
    ``price_history_query._provenance_by_date`` (keeps them, for per-date
    provenance) share.

    First, :func:`_chosen_retrieved_at` selects the latest retrieval in the
    pinned snapshot only when ``latest_snapshot=True`` and ``cutoff`` is
    omitted. Otherwise a string cutoff is required for retrieval-time as-of
    replay. Every stored row through the selected retrieval is included,
    latest version per date winning -- a forward-fill reconstruction across
    ALL sources. The price-history reader requests the latest pinned retrieval
    and applies its data-date ceiling after reconstruction. Never
    source_kind-filtered: rows are diff-only, so an unchanged value across
    two different-source retrievals is stored only once, under whichever
    retrieval first saw it; filtering by source_kind would silently drop it.
    """
    if stored_rows.empty:
        return stored_rows.reindex(columns=_STORED_ROW_COLUMNS)
    chosen = _chosen_retrieved_at(stored_rows, cutoff, latest_snapshot=latest_snapshot)
    eligible = stored_rows[stored_rows["retrieved_at"] <= chosen]
    idx = eligible.groupby("date")["retrieved_at"].idxmax()
    return eligible.loc[idx].reset_index(drop=True)


def as_of_view(stored_rows: pd.DataFrame, cutoff: str | None = None, *,
               latest_snapshot: bool = False) -> pd.DataFrame:
    """The reconstructed ``(date, close_adj, close_raw, high_raw)`` series
    for one ticker, tombstones dropped. A string ``cutoff`` replays a
    retrieval-time as-of view; ``latest_snapshot=True`` selects the
    latest retrieval in the pinned snapshot. See :func:`resolve_pool` for the
    shared reconstruction; this drops tombstones and the provenance columns.
    """
    pool = resolve_pool(stored_rows, cutoff, latest_snapshot=latest_snapshot)
    view = pool[~pool["deleted"].astype(bool)].sort_values("date").reset_index(drop=True)
    return view[["date", "close_adj", "close_raw", "high_raw"]]
