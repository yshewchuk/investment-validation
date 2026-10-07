"""``computed_moves`` capture -- the natively-owned store for the table
``engine/v2/data/computed_moves_table.py`` contracts (spec s4b Change 3).

This replaces the legacy-adapter wrapper the HEAD commit added: the job now
reads its two source tables out of the v2 catalog (``earnings_events`` and
``daily_market`` through :class:`~engine.v2.data.repository.Repository`),
fetches yfinance through the shared provider-budget admission the daily
refresh already uses (``incremental_data.plan_refresh`` /
``classify_response``), and stages the resulting rows into the object store
with the same immutable-object/manifest/atomic-head primitives
``price_history_store`` uses directly -- never ``generic_incremental``, because
``computed_moves`` has no manifest in the parent snapshot the way an existing
contracted table does. One fragment per ticker; the append-only capture log is
``data_computed_moves_captures`` (schema v12).

Spec s4c rewrites two things here: the yfinance edge is the injected
``yfinance_history_fetcher`` (never ``legacy_adapter.new_fetcher``), and the
source tables are never scanned per ticker (the old ``_events_for``/
``_daily_for`` made 2xN full-table scans for N targets). Issue #362: one
scan keeps only per-ticker counters for selection, and capture rescans one
bounded ticker chunk at a time (``_TickerChunks``), so retained source rows
never grow with total history.

The pure close-to-close math lives in :mod:`engine.v2.data.computed_moves`;
this module is the impure orchestrator.
"""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from engine.v2.contracts import ObjectRef, TableContractRef
from engine.v2.data import catalog as data_catalog
from engine.v2.data import manifests, objects, reference_catalog
from engine.v2.data.computed_moves import MIN_SCOREABLE, build_rows
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_TABLE_NAME
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.generation_binding import record_price_history_lineage
from engine.v2.ops.incremental_data import (
    RefreshCallbackResult,
    RefreshUnit,
    plan_refresh,
    refresh_job_kind,
)
from engine.v2.ops.legacy_adapter import iter_raw_fetch_cache
from engine.v2.ops.lifecycle import validated_attempt_fence_pair, verify_fence
from engine.v2.ops.pinned_partition_reader import (
    RetainedBatch,
    RetainedRowCount,
    iter_pinned_scan_batches,
)
from engine.v2.ops.unit_receipts import (
    NATIVE_COMPUTED_MOVES_ACCOUNT,
    cached_unit_outcomes,
    cached_unit_payloads,
    provider_failure_code,
    record_unit_receipt,
)

__all__ = [
    "computed_moves_units",
    "fetch_history",
    "run_computed_moves_refresh",
    "target_tickers_from_snapshot",
]

INPUT_PATH = "computed_moves_refresh_input.json"
MAX_SCAN_ROWS = 2_000_000
#: The reader's own batch ceiling; a chunk scan leases at most this, or half the guard if smaller.
_LEASE_ALLOWANCE = 50_000
_EVENT_COLUMNS = ("ticker", "event_date", "session", "src_orats")
_DAILY_COLUMNS = ("ticker", "date", "implied_move")
FRAGMENT_COLUMNS = ("ticker", "event_date", "realized_move_pct", "available_as_of_date",
                    "implied_move_pct", "quarter_ordinal", "skipped", "computed_at",
                    "source_hash", "capture_id")

_EMPTY_DAILY = pd.DataFrame(columns=["date", "implied_move"])

_ARROW_SCHEMA = pa.schema([
    ("ticker", pa.string()), ("event_date", pa.string()), ("realized_move_pct", pa.float64()),
    ("available_as_of_date", pa.string()),
    ("implied_move_pct", pa.float64()), ("quarter_ordinal", pa.int64()), ("skipped", pa.bool_()),
    ("computed_at", pa.string()), ("source_hash", pa.string()), ("capture_id", pa.string()),
])

_CONTRACT_REF = TableContractRef(contract_id=COMPUTED_MOVES_CONTRACT.contract_id,
                                 definition_hash=COMPUTED_MOVES_CONTRACT.definition_hash)

#: Every key the staged input document may carry; an unrecognized key is
#: refused rather than silently ignored (Opus review, PR #39).
_ALLOWED_DOCUMENT_KEYS = frozenset({
    "catalog_path", "objects_root", "scope", "expected_head_generation",
    "expected_head_snapshot_id", "parent_receipt_id", "attempt_id", "fence",
    "all_scoreable", "since", "as_of",
})

#: The namespaces this store's document may commit into. This store has no
#: ``JobKind`` of its own yet (see ARCHITECTURE.md "not yet wired"), so there
#: is no allowlist of its own to diverge from -- it reuses the sibling
#: ``incremental_refresh`` job kind's, the same ``{"shadow", "smoke"}`` every
#: other v2 ops entry point is scoped to.
_ALLOWED_SCOPES = refresh_job_kind().namespaces


def tier1_yfinance_history_fetcher(source_root):
    """Return the successful max-history cache index and a no-network fetcher."""
    newest = {}
    for entry in iter_raw_fetch_cache(source_root, "yfinance"):
        if entry.endpoint != "history" or entry.params.get("period") != "max":
            continue
        try:
            if int(entry.meta.get("status", 0)) != 200:
                continue
        except (TypeError, ValueError):
            continue
        ticker = entry.params.get("ticker")
        if not ticker:
            continue
        ticker = str(ticker)
        fetched_at = str(entry.meta.get("fetched_at") or "")
        ordering = (fetched_at, str(entry.path))
        if ticker not in newest or ordering > newest[ticker][0]:
            newest[ticker] = (ordering, entry)
    cache = {ticker: value[1] for ticker, value in newest.items()}

    def fetch(ticker):
        entry = cache.get(ticker)
        if entry is None:
            return b"", "legitimate_empty", {}, []
        return entry.body(), "complete", {}, []

    return cache, fetch


def parent_receipt_id_for_snapshot(conn, scope, snapshot_id) -> str:
    receipt_id = reference_catalog.committed_receipt_for_snapshot(
        conn, scope=scope, snapshot_id=snapshot_id)
    if receipt_id is None:
        raise fail("SNAPSHOT_NOT_READY", "scope's parent snapshot has no committed import receipt",
                   details={"scope": scope, "snapshot_id": snapshot_id})
    return _validated_parent_receipt_id(receipt_id)


def _as_of_day(as_of) -> str:
    """The job's as_of date as ``YYYY-MM-DD``.

    Spec R4/R5: the as_of is part of every unit id and the only date selection
    may use, so it is required, never defaulted to the wall clock, and every
    bad shape is refused (``INVALID_REQUEST``) rather than silently coerced:
    ``None``, a bool, a raw number (pandas reads an int/float as a UNIX
    timestamp, not a calendar date -- a defect class distinct from what it
    looks like), NaT, a tz-aware value (silently normalizing one would drop
    the timezone rather than refuse it), or a string pandas cannot parse.
    """
    if as_of is None or isinstance(as_of, bool) or isinstance(as_of, (int, float)):
        raise fail("INVALID_REQUEST",
                   "computed moves refresh as_of must be a date-like value, not a number")
    try:
        day = pd.Timestamp(as_of)
    except (ValueError, TypeError):
        raise fail("INVALID_REQUEST", "computed moves refresh as_of could not be parsed as a date")
    if pd.isna(day):
        raise fail("INVALID_REQUEST", "computed moves refresh needs an as_of date")
    if day.tzinfo is not None:
        raise fail("INVALID_REQUEST", "computed moves refresh as_of must be tz-naive")
    return str(day.normalize().date())


# --------------------------------------------------------------------------
# snapshot reads (the v2 read path, never engine.data.store)
# --------------------------------------------------------------------------


def _scan_rows(repository: Repository, snapshot, table_name: str,
               columns) -> Iterator[RetainedBatch]:
    """One pinned snapshot's ``table_name``, in the table contract's primary-key
    order, through the shared bounded pinned-partition reader.

    The reader keeps this store's exact query scope -- the snapshot's own
    contract ref, every represented year partition, these ``columns`` -- and
    yields the selection as the ordered :class:`RetainedBatch` leases accounted
    by a fresh local live-row account: each lease is the caller's to enter,
    process and exit before the reader advances, so the live count stays
    bounded by ``max_retained_rows`` (``MAX_SCAN_ROWS``, unchanged) and every
    batch is discharged as the reader advances, on scan error and on iterator
    close. Exhaustion simply ends the lease sequence once the selection's last
    batch has been released. Every repository exception (a missing or corrupt
    pinned input, a scan exceeding that bound) propagates as the repository's
    own typed refusal -- never an empty result.
    """
    account = RetainedRowCount()
    yield from iter_pinned_scan_batches(
        repository, snapshot, table_name, columns,
        max_retained_rows=MAX_SCAN_ROWS, retained_rows=account)


def _lease_cap() -> int:
    """The most rows one chunk-scan lease may hold: its share of ``MAX_SCAN_ROWS``."""
    return max(1, min(_LEASE_ALLOWANCE, MAX_SCAN_ROWS // 2))


def _scan_frame(repository: Repository, snapshot, table_name: str, columns,
                tickers=None, held: RetainedRowCount | None = None) -> pd.DataFrame:
    """One pinned snapshot's ``table_name`` as one frame, built per leased batch.

    Consumes the ordered :class:`RetainedBatch` leases from :func:`_scan_rows`.
    Handing released row dictionaries to ``pd.DataFrame`` after the fact would
    fail -- a lease clears its rows on release -- so exactly one frame chunk is
    constructed per batch while that batch's lease is live, the lease exits
    before the reader advances, and the completed chunks concatenate --
    ``ignore_index=True`` -- only after full exhaustion. A scan whose lease
    sequence yields no batches returns an empty frame, the previous generator's
    shape. Every repository or integrity error propagates unchanged as this
    helper unwinds, discarding the local provisional chunks; no row dictionary
    is ever retained or accumulated across batches.

    ``tickers`` (a set) keeps only those tickers' rows, so the frame is the
    selection's bounded slice rather than the whole table. That scan leases at
    most :func:`_lease_cap` rows and charges those leases and every kept row to
    ``held``; the caller discharges the kept rows when it drops the frame.
    """
    chunks: list[pd.DataFrame] = []
    leases = (_scan_rows(repository, snapshot, table_name, columns) if tickers is None
              else iter_pinned_scan_batches(
                  repository, snapshot, table_name, columns,
                  max_retained_rows=_lease_cap(), retained_rows=held))
    for lease in leases:
        with lease as batch:
            rows = batch if tickers is None else [row for row in batch if row["ticker"] in tickers]
            if tickers is not None and not rows:
                continue
            chunks.append(pd.DataFrame(rows))
            if held is not None:
                held.retain(len(rows))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _scan_once(repository: Repository, snapshot, tickers=None,
               held: RetainedRowCount | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One scan of each source table, optionally narrowed to ``tickers``.

    Returns the ORATS-confirmed ``earnings_events`` frame (``event_date``
    parsed) and the ``daily_market`` frame, both sorted by key. Each frame
    arrives through the batch-aware :func:`_scan_frame` consumer, so no scan
    retains the whole selection's row dictionaries at once. With ``tickers``
    omitted the frames are the whole tables; the refresh passes one bounded
    ticker chunk at a time (:class:`_TickerChunks`).
    """
    events = _scan_frame(repository, snapshot, "earnings_events", _EVENT_COLUMNS, tickers, held)
    if not events.empty:
        events = events[events["src_orats"] & events["session"].notna()].copy()
        events["event_date"] = pd.to_datetime(events["event_date"])
        events = events.sort_values("event_date").reset_index(drop=True)
    daily = _scan_frame(repository, snapshot, "daily_market", _DAILY_COLUMNS, tickers, held)
    if not daily.empty:
        daily = daily.sort_values("date").reset_index(drop=True)
    return events, daily


class _SourceStats:
    """Per-ticker aggregates of the two source tables: O(tickers) state, never rows."""

    def __init__(self) -> None:
        self.event_rows, self.daily_rows, self.past = Counter(), Counter(), Counter()
        self.latest: dict[str, pd.Timestamp] = {}

    def add_confirmed(self, events: pd.DataFrame, day) -> None:
        """Fold in ORATS-confirmed sessioned events: those before ``day``, and each ticker's newest."""
        if events.empty:
            return
        dates, tickers = pd.to_datetime(events["event_date"]), events["ticker"].astype(str)
        if day is not None:
            self.past.update(tickers[dates < day].value_counts().to_dict())
        for ticker, newest in dates.groupby(tickers).max().items():
            if pd.notna(newest) and (ticker not in self.latest or newest > self.latest[ticker]):
                self.latest[ticker] = newest


def _scan_stats(repository: Repository, snapshot, stats: _SourceStats, day, *,
                events: bool = True, daily: bool = True) -> None:
    """Stream the source tables into ``stats``, one lease at a time."""
    if events:
        for lease in _scan_rows(repository, snapshot, "earnings_events", _EVENT_COLUMNS):
            with lease as batch:
                frame = pd.DataFrame(batch)
                if not frame.empty:
                    stats.event_rows.update(frame["ticker"].astype(str))
                    stats.add_confirmed(
                        frame[frame["src_orats"] & frame["session"].notna()], day)
    if daily:
        for lease in _scan_rows(repository, snapshot, "daily_market", _DAILY_COLUMNS):
            with lease as batch:
                stats.daily_rows.update(str(row["ticker"]) for row in batch)


class _TickerChunks:
    """Each target ticker's source frames, loaded one bounded ticker chunk at a time.

    Targets, in plan order, are packed into chunks whose event plus daily rows
    total at most ``MAX_SCAN_ROWS`` less :func:`_lease_cap`, and chunk scans
    lease at most that cap, so the rows held here and the in-flight lease
    together stay within ``MAX_SCAN_ROWS`` however long the source history is
    (``held.peak_rows`` is that total). A chunk is loaded on its first ticker's
    request and released (the charge returned) before the next one loads. One
    ticker alone above the bound is refused: nothing partial is built.
    """

    def __init__(self, repository: Repository, snapshot, order, stats: _SourceStats) -> None:
        self._repository, self._snapshot = repository, snapshot
        self.held = RetainedRowCount()
        budget = MAX_SCAN_ROWS - _lease_cap()
        self._chunks: list[list[str]] = []
        self._index: dict[str, int] = {}
        size = 0
        for ticker in order:
            rows = stats.event_rows[ticker] + stats.daily_rows[ticker]
            if rows > budget:
                raise fail("RESOURCE_LIMIT_EXCEEDED",
                           "computed moves source rows for one ticker exceed the retained-row guard",
                           details={"limit": budget})
            if not self._chunks or size + rows > budget:
                self._chunks.append([])
                size = 0
            self._chunks[-1].append(ticker)
            size += rows
            self._index[ticker] = len(self._chunks) - 1
        self._loaded: int | None = None
        self._groups: tuple[dict, dict] = ({}, {})

    def _for(self, ticker: str) -> tuple[dict, dict]:
        index = self._index.get(ticker)
        if index is None:
            return {}, {}
        if index != self._loaded:
            self._groups, self._loaded = ({}, {}), None
            self.held.discharge(self.held.live_rows)
            events, daily = _scan_once(self._repository, self._snapshot,
                                       set(self._chunks[index]), self.held)
            self._groups, self._loaded = (_group_by_ticker(events), _group_by_ticker(daily)), index
        return self._groups

    def events(self, ticker: str) -> pd.DataFrame | None:
        return self._for(ticker)[0].get(ticker)

    def daily(self, ticker: str) -> pd.DataFrame:
        return self._for(ticker)[1].get(ticker, _EMPTY_DAILY)


def _group_by_ticker(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if frame.empty:
        return {}
    return {str(ticker): group for ticker, group in frame.groupby("ticker")}


def _selection_stats(repository: Repository, parent_snapshot_id: str,
                     events: pd.DataFrame | None, daily: pd.DataFrame | None, as_of) -> _SourceStats:
    """Selection aggregates from the caller's frames, streaming whichever table is absent.

    Counters only, never a whole-table frame. A bad ``as_of`` refuses after the
    scan, so a source refusal still outranks it exactly as before.
    """
    try:
        day, bad_as_of = _as_of_day(as_of), None
    except OpsError as error:
        day, bad_as_of = None, error
    stats = _SourceStats()
    if events is not None:
        stats.add_confirmed(events, day)
    if daily is not None and not daily.empty:
        stats.daily_rows.update(daily["ticker"].astype(str))
    if events is None or daily is None:
        _scan_stats(repository, repository.resolve(parent_snapshot_id), stats, day,
                    events=events is None, daily=daily is None)
    if bad_as_of is not None:
        raise bad_as_of
    return stats


def target_tickers_from_snapshot(repository: Repository, parent_snapshot_id: str, *,
                                 all_scoreable: bool = True, since=None,
                                 oquants_tickers=(), events: pd.DataFrame | None = None,
                                 daily: pd.DataFrame | None = None,
                                 stats: _SourceStats | None = None,
                                 as_of) -> tuple[list[str], dict]:
    """The legacy ``target_tickers`` rule, read through the v2 snapshot.

    Same selection as ``engine.data.pulls.computed_moves.target_tickers``:
    tickers with at least ``MIN_SCOREABLE`` past ORATS-confirmed sessioned
    events AND at least one ``daily_market`` row. ``oquants_tickers`` is the
    caller's optional replacement for the legacy extension-only oquants glob
    (the v2 catalog carries no oquants moves table); it only matters for
    ``all_scoreable=False``. ``events``/``daily`` are the caller's already
    scanned frames (spec s4c Rewrite 3) and ``stats`` its already streamed
    per-ticker aggregates; omitted, this streams them itself, keeping counters
    only -- never a whole-table frame.
    ``as_of`` is the job's own session date (spec R5): selection NEVER reads
    the wall clock, so the store's plan hash equals the nightly's.
    """
    if stats is None:
        stats = _selection_stats(repository, parent_snapshot_id, events, daily, as_of)
    _as_of_day(as_of)  # a stats-bearing caller is validated too
    scoreable = {ticker for ticker, count in stats.past.items() if count >= MIN_SCOREABLE}
    dm_tickers = set(stats.daily_rows)

    oq_tickers = {str(ticker) for ticker in oquants_tickers}
    pool = scoreable if all_scoreable else (scoreable - oq_tickers)
    targets = sorted(pool & dm_tickers)
    report = {
        "mode": "all_scoreable" if all_scoreable else "extension_only",
        "scoreable_on_orats_calendar": len(scoreable),
        "also_in_oquants": len(scoreable & oq_tickers),
        "no_daily_market_rows": len(pool - dm_tickers),
    }
    if since is not None:
        since = pd.Timestamp(_as_of_day(since))  # same validation as_of gets (Opus review)
        printed = {ticker for ticker, newest in stats.latest.items() if newest >= since}
        targets = [ticker for ticker in targets if ticker in printed]
        report["since"] = str(since.date())
        report["printed_since"] = len(printed)
    report["targets"] = len(targets)
    return targets, report


def computed_moves_units(targets, *, as_of) -> tuple[RefreshUnit, ...]:
    """One cacheable unit per target ticker -- the job's coverage denominator.

    The as_of date is part of the request id (spec R4): each session refetches
    its own units, so a later as_of never reuses a previous session's receipt.
    """
    day = _as_of_day(as_of)
    return tuple(RefreshUnit(
        request_id="computed_moves:" + str(ticker) + ":" + day,
        table_name=COMPUTED_MOVES_TABLE_NAME,
        partition_key=str(ticker), expected_keys=(str(ticker),)) for ticker in targets)


# --------------------------------------------------------------------------
# yfinance retrieval -- moved from the legacy pull's ``fetch_history``
# --------------------------------------------------------------------------


def fetch_history(fetcher, ticker: str) -> tuple[np.ndarray, np.ndarray] | None:
    """yfinance Close series (split-adjusted, not dividend-adjusted).

    ``fetcher(ticker)`` is the injected provider edge (the ops layer binds
    ``yfinance_history_fetcher()``); it returns
    ``(raw_bytes, response_kind, response_meta, rows)`` and classifies its own
    failures (spec R1). Only a ``complete`` response with parseable bytes is a
    history; a ``legitimate_empty`` name, a transient outage and unparseable
    bytes all yield ``None`` here, and the caller owns failing the job (R3).
    """
    raw, kind = _history_result(fetcher, ticker)
    if kind != "complete":
        return None
    return _parse_history(raw)


def _history_result(fetcher, ticker: str) -> tuple[bytes, str]:
    """Call the provider edge once and return ``(raw, response_kind)``.

    The edge classifies its own HTTP/library failures (R1); only a network
    exception out of a raising edge is classified ``transient`` here -- a
    programming error or an ``OpsError`` propagates rather than being silently
    treated as "no history" (R3). The caller aggregates kinds and fails the
    job when any unit ends non-complete.
    """
    try:
        raw, kind, _meta, _rows = fetcher(str(ticker))
    except (OSError, TimeoutError):
        return b"", "transient"
    if not isinstance(kind, str):
        return b"", "refused"
    return (raw or b""), kind


def _parse_history(raw: bytes) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except (ValueError, OSError):
        return None
    if frame.empty or "Close" not in frame.columns:
        return None
    date_col = frame.columns[0]
    dates = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
    dates = dates.dt.tz_localize(None).dt.normalize()
    closes = pd.to_numeric(frame["Close"], errors="coerce").to_numpy(dtype=float)
    ok = dates.notna() & np.isfinite(closes) & (closes > 0)
    dates = dates[ok].to_numpy(dtype="datetime64[ns]")
    closes = closes[ok]
    if closes.size == 0:
        return None
    order = np.argsort(dates, kind="stable")
    return dates[order], closes[order]


# --------------------------------------------------------------------------
# staging one ticker's fragment and committing the generation
# --------------------------------------------------------------------------


def _write_ticker_fragment(store: ArtifactStore, ticker: str, rows: list[dict], *,
                           request_hash: str):
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["event_date"]).reset_index(drop=True).copy()
    table = pa.Table.from_pandas(frame[list(FRAGMENT_COLUMNS)], schema=_ARROW_SCHEMA,
                                 preserve_index=False)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    raw = sink.getvalue().to_pybytes()
    ref = store.publish_bytes(raw, schema_ref=objects.PARQUET_FRAGMENT_SCHEMA_REF)
    object_ref = ObjectRef(kind="parquet_fragment", object_id=ref.artifact_id,
                           content_hash=ref.content_hash, byte_size=ref.byte_size)
    inspection = objects.inspect_fragment(store, object_ref, COMPUTED_MOVES_CONTRACT,
                                          _CONTRACT_REF, ticker)
    return manifests.fragment_record(inspection, _CONTRACT_REF, input_receipt_refs=(),
                                     import_request_hash=request_hash)


def _insert_captures(conn: sqlite3.Connection, attempts: list[dict]) -> None:
    """Record this run's captures once; a capture already logged is not re-logged.

    ``capture_id`` (``_capture_id_for``) folds in the unit's own ``request_id``
    (ticker + as_of day), the table contract's ``definition_hash``, and -- when
    a series was actually obtained -- the fetched source bytes' hash. Never
    the run's wall-clock time. Two attempts for the SAME unit with the SAME
    underlying content resolve to the SAME id (a retry never double-logs);
    different underlying content (a same-day upstream data change) or a
    different contract version gets a genuinely different id, never silently
    reused.
    """
    for attempt in attempts:
        existing = conn.execute(
            "SELECT 1 FROM data_computed_moves_captures WHERE capture_id = ?",
            (attempt["capture_id"],)).fetchone()
        if existing is not None:
            continue
        conn.execute(
            "INSERT INTO data_computed_moves_captures "
            "(capture_id, ticker, created_at, contract_id, outcome) VALUES (?, ?, ?, ?, ?)",
            (attempt["capture_id"], attempt["ticker"], attempt["created_at"],
             COMPUTED_MOVES_CONTRACT.contract_id, attempt["outcome"]))


def _fence_check_for(staged_attempt_id, staged_fence, clock):
    """The commit's own fence check against ``engine.v2.ops.lifecycle.verify_fence``'s
    REAL signature (``conn, attempt_id, fence, now`` -- it has no
    ``check_lease_time`` parameter to disable the wall-clock lease-expiry
    check with). Never disables that check: the production lease-expiry gate
    stays active here, same as every other fenced commit in this package.
    ``None`` staged_attempt_id (no live job behind this call, e.g. a manual or
    test invocation) is a no-op fence check.
    """
    if staged_attempt_id is None:
        return lambda connection: None
    return lambda connection: verify_fence(connection, staged_attempt_id, staged_fence,
                                           clock.now())


def _commit_generation(conn, store, scope, *, parent, parent_receipt_id, records_by_ticker,
                       attempts, clock, expected_head, generation, request_hash, as_of,
                       staged_attempt_id=None, staged_fence=None):
    prior_manifest = parent.table_manifests.get(COMPUTED_MOVES_TABLE_NAME)
    prior_records = tuple(record for record in parent.records
                          if record.table_contract_ref.contract_id
                          == COMPUTED_MOVES_CONTRACT.contract_id)
    rewritten = set(records_by_ticker)
    kept = tuple(record for record in prior_records if record.partition_key not in rewritten)
    # Issue #179 (contract: engine/v2/ops/ARCHITECTURE.md, "capture and
    # inherited fragments respect as_of"): capture-time truncation only
    # bounds the fragments THIS run writes, so a run at an earlier as_of
    # would inherit a prior fragment whose committed event dates reach on or
    # after that as_of -- realized data from the future, carried forward
    # untouched. Refuse the whole generation, non-retryably, BEFORE the
    # catalog transaction opens: the parent snapshot, head, and capture-log
    # rows all stay exactly as they were. Only inherited fragments are
    # inspected -- a rewritten ticker's old fragment is excluded here and
    # bounded by the capture-time truncation instead. `event_date` is the
    # second primary-key component and a normalized YYYY-MM-DD string, so it
    # compares directly against the job's own normalized as_of day.
    day = _as_of_day(as_of)
    for record in kept:
        if str(record.primary_key_max[1]) >= day:
            raise fail("VALIDATION_FAILED",
                       "computed moves refresh would inherit a committed fragment with "
                       "event dates on or after this run's as_of")
    new_records = tuple(sorted((*kept, *records_by_ticker.values()),
                               key=lambda item: (item.partition_key, item.primary_key_min)))
    parent_version = (prior_manifest.dataset_version_ref.dataset_version_id
                      if prior_manifest else None)
    manifest = manifests.dataset_manifest(
        _CONTRACT_REF, new_records, knowledge_mode="reconstructed",
        coverage_receipt_refs=(), availability_evidence_refs=(),
        parent_dataset_version_id=parent_version)

    table_manifests = {name: manifest_ for name, manifest_ in parent.table_manifests.items()
                       if name != COMPUTED_MOVES_TABLE_NAME}
    table_manifests[COMPUTED_MOVES_TABLE_NAME] = manifest
    snapshot = manifests.snapshot_ref(
        table_manifests, calendar_version=parent.snapshot.calendar_version,
        source_priority_version=parent.snapshot.source_priority_version,
        finality_receipt_refs=parent.snapshot.finality_receipt_refs,
        parent_snapshot_id=parent.snapshot.snapshot_id)

    other_records = tuple(record for record in parent.records
                          if record.table_contract_ref.contract_id
                          != COMPUTED_MOVES_CONTRACT.contract_id)
    all_records = other_records + new_records
    contracts = {item.contract_id: item for item in parent.contracts}
    contracts[COMPUTED_MOVES_CONTRACT.contract_id] = COMPUTED_MOVES_CONTRACT
    all_objects = tuple({record.object_ref.object_id: record.object_ref
                         for record in all_records}.values())
    receipt_id = "receipt_cm_" + request_hash.removeprefix("sha256:")[:32]
    attempt_id = "attempt_cm_" + request_hash.removeprefix("sha256:")[:32]
    # Resolve and validate the parent receipt BEFORE the commit opens: the
    # legacy lookup reads the newest committed receipt for the parent
    # snapshot, and an unchanged candidate names that same snapshot, so a
    # lookup after the candidate's own insert would find the candidate (no
    # reference inputs yet) and raise SNAPSHOT_NOT_READY.
    resolved_parent_receipt_id = _parent_receipt_id_for_commit(
        conn, scope, parent.snapshot.snapshot_id, parent_receipt_id)

    def _record_references(connection, rid):
        inputs = reference_catalog.reference_inputs_for_receipt(
            connection, receipt_id=resolved_parent_receipt_id)
        reference_catalog.insert_reference_inputs(connection, rid, inputs)
        _insert_captures(connection, attempts)
        record_price_history_lineage(
            connection, receipt_id=rid, base_receipt_id=resolved_parent_receipt_id)

    return data_catalog.commit_snapshot(
        conn, scope=scope, request_hash=request_hash, contracts=tuple(contracts.values()),
        objects=all_objects, records=all_records, manifests=tuple(table_manifests.values()),
        snapshot=snapshot, expected_head_snapshot_id=expected_head,
        expected_head_generation=generation, receipt_id=receipt_id, attempt_id=attempt_id,
        fence=1,
        fence_check=_fence_check_for(staged_attempt_id, staged_fence, clock),
        clock=clock, store=store, record_references=_record_references,
        audit_partitions=False)


# --------------------------------------------------------------------------
# the standalone runner (Part 3 adapts this to RefreshCallback)
# --------------------------------------------------------------------------


def _noop_result(parameters, completed_ids) -> RefreshCallbackResult:
    """A truthful rerun result: nothing committed, nothing advanced.

    The plan hash is the job's own binding (``parameters.refresh_plan_hash``),
    never a recomputed plan: a retry sees its units already cached, so its
    recomputed plan is not the plan the job reserved budget for.
    """
    return RefreshCallbackResult(
        status="noop", completed_ids=completed_ids, coverage_advanced=False,
        parent_snapshot_id=parameters.parent_snapshot_id,
        refresh_plan_hash=parameters.refresh_plan_hash)


def _no_fragment_result(conn, parameters, attempts, targets, document, parent, clock):
    """Persist fenced audit attempts without publishing a generation."""
    with transaction(conn):
        _fence_check_for(document.get("attempt_id"), document.get("fence"), clock)(conn)
        _parent_receipt_id_for_commit(
            conn, document["scope"], parent.snapshot.snapshot_id,
            parameters.parent_receipt_id)
        _insert_captures(conn, attempts)
    return _noop_result(parameters, tuple(sorted(targets)))


def _input_document(root: Path) -> dict | None:
    path = root / INPUT_PATH
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict):
        return None
    return document


def _validate_document_identity(document: dict) -> None:
    """Unknown keys, and the three bounded non-empty string fields every
    input document must carry."""
    unknown = set(document) - _ALLOWED_DOCUMENT_KEYS
    if unknown:
        raise fail("INVALID_REQUEST",
                   "computed moves refresh input document has unknown keys",
                   details={"unknown_keys": sorted(unknown)})
    for name in ("catalog_path", "objects_root", "scope"):
        value = document.get(name)
        if not isinstance(value, str) or not value:
            raise fail("INVALID_REQUEST",
                       f"computed moves refresh input document needs a non-empty string {name}")
    if document.get("parent_receipt_id") is not None:
        # An explicit null is the same legacy/unpinned state as an omitted
        # field: production documents staged before ``parent_receipt_id``
        # existed serialize it as null, and ``_parent_receipt_id_for_commit``
        # resolves the receipt for that state. Every non-null value is still
        # validated here, and a non-null pin that disagrees with the job or is
        # not committed for the parent is still refused.
        _validated_parent_receipt_id(document["parent_receipt_id"])
    if not Path(document["catalog_path"]).is_file():
        # ``sqlite3.connect`` is never allowed to silently create a fresh,
        # empty database at a path that does not already hold the real
        # catalog -- the same rule sibling PR #40 applies in
        # forward_calendar_store.py.
        raise fail("INVALID_REQUEST",
                   f"catalog_path does not exist as a file: {document['catalog_path']!r}")
    if not Path(document["objects_root"]).is_dir():
        raise fail("INVALID_REQUEST",
                   f"objects_root does not exist as a directory: {document['objects_root']!r}")
    if document["scope"] not in _ALLOWED_SCOPES:
        raise fail("INVALID_REQUEST",
                   "computed moves refresh scope is not an allowed namespace",
                   details={"scope": document["scope"], "allowed": sorted(_ALLOWED_SCOPES)})


def _validate_document_head(document: dict) -> None:
    """The commit-target fields: generation and the optional expected head."""
    generation = document.get("expected_head_generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise fail("INVALID_REQUEST", "computed moves refresh input document needs a "
                                       "non-negative integer expected_head_generation")
    head = document.get("expected_head_snapshot_id")
    if head is not None and (not isinstance(head, str) or not head or len(head) > 128):
        raise fail("INVALID_REQUEST",
                   "expected_head_snapshot_id must be a bounded non-empty string when present")


def _validate_document_attempt(document: dict) -> None:
    """The optional staged-job identity fields: attempt id and fence. Never
    checked against the catalog here -- a head mismatch is caught at commit
    time (``SNAPSHOT_CONFLICT``), after any fetches; this is format-only,
    before any I/O. The two fields are then cross-checked against each
    other (``engine.v2.ops.lifecycle.validated_attempt_fence_pair``, issue
    #58): exactly one set is refused up front, before any I/O, as
    ``INVALID_REQUEST`` -- a bare ``fence`` with no ``attempt_id`` would
    otherwise make this module's own ``_fence_check_for`` a no-op,
    committing unfenced (fail-open), and a bare ``attempt_id`` with no
    ``fence`` would otherwise only be refused later, inside
    ``verify_fence`` itself, after I/O. ``None``/``None`` (no live job
    behind this call) and both set are the only two valid shapes; this
    same helper also guards ``forward_calendar_store``'s identical
    ``attempt_id``/``fence`` keyword pair."""
    attempt_id = document.get("attempt_id")
    if attempt_id is not None and (not isinstance(attempt_id, str) or not attempt_id):
        raise fail("INVALID_REQUEST", "attempt_id must be a non-empty string when present")
    fence = document.get("fence")
    if fence is not None and (isinstance(fence, bool) or not isinstance(fence, int)
                              or fence < 1):
        raise fail("INVALID_REQUEST", "fence must be an int >= 1 when present")
    validated_attempt_fence_pair(attempt_id, fence)


def _validate_document_selection(document: dict) -> None:
    """``all_scoreable``/``since``, the two fields that steer target selection."""
    if "all_scoreable" in document and not isinstance(document["all_scoreable"], bool):
        # The old call site did ``bool(document.get("all_scoreable", True))``:
        # ``bool("false")`` is ``True`` in Python, so a string value was
        # silently inverted rather than refused. A real JSON boolean decodes
        # as a Python ``bool`` already; anything else is refused.
        raise fail("INVALID_REQUEST", "all_scoreable must be a real boolean, not a coerced value")
    if document.get("since") is not None:
        _as_of_day(document["since"])  # same validation as_of gets; raises on a bad value


def _validate_document_matches_job(document: dict, parameters, *, as_of: str) -> None:
    """The document's own identity must agree with ``parameters`` and the
    validated ``as_of`` keyword -- never silently diverge."""
    if "as_of" in document:
        # The document's own as_of was staged but never read -- dead data a
        # caller could set to anything without effect (the same defect class
        # as sibling PR #40's RefreshParameters.as_of/tickers mismatch).
        # Rather than refuse its presence outright (every staged input
        # already writes it), this store requires it to agree with the
        # validated ``as_of`` keyword: a documented choice, not a silent one.
        if _as_of_day(document["as_of"]) != _as_of_day(as_of):
            raise fail("INVALID_REQUEST",
                       "computed moves refresh input document's as_of disagrees with the "
                       "job's as_of")
    if document["catalog_path"] != parameters.catalog_path:
        raise fail("INVALID_REQUEST", "computed moves refresh input document's catalog_path "
                                       "disagrees with the job's RefreshParameters")
    if document["objects_root"] != parameters.objects_root:
        raise fail("INVALID_REQUEST", "computed moves refresh input document's objects_root "
                                       "disagrees with the job's RefreshParameters")
    if document["scope"] != parameters.scope:
        raise fail("INVALID_REQUEST", "computed moves refresh input document's scope disagrees "
                                       "with the job's RefreshParameters")
    if document.get("parent_receipt_id") != parameters.parent_receipt_id:
        raise fail("INVALID_REQUEST", "computed moves input parent_receipt_id disagrees "
                                      "with the job parameters")
    if document.get("expected_head_generation") != parameters.expected_head_generation:
        raise fail("INVALID_REQUEST", "computed moves refresh input document's "
                                       "expected_head_generation disagrees with the job's "
                                       "RefreshParameters")


def _validate_input_document(document: dict, parameters, *, as_of: str) -> None:
    """Every field the staged input document carries, validated before any
    I/O (sqlite connect, snapshot resolve, fetch, or receipt write) -- the
    same discipline ``_as_of_day`` already applies to the ``as_of`` keyword
    (Opus review, PR #39). An unknown key, a wrong type, or a value that
    disagrees with the job's own ``RefreshParameters`` is refused, never
    coerced or silently ignored.
    """
    _validate_document_identity(document)
    _validate_document_head(document)
    _validate_document_attempt(document)
    _validate_document_selection(document)
    _validate_document_matches_job(document, parameters, as_of=as_of)


def _validated_parent_snapshot_id(parent_snapshot_id) -> str:
    """Matches ``incremental_data``'s own ``RefreshParameters`` rule for this
    same field: a bounded nonempty string. Mirrors sibling PR #40's
    ``_validated_parent_snapshot_id`` (not imported: unmerged sibling)."""
    if (not isinstance(parent_snapshot_id, str) or not parent_snapshot_id
            or len(parent_snapshot_id) > 128):
        raise fail("INVALID_REQUEST",
                   f"parent_snapshot_id must be a bounded nonempty str, "
                   f"got {parent_snapshot_id!r}")
    return parent_snapshot_id


def _validated_parent_receipt_id(parent_receipt_id) -> str:
    if (not isinstance(parent_receipt_id, str) or not parent_receipt_id
            or len(parent_receipt_id) > 128):
        raise fail("INVALID_REQUEST",
                   f"parent_receipt_id must be a bounded nonempty str, "
                   f"got {parent_receipt_id!r}")
    return parent_receipt_id


def _parent_receipt_id_for_commit(conn, scope, snapshot_id, pinned_receipt_id):
    if pinned_receipt_id is None:
        # Compatibility for jobs serialized before parent_receipt_id was added.
        return parent_receipt_id_for_snapshot(conn, scope, snapshot_id)
    receipt_id = _validated_parent_receipt_id(pinned_receipt_id)
    row = conn.execute(
        "SELECT 1 FROM data_import_receipts WHERE receipt_id = ? AND scope = ? "
        "AND status = 'committed' AND result_snapshot_id = ?",
        (receipt_id, scope, snapshot_id)).fetchone()
    if row is None:
        raise fail("SNAPSHOT_NOT_READY",
                   "pinned parent receipt is not committed for the parent snapshot",
                   details={"scope": scope, "snapshot_id": snapshot_id,
                            "receipt_id": receipt_id})
    return receipt_id


def _validated_refresh_plan_hash(refresh_plan_hash) -> str:
    """Matches ``incremental_data._is_hash``'s own sha256-hex check for this
    same field (mirrored rather than imported: that name is private, and #40's
    copy of it is an unmerged sibling)."""
    valid = (isinstance(refresh_plan_hash, str)
             and refresh_plan_hash.startswith("sha256:") and len(refresh_plan_hash) == 71
             and all(char in "0123456789abcdef" for char in refresh_plan_hash[7:]))
    if not valid:
        raise fail("INVALID_REQUEST",
                   f"refresh_plan_hash must be a sha256 content hash, "
                   f"got {refresh_plan_hash!r}")
    return refresh_plan_hash


def _validate_job_identity(parameters) -> None:
    """``parameters``' own identity fields, validated before any I/O: this
    runner is called directly, not through the job-submission pipeline that
    would otherwise have validated them via ``refresh_parameter_problems``."""
    _validated_parent_snapshot_id(parameters.parent_snapshot_id)
    if parameters.parent_receipt_id is not None:
        _validated_parent_receipt_id(parameters.parent_receipt_id)
    _validated_refresh_plan_hash(parameters.refresh_plan_hash)


def run_computed_moves_refresh(parameters, root, *, as_of, fetcher=None,
                               use_cached_receipts: bool = True) -> RefreshCallbackResult:
    """This job's own callback (spec s4b Change 3), called directly -- NOT
    bound to ``RefreshCallback`` via a bare ``functools.partial``: ``main``'s
    ``RefreshParameters`` has no ``as_of`` field, so ``as_of`` is an explicit,
    validated, required keyword. Validates the staged input document up front,
    selects targets from the pinned parent snapshot with ONE scan per source
    table, and commits one new snapshot. A same-``as_of`` rerun genuinely
    no-ops: every committed row's ``computed_at`` derives from ``as_of``, so
    identical inputs commit identical bytes. ``use_cached_receipts`` picks the
    plan's cache policy (``_plan_cached_outcomes``); the supervised default
    is unchanged."""
    as_of_day = _as_of_day(as_of)  # validated before any I/O; a bad value raises
    _validate_job_identity(parameters)
    root = Path(root)
    document = _input_document(root)
    if document is None:
        raise fail("INVALID_REQUEST",
                   "computed moves refresh input document is missing or malformed")
    _validate_input_document(document, parameters, as_of=as_of)
    if fetcher is None:
        raise fail("RESOURCE_UNAVAILABLE", "no computed_moves history fetcher is configured")
    # mode=rw: never let sqlite3 silently create a fresh, empty database if
    # the file was removed between _validate_document_identity's is_file()
    # check and this connect (TOCTOU) -- it raises instead.
    conn = sqlite3.connect(f"file:{document['catalog_path']}?mode=rw", uri=True)
    conn.row_factory = sqlite3.Row
    clock = SystemClock()
    try:
        store = ArtifactStore(document["objects_root"])
        repository = Repository(conn, store)
        parent = repository.resolve_full(parameters.parent_snapshot_id)
        stats = _SourceStats()
        _scan_stats(repository, parent.snapshot, stats, as_of_day)
        targets, _selection = target_tickers_from_snapshot(
            repository, parameters.parent_snapshot_id,
            all_scoreable=document.get("all_scoreable", True),
            since=document.get("since"), stats=stats, as_of=as_of)
        units = computed_moves_units(targets, as_of=as_of)
        plan = plan_refresh(
            parent.snapshot, units,
            cached_outcomes=_plan_cached_outcomes(conn, units, use_cached_receipts),
            provider_account=NATIVE_COMPUTED_MOVES_ACCOUNT,
            expected_head_generation=int(document["expected_head_generation"]))
        fragment_records, attempts = _capture_targets(
            conn, store, plan, fetcher, clock,
            inputs=_TickerChunks(repository, parent.snapshot,
                                 [unit.expected_keys[0] for unit in plan.units], stats),
            as_of_day=as_of_day)
        if not fragment_records:
            return _no_fragment_result(
                conn, parameters, attempts, targets, document, parent, clock)

        request_hash = content_hash({
            "kind": "computed_moves_generation", "scope": document["scope"],
            "base_snapshot_id": parent.snapshot.snapshot_id,
            "fragments": {ticker: record.fragment_id
                          for ticker, record in sorted(fragment_records.items())}})
        receipt = _commit_generation(
            conn, store, str(document["scope"]), parent=parent,
            parent_receipt_id=parameters.parent_receipt_id,
            records_by_ticker=fragment_records, attempts=attempts, clock=clock, as_of=as_of_day,
            expected_head=document.get("expected_head_snapshot_id",
                                       parameters.parent_snapshot_id),
            generation=int(document["expected_head_generation"]), request_hash=request_hash,
            staged_attempt_id=document.get("attempt_id"), staged_fence=document.get("fence"))
        if receipt.resulting_head_snapshot_id == parent.snapshot.snapshot_id:
            # The commit layer's own result decides: the candidate resolved
            # back to the parent snapshot, so the head did not move and
            # nothing was committed. Key presence in the parent is never
            # consulted -- yesterday's partition or a cached receipt is not
            # today's committed content.
            return _noop_result(parameters, tuple(sorted(targets)))
        return RefreshCallbackResult(
            status="complete", completed_ids=tuple(sorted(targets)),
            coverage_advanced=True, parent_snapshot_id=parameters.parent_snapshot_id,
            refresh_plan_hash=parameters.refresh_plan_hash,
            candidate_snapshot_id=receipt.resulting_head_snapshot_id)
    finally:
        conn.close()


def _plan_cached_outcomes(conn, units, use_cached_receipts):
    """The plan's cache set under this run's explicit policy.

    ``use_cached_receipts`` is the narrowest explicit control over the catalog
    raw-receipt cache. ``True`` -- the supervised worker's existing behavior --
    reuses a durable same-unit receipt instead of re-fetching (spec R2). The
    local ``computed-moves capture`` CLI sets it ``False`` because its selected
    Tier-1 source root is authoritative: an old same-unit catalog receipt must
    never override changed bytes, or a missing entry, in that selected source.
    Both the CLI and the runner build their plan with this same policy, so
    their plan identity agrees.
    """
    if not use_cached_receipts:
        return {}
    return cached_unit_outcomes(
        conn, units, source=COMPUTED_MOVES_TABLE_NAME,
        endpoint=COMPUTED_MOVES_TABLE_NAME)


def _unit_history(conn, store, unit, fetcher, cached, *, created_at):
    """One unit's parsed Close series, from the edge or a cached receipt.

    A unit in the plan's cache set is never re-fetched (spec R2): its verified
    receipt bytes are re-parsed instead, so a same-session retry rebuilds the
    series a clean run had. Only a fresh unit's bytes are cached, and only
    after they parse (R2). Returns ``(series, kind, fresh)``.
    """
    fresh = unit.request_id not in cached
    if fresh:
        raw, kind = _history_result(fetcher, unit.expected_keys[0])
    else:
        raw, kind = cached[unit.request_id], "complete"
    series = (_parse_history(raw) if raw and kind in ("complete", "legitimate_empty")
              else None)
    if kind == "complete" and series is None:
        kind = "refused"
    if series is not None and fresh:
        record_unit_receipt(conn, store, unit, raw, source=COMPUTED_MOVES_TABLE_NAME,
                            endpoint=COMPUTED_MOVES_TABLE_NAME, received_at=created_at,
                            response_kind=kind)
    return series, kind, fresh


def _capture_id_for(unit, *, source_hash: str | None = None,
                    definition_hash: str = COMPUTED_MOVES_CONTRACT.definition_hash) -> str:
    """The stable capture identity for one refresh unit's ACTUAL content.

    Same (ticker, as_of) always yields the same id, regardless of when this
    run's wall clock reads: derived from the unit's own ``request_id``
    (``computed_moves:<ticker>:<day>``, spec R4/R5's per-session request id),
    never from the run's ``created_at``. request_id ALONE is not enough,
    though: two attempts for the same unit can carry genuinely different
    content -- a same-day yfinance correction/backfill changes the fetched
    bytes, or a code change moves the table's ``definition_hash`` -- and a
    request_id-only id would collide those into one row, silently keeping
    only the first attempt's outcome. ``source_hash`` (the fetched series'
    hash, when a series was actually obtained) and ``definition_hash`` (the
    contract's own schema identity) are folded in too, so a same-request-id
    attempt with different real content gets a genuinely different id.
    ``source_hash=None`` (no series obtained -- a failed/too-few outcome) omits
    it: there is no immutable source content to distinguish by, so two failed
    attempts for the same unit under the same code are correctly the same
    capture. A rerun of the same unit with the SAME content -- a crash-retry,
    or a same-session no-op replay -- must still resolve to the identity
    ``_insert_captures`` already logged, or its dedup-by-``capture_id`` check
    never dedups anything.
    """
    payload = {"request_id": unit.request_id, "contract_definition_hash": definition_hash}
    if source_hash is not None:
        payload["source_hash"] = source_hash
    return "capture_" + content_hash(payload).removeprefix("sha256:")[:32]


def _capture_targets(conn, store, plan, fetcher, clock, *, inputs: _TickerChunks,
                     as_of_day: str):
    """Acquire and stage one fragment per unit, fresh or cached, exactly once.

    The caller reaches this only when the plan has at least one fresh fetch, and
    EVERY wanted unit is rebuilt here -- the fresh fetches plus the cached
    ``complete`` receipts re-read and re-parsed by receipt -- so a same-session
    retry commits exactly what a clean single run would. Any unit that ends
    transient/refused/not_final fails the whole job (R3) after every unit has
    been attempted, with the good receipts already cached.

    ``created_at`` (this run's wall clock) tags the audit-only capture log and
    raw-receipt rows -- when this attempt happened, never content identity.
    Every COMMITTED row's ``computed_at`` uses ``as_of_day`` instead (Opus
    review, PR #39; was tracked as #41): the fragment's bytes, and so its
    content-addressed identity, must be a pure function of ``as_of`` and the
    fetched source, or a same-``as_of`` rerun with identical inputs commits a
    new generation instead of resolving to a true no-op.
    """
    created_at = clock.now().isoformat()
    cached = cached_unit_payloads(conn, store, plan)
    fragment_records, attempts, kinds = {}, [], []
    for unit in plan.units:
        ticker = unit.expected_keys[0]
        series, kind, fresh = _unit_history(conn, store, unit, fetcher, cached,
                                            created_at=created_at)
        kinds.append(kind)
        if series is None:
            capture_id = _capture_id_for(unit)
            attempts.append({"capture_id": capture_id, "ticker": ticker,
                             "created_at": created_at,
                             "outcome": ("no_history" if kind == "legitimate_empty"
                                         else kind)})
            continue
        events = inputs.events(ticker)
        if events is None:
            capture_id = _capture_id_for(unit)
            attempts.append({"capture_id": capture_id, "ticker": ticker,
                             "created_at": created_at, "outcome": "too_few"})
            continue
        sd, sc = series
        # Issue #99: the fetch returns history up to the run's real wall
        # clock, later than as_of_day on a backfill re-run. Truncate the
        # series BEFORE hashing (capture identity stays a pure function of
        # as_of + the truncated source) and bound events to strictly before
        # as_of_ts, or a post-as_of event scored against real future closes
        # commits as a row stamped computed_at = as_of_day.
        as_of_ts = pd.Timestamp(as_of_day)
        keep = sd <= np.datetime64(as_of_ts)
        sd, sc = sd[keep], sc[keep]
        if sd.size == 0:
            capture_id = _capture_id_for(unit)
            attempts.append({"capture_id": capture_id, "ticker": ticker,
                             "created_at": created_at, "outcome": "too_few"})
            continue
        source_hash = hashlib.sha256(np.ascontiguousarray(sc).tobytes()).hexdigest()
        capture_id = _capture_id_for(unit, source_hash=source_hash)
        events = events[(events["event_date"] >= pd.Timestamp(sd[0]))
                        & (events["event_date"] < as_of_ts)]
        rows = build_rows(
            ticker, events, sd, sc, inputs.daily(ticker),
            computed_at=as_of_day, source_hash=source_hash, capture_id=capture_id)
        if not rows:
            attempts.append({"capture_id": capture_id, "ticker": ticker,
                             "created_at": created_at, "outcome": "too_few"})
            continue
        request_hash = content_hash({"kind": "computed_moves_fragment", "ticker": ticker,
                                     "capture_id": capture_id})
        fragment_records[ticker] = _write_ticker_fragment(store, ticker, rows,
                                                          request_hash=request_hash)
        attempts.append({"capture_id": capture_id, "ticker": ticker,
                         "created_at": created_at,
                         "outcome": "added" if fresh else "cached"})
    code = provider_failure_code(kinds)
    if code:
        raise fail(code, "computed moves provider response was not complete")
    return fragment_records, attempts
