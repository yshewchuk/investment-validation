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
``yfinance_history_fetcher`` (never ``legacy_adapter.new_fetcher``), and both
source tables are scanned exactly ONCE per run -- selection and capture share
the same two frames (``_scan_once``), because the old per-ticker
``_events_for``/``_daily_for`` made 2xN full-table scans for N targets.

The pure close-to-close math lives in :mod:`engine.v2.data.computed_moves`;
this module is the impure orchestrator.
"""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from engine.v2.contracts import DataQuery, KeyPredicate, ObjectRef, TableContractRef
from engine.v2.data import catalog as data_catalog
from engine.v2.data import manifests, objects
from engine.v2.data.computed_moves import MIN_SCOREABLE, build_rows
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_TABLE_NAME
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.ops.calendar_moves_jobs import (
    NATIVE_COMPUTED_MOVES_ACCOUNT,
    cached_unit_outcomes,
    cached_unit_payloads,
    provider_failure_code,
    record_unit_receipt,
)
from engine.v2.ops.errors import fail
from engine.v2.ops.incremental_data import (
    RefreshCallbackResult,
    RefreshUnit,
    plan_refresh,
)

__all__ = [
    "computed_moves_units",
    "fetch_history",
    "run_computed_moves_refresh",
    "target_tickers_from_snapshot",
]

INPUT_PATH = "computed_moves_refresh_input.json"
MAX_SCAN_ROWS = 2_000_000
FRAGMENT_COLUMNS = ("ticker", "event_date", "realized_move_pct", "implied_move_pct",
                    "quarter_ordinal", "skipped", "computed_at", "source_hash", "capture_id")

_EMPTY_DAILY = pd.DataFrame(columns=["date", "implied_move"])

_ARROW_SCHEMA = pa.schema([
    ("ticker", pa.string()), ("event_date", pa.string()), ("realized_move_pct", pa.float64()),
    ("implied_move_pct", pa.float64()), ("quarter_ordinal", pa.int64()), ("skipped", pa.bool_()),
    ("computed_at", pa.string()), ("source_hash", pa.string()), ("capture_id", pa.string()),
])

_CONTRACT_REF = TableContractRef(contract_id=COMPUTED_MOVES_CONTRACT.contract_id,
                                 definition_hash=COMPUTED_MOVES_CONTRACT.definition_hash)


def _as_of_day(as_of) -> str:
    """The job's as_of date as ``YYYY-MM-DD``; a missing as_of is refused.

    Spec R4/R5: the as_of is part of every unit id and the only date selection
    may use, so it is required, never defaulted to the wall clock.
    """
    day = pd.Timestamp(as_of).normalize()
    if pd.isna(day):
        raise fail("INVALID_REQUEST", "computed moves refresh needs an as_of date")
    return str(day.date())


# --------------------------------------------------------------------------
# snapshot reads (the v2 read path, never engine.data.store)
# --------------------------------------------------------------------------


def _scan_rows(repository: Repository, snapshot, table_name: str, columns) -> list[dict]:
    contract_ref = snapshot.table_versions[table_name].table_contract_ref
    contract = repository.table_contract(snapshot, table_name)
    years = tuple(sorted({int(record.partition_key)
                          for record in repository.fragment_records(snapshot, table_name)}))
    if not years:
        return []
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=contract_ref,
        columns=tuple(columns),
        key_filter=(KeyPredicate(column="year", operator="in", values=years),),
        order_by=tuple(contract.primary_key),
        max_batch_rows=min(contract.maximum_batch_rows, 50_000),
        max_result_rows=min(contract.maximum_result_rows, MAX_SCAN_ROWS))
    rows: list[dict] = []
    for batch in repository.scan(query, table_name=table_name):
        rows.extend(batch.to_pylist())
    return rows


def _scan_once(repository: Repository, snapshot) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One scan of each source table, shared by target selection and capture.

    Returns the ORATS-confirmed ``earnings_events`` frame (``event_date``
    parsed) and the ``daily_market`` frame, both sorted by key. Every consumer
    in this run indexes these frames; nothing scans per ticker again.
    """
    events = pd.DataFrame(_scan_rows(repository, snapshot, "earnings_events",
                                     ("ticker", "event_date", "session", "src_orats")))
    if not events.empty:
        events = events[events["src_orats"] & events["session"].notna()].copy()
        events["event_date"] = pd.to_datetime(events["event_date"])
        events = events.sort_values("event_date").reset_index(drop=True)
    daily = pd.DataFrame(_scan_rows(repository, snapshot, "daily_market",
                                    ("ticker", "date", "implied_move")))
    if not daily.empty:
        daily = daily.sort_values("date").reset_index(drop=True)
    return events, daily


def _group_by_ticker(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if frame.empty:
        return {}
    return {str(ticker): group for ticker, group in frame.groupby("ticker")}


def target_tickers_from_snapshot(repository: Repository, parent_snapshot_id: str, *,
                                 all_scoreable: bool = True, since=None,
                                 oquants_tickers=(), events: pd.DataFrame | None = None,
                                 daily: pd.DataFrame | None = None,
                                 as_of) -> tuple[list[str], dict]:
    """The legacy ``target_tickers`` rule, read through the v2 snapshot.

    Same selection as ``engine.data.pulls.computed_moves.target_tickers``:
    tickers with at least ``MIN_SCOREABLE`` past ORATS-confirmed sessioned
    events AND at least one ``daily_market`` row. ``oquants_tickers`` is the
    caller's optional replacement for the legacy extension-only oquants glob
    (the v2 catalog carries no oquants moves table); it only matters for
    ``all_scoreable=False``. ``events``/``daily`` are the caller's already
    scanned frames (spec s4c Rewrite 3); omitted, this scans once itself.
    ``as_of`` is the job's own session date (spec R5): selection NEVER reads
    the wall clock, so the store's plan hash equals the nightly's.
    """
    if events is None or daily is None:
        snapshot = repository.resolve(parent_snapshot_id)
        scanned_events, scanned_daily = _scan_once(repository, snapshot)
        events = scanned_events if events is None else events
        daily = scanned_daily if daily is None else daily
    day = _as_of_day(as_of)
    if events.empty:
        scoreable: set[str] = set()
    else:
        hist = events[pd.to_datetime(events["event_date"]) < day]
        counts = hist.groupby("ticker")["event_date"].size()
        scoreable = set(counts[counts >= MIN_SCOREABLE].index)
    dm_tickers = set(daily["ticker"].astype(str)) if not daily.empty else set()

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
        since = pd.Timestamp(since).normalize()
        recent = events[pd.to_datetime(events["event_date"]) >= since]
        printed = set(recent["ticker"].astype(str))
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
    for attempt in attempts:
        conn.execute(
            "INSERT INTO data_computed_moves_captures "
            "(capture_id, ticker, created_at, contract_id, outcome) VALUES (?, ?, ?, ?, ?)",
            (attempt["capture_id"], attempt["ticker"], attempt["created_at"],
             COMPUTED_MOVES_CONTRACT.contract_id, attempt["outcome"]))


def _commit_generation(conn, store, scope, *, parent, records_by_ticker, attempts, clock,
                       expected_head, generation, request_hash):
    prior_manifest = parent.table_manifests.get(COMPUTED_MOVES_TABLE_NAME)
    prior_records = tuple(record for record in parent.records
                          if record.table_contract_ref.contract_id
                          == COMPUTED_MOVES_CONTRACT.contract_id)
    rewritten = set(records_by_ticker)
    kept = tuple(record for record in prior_records if record.partition_key not in rewritten)
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
    return data_catalog.commit_snapshot(
        conn, scope=scope, request_hash=request_hash, contracts=tuple(contracts.values()),
        objects=all_objects, records=all_records, manifests=tuple(table_manifests.values()),
        snapshot=snapshot, expected_head_snapshot_id=expected_head,
        expected_head_generation=generation, receipt_id=receipt_id, attempt_id=attempt_id,
        fence=1, fence_check=lambda connection: None, clock=clock, store=store,
        record_references=lambda connection, rid: _insert_captures(connection, attempts),
        audit_partitions=False)


# --------------------------------------------------------------------------
# the RefreshCallback-compatible entrypoint
# --------------------------------------------------------------------------


def _committed_targets(parent, targets) -> tuple[str, ...]:
    """The wanted tickers whose rows the pinned snapshot already committed.

    A cache-only rerun commits nothing, so its ``completed_ids`` is only the
    units whose rows are already in the parent snapshot's ``computed_moves``
    fragments -- never the whole target list on faith.
    """
    committed = {record.partition_key for record in parent.records
                 if record.table_contract_ref.contract_id
                 == COMPUTED_MOVES_CONTRACT.contract_id}
    return tuple(ticker for ticker in targets if ticker in committed)


def _noop_result(parameters, completed_ids, plan_hash) -> RefreshCallbackResult:
    """A truthful rerun result: nothing committed, nothing advanced."""
    return RefreshCallbackResult(
        status="noop", completed_ids=completed_ids, coverage_advanced=False,
        parent_snapshot_id=parameters.parent_snapshot_id, refresh_plan_hash=plan_hash)


def _input_document(root: Path) -> dict | None:
    path = root / INPUT_PATH
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not document.get("catalog_path") or not document.get("objects_root"):
        return None
    return document


def _failed(parameters) -> RefreshCallbackResult:
    return RefreshCallbackResult(
        status="failed", completed_ids=(), coverage_advanced=False,
        parent_snapshot_id=parameters.parent_snapshot_id,
        refresh_plan_hash=parameters.refresh_plan_hash)


def run_computed_moves_refresh(parameters, root, *, fetcher=None) -> RefreshCallbackResult:
    """The ``RefreshCallback`` this job's worker calls (spec s4b Change 3).

    Reads the executor-staged ``computed_moves_refresh_input.json`` (catalog
    path, objects root, scope, expected head generation), selects targets from
    the pinned parent snapshot with ONE scan of each source table, reserves
    budget through ``incremental_data.plan_refresh`` per ticker, and commits
    one new snapshot carrying every other table forward alongside a fresh
    ``computed_moves`` dataset version. A unit already backed by a durable raw
    receipt is a cache hit: it is never re-fetched. A same-session retry
    rebuilds EVERY unit's fragment -- the fresh fetches plus the cached
    complete receipts re-read by receipt -- so it commits exactly what a clean
    single run would, while a run whose units are all cache-satisfied is a
    no-op with no provider call. Never touches ``INVESTING_PLAN_ROOT`` or any
    legacy path.
    """
    root = Path(root)
    document = _input_document(root)
    if document is None:
        return _failed(parameters)
    if fetcher is None:
        raise fail("RESOURCE_UNAVAILABLE", "no computed_moves history fetcher is configured")
    conn = sqlite3.connect(document["catalog_path"])
    conn.row_factory = sqlite3.Row
    clock = SystemClock()
    try:
        store = ArtifactStore(document["objects_root"])
        repository = Repository(conn, store)
        parent = repository.resolve_full(parameters.parent_snapshot_id)
        events, daily = _scan_once(repository, parent.snapshot)
        as_of = document.get("as_of") or parameters.as_of
        targets, _selection = target_tickers_from_snapshot(
            repository, parameters.parent_snapshot_id,
            all_scoreable=bool(document.get("all_scoreable", True)),
            since=document.get("since"), events=events, daily=daily, as_of=as_of)
        units = computed_moves_units(targets, as_of=as_of)
        plan = plan_refresh(
            parent.snapshot, units,
            cached_outcomes=cached_unit_outcomes(
                conn, units, source=COMPUTED_MOVES_TABLE_NAME,
                endpoint=COMPUTED_MOVES_TABLE_NAME),
            provider_account=NATIVE_COMPUTED_MOVES_ACCOUNT,
            expected_head_generation=int(document["expected_head_generation"]))
        if not plan.fetch_units:
            # A cached receipt is not a committed row: a commit that failed
            # leaves the receipt durable while the fragment is absent. Only a
            # rerun whose wanted rows are ALL already committed is a no-op;
            # anything missing falls through and is rebuilt from cache.
            committed = _committed_targets(parent, targets)
            if len(committed) == len(targets):
                return _noop_result(parameters, committed, plan.plan_hash)
        fragment_records, attempts = _capture_targets(
            conn, store, plan, fetcher, clock,
            events_by_ticker=_group_by_ticker(events),
            daily_by_ticker=_group_by_ticker(daily))
        if not fragment_records:
            return _noop_result(parameters, _committed_targets(parent, targets), plan.plan_hash)

        request_hash = content_hash({
            "kind": "computed_moves_generation", "scope": document["scope"],
            "base_snapshot_id": parent.snapshot.snapshot_id,
            "tickers": sorted(fragment_records)})
        receipt = _commit_generation(
            conn, store, str(document["scope"]), parent=parent,
            records_by_ticker=fragment_records, attempts=attempts, clock=clock,
            expected_head=document.get("expected_head_snapshot_id",
                                       parameters.parent_snapshot_id),
            generation=int(document["expected_head_generation"]), request_hash=request_hash)
        return RefreshCallbackResult(
            status="complete", completed_ids=tuple(sorted(fragment_records)),
            coverage_advanced=True, parent_snapshot_id=parameters.parent_snapshot_id,
            refresh_plan_hash=plan.plan_hash,
            candidate_snapshot_id=receipt.resulting_head_snapshot_id)
    finally:
        conn.close()


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


def _capture_targets(conn, store, plan, fetcher, clock, *, events_by_ticker,
                     daily_by_ticker):
    """Acquire and stage one fragment per unit, fresh or cached, exactly once.

    The caller reaches this only when the plan has at least one fresh fetch, and
    EVERY wanted unit is rebuilt here -- the fresh fetches plus the cached
    ``complete`` receipts re-read and re-parsed by receipt -- so a same-session
    retry commits exactly what a clean single run would. Any unit that ends
    transient/refused/not_final fails the whole job (R3) after every unit has
    been attempted, with the good receipts already cached.
    """
    created_at = clock.now().isoformat()
    cached = cached_unit_payloads(conn, store, plan)
    fragment_records, attempts, kinds = {}, [], []
    for unit in plan.units:
        ticker = unit.expected_keys[0]
        capture_id = "capture_" + content_hash(
            {"ticker": ticker, "created_at_request": created_at}).removeprefix("sha256:")[:32]
        series, kind, fresh = _unit_history(conn, store, unit, fetcher, cached,
                                            created_at=created_at)
        kinds.append(kind)
        if series is None:
            attempts.append({"capture_id": capture_id, "ticker": ticker,
                             "created_at": created_at,
                             "outcome": ("no_history" if kind == "legitimate_empty"
                                         else kind)})
            continue
        events = events_by_ticker.get(ticker)
        if events is None:
            attempts.append({"capture_id": capture_id, "ticker": ticker,
                             "created_at": created_at, "outcome": "too_few"})
            continue
        sd, sc = series
        events = events[events["event_date"] >= pd.Timestamp(sd[0])]
        rows = build_rows(
            ticker, events, sd, sc, daily_by_ticker.get(ticker, _EMPTY_DAILY),
            computed_at=created_at,
            source_hash=hashlib.sha256(np.ascontiguousarray(sc).tobytes()).hexdigest(),
            capture_id=capture_id)
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
