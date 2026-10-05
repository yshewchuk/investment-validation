"""``forward_calendar`` capture -- the natively-owned store for the forward
earnings calendar (spec s4b Change 5).

The legacy ``engine.data.pulls.forward_calendar`` pull fetched Nasdaq and
yfinance directly and then called ``engine.data.rebuild.rebuild(tables=
("events",))`` to merge Tier 2 -- which is exactly the wrapping mistake this
store removes: the job now fetches through the shared provider-budget
admission (``incremental_data.plan_refresh``/``classify_response``), resolves
each claim's session with the SAME priority rule the legacy ``build_calendar``
applies (``engine.calendar.SESSION_PRIORITY``, moved here verbatim as pure
code -- a v2 layer may not import the legacy module), and merges the claims
into the EXISTING ``legacy.earnings_events.v1`` contract through
``generic_incremental`` (``incremental_tables``' own docstring names
"calendar" as a table family meant for these primitives before it gets its own
provider adapter -- this is that adapter).

Spec s4c rewrites two things here: the trading calendar is derived from the
pinned snapshot's own ``daily_market`` sessions (``data.computed_moves.
native_trading_calendar``, extending with ported pure rules) instead of the
legacy ``GSPC_DAILY`` CSV, and both network edges are the injected
``nasdaq_calendar_fetcher``/``yfinance_earnings_fetcher``, whose acquired
bytes are cached as raw receipts so a second same-catalog run makes zero
provider calls and commits exactly the rows its receipts rebuild.

The pure functions are testable without a catalog or a network; the runner is
the impure orchestrator.
"""
from __future__ import annotations

import io
import json
import logging
import numbers
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from engine.v2.contracts import (
    CompletedCoverage,
    CoverageKey,
    CoverageOutcome,
    RevisionCandidate,
    TimeInterval,
)
from engine.v2.data import generic_incremental, incremental_tables
from engine.v2.data.computed_moves import native_trading_calendar
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.ops.errors import fail
from engine.v2.ops.incremental_data import (
    RefreshCallbackResult,
    RefreshUnit,
    plan_refresh,
)
from engine.v2.ops.lifecycle import validated_attempt_fence_pair as _validated_attempt_fence_pair
from engine.v2.ops.lifecycle import verify_fence
from engine.v2.ops.unit_receipts import (
    NATIVE_NASDAQ_ACCOUNT,
    NATIVE_YFINANCE_ACCOUNT,
    cached_unit_outcomes,
    cached_unit_payloads,
    provider_failure_code,
    record_unit_receipt,
)

__all__ = [
    "SESSION_PRIORITY",
    "daily_by_ticker",
    "date_units",
    "horizon_dates",
    "plan_forward_calendar",
    "resolve_session_claims",
    "run_forward_calendar_refresh",
    "ticker_units",
]

TABLE_NAME = "earnings_events"
MAX_SCAN_ROWS = 2_000_000
NASDAQ_SOURCE = "nasdaq"
YFINANCE_SOURCE = "yfinance"

#: No existing forward-calendar or board-universe horizon constant already
#: bounds a maximum; this is a new, deliberately generous ceiling (one year
#: plus a leap day), not a tuned limit.
MAX_HORIZON_DAYS = 366

#: Which source's session wins, best first -- moved verbatim from
#: ``engine.calendar.SESSION_PRIORITY`` (the legacy module stays untouched and
#: is imported by the test as the oracle).
SESSION_PRIORITY = ("orats", "yfinance", "nasdaq")

#: Moved verbatim from ``engine.data.sources.nasdaq.SESSION_BY_TIME``: the two
#: ``time`` strings Nasdaq's forward-calendar rows can carry.
SESSION_BY_TIME = {
    "time-pre-market": "BMO",
    "time-after-hours": "AMC",
}

_LOGGER = logging.getLogger(__name__)


def horizon_dates(as_of, horizon_days: int, *, calendar=None) -> list[pd.Timestamp]:
    """Trading days in ``[as_of, as_of + horizon_days]``.

    Moved from the legacy pull (76-90) with the process-global
    ``trading_calendar()`` lookup made a parameter: the runner injects the
    snapshot-native calendar (``native_trading_calendar``), and a missing
    calendar (``None`` -- the snapshot carries no ``daily_market`` session, a
    condition ``_native_calendar`` catches specifically) falls back to weekdays
    exactly as the legacy function's own ``except`` branch does. Reading the
    calendar's own days is never wrapped in a broad catch: a programming error
    propagates instead of silently degrading the horizon.
    """
    as_of = pd.Timestamp(as_of).normalize()
    end = as_of + pd.Timedelta(days=horizon_days)
    days = [d for d in calendar.days if as_of <= d <= end] if calendar is not None else []
    if not days:
        days = [d for d in pd.date_range(as_of, end, freq="D") if d.weekday() < 5]
    return list(days)


# --------------------------------------------------------------------------
# snapshot reads (the v2 read path, never engine.data.store)
# --------------------------------------------------------------------------


def _scan_rows(repository: Repository, snapshot, table_name: str, columns) -> list[dict]:
    from engine.v2.contracts import DataQuery, KeyPredicate

    contract_ref = snapshot.table_versions[table_name].table_contract_ref
    contract = repository.table_contract(snapshot, table_name)
    years = tuple(sorted({int(record.partition_key)
                          for record in repository.fragment_records(snapshot, table_name)}))
    if not years:
        return []
    key_filter = (KeyPredicate(column="year", operator="in", values=years),)
    population_bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=table_name, table_contract_ref=contract_ref,
        key_filter=key_filter, time_interval=None)
    max_result_rows = min(MAX_SCAN_ROWS, population_bound)
    max_batch_rows = min(contract.maximum_batch_rows, 50_000)
    if max_result_rows > 0:
        max_batch_rows = min(max_batch_rows, max_result_rows)
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=contract_ref,
        columns=tuple(columns),
        key_filter=key_filter,
        order_by=tuple(contract.primary_key),
        max_batch_rows=max_batch_rows,
        max_result_rows=max_result_rows)
    rows: list[dict] = []
    for batch in repository.scan(query, table_name=table_name):
        rows.extend(batch.to_pylist())
    return rows


def daily_by_ticker(repository: Repository, snapshot) -> dict[str, pd.DataFrame]:
    """One scan of the pinned snapshot's ``daily_market``, grouped by ticker.

    This single scan feeds ``native_trading_calendar`` (spec s4c Rewrite 1) --
    there is no second daily_market read and no legacy CSV touch.
    """
    rows = _scan_rows(repository, snapshot, "daily_market", ("ticker", "date"))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    frame["date"] = pd.to_datetime(frame["date"])
    return {str(ticker): group for ticker, group in frame.groupby("ticker")}


def resolve_session_claims(claims: dict) -> tuple[str | None, str | None]:
    """The legacy ``build_calendar`` session tie-break, moved verbatim.

    The legacy loop walks ``SESSION_PRIORITY`` in order and takes the first
    source whose session column is populated; this function is the same loop
    over an already-collected ``{source: session}`` mapping. A clean
    one-for-one port, so a forward claim can never resolve a session
    differently from the legacy ``rebuild`` pipeline.
    """
    session = None
    session_src = None
    for name in SESSION_PRIORITY:
        value = claims.get(name)
        if session is None and value is not None and not pd.isna(value):
            session, session_src = value, name
    return session, session_src


def _as_of_day(as_of) -> str:
    """The job's as_of date as ``YYYY-MM-DD``, strictly validated (R4/R5).

    Mirrors ``engine.v2.ops.native_board_universe._validated_as_of`` (PR #16,
    not yet on ``main``, so mirrored here rather than imported): refuses
    ``None``, a bare number/``bool`` (which would misread as epoch time), an
    unparseable value, ``NaT``, and a timezone-aware value -- this job only
    ever deals in a timezone-naive calendar date.
    """
    if as_of is None:
        raise fail("INVALID_REQUEST", "as_of must not be None")
    if isinstance(as_of, bool) or isinstance(as_of, (numbers.Number, np.number)):
        raise fail("INVALID_REQUEST",
                   f"as_of must be a date/timestamp, not a bare number ({as_of!r}); a "
                   f"numeric value would be misread as epoch time rather than a calendar date")
    try:
        day = pd.Timestamp(as_of)
    except (TypeError, ValueError) as exc:
        raise fail("INVALID_REQUEST",
                   f"as_of could not be parsed as a timestamp: {exc}") from exc
    if pd.isna(day):
        raise fail("INVALID_REQUEST", "as_of must not be NaT")
    day = day.normalize()
    if day.tzinfo is not None:
        raise fail("INVALID_REQUEST",
                   "as_of is timezone-aware; forward_calendar only supports a "
                   "timezone-naive as_of")
    return str(day.date())


def _validated_tickers(tickers) -> tuple[str, ...]:
    """``tickers`` must be a non-string iterable of non-empty ``str``.

    A bare ``str`` is refused outright (rather than silently iterated
    character-by-character, e.g. ``set("AAPL")``); any other non-iterable is
    refused the same way; every element must be a non-empty ``str``.
    """
    if tickers is None or isinstance(tickers, (str, bytes)):
        raise fail("INVALID_REQUEST",
                   f"tickers must be a non-string iterable of ticker symbols, "
                   f"got {tickers!r}")
    try:
        items = list(tickers)
    except TypeError as exc:
        raise fail("INVALID_REQUEST",
                   f"tickers must be an iterable of ticker symbols, got {tickers!r}") from exc
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise fail("INVALID_REQUEST",
                       f"every ticker must be a non-empty str, got {item!r}")
    return tuple(items)


def _validated_horizon_days(horizon_days) -> int:
    """``horizon_days`` must be a real ``int`` (not ``bool``) in
    ``[1, MAX_HORIZON_DAYS]``."""
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int):
        raise fail("INVALID_REQUEST",
                   f"horizon_days must be an int, got {horizon_days!r}")
    if not 1 <= horizon_days <= MAX_HORIZON_DAYS:
        raise fail("INVALID_REQUEST",
                   f"horizon_days must be between 1 and {MAX_HORIZON_DAYS}, "
                   f"got {horizon_days!r}")
    return horizon_days


#: The only two commit-destination namespaces v2 ops ever authorizes for a
#: job like this one -- the same pair every ``JobKind`` in ``stages.py``/
#: ``cli.py``/``incremental_data.refresh_job_kind`` registers as
#: ``namespaces=frozenset({"shadow", "smoke"})``. Reused, not redefined: no
#: single module exports this pair as an importable name, so this is a local
#: copy of the same two values, not a new list.
_VALID_SCOPES = frozenset({"shadow", "smoke"})


def _validated_scope(scope) -> str:
    """``scope`` is required and must be one of ``_VALID_SCOPES``, refused
    before any I/O."""
    if not isinstance(scope, str) or scope not in _VALID_SCOPES:
        raise fail("INVALID_REQUEST",
                   f"scope must be one of {sorted(_VALID_SCOPES)}, got {scope!r}")
    return scope


def _validated_expected_head_generation(value) -> int:
    """``expected_head_generation`` is required: a non-negative ``int``,
    refused before any I/O."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise fail("INVALID_REQUEST",
                   f"expected_head_generation is required and must be an int, "
                   f"got {value!r}")
    if value < 0:
        raise fail("INVALID_REQUEST",
                   f"expected_head_generation must not be negative, got {value!r}")
    return value


def _validated_expected_head_snapshot_id(value) -> str | None:
    """Optional; when present, the same bounded nonempty string shape as
    ``parent_snapshot_id`` (1..128 chars) -- refused before any I/O, never
    read straight into ``_commit_claims``/``generic_incremental`` unchecked."""
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 128:
        raise fail("INVALID_REQUEST",
                   f"expected_head_snapshot_id must be a bounded nonempty str or None, "
                   f"got {value!r}")
    return value


def _validated_attempt_id(value) -> str | None:
    """Optional; when present, a non-empty ``str`` -- the staged job attempt
    this commit is fenced to (issue #52). ``None`` is a valid, meaningful
    request: a manual/ad-hoc invocation with no live job behind it, the same
    contract ``computed_moves_store``'s own optional ``attempt_id``/``fence``
    document fields have."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise fail("INVALID_REQUEST",
                   f"attempt_id must be a non-empty str or None, got {value!r}")
    return value


def _validated_fence(value) -> int | None:
    """Optional; when present, an ``int >= 1`` (never a ``bool``, which is
    an ``int`` subclass in Python)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise fail("INVALID_REQUEST",
                   f"fence must be an int >= 1 or None, got {value!r}")
    return value


def _validated_catalog_path(catalog_path) -> str:
    """``catalog_path`` must already exist as a file -- ``sqlite3.connect``
    is never allowed to silently create one that does not."""
    if not isinstance(catalog_path, str) or not catalog_path:
        raise fail("INVALID_REQUEST",
                   f"catalog_path must be a non-empty str, got {catalog_path!r}")
    if not Path(catalog_path).is_file():
        raise fail("INVALID_REQUEST",
                   f"catalog_path does not exist as a file: {catalog_path!r}")
    return catalog_path


def _validated_objects_root(objects_root) -> str:
    """``objects_root`` must already exist as a directory."""
    if not isinstance(objects_root, str) or not objects_root:
        raise fail("INVALID_REQUEST",
                   f"objects_root must be a non-empty str, got {objects_root!r}")
    if not Path(objects_root).is_dir():
        raise fail("INVALID_REQUEST",
                   f"objects_root does not exist as a directory: {objects_root!r}")
    return objects_root


def _validated_parent_snapshot_id(parent_snapshot_id) -> str:
    """Matches ``incremental_data``'s own ``RefreshParameters`` rule for this
    same field: a bounded nonempty string (a parent snapshot id is not
    restricted to one minted shape -- a bootstrap/root reference is
    legitimate too)."""
    if (not isinstance(parent_snapshot_id, str) or not parent_snapshot_id
            or len(parent_snapshot_id) > 128):
        raise fail("INVALID_REQUEST",
                   f"parent_snapshot_id must be a bounded nonempty str, "
                   f"got {parent_snapshot_id!r}")
    return parent_snapshot_id


def _validated_refresh_plan_hash(refresh_plan_hash) -> str:
    """Matches ``incremental_data._is_hash``'s own sha256-hex check for this
    same field (mirrored rather than imported: that name is private)."""
    valid = (isinstance(refresh_plan_hash, str)
             and refresh_plan_hash.startswith("sha256:") and len(refresh_plan_hash) == 71
             and all(char in "0123456789abcdef" for char in refresh_plan_hash[7:]))
    if not valid:
        raise fail("INVALID_REQUEST",
                   f"refresh_plan_hash must be a sha256 content hash, "
                   f"got {refresh_plan_hash!r}")
    return refresh_plan_hash


def date_units(dates, *, as_of) -> tuple[RefreshUnit, ...]:
    """One Nasdaq discovery unit per date; the as_of is part of every id (R4)."""
    day = _as_of_day(as_of)
    return tuple(RefreshUnit(
        request_id="nasdaq:calendar/earnings:" + str(pd.Timestamp(item).date()) + ":" + day,
        table_name=TABLE_NAME, partition_key=str(pd.Timestamp(item).date()),
        expected_keys=(str(pd.Timestamp(item).date()),)) for item in dates)


def ticker_units(tickers, *, as_of) -> tuple[RefreshUnit, ...]:
    """One yfinance confirmation unit per ticker; as_of is in the id (R4)."""
    day = _as_of_day(as_of)
    return tuple(RefreshUnit(
        request_id="yfinance:earnings:" + str(ticker) + ":" + day, table_name=TABLE_NAME,
        partition_key=str(ticker), expected_keys=(str(ticker),)) for ticker in tickers)


def plan_forward_calendar(parent_snapshot, dates, tickers, *, as_of, cached_nasdaq=None,
                          cached_yfinance=None, max_attempts: int = 3,
                          expected_head_generation: int = 0):
    """The two cache-first RefreshPlans this job reserves budget for.

    One plan per source, exactly the ``plan_refresh`` formula the daily
    refresh already uses: ``provider_calls == len(fetch_units) *
    max_attempts``, cache-satisfied units never reserve a call. Returned as a
    pair because a ``RefreshPlan`` names a single provider account; the
    runner executes the Nasdaq plan first and builds the yfinance plan from
    the tickers still missing a session.
    """
    nasdaq = plan_refresh(parent_snapshot, date_units(dates, as_of=as_of),
                          cached_outcomes=dict(cached_nasdaq or {}),
                          provider_account=NATIVE_NASDAQ_ACCOUNT, max_attempts=max_attempts,
                          expected_head_generation=expected_head_generation)
    yfinance = plan_refresh(parent_snapshot, ticker_units(tickers, as_of=as_of),
                            cached_outcomes=dict(cached_yfinance or {}),
                            provider_account=NATIVE_YFINANCE_ACCOUNT, max_attempts=max_attempts,
                            expected_head_generation=expected_head_generation)
    return nasdaq, yfinance


# --------------------------------------------------------------------------
# claim collection through the shared admission
# --------------------------------------------------------------------------


def _edge_result(fetcher, argument) -> tuple[bytes, str, dict, list]:
    """Call one provider edge; only a network failure is a classified transient.

    The edges classify their own provider responses (R1); a programming error
    or an ``OpsError`` out of a raising edge propagates (R3), it never becomes
    a silent outage.
    """
    try:
        raw, kind, meta, rows = fetcher(argument)
    except (OSError, TimeoutError) as exc:
        return b"", "transient", {"error": type(exc).__name__}, []
    if not isinstance(kind, str):
        return b"", "refused", {}, []
    return (raw or b""), kind, dict(meta or {}), list(rows or ())


def _parseable_json(raw: bytes) -> bool:
    """The store re-validates bytes before caching them (spec R2)."""
    try:
        json.loads(raw.decode("utf-8"))
    except (AttributeError, TypeError, UnicodeDecodeError, ValueError):
        return False
    return True


def _parse_earnings(raw: bytes):
    try:
        return pd.read_csv(io.BytesIO(raw))
    except (ValueError, OSError, AttributeError):
        return None


def nasdaq_rows_from_payload(raw: bytes) -> list[dict]:
    """The provider's own ``data.rows`` parser, over a cached receipt's bytes."""
    from engine.v2.ops.providers.nasdaq_calendar import _rows_field

    try:
        document = json.loads(raw.decode("utf-8"))
    except (AttributeError, TypeError, UnicodeDecodeError, ValueError):
        return []
    return _rows_field(document) or []


def nasdaq_claims_from_rows(claims: dict, day: str, rows, wanted) -> None:
    """Fold one Nasdaq date's rows into ``claims`` in place.

    ``claims`` maps ``(ticker, day) -> {"nasdaq": session_or_None}``. Shared by
    ``_fetch_nasdaq`` (fresh-or-cached fetch path) and the nightly planner's
    zero-fetch path, so the two can never drift on the symbol strip, the
    wanted-ticker filter, or the ``SESSION_BY_TIME`` lookup (spec R6: the
    reserved ``provider_calls`` must cover every yfinance fetch the store
    makes, so both call sites must agree on which tickers are "pending").
    """
    for row in rows:
        ticker = str(row.get("symbol") or "").strip()
        if not ticker or (wanted and ticker not in wanted):
            continue
        claims.setdefault((ticker, day), {})["nasdaq"] = SESSION_BY_TIME.get(row.get("time"))


def pending_tickers(claims: dict) -> list[str]:
    """Tickers whose Nasdaq claim never resolved a session.

    These still need a yfinance confirmation pass.
    """
    return sorted({ticker for (ticker, _day), sources in claims.items()
                  if not sources.get("nasdaq")})


def _nasdaq_unit(conn, store, fetcher, unit, cached, *, received_at: str):
    """One Nasdaq date's rows: a fresh fetch or the cached receipt's bytes.

    Returns ``(kind, rows)``; a fresh parseable payload is recorded as a
    receipt, so it survives a later unit's failure in the same run (R3).
    """
    from engine.v2.ops.unit_receipts import _unit_request

    fresh = unit.request_id not in cached
    if fresh:
        raw, kind, _meta, rows = _edge_result(fetcher, _unit_request(unit))
    else:
        raw, kind = cached[unit.request_id], "complete"
        rows = nasdaq_rows_from_payload(raw)
    if kind in ("complete", "legitimate_empty"):
        if not raw or not _parseable_json(raw):
            kind = "refused"  # unparseable bytes are never cached as complete
        elif fresh:
            record_unit_receipt(conn, store, unit, raw, source=NASDAQ_SOURCE,
                                endpoint="calendar/earnings", received_at=received_at,
                                response_kind=kind)
    return kind, (rows if kind == "complete" else [])


def _fetch_nasdaq(conn, store, fetcher, plan, wanted: set[str], *, received_at: str):
    """Every Nasdaq date unit's claims, fresh and cached receipt alike.

    A same-session retry re-reads the 20 good units' cached bytes and fetches
    only the failed one, yet still builds the claims a clean run would (spec
    s4c round 3). Returns ``(claims, kinds)``.
    """
    cached = cached_unit_payloads(conn, store, plan)
    claims: dict[tuple[str, str], dict] = {}
    kinds: list[str] = []
    for unit in plan.units:
        day = unit.expected_keys[0]
        kind, rows = _nasdaq_unit(conn, store, fetcher, unit, cached,
                                  received_at=received_at)
        kinds.append(kind)
        if kind != "complete":
            continue
        nasdaq_claims_from_rows(claims, day, rows, wanted)
    return claims, kinds


def _yfinance_unit(conn, store, fetcher, unit, cached, *, received_at: str):
    """One ticker's earnings frame: a fresh fetch or the cached receipt's bytes."""
    fresh = unit.request_id not in cached
    if fresh:
        raw, kind, _meta, _rows = _edge_result(fetcher, unit.expected_keys[0])
    else:
        raw, kind = cached[unit.request_id], "complete"
    frame = (_parse_earnings(raw)
             if raw and kind in ("complete", "legitimate_empty") else None)
    if kind == "complete" and frame is None:
        kind = "refused"
    if frame is not None and kind in ("complete", "legitimate_empty") and fresh:
        record_unit_receipt(conn, store, unit, raw, source=YFINANCE_SOURCE,
                            endpoint="earnings", received_at=received_at,
                            response_kind=kind)
    return frame, kind


def _fetch_yfinance(conn, store, fetcher, plan, claims: dict, *, received_at: str) -> list[str]:
    """Warm and read the per-ticker yfinance earnings CSV.

    The legacy pull only warmed this cache here and left the session parse to
    ``engine.calendar.load_yfinance_earnings`` at rebuild time; this store is
    that read, moved into the job that owns the fetch (same CSV columns, same
    ``session`` mapping). Every unit in the plan is applied -- a fresh fetch or
    the cached complete receipt re-read by receipt -- so a same-session retry
    resolves the same claims a clean run would. Bytes are parsed BEFORE they
    are cached (R2); the unit kinds are returned so the caller can fail the job
    on any transient/refused/not_final unit (R3).
    """
    cached = cached_unit_payloads(conn, store, plan)
    kinds: list[str] = []
    for unit in plan.units:
        ticker = unit.expected_keys[0]
        frame, kind = _yfinance_unit(conn, store, fetcher, unit, cached,
                                     received_at=received_at)
        kinds.append(kind)
        if frame is None or frame.empty or "session" not in frame.columns:
            continue
        for key, claim in claims.items():
            if key[0] != ticker:
                continue
            match = frame[frame["event_date"].astype(str).str[:10] == key[1]]
            if not match.empty:
                claim["yfinance"] = match.iloc[-1]["session"]
    return kinds


# --------------------------------------------------------------------------
# merge into the existing earnings_events contract and commit
# --------------------------------------------------------------------------


def _existing_index(repository, snapshot) -> dict[tuple[str, str], dict]:
    """Every existing ``earnings_events`` row, keyed ``(ticker, event_date)``."""
    contract = next(item for item in snapshot.contracts if item.table_name == TABLE_NAME)
    rows = {}
    for row in _scan_rows(repository, snapshot.snapshot, TABLE_NAME,
                          tuple(column.name for column in contract.columns)):
        rows[(str(row["ticker"]), str(row["event_date"])[:10])] = row
    return rows


def _merged_row(existing, ticker: str, day: str, claims: dict, *, updated_at: str) -> dict:
    row = dict(existing) if existing else {
        "event_id": f"{ticker}_{day}", "ticker": ticker,
        "event_date": pd.Timestamp(day).to_pydatetime(), "year": int(day[:4]),
        "annc_tod": None, "session": None, "session_src": None,
        "src_orats": False, "src_oquants": False, "src_nasdaq": False,
        "src_yfinance": False, "date_agree": False, "date_conflict": False,
        "updated_at": updated_at, "event_cluster_id": f"{ticker}_{day}",
        "claim_count": 1, "reconciliation": "unreconciled",
    }
    attributed = {}
    if row.get("session") is not None and row.get("session_src"):
        attributed[str(row["session_src"])] = row["session"]
    for source in ("nasdaq", "yfinance"):
        if source in claims:
            row[f"src_{source}"] = True
        if claims.get(source):
            attributed[source] = claims[source]
    session, session_src = resolve_session_claims(attributed)
    row["session"] = session
    row["session_src"] = session_src
    row["date_agree"] = bool(row.get("src_orats")) and bool(row.get("src_oquants"))
    row["updated_at"] = updated_at
    return row


def _revision(contract, row: dict, *, received_at: str) -> incremental_tables.GenericRevision:
    logical_key = incremental_tables.logical_key_for_row(contract, row)
    payload = incremental_tables.revision_hash(logical_key=logical_key, row=row, deleted=False)
    # Microsecond resolution: a later session's correction must outrank the same
    # key's retained revision even when both commits happen in the same second
    # (second-resolution ordinals tie and the equal-rank guard then refuses).
    candidate = RevisionCandidate(
        revision_id="fwd_cal_" + payload.removeprefix("sha256:")[:32],
        logical_key=logical_key, source="forward_calendar", source_priority=0,
        finality="final",
        revision_ordinal=int(pd.Timestamp(received_at).timestamp() * 1_000_000),
        received_at=received_at, content_hash=payload)
    return incremental_tables.GenericRevision(candidate=candidate, row=row, deleted=False)


def _coverage(contract, revision_ids, tickers, *, day_range, created_at: str) -> CompletedCoverage:
    from engine.v2.contracts import TableContractRef
    ref = TableContractRef(contract_id=contract.contract_id,
                           definition_hash=contract.definition_hash)
    expected = tuple(CoverageKey(
        item_key=ticker + "|" + day, session_date=day, ticker=ticker,
        contract_id=contract.contract_id) for ticker, day in sorted(revision_ids))
    outcomes = tuple(CoverageOutcome(
        key=key, status="present", receipt_id="forward_calendar", revision_id=None,
        finality="final") for key in expected)
    identity = {"kind": "completed_coverage_id.v1", "source": "forward_calendar",
                "expected": [item.item_key for item in expected]}
    return CompletedCoverage(
        coverage_id="cov_" + content_hash(identity).removeprefix("sha256:")[:32],
        table_contract_ref=ref, source="forward_calendar", endpoint="calendar/earnings",
        interval=TimeInterval(column="event_date", start_inclusive=day_range[0],
                              end_exclusive=None),
        expected=expected, outcomes=outcomes, covered_tickers=tuple(sorted(set(tickers))),
        acquisition_receipt_refs=(), state="complete", completed_at=created_at)


def _fence_check_for(staged_attempt_id, staged_fence, clock):
    """The commit's own fence check against
    ``engine.v2.ops.lifecycle.verify_fence``'s REAL signature (``conn,
    attempt_id, fence, now`` -- it has no ``check_lease_time`` parameter to
    disable the wall-clock lease-expiry check with). Never disables that
    check: the production lease-expiry gate stays active here, the same
    contract ``computed_moves_store``'s own ``_fence_check_for`` has (issue
    #52). ``None`` staged_attempt_id (no live job behind this call, e.g. a
    manual or test invocation) is a no-op fence check.
    """
    if staged_attempt_id is None:
        return lambda connection: None
    return lambda connection: verify_fence(connection, staged_attempt_id, staged_fence,
                                           clock.now())


def _commit_claims(conn, store, parent, claims: dict, existing: dict, *, scope, clock,
                   expected_head_generation: int, expected_head_snapshot_id: str | None,
                   attempt_id: str | None = None, fence: int | None = None):
    """Merge ``claims`` into the existing ``earnings_events`` contract and
    commit one new snapshot generation through ``generic_incremental``.

    ``attempt_id``/``fence`` (issue #52), when not ``None``, fence this
    commit to a live job attempt via ``_fence_check_for``: a cancelled job
    or an expired lease is refused inside the same transaction the head
    compare-and-swap runs in, before any row is inserted. Returns ``None``
    when ``claims`` is empty (nothing to merge, nothing committed).
    """
    contract = next(item for item in parent.contracts if item.table_name == TABLE_NAME)
    received_at = clock.now().isoformat()
    revisions = []
    for (ticker, day), sources in sorted(claims.items()):
        row = _merged_row(existing.get((ticker, day)), ticker, day, sources,
                          updated_at=received_at)
        revisions.append(_revision(contract, row, received_at=received_at))
    if not revisions:
        return None
    coverage = _coverage(contract, list(claims), [ticker for ticker, _ in claims],
                         day_range=(min(day for _, day in claims),), created_at=received_at)
    retained = generic_incremental.load_generic_revisions(conn, TABLE_NAME, contract)
    candidate = generic_incremental.build_generic_table_candidate(
        parent, store, TABLE_NAME, tuple(revisions), coverage=coverage, retained=retained,
        parent_snapshot_id=parent.snapshot.snapshot_id)
    return generic_incremental.commit_generic_table_candidate(
        conn, store, candidate, scope=scope,
        expected_head_snapshot_id=expected_head_snapshot_id or parent.snapshot.snapshot_id,
        expected_head_generation=expected_head_generation, clock=clock,
        request_hash=content_hash({"kind": "forward_calendar_generation", "scope": scope,
                                   "base": parent.snapshot.snapshot_id,
                                   "revisions": sorted(r.candidate.revision_id
                                                       for r in revisions)}),
        fence_check=_fence_check_for(attempt_id, fence, clock))


# --------------------------------------------------------------------------
# the standalone runner (NOT RefreshCallback-shaped -- see module docstring)
# --------------------------------------------------------------------------


def _cached_nasdaq(conn, units) -> dict:
    return cached_unit_outcomes(conn, units, source=NASDAQ_SOURCE, endpoint="calendar/earnings")


def _cached_yfinance(conn, units) -> dict:
    return cached_unit_outcomes(conn, units, source=YFINANCE_SOURCE, endpoint="earnings")


def _native_calendar(repository, parent, *, as_of, horizon_days) \
        -> tuple[object | None, tuple[str, ...]]:
    """The pinned snapshot's own trading calendar, or ``None`` for the fallback.

    ``native_trading_calendar`` raises ``ValueError`` when the snapshot carries
    no ``daily_market`` session; that one specific error selects the weekday
    fallback. The error text is returned as a job-result warning so the
    diagnostics in the evidence name the fallback instead of the horizon
    silently changing shape. ``as_of``/``horizon_days`` compute this run's
    actual requested horizon end, passed through as ``horizon_end`` so a
    stale panel or a wide horizon can never silently truncate the returned
    calendar's date range (the projection always extends through the greater
    of the default 400-day window and the requested horizon). The
    ``try``/``except`` below wraps ONLY the ``native_trading_calendar`` call:
    a ``ValueError`` from the calendar scan or the horizon arithmetic before
    it is a real bug, not the documented no-session fallback, and must not be
    swallowed into it.
    """
    horizon_end = pd.Timestamp(as_of).normalize() + pd.Timedelta(days=int(horizon_days))
    daily = daily_by_ticker(repository, parent.snapshot)
    try:
        return native_trading_calendar(daily, horizon_end=horizon_end), ()
    except ValueError as exc:
        warning = f"weekday calendar fallback: {exc}"
        _LOGGER.warning("forward_calendar: %s", warning)
        return None, (warning,)


def _result(*, parent_snapshot_id: str, refresh_plan_hash: str, status: str, completed_ids,
            coverage_advanced: bool, warnings, candidate_snapshot_id=None) \
        -> RefreshCallbackResult:
    """One job-result document; ``warnings`` carries any degradation evidence."""
    return RefreshCallbackResult(
        status=status, completed_ids=completed_ids, coverage_advanced=coverage_advanced,
        parent_snapshot_id=parent_snapshot_id, refresh_plan_hash=refresh_plan_hash,
        candidate_snapshot_id=candidate_snapshot_id, warnings=warnings)


def run_forward_calendar_refresh(*, catalog_path: str, objects_root: str,
                                 parent_snapshot_id: str, refresh_plan_hash: str,
                                 as_of=None, tickers=None, horizon_days=None,
                                 scope=None, expected_head_generation=None,
                                 expected_head_snapshot_id: str | None = None,
                                 attempt_id: str | None = None, fence: int | None = None,
                                 nasdaq_fetcher=None, earnings_fetcher=None) \
        -> RefreshCallbackResult:
    """The forward-calendar refresh's standalone runner (spec s4b Change 5).

    NOT ``RefreshCallback``-shaped: ``main``'s ``RefreshParameters`` has no
    ``as_of``/``tickers`` field (only a later job-kind-specific parameters
    dataclass will), so this runner takes every value it needs as an
    explicit keyword argument instead, and validates every one of them up
    front -- before the catalog connection opens, before the artifact store
    is constructed, and before any provider call or receipt write -- raising
    a typed ``INVALID_REQUEST`` for each: ``catalog_path`` must already exist
    as a file (``sqlite3.connect`` is never allowed to silently create one),
    ``objects_root`` must already exist as a directory,
    ``parent_snapshot_id``/``refresh_plan_hash`` are checked against the same
    formats ``incremental_data.RefreshParameters`` already uses for these
    fields, and ``as_of``/``tickers``/``horizon_days``/``scope``/
    ``expected_head_generation`` are the request-shaped fields below. A
    later PR wires a ``RefreshCallback``-shaped adapter that reads a job's
    staged parameters/input document and calls this runner with explicit
    keyword arguments.

    ``tickers=()`` means the whole market (my call, since the nightly
    refreshes everything): an empty ``wanted`` set never filters a Nasdaq
    date's rows (``nasdaq_claims_from_rows``), so every ticker Nasdaq reports
    for the requested dates is claimed. The yfinance fan-out is bounded by
    that Nasdaq result for the horizon, not by ``tickers``: only tickers
    Nasdaq left without a resolved session (``pending_tickers``) ever reach
    yfinance, whether ``tickers`` named a handful of symbols or the whole
    market.

    Never calls ``engine.data.rebuild.rebuild``: the claims merge into the
    existing ``earnings_events`` contract through ``generic_incremental``
    against the pinned parent snapshot. The horizon calendar comes from that
    snapshot's own ``daily_market`` dates (a snapshot without one records the
    weekday fallback as a result warning); each source's units are planned
    cache-first, so a unit backed by a durable complete receipt never reaches
    the provider again (spec R2), and any unit that ends
    transient/refused/not_final fails the job with its typed code before
    anything is committed (spec R3). A same-session retry rebuilds EVERY unit's
    claims -- fresh fetches plus the cached complete receipts re-read by
    receipt -- so it commits exactly what a clean single run would. Whether the
    result is a no-op is decided only by the commit itself (see
    ``_execute_forward_calendar_refresh``): a candidate whose merged rows equal
    the parent's resolves back to the parent snapshot, and key presence in the
    parent is never mistaken for this run's content.

    ``attempt_id``/``fence`` (issue #52): ``None``/``None`` is a no-op fence
    check; otherwise the commit is fenced via
    ``engine.v2.ops.lifecycle.verify_fence`` before anything commits. Exactly
    one set (Opus gate finding) is refused up front, before any I/O.
    """
    catalog_path = _validated_catalog_path(catalog_path)
    objects_root = _validated_objects_root(objects_root)
    parent_snapshot_id = _validated_parent_snapshot_id(parent_snapshot_id)
    refresh_plan_hash = _validated_refresh_plan_hash(refresh_plan_hash)
    as_of = _as_of_day(as_of)
    tickers = _validated_tickers(tickers)
    horizon_days = _validated_horizon_days(horizon_days)
    scope = _validated_scope(scope)
    expected_head_generation = _validated_expected_head_generation(expected_head_generation)
    expected_head_snapshot_id = _validated_expected_head_snapshot_id(expected_head_snapshot_id)
    attempt_id = _validated_attempt_id(attempt_id)
    fence = _validated_fence(fence)
    _validated_attempt_fence_pair(attempt_id, fence)
    if nasdaq_fetcher is None or earnings_fetcher is None:
        raise fail("RESOURCE_UNAVAILABLE", "no forward_calendar fetchers are configured")
    return _execute_forward_calendar_refresh(
        catalog_path=catalog_path, objects_root=objects_root,
        parent_snapshot_id=parent_snapshot_id, refresh_plan_hash=refresh_plan_hash,
        as_of=as_of, tickers=tickers, horizon_days=horizon_days, scope=scope,
        expected_head_generation=expected_head_generation,
        expected_head_snapshot_id=expected_head_snapshot_id,
        attempt_id=attempt_id, fence=fence,
        nasdaq_fetcher=nasdaq_fetcher, earnings_fetcher=earnings_fetcher)


def _execute_forward_calendar_refresh(*, catalog_path: str, objects_root: str,
                                      parent_snapshot_id: str, refresh_plan_hash: str,
                                      as_of: str, tickers: tuple[str, ...], horizon_days: int,
                                      scope: str, expected_head_generation: int,
                                      expected_head_snapshot_id: str | None,
                                      attempt_id: str | None = None, fence: int | None = None,
                                      nasdaq_fetcher, earnings_fetcher) \
        -> RefreshCallbackResult:
    """Every argument already validated -- opens the catalog and runs the
    two-source cache-first fetch, merge, and commit (see
    ``run_forward_calendar_refresh``'s docstring for the full contract)."""
    conn = sqlite3.connect(catalog_path)
    conn.row_factory = sqlite3.Row
    clock = SystemClock()
    try:
        store = ArtifactStore(objects_root)
        repository = Repository(conn, store)
        parent = repository.resolve_full(parent_snapshot_id)
        wanted = set(tickers)
        calendar, warnings = _native_calendar(repository, parent, as_of=as_of,
                                              horizon_days=horizon_days)
        dates = horizon_dates(as_of, horizon_days, calendar=calendar)
        received_at = clock.now().isoformat()

        nasdaq_units = date_units(dates, as_of=as_of)
        nasdaq_plan, _ = plan_forward_calendar(
            parent.snapshot, dates, (), as_of=as_of,
            cached_nasdaq=_cached_nasdaq(conn, nasdaq_units))
        claims, nasdaq_kinds = _fetch_nasdaq(conn, store, nasdaq_fetcher, nasdaq_plan, wanted,
                                             received_at=received_at)
        pending = pending_tickers(claims)
        yfinance_units = ticker_units(pending, as_of=as_of)
        _, yfinance_plan = plan_forward_calendar(
            parent.snapshot, (), pending, as_of=as_of,
            cached_yfinance=_cached_yfinance(conn, yfinance_units))
        yfinance_kinds = _fetch_yfinance(conn, store, earnings_fetcher, yfinance_plan, claims,
                                         received_at=received_at)
        code = provider_failure_code((*nasdaq_kinds, *yfinance_kinds))
        if code:
            raise fail(code, "forward calendar provider response was not complete")
        if not claims:
            return _result(parent_snapshot_id=parent_snapshot_id,
                           refresh_plan_hash=refresh_plan_hash, status="noop",
                           completed_ids=tuple(sorted(wanted)),
                           coverage_advanced=False, warnings=warnings)
        existing = _existing_index(repository, parent)
        receipt = _commit_claims(conn, store, parent, claims, existing,
                                 scope=scope, clock=clock,
                                 expected_head_generation=expected_head_generation,
                                 expected_head_snapshot_id=expected_head_snapshot_id,
                                 attempt_id=attempt_id, fence=fence)
        if receipt is None or receipt.resulting_head_snapshot_id == parent.snapshot.snapshot_id:
            # A candidate whose merged rows equal the parent's resolves back to
            # the parent snapshot, so the head did not move and nothing was
            # committed. Key presence in the parent is never consulted -- a
            # cached receipt or a foreign source's row is not this run's
            # committed content.
            return _result(parent_snapshot_id=parent_snapshot_id,
                           refresh_plan_hash=refresh_plan_hash, status="noop",
                           completed_ids=tuple(sorted(wanted)),
                           coverage_advanced=False, warnings=warnings)
        return _result(parent_snapshot_id=parent_snapshot_id,
                       refresh_plan_hash=refresh_plan_hash, status="complete",
                       completed_ids=tuple(sorted(wanted)),
                       coverage_advanced=True, warnings=warnings,
                       candidate_snapshot_id=receipt.resulting_head_snapshot_id)
    finally:
        conn.close()
