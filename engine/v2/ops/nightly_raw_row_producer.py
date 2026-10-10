"""Read-only native raw-row composition for cutover PR-6 slice 4b.

``build_native_score_batch_events`` enumerates the forward board through
``nightly_raw_rows.scan_forward_board_requests`` -- the slice's only
enumeration, so request order and each original ``BoardRequest`` timestamp
identity survive -- classifies every intraday key before any context read, reads
one build-scoped decision calendar plus one panel row/anchor per distinct
``(ticker, event_date, session)``, then composes calendar and quote rows per
request. The calendar helper still refuses an intraday event date, so this
producer is the admission boundary #243 requires: such a key is refused under
its exact, never-normalized identity. Nothing scores, publishes or writes; only
a complete calendar-and-quote composition becomes an event, and a failure
propagates instead of returning a partial tuple. Like ``native_board_universe``,
this module never imports ``engine.score``/``engine.structures``/
``engine.replay``/``engine.fills``.
"""
from __future__ import annotations

import json
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from typing import Any

import pandas as pd

from engine.v2.contracts import SnapshotRef
from engine.v2.data.errors import DataError, fail
from engine.v2.data.price_history_query import tickers_with_price_history
from engine.v2.data.price_history_table import PRICE_HISTORY_TABLE_NAME
from engine.v2.data.repository import Repository
from engine.v2.features.panel_row_inputs import PanelRowInputs, scan_panel_row
from engine.v2.ops.errors import OpsError
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_score_batch_types import NativeScoreBatchRowRefusal
from engine.v2.ops.nightly_calendar_inputs import (
    scan_calendar_row_inputs,
    scan_decision_calendar,
)
from engine.v2.ops.nightly_quote_rows import scan_quote_rows
from engine.v2.ops.nightly_raw_rows import scan_forward_board_requests
from engine.v2.scoring.nightly_source_bundle import validated_as_of

__all__ = ["build_native_score_batch_events"]

#: The refusal sidecar's schema, the one ``producer_refusals.json``'s decoder takes.
_REFUSAL_SCHEMA_VERSION = "native_score_batch_producer_refusals.v1.0"
_INTRADAY_CODE = "INTRADAY_EVENT_NOT_ADMITTED"
#: Fixed public-safe detail text (CWE-209): never exception text or a key value.
_INTRADAY_DETAIL = "the board request carries an intraday event timestamp"
_PANEL_HISTORY_DETAIL = "the pinned snapshot lacks required earlier panel sessions"
_PRICE_HISTORY_CODE = "PRICE_HISTORY_NOT_AVAILABLE"
_PRICE_HISTORY_DETAIL = "the pinned snapshot has no price history for this ticker"
_ROW_REFUSAL_DETAILS = {
    "NO_RESOLVABLE_EXPIRY": "no strategy-eligible listed expiry for the board request",
    "EVENT_NOT_FOUND": "no exact calendar event for the board request in the snapshot",
    "IDENTITY_CONFLICT": "multiple exact calendar events for the board request",
}
#: The only event refusals the calendar helper may expose as a typed error.
_EVENT_ERROR_CODES = ("EVENT_NOT_FOUND", "IDENTITY_CONFLICT")


def _event_day(value: Any) -> str:
    """One already-classified midnight event timestamp as ``YYYY-MM-DD``."""
    return pd.Timestamp(value).date().isoformat()


def _panel_marker(key: BoardRequest) -> tuple[str, str, str]:
    """The ``(ticker, event day, session)`` identity a panel row is shared by."""
    return key.ticker, _event_day(key.event_date), key.session


def _is_intraday(key: BoardRequest) -> bool:
    """Whether this request's event timestamp is past midnight of its own day."""
    stamp = pd.Timestamp(key.event_date)
    return stamp != stamp.normalize()


def _json_scalar(value: Any) -> Any:
    """One non-container value as a JSON-ready builtin: NumPy scalars normalized via
    ``.item()``, missing/non-finite as ``None``, ISO text (midnight ``YYYY-MM-DD``)."""
    import numpy as np
    value = value.item() if isinstance(value, np.generic) else value
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (str, bool, numbers.Integral)):
        return value if isinstance(value, (str, bool)) else int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (datetime, date)):
        stamp = pd.Timestamp(value)
        return (stamp.date().isoformat() if stamp.normalize() == stamp else stamp.isoformat())
    return value


def _json_value(value: Any) -> Any:
    """Recursively convert one staged value into JSON-ready builtins."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return _json_scalar(value)


def _document(value: Any) -> dict[str, Any]:
    """One converted document, validated as encodable before it is returned."""
    document = _json_value(value)
    json.dumps(document, allow_nan=False)
    return document


def _refusal(key: BoardRequest, code: str, detail: str) -> dict[str, Any]:
    """The worker's own ``{key, code, detail}`` refusal shape, fixed detail."""
    return NativeScoreBatchRowRefusal(key, code, detail).as_document()


def _refusals_document(refusal_documents: list[dict[str, Any]]) -> dict[str, Any]:
    return _document({"schema_version": _REFUSAL_SCHEMA_VERSION, "refusals": refusal_documents})


def _panel_history_start(calendar: Any, decision_session: str) -> str | None:
    """The panel history floor, or ``None`` when calendar history is not admitted.

    Pure predicate over the build-scoped decision calendar: the observed sessions
    are its days no later than ``observed_through``; history is admitted only
    when that through-day is the decision session itself and at least 253
    observed sessions exist, and the floor is then the first observed session.
    """
    observed_sessions = tuple(day for day in calendar.days if day <= calendar.observed_through)
    if calendar.observed_through != decision_session or len(observed_sessions) < 253:
        return None
    return observed_sessions[0]


def _shared_panel_rows(repository: Repository, snapshot: SnapshotRef,
                       keys: Sequence[BoardRequest], *, decision_session: str,
                       history_start: Any) -> dict[tuple[str, str, str], PanelRowInputs]:
    """One ``scan_panel_row`` read per distinct ``(ticker, event day, session)``."""
    panels: dict[tuple[str, str, str], PanelRowInputs] = {}
    spy_market_cache: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for key in keys:
        marker = _panel_marker(key)
        if marker not in panels:
            panels[marker] = scan_panel_row(
                repository, snapshot, key, decision_session=decision_session,
                history_start=history_start, spy_market_cache=spy_market_cache)
    return panels


def _regime_history_available(panel: PanelRowInputs) -> bool:
    """Whether the shared panel row carries a usable pinned SPY ``daily_market`` regime value."""
    try:
        return math.isfinite(float(panel.panel_row.get("spy_ret252")))
    except (TypeError, ValueError, OverflowError):
        return False


def _admit_price_history_requests(
    repository: Repository,
    snapshot: SnapshotRef,
    admitted: list[tuple[int, BoardRequest]],
    refusals: dict[int, dict[str, Any]],
) -> list[tuple[int, BoardRequest]]:
    """Drop each admitted request whose ticker the pinned table lacks, refusing it in place."""
    if not admitted or PRICE_HISTORY_TABLE_NAME not in snapshot.table_versions:
        return admitted
    present = tickers_with_price_history(
        repository, snapshot, {key.ticker for _, key in admitted})
    for position, key in admitted:
        if key.ticker not in present:
            refusals[position] = _refusal(
                key, _PRICE_HISTORY_CODE, _PRICE_HISTORY_DETAIL)
    return [(position, key) for position, key in admitted if key.ticker in present]


def _compose_event(
    repository: Repository,
    snapshot: SnapshotRef,
    key: BoardRequest,
    *,
    decision_session: str,
    calendar: Any,
    panel: PanelRowInputs,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """One admitted request's event document, or its in-place refusal document."""
    if not _regime_history_available(panel):
        return None, _refusal(key, "PANEL_HISTORY_NOT_AVAILABLE", _PANEL_HISTORY_DETAIL)
    try:
        calendar_row = scan_calendar_row_inputs(
            repository, snapshot, key, decision_session=decision_session,
            calendar=calendar).calendar_row
        quotes = scan_quote_rows(
            repository, snapshot, key, expiry=calendar_row["expiry"],
            decision_session=decision_session)
    except NativeScoreBatchRowRefusal as refusal:
        detail = _ROW_REFUSAL_DETAILS.get(refusal.code)
        if detail is None:
            raise
        return None, _refusal(key, refusal.code, detail)
    except (OpsError, DataError) as error:
        if error.code not in _EVENT_ERROR_CODES:
            raise
        return None, _refusal(key, error.code, _ROW_REFUSAL_DETAILS[error.code])
    event = _document({
        "key": {"ticker": key.ticker, "strategy": key.strategy,
                "event_date": _event_day(key.event_date), "session": key.session},
        "calendar_row": calendar_row, "panel_row": panel.panel_row,
        "panel_anchor": panel.panel_anchor, "tier4_row": {},
        "quote_rows": quotes.quote_rows, "quote_status": quotes.quote_status,
    })
    return event, None


def build_native_score_batch_events(
    repository: Repository,
    snapshot: SnapshotRef,
    *,
    as_of: Any,
    horizon_days: int,
    tickers: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compose the pinned forward board into raw-row event documents.

    Returns ``(events_document, refusals_document)``: one JSON-ready event
    document (``key``, ``calendar_row``, ``panel_row``, ``panel_anchor``,
    ``tier4_row`` -- always ``{}`` -- ``quote_rows``, ``quote_status``) per fully
    composed request, plus the disjoint ``producer_refusals.v1.0`` document. An
    empty enumeration returns both empty before any calendar, panel, spot or
    quote read, and an all-intraday one returns only its refusals. Only
    ``NO_RESOLVABLE_EXPIRY``/``EVENT_NOT_FOUND``/``IDENTITY_CONFLICT`` become
    per-key refusals, plus ``PRICE_HISTORY_NOT_AVAILABLE`` as a per-key absence
    from pinned table membership, refused at its original position while the
    rest of the build proceeds; a malformed series, an absent ``price_history``
    table, and an exact-session spot failure propagate, as does every other
    refusal, repository or caller-input failure, and an absent
    ``option_chains`` table once price-history admission refuses every
    non-intraday request, and the whole build fails.
    """
    requests = tuple(scan_forward_board_requests(
        repository, snapshot, as_of=as_of, horizon_days=horizon_days, tickers=tickers))
    refusals: dict[int, dict[str, Any]] = {}
    admitted: list[tuple[int, BoardRequest]] = []

    def ordered() -> list[dict[str, Any]]:
        return [refusals[position] for position in sorted(refusals)]

    for position, key in enumerate(requests):
        if _is_intraday(key):
            refusals[position] = _refusal(key, _INTRADAY_CODE, _INTRADAY_DETAIL)
        else:
            admitted.append((position, key))
    had_non_intraday = bool(admitted)
    admitted = _admit_price_history_requests(repository, snapshot, admitted, refusals)
    if not admitted:
        if had_non_intraday and "option_chains" not in snapshot.table_versions:
            raise fail("CONTRACT_MISMATCH", "table is not part of this snapshot",
                       details={"table_name": "option_chains"})
        return [], _refusals_document(ordered())

    decision_session = validated_as_of(as_of).normalize().date().isoformat()
    calendar = scan_decision_calendar(
        repository, snapshot, decision_session=decision_session,
        event_through=max(_event_day(key.event_date) for _, key in admitted))
    history_start = _panel_history_start(calendar, decision_session)
    if history_start is None:
        for position, key in admitted:
            refusals[position] = _refusal(key, "PANEL_HISTORY_NOT_AVAILABLE", _PANEL_HISTORY_DETAIL)
        return [], _refusals_document(ordered())
    panels = _shared_panel_rows(repository, snapshot, [key for _, key in admitted],
                                decision_session=decision_session,
                                history_start=history_start)
    events: list[dict[str, Any]] = []
    for position, key in admitted:
        event, refusal = _compose_event(
            repository, snapshot, key, decision_session=decision_session,
            calendar=calendar, panel=panels[_panel_marker(key)])
        if refusal is None:
            events.append(event)
        else:
            refusals[position] = refusal
    return events, _refusals_document(ordered())
