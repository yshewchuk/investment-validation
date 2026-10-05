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
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.features.panel_row_inputs import PanelRowInputs, scan_panel_row
from engine.v2.ops.errors import OpsError
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_score_batch import NativeScoreBatchRowRefusal
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


def _shared_panel_rows(repository: Repository, snapshot: SnapshotRef,
                       keys: Sequence[BoardRequest], *, decision_session: str,
                       history_start: Any) -> dict[tuple[str, str, str], PanelRowInputs]:
    """One ``scan_panel_row`` read per distinct ``(ticker, event day, session)``."""
    panels: dict[tuple[str, str, str], PanelRowInputs] = {}
    for key in keys:
        marker = _panel_marker(key)
        if marker not in panels:
            panels[marker] = scan_panel_row(
                repository, snapshot, key, decision_session=decision_session,
                history_start=history_start)
    return panels


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
    per-key refusals; every other refusal, repository or caller-input failure
    propagates and the whole build fails.
    """
    requests = tuple(scan_forward_board_requests(
        repository, snapshot, as_of=as_of, horizon_days=horizon_days, tickers=tickers))
    refusal_documents: list[dict[str, Any]] = []
    admitted: list[BoardRequest] = []
    for key in requests:
        if _is_intraday(key):
            refusal_documents.append(_refusal(key, _INTRADAY_CODE, _INTRADAY_DETAIL))
        else:
            admitted.append(key)
    if not admitted:
        return [], _refusals_document(refusal_documents)

    decision_session = validated_as_of(as_of).normalize().date().isoformat()
    calendar = scan_decision_calendar(
        repository, snapshot, decision_session=decision_session,
        event_through=max(_event_day(key.event_date) for key in admitted))
    observed_sessions = tuple(day for day in calendar.days if day <= calendar.observed_through)
    if calendar.observed_through != decision_session or len(observed_sessions) < 253:
        history_detail = "the pinned snapshot lacks required earlier panel sessions"
        return [], _refusals_document([
            _refusal(key, _INTRADAY_CODE, _INTRADAY_DETAIL) if _is_intraday(key)
            else _refusal(key, "PANEL_HISTORY_NOT_AVAILABLE", history_detail)
            for key in requests])
    panels = _shared_panel_rows(repository, snapshot, admitted,
                                decision_session=decision_session,
                                history_start=observed_sessions[-253])

    events: list[dict[str, Any]] = []
    for key in admitted:
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
            refusal_documents.append(_refusal(key, refusal.code, detail))
            continue
        except (OpsError, DataError) as error:
            if error.code not in _EVENT_ERROR_CODES:
                raise
            refusal_documents.append(_refusal(key, error.code, _ROW_REFUSAL_DETAILS[error.code]))
            continue
        panel = panels[_panel_marker(key)]
        events.append(_document({
            "key": {"ticker": key.ticker, "strategy": key.strategy,
                    "event_date": _event_day(key.event_date), "session": key.session},
            "calendar_row": calendar_row, "panel_row": panel.panel_row,
            "panel_anchor": panel.panel_anchor, "tier4_row": {},
            "quote_rows": quotes.quote_rows, "quote_status": quotes.quote_status,
        }))
    return events, _refusals_document(refusal_documents)
