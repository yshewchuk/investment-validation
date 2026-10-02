"""Pinned ``option_chains`` quote-row staging for cutover PR-6.

Stages one exact ``(ticker, obs_date)`` chain slice -- read directly off the
pinned snapshot, explicitly un-admitted -- into the plain quote rows the
raw-row producer's assembler (cutover PR-6 slice 4, not built yet) passes
to ``nightly_source_bundle``. The read is SHADOW-only and safe by namespace
isolation, not by source admission: ``native_score_batch`` is registered
``namespaces=frozenset({"shadow", "smoke"})`` with no ``store_domains``, so
it commits no legacy-store head and holds no read/write lease the legacy
board depends on (the rationale written in ``ARCHITECTURE.md``'s "Cutover
PR-6" section). Causality is by data dates only: ``obs_date`` must equal
the caller's resolved ``decision_session`` exactly -- never a
latest-at-or-before lookback -- mirroring ``engine/v2/data/chains.py``'s
``get_chain`` convention rather than inventing a new policy.

Like ``native_board_universe`` and ``nightly_raw_rows``, this module never
imports ``engine.score`` / ``engine.structures`` / ``engine.replay`` /
``engine.fills``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from engine.v2.contracts import DataQuery, KeyPredicate, SnapshotRef
from engine.v2.data.errors import fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import fail
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.scoring.nightly_source_bundle import (
    NightlySourceBundleRefusal,
    validated_as_of,
)

__all__ = ["QuoteRowInputs", "scan_quote_rows"]

_QUOTE_TABLE = "option_chains"
_QUOTE_COLUMNS = ("ticker", "obs_date", "expiry", "strike", "right", "bid", "ask")
_BATCH_CAP = 50_000
_RESULT_CAP = 2_000_000


@dataclass(frozen=True, slots=True)
class QuoteRowInputs:
    """Staged quote rows and the status the assembler needs for them."""

    quote_rows: tuple[Mapping[str, Any], ...]
    quote_status: str


def _calendar_day(value: Any) -> str:
    """Reject non-day values without exposing submitted text in refusals."""
    try:
        day = validated_as_of(value)
    except NightlySourceBundleRefusal:
        raise fail("INVALID_REQUEST", "quote context requires valid naive dates") from None
    if day != day.normalize():
        raise fail("INVALID_REQUEST", "quote context requires midnight dates")
    return day.date().isoformat()


def _date_string(value: Any) -> str:
    """``chains.py::_date_string``'s own idiom, reproduced locally."""
    return value.date().isoformat() if hasattr(value, "date") else str(value)[:10]


def _quote_number(value: Any) -> float | None:
    """A null/NaN stored quote passes through as ``None``; else its float.

    Whether an incomplete quote makes the domain unusable is
    ``nightly_source_bundle.quote_domain_map``'s own validation, not this
    staging function's.
    """
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) else number


def _matching_quote_rows(rows: list[dict], expiry: str,
                         decision_session: str) -> tuple[Mapping[str, Any], ...]:
    staged: list[Mapping[str, Any]] = []
    for row in rows:
        if _date_string(row["expiry"]) != expiry:
            continue
        staged.append({
            "ticker": row["ticker"], "right": row["right"],
            "strike": float(row["strike"]), "expiry": expiry,
            "bid": _quote_number(row["bid"]), "ask": _quote_number(row["ask"]),
            "observed_at": decision_session,
        })
    return tuple(staged)


def scan_quote_rows(
    repository: Repository,
    snapshot: SnapshotRef,
    key: BoardRequest,
    *,
    expiry: Any,
    decision_session: Any,
) -> QuoteRowInputs:
    """Stage one exact ``option_chains`` chain slice for a board key.

    ``expiry`` and ``decision_session`` must be naive midnight dates; the
    resolved ``decision_session`` must not be after the resolved ``expiry``
    (a quote as of a date past the contract's own expiry is never
    meaningful -- ``QUERY_NOT_BOUNDED``, mirroring ``ChainQuery``'s own
    ceiling check). One bounded ``option_chains`` scan reads exactly
    ``(key.ticker, decision_session)``; the requested ``expiry`` narrows the
    fetched rows in Python, because a chain's own expiry is data, not a key
    predicate. A slice with no surviving row returns an empty
    ``QuoteRowInputs`` with ``quote_status="empty"``, never a raise; a
    missing ``option_chains`` table raises ``CONTRACT_MISMATCH``
    (``Repository.table_contract``'s own typed refusal). Malformed
    key/date inputs raise ``INVALID_REQUEST``; repository failures
    propagate unchanged.
    """
    if not isinstance(key, BoardRequest) or any(
        not isinstance(value, str) or not value.strip()
        for value in (key.ticker, key.strategy, key.session)
    ):
        raise fail("INVALID_REQUEST", "quote key requires non-empty identity fields")
    expiry_day = _calendar_day(expiry)
    session_day = _calendar_day(decision_session)
    if session_day > expiry_day:
        raise data_fail("QUERY_NOT_BOUNDED", "decision_session is after the contract expiry")
    contract = repository.table_contract(snapshot, _QUOTE_TABLE)
    version = snapshot.table_versions[_QUOTE_TABLE]
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=version.table_contract_ref,
        columns=_QUOTE_COLUMNS,
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=(key.ticker,)),
                    KeyPredicate(column="obs_date", operator="eq", values=(session_day,))),
        order_by=tuple(contract.primary_key),
        max_batch_rows=min(contract.maximum_batch_rows, _BATCH_CAP),
        max_result_rows=min(contract.maximum_result_rows, _RESULT_CAP))
    rows: list[dict] = []
    for batch in repository.scan(query, table_name=_QUOTE_TABLE):
        rows.extend(batch.to_pylist())
    quote_rows = _matching_quote_rows(rows, expiry_day, session_day)
    if not quote_rows:
        return QuoteRowInputs(quote_rows=(), quote_status="empty")
    return QuoteRowInputs(quote_rows=quote_rows, quote_status="recorded")
