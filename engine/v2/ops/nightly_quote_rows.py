"""Pinned ``option_chains`` quote-row staging for the raw-row producer.

Stages one exact ``(ticker, obs_date)`` chain slice -- read off the pinned
snapshot (once per build by ``nightly_chain_batch``, handed in as ``chain_rows``,
or by its own per-key scan when none is given), explicitly un-admitted -- into the plain quote rows the
raw-row producer's assembler passes
to ``nightly_source_bundle``. The read is SHADOW-only and safe by namespace
isolation, not by source admission: ``native_score_batch`` is registered
``namespaces=frozenset({"shadow", "smoke"})`` with no ``store_domains``, so
it commits no legacy-store head and holds no read/write lease the legacy
board depends on (the rationale written in ``ARCHITECTURE.md``'s "Raw-row
producer" section). Causality is by data dates only: ``obs_date`` must equal
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
from typing import Any, Mapping, Sequence

from engine.v2.contracts import DataQuery, KeyPredicate, SnapshotRef
from engine.v2.data.errors import fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import fail
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_score_batch_types import NativeScoreBatchRowRefusal
from engine.v2.ops.nightly_chain_batch import CHAIN_COLUMNS, require_chain_rows
from engine.v2.scoring.nightly_source_bundle import (
    NightlySourceBundleRefusal,
    validated_as_of,
)

__all__ = ["QuoteRowInputs", "scan_quote_rows"]

_QUOTE_TABLE = "option_chains"
_BATCH_CAP = 50_000
_RESULT_CAP = 2_000_000
#: The per-row refusal an unusable/disagreeing chain spot becomes, with its
#: fixed public-safe detail (CWE-209): never a staged value.
_CHAIN_SPOT_UNAVAILABLE = "CHAIN_SPOT_UNAVAILABLE"
_CHAIN_SPOT_DETAIL = "the option chain carries no usable spot for the board request"


class _UnusableChainSpot(ValueError):
    """A surviving quote row's chain spot is unusable or disagrees with the slice."""


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


def _chain_spot(value: Any) -> float:
    """One surviving quote row's chain ``spot`` as a float.

    The chain spot is the legacy ``entry_cost_pct`` denominator, so an unusable
    one is a per-row ``CHAIN_SPOT_UNAVAILABLE`` refusal, not a build failure:
    ``None``, NaN, +/-inf, a non-numeric or bool value, or a non-positive number
    raises ``_UnusableChainSpot`` with a fixed message that never echoes the
    staged value, and ``scan_quote_rows`` turns it into that typed refusal. Whether an
    incomplete bid/ask makes the domain unusable stays
    ``nightly_source_bundle.quote_domain_map``'s own validation; only the spot
    is judged here.
    """
    if isinstance(value, bool):
        raise _UnusableChainSpot("option chain row has an unusable spot")
    try:
        spot = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _UnusableChainSpot("option chain row has an unusable spot") from None
    if not math.isfinite(spot) or spot <= 0.0:
        raise _UnusableChainSpot("option chain row has an unusable spot")
    return spot


def _matching_quote_rows(rows: list[dict], expiry: str,
                         decision_session: str) -> tuple[Mapping[str, Any], ...]:
    staged: list[Mapping[str, Any]] = []
    chain_spot: float | None = None
    for row in rows:
        if _date_string(row["expiry"]) != expiry:
            continue
        spot = _chain_spot(row["spot"])
        if chain_spot is None:
            chain_spot = spot
        elif spot != chain_spot:
            raise _UnusableChainSpot(
                "option chain rows disagree on the spot for one expiry")
        staged.append({
            "ticker": row["ticker"], "right": row["right"],
            "strike": float(row["strike"]), "expiry": expiry,
            "bid": _quote_number(row["bid"]), "ask": _quote_number(row["ask"]),
            "chain_spot": spot, "observed_at": decision_session,
        })
    return tuple(staged)


def _scan_key_rows(repository: Repository, snapshot: SnapshotRef, ticker: str,
                   session_day: str) -> list[Mapping[str, Any]]:
    """One bounded ``option_chains`` scan of exactly ``(ticker, session_day)``.

    The requested expiry is data, not a key predicate, so it is applied in
    Python by the caller; a missing table is ``Repository.table_contract``'s
    own typed ``CONTRACT_MISMATCH``.
    """
    contract = repository.table_contract(snapshot, _QUOTE_TABLE)
    version = snapshot.table_versions[_QUOTE_TABLE]
    key_filter = (KeyPredicate(column="ticker", operator="eq", values=(ticker,)),
                  KeyPredicate(column="obs_date", operator="eq", values=(session_day,)))
    population_bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=_QUOTE_TABLE,
        table_contract_ref=version.table_contract_ref,
        key_filter=key_filter, time_interval=None)
    max_result_rows = min(_RESULT_CAP, population_bound)
    max_batch_rows = (min(contract.maximum_batch_rows, _BATCH_CAP, max_result_rows)
                      if max_result_rows > 0
                      else min(contract.maximum_batch_rows, _BATCH_CAP))
    query = DataQuery(
        snapshot_id=snapshot.snapshot_id, table_contract_ref=version.table_contract_ref,
        columns=CHAIN_COLUMNS,
        key_filter=key_filter,
        order_by=tuple(contract.primary_key),
        max_batch_rows=max_batch_rows,
        max_result_rows=max_result_rows)
    rows = []
    for batch in repository.scan(query, table_name=_QUOTE_TABLE):
        rows.extend(batch.to_pylist())
    return require_chain_rows(rows, ticker, session_day)


def scan_quote_rows(
    repository: Repository,
    snapshot: SnapshotRef,
    key: BoardRequest,
    *,
    expiry: Any,
    decision_session: Any,
    chain_rows: Sequence[Mapping[str, Any]] | None = None,
) -> QuoteRowInputs:
    """Stage one exact ``option_chains`` chain slice for a board key.

    ``expiry`` and ``decision_session`` must be naive midnight dates; the
    resolved ``decision_session`` must not be after the resolved ``expiry``
    (a quote as of a date past the contract's own expiry is never
    meaningful -- ``QUERY_NOT_BOUNDED``, mirroring ``ChainQuery``'s own
    ceiling check). When ``chain_rows`` is None, one bounded ``option_chains`` scan reads
    exactly ``(key.ticker, decision_session)``; the requested ``expiry`` narrows the
    fetched rows in Python, because a chain's own expiry is data, not a key
    predicate. A slice with no surviving row returns an empty
    ``QuoteRowInputs`` with ``quote_status="empty"``, never a raise; a
    missing ``option_chains`` table raises ``CONTRACT_MISMATCH``
    (``Repository.table_contract``'s own typed refusal). Malformed
    key/date inputs raise ``INVALID_REQUEST``; a row (scanned or given) with a
    null expiry, strike or right, an unusable strike or right, or a non-numeric
    bid or ask raises ``CONTRACT_MISMATCH``; repository failures
    propagate unchanged. ``chain_rows`` (that ticker's rows for the session, from
    one batched build read) replaces the scan when given; ``None`` keeps
    the per-key scan.

    Every row that survives the expiry filter also carries ``chain_spot``: the
    chain's own ``spot`` as a float, staged for a later ``entry_cost_pct``
    consumer and read by nothing here. Only surviving rows are judged -- a row
    for another expiry or date with a null spot never fails the slice, because
    ``require_chain_rows`` checks presence, never value. A surviving row whose
    spot is ``None``, NaN, non-finite, non-numeric, bool or non-positive, or
    surviving rows that disagree on it, make this call raise the typed per-row
    ``NativeScoreBatchRowRefusal`` with code ``CHAIN_SPOT_UNAVAILABLE`` and a
    fixed detail (no staged value) -- like the calendar reader's
    ``PRICE_HISTORY_NOT_AVAILABLE``/``NO_RESOLVABLE_EXPIRY`` refusals, it
    refuses this key in place while other rows continue; the raw-row producer
    maps it to its refusal document. Unlike those, an unusable bid/ask or
    strike still raises ``CONTRACT_MISMATCH`` and fails the build.
    """
    if not isinstance(key, BoardRequest) or any(
        not isinstance(value, str) or not value.strip()
        for value in (key.ticker, key.strategy, key.session)
    ):
        raise fail("INVALID_REQUEST", "quote key requires non-empty identity fields")
    expiry_day = _calendar_day(expiry)
    session_day = _calendar_day(decision_session)
    checked_rows = (None if chain_rows is None
                    else require_chain_rows(chain_rows, key.ticker, session_day))
    if session_day > expiry_day:
        raise data_fail("QUERY_NOT_BOUNDED", "decision_session is after the contract expiry")
    if checked_rows is not None:
        rows = checked_rows
    else:
        rows = _scan_key_rows(repository, snapshot, key.ticker, session_day)
    try:
        quote_rows = _matching_quote_rows(rows, expiry_day, session_day)
    except _UnusableChainSpot:
        raise NativeScoreBatchRowRefusal(
            key, _CHAIN_SPOT_UNAVAILABLE, _CHAIN_SPOT_DETAIL) from None
    if not quote_rows:
        return QuoteRowInputs(quote_rows=(), quote_status="empty")
    return QuoteRowInputs(quote_rows=quote_rows, quote_status="recorded")
