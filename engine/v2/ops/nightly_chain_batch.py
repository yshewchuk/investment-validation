"""One batched ``option_chains`` read per producer build.

``scan_chain_slices`` replaces the per-event pair of exact ``(ticker, obs_date)``
scans: one ``DataQuery`` with ``ticker in <tickers>`` and ``obs_date eq
<decision session>`` over the union of the candidate, quote and chain-spot
columns, grouped by ticker in primary-key order (so each group equals what the
per-key scan returns). Failure semantics: a ticker with no rows maps to ``[]``; a missing
table is ``Repository.table_contract``'s own ``CONTRACT_MISMATCH``; every
repository error (including ``RESULT_LIMIT_EXCEEDED``) propagates and never
falls back to per-key scans. The rows-returned worker step and the near-cap
warning are observation only: the rows returned and every failure are
unchanged.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from engine.v2.contracts import DataQuery, KeyPredicate, SnapshotRef
from engine.v2.data.errors import fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.ops import worker_progress
from engine.v2.ops.errors import fail
from engine.v2.scoring.nightly_source_bundle import NightlySourceBundleRefusal, validated_as_of

__all__ = ["CHAIN_COLUMNS", "CHAIN_TABLE", "require_chain_rows", "require_usable_fields",
           "scan_chain_slices"]

CHAIN_TABLE = "option_chains"
#: Union of the candidate-expiry columns, the quote-row columns and the chain
#: ``spot`` the quote staging carries through (never validated here: rows for
#: other expiries or dates must not fail on it; only the surviving quote rows
#: are checked, in ``nightly_quote_rows``).
CHAIN_COLUMNS = ("ticker", "obs_date", "expiry", "strike", "right", "bid", "ask", "spot")
RIGHTS = frozenset({"C", "P"})
_BATCH_CAP = 50_000
_RESULT_CAP = 2_000_000


def _day(value: Any) -> str:
    try:
        stamp = validated_as_of(value)
    except NightlySourceBundleRefusal:
        raise fail("INVALID_REQUEST", "chain rows carry an invalid obs_date") from None
    if stamp != stamp.normalize():
        raise fail("INVALID_REQUEST", "chain rows carry a non-midnight obs_date")
    return stamp.date().isoformat()


def _number_or_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def require_usable_fields(row: Mapping[str, Any], check_quotes: bool = True) -> None:
    """The candidate/quote field checks both readers rely on; malformed pinned
    input is ``CONTRACT_MISMATCH`` (the per-key candidate path's own code)."""
    if row["expiry"] is None or row["strike"] is None or row["right"] is None:
        raise data_fail("CONTRACT_MISMATCH", "option chain row has a null expiry, strike or right")
    if str(row["right"]).upper() not in RIGHTS:
        raise data_fail("CONTRACT_MISMATCH", "option chain row has a right outside C/P")
    try:
        strike = float(row["strike"])
    except (TypeError, ValueError):
        raise data_fail("CONTRACT_MISMATCH",
                        "option chain row has an unconvertible strike") from None
    if not math.isfinite(strike):
        raise data_fail("CONTRACT_MISMATCH", "option chain row has a non-finite strike")
    if check_quotes and not (_number_or_missing(row["bid"]) and _number_or_missing(row["ask"])):
        raise data_fail("CONTRACT_MISMATCH", "option chain row has an unconvertible quote")


def require_chain_rows(rows: Sequence[Mapping[str, Any]], ticker: str,
                       session_day: str, *, check_quotes: bool = True) -> list[Mapping[str, Any]]:
    """A materialised copy of ``rows`` after checking every row is ``ticker``'s
    at ``session_day`` and carries every chain column (any other row is
    ``INVALID_REQUEST``, never echoed) and usable expiry/strike/right fields (and
    bid/ask when ``check_quotes``; otherwise ``CONTRACT_MISMATCH``)."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Iterable):
        raise fail("INVALID_REQUEST", "chain rows must be a collection of rows")
    staged = list(rows)
    for row in staged:
        if not isinstance(row, Mapping) or any(column not in row for column in CHAIN_COLUMNS):
            raise fail("INVALID_REQUEST", "chain rows must carry every chain column")
        if row["ticker"] != ticker or _day(row["obs_date"]) != session_day:
            raise fail("INVALID_REQUEST", "chain rows do not match the key and session")
        require_usable_fields(row, check_quotes)
    return staged


def scan_chain_slices(repository: Repository, snapshot: SnapshotRef, tickers: Iterable[str], *,
                      decision_session: str) -> dict[str, list[dict[str, Any]]]:
    """Every ``tickers`` entry's ``option_chains`` rows for ``decision_session``
    (a naive midnight date), from a single ``Repository.scan``."""
    try:
        day = validated_as_of(decision_session)
    except NightlySourceBundleRefusal:
        raise fail("INVALID_REQUEST", "decision_session must be a valid naive date") from None
    if day != day.normalize():
        raise fail("INVALID_REQUEST", "decision_session must be a midnight date")
    decision_session = day.date().isoformat()
    if isinstance(tickers, (str, bytes)) or not isinstance(tickers, Iterable):
        raise fail("INVALID_REQUEST", "tickers must be a collection of strings")
    listed = list(tickers)
    if any(not isinstance(ticker, str) or not ticker.strip() for ticker in listed):
        raise fail("INVALID_REQUEST", "tickers must be non-empty strings")
    wanted = sorted(set(listed))
    slices: dict[str, list[dict[str, Any]]] = {ticker: [] for ticker in wanted}
    if not wanted:
        return slices
    contract = repository.table_contract(snapshot, CHAIN_TABLE)
    version = snapshot.table_versions[CHAIN_TABLE]
    key_filter = (KeyPredicate(column="ticker", operator="in", values=tuple(wanted)),
                  KeyPredicate(column="obs_date", operator="eq", values=(decision_session,)))
    population_bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name=CHAIN_TABLE,
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
    count = 0
    for batch in repository.scan(query, table_name=CHAIN_TABLE):
        rows = batch.to_pylist()
        for row in rows:
            slices[row["ticker"]].append(row)
        count += len(rows)
    worker_progress.step_start("option_chains_rows_returned")
    worker_progress.step_end("option_chains_rows_returned", units=count)
    if count >= 0.8 * _RESULT_CAP:
        logging.getLogger(__name__).warning("option_chains read returned %d of %d allowed rows",
                                            count, _RESULT_CAP)
    return slices
