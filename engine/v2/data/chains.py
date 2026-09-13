"""Pure legacy chain mapping: ``option_chains`` rows -> ``ChainSnapshot``,
versioned ``legacy_stored_quote.v1`` — phase-2 guide §5.4, §8.3, task brief
decision 4.

``get_chain`` is the one impure entry point (bounded scans through a
caller-supplied ``Repository``, plus one ``get_event`` call — see the
judgement call below); :func:`map_row`, :func:`contract_id_for`, and
:func:`exact_decimal_string` are pure and independently testable.

Judgement call, revised per coordinator review (task brief decision 4 names
the row mapping but not how a ``ChainQuery.security_id`` — a one-way sha256
— is turned back into the legacy ``ticker`` ``option_chains`` is actually
keyed by): ``event_ref`` is optional on the contract, so ``get_chain`` does
not require it.

* When ``event_ref`` is given, it is resolved via :func:`events.get_event`
  to recover ``ticker_at_event``, and ``events.security_id_for_ticker(ticker)
  == chain_query.security_id`` is verified — ``IDENTITY_CONFLICT`` on a
  mismatch.
* When it is absent, the ticker is recovered by scanning the pinned
  ``securities`` table (``columns=("ticker",)``, bounded by a ``year`` key
  predicate derived from ``session_date`` — no new filterable column: `year`
  is already filterable on ``securities``) and keeping the rows whose
  ``legacy_ticker.v1`` security_id matches. Zero matches is
  ``IDENTITY_CONFLICT`` ("unknown security"); more than one distinct ticker
  matching is also ``IDENTITY_CONFLICT`` ("ambiguous symbology") — two
  different tickers can never share one sha256 in practice, but the check
  costs nothing and names the right failure if they ever did.

Judgement call: ``expected_contracts`` counts every distinct
``(expiry, right, strike)`` combination the bounded ticker/session scan
returns, *before* ``expiry_interval`` narrowing; ``supported_contracts`` and
``returned_contracts`` count only members that also fall inside
``expiry_interval`` (and, defensively, on or before the ceiling). This is
this module's own reading of decision 4's "within the interval" phrase —
the alternative (interval-scoped ``expected`` too) makes the three
populations coincide in every case this legacy source can produce, so it
would leave "expected, supported, and returned populations differ where
they should" (D07) unexercisable by any real data.

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts``, ``engine.v2.foundation``, and this package's own
``errors``/``events`` — never ``engine.v2.ops`` or legacy ``engine.*``.
"""
from __future__ import annotations

from datetime import date

from engine.v2.contracts.data import (
    ChainMember,
    ChainQuery,
    ChainSnapshot,
    ContractId,
    DataQuery,
    KeyPredicate,
    SnapshotRef,
)
from engine.v2.foundation import content_hash, parse_timestamp

from . import events
from .errors import fail

__all__ = [
    "QUOTE_POLICY_REF",
    "TABLE_NAME",
    "contract_id_for",
    "exact_decimal_string",
    "get_chain",
    "map_row",
]

TABLE_NAME = "option_chains"
SECURITIES_TABLE_NAME = "securities"
QUOTE_POLICY_REF = "legacy_stored_quote.v1"

_CHAIN_COLUMNS = ("ticker", "obs_date", "expiry", "strike", "right", "bid", "ask", "mid",
                  "iv", "delta", "volume", "open_interest", "bid_size", "ask_size",
                  "src", "src_file", "quote_repaired")
_BATCH_CAP = 50000
_RESULT_CAP = 2_000_000


def exact_decimal_string(value: float) -> str:
    """The full stored value as a decimal string — ``repr`` of a Python float
    is already its shortest round-trip representation, never display-rounded
    (phase-2 guide §5.4)."""
    return repr(float(value))


def contract_id_for(security_id: str, expiry: str, right: str, exact_strike: str) -> ContractId:
    """``contract_id = sha256({scheme: legacy_option.v1, security_id, expiry,
    right, exact_strike, multiplier: "100", adjustment_identity:
    "legacy_standard.v1"})`` (phase-2 guide §5.4)."""
    payload = {"scheme": "legacy_option.v1", "security_id": security_id, "expiry": expiry,
              "right": right, "exact_strike": exact_strike, "multiplier": "100",
              "adjustment_identity": "legacy_standard.v1"}
    return ContractId(contract_id=content_hash(payload), security_id=security_id, vendor_mappings={},
                      expiry=expiry, right=right, exact_strike=exact_strike, multiplier="100",
                      adjustment_identity="legacy_standard.v1")


def map_row(row: dict, *, ticker_at_event: str, security_id: str) -> ChainMember:
    """One decoded ``option_chains`` row -> ``ChainMember`` (decision 4).

    Refuses with ``IDENTITY_CONFLICT`` when the row's own ``ticker`` differs
    from the ticker this chain was resolved for — a contract from another
    security (§8.3: "refuse a contract from another security").
    """
    if row["ticker"] != ticker_at_event:
        raise fail("IDENTITY_CONFLICT", "chain row belongs to a different ticker than requested",
                  details={"row_ticker": row["ticker"], "requested_ticker": ticker_at_event})
    expiry = _date_string(row["expiry"])
    exact_strike = exact_decimal_string(row["strike"])
    contract_id = contract_id_for(security_id, expiry, row["right"], exact_strike)
    quality_flags = ("legacy_quote_repaired",) if row.get("quote_repaired") else ()
    usable = (row["bid"] is not None and row["ask"] is not None
              and 0 <= row["bid"] <= row["ask"])
    if not usable:
        return ChainMember(
            contract_id=contract_id, bid=None, ask=None, mid=None, iv=None, delta=None,
            volume=_int_or_none(row.get("volume")), open_interest=_int_or_none(row.get("open_interest")),
            bid_size=_int_or_none(row.get("bid_size")), ask_size=_int_or_none(row.get("ask_size")),
            source=row.get("src") or "unknown", source_row_ref=row.get("src_file") or "unknown",
            availability_status="unavailable", missing_reason="no_usable_quote",
            quality_flags=quality_flags)
    return ChainMember(
        contract_id=contract_id, bid=exact_decimal_string(row["bid"]), ask=exact_decimal_string(row["ask"]),
        mid=exact_decimal_string(row["mid"]) if row.get("mid") is not None else None,
        iv=row.get("iv"), delta=row.get("delta"),
        volume=_int_or_none(row.get("volume")), open_interest=_int_or_none(row.get("open_interest")),
        bid_size=_int_or_none(row.get("bid_size")), ask_size=_int_or_none(row.get("ask_size")),
        source=row.get("src") or "unknown", source_row_ref=row.get("src_file") or "unknown",
        # "get_chain may return reconstructed history; it may not call it
        # observed" (phase-2 guide §5.4) — never "observed" here.
        availability_status="reconstructed", missing_reason=None, quality_flags=quality_flags)


def get_chain(repository, chain_query: ChainQuery, snapshot_ref: SnapshotRef) -> ChainSnapshot:
    """§8.3: one bounded ticker/session scan, organized under the named quote
    policy. Never selects legs, ranks a structure, or computes PnL."""
    if chain_query.quote_policy_ref != QUOTE_POLICY_REF:
        raise fail("UNSUPPORTED_CONTRACT", "unknown quote_policy_ref",
                  details={"contract": chain_query.quote_policy_ref})
    if TABLE_NAME not in snapshot_ref.table_versions:
        raise fail("CONTRACT_MISMATCH", "snapshot has no option_chains table")
    ceiling_date = parse_timestamp(chain_query.observation_ceiling).date()
    session_date = date.fromisoformat(chain_query.session_date)
    if session_date > ceiling_date:
        raise fail("QUERY_NOT_BOUNDED", "session_date is after the observation ceiling")
    ticker, security_id = _resolve_ticker(repository, chain_query, snapshot_ref)

    dvr = snapshot_ref.table_versions[TABLE_NAME]
    rows = _fetch_rows(repository, snapshot_ref, dvr.table_contract_ref, ticker, chain_query.session_date)
    expected = {(r["expiry"], r["right"], r["strike"]) for r in rows}
    if not expected:
        raise fail("POPULATION_COLLAPSED", "no contracts exist for this ticker/session")

    members, supported = _build_members(rows, chain_query, ticker, security_id, ceiling_date)
    if len(members) > chain_query.max_contracts:
        raise fail("RESULT_LIMIT_EXCEEDED", "chain exceeds max_contracts",
                  details={"max_contracts": chain_query.max_contracts})
    return ChainSnapshot(
        chain_id=content_hash({"scheme": "legacy_chain.v1", "security_id": security_id,
                               "session_date": chain_query.session_date,
                               "quote_policy_ref": QUOTE_POLICY_REF}),
        source_snapshot_ref=snapshot_ref.snapshot_id, security_id=security_id,
        observed_at=f"{chain_query.session_date}T00:00:00.000000Z",
        available_at=None, received_at=None, session_date=chain_query.session_date,
        quote_policy_ref=QUOTE_POLICY_REF, spot_unadjusted=None, spot_adjusted=None,
        rows=tuple(members), expected_contracts=len(expected), supported_contracts=len(supported),
        returned_contracts=len(members), coverage_ref=dvr.dataset_version_id,
        knowledge_mode=snapshot_ref.knowledge_mode_by_table.get(TABLE_NAME, "reconstructed"))


def _resolve_ticker(repository, chain_query: ChainQuery, snapshot_ref: SnapshotRef) -> tuple[str, str]:
    if chain_query.event_ref is not None:
        event = events.get_event(repository, chain_query.event_ref, snapshot_ref)
        ticker = event.ticker_at_event
        if events.security_id_for_ticker(ticker) != chain_query.security_id:
            raise fail("IDENTITY_CONFLICT", "security_id does not match the event's ticker mapping",
                      details={"security_id": chain_query.security_id})
        return ticker, chain_query.security_id
    return _ticker_from_securities(repository, chain_query, snapshot_ref), chain_query.security_id


def _ticker_from_securities(repository, chain_query: ChainQuery, snapshot_ref: SnapshotRef) -> str:
    """No ``event_ref``: recover the ticker by scanning the pinned
    ``securities`` table for the ``year`` of ``session_date`` (already a
    filterable column there — no new one needed) and keeping the rows whose
    ``legacy_ticker.v1`` security_id matches ``chain_query.security_id``."""
    if SECURITIES_TABLE_NAME not in snapshot_ref.table_versions:
        raise fail("CONTRACT_MISMATCH", "snapshot has no securities table")
    dvr = snapshot_ref.table_versions[SECURITIES_TABLE_NAME]
    year = date.fromisoformat(chain_query.session_date).year
    query = DataQuery(
        snapshot_id=snapshot_ref.snapshot_id, table_contract_ref=dvr.table_contract_ref,
        columns=("ticker", "year"),
        key_filter=(KeyPredicate(column="year", operator="eq", values=(year,)),),
        order_by=("ticker", "year"), max_batch_rows=_BATCH_CAP, max_result_rows=_RESULT_CAP)
    tickers = set()
    for batch in repository.scan(query, table_name=SECURITIES_TABLE_NAME):
        for row in batch.to_pylist():
            if events.security_id_for_ticker(row["ticker"]) == chain_query.security_id:
                tickers.add(row["ticker"])
    if not tickers:
        raise fail("IDENTITY_CONFLICT", "unknown security",
                  details={"security_id": chain_query.security_id})
    if len(tickers) > 1:
        raise fail("IDENTITY_CONFLICT", "ambiguous symbology",
                  details={"security_id": chain_query.security_id, "tickers": sorted(tickers)})
    return next(iter(tickers))


def _build_members(rows, chain_query: ChainQuery, ticker: str, security_id: str,
                   ceiling_date) -> tuple[list[ChainMember], set]:
    members: list[ChainMember] = []
    supported: set = set()
    for row in rows:
        if not _within_expiry_interval(row, chain_query.expiry_interval):
            continue
        if _date_string(row["obs_date"]) > ceiling_date.isoformat():
            continue
        member = map_row(row, ticker_at_event=ticker, security_id=security_id)
        members.append(member)
        supported.add((member.contract_id.expiry, member.contract_id.right, member.contract_id.exact_strike))
    return members, supported


def _fetch_rows(repository, snapshot_ref: SnapshotRef, contract_ref, ticker: str,
                session_date: str) -> list[dict]:
    query = DataQuery(
        snapshot_id=snapshot_ref.snapshot_id, table_contract_ref=contract_ref, columns=_CHAIN_COLUMNS,
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=(ticker,)),
                    KeyPredicate(column="obs_date", operator="eq", values=(session_date,))),
        order_by=("ticker", "obs_date", "expiry", "strike", "right"),
        max_batch_rows=_BATCH_CAP, max_result_rows=_RESULT_CAP)
    rows: list[dict] = []
    for batch in repository.scan(query, table_name=TABLE_NAME):
        rows.extend(batch.to_pylist())
    return rows


def _within_expiry_interval(row: dict, interval) -> bool:
    if interval is None:
        return True
    expiry_str = _date_string(row["expiry"])
    if interval.start_inclusive is not None and expiry_str < interval.start_inclusive[:10]:
        return False
    if interval.end_exclusive is not None and expiry_str >= interval.end_exclusive[:10]:
        return False
    return True


def _date_string(value) -> str:
    return value.date().isoformat() if hasattr(value, "date") else str(value)[:10]


def _int_or_none(value):
    return None if value is None else int(value)
