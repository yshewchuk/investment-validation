"""D07: exact event lookup and legacy chain mapping — phase-2 guide §5.4,
§8.3, task brief decision 3/4.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts.data import ChainQuery, EventRef  # noqa: E402
from engine.v2.data import chains, events  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    hand_built_record,
    publish_and_inspect,
    table_from_rows,
)

_EVENTS = contract_for("earnings_events")
_EVENTS_REF = contract_ref_for(_EVENTS)
_CHAINS = contract_for("option_chains")
_CHAINS_REF = contract_ref_for(_CHAINS)


def _event_row(event_id: str, ticker: str, event_date: datetime, *, session="BMO",
              session_src="orats", date_agree=True, date_conflict=False,
              event_cluster_id=None) -> dict:
    return dict(
        event_id=event_id, ticker=ticker, event_date=event_date, year=event_date.year,
        session=session, session_src=session_src, annc_tod=None, src_orats=True,
        src_oquants=True, src_nasdaq=False, src_yfinance=False, date_agree=date_agree,
        date_conflict=date_conflict, updated_at=None, event_cluster_id=event_cluster_id,
        claim_count=1 if event_cluster_id else None,
        reconciliation="single_source" if event_cluster_id else None)


def _chain_row(ticker: str, obs_date: datetime, expiry: datetime, strike: float, *, right="C",
              bid=1.0, ask=1.2, mid=1.1, quote_repaired=False) -> dict:
    return dict(ticker=ticker, obs_date=obs_date, year=obs_date.year, expiry=expiry,
               dte=(expiry - obs_date).days, strike=strike, right=right, bid=bid, ask=ask, mid=mid,
               iv=30.0, delta=0.5, spot=100.0, src="orats", src_file="f.parquet", chain_kind="entry",
               volume=None, open_interest=None, bid_size=None, ask_size=None,
               quote_repaired=quote_repaired)


def _events_snapshot(tmp_path, rows, *, year="2024"):
    conn, clock, store = catalog_and_store(tmp_path)
    record = publish_and_inspect(store, _EVENTS, _EVENTS_REF, rows, year)
    snap = commit_tables(conn, clock, {"earnings_events": [record]}, {"earnings_events": _EVENTS})
    return conn, store, snap


def _combined_snapshot(tmp_path, event_rows, chain_rows, *, year="2024"):
    conn, clock, store = catalog_and_store(tmp_path)
    event_record = publish_and_inspect(store, _EVENTS, _EVENTS_REF, event_rows, year)
    chain_record = publish_and_inspect(store, _CHAINS, _CHAINS_REF, chain_rows, year)
    snap = commit_tables(conn, clock,
                         {"earnings_events": [event_record], "option_chains": [chain_record]},
                         {"earnings_events": _EVENTS, "option_chains": _CHAINS})
    return conn, store, snap


# --------------------------------------------------------------------------
# get_event
# --------------------------------------------------------------------------


def test_get_event_maps_a_row(tmp_path):
    conn, store, snap = _events_snapshot(
        tmp_path, [_event_row("AAA_2024-01-05", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    revision = snap.table_versions["earnings_events"].dataset_version_id
    ref = EventRef(event_id="AAA_2024-01-05", calendar_revision=revision)
    event = repo.get_event(ref, snap)
    assert event.ticker_at_event == "AAA"
    assert event.scheduled_event_date == "2024-01-05"
    assert event.security_id == events.security_id_for_ticker("AAA")
    assert event.confidence == 1.0
    assert event.conflict_status == "none"


def test_revision_mismatch_is_contract_mismatch(tmp_path):
    conn, store, snap = _events_snapshot(
        tmp_path, [_event_row("AAA_2024-01-05", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    ref = EventRef(event_id="AAA_2024-01-05", calendar_revision="dsv_" + "0" * 32)
    with pytest.raises(DataError) as err:
        repo.get_event(ref, snap)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_duplicate_event_id_is_manifest_corrupt(tmp_path):
    """Two rows sharing one ``event_id`` — unreachable through the normal
    ``inspect_fragment`` publish path (it already refuses a duplicate primary
    key at ingest time), so this bypasses it via a hand-built record, the
    same technique ``test_v2_data_query.py``'s required-column test uses."""
    conn, clock, store = catalog_and_store(tmp_path)
    rows = [_event_row("AAA_2024-01-05", "AAA", datetime(2024, 1, 5)),
            _event_row("AAA_2024-01-05", "AAA", datetime(2024, 1, 5), session="AMC")]
    table = table_from_rows(_EVENTS, rows)
    record = hand_built_record(store, _EVENTS, _EVENTS_REF, table, partition_key="2024", row_count=2,
                               primary_key_min=("AAA_2024-01-05",), primary_key_max=("AAA_2024-01-05",))
    snap = commit_tables(conn, clock, {"earnings_events": [record]}, {"earnings_events": _EVENTS})
    repo = Repository(conn, store)
    revision = snap.table_versions["earnings_events"].dataset_version_id
    ref = EventRef(event_id="AAA_2024-01-05", calendar_revision=revision)
    with pytest.raises(DataError) as err:
        repo.get_event(ref, snap)
    assert err.value.code == "MANIFEST_CORRUPT"


def test_cluster_conflict_surfaces_as_conflict_status(tmp_path):
    conn, store, snap = _events_snapshot(tmp_path, [
        _event_row("AAA_2024-01-04", "AAA", datetime(2024, 1, 4), event_cluster_id="AAA_2024-01"),
        _event_row("AAA_2024-01-05", "AAA", datetime(2024, 1, 5), event_cluster_id="AAA_2024-01"),
    ])
    repo = Repository(conn, store)
    revision = snap.table_versions["earnings_events"].dataset_version_id
    ref = EventRef(event_id="AAA_2024-01-04", calendar_revision=revision)
    event = repo.get_event(ref, snap)
    assert event.conflict_status == "conflict"


def test_no_cluster_conflict_when_cluster_is_singleton(tmp_path):
    conn, store, snap = _events_snapshot(tmp_path, [
        _event_row("AAA_2024-01-04", "AAA", datetime(2024, 1, 4), event_cluster_id="AAA_2024-01"),
    ])
    repo = Repository(conn, store)
    revision = snap.table_versions["earnings_events"].dataset_version_id
    ref = EventRef(event_id="AAA_2024-01-04", calendar_revision=revision)
    event = repo.get_event(ref, snap)
    assert event.conflict_status == "none"


# --------------------------------------------------------------------------
# chains.map_row — cross-security row
# --------------------------------------------------------------------------


def test_cross_security_chain_row_is_identity_conflict():
    row = _chain_row("ZZZ", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0)
    with pytest.raises(DataError) as err:
        chains.map_row(row, ticker_at_event="AAA", security_id=events.security_id_for_ticker("AAA"))
    assert err.value.code == "IDENTITY_CONFLICT"


def test_exact_strike_is_preserved_without_display_rounding():
    row = _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.00000000000001)
    member = chains.map_row(row, ticker_at_event="AAA", security_id=events.security_id_for_ticker("AAA"))
    assert member.contract_id.exact_strike == "100.00000000000001"


def test_unusable_quote_is_retained_with_reason():
    row = _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0, bid=None, ask=None)
    member = chains.map_row(row, ticker_at_event="AAA", security_id=events.security_id_for_ticker("AAA"))
    assert member.bid is None and member.ask is None and member.mid is None
    assert member.availability_status == "unavailable"
    assert member.missing_reason == "no_usable_quote"


def test_quote_repaired_flag_becomes_a_quality_flag():
    row = _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0, quote_repaired=True)
    member = chains.map_row(row, ticker_at_event="AAA", security_id=events.security_id_for_ticker("AAA"))
    assert "legacy_quote_repaired" in member.quality_flags


# --------------------------------------------------------------------------
# get_chain
# --------------------------------------------------------------------------


def _chain_query(**overrides) -> ChainQuery:
    base = dict(event_ref=None, security_id=events.security_id_for_ticker("AAA"),
               observation_ceiling="2024-01-05T23:59:59.000000Z", session_date="2024-01-05",
               expiry_interval=None, quote_policy_ref=chains.QUOTE_POLICY_REF, max_contracts=10)
    base.update(overrides)
    return ChainQuery(**base)


def test_get_chain_organizes_rows_for_the_session(tmp_path):
    event_rows = [_event_row("AAA_2024-01-05", "AAA", datetime(2024, 1, 5))]
    chain_rows = [_chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0),
                  _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 105.0, right="P")]
    conn, store, snap = _combined_snapshot(tmp_path, event_rows, chain_rows)
    repo = Repository(conn, store)
    ref = EventRef(event_id="AAA_2024-01-05",
                   calendar_revision=snap.table_versions["earnings_events"].dataset_version_id)
    query = _chain_query(event_ref=ref)
    chain = repo.get_chain(query, snap)
    assert chain.expected_contracts == 2
    assert chain.supported_contracts == 2
    assert chain.returned_contracts == 2
    assert chain.knowledge_mode == "reconstructed"


def test_post_ceiling_session_is_refused(tmp_path):
    event_rows = [_event_row("AAA_2024-01-05", "AAA", datetime(2024, 1, 5))]
    chain_rows = [_chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0)]
    conn, store, snap = _combined_snapshot(tmp_path, event_rows, chain_rows)
    repo = Repository(conn, store)
    ref = EventRef(event_id="AAA_2024-01-05",
                   calendar_revision=snap.table_versions["earnings_events"].dataset_version_id)
    query = _chain_query(event_ref=ref, observation_ceiling="2024-01-04T00:00:00.000000Z")
    with pytest.raises(DataError) as err:
        repo.get_chain(query, snap)
    assert err.value.code == "QUERY_NOT_BOUNDED"


def test_collapsed_expected_population_is_refused(tmp_path):
    event_rows = [_event_row("AAA_2024-01-05", "AAA", datetime(2024, 1, 5))]
    chain_rows = [_chain_row("AAA", datetime(2024, 1, 6), datetime(2024, 2, 16), 100.0)]
    conn, store, snap = _combined_snapshot(tmp_path, event_rows, chain_rows)
    repo = Repository(conn, store)
    ref = EventRef(event_id="AAA_2024-01-05",
                   calendar_revision=snap.table_versions["earnings_events"].dataset_version_id)
    # session_date (2024-01-05) has no matching option_chains rows at all
    # (the only row is obs_date=2024-01-06): expected_contracts collapses to 0.
    query = _chain_query(event_ref=ref)
    with pytest.raises(DataError) as err:
        repo.get_chain(query, snap)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_max_contracts_exceeded_is_result_limit_exceeded(tmp_path):
    event_rows = [_event_row("AAA_2024-01-05", "AAA", datetime(2024, 1, 5))]
    chain_rows = [_chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), strike)
                 for strike in (95.0, 100.0, 105.0)]
    conn, store, snap = _combined_snapshot(tmp_path, event_rows, chain_rows)
    repo = Repository(conn, store)
    ref = EventRef(event_id="AAA_2024-01-05",
                   calendar_revision=snap.table_versions["earnings_events"].dataset_version_id)
    query = _chain_query(event_ref=ref, max_contracts=2)
    with pytest.raises(DataError) as err:
        repo.get_chain(query, snap)
    assert err.value.code == "RESULT_LIMIT_EXCEEDED"


def test_expected_supported_returned_differ_with_expiry_interval(tmp_path):
    from engine.v2.contracts.data import TimeInterval
    event_rows = [_event_row("AAA_2024-01-05", "AAA", datetime(2024, 1, 5))]
    chain_rows = [_chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0),
                  _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 3, 15), 100.0)]
    conn, store, snap = _combined_snapshot(tmp_path, event_rows, chain_rows)
    repo = Repository(conn, store)
    ref = EventRef(event_id="AAA_2024-01-05",
                   calendar_revision=snap.table_versions["earnings_events"].dataset_version_id)
    interval = TimeInterval(column="expiry", start_inclusive="2024-02-01", end_exclusive="2024-03-01")
    query = _chain_query(event_ref=ref, expiry_interval=interval)
    chain = repo.get_chain(query, snap)
    assert chain.expected_contracts == 2
    assert chain.supported_contracts == 1
    assert chain.returned_contracts == 1


def test_missing_event_ref_is_refused(tmp_path):
    event_rows = [_event_row("AAA_2024-01-05", "AAA", datetime(2024, 1, 5))]
    chain_rows = [_chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0)]
    conn, store, snap = _combined_snapshot(tmp_path, event_rows, chain_rows)
    repo = Repository(conn, store)
    query = _chain_query(event_ref=None)
    with pytest.raises(DataError) as err:
        repo.get_chain(query, snap)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_security_id_mismatch_is_identity_conflict(tmp_path):
    event_rows = [_event_row("AAA_2024-01-05", "AAA", datetime(2024, 1, 5))]
    chain_rows = [_chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0)]
    conn, store, snap = _combined_snapshot(tmp_path, event_rows, chain_rows)
    repo = Repository(conn, store)
    ref = EventRef(event_id="AAA_2024-01-05",
                   calendar_revision=snap.table_versions["earnings_events"].dataset_version_id)
    query = _chain_query(event_ref=ref, security_id=events.security_id_for_ticker("ZZZ"))
    with pytest.raises(DataError) as err:
        repo.get_chain(query, snap)
    assert err.value.code == "IDENTITY_CONFLICT"


def test_unknown_quote_policy_ref_is_unsupported_contract(tmp_path):
    event_rows = [_event_row("AAA_2024-01-05", "AAA", datetime(2024, 1, 5))]
    chain_rows = [_chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0)]
    conn, store, snap = _combined_snapshot(tmp_path, event_rows, chain_rows)
    repo = Repository(conn, store)
    ref = EventRef(event_id="AAA_2024-01-05",
                   calendar_revision=snap.table_versions["earnings_events"].dataset_version_id)
    query = _chain_query(event_ref=ref, quote_policy_ref="some_other_policy.v1")
    with pytest.raises(DataError) as err:
        repo.get_chain(query, snap)
    assert err.value.code == "UNSUPPORTED_CONTRACT"


def test_expiry_interval_start_inclusive_excludes_earlier_expiries():
    row_early = _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 1, 20), 100.0)
    row_late = _chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 100.0)
    from engine.v2.contracts.data import TimeInterval
    interval = TimeInterval(column="expiry", start_inclusive="2024-02-01")
    assert chains._within_expiry_interval(row_early, interval) is False
    assert chains._within_expiry_interval(row_late, interval) is True


def _securities_only_snapshot(tmp_path):
    sec = contract_for("securities")
    sec_ref = contract_ref_for(sec)
    conn, clock, store = catalog_and_store(tmp_path)
    row = dict(ticker="AAA", year=2024, first_date=None, last_date=None, mcap_usd=1.5e9,
              mcap_log=21.1, mcap_raw=1.5, mcap_unit_era="billions", mcap_quantized=False,
              n_obs=250, src="orats")
    record = publish_and_inspect(store, sec, sec_ref, [row], "2024")
    snap = commit_tables(conn, clock, {"securities": [record]}, {"securities": sec})
    return conn, store, snap


def test_get_event_without_earnings_events_table_is_contract_mismatch(tmp_path):
    conn, store, snap = _securities_only_snapshot(tmp_path)
    repo = Repository(conn, store)
    ref = EventRef(event_id="AAA_2024-01-05", calendar_revision="dsv_" + "0" * 32)
    with pytest.raises(DataError) as err:
        repo.get_event(ref, snap)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_get_event_zero_rows_is_contract_mismatch(tmp_path):
    conn, store, snap = _events_snapshot(
        tmp_path, [_event_row("AAA_2024-01-05", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    revision = snap.table_versions["earnings_events"].dataset_version_id
    ref = EventRef(event_id="BBB_2024-01-05", calendar_revision=revision)
    with pytest.raises(DataError) as err:
        repo.get_event(ref, snap)
    assert err.value.code == "CONTRACT_MISMATCH"


def test_get_chain_without_option_chains_table_is_contract_mismatch(tmp_path):
    conn, store, snap = _securities_only_snapshot(tmp_path)
    repo = Repository(conn, store)
    query = _chain_query(event_ref=EventRef(event_id="x", calendar_revision="y"))
    with pytest.raises(DataError) as err:
        repo.get_chain(query, snap)
    assert err.value.code == "CONTRACT_MISMATCH"
