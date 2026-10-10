"""Focused tests for ``nightly_calendar_inputs`` (cutover PR-6 slice 4a.2).

Three public helpers over a real catalog + ArtifactStore snapshot with
deterministic synthetic fragments: the pinned SPY decision calendar, the
exact-session listed candidate expiries, and one row's exact raw-close spot,
native resolved expiry and independent planned exit. No market data, no
providers, no data/fixtures contents.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pandas as pd
import pytest

from engine.v2.contracts import PriceQuery
from engine.v2.data.errors import DataError, fail as data_fail
from engine.v2.data.price_history_table import PRICE_HISTORY_CONTRACT
from engine.v2.data.repository import Repository
from engine.v2.domain import generation
from engine.v2.domain.generation import GeometryRefusal
from engine.v2.foundation.market_calendar import CalendarSessions
from engine.v2.ops import nightly_calendar_inputs as nci
from engine.v2.ops.errors import OpsError
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_score_batch import NativeScoreBatchRowRefusal
from engine.v2.ops.nightly_calendar_inputs import (
    scan_calendar_row_inputs,
    scan_candidate_expiries,
    scan_decision_calendar,
)
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    fake_hash,
    publish_and_inspect,
)

_EVENTS = contract_for("earnings_events")
_EVENTS_REF = contract_ref_for(_EVENTS)
_CHAINS = contract_for("option_chains")
_CHAINS_REF = contract_ref_for(_CHAINS)
_PH_REF = contract_ref_for(PRICE_HISTORY_CONTRACT)

_PH_NAME = "price_history"
_CHAINS_NAME = "option_chains"
_EVENTS_NAME = "earnings_events"

_SESSION = "2024-01-05"
_EVENT = "2024-01-16"
_STRADDLE_KEY = BoardRequest("AAA", "STR-THRU", pd.Timestamp(_EVENT), "BMO")
_RUNUP_KEY = BoardRequest("AAA", "STR-RUNUP", pd.Timestamp(_EVENT), "BMO")
_CALENDAR = CalendarSessions(
    days=("2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09",
          "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-16", "2024-01-17",
          "2024-01-18", "2024-01-19"),
    observed_through=_SESSION)
_BAD_DAYS = (None, True, 20240105, pd.NaT, "not-a-date", "2024-01-05T01:00:00",
             "2024-01-05T00:00:00Z", pd.Timestamp("2024-01-05", tz="UTC"))


def _price_row(ticker: str, date: str, *, close_raw=10.0, close_adj=9.0) -> dict:
    return dict(
        ticker=ticker,
        date=date,
        close_adj=close_adj,
        close_raw=close_raw,
        high_raw=close_raw,
        retrieved_at="2024-01-01T00:00:00Z",
        deleted=False,
        source_kind="yfinance",
        source_hash=fake_hash(f"px-{ticker}-{date}"),
        capture_id=f"cap-{ticker}-{date}",
    )


def _chain_row(ticker: str, obs_date: datetime, expiry: datetime, strike: float, *,
               right="C", bid=1.0, ask=1.2) -> dict:
    return dict(
        ticker=ticker,
        obs_date=obs_date,
        year=obs_date.year,
        expiry=expiry,
        dte=(expiry - obs_date).days,
        strike=strike,
        right=right,
        bid=bid,
        ask=ask,
        mid=1.1,
        iv=30.0,
        delta=0.5,
        spot=100.0,
        src="orats",
        src_file="f.parquet",
        chain_kind="entry",
        volume=None,
        open_interest=None,
        bid_size=None,
        ask_size=None,
        quote_repaired=False,
    )


def _event_row(ticker: str, event_date: str, session: str, *, event_id=None) -> dict:
    d = pd.Timestamp(event_date).to_pydatetime()
    return dict(
        event_id=event_id or f"{ticker}_{d.date()}",
        ticker=ticker,
        event_date=d,
        year=d.year,
        session=session,
        session_src="orats",
        annc_tod=None,
        src_orats=False,
        src_oquants=False,
        src_nasdaq=False,
        src_yfinance=False,
        date_agree=True,
        date_conflict=False,
        updated_at=None,
        event_cluster_id=None,
        claim_count=None,
        reconciliation=None,
    )


def _sorted_rows(contract, rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: tuple(row[field] for field in contract.primary_key))


def _snapshot(tmp_path, *, price=None, chains=None, events=None):
    conn, clock, store = catalog_and_store(tmp_path)
    tables: dict[str, list] = {}
    contracts: dict[str, object] = {}
    if price is not None:
        tables[_PH_NAME] = [
            publish_and_inspect(store, PRICE_HISTORY_CONTRACT, _PH_REF,
                                _sorted_rows(PRICE_HISTORY_CONTRACT, rows), ticker)
            for ticker, rows in sorted(price.items())
        ]
        contracts[_PH_NAME] = PRICE_HISTORY_CONTRACT
    if chains is not None:
        tables[_CHAINS_NAME] = [
            publish_and_inspect(store, _CHAINS, _CHAINS_REF, _sorted_rows(_CHAINS, chains), "2024")
        ]
        contracts[_CHAINS_NAME] = _CHAINS
    if events is not None:
        by_year: dict[str, list[dict]] = {}
        for row in events:
            by_year.setdefault(str(row["year"]), []).append(row)
        tables[_EVENTS_NAME] = [
            publish_and_inspect(store, _EVENTS, _EVENTS_REF, _sorted_rows(_EVENTS, rows), year)
            for year, rows in sorted(by_year.items())
        ]
        contracts[_EVENTS_NAME] = _EVENTS
    snapshot = commit_tables(conn, clock, tables, contracts, scope="test")
    return Repository(conn, store), snapshot


def _common_pair(ticker: str, obs: datetime, expiry: datetime, strike: float) -> list[dict]:
    return [
        _chain_row(ticker, obs, expiry, strike, right="C"),
        _chain_row(ticker, obs, expiry, strike, right="P"),
    ]


def _domain_chain_rows() -> list[dict]:
    obs = datetime(2024, 1, 5)
    return [
        _chain_row("AAA", obs, datetime(2024, 1, 19), 100.0, right="C", bid=0.0, ask=0.0),
        _chain_row("AAA", obs, datetime(2024, 1, 19), 100.0, right="P", bid=0.0, ask=0.0),
        _chain_row("AAA", obs, datetime(2024, 1, 19), 105.0, right="P", bid=None, ask=None),
        _chain_row("AAA", obs, datetime(2024, 2, 16), 100.0, right="C"),
        _chain_row("AAA", obs, datetime(2024, 3, 15), 90.0, right="P", bid=None, ask=None),
        _chain_row("AAA", datetime(2024, 1, 4), datetime(2024, 1, 19), 100.0, right="C"),
        _chain_row("AAA", datetime(2024, 1, 4), datetime(2024, 1, 19), 100.0, right="P"),
        _chain_row("BBB", obs, datetime(2024, 1, 19), 100.0, right="C"),
        _chain_row("BBB", obs, datetime(2024, 1, 19), 100.0, right="P"),
    ]


class _ProjectedBatch:
    """A scan batch carrying exactly the projected ``(right, strike, expiry)``
    row dicts it was given -- synthetic malformed input the physical publish
    path itself would refuse."""

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def to_pylist(self) -> list[dict]:
        return self._rows


def _stage_projected_chain_rows(monkeypatch, repository, rows: list[dict]):
    original = repository.scan

    def staged(query, *, table_name):
        if table_name != _CHAINS_NAME:
            yield from original(query, table_name=table_name)
            return
        yield _ProjectedBatch(rows)

    monkeypatch.setattr(repository, "scan", staged)


def test_decision_calendar_pins_exact_spy_query_and_keeps_source_observed_max(
        tmp_path, monkeypatch):
    rows = [
        _price_row("SPY", "2024-01-08", close_raw=105.0),
        _price_row("SPY", "2024-01-03", close_raw=101.0),
        _price_row("SPY", "2024-01-05", close_raw=103.0),
        _price_row("SPY", "2024-01-04", close_raw=102.0),
    ]
    repository, snapshot = _snapshot(tmp_path, price={"SPY": rows})
    queries, factory_calls = [], []
    real_series, real_factory = nci.get_price_series, nci.build_calendar_sessions

    def spy_series(repo, query, snap):
        queries.append((repo, query, snap))
        return real_series(repo, query, snap)

    def spy_factory(observed, *, event_through):
        factory_calls.append((observed, event_through))
        return real_factory(observed, event_through=event_through)

    monkeypatch.setattr(nci, "get_price_series", spy_series)
    monkeypatch.setattr(nci, "build_calendar_sessions", spy_factory)

    calendar = scan_decision_calendar(repository, snapshot, decision_session=_SESSION,
                                      event_through=_EVENT)

    assert len(queries) == 1
    assert queries[0][1] == PriceQuery(ticker="SPY", session_date=_SESSION,
                                       observation_ceiling=_SESSION, lookback_sessions=0)
    assert factory_calls == [((("2024-01-03"), "2024-01-04", "2024-01-05"), _EVENT)]
    assert calendar.observed_through == _SESSION
    assert calendar.observed_through != calendar.days[-1]
    assert calendar.days == ("2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
                             "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
                             "2024-01-16", "2024-01-17")


def test_decision_calendar_without_spy_rows_is_contract_mismatch(tmp_path):
    repository, snapshot = _snapshot(tmp_path, price={"QQQ": [_price_row("QQQ", "2024-01-05")]})
    with pytest.raises(DataError) as exc:
        scan_decision_calendar(repository, snapshot, decision_session=_SESSION,
                               event_through=_EVENT)
    assert exc.value.code == "CONTRACT_MISMATCH"


def test_decision_calendar_without_price_table_is_contract_mismatch(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path, chains=_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0))
    with pytest.raises(DataError) as exc:
        scan_decision_calendar(repository, snapshot, decision_session=_SESSION,
                               event_through=_EVENT)
    assert exc.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("strategy", ["STR-THRU", "STR-RUNUP"])
def test_candidate_expiries_straddle_domain_requires_common_strike_and_one_bounded_scan(
        tmp_path, monkeypatch, strategy):
    repository, snapshot = _snapshot(tmp_path, chains=_domain_chain_rows())
    calls = []
    original = repository.scan

    def recorded(query, *, table_name):
        calls.append((query, table_name))
        return original(query, table_name=table_name)

    monkeypatch.setattr(repository, "scan", recorded)
    key = replace(_STRADDLE_KEY, strategy=strategy)

    result = scan_candidate_expiries(repository, snapshot, key, decision_session=_SESSION)

    assert result == ("2024-01-19",)
    assert len(calls) == 1
    query, table = calls[0]
    assert table == _CHAINS_NAME
    assert query.snapshot_id == snapshot.snapshot_id
    assert query.table_contract_ref == snapshot.table_versions[_CHAINS_NAME].table_contract_ref
    assert query.columns == ("right", "strike", "expiry")
    assert query.time_interval is None
    assert [(p.column, p.operator, p.values) for p in query.key_filter] == [
        ("ticker", "eq", ("AAA",)), ("obs_date", "eq", (_SESSION,))]
    assert query.order_by == tuple(_CHAINS.primary_key)
    assert query.max_batch_rows == min(_CHAINS.maximum_batch_rows, 50_000, query.max_result_rows)
    assert query.max_result_rows == query.max_batch_rows


@pytest.mark.parametrize("strategy", ["DYN-SV", "TWIN-P"])
def test_candidate_expiries_put_menu_domain_accepts_listed_puts_without_usable_quotes(
        tmp_path, strategy):
    repository, snapshot = _snapshot(tmp_path, chains=_domain_chain_rows())
    key = replace(_STRADDLE_KEY, strategy=strategy)

    result = scan_candidate_expiries(repository, snapshot, key, decision_session=_SESSION)

    assert result == ("2024-01-19", "2024-03-15")


def test_candidate_expiries_empty_domain_returns_empty_tuple(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path,
        chains=[_chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0, right="C")])
    key = replace(_STRADDLE_KEY, strategy="DYN-SV")
    assert scan_candidate_expiries(repository, snapshot, key, decision_session=_SESSION) == ()


def test_candidate_expiries_missing_option_table_is_contract_mismatch(tmp_path):
    repository, snapshot = _snapshot(tmp_path, price={"AAA": [_price_row("AAA", _SESSION)]})
    with pytest.raises(DataError) as exc:
        scan_candidate_expiries(repository, snapshot, _STRADDLE_KEY, decision_session=_SESSION)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"table_name": _CHAINS_NAME}


@pytest.mark.parametrize("code", ["RESULT_LIMIT_EXCEEDED", "QUERY_NOT_BOUNDED"])
def test_candidate_expiries_scan_failure_propagates(tmp_path, monkeypatch, code):
    repository, snapshot = _snapshot(
        tmp_path,
        price={"AAA": [_price_row("AAA", _SESSION)]},
        chains=_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0))
    problem = data_fail(code, "synthetic option scan failure")

    def broken(query, *, table_name):
        raise problem

    monkeypatch.setattr(repository, "scan", broken)
    with pytest.raises(DataError) as exc:
        scan_candidate_expiries(repository, snapshot, _STRADDLE_KEY, decision_session=_SESSION)
    assert exc.value is problem


_PRE_SESSION = datetime(2024, 1, 4)
_MALFORMED_ROWS = [
    ("null_expiry", dict(expiry=None, strike=100.0, right="C")),
    ("null_strike", dict(expiry=_PRE_SESSION, strike=None, right="C")),
    ("null_right", dict(expiry=_PRE_SESSION, strike=100.0, right=None)),
    ("invalid_right", dict(expiry=_PRE_SESSION, strike=100.0, right="X")),
    ("nan_strike", dict(expiry=_PRE_SESSION, strike=float("nan"), right="P")),
    ("infinite_strike", dict(expiry=_PRE_SESSION, strike=float("inf"), right="P")),
    ("unconvertible_strike", dict(expiry=_PRE_SESSION, strike="not-a-strike", right="C")),
]


@pytest.mark.parametrize("row", [row for _, row in _MALFORMED_ROWS],
                         ids=[name for name, _ in _MALFORMED_ROWS])
def test_candidate_expiries_rejects_malformed_row_before_expiry_filtering(
        tmp_path, monkeypatch, row):
    repository, snapshot = _snapshot(
        tmp_path,
        chains=_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0))
    _stage_projected_chain_rows(monkeypatch, repository, [row])
    with pytest.raises(DataError) as exc:
        scan_candidate_expiries(repository, snapshot, _STRADDLE_KEY, decision_session=_SESSION)
    assert exc.value.code == "CONTRACT_MISMATCH"


def test_candidate_expiries_normalizes_right_once_and_still_filters_pre_session(
        tmp_path, monkeypatch):
    repository, snapshot = _snapshot(
        tmp_path,
        chains=_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0))
    _stage_projected_chain_rows(monkeypatch, repository, [
        dict(expiry=datetime(2024, 1, 19), strike=100.0, right="c"),
        dict(expiry=datetime(2024, 1, 19), strike=100.0, right="P"),
        dict(expiry=_PRE_SESSION, strike=100.0, right="P"),
    ])
    assert scan_candidate_expiries(repository, snapshot, _STRADDLE_KEY,
                                   decision_session=_SESSION) == ("2024-01-19",)


def test_calendar_row_inputs_uses_exact_raw_close_and_forwards_pinned_arguments(
        tmp_path, monkeypatch):
    rows = {
        "SPY": [_price_row("SPY", day) for day in
                ("2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08")],
        "AAA": [_price_row("AAA", "2024-01-04", close_raw=11.0, close_adj=111.0),
                _price_row("AAA", _SESSION, close_raw=12.5, close_adj=99.9)],
    }
    chains = (_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 16), 100.0)
              + [_chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 1, 16), 105.0, right="P")]
              + _common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 2, 16), 110.0)
              + _common_pair("BBB", datetime(2024, 1, 5), datetime(2024, 1, 16), 100.0))
    events = [_event_row("AAA", _EVENT, "BMO", event_id="persisted-aaa")]
    repository, snapshot = _snapshot(tmp_path, price=rows, chains=chains, events=events)
    calendar = scan_decision_calendar(repository, snapshot, decision_session=_SESSION,
                                      event_through=_EVENT)
    assert calendar.observed_through == _SESSION

    queries, forwards = [], []
    real_series, real_row = nci.get_price_series, nci.scan_calendar_row

    def spy_series(repo, query, snap):
        queries.append(query)
        return real_series(repo, query, snap)

    def spy_row(repo, snap, key, **kwargs):
        forwards.append((repo, snap, key, kwargs))
        return real_row(repo, snap, key, **kwargs)

    monkeypatch.setattr(nci, "get_price_series", spy_series)
    monkeypatch.setattr(nci, "scan_calendar_row", spy_row)

    result = scan_calendar_row_inputs(repository, snapshot, _STRADDLE_KEY,
                                      decision_session=_SESSION, calendar=calendar)

    assert queries == [PriceQuery(ticker="AAA", session_date=_SESSION,
                                  observation_ceiling=_SESSION, lookback_sessions=0)]
    assert len(forwards) == 1 and forwards[0][:3] == (repository, snapshot, _STRADDLE_KEY)
    assert forwards[0][3] == {
        "entry_date": _SESSION, "exit_date": _EVENT, "expiry": _EVENT, "spot": 12.5,
        "calendar_observed_through": _SESSION}
    assert result.calendar_row == {
        "event_id": "persisted-aaa", "ticker": "AAA", "event_date": _EVENT, "session": "BMO",
        "entry_date": _SESSION, "exit_date": _EVENT, "expiry": _EVENT, "spot": 12.5,
        "calendar_observed_through": _SESSION}
    assert result.calendar_revision == snapshot.table_versions[_EVENTS_NAME].dataset_version_id
    assert result.calendar_revision != snapshot.calendar_version


def test_calendar_row_inputs_missing_price_table_is_contract_mismatch(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path, chains=_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 16), 100.0))
    with pytest.raises(DataError) as exc:
        scan_calendar_row_inputs(repository, snapshot, _STRADDLE_KEY,
                                 decision_session=_SESSION, calendar=_CALENDAR)
    assert exc.value.code == "CONTRACT_MISMATCH"


def test_calendar_row_inputs_missing_ticker_is_contract_mismatch(tmp_path):
    repository, snapshot = _snapshot(tmp_path, price={"SPY": [_price_row("SPY", _SESSION)]})
    with pytest.raises(DataError) as exc:
        scan_calendar_row_inputs(repository, snapshot, _STRADDLE_KEY,
                                 decision_session=_SESSION, calendar=_CALENDAR)
    assert exc.value.code == "CONTRACT_MISMATCH"


def test_calendar_row_inputs_stale_only_series_is_price_history_not_available(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path, price={"AAA": [_price_row("AAA", "2024-01-04", close_raw=11.0)]})
    with pytest.raises(NativeScoreBatchRowRefusal) as exc:
        scan_calendar_row_inputs(repository, snapshot, _STRADDLE_KEY,
                                 decision_session=_SESSION, calendar=_CALENDAR)
    assert exc.value.code == "PRICE_HISTORY_NOT_AVAILABLE"


def test_calendar_row_inputs_empty_eligible_series_is_price_history_not_available(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path, price={"AAA": [_price_row("AAA", "2024-01-08", close_raw=11.0)]})
    with pytest.raises(NativeScoreBatchRowRefusal) as exc:
        scan_calendar_row_inputs(repository, snapshot, _STRADDLE_KEY,
                                 decision_session=_SESSION, calendar=_CALENDAR)
    assert exc.value.code == "PRICE_HISTORY_NOT_AVAILABLE"


@pytest.mark.parametrize("bad", [None, 0.0, -3.0, float("nan")])
def test_calendar_row_inputs_unusable_raw_close_is_price_history_not_available(tmp_path, bad):
    repository, snapshot = _snapshot(
        tmp_path,
        price={"AAA": [_price_row("AAA", "2024-01-04", close_raw=11.0, close_adj=11.0),
                       _price_row("AAA", _SESSION, close_raw=bad, close_adj=50.0)]},
        chains=_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 16), 100.0),
        events=[_event_row("AAA", _EVENT, "BMO")])
    with pytest.raises(NativeScoreBatchRowRefusal) as exc:
        scan_calendar_row_inputs(repository, snapshot, _STRADDLE_KEY,
                                 decision_session=_SESSION, calendar=_CALENDAR)
    assert exc.value.code == "PRICE_HISTORY_NOT_AVAILABLE"


def test_calendar_row_inputs_option_scan_failure_propagates(tmp_path, monkeypatch):
    repository, snapshot = _snapshot(
        tmp_path,
        price={"AAA": [_price_row("AAA", _SESSION, close_raw=12.5)]},
        chains=_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 16), 100.0))
    problem = data_fail("RESULT_LIMIT_EXCEEDED", "synthetic option scan failure")
    original = repository.scan

    def broken(query, *, table_name):
        if table_name == _CHAINS_NAME:
            raise problem
        return original(query, table_name=table_name)

    monkeypatch.setattr(repository, "scan", broken)
    with pytest.raises(DataError) as exc:
        scan_calendar_row_inputs(repository, snapshot, _STRADDLE_KEY,
                                 decision_session=_SESSION, calendar=_CALENDAR)
    assert exc.value is problem


def test_calendar_row_inputs_propagates_malformed_option_row_contract_mismatch(
        tmp_path, monkeypatch):
    repository, snapshot = _snapshot(
        tmp_path,
        price={"AAA": [_price_row("AAA", _SESSION, close_raw=12.5)]},
        chains=_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 16), 100.0))
    _stage_projected_chain_rows(monkeypatch, repository,
                                [dict(expiry=_PRE_SESSION, strike=100.0, right="X")])
    with pytest.raises(DataError) as exc:
        scan_calendar_row_inputs(repository, snapshot, _STRADDLE_KEY,
                                 decision_session=_SESSION, calendar=_CALENDAR)
    assert exc.value.code == "CONTRACT_MISMATCH"


def test_calendar_row_inputs_empty_candidates_is_no_resolvable_expiry(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path,
        price={"AAA": [_price_row("AAA", _SESSION, close_raw=12.5)]},
        chains=[_chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0, right="C")])
    key = replace(_STRADDLE_KEY, strategy="DYN-SV")
    with pytest.raises(NativeScoreBatchRowRefusal) as exc:
        scan_calendar_row_inputs(repository, snapshot, key,
                                 decision_session=_SESSION, calendar=_CALENDAR)
    assert exc.value.code == "NO_RESOLVABLE_EXPIRY"
    assert exc.value.detail == "no strategy-eligible listed expiry"
    assert exc.value.key is key


def test_calendar_row_inputs_translates_amc_native_no_expiry(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path,
        price={"AAA": [_price_row("AAA", _SESSION, close_raw=12.5)]},
        chains=(_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 10), 100.0)
                + _common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 16), 100.0)))
    key = replace(_STRADDLE_KEY, session="AMC")
    with pytest.raises(NativeScoreBatchRowRefusal) as exc:
        scan_calendar_row_inputs(repository, snapshot, key,
                                 decision_session=_SESSION, calendar=_CALENDAR)
    assert exc.value.code == "NO_RESOLVABLE_EXPIRY"
    assert exc.value.detail == "no strategy-eligible listed expiry"
    assert exc.value.key is key


def test_calendar_row_inputs_translates_runup_dte_no_expiry(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path,
        price={"AAA": [_price_row("AAA", _SESSION, close_raw=12.5)]},
        chains=_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0))
    key = BoardRequest("AAA", "STR-RUNUP", pd.Timestamp("2024-01-31"), "BMO")
    with pytest.raises(NativeScoreBatchRowRefusal) as exc:
        scan_calendar_row_inputs(repository, snapshot, key,
                                 decision_session=_SESSION, calendar=_CALENDAR)
    assert exc.value.code == "NO_RESOLVABLE_EXPIRY"
    assert exc.value.detail == "no strategy-eligible listed expiry"
    assert exc.value.key is key


_UNSUPPORTED_DOMAINS = [
    ("strategy_empty_candidates", replace(_STRADDLE_KEY, strategy="NOT-A-STRATEGY"),
     [_chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0, right="C")]),
    ("strategy_candidates", replace(_STRADDLE_KEY, strategy="NOT-A-STRATEGY"),
     _common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 10), 100.0)),
    ("session_empty_candidates", replace(_STRADDLE_KEY, session="OVD"),
     [_chain_row("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0, right="C")]),
    ("session_candidates", replace(_STRADDLE_KEY, session="OVD"),
     _common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 10), 100.0)),
]


@pytest.mark.parametrize("key, chains",
                         [(key, chains) for _, key, chains in _UNSUPPORTED_DOMAINS],
                         ids=[name for name, _, _ in _UNSUPPORTED_DOMAINS])
def test_calendar_row_inputs_rejects_unsupported_strategy_or_session_before_expiry(
        tmp_path, key, chains):
    """A genuinely empty eligible domain is ``NO_RESOLVABLE_EXPIRY``; an
    unsupported nonempty strategy/session is malformed input and must refuse
    ``INVALID_REQUEST`` -- whether the candidate scan came back empty (only a
    call listed) or populated (the listed expiry is pre-event, the native
    refusal this staging translates)."""
    repository, snapshot = _snapshot(
        tmp_path,
        price={"AAA": [_price_row("AAA", _SESSION, close_raw=12.5)]},
        chains=chains)
    with pytest.raises(OpsError) as exc:
        scan_calendar_row_inputs(repository, snapshot, key,
                                 decision_session=_SESSION, calendar=_CALENDAR)
    assert exc.value.code == "INVALID_REQUEST"


def test_calendar_row_inputs_rejects_unsupported_domains_before_source_reads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source read before strategy/session validation")

    monkeypatch.setattr(nci, "get_price_series", forbidden)
    for key in (replace(_STRADDLE_KEY, strategy="NOT-A-STRATEGY"),
                replace(_STRADDLE_KEY, session="OVD")):
        with pytest.raises(OpsError) as exc:
            scan_calendar_row_inputs(None, None, key,
                                     decision_session=_SESSION, calendar=_CALENDAR)
        assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("code", [
    "EXPIRY_NOT_LISTED:2024-02-16", "NO_EXPIRY_ON_OR_AFTER", "UNRELATED_GEOMETRY"])
def test_calendar_row_inputs_propagates_untranslated_geometry_refusals(
        tmp_path, monkeypatch, code):
    repository, snapshot = _snapshot(
        tmp_path,
        price={"AAA": [_price_row("AAA", _SESSION, close_raw=12.5)]},
        chains=_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0))
    boom = GeometryRefusal(code)

    def refuse(*args, **kwargs):
        raise boom

    monkeypatch.setattr(generation, "resolve_expiry", refuse)
    with pytest.raises(GeometryRefusal) as exc:
        scan_calendar_row_inputs(repository, snapshot, _STRADDLE_KEY,
                                 decision_session=_SESSION, calendar=_CALENDAR)
    assert exc.value is boom


def test_calendar_row_inputs_resolves_runup_dte_through_public_resolver(tmp_path, monkeypatch):
    repository, snapshot = _snapshot(
        tmp_path,
        price={"AAA": [_price_row("AAA", _SESSION, close_raw=12.5)]},
        chains=(_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 2, 1), 100.0)
                + _common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 2, 5), 100.0)),
        events=[_event_row("AAA", _EVENT, "BMO")])
    calls = []
    real = generation.resolve_expiry

    def recorded(strategy, inputs, expiries):
        calls.append((strategy, dict(inputs), list(expiries)))
        return real(strategy, inputs, expiries)

    monkeypatch.setattr(generation, "resolve_expiry", recorded)
    result = scan_calendar_row_inputs(repository, snapshot, _RUNUP_KEY,
                                      decision_session=_SESSION, calendar=_CALENDAR)

    assert calls == [("STR-RUNUP",
                      {"event_date": _RUNUP_KEY.event_date, "session": "BMO",
                       "entry_date": _SESSION, "quote_date": _SESSION},
                      ["2024-02-01", "2024-02-05"])]
    assert result.calendar_row["expiry"] == "2024-02-05"
    assert result.calendar_row["exit_date"] == "2024-01-12"


def test_calendar_row_inputs_keeps_resolver_off_the_module_import_path():
    assert not hasattr(nci, "resolve_expiry")
    assert not hasattr(nci, "GeometryRefusal")


@pytest.mark.parametrize("session, expiry, exit_date", [
    ("BMO", "2024-01-16", "2024-01-16"),
    ("AMC", "2024-01-19", "2024-01-17"),
])
def test_calendar_row_inputs_event_day_expiry_follows_session(
        tmp_path, session, expiry, exit_date):
    repository, snapshot = _snapshot(
        tmp_path,
        price={"AAA": [_price_row("AAA", _SESSION, close_raw=12.5)]},
        chains=(_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 16), 100.0)
                + _common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0)),
        events=[_event_row("AAA", _EVENT, session)])
    key = replace(_STRADDLE_KEY, session=session)
    result = scan_calendar_row_inputs(repository, snapshot, key,
                                      decision_session=_SESSION, calendar=_CALENDAR)
    assert result.calendar_row["expiry"] == expiry
    assert result.calendar_row["exit_date"] == exit_date


def test_calendar_row_inputs_keeps_expiry_beyond_horizon_with_independent_exit(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path,
        price={"AAA": [_price_row("AAA", _SESSION, close_raw=12.5)]},
        chains=(_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0)
                + _common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 7, 19), 100.0)),
        events=[_event_row("AAA", _EVENT, "BMO")])
    result = scan_calendar_row_inputs(repository, snapshot, _RUNUP_KEY,
                                      decision_session=_SESSION, calendar=_CALENDAR)
    assert result.calendar_row["expiry"] == "2024-07-19"
    assert result.calendar_row["exit_date"] == "2024-01-12"
    assert result.calendar_row["exit_date"] < result.calendar_row["expiry"]


def test_decision_calendar_rejects_bad_days_before_reading(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source read before validation")

    monkeypatch.setattr(nci, "get_price_series", forbidden)
    for field in ("decision_session", "event_through"):
        for bad in _BAD_DAYS:
            staged = {"decision_session": _SESSION, "event_through": _EVENT}
            staged[field] = bad
            with pytest.raises(OpsError) as exc:
                scan_decision_calendar(None, None, **staged)
            assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("bad", _BAD_DAYS)
def test_candidate_expiries_rejects_bad_session_before_scan(bad):
    with pytest.raises(OpsError) as exc:
        scan_candidate_expiries(None, None, _STRADDLE_KEY, decision_session=bad)
    assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("key", [
    None, "not-a-key",
    replace(_STRADDLE_KEY, ticker=""),
    replace(_STRADDLE_KEY, strategy=" "),
    replace(_STRADDLE_KEY, session=None),
])
def test_candidate_expiries_rejects_bad_key_before_scan(key):
    with pytest.raises(OpsError) as exc:
        scan_candidate_expiries(None, None, key, decision_session=_SESSION)
    assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("key", [
    replace(_STRADDLE_KEY, strategy="NOT-A-STRATEGY"),
    replace(_STRADDLE_KEY, session="OVD"),
])
def test_candidate_expiries_rejects_unsupported_domains_before_expiry_scan(
        tmp_path, monkeypatch, key):
    """Direct-entrypoint guard matching ``scan_calendar_row_inputs``: an
    unsupported nonempty strategy/session is malformed input and must refuse
    ``INVALID_REQUEST`` before the ``option_chains`` scan, never staged as a
    legitimately empty eligible expiry domain. The repository scan is made to
    raise on call, so reaching it would fail the test rather than return ()."""
    repository, snapshot = _snapshot(
        tmp_path,
        chains=_common_pair("AAA", datetime(2024, 1, 5), datetime(2024, 1, 19), 100.0))

    def forbidden(query, *, table_name):
        raise AssertionError("expiry scan before strategy/session validation")

    monkeypatch.setattr(repository, "scan", forbidden)
    with pytest.raises(OpsError) as exc:
        scan_candidate_expiries(repository, snapshot, key, decision_session=_SESSION)
    assert exc.value.code == "INVALID_REQUEST"


def test_calendar_row_inputs_validates_before_source_scans(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source read before validation")

    monkeypatch.setattr(nci, "get_price_series", forbidden)
    for bad in _BAD_DAYS:
        with pytest.raises(OpsError) as exc:
            scan_calendar_row_inputs(None, None, _STRADDLE_KEY,
                                     decision_session=bad, calendar=_CALENDAR)
        assert exc.value.code == "INVALID_REQUEST"
    with pytest.raises(OpsError) as exc:
        scan_calendar_row_inputs(None, None, "not-a-key",
                                 decision_session=_SESSION, calendar=_CALENDAR)
    assert exc.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("bad_event_date", [
    pd.Timestamp("2024-01-16", tz="UTC"),
    pd.Timestamp("2024-01-16 01:00:00"),
    pd.NaT,
    "not-a-date",
    20240116,
])
def test_calendar_row_inputs_rejects_bad_event_date_before_source(bad_event_date, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source read before event_date validation")

    monkeypatch.setattr(nci, "get_price_series", forbidden)
    key = replace(_STRADDLE_KEY, event_date=bad_event_date)
    with pytest.raises(OpsError) as exc:
        scan_calendar_row_inputs(None, None, key,
                                 decision_session=_SESSION, calendar=_CALENDAR)
    assert exc.value.code == "INVALID_REQUEST"
