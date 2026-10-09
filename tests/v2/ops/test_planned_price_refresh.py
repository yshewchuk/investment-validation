"""Planned watchlists reach genuine Tier-1 and snapshot-backed price history.

Only the provider-facing Fetcher and the final plan/submit/serve boundaries are
faked. Planning, rooted cache discovery, capture, and repository reads are real;
all inputs and catalogs live under pytest's temporary directory.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from engine import paths
from engine.data import fetch
from engine.data.pulls import price_refresh
from engine.v2.contracts import PriceQuery
from engine.v2.data import reference_catalog
from engine.v2.data.reference_catalog import ReferenceInput
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore
from engine.v2.ops import nightly_trigger
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.legacy_adapter import invoke_price_refresh
from tests.data_scan_support import commit_tables, publish_and_inspect
from tests.ops_support import FakeClock
from tests.test_v2_ops_price_history import (
    _SEC,
    _SEC_REF,
    _scan_price_history,
    _securities_row,
    _write_tier1,
)

SESSION = "2026-09-11"
FETCHED_AT = SESSION + "T20:00:00+00:00"
PRICES = {"2026-09-10": 100.0, SESSION: 101.0}


def _unexpected_legacy_universe(*args, **kwargs):
    pytest.fail("planned refresh must not consult the legacy events/inventory universe")


@pytest.fixture(autouse=True)
def isolated_legacy_sources(tmp_path, monkeypatch):
    """A regression must fail safely rather than read the checkout's data."""
    global_root = tmp_path / "global-source"
    monkeypatch.setattr(paths, "RAW_FETCH", global_root / "data" / "raw" / "fetch")
    monkeypatch.setattr(paths, "RAW_YF", global_root / "earnings_predictions" /
                        "data" / "raw" / "yfinance")
    monkeypatch.setattr(price_refresh, "load_events", _unexpected_legacy_universe)
    monkeypatch.setattr(price_refresh, "load_price_universe", _unexpected_legacy_universe)
    return global_root


@pytest.fixture
def shadow_root(tmp_path):
    root = tmp_path / "selected-source"
    ops_root = root / "data" / "operations"
    ops_root.mkdir(parents=True)
    clock = FakeClock()
    conn = open_catalog(ops_root / "catalog.sqlite", clock=clock)
    store = ArtifactStore(ops_root)
    record = publish_and_inspect(
        store, _SEC, _SEC_REF, [_securities_row("AAPL", 2024)], "2024")
    base = commit_tables(conn, clock, {"securities": [record]}, {"securities": _SEC},
                         scope="shadow", receipt_id="base-r1")
    calendar = store.publish_bytes(b"synthetic-calendar", schema_ref="calendar.v1")
    reference = ReferenceInput(kind="calendar", legacy_path="calendar.csv",
                               object_id=calendar.artifact_id, content_hash=calendar.content_hash,
                               byte_size=calendar.byte_size)
    with transaction(conn):
        reference_catalog.insert_reference_inputs(conn, "base-r1", [reference])
    yield SimpleNamespace(root=root, conn=conn, clock=clock, store=store, base=base)
    conn.close()


@pytest.fixture
def install_fetcher(monkeypatch):
    def install(root, *, failures=None, fetched_at=FETCHED_AT):
        calls = []
        failures = failures or {}
        expected_root = root / "data" / "raw" / "fetch"

        class SyntheticFetcher:
            def __init__(self, *, root):
                assert root == expected_root

            def fetch(self, source, endpoint, params, *, live, note):
                ticker = params["ticker"]
                assert (source, endpoint, params, live, note) == (
                    "yfinance", "history", {"ticker": ticker, "period": "max"},
                    True, "price-refresh")
                calls.append(ticker)
                failure = failures.get(ticker)
                if isinstance(failure, Exception):
                    raise failure
                if failure is not None:
                    return SimpleNamespace(status=failure)
                _write_tier1(root, ticker, PRICES, key="fetched-" + ticker,
                             fetched_at=fetched_at)
                return SimpleNamespace(status=200)

        monkeypatch.setattr(fetch, "Fetcher", SyntheticFetcher)
        return calls

    return install


def _read_report(root):
    return json.loads(root.joinpath(
        *nightly_trigger.STATE_DIR, SESSION + ".price_refresh.json").read_text())


def _assert_prices(state, snapshot_id, ticker):
    rows = _scan_price_history(state.conn, state.store, snapshot_id, ticker=ticker)
    assert dict(zip(rows["date"], rows["close_adj"], strict=True)) == PRICES
    assert set(rows["deleted"]) == {False}


def test_never_fetched_planned_ticker_is_fetched_and_captured(shadow_root, install_fetcher):
    state = shadow_root
    calls = install_fetcher(state.root)
    assert not (state.root / "data" / "raw").exists()
    assert not (state.root / "earnings_predictions").exists()
    assert "earnings_events" not in state.base.table_versions

    snapshot_id = nightly_trigger._refresh_plan_prices(
        state.root, SESSION, ("ZZNEW",), state.clock, state.base.snapshot_id)

    assert calls == ["SPY", "ZZNEW"]
    assert snapshot_id != state.base.snapshot_id
    _assert_prices(state, snapshot_id, "ZZNEW")
    _assert_prices(state, snapshot_id, "SPY")
    snapshot = Repository(state.conn, state.store).resolve(snapshot_id)
    assert snapshot.table_versions["securities"] == state.base.table_versions["securities"]
    report = _read_report(state.root)
    assert report["schema_version"] == "nightly_price_refresh.v1.0"
    assert report["plan"] == {
        "session": SESSION, "daily": ["SPY", "ZZNEW"], "monthly": [],
        "skipped_already_fetched": [],
    }
    assert report["report"]["counts"]["fetched"] == 2
    assert report["capture"]["result_snapshot_id"] == snapshot_id
    assert report["missing_price_history"] == []
    assert not list(state.root.joinpath(*nightly_trigger.STATE_DIR).glob("*.tmp"))


@pytest.mark.parametrize("failure,error_class", [(OSError("unavailable"), "OSError"),
                                                (404, "http_404")])
def test_ordinary_fetch_failure_is_nonfatal_and_reported(
        shadow_root, install_fetcher, failure, error_class):
    state = shadow_root
    calls = install_fetcher(state.root, failures={"BROKEN": failure})

    snapshot_id = nightly_trigger._refresh_plan_prices(
        state.root, SESSION, ("BROKEN", "HEALTHY"), state.clock, state.base.snapshot_id)

    assert calls == ["BROKEN", "HEALTHY", "SPY"]
    _assert_prices(state, snapshot_id, "HEALTHY")
    _assert_prices(state, snapshot_id, "SPY")
    report = _read_report(state.root)
    assert report["report"]["failed"] == [
        {"ticker": "BROKEN", "group": "daily", "error_class": error_class}]
    assert report["report"]["counts"]["fetched"] == 2
    assert report["missing_price_history"] == [
        {"ticker": "BROKEN", "reason_code": "PRICE_HISTORY_NOT_AVAILABLE"}]
    assert report["capture"]["result_snapshot_id"] == snapshot_id
    assert state.conn.execute(
        "SELECT snapshot_id FROM data_snapshot_heads WHERE scope='shadow'"
    ).fetchone()["snapshot_id"] == snapshot_id


def test_selected_root_controls_daily_cache_and_fetch_writes(
        tmp_path, isolated_legacy_sources, install_fetcher):
    root = tmp_path / "selected-source"
    global_root = isolated_legacy_sources
    for ticker in ("NEW", "SPY"):
        _write_tier1(global_root, ticker, PRICES, key="global-" + ticker,
                     fetched_at=FETCHED_AT)
    _write_tier1(root, "CACHED", PRICES, key="cached", fetched_at=FETCHED_AT)
    _write_tier1(root, "OLD", PRICES, key="old", fetched_at="2026-09-10T20:00:00+00:00")
    _write_tier1(root, "UNRELATED", PRICES, key="unrelated",
                 fetched_at="2026-09-10T20:00:00+00:00")
    global_before = {p: p.read_bytes() for p in global_root.rglob("*") if p.is_file()}
    calls = install_fetcher(root)

    first = invoke_price_refresh(
        SESSION, planned_tickers=("CACHED", "NEW", "OLD"), source_root=root)

    assert first["plan"]["daily"] == ["NEW", "OLD", "SPY"]
    assert first["plan"]["monthly"] == []
    assert first["plan"]["skipped_already_fetched"] == ["CACHED"]
    assert calls == ["NEW", "OLD", "SPY"]
    second = invoke_price_refresh(
        SESSION, planned_tickers=("CACHED", "NEW", "OLD"), source_root=root)
    assert second["plan"]["daily"] == []
    assert second["plan"]["skipped_already_fetched"] == ["CACHED", "NEW", "OLD", "SPY"]
    assert second["report"]["counts"]["fetched"] == 0
    assert calls == ["NEW", "OLD", "SPY"]
    assert {p: p.read_bytes() for p in global_root.rglob("*") if p.is_file()} == global_before


def test_planned_dry_run_never_constructs_fetcher_or_writes(tmp_path, monkeypatch):
    root = tmp_path / "dry-source"

    def unexpected_fetcher(*args, **kwargs):
        pytest.fail("dry run constructed a Fetcher")

    monkeypatch.setattr(fetch, "Fetcher", unexpected_fetcher)
    result = invoke_price_refresh(
        SESSION, planned_tickers=("ZZNEW",), source_root=root, dry_run=True)

    assert result == {"plan": {
        "session": SESSION, "daily": ["SPY", "ZZNEW"], "monthly": [],
        "skipped_already_fetched": [],
    }, "report": None}
    assert not root.exists()


@pytest.mark.parametrize("use_population", [False, True])
def test_production_submit_prepares_exact_watchlist_before_planning(
        shadow_root, install_fetcher, monkeypatch, use_population):
    state = shadow_root
    calls = install_fetcher(state.root)
    population_path = state.root.joinpath(*nightly_trigger.STATE_DIR,
                                          nightly_trigger.QUALIFICATION_POPULATION)
    population_path.parent.mkdir(parents=True)
    population_path.write_text(json.dumps([
        "ZZNEW|CALL|2026-09-18", "HEALTHY|CALL|2026-09-18", "ZZNEW|PUT|2026-09-18"]))
    tickers = () if use_population else ("ZZNEW",)
    expected = ("HEALTHY", "ZZNEW") if use_population else tickers
    context = () if use_population else ("CONTEXT_ONLY",)
    planned = []

    def final_plan(root, as_of, actual_tickers, actual_context, clock, *, full_run,
                   expected_shadow_snapshot_id):
        assert (root, as_of, actual_tickers, actual_context, clock, full_run) == (
            state.root, SESSION, expected, context or expected, state.clock, True)
        report = _read_report(root)
        assert expected_shadow_snapshot_id == report["capture"]["result_snapshot_id"]
        assert expected_shadow_snapshot_id != state.base.snapshot_id
        assert report["plan"]["daily"] == sorted((*expected, "SPY"))
        assert report["missing_price_history"] == []
        for ticker in (*expected, "SPY"):
            _assert_prices(state, expected_shadow_snapshot_id, ticker)
        planned.append(expected_shadow_snapshot_id)
        return "synthetic-plan-ref"

    monkeypatch.setattr(nightly_trigger, "_default_plan", final_plan)
    receipt = nightly_trigger._submit_plan(
        state.root, SESSION, tickers=tickers, context_tickers=context, clock=state.clock,
        plan_fn=None, submit_fn=lambda *args: None, serve_fn=lambda *args: "completed",
        ensure_snapshot_fn=lambda *args: ("ready", state.base.snapshot_id),
        full_run=True, prior=None, plan_ref=None)

    assert receipt.status == "completed"
    assert receipt.plan_ref == "synthetic-plan-ref"
    assert len(planned) == 1
    assert calls == sorted((*expected, "SPY"))


def test_first_fetch_next_morning_keeps_honest_retrieval_time(
        shadow_root, install_fetcher):
    state = shadow_root
    install_fetcher(state.root, fetched_at="2026-09-12T06:00:00+00:00")
    snapshot_id = nightly_trigger._refresh_plan_prices(
        state.root, SESSION, ("ZZNEW",), state.clock, state.base.snapshot_id)
    repository = Repository(state.conn, state.store)
    rows = repository.get_price_series(PriceQuery(
        ticker="ZZNEW", session_date=SESSION,
        observation_ceiling=SESSION + "T23:59:59.000000Z", lookback_sessions=10),
        repository.resolve(snapshot_id))
    assert {row.date: row.close_adj for row in rows} == PRICES
    assert all(row.retrieved_at.startswith("2026-09-12T06:00:00") for row in rows)
