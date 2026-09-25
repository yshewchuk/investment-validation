"""S4C end-to-end: the native computed_moves and forward_calendar refresh jobs.

The worker subprocess is the real ``engine.v2.ops.worker`` (its argv swapped
for a stub -- the seam ``test_v2_ops_native_refresh_nightly_e2e.py`` uses) and
the real nightly submit path builds each job. Only the new provider fetchers
are faked: ``history_fn`` for computed_moves, ``http_get``+``earnings_fn`` for
the forward calendar. A cache-only second run with an exploding fetcher proves
the durable raw-receipt cache, and a missing-staged-input run pins each kind's
own ``WORKER_FAILED`` message.
"""
from __future__ import annotations

import collections
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from engine import calendar as legacy_calendar
from engine import paths
from engine.v2.data.computed_moves import native_trading_calendar
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore
from engine.v2.ops import (
    cli,
    computed_moves_store,
    executor,
    forward_calendar_store,
    refresh_staging,
)
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.calendar_moves_jobs import (
    COMPUTED_MOVES_REFRESH_ACTION,
    FORWARD_CALENDAR_REFRESH_ACTION,
    NATIVE_COMPUTED_MOVES_ACCOUNT,
    NATIVE_NASDAQ_ACCOUNT,
)
from engine.v2.ops.nightly import build_legacy_job_requests
from engine.v2.ops.plans import nightly_plan
from engine.v2.ops.profiles import MIB
from engine.v2.ops.provider_budget import configure_account
from engine.v2.ops.providers import PROVIDER_CREDENTIAL_VARIABLES, provider_credentials
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit
from engine.v2.ops.supervisor import Service
from tests.data_scan_support import (
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.ops_support import TEST_POLICY, FakeClock, catalog, run_until

ROOT = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})
#: TEST_POLICY with every profile's memory capped small: these tests drive the
#: real host admission sampler, and a 384 MiB io_fetch reservation loses to a
#: busy shared host for reasons unrelated to the code under test.
RUN_POLICY = replace(TEST_POLICY, profiles=tuple(
    replace(profile, memory_bytes=min(profile.memory_bytes, 256 * MIB))
    for profile in TEST_POLICY.profiles))
SCOPE = "shadow"
SESSION = "2026-11-20"
TICKERS = ("AAA", "BBB", "CCC")
TICKER = TICKERS[0]
_REAL_POPEN = subprocess.Popen


@pytest.fixture(autouse=True)
def _patient_admission(monkeypatch):
    """These tests drive the real host sampler; a transient MEMORY_HEADROOM
    shortage on a shared box is not a code failure, so wait it out."""
    monkeypatch.setattr("tests.ops_support.ADMISSION_WAIT_SECONDS", 200.0)

_EVENT_DAYS = (
    "2024-01-16", "2024-02-15", "2024-04-16", "2024-05-15", "2024-07-16",
    "2024-08-15", "2024-10-15", "2024-11-15", "2025-01-15", "2025-02-18",
    "2025-04-15", "2025-05-15",
)


# --------------------------------------------------------------------------
# synthetic parent snapshots (real contracts, real Parquet, no market data)
# --------------------------------------------------------------------------


def _event_rows(ticker: str, year: int) -> list[dict]:
    return [dict(event_id=f"EE_{ticker}_{day}", ticker=ticker,
                 event_date=datetime.fromisoformat(day), year=year, annc_tod="0800",
                 session="BMO", session_src="orats", src_orats=True, src_oquants=False,
                 src_nasdaq=False, src_yfinance=False, date_agree=True, date_conflict=False,
                 event_cluster_id=f"{ticker}_{day}", claim_count=1,
                 reconciliation="reconciled")
            for day in _EVENT_DAYS if day.startswith(str(year))]


_DAILY_COMMON = dict(spot=100.0, iv10=30.0, iv30=32.0, exern_iv10=29.0, exern_iv30=31.0,
                     implied_move=5.0, implied_reconstructed=False, rvol30=28.0, skew=1.1,
                     contango=0.5, fwd90_30=33.0, fexern90_30=34.0, iee=0.2, mcap_usd=1e9,
                     mcap_log=20.7, mcap_asof=datetime(2026, 8, 3), mcap_age_days=0.0,
                     src_spot="orats", src_iv="orats", src_mcap="orats")


def _daily_rows() -> list[dict]:
    days = pd.bdate_range("2026-08-03", "2026-12-01")
    return [dict(ticker=ticker, date=day.to_pydatetime(), year=2026, **_DAILY_COMMON)
            for ticker in TICKERS for day in days]


def _commit_parent(conn, store, clock, *, daily_rows=None, tickers=TICKERS):
    contracts = {"daily_market": contract_for("daily_market"),
                 "earnings_events": contract_for("earnings_events")}
    dm_ref = contract_ref_for(contracts["daily_market"])
    rows = _daily_rows() if daily_rows is None else daily_rows
    dm_records = [publish_and_inspect(store, contracts["daily_market"], dm_ref, rows, "2026")]
    ee_ref = contract_ref_for(contracts["earnings_events"])
    ee_records = []
    for year in (2024, 2025):
        year_rows = [row for ticker in tickers for row in _event_rows(ticker, year)]
        ee_records.append(publish_and_inspect(
            store, contracts["earnings_events"], ee_ref, year_rows, str(year)))
    commit_tables(conn, clock, {"earnings_events": ee_records, "daily_market": dm_records},
                  contracts, store=store)


def _head(conn):
    return conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?",
        (SCOPE,)).fetchone()


def _build_requests(conn, store, clock, root, *, tickers=TICKERS, context_tickers=None):
    catalog_path = str(root / "ops.sqlite")
    context_tickers = tuple(context_tickers) if context_tickers is not None else tuple(tickers)
    plan = nightly_plan(
        ROOT, SESSION, manifest_ref="artifact:parent",
        expected_population=(TICKER + "|x|" + SESSION,),
        tickers=tickers, context_tickers=context_tickers, refresh_mode="native",
        catalog_path=catalog_path, objects_root=str(root), clock=clock)
    requests = build_legacy_job_requests(
        plan, tickers=tickers, context_tickers=context_tickers,
        year_start=plan["year_start"], year_end=plan["year_end"],
        refresh_mode="native", catalog_path=catalog_path,
        objects_root=str(root), conn=conn, store=store, clock=clock)
    return {request.job.kind: request for request in requests}


# --------------------------------------------------------------------------
# the worker stubs -- only the new provider factory is faked
# --------------------------------------------------------------------------


def _canned_history() -> dict:
    days = pd.bdate_range("2023-12-01", "2026-06-30")
    return {"dates": [day.strftime("%Y-%m-%d") for day in days],
            "closes": [100.0 + 0.25 * index for index in range(len(days))]}


def _install_history_stub(monkeypatch, *, explode):
    payload = json.dumps(_canned_history())
    body = (
        "def fake_history_fn(ticker):\n"
        "    raise AssertionError('yfinance ran on a cache-only refresh: ' + ticker)\n"
        if explode else
        "PAYLOAD = json.loads(%r)\n"
        "FRAME = pd.DataFrame({'Close': PAYLOAD['closes']},\n"
        "                     index=pd.to_datetime(PAYLOAD['dates']))\n"
        "def fake_history_fn(ticker):\n"
        "    return FRAME\n" % payload
    )
    _swap_worker(monkeypatch, _worker_stub_prelude() + body + (
        "patched = functools.partial(yfinance_edge.yfinance_history_fetcher,\n"
        "                            history_fn=fake_history_fn)\n"
        "providers.yfinance_history_fetcher = patched\n"
        "raise SystemExit(worker.main())\n"))


def _install_calendar_stub(monkeypatch, *, explode):
    earnings = {"ticker": TICKER, "event_date": SESSION, "annc_tod": "1650", "session": "AMC"}
    if explode:
        body = (
            "def fake_http_get(url, *, timeout):\n"
            "    raise AssertionError('nasdaq ran on a cache-only refresh: ' + url)\n"
            "def fake_earnings_fn(ticker):\n"
            "    raise AssertionError('yfinance ran on a cache-only refresh: ' + ticker)\n")
    else:
        body = (
            "def fake_http_get(url, *, timeout):\n"
            "    row = {'symbol': %r, 'time': 'time-not-supplied'}\n"
            "    payload = {'data': {'rows': [row]}, 'message': 'ok'}\n"
            "    return (200, {}, json.dumps(payload).encode())\n"
            "EARNINGS = pd.DataFrame([json.loads(%r)],\n"
            "                        columns=['ticker', 'event_date', 'annc_tod', 'session'])\n"
            "def fake_earnings_fn(ticker):\n"
            "    return EARNINGS if ticker == %r else None\n"
            % (TICKER, json.dumps(earnings), TICKER))
    _swap_worker(monkeypatch, _worker_stub_prelude() + body + (
        "nasdaq = functools.partial(nasdaq_calendar.nasdaq_calendar_fetcher,\n"
        "                           http_get=fake_http_get)\n"
        "providers.nasdaq_calendar_fetcher = nasdaq\n"
        "earnings = functools.partial(yfinance_edge.yfinance_earnings_fetcher,\n"
        "                             earnings_fn=fake_earnings_fn)\n"
        "providers.yfinance_earnings_fetcher = earnings\n"
        "raise SystemExit(worker.main())\n"))


def _worker_stub_prelude() -> str:
    return (
        "import functools, json, sys\n"
        "import pandas as pd\n"
        "import engine.v2.ops.providers as providers\n"
        "from engine.v2.ops.providers import nasdaq_calendar, yfinance_edge\n"
        "from engine.v2.ops import worker\n")


def _swap_worker(monkeypatch, stub):
    def popen(args, **kwargs):
        if list(args[-2:]) == ["-m", "engine.v2.ops.worker"]:
            args = [sys.executable, "-u", "-c", stub]
        return _REAL_POPEN(args, **kwargs)

    monkeypatch.setattr(executor.subprocess, "Popen", popen)


# --------------------------------------------------------------------------
# the real supervisor -> executor -> worker path
# --------------------------------------------------------------------------


def _run_native(conn, clock, root, receipt, monkeypatch, install):
    install(monkeypatch, explode=False)
    service = Service(conn, root, registry(), RUN_POLICY, clock=clock, code_source=ROOT)
    service.start()
    try:
        state = run_until(service, conn, receipt.job_id, timeout=300)
        if state != "succeeded":
            _raise_attempt_failure(conn, service, receipt.job_id, state)
    finally:
        service.close()


def _raise_attempt_failure(conn, service, job_id, state):
    attempt = conn.execute(
        "SELECT attempt_id FROM attempts WHERE job_id = ? "
        "ORDER BY attempt_number DESC LIMIT 1", (job_id,)).fetchone()
    stderr = ""
    if attempt is not None:
        path = service.store.staging_dir(attempt["attempt_id"]) / "diagnostics" / "worker.stderr"
        if path.is_file():
            stderr = path.read_text()
    failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id = ?",
                           (job_id,)).fetchone()[0]
    raise AssertionError(f"calendar/moves refresh {state}: {failure}\n{stderr}")


# --------------------------------------------------------------------------
# 1. each kind lands its dataset version in the head
# --------------------------------------------------------------------------


def test_native_computed_moves_refresh_e2e(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)
    configure_account(conn, NATIVE_COMPUTED_MOVES_ACCOUNT, 1, remaining=20, live_reserve=1)

    request = _build_requests(conn, store, clock, tmp_path)[COMPUTED_MOVES_REFRESH_ACTION]
    assert request.job.parameters["provider_calls"] == len(TICKERS) * 3
    assert request.job.provider_budget_ref == NATIVE_COMPUTED_MOVES_ACCOUNT
    receipt = submit(conn, registry(), POLICY, request, clock=clock)

    _run_native(conn, clock, tmp_path, receipt, monkeypatch, _install_history_stub)

    new_head = _head(conn)
    assert new_head["generation"] == head["generation"] + 1
    committed = Repository(conn).resolve_full(new_head["snapshot_id"])
    assert "computed_moves" in committed.table_manifests
    captures = conn.execute(
        "SELECT ticker, outcome FROM data_computed_moves_captures ORDER BY ticker").fetchall()
    assert [(row["ticker"], row["outcome"]) for row in captures] == [
        (ticker, "added") for ticker in TICKERS]
    assert conn.execute(
        "SELECT COUNT(*) FROM data_raw_receipts WHERE source = 'computed_moves'"
    ).fetchone()[0] == len(TICKERS)


def test_native_forward_calendar_refresh_e2e(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)
    configure_account(conn, NATIVE_NASDAQ_ACCOUNT, 1, remaining=200, live_reserve=1)

    request = _build_requests(conn, store, clock, tmp_path)[FORWARD_CALENDAR_REFRESH_ACTION]
    assert request.job.parameters["provider_calls"] > 0
    assert request.job.provider_budget_ref == NATIVE_NASDAQ_ACCOUNT
    receipt = submit(conn, registry(), POLICY, request, clock=clock)

    _run_native(conn, clock, tmp_path, receipt, monkeypatch, _install_calendar_stub)

    new_head = _head(conn)
    attempt = conn.execute(
        "SELECT attempt_id, attempt_number, state FROM attempts WHERE job_id = ? "
        "ORDER BY attempt_number DESC LIMIT 1", (receipt.job_id,)).fetchone()
    result_path = ArtifactStore(tmp_path).staging_dir(attempt["attempt_id"]) / \
        "forward_calendar_refresh_result.json"
    assert new_head["generation"] == head["generation"] + 1, {
        "result": result_path.read_text() if result_path.is_file() else None,
        "attempt": tuple(attempt),
        "receipts": conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0],
    }
    revisions = conn.execute(
        "SELECT COUNT(*) FROM data_table_revisions WHERE table_name = 'earnings_events'"
    ).fetchone()[0]
    assert revisions == len(pd.bdate_range(SESSION, "2026-12-11"))
    sessions = [json.loads(row["row_json"])["session"] for row in conn.execute(
        "SELECT row_json FROM data_table_revisions WHERE table_name = 'earnings_events'")]
    assert "AMC" in sessions  # the yfinance session claim merged into its date's row


# --------------------------------------------------------------------------
# 2. negative controls: cache-only and missing staged identity
# --------------------------------------------------------------------------


def test_second_computed_moves_refresh_for_the_same_catalog_is_cache_only(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    configure_account(conn, NATIVE_COMPUTED_MOVES_ACCOUNT, 1, remaining=20, live_reserve=1)

    first = _build_requests(conn, store, clock, tmp_path)[COMPUTED_MOVES_REFRESH_ACTION]
    receipt = submit(conn, registry(), POLICY, first, clock=clock)
    _run_native(conn, clock, tmp_path, receipt, monkeypatch, _install_history_stub)
    head_after_first = _head(conn)
    receipts = conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0]
    captures = conn.execute("SELECT COUNT(*) FROM data_computed_moves_captures").fetchone()[0]

    clock.advance(60)  # a fresh same-session generation, not a retry of plan one
    second = _build_requests(conn, store, clock, tmp_path)[COMPUTED_MOVES_REFRESH_ACTION]
    assert second.job.parameters["provider_calls"] == 0
    assert second.job.provider_budget_ref is None
    receipt = submit(conn, registry(), POLICY, second, clock=clock)
    _run_native(conn, clock, tmp_path, receipt, monkeypatch,
                lambda monkeypatch, explode: _install_history_stub(monkeypatch, explode=True))

    head_after_second = _head(conn)
    assert (head_after_second["snapshot_id"], head_after_second["generation"]) == (
        head_after_first["snapshot_id"], head_after_first["generation"])
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == receipts
    assert conn.execute(
        "SELECT COUNT(*) FROM data_computed_moves_captures").fetchone()[0] == captures


def test_second_forward_calendar_refresh_for_the_same_catalog_is_cache_only(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    configure_account(conn, NATIVE_NASDAQ_ACCOUNT, 1, remaining=200, live_reserve=1)

    first = _build_requests(conn, store, clock, tmp_path)[FORWARD_CALENDAR_REFRESH_ACTION]
    receipt = submit(conn, registry(), POLICY, first, clock=clock)
    _run_native(conn, clock, tmp_path, receipt, monkeypatch, _install_calendar_stub)
    head_after_first = _head(conn)
    revisions = conn.execute("SELECT COUNT(*) FROM data_table_revisions").fetchone()[0]

    clock.advance(60)
    second = _build_requests(conn, store, clock, tmp_path)[FORWARD_CALENDAR_REFRESH_ACTION]
    assert second.job.parameters["provider_calls"] == 0
    assert second.job.provider_budget_ref is None
    receipt = submit(conn, registry(), POLICY, second, clock=clock)
    _run_native(conn, clock, tmp_path, receipt, monkeypatch,
                lambda monkeypatch, explode: _install_calendar_stub(monkeypatch, explode=True))

    head_after_second = _head(conn)
    assert (head_after_second["snapshot_id"], head_after_second["generation"]) == (
        head_after_first["snapshot_id"], head_after_first["generation"])
    assert conn.execute("SELECT COUNT(*) FROM data_table_revisions").fetchone()[0] == revisions


@pytest.mark.parametrize("kind,account,message", [
    (COMPUTED_MOVES_REFRESH_ACTION, NATIVE_COMPUTED_MOVES_ACCOUNT,
     "computed moves refresh did not produce complete coverage"),
    (FORWARD_CALENDAR_REFRESH_ACTION, NATIVE_NASDAQ_ACCOUNT,
     "forward calendar refresh did not produce complete coverage"),
])
def test_missing_staged_identity_fails_with_the_kinds_own_message(
        tmp_path, monkeypatch, kind, account, message):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    configure_account(conn, account, 1, remaining=200, live_reserve=1)
    request = _build_requests(conn, store, clock, tmp_path)[kind]
    receipt = submit(conn, registry(), POLICY, request, clock=clock)

    no_op = lambda claim, staging: None
    monkeypatch.setattr(executor, "stage_refresh_input", no_op)
    monkeypatch.setattr(refresh_staging, "stage_refresh_input", no_op)
    _install_history_stub(monkeypatch, explode=False)
    _install_calendar_stub(monkeypatch, explode=False)

    service = Service(conn, tmp_path, registry(), RUN_POLICY, clock=clock, code_source=ROOT)
    service.start()
    try:
        states = ("succeeded", "failed", "retry_wait")
        state = run_until(service, conn, receipt.job_id, timeout=180, states=states)
        for _ in range(4):
            if state != "retry_wait":
                break
            clock.advance(70)
            state = run_until(service, conn, receipt.job_id, timeout=180, states=states)
    finally:
        service.close()

    assert state == "failed"
    failure = json.loads(conn.execute(
        "SELECT failure_json FROM jobs WHERE job_id = ?",
        (receipt.job_id,)).fetchone()[0])
    assert failure["code"] == "WORKER_FAILED"
    assert failure["message"] == message


# --------------------------------------------------------------------------
# 3. the calendar reads the pinned snapshot, never the legacy CSV
# --------------------------------------------------------------------------


def test_forward_calendar_calendar_reads_pinned_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "GSPC_DAILY", tmp_path / "missing-gspc.csv")
    legacy_calendar.trading_calendar.cache_clear()
    with pytest.raises(FileNotFoundError):
        legacy_calendar.trading_calendar()  # the trap is armed for an accidental read

    def calendar_for(directory, end):
        directory.mkdir()
        clock = FakeClock()
        conn = open_catalog(directory / "catalog.sqlite", clock=clock)
        store = ArtifactStore(directory / "store")
        contract = contract_for("daily_market")
        ref = contract_ref_for(contract)
        days = pd.bdate_range("2026-11-02", end)
        rows = [dict(ticker=TICKER, date=day.to_pydatetime(), year=2026, **_DAILY_COMMON)
                for day in days]
        snapshot = commit_tables(
            conn, clock, {"daily_market": [publish_and_inspect(store, contract, ref, rows, "2026")]},
            {"daily_market": contract}, store=store)
        try:
            frames = forward_calendar_store.daily_by_ticker(Repository(conn, store), snapshot)
            return native_trading_calendar(frames)
        finally:
            conn.close()

    observed_to_thanksgiving = calendar_for(tmp_path / "a", "2026-12-01")
    projected_from_before = calendar_for(tmp_path / "b", "2026-11-19")
    dates_a = forward_calendar_store.horizon_dates(
        SESSION, 15, calendar=observed_to_thanksgiving)
    dates_b = forward_calendar_store.horizon_dates(
        SESSION, 15, calendar=projected_from_before)

    thanksgiving = pd.Timestamp("2026-11-26")
    assert thanksgiving in dates_a  # the snapshot's own observed session
    assert thanksgiving not in dates_b  # projection excludes the scheduled holiday
    assert dates_a != dates_b


# --------------------------------------------------------------------------
# 4. one scan per table per run, however many targets
# --------------------------------------------------------------------------


def test_computed_moves_scan_once_per_run(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)

    root = tmp_path / "attempt"
    root.mkdir()
    (root / "computed_moves_refresh_input.json").write_text(json.dumps({
        "catalog_path": str(tmp_path / "ops.sqlite"), "objects_root": str(tmp_path),
        "scope": SCOPE, "expected_head_generation": head["generation"],
        "expected_head_snapshot_id": head["snapshot_id"], "all_scoreable": True}))
    from engine.v2.ops.calendar_moves_jobs import CalendarMovesParameters

    parameters = CalendarMovesParameters(
        expected_ids=TICKERS, parent_snapshot_id=head["snapshot_id"],
        refresh_plan_hash="sha256:" + "b" * 64, catalog_path=str(tmp_path / "ops.sqlite"),
        objects_root=str(tmp_path), scope=SCOPE, expected_head_generation=head["generation"])

    calls = collections.Counter()
    real_scan = computed_moves_store._scan_rows

    def counting_scan(repository, snapshot, table_name, columns):
        calls[table_name] += 1
        return real_scan(repository, snapshot, table_name, columns)

    monkeypatch.setattr(computed_moves_store, "_scan_rows", counting_scan)
    payload = _canned_history()
    frame = pd.DataFrame({"Close": payload["closes"]},
                         index=pd.to_datetime(payload["dates"]))
    result = computed_moves_store.run_computed_moves_refresh(
        parameters, root, fetcher=lambda ticker: frame.to_csv().encode())

    assert result.status == "complete"
    assert tuple(result.completed_ids) == TICKERS
    assert calls["earnings_events"] == 1
    assert calls["daily_market"] == 1


# --------------------------------------------------------------------------
# 5. the new provider accounts are provisioned through the CLI
# --------------------------------------------------------------------------


def test_provider_account_cli_admits_new_accounts(tmp_path, capsys):
    clock = FakeClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    try:
        begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
        for account in (NATIVE_COMPUTED_MOVES_ACCOUNT, NATIVE_NASDAQ_ACCOUNT):
            assert cli.main(["provider-account", "--root", str(tmp_path), "--account",
                             account, "--remaining", "25", "--live-reserve", "1"]) == 0
            created = json.loads(capsys.readouterr().out)
            assert created["account"] == account
            assert created["remaining"] == 25
            assert PROVIDER_CREDENTIAL_VARIABLES[account] == ()
            assert provider_credentials({"provider_account": account}) == {}
    finally:
        conn.close()
