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
import pyarrow.parquet as pq
import pytest

from engine import calendar as legacy_calendar
from engine import paths
from engine.v2.data import objects
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
    DEFAULT_HORIZON_DAYS,
    FORWARD_CALENDAR_REFRESH_ACTION,
    NATIVE_COMPUTED_MOVES_ACCOUNT,
    NATIVE_NASDAQ_ACCOUNT,
    CalendarMovesParameters,
)
from engine.v2.ops.errors import OpsError, fail
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


def _daily_rows(tickers=TICKERS) -> list[dict]:
    days = pd.bdate_range("2026-08-03", "2026-12-01")
    return [dict(ticker=ticker, date=day.to_pydatetime(), year=2026, **_DAILY_COMMON)
            for ticker in tickers for day in days]


def _commit_parent(conn, store, clock, *, daily_rows=None, tickers=TICKERS):
    contracts = {"daily_market": contract_for("daily_market"),
                 "earnings_events": contract_for("earnings_events")}
    dm_ref = contract_ref_for(contracts["daily_market"])
    rows = _daily_rows(tickers) if daily_rows is None else daily_rows
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
    repository = Repository(conn, store)
    parent = repository.resolve(head["snapshot_id"])
    calendar = native_trading_calendar(
        forward_calendar_store.daily_by_ticker(repository, parent))
    dates = forward_calendar_store.horizon_dates(
        SESSION, DEFAULT_HORIZON_DAYS, calendar=calendar)
    # R6: every provider request is reserved -- the Nasdaq discovery calls AND
    # the yfinance session-confirmation calls the store can make this run.
    assert request.job.parameters["provider_calls"] == (len(dates) + len(TICKERS)) * 3
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
        "expected_head_snapshot_id": head["snapshot_id"], "all_scoreable": True,
        "as_of": SESSION}))

    parameters = CalendarMovesParameters(
        expected_ids=TICKERS, parent_snapshot_id=head["snapshot_id"],
        refresh_plan_hash="sha256:" + "b" * 64, catalog_path=str(tmp_path / "ops.sqlite"),
        objects_root=str(tmp_path), scope=SCOPE, expected_head_generation=head["generation"],
        as_of=SESSION)

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
        parameters, root,
        fetcher=lambda ticker: (frame.to_csv().encode(), "complete", {}, []))

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


# --------------------------------------------------------------------------
# 6. failure semantics and per-session unit ids (spec R1-R6)
# --------------------------------------------------------------------------


def _write_store_input(root, name, head, **extra):
    document = {"catalog_path": str(root / "ops.sqlite"), "objects_root": str(root),
                "scope": SCOPE, "expected_head_generation": head["generation"],
                "expected_head_snapshot_id": head["snapshot_id"], **extra}
    (root / name).write_text(json.dumps(document))
    return document


def _store_parameters(root, head, *, expected_ids=TICKERS):
    return CalendarMovesParameters(
        expected_ids=expected_ids, parent_snapshot_id=head["snapshot_id"],
        refresh_plan_hash="sha256:" + "b" * 64, catalog_path=str(root / "ops.sqlite"),
        objects_root=str(root), scope=SCOPE, expected_head_generation=head["generation"])


def _nasdaq_body(*, ticker=TICKER, time="time-after-hours"):
    return json.dumps({"data": {"rows": [{"symbol": ticker, "time": time}]}}).encode()


def _empty_earnings(ticker):
    return b"", "legitimate_empty", {}, []


def test_legitimate_empty_receipts_are_recorded_but_never_reused(tmp_path):
    """Spec R2: a legitimate-empty receipt keeps its own kind and is refetched."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)
    _write_store_input(tmp_path, "forward_calendar_refresh_input.json", head,
                       as_of=SESSION, horizon_days=7, tickers=list(TICKERS))
    parameters = _store_parameters(tmp_path, head)
    calls = []
    body = json.dumps({"data": {"rows": []}}).encode()

    def nasdaq(unit):
        calls.append(unit["partition_key"])
        return body, "legitimate_empty", {"status": 200}, []

    first = forward_calendar_store.run_forward_calendar_refresh(
        parameters, tmp_path, nasdaq_fetcher=nasdaq, earnings_fetcher=_empty_earnings)
    assert first.status == "noop"
    first_calls = len(calls)
    assert first_calls > 0
    kinds = {row["response_kind"] for row in conn.execute(
        "SELECT response_kind FROM data_raw_receipts WHERE source = 'nasdaq'")}
    assert kinds == {"legitimate_empty"}

    second = forward_calendar_store.run_forward_calendar_refresh(
        parameters, tmp_path, nasdaq_fetcher=nasdaq, earnings_fetcher=_empty_earnings)
    assert second.status == "noop"
    assert len(calls) == 2 * first_calls


def test_a_credential_invalid_response_is_not_cached_and_fails_non_retryably(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)
    _write_store_input(tmp_path, "forward_calendar_refresh_input.json", head,
                       as_of=SESSION, horizon_days=7, tickers=list(TICKERS))

    def forbidden(unit):
        return b"forbidden", "credential_invalid", {"status": 403}, []

    with pytest.raises(OpsError) as exc:
        forward_calendar_store.run_forward_calendar_refresh(
            _store_parameters(tmp_path, head), tmp_path, nasdaq_fetcher=forbidden,
            earnings_fetcher=_empty_earnings)
    assert exc.value.code == "CREDENTIAL_INVALID"
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts WHERE source = 'nasdaq'"
                        ).fetchone()[0] == 0
    assert _head(conn)["generation"] == head["generation"]


def test_all_transient_units_fail_with_transient_source_and_no_result(tmp_path):
    """Spec R3: a network outage on every unit can never be a noop/complete."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)
    _write_store_input(tmp_path, "computed_moves_refresh_input.json", head,
                       as_of=SESSION, all_scoreable=True)

    def down(ticker):
        raise TimeoutError("network down")

    with pytest.raises(OpsError) as exc:
        computed_moves_store.run_computed_moves_refresh(
            _store_parameters(tmp_path, head), tmp_path, fetcher=down)
    assert exc.value.code == "TRANSIENT_SOURCE"
    assert not (tmp_path / "computed_moves_refresh_result.json").is_file()
    assert _head(conn)["generation"] == head["generation"]
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts WHERE source = 'computed_moves'"
                        ).fetchone()[0] == 0


def test_unparseable_computed_moves_bytes_are_never_cached(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)
    _write_store_input(tmp_path, "computed_moves_refresh_input.json", head,
                       as_of=SESSION, all_scoreable=True)

    def garbage(ticker):
        return b"not,a,close\n1,2\n", "complete", {}, []

    with pytest.raises(OpsError) as exc:
        computed_moves_store.run_computed_moves_refresh(
            _store_parameters(tmp_path, head), tmp_path, fetcher=garbage)
    assert exc.value.code == "SOURCE_INVALID"
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts WHERE source = 'computed_moves'"
                        ).fetchone()[0] == 0


def test_unparseable_nasdaq_bytes_are_never_cached(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)
    _write_store_input(tmp_path, "forward_calendar_refresh_input.json", head,
                       as_of=SESSION, horizon_days=7, tickers=list(TICKERS))

    def garbage(unit):
        return b"not-json", "complete", {"status": 200}, []

    with pytest.raises(OpsError) as exc:
        forward_calendar_store.run_forward_calendar_refresh(
            _store_parameters(tmp_path, head), tmp_path, nasdaq_fetcher=garbage,
            earnings_fetcher=_empty_earnings)
    assert exc.value.code == "SOURCE_INVALID"
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts WHERE source = 'nasdaq'"
                        ).fetchone()[0] == 0


def _history_frame() -> pd.DataFrame:
    payload = _canned_history()
    return pd.DataFrame({"Close": payload["closes"]},
                        index=pd.to_datetime(payload["dates"]))


def test_a_new_session_refetches_computed_moves(tmp_path):
    """Spec R4: unit ids carry the as_of, so a later session refetches."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)
    frame = _history_frame()
    calls = []

    def history(ticker):
        calls.append(ticker)
        return frame.to_csv().encode(), "complete", {}, []

    _write_store_input(tmp_path, "computed_moves_refresh_input.json", head,
                       as_of=SESSION, all_scoreable=True)
    first = computed_moves_store.run_computed_moves_refresh(
        _store_parameters(tmp_path, head), tmp_path, fetcher=history)
    assert first.status == "complete"
    assert sorted(calls) == sorted(TICKERS)
    after = _head(conn)
    assert after["generation"] == head["generation"] + 1
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts WHERE source = 'computed_moves'"
                        ).fetchone()[0] == len(TICKERS)

    calls.clear()
    _write_store_input(tmp_path, "computed_moves_refresh_input.json", after,
                       as_of="2026-11-23", all_scoreable=True)
    second = computed_moves_store.run_computed_moves_refresh(
        _store_parameters(tmp_path, after), tmp_path, fetcher=history)
    assert second.status == "complete"
    assert sorted(calls) == sorted(TICKERS)


def test_a_new_session_refetches_nasdaq_dates(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)
    rows = [{"symbol": TICKER, "time": "time-after-hours"}]
    calls = []

    def nasdaq(unit):
        calls.append(unit["partition_key"])
        return _nasdaq_body(), "complete", {"status": 200}, rows

    _write_store_input(tmp_path, "forward_calendar_refresh_input.json", head,
                       as_of=SESSION, horizon_days=7, tickers=list(TICKERS))
    first = forward_calendar_store.run_forward_calendar_refresh(
        _store_parameters(tmp_path, head), tmp_path, nasdaq_fetcher=nasdaq,
        earnings_fetcher=_empty_earnings)
    assert first.status == "complete"
    first_calls = len(calls)
    assert first_calls > 0
    after = _head(conn)
    assert after["generation"] == head["generation"] + 1

    calls.clear()
    _write_store_input(tmp_path, "forward_calendar_refresh_input.json", after,
                       as_of="2026-11-23", horizon_days=7, tickers=list(TICKERS))
    second = forward_calendar_store.run_forward_calendar_refresh(
        _store_parameters(tmp_path, after), tmp_path, nasdaq_fetcher=nasdaq,
        earnings_fetcher=_empty_earnings)
    assert second.status == "complete"
    assert len(calls) == first_calls


def test_target_selection_uses_as_of_not_the_wall_clock(tmp_path, monkeypatch):
    """Spec R5: a frozen clock far from as_of must not move the targets."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)
    repository = Repository(conn, store)

    class FrozenTimestamp(pd.Timestamp):
        @classmethod
        def today(cls):
            return cls("2023-01-02")

    monkeypatch.setattr(pd, "Timestamp", FrozenTimestamp)
    targets, report = computed_moves_store.target_tickers_from_snapshot(
        repository, head["snapshot_id"], as_of=SESSION)
    assert targets == sorted(TICKERS)
    assert report["scoreable_on_orats_calendar"] == len(TICKERS)


# --------------------------------------------------------------------------
# 7. a same-session retry rebuilds every unit, cached receipts included
# --------------------------------------------------------------------------


RETRY_TICKERS = tuple(f"CM{index:02d}" for index in range(21))


def _fixed_system_clock(monkeypatch, module, clock):
    monkeypatch.setattr(module, "SystemClock", lambda: clock)


def _fragment_rows(conn, store, snapshot_id, table_name):
    resolved = Repository(conn, store).resolve_full(snapshot_id)
    contract_id = next(item.contract_id for item in resolved.contracts
                       if item.table_name == table_name)
    rows = []
    for record in resolved.records:
        if record.table_contract_ref.contract_id != contract_id:
            continue
        rows.extend(pq.read_table(objects.verify_object_path(store, record.object_ref)).to_pylist())
    return rows


def _row_key(row):
    return (str(row.get("ticker")), str(row.get("event_date")))


def _prepared_root(root, tickers):
    root.mkdir()
    conn, clock, _ = catalog(root)
    store = ArtifactStore(root)
    _commit_parent(conn, store, clock, tickers=tickers)
    return conn, clock, store


def test_computed_moves_retry_commits_every_cached_unit(tmp_path, monkeypatch):
    """20 good units + 1 transient, then the transient recovers: the retry
    commits the same rows a clean single run over all 21 units would, and
    refetches only the one unit that failed."""
    retry_root = tmp_path / "cm-retry"
    clean_root = tmp_path / "cm-clean"
    retry_conn, retry_clock, retry_store = _prepared_root(retry_root, RETRY_TICKERS)
    head = _head(retry_conn)
    _write_store_input(retry_root, "computed_moves_refresh_input.json", head,
                       as_of=SESSION, all_scoreable=True)
    parameters = _store_parameters(retry_root, head, expected_ids=RETRY_TICKERS)

    frame = _history_frame()
    calls, failing = [], {RETRY_TICKERS[-1]}

    def history(ticker):
        calls.append(ticker)
        if ticker in failing:
            raise TimeoutError("transient outage")
        return frame.to_csv().encode(), "complete", {}, []

    _fixed_system_clock(monkeypatch, computed_moves_store, retry_clock)
    with pytest.raises(OpsError) as exc:
        computed_moves_store.run_computed_moves_refresh(parameters, retry_root,
                                                        fetcher=history)
    assert exc.value.code == "TRANSIENT_SOURCE"
    assert len(calls) == len(RETRY_TICKERS)
    assert retry_conn.execute(
        "SELECT COUNT(*) FROM data_raw_receipts WHERE source = 'computed_moves'"
    ).fetchone()[0] == len(RETRY_TICKERS) - 1
    assert _head(retry_conn)["generation"] == head["generation"]

    failing.clear()
    calls.clear()
    retried = computed_moves_store.run_computed_moves_refresh(parameters, retry_root,
                                                             fetcher=history)
    assert retried.status == "complete"
    assert tuple(retried.completed_ids) == RETRY_TICKERS
    assert calls == [RETRY_TICKERS[-1]]  # the 20 cached units were not refetched

    clean_conn, clean_clock, clean_store = _prepared_root(clean_root, RETRY_TICKERS)
    clean_head = _head(clean_conn)
    _write_store_input(clean_root, "computed_moves_refresh_input.json", clean_head,
                       as_of=SESSION, all_scoreable=True)
    _fixed_system_clock(monkeypatch, computed_moves_store, clean_clock)
    clean = computed_moves_store.run_computed_moves_refresh(
        _store_parameters(clean_root, clean_head, expected_ids=RETRY_TICKERS), clean_root,
        fetcher=lambda ticker: (frame.to_csv().encode(), "complete", {}, []))
    assert clean.status == "complete"

    retry_rows = _fragment_rows(retry_conn, retry_store, _head(retry_conn)["snapshot_id"],
                                "computed_moves")
    clean_rows = _fragment_rows(clean_conn, clean_store, _head(clean_conn)["snapshot_id"],
                                "computed_moves")
    assert sorted(retry_rows, key=_row_key) == sorted(clean_rows, key=_row_key)


def _injected_commit_fault(point):
    raise fail("INPUT_CHANGED", "commit stage failed after every unit fetched: " + point)


def test_computed_moves_retry_after_commit_failure_rebuilds_every_cached_unit(
        tmp_path, monkeypatch):
    """Every unit fetches, the fenced commit fails INPUT_CHANGED (receipts are
    durable, rows are not), then the retry makes zero provider calls and
    commits every row a clean run would; a further retry is then a true noop."""
    retry_root = tmp_path / "cm-commit-fail"
    clean_root = tmp_path / "cm-commit-clean"
    retry_conn, retry_clock, retry_store = _prepared_root(retry_root, TICKERS)
    head = _head(retry_conn)
    _write_store_input(retry_root, "computed_moves_refresh_input.json", head,
                       as_of=SESSION, all_scoreable=True)
    parameters = _store_parameters(retry_root, head)

    frame = _history_frame()
    calls = []

    def history(ticker):
        calls.append(ticker)
        return frame.to_csv().encode(), "complete", {}, []

    state = {"fail": True}
    real_commit = computed_moves_store.data_catalog.commit_snapshot

    def commit(*args, **kwargs):
        if state["fail"]:
            kwargs["fault"] = _injected_commit_fault
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(computed_moves_store.data_catalog, "commit_snapshot", commit)
    _fixed_system_clock(monkeypatch, computed_moves_store, retry_clock)

    with pytest.raises(OpsError) as exc:
        computed_moves_store.run_computed_moves_refresh(parameters, retry_root, fetcher=history)
    assert exc.value.code == "INPUT_CHANGED"
    assert sorted(calls) == sorted(TICKERS)  # every unit fetched before the commit ran
    assert retry_conn.execute(
        "SELECT COUNT(*) FROM data_raw_receipts WHERE source = 'computed_moves'"
    ).fetchone()[0] == len(TICKERS)  # receipts are durable ...
    assert _head(retry_conn)["generation"] == head["generation"]  # ... rows are not committed

    state["fail"] = False
    calls.clear()
    retried = computed_moves_store.run_computed_moves_refresh(parameters, retry_root,
                                                             fetcher=history)
    assert retried.status == "complete"
    assert tuple(retried.completed_ids) == TICKERS
    assert calls == []  # every unit was re-read from its cached receipt
    after = _head(retry_conn)
    assert after["generation"] == head["generation"] + 1

    clean_conn, clean_clock, clean_store = _prepared_root(clean_root, TICKERS)
    clean_head = _head(clean_conn)
    _write_store_input(clean_root, "computed_moves_refresh_input.json", clean_head,
                       as_of=SESSION, all_scoreable=True)
    _fixed_system_clock(monkeypatch, computed_moves_store, clean_clock)
    clean = computed_moves_store.run_computed_moves_refresh(
        _store_parameters(clean_root, clean_head), clean_root,
        fetcher=lambda ticker: (frame.to_csv().encode(), "complete", {}, []))
    assert clean.status == "complete"

    retry_rows = _fragment_rows(retry_conn, retry_store, after["snapshot_id"], "computed_moves")
    clean_rows = _fragment_rows(clean_conn, clean_store, _head(clean_conn)["snapshot_id"],
                                "computed_moves")
    assert sorted(retry_rows, key=_row_key) == sorted(clean_rows, key=_row_key)

    captures = retry_conn.execute(
        "SELECT COUNT(*) FROM data_computed_moves_captures").fetchone()[0]
    _write_store_input(retry_root, "computed_moves_refresh_input.json", after,
                       as_of=SESSION, all_scoreable=True)
    noop = computed_moves_store.run_computed_moves_refresh(
        _store_parameters(retry_root, after), retry_root, fetcher=history)
    assert noop.status == "noop"
    assert calls == []
    assert (_head(retry_conn)["snapshot_id"], _head(retry_conn)["generation"]) == (
        after["snapshot_id"], after["generation"])
    assert retry_conn.execute(
        "SELECT COUNT(*) FROM data_computed_moves_captures").fetchone()[0] == captures


def test_forward_calendar_retry_commits_every_cached_unit(tmp_path, monkeypatch):
    """20 good date units + 1 transient, then it recovers: the retry rebuilds
    the cached Nasdaq date receipts AND the cached yfinance receipt, commits
    the committed-row set a clean single run would, and fetches only the unit
    that failed."""
    retry_root = tmp_path / "fc-retry"
    clean_root = tmp_path / "fc-clean"
    retry_conn, retry_clock, retry_store = _prepared_root(retry_root, TICKERS)
    head = _head(retry_conn)
    repository = Repository(retry_conn, retry_store)
    parent = repository.resolve(head["snapshot_id"])
    dates = forward_calendar_store.horizon_dates(
        SESSION, 30, calendar=native_trading_calendar(
            forward_calendar_store.daily_by_ticker(repository, parent)))
    assert len(dates) == 21
    bad_day = str(dates[-1].date())
    _write_store_input(retry_root, "forward_calendar_refresh_input.json", head,
                       as_of=SESSION, horizon_days=30, tickers=[TICKER])
    parameters = _store_parameters(retry_root, head, expected_ids=(TICKER,))

    body = _nasdaq_body(time="time-not-supplied")
    nasdaq_calls, earnings_calls = [], []
    failing = {bad_day}

    def nasdaq(unit):
        nasdaq_calls.append(unit["partition_key"])
        if unit["partition_key"] in failing:
            raise TimeoutError("transient outage")
        rows = [{"symbol": TICKER, "time": "time-not-supplied"}]
        return body, "complete", {"status": 200}, rows

    earnings = pd.DataFrame([{"ticker": TICKER, "event_date": SESSION,
                              "annc_tod": "1650", "session": "AMC"}])

    def earn(ticker):
        earnings_calls.append(ticker)
        return earnings.to_csv(index=False).encode(), "complete", {}, []

    _fixed_system_clock(monkeypatch, forward_calendar_store, retry_clock)
    with pytest.raises(OpsError) as exc:
        forward_calendar_store.run_forward_calendar_refresh(
            parameters, retry_root, nasdaq_fetcher=nasdaq, earnings_fetcher=earn)
    assert exc.value.code == "TRANSIENT_SOURCE"
    assert len(nasdaq_calls) == 21
    assert earnings_calls == [TICKER]
    assert retry_conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 21
    assert _head(retry_conn)["generation"] == head["generation"]

    failing.clear()
    nasdaq_calls.clear()
    earnings_calls.clear()
    retried = forward_calendar_store.run_forward_calendar_refresh(
        parameters, retry_root, nasdaq_fetcher=nasdaq, earnings_fetcher=earn)
    assert retried.status == "complete"
    assert tuple(retried.completed_ids) == (TICKER,)
    assert nasdaq_calls == [bad_day]
    assert earnings_calls == []  # its receipt was re-read, not refetched

    clean_conn, clean_clock, clean_store = _prepared_root(clean_root, TICKERS)
    clean_head = _head(clean_conn)
    _write_store_input(clean_root, "forward_calendar_refresh_input.json", clean_head,
                       as_of=SESSION, horizon_days=30, tickers=[TICKER])
    clean_calls = []

    def clean_nasdaq(unit):
        clean_calls.append(unit["partition_key"])
        rows = [{"symbol": TICKER, "time": "time-not-supplied"}]
        return body, "complete", {"status": 200}, rows

    _fixed_system_clock(monkeypatch, forward_calendar_store, clean_clock)
    clean = forward_calendar_store.run_forward_calendar_refresh(
        _store_parameters(clean_root, clean_head, expected_ids=(TICKER,)), clean_root,
        nasdaq_fetcher=clean_nasdaq, earnings_fetcher=earn)
    assert clean.status == "complete"
    assert len(clean_calls) == 21

    retry_rows = _fragment_rows(retry_conn, retry_store, _head(retry_conn)["snapshot_id"],
                                "earnings_events")
    clean_rows = _fragment_rows(clean_conn, clean_store, _head(clean_conn)["snapshot_id"],
                                "earnings_events")
    assert sorted(retry_rows, key=_row_key) == sorted(clean_rows, key=_row_key)


def test_forward_calendar_retry_after_commit_failure_rebuilds_every_cached_unit(
        tmp_path, monkeypatch):
    """Every date and ticker unit fetches, the fenced commit fails
    INPUT_CHANGED (receipts are durable, rows are not), then the retry makes
    zero provider calls and commits every row a clean run would; a further
    retry when those rows are committed is a true noop."""
    retry_root = tmp_path / "fc-commit-fail"
    clean_root = tmp_path / "fc-commit-clean"
    retry_conn, retry_clock, retry_store = _prepared_root(retry_root, TICKERS)
    head = _head(retry_conn)
    _write_store_input(retry_root, "forward_calendar_refresh_input.json", head,
                       as_of=SESSION, horizon_days=7, tickers=list(TICKERS))
    parameters = _store_parameters(retry_root, head, expected_ids=TICKERS)

    rows = [{"symbol": ticker, "time": "time-not-supplied"} for ticker in TICKERS]
    body = json.dumps({"data": {"rows": rows}}).encode()
    nasdaq_calls, earnings_calls = [], []

    def nasdaq(unit):
        nasdaq_calls.append(unit["partition_key"])
        return body, "complete", {"status": 200}, rows

    earnings = pd.DataFrame([{"ticker": ticker, "event_date": SESSION,
                              "annc_tod": "1650", "session": "AMC"} for ticker in TICKERS])

    def earn(ticker):
        earnings_calls.append(ticker)
        return earnings.to_csv(index=False).encode(), "complete", {}, []

    state = {"fail": True}
    real_commit = forward_calendar_store.generic_incremental.commit_generic_table_candidate

    def commit(*args, **kwargs):
        if state["fail"]:
            kwargs["fault"] = _injected_commit_fault
        return real_commit(*args, **kwargs)

    monkeypatch.setattr(forward_calendar_store.generic_incremental,
                        "commit_generic_table_candidate", commit)
    _fixed_system_clock(monkeypatch, forward_calendar_store, retry_clock)

    with pytest.raises(OpsError) as exc:
        forward_calendar_store.run_forward_calendar_refresh(
            parameters, retry_root, nasdaq_fetcher=nasdaq, earnings_fetcher=earn)
    assert exc.value.code == "INPUT_CHANGED"
    assert len(nasdaq_calls) > 0 and len(set(nasdaq_calls)) == len(nasdaq_calls)
    assert earnings_calls == list(TICKERS)
    receipts = nasdaq_calls[:]
    assert retry_conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == (
        len(receipts) + len(TICKERS))  # receipts are durable ...
    assert _head(retry_conn)["generation"] == head["generation"]  # ... rows are not committed

    state["fail"] = False
    nasdaq_calls.clear()
    earnings_calls.clear()
    retried = forward_calendar_store.run_forward_calendar_refresh(
        parameters, retry_root, nasdaq_fetcher=nasdaq, earnings_fetcher=earn)
    assert retried.status == "complete"
    assert tuple(retried.completed_ids) == TICKERS
    assert nasdaq_calls == [] and earnings_calls == []  # every receipt was re-read
    after = _head(retry_conn)
    assert after["generation"] == head["generation"] + 1

    clean_conn, clean_clock, clean_store = _prepared_root(clean_root, TICKERS)
    clean_head = _head(clean_conn)
    _write_store_input(clean_root, "forward_calendar_refresh_input.json", clean_head,
                       as_of=SESSION, horizon_days=7, tickers=list(TICKERS))
    _fixed_system_clock(monkeypatch, forward_calendar_store, clean_clock)
    clean = forward_calendar_store.run_forward_calendar_refresh(
        _store_parameters(clean_root, clean_head, expected_ids=TICKERS), clean_root,
        nasdaq_fetcher=nasdaq, earnings_fetcher=earn)
    assert clean.status == "complete"

    retry_rows = _fragment_rows(retry_conn, retry_store, after["snapshot_id"], "earnings_events")
    clean_rows = _fragment_rows(clean_conn, clean_store, _head(clean_conn)["snapshot_id"],
                                "earnings_events")
    assert sorted(retry_rows, key=_row_key) == sorted(clean_rows, key=_row_key)

    nasdaq_calls.clear()
    earnings_calls.clear()
    revisions = retry_conn.execute(
        "SELECT COUNT(*) FROM data_table_revisions WHERE table_name = 'earnings_events'"
    ).fetchone()[0]
    _write_store_input(retry_root, "forward_calendar_refresh_input.json", after,
                       as_of=SESSION, horizon_days=7, tickers=list(TICKERS))
    noop = forward_calendar_store.run_forward_calendar_refresh(
        _store_parameters(retry_root, after, expected_ids=TICKERS), retry_root,
        nasdaq_fetcher=nasdaq, earnings_fetcher=earn)
    assert noop.status == "noop"
    assert nasdaq_calls == [] and earnings_calls == []
    assert (_head(retry_conn)["snapshot_id"], _head(retry_conn)["generation"]) == (
        after["snapshot_id"], after["generation"])
    assert retry_conn.execute(
        "SELECT COUNT(*) FROM data_table_revisions WHERE table_name = 'earnings_events'"
    ).fetchone()[0] == revisions
