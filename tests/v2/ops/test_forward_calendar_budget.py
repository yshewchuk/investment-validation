"""Forward-calendar workers charge uncached units through real fenced budgets."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, content_hash, to_document
from engine.v2.ops import (
    catalog as catalog_module,
)
from engine.v2.ops import (
    forward_calendar_store,
    incremental_data,
    provider_budget,
    providers,
    worker,
)
from engine.v2.ops.calendar_moves_jobs import (
    FORWARD_CALENDAR_RESULT_PATH,
    FORWARD_CALENDAR_RESULT_SCHEMA,
    CalendarMovesParameters,
)
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError
from engine.v2.ops.lifecycle import release_reservations, request_cancel
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.providers import nasdaq_calendar, yfinance_edge
from engine.v2.ops.refresh_staging import stage_refresh_input
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit
from engine.v2.ops.unit_receipts import record_unit_receipt
from tests.ops_support import catalog, sample
from tests.v2.ops.test_forward_calendar_store import _seeded_parent

AS_OF = "2026-09-18"
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


def _nasdaq_payload(session="time-after-hours"):
    rows = [{"symbol": "AAPL", "time": session}]
    return json.dumps({"data": {"rows": rows}}).encode(), rows


def _earnings_frame():
    return pd.DataFrame([{
        "ticker": "AAPL", "event_date": AS_OF, "annc_tod": "1600", "session": "AMC",
    }])


def _assert_closed(connections):
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def _counts(conn, account):
    row = conn.execute(
        "SELECT a.remaining, a.uncertain, r.reserved_calls, r.used_calls "
        "FROM provider_accounts a JOIN provider_reservations r USING(account) "
        "WHERE account = ?", (account,)).fetchone()
    return None if row is None else tuple(row)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Real stores and admission; only the HTTP and library boundaries are fake."""
    conn, clock, supervisor = catalog(tmp_path)
    clock.advance(6 * 24 * 60 * 60)
    store = ArtifactStore(tmp_path / "objects")
    parent = _seeded_parent(conn, store, clock, scope="shadow", daily_market_rows=())
    state = SimpleNamespace(
        conn=conn, clock=clock, supervisor=supervisor, store=store, parent=parent,
        root=tmp_path, catalog_path=tmp_path / "ops.sqlite", calls=[], connections=[],
        session="time-after-hours", fail_source=None, claims=0,
    )
    monkeypatch.setattr(incremental_data, "SystemClock", lambda: clock)
    monkeypatch.setattr(forward_calendar_store, "SystemClock", lambda: clock)

    def guard_connect(path, **kwargs):
        assert kwargs == {"must_exist": True}
        connection = catalog_module.connect(path, **kwargs)
        state.connections.append(connection)
        return connection

    monkeypatch.setattr(provider_budget, "connect", guard_connect)

    def observe_call(source, argument):
        """A separate writer sees the committed debit before fake provider I/O."""
        assert state.connections
        _assert_closed(state.connections)
        count = 1 + sum(name == source for name, _ in state.calls)
        with closing(catalog_module.connect(state.catalog_path, busy_timeout_ms=0)) as reader:
            with transaction(reader):
                assert _counts(reader, source) == (20 - count, 1, 3, count)
        state.calls.append((source, argument))
        if state.fail_source == source:
            raise OSError("synthetic provider failure")

    def http_get(url, *, timeout):
        assert url == nasdaq_calendar.BASE_URL + "/calendar/earnings?date=" + AS_OF
        assert timeout == nasdaq_calendar.REQUEST_TIMEOUT_SECONDS
        observe_call("nasdaq", AS_OF)
        return 200, {}, _nasdaq_payload(state.session)[0]

    def earnings(ticker):
        assert ticker == "AAPL"
        observe_call("yfinance", ticker)
        return _earnings_frame()

    state.observe_call = observe_call
    monkeypatch.setattr(nasdaq_calendar, "_requests_get", http_get)
    monkeypatch.setattr(yfinance_edge, "_default_earnings", earnings)
    yield state
    _assert_closed(state.connections)
    conn.close()


def _claim(state, *, account="nasdaq"):
    """Submit to the production registry, claim its scalar budget, then stage it."""
    state.claims += 1
    if account is not None:
        provider_budget.configure_account(
            state.conn, account, "generation-1", remaining=20, live_reserve=2)
    parameters = CalendarMovesParameters(
        expected_ids=("AAPL",), parent_snapshot_id=state.parent.snapshot.snapshot_id,
        refresh_plan_hash=content_hash({"forward_calendar": AS_OF}),
        provider_calls=3 if account else 0, provider_account=account,
        catalog_path=str(state.catalog_path), objects_root=str(state.root / "objects"),
        scope="shadow", expected_head_generation=1,
        expected_head_snapshot_id=state.parent.snapshot.snapshot_id,
        as_of=AS_OF, horizon_days=1, tickers=("AAPL",),
    )
    spec = JobSpec(
        kind="forward_calendar_refresh", implementation_ref="code", spec_hash=None,
        environment_ref="env", parameters=to_document(parameters), input_refs=(),
        output_namespace="shadow", resource_class="io_fetch", provider_budget_ref=account,
        retry_policy_ref="bounded", checkpoint_contract_ref=FORWARD_CALENDAR_RESULT_SCHEMA,
    )
    receipt = submit(state.conn, registry(), POLICY, SubmitRequest(
        namespace="shadow", idempotency_key=f"calendar-{state.claims}",
        principal="operator", job=spec), clock=state.clock)
    claim = claim_next(
        state.conn, policy=DEFAULT_POLICY, sample=sample(state.clock),
        supervisor=state.supervisor, clock=state.clock, registry=registry())
    assert claim is not None
    assert claim.job_id == receipt.job_id
    if account:
        assert _counts(state.conn, account) == (20, 0, 3, 0)
    staging = state.root / claim.attempt_id
    staging.mkdir()
    stage_refresh_input(claim, staging)
    return claim, staging


def _dispatch(claim, staging):
    return worker.dispatch("forward_calendar_refresh", claim.spec.parameters, staging)


def _cache(state, source):
    """Persist a complete raw receipt using the same unit identities as the store."""
    if source == "nasdaq":
        unit, = forward_calendar_store.date_units([AS_OF], as_of=AS_OF)
        payload = _nasdaq_payload("time-not-supplied")[0]
        endpoint = "calendar/earnings"
    else:
        unit, = forward_calendar_store.ticker_units(["AAPL"], as_of=AS_OF)
        payload = _earnings_frame().to_csv(index=False).encode()
        endpoint = "earnings"
    return record_unit_receipt(
        state.conn, state.store, unit, payload, source=source, endpoint=endpoint,
        received_at=state.clock.now().isoformat())


def _head(state):
    return tuple(state.conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = 'shadow'"
    ).fetchone())


def _receipts(state):
    return [tuple(row) for row in state.conn.execute(
        "SELECT source, raw_receipt_id, raw_hash, response_kind "
        "FROM data_raw_receipts ORDER BY source, raw_receipt_id")]


def _budget_state(state):
    return tuple([tuple(row) for row in state.conn.execute(
        f"SELECT * FROM {table} ORDER BY account, rowid")]
        for table in ("provider_accounts", "provider_reservations"))


def _assert_published(state, staging, result):
    assert result["completed_ids"] == ["AAPL"]
    assert result["no_work"] is False
    snapshot, generation = _head(state)
    assert snapshot != state.parent.snapshot.snapshot_id
    assert generation == 2
    evidence = json.loads((staging / FORWARD_CALENDAR_RESULT_PATH).read_text())
    assert evidence["candidate_snapshot_id"] == snapshot
    assert evidence["status"] == "complete"


def test_fresh_nasdaq_charges_once_before_default_http_edge(harness):
    claim, staging = _claim(harness)
    _assert_published(harness, staging, _dispatch(claim, staging))
    assert harness.calls == [("nasdaq", AS_OF)]
    assert _counts(harness.conn, "nasdaq") == (19, 1, 3, 1)
    assert len(harness.connections) == 1
    assert [row[0] for row in _receipts(harness)] == ["nasdaq"]


def test_cached_nasdaq_charges_only_fresh_yfinance(harness):
    _cache(harness, "nasdaq")
    claim, staging = _claim(harness, account="yfinance")
    _assert_published(harness, staging, _dispatch(claim, staging))
    assert harness.calls == [("yfinance", "AAPL")]
    assert _counts(harness.conn, "yfinance") == (19, 1, 3, 1)
    assert len(harness.connections) == 1
    assert [row[0] for row in harness.conn.execute(
        "SELECT account FROM provider_accounts")] == ["yfinance"]


def test_complete_cache_needs_no_budget_accounts_or_live_calls(harness):
    _cache(harness, "nasdaq")
    _cache(harness, "yfinance")
    claim, staging = _claim(harness, account=None)
    before = _receipts(harness)
    _assert_published(harness, staging, _dispatch(claim, staging))
    assert harness.calls == harness.connections == []
    assert _budget_state(harness) == ([], [])
    assert _receipts(harness) == before


def test_cached_commit_checks_stale_head_before_cancelled_attempt(harness):
    _cache(harness, "nasdaq")
    _cache(harness, "yfinance")
    claim, staging = _claim(harness, account=None)
    request_cancel(harness.conn, claim.job_id, claim.attempt_id, clock=harness.clock)
    with transaction(harness.conn):
        harness.conn.execute(
            "UPDATE data_snapshot_heads SET generation = generation + 1 WHERE scope = 'shadow'")
    before = _receipts(harness), _head(harness)
    with pytest.raises(DataError) as caught:
        _dispatch(claim, staging)
    assert caught.value.code == "SNAPSHOT_CONFLICT"
    assert harness.calls == harness.connections == []
    assert _budget_state(harness) == ([], [])
    assert (_receipts(harness), _head(harness)) == before
    assert harness.conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0] == 1
    assert not (staging / FORWARD_CALENDAR_RESULT_PATH).exists()


def test_standalone_whole_market_cache_preserves_unfenced_research_contract(harness):
    _cache(harness, "nasdaq")
    _cache(harness, "yfinance")
    receipts = _receipts(harness)
    result = forward_calendar_store.run_forward_calendar_refresh(
        catalog_path=str(harness.catalog_path), objects_root=str(harness.root / "objects"),
        parent_snapshot_id=harness.parent.snapshot.snapshot_id,
        refresh_plan_hash=content_hash({"research_calendar": AS_OF}),
        as_of=AS_OF, tickers=(), horizon_days=1, scope="shadow",
        expected_head_generation=1,
        expected_head_snapshot_id=harness.parent.snapshot.snapshot_id,
        nasdaq_fetcher=providers.nasdaq_calendar_fetcher(),
        earnings_fetcher=providers.yfinance_earnings_fetcher())
    assert result.status == "complete"
    assert _head(harness) == (result.candidate_snapshot_id, 2)
    repository = Repository(harness.conn, harness.store)
    snapshot = repository.resolve_full(result.candidate_snapshot_id)
    row = forward_calendar_store._existing_index(repository, snapshot, {("AAPL", AS_OF)})[("AAPL", AS_OF)]
    assert (row["session"], row["src_nasdaq"], row["src_yfinance"]) == ("AMC", True, True)
    assert harness.calls == harness.connections == []
    assert _budget_state(harness) == ([], [])
    assert _receipts(harness) == receipts
    assert harness.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0


@pytest.mark.parametrize("source", ["nasdaq", "yfinance"])
def test_cache_invalidated_after_unbudgeted_admission_refuses_live_fetch(harness, source):
    """A newer non-reusable receipt invalidates a complete cache after admission."""
    _cache(harness, "nasdaq")
    _cache(harness, "yfinance")
    assert [row[3] for row in _receipts(harness)] == ["complete", "complete"]
    claim, staging = _claim(harness, account=None)
    harness.clock.advance(1)
    if source == "nasdaq":
        unit, = forward_calendar_store.date_units([AS_OF], as_of=AS_OF)
        payload, endpoint = b'{"data":{"rows":[]}}', "calendar/earnings"
    else:
        unit, = forward_calendar_store.ticker_units(["AAPL"], as_of=AS_OF)
        payload = _earnings_frame().iloc[:0].to_csv(index=False).encode()
        endpoint = "earnings"
    record_unit_receipt(
        harness.conn, harness.store, unit, payload, source=source, endpoint=endpoint,
        received_at=harness.clock.now().isoformat(), response_kind="legitimate_empty")
    before = _receipts(harness), _head(harness)
    with pytest.raises(OpsError) as caught:
        _dispatch(claim, staging)
    assert caught.value.code == "CREDENTIAL_INVALID"
    assert harness.calls == []
    assert len(harness.connections) == 1
    _assert_closed(harness.connections)
    assert _budget_state(harness) == ([], [])
    assert (_receipts(harness), _head(harness)) == before
    assert not (staging / FORWARD_CALENDAR_RESULT_PATH).exists()


@pytest.mark.parametrize("injected_factories", [False, True])
def test_two_reserved_sources_charge_their_own_accounts(
        harness, monkeypatch, injected_factories):
    """A second real reservation models existing state, not multi-account submission."""
    harness.session = "time-not-supplied"
    claim, staging = _claim(harness)
    provider_budget.configure_account(
        harness.conn, "yfinance", "generation-1", remaining=20, live_reserve=2)
    provider_budget.reserve(harness.conn, claim, "yfinance", 3, clock=harness.clock)
    if injected_factories:
        def nasdaq(unit):
            assert unit["partition_key"] == AS_OF
            harness.observe_call("nasdaq", AS_OF)
            payload, rows = _nasdaq_payload(harness.session)
            return payload, "complete", {}, rows

        def earnings(ticker):
            harness.observe_call("yfinance", ticker)
            return _earnings_frame().to_csv(index=False).encode(), "complete", {}, []

        monkeypatch.setattr(providers, "nasdaq_calendar_fetcher", lambda: nasdaq)
        monkeypatch.setattr(providers, "yfinance_earnings_fetcher", lambda: earnings)
    _assert_published(harness, staging, _dispatch(claim, staging))
    assert harness.calls == [("nasdaq", AS_OF), ("yfinance", "AAPL")]
    assert len(harness.connections) == 2
    for source in ("nasdaq", "yfinance"):
        assert _counts(harness.conn, source) == (19, 1, 3, 1)
    assert [row[0] for row in _receipts(harness)] == ["nasdaq", "yfinance"]


def test_scalar_nasdaq_reservation_refuses_needed_yfinance_and_keeps_receipt(harness):
    harness.session = "time-not-supplied"
    claim, staging = _claim(harness)
    head = _head(harness)
    with pytest.raises(OpsError) as caught:
        _dispatch(claim, staging)
    assert caught.value.code == "CREDENTIAL_INVALID"
    assert harness.calls == [("nasdaq", AS_OF)]
    assert _counts(harness.conn, "nasdaq") == (19, 1, 3, 1)
    assert len(harness.connections) == 2
    assert [row[0] for row in _receipts(harness)] == ["nasdaq"]
    assert _head(harness) == head
    assert not (staging / FORWARD_CALENDAR_RESULT_PATH).exists()


@pytest.mark.parametrize(("condition", "code"), [
    ("cancelled", "CANCELLED"), ("backoff", "RATE_LIMITED"),
])
def test_second_source_rechecks_live_state_after_nasdaq_success(
        harness, monkeypatch, condition, code):
    """An admitted second account must still pass a fresh check at its own edge."""
    harness.session = "time-not-supplied"
    claim, staging = _claim(harness)
    provider_budget.configure_account(
        harness.conn, "yfinance", "generation-1", remaining=20, live_reserve=2)
    provider_budget.reserve(harness.conn, claim, "yfinance", 3, clock=harness.clock)
    http_get = nasdaq_calendar._requests_get

    def nasdaq_then_invalidate(url, *, timeout):
        response = http_get(url, timeout=timeout)
        if condition == "cancelled":
            request_cancel(harness.conn, claim.job_id, claim.attempt_id, clock=harness.clock)
        else:
            provider_budget.record_response(harness.conn, "yfinance", 429, clock=harness.clock)
        return response

    monkeypatch.setattr(nasdaq_calendar, "_requests_get", nasdaq_then_invalidate)
    head = _head(harness)
    with pytest.raises(OpsError) as caught:
        _dispatch(claim, staging)
    assert caught.value.code == code
    assert harness.calls == [("nasdaq", AS_OF)]
    assert _counts(harness.conn, "nasdaq") == (19, 1, 3, 1)
    assert _counts(harness.conn, "yfinance") == (20, 0, 3, 0)
    assert len(harness.connections) == 2
    _assert_closed(harness.connections)
    assert [row[0] for row in _receipts(harness)] == ["nasdaq"]
    assert _head(harness) == head
    assert not (staging / FORWARD_CALENDAR_RESULT_PATH).exists()


@pytest.mark.parametrize("source", ["nasdaq", "yfinance"])
def test_sqlite_charge_failure_rolls_back_both_counters_before_provider(harness, source):
    """An account-write failure rolls back the preceding reservation debit too."""
    if source == "yfinance":
        _cache(harness, "nasdaq")
    claim, staging = _claim(harness, account=source)
    with transaction(harness.conn):
        harness.conn.execute(
            "CREATE TRIGGER reject_provider_charge AFTER UPDATE OF remaining "
            "ON provider_accounts WHEN (SELECT used_calls FROM provider_reservations "
            "WHERE account = NEW.account) = 1 BEGIN "
            "SELECT RAISE(ABORT, 'private fixture database failure'); END")
    before = _budget_state(harness), _receipts(harness), _head(harness)
    with pytest.raises(OpsError) as caught:
        _dispatch(claim, staging)
    assert caught.value.code == "RESOURCE_UNAVAILABLE"
    assert caught.value.problem.message == "provider budget catalog is unavailable"
    assert caught.value.problem.details == {}
    assert "private fixture" not in str(caught.value)
    assert harness.calls == []
    assert len(harness.connections) == 1
    _assert_closed(harness.connections)
    assert (_budget_state(harness), _receipts(harness), _head(harness)) == before
    assert not (staging / FORWARD_CALENDAR_RESULT_PATH).exists()


@pytest.mark.parametrize("source", ["nasdaq", "yfinance"])
@pytest.mark.parametrize(("condition", "code"), [
    ("missing_account", "CREDENTIAL_INVALID"),
    ("missing", "CREDENTIAL_INVALID"),
    ("wrong_account", "CREDENTIAL_INVALID"),
    ("released", "CREDENTIAL_INVALID"),
    ("exhausted", "RESOURCE_UNAVAILABLE"),
    ("blocked", "CREDENTIAL_INVALID"),
    ("backoff", "RATE_LIMITED"),
    ("cancelled", "CANCELLED"),
    ("expired", "LEASE_LOST"),
    ("stale", "LEASE_LOST"),
])
def test_unusable_reservation_or_attempt_refuses_before_provider(
        harness, source, condition, code):
    if source == "yfinance":
        _cache(harness, "nasdaq")
    account = "other-account" if condition == "wrong_account" else source
    if condition == "missing_account":
        account = None
    claim, staging = _claim(harness, account=account)
    if condition == "missing":
        with transaction(harness.conn):
            harness.conn.execute("DELETE FROM provider_reservations WHERE account = ?",
                                 (account,))
    elif condition == "released":
        with transaction(harness.conn):
            release_reservations(harness.conn, claim.attempt_id, harness.clock.now(),
                                 reason="fixture release before worker dispatch")
    elif condition == "exhausted":
        for _ in range(3):
            provider_budget.before_request(harness.conn, claim, account, clock=harness.clock)
    elif condition in ("blocked", "backoff"):
        provider_budget.record_response(
            harness.conn, account, 401 if condition == "blocked" else 429,
            clock=harness.clock)
    elif condition == "cancelled":
        request_cancel(harness.conn, claim.job_id, claim.attempt_id, clock=harness.clock)
    elif condition == "expired":
        harness.clock.advance(harness.supervisor.lease_seconds)
    elif condition == "stale":
        with transaction(harness.conn):
            harness.conn.execute("UPDATE jobs SET fence = fence + 1 WHERE job_id = ?",
                                 (claim.job_id,))
    before = _budget_state(harness), _receipts(harness), _head(harness)
    with pytest.raises(OpsError) as caught:
        _dispatch(claim, staging)
    assert caught.value.code == code
    assert harness.calls == []
    assert len(harness.connections) == 1
    assert (_budget_state(harness), _receipts(harness), _head(harness)) == before
    assert not (staging / FORWARD_CALENDAR_RESULT_PATH).exists()


@pytest.mark.parametrize("source", ["nasdaq", "yfinance"])
def test_failed_provider_fetch_stays_charged(harness, source):
    if source == "yfinance":
        _cache(harness, "nasdaq")
    claim, staging = _claim(harness, account=source)
    harness.fail_source = source
    before = _head(harness), _receipts(harness)
    with pytest.raises(OpsError) as caught:
        _dispatch(claim, staging)
    assert caught.value.code == "TRANSIENT_SOURCE"
    assert len(harness.calls) == len(harness.connections) == 1
    assert _counts(harness.conn, source) == (19, 1, 3, 1)
    assert (_head(harness), _receipts(harness)) == before
    assert not (staging / FORWARD_CALENDAR_RESULT_PATH).exists()


@pytest.mark.parametrize(("field", "value"), [
    ("as_of", "not-a-date"), ("horizon_days", 0), ("tickers", [""]),
    ("scope", "unapproved"), ("expected_head_generation", -1),
    ("parent_snapshot_id", ""), ("refresh_plan_hash", "not-a-hash"),
    ("expected_head_snapshot_id", ""),
    ("catalog_path", "missing-catalog"), ("objects_root", "missing-objects"),
])
def test_worker_revalidates_store_inputs_before_lazy_budget_connection(
        harness, field, value):
    """Corrupted worker parameters cannot open a guard or spend an admitted budget."""
    claim, staging = _claim(harness)
    if field in ("catalog_path", "objects_root"):
        value = str(harness.root / value)
    claim.spec.parameters[field] = value
    before = _budget_state(harness), _receipts(harness), _head(harness)
    with pytest.raises(OpsError) as caught:
        _dispatch(claim, staging)
    assert caught.value.code == "INVALID_REQUEST"
    assert harness.calls == harness.connections == []
    assert (_budget_state(harness), _receipts(harness), _head(harness)) == before


def test_missing_budget_catalog_refuses_lazily_without_creating_file(tmp_path):
    path = tmp_path / "missing #?.sqlite"
    calls = []
    fetch = provider_budget.budgeted_fetcher(
        lambda unit: calls.append(unit), catalog_path=path,
        attempt_id="missing-attempt", fence=1, account="nasdaq", clock=None)
    assert not path.exists()
    with pytest.raises(OpsError) as caught:
        fetch({"partition_key": AS_OF})
    assert caught.value.code == "RESOURCE_UNAVAILABLE"
    assert calls == []
    assert not path.exists()


def test_catalog_must_exist_refuses_creation_and_default_still_creates(tmp_path):
    path = tmp_path / "catalog #?.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        catalog_module.connect(path, must_exist=True)
    assert not path.exists()
    with closing(catalog_module.connect(path)) as conn:
        conn.execute("CREATE TABLE example (value INTEGER)")
        conn.execute("INSERT INTO example VALUES (7)")
    with closing(catalog_module.connect(path, must_exist=True)) as conn:
        assert conn.execute("SELECT value FROM example").fetchone()[0] == 7
