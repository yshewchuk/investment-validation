"""Synthetic catalogs and immutable bytes exercise optional accounting hooks."""
import socket
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from types import SimpleNamespace

import pytest

from engine.v2.data.incremental import (
    RAW_SCHEMA_REF,
    RawPayload,
    RawReceiptRecord,
    cache_raw_receipt,
)
from engine.v2.foundation import ArtifactError, ArtifactStore, artifact_reference, format_timestamp
from engine.v2.ops import provider_budget
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, make_problem
from engine.v2.ops.lifecycle import Outcome, commit_attempt, request_cancel, verify_fence
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.scheduler import claim_next
from tests.ops_support import catalog, enqueue_claim, sample

ACCOUNT = "massive-stocks"
OTHER_ACCOUNT = "synthetic-other"
BODY = b'{"status":"OK","results":[{"ticker":"SYNTH"}]}'


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("accounting hook test attempted network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def harness(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    for account in (ACCOUNT, OTHER_ACCOUNT):
        provider_budget.configure_account(conn, account, "synthetic", 20, 0)
    conn.execute(
        "CREATE TABLE synthetic_occurrences ("
        "raw_receipt_id TEXT NOT NULL REFERENCES data_raw_receipts(raw_receipt_id), "
        "first_received_at TEXT NOT NULL, observed_at TEXT NOT NULL, cache_hit INTEGER NOT NULL)")
    yield SimpleNamespace(conn=conn, clock=clock, supervisor=supervisor, path=tmp_path / "ops.sqlite",
                          store=ArtifactStore(tmp_path / "objects"))
    conn.close()


def _claim(h, key="one", calls=3):
    claim = enqueue_claim(h.conn, h.clock, h.supervisor, key)
    provider_budget.reserve(h.conn, claim, ACCOUNT, calls, clock=h.clock)
    return claim


def _finish(h, claim, *, retry=False):
    outcome = Outcome(succeeded=not retry, process_state="exited", exit_code=int(retry),
                      failure=make_problem("TRANSIENT_SOURCE", "synthetic retry") if retry else None)
    return commit_attempt(h.conn, claim.attempt_id, claim.fence, outcome, clock=h.clock)


def _accounting(h):
    return tuple(tuple(row) for table in ("provider_accounts", "provider_reservations")
                 for row in h.conn.execute(f"SELECT * FROM {table} ORDER BY account"))


def _retry(h, previous):
    assert _finish(h, previous, retry=True) == "retry_wait"
    h.clock.advance(2)
    claim = claim_next(h.conn, policy=DEFAULT_POLICY, sample=sample(h.clock),
                       supervisor=h.supervisor, clock=h.clock)
    assert claim.job_id == previous.job_id and claim.attempt_id != previous.attempt_id
    provider_budget.reserve(h.conn, claim, ACCOUNT, 3, clock=h.clock)
    return claim


def _assert_limited(h, attempt):
    before = _accounting(h)
    with pytest.raises(OpsError, match="RESOURCE_LIMIT_EXCEEDED"):
        provider_budget.before_request(h.conn, attempt, ACCOUNT, clock=h.clock,
                                       max_calls_per_job=2, minimum_interval_seconds=13)
    assert _accounting(h) == before


def test_job_cap_counts_released_attempts_and_current_calls(harness):
    h = harness
    first = _claim(h)
    provider_budget.before_request(h.conn, first, ACCOUNT, clock=h.clock, max_calls_per_job=2)
    second = _retry(h, first)
    # Supplied job identity has no authority; the live attempt's catalog row does.
    trusted = SimpleNamespace(attempt_id=second.attempt_id, fence=second.fence,
                              job_id="unrelated-job")
    provider_budget.before_request(h.conn, trusted, ACCOUNT, clock=h.clock, max_calls_per_job=2)
    _assert_limited(h, second)
    third = _retry(h, second)
    _assert_limited(h, third)
    rows = h.conn.execute(
        "SELECT used_calls,released_at FROM provider_reservations ORDER BY rowid").fetchall()
    assert [row["used_calls"] for row in rows] == [1, 1, 0]
    assert all(row["released_at"] is not None for row in rows[:2])
    assert h.conn.execute("SELECT remaining FROM provider_accounts WHERE account=?",
                          (ACCOUNT,)).fetchone()[0] == 18


def test_job_cap_excludes_other_jobs_and_other_accounts(harness):
    h = harness
    other_job = _claim(h, "other")
    for _ in range(2):
        provider_budget.before_request(h.conn, other_job, ACCOUNT, clock=h.clock)
    _finish(h, other_job)
    current = _claim(h)
    provider_budget.reserve(h.conn, current, OTHER_ACCOUNT, 3, clock=h.clock)
    provider_budget.before_request(h.conn, current, OTHER_ACCOUNT, clock=h.clock)
    provider_budget.before_request(h.conn, current, ACCOUNT, clock=h.clock, max_calls_per_job=1)
    before = _accounting(h)
    with pytest.raises(OpsError, match="RESOURCE_LIMIT_EXCEEDED"):
        provider_budget.before_request(h.conn, current, ACCOUNT, clock=h.clock, max_calls_per_job=1)
    assert _accounting(h) == before
    assert [tuple(row) for row in h.conn.execute(
        "SELECT account,remaining FROM provider_accounts ORDER BY account")] == [
            (ACCOUNT, 17), (OTHER_ACCOUNT, 19)]


class IntSubclass(int):
    pass


@pytest.mark.parametrize("limit", [False, True, 0, -1, 1.0, "1", [], {}, IntSubclass(1)])
def test_invalid_job_cap_refuses_before_catalog_access(harness, limit):
    h = harness
    claim = _claim(h)
    before = _accounting(h)
    statements = []
    h.conn.set_trace_callback(statements.append)
    try:
        with pytest.raises(OpsError, match="INVALID_REQUEST"):
            provider_budget.before_request(h.conn, claim, ACCOUNT, clock=h.clock,
                                           max_calls_per_job=limit)
    finally:
        h.conn.set_trace_callback(None)
    assert statements == [] and _accounting(h) == before


@pytest.mark.parametrize("limit", [1, 2 ** 100])
def test_positive_exact_integer_cap_accepts_one_charge(harness, limit):
    h = harness
    claim = _claim(h)
    provider_budget.before_request(h.conn, claim, ACCOUNT, clock=h.clock, max_calls_per_job=limit)
    assert h.conn.execute("SELECT used_calls FROM provider_reservations").fetchone()[0] == 1


@pytest.mark.parametrize("options", [{}, {"max_calls_per_job": None}])
def test_omitted_job_cap_preserves_reservation_limit(harness, options):
    h = harness
    claim = _claim(h, calls=2)
    provider_budget.before_request(h.conn, claim, ACCOUNT, clock=h.clock, max_calls_per_job=1)
    provider_budget.before_request(h.conn, claim, ACCOUNT, clock=h.clock, **options)
    before = _accounting(h)
    with pytest.raises(OpsError, match="RESOURCE_UNAVAILABLE"):
        provider_budget.before_request(h.conn, claim, ACCOUNT, clock=h.clock, **options)
    assert _accounting(h) == before


def test_cancelled_fence_refuses_before_job_cap_or_charge(harness):
    h = harness
    claim = _claim(h)
    provider_budget.before_request(h.conn, claim, ACCOUNT, clock=h.clock, max_calls_per_job=1)
    request_cancel(h.conn, claim.job_id, claim.attempt_id, clock=h.clock)
    before = _accounting(h)
    with pytest.raises(OpsError, match="CANCELLED"):
        provider_budget.before_request(h.conn, claim, ACCOUNT, clock=h.clock, max_calls_per_job=1)
    assert _accounting(h) == before


def test_concurrent_requests_cannot_both_spend_last_job_call(harness):
    h = harness
    claim = _claim(h)
    def charge():
        with closing(connect(h.path, must_exist=True)) as conn:
            try:
                provider_budget.before_request(conn, claim, ACCOUNT, clock=h.clock, max_calls_per_job=1)
                return "charged"
            except OpsError as error:
                return error.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: charge(), range(2)))
    assert sorted(results) == ["RESOURCE_LIMIT_EXCEEDED", "charged"]
    assert h.conn.execute("SELECT used_calls FROM provider_reservations").fetchone()[0] == 1
    assert h.conn.execute("SELECT remaining,uncertain FROM provider_accounts WHERE account=?",
                          (ACCOUNT,)).fetchone()[:] == (19, 1)


def _cache(h, *, response_meta=None, **options):
    return cache_raw_receipt(
        h.conn, h.store, RawPayload(payload=BODY, response_kind="complete",
                                   response_meta={"first": True} if response_meta is None else response_meta),
        source=ACCOUNT, endpoint="/synthetic", request={"ticker": "SYNTH"},
        received_at=format_timestamp(h.clock.now()), **options)


def _record_occurrence(h, record):
    assert h.conn.in_transaction and isinstance(record, RawReceiptRecord)
    raw = h.conn.execute("SELECT * FROM data_raw_receipts WHERE raw_receipt_id=?",
                         (record.raw_receipt_id,)).fetchone()
    assert raw["received_at"] == record.received_at and raw["raw_hash"] == record.raw_hash
    h.conn.execute("INSERT INTO synthetic_occurrences VALUES (?,?,?,?)",
                   (record.raw_receipt_id, record.received_at,
                    format_timestamp(h.clock.now()), int(record.cache_hit)))


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("fenced", [False, True])
def test_receipt_callback_runs_atomically_after_verified_io(harness, monkeypatch, cached, fenced):
    h = harness
    claim = _claim(h)
    original = _cache(h) if cached else None
    h.clock.advance(10)
    events = []
    for name in ("publish_bytes", "verify"):
        method = getattr(h.store, name)
        def observe(*args, _method=method, _name=name, **kwargs):
            assert not h.conn.in_transaction
            events.append(_name)
            return _method(*args, **kwargs)
        monkeypatch.setattr(h.store, name, observe)
    def fence():
        assert h.conn.in_transaction
        verify_fence(h.conn, claim.attempt_id, claim.fence, h.clock.now())
        events.append("fence")
    def record(receipt):
        events.append("record")
        _record_occurrence(h, receipt)
    result = _cache(h, response_meta={"second": True}, record_receipt=record,
                    fence_check=fence if fenced else None)
    assert events == (["verify"] if cached else ["publish_bytes"]) + (
        ["fence", "record"] if fenced else ["record"])
    assert result.cache_hit is cached and not h.conn.in_transaction
    expected_time = original.received_at if cached else format_timestamp(h.clock.now())
    assert result.received_at == expected_time
    assert result.response_meta == ({"first": True} if cached else {"second": True})
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 1
    assert tuple(h.conn.execute("SELECT * FROM synthetic_occurrences").fetchone()) == (
        result.raw_receipt_id, expected_time, format_timestamp(h.clock.now()), int(cached))


@pytest.mark.parametrize("cached", [False, True])
def test_receipt_callback_failure_rolls_back_receipt_and_occurrence(harness, cached):
    h = harness
    claim = _claim(h)
    original = _cache(h) if cached else None
    h.clock.advance(10)
    def record(receipt):
        _record_occurrence(h, receipt)
        raise RuntimeError("synthetic occurrence failure")
    with pytest.raises(RuntimeError, match="synthetic occurrence failure"):
        _cache(h, record_receipt=record,
               fence_check=lambda: verify_fence(h.conn, claim.attempt_id, claim.fence, h.clock.now()))
    assert not h.conn.in_transaction
    assert h.conn.execute("SELECT count(*) FROM synthetic_occurrences").fetchone()[0] == 0
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == int(cached)
    assert h.store.read_verified(artifact_reference(BODY, RAW_SCHEMA_REF)) == BODY
    if cached:
        assert h.conn.execute("SELECT received_at FROM data_raw_receipts").fetchone()[0] == original.received_at
    recovered = _cache(h, record_receipt=lambda receipt: _record_occurrence(h, receipt))
    assert recovered.cache_hit is cached
    assert h.conn.execute("SELECT count(*) FROM synthetic_occurrences").fetchone()[0] == 1


@pytest.mark.parametrize("cached", [False, True])
def test_rejected_live_fence_prevents_receipt_callback(harness, cached):
    h = harness
    claim = _claim(h)
    if cached:
        _cache(h)
    request_cancel(h.conn, claim.job_id, claim.attempt_id, clock=h.clock)
    def forbidden(_):
        pytest.fail("receipt callback ran after rejected fence")
    with pytest.raises(OpsError, match="CANCELLED"):
        _cache(h, record_receipt=forbidden,
               fence_check=lambda: verify_fence(h.conn, claim.attempt_id, claim.fence, h.clock.now()))
    assert not h.conn.in_transaction
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == int(cached)
    assert h.conn.execute("SELECT count(*) FROM synthetic_occurrences").fetchone()[0] == 0


def test_corrupt_replay_body_refuses_before_receipt_callback(harness):
    h = harness
    _cache(h)
    path = h.store.verify(artifact_reference(BODY, RAW_SCHEMA_REF))
    path.chmod(0o644)
    path.write_bytes(BODY + b"corruption")
    def forbidden(_):
        pytest.fail("receipt callback trusted corrupt raw bytes")
    with pytest.raises(ArtifactError, match="INTEGRITY_FAILED"):
        _cache(h, record_receipt=forbidden)
    assert not h.conn.in_transaction
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 1
    assert h.conn.execute("SELECT count(*) FROM synthetic_occurrences").fetchone()[0] == 0
