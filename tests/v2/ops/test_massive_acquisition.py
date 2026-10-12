"""Real synthetic catalogs, leases and raw objects; no provider network or credentials."""
import hashlib
import json
import os
import socket
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.data.incremental import RawPayload, cache_raw_receipt
from engine.v2.foundation import (
    ArtifactStore,
    canonical_json,
    content_hash,
    format_timestamp,
    to_document,
)
from engine.v2.ops import provider_budget
from engine.v2.ops.catalog import connect, transaction
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.lifecycle import Outcome, commit_attempt, request_cancel
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.stores.massive import acquisition
from engine.v2.ops.submission import submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import POLICY, catalog, sample

SOURCE = "massive-stocks"
REQUEST = {"schema_version": "massive_request.v1", "source": SOURCE, "transport": "rest",
           "endpoint": "/v2/aggs/ticker/SYNTH/range/1/day/2026-08-01/2026-08-31",
           "dataset": "daily", "ticker": "SYNTH", "start": "2026-08-01", "end": "2026-08-31",
           "adjustment_basis": "raw", "params": {"adjusted": False, "sort": "asc", "limit": 50000}}
BODY = canonical_json({"status": "OK", "ticker": "SYNTH", "adjusted": False,
                       "resultsCount": 1, "results": [{"t": 1785542400000, "o": 10.,
                                                          "h": 12., "l": 9., "c": 11., "v": 100.}]}).encode()
PARAMS = acquisition.Parameters(ticker="SYNTH", dataset="daily", start="2026-08-01", end="2026-08-31",
                                as_of="2026-09-12", adjustment="raw", expected_ids=(content_hash(REQUEST),))


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic acquisition attempted network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def harness(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    provider_budget.configure_account(conn, SOURCE, "synthetic", 10, 0)
    h = SimpleNamespace(conn=conn, clock=clock, supervisor=supervisor, root=tmp_path,
                        store=ArtifactStore(tmp_path / "store"))
    yield h
    conn.close()


def enqueue(h, key="one", params=PARAMS, **changes):
    job = JobSpec(kind="massive_acquisition", implementation_ref="synthetic-code", spec_hash=None,
                  environment_ref="synthetic-env", parameters=to_document(params), output_namespace="shadow",
                  resource_class="io_fetch", provider_budget_ref=SOURCE, retry_policy_ref="bounded",
                  checkpoint_contract_ref="massive_acquisition.v1")
    return submit(h.conn, registry(), POLICY, SubmitRequest(namespace="shadow", idempotency_key=key,
                  principal="operator", job=replace(job, **changes)), clock=h.clock)


def claim(h):
    return claim_next(h.conn, policy=DEFAULT_POLICY, sample=sample(h.clock),
                      supervisor=h.supervisor, clock=h.clock)


def run(h, attempt, fetch, params=PARAMS):
    root = h.root / attempt.attempt_id
    root.mkdir(exist_ok=True)
    result = acquisition.run(to_document(params), root,
                             {"attempt_id": attempt.attempt_id, "fence": attempt.fence,
                              "catalog_path": str(h.root / "ops.sqlite"), "objects_root": str(h.store.root)},
                             fetch=fetch, clock=h.clock)
    return result, json.loads((root / "massive_acquisition.json").read_bytes())


def successful(request, charge):
    assert request == REQUEST
    charge()
    return 200, BODY


def finish(h, attempt):
    commit_attempt(h.conn, attempt.attempt_id, attempt.fence,
                   Outcome(succeeded=True, process_state="exited", exit_code=0), clock=h.clock)


def test_fresh_acquisition_and_repeated_job_reuse_only_raw_cache(harness):
    h = harness
    enqueue(h)
    first = claim(h)
    protocol, record = run(h, first, successful)
    raw = h.conn.execute("SELECT * FROM data_raw_receipts").fetchone()
    assert json.loads(raw["request_json"]) == REQUEST
    assert raw["raw_hash"] == "sha256:" + hashlib.sha256(BODY).hexdigest()
    assert record == {"schema_version": "massive_acquisition.v1", "request_hash": content_hash(REQUEST),
                      "raw_receipt_id": raw["raw_receipt_id"], "raw_hash": raw["raw_hash"],
                      "first_received_at": "2026-09-12T00:00:00.000000Z",
                      "observed_at": "2026-09-12T00:00:00.000000Z", "attempt_id": first.attempt_id,
                      "fence": first.fence, "cache_hit": False, "http_attempts": 1,
                      "policy_version": "massive_stocks_basic.2026-10-10"}
    assert protocol == {"outputs": [{"name": "massive_acquisition", "path": "massive_acquisition.json",
                                     "schema": "massive_acquisition.v1"}],
                        "completed_ids": [content_hash(REQUEST)], "no_work": False}
    assert h.conn.execute("SELECT remaining,uncertain FROM provider_accounts").fetchone()[:] == (9, 1)
    assert h.conn.execute("SELECT used_calls FROM provider_reservations").fetchone()[0] == 1
    # Both pending same-key work and a later distinct job must not issue duplicate HTTP.
    assert enqueue(h).job_id == first.job_id
    enqueue(h, "two")
    assert claim(h) is None
    finish(h, first)
    h.clock.advance(12)
    assert claim(h) is None
    h.clock.advance(1)
    second = claim(h)
    assert second is not None
    def forbidden(*_):
        raise AssertionError("verified cache hit attempted HTTP")
    protocol2, reused = run(h, second, forbidden)
    assert reused["raw_receipt_id"] == record["raw_receipt_id"]
    assert reused["first_received_at"] == record["first_received_at"]
    assert reused["observed_at"] == "2026-09-12T00:00:13.000000Z"
    assert reused["attempt_id"] == second.attempt_id
    assert reused["cache_hit"] is True and reused["http_attempts"] == 0 and protocol2["no_work"] is True
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 1
    assert h.conn.execute("SELECT remaining FROM provider_accounts").fetchone()[0] == 9
    assert h.conn.execute("SELECT used_calls FROM provider_reservations WHERE attempt_id=?",
                          (second.attempt_id,)).fetchone()[0] == 0
    service = object.__new__(Service)
    assert service._cache_allowed(second) is False
    service._store_domains = lambda _: ()
    assert service._cache_allowed(replace(second, spec=replace(second.spec, kind="artifact_check"))) is True


SCOPE_REFUSALS = [{"provider_calls": 0},
    {"provider_calls": 2}, {"provider_account": "other"}, {"policy_version": "other"},
    {"expected_ids": ("wrong",)}, {"dataset": "minute"}, {"adjustment": "dividend_adjusted"}]


@pytest.mark.parametrize("change", [*SCOPE_REFUSALS, {"provider_calls": True}])
def test_parameter_refusals_precede_storage(harness, monkeypatch, change):
    enqueue(harness)
    attempt = claim(harness)
    def forbidden(*_, **__):
        pytest.fail("invalid worker scope reached storage or provider access")
    monkeypatch.setattr(acquisition, "connect", forbidden)
    with pytest.raises(OpsError) as error:
        run(harness, attempt, forbidden, replace(PARAMS, **change))
    assert error.value.code == "INVALID_REQUEST"
    assert harness.conn.execute("SELECT used_calls FROM provider_reservations").fetchone()[0] == 0


@pytest.mark.parametrize("change", SCOPE_REFUSALS)
def test_submission_scope_refusals_use_registered_validator(harness, monkeypatch, change):
    checked = []
    original = acquisition.parameter_problems
    def validate(job, params):
        checked.append(params)
        return original(job, params)
    monkeypatch.setattr(acquisition, "parameter_problems", validate)
    params = replace(PARAMS, **change)
    with pytest.raises(OpsError) as error:
        enqueue(harness, params=params)
    assert error.value.code == "INVALID_REQUEST"
    assert checked == [params]
    assert harness.conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    assert harness.conn.execute("SELECT count(*) FROM provider_reservations").fetchone()[0] == 0


@pytest.mark.parametrize("change", [{"provider_budget_ref": "other"}, {"input_refs": ("artifact-1",)},
                                    {"dependency_job_ids": ("job-1",)}])
def test_submission_rejects_unbound_account_and_dependencies(harness, change):
    with pytest.raises(OpsError) as error:
        enqueue(harness, **change)
    assert error.value.code == "INVALID_REQUEST"
    assert harness.conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0


@pytest.mark.parametrize("status,code,blocked,cooldown", [(401, "CREDENTIAL_INVALID", "CREDENTIAL_INVALID", 13),
    (403, "SOURCE_INVALID", None, 13), (404, "SOURCE_NOT_FOUND", None, 13),
    (429, "RATE_LIMITED", None, 65), (503, "TRANSIENT_SOURCE", None, 13), (302, "SOURCE_INVALID", None, 13)])
def test_http_refusals_accounting_and_no_cache(harness, status, code, blocked, cooldown):
    h = harness
    enqueue(h)
    attempt = claim(h)
    def fetch(_, charge):
        charge()
        return status, b"never persisted provider error body"
    with pytest.raises(OpsError) as error:
        run(h, attempt, fetch)
    assert error.value.code == code
    if status == 403:
        assert error.value.problem.details == {"reason": "CAPABILITY_UNAVAILABLE"}
    row = h.conn.execute("SELECT * FROM provider_accounts").fetchone()
    assert row["remaining"] == 9 and row["uncertain"] == 1 and row["blocked_code"] == blocked
    h.clock.advance(cooldown)
    assert row["next_eligible_at"] == format_timestamp(h.clock.now())
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 0
    assert not (h.root / attempt.attempt_id / "massive_acquisition.json").exists()


@pytest.mark.parametrize("body,code", [(b"malformed", "SOURCE_INVALID"),
    (canonical_json(json.loads(BODY) | {"results": [], "resultsCount": 0}).encode(), "SOURCE_EMPTY"),
    (canonical_json(json.loads(BODY) | {"next_url": "https://untrusted.invalid/secret"}).encode(), "SOURCE_NOT_FINAL"),
    (canonical_json(json.loads(BODY) | {"adjusted": True}).encode(), "SOURCE_INVALID"),
    (b"x" * (16 * 1024 * 1024 + 1), "RESOURCE_LIMIT_EXCEEDED")])
def test_unusable_bodies_never_commit_complete_receipts(harness, body, code):
    h = harness
    enqueue(h)
    attempt = claim(h)
    def fetch(_, charge):
        charge()
        h.clock.advance(5)
        return 200, body
    with pytest.raises(OpsError) as error:
        run(h, attempt, fetch)
    assert error.value.code == code
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 0
    assert not h.store.root.exists()
    assert h.conn.execute("SELECT remaining,uncertain,next_eligible_at FROM provider_accounts").fetchone()[:] == (
        9, 1, "2026-09-12T00:00:18.000000Z")


@pytest.mark.parametrize("when", ["before_charge", "during_fetch", "during_publication"])
def test_cancellation_fences_raw_catalog_visibility(harness, monkeypatch, when):
    h = harness
    enqueue(h)
    attempt = claim(h)
    def cancel():
        request_cancel(h.conn, attempt.job_id, expected_attempt_id=attempt.attempt_id, clock=h.clock)
    original = ArtifactStore.publish_bytes
    def publish(store, *args, **kwargs):
        result = original(store, *args, **kwargs)
        cancel()
        return result
    if when == "before_charge":
        cancel()
    if when == "during_publication":
        monkeypatch.setattr(ArtifactStore, "publish_bytes", publish)
    def fetch(request, charge):
        charge()
        if when == "during_fetch":
            cancel()
        return 200, BODY
    with pytest.raises(OpsError) as error:
        run(h, attempt, fetch)
    assert error.value.code == "CANCELLED"
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 0
    assert not (h.root / attempt.attempt_id / "massive_acquisition.json").exists()
    assert h.conn.execute("SELECT used_calls FROM provider_reservations").fetchone()[0] == (when != "before_charge")


@pytest.mark.parametrize("status", [200, 401, 429])
def test_cancelled_fetch_cannot_mutate_accounting_after_charge(harness, status):
    h = harness
    enqueue(h)
    attempt = claim(h)
    before = []
    def fetch(_, charge):
        charge()
        before.append(tuple(h.conn.execute("SELECT * FROM provider_accounts").fetchone()))
        h.clock.advance(5)
        request_cancel(h.conn, attempt.job_id, expected_attempt_id=attempt.attempt_id, clock=h.clock)
        return status, BODY
    with pytest.raises(OpsError) as error:
        run(h, attempt, fetch)
    assert error.value.code == "CANCELLED"
    assert tuple(h.conn.execute("SELECT * FROM provider_accounts").fetchone()) == before[0]
    assert h.conn.execute("SELECT remaining,uncertain,next_eligible_at,blocked_code FROM provider_accounts").fetchone()[:] == (
        9, 1, "2026-09-12T00:00:13.000000Z", None)
    assert h.conn.execute("SELECT used_calls FROM provider_reservations").fetchone()[0] == 1
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 0
    assert not (h.root / attempt.attempt_id / "massive_acquisition.json").exists()


def test_transport_failure_keeps_charge_and_closes_budget_transaction(harness):
    h = harness
    enqueue(h)
    attempt = claim(h)
    def fetch(_, charge):
        charge()
        with closing(connect(h.root / "ops.sqlite", must_exist=True)) as independent:
            with transaction(independent):
                assert independent.execute("SELECT used_calls FROM provider_reservations").fetchone()[0] == 1
        raise fail("TRANSIENT_SOURCE", "synthetic transport interruption")
    with pytest.raises(OpsError) as error:
        run(h, attempt, fetch)
    assert error.value.code == "TRANSIENT_SOURCE"
    assert h.conn.execute("SELECT remaining,uncertain FROM provider_accounts").fetchone()[:] == (9, 1)
    h.clock.advance(13)
    with pytest.raises(OpsError) as exhausted:
        provider_budget.before_request(h.conn, attempt, SOURCE, clock=h.clock)
    assert exhausted.value.code == "RESOURCE_UNAVAILABLE"
    assert h.conn.execute("SELECT max_attempts FROM jobs").fetchone()[0] == 1
    state = commit_attempt(h.conn, attempt.attempt_id, attempt.fence,
                           Outcome(succeeded=False, process_state="exited", failure=error.value.problem), clock=h.clock)
    assert state == "failed"
    assert claim(h) is None
    assert h.conn.execute("SELECT count(*) FROM attempts").fetchone()[0] == 1


@pytest.mark.parametrize("code,recorded_status", [("TRANSIENT_SOURCE", 503),
    ("SOURCE_INVALID", 200), ("RESOURCE_LIMIT_EXCEEDED", 200)])
def test_transport_refusal_accounting_preserves_observed_status(harness, monkeypatch, code, recorded_status):
    h = harness
    enqueue(h)
    attempt = claim(h)
    statuses = []
    original = provider_budget.record_response
    def record(conn, account, status, **kwargs):
        statuses.append(status)
        return original(conn, account, status, **kwargs)
    monkeypatch.setattr(provider_budget, "record_response", record)
    def fetch(_, charge):
        charge()
        h.clock.advance(5)
        raise fail(code, "synthetic transport refusal")
    with pytest.raises(OpsError) as error:
        run(h, attempt, fetch)
    assert error.value.code == code
    assert statuses == [recorded_status]
    row = h.conn.execute("SELECT remaining,uncertain,next_eligible_at,blocked_code FROM provider_accounts").fetchone()
    assert tuple(row) == (9, 1, "2026-09-12T00:00:18.000000Z", None)
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 0


def test_existing_raw_receipt_replay_is_fenced_in_its_own_transaction(harness):
    h = harness
    payload = RawPayload(payload=BODY, response_kind="complete", response_meta={})
    arguments = dict(source=SOURCE, endpoint=REQUEST["endpoint"], request=REQUEST,
                     received_at=format_timestamp(h.clock.now()))
    original = cache_raw_receipt(h.conn, h.store, payload, **arguments)
    def refuse():
        assert h.conn.in_transaction
        raise fail("LEASE_LOST", "synthetic stale fence")
    with pytest.raises(OpsError) as error:
        cache_raw_receipt(h.conn, h.store, payload, fence_check=refuse, **arguments)
    assert error.value.code == "LEASE_LOST"
    assert not h.conn.in_transaction
    assert h.conn.execute("SELECT raw_receipt_id FROM data_raw_receipts").fetchone()[0] == original.raw_receipt_id
    checked = []
    replay = cache_raw_receipt(h.conn, h.store, payload, fence_check=lambda: checked.append(h.conn.in_transaction),
                               **(arguments | {"received_at": "2026-09-13T00:00:00.000000Z"}))
    assert checked == [True] and replay.cache_hit is True
    assert replay.received_at == original.received_at


@pytest.mark.parametrize("cached,forwarded", [(False, False), (True, False), (True, True)])
def test_real_executor_and_worker_use_trusted_store_and_no_ambient_credentials(harness, monkeypatch, cached, forwarded):
    from engine.v2.ops import executor, providers
    from engine.v2.ops.stages import validate_result

    h = harness
    h.clock.value = datetime.now(timezone.utc)
    if cached:
        cache_raw_receipt(h.conn, h.store, RawPayload(payload=BODY, response_kind="complete", response_meta={}),
                          source=SOURCE, endpoint=REQUEST["endpoint"], request=REQUEST,
                          received_at=format_timestamp(h.clock.now()))
    enqueue(h)
    attempt = claim(h)
    attempt = replace(attempt, resources=replace(attempt.resources,
                      assigned_cpu_ids=(min(os.sched_getaffinity(0)),)))
    marker = "synthetic-massive-key-not-a-credential"
    synthetic = {"MASSIVE_API_KEY": marker, "ORATS_API_KEY": "unrelated-synthetic"} if forwarded else {}
    monkeypatch.setattr(providers, "os", SimpleNamespace(environ=synthetic))
    launched = []
    original = executor.subprocess.Popen
    def launch(*args, **kwargs):
        environment = dict(kwargs["env"])
        assert "ORATS_API_KEY" not in environment
        launched.append(environment.pop("MASSIVE_API_KEY", None))
        # Verify the real allowlist/launch boundary, then remove the synthetic
        # key so even a cache regression cannot make a provider call in the child.
        return original(*args, **(kwargs | {"env": environment}))
    monkeypatch.setattr(executor.subprocess, "Popen", launch)
    running = executor.launch(h.conn, attempt, registry().get("massive_acquisition"), h.store,
                              Path(__file__).resolve().parents[3], clock=h.clock, boot_id="boot")
    try:
        exit_code = running.process.wait(timeout=30)
        executor._read_result(running)
        document = json.loads(running.data)
        assert launched == [marker if forwarded else None]
        assert marker not in running.data.decode() + str(tuple(h.conn.iterdump()))
        assert exit_code == (0 if cached else 1)
        if cached:
            assert validate_result(attempt, document) == [
                {"name": "massive_acquisition", "path": "massive_acquisition.json", "schema": "massive_acquisition.v1"}]
            output = h.store.staging_dir(attempt.attempt_id) / "massive_acquisition.json"
            assert json.loads(output.read_bytes())["cache_hit"] is True
        else:
            assert document["failure"] == "CREDENTIAL_INVALID"
            assert document["problem"]["retryable"] is False
        assert h.conn.execute("SELECT used_calls FROM provider_reservations").fetchone()[0] == 0
    finally:
        if running.process.poll() is None:
            running.process.kill()
            running.process.wait(timeout=5)
        os.close(running.result_fd)


@pytest.mark.parametrize("dataset,date_field", [("splits", "execution_date"), ("dividends", "ex_dividend_date")])
def test_corporate_actions_are_stored_under_their_own_canonical_identity(harness, dataset, date_field):
    h = harness
    expected = REQUEST | {"dataset": dataset, "endpoint": "/stocks/v1/" + dataset,
                          "adjustment_basis": "corporate_action", "params": {
                              "ticker": "SYNTH", date_field + ".gte": "2026-08-01",
                              date_field + ".lte": "2026-08-31", "sort": date_field + ".asc", "limit": 5000}}
    params = replace(PARAMS, dataset=dataset, expected_ids=(content_hash(expected),))
    enqueue(h, params=params)
    def fetch(request, charge):
        assert request == expected
        charge()
        return 200, canonical_json({"status": "OK", "results": [{"ticker": "SYNTH", date_field: "2026-08-05"}]}).encode()
    _, record = run(h, claim(h), fetch, params)
    row = h.conn.execute("SELECT request_json,response_kind FROM data_raw_receipts").fetchone()
    assert json.loads(row["request_json"]) == expected and row["response_kind"] == "complete"
    assert record["request_hash"] == content_hash(expected)


@pytest.mark.parametrize("interval", [-1, True, 1.5, "13", None])
def test_pacing_arguments_refuse_before_catalog_access(interval):
    for invoke in (lambda: provider_budget.before_request(None, None, SOURCE, clock=None,
                                                         minimum_interval_seconds=interval),
                   lambda: provider_budget.record_response(None, SOURCE, 200, clock=None,
                                                          minimum_interval_seconds=interval)):
        with pytest.raises(OpsError) as error:
            invoke()
        assert error.value.code == "INVALID_REQUEST"


def test_response_backoff_never_shortens_an_existing_longer_pacing_window(harness):
    h = harness
    assert provider_budget.record_response(h.conn, SOURCE, 200, clock=h.clock,
                                            minimum_interval_seconds=100) is None
    assert provider_budget.record_response(h.conn, SOURCE, 429, clock=h.clock) == "RATE_LIMITED"
    h.clock.advance(100)
    assert h.conn.execute("SELECT next_eligible_at FROM provider_accounts").fetchone()[0] == format_timestamp(h.clock.now())


@pytest.mark.parametrize("missing", [False, True])
def test_unusable_existing_raw_bytes_refuse_without_http_fallback(harness, missing):
    h = harness
    record = cache_raw_receipt(h.conn, h.store, RawPayload(payload=BODY, response_kind="complete", response_meta={}),
                               source=SOURCE, endpoint=REQUEST["endpoint"], request=REQUEST,
                               received_at=format_timestamp(h.clock.now()))
    digest = record.raw_hash.removeprefix("sha256:")
    path = h.store.root / "objects" / digest[:2] / digest
    if missing:
        path.unlink()
    else:
        path.chmod(0o600)
        path.write_bytes(b"synthetic corrupt object")
    enqueue(h)
    def forbidden(*_):
        raise AssertionError("corrupt cache must never trigger HTTP fallback")
    with pytest.raises(OpsError) as error:
        run(h, claim(h), forbidden)
    assert error.value.code == "INTEGRITY_FAILED"
    assert error.value.problem.retryable is False
    assert h.conn.execute("SELECT used_calls FROM provider_reservations").fetchone()[0] == 0


def test_missing_catalog_is_typed_and_not_recreated(harness):
    h = harness
    missing = h.root / "absent.sqlite"
    with pytest.raises(OpsError) as error:
        acquisition.run(to_document(PARAMS), h.root, {"attempt_id": "synthetic", "fence": 1,
                        "catalog_path": str(missing), "objects_root": str(h.store.root)}, fetch=successful)
    assert error.value.code == "RESOURCE_UNAVAILABLE"
    assert not missing.exists()


def test_output_write_failure_leaves_only_the_fenced_reusable_raw_receipt(harness):
    h = harness
    enqueue(h)
    attempt = claim(h)
    output = h.root / "not-a-directory"
    output.write_bytes(b"unchanged")
    with pytest.raises(OpsError) as error:
        acquisition.run(to_document(PARAMS), output,
                        {"attempt_id": attempt.attempt_id, "fence": attempt.fence,
                         "catalog_path": str(h.root / "ops.sqlite"), "objects_root": str(h.store.root)},
                        fetch=successful, clock=h.clock)
    assert error.value.code == "RESOURCE_UNAVAILABLE"
    assert output.read_bytes() == b"unchanged"
    assert h.conn.execute("SELECT response_kind FROM data_raw_receipts").fetchone()[0] == "complete"
    assert acquisition.scope_plan(PARAMS, h.conn, h.store)["minimum_initial_calls"] == 0
