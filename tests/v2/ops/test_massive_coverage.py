"""Synthetic end-to-end interval evidence, real leases, raw objects and retries."""
import json
import socket
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from engine.v2.data.incremental import RawPayload, cache_raw_receipt
from engine.v2.foundation import (
    ArtifactStore,
    SystemClock,
    canonical_json,
    content_hash,
    format_timestamp,
    to_document,
)
from engine.v2.ops import massive_acquire as submit_cli
from engine.v2.ops import provider_budget, providers
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError
from engine.v2.ops.lifecycle import Outcome, commit_attempt, request_cancel, verify_fence
from engine.v2.ops.stores.massive import acquisition, coverage
from tests.ops_support import FakeClock
from tests.v2.ops import test_massive_acquisition as exact
from tests.v2.ops import test_massive_submission as submission_tests
from tests.v2.ops.test_massive_acquisition import (
    PARAMS,
    SOURCE,
    claim,
    finish,
    run,
)
from tests.v2.ops.test_massive_acquisition import (
    enqueue as enqueue_exact,
)
from tools import v2_massive_plan as plan_cli

harness = exact.harness
isolated = submission_tests.isolated


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic coverage test attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def request(start="2026-08-01", end="2026-08-31", dataset="daily", ticker="SYNTH"):
    result = {"schema_version": "massive_request.v1", "source": SOURCE,
              "transport": "rest", "dataset": dataset, "ticker": ticker,
              "start": start, "end": end, "adjustment_basis": "raw"}
    if dataset == "daily":
        return result | {"endpoint": f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
                         "params": {"adjusted": False, "sort": "asc", "limit": 50000}}
    field = "execution_date" if dataset == "splits" else "ex_dividend_date"
    return result | {"endpoint": f"/stocks/v1/{dataset}", "adjustment_basis": "corporate_action",
                     "params": {"ticker": ticker, field + ".gte": start, field + ".lte": end,
                                "sort": field + ".asc", "limit": 5000}}


def parameters(scope=None, *, generation="initial", cap=1):
    scope = scope or request()
    return replace(PARAMS, ticker=scope["ticker"], dataset=scope["dataset"],
                   start=scope["start"], end=scope["end"], expected_ids=(content_hash(scope),),
                   generation=generation, max_http_calls=cap)


def enqueue(h, key="history", params=None):
    return enqueue_exact(h, key, params or parameters(), kind="massive_history_acquisition",
                         checkpoint_contract_ref="massive_history_acquisition.v1")


def body(scope):
    if scope["dataset"] == "daily":
        instant = datetime.fromisoformat(scope["start"]).replace(tzinfo=ZoneInfo("America/New_York"))
        row = {"t": int(instant.timestamp() * 1000), "o": 10., "h": 12., "l": 9., "c": 11., "v": 100.}
        document = {"status": "OK", "ticker": scope["ticker"], "adjusted": False,
                    "resultsCount": 1, "results": [row]}
    else:
        field = "execution_date" if scope["dataset"] == "splits" else "ex_dividend_date"
        document = {"status": "OK", "results": [{"ticker": scope["ticker"], field: scope["start"]}]}
    return canonical_json(document).encode()


def fetching(calls, payload=None):
    def fetch(scope, charge):
        charge()
        calls.append(scope)
        return 200, body(scope) if payload is None else payload
    return fetch


def no_fetch(*_):
    pytest.fail("verified interval reuse or refusal must not issue HTTP")


def acquire(h, key, scope=None, *, generation="initial", fetch=None):
    params = parameters(scope, generation=generation)
    enqueue(h, key, params)
    attempt = claim(h)
    assert attempt is not None
    protocol, record = run(h, attempt, fetch or fetching([]), params)
    finish(h, attempt)
    h.clock.advance(13)
    return protocol, record


def occurrences(h):
    return h.conn.execute("SELECT * FROM ops_massive_occurrences ORDER BY occurrence_id").fetchall()


def retry(h, attempt, error):
    state = commit_attempt(h.conn, attempt.attempt_id, attempt.fence,
                           Outcome(succeeded=False, process_state="exited", exit_code=1,
                                   failure=error.problem), clock=h.clock)
    h.clock.advance(13)
    return state


def seed_overlap(h, key, scope, observed, *, payload=None):
    """Construct verified historical overlaps through the real atomic receipt API."""
    enqueue(h, key, parameters(scope))
    attempt = claim(h)
    documents = []
    def record(raw):
        document = {"schema_version": coverage.VERSION, "policy_version": PARAMS.policy_version,
                    "job_id": attempt.job_id, "generation": "initial", "request": scope,
                    "raw_receipt_id": raw.raw_receipt_id, "raw_hash": raw.raw_hash,
                    "first_received_at": raw.received_at, "origin": "http_response",
                    "observed_at": observed, "attempt_id": attempt.attempt_id, "fence": attempt.fence}
        coverage.record(h.conn, document)
        documents.append(document)
    cache_raw_receipt(h.conn, h.store,
                      RawPayload(payload=body(scope) if payload is None else payload,
                                 response_kind="complete", response_meta={}),
                      source=SOURCE, endpoint=scope["endpoint"], request=scope, received_at=observed,
                      fence_check=lambda: verify_fence(h.conn, attempt.attempt_id, attempt.fence, h.clock.now()),
                      record_receipt=record)
    finish(h, attempt)
    return content_hash(documents[0]), documents[0]


def test_superset_reuse_clips_bindings_without_rewriting_original_evidence(harness):
    h = harness
    calls = []
    _, first = acquire(h, "whole", fetch=fetching(calls))
    subset = request("2026-08-04", "2026-08-19")
    protocol, second = acquire(h, "subset", subset, fetch=no_fetch)
    original = first["coverage"]["bindings"][0]
    clipped = second["coverage"]["bindings"][0]
    assert clipped == original | {"start": subset["start"], "end": subset["end"]}
    assert clipped["evidence"]["request"] == request()
    assert second["coverage"]["gaps"] == []
    assert protocol["completed_ids"] == [content_hash(subset)] and not protocol["no_work"]
    assert protocol["outputs"][0]["schema"] == "massive_history_acquisition.v1"
    plan = second["coverage"]
    assert plan["plan_hash"] == content_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    assert plan == coverage.plan(h.conn, h.store, subset, "initial")
    assert calls == [request()] and len(occurrences(h)) == 1
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 1
    assert h.conn.execute("SELECT SUM(used_calls) FROM provider_reservations").fetchone()[0] == 1


@pytest.mark.parametrize("intervals,missing", [
    ([(3, 5), (5, 7), (10, 12), (15, 31)], [(1, 2), (8, 9), (13, 14)]),
    ([(1, 5), (6, 10), (11, 31)], []),
    ([(1, 5), (20, 31)], [(6, 19)]),
])
def test_union_plans_maximal_canonical_gaps(harness, intervals, missing):
    h = harness
    for index, (start, end) in enumerate(intervals):
        seed_overlap(h, str(index), request(f"2026-08-{start:02}", f"2026-08-{end:02}"),
                     "2026-09-10T12:00:00.000000Z")
    plan = coverage.plan(h.conn, h.store, request(), "initial")
    assert plan["gaps"] == [request(f"2026-08-{start:02}", f"2026-08-{end:02}") for start, end in missing]
    assert plan == coverage.plan(h.conn, h.store, request(), "initial")


def test_overlapping_binding_uses_newest_observation_then_stable_occurrence_id(harness):
    h = harness
    old, _ = seed_overlap(h, "old", request("2026-08-01", "2026-08-20"), "2026-09-10T12:00:00.000000Z")
    new_scope = request("2026-08-05", "2026-08-10")
    new, _ = seed_overlap(h, "new", new_scope, "2026-09-11T12:00:00.000000Z")
    tied, _ = seed_overlap(h, "tie", new_scope, "2026-09-11T12:00:00.000000Z")
    plan = coverage.plan(h.conn, h.store, request(), "initial")
    assert [(b["start"], b["end"], b["occurrence_id"]) for b in plan["bindings"]] == [
        ("2026-08-01", "2026-08-04", old), ("2026-08-05", "2026-08-10", max(new, tied)),
        ("2026-08-11", "2026-08-20", old)]
    assert plan["gaps"] == [request("2026-08-21", "2026-08-31")]


def test_refresh_generation_fetches_identical_bytes_but_retains_first_raw_timestamp(harness):
    h = harness
    calls = []
    _, first = acquire(h, "initial", fetch=fetching(calls))
    first_stamp = h.conn.execute("SELECT received_at FROM data_raw_receipts").fetchone()[0]
    _, refresh = acquire(h, "refresh", generation="review-2", fetch=fetching(calls))
    initial = first["coverage"]["bindings"][0]
    renewed = refresh["coverage"]["bindings"][0]
    assert calls == [request(), request()]
    assert initial["occurrence_id"] != renewed["occurrence_id"]
    assert initial["evidence"]["raw_receipt_id"] == renewed["evidence"]["raw_receipt_id"]
    assert initial["evidence"]["observed_at"] == first_stamp
    assert renewed["evidence"]["observed_at"] > first_stamp
    assert h.conn.execute("SELECT received_at FROM data_raw_receipts").fetchall()[0][0] == first_stamp
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 1
    _, reused = acquire(h, "same-generation", generation="review-2", fetch=no_fetch)
    assert reused["coverage"] == refresh["coverage"]
    assert coverage.plan(h.conn, h.store, request(), "initial")["bindings"] == [initial]
    assert len(occurrences(h)) == 2


def test_exact_kind_remains_exact_only_despite_history_superset(harness):
    h = harness
    acquire(h, "history")
    subset = request("2026-08-05", "2026-08-10")
    params = parameters(subset, generation=None)
    enqueue_exact(h, "exact-subset", params)
    calls = []
    protocol, result = run(h, claim(h), fetching(calls), params)
    assert calls == [subset] and result["cache_hit"] is False
    assert protocol["outputs"][0]["schema"] == "massive_acquisition.v1"
    assert len(occurrences(h)) == 1


@pytest.mark.parametrize("dataset", ["splits", "dividends"])
def test_corporate_history_reuses_its_own_family_only(harness, dataset):
    h = harness
    scope = request(dataset=dataset)
    _, first = acquire(h, "corporate", scope)
    subset = request("2026-08-10", "2026-08-15", dataset=dataset)
    _, second = acquire(h, "corporate-subset", subset, fetch=no_fetch)
    assert second["coverage"]["bindings"][0]["evidence"] == first["coverage"]["bindings"][0]["evidence"]
    for unrelated in (request(), request(dataset=dataset, ticker="OTHER"),
                      request(dataset="splits" if dataset == "dividends" else "dividends")):
        assert coverage.plan(h.conn, h.store, unrelated, "initial")["gaps"] == [unrelated]


@pytest.mark.parametrize("dataset", ["daily", "splits", "dividends"])
def test_initial_generation_promotes_legacy_receipt_at_original_observation(harness, dataset):
    h = harness
    scope = request(dataset=dataset)
    params = parameters(scope, generation=None)
    enqueue_exact(h, params=params)
    attempt = claim(h)
    _, legacy = run(h, attempt, fetching([]), params)
    finish(h, attempt)
    h.clock.advance(13)
    _, result = acquire(h, "promote", scope, fetch=no_fetch)
    evidence = result["coverage"]["bindings"][0]["evidence"]
    assert evidence["raw_receipt_id"] == legacy["raw_receipt_id"]
    assert evidence["observed_at"] == legacy["first_received_at"]
    assert evidence["observed_at"] < format_timestamp(h.clock.now())
    assert h.conn.execute("SELECT SUM(used_calls) FROM provider_reservations").fetchone()[0] == 1


def assert_terminal_evidence_failure(h, attempt, params):
    with pytest.raises(OpsError) as error:
        run(h, attempt, no_fetch, params)
    assert error.value.code == "INTEGRITY_FAILED" and not error.value.problem.retryable
    assert retry(h, attempt, error.value) == "failed"
    assert claim(h) is None
    assert h.conn.execute("SELECT count(*) FROM attempts WHERE job_id=?",
                          (attempt.job_id,)).fetchone()[0] == 1
    assert h.conn.execute("SELECT used_calls FROM provider_reservations WHERE attempt_id=?",
                          (attempt.attempt_id,)).fetchone()[0] == 0
    assert not (h.root / attempt.attempt_id / "massive_acquisition.json").exists()


@pytest.mark.parametrize("dataset", ["daily", "splits", "dividends"])
@pytest.mark.parametrize("observed", ["2026-09-12T00:00:00+00:00", "2026-09-12T03:59:59+00:00"])
def test_preclose_legacy_receipt_is_terminal_when_initial_generation_adopts_it(harness, dataset, observed):
    h = harness
    h.clock.value = datetime.fromisoformat(observed)
    scope = request("2026-09-10", "2026-09-11", dataset=dataset)
    exact = parameters(scope, generation=None)
    enqueue_exact(h, params=exact)
    attempt = claim(h)
    _, legacy = run(h, attempt, fetching([]), exact)
    finish(h, attempt)
    h.clock.advance(86400)
    params = parameters(scope)
    enqueue(h, params=params)
    assert_terminal_evidence_failure(h, claim(h), params)
    assert occurrences(h) == []
    assert h.conn.execute("SELECT received_at FROM data_raw_receipts").fetchone()[0] == legacy["first_received_at"]
    assert h.conn.execute("SELECT SUM(used_calls) FROM provider_reservations").fetchone()[0] == 1


@pytest.mark.parametrize("dataset", ["daily", "splits", "dividends"])
@pytest.mark.parametrize("observed,allowed", [
    ("2026-09-12T00:00:00.000000Z", False),
    ("2026-09-12T03:59:59.000000Z", False),
    ("2026-09-12T04:00:00.000000Z", True),
    ("2026-09-13T12:00:00.000000Z", True),
])
def test_occurrence_reuse_checks_original_scope_at_its_observation(harness, dataset, observed, allowed):
    h = harness
    scope = request("2026-09-10", "2026-09-11", dataset=dataset)
    identity, evidence = seed_overlap(h, "original", scope, observed)
    before = tuple(occurrences(h)[0])
    h.clock.advance(2 * 86400)
    # This subset was already closed at every observation. Only validation of
    # the original evidence scope can reject its prematurely observed superset.
    subset = request("2026-09-10", "2026-09-10", dataset=dataset)
    if allowed:
        _, result = acquire(h, "reuse", subset, fetch=no_fetch)
        assert result["coverage"]["gaps"] == []
        assert result["coverage"]["bindings"] == [
            {"start": subset["start"], "end": subset["end"],
             "occurrence_id": identity, "evidence": evidence}]
    else:
        with pytest.raises(OpsError) as error:
            coverage.plan(h.conn, h.store, subset, "initial")
        assert error.value.code == "INTEGRITY_FAILED" and not error.value.problem.retryable
        params = parameters(subset)
        enqueue(h, "reuse", params)
        assert_terminal_evidence_failure(h, claim(h), params)
    assert [tuple(row) for row in occurrences(h)] == [before]
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 1
    assert h.conn.execute("SELECT SUM(used_calls) FROM provider_reservations").fetchone()[0] == 0


@pytest.mark.parametrize("dataset", ["daily", "splits", "dividends"])
@pytest.mark.parametrize("stored_as", ["occurrence", "legacy_exact"])
def test_stored_out_of_scope_rows_are_terminal_integrity_failures(harness, dataset, stored_as):
    h = harness
    scope = request(dataset=dataset)
    # A complete, nonempty envelope accepted by the old exact validator, with
    # one row outside the requested dates. Stronger validation must not retry it.
    payload = body(request("2026-07-31", "2026-08-31", dataset=dataset))
    if stored_as == "occurrence":
        seed_overlap(h, "original", scope, "2026-09-12T00:00:00.000000Z", payload=payload)
    else:
        params = parameters(scope, generation=None)
        enqueue_exact(h, params=params)
        attempt = claim(h)
        run(h, attempt, fetching([], payload), params)
        finish(h, attempt)
    h.clock.advance(13)
    before = [tuple(row) for row in occurrences(h)]
    params = parameters(scope)
    enqueue(h, "reuse", params)
    assert_terminal_evidence_failure(h, claim(h), params)
    assert [tuple(row) for row in occurrences(h)] == before
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 1


def test_crash_after_atomic_commit_before_output_recovers_without_duplicate_fetch(harness):
    h = harness
    params = parameters(generation="refresh")
    enqueue(h, params=params)
    first = claim(h)
    output = h.root / "not-a-directory"
    output.write_bytes(b"keep")
    calls = []
    with pytest.raises(OpsError) as error:
        acquisition.run(to_document(params), output,
                        {"attempt_id": first.attempt_id, "fence": first.fence,
                         "catalog_path": str(h.root / "ops.sqlite"), "objects_root": str(h.store.root)},
                        fetch=fetching(calls), clock=h.clock)
    assert error.value.code == "RESOURCE_UNAVAILABLE" and output.read_bytes() == b"keep"
    committed = tuple(occurrences(h)[0])
    assert retry(h, first, error.value) == "retry_wait"
    second = claim(h)
    assert second.job_id == first.job_id and second.attempt_id != first.attempt_id
    _, result = run(h, second, no_fetch, params)
    assert result["coverage"]["bindings"][0]["evidence"]["attempt_id"] == first.attempt_id
    assert tuple(occurrences(h)[0]) == committed and len(occurrences(h)) == 1
    assert calls == [request()]
    assert h.conn.execute("SELECT SUM(used_calls) FROM provider_reservations").fetchone()[0] == 1


@pytest.mark.parametrize("cap", [1, 2])
def test_multiple_gaps_retry_with_cumulative_job_call_cap(harness, cap):
    h = harness
    acquire(h, "middle", request("2026-08-11", "2026-08-20"))
    params = parameters(cap=cap)
    submitted = enqueue(h, "union", params)
    assert h.conn.execute("SELECT max_attempts FROM jobs WHERE job_id=?", (submitted.job_id,)).fetchone()[0] == 101
    calls = []
    first = claim(h)
    with pytest.raises(OpsError) as error:
        run(h, first, fetching(calls), params)
    assert error.value.code == "SOURCE_NOT_FINAL"
    assert retry(h, first, error.value) == "retry_wait"
    second = claim(h)
    if cap == 1:
        with pytest.raises(OpsError) as error:
            run(h, second, fetching(calls), params)
        assert error.value.code == "RESOURCE_LIMIT_EXCEEDED"
        assert retry(h, second, error.value) == "failed"
        assert coverage.plan(h.conn, h.store, request(), "initial")["gaps"] == [request("2026-08-21", "2026-08-31")]
    else:
        _, result = run(h, second, fetching(calls), params)
        assert result["coverage"]["gaps"] == []
        finish(h, second)
    assert calls == [request("2026-08-01", "2026-08-10"), request("2026-08-21", "2026-08-31")][:cap]
    assert h.conn.execute("SELECT SUM(r.used_calls) FROM provider_reservations r JOIN attempts a USING(attempt_id) "
                          "WHERE a.job_id=?", (submitted.job_id,)).fetchone()[0] == cap


@pytest.mark.parametrize("failure", ["cancel", "expired"])
def test_failed_fence_rolls_back_raw_and_occurrence_together(harness, monkeypatch, failure):
    h = harness
    enqueue(h)
    attempt = claim(h)
    original = type(h.store).publish_bytes
    def publish(store, *args, **kwargs):
        ref = original(store, *args, **kwargs)
        if failure == "cancel":
            request_cancel(h.conn, attempt.job_id, expected_attempt_id=attempt.attempt_id, clock=h.clock)
        else:
            h.clock.advance(3600)
        return ref
    monkeypatch.setattr(type(h.store), "publish_bytes", publish)
    with pytest.raises(OpsError) as error:
        run(h, attempt, fetching([]), parameters())
    assert error.value.code == ("CANCELLED" if failure == "cancel" else "LEASE_LOST")
    assert occurrences(h) == []
    assert h.conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 0
    assert not (h.root / attempt.attempt_id / "massive_acquisition.json").exists()
    assert h.conn.execute("SELECT used_calls FROM provider_reservations").fetchone()[0] == 1


@pytest.mark.parametrize("existing_raw", [False, True])
def test_occurrence_insert_failure_rolls_back_new_raw_but_preserves_existing_raw(harness, existing_raw):
    h = harness
    if existing_raw:
        enqueue_exact(h)
        attempt = claim(h)
        run(h, attempt, fetching([]))
        finish(h, attempt)
        h.clock.advance(13)
    before = [tuple(row) for row in h.conn.execute("SELECT * FROM data_raw_receipts")]
    # Fault injection at SQLite's write boundary, without replacing either
    # cache_raw_receipt or the production occurrence callback under test.
    with transaction(h.conn):
        h.conn.execute("CREATE TRIGGER reject_occurrence BEFORE INSERT ON ops_massive_occurrences "
                       "BEGIN SELECT RAISE(ABORT, 'synthetic occurrence write failure'); END")
    params = parameters()
    enqueue(h, params=params)
    attempt = claim(h)
    with pytest.raises(OpsError) as error:
        run(h, attempt, no_fetch if existing_raw else fetching([]), params)
    assert error.value.code == "RESOURCE_UNAVAILABLE"
    assert occurrences(h) == []
    assert [tuple(row) for row in h.conn.execute("SELECT * FROM data_raw_receipts")] == before
    assert not (h.root / attempt.attempt_id / "massive_acquisition.json").exists()


@pytest.mark.parametrize("missing", [False, True])
def test_corrupt_pinned_object_refuses_without_http_fallback(harness, missing):
    h = harness
    _, result = acquire(h, "original")
    evidence = result["coverage"]["bindings"][0]["evidence"]
    digest = evidence["raw_hash"].removeprefix("sha256:")
    path = h.store.root / "objects" / digest[:2] / digest
    if missing:
        path.unlink()
    else:
        path.chmod(0o600)
        path.write_bytes(b"synthetic corruption")
    params = parameters(request("2026-08-05", "2026-08-08"))
    enqueue(h, "overlap", params)
    with pytest.raises(OpsError) as error:
        run(h, claim(h), no_fetch, params)
    assert error.value.code == "INTEGRITY_FAILED" and not error.value.problem.retryable
    assert len(occurrences(h)) == 1
    assert h.conn.execute("SELECT SUM(used_calls) FROM provider_reservations").fetchone()[0] == 1


@pytest.mark.parametrize("dataset", ["daily", "splits", "dividends"])
@pytest.mark.parametrize("invalid", ["before", "after", "ticker", "empty", "partial"])
def test_out_of_scope_or_nonterminal_corpus_never_becomes_positive(harness, dataset, invalid):
    h = harness
    scope = request(dataset=dataset)
    params = parameters(scope)
    document = json.loads(body(scope))
    row = document["results"][0]
    if invalid in ("before", "after"):
        day = "2026-07-31" if invalid == "before" else "2026-09-01"
        if dataset == "daily":
            row["t"] = int(datetime.fromisoformat(day).replace(tzinfo=ZoneInfo("America/New_York")).timestamp() * 1000)
        else:
            row["execution_date" if dataset == "splits" else "ex_dividend_date"] = day
    elif invalid == "ticker":
        (document if dataset == "daily" else row)["ticker"] = "OTHER"
    elif invalid == "empty":
        document.update(results=[], resultsCount=0)
    else:
        document["next_url"] = "https://api.massive.com" + scope["endpoint"] + "?cursor=next"
    enqueue(h, params=params)
    with pytest.raises(OpsError) as error:
        run(h, claim(h), fetching([], canonical_json(document).encode()), params)
    assert error.value.code == {"empty": "SOURCE_EMPTY", "partial": "SOURCE_NOT_FINAL"}.get(invalid, "SOURCE_INVALID")
    assert occurrences(h) == []
    raw = h.conn.execute("SELECT response_kind FROM data_raw_receipts").fetchall()
    if invalid == "partial":
        assert [row[0] for row in raw] == ["partial"]
        assert h.conn.execute("SELECT count(*) FROM ops_massive_pages").fetchone()[0] == 1
        assert coverage.plan(h.conn, h.store, scope, "initial")["gaps"] == [scope]
    else:
        assert raw == []


@pytest.mark.parametrize("instant,allowed", [("2026-09-12T03:59:59+00:00", False),
                                            ("2026-09-12T04:00:00+00:00", True)])
@pytest.mark.parametrize("dataset", ["daily", "splits", "dividends"])
def test_history_eligibility_uses_current_eastern_date(harness, instant, allowed, dataset):
    h = harness
    h.clock.value = datetime.fromisoformat(instant)
    scope = request("2026-09-10", "2026-09-11", dataset=dataset)
    params = parameters(scope)
    enqueue(h, params=params)
    attempt = claim(h)
    if allowed:
        _, result = run(h, attempt, fetching([]), params)
        assert result["coverage"]["gaps"] == []
    else:
        with pytest.raises(OpsError) as error:
            run(h, attempt, no_fetch, params)
        assert error.value.code == "SOURCE_NOT_FINAL"
        assert occurrences(h) == []
        assert h.conn.execute("SELECT used_calls FROM provider_reservations").fetchone()[0] == 0


@pytest.mark.parametrize("mutation", ["UPDATE ops_massive_occurrences SET generation='changed'",
                                     "DELETE FROM ops_massive_occurrences",
                                     "INSERT OR REPLACE INTO ops_massive_occurrences SELECT * FROM ops_massive_occurrences"])
def test_occurrences_are_immutable(harness, mutation):
    h = harness
    acquire(h, "original")
    before = tuple(occurrences(h)[0])
    with pytest.raises(sqlite3.IntegrityError):
        with transaction(h.conn):
            h.conn.execute(mutation)
    assert tuple(occurrences(h)[0]) == before


@pytest.mark.parametrize("change", [{"generation": ""}, {"generation": "with space"},
                                    {"generation": "x" * 65}, {"max_http_calls": 0},
                                    {"max_http_calls": 101}, {"max_http_calls": True},
                                    {"max_http_calls": "2"}])
def test_invalid_generation_or_call_cap_is_refused_before_submission(harness, change):
    with pytest.raises(OpsError) as error:
        enqueue(harness, params=replace(parameters(), **change))
    assert error.value.code == "INVALID_REQUEST"
    assert harness.conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0


@pytest.mark.parametrize("cap", [1, 7, 100])
def test_history_cli_submits_registered_kind_and_preserves_idempotency(tmp_path, isolated, capsys, cap):
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=FakeClock())
    try:
        args = submission_tests.arguments(tmp_path) + ["--generation", "initial", "--max-http-calls", str(cap)]
        assert submit_cli.main(args) == 0
        first = json.loads(capsys.readouterr().out)
        assert submit_cli.main(args) == 0
        second = json.loads(capsys.readouterr().out)
        assert first["job_id"] == second["job_id"] and first["state"] == "queued"
        assert first["kind"] == "massive_history_acquisition"
        row = conn.execute("SELECT * FROM jobs").fetchone()
        spec = json.loads(row["spec_json"])
        assert spec["checkpoint_contract_ref"] == "massive_history_acquisition.v1"
        assert spec["parameters"]["generation"] == "initial"
        assert spec["parameters"]["max_http_calls"] == cap
        assert row["max_attempts"] == 101
        assert json.loads(row["retry_json"])["backoff_seconds"] == [13]
        assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM provider_reservations").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM data_raw_receipts").fetchone()[0] == 0
        assert isolated not in str(tuple(conn.iterdump())) + canonical_json(first)
    finally:
        conn.close()


@pytest.mark.parametrize("dataset", ["daily", "splits", "dividends"])
@pytest.mark.parametrize("instant,allowed", [
    ("2026-09-12T00:00:00+00:00", False),
    ("2026-09-12T03:59:59+00:00", False),
    ("2026-09-12T04:00:00+00:00", True),
    ("2026-09-13T12:00:00+00:00", True),
])
def test_history_cli_closed_window_precedes_fingerprints_and_submission(
        tmp_path, isolated, monkeypatch, capsys, dataset, instant, allowed):
    clock = FakeClock()
    clock.value = datetime.fromisoformat(instant)
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    fingerprints = []
    def source(_):
        fingerprints.append("source")
        return {"synthetic_source": "v1"}
    def environment(threads):
        fingerprints.append("environment")
        return {"synthetic_threads": threads}
    monkeypatch.setattr(submit_cli, "worker_source_manifest", source)
    monkeypatch.setattr(submit_cli, "environment_identity", environment)
    monkeypatch.setattr(providers, "provider_credentials", no_fetch)
    monkeypatch.setattr(provider_budget, "before_request", no_fetch)
    try:
        args = submission_tests.arguments(tmp_path) + ["--generation", "initial"]
        for flag, value in (("--dataset", dataset), ("--start", "2026-09-10"), ("--end", "2026-09-11")):
            args[args.index(flag) + 1] = value
        assert submit_cli.main(args, clock=clock) == (0 if allowed else 2)
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert captured.err == "" and isolated not in captured.out
        if allowed:
            assert result["kind"] == "massive_history_acquisition" and result["state"] == "queued"
            assert fingerprints == ["source", "environment"]
        else:
            assert result == {"refused": "SOURCE_NOT_FINAL"}
            assert fingerprints == []
        assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == int(allowed)
        for table in ("attempts", "provider_accounts", "provider_reservations", "data_raw_receipts",
                      "ops_massive_occurrences", "ops_massive_pages"):
            assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    finally:
        conn.close()


@pytest.mark.parametrize("flags", [["--generation", ""], ["--generation", "with space"],
    ["--generation", "x" * 65], ["--generation", "initial", "--max-http-calls", "True"],
    ["--generation", "initial", "--max-http-calls", "0"],
    ["--generation", "initial", "--max-http-calls", "-1"],
    ["--generation", "initial", "--max-http-calls", "101"], ["--max-http-calls", "2"]])
def test_history_cli_invalid_scope_refuses_before_catalog_and_fingerprints(tmp_path, isolated, monkeypatch, capsys, flags):
    def forbidden(*_args, **_kwargs):
        pytest.fail("invalid history CLI scope reached catalog or fingerprints")
    for name in ("connect", "worker_source_manifest", "environment_identity"):
        monkeypatch.setattr(submit_cli, name, forbidden)
    assert submit_cli.main(submission_tests.arguments(tmp_path) + flags) == 2
    captured = capsys.readouterr()
    assert captured.err == "" and json.loads(captured.out) == {"refused": "INVALID_REQUEST"}
    assert isolated not in captured.out
    assert not (tmp_path / "catalog.sqlite").exists()


@pytest.mark.parametrize("kind,generation", [("massive_acquisition", "initial"),
                                            ("massive_history_acquisition", None)])
def test_generation_and_job_kind_mismatch_is_refused_before_admission(harness, kind, generation):
    checkpoint = "massive_acquisition.v1" if kind == "massive_acquisition" else "massive_history_acquisition.v1"
    with pytest.raises(OpsError) as error:
        enqueue_exact(harness, params=parameters(generation=generation), kind=kind,
                      checkpoint_contract_ref=checkpoint)
    assert error.value.code == "INVALID_REQUEST"
    assert harness.conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
    assert harness.conn.execute("SELECT count(*) FROM provider_reservations").fetchone()[0] == 0


def test_interval_planning_cli_is_readonly_clipped_gap_aware_and_legacy_compatible(harness, monkeypatch, capsys):
    h = harness
    acquire(h, "left", request("2026-08-01", "2026-08-10"))
    acquire(h, "right", request("2026-08-20", "2026-08-31"))
    before_catalog = tuple(h.conn.iterdump())
    before_objects = {p.relative_to(h.store.root): (p.read_bytes(), p.stat().st_mtime_ns)
                      for p in h.store.root.rglob("*") if p.is_file()}
    marker = "synthetic-credential-must-not-appear"
    monkeypatch.setattr(providers, "os", SimpleNamespace(environ={"MASSIVE_API_KEY": marker}))
    original, statements = sqlite3.connect, []
    def readonly(database, **kwargs):
        assert database.endswith("?mode=ro") and kwargs["uri"] is True
        result = original(database, **kwargs)
        result.set_trace_callback(statements.append)
        return result
    monkeypatch.setattr(plan_cli, "sqlite3", SimpleNamespace(connect=readonly, Row=sqlite3.Row, Error=sqlite3.Error))
    args = ["--catalog", str(h.root / "ops.sqlite"), "--store-root", str(h.store.root),
            "--ticker", "SYNTH", "--start", "2026-08-05", "--end", "2026-08-25",
            "--as-of", "2026-09-12", "--adjustment", "raw", "--max-calls-per-request", "7"]
    assert plan_cli._main(args + ["--generation", "initial"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert captured.err == "" and result["schema_version"] == "massive_interval_plan.v1"
    assert result["plan_only"] is True
    daily = next(p for p in result["plans"] if p["request"]["dataset"] == "daily")
    assert [(b["start"], b["end"]) for b in daily["bindings"]] == [
        ("2026-08-05", "2026-08-10"), ("2026-08-20", "2026-08-25")]
    assert daily["gaps"] == [request("2026-08-11", "2026-08-19")]
    assert all(p["generation"] == "initial" for p in result["plans"])
    assert plan_cli._main(args) == 0
    legacy = capsys.readouterr()
    assert legacy.err == "" and json.loads(legacy.out)["schema_version"] == "massive_cache_plan.v1"
    assert "BEGIN" in statements and "PRAGMA query_only=ON" in statements
    assert not any(s.startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "COMMIT")) for s in statements)
    assert tuple(h.conn.iterdump()) == before_catalog
    assert {p.relative_to(h.store.root): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in h.store.root.rglob("*") if p.is_file()} == before_objects
    assert marker not in captured.out + legacy.out + str(before_catalog)
    assert "MASSIVE_API_KEY" not in captured.out + legacy.out


def test_real_service_launches_each_cached_history_job_and_publishes_own_output(tmp_path, monkeypatch, capsys):
    from engine.v2.ops import executor
    from engine.v2.ops.checkpoints import artifact
    from engine.v2.ops.fingerprints import worker_source_manifest
    from engine.v2.ops.stages import registry
    from engine.v2.ops.supervisor import Service
    from tests.ops_support import TEST_POLICY, run_until

    clock = SystemClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    store = ArtifactStore(tmp_path)
    provider_budget.configure_account(conn, SOURCE, "synthetic", 10, 0)
    scope = request()
    raw = cache_raw_receipt(conn, store, RawPayload(payload=body(scope), response_kind="complete", response_meta={}),
                            source=SOURCE, endpoint=scope["endpoint"], request=scope,
                            received_at="2026-09-12T00:00:00.000000Z")
    source = Path(__file__).resolve().parents[3]
    manifest = worker_source_manifest(source)
    assert {"engine/v2/ops/stores/massive/coverage.py", "engine/v2/ops/stores/massive/pages.py"} <= set(manifest)
    marker = "synthetic-massive-key-not-a-credential"
    monkeypatch.setattr(providers, "os", SimpleNamespace(environ={"MASSIVE_API_KEY": marker,
                                                                "ORATS_API_KEY": "unrelated-synthetic"}))
    original, launched = executor.subprocess.Popen, []
    def launch(*args, **kwargs):
        environment = dict(kwargs["env"])
        assert "ORATS_API_KEY" not in environment
        launched.append(environment.pop("MASSIVE_API_KEY", None))
        # Even a regression in cache selection cannot perform live HTTP: the
        # real child receives no key, and its legacy root is synthetic too.
        return original(*args, **(kwargs | {"env": environment}))
    monkeypatch.setattr(executor.subprocess, "Popen", launch)
    service = Service(conn, tmp_path, registry(), TEST_POLICY, clock=clock, code_source=source,
                      store_root=tmp_path / "synthetic-legacy")
    service.start()
    try:
        records = []
        for key in ("first-history", "second-history"):
            assert marker not in capsys.readouterr().out
            args = submission_tests.arguments(tmp_path) + ["--generation", "initial"]
            args[args.index("--idempotency-key") + 1] = key
            assert submit_cli.main(args, clock=clock) == 0
            submitted = json.loads(capsys.readouterr().out)
            assert run_until(service, conn, submitted["job_id"], timeout=30) == "succeeded"
            output = conn.execute("SELECT o.artifact_id FROM attempt_outputs o JOIN attempts a USING(attempt_id) "
                                  "WHERE a.job_id=? AND o.name='massive_acquisition'", (submitted["job_id"],)).fetchone()
            ref = artifact(conn, store, output[0])
            records.append(json.loads(store.read_verified(ref)))
        assert launched == [marker, marker]
        assert records[0]["schema_version"] == records[1]["schema_version"] == "massive_history_acquisition.v1"
        assert records[0]["attempt_id"] != records[1]["attempt_id"]
        assert records[0]["coverage"] == records[1]["coverage"]
        assert records[0]["coverage"]["bindings"][0]["evidence"]["raw_receipt_id"] == raw.raw_receipt_id
        assert conn.execute("SELECT count(*) FROM ops_massive_occurrences").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM checkpoints").fetchone()[0] == 0
        assert conn.execute("SELECT SUM(used_calls) FROM provider_reservations").fetchone()[0] == 0
        assert marker not in str(tuple(conn.iterdump())) + canonical_json(records)
    finally:
        service.close()
        conn.close()


def test_candidate_limit_is_inclusive_and_overflow_precedes_object_reads(harness, monkeypatch):
    h = harness
    assert coverage.MAX_OCCURRENCES == 100
    for index in range(100):
        seed_overlap(h, f"candidate-{index}", request(), "2026-09-10T12:00:00.000000Z")
    assert coverage.plan(h.conn, h.store, request(), "initial")["gaps"] == []
    seed_overlap(h, "candidate-overflow", request(), "2026-09-10T12:00:00.000000Z")
    monkeypatch.setattr(h.store, "read_verified", no_fetch)
    with pytest.raises(OpsError) as refused:
        coverage.plan(h.conn, h.store, request(), "initial")
    assert refused.value.code == "RESOURCE_LIMIT_EXCEEDED"


def test_aggregate_verification_bytes_are_inclusive_and_stop_before_excess_read(harness, monkeypatch):
    h = harness
    assert coverage.MAX_VERIFIED_BYTES == 64 << 20
    scopes = [request(f"2026-08-{start:02}", f"2026-08-{end:02}")
              for start, end in ((1, 5), (8, 12), (20, 25))]
    for index, scope in enumerate(scopes[:2]):
        seed_overlap(h, f"bytes-{index}", scope, "2026-09-10T12:00:00.000000Z")
    assert len({len(body(scope)) for scope in scopes}) == 1
    monkeypatch.setattr(coverage, "MAX_VERIFIED_BYTES", sum(len(body(scope)) for scope in scopes[:2]))
    assert len(coverage.plan(h.conn, h.store, request(), "initial")["bindings"]) == 2
    seed_overlap(h, "bytes-overflow", scopes[2], "2026-09-10T12:00:00.000000Z")
    reads, original = [], h.store.read_verified
    def counted(ref):
        reads.append(ref)
        return original(ref)
    monkeypatch.setattr(h.store, "read_verified", counted)
    with pytest.raises(OpsError) as refused:
        coverage.plan(h.conn, h.store, request(), "initial")
    assert refused.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert len(reads) == 2
