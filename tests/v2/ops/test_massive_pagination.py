"""Real fenced pagination/recovery with synthetic HTTP, catalogs and raw objects."""
import json
import socket
import sqlite3
from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import pytest

from engine.v2.foundation import canonical_json, content_hash, to_document
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.lifecycle import Outcome, commit_attempt, request_cancel
from engine.v2.ops.providers import massive as http
from engine.v2.ops.stores.massive import acquisition, coverage, journal
from tests.v2.ops import test_massive_coverage as support

harness = support.harness


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic pagination attempted a real network connection")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(http.requests, "Session", forbidden)
    monkeypatch.setattr(http, "os", SimpleNamespace(environ={}))


def continuation(scope, index, *, day=None):
    endpoint = scope["endpoint"]
    if scope["dataset"] == "daily" and day is not None:
        instant = datetime.fromisoformat(day).replace(tzinfo=ZoneInfo("America/New_York"))
        endpoint = (f"/v2/aggs/ticker/{scope['ticker']}/range/1/day/"
                    f"{int(instant.timestamp() * 1000)}/{scope['end']}")
    return scope | {"endpoint": endpoint, "params": scope["params"] | {"cursor": f"page-{index}"}}


def next_url(scope):
    return "https://api.massive.com" + scope["endpoint"] + "?" + urlencode(scope["params"] | {
        key: str(value).lower() for key, value in scope["params"].items() if type(value) is bool})


def payload(scope, *, day=None, following=None):
    document = json.loads(support.body(scope | {"start": day or scope["start"]}))
    if following is not None:
        document["next_url"] = next_url(following)
    return canonical_json(document).encode()


def fetcher(calls, expected, body, status=200):
    def fetch(request, charge):
        assert request == expected
        charge()
        calls.append(request)
        return status, body
    return fetch


def pages(h):
    return h.conn.execute("SELECT * FROM ops_massive_pages ORDER BY page_index").fetchall()


def raw_rows(h):
    return h.conn.execute("SELECT * FROM data_raw_receipts ORDER BY raw_receipt_id").fetchall()


def spent(h, job_id):
    return h.conn.execute(
        "SELECT COALESCE(SUM(r.used_calls),0) FROM provider_reservations r "
        "JOIN attempts a USING(attempt_id) WHERE a.job_id=?", (job_id,)).fetchone()[0]


def refused(h, attempt, params, fetch, code):
    with pytest.raises(OpsError) as error:
        support.run(h, attempt, fetch, params)
    assert error.value.code == code
    assert not (h.root / attempt.attempt_id / "massive_acquisition.json").exists()
    return error.value


def prefix(h, *, params=None, count=1):
    params = params or support.parameters(cap=count + 1)
    scope = support.request(params.start, params.end, params.dataset, params.ticker)
    support.enqueue(h, params=params)
    current, calls, attempts = scope, [], []
    for index in range(count):
        attempt = support.claim(h)
        assert attempt is not None
        attempts.append(attempt)
        following = continuation(scope, index + 1, day=f"2026-08-{index + 2:02}")
        error = refused(h, attempt, params,
                        fetcher(calls, current, payload(scope, day=f"2026-08-{index + 1:02}",
                                                       following=following)), "SOURCE_NOT_FINAL")
        assert support.retry(h, attempt, error) == "retry_wait"
        current = following
    return params, attempts, current, calls


@pytest.mark.parametrize("dataset", ["daily", "splits", "dividends"])
@pytest.mark.parametrize("count", [2, 3])
def test_pages_commit_as_partial_prefix_then_receipt_pinned_complete_coverage(harness, dataset, count):
    h = harness
    scope = support.request(dataset=dataset)
    params = support.parameters(scope, cap=count)
    params, attempts, current, calls = prefix(h, params=params, count=count - 1)
    before = [tuple(row) for row in pages(h)]
    assert len(before) == count - 1
    assert {row["response_kind"] for row in raw_rows(h)} == {"partial"}
    assert support.occurrences(h) == []
    assert coverage.plan(h.conn, h.store, scope, "initial")["gaps"] == [scope]
    progress = coverage.progress(h.conn, h.store, scope, "initial", attempts[0].job_id)
    assert progress["next_request"] == current
    assert progress["page_ids"] == [row["page_id"] for row in pages(h)]
    last = support.claim(h)
    protocol, result = support.run(h, last, fetcher(calls, current, payload(
        scope, day=f"2026-08-{count:02}")), params)
    assert len(calls) == count
    assert [tuple(row) for row in pages(h)[:-1]] == before
    evidence = result["coverage"]["bindings"][0]["evidence"]
    assert evidence["request"] == scope
    assert evidence["pages"] == [row["page_id"] for row in pages(h)]
    assert evidence["attempt_id"] == last.attempt_id and evidence["fence"] == last.fence
    assert evidence["raw_receipt_id"] == pages(h)[-1]["raw_receipt_id"]
    assert len(support.occurrences(h)) == 1 and result["coverage"]["gaps"] == []
    assert protocol["completed_ids"] == [content_hash(scope)] and protocol["no_work"] is False
    assert [json.loads(row["document_json"])["page_index"] for row in pages(h)] == list(range(count))
    assert sorted(row["response_kind"] for row in raw_rows(h)) == ["complete"] + ["partial"] * (count - 1)
    assert spent(h, last.job_id) == count
    reservations = h.conn.execute("SELECT reserved_calls,used_calls FROM provider_reservations").fetchall()
    assert [tuple(row) for row in reservations] == [(1, 1)] * count
    support.finish(h, last)
    h.clock.advance(13)
    subset = support.request("2026-08-10", "2026-08-20", dataset=dataset)
    _, reused = support.acquire(h, "subset", subset, fetch=support.no_fetch)
    binding = reused["coverage"]["bindings"][0]
    assert (binding["start"], binding["end"]) == (subset["start"], subset["end"])
    assert binding["evidence"] == evidence
    assert len(raw_rows(h)) == count and len(support.occurrences(h)) == 1


@pytest.mark.parametrize("interleaved", [False, True])
def test_remaining_completion_slot_is_rechecked_for_multi_page_retry(harness, interleaved):
    h = harness
    for index in range(99):
        support.seed_overlap(h, f"candidate-{index}", support.request("2026-08-01", "2026-08-20"),
                             "2026-09-10T12:00:00.000000Z")
    scope, gap = support.request(), support.request("2026-08-21", "2026-08-31")
    params = support.parameters(cap=2)
    support.enqueue(h, "two-page-last-slot", params)
    first, calls = support.claim(h), []
    following = continuation(gap, 1, day="2026-08-22")
    error = refused(h, first, params, fetcher(calls, gap, payload(gap, following=following)), "SOURCE_NOT_FINAL")
    assert support.retry(h, first, error) == "retry_wait"
    planned = coverage.plan(h.conn, h.store, scope, "initial")
    assert planned["candidate_count"] == 99 and planned["gaps"] == [gap]
    assert len(pages(h)) == 1  # A partial page consumes no completion slot.
    if interleaved:
        other_scope = support.request("2026-08-21", "2026-08-21")
        other_params = support.parameters(other_scope)
        support.enqueue(h, "higher-priority-gap", other_params, priority=10)
        other = support.claim(h)
        assert other.job_id != first.job_id
        _, completed = support.run(h, other, support.fetching([]), other_params)
        assert completed["coverage"]["gaps"] == []
        support.finish(h, other)
        h.clock.advance(13)
    last = support.claim(h)
    assert last.job_id == first.job_id
    if interleaved:
        before = tuple(h.conn.iterdump())
        objects = {p.relative_to(h.store.root): p.read_bytes() for p in h.store.root.rglob("*") if p.is_file()}
        error = refused(h, last, params, support.no_fetch, "RESOURCE_LIMIT_EXCEEDED")
        assert tuple(h.conn.iterdump()) == before
        assert {p.relative_to(h.store.root): p.read_bytes() for p in h.store.root.rglob("*") if p.is_file()} == objects
        assert len(pages(h)) == 1 and len(support.occurrences(h)) == 100
        assert calls == [gap] and spent(h, last.job_id) == 1
        assert support.retry(h, last, error) == "failed" and support.claim(h) is None
        return
    _, result = support.run(h, last, fetcher(calls, following, payload(gap, day="2026-08-22")), params)
    assert result["coverage"]["candidate_count"] == 100 and result["coverage"]["gaps"] == []
    assert len(pages(h)) == 2 and len(support.occurrences(h)) == 100
    assert calls == [gap, following] and spent(h, last.job_id) == 2
    support.finish(h, last)


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_call_cap_spans_released_page_attempts_and_never_refunds_prefix(harness, cap):
    h = harness
    params = support.parameters(cap=cap)
    params, attempts, current, calls = prefix(h, params=params, count=cap)
    before = [tuple(row) for row in pages(h)]
    attempt = support.claim(h)
    error = refused(h, attempt, params, fetcher(calls, current, payload(support.request(), day="2026-08-10")),
                    "RESOURCE_LIMIT_EXCEEDED")
    assert support.retry(h, attempt, error) == "failed"
    assert calls == [support.request()] + [continuation(support.request(), i, day=f"2026-08-{i + 1:02}")
                                         for i in range(1, cap)]
    assert spent(h, attempts[0].job_id) == cap
    assert [tuple(row) for row in pages(h)] == before and support.occurrences(h) == []
    assert len(raw_rows(h)) == cap
    assert h.conn.execute("SELECT used_calls FROM provider_reservations WHERE attempt_id=?",
                          (attempt.attempt_id,)).fetchone()[0] == 0


@pytest.mark.parametrize("failure", ["503", "precommit"])
def test_failed_continuation_spends_call_but_preserves_only_committed_prefix(harness, failure):
    h = harness
    params, attempts, current, calls = prefix(h, params=support.parameters(cap=3))
    before = [tuple(row) for row in pages(h)], [tuple(row) for row in raw_rows(h)]
    if failure == "precommit":
        with transaction(h.conn):
            h.conn.execute("CREATE TRIGGER reject_page BEFORE INSERT ON ops_massive_pages "
                           "BEGIN SELECT * FROM synthetic_missing_page_table; END")
    attempt = support.claim(h)
    code = "TRANSIENT_SOURCE" if failure == "503" else "RESOURCE_UNAVAILABLE"
    error = refused(h, attempt, params, fetcher(calls, current, payload(support.request(), day="2026-08-02"),
                                               status=503 if failure == "503" else 200), code)
    assert ([tuple(row) for row in pages(h)], [tuple(row) for row in raw_rows(h)]) == before
    assert support.occurrences(h) == [] and spent(h, attempt.job_id) == 2
    assert support.retry(h, attempt, error) == "retry_wait"
    if failure == "precommit":
        with transaction(h.conn):
            h.conn.execute("DROP TRIGGER reject_page")
    final = support.claim(h)
    _, result = support.run(h, final, fetcher(calls, current, payload(support.request(), day="2026-08-02")), params)
    assert calls == [support.request(), current, current]
    assert spent(h, attempts[0].job_id) == 3 and len(pages(h)) == 2
    assert result["coverage"]["gaps"] == [] and len(support.occurrences(h)) == 1


def test_failure_before_first_page_commit_restarts_original_request_without_refunding(harness):
    h = harness
    params = support.parameters(cap=2)
    support.enqueue(h, params=params)
    first = support.claim(h)
    calls = []
    following = continuation(support.request(), 1, day="2026-08-02")
    with transaction(h.conn):
        h.conn.execute("CREATE TRIGGER reject_page BEFORE INSERT ON ops_massive_pages "
                       "BEGIN SELECT * FROM synthetic_missing_page_table; END")
    error = refused(h, first, params, fetcher(calls, support.request(), payload(
        support.request(), following=following)), "RESOURCE_UNAVAILABLE")
    assert pages(h) == raw_rows(h) == support.occurrences(h) == []
    assert spent(h, first.job_id) == 1
    assert support.retry(h, first, error) == "retry_wait"
    with transaction(h.conn):
        h.conn.execute("DROP TRIGGER reject_page")
    second = support.claim(h)
    _, result = support.run(h, second, fetcher(calls, support.request(), payload(support.request())), params)
    assert calls == [support.request(), support.request()] and spent(h, first.job_id) == 2
    assert result["coverage"]["gaps"] == [] and pages(h) == []


def test_terminal_commit_survives_output_failure_and_reuses_entire_chain_without_http(harness):
    h = harness
    params, attempts, current, calls = prefix(h)
    final = support.claim(h)
    output = h.root / "not-a-directory"
    output.write_bytes(b"keep")
    with pytest.raises(OpsError) as error:
        acquisition.run(to_document(params), output,
                        {"attempt_id": final.attempt_id, "fence": final.fence,
                         "catalog_path": str(h.root / "ops.sqlite"), "objects_root": str(h.store.root)},
                        fetch=fetcher(calls, current, payload(support.request(), day="2026-08-02")), clock=h.clock)
    assert error.value.code == "RESOURCE_UNAVAILABLE" and output.read_bytes() == b"keep"
    before = [tuple(row) for row in pages(h)], [tuple(row) for row in support.occurrences(h)]
    assert len(before[0]) == 2 and len(before[1]) == 1
    assert support.retry(h, final, error.value) == "retry_wait"
    recovered = support.claim(h)
    _, result = support.run(h, recovered, support.no_fetch, params)
    assert result["coverage"]["bindings"][0]["evidence"]["attempt_id"] == final.attempt_id
    assert ([tuple(row) for row in pages(h)], [tuple(row) for row in support.occurrences(h)]) == before
    assert spent(h, attempts[0].job_id) == 2 and len(calls) == 2


@pytest.mark.parametrize("failure", ["cancel", "expired", "stale"])
@pytest.mark.parametrize("terminal", [False, True])
def test_lost_fence_cannot_commit_partial_or_terminal_raw_page_occurrence(harness, monkeypatch, failure, terminal):
    h = harness
    if terminal:
        params, attempts, current, calls = prefix(h)
    else:
        params, calls, current = support.parameters(cap=2), [], support.request()
        support.enqueue(h, params=params)
    attempt = support.claim(h)
    before = [tuple(row) for row in pages(h)], [tuple(row) for row in raw_rows(h)]
    original = type(h.store).publish_bytes
    def publish(store, *args, **kwargs):
        result = original(store, *args, **kwargs)
        if failure == "cancel":
            request_cancel(h.conn, attempt.job_id, expected_attempt_id=attempt.attempt_id, clock=h.clock)
        elif failure == "expired":
            h.clock.advance(3600)
        else:
            with transaction(h.conn):
                h.conn.execute("UPDATE jobs SET fence=fence+1 WHERE job_id=?", (attempt.job_id,))
        return result
    monkeypatch.setattr(type(h.store), "publish_bytes", publish)
    following = None if terminal else continuation(support.request(), 1, day="2026-08-02")
    refused(h, attempt, params, fetcher(calls, current, payload(support.request(),
            day="2026-08-02" if terminal else None, following=following)),
            "CANCELLED" if failure == "cancel" else "LEASE_LOST")
    assert ([tuple(row) for row in pages(h)], [tuple(row) for row in raw_rows(h)]) == before
    assert support.occurrences(h) == [] and spent(h, attempt.job_id) == (2 if terminal else 1)


def test_completion_write_failure_rolls_back_terminal_page_and_raw_together(harness):
    h = harness
    params, _, current, calls = prefix(h, params=support.parameters(cap=3))
    before = [tuple(row) for row in pages(h)], [tuple(row) for row in raw_rows(h)]
    with transaction(h.conn):
        h.conn.execute("CREATE TRIGGER reject_completion BEFORE INSERT ON ops_massive_occurrences "
                       "BEGIN SELECT RAISE(ABORT, 'synthetic terminal commit failure'); END")
    attempt = support.claim(h)
    error = refused(h, attempt, params, fetcher(calls, current, payload(support.request(), day="2026-08-02")),
                    "RESOURCE_UNAVAILABLE")
    assert ([tuple(row) for row in pages(h)], [tuple(row) for row in raw_rows(h)]) == before
    assert support.occurrences(h) == [] and spent(h, attempt.job_id) == 2
    assert support.retry(h, attempt, error) == "retry_wait"
    with transaction(h.conn):
        h.conn.execute("DROP TRIGGER reject_completion")
    final = support.claim(h)
    support.run(h, final, fetcher(calls, current, payload(support.request(), day="2026-08-02")), params)
    assert len(pages(h)) == 2 and len(raw_rows(h)) == 2 and len(support.occurrences(h)) == 1
    assert calls == [support.request(), current, current]


@pytest.mark.parametrize("invalid", ["empty", "cycle", "credential", "foreign-origin"])
def test_rejected_final_or_continuation_response_never_persists_new_raw(harness, invalid):
    h = harness
    params, _, current, calls = prefix(h, count=2)
    before = [tuple(row) for row in pages(h)], [tuple(row) for row in raw_rows(h)]
    document = json.loads(payload(support.request(), day="2026-08-03"))
    if invalid == "empty":
        document.update(results=[], resultsCount=0)
    elif invalid == "cycle":
        # Same daily lower bound with a previously used cursor is forbidden.
        document["next_url"] = next_url(current)
    else:
        document["next_url"] = next_url(continuation(support.request(), 3, day="2026-08-04"))
        if invalid == "credential":
            document["next_url"] += "&apiKey=synthetic-secret"
        else:
            document["next_url"] = document["next_url"].replace("api.massive.com", "foreign.invalid")
    attempt = support.claim(h)
    error = refused(h, attempt, params, fetcher(calls, current, canonical_json(document).encode()),
                    "SOURCE_EMPTY" if invalid == "empty" else "SOURCE_INVALID")
    assert ([tuple(row) for row in pages(h)], [tuple(row) for row in raw_rows(h)]) == before
    assert support.occurrences(h) == [] and spent(h, attempt.job_id) == 3
    assert "synthetic-secret" not in str(error) + str(tuple(h.conn.iterdump()))
    assert all(b"synthetic-secret" not in file.read_bytes() for file in h.store.root.rglob("*") if file.is_file())


@pytest.mark.parametrize("fault", ["missing-first", "index", "previous", "hash", "schema", "next", "request",
                                   "raw-hash", "fence", "timestamp", "missing-object", "corrupt-object"])
def test_corrupt_committed_prefix_refuses_without_fetch_or_new_charge(harness, fault):
    h = harness
    params, attempts, current, calls = prefix(h, count=2)
    rows = pages(h)
    row = rows[-1]
    document = json.loads(row["document_json"])
    if fault in ("missing-object", "corrupt-object"):
        digest = document["raw_hash"].removeprefix("sha256:")
        path = h.store.root / "objects" / digest[:2] / digest
        if fault == "missing-object":
            path.unlink()
        else:
            path.chmod(0o600)
            path.write_bytes(b"synthetic corrupt page bytes")
    else:
        with transaction(h.conn):
            h.conn.execute("DROP TRIGGER massive_pages_no_update")
            h.conn.execute("DROP TRIGGER massive_pages_no_delete")
            if fault == "missing-first":
                h.conn.execute("DELETE FROM ops_massive_pages WHERE page_id=?", (rows[0]["page_id"],))
            else:
                if fault == "index":
                    document["page_index"] = 7
                elif fault == "previous":
                    document["previous_page_id"] = "sha256:" + "0" * 64
                elif fault == "schema":
                    document["schema_version"] = "unsupported.v9"
                elif fault == "next":
                    document["next_request"] = continuation(support.request(), 99, day="2026-08-03")
                elif fault == "request":
                    document["request"] = document["request"] | {"params": document["request"]["params"] | {"limit": 1}}
                elif fault == "raw-hash":
                    document["raw_hash"] = "sha256:" + "0" * 64
                elif fault == "fence":
                    document["fence"] += 1
                elif fault == "timestamp":
                    document["observed_at"] = "2026-09-12"
                identity = "sha256:" + "0" * 64 if fault == "hash" else content_hash(document)
                h.conn.execute("UPDATE ops_massive_pages SET page_id=?,page_index=?,document_json=? WHERE page_id=?",
                               (identity, document["page_index"], canonical_json(document), row["page_id"]))
    attempt = support.claim(h)
    error = refused(h, attempt, params, support.no_fetch, "INTEGRITY_FAILED")
    assert not error.problem.retryable
    assert spent(h, attempts[0].job_id) == 2 and len(calls) == 2
    assert support.occurrences(h) == []
    # Claim changes only supervisor state, never reconstructs missing evidence.
    assert len(pages(h)) == (1 if fault == "missing-first" else 2)


def test_terminal_journal_without_completion_is_corrupt_and_cannot_be_repaired(harness):
    h = harness
    params, _, current, calls = prefix(h)
    terminal = support.claim(h)
    support.run(h, terminal, fetcher(calls, current, payload(support.request(), day="2026-08-02")), params)
    with transaction(h.conn):
        h.conn.execute("DROP TRIGGER massive_occurrences_no_delete")
        h.conn.execute("DELETE FROM ops_massive_occurrences")
    # Model a corrupt catalog found while this exact job is still recoverable.
    failure = fail("RESOURCE_UNAVAILABLE", "synthetic output crash")
    assert support.retry(h, terminal, failure) == "retry_wait"
    attempt = support.claim(h)
    refused(h, attempt, params, support.no_fetch, "INTEGRITY_FAILED")
    assert len(pages(h)) == 2 and support.occurrences(h) == [] and spent(h, attempt.job_id) == 2


def test_another_job_cannot_inherit_partial_chain_or_its_used_call_allowance(harness):
    h = harness
    params, attempts, current, calls = prefix(h, params=support.parameters(cap=1))
    exhausted = support.claim(h)
    error = refused(h, exhausted, params, fetcher(calls, current, payload(support.request(), day="2026-08-02")),
                    "RESOURCE_LIMIT_EXCEEDED")
    assert support.retry(h, exhausted, error) == "failed"
    old = tuple(pages(h)[0])
    support.enqueue(h, "independent", params)
    independent = support.claim(h)
    assert independent.job_id != attempts[0].job_id
    _, result = support.run(h, independent, fetcher(calls, support.request(), payload(support.request())), params)
    assert calls == [support.request(), support.request()]
    assert spent(h, independent.job_id) == spent(h, attempts[0].job_id) == 1
    assert tuple(pages(h)[0]) == old and len(pages(h)) == 1
    assert result["coverage"]["bindings"][0]["evidence"]["job_id"] == independent.job_id
    assert "pages" not in result["coverage"]["bindings"][0]["evidence"]


def test_other_completed_acquisition_changes_gap_and_abandons_original_prefix(harness):
    h = harness
    params = support.parameters(cap=2)
    support.enqueue(h, params=params)
    first = support.claim(h)
    following = continuation(support.request(), 1, day="2026-08-02")
    calls = []
    error = refused(h, first, params, fetcher(calls, support.request(), payload(
        support.request(), following=following)), "SOURCE_NOT_FINAL")
    # Make the original retry temporarily ineligible, allowing another admitted job to finish.
    assert commit_attempt(h.conn, first.attempt_id, first.fence,
                          Outcome(succeeded=False, process_state="exited", exit_code=1, failure=error.problem),
                          clock=h.clock) == "retry_wait"
    with transaction(h.conn):
        h.conn.execute("UPDATE jobs SET next_eligible_at='2026-09-13T00:00:00.000000Z' WHERE job_id=?",
                       (first.job_id,))
    h.clock.advance(13)
    support.acquire(h, "other-completes-left", support.request("2026-08-01", "2026-08-10"))
    with transaction(h.conn):
        h.conn.execute("UPDATE jobs SET next_eligible_at='2026-09-12T00:00:00.000000Z' WHERE job_id=?",
                       (first.job_id,))
    before = tuple(pages(h)[0])
    attempt = support.claim(h)
    assert attempt.job_id == first.job_id
    gap = support.request("2026-08-11", "2026-08-31")
    _, result = support.run(h, attempt, fetcher(calls, gap, payload(gap)), params)
    assert calls == [support.request(), gap] and spent(h, first.job_id) == 2
    assert tuple(pages(h)[0]) == before
    assert [(binding["start"], binding["end"]) for binding in result["coverage"]["bindings"]] == [
        ("2026-08-01", "2026-08-10"), ("2026-08-11", "2026-08-31")]
    assert result["coverage"]["gaps"] == []


@pytest.mark.parametrize("mutation", ["UPDATE ops_massive_pages SET page_index=7",
                                     "DELETE FROM ops_massive_pages",
                                     "INSERT OR REPLACE INTO ops_massive_pages SELECT * FROM ops_massive_pages"])
def test_page_progress_is_append_only_in_actual_catalog(harness, mutation):
    h = harness
    prefix(h)
    before = tuple(pages(h)[0])
    with pytest.raises(sqlite3.IntegrityError):
        with transaction(h.conn):
            h.conn.execute(mutation)
    assert tuple(pages(h)[0]) == before


def test_older_cursor_cycle_is_rejected_before_any_new_raw_commit(harness):
    h = harness
    scope = support.request(dataset="splits")
    params, _, current, calls = prefix(h, params=support.parameters(scope, cap=3), count=2)
    before = [tuple(row) for row in pages(h)], [tuple(row) for row in raw_rows(h)]
    attempt = support.claim(h)
    refused(h, attempt, params, fetcher(calls, current, payload(
        scope, following=continuation(scope, 1))), "SOURCE_INVALID")
    assert ([tuple(row) for row in pages(h)], [tuple(row) for row in raw_rows(h)]) == before
    assert support.occurrences(h) == [] and spent(h, attempt.job_id) == 3


@pytest.mark.parametrize("generation", ["initial", "fresh-review"])
def test_only_initial_may_promote_exact_raw_only_evidence(harness, generation):
    h = harness
    support.enqueue_exact(h)
    first = support.claim(h)
    calls = []
    _, exact = support.run(h, first, fetcher(calls, support.request(), payload(support.request())))
    support.finish(h, first)
    h.clock.advance(13)
    assert support.occurrences(h) == pages(h) == []
    _, result = support.acquire(h, "history", generation=generation,
                                fetch=support.no_fetch if generation == "initial" else
                                fetcher(calls, support.request(), payload(support.request())))
    evidence = result["coverage"]["bindings"][0]["evidence"]
    assert evidence["raw_receipt_id"] == exact["raw_receipt_id"]
    assert evidence["first_received_at"] == exact["first_received_at"]
    assert evidence["origin"] == ("raw_reuse" if generation == "initial" else "http_response")
    assert len(calls) == (1 if generation == "initial" else 2)
    assert len(raw_rows(h)) == 1 and pages(h) == [] and "pages" not in evidence


def test_one_hundred_committed_pages_stop_at_bound_without_page_101_or_extra_charge(harness):
    h = harness
    scope = support.request(dataset="splits")
    params = support.parameters(scope, cap=100)
    with transaction(h.conn):
        h.conn.execute("UPDATE provider_accounts SET remaining=200")
    submitted = support.enqueue(h, params=params)
    row = h.conn.execute("SELECT max_attempts,retry_json FROM jobs WHERE job_id=?", (submitted.job_id,)).fetchone()
    assert row["max_attempts"] == 101
    assert json.loads(row["retry_json"])["backoff_seconds"] == [13]
    assert journal.MAX_PAGES == 100
    current, calls = scope, []
    for index in range(100):
        attempt = support.claim(h)
        assert attempt is not None and attempt.attempt_number == index + 1
        following = continuation(scope, index + 1)
        error = refused(h, attempt, params, fetcher(calls, current, payload(scope, following=following)),
                        "SOURCE_NOT_FINAL")
        assert support.retry(h, attempt, error) == "retry_wait"
        current = following
    before = [tuple(row) for row in pages(h)]
    last = support.claim(h)
    assert last.attempt_number == 101
    error = refused(h, last, params, support.no_fetch, "RESOURCE_LIMIT_EXCEEDED")
    assert "page limit" in error.problem.message
    assert support.retry(h, last, error) == "failed"
    assert [tuple(row) for row in pages(h)] == before
    assert len(raw_rows(h)) == len(calls) == spent(h, submitted.job_id) == 100
    assert support.occurrences(h) == []
    assert h.conn.execute("SELECT COUNT(*) FROM provider_reservations WHERE reserved_calls<>1").fetchone()[0] == 0


@pytest.mark.parametrize("cap", [3, 4])
def test_paginated_gap_and_failed_retry_share_cap_with_later_maximal_gap(harness, cap):
    h = harness
    support.acquire(h, "middle", support.request("2026-08-11", "2026-08-20"))
    params = support.parameters(cap=cap)
    submitted = support.enqueue(h, "mosaic", params)
    left = support.request("2026-08-01", "2026-08-10")
    right = support.request("2026-08-21", "2026-08-31")
    following = continuation(left, 1, day="2026-08-02")
    calls = []
    first = support.claim(h)
    error = refused(h, first, params, fetcher(calls, left, payload(left, following=following)), "SOURCE_NOT_FINAL")
    assert support.retry(h, first, error) == "retry_wait"
    failed = support.claim(h)
    error = refused(h, failed, params, fetcher(calls, following, b"", status=503), "TRANSIENT_SOURCE")
    assert support.retry(h, failed, error) == "retry_wait"
    terminal = support.claim(h)
    error = refused(h, terminal, params, fetcher(calls, following, payload(left, day="2026-08-02")),
                    "SOURCE_NOT_FINAL")
    assert support.retry(h, terminal, error) == "retry_wait"
    assert coverage.plan(h.conn, h.store, support.request(), "initial")["gaps"] == [right]
    assert len(support.occurrences(h)) == 2 and len(pages(h)) == 2
    last = support.claim(h)
    final_fetch = fetcher(calls, right, payload(right))
    if cap == 3:
        error = refused(h, last, params, final_fetch, "RESOURCE_LIMIT_EXCEEDED")
        assert support.retry(h, last, error) == "failed"
        assert len(support.occurrences(h)) == 2
    else:
        _, result = support.run(h, last, final_fetch, params)
        assert result["coverage"]["gaps"] == []
        bindings = result["coverage"]["bindings"]
        assert [(binding["start"], binding["end"]) for binding in bindings] == [
            ("2026-08-01", "2026-08-10"), ("2026-08-11", "2026-08-20"), ("2026-08-21", "2026-08-31")]
        assert bindings[0]["evidence"]["pages"] == [row["page_id"] for row in pages(h)]
        assert len(support.occurrences(h)) == 3
    assert calls == [left, following, following, right][:cap]
    assert spent(h, submitted.job_id) == cap


def test_continuation_uses_shared_fixed_https_edge_and_header_only_auth(harness, monkeypatch):
    h = harness
    params, _, current, calls = prefix(h)
    seen = []
    class Response:
        status_code = 200
        headers = {}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def iter_content(self, size):
            assert size == 64 * 1024
            yield payload(support.request(), day="2026-08-02")
    class Session:
        trust_env = True
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, url, **kwargs):
            assert self.trust_env is False
            assert spent(h, attempt.job_id) == 2
            seen.append((url, kwargs))
            return Response()
    marker = "synthetic-header-secret"
    monkeypatch.setattr(http, "os", SimpleNamespace(environ={"MASSIVE_API_KEY": marker}))
    monkeypatch.setattr(http.requests, "Session", Session)
    monkeypatch.setattr(http, "_deadline", nullcontext)
    attempt = support.claim(h)
    support.run(h, attempt, http.fetch, params)
    assert seen == [("https://api.massive.com" + current["endpoint"], {
        "params": current["params"] | {"adjusted": "false"},
        "headers": {"Authorization": "Bearer " + marker, "Accept-Encoding": "identity"},
        "stream": True, "allow_redirects": False, "timeout": (10, 15)})]
    assert marker not in str(tuple(h.conn.iterdump()))
    assert all(marker.encode() not in file.read_bytes() for file in h.root.rglob("*") if file.is_file())
