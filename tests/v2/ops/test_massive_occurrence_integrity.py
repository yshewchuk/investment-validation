"""Independent native-catalog tamper oracles for completed occurrence reuse."""
import json

import pytest

from engine.v2.data.incremental import RawPayload, cache_raw_receipt
from engine.v2.foundation import canonical_json, content_hash, format_timestamp
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError
from engine.v2.ops.lifecycle import verify_fence
from engine.v2.ops.stores.massive import coverage
from tests.v2.ops import test_massive_coverage as support
from tests.v2.ops import test_massive_pagination as pagination

harness = support.harness
forbid_network = pagination.forbid_network


@pytest.mark.parametrize("changed", [None, "observed_at", "raw_hash"])
def test_occurrence_replay_is_idempotent_but_conflicting_evidence_refuses(harness, changed):
    h = harness
    support.enqueue(h)
    attempt = support.claim(h)
    calls = []
    _, result = support.run(h, attempt, support.fetching(calls), support.parameters())
    binding = result["coverage"]["bindings"][0]
    original = binding["evidence"]
    document = dict(original)
    if changed is not None:
        document[changed] = ("2026-09-12T00:00:01.000000Z" if changed == "observed_at" else "sha256:" + "0" * 64)
        assert content_hash(document) != binding["occurrence_id"]
    before = tuple(h.conn.iterdump())
    objects = {p.relative_to(h.store.root): (p.read_bytes(), p.stat().st_mtime_ns)
               for p in h.store.root.rglob("*") if p.is_file()}
    def replay():
        with transaction(h.conn):
            verify_fence(h.conn, attempt.attempt_id, attempt.fence, h.clock.now())
            coverage.record(h.conn, document)
    if changed is None:
        assert replay() is None
    else:
        with pytest.raises(OpsError) as refused:
            replay()
        assert refused.value.code == "INTEGRITY_FAILED" and not refused.value.problem.retryable
        assert str(refused.value) == "INTEGRITY_FAILED: Massive acquisition occurrence conflicts"
    assert tuple(h.conn.iterdump()) == before
    assert len(support.occurrences(h)) == 1
    assert json.loads(support.occurrences(h)[0]["document_json"]) == original
    assert {p.relative_to(h.store.root): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in h.store.root.rglob("*") if p.is_file()} == objects
    assert calls == [support.request()]


def rewrite_occurrence(h, row, document):
    """Coherently rehash only this synthetic catalog's corrupted completion.

    Leaving a stale digest/projection would mask schema, request and provenance
    regressions behind the unrelated row-projection guard.
    """
    request = document["request"]
    values = (content_hash(document), document["job_id"], content_hash(request),
              coverage.family(request), document["generation"], request["start"],
              request["end"], document["raw_receipt_id"], canonical_json(document))
    with transaction(h.conn):
        h.conn.execute("DROP TRIGGER massive_occurrences_no_update")
        h.conn.execute(
            "UPDATE ops_massive_occurrences SET occurrence_id=?,job_id=?,request_hash=?,"
            "family=?,generation=?,start_date=?,end_date=?,raw_receipt_id=?,document_json=? "
            "WHERE occurrence_id=?", (*values, row["occurrence_id"]))
    assert tuple(support.occurrences(h)[0]) == values


def refuse_reuse(h):
    """Exercise worker acquisition, failure commit, and next supervisor claim."""
    before = ([tuple(row) for row in support.occurrences(h)],
              [tuple(row) for row in pagination.pages(h)],
              [tuple(row) for row in pagination.raw_rows(h)])
    params = support.parameters()
    support.enqueue(h, "reuse-corrupt-evidence", params)
    attempt = support.claim(h)
    assert attempt is not None
    support.assert_terminal_evidence_failure(h, attempt, params)
    assert ([tuple(row) for row in support.occurrences(h)],
            [tuple(row) for row in pagination.pages(h)],
            [tuple(row) for row in pagination.raw_rows(h)]) == before


def completed_chain(h):
    params, _, current, calls = pagination.prefix(h)
    last = support.claim(h)
    support.run(h, last, pagination.fetcher(calls, current, pagination.payload(
        support.request(), day="2026-08-02")), params)
    support.finish(h, last)
    h.clock.advance(13)
    assert calls == [support.request(), current]
    assert len(pagination.pages(h)) == 2
    return support.occurrences(h)[0]


@pytest.mark.parametrize("column", ["occurrence_id", "request_hash"])
def test_occurrence_row_projection_tamper_is_terminal_without_fetch(harness, column):
    h = harness
    support.acquire(h, "original")
    row = support.occurrences(h)[0]
    document = json.loads(row["document_json"])
    with transaction(h.conn):
        h.conn.execute("DROP TRIGGER massive_occurrences_no_update")
        # Both columns are outside the interval lookup predicate. The genuine
        # document, raw receipt and body remain valid and must not hide damage.
        h.conn.execute(f"UPDATE ops_massive_occurrences SET {column}=?",
                       ("sha256:" + "0" * 64,))
    assert json.loads(support.occurrences(h)[0]["document_json"]) == document
    refuse_reuse(h)


@pytest.mark.parametrize("field", ["schema_version", "policy_version"])
def test_coherent_occurrence_with_wrong_version_is_terminal_without_fetch(harness, field):
    h = harness
    support.acquire(h, "original")
    row = support.occurrences(h)[0]
    document = json.loads(row["document_json"])
    document[field] = "unsupported.synthetic.v9"
    rewrite_occurrence(h, row, document)
    refuse_reuse(h)


def test_coherent_noncanonical_request_is_terminal_without_fetch(harness):
    h = harness
    support.acquire(h, "original")
    row = support.occurrences(h)[0]
    document = json.loads(row["document_json"])
    request = document["request"] | {"schema_version": "massive_request.synthetic.v9"}
    # Preserve valid raw provenance for this exact altered request. Otherwise
    # deleting canonical-request validation still fails raw identity checking.
    raw = cache_raw_receipt(
        h.conn, h.store,
        RawPayload(payload=support.body(request), response_kind="complete", response_meta={}),
        source=support.SOURCE, endpoint=request["endpoint"], request=request,
        received_at=document["first_received_at"])
    document.update(request=request, raw_receipt_id=raw.raw_receipt_id, raw_hash=raw.raw_hash)
    rewrite_occurrence(h, row, document)
    refuse_reuse(h)


@pytest.mark.parametrize("field", ["raw_receipt_id", "raw_hash", "first_received_at",
                                   "observed_at", "origin", "attempt_id", "fence"])
def test_completion_must_pin_each_last_page_field(harness, field):
    h = harness
    row = completed_chain(h)
    document = json.loads(row["document_json"])
    first = json.loads(pagination.pages(h)[0]["document_json"])
    last = json.loads(pagination.pages(h)[-1]["document_json"])
    changed = first[field] if field != "origin" else "raw_reuse"
    assert changed != last[field]
    document[field] = changed
    # Keep every page and raw object untouched, with all other completion
    # fields still pinned. Only the selected last-page equality can refuse.
    assert {key for key in last if key in document and document[key] != last[key]} == {
        "schema_version", "request", field}
    rewrite_occurrence(h, row, document)
    refuse_reuse(h)


@pytest.mark.parametrize("page_ids", ["empty", "reversed"])
def test_completion_rejects_empty_or_reordered_page_ids(harness, page_ids):
    h = harness
    row = completed_chain(h)
    document = json.loads(row["document_json"])
    document["pages"] = [] if page_ids == "empty" else list(reversed(document["pages"]))
    # Empty-list rejection intentionally has overlapping guards: a genuine
    # terminal native journal cannot itself have an empty page-ID sequence.
    rewrite_occurrence(h, row, document)
    refuse_reuse(h)


def test_raw_reuse_cannot_claim_a_later_observation_than_its_raw_receipt(harness):
    h = harness
    support.enqueue_exact(h)
    original = support.claim(h)
    support.run(h, original, support.fetching([]))
    support.finish(h, original)
    h.clock.advance(13)
    support.acquire(h, "adopt-raw", fetch=support.no_fetch)
    row = support.occurrences(h)[0]
    document = json.loads(row["document_json"])
    assert document["origin"] == "raw_reuse"
    assert document["observed_at"] == document["first_received_at"]
    # Both timestamps are canonical and after scope close. Thus timestamp
    # syntax and closed-day validation cannot mask the raw-reuse time guard.
    document["observed_at"] = format_timestamp(h.clock.now())
    assert document["observed_at"] > document["first_received_at"]
    rewrite_occurrence(h, row, document)
    refuse_reuse(h)
