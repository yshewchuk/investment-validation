"""Network-free SQLite journal oracles, corruption refusals and transaction ownership."""
import json
import socket
import sqlite3
from copy import deepcopy

import pytest

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.ops import massive_plan
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.stores.massive import journal

ORIGINAL = {"schema_version": "massive_request.v1", "source": "massive-stocks",
            "transport": "rest", "dataset": "splits", "ticker": "SYNTH",
            "start": "2026-08-01", "end": "2026-08-31",
            "endpoint": "/stocks/v1/splits", "adjustment_basis": "corporate_action",
            "params": {"ticker": "SYNTH", "execution_date.gte": "2026-08-01",
                       "execution_date.lte": "2026-08-31", "sort": "execution_date.asc",
                       "limit": 5000}}


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic journal test attempted a network connection")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def conn():
    database = sqlite3.connect(":memory:")
    database.row_factory = sqlite3.Row
    database.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE jobs (job_id TEXT PRIMARY KEY);
        CREATE TABLE raw (raw_receipt_id TEXT PRIMARY KEY);
        CREATE TABLE ops_massive_pages (
            page_id TEXT PRIMARY KEY,
            job_id TEXT REFERENCES jobs(job_id),
            request_hash TEXT,
            page_index INTEGER CHECK(page_index>=0),
            raw_receipt_id TEXT REFERENCES raw(raw_receipt_id),
            document_json TEXT,
            UNIQUE(job_id, request_hash, page_index)
        );
        INSERT INTO jobs VALUES ('job'), ('other-job');
    """)
    database.executemany("INSERT INTO raw VALUES (?)", [(f"raw-{i}",) for i in range(101)])
    database.commit()
    yield database
    database.close()


def request(index):
    result = deepcopy(ORIGINAL)
    if index:
        result["params"]["cursor"] = f"page-{index}"
    return result


def page(index=0, *, previous=None, next_request=None):
    return {"schema_version": "massive_acquisition_page.v1",
            "policy_version": massive_plan.POLICY_VERSION,
            "job_id": "job", "generation": "initial", "scope_hash": content_hash(ORIGINAL),
            "page_index": index, "previous_page_id": previous, "request": request(index),
            "next_request": next_request, "raw_receipt_id": f"raw-{index}",
            "raw_hash": content_hash({"synthetic_page": index}),
            "first_received_at": "2026-09-01T12:00:00.000000Z",
            "observed_at": "2026-09-01T12:00:00.000000Z", "origin": "http_response",
            "attempt_id": "attempt", "fence": 1}


def insert(conn, document, **columns):
    """Seed independent row projections, including deliberately corrupt evidence."""
    values = {"page_id": content_hash(document), "job_id": "job",
              "request_hash": content_hash(ORIGINAL), "page_index": document.get("page_index", 0),
              "raw_receipt_id": document.get("raw_receipt_id", "raw-0"),
              "document_json": canonical_json(document)}
    values.update(columns)
    conn.execute("INSERT INTO ops_massive_pages VALUES (?,?,?,?,?,?)", tuple(values.values()))
    return values["page_id"]


def progress(conn, verify, *, generation="initial", job_id="job", original=ORIGINAL):
    return journal.progress(conn, original, generation, job_id, verify)


def no_verify(_):
    pytest.fail("malformed metadata must refuse before raw verification")


def test_empty_progress_returns_original_gap_without_verification(conn):
    assert progress(conn, no_verify) == {
        "page_ids": [], "next_request": ORIGINAL, "last": None, "requests": []}


def test_append_identity_replay_and_rollback_belong_to_caller(conn):
    document = page(next_request=request(1))
    expected = content_hash(document)
    assert journal.append(conn, document) == expected
    assert journal.append(conn, document) == expected
    assert conn.in_transaction
    assert conn.execute("SELECT count(*) FROM ops_massive_pages").fetchone()[0] == 1
    row = conn.execute("SELECT * FROM ops_massive_pages").fetchone()
    assert tuple(row) == (expected, "job", content_hash(ORIGINAL), 0,
                          "raw-0", canonical_json(document))
    conn.rollback()
    assert conn.execute("SELECT count(*) FROM ops_massive_pages").fetchone()[0] == 0


def test_append_refuses_conflicting_page_position_or_existing_identity(conn):
    document = page()
    identity = journal.append(conn, document)
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        journal.append(conn, document | {"raw_hash": content_hash("different response")})
    conn.execute("UPDATE ops_massive_pages SET raw_receipt_id='raw-1' WHERE page_id=?", (identity,))
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        journal.append(conn, document)
    assert conn.execute("SELECT count(*) FROM ops_massive_pages").fetchone()[0] == 1


def test_append_foreign_key_failure_is_sanitized(conn):
    with pytest.raises(OpsError, match="INTEGRITY_FAILED") as error:
        journal.append(conn, page() | {"raw_receipt_id": "missing-private-receipt"})
    assert "missing-private-receipt" not in str(error.value)


@pytest.mark.parametrize("terminal", [False, True])
def test_verified_chain_returns_ordered_progress_and_actual_request_hashes(conn, terminal):
    documents, identities = [], []
    for index in range(3):
        document = page(index, previous=identities[-1] if identities else None,
                        next_request=None if terminal and index == 2 else request(index + 1))
        documents.append(document)
        identities.append(insert(conn, document))
    calls = []
    def verify(document):
        index = len(calls)
        assert document == documents[index]
        calls.append(document["raw_receipt_id"])
        return None if terminal and index == 2 else request(index + 1)
    assert progress(conn, verify) == {
        "page_ids": identities, "next_request": None if terminal else request(3),
        "last": documents[-1], "requests": [content_hash(request(i)) for i in range(3)]}
    assert calls == ["raw-0", "raw-1", "raw-2"]


def test_progress_is_scoped_to_job_and_original_gap(conn):
    insert(conn, page())
    assert progress(conn, no_verify, job_id="other-job")["page_ids"] == []
    assert progress(conn, no_verify, original=ORIGINAL | {"end": "2026-08-30"})["page_ids"] == []
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        progress(conn, no_verify, generation="refresh")


@pytest.mark.parametrize("count", [100, 101])
def test_page_count_bound_precedes_all_verification(conn, count):
    previous = None
    for index in range(count):
        previous = insert(conn, page(index, previous=previous,
                                     next_request=None if index == count - 1 else request(index + 1)))
    calls = []
    def verify(document):
        index = len(calls)
        calls.append(index)
        return None if index == count - 1 else request(index + 1)
    if count == 101:
        with pytest.raises(OpsError, match="RESOURCE_LIMIT_EXCEEDED"):
            progress(conn, verify)
        assert calls == []
    else:
        result = progress(conn, verify)
        assert result["next_request"] is None
        assert len(result["page_ids"]) == len(calls) == 100


@pytest.mark.parametrize("change", [
    {"unknown": "private"}, {"schema_version": "unknown"}, {"policy_version": "unknown"},
    {"page_index": True}, {"page_index": 0.0}, {"page_index": "0"}, {"page_index": -1},
    {"fence": True}, {"fence": -1}, {"fence": 0}, {"generation": None}, {"job_id": ""},
    {"request": []}, {"next_request": "private"}, {"previous_page_id": False},
    {"origin": "unverified"}, {"raw_hash": None},
])
def test_append_rejects_invalid_document_shape(conn, change):
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        journal.append(conn, page() | change)
    assert conn.execute("SELECT count(*) FROM ops_massive_pages").fetchone()[0] == 0


@pytest.mark.parametrize("field", sorted(page()))
def test_missing_fields_refuse_in_append_and_progress(conn, field):
    document = page()
    del document[field]
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        journal.append(conn, document)
    insert(conn, document)
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        progress(conn, no_verify)


@pytest.mark.parametrize("change", [
    {"unknown": "private"}, {"schema_version": "unknown"}, {"policy_version": "unknown"},
    {"generation": "other"}, {"job_id": "other-job"}, {"scope_hash": content_hash("other")},
    {"page_index": True}, {"previous_page_id": "missing"}, {"request": request(2)},
    {"origin": "unverified"}, {"fence": False}, {"fence": 0}, {"raw_hash": None},
])
def test_persisted_metadata_is_reverified_before_callback(conn, change):
    insert(conn, page() | change)
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        progress(conn, no_verify)


@pytest.mark.parametrize("columns", [
    {"page_id": content_hash("wrong identity")}, {"raw_receipt_id": "raw-1"},
    {"page_index": 1}, {"document_json": "{"}, {"document_json": "null"},
    {"document_json": "[]"}, {"document_json": json.dumps(page(), indent=2)},
    {"document_json": canonical_json(page())[:-1] + ',"page_index":0}'},
])
def test_corrupt_rows_or_noncanonical_documents_refuse(conn, columns):
    insert(conn, page(), **columns)
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        progress(conn, no_verify)


@pytest.mark.parametrize("damage", ["missing-first", "missing-middle", "reordered", "predecessor"])
def test_missing_reordered_or_unlinked_pages_never_form_a_prefix(conn, damage):
    first = page(next_request=request(1))
    first_id = content_hash(first)
    second = page(1, previous=first_id, next_request=request(2))
    third = page(2, previous=content_hash(second))
    if damage != "missing-first":
        insert(conn, first, page_index=1 if damage == "reordered" else 0)
    if damage != "missing-middle":
        insert(conn, second, page_index=0 if damage == "reordered" else 1)
    if damage == "predecessor":
        third["previous_page_id"] = first_id
    insert(conn, third)
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        progress(conn, lambda document: None if document["page_index"] == 2
                 else request(document["page_index"] + 1))


def test_no_page_is_allowed_after_a_terminal_page(conn):
    first_id = insert(conn, page())
    insert(conn, page(1, previous=first_id))
    calls = []
    def verify(document):
        calls.append(document["page_index"])
        return None
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        progress(conn, verify)
    assert calls == [0]


@pytest.mark.parametrize("cycle", ["self", "previous"])
def test_continuation_cycles_refuse_even_at_incomplete_tail(conn, cycle):
    if cycle == "self":
        insert(conn, page(next_request=request(0)))
    else:
        first_id = insert(conn, page(next_request=request(1)))
        insert(conn, page(1, previous=first_id, next_request=request(0)))
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        progress(conn, lambda document: request(1) if document["page_index"] == 0
                 and cycle == "previous" else request(0))


@pytest.mark.parametrize("checked_next", [None, request(2), {"params": {"private": "secret"}}])
def test_stored_next_request_must_equal_callback_verified_continuation(conn, checked_next):
    insert(conn, page(next_request=request(1)))
    with pytest.raises(OpsError, match="INTEGRITY_FAILED") as error:
        progress(conn, lambda _: checked_next)
    assert "secret" not in str(error.value)


def test_boolean_does_not_alias_a_request_integer(conn):
    original = request(0)
    original["params"]["limit"] = 1
    document = page() | {"scope_hash": content_hash(original), "request": deepcopy(original)}
    document["request"]["params"]["limit"] = True
    insert(conn, document, request_hash=content_hash(original))
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        progress(conn, no_verify, original=original)


@pytest.mark.parametrize("error", [ValueError, TypeError, KeyError, RecursionError])
def test_malformed_verifier_data_is_sanitized(conn, error):
    insert(conn, page())
    def verify(_):
        raise error("private /raw/path?apiKey=secret")
    with pytest.raises(OpsError, match="INTEGRITY_FAILED") as caught:
        progress(conn, verify)
    assert "private" not in str(caught.value) and "secret" not in str(caught.value)


@pytest.mark.parametrize("code", ["RESOURCE_LIMIT_EXCEEDED", "SOURCE_NOT_FINAL", "INTEGRITY_FAILED"])
def test_typed_verifier_failure_is_preserved(conn, code):
    insert(conn, page())
    error = fail(code, "synthetic typed refusal")
    def verify(_):
        raise error
    with pytest.raises(OpsError) as caught:
        progress(conn, verify)
    assert caught.value is error


def test_recursive_input_and_deep_persisted_json_are_sanitized(conn):
    document = page()
    document["request"]["recursive"] = document
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        journal.append(conn, document)
    insert(conn, page(), document_json="[" * 2000 + "0" + "]" * 2000)
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        progress(conn, no_verify)
