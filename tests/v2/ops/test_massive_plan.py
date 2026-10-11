"""Synthetic real receipts and read-only CLI checks; no provider or private data."""
import json
import os
import socket
import sqlite3

import pytest

from engine.v2.data.incremental import RawPayload, cache_raw_receipt
from engine.v2.foundation import ArtifactStore, canonical_json, content_hash
from engine.v2.foundation.artifacts import ArtifactError
from engine.v2.ops import massive_plan
from engine.v2.ops.errors import OpsError
from tests.ops_support import catalog
from tools.v2_massive_plan import main

PARAMS = {"tickers": ["TEST.A"], "start": "2026-01-01", "end": "2026-01-31",
          "as_of": "2026-10-10", "adjustment": "raw", "max_calls_per_request": 7}


@pytest.fixture(autouse=True)
def no_network_or_provider_lookups(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the plan attempted network I/O")

    original = type(os.environ).__getitem__

    def guarded(environment, key):
        assert not key.startswith("MASSIVE_"), "the plan read provider configuration"
        return original(environment, key)

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(type(os.environ), "__getitem__", guarded)


@pytest.fixture
def cache(tmp_path):
    conn, _, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "store")
    yield conn, store, tmp_path
    conn.close()


def inventory(cache=None, **changes):
    conn, store = (None, None) if cache is None else cache[:2]
    return massive_plan.plan(conn, store, **(PARAMS | changes))


def body(request, **changes):
    result = {"status": "OK", "results": [{"ticker": request["ticker"], "v": 1.25}],
              "resultsCount": 1}
    if request["dataset"] == "daily":
        result.update(ticker=request["ticker"], adjusted=request["params"]["adjusted"])
    return result | changes


def seed(cache, request, *, document=None, kind="complete", source=massive_plan.SOURCE,
         received_at="2026-10-09T12:00:00Z", raw=None):
    conn, store, _ = cache
    encoded = canonical_json(body(request) if document is None else document).encode()
    return cache_raw_receipt(conn, store,
                             RawPayload(payload=encoded if raw is None else raw,
                                        response_kind=kind, response_meta={}),
                             source=source, endpoint=request["endpoint"], request=request,
                             received_at=received_at)


def mutate_receipt(cache, record, field, value):
    """Deliberate corruption of this test's isolated catalog, never real data."""
    conn = cache[0]
    conn.execute("DROP TRIGGER data_raw_receipts_no_update")
    conn.execute(f"UPDATE data_raw_receipts SET {field}=? WHERE raw_receipt_id=?",
                 (value, record.raw_receipt_id))
    conn.commit()


def test_descriptors_and_budgets_are_canonical_and_explicit():
    result = inventory(tickers=["TEST.B", "TEST.A", "TEST.A"])
    assert result == inventory(tickers=["TEST.A", "TEST.B"])
    assert result["policy_version"] == "massive_stocks_basic.2026-10-10"
    assert result["requests_per_minute"] == 5
    assert result["minimum_initial_calls"] == 6
    assert result["maximum_http_attempts"] == 42
    requests = {u["request"]["dataset"]: u["request"] for u in result["units"][:3]}
    assert requests["daily"]["endpoint"].endswith("/TEST.A/range/1/day/2026-01-01/2026-01-31")
    assert requests["daily"]["params"] == {"adjusted": False, "sort": "asc", "limit": 50000}
    assert requests["splits"]["endpoint"] == "/stocks/v1/splits"
    assert requests["splits"]["params"]["execution_date.gte"] == "2026-01-01"
    assert requests["dividends"]["endpoint"] == "/stocks/v1/dividends"
    assert requests["dividends"]["params"]["sort"] == "ex_dividend_date.asc"
    assert requests["dividends"]["adjustment_basis"] == "corporate_action"
    text = json.dumps(result)
    assert "apiKey" not in text and "MASSIVE_" not in text and "https://" not in text


def test_adjustment_case_and_date_changes_separate_keys():
    raw = inventory(datasets=["daily"])["units"][0]
    adjusted = inventory(datasets=["daily"], adjustment="split_adjusted")["units"][0]
    assert raw["request_hash"] != adjusted["request_hash"]
    assert adjusted["request"]["params"]["adjusted"] is True
    assert raw["request_hash"] != inventory(datasets=["daily"], tickers=["test.a"])["units"][0]["request_hash"]
    assert raw["request_hash"] != inventory(datasets=["daily"], end="2026-01-30")["units"][0]["request_hash"]
    assert raw["request_hash"] == inventory(datasets=["daily"], as_of="2026-10-11")["units"][0]["request_hash"]


def test_calendar_year_boundaries_and_no_same_day_claim():
    result = inventory(start="2024-10-10", end="2026-10-09")
    assert result["earliest_date"] == "2024-10-10"
    assert result["latest_date"] == "2026-10-09"
    leap = inventory(as_of="2024-02-29", start="2022-02-28", end="2024-02-28")
    assert leap["earliest_date"] == "2022-02-28"


@pytest.mark.parametrize("change", [
    {"start": "2024-10-09"}, {"end": "2026-10-10"}, {"end": "2026-10-11"},
    {"start": "2026-02-01"}, {"start": "20260101"}, {"start": None},
    {"as_of": "0002-01-01"}, {"as_of": "2026-02-30"}, {"adjustment": None},
    {"adjustment": "total_return"}, {"transport": "s3"}, {"transport": "other"},
    {"tickers": []}, {"tickers": "TEST"}, {"tickers": ["TEST"] * 101},
    {"tickers": [1]}, {"tickers": ["TEST?apiKey=secret"]}, {"tickers": ["../TEST"]},
    {"tickers": [""]}, {"tickers": ["TEST\n"]}, {"datasets": ["minute"]},
    {"datasets": []}, {"datasets": "daily"}, {"datasets": [None]},
    {"datasets": ["daily"] * 4}, {"max_calls_per_request": 0},
    {"max_calls_per_request": 101}, {"max_calls_per_request": True},
    {"max_calls_per_request": 1.5}, {"max_calls_per_request": float("nan")},
])
def test_invalid_inputs_refuse_before_catalog_access(change):
    class Poison:
        def execute(self, *args):
            raise AssertionError("invalid input reached the catalog")

    with pytest.raises(OpsError, match="INVALID_REQUEST"):
        massive_plan.plan(Poison(), None, **(PARAMS | change))


def test_real_cache_hits_are_read_only_and_do_not_fetch(cache):
    conn, store, root = cache
    initial = inventory(cache)
    receipts = [seed(cache, u["request"]) for u in initial["units"]]
    before = (conn.total_changes, tuple(conn.iterdump()), sorted(p.relative_to(root) for p in store.root.rglob("*")))
    result = inventory(cache)
    assert result["minimum_initial_calls"] == result["maximum_http_attempts"] == 0
    assert [u["raw_receipt_id"] for u in result["units"]] == [r.raw_receipt_id for r in receipts]
    assert inventory(cache) == result
    assert before == (conn.total_changes, tuple(conn.iterdump()), sorted(p.relative_to(root) for p in store.root.rglob("*")))


def test_exact_only_no_other_source_or_superset_fallback(cache):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    seed(cache, req, source="other-public-provider")
    assert inventory(cache, datasets=["daily"])["minimum_initial_calls"] == 1
    seed(cache, req)
    assert inventory(cache, datasets=["daily"])["minimum_initial_calls"] == 0
    assert inventory(cache, datasets=["daily"], end="2026-01-30")["minimum_initial_calls"] == 1
    assert inventory(cache, datasets=["daily"], adjustment="split_adjusted")["minimum_initial_calls"] == 1


def test_only_missing_units_spend_the_declared_call_cap(cache):
    request = inventory()["units"][0]["request"]
    seed(cache, request)
    result = inventory(cache)
    assert result["minimum_initial_calls"] == 2
    assert result["maximum_http_attempts"] == 14
    assert result["max_calls_per_request"] == 7


def test_corrupt_catalog_evidence_json_is_typed(cache):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    record = seed(cache, req)
    malformed = "[" * 2000 + "0" + "]" * 2000
    # The real schema's JSON guard rejects excessively nested JSON itself.
    with pytest.raises(sqlite3.IntegrityError):
        mutate_receipt(cache, record, "request_json", malformed)


@pytest.mark.parametrize("kind", ["partial", "failed", "auth_failed", "rate_limited", "legitimate_empty"])
def test_newer_noncomplete_receipt_never_falls_back(cache, kind):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    seed(cache, req)
    seed(cache, req, document=body(req, request_id="second"), kind=kind)
    assert inventory(cache, datasets=["daily"])["minimum_initial_calls"] == 1


def test_latest_tie_break_and_timestamp_order(cache):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    seed(cache, req, kind="partial")
    latest = seed(cache, req, document=body(req, request_id="second"))
    seed(cache, req, document=body(req, request_id="older"), kind="failed",
         received_at="2026-10-08T12:00:00Z")
    assert inventory(cache, datasets=["daily"])["units"][0]["raw_receipt_id"] == latest.raw_receipt_id


@pytest.mark.parametrize("changes", [
    {"results": [], "resultsCount": 0}, {"next_url": "https://api.massive.com/next?apiKey=secret"},
])
def test_empty_or_paginated_is_missing_not_complete(cache, changes):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    seed(cache, req, document=body(req, **changes))
    result = inventory(cache, datasets=["daily"])
    assert result["minimum_initial_calls"] == 1
    assert result["maximum_http_attempts"] == 7
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("changes", [
    {"status": "ERROR"}, {"resultsCount": 2}, {"resultsCount": True},
    {"results": None}, {"results": [1]}, {"ticker": "WRONG"}, {"adjusted": 0},
])
def test_invalid_cached_envelopes_refuse(cache, changes):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    seed(cache, req, document=body(req, **changes))
    with pytest.raises(OpsError, match="SOURCE_INVALID"):
        inventory(cache, datasets=["daily"])


@pytest.mark.parametrize("raw", [b"not-json", b"[]", b"[" * 2000 + b"0" + b"]" * 2000])
def test_malformed_json_is_redacted(cache, raw):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    seed(cache, req, raw=raw)
    with pytest.raises(OpsError, match="SOURCE_INVALID"):
        inventory(cache, datasets=["daily"])


@pytest.mark.parametrize("field,value", [
    ("kind", "not_raw_response"), ("object_id", "art_" + "0" * 32),
    ("content_hash", "sha256:" + "0" * 64),
])
def test_corrupted_object_manifest_binding_refuses(cache, field, value):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    record = seed(cache, req)
    document = json.loads(cache[0].execute("SELECT artifact_ref_json FROM data_raw_receipts").fetchone()[0])
    document[field] = value
    mutate_receipt(cache, record, "artifact_ref_json", json.dumps(document))
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        inventory(cache, datasets=["daily"])


def test_request_hash_recomputed_without_false_integer_alias(cache):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    record = seed(cache, req)
    req["params"]["adjusted"] = 0
    mutate_receipt(cache, record, "request_json", json.dumps(req))
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        inventory(cache, datasets=["daily"])


def test_receipt_identity_is_recomputed(cache):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    record = seed(cache, req)
    mutate_receipt(cache, record, "raw_receipt_id", "raw_" + "0" * 32)
    with pytest.raises(OpsError, match="INTEGRITY_FAILED"):
        inventory(cache, datasets=["daily"])


def test_oversized_manifest_refuses_before_object_read(cache, monkeypatch):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    record = seed(cache, req)
    document = json.loads(cache[0].execute("SELECT artifact_ref_json FROM data_raw_receipts").fetchone()[0])
    document["byte_size"] = massive_plan.MAX_RAW_BYTES + 1
    mutate_receipt(cache, record, "artifact_ref_json", json.dumps(document))
    monkeypatch.setattr(massive_plan, "load_raw_receipt", lambda *a: pytest.fail("read oversized bytes"))
    with pytest.raises(OpsError, match="RESOURCE_LIMIT_EXCEEDED"):
        inventory(cache, datasets=["daily"])


@pytest.mark.parametrize("missing", [False, True])
def test_corrupt_or_missing_object_is_never_a_hit(cache, missing):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    record = seed(cache, req)
    digest = record.raw_hash.removeprefix("sha256:")
    path = cache[1].root / "objects" / digest[:2] / digest
    if missing:
        path.unlink()
    else:
        path.chmod(0o600)
        path.write_bytes(b"tampered")
    with pytest.raises((ArtifactError, OSError)):
        inventory(cache, datasets=["daily"])


def cli_args(root, **changes):
    args = {"catalog": root / "ops.sqlite", "store-root": root / "store", "ticker": "TEST.A",
            "start": PARAMS["start"], "end": PARAMS["end"], "as-of": PARAMS["as_of"],
            "adjustment": "raw", "max-calls-per-request": 7} | changes
    return [part for name, value in args.items() for part in ("--" + name, str(value))]


def test_cli_is_reachable_and_opens_a_read_only_transaction(cache, monkeypatch, capsys):
    conn, _, root = cache
    for unit in inventory(cache)["units"]:
        seed(cache, unit["request"])
    before = tuple(conn.iterdump())
    original, statements = sqlite3.connect, []

    def guarded(database, **kwargs):
        assert database.endswith("?mode=ro") and kwargs["uri"] is True
        result = original(database, **kwargs)
        result.set_trace_callback(statements.append)
        return result

    monkeypatch.setattr(sqlite3, "connect", guarded)
    assert main(cli_args(root)) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["plan_only"] and output["maximum_http_attempts"] == 0
    assert "BEGIN" in statements and "PRAGMA query_only=ON" in statements
    assert not any(s.startswith(("INSERT", "UPDATE", "CREATE", "DELETE", "COMMIT")) for s in statements)
    assert tuple(conn.iterdump()) == before


def test_missing_catalog_is_not_created(tmp_path, capsys):
    assert main(cli_args(tmp_path)) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "RESOURCE_UNAVAILABLE"}
    assert not (tmp_path / "ops.sqlite").exists()
    assert not (tmp_path / "store").exists()


def test_cli_s3_refuses_before_connect(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **kw: pytest.fail("opened catalog"))
    assert main(cli_args(tmp_path, transport="s3")) == 2
    assert json.loads(capsys.readouterr().out) == {
        "refused": "INVALID_REQUEST", "reason": "CAPABILITY_UNAVAILABLE"}


def test_request_hash_matches_existing_raw_receipt_contract(cache):
    unit = inventory(datasets=["daily"])["units"][0]
    record = seed(cache, unit["request"])
    assert unit["request_hash"] == record.request_hash == content_hash(unit["request"])
