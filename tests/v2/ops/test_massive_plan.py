"""Synthetic real receipts and read-only CLI checks; no provider or private data."""
import json
import os
import socket
import sqlite3
import subprocess
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.incremental import RawPayload, cache_raw_receipt
from engine.v2.foundation import ArtifactStore, canonical_json, content_hash
from engine.v2.foundation.artifacts import ArtifactError
from engine.v2.ops import massive_plan
from engine.v2.ops.errors import OpsError
from tests.ops_support import catalog
from tools import v2_massive_plan as cli
from tools.v2_massive_plan import main

PARAMS = {"tickers": ["TEST.A"], "start": "2026-01-01", "end": "2026-01-31",
          "as_of": "2026-10-10", "adjustment": "raw", "max_calls_per_request": 7}
DATASETS = ("daily", "splits", "dividends")
PRIVATE_PAYLOAD_MARKER = "synthetic-private-raw-payload-marker"


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


@pytest.mark.parametrize("ticker,start,end,adjustment,adjusted", [
    ("TEST.A", "2026-01-01", "2026-01-31", "raw", False),
    ("test.b", "2025-03-07", "2025-04-09", "split_adjusted", True),
    ("BRK.B", "2026-07-08", "2026-08-19", "raw", False),
    ("T" * 32, "2026-01-01", "2026-01-31", "raw", False),
    ("ONE", "2026-07-08", "2026-07-08", "raw", False),
])
def test_descriptors_and_budgets_are_canonical_and_explicit(ticker, start, end, adjustment, adjusted):
    scope = {"start": start, "end": end, "adjustment": adjustment}
    result = inventory(tickers=["ZZZ", ticker, ticker], **scope)
    assert result == inventory(tickers=[ticker, "ZZZ"], **scope)
    assert result["schema_version"] == "massive_cache_plan.v1"
    assert result["policy_version"] == "massive_stocks_basic.2026-10-10"
    assert result["requests_per_minute"] == 5
    assert result["minimum_initial_calls"] == 6
    assert result["maximum_http_attempts"] == 42
    expected = {}
    for stock in {ticker, "ZZZ"}:
        common = {"schema_version": "massive_request.v1", "source": "massive-stocks",
                  "transport": "rest", "ticker": stock, "start": start, "end": end}
        expected[stock] = {
            "daily": common | {"dataset": "daily", "adjustment_basis": adjustment,
                                "endpoint": f"/v2/aggs/ticker/{stock}/range/1/day/{start}/{end}",
                                "params": {"adjusted": adjusted, "sort": "asc", "limit": 50000}},
            "splits": common | {"dataset": "splits", "adjustment_basis": "corporate_action",
                                 "endpoint": "/stocks/v1/splits",
                                 "params": {"ticker": stock, "execution_date.gte": start,
                                            "execution_date.lte": end,
                                            "sort": "execution_date.asc", "limit": 5000}},
            "dividends": common | {"dataset": "dividends", "adjustment_basis": "corporate_action",
                                    "endpoint": "/stocks/v1/dividends",
                                    "params": {"ticker": stock, "ex_dividend_date.gte": start,
                                               "ex_dividend_date.lte": end,
                                               "sort": "ex_dividend_date.asc", "limit": 5000}},
        }
    assert result["units"] == [
        {"request": request, "request_hash": content_hash(request), "raw_receipt_id": None}
        for stock in sorted(expected) for _, request in sorted(expected[stock].items())
    ]
    text = json.dumps(result)
    assert "apiKey" not in text and "MASSIVE_" not in text and "https://" not in text


@pytest.mark.parametrize("datasets,unique", [
    (["splits", "daily", "splits"], ["daily", "splits"]),
    (("dividends", "dividends", "dividends"), ["dividends"]),
])
def test_repeated_datasets_have_unique_units_and_budgets(cache, datasets, unique):
    units = inventory(datasets=unique)["units"]
    result = inventory(cache, datasets=datasets)
    assert [unit["request"]["dataset"] for unit in result["units"]] == unique
    assert result["units"] == units
    assert result["minimum_initial_calls"] == len(unique)
    assert result["maximum_http_attempts"] == 7 * len(unique)
    record = seed(cache, units[0]["request"])
    result = inventory(cache, datasets=datasets)
    assert result["units"] == [units[0] | {"raw_receipt_id": record.raw_receipt_id}, *units[1:]]
    assert result["minimum_initial_calls"] == len(unique) - 1
    assert result["maximum_http_attempts"] == 7 * (len(unique) - 1)


def test_exactly_one_hundred_tickers_are_accepted():
    tickers = [f"T{index:03d}" for index in range(100)]
    result = inventory(tickers=tickers, datasets=["daily"])
    assert [unit["request"]["ticker"] for unit in result["units"]] == tickers
    assert len(result["units"]) == result["minimum_initial_calls"] == 100
    assert result["maximum_http_attempts"] == 700


def test_adjustment_case_and_date_changes_separate_keys():
    raw = inventory(datasets=["daily"])["units"][0]
    adjusted = inventory(datasets=["daily"], adjustment="split_adjusted")["units"][0]
    assert raw["request_hash"] != adjusted["request_hash"]
    assert adjusted["request"]["params"]["adjusted"] is True
    assert raw["request_hash"] != inventory(datasets=["daily"], tickers=["test.a"])["units"][0]["request_hash"]
    assert raw["request_hash"] != inventory(datasets=["daily"], end="2026-01-30")["units"][0]["request_hash"]
    assert raw["request_hash"] == inventory(datasets=["daily"], as_of="2026-10-11")["units"][0]["request_hash"]


def test_calendar_year_boundaries_and_no_same_day_claim():
    result = inventory(as_of="2026-10-10", start="2024-10-10", end="2026-10-09")
    assert result["as_of"] == "2026-10-10"
    assert result["earliest_date"] == "2024-10-10"
    assert result["latest_date"] == "2026-10-09"
    leap = inventory(as_of="2024-02-29", start="2022-02-28", end="2024-02-28")
    assert leap["as_of"] == "2024-02-29"
    assert leap["earliest_date"] == "2022-02-28"
    assert leap["latest_date"] == "2024-02-28"


@pytest.mark.parametrize("change", [
    {"start": "20260101"}, {"start": None},
    {"as_of": "0002-01-01"}, {"as_of": "2026-02-30"}, {"adjustment": None},
    {"adjustment": "total_return"},
    {"tickers": []}, {"tickers": "TEST"}, {"tickers": ["TEST"] * 101},
    {"tickers": [1]}, {"tickers": ["TEST?apiKey=secret"]}, {"tickers": ["../TEST"]},
    {"tickers": [""]}, {"tickers": ["TEST\n"]}, {"tickers": ["T" * 33]},
    {"datasets": []}, {"datasets": "daily"},
    {"datasets": ["daily"] * 4}, {"max_calls_per_request": 0},
    {"max_calls_per_request": 101}, {"max_calls_per_request": True},
    {"max_calls_per_request": 1.5}, {"max_calls_per_request": float("nan")},
])
def test_invalid_inputs_refuse_before_catalog_access(change):
    class Poison:
        def execute(self, *args):
            raise AssertionError("invalid input reached the catalog")

    with pytest.raises(OpsError) as error:
        massive_plan.plan(Poison(), None, **(PARAMS | change))
    assert error.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("start,end", [
    ("2026-02-01", "2026-01-31"),
    ("2024-10-09", "2024-10-08"),
    ("2026-10-11", "2026-10-10"),
])
def test_reversed_dates_refuse_as_invalid_before_history_or_catalog(start, end):
    class Poison:
        def execute(self, *args):
            raise AssertionError("reversed dates reached the catalog")

    with pytest.raises(OpsError) as error:
        massive_plan.plan(Poison(), None, **(PARAMS | {"start": start, "end": end}))
    assert error.value.code == "INVALID_REQUEST"
    assert error.value.problem.details == {}


@pytest.mark.parametrize("change,reason", [
    ({"start": "2024-10-09"}, "HISTORY_UNAVAILABLE"),
    ({"end": "2026-10-10"}, "HISTORY_UNAVAILABLE"),
    ({"end": "2026-10-11"}, "HISTORY_UNAVAILABLE"),
    ({"transport": "s3"}, "CAPABILITY_UNAVAILABLE"),
    ({"transport": "other"}, "CAPABILITY_UNAVAILABLE"),
    ({"datasets": ["minute"]}, "CAPABILITY_UNAVAILABLE"),
    ({"datasets": [None]}, "CAPABILITY_UNAVAILABLE"),
])
def test_policy_refusals_preserve_exact_reason_before_catalog_access(change, reason):
    class Poison:
        def execute(self, *args):
            raise AssertionError("unavailable capability reached the catalog")

    with pytest.raises(OpsError) as error:
        massive_plan.plan(Poison(), None, **(PARAMS | change))
    assert error.value.code == "INVALID_REQUEST"
    assert error.value.problem.details == {"reason": reason}


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


@pytest.mark.parametrize("dataset", ["splits", "dividends"])
@pytest.mark.parametrize("row_count", [1, 2, 3])
def test_countless_corporate_action_envelopes_are_complete_hits(cache, dataset, row_count):
    unit = inventory(datasets=[dataset])["units"][0]
    document = {"status": "OK", "results": [
        {"ticker": unit["request"]["ticker"], "v": index + 1.25} for index in range(row_count)
    ]}
    assert "resultsCount" not in document
    record = seed(cache, unit["request"], document=document)
    result = inventory(cache, datasets=[dataset])
    assert result["units"] == [unit | {"raw_receipt_id": record.raw_receipt_id}]
    assert result["minimum_initial_calls"] == result["maximum_http_attempts"] == 0


def test_exact_only_no_other_source_or_superset_fallback(cache):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    seed(cache, req, source="other-public-provider")
    assert inventory(cache, datasets=["daily"])["minimum_initial_calls"] == 1
    seed(cache, req)
    assert inventory(cache, datasets=["daily"])["minimum_initial_calls"] == 0
    assert inventory(cache, datasets=["daily"], end="2026-01-30")["minimum_initial_calls"] == 1
    assert inventory(cache, datasets=["daily"], adjustment="split_adjusted")["minimum_initial_calls"] == 1


@pytest.mark.parametrize("cap", [1, 7, 100])
def test_only_missing_units_spend_the_declared_call_cap(cache, cap):
    request = inventory()["units"][0]["request"]
    seed(cache, request)
    result = inventory(cache, max_calls_per_request=cap)
    assert result["minimum_initial_calls"] == 2
    assert result["maximum_http_attempts"] == 2 * cap
    assert result["max_calls_per_request"] == cap


@pytest.mark.parametrize("missing_key,error_type,code", [
    ("kind", OpsError, "INTEGRITY_FAILED"), ("content_hash", OpsError, "INTEGRITY_FAILED"),
    ("byte_size", OpsError, "INTEGRITY_FAILED"), ("object_id", DataError, "MANIFEST_CORRUPT"),
])
def test_schema_admissible_incomplete_manifest_is_typed(cache, missing_key, error_type, code):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    record = seed(cache, req)
    document = json.loads(cache[0].execute("SELECT artifact_ref_json FROM data_raw_receipts").fetchone()[0])
    del document[missing_key]
    malformed = json.dumps(document)
    mutate_receipt(cache, record, "artifact_ref_json", malformed)
    assert cache[0].execute("SELECT artifact_ref_json FROM data_raw_receipts").fetchone()[0] == malformed
    with pytest.raises(error_type) as error:
        inventory(cache, datasets=["daily"])
    assert error.value.code == code


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("kind", [
    "legitimate_empty", "partial", "failed", "auth_failed", "rate_limited", "delayed", "unavailable",
])
def test_newer_noncomplete_receipt_never_falls_back(cache, dataset, kind):
    missing = inventory(datasets=[dataset])
    req = missing["units"][0]["request"]
    seed(cache, req)
    seed(cache, req, document=body(req, request_id="second"), kind=kind,
         received_at="2026-10-10T12:00:00Z")
    assert inventory(cache, datasets=[dataset]) == missing


@pytest.mark.parametrize("indexed", [True, False])
def test_latest_tie_break_and_timestamp_order(cache, indexed):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    first = seed(cache, req, kind="partial")
    latest = seed(cache, req, document=body(req, request_id="second"))
    older = seed(cache, req, document=body(req, request_id="older"), kind="failed",
                 received_at="2026-10-08T12:00:00Z")
    if not indexed:
        # Shadow the real receipts with an index-free copy, including no UNIQUE autoindex.
        cache[0].execute("CREATE TEMP TABLE data_raw_receipts AS "
                         "SELECT * FROM main.data_raw_receipts ORDER BY rowid")
        assert cache[0].execute("PRAGMA temp.index_list(data_raw_receipts)").fetchall() == []
        assert cache[0].execute("SELECT raw_receipt_id FROM data_raw_receipts "
                                "ORDER BY received_at DESC LIMIT 1").fetchone()[0] == first.raw_receipt_id
        assert cache[0].execute("SELECT raw_receipt_id FROM data_raw_receipts "
                                "ORDER BY rowid DESC LIMIT 1").fetchone()[0] == older.raw_receipt_id
    assert inventory(cache, datasets=["daily"])["units"][0]["raw_receipt_id"] == latest.raw_receipt_id


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("changes", [
    {"results": [], "resultsCount": 0}, {"next_url": "https://api.massive.com/next?apiKey=secret"},
])
def test_empty_or_paginated_is_missing_not_complete(cache, dataset, changes):
    missing = inventory(datasets=[dataset])
    req = missing["units"][0]["request"]
    seed(cache, req, document=body(req, **changes))
    result = inventory(cache, datasets=[dataset])
    assert result == missing
    assert result["minimum_initial_calls"] == 1
    assert result["maximum_http_attempts"] == 7
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("changes", [
    {"status": "ERROR"}, {"resultsCount": 2}, {"resultsCount": True},
    {"results": None}, {"results": [1]},
])
def test_invalid_cached_envelopes_refuse(cache, dataset, changes):
    req = inventory(datasets=[dataset])["units"][0]["request"]
    seed(cache, req, document=body(req, **changes))
    with pytest.raises(OpsError) as error:
        inventory(cache, datasets=[dataset])
    assert error.value.code == "SOURCE_INVALID"


@pytest.mark.parametrize("changes", [{"ticker": "WRONG"}, {"adjusted": 0}])
def test_daily_identity_mismatch_refuses(cache, changes):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    seed(cache, req, document=body(req, **changes))
    with pytest.raises(OpsError) as error:
        inventory(cache, datasets=["daily"])
    assert error.value.code == "SOURCE_INVALID"


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("raw", [
    (PRIVATE_PAYLOAD_MARKER + ":not-json").encode(),
    json.dumps([PRIVATE_PAYLOAD_MARKER]).encode(),
    b"[" * 2000 + json.dumps(PRIVATE_PAYLOAD_MARKER).encode() + b"]" * 2000,
], ids=["invalid-json", "not-an-object", "too-deep"])
def test_malformed_json_is_redacted(cache, dataset, raw):
    req = inventory(datasets=[dataset])["units"][0]["request"]
    seed(cache, req, raw=raw)
    with pytest.raises(OpsError) as error:
        inventory(cache, datasets=[dataset])
    assert error.value.code == "SOURCE_INVALID"
    rendered = "\n".join((str(error.value), repr(error.value), json.dumps(error.value.args),
                          canonical_json(asdict(error.value.problem))))
    assert PRIVATE_PAYLOAD_MARKER in raw.decode()
    assert PRIVATE_PAYLOAD_MARKER not in rendered
    assert raw.decode() not in rendered


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
    with pytest.raises(OpsError) as error:
        inventory(cache, datasets=["daily"])
    assert error.value.code == "INTEGRITY_FAILED"


def test_request_hash_recomputed_without_false_integer_alias(cache):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    record = seed(cache, req)
    req["params"]["adjusted"] = 0
    mutate_receipt(cache, record, "request_json", json.dumps(req))
    with pytest.raises(OpsError) as error:
        inventory(cache, datasets=["daily"])
    assert error.value.code == "INTEGRITY_FAILED"


def test_receipt_identity_is_recomputed(cache):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    record = seed(cache, req)
    mutate_receipt(cache, record, "raw_receipt_id", "raw_" + "0" * 32)
    with pytest.raises(OpsError) as error:
        inventory(cache, datasets=["daily"])
    assert error.value.code == "INTEGRITY_FAILED"


@pytest.mark.parametrize("size", [-1, True, False, 1.0, "1", None, 16 * 1024 * 1024 + 1])
def test_invalid_manifest_size_refuses_before_object_read(cache, monkeypatch, size):
    req = inventory(datasets=["daily"])["units"][0]["request"]
    record = seed(cache, req)
    document = json.loads(cache[0].execute("SELECT artifact_ref_json FROM data_raw_receipts").fetchone()[0])
    document["byte_size"] = size
    mutate_receipt(cache, record, "artifact_ref_json", json.dumps(document))
    monkeypatch.setattr(massive_plan, "load_raw_receipt", lambda *a: pytest.fail("read invalid-size bytes"))
    with pytest.raises(OpsError) as error:
        inventory(cache, datasets=["daily"])
    assert error.value.code == "RESOURCE_LIMIT_EXCEEDED"


def test_exact_maximum_raw_size_is_a_verified_cache_hit(cache):
    unit = inventory(datasets=["daily"])["units"][0]
    encoded = canonical_json(body(unit["request"])).encode()
    raw = encoded + b" " * (16 * 1024 * 1024 - len(encoded))
    assert len(raw) == 16 * 1024 * 1024
    record = seed(cache, unit["request"], raw=raw)
    assert record.object_ref.byte_size == 16 * 1024 * 1024
    result = inventory(cache, datasets=["daily"])
    assert result["units"] == [unit | {"raw_receipt_id": record.raw_receipt_id}]
    assert result["minimum_initial_calls"] == result["maximum_http_attempts"] == 0


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
    with pytest.raises(ArtifactError) as error:
        inventory(cache, datasets=["daily"])
    assert error.value.code == ("MISSING" if missing else "INTEGRITY_FAILED")


@pytest.mark.parametrize("condition,code", [
    ("store_symlink", "UNSAFE_PATH"), ("object_symlink", "UNSAFE_PATH"),
    ("missing_parent", "MISSING"), ("missing_store", "MISSING"),
])
def test_real_object_path_failures_preserve_exact_codes(cache, condition, code):
    conn, store, root = cache
    unit = inventory(datasets=["daily"])["units"][0]
    record = seed(cache, unit["request"])
    assert inventory(cache, datasets=["daily"])["units"] == [
        unit | {"raw_receipt_id": record.raw_receipt_id}]
    before = tuple(conn.iterdump())
    digest = record.raw_hash.removeprefix("sha256:")
    path = store.root / "objects" / digest[:2] / digest
    if condition in ("store_symlink", "missing_store"):
        saved = root / "saved-store"
        store.root.rename(saved)
        if condition == "store_symlink":
            store.root.symlink_to(saved, target_is_directory=True)
    elif condition == "object_symlink":
        saved = root / "saved-object"
        path.rename(saved)
        path.symlink_to(saved)
    else:
        path.unlink()
        path.parent.rmdir()
    with pytest.raises(ArtifactError) as error:
        inventory(cache, datasets=["daily"])
    assert error.value.code == code
    assert tuple(conn.iterdump()) == before
    if condition == "missing_store":
        assert not store.root.exists()


@pytest.mark.parametrize("kind", [None, "partial"])
def test_missing_store_is_not_required_without_a_complete_receipt(cache, kind):
    missing = inventory(datasets=["daily"])
    if kind:
        seed(cache, missing["units"][0]["request"], kind=kind)
        cache[1].root.rename(cache[2] / "saved-store")
    assert not cache[1].root.exists()
    assert inventory(cache, datasets=["daily"]) == missing
    assert not cache[1].root.exists()


def test_uninitialized_catalog_refuses_without_retry_or_storage_access():
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(OpsError) as error:
            massive_plan.plan(conn, None, **PARAMS)
        assert error.value.code == "VALIDATION_FAILED"
        assert error.value.problem.category == "validation"
        assert error.value.problem.retryable is False
        assert error.value.problem.details == {}
        assert conn.execute("SELECT name FROM sqlite_master").fetchall() == []
    finally:
        conn.close()


def test_selected_receipt_disappearing_before_real_load_is_input_changed(cache, monkeypatch):
    conn, store, _ = cache
    unit = inventory(datasets=["daily"])["units"][0]
    record = seed(cache, unit["request"])
    assert inventory(cache, datasets=["daily"])["units"] == [
        unit | {"raw_receipt_id": record.raw_receipt_id}]
    original_load = massive_plan.load_raw_receipt

    def remove_then_load(connection, objects, receipt_id):
        assert connection is conn and objects is store and receipt_id == record.raw_receipt_id
        connection.execute("DROP TRIGGER data_raw_receipts_no_delete")
        connection.execute("DELETE FROM data_raw_receipts WHERE raw_receipt_id=?", (receipt_id,))
        connection.commit()
        return original_load(connection, objects, receipt_id)

    monkeypatch.setattr(massive_plan, "load_raw_receipt", remove_then_load)
    with pytest.raises(DataError) as error:
        inventory(cache, datasets=["daily"])
    assert error.value.code == "INPUT_CHANGED"
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 0


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

    monkeypatch.setattr(cli, "sqlite3", SimpleNamespace(
        connect=guarded, Row=sqlite3.Row, Error=sqlite3.Error))
    # Coverage/mutation plugins use SQLite while this test's guard is active.
    assert sqlite3.connect is original
    with sqlite3.connect(":memory:") as unrelated:
        assert unrelated.execute("SELECT 1").fetchone() == (1,)
    assert cli._main(cli_args(root)) == 0
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


@pytest.mark.parametrize("transport", ["s3", "synthetic-unsupported-transport"])
def test_cli_unsupported_transport_refuses_before_connect(tmp_path, monkeypatch, capsys, transport):
    monkeypatch.setattr(cli, "sqlite3", SimpleNamespace(
        connect=lambda *a, **kw: pytest.fail("opened catalog"),
        Row=sqlite3.Row, Error=sqlite3.Error))
    assert cli._main(cli_args(tmp_path, transport=transport)) == 2
    assert json.loads(capsys.readouterr().out) == {
        "refused": "INVALID_REQUEST", "reason": "CAPABILITY_UNAVAILABLE"}


def test_cli_process_launch_failure_is_redacted(monkeypatch, capsys):
    def unavailable(*args, **kwargs):
        assert kwargs["timeout"] == 30
        raise OSError("synthetic private launch diagnostic")

    monkeypatch.setattr(cli, "subprocess", SimpleNamespace(
        run=unavailable, TimeoutExpired=subprocess.TimeoutExpired))
    assert main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == '{"refused": "RESOURCE_UNAVAILABLE"}\n'
    assert captured.err == ""


def test_request_hash_matches_existing_raw_receipt_contract(cache):
    unit = inventory(datasets=["daily"])["units"][0]
    record = seed(cache, unit["request"])
    assert unit["request_hash"] == record.request_hash == content_hash(unit["request"])
