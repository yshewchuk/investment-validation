"""Synthetic page boundaries only: no provider, credentials, or private data."""
import json
import socket
from datetime import datetime, timezone
from urllib.parse import urlencode

import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.ops.providers.massive_policy import MAX_RAW_BYTES
from engine.v2.ops.stores.massive.pages import closed_day, validate_page, validate_rows

REQUEST = {
    "schema_version": "massive_request.v1", "source": "massive-stocks", "transport": "rest",
    "endpoint": "/v2/aggs/ticker/SYNTH/range/1/day/2020-01-01/2020-01-10",
    "dataset": "daily", "ticker": "SYNTH", "start": "2020-01-01", "end": "2020-01-10",
    "adjustment_basis": "raw", "params": {"adjusted": False, "sort": "asc", "limit": 50000},
}
ROW = {"t": 1577941200000, "o": 10, "h": 12, "l": 9, "c": 11, "v": 100}
DOCUMENT = {"status": "OK", "ticker": "SYNTH", "adjusted": False,
            "resultsCount": 1, "results": [ROW]}
CURSOR = "bGltaXQ9MiZzb3J0PWFzYw"


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("page validation attempted network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def body(**changes):
    return json.dumps(DOCUMENT | changes).encode()


def next_url(endpoint=REQUEST["endpoint"], **params):
    return "https://api.massive.com" + endpoint + "?" + urlencode({"cursor": CURSOR} | params)


def refusal(payload, request=REQUEST, page=None, code="SOURCE_INVALID"):
    with pytest.raises(OpsError) as error:
        validate_page(payload, request, page or request)
    assert error.value.code == code
    assert "synthetic-secret" not in str(error.value)
    assert "api.massive.com" not in str(error.value)


def corporate(dataset):
    field = "execution_date" if dataset == "splits" else "ex_dividend_date"
    request = REQUEST | {
        "dataset": dataset, "endpoint": "/stocks/v1/" + dataset,
        "adjustment_basis": "corporate_action",
        "params": {"ticker": "SYNTH", field + ".gte": REQUEST["start"],
                   field + ".lte": REQUEST["end"], "sort": field + ".asc", "limit": 5000},
    }
    row = {"ticker": "SYNTH", field: "2020-01-02"}
    return request, row, field


def test_terminal_envelope_and_rows_use_shared_validation_without_normalizing():
    payload = body(ticker="synth")
    assert validate_page(payload, REQUEST, REQUEST) is None
    assert validate_rows(payload, REQUEST, REQUEST) == DOCUMENT | {"ticker": "synth"}
    assert validate_page(body(results=[{"t": ROW["t"]}]), REQUEST, REQUEST) is None
    adjusted = REQUEST | {"adjustment_basis": "split_adjusted",
                          "params": REQUEST["params"] | {"adjusted": True}}
    assert validate_page(body(adjusted=True), adjusted, adjusted) is None


@pytest.mark.parametrize("change", [
    {"status": "ERROR"}, {"status": True}, {"ticker": "OTHER"}, {"ticker": None},
    {"adjusted": True}, {"adjusted": 0}, {"resultsCount": True}, {"resultsCount": 1.0},
    {"resultsCount": 0}, {"results": {}}, {"results": [None]}, {"results": [False]},
])
def test_envelope_types_and_identity_are_checked(change):
    refusal(body(**change))


@pytest.mark.parametrize("payload", [b"null", b"[]", b"1", b"{", b"\xff", "{}", None, bytearray(b"{}")])
def test_malformed_body_is_sanitized(payload):
    refusal(payload)


def test_raw_size_bound_is_shared_and_checked_before_json():
    payload = body()
    assert validate_page(payload + b" " * (MAX_RAW_BYTES - len(payload)), REQUEST, REQUEST) is None
    refusal(b" " * (MAX_RAW_BYTES + 1), code="RESOURCE_LIMIT_EXCEEDED")


def test_duplicate_json_keys_cannot_hide_secret_bearing_continuation():
    hidden = ', "next_url": "https://api.massive.com/?apiKey=synthetic-secret", "next_url": null}'
    refusal(body()[:-1] + hidden.encode())
    escaped = hidden.replace('"next_url": null', '"\\u006eext_url": null')
    refusal(body()[:-1] + escaped.encode())
    refusal(body().replace(b'"t": 1577941200000', b'"t": false, "t": 1577941200000'))


@pytest.mark.parametrize("continuation", [None, next_url()])
def test_empty_page_never_establishes_coverage(continuation):
    refusal(body(results=[], resultsCount=0, next_url=continuation), code="SOURCE_EMPTY")


@pytest.mark.parametrize("timestamp", [True, False, 1577941200000.0, "1577941200000", None,
                                       -10**30, 10**30, 1577854799999, 1578718800000])
def test_daily_timestamp_requires_integer_inside_eastern_dates(timestamp):
    refusal(body(results=[ROW | {"t": timestamp}]))


def test_missing_daily_timestamp_refuses():
    refusal(body(results=[{"o": 10}]))


@pytest.mark.parametrize("timestamp", [1577854800000, 1578718799999])
def test_daily_date_range_is_inclusive_at_eastern_boundaries(timestamp):
    assert validate_page(body(results=[ROW | {"t": timestamp}]), REQUEST, REQUEST) is None


@pytest.mark.parametrize("day,start_utc,end_utc", [
    ("2020-03-08", "2020-03-08T05:00:00+00:00", "2020-03-09T04:00:00+00:00"),
    ("2020-11-01", "2020-11-01T04:00:00+00:00", "2020-11-02T05:00:00+00:00"),
])
def test_daily_date_scope_handles_both_dst_transitions(day, start_utc, end_utc):
    request = REQUEST | {"start": day, "end": day,
                         "endpoint": f"/v2/aggs/ticker/SYNTH/range/1/day/{day}/{day}"}
    first = int(datetime.fromisoformat(start_utc).timestamp()) * 1000
    after = int(datetime.fromisoformat(end_utc).timestamp()) * 1000
    for instant in (first, after - 1):
        assert validate_page(body(results=[ROW | {"t": instant}]), request, request) is None
    for instant in (first - 1, after):
        refusal(body(results=[ROW | {"t": instant}]), request)


@pytest.mark.parametrize("dataset", ["splits", "dividends"])
def test_corporate_row_scope_and_initial_filters_are_preserved(dataset):
    request, row, _ = corporate(dataset)
    payload = body(results=[row | {"ticker": "synth"}], next_url=next_url(request["endpoint"]))
    result = validate_page(payload, request, request)
    assert result == request | {"params": request["params"] | {"cursor": CURSOR}}
    assert "cursor" not in request["params"]


@pytest.mark.parametrize("dataset", ["splits", "dividends"])
@pytest.mark.parametrize("change", ["ticker", "missing_ticker", "date", "compact_date", "date_time",
                                    "bool_date", "missing_date", "early_date", "late_date"])
def test_corporate_row_ticker_and_canonical_date_are_required(dataset, change):
    request, row, field = corporate(dataset)
    if change == "ticker":
        row["ticker"] = "OTHER"
    elif change == "missing_ticker":
        del row["ticker"]
    elif change == "missing_date":
        del row[field]
    else:
        row[field] = {"date": "synthetic-secret", "compact_date": "20200102",
                      "date_time": "2020-01-02T00:00:00", "bool_date": True,
                      "early_date": "2019-12-31", "late_date": "2020-01-11"}[change]
    refusal(body(results=[row]), request)


@pytest.mark.parametrize("dataset", ["splits", "dividends"])
@pytest.mark.parametrize("ticker", ["ſYNTH", "SYNTH ", " SYNTH", "SYNTH/", True, None])
def test_corporate_ticker_is_strict_ascii(dataset, ticker):
    request, row, _ = corporate(dataset)
    refusal(body(results=[row | {"ticker": ticker}]), request)


def test_official_millisecond_continuation_shape_preserves_original_scope():
    request = REQUEST | {"ticker": "AAPL", "endpoint": REQUEST["endpoint"].replace("SYNTH", "AAPL")}
    endpoint = "/v2/aggs/ticker/AAPL/range/1/day/1578114000000/2020-01-10"
    payload = body(ticker="AAPL", next_url=next_url(endpoint))
    result = validate_page(payload, request, request)
    assert result == request | {"endpoint": endpoint,
                               "params": request["params"] | {"cursor": CURSOR}}
    assert result["start"] == "2020-01-01"
    assert result["params"]["adjusted"] is False
    assert validate_page(body(ticker="AAPL", results=[ROW | {"t": 1578114000000}]), request, result) is None
    refusal(body(ticker="AAPL", results=[ROW]), request, result)


def test_canonical_iso_continuation_and_exact_millisecond_lower_bound():
    endpoint = REQUEST["endpoint"].replace("2020-01-01", "2020-01-02")
    result = validate_page(body(next_url=next_url(endpoint)), REQUEST, REQUEST)
    assert validate_page(body(), REQUEST, result) is None
    millisecond = endpoint.replace("2020-01-02", str(ROW["t"] + 1))
    result = validate_page(body(next_url=next_url(millisecond)), REQUEST, REQUEST)
    refusal(body(), REQUEST, result)


def test_millisecond_page_lower_bound_compares_instants_across_fall_dst_fold():
    request = REQUEST | {"start": "2020-11-01", "end": "2020-11-01",
                         "endpoint": "/v2/aggs/ticker/SYNTH/range/1/day/2020-11-01/2020-11-01"}
    # 01:30 EDT, then 01:15 EST: the latter is chronologically later.
    lower = 1604208600000
    page = request | {"endpoint": request["endpoint"].replace("/2020-11-01/", f"/{lower}/"),
                      "params": request["params"] | {"cursor": CURSOR}}
    assert validate_page(body(results=[ROW | {"t": 1604211300000}]), request, page) is None
    refusal(body(results=[ROW | {"t": lower - 1}]), request, page)


@pytest.mark.parametrize("url", [
    "http://api.massive.com", "https://other.example", "https://api.massive.com.evil.example",
    "https://synthetic-secret@api.massive.com", "https://api.massive.com:443",
    "https://API.massive.com", "https://api.massive.com.", "//api.massive.com",
])
def test_continuation_authority_is_exact_https(url):
    refusal(body(next_url=url + REQUEST["endpoint"] + "?cursor=" + CURSOR))


@pytest.mark.parametrize("suffix", [
    "#", "#synthetic-secret", "&apiKey=synthetic-secret", "&token=synthetic-secret",
    "&unknown=value", "&cursor=another", "&adjusted=true", "&sort=desc", "&limit=1",
    "&api%4bey=synthetic-secret", "\n", "\t", " ",
])
def test_continuation_credentials_fragments_unknown_and_conflicting_params_refuse(suffix):
    refusal(body(next_url=next_url() + suffix))


@pytest.mark.parametrize("cursor", ["", "synthetic-secret value", "a/b", "a+b", "a=b", "a===", "é", "a" * 2049])
def test_continuation_cursor_is_bounded_ascii_urlsafe(cursor):
    refusal(body(next_url=next_url(cursor=cursor)))


@pytest.mark.parametrize("cursor", ["abc_-123", "ab==", "abc=", "a" * 2048])
def test_supported_cursor_syntax(cursor):
    result = validate_page(body(next_url=next_url(cursor=cursor)), REQUEST, REQUEST)
    assert result["params"]["cursor"] == cursor


@pytest.mark.parametrize("url", [False, 123, {}, [], "", "https://api.massive.com/" + "a" * 4096])
def test_continuation_type_and_total_size_are_bounded(url):
    refusal(body(next_url=url))


@pytest.mark.parametrize("endpoint", [
    REQUEST["endpoint"].replace("SYNTH", "OTHER"),
    REQUEST["endpoint"].replace("/day/", "/minute/"),
    REQUEST["endpoint"].replace("2020-01-10", "2020-01-11"),
    REQUEST["endpoint"].replace("2020-01-01", "2019-12-31"),
    REQUEST["endpoint"].replace("2020-01-01", "20200101"),
    REQUEST["endpoint"].replace("2020-01-01", "01577854800000"),
    REQUEST["endpoint"].replace("2020-01-01", "1577854799999"),
    REQUEST["endpoint"].replace("2020-01-01", "9999999999999999"),
    REQUEST["endpoint"].replace("SYNTH", "%53YNTH"),
    REQUEST["endpoint"] + "/", "/stocks/v1/splits",
])
def test_continuation_endpoint_remains_canonical_and_within_original_scope(endpoint):
    refusal(body(next_url=next_url(endpoint)))


@pytest.mark.parametrize("dataset", ["splits", "dividends"])
def test_corporate_endpoint_cannot_change(dataset):
    request, row, _ = corporate(dataset)
    refusal(body(results=[row], next_url=next_url("/stocks/v1/other")), request)


def test_matching_original_params_are_allowed_but_never_replace_typed_params():
    result = validate_page(body(next_url=next_url(adjusted="false", sort="asc", limit="50000")),
                           REQUEST, REQUEST)
    assert result["params"] == REQUEST["params"] | {"cursor": CURSOR}
    request, row, _ = corporate("splits")
    url = next_url(request["endpoint"], **request["params"])
    assert validate_page(body(results=[row], next_url=url), request, request)["params"] == (
        request["params"] | {"cursor": CURSOR})


def test_missing_duplicate_or_malformed_query_is_refused():
    for query in ("", "limit=50000", "cursor", "cursor=" + CURSOR + "&", "cursor=x&" * 11):
        refusal(body(next_url="https://api.massive.com" + REQUEST["endpoint"] + "?" + query))


def test_immediate_descriptor_cycle_and_backwards_lower_bound_are_refused():
    payload = body(next_url=next_url())
    page = validate_page(payload, REQUEST, REQUEST)
    refusal(payload, REQUEST, page)
    endpoint = REQUEST["endpoint"].replace("2020-01-01", "2020-01-02")
    page = REQUEST | {"endpoint": endpoint, "params": REQUEST["params"] | {"cursor": CURSOR}}
    refusal(body(next_url=next_url(cursor="different")), REQUEST, page)


@pytest.mark.parametrize("dataset", ["daily", "splits", "dividends"])
@pytest.mark.parametrize("change", ["ticker", "start", "end", "adjustment_basis", "source",
                                    "endpoint", "params", "cursor", "limit_type", "extra"])
def test_actual_page_descriptor_must_preserve_the_original_scope(dataset, change):
    request, row = (REQUEST, ROW) if dataset == "daily" else corporate(dataset)[:2]
    page = request | {"params": request["params"] | {"cursor": CURSOR}}
    if change == "params":
        page["params"] = page["params"] | {"apiKey": "synthetic-secret"}
    elif change == "cursor":
        page["params"] = page["params"] | {"cursor": "a/b"}
    elif change == "limit_type":
        page["params"] = page["params"] | {"limit": float(request["params"]["limit"])}
    else:
        page[change] = "synthetic-secret"
    refusal(body(results=[row]), request, page)


@pytest.mark.parametrize("params", [None, [], False, {"cursor": CURSOR}])
def test_actual_page_descriptor_malformed_params_are_sanitized(params):
    refusal(body(), page=REQUEST | {"params": params})


def test_changed_daily_start_requires_a_checked_cursor_descriptor():
    page = REQUEST | {"endpoint": REQUEST["endpoint"].replace("2020-01-01", "2020-01-02")}
    refusal(body(), page=page)
    refusal(body(), page=REQUEST | {"params": REQUEST["params"] | {"adjusted": 0}})


@pytest.mark.parametrize("continuation", [None, next_url(cursor="another")])
def test_empty_later_page_never_establishes_coverage(continuation):
    page = validate_page(body(next_url=next_url()), REQUEST, REQUEST)
    refusal(body(results=[], resultsCount=0, next_url=continuation), page=page, code="SOURCE_EMPTY")


@pytest.mark.parametrize("observed,refused", [
    ("2020-01-11T04:59:59+00:00", True), ("2020-01-11T05:00:00+00:00", False),
    ("2020-01-10T23:59:59-05:00", True), ("2020-01-11T00:00:00-05:00", False),
    ("2020-01-11T00:00:00", True),
])
def test_closed_day_uses_eastern_observation_date(observed, refused):
    observed_at = datetime.fromisoformat(observed)
    if refused:
        with pytest.raises(OpsError) as error:
            closed_day(REQUEST, observed_at)
        assert error.value.code == "SOURCE_NOT_FINAL"
    else:
        assert closed_day(REQUEST, observed_at) is None


@pytest.mark.parametrize("end,before,after", [
    ("2020-03-08", "2020-03-09T03:59:59+00:00", "2020-03-09T04:00:00+00:00"),
    ("2020-11-01", "2020-11-02T04:59:59+00:00", "2020-11-02T05:00:00+00:00"),
])
def test_closed_day_handles_dst_without_fixed_offset(end, before, after):
    request = REQUEST | {"end": end}
    with pytest.raises(OpsError) as error:
        closed_day(request, datetime.fromisoformat(before))
    assert error.value.code == "SOURCE_NOT_FINAL"
    assert closed_day(request, datetime.fromisoformat(after)) is None


def test_closed_day_is_only_daily_local_eligibility():
    request, _, _ = corporate("splits")
    assert closed_day(request, datetime(2020, 1, 10, tzinfo=timezone.utc)) is None
