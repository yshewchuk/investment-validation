"""S4C: the native Nasdaq forward-calendar fetcher.

No network: every test injects ``http_get``. The URL builder and the
``data.rows`` parser are ported from ``engine/data/sources/nasdaq.py`` and the
legacy pull's own ``_nasdaq_rows``; only the HTTP edge is faked. Classification
(spec R1) is the provider's own job, so every kind is asserted here.
"""
from __future__ import annotations

import json

import pytest

from engine.v2.ops.providers import PROVIDER_CREDENTIAL_VARIABLES, provider_credentials
from engine.v2.ops.providers.nasdaq_calendar import (
    BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
    SESSION_BY_TIME,
    nasdaq_calendar_fetcher,
)

DAY = "2026-05-01"
UNIT = {"request_id": "nasdaq:calendar/earnings:" + DAY + ":2026-04-30",
        "table_name": "earnings_events", "partition_key": DAY, "expected_keys": [DAY]}
ROW = {"symbol": "AAA", "time": "time-after-hours", "name": "Agilent"}


class _FakeHttp:
    """One canned response; records every ``(url, timeout)`` call."""

    def __init__(self, status=200, body=None):
        self.status = status
        self.body = body if body is not None else json.dumps(
            {"data": {"rows": [ROW]}, "message": "ok"}).encode()
        self.calls = []

    def __call__(self, url, *, timeout):
        self.calls.append((url, timeout))
        return self.status, {}, self.body


def test_rows_are_parsed_and_the_url_is_the_ported_endpoint():
    fake = _FakeHttp()
    fetcher = nasdaq_calendar_fetcher(http_get=fake)

    raw, kind, meta, rows = fetcher(dict(UNIT))

    assert raw == fake.body
    assert kind == "complete"
    assert meta == {"status": 200, "date": DAY}
    assert rows == [ROW]
    assert fake.calls == [
        (f"{BASE_URL}/calendar/earnings?date={DAY}", REQUEST_TIMEOUT_SECONDS)]
    assert set(SESSION_BY_TIME) == {"time-pre-market", "time-after-hours"}


def test_an_empty_200_is_a_legitimate_empty_not_a_complete_claim():
    fetcher = nasdaq_calendar_fetcher(
        http_get=_FakeHttp(body=json.dumps({"data": {"rows": []}}).encode()))
    _raw, kind, _meta, rows = fetcher(dict(UNIT))
    assert kind == "legitimate_empty"
    assert rows == []


def test_a_malformed_body_is_refused_not_empty():
    """Unparseable bytes are refused, never mistaken for a legitimate empty."""
    fetcher = nasdaq_calendar_fetcher(http_get=_FakeHttp(body=b"not-json"))
    _raw, kind, _meta, rows = fetcher(dict(UNIT))
    assert kind == "refused"
    assert rows == []


def test_a_2xx_without_the_documented_shape_is_refused():
    fetcher = nasdaq_calendar_fetcher(http_get=_FakeHttp(body=b"{}"))
    _raw, kind, _meta, _rows = fetcher(dict(UNIT))
    assert kind == "refused"


def test_a_404_is_not_final_not_complete():
    fetcher = nasdaq_calendar_fetcher(http_get=_FakeHttp(status=404, body=b"gone"))
    _raw, kind, _meta, rows = fetcher(dict(UNIT))
    assert kind == "not_final"
    assert rows == []


def test_a_403_is_credential_invalid_not_a_generic_refusal():
    fetcher = nasdaq_calendar_fetcher(http_get=_FakeHttp(status=403, body=b"forbidden"))
    _raw, kind, _meta, _rows = fetcher(dict(UNIT))
    assert kind == "credential_invalid"


def test_a_400_is_refused_bad_source_data_not_a_credential_failure():
    fetcher = nasdaq_calendar_fetcher(http_get=_FakeHttp(status=400, body=b"bad request"))
    _raw, kind, _meta, _rows = fetcher(dict(UNIT))
    assert kind == "refused"


def test_a_5xx_is_transient():
    fetcher = nasdaq_calendar_fetcher(http_get=_FakeHttp(status=503, body=b"down"))
    _raw, kind, _meta, _rows = fetcher(dict(UNIT))
    assert kind == "transient"


def test_a_network_error_is_transient_not_an_empty_row_list():
    def boom(url, *, timeout):
        raise TimeoutError("timed out")

    fetcher = nasdaq_calendar_fetcher(http_get=boom)
    raw, kind, meta, rows = fetcher(dict(UNIT))
    assert (raw, kind, rows) == (b"", "transient", [])
    assert meta["error"] == "TimeoutError"


def test_a_programming_error_out_of_http_get_propagates():
    def boom(url, *, timeout):
        raise TypeError("bad call signature")

    fetcher = nasdaq_calendar_fetcher(http_get=boom)
    with pytest.raises(TypeError):
        fetcher(dict(UNIT))


def test_the_unmetered_account_needs_no_credentials(monkeypatch):
    monkeypatch.setenv("ORATS_API_KEY", "unused")
    assert PROVIDER_CREDENTIAL_VARIABLES["nasdaq"] == ()
    assert provider_credentials({"provider_account": "nasdaq"}) == {}
    # constructing the fetcher touches nothing
    assert callable(nasdaq_calendar_fetcher())
