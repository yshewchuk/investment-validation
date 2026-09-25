"""S4C: the native Nasdaq forward-calendar fetcher.

No network: every test injects ``http_get``. The URL builder and the
``data.rows`` parser are ported from ``engine/data/sources/nasdaq.py`` and the
legacy pull's own ``_nasdaq_rows``; only the HTTP edge is faked.
"""
from __future__ import annotations

import json

from engine.v2.ops.providers import PROVIDER_CREDENTIAL_VARIABLES, provider_credentials
from engine.v2.ops.providers.nasdaq_calendar import (
    BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
    SESSION_BY_TIME,
    nasdaq_calendar_fetcher,
)

DAY = "2026-05-01"
UNIT = {"request_id": "nasdaq:calendar/earnings:" + DAY, "table_name": "earnings_events",
        "partition_key": DAY, "expected_keys": [DAY]}
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
    assert kind == "empty"
    assert rows == []


def test_a_malformed_body_is_an_empty_row_list():
    fetcher = nasdaq_calendar_fetcher(http_get=_FakeHttp(body=b"not-json"))
    _raw, kind, _meta, rows = fetcher(dict(UNIT))
    assert kind == "empty"
    assert rows == []


def test_a_404_is_unsupported_not_complete():
    fetcher = nasdaq_calendar_fetcher(http_get=_FakeHttp(status=404, body=b"gone"))
    _raw, kind, _meta, rows = fetcher(dict(UNIT))
    assert kind == "unsupported"
    assert rows == []


def test_the_unmetered_account_needs_no_credentials(monkeypatch):
    monkeypatch.setenv("ORATS_API_KEY", "unused")
    assert PROVIDER_CREDENTIAL_VARIABLES["nasdaq"] == ()
    assert provider_credentials({"provider_account": "nasdaq"}) == {}
    # constructing the fetcher touches nothing
    assert callable(nasdaq_calendar_fetcher())
