"""The Tier-1 fetch wrapper.

The live network path cannot be exercised here — the ORATS monthly quota is
spent, and the guide forbids burning it on a test — so every guarantee is
asserted against a fake adapter that records exactly what the wrapper asked it
to do. What that leaves unverified is the real HTTP round trip for each source;
the first Sep-1 pull is where that seam gets its live check.
"""
from __future__ import annotations

import csv
import gzip
import json

import pytest

from engine.data.fetch import (
    FETCH_LOG_COLUMNS,
    Fetcher,
    cache_key,
    canonical_params,
)
from engine.data.sources.base import CredentialRotated, FetchError, Response, redact
from engine.data.throttle import QuotaExhausted, SourceConfig, Throttle


class FakeAdapter:
    """Records calls and replays a scripted sequence of responses."""

    name = "orats"

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls: list[tuple[str, dict]] = []
        self.default = Response(200, b'{"ok": true}', {}, "https://example/test?token=<redacted>")

    def request(self, endpoint, params, timeout):
        self.calls.append((endpoint, dict(params)))
        if self.responses:
            nxt = self.responses.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        return self.default

    def quota_from(self, response):
        value = response.headers.get("X-Monthly-Quota-Remaining")
        return int(value) if value is not None else None

    def is_auth_failure(self, response):
        return response.status == 401


@pytest.fixture
def fetcher(tmp_path):
    adapter = FakeAdapter()
    throttle = Throttle(
        {"orats": SourceConfig("orats", 0.0, 0.0, 3, 10.0, 20_000, 3_000)},
        sleep_fn=lambda s: None,
    )
    return Fetcher(tmp_path / "fetch", throttle=throttle, adapters={"orats": adapter}), adapter


class TestCacheKeys:
    def test_parameter_order_does_not_change_the_key(self):
        a = cache_key("orats", "hist/earnings", {"ticker": "AAPL", "dte": 14})
        b = cache_key("orats", "hist/earnings", {"dte": 14, "ticker": "AAPL"})
        assert a == b

    def test_equivalent_values_typed_differently_share_a_key(self):
        # The same request written by two call sites must not cost two pulls.
        assert canonical_params({"dte": 14}) == canonical_params({"dte": "14"})

    def test_none_valued_parameters_are_dropped(self):
        assert canonical_params({"a": 1, "b": None}) == canonical_params({"a": 1})

    def test_endpoint_slashes_are_normalized(self):
        assert cache_key("orats", "/hist/earnings/", {}) == cache_key("orats", "hist/earnings", {})

    def test_different_requests_get_different_keys(self):
        assert cache_key("orats", "hist/earnings", {"ticker": "AAPL"}) != cache_key(
            "orats", "hist/earnings", {"ticker": "MSFT"}
        )
        assert cache_key("orats", "hist/earnings", {}) != cache_key("orats", "hist/cores", {})

    def test_live_requests_get_one_entry_per_day(self):
        historical = cache_key("orats", "summaries", {"ticker": "AAPL"})
        live = cache_key("orats", "summaries", {"ticker": "AAPL"}, live=True)
        assert historical != live


class TestCacheFirst:
    def test_a_repeated_request_never_touches_the_network(self, fetcher):
        f, adapter = fetcher
        first = f.fetch("orats", "hist/earnings", {"ticker": "AAPL"})
        second = f.fetch("orats", "hist/earnings", {"ticker": "AAPL"})

        assert first.from_cache is False
        assert second.from_cache is True
        assert len(adapter.calls) == 1  # the whole point of Tier 1
        assert f.network_calls == 1
        assert second.body == first.body

    def test_only_the_network_call_is_logged(self, fetcher):
        f, _ = fetcher
        f.fetch("orats", "hist/earnings", {"ticker": "AAPL"})
        f.fetch("orats", "hist/earnings", {"ticker": "AAPL"})
        with open(f.fetch_log, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["from_cache"] == "0"

    def test_has_reports_cache_membership(self, fetcher):
        f, _ = fetcher
        assert not f.has("orats", "hist/earnings", {"ticker": "AAPL"})
        f.fetch("orats", "hist/earnings", {"ticker": "AAPL"})
        assert f.has("orats", "hist/earnings", {"ticker": "AAPL"})

    def test_different_params_are_separate_entries(self, fetcher):
        f, adapter = fetcher
        f.fetch("orats", "hist/earnings", {"ticker": "AAPL"})
        f.fetch("orats", "hist/earnings", {"ticker": "MSFT"})
        assert len(adapter.calls) == 2


class TestPersistence:
    def test_the_body_is_stored_verbatim_and_gzipped(self, fetcher):
        f, adapter = fetcher
        payload = b'{"rows": [1, 2, 3], "note": "exact bytes"}'
        adapter.responses = [Response(200, payload, {}, "https://x?token=<redacted>")]
        record = f.fetch("orats", "hist/strikes", {"tradeDate": "2024-05-01"})

        assert record.body == payload
        with gzip.open(record.path, "rb") as fh:
            assert fh.read() == payload

    def test_metadata_lands_beside_the_body(self, fetcher):
        f, _ = fetcher
        record = f.fetch("orats", "hist/earnings", {"ticker": "AAPL"}, note="smoke")
        meta = json.loads(f.meta_path("orats", record.key).read_text())
        assert meta["endpoint"] == "hist/earnings"
        assert meta["params"] == {"ticker": "AAPL"}
        assert meta["status"] == 200
        assert meta["note"] == "smoke"
        assert "fetched_at" in meta

    def test_the_record_parses_without_mutating_the_stored_bytes(self, fetcher):
        f, adapter = fetcher
        adapter.responses = [Response(200, b'{"a": 1}', {}, "u")]
        record = f.fetch("orats", "hist/earnings", {"ticker": "AAPL"})
        assert record.json() == {"a": 1}
        assert record.body == b'{"a": 1}'

    def test_a_corrupt_entry_fails_loudly_rather_than_silently_refetching(self, fetcher):
        f, _ = fetcher
        record = f.fetch("orats", "hist/earnings", {"ticker": "AAPL"})
        record.path.write_bytes(b"not gzip at all")
        with pytest.raises(FetchError, match="corrupt Tier-1 entry"):
            f.fetch("orats", "hist/earnings", {"ticker": "AAPL"})


class TestFailures:
    def test_a_failed_response_is_not_cached(self, fetcher):
        f, adapter = fetcher
        adapter.responses = [Response(404, b"nope", {}, "u")]
        with pytest.raises(FetchError, match="HTTP 404"):
            f.fetch("orats", "hist/earnings", {"ticker": "ZZZZ"})

        # Nothing poisoned: a later retry is a clean miss, not a bad hit.
        assert not f.has("orats", "hist/earnings", {"ticker": "ZZZZ"})
        adapter.responses = [Response(200, b"{}", {}, "u")]
        assert f.fetch("orats", "hist/earnings", {"ticker": "ZZZZ"}).status == 200

    def test_a_failure_is_still_logged(self, fetcher):
        f, adapter = fetcher
        adapter.responses = [Response(404, b"nope", {}, "u")]
        with pytest.raises(FetchError):
            f.fetch("orats", "hist/earnings", {"ticker": "ZZZZ"})
        with open(f.fetch_log, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["status"] == "404"

    def test_client_errors_are_not_retried(self, fetcher):
        f, adapter = fetcher
        adapter.responses = [Response(400, b"bad", {}, "u"), Response(200, b"{}", {}, "u")]
        with pytest.raises(FetchError, match="HTTP 400"):
            f.fetch("orats", "hist/earnings", {})
        assert len(adapter.calls) == 1  # a 4xx will not fix itself

    def test_rate_limits_and_server_errors_are_retried(self, fetcher):
        f, adapter = fetcher
        adapter.responses = [
            Response(429, b"slow down", {}, "u"),
            Response(502, b"gateway", {}, "u"),
            Response(200, b'{"ok":1}', {}, "u"),
        ]
        record = f.fetch("orats", "hist/cores", {"ticker": "AAPL"})
        assert record.status == 200
        assert len(adapter.calls) == 3

    def test_retries_are_bounded(self, fetcher):
        f, adapter = fetcher
        adapter.responses = [Response(500, b"x", {}, "u")] * 10
        with pytest.raises(FetchError, match="after 3 attempts"):
            f.fetch("orats", "hist/cores", {"ticker": "AAPL"})
        assert len(adapter.calls) == 3

    def test_transport_errors_are_retried_then_surfaced(self, fetcher):
        f, adapter = fetcher
        adapter.responses = [ConnectionError("reset"), Response(200, b"{}", {}, "u")]
        assert f.fetch("orats", "hist/cores", {}).status == 200

    def test_a_rotated_credential_stops_immediately(self, fetcher):
        # A rotated key produces an unbounded stream of identical failures and
        # the only fix is a human updating .env — so it must never retry-loop.
        f, adapter = fetcher
        adapter.responses = [Response(401, b"Unknown API Key", {}, "u")] * 5
        with pytest.raises(CredentialRotated):
            f.fetch("orats", "hist/earnings", {"ticker": "AAPL"})
        assert len(adapter.calls) == 1


class TestQuota:
    def test_quota_headers_are_recorded(self, fetcher):
        f, adapter = fetcher
        adapter.responses = [
            Response(200, b"{}", {"X-Monthly-Quota-Remaining": "11000"}, "u")
        ]
        f.fetch("orats", "hist/earnings", {"ticker": "AAPL"})
        with open(f.quota_log, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[-1]["quota_remaining"] == "11000"

    def test_the_guard_refuses_the_next_call_once_inside_the_reserve(self, fetcher):
        f, adapter = fetcher
        adapter.responses = [
            Response(200, b"{}", {"X-Monthly-Quota-Remaining": "2500"}, "u"),
            Response(200, b"{}", {}, "u"),
        ]
        f.fetch("orats", "hist/earnings", {"ticker": "AAPL"})
        with pytest.raises(QuotaExhausted, match="reserve"):
            f.fetch("orats", "hist/earnings", {"ticker": "MSFT"})

    def test_the_guard_reads_the_quota_log_in_a_fresh_process(self, tmp_path):
        root = tmp_path / "fetch"
        throttle = Throttle(
            {"orats": SourceConfig("orats", 0.0, 0.0, 3, 10.0, 20_000, 3_000)},
            sleep_fn=lambda s: None,
        )
        first = Fetcher(root, throttle=throttle, adapters={"orats": FakeAdapter(
            [Response(200, b"{}", {"X-Monthly-Quota-Remaining": "100"}, "u")]
        )})
        first.fetch("orats", "hist/earnings", {"ticker": "AAPL"})

        # A new process must not start spending again from zero knowledge.
        second = Fetcher(root, throttle=throttle, adapters={"orats": FakeAdapter()})
        with pytest.raises(QuotaExhausted):
            second.fetch("orats", "hist/earnings", {"ticker": "MSFT"})


class TestLogging:
    def test_the_log_has_the_declared_columns(self, fetcher):
        f, _ = fetcher
        f.fetch("orats", "hist/earnings", {"ticker": "AAPL"})
        with open(f.fetch_log, newline="") as fh:
            assert next(csv.reader(fh)) == list(FETCH_LOG_COLUMNS)

    def test_no_secret_reaches_the_log_or_the_metadata(self, fetcher):
        f, adapter = fetcher
        adapter.responses = [
            Response(200, b"{}", {}, redact("https://api.orats.io/x?token=SUPERSECRET123"))
        ]
        record = f.fetch("orats", "hist/earnings", {"ticker": "AAPL"})
        assert b"SUPERSECRET123" not in f.fetch_log.read_bytes()
        assert "SUPERSECRET123" not in f.meta_path("orats", record.key).read_text()


class TestRedaction:
    @pytest.mark.parametrize("param", ["token", "apiKey", "api_key", "key", "access_token"])
    def test_secret_query_parameters_are_redacted(self, param):
        out = redact(f"https://example.com/x?{param}=abc123&ticker=AAPL")
        assert "abc123" not in out
        assert "ticker=AAPL" in out

    def test_redaction_leaves_ordinary_parameters_alone(self):
        assert redact("https://x/y?ticker=AAPL&dte=14") == "https://x/y?ticker=AAPL&dte=14"


class TestQuotaMonthBoundary:
    """The quota resets on the 1st; a stale reading must not lock out the month."""

    def _fetcher(self, root):
        throttle = Throttle(
            {"orats": SourceConfig("orats", 0.0, 0.0, 3, 10.0, 20_000, 3_000)},
            sleep_fn=lambda s: None,
        )
        return Fetcher(root, throttle=throttle, adapters={"orats": FakeAdapter()})

    def _write_quota_log(self, fetcher, ts, remaining):
        import csv as _csv

        path = fetcher.quota_log
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            writer = _csv.DictWriter(
                fh, fieldnames=["ts", "source", "endpoint", "key", "status",
                                "quota_remaining", "elapsed_s"]
            )
            writer.writeheader()
            writer.writerow(
                {"ts": ts, "source": "orats", "endpoint": "hist/strikes",
                 "key": "k", "status": 200, "quota_remaining": remaining,
                 "elapsed_s": 0.1}
            )

    def test_a_sub_reserve_reading_from_a_previous_month_is_ignored(self, tmp_path):
        # September correctly stops at the reserve floor, leaving a sub-reserve
        # reading on disk. On October 1 the guard runs BEFORE the first call
        # that would refresh the header. Replaying September's number would
        # block the whole month.
        fetcher = self._fetcher(tmp_path / "fetch")
        self._write_quota_log(fetcher, "2020-01-15T00:00:00+00:00", 100)
        assert fetcher._last_known_quota("orats") is None
        fetcher.fetch("orats", "hist/strikes", {"tradeDate": "x"})  # must not raise

    def test_a_sub_reserve_reading_from_this_month_still_guards(self, tmp_path):
        from datetime import datetime, timezone

        fetcher = self._fetcher(tmp_path / "fetch")
        now = datetime.now(timezone.utc).isoformat()
        self._write_quota_log(fetcher, now, 100)
        assert fetcher._last_known_quota("orats") == 100
        with pytest.raises(QuotaExhausted):
            fetcher.fetch("orats", "hist/strikes", {"tradeDate": "y"})

    def test_an_ample_reading_from_this_month_is_used(self, tmp_path):
        from datetime import datetime, timezone

        fetcher = self._fetcher(tmp_path / "fetch")
        self._write_quota_log(fetcher, datetime.now(timezone.utc).isoformat(), 15_000)
        assert fetcher._last_known_quota("orats") == 15_000
