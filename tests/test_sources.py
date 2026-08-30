"""Source adapters: auth handling, URL construction, and the standing bans.

Every adapter is pure enough to test without a network. What makes these worth
testing is not the URL string-building — it is that three of the program's
standing rules are enforced *here* and nowhere else:

* the ORATS token travels in the URL, so redaction is what keeps it out of the
  Tier-1 metadata written next to every cached body;
* oquants backtest endpoints return model-fitted marks that are banned from P&L;
* known-bad ORATS paths must fail fast instead of burning retries.

A guard with no test is a guard you do not have.
"""
from __future__ import annotations

import pytest

from engine.data.sources import get_adapter
from engine.data.sources.base import CredentialRotated, Response, redact
from engine.data.sources.oquants import PNL_BANNED_ENDPOINTS, OquantsAdapter
from engine.data.sources.orats import KNOWN_BAD_PATHS, OratsAdapter, OratsResponse
from engine.data.sources.polygon import PolygonAdapter, option_ticker
from engine.data.sources.yf import YFinanceAdapter

SECRET = "orats-secret-key-abcdef123456"


class TestOratsUrls:
    @pytest.fixture
    def adapter(self):
        return OratsAdapter(api_key=SECRET)

    def test_auth_is_a_query_param_not_a_header(self, adapter):
        # ORATS is the odd one out: the token is in the URL, which is why the
        # URL itself is a secret.
        url = adapter.build_url("hist/earnings", {"ticker": "AAPL"})
        assert f"token={SECRET}" in url
        assert url.startswith("https://api.orats.io/datav2/hist/earnings?")

    def test_params_are_sorted_so_the_url_is_deterministic(self, adapter):
        a = adapter.build_url("hist/strikes", {"tradeDate": "2024-05-01", "dte": "1,45"})
        b = adapter.build_url("hist/strikes", {"dte": "1,45", "tradeDate": "2024-05-01"})
        assert a == b

    def test_none_valued_params_are_dropped(self, adapter):
        assert "expiry" not in adapter.build_url("hist/strikes", {"expiry": None})

    def test_leading_and_trailing_slashes_are_normalized(self, adapter):
        assert adapter.build_url("/hist/earnings/", {}) == adapter.build_url("hist/earnings", {})

    @pytest.mark.parametrize("path", sorted(KNOWN_BAD_PATHS))
    def test_known_bad_paths_fail_fast(self, adapter, path):
        # These 403 because they are not real endpoint names. Retrying them
        # only spends time, so they must raise before any request is made.
        with pytest.raises(ValueError, match="known-bad"):
            adapter.build_url(path, {})

    def test_a_missing_key_raises_rather_than_calling_anonymously(self):
        with pytest.raises(CredentialRotated, match="ORATS_API_KEY"):
            OratsAdapter(api_key="").request("hist/earnings", {}, 10.0)


class TestOratsRedaction:
    def test_the_token_never_survives_redaction(self):
        adapter = OratsAdapter(api_key=SECRET)
        url = adapter.build_url("hist/earnings", {"ticker": "AAPL"})
        cleaned = redact(url)
        assert SECRET not in cleaned
        assert "token=<redacted>" in cleaned
        assert "ticker=AAPL" in cleaned  # the rest of the request stays legible


class TestOratsQuota:
    def _response(self, headers):
        return OratsResponse(200, b"{}", headers, "u")

    def test_quota_remaining_is_parsed(self):
        adapter = OratsAdapter(api_key=SECRET)
        response = self._response({"X-Monthly-Quota-Remaining": "12345"})
        assert adapter.quota_from(response) == 12345
        assert response.quota_remaining() == 12345

    def test_header_casing_does_not_matter(self):
        adapter = OratsAdapter(api_key=SECRET)
        assert adapter.quota_from(self._response({"x-monthly-quota-remaining": "7"})) == 7

    def test_absent_headers_yield_none_not_zero(self):
        # ORATS omits quota headers on CDN-cached responses. Reading that as 0
        # would trip the reserve guard and stall every subsequent run.
        adapter = OratsAdapter(api_key=SECRET)
        assert adapter.quota_from(self._response({})) is None

    def test_unparseable_header_yields_none(self):
        adapter = OratsAdapter(api_key=SECRET)
        assert adapter.quota_from(self._response({"X-Monthly-Quota-Remaining": "n/a"})) is None

    def test_used_and_rate_headers_are_available(self):
        adapter = OratsAdapter(api_key=SECRET)
        response = self._response(
            {"X-Monthly-Quota-Used": "500", "X-RateLimit-Remaining": "998"}
        )
        assert adapter.quota_used(response) == 500
        assert adapter.rate_remaining(response) == 998


class TestOratsAuthDetection:
    @pytest.fixture
    def adapter(self):
        return OratsAdapter(api_key=SECRET)

    def test_401_is_an_auth_failure(self, adapter):
        assert adapter.is_auth_failure(Response(401, b"", {}, "u"))

    def test_403_about_a_key_is_an_auth_failure(self, adapter):
        assert adapter.is_auth_failure(Response(403, b"Unknown API Key", {}, "u"))

    def test_403_on_a_mistyped_path_is_not_an_auth_failure(self, adapter):
        # Both a bad key and a bad path answer 403; only the body separates them,
        # and treating a typo as a rotated credential would stop a good run.
        assert not adapter.is_auth_failure(Response(403, b"resource not found", {}, "u"))

    def test_a_normal_response_is_not_an_auth_failure(self, adapter):
        assert not adapter.is_auth_failure(Response(200, b"[]", {}, "u"))


class TestPolygonContractIds:
    """OCC ids are pure formatting and silently wrong if the padding slips."""

    def test_a_known_contract_id(self):
        assert option_ticker("MRVL", "2024-09-06", "C", 93.0) == "O:MRVL240906C00093000"

    def test_puts_use_p(self):
        assert option_ticker("AAPL", "2024-01-19", "P", 185.0) == "O:AAPL240119P00185000"

    def test_fractional_strikes_scale_by_a_thousand(self):
        assert option_ticker("F", "2024-01-19", "C", 12.5) == "O:F240119C00012500"

    def test_sub_dollar_strikes_pad_correctly(self):
        assert option_ticker("SIRI", "2024-01-19", "C", 0.5) == "O:SIRI240119C00000500"

    def test_large_strikes_do_not_overflow_the_field(self):
        assert option_ticker("BRK", "2024-01-19", "C", 4000.0) == "O:BRK240119C04000000"

    def test_the_symbol_is_upper_cased(self):
        assert option_ticker("spy", "2024-01-19", "C", 500.0).startswith("O:SPY")

    def test_an_invalid_right_raises(self):
        with pytest.raises(ValueError, match="right must be"):
            option_ticker("AAPL", "2024-01-19", "X", 100.0)


class TestPolygonAdapter:
    def test_no_per_call_quota_is_reported(self):
        # Polygon bills by plan tier; there is no per-call quota to guard.
        assert PolygonAdapter(api_key="k").quota_from(Response(200, b"", {}, "u")) is None

    def test_the_unknown_api_key_body_is_an_auth_failure(self):
        # The documented fresh-process failure mode: a valid key, a 200-ish
        # response, and "Unknown API Key" in the body.
        adapter = PolygonAdapter(api_key="k")
        assert adapter.is_auth_failure(Response(200, b'{"error":"Unknown API Key"}', {}, "u"))

    def test_a_missing_key_raises(self):
        with pytest.raises(CredentialRotated, match="POLYGON_API_KEY"):
            PolygonAdapter(api_key="").request("v2/aggs", {}, 10.0)

    def test_query_params_are_sorted(self):
        adapter = PolygonAdapter(api_key="k")
        a = adapter.build_url("v3/reference/options/contracts", {"b": 2, "a": 1})
        b = adapter.build_url("v3/reference/options/contracts", {"a": 1, "b": 2})
        assert a == b and a.endswith("?a=1&b=2")


class TestOquantsPnlBan:
    """The standing rule: model-fitted marks never reach a P&L path."""

    @pytest.fixture
    def adapter(self):
        return OquantsAdapter(cookie_name="c", cookie_value="v")

    @pytest.mark.parametrize("endpoint", sorted(PNL_BANNED_ENDPOINTS))
    def test_banned_endpoints_refuse_to_be_called(self, adapter, endpoint):
        # oquants marks positions with a model-fitted "Smooth Straddle Px", not
        # a traded price: universe-wide the gap is systematic (-14pp on cheap
        # straddles, +15pp on rich ones), and a reconstruction at real prices
        # turned +66% into -48%. The ban is enforced before the request.
        with pytest.raises(ValueError, match="banned from P&L"):
            adapter.request(endpoint, {}, 10.0)

    def test_the_ban_survives_slash_variations(self, adapter):
        with pytest.raises(ValueError, match="banned from P&L"):
            adapter.request("/research/backtest/", {}, 10.0)

    def test_permitted_endpoints_are_not_blocked_by_the_ban(self, adapter):
        # It should fail on auth (no browser here), NOT on the ban.
        with pytest.raises(Exception) as excinfo:
            adapter.request("dashboard/volatility/skew-timeseries", {"ticker": "AAPL"}, 10.0)
        assert "banned from P&L" not in str(excinfo.value)

    def test_missing_cookie_raises_credential_rotated(self):
        with pytest.raises(CredentialRotated, match="OQUANTS_COOKIE"):
            OquantsAdapter(cookie_name="", cookie_value="").token()

    def test_a_token_is_reused_within_a_process(self, adapter, monkeypatch):
        # A browser launch per call turns a bulk pull into a launch loop.
        calls = {"n": 0}

        def fake_mint():
            calls["n"] += 1
            return "tok"

        monkeypatch.setattr(adapter, "_mint_token", fake_mint)
        assert adapter.token() == "tok"
        assert adapter.token() == "tok"
        assert calls["n"] == 1
        adapter.token(refresh=True)
        assert calls["n"] == 2


class TestOquantsAuthDetection:
    @pytest.fixture
    def adapter(self):
        return OquantsAdapter(cookie_name="c", cookie_value="v")

    def test_a_success_false_auth_body_is_detected(self):
        # oquants answers errors 200-with-a-body.
        adapter = OquantsAdapter(cookie_name="c", cookie_value="v")
        body = b'{"success": false, "data": null, "message": "invalid session token"}'
        assert adapter.is_auth_failure(Response(200, body, {}, "u"))

    def test_a_non_auth_error_body_is_not_an_auth_failure(self, adapter):
        body = b'{"success": false, "data": null, "message": "no data for ticker"}'
        assert not adapter.is_auth_failure(Response(200, body, {}, "u"))

    def test_a_normal_payload_is_not_an_auth_failure(self, adapter):
        assert not adapter.is_auth_failure(Response(200, b'{"data": [1,2,3]}', {}, "u"))

    def test_non_json_bodies_do_not_crash_the_check(self, adapter):
        assert not adapter.is_auth_failure(Response(200, b"<html>hello", {}, "u"))


class TestYFinanceAdapter:
    def test_only_the_history_endpoint_is_supported(self):
        with pytest.raises(ValueError, match="unsupported yfinance endpoint"):
            YFinanceAdapter().request("quote", {"ticker": "AAPL"}, 10.0)

    def test_a_ticker_is_required(self):
        with pytest.raises(ValueError, match="requires a `ticker`"):
            YFinanceAdapter().request("history", {}, 10.0)

    def test_it_reports_no_quota_and_never_auth_fails(self):
        adapter = YFinanceAdapter()
        assert adapter.quota_from(Response(200, b"", {}, "u")) is None
        assert not adapter.is_auth_failure(Response(401, b"", {}, "u"))


class TestAdapterRegistry:
    @pytest.mark.parametrize("name", ["orats", "polygon", "oquants", "yfinance"])
    def test_every_source_resolves_to_an_adapter(self, name):
        adapter = get_adapter(name)
        assert adapter.name == name
        # The fetch wrapper depends on all three of these existing.
        for method in ("request", "quota_from", "is_auth_failure"):
            assert callable(getattr(adapter, method))

    def test_an_unknown_source_is_rejected(self):
        with pytest.raises(KeyError, match="unknown source"):
            get_adapter("bloomberg")
