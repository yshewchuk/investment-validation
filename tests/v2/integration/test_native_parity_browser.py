"""Native-parity summary browser checks (#327); mismatch rendering is covered and unpaired rendering/pagination are an owner follow-up. Synthetic page.route mocks ONLY /api/v1/native_parity** and /api/v1/releases/current, which always answers a valid 503 Problem, plus a /__data_client__/ harness serving ui/src/api/client.ts transpiled to ESM by the esbuild already installed in ui/node_modules, so the real DataClient parity detail reads run their own fetches through the mocked detail routes."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from playwright.sync_api import expect

from tests.fixtures.v2_ui_mock_api import build_default_state, serve_in_thread

# packages: engine.v2.serving, engine.v2.dashboard
pytestmark = [pytest.mark.xdist_group("serial"), pytest.mark.browser]
TOKEN = "browser-secret"
PROBLEM = {"code": "NO_CURRENT_RELEASE", "category": "unavailable", "retryable": True, "message": "no release"}
SUMMARY = {"schema_version": "native_parity_summary.v1.0", "status": "available", "partial": False, "source_schema_version": "native_parity_report.v1.2", "as_of": "2026-09-30", "generated_at": "2026-10-01T02:00:00Z", "tolerance_policy_id": "tol-1", "compared_count": 12, "matched_row_count": 10, "mismatched_row_count": 2, "only_legacy_count": 1, "only_native_count": 1, "native_refused_count": 3, "native_refused_unmatched_count": 1, "native_refused_tickers": [], "native_refused_reasons": {"unknown_ticker": 2, "missing_field": 1}}
NO_REPORT = {"status": "no_report"}

@pytest.fixture(scope="module")
def dist_dir(ui_dist_dir): return ui_dist_dir

@pytest.fixture
def server(dist_dir):
    srv, thread = serve_in_thread(build_default_state(dist_root=dist_dir, token=TOKEN))
    try:
        yield srv
    finally:
        srv.shutdown(); thread.join(timeout=2)

@pytest.fixture(autouse=True)
def _close_new_contexts(browser):
    existing = set(browser.contexts)
    yield
    for context in list(browser.contexts):
        if context not in existing:
            context.close()

EMPTY_MISMATCH_PAGE = (200, {"status": "available", "as_of": None, "generated_at": None,
    "tolerance_policy_id": None, "items": [], "next_cursor": None})

def _empty_mismatches(cursor): return EMPTY_MISMATCH_PAGE

def _open_parity(browser, server, *, code=200, body=SUMMARY, hold=False, mismatches=None):
    base = f"http://127.0.0.1:{server.server_port}"
    context = browser.new_context()
    context.add_cookies([{"name": "operations_token", "value": TOKEN, "url": base}])
    page = context.new_page()
    parity, pending = [], []
    page.mismatch_calls = []
    if mismatches is None:
        mismatches = _empty_mismatches

    def handle(route):
        url = route.request.url
        path = url.split("?")[0]
        if path.endswith("/native_parity/mismatches"):
            page.mismatch_calls.append(url)
            status, response = mismatches(route.request.url.split("cursor=")[1].split("&")[0] if "cursor=" in route.request.url else None)
            route.fulfill(status=status, content_type="application/json", body=json.dumps(response))
            return
        parity.append(url)
        if not path.endswith("/api/v1/native_parity"):
            route.fulfill(status=500, content_type="application/json", body=json.dumps(PROBLEM))
        elif hold:
            pending.append(route)
        else:
            payload = body() if callable(body) else body
            route.fulfill(status=code, content_type="application/json", body=json.dumps(payload))

    page.route("**/api/v1/native_parity**", handle)
    page.route("**/api/v1/releases/current", lambda r: r.fulfill(status=503, content_type="application/json", body=json.dumps(PROBLEM)))
    return base, context, page, parity, pending

def test_loading_then_saved_identity(browser, server):
    base, context, page, parity, pending = _open_parity(browser, server, hold=True)
    page.goto(base + "/#/native-parity")
    expect(page.get_by_test_id("parity-loading")).to_be_visible()
    expect(page.get_by_test_id("parity-summary")).to_have_count(0)
    pending[0].fulfill(status=200, content_type="application/json", body=json.dumps(SUMMARY))
    expect(page.get_by_test_id("parity-loading")).to_have_count(0)
    expect(page.get_by_test_id("parity-as-of")).to_contain_text("2026-09-30")
    expect(page.get_by_test_id("parity-generated-at")).to_contain_text("2026-10-01T02:00:00Z")
    expect(page.get_by_test_id("parity-tolerance")).to_contain_text("tol-1")
    assert len(parity) == 1
    context.close()

def test_401_summary_shows_unauthenticated(browser, server):
    base, context, page, parity, _ = _open_parity(browser, server, code=401,
        body={"code": "UNAUTHENTICATED", "category": "auth", "retryable": False, "message": "sign in"})
    page.goto(base + "/#/native-parity")
    expect(page.get_by_test_id("parity-unauthenticated")).to_be_visible()
    expect(page.get_by_test_id("parity-summary")).to_have_count(0)
    assert len(parity) == 1
    context.close()

def test_no_report_summary(browser, server):
    base, context, page, parity, _ = _open_parity(browser, server, body=NO_REPORT)
    page.goto(base + "/#/native-parity")
    expect(page.get_by_test_id("parity-no-report")).to_be_visible()
    expect(page.get_by_test_id("parity-summary")).to_have_count(0)
    assert len(parity) == 1
    context.close()

COUNT_FIELDS = [("parity-compared", "compared_count"), ("parity-matched", "matched_row_count"),
    ("parity-mismatched", "mismatched_row_count"), ("parity-only-legacy", "only_legacy_count"),
    ("parity-only-native", "only_native_count"), ("parity-native-refused", "native_refused_count"),
    ("parity-native-refused-unmatched", "native_refused_unmatched_count")]
MALFORMED = {"code": "NATIVE_PARITY_REPORT_MALFORMED", "category": "unavailable", "retryable": True, "message": "malformed report"}
STALE = dict(SUMMARY, status="stale")
ZERO = dict(SUMMARY, native_refused_reasons={},
    **{field: 0 for _, field in COUNT_FIELDS})
PARTIAL = dict(ZERO, partial=True, source_schema_version="native_parity_report.v1.1")

def test_stale_shows_saved_summary(browser, server):
    base, context, page, parity, _ = _open_parity(browser, server, body=STALE)
    page.goto(base + "/#/native-parity")
    expect(page.get_by_test_id("parity-stale")).to_be_visible()
    expect(page.get_by_test_id("parity-summary")).to_be_visible()
    expect(page.get_by_test_id("parity-as-of")).to_have_text(STALE["as_of"])
    expect(page.get_by_test_id("parity-generated-at")).to_have_text(STALE["generated_at"])
    expect(page.get_by_test_id("parity-tolerance")).to_have_text(STALE["tolerance_policy_id"])
    for testid, field in COUNT_FIELDS:
        expect(page.get_by_test_id(testid)).to_have_text(str(STALE[field]))
    expect(page.get_by_test_id("parity-refusal-reason")).to_have_count(2)
    expect(page.get_by_test_id("parity-refusal-reason").filter(has_text="unknown_ticker: 2")).to_have_text("unknown_ticker: 2")
    expect(page.get_by_test_id("parity-refusal-reason").filter(has_text="missing_field: 1")).to_have_text("missing_field: 1")
    expect(page.get_by_test_id("parity-unavailable")).to_have_count(0)
    expect(page.get_by_test_id("parity-no-report")).to_have_count(0)
    assert len(parity) == 1 and parity[0].split("?")[0].endswith("/api/v1/native_parity")
    context.close()

def test_503_malformed_withheld_as_unavailable(browser, server):
    base, context, page, parity, _ = _open_parity(browser, server, code=503, body=MALFORMED)
    page.goto(base + "/#/native-parity")
    expect(page.get_by_test_id("parity-unavailable")).to_contain_text("NATIVE_PARITY_REPORT_MALFORMED")
    expect(page.get_by_test_id("parity-summary")).to_have_count(0)
    expect(page.get_by_test_id("parity-no-report")).to_have_count(0)
    expect(page.get_by_test_id("parity-stale")).to_have_count(0)
    assert len(parity) == 1
    context.close()

def test_zero_saved_counts_render_actual_zero(browser, server):
    base, context, page, parity, _ = _open_parity(browser, server, body=ZERO)
    page.goto(base + "/#/native-parity")
    expect(page.get_by_test_id("parity-summary")).to_be_visible()
    for testid, field in COUNT_FIELDS:
        expect(page.get_by_test_id(testid)).to_have_text("0")
    expect(page.get_by_test_id("parity-as-of")).to_have_text(ZERO["as_of"])
    expect(page.get_by_test_id("parity-tolerance")).to_have_text(ZERO["tolerance_policy_id"])
    expect(page.get_by_test_id("parity-refusal-reasons-empty")).to_have_text("No refusal reasons.")
    expect(page.get_by_test_id("parity-refusal-reason")).to_have_count(0)
    expect(page.get_by_test_id("parity-refused-tickers-heading")).to_have_count(0)
    expect(page.get_by_test_id("parity-no-report")).to_have_count(0)
    expect(page.get_by_test_id("parity-unavailable")).to_have_count(0)
    expect(page.get_by_test_id("parity-stale")).to_have_count(0)
    assert len(parity) == 1
    context.close()

def test_partial_saved_summary_shows_incomplete_refusals_banner(browser, server):
    base, context, page, parity, _ = _open_parity(browser, server, body=PARTIAL)
    page.goto(base + "/#/native-parity")
    expect(page.get_by_test_id("parity-summary")).to_be_visible()
    expect(page.get_by_test_id("parity-partial")).to_be_visible()
    expect(page.get_by_test_id("parity-partial")).to_contain_text("refusal counts below are incomplete")
    expect(page.get_by_test_id("parity-native-refused")).to_have_text("0")
    expect(page.get_by_test_id("parity-refusal-reasons-empty")).to_have_text("No refusal reasons.")
    assert len(parity) == 1
    context.close()

def test_refused_tickers_render_in_saved_summary(browser, server):
    body = dict(SUMMARY, native_refused_tickers=[{"ticker": "XYZ", "reason": "PRICE_HISTORY_NOT_AVAILABLE"}])
    base, context, page, parity, _ = _open_parity(browser, server, body=body)
    page.goto(base + "/#/native-parity")
    expect(page.get_by_test_id("parity-refused-tickers-heading")).to_be_visible()
    expect(page.get_by_test_id("parity-refused-tickers-heading")).to_have_text("Refused tickers (1)")
    expect(page.get_by_test_id("parity-refused-ticker")).to_have_count(1)
    expect(page.get_by_test_id("parity-refused-ticker")).to_have_text("XYZ: PRICE_HISTORY_NOT_AVAILABLE")
    assert len(parity) == 1
    context.close()

def test_shared_board_link_reaches_parity_after_no_release(browser, server):
    # releases/current is 503 in _open_parity, so the board opens into its
    # no-release state; the shared link must still reach the parity summary.
    base, context, page, parity, _ = _open_parity(browser, server)
    try:
        page.goto(base + "/#/")
        expect(page.get_by_test_id("no-release")).to_be_visible()
        page.get_by_test_id("native-parity-link").click()
        expect(page).to_have_url(base + "/#/native-parity")
        expect(page.get_by_test_id("parity-summary")).to_be_visible()
        expect(page.get_by_test_id("parity-as-of")).to_have_text(SUMMARY["as_of"])
        assert len(parity) == 1
    finally:
        context.close()

UI_ROOT = Path(__file__).resolve().parents[3] / "ui"
HARNESS_PATH = "/__data_client__/harness.html"
CLIENT_PATH = "/__data_client__/client.js"
MISMATCH_PAGE = {"status": "available", "as_of": "2026-09-30", "generated_at": "2026-10-01T02:00:00Z", "tolerance_policy_id": "tol-1",
    "items": [
        {"row_key": "AAPL|2026-09-15|STR-THRU", "dimension": "verdicts", "fields": {
            "gate_score": {"status": "differ", "legacy": 0.61, "native": 0.63},
            "gate_threshold": {"status": "agree"}, "gate_pass": {"status": "agree"}}},
        {"row_key": "MSFT|2026-09-18|STR-THRU", "dimension": "financial_diagnostics", "fields": {
            "entry_cost_pct": {"status": "agree"},
            "model_vs_market": {"status": "differ", "legacy": -0.015, "native": 0.02},
            "fair_premium_pct": {"status": "agree"}, "premium_vs_fair": {"status": "agree"},
            "cost_over_width": {"status": "agree"}}}],
    "next_cursor": "mm-2"}
UNPAIRED_PAGE = {"status": "available", "as_of": "2026-09-30", "generated_at": "2026-10-01T02:00:00Z", "tolerance_policy_id": "tol-1",
    "items": ["NVDA|2026-09-22|STR-THRU", "TSLA|2026-09-29|STR-THRU"], "next_cursor": None}

def _transpile_client(tmp_path):
    """`ui/src/api/client.ts` transpiled to plain ESM by the esbuild already in
    `ui/node_modules` (present because the session `ui_dist_dir` fixture ran
    `npm ci`), written under `tmp_path` for a Playwright route to serve. Only
    the network is mocked below; the module, its `getJson`, `queryString` and
    the two detail methods are the real client code."""
    out = tmp_path / "client.js"
    script = tmp_path / "transpile.cjs"
    script.write_text(
        "const fs = require('fs');\n"
        f"const esbuild = require({json.dumps(str(UI_ROOT / 'node_modules' / 'esbuild'))});\n"
        f"const source = fs.readFileSync({json.dumps(str(UI_ROOT / 'src' / 'api' / 'client.ts'))}, 'utf8');\n"
        "const transpiled = esbuild.transformSync(source, {loader: 'ts', format: 'esm'});\n"
        f"fs.writeFileSync({json.dumps(str(out))}, transpiled.code);\n", encoding="utf-8")
    subprocess.run(["node", str(script)], check=True, capture_output=True, text=True, timeout=120)
    js = out.read_text(encoding="utf-8")
    assert "createHttpDataClient" in js and "import type" not in js
    return js

def _open_data_client(browser, server, client_js, route_name, body):
    base = f"http://127.0.0.1:{server.server_port}"
    context = browser.new_context()
    page = context.new_page()
    calls = []
    page.route(f"**{HARNESS_PATH}",
        lambda r: r.fulfill(status=200, content_type="text/html", body="<html><body>data-client harness</body></html>"))
    page.route(f"**{CLIENT_PATH}",
        lambda r: r.fulfill(status=200, content_type="text/javascript", body=client_js))

    def handle(route):
        calls.append(route.request.url)
        route.fulfill(status=200, content_type="application/json", body=json.dumps(body))

    page.route(f"**/api/v1/native_parity/{route_name}**", handle)
    return base, context, page, calls

IMPORT_AND_CALL = ("async ([method, args]) => {"
    " const mod = await import(" + json.dumps(CLIENT_PATH) + ");"
    " const client = mod.createHttpDataClient(" + json.dumps("/api/v1") + ");"
    " return await client[method](...args); }")

def test_mismatches_no_report_is_status_only(browser, server, tmp_path):
    base, context, page, calls = _open_data_client(
        browser, server, _transpile_client(tmp_path), "mismatches", NO_REPORT)
    try:
        page.goto(base + HARNESS_PATH)
        result = page.evaluate(IMPORT_AND_CALL, ["getNativeParityMismatches", [5, None]])
        assert result == {"status": "no_report"}
        assert calls == [base + "/api/v1/native_parity/mismatches?limit=5"]
    finally:
        context.close()

def test_unpaired_no_report_is_status_only(browser, server, tmp_path):
    base, context, page, calls = _open_data_client(
        browser, server, _transpile_client(tmp_path), "unpaired", NO_REPORT)
    try:
        page.goto(base + HARNESS_PATH)
        result = page.evaluate(IMPORT_AND_CALL, ["getNativeParityUnpaired", ["native", 4, None]])
        assert result == {"status": "no_report"}
        assert calls == [base + "/api/v1/native_parity/unpaired?side=native&limit=4"]
    finally:
        context.close()

def test_mismatches_available_page_round_trips_exactly(browser, server, tmp_path):
    base, context, page, calls = _open_data_client(
        browser, server, _transpile_client(tmp_path), "mismatches", MISMATCH_PAGE)
    try:
        page.goto(base + HARNESS_PATH)
        result = page.evaluate(IMPORT_AND_CALL, ["getNativeParityMismatches", [2, "mm-1"]])
        assert result == MISMATCH_PAGE
        assert calls == [base + "/api/v1/native_parity/mismatches?limit=2&cursor=mm-1"]
    finally:
        context.close()

def test_unpaired_available_page_round_trips_exactly(browser, server, tmp_path):
    base, context, page, calls = _open_data_client(
        browser, server, _transpile_client(tmp_path), "unpaired", UNPAIRED_PAGE)
    try:
        page.goto(base + HARNESS_PATH)
        result = page.evaluate(IMPORT_AND_CALL, ["getNativeParityUnpaired", ["legacy", 3, "uc-1"]])
        assert result == UNPAIRED_PAGE
        assert calls == [base + "/api/v1/native_parity/unpaired?side=legacy&limit=3&cursor=uc-1"]
    finally:
        context.close()

def _page(items, next_cursor=None, as_of="2026-09-30", generated_at="2026-10-01T02:00:00Z"):
    return {"status": "available", "as_of": as_of, "generated_at": generated_at,
        "tolerance_policy_id": "tol-1", "items": items, "next_cursor": next_cursor}

def test_mismatch_table_shows_only_differing_fields(browser, server):
    first = _page(MISMATCH_PAGE["items"])
    base, context, page, _, _ = _open_parity(browser, server, mismatches=lambda c: (200, first))
    try:
        page.goto(base + "/#/native-parity")
        expect(page.get_by_test_id("parity-mismatch-table")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_disabled()
        expect(page.get_by_test_id("parity-mismatch-row")).to_have_count(2)
        aapl = page.get_by_test_id("parity-mismatch-row").filter(has_text="AAPL|2026-09-15|STR-THRU")
        expect(aapl).to_contain_text("gate_score")
        expect(aapl.locator("td").nth(1)).to_have_text("verdicts")
        expect(aapl.get_by_test_id("parity-mismatch-legacy")).to_have_text("0.61")
        expect(aapl.get_by_test_id("parity-mismatch-native")).to_have_text("0.63")
        msft = page.get_by_test_id("parity-mismatch-row").filter(has_text="MSFT|2026-09-18|STR-THRU")
        expect(msft).to_contain_text("model_vs_market")
        expect(msft.locator("td").nth(1)).to_have_text("financial_diagnostics")
        expect(msft.get_by_test_id("parity-mismatch-legacy")).to_have_text("-0.015")
        expect(msft.get_by_test_id("parity-mismatch-native")).to_have_text("0.02")
        page_text = page.get_by_test_id("parity-mismatch-table").inner_text()
        assert "gate_threshold" not in page_text
        assert "entry_cost_pct" not in page_text
        expect(page.get_by_test_id("parity-mismatch-next")).to_be_disabled()
        expect(page.get_by_test_id("parity-mismatch-identity-changed")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-identity")).to_contain_text("2026-10-01T02:00:00Z")
    finally:
        context.close()

def test_mismatch_item_with_multiple_differing_fields_and_value_kinds(browser, server):
    page_body = _page([{"row_key": "KO|2026-09-10|STR-THRU", "dimension": "verdicts", "fields": {
            "gate_pass": {"status": "differ", "legacy": True, "native": False},
            "reason": {"status": "differ", "legacy": "ok", "native": "withheld"},
            "gate_score": {"status": "differ", "legacy": None, "native": 0.5},
            "gate_threshold": {"status": "differ", "native": 0.7},
            "zero_flag": {"status": "differ", "legacy": 0, "native": 1},
            "zero_count": {"status": "differ", "legacy": 2, "native": 0},
            "n_analogs": {"status": "agree"}}}])
    base, context, page, _, _ = _open_parity(browser, server, mismatches=lambda c: (200, page_body))
    try:
        page.goto(base + "/#/native-parity")
        rows = page.get_by_test_id("parity-mismatch-row")
        expect(rows).to_have_count(6)
        expect(rows.filter(has_text="KO|2026-09-10|STR-THRU")).to_have_count(6)
        gate_pass = rows.filter(has_text="gate_pass")
        expect(gate_pass.get_by_test_id("parity-mismatch-legacy")).to_have_text("true")
        expect(gate_pass.get_by_test_id("parity-mismatch-native")).to_have_text("false")
        reason = rows.filter(has_text="reason")
        expect(reason.get_by_test_id("parity-mismatch-legacy")).to_have_text("ok")
        expect(reason.get_by_test_id("parity-mismatch-native")).to_have_text("withheld")
        gate_score = rows.filter(has_text="gate_score")
        expect(gate_score.get_by_test_id("parity-mismatch-legacy")).to_have_text("—")
        expect(gate_score.get_by_test_id("parity-mismatch-native")).to_have_text("0.5")
        gate_threshold = rows.filter(has_text="gate_threshold")
        expect(gate_threshold.get_by_test_id("parity-mismatch-legacy")).to_have_text("—")
        expect(gate_threshold.get_by_test_id("parity-mismatch-native")).to_have_text("0.7")
        zero_flag = rows.filter(has_text="zero_flag")
        expect(zero_flag.get_by_test_id("parity-mismatch-legacy")).to_have_text("0")
        expect(zero_flag.get_by_test_id("parity-mismatch-native")).to_have_text("1")
        zero_count = rows.filter(has_text="zero_count")
        expect(zero_count.get_by_test_id("parity-mismatch-legacy")).to_have_text("2")
        expect(zero_count.get_by_test_id("parity-mismatch-native")).to_have_text("0")
        table_text = page.get_by_test_id("parity-mismatch-table").inner_text()
        assert "n_analogs" not in table_text
        assert table_text.count("—") == 2
    finally:
        context.close()

def test_mismatch_table_pages_past_first_page(browser, server):
    page1 = {"status": "available", "as_of": None, "generated_at": None, "tolerance_policy_id": None,
        "items": [{"row_key": "AAPL|2026-09-15|STR-THRU", "dimension": "verdicts",
            "fields": {"gate_score": {"status": "differ", "legacy": 0.61, "native": 0.63}}}],
        "next_cursor": "c2"}
    page2 = {"status": "available", "as_of": None, "generated_at": None, "tolerance_policy_id": None,
        "items": [{"row_key": "NVDA|2026-09-22|STR-THRU", "dimension": "verdicts",
            "fields": {"gate_score": {"status": "differ", "legacy": 1.0, "native": 2.0}}}],
        "next_cursor": None}

    def mismatches(cursor):
        return (200, page1 if cursor is None else page2)

    base, context, page, _, _ = _open_parity(browser, server, mismatches=mismatches)
    try:
        page.goto(base + "/#/native-parity")
        aapl = page.get_by_test_id("parity-mismatch-row").filter(has_text="AAPL|2026-09-15|STR-THRU")
        expect(aapl).to_be_visible()
        expect(aapl.get_by_test_id("parity-mismatch-legacy")).to_have_text("0.61")
        expect(aapl.get_by_test_id("parity-mismatch-native")).to_have_text("0.63")
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 1")
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_disabled()
        page.get_by_test_id("parity-mismatch-next").click()
        nvda = page.get_by_test_id("parity-mismatch-row").filter(has_text="NVDA|2026-09-22|STR-THRU")
        expect(nvda).to_be_visible()
        expect(nvda.get_by_test_id("parity-mismatch-legacy")).to_have_text("1")
        expect(nvda.get_by_test_id("parity-mismatch-native")).to_have_text("2")
        expect(aapl).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 2")
        expect(page.get_by_test_id("parity-mismatch-next")).to_be_disabled()
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_enabled()
        assert any("cursor=c2" in u for u in page.mismatch_calls)
        page.get_by_test_id("parity-mismatch-prev").click()
        expect(aapl).to_be_visible()
        expect(aapl.get_by_test_id("parity-mismatch-legacy")).to_have_text("0.61")
        expect(aapl.get_by_test_id("parity-mismatch-native")).to_have_text("0.63")
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 1")
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_disabled()
    finally:
        context.close()

def test_mismatch_empty_and_no_report(browser, server):
    base, context, page, _, _ = _open_parity(browser, server, mismatches=lambda c: EMPTY_MISMATCH_PAGE)
    try:
        page.goto(base + "/#/native-parity")
        expect(page.get_by_test_id("parity-mismatch-empty")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-loading")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-unavailable")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-no-report")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-table")).to_have_count(0)
        expect(page.get_by_test_id("parity-summary")).to_be_visible()
    finally:
        context.close()
    base, context, page, _, _ = _open_parity(browser, server, mismatches=lambda c: (200, {"status": "no_report"}))
    try:
        page.goto(base + "/#/native-parity")
        expect(page.get_by_test_id("parity-mismatch-no-report")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-empty")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-table")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-unavailable")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-loading")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-pager")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-identity-changed")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-identity")).to_have_count(0)
    finally:
        context.close()

def test_mismatch_table_walks_three_pages_with_re_read(browser, server):
    page1 = _page([{"row_key": "AAPL|2026-09-15|STR-THRU", "dimension": "verdicts",
        "fields": {"gate_score": {"status": "differ", "legacy": 0.61, "native": 0.63}}}], next_cursor="c2")
    page2 = _page([{"row_key": "NVDA|2026-09-22|STR-THRU", "dimension": "verdicts",
        "fields": {"gate_score": {"status": "differ", "legacy": 1.5, "native": 2.5}}}], next_cursor="c3")
    page3 = _page([{"row_key": "TSLA|2026-09-29|STR-THRU", "dimension": "verdicts",
        "fields": {"gate_score": {"status": "differ", "legacy": 7, "native": 8}}}])

    def mismatches(cursor):
        return (200, {None: page1, "c2": page2, "c3": page3}[cursor])

    base, context, page, _, _ = _open_parity(browser, server, mismatches=mismatches)
    try:
        page.goto(base + "/#/native-parity")
        rows = page.get_by_test_id("parity-mismatch-row")
        expect(rows).to_have_count(1)
        expect(rows.filter(has_text="AAPL|2026-09-15|STR-THRU")).to_be_visible()
        expect(rows.filter(has_text="AAPL|2026-09-15|STR-THRU").get_by_test_id("parity-mismatch-legacy")).to_have_text("0.61")
        expect(rows.filter(has_text="AAPL|2026-09-15|STR-THRU").get_by_test_id("parity-mismatch-native")).to_have_text("0.63")
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 1")
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_disabled()
        page.get_by_test_id("parity-mismatch-next").click()
        expect(rows).to_have_count(1)
        expect(rows.filter(has_text="NVDA|2026-09-22|STR-THRU")).to_be_visible()
        expect(rows.filter(has_text="NVDA|2026-09-22|STR-THRU").get_by_test_id("parity-mismatch-legacy")).to_have_text("1.5")
        expect(rows.filter(has_text="NVDA|2026-09-22|STR-THRU").get_by_test_id("parity-mismatch-native")).to_have_text("2.5")
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 2")
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_enabled()
        page.get_by_test_id("parity-mismatch-next").click()
        expect(rows).to_have_count(1)
        expect(rows.filter(has_text="TSLA|2026-09-29|STR-THRU")).to_be_visible()
        expect(rows.filter(has_text="TSLA|2026-09-29|STR-THRU").get_by_test_id("parity-mismatch-legacy")).to_have_text("7")
        expect(rows.filter(has_text="TSLA|2026-09-29|STR-THRU").get_by_test_id("parity-mismatch-native")).to_have_text("8")
        expect(rows.filter(has_text="AAPL|2026-09-15|STR-THRU")).to_have_count(0)
        expect(rows.filter(has_text="NVDA|2026-09-22|STR-THRU")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 3")
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_enabled()
        expect(page.get_by_test_id("parity-mismatch-next")).to_be_disabled()
        page.get_by_test_id("parity-mismatch-prev").click()
        expect(rows).to_have_count(1)
        expect(rows.filter(has_text="NVDA|2026-09-22|STR-THRU")).to_be_visible()
        expect(rows.filter(has_text="NVDA|2026-09-22|STR-THRU").get_by_test_id("parity-mismatch-legacy")).to_have_text("1.5")
        expect(rows.filter(has_text="NVDA|2026-09-22|STR-THRU").get_by_test_id("parity-mismatch-native")).to_have_text("2.5")
        expect(rows.filter(has_text="AAPL|2026-09-15|STR-THRU")).to_have_count(0)
        expect(rows.filter(has_text="TSLA|2026-09-29|STR-THRU")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 2")
        page.get_by_test_id("parity-mismatch-prev").click()
        expect(rows.filter(has_text="AAPL|2026-09-15|STR-THRU")).to_be_visible()
        expect(rows.filter(has_text="AAPL|2026-09-15|STR-THRU").get_by_test_id("parity-mismatch-legacy")).to_have_text("0.61")
        expect(rows.filter(has_text="AAPL|2026-09-15|STR-THRU").get_by_test_id("parity-mismatch-native")).to_have_text("0.63")
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 1")
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_disabled()
        page.get_by_test_id("parity-mismatch-next").click()
        expect(rows.filter(has_text="NVDA|2026-09-22|STR-THRU")).to_be_visible()
        expect(rows.filter(has_text="NVDA|2026-09-22|STR-THRU").get_by_test_id("parity-mismatch-legacy")).to_have_text("1.5")
        expect(rows.filter(has_text="NVDA|2026-09-22|STR-THRU").get_by_test_id("parity-mismatch-native")).to_have_text("2.5")
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 2")
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_enabled()
        cursors = [u.split("cursor=")[1].split("&")[0] if "cursor=" in u else None
            for u in page.mismatch_calls]
        assert cursors == [None, "c2", "c3", "c2", None, "c2"]
    finally:
        context.close()

def test_mismatch_loading_state(browser, server):
    mismatch_pending = []
    base, context, page, _, _ = _open_parity(browser, server, hold=True)
    page.unroute("**/api/v1/native_parity**")
    parity = []

    def handle(route):
        url = route.request.url
        path = url.split("?")[0]
        if path.endswith("/native_parity/mismatches"):
            page.mismatch_calls.append(url)
            mismatch_pending.append(route)
        elif path.endswith("/api/v1/native_parity"):
            parity.append(url)
            route.fulfill(status=200, content_type="application/json", body=json.dumps(SUMMARY))
        else:
            route.fulfill(status=500, content_type="application/json", body=json.dumps(PROBLEM))

    page.route("**/api/v1/native_parity**", handle)
    try:
        page.goto(base + "/#/native-parity")
        expect(page.get_by_test_id("parity-summary")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-loading")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-no-report")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-empty")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-table")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-unavailable")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-unauthenticated")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-pager")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-identity-changed")).to_have_count(0)
        mismatch_pending[0].fulfill(status=200, content_type="application/json",
            body=json.dumps(EMPTY_MISMATCH_PAGE[1]))
        expect(page.get_by_test_id("parity-mismatch-loading")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-empty")).to_be_visible()
    finally:
        context.close()

def test_mismatch_error_state(browser, server):
    base, context, page, _, _ = _open_parity(browser, server, mismatches=lambda c: (503, PROBLEM))
    try:
        page.goto(base + "/#/native-parity")
        unavailable = page.get_by_test_id("parity-mismatch-unavailable")
        expect(unavailable).to_contain_text("NO_CURRENT_RELEASE")
        expect(unavailable).to_contain_text("no release")
        expect(unavailable).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-empty")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-table")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-no-report")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-loading")).to_have_count(0)
        expect(page.get_by_test_id("parity-summary")).to_be_visible()
    finally:
        context.close()

def test_mismatch_error_recovery_restarts(browser, server):
    summary = {"d": SUMMARY}
    failing = {"on": True}
    base, context, page, parity, _ = _open_parity(browser, server, body=lambda: summary["d"],
        mismatches=lambda c: (503, PROBLEM) if failing["on"] else (200, EMPTY_MISMATCH_PAGE[1]))
    try:
        page.goto(base + "/#/native-parity")
        expect(page.get_by_test_id("parity-mismatch-unavailable")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_disabled()
        failing["on"] = False
        summary["d"] = dict(SUMMARY, as_of="2026-10-01", generated_at="2026-10-02T02:00:00Z", compared_count=99)
        page.get_by_test_id("parity-mismatch-restart").click()
        expect(page.get_by_test_id("parity-mismatch-unavailable")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-empty")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-table")).to_have_count(0)
        expect(page.get_by_test_id("parity-as-of")).to_have_text("2026-10-01")
        expect(page.get_by_test_id("parity-generated-at")).to_have_text("2026-10-02T02:00:00Z")
        expect(page.get_by_test_id("parity-compared")).to_have_text("99")
        expect(page.get_by_test_id("parity-mismatch-identity-changed")).to_be_visible()
        assert len(parity) == 2
    finally:
        context.close()

def test_mismatch_later_page_error_recovers_via_previous_and_restart(browser, server):
    fail = {"on": True}
    page_a = _page([{"row_key": "AAPL|2026-09-15|STR-THRU", "dimension": "verdicts",
        "fields": {"gate_score": {"status": "differ", "legacy": 0.61, "native": 0.63}}}], next_cursor="c2")
    page_b = _page([{"row_key": "NVDA|2026-09-22|STR-THRU", "dimension": "verdicts",
        "fields": {"gate_score": {"status": "differ", "legacy": 1.5, "native": 2.5}}}])

    def mismatches(cursor):
        if cursor == "c2" and fail["on"]:
            return (503, PROBLEM)
        return (200, page_a if cursor is None else page_b)

    base, context, page, parity, _ = _open_parity(browser, server, mismatches=mismatches)
    try:
        page.goto(base + "/#/native-parity")
        expect(page.get_by_test_id("parity-mismatch-row").filter(has_text="AAPL|2026-09-15|STR-THRU")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-row").filter(has_text="AAPL|2026-09-15|STR-THRU").get_by_test_id("parity-mismatch-legacy")).to_have_text("0.61")
        expect(page.get_by_test_id("parity-mismatch-row").filter(has_text="AAPL|2026-09-15|STR-THRU").get_by_test_id("parity-mismatch-native")).to_have_text("0.63")
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 1")
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_disabled()
        page.get_by_test_id("parity-mismatch-next").click()
        expect(page.get_by_test_id("parity-mismatch-unavailable")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_enabled()
        expect(page.get_by_test_id("parity-mismatch-restart")).to_be_visible()
        page.get_by_test_id("parity-mismatch-prev").click()
        expect(page.get_by_test_id("parity-mismatch-row").filter(has_text="AAPL|2026-09-15|STR-THRU")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-row").filter(has_text="AAPL|2026-09-15|STR-THRU").get_by_test_id("parity-mismatch-legacy")).to_have_text("0.61")
        expect(page.get_by_test_id("parity-mismatch-row").filter(has_text="AAPL|2026-09-15|STR-THRU").get_by_test_id("parity-mismatch-native")).to_have_text("0.63")
        expect(page.get_by_test_id("parity-mismatch-unavailable")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 1")
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_disabled()
        page.get_by_test_id("parity-mismatch-next").click()
        expect(page.get_by_test_id("parity-mismatch-unavailable")).to_be_visible()
        fail["on"] = False
        page.get_by_test_id("parity-mismatch-restart").click()
        expect(page.get_by_test_id("parity-mismatch-row").filter(has_text="AAPL|2026-09-15|STR-THRU")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-row").filter(has_text="AAPL|2026-09-15|STR-THRU").get_by_test_id("parity-mismatch-legacy")).to_have_text("0.61")
        expect(page.get_by_test_id("parity-mismatch-row").filter(has_text="AAPL|2026-09-15|STR-THRU").get_by_test_id("parity-mismatch-native")).to_have_text("0.63")
        expect(page.get_by_test_id("parity-mismatch-row").filter(has_text="NVDA|2026-09-22|STR-THRU")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-unavailable")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 1")
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_disabled()
        assert len(parity) == 2
        assert "cursor=" not in page.mismatch_calls[-1]
        page.get_by_test_id("parity-mismatch-next").click()
        expect(page.get_by_test_id("parity-mismatch-row").filter(has_text="NVDA|2026-09-22|STR-THRU")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-row").filter(has_text="NVDA|2026-09-22|STR-THRU").get_by_test_id("parity-mismatch-legacy")).to_have_text("1.5")
        expect(page.get_by_test_id("parity-mismatch-row").filter(has_text="NVDA|2026-09-22|STR-THRU").get_by_test_id("parity-mismatch-native")).to_have_text("2.5")
    finally:
        context.close()

def test_mismatch_restart_from_successful_later_page(browser, server):
    summary = {"d": SUMMARY}
    page_a = _page([{"row_key": "AAPL|2026-09-15|STR-THRU", "dimension": "verdicts",
        "fields": {"gate_score": {"status": "differ", "legacy": 0.61, "native": 0.63}}}], next_cursor="c2")
    page_b = _page([{"row_key": "NVDA|2026-09-22|STR-THRU", "dimension": "verdicts",
        "fields": {"gate_score": {"status": "differ", "legacy": 1.0, "native": 2.0}}}])

    def mismatches(cursor):
        return (200, page_a if cursor is None else page_b)

    base, context, page, parity, _ = _open_parity(browser, server, body=lambda: summary["d"], mismatches=mismatches)
    try:
        page.goto(base + "/#/native-parity")
        aapl = page.get_by_test_id("parity-mismatch-row").filter(has_text="AAPL|2026-09-15|STR-THRU")
        expect(aapl).to_be_visible()
        expect(aapl.get_by_test_id("parity-mismatch-legacy")).to_have_text("0.61")
        expect(aapl.get_by_test_id("parity-mismatch-native")).to_have_text("0.63")
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 1")
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_disabled()
        page.get_by_test_id("parity-mismatch-next").click()
        nvda = page.get_by_test_id("parity-mismatch-row").filter(has_text="NVDA|2026-09-22|STR-THRU")
        expect(nvda).to_be_visible()
        expect(nvda.get_by_test_id("parity-mismatch-legacy")).to_have_text("1")
        expect(nvda.get_by_test_id("parity-mismatch-native")).to_have_text("2")
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 2")
        expect(page.get_by_test_id("parity-mismatch-restart")).to_be_visible()
        summary["d"] = dict(SUMMARY, as_of="2026-10-01", generated_at="2026-10-02T02:00:00Z", compared_count=99)
        page.get_by_test_id("parity-mismatch-restart").click()
        expect(aapl).to_be_visible()
        expect(aapl.get_by_test_id("parity-mismatch-legacy")).to_have_text("0.61")
        expect(aapl.get_by_test_id("parity-mismatch-native")).to_have_text("0.63")
        expect(page.get_by_test_id("parity-mismatch-row").filter(has_text="NVDA|2026-09-22|STR-THRU")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-page")).to_have_text("Page 1")
        expect(page.get_by_test_id("parity-mismatch-prev")).to_be_disabled()
        expect(page.get_by_test_id("parity-as-of")).to_have_text("2026-10-01")
        expect(page.get_by_test_id("parity-generated-at")).to_have_text("2026-10-02T02:00:00Z")
        expect(page.get_by_test_id("parity-compared")).to_have_text("99")
        expect(page.get_by_test_id("parity-mismatch-identity-changed")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-identity")).to_have_text(
            "Rows from report generated 2026-10-01T02:00:00Z (as of 2026-09-30)")
        assert "cursor=" not in page.mismatch_calls[-1]
        assert len(parity) == 2
    finally:
        context.close()

def test_mismatch_401_shows_unauthenticated(browser, server):
    base, context, page, _, _ = _open_parity(browser, server,
        mismatches=lambda c: (401, {"code": "UNAUTHENTICATED", "category": "auth", "retryable": False, "message": "sign in"}))
    try:
        page.goto(base + "/#/native-parity")
        expect(page.get_by_test_id("parity-mismatch-unauthenticated")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-table")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-unavailable")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-empty")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-no-report")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-loading")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-pager")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-identity")).to_have_count(0)
        expect(page.get_by_test_id("parity-summary")).to_be_visible()
    finally:
        context.close()

@pytest.mark.parametrize("as_of,generated_at", [
    ("2026-10-01", SUMMARY["generated_at"]),
    (SUMMARY["as_of"], "2026-10-02T02:00:00Z"),
    ("2026-10-01", "2026-10-02T02:00:00Z"),
])
def test_mismatch_identity_changing_detail_banner(browser, server, as_of, generated_at):
    identity = {"as_of": as_of, "generated_at": generated_at}
    items = [{"row_key": "AAPL|2026-09-15|STR-THRU", "dimension": "verdicts",
        "fields": {"gate_score": {"status": "differ", "legacy": 0.61, "native": 0.63}}}]

    def mismatches(cursor):
        return (200, _page(items, as_of=identity["as_of"], generated_at=identity["generated_at"]))

    base, context, page, parity, _ = _open_parity(browser, server, mismatches=mismatches)
    try:
        page.goto(base + "/#/native-parity")
        expect(page.get_by_test_id("parity-mismatch-identity-changed")).to_be_visible()
        expect(page.get_by_test_id("parity-mismatch-identity")).to_have_text(
            f"Rows from report generated {generated_at} (as of {as_of})")
        identity["as_of"] = SUMMARY["as_of"]
        identity["generated_at"] = SUMMARY["generated_at"]
        page.get_by_test_id("parity-mismatch-restart").click()
        expect(page.get_by_test_id("parity-mismatch-identity-changed")).to_have_count(0)
        expect(page.get_by_test_id("parity-mismatch-identity")).to_have_text(
            f"Rows from report generated {SUMMARY['generated_at']} (as of {SUMMARY['as_of']})")
        assert len(parity) == 2
    finally:
        context.close()
