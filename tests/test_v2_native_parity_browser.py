"""Native-parity summary browser checks (#327); mismatch/unpaired rendering and pagination are an owner follow-up. Synthetic page.route mocks ONLY /api/v1/native_parity** and /api/v1/releases/current, which always answers a valid 503 Problem."""
from __future__ import annotations

import json

import pytest
from playwright.sync_api import expect

from tests.fixtures.v2_ui_mock_api import build_default_state, serve_in_thread

pytestmark = [pytest.mark.xdist_group("serial"), pytest.mark.browser]
TOKEN = "browser-secret"
PROBLEM = {"code": "NO_CURRENT_RELEASE", "category": "unavailable", "retryable": True, "message": "no release"}
SUMMARY = {"status": "available", "as_of": "2026-09-30", "generated_at": "2026-10-01T02:00:00Z", "tolerance_policy_id": "tol-1", "compared_count": 12, "matched_row_count": 10, "mismatched_row_count": 2, "only_legacy_count": 1, "only_native_count": 1, "native_refused_count": 3, "native_refused_unmatched_count": 1, "native_refused_reasons": {"unknown_ticker": 2, "missing_field": 1}}
NO_REPORT = dict.fromkeys(SUMMARY, None) | {"status": "no_report", "native_refused_reasons": {}}

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

def _open_parity(browser, server, *, code=200, body=SUMMARY, hold=False):
    base = f"http://127.0.0.1:{server.server_port}"
    context = browser.new_context()
    context.add_cookies([{"name": "operations_token", "value": TOKEN, "url": base}])
    page = context.new_page()
    parity, pending = [], []

    def handle(route):
        url = route.request.url
        parity.append(url)
        if not url.split("?")[0].endswith("/api/v1/native_parity"):
            route.fulfill(status=500, content_type="application/json", body=json.dumps(PROBLEM))
        elif hold:
            pending.append(route)
        else:
            route.fulfill(status=code, content_type="application/json", body=json.dumps(body))

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
ZERO = dict(SUMMARY, as_of=None, tolerance_policy_id=None, native_refused_reasons={},
    **{field: 0 for _, field in COUNT_FIELDS})

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
    expect(page.get_by_test_id("parity-as-of")).to_have_text("—")
    expect(page.get_by_test_id("parity-tolerance")).to_have_text("—")
    expect(page.get_by_test_id("parity-refusal-reasons-empty")).to_have_text("No refusal reasons.")
    expect(page.get_by_test_id("parity-refusal-reason")).to_have_count(0)
    expect(page.get_by_test_id("parity-no-report")).to_have_count(0)
    expect(page.get_by_test_id("parity-unavailable")).to_have_count(0)
    expect(page.get_by_test_id("parity-stale")).to_have_count(0)
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
