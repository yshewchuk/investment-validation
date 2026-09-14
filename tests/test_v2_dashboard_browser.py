"""P3-3a browser checks: the React board against the §6 mock API.

Rearchitecture Phase 3 guide §8 P3-3, exit commands:
    npm --prefix ui run typecheck
    npm --prefix ui run build
    python3 -m pytest -q tests/test_v2_dashboard_browser.py

The whole module drives a real Playwright browser (same reason
``tests/test_v2_ops_serving_browser.py`` groups its whole file, per
``tests/conftest.py``'s xdist grouping rule) and starts one real subprocess
per module (``npm --prefix ui run build``), so it is pinned to a single
xdist worker.

The mock API (``tests/fixtures/v2_ui_mock_api.py``) is a stdlib stand-in for
the real read API (P3-2, in progress) shaped exactly from
``engine/v2/contracts/serving.py``; the board never knows the difference.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright

from tests.fixtures.v2_ui_mock_api import build_default_state, serve_in_thread

pytestmark = pytest.mark.xdist_group("serial")

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = REPO_ROOT / "ui"
TOKEN = "browser-secret"


def _node_available() -> bool:
    return shutil.which("node") is not None and shutil.which("npm") is not None


@pytest.fixture(scope="module")
def dist_dir():
    """Builds ``ui/dist`` once per module via the real ``npm run build``
    (guide §8 P3-3 exit command), skipped with a clear reason only if node
    is missing."""
    if not _node_available():
        pytest.skip("node/npm not available in this environment")
    result = subprocess.run(
        ["npm", "--prefix", str(UI_ROOT), "run", "build"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        pytest.fail(f"npm --prefix ui run build failed:\n{result.stdout}\n{result.stderr}")
    dist = UI_ROOT / "dist"
    assert dist.is_dir(), "build did not produce ui/dist"
    return dist


@pytest.fixture
def state(dist_dir):
    return build_default_state(dist_root=dist_dir, token=TOKEN)


@pytest.fixture
def server(state):
    srv, thread = serve_in_thread(state)
    try:
        yield srv
    finally:
        srv.shutdown()
        thread.join(timeout=2)


@pytest.fixture(scope="module")
def playwright_instance():
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="module")
def browser(playwright_instance):
    b = playwright_instance.chromium.launch(headless=True)
    try:
        yield b
    finally:
        b.close()


def _authed_page(browser, server, base):
    context = browser.new_context()
    context.add_cookies([{"name": "operations_token", "value": TOKEN, "url": base}])
    page = context.new_page()
    return context, page


# --------------------------------------------------------------------------
# release banner
# --------------------------------------------------------------------------


def test_board_renders_release_id_and_as_of(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/")
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")
        expect(page.get_by_test_id("release-as-of")).to_contain_text(
            state.releases["r1"].release["resolved_as_of"])
    finally:
        context.close()


def test_release_banner_shows_stale_reasons(browser, server, state):
    state.set_current("r2")
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/")
        expect(page.get_by_test_id("release-stale")).to_contain_text("model_evidence_unavailable")
    finally:
        context.close()


# --------------------------------------------------------------------------
# pagination: walks all pages, no duplicates
# --------------------------------------------------------------------------


def test_pagination_walks_all_pages_without_duplicates(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/")
        expect(page.get_by_test_id("event-table")).to_be_visible()

        seen_tickers_dates: list[str] = []
        for _ in range(3):  # 120 events / 50-per-page = 3 pages
            rows = page.get_by_test_id("score-row")
            count = rows.count()
            for i in range(count):
                row = rows.nth(i)
                cells = row.locator("td")
                seen_tickers_dates.append(cells.nth(0).inner_text() + "|" + cells.nth(1).inner_text()
                                          + "|" + row.get_attribute("data-score-id"))
            next_button = page.get_by_test_id("page-next")
            if next_button.is_disabled():
                break
            next_button.click()
            page.wait_for_timeout(200)

        assert len(seen_tickers_dates) == len(set(seen_tickers_dates)), "duplicate rows across pages"
        assert len(seen_tickers_dates) > 50, "expected more than one page of rows"
    finally:
        context.close()


# --------------------------------------------------------------------------
# null vs zero, refusal, headline expected-return choice
# --------------------------------------------------------------------------


def test_null_field_shows_missing_and_zero_shows_zero(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/")
        row = page.locator('[data-score-id="r1-score-0-a"]')
        expect(row.get_by_test_id("entry-premium-cell")).to_have_text("—")  # null
        expect(row.get_by_test_id("expected-return-cell")).to_contain_text("0.0%")  # real zero
    finally:
        context.close()


def test_refusal_row_shown_as_refusal(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/")
        row = page.locator('[data-score-id="r1-score-1-a"]')
        expect(row.get_by_test_id("refusal-badge")).to_contain_text("entry cost exceeds ceiling")
    finally:
        context.close()


def test_headline_expected_return_model_null_sim_present_and_both_null(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/")
        sim_row = page.locator('[data-score-id="r1-score-2-a"]')
        expect(sim_row.get_by_test_id("expected-return-cell")).to_contain_text("11.0%")
        expect(sim_row.get_by_test_id("sim-badge")).to_be_visible()

        both_null_row = page.locator('[data-score-id="r1-score-2-b"]')
        expect(both_null_row.get_by_test_id("expected-return-cell")).to_contain_text("—")
        expect(both_null_row.get_by_test_id("sim-badge")).to_have_count(0)
    finally:
        context.close()


def test_no_scores_row_renders_unavailable(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/")
        expect(page.get_by_test_id("no-scores-row").first).to_contain_text("no scores for this event")
    finally:
        context.close()


# --------------------------------------------------------------------------
# release switch mid-session: page 2 stays on the pinned release
# --------------------------------------------------------------------------


def test_release_switch_mid_session_keeps_pinned_release_and_shows_notice(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}?pollMs=250"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base)
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")

        page.get_by_test_id("page-next").click()
        page.wait_for_timeout(200)
        first_page2_score = page.get_by_test_id("score-row").first.get_attribute("data-score-id")

        state.set_current("r2")
        expect(page.get_by_test_id("release-changed-notice")).to_be_visible(timeout=5000)
        expect(page.get_by_test_id("release-changed-notice")).to_contain_text("r2")

        # Page 2 is still served from the pinned release r1 -- same row set.
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")
        assert page.get_by_test_id("score-row").first.get_attribute("data-score-id") == first_page2_score

        page.get_by_test_id("release-changed-notice").get_by_role("button").click()
        # The background poll (pollMs=250) keeps firing after reload, so this
        # page never reaches "networkidle" -- wait for the pinned release
        # itself to change instead of a load-state event.
        expect(page.get_by_test_id("release-id")).to_contain_text("r2", timeout=10000)
    finally:
        context.close()


# --------------------------------------------------------------------------
# empty release / API error state
# --------------------------------------------------------------------------


def test_empty_release_renders_no_matches(browser, server, state):
    state.set_current("r-empty")
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/")
        expect(page.get_by_test_id("no-matches")).to_be_visible()
        expect(page.get_by_test_id("pagination-count")).to_contain_text("0 of 0")
    finally:
        context.close()


def test_api_error_state_renders(browser, server, state):
    state.force_events_error = True
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/")
        expect(page.get_by_test_id("page-error")).to_be_visible()
    finally:
        state.force_events_error = False
        context.close()


def test_no_current_release_renders_unavailable(browser, server, state):
    state.set_current(None)
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/")
        expect(page.get_by_test_id("no-release")).to_be_visible()
    finally:
        context.close()


# --------------------------------------------------------------------------
# auth / credential hygiene
# --------------------------------------------------------------------------


def test_unauthenticated_state_renders_without_cookie(browser, server):
    base = f"http://127.0.0.1:{server.server_port}"
    context = browser.new_context()
    page = context.new_page()
    try:
        page.goto(base + "/")
        expect(page.get_by_test_id("unauthenticated")).to_be_visible()
    finally:
        context.close()


def test_no_token_in_url_or_local_storage(browser, server):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/")
        expect(page.get_by_test_id("release-banner")).to_be_visible()
        assert TOKEN not in page.url
        storage = page.evaluate(
            "() => Object.entries(localStorage).map(([k,v]) => k + '=' + v).join(';')")
        assert TOKEN not in storage
    finally:
        context.close()
