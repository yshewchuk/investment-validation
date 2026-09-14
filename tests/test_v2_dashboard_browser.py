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

import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
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


# --------------------------------------------------------------------------
# P3-3b: event and score detail views
# --------------------------------------------------------------------------


def test_board_to_event_to_score_and_back_keeps_pagination(browser, server, state):
    """Guide P3-3b deliverable 6: "board -> event -> score -> back keeps
    pagination". Uses page 2 (a row not covered by DETAIL_OVERRIDES) so the
    assertion is purely about navigation/state, not detail content."""
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/")
        expect(page.get_by_test_id("event-table")).to_be_visible()
        page.get_by_test_id("page-next").click()
        page.wait_for_timeout(200)

        first_row = page.get_by_test_id("score-row").first
        page2_score_id = first_row.get_attribute("data-score-id")
        assert page2_score_id is not None

        first_row.get_by_test_id("open-event-link").click()
        expect(page.get_by_test_id("event-detail")).to_be_visible()
        expect(page.get_by_test_id("event-scores-table")).to_be_visible()

        page.get_by_test_id("open-score-link").first.click()
        expect(page.get_by_test_id("score-detail")).to_be_visible()
        expect(page.get_by_test_id("score-detail-header")).to_be_visible()

        page.get_by_test_id("back-to-board").click()
        expect(page.get_by_test_id("event-table")).to_be_visible()
        expect(page.get_by_test_id("score-row").first).to_have_attribute(
            "data-score-id", page2_score_id)
    finally:
        context.close()


def test_deep_link_score_on_non_current_release_shows_notice(browser, server, state):
    """P3-3b deliverable 3/6: a deep link to a score on a release that is
    not current still loads it, and shows the notice rather than silently
    switching to the actually-current release (r1)."""
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/#/release/r2/scores/r2-score-0-a")
        expect(page.get_by_test_id("score-detail")).to_be_visible()
        expect(page.get_by_test_id("score-id-value")).to_contain_text("r2-score-0-a")
        expect(page.get_by_test_id("score-release-id-value")).to_contain_text("r2")
        expect(page.get_by_test_id("release-id")).to_contain_text("r2")
        expect(page.get_by_test_id("release-not-current-notice")).to_be_visible()
    finally:
        context.close()


def test_release_switch_mid_session_keeps_detail_fetches_pinned(browser, server, state):
    """P3-3b deliverable 6: switching the release mid-session keeps detail
    fetches on the pinned release. `r1-score-4-a` only exists under release
    r1's fixture -- if the score fetch had drifted onto the now-current r2
    instead of staying pinned to r1, this would 404 (`score-not-found`)."""
    base = f"http://127.0.0.1:{server.server_port}?pollMs=250"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base)
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")

        row = page.locator('[data-score-id="r1-score-4-a"]')
        row.get_by_test_id("open-event-link").click()
        expect(page.get_by_test_id("event-detail")).to_be_visible()
        expect(page.get_by_test_id("event-scores-table")).to_be_visible()

        state.set_current("r2")
        expect(page.get_by_test_id("release-changed-notice")).to_be_visible(timeout=5000)

        page.get_by_test_id("open-score-link").first.click()
        expect(page.get_by_test_id("score-detail")).to_be_visible()
        expect(page.get_by_test_id("score-id-value")).to_contain_text("r1-score-4-a")
        expect(page.get_by_test_id("score-not-found")).to_have_count(0)
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")
    finally:
        context.close()


def test_score_detail_null_shows_missing_zero_shows_zero_and_other_category(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/#/release/r1/scores/r1-score-0-a")
        expect(page.get_by_test_id("score-detail")).to_be_visible()

        entry_cost_row = page.locator('[data-testid="field-row"][data-field="entry_cost"]')
        expect(entry_cost_row.get_by_test_id("field-value")).to_have_text("—")

        exp_pnl_row = page.locator('[data-testid="field-row"][data-field="exp_pnl_model"]')
        expect(exp_pnl_row.get_by_test_id("field-value")).to_have_text("0")

        # "note" is not in the mapping spec's category table -- falls back
        # to the "other" category, per guide "otherwise alphabetically".
        categories = page.get_by_test_id("field-group-category").all_inner_texts()
        assert "other" in categories
        note_row = page.locator('[data-testid="field-row"][data-field="note"]')
        expect(note_row).to_be_visible()
    finally:
        context.close()


def test_score_detail_refusal_shown(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/#/release/r1/scores/r1-score-1-a")
        expect(page.get_by_test_id("score-raw-verdict")).to_contain_text("false")
        detail_row = page.locator('[data-testid="field-row"][data-field="detail"]')
        expect(detail_row.get_by_test_id("field-value")).to_have_text("entry cost exceeds ceiling")
    finally:
        context.close()


def test_score_detail_dynsv_choice_shown(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/#/release/r1/scores/r1-score-2-b")
        chosen_row = page.locator('[data-testid="field-row"][data-field="chosen_strategy"]')
        expect(chosen_row.get_by_test_id("field-value")).to_have_text("STR-THRU")
        margin_row = page.locator('[data-testid="field-row"][data-field="chosen_margin"]')
        expect(margin_row.get_by_test_id("field-value")).to_have_text("0.014")
    finally:
        context.close()


def test_payoff_svg_points_match_fixture_exactly(browser, server, state):
    """P3-3b deliverable 6: "payoff SVG point count equals the fixture" --
    and, more strongly, each point's own data matches the fixture's x/y
    arrays exactly (no interpolation, no dropped/added point)."""
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/#/release/r1/scores/r1-score-0-a")
        expect(page.get_by_test_id("payoff-svg")).to_be_visible()
        points = page.get_by_test_id("payoff-point")
        assert points.count() == 5
        xs = [float(points.nth(i).get_attribute("data-x") or "nan") for i in range(5)]
        ys = [float(points.nth(i).get_attribute("data-y") or "nan") for i in range(5)]
        assert xs == [90.0, 95.0, 100.0, 105.0, 110.0]
        assert ys == [5.0, 5.0, 0.0, 0.0, 0.0]
    finally:
        context.close()


def test_payoff_curve_empty_state_renders(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/#/release/r1/scores/r1-score-2-a")
        expect(page.get_by_test_id("payoff-empty")).to_be_visible()
        expect(page.get_by_test_id("payoff-svg")).to_have_count(0)
    finally:
        context.close()


def test_payoff_curve_missing_state_renders(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/#/release/r1/scores/r1-score-1-a")
        expect(page.get_by_test_id("payoff-missing")).to_be_visible()
    finally:
        context.close()


def test_score_detail_shows_ids_and_collapsed_engine_evidence(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/#/release/r1/scores/r1-score-0-a")
        expect(page.get_by_test_id("score-id-value")).to_contain_text("r1-score-0-a")
        expect(page.get_by_test_id("score-release-id-value")).to_contain_text("r1")

        details = page.get_by_test_id("engine-evidence")
        expect(details).to_be_visible()
        assert details.get_attribute("open") is None  # collapsed by default
        expect(page.get_by_test_id("engine-evidence-json")).to_contain_text("raw_model_state_ref")
    finally:
        context.close()


def test_unknown_score_shows_not_found(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/#/release/r1/scores/does-not-exist")
        expect(page.get_by_test_id("score-not-found")).to_be_visible()
    finally:
        context.close()


def test_unknown_event_shows_not_found(browser, server, state):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/#/release/r1/events/does-not-exist")
        expect(page.get_by_test_id("event-not-found")).to_be_visible()
    finally:
        context.close()


def _mock_get(server, path: str) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        headers={"Authorization": "Bearer " + TOKEN})
    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_release_id_required_on_event_scores_and_score_routes(server, state):
    """Coordinator's P3-2 contract decision: `release_id` is required on
    `GET /api/v1/events/{id}/scores` and `GET /api/v1/scores/{id}`, 400
    `RELEASE_ID_REQUIRED` if missing -- distinct from a present-but-unknown
    release id (404 `UNKNOWN_RELEASE`). Hits the mock directly (no browser
    needed) since this is a server-contract check, not a UI-rendering one."""
    status, body = _mock_get(server, "/api/v1/events/evt-r1-000/scores")
    assert status == 400
    assert body["code"] == "RELEASE_ID_REQUIRED"
    assert "status" not in body and "title" not in body  # real Problem shape only

    status, body = _mock_get(server, "/api/v1/scores/r1-score-0-a")
    assert status == 400
    assert body["code"] == "RELEASE_ID_REQUIRED"

    status, body = _mock_get(server, "/api/v1/events/evt-r1-000/scores?release_id=does-not-exist")
    assert status == 404
    assert body["code"] == "UNKNOWN_RELEASE"

    status, body = _mock_get(server, "/api/v1/events/evt-r1-000/scores?release_id=r1")
    assert status == 200
    assert isinstance(body, list)  # bare array, no envelope


def test_no_token_in_url_or_local_storage_on_detail_views(browser, server):
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/#/release/r1/scores/r1-score-0-a")
        expect(page.get_by_test_id("score-detail")).to_be_visible()
        assert TOKEN not in page.url
        storage = page.evaluate(
            "() => Object.entries(localStorage).map(([k,v]) => k + '=' + v).join(';')")
        assert TOKEN not in storage
    finally:
        context.close()
