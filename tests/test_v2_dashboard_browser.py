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
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import expect

from tests.fixtures.v2_ui_mock_api import build_default_state, serve_in_thread

pytestmark = [pytest.mark.xdist_group("serial"), pytest.mark.browser]  # drives a real Playwright browser or needs node/npm (ui/ build)

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = REPO_ROOT / "ui"
TOKEN = "browser-secret"


@pytest.fixture(scope="module")
def dist_dir(ui_dist_dir):
    """``ui/dist``, built (and ``ui/node_modules`` installed first if
    needed) by the shared session-scoped ``ui_dist_dir`` fixture in
    ``tests/conftest.py`` -- guide §8 P3-3 exit command, now fixture-safe in
    a fresh worktree with no ``ui/node_modules`` at all."""
    return ui_dist_dir


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


def test_operations_status_marks_old_pinned_board_and_missing_scheduled_session(browser, server, state):
    """Issue reproduction: a 24-day-old operations-status sidecar that reports a
    missing scheduled session must never read as ``current``, and the board stays
    pinned to the release it loaded even after ``current`` moves on -- asserted
    without a click or a reload."""
    operations_status = {
        "schema_version": "operations_status.v1.0",
        "generated_at": "2026-09-10T00:00:00Z",
        "release_id": "r1",
        "attempted_release_id": "r1",
        "requested_session": "eng-night-2026-09-10",
        "resolved_session": "eng-night-2026-09-10",
        "engineering_history": [{"occurrence": "2026-09-20", "status": "unknown"}],
        "stale": False,
        "withheld": False,
        "failed_update": False,
    }
    # The route reads this holder at fulfill time, so the payload can be
    # swapped mid-test and the next 100ms operations poll picks it up.
    served = {"status": operations_status}
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    page.route(
        "**/api/v1/operations",
        lambda route: route.fulfill(
            status=200, content_type="application/json",
            body=json.dumps(served["status"])))
    # Freeze only the browser wall clock (not setInterval -- the release poll
    # still ticks in real time); `?pollMs=100` accelerates the polls instead.
    page.add_init_script(
        'Date.now = function () { return Date.parse("2026-10-04T00:00:00Z"); };')
    try:
        page.goto(base + "/?pollMs=100")

        expect(page.get_by_test_id("release-id")).to_contain_text("r1")
        status = page.get_by_test_id("operations-status")
        expect(status).to_contain_text("2026-09-20")  # the missing occurrence is named
        expect(status).to_contain_text("scheduled observations missing")  # the explicit reason phrase
        expect(status).to_contain_text("24d")  # frozen clock -> observation age
        expect(status).not_to_contain_text("operations: current")
        identities = page.get_by_test_id("operations-identities")
        expect(identities).to_contain_text("board pin r1")
        expect(identities).to_contain_text("health/status-described r1")
        expect(identities).to_contain_text("latest published r1")
        sessions = page.get_by_test_id("operations-sessions")
        expect(sessions).to_contain_text("requested session eng-night-2026-09-10")
        expect(sessions).to_contain_text("resolved session eng-night-2026-09-10")
        expect(page.get_by_test_id("operations-attempt")).to_contain_text("attempted release r1")

        state.set_current("r2")
        stale = page.get_by_test_id("operations-board-stale")
        expect(stale).to_be_visible()
        expect(stale).to_contain_text("r1")
        expect(stale).to_contain_text("r2")
        expect(stale).to_contain_text("24d")  # still shows the observation age
        expect(stale.get_by_role("button")).to_be_visible()
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")

        # A fresh status write lands while the clock stays frozen at
        # 2026-10-04T00:00:00Z: generated_at is now 29m old. The sub-hour
        # age must render in minutes on the status row, and the stale-board
        # notice must label it explicitly as the operations-status
        # observation age, adjacent to the latest published id -- not leave
        # it unqualified and not round it to "an hour".
        served["status"] = {**operations_status, "generated_at": "2026-10-03T23:31:00Z"}
        expect(status).to_contain_text("observed 29m ago")
        expect(stale).to_contain_text("r2 (operations status observed 29m ago)")
    finally:
        context.close()


# --------------------------------------------------------------------------
# issue #161 regressions: a failed current poll must not keep presenting the
# last-known identity as authoritative; a rollback to the pin clears the drift
# hint; malformed or future-resolved session evidence renders unknown
# --------------------------------------------------------------------------


def test_current_release_poll_failure_makes_operations_unknown(browser, server, state):
    """The r2-pinned board with valid operations evidence agreeing with r2
    renders operations current; a later `/api/v1/releases/current` POLL
    FAILURE must make operations unknown -- publication identity is only ever
    the latest SUCCESSFUL discovery, so the old "r2" is never kept on being
    presented as the authoritative latest published release. The pin and the
    loaded board survive, nothing repins or swaps, and the next scheduled
    poll recovers on its own (no immediate browser-invented retry)."""
    state.set_current("r2")
    status = {
        "schema_version": "operations_status.v1.0",
        "generated_at": "2026-10-03T23:31:00Z",
        "release_id": "r2",
        "attempted_release_id": "r2",
        "requested_session": "eng-night-2026-10-03",
        "resolved_session": "eng-night-2026-10-03",
        "engineering_history": [{"occurrence": "2026-10-03", "status": "pass"}],
        "stale": False,
        "withheld": False,
        "failed_update": False,
    }
    problem = {
        "code": "SIMULATED_ERROR", "category": "internal", "retryable": True,
        "message": "simulated current-release poll failure", "stage": None,
        "trace_id": None, "dependency_refs": [], "retry_after_seconds": None,
        "diagnostic_ref": None, "details": {}, "schema_version": "problem.v1.0",
    }

    def _fail_current(route):
        route.fulfill(status=503, content_type="application/json", body=json.dumps(problem))

    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    page.route(
        "**/api/v1/operations",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(status)))
    page.add_init_script(
        'Date.now = function () { return Date.parse("2026-10-04T00:00:00Z"); };')
    try:
        page.goto(base + "/?pollMs=100")

        # Initial current discovery succeeded: r2 pinned, and with status,
        # published and pin all naming r2, operations renders current.
        expect(page.get_by_test_id("release-id")).to_contain_text("r2")
        expect(page.get_by_test_id("operations-status")).to_contain_text("operations: current")
        expect(page.get_by_test_id("operations-identities")).to_contain_text("latest published r2")

        # A later /api/v1/releases/current poll now fails outright.
        page.route("**/api/v1/releases/current", handler=_fail_current)

        # The pin and the loaded board survive the failed poll...
        expect(page.get_by_test_id("release-id")).to_contain_text("r2")
        expect(page.get_by_test_id("event-table")).to_be_visible()
        expect(page.get_by_test_id("release-changed-notice")).to_have_count(0)
        # ...but operations is no longer current: a failed read never turns
        # into an authoritative published identity, so latest published is
        # unavailable and the stale "r2" is not presented as authoritative.
        identities = page.get_by_test_id("operations-identities")
        expect(page.get_by_test_id("operations-status")).to_contain_text("operations: unknown")
        expect(page.get_by_test_id("operations-status")).not_to_contain_text("operations: current")
        expect(identities).to_contain_text("latest published unavailable")
        expect(identities).not_to_contain_text("latest published r2")
        expect(page.get_by_test_id("operations-board-stale")).to_have_count(0)

        # The next scheduled poll (still 100ms away, no reload, no extra
        # retry) recovers: the latest successful discovery is r2 again.
        page.unroute("**/api/v1/releases/current", handler=_fail_current)
        expect(page.get_by_test_id("operations-status")).to_contain_text("operations: current")
        expect(identities).to_contain_text("latest published r2")
    finally:
        context.close()


def test_current_release_rollback_to_pin_clears_drift_hint(browser, server, state):
    """Deep link pins r1 while `current` initially reports r2, so the r2 drift
    hint (stale-board notice + "latest published r2") is up and operations is
    unknown. When the next current poll reports r1 again -- a rollback to the
    pin -- the latest SUCCESSFUL identity replaces the stale r2: the drift
    hint clears, latest published shows r1, and only now, with pin, published
    and status all agreeing on r1, may operations be labelled current."""
    state.set_current("r2")
    status = {
        "schema_version": "operations_status.v1.0",
        "generated_at": "2026-10-03T23:31:00Z",
        "release_id": "r1",
        "attempted_release_id": "r1",
        "requested_session": "eng-night-2026-10-03",
        "resolved_session": "eng-night-2026-10-03",
        "engineering_history": [{"occurrence": "2026-10-03", "status": "pass"}],
        "stale": False,
        "withheld": False,
        "failed_update": False,
    }
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    page.route(
        "**/api/v1/operations",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(status)))
    page.add_init_script(
        'Date.now = function () { return Date.parse("2026-10-04T00:00:00Z"); };')
    try:
        page.goto(base + "/?pollMs=100#/release/r1")

        # Load-time drift: pin r1, published r2, status describing r1 -- the
        # identities do not all agree, so the hint is up and operations is
        # unknown.
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")
        drift = page.get_by_test_id("operations-board-stale")
        expect(drift).to_be_visible()
        expect(drift).to_contain_text("r2")
        expect(page.get_by_test_id("operations-identities")).to_contain_text("latest published r2")
        expect(page.get_by_test_id("operations-status")).to_contain_text("operations: unknown")

        # `current` rolls back to the pinned release; the next scheduled poll
        # sees it and the old r2 hint must go away.
        state.set_current("r1")
        expect(drift).to_have_count(0)
        identities = page.get_by_test_id("operations-identities")
        expect(identities).to_contain_text("latest published r1")
        expect(identities).not_to_contain_text("latest published r2")
        expect(page.get_by_test_id("operations-status")).to_contain_text("operations: current")
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")
    finally:
        context.close()


def test_malformed_or_future_resolved_session_is_unknown(browser, server, state):
    """Operations session evidence the board cannot stand behind renders
    unknown, never current, on an otherwise fully valid r1 status: nonempty
    but malformed requested/resolved identifiers, malformed identifiers whose
    only well-formed part is a trailing ISO date, and ISO date-suffixed
    identifiers whose resolved date is later than the requested date. Neither
    may crash the page or disturb the pinned r1 board."""
    base_status = {
        "schema_version": "operations_status.v1.0",
        "generated_at": "2026-10-03T23:31:00Z",
        "release_id": "r1",
        "attempted_release_id": "r1",
        "requested_session": "eng-night-2026-10-03",
        "resolved_session": "eng-night-2026-10-03",
        "engineering_history": [{"occurrence": "2026-10-03", "status": "pass"}],
        "stale": False,
        "withheld": False,
        "failed_update": False,
    }
    # The route reads this holder at fulfill time, so each 100ms operations
    # poll picks up the swapped session evidence without a reload.
    served = {"status": base_status}
    errors: list[str] = []
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.route(
        "**/api/v1/operations",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(served["status"])))
    page.add_init_script(
        'Date.now = function () { return Date.parse("2026-10-04T00:00:00Z"); };')
    try:
        page.goto(base + "/?pollMs=100")
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")
        # Baseline: with well-formed, agreeing session evidence the fully
        # valid status renders current -- the unknowns below are caused by
        # the session evidence alone.
        expect(page.get_by_test_id("operations-status")).to_contain_text("operations: current")

        # Regression: a nonempty malformed identifier that merely ENDS in an
        # ISO date. Identical requested/resolved values must not be read as
        # agreeing session evidence -- the date tail never rescues an unknown
        # prefix, so this is unknown, never current, and the pin stands.
        served["status"] = {**base_status,
                            "requested_session": "unexpected-2026-10-03",
                            "resolved_session": "unexpected-2026-10-03"}
        expect(page.get_by_test_id("operations-status")).to_contain_text("operations: unknown")
        expect(page.get_by_test_id("operations-status")).not_to_contain_text("operations: current")
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")
        expect(page.get_by_test_id("event-table")).to_be_visible()

        # Nonempty malformed requested/resolved session identifiers.
        served["status"] = {**base_status,
                            "requested_session": "!!!not-a-session!!!",
                            "resolved_session": "????"}
        expect(page.get_by_test_id("operations-status")).to_contain_text("operations: unknown")
        expect(page.get_by_test_id("operations-status")).not_to_contain_text("operations: current")
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")
        expect(page.get_by_test_id("event-table")).to_be_visible()

        # ISO date-suffixed identifiers whose resolved date is LATER than the
        # requested date -- evidence no real walk-back can produce.
        served["status"] = {**base_status,
                            "requested_session": "eng-night-2026-10-03",
                            "resolved_session": "eng-night-2026-10-04"}
        expect(page.get_by_test_id("operations-status")).to_contain_text("operations: unknown")
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")
        expect(page.get_by_test_id("event-table")).to_be_visible()

        # Neither bad payload reached the browser as an exception.
        assert errors == []
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
# issue #408 regressions: a failed current poll clears the published identity
# rather than reusing the drifted id; a rollback to the pin clears the drift
# hint; and out-of-order polls never let a superseded reply publish
# --------------------------------------------------------------------------


def _pinned_board_page(browser, server):
    """An r1-pinned board, frozen clock, valid r1 operations sidecar."""
    status = {
        "schema_version": "operations_status.v1.0", "generated_at": "2026-10-04T00:00:00Z",
        "release_id": "r1", "attempted_release_id": "r1",
        "requested_session": "eng-night-2026-10-03", "resolved_session": "eng-night-2026-10-03",
        "engineering_history": [{"occurrence": "2026-10-03", "status": "pass"}],
        "stale": False, "withheld": False, "failed_update": False,
    }
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    page.route("**/api/v1/operations", lambda route: route.fulfill(
        status=200, content_type="application/json", body=json.dumps(status)))
    page.add_init_script(
        'Date.now = function () { return Date.parse("2026-10-04T00:00:00Z"); };')
    return context, page, base


def test_current_poll_failure_after_drift_clears_published_identity(browser, server, state):
    context, page, base = _pinned_board_page(browser, server)
    try:
        page.goto(base + "/?pollMs=100")
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")

        state.set_current("r2")
        expect(page.get_by_test_id("release-changed-notice")).to_contain_text("r2")

        # A failed read is never an authority: the r2 drift goes with it.
        page.route("**/api/v1/releases/current", lambda route: route.fulfill(
            status=503, content_type="application/json", body="{}"))

        expect(page.get_by_test_id("release-changed-notice")).to_have_count(0)
        expect(page.get_by_test_id("operations-identities")).to_contain_text(
            "latest published unavailable")
        status = page.get_by_test_id("operations-status")
        expect(status).to_contain_text("operations: unknown")
        expect(status).to_contain_text("latest published release identity unavailable")
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")
    finally:
        context.close()


def test_current_switch_then_rollback_to_pin_clears_drift_hint(browser, server, state):
    context, page, base = _pinned_board_page(browser, server)
    try:
        page.goto(base + "/?pollMs=100")
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")

        state.set_current("r2")
        expect(page.get_by_test_id("release-changed-notice")).to_contain_text("r2")

        state.set_current("r1")
        expect(page.get_by_test_id("release-changed-notice")).to_have_count(0)
        expect(page.get_by_test_id("operations-identities")).to_contain_text("latest published r1")
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")
    finally:
        context.close()


def test_out_of_order_current_polls_keep_newest_successful_identity(browser, server, state):
    held: list = []
    seen = [0]
    r2_release = state.releases["r2"].release

    def _ordered_current(route):
        seen[0] += 1
        if seen[0] == 2:
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps(r2_release))
        else:
            held.append(route)  # every other poll waits: r2 lands first

    context, page, base = _pinned_board_page(browser, server)
    try:
        page.goto(base + "/?pollMs=100")
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")

        page.route("**/api/v1/releases/current", handler=_ordered_current)
        expect(page.get_by_test_id("release-changed-notice")).to_contain_text("r2")
        expect(page.get_by_test_id("operations-identities")).to_contain_text("latest published r2")

        with page.expect_event(
            "requestfinished",
            predicate=lambda request: request.url.endswith("/api/v1/releases/current"),
        ):
            held[0].fulfill(status=200, content_type="application/json",
                            body=json.dumps({**r2_release, "release_id": "r3"}))
        # Let the resolved fetch continuation and React commit reach a paint.
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")

        expect(page.get_by_test_id("release-changed-notice")).to_contain_text("r2")
        identities = page.get_by_test_id("operations-identities")
        expect(identities).to_contain_text("latest published r2")
        expect(identities).not_to_contain_text("latest published r3")
        expect(page.get_by_test_id("release-id")).to_contain_text("r1")
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
# unit regression: percentage-point fields and counts render in their own
# units (test-first: expected RED on the current UI, which re-scales pp
# values by 100 and renders coverage counts as percentages)
# --------------------------------------------------------------------------


_UNIT_FIELD_OVERRIDES = {
    "driver_forecast": 5.0,        # already percentage points -> "5.0%"
    "market_implied_move": 5.0,    # already percentage points -> "5.0%"
    "expected_return_model": 0.05, # fraction -> "5.0%"
    "planned_population": 121,     # count -> "121", never a %
    "entry_premium": None,         # null -> "—"
}


def _override_fields_by_key(node, _seen: set[int] | None = None) -> None:
    """Set every `_UNIT_FIELD_OVERRIDES` key by name, recursively, across all
    dicts, lists and plain objects reachable from ``node`` -- without relying
    on the mock fixture's internal structure."""
    if node is None or isinstance(node, (str, bytes, bytearray, int, float,
                                         complex, bool, type, types.ModuleType)):
        return
    if callable(node):
        return
    if _seen is None:
        _seen = set()
    if id(node) in _seen:
        return
    _seen.add(id(node))
    if isinstance(node, dict):
        for key in list(node):
            if key == "coverage_summary" and isinstance(node[key], dict):
                node[key]["compared_population"] = 87
                _override_fields_by_key(node[key], _seen)
            elif key in _UNIT_FIELD_OVERRIDES:
                node[key] = _UNIT_FIELD_OVERRIDES[key]
            else:
                _override_fields_by_key(node[key], _seen)
    elif isinstance(node, (list, tuple, set, frozenset)):
        for item in node:
            _override_fields_by_key(item, _seen)
    else:
        attrs = getattr(node, "__dict__", None)
        if isinstance(attrs, dict):
            for key in list(attrs):
                if key in _UNIT_FIELD_OVERRIDES:
                    try:
                        setattr(node, key, _UNIT_FIELD_OVERRIDES[key])
                    except (AttributeError, TypeError):
                        pass
                else:
                    _override_fields_by_key(attrs[key], _seen)


def test_percentage_point_and_count_fields_render_in_their_own_units(
        browser, server, state):
    """driver_forecast / market_implied_move are percentage points: 5.0 must
    render as "5.0%", never the re-scaled "500.0%"; expected_return_model is
    a fraction: 0.05 -> "5.0%"; null entry_premium -> "—"; coverage
    planned_population is a count: its .coverage-item has exact text
    "planned_population: 121" with no "%" (other fractional coverage items
    legitimately render "%"). Asserted on the board row and again on the
    event detail row opened from it."""
    _override_fields_by_key(state)
    base = f"http://127.0.0.1:{server.server_port}"
    context, page = _authed_page(browser, server, base)
    try:
        page.goto(base + "/")
        row = page.locator('[data-score-id="r1-score-0-a"]')
        expect(row).to_be_visible()
        expect(row.get_by_test_id("driver-forecast-cell")).to_have_text("5.0%")
        expect(row.locator("td").nth(6)).to_have_text("5.0%")  # market implied move
        expect(row).not_to_contain_text("500.0%")
        expect(row.get_by_test_id("expected-return-cell")).to_contain_text("5.0%")
        expect(row.get_by_test_id("entry-premium-cell")).to_have_text("—")

        coverage = page.get_by_test_id("release-coverage")
        population_item = coverage.locator(
            ".coverage-item", has_text="planned_population")
        expect(population_item).to_have_text("planned_population: 121")
        expect(population_item).not_to_contain_text("%")

        compared_item = coverage.locator(
            ".coverage-item", has_text="compared_population")
        expect(compared_item).to_have_text("compared_population: 87")
        expect(compared_item).not_to_contain_text("%")

        row.get_by_test_id("open-event-link").click()
        expect(page.get_by_test_id("event-detail")).to_be_visible()
        expect(page.get_by_test_id("event-scores-table")).to_be_visible()
        detail_row = page.locator(
            '[data-testid="event-score-row"][data-score-id="r1-score-0-a"]')
        expect(detail_row).to_be_visible()
        expect(detail_row.locator("td").nth(2)).to_have_text("5.0%")  # driver forecast
        expect(detail_row.locator("td").nth(3)).to_have_text("5.0%")  # market implied move
        expect(detail_row).not_to_contain_text("500.0%")
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
