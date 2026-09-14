"""P3-3c: the merged React UI against the REAL read API, end to end.

``ui/`` (P3-3a/P3-3b) was built and tested only against a stdlib mock
(``tests/fixtures/v2_ui_mock_api.py``) shaped from ``engine/v2/contracts/
serving.py``; ``engine/v2/serving/api.py`` (P3-2) is the real FastAPI server.
This module wires a real synthetic release all the way through the real
pipeline -- ``engine.v2.serving.projections.build_candidate`` (same
technique ``tests/test_v2_serving_projections.py`` and ``tests/
test_v2_serving_publication_binding.py`` use), a real fenced ops publish
(``engine.v2.ops.publication.stage_release``/``publish_local``, same
technique as ``tests/test_v2_serving_publication_binding.py``), and a real
``uvicorn.Server`` running ``engine.v2.serving.api.create_app(...)`` -- then
drives a real Playwright browser against ``ui/dist`` served same-origin.

**Static serving choice**: a ``StaticFiles`` mount is added to the app
returned by ``create_app`` ONLY in this test module (``_serve_ui``, below),
never inside ``engine/v2/serving/api.py`` itself -- the routing is
hash-based (``#/release/<id>/...``), so the server only ever needs to serve
``index.html`` and the built asset files, never a path-based SPA fallback.
No production static route is added; §6/§7 do not ask for one.

No ``engine/v2/ops`` file is touched, no ``checks/rearchitecture_phase3_*``,
no ``/root/phase2-shadow-ops`` or ``/root/phase2-heavy-*`` data -- entirely
synthetic, tmp-path scoped.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.staticfiles import StaticFiles
from playwright.sync_api import expect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import PreviewRelease  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore  # noqa: E402
from engine.v2.serving import projections  # noqa: E402
from engine.v2.serving.api import create_app  # noqa: E402
from tests.ops_support import catalog, enqueue_claim  # noqa: E402
from tests.test_v2_serving_api import _get, _start, _stop  # noqa: E402
from tests.test_v2_serving_projections import (  # noqa: E402
    _bundle,
    _compact,
    _event_row,
    _events_snapshot,
    _preview_input,
    _row,
    _score_doc,
    _serving,
)
from tests.test_v2_serving_publication_binding import (  # noqa: E402
    _stage_and_publish,
    _stage_files,
)

pytestmark = pytest.mark.xdist_group("serial")

TOKEN = "integration-secret-9c2e"
TOKEN_COOKIE = "operations_token"
N_EVENTS = 55  # > the board's hardcoded 50-per-page default (App.tsx DEFAULT_LIMIT)

# The 5-point payoff curve mirrors the mock fixture's own
# ``test_payoff_svg_points_match_fixture_exactly`` shape, so the same
# assertion style applies against the real API.
_PAYOFF_CURVE = {
    "x": [90.0, 95.0, 100.0, 105.0, 110.0], "y": [5.0, 5.0, 0.0, 0.0, 0.0],
    "max": 5.0, "min": 0.0, "strikes": [], "shape": "centre",
}


# --------------------------------------------------------------------------
# a real, 55-event synthetic release ("release A") plus a small second one
# ("release B", generation 2) -- built the same way
# tests/test_v2_serving_publication_binding.py does
# --------------------------------------------------------------------------


def _ticker(i: int) -> str:
    return f"TICK{i:03d}"


def _event_date(i: int) -> datetime:
    return datetime(2024, 1, 1) + timedelta(days=i)


def _release_a_rows() -> list[dict]:
    rows = []
    for i in range(N_EVENTS):
        date_str = _event_date(i).date().isoformat()
        ticker = _ticker(i)
        if i == 0:
            # Null field vs. a real value elsewhere on the same row --
            # `entry_cost`/`entry_cost_pct` null, never silently 0.
            row = _row(ticker=ticker, event_date=date_str, strike=100.0,
                       entry_cost=None, entry_cost_pct=None)
        elif i == 1:
            # A refusal row: gate_pass False, a `detail` reason, no legs.
            row = _row(ticker=ticker, event_date=date_str, strike=100.0,
                       gate_pass=False, detail="entry cost exceeds ceiling",
                       exp_pnl_model=None, exp_pnl_analog=None, gate_score=None,
                       entry_cost=None, entry_cost_pct=None, legs=[])
        elif i == 2:
            # The row with a distinctive, known payoff curve.
            row = _row(ticker=ticker, event_date=date_str, strike=100.0, exp_pnl_model=0.0123)
        else:
            row = _row(ticker=ticker, event_date=date_str, strike=100.0 + i,
                       exp_pnl_model=round(0.001 * i, 6))
        rows.append(row)
    return rows


def _release_a_compacts(rows: list[dict]) -> list[dict]:
    compacts = []
    for i, row in enumerate(rows):
        if i == 1:
            compacts.append(_compact(row, payoff_curve=None))  # no curve was saved
        elif i == 2:
            compacts.append(_compact(row, payoff_curve=dict(_PAYOFF_CURVE)))
        else:
            compacts.append(_compact(row))
    return compacts


def _build_releases(tmp_path):
    """One shared Phase-2 snapshot (55 events for release A, 2 more for
    release B), a real synthetic serving index, and two committed
    candidates over disjoint score subsets -- ``release_a``/``release_b``,
    reused by ``tests/test_v2_serving_publication_binding.py``'s own
    ``_releases`` for the same reason: two distinct generations from one
    snapshot, without needing two Phase-2 catalogs."""
    (tmp_path / "phase2").mkdir()
    event_rows = [_event_row(f"evt-{i:03d}", _ticker(i), _event_date(i)) for i in range(N_EVENTS)]
    b_dates = [datetime(2024, 3, 1) + timedelta(days=i) for i in range(2)]
    event_rows += [_event_row(f"evt-b-{i:03d}", f"TICKB{i}", b_dates[i]) for i in range(2)]

    conn, store, snap = _events_snapshot(tmp_path / "phase2", event_rows)
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)

    rows_a = _release_a_rows()
    compacts_a = _release_a_compacts(rows_a)
    release_a = projections.build_candidate(
        _preview_input(), _score_doc(rows=rows_a), _bundle(*compacts_a),
        repository=repo, snapshot_ref=snap, store=serving_store, conn=serving_conn,
        requested_as_of="2024-01-04", resolved_as_of="2024-01-04")
    assert isinstance(release_a, PreviewRelease), release_a

    rows_b = [_row(ticker=f"TICKB{i}", event_date=b_dates[i].date().isoformat(), strike=200.0 + i)
             for i in range(2)]
    compacts_b = [_compact(r) for r in rows_b]
    release_b = projections.build_candidate(
        _preview_input(source_release_id="rel_2"), _score_doc(rows=rows_b), _bundle(*compacts_b),
        repository=repo, snapshot_ref=snap, store=serving_store, conn=serving_conn,
        requested_as_of="2024-03-01", resolved_as_of="2024-03-01")
    assert isinstance(release_b, PreviewRelease), release_b

    return conn, repo, serving_conn, serving_store, release_a, release_b


def _score_id_for_event(serving_conn, release_id: str, event_id: str) -> str:
    item = projections.get_event(serving_conn, release_id, event_id)
    assert item is not None and len(item.scores) == 1
    return item.scores[0].score_id


def _fmt_unknown(value) -> str:
    """Python mirror of ``ui/src/format.ts`` ``fmtUnknown`` -- only for the
    handful of clean, hand-picked field values this test uses (whole-number
    or short-decimal floats), so no general JS-number-to-string emulation is
    needed."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value)


def _serve_ui(app, dist_dir: Path):
    """Adds a ``StaticFiles`` mount for the built UI to a real ``create_app``
    instance -- ONLY here, in the test, per this module's docstring. Mounted
    at ``/`` AFTER every ``/api/v1/...`` route is already registered, so
    FastAPI/Starlette's in-order route matching tries the specific API
    routes first and only falls through to static files for anything else
    (``/`` and ``/assets/...`` -- routing is hash-based, so nothing else is
    ever requested from the server)."""
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="ui-dist")
    return app


def _authed_context(browser, base: str):
    context = browser.new_context()
    context.add_cookies([{"name": TOKEN_COOKIE, "value": TOKEN, "url": base}])
    return context


# --------------------------------------------------------------------------
# the full flow
# --------------------------------------------------------------------------


def test_dashboard_against_real_api_full_flow(tmp_path, ui_dist_dir, browser):
    conn, repo, serving_conn, serving_store, release_a, release_b = _build_releases(tmp_path)
    ops_conn, clock, supervisor = catalog(tmp_path)
    ops_store = ArtifactStore(tmp_path / "opsstore")
    target = tmp_path / "opsrelease"
    scope = "shadow"
    try:
        # Publish release A as generation 1 -- the real fenced ops publish
        # path, not a test-only shortcut.
        binding_a = projections.projection_binding(serving_conn, release_a.release_id)
        files_a = _stage_files(ops_store, "release-a", binding_a)
        claim1 = enqueue_claim(ops_conn, clock, supervisor, key="pub-gen1")
        _stage_and_publish(ops_conn, ops_store, claim1, "rel-gen1", "2026-09-14", files_a,
                           expected_current=None, target=target, scope=scope, clock=clock,
                           generation="gen1")

        app = _serve_ui(
            create_app(serving_db=str(tmp_path / "serving" / "serving.sqlite"),
                      store_root=str(tmp_path / "serving" / "objects"),
                      serving_root=str(tmp_path / "serving"),
                      publication_root=str(target), token=TOKEN),
            ui_dist_dir)
        server, thread, base = _start(app)
        try:
            null_score_id = _score_id_for_event(serving_conn, release_a.release_id, "evt-000")
            refusal_score_id = _score_id_for_event(serving_conn, release_a.release_id, "evt-001")
            payoff_score_id = _score_id_for_event(serving_conn, release_a.release_id, "evt-002")

            # -- 3 rows' displayed values equal the real API's own
            #    display_record, fetched independently over real HTTP --
            #    including the null-field row and the refusal row.
            for score_id, fields in (
                (null_score_id, ("entry_cost", "entry_cost_pct", "ticker")),
                (refusal_score_id, ("gate_pass", "detail", "exp_pnl_model")),
                (payoff_score_id, ("exp_pnl_model", "gate_pass", "strategy")),
            ):
                code, body, _ = _get(base, f"/api/v1/scores/{score_id}", token=TOKEN,
                                     params={"release_id": release_a.release_id})
                assert code == 200
                display_record = json.loads(body)["display_record"]

                context = _authed_context(browser, base)
                page = context.new_page()
                try:
                    page.goto(f"{base}/#/release/{release_a.release_id}/scores/{score_id}")
                    expect(page.get_by_test_id("score-detail")).to_be_visible()
                    for field in fields:
                        row = page.locator(f'[data-testid="field-row"][data-field="{field}"]')
                        expect(row.get_by_test_id("field-value")).to_have_text(
                            _fmt_unknown(display_record[field]))
                finally:
                    context.close()

            # -- board / pagination / event -> score / payoff SVG, all in
            #    one browsing session so the mid-session release switch
            #    later in this test can rely on the SAME page.
            context = _authed_context(browser, f"{base}/?pollMs=250")
            page = context.new_page()
            try:
                page.goto(f"{base}/?pollMs=250")
                expect(page.get_by_test_id("release-id")).to_contain_text(release_a.release_id)
                assert TOKEN not in page.url
                storage = page.evaluate(
                    "() => Object.entries(localStorage).map(([k,v]) => k + '=' + v).join(';')")
                assert TOKEN not in storage

                expect(page.get_by_test_id("event-table")).to_be_visible()
                expect(page.get_by_test_id("pagination-count")).to_contain_text(f"of {N_EVENTS}")

                page.get_by_test_id("page-next").click()
                page.wait_for_timeout(200)
                page2_rows = page.get_by_test_id("score-row").count()
                assert 0 < page2_rows < 50  # the remaining N_EVENTS - 50 rows
                expect(page.get_by_test_id("page-next")).to_be_disabled()
                page.get_by_test_id("page-prev").click()
                page.wait_for_timeout(200)
                expect(page.get_by_test_id("event-table")).to_be_visible()

                # open event -> score for the distinctive-payoff row.
                row = page.locator(f'[data-score-id="{payoff_score_id}"]')
                row.get_by_test_id("open-event-link").click()
                expect(page.get_by_test_id("event-detail")).to_be_visible()
                expect(page.get_by_test_id("event-scores-table")).to_be_visible()
                page.get_by_test_id("open-score-link").first.click()
                expect(page.get_by_test_id("score-detail")).to_be_visible()
                expect(page.get_by_test_id("score-id-value")).to_contain_text(payoff_score_id)

                expect(page.get_by_test_id("payoff-svg")).to_be_visible()
                points = page.get_by_test_id("payoff-point")
                assert points.count() == len(_PAYOFF_CURVE["x"])
                xs = [float(points.nth(i).get_attribute("data-x") or "nan")
                     for i in range(points.count())]
                ys = [float(points.nth(i).get_attribute("data-y") or "nan")
                     for i in range(points.count())]
                assert xs == _PAYOFF_CURVE["x"]
                assert ys == _PAYOFF_CURVE["y"]

                page.get_by_test_id("back-to-event").click()
                expect(page.get_by_test_id("event-detail")).to_be_visible()
                page.get_by_test_id("back-to-board").click()
                expect(page.get_by_test_id("event-table")).to_be_visible()

                # -- publish generation 2 (release B) mid-session: the
                #    pinned release stays release A, and the notice appears.
                binding_b = projections.projection_binding(serving_conn, release_b.release_id)
                files_b = _stage_files(ops_store, "release-b", binding_b)
                claim2 = enqueue_claim(ops_conn, clock, supervisor, key="pub-gen2")
                _stage_and_publish(ops_conn, ops_store, claim2, "rel-gen2", "2026-09-14", files_b,
                                   expected_current="rel-gen1", target=target, scope=scope,
                                   clock=clock, generation="gen2")

                expect(page.get_by_test_id("release-changed-notice")).to_be_visible(timeout=5000)
                expect(page.get_by_test_id("release-changed-notice")).to_contain_text(
                    release_b.release_id)
                expect(page.get_by_test_id("release-id")).to_contain_text(release_a.release_id)
            finally:
                context.close()

            # -- a deep link to generation 1 (release A), now NOT current,
            #    shows the non-current banner via a direct
            #    GET /api/v1/releases/{id} fetch (not inherited from
            #    `current`) -- its own as-of, and a link to what IS current.
            context = _authed_context(browser, base)
            page = context.new_page()
            try:
                page.goto(f"{base}/#/release/{release_a.release_id}")
                expect(page.get_by_test_id("release-id")).to_contain_text(release_a.release_id)
                expect(page.get_by_test_id("release-as-of")).to_contain_text(
                    release_a.resolved_as_of)
                expect(page.get_by_test_id("release-not-current-notice")).to_be_visible()
                link = page.get_by_test_id("current-release-link")
                expect(link).to_be_visible()
                expect(link).to_contain_text(release_b.release_id)
                assert TOKEN not in page.url

                link.click()
                expect(page.get_by_test_id("release-id")).to_contain_text(
                    release_b.release_id, timeout=10000)
                expect(page.get_by_test_id("release-not-current-notice")).to_have_count(0)
            finally:
                context.close()

            # -- 404 UNKNOWN_RELEASE: a deep link naming a release the
            #    server has never heard of.
            context = _authed_context(browser, base)
            page = context.new_page()
            try:
                page.goto(f"{base}/#/release/does-not-exist-at-all")
                expect(page.get_by_test_id("unknown-release")).to_be_visible()
                expect(page.get_by_test_id("current-release-link")).to_be_visible()
            finally:
                context.close()
        finally:
            _stop(server, thread)
    finally:
        conn.close()
        serving_conn.close()
        ops_conn.close()
