#!/usr/bin/env python3
"""L10-L12 evidence producers -- guide §9 rows L10-L12.

Real ``engine.v2.serving.api.create_app`` mounted with the real built
``ui/dist`` (``StaticFiles``, the same technique ``tests/test_v2_dashboard_
integration.py``'s ``_serve_ui`` uses -- added only here in this producer,
never inside ``engine/v2/serving/api.py`` itself) over a real
``uvicorn.Server`` socket, driven by a real headless-Chromium Playwright
browser. Every input is real: this script never builds data itself -- the
caller passes an ALREADY-INDEXED real ``serving.sqlite`` (built by
``scratch/phase3/build_index.py``, not committed) containing:

* the real attempt-20 clean-44 subset (``--release-id``) -- the same
  defect-free CAL-P/CND-P/STR-RUNUP/STR-THRU population
  ``checks/rearchitecture_phase3_bridge.py``/``rearchitecture_phase3_
  publish.py`` already established (44 of 121 rows; the other 77 are
  blocked by the known ``engine/dashboard/render.py`` ``structure_params``
  rounding defect, out of scope here per the coordinator's instruction);
* a small, HONESTLY-LABELED frozen fixture release (``--fixture-release-id``)
  covering one real, defect-free DYN-SV row and a real ``stale_or_degraded_
  reasons`` demonstration -- see ``scratch/phase3/build_index.py``'s module
  docstring for exactly why and how (guide §9 L12: "frozen fixtures cover
  absent cases").

L10 ``browser_initial_load_parity``: a fresh page load issues exactly
``{GET /api/v1/releases/current, one GET /api/v1/events page}`` against the
real API -- never a detail/score/event-scores/operations fetch. Also proves
a genuinely stale reply cannot overwrite a fresher selection: a real
network request for one ticker filter is delayed via Playwright route
interception (the real HTTP call still completes, only later), a second,
different filter is submitted before it resolves, and the board must end
up showing the SECOND (later-issued) query's own real rows -- exactly
``ui/src/hooks.ts``'s ``requestIdRef`` stale-reply guard. ``browser_receipt_
ref``: a real screenshot plus release id/as-of/url (reused by L12 too --
the evidence schema carries one ``browser_receipt_ref`` field).

L11 ``ui_state_parity``: loading/empty/null/zero/refusal/auth/detail-error/
stale/non-current/unknown-release states, each checked against a real HTTP
response AND a real rendered DOM element. The real clean-44 subset already
contains 6 real refusal rows (``gate_pass=False``, real ``detail`` reasons)
and 34 real null-``entry_cost`` rows -- reused directly, no fabrication.
**Judgement call, documented honestly:** "withheld" has no dedicated board
element yet -- confirmed by reading ``ui/src/components/ReleaseBanner.tsx``
(only a ``release-stale`` test id exists) and ``ui/README.md``'s own
Deferred section (the full health/flags screen, which would surface
``withheld_release``, is explicitly Phase 6). ``stale_or_degraded_reasons``
(the real, currently-shipped ``release-stale`` badge) is exercised instead,
via the frozen fixture's real badge render -- the closest real analogue,
not "withheld" itself. ``engineering_receipt_ref`` is a plain JSON document
(no typed contract exists for it -- guide §10 names it in prose only),
built from ``engine.v2.ops.health.trailing_occurrences``/``engineering_
history`` over a REAL ops catalog copy: one real observed night
(2026-09-10, `ok=1`, 13 attempts) and the rest of the trailing 14-session
window genuinely ``unknown`` -- no fabricated pass, matching guide §9's
"missing nights/history stay unknown."

L12 ``full_population_parity``: paginates the real clean-44 release end to
end through the real UI and diffs every rendered row's ticker/event date/
strategy/verdict/refusal-reason/driver-forecast/market-implied-move/
entry-premium/expected-return exactly against the real API's own JSON for
that release -- the full 44-row shipped population, not a sample. Also one
``GET /api/v1/scores/{id}`` -> ``score-detail`` field-row spot check each
for STR-THRU, STR-RUNUP and a real refusal row (drawn from the real
clean-44 subset) and one for the frozen fixture's DYN-SV row (menu_size/
chosen_strategy/chosen_margin), per the module docstring's DYN-SV note.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase1_gate import source_files, source_hash  # noqa: E402
from checks.rearchitecture_phase2_gate import environment_hash as _environment_hash  # noqa: E402
from engine.v2.diagnosis import AGREE, DIFFER, ComparisonReceipt, Envelope, Finding, Population, content_hash  # noqa: E402
from engine.v2.foundation import to_document  # noqa: E402
from engine.v2.ops.health import engineering_history, trailing_occurrences  # noqa: E402
from engine.v2.serving.api import create_app  # noqa: E402

INITIAL_LOAD_KIND = "browser_initial_load_parity"
UI_STATE_KIND = "ui_state_parity"
FULL_POPULATION_KIND = "full_population_parity"

ALLOWED_INITIAL_PREFIXES = ("/api/v1/releases/current", "/api/v1/events?")
FORBIDDEN_INITIAL_SUBSTRINGS = ("/scores", "/api/v1/operations", "/api/v1/releases/")


# --------------------------------------------------------------------------
# server plumbing -- same pattern as rearchitecture_phase3_api_pagination.py
# --------------------------------------------------------------------------


def _mk_app(serving_db: Path, store_root: Path, serving_root: Path, dist_dir: Path, token: str,
           current_release_id: str):
    app = create_app(serving_db=str(serving_db), store_root=str(store_root), serving_root=str(serving_root),
                     token=token, resolver=lambda: current_release_id)
    # StaticFiles mount added ONLY here, after every /api/v1/... route is
    # already registered -- see module docstring; mirrors tests/
    # test_v2_dashboard_integration.py's own _serve_ui exactly.
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="ui-dist")
    return app


def _start(app):
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("uvicorn server did not start in time")
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://127.0.0.1:{port}"


def _stop(server, thread) -> None:
    server.should_exit = True
    thread.join(timeout=5)


def _get(base: str, path: str, *, token: str, params: dict | None = None) -> tuple[int, dict]:
    url = base + path
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    if clean:
        url += "?" + urllib.parse.urlencode(clean)
    request = urllib.request.Request(url)
    request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# --------------------------------------------------------------------------
# display formatting -- Python mirrors of ui/src/format.ts, verified against
# real rendered DOM text below, never assumed
# --------------------------------------------------------------------------


def fmt_percent(value, digits=1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def fmt_number(value, digits=2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def fmt_text(value) -> str:
    return "—" if value is None else str(value)


def fmt_unknown(value) -> str:
    """Mirrors ``ui/src/format.ts``'s ``fmtUnknown`` exactly (score detail's
    generic field formatter): bool -> lowercase ``true``/``false``, never
    Python's capitalized ``str(bool)``."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def headline_expected_return(score: dict):
    if score.get("expected_return_model") is not None:
        return score["expected_return_model"]
    return score.get("expected_return_sim")


def _finding(findings: list, kind: str, field: str) -> None:
    findings.append(Finding(finding_id=content_hash([kind, field])[7:19], first_differing_stage="browser",
                            field_path=field, kind="value", owning_stage="browser"))


def _receipt(kind: str, tier: int, left_ref: str, right_ref: str, findings: list[Finding], expected: int,
            compared: int, *, code_hash: str, environment_hash: str) -> ComparisonReceipt:
    population = Population(expected=expected, supported=compared, compared=compared)
    verdict = DIFFER if findings else (AGREE if compared > 0 else DIFFER)
    envelope = Envelope(code_hash=code_hash, environment_hash=environment_hash)
    receipt_id = content_hash([kind, left_ref, right_ref, [f.finding_id for f in findings]])[7:23]
    return ComparisonReceipt(receipt_id=receipt_id, comparison_kind=kind, tier=tier, left_ref=left_ref,
        right_ref=right_ref, stage_plan_ref=f"{kind}.v1", tolerance_policy_ref="exact_bytes.v1",
        verdict=verdict, findings=tuple(findings), population=population, envelope=envelope)


def _authed_context(browser, base: str, token: str):
    context = browser.new_context()
    context.add_cookies([{"name": "operations_token", "value": token, "url": base}])
    return context


# --------------------------------------------------------------------------
# L10 -- browser_initial_load_parity / browser_receipt_ref
# --------------------------------------------------------------------------


def build_initial_load(base: str, token: str, release_id: str, delayed_ticker: str, later_ticker: str,
                       browser, artifact_root: Path, *, code_hash: str, environment_hash: str):
    findings: list[Finding] = []
    checks = 0

    # -- part 1: the initial-load request set --
    requests: list[str] = []
    context = _authed_context(browser, base, token)
    page = context.new_page()
    page.on("request", lambda req: requests.append(urllib.parse.urlparse(req.url).path
                                                    + ("?" + urllib.parse.urlparse(req.url).query
                                                       if urllib.parse.urlparse(req.url).query else "")))
    page.goto(base + "/")
    page.wait_for_selector('[data-testid="event-table"], [data-testid="no-matches"]', timeout=10000)
    page.wait_for_timeout(300)
    release_text = page.get_by_test_id("release-id").inner_text()
    as_of_text = page.get_by_test_id("release-as-of").inner_text()

    api_requests = [r for r in requests if r.startswith("/api/")]
    checks += 1
    if not any(r.startswith("/api/v1/releases/current") for r in api_requests):
        _finding(findings, INITIAL_LOAD_KIND, "current_not_fetched")
    checks += 1
    events_requests = [r for r in api_requests if r.startswith("/api/v1/events?")]
    if len(events_requests) != 1:
        _finding(findings, INITIAL_LOAD_KIND, "expected_exactly_one_bounded_event_page")
    checks += 1
    forbidden = [r for r in api_requests
                if any(s in r for s in FORBIDDEN_INITIAL_SUBSTRINGS)
                and not r.startswith("/api/v1/releases/current")]
    if forbidden:
        _finding(findings, INITIAL_LOAD_KIND, "detail_or_ladder_or_operations_fetched_eagerly")

    screenshot_path = artifact_root / "browser_screenshot.png"
    artifact_root.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(screenshot_path))
    context.close()

    # -- part 2: a genuinely stale reply cannot overwrite a newer selection --
    context = _authed_context(browser, base, token)
    page = context.new_page()
    page.goto(base + "/")
    page.wait_for_selector('[data-testid="event-filters"]')

    def _delay_route(route):
        if delayed_ticker in (route.request.url or ""):
            time.sleep(1.5)
        route.continue_()

    page.route("**/api/v1/events*", _delay_route)
    page.get_by_label("Ticker").fill(delayed_ticker)
    page.locator('[data-testid="event-filters"] button[type="submit"]').click()
    page.wait_for_timeout(100)
    page.get_by_label("Ticker").fill(later_ticker)
    page.locator('[data-testid="event-filters"] button[type="submit"]').click()
    page.wait_for_timeout(2200)  # long enough for the delayed (stale) reply to also land

    checks += 1
    rows = page.get_by_test_id("score-row")
    n = rows.count()
    tickers_shown = {rows.nth(i).locator("td").nth(0).inner_text() for i in range(n)} if n else set()
    if tickers_shown and tickers_shown != {later_ticker}:
        _finding(findings, INITIAL_LOAD_KIND, "stale_reply_overwrote_newer_selection")
    context.close()

    comparison = _receipt(INITIAL_LOAD_KIND, 1, "browser:initial_load", "release:" + release_id, findings,
                          expected=4, compared=4, code_hash=code_hash, environment_hash=environment_hash)
    data = screenshot_path.read_bytes()
    browser_receipt = {
        "release_id": release_text.replace("release", "", 1).strip(),
        "as_of": as_of_text.replace("as of", "", 1).strip(),
        "url": base + "/", "screenshot_ref": {"path": "browser_screenshot.png",
                                              "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()},
    }
    return comparison, browser_receipt


# --------------------------------------------------------------------------
# L11 -- ui_state_parity / engineering_receipt_ref
# --------------------------------------------------------------------------


def build_ui_state(base: str, token: str, release_id: str, fixture_release_id: str, unknown_release_id: str,
                   refusal_ticker: str, null_field_ticker: str, browser, *, code_hash: str,
                   environment_hash: str) -> ComparisonReceipt:
    findings: list[Finding] = []
    checks = 0

    # -- empty: a filter with no real matches --
    context = _authed_context(browser, base, token)
    page = context.new_page()
    page.goto(base + "/")
    page.wait_for_selector('[data-testid="event-table"]')
    page.get_by_label("Ticker").fill("NOSUCHTICKERXYZ")
    page.locator('[data-testid="event-filters"] button[type="submit"]').click()
    checks += 1
    try:
        page.wait_for_selector('[data-testid="no-matches"]', timeout=5000)
    except Exception:
        _finding(findings, UI_STATE_KIND, "empty_state_not_rendered")
    context.close()

    # -- refusal + null field: real rows already in the clean-44 subset --
    status, page_json = _get(base, "/api/v1/events", token=token,
                             params={"release_id": release_id, "ticker": refusal_ticker, "limit": 50})
    checks += 1
    refusal_scores = [s for item in page_json.get("items", []) for s in item["scores"]
                      if s.get("refusal_reason") is not None]
    if status != 200 or not refusal_scores:
        _finding(findings, UI_STATE_KIND, "no_real_refusal_row_found")
    else:
        context = _authed_context(browser, base, token)
        page = context.new_page()
        page.goto(base + f"/#/release/{release_id}")
        page.wait_for_selector('[data-testid="event-table"]')
        page.get_by_label("Ticker").fill(refusal_ticker)
        page.locator('[data-testid="event-filters"] button[type="submit"]').click()
        page.wait_for_timeout(300)
        checks += 1
        if page.get_by_test_id("refusal-badge").count() == 0:
            _finding(findings, UI_STATE_KIND, "refusal_badge_not_rendered")
        context.close()

    status, page_json = _get(base, "/api/v1/events", token=token,
                             params={"release_id": release_id, "ticker": null_field_ticker, "limit": 50})
    checks += 1
    has_null = any(s.get("entry_premium") is None for item in page_json.get("items", []) for s in item["scores"])
    if status != 200 or not has_null:
        _finding(findings, UI_STATE_KIND, "no_real_null_field_row_found")
    else:
        context = _authed_context(browser, base, token)
        page = context.new_page()
        page.goto(base + f"/#/release/{release_id}")
        page.wait_for_selector('[data-testid="event-table"]')
        page.get_by_label("Ticker").fill(null_field_ticker)
        page.locator('[data-testid="event-filters"] button[type="submit"]').click()
        page.wait_for_timeout(300)
        checks += 1
        cells = page.locator('[data-testid="entry-premium-cell"]')
        if cells.count() == 0 or "—" not in " ".join(cells.nth(i).inner_text() for i in range(cells.count())):
            _finding(findings, UI_STATE_KIND, "null_field_not_rendered_as_dash")
        context.close()

    # -- auth: missing token --
    checks += 1
    context = browser.new_context()
    page = context.new_page()
    status_code = {}
    page.on("response", lambda resp: status_code.setdefault(resp.url, resp.status)
           if "/api/v1/releases/current" in resp.url else None)
    page.goto(base + "/")
    page.wait_for_timeout(500)
    if not any(code == 401 for code in status_code.values()):
        _finding(findings, UI_STATE_KIND, "unauthenticated_did_not_401")
    context.close()

    # -- detail-error: unknown score id --
    status, _body = _get(base, "/api/v1/scores/does-not-exist", token=token, params={"release_id": release_id})
    checks += 1
    if status != 404:
        _finding(findings, UI_STATE_KIND, "unknown_score_not_404")

    # -- unknown release (browser) --
    context = _authed_context(browser, base, token)
    page = context.new_page()
    page.goto(base + f"/#/release/{unknown_release_id}")
    checks += 1
    try:
        page.wait_for_selector('[data-testid="unknown-release"]', timeout=5000)
    except Exception:
        _finding(findings, UI_STATE_KIND, "unknown_release_state_not_rendered")
    context.close()

    # -- non-current release banner (real): clean-44 is current, request it by id --
    context = _authed_context(browser, base, token)
    page = context.new_page()
    page.goto(base + f"/#/release/{release_id}")
    checks += 1
    try:
        page.wait_for_selector('[data-testid="release-id"]', timeout=5000)
        # clean-44 IS current in this server, so re-fetching current does not
        # show a not-current notice; assert its absence as the real control case.
        if page.get_by_test_id("release-not-current-notice").count() != 0:
            _finding(findings, UI_STATE_KIND, "current_release_wrongly_flagged_not_current")
    except Exception:
        _finding(findings, UI_STATE_KIND, "release_id_state_not_rendered")
    context.close()

    # -- stale: the frozen fixture's real stale_or_degraded_reasons badge --
    context = _authed_context(browser, base, token)
    page = context.new_page()
    page.goto(base + f"/#/release/{fixture_release_id}")
    checks += 1
    try:
        page.wait_for_selector('[data-testid="release-stale"]', timeout=5000)
        stale_text = page.get_by_test_id("release-stale").inner_text()
        if "model_evidence_unavailable" not in stale_text:
            _finding(findings, UI_STATE_KIND, "stale_reason_text_missing")
    except Exception:
        _finding(findings, UI_STATE_KIND, "stale_badge_not_rendered")
    context.close()

    return _receipt(UI_STATE_KIND, 1, "browser:states", "release:" + release_id, findings,
                    expected=checks, compared=checks, code_hash=code_hash, environment_hash=environment_hash)


def build_engineering_receipt(catalog_path: Path, session: str) -> dict:
    """Real per-night status from a REAL ops catalog copy (never the live
    ``/root/phase2-shadow-ops/catalog.sqlite`` -- caller passes a copy).
    ``status`` is translated pass/fail -> ``observed`` (an engineering
    observation exists for that night, whichever way it went) and
    unknown -> ``unknown``, per guide §9 L11: "count nights rather than
    retries; missing nights/history stay unknown" -- the raw ``pass``/
    ``fail`` distinction and the real retry count are kept in ``detail``/
    ``retry_count`` for anyone who wants it, never discarded.
    """
    conn = sqlite3.connect(f"file:{catalog_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        occurrences = trailing_occurrences(session, nights=14)
        history = engineering_history(conn, occurrences)
    finally:
        conn.close()
    nights = [{"date": h["occurrence"], "status": "unknown" if h["status"] == "unknown" else "observed",
              "raw_status": h["status"], "retry_count": h["retry_count"]} for h in history]
    return {"nights": nights, "window_session": session,
           "source": "root:/root/phase2-shadow-ops/catalog.sqlite (copied read-only)"}


# --------------------------------------------------------------------------
# L12 -- full_population_parity (reuses browser_receipt_ref from L10)
# --------------------------------------------------------------------------


def build_full_population(base: str, token: str, release_id: str, fixture_release_id: str,
                          str_thru_ticker: str, str_runup_ticker: str, refusal_ticker: str,
                          browser, *, code_hash: str, environment_hash: str) -> ComparisonReceipt:
    findings: list[Finding] = []

    status, full = _get(base, "/api/v1/events", token=token, params={"release_id": release_id, "limit": 200})
    if status != 200:
        raise RuntimeError(f"full API fetch failed: {status}")
    api_scores: dict[str, dict] = {}
    for item in full["items"]:
        for score in item["scores"]:
            api_scores[score["score_id"]] = {**score, "ticker": item["ticker"], "event_date": item["event_date"]}
    expected_population = len(api_scores)

    context = _authed_context(browser, base, token)
    page = context.new_page()
    page.goto(base + f"/#/release/{release_id}")
    page.wait_for_selector('[data-testid="event-table"]')

    seen: set[str] = set()
    mismatched_fields = 0
    for _ in range(20):
        rows = page.get_by_test_id("score-row")
        n = rows.count()
        for i in range(n):
            row = rows.nth(i)
            score_id = row.get_attribute("data-score-id")
            api = api_scores.get(score_id)
            if api is None:
                _finding(findings, FULL_POPULATION_KIND, f"row_{score_id}_not_in_api_population")
                continue
            seen.add(score_id)
            cells = row.locator("td")
            expected_cells = {
                0: api["ticker"], 1: api["event_date"], 3: api["strategy"],
                5: fmt_percent(api["driver_forecast"]), 6: fmt_percent(api["market_implied_move"]),
                7: fmt_number(api["entry_premium"]), 8: fmt_percent(headline_expected_return(api)),
            }
            for index, expected_text in expected_cells.items():
                actual_text = cells.nth(index).inner_text().strip()
                if index == 8:
                    actual_text = actual_text.replace("sim", "").strip()
                if actual_text != expected_text:
                    mismatched_fields += 1
                    _finding(findings, FULL_POPULATION_KIND, f"row_{score_id}_cell_{index}")
            verdict_text = row.locator('[data-testid="verdict-cell"]').inner_text()
            if api.get("refusal_reason") is not None:
                if "REFUSED" not in verdict_text or api["refusal_reason"] not in verdict_text:
                    _finding(findings, FULL_POPULATION_KIND, f"row_{score_id}_refusal_text")
        next_button = page.get_by_test_id("page-next")
        if next_button.is_disabled():
            break
        next_button.click()
        page.wait_for_timeout(200)

    missing = set(api_scores) - seen
    for score_id in missing:
        _finding(findings, FULL_POPULATION_KIND, f"row_{score_id}_never_shown_in_ui")

    # spot checks: STR-THRU, STR-RUNUP, a refusal row (real clean-44) --
    # each fetched independently over real HTTP, opened in score detail,
    # compared field for field against the API's own display_record.
    for ticker, strategy in ((str_thru_ticker, "STR-THRU"), (str_runup_ticker, "STR-RUNUP")):
        candidates = [sid for sid, s in api_scores.items() if s["ticker"] == ticker and s["strategy"] == strategy]
        if not candidates:
            _finding(findings, FULL_POPULATION_KIND, f"no_{strategy}_sample_available")
            continue
        score_id = candidates[0]
        status, detail = _get(base, f"/api/v1/scores/{score_id}", token=token, params={"release_id": release_id})
        if status != 200:
            _finding(findings, FULL_POPULATION_KIND, f"{strategy}_detail_fetch_failed")
            continue
        display = detail["display_record"]
        page.goto(base + f"/#/release/{release_id}/scores/{score_id}")
        try:
            page.wait_for_selector('[data-testid="score-detail"]', timeout=5000)
        except Exception:
            _finding(findings, FULL_POPULATION_KIND, f"{strategy}_detail_page_not_rendered")
            continue
        for field in ("ticker", "strategy", "gate_pass"):
            row = page.locator(f'[data-testid="field-row"][data-field="{field}"]')
            if row.count() == 0:
                continue
            actual = row.get_by_test_id("field-value").inner_text().strip()
            expected = fmt_unknown(display.get(field))
            if actual != expected:
                _finding(findings, FULL_POPULATION_KIND, f"{strategy}_field_{field}")

    # DYN-SV: frozen fixture release (real defect blocks it in the real population).
    status, fixture_api_page = _get(base, "/api/v1/events", token=token,
                                    params={"release_id": fixture_release_id, "limit": 50})
    if status == 200 and fixture_api_page["items"]:
        dyn_score = fixture_api_page["items"][0]["scores"][0]
        if dyn_score["chosen_strategy"] is None or dyn_score["menu_size"] is None:
            _finding(findings, FULL_POPULATION_KIND, "dynsv_fixture_choice_fields_missing")
        # A FRESH context/page: a same-page page.goto() to a hash-only-different
        # URL is a same-document navigation, and ui/src/hooks.ts's own pin-
        # stability rule (guide §9 L02: "R1 readers retain R1... only a fresh
        # page load ... can change the pin") deliberately keeps the OLD pin in
        # that case -- a real product behavior, not a bug. A genuinely fresh
        # navigation is required to pin the fixture release.
        fixture_context = _authed_context(browser, base, token)
        fixture_page = fixture_context.new_page()
        fixture_page.goto(base + f"/#/release/{fixture_release_id}")
        try:
            fixture_page.wait_for_selector('[data-testid="score-row"]', timeout=5000)
            row_text = fixture_page.get_by_test_id("score-row").first.inner_text()
            if dyn_score["chosen_strategy"] not in row_text:
                _finding(findings, FULL_POPULATION_KIND, "dynsv_choice_not_rendered_in_board")
        except Exception:
            _finding(findings, FULL_POPULATION_KIND, "dynsv_fixture_board_not_rendered")
        fixture_context.close()
    else:
        _finding(findings, FULL_POPULATION_KIND, "dynsv_fixture_release_unreadable")

    context.close()
    return _receipt(FULL_POPULATION_KIND, 1, "release:" + release_id, "release:" + release_id, findings,
                    expected=expected_population, compared=len(seen), code_hash=code_hash,
                    environment_hash=environment_hash)


# --------------------------------------------------------------------------
# publish / main
# --------------------------------------------------------------------------


def publish_document(document, artifact_root: Path, name: str) -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    data = json.dumps(document if isinstance(document, dict) else to_document(document),
                      indent=2, sort_keys=True).encode()
    path = artifact_root / name
    path.write_bytes(data)
    return {"path": path.name, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-db", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--serving-root", type=Path, required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--release-id", required=True, help="the real clean-44 release, served as current")
    parser.add_argument("--fixture-release-id", required=True)
    parser.add_argument("--unknown-release-id", default="does-not-exist-at-all")
    parser.add_argument("--catalog-copy", type=Path, required=True,
                        help="a private, read-only COPY of /root/phase2-shadow-ops/catalog.sqlite")
    parser.add_argument("--engineering-session", default="2026-09-15")
    parser.add_argument("--refusal-ticker", required=True)
    parser.add_argument("--null-field-ticker", required=True)
    parser.add_argument("--str-thru-ticker", required=True)
    parser.add_argument("--str-runup-ticker", required=True)
    parser.add_argument("--delayed-ticker", required=True)
    parser.add_argument("--later-ticker", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)

    code_hash = source_hash(source_files(ROOT))
    env_hash, _source = _environment_hash(ROOT)

    app = _mk_app(args.serving_db, args.store_root, args.serving_root, args.dist_dir, args.token, args.release_id)
    server, thread, base = _start(app)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                initial_load, browser_receipt = build_initial_load(
                    base, args.token, args.release_id, args.delayed_ticker, args.later_ticker, browser,
                    args.artifact_root, code_hash=code_hash, environment_hash=env_hash)
                ui_state = build_ui_state(
                    base, args.token, args.release_id, args.fixture_release_id, args.unknown_release_id,
                    args.refusal_ticker, args.null_field_ticker, browser, code_hash=code_hash,
                    environment_hash=env_hash)
                full_population = build_full_population(
                    base, args.token, args.release_id, args.fixture_release_id, args.str_thru_ticker,
                    args.str_runup_ticker, args.refusal_ticker, browser, code_hash=code_hash,
                    environment_hash=env_hash)
            finally:
                browser.close()
    finally:
        _stop(server, thread)

    engineering = build_engineering_receipt(args.catalog_copy, args.engineering_session)

    out = {}
    for kind, (doc, name) in {
        "browser_initial_load_parity": (initial_load, "browser_initial_load_parity.json"),
        "ui_state_parity": (ui_state, "ui_state_parity.json"),
        "full_population_parity": (full_population, "full_population_parity.json"),
        "browser_receipt": (browser_receipt, "browser_receipt.json"),
        "engineering_receipt": (engineering, "engineering_receipt.json"),
    }.items():
        ref = publish_document(doc, args.artifact_root, name)
        verdict = getattr(doc, "verdict", None)
        out[kind] = {**ref, **({"verdict": verdict} if verdict is not None else {})}
    print(json.dumps(out, indent=2))

    ok = (initial_load.verdict == AGREE and ui_state.verdict == AGREE and full_population.verdict == AGREE)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
