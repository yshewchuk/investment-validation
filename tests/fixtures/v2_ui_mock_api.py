"""Mock §6 read API plus static ``ui/dist`` serving, for the browser tests in
``tests/test_v2_dashboard_browser.py``.

The real FastAPI read API is P3-2 (guide §8), still in progress alongside
this file (see ``engine/v2/serving/projections.py``'s ``list_events``/
``get_release``/``event_scores``/``get_score_detail`` — the bounded queries a
future route wraps). This module is a **stand-in**, stdlib-only, so the React
board can be built and tested against a stable §6 shape before P3-2 exists.
It is never imported by production code and never touches the real serving
index.

JSON field names are taken verbatim from ``engine/v2/contracts/serving.py``
(``PreviewRelease``, ``EventPage``, ``EventPageItem``, ``EventScoreSummary``
v1.1, ``LegacyScoreBridge``) so the UI's TS types (``ui/src/api/types.ts``)
need no translation layer when P3-2 lands. Auth mirrors
``engine/v2/serving/operations.py::_authorized``: a bearer token OR a
same-origin ``operations_token`` cookie, never a URL parameter.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import http.cookies
import http.server
import json
import mimetypes
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

HTTPStatus = http.HTTPStatus
TOKEN_COOKIE = "operations_token"
DEFAULT_LIMIT = 50
MAX_LIMIT = 200

__all__ = [
    "MockState",
    "ReleaseFixture",
    "build_default_state",
    "create_mock_server",
    "make_event_item",
    "make_score",
]


# --------------------------------------------------------------------------
# fixture shapes (exactly the contracts' field names)
# --------------------------------------------------------------------------


def make_score(
    score_id: str,
    strategy: str,
    *,
    verdict: str | None = None,
    refusal_reason: str | None = None,
    driver_forecast: float | None = None,
    market_implied_move: float | None = None,
    entry_premium: float | None = None,
    expected_return_model: float | None = None,
    expected_return_analog: float | None = None,
    expected_return_sim: float | None = None,
    chosen_strategy: str | None = None,
    chosen_margin: float | None = None,
    menu_size: int | None = None,
) -> dict:
    """One ``EventScoreSummary`` (v1.1) — ``expected_return`` is always null,
    per the README gap this contract records: the rendered row carries no
    single merged headline, only the three per-producer reads."""
    return {
        "score_id": score_id,
        "strategy": strategy,
        "verdict": verdict,
        "refusal_reason": refusal_reason,
        "driver_forecast": driver_forecast,
        "market_implied_move": market_implied_move,
        "entry_premium": entry_premium,
        "expected_return": None,
        "expected_return_model": expected_return_model,
        "expected_return_analog": expected_return_analog,
        "expected_return_sim": expected_return_sim,
        "chosen_strategy": chosen_strategy,
        "chosen_margin": chosen_margin,
        "menu_size": menu_size,
        "schema_version": "event_score_summary.v1.1",
    }


def make_event_item(
    event_id: str,
    ticker: str,
    event_date: str,
    *,
    calendar_revision: str = "rev1",
    session: str | None = "AMC",
    clock_id: str = "legacy.entry_close.v1",
    readiness: str = "ready",
    scores: tuple[dict, ...] = (),
) -> dict:
    """One ``EventPageItem``."""
    return {
        "event_ref": {
            "event_id": event_id,
            "calendar_revision": calendar_revision,
            "schema_version": "event_ref.v1.0",
        },
        "ticker": ticker,
        "event_date": event_date,
        "session": session,
        "clock_id": clock_id,
        "readiness": readiness,
        "scores": list(scores),
        "schema_version": "event_page_item.v1.0",
    }


def _preview_release(
    release_id: str,
    *,
    resolved_as_of: str,
    stale_or_degraded_reasons: tuple[str, ...] = (),
    coverage_summary: dict[str, float] | None = None,
) -> dict:
    return {
        "release_id": release_id,
        "source_release_id": "source-" + release_id,
        "projection_manifest_ref": "manifest-" + release_id,
        "snapshot_ref": "snapshot-" + release_id,
        "score_batch_ref": "batch-" + release_id,
        "bundle_manifest_ref": "bundle-" + release_id,
        "model_registry_artifact_refs": ["model-1"],
        "model_evidence_ref": None,
        "comparison_receipt_refs": ["receipt-" + release_id],
        "source_code_hash": "sha256:" + hashlib.sha256(release_id.encode()).hexdigest(),
        "projection_code_hash": "sha256:mock",
        "requested_as_of": resolved_as_of,
        "resolved_as_of": resolved_as_of,
        "clock_ids": ["legacy.entry_close.v1"],
        "coverage_summary": coverage_summary or {},
        "stale_or_degraded_reasons": list(stale_or_degraded_reasons),
        "score_format": "legacy_score_bridge.v1.0",
        "producer": "legacy_via_v2",
        "capabilities": {"read": True, "submit_jobs": False, "collect_live": False},
        "schema_version": "preview_release.v1.0",
    }


@dataclass
class ReleaseFixture:
    release: dict
    items: list[dict]  # EventPageItem dicts, pre-sorted (event_date, ticker, event_id)
    scores_by_id: dict[str, dict] = field(default_factory=dict)
    event_ref_by_score_id: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.items = sorted(self.items, key=lambda it: (it["event_date"], it["ticker"], it["event_ref"]["event_id"]))
        for item in self.items:
            for score in item["scores"]:
                self.scores_by_id[score["score_id"]] = score
                self.event_ref_by_score_id[score["score_id"]] = item["event_ref"]


@dataclass
class MockState:
    """Mutable in-process fixture set. Tests hold a reference to this (the
    server runs on a daemon thread in the SAME process, exactly like
    ``tests/test_v2_dashboard_preview.py``'s ``_serve``/``_write_current``
    pattern) and call ``set_current`` directly — no HTTP control channel."""

    releases: dict[str, ReleaseFixture]
    current_release_id: str | None
    token: str = "mock-secret"
    dist_root: Path | None = None
    force_events_error: bool = False
    operations: dict = field(default_factory=lambda: {
        "schema_version": "operations_health.v1.0",
        "generated_at": "2026-09-14T00:00:00Z",
        "withheld_release": None,
        "code_budgets": {"consecutive_nights": 0},
    })

    def set_current(self, release_id: str | None) -> None:
        if release_id is not None and release_id not in self.releases:
            raise KeyError(release_id)
        self.current_release_id = release_id


# --------------------------------------------------------------------------
# score-detail fixtures (P3-3b): (engine_record, display_record) pairs for
# a handful of the summary scores above, shaped with the LEGACY board's own
# field names (``engine/v2/serving/bridge.py::_BOARD_FIELD_NAMES``/``_UNITS``,
# transcribed in ``ui/src/displayFieldSpec.ts``) -- not the EventScoreSummary
# names those rows also carry. ``engine_record`` always adds one field
# (``raw_model_state_ref``) that ``display_record`` never gets, so the two
# sections of the score-detail view are provably separate objects, not one
# dict shown twice.
# --------------------------------------------------------------------------


def _legacy_pair(score_id: str, **fields: Any) -> tuple[dict, dict]:
    base = {
        "row_id": score_id, "ticker": "TICK", "strategy": "STR-THRU",
        "as_of": "legacy.entry_close.v1", "event_date": "2026-09-01", "session": "AMC",
        "entry_date": "2026-08-31", "exit_date": None, "strike": 100.0, "expiry": "2026-09-18",
        "quote_date": "2026-08-31", "spot": 101.25, "entry_cost": 0.62,
        "entry_cost_pct": 0.0061, "exp_pnl_model": 0.02, "exp_pnl_analog": 0.018,
        "exp_pnl_sim": 0.019, "win_model": 0.55, "gate_score": 0.71, "gate_threshold": 0.5,
        "gate_pass": True, "forecast_abs_move": 0.05, "forecast_model": "driver-v3",
        "detail": None, "scored": True, "flags": [], "model_versions": {"driver": "v3.2"},
        "chosen_strategy": None, "chosen_margin": None, "menu_size": None,
        "legs": [{"strike": 95.0, "qty": 1, "side": "buy", "right": "P"},
                 {"strike": 100.0, "qty": 1, "side": "sell", "right": "P"}],
        "payoff_curve": {
            "x": [90.0, 95.0, 100.0, 105.0, 110.0], "y": [5.0, 5.0, 0.0, 0.0, 0.0],
            "max": 5.0, "min": 0.0,
            "strikes": [{"strike": 95.0, "qty": 1, "side": "buy"},
                        {"strike": 100.0, "qty": 1, "side": "sell"}],
            "shape": "centre",
        },
        "digest": "sha256:mock-" + score_id,
    }
    base.update(fields)
    engine_record = {**base, "raw_model_state_ref": "state-ref-" + score_id}
    display_record = dict(base)
    return engine_record, display_record


def _build_detail_overrides() -> dict[str, tuple[dict, dict]]:
    overrides: dict[str, tuple[dict, dict]] = {}

    # Null vs. zero (§9 L11) at detail granularity too, plus the one score
    # whose payoff curve has a known, non-trivial point count (5), and one
    # display-only field ("note") absent from the mapping spec entirely --
    # exercises the "otherwise alphabetically" / "other" fallback category.
    overrides["r1-score-0-a"] = _legacy_pair(
        "r1-score-0-a", ticker="TICK0", event_date="2026-09-01",
        entry_cost=None, entry_cost_pct=None,  # null: renders as missing
        exp_pnl_model=0.0,  # a real zero: renders as "0"
        note="mock-only annotation field, not in the mapping spec",
    )

    # Refusal: gate_pass False plus a "detail" reason string, no legs/curve
    # (refused before sizing).
    overrides["r1-score-1-a"] = _legacy_pair(
        "r1-score-1-a", ticker="TICK1", strategy="STR-RUNUP", event_date="2026-09-02",
        strike=None, expiry=None, entry_cost=None, entry_cost_pct=None,
        exp_pnl_model=None, exp_pnl_analog=None, exp_pnl_sim=None, win_model=None,
        gate_score=None, gate_pass=False, detail="entry cost exceeds ceiling",
        scored=False, flags=["ENTRY_COST_CEILING"], legs=[], payoff_curve=None,
    )

    # Empty payoff curve: a real curve object with zero points -- distinct
    # from "no curve at all" (r1-score-1-a, above).
    overrides["r1-score-2-a"] = _legacy_pair(
        "r1-score-2-a", ticker="TICK2", strategy="TWIN-P", event_date="2026-09-03",
        payoff_curve={"x": [], "y": [], "max": 0.0, "min": 0.0, "strikes": [], "shape": "centre"},
    )

    # DYN-SV: chosen_strategy/chosen_margin/menu_size carried through, gate
    # verdict unavailable (null), no payoff curve of its own (it wraps
    # STR-THRU's).
    overrides["r1-score-2-b"] = _legacy_pair(
        "r1-score-2-b", ticker="TICK2", strategy="DYN-SV", event_date="2026-09-03",
        gate_pass=None, gate_score=None, exp_pnl_model=None, exp_pnl_analog=None,
        exp_pnl_sim=None, chosen_strategy="STR-THRU", chosen_margin=0.014, menu_size=4,
        legs=[], payoff_curve=None,
    )

    return overrides


DETAIL_OVERRIDES: dict[str, tuple[dict, dict]] = _build_detail_overrides()


def build_default_state(dist_root: Path | None = None, *, token: str = "mock-secret") -> MockState:
    """Two releases (``r1``: 120 synthetic events across default pages;
    ``r2``: a small second release for the release-switch test) plus an
    empty release used by the empty-release-state test."""
    r1_items = []
    for i in range(120):
        scores: tuple[dict, ...]
        if i == 0:
            # null field vs. zero value, side by side (§9 L11).
            scores = (
                make_score(
                    "r1-score-0-a", "STR-THRU", verdict="true",
                    driver_forecast=0.05, market_implied_move=0.04,
                    entry_premium=None,  # null: renders as missing, never 0
                    expected_return_model=0.0,  # a real zero: renders as "0.0%"
                ),
            )
        elif i == 1:
            # refusal row.
            scores = (
                make_score(
                    "r1-score-1-a", "STR-RUNUP", verdict="false",
                    refusal_reason="entry cost exceeds ceiling",
                    driver_forecast=0.03, market_implied_move=0.06, entry_premium=1.2,
                ),
            )
        elif i == 2:
            # headline expected-return cases: model-null+sim-present, and both-null.
            scores = (
                make_score(
                    "r1-score-2-a", "TWIN-P", verdict="true",
                    driver_forecast=0.02, market_implied_move=0.03, entry_premium=0.8,
                    expected_return_model=None, expected_return_sim=0.11,
                ),
                make_score(
                    "r1-score-2-b", "DYN-SV", verdict=None,
                    driver_forecast=None, market_implied_move=0.03, entry_premium=0.9,
                    expected_return_model=None, expected_return_sim=None,
                    chosen_strategy="STR-THRU", chosen_margin=0.014, menu_size=4,
                ),
            )
        elif i == 3:
            scores = ()  # unavailable/no-scores row
        else:
            scores = (
                make_score(
                    f"r1-score-{i}-a", "STR-THRU", verdict="true" if i % 2 == 0 else "false",
                    driver_forecast=0.01 * (i % 7), market_implied_move=0.02 * (i % 5),
                    entry_premium=0.5 + 0.01 * i, expected_return_model=0.001 * i,
                ),
            )
        r1_items.append(make_event_item(
            f"evt-r1-{i:03d}", f"TICK{i % 9}", f"2026-09-{(i % 27) + 1:02d}",
            readiness="unavailable" if i == 3 else "ready", scores=scores))

    r2_items = [
        make_event_item("evt-r2-000", "TICK0", "2026-09-15",
                        scores=(make_score("r2-score-0-a", "STR-THRU", verdict="true",
                                           driver_forecast=0.02, market_implied_move=0.02,
                                           entry_premium=0.6, expected_return_model=0.01),)),
        make_event_item("evt-r2-001", "TICK1", "2026-09-16",
                        scores=(make_score("r2-score-1-a", "STR-RUNUP", verdict="false",
                                           refusal_reason="spread too wide",
                                           driver_forecast=0.01, market_implied_move=0.05,
                                           entry_premium=1.1),)),
    ]

    releases = {
        "r1": ReleaseFixture(
            release=_preview_release("r1", resolved_as_of="2026-09-14",
                                     coverage_summary={"planned_population": 120.0, "coverage": 1.0}),
            items=r1_items),
        "r2": ReleaseFixture(
            release=_preview_release("r2", resolved_as_of="2026-09-15",
                                     stale_or_degraded_reasons=("model_evidence_unavailable",)),
            items=r2_items),
        "r-empty": ReleaseFixture(
            release=_preview_release("r-empty", resolved_as_of="2026-09-10"), items=[]),
    }
    return MockState(releases=releases, current_release_id="r1", token=token, dist_root=dist_root)


# --------------------------------------------------------------------------
# cursor: opaque, integrity-protected, bound to release + normalized query
# --------------------------------------------------------------------------


def _query_hash(release_id: str, filters: dict[str, Any], limit: int) -> str:
    payload = json.dumps({"release_id": release_id, "filters": filters, "limit": limit},
                         sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _sign(secret: str, payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(secret.encode(), body, "sha256").hexdigest()


def _encode_cursor(secret: str, *, release_id: str, query_hash: str, offset: int) -> str:
    payload = {"release_id": release_id, "query_hash": query_hash, "offset": offset}
    signed = {"p": payload, "s": _sign(secret, payload)}
    return base64.urlsafe_b64encode(json.dumps(signed).encode()).decode()


class _CursorMismatch(Exception):
    pass


def _decode_cursor(secret: str, cursor: str, *, release_id: str, query_hash: str) -> int:
    try:
        signed = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        payload = signed["p"]
        if not hmac.compare_digest(signed["s"], _sign(secret, payload)):
            raise _CursorMismatch()
    except (ValueError, KeyError, TypeError):
        raise _CursorMismatch() from None
    if payload["release_id"] != release_id or payload["query_hash"] != query_hash:
        raise _CursorMismatch()
    return int(payload["offset"])


# --------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------


def _matches(item: dict, filters: dict[str, str]) -> tuple[bool, list[dict]]:
    """Returns (event matches, visible score summaries for this page).

    §6: "Strategy/verdict filters select matching events and their matching
    visible summaries consistently." — an event with at least one matching
    score is included, and only the matching scores are shown on it.
    """
    if filters.get("ticker") and item["ticker"].lower() != filters["ticker"].lower():
        return False, []
    date_from, date_to = filters.get("date_from"), filters.get("date_to")
    if date_from and item["event_date"] < date_from:
        return False, []
    if date_to and item["event_date"] > date_to:
        return False, []
    strategy, verdict = filters.get("strategy"), filters.get("verdict")
    if not strategy and not verdict:
        return True, item["scores"]
    visible = [
        s for s in item["scores"]
        if (not strategy or s["strategy"].lower() == strategy.lower())
        and (not verdict or (s["verdict"] or "").lower() == verdict.lower())
    ]
    return (len(visible) > 0), visible


def _filtered_items(fixture: ReleaseFixture, filters: dict[str, str]) -> list[dict]:
    out = []
    for item in fixture.items:
        matched, visible = _matches(item, filters)
        if matched:
            out.append({**item, "scores": visible})
    return out


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------


class MockApiHandler(http.server.BaseHTTPRequestHandler):
    server_version = "v2-ui-mock/1"

    def do_GET(self):  # noqa: N802
        state: MockState = self.server.state
        split = urlsplit(self.path)
        path = unquote(split.path)
        query = {k: v[0] for k, v in parse_qs(split.query).items()}
        try:
            if path == "/api/v1/releases/current":
                return self._current(state)
            if path.startswith("/api/v1/releases/"):
                release_id = path[len("/api/v1/releases/"):]
                return self._release_by_id(state, release_id)
            if path == "/api/v1/events":
                return self._events(state, query)
            if path.startswith("/api/v1/events/") and path.endswith("/scores"):
                event_id = path[len("/api/v1/events/"):-len("/scores")]
                return self._event_scores(state, event_id, query)
            if path.startswith("/api/v1/scores/"):
                score_id = path[len("/api/v1/scores/"):]
                return self._score(state, score_id, query)
            if path == "/api/v1/operations":
                return self._operations(state)
            return self._static(state, path)
        except BrokenPipeError:  # pragma: no cover - client hung up
            return

    # -- auth ---------------------------------------------------------

    def _authorized(self, state: MockState) -> bool:
        supplied = self.headers.get("Authorization", "")
        cookie = http.cookies.SimpleCookie()
        cookie.load(self.headers.get("Cookie", ""))
        cookie_value = cookie.get(TOKEN_COOKIE)
        return hmac.compare_digest(supplied, "Bearer " + state.token) or (
            cookie_value is not None and hmac.compare_digest(cookie_value.value, state.token))

    def _require_auth(self, state: MockState) -> bool:
        if not self._authorized(state):
            self._problem(HTTPStatus.UNAUTHORIZED, "UNAUTHORIZED", "missing or invalid credential")
            return False
        return True

    # -- routes ---------------------------------------------------------

    def _current(self, state: MockState) -> None:
        if not self._require_auth(state):
            return
        if state.current_release_id is None:
            return self._problem(HTTPStatus.SERVICE_UNAVAILABLE, "NO_CURRENT_RELEASE",
                                 "no current release is published", category="resource", retryable=True)
        fixture = state.releases[state.current_release_id]
        self._json(HTTPStatus.OK, fixture.release, etag=fixture.release["release_id"])

    def _release_by_id(self, state: MockState, release_id: str) -> None:
        """``GET /api/v1/releases/{id}`` -- ``engine/v2/serving/api.py``
        ``releases_by_id``: one specific release's own metadata, current or
        not (404 ``UNKNOWN_RELEASE`` for an id this index never committed).
        Added for P3-3c: the non-current-release banner needs a NAMED
        release's ``resolved_as_of``/coverage/stale reasons directly, not
        only ``current``'s -- this route did not exist on the mock before
        (mock-vs-real shape gap, closed here)."""
        if not self._require_auth(state):
            return
        fixture = state.releases.get(release_id)
        if fixture is None:
            return self._problem(HTTPStatus.NOT_FOUND, "UNKNOWN_RELEASE", "unknown release id")
        self._json(HTTPStatus.OK, fixture.release, etag=fixture.release["release_id"])

    def _events(self, state: MockState, query: dict[str, str]) -> None:
        if not self._require_auth(state):
            return
        if state.force_events_error:
            return self._problem(HTTPStatus.INTERNAL_SERVER_ERROR, "SIMULATED_ERROR",
                                 "simulated server error", category="internal")
        release_id = query.get("release_id")
        if not release_id:
            return self._problem(HTTPStatus.UNPROCESSABLE_ENTITY, "INVALID_REQUEST",
                                 "release_id is required")
        fixture = state.releases.get(release_id)
        if fixture is None:
            return self._problem(HTTPStatus.NOT_FOUND, "UNKNOWN_RELEASE", "unknown release id")
        try:
            limit = max(1, min(int(query.get("limit", DEFAULT_LIMIT)), MAX_LIMIT))
        except ValueError:
            return self._problem(HTTPStatus.UNPROCESSABLE_ENTITY, "INVALID_REQUEST", "invalid limit")
        filters = {k: query[k] for k in ("ticker", "strategy", "verdict", "date_from", "date_to")
                  if query.get(k)}
        matching = _filtered_items(fixture, filters)
        query_hash = _query_hash(release_id, filters, limit)
        offset = 0
        cursor = query.get("cursor")
        if cursor:
            try:
                offset = _decode_cursor(state.token, cursor, release_id=release_id, query_hash=query_hash)
            except _CursorMismatch:
                return self._problem(HTTPStatus.CONFLICT, "CURSOR_MISMATCH",
                                     "cursor does not match this release/query")
        page = matching[offset:offset + limit]
        has_more = offset + limit < len(matching)
        next_cursor = (
            _encode_cursor(state.token, release_id=release_id, query_hash=query_hash, offset=offset + limit)
            if has_more else None
        )
        body = {
            "release_id": release_id, "query_hash": query_hash, "items": page,
            "next_cursor": next_cursor, "total_matching": len(matching),
            "schema_version": "event_page.v1.0",
        }
        self._json(HTTPStatus.OK, body)

    def _event_scores(self, state: MockState, event_id: str, query: dict[str, str]) -> None:
        """§6/P3-2 review: ``release_id`` is required, not merely validated
        when supplied -- ``score``/event identity is only unique WITHIN a
        release (§5.4), so this route never searches across releases.
        Missing is 400 ``RELEASE_ID_REQUIRED`` (a client error), distinct
        from a present-but-unknown id, which is 404 ``UNKNOWN_RELEASE``
        (`engine/v2/serving/api.py::_require_release_id`)."""
        if not self._require_auth(state):
            return
        release_id = query.get("release_id")
        if not release_id:
            return self._problem(HTTPStatus.BAD_REQUEST, "RELEASE_ID_REQUIRED", "release_id is required")
        fixture = state.releases.get(release_id)
        if fixture is None:
            return self._problem(HTTPStatus.NOT_FOUND, "UNKNOWN_RELEASE", "unknown release id")
        for item in fixture.items:
            if item["event_ref"]["event_id"] == event_id:
                return self._json(HTTPStatus.OK, item["scores"])
        self._problem(HTTPStatus.NOT_FOUND, "UNKNOWN_EVENT", "unknown event id")

    def _score(self, state: MockState, score_id: str, query: dict[str, str]) -> None:
        """Same ``release_id``-required rule as ``_event_scores`` (P3-2
        review, `_require_release_id`) -- no more searching every release
        for a bare score id; the caller (always `ui/src/api/client.ts`
        `getScore(scoreId, releaseId)`, non-optional) must name the release
        it saw the score under."""
        if not self._require_auth(state):
            return
        release_id = query.get("release_id")
        if not release_id:
            return self._problem(HTTPStatus.BAD_REQUEST, "RELEASE_ID_REQUIRED", "release_id is required")
        fixture = state.releases.get(release_id)
        if fixture is None:
            return self._problem(HTTPStatus.NOT_FOUND, "UNKNOWN_RELEASE", "unknown release id")
        score = fixture.scores_by_id.get(score_id)
        if score is None:
            return self._problem(HTTPStatus.NOT_FOUND, "UNKNOWN_SCORE", "unknown score id")
        event_ref = fixture.event_ref_by_score_id.get(score_id) or {
            "event_id": "unknown", "calendar_revision": "rev1",
            "schema_version": "event_ref.v1.0",
        }
        engine_record, display_record = DETAIL_OVERRIDES.get(score_id, (score, score))
        bridge = {
            "score_id": score_id,
            "event_ref": event_ref,
            "clock_id": "legacy.entry_close.v1", "legacy_row_id": score_id,
            "score_batch_ref": fixture.release["score_batch_ref"],
            "source_row_key": score_id, "source_record_hash": "sha256:mock",
            "request_provenance_refs": [], "snapshot_ref": fixture.release["snapshot_ref"],
            "model_registry_artifact_refs": fixture.release["model_registry_artifact_refs"],
            "engine_record": engine_record, "display_record": display_record,
            "detail_refs": [], "unavailable_detail_reasons": [],
            "schema_version": "legacy_score_bridge.v1.0",
        }
        self._json(HTTPStatus.OK, bridge)

    def _operations(self, state: MockState) -> None:
        if not self._require_auth(state):
            return
        self._json(HTTPStatus.OK, state.operations)

    def _static(self, state: MockState, path: str) -> None:
        if state.dist_root is None:
            return self._problem(HTTPStatus.NOT_FOUND, "NOT_FOUND", "no static root configured")
        rel = path.lstrip("/") or "index.html"
        parts = [p for p in rel.split("/") if p not in ("", ".")]
        if any(p == ".." for p in parts):
            return self._problem(HTTPStatus.NOT_FOUND, "NOT_FOUND", "missing")
        target = state.dist_root.joinpath(*parts)
        if target.is_dir():
            target = target / "index.html"
        # SPA fallback: any non-file, non-API path serves index.html.
        if not target.is_file() and not path.startswith("/assets/"):
            target = state.dist_root / "index.html"
        if target.is_symlink() or not target.is_file():
            return self._problem(HTTPStatus.NOT_FOUND, "NOT_FOUND", "missing")
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- response helpers ---------------------------------------------------

    def _json(self, status, body: Any, *, etag: str | None = None) -> None:
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "private, no-cache")
        if etag is not None:
            self.send_header("ETag", '"' + etag + '"')
        self.end_headers()
        self.wfile.write(encoded)

    def _problem(self, status, code: str, message: str, *, category: str = "validation",
                retryable: bool = False) -> None:
        """The real ``Problem`` envelope (``problem.v1.0``,
        `engine/v2/serving/api.py::_problem`), field for field -- NO
        ``title``/``status`` alias (P3-2 review: the exception handler
        returns ``exc.problem`` verbatim). ``ui/src/api/types.ts``
        ``ProblemEnvelope`` and ``ui/src/api/client.ts`` read exactly these
        names; the HTTP status is read from the response itself, not from
        this body."""
        self._json(status, {
            "code": code, "category": category, "retryable": retryable, "message": message,
            "stage": None, "trace_id": None, "dependency_refs": [], "retry_after_seconds": None,
            "diagnostic_ref": None, "details": {}, "schema_version": "problem.v1.0",
        })

    def log_message(self, format, *args):  # noqa: A002
        return


def create_mock_server(state: MockState, address: tuple[str, int] = ("127.0.0.1", 0)):
    server = http.server.ThreadingHTTPServer(address, MockApiHandler)
    server.state = state
    return server


def serve_in_thread(state: MockState, address: tuple[str, int] = ("127.0.0.1", 0)):
    server = create_mock_server(state, address)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
