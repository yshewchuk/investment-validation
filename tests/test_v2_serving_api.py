"""P3-2: the read API -- rearchitecture phase-3 guide §6.

Real ``serving.sqlite`` (built with ``projections.build_candidate`` over
synthetic score/bundle pairs, reusing ``tests/test_v2_serving_projections.py``'s
own fixtures/helpers) served over REAL HTTP: a real ``uvicorn.Server`` bound
to an ephemeral loopback port in a background thread for the bulk of the
route/auth/pagination/cursor/ETag cases (the same "bind port 0, poll
``server.started``, real socket" technique ``tests/test_v2_ops_serving.py``
already uses for the stdlib operations server), plus real subprocesses for
the launcher-refusal cases and the no-scoring/provider import guard, where a
SEPARATE process is the only way to make either check meaningful. Never
``TestClient`` alone.
"""
from __future__ import annotations

import hmac
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request as URLRequest
from urllib.request import urlopen

import pytest
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import Problem  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore, content_hash  # noqa: E402
from engine.v2.serving import api as api_module  # noqa: E402
from engine.v2.serving import projections  # noqa: E402
from engine.v2.serving.api import create_app  # noqa: E402
from tests.data_scan_support import catalog_and_store, commit_tables  # noqa: E402
from tests.test_v2_serving_projections import (  # noqa: E402
    _bundle,
    _compact,
    _event_row,
    _events_snapshot,
    _preview_input,
    _row,
    _score_doc,
)

TOKEN = "test-token-3f8c"

# --------------------------------------------------------------------------
# real HTTP over a real socket -- uvicorn.Server in a background thread
# --------------------------------------------------------------------------


def _start(app) -> tuple[uvicorn.Server, threading.Thread, str]:
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


def _stop(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=5)


def _get(base: str, path: str, *, token: str | None = None, cookie_token: str | None = None,
        params: dict | None = None, headers: dict | None = None):
    url = base + path
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    if clean:
        url += "?" + urlencode(clean)
    request = URLRequest(url)
    if token is not None:
        request.add_header("Authorization", "Bearer " + token)
    if cookie_token is not None:
        request.add_header("Cookie", "operations_token=" + cookie_token)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        response = urlopen(request, timeout=10)
        return response.getcode(), response.read(), response.headers
    except HTTPError as error:
        return error.code, error.read(), error.headers


# --------------------------------------------------------------------------
# synthetic two-release serving fixture
# --------------------------------------------------------------------------


def _shared_row() -> dict:
    """Byte-identical across both releases (same ticker/event_date/strike and
    every other field), so it hashes to the SAME `score_id` in both --
    `test_score_detail_scopes_a_shared_score_id_to_the_requested_release`'s
    fixture for §5.4's "event/score IDs are unique within a release" (not
    globally)."""
    return _row(ticker="T6", event_date="2024-01-07", strike=999.0)


def _rows_for(strike_base: float) -> list[dict]:
    rows = [_row(ticker=f"T{i}", event_date=f"2024-01-0{i + 1}", strike=strike_base + i) for i in range(5)]
    rows.append(_row(ticker="T5", event_date="2024-01-06", strike=strike_base + 5,
                     gate_pass=False, exp_pnl_model=None, win_model=None,
                     detail="gate score below threshold"))
    rows.append(_shared_row())
    return rows


def _two_release_app(tmp_path):
    (tmp_path / "phase2").mkdir()
    conn, store, snap = _events_snapshot(
        tmp_path / "phase2", [_event_row(f"e{i}", f"T{i}", datetime(2024, 1, i + 1)) for i in range(7)])
    repo = Repository(conn, store)
    serving_root = tmp_path / "serving"
    serving_root.mkdir()
    serving_store = ArtifactStore(serving_root / "objects")
    serving_conn = projections.connect(str(serving_root / "serving.sqlite"))

    rows_a = _rows_for(100.0)
    release_a = projections.build_candidate(
        _preview_input(), _score_doc(rows=rows_a), _bundle(*[_compact(r) for r in rows_a]),
        repository=repo, snapshot_ref=snap, store=serving_store, conn=serving_conn,
        requested_as_of="2024-01-04", resolved_as_of="2024-01-04")
    rows_b = _rows_for(200.0)
    release_b = projections.build_candidate(
        _preview_input(), _score_doc(rows=rows_b), _bundle(*[_compact(r) for r in rows_b]),
        repository=repo, snapshot_ref=snap, store=serving_store, conn=serving_conn,
        requested_as_of="2024-01-05", resolved_as_of="2024-01-05")
    serving_conn.close()

    # §5.4/P3-1c: the one authoritative pointer this API resolves "current"
    # through is the existing fenced ops publisher's own CURRENT, never a
    # bare file directly under serving_root -- nested under serving_root
    # only so `live`'s 4-tuple shape (every existing call site already
    # destructures it) does not need to grow a fifth element for this.
    publication_root = serving_root / "_publication"
    app = create_app(serving_db=str(serving_root / "serving.sqlite"),
                     store_root=str(serving_root / "objects"), serving_root=str(serving_root),
                     publication_root=str(publication_root), token=TOKEN)
    return app, serving_root, release_a, release_b


def _write_publication(serving_root: Path, ops_release_id: str, binding: dict | None) -> None:
    """Materialize a minimal fenced-publisher release directory carrying
    just a ``projection_binding.json`` (mirroring what ``publication_
    effect`` binds -- §5.4/P3-1c) and move the ops ``CURRENT`` pointer to
    it. Real gate/staging machinery is ops-side coverage
    (``tests/test_v2_ops_effects_graph.py``,
    ``tests/test_v2_serving_publication_binding.py``); this helper only
    reproduces the on-disk SHAPE the API's own resolver reads. ``binding``
    of ``None`` writes an ops release with no bound projection at all (the
    plain P3-0 compatibility-preview case)."""
    publication_root = serving_root / "_publication"
    release_dir = publication_root / "releases" / ops_release_id
    release_dir.mkdir(parents=True, exist_ok=True)
    if binding is not None:
        (release_dir / "projection_binding.json").write_text(json.dumps(binding))
    (publication_root / "CURRENT").write_text(ops_release_id)


def _ops_release_id(release_id: str) -> str:
    return "ops" + content_hash({"ops_release_for": release_id}).split(":")[1][:24]


def _set_current(serving_root: Path, release_id: str) -> None:
    conn = projections.connect(str(serving_root / "serving.sqlite"))
    try:
        binding = projections.projection_binding(conn, release_id)
    finally:
        conn.close()
    assert binding is not None, f"{release_id} is not a committed candidate"
    _write_publication(serving_root, _ops_release_id(release_id), binding)


@pytest.fixture
def live(tmp_path):
    app, serving_root, release_a, release_b = _two_release_app(tmp_path)
    server, thread, base = _start(app)
    try:
        yield base, serving_root, release_a, release_b
    finally:
        _stop(server, thread)


# --------------------------------------------------------------------------
# auth: every route, missing and invalid token, never echoed
# --------------------------------------------------------------------------


_ROUTES = (
    ("/api/v1/releases/current", {}),
    ("/api/v1/releases/{release_id}", {}),
    ("/api/v1/events", {}),
    ("/api/v1/events/{event_id}/scores", {}),
    ("/api/v1/scores/{score_id}", {}),
    ("/api/v1/operations", {}),
)


def test_auth_missing_or_invalid_token_401_on_every_route_and_token_never_echoed(live):
    base, serving_root, release_a, _ = live
    _set_current(serving_root, release_a.release_id)
    placeholders = {"release_id": release_a.release_id, "event_id": "e0", "score_id": "anything"}
    for template, params in _ROUTES:
        path = template.format(**placeholders)
        for token in (None, "wrong-token"):
            code, body, _ = _get(base, path, token=token, params=params)
            assert code == 401, path
            document = json.loads(body)
            assert document["code"] == "UNAUTHORIZED"
            assert TOKEN not in body.decode("utf-8")


def test_cookie_only_auth_works_on_every_route(live):
    """`ui/src/api/client.ts` never sends `Authorization` -- every request
    carries `credentials: "same-origin"` and relies solely on a same-origin
    `operations_token` cookie, mirroring `engine/v2/serving/operations.py`
    exactly. Every route must accept cookie-only auth."""
    base, serving_root, release_a, _ = live
    _set_current(serving_root, release_a.release_id)
    placeholders = {"release_id": release_a.release_id, "event_id": "e0", "score_id": "anything"}
    for template, params in _ROUTES:
        path = template.format(**placeholders)
        code, body, _ = _get(base, path, cookie_token=TOKEN, params=params)
        assert code != 401, (path, body)


# --------------------------------------------------------------------------
# releases/current, releases/{id}: 503, 404, typed errors, ETag/304
# --------------------------------------------------------------------------


def test_no_current_release_returns_503(live):
    base, _, _, _ = live
    code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
    assert code == 503
    assert json.loads(body)["code"] == "NO_CURRENT_RELEASE"


def test_published_release_with_no_bound_projection_is_no_current_release(live):
    """The plain P3-0 compatibility-preview case (an ops release published
    with a bundle but no bound projection candidate at all) is the ordinary
    "not configured yet" 503 -- never the distinct binding-integrity
    refusal, which is reserved for a binding that names a release id and
    then fails to verify."""
    base, serving_root, _, _ = live
    _write_publication(serving_root, "ops-bundle-only", binding=None)
    code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
    assert code == 503
    assert json.loads(body)["code"] == "NO_CURRENT_RELEASE"


def test_current_pointer_names_an_uncommitted_projection_is_a_typed_integrity_refusal(live):
    """§5.4/P3-1c: "refuse ... if the projection isn't fully committed in
    serving.sqlite" -- a bound projection_binding.json naming a release id
    the index never actually committed is CURRENT_BINDING_INVALID, distinct
    from NO_CURRENT_RELEASE (`test_no_current_release_returns_503`)."""
    base, serving_root, _, _ = live
    _write_publication(serving_root, "ops-bogus", {
        "schema_version": "projection_binding.v1.0",
        "projection_release_id": "not_a_real_release_id",
        "source_release_id": "src", "projection_manifest_ref": "art_x",
        "projection_manifest_hash": "sha256:" + "0" * 64,
        "serving_index_identity": "sha256:" + "0" * 64,
        "comparison_receipt_refs": [], "requested_as_of": "2024-01-01", "resolved_as_of": "2024-01-01"})
    code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
    assert code == 500
    assert json.loads(body)["code"] == "CURRENT_BINDING_INVALID"


def test_current_pointer_binding_manifest_hash_mismatch_is_a_typed_integrity_refusal(live):
    """§5.4/P3-1c: a real, committed candidate whose bound
    projection_manifest_hash has been tampered (or drifted from the index)
    still refuses closed, never silently trusting a stale/edited hash."""
    base, serving_root, release_a, _ = live
    conn = projections.connect(str(serving_root / "serving.sqlite"))
    try:
        binding = dict(projections.projection_binding(conn, release_a.release_id))
    finally:
        conn.close()
    binding["projection_manifest_hash"] = "sha256:" + "f" * 64
    _write_publication(serving_root, _ops_release_id(release_a.release_id), binding)
    code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
    assert code == 500
    assert json.loads(body)["code"] == "CURRENT_BINDING_INVALID"


def test_release_current_resolves_and_matches_explicit_lookup(live):
    base, serving_root, release_a, _ = live
    _set_current(serving_root, release_a.release_id)
    code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
    assert code == 200
    assert json.loads(body)["release_id"] == release_a.release_id


def test_release_current_is_no_store_with_etag_304_for_the_4s_poll(live):
    """The UI polls `/api/v1/releases/current` roughly every 4s (guide §7,
    `ui/src/hooks.ts` `usePinnedRelease`'s background poll) to detect a
    release switch -- coordinator instruction: "no-store, small body, and
    ETag/304 where it fits". `no-store` alone (this route's original
    design) and ETag/304 are not mutually exclusive: no-store says a cache
    must not reuse the response without asking again; ETag/304 makes asking
    again cheap when nothing changed."""
    base, serving_root, release_a, _ = live
    _set_current(serving_root, release_a.release_id)
    code, _, headers = _get(base, "/api/v1/releases/current", token=TOKEN)
    assert code == 200
    assert headers.get("Cache-Control") == "no-store"
    etag = headers.get("ETag")
    assert etag
    code2, body2, headers2 = _get(base, "/api/v1/releases/current", token=TOKEN,
                                  headers={"If-None-Match": etag})
    assert code2 == 304
    assert body2 == b""
    assert headers2.get("Cache-Control") == "no-store"


def test_releases_by_id_unknown_is_404(live):
    base, _, _, _ = live
    code, body, _ = _get(base, "/api/v1/releases/unknown-id", token=TOKEN)
    assert code == 404
    assert json.loads(body)["code"] == "UNKNOWN_RELEASE"


def test_etag_gives_304(live):
    base, _, release_a, _ = live
    code, _, headers = _get(base, "/api/v1/releases/" + release_a.release_id, token=TOKEN)
    assert code == 200
    etag = headers.get("ETag")
    assert etag
    code2, body2, _ = _get(base, "/api/v1/releases/" + release_a.release_id, token=TOKEN,
                           headers={"If-None-Match": etag})
    assert code2 == 304
    assert body2 == b""


# --------------------------------------------------------------------------
# events: pagination, filters, cursor integrity, limit
# --------------------------------------------------------------------------


def test_pagination_full_walk_is_complete_ordered_and_has_no_duplicates(live):
    base, _, release_a, _ = live
    seen: list[str] = []
    cursor = None
    for _ in range(10):
        params = {"release_id": release_a.release_id, "limit": "2"}
        if cursor:
            params["cursor"] = cursor
        code, body, _ = _get(base, "/api/v1/events", token=TOKEN, params=params)
        assert code == 200
        document = json.loads(body)
        assert document["total_matching"] == 7
        seen.extend(item["event_ref"]["event_id"] for item in document["items"])
        cursor = document["next_cursor"]
        if cursor is None:
            break
    assert len(seen) == 7
    assert len(set(seen)) == 7
    assert seen == sorted(seen, key=lambda eid: int(eid[1:]))


def test_verdict_filter_selects_matching_events_and_summaries_consistently(live):
    base, _, release_a, _ = live
    code, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                         params={"release_id": release_a.release_id, "limit": "10", "verdict": "false"})
    assert code == 200
    document = json.loads(body)
    assert document["total_matching"] == 1
    assert [item["event_ref"]["event_id"] for item in document["items"]] == ["e5"]
    scores = document["items"][0]["scores"]
    assert scores and all(s["verdict"] == "false" for s in scores)
    assert scores[0]["refusal_reason"] == "gate score below threshold"


def test_limit_above_max_is_clamped_not_rejected(live):
    base, _, release_a, _ = live
    code, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                         params={"release_id": release_a.release_id, "limit": "9999"})
    assert code == 200
    document = json.loads(body)
    assert len(document["items"]) == 7  # every event fits; proves no 400/rejection


def test_limit_non_integer_is_422(live):
    base, _, release_a, _ = live
    code, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                         params={"release_id": release_a.release_id, "limit": "abc"})
    assert code == 422
    assert json.loads(body)["code"] == "INVALID_REQUEST"


#: §6/component_contracts.md §13.2 and `tests/fixtures/v2_ui_mock_api.py`
#: (`HTTPStatus.CONFLICT`) agree: a mismatched cursor is 409, not the 400
#: this task's own brief text used before the coordinator's UI-alignment
#: instruction settled the question in the guide/mock's favor.
_CURSOR_MISMATCH_STATUS = 409


def test_cursor_tampered_returns_409(live):
    base, _, release_a, _ = live
    _, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                      params={"release_id": release_a.release_id, "limit": "2"})
    cursor = json.loads(body)["next_cursor"]
    assert cursor is not None
    tampered = cursor[:-1] + ("0" if cursor[-1] != "0" else "1")
    code, body2, _ = _get(base, "/api/v1/events", token=TOKEN,
                          params={"release_id": release_a.release_id, "limit": "2", "cursor": tampered})
    assert code == _CURSOR_MISMATCH_STATUS
    assert json.loads(body2)["code"] == "CURSOR_MISMATCH"


def test_cursor_reused_with_a_different_query_returns_409(live):
    base, _, release_a, _ = live
    _, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                      params={"release_id": release_a.release_id, "limit": "2"})
    cursor = json.loads(body)["next_cursor"]
    code, body2, _ = _get(base, "/api/v1/events", token=TOKEN,
                          params={"release_id": release_a.release_id, "limit": "2",
                                  "cursor": cursor, "ticker": "T0"})
    assert code == _CURSOR_MISMATCH_STATUS
    assert json.loads(body2)["code"] == "CURSOR_MISMATCH"


def test_cursor_reused_with_a_different_release_returns_409(live):
    base, _, release_a, release_b = live
    _, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                      params={"release_id": release_a.release_id, "limit": "2"})
    cursor = json.loads(body)["next_cursor"]
    code, body2, _ = _get(base, "/api/v1/events", token=TOKEN,
                          params={"release_id": release_b.release_id, "limit": "2", "cursor": cursor})
    assert code == _CURSOR_MISMATCH_STATUS
    assert json.loads(body2)["code"] == "CURSOR_MISMATCH"


def test_date_range_filter_uses_client_param_names(live):
    """`ui/src/api/client.ts`'s `EventQuery` sends `date_from`/`date_to`,
    not `event_date_from`/`event_date_to` -- this pins the wire name."""
    base, _, release_a, _ = live
    code, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                         params={"release_id": release_a.release_id, "limit": "10",
                                 "date_from": "2024-01-03", "date_to": "2024-01-04"})
    assert code == 200
    document = json.loads(body)
    assert [item["event_ref"]["event_id"] for item in document["items"]] == ["e2", "e3"]


# --------------------------------------------------------------------------
# release switch mid pagination / mid detail
# --------------------------------------------------------------------------


def test_release_switch_mid_pagination_keeps_pinned_release_new_session_gets_new_one(live):
    base, serving_root, release_a, release_b = live
    _set_current(serving_root, release_a.release_id)
    code, body, _ = _get(base, "/api/v1/events", token=TOKEN, params={"limit": "2"})
    assert code == 200
    first = json.loads(body)
    assert first["release_id"] == release_a.release_id
    cursor = first["next_cursor"]
    assert cursor is not None

    _set_current(serving_root, release_b.release_id)

    # A client that keeps carrying the resolved release forward stays pinned.
    code, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                         params={"limit": "2", "cursor": cursor, "release_id": release_a.release_id})
    assert code == 200
    assert json.loads(body)["release_id"] == release_a.release_id

    # A fresh resolution (no cursor, no explicit release) now returns the new one.
    code, body, _ = _get(base, "/api/v1/events", token=TOKEN, params={"limit": "2"})
    assert code == 200
    assert json.loads(body)["release_id"] == release_b.release_id


def test_release_switch_mid_detail_keeps_pinned_release_new_session_gets_new_one(live):
    base, serving_root, release_a, release_b = live
    _set_current(serving_root, release_a.release_id)
    code, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                         params={"release_id": release_a.release_id, "limit": "1"})
    score_id_a = json.loads(body)["items"][0]["scores"][0]["score_id"]

    _set_current(serving_root, release_b.release_id)

    code, body, _ = _get(base, "/api/v1/scores/" + score_id_a, token=TOKEN,
                         params={"release_id": release_a.release_id})
    assert code == 200
    assert json.loads(body)["score_id"] == score_id_a

    code, body, _ = _get(base, "/api/v1/events", token=TOKEN, params={"limit": "1"})
    assert json.loads(body)["release_id"] == release_b.release_id


# --------------------------------------------------------------------------
# events/{id}/scores
# --------------------------------------------------------------------------


def test_event_scores_requires_release_id(live):
    """§5.4: event/score IDs are unique only WITHIN a release, so this route
    must be given one (P3-2 review) -- a missing `release_id` is 400
    `RELEASE_ID_REQUIRED`, distinct from the 404 an unknown-but-present one
    gets below."""
    base, _, _, _ = live
    code, body, _ = _get(base, "/api/v1/events/e0/scores", token=TOKEN)
    assert code == 400
    assert json.loads(body)["code"] == "RELEASE_ID_REQUIRED"


def test_event_scores_unknown_event_is_404(live):
    base, _, release_a, _ = live
    code, body, _ = _get(base, "/api/v1/events/unknown/scores", token=TOKEN,
                         params={"release_id": release_a.release_id})
    assert code == 404
    assert json.loads(body)["code"] == "UNKNOWN_EVENT"


def test_event_scores_response_is_a_bare_array_and_includes_refusals(live):
    """§6 names no envelope beyond "strategy score summaries"; `ui/src/api/
    client.ts` (`Promise<EventScoreSummary[]>`) and the mock server both
    serve a bare array, so this does too -- not `{release_id, event_id,
    scores}`."""
    base, _, release_a, _ = live
    code, body, _ = _get(base, "/api/v1/events/e5/scores", token=TOKEN,
                         params={"release_id": release_a.release_id})
    assert code == 200
    scores = json.loads(body)
    assert isinstance(scores, list)
    assert len(scores) == 1
    assert scores[0]["verdict"] == "false"
    assert scores[0]["refusal_reason"] == "gate score below threshold"


def test_event_scores_unknown_release_is_404(live):
    base, _, _, _ = live
    code, body, _ = _get(base, "/api/v1/events/e0/scores", token=TOKEN, params={"release_id": "nope"})
    assert code == 404
    assert json.loads(body)["code"] == "UNKNOWN_RELEASE"


# --------------------------------------------------------------------------
# scores/{id}
# --------------------------------------------------------------------------


def test_score_detail_unknown_id_is_404(live):
    base, _, release_a, _ = live
    code, body, _ = _get(base, "/api/v1/scores/unknown", token=TOKEN,
                         params={"release_id": release_a.release_id})
    assert code == 404
    assert json.loads(body)["code"] == "UNKNOWN_SCORE"


def test_score_detail_membership_validated_when_release_supplied(live):
    base, _, release_a, release_b = live
    code, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                         params={"release_id": release_a.release_id, "limit": "1"})
    score_id = json.loads(body)["items"][0]["scores"][0]["score_id"]
    code, body, _ = _get(base, "/api/v1/scores/" + score_id, token=TOKEN,
                         params={"release_id": release_b.release_id})
    assert code == 404
    assert json.loads(body)["code"] == "UNKNOWN_SCORE"


def test_score_detail_requires_release_id(live):
    """§5.4: score IDs are unique only WITHIN a release; a client always
    pins one (P3-2 review). A missing `release_id` is 400
    `RELEASE_ID_REQUIRED`, never a cross-release search."""
    base, _, _, _ = live
    code, body, _ = _get(base, "/api/v1/scores/anything", token=TOKEN)
    assert code == 400
    assert json.loads(body)["code"] == "RELEASE_ID_REQUIRED"


def test_score_detail_scopes_a_shared_score_id_to_the_requested_release(live):
    """`_shared_row` is byte-identical in both releases, so it hashes to the
    SAME `score_id` in `release_a` and `release_b` (a content-addressed id
    that two releases share necessarily shares its content, by
    construction) -- exercising exactly the shared-id case §5.4's
    within-a-release uniqueness rule is about: fetching it under EITHER
    explicit release succeeds and returns that id's own record, never an
    error from treating the id as ambiguous or globally owned by one
    release. Paired with `test_score_detail_membership_validated_when_
    release_supplied` (a score minted under `release_a` alone, requested
    under `release_b`, 404s rather than falling back) this proves the route
    is scoped strictly to the requested release in both directions."""
    base, _, release_a, release_b = live
    code, body, _ = _get(base, "/api/v1/events/e6/scores", token=TOKEN,
                         params={"release_id": release_a.release_id})
    score_id = json.loads(body)[0]["score_id"]
    code, body, _ = _get(base, "/api/v1/events/e6/scores", token=TOKEN,
                         params={"release_id": release_b.release_id})
    assert json.loads(body)[0]["score_id"] == score_id  # fixture sanity: truly shared

    for release in (release_a, release_b):
        code, body, _ = _get(base, "/api/v1/scores/" + score_id, token=TOKEN,
                             params={"release_id": release.release_id})
        assert code == 200
        document = json.loads(body)
        assert document["score_id"] == score_id
        assert document["engine_record"]["ticker"] == "T6"


# --------------------------------------------------------------------------
# operations
# --------------------------------------------------------------------------


def test_operations_route_unavailable_without_health_document(live):
    base, _, _, _ = live
    code, body, _ = _get(base, "/api/v1/operations", token=TOKEN)
    assert code == 503
    assert json.loads(body)["code"] == "OPERATIONS_UNAVAILABLE"


def test_operations_route_serves_a_valid_health_document(live):
    base, serving_root, _, _ = live
    (serving_root / "health.json").write_text(json.dumps(
        {"schema_version": "operations_health.v1.0", "generated_at": "t1", "withheld_release": None}))
    code, body, _ = _get(base, "/api/v1/operations", token=TOKEN)
    assert code == 200
    assert json.loads(body)["generated_at"] == "t1"


# --------------------------------------------------------------------------
# no-scoring/provider guard -- real subprocess, real HTTP, two sys.modules
# snapshots (after startup, after a request pass)
# --------------------------------------------------------------------------


_GUARD_SCRIPT = """
import json, sys, threading, time, urllib.error, urllib.request
sys.path.insert(0, {root!r})
from engine.v2.serving.api import create_app
app = create_app(serving_db={serving_db!r}, store_root={store_root!r},
                 serving_root={serving_root!r}, token={token!r})
after_create = sorted(sys.modules)
import uvicorn
config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
server = uvicorn.Server(config)
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
deadline = time.monotonic() + 10
while not server.started and time.monotonic() < deadline:
    time.sleep(0.01)
port = server.servers[0].sockets[0].getsockname()[1]
req = urllib.request.Request("http://127.0.0.1:" + str(port) + "/api/v1/operations",
                             headers={{"Authorization": "Bearer " + {token!r}}})
try:
    urllib.request.urlopen(req, timeout=10).read()
except urllib.error.HTTPError:
    pass
after_request = sorted(sys.modules)
server.should_exit = True
thread.join(timeout=5)
print(json.dumps({{"after_create": after_create, "after_request": after_request}}))
"""

_FORBIDDEN_MODULE_SUBSTRINGS = ("engine.score", "engine.v2.ops", "engine.data.pulls",
                                "engine.data.sources", "yfinance")


def test_no_scoring_or_provider_import_after_startup_and_after_a_request(tmp_path):
    script = _GUARD_SCRIPT.format(root=str(ROOT), serving_db=str(tmp_path / "s.sqlite"),
                                  store_root=str(tmp_path / "objects"),
                                  serving_root=str(tmp_path / "serving"), token=TOKEN)
    script_path = tmp_path / "guard.py"
    script_path.write_text(script)
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(ROOT),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    for key in ("after_create", "after_request"):
        modules = payload[key]
        for forbidden in _FORBIDDEN_MODULE_SUBSTRINGS:
            hit = [m for m in modules if forbidden in m]
            assert not hit, f"{key}: forbidden module(s) imported: {hit}"


# --------------------------------------------------------------------------
# launcher: refusals, and a real subprocess round trip
# --------------------------------------------------------------------------


def _launcher_argv(tmp_path, *, host="127.0.0.1", port="0", extra=()):
    return [sys.executable, "-m", "engine.v2.serving.api", "--host", host, "--port", str(port),
           "--serving-db", str(tmp_path / "s.sqlite"), "--store-root", str(tmp_path / "objects"),
           "--serving-root", str(tmp_path / "serving"), *extra]


def test_launcher_refuses_without_token(tmp_path):
    env = dict(os.environ)
    env.pop("V2_DASHBOARD_TOKEN", None)
    result = subprocess.run(_launcher_argv(tmp_path), cwd=str(ROOT), env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert TOKEN not in result.stdout and TOKEN not in result.stderr


def test_launcher_refuses_non_loopback_host_without_flag(tmp_path):
    env = dict(os.environ, V2_DASHBOARD_TOKEN=TOKEN)
    result = subprocess.run(_launcher_argv(tmp_path, host="8.8.8.8"), cwd=str(ROOT), env=env,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0


def test_launcher_allows_non_loopback_host_with_flag_and_serves(tmp_path):
    env = dict(os.environ, V2_DASHBOARD_TOKEN=TOKEN)
    process = subprocess.Popen(
        _launcher_argv(tmp_path, host="127.0.0.1", extra=("--allow-non-loopback",)),
        cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert process.poll() is None
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


# --------------------------------------------------------------------------
# coverage: error branches, resolver refusals, launcher (P3-2 review point 1)
#
# Unit-level, importing the modules' own private helpers directly rather
# than contriving an HTTP scenario for every internal branch -- exactly what
# the review asked for ("error branches, launcher, resolver refusals").
# `main()`/`_parse_args` are called IN-PROCESS here (not via subprocess) so
# `coverage run` (which never instruments a spawned child) can see them; the
# real subprocess launcher tests above stay as the end-to-end proof.
# --------------------------------------------------------------------------


def test_authorized_is_false_without_a_token_before_touching_the_request():
    assert api_module._authorized(None, "") is False


def test_read_ops_current_refuses_a_symlinked_current_pointer(tmp_path):
    real = tmp_path / "real"
    real.write_text("some-release")
    link = tmp_path / "CURRENT"
    link.symlink_to(real)
    assert api_module._read_ops_current(tmp_path) is None


def test_read_ops_current_refuses_traversal_content(tmp_path):
    (tmp_path / "CURRENT").write_text("../escape")
    assert api_module._read_ops_current(tmp_path) is None


def test_read_ops_current_refuses_multi_segment_content(tmp_path):
    (tmp_path / "CURRENT").write_text("a/b")
    assert api_module._read_ops_current(tmp_path) is None


def test_read_ops_current_missing_pointer_is_none(tmp_path):
    assert api_module._read_ops_current(tmp_path) is None


def test_read_projection_binding_refuses_a_symlinked_binding_file(tmp_path):
    release_dir = tmp_path / "releases" / "rel1"
    release_dir.mkdir(parents=True)
    real = tmp_path / "real_binding.json"
    real.write_text(json.dumps({"projection_release_id": "x"}))
    (release_dir / "projection_binding.json").symlink_to(real)
    assert api_module._read_projection_binding(tmp_path, "rel1") is None


def test_read_projection_binding_refuses_an_unsafe_release_id(tmp_path):
    assert api_module._read_projection_binding(tmp_path, "../escape") is None
    assert api_module._read_projection_binding(tmp_path, "a/b") is None


def test_read_projection_binding_missing_file_is_none(tmp_path):
    (tmp_path / "releases" / "rel1").mkdir(parents=True)
    assert api_module._read_projection_binding(tmp_path, "rel1") is None


def test_read_projection_binding_malformed_json_is_none(tmp_path):
    release_dir = tmp_path / "releases" / "rel1"
    release_dir.mkdir(parents=True)
    (release_dir / "projection_binding.json").write_text("{not json")
    assert api_module._read_projection_binding(tmp_path, "rel1") is None


def test_publication_resolver_with_no_publication_root_or_resolver_is_no_current_release(tmp_path):
    app = create_app(serving_db=str(tmp_path / "s.sqlite"), store_root=str(tmp_path / "o"),
                     serving_root=str(tmp_path), token=TOKEN)
    server, thread, base = _start(app)
    try:
        code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
        assert code == 503
        assert json.loads(body)["code"] == "NO_CURRENT_RELEASE"
    finally:
        _stop(server, thread)


def test_unpack_cursor_without_a_dot_separator_is_malformed():
    key = api_module._cursor_key(TOKEN)
    with pytest.raises(api_module.ApiError) as excinfo:
        api_module._unpack_cursor(key, "no-dot-here")
    assert excinfo.value.problem["code"] == "CURSOR_MISMATCH"


def test_unpack_cursor_payload_with_wrong_field_count_is_malformed():
    key = api_module._cursor_key(TOKEN)
    payload = b"only-one-field"
    signature = hmac.new(key, payload, digestmod="sha256").hexdigest()
    cursor = payload.hex() + "." + signature
    with pytest.raises(api_module.ApiError) as excinfo:
        api_module._unpack_cursor(key, cursor)
    assert excinfo.value.problem["code"] == "CURSOR_MISMATCH"


def test_create_app_refuses_an_empty_token(tmp_path):
    with pytest.raises(ValueError):
        create_app(serving_db=str(tmp_path / "s.sqlite"), store_root=str(tmp_path / "o"),
                  serving_root=str(tmp_path), token="")


def test_limit_zero_or_negative_is_422(live):
    base, _, release_a, _ = live
    for limit in ("0", "-1"):
        code, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                             params={"release_id": release_a.release_id, "limit": limit})
        assert code == 422, limit
        assert json.loads(body)["code"] == "INVALID_REQUEST"


def test_date_from_bad_format_is_422(live):
    base, _, release_a, _ = live
    code, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                         params={"release_id": release_a.release_id, "date_from": "not-a-date"})
    assert code == 422
    assert json.loads(body)["code"] == "INVALID_REQUEST"


def test_operations_route_unavailable_on_malformed_health_json(live):
    base, serving_root, _, _ = live
    (serving_root / "health.json").write_text("{not json")
    code, body, _ = _get(base, "/api/v1/operations", token=TOKEN)
    assert code == 503
    assert json.loads(body)["code"] == "OPERATIONS_UNAVAILABLE"


def test_operations_route_unavailable_on_wrong_schema_version(live):
    base, serving_root, _, _ = live
    (serving_root / "health.json").write_text(json.dumps({"schema_version": "not_the_right.v1.0"}))
    code, body, _ = _get(base, "/api/v1/operations", token=TOKEN)
    assert code == 503
    assert json.loads(body)["code"] == "OPERATIONS_UNAVAILABLE"


def test_events_with_no_release_id_and_no_current_is_503(live):
    base, _, _, _ = live  # CURRENT never set by this fixture
    code, body, _ = _get(base, "/api/v1/events", token=TOKEN)
    assert code == 503
    assert json.loads(body)["code"] == "NO_CURRENT_RELEASE"


def test_event_scores_clock_id_mismatch_is_422(live):
    base, _, release_a, _ = live
    code, body, _ = _get(base, "/api/v1/events/e0/scores", token=TOKEN,
                         params={"release_id": release_a.release_id, "clock_id": "not-the-real-clock"})
    assert code == 422
    assert json.loads(body)["code"] == "INVALID_REQUEST"


def test_ticker_filter_alone(live):
    base, _, release_a, _ = live
    code, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                         params={"release_id": release_a.release_id, "ticker": "T0"})
    assert code == 200
    document = json.loads(body)
    assert [item["ticker"] for item in document["items"]] == ["T0"]


def test_strategy_filter_alone(live):
    base, _, release_a, _ = live
    code, body, _ = _get(base, "/api/v1/events", token=TOKEN,
                         params={"release_id": release_a.release_id, "limit": "10", "strategy": "STR-THRU"})
    assert code == 200
    document = json.loads(body)
    assert document["total_matching"] == 7
    for item in document["items"]:
        assert all(s["strategy"] == "STR-THRU" for s in item["scores"])


def test_main_refuses_without_token_in_process(tmp_path, monkeypatch):
    monkeypatch.delenv("V2_DASHBOARD_TOKEN", raising=False)
    argv = ["--port", "0", "--serving-db", str(tmp_path / "s.sqlite"),
           "--store-root", str(tmp_path / "o"), "--serving-root", str(tmp_path)]
    assert api_module.main(argv) == 2


def test_main_refuses_non_loopback_without_flag_in_process(tmp_path, monkeypatch):
    monkeypatch.setenv("V2_DASHBOARD_TOKEN", TOKEN)
    argv = ["--host", "8.8.8.8", "--port", "0", "--serving-db", str(tmp_path / "s.sqlite"),
           "--store-root", str(tmp_path / "o"), "--serving-root", str(tmp_path)]
    assert api_module.main(argv) == 2


def test_main_starts_uvicorn_when_allowed_in_process(tmp_path, monkeypatch):
    """Patches ``uvicorn.run`` to a no-op so ``main()``'s success path
    (build the real app, hand it to uvicorn, return 0) runs without
    actually binding a socket or blocking -- the only way to reach that
    line under `coverage run`, which never instruments a subprocess."""
    monkeypatch.setenv("V2_DASHBOARD_TOKEN", TOKEN)
    calls = []
    monkeypatch.setattr(api_module.uvicorn, "run", lambda app, **kw: calls.append(kw))
    argv = ["--host", "127.0.0.1", "--port", "0", "--serving-db", str(tmp_path / "s.sqlite"),
           "--store-root", str(tmp_path / "o"), "--serving-root", str(tmp_path)]
    assert api_module.main(argv) == 0
    assert calls == [{"host": "127.0.0.1", "port": 0}]


# -- projections.py: internal error branches (ServingIndexError, migration
# integrity, empty-catalog resolver, malformed raw cursor) --------------


def test_serving_index_error_carries_its_problem():
    problem = Problem(code="X", category="integrity", retryable=False, message="boom")
    error = projections.ServingIndexError(problem)
    assert "X: boom" in str(error)
    assert error.problem is problem


def test_transaction_refuses_nesting(tmp_path):
    conn = projections.connect(str(tmp_path / "s.sqlite"))
    conn.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(RuntimeError):
            with projections._transaction(conn):
                pass
    finally:
        conn.execute("ROLLBACK")
        conn.close()


def test_check_applied_refuses_a_newer_unknown_version():
    with pytest.raises(projections.ServingIndexError):
        projections._check_applied({1: "a"}, {1: "a", 2: "b"})


def test_check_applied_refuses_a_changed_migration():
    with pytest.raises(projections.ServingIndexError):
        projections._check_applied({1: "a"}, {1: "different"})


def test_resolve_event_refs_with_no_events_table_is_empty(tmp_path):
    (tmp_path / "phase2").mkdir()
    conn, clock, store = catalog_and_store(tmp_path / "phase2")
    snap = commit_tables(conn, clock, {}, {})  # no tables committed at all
    repo = Repository(conn, store)
    result = projections.resolve_event_refs(repo, snap, {("AAA", "2024-01-01")})
    assert result == {}


def test_load_ref_refuses_an_unregistered_artifact_id(tmp_path):
    conn = projections.connect(str(tmp_path / "s.sqlite"))
    with pytest.raises(projections.ServingIndexError):
        projections._load_ref(conn, "no-such-artifact")
    conn.close()


def test_decode_cursor_refuses_a_malformed_raw_cursor():
    with pytest.raises(projections.ServingIndexError):
        projections._decode_cursor("not-enough-fields")
