"""Issue #342: an API-owned serving-index read that fails AFTER
``projections.connect`` succeeds -- a ``sqlite3.Error`` from a corrupt-but-
openable index (a ``DROP TABLE`` the migrations still tolerate) or a
``json.JSONDecodeError`` from a stored malformed row -- must become HTTP 503
with the existing ``Problem`` envelope, code ``SERVING_INDEX_UNREADABLE``.

Durable regression coverage for that contract: every API-owned read boundary
(pinned release, explicit and resolved ``/events``, ``/releases/current``,
event scores, score detail), both stored-JSON variants, connection-leak
closure, auth short-circuit, conditional-request handling, and the artifact
negative control (a malformed *artifact*, not an index row, still 500s and
never ``SERVING_INDEX_UNREADABLE``) pin the boundary on either side. The
before-fix red evidence (500/untyped against the unchanged ``api.py``) is
recorded in the issue hand-back, not re-asserted here.

Reuses ``tests/v2/serving/test_v2_serving_api.py``'s real-HTTP fixture machinery (two
synthetic releases over a real ``serving.sqlite``, a ``uvicorn.Server`` on an
ephemeral loopback port) and ``engine.v2.serving.projections``' own read helpers;
never mocks a projection exception and never zeroes a header so ``connect``
itself fails.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from engine.paths import ROOT

sys.path.insert(0, str(ROOT))

from engine.v2.foundation import ArtifactStore  # noqa: E402
from engine.v2.serving import projections  # noqa: E402
from tests.v2.serving.test_v2_serving_api import (  # noqa: E402
    TOKEN,
    _get,
    _set_current,
    _start,
    _stop,
    _two_release_app,
)

#: A fixed invalid-JSON value written into exactly the row/field under test. A
#: distinctive sentinel so the leak-free assertions are meaningful.
INVALID = "[[[malformed-index-sentinel]]]"

#: The exact, complete 503 envelope the chosen contract requires -- fixed here,
#: NOT imported from a (yet nonexistent) production refusal helper. The
#: operational fields are ``api._problem``'s own defaults, so full-document
#: equality stays exact.
EXPECTED_503 = {
    "code": "SERVING_INDEX_UNREADABLE",
    "category": "resource",
    "retryable": True,
    "message": "the serving index cannot be read",
    "stage": None,
    "trace_id": None,
    "dependency_refs": [],
    "retry_after_seconds": None,
    "diagnostic_ref": None,
    "details": {},
    "schema_version": "problem.v1.0",
}


# --------------------------------------------------------------------------
# fixtures / small helpers
# --------------------------------------------------------------------------


@dataclass
class _Ctx:
    base: str
    serving_root: Path
    release_a: object
    db: Path
    tmp: Path

    @property
    def release_id(self) -> str:
        return self.release_a.release_id


@contextlib.contextmanager
def _live_app(tmp_path):
    app, serving_root, release_a, _release_b = _two_release_app(tmp_path)
    server, thread, base = _start(app)
    ctx = _Ctx(base=base, serving_root=serving_root, release_a=release_a,
               db=serving_root / "serving.sqlite", tmp=tmp_path)
    try:
        yield ctx
    finally:
        _stop(server, thread)


def _store(ctx: _Ctx) -> ArtifactStore:
    return ArtifactStore(str(ctx.serving_root / "objects"))


def _run_sql(ctx: _Ctx, sql: str, params: tuple = ()) -> None:
    """Damage the on-disk index through a real, autocommit connection.
    ``foreign_keys`` is switched off so ``DROP TABLE serving_release`` (whose
    own committed children reference it) does not fail on the implicit DELETE;
    the migrations in ``connect`` still see every ``schema_versions`` row, so
    the index stays openable and only the targeted later query fails."""
    conn = projections.connect(str(ctx.db))
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(sql, params)
    finally:
        conn.close()


def _binding_before_damage(ctx: _Ctx) -> dict:
    conn = projections.connect(str(ctx.db))
    try:
        return projections.projection_binding(conn, ctx.release_id)
    finally:
        conn.close()


def _score_id(ctx: _Ctx, event_id: str = "e0") -> str:
    code, body, _ = _get(ctx.base, f"/api/v1/events/{event_id}/scores", token=TOKEN,
                        params={"release_id": ctx.release_id})
    assert code == 200, (code, body[:200])
    return json.loads(body)[0]["score_id"]


def _detail_ref_before_damage(ctx: _Ctx, score_id: str) -> str:
    conn = projections.connect(str(ctx.db))
    try:
        row = conn.execute(
            "SELECT detail_artifact_id FROM serving_score_summary WHERE release_id = ? AND score_id = ?",
            (ctx.release_id, score_id)).fetchone()
        return row["detail_artifact_id"]
    finally:
        conn.close()


@contextlib.contextmanager
def _track_connects():
    """Record only the connections the API opens during the request. Probe /
    setup / damage / demonstration connections are opened outside this block,
    so they are excluded; ``api._open`` and the publication resolver both call
    ``projections.connect`` by attribute, so the patch is observed in the
    uvicorn worker thread too."""
    opened: list[sqlite3.Connection] = []
    original = projections.connect

    def wrapper(path, *, clock=None):
        conn = original(path, clock=clock)
        opened.append(conn)
        return conn

    projections.connect = wrapper
    try:
        yield opened
    finally:
        projections.connect = original


def _demonstrate(ctx: _Ctx, expect: type[BaseException], query) -> None:
    """Before any HTTP: open the damaged index for real and run the exact
    relevant projection query, proving it is the query -- not ``connect``, not
    a mock -- that raises. Guaranteed close."""
    conn = projections.connect(str(ctx.db))
    try:
        with pytest.raises(expect):
            query(conn)
    finally:
        conn.close()


def _assert_closed(opened: list[sqlite3.Connection]) -> None:
    assert opened, "expected at least one API-owned connection to have opened"
    for conn in opened:
        # These connections are created in a uvicorn worker thread. On this
        # Python, ``execute`` checks the creating thread BEFORE the handle, so
        # an OPEN leaked connection and a CLOSED one raise the same thread
        # ProgrammingError -- SELECT 1 cannot distinguish them. ``total_changes``
        # has no thread guard: a closed handle raises the real sqlite3
        # "Cannot operate on a closed database." ProgrammingError, while a
        # still-open handle would return a count and fail this test. Never
        # trusts a fictitious ``conn.closed`` property.
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.total_changes


def _assert_clean(body: bytes, ctx: _Ctx) -> None:
    text = body.decode("utf-8", "replace")
    for forbidden in (TOKEN, str(ctx.tmp), str(ctx.db), INVALID, "no such table",
                      "malformed JSON", "Traceback", "sqlite3", "detail_artifact_id",
                      "ref_json", "document_json"):
        assert forbidden not in text, forbidden


def _assert_envelope(body: bytes) -> None:
    assert json.loads(body) == EXPECTED_503


def _expect_unreadable(ctx: _Ctx, path: str, params: dict | None = None, *,
                       headers: dict | None = None, expected_connections: int = 1):
    opened: list[sqlite3.Connection] = []
    with _track_connects() as opened:
        code, body, resp = _get(ctx.base, path, token=TOKEN, params=params, headers=headers)
    assert code == 503, (code, body[:200])
    assert len(opened) == expected_connections, len(opened)
    _assert_envelope(body)
    _assert_clean(body, ctx)
    # Never a success-cache 304 nor an immutable 200 cache header on a failure.
    assert "immutable" not in (resp.get("Cache-Control") or "")
    _assert_closed(opened)
    return code, body, resp


# --------------------------------------------------------------------------
# 1. every API-owned read boundary, DROP TABLE (a real corrupt-but-openable
#    index; connect succeeds, the targeted later query raises sqlite3.Error)
# --------------------------------------------------------------------------


def test_pinned_release_drops_serving_release(ctx):
    _run_sql(ctx, "DROP TABLE serving_release")
    _demonstrate(ctx, sqlite3.Error, lambda conn: projections.get_release(conn, ctx.release_id))
    _expect_unreadable(ctx, "/api/v1/releases/" + ctx.release_id)


def test_events_explicit_release_drops_serving_event_summary(ctx):
    _run_sql(ctx, "DROP TABLE serving_event_summary")
    _demonstrate(ctx, sqlite3.Error, lambda conn: projections.list_events(conn, ctx.release_id))
    _expect_unreadable(ctx, "/api/v1/events", {"release_id": ctx.release_id})


def test_events_unpinned_resolver_drops_serving_release(ctx):
    _set_current(ctx.serving_root, ctx.release_id)
    binding = _binding_before_damage(ctx)
    _run_sql(ctx, "DROP TABLE serving_release")
    _demonstrate(ctx, sqlite3.Error, lambda conn: projections.verify_projection_binding(conn, binding))
    # No release pin: the API opens its own outer connection first, then the
    # resolver opens a second one to verify the binding -- both must close.
    _expect_unreadable(ctx, "/api/v1/events", expected_connections=2)


def test_release_current_resolver_drops_serving_release(ctx):
    _set_current(ctx.serving_root, ctx.release_id)
    binding = _binding_before_damage(ctx)
    _run_sql(ctx, "DROP TABLE serving_release")
    _demonstrate(ctx, sqlite3.Error, lambda conn: projections.verify_projection_binding(conn, binding))
    _expect_unreadable(ctx, "/api/v1/releases/current")


def test_event_scores_pinned_drops_serving_event_summary(ctx):
    _run_sql(ctx, "DROP TABLE serving_event_summary")
    _demonstrate(ctx, sqlite3.Error,
                 lambda conn: projections.get_event(conn, ctx.release_id, "e0"))
    _expect_unreadable(ctx, "/api/v1/events/e0/scores", {"release_id": ctx.release_id})


def test_score_detail_pinned_drops_serving_score_summary(ctx):
    score_id = _score_id(ctx, "e0")
    _run_sql(ctx, "DROP TABLE serving_score_summary")
    _demonstrate(ctx, sqlite3.Error,
                 lambda conn: projections.get_score_detail(conn, _store(ctx), ctx.release_id, score_id))
    _expect_unreadable(ctx, "/api/v1/scores/" + score_id, {"release_id": ctx.release_id})


# --------------------------------------------------------------------------
# 2. stored malformed JSON in exactly the row/field that read uses it
#    (real UPDATE; the relevant projection raises json.JSONDecodeError)
# --------------------------------------------------------------------------


def test_pinned_release_malformed_document_json(ctx):
    _run_sql(ctx, "UPDATE serving_release SET document_json = ? WHERE release_id = ?",
             (INVALID, ctx.release_id))
    _demonstrate(ctx, json.JSONDecodeError, lambda conn: projections.get_release(conn, ctx.release_id))
    _expect_unreadable(ctx, "/api/v1/releases/" + ctx.release_id)


def test_current_binding_malformed_document_json(ctx):
    _set_current(ctx.serving_root, ctx.release_id)
    binding = _binding_before_damage(ctx)
    _run_sql(ctx, "UPDATE serving_release SET document_json = ? WHERE release_id = ?",
             (INVALID, ctx.release_id))
    _demonstrate(ctx, json.JSONDecodeError,
                 lambda conn: projections.verify_projection_binding(conn, binding))
    _expect_unreadable(ctx, "/api/v1/releases/current")


def test_current_binding_malformed_findings_json(ctx):
    _set_current(ctx.serving_root, ctx.release_id)
    binding = _binding_before_damage(ctx)
    _run_sql(ctx, "UPDATE serving_release SET findings_json = ? WHERE release_id = ?",
             (INVALID, ctx.release_id))
    _demonstrate(ctx, json.JSONDecodeError,
                 lambda conn: projections.verify_projection_binding(conn, binding))
    _expect_unreadable(ctx, "/api/v1/releases/current")


def test_current_binding_malformed_object_ref_json(ctx):
    _set_current(ctx.serving_root, ctx.release_id)
    binding = _binding_before_damage(ctx)
    _run_sql(ctx, "UPDATE serving_object SET ref_json = ? WHERE artifact_id = ?",
             (INVALID, ctx.release_a.projection_manifest_ref))
    _demonstrate(ctx, json.JSONDecodeError,
                 lambda conn: projections.verify_projection_binding(conn, binding))
    _expect_unreadable(ctx, "/api/v1/releases/current")


def test_score_detail_malformed_object_ref_json(ctx):
    score_id = _score_id(ctx, "e0")
    detail_ref = _detail_ref_before_damage(ctx, score_id)
    _run_sql(ctx, "UPDATE serving_object SET ref_json = ? WHERE artifact_id = ?",
             (INVALID, detail_ref))
    _demonstrate(ctx, json.JSONDecodeError,
                 lambda conn: projections.get_score_detail(conn, _store(ctx), ctx.release_id, score_id))
    _expect_unreadable(ctx, "/api/v1/scores/" + score_id, {"release_id": ctx.release_id})


def test_events_list_malformed_score_flags(ctx):
    score_id = _score_id(ctx, "e0")
    _run_sql(ctx, "UPDATE serving_score_summary SET flags = ? WHERE release_id = ? AND score_id = ?",
             (INVALID, ctx.release_id, score_id))
    # /events with both toggles on really decodes flags on the requested
    # path: ``list_events`` builds each returned page item via
    # ``_event_page_item`` (projections.py:804) -> ``event_scores``
    # (projections.py:669) -> ``_score_summary_from_row``'s
    # ``json.loads(row["flags"])`` (projections.py:629). With both toggles
    # true ``_score_row_clauses`` adds no ``json_each(flags)`` SQL predicate
    # (projections.py:738-743), so the Python decode is what raises here.
    _demonstrate(ctx, json.JSONDecodeError, lambda conn: projections.list_events(
        conn, ctx.release_id, out_of_domain=True, disabled=True))
    _expect_unreadable(ctx, "/api/v1/events",
                       {"release_id": ctx.release_id, "out_of_domain": "true", "disabled": "true"})


def test_event_scores_malformed_score_flags(ctx):
    score_id = _score_id(ctx, "e0")
    _run_sql(ctx, "UPDATE serving_score_summary SET flags = ? WHERE release_id = ? AND score_id = ?",
             (INVALID, ctx.release_id, score_id))
    _demonstrate(ctx, json.JSONDecodeError, lambda conn: projections.event_scores(
        conn, ctx.release_id, "e0", out_of_domain=True, disabled=True))
    _expect_unreadable(ctx, "/api/v1/events/e0/scores", {"release_id": ctx.release_id})


# --------------------------------------------------------------------------
# 4. same damaged index: auth short-circuits before the index is opened
# --------------------------------------------------------------------------


def test_unauthorized_and_invalid_token_do_not_open_a_damaged_index(ctx):
    _run_sql(ctx, "DROP TABLE serving_release")
    for template, params in (
        ("/api/v1/releases/" + ctx.release_id, {}),
        ("/api/v1/releases/current", {}),
        ("/api/v1/events", {}),
    ):
        for token in (None, "wrong-token"):
            opened: list[sqlite3.Connection] = []
            with _track_connects() as opened:
                code, body, _ = _get(ctx.base, template, token=token, params=params)
            assert code == 401, (template, token, body[:200])
            assert json.loads(body)["code"] == "UNAUTHORIZED"
            assert opened == []  # the index is never opened on the auth path
            _assert_clean(body, ctx)


# --------------------------------------------------------------------------
# 5. a healthy ETag must not make a damaged read a 304 / immutable success
# --------------------------------------------------------------------------


def _capture_etag(ctx: _Ctx, path: str, params: dict | None = None) -> str:
    code, body, resp = _get(ctx.base, path, token=TOKEN, params=params)
    assert code == 200, (code, body[:200])
    etag = resp.get("ETag")
    assert etag
    return etag


def test_damaged_pinned_release_if_none_match_is_503_not_304(ctx):
    etag = _capture_etag(ctx, "/api/v1/releases/" + ctx.release_id)
    _run_sql(ctx, "DROP TABLE serving_release")
    _expect_unreadable(ctx, "/api/v1/releases/" + ctx.release_id, headers={"If-None-Match": etag})


def test_damaged_event_scores_if_none_match_is_503_not_304(ctx):
    etag = _capture_etag(ctx, "/api/v1/events/e0/scores", {"release_id": ctx.release_id})
    _run_sql(ctx, "DROP TABLE serving_event_summary")
    _expect_unreadable(ctx, "/api/v1/events/e0/scores", {"release_id": ctx.release_id},
                       headers={"If-None-Match": etag})


def test_damaged_score_detail_if_none_match_is_503_not_304(ctx):
    score_id = _score_id(ctx, "e0")
    etag = _capture_etag(ctx, "/api/v1/scores/" + score_id, {"release_id": ctx.release_id})
    _run_sql(ctx, "DROP TABLE serving_score_summary")
    _expect_unreadable(ctx, "/api/v1/scores/" + score_id, {"release_id": ctx.release_id},
                       headers={"If-None-Match": etag})


# --------------------------------------------------------------------------
# 6. a malformed ARTIFACT (existing read_verified seam, index bytes untouched)
#    keeps the existing 500 -- never SERVING_INDEX_UNREADABLE -- and still
#    closes every index connection.
# --------------------------------------------------------------------------


def test_malformed_artifact_detail_stays_500_not_index_unreadable(ctx, monkeypatch):
    score_id = _score_id(ctx, "e0")
    calls: list = []

    def fake_read_verified(self, ref):
        calls.append(ref)
        return b"{malformed-artifact-not-index"

    monkeypatch.setattr(ArtifactStore, "read_verified", fake_read_verified)
    opened: list[sqlite3.Connection] = []
    with _track_connects() as opened:
        code, body, _ = _get(ctx.base, "/api/v1/scores/" + score_id, token=TOKEN,
                            params={"release_id": ctx.release_id})
    assert code == 500, (code, body[:200])
    assert len(calls) == 1
    assert b"SERVING_INDEX_UNREADABLE" not in body
    _assert_closed(opened)


# --------------------------------------------------------------------------
# bounded control: this really is the live serving fixture, and a missing
# release is still the 404 UNKNOWN_RELEASE -- never a 503.
# --------------------------------------------------------------------------


def test_healthy_and_missing_release_control(ctx):
    code, _, _ = _get(ctx.base, "/api/v1/releases/" + ctx.release_id, token=TOKEN)
    assert code == 200
    code, body, _ = _get(ctx.base, "/api/v1/releases/does-not-exist", token=TOKEN)
    assert code == 404
    assert json.loads(body)["code"] == "UNKNOWN_RELEASE"


@pytest.fixture
def ctx(tmp_path):
    with _live_app(tmp_path) as live_ctx:
        yield live_ctx
