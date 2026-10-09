"""The read-only v2 dashboard API -- rearchitecture phase-3 guide §6 (P3-2).

``create_app`` wires FastAPI routes over the bounded read helpers
:mod:`engine.v2.serving.projections` already exposes (§5.4/P3-1b): no route
and no import at module load time constructs a scorer, opens a provider
connection, or touches ``engine.v2.ops``/legacy ``engine.*`` -- every value
served here was already computed and indexed by the offline projection
coordinator (``tools/v2_dashboard_project.py``).

**Current-release rule (§5.4/P3-1c).** One authoritative published pointer:
the existing fenced ops publisher's own ``CURRENT`` under its release root
(``engine/v2/ops/publication.py``). API "current" follows it one more hop
-- that release's bound ``projection_binding.json`` names the projection
``release_id`` -- never a second, independently-advancing "UI latest"
pointer. The default resolver (``_publication_resolver``) reads both files
directly (the same symlink/one-segment safety ``operations.py``'s
``_resolve_current_id`` applies, reimplemented since serving may not
import ops), then reverifies the named release against the LIVE
``serving.sqlite`` index (``projections.verify_projection_binding``): a
binding that does not (yet, or any longer) match is a typed, non-retryable
 ``CURRENT_BINDING_INVALID``, never a silent "latest" fallback and never the
 plain "no current release configured" (503) a release with no bound
 projection at all (the P3-0 compatibility preview) still gets. A post-open
 index read failing during that reverification is the retryable
 ``SERVING_INDEX_UNREADABLE`` 503 (``_index_unreadable``, #342).


``create_app``'s ``resolver`` parameter remains the seam: any zero-argument
``Callable[[], str | None]`` (may also raise ``ApiError``) may replace it,
as tests do. Without an explicit ``resolver``, ``publication_root`` selects
``_publication_resolver``; with neither, "current" always resolves to
``None`` -- the temporary ``serving_root/CURRENT`` default this module used
before P3-1c is retired, not replaced by a new fallback.

**Errors are one ``Problem``-shaped envelope everywhere** (§6): ``code``,
``category``, ``retryable``, ``message``, plus the operational fields the
contract carries (``stage``, ``trace_id``, ``dependency_refs``,
``retry_after_seconds``, ``diagnostic_ref``, ``details``,
``schema_version``). Built as a plain dict (``_problem``), not the
``engine.v2.contracts.Problem`` dataclass -- constructing real instances
would need a contracts import this module has no other use for, and the
fan-out budget (§4.3, 8 distinct modules) is otherwise exactly spent on
``fastapi``, ``engine.v2.foundation``, this package's own ``projections``,
plus ``hmac``/``os``/``json``/``argparse``/``uvicorn`` for auth, cursor
signing, the CURRENT/health files and the launcher. The shape is pinned by
 ``tests/test_v2_serving_api.py`` against the real dataclass's own field
 names, so the two cannot drift silently.

**A serving index that opens but cannot be read is one typed 503** (#342):
every connection-owning read block -- the publication resolver's binding
reverification, ``/releases/{id}`` + ``/releases/current``, ``/events``,
``/events/{id}/scores``, ``/scores/{id}`` -- wraps only its projection calls in
``except (projections.sqlite3.Error, json.JSONDecodeError)`` and raises
``_index_unreadable()``, keeping the existing guaranteed-close ``finally``.
Open-time behavior, artifact decoding (still the existing 500), binding-invalid
(500), unknown release/event/score (404) and every validation refusal are
unchanged; no global SQLite/JSON exception hook is registered.


**Cursors** are opaque and integrity-protected (§6): ``release_id`` plus
every normalized filter (``projections.event_query_hash``, over the wire
names ``date_from``/``date_to``/``ticker``/``strategy``/``verdict`` ``ui/
src/api/client.ts`` sends) plus the projection's own raw keyset cursor are
HMAC-signed with a key derived from the server's auth token (never the
token itself -- ``_cursor_key`` domain-separates it). A cursor whose
signature fails, or whose embedded release/query does not match the
current request, is refused as ``CURSOR_MISMATCH`` (409, matching
component_contracts.md §13.2 and ``tests/fixtures/v2_ui_mock_api.py`` --
see the package README's judgement-call note for why this superseded the
400 an earlier version of this module used).

**ETags** are strong: hashing ``{identity, document}`` (``_etag``).
Release-scoped immutable objects (a specific release, an event's scores,
one score detail) carry a long, private ``Cache-Control``. ``current`` and
``operations`` stay ``Cache-Control: no-store`` since their *resolution*
can change between requests -- but ``current`` also now carries an ETag and
answers ``If-None-Match`` with 304, so the UI's ~4s release-change poll
(``ui/src/hooks.ts`` ``usePinnedRelease``) stays cheap without ever letting
a cache reuse a stale resolution.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os

import uvicorn
from fastapi import Depends, FastAPI, Request, Response

from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    canonical_json,
    content_hash,
    safe_relative_path,
    to_document,
)

from . import native_parity_projection, projections

__all__ = ["ApiError", "create_app", "main"]

_LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
_OPERATIONS_STATUS_SCHEMA = "operations_status.v1.0"
_IMMUTABLE_CACHE = "private, max-age=31536000, immutable"

#: §Native parity: an ops root with no committed succeeded shadow parity job.
_NATIVE_PARITY_JOB_NOT_FOUND = "NATIVE_PARITY_JOB_NOT_FOUND"
#: Report-source result distinct from "no configured report" (``None``).
_JOB_NOT_FOUND = object()


class ApiError(Exception):
    """One route or dependency refusal, carrying its own ``Problem`` body."""

    def __init__(self, status_code: int, problem: dict) -> None:
        super().__init__(problem["message"])
        self.status_code = status_code
        self.problem = problem


def _problem(code: str, category: str, message: str, *, retryable: bool = False,
            details: dict | None = None) -> dict:
    return {"code": code, "category": category, "retryable": retryable, "message": message,
            "stage": None, "trace_id": None, "dependency_refs": [], "retry_after_seconds": None,
            "diagnostic_ref": None, "details": details or {}, "schema_version": "problem.v1.0"}


def _handle_api_error(request: Request, exc: ApiError) -> Response:
    """Render one Problem envelope; native parity responses are never cached or reused."""
    headers = ({"Cache-Control": "no-store"}
               if request.url.path.startswith("/api/v1/native_parity") else {})
    return Response(content=canonical_json(exc.problem), media_type="application/json",
                    status_code=exc.status_code, headers=headers)


# --------------------------------------------------------------------------
# post-open serving-index read failures (#342) -- the narrow boundary each
# connection-owning read applies, same shape as the operations listener's #339
# listener fix: a ``try`` around only the projection calls, the existing
# ``conn.close()`` ``finally`` kept, and this one typed refusal. No global
# SQLite/JSON hooks, no open-time behavior change (an index that fails to
# OPEN stays exactly what ``projections.connect`` already did), and artifact
# decoding failures stay their existing 500 -- only INDEX reads/rows translate.
# --------------------------------------------------------------------------


def _index_unreadable() -> ApiError:
    """A connection ``projections.connect`` opened successfully, but a later
    API-owned index read failed: ``sqlite3.Error`` (a dropped/corrupt table the
    migrations still tolerate) or ``json.JSONDecodeError`` (malformed stored
    ``document_json``/``findings_json``/``ref_json``/``flags``). Retryable 503
    with the existing ``Problem`` shape and this FIXED message -- never raw
    exception text, a path, the token, a traceback or the malformed bytes.
    ``projections.sqlite3`` is reached through the sibling module's own import,
    the way ``operations.py`` does it, because this module's fan-out budget
    (§4.3, 8 distinct modules) is otherwise exactly spent."""
    return ApiError(503, _problem("SERVING_INDEX_UNREADABLE", "resource",
                                  "the serving index cannot be read", retryable=True))


# --------------------------------------------------------------------------
# auth -- reuses operations.py's bearer/cookie rule, not its code (a peer
# module this task does not touch)
# --------------------------------------------------------------------------


def _authorized(request: Request, token: str) -> bool:
    if not token:
        return False
    header = request.headers.get("authorization", "")
    if hmac.compare_digest(header, "Bearer " + token):
        return True
    cookie = request.cookies.get("operations_token")
    return cookie is not None and hmac.compare_digest(cookie, token)


# --------------------------------------------------------------------------
# the default CURRENT resolver -- §5.4/P3-1c: published ops release ->
# its bound projection_binding.json -> the projection release_id, reverified
# against the live serving index
# --------------------------------------------------------------------------


def _one_segment_pointer(raw: str) -> str | None:
    """A pointer file's content, accepted only as one clean path segment --
    the same safety rule ``engine/v2/ops/publication.py::current`` and
    ``engine/v2/serving/operations.py::_resolve_current_id`` both apply,
    reimplemented here (serving may not import ops; operations.py is an
    untouched peer)."""
    try:
        parts = safe_relative_path(raw)
    except ArtifactError:
        return None
    return raw if len(parts) == 1 else None


def _read_pointer_file(path: str) -> str | None:
    if os.path.islink(path) or not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return _one_segment_pointer(handle.read().strip())


def _read_ops_current(release_root) -> str | None:
    """File-only re-read of the fenced ops publisher's own ``CURRENT`` --
    never an ``engine.v2.ops`` import."""
    return _read_pointer_file(os.path.join(str(release_root), "CURRENT"))


def _read_projection_binding(release_root, ops_release_id: str) -> dict | None:
    """Read ``<release_root>/releases/<id>/projection_binding.json`` -- the
    file ``publication_effect`` binds when the operator supplies one (§5.4/
    P3-1c). ``None`` for a missing/symlinked/malformed file or unsafe id --
    an unbound release is normal, not a corruption."""
    if _one_segment_pointer(ops_release_id) != ops_release_id:
        return None
    path = os.path.join(str(release_root), "releases", ops_release_id, "projection_binding.json")
    if os.path.islink(path) or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.loads(handle.read())
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _publication_resolver(release_root, serving_db):
    """The one pointer chain (§5.4/P3-1c): ops ``CURRENT`` -> that
    release's bound ``projection_binding.json`` -> the projection
    ``release_id`` it names, reverified against the LIVE serving index.
    Files (plus one bounded ``serving.sqlite`` connection) only -- keeps
    serving/ops layering intact. ``None`` when there is genuinely no
    publication yet (absent/malformed pointer or binding -- ordinary "no
    current release" 503). Raises :class:`ApiError` -- typed, non-
    retryable, distinct from "no current release" -- only when a binding
 names a release id that fails to verify: not committed, or its
 manifest/index hash no longer matches (a tampered doc or changed index row).
 Post-open failures of that reverification itself -- a dropped/corrupt table
 (``sqlite3.Error``) or a malformed stored row (``json.JSONDecodeError``) --
 are the retryable ``SERVING_INDEX_UNREADABLE`` 503 (#342); the connection
 closes either way."""


    def resolve() -> str | None:
        ops_release_id = _read_ops_current(release_root)
        if ops_release_id is None:
            return None
        binding = _read_projection_binding(release_root, ops_release_id)
        if binding is None:
            return None
        conn = projections.connect(str(serving_db))
        try:
            if not projections.verify_projection_binding(conn, binding):
                raise ApiError(500, _problem(
                    "CURRENT_BINDING_INVALID", "integrity",
                    "the published pointer's projection binding does not match the "
                    "live serving index", details={"ops_release_id": ops_release_id}))
            release_id = binding.get("projection_release_id")
        except (projections.sqlite3.Error, json.JSONDecodeError):
            raise _index_unreadable() from None
        finally:
            conn.close()
        return release_id if isinstance(release_id, str) else None

    return resolve


def _no_publication_configured() -> str | None:
    return None


# --------------------------------------------------------------------------
# ETags and cache headers
# --------------------------------------------------------------------------


def _etag(identity: str, document) -> str:
    return '"' + content_hash({"identity": identity, "document": document}) + '"'


def _not_modified(request: Request, etag: str) -> bool:
    header = request.headers.get("if-none-match")
    if not header:
        return False
    return etag in {part.strip() for part in header.split(",")}


def _cached_response(request: Request, response: Response, identity: str, document, *,
                     cache_control: str) -> dict | Response:
    etag = _etag(identity, document)
    if _not_modified(request, etag):
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": cache_control})
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = cache_control
    return document


def _immutable_response(request: Request, response: Response, identity: str, document) -> dict | Response:
    return _cached_response(request, response, identity, document, cache_control=_IMMUTABLE_CACHE)


# --------------------------------------------------------------------------
# opaque, integrity-protected cursors -- §6
# --------------------------------------------------------------------------


def _cursor_key(token: str) -> bytes:
    return hmac.new(token.encode("utf-8"), b"v2-serving-cursor-v1", digestmod="sha256").digest()


def _pack_cursor(key: bytes, release_id: str, query_hash: str, raw: str) -> str:
    payload = "|".join((release_id, query_hash, raw)).encode("utf-8")
    signature = hmac.new(key, payload, digestmod="sha256").hexdigest()
    return payload.hex() + "." + signature


#: §6/component_contracts.md §13.2: "A cursor from another release/query
#: returns CURSOR_MISMATCH" at 409 Conflict. `tests/fixtures/v2_ui_mock_api.py`
#: (built against §6 for P3-3a) agrees: `HTTPStatus.CONFLICT`. An earlier
#: version of this module used 400 per this task's own brief text; 409 is the
#: one both the guide and the UI's own mock server actually implement, so it
#: wins (see the package README's judgement-call note).
_CURSOR_MISMATCH_STATUS = 409


def _unpack_cursor(key: bytes, cursor: str) -> tuple[str, str, str]:
    mismatch = ApiError(_CURSOR_MISMATCH_STATUS,
                        _problem("CURSOR_MISMATCH", "validation", "cursor is malformed or tampered"))
    try:
        hex_part, signature = cursor.rsplit(".", 1)
        payload = bytes.fromhex(hex_part)
    except ValueError:
        raise mismatch from None
    expected = hmac.new(key, payload, digestmod="sha256").hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise mismatch
    try:
        release_id, query_hash, raw = payload.decode("utf-8").split("|", 2)
    except ValueError:
        raise mismatch from None
    return release_id, query_hash, raw


def _validated_raw_cursor(cursor: str | None, cursor_key: bytes, release_id: str, expected_hash: str) -> str | None:
    if cursor is None:
        return None
    c_release, c_hash, raw = _unpack_cursor(cursor_key, cursor)
    if c_release != release_id or c_hash != expected_hash:
        raise ApiError(_CURSOR_MISMATCH_STATUS, _problem("CURSOR_MISMATCH", "validation",
                                                          "cursor was issued for a different release or query"))
    return raw


def _event_page_document(page, cursor_key: bytes) -> dict:
    document = to_document(page)
    if page.next_cursor is not None:
        document["next_cursor"] = _pack_cursor(cursor_key, page.release_id, page.query_hash, page.next_cursor)
    return document


# --------------------------------------------------------------------------
# request validation
# --------------------------------------------------------------------------


def _parse_limit(raw: str | None) -> int:
    if raw is None:
        return projections.DEFAULT_PAGE_SIZE
    try:
        value = int(raw)
    except ValueError:
        raise ApiError(422, _problem("INVALID_REQUEST", "validation", "limit must be an integer")) from None
    if value < 1:
        raise ApiError(422, _problem("INVALID_REQUEST", "validation", "limit must be positive"))
    return value


def _validate_date(value: str | None, field_name: str) -> None:
    if value is None:
        return
    ok = (len(value) == 10 and value[4] == "-" and value[7] == "-"
         and value[:4].isdigit() and value[5:7].isdigit() and value[8:10].isdigit())
    if not ok:
        raise ApiError(422, _problem("INVALID_REQUEST", "validation", f"{field_name} must be YYYY-MM-DD"))


_GATE_VALUES = ("pass", "fail", "na")


def _validate_gate(value: str | None) -> None:
    if value is not None and value not in _GATE_VALUES:
        raise ApiError(422, _problem("INVALID_REQUEST", "validation",
                                     "gate must be one of pass, fail, na"))


def _parse_bool_flag(raw: str | None, field_name: str) -> bool:
    if raw is None:
        return False
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ApiError(422, _problem("INVALID_REQUEST", "validation",
                                 f"{field_name} must be true or false"))


# --------------------------------------------------------------------------
# operations status -- §6/§5.5 items 2-3: the sidecar ``publication_effect``
# writes into the publisher's scope root, read like ``_read_projection_
# binding`` (files only, never an ``engine.v2.ops`` import), as a plain
# untyped document (no ``engine.v2.contracts`` import; budget already spent).
# --------------------------------------------------------------------------


def _read_operations_status(publication_root) -> dict | None:
    if not publication_root:
        return None
    path = os.path.join(str(publication_root), "operations_status.json")
    if os.path.islink(path) or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.loads(handle.read())
    except (OSError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("schema_version") != _OPERATIONS_STATUS_SCHEMA:
        return None
    return document


# --------------------------------------------------------------------------
# route bodies -- one function per route, kept independent of FastAPI's own
# decorator wiring so each stays under the function-length/complexity budget
# --------------------------------------------------------------------------


def _open(serving_db):
    return projections.connect(str(serving_db))


def _release_response(serving_db, release_id: str | None, response: Response, request: Request, *,
                      require_current: bool) -> dict | Response:
    if release_id is None:
        raise ApiError(503, _problem("NO_CURRENT_RELEASE", "resource",
                                     "no current release is configured", retryable=True))
    conn = _open(serving_db)
    try:
        release = projections.get_release(conn, release_id)
    except (projections.sqlite3.Error, json.JSONDecodeError):
        raise _index_unreadable() from None
    finally:
        conn.close()
    if release is None:
        if require_current:
            raise ApiError(503, _problem("NO_CURRENT_RELEASE", "resource",
                                         "the current pointer names a release with no committed "
                                         "projection", retryable=True))
        raise ApiError(404, _problem("UNKNOWN_RELEASE", "validation", "unknown release id"))
    document = to_document(release)
    cache_control = "no-store" if require_current else _IMMUTABLE_CACHE
    return _cached_response(request, response, release.release_id, document, cache_control=cache_control)


def _require_release(conn, release_id: str) -> None:
    if projections.get_release(conn, release_id) is None:
        raise ApiError(404, _problem("UNKNOWN_RELEASE", "validation", "unknown release id"))


def _events_response(serving_db, resolve_current, cursor_key: bytes, response: Response, request: Request, *,
                     release_id: str | None, event_date_from: str | None, event_date_to: str | None,
                     ticker: str | None, strategy: str | None, verdict: str | None,
                     gate: str | None, out_of_domain: str | None, disabled: str | None,
                     limit: str | None, cursor: str | None) -> dict:
    explicit = release_id is not None
    _validate_date(event_date_from, "date_from")
    _validate_date(event_date_to, "date_to")
    _validate_gate(gate)
    out_of_domain_value = _parse_bool_flag(out_of_domain, "out_of_domain")
    disabled_value = _parse_bool_flag(disabled, "disabled")
    limit_value = _parse_limit(limit)
    conn = _open(serving_db)
    try:
        if not explicit:
            release_id = resolve_current()
            if release_id is None:
                raise ApiError(503, _problem("NO_CURRENT_RELEASE", "resource",
                                             "no current release is configured", retryable=True))
        _require_release(conn, release_id)
        expected_hash = projections.event_query_hash(
            release_id, event_date_from=event_date_from, event_date_to=event_date_to,
            ticker=ticker, strategy=strategy, verdict=verdict, gate=gate,
            out_of_domain=out_of_domain_value, disabled=disabled_value)
        raw_cursor = _validated_raw_cursor(cursor, cursor_key, release_id, expected_hash)
        page = projections.list_events(
            conn, release_id, limit=limit_value, cursor=raw_cursor,
            event_date_from=event_date_from, event_date_to=event_date_to,
            ticker=ticker, strategy=strategy, verdict=verdict, gate=gate,
            out_of_domain=out_of_domain_value, disabled=disabled_value)
    except (projections.sqlite3.Error, json.JSONDecodeError):
        raise _index_unreadable() from None
    finally:
        conn.close()
    document = _event_page_document(page, cursor_key)
    if explicit:
        response.headers["ETag"] = _etag(release_id, document)
        response.headers["Cache-Control"] = _IMMUTABLE_CACHE
    else:
        response.headers["Cache-Control"] = "no-store"
    return document


def _require_release_id(release_id: str | None) -> str:
    """§5.4: "event/score IDs are unique within a release" -- not globally,
    so a route keyed by one must be given the release, never search for it.
    Missing is a client error distinct from an unknown one (P3-2 review):
    400 `RELEASE_ID_REQUIRED`, not the 404 an unknown-but-present id gets."""
    if release_id is None:
        raise ApiError(400, _problem("RELEASE_ID_REQUIRED", "validation",
                                     "release_id is required"))
    return release_id


def _event_scores_response(serving_db, response: Response, request: Request, *,
                           event_id: str, release_id: str | None, clock_id: str | None) -> list | Response:
    """§6: bare array of `EventScoreSummary` -- matches `ui/src/api/client.ts`
    `getEventScores(): Promise<EventScoreSummary[]>` and `tests/fixtures/
    v2_ui_mock_api.py`'s `_event_scores`; §6 names no different envelope, so
    there is no reason to wrap it."""
    release_id = _require_release_id(release_id)
    conn = _open(serving_db)
    try:
        _require_release(conn, release_id)
        item = projections.get_event(conn, release_id, event_id)
        if item is None:
            raise ApiError(404, _problem("UNKNOWN_EVENT", "validation", "unknown event id"))
        if clock_id is not None and clock_id != item.clock_id:
            raise ApiError(422, _problem("INVALID_REQUEST", "validation",
                                         "clock_id does not match this event"))
    except (projections.sqlite3.Error, json.JSONDecodeError):
        raise _index_unreadable() from None
    finally:
        conn.close()
    document = [to_document(s) for s in item.scores]
    return _immutable_response(request, response, release_id + "|" + event_id, document)


def _score_detail_response(serving_db, store, response: Response, request: Request, *,
                           score_id: str, release_id: str | None) -> dict | Response:
    """§6: "validate membership if a release ID is supplied" presumes one
    normally is; §5.4 scopes score-id uniqueness to within one release, so
    this never searches across releases (P3-2 review) -- `release_id` is
    required, matching `/events/{id}/scores`."""
    release_id = _require_release_id(release_id)
    conn = _open(serving_db)
    try:
        _require_release(conn, release_id)
        ref = projections._score_detail_ref(conn, release_id, score_id)
    except (projections.sqlite3.Error, json.JSONDecodeError):
        raise _index_unreadable() from None
    finally:
        conn.close()
    if ref is None:
        raise ApiError(404, _problem("UNKNOWN_SCORE", "validation", "unknown score id"))
    # Deliberately outside the catch and after the guaranteed close: this reads
    # the ARTIFACT store, not the index. A malformed artifact stays the existing
    # unhandled 500 and is never translated into SERVING_INDEX_UNREADABLE.
    detail = projections._score_detail_from_ref(store, ref)
    document = to_document(detail)
    return _immutable_response(request, response, detail.score_id, document)


def _operations_response(publication_root, response: Response, *, release_id: str | None = None) -> dict:
    """§6 sidecar; missing history is the same typed refusal as a malformed
    document -- never a silent green default (§5.5 item 3).

    2026-09-14 review fix: the sidecar always describes the SCOPE's current
    state, not any one release -- a client that pinned an earlier
    ``release_id`` (from a prior response's own field, or from
    ``/api/v1/releases/current``) and polls this route later can otherwise
    be handed a DIFFERENT release's status without noticing, once a newer
    publication attempt has run. ``release_id``, given, is checked against
    the document's own field; a mismatch is a typed, non-retryable refusal
    (the caller asked for a pinned release this document no longer
    describes) rather than silently substituting another release's status.
    """
    document = _read_operations_status(publication_root)
    if document is None:
        raise ApiError(503, _problem("OPERATIONS_UNAVAILABLE", "resource",
                                     "no operations status recorded", retryable=True))
    if release_id is not None and document.get("release_id") != release_id:
        raise ApiError(409, _problem(
            "OPERATIONS_STATUS_NOT_FOR_RELEASE", "validation",
            "operations status describes a different release than requested",
            details={"requested_release_id": release_id,
                     "status_release_id": document.get("release_id")}))
    response.headers["Cache-Control"] = "no-store"
    return document


def _parity_malformed() -> ApiError:
    return ApiError(503, _problem(native_parity_projection.NATIVE_PARITY_REPORT_MALFORMED, "integrity",
                                  "the native parity report is malformed"))


# ops-root native-parity discovery: strictly read-only and file-only about the
# ops catalog (serving may not import ``engine.v2.ops``); the selected report is
# read through the same verified foundation store reader.


def _no_report_source():
    """No configured report source: the unchanged ``no_report`` result."""
    return None


def _fixed_report_source(report_path):
    """The explicit ``--native-parity-report-path`` always wins: only it."""
    return lambda: report_path


def _native_parity_report_source(native_parity_report_path, ops_root):
    """The one precedence rule: an explicit report path, else ops discovery."""
    if native_parity_report_path is not None:
        return _fixed_report_source(native_parity_report_path)
    if ops_root is not None:
        return _ops_native_parity_report_source(ops_root)
    return _no_report_source


def _job_not_found_report(section: str | None) -> dict:
    """The typed ``no_report`` body for an ops root with no succeeded job."""
    reason = {"reason_code": _NATIVE_PARITY_JOB_NOT_FOUND}
    if section is None:
        return {"schema_version": native_parity_projection.NATIVE_PARITY_SUMMARY_V1,
                "status": "no_report", **reason}
    return {"status": "no_report", **reason}


def _native_parity_as_of(idempotency_key: str) -> str | None:
    """Recover the as-of from a ``nightly:<as_of>:<scope_hash>:native_parity`` key;
    mirrors ``engine.v2.ops.nightly``'s parser (``partition``/``rpartition``)."""
    prefix, sep1, rest = idempotency_key.partition(":")
    if prefix != "nightly" or not sep1:
        return None
    as_of, sep2, remainder = rest.partition(":")
    if not sep2 or not as_of:
        return None
    _scope_hash, sep3, stage = remainder.rpartition(":")
    if not sep3 or stage != "native_parity":
        return None
    return as_of


#: RFC 3986 unreserved characters plus ``/``, kept literal in a SQLite URI path.
_URI_PATH_SAFE = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~/")


def _open_ops_catalog(catalog_path: str):
    """A strictly read-only ``mode=ro`` connection that never migrates or writes.

    ``catalog_path`` is percent-encoded as the URI path component, so a ``?``
    or ``#`` in an ops-root directory name stays a filename character instead
    of starting the query string or a fragment (SQLite decodes it back).
    Encoding is byte-level via ``os.fsencode``: ordinary Unicode becomes its
    UTF-8 bytes and surrogateescape filename bytes are preserved, so any
    directory name the filesystem itself accepts round-trips."""
    uri_path = "".join(
        chr(byte) if chr(byte) in _URI_PATH_SAFE
        else f"%{byte:02X}" for byte in os.fsencode(catalog_path))
    conn = projections.sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    conn.row_factory = projections.sqlite3.Row
    return conn


#: Jobs only -- a succeeded job with no committed output is still considered.
_NEWEST_NATIVE_PARITY_JOB = (
    "SELECT j.job_id, j.idempotency_key, "
    "(SELECT MAX(a.ended_at) FROM attempts a "
    " WHERE a.job_id = j.job_id AND a.state = 'succeeded') AS commit_at "
    "FROM jobs j WHERE j.kind = 'native_parity' AND j.namespace = 'shadow' "
    "AND j.state = 'succeeded'")

#: The producer registers its report output under the name ``report``.
_NATIVE_PARITY_REPORT_OUTPUT = (
    "SELECT ao.artifact_id FROM attempts a "
    "JOIN attempt_outputs ao ON ao.attempt_id = a.attempt_id "
    "WHERE a.job_id = ? AND a.state = 'succeeded' AND ao.name = 'report' "
    "ORDER BY a.attempt_number DESC LIMIT 1")


def _newest_native_parity_job(conn) -> str | None:
    """The newest succeeded shadow parity job id, or ``None`` (ordered by
    as-of, then output commit time, then job id)."""
    candidates = []
    for row in conn.execute(_NEWEST_NATIVE_PARITY_JOB):
        as_of = _native_parity_as_of(row["idempotency_key"])
        if as_of is None:
            continue
        candidates.append((as_of, row["commit_at"] or "", row["job_id"]))
    if not candidates:
        return None
    return max(candidates)[2]


def _selected_native_parity_ref(conn, job_id: str):
    """The registered ``report`` output ref for a succeeded attempt, or ``None``."""
    row = conn.execute(_NATIVE_PARITY_REPORT_OUTPUT, (job_id,)).fetchone()
    if row is None:
        return None
    artifact = conn.execute("SELECT ref_json FROM artifacts WHERE artifact_id = ?",
                            (row["artifact_id"],)).fetchone()
    if artifact is None:
        return None
    try:
        return projections.from_document(projections.ArtifactRef,
                                         json.loads(artifact["ref_json"]))
    except (ValueError, TypeError):
        # A malformed stored reference is this job's own broken output:
        # the caller's typed malformed refusal, never an older fallback.
        raise _parity_malformed() from None


def _ops_native_parity_report_source(ops_root):
    """Discover the newest succeeded shadow ``native_parity`` report read-only.

    The returned callable yields the verified report path, ``_JOB_NOT_FOUND``
    when none exists, or raises the typed 503 refusal; its connection always closes."""
    root = os.path.realpath(str(ops_root))
    catalog_path = os.path.join(root, "catalog.sqlite")
    store = ArtifactStore(root)

    def source():
        if not os.path.isfile(catalog_path):
            return _JOB_NOT_FOUND
        try:
            conn = _open_ops_catalog(catalog_path)
        except projections.sqlite3.Error:
            raise _parity_malformed() from None
        try:
            job_id = _newest_native_parity_job(conn)
            ref = None if job_id is None else _selected_native_parity_ref(conn, job_id)
        except projections.sqlite3.Error:
            raise _parity_malformed() from None
        finally:
            conn.close()
        if job_id is None:
            return _JOB_NOT_FOUND
        if ref is None:
            raise _parity_malformed()
        try:
            return str(store.verify(ref))
        except (ArtifactError, OSError):
            raise _parity_malformed() from None

    return source


def _native_parity_freshness(serving_db, resolve_current, as_of: str | None) -> str:
    """Classify report freshness; an ApiError/OSError resolver means unknown current."""
    return native_parity_projection.native_parity_freshness(
        serving_db, resolve_current, as_of, _open, projections.get_release,
        (ApiError, OSError))


def _native_parity_encoded_response(document) -> Response:
    """Serialize validated JSON directly, avoiding FastAPI's recursive re-encoding."""
    try:
        body = json.dumps(document, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError):
        raise _parity_malformed() from None
    return Response(content=body, media_type="application/json",
                    headers={"Cache-Control": "no-store"})


def _native_parity_screen_items(report, section: str, side: str | None, row_key: str | None):
    """Return one screen's native parity items, mapping lookup/malformed failures to ApiError."""
    try:
        return native_parity_projection.native_parity_items(
            report, section, side=side, row_key=row_key)
    except LookupError:
        raise ApiError(404, _problem("NATIVE_PARITY_ROW_NOT_FOUND", "validation",
                                     "unknown native parity row key")) from None
    except (ValueError, TypeError, RecursionError):
        raise _parity_malformed() from None


def _native_parity_response(serving_db, resolve_current, cursor_key: bytes, report_source,
                            response: Response, *, section: str | None = None,
                            side: str | None = None, row_key: str | None = None,
                            limit: str | None = None, cursor: str | None = None) -> dict:
    """Load, validate, then serve either the summary or one filtered detail screen."""
    report_path = report_source()
    response.headers["Cache-Control"] = "no-store"
    if report_path is _JOB_NOT_FOUND:
        return _job_not_found_report(section)
    _status, summary, report = native_parity_projection.native_parity_snapshot(report_path)
    if summary["status"] == "unavailable":
        raise _parity_malformed()
    if report is None:
        return summary if section is None else {"status": "no_report"}
    if (report["schema_version"] == "native_parity_report.v1.2"
            and not {"as_of", "generated_at"} <= report.keys()):
        raise _parity_malformed()
    summary["status"] = _native_parity_freshness(serving_db, resolve_current, summary.get("as_of"))
    if section is None:
        return summary
    if section == "unpaired" and side not in ("legacy", "native"):
        raise ApiError(422, _problem("INVALID_REQUEST", "validation",
                                     "side must be legacy or native"))
    limit_value = min(_parse_limit(limit), projections.MAX_PAGE_SIZE)
    try:
        release_id = content_hash(report)
    except (ValueError, TypeError, RecursionError):
        raise _parity_malformed() from None
    query_hash = content_hash({"section": section, "side": side, "row_key": row_key})
    raw_cursor = _validated_raw_cursor(cursor, cursor_key, release_id, query_hash)
    items = _native_parity_screen_items(report, section, side, row_key)
    try:
        offset = 0 if raw_cursor is None else int(raw_cursor)
        if not 0 <= offset <= len(items):
            raise ValueError(raw_cursor)
    except ValueError:
        raise ApiError(_CURSOR_MISMATCH_STATUS, _problem(
            "CURSOR_MISMATCH", "validation", "cursor offset is invalid")) from None
    end = offset + limit_value
    next_cursor = (_pack_cursor(cursor_key, release_id, query_hash, str(end))
                   if end < len(items) else None)
    return {"status": summary["status"], "as_of": summary.get("as_of"),
            "generated_at": summary.get("generated_at"),
            "tolerance_policy_id": summary.get("tolerance_policy_id"),
            "items": items[offset:end], "next_cursor": next_cursor}


def _register_native_parity_routes(app, auth, serving_db, resolve_current, cursor_key,
                                   report_source) -> None:
    @app.get("/api/v1/native_parity", dependencies=auth)
    def native_parity_summary_route(response: Response):
        return _native_parity_encoded_response(
            _native_parity_response(serving_db, resolve_current, cursor_key, report_source, response))

    @app.get("/api/v1/native_parity/mismatches", dependencies=auth)
    def native_parity_mismatches_route(response: Response, row_key: str | None = None,
                                       limit: str | None = None, cursor: str | None = None):
        return _native_parity_encoded_response(
            _native_parity_response(serving_db, resolve_current, cursor_key, report_source, response,
                                    section="mismatches", row_key=row_key, limit=limit, cursor=cursor))

    @app.get("/api/v1/native_parity/unpaired", dependencies=auth)
    def native_parity_unpaired_route(response: Response, side: str | None = None,
                                     row_key: str | None = None, limit: str | None = None,
                                     cursor: str | None = None):
        return _native_parity_encoded_response(
            _native_parity_response(serving_db, resolve_current, cursor_key, report_source, response,
                                    section="unpaired", side=side, row_key=row_key,
                                    limit=limit, cursor=cursor))


# --------------------------------------------------------------------------
# app wiring
# --------------------------------------------------------------------------


def create_app(*, serving_db, store_root, serving_root, token: str, resolver=None,
              publication_root=None, ops_root=None, native_parity_report_path=None) -> FastAPI:
    """Build the read-only API. ``resolver``, given, replaces the default
    current-release resolution with any zero-argument ``Callable[[], str |
    None]`` (may also raise ``ApiError``) -- tests use this to pin a
    release without touching the filesystem. Without one, ``publication_
    root`` -- the fenced ops publisher's own release root for this shadow
    scope (``<ops_root>/releases/<scope>``) -- selects the real chain
    (§5.4/P3-1c): its ``CURRENT``, that release's bound ``projection_
    binding.json``, reverified against ``serving_db``. With neither,
    "current" always resolves to ``None``.

    ``ops_root`` is an independent, never-inferred path: when given and no
    explicit ``native_parity_report_path`` is supplied, the native-parity
    routes discover the newest committed succeeded shadow report read-only
    from ``<ops_root>/catalog.sqlite``. An explicit report path always wins."""
    if not token:
        raise ValueError("a nonempty token is required")
    store = ArtifactStore(store_root)
    if resolver is not None:
        resolve_current = resolver
    elif publication_root is not None:
        resolve_current = _publication_resolver(publication_root, serving_db)
    else:
        resolve_current = _no_publication_configured
    report_source = _native_parity_report_source(native_parity_report_path, ops_root)
    cursor_key = _cursor_key(token)
    app = FastAPI(title="v2 serving read API", docs_url=None, redoc_url=None, openapi_url=None)

    def require_auth(request: Request) -> None:
        if not _authorized(request, token):
            raise ApiError(401, _problem("UNAUTHORIZED", "validation", "missing or invalid credentials"))

    auth = [Depends(require_auth)]
    app.add_exception_handler(ApiError, _handle_api_error)

    @app.get("/api/v1/releases/current", dependencies=auth)
    def releases_current(request: Request, response: Response):
        return _release_response(serving_db, resolve_current(), response, request, require_current=True)

    @app.get("/api/v1/releases/{release_id}", dependencies=auth)
    def releases_by_id(release_id: str, request: Request, response: Response):
        return _release_response(serving_db, release_id, response, request, require_current=False)

    @app.get("/api/v1/events", dependencies=auth)
    def events_route(request: Request, response: Response, release_id: str | None = None,
                     date_from: str | None = None, date_to: str | None = None,
                     ticker: str | None = None, strategy: str | None = None, verdict: str | None = None,
                     gate: str | None = None, out_of_domain: str | None = None,
                     disabled: str | None = None,
                     limit: str | None = None, cursor: str | None = None):
        # `date_from`/`date_to` are `ui/src/api/client.ts` `EventQuery`'s own
        # names; `_events_response`'s internal `event_date_from`/
        # `event_date_to` stay projections.py's storage-facing names.
        return _events_response(serving_db, resolve_current, cursor_key, response, request,
                                release_id=release_id, event_date_from=date_from,
                                event_date_to=date_to, ticker=ticker, strategy=strategy,
                                verdict=verdict, gate=gate, out_of_domain=out_of_domain,
                                disabled=disabled, limit=limit, cursor=cursor)

    @app.get("/api/v1/events/{event_id}/scores", dependencies=auth)
    def event_scores_route(event_id: str, request: Request, response: Response,
                           release_id: str | None = None, clock_id: str | None = None):
        return _event_scores_response(serving_db, response, request, event_id=event_id,
                                      release_id=release_id, clock_id=clock_id)

    @app.get("/api/v1/scores/{score_id}", dependencies=auth)
    def score_detail_route(score_id: str, request: Request, response: Response, release_id: str | None = None):
        return _score_detail_response(serving_db, store, response, request,
                                      score_id=score_id, release_id=release_id)

    @app.get("/api/v1/operations", dependencies=auth)
    def operations_route(response: Response, release_id: str | None = None):
        return _operations_response(publication_root, response, release_id=release_id)

    _register_native_parity_routes(app, auth, serving_db, resolve_current, cursor_key,
                                   report_source)

    return app


# --------------------------------------------------------------------------
# launcher -- `python3 -m engine.v2.serving.api ...`
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 serving read API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--serving-db", required=True)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--serving-root", required=True)
    parser.add_argument("--publication-root", default=None,
                        help="the fenced ops publisher's release root for this shadow scope "
                             "(<ops_root>/releases/<scope>); omit for no configured pointer")
    parser.add_argument("--native-parity-report-path", default=None,
                        help="the producer's native parity report JSON; omit for no_report")
    parser.add_argument("--ops-root", default=None,
                        help="the ops root (<ops_root>/catalog.sqlite); when given and no "
                             "explicit report path is set, discover the newest succeeded shadow "
                             "native_parity report read-only from the ops catalog")
    parser.add_argument("--allow-non-loopback", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    token = os.environ.get("V2_DASHBOARD_TOKEN", "")
    if not token:
        print("V2_DASHBOARD_TOKEN is not set; refusing to start")
        return 2
    if args.host not in _LOOPBACK_HOSTS and not args.allow_non_loopback:
        print("refusing a non-loopback host without --allow-non-loopback")
        return 2
    app = create_app(serving_db=args.serving_db, store_root=args.store_root,
                     serving_root=args.serving_root, token=token,
                     publication_root=args.publication_root,
                     ops_root=args.ops_root,
                     native_parity_report_path=args.native_parity_report_path)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
