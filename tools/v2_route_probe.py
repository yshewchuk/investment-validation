#!/usr/bin/env python3
"""P6 route probe: hit every GET route the RUNNING v2 preview server serves.

The route list is imported from ``engine.v2.serving.operations.route_table``
-- the server's own declarative table -- never a hand-maintained copy here, so
a route the server stops answering cannot stay silently unprobed. Only GET
routes are requested: every declared POST route is listed in the receipt as
``skipped_post`` and never sent, because an evidence probe has no business
mutating the server. Each GET request carries the bearer token read from the
``V2_PROBE_TOKEN`` environment variable and is recorded to a session receipt
(``<evidence-dir>/<session>-route_probe.json``) with its status, response size,
latency and timestamp. The token is never written to the receipt or any log.
Redirects are never followed: a 3xx is recorded as the route's own status and
the bearer token is never re-sent to a ``Location`` URL. The probe is
fail-closed: if any probed route does not answer 2xx, ``all_2xx`` is false,
the failures are printed and the process exits 1 -- a 401/403/404/5xx is a
failure here, not a security result.

Parameterized routes are resolved to one concrete in-session path before
probing. The resolution source is the third element the server declares on
each ``PARAMETERIZED_ROUTES`` entry: ``literal:<suffix>`` uses that suffix
directly (``/legacy/*`` ignores it), ``current-release-id`` uses the
``release_id`` from the server's own ``/release/current.json`` body, and
``prior-whatif-job-id`` uses the ``job_id`` a prior ``POST /actions/whatif``
returned -- and since POST routes are never requested, that row is recorded as
``skipped`` and never fabricated as a 2xx.

Receipt rows carry ``declared_path`` on parameterized routes so a reader can
map a concrete request back to the route table entry it covers.

Usage::

    V2_PROBE_TOKEN=... python3 tools/v2_route_probe.py \\
        --base-url http://127.0.0.1:8765 --session 2026-09-25 \\
        --evidence-dir reports/phase6_evidence/route_probe
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.serving.operations import PARAMETERIZED_ROUTES, route_table  # noqa: E402

__all__ = [
    "SCHEMA_VERSION",
    "TOKEN_ENV",
    "RouteProbeError",
    "resolve_parameterized_path",
    "probe",
    "main",
]

SCHEMA_VERSION = "route_probe_receipt.v1.0"

#: The bearer token is read ONLY from this environment variable -- never a
#: command-line argument, so it cannot leak into a shell history or a process
#: listing, and it is never written to a receipt or a log line.
TOKEN_ENV = "V2_PROBE_TOKEN"

#: ``example_suffix_source`` values the probe knows how to resolve.
_LITERAL_PREFIX = "literal:"
_CURRENT_RELEASE_ID = "current-release-id"
_PRIOR_WHATIF_JOB_ID = "prior-whatif-job-id"


class RouteProbeError(RuntimeError):
    """The probe cannot run at all (no token in the environment)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_2xx(row: dict) -> bool:
    status = row.get("status")
    return isinstance(status, int) and not isinstance(status, bool) and 200 <= status < 300


def _token_from_env() -> str:
    token = os.environ.get(TOKEN_ENV, "")
    if not token:
        raise RouteProbeError(f"no probe token: set {TOKEN_ENV} in the environment")
    return token


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: a 3xx is the route's recorded result, and the
    bearer token is never re-sent to the URL a ``Location`` header names."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _document_field(payload: bytes, name: str):
    """A nonempty string field of a JSON response body, else ``None``."""
    try:
        document = json.loads(payload)
    except ValueError:
        return None
    value = document.get(name) if isinstance(document, dict) else None
    return value if isinstance(value, str) and value else None


def _request(base_url: str, path: str, method: str, token: str, *,
             timeout: float):
    """Issue one real request; return ``(row, response_bytes)``.

    Redirects are refused, never followed: a 3xx is recorded as the route's
    own status and the bearer token is never re-sent to the URL a ``Location``
    header names. An unreached route records ``status: null`` plus the error,
    never a fabricated success.
    """
    request = urllib.request.Request(base_url.rstrip("/") + path, method=method)
    request.add_header("Authorization", "Bearer " + token)
    opener = urllib.request.build_opener(_NoRedirects())
    requested_at = _now()
    started = time.perf_counter()
    status = None
    error = None
    payload = b""
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read()
        finally:
            exc.close()
        status = exc.code
    except (urllib.error.URLError, OSError, ValueError) as exc:
        error = str(exc)
    row = {
        "method": method,
        "path": path,
        "status": status,
        "bytes": len(payload),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "requested_at": requested_at,
    }
    if error is not None:
        row["error"] = error
    return row, payload


def _skipped_post(route: dict) -> dict:
    return {"method": route["method"], "path": route["path"],
            "skipped_post": "POST routes are never requested by this probe"}


def resolve_parameterized_path(declared_path: str, example_suffix_source: str, *,
                               release_id=None, job_id=None):
    """``(concrete_path, None)`` or ``(None, reason)`` for one declared route.

    A ``None`` reason never means "close enough": the caller records the skip
    or the failure, never a 2xx it did not observe.
    """
    if example_suffix_source.startswith(_LITERAL_PREFIX):
        return declared_path + example_suffix_source[len(_LITERAL_PREFIX):], None
    if example_suffix_source == _CURRENT_RELEASE_ID:
        if release_id is None:
            return None, "no current release id in this session"
        return f"{declared_path}{release_id}/index.html", None
    if example_suffix_source == _PRIOR_WHATIF_JOB_ID:
        if job_id is None:
            return None, "no job id in this session"
        if release_id is None:
            return None, "no current release id in this session"
        return f"{declared_path}{job_id}?release_id={release_id}", None
    return None, f"unknown example_suffix_source for {declared_path}: {example_suffix_source!r}"


def _probed(row: dict) -> bool:
    """A row the probe actually requested (not a skip of either kind)."""
    return "skipped" not in row and "skipped_post" not in row


def probe(*, base_url: str, session: str, evidence_dir, timeout: float = 10.0) -> dict:
    """Probe every declared GET route and write ``<session>-route_probe.json``."""
    token = _token_from_env()
    routes = route_table()

    rows: list[dict] = []
    current_release_id = None
    for route in routes:
        if route["method"] != "GET":
            rows.append(_skipped_post(route))
            continue
        if route["parameterized"]:
            continue
        declared = route["path"]
        row, payload = _request(base_url, declared, route["method"], token,
                                timeout=timeout)
        rows.append(row)
        if declared == "/release/current.json":
            current_release_id = _document_field(payload, "release_id")

    sources = {prefix: source for _, prefix, source in PARAMETERIZED_ROUTES}
    for route in routes:
        if not route["parameterized"] or route["method"] != "GET":
            continue
        declared = route["path"]
        source = sources.get(declared, "")
        if source == _PRIOR_WHATIF_JOB_ID:
            rows.append({"method": route["method"], "path": declared,
                         "skipped": "no job id in this session"})
            continue
        path, reason = resolve_parameterized_path(
            declared, source, release_id=current_release_id, job_id=None)
        if path is None:
            rows.append({"method": route["method"], "path": declared,
                         "declared_path": declared, "status": None, "bytes": 0,
                         "latency_ms": 0.0, "requested_at": _now(), "error": reason})
            continue
        row, _ = _request(base_url, path, route["method"], token, timeout=timeout)
        row["declared_path"] = declared
        rows.append(row)

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "session": session,
        "base_url": base_url,
        "generated_at": _now(),
        "routes": rows,
        "all_2xx": all(_is_2xx(row) for row in rows if _probed(row)),
    }
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / f"{session}-route_probe.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=10.0,
                        help=f"per-request timeout; the token comes from {TOKEN_ENV}")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        receipt = probe(base_url=args.base_url, session=args.session,
                        evidence_dir=args.evidence_dir, timeout=args.timeout)
    except RouteProbeError as exc:
        print(f"route-probe: refused: {exc}", file=sys.stderr)
        return 2
    for row in receipt["routes"]:
        if "skipped" in row:
            print(f"route-probe: SKIPPED {row['method']} {row['path']}: {row['skipped']}")
        elif "skipped_post" in row:
            print(f"route-probe: SKIPPED-POST {row['method']} {row['path']}: "
                  f"{row['skipped_post']}")
        elif not _is_2xx(row):
            detail = row.get("error") or f"status {row.get('status')}"
            print(f"route-probe: FAIL {row['method']} {row['path']}: {detail}",
                  file=sys.stderr)
    print(f"route-probe: {len(receipt['routes'])} route(s), all_2xx={receipt['all_2xx']}")
    return 0 if receipt["all_2xx"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
