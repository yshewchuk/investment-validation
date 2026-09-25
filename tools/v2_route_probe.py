#!/usr/bin/env python3
"""P6 route probe: hit every route the RUNNING v2 preview server serves.

The route list is imported from ``engine.v2.serving.operations.route_table``
-- the server's own declarative table -- never a hand-maintained copy here, so
a route the server stops answering cannot stay silently unprobed. Each request
carries the bearer token and is recorded to a session receipt
(``<evidence-dir>/<session>-route_probe.json``) with its status, response size,
latency and timestamp. The probe is fail-closed: if any declared route does
not answer 2xx (after redirects), ``all_2xx`` is false, the failures are
printed and the process exits 1 -- a 401/403/404/5xx is a failure here, not a
security result.

Parameterized routes are resolved to one concrete in-session path before
probing. The resolution source is the third element the server declares on
each ``PARAMETERIZED_ROUTES`` entry: ``literal:<suffix>`` uses that suffix
directly (``/legacy/*`` ignores it), ``current-release-id`` uses the
``release_id`` from the server's own ``/release/current.json`` body, and
``prior-whatif-job-id`` uses the ``job_id`` the static ``POST /actions/whatif``
returned in this same run -- without one, that row is recorded as
``skipped`` and never fabricated as a 2xx. A declared POST route with no
supplied body is refused outright: never silently skipped.

Receipt rows carry ``declared_path`` on parameterized routes so a reader can
map a concrete request back to the route table entry it covers.

Usage::

    python3 tools/v2_route_probe.py --base-url http://127.0.0.1:8765 \\
        --token "$V2_DASHBOARD_TOKEN" --session 2026-09-25 \\
        --refresh-body '{"plan_ref": "plan-1"}' \\
        --whatif-body '{"request": {}, "native_inputs": {}}' \\
        --evidence-dir reports/phase6_evidence/route_probe
"""
from __future__ import annotations

import argparse
import json
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
    "RouteProbeError",
    "resolve_parameterized_path",
    "probe",
    "main",
]

SCHEMA_VERSION = "route_probe_receipt.v1.0"

#: ``example_suffix_source`` values the probe knows how to resolve.
_LITERAL_PREFIX = "literal:"
_CURRENT_RELEASE_ID = "current-release-id"
_PRIOR_WHATIF_JOB_ID = "prior-whatif-job-id"

#: POST route path -> the CLI argument that must supply its probe body.
_POST_BODY_ARGS = {"/actions/refresh": "refresh_body", "/actions/whatif": "whatif_body"}


class RouteProbeError(RuntimeError):
    """A declared route cannot be probed at all (no body, bad JSON)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_2xx(row: dict) -> bool:
    status = row.get("status")
    return isinstance(status, int) and not isinstance(status, bool) and 200 <= status < 300


def _json_object(raw, label: str):
    """Parse a CLI body argument into a JSON object, or refuse clearly."""
    if raw is None:
        return None
    try:
        document = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except ValueError as exc:
        raise RouteProbeError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise RouteProbeError(f"{label} must be a JSON object")
    return document


def _document_field(payload: bytes, name: str):
    """A nonempty string field of a JSON response body, else ``None``."""
    try:
        document = json.loads(payload)
    except ValueError:
        return None
    value = document.get(name) if isinstance(document, dict) else None
    return value if isinstance(value, str) and value else None


def _request(base_url: str, path: str, method: str, token: str, *,
             body=None, timeout: float):
    """Issue one real request; return ``(row, response_bytes)``.

    Redirects are followed (urllib's default), so ``/release/current``'s
    documented 302-to-200 is recorded as its final status; an unreached route
    records ``status: null`` plus the error, never a fabricated success.
    """
    data = json.dumps(body, sort_keys=True).encode() if body is not None else None
    request = urllib.request.Request(base_url.rstrip("/") + path, data=data, method=method)
    request.add_header("Authorization", "Bearer " + token)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    requested_at = _now()
    started = time.perf_counter()
    status = None
    error = None
    payload = b""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
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


def probe(*, base_url: str, token: str, session: str, evidence_dir, refresh_body=None,
          whatif_body=None, timeout: float = 10.0) -> dict:
    """Probe every declared route and write ``<session>-route_probe.json``."""
    bodies = {
        "refresh_body": _json_object(refresh_body, "--refresh-body"),
        "whatif_body": _json_object(whatif_body, "--whatif-body"),
    }
    routes = route_table()

    # Refuse before any request when a declared POST route has no body: never
    # silently skip a declared route, and never half-probe a session first.
    post_bodies: dict[str, dict] = {}
    for route in routes:
        if route["parameterized"] or route["method"] != "POST":
            continue
        argument = _POST_BODY_ARGS.get(route["path"])
        if argument is None:
            raise RouteProbeError(
                f"declared POST route {route['path']} has no known body argument; "
                "the probe refuses to skip a declared route")
        if bodies[argument] is None:
            raise RouteProbeError(
                f"declared POST route {route['path']} needs a probe body; pass "
                f"--{argument.replace('_', '-')}")
        post_bodies[route["path"]] = bodies[argument]

    rows: list[dict] = []
    current_release_id = None
    whatif_job_id = None
    for route in routes:
        if route["parameterized"]:
            continue
        declared = route["path"]
        row, payload = _request(base_url, declared, route["method"], token,
                                body=post_bodies.get(declared), timeout=timeout)
        rows.append(row)
        if declared == "/release/current.json":
            current_release_id = _document_field(payload, "release_id")
        if declared == "/actions/whatif":
            whatif_job_id = _document_field(payload, "job_id")

    sources = {prefix: source for _, prefix, source in PARAMETERIZED_ROUTES}
    for route in routes:
        if not route["parameterized"]:
            continue
        declared = route["path"]
        source = sources.get(declared, "")
        if source == _PRIOR_WHATIF_JOB_ID and whatif_job_id is None:
            rows.append({"method": route["method"], "path": declared,
                         "skipped": "no job id in this session"})
            continue
        path, reason = resolve_parameterized_path(
            declared, source, release_id=current_release_id, job_id=whatif_job_id)
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
        "all_2xx": all(_is_2xx(row) for row in rows if "skipped" not in row),
    }
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / f"{session}-route_probe.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--refresh-body", default=None,
                        help="JSON object for POST /actions/refresh; required when declared")
    parser.add_argument("--whatif-body", default=None,
                        help="JSON object for POST /actions/whatif; required when declared")
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        receipt = probe(base_url=args.base_url, token=args.token, session=args.session,
                        evidence_dir=args.evidence_dir, refresh_body=args.refresh_body,
                        whatif_body=args.whatif_body, timeout=args.timeout)
    except RouteProbeError as exc:
        print(f"route-probe: refused: {exc}", file=sys.stderr)
        return 2
    for row in receipt["routes"]:
        if "skipped" in row:
            print(f"route-probe: SKIPPED {row['method']} {row['path']}: {row['skipped']}")
        elif not _is_2xx(row):
            detail = row.get("error") or f"status {row.get('status')}"
            print(f"route-probe: FAIL {row['method']} {row['path']}: {detail}",
                  file=sys.stderr)
    print(f"route-probe: {len(receipt['routes'])} route(s), all_2xx={receipt['all_2xx']}")
    return 0 if receipt["all_2xx"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
