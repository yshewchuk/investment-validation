"""Compatibility preview launcher — composes ``engine.v2.serving`` only.

Starts the authenticated operations server (``engine.v2.serving.operations``),
loopback-only unless told otherwise, resolves the pinned current release id
exactly once by calling the server's own ``/release/current.json`` route, and
prints the URL plus that id. The token is read from an environment variable
and is never printed or logged.

``--model-release-root`` points the server's ``/models/release.json`` view at
the deployed model release store — a distinct store from the dashboard bundle
``--release-root``, never guessed from it. Omitting it keeps the route's
explicit ``MODEL_RELEASE_NOT_CONFIGURED`` refusal.

``--calibration-health-path`` points the server's ``/calibration-health.json``
view at an exported copy of the ledger calibration producer's own
``ledger_health.v1`` document, again never guessed from ``--health-path``.
Omitting it keeps that route's explicit 503 ``not configured`` refusal.

``--ops-root`` points the authenticated ``POST /actions/refresh`` route at the
catalog/artifact root a nightly plan is published under — again distinct from
``--release-root``, never guessed from it. When set, the route's
``submit_refresh`` callback calls ``engine.v2.ops.cli.refresh_action`` on
exactly that root, so a refresh POST over an already-published nightly plan
submits the shadow nightly (a ``202`` with job ids) rather than the read-only
``503``. Omitting it keeps that explicit 503 refusal; this wires nothing for
production authority and never runs the refresh inline.

``--serving-index-path`` points the authenticated ``GET /analogs.json`` route
at the serving index sqlite file it reads persisted analog row ids from —
again distinct from ``--release-root``, never guessed from it. Omitting it
keeps that route's explicit 503 ``analogs not configured`` refusal.

Run as::

    V2_DASHBOARD_TOKEN=... python3 -m engine.v2.dashboard.preview \\
        --host 127.0.0.1 --port 8765 --release-root R --health-path H \\
        --model-release-root M --ops-root O --calibration-health-path C \\
        --serving-index-path S
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
import threading
import urllib.error
import urllib.request

from engine.v2.dashboard._server import build_server

__all__ = ["TOKEN_ENV_VAR", "is_loopback", "resolve_release_id", "run", "main"]

TOKEN_ENV_VAR = "V2_DASHBOARD_TOKEN"


def is_loopback(host: str) -> bool:
    """True for a literal loopback address or the ``localhost`` name.

    An unparsable host (a real DNS name, or ``0.0.0.0``/``::`` which bind
    every interface rather than naming one) is treated as NOT loopback, so it
    is refused without an explicit opt-in rather than guessed at.
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def resolve_release_id(base_url: str, token: str, *, timeout: float = 5.0) -> str | None:
    """Hit the server's own pinned-id route once; ``None`` if unavailable."""
    request = urllib.request.Request(base_url + "/release/current.json")
    request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())["release_id"]
    except (urllib.error.HTTPError, urllib.error.URLError, ValueError, KeyError, OSError):
        return None


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Read-only v2 compatibility preview launcher.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--health-path", required=True)
    parser.add_argument("--model-release-root", default=None,
                        help="deployed model release store for /models/release.json; distinct "
                             "from --release-root, never inferred from it. Omit to keep the "
                             "route's explicit MODEL_RELEASE_NOT_CONFIGURED refusal.")
    parser.add_argument("--calibration-health-path", default=None,
                        help="exported ledger calibration health JSON (ledger_health.v1) for "
                             "/calibration-health.json; distinct from --health-path, never "
                             "inferred from it. Omit to keep the route's explicit 503 "
                             "'not configured' refusal.")
    parser.add_argument("--ops-root", default=None,
                        help="catalog/artifact root POST /actions/refresh submits an "
                             "already-published nightly plan against, via "
                             "engine.v2.ops.cli.refresh_action; distinct from --release-root, "
                             "never inferred from it. Omit to keep the route's read-only "
                             "503 'refresh not configured' refusal.")
    parser.add_argument("--serving-index-path", default=None,
                        help="the serving index sqlite file GET /analogs.json reads analog "
                             "row ids from; distinct from --release-root, never inferred from "
                             "it. Omit to keep the route's explicit 503 'analogs not "
                             "configured' refusal.")
    parser.add_argument("--frozen-at", default="unknown")
    parser.add_argument("--allow-non-loopback", action="store_true")
    return parser.parse_args(argv)


def run(argv=None):
    """Start the server without blocking; returns ``(server, thread, release_id)``.

    Raises ``SystemExit`` with a token- and secret-free message if the token
    env var is missing or a non-loopback host was given without the flag.
    """
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    token = os.environ.get(TOKEN_ENV_VAR)
    if not token:
        raise SystemExit(f"refusing to start: set {TOKEN_ENV_VAR} to a nonempty token")
    if not args.allow_non_loopback and not is_loopback(args.host):
        raise SystemExit(
            f"refusing non-loopback host {args.host!r} without --allow-non-loopback")
    server = build_server(host=args.host, port=args.port, token=token,
                          health_path=args.health_path, release_root=args.release_root,
                          frozen_at=args.frozen_at, model_release_root=args.model_release_root,
                          ops_root=args.ops_root,
                          calibration_health_path=args.calibration_health_path,
                          serving_index_path=args.serving_index_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    probe_host = args.host if args.host not in ("0.0.0.0", "::") else "127.0.0.1"
    release_id = resolve_release_id(f"http://{probe_host}:{server.server_port}", token)
    return server, thread, release_id


def _shutdown(server, thread):
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


def main(argv=None) -> int:
    try:
        server, thread, release_id = run(argv)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"preview: http://{server.server_address[0]}:{server.server_port}", flush=True)
    print(f"current release: {release_id if release_id is not None else 'unavailable'}", flush=True)
    try:
        thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown(server, thread)
    return 0


# The `== to !=` mutant on the guard below cannot be scored: pytest imports this
# module to collect the serving tests, and a flipped guard runs main() at import,
# so argparse raises SystemExit(2) for the required --release-root/--health-path and
# the runner collects nothing (ERROR, not a kill). The guard's live path is still
# exercised end-to-end by test_launcher_subprocess_starts_on_ephemeral_port_and_serves_shell.
if __name__ == "__main__":  # gremlin: pardon[untestable] flipping this guard runs main() during pytest's import for collection (argparse SystemExit(2) on required args), so no serving test can run; covered by the subprocess launcher test
    raise SystemExit(main())
