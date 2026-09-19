#!/usr/bin/env python3
"""P6-5 export-remote-delivery: the phone/remote access path must never serve
a snapshot unauthenticated, on ANY bind address.

Context (see ``dashboard-bind-all-interfaces`` in project memory): the
LEGACY desk app (``dashboard/earnings_app.py``) defaults to binding
``0.0.0.0`` with no auth layer at all, and its ``POST /api/refresh`` is
reachable by anyone who can route to the port. The v2 replacement for
phone/remote access is ``engine.v2.dashboard.preview`` (CLI launcher) over
``engine.v2.serving.operations.create_server`` -- a token-gated transport
where Cloudflare Access (or any perimeter control) is defense in depth, not
the only thing standing between the board and the network, because the app
itself refuses every route without a valid bearer token.

This check fails if ANY of the following is true:

* a request with no token, or the wrong token, against a server bound to
  loopback still returns 200 for a protected route (data/release/health);
* the SAME probe against a server bound to ``0.0.0.0`` (every interface)
  still returns 200 -- binding wide must not be an alternate way to skip
  auth;
* ``create_server``'s ``token`` parameter has a default, i.e. some caller
  could start it without one;
* the CLI launcher (``engine.v2.dashboard.preview.run``) starts without
  ``V2_DASHBOARD_TOKEN`` set, or silently accepts a non-loopback host with
  no explicit ``--allow-non-loopback`` opt-in.

Usage::

    python3 checks/rearchitecture_phase6_remote_access.py
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.dashboard import preview  # noqa: E402
from engine.v2.serving.operations import create_server  # noqa: E402

TOKEN = "remote-access-check-token"


def _write_release(root: Path, release_id: str = "r1") -> None:
    release_dir = root / "releases" / release_id
    (release_dir / "data").mkdir(parents=True)
    (release_dir / "index.html").write_text("<!doctype html><meta charset=\"utf-8\">ok")
    (release_dir / "data" / "board.json").write_text(json.dumps({"release": release_id}))
    (root / "CURRENT").write_text(release_id + "\n")
    (root / "health.json").write_text(json.dumps({
        "schema_version": "operations_health.v1.0", "generated_at": "2026-09-19T00:00:00Z",
        "withheld_release": None}))


def _finding(code: str, detail: str) -> dict:
    return {"code": code, "detail": detail}


def _get(base: str, path: str, token: str | None) -> int:
    request = Request(base + path)
    if token is not None:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urlopen(request, timeout=5) as response:
            return response.status
    except HTTPError as exc:
        return exc.code


def _probe_bind(root: Path, host: str) -> list[dict]:
    """Start a real server on ``host`` and confirm every protected route
    refuses a missing/wrong token, regardless of the bind address."""
    server = create_server((host, 0), token=TOKEN, health_path=root / "health.json",
                           release_root=root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    findings: list[dict] = []
    client_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    base = f"http://{client_host}:{server.server_port}"
    label = "BIND_ALL" if host in ("0.0.0.0", "::") else "LOOPBACK"
    try:
        for path in ("/release/current.json", "/release/r1/index.html",
                     "/release/r1/data/board.json", "/health.json"):
            for bad_token in (None, "wrong-" + TOKEN):
                status = _get(base, path, bad_token)
                if status == 200:
                    findings.append(_finding(
                        f"UNAUTHENTICATED_{label}",
                        f"{path} returned 200 with token={bad_token!r} on host={host!r}"))
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    return findings


def _check_token_has_no_default() -> list[dict]:
    signature = inspect.signature(create_server)
    token_param = signature.parameters.get("token")
    if token_param is None or token_param.default is not inspect.Parameter.empty:
        return [_finding("TOKEN_PARAM_HAS_DEFAULT",
                         "create_server's token parameter is optional or missing")]
    return []


def _check_cli_refuses_without_token(root: Path) -> list[dict]:
    findings = []
    saved = os.environ.pop(preview.TOKEN_ENV_VAR, None)
    try:
        try:
            preview.run(["--host", "127.0.0.1", "--port", "0",
                        "--release-root", str(root), "--health-path", str(root / "health.json")])
            findings.append(_finding("CLI_STARTS_WITHOUT_TOKEN",
                                     "preview.run started with no V2_DASHBOARD_TOKEN set"))
        except SystemExit:
            pass
    finally:
        if saved is not None:
            os.environ[preview.TOKEN_ENV_VAR] = saved
    return findings


def _check_cli_refuses_bind_all_without_flag(root: Path) -> list[dict]:
    findings = []
    saved = os.environ.get(preview.TOKEN_ENV_VAR)
    os.environ[preview.TOKEN_ENV_VAR] = TOKEN
    try:
        try:
            server, thread, _release_id = preview.run(
                ["--host", "0.0.0.0", "--port", "0",
                 "--release-root", str(root), "--health-path", str(root / "health.json")])
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()
            findings.append(_finding("CLI_ALLOWS_BIND_ALL_WITHOUT_FLAG",
                                     "preview.run bound 0.0.0.0 with no --allow-non-loopback"))
        except SystemExit:
            pass
    finally:
        if saved is None:
            os.environ.pop(preview.TOKEN_ENV_VAR, None)
        else:
            os.environ[preview.TOKEN_ENV_VAR] = saved
    return findings


def run_checks(tmp_root: Path) -> list[dict]:
    _write_release(tmp_root)
    findings: list[dict] = []
    findings += _check_token_has_no_default()
    findings += _probe_bind(tmp_root, "127.0.0.1")
    findings += _probe_bind(tmp_root, "0.0.0.0")
    findings += _check_cli_refuses_without_token(tmp_root)
    findings += _check_cli_refuses_bind_all_without_flag(tmp_root)
    return findings


def main(argv=None) -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        findings = run_checks(Path(scratch))
    for finding in findings:
        print(f"{finding['code']}: {finding['detail']}")
    verdict = "FAIL" if findings else "PASS"
    print(f"remote_access: {verdict} ({len(findings)} finding(s))")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
