#!/usr/bin/env python3
"""L08 evidence producer -- guide §9 row L08.

One ``api_auth_traversal_negative_control`` receipt covering both real
servers this package ships:

* ``engine.v2.serving.api`` (the read API): unauthenticated/wrong-token GETs
  against data, detail and operations routes.
* ``engine.v2.serving.operations`` (the compatibility preview / "legacy
  file" server): unauthenticated GETs, a ``..``-traversal path, and a real
  symlink planted inside the release tree that tries to escape the release
  root.

**Convention (a judgement call, consistent with every other negative-control
producer here): ``verdict=DIFFER`` means every probe was correctly refused
(the safe, desired outcome); ``verdict=AGREE`` means at least one probe
leaked real content** (a :class:`Finding` is recorded for it -- exactly what
``NEGATIVE_CONTROL_NOT_TRIGGERED`` exists to catch).

Read-only over whatever release tree/serving db it is pointed at; the one
symlink it creates lives under a caller-supplied SCRATCH directory, never a
shared root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase1_gate import source_files, source_hash  # noqa: E402
from checks.rearchitecture_phase2_gate import environment_hash as _environment_hash  # noqa: E402
from engine.v2.diagnosis import AGREE, DIFFER, ComparisonReceipt, Envelope, Finding, Population, content_hash  # noqa: E402
from engine.v2.foundation import to_document  # noqa: E402
from engine.v2.serving.api import create_app  # noqa: E402
from engine.v2.serving.operations import create_server  # noqa: E402

KIND = "api_auth_traversal_negative_control"


def _start_operations(release_root: Path, health_path: Path, token: str):
    server = create_server(("127.0.0.1", 0), token=token, health_path=health_path, release_root=release_root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def _stop_operations(server, thread) -> None:
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def _start_api(app):
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


def _stop_api(server, thread) -> None:
    server.should_exit = True
    thread.join(timeout=5)


def _get(base: str, path: str, token: str | None) -> tuple[int, bytes]:
    request = Request(base + path)
    if token is not None:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def _plant_symlink(release_root: Path, release_id: str, *, target: Path) -> str:
    """A real symlink under ``releases/<release_id>/`` pointing OUTSIDE the
    release tree, at the caller-given ``target`` (a scratch file this
    process itself creates -- never a real secret). Returns the relative
    path the probe requests."""
    link = release_root / "releases" / release_id / "escape.json"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)
    return "escape.json"


def build(release_root: Path, health_path: Path, *, release_id: str, token: str,
         serving_db: Path, store_root: Path, serving_root: Path, scratch_dir: Path,
         api_release_id: str | None) -> ComparisonReceipt:
    findings: list[Finding] = []
    checks = 0

    scratch_dir.mkdir(parents=True, exist_ok=True)
    outside_secret = scratch_dir / "outside_release_root.txt"
    outside_secret.write_text("not part of any release\n")
    symlink_relpath = _plant_symlink(release_root, release_id, target=outside_secret)

    op_server, op_thread, op_base = _start_operations(release_root, health_path, token)
    try:
        for label, path, bad_token in (
            ("op_current_json_no_token", "/release/current.json", None),
            ("op_current_json_wrong_token", "/release/current.json", "wrong-" + token),
            ("op_health_no_token", "/health.json", None),
            ("op_release_file_no_token", f"/release/{release_id}/index.html", None),
        ):
            checks += 1
            status, body = _get(op_base, path, bad_token)
            if status == 200:
                findings.append(Finding(
                    finding_id=content_hash([KIND, label])[7:19], first_differing_stage="auth",
                    field_path=label, kind="value", owning_stage="auth"))

        checks += 1
        traversal_path = f"/release/{release_id}/../../../etc/passwd"
        status, body = _get(op_base, traversal_path, token)
        if status == 200:
            findings.append(Finding(
                finding_id=content_hash([KIND, "traversal_etc_passwd"])[7:19],
                first_differing_stage="traversal", field_path="traversal_etc_passwd",
                kind="value", owning_stage="traversal"))

        checks += 1
        status, body = _get(op_base, f"/release/{release_id}/{symlink_relpath}", token)
        if status == 200 and body == outside_secret.read_bytes():
            findings.append(Finding(
                finding_id=content_hash([KIND, "symlink_escape"])[7:19],
                first_differing_stage="traversal", field_path="symlink_escape",
                kind="value", owning_stage="traversal"))
    finally:
        _stop_operations(op_server, op_thread)
        (release_root / "releases" / release_id / symlink_relpath).unlink(missing_ok=True)

    app = create_app(serving_db=str(serving_db), store_root=str(store_root), serving_root=str(serving_root),
                     token=token, resolver=(lambda: api_release_id) if api_release_id else None)
    api_server, api_thread, api_base = _start_api(app)
    try:
        for label, path, params in (
            ("api_current_no_token", "/api/v1/releases/current", None),
            ("api_events_wrong_token", "/api/v1/events", None),
            ("api_operations_no_token", "/api/v1/operations", None),
        ):
            checks += 1
            url = path
            status, body = _get(api_base, url, None if "no_token" in label else "wrong-" + token)
            if status == 200:
                findings.append(Finding(
                    finding_id=content_hash([KIND, label])[7:19], first_differing_stage="auth",
                    field_path=label, kind="value", owning_stage="auth"))
    finally:
        _stop_api(api_server, api_thread)

    population = Population(expected=checks, supported=checks, compared=checks)
    verdict = AGREE if findings else DIFFER
    code_hash = source_hash(source_files(ROOT))
    env_hash, _source = _environment_hash(ROOT)
    envelope = Envelope(code_hash=code_hash, environment_hash=env_hash)
    receipt_id = content_hash([KIND, release_id, [f.finding_id for f in findings]])[7:23]
    return ComparisonReceipt(
        receipt_id=receipt_id, comparison_kind=KIND, tier=1,
        left_ref="authorized:" + release_id, right_ref="unauthorized_or_traversal:" + release_id,
        stage_plan_ref="api_auth_traversal.v1", tolerance_policy_ref="exact_bytes.v1",
        verdict=verdict, findings=tuple(findings), population=population, envelope=envelope)


def publish(receipt: ComparisonReceipt, artifact_root: Path) -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    data = json.dumps(to_document(receipt), indent=2, sort_keys=True).encode()
    path = artifact_root / "api_auth_traversal_negative_control.json"
    path.write_bytes(data)
    return {"path": path.name, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--health-path", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--serving-db", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--serving-root", type=Path, required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--api-release-id", default=None)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    receipt = build(args.release_root, args.health_path, release_id=args.release_id, token=args.token,
                    serving_db=args.serving_db, store_root=args.store_root, serving_root=args.serving_root,
                    scratch_dir=args.scratch_dir, api_release_id=args.api_release_id)
    ref = publish(receipt, args.artifact_root)
    print(json.dumps({**ref, "verdict": receipt.verdict}, indent=2))
    return 0 if receipt.verdict == DIFFER else 1


if __name__ == "__main__":
    raise SystemExit(main())
