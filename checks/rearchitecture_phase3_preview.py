#!/usr/bin/env python3
"""P3-0 compatibility preview evidence producer -- guide §9 row L01.

Builds two real ``ComparisonReceipt``s against an already-materialized
operations release tree -- the exact shape
``engine.v2.serving.operations.create_server`` expects::

    <release-root>/CURRENT
    <release-root>/releases/<id>/...   (index.html, data/board.json, ...)
    <release-root>/health.json

Starts a REAL ``OperationsHandler`` server (real bytes on disk, real HTTP
over a loopback socket), probes it, and shuts it down. Never writes to
``--release-root``.

* ``preview_open_parity``: every file under ``releases/<id>/`` plus
  ``health.json`` and the ``/release/current.json`` resolution are compared,
  byte for byte, between what is on disk and what the authenticated server
  actually serves. ``AGREE`` only when the compared population is nonempty
  and every file matches.

* ``preview_auth_negative_control``: for the same protected routes, an
  authenticated response is compared against a no-token and a wrong-token
  response. **Convention (a judgement call, documented here since the guide
  does not spell out the mechanism -- mirrors every other negative-control
  producer in this package for consistency):** unlike a normal comparison
  receipt, a negative control's two sides are EXPECTED to differ -- that
  is what proves the fault (missing/bad auth) was refused rather than
  served. So here ``verdict=DIFFER`` means "the authenticated and
  unauthenticated responses differed, i.e. auth correctly gated real
  content" (the desired, safe outcome), and ``verdict=AGREE`` means "the
  unauthenticated response leaked the same bytes as the authenticated one"
  (a real auth bypass -- a :class:`Finding` is recorded for it, and
  ``checks/rearchitecture_phase3_evidence.py``'s
  ``NEGATIVE_CONTROL_NOT_TRIGGERED`` is exactly the code that should catch
  this if it ever happens for real).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase1_gate import source_files, source_hash  # noqa: E402
from checks.rearchitecture_phase2_gate import environment_hash as _environment_hash  # noqa: E402
from engine.v2.diagnosis import AGREE, DIFFER, ComparisonReceipt, Envelope, Finding, Population, content_hash  # noqa: E402
from engine.v2.foundation import to_document  # noqa: E402
from engine.v2.serving.operations import create_server  # noqa: E402

OPEN_PARITY_KIND = "preview_open_parity"
AUTH_NEGATIVE_KIND = "preview_auth_negative_control"


def _start(release_root: Path, health_path: Path, token: str):
    server = create_server(("127.0.0.1", 0), token=token, health_path=health_path,
                           release_root=release_root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def _stop(server, thread) -> None:
    server.shutdown()
    thread.join(timeout=5)
    server.server_close()


def _get(base: str, path: str, token: str | None) -> tuple[int, bytes]:
    request = urllib.request.Request(base + path)
    if token is not None:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _current_release_id(release_root: Path) -> str:
    return (release_root / "CURRENT").read_text().strip()


def _release_files(release_root: Path, release_id: str) -> list[str]:
    base = release_root / "releases" / release_id
    return sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())


def build_open_parity(release_root: Path, health_path: Path, *, token: str, code_hash: str,
                      environment_hash: str) -> ComparisonReceipt:
    release_id = _current_release_id(release_root)
    files = _release_files(release_root, release_id)
    base, server, thread = None, None, None
    findings: list[Finding] = []
    compared = 0
    if not files:
        # An empty/missing release directory is a collapsed population, not
        # agreement -- mirrors this codebase's general Population.collapsed
        # rule (engine/v2/diagnosis/receipt.py): zero files compared can
        # never be reported as "the preview matches the release".
        findings.append(Finding(
            finding_id=content_hash(["preview_open_parity", "empty_release"])[7:19],
            first_differing_stage="serve", field_path="releases/" + release_id,
            kind="missing_field", owning_stage="serve"))
    server, thread, base = _start(release_root, health_path, token)
    try:
        status, body = _get(base, "/release/current.json", token)
        compared += 1
        if status != 200 or json.loads(body).get("release_id") != release_id:
            findings.append(Finding(
                finding_id=content_hash(["preview_open_parity", "current.json"])[7:19],
                first_differing_stage="serve", field_path="release/current.json",
                kind="value", owning_stage="serve"))
        status, body = _get(base, "/health.json", token)
        compared += 1
        on_disk = health_path.read_bytes()
        if status != 200 or json.loads(body) != json.loads(on_disk):
            findings.append(Finding(
                finding_id=content_hash(["preview_open_parity", "health.json"])[7:19],
                first_differing_stage="serve", field_path="health.json",
                kind="value", owning_stage="serve"))
        for relative in files:
            compared += 1
            on_disk_bytes = (release_root / "releases" / release_id / relative).read_bytes()
            status, body = _get(base, f"/release/{release_id}/{relative}", token)
            if status != 200 or body != on_disk_bytes:
                findings.append(Finding(
                    finding_id=content_hash(["preview_open_parity", relative])[7:19],
                    first_differing_stage="serve", field_path=relative,
                    kind="value", owning_stage="serve"))
    finally:
        _stop(server, thread)
    population = Population(expected=compared, supported=compared, compared=compared)
    verdict = DIFFER if findings else (AGREE if compared > 0 else DIFFER)
    envelope = Envelope(code_hash=code_hash, environment_hash=environment_hash)
    receipt_id = content_hash(["preview_open_parity", release_id, [f.finding_id for f in findings]])[7:23]
    return ComparisonReceipt(
        receipt_id=receipt_id, comparison_kind=OPEN_PARITY_KIND, tier=1,
        left_ref="disk:" + release_id, right_ref="release:" + release_id,
        stage_plan_ref="preview_open.v1", tolerance_policy_ref="exact_bytes.v1",
        verdict=verdict, findings=tuple(findings), population=population, envelope=envelope)


_PROTECTED_ROUTES_TEMPLATE = ("/release/current.json", "/release/{release_id}/index.html", "/health.json")


def build_auth_negative_control(release_root: Path, health_path: Path, *, token: str, code_hash: str,
                                environment_hash: str) -> ComparisonReceipt:
    release_id = _current_release_id(release_root)
    routes = [r.format(release_id=release_id) for r in _PROTECTED_ROUTES_TEMPLATE]
    findings: list[Finding] = []
    compared = 0
    server, thread, base = _start(release_root, health_path, token)
    try:
        for route in routes:
            auth_status, auth_body = _get(base, route, token)
            for label, bad_token in (("missing_token", None), ("wrong_token", "wrong-" + token)):
                compared += 1
                bad_status, bad_body = _get(base, route, bad_token)
                leaked = bad_status == 200 and auth_status == 200 and bad_body == auth_body
                if leaked:
                    findings.append(Finding(
                        finding_id=content_hash(["preview_auth_negative_control", route, label])[7:19],
                        first_differing_stage="auth", field_path=f"{route}::{label}",
                        kind="value", owning_stage="auth"))
    finally:
        _stop(server, thread)
    population = Population(expected=compared, supported=compared, compared=compared)
    # judgement call (module docstring): a fired control differs; a bypassed one agrees.
    verdict = AGREE if findings else DIFFER
    envelope = Envelope(code_hash=code_hash, environment_hash=environment_hash)
    receipt_id = content_hash(["preview_auth_negative_control", release_id,
                               [f.finding_id for f in findings]])[7:23]
    return ComparisonReceipt(
        receipt_id=receipt_id, comparison_kind=AUTH_NEGATIVE_KIND, tier=1,
        left_ref="authenticated:" + release_id, right_ref="unauthenticated:" + release_id,
        stage_plan_ref="preview_auth.v1", tolerance_policy_ref="exact_bytes.v1",
        verdict=verdict, findings=tuple(findings), population=population, envelope=envelope)


def build(release_root: Path, health_path: Path, *, token: str) -> tuple[ComparisonReceipt, ComparisonReceipt]:
    code_hash = source_hash(source_files(ROOT))
    env_hash, _source = _environment_hash(ROOT)
    open_receipt = build_open_parity(release_root, health_path, token=token, code_hash=code_hash,
                                     environment_hash=env_hash)
    auth_receipt = build_auth_negative_control(release_root, health_path, token=token, code_hash=code_hash,
                                               environment_hash=env_hash)
    return open_receipt, auth_receipt


def publish(receipt: ComparisonReceipt, artifact_root: Path, name: str) -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    data = json.dumps(to_document(receipt), indent=2, sort_keys=True).encode()
    path = artifact_root / name
    path.write_bytes(data)
    return {"path": path.name, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--health-path", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    open_receipt, auth_receipt = build(args.release_root, args.health_path, token=args.token)
    open_ref = publish(open_receipt, args.artifact_root, "preview_open_parity.json")
    auth_ref = publish(auth_receipt, args.artifact_root, "preview_auth_negative_control.json")
    print(json.dumps({
        "preview_open_parity": {**open_ref, "verdict": open_receipt.verdict},
        "preview_auth_negative_control": {**auth_ref, "verdict": auth_receipt.verdict},
    }, indent=2))
    return 0 if open_receipt.verdict == AGREE and auth_receipt.verdict == DIFFER else 1


if __name__ == "__main__":
    raise SystemExit(main())
