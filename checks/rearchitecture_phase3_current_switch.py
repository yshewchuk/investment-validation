#!/usr/bin/env python3
"""L02 evidence producer -- guide §9 row L02 / §5.4's "resolve-once" rule.

Proves, over a REAL operations release tree with two already-materialized
releases (``--release-root/releases/<r1>``, ``<r2>``), that a URL pinned by
an earlier ``/release/current.json`` resolution keeps returning that
release's bytes after ``CURRENT`` is switched underneath it, and that a
FRESH resolution picks up the new release. This producer itself flips
``CURRENT`` inside ``--release-root`` -- always a private scratch tree this
process owns, never a shared/live root (callers must not point it at a live
ops release root).

One ``current_switch_parity`` ``ComparisonReceipt`` (guide §9 L02, must
``AGREE``): every step below is a check; any deviation is a
:class:`Finding` and the verdict is ``differ`` if any exist.
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

CURRENT_SWITCH_KIND = "current_switch_parity"


def _write_current(release_root: Path, release_id: str) -> None:
    (release_root / "CURRENT").write_text(release_id + "\n")


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


def _get(base: str, path: str, token: str) -> tuple[int, bytes]:
    request = urllib.request.Request(base + path)
    request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _finding(findings: list, field: str) -> None:
    findings.append(Finding(
        finding_id=content_hash(["current_switch_parity", field])[7:19],
        first_differing_stage="switch", field_path=field, kind="value", owning_stage="switch"))


def build(release_root: Path, health_path: Path, *, r1: str, r2: str, token: str,
         data_relpath: str = "data/board.json", page_relpath: str = "index.html") -> ComparisonReceipt:
    on_disk_r1 = (release_root / "releases" / r1 / data_relpath).read_bytes()
    on_disk_r2 = (release_root / "releases" / r2 / data_relpath).read_bytes()
    findings: list[Finding] = []
    checks = 0
    _write_current(release_root, r1)
    server, thread, base = _start(release_root, health_path, token)
    try:
        resolved = json.loads(_get(base, "/release/current.json", token)[1])["release_id"]
        checks += 1
        if resolved != r1:
            _finding(findings, "initial_resolution")

        pinned_page = f"/release/{resolved}/{page_relpath}"
        pinned_data = f"/release/{resolved}/{data_relpath}"
        status, body = _get(base, pinned_data, token)
        checks += 1
        if status != 200 or body != on_disk_r1:
            _finding(findings, "pinned_data_before_switch")
        status, page_body = _get(base, pinned_page, token)
        checks += 1
        if status != 200:
            _finding(findings, "pinned_page_before_switch")

        _write_current(release_root, r2)

        status, body_after = _get(base, pinned_data, token)
        checks += 1
        if status != 200 or body_after != on_disk_r1:
            _finding(findings, "pinned_data_survives_switch")
        status, page_after = _get(base, pinned_page, token)
        checks += 1
        if status != 200 or page_after != page_body:
            _finding(findings, "pinned_page_survives_switch")

        fresh = json.loads(_get(base, "/release/current.json", token)[1])["release_id"]
        checks += 1
        if fresh != r2:
            _finding(findings, "fresh_resolution_after_switch")
        status, fresh_data = _get(base, f"/release/{fresh}/{data_relpath}", token)
        checks += 1
        if status != 200 or fresh_data != on_disk_r2:
            _finding(findings, "fresh_data_matches_r2")
    finally:
        _stop(server, thread)

    population = Population(expected=checks, supported=checks, compared=checks)
    verdict = DIFFER if findings else AGREE
    code_hash = source_hash(source_files(ROOT))
    env_hash, _source = _environment_hash(ROOT)
    envelope = Envelope(code_hash=code_hash, environment_hash=env_hash)
    receipt_id = content_hash(["current_switch_parity", r1, r2, [f.finding_id for f in findings]])[7:23]
    return ComparisonReceipt(
        receipt_id=receipt_id, comparison_kind=CURRENT_SWITCH_KIND, tier=1,
        left_ref="operations_release:" + r1, right_ref="operations_release:" + r2,
        stage_plan_ref="current_switch.v1", tolerance_policy_ref="exact_bytes.v1",
        verdict=verdict, findings=tuple(findings), population=population, envelope=envelope)


def publish(receipt: ComparisonReceipt, artifact_root: Path) -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    data = json.dumps(to_document(receipt), indent=2, sort_keys=True).encode()
    path = artifact_root / "current_switch_parity.json"
    path.write_bytes(data)
    return {"path": path.name, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--health-path", type=Path, required=True)
    parser.add_argument("--r1", required=True)
    parser.add_argument("--r2", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build(args.release_root, args.health_path, r1=args.r1, r2=args.r2, token=args.token)
    ref = publish(receipt, args.artifact_root)
    print(json.dumps({**ref, "verdict": receipt.verdict}, indent=2))
    return 0 if receipt.verdict == AGREE else 1


if __name__ == "__main__":
    raise SystemExit(main())
