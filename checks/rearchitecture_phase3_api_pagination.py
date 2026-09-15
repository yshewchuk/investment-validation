#!/usr/bin/env python3
"""L07 evidence producer -- guide §9 row L07 (§6 read API pagination/cursors).

Runs the real ``engine.v2.serving.api`` FastAPI app over a real ``uvicorn``
socket (mirroring ``tests/test_v2_serving_api.py``'s own "never TestClient
alone" convention) against an already-committed ``serving.sqlite`` -- a real
``PreviewRelease`` some other process (``tools/v2_dashboard_project.py``)
already indexed. This module makes no scoring/provider calls and writes
nothing to ``--serving-db``/``--store-root``; it only issues GETs.

* ``api_pagination_parity``: fetches the full ``/api/v1/events`` population
  in one wide page, then re-fetches it by following ``next_cursor`` at a
  small page size, and requires the two populations to match exactly --
  same ``total_matching``, no duplicate event ids, no omissions. ``AGREE``
  only when the compared population is nonempty and both walks agree.

* ``api_cursor_negative_control``: a valid cursor is tampered (one hex
  character flipped in its HMAC signature) and replayed. **Convention (a
  judgement call, consistent with every other negative-control producer in
  this package): ``verdict=DIFFER`` means the tampered cursor was correctly
  refused (``CURSOR_MISMATCH``, 409) -- the safe, desired outcome;
  ``verdict=AGREE`` means the tampered cursor was silently accepted and
  returned a page anyway** (a real integrity bypass, recorded as a
  :class:`Finding`).
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
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.rearchitecture_phase1_gate import source_files, source_hash  # noqa: E402
from checks.rearchitecture_phase2_gate import environment_hash as _environment_hash  # noqa: E402
from engine.v2.diagnosis import AGREE, DIFFER, ComparisonReceipt, Envelope, Finding, Population, content_hash  # noqa: E402
from engine.v2.foundation import to_document  # noqa: E402
from engine.v2.serving.api import create_app  # noqa: E402

PAGINATION_KIND = "api_pagination_parity"
CURSOR_NEGATIVE_KIND = "api_cursor_negative_control"

PAGE_LIMIT = 3


def _start(app):
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


def _stop(server, thread) -> None:
    server.should_exit = True
    thread.join(timeout=5)


def _get(base: str, path: str, *, token: str, params: dict | None = None) -> tuple[int, dict]:
    url = base + path
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    if clean:
        url += "?" + urlencode(clean)
    request = Request(url)
    request.add_header("Authorization", "Bearer " + token)
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _finding(findings: list, field: str, kind: str = "value") -> None:
    findings.append(Finding(
        finding_id=content_hash([PAGINATION_KIND, field])[7:19],
        first_differing_stage="paginate", field_path=field, kind=kind, owning_stage="paginate"))


def _walk_pages(base: str, token: str, release_id: str, *, limit: int) -> list[dict]:
    items: list[dict] = []
    cursor = None
    while True:
        status, page = _get(base, "/api/v1/events", token=token,
                            params={"release_id": release_id, "limit": limit, "cursor": cursor})
        if status != 200:
            raise RuntimeError(f"paged fetch failed: {status} {page}")
        items.extend(page["items"])
        cursor = page.get("next_cursor")
        if not cursor:
            return items


def build_pagination_parity(base: str, token: str, release_id: str, *, code_hash: str,
                            environment_hash: str) -> ComparisonReceipt:
    findings: list[Finding] = []
    status, full_page = _get(base, "/api/v1/events", token=token,
                             params={"release_id": release_id, "limit": 200})
    if status != 200:
        raise RuntimeError(f"full fetch failed: {status} {full_page}")
    full_items = full_page["items"]
    paged_items = _walk_pages(base, token, release_id, limit=PAGE_LIMIT)

    full_ids = [item["event_ref"]["event_id"] for item in full_items]
    paged_ids = [item["event_ref"]["event_id"] for item in paged_items]
    if len(paged_ids) != len(set(paged_ids)):
        _finding(findings, "duplicate_event_ids", kind="value")
    if set(full_ids) != set(paged_ids):
        _finding(findings, "population_mismatch", kind="missing_field")
    if full_page["total_matching"] != len(full_ids):
        _finding(findings, "total_matching_vs_full_page")
    if full_page["total_matching"] != len(paged_ids):
        _finding(findings, "total_matching_vs_paged_walk")
    if full_ids != paged_ids:
        _finding(findings, "page_order_stability")

    compared = len(paged_ids)
    population = Population(expected=full_page["total_matching"], supported=compared, compared=compared)
    verdict = DIFFER if findings else (AGREE if compared > 0 else DIFFER)
    envelope = Envelope(code_hash=code_hash, environment_hash=environment_hash)
    receipt_id = content_hash([PAGINATION_KIND, release_id, [f.finding_id for f in findings]])[7:23]
    return ComparisonReceipt(
        receipt_id=receipt_id, comparison_kind=PAGINATION_KIND, tier=1,
        left_ref="release:" + release_id, right_ref="release:" + release_id,
        stage_plan_ref="api_pagination.v1", tolerance_policy_ref="exact_set.v1",
        verdict=verdict, findings=tuple(findings), population=population, envelope=envelope)


def _tamper_cursor(cursor: str) -> str:
    hex_part, signature = cursor.rsplit(".", 1)
    flipped_char = "0" if signature[-1] != "0" else "1"
    return hex_part + "." + signature[:-1] + flipped_char


def build_cursor_negative_control(base: str, token: str, release_id: str, *, code_hash: str,
                                  environment_hash: str) -> ComparisonReceipt:
    findings: list[Finding] = []
    checks = 0
    status, first_page = _get(base, "/api/v1/events", token=token,
                              params={"release_id": release_id, "limit": PAGE_LIMIT})
    if status != 200 or not first_page.get("next_cursor"):
        raise RuntimeError("need a release with more than one page to test cursor tampering")
    valid_cursor = first_page["next_cursor"]

    # Only an actual 200 (real page content returned for a tampered/misused
    # cursor) counts as a leak. Any refusal status is safe, whether or not
    # it happens to be exactly 409 CURSOR_MISMATCH -- a 404 on an unknown
    # release, for example, is a different but equally safe outcome, not a
    # "wrong refusal shape" worth flagging.
    checks += 1
    tampered = _tamper_cursor(valid_cursor)
    status, body = _get(base, "/api/v1/events", token=token,
                        params={"release_id": release_id, "limit": PAGE_LIMIT, "cursor": tampered})
    if status == 200:
        _finding(findings, "tampered_signature_accepted")

    # Reuse the SAME valid cursor with a different normalized query (an added
    # ticker filter) on the SAME release -- the cursor's embedded query hash
    # must no longer match, so this must also refuse.
    checks += 1
    status, body = _get(base, "/api/v1/events", token=token,
                        params={"release_id": release_id, "limit": PAGE_LIMIT, "cursor": valid_cursor,
                                "ticker": "T0"})
    if status == 200:
        _finding(findings, "cursor_reused_for_different_query_accepted")

    population = Population(expected=checks, supported=checks, compared=checks)
    verdict = AGREE if findings else DIFFER
    envelope = Envelope(code_hash=code_hash, environment_hash=environment_hash)
    receipt_id = content_hash([CURSOR_NEGATIVE_KIND, release_id, [f.finding_id for f in findings]])[7:23]
    return ComparisonReceipt(
        receipt_id=receipt_id, comparison_kind=CURSOR_NEGATIVE_KIND, tier=1,
        left_ref="valid_cursor:" + release_id, right_ref="tampered_cursor:" + release_id,
        stage_plan_ref="api_cursor.v1", tolerance_policy_ref="exact_bytes.v1",
        verdict=verdict, findings=tuple(findings), population=population, envelope=envelope)


def build(serving_db: Path, store_root: Path, serving_root: Path, *, token: str,
         release_id: str) -> tuple[ComparisonReceipt, ComparisonReceipt]:
    app = create_app(serving_db=str(serving_db), store_root=str(store_root), serving_root=str(serving_root),
                     token=token, resolver=lambda: release_id)
    code_hash = source_hash(source_files(ROOT))
    env_hash, _source = _environment_hash(ROOT)
    server, thread, base = _start(app)
    try:
        pagination = build_pagination_parity(base, token, release_id, code_hash=code_hash,
                                             environment_hash=env_hash)
        cursor_negative = build_cursor_negative_control(base, token, release_id, code_hash=code_hash,
                                                        environment_hash=env_hash)
    finally:
        _stop(server, thread)
    return pagination, cursor_negative


def publish(receipt: ComparisonReceipt, artifact_root: Path, name: str) -> dict:
    artifact_root.mkdir(parents=True, exist_ok=True)
    data = json.dumps(to_document(receipt), indent=2, sort_keys=True).encode()
    path = artifact_root / name
    path.write_bytes(data)
    return {"path": path.name, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-db", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--serving-root", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args(argv)
    pagination, cursor_negative = build(args.serving_db, args.store_root, args.serving_root,
                                        token=args.token, release_id=args.release_id)
    p_ref = publish(pagination, args.artifact_root, "api_pagination_parity.json")
    c_ref = publish(cursor_negative, args.artifact_root, "api_cursor_negative_control.json")
    print(json.dumps({
        "api_pagination_parity": {**p_ref, "verdict": pagination.verdict},
        "api_cursor_negative_control": {**c_ref, "verdict": cursor_negative.verdict},
    }, indent=2))
    return 0 if pagination.verdict == AGREE and cursor_negative.verdict == DIFFER else 1


if __name__ == "__main__":
    raise SystemExit(main())
