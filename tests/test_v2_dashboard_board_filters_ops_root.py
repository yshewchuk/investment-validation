"""Phase 6 slice 8: the new board-list filters are reachable through the real
serving-API launcher.

`tests/test_v2_dashboard_preview_ops_root.py` drives a real
``engine.v2.dashboard.preview.run`` over an ephemeral loopback port to prove
argv-to-route reachability through a real server composition.
``/api/v1/events`` is not mounted by that compatibility shell
(``engine/v2/serving/operations.py`` serves the shell, release and action
routes only); its real argv/launcher is ``python3 -m engine.v2.serving.api``.
So this module composes the same way one layer down: a real ``build_candidate``
serving root, the real ``engine.v2.serving.api`` launcher as a subprocess over
argv, and one real ``GET /api/v1/events`` request per new filter -- proving
``gate``/``out_of_domain``/``disabled`` reach ``projections.list_events``
through the launcher, not only through ``create_app`` directly (that unit
level is ``tests/test_v2_serving_api.py``).
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore  # noqa: E402
from engine.v2.serving import projections  # noqa: E402
from tests.test_v2_serving_projections import (  # noqa: E402
    _bundle,
    _compact,
    _event_row,
    _events_snapshot,
    _preview_input,
    _row,
    _score_doc,
)

TOKEN = "board-filter-launcher-secret"


def _committed_release(tmp_path):
    """A real serving root holding one committed candidate: gate-pass,
    gate-fail and OUT_OF_DOMAIN-flagged rows, matching the board fixture
    shapes ``tests/test_v2_serving_projections.py`` already builds."""
    (tmp_path / "phase2").mkdir()
    conn, store, snap = _events_snapshot(
        tmp_path / "phase2",
        [_event_row(f"e{i}", f"T{i}", datetime(2024, 1, i + 1)) for i in range(3)])
    repo = Repository(conn, store)
    serving_root = tmp_path / "serving"
    serving_root.mkdir()
    serving_store = ArtifactStore(serving_root / "objects")
    serving_conn = projections.connect(str(serving_root / "serving.sqlite"))
    rows = [
        _row(ticker="T0", event_date="2024-01-01", strike=100.0),
        _row(ticker="T1", event_date="2024-01-02", strike=101.0, flags=["OUT_OF_DOMAIN"]),
        _row(ticker="T2", event_date="2024-01-03", strike=102.0, gate_pass=False,
             exp_pnl_model=None, win_model=None, detail="gate score below threshold"),
    ]
    release = projections.build_candidate(
        _preview_input(), _score_doc(rows=rows), _bundle(*[_compact(r) for r in rows]),
        repository=repo, snapshot_ref=snap, store=serving_store, conn=serving_conn,
        requested_as_of="2024-01-03", resolved_as_of="2024-01-03")
    serving_conn.close()
    return serving_root, release


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _get(base: str, path: str, *, params: dict | None = None) -> tuple[int, dict]:
    url = base + path
    clean = {k: v for k, v in (params or {}).items() if v is not None}
    if clean:
        url += "?" + urlencode(clean)
    request = Request(url)
    request.add_header("Authorization", "Bearer " + TOKEN)
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except HTTPError as error:
        return error.code, json.loads(error.read())


def _wait_ready(base: str, process, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            status, _ = _get(base, "/api/v1/events")
        except (URLError, OSError):
            time.sleep(0.05)
            continue
        if status is not None:
            return True
    return False


def _terminate(process) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _start_launcher(serving_root):
    env = dict(os.environ, V2_DASHBOARD_TOKEN=TOKEN)
    last_error = ""
    for _ in range(3):
        port = _free_port()
        log_path = serving_root / f"launcher-{port}.log"
        with open(log_path, "wb") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "engine.v2.serving.api", "--host", "127.0.0.1",
                 "--port", str(port), "--serving-db", str(serving_root / "serving.sqlite"),
                 "--store-root", str(serving_root / "objects"), "--serving-root", str(serving_root)],
                cwd=str(ROOT), env=env, stdout=log, stderr=log)
        base = f"http://127.0.0.1:{port}"
        if _wait_ready(base, process):
            return process, base
        alive = process.poll() is None
        _terminate(process)
        detail = log_path.read_text(errors="replace")[-2000:]
        last_error = ("still alive after wait" if alive else f"rc={process.returncode}") + f" log={detail}"
    raise RuntimeError(f"serving API launcher did not start in time; last {last_error}")


@pytest.fixture
def launcher(tmp_path):
    serving_root, release = _committed_release(tmp_path)
    process, base = _start_launcher(serving_root)
    try:
        yield base, release
    finally:
        _terminate(process)


def test_gate_filter_is_reachable_through_the_real_launcher(launcher):
    base, release = launcher
    status, page = _get(base, "/api/v1/events",
                        params={"release_id": release.release_id, "limit": "10", "gate": "pass"})
    assert status == 200
    assert page["total_matching"] == 1  # e1 is gate-pass but OUT_OF_DOMAIN, hidden by default
    assert [item["event_ref"]["event_id"] for item in page["items"]] == ["e0"]
    assert page["items"][0]["scores"][0]["verdict"] == "true"

    status, page = _get(base, "/api/v1/events",
                        params={"release_id": release.release_id, "limit": "10", "gate": "fail"})
    assert status == 200
    assert page["total_matching"] == 1
    assert [item["event_ref"]["event_id"] for item in page["items"]] == ["e2"]
    assert page["items"][0]["scores"][0]["refusal_reason"] == "gate score below threshold"

    status, body = _get(base, "/api/v1/events",
                        params={"release_id": release.release_id, "gate": "bogus"})
    assert status == 422
    assert body["code"] == "INVALID_REQUEST"


def test_out_of_domain_and_disabled_toggles_are_reachable_through_the_real_launcher(launcher):
    base, release = launcher
    status, page = _get(base, "/api/v1/events",
                        params={"release_id": release.release_id, "limit": "10"})
    assert status == 200
    assert page["total_matching"] == 2
    assert [item["event_ref"]["event_id"] for item in page["items"]] == ["e0", "e2"]

    status, page = _get(base, "/api/v1/events",
                        params={"release_id": release.release_id, "limit": "10",
                                "out_of_domain": "true"})
    assert status == 200
    assert page["total_matching"] == 3
    assert [item["event_ref"]["event_id"] for item in page["items"]] == ["e0", "e1", "e2"]
    assert page["items"][1]["scores"][0]["flags"] == ["OUT_OF_DOMAIN"]

    status, page = _get(base, "/api/v1/events",
                        params={"release_id": release.release_id, "limit": "10",
                                "disabled": "true"})
    assert status == 200
    assert page["total_matching"] == 2  # no UNVALIDATED_STRUCTURE row in this fixture
