"""Phase 6 P6-4: the launcher's ``--ops-root`` makes ``POST /actions/refresh`` reachable.

Drives a real ``preview.run`` on a loopback ephemeral port against an ops root
that already holds a published nightly plan, then POSTs ``/actions/refresh``
back over HTTP. That single boundary covers argv parsing,
``_server.build_server``'s ``submit_refresh`` callback,
``engine.v2.ops.cli.refresh_action`` and the operations route together -- the
runtime reachability the static inventory could not see. Shadow submission
only: nothing here runs the nightly, serves production reads from ops, or
touches production authority. Refresh/what-if remain the only wired actions;
``--ops-root`` is never inferred from ``--release-root``.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from engine.v2.dashboard import preview
from engine.v2.foundation import SystemClock
from engine.v2.ops import cli
from engine.v2.ops.bootstrap import open_catalog

REPO = Path(__file__).resolve().parents[1]
TOKEN = "refresh-launcher-secret"
SESSION = "2026-09-10"
_NIGHTLY_JOB_COUNT = 14  # the legacy-mode DAG (P6-2 added "features")


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _dashboard_bundle(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    release_dir = bundle / "releases" / "r1"
    release_dir.mkdir(parents=True)
    (release_dir / "index.html").write_text("<!doctype html><title>legacy</title>")
    (bundle / "CURRENT").write_text("r1\n")
    health = tmp_path / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0"}))
    return bundle, health


def _nightly_manifest_dict() -> dict:
    from engine.v2.data.legacy_nightly_read_plan import NIGHTLY_CAPTURE_IMPLEMENTATION_REF

    tables = ("daily_market", "option_chains", "earnings_events", "trades")
    file_refs = [{"path": f"data/curated/{table}/year=2024/part-0000.parquet",
                  "content_hash": "sha256:" + "0" * 64, "byte_size": 1} for table in tables]
    file_refs.append({"path": "data/raw/fetch/orats/ab/placeholder.meta.json",
                      "content_hash": "sha256:" + "0" * 64, "byte_size": 1})
    return {"manifest_id": "m1", "note": "", "file_refs": file_refs, "table_contract_refs": [],
            "registry_and_model_refs": ["placeholder::sha256:" + "0" * 64],
            "calendar_ref": "placeholder::sha256:" + "0" * 64, "selected_session": SESSION,
            "finality_receipt_refs": [], "knowledge_mode_by_table": {},
            "availability_evidence_refs": [], "read_set_complete": True,
            "capture_implementation_ref": NIGHTLY_CAPTURE_IMPLEMENTATION_REF}


def _published_nightly_plan(tmp_path: Path, capsys) -> tuple[Path, str]:
    """An ops root holding one already-published, unblocked nightly plan.

    Uses the real ``ops plan nightly`` CLI -- the same path the desk's CLI
    documented under UD-1 -- so the plan artifact is registered exactly as a
    production refresh submission would reference it.
    """
    ops_root = tmp_path / "ops"
    population = tmp_path / "population.json"
    population.write_text(json.dumps(["FAKE|TWIN-P|" + SESSION]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_nightly_manifest_dict()))
    assert cli.main(["--root", str(ops_root), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(ops_root), "plan", "nightly", "--as-of", SESSION,
                     "--tickers", "FAKE", "--input-manifest", str(manifest),
                     "--expected-population", str(population)]) == 0
    plan_ref = json.loads(capsys.readouterr().out)["plan_ref"]
    return ops_root, plan_ref


# --------------------------------------------------------------------------
# launcher / HTTP helpers
# --------------------------------------------------------------------------


def _run_launcher(tmp_path, monkeypatch, *, ops_root=None, calibration_health_path=None):
    monkeypatch.setenv(preview.TOKEN_ENV_VAR, TOKEN)
    bundle, health = _dashboard_bundle(tmp_path)
    argv = ["--host", "127.0.0.1", "--port", "0",
            "--release-root", str(bundle), "--health-path", str(health)]
    if ops_root is not None:
        argv += ["--ops-root", str(ops_root)]
    if calibration_health_path is not None:
        argv += ["--calibration-health-path", str(calibration_health_path)]
    return preview.run(argv)


def _get_calibration_health(server, *, token=TOKEN):
    request = Request(f"http://127.0.0.1:{server.server_port}/calibration-health.json")
    if token is not None:
        request.add_header("Authorization", "Bearer " + token)
    return urlopen(request, timeout=5)


def _post_refresh(server, body, *, token=TOKEN):
    request = Request(f"http://127.0.0.1:{server.server_port}/actions/refresh",
                      data=json.dumps(body).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    if token is not None:
        request.add_header("Authorization", "Bearer " + token)
    return urlopen(request, timeout=5)


def _job_count(ops_root: Path) -> int:
    conn = open_catalog(ops_root / "catalog.sqlite", clock=SystemClock())
    try:
        return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    finally:
        conn.close()


def _stop(server, thread) -> None:
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


# --------------------------------------------------------------------------
# acceptance
# --------------------------------------------------------------------------


def test_ops_root_refresh_returns_202_jobs_and_repeat_is_idempotent(tmp_path, monkeypatch, capsys):
    ops_root, plan_ref = _published_nightly_plan(tmp_path, capsys)
    assert _job_count(ops_root) == 0  # planning publishes no jobs
    server, thread, release_id = _run_launcher(tmp_path, monkeypatch, ops_root=ops_root)
    try:
        assert release_id == "r1"  # the read-only launcher still starts and pins normally
        first = _post_refresh(server, {"plan_ref": plan_ref})
        assert first.status == 202
        body = json.loads(first.read())
        job_ids = [job["job_id"] for job in body["jobs"]]
        assert len(job_ids) == _NIGHTLY_JOB_COUNT
        assert TOKEN not in json.dumps(body)

        second = _post_refresh(server, {"plan_ref": plan_ref})
        assert second.status == 202
        repeat = [job["job_id"] for job in json.loads(second.read())["jobs"]]
        assert repeat == job_ids  # same plan_ref -> identical jobs, no new rows
        assert _job_count(ops_root) == _NIGHTLY_JOB_COUNT
    finally:
        _stop(server, thread)


def test_missing_or_wrong_token_is_401_and_creates_no_job(tmp_path, monkeypatch, capsys):
    ops_root, plan_ref = _published_nightly_plan(tmp_path, capsys)
    server, thread, _ = _run_launcher(tmp_path, monkeypatch, ops_root=ops_root)
    try:
        for bad in (None, "wrong-token"):
            with pytest.raises(HTTPError) as error:
                _post_refresh(server, {"plan_ref": plan_ref}, token=bad)
            assert error.value.code == 401
        assert _job_count(ops_root) == 0  # an unauthorised POST never reaches the callback
    finally:
        _stop(server, thread)


def test_malformed_plan_ref_is_a_safe_400_and_creates_no_job(tmp_path, monkeypatch, capsys):
    ops_root, _ = _published_nightly_plan(tmp_path, capsys)
    server, thread, _ = _run_launcher(tmp_path, monkeypatch, ops_root=ops_root)
    try:
        for payload in ({"plan_ref": 123}, {}, {"plan_ref": ""}):
            with pytest.raises(HTTPError) as error:
                _post_refresh(server, payload)
            assert error.value.code == 400
        assert _job_count(ops_root) == 0
    finally:
        _stop(server, thread)


def test_without_ops_root_valid_plan_ref_still_503_and_is_not_guessed(tmp_path, monkeypatch, capsys):
    # An ops root holding a publishable plan, but launched WITHOUT --ops-root:
    # the route must keep its read-only 503 and must never guess a root from
    # --release-root (the bundle store is a different tree entirely).
    ops_root, plan_ref = _published_nightly_plan(tmp_path, capsys)
    server, thread, release_id = _run_launcher(tmp_path, monkeypatch)  # no ops root
    try:
        assert release_id == "r1"
        with pytest.raises(HTTPError) as error:
            _post_refresh(server, {"plan_ref": plan_ref})
        assert error.value.code == 503
        assert _job_count(ops_root) == 0
    finally:
        _stop(server, thread)


def test_calibration_health_reachable_via_preview_run(tmp_path, monkeypatch):
    # A ledger_health.v1-shaped export (no schema_version, exactly what
    # engine.v2.ledger.calibration.export_health_file writes): the route must
    # serve the file's own bytes, canonicalized the way /health.json does.
    payload = {"generated_at": "2026-09-19T00:00:00+00:00", "n_scored": 3,
               "per_strategy": {"STR-THRU": {"available": True}}}
    expected = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    export = tmp_path / "calibration_health.json"
    export.write_bytes(expected)

    server, thread, release_id = _run_launcher(
        tmp_path, monkeypatch, calibration_health_path=export)
    try:
        assert release_id == "r1"
        response = _get_calibration_health(server)
        assert response.status == 200
        assert response.read() == expected
    finally:
        _stop(server, thread)


def test_calibration_health_503_when_not_configured(tmp_path, monkeypatch):
    # Omitting --calibration-health-path must not guess a path from
    # --health-path or --release-root: the route keeps its explicit 503 and
    # must never answer 200 with stale or empty content.
    server, thread, release_id = _run_launcher(tmp_path, monkeypatch)
    try:
        assert release_id == "r1"
        with pytest.raises(HTTPError) as error:
            _get_calibration_health(server)
        assert error.value.code == 503
    finally:
        _stop(server, thread)
