"""Ops-root native-parity discovery for the read API (serving P3-2).

A real ops catalog at the production ``<ops_root>/catalog.sqlite`` path
(``open_catalog``) and its committed objects under ``ArtifactStore(ops_root)``
back the discovery; the report bytes are built by the REAL production
comparison functions (``compare_native_vs_legacy`` -> ``_stamp_report_identity``)
and served over REAL HTTP (the ``_start``/``_stop``/``_get`` harness from
``tests/test_v2_serving_api.py``). Only the nightly producer's own job/attempt/
output rows are written directly, mirroring
``tests/test_v2_ops_nightly_native_parity_helpers.py``.
"""
from __future__ import annotations

import contextlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.foundation import ArtifactStore, format_timestamp
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.native_parity_report import (
    SCHEMA_VERSION,
    _stamp_report_identity,
    compare_native_vs_legacy,
    write_parity_report,
)
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor
from engine.v2.ops.submission import job_id_for
from engine.v2.parity.dimensions import FORECAST_FIELDS, SIMULATION_FIELDS
from engine.v2.serving.api import create_app
from tests.ops_support import FakeClock
from tests.test_v2_serving_api import _get, _start, _stop

TOKEN = "test-token-ops-root-5b0e"
AS_OF = "2026-01-02"
_FIELDS = (*FORECAST_FIELDS, *SIMULATION_FIELDS)
_PARITY_ROUTES = ("/api/v1/native_parity", "/api/v1/native_parity/mismatches",
                  "/api/v1/native_parity/unpaired")


class _FixedClock:
    def now(self):
        return datetime(2026, 9, 12, tzinfo=timezone.utc)


def _row():
    return {name: 0.0 for name in _FIELDS}


def _report(row_count: int, *, as_of: str = AS_OF) -> dict:
    legacy = {f"K{i}|S|2026-01-{i + 1:02d}": _row() for i in range(row_count)}
    native = {f"K{i}|S|2026-01-{i + 1:02d}": _row() for i in range(row_count)}
    report = compare_native_vs_legacy(legacy, native, ("forecasts", "simulation"))
    return _stamp_report_identity(report, as_of=as_of, clock=_FixedClock())


def _payload(row_count: int, *, as_of: str = AS_OF) -> bytes:
    return json.dumps(_report(row_count, as_of=as_of)).encode("utf-8")


def _key(as_of: str, scope_hash: str = "H1") -> str:
    return "nightly:" + as_of + ":" + scope_hash + ":native_parity"


def _ops_at(root):
    clock = FakeClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    epoch_id = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
    return SimpleNamespace(root=root, conn=conn, clock=clock,
                           supervisor=Supervisor(epoch_id, "boot"),
                           store=ArtifactStore(root))


@pytest.fixture
def ops(tmp_path):
    ops = _ops_at(tmp_path)
    try:
        yield ops
    finally:
        ops.conn.close()


def _insert_job(ops, *, as_of, scope_hash="H1", state="succeeded", stamp=None):
    job_id = job_id_for("shadow", _key(as_of, scope_hash))
    ts = stamp or format_timestamp(ops.clock.now())
    ops.conn.execute(
        "INSERT INTO jobs (job_id, namespace, idempotency_key, request_digest, principal, "
        "kind, spec_json, resource_class, checkpoint_contract_ref, retry_json, state, "
        "priority, max_attempts, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?)",
        (job_id, "shadow", _key(as_of, scope_hash), "digest", "operator", "native_parity",
         "{}", "validation", SCHEMA_VERSION, "{}", state, 0, 1, ts, ts))
    ops.conn.commit()
    return job_id


def _publish_report(ops, job_id, payload, *, stamp=None):
    attempt_id = "attempt-" + job_id
    ts = stamp or format_timestamp(ops.clock.now())
    lease = format_timestamp(ops.clock.now() + timedelta(seconds=1))
    ops.conn.execute(
        "INSERT INTO attempts (attempt_id, job_id, attempt_number, fence, supervisor_epoch, "
        "host_boot_id, state, process_state, resources_json, created_at, lease_expires_at, "
        "ended_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'verified_dead', ?, ?, ?, ?)",
        (attempt_id, job_id, 1, 1, ops.supervisor.epoch_id, ops.supervisor.boot_id,
         "succeeded", "{}", ts, lease, ts))
    ref = ops.store.publish_bytes(payload, schema_ref=SCHEMA_VERSION)
    register_artifact(ops.conn, ref, attempt_id, ops.clock)
    ops.conn.execute("INSERT INTO attempt_outputs VALUES (?, ?, ?)",
                     (attempt_id, "report", ref.artifact_id))
    ops.conn.commit()
    return ref


def _seed(ops, *, as_of, row_count=1, scope_hash="H1", state="succeeded", stamp=None):
    job_id = _insert_job(ops, as_of=as_of, scope_hash=scope_hash, state=state, stamp=stamp)
    if state == "succeeded":
        _publish_report(ops, job_id, _payload(row_count, as_of=as_of), stamp=stamp)
    return job_id


def _app(ops, *, with_ops_root=True, report_path=None):
    return create_app(
        serving_db=str(ops.root / "serving.sqlite"),
        store_root=str(ops.root / "serving-objects"),
        serving_root=str(ops.root / "serving"),
        token=TOKEN,
        ops_root=str(ops.root) if with_ops_root else None,
        native_parity_report_path=None if report_path is None else str(report_path))


@contextlib.contextmanager
def _running(ops, **kwargs):
    server, thread, base = _start(_app(ops, **kwargs))
    try:
        yield base
    finally:
        _stop(server, thread)


def _summary(ops, **kwargs):
    with _running(ops, **kwargs) as base:
        code, body, headers = _get(base, "/api/v1/native_parity", token=TOKEN)
    return code, headers, json.loads(body)


def test_ops_root_selects_newest_succeeded_shadow_report(ops):
    """As-of session descending, then commit time descending; a newer failed
    job and an older report are both ignored -- and the catalog is untouched."""
    catalog_path = ops.root / "catalog.sqlite"
    _seed(ops, as_of="2026-01-01", row_count=1)
    ops.clock.advance(60)
    _seed(ops, as_of="2026-01-02", row_count=2, scope_hash="HB", stamp=format_timestamp(ops.clock.now()))
    ops.clock.advance(60)
    _seed(ops, as_of="2026-01-02", row_count=4, scope_hash="HC", stamp=format_timestamp(ops.clock.now()))
    ops.clock.advance(60)
    _seed(ops, as_of="2026-01-03", state="failed")

    before = catalog_path.read_bytes()
    with _running(ops) as base:
        code, body, headers = _get(base, "/api/v1/native_parity", token=TOKEN)
        assert code == 200
        assert headers.get("Cache-Control") == "no-store"
        summary = json.loads(body)
        assert summary["status"] == "available"
        assert summary["as_of"] == "2026-01-02"
        assert summary["generated_at"] == "2026-09-12T00:00:00.000000Z"
        assert summary["compared_count"] == 4

        for path in _PARITY_ROUTES[1:]:
            params = {"side": "legacy"} if path.endswith("unpaired") else None
            code, body, headers = _get(base, path, token=TOKEN, params=params)
            assert code == 200, path
            assert json.loads(body)["status"] == "available", path

    assert catalog_path.read_bytes() == before


def test_ops_root_path_with_uri_characters_resolves_report(tmp_path):
    """A ``?``, ``#``, or literal ``%``, non-ASCII text, and a surrogateescape
    byte in the ops-root directory are all filename characters -- not URI
    query/fragment syntax, a stray percent-escape, or a decode failure: the real
    catalog still opens read-only and the committed succeeded shadow report
    resolves over the real API path."""
    root = tmp_path / "ops?root#a%b-\u00e9\udcff"
    root.mkdir()
    ops = _ops_at(root)
    try:
        _seed(ops, as_of="2026-01-02", row_count=3)
        code, headers, summary = _summary(ops)
    finally:
        ops.conn.close()
    assert code == 200
    assert headers.get("Cache-Control") == "no-store"
    assert summary["status"] == "available"
    assert summary["as_of"] == "2026-01-02"
    assert summary["compared_count"] == 3


def test_ops_root_without_succeeded_job_is_typed_no_report(ops):
    _seed(ops, as_of="2026-01-02", state="failed")

    with _running(ops) as base:
        for path in _PARITY_ROUTES:
            params = {"side": "legacy"} if path.endswith("unpaired") else None
            code, body, headers = _get(base, path, token=TOKEN, params=params)
            assert code == 200, path
            assert headers.get("Cache-Control") == "no-store"
            document = json.loads(body)
            assert document["status"] == "no_report", path
            assert document["reason_code"] == "NATIVE_PARITY_JOB_NOT_FOUND", path


def test_explicit_report_path_wins_over_ops_root(ops, tmp_path):
    _seed(ops, as_of="2026-01-02", row_count=1)
    explicit = write_parity_report(_report(3, as_of="2026-12-31"), tmp_path / "explicit.json")

    code, _headers, summary = _summary(ops, report_path=explicit)
    assert code == 200
    assert summary["status"] == "available"
    assert summary["as_of"] == "2026-12-31"
    assert summary["compared_count"] == 3


def test_no_ops_root_preserves_untyped_no_report(ops):
    for path in _PARITY_ROUTES:
        params = {"side": "legacy"} if path.endswith("unpaired") else None
        with _running(ops, with_ops_root=False) as base:
            code, body, headers = _get(base, path, token=TOKEN, params=params)
        assert code == 200, path
        assert headers.get("Cache-Control") == "no-store"
        document = json.loads(body)
        assert document["status"] == "no_report", path
        assert "reason_code" not in document, path


def _assert_malformed(ops, **kwargs):
    with _running(ops, **kwargs) as base:
        for path in _PARITY_ROUTES:
            params = {"side": "legacy"} if path.endswith("unpaired") else None
            code, body, headers = _get(base, path, token=TOKEN, params=params)
            assert code == 503, path
            assert headers.get("Cache-Control") == "no-store"
            problem = json.loads(body)
            assert problem["code"] == "NATIVE_PARITY_REPORT_MALFORMED", path
            assert "compared_count" not in problem, path
            assert "items" not in problem, path


def test_selected_missing_output_refuses_without_older_fallback(ops):
    _seed(ops, as_of="2026-01-01", row_count=3)
    ops.clock.advance(60)
    _insert_job(ops, as_of="2026-01-02")  # succeeded, but committed no report output

    _assert_malformed(ops)


def test_selected_schema_invalid_report_refuses_without_older_fallback(ops):
    _seed(ops, as_of="2026-01-01", row_count=3)
    ops.clock.advance(60)
    job_id = _insert_job(ops, as_of="2026-01-02")
    _publish_report(ops, job_id, b"{}")  # valid JSON, but no report schema

    _assert_malformed(ops)


def test_selected_malformed_json_report_refuses_without_older_fallback(ops):
    _seed(ops, as_of="2026-01-01", row_count=3)
    ops.clock.advance(60)
    job_id = _insert_job(ops, as_of="2026-01-02")
    _publish_report(ops, job_id, b"{not json")

    _assert_malformed(ops)


def test_selected_unverifiable_output_refuses_without_older_fallback(ops):
    _seed(ops, as_of="2026-01-01", row_count=3)
    ops.clock.advance(60)
    job_id = _insert_job(ops, as_of="2026-01-02")
    ref = _publish_report(ops, job_id, _payload(1))
    object_path = ops.store.root / ref.storage_key
    payload = object_path.read_bytes()
    object_path.chmod(object_path.stat().st_mode | 0o200)
    object_path.write_bytes(b"!" + payload[1:])

    _assert_malformed(ops)


def test_selected_deeply_nested_ref_json_refuses_without_older_fallback(ops):
    """A stored reference that is valid JSON text but too deeply nested for
    ``json.loads`` to parse (``RecursionError``) is the selected job's own
    broken output: every parity route refuses with the typed 503 malformed
    refusal -- no untyped 500, and no fallback to the older valid report."""
    _seed(ops, as_of="2026-01-01", row_count=3)
    ops.clock.advance(60)
    job_id = _insert_job(ops, as_of="2026-01-02")
    ref = _publish_report(ops, job_id, _payload(1))
    ops.conn.execute("UPDATE artifacts SET ref_json = ? WHERE artifact_id = ?",
                     ("[" * 100000 + "]" * 100000, ref.artifact_id))
    ops.conn.commit()

    _assert_malformed(ops)
