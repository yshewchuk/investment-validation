"""Slice-2 HTTP tests for the native-parity read routes (P3-2 follow-on).

The report fixture is built by calling the REAL production functions in
``engine.v2.ops.native_parity_report`` (``compare_native_vs_legacy`` ->
``_stamp_report_identity`` -> ``write_parity_report``) against independent
legacy/native synthetic rows; this module never hand-writes a report literal.
Served over REAL HTTP (a real ``uvicorn.Server`` in a background thread),
reusing ``tests/test_v2_serving_api.py``'s own ``_start``/``_stop``/``_get``
helpers exactly like ``tests/test_v2_serving_publication_binding.py`` does.
"""
from __future__ import annotations

import builtins
import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import Problem  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore, content_hash  # noqa: E402
from engine.v2.ops.native_parity_report import (  # noqa: E402
    _stamp_report_identity,
    apply_native_refusals,
    compare_native_vs_legacy,
    write_parity_report,
)
from engine.v2.parity.dimensions import FORECAST_FIELDS, SIMULATION_FIELDS  # noqa: E402
from engine.v2.serving import api as api_module  # noqa: E402
from engine.v2.serving import native_parity_projection, projections  # noqa: E402
from engine.v2.serving.api import ApiError, create_app  # noqa: E402
from engine.v2.serving.native_parity_projection import (  # noqa: E402
    NATIVE_PARITY_REPORT_MALFORMED,
    native_parity_summary,
)
from tests.test_v2_serving_api import (  # noqa: E402
    _bundle,
    _compact,
    _event_row,
    _events_snapshot,
    _get,
    _preview_input,
    _rows_for,
    _score_doc,
    _start,
    _stop,
)

TOKEN = "test-token-parity-7d21"
AS_OF = "2026-01-02"
GENERATED_AT = "2026-09-12T00:00:00.000000Z"

_DEFAULT_VALUES = {name: 0.0 for name in FORECAST_FIELDS + SIMULATION_FIELDS}


def _row(**overrides):
    row = dict(_DEFAULT_VALUES)
    row.update(overrides)
    return row


def _source_rows():
    """3 paired keys (one agreeing, one forecasts mismatch, one simulation
    mismatch), 2 legacy-only, 2 native-only."""
    legacy = {
        "AAA|S|2026-01-01": _row(),
        "BBB|S|2026-01-02": _row(forecast_p10=1.0),
        "CCC|S|2026-01-03": _row(exp_pnl_sim=1.0),
        "DDD|S|2026-01-04": _row(),
        "EEE|S|2026-01-05": _row(),
    }
    native = {
        "AAA|S|2026-01-01": _row(),
        "BBB|S|2026-01-02": _row(forecast_p10=1.25),
        "CCC|S|2026-01-03": _row(exp_pnl_sim=1.5),
        "FFF|S|2026-01-06": _row(),
        "GGG|S|2026-01-07": _row(),
    }
    return legacy, native


class _FixedClock:
    def now(self):
        return datetime(2026, 9, 12, tzinfo=timezone.utc)


def _build_report():
    legacy, native = _source_rows()
    report = compare_native_vs_legacy(legacy, native, ("forecasts", "simulation"))
    return _stamp_report_identity(report, as_of=AS_OF, clock=_FixedClock())


@dataclass
class ParityApi:
    report_path: Path
    report: dict
    base: str
    token: str


@pytest.fixture
def parity(tmp_path):
    report = _build_report()
    report_path = write_parity_report(report, tmp_path / "report.json")
    app = create_app(serving_db=str(tmp_path / "serving.sqlite"),
                     store_root=str(tmp_path / "objects"),
                     serving_root=str(tmp_path / "serving"),
                     token=TOKEN, native_parity_report_path=str(report_path))
    server, thread, base = _start(app)
    try:
        yield ParityApi(report_path=report_path, report=report, base=base, token=TOKEN)
    finally:
        _stop(server, thread)


def _page(parity, path, params, expected):
    code, body, headers = _get(parity.base, path, token=parity.token, params=params)
    assert code == 200
    assert headers.get("Cache-Control") == "no-store"
    assert headers.get("ETag") is None
    document = json.loads(body)
    assert document["items"] == [expected]
    return document["next_cursor"]


def test_authenticated_summary_and_bounded_detail_pages(parity):
    code, body, headers = _get(parity.base, "/api/v1/native_parity", token=parity.token)
    assert code == 200
    summary = json.loads(body)
    assert summary["schema_version"] == "native_parity_summary.v1.0"
    assert summary["status"] == "available"
    assert summary["source_schema_version"] == "native_parity_report.v1.3"
    assert summary["as_of"] == AS_OF
    assert summary["generated_at"] == GENERATED_AT
    assert summary["tolerance_policy_id"] == "score_record.exact.v1"
    assert summary["compared_count"] == 3
    assert summary["matched_row_count"] == 1
    assert summary["mismatched_row_count"] == 2
    assert summary["only_legacy_count"] == 2
    assert summary["only_native_count"] == 2
    assert summary["native_refused_count"] == 0
    assert summary["native_refused_unmatched_count"] == 0
    assert summary["native_refused_reasons"] == {}
    assert summary["native_refused_tickers"] == []
    assert headers.get("Cache-Control") == "no-store"
    assert headers.get("ETag") is None

    mismatch_items = []
    cursor = None
    for _ in range(3):
        params = {"limit": "1"}
        if cursor:
            params["cursor"] = cursor
        code, body, headers = _get(parity.base, "/api/v1/native_parity/mismatches",
                                   token=parity.token, params=params)
        assert code == 200
        assert headers.get("Cache-Control") == "no-store"
        assert headers.get("ETag") is None
        document = json.loads(body)
        mismatch_items.extend(document["items"])
        cursor = document["next_cursor"]
        if cursor is None:
            break
    assert cursor is None
    assert [(item["row_key"], item["dimension"]) for item in mismatch_items] == [
        ("BBB|S|2026-01-02", "forecasts"), ("CCC|S|2026-01-03", "simulation")]
    forecasts, simulation = mismatch_items
    assert set(forecasts["fields"]) == set(FORECAST_FIELDS)
    assert forecasts["fields"]["forecast_p10"] == {
        "status": "differ", "legacy": 1.0, "native": 1.25}
    assert forecasts["fields"]["driver_prediction"] == {"status": "agree"}
    assert set(simulation["fields"]) == set(SIMULATION_FIELDS)
    assert simulation["fields"]["exp_pnl_sim"] == {
        "status": "differ", "legacy": 1.0, "native": 1.5}
    assert simulation["fields"]["win_sim"] == {"status": "agree"}

    for side, expected in (("legacy", ["DDD|S|2026-01-04", "EEE|S|2026-01-05"]),
                           ("native", ["FFF|S|2026-01-06", "GGG|S|2026-01-07"])):
        seen = []
        cursor = None
        for index in range(3):
            params = {"side": side, "limit": "1"}
            if cursor:
                params["cursor"] = cursor
            cursor = _page(parity, "/api/v1/native_parity/unpaired", params,
                           expected[index])
            seen.append(expected[index])
            if cursor is None:
                break
        assert cursor is None
        assert seen == expected


def test_summary_projects_keyed_refusal_ticker_and_reason(parity):
    report = apply_native_refusals(
        _build_report(), {"DDD|S|2026-01-04": "PRICE_HISTORY_NOT_AVAILABLE"})
    write_parity_report(report, parity.report_path)

    code, body, _ = _get(parity.base, "/api/v1/native_parity", token=parity.token)
    assert code == 200
    summary = json.loads(body)
    assert summary["native_refused_tickers"] == [
        {"ticker": "DDD", "reason": "PRICE_HISTORY_NOT_AVAILABLE"}]
    assert summary["native_refused_count"] == 1


def test_summary_projects_unmatched_refusal_ticker_and_count(parity):
    report = apply_native_refusals(
        _build_report(), {"HHH|S|2026-01-08": "PRICE_HISTORY_NOT_AVAILABLE"})
    assert report["native_refused_unmatched"] == [
        {"row_key": "HHH|S|2026-01-08", "refusal_code": "PRICE_HISTORY_NOT_AVAILABLE",
         "ticker": "HHH", "reason": "PRICE_HISTORY_NOT_AVAILABLE"}]
    write_parity_report(report, parity.report_path)

    code, body, _ = _get(parity.base, "/api/v1/native_parity", token=parity.token)
    assert code == 200
    summary = json.loads(body)
    assert summary["native_refused_tickers"] == [
        {"ticker": "HHH", "reason": "PRICE_HISTORY_NOT_AVAILABLE"}]
    assert summary["native_refused_unmatched_count"] == 1


def test_v12_compat_report_projects_row_key_prefix_and_refusal_code(parity):
    report = apply_native_refusals(
        _build_report(), {"DDD|S|2026-01-04": "PRICE_HISTORY_NOT_AVAILABLE"})
    report["schema_version"] = "native_parity_report.v1.2"
    entry = report["native_refused"][0]
    del entry["ticker"]
    del entry["reason"]
    parity.report_path.write_text(json.dumps(report))

    code, body, _ = _get(parity.base, "/api/v1/native_parity", token=parity.token)
    assert code == 200
    summary = json.loads(body)
    assert summary["native_refused_tickers"] == [
        {"ticker": "DDD", "reason": "PRICE_HISTORY_NOT_AVAILABLE"}]


_PARITY_ROUTES = ("/api/v1/native_parity", "/api/v1/native_parity/mismatches",
                  "/api/v1/native_parity/unpaired")


def _assert_no_store(headers):
    assert headers.get("Cache-Control") == "no-store"
    assert headers.get("ETag") is None


def _assert_no_report(base, token):
    for path in _PARITY_ROUTES:
        code, body, headers = _get(base, path, token=token)
        assert code == 200, path
        _assert_no_store(headers)
        document = json.loads(body)
        assert document["status"] == "no_report", path
        assert "items" not in document
        assert "compared_count" not in document


def test_no_report(parity, tmp_path):
    parity.report_path.unlink()
    _assert_no_report(parity.base, parity.token)

    app = create_app(serving_db=str(tmp_path / "unconfigured.sqlite"),
                     store_root=str(tmp_path / "unconfigured-objects"),
                     serving_root=str(tmp_path / "unconfigured-serving"),
                     token=TOKEN, native_parity_report_path=None)
    server, thread, base = _start(app)
    try:
        _assert_no_report(base, TOKEN)
    finally:
        _stop(server, thread)


_MALFORMED_CASES = ("not_json", "missing_identity_keys", "missing_both_identity_keys",
                    "malformed_as_of", "missing_value_pair")


def _corrupt_report(parity, case):
    if case == "not_json":
        parity.report_path.write_text("not json{")
        return
    document = json.loads(json.dumps(parity.report))
    if case == "missing_identity_keys":
        del document["as_of"]
    elif case == "missing_both_identity_keys":
        del document["as_of"]
        del document["generated_at"]
    elif case == "malformed_as_of":
        document["as_of"] = "2026-13-99"
    elif case == "missing_value_pair":
        mismatch = next(entry for entry in document["mismatches"]
                        if entry["finding_fields"])
        mismatch["values"][mismatch["finding_fields"][0]].pop("native")
    else:
        raise AssertionError(case)
    parity.report_path.write_text(json.dumps(document))


@pytest.mark.parametrize("case", _MALFORMED_CASES)
def test_unavailable(parity, case):
    _corrupt_report(parity, case)
    for path in _PARITY_ROUTES:
        code, body, headers = _get(parity.base, path, token=parity.token)
        assert code == 503, (case, path)
        _assert_no_store(headers)
        problem = json.loads(body)
        assert problem["code"] == "NATIVE_PARITY_REPORT_MALFORMED", (case, path)
        assert "items" not in problem
        assert "compared_count" not in problem
        text = body.decode("utf-8")
        assert str(parity.report_path) not in text
        assert "Traceback" not in text


def test_unknown_row_key(parity):
    unknown = "ZZZ|S|2099-12-31"
    for path in ("/api/v1/native_parity/mismatches", "/api/v1/native_parity/unpaired"):
        params = {"row_key": unknown}
        if path.endswith("unpaired"):
            params["side"] = "legacy"
        code, body, headers = _get(parity.base, path, token=parity.token, params=params)
        assert code == 404, path
        _assert_no_store(headers)
        assert json.loads(body)["code"] == "NATIVE_PARITY_ROW_NOT_FOUND", path

    known = "AAA|S|2026-01-01"
    for path, params in (("/api/v1/native_parity/mismatches", {"row_key": known}),
                         ("/api/v1/native_parity/unpaired",
                          {"side": "legacy", "row_key": known})):
        code, body, headers = _get(parity.base, path, token=parity.token, params=params)
        assert code == 200, path
        _assert_no_store(headers)
        document = json.loads(body)
        assert document["items"] == [], path
        assert document["next_cursor"] is None, path


_REFUSAL_CODE = "MODEL_NOT_READY"
_REFUSED_COLLECTIONS = (
    pytest.param("native_refused", "DDD|S|2026-01-04", id="native_refused"),
    pytest.param("native_refused_unmatched", "HHH|S|2026-01-08", id="native_refused_unmatched"),
)


@pytest.mark.parametrize("collection,refused_key", _REFUSED_COLLECTIONS)
def test_keyed_refusal_entries_are_known_row_keys(parity, collection, refused_key):
    """A keyed entry in either optional refusal collection is a report row:
    the writer moved it out of ``only_legacy`` (or native-only refused it),
    so it names no row in any other population -- both detail routes accept
    the key and serve an empty page, and 404 stays reserved for keys no
    population names."""
    legacy, native = _source_rows()
    report = compare_native_vs_legacy(legacy, native, ("forecasts", "simulation"))
    report = apply_native_refusals(report, {refused_key: _REFUSAL_CODE})
    report = _stamp_report_identity(report, as_of=AS_OF, clock=_FixedClock())
    assert report[collection] == [{"row_key": refused_key, "refusal_code": _REFUSAL_CODE,
                                   "ticker": refused_key.split("|", 1)[0],
                                   "reason": _REFUSAL_CODE}]
    other = ({"native_refused": "native_refused_unmatched",
              "native_refused_unmatched": "native_refused"}[collection])
    assert report[other] == [], collection
    for field in ("compared", "only_legacy", "only_native"):
        assert refused_key not in report[field], (collection, field)
    write_parity_report(report, parity.report_path)

    screens = (("/api/v1/native_parity/mismatches", {}),
               ("/api/v1/native_parity/unpaired", {"side": "legacy"}))
    for path, base_params in screens:
        code, body, headers = _get(parity.base, path, token=parity.token,
                                   params={**base_params, "row_key": refused_key})
        assert code == 200, (collection, path)
        _assert_no_store(headers)
        document = json.loads(body)
        assert document["items"] == [], (collection, path)
        assert document["next_cursor"] is None, (collection, path)

        code, body, headers = _get(parity.base, path, token=parity.token,
                                   params={**base_params, "row_key": "ZZZ|S|2099-12-31"})
        assert code == 404, (collection, path)
        _assert_no_store(headers)
        assert json.loads(body)["code"] == "NATIVE_PARITY_ROW_NOT_FOUND", (collection, path)


def test_auth_required(parity):
    for path in _PARITY_ROUTES:
        for token in (None, "wrong-token"):
            code, body, headers = _get(parity.base, path, token=token)
            assert code == 401, path
            _assert_no_store(headers)
            assert json.loads(body)["code"] == "UNAUTHORIZED", path
            assert TOKEN not in body.decode("utf-8")


# --------------------------------------------------------------------------
# freshness, cursor binding, resolver failure, null as_of
# --------------------------------------------------------------------------


def _persist_release(root, *, resolved_as_of):
    """Commit a REAL serving candidate through the API tests' own candidate
    pattern (same helpers ``_two_release_app`` uses), returning its serving
    root and committed release id."""
    root.mkdir()
    (root / "phase2").mkdir()
    conn, store, snap = _events_snapshot(
        root / "phase2",
        [_event_row(f"e{i}", f"T{i}", datetime(2024, 1, i + 1)) for i in range(7)])
    repo = Repository(conn, store)
    serving_root = root / "serving"
    serving_root.mkdir()
    serving_store = ArtifactStore(serving_root / "objects")
    serving_conn = projections.connect(str(serving_root / "serving.sqlite"))
    rows = _rows_for(100.0)
    release = projections.build_candidate(
        _preview_input(), _score_doc(rows=rows), _bundle(*[_compact(row) for row in rows]),
        repository=repo, snapshot_ref=snap, store=serving_store, conn=serving_conn,
        requested_as_of=resolved_as_of, resolved_as_of=resolved_as_of)
    serving_conn.close()
    return serving_root, release.release_id


def test_stale(parity, tmp_path):
    serving_root, release_id = _persist_release(tmp_path / "stale-serving",
                                                resolved_as_of="2026-05-05")
    app = create_app(serving_db=str(serving_root / "serving.sqlite"),
                     store_root=str(serving_root / "objects"),
                     serving_root=str(serving_root), token=TOKEN,
                     resolver=lambda: release_id,
                     native_parity_report_path=str(parity.report_path))
    server, thread, base = _start(app)
    try:
        code, body, headers = _get(base, "/api/v1/native_parity", token=TOKEN)
        assert code == 200
        _assert_no_store(headers)
        summary = json.loads(body)
        assert summary["status"] == "stale"
        assert summary["as_of"] == AS_OF
        assert summary["generated_at"] == GENERATED_AT
        assert summary["compared_count"] == 3
        assert summary["matched_row_count"] == 1
        assert summary["mismatched_row_count"] == 2
        assert summary["only_legacy_count"] == 2
        assert summary["only_native_count"] == 2

        code, body, headers = _get(base, "/api/v1/native_parity/mismatches", token=TOKEN,
                                   params={"limit": "10"})
        assert code == 200
        _assert_no_store(headers)
        document = json.loads(body)
        assert document["status"] == "stale"
        assert [(item["row_key"], item["dimension"]) for item in document["items"]] == [
            ("BBB|S|2026-01-02", "forecasts"), ("CCC|S|2026-01-03", "simulation")]
        assert document["next_cursor"] is None

        for side, expected in (("legacy", ["DDD|S|2026-01-04", "EEE|S|2026-01-05"]),
                               ("native", ["FFF|S|2026-01-06", "GGG|S|2026-01-07"])):
            code, body, headers = _get(base, "/api/v1/native_parity/unpaired", token=TOKEN,
                                       params={"side": side, "limit": "10"})
            assert code == 200
            _assert_no_store(headers)
            document = json.loads(body)
            assert document["status"] == "stale", side
            assert document["items"] == expected, side
    finally:
        _stop(server, thread)


def _assert_cursor_mismatch(base, path, params):
    code, body, headers = _get(base, path, token=TOKEN, params=params)
    assert code == 409, path
    assert json.loads(body)["code"] == "CURSOR_MISMATCH", path
    _assert_no_store(headers)


def test_cursor_mismatch(parity):
    mismatches = "/api/v1/native_parity/mismatches"
    unpaired = "/api/v1/native_parity/unpaired"
    code, body, _ = _get(parity.base, mismatches, token=TOKEN, params={"limit": "1"})
    assert code == 200
    first_cursor = json.loads(body)["next_cursor"]
    assert first_cursor is not None

    _assert_cursor_mismatch(parity.base, mismatches,
                            {"limit": "1", "cursor": first_cursor,
                             "row_key": "BBB|S|2026-01-02"})
    _assert_cursor_mismatch(parity.base, unpaired,
                            {"side": "legacy", "limit": "1", "cursor": first_cursor})

    code, body, _ = _get(parity.base, unpaired, token=TOKEN,
                         params={"side": "legacy", "limit": "1"})
    assert code == 200
    legacy_cursor = json.loads(body)["next_cursor"]
    assert legacy_cursor is not None
    _assert_cursor_mismatch(parity.base, unpaired,
                            {"side": "native", "limit": "1", "cursor": legacy_cursor})

    document = json.loads(json.dumps(parity.report))
    entry = next(item for item in document["mismatches"] if item["finding_fields"])
    entry["values"][entry["finding_fields"][0]]["native"] = 9.99
    parity.report_path.write_text(json.dumps(document))

    _assert_cursor_mismatch(parity.base, mismatches, {"limit": "1", "cursor": first_cursor})


def _signed_offset_cursor(parity, raw):
    document = json.loads(parity.report_path.read_text())
    query_hash = content_hash({"section": "mismatches", "side": None, "row_key": None})
    return api_module._pack_cursor(api_module._cursor_key(TOKEN), content_hash(document),
                                   query_hash, raw)


@pytest.mark.parametrize("raw", ("", "nope", "-1", "3"))
def test_signed_cursor_offsets_are_checked(parity, raw):
    _assert_cursor_mismatch(parity.base, "/api/v1/native_parity/mismatches",
                            {"limit": "1", "cursor": _signed_offset_cursor(parity, raw)})


def test_signed_cursor_exact_collection_length_is_empty(parity):
    code, body, headers = _get(parity.base, "/api/v1/native_parity/mismatches", token=TOKEN,
                               params={"limit": "1", "cursor": _signed_offset_cursor(parity, "2")})
    assert code == 200
    _assert_no_store(headers)
    document = json.loads(body)
    assert document["items"] == []
    assert document["next_cursor"] is None


_BINDING_INVALID_PROBLEM = {
    "code": "CURRENT_BINDING_INVALID", "category": "integrity", "retryable": False,
    "message": "the published pointer's projection binding does not match the live serving index",
    "stage": None, "trace_id": None, "dependency_refs": [], "retry_after_seconds": None,
    "diagnostic_ref": None, "details": {}, "schema_version": "problem.v1.0",
}


def test_current_release_failure(parity, tmp_path):
    def resolver():
        raise ApiError(500, dict(_BINDING_INVALID_PROBLEM))

    app = create_app(serving_db=str(tmp_path / "serving.sqlite"),
                     store_root=str(tmp_path / "objects"), serving_root=str(tmp_path / "serving"),
                     token=TOKEN, resolver=resolver,
                     native_parity_report_path=str(parity.report_path))
    server, thread, base = _start(app)
    try:
        code, body, _ = _get(base, "/api/v1/releases/current", token=TOKEN)
        assert code == 500
        assert json.loads(body)["code"] == "CURRENT_BINDING_INVALID"

        for path, params, expected in (
                ("/api/v1/native_parity", {}, None),
                ("/api/v1/native_parity/mismatches", {"limit": "10"}, 2),
                ("/api/v1/native_parity/unpaired", {"side": "legacy", "limit": "10"}, 2)):
            code, body, headers = _get(base, path, token=TOKEN, params=params)
            assert code == 200, path
            _assert_no_store(headers)
            document = json.loads(body)
            assert document["status"] == "available", path
            if expected is None:
                assert document["compared_count"] == 3, path
            else:
                assert len(document["items"]) == expected, path
    finally:
        _stop(server, thread)


def test_current_pointer_io_failure_leaves_report_available(parity, tmp_path, monkeypatch):
    """A real I/O failure reading the ops publisher's ``CURRENT`` is the same
    availability contract as the resolver's own ``ApiError`` above: the parity
    report stays available. The failure is injected only for that exact file
    through ``builtins.open`` so the real ``_publication_resolver`` chain is
    exercised, never a mocked resolver or freshness check."""
    publication_root = tmp_path / "publication"
    publication_root.mkdir()
    current_path = publication_root / "CURRENT"
    current_path.write_text("shadow-release-0001\n", encoding="utf-8")

    app = create_app(serving_db=str(tmp_path / "serving.sqlite"),
                     store_root=str(tmp_path / "objects"),
                     serving_root=str(tmp_path / "serving"),
                     token=TOKEN, publication_root=str(publication_root),
                     native_parity_report_path=str(parity.report_path))
    server, thread, base = _start(app)
    injected = []
    real_open = builtins.open

    def deny_current_candidate(file, *args, **kwargs):
        if isinstance(file, int):
            return real_open(file, *args, **kwargs)
        if os.fspath(file) == str(current_path):
            injected.append(str(current_path))
            raise PermissionError(13, "Permission denied", str(current_path))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny_current_candidate)
    try:
        code, body, headers = _get(base, "/api/v1/native_parity", token=TOKEN)
        assert code == 200
        _assert_no_store(headers)
        summary = json.loads(body)
        assert summary["status"] == "available"
        assert summary["as_of"] == AS_OF
        assert summary["compared_count"] == 3
        assert summary["matched_row_count"] == 1
        assert summary["mismatched_row_count"] == 2
        assert summary["only_legacy_count"] == 2
        assert summary["only_native_count"] == 2

        code, body, headers = _get(base, "/api/v1/native_parity/mismatches", token=TOKEN,
                                   params={"limit": "10"})
        assert code == 200
        _assert_no_store(headers)
        document = json.loads(body)
        assert document["status"] == "available"
        assert [(item["row_key"], item["dimension"]) for item in document["items"]] == [
            ("BBB|S|2026-01-02", "forecasts"), ("CCC|S|2026-01-03", "simulation")]
        assert document["next_cursor"] is None

        for side, expected in (("legacy", ["DDD|S|2026-01-04", "EEE|S|2026-01-05"]),
                               ("native", ["FFF|S|2026-01-06", "GGG|S|2026-01-07"])):
            code, body, headers = _get(base, "/api/v1/native_parity/unpaired", token=TOKEN,
                                       params={"side": side, "limit": "10"})
            assert code == 200
            _assert_no_store(headers)
            document = json.loads(body)
            assert document["status"] == "available", side
            assert document["items"] == expected, side

        assert injected == [str(current_path)] * 4
    finally:
        monkeypatch.undo()
        _stop(server, thread)


_SAFE_RELEASE_ID = "operational-fallback-release"


def _operational_resolver(phase):
    def resolver():
        if phase == "resolver":
            raise sqlite3.OperationalError("database is locked")
        return _SAFE_RELEASE_ID
    return resolver


def _assert_available_report(base):
    code, body, headers = _get(base, "/api/v1/native_parity", token=TOKEN)
    assert code == 200
    _assert_no_store(headers)
    summary = json.loads(body)
    assert summary["status"] == "available"
    assert summary["as_of"] == AS_OF
    assert summary["generated_at"] == GENERATED_AT
    assert summary["compared_count"] == 3
    assert summary["matched_row_count"] == 1
    assert summary["mismatched_row_count"] == 2
    assert summary["only_legacy_count"] == 2
    assert summary["only_native_count"] == 2

    code, body, headers = _get(base, "/api/v1/native_parity/mismatches", token=TOKEN,
                               params={"limit": "10"})
    assert code == 200
    _assert_no_store(headers)
    document = json.loads(body)
    assert document["status"] == "available"
    assert [(item["row_key"], item["dimension"]) for item in document["items"]] == [
        ("BBB|S|2026-01-02", "forecasts"), ("CCC|S|2026-01-03", "simulation")]
    assert document["next_cursor"] is None

    for side, expected in (("legacy", ["DDD|S|2026-01-04", "EEE|S|2026-01-05"]),
                           ("native", ["FFF|S|2026-01-06", "GGG|S|2026-01-07"])):
        code, body, headers = _get(base, "/api/v1/native_parity/unpaired", token=TOKEN,
                                   params={"side": side, "limit": "10"})
        assert code == 200
        _assert_no_store(headers)
        document = json.loads(body)
        assert document["status"] == "available", side
        assert document["items"] == expected, side


@pytest.mark.parametrize("phase", ("resolver", "open", "get_release"))
def test_operational_index_failure_leaves_report_available(parity, tmp_path, monkeypatch, phase):
    """A sqlite3.OperationalError -- from the resolver, the serving-db open, or
    the release lookup -- is the same availability contract as the ApiError and
    real OSError cases above. Injected at the dependency boundary (resolver,
    api._open, projections.get_release), never by mocking freshness itself."""
    connections = []
    if phase == "open":
        def fail_open(serving_db):
            raise sqlite3.OperationalError("unable to open database file")

        monkeypatch.setattr(api_module, "_open", fail_open)
    elif phase == "get_release":
        def memory_open(serving_db):
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            connections.append(conn)
            return conn

        def fail_get_release(conn, release_id):
            raise sqlite3.OperationalError("database disk image is malformed")

        monkeypatch.setattr(api_module, "_open", memory_open)
        monkeypatch.setattr(projections, "get_release", fail_get_release)

    app = create_app(serving_db=str(tmp_path / "serving.sqlite"),
                     store_root=str(tmp_path / "objects"),
                     serving_root=str(tmp_path / "serving"),
                     token=TOKEN, resolver=_operational_resolver(phase),
                     native_parity_report_path=str(parity.report_path))
    server, thread, base = _start(app)
    try:
        _assert_available_report(base)
    finally:
        monkeypatch.undo()
        _stop(server, thread)

    if phase == "get_release":
        assert len(connections) == 4
        for conn in connections:
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                conn.execute("SELECT 1")


def _integrity_error():
    return projections.ServingIndexError(Problem(
        code="SERVING_INDEX_UNREADABLE", category="integrity", retryable=False,
        message="the serving index cannot be read"))


def test_index_integrity_errors_propagate_from_freshness(monkeypatch):
    """The negative control: ServingIndexError is an integrity failure, not an
    availability signal -- resolver and lookup failures both propagate, and the
    lookup connection is still closed on the way out."""
    def fail_resolver():
        raise _integrity_error()

    with pytest.raises(projections.ServingIndexError):
        api_module._native_parity_freshness("unused", fail_resolver, AS_OF)

    connections = []

    def memory_open(serving_db):
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        connections.append(conn)
        return conn

    def fail_get_release(conn, release_id):
        raise _integrity_error()

    monkeypatch.setattr(api_module, "_open", memory_open)
    monkeypatch.setattr(projections, "get_release", fail_get_release)
    with pytest.raises(projections.ServingIndexError):
        api_module._native_parity_freshness("unused", lambda: _SAFE_RELEASE_ID, AS_OF)
    assert len(connections) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        connections[0].execute("SELECT 1")


def test_null_as_of(parity):
    legacy, native = _source_rows()
    report = compare_native_vs_legacy(legacy, native, ("forecasts", "simulation"))
    report = _stamp_report_identity(report, as_of=None, clock=_FixedClock())
    write_parity_report(report, parity.report_path)

    code, body, headers = _get(parity.base, "/api/v1/native_parity", token=TOKEN)
    assert code == 200
    _assert_no_store(headers)
    summary = json.loads(body)
    assert summary["status"] == "available"
    assert summary["as_of"] is None
    assert summary["generated_at"] == GENERATED_AT
    assert summary["compared_count"] == 3
    assert summary["matched_row_count"] == 1
    assert summary["mismatched_row_count"] == 2
    assert summary["only_legacy_count"] == 2
    assert summary["only_native_count"] == 2


# --------------------------------------------------------------------------
# shared-reader boundary: the modern identity/policy/value checks live in
# engine/v2/serving/native_parity_projection and the API route keeps only
# its required-identity refusal for stamped v1.2 reports
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", ("missing_identity_keys", "malformed_as_of",
                                  "missing_value_pair"))
def test_shared_reader_refuses_malformed_modern_fields(parity, case):
    _corrupt_report(parity, case)
    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.SERVICE_UNAVAILABLE, case
    assert summary["status"] == "unavailable", case
    assert summary["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED, case
    assert "compared_count" not in summary, case


def test_space_padded_as_of_day_is_refused_as_malformed(parity):
    document = json.loads(json.dumps(parity.report))
    document["as_of"] = "2026-01- 2"
    write_parity_report(document, parity.report_path)

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.SERVICE_UNAVAILABLE
    assert summary["status"] == "unavailable"
    assert summary["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED
    assert "compared_count" not in summary

    _assert_malformed_routes(parity)


def test_raw_v12_diagnostic_reads_shared_but_api_refuses_missing_identity(parity):
    """The captured exporter's raw v1.2 diagnostic (no run identity) stays
    readable by the shared reader, while the API route refuses it."""
    legacy, native = _source_rows()
    report = compare_native_vs_legacy(legacy, native, ("forecasts", "simulation"))
    report["schema_version"] = "native_parity_report.v1.2"
    assert "as_of" not in report and "generated_at" not in report
    write_parity_report(report, parity.report_path)

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.OK
    assert summary["status"] == "available"
    assert summary["compared_count"] == 3

    code, body, headers = _get(parity.base, "/api/v1/native_parity", token=TOKEN)
    assert code == 503
    _assert_no_store(headers)
    assert json.loads(body)["code"] == NATIVE_PARITY_REPORT_MALFORMED


def test_pre_v12_report_without_identity_stays_readable(parity):
    legacy, native = _source_rows()
    report = compare_native_vs_legacy(legacy, native, ("forecasts", "simulation"))
    report["schema_version"] = "native_parity_report.v1.1"
    write_parity_report(report, parity.report_path)

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.OK
    assert summary["status"] == "available"
    assert summary["source_schema_version"] == "native_parity_report.v1.1"

    code, body, headers = _get(parity.base, "/api/v1/native_parity", token=TOKEN)
    assert code == 200
    _assert_no_store(headers)
    assert json.loads(body)["status"] == "available"


@pytest.mark.parametrize("case", ("unknown_dimension", "wrong_group_field"))
def test_shared_reader_refuses_unknown_dimension_or_wrong_group_field(parity, case):
    document = json.loads(parity.report_path.read_text())
    mismatch = document["mismatches"][0]
    assert mismatch["dimension"] == "forecasts", case
    if case == "unknown_dimension":
        mismatch["dimension"] = "unknown_dimension"
    else:
        name = mismatch["finding_fields"][0]
        mismatch["finding_fields"][0] = "exp_pnl_sim"
        mismatch["values"]["exp_pnl_sim"] = mismatch["values"].pop(name)
    write_parity_report(document, parity.report_path)

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.SERVICE_UNAVAILABLE, case
    assert summary["status"] == "unavailable", case
    assert summary["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED, case

    for path, params in (("/api/v1/native_parity", None),
                         ("/api/v1/native_parity/mismatches", None),
                         ("/api/v1/native_parity/unpaired", {"side": "legacy"})):
        code, body, headers = _get(parity.base, path, token=TOKEN, params=params)
        assert code == 503, (case, path)
        _assert_no_store(headers)
        problem = json.loads(body)
        assert problem["code"] == NATIVE_PARITY_REPORT_MALFORMED, (case, path)
        assert "compared_count" not in problem, (case, path)
        assert "items" not in problem, (case, path)


def test_pre_v11_raw_comparison_unknown_dimension_and_field_stays_readable(parity):
    legacy, native = _source_rows()
    report = compare_native_vs_legacy(legacy, native, ("forecasts", "simulation"))
    mismatch = report["mismatches"][0]
    assert mismatch["dimension"] == "forecasts"
    name = mismatch["finding_fields"][0]
    mismatch["finding_fields"][0] = "unknown_field"
    mismatch["values"]["unknown_field"] = mismatch["values"].pop(name)
    mismatch["dimension"] = "unknown_dimension"
    report["schema_version"] = "native_parity_report.v1.1"
    write_parity_report(report, parity.report_path)

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.OK
    assert summary["status"] == "available"


# --------------------------------------------------------------------------
# non-finite saved value pairs: the shared reader refuses them before any
# route can serialize a response, and every valid saved value still serves
# --------------------------------------------------------------------------


_NON_FINITE_LITERALS = ("NaN", "Infinity", "-Infinity", "1e999", "-1e999")
_SAVED_VALUE_SENTINEL = "saved-value-sentinel-4f19c2"
_POSITIVE_SAVED_VALUES = (
    pytest.param(1.25, id="finite"),
    pytest.param(None, id="null"),
    pytest.param("NaN", id="string_nan"),
    pytest.param("Infinity", id="string_infinity"),
    pytest.param(2 ** 53 + 1, id="large_int"),
    pytest.param(True, id="bool"),
)


def _first_mismatch(document):
    mismatch = next(entry for entry in document["mismatches"] if entry["finding_fields"])
    assert mismatch["row_key"] == "BBB|S|2026-01-02"
    assert mismatch["dimension"] == "forecasts"
    assert mismatch["finding_fields"] == ["forecast_p10"]
    return mismatch


def _assert_malformed_routes(parity):
    for path in _PARITY_ROUTES:
        params = {"side": "legacy"} if path.endswith("unpaired") else None
        code, body, headers = _get(parity.base, path, token=parity.token, params=params)
        assert code == 503, path
        _assert_no_store(headers)
        problem = json.loads(body)
        assert problem["code"] == NATIVE_PARITY_REPORT_MALFORMED, path
        assert "items" not in problem, path
        assert "compared_count" not in problem, path


@pytest.mark.parametrize("value", (None, True, 1, [], {}))
def test_nonstring_generated_at_is_refused(parity, value):
    document = json.loads(json.dumps(parity.report))
    document["generated_at"] = value
    parity.report_path.write_text(json.dumps(document))

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.SERVICE_UNAVAILABLE
    assert summary["status"] == "unavailable"
    assert summary["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED
    _assert_malformed_routes(parity)


@pytest.mark.parametrize("side", ("legacy", "native"))
@pytest.mark.parametrize("literal", _NON_FINITE_LITERALS)
def test_non_finite_saved_value_is_refused_everywhere(parity, side, literal):
    document = json.loads(json.dumps(parity.report))
    _first_mismatch(document)["values"]["forecast_p10"][side] = _SAVED_VALUE_SENTINEL
    write_parity_report(document, parity.report_path)
    text = parity.report_path.read_text()
    assert json.dumps(_SAVED_VALUE_SENTINEL) in text
    parity.report_path.write_text(
        text.replace(json.dumps(_SAVED_VALUE_SENTINEL), literal, 1))

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.SERVICE_UNAVAILABLE, (side, literal)
    assert summary["status"] == "unavailable", (side, literal)
    assert summary["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED, (side, literal)
    assert "compared_count" not in summary, (side, literal)

    _assert_malformed_routes(parity)


def test_nested_non_finite_saved_value_is_refused(parity):
    document = json.loads(json.dumps(parity.report))
    _first_mismatch(document)["values"]["forecast_p10"]["legacy"] = {
        "nested": [1.0, {"deeper": float("nan")}]}
    write_parity_report(document, parity.report_path)

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.SERVICE_UNAVAILABLE
    assert summary["status"] == "unavailable"
    assert summary["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED

    _assert_malformed_routes(parity)


@pytest.mark.parametrize("side", ("legacy", "native"))
@pytest.mark.parametrize("value", _POSITIVE_SAVED_VALUES)
def test_valid_saved_values_serve_exactly(parity, side, value):
    document = json.loads(json.dumps(parity.report))
    pair = _first_mismatch(document)["values"]["forecast_p10"]
    pair[side] = value
    write_parity_report(document, parity.report_path)

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.OK, (side, value)
    assert summary["status"] == "available", (side, value)
    assert summary["mismatched_row_count"] == 2, (side, value)

    code, body, headers = _get(parity.base, "/api/v1/native_parity", token=parity.token)
    assert code == 200, (side, value)
    _assert_no_store(headers)
    assert json.loads(body)["status"] == "available", (side, value)

    code, body, headers = _get(parity.base, "/api/v1/native_parity/mismatches",
                               token=parity.token, params={"limit": "10"})
    assert code == 200, (side, value)
    _assert_no_store(headers)
    first = json.loads(body)["items"][0]
    assert (first["row_key"], first["dimension"]) == ("BBB|S|2026-01-02", "forecasts")
    served = first["fields"]["forecast_p10"]
    assert served["status"] == "differ", (side, value)
    assert served[side] == value, (side, value)
    assert type(served[side]) is type(value), (side, value)
    other = "native" if side == "legacy" else "legacy"
    assert served[other] == (1.25 if other == "native" else 1.0), (side, value)
    if isinstance(value, int) and not isinstance(value, bool):
        assert served[side] == 2 ** 53 + 1, (side, value)


# --------------------------------------------------------------------------
# forwarding-safety round: unstamped v1.1 compatibility, non-finite forwarded
# metadata, lone surrogates, recursion and strict-encoder round-trips
# --------------------------------------------------------------------------


def _unstamped_document(parity):
    """The real stamped fixture in the pre-identity v1.1 shape."""
    document = json.loads(json.dumps(parity.report))
    document["schema_version"] = "native_parity_report.v1.1"
    del document["as_of"]
    del document["generated_at"]
    return document


def _write_unstamped(parity, mutate):
    document = _unstamped_document(parity)
    mutate(document)
    parity.report_path.write_text(json.dumps(document))
    return document


@pytest.mark.parametrize("side", ("legacy", "native"))
@pytest.mark.parametrize("literal", _NON_FINITE_LITERALS)
def test_unstamped_v11_non_finite_saved_value_serves_summary_and_unpaired(parity, side, literal):
    """An ignored legacy saved value stays out of the summary and both unpaired
    collections (which never project it); only the mismatch screen, which would
    forward the raw value, is the refused one."""
    document = _unstamped_document(parity)
    _first_mismatch(document)["values"]["forecast_p10"][side] = _SAVED_VALUE_SENTINEL
    parity.report_path.write_text(json.dumps(document))
    text = parity.report_path.read_text()
    assert json.dumps(_SAVED_VALUE_SENTINEL) in text
    parity.report_path.write_text(text.replace(json.dumps(_SAVED_VALUE_SENTINEL), literal, 1))

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.OK, (side, literal)
    assert summary["status"] == "available", (side, literal)
    assert summary["compared_count"] == 3, (side, literal)
    assert summary["mismatched_row_count"] == 2, (side, literal)

    code, body, headers = _get(parity.base, "/api/v1/native_parity", token=parity.token)
    assert code == 200, (side, literal)
    _assert_no_store(headers)
    authenticated = json.loads(body)
    assert authenticated["status"] == "available", (side, literal)
    assert authenticated["compared_count"] == 3, (side, literal)
    assert authenticated["only_legacy_count"] == 2, (side, literal)

    code, body, headers = _get(parity.base, "/api/v1/native_parity/mismatches",
                               token=parity.token, params={"limit": "10"})
    assert code == 503, (side, literal)
    _assert_no_store(headers)
    problem = json.loads(body)
    assert problem["code"] == NATIVE_PARITY_REPORT_MALFORMED, (side, literal)
    assert "items" not in problem, (side, literal)
    assert "compared_count" not in problem, (side, literal)

    code, body, headers = _get(parity.base, "/api/v1/native_parity/unpaired",
                               token=parity.token, params={"side": "legacy", "limit": "10"})
    assert code == 200, (side, literal)
    _assert_no_store(headers)
    assert json.loads(body)["items"] == ["DDD|S|2026-01-04", "EEE|S|2026-01-05"], (side, literal)


def test_unstamped_v11_nested_non_finite_saved_value_refuses_only_mismatches(parity):
    def mutate(document):
        _first_mismatch(document)["values"]["forecast_p10"]["legacy"] = {
            "nested": [1.0, {"deeper": float("nan")}]}

    _write_unstamped(parity, mutate)

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.OK
    assert summary["status"] == "available"

    code, body, headers = _get(parity.base, "/api/v1/native_parity/mismatches",
                               token=parity.token, params={"limit": "10"})
    assert code == 503
    _assert_no_store(headers)
    assert json.loads(body)["code"] == NATIVE_PARITY_REPORT_MALFORMED

    code, body, headers = _get(parity.base, "/api/v1/native_parity/unpaired",
                               token=parity.token, params={"side": "native", "limit": "10"})
    assert code == 200
    _assert_no_store(headers)
    assert json.loads(body)["items"] == ["FFF|S|2026-01-06", "GGG|S|2026-01-07"]


_NON_FINITE_FORWARDED_METADATA = (
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="infinity"),
    pytest.param([1.0, [{"deep": float("-inf")}]], id="nested_non_finite"),
)


@pytest.mark.parametrize("policy", _NON_FINITE_FORWARDED_METADATA)
def test_unstamped_v11_non_finite_forwarded_policy_refuses_every_route(parity, policy):
    """The summary passes legacy tolerance_policy_id through verbatim: a
    non-finite value there is the strict encoder's refusal at every route, with
    no new policy type rule introduced for old reports."""
    def mutate(document):
        document["tolerance_policy_id"] = policy

    _write_unstamped(parity, mutate)

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.SERVICE_UNAVAILABLE, policy
    assert summary["status"] == "unavailable", policy
    assert summary["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED, policy
    assert "compared_count" not in summary, policy

    _assert_malformed_routes(parity)


_SURROGATE = "\ud800"
_SURROGATE_SUMMARY_CASES = ("schema_version", "tolerance_policy_id", "mismatch_key",
                            "dimension", "finding_field", "refusal_code")


def _apply_surrogate_summary_case(document, case):
    if case == "schema_version":
        document["schema_version"] = _SURROGATE
    elif case == "tolerance_policy_id":
        document["tolerance_policy_id"] = _SURROGATE
    elif case == "mismatch_key":
        mismatch = _first_mismatch(document)
        document["compared"][document["compared"].index(mismatch["row_key"])] = _SURROGATE
        mismatch["row_key"] = _SURROGATE
    elif case == "dimension":
        _first_mismatch(document)["dimension"] = _SURROGATE
    elif case == "finding_field":
        mismatch = _first_mismatch(document)
        name = mismatch["finding_fields"][0]
        mismatch["finding_fields"][0] = _SURROGATE
        mismatch["values"][_SURROGATE] = mismatch["values"].pop(name)
    elif case == "refusal_code":
        document["native_refused"] = [{"row_key": "AAA|S|2026-01-01",
                                       "refusal_code": _SURROGATE}]
    else:
        raise AssertionError(case)


@pytest.mark.parametrize("case", _SURROGATE_SUMMARY_CASES)
def test_unstamped_v11_lone_surrogate_in_summary_field_is_refused(parity, case):
    """The summary forwards these strings/keys, so a lone surrogate is the
    malformed-report refusal everywhere. The stored JSON is written escaped,
    never as raw surrogate UTF-8 bytes."""
    document = _unstamped_document(parity)
    _apply_surrogate_summary_case(document, case)
    parity.report_path.write_text(json.dumps(document))

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.SERVICE_UNAVAILABLE, case
    assert summary["status"] == "unavailable", case
    assert summary["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED, case

    _assert_malformed_routes(parity)


_SURROGATE_DETAIL_CASES = ("unpaired_key", "saved_value", "nested_saved_key")


def _apply_surrogate_detail_case(document, case):
    if case == "unpaired_key":
        document["only_legacy"].append(_SURROGATE)
    elif case == "saved_value":
        _first_mismatch(document)["values"]["forecast_p10"]["legacy"] = _SURROGATE
    elif case == "nested_saved_key":
        _first_mismatch(document)["values"]["forecast_p10"]["native"] = {_SURROGATE: 1.0}
    else:
        raise AssertionError(case)


@pytest.mark.parametrize("case", _SURROGATE_DETAIL_CASES)
def test_unstamped_v11_lone_surrogate_in_ignored_details_keeps_summary(parity, case):
    """The summary ignores these contents (unpaired keys are counts, saved
    values are not projected), so it stays readable; every detail route that
    would forward them refuses as malformed."""
    document = _unstamped_document(parity)
    _apply_surrogate_detail_case(document, case)
    parity.report_path.write_text(json.dumps(document))

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.OK, case
    assert summary["status"] == "available", case
    assert summary["source_schema_version"] == "native_parity_report.v1.1", case

    code, body, headers = _get(parity.base, "/api/v1/native_parity", token=parity.token)
    assert code == 200, case
    _assert_no_store(headers)
    assert json.loads(body)["status"] == "available", case

    for path, params in (("/api/v1/native_parity/mismatches", {"limit": "10"}),
                         ("/api/v1/native_parity/unpaired", {"side": "legacy", "limit": "10"})):
        code, body, headers = _get(parity.base, path, token=parity.token, params=params)
        assert code == 503, (case, path)
        _assert_no_store(headers)
        assert json.loads(body)["code"] == NATIVE_PARITY_REPORT_MALFORMED, (case, path)


@pytest.mark.parametrize("case", ("tolerance_policy_id", "saved_value", "nested_saved_key",
                                  "refusal_code"))
def test_stamped_v12_lone_surrogate_in_modern_fields_is_refused(parity, case):
    document = json.loads(json.dumps(parity.report))
    if case == "tolerance_policy_id":
        document["tolerance_policy_id"] = _SURROGATE
    elif case == "saved_value":
        _first_mismatch(document)["values"]["forecast_p10"]["legacy"] = _SURROGATE
    elif case == "nested_saved_key":
        _first_mismatch(document)["values"]["forecast_p10"]["native"] = {_SURROGATE: 1.0}
    else:
        document["native_refused"] = [{"row_key": "AAA|S|2026-01-01",
                                       "refusal_code": _SURROGATE}]
    parity.report_path.write_text(json.dumps(document))

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.SERVICE_UNAVAILABLE, case
    assert summary["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED, case

    _assert_malformed_routes(parity)


def test_stamped_v12_lone_surrogate_unpaired_key_keeps_summary_but_refuses_details(parity):
    document = json.loads(json.dumps(parity.report))
    document["only_legacy"].append(_SURROGATE)
    parity.report_path.write_text(json.dumps(document))

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.OK
    assert summary["status"] == "available"

    code, body, headers = _get(parity.base, "/api/v1/native_parity", token=parity.token)
    assert code == 200
    _assert_no_store(headers)

    code, body, headers = _get(parity.base, "/api/v1/native_parity/unpaired",
                               token=parity.token, params={"side": "legacy", "limit": "10"})
    assert code == 503
    _assert_no_store(headers)
    assert json.loads(body)["code"] == NATIVE_PARITY_REPORT_MALFORMED


_DEEP_FORWARDED_LEAF = "deep-forwarded-leaf-8f42"


def _deep_forwarded_text(depth):
    body = json.dumps(_DEEP_FORWARDED_LEAF)
    for _ in range(depth):
        body = "[" + body + "]"
    return body


def test_unstamped_v11_deep_forwarded_metadata_serves_without_transport_500(parity):
    """A nested-but-valid forwarded legacy value passes the projection boundary
    and must not be re-rejected by FastAPI's own recursive encoder at the
    transport seam: the shared reader and the authenticated summary both serve
    it, and the test unwraps iteratively to the exact leaf."""
    depth = sys.getrecursionlimit() + 100
    deep_text = _deep_forwarded_text(depth)
    try:
        json.dumps(json.loads(deep_text))
    except RecursionError:
        depth = sys.getrecursionlimit() // 2 + 50
        deep_text = _deep_forwarded_text(depth)
    document = _unstamped_document(parity)
    document["tolerance_policy_id"] = "__deep_forwarded_policy__"
    text = json.dumps(document)
    parity.report_path.write_text(
        text.replace(json.dumps("__deep_forwarded_policy__"), deep_text, 1))

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.OK
    assert summary["status"] == "available"
    assert summary["compared_count"] == 3

    code, body, headers = _get(parity.base, "/api/v1/native_parity", token=parity.token)
    assert code == 200
    _assert_no_store(headers)
    authenticated = json.loads(body)
    assert authenticated["status"] == "available"
    assert authenticated["compared_count"] == 3

    leaves = []
    for forwarded in (summary["tolerance_policy_id"],
                      authenticated["tolerance_policy_id"]):
        assert type(forwarded) is list
        assert len(forwarded) == 1
        value = forwarded
        for _ in range(depth):
            assert type(value) is list and len(value) == 1
            value = value[0]
        leaves.append(value)
    assert leaves == [_DEEP_FORWARDED_LEAF, _DEEP_FORWARDED_LEAF]


@pytest.mark.parametrize("schema_version, unstamped", (
    ("native_parity_report.v1.1", True),
    ("native_parity_report.v1.2", False),
))
def test_parser_recursion_is_503_not_500(parity, monkeypatch, schema_version, unstamped):
    """A RecursionError from the projection's own json.load parser is the
    malformed-report refusal for both schemas, never an untyped transport
    failure, and every opened report handle is closed again."""
    document = json.loads(json.dumps(parity.report))
    document["schema_version"] = schema_version
    if unstamped:
        del document["as_of"]
        del document["generated_at"]
    write_parity_report(document, parity.report_path)

    handles = []

    def exploding_load(handle, *args, **kwargs):
        handles.append(handle)
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(native_parity_projection.json, "load", exploding_load)

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.SERVICE_UNAVAILABLE, schema_version
    assert summary["status"] == "unavailable", schema_version
    assert summary["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED, schema_version
    assert handles and all(handle.closed for handle in handles), schema_version

    for path in _PARITY_ROUTES:
        params = {"side": "legacy"} if path.endswith("unpaired") else None
        code, body, headers = _get(parity.base, path, token=parity.token, params=params)
        assert code == 503, (schema_version, path)
        _assert_no_store(headers)
        assert json.loads(body)["code"] == NATIVE_PARITY_REPORT_MALFORMED, (schema_version, path)
        assert b"maximum recursion depth exceeded" not in body, (schema_version, path)

    assert len(handles) == 1 + len(_PARITY_ROUTES), schema_version
    assert all(handle.closed for handle in handles), schema_version


def test_cursor_hashing_recursion_on_ignored_extras_is_typed_503(parity, monkeypatch):
    """The summary ignores unknown report extras, but a detail route's cursor
    hash recursion over them must classify as the malformed-report 503, never a
    500. Injecting RecursionError at the native parity content_hash boundary is
    the stable way to exercise that classification."""
    document = json.loads(json.dumps(parity.report))
    document["unknown_extra"] = {"ignored": [1, 2, 3]}
    parity.report_path.write_text(json.dumps(document))
    real_hash = api_module.content_hash

    def exploding_hash(value, **kwargs):
        if isinstance(value, dict) and "mismatches" in value:
            raise RecursionError("maximum recursion depth exceeded")
        return real_hash(value, **kwargs)

    monkeypatch.setattr(api_module, "content_hash", exploding_hash)

    code, body, headers = _get(parity.base, "/api/v1/native_parity", token=parity.token)
    assert code == 200
    _assert_no_store(headers)
    assert json.loads(body)["status"] == "available"

    code, body, headers = _get(parity.base, "/api/v1/native_parity/mismatches",
                               token=parity.token, params={"limit": "10"})
    assert code == 503
    _assert_no_store(headers)
    assert json.loads(body)["code"] == NATIVE_PARITY_REPORT_MALFORMED


_LEGACY_DETAIL_VALUES = (
    pytest.param(1.25, id="finite"),
    pytest.param(-0.0, id="negative_zero"),
    pytest.param(None, id="null"),
    pytest.param(True, id="bool"),
    pytest.param(2 ** 53 + 1, id="large_int"),
    pytest.param(10 ** 400, id="arbitrary_precision_int"),
    pytest.param(sys.float_info.max, id="max_finite_float"),
    pytest.param("NaN", id="string_nan_marker"),
    pytest.param("Infinity", id="string_infinity_marker"),
    pytest.param({"nested": [1, 2, {"deep": False}]}, id="nested_finite"),
    pytest.param("\U0001F600 safe \u2713 \u00e9", id="non_bmp_unicode"),
)


@pytest.mark.parametrize("side", ("legacy", "native"))
@pytest.mark.parametrize("value", _LEGACY_DETAIL_VALUES)
def test_unstamped_v11_finite_saved_value_round_trips_through_pages(parity, side, value):
    document = _unstamped_document(parity)
    _first_mismatch(document)["values"]["forecast_p10"][side] = value
    parity.report_path.write_text(json.dumps(document))

    code, body, headers = _get(parity.base, "/api/v1/native_parity/mismatches",
                               token=parity.token, params={"limit": "1"})
    assert code == 200, (side, value)
    _assert_no_store(headers)
    first_page = json.loads(body)
    assert first_page["next_cursor"] is not None, (side, value)

    code, body, headers = _get(parity.base, "/api/v1/native_parity/mismatches",
                               token=parity.token,
                               params={"limit": "1", "cursor": first_page["next_cursor"]})
    assert code == 200, (side, value)
    _assert_no_store(headers)
    second_page = json.loads(body)
    assert second_page["next_cursor"] is None, (side, value)

    first = first_page["items"][0]
    assert (first["row_key"], first["dimension"]) == ("BBB|S|2026-01-02", "forecasts")
    served = first["fields"]["forecast_p10"]
    assert served["status"] == "differ", (side, value)
    assert served[side] == value, (side, value)
    assert type(served[side]) is type(value), (side, value)
    if isinstance(value, float) and value == 0.0:
        assert math.copysign(1.0, served[side]) == -1.0, (side, value)


_LEGACY_READABILITY_CASES = ("missing_saved_pair", "unknown_dimension_field",
                             "absent_policy", "null_policy")


def _apply_readability_case(document, case):
    if case == "missing_saved_pair":
        _first_mismatch(document)["values"]["forecast_p10"].pop("native")
    elif case == "unknown_dimension_field":
        mismatch = _first_mismatch(document)
        name = mismatch["finding_fields"][0]
        mismatch["finding_fields"][0] = "unknown_field"
        mismatch["values"]["unknown_field"] = mismatch["values"].pop(name)
        mismatch["dimension"] = "unknown_dimension"
    elif case == "absent_policy":
        del document["tolerance_policy_id"]
    elif case == "null_policy":
        document["tolerance_policy_id"] = None
    else:
        raise AssertionError(case)


@pytest.mark.parametrize("case", _LEGACY_READABILITY_CASES)
def test_unstamped_v11_ignored_or_missing_modern_fields_stay_readable(parity, case):
    document = _unstamped_document(parity)
    _apply_readability_case(document, case)
    parity.report_path.write_text(json.dumps(document))

    code, summary = native_parity_summary(str(parity.report_path))
    assert code == HTTPStatus.OK, case
    assert summary["status"] == "available", case
    assert summary["source_schema_version"] == "native_parity_report.v1.1", case

    code, body, headers = _get(parity.base, "/api/v1/native_parity/mismatches",
                               token=parity.token, params={"limit": "10"})
    assert code == 200, case
    _assert_no_store(headers)
    items = json.loads(body)["items"]
    expected_dimension = ("unknown_dimension" if case == "unknown_dimension_field"
                          else "forecasts")
    assert [(item["row_key"], item["dimension"]) for item in items] == [
        ("BBB|S|2026-01-02", expected_dimension),
        ("CCC|S|2026-01-03", "simulation")], case

    first = items[0]
    if case == "missing_saved_pair":
        assert first["fields"]["forecast_p10"] == {"status": "differ"}, case
    elif case == "unknown_dimension_field":
        assert first["fields"] == {"unknown_field": {
            "status": "differ", "legacy": 1.0, "native": 1.25}}, case
    else:
        assert first["fields"]["forecast_p10"] == {
            "status": "differ", "legacy": 1.0, "native": 1.25}, case

    if case in ("absent_policy", "null_policy"):
        assert summary["tolerance_policy_id"] is None, case
