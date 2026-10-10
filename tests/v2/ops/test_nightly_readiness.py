"""Real-file tests for the legacy-nightly readiness check (slice 3 of #564).

Reports are written with the legacy producer's own ``NightlyReport.as_dict`` so a change in
its shape breaks these tests instead of silently passing a hand-written fixture.
"""
from __future__ import annotations

import json

import pytest

from engine.dashboard.nightly import NightlyReport
from engine.v2.ops import nightly_readiness as nrd
from engine.v2.ops.errors import OpsError

D = "2026-10-01"
GOOD_TIERS = {"rebuilt": ["panel", "tier4"], "tier4_since": "2026-09-01", "elapsed_s": 1.0}
DEGRADED = {"degraded": True, "error": "Tier4Error: fold"}


def _write(directory, requested, *, resolved=D, tiers=GOOD_TIERS, final=True, stopped=None,
           finality_date=None):
    report = NightlyReport(as_of=resolved, requested_as_of=requested, resolved_as_of=resolved)
    report.finality = {"date": finality_date or resolved, "is_final": final}
    report.stopped = stopped
    if tiers is not None:
        report.steps["tiers"] = tiers
    path = directory / f"nightly_{requested}.json"
    path.write_text(json.dumps(report.as_dict(), default=str))
    return path


def _code(excinfo) -> str:
    return excinfo.value.code


def test_ready_report_for_the_session_is_returned(tmp_path):
    _write(tmp_path, "2026-10-02")
    got = nrd.check_legacy_report(tmp_path, D)
    assert got == nrd.LegacyReadiness("nightly_2026-10-02.json", "2026-10-02", D)


def test_no_report_or_a_report_for_another_session_is_source_not_found(tmp_path):
    with pytest.raises(OpsError) as exc:
        nrd.check_legacy_report(tmp_path, D)
    assert _code(exc) == "SOURCE_NOT_FOUND"
    _write(tmp_path, "2026-10-02", resolved="2026-09-30")  # walked back to another session
    with pytest.raises(OpsError) as exc:
        nrd.check_legacy_report(tmp_path, D)
    assert _code(exc) == "SOURCE_NOT_FOUND"


def test_report_requested_outside_the_candidate_window_is_not_considered(tmp_path):
    _write(tmp_path, "2026-10-10")
    with pytest.raises(OpsError) as exc:
        nrd.check_legacy_report(tmp_path, D)
    assert _code(exc) == "SOURCE_NOT_FOUND"


@pytest.mark.parametrize("tiers", [DEGRADED, {"error": "boom"}, {"degraded": True}, None])
def test_degraded_or_missing_tier_step_stops_the_chain(tmp_path, tiers):
    _write(tmp_path, "2026-10-02", tiers=tiers)
    with pytest.raises(OpsError) as exc:
        nrd.check_legacy_report(tmp_path, D)
    assert _code(exc) == "DEPENDENCY_FAILED"
    assert exc.value.problem.details["step"] == "tiers"


def test_stopped_report_is_dependency_failed(tmp_path):
    _write(tmp_path, "2026-10-02", stopped="validate")
    with pytest.raises(OpsError) as exc:
        nrd.check_legacy_report(tmp_path, D)
    assert _code(exc) == "DEPENDENCY_FAILED"
    assert exc.value.problem.details["step"] == "validate"


@pytest.mark.parametrize("kwargs", [{"final": False}, {"finality_date": "2026-09-30"}])
def test_session_not_recorded_final_is_source_not_final(tmp_path, kwargs):
    _write(tmp_path, "2026-10-02", **kwargs)
    with pytest.raises(OpsError) as exc:
        nrd.check_legacy_report(tmp_path, D)
    assert _code(exc) == "SOURCE_NOT_FINAL"


def test_the_latest_requested_report_for_the_session_decides(tmp_path):
    _write(tmp_path, "2026-10-01", tiers=DEGRADED)
    _write(tmp_path, "2026-10-03")
    assert nrd.check_legacy_report(tmp_path, D).requested_as_of == "2026-10-03"
    _write(tmp_path, "2026-10-04", tiers=DEGRADED)
    with pytest.raises(OpsError) as exc:
        nrd.check_legacy_report(tmp_path, D)
    assert _code(exc) == "DEPENDENCY_FAILED"


@pytest.mark.parametrize("text", ["{not json", "[]", json.dumps({"as_of": D}),
                                  json.dumps({"as_of": D, "steps": []}),
                                  json.dumps({"as_of": 5, "steps": {}})])
def test_unreadable_candidate_is_an_integrity_failure_never_absent(tmp_path, text):
    (tmp_path / "nightly_2026-10-02.json").write_text(text)
    with pytest.raises(OpsError) as exc:
        nrd.check_legacy_report(tmp_path, D)
    assert _code(exc) == "INTEGRITY_FAILED"


@pytest.mark.parametrize("as_of", ["2026-10-1", "", None, "tomorrow"])
def test_invalid_session_is_invalid_request(tmp_path, as_of):
    with pytest.raises(OpsError) as exc:
        nrd.check_legacy_report(tmp_path, as_of)
    assert _code(exc) == "INVALID_REQUEST"


def test_the_check_writes_nothing(tmp_path):
    _write(tmp_path, "2026-10-02")
    before = sorted((p.name, p.read_bytes()) for p in tmp_path.iterdir())
    nrd.check_legacy_report(tmp_path, D)
    assert sorted((p.name, p.read_bytes()) for p in tmp_path.iterdir()) == before


def test_report_without_resolved_as_of_uses_as_of(tmp_path):
    path = _write(tmp_path, "2026-10-02")
    doc = json.loads(path.read_text())
    doc["resolved_as_of"] = None
    path.write_text(json.dumps(doc))
    assert nrd.check_legacy_report(tmp_path, D).resolved_as_of == D


def test_report_without_finality_is_source_not_final(tmp_path):
    path = _write(tmp_path, "2026-10-02")
    doc = json.loads(path.read_text())
    doc["finality"] = None
    path.write_text(json.dumps(doc))
    with pytest.raises(OpsError) as exc:
        nrd.check_legacy_report(tmp_path, D)
    assert _code(exc) == "SOURCE_NOT_FINAL"