"""Unit tests for report-only implied input collection in the IV crush search report."""
from types import SimpleNamespace

import pytest

from experiments import iv_crush_search_report as search_report
from experiments.iv_crush_search_report import _implied_inputs


def _fold():
    return {"evaluation_event_ids": ["e1", "e2"]}


def _inputs():
    return {"features": [
        {"event_id": "e1", "horizon_sessions": 1, "iv30": 50., "exern_iv30": 40., "other": 9},
        {"event_id": "e1", "horizon_sessions": 3, "iv30": None, "exern_iv30": 41.},
        {"event_id": "e1", "horizon_sessions": 5, "iv30": 52., "exern_iv30": 42.},
        {"event_id": "e2", "horizon_sessions": 1, "iv30": 60., "exern_iv30": 0.},
        {"event_id": "t1", "horizon_sessions": 1, "iv30": 70., "exern_iv30": 30.},
    ]}


def test_implied_inputs_selects_evaluation_events_and_horizons_only():
    sink = {}
    _implied_inputs(_fold(), _inputs(), (1, 3), sink)
    assert set(sink) == {("e1", 1), ("e1", 3), ("e2", 1)}


def test_implied_inputs_ignores_training_rows():
    sink = {}
    _implied_inputs(_fold(), _inputs(), (1, 3), sink)
    assert not any(event == "t1" for event, _ in sink)


def test_implied_inputs_copies_only_the_two_fields_and_preserves_none():
    sink = {}
    _implied_inputs(_fold(), _inputs(), (1, 3), sink)
    assert sink[("e1", 1)] == {"iv30": 50., "exern_iv30": 40.}
    assert sink[("e1", 3)] == {"iv30": None, "exern_iv30": 41.}
    assert sink[("e2", 1)] == {"iv30": 60., "exern_iv30": 0.}


def test_implied_inputs_accumulates_across_calls_without_clearing():
    sink = {("prior", 1): {"iv30": 1., "exern_iv30": 2.}}
    _implied_inputs(_fold(), _inputs(), (5,), sink)
    assert set(sink) == {("prior", 1), ("e1", 5)}
    assert sink[("e1", 5)] == {"iv30": 52., "exern_iv30": 42.}


CHALLENGER = [{"event_id": "E1", "horizon_sessions": 1, "predicted_crush_pct_iv30": -10.0}]
INCUMBENT = [{"event_id": "E1", "horizon_sessions": 1, "predicted_crush_pct_iv30": -8.0,
              "abstention": None, "training_mean": -60.0}]
DISTRIBUTION = {"schema_version": "iv_distribution.v1", "status": "ok", "marker": 1}


def _plan():
    return {"plan_id": "p", "horizons": (1,),
            "folds": [{"outer": {"year": "2025", "evaluation_event_ids": ["E1"]}}],
            "request": {"eligibility_policy_ref": "x"}}


def _registration():
    event = {"event_id": "E1", "ticker": "T", "event_date": "2025-03-03"}
    return SimpleNamespace(run_id="r", document={"regression_plan": {"events": [event]}})


def _fake_outer(calls):
    """Record whether a callback was passed; for v2 feed one fold's verified inputs through it."""
    def fake(*args, input_callback=None):
        calls.append(input_callback is None)
        if input_callback is not None:
            input_callback({"evaluation_event_ids": ["E1"]},
                           {"features": [{"event_id": "E1", "horizon_sessions": 1,
                                          "iv30": 50.0, "exern_iv30": 40.0}]})
        return {"predictions": [{"event_id": "E1", "horizon_sessions": 1,
                                 "crush_pct_iv30": -12.0, "outer_year": "2025"}],
                "folds": [{"outer_year": "2025", "candidate": {"candidate_id": "c"}}]}
    return fake


def _read_record(conn, store, run_id, slot):
    if slot.startswith("outer-forecasts:"):
        return CHALLENGER
    if slot.startswith("comparison-terminal:"):
        return {"result": {"forecasts": INCUMBENT, "fit_details": []}}
    raise AssertionError(slot)


def _patch_collect(monkeypatch, calls):
    monkeypatch.setattr(search_report, "validate_registration", lambda document: None)
    monkeypatch.setattr(search_report, "_admit_driver", lambda *a, **k: _plan())
    monkeypatch.setattr(search_report, "collect_outer_results", _fake_outer(calls))
    monkeypatch.setattr(search_report, "read_trial_arrays", lambda *a, **k: {})
    monkeypatch.setattr(search_report, "outer_arrays", lambda *a, **k: {})
    monkeypatch.setattr(search_report, "collect_distribution", lambda *a, **k: dict(DISTRIBUTION))
    monkeypatch.setattr(search_report, "read_record", _read_record)
    monkeypatch.setattr(search_report, "_candidate_evidence", lambda *a, **k: [])
    monkeypatch.setattr(search_report, "resource_report", lambda *a, **k: {})


def test_collect_v2_request_passes_callback_and_emits_paired_report_v2(monkeypatch):
    calls = []
    _patch_collect(monkeypatch, calls)
    report = search_report.collect(None, None, None, _registration(),
                                   {"schema_version": "regression_search_request.v2"}, "ref")
    assert calls == [False]  # callback was passed
    paired = report["paired"]
    assert paired["schema_version"] == "iv_crush_paired_report.v2"
    assert paired["challenger_distribution"]["marker"] == 1
    implied = paired["arm_metrics"]["implied_ex_earnings"]
    assert implied is not None
    # implied = 100*(40/50-1) = -20; truth -12; one row => bias -8, count 1
    assert implied["count"] == 1
    assert implied["bias"] == pytest.approx(-8.0)
    assert report["schema_version"] == "iv_crush_search_report.v1"
    assert report["distribution"]["marker"] == 1


def test_collect_v1_request_passes_no_callback_and_emits_paired_report_v1(monkeypatch):
    calls = []
    _patch_collect(monkeypatch, calls)
    report = search_report.collect(None, None, None, _registration(),
                                   {"schema_version": "regression_search_request.v1"}, "ref")
    assert calls == [True]  # callback is None
    assert report["paired"]["schema_version"] == "iv_crush_paired_report.v1"
