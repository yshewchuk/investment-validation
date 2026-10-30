"""Non-strict target planning in the eligible-history backend."""
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data import errors as data_errors
from engine.v2.research import eligible_history as backend


def _selected(event_ids):
    return pd.DataFrame([{"event_id": key, "ticker": "AAAA", "session": "AMC",
                          "event_date": pd.Timestamp("2024-01-05")} for key in event_ids])


def _call(strict, monkeypatch):
    monkeypatch.setattr(backend, "_anchors", lambda *args, **kwargs: {})
    monkeypatch.setattr(backend, "_bound", lambda *args: None)
    repository = SimpleNamespace(fragment_records=lambda snapshot, table: [])
    return backend._groups(repository, object(), _selected(["ev-1", "ev-2"]), {}, ["ev-1"],
                           [1, 3, 5], None, {}, {}, strict=strict)


def test_nonstrict_groups_records_unavailable_target(monkeypatch):
    groups, unavailable, qualified = _call(False, monkeypatch)
    assert groups == [] and qualified == {}
    assert unavailable == ["ev-1", "ev-2"]


def test_strict_groups_refuses_unavailable_target(monkeypatch):
    with pytest.raises(data_errors.DataError) as raised:
        _call(True, monkeypatch)
    assert raised.value.code == "CONTRACT_MISMATCH"


def test_plan_targets_intersects_requests_without_materializing(monkeypatch):
    captured = {}

    def fake_plan(repository, snapshot, **kwargs):
        captured.update(kwargs)
        return {"eligibility": {"plan_id": "plan_1",
                                "retained_event_ids": ["ev-b", "contributor"],
                                "exclusions": [{"event_id": "ev-a", "reasons": ["embargo_ticker"]},
                                               {"event_id": "absent", "reasons": ["other"]}]},
                "unavailable": ["ev-c", "ghost"], "start": date(2020, 1, 1),
                "population": pd.DataFrame([
                    {"event_id": "ev-a", "event_date": pd.Timestamp("2024-01-05")},
                    {"event_id": "ev-b", "event_date": pd.Timestamp("2024-01-08")},
                    {"event_id": "ev-c", "event_date": pd.Timestamp("2024-01-10")}])}

    monkeypatch.setattr(backend, "_plan", fake_plan)
    monkeypatch.setattr(backend, "_materialize",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("materialized")))
    result = backend.plan_targets(object(), object(), as_of_month="2024-01", policy={},
                                  event_ids=["ev-a", "ev-b", "ev-c"], horizons=[1, 3, 5],
                                  history_start="2020-01-01", limits={})
    assert captured["strict"] is False
    assert result == {"plan_id": "plan_1", "retained": ["ev-b"], "unavailable": ["ev-c"],
                      "exclusions": [{"event_id": "ev-a", "reasons": ["embargo_ticker"]}],
                      "before_start": []}


def test_plan_targets_reports_events_before_history_start(monkeypatch):
    def fake_plan(repository, snapshot, **kwargs):
        return {"eligibility": {"plan_id": "plan_1", "retained_event_ids": ["ev-in"],
                                "exclusions": [{"event_id": "ev-x", "reasons": ["embargo_ticker"]}]},
                "unavailable": [], "start": date(2024, 1, 1),
                "population": pd.DataFrame([
                    {"event_id": "ev-before", "ticker": "AAAA", "event_date": pd.Timestamp("2023-12-31")},
                    {"event_id": "ev-in", "ticker": "AAAA", "event_date": pd.Timestamp("2024-01-05")},
                    {"event_id": "ev-x", "ticker": "AAAA", "event_date": pd.Timestamp("2024-01-15")}])}

    monkeypatch.setattr(backend, "_plan", fake_plan)
    monkeypatch.setattr(backend, "_materialize",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("materialized")))
    result = backend.plan_targets(object(), object(), as_of_month="2024-01", policy={},
                                  event_ids=["ev-before", "ev-in", "ev-x"], horizons=[1, 3, 5],
                                  history_start="2024-01-01", limits={})
    assert result["before_start"] == ["ev-before"]


def test_plan_targets_reports_requested_events_before_history_start(monkeypatch):
    def fake_plan(repository, snapshot, **kwargs):
        return {"eligibility": {"plan_id": "plan_1", "retained_event_ids": [], "exclusions": []},
                "unavailable": [], "start": date(2020, 1, 1),
                "population": pd.DataFrame([
                    {"event_id": "ev-before", "event_date": pd.Timestamp("2019-06-01")},
                    {"event_id": "ev-early", "event_date": pd.Timestamp("2019-12-31")},
                    {"event_id": "ev-onstart", "event_date": pd.Timestamp("2020-01-01")},
                    {"event_id": "ev-after", "event_date": pd.Timestamp("2024-01-05")},
                    {"event_id": "unrequested-old", "event_date": pd.Timestamp("2018-01-01")}])}

    monkeypatch.setattr(backend, "_plan", fake_plan)
    result = backend.plan_targets(object(), object(), as_of_month="2024-01", policy={},
                                  event_ids=["ev-before", "ev-early", "ev-onstart", "ev-after"],
                                  horizons=[1, 3, 5], history_start="2020-01-01", limits={})
    assert result["before_start"] == ["ev-before", "ev-early"]
    assert "unrequested-old" not in result["before_start"]
