"""Non-strict target planning in the eligible-history backend."""
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


def _stub_plan(monkeypatch, population_rows, ceilings, *, retained, exclusions, unavailable=()):
    """Drive the real _plan with only its collaborators stubbed; never materialize."""
    population = pd.DataFrame(population_rows)
    calls = {"groups": [], "eligibility": []}

    def load(repository, snapshot, **kwargs):
        event_ids = kwargs.get("event_ids")
        if event_ids is None:
            return population.copy()
        return population[population.event_id.isin(event_ids)].copy()

    def forecasts(event, horizons, calendar, clocks):
        return [{"horizon_sessions": h, "cutoff_id": event.event_id + "-" + str(h),
                 "source_session_ceiling": ceilings[event.event_id]} for h in horizons]

    def groups(repository, snapshot, selected, event_forecasts, event_ids, horizons, calendar,
               clocks, limits, strict):
        calls["groups"].append({"event_ids": list(event_ids), "selected": selected, "strict": strict})
        return [], list(unavailable), {}

    def plan_eligibility(repository, snapshot, **kwargs):
        calls["eligibility"].append(kwargs)
        return {"plan_id": "plan_1", "retained_event_ids": list(retained), "exclusions": list(exclusions)}

    monkeypatch.setattr(backend, "_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "load_population", load)
    monkeypatch.setattr(backend, "calendar_input", lambda repository, snapshot: (None, "calendar-source"))
    monkeypatch.setattr(backend, "_forecasts", forecasts)
    monkeypatch.setattr(backend, "_groups", groups)
    monkeypatch.setattr(backend, "_materialize",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("materialized")))
    monkeypatch.setattr(backend.eligibility_policy, "plan_eligibility", plan_eligibility)
    return calls


def _plan_targets(event_ids, history_start="2024-01-01"):
    return backend.plan_targets(object(), {}, as_of_month="2024-01", policy={"policy_id": "pol"},
                                event_ids=event_ids, horizons=[1, 3, 5],
                                history_start=history_start, limits={})


def test_plan_targets_intersects_requests_without_materializing(monkeypatch):
    calls = _stub_plan(monkeypatch,
                       [{"event_id": "ev-a", "ticker": "AAAA", "event_date": pd.Timestamp("2024-01-05")},
                        {"event_id": "ev-b", "ticker": "AAAA", "event_date": pd.Timestamp("2024-01-08")},
                        {"event_id": "ev-c", "ticker": "AAAA", "event_date": pd.Timestamp("2024-01-10")}],
                       {"ev-a": "2024-01-12", "ev-b": "2024-01-15", "ev-c": "2024-01-18"},
                       retained=["ev-b", "contributor"],
                       exclusions=[{"event_id": "ev-a", "reasons": ["embargo_ticker"]},
                                   {"event_id": "absent", "reasons": ["other"]}],
                       unavailable=["ev-c", "ghost"])
    result = _plan_targets(["ev-a", "ev-b", "ev-c"])
    assert calls["groups"][0]["strict"] is False
    assert result == {"plan_id": "plan_1", "retained": ["ev-b"], "unavailable": ["ev-c"],
                      "exclusions": [{"event_id": "ev-a", "reasons": ["embargo_ticker"]}],
                      "before_start": []}


def test_ceiling_before_history_start_goes_to_before_start_not_window_refusal(monkeypatch):
    calls = _stub_plan(monkeypatch,
                       [{"event_id": "ev-early", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2023-12-10")},
                        {"event_id": "ev-in", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2024-01-05")},
                        {"event_id": "ev-x", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2024-01-15")}],
                       {"ev-early": "2023-12-15", "ev-in": "2024-01-20", "ev-x": "2024-01-25"},
                       retained=["ev-early", "ev-in"],
                       exclusions=[{"event_id": "ev-x", "reasons": ["embargo_ticker"]}])
    result = _plan_targets(["ev-early", "ev-in", "ev-x"])
    assert result == {"plan_id": "plan_1", "retained": ["ev-in"], "unavailable": [],
                      "exclusions": [{"event_id": "ev-x", "reasons": ["embargo_ticker"]}],
                      "before_start": ["ev-early"]}
    groups_call = calls["groups"][0]
    assert groups_call["event_ids"] == ["ev-in", "ev-x"]
    assert "ev-early" not in groups_call["event_ids"]
    assert "ev-early" not in groups_call["selected"]["event_id"].tolist()
    assert len(calls["eligibility"]) == 1


def test_all_early_targets_return_empty_plan_without_eligibility(monkeypatch):
    calls = _stub_plan(monkeypatch,
                       [{"event_id": "ev-early", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2023-11-20")},
                        {"event_id": "ev-in", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2024-01-05")}],
                       {"ev-early": "2023-11-30", "ev-in": "2024-01-20"},
                       retained=[], exclusions=[])
    result = _plan_targets(["ev-early"])
    assert result == {"plan_id": None, "retained": [], "unavailable": [], "exclusions": [],
                      "before_start": ["ev-early"]}
    assert calls["groups"] == [] and calls["eligibility"] == []


def test_strict_plan_still_refuses_ceiling_before_history_start(monkeypatch):
    calls = _stub_plan(monkeypatch,
                       [{"event_id": "ev-early", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2023-12-10")},
                        {"event_id": "ev-in", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2024-01-05")}],
                       {"ev-early": "2023-12-15", "ev-in": "2024-01-20"},
                       retained=[], exclusions=[])
    with pytest.raises(data_errors.DataError) as raised:
        backend._plan(object(), {}, as_of_month="2024-01", policy={"policy_id": "pol"},
                      event_ids=["ev-early", "ev-in"], horizons=[1, 3, 5],
                      history_start="2024-01-01", limits={}, strict=True)
    assert raised.value.code == "CONTRACT_MISMATCH"
    assert calls["groups"] == [] and calls["eligibility"] == []
