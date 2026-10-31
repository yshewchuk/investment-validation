"""Non-strict target planning in the eligible-history backend."""
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data import errors as data_errors
from engine.v2.research import eligible_history as backend
from experiments import eligible_history as public
from tests.v2.research.test_eligible_history import source


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
    calls = {"groups": [], "eligibility": [], "load": []}

    def load(repository, snapshot, **kwargs):
        calls["load"].append(kwargs)
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


def _plan_targets(event_ids, history_start="2024-01-01", limits=None):
    return backend.plan_targets(object(), {}, as_of_month="2024-01", policy={"policy_id": "pol"},
                                event_ids=event_ids, horizons=[1, 3, 5],
                                history_start=history_start, limits=limits if limits else {})


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
                      "before_start": [], "dated_before_start": []}


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
                      "before_start": ["ev-early"], "dated_before_start": ["ev-early"]}
    groups_call = calls["groups"][0]
    assert groups_call["event_ids"] == ["ev-in", "ev-x"]
    assert "ev-early" not in groups_call["event_ids"]
    assert "ev-early" not in groups_call["selected"]["event_id"].tolist()
    assert len(calls["eligibility"]) == 1


def test_event_date_before_history_start_goes_to_before_start_despite_later_ceiling(monkeypatch):
    """Defensive: an artificial ceiling at or after history_start still cannot keep a pre-history event."""
    calls = _stub_plan(monkeypatch,
                       [{"event_id": "ev-before", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2023-12-10")},
                        {"event_id": "ev-in", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2024-01-05")}],
                       {"ev-before": "2024-01-20", "ev-in": "2024-01-25"},
                       retained=["ev-before", "ev-in"],
                       exclusions=[])
    result = _plan_targets(["ev-before", "ev-in"])
    assert result == {"plan_id": "plan_1", "retained": ["ev-in"], "unavailable": [],
                      "exclusions": [], "before_start": ["ev-before"],
                      "dated_before_start": ["ev-before"]}
    groups_call = calls["groups"][0]
    assert groups_call["event_ids"] == ["ev-in"]
    assert "ev-before" not in groups_call["selected"]["event_id"].tolist()
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
                      "before_start": ["ev-early"], "dated_before_start": ["ev-early"]}
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


def test_dated_before_history_start_target_never_reaches_forecasts(monkeypatch):
    """forecast_cutoff can raise CONTRACT_MISMATCH for old events outside the calendar; the
    non-strict planner must remove pre-history_start targets before deriving any forecast."""
    calls = _stub_plan(monkeypatch,
                       [{"event_id": "ev-old", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2020-06-10")},
                        {"event_id": "ev-in", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2024-01-05")}],
                       {"ev-old": "2020-06-20", "ev-in": "2024-01-20"},
                       retained=["ev-old", "ev-in"], exclusions=[])
    stub = backend._forecasts
    seen = []

    def guarded(event, horizons, calendar, clocks):
        seen.append(event.event_id)
        if event.event_id == "ev-old":
            raise data_errors.fail("CONTRACT_MISMATCH", "old event is outside the forecast calendar")
        return stub(event, horizons, calendar, clocks)

    monkeypatch.setattr(backend, "_forecasts", guarded)
    result = _plan_targets(["ev-old", "ev-in"])
    assert "ev-old" not in seen
    assert result == {"plan_id": "plan_1", "retained": ["ev-in"], "unavailable": [],
                      "exclusions": [], "before_start": ["ev-old"],
                      "dated_before_start": ["ev-old"]}
    assert calls["groups"][0]["event_ids"] == ["ev-in"]


def test_ceiling_early_target_dated_in_window_is_not_dated_before_start(monkeypatch):
    """A target inside the event window whose earliest ceiling precedes history_start is
    before_start but never dated_before_start."""
    calls = _stub_plan(monkeypatch,
                       [{"event_id": "ev-ceil", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2024-01-05")},
                        {"event_id": "ev-in", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2024-01-10")}],
                       {"ev-ceil": "2023-12-20", "ev-in": "2024-01-20"},
                       retained=["ev-ceil", "ev-in"], exclusions=[])
    result = _plan_targets(["ev-ceil", "ev-in"])
    assert result == {"plan_id": "plan_1", "retained": ["ev-in"], "unavailable": [],
                      "exclusions": [], "before_start": ["ev-ceil"], "dated_before_start": []}
    assert calls["groups"][0]["event_ids"] == ["ev-in"]


def test_nonstrict_plan_forwards_decoded_cap_to_reads_and_eligibility(monkeypatch):
    calls = _stub_plan(monkeypatch,
                       [{"event_id": "ev-a", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2024-01-05")}],
                       {"ev-a": "2024-01-12"}, retained=["ev-a"], exclusions=[])
    _plan_targets(["ev-a"], limits={"max_object_bytes": 4096})
    assert [kwargs["max_object_bytes"] for kwargs in calls["load"]] == [4096, 4096]
    assert calls["eligibility"][0]["max_object_bytes"] == 4096


def test_strict_plan_keeps_eligibility_cap_unset(monkeypatch):
    calls = _stub_plan(monkeypatch,
                       [{"event_id": "ev-a", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2024-01-05")}],
                       {"ev-a": "2024-01-12"}, retained=["ev-a"], exclusions=[])
    backend._plan(object(), {}, as_of_month="2024-01", policy={"policy_id": "pol"},
                  event_ids=["ev-a"], horizons=[1, 3, 5], history_start="2024-01-01",
                  limits={"max_object_bytes": 4096}, strict=True)
    assert [kwargs.get("max_object_bytes", "absent") for kwargs in calls["load"]] == ["absent", "absent"]
    assert calls["eligibility"][0]["max_object_bytes"] is None


def test_all_targets_dated_before_history_start_forecast_nothing(monkeypatch):
    calls = _stub_plan(monkeypatch,
                       [{"event_id": "ev-old-1", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2020-06-10")},
                        {"event_id": "ev-old-2", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2020-07-10")}],
                       {"ev-old-1": "2020-06-20", "ev-old-2": "2020-07-20"},
                       retained=["ev-old-1", "ev-old-2"], exclusions=[])

    def forbidden(event, horizons, calendar, clocks):
        raise AssertionError("forecasts derived for targets dated before history_start")

    monkeypatch.setattr(backend, "_forecasts", forbidden)
    result = _plan_targets(["ev-old-1", "ev-old-2"])
    assert result == {"plan_id": None, "retained": [], "unavailable": [], "exclusions": [],
                      "before_start": ["ev-old-1", "ev-old-2"],
                      "dated_before_start": ["ev-old-1", "ev-old-2"]}
    assert calls["groups"] == [] and calls["eligibility"] == []


def test_strict_plan_forecasts_every_target_including_dated_early(monkeypatch):
    calls = _stub_plan(monkeypatch,
                       [{"event_id": "ev-old", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2023-12-10")},
                        {"event_id": "ev-in", "ticker": "AAAA",
                         "event_date": pd.Timestamp("2024-01-05")}],
                       {"ev-old": "2024-01-20", "ev-in": "2024-01-25"},
                       retained=["ev-old", "ev-in"], exclusions=[])
    stub = backend._forecasts
    seen = []

    def recording(event, horizons, calendar, clocks):
        seen.append(event.event_id)
        return stub(event, horizons, calendar, clocks)

    monkeypatch.setattr(backend, "_forecasts", recording)
    plan = backend._plan(object(), {}, as_of_month="2024-01", policy={"policy_id": "pol"},
                         event_ids=["ev-old", "ev-in"], horizons=[1, 3, 5],
                         history_start="2024-01-01", limits={}, strict=True)
    assert sorted(seen) == ["ev-in", "ev-old"]
    assert plan["early"] == [] and plan["remaining"] is True
    assert sorted(plan["forecasts"]) == ["ev-in", "ev-old"]
    assert calls["groups"][0]["strict"] is True


# --- real planner over the synthetic native fixture (nothing stubbed) ---

@pytest.fixture
def real(tmp_path):
    data = source(tmp_path)
    yield data
    data.conn.close()


def _request(data, indices, **changes):
    return {**data.request, "event_ids": [data.events[i]["event_id"] for i in indices], **changes}


def test_real_plan_targets_match_materialized_targets_and_plan_id(real):
    request = _request(real, (1, 5, -1))
    artifact = public.materialize(real.repository, real.snapshot, **request)
    result = backend.plan_targets(real.repository, real.snapshot, **request)
    expected = sorted(t["event_id"] for t in artifact["context"]["targets"])
    assert expected == sorted(request["event_ids"])
    assert result["retained"] == expected
    assert result["unavailable"] == [] and result["before_start"] == []
    assert result["plan_id"] == artifact["context"]["eligibility"]["plan_id"]


def test_real_plan_targets_move_early_target_to_before_start(real):
    event_ids = [real.events[i]["event_id"] for i in (1, 5, -1)]
    result = backend.plan_targets(real.repository, real.snapshot,
                                  **_request(real, (1, 5, -1), history_start="2023-03-01"))
    assert result["before_start"] == [event_ids[0]]
    assert result["retained"] == sorted(event_ids[1:])
    assert result["unavailable"] == [] and result["plan_id"] is not None


@pytest.mark.parametrize("changes", [{"history_start": "bad"}, {"history_start": 5},
                                     {"as_of_month": "bad"}])
def test_real_plan_targets_refuse_noncanonical_start_or_month_without_reads(real, monkeypatch, changes):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid request reached the repository")
    monkeypatch.setattr(real.repository, "resolve", forbidden)
    monkeypatch.setattr(real.repository, "scan", forbidden)
    with pytest.raises(data_errors.DataError) as raised:
        backend.plan_targets(real.repository, real.snapshot, **_request(real, (1, 5, -1), **changes))
    assert raised.value.code == "CONTRACT_MISMATCH"


def test_real_plan_targets_date_object_matches_equal_iso_string(real):
    iso = _request(real, (1, 5, -1))
    as_date = {**iso, "history_start": date.fromisoformat(iso["history_start"])}
    assert backend.plan_targets(real.repository, real.snapshot, **as_date) == \
        backend.plan_targets(real.repository, real.snapshot, **iso)


@pytest.mark.parametrize("changes", [{"event_ids": []}, {"event_ids": ["ev", "ev"]},
                                     {"event_ids": ["ev", 1]}, {"event_ids": ("ev-1",)},
                                     {"event_ids": ["", " a", "a "]},
                                     {"horizons": [1, 3]}, {"horizons": [1, 3, 5, 1]}])
def test_real_plan_targets_refuse_malformed_targets_or_horizons_before_reads(real, monkeypatch, changes):
    def forbidden(*args, **kwargs):
        raise AssertionError("malformed request reached the source")
    monkeypatch.setattr(backend, "admit_source", forbidden)
    monkeypatch.setattr(real.repository, "resolve", forbidden)
    monkeypatch.setattr(real.repository, "scan", forbidden)
    with pytest.raises(data_errors.DataError) as raised:
        backend.plan_targets(real.repository, real.snapshot, **_request(real, (1, 5, -1), **changes))
    assert raised.value.code == "CONTRACT_MISMATCH"
    assert "explicit unique targets and ordered native horizons required" in str(raised.value)


def test_real_early_rule_is_strictly_below_the_exact_ceiling(real):
    """history_start equal to a target's earliest ceiling keeps it; one day later drops it."""
    request = _request(real, (1, 5, -1))
    artifact = public.materialize(real.repository, real.snapshot, **request)
    earliest = request["event_ids"][0]
    target = next(t for t in artifact["context"]["targets"] if t["event_id"] == earliest)
    ceiling = min(c["source_session_ceiling"] for c in target["forecasts"])
    exact = backend.plan_targets(real.repository, real.snapshot,
                                 **{**request, "history_start": ceiling})
    assert earliest not in exact["before_start"] and exact["plan_id"] is not None
    later = (date.fromisoformat(ceiling) + timedelta(days=1)).isoformat()
    after = backend.plan_targets(real.repository, real.snapshot,
                                 **{**request, "history_start": later})
    assert after["before_start"] == [earliest]
    assert after["plan_id"] is not None
