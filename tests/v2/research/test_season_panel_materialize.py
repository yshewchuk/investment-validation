# ruff: noqa: F811
"""Season heat panel wiring: published columns, persisted-value oracle, replay, identity and refusal."""
import json
from datetime import date, timedelta

import pytest

from engine.v2.data.errors import DataError
from engine.v2.foundation import content_hash
from engine.v2.research import _magnitude_panel_store as store
from engine.v2.research import magnitude_panel_qualified as builder
from engine.v2.research import magnitude_panel_qualified_table as table
from engine.v2.research import season_heat
from experiments import size_panel
from tests.v2.research.magnitude_trial_support import calendar_sessions
from tests.v2.research.test_magnitude_panel_qualified import panel_data, read  # noqa: F401

CALENDAR = ("2022-05-20", "2024-06-20")
ALL_COLUMNS = [*table.FEATURES_HMOS, "has_implied_quote"]
MEANS = ("season_heat_mean_abs_5", "season_heat_mean_abs_20", "season_heat_mean_abs_60")


def stored_rows(data, artifact):
    """Every stored v4 column of the published panel, bypassing the public projection gate."""
    return store.read(data.repository, artifact["feature_snapshot"], artifact["context"], ALL_COLUMNS)


def heat_targets(context):
    """The context's deduplicated (event, decision clock) targets as compute_season_heat wants them."""
    return list({(target["event_id"], cutoff["source_session_ceiling"]):
        {"event_id": target["event_id"], "ticker": target["ticker"], "event_date": target["event_date"],
         "decision_session": cutoff["source_session_ceiling"]}
        for target in context["target_events"] for cutoff in target["forecasts"]}.values())


def synthetic_labels(sessions, targets):
    """Six labels on each of the twenty sessions before the earliest decision clock, in window.

    Six per session, not two, so the 5-session window of the earliest decision clock still holds
    4 x 6 = 24 labels and mean_abs_5 is non-null somewhere (the mean threshold is 20 labels).
    """
    start = max(0, min(sessions.index(target["decision_session"]) for target in targets) - 20)
    return [{"event_id": f"synthetic-{offset}-{ticker}", "ticker": ticker,
             "event_date": sessions[start + offset], "available_as_of_date": sessions[start + offset + 1],
             "abs_move": 1.0 + offset / 10}
            for offset in range(len(sessions[start:start + 20]))
            for ticker in ("LAB", "LAC", "LAD", "LAE", "LAF", "LAG")]


def prepare(data):
    return builder._prepare(data.repository, data.panel_source, producer=size_panel._producer({}),
        history_admitted=size_panel._history(data.repository, data.panel_source, data.panel_request),
        **data.panel_request)[0]


def test_published_rows_carry_season_columns_and_receipt_lineage(panel_data):
    data = panel_data
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    receipt = artifact["context"]["season_heat_population"]
    assert receipt["schema_version"] == "season_heat_population.v1"
    assert receipt["receipt_id"] == content_hash({k: v for k, v in receipt.items() if k != "receipt_id"})
    for row in stored_rows(data, artifact):
        assert all(name in row for name in table.SEASON)
        assert type(row["season_heat_count_20"]) is int and row["season_heat_count_20"] >= 0
        assert all(row[name] is None or row[name] >= 0. for name in MEANS)
        assert json.loads(row["lineage"])["season_heat"] == {"receipt_id": receipt["receipt_id"]}
    assert [json.loads(row["lineage"])["season_heat"] for row in read(data, artifact)["rows"]] \
        == [{"receipt_id": receipt["receipt_id"]}] * 3


def test_stored_season_columns_equal_an_independent_heat_recompute(panel_data, monkeypatch):
    data = panel_data
    sessions = calendar_sessions(*CALENDAR).days
    labels = synthetic_labels(sessions, heat_targets(prepare(data)))
    assert len(labels) >= 25
    calls = []
    monkeypatch.setattr(builder.season_population, "read_labels",
                        lambda *args, **kwargs: calls.append(kwargs) or labels)
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    assert len(calls) == 1  # one population read for the whole build
    assert calls[0]["latest_decision"] \
        == artifact["context"]["season_heat_population"]["window"]["latest_decision"]
    expected = season_heat.compute_season_heat(labels, heat_targets(artifact["context"]), sessions)
    rows = stored_rows(data, artifact)
    assert all(row["season_heat_count_20"] >= 20 for row in rows)  # labels reach the mean threshold
    assert any(row["season_heat_ratio_20_252"] is not None for row in rows)
    assert any(row["season_heat_mean_abs_5"] is not None for row in rows)
    assert any(row["season_heat_mean_abs_60"] is not None for row in rows)
    assert any(row["season_heat_mean_abs_5"] is None for row in rows)  # latest clock: 0 in its 5-window
    for row in rows:
        assert {name: row[name] for name in table.SEASON} \
            == expected[(row["event_id"], row["source_session_ceiling"])]


def test_stored_season_heat_equals_hand_computed_values(panel_data, monkeypatch):
    """Independent numeric oracle: no compute_season_heat call, only the labels and their windows.

    Uniform 2.0 moves make every populated mean exactly 2.0 and the 20/252 ratio exactly 1.0.
    """
    data = panel_data
    sessions = calendar_sessions(*CALENDAR).days
    decisions = sorted({sessions.index(target["decision_session"])
                        for target in heat_targets(prepare(data))})
    assert decisions == [515, 517, 519]  # horizons 1, 3, 5 of the fixture's single 2024-06-17 event
    # The three 20-session windows are [496,515], [498,517], [500,519], so sessions 500..514
    # (decisions[-1] - 19 up to decisions[0] - 1) are inside every window and strictly before the
    # earliest clock. Two off-target tickers there = (515 - 500) * 2 = 30 labels, each available the
    # next calendar day, at latest 2024-06-07 + 1 day = 2024-06-08 < the 2024-06-10 first clock.
    lo, per_session = decisions[-1] - 19, 2
    expected_count_20 = (decisions[0] - lo) * per_session  # (515 - 500) * 2 = 30
    labels = [{"event_id": f"oracle-{index}-{ticker}", "ticker": ticker, "event_date": sessions[index],
               "available_as_of_date": (date.fromisoformat(sessions[index])
                                        + timedelta(days=1)).isoformat(), "abs_move": 2.0}
              for index in range(lo, decisions[0]) for ticker in ("LAB", "LAC")]
    assert len(labels) == expected_count_20 == 30 >= 20  # 20 labels is the mean threshold
    monkeypatch.setattr(builder.season_population, "read_labels", lambda *args, **kwargs: labels)
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    rows = stored_rows(data, artifact)
    assert len(rows) == 3
    for row in rows:
        # mean_abs_5 is null everywhere: its window holds only [511,515] -> 4 x 2 = 8 labels at the
        # 2024-06-10 clock, 2 x 2 = 4 at 2024-06-12 and 0 at 2024-06-14, all below the 20 threshold.
        # mean_abs_20 and mean_abs_60 both hold all 30 labels: 60.0 / 30 = 2.0 exactly.
        assert row["season_heat_count_20"] == expected_count_20
        assert [row[name] for name in MEANS] == [None, 2.0, 2.0]
        assert row["season_heat_ratio_20_252"] == 1.0  # all 30 labels are within 252 sessions too


def test_replay_returns_the_same_artifact_without_reading_labels_again(panel_data, monkeypatch):
    data = panel_data
    original = builder.season_population.read_labels
    calls = []

    def counted(repository, snapshot, roster, **kwargs):
        calls.append(roster)
        return original(repository, snapshot, roster, **kwargs)

    monkeypatch.setattr(builder.season_population, "read_labels", counted)
    first = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    assert len(calls) == 1
    second = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    assert second == first and len(calls) == 1
    assert read(data, second) == read(data, first)


def test_roster_digest_change_moves_the_context_identity(panel_data, monkeypatch):
    data = panel_data
    first = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    plan_roster = builder.season_population.plan_roster

    def changed(repository, snapshot, **kwargs):
        planned = plan_roster(repository, snapshot, **kwargs)
        planned["receipt"]["roster_digest"] = "sha256:" + "f" * 64
        return planned

    monkeypatch.setattr(builder.season_population, "plan_roster", changed)
    second = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    receipt = second["context"]["season_heat_population"]
    assert receipt["roster_digest"] != first["context"]["season_heat_population"]["roster_digest"]
    identity = {k: v for k, v in second["context"].items() if k != "context_id"}
    assert identity == {**{k: v for k, v in first["context"].items() if k != "context_id"},
        "season_heat_population": receipt}  # the roster digest is the only context change
    assert content_hash(identity) == second["context"]["context_id"]
    assert second["context"]["context_id"] != first["context"]["context_id"]


@pytest.mark.parametrize("column,value", [
    ("season_heat_count_20", -1), ("season_heat_count_20", 1.0), ("season_heat_count_20", True),
    ("season_heat_mean_abs_20", 4.), ("season_heat_mean_abs_5", -1.),
    ("season_heat_mean_abs_60", float("inf")), ("season_heat_ratio_20_252", 1.5),
    ("season_heat_ratio_20_252", "1.5")])
def test_store_refuses_tampered_season_columns(panel_data, column, value):
    data = panel_data
    artifact = size_panel.materialize(data.repository, data.panel_source, **data.panel_request)
    rows = stored_rows(data, artifact)
    assert all(row["season_heat_mean_abs_20"] is None and row["season_heat_count_20"] == 0 for row in rows)
    rows[0][column] = value
    rows[0]["row_id"] = content_hash({k: v for k, v in rows[0].items() if k != "row_id"})
    with pytest.raises(DataError) as refused:
        list(store._validate(rows, artifact["context"]))
    assert refused.value.code == "CONTRACT_MISMATCH"
