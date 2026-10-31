"""Target-aware planning over real synthetic native catalogs, without outcomes."""
from copy import deepcopy

import pytest

from engine.v2.foundation import content_hash
from engine.v2.models.regression_target import target_descriptor, validate_target
from engine.v2.research.magnitude_folds import plan_magnitude_folds, plan_regression_folds
from experiments import iv_crush_plan
from tests.data_scan_support import commit_tables
from tests.ops_support import FakeClock
from tests.v2.research.magnitude_trial_support import history_event
from tests.v2.research.test_magnitude_folds import magnitude_fold_fixture

REQUEST = dict(as_of_month="2026-01", evaluation_years=["2024", "2025"], training_start="2023-12-10")


def test_real_planning_without_either_outcome_table_preserves_legacy_clocks(tmp_path, monkeypatch):
    events = [history_event(day) for day in ("2023-12-12", "2024-01-03", "2025-01-07")]
    events.append(history_event("2023-12-13", protected=True))
    conn, repository, original = magnitude_fold_fixture(tmp_path, events)
    try:
        names = ("earnings_events", "daily_market")
        snapshot = commit_tables(conn, FakeClock(),
            {name: repository.fragment_records(original, name) for name in names},
            {name: repository.table_contract(original, name) for name in names},
            scope="metadata-only", receipt_id="metadata-only", attempt_id="metadata-only")
        assert set(snapshot.table_versions) == set(names)
        before = conn.total_changes
        for method in ("scan", "fragment_records", "table_contract"):
            native = getattr(repository, method)
            def guarded(*args, native=native, **kwargs):
                name = kwargs.get("table_name", args[1] if len(args) > 1 else None)
                assert name in names
                return native(*args, **kwargs)
            monkeypatch.setattr(repository, method, guarded)
        legacy = plan_magnitude_folds(repository, snapshot, **REQUEST)
        result = iv_crush_plan.plan(repository, snapshot, **REQUEST)
        expected = {**legacy, "schema_version": "regression_temporal_plan.v1",
                    "target_recipe": iv_crush_plan.target()}
        expected["plan_id"] = content_hash({k: v for k, v in expected.items() if k != "plan_id"})
        assert result == expected
        assert result == iv_crush_plan.plan(repository, snapshot, **REQUEST)
        assert result["plan_id"] != legacy["plan_id"]
        assert result["horizons"] == [5, 3, 1]
        assert len(result["folds"]) == 2
        assert events[-1]["event_id"] not in [e["event_id"] for e in result["events"]]
        assert conn.total_changes == before
        assert all(name in names for name, _ in repository.scans)
    finally:
        conn.close()


@pytest.mark.parametrize("change", [
    {"domain": "nonnegative"}, {"column": "other_label"},
    {"prediction_column": "other_prediction"}, {"definition_id": "sha256:" + "1" * 64},
])
def test_every_semantic_change_changes_plan_identity(tmp_path, change):
    conn, repository, snapshot = magnitude_fold_fixture(tmp_path)
    try:
        target = iv_crush_plan.target()
        args = {k: target[k] for k in ("column", "prediction_column", "domain", "definition_id")}
        other = target_descriptor(**{**args, **change})
        request = {**REQUEST, "evaluation_years": ["2024"]}
        left = plan_regression_folds(repository, snapshot, target=target, **request)
        right = plan_regression_folds(repository, snapshot, target=other, **request)
        assert left["events"] == right["events"] and left["folds"] == right["folds"]
        assert left["plan_id"] != right["plan_id"]
        assert left["target_recipe"]["target_id"] != right["target_recipe"]["target_id"]
        target["domain"] = "nonnegative"
        assert left["target_recipe"]["domain"] == "signed"
    finally:
        conn.close()


@pytest.mark.parametrize("change", [
    {"extra": 1}, {"schema_version": "unknown"}, {"target_id": "sha256:" + "0" * 64},
    {"domain": "positive"}, {"domain": True}, {"column": "event_id"}, {"column": "values"},
    {"column": "has space"}, {"column": "predicted_crush_pct_iv30"},
    {"prediction_column": "horizon_sessions"}, {"definition_id": "not-a-hash"},
    {"definition_id": "sha256:" + "A" * 64}, {"column": None},
])
def test_invalid_or_tampered_target_refuses_before_repository_access(change):
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError("repository touched")
    with pytest.raises(ValueError):
        plan_regression_folds(Poison(), Poison(), target={**iv_crush_plan.target(), **change}, **REQUEST)


def test_closed_descriptor_round_trip_and_independent_identity():
    target = iv_crush_plan.target()
    assert validate_target(deepcopy(target)) == target
    assert validate_target(target) is not target
    assert target["target_id"] == content_hash({k: v for k, v in target.items() if k != "target_id"})
    assert target["column"] == "crush_pct_iv30" and target["domain"] == "signed"
    for value in (None, [], {k: v for k, v in target.items() if k != "domain"}):
        with pytest.raises(ValueError):
            validate_target(value)
