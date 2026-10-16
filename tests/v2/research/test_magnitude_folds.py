"""Temporal metadata plans over real synthetic catalogs; every outcome is poison."""
from dataclasses import replace

import pytest

from engine.v2.data import objects
from engine.v2.data.errors import DataError, fail
from engine.v2.foundation import content_hash
from engine.v2.research import magnitude_folds
from tests.data_scan_support import commit_tables
from tests.ops_support import FakeClock
from tests.v2.research.magnitude_trial_support import (
    calendar_sessions,
    commit_history_inputs,
    history_event,
    outcome_move,
    poison_outcomes,
)


def magnitude_fold_fixture(tmp_path, events=None, **changes):
    """Commit actual native metadata and deliberately unusable outcome payloads."""
    events = events if events is not None else [
        history_event("2023-12-12"), {**history_event("2024-01-03"), "session": "BMO"}, history_event("2024-01-04")]
    unique = {(event["ticker"], event["event_date"]): event for event in events}
    return commit_history_inputs(tmp_path, events, [outcome_move(event, None) for event in unique.values()],
                   **{"days": calendar_sessions("2023-12-01", "2025-01-10").days, **changes})


def plan_test_magnitude_folds(repository, snapshot, **changes):
    """The caller explicitly declares all temporal request fields."""
    return magnitude_folds.plan_magnitude_folds(repository, snapshot, **{
        "as_of_month": "2026-01", "evaluation_years": ["2024"],
        "training_start": "2023-12-10", **changes})


def forbid_fold_outcomes(monkeypatch, repository, snapshot):
    """Poison physical outcome verification and even outcome-table planning."""
    opened = poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
    for name in ("fragment_records", "table_contract"):
        original = getattr(repository, name)
        def guarded(snap, table_name, *args, original=original, **kwargs):
            assert table_name != "computed_moves", "outcome planning is forbidden"
            return original(snap, table_name, *args, **kwargs)
        monkeypatch.setattr(repository, name, guarded)
    return opened


def test_plan_groups_all_horizons_and_removes_january_training_leak(tmp_path, monkeypatch):
    train, on_boundary, late = (history_event(day) for day in ("2023-12-12", "2023-12-22", "2023-12-26"))
    first, second = {**history_event("2024-01-03"), "session": "BMO"}, history_event("2024-06-12")
    next_year = {**history_event("2025-01-07"), "session": "BMO"}
    random, rolling = history_event("2023-12-13", protected=True), history_event("2025-12-03")
    alias = history_event("2024-01-04")
    sibling = {**alias, "event_id": "conflicting-identity", "session": "BMO"}
    before, future = history_event("2023-11-01"), history_event("2026-02-02")
    events = [train, on_boundary, late, first, second, next_year, random, rolling, alias, before, future]
    conn, repository, snapshot = magnitude_fold_fixture(tmp_path, [*events, sibling],
        mixed_days=("2023-12-12", "2023-12-13"))
    try:
        forbid_fold_outcomes(monkeypatch, repository, snapshot)
        changes = conn.total_changes
        result = plan_test_magnitude_folds(repository, snapshot, evaluation_years=["2024", "2025"])
        assert result == plan_test_magnitude_folds(repository, snapshot, evaluation_years=("2024", "2025"))
        assert {key: value for key, value in result.items()
                if key not in ("plan_id", "events", "folds", "population_input", "calendar_input")} == {
            "schema_version": "magnitude_temporal_plan.v1", "snapshot_id": snapshot.snapshot_id,
            "holdout_as_of_month": "2026-01", "random_membership_version": "canonical-event-sha256.v1",
            "rolling_membership_version": "calendar-months.v1", "training_start": "2023-12-10",
            "evaluation_years": ["2024", "2025"], "horizons": [5, 3, 1],
            "target_recipe": {"schema_version": "earnings_magnitude.v1", "units": "percentage_points",
                "target": "absolute_earnings_close_to_close_return",
                "price_basis": "split_neutral_non_dividend_adjusted", "estimand": "conditional_mean",
                "primary_loss": "pooled_rmse", "secondary_diagnostics": ["mae", "quantile_calibration"]},
            "fold_policy": {"kind": "expanding_event_year", "group_key": "event_id",
                "fit_boundary": "min_year_start_and_evaluation_forecast_ceilings",
                "training_label_availability": "strictly_before_fit_boundary_required_later",
                "issuer_overlap": "allowed_across_time", "inner_search": "not_defined"},
            "availability": {"basis": "reconstructed_date_bounds", "event_schedule_as_known": "unverified",
                             "source_publication": "unverified"}}
        assert result["plan_id"] == content_hash({k: v for k, v in result.items() if k != "plan_id"})
        ids = [r["event_id"] for r in result["events"]]
        assert ids == [e["event_id"] for e in (train, on_boundary, late, first, second, next_year)]
        early, later = result["folds"]
        assert early == {"year": "2024", "training_labels_before": "2023-12-22",
            "training_candidate_event_ids": [train["event_id"]],
            "evaluation_event_ids": [first["event_id"], second["event_id"]]}
        assert later == {"year": "2025", "training_labels_before": "2024-12-27",
            "training_candidate_event_ids": [e["event_id"] for e in (train, on_boundary, late, first, second)],
            "evaluation_event_ids": [next_year["event_id"]]}
        for event in result["events"]:
            assert [f["horizon_sessions"] for f in event["forecasts"]] == [5, 3, 1]
            assert all(f["event_date"] == event["event_date"] and f["session"] == event["session"]
                       for f in event["forecasts"])
        assert all(q.snapshot_id == snapshot.snapshot_id for _, q in repository.scans)
        assert {name for name, _ in repository.scans} == {"earnings_events", "daily_market"}
        assert all(list(q.columns) == result["population_input"]["columns"]
                   for name, q in repository.scans if name == "earnings_events")
        assert all(q.columns == ("date",) for name, q in repository.scans if name == "daily_market")
        for table, field in (("earnings_events", "population_input"), ("daily_market", "calendar_input")):
            expected = {"table": table, "dataset_version": snapshot.table_versions[table].dataset_version_id,
                "columns": ["date"] if table == "daily_market" else [
                    "event_id", "ticker", "event_date", "session", "date_conflict", "event_cluster_id"],
                "objects": [{"fragment_id": record.fragment_id, "object_id": record.object_ref.object_id,
                             "content_hash": record.object_ref.content_hash}
                            for record in repository.fragment_records(snapshot, table)]}
            if table == "daily_market":
                expected.update(snapshot_id=snapshot.snapshot_id, calendar_version=snapshot.calendar_version)
            assert result[field] == expected
        assert conn.total_changes == changes
        assert not list(tmp_path.rglob("REPORT.md")) and not list(tmp_path.rglob("LEDGER.csv"))
    finally:
        conn.close()


def test_midyear_evaluation_keeps_january_fit_boundary_and_candidate_is_not_a_label(tmp_path, monkeypatch):
    train, late = history_event("2023-12-12"), history_event("2023-12-29")
    test = history_event("2024-06-12")
    conn, repository, snapshot = magnitude_fold_fixture(tmp_path, [train, late, test])
    try:
        forbid_fold_outcomes(monkeypatch, repository, snapshot)
        result = plan_test_magnitude_folds(repository, snapshot)
        fold = result["folds"][0]
        assert fold["training_labels_before"] == "2024-01-01"
        assert fold["training_candidate_event_ids"] == [train["event_id"], late["event_id"]]
        assert result["fold_policy"]["training_label_availability"] == "strictly_before_fit_boundary_required_later"
        assert all("absolute_move_pct" not in event and "features" not in event for event in result["events"])
    finally:
        conn.close()


@pytest.mark.parametrize("changes", [
    {"evaluation_years": value} for value in (None, [], "2024", {"2024"}, [2024], [True],
        ["2024", "2024"], ["2025", "2024"], ["024"], ["２０２４"], ["0001"], ["9999"], [["2024"]])
] + [{"training_start": value} for value in (None, True, "2023-12-10T00:00:00", "2024-01-01", "bad")])
def test_malformed_declarations_refuse_before_repository_access(changes):
    """A sentinel cannot be queried; malformed requests must fail first."""
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        plan_test_magnitude_folds(object(), object(), **changes)


@pytest.mark.parametrize("kind", ["random", "rolling", "alias", "unknown", "snapshot", "window"])
def test_admission_and_explicit_window_refuse_before_calendar(tmp_path, monkeypatch, kind):
    target = history_event("2025-12-03" if kind == "rolling" else "2024-01-03", protected=kind == "random")
    events = [history_event("2023-12-12"), target]
    if kind == "alias":
        events.append({**target, "event_id": "alias", "session": "BMO"})
    conn, repository, snapshot = magnitude_fold_fixture(tmp_path, events)
    original = objects.verify_object_path
    blocked = {r.object_ref.object_id for table in ("daily_market", "computed_moves")
               for r in repository.fragment_records(snapshot, table)}
    def verify(store, ref):
        assert ref.object_id not in blocked, "admission must precede calendar/outcomes"
        return original(store, ref)
    monkeypatch.setattr(objects, "verify_object_path", verify)
    try:
        ids = ["unknown" if kind == "unknown" else target["event_id"]]
        changes = {"training_start": "2024-01-04", "evaluation_years": ["2025"]} if kind == "window" else {}
        if kind == "snapshot":
            snapshot = replace(snapshot, calendar_version="wrong")
        code = "SNAPSHOT_UNRESOLVED" if kind == "snapshot" else (
            "CONTRACT_MISMATCH" if kind == "window" else "HOLDOUT_ACCESS_DENIED")
        with pytest.raises(DataError, match=code):
            plan_test_magnitude_folds(repository, snapshot, event_ids=ids, **changes)
        assert all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()


@pytest.mark.parametrize("kind", ["empty_window", "no_train", "no_test", "missing_calendar", "calendar_gap"])
def test_empty_or_unprovable_fold_refuses_without_any_outcome_access(tmp_path, monkeypatch, kind):
    events = [history_event("2023-12-12"), history_event("2024-01-03")]
    changes, commit = {}, {}
    if kind == "empty_window":
        changes = {"training_start": "2025-01-01", "evaluation_years": ["2026"]}
    elif kind == "no_train":
        events = events[1:]
    elif kind == "no_test":
        changes = {"evaluation_years": ["2024", "2025"]}
    elif kind == "missing_calendar":
        commit = {"include_calendar": False}
    else:
        commit = {"days": tuple(day for day in calendar_sessions("2023-12-01", "2025-01-10").days
                                if day != "2023-12-29")}
    conn, repository, snapshot = magnitude_fold_fixture(tmp_path, events, **commit)
    try:
        forbid_fold_outcomes(monkeypatch, repository, snapshot)
        with pytest.raises(DataError, match="CONTRACT_MISMATCH" if "calendar" in kind else "POPULATION_COLLAPSED"):
            plan_test_magnitude_folds(repository, snapshot, **changes)
    finally:
        conn.close()


def test_request_context_and_calendar_object_bytes_change_plan_identity(tmp_path):
    plans = []
    for marker in (1.0, 2.0):
        conn, repository, snapshot = magnitude_fold_fixture(tmp_path / str(marker), marker=marker)
        try:
            base = plan_test_magnitude_folds(repository, snapshot)
            later = plan_test_magnitude_folds(repository, snapshot, as_of_month="2026-02")
            shorter = plan_test_magnitude_folds(repository, snapshot, training_start="2023-12-11")
            assert len({plan["plan_id"] for plan in (base, later, shorter)}) == 3
            plans.append(base)
        finally:
            conn.close()
    assert plans[0]["plan_id"] != plans[1]["plan_id"]
    assert plans[0]["events"] == plans[1]["events"] and plans[0]["folds"] == plans[1]["folds"]


def test_late_calendar_scan_failure_propagates_without_retry_or_partial_plan(tmp_path, monkeypatch):
    conn, repository, snapshot = magnitude_fold_fixture(tmp_path)
    original, attempts = repository.scan, []
    def scan(query, *, table_name):
        yield from original(query, table_name=table_name)
        if table_name == "daily_market":
            attempts.append(query)
            raise fail("OBJECT_CORRUPT", "synthetic late scan failure")
    monkeypatch.setattr(repository, "scan", scan)
    try:
        with pytest.raises(DataError, match="OBJECT_CORRUPT"):
            plan_test_magnitude_folds(repository, snapshot)
        assert len(attempts) == 1
    finally:
        conn.close()


def test_equivalent_explicit_population_order_and_call_local_clock_reuse(tmp_path, monkeypatch):
    train = history_event("2023-12-12")
    # Native announcement dates may be non-session days; this lies between the two deadlines.
    between = history_event("2023-12-25")
    first, second = history_event("2024-01-03", ticker="A"), history_event("2024-01-03", ticker="B")
    third = {**history_event("2024-01-03", ticker="C"), "session": "BMO"}
    conn, repository, snapshot = magnitude_fold_fixture(tmp_path, [third, second, train, first, between])
    original, calls = magnitude_folds.forecast_cutoff, []
    def clock(*args):
        calls.append(args[:3])
        return original(*args)
    monkeypatch.setattr(magnitude_folds, "forecast_cutoff", clock)
    try:
        forbid_fold_outcomes(monkeypatch, repository, snapshot)
        bulk = plan_test_magnitude_folds(repository, snapshot)
        assert len(calls) == 12  # Four unique date/session groups, three horizons each.
        selected = plan_test_magnitude_folds(repository, snapshot,
            event_ids=[event["event_id"] for event in (third, second, first, between, train)])
        assert len(calls) == 24  # There is no cross-call cache.
        assert bulk == selected
        assert bulk["folds"] == [{"year": "2024", "training_labels_before": "2023-12-22",
            "training_candidate_event_ids": [train["event_id"]],
            "evaluation_event_ids": [event["event_id"] for event in (first, second, third)]}]
        assert [{key: value for key, value in event.items() if key != "forecasts"}
                for event in bulk["events"]] == [
            {"event_id": event["event_id"], "ticker": event["ticker"],
             "event_date": event["event_date"].date().isoformat(), "session": event["session"],
             "population_use": "post-release selection"}
            for event in (train, between, first, second, third)]
        a, b, c = bulk["events"][2:]
        calendar = calendar_sessions("2023-12-01", "2025-01-10")
        calendar_id = content_hash({"days": list(calendar.days), "observed_through": calendar.days[-1]})
        expected_forecasts = {}
        for session, anchor, ceilings in (
            ("AMC", "2024-01-03", ("2023-12-26", "2023-12-28", "2024-01-02")),
            ("BMO", "2024-01-02", ("2023-12-22", "2023-12-27", "2023-12-29")),
        ):
            expected_forecasts[session] = []
            for horizon, ceiling in zip((5, 3, 1), ceilings):
                expected = {"schema_version": "earnings_forecast_cutoff.v1", "event_date": "2024-01-03",
                    "session": session, "horizon_sessions": horizon, "pre_announcement_close": anchor,
                    "source_session_ceiling": ceiling, "calendar_id": calendar_id}
                expected_forecasts[session].append({**expected, "cutoff_id": content_hash(expected)})
        assert a["forecasts"] == b["forecasts"] == expected_forecasts["AMC"]
        assert c["forecasts"] == expected_forecasts["BMO"]
        assert a["forecasts"][0] is not b["forecasts"][0]
    finally:
        conn.close()


def test_outcome_table_is_not_required_for_a_metadata_plan(tmp_path, monkeypatch):
    conn, repository, full = magnitude_fold_fixture(tmp_path)
    try:
        names = ("earnings_events", "daily_market")
        snapshot = commit_tables(conn, FakeClock(),
            {name: repository.fragment_records(full, name) for name in names},
            {name: repository.table_contract(full, name) for name in names},
            scope="metadata-only", receipt_id="metadata-only", attempt_id="metadata-only",
            store=repository._store)
        forbid_fold_outcomes(monkeypatch, repository, full)
        assert "computed_moves" not in snapshot.table_versions
        result = plan_test_magnitude_folds(repository, snapshot)
        assert result["snapshot_id"] == snapshot.snapshot_id and len(result["folds"]) == 1
    finally:
        conn.close()


def test_nonconsecutive_evaluation_years_do_not_drop_intervening_training_history(tmp_path, monkeypatch):
    events = [history_event(day) for day in ("2023-12-12", "2024-06-12", "2025-01-07", "2026-01-07")]
    conn, repository, snapshot = magnitude_fold_fixture(tmp_path, events,
        days=calendar_sessions("2023-12-01", "2026-01-08").days)
    try:
        forbid_fold_outcomes(monkeypatch, repository, snapshot)
        result = plan_test_magnitude_folds(repository, snapshot, as_of_month="2026-10", evaluation_years=["2024", "2026"])
        assert [fold["year"] for fold in result["folds"]] == ["2024", "2026"]
        assert result["folds"][1]["training_candidate_event_ids"] == [e["event_id"] for e in events[:-1]]
        assert events[2]["event_id"] not in sum([f["evaluation_event_ids"] for f in result["folds"]], [])
    finally:
        conn.close()


@pytest.mark.parametrize("session", ["AMC", "BMO"])
def test_non_session_announcement_preserves_native_date_and_anchor(tmp_path, session):
    events = [history_event("2023-12-12"), {**history_event("2024-01-01"), "session": session}]
    conn, repository, snapshot = magnitude_fold_fixture(tmp_path, events)
    try:
        event = plan_test_magnitude_folds(repository, snapshot)["events"][-1]
        assert event["event_date"] == "2024-01-01" and event["session"] == session
        assert all(f["pre_announcement_close"] == "2023-12-29" for f in event["forecasts"])
        assert event["forecasts"][0]["source_session_ceiling"] == "2023-12-21"
    finally:
        conn.close()


def test_request_window_includes_start_excludes_end_and_refuses_explicit_end_before_calendar(tmp_path, monkeypatch):
    """Both exact endpoints are admitted metadata, so membership follows the window alone."""
    start, test, end = [history_event(day) for day in ("2023-12-12", "2024-06-12", "2025-01-01")]
    conn, repository, snapshot = magnitude_fold_fixture(tmp_path, [start, test, end])
    try:
        forbid_fold_outcomes(monkeypatch, repository, snapshot)
        plan = plan_test_magnitude_folds(repository, snapshot, training_start="2023-12-12")
        assert [event["event_id"] for event in plan["events"]] == [start["event_id"], test["event_id"]]
        assert plan["folds"] == [{"year": "2024", "training_labels_before": "2024-01-01",
            "training_candidate_event_ids": [start["event_id"]], "evaluation_event_ids": [test["event_id"]]}]
        blocked = {record.object_ref.object_id for record in repository.fragment_records(snapshot, "daily_market")}
        original = objects.verify_object_path
        def verify(store, ref):
            assert ref.object_id not in blocked, "explicit end-boundary refusal must precede calendar access"
            return original(store, ref)
        monkeypatch.setattr(objects, "verify_object_path", verify)
        repository.scans.clear()
        with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
            plan_test_magnitude_folds(repository, snapshot, training_start="2023-12-12", event_ids=[end["event_id"]])
        assert repository.scans and all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()
