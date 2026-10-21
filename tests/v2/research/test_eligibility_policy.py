"""Metadata-only shared admission: no caller-supplied membership is trusted."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace

import pytest

from engine.v2.data.errors import DataError
from engine.v2.foundation import content_hash, to_document
from engine.v2.research import eligibility_policy as eligibility
from tests.v2.research.magnitude_trial_support import (
    calendar_sessions,
    commit_history_inputs,
    history_event,
    outcome_move,
    poison_outcomes,
)

MONTH = "2026-01"
VERSION = "grouped_ticker_embargo.v1"


@contextmanager
def refuses(code, message):
    with pytest.raises(DataError) as caught:
        yield
    assert caught.value.code == code
    assert caught.value.problem.message == message


def group(event, *, start=None, end=None, decision=None, source=None):
    day = event["event_date"].date().isoformat()
    return {"event_id": event["event_id"], "ticker": event["ticker"], "event_date": day,
        "label_start": start or day, "label_end": end or day,
        "forecasts": [{"horizon_sessions": h, "decision_date": decision or day,
                       "source_session_ceiling": source or decision or day} for h in (1, 3, 5)]}


def seal(snapshot, policy, groups, **changes):
    identity = {"schema_version": "event_group_bounds.v1", "source_snapshot": to_document(snapshot),
        "holdout_as_of_month": MONTH, "policy_id": policy["policy_id"],
        "horizons": [1, 3, 5], "groups": deepcopy(groups), **changes}
    return {**identity, "metadata_id": content_hash(identity)}


def setup(tmp_path, monkeypatch, events):
    conn, repository, snapshot = commit_history_inputs(tmp_path, events,
        [outcome_move(e, None) for e in events], days=calendar_sessions().days)
    opened = poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
    original = repository.scan
    def metadata_only(query, *, table_name):
        assert table_name == "earnings_events"
        assert tuple(query.columns) == (
            "event_id", "ticker", "event_date", "session", "date_conflict", "event_cluster_id")
        yield from original(query, table_name=table_name)
    monkeypatch.setattr(repository, "scan", metadata_only)
    return conn, repository, snapshot, opened


def plan(repository, snapshot, groups, *, evaluation=None, days=30, **changes):
    policy = eligibility.eligibility_policy(version=VERSION, embargo_days=days)
    return eligibility.plan_eligibility(repository, snapshot, as_of_month=MONTH,
        policy=policy, metadata=seal(snapshot, policy, groups),
        evaluation_event_ids=[] if evaluation is None else evaluation, **changes)


def test_inclusive_boundaries_ticker_isolation_and_full_group_oracle(tmp_path, monkeypatch):
    held = history_event("2024-06-14", protected=True)
    before = history_event("2024-05-14")  # day -31
    left = history_event("2024-05-15")  # day -30
    right = history_event("2024-07-14")  # day +30
    after = history_event("2024-07-15")  # day +31
    other = history_event("2024-06-14", ticker="OTHER")
    rolling = history_event("2025-12-01")
    events = [held, before, left, right, after, other, rolling]
    conn, repository, snapshot, opened = setup(tmp_path, monkeypatch, events)
    try:
        groups = [group(e) for e in reversed(events)]
        saved, changes = deepcopy(groups), conn.total_changes
        result = plan(repository, snapshot, groups, evaluation=[held["event_id"], right["event_id"]])
        assert result["retained_event_ids"] == ["OTHER:2024-06-14:0", "SYNTH:2024-05-14:0", "SYNTH:2024-07-15:0"]
        full = ["embargo_decision_date", "embargo_event_window", "embargo_label_window", "embargo_source_session_ceiling"]
        assert result["exclusions"] == [
            {"event_id": left["event_id"], "reasons": full},
            {"event_id": held["event_id"], "reasons": ["random", *full]},
            {"event_id": right["event_id"], "reasons": full},
            {"event_id": rolling["event_id"], "reasons": ["rolling"]}]
        assert result["random_windows"] == [{"ticker": "SYNTH", "start": "2024-05-15", "end": "2024-07-14"}]
        assert result["evaluation_event_ids"] == sorted([held["event_id"], right["event_id"]])
        assert result["coverage"] == {"development_groups": 7, "retained_groups": 3,
                                      "excluded_groups": 4, "evaluation_groups": 2}
        assert result["groups"] == sorted(groups, key=lambda g: g["event_id"])
        expected_policy = eligibility.eligibility_policy(version=VERSION, embargo_days=30)
        expected = {"schema_version": "group_eligibility_plan.v1", "snapshot_id": snapshot.snapshot_id,
            "holdout_as_of_month": "2026-01", "random_membership_version": "canonical-event-sha256.v1",
            "rolling_membership_version": "calendar-months.v1", "source_snapshot": to_document(snapshot),
            "policy": expected_policy, "metadata_id": seal(snapshot, expected_policy, saved)["metadata_id"],
            "groups": sorted(saved, key=lambda g: g["event_id"]),
            "random_windows": [{"ticker": "SYNTH", "start": "2024-05-15", "end": "2024-07-14"}],
            "retained_event_ids": ["OTHER:2024-06-14:0", "SYNTH:2024-05-14:0", "SYNTH:2024-07-15:0"],
            "exclusions": [{"event_id": left["event_id"], "reasons": full},
                {"event_id": held["event_id"], "reasons": ["random", *full]},
                {"event_id": right["event_id"], "reasons": full},
                {"event_id": rolling["event_id"], "reasons": ["rolling"]}],
            "evaluation_event_ids": sorted([held["event_id"], right["event_id"]]),
            "coverage": {"development_groups": 7, "retained_groups": 3, "excluded_groups": 4, "evaluation_groups": 2}}
        assert result == {**expected, "plan_id": content_hash(expected)}
        assert result == plan(repository, snapshot, groups, evaluation=[right["event_id"], held["event_id"]])
        assert groups == saved and conn.total_changes == changes
        result["groups"][0]["forecasts"].clear()
        assert groups == saved  # detached result, not a mutable capability
        assert len(opened) == 4  # two ticker metadata fragments, once per call
        assert [name for name, _ in repository.scans] == ["earnings_events"] * 4
    finally:
        conn.close()


def test_policy_literal_identity_and_no_default():
    policy = eligibility.eligibility_policy(version=VERSION, embargo_days=30)
    assert policy == {"schema_version": VERSION, "embargo_days": 30, "endpoints": "inclusive",
        "group_key": "event_id", "window_key": "ticker", "applies_to": "development",
        "evaluation": "preserve_membership", "policy_id": "sha256:dff629fde866872115a22f90305909b9f7176451439a3cc8f27e3e884c22376c"}
    with pytest.raises(TypeError):
        eligibility.eligibility_policy(version=VERSION)


@pytest.mark.parametrize("days", [-1, True, 30.0, float("nan"), float("inf"), None, "30"])
def test_bad_policy_width(days):
    assert eligibility.eligibility_policy(version=VERSION, embargo_days=0)["embargo_days"] == 0
    with refuses("CONTRACT_MISMATCH", "explicit supported policy and nonnegative integer days required"):
        eligibility.eligibility_policy(version=VERSION, embargo_days=days)


def test_unknown_version():
    assert eligibility.eligibility_policy(version=VERSION, embargo_days=30)["schema_version"] == VERSION
    with refuses("CONTRACT_MISMATCH", "explicit supported policy and nonnegative integer days required"):
        eligibility.eligibility_policy(version="v2", embargo_days=30)


def test_windows_merge_and_zero_width(tmp_path, monkeypatch):
    held = [history_event(day, protected=True) for day in ("2024-06-14", "2024-07-01", "2024-11-01")]
    dev = history_event("2024-07-31")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [*held, dev])
    try:
        result = plan(repository, snapshot, [group(dev)])
        assert result["random_windows"] == [
            {"ticker": "SYNTH", "start": "2024-05-15", "end": "2024-07-31"},
            {"ticker": "SYNTH", "start": "2024-10-02", "end": "2024-12-01"}]
        assert result["retained_event_ids"] == []
        zero = plan(repository, snapshot, [group(dev)], days=0)
        assert zero["random_windows"] == [{"ticker": "SYNTH", "start": day, "end": day}
            for day in ("2024-06-14", "2024-07-01", "2024-11-01")]
        assert zero["retained_event_ids"] == [dev["event_id"]]
        assert zero["policy"]["policy_id"] != result["policy"]["policy_id"]
    finally:
        conn.close()


@pytest.mark.parametrize("field,reason", [("label_start", "embargo_label_window"),
    ("decision_date", "embargo_decision_date"), ("source_session_ceiling", "embargo_source_session_ceiling")])
def test_one_horizon_or_label_overlap_excludes_whole_event(tmp_path, monkeypatch, field, reason):
    held, dev = history_event("2024-06-14", protected=True), history_event("2024-07-15")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [held, dev])
    try:
        row = group(dev)
        if field == "label_start":
            row[field] = "2024-07-14"
        elif field == "decision_date":
            row["forecasts"][2][field] = "2024-07-14"
            row["forecasts"][2]["source_session_ceiling"] = "2024-05-14"
        else:
            row["forecasts"][2][field] = "2024-07-14"
        result = plan(repository, snapshot, [row], evaluation=[dev["event_id"]])
        assert result["exclusions"] == [{"event_id": dev["event_id"], "reasons": [reason]}]
        assert result["retained_event_ids"] == [] and result["evaluation_event_ids"] == [dev["event_id"]]
        assert result["groups"] == [row]
    finally:
        conn.close()


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(unknown=True), lambda m: m.update(schema_version="v2"),
    lambda m: m.update(holdout_as_of_month="2025-12"), lambda m: m.update(policy_id="sha256:wrong"),
    lambda m: m["source_snapshot"].update(calendar_version="stale"),
    lambda m: m.update(horizons=[1, 1, 5]), lambda m: m.update(horizons=[True, 3, 5]),
    lambda m: m["groups"].append(deepcopy(m["groups"][0])),
    lambda m: m["groups"][0].pop("label_start"),
    lambda m: m["groups"][0].update(label_start="2024-07-01"),
    lambda m: m["groups"][0].update(label_end="2024-05-01"),
    lambda m: m["groups"][0].update(label_end="2024-06-14T00:00:00Z"),
    lambda m: m["groups"][0]["forecasts"].pop(),
    lambda m: m["groups"][0]["forecasts"][0].update(horizon_sessions=True),
    lambda m: m["groups"][0]["forecasts"][0].update(decision_date="2024-06-15"),
    lambda m: m["groups"][0]["forecasts"][0].update(source_session_ceiling="2024-06-15"),
    lambda m: m["groups"][0].update(random_held=False),
])
def test_closed_metadata_refuses_before_scans(tmp_path, monkeypatch, mutation):
    event = history_event("2024-06-14")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        policy = eligibility.eligibility_policy(version=VERSION, embargo_days=30)
        metadata = seal(snapshot, policy, [group(event)])
        assert plan(repository, snapshot, [group(event)])["retained_event_ids"] == [event["event_id"]]
        repository.scans.clear()
        mutation(metadata)
        metadata["metadata_id"] = content_hash({k: v for k, v in metadata.items() if k != "metadata_id"})
        with refuses("CONTRACT_MISMATCH", "invalid or stale eligibility policy/group metadata"):
            eligibility.plan_eligibility(repository, snapshot, as_of_month=MONTH, policy=policy,
                                        metadata=metadata, evaluation_event_ids=[])
        assert repository.scans == []
    finally:
        conn.close()


def test_hash_and_snapshot_refusal(tmp_path, monkeypatch):
    event = history_event("2024-06-14")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        policy = eligibility.eligibility_policy(version=VERSION, embargo_days=30)
        metadata = seal(snapshot, policy, [group(event)])
        assert plan(repository, snapshot, [group(event)])["retained_event_ids"] == [event["event_id"]]
        repository.scans.clear()
        metadata["metadata_id"] = "sha256:stale"
        with refuses("CONTRACT_MISMATCH", "invalid or stale eligibility policy/group metadata"):
            eligibility.plan_eligibility(repository, snapshot, as_of_month=MONTH, policy=policy,
                                        metadata=metadata, evaluation_event_ids=[])
        with refuses("SNAPSHOT_UNRESOLVED", "snapshot differs from its committed identity"):
            plan(repository, replace(snapshot, calendar_version="stale"), [group(event)])
        assert repository.scans == []
    finally:
        conn.close()


@pytest.mark.parametrize("field,value", [("ticker", "WRONG"), ("event_date", "2024-06-13"),
                                        ("event_id", "missing")])
def test_canonical_population_not_caller_claim(tmp_path, monkeypatch, field, value):
    event = history_event("2024-06-14", protected=True)
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        row = group(event, start="2024-06-01", end="2024-06-30", decision="2024-06-01")
        valid = plan(repository, snapshot, [row])
        assert valid["exclusions"][0]["event_id"] == event["event_id"]
        row[field] = value
        with refuses("HOLDOUT_ACCESS_DENIED", "group identity differs from canonical population"):
            plan(repository, snapshot, [row])
    finally:
        conn.close()


@pytest.mark.parametrize("case", ["ambiguous", "evaluation", "overflow"])
def test_independent_window_and_evaluation_refusals(tmp_path, monkeypatch, case):
    event = history_event("2024-06-14", protected=True)
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        assert plan(repository, snapshot, [], evaluation=[event["event_id"]])["evaluation_event_ids"] == [event["event_id"]]
        if case == "ambiguous":
            original = eligibility.canonical_events
            def ambiguous(repository, snapshot):
                rows = original(repository, snapshot)
                rows.loc[0, "date_conflict"] = True
                return rows
            monkeypatch.setattr(eligibility, "canonical_events", ambiguous)
        with refuses("HOLDOUT_ACCESS_DENIED", ("evaluation membership cannot be resolved" if case == "evaluation" else "random holdout window bounds cannot be established")):
            plan(repository, snapshot, [], evaluation=["missing"] if case == "evaluation" else [],
                 days=10**20 if case == "overflow" else 30)
    finally:
        conn.close()


@pytest.mark.parametrize("value", [None, "event", ["x", "x"], [True], [" padded"]])
def test_malformed_evaluation_ids_are_not_silently_dropped(tmp_path, monkeypatch, value):
    event = history_event("2024-06-14")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        assert plan(repository, snapshot, [group(event)])["retained_event_ids"] == [event["event_id"]]
        repository.scans.clear()
        policy = eligibility.eligibility_policy(version=VERSION, embargo_days=30)
        with refuses("CONTRACT_MISMATCH", "explicit unique evaluation event IDs required"):
            eligibility.plan_eligibility(repository, snapshot, as_of_month=MONTH, policy=policy,
                metadata=seal(snapshot, policy, [group(event)]), evaluation_event_ids=value)
        assert repository.scans == []
    finally:
        conn.close()


def test_policy_hash_tampering_and_unknown_fields(tmp_path, monkeypatch):
    event = history_event("2024-06-14")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        assert plan(repository, snapshot, [group(event)])["retained_event_ids"] == [event["event_id"]]
        repository.scans.clear()
        original = eligibility.eligibility_policy(version=VERSION, embargo_days=30)
        for policy in ({**original, "policy_id": "wrong"}, {**original, "unknown": True}, None):
            with refuses("CONTRACT_MISMATCH", "invalid or stale eligibility policy/group metadata"):
                eligibility.plan_eligibility(repository, snapshot, as_of_month=MONTH, policy=policy,
                    metadata=seal(snapshot, original, [group(event)]), evaluation_event_ids=[])
        assert repository.scans == []
    finally:
        conn.close()


def test_canonical_collision_refuses(tmp_path, monkeypatch):
    event = history_event("2024-06-14")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        assert plan(repository, snapshot, [group(event)])["retained_event_ids"] == [event["event_id"]]
        original = eligibility.canonical_events
        def collision(repository, snapshot):
            import pandas as pd
            rows = original(repository, snapshot)
            return pd.concat([rows, rows], ignore_index=True)
        monkeypatch.setattr(eligibility, "canonical_events", collision)
        with refuses("HOLDOUT_ACCESS_DENIED", "canonical event identities collide or are invalid"):
            plan(repository, snapshot, [group(event)])
    finally:
        conn.close()


def test_malformed_canonical_date_is_typed_refusal(tmp_path, monkeypatch):
    event = history_event("2024-06-14")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        assert plan(repository, snapshot, [group(event)])["retained_event_ids"] == [event["event_id"]]
        original = eligibility.canonical_events
        def invalid(repository, snapshot):
            import pandas as pd
            rows = original(repository, snapshot)
            rows.loc[0, "event_date"] = pd.NaT
            return rows
        monkeypatch.setattr(eligibility, "canonical_events", invalid)
        with refuses("HOLDOUT_ACCESS_DENIED", "group identity differs from canonical population"):
            plan(repository, snapshot, [group(event)])
    finally:
        conn.close()


def test_future_random_metadata_has_window_but_never_development_label(tmp_path, monkeypatch):
    future = history_event("2026-02-01", protected=True)
    dev = history_event("2024-06-14")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [future, dev])
    try:
        result = plan(repository, snapshot, [group(dev), group(future)])
        assert result["random_windows"] == [{"ticker": "SYNTH", "start": "2026-01-02", "end": "2026-03-03"}]
        assert result["retained_event_ids"] == [dev["event_id"]]
        assert result["exclusions"] == [{"event_id": future["event_id"], "reasons": ["ambiguous", "random",
            "embargo_decision_date", "embargo_event_window", "embargo_label_window", "embargo_source_session_ceiling"]}]
    finally:
        conn.close()


@pytest.mark.parametrize("kind", ["unknown", "missing", "blank", "wrong_type", "subclass"])
def test_snapshot_refusals_are_typed_and_precede_scans(tmp_path, monkeypatch, capsys, kind):
    event = history_event("2024-06-14")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        assert plan(repository, snapshot, [group(event)])["retained_event_ids"] == [event["event_id"]]
        repository.scans.clear()
        capsys.readouterr()
        class ForeignSnapshot(type(snapshot)):
            pass
        invalid = {"unknown": replace(snapshot, snapshot_id="missing"), "missing": None,
            "blank": replace(snapshot, snapshot_id=" "), "wrong_type": replace(snapshot, snapshot_id=[]),
            "subclass": ForeignSnapshot(**snapshot.__dict__)}[kind]
        policy = eligibility.eligibility_policy(version=VERSION, embargo_days=30)
        with refuses("SNAPSHOT_UNRESOLVED", ("the requested snapshot cannot be resolved" if kind == "unknown" else "an exact native snapshot reference is required")):
            eligibility.plan_eligibility(repository, invalid, as_of_month=MONTH, policy=policy,
                metadata=seal(snapshot, policy, [group(event)]), evaluation_event_ids=[])
        assert repository.scans == []
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "refused SNAPSHOT_UNRESOLVED" in captured.err
        assert "peak RSS" in captured.err
    finally:
        conn.close()


def test_refusal_telemetry_and_membership_cadence(tmp_path, monkeypatch, capsys):
    event = history_event("2024-06-14")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        meters = []
        real_progress = eligibility.progress.Progress
        def meter(name, total):
            ticks = iter([0, 31, 32])
            result = real_progress(name, total, clock=lambda: next(ticks))
            meters.append(result)
            return result
        monkeypatch.setattr(eligibility.progress, "Progress", meter)
        assert plan(repository, snapshot, [group(event)])["retained_event_ids"] == [event["event_id"]]
        membership = [m for m in meters if m.name == "eligibility membership"]
        assert len(membership) == 1 and membership[0].done == 1 and membership[0].lines == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "[eligibility membership] 1/1, 31s elapsed" in captured.err
        assert "ETA 0s" in captured.err
        row = group(event)
        row["forecasts"][0]["source_session_ceiling"] = "2024-06-15"
        with refuses("CONTRACT_MISMATCH", "invalid or stale eligibility policy/group metadata"):
            plan(repository, snapshot, [row])
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "refused CONTRACT_MISMATCH" in captured.err
        assert event["event_id"] not in captured.err
    finally:
        conn.close()


def test_public_metadata_helpers_detach_and_refuse_invalid_contracts(tmp_path, monkeypatch):
    import pandas as pd

    event = history_event("2024-06-14", protected=True)
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        events = eligibility.canonical_events(repository, snapshot)
        saved = events.copy(deep=True)
        canonical, labels, ambiguous, context = eligibility.classify_events(events, snapshot, as_of_month=MONTH)
        assert labels == {event["event_id"]: {"random"}} and ambiguous == set()
        assert context == {"snapshot_id": snapshot.snapshot_id, "holdout_as_of_month": MONTH,
            "random_membership_version": "canonical-event-sha256.v1",
            "rolling_membership_version": "calendar-months.v1"}
        expected = saved[["event_id", "ticker", "event_date"]].copy()
        expected["canonical_date"] = saved["event_date"]
        expected["canonical_ticker"] = saved["ticker"]
        for name in ("date_conflict", "event_cluster_id", "session"):
            expected[name] = saved[name]
        pd.testing.assert_frame_equal(canonical, expected, check_like=True)
        canonical.loc[0, "ticker"] = "changed"
        pd.testing.assert_frame_equal(events, saved)
        for invalid in (None, replace(snapshot, snapshot_id=[]), replace(snapshot, snapshot_id=" ")):
            with refuses("SNAPSHOT_UNRESOLVED", "an exact native snapshot reference is required"):
                eligibility.classify_events(events, invalid, as_of_month=MONTH)
            with refuses("SNAPSHOT_UNRESOLVED", "an exact native snapshot reference is required"):
                eligibility.canonical_events(repository, invalid)
        for invalid in (None, events.drop(columns="event_id"), pd.concat([events, events], axis=1)):
            with refuses("CONTRACT_MISMATCH", "canonical event metadata columns required"):
                eligibility.classify_events(invalid, snapshot, as_of_month=MONTH)
    finally:
        conn.close()


@pytest.mark.parametrize("field,value", [("snapshot_id", "unknown"), ("calendar_version", "stale"),
    ("manifest_hash", "sha256:stale"), ("table_versions", {})])
def test_public_reader_requires_exact_commit_before_scanning(tmp_path, monkeypatch, field, value):
    event = history_event("2024-06-14")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        assert eligibility.canonical_events(repository, snapshot).event_id.tolist() == [event["event_id"]]
        repository.scans.clear()
        message = ("the requested snapshot cannot be resolved" if field == "snapshot_id"
                   else "snapshot differs from its committed identity")
        with refuses("SNAPSHOT_UNRESOLVED", message):
            eligibility.canonical_events(repository, replace(snapshot, **{field: value}))
        assert repository.scans == []
    finally:
        conn.close()


@pytest.mark.parametrize("code", ["OBJECT_CORRUPT", "MANIFEST_CORRUPT", "CONTRACT_MISMATCH"])
def test_admission_preserves_native_integrity_problem(tmp_path, monkeypatch, code):
    from engine.v2.data import errors

    event = history_event("2024-06-14")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        assert plan(repository, snapshot, [group(event)])["retained_event_ids"] == [event["event_id"]]
        assert eligibility.canonical_events(repository, snapshot).event_id.tolist() == [event["event_id"]]
        repository.scans.clear()
        original = errors.fail(code, "native repository refusal", stage="resolve", details={"sentinel": True})
        def refuse(_):
            raise original
        monkeypatch.setattr(repository, "resolve", refuse)
        for call in (lambda: eligibility.canonical_events(repository, snapshot),
                     lambda: plan(repository, snapshot, [group(event)])):
            with pytest.raises(DataError) as caught:
                call()
            assert caught.value is original
            assert caught.value.problem.message == "native repository refusal"
            assert caught.value.problem.details == {"sentinel": True}
        assert repository.scans == []
    finally:
        conn.close()


@pytest.mark.parametrize("case", ["event_id", "cluster", "session"])
def test_classifier_structural_duplicates_are_not_future_membership(tmp_path, monkeypatch, case):
    import pandas as pd

    events = [history_event("2024-06-14"), history_event("2024-07-15")]
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, events)
    try:
        frame = eligibility.canonical_events(repository, snapshot)
        _, labels, structural, _ = eligibility.classify_events(frame, snapshot, as_of_month=MONTH)
        assert labels == {} and structural == set()
        if case == "event_id":
            frame = pd.concat([frame.iloc[:1], frame.iloc[:1]], ignore_index=True)
        elif case == "cluster":
            frame["event_cluster_id"] = "same-cluster"
        else:
            frame["event_date"] = frame.iloc[0].event_date
            frame["session"] = "AMC"
        expected_ids = set(frame.event_id)
        canonical, labels, structural, _ = eligibility.classify_events(frame, snapshot, as_of_month=MONTH)
        assert structural == expected_ids
        assert labels == {key: {"ambiguous"} for key in expected_ids}
        assert canonical.event_id.tolist() == frame.event_id.tolist()
    finally:
        conn.close()


@pytest.mark.parametrize("horizons", [[], [0, 3, 5], [-1, 3, 5], [True, 3, 5], [1, 1, 5], [5, 3, 1]])
def test_horizon_declaration_guard_independent_of_matching_forecasts(tmp_path, monkeypatch, horizons):
    event = history_event("2024-06-14")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        assert plan(repository, snapshot, [group(event)])["retained_event_ids"] == [event["event_id"]]
        repository.scans.clear()
        policy = eligibility.eligibility_policy(version=VERSION, embargo_days=30)
        row = group(event)
        template = row["forecasts"][0]
        row["forecasts"] = [{**template, "horizon_sessions": h} for h in horizons]
        # Empty groups isolate envelope guards from all forecast guards.
        assert plan(repository, snapshot, [])["groups"] == []
        empty = seal(snapshot, policy, [], horizons=horizons)
        with refuses("CONTRACT_MISMATCH", "invalid or stale eligibility policy/group metadata"):
            eligibility.plan_eligibility(repository, snapshot, as_of_month=MONTH, policy=policy,
                                        metadata=empty, evaluation_event_ids=[])
        repository.scans.clear()
        metadata = seal(snapshot, policy, [row], horizons=horizons)
        with refuses("CONTRACT_MISMATCH", "invalid or stale eligibility policy/group metadata"):
            eligibility.plan_eligibility(repository, snapshot, as_of_month=MONTH, policy=policy,
                                        metadata=metadata, evaluation_event_ids=[])
        assert repository.scans == []
    finally:
        conn.close()


@pytest.mark.parametrize("month,expected", [("2025-12", {"rolling"}), ("2026-01", set())])
def test_asof_month_changes_both_public_classification_paths(tmp_path, monkeypatch, month, expected):
    event = history_event("2025-07-15")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        frame = eligibility.canonical_events(repository, snapshot)
        _, labels, structural, context = eligibility.classify_events(frame, snapshot, as_of_month=month)
        assert labels == ({event["event_id"]: expected} if expected else {})
        assert structural == set() and context["holdout_as_of_month"] == month
        policy = eligibility.eligibility_policy(version=VERSION, embargo_days=30)
        metadata = seal(snapshot, policy, [group(event)], holdout_as_of_month=month)
        result = eligibility.plan_eligibility(repository, snapshot, as_of_month=month, policy=policy,
                                             metadata=metadata, evaluation_event_ids=[])
        assert result["holdout_as_of_month"] == month
        assert result["exclusions"] == ([{"event_id": event["event_id"], "reasons": ["rolling"]}] if expected else [])
        assert result["retained_event_ids"] == ([] if expected else [event["event_id"]])
    finally:
        conn.close()


@pytest.mark.parametrize("month", [None, "2026-13", "9999-01"])
def test_invalid_asof_month_refuses_both_paths(tmp_path, monkeypatch, month):
    event = history_event("2024-06-14")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        frame = eligibility.canonical_events(repository, snapshot)
        assert eligibility.classify_events(frame, snapshot, as_of_month=MONTH)[1] == {}
        assert plan(repository, snapshot, [group(event)])["retained_event_ids"] == [event["event_id"]]
        policy = eligibility.eligibility_policy(version=VERSION, embargo_days=30)
        metadata = seal(snapshot, policy, [group(event)], holdout_as_of_month=month)
        for call in (lambda: eligibility.classify_events(frame, snapshot, as_of_month=month),
                     lambda: eligibility.plan_eligibility(repository, snapshot, as_of_month=month,
                                policy=policy, metadata=metadata, evaluation_event_ids=[])):
            with refuses("HOLDOUT_ACCESS_DENIED", "explicit valid holdout context is required"):
                call()
    finally:
        conn.close()


def test_malformed_classifier_values_refuse_without_builtin_exception(tmp_path, monkeypatch):
    import pandas as pd

    event = history_event("2024-06-14")
    conn, repository, snapshot, _ = setup(tmp_path, monkeypatch, [event])
    try:
        frame = eligibility.canonical_events(repository, snapshot)
        assert eligibility.classify_events(frame, snapshot, as_of_month=MONTH)[1] == {}
        frame["event_id"] = pd.Series([["not", "scalar"]], dtype=object)
        with refuses("CONTRACT_MISMATCH", "canonical event metadata values are malformed"):
            eligibility.classify_events(frame, snapshot, as_of_month=MONTH)
    finally:
        conn.close()
