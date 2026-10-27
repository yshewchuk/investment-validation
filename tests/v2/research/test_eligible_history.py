"""Synthetic policy/physical-read, lineage, native math and replay oracles."""
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine.v2.data.computed_move_bounds import CONTRACT as BOUNDS_CONTRACT
from engine.v2.data.computed_move_bounds import TABLE as BOUNDS_TABLE
from engine.v2.data.errors import DataError
from engine.v2.foundation import SystemClock
from engine.v2.ops.computed_move_bounds_store import write_bounds
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.eligible_history_table import CONTRACT, FEATURES, TABLE, admit_fields
from engine.v2.research.magnitude_history_table import CONTRACT as LIVE_CONTRACT
from experiments import eligible_history as history
from experiments import magnitude_eligible_history as size
from tests.data_scan_support import commit_tables
from tests.v2.research.magnitude_trial_support import (
    calendar_sessions,
    catalog_rows,
    commit_history_inputs,
    history_event,
    outcome_move,
    poison_outcomes,
)

LIMITS = dict(max_source_rows=10000, max_fragments=100, max_object_bytes=1 << 20,
              max_total_bytes=16 << 20, max_output_rows=100)
POLICY = eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30)


def source(tmp_path, *, null_day="2023-05-10", changes=None, days=None, values=None, extra_ticker=False):
    days = days or ["2022-06-02", "2023-01-03", "2023-03-20", "2023-04-03", "2023-05-10", "2023-06-02", "2024-06-17"]
    events = [history_event(day, protected=day == "2023-04-03") for day in days]
    moves = [outcome_move(e, value) for e, value in zip(events, values or [-2., -4., 9999., 9999., 9999., 6., 8.])]
    if extra_ticker:
        extra = history_event("2023-06-02", ticker="OTHER")
        events.insert(0, extra)
        moves.insert(0, outcome_move(extra, 9.))
    for row in moves:
        if row["event_date"] == null_day:
            row.update(skipped=True, realized_move_pct=None, available_as_of_date=None)
    conn, repository, original = commit_history_inputs(tmp_path, events, moves,
        days=calendar_sessions("2022-05-20", "2024-06-20").days)
    records = repository.fragment_records(original, "computed_moves")
    dates = sorted({str(e["event_date"].date() + timedelta(days=n)) for e in events for n in (0, 1)})
    bounds = tuple(record for ticker in sorted({e["ticker"] for e in events})
        for record in write_bounds(repository._store, [r for r in moves if r["ticker"] == ticker],
            pd.DataFrame([e for e in events if e["ticker"] == ticker]),
            np.array(dates, dtype="datetime64[ns]"), records))
    parent = repository.resolve_full(original.snapshot_id)
    tables = {name: list(repository.fragment_records(original, name)) for name in original.table_versions}
    tables[BOUNDS_TABLE] = list(bounds)
    contracts = {name: repository.table_contract(original, name) for name in original.table_versions}
    contracts[BOUNDS_TABLE] = BOUNDS_CONTRACT
    if changes:
        changes(tables, contracts)
    snapshot = commit_tables(conn, SystemClock(), tables, contracts, scope="qualified",
        receipt_id="bounds-source", attempt_id="bounds-attempt", store=repository._store,
        partition_logical_hashes={name: m.partition_logical_hashes for name, m in parent.table_manifests.items()})
    request = dict(as_of_month="2026-01", policy=POLICY, event_ids=[events[-1]["event_id"]],
        horizons=[1, 3, 5], history_start="2022-01-01", limits=LIMITS)
    return SimpleNamespace(conn=conn, repository=repository, snapshot=snapshot, original=original,
                           events=events, request=request)


@pytest.fixture
def data(tmp_path):
    result = source(tmp_path)
    yield result
    result.conn.close()


def read(data, artifact, **changes):
    return history.read(data.repository, data.snapshot, artifact["feature_snapshot"],
        columns=list(FEATURES), **{**data.request, **changes})


def test_rebuild_does_not_read_protected_embargoed_null_or_target_outcomes(data, monkeypatch):
    poison_outcomes(monkeypatch, data.repository, data.snapshot,
        ["2023-03-20", "2023-04-03", "2023-05-10", "2024-06-17"])
    artifact = history.materialize(data.repository, data.snapshot, **data.request)
    result = read(data, artifact)
    assert set(artifact["feature_snapshot"].table_versions) == {TABLE}
    assert artifact["feature_snapshot"] != data.snapshot
    for row in result["rows"]:
        assert row["n_prior"] == 3
        assert row["mean_prior_move"] == 0
        assert row["mean_prior_abs_move"] == 4
        assert row["ema2_prior_move"] == pytest.approx(26 / 9)
        assert row["ema2_prior_abs_move"] == pytest.approx(46 / 9)
        assert all(row[f"ema{span}_prior_move"] is None for span in (4, 8, 12))
    context = artifact["context"]
    assert context["unavailable_contributors"] == [data.events[4]["event_id"]]
    assert context["eligibility"]["random_windows"] == [
        {"ticker": "SYNTH", "start": "2023-03-04", "end": "2023-05-03"}]
    assert context["availability"] == {"basis": "reconstructed_date_bounds",
        "source_publication": "unverified", "event_schedule_as_known": "unverified"}
    before = catalog_rows(data.conn)
    poison_outcomes(monkeypatch, data.repository, data.snapshot, (), all_outcomes=True)
    assert history.materialize(data.repository, data.snapshot, **data.request) == artifact
    assert read(data, artifact) == result
    assert catalog_rows(data.conn) == before


def test_null_protected_label_still_establishes_full_canonical_embargo(tmp_path, monkeypatch):
    data = source(tmp_path, null_day="2023-04-03")
    try:
        poison_outcomes(monkeypatch, data.repository, data.snapshot, ["2023-04-03", "2023-03-20"])
        result = history.materialize(data.repository, data.snapshot, **data.request)
        assert result["context"]["eligibility"]["random_windows"][0]["start"] == "2023-03-04"
        assert data.events[2]["event_id"] not in {r["event_id"] for r in result["context"]["contributors"]}
    finally:
        data.conn.close()


@pytest.mark.parametrize("change", [
    {"as_of_month": "2026-02"}, {"history_start": "2022-07-01"},
    {"policy": eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=31)},
    {"horizons": [1]},
])
def test_exact_context_replay_rejects_different_policy_month_start_or_cutoff(data, change):
    artifact = history.materialize(data.repository, data.snapshot, **data.request)
    with pytest.raises(DataError):
        read(data, artifact, **change)


def test_missing_sidecar_refuses_before_any_outcome(data, monkeypatch):
    poison_outcomes(monkeypatch, data.repository, data.original, (), all_outcomes=True)
    with pytest.raises(DataError, match="requires native source bounds"):
        history.materialize(data.repository, data.original, **data.request)


def test_partial_default_off_bounds_only_require_selected_tickers(tmp_path, monkeypatch):
    from engine.v2.ops.computed_move_bounds_store import compose
    data = source(tmp_path, extra_ticker=True)
    try:
        parent = data.repository.resolve_full(data.snapshot.snapshot_id)
        tables = {name: list(data.repository.fragment_records(data.snapshot, name))
            for name in data.snapshot.table_versions}
        contracts = {name: data.repository.table_contract(data.snapshot, name) for name in tables}
        # Actual default-off composition drops OTHER's bounds but retains its canonical metadata/outcome.
        tables[BOUNDS_TABLE] = list(compose(parent, {"OTHER"}, None, {}))
        partial = commit_tables(data.conn, SystemClock(), tables, contracts, scope="partial",
            receipt_id="partial", attempt_id="partial", store=data.repository._store,
            partition_logical_hashes={name: m.partition_logical_hashes for name, m in parent.table_manifests.items()
                if name != BOUNDS_TABLE})
        artifact = history.materialize(data.repository, partial, **data.request)
        rows = history.read(data.repository, partial, artifact["feature_snapshot"],
            columns=list(FEATURES), **data.request)["rows"]
        assert len(rows) == 3
        assert all((r["n_prior"], r["mean_prior_abs_move"]) == (3, 4) for r in rows)
        poison_outcomes(monkeypatch, data.repository, partial, (), all_outcomes=True)
        scan = data.repository.scan
        def no_bounds_scan(query, *, table_name, **kwargs):
            assert table_name != BOUNDS_TABLE, "missing partition must refuse before sidecar scan"
            return scan(query, table_name=table_name, **kwargs)
        monkeypatch.setattr(data.repository, "scan", no_bounds_scan)
        with pytest.raises(DataError) as caught:
            history.materialize(data.repository, partial,
                **{**data.request, "event_ids": [data.events[0]["event_id"]]})
        assert caught.value.code == "CONTRACT_MISMATCH"
        assert "ticker" in str(caught.value)
    finally:
        data.conn.close()


def test_missing_event_bound_refuses_with_partition_present_before_outcomes(data, monkeypatch):
    poison_outcomes(monkeypatch, data.repository, data.snapshot, (), all_outcomes=True)
    scan = data.repository.scan
    def missing_row(query, *, table_name, **kwargs):
        for batch in scan(query, table_name=table_name, **kwargs):
            if table_name == BOUNDS_TABLE:
                # Preserve row count/partition while replacing one required event with an unrelated date.
                rows = batch.to_pylist()
                rows[0]["event_date"] = "2021-01-01"
                yield type(batch).from_pylist(rows, schema=batch.schema)
            else:
                yield batch
    monkeypatch.setattr(data.repository, "scan", missing_row)
    with pytest.raises(DataError) as caught:
        history.materialize(data.repository, data.snapshot, **data.request)
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert "event" in str(caught.value)


@pytest.mark.parametrize("limit", list(LIMITS))
def test_caps_refuse_without_outcomes(data, monkeypatch, limit):
    poison_outcomes(monkeypatch, data.repository, data.snapshot, (), all_outcomes=True)
    request = {**data.request, "limits": {**LIMITS, limit: 1}}
    with pytest.raises(DataError) as caught:
        history.materialize(data.repository, data.snapshot, **request)
    assert caught.value.code == "RESULT_LIMIT_EXCEEDED"


@pytest.mark.parametrize("columns", [["renamed_mean"], ["ema12r_abs"], ["abs(mean_prior_move)"],
                                      [{"name": "safe", "input": "mean_prior_move"}]])
def test_alias_and_derived_requests_cannot_launder_sources(data, columns):
    artifact = history.materialize(data.repository, data.snapshot, **data.request)
    with pytest.raises(DataError, match="exact eligible-history provenance"):
        history.read(data.repository, data.snapshot, artifact["feature_snapshot"], columns=columns,
                     **data.request)


def test_live_contract_and_forged_policy_are_not_research_capabilities(data):
    artifact = history.materialize(data.repository, data.snapshot, **data.request)
    for contract, policy_id in [(LIVE_CONTRACT, POLICY["policy_id"]), (CONTRACT, "unknown")]:
        with pytest.raises(DataError, match="exact eligible-history provenance"):
            admit_fields(contract, artifact["context"], policy_id=policy_id, columns=list(FEATURES))
    altered = deepcopy(artifact["context"])
    altered["feature_recipe"]["usage"] = "live_scoring_only"
    with pytest.raises(DataError):
        admit_fields(CONTRACT, altered, policy_id=POLICY["policy_id"], columns=list(FEATURES))


def test_size_comparison_uses_one_native_artifact(data):
    artifact = size.build(data.repository, data.snapshot, as_of_month="2026-01",
        evaluation_years=["2024"], training_start="2023-01-01", history_start="2022-01-01",
        policy=POLICY, limits=LIMITS,
        event_ids=[data.events[1]["event_id"], data.events[5]["event_id"], data.events[-1]["event_id"]])
    result = size.comparison_inputs(data.repository, data.snapshot, artifact)
    expected = {data.events[i]["event_id"]: values for i, values in
        [(1, (1, -2, 2)), (5, (2, -3, 3)), (-1, (3, 0, 4))]}
    assert {(r["event_id"], r["horizon_sessions"]) for r in result["rows"]} == {
        (event_id, horizon) for event_id in expected for horizon in (1, 3, 5)}
    assert len(result["rows"]) == 9
    for row in result["rows"]:
        assert (row["n_prior"], row["mean_prior_move"], row["mean_prior_abs_move"]) == expected[row["event_id"]]
    assert artifact["temporal_plan"]["fold_policy"]["kind"] == "expanding_event_year"


@pytest.mark.parametrize("count", [0, 1, 2, 4, 8, 12])
def test_literal_expanding_mean_ema_recursion_and_warmup(tmp_path, count):
    days = ["2022-06-02", "2022-07-05", "2022-08-02", "2022-09-06", "2022-10-04", "2022-11-01",
            "2022-12-06", "2023-01-03", "2023-02-07", "2023-03-07", "2023-04-04", "2023-05-02"][:count]
    values = [float((-1 if i % 2 else 1) * (i + 1)) for i in range(count)]
    data = source(tmp_path, days=[*days, "2024-06-17"], values=[*values, 1.], null_day=None)
    try:
        artifact = history.materialize(data.repository, data.snapshot, **data.request)
        rows = read(data, artifact)["rows"]
        for row in rows:
            assert row["n_prior"] == count
            for kind, numbers in [("move", values), ("abs_move", [abs(v) for v in values])]:
                assert row[f"mean_prior_{kind}"] == (sum(numbers) / count if count else None)
                for span in (2, 4, 8, 12):
                    expected = None
                    if count >= span:
                        expected = numbers[0]
                        for value in numbers[1:]:
                            expected = 2 / (span + 1) * value + (1 - 2 / (span + 1)) * expected
                    assert row[f"ema{span}_prior_{kind}"] == (pytest.approx(expected) if expected is not None else None)
    finally:
        data.conn.close()


def test_source_anchor_coverage_missing_is_not_silently_empty_history(tmp_path, monkeypatch):
    data = source(tmp_path, changes=lambda tables, _: tables.update({BOUNDS_TABLE: []}))
    try:
        poison_outcomes(monkeypatch, data.repository, data.snapshot, (), all_outcomes=True)
        with pytest.raises(DataError, match="missing native source bounds") as caught:
            history.materialize(data.repository, data.snapshot, **data.request)
        assert caught.value.code == "CONTRACT_MISMATCH"
    finally:
        data.conn.close()


def test_live_source_snapshot_cannot_be_passed_as_research_table(data, monkeypatch):
    poison_outcomes(monkeypatch, data.repository, data.snapshot, (), all_outcomes=True)
    with pytest.raises(DataError, match="exact eligible-history artifact"):
        history.read(data.repository, data.snapshot, data.snapshot, columns=list(FEATURES), **data.request)


def test_unknown_target_policy_and_horizons_fail_before_values(data, monkeypatch):
    poison_outcomes(monkeypatch, data.repository, data.snapshot, (), all_outcomes=True)
    for changes in ({"policy": {}}, {"horizons": [2]}, {"horizons": [True]},
                    {"event_ids": [data.events[3]["event_id"]]}, {"event_ids": ["unknown"]}):
        with pytest.raises(DataError):
            history.materialize(data.repository, data.snapshot, **{**data.request, **changes})


def test_embargoed_development_target_refuses_even_when_annual_evaluation(data, monkeypatch):
    poison_outcomes(monkeypatch, data.repository, data.snapshot, (), all_outcomes=True)
    embargoed = data.events[2]["event_id"]
    with pytest.raises(DataError, match="development targets are excluded"):
        history.materialize(data.repository, data.snapshot,
            **{**data.request, "event_ids": [embargoed]})
    with pytest.raises(DataError, match="development targets are excluded"):
        size.build(data.repository, data.snapshot, as_of_month="2026-01", evaluation_years=["2023"],
            training_start="2022-01-01", history_start="2022-01-01", policy=POLICY, limits=LIMITS,
            event_ids=[data.events[0]["event_id"], embargoed])


def test_output_horizon_choice_cannot_weaken_frozen_group_admission(data, monkeypatch):
    poison_outcomes(monkeypatch, data.repository, data.snapshot, (), all_outcomes=True)
    for horizons in ([1], [3], [5], [1, 3], [3, 5], [5, 3, 1]):
        with pytest.raises(DataError, match="ordered native horizons"):
            history.materialize(data.repository, data.snapshot, **{**data.request, "horizons": horizons})


def test_changed_installed_producer_cannot_replay_previous_artifact(data, monkeypatch):
    artifact = history.materialize(data.repository, data.snapshot, **data.request)
    producer = deepcopy(artifact["context"]["producer"])
    assert "engine/v2/features/panel_math.py" in producer["source_closure"]
    assert "engine/v2/research/eligibility_policy.py" in producer["source_closure"]
    producer["source_closure"]["engine/v2/features/panel_math.py"] = "sha256:" + "0" * 64
    monkeypatch.setattr(history, "_producer", lambda: producer)
    poison_outcomes(monkeypatch, data.repository, data.snapshot, (), all_outcomes=True)
    with pytest.raises(DataError, match="exact eligible-history artifact"):
        read(data, artifact)
    with pytest.raises(DataError) as caught:
        size.comparison_inputs(data.repository, data.snapshot, {**artifact, "request": data.request})
    assert caught.value.problem.code == "HOLDOUT_ACCESS_DENIED"


def test_only_public_platform_admits_producer_identity(data, monkeypatch):
    from engine.v2.research import eligible_history as backend
    assert not hasattr(backend, "materialize")
    assert not hasattr(backend, "read")
    def forbidden(*args, **kwargs):
        raise AssertionError("caller producer reached fingerprinting or source access")
    monkeypatch.setattr(history, "_producer", forbidden)
    monkeypatch.setattr(data.repository, "resolve", forbidden)
    request = {**data.request, "producer": {"source_closure": {}, "environment": {}}}
    calls = [
        lambda: history.materialize(data.repository, data.snapshot, **request),
        lambda: history.read(data.repository, data.snapshot, data.snapshot, columns=list(FEATURES), **request),
        lambda: size.comparison_inputs(data.repository, data.snapshot,
            {"request": request, "feature_snapshot": data.snapshot}),
    ]
    for call in calls:
        with pytest.raises(DataError) as caught:
            call()
        assert caught.value.problem.code == "HOLDOUT_ACCESS_DENIED"
    with pytest.raises(TypeError, match="producer"):
        size.build(data.repository, data.snapshot, producer=request["producer"])


def test_fabricated_internal_producer_is_not_publicly_replayable(data):
    from engine.v2.research import eligible_history as backend
    # Internal helpers are trusted orchestration, not an admitted API or Python sandbox.
    fabricated = backend._materialize(data.repository, data.snapshot,
        producer={"source_closure": {"old.py": "old"}, "environment": {}}, **data.request)
    for call in [lambda: read(data, fabricated), lambda: size.comparison_inputs(
            data.repository, data.snapshot, {**fabricated, "request": data.request})]:
        with pytest.raises(DataError) as caught:
            call()
        assert caught.value.problem.code == "HOLDOUT_ACCESS_DENIED"
    canonical = history.materialize(data.repository, data.snapshot, **data.request)
    assert canonical["feature_snapshot"] != fabricated["feature_snapshot"]
    assert canonical["context"]["producer"] == history._producer()


def test_per_forecast_strict_prior_and_label_ceiling_do_not_backfill_earlier_rows(tmp_path, monkeypatch):
    data = source(tmp_path, days=["2024-06-12", "2024-06-17"], values=[10., 1.], null_day=None)
    try:
        poison_outcomes(monkeypatch, data.repository, data.snapshot, ["2024-06-17"])
        artifact = history.materialize(data.repository, data.snapshot, **data.request)
        by_horizon = {row["horizon_sessions"]: row for row in read(data, artifact)["rows"]}
        assert by_horizon[1]["source_session_ceiling"] == "2024-06-14"
        assert by_horizon[1]["n_prior"] == 1
        assert by_horizon[1]["mean_prior_move"] == 10.
        for horizon in (3, 5):
            assert by_horizon[horizon]["n_prior"] == 0
            assert by_horizon[horizon]["mean_prior_move"] is None
    finally:
        data.conn.close()


def test_bound_rejects_stale_object_identity_and_malformed_dates_before_reader(data):
    from engine.v2.research import eligible_history as backend
    event = SimpleNamespace(ticker="SYNTH", event_date=data.events[0]["event_date"], session="AMC")
    records = {r.primary_key_min: r for r in data.repository.fragment_records(data.snapshot, "computed_moves")}
    anchors = backend._anchors(data.repository, data.snapshot, LIMITS)
    key = (event.ticker, str(event.event_date.date()))
    for changes in ({"outcome_object_id": "wrong"}, {"outcome_content_hash": "sha256:" + "0" * 64},
                    {"session": "BMO"}, {"pre_close_date": None}, {"post_close_date": "bad"},
                    {"pre_close_date": "2022-01-01"}):
        malformed = {**anchors, key: {**anchors[key], **changes}}
        with pytest.raises(DataError) as caught:
            backend._bound(event, malformed, records)
        assert caught.value.code == ("HOLDOUT_ACCESS_DENIED" if "outcome_object_id" in changes
            or "outcome_content_hash" in changes else "CONTRACT_MISMATCH")
    with pytest.raises(DataError) as caught:
        backend._bound(event, anchors, {})
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"


def test_nonfinite_aggregate_refuses_before_publication(tmp_path):
    data = source(tmp_path, days=["2022-06-02", "2023-01-03", "2024-06-17"],
                  values=[1e308, 1e308, 1.], null_day=None)
    try:
        before = catalog_rows(data.conn)
        with pytest.warns(RuntimeWarning, match="overflow"), pytest.raises(DataError, match="history features are invalid"):
            history.materialize(data.repository, data.snapshot, **data.request)
        assert catalog_rows(data.conn) == before
    finally:
        data.conn.close()


def test_size_frozen_policy_refuses_build_and_read_before_source_access(data, monkeypatch):
    policy = eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=31)
    request = {**data.request, "policy": policy}
    artifact = history.materialize(data.repository, data.snapshot, **request)
    assert history.read(data.repository, data.snapshot, artifact["feature_snapshot"],
        columns=list(FEATURES), **request)["rows"]  # Generic platform remains policy-parameterized.
    def forbidden(*args, **kwargs):
        raise AssertionError("SIZE touched source before frozen-policy admission")
    monkeypatch.setattr(data.repository, "resolve", forbidden)
    with pytest.raises(DataError, match="frozen 30-day"):
        size.build(data.repository, data.snapshot, as_of_month="2026-01", evaluation_years=["2024"],
            training_start="2023-01-01", history_start="2022-01-01", policy=policy, limits=LIMITS)
    with pytest.raises(DataError, match="frozen 30-day"):
        size.comparison_inputs(data.repository, data.snapshot, {**artifact, "request": request})


def test_clock_memo_spans_target_and_contributor_loops_but_not_invocations(tmp_path, monkeypatch):
    from collections import Counter
    from engine.v2.research import eligible_history as backend
    data = source(tmp_path, extra_ticker=True)
    calls, original = Counter(), backend.forecast_cutoff
    def counted(day, session, horizon, calendar):
        calls[(day.date().isoformat(), session, horizon)] += 1
        return original(day, session, horizon, calendar)
    monkeypatch.setattr(backend, "forecast_cutoff", counted)
    try:
        request = {**data.request, "event_ids": [data.events[0]["event_id"], data.events[-1]["event_id"]]}
        artifact = history.materialize(data.repository, data.snapshot, **request)
        assert len(calls) == 15
        assert set(calls.values()) == {1}
        result = history.read(data.repository, data.snapshot, artifact["feature_snapshot"],
            columns=list(FEATURES), **request)
        assert len(result["rows"]) == 6
        assert set(calls.values()) == {2}  # Read revalidates; no stale global/calendar cache.
        synth = [r for r in result["rows"] if r["event_id"] == data.events[-1]["event_id"]]
        assert len(synth) == 3
        assert all((r["n_prior"], r["mean_prior_move"], r["mean_prior_abs_move"]) == (3, 0, 4) for r in synth)
    finally:
        data.conn.close()


def test_clock_memo_scales_with_distinct_keys_and_returns_detached_values(monkeypatch):
    from engine.v2.research import eligible_history as backend
    calendar = calendar_sessions("2024-05-01", "2024-06-20")
    original, calls = backend.forecast_cutoff, []
    def counted(*args):
        calls.append(args[:3])
        return original(*args)
    monkeypatch.setattr(backend, "forecast_cutoff", counted)
    event = SimpleNamespace(event_date=pd.Timestamp("2024-06-17"), session="AMC")
    memo = {}
    expected = [original(event.event_date, event.session, h, calendar) for h in (1, 3, 5)]
    for _ in range(1000):
        result = backend._forecasts(event, [1, 3, 5], calendar, memo)
        assert result == expected
        result[0]["cutoff_id"] = "mutated caller result"
    assert len(calls) == 3
    for day, session in [("2024-06-17", "BMO"), ("2024-06-18", "AMC")]:
        different = SimpleNamespace(event_date=pd.Timestamp(day), session=session)
        assert backend._forecasts(different, [1, 3, 5], calendar, memo) == [
            original(different.event_date, session, h, calendar) for h in (1, 3, 5)]
    assert len(calls) == 9
    invalid = SimpleNamespace(event_date=pd.Timestamp("2024-01-01"), session="AMC")
    for _ in range(2):
        with pytest.raises(DataError) as caught:
            backend._forecasts(invalid, [1, 3, 5], calendar, memo)
        assert caught.value.code == "CONTRACT_MISMATCH"
    assert len(calls) == 11  # Failed validation is neither cached nor turned into a clock.
    assert len(memo) == 9
