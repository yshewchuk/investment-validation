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


def source(tmp_path, *, null_day="2023-05-10", changes=None, days=None, values=None, extra_ticker=False,
           extra_events=(), missing_bounds=()):
    days = days or ["2022-06-02", "2023-01-03", "2023-03-20", "2023-04-03", "2023-05-10", "2023-06-02", "2024-06-17"]
    events = [history_event(day, protected=day == "2023-04-03") for day in days]
    moves = [outcome_move(e, value) for e, value in zip(events, values or [-2., -4., 9999., 9999., 9999., 6., 8.])]
    if extra_ticker:
        extra = history_event("2023-06-02", ticker="OTHER")
        events.insert(0, extra)
        moves.insert(0, outcome_move(extra, 9.))
    events.extend(extra_events)
    moves.extend(outcome_move(e, 7.) for e in extra_events)
    for row in moves:
        if row["event_date"] == null_day:
            row.update(skipped=True, realized_move_pct=None, available_as_of_date=None)
    conn, repository, original = commit_history_inputs(tmp_path, events, moves,
        days=calendar_sessions("2022-05-20", "2024-06-20").days)
    records = repository.fragment_records(original, "computed_moves")
    dates = sorted({str(e["event_date"].date() + timedelta(days=n)) for e in events for n in (0, 1)})
    bounds = tuple(record for ticker in sorted({e["ticker"] for e in events})
        for record in write_bounds(repository._store, [r for r in moves if r["ticker"] == ticker
            and (r["ticker"], r["event_date"]) not in missing_bounds],
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
    with pytest.raises(DataError, match="drifted since import") as caught:
        read(data, artifact)
    assert caught.value.code == "INPUT_CHANGED"
    with pytest.raises(DataError) as caught:
        size.comparison_inputs(data.repository, data.snapshot, {**artifact, "request": data.request})
    assert caught.value.problem.code == "INPUT_CHANGED"


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
    with pytest.raises(DataError) as caught:
        size.build(data.repository, data.snapshot, producer=request["producer"])
    assert caught.value.code == "HOLDOUT_ACCESS_DENIED"


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


def test_public_request_shape_matrix_refuses_before_fingerprints_or_sources(data, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("malformed public input reached fingerprinting or source access")

    monkeypatch.setattr(history, "_producer", forbidden)
    monkeypatch.setattr(history, "admit_source", forbidden)
    monkeypatch.setattr(data.repository, "resolve", forbidden)
    monkeypatch.setattr(data.repository, "scan", forbidden)
    monkeypatch.setattr(data.repository._store, "publish_bytes", forbidden)
    for invalid in (None, [], "request"):
        with pytest.raises(DataError) as caught:
            history.validate_request(invalid)
        assert caught.value.code == "CONTRACT_MISMATCH"
    bad_values = {
        "as_of_month": [None, [], 202601, "2026-13", "2026-1"],
        "history_start": [None, [], 2022, "2022-02-30", "20220101"],
        "event_ids": [None, {}, "event", [], [None], [""], [" x"], ["x", "x"]],
        "horizons": [None, {}, "1", [1], [True, 3, 5]],
        "limits": [None, [], {}, {**LIMITS, "extra": 1},
                   *[{**LIMITS, key: value} for key in LIMITS for value in (None, True, 0, "1")]],
        "policy": [None, [], {}, {**POLICY, "extra": 1},
                   *[{k: v for k, v in POLICY.items() if k != key} for key in POLICY],
                   *[{**POLICY, key: None} for key in POLICY]],
    }
    invalid = [{k: v for k, v in data.request.items() if k != key} for key in data.request]
    invalid += [{**data.request, "extra": None}]
    invalid += [{**data.request, key: value} for key, values in bad_values.items() for value in values]
    for request in invalid:
        for call in (
            lambda: history.materialize(data.repository, data.snapshot, **request),
            lambda: history.read(data.repository, data.snapshot, data.snapshot,
                                 columns=list(FEATURES), **request),
            lambda: size.comparison_inputs(data.repository, data.snapshot,
                {"request": request, "feature_snapshot": data.snapshot}),
        ):
            with pytest.raises(DataError) as caught:
                call()
            assert caught.value.code == "CONTRACT_MISMATCH"
    build = {k: v for k, v in data.request.items() if k != "horizons"}
    build.update(evaluation_years=["2024"], training_start="2023-01-01")
    invalid_builds = [{k: v for k, v in build.items() if k != key}
                     for key in build if key != "event_ids"]
    invalid_builds += [{**build, "extra": 1}]
    invalid_builds += [{**build, key: value} for key, values in bad_values.items() if key != "horizons"
                      for value in values if key != "event_ids" or value is not None]
    invalid_builds += [{**build, key: value} for key in ("evaluation_years", "training_start")
                      for value in (None, {}, 2024, [], "bad")]
    for request in invalid_builds:
        with pytest.raises(DataError) as caught:
            size.build(data.repository, data.snapshot, **request)
        assert caught.value.code == "CONTRACT_MISMATCH"


def test_public_artifact_columns_and_snapshot_shapes_refuse_without_io(data, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("malformed envelope reached installed fingerprint or source")

    monkeypatch.setattr(history, "_producer", forbidden)
    monkeypatch.setattr(history, "admit_source", forbidden)
    artifact = {"request": data.request, "feature_snapshot": data.snapshot}
    for invalid in (None, [], "artifact", {}, {"request": data.request},
                    {"feature_snapshot": data.snapshot}, {**artifact, "extra": None},
                    {**artifact, "context": None}, {**artifact, "temporal_plan": []},
                    {**artifact, "request": None}, {**artifact, "request": []}):
        with pytest.raises(DataError) as caught:
            size.comparison_inputs(data.repository, data.snapshot, invalid)
        assert caught.value.code == "CONTRACT_MISMATCH"
    for columns in (None, [], {}, "n_prior", [None], ["n_prior", "n_prior"], ["live_mean"]):
        with pytest.raises(DataError) as caught:
            history.read(data.repository, data.snapshot, data.snapshot, columns=columns, **data.request)
        assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
    for invalid in (None, {}, "snap_invalid"):
        for call in (
            lambda: history.materialize(data.repository, invalid, **data.request),
            lambda: history.read(data.repository, data.snapshot, invalid,
                                 columns=list(FEATURES), **data.request),
            lambda: size.comparison_inputs(data.repository, data.snapshot,
                {**artifact, "feature_snapshot": invalid}),
        ):
            with pytest.raises(DataError) as caught:
                call()
            assert caught.value.code == "SNAPSHOT_UNRESOLVED"


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


@pytest.mark.parametrize("protected", [False, True])
def test_each_ticker_uses_its_own_history_ceiling_but_full_protected_metadata(tmp_path, protected):
    later = history_event("2023-06-20" if protected else "2024-04-02", ticker="OTHER", protected=protected)
    data = source(tmp_path, extra_ticker=True, extra_events=[later],
        missing_bounds=[("OTHER", later["event_date"].date().isoformat())])
    try:
        request = {**data.request, "event_ids": [data.events[0]["event_id"], data.events[-2]["event_id"]]}
        if protected:
            with pytest.raises(DataError) as caught:
                history.materialize(data.repository, data.snapshot, **request)
            assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
        else:
            artifact = history.materialize(data.repository, data.snapshot, **request)
            result = history.read(data.repository, data.snapshot, artifact["feature_snapshot"],
                columns=list(FEATURES), **request)
            assert len(result["rows"]) == 6
            assert {r["event_id"] for r in result["rows"]} == set(request["event_ids"])
            assert later["event_id"] not in {g["event_id"] for g in artifact["context"]["eligibility"]["groups"]}
    finally:
        data.conn.close()


def test_on_disk_source_drift_after_import_refuses_before_source_access(data, tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    root = Path(history.__file__).resolve().parents[1]
    copied = tmp_path / "code"
    for relative in history._producer()["source_closure"]:
        target = copied / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / relative).read_bytes())
    spec = importlib.util.spec_from_file_location("isolated_history_api", copied / "experiments/eligible_history.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    artifact = api.materialize(data.repository, data.snapshot, **data.request)
    dependency = copied / "engine/v2/features/panel_math.py"
    dependency.write_text(dependency.read_text() + "\n# changed after API import\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("source drift reached repository")
    monkeypatch.setattr(data.repository, "resolve", forbidden)
    for call in [lambda: api.materialize(data.repository, data.snapshot, **data.request),
            lambda: api.read(data.repository, data.snapshot, artifact["feature_snapshot"],
                columns=list(FEATURES), **data.request)]:
        with pytest.raises(DataError) as caught:
            call()
        assert caught.value.code == "INPUT_CHANGED"


@pytest.mark.parametrize("damage", ["scope", "failed", "snapshot"])
def test_receipt_lookup_projects_bounded_identity_not_diagnostics(data, damage):
    import sqlite3
    from engine.v2.research import eligible_history as backend
    artifact = history.materialize(data.repository, data.snapshot, **data.request)
    slot = backend._slot(artifact["context"])
    # Deliberately damage this synthetic catalog; production immutability remains unchanged.
    data.conn.execute("DROP TRIGGER data_import_receipts_no_update")
    # Large diagnostic must never cross the read boundary even in a damaged catalog.
    problem = '{"message":"' + "x" * (1 << 20) + '"}'
    if damage == "scope":
        data.conn.execute("UPDATE data_import_receipts SET scope=?,problem_json=? WHERE receipt_id=?",
            ("x" * (1 << 20), problem, slot))
    elif damage == "failed":
        data.conn.execute("UPDATE data_import_receipts SET status='failed',result_snapshot_id=NULL,problem_json=? WHERE receipt_id=?",
            (problem, slot))
    else:
        data.conn.commit()
        data.conn.execute("PRAGMA foreign_keys=OFF")
        data.conn.execute("UPDATE data_import_receipts SET result_snapshot_id=?,problem_json=? WHERE receipt_id=?",
            ("snap_" + "g" * 32, problem, slot))
    def authorizer(action, table, column, *rest):
        if action == sqlite3.SQLITE_READ and table == "data_import_receipts" and column in {"problem_json", "registered_at"}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    data.conn.set_authorizer(authorizer)
    try:
        with pytest.raises(DataError) as caught:
            backend._existing(data.repository, artifact["context"])
        assert caught.value.code == "IDENTITY_CONFLICT"
    finally:
        data.conn.set_authorizer(None)


@pytest.mark.parametrize("offset", [-1, 0])
def test_compressible_output_obeys_native_expanded_read_cap_before_publish(data, monkeypatch, offset):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from engine.v2.research import eligible_history as backend
    original = backend._rows
    def compressible_rows(*args):
        for rows in original(*args):
            for row in rows:
                row["history_json"] += " " * (256 << 10)  # Valid JSON trailing whitespace, highly compressible.
            yield rows
    monkeypatch.setattr(backend, "_rows", compressible_rows)
    prepared = backend._prepare(data.repository, data.snapshot, producer=history._producer(), **data.request)
    rows = next(compressible_rows(data.snapshot, prepared))
    schema = pa.schema([pa.field(c.name, backend.query.arrow_type_for(c.physical_type), nullable=c.nullable)
        for c in CONTRACT.columns])
    sink = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), sink)
    payload = sink.getvalue().to_pybytes()
    with pq.ParquetFile(pa.BufferReader(payload)) as file:
        expanded = sum(file.metadata.row_group(i).total_byte_size for i in range(file.num_row_groups))
    assert len(payload) < expanded - 1
    request = {**data.request, "limits": {**LIMITS, "max_object_bytes": expanded + offset}}
    before = catalog_rows(data.conn)
    if offset < 0:
        def forbidden(*args, **kwargs):
            raise AssertionError("unreadable output reached object publication")
        monkeypatch.setattr(data.repository._store, "publish_bytes", forbidden)
        with pytest.raises(DataError) as caught:
            history.materialize(data.repository, data.snapshot, **request)
        assert caught.value.code == "RESULT_LIMIT_EXCEEDED"
        assert catalog_rows(data.conn) == before
    else:
        artifact = history.materialize(data.repository, data.snapshot, **request)
        result = history.read(data.repository, data.snapshot, artifact["feature_snapshot"],
            columns=list(FEATURES), **request)
        assert len(result["rows"]) == 3
        assert all((r["n_prior"], r["mean_prior_abs_move"]) == (3, 4) for r in result["rows"])
