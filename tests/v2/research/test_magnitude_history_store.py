"""Synthetic native publication/read oracles, including physical poison guards."""
import json
import sqlite3
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.data import catalog, objects
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import SystemClock, canonical_json, content_hash
from engine.v2.research import event_history, magnitude_features
from engine.v2.research import magnitude_history_store as stored
from engine.v2.research.magnitude_history_table import CONTRACT, FEATURES, TABLE
from experiments import magnitude_history_store as entry
from tests.v2.research.test_event_history import _event, _poison
from tests.v2.research.test_magnitude_features import _commit
from tests.v2.research.test_magnitude_inputs import _calendar
from tests.v2.research.test_native_registration import _catalog, _objects
from tests.v2.research.test_prediction_inputs import _move

REQUEST = dict(as_of_month="2026-01", evaluation_years=["2024"],
               training_start="2023-01-01", history_start="2022-01-01")


def _source(tmp_path, *, mixed_days=(), overrides=None):
    events = [_event(day) for day in ("2022-06-02", "2023-01-03", "2023-06-02", "2024-06-12", "2024-06-17")]
    moves = [_move(event, value) for event, value in zip(events, (-2., -4., 6., 10., None))]
    for index, changes in (overrides or {}).items():
        moves[index].update(changes)
    forbidden = [_event("2023-04-03", protected=True), _event("2025-12-03")]
    ambiguous = _event("2023-05-03")
    forbidden.append(ambiguous)
    sibling = {**ambiguous, "event_id": "ambiguous-sibling", "session": "BMO"}
    moves += [_move(event, None) for event in forbidden]
    conn, repository, snapshot = _commit(tmp_path, [*events, *forbidden, sibling], moves,
        mixed_days=mixed_days, days=_calendar("2022-05-25", "2024-06-18").days)
    return SimpleNamespace(conn=conn, repository=repository, snapshot=snapshot,
        store=repository._store, events=events, forbidden=forbidden, root=tmp_path)


@pytest.fixture
def source(tmp_path):
    source = _source(tmp_path)
    yield source
    source.conn.close()


def _build(source, **changes):
    result = entry.build(source.repository, source.snapshot, **{**REQUEST, **changes})
    return result, source.repository.resolve(result["feature_snapshot"]["snapshot_id"])


def _read(source, output, **changes):
    return entry.read(source.repository, source.snapshot, output, **{**REQUEST, **changes})


def _forbid(*args, **kwargs):
    raise AssertionError("forbidden path was called")


def test_native_roundtrip_math_lineage_units_identity_and_replay(source, monkeypatch):
    forbidden = [e["event_date"].date().isoformat() for e in source.forbidden]
    forbidden.append("2024-06-17")  # Latest target is never historical input.
    _poison(monkeypatch, source.repository, source.snapshot, forbidden)
    before_source = Repository.resolve_pinned(source.repository, "shadow")
    build, output = _build(source)
    context = build["context"]
    assert output != source.snapshot and set(output.table_versions) == {TABLE}
    assert Repository.resolve_pinned(source.repository, "shadow") == before_source
    assert output.knowledge_mode_by_table == {TABLE: "reconstructed"}
    assert context["context_id"] == content_hash({k: v for k, v in context.items() if k != "context_id"})
    assert context["source_snapshot"]["snapshot_id"] == source.snapshot.snapshot_id
    assert context["producer"]["source_closure"]["experiments/magnitude_history_store.py"].startswith("sha256:")
    assert context["feature_recipe"]["features"] == [
        {"name": name, "on_missing": "zero_count" if name == "n_prior" else "null"} for name in FEATURES]
    before = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(magnitude_features, "load_magnitude_history", _forbid)
    monkeypatch.setattr(event_history, "load_event_history", _forbid)
    monkeypatch.setattr(stored, "_preflight_source", _forbid)
    first = _read(source, output)
    assert first == _read(source, output)
    assert _build(source) == (build, output)  # Metadata admission, then artifact-only replay.
    assert (_catalog(source.conn), _objects(source.store)) == before
    assert first["inputs_id"] == content_hash({k: v for k, v in first.items() if k != "inputs_id"})
    assert len(first["rows"]) == 12
    rows = {(r["event_date"], r["horizon_sessions"]): r for r in first["rows"]}
    for horizon in (1, 3, 5):
        one = rows["2023-01-03", horizon]
        assert one["n_prior"] == 1 and one["mean_prior_move"] == -2 and one["mean_prior_abs_move"] == 2
        assert one["ema2_prior_move"] is None
        three = rows["2024-06-12", horizon]
        assert three["n_prior"] == 3 and three["mean_prior_move"] == 0 and three["mean_prior_abs_move"] == 4
        assert three["ema2_prior_move"] == pytest.approx(26 / 9)
        assert three["ema2_prior_abs_move"] == pytest.approx(46 / 9)
        assert three["ema4_prior_abs_move"] is None
    assert rows["2024-06-17", 1]["n_prior"] == 4
    assert rows["2024-06-17", 3]["n_prior"] == 3
    assert rows["2024-06-17", 5]["n_prior"] == 3
    history = json.loads(rows["2024-06-17", 1]["history_json"])
    assert [h["event_id"] for h in history["history"]] == [e["event_id"] for e in source.events[:-1]]
    assert history["omitted_history"] == [] and len(history["read_inputs"]) == 4
    records = source.repository.fragment_records(output, TABLE)
    assert len(records) == 4 and all(r.row_count == 3 for r in records)
    assert all(r.primary_key_min[-1] == 1 and r.primary_key_max[-1] == 5 for r in records)


def test_empty_and_zero_history_keep_distinct_missingness(source):
    _, output = _build(source, history_start="2022-12-01")
    rows = _read(source, output, history_start="2022-12-01")["rows"]
    empty = next(r for r in rows if r["event_date"] == "2023-01-03")
    assert empty["n_prior"] == 0 and empty["history_status"] == "empty"
    assert all(empty[name] is None for name in FEATURES[1:])
    assert json.loads(empty["history_json"])["history"] == []
    zero = next(r for r in rows if r["event_date"] == "2024-06-12")
    assert zero["n_prior"] == 2 and zero["mean_prior_move"] == 1 and zero["mean_prior_abs_move"] == 5


def test_later_mixed_candidate_refuses_before_even_first_safe_outcome(tmp_path, monkeypatch):
    source = _source(tmp_path, mixed_days=("2023-06-02", "2024-06-12"))
    try:
        _poison(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
        before = _catalog(source.conn), _objects(source.store)
        with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
            _build(source)
        assert (_catalog(source.conn), _objects(source.store)) == before
    finally:
        source.conn.close()


@pytest.mark.parametrize("change", [
    {"as_of_month": "2026-02"}, {"history_start": "2022-02-01"},
    {"training_start": "2023-02-01"},
])
def test_changed_context_cannot_reuse_feature_snapshot(source, monkeypatch, change):
    _, output = _build(source)
    monkeypatch.setattr(stored, "_scan", _forbid)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        _read(source, output, **change)


def test_source_or_output_reference_drift_refuses_before_feature_payload(source, monkeypatch):
    _, output = _build(source)
    monkeypatch.setattr(stored, "_scan", _forbid)
    with pytest.raises(DataError, match="SNAPSHOT_UNRESOLVED"):
        entry.read(source.repository, replace(source.snapshot, calendar_version="changed"), output, **REQUEST)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        _read(source, replace(output, manifest_hash="sha256:" + "a" * 64))
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        _read(source, source.snapshot)


def test_success_receipt_is_immutable_native_catalog_evidence(source):
    built, output = _build(source)
    _, receipt = stored._names(built["context"])
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        source.conn.execute("UPDATE data_import_receipts SET scope='wrong' WHERE receipt_id=?", (receipt,))
    assert len(_read(source, output)["rows"]) == 12


@pytest.mark.parametrize("field,value", [("scope", "wrong"), ("attempt_id", "wrong"),
                                        ("fence", 7), ("request_hash", "sha256:" + "a" * 64)])
def test_colliding_native_success_receipt_refuses_before_outcomes(source, monkeypatch, field, value):
    binding = entry.context(source.repository, source.snapshot, **REQUEST)
    scope, receipt = stored._names(binding)
    parent = source.repository.resolve_full(source.snapshot.snapshot_id)
    request = dict(scope=scope, request_hash=binding["context_id"], receipt_id=receipt,
        attempt_id=receipt, fence=1, expected_head_snapshot_id=None, expected_head_generation=0,
        contracts=parent.contracts, objects=parent.objects, records=parent.records,
        manifests=tuple(parent.table_manifests.values()), snapshot=parent.snapshot,
        fence_check=lambda conn: None, clock=SystemClock(), store=source.store, audit_partitions=False)
    catalog.commit_snapshot(source.conn, **{**request, field: value})
    monkeypatch.setattr(stored, "_preflight_source", _forbid)
    before = _catalog(source.conn), _objects(source.store)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        _build(source)
    assert (_catalog(source.conn), _objects(source.store)) == before


def test_all_metadata_is_admitted_before_any_feature_column_decode(source, monkeypatch):
    original = stored._event_rows

    def malformed(*args):
        rows = original(*args)
        if rows[0]["event_date"] == "2024-06-17":
            rows[-1]["cutoff_id"] = "wrong"
        return rows

    monkeypatch.setattr(stored, "_event_rows", malformed)
    _, output = _build(source)
    scan = Repository.scan
    seen = []

    def guarded(repository, request, *, table_name):
        if table_name == TABLE:
            assert not set(request.columns) & set(FEATURES), "feature columns decoded before full metadata admission"
            seen.append(request)
        yield from scan(repository, request, table_name=table_name)

    monkeypatch.setattr(Repository, "scan", guarded)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _read(source, output)
    assert len(seen) == 1


@pytest.mark.parametrize("field,value", [
    ("n_prior", 99), ("mean_prior_move", float("nan")),
    ("mean_prior_abs_move", -1.), ("ema12_prior_move", 0.), ("mean_prior_move", None),
])
def test_real_stored_bad_feature_or_missingness_refuses_without_partial_return(source, monkeypatch, field, value):
    original = stored._event_rows

    def malformed(*args):
        rows = original(*args)
        if rows[0]["event_date"] == "2024-06-17":
            rows[-1][field] = value
        return rows

    monkeypatch.setattr(stored, "_event_rows", malformed)
    _, output = _build(source)
    before = _catalog(source.conn), _objects(source.store)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _read(source, output)
    assert (_catalog(source.conn), _objects(source.store)) == before


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_native_inspection_refuses_nonfinite_features_before_commit(source, monkeypatch, value):
    original = stored._event_rows

    def malformed(*args):
        rows = original(*args)
        rows[-1]["mean_prior_move"] = value
        return rows

    monkeypatch.setattr(stored, "_event_rows", malformed)
    before = _catalog(source.conn)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _build(source)
    assert _catalog(source.conn) == before


def test_global_fragment_preflight_precedes_all_feature_object_opens(source, monkeypatch):
    _, output = _build(source)
    records = source.repository.fragment_records(output, TABLE)
    fragment_records = source.repository.fragment_records

    def malformed(snapshot, table):
        if table == TABLE:
            return (*records[:-1], replace(records[-1], row_count=4))
        return fragment_records(snapshot, table)

    monkeypatch.setattr(source.repository, "fragment_records", malformed)
    forbidden = {r.object_ref.object_id for r in records}
    verify = objects.verify_object_path

    def guarded(store, ref):
        assert ref.object_id not in forbidden
        return verify(store, ref)

    monkeypatch.setattr(objects, "verify_object_path", guarded)
    with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
        _read(source, output)


def test_commit_failure_rolls_back_and_retry_uses_unpoisoned_success_identity(source, monkeypatch):
    commit = catalog.commit_snapshot
    before = _catalog(source.conn)

    def interrupted(*args, **kwargs):
        def fault(point):
            if point == "before_commit":
                raise RuntimeError("synthetic interruption")
        return commit(*args, **kwargs, fault=fault)

    monkeypatch.setattr(catalog, "commit_snapshot", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _build(source)
    assert _catalog(source.conn) == before
    monkeypatch.setattr(catalog, "commit_snapshot", commit)
    _, output = _build(source)
    assert len(_read(source, output)["rows"]) == 12


def test_existing_context_scope_conflict_preserves_head(source):
    binding = entry.context(source.repository, source.snapshot, **REQUEST)
    scope, _ = stored._names(binding)
    source.conn.execute("INSERT INTO data_snapshot_heads SELECT ?, snapshot_id, generation, updated_at, update_receipt_ref "
                        "FROM data_snapshot_heads WHERE scope='shadow'", (scope,))
    before = _catalog(source.conn)
    with pytest.raises(DataError, match="SNAPSHOT_CONFLICT"):
        _build(source)
    assert _catalog(source.conn) == before


def test_replay_after_head_moves_returns_original_artifact_without_head_effect(source, monkeypatch):
    built, output = _build(source)
    scope, _ = stored._names(built["context"])
    source.conn.execute("UPDATE data_snapshot_heads SET snapshot_id=?, generation=generation+1 WHERE scope=?",
                        (source.snapshot.snapshot_id, scope))
    before = _catalog(source.conn), _objects(source.store)
    monkeypatch.setattr(magnitude_features, "load_magnitude_history", _forbid)
    assert _build(source) == (built, output)
    assert (_catalog(source.conn), _objects(source.store)) == before


def test_cli_build_read_and_typed_refusal(source, capsys):
    args = ["--catalog", str(source.root / "catalog.sqlite"), "--store-root", str(source.store.root),
        "--source-snapshot-id", source.snapshot.snapshot_id, "--as-of-month", REQUEST["as_of_month"],
        "--evaluation-years", *REQUEST["evaluation_years"], "--training-start", REQUEST["training_start"],
        "--history-start", REQUEST["history_start"]]
    assert entry.main(["build", *args]) == 0
    built = json.loads(capsys.readouterr().out)
    assert entry.main(["read", *args, "--feature-snapshot-id", built["feature_snapshot"]["snapshot_id"]]) == 0
    rows = json.loads(capsys.readouterr().out)["rows"]
    assert len(rows) == 12
    row = next(r for r in rows if r["event_date"] == "2024-06-12" and r["horizon_sessions"] == 1)
    assert row["n_prior"] == 3 and row["mean_prior_abs_move"] == 4
    assert row["ema2_prior_abs_move"] == pytest.approx(46 / 9)
    assert entry.main(["read", *args, "--feature-snapshot-id", source.snapshot.snapshot_id]) == 2
    assert json.loads(capsys.readouterr().out)["refused"] == "IDENTITY_CONFLICT"
    with pytest.raises(SystemExit, match="2"):
        entry.main(["read", *args])
    with pytest.raises(SystemExit, match="2"):
        entry.main(["build", *args, "--feature-snapshot-id", source.snapshot.snapshot_id])


def test_recipe_and_physical_contract_are_closed_and_source_rows_are_not_outputs():
    assert CONTRACT.primary_key == ("event_id", "context_id", "horizon_sessions")
    assert CONTRACT.partition_columns == ("context_id",)
    assert [c.name for c in CONTRACT.columns] == [*stored.METADATA, *FEATURES]
    assert all(c.nullable == (c.name in FEATURES[1:]) for c in CONTRACT.columns)
    assert len(FEATURES) == 11


def test_context_or_source_mismatch_refuses_before_any_source_or_feature_payload(source, monkeypatch):
    binding = entry.context(source.repository, source.snapshot, **REQUEST)
    _poison(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    before = _catalog(source.conn), _objects(source.store)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        stored.materialize(source.repository, source.snapshot, {**binding, "history_start": "2022-02-01"})
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        stored.materialize(source.repository, replace(source.snapshot, calendar_version="different"), binding)
    assert (_catalog(source.conn), _objects(source.store)) == before


def test_equal_feature_values_under_new_context_have_distinct_physical_objects(source):
    _, first = _build(source)
    _, second = _build(source, history_start="2022-02-01")
    left, right = _read(source, first)["rows"], _read(source, second, history_start="2022-02-01")["rows"]
    assert [[r[k] for k in FEATURES] for r in left] == [[r[k] for k in FEATURES] for r in right]
    assert first != second
    left_objects = {r.object_ref.object_id for r in source.repository.fragment_records(first, TABLE)}
    right_objects = {r.object_ref.object_id for r in source.repository.fragment_records(second, TABLE)}
    assert not left_objects & right_objects


def test_explicit_protected_population_fails_before_any_outcome_or_publication(source, monkeypatch):
    _poison(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    before = _catalog(source.conn), _objects(source.store)
    with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
        _build(source, event_ids=[source.forbidden[0]["event_id"]])
    assert (_catalog(source.conn), _objects(source.store)) == before


@pytest.mark.parametrize("name,value", [
    ("random_membership_version", "wrong"), ("rolling_membership_version", "wrong"),
    ("holdout_as_of_month", "2026-02"), ("computed_moves_version", "wrong"),
    ("purpose", "final"), ("schema_version", "wrong"), ("read_inputs", None),
])
def test_rehashed_cross_context_or_missing_lineage_refuses_before_features(source, monkeypatch, name, value):
    original = stored._event_rows

    def malformed(*args):
        rows = original(*args)
        if rows[0]["event_date"] == "2024-06-17":
            history = json.loads(rows[-1]["history_json"])
            if value is None:
                del history[name]
            else:
                history[name] = value
            history["history_id"] = content_hash({k: v for k, v in history.items()
                if k not in ("history_id", "history_status", "holdout_exclusions")})
            rows[-1]["history_json"] = canonical_json(history)
        return rows

    monkeypatch.setattr(stored, "_event_rows", malformed)
    _, output = _build(source)
    scan = stored._scan

    def guarded(repository, snapshot, context, columns, count):
        assert not set(columns) & set(FEATURES)
        return scan(repository, snapshot, context, columns, count)

    monkeypatch.setattr(stored, "_scan", guarded)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _read(source, output)


def test_native_short_horizon_object_refuses_before_opening_any_feature_object(source, monkeypatch):
    original = stored._event_rows

    def incomplete(*args):
        rows = original(*args)
        return rows[:-1] if rows[0]["event_date"] == "2024-06-17" else rows

    monkeypatch.setattr(stored, "_event_rows", incomplete)
    _, output = _build(source)
    forbidden = {r.object_ref.object_id for r in source.repository.fragment_records(output, TABLE)}
    verify = objects.verify_object_path

    def guarded(store, ref):
        assert ref.object_id not in forbidden
        return verify(store, ref)

    monkeypatch.setattr(objects, "verify_object_path", guarded)
    with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
        _read(source, output)


def test_old_native_contract_refuses_before_feature_scan(source, monkeypatch):
    original = stored.CONTRACT
    from engine.v2.data.manifests import table_contract_hash
    old = replace(original, contract_id=TABLE + ".v0", semantic_version="0.0.0")
    old = replace(old, definition_hash=table_contract_hash(old))
    monkeypatch.setattr(stored, "CONTRACT", old)
    _, output = _build(source)
    monkeypatch.setattr(stored, "CONTRACT", original)
    monkeypatch.setattr(stored, "_scan", _forbid)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _read(source, output)


def test_late_unparseable_and_skipped_labels_are_stored_as_omissions(tmp_path):
    source = _source(tmp_path, overrides={0: {"realized_move_pct": None, "available_as_of_date": "2026-01-01"},
        1: {"realized_move_pct": None, "skipped": True, "available_as_of_date": None},
        2: {"realized_move_pct": 0.}})
    try:
        _, output = _build(source)
        rows = _read(source, output)["rows"]
        row = next(r for r in rows if r["event_date"] == "2024-06-12")
        assert row["n_prior"] == 1 and row["mean_prior_abs_move"] == 0 and row["mean_prior_move"] == 0
        history = json.loads(row["history_json"])
        assert len(history["read_inputs"]) == 3
        assert history["omitted_history"] == [
            {"event_id": source.events[0]["event_id"], "reason": "unavailable"},
            {"event_id": source.events[1]["event_id"], "reason": "skipped"}]
    finally:
        source.conn.close()


def test_build_scans_calendar_once_population_twice_and_each_outcome_once(source, monkeypatch):
    monkeypatch.setattr(magnitude_features, "load_magnitude_history", _forbid)
    monkeypatch.setattr(event_history, "load_event_history", _forbid)
    source.repository.scans.clear()
    _, output = _build(source)
    scans = list(source.repository.scans)
    # Four event-year partitions (two passes), three market-year partitions (one pass).
    assert Counter(name for name, _ in scans) == {"earnings_events": 8, "daily_market": 3, "computed_moves": 4}
    outcome_queries = [q for name, q in scans if name == "computed_moves"]
    assert all(q.max_result_rows == 1 and q.max_batch_rows == 1 for q in outcome_queries)
    assert len({q.time_interval.start_inclusive for q in outcome_queries}) == 4
    assert len(_read(source, output)["rows"]) == 12
    source.repository.scans.clear()
    _build(source, history_start="2022-02-01")
    assert Counter(name for name, _ in source.repository.scans)["computed_moves"] == 4  # No cross-build cache.


def test_lazy_cache_preserves_first_invalid_label_refusal_before_later_objects(tmp_path, monkeypatch):
    source = _source(tmp_path, overrides={0: {"realized_move_pct": None}})
    try:
        _poison(monkeypatch, source.repository, source.snapshot,
                [e["event_date"].date().isoformat() for e in source.events[1:] + source.forbidden])
        before = _catalog(source.conn), _objects(source.store)
        with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
            _build(source)
        assert (_catalog(source.conn), _objects(source.store)) == before
    finally:
        source.conn.close()


def test_prepared_histories_preserve_existing_per_forecast_features_and_full_lineage(source):
    _, output = _build(source)
    rows = _read(source, output)["rows"]
    for row in rows:
        expected = magnitude_features.load_magnitude_history(source.repository, source.snapshot,
            event_id=row["event_id"], as_of_month=REQUEST["as_of_month"], horizon=row["horizon_sessions"],
            history_start=REQUEST["history_start"])
        assert {key: row[key] for key in FEATURES} == expected["features"]
        assert row["cutoff_id"] == expected["forecast_cutoff"]["cutoff_id"]
        assert json.loads(row["history_json"]) == {
            key: value for key, value in expected["history_input"].items() if key != "features"}


@pytest.mark.parametrize("change", ["both_statuses", "lineage_status"])
def test_inconsistent_stored_history_status_refuses_before_feature_columns(source, monkeypatch, change):
    original = stored._event_rows

    def malformed(*args):
        rows = original(*args)
        if rows[0]["event_date"] == "2024-06-17":
            history = json.loads(rows[-1]["history_json"])
            history["history_status"] = "empty"
            rows[-1]["history_json"] = canonical_json(history)
            if change == "both_statuses":
                rows[-1]["history_status"] = "empty"
        return rows

    monkeypatch.setattr(stored, "_event_rows", malformed)
    _, output = _build(source)
    scan = stored._scan

    def guarded(repository, snapshot, context, columns, count):
        assert not set(columns) & set(FEATURES)
        return scan(repository, snapshot, context, columns, count)

    monkeypatch.setattr(stored, "_scan", guarded)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _read(source, output)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_fragment_set_coverage_refuses_before_any_feature_object(source, monkeypatch, change):
    _, output = _build(source)
    records = source.repository.fragment_records(output, TABLE)
    original = source.repository.fragment_records

    def malformed(snapshot, table):
        if table == TABLE:
            return records[:-1] if change == "missing" else (*records, records[-1])
        return original(snapshot, table)

    monkeypatch.setattr(source.repository, "fragment_records", malformed)
    forbidden = {r.object_ref.object_id for r in records}
    verify = objects.verify_object_path

    def guarded(store, ref):
        assert ref.object_id not in forbidden
        return verify(store, ref)

    monkeypatch.setattr(objects, "verify_object_path", guarded)
    with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
        _read(source, output)


@pytest.mark.parametrize("phase", ["metadata", "features"])
def test_scan_row_coverage_guard_prevents_a_partial_stored_result(source, monkeypatch, phase):
    _, output = _build(source)
    scan = stored._scan

    def truncated(repository, snapshot, context, columns, count):
        is_feature_scan = bool(set(columns) & set(FEATURES))
        if phase == "metadata":
            assert not is_feature_scan
        rows = scan(repository, snapshot, context, columns, count)
        return rows[:-1] if is_feature_scan == (phase == "features") else rows

    monkeypatch.setattr(stored, "_scan", truncated)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _read(source, output)
