"""Synthetic native publication/read oracles, including physical poison guards."""
import hashlib
import importlib.metadata
import json
import platform
import sqlite3
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from engine.v2.data import catalog, objects
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
    ArtifactError,
    SystemClock,
    canonical_json,
    content_hash,
    to_document,
)
from engine.v2.ops import bootstrap
from engine.v2.research import event_history, magnitude_features
from engine.v2.research import magnitude_history_store as stored
from engine.v2.research.magnitude_history_table import CONTRACT, FEATURES, TABLE
from experiments import magnitude_history_store as entry
from tests.v2.research import magnitude_trial_support as support
from tests.v2.research.magnitude_trial_support import (
    HISTORY_REQUEST,
    build_history,
    calendar_sessions,
    catalog_rows,
    commit_history_inputs,
    history_event,
    outcome_move,
    poison_outcomes,
)
from tests.v2.research.test_native_registration import artifact_object_bytes


def _source(tmp_path, *, mixed_days=(), overrides=None, recent_day="2024-06-12"):
    events = [history_event(day) for day in ("2022-06-02", "2023-01-03", "2023-06-02", recent_day, "2024-06-17")]
    moves = [outcome_move(event, value) for event, value in zip(events, (-2., -4., 6., 10., None))]
    for index, changes in (overrides or {}).items():
        moves[index].update(changes)
    forbidden = [history_event("2023-04-03", protected=True), history_event("2025-12-03")]
    ambiguous = history_event("2023-05-03")
    forbidden.append(ambiguous)
    sibling = {**ambiguous, "event_id": "ambiguous-sibling", "session": "BMO"}
    moves += [outcome_move(event, None) for event in forbidden]
    conn, repository, snapshot = commit_history_inputs(tmp_path, [*events, *forbidden, sibling], moves,
        mixed_days=mixed_days, days=calendar_sessions("2022-05-25", "2024-06-18").days)
    return SimpleNamespace(conn=conn, repository=repository, snapshot=snapshot,
        store=repository._store, events=events, forbidden=forbidden, root=tmp_path)


@pytest.fixture
def source(tmp_path):
    source = _source(tmp_path)
    yield source
    source.conn.close()


def _long_source(tmp_path, count):
    dates = ("2022-06-02", "2022-07-05", "2022-08-02", "2022-09-06",
        "2022-10-04", "2022-11-01", "2022-12-06", "2023-01-03",
        "2023-02-07", "2023-03-07", "2023-04-04", "2023-05-02")
    events = [history_event(day) for day in (*dates[:count], "2024-06-17")]
    values = [float((-1 if n % 2 else 1) * 2 * n) for n in range(1, count + 1)]
    moves = [outcome_move(event, value) for event, value in zip(events, [*values, None])]
    conn, repository, snapshot = commit_history_inputs(tmp_path, events, moves,
        days=calendar_sessions("2022-05-25", "2024-06-18").days)
    return SimpleNamespace(conn=conn, repository=repository, snapshot=snapshot,
        store=repository._store, events=events, root=tmp_path)


@pytest.fixture
def long_source(tmp_path):
    source = _long_source(tmp_path, 12)
    yield source
    source.conn.close()


def _read(source, output, **changes):
    return entry.read(source.repository, source.snapshot, output, **{**HISTORY_REQUEST, **changes})


def _forbid(*args, **kwargs):
    raise AssertionError("forbidden path was called")


def test_native_roundtrip_math_lineage_units_identity_and_replay(source, monkeypatch):
    forbidden = [e["event_date"].date().isoformat() for e in source.forbidden]
    forbidden.append("2024-06-17")  # Latest target is never historical input.
    poison_outcomes(monkeypatch, source.repository, source.snapshot, forbidden)
    before_source = Repository.resolve_pinned(source.repository, "shadow")
    build, output = build_history(source)
    context = build["context"]
    committed = Repository.resolve_pinned(source.repository,
        "magnitude-history-" + context["context_id"].removeprefix("sha256:"))
    assert output == committed
    assert build == {"context": context, "feature_snapshot": to_document(committed)}
    assert set(context) == {"schema_version", "context_id", "source_snapshot", "temporal_plan",
                            "history_start", "feature_recipe", "producer"}
    assert context["schema_version"] == "magnitude_materialization.v1"
    assert context["temporal_plan"]["schema_version"] == "magnitude_temporal_plan.v1"
    assert context["history_start"] == "2022-01-01"
    assert set(context["producer"]) == {"source_closure", "environment"}
    assert output != source.snapshot and set(output.table_versions) == {TABLE}
    assert Repository.resolve_pinned(source.repository, "shadow") == before_source
    assert output.knowledge_mode_by_table == {TABLE: "reconstructed"}
    assert context["context_id"] == content_hash({k: v for k, v in context.items() if k != "context_id"})
    assert context["source_snapshot"] == to_document(source.snapshot)
    closure = context["producer"]["source_closure"]
    root = Path(entry.__file__).resolve().parents[1]
    assert {"experiments/magnitude_history_store.py", "engine/v2/research/magnitude_history_store.py",
            "engine/v2/research/event_history.py", "engine/v2/features/panel_math.py",
            "engine/v2/research/magnitude_history_table.py", "engine/v2/research/magnitude_features.py",
            "engine/v2/research/magnitude_inputs.py", "engine/v2/research/magnitude_folds.py",
            "engine/v2/research/experiment_population.py", "engine/v2/foundation/experiment_holdouts.py",
            "engine/v2/foundation/market_calendar.py"} <= closure.keys()
    assert closure == {name: "sha256:" + hashlib.sha256((root / name).read_bytes()).hexdigest() for name in closure}
    libraries = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow"):
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libraries[name] = None
    assert context["producer"]["environment"] == {
        "python": platform.python_version(), "machine": platform.machine(),
        "libraries": libraries, "numerical_threads": 1}
    assert context["feature_recipe"]["features"] == [
        {"name": name, "on_missing": "zero_count" if name == "n_prior" else "null"} for name in FEATURES]
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(magnitude_features, "load_magnitude_history", _forbid)
    monkeypatch.setattr(event_history, "load_event_history", _forbid)
    monkeypatch.setattr(stored, "_preflight_source", _forbid)
    first = _read(source, output)
    assert set(first) == {"context", "feature_snapshot", "rows", "inputs_id"}
    assert first["context"] == context and first["feature_snapshot"] == to_document(committed)
    assert first == _read(source, output)  # Two separate calls pin stored-read determinism.
    assert build_history(source) == (build, output)  # Metadata admission, then artifact-only replay.
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    assert first["inputs_id"] == content_hash({"context": context,
        "feature_snapshot": to_document(committed), "rows": first["rows"]})
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


def test_native_persisted_rows_omit_exclusion_roster_but_keep_transient_history(source, monkeypatch):
    poison_outcomes(monkeypatch, source.repository, source.snapshot,
            [e["event_date"].date().isoformat() for e in [*source.forbidden, source.events[-1]]])
    transient = event_history.load_event_history(source.repository, source.snapshot,
        event_id=source.events[-1]["event_id"], as_of_month=HISTORY_REQUEST["as_of_month"],
        history_start=HISTORY_REQUEST["history_start"], decision_session="2024-06-14")
    exclusions = {r["event_id"]: r["memberships"] for r in transient["holdout_exclusions"]}
    assert exclusions and source.forbidden[0]["event_id"] in exclusions
    built, output = build_history(source)
    persisted = []
    for record in source.repository.fragment_records(output, TABLE):
        path = objects.verify_object_path(source.store, record.object_ref)
        persisted.extend(pq.read_table(path).to_pylist())
    assert len(persisted) == 12
    for row in persisted:
        lineage = json.loads(row["history_json"])
        assert "holdout_exclusions" not in lineage
        assert lineage["history_id"] == content_hash({k: v for k, v in lineage.items()
            if k not in ("history_id", "history_status")})
    serialized = canonical_json({"build": built, "rows": persisted})
    assert all(event_id not in serialized for event_id in exclusions)
    latest = next(r for r in persisted if r["event_date"] == "2024-06-17" and r["horizon_sessions"] == 1)
    assert json.loads(latest["history_json"])["history_id"] == transient["history_id"]


def test_empty_and_zero_history_keep_distinct_missingness(source):
    _, output = build_history(source, history_start="2022-12-01")
    rows = _read(source, output, history_start="2022-12-01")["rows"]
    empty = next(r for r in rows if r["event_date"] == "2023-01-03")
    assert empty["n_prior"] == 0 and empty["history_status"] == "empty"
    assert all(empty[name] is None for name in FEATURES[1:])
    assert json.loads(empty["history_json"])["history"] == []
    zero = next(r for r in rows if r["event_date"] == "2024-06-12")
    assert zero["n_prior"] == 2 and zero["mean_prior_move"] == 1 and zero["mean_prior_abs_move"] == 5


def test_two_issuers_with_overlapping_dates_keep_exact_stored_features_and_lineage(tmp_path, monkeypatch):
    dates = ("2022-06-02", "2023-01-03", "2023-06-02", "2024-06-12", "2024-06-17")
    events = {ticker: [history_event(day, ticker=ticker) for day in dates] for ticker in ("SYNTH", "SYNTH_B")}
    values = {"SYNTH": [-2., -4., 6., 10., None], "SYNTH_B": [2., 8., -12., 20., None]}
    moves = {ticker: [outcome_move(event, value) for event, value in zip(group, values[ticker])]
             for ticker, group in events.items()}
    conn, repository, snapshot = commit_history_inputs(tmp_path,
        [event for pair in zip(*events.values()) for event in reversed(pair)],
        [move for pair in zip(*moves.values()) for move in pair],
        days=calendar_sessions("2022-05-25", "2024-06-18").days)
    source = SimpleNamespace(conn=conn, repository=repository, snapshot=snapshot, store=repository._store)
    selected = [events[ticker][i]["event_id"] for i in (2, 4) for ticker in ("SYNTH_B", "SYNTH")]
    by_id = {event["event_id"]: (ticker, i) for ticker, group in events.items() for i, event in enumerate(group)}
    # Independent tuples: signed/absolute means, EMA2 pair, optional EMA4 pair.
    expected = {"SYNTH": {2: (-3., 3., -10 / 3, 10 / 3, None, None),
        3: (0., 4., 26 / 9, 46 / 9, None, None),
        4: (2.5, 5.5, 206 / 27, 226 / 27, 554 / 125, 806 / 125)},
        "SYNTH_B": {2: (5., 5., 6., 6., None, None),
        3: (-2 / 3, 22 / 3, -6., 10., None, None),
        4: (4.5, 10.5, 34 / 3, 50 / 3, 838 / 125, 1558 / 125)}}
    try:
        records = {(r.partition_key, r.time_min): r for r in repository.fragment_records(snapshot, "computed_moves")}
        poison_outcomes(monkeypatch, repository, snapshot, [dates[-1]])
        _, output = build_history(source, event_ids=selected)
        poison_outcomes(monkeypatch, repository, snapshot, [], all_outcomes=True)
        monkeypatch.setattr(stored, "_prepare_history", _forbid)
        before = catalog_rows(conn), artifact_object_bytes(source.store)
        rows = _read(source, output, event_ids=selected)["rows"]
        assert {(row["event_id"], row["horizon_sessions"]) for row in rows} == {
            (event_id, horizon) for event_id in selected for horizon in (1, 3, 5)}
        assert len(rows) == 12
        for row in rows:
            ticker, index = by_id[row["event_id"]]
            count = 2 if index == 2 else 4 if row["horizon_sessions"] == 1 else 3
            mean, absolute, ema2, abs2, ema4, abs4 = expected[ticker][count]
            wanted = dict(n_prior=count, mean_prior_move=mean, mean_prior_abs_move=absolute,
                ema2_prior_move=ema2, ema2_prior_abs_move=abs2, ema4_prior_move=ema4,
                ema4_prior_abs_move=abs4, ema8_prior_move=None, ema8_prior_abs_move=None,
                ema12_prior_move=None, ema12_prior_abs_move=None)
            assert {name: row[name] for name in wanted} == pytest.approx(wanted)
            lineage = json.loads(row["history_json"])
            assert lineage["ticker"] == ticker and lineage["event_id"] == row["event_id"]
            assert lineage["omitted_history"] == []
            history, inputs = [], []
            for event, move in zip(events[ticker][:count], moves[ticker][:count]):
                record = records[ticker, move["event_date"]]
                identity = dict(event_id=event["event_id"], fragment_id=record.fragment_id,
                    object_id=record.object_ref.object_id, content_hash=record.object_ref.content_hash)
                inputs.append(identity)
                history.append({**identity, "event_date": move["event_date"],
                                "target_available_on": move["available_as_of_date"]})
            assert lineage["history"] == history and lineage["read_inputs"] == inputs
        assert (catalog_rows(conn), artifact_object_bytes(source.store)) == before
    finally:
        conn.close()


@pytest.mark.parametrize("count,expected", [
    (8, {2: (6014 / 729, 32806 / 2187), 4: (324278 / 78125, 1022186 / 78125),
         8: (2912650 / 1594323, 48811522 / 4782969)}),
    (12, {2: (723350 / 59049, 4074382 / 177147),
          4: (301891518 / 48828125, 1025922066 / 48828125),
          8: (31443810178 / 10460353203, 547319300554 / 31381059609),
          12: (3219945174670 / 1792160394037, 26436513499202 / 1792160394037)}),
])
def test_eight_and_twelve_event_histories_store_independent_emas(tmp_path, monkeypatch, count, expected):
    # Rational oracles use q**(n-1)*x[0] + sum(a*q**(n-i-1)*x[i]),
    # where a=2/(span+1), q=1-a; no production EMA helper or recurrence.
    source = _long_source(tmp_path, count)
    try:
        target = source.events[-1]
        poison_outcomes(monkeypatch, source.repository, source.snapshot, ["2024-06-17"])
        _, output = build_history(source)
        poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
        monkeypatch.setattr(stored, "_prepare_history", _forbid)
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        rows = [row for row in _read(source, output)["rows"] if row["event_id"] == target["event_id"]]
        assert [row["horizon_sessions"] for row in rows] == [1, 3, 5]
        for row in rows:
            assert row["n_prior"] == count
            assert row["mean_prior_move"] == 1 and row["mean_prior_abs_move"] == count + 1
            lineage = json.loads(row["history_json"])
            assert [item["event_id"] for item in lineage["history"]] == [
                event["event_id"] for event in source.events[:-1]]
            for span in (2, 4, 8, 12):
                if span <= count:
                    signed, absolute = expected[span]
                    assert row[f"ema{span}_prior_move"] == pytest.approx(signed, rel=1e-12)
                    assert row[f"ema{span}_prior_abs_move"] == pytest.approx(absolute, rel=1e-12)
                else:
                    assert row[f"ema{span}_prior_move"] is None
                    assert row[f"ema{span}_prior_abs_move"] is None
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    finally:
        source.conn.close()


def test_later_mixed_candidate_refuses_before_even_first_safe_outcome(tmp_path, monkeypatch):
    source = _source(tmp_path, mixed_days=("2023-06-02", "2024-06-12"))
    try:
        poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
            build_history(source)
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    finally:
        source.conn.close()


def test_non_native_computed_moves_contract_refuses_before_outcome_objects(tmp_path, monkeypatch):
    from engine.v2.data.manifests import table_contract_hash
    foreign = replace(stored.COMPUTED_MOVES_CONTRACT, contract_id="computed_moves.v0", semantic_version="0.0.0")
    foreign = replace(foreign, definition_hash=table_contract_hash(foreign))
    with monkeypatch.context() as fixture_contract:
        fixture_contract.setattr(support, "COMPUTED_MOVES_CONTRACT", foreign)
        source = _source(tmp_path)
    try:
        assert source.repository.table_contract(source.snapshot, "computed_moves") == foreign
        assert stored.COMPUTED_MOVES_CONTRACT != foreign
        assert source.repository.resolve(source.snapshot.snapshot_id) == source.snapshot
        poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
            build_history(source)
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    finally:
        source.conn.close()


@pytest.mark.parametrize("change", [
    {"as_of_month": "2026-02"}, {"history_start": "2022-02-01"},
    {"training_start": "2023-02-01"},
])
def test_changed_context_cannot_reuse_feature_snapshot(source, monkeypatch, change):
    _, output = build_history(source)
    monkeypatch.setattr(stored, "_scan", _forbid)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        _read(source, output, **change)


@pytest.mark.parametrize("field", ["python", "machine", "libraries", "numerical_threads"])
def test_environment_only_drift_changes_context_and_refuses_old_feature_snapshot(source, monkeypatch, field):
    built, output = build_history(source)
    original = built["context"]
    environment = json.loads(json.dumps(original["producer"]["environment"]))
    environment[field] = ({**environment[field], "numpy": "synthetic-drift"} if field == "libraries"
        else 2 if field == "numerical_threads" else environment[field] + "-synthetic-drift")
    calls = []

    def changed_environment(threads):
        calls.append(threads)
        return environment

    monkeypatch.setattr(entry, "environment_identity", changed_environment)
    changed = entry.context(source.repository, source.snapshot, **HISTORY_REQUEST)
    identity = {**{k: v for k, v in original.items() if k != "context_id"},
                "producer": {**original["producer"], "environment": environment}}
    assert changed == {**identity, "context_id": content_hash(identity)}
    assert changed["context_id"] != original["context_id"]
    poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    monkeypatch.setattr(stored, "_scan", _forbid)
    monkeypatch.setattr(stored, "_prepare_history", _forbid)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        _read(source, output)
    assert calls == [1, 1]
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("path", ["engine/v2/features/panel_math.py",
    "engine/v2/research/magnitude_history_table.py", "engine/v2/foundation/experiment_holdouts.py",
    "engine/v2/research/magnitude_folds.py"])
def test_source_only_drift_changes_context_and_refuses_old_feature_snapshot(source, monkeypatch, path):
    built, output = build_history(source)
    original = built["context"]
    closure = original["producer"]["source_closure"]
    assert path in closure and closure[path] != "sha256:" + "0" * 64
    changed_sources = {**closure, path: "sha256:" + "0" * 64}
    calls = []

    def changed_closure(root, entries):
        assert root == Path(entry.__file__).resolve().parents[1]
        assert entries == ["experiments/magnitude_history_store.py"]
        calls.append(1)
        return changed_sources

    monkeypatch.setattr(entry, "source_closure", changed_closure)
    changed = entry.context(source.repository, source.snapshot, **HISTORY_REQUEST)
    identity = {**{k: v for k, v in original.items() if k != "context_id"},
                "producer": {**original["producer"], "source_closure": changed_sources}}
    assert changed == {**identity, "context_id": content_hash(identity)}
    assert changed["context_id"] != original["context_id"]
    poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    monkeypatch.setattr(stored, "_scan", _forbid)
    monkeypatch.setattr(stored, "_prepare_history", _forbid)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        _read(source, output)
    assert calls == [1, 1]
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("changes,code", [
    ({"evaluation_years": ["2021"], "training_start": "2020-01-01"}, "POPULATION_COLLAPSED"),
    ({"evaluation_years": ["2022"], "training_start": "2021-01-01"}, "POPULATION_COLLAPSED"),
    ({"history_start": "2024-01-01"}, "CONTRACT_MISMATCH"),
])
def test_empty_folds_and_late_history_start_refuse_before_build_payloads(source, monkeypatch, changes, code):
    poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    monkeypatch.setattr(stored, "_prepare_history", _forbid)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(DataError, match=code):
        build_history(source, **changes)
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_source_or_output_reference_drift_refuses_before_feature_payload(source, monkeypatch):
    _, output = build_history(source)
    monkeypatch.setattr(stored, "_scan", _forbid)
    with pytest.raises(DataError, match="SNAPSHOT_UNRESOLVED"):
        entry.build(source.repository, replace(source.snapshot, calendar_version="changed"), **HISTORY_REQUEST)
    with pytest.raises(DataError, match="SNAPSHOT_UNRESOLVED"):
        entry.read(source.repository, replace(source.snapshot, calendar_version="changed"), output, **HISTORY_REQUEST)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        _read(source, replace(output, manifest_hash="sha256:" + "a" * 64))
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        _read(source, source.snapshot)


def test_success_receipt_is_immutable_native_catalog_evidence(source):
    built, output = build_history(source)
    _, receipt = stored._names(built["context"])
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        source.conn.execute("UPDATE data_import_receipts SET scope='wrong' WHERE receipt_id=?", (receipt,))
    assert len(_read(source, output)["rows"]) == 12


@pytest.mark.parametrize("field,value", [("scope", "wrong"), ("attempt_id", "wrong"),
                                        ("fence", 7), ("request_hash", "sha256:" + "a" * 64)])
def test_colliding_native_success_receipt_refuses_before_outcomes(source, monkeypatch, field, value):
    binding = entry.context(source.repository, source.snapshot, **HISTORY_REQUEST)
    scope, receipt = stored._names(binding)
    parent = source.repository.resolve_full(source.snapshot.snapshot_id)
    request = dict(scope=scope, request_hash=binding["context_id"], receipt_id=receipt,
        attempt_id=receipt, fence=1, expected_head_snapshot_id=None, expected_head_generation=0,
        contracts=parent.contracts, objects=parent.objects, records=parent.records,
        manifests=tuple(parent.table_manifests.values()), snapshot=parent.snapshot,
        fence_check=lambda conn: None, clock=SystemClock(), store=source.store, audit_partitions=False)
    catalog.commit_snapshot(source.conn, **{**request, field: value})
    monkeypatch.setattr(stored, "_preflight_source", _forbid)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        build_history(source)
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("status", ["failed", "conflict"])
@pytest.mark.parametrize("action", ["build", "read"])
def test_noncommitted_matching_receipt_refuses_before_result_resolution(source, monkeypatch, status, action):
    binding = entry.context(source.repository, source.snapshot, **HISTORY_REQUEST)
    scope, receipt = stored._names(binding)
    # The failed-import helper leaves scope empty; insert a valid native row so only status differs.
    source.conn.execute("INSERT INTO data_import_receipts "
        "(receipt_id, attempt_id, fence, source_manifest_hash, result_snapshot_id, status, registered_at, scope) "
        "VALUES (?, ?, 1, ?, NULL, ?, '2026-01-01T00:00:00Z', ?)",
        (receipt, receipt, binding["context_id"], status, scope))
    poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    monkeypatch.setattr(stored, "_preflight_source", _forbid)
    monkeypatch.setattr(stored, "_scan", _forbid)
    resolve = source.repository.resolve

    def guarded(snapshot_id):
        assert snapshot_id is not None, "noncommitted result was resolved"
        return resolve(snapshot_id)

    monkeypatch.setattr(source.repository, "resolve", guarded)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        build_history(source) if action == "build" else _read(source, source.snapshot)
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_all_metadata_is_admitted_before_any_feature_column_decode(source, monkeypatch):
    original = stored._event_rows

    def malformed(*args):
        rows = original(*args)
        if rows[0]["event_date"] == "2024-06-17":
            rows[-1]["cutoff_id"] = "wrong"
        return rows

    monkeypatch.setattr(stored, "_event_rows", malformed)
    _, output = build_history(source)
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


@pytest.mark.parametrize("name,value", [
    ("event_id", "wrong"), ("context_id", "wrong"), ("horizon_sessions", 2),
    ("event_date", "2024-06-16"), ("cutoff_id", "wrong"), ("source_session_ceiling", "2024-06-01"),
])
def test_each_decoded_metadata_pin_refuses_before_feature_columns(source, monkeypatch, name, value):
    _, output = build_history(source)
    scan = stored._scan
    seen = []

    def malformed(repository, snapshot, context, columns, count):
        assert not set(columns) & set(FEATURES), "feature decode preceded complete metadata admission"
        rows = scan(repository, snapshot, context, columns, count)
        rows[-1][name] = value  # Exercise row admission independently of physical fragment guards.
        seen.append(columns)
        return rows

    monkeypatch.setattr(stored, "_scan", malformed)
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
    _, output = build_history(source)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _read(source, output)
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def _observe_long_target_validation(source, monkeypatch, field):
    validate, observed = stored._validate_features, []

    def checked(row, count):
        if row["event_id"] == source.events[-1]["event_id"] and row["horizon_sessions"] == 5:
            assert count == row["n_prior"] == 12
            observed.append(row[field])
        return validate(row, count)

    monkeypatch.setattr(stored, "_validate_features", checked)
    return observed


@pytest.mark.parametrize("field,value", [
    ("ema8_prior_move", None), ("ema8_prior_abs_move", None),
    ("ema12_prior_move", None), ("ema12_prior_abs_move", None),
    ("ema2_prior_abs_move", -1.), ("ema4_prior_abs_move", -1.),
    ("ema8_prior_abs_move", -1.), ("ema12_prior_abs_move", -1.),
])
def test_sufficient_history_stored_null_and_negative_emas_refuse(long_source, monkeypatch, field, value):
    source, original, changed = long_source, stored._event_rows, []

    def malformed(*args):
        rows = original(*args)
        if rows[-1]["event_id"] == source.events[-1]["event_id"]:
            assert rows[-1]["n_prior"] == 12
            rows[-1][field] = value
            changed.append(field)
        return rows

    monkeypatch.setattr(stored, "_event_rows", malformed)
    _, output = build_history(source)  # Null and negative finite values pass native physical inspection.
    assert changed == [field]
    poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    monkeypatch.setattr(stored, "_prepare_history", _forbid)
    observed = _observe_long_target_validation(source, monkeypatch, field)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH") as failure:
        _read(source, output)
    assert failure.value.problem.message == "stored history feature or missingness is invalid"
    assert observed == [value]
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("field", ["ema8_prior_move", "ema8_prior_abs_move",
                                  "ema12_prior_move", "ema12_prior_abs_move"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_sufficient_history_nonfinite_emas_refuse_after_real_feature_scan(long_source, monkeypatch, field, value):
    source = long_source
    _, output = build_history(source)
    poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    monkeypatch.setattr(stored, "_prepare_history", _forbid)
    scan, seen, changed = stored._scan, [], []

    def malformed(repository, snapshot, context, columns, count):
        rows = scan(repository, snapshot, context, columns, count)
        seen.append(tuple(columns))
        if tuple(columns) == (*stored.METADATA, *FEATURES):
            for row in rows:
                if row["event_id"] == source.events[-1]["event_id"] and row["horizon_sessions"] == 5:
                    assert row["n_prior"] == 12
                    row[field] = value  # Native inspection rejects infinities before publication.
                    changed.append(field)
        return rows

    monkeypatch.setattr(stored, "_scan", malformed)
    observed = _observe_long_target_validation(source, monkeypatch, field)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH") as failure:
        _read(source, output)
    assert failure.value.problem.message == "stored history feature or missingness is invalid"
    assert len(observed) == 1 and observed[0] is value
    assert seen == [tuple(stored.METADATA), (*stored.METADATA, *FEATURES)] and changed == [field]
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_native_inspection_refuses_nonfinite_features_before_commit(source, monkeypatch, value):
    original = stored._event_rows

    def malformed(*args):
        rows = original(*args)
        rows[-1]["mean_prior_move"] = value
        return rows

    monkeypatch.setattr(stored, "_event_rows", malformed)
    before = catalog_rows(source.conn)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        build_history(source)
    assert catalog_rows(source.conn) == before


def test_global_fragment_preflight_precedes_all_feature_object_opens(source, monkeypatch):
    _, output = build_history(source)
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
    before = catalog_rows(source.conn)
    attempts = []

    def interrupted(*args, **kwargs):
        attempts.append(1)
        def fault(point):
            if point == "before_commit":
                raise RuntimeError("synthetic interruption")
        return commit(*args, **kwargs, fault=fault)

    monkeypatch.setattr(catalog, "commit_snapshot", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        build_history(source)
    assert attempts == [1]
    assert catalog_rows(source.conn) == before
    monkeypatch.setattr(catalog, "commit_snapshot", commit)
    _, output = build_history(source)
    assert len(_read(source, output)["rows"]) == 12


def test_existing_context_scope_conflict_preserves_head(source):
    binding = entry.context(source.repository, source.snapshot, **HISTORY_REQUEST)
    scope, _ = stored._names(binding)
    source.conn.execute("INSERT INTO data_snapshot_heads SELECT ?, snapshot_id, generation, updated_at, update_receipt_ref "
                        "FROM data_snapshot_heads WHERE scope='shadow'", (scope,))
    before = catalog_rows(source.conn)
    with pytest.raises(DataError, match="SNAPSHOT_CONFLICT"):
        build_history(source)
    assert catalog_rows(source.conn) == before


def test_replay_after_head_moves_returns_original_artifact_without_head_effect(source, monkeypatch):
    built, output = build_history(source)
    scope, _ = stored._names(built["context"])
    source.conn.execute("UPDATE data_snapshot_heads SET snapshot_id=?, generation=generation+1 WHERE scope=?",
                        (source.snapshot.snapshot_id, scope))
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(magnitude_features, "load_magnitude_history", _forbid)
    assert build_history(source) == (built, output)
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_replay_reverifies_tampered_stored_artifact_without_rebuild_or_publication(source, monkeypatch):
    _, output = build_history(source)
    record = source.repository.fragment_records(output, TABLE)[0]
    path = objects.verify_object_path(source.store, record.object_ref)
    path.chmod(0o644)
    with path.open("ab") as handle:
        handle.write(b"x")  # Changed size invalidates the native checksum cache.
    poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    monkeypatch.setattr(stored, "_prepare_history", _forbid)
    monkeypatch.setattr(source.store, "publish_bytes", _forbid)
    monkeypatch.setattr(catalog, "commit_snapshot", _forbid)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(DataError, match="OBJECT_CORRUPT"):
        build_history(source)
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def _cli_args(source):
    return ["--catalog", str(source.root / "catalog.sqlite"), "--store-root", str(source.store.root),
        "--source-snapshot-id", source.snapshot.snapshot_id, "--as-of-month", HISTORY_REQUEST["as_of_month"],
        "--evaluation-years", *HISTORY_REQUEST["evaluation_years"], "--training-start", HISTORY_REQUEST["training_start"],
        "--history-start", HISTORY_REQUEST["history_start"]]


def test_cli_build_read_and_typed_refusal(source, capsys):
    args = _cli_args(source)
    assert entry.main(["build", *args]) == 0
    built = json.loads(capsys.readouterr().out)
    assert entry.main(["read", *args, "--feature-snapshot-id", built["feature_snapshot"]["snapshot_id"]]) == 0
    rows = json.loads(capsys.readouterr().out)["rows"]
    assert len(rows) == 12
    row = next(r for r in rows if r["event_date"] == "2024-06-12" and r["horizon_sessions"] == 1)
    assert row["n_prior"] == 3 and row["mean_prior_abs_move"] == 4
    assert row["ema2_prior_abs_move"] == pytest.approx(46 / 9)
    assert entry.main(["read", *args, "--feature-snapshot-id", source.snapshot.snapshot_id]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"refused": "IDENTITY_CONFLICT",
        "message": "IDENTITY_CONFLICT: feature snapshot is not the committed context output"}
    assert all(line.startswith("  [") for line in captured.err.splitlines())  # progress only
    with pytest.raises(SystemExit) as error:
        entry.main(["read", *args])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "read requires --feature-snapshot-id; build does not accept it" in captured.err and captured.out == ""
    with pytest.raises(SystemExit) as error:
        entry.main(["build", *args, "--feature-snapshot-id", source.snapshot.snapshot_id])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "read requires --feature-snapshot-id; build does not accept it" in captured.err and captured.out == ""


@pytest.mark.parametrize("action", ["build", "read"])
@pytest.mark.parametrize("kind,expected", [
    ("data", {"refused": "SNAPSHOT_NOT_FOUND", "message": "SNAPSHOT_NOT_FOUND: unknown snapshot id"}),
    ("ops", {"refused": "INTEGRITY_FAILED",
             "message": "INTEGRITY_FAILED: ops migration differs from the one applied"}),
])
def test_cli_native_typed_errors_preserve_complete_envelope_and_close_catalog(
        source, monkeypatch, capsys, action, kind, expected):
    args = _cli_args(source)
    if kind == "ops":
        changed = source.conn.execute("UPDATE schema_versions SET checksum='synthetic-changed' "
                                      "WHERE owner='ops' AND version=1")
        assert changed.rowcount == 1
    else:
        args[5] = "unknown-synthetic-source"
    if action == "read":
        args += ["--feature-snapshot-id", "unused"]
    opened = []
    connect = bootstrap.connect

    def track(*args, **kwargs):
        conn = connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(bootstrap, "connect", track)
    poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    assert entry.main([action, *args]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == expected and captured.err == ""
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")


@pytest.mark.parametrize("action", ["build", "read"])
def test_cli_bootstrap_migrates_each_owner_before_source_resolution(source, capsys, action):
    path = source.root / "empty.sqlite"
    sqlite3.connect(path).close()
    args = _cli_args(source)
    args[1] = str(path)
    if action == "read":
        args += ["--feature-snapshot-id", "unused"]
    assert entry.main([action, *args]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"refused": "SNAPSHOT_NOT_FOUND",
                                       "message": "SNAPSHOT_NOT_FOUND: unknown snapshot id"}
    assert captured.err == ""
    with sqlite3.connect(path) as conn:
        versions = conn.execute("SELECT * FROM schema_versions ORDER BY owner, version").fetchall()
        assert {row[0] for row in versions} == {"ops", "ledger", "data"}
        assert conn.execute("SELECT COUNT(*) FROM data_snapshots").fetchone()[0] == 0
    assert entry.main([action, *args]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"refused": "SNAPSHOT_NOT_FOUND",
                                       "message": "SNAPSHOT_NOT_FOUND: unknown snapshot id"}
    assert captured.err == ""
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT * FROM schema_versions ORDER BY owner, version").fetchall() == versions


@pytest.mark.parametrize("action", ["build", "read"])
def test_cli_unknown_option_refuses_before_execution_and_bootstrap(source, monkeypatch, capsys, action):
    args = _cli_args(source)
    if action == "read":
        args += ["--feature-snapshot-id", "unused"]
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    monkeypatch.setattr(entry, "open_catalog", _forbid)
    monkeypatch.setattr(entry, "_execute", _forbid)
    with pytest.raises(SystemExit) as error:
        entry.main([action, *args, "--caller-features", "unused"])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "--caller-features" in captured.err and captured.out == ""
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("action", ["build", "read"])
def test_cli_missing_catalog_refuses_before_bootstrap_or_directory_creation(tmp_path, monkeypatch, capsys, action):
    missing = tmp_path / "missing" / "catalog.sqlite"
    args = [action, "--catalog", str(missing), "--store-root", str(tmp_path / "objects"),
        "--source-snapshot-id", "source", "--as-of-month", "2026-01", "--evaluation-years", "2024",
        "--training-start", "2023-01-01", "--history-start", "2022-01-01"]
    if action == "read":
        args += ["--feature-snapshot-id", "features"]
    monkeypatch.setattr(entry, "open_catalog", _forbid)
    with pytest.raises(SystemExit) as error:
        entry.main(args)
    assert error.value.code == 2
    assert "existing native catalog file" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_cli_event_subset_build_and_read_keep_full_eligible_history(source, monkeypatch, capsys):
    selected = [source.events[1]["event_id"], source.events[-1]["event_id"]]
    args = _cli_args(source)
    for event_id in selected:
        args += ["--event-id", event_id]
    poison_outcomes(monkeypatch, source.repository, source.snapshot,
            [e["event_date"].date().isoformat() for e in [*source.forbidden, source.events[-1]]])
    assert entry.main(["build", *args]) == 0
    built = json.loads(capsys.readouterr().out)
    assert [e["event_id"] for e in built["context"]["temporal_plan"]["events"]] == selected
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    assert entry.main(["read", *args, "--feature-snapshot-id", built["feature_snapshot"]["snapshot_id"]]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["context"] == built["context"]
    assert {r["event_id"] for r in result["rows"]} == set(selected) and len(result["rows"]) == 6
    latest = {r["horizon_sessions"]: r for r in result["rows"] if r["event_id"] == selected[-1]}
    assert {h: r["mean_prior_abs_move"] for h, r in latest.items()} == {1: 5.5, 3: 4., 5: 4.}
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("action", ["build", "read"])
@pytest.mark.parametrize("kind", ["missing", "nested_missing", "file"])
def test_cli_invalid_store_root_refuses_before_bootstrap_or_store_construction(
        source, monkeypatch, capsys, action, kind):
    root = source.root / "invalid-store-root"
    if kind == "nested_missing":
        root = root / "objects"
    if kind == "file":
        root.write_text("synthetic non-directory store path")
    args = _cli_args(source)
    args[3] = str(root)
    if action == "read":
        args += ["--feature-snapshot-id", "unused"]
    monkeypatch.setattr(entry, "_execute", _forbid)
    monkeypatch.setattr(entry, "open_catalog", _forbid)
    monkeypatch.setattr(entry, "ArtifactStore", _forbid)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    assert entry.main([action, *args]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"refused": "UNSAFE_PATH"} and captured.err == ""
    assert root.exists() == (kind == "file")
    if kind == "nested_missing":
        assert not root.parent.exists()
    if kind == "file":
        assert root.read_text() == "synthetic non-directory store path"
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("action", ["build", "read"])
def test_cli_existing_empty_store_reaches_native_object_admission(source, capsys, action):
    _, output = build_history(source)
    root = source.root / "empty-store"
    root.mkdir()
    args = _cli_args(source)
    args[3] = str(root)
    if action == "read":
        args += ["--feature-snapshot-id", output.snapshot_id]
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    assert entry.main([action, *args]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"refused": "OBJECT_CORRUPT",
        "message": "OBJECT_CORRUPT: published object bytes do not match its recorded hash"}
    assert all(line.startswith("  [") for line in captured.err.splitlines())  # progress only
    assert list(root.iterdir()) == []
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("action", ["build", "read"])
@pytest.mark.parametrize("fault", ["stat_os", "store_stat_os", "open_os", "open_sqlite", "store_os", "store_artifact"])
def test_cli_raw_local_failures_are_redacted_and_close_open_catalogs(source, monkeypatch, capsys, action, fault):
    args = _cli_args(source)
    if action == "read":
        args += ["--feature-snapshot-id", "unused"]
    opened = []
    original = entry.open_catalog

    def fail(*args, **kwargs):
        if fault == "store_artifact":
            raise ArtifactError("UNSAFE_PATH", "unpublishable exception detail")
        raise OSError("unpublishable exception detail")

    def track(*args, **kwargs):
        conn = original(*args, **kwargs)
        opened.append(conn)
        return conn

    if fault == "stat_os":
        monkeypatch.setattr(entry, "Path", lambda path: SimpleNamespace(is_file=fail))
        monkeypatch.setattr(entry, "open_catalog", _forbid)
    elif fault == "store_stat_os":
        monkeypatch.setattr(entry, "Path", lambda path: SimpleNamespace(is_file=lambda: True, is_dir=fail))
        monkeypatch.setattr(entry, "open_catalog", _forbid)
        monkeypatch.setattr(entry, "ArtifactStore", _forbid)
    elif fault == "open_os":
        monkeypatch.setattr(entry, "open_catalog", fail)
        monkeypatch.setattr(entry, "ArtifactStore", _forbid)
    elif fault == "open_sqlite":
        invalid = source.root / "invalid.sqlite"
        invalid.write_text("synthetic invalid SQLite contents")
        args[1] = str(invalid)
        monkeypatch.setattr(entry, "ArtifactStore", _forbid)
    else:
        monkeypatch.setattr(entry, "open_catalog", track)
        monkeypatch.setattr(entry, "ArtifactStore", fail)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    assert entry.main([action, *args]) == 2
    captured = capsys.readouterr()
    expected = "UNSAFE_PATH" if fault == "store_artifact" else "INPUT_CHANGED"
    assert json.loads(captured.out) == {"refused": expected} and captured.err == ""
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    assert len(opened) == (1 if fault in ("store_os", "store_artifact") else 0)
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")


def test_recipe_and_physical_contract_are_closed():
    names = ("n_prior", "mean_prior_move", "mean_prior_abs_move", "ema2_prior_move", "ema2_prior_abs_move",
        "ema4_prior_move", "ema4_prior_abs_move", "ema8_prior_move", "ema8_prior_abs_move",
        "ema12_prior_move", "ema12_prior_abs_move")
    assert FEATURES == names
    assert stored.feature_recipe() == {"schema_version": "stored_magnitude_history.v1",
        "arithmetic": "panel_math.history_features", "units": "percentage_points", "imputation": "none",
        "features": [{"name": name, "on_missing": "zero_count" if name == "n_prior" else "null"}
                     for name in names]}
    assert CONTRACT.primary_key == ("event_id", "context_id", "horizon_sessions")
    assert CONTRACT.partition_columns == ("context_id",)
    assert [c.name for c in CONTRACT.columns] == [*stored.METADATA, *FEATURES]
    assert all(c.nullable == (c.name in FEATURES[1:]) for c in CONTRACT.columns)
    assert len(FEATURES) == 11
    assert {c.name: c.unit for c in CONTRACT.columns if c.name in names} == {
        name: "count" if name == "n_prior" else "percent" for name in names}
    assert {c.name: c.null_policy for c in CONTRACT.columns if c.name in names} == {
        name: "zero count" if name == "n_prior" else "null when history is insufficient" for name in names}


def test_context_or_source_mismatch_refuses_before_any_source_or_feature_payload(source, monkeypatch):
    binding = entry.context(source.repository, source.snapshot, **HISTORY_REQUEST)
    poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        stored.materialize(source.repository, source.snapshot, {**binding, "history_start": "2022-02-01"})
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        stored.materialize(source.repository, replace(source.snapshot, calendar_version="different"), binding)
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("action", ["materialize", "read_stored"])
@pytest.mark.parametrize("missing", ["context_id", "source_snapshot", "temporal_plan", "history_start",
                                    "not_a_context", "null_plan"])
def test_missing_direct_context_identity_has_a_typed_refusal_before_payloads(source, monkeypatch, action, missing):
    binding = entry.context(source.repository, source.snapshot, **HISTORY_REQUEST)
    if missing == "not_a_context":
        binding = None
    else:
        if missing == "null_plan":
            binding["temporal_plan"] = None
        else:
            del binding[missing]
        if missing != "context_id":
            binding["context_id"] = content_hash({k: v for k, v in binding.items() if k != "context_id"})
    poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT"):
        getattr(stored, action)(source.repository, source.snapshot, binding)
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("scope,name,value", [
    ("event", "ticker", "OTHER"), ("event", "event_date", "2024-06-18"),
    ("event", "session", "BMO"), ("event", "population_use", "wrong"),
    ("event", "event_id", "unknown"), ("event", "event_id", "protected"),
    ("event", "forecasts", []), ("cutoff", "horizon_sessions", 3),
    ("cutoff", "horizon_sessions", True), ("cutoff", "event_date", "2024-06-18"),
    ("cutoff", "session", "BMO"), ("cutoff", "source_session_ceiling", "2024-06-18"),
    ("cutoff", "source_session_ceiling", "2024-06-17"),
    ("cutoff", "source_session_ceiling", "2021-12-31"),
    ("event_and_cutoffs", "event_date", "2024-06-18"),
])
def test_rehashed_direct_context_guards_all_targets_before_outcomes(source, monkeypatch, scope, name, value):
    binding = entry.context(source.repository, source.snapshot, **HISTORY_REQUEST)
    plan = binding["temporal_plan"]
    event = plan["events"][-1]  # A malformed later target must stop earlier safe-target reads too.
    assert event["event_date"] == "2024-06-17"
    target = event if scope != "cutoff" else event["forecasts"][-1]
    target[name] = source.forbidden[0]["event_id"] if value == "protected" else value
    for forecast in event["forecasts"]:
        if scope == "event_and_cutoffs":
            forecast[name] = value
        forecast["cutoff_id"] = content_hash({k: v for k, v in forecast.items() if k != "cutoff_id"})
    plan["plan_id"] = content_hash({k: v for k, v in plan.items() if k != "plan_id"})
    binding["context_id"] = content_hash({k: v for k, v in binding.items() if k != "context_id"})
    poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    code = "HOLDOUT_ACCESS_DENIED" if name == "event_id" else "CONTRACT_MISMATCH"
    with pytest.raises(DataError, match=code):
        stored.materialize(source.repository, source.snapshot, binding)
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_direct_engine_accepts_rehashed_calendar_metadata_but_public_read_rederives_context(source, monkeypatch):
    authentic = entry.context(source.repository, source.snapshot, **HISTORY_REQUEST)
    binding = json.loads(json.dumps(authentic))
    plan = binding["temporal_plan"]
    plan["calendar_input"]["calendar_version"] = "synthetic-rehashed-calendar"
    for event in plan["events"]:
        for forecast in event["forecasts"]:
            forecast["calendar_id"] = "sha256:" + "0" * 64
            forecast["cutoff_id"] = content_hash({k: v for k, v in forecast.items() if k != "cutoff_id"})
    plan["plan_id"] = content_hash({k: v for k, v in plan.items() if k != "plan_id"})
    binding["context_id"] = content_hash({k: v for k, v in binding.items() if k != "context_id"})
    assert binding["context_id"] != authentic["context_id"]
    poison_outcomes(monkeypatch, source.repository, source.snapshot,
            [e["event_date"].date().isoformat() for e in [*source.forbidden, source.events[-1]]])
    output = stored.materialize(source.repository, source.snapshot, binding)
    poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    monkeypatch.setattr(stored, "_prepare_history", _forbid)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    result = stored.read_stored(source.repository, output, binding)
    assert result["context"] == binding and result["feature_snapshot"] == to_document(output)
    assert len(result["rows"]) == 12
    latest = next(row for row in result["rows"]
                  if row["event_date"] == "2024-06-17" and row["horizon_sessions"] == 1)
    assert latest["n_prior"] == 4 and latest["mean_prior_abs_move"] == 5.5
    monkeypatch.setattr(stored, "_scan", _forbid)
    with pytest.raises(DataError, match="IDENTITY_CONFLICT: feature snapshot is not the committed context output"):
        _read(source, output)
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


def test_equal_feature_values_under_new_context_have_distinct_physical_objects(source):
    _, first = build_history(source)
    _, second = build_history(source, history_start="2022-02-01")
    left, right = _read(source, first)["rows"], _read(source, second, history_start="2022-02-01")["rows"]
    assert [[r[k] for k in FEATURES] for r in left] == [[r[k] for k in FEATURES] for r in right]
    assert first != second
    left_objects = {r.object_ref.object_id for r in source.repository.fragment_records(first, TABLE)}
    right_objects = {r.object_ref.object_id for r in source.repository.fragment_records(second, TABLE)}
    assert not left_objects & right_objects


def test_explicit_protected_population_fails_before_any_outcome_or_publication(source, monkeypatch):
    poison_outcomes(monkeypatch, source.repository, source.snapshot, [], all_outcomes=True)
    before = catalog_rows(source.conn), artifact_object_bytes(source.store)
    with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
        build_history(source, event_ids=[source.forbidden[0]["event_id"]])
    assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before


@pytest.mark.parametrize("name,value", [
    ("random_membership_version", "wrong"), ("rolling_membership_version", "wrong"),
    ("holdout_as_of_month", "2026-02"), ("computed_moves_version", "wrong"),
    ("purpose", "final"), ("schema_version", "wrong"), ("read_inputs", None),
    ("snapshot_id", "wrong"), ("event_id", "wrong"), ("ticker", "OTHER"),
    ("event_date", "2024-06-16"), ("population_use", "wrong"),
    ("history_start", "2022-02-01"), ("decision_session", "2024-06-01"),
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
    _, output = build_history(source)
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
    _, output = build_history(source)
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
    _, output = build_history(source)
    monkeypatch.setattr(stored, "CONTRACT", original)
    monkeypatch.setattr(stored, "_scan", _forbid)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _read(source, output)


def test_late_missing_and_skipped_labels_are_stored_as_omissions(tmp_path):
    source = _source(tmp_path, overrides={0: {"realized_move_pct": None, "available_as_of_date": "2026-01-01"},
        1: {"realized_move_pct": None, "skipped": True, "available_as_of_date": None},
        2: {"realized_move_pct": 0.}})
    try:
        _, output = build_history(source)
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
    _, output = build_history(source)
    scans = list(source.repository.scans)
    # Four event-year partitions (two passes), three market-year partitions (one pass).
    assert Counter(name for name, _ in scans) == {"earnings_events": 8, "daily_market": 3, "computed_moves": 4}
    outcome_queries = [q for name, q in scans if name == "computed_moves"]
    assert all(q.max_result_rows == 1 and q.max_batch_rows == 1 for q in outcome_queries)
    assert len({q.time_interval.start_inclusive for q in outcome_queries}) == 4
    assert len(_read(source, output)["rows"]) == 12
    source.repository.scans.clear()
    build_history(source, history_start="2022-02-01")
    assert Counter(name for name, _ in source.repository.scans)["computed_moves"] == 4  # No cross-build cache.


def test_candidate_equal_to_a_forecast_ceiling_is_absent_from_that_forecast_reads(source):
    _, output = build_history(source)
    rows = _read(source, output)["rows"]
    row = next(r for r in rows if r["event_date"] == "2024-06-17" and r["horizon_sessions"] == 3)
    assert row["source_session_ceiling"] == "2024-06-12"
    history = json.loads(row["history_json"])
    assert [r["event_id"] for r in history["read_inputs"]] == [e["event_id"] for e in source.events[:3]]
    assert history["omitted_history"] == []


def test_candidate_on_latest_ceiling_is_never_opened(tmp_path, monkeypatch):
    source = _source(tmp_path, recent_day="2024-06-14")
    try:
        poison_outcomes(monkeypatch, source.repository, source.snapshot, ["2024-06-14", "2024-06-17"])
        _, output = build_history(source)
        rows = _read(source, output)["rows"]
        row = next(r for r in rows if r["event_date"] == "2024-06-17" and r["horizon_sessions"] == 1)
        assert row["source_session_ceiling"] == "2024-06-14" and row["n_prior"] == 3
    finally:
        source.conn.close()


@pytest.mark.parametrize("invalid_index", [0, 2])
def test_lazy_cache_preserves_invalid_label_order_without_catalog_or_receipt_effect(tmp_path, monkeypatch, invalid_index):
    source = _source(tmp_path, overrides={invalid_index: {"realized_move_pct": None}})
    try:
        # Same-ticker IDs place the ISO date before the membership suffix; hash choice cannot reorder dates.
        ids = [e["event_id"] for e in source.events]
        assert ids == sorted(ids)
        poison_outcomes(monkeypatch, source.repository, source.snapshot,
                [e["event_date"].date().isoformat() for e in source.events[invalid_index + 1:] + source.forbidden])
        context = entry.context(source.repository, source.snapshot, **HISTORY_REQUEST)
        scope, receipt = stored._names(context)
        before, existing_objects = catalog_rows(source.conn), artifact_object_bytes(source.store)
        with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
            build_history(source)
        assert catalog_rows(source.conn) == before
        assert source.conn.execute("SELECT 1 FROM data_import_receipts WHERE receipt_id=?", (receipt,)).fetchone() is None
        assert source.conn.execute("SELECT 1 FROM data_snapshot_heads WHERE scope=?", (scope,)).fetchone() is None
        after = artifact_object_bytes(source.store)
        assert {key: after[key] for key in existing_objects} == existing_objects
        if invalid_index == 2:
            assert len(after) == len(existing_objects) + 2  # Earlier complete objects are unreferenced.
        else:
            assert after == existing_objects  # The first target failed before any feature publication.
    finally:
        source.conn.close()


def test_prepared_histories_preserve_existing_per_forecast_features_and_full_lineage(source):
    _, output = build_history(source)
    rows = _read(source, output)["rows"]
    for row in rows:
        expected = magnitude_features.load_magnitude_history(source.repository, source.snapshot,
            event_id=row["event_id"], as_of_month=HISTORY_REQUEST["as_of_month"], horizon=row["horizon_sessions"],
            history_start=HISTORY_REQUEST["history_start"])
        assert {key: row[key] for key in FEATURES} == expected["features"]
        assert row["cutoff_id"] == expected["forecast_cutoff"]["cutoff_id"]
        assert json.loads(row["history_json"]) == {
            key: value for key, value in expected["history_input"].items()
            if key not in ("features", "holdout_exclusions")}


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
    _, output = build_history(source)
    scan = stored._scan

    def guarded(repository, snapshot, context, columns, count):
        assert not set(columns) & set(FEATURES)
        return scan(repository, snapshot, context, columns, count)

    monkeypatch.setattr(stored, "_scan", guarded)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _read(source, output)


@pytest.mark.parametrize("change", ["missing", "extra", "row_count", "partition_key",
                                    "primary_key_min", "primary_key_max", "time_min", "time_max"])
def test_fragment_coverage_and_bounds_refuse_before_any_feature_object(source, monkeypatch, change):
    _, output = build_history(source)
    records = source.repository.fragment_records(output, TABLE)
    original = source.repository.fragment_records

    def malformed(snapshot, table):
        if table == TABLE:
            if change == "missing":
                return records[:-1]
            if change == "extra":
                return (*records, records[-1])
            changed = {"row_count": 2, "partition_key": "wrong",
                "primary_key_min": (*records[-1].primary_key_min[:-1], 0),
                "primary_key_max": (*records[-1].primary_key_max[:-1], 6),
                "time_min": "2024-06-16", "time_max": "2024-06-18"}
            return (*records[:-1], replace(records[-1], **{change: changed[change]}))
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


@pytest.mark.parametrize("change", ["history_content", "read_inputs_content", "history_id", "noncanonical",
                                    "history_type", "omitted_history_type", "read_inputs_type", "extra_key",
                                    "holdout_exclusions", "rehashed_roster"])
def test_lineage_digest_encoding_and_closed_types_refuse_before_features(source, monkeypatch, change):
    original = stored._event_rows

    def malformed(*args):
        rows = original(*args)
        if rows[0]["event_date"] == "2024-06-17":
            history = json.loads(rows[-1]["history_json"])
            if change.endswith("_content"):
                history[change.removesuffix("_content")][0]["object_id"] = "wrong"
            elif change == "history_id":
                history["history_id"] = "wrong"
            elif change.endswith("_type"):
                name = change.removesuffix("_type")
                history[name] = {"one": 1, "two": 2, "three": 3} if name == "history" else {}
            elif change == "extra_key":
                history["extra"] = "unused"
            elif change in ("holdout_exclusions", "rehashed_roster"):
                history["holdout_exclusions"] = [{"event_id": source.forbidden[0]["event_id"], "memberships": ["random"]}]
                if change == "rehashed_roster":
                    history["history_id"] = content_hash({k: v for k, v in history.items()
                        if k not in ("history_id", "history_status")})
            if change.endswith("_type") or change == "extra_key":
                history["history_id"] = content_hash({k: v for k, v in history.items()
                    if k not in ("history_id", "history_status", "holdout_exclusions")})
            rows[-1]["history_json"] = json.dumps(history, indent=2) if change == "noncanonical" else canonical_json(history)
        return rows

    monkeypatch.setattr(stored, "_event_rows", malformed)
    _, output = build_history(source)
    scan = stored._scan

    def guarded(repository, snapshot, context, columns, count):
        assert not set(columns) & set(FEATURES)
        return scan(repository, snapshot, context, columns, count)

    monkeypatch.setattr(stored, "_scan", guarded)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _read(source, output)


@pytest.mark.parametrize("name", stored.METADATA)
def test_feature_pass_metadata_drift_with_equal_counts_never_returns_partial_rows(source, monkeypatch, name):
    _, output = build_history(source)
    scan = stored._scan
    seen = []

    def changed(repository, snapshot, context, columns, count):
        rows = scan(repository, snapshot, context, columns, count)
        if set(columns) & set(FEATURES):
            rows[-1][name] = 2 if name == "horizon_sessions" else "changed"
        seen.append(len(rows))
        return rows

    monkeypatch.setattr(stored, "_scan", changed)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        _read(source, output)
    assert seen == [12, 12]


@pytest.mark.parametrize("phase", ["metadata", "features"])
def test_scan_row_coverage_guard_prevents_a_partial_stored_result(source, monkeypatch, phase):
    _, output = build_history(source)
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
