"""Synthetic committed IV targets: protected bytes, clocks and independent identities."""
from contextlib import contextmanager
from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data import objects
from engine.v2.data.errors import DataError, fail
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT as CONTRACT
from engine.v2.data.manifests import table_contract_hash
from engine.v2.data.repository import Repository
from engine.v2.foundation import content_hash, to_document
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.research import iv_crush_inputs as reader
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.v2.research.magnitude_trial_support import TrackingRepository, history_event


class BoundedTrackingRepository(TrackingRepository):
    def scan(self, query, *, table_name, max_object_bytes=None):
        self.scans.append((table_name, query))
        yield from Repository.scan(self, query, table_name=table_name, max_object_bytes=max_object_bytes)


MONTH, CUTOFF, TABLE = "2025-01", "2024-07-01", "iv_crush_targets"


def target(event, value=-40., **changes):
    day = event["event_date"].date()
    return {**dict(ticker=event["ticker"], event_date=day.isoformat(), session=event["session"],
        pre_date=day.isoformat(), post_date=(day + timedelta(days=1)).isoformat(),
        pre_iv30=50., post_iv30=30., crush_pct_iv30=value,
        available_as_of_date=(day + timedelta(days=2)).isoformat(), skipped=False, skip_reason=None,
        source_id="synthetic-caller", input_hash="sha256:" + "a" * 64), **changes}


@contextmanager
def committed(tmp_path, events, targets, *, mixed_days=(), contract=CONTRACT):
    conn, clock, store = catalog_and_store(tmp_path)
    contracts = {"earnings_events": contract_for("earnings_events"), TABLE: contract}
    tables, hashes = {}, {}
    for name, rows in (("earnings_events", events), (TABLE, targets)):
        definition, groups = contracts[name], {}
        for row in rows:
            partition = str(row[definition.partition_columns[0]])
            piece = "metadata" if name != TABLE else (
                "mixed" if row["event_date"] in mixed_days else row["event_date"])
            groups.setdefault((partition, piece), []).append(row)
        tables[name] = sorted([publish_and_inspect(store, definition, contract_ref_for(definition),
            sorted(group, key=lambda r: tuple(r[k] for k in definition.primary_key)), partition)
            for (partition, _), group in groups.items()], key=lambda r: (r.partition_key, r.primary_key_min))
        hashes[name] = {}
        for partition in {r.partition_key for r in tables[name]}:
            group = [r.object_ref for r in tables[name] if r.partition_key == partition]
            if len(group) > 1:
                hashes[name][partition] = objects.partition_logical_hash(
                    store, group, definition, contract_ref_for(definition), partition)
    try:
        snapshot = commit_tables(conn, clock, tables, contracts, partition_logical_hashes=hashes, store=store)
        yield conn, BoundedTrackingRepository(conn, store), snapshot
    finally:
        conn.close()


def poison(monkeypatch, repository, snapshot, days=None):
    forbidden = {r.object_ref.object_id for r in repository.fragment_records(snapshot, TABLE)
                 if days is None or any(r.time_min <= day <= r.time_max for day in days)}
    original = objects.verify_object_path
    def verify(store, ref):
        assert ref.object_id not in forbidden, "protected IV target bytes opened"
        return original(store, ref)
    monkeypatch.setattr(objects, "verify_object_path", verify)
    original_bounded = objects.read_verified_bytes
    def bounded(store, ref, *, max_bytes):
        assert ref.object_id not in forbidden, "protected IV target bytes opened"
        return original_bounded(store, ref, max_bytes=max_bytes)
    monkeypatch.setattr(objects, "read_verified_bytes", bounded)


def load(repository, snapshot, **changes):
    return reader.load_iv_crush_targets(repository, snapshot,
        **{**dict(as_of_month=MONTH, labels_before=CUTOFF), **changes})


def exact_problem(caught, code, message, category=None, details=None):
    category = category or ("internal" if code == "EXPERIMENT_VARIANT_FAILED" else "validation")
    problem = caught.value.problem
    assert (problem.code, problem.message, problem.category, problem.retryable,
            problem.stage, problem.details) == (code, message, category, False, None, details or {})
    assert str(caught.value) == f"{code}: {message}"


def test_complete_signed_rows_identity_purity_and_protected_bytes(tmp_path, monkeypatch):
    eligible = [history_event(day) for day in ("2024-01-04", "2024-02-05", "2024-03-05")]
    random, rolling = history_event("2024-04-05", protected=True), history_event("2024-12-03")
    alias, future, same_day = history_event("2024-05-06"), history_event("2024-07-02"), history_event(CUTOFF)
    excluded = [random, rolling, alias, future, same_day]
    events = [*eligible, *excluded, {**alias, "event_id": "ambiguous-sibling", "session": "BMO"}]
    targets = [target(e, v) for e, v in zip(eligible, (-110., 0., 75.), strict=True)]
    targets += [target(e, None) for e in excluded]
    with committed(tmp_path, events, targets) as (conn, repository, snapshot):
        poison(monkeypatch, repository, snapshot, [e["event_date"].date().isoformat() for e in excluded])
        before = conn.total_changes
        result = load(repository, snapshot)
        expected = [(e["event_id"], e["ticker"], e["event_date"], v,
                     (e["event_date"].date() + timedelta(days=2)).isoformat())
                    for e, v in zip(eligible, (-110., 0., 75.), strict=True)]
        assert list(result[["event_id", "ticker", "event_date", "crush_pct_iv30",
                            "target_available_on"]].itertuples(index=False, name=None)) == expected
        records = repository.fragment_records(snapshot, TABLE)
        consumed = [{"event_id": e["event_id"], "fragment_id": r.fragment_id,
            "object_id": r.object_ref.object_id, "content_hash": r.object_ref.content_hash,
            "source_id": "synthetic-caller", "input_hash": "sha256:" + "a" * 64}
            for e in eligible for r in records if r.time_min == e["event_date"].date().isoformat()]
        identity = {"schema_version": "research_iv_crush_targets.v1", "snapshot_id": snapshot.snapshot_id,
            "holdout_as_of_month": MONTH, "random_membership_version": ExperimentHoldouts.random_version,
            "rolling_membership_version": ExperimentHoldouts.rolling_version,
            "population_use": "post-release selection", "purpose": "selection",
            "target_version": to_document(snapshot.table_versions[TABLE]),
            "availability_basis": "modeled_post_close_plus_one_calendar_day",
            "source_publication": "unverified", "source_finality": "unverified",
            "labels_before": CUTOFF, "consumed_inputs": consumed,
            "excluded_future_events": sorted([future["event_id"], same_day["event_id"]])}
        assert {k: v for k, v in result.attrs.items() if k != "holdout_exclusions"} == {
            **identity, "inputs_id": content_hash(identity)}
        assert result.target_object_id.tolist() == [r["object_id"] for r in consumed]
        assert result.attrs["holdout_exclusions"] == sorted([
            {"event_id": random["event_id"], "memberships": ["random"]},
            {"event_id": rolling["event_id"], "memberships": ["rolling"]},
            {"event_id": alias["event_id"], "memberships": ["ambiguous"]},
            {"event_id": "ambiguous-sibling", "memberships": ["ambiguous"]},
        ], key=lambda entry: entry["event_id"])
        again = load(repository, snapshot)
        pd.testing.assert_frame_equal(result, again)
        assert result.attrs == again.attrs
        assert load(repository, snapshot, labels_before="2024-06-30").attrs["inputs_id"] != result.attrs["inputs_id"]
        assert conn.total_changes == before
        queries = [q for name, q in repository.scans if name == TABLE]
        assert len(queries) == 9 and all(q.max_result_rows == q.max_batch_rows == 1 for q in queries)
        assert all(q.columns == tuple(c.name for c in CONTRACT.columns) for q in queries)


def test_preflight_binds_exact_participants_and_objects_without_opening_targets(tmp_path, monkeypatch):
    events = [history_event(day) for day in ("2024-01-04", "2024-03-05")]
    with committed(tmp_path, events, [target(e) for e in events]) as (_, repository, snapshot):
        poison(monkeypatch, repository, snapshot)
        ids = [e["event_id"] for e in events]
        result = reader.preflight_iv_crush_targets(repository, snapshot, as_of_month=MONTH, event_ids=ids[::-1])
        records = repository.fragment_records(snapshot, TABLE)
        expected = {"schema_version": "research_iv_crush_preflight.v1", "snapshot_id": snapshot.snapshot_id,
            "holdout_as_of_month": MONTH, "random_membership_version": ExperimentHoldouts.random_version,
            "rolling_membership_version": ExperimentHoldouts.rolling_version,
            "population_use": "post-release selection", "purpose": "selection",
            "target_version": to_document(snapshot.table_versions[TABLE]),
            "availability_basis": "modeled_post_close_plus_one_calendar_day",
            "source_publication": "unverified", "source_finality": "unverified", "event_ids": ids,
            "planned_inputs": [{"event_id": e["event_id"], "fragment_id": r.fragment_id,
                "object_id": r.object_ref.object_id, "content_hash": r.object_ref.content_hash}
                for e, r in zip(events, records, strict=True)]}
        assert result == {**expected, "preflight_id": content_hash(expected)}
        assert all(name == "earnings_events" for name, _ in repository.scans)


@pytest.mark.parametrize("kind", ["random", "rolling", "alias", "unknown", "missing", "mixed", "old_contract", "v1_contract"])
def test_all_admission_precedes_first_target_open(tmp_path, monkeypatch, kind):
    first = history_event("2024-01-04")
    second = history_event("2024-12-03" if kind == "rolling" else "2024-03-05", protected=kind == "random")
    events, targets, contract = [first, second], [target(first), target(second)], CONTRACT
    if kind == "alias":
        events.append({**second, "event_id": "alias", "session": "BMO"})
    if kind == "missing":
        targets.pop()
    if kind == "old_contract":
        # Alter semantic metadata while retaining the current name; exact equality is required.
        policies = {"available_as_of_date": "null exactly when skipped; unavailable labels cannot enter training",
            "skip_reason": "null exactly when not skipped",
            "source_id": "caller-declared identity, not verified by the pure builder",
            "input_hash": "canonical hash of the supplied typed event/daily projections"}
        columns = tuple(replace(c, null_policy=policies[c.name]) if c.name in policies else c
                        for c in contract.columns)
        contract = replace(contract, columns=columns,
            provenance_semantics="input_hash binds supplied projections; source_id is unverified caller metadata",
            coverage_semantics="one row per supplied event, skipped rows included; no completeness inference")
        contract = replace(contract, definition_hash=table_contract_hash(contract))
    if kind == "v1_contract":
        contract = replace(contract, contract_id="iv_crush_targets.v1", semantic_version="1.0.0")
        contract = replace(contract, definition_hash=table_contract_hash(contract))
    mixed = ("2024-01-04", "2024-03-05") if kind == "mixed" else ()
    with committed(tmp_path, events, targets, mixed_days=mixed, contract=contract) as (_, repository, snapshot):
        poison(monkeypatch, repository, snapshot)
        ids = [first["event_id"], "unknown" if kind == "unknown" else second["event_id"]]
        for action in (lambda: load(repository, snapshot, event_ids=ids),
                       lambda: reader.preflight_iv_crush_targets(repository, snapshot,
                           as_of_month=MONTH, event_ids=ids)):
            with pytest.raises(DataError) as caught:
                action()
            if kind in ("old_contract", "v1_contract"):
                exact_problem(caught, "CONTRACT_MISMATCH", "IV-crush labels require the exact native contract")
            elif kind in ("missing", "mixed"):
                exact_problem(caught, "HOLDOUT_ACCESS_DENIED", "research requires event-isolated outcome fragments")
            elif kind == "unknown":
                exact_problem(caught, "HOLDOUT_ACCESS_DENIED", "requested event membership cannot be resolved")
            else:
                exact_problem(caught, "HOLDOUT_ACCESS_DENIED", "requested events are excluded from experiment reads",
                    details={"snapshot_id": snapshot.snapshot_id, "holdout_as_of_month": MONTH,
                        "random_membership_version": "canonical-event-sha256.v1",
                        "rolling_membership_version": "calendar-months.v1", "purpose": "selection",
                        "holdout_exclusions": [{"event_id": second["event_id"],
                            "memberships": [{"random": "random", "rolling": "rolling", "alias": "ambiguous"}[kind]]}]})
        assert all(name == "earnings_events" for name, _ in repository.scans)


@pytest.mark.parametrize("changes", [
    {"skipped": True}, {"skipped": None}, {"skipped": 0}, {"skip_reason": "invalid_pre_iv30"},
    {"session": "BMO"}, {"event_date": "2024-05-07"}, {"ticker": "OTHER"},
    {"available_as_of_date": None}, {"available_as_of_date": "bad"},
    {"available_as_of_date": "2024-05-06"}, {"available_as_of_date": CUTOFF},
    {"available_as_of_date": "2024-07-02"}, {"post_date": None}, {"post_date": "2024-05-08"},
])
def test_exact_identity_clock_and_skip_refusal_precedes_numeric_access(changes):
    event = history_event("2024-05-06")
    row = target(event, **changes)
    del row["crush_pct_iv30"]
    with pytest.raises(DataError) as caught:
        reader._label_before([row], SimpleNamespace(**event), date(2024, 7, 1))
    exact_problem(caught, "EXPERIMENT_VARIANT_FAILED", "IV-crush label identity or availability is invalid")


@pytest.mark.parametrize("changes", [
    {"crush_pct_iv30": value} for value in (None, True, "30", float("nan"), float("inf"), -float("inf"), 10**400)
] + [{"source_id": value} for value in (None, "", " padded", "padded ", "x" * 257)]
  + [{"input_hash": value} for value in (None, "sha256:" + "A" * 64, "sha256:" + "a" * 63, "foobar:" + "a" * 64)])
def test_exact_value_and_declared_provenance_refusal(changes):
    event = history_event("2024-05-06")
    with pytest.raises(DataError) as caught:
        reader._label_before([target(event, **changes)], SimpleNamespace(**event), date(2024, 7, 1))
    exact_problem(caught, "EXPERIMENT_VARIANT_FAILED", "IV-crush label value or provenance is invalid")


@pytest.mark.parametrize("count", [0, 2])
def test_decoded_cardinality_is_exact(count):
    event = history_event("2024-05-06")
    with pytest.raises(DataError) as caught:
        reader._label_before([target(event)] * count, SimpleNamespace(**event), date(2024, 7, 1))
    exact_problem(caught, "EXPERIMENT_VARIANT_FAILED", "IV-crush label identity or availability is invalid")


@pytest.mark.parametrize("cutoff", [None, True, "20240701", "2024-07-01T00:00:00Z"])
def test_invalid_cutoff_refuses_before_repository_access(cutoff):
    with pytest.raises(DataError) as caught:
        load(None, None, labels_before=cutoff)
    exact_problem(caught, "CONTRACT_MISMATCH", "an explicit IV-crush label cutoff is required")


def test_preflight_requires_explicit_participants_before_access():
    with pytest.raises(DataError) as caught:
        reader.preflight_iv_crush_targets(None, None, as_of_month=MONTH, event_ids=None)
    exact_problem(caught, "CONTRACT_MISMATCH", "IV-crush preflight requires explicit participants")


@pytest.mark.parametrize("day", [CUTOFF, "2024-07-02"])
def test_explicit_future_events_and_empty_bulk_refuse_before_target_bytes(tmp_path, monkeypatch, day):
    event = history_event(day)
    with committed(tmp_path, [event], [target(event)]) as (_, repository, snapshot):
        poison(monkeypatch, repository, snapshot)
        with pytest.raises(DataError) as caught:
            load(repository, snapshot, event_ids=[event["event_id"]])
        exact_problem(caught, "CONTRACT_MISMATCH", "requested IV-crush events must precede label cutoff")
        with pytest.raises(DataError) as caught:
            load(repository, snapshot)
        exact_problem(caught, "POPULATION_COLLAPSED", "no IV-crush events precede label cutoff")
        assert all(name == "earnings_events" for name, _ in repository.scans)


@pytest.mark.parametrize("changes", [{"as_of_month": None}, {"as_of_month": "2025-13"},
    {"purpose": "final"}, {"purpose": "holdout"}, {"event_ids": "not-a-list"}])
def test_invalid_holdout_context_refuses_before_any_scan(tmp_path, monkeypatch, changes):
    event = history_event("2024-05-06")
    with committed(tmp_path, [event], [target(event)]) as (_, repository, snapshot):
        poison(monkeypatch, repository, snapshot)
        with pytest.raises(DataError) as caught:
            load(repository, snapshot, **changes)
        exact_problem(caught, "HOLDOUT_ACCESS_DENIED", "explicit valid holdout context is required")
        assert repository.scans == []


def test_label_error_returns_no_partial_result_after_an_earlier_valid_read(tmp_path):
    events = [history_event(day) for day in ("2024-01-04", "2024-03-05")]
    with committed(tmp_path, events, [target(events[0]), target(events[1], skipped=True)]) as (_, repository, snapshot):
        with pytest.raises(DataError) as caught:
            load(repository, snapshot)
        exact_problem(caught, "EXPERIMENT_VARIANT_FAILED", "IV-crush label identity or availability is invalid")
        assert len([q for name, q in repository.scans if name == TABLE]) == 2


def test_last_day_clock_strictness_and_native_refusal_propagation(tmp_path, monkeypatch):
    event = history_event("2024-06-28")
    with committed(tmp_path, [event], [target(event)]) as (_, repository, snapshot):
        result = load(repository, snapshot)
        assert result.crush_pct_iv30.tolist() == [-40.]
        assert result.target_available_on.tolist() == ["2024-06-30"]
        with pytest.raises(DataError) as caught:
            load(repository, snapshot, labels_before="2024-06-30")
        exact_problem(caught, "EXPERIMENT_VARIANT_FAILED", "IV-crush label identity or availability is invalid")
        original = repository.scan
        refusal = fail("OBJECT_CORRUPT", "synthetic object corruption")
        def scan(query, *, table_name, **kwargs):
            if table_name == TABLE:
                raise refusal
            yield from original(query, table_name=table_name, **kwargs)
        monkeypatch.setattr(repository, "scan", scan)
        with pytest.raises(DataError) as caught:
            load(repository, snapshot)
        assert caught.value is refusal


@pytest.mark.parametrize("session,pre,post,available,valid", [
    ("AMC", "2024-05-06", "2024-05-06", "2024-05-07", False),
    ("AMC", "2024-05-03", "2024-05-06", "2024-05-07", False),
    ("AMC", "2024-05-07", "2024-05-08", "2024-05-09", False),
    ("AMC", "2024-05-01", "2024-05-07", "2024-05-08", False),
    ("AMC", "2024-05-02", "2024-05-07", "2024-05-08", True),
    ("AMC", "2024-05-03", "2024-05-07", "2024-05-08", True),
    ("BMO", "2024-05-06", "2024-05-07", "2024-05-08", False),
    ("BMO", "2024-05-02", "2024-05-05", "2024-05-06", False),
    ("BMO", "2024-05-03", "2024-05-06", "2024-05-07", True),
    ("BMO", "2024-05-03", "2024-05-07", "2024-05-08", True),
    ("AMC", "bad", "2024-05-07", "2024-05-08", False),
    ("AMC", None, "2024-05-07", "2024-05-08", False),
    ("AMC", "2024-05-07", "2024-05-06", "2024-05-07", False),
    ("AMC", "2024-05-06", "bad", "2024-05-08", False),
    ("AMC", "2024-05-06", None, "2024-05-08", False),
])
def test_session_anchor_admission_without_invented_calendar_or_source_claim(tmp_path, session, pre, post, available, valid):
    event = history_event("2024-05-06", session=session)
    row = target(event, -110., pre_date=pre, post_date=post, available_as_of_date=available)
    with committed(tmp_path, [event], [row]) as (_, repository, snapshot):
        if valid:
            result = load(repository, snapshot)
            assert result[["crush_pct_iv30", "target_available_on"]].values.tolist() == [[-110., available]]
        else:
            with pytest.raises(DataError) as caught:
                load(repository, snapshot)
            exact_problem(caught, "EXPERIMENT_VARIANT_FAILED", "IV-crush label identity or availability is invalid")


@pytest.mark.parametrize("extra", [0, 1])
def test_all_declared_byte_bounds_precede_first_target_open(tmp_path, monkeypatch, extra):
    events = [history_event(day) for day in ("2024-01-04", "2024-03-05")]
    assert reader.MAX_TARGET_OBJECT_BYTES == 1048576
    with committed(tmp_path, events, [target(e) for e in events]) as (_, repository, snapshot):
        records = list(repository.fragment_records(snapshot, TABLE))
        records[-1] = replace(records[-1], object_ref=replace(records[-1].object_ref, byte_size=1048576 + extra))
        original = repository.fragment_records
        monkeypatch.setattr(repository, "fragment_records", lambda snap, name:
                            records if name == TABLE else original(snap, name))
        poison(monkeypatch, repository, snapshot)
        def preflight():
            return reader.preflight_iv_crush_targets(repository, snapshot,
                as_of_month=MONTH, event_ids=[e["event_id"] for e in events])
        if extra == 0:
            assert preflight()["event_ids"] == [e["event_id"] for e in events]
        else:
            for action in (preflight, lambda: load(repository, snapshot)):
                with pytest.raises(DataError) as caught:
                    action()
                exact_problem(caught, "RESULT_LIMIT_EXCEEDED",
                    "IV-crush target object exceeds the 1 MiB byte bound", category="resource")
        assert all(name == "earnings_events" for name, _ in repository.scans)


def test_iv_scan_passes_expansion_bound_before_decoding_large_scalar(tmp_path, monkeypatch):
    event = history_event("2024-05-06")
    with committed(tmp_path, [event], [target(event, source_id="x" * 1100000)]) as (_, repository, snapshot):
        record, = repository.fragment_records(snapshot, TABLE)
        assert record.object_ref.byte_size < 1048576
        original = repository._iter_batches
        def batches(parquet, present, cap):
            assert "crush_pct_iv30" not in present, "oversized IV scalar decoded"
            yield from original(parquet, present, cap)
        monkeypatch.setattr(repository, "_iter_batches", batches)
        with pytest.raises(DataError) as caught:
            load(repository, snapshot)
        exact_problem(caught, "RESULT_LIMIT_EXCEEDED",
            "parquet exceeds max_object_bytes when uncompressed", category="resource")


def test_current_v2_identity_and_valid_text_boundaries(tmp_path):
    assert (CONTRACT.contract_id, CONTRACT.semantic_version, CONTRACT.definition_hash) == (
        "iv_crush_targets.v2", "2.0.0", "sha256:f481bb155dac4d735483f24e5516ec5866a358c038de4f24ab36f25eddb58a9c")
    event = history_event("2024-05-06")
    row = target(event, -40, source_id="x" * 256, input_hash="sha256:" + "0123456789abcdef" * 4)
    with committed(tmp_path, [event], [row]) as (_, repository, snapshot):
        result = load(repository, snapshot)
        assert result.crush_pct_iv30.tolist() == [-40.]
        assert result.attrs["target_version"]["table_contract_ref"] == {
            "schema_version": "table_contract_ref.v1.0", "contract_id": "iv_crush_targets.v2",
            "definition_hash": "sha256:f481bb155dac4d735483f24e5516ec5866a358c038de4f24ab36f25eddb58a9c"}
        assert result.attrs["consumed_inputs"][0]["source_id"] == "x" * 256
        assert result.attrs["consumed_inputs"][0]["input_hash"] == "sha256:" + "0123456789abcdef" * 4
