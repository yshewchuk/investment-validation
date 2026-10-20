"""Public synthetic native-history fixtures shared by magnitude research tests.

Keep fixture construction independent of test modules so import order and pytest
collection do not define the available helpers.
"""
import json
from contextlib import contextmanager
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data import objects
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT
from engine.v2.data.repository import Repository
from engine.v2.features import panel_math
from engine.v2.foundation import market_calendar
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.ops.experiments import ExperimentSpec
from engine.v2.research import event_history, magnitude_features
from engine.v2.research import magnitude_history_store as stored
from experiments import magnitude_history_store as history
from experiments import native_registration as native
from experiments import prediction_report as prediction
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)

MONTH, START, HISTORY, CUTOFF = "2026-01", "2023-01-01", "2022-01-01", "2025-02-01"
ROOT = Path(__file__).resolve().parents[3]
HISTORY_REQUEST = dict(as_of_month=MONTH, evaluation_years=["2024"],
                      training_start=START, history_start=HISTORY)


def outcome_move(event, value=2.0, **overrides):
    day = event["event_date"].date()
    row = dict(ticker=event["ticker"], event_date=day.isoformat(), realized_move_pct=value,
               available_as_of_date=(day + timedelta(days=2)).isoformat(), implied_move_pct=3.0,
               event_ordinal=0, admitted=False, mean_prior_or_implied=None, n_prior_implied=0,
               skipped=False, computed_at="2025-01-01T00:00:00Z",
               source_hash="synthetic-price-history", capture_id="synthetic-capture")
    return {**row, **overrides}


class TrackingRepository(Repository):
    def __init__(self, conn, store):
        super().__init__(conn, store)
        self.scans = []

    def scan(self, query, *, table_name):
        self.scans.append((table_name, query))
        yield from super().scan(query, table_name=table_name)

    def resolve_pinned(self, _scope):
        raise AssertionError("implicit latest is forbidden")


class PlanRepository(TrackingRepository):
    """Allow the registrar's one head pin while retaining real tracked scans."""
    def __init__(self, conn, store):
        super().__init__(conn, store)
        self.pins = []

    def resolve_pinned(self, scope):
        self.pins.append(scope)
        return Repository.resolve_pinned(self, scope)


def forbid_access(*args, **kwargs):
    """Fail if a pre-admission refusal touches metadata or an effect boundary."""
    raise AssertionError("forbidden access")


@contextmanager
def refuses(error_type, code):
    """Assert the exact documented nonretryable envelope, never exception text."""
    with pytest.raises(error_type) as caught:
        yield
    assert caught.value.code == code
    assert caught.value.problem.retryable is False


def history_event(day, *, protected=False, ticker="SYNTH", **overrides):
    event_id = next(f"{ticker}:{day}:{index}" for index in range(1000)
        if ExperimentHoldouts.random_membership(f"{ticker}:{day}:{index}") == protected)
    return {**dict(event_id=event_id, ticker=ticker, event_date=pd.Timestamp(day), year=int(day[:4]),
        session="AMC", session_src="orats", src_orats=True, src_oquants=False, src_nasdaq=False,
        src_yfinance=False, date_agree=True, date_conflict=False), **overrides}


def poison_outcomes(monkeypatch, repository, snapshot, protected_days, *, all_outcomes=False):
    records = repository.fragment_records(snapshot, "computed_moves")
    forbidden = {r.object_ref.object_id for r in records
        if all_outcomes or any(r.time_min <= day <= r.time_max for day in protected_days)}
    original = objects.verify_object_path
    opened = []

    def verify(store, ref):
        assert ref.object_id not in forbidden, "protected outcome bytes were requested"
        opened.append(ref.object_id)
        return original(store, ref)

    monkeypatch.setattr(objects, "verify_object_path", verify)
    return opened


def calendar_sessions(start="2024-06-03", end="2024-06-14"):
    projected = market_calendar.build_calendar_sessions((start,), event_through=end)
    days = tuple(d for d in projected.days if d <= end)
    return market_calendar.CalendarSessions(days, days[-1])


def commit_history_inputs(tmp_path, events, moves, *, days=None, mixed_days=(), include_calendar=True,
                          marker=None, contract=COMPUTED_MOVES_CONTRACT):
    tmp_path.mkdir(parents=True, exist_ok=True)
    conn, clock, store = catalog_and_store(tmp_path)
    contracts = {"earnings_events": contract_for("earnings_events"), "computed_moves": contract}
    inputs = {"earnings_events": events, "computed_moves": moves}
    if include_calendar:
        contracts["daily_market"] = contract_for("daily_market")
        inputs["daily_market"] = [dict(ticker=ticker, date=pd.Timestamp(day), year=int(day[:4]), spot=marker)
            for day in (calendar_sessions().days if days is None else days) for ticker in ("CAL-A", "CAL-B")]
    tables, hashes = {}, {}
    for name, rows in inputs.items():
        contract, groups = contracts[name], {}
        for row in rows:
            partition = str(row[contract.partition_columns[0]])
            piece = "metadata" if name != "computed_moves" else (
                "mixed" if row["event_date"] in mixed_days else row["event_date"])
            groups.setdefault((partition, piece), []).append(row)
        records = [publish_and_inspect(store, contract, contract_ref_for(contract),
            sorted(group, key=lambda r: tuple(r[k] for k in contract.primary_key)), partition)
            for (partition, _), group in groups.items()]
        tables[name] = sorted(records, key=lambda r: (r.partition_key, r.primary_key_min))
        partitions = {}
        for record in tables[name]:
            partitions.setdefault(record.partition_key, []).append(record)
        hashes[name] = {key: objects.partition_logical_hash(store,
            [r.object_ref for r in group], contract, contract_ref_for(contract), key)
            for key, group in partitions.items() if len(group) > 1}
    snapshot = commit_tables(conn, clock, tables, contracts, partition_logical_hashes=hashes, store=store)
    return conn, TrackingRepository(conn, store), snapshot


def prediction_spec(**changes):
    return ExperimentSpec(**{
        "experiment_id": "synthetic-prediction", "hypothesis": "A fixed prior baseline provides a calibration reference.",
        "primary_arm_id": "prior", "arms": ("prior",), "seed": 0,
        "folds": ("2023", "2024"), "economic_params": {}, "price_source": "computed_moves",
        "runner": prediction.RUNNER, **changes,
    })


def catalog_rows(conn):
    """Snapshot every catalog row so refusals cannot silently change other tables."""
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    return {table: sorted(tuple(row) for row in conn.execute(f'SELECT * FROM "{table}"'))
            for table in tables}


def build_history(source, **changes):
    result = history.build(source.repository, source.snapshot, **{**HISTORY_REQUEST, **changes})
    return result, source.repository.resolve(result["feature_snapshot"]["snapshot_id"])


def magnitude_dataset():
    dates = ("2022-06-02", "2023-01-03", "2023-06-02", "2024-06-12", "2024-06-17", "2025-01-07")
    values = {"SYNTH_A": (-2., -4., 6., 10., 8., 12.),
              "SYNTH_B": (2., 8., -12., 20., 4., 16.)}
    groups = {ticker: [history_event(day, ticker=ticker) for day in dates] for ticker in values}
    # Overlapping dates and interleaved insertion must never join histories across issuers.
    events = [event for pair in zip(*groups.values()) for event in reversed(pair)]
    moves = [outcome_move(event, value) for ticker, group in groups.items()
             for event, value in zip(group, values[ticker])]
    excluded = [history_event("2023-04-03", protected=True), history_event("2025-12-03")]
    return events + excluded, moves + [outcome_move(event, None) for event in excluded]


def magnitude_source(tmp_path, *, register=True, build=True, events=None, moves=None, folds=("2024", "2025"), mixed_days=()):
    if events is None:
        events, moves = magnitude_dataset()
    conn, original, snapshot = commit_history_inputs(tmp_path, events, moves, mixed_days=mixed_days,
        days=calendar_sessions("2022-05-25", "2025-12-05").days)
    spec = prediction_spec(experiment_id="synthetic-stored-magnitude", primary_arm_id="history",
        arms=("history",), folds=folds)
    source = SimpleNamespace(conn=conn, repository=PlanRepository(conn, original._store),
        store=original._store, snapshot=snapshot, events=events, spec=spec, root=tmp_path,
        spec_path=tmp_path / "spec.json")
    source.spec_path.write_text(json.dumps(asdict(spec)))
    if build:
        source.build, source.feature_snapshot = build_history(source, evaluation_years=list(folds))
    if register:
        source.registration = register_history(source)
        source.repository.scans.clear()
    return source


def history_binding(source, **changes):
    return dict(code_root=ROOT, as_of_month=MONTH, magnitude_training_start=START,
        magnitude_labels_before=CUTOFF, magnitude_feature_snapshot_id=source.feature_snapshot.snapshot_id,
        magnitude_history_start=HISTORY, **changes)


def register_history(source):
    return native.register_native(source.conn, source.store, source.repository, source.spec, **history_binding(source))


def forbid_history_rebuild(monkeypatch):
    for owner, name in ((history, "build"), (stored, "materialize"), (stored, "_prepare_history"),
        (stored, "_event_rows"), (event_history, "load_event_history"),
        (event_history, "_assemble_history"), (magnitude_features, "load_magnitude_history"),
        (panel_math, "history_features")):
        monkeypatch.setattr(owner, name, forbid_access)


def trial_cases(case):
    return [(outer["outer"]["year"], inner["year"], candidate["candidate_id"])
            for outer in case.plan["folds"] for inner in outer["inner"] for candidate in case.plan["candidates"]]


def trial_forecasts(case, args, value=7):
    outer, year, _ = args
    fold = next(inner for item in case.plan["folds"] if item["outer"]["year"] == outer
                for inner in item["inner"] if inner["year"] == year)
    return [{"event_id": event, "horizon_sessions": horizon, "predicted_absolute_move_pct": value}
            for event in fold["evaluation_event_ids"] for horizon in (5, 3, 1)]


def trial_slots(case):
    return [tuple(row) for row in case.source.conn.execute(
        "SELECT slot, ref_json FROM magnitude_search_records ORDER BY slot")]
