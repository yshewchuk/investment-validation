"""Public synthetic partial-block construction; no private data or observed-PIT claim."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.paths import ROOT
from engine.v2.data import objects
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import content_hash, to_document
from engine.v2.ops.fingerprints import environment_identity, source_closure
from engine.v2.research import eligibility_policy, magnitude_panel_daily as daily, magnitude_panel_plan, magnitude_panel_table
from tests.data_scan_support import (
    catalog_and_store, commit_tables, contract_for, contract_ref_for, publish_and_inspect,
)
from tests.v2.research.magnitude_trial_support import (
    TrackingRepository, build_history, calendar_sessions, history_event, outcome_move, poison_outcomes,
)

DAILY = ["implied_move", "iv10", "iv30", "exern_iv10", "exern_iv30", "iee", "skew",
         "contango", "fwd90_30", "fexern90_30", "rvol30", "spot", "mcap_log"]
SPY = ["spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol5", "spy_vol20",
       "spy_vol60", "spy_vol252", "spy_vol20_rel252"]
START = "2022-05-25"


class PinnedRepository(TrackingRepository):
    def __init__(self, conn, store):
        Repository.__init__(self, conn, store, pin_snapshots=True)
        self.scans = []


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    root = tmp_path_factory.mktemp("panel-daily")
    conn, clock, store = catalog_and_store(root)
    events = [history_event(day) for day in ("2022-06-02", "2023-01-03", "2024-06-12", "2024-08-16")]
    held = history_event("2024-05-15", protected=True)
    all_events = [*events, held]
    days = calendar_sessions(START, "2024-08-19").days
    market = []
    for ticker in ("SYNTH", "SPY"):
        for index, day in enumerate(days):
            row = {name: float(i + 1) for i, name in enumerate(DAILY)}
            row.update(ticker=ticker, date=pd.Timestamp(day), year=int(day[:4]), src_iv=f"synthetic-{index}-" + "x" * 256,
                       spot=100.0 + index, mcap_asof=pd.Timestamp(day), mcap_log=20.0)
            if day == "2024-08-13":
                row.update(implied_move=0.0, mcap_asof=pd.Timestamp("2030-01-01"), mcap_log=999999.0)
            if day == "2024-08-14":
                row["src_iv"] = None
            market.append(row)
    inputs = {"earnings_events": all_events, "daily_market": market,
              "computed_moves": [outcome_move(event, 2.0) for event in all_events]}
    contracts = {"earnings_events": contract_for("earnings_events"),
                 "daily_market": contract_for("daily_market"), "computed_moves": COMPUTED_MOVES_CONTRACT}
    tables, hashes = {}, {}
    for table, rows in inputs.items():
        contract, groups = contracts[table], {}
        for row in rows:
            partition = str(row[contract.partition_columns[0]])
            piece = row["event_date"] if table == "computed_moves" else "all"
            groups.setdefault((partition, piece), []).append(row)
        records = [publish_and_inspect(store, contract, contract_ref_for(contract),
            sorted(group, key=lambda row: tuple(row[key] for key in contract.primary_key)), partition)
            for (partition, _), group in groups.items()]
        tables[table] = sorted(records, key=lambda record: (record.partition_key, record.primary_key_min))
        partitions = {}
        for record in tables[table]:
            partitions.setdefault(record.partition_key, []).append(record)
        hashes[table] = {key: objects.partition_logical_hash(store, [r.object_ref for r in group],
            contract, contract_ref_for(contract), key) for key, group in partitions.items() if len(group) > 1}
    snapshot = commit_tables(conn, clock, tables, contracts, partition_logical_hashes=hashes, store=store)
    source = SimpleNamespace(conn=conn, repository=PinnedRepository(conn, store), snapshot=snapshot)
    build, stored = build_history(source)
    parent = magnitude_panel_plan.build_context(source.repository, snapshot, stored,
        build["context"], producer=build["context"]["producer"])
    policy = eligibility_policy.eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30)
    groups = []
    for event in parent["history_context"]["temporal_plan"]["events"]:
        groups.append({"event_id": event["event_id"], "ticker": event["ticker"],
            "event_date": event["event_date"], "label_start": event["event_date"],
            "label_end": (pd.Timestamp(event["event_date"]) + timedelta(days=2)).date().isoformat(),
            "forecasts": [{"horizon_sessions": c["horizon_sessions"],
                "decision_date": c["source_session_ceiling"], "source_session_ceiling": c["source_session_ceiling"]}
                for c in event["forecasts"]]})
    metadata = {"schema_version": "event_group_bounds.v1", "source_snapshot": to_document(snapshot),
        "holdout_as_of_month": "2026-01", "policy_id": policy["policy_id"],
        "horizons": [1, 3, 5], "groups": groups}
    metadata["metadata_id"] = content_hash(metadata)
    producer = {"source_closure": source_closure(ROOT, ["engine/v2/research/magnitude_panel_daily.py"]),
                "environment": environment_identity(1)}
    yield SimpleNamespace(**vars(source), stored=stored, parent=parent, policy=policy,
                          metadata=metadata, events=events, days=days, producer=producer)
    conn.close()


def build(native, **changes):
    return daily.build_daily_blocks(native.repository, native.snapshot, native.stored, native.parent,
        **{"producer": native.producer, "policy": native.policy, "eligibility_metadata": native.metadata, "source_start": START,
           "max_source_rows": 1000, "max_events": 10, "max_object_bytes": 2**20, **changes})


def test_vertical_native_sources_to_complete_partial_blocks(native, monkeypatch):
    poison_outcomes(monkeypatch, native.repository, native.snapshot, (), all_outcomes=True)
    before = native.conn.total_changes
    saved = deepcopy(native.parent)
    native.repository.scans.clear()
    result = build(native)
    assert native.parent == saved and saved["admission"]["raw_payload_access"] == "not_authorized"
    assert native.conn.total_changes == before
    assert result["context"]["status"] == "reconstructed_partial_blocks_not_fit_admissible"
    assert result["context"]["label_bounds"] == "unverified_producer_declaration"
    assert result["context"]["daily_columns"] == DAILY and result["context"]["spy_columns"] == SPY
    assert len(result["rows"]) == 6  # All horizons of the June 2024 event are embargoed.
    assert {r["event_id"] for r in result["rows"]} == {native.events[1]["event_id"], native.events[3]["event_id"]}
    assert all(r["status"] == "reconstructed_partial_blocks_not_fit_admissible" for r in result["rows"])
    assert all(set(r["daily_market"]["values"]) == set(DAILY) and set(r["spy"]["values"]) == set(SPY)
               for r in result["rows"])
    reads = [query for table, query in native.repository.scans if table == "daily_market" and "spot" in query.columns]
    counts = Counter((q.key_filter[0].values[0], q.key_filter[1].values[0]) for q in reads)
    assert counts == Counter({(ticker, year): 1 for ticker in ("SPY", "SYNTH") for year in (2022, 2023, 2024)})
    for row in result["rows"]:
        assert row["row_id"] == content_hash({k: v for k, v in row.items() if k != "row_id"})
        assert row["daily_market"]["anchor"]["source_date"] <= row["source_session_ceiling"]
        assert row["daily_market"]["anchor"]["available_as_of"] is None
        assert max(row["spy"]["source_dates"]) <= row["source_session_ceiling"]
        assert row["spy"]["knowledge_mode"] == "reconstructed"


def test_literal_daily_values_quote_and_late_cap(native):
    row = next(r for r in build(native)["rows"] if r["event_id"] == native.events[3]["event_id"]
               and r["horizon_sessions"] == 3)
    block = row["daily_market"]
    assert row["source_session_ceiling"] == "2024-08-13"
    expected = {name: float(i + 1) for i, name in enumerate(DAILY)}
    expected.update(implied_move=None, mcap_log=None, spot=100.0 + native.days.index("2024-08-13"))
    assert block["values"] == expected
    assert block["missing"] == {"implied_move": "no_quote", "mcap_log": "late_capture"}
    assert block["quote_status"] == "zero" and block["anchor"]["mcap_asof"] == "2030-01-01"


def test_spy_independent_return_oracles(native):
    result = build(native)
    for row in result["rows"]:
        idx = native.days.index(row["source_session_ceiling"])
        values = row["spy"]["values"]
        for window in (21, 63, 252):
            expected = ((100 + idx) / (100 + idx - window) - 1) * 100 if idx >= window else None
            assert values[f"spy_ret{window}"] == pytest.approx(expected) if expected is not None else values[f"spy_ret{window}"] is None


@pytest.mark.parametrize("field,value", [("max_events", 0), ("max_source_rows", True), ("max_events", 1_000_001)])
def test_invalid_limits_refuse_before_repository(native, monkeypatch, field, value):
    monkeypatch.setattr(native.repository, "resolve", lambda *a: pytest.fail("access before limit admission"))
    with pytest.raises(DataError, match="bounded panel limits") as caught:
        build(native, **{field: value})
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_changed_forecast_clock_refuses_before_value_reads(native, monkeypatch):
    metadata = deepcopy(native.metadata)
    metadata["groups"][0]["forecasts"][0]["source_session_ceiling"] = "2022-01-01"
    metadata["metadata_id"] = content_hash({k: v for k, v in metadata.items() if k != "metadata_id"})
    original = native.repository.scan
    def no_values(query, *, table_name, **kwargs):
        assert "spot" not in query.columns
        yield from original(query, table_name=table_name, **kwargs)
    monkeypatch.setattr(native.repository, "scan", no_values)
    with pytest.raises(DataError, match="clocks differ") as caught:
        build(native, eligibility_metadata=metadata)
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_source_bound_refuses_without_partial_return(native):
    with pytest.raises(DataError) as caught:
        build(native, max_source_rows=10)
    assert caught.value.code == "RESULT_LIMIT_EXCEEDED"


def test_exact_replay_rereads_and_has_no_writes(native):
    first = build(native)
    native.repository.scans.clear()
    changes = native.conn.total_changes
    assert build(native) == first
    assert any("spot" in query.columns for _, query in native.repository.scans)
    assert native.conn.total_changes == changes


def test_late_cap_poison_is_not_numerically_consumed():
    row = dict.fromkeys(DAILY, 1.0)
    row.update(ticker="SYNTH", date="2024-01-02", src_iv="synthetic", mcap_asof="2025-01-01")
    row["mcap_log"] = object()  # Would fail numeric conversion if late capture entered the kernel.
    normalized = daily._source_row(row, "SYNTH", 2024, "2024-01-01", "2024-01-03", set())
    assert normalized["mcap_log"] is None


def test_source_duplicate_refuses():
    row = dict.fromkeys(DAILY, 1.0)
    row.update(ticker="SYNTH", date="2024-01-02", src_iv="synthetic", mcap_asof=None)
    with pytest.raises(DataError, match="invalid daily block source row") as caught:
        daily._source_row(row, "SYNTH", 2024, "2024-01-01", "2024-01-03", {"2024-01-02"})
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_partial_blocks_are_rejected_as_stored_panel_rows(native):
    result = build(native)
    with pytest.raises(DataError, match="closed contract") as caught:
        magnitude_panel_table.validate_rows(result["rows"], native.parent)
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_future_source_changes_cannot_change_earlier_spy_prefix():
    dates = pd.bdate_range("2020-01-01", periods=310)
    rows = [{"date": day.date().isoformat(), "spot": 100.0 + i} for i, day in enumerate(dates)]
    events = [{"event_id": "early", "event_date": dates[270].date().isoformat(),
               "forecasts": [{"horizon_sessions": 1, "source_session_ceiling": dates[269].date().isoformat()}]},
              {"event_id": "late", "event_date": dates[-1].date().isoformat(),
               "forecasts": [{"horizon_sessions": 1, "source_session_ceiling": dates[-2].date().isoformat()}]}]
    expected = daily._spy_blocks(events, rows)
    changed = deepcopy(rows)
    for row in changed[270:]:
        row["spot"] *= 123
    actual = daily._spy_blocks(events, changed)
    assert actual["early", 1] == expected["early", 1]
    assert actual["late", 1]["values"] != expected["late", 1]["values"]


@pytest.mark.parametrize("cap_date,reason", [(None, "unavailable"), ("2024-01-03", "late_capture")])
def test_missing_and_late_cap_do_not_inherit_values(cap_date, reason):
    raw = dict.fromkeys(DAILY, 1.0)
    raw.update(ticker="SYNTH", date="2024-01-02", mcap_asof=cap_date, src_iv="synthetic")
    raw["mcap_log"] = object()
    row = daily._source_row(raw, "SYNTH", 2024, "2024-01-01", "2024-01-03", set())
    block = daily._daily([row], ["2024-01-02"], "2024-01-02")
    assert block["values"]["mcap_log"] is None and block["missing"]["mcap_log"] == reason


@pytest.mark.parametrize("changes", [{"ticker": "WRONG"}, {"date": "2024-01-02T12:00:00"},
    {"date": "2024-01-02T00:00:00Z"}, {"date": "2025-01-02"}, {"iv10": True},
    {"iv10": float("inf")}, {"implied_move": -1.0}])
def test_malformed_source_rows_refuse(changes):
    row = dict.fromkeys(DAILY, 1.0)
    row.update(ticker="SYNTH", date="2024-01-02", src_iv="synthetic", mcap_asof=None)
    row.update(changes)
    with pytest.raises(DataError, match="invalid daily block source row") as caught:
        daily._source_row(row, "SYNTH", 2024, "2024-01-01", "2024-01-03", set())
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_no_source_or_iv_surface_is_explicitly_missing():
    block = daily._daily([], [], "2024-01-01")
    assert block == {"values": dict.fromkeys(DAILY), "anchor": None, "quote_status": "unavailable",
                     "missing": dict.fromkeys(DAILY, "unavailable")}


def test_unknown_source_contract_refuses_before_values(native, monkeypatch):
    original = native.repository.table_contract
    def changed(snapshot, table):
        contract = original(snapshot, table)
        return replace(contract, contract_id="legacy.daily_market.v99") if table == "daily_market" else contract
    monkeypatch.setattr(native.repository, "table_contract", changed)
    with pytest.raises(DataError, match="unsupported daily block source contract") as caught:
        daily._query(native.repository, native.snapshot, "SPY", 2024, "2024-01-01", "2024-02-01", 100)
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_snapshot_and_policy_drift_cannot_reuse_context(native):
    with pytest.raises(DataError) as caught:
        daily.build_daily_blocks(native.repository, replace(native.snapshot, calendar_version="changed"),
            native.stored, native.parent, producer=native.producer, policy=native.policy, eligibility_metadata=native.metadata,
            source_start=START, max_source_rows=1000, max_events=10, max_object_bytes=2**20)
    assert caught.value.code == "IDENTITY_CONFLICT"
    with pytest.raises(DataError, match="stage-two eligibility policy") as caught:
        build(native, policy=eligibility_policy.eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=0))
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_event_and_source_start_bounds(native):
    with pytest.raises(DataError, match="event bound exceeded") as caught:
        build(native, max_events=1)
    assert caught.value.code == "RESULT_LIMIT_EXCEEDED"
    with pytest.raises(DataError, match="source start exceeds") as caught:
        build(native, source_start="2024-08-01")
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_producer_identity_is_required_detached_and_context_bound(native):
    with pytest.raises(DataError, match="producer identity") as caught:
        build(native, producer=native.parent["producer"])
    assert caught.value.code == "CONTRACT_MISMATCH"
    original = build(native)
    changed = deepcopy(native.producer)
    changed["environment"]["synthetic_change"] = "new"
    result = build(native, producer=changed)
    assert result["context"]["context_id"] != original["context"]["context_id"]
    changed["environment"]["synthetic_change"] = "mutated"
    assert result["context"]["producer"]["environment"]["synthetic_change"] == "new"


def test_unpinned_repository_refuses_before_access(native, monkeypatch):
    repository = TrackingRepository(native.conn, native.repository._store)
    monkeypatch.setattr(repository, "resolve", lambda *args: pytest.fail("unpinned access"))
    with pytest.raises(DataError, match="snapshot-pinned repository") as caught:
        daily.build_daily_blocks(repository, native.snapshot, native.stored, native.parent,
            producer=native.producer, policy=native.policy, eligibility_metadata=native.metadata,
            source_start=START, max_source_rows=1000, max_events=10, max_object_bytes=2**20)
    assert caught.value.code == "CONTRACT_MISMATCH"


class NonPlainList(list):
    pass


@pytest.mark.parametrize("value", [("synthetic",), NonPlainList(["synthetic"]), float("nan"), float("inf")])
def test_nonplain_producer_refuses_before_normalization_and_repository(native, monkeypatch, value):
    producer = deepcopy(native.producer)
    producer["environment"]["synthetic_probe"] = value
    monkeypatch.setattr(native.repository, "resolve", lambda *args: pytest.fail("unvalidated producer reached repository"))
    with pytest.raises(DataError) as caught:
        build(native, producer=producer)
    assert caught.value.code == "CONTRACT_MISMATCH"
