"""Tier-0 tests for the v2 experiments trades loader (decision 2026-09-29,
option A: a new loader over the committed v2 ``trades`` version, migrating
nothing yet, ``snapshot_id`` always explicit). Fixture machinery is reused
from ``tests/test_v2_research_build_trades.py`` and its own helpers.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.research import experiment_trades  # noqa: E402
from experiments import common_v2  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.test_v2_research_build_trades import _commit_all, _trade_row  # noqa: E402
from tests.test_v2_research_replay import _event_rows  # noqa: E402

PROVENANCE = experiment_trades.PROVENANCE


def _str_thru_fixture_rows() -> list[dict]:
    """The two STR-THRU/PROVENANCE rows both value-comparison tests seed."""
    return [
        _trade_row("T-THRU-A", "STR-THRU", PROVENANCE, fill_alpha=0.0),
        _trade_row("T-THRU-B", "STR-THRU", PROVENANCE, fill_alpha=0.5),
    ]


def _expected_str_thru_frame(columns: list[str]) -> pd.DataFrame:
    """The explicit expected frame for every seeded value of
    ``_str_thru_fixture_rows()`` plus the ``session`` the join adds."""
    rows = [{**row, "session": "AMC"} for row in _str_thru_fixture_rows()]
    frame = pd.DataFrame(rows)[columns]
    for column in ("event_date", "entry_date", "exit_date", "expiry"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def _load_str_thru_sorted(tmp_path) -> pd.DataFrame:
    """One committed fixture, loaded for STR-THRU, sorted by ``trade_id``
    (fixture insertion order is not guaranteed to survive the merge)."""
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit_all(conn, clock, store, receipt_id="r1", trades_rows=[
        *_str_thru_fixture_rows(),
        _trade_row("T-CALP-A", "CAL-P", PROVENANCE),
        _trade_row("T-THRU-LEGACY", "STR-THRU", "engine.replay"),
    ])
    repository = Repository(conn, store)
    frame = experiment_trades.load_trades(repository, snapshot, "STR-THRU")
    conn.close()
    return frame.sort_values("trade_id").reset_index(drop=True)


def test_load_trades_returns_the_strategy_rows_with_session_joined(tmp_path):
    actual = _load_str_thru_sorted(tmp_path)

    assert set(actual["trade_id"].astype(str)) == {"T-THRU-A", "T-THRU-B"}
    expected = _expected_str_thru_frame(list(actual.columns)).astype(
        actual.dtypes.to_dict())
    pd.testing.assert_frame_equal(actual, expected)
    for column in ("event_date", "entry_date", "exit_date"):
        assert pd.api.types.is_datetime64_any_dtype(actual[column])


def test_load_trades_comparison_catches_a_corrupted_value(tmp_path):
    """The planted defect proves the frame comparison above is real: it
    fails on a corrupted cell, so its pass is not vacuous."""
    actual = _load_str_thru_sorted(tmp_path)
    expected = _expected_str_thru_frame(list(actual.columns)).astype(
        actual.dtypes.to_dict())

    actual.loc[actual["trade_id"] == "T-THRU-A", "entry_cost"] = 999.0

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(actual, expected)


def test_load_trades_refuses_empty_result(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit_all(conn, clock, store, receipt_id="r1", trades_rows=[
        _trade_row("T-CALP-A", "CAL-P", PROVENANCE),
    ])
    repository = Repository(conn, store)

    with pytest.raises(DataError) as excinfo:
        experiment_trades.load_trades(repository, snapshot, "STR-THRU")
    assert excinfo.value.code == "POPULATION_COLLAPSED"
    conn.close()


def test_load_trades_refuses_a_trades_row_with_no_matching_event(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    orphan = _trade_row("T-THRU-ORPHAN", "STR-THRU", PROVENANCE)
    orphan["event_id"] = "GHOST_2024-05-02"
    snapshot = _commit_all(conn, clock, store, receipt_id="r1",
                           trades_rows=[orphan])
    repository = Repository(conn, store)

    with pytest.raises(DataError) as excinfo:
        experiment_trades.load_trades(repository, snapshot, "STR-THRU")
    assert excinfo.value.code == "CONTRACT_MISMATCH"
    conn.close()


def test_load_trades_refuses_an_empty_trades_table(tmp_path):
    """A ``trades`` table registered in the snapshot with ZERO fragments (the
    ``commit_tables(conn, clock, {"<table>": []}, ...)`` precedent from
    ``test_v2_research_snapshot_read_table``): ``_snapshot.read_table``'s bare
    ``ValueError`` (issue #70) must reach the caller as a typed refusal.
    Distinct from the no-trades-table-entry case below (``CONTRACT_MISMATCH``).
    """
    conn, clock, store = catalog_and_store(tmp_path)
    trades_contract = contract_for("trades")
    events_contract = contract_for("earnings_events")
    events_record = publish_and_inspect(
        store, events_contract, contract_ref_for(events_contract),
        _event_rows(), "2024")
    snapshot = commit_tables(
        conn, clock, {"trades": [], "earnings_events": [events_record]},
        {"trades": trades_contract, "earnings_events": events_contract},
        store=store)
    repository = Repository(conn, store)

    with pytest.raises(DataError) as excinfo:
        experiment_trades.load_trades(repository, snapshot, "STR-THRU")
    assert excinfo.value.code == "POPULATION_COLLAPSED"
    conn.close()


def test_load_trades_refuses_a_snapshot_with_no_trades_table(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    events_contract = contract_for("earnings_events")
    events_record = publish_and_inspect(
        store, events_contract, contract_ref_for(events_contract),
        _event_rows(), "2024")
    snapshot = commit_tables(
        conn, clock, {"earnings_events": [events_record]},
        {"earnings_events": events_contract}, store=store)
    repository = Repository(conn, store)

    with pytest.raises(DataError) as excinfo:
        experiment_trades.load_trades(repository, snapshot, "STR-THRU")
    assert excinfo.value.code == "CONTRACT_MISMATCH"
    conn.close()


def test_load_v2_trades_requires_an_explicit_snapshot_id():
    param = inspect.signature(common_v2.load_v2_trades).parameters["snapshot_id"]
    assert param.default is inspect.Parameter.empty
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_load_v2_trades_raises_snapshot_not_found_for_an_unknown_id(tmp_path):
    conn, _clock, _store = catalog_and_store(tmp_path)
    conn.close()

    with pytest.raises(DataError) as excinfo:
        common_v2.load_v2_trades(
            "STR-THRU", catalog=tmp_path / "catalog.sqlite",
            store_root=tmp_path / "store", snapshot_id="does-not-exist")
    assert excinfo.value.code == "SNAPSHOT_NOT_FOUND"
