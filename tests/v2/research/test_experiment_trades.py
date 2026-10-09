"""Tier-0 tests for the v2 experiments trades loader (decision 2026-09-29,
option A: a new loader over the committed v2 ``trades`` version, migrating
nothing yet, ``snapshot_id`` always explicit). Fixture machinery is reused
from ``tests/test_v2_research_build_trades.py`` and its own helpers.
"""
from __future__ import annotations

import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.research import experiment_trades  # noqa: E402
from engine.v2.research._trades_revisions import _TRADES_COLUMNS  # noqa: E402
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

#: The compared columns: the trades-table contract plus the one column the
#: ``earnings_events`` join adds -- an explicit list, never read back from the
#: frame under test.
_COMPARED_COLUMNS = (*_TRADES_COLUMNS, "session")

#: Columns ``load_trades`` returns as datetimes, and columns that are floats.
_DATE_COLUMNS = ("event_date", "entry_date", "exit_date", "expiry")
_FLOAT_COLUMNS = ("strike", "fill_alpha", "entry_cost", "exit_value", "ret")


def _str_thru_fixture_rows() -> list[dict]:
    """The two STR-THRU/PROVENANCE rows both value-comparison tests seed."""
    return [
        _trade_row("T-THRU-A", "STR-THRU", PROVENANCE, fill_alpha=0.0),
        _trade_row("T-THRU-B", "STR-THRU", PROVENANCE, fill_alpha=0.5),
    ]


def _expected_str_thru_frame() -> pd.DataFrame:
    """The explicit expected frame for every seeded value of
    ``_str_thru_fixture_rows()`` plus the ``session`` the join adds.

    Column set and dtypes are chosen here, independently of the frame under
    test: ``_COMPARED_COLUMNS`` comes from the trades-table contract, and the
    datetimes/floats are coerced by name.
    """
    rows = [{**row, "session": "AMC"} for row in _str_thru_fixture_rows()]
    frame = pd.DataFrame(rows)[list(_COMPARED_COLUMNS)]
    for column in _DATE_COLUMNS:
        # Pinned to nanoseconds: pandas 3 parses in-process dates at
        # microsecond resolution while the scan returns the contract's
        # ``datetime64[ns]`` (``engine/data/schemas.py`` pins ns for the same
        # reason).
        frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    for column in _FLOAT_COLUMNS:
        frame[column] = frame[column].astype("float64")
    return frame


def _assert_str_thru_dtypes(frame: pd.DataFrame) -> None:
    """The actual frame's dtypes, checked by name against this test's own
    expectations -- never against the actual frame's own ``dtypes`` mapping."""
    for column in _DATE_COLUMNS:
        assert pd.api.types.is_datetime64_any_dtype(frame[column]), column
    for column in _FLOAT_COLUMNS:
        assert pd.api.types.is_float_dtype(frame[column]), column


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
    frame = experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2025-01")
    conn.close()
    return frame.sort_values("trade_id").reset_index(drop=True)


def test_load_trades_returns_the_strategy_rows_with_session_joined(tmp_path):
    actual = _load_str_thru_sorted(tmp_path)

    assert set(actual["trade_id"].astype(str)) == {"T-THRU-A", "T-THRU-B"}
    compared = actual[list(_COMPARED_COLUMNS)]
    _assert_str_thru_dtypes(compared)
    # ``check_dtype=False`` is only the resolution/backing exemption
    # ``test_v2_research_replay.py`` documents (Arrow gives datetime64[ns] and
    # object strings, in-process pandas 3 gives microsecond-resolution dates
    # and str strings); the dtype check above is the real one, and every value
    # is still compared exactly.
    pd.testing.assert_frame_equal(
        compared, _expected_str_thru_frame(), check_dtype=False)


def test_load_trades_comparison_catches_a_corrupted_value(tmp_path):
    """The planted defect proves the frame comparison above is real: it
    fails on a corrupted cell, so its pass is not vacuous."""
    actual = _load_str_thru_sorted(tmp_path)
    _assert_str_thru_dtypes(actual[list(_COMPARED_COLUMNS)])

    actual.loc[actual["trade_id"] == "T-THRU-A", "entry_cost"] = 999.0

    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            actual[list(_COMPARED_COLUMNS)], _expected_str_thru_frame(),
            check_dtype=False)


def test_load_trades_refuses_empty_result(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit_all(conn, clock, store, receipt_id="r1", trades_rows=[
        _trade_row("T-CALP-A", "CAL-P", PROVENANCE),
    ])
    repository = Repository(conn, store)

    with pytest.raises(DataError) as excinfo:
        experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2025-01")
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
        experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2025-01")
    assert excinfo.value.code == "HOLDOUT_ACCESS_DENIED"
    assert excinfo.value.problem.details["holdout_exclusions"][0]["memberships"] == ["ambiguous"]
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
        experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2025-01")
    assert excinfo.value.code == "POPULATION_COLLAPSED"
    conn.close()


def test_load_trades_refuses_a_snapshot_with_an_empty_events_table(tmp_path):
    """Mirror of the empty-``trades``-table case above, for
    ``earnings_events``: a normal one-row ``trades`` table plus a ZERO-fragment
    ``earnings_events`` table (the same ``commit_tables(conn, clock,
    {"<table>": []}, ...)`` pattern) must reach the caller as a typed
    ``CONTRACT_MISMATCH`` -- ``read_event_rows``' bare ``ValueError`` (issue
    #70) is the same exposure the ``read_existing_trades`` call already
    converts, and an empty events table trivially leaves every row's
    ``session`` unmatched."""
    conn, clock, store = catalog_and_store(tmp_path)
    trades_contract = contract_for("trades")
    events_contract = contract_for("earnings_events")
    trades_record = publish_and_inspect(
        store, trades_contract, contract_ref_for(trades_contract),
        [_trade_row("T-THRU-A", "STR-THRU", PROVENANCE)], "2024")
    snapshot = commit_tables(
        conn, clock, {"trades": [trades_record], "earnings_events": []},
        {"trades": trades_contract, "earnings_events": events_contract},
        store=store)
    repository = Repository(conn, store)

    with pytest.raises(DataError) as excinfo:
        experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2025-01")
    assert excinfo.value.code == "CONTRACT_MISMATCH"
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
        experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2025-01")
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


def _holdout_snapshot(tmp_path, members, event_overrides=None, extra_trades=()):
    """Real committed tables; membership is never mocked or pre-labelled."""
    conn, clock, store = catalog_and_store(tmp_path)
    events, trades = [], []
    for event_id, date, trade_date in members:
        events.append({**_event_rows()[0], "event_id": event_id,
                       "event_date": pd.Timestamp(date),
                       **(event_overrides or {}).get(event_id, {})})
        trades.append({**_trade_row(f"T-{event_id}", "STR-THRU", PROVENANCE),
                       "event_id": event_id, "event_date": pd.Timestamp(trade_date or date)})
    trades.extend(extra_trades)
    tables, contracts = {}, {}
    for name, rows in (("earnings_events", events), ("trades", trades)):
        contract = contract_for(name)
        contracts[name] = contract
        key = "event_id" if name == "earnings_events" else "trade_id"
        tables[name] = [publish_and_inspect(store, contract, contract_ref_for(contract),
                                          sorted(rows, key=lambda row: row[key]), "2024")]
    snapshot = commit_tables(conn, clock, tables, contracts, store=store)
    return conn, Repository(conn, store), snapshot


# Golden IDs for the versioned SHA-256 definition, not a search using the
# classifier under test. EVENT-5/30 are random members; EVENT-0/1/2 are not.
_SAFE = ("EVENT-0", "2024-01-01", None)
_RANDOM = ("EVENT-5", "2024-01-16", None)
_ROLLING = ("EVENT-1", "2024-09-15", None)
_OVERLAP = ("EVENT-30", "2024-09-16", None)
_AMBIGUOUS = ("EVENT-2", "2024-01-15", "2024-02-15")


@pytest.mark.parametrize("purpose", ["training", "selection", "sweep"])
@pytest.mark.parametrize("members,excluded", [
    ([_RANDOM], {"EVENT-5": ["random"]}),
    ([_ROLLING], {"EVENT-1": ["rolling"]}),
    ([_RANDOM, _ROLLING, _OVERLAP], {"EVENT-5": ["random"],
        "EVENT-1": ["rolling"], "EVENT-30": ["random", "rolling"]}),
    ([_AMBIGUOUS], {"EVENT-2": ["ambiguous"]}),
])
def test_real_loader_excludes_each_holdout_union_and_ambiguity(tmp_path, purpose, members, excluded):
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_SAFE, *members])
    try:
        frame = experiment_trades.load_trades(repository, snapshot, "STR-THRU",
                                              as_of_month="2024-10", purpose=purpose)
        assert frame["event_id"].tolist() == ["EVENT-0"]
        assert {x["event_id"]: x["memberships"] for x in frame.attrs["holdout_exclusions"]} == excluded
        assert frame["snapshot_id"].tolist() == [snapshot.snapshot_id]
        assert frame["holdout_as_of_month"].tolist() == ["2024-10"]
        assert frame["random_membership_version"].tolist() == ["canonical-event-sha256.v1"]
        assert frame["rolling_membership_version"].tolist() == ["calendar-months.v1"]
        assert frame["population_use"].tolist() == ["post-release selection"]
    finally:
        conn.close()


@pytest.mark.parametrize("purpose,member", [("training", _RANDOM), ("training", _ROLLING),
    ("selection", _OVERLAP), ("sweep", _RANDOM), ("sweep", _ROLLING), ("sweep", _AMBIGUOUS)])
def test_requested_holdout_refuses_before_any_metric_or_partial_report(tmp_path, purpose, member):
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_SAFE, member])
    output = tmp_path / "report.json"
    metrics = []
    try:
        with pytest.raises(DataError) as caught:
            frame = experiment_trades.load_trades(repository, snapshot, "STR-THRU",
                as_of_month="2024-10", purpose=purpose, event_ids=[_SAFE[0], member[0]])
            metrics.append(frame["ret"].mean())
            output.write_text(str(metrics))
        assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
        assert caught.value.problem.retryable is False
        assert caught.value.problem.details["snapshot_id"] == snapshot.snapshot_id
        assert caught.value.problem.details["holdout_exclusions"][0]["event_id"] == member[0]
        assert not metrics
        assert not output.exists()
    finally:
        conn.close()


def test_monthly_release_does_not_release_random_overlap(tmp_path):
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_SAFE, _RANDOM, _ROLLING, _OVERLAP])
    try:
        released = experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2025-05")
        assert set(released["event_id"]) == {"EVENT-0", "EVENT-1"}
        assert released.attrs["holdout_exclusions"] == [
            {"event_id": "EVENT-30", "memberships": ["random"]},
            {"event_id": "EVENT-5", "memberships": ["random"]}]
    finally:
        conn.close()


@pytest.mark.parametrize("kwargs", [{}, {"as_of_month": "bad"},
    {"as_of_month": "9999-12"},
    {"as_of_month": "2024-10", "purpose": "final_evaluation"},
    {"as_of_month": "2024-10", "event_ids": "EVENT-5"}])
def test_no_missing_context_fallback_or_final_read_api(tmp_path, kwargs):
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_SAFE, _RANDOM])
    try:
        with pytest.raises(DataError) as caught:
            experiment_trades.load_trades(repository, snapshot, "STR-THRU", **kwargs)
        assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
    finally:
        conn.close()


def test_entirely_excluded_population_refuses(tmp_path):
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM, _OVERLAP])
    try:
        with pytest.raises(DataError) as caught:
            experiment_trades.load_trades(repository, snapshot, "STR-THRU",
                                          as_of_month="2024-10", purpose="sweep")
        assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
        assert len(caught.value.problem.details["holdout_exclusions"]) == 2
    finally:
        conn.close()


@pytest.mark.parametrize("override", [{"date_conflict": True}, {"session": None}])
def test_canonical_calendar_ambiguity_is_excluded_and_labelled(tmp_path, override):
    member = ("EVENT-2", "2024-01-15", None)
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_SAFE, member], {"EVENT-2": override})
    try:
        frame = experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2024-10")
        assert frame["event_id"].tolist() == ["EVENT-0"]
        assert frame.attrs["holdout_exclusions"] == [{"event_id": "EVENT-2", "memberships": ["ambiguous"]}]
    finally:
        conn.close()


def test_conflicting_cluster_rows_cannot_change_membership_by_identity(tmp_path):
    members = [_SAFE, ("EVENT-1", "2024-01-15", None), ("EVENT-2", "2024-01-16", None)]
    overrides = {key: {"event_cluster_id": "shared"} for key in ("EVENT-1", "EVENT-2")}
    conn, repository, snapshot = _holdout_snapshot(tmp_path, members, overrides)
    try:
        frame = experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2024-10")
        assert frame["event_id"].tolist() == ["EVENT-0"]
        assert frame.attrs["holdout_exclusions"] == [
            {"event_id": key, "memberships": ["ambiguous"]} for key in ("EVENT-1", "EVENT-2")]
    finally:
        conn.close()


def test_future_as_of_cannot_release_events_before_the_current_utc_month(tmp_path, monkeypatch):
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_SAFE, _ROLLING])
    monkeypatch.setattr(experiment_trades.SystemClock, "now",
                        lambda self: datetime(2024, 10, 1, tzinfo=timezone.utc))
    try:
        with pytest.raises(DataError) as caught:
            experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2024-11")
        assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
        frame = experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2024-10")
        assert frame["event_id"].tolist() == ["EVENT-0"]
    finally:
        conn.close()


def test_explicit_population_with_unresolved_event_refuses_without_partial_frame(tmp_path):
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_SAFE])
    try:
        with pytest.raises(DataError) as caught:
            experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2024-10",
                                          event_ids=["EVENT-0", "MISSING"])
        assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
        assert caught.value.problem.details["holdout_exclusions"] == [
            {"event_id": "MISSING", "memberships": ["ambiguous"]}]
    finally:
        conn.close()


@pytest.mark.parametrize("include_safe", [True, False])
def test_null_identity_and_random_exclusions_are_typed_and_json_safe(tmp_path, include_safe):
    orphan = {**_trade_row("T-NULL", "STR-THRU", PROVENANCE), "event_id": None}
    members = [_RANDOM, _SAFE] if include_safe else [_RANDOM]
    conn, repository, snapshot = _holdout_snapshot(tmp_path, members, extra_trades=[orphan])
    expected = [{"event_id": None, "memberships": ["ambiguous"]},
                {"event_id": "EVENT-5", "memberships": ["random"]}]
    try:
        if include_safe:
            frame = experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2024-10")
            assert frame["event_id"].tolist() == ["EVENT-0"]
            evidence = frame.attrs["holdout_exclusions"]
        else:
            with pytest.raises(DataError) as caught:
                experiment_trades.load_trades(repository, snapshot, "STR-THRU", as_of_month="2024-10")
            assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
            evidence = caught.value.problem.details["holdout_exclusions"]
        assert evidence == expected
        assert json.loads(json.dumps(evidence, allow_nan=False)) == expected
    finally:
        conn.close()


@pytest.mark.parametrize("purpose", ["training", "selection", "sweep"])
@pytest.mark.parametrize("explicit", [False, True])
def test_duplicate_canonical_coordinates_cannot_alias_a_random_holdout(tmp_path, purpose, explicit):
    alias = (_RANDOM[0], _SAFE[1], None)
    independent = ("EVENT-2", "2023-12-01", None)
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_SAFE, alias, independent])
    output = tmp_path / "report.json"
    try:
        kwargs = dict(as_of_month="2024-10", purpose=purpose)
        if explicit:
            # The random alias is outside the requested population, but still
            # makes its other ID's canonical identity ambiguous.
            with pytest.raises(DataError) as caught:
                frame = experiment_trades.load_trades(repository, snapshot, "STR-THRU",
                                                      event_ids=[_SAFE[0]], **kwargs)
                output.write_text(str(frame["ret"].mean()))
            assert caught.value.code == "HOLDOUT_ACCESS_DENIED"
            assert caught.value.problem.details["holdout_exclusions"] == [
                {"event_id": _SAFE[0], "memberships": ["ambiguous"]}]
            assert not output.exists()
        else:
            frame = experiment_trades.load_trades(repository, snapshot, "STR-THRU", **kwargs)
            assert frame["event_id"].tolist() == [independent[0]]
            assert frame.attrs["holdout_exclusions"] == [
                {"event_id": _SAFE[0], "memberships": ["ambiguous"]},
                {"event_id": _RANDOM[0], "memberships": ["ambiguous", "random"]}]
    finally:
        conn.close()
