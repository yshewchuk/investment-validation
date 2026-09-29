"""Issue #107: ``engine.v2.research._snapshot.read_table`` delegates its scan
to ``_scan.read_table`` so each requested partition is bounded and split into
calendar months (then days) on ``RESULT_LIMIT_EXCEEDED``.

Every case commits a real ``option_chains``- or ``trades``-shaped snapshot
through ``tests/data_scan_support.py``'s helpers (the same pattern
``tests/test_v2_research_replay.py`` uses) with a SMALL test
``maximum_result_rows`` override, then reads through
``_snapshot.read_table`` — directly, and through the two callers issue #107
names (``_chains.read_chain_keys`` / ``_chains.load_chain_index``).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data import manifests  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.research import _chains  # noqa: E402
from engine.v2.research._snapshot import read_table  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)

_COLUMNS = ("ticker", "obs_date", "year")


def _capped_contract(maximum_result_rows: int):
    """The real option_chains contract with a small test result cap.

    ``maximum_result_rows`` is covered by ``definition_hash``, so the hash is
    recomputed after the override — otherwise ``commit_snapshot`` would refuse
    the contract as MANIFEST_CORRUPT before any read could run.
    """
    contract = dataclasses.replace(contract_for("option_chains"),
                                   maximum_result_rows=maximum_result_rows)
    return dataclasses.replace(contract,
                               definition_hash=manifests.table_contract_hash(contract))


def _capped_trades_contract(maximum_result_rows: int):
    """The real trades contract with a small test result cap.

    Same hash-recompute reason as :func:`_capped_contract`: ``trades`` declares
    a nullable observation column (``entry_date``), which is exactly what the
    three cases below exercise.
    """
    contract = dataclasses.replace(contract_for("trades"),
                                   maximum_result_rows=maximum_result_rows)
    return dataclasses.replace(contract,
                               definition_hash=manifests.table_contract_hash(contract))


def _chain_rows(ticker: str, counts_by_date: dict[str, int], year: int) -> list[dict]:
    """Synthetic option_chains rows, ``counts_by_date`` rows per obs_date,
    already ordered by the table's full primary key."""
    expiry = pd.Timestamp(f"{year}-12-20")
    rows: list[dict] = []
    for obs_date, count in counts_by_date.items():
        obs = pd.Timestamp(obs_date)
        for i in range(count):
            rows.append(
                {
                    "ticker": ticker, "obs_date": obs, "year": year,
                    "expiry": expiry, "dte": 30, "strike": 100.0 + i,
                    "right": "C" if i % 2 == 0 else "P",
                    "bid": 1.0, "ask": 1.2, "spot": 100.0,
                    "quote_repaired": False,
                }
            )
    rows.sort(key=lambda row: (row["ticker"], row["obs_date"], row["expiry"],
                               row["strike"], row["right"]))
    return rows


def _trades_rows(year: int, entries: dict[str, str | None]) -> list[dict]:
    """Synthetic trades rows, one per ``trade_id -> entry_date`` (``None``
    keeps the nullable observation time NULL), already ordered by the table's
    primary key. Every other nullable column may be absent: ``table_from_rows``
    fills it with a typed null.
    """
    rows = [
        {
            "trade_id": trade_id, "kind": "sim", "strategy": "CAL-P", "ticker": "TEST",
            "year": year,
            "entry_date": None if entry_date is None else pd.Timestamp(entry_date),
        }
        for trade_id, entry_date in entries.items()
    ]
    rows.sort(key=lambda row: row["trade_id"])
    return rows


def _commit_trades(tmp_path, fragments: dict[str, list[dict]], *, maximum_result_rows: int):
    conn, clock, store = catalog_and_store(tmp_path)
    contract = _capped_trades_contract(maximum_result_rows)
    ref = contract_ref_for(contract)
    records = [publish_and_inspect(store, contract, ref, rows, partition_key)
               for partition_key, rows in fragments.items()]
    snap = commit_tables(conn, clock, {"trades": records},
                         {"trades": contract}, store=store)
    return conn, store, snap


def _commit_chains(tmp_path, fragments: dict[str, list[dict]], *, maximum_result_rows: int):
    conn, clock, store = catalog_and_store(tmp_path)
    contract = _capped_contract(maximum_result_rows)
    ref = contract_ref_for(contract)
    records = [publish_and_inspect(store, contract, ref, rows, partition_key)
               for partition_key, rows in fragments.items()]
    snap = commit_tables(conn, clock, {"option_chains": records},
                         {"option_chains": contract}, store=store)
    return conn, store, snap


def test_read_table_splits_a_partition_over_the_cap(tmp_path):
    conn, store, snap = _commit_chains(
        tmp_path,
        {"2024": _chain_rows("TEST", {"2024-01-10": 2, "2024-02-12": 3}, 2024)},
        maximum_result_rows=4,
    )
    repository = Repository(conn, store)

    frame = read_table(repository, snap, "option_chains", ("ticker", "obs_date"),
                       partition_keys=["2024"])
    assert len(frame) == 5

    keys = {("TEST", pd.Timestamp("2024-01-10")), ("TEST", pd.Timestamp("2024-02-12"))}
    assert _chains.read_chain_keys(repository, snap) == keys

    index = _chains.load_chain_index(repository, snap, keys)
    assert len(index) == 2
    entry = index.get("TEST", "2024-01-10")
    exit_ = index.get("TEST", "2024-02-12")
    assert entry is not None and len(entry) == 2
    assert exit_ is not None and len(exit_) == 3
    conn.close()


def test_read_table_splits_a_month_over_the_cap_by_day(tmp_path):
    """Two days in the SAME month, each under the cap, whose combined month
    total exceeds it: the month-level scan overflows and is re-scanned one
    day at a time (``_scan.py``'s day-level split), never refusing."""
    conn, store, snap = _commit_chains(
        tmp_path,
        {"2024": _chain_rows("TEST", {"2024-01-10": 3, "2024-01-20": 3}, 2024)},
        maximum_result_rows=4,
    )
    repository = Repository(conn, store)

    frame = read_table(repository, snap, "option_chains", ("ticker", "obs_date"),
                       partition_keys=["2024"])
    assert len(frame) == 6

    keys = {("TEST", pd.Timestamp("2024-01-10")), ("TEST", pd.Timestamp("2024-01-20"))}
    assert _chains.read_chain_keys(repository, snap) == keys
    conn.close()


def test_read_table_without_partition_keys_scans_every_year(tmp_path):
    conn, store, snap = _commit_chains(
        tmp_path,
        {
            "2024": _chain_rows("TEST", {"2024-01-10": 2, "2024-02-12": 3}, 2024),
            "2025": _chain_rows("TEST", {"2025-03-11": 2, "2025-04-15": 3}, 2025),
        },
        maximum_result_rows=4,
    )
    repository = Repository(conn, store)

    frame = read_table(repository, snap, "option_chains", _COLUMNS)
    assert len(frame) == 10
    assert set(pd.to_datetime(frame["obs_date"])) == {
        pd.Timestamp("2024-01-10"), pd.Timestamp("2024-02-12"),
        pd.Timestamp("2025-03-11"), pd.Timestamp("2025-04-15"),
    }
    assert set(frame["year"]) == {2024, 2025}
    conn.close()


def test_read_table_still_refuses_a_day_over_the_cap(tmp_path):
    conn, store, snap = _commit_chains(
        tmp_path,
        {"2024": _chain_rows("TEST", {"2024-01-15": 5}, 2024)},
        maximum_result_rows=4,
    )
    repository = Repository(conn, store)

    with pytest.raises(DataError) as err:
        read_table(repository, snap, "option_chains", ("ticker", "obs_date"),
                   partition_keys=["2024"])
    assert err.value.code == "RESULT_LIMIT_EXCEEDED"
    conn.close()


def test_read_table_empty_result_keeps_object_dtypes(tmp_path):
    conn, store, snap = _commit_chains(
        tmp_path,
        {"2024": _chain_rows("TEST", {"2024-01-10": 2}, 2024)},
        maximum_result_rows=4,
    )
    repository = Repository(conn, store)

    frame = read_table(repository, snap, "option_chains", _COLUMNS,
                       partition_keys=["1999"])
    assert len(frame) == 0
    for column in _COLUMNS:
        assert frame[column].dtype == np.dtype("object")
    conn.close()


def test_read_table_normalizes_a_non_canonical_int_partition_key(tmp_path):
    conn, store, snap = _commit_chains(
        tmp_path,
        {"2024": _chain_rows("TEST", {"2024-01-10": 2}, 2024)},
        maximum_result_rows=4,
    )
    repository = Repository(conn, store)

    frame = read_table(repository, snap, "option_chains", _COLUMNS,
                       partition_keys=["02024"])
    assert len(frame) == 2
    conn.close()


def test_read_table_with_no_fragments_still_refuses_a_bare_value_error(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    contract = _capped_contract(4)
    snap = commit_tables(conn, clock, {"option_chains": []},
                         {"option_chains": contract}, store=store)
    repository = Repository(conn, store)

    with pytest.raises(ValueError) as err:
        read_table(repository, snap, "option_chains", ("ticker", "obs_date"))
    assert type(err.value) is ValueError
    conn.close()


# --------------------------------------------------------------------------
# trades: nullable observation column (entry_date is NULLABLE; year is the
# partition column), so a partition scan must be scoped to its own year and
# must not rely on an interval that can never match a null entry_date
# --------------------------------------------------------------------------


def test_read_table_scopes_each_partition_and_returns_no_duplicate_rows(tmp_path):
    """Cross-partition dedup (#107): a row's ``entry_date`` may fall in the
    OTHER partition's calendar year (a trade entered just after its earnings
    event crossed a year boundary), so each partition's scan is scoped to its
    own ``year``. Keyed off ``trade_id`` uniqueness, not a bare row count."""
    conn, store, snap = _commit_trades(
        tmp_path,
        {
            "2024": _trades_rows(2024, {"T1": "2025-01-02", "T2": "2024-03-01"}),
            "2025": _trades_rows(2025, {"T3": "2024-12-30", "T4": "2025-05-01"}),
        },
        maximum_result_rows=2,
    )
    repository = Repository(conn, store)

    frame = read_table(repository, snap, "trades", ("trade_id", "year", "entry_date"),
                       partition_keys=["2024", "2025"])
    assert len(frame) == 4
    assert frame["trade_id"].is_unique
    assert set(frame["trade_id"]) == {"T1", "T2", "T3", "T4"}
    conn.close()


def test_read_table_keeps_a_null_entry_date_row_under_the_cap(tmp_path):
    """Null observation-column rows (#107): a partition comfortably under the
    cap is scanned whole, so the row with ``entry_date=None`` comes back too
    (checked by ``trade_id``: a bare count could hide it behind another)."""
    conn, store, snap = _commit_trades(
        tmp_path,
        {"2024": _trades_rows(2024, {"T1": "2024-01-10", "T2": None,
                                     "T3": "2024-02-12"})},
        maximum_result_rows=4,
    )
    repository = Repository(conn, store)

    frame = read_table(repository, snap, "trades", ("trade_id", "entry_date"),
                       partition_keys=["2024"])
    assert set(frame["trade_id"]) == {"T1", "T2", "T3"}
    null_row = frame.loc[frame["trade_id"] == "T2"]
    assert len(null_row) == 1
    assert pd.isna(null_row["entry_date"]).all()
    conn.close()


def test_read_table_refuses_to_split_a_partition_with_a_nullable_observation(tmp_path):
    """A nullable-observation partition over the cap (#107): the month/day
    split can never place a null ``entry_date`` row in an interval, so the read
    refuses with ``RESULT_LIMIT_EXCEEDED`` rather than silently returning a
    frame missing that row (which the pre-fix code did)."""
    conn, store, snap = _commit_trades(
        tmp_path,
        {"2024": _trades_rows(2024, {"T1": "2024-01-10", "T2": "2024-02-12",
                                     "T3": None, "T4": "2024-01-20",
                                     "T5": "2024-02-22"})},
        maximum_result_rows=4,
    )
    repository = Repository(conn, store)

    with pytest.raises(DataError) as err:
        read_table(repository, snap, "trades", ("trade_id", "entry_date"),
                   partition_keys=["2024"])
    assert err.value.code == "RESULT_LIMIT_EXCEEDED"
    conn.close()
