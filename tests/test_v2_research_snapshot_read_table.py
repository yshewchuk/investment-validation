"""Issue #107: ``engine.v2.research._snapshot.read_table`` delegates its scan
to ``_scan.read_table`` so each requested partition is bounded and split into
calendar months (then days) on ``RESULT_LIMIT_EXCEEDED``.

Every case commits a real ``option_chains``-shaped snapshot through
``tests/data_scan_support.py``'s helpers (the same pattern
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
