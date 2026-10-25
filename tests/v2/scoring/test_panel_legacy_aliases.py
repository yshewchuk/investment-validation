"""Legacy-named panel aliases: ``or_implied``, ``or_rvol30``, ``abs_dist_*``,
``has_implied_quote``.

Legacy sources: ``engine/features.py`` ``add_absolute_features`` /
``add_quote_indicators``; ``engine/data/features/panel.py`` ``ORATS_FEATURES``.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.features import add_absolute_features, add_quote_indicators
from engine.v2.contracts.data import DatasetVersionRef, SnapshotRef, TableContractRef
from engine.v2.data.computed_moves_table import (
    COMPUTED_MOVES_CONTRACT,
    COMPUTED_MOVES_TABLE_NAME,
)
from engine.v2.data.price_history_table import (
    PRICE_HISTORY_CONTRACT,
    PRICE_HISTORY_TABLE_NAME,
)
from engine.v2.data.errors import DataError
from engine.v2.features import panel_row_inputs
from engine.v2.features.panel_row_inputs import scan_panel_row
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.scoring.nightly_source_bundle import _project_features
from tests.data_scan_support import contract_for, contract_ref_for, fake_hash
from tests.test_v2_features_daily_state_inputs import _bound_row_selected

_DM = contract_for("daily_market")
_DM_REF = contract_ref_for(_DM)
_CM_REF = TableContractRef(
    contract_id=COMPUTED_MOVES_CONTRACT.contract_id,
    definition_hash=COMPUTED_MOVES_CONTRACT.definition_hash,
)
_PH_REF = TableContractRef(
    contract_id=PRICE_HISTORY_CONTRACT.contract_id,
    definition_hash=PRICE_HISTORY_CONTRACT.definition_hash,
)

_HISTORY_START = pd.Timestamp("2024-01-02")
_EVENT = pd.Timestamp("2024-02-15")
_DECISION = pd.Timestamp("2024-02-14")


class _Batch:
    def __init__(self, rows) -> None:
        self._rows = rows

    def to_pylist(self):
        return self._rows


class _FakeRepository:
    def __init__(self, snapshot, contracts, batches_by_key) -> None:
        self._snapshot = snapshot
        self._contracts = contracts
        self._batches = batches_by_key

    def resolve(self, snapshot_id: str) -> SnapshotRef:
        assert snapshot_id == self._snapshot.snapshot_id
        return self._snapshot

    def resolve_pinned(self, scope: str):
        raise AssertionError("scan_panel_row must never consult a head")

    def resolve_full_pinned(self, scope: str):
        raise AssertionError("scan_panel_row must never consult a head")

    def table_contract(self, snapshot_ref, table_name):
        if table_name not in snapshot_ref.table_versions:
            raise DataError(_problem("table is not part of this snapshot", table_name))
        return self._contracts[table_name]

    def scan_population_bound(self, snapshot_id, *, table_name, table_contract_ref,
                              key_filter=(), time_interval=None):
        snapshot = self.resolve(snapshot_id)
        version = snapshot.table_versions.get(table_name)
        if version is None:
            raise DataError(_problem("table is not part of this snapshot", table_name))
        if table_contract_ref != version.table_contract_ref:
            raise DataError(_problem("table_contract_ref does not match the pinned version",
                                     table_name))
        return sum(1 for (stored_table, _ticker), rows in self._batches.items()
                   if stored_table == table_name
                   for row in rows if _bound_row_selected(row, key_filter, time_interval))

    def scan(self, query, *, table_name, max_object_bytes=None):
        ticker = query.key_filter[0].values[0]
        rows = self._batches.get((table_name, ticker), [])
        selected = [row for row in rows if _bound_row_selected(
            row, query.key_filter, query.time_interval)]
        yield _Batch(selected)


def _problem(message, table_name):
    from engine.v2.data import errors
    return errors.make_problem("CONTRACT_MISMATCH", message, details={"table_name": table_name})


def _snapshot() -> SnapshotRef:
    versions = {
        "daily_market": DatasetVersionRef(
            dataset_version_id="dsv-dm", table_contract_ref=_DM_REF,
            manifest_hash=fake_hash("dm-manifest")),
        COMPUTED_MOVES_TABLE_NAME: DatasetVersionRef(
            dataset_version_id="dsv-cm", table_contract_ref=_CM_REF,
            manifest_hash=fake_hash("cm-manifest")),
        PRICE_HISTORY_TABLE_NAME: DatasetVersionRef(
            dataset_version_id="dsv-ph", table_contract_ref=_PH_REF,
            manifest_hash=fake_hash("ph-manifest")),
    }
    modes = {"daily_market": "reconstructed",
             COMPUTED_MOVES_TABLE_NAME: "observed",
             PRICE_HISTORY_TABLE_NAME: "reconstructed"}
    return SnapshotRef(snapshot_id="snap-panel", manifest_hash=fake_hash("panel-snapshot"),
                       table_versions=versions, calendar_version="cal.v1",
                       source_priority_version="prio.v1", finality_receipt_refs=(),
                       knowledge_mode_by_table=modes)


def _contracts():
    return {"daily_market": _DM, COMPUTED_MOVES_TABLE_NAME: COMPUTED_MOVES_CONTRACT,
            PRICE_HISTORY_TABLE_NAME: PRICE_HISTORY_CONTRACT}


def _dm_rows(ticker, days, *, src_iv="orats", implied_move=None, rvol30=None):
    rows = []
    for day in days:
        row = {"ticker": ticker, "date": pd.Timestamp(day), "src_iv": src_iv}
        if implied_move is not None:
            row["implied_move"] = implied_move
        if rvol30 is not None:
            row["rvol30"] = rvol30
        rows.append(row)
    return rows


def _computed_rows(rows_spec):
    rows = []
    for index, (day, move, skipped, implied) in enumerate(rows_spec):
        prior_implied = [row["implied_move_pct"] for row in rows
                         if row["implied_move_pct"] is not None]
        rows.append({
            "ticker": "AAA", "event_date": day, "realized_move_pct": move,
            "skipped": skipped, "implied_move_pct": implied,
            "available_as_of_date": None if skipped else
            (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat(),
            "event_ordinal": index, "admitted": True,
            "mean_prior_or_implied": (sum(prior_implied) / len(prior_implied)
                                      if prior_implied else None),
            "n_prior_implied": len(prior_implied),
        })
    return rows


def _price_rows(n=320, *, end="2024-01-20"):
    dates = pd.date_range(end=end, periods=n, freq="D")
    return [{"date": stamp.date().isoformat(), "close_adj": 100.0 + i, "close_raw": 100.0 + i,
             "high_raw": 100.0 + i, "retrieved_at": "2024-02-01T00:00:00Z", "deleted": False,
             "source_kind": "yfinance", "source_hash": fake_hash("px")}
            for i, stamp in enumerate(dates)]


def _gspc_rows(periods=22):
    days = [str(d.date()) for d in pd.date_range(end="2024-02-01", periods=periods, freq="D")]
    return [{"date": day, "close_adj": 500.0 + i, "close_raw": 500.0 + i,
             "high_raw": 500.0 + i, "retrieved_at": "2024-02-01T00:00:00Z",
             "deleted": False, "source_kind": "legacy_gspc_csv",
             "source_hash": fake_hash("gspc")}
            for i, day in enumerate(days)]


def _batches():
    return {
        ("daily_market", "AAA"): _dm_rows("AAA", ["2024-02-04", "2024-02-05"],
                                          implied_move=2.0, rvol30=1.5),
        (PRICE_HISTORY_TABLE_NAME, "^GSPC"): _gspc_rows(),
        (COMPUTED_MOVES_TABLE_NAME, "AAA"): _computed_rows([
            ("2024-01-10", 1.0, False, 4.0),
            ("2024-01-11", 1.0, False, 6.0),
            ("2024-01-12", None, True, None),
        ]),
        (PRICE_HISTORY_TABLE_NAME, "AAA"): _price_rows(),
    }


def _scan():
    snapshot = _snapshot()
    repo = _FakeRepository(snapshot, _contracts(), _batches())
    return scan_panel_row(repo, snapshot,
                          BoardRequest(ticker="AAA", strategy="STR-X",
                                       event_date=_EVENT, session="AMC"),
                          history_start=_HISTORY_START, decision_session=_DECISION)


def test_scan_panel_row_emits_legacy_named_inputs():
    panel = _scan().panel_row
    assert panel["or_implied"] == panel["im"]
    assert panel["or_rvol30"] == panel["rvol30"]
    assert panel["abs_dist_high"] == abs(panel["dist_high"])
    assert panel["abs_dist_ema"] == abs(panel["dist_ema"])
    assert panel["has_implied_quote"] == 1.0
    assert panel["mean_prior_or_implied"] == pytest.approx(5.0)


def test_derivations_match_legacy():
    frame = pd.DataFrame({"dist_high": [-3.0], "dist_ema": [2.5]})
    frame = add_absolute_features(frame)
    assert frame["abs_dist_high"][0] == pytest.approx(3.0)
    assert frame["abs_dist_ema"][0] == pytest.approx(2.5)

    quote = pd.DataFrame({"or_implied": [0.0, 2.0, float("nan")]})
    quote = add_quote_indicators(quote)
    assert list(quote["has_implied_quote"]) == [0.0, 1.0, 0.0]

    daily = {"im": 2.0}
    runup = {"dist_high": -3.0, "dist_ema": 2.5}
    out = panel_row_inputs._legacy_named_inputs(daily, runup)
    assert out["abs_dist_high"] == pytest.approx(3.0)
    assert out["abs_dist_ema"] == pytest.approx(2.5)
    assert out["has_implied_quote"] == 1.0

    assert out["or_implied"] == 2.0

    no_implied = panel_row_inputs._legacy_named_inputs({}, runup)
    assert no_implied["has_implied_quote"] == 0.0
    assert "or_implied" not in no_implied


def test_size_driver_names_not_missing_after_projection():
    feature_names = ("has_implied_quote", "abs_dist_ema", "or_implied", "or_rvol30",
                     "mean_prior_or_implied")
    _, mask = _project_features(
        tier4_row={}, panel_row=_scan().panel_row, feature_names=feature_names,
        as_of_ts=pd.Timestamp("2026-01-15"))
    assert all(not mask[name] for name in feature_names)