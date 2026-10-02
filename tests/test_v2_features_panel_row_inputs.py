"""Direct tests for ``engine.v2.features.panel_row_inputs.scan_panel_row``.

One fake repository serves the four pinned-snapshot reads the composition needs
from a single fixture: ``daily_market`` (for both ``key.ticker`` and the fixed
``"SPY"``), ``computed_moves``, and ``price_history``. Every row of the
ARCHITECTURE.md panel-row R1-R6 table is exercised, plus retry idempotency and
strategy independence. The construction style mirrors
``tests/test_v2_features_daily_state_inputs.py``.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from engine.v2.contracts.data import DatasetVersionRef, SnapshotRef, TableContractRef
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_TABLE_NAME
from engine.v2.data.errors import DataError
from engine.v2.data.price_history_table import PRICE_HISTORY_CONTRACT, PRICE_HISTORY_TABLE_NAME
from engine.v2.features import panel_math
from engine.v2.features.panel_row_inputs import PanelRowInputs, scan_panel_row
from engine.v2.ops.native_board_universe import BoardRequest
from tests.data_scan_support import contract_for, contract_ref_for, fake_hash

_DM = contract_for("daily_market")
_DM_REF = contract_ref_for(_DM)
_CM_REF = TableContractRef(contract_id=COMPUTED_MOVES_CONTRACT.contract_id,
                           definition_hash=COMPUTED_MOVES_CONTRACT.definition_hash)
_PH_REF = TableContractRef(contract_id=PRICE_HISTORY_CONTRACT.contract_id,
                           definition_hash=PRICE_HISTORY_CONTRACT.definition_hash)

_HISTORY_START = pd.Timestamp("2024-01-02")
_EVENT = pd.Timestamp("2024-02-15")
_DECISION = pd.Timestamp("2024-02-14")

# The three contributing source dates, deliberately all different so the happy
# path exercises ``min`` (not a tie): daily_state 02-05, regime 02-01, runup 01-20.
_DAILY_STATE_SOURCE = "2024-02-05"
_REGIME_SOURCE = "2024-02-01"
_RUNUP_SOURCE = "2024-01-20"


class _Batch:
    """A batch whose rows are handed back exactly as built (no copy)."""

    def __init__(self, rows) -> None:
        self._rows = rows

    def to_pylist(self):
        return self._rows


class _FakeRepository:
    """The ``Repository`` surface the composition needs, dispatched per table/ticker."""

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

    def scan(self, query, *, table_name):
        ticker = query.key_filter[0].values[0]
        yield _Batch(self._batches.get((table_name, ticker), []))


def _problem(message, table_name):
    from engine.v2.data import errors
    return errors.make_problem("CONTRACT_MISMATCH", message, details={"table_name": table_name})


def _snapshot(*, with_daily=True, with_computed=True, with_price=True) -> SnapshotRef:
    versions, modes = {}, {}
    if with_daily:
        versions["daily_market"] = DatasetVersionRef(
            dataset_version_id="dsv-dm", table_contract_ref=_DM_REF,
            manifest_hash=fake_hash("dm-manifest"))
        modes["daily_market"] = "reconstructed"
    if with_computed:
        versions[COMPUTED_MOVES_TABLE_NAME] = DatasetVersionRef(
            dataset_version_id="dsv-cm", table_contract_ref=_CM_REF,
            manifest_hash=fake_hash("cm-manifest"))
        modes[COMPUTED_MOVES_TABLE_NAME] = "observed"
    if with_price:
        versions[PRICE_HISTORY_TABLE_NAME] = DatasetVersionRef(
            dataset_version_id="dsv-ph", table_contract_ref=_PH_REF,
            manifest_hash=fake_hash("ph-manifest"))
        modes[PRICE_HISTORY_TABLE_NAME] = "reconstructed"
    return SnapshotRef(snapshot_id="snap-panel", manifest_hash=fake_hash("panel-snapshot"),
                       table_versions=versions, calendar_version="cal.v1",
                       source_priority_version="prio.v1", finality_receipt_refs=(),
                       knowledge_mode_by_table=modes)


def _contracts():
    return {"daily_market": _DM, COMPUTED_MOVES_TABLE_NAME: COMPUTED_MOVES_CONTRACT,
            PRICE_HISTORY_TABLE_NAME: PRICE_HISTORY_CONTRACT}


def _dm_rows(ticker, days, *, src_iv="orats", implied_move=1.0, close=None):
    rows = []
    for i, day in enumerate(days):
        row = {"ticker": ticker, "date": pd.Timestamp(day), "src_iv": src_iv,
               "implied_move": implied_move + i}
        if close is not None:
            row["close"] = close + i
        rows.append(row)
    return rows


def _computed_rows(rows_spec):
    return [{"ticker": "AAA", "event_date": day, "realized_move_pct": move, "skipped": skipped}
            for day, move, skipped in rows_spec]


def _price_rows(n=320, *, end="2024-01-20"):
    dates = pd.date_range(end=end, periods=n, freq="D")
    return [{"date": stamp.date().isoformat(), "close_adj": 100.0 + i, "close_raw": 100.0 + i,
             "high_raw": 100.0 + i, "retrieved_at": "2024-02-01T00:00:00Z", "deleted": False,
             "source_kind": "yfinance", "source_hash": fake_hash("px")}
            for i, stamp in enumerate(dates)]


def _default_batches():
    return {
        ("daily_market", "AAA"): _dm_rows("AAA", ["2024-02-04", _DAILY_STATE_SOURCE]),
        ("daily_market", "SPY"): _spy_rows(),
        (COMPUTED_MOVES_TABLE_NAME, "AAA"): _computed_rows(
            [("2024-01-10", 2.0, False), ("2024-01-11", -1.0, False)]),
        (PRICE_HISTORY_TABLE_NAME, "AAA"): _price_rows(),
    }


def _spy_rows():
    days = [str(d.date()) for d in pd.date_range(end=_REGIME_SOURCE, periods=20, freq="D")]
    return [{"ticker": "SPY", "date": pd.Timestamp(day), "close": 500.0 + i,
             "src_iv": "orats", "implied_move": 1.0}
            for i, day in enumerate(days)]


def _key(strategy="STR-X"):
    return BoardRequest(ticker="AAA", strategy=strategy, event_date=_EVENT, session="AMC")


def _scan(snapshot=None, batches=None, *, strategy="STR-X"):
    snapshot = _snapshot() if snapshot is None else snapshot
    batches = _default_batches() if batches is None else batches
    repo = _FakeRepository(snapshot, _contracts(), batches)
    return scan_panel_row(repo, snapshot, _key(strategy),
                          history_start=_HISTORY_START, decision_session=_DECISION)


def _nan_equal(left, right):
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    return left == right


def _rows_equal(a, b):
    assert set(a) == set(b)
    assert all(_nan_equal(a[k], b[k]) for k in a)


# --------------------------------------------------------------------------
# R1-R6 table coverage
# --------------------------------------------------------------------------


def test_1_happy_path_all_four_reads_and_min_anchor():
    result = _scan()
    assert isinstance(result, PanelRowInputs)
    panel = result.panel_row
    # One key from each of the four sources.
    assert "im" in panel and panel["im"] == 2.0
    assert "n_prior" in panel and panel["n_prior"] == 2
    assert "spy_ret21" in panel
    assert "signed_streak" in panel
    # The three contributing dates differ; the anchor is the earliest.
    assert result.panel_anchor == pd.Timestamp(_RUNUP_SOURCE)
    # spy_* and runup_* keys survive as plain floats; the ema fallback is the mean.
    assert isinstance(panel["spy_ret21"], float)
    assert isinstance(panel["signed_streak"], float)
    assert panel["ema12r_abs"] == 1.5  # n_prior < 12 -> falls back to mean_prior_abs_move


def test_2_missing_daily_market_or_computed_moves_propagates_contract_mismatch():
    # No computed_moves table.
    with pytest.raises(DataError) as exc:
        _scan(snapshot=_snapshot(with_computed=False))
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"table_name": COMPUTED_MOVES_TABLE_NAME}

    # No daily_market table: scan_daily_state_inputs refuses it.
    with pytest.raises(DataError) as exc:
        _scan(snapshot=_snapshot(with_daily=False))
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"table_name": "daily_market"}


def test_3_missing_price_history_propagates_contract_mismatch():
    with pytest.raises(DataError) as exc:
        _scan(snapshot=_snapshot(with_price=False))
    assert exc.value.code == "CONTRACT_MISMATCH"


def test_4_no_computed_moves_rows_keeps_history_empty():
    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = []
    panel = _scan(batches=batches).panel_row
    assert panel["n_prior"] == 0
    for key in ("mean_prior_move", "mean_prior_abs_move", "ema2_prior_move",
                "ema4_prior_abs_move", "ema8_prior_move", "ema12_prior_abs_move"):
        assert panel[key] is None
    # Independent reads are unaffected.
    assert "im" in panel
    assert "spy_ret21" in panel
    assert "signed_streak" in panel


def test_5_no_spy_rows_gives_nan_regime_features():
    batches = _default_batches()
    batches[("daily_market", "SPY")] = []
    panel = _scan(batches=batches).panel_row
    for col in ("spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol5",
                "spy_vol20", "spy_vol60", "spy_vol252", "spy_vol20_rel252"):
        assert math.isnan(panel[col])
    # Independent reads are unaffected.
    assert panel["n_prior"] == 2
    assert "signed_streak" in panel


def test_6_only_non_skipped_computed_moves_feed_history():
    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = _computed_rows([
        ("2024-01-10", 2.0, False),
        ("2024-01-11", None, True),   # skipped: realized_move_pct always None
        ("2024-01-12", -1.0, False),
        ("2024-01-13", None, True),
    ])
    panel = _scan(batches=batches).panel_row
    assert panel["n_prior"] == 2  # the two non-skipped rows, not all four
    assert panel["mean_prior_move"] == 0.5
    assert panel["mean_prior_abs_move"] == 1.5


def test_7_retry_is_idempotent():
    first = _scan()
    second = _scan()
    assert first.panel_anchor == second.panel_anchor
    _rows_equal(first.panel_row, second.panel_row)


def test_8_all_three_anchors_absent_gives_none():
    snapshot = _snapshot()
    batches = _default_batches()
    # No eligible daily_state row (src_iv absent) -> source_session None.
    batches[("daily_market", "AAA")] = _dm_rows("AAA", ["2024-02-05"], src_iv=None)
    # No SPY history -> regime NaT anchor.
    batches[("daily_market", "SPY")] = []
    # No computed history -> n_prior 0.
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = []
    # Too little price history to resolve a runup anchor -> runup_asof NaT.
    batches[(PRICE_HISTORY_TABLE_NAME, "AAA")] = _price_rows(n=10)
    result = _scan(snapshot=snapshot, batches=batches)
    assert result.panel_anchor is None


def test_9_strategy_never_changes_the_result():
    one = _scan(strategy="STR-X")
    other = _scan(strategy="DYN-SV")
    assert one.panel_anchor == other.panel_anchor
    _rows_equal(one.panel_row, other.panel_row)


def test_10_non_skipped_null_realized_move_raises_contract_mismatch():
    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = _computed_rows([
        ("2024-01-10", None, False),  # repository-integrity violation
    ])
    with pytest.raises(DataError) as exc:
        _scan(batches=batches)
    assert exc.value.code == "CONTRACT_MISMATCH"


def test_history_keys_are_the_panel_math_superset():
    panel = _scan().panel_row
    for key in panel_math.history_features([2.0, -1.0], [2.0, 1.0]):
        assert key in panel
