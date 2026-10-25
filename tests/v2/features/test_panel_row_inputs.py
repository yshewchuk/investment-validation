"""Direct tests for ``engine.v2.features.panel_row_inputs.scan_panel_row``.

One fake repository serves the four pinned-snapshot reads the composition needs
from a single fixture: ``daily_market`` (``key.ticker``), ``computed_moves``,
and ``price_history`` (both the ``REGIME_TICKER`` ``"^GSPC"`` regime series and
``key.ticker``'s run-up series). Every row of the ARCHITECTURE.md panel-row
R1-R6 table is exercised, plus retry idempotency and strategy independence.
The construction style mirrors ``tests/test_v2_features_daily_state_inputs.py``.
"""
from __future__ import annotations

import dataclasses
import math

import pandas as pd
import pytest

from engine.v2.contracts.data import (
    DatasetVersionRef,
    KeyPredicate,
    SnapshotRef,
    TableContractRef,
    TimeInterval,
)
from engine.v2.data.computed_moves import build_rows
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_TABLE_NAME
from engine.v2.data.errors import DataError
from engine.v2.data.price_history_table import (
    PRICE_HISTORY_CONTRACT,
    PRICE_HISTORY_TABLE_NAME,
    REGIME_TICKER,
)
from engine.v2.features import panel_math, panel_row_inputs, regime
from engine.v2.features.panel_row_inputs import PanelRowInputs, scan_panel_row
from engine.v2.ops.native_board_universe import BoardRequest
from tests.data_scan_support import contract_for, contract_ref_for, fake_hash
from tests.test_v2_features_daily_state_inputs import _bound_row_selected

_DM = contract_for("daily_market")
_DM_REF = contract_ref_for(_DM)
_CM_REF = TableContractRef(contract_id=COMPUTED_MOVES_CONTRACT.contract_id,
                           definition_hash=COMPUTED_MOVES_CONTRACT.definition_hash)
_PH_REF = TableContractRef(contract_id=PRICE_HISTORY_CONTRACT.contract_id,
                           definition_hash=PRICE_HISTORY_CONTRACT.definition_hash)

_HISTORY_START = pd.Timestamp("2024-01-02")
_EVENT = pd.Timestamp("2024-02-15")
_DECISION = pd.Timestamp("2024-02-14")

# The three contributing MARKET source dates, deliberately all different so the
# happy path exercises ``max`` (not a tie): daily_state 02-05, regime 02-01,
# runup 01-20; the default kept history's availability (01-12) is earlier still,
# so it never moves the default anchor.
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
        self.queries: list = []

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
        """The real method's metadata-only bound, faked over stored rows: the same
        pinned-snapshot/contract identity checks ``resolve``/``table_contract``
        make, then exactly the rows the eq/in predicates and half-open interval
        select -- the selection ``scan`` below streams for that table."""
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
        self.queries.append(query)
        ticker = query.key_filter[0].values[0]
        rows = self._batches.get((table_name, ticker), [])
        selected = [row for row in rows if _bound_row_selected(
            row, query.key_filter, query.time_interval)]
        yield _Batch(selected)


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
    # available_as_of_date follows the COMPUTED_MOVES_CONTRACT null policy:
    # a non-skipped row is knowable the day after its event, a skipped row
    # (null realized_move_pct) carries a null availability -- never "assume
    # available". Delayed-close fixtures override it with producer-derived
    # dates (see test_19).
    return [{"ticker": "AAA", "event_date": day, "realized_move_pct": move,
             "skipped": skipped,
             "available_as_of_date": None if skipped else
             (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat(),
             "event_ordinal": index, "admitted": False,
             "implied_move_pct": None, "mean_prior_or_implied": None,
             "n_prior_implied": 0}
            for index, (day, move, skipped) in enumerate(rows_spec)]


def _with_implied(row, **overrides):
    return {**row, **overrides}


def _price_rows(n=320, *, end="2024-01-20"):
    dates = pd.date_range(end=end, periods=n, freq="D")
    return [{"date": stamp.date().isoformat(), "close_adj": 100.0 + i, "close_raw": 100.0 + i,
             "high_raw": 100.0 + i, "retrieved_at": "2024-02-01T00:00:00Z", "deleted": False,
             "source_kind": "yfinance", "source_hash": fake_hash("px")}
            for i, stamp in enumerate(dates)]


def _default_batches():
    return {
        ("daily_market", "AAA"): _dm_rows("AAA", ["2024-02-04", _DAILY_STATE_SOURCE]),
        (PRICE_HISTORY_TABLE_NAME, REGIME_TICKER): _gspc_rows(),
        (COMPUTED_MOVES_TABLE_NAME, "AAA"): _computed_rows(
            [("2024-01-10", 2.0, False), ("2024-01-11", -1.0, False)]),
        (PRICE_HISTORY_TABLE_NAME, "AAA"): _price_rows(),
    }


def _gspc_rows(periods=20, *, end=_REGIME_SOURCE):
    """The ``REGIME_TICKER`` ``price_history`` fixture rows: the regime
    series' closes (``close_raw``, 500.0+i on consecutive days ending
    ``end``). ``close_adj`` (900.0+i) and ``high_raw`` (550.0+i) are
    deliberately distinct so a regime read picking the wrong column fails."""
    days = [str(d.date()) for d in pd.date_range(end=end, periods=periods, freq="D")]
    return [{"date": day, "close_adj": 900.0 + i, "close_raw": 500.0 + i,
             "high_raw": 550.0 + i, "retrieved_at": "2024-02-01T00:00:00Z",
             "deleted": False, "source_kind": "legacy_gspc_csv",
             "source_hash": fake_hash("gspc")}
            for i, day in enumerate(days)]


def _gspc_rows_after_decision(periods=3, *, end="2024-02-18"):
    """Rows that exist but resolve no eligible bar at ``_DECISION``: the
    regime series' own no-history input (NaN features, NaT anchor), never a
    missing-ticker refusal. Pass a later ``end`` for a later decision."""
    return _gspc_rows(periods=periods, end=end)


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


def test_1_happy_path_all_four_reads_and_max_anchor():
    # add_regime_features needs j >= 21 (22 rows, anchor at the last) for a
    # non-NaN spy_ret21, so this test widens the ^GSPC batch; every other test
    # keeps the default 20-row fixture.
    gspc = _gspc_rows(periods=22)
    batches = _default_batches()
    batches[(PRICE_HISTORY_TABLE_NAME, REGIME_TICKER)] = gspc
    result = _scan(batches=batches)
    assert isinstance(result, PanelRowInputs)
    panel = result.panel_row
    # One key from each of the four sources.
    assert "im" in panel and panel["im"] == 2.0
    assert "n_prior" in panel and panel["n_prior"] == 2
    assert "spy_ret21" in panel
    assert "signed_streak" in panel
    # The three MARKET source dates differ (the default kept history resolves to
    # the earlier 2024-01-12); the anchor is the latest.
    assert result.panel_anchor == pd.Timestamp(_DAILY_STATE_SOURCE)
    # spy_* and runup_* keys survive as plain floats; the ema fallback is the mean.
    # regime.py's formula, mirrored over the same closes: every ^GSPC date is
    # strictly before both the event and the decision, so the anchor is the
    # last row (j = len - 1) and spy_ret21 = (close / closes[j - 21] - 1) * 100.
    closes = [row["close_raw"] for row in gspc]
    expected_ret21 = (closes[-1] / closes[-1 - 21] - 1.0) * 100
    assert panel["spy_ret21"] == pytest.approx(expected_ret21)
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
    # Rows exist (the read itself succeeds) but none is eligible through the
    # decision: regime's own no-history input, never a missing-ticker refusal.
    batches[(PRICE_HISTORY_TABLE_NAME, REGIME_TICKER)] = _gspc_rows_after_decision()
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


def test_6b_v5_refusal_rows_never_clear_the_stored_implied_mean():
    """A newest v5 refusal row (not_orats) is dropped before the carry step, so the prior
    admitted row's stored mean still feeds the panel row."""
    from engine.v2.data.computed_moves_table import COMPUTED_MOVES_V5_CONTRACT

    v5_ref = TableContractRef(contract_id=COMPUTED_MOVES_V5_CONTRACT.contract_id,
                              definition_hash=COMPUTED_MOVES_V5_CONTRACT.definition_hash)
    snapshot = _snapshot()
    versions = dict(snapshot.table_versions)
    versions[COMPUTED_MOVES_TABLE_NAME] = dataclasses.replace(
        versions[COMPUTED_MOVES_TABLE_NAME], table_contract_ref=v5_ref)
    snapshot = dataclasses.replace(snapshot, table_versions=versions)
    contracts = {**_contracts(), COMPUTED_MOVES_TABLE_NAME: COMPUTED_MOVES_V5_CONTRACT}
    rows = _computed_rows([("2024-01-10", 2.0, False), ("2024-01-11", -1.0, False),
                           ("2024-01-12", None, True)])
    for row in rows:
        row["not_computed_reason"] = None
    for row in rows[:2]:  # admitted rows carry the stored aggregate
        row.update(admitted=True, mean_prior_or_implied=1.5, n_prior_implied=3)
    rows[2]["not_computed_reason"] = "not_orats"
    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = rows
    repo = _FakeRepository(snapshot, contracts, batches)

    panel = scan_panel_row(repo, snapshot, _key(), history_start=_HISTORY_START,
                           decision_session=_DECISION).panel_row

    assert panel["mean_prior_or_implied"] == 1.5


def test_7_retry_is_idempotent():
    first = _scan()
    second = _scan()
    assert first.panel_anchor == second.panel_anchor
    _rows_equal(first.panel_row, second.panel_row)


def test_8_all_contributing_anchors_absent_gives_none():
    snapshot = _snapshot()
    batches = _default_batches()
    # No eligible daily_state row (src_iv absent) -> source_session None.
    batches[("daily_market", "AAA")] = _dm_rows("AAA", ["2024-02-05"], src_iv=None)
    # No eligible ^GSPC row -> regime NaT anchor.
    batches[(PRICE_HISTORY_TABLE_NAME, REGIME_TICKER)] = _gspc_rows_after_decision()
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
    assert exc.value.problem.details == {"table_name": COMPUTED_MOVES_TABLE_NAME}
    assert "realized_move_pct" in exc.value.problem.message


def test_11_computed_moves_rows_are_sorted_before_feeding_history():
    batches = _default_batches()
    # Fed out of order on purpose; the real chronological order is 01-10, 01-11, 01-12.
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = _computed_rows([
        ("2024-01-12", 3.0, False),
        ("2024-01-10", 2.0, False),
        ("2024-01-11", -1.0, False),
    ])
    panel = _scan(batches=batches).panel_row
    expected = panel_math.history_features([2.0, -1.0, 3.0], [2.0, 1.0, 3.0])
    assert panel["ema2_prior_move"] == expected["ema2_prior_move"]
    assert panel["mean_prior_move"] == expected["mean_prior_move"]


def test_history_keys_are_the_panel_math_superset():
    panel = _scan().panel_row
    for key in panel_math.history_features([2.0, -1.0], [2.0, 1.0]):
        assert key in panel


def test_12_computed_moves_capped_at_decision_session_not_just_event_date():
    """A row after decision_session is not "prior" even when it precedes the event."""
    snapshot = _snapshot()
    batches = _default_batches()
    # The scored event moves to 02-20 so a day strictly between it and _DECISION exists.
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = _computed_rows([("2024-02-17", 3.0, False)])

    class _WindowedRepository(_FakeRepository):
        """The shared fake, but honoring the emitted window for ``computed_moves``
        (which the base fake ignores, returning every row for the ticker)."""

        def scan(self, query, *, table_name, max_object_bytes=None):
            rows = self._batches.get((table_name, query.key_filter[0].values[0]), [])
            interval = query.time_interval
            if table_name == COMPUTED_MOVES_TABLE_NAME and interval is not None:
                start = pd.Timestamp(interval.start_inclusive)
                end = pd.Timestamp(interval.end_exclusive)
                rows = [row for row in rows
                        if start <= pd.Timestamp(row[interval.column]) < end]
            yield _Batch(rows)

    def _panel_at(decision_session):
        repo = _WindowedRepository(snapshot, _contracts(), batches)
        key = BoardRequest(ticker="AAA", strategy="STR-X",
                           event_date=pd.Timestamp("2024-02-20"), session="AMC")
        return scan_panel_row(repo, snapshot, key, history_start=_HISTORY_START,
                              decision_session=decision_session).panel_row

    assert _panel_at(_DECISION)["n_prior"] == 0
    # Excluded by the cap, not absent: five sessions later (still before the
    # event) the same row is ordinary prior history.
    assert _panel_at(_DECISION + pd.Timedelta(days=5))["n_prior"] == 1


def test_13_signed_streak_reflects_prior_computed_moves():
    """Three same-signed prior moves put the current event at streak length 3.

    Before the multi-row-frame fix, _runup_from_prices fed add_runup_features a
    one-row frame whose group row 0 always resets the streak, so this value was
    0.0 unconditionally regardless of input."""
    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = _computed_rows([
        ("2024-02-11", 1.0, False),
        ("2024-02-12", 2.0, False),
        ("2024-02-13", 3.0, False),
    ])
    panel = _scan(batches=batches).panel_row
    assert panel["signed_streak"] == 3.0


def test_14_regime_features_read_price_history_closes_not_daily_market_spot():
    """The nine ``spy_*`` features come from the ``REGIME_TICKER``
    ``price_history`` closes -- never a ``daily_market`` spot.

    The fixture seeds ``daily_market`` SPY rows with wildly different
    (quadratic) spot values alongside the default linear ^GSPC closes: the
    panel's ``spy_*`` values must equal ``regime.add_regime_features`` over
    the closes and must differ from the same computation over the spots.
    After the cutover the SPY ``daily_market`` rows are read by nobody on
    this path -- they exist only to make a spot-based computation observable
    and unmistakably wrong."""
    batches = _default_batches()
    gspc_fixture = [dict(row) for row in batches[(PRICE_HISTORY_TABLE_NAME, REGIME_TICKER)]]
    spot_rows = [{"ticker": "SPY", "date": pd.Timestamp(row["date"]),
                  "spot": 400.0 + i * i, "src_iv": "orats", "implied_move": 1.0}
                 for i, row in enumerate(gspc_fixture)]
    batches[("daily_market", "SPY")] = spot_rows
    panel = _scan(batches=batches).panel_row

    events = pd.DataFrame({"date": [_EVENT], "decision": [_DECISION]})
    expected = regime.add_regime_features(
        events,
        pd.DataFrame({"date": pd.to_datetime([row["date"] for row in gspc_fixture]),
                      "close": [float(row["close_raw"]) for row in gspc_fixture]}),
        as_of_column="decision").iloc[0]
    spot_based = regime.add_regime_features(
        events,
        pd.DataFrame({"date": pd.to_datetime([row["date"] for row in spot_rows]),
                      "close": [float(row["spot"]) for row in spot_rows]}),
        as_of_column="decision").iloc[0]
    literal_regime_names = ("spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol5",
                            "spy_vol20", "spy_vol60", "spy_vol252", "spy_vol20_rel252")
    for column in literal_regime_names:
        value, other = panel[column], spot_based[column]
        if math.isnan(value):
            assert math.isnan(expected[column])
        else:
            assert value == pytest.approx(expected[column])
            if not math.isnan(other):
                assert value != pytest.approx(other, rel=1e-6)


def test_15_null_regime_close_raises_contract_mismatch():
    batches = _default_batches()
    rows = _gspc_rows()
    rows[3] = {**rows[3], "close_raw": None}  # a null close conversion refuses
    batches[(PRICE_HISTORY_TABLE_NAME, REGIME_TICKER)] = rows
    with pytest.raises(DataError) as exc:
        _scan(batches=batches)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert "price_history value conversion failed" in str(exc.value)
    assert exc.value.problem.details == {"table_name": PRICE_HISTORY_TABLE_NAME}


@pytest.mark.parametrize("close_raw", [0.0, -4700.0, float("inf")])
def test_15b_nonpositive_or_nonfinite_regime_close_raises_contract_mismatch(close_raw):
    """The regime read mirrors the research path's positive-close check: a
    stored ``close_raw`` that is zero, negative or non-finite refuses before
    any ``regime.add_regime_features`` arithmetic runs. (A NaN close never
    reaches this guard: ``get_price_series`` nulls it, so it refuses as the
    null-conversion case of ``test_15``.)"""
    batches = _default_batches()
    rows = _gspc_rows()
    rows[3] = {**rows[3], "close_raw": close_raw}
    batches[(PRICE_HISTORY_TABLE_NAME, REGIME_TICKER)] = rows
    with pytest.raises(DataError) as exc:
        _scan(batches=batches)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert "regime source closes must be positive" in str(exc.value)
    assert exc.value.problem.details == {"table_name": PRICE_HISTORY_TABLE_NAME}


def test_15c_regime_price_query_bounds_pinned_to_decision(monkeypatch):
    """The regime read is exactly one ``get_price_series`` of the pinned
    ``REGIME_TICKER`` as of ``decision_session``: the ``PriceQuery`` bounds
    (ticker, session_date/observation_ceiling, ``lookback_sessions=0``) are
    asserted, not just the values they return, and the read is byte-bounded by
    the module's own cap."""
    from engine.v2.data import price_history_query

    calls: list = []
    real = price_history_query.get_price_series

    def recording(repository, query, snapshot, **kwargs):
        calls.append((query, kwargs))
        return real(repository, query, snapshot, **kwargs)

    monkeypatch.setattr(price_history_query, "get_price_series", recording)
    _scan()
    regime_calls = [(query, kwargs) for query, kwargs in calls if query.ticker == REGIME_TICKER]
    assert len(regime_calls) == 1
    query, kwargs = regime_calls[0]
    assert kwargs == {"max_object_bytes": panel_row_inputs._REGIME_MAX_OBJECT_BYTES}
    assert query.session_date == _DECISION.date().isoformat()
    assert query.observation_ceiling == _DECISION.date().isoformat()
    assert query.lookback_sessions == 0


def test_16_zero_day_computed_moves_window_returns_empty_not_a_query_error():
    """``history_start == min(event_date, decision)`` means a zero-row window.

    ``_window`` only requires ``history_start <= decision_session``, so a caller
    may pass them equal; with ``_DECISION < _EVENT`` that makes
    ``_read_computed_moves``'s ``start_inclusive == end_exclusive``. The real
    repository's ``DataQuery`` validation (``documents._check_time_interval``,
    reached through ``Repository.scan`` -> ``decode_document``) rejects equal
    bounds with TIME_BOUNDS_OUT_OF_ORDER, but the in-memory ``_FakeRepository``
    skips encode/decode validation and so cannot reproduce that. This pins the
    scan_panel_row-level contract -- an empty history, never a raise -- for that
    zero-day window. The default ``daily_market`` rows (02-04/02-05) would fall
    outside a 02-14 start and trip daily_state_inputs' own window check first,
    so one in-window row is supplied to reach the computed_moves read."""
    batches = _default_batches()
    batches[("daily_market", "AAA")] = _dm_rows("AAA", ["2024-02-14"])
    snapshot = _snapshot()
    repo = _BoundRecordingRepository(snapshot, _contracts(), batches)
    result = scan_panel_row(repo, snapshot, _key(),
                            history_start=_DECISION, decision_session=_DECISION)
    assert result.panel_row["n_prior"] == 0
    # The zero-day window returns before any bound planning or scan: no
    # computed_moves bound call and no computed_moves query is ever emitted.
    assert not [call for call in repo.bound_calls
                if call["table_name"] == COMPUTED_MOVES_TABLE_NAME]
    assert not [query for table, query in repo.scans
                if table == COMPUTED_MOVES_TABLE_NAME]


def test_17_zero_day_window_still_validates_missing_computed_moves_table():
    """The zero-day short-circuit skips only the query, never the validation."""
    batches = _default_batches()
    # One in-window daily_market row so the read reaches _read_computed_moves.
    batches[("daily_market", "AAA")] = _dm_rows("AAA", ["2024-02-14"])
    snapshot = _snapshot(with_computed=False)
    repo = _FakeRepository(snapshot, _contracts(), batches)
    with pytest.raises(DataError) as exc:
        scan_panel_row(repo, snapshot, _key(),
                       history_start=_DECISION, decision_session=_DECISION)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"table_name": COMPUTED_MOVES_TABLE_NAME}


def test_18_panel_anchor_is_the_latest_not_the_earliest_source_date():
    # Vary the price history so runup resolves to 2024-02-13, the latest
    # contributing date (daily_state is 2024-02-05, regime 2024-02-01). A
    # freshness cutoff strictly between the earliest and latest (e.g.
    # 2024-02-07) therefore exists. The anchor must be the latest contributing
    # date -- under the old `min` behavior this exact fixture returned the
    # earliest (2024-02-01), which would wrongly let a 02-07 cutoff pass
    # despite the 02-13 price data, the defect this test guards against.
    batches = _default_batches()
    batches[(PRICE_HISTORY_TABLE_NAME, "AAA")] = _price_rows(end="2024-02-13")
    result = _scan(batches=batches)
    assert result.panel_anchor == pd.Timestamp("2024-02-13")
    assert result.panel_anchor != pd.Timestamp(_REGIME_SOURCE)


# --------------------------------------------------------------------------
# available_as_of_date eligibility
# --------------------------------------------------------------------------


class _WindowedComputedRepository(_FakeRepository):
    """The shared fake, but honoring the emitted event_date window for
    ``computed_moves`` (like test_12's local variant), so the availability
    filter -- not the fake -- is what the eligibility tests exercise."""

    def scan(self, query, *, table_name, max_object_bytes=None):
        rows = self._batches.get((table_name, query.key_filter[0].values[0]), [])
        interval = query.time_interval
        if table_name == COMPUTED_MOVES_TABLE_NAME and interval is not None:
            start = pd.Timestamp(interval.start_inclusive)
            end = pd.Timestamp(interval.end_exclusive)
            rows = [row for row in rows
                    if start <= pd.Timestamp(row[interval.column]) < end]
        yield _Batch(rows)


def test_19_delayed_close_amc_event_enters_history_only_at_producer_derived_availability():
    """A real delayed close: a Tuesday 2024-02-13 AMC event whose measured
    close slides to Thursday (Wednesday's session is missing) is knowable
    only on Friday 2024-02-16.

    The availability dates and moves are not hand-written fixtures -- they
    are derived from the actual producer, ``build_rows``. The expected panel
    values are hand-computed independently (never via the scanner): at the
    Wednesday 02-14 decision the delayed row is inside the event_date window
    but not yet knowable, so history and streak carry only the ordinary
    +5.0 prior; at a Friday 02-16 decision (availability date == decision,
    equality enters) both priors count."""
    closes = {
        "2024-01-09": 100.0, "2024-01-10": 100.0, "2024-01-11": 105.0,
        "2024-01-12": 105.0, "2024-02-07": 100.0, "2024-02-08": 100.0,
        "2024-02-09": 100.0, "2024-02-12": 100.0, "2024-02-13": 100.0,
        # No Wednesday 2024-02-14 session: the Tuesday AMC print's measured
        # close is Thursday 2024-02-15, so availability is 2024-02-16.
        "2024-02-15": 110.0,
    }
    days = sorted(closes)
    events = pd.DataFrame({
        "event_date": pd.to_datetime(["2024-01-10", "2024-01-11", "2024-02-08",
                                      "2024-02-09", "2024-02-13"]),
        "session": ["BMO", "BMO", "BMO", "BMO", "AMC"],
    })
    built = build_rows(
        "AAA", events, pd.to_datetime(days).to_numpy(),
        [closes[day] for day in days],
        pd.DataFrame({"date": pd.to_datetime(["2024-01-08"]), "implied_move": [1.0],
                      "src_iv": ["orats"]}),
        computed_at="2024-03-01T00:00:00+00:00",
        source_hash=fake_hash("cm-producer"), capture_id="cap-1")
    by_date = {row["event_date"]: row for row in built}

    delayed, ordinary = by_date["2024-02-13"], by_date["2024-01-11"]
    assert delayed["available_as_of_date"] == "2024-02-16"
    assert delayed["realized_move_pct"] == pytest.approx(10.0)
    assert ordinary["available_as_of_date"] == "2024-01-12"
    assert ordinary["realized_move_pct"] == pytest.approx(5.0)

    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [ordinary, delayed]
    snapshot = _snapshot()

    def _panel_at(decision_session):
        repo = _WindowedComputedRepository(snapshot, _contracts(), batches)
        key = BoardRequest(ticker="AAA", strategy="STR-X",
                           event_date=pd.Timestamp("2024-02-20"), session="AMC")
        return scan_panel_row(repo, snapshot, key, history_start=_HISTORY_START,
                              decision_session=decision_session).panel_row

    panel = _panel_at(_DECISION)
    assert panel["n_prior"] == 1
    assert panel["mean_prior_move"] == pytest.approx(5.0)
    assert panel["mean_prior_abs_move"] == pytest.approx(5.0)
    assert panel["signed_streak"] == 1.0

    panel = _panel_at(pd.Timestamp("2024-02-16"))
    assert panel["n_prior"] == 2
    assert panel["mean_prior_move"] == pytest.approx(7.5)
    assert panel["signed_streak"] == 2.0


def test_20_late_computed_at_is_irrelevant_when_availability_is_by_decision():
    """Backfill negative control: a prior row computed long after the
    decision, but available by it, stays eligible -- eligibility is the
    availability date, never ``computed_at`` (which production does not even
    request)."""
    batches = _default_batches()
    row = {**_computed_rows([("2024-01-10", 2.0, False)])[0],
           "computed_at": "2025-06-01T00:00:00+00:00"}
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [row]
    panel = _scan(batches=batches).panel_row
    assert panel["n_prior"] == 1
    assert panel["mean_prior_move"] == pytest.approx(2.0)
    assert panel["signed_streak"] == 1.0


def test_21_null_available_as_of_date_excludes_a_numeric_prior_move():
    """A non-skipped row with a numeric realized move but a NULL
    availability date is unavailable to every decision -- never fabricated
    into "always available"."""
    batches = _default_batches()
    row = {**_computed_rows([("2024-01-10", 2.0, False)])[0],
           "available_as_of_date": None}
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [row]
    panel = _scan(batches=batches).panel_row
    assert panel["n_prior"] == 0
    assert panel["mean_prior_move"] is None
    assert panel["signed_streak"] == 0.0


def test_22_computed_projection_names_available_as_of_date_and_only_real_columns():
    """The requested computed_moves projection asks for availability and
    names only real contract columns -- the permissive fake never validates
    column names against the contract, and ``computed_at`` must stay
    unrequested."""
    from engine.v2.features import panel_row_inputs
    assert "available_as_of_date" in panel_row_inputs._COMPUTED_COLUMNS
    assert "computed_at" not in panel_row_inputs._COMPUTED_COLUMNS
    real_columns = {column.name for column in COMPUTED_MOVES_CONTRACT.columns}
    assert set(panel_row_inputs._COMPUTED_COLUMNS) <= real_columns


def test_22b_computed_moves_emitted_projection_is_the_pinned_literal_tuple():
    """The DataQuery the scan actually emits for ``computed_moves`` projects
    exactly these columns, in this order -- the permissive ``_FakeRepository``
    never validates column names against the contract, so a rename or reorder
    would otherwise pass every other test in this file."""
    snapshot = _snapshot()
    repo = _FakeRepository(snapshot, _contracts(), _default_batches())
    scan_panel_row(repo, snapshot, _key(), history_start=_HISTORY_START,
                   decision_session=_DECISION)
    computed = [query for query in repo.queries
                if "realized_move_pct" in query.columns]
    assert len(computed) == 1
    query = computed[0]
    assert query.key_filter[0].values == ("AAA",)
    assert query.columns == (
        "ticker", "event_date", "realized_move_pct", "skipped",
        "available_as_of_date", "event_ordinal", "admitted", "implied_move_pct",
        "mean_prior_or_implied", "n_prior_implied")


# --------------------------------------------------------------------------
# staged-input presence contract (nightly_source_bundle)
# --------------------------------------------------------------------------


def test_23_panel_row_date_is_the_scored_event_and_satisfies_the_presence_guard():
    """The assembled panel_row carries the scored event's ISO calendar date.

    Presence-contract regression for the real consumer guard,
    ``nightly_source_bundle._require_staged_inputs_present``, which refuses
    ``MISSING_STAGED_INPUT`` on a panel_row without its ``date`` key. The
    expected date is independent of scanner output, and must be the event
    date -- never the decision session or any contributing source anchor
    (the mixed-date fixture puts those at 02-14 / 02-05)."""
    from engine.v2.scoring.nightly_source_bundle import (
        NightlySourceBundleRefusal,
        _CALENDAR_REQUIRED_FIELDS,
        _require_staged_inputs_present,
    )

    result = _scan()
    panel = result.panel_row
    assert panel["date"] == _EVENT.date().isoformat() == "2024-02-15"
    assert panel["date"] != _DECISION.date().isoformat()
    assert panel["date"] != result.panel_anchor.date().isoformat()

    calendar_row = {field: None for field in _CALENDAR_REQUIRED_FIELDS}
    _require_staged_inputs_present(calendar_row, panel, {}, [])

    without_date = {name: value for name, value in panel.items() if name != "date"}
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        _require_staged_inputs_present(calendar_row, without_date, {}, [])
    assert exc.value.code == "MISSING_STAGED_INPUT"
    assert "'date'" in exc.value.detail


# --------------------------------------------------------------------------
# event_date normalization at the scan boundary
# --------------------------------------------------------------------------


def test_24_event_day_decision_reads_identically_for_midnight_and_intraday_event_keys():
    """A 10:30 ``key.event_date`` names the same calendar event day as midnight,
    so on an event-day decision both must produce the SAME row -- and neither
    may observe the event day's own close.

    Board enumeration legitimately preserves the intraday event clock
    (``native_board_universe``), so this is a real key shape, not a malformed
    one. The fixture spikes the event day's own ^GSPC close and adjusted close so
    that an unnormalized key -- whose ``_anchor_index`` bound would be the
    event-day timestamp, selecting the final bar -- is unmistakably wrong
    instead of merely different. Both regime and runup math run for real here;
    neither is mocked.
    """
    event_day_decision = pd.Timestamp("2024-02-15")

    class _WindowedDailyMarketRepository(_FakeRepository):
        """The shared fake, but honoring the emitted half-open window on the
        ``daily_market`` scan (which the base fake ignores, returning every
        stored row for the ticker). Key/table selection and every fixture
        value are otherwise unchanged."""

        def scan(self, query, *, table_name, max_object_bytes=None):
            rows = self._batches.get((table_name, query.key_filter[0].values[0]), [])
            interval = query.time_interval
            if table_name == "daily_market" and interval is not None:
                start = pd.Timestamp(interval.start_inclusive)
                end = pd.Timestamp(interval.end_exclusive)
                rows = [row for row in rows
                        if start <= pd.Timestamp(row[interval.column]) < end]
            yield _Batch(rows)

    def _scan_with(event_date):
        batches = _default_batches()
        gspc = _gspc_rows(periods=320, end="2024-02-15")
        gspc[-1] = {**gspc[-1], "close_raw": 1500.0,
                    "close_adj": 1500.0}  # the event day's own close, spiked
        batches[(PRICE_HISTORY_TABLE_NAME, REGIME_TICKER)] = gspc
        prices = _price_rows(n=320, end="2024-02-15")
        prices[-1] = {**prices[-1], "close_adj": 4000.0}  # the event day's own close, spiked
        batches[(PRICE_HISTORY_TABLE_NAME, "AAA")] = prices
        snapshot = _snapshot()
        repo = _WindowedDailyMarketRepository(snapshot, _contracts(), batches)
        key = BoardRequest(ticker="AAA", strategy="STR-X", event_date=event_date, session="AMC")
        return scan_panel_row(repo, snapshot, key, history_start=_HISTORY_START,
                              decision_session=event_day_decision)

    midnight = _scan_with(pd.Timestamp("2024-02-15"))
    intraday = _scan_with(pd.Timestamp("2024-02-15T10:30:00"))

    assert midnight.panel_row["date"] == "2024-02-15"
    assert intraday.panel_row["date"] == "2024-02-15"
    _rows_equal(midnight.panel_row, intraday.panel_row)
    assert midnight.panel_anchor == intraday.panel_anchor == pd.Timestamp("2024-02-14")
    # The daily-state source is deliberately the older 2024-02-05, so the anchor
    # above is the two normalized event-day-14 anchors, never the event itself.

    panel = intraday.panel_row
    # The anchored bar is the PRIOR day: ^GSPC index 318 is 818 vs index 297's 797;
    # price index 318 is 418 vs index 313's 413 (the fixture starts at 100.0+i).
    assert panel["spy_ret21"] == pytest.approx((818.0 / 797.0 - 1.0) * 100)
    assert panel["ret5"] == pytest.approx((418.0 / 413.0 - 1.0) * 100)
    # Negative control: selecting the spiked final bar of either series gives
    # completely different values, so neither assertion passes by coincidence.
    assert panel["spy_ret21"] != pytest.approx((1500.0 / 798.0 - 1.0) * 100)
    assert panel["ret5"] != pytest.approx((4000.0 / 414.0 - 1.0) * 100)


class _NoReadRepository(_FakeRepository):
    """A fake on which the first contract lookup or scan IS the test failure:
    it proves a key-level refusal happens before any repository access."""

    def table_contract(self, snapshot_ref, table_name):
        raise AssertionError(f"event_date refusal must precede the {table_name} contract lookup")

    def scan_population_bound(self, snapshot_id, *, table_name, table_contract_ref,
                              key_filter=(), time_interval=None):
        raise AssertionError(f"event_date refusal must precede the {table_name} population bound")

    def scan(self, query, *, table_name, max_object_bytes=None):
        raise AssertionError(f"event_date refusal must precede the {table_name} scan")


@pytest.mark.parametrize("event_date", [
    pd.NaT,
    None,
    "not-a-date",
    20240215,
    True,
    pd.Timestamp("2024-02-15T10:30:00Z"),
])
def test_25_invalid_directly_constructed_event_date_refuses_before_any_read(event_date):
    """``board_requests`` validates its own event dates, but a directly
    constructed ``BoardRequest`` bypasses that; the scan boundary is what
    refuses numeric, invalid, missing and timezone-aware event dates -- before
    any read, with the field named."""
    snapshot = _snapshot()
    repo = _NoReadRepository(snapshot, _contracts(), {})
    key = BoardRequest(ticker="AAA", strategy="STR-X", event_date=event_date, session="AMC")
    with pytest.raises(DataError) as exc:
        scan_panel_row(repo, snapshot, key, history_start=_HISTORY_START,
                       decision_session=_DECISION)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"field": "event_date"}


def test_26_intraday_normalization_is_event_dates_alone():
    """A valid naive intraday event key still scans; every other date field
    keeps refusing any clock time exactly as before."""
    snapshot = _snapshot()
    key = BoardRequest(ticker="AAA", strategy="STR-X",
                       event_date=pd.Timestamp("2024-02-15T10:30:00"), session="AMC")
    repo = _FakeRepository(snapshot, _contracts(), _default_batches())
    result = scan_panel_row(repo, snapshot, key, history_start=_HISTORY_START,
                            decision_session=_DECISION)
    assert result.panel_row["date"] == "2024-02-15"

    refusals = [
        ("history_start", {"history_start": pd.Timestamp("2024-01-02T09:30:00"),
                           "decision_session": _DECISION}),
        ("decision_session", {"history_start": _HISTORY_START,
                              "decision_session": pd.Timestamp("2024-02-14T16:00:00")}),
    ]
    for field, window in refusals:
        repo = _FakeRepository(snapshot, _contracts(), _default_batches())
        with pytest.raises(DataError) as exc:
            scan_panel_row(repo, snapshot, _key(), **window)
        assert exc.value.code == "CONTRACT_MISMATCH"
        assert exc.value.problem.details == {"field": field}

    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [
        {**_computed_rows([("2024-01-10", 2.0, False)])[0],
         "available_as_of_date": "2024-01-11T09:00:00"}]
    with pytest.raises(DataError) as exc:
        _scan(batches=batches)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {"field": "available_as_of_date"}


def test_27_negative_control_delayed_close_availability_bypass_is_detected(monkeypatch):
    """Availability-regression control on test_19's teeth: bypass ONLY the
    outcome availability comparison and its one-prior assertion fails on the
    spot.

    ``test_19`` is called unchanged, so nothing here weakens it. With
    ``available_as_of_date`` validation made to return a date at or before every
    decision, the delayed 2024-02-13 close looks knowable at the 2024-02-14
    decision, both priors feed history, and ``assert panel["n_prior"] == 1``
    raises ``assert 2 == 1``. That observed two-versus-one AssertionError proves
    the availability regression detects a bypassed comparison. No actual
    mutation tooling ran; production filtering is untouched and the patch is
    confined to this test."""
    from engine.v2.features import panel_row_inputs

    original = panel_row_inputs._session_day

    def _bypass(value, field, **kwargs):
        if field == "available_as_of_date":
            return _HISTORY_START
        return original(value, field, **kwargs)

    monkeypatch.setattr(panel_row_inputs, "_session_day", _bypass)
    with pytest.raises(AssertionError, match=r"assert 2 == 1"):
        test_19_delayed_close_amc_event_enters_history_only_at_producer_derived_availability()


# --------------------------------------------------------------------------
# panel_anchor: used-history outcome availability as a contributing date
# --------------------------------------------------------------------------


def _used_history_row():
    """One kept prior move whose outcome became available late (2024-02-19) yet
    still on or before the 2024-02-20 decision, with a legitimate far-later
    computed_at write (2024-03-01) that must never become an anchor."""
    return {**_computed_rows([("2024-02-12", 2.0, False)])[0],
            "available_as_of_date": "2024-02-19",
            "computed_at": "2024-03-01T00:00:00Z"}


def _scan_at(batches, *, decision, event):
    snapshot = _snapshot()
    repo = _WindowedComputedRepository(snapshot, _contracts(), batches)
    key = BoardRequest(ticker="AAA", strategy="STR-X", event_date=event, session="AMC")
    return scan_panel_row(repo, snapshot, key, history_start=_HISTORY_START,
                          decision_session=decision)


@pytest.mark.parametrize("price_count", [320, 10])
def test_28_history_availability_is_latest_contributor(price_count):
    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [_used_history_row()]
    batches[(PRICE_HISTORY_TABLE_NAME, "AAA")] = _price_rows(n=price_count)
    result = _scan_at(batches, decision=pd.Timestamp("2024-02-20"),
                      event=pd.Timestamp("2024-02-21"))
    panel = result.panel_row
    assert result.panel_anchor == pd.Timestamp("2024-02-19")
    assert result.panel_anchor <= pd.Timestamp("2024-02-20")
    for market_source in (_DAILY_STATE_SOURCE, _REGIME_SOURCE, _RUNUP_SOURCE):
        assert result.panel_anchor > pd.Timestamp(market_source)
    # A 2024-02-10 freshness cutoff would have accepted the old max market
    # anchor (2024-02-05); the used history's availability now outranks it.
    assert result.panel_anchor > pd.Timestamp("2024-02-10") >= pd.Timestamp(_DAILY_STATE_SOURCE)
    assert panel["n_prior"] == 1
    assert panel["mean_prior_move"] == pytest.approx(2.0)
    assert panel["signed_streak"] == 1.0
    assert panel["ema12r_abs"] == 2.0
    assert panel["date"] == "2024-02-21"
    if price_count == 10:
        assert math.isnan(panel["ret5"])


@pytest.mark.parametrize("case", [
    "empty_history",
    "numeric_move_null_availability",
    "numeric_move_availability_after_decision",
    "skipped_row_with_availability",
])
def test_29_noncontributing_computed_rows_do_not_advance_anchor(case):
    batches = _default_batches()
    if case == "empty_history":
        history_rows = []
    elif case == "numeric_move_null_availability":
        history_rows = [{**_computed_rows([("2024-02-12", 2.0, False)])[0],
                         "available_as_of_date": None}]
    elif case == "numeric_move_availability_after_decision":
        history_rows = [{**_computed_rows([("2024-02-12", 2.0, False)])[0],
                         "available_as_of_date": "2024-02-21"}]
    else:
        # Eligible to the availability scan yet deliberately unused by the
        # arithmetic: the bound must come from kept_moves, not all scanned rows.
        history_rows = [{**_computed_rows([("2024-02-12", None, True)])[0],
                         "available_as_of_date": "2024-02-19"}]
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = history_rows
    result = _scan_at(batches, decision=pd.Timestamp("2024-02-20"),
                      event=pd.Timestamp("2024-02-21"))
    panel = result.panel_row
    assert panel["n_prior"] == 0
    assert panel["signed_streak"] == 0.0
    assert result.panel_anchor == pd.Timestamp(_DAILY_STATE_SOURCE)


def test_30_used_history_alone_resolves_anchor():
    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [_used_history_row()]
    # Every market contributor's own date is unresolvable -- the daily-state
    # query returns only rows without an eligible src_iv, no ^GSPC row is
    # eligible through the 2024-02-20 decision, and ten price rows cannot
    # resolve a runup anchor -- yet the queries still have real rows where
    # required, so no empty-source refusal happens.
    batches[("daily_market", "AAA")] = _dm_rows("AAA", ["2024-02-05"], src_iv=None)
    batches[(PRICE_HISTORY_TABLE_NAME, REGIME_TICKER)] = \
        _gspc_rows_after_decision(end="2024-02-25")
    batches[(PRICE_HISTORY_TABLE_NAME, "AAA")] = _price_rows(n=10)
    result = _scan_at(batches, decision=pd.Timestamp("2024-02-20"),
                      event=pd.Timestamp("2024-02-21"))
    panel = result.panel_row
    assert result.panel_anchor == pd.Timestamp("2024-02-19")
    assert panel["n_prior"] == 1
    assert panel["signed_streak"] == 1.0
    assert panel["ema12r_abs"] == 2.0
    assert math.isnan(panel["ret5"])


# --------------------------------------------------------------------------
# shared-bound preparation (scan_population_bound -> emitted query limits)
# --------------------------------------------------------------------------


class _BoundRecordingRepository(_FakeRepository):
    """The shared fake, recording every bound-plan call and emitted scan,
    with a selected bound injected per table. ``scan`` yields the full
    selected fixture rows without clipping to the query's limit, so the
    returned-row assertions are real checks. Tables absent from ``bounds``
    keep the base fake's row-count bound, so unrelated reads are
    unaffected."""

    def __init__(self, snapshot, contracts, batches, bounds=None) -> None:
        super().__init__(snapshot, contracts, batches)
        self._bounds = dict(bounds or {})
        self.bound_calls: list[dict] = []
        self.scans: list[tuple] = []

    def scan_population_bound(self, snapshot_id, *, table_name, table_contract_ref,
                              key_filter=(), time_interval=None):
        self.bound_calls.append({"snapshot_id": snapshot_id, "table_name": table_name,
                                 "table_contract_ref": table_contract_ref,
                                 "key_filter": key_filter, "time_interval": time_interval})
        if table_name in self._bounds:
            return self._bounds[table_name]
        return super().scan_population_bound(snapshot_id, table_name=table_name,
                                             table_contract_ref=table_contract_ref,
                                             key_filter=key_filter,
                                             time_interval=time_interval)

    def scan(self, query, *, table_name, max_object_bytes=None):
        self.scans.append((table_name, query))
        rows = self._batches.get((table_name, query.key_filter[0].values[0]), [])
        selected = [row for row in rows if _bound_row_selected(
            row, query.key_filter, query.time_interval)]
        yield _Batch(selected)


def _bound_reads(bounds):
    """Both bounded reads through the recording fake on the default fixture."""
    snapshot = _snapshot()
    batches = _default_batches()
    cm_fixture = batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")]
    regime_fixture = batches[(PRICE_HISTORY_TABLE_NAME, REGIME_TICKER)]
    repo = _BoundRecordingRepository(snapshot, _contracts(), batches, bounds)
    computed = panel_row_inputs._read_computed_moves(repo, snapshot, "AAA",
                                                     _HISTORY_START, _DECISION, _EVENT)
    regime_rows = panel_row_inputs._read_regime_prices(repo, snapshot,
                                                       _HISTORY_START, _DECISION)
    return computed, regime_rows, repo, cm_fixture, regime_fixture


def _assert_bound_matches_query(call, table_name, expected_ref, query):
    """The bound plan saw exactly the identity the read's DataQuery carries."""
    assert call["snapshot_id"] == query.snapshot_id == "snap-panel"
    assert call["table_name"] == table_name
    assert call["table_contract_ref"] == expected_ref == query.table_contract_ref
    assert call["key_filter"] == query.key_filter
    assert call["time_interval"] == query.time_interval


def test_31_bound_plan_receives_the_exact_query_identity_for_both_reads():
    _, _, repo, _, _ = _bound_reads(
        {COMPUTED_MOVES_TABLE_NAME: 5, PRICE_HISTORY_TABLE_NAME: 25})
    assert [call["table_name"] for call in repo.bound_calls] == [
        COMPUTED_MOVES_TABLE_NAME, PRICE_HISTORY_TABLE_NAME]
    (cm_table, cm_query), (reg_table, reg_query) = repo.scans
    assert (cm_table, reg_table) == (COMPUTED_MOVES_TABLE_NAME, PRICE_HISTORY_TABLE_NAME)
    _assert_bound_matches_query(repo.bound_calls[0], COMPUTED_MOVES_TABLE_NAME, _CM_REF, cm_query)
    _assert_bound_matches_query(repo.bound_calls[1], PRICE_HISTORY_TABLE_NAME,
                                _PH_REF, reg_query)
    # And the exact literals, not just internal consistency.
    assert repo.bound_calls[0]["key_filter"] == (
        KeyPredicate(column="ticker", operator="eq", values=("AAA",)),)
    assert repo.bound_calls[1]["key_filter"] == (
        KeyPredicate(column="ticker", operator="eq", values=(REGIME_TICKER,)),)
    assert repo.bound_calls[0]["time_interval"] == TimeInterval(
        column="event_date", start_inclusive="2024-01-02",
        end_exclusive=_DECISION.date().isoformat())
    # The regime read is a plain get_price_series: pinned membership bound,
    # no time interval -- the session/window bounds are its own.
    assert repo.bound_calls[1]["time_interval"] is None


def test_32_positive_selected_bound_lowers_query_limits_and_rows_match_fixtures():
    cm_count = len(_computed_rows([("2024-01-10", 2.0, False),
                                   ("2024-01-11", -1.0, False)]))
    regime_count = len(_gspc_rows())
    computed, regime_rows, repo, cm_fixture, regime_fixture = _bound_reads(
        {COMPUTED_MOVES_TABLE_NAME: cm_count, PRICE_HISTORY_TABLE_NAME: regime_count})
    (_cm, cm_query), (_reg, reg_query) = repo.scans
    cases = ((cm_query, len(cm_fixture)), (reg_query, len(regime_fixture)))
    for query, bound in cases:
        assert 0 < bound < panel_row_inputs._RESULT_LIMIT
        assert query.max_result_rows == bound
        assert 0 < query.max_batch_rows <= bound
    assert len(computed) <= cm_query.max_result_rows
    assert computed == cm_fixture  # both fixture rows, in fixture order
    assert len(regime_rows) <= reg_query.max_result_rows
    assert [(row.date, row.close_raw) for row in regime_rows] == \
        [(row["date"], row["close_raw"]) for row in regime_fixture]


def test_33_zero_selected_bound_uses_zero_result_limit_and_positive_batch():
    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = []
    batches[(PRICE_HISTORY_TABLE_NAME, REGIME_TICKER)] = []
    snapshot = _snapshot()
    # Inject the zero bound: the fake's row-count bound counts price_history
    # rows across the AAA fixture too (fixture dicts carry no ticker column),
    # so the empty ^GSPC selection's bound is stated, not counted.
    repo = _BoundRecordingRepository(snapshot, _contracts(), batches,
                                     {PRICE_HISTORY_TABLE_NAME: 0})
    computed = panel_row_inputs._read_computed_moves(
        repo, snapshot, "AAA", _HISTORY_START, _DECISION, _EVENT)
    # A zero-row pinned membership is still a bounded query; the read then
    # propagates get_price_series' missing-ticker CONTRACT_MISMATCH as-is.
    with pytest.raises(DataError) as exc:
        panel_row_inputs._read_regime_prices(repo, snapshot, _HISTORY_START, _DECISION)
    assert exc.value.code == "CONTRACT_MISMATCH"
    (_cm, cm_query), (_reg, reg_query) = repo.scans
    assert cm_query.max_result_rows == 0
    assert cm_query.max_batch_rows > 0
    assert reg_query.max_result_rows == 0
    assert reg_query.max_batch_rows > 0
    assert computed == []


# --------------------------------------------------------------------------
# stored implied-history carry step (mean_prior_or_implied)
# --------------------------------------------------------------------------


def _implied_row(day, move, skipped, **overrides):
    return _with_implied(_computed_rows([(day, move, skipped)])[0], **overrides)


@pytest.mark.parametrize("admitted,mean,n,implied,expected", [
    (True, 2.0, 2, 5.0, 3.0),
    (True, 1.5, 3, 5.0, 2.375),   # (1.5*3 + 5.0) / (3 + 1) = 9.5 / 4
    (True, 10.0, 4, 0.0, 8.0),    # (10.0*4 + 0.0) / (4 + 1) = 40.0 / 5
    (True, -2.0, 2, 7.0, 1.0),    # (-2.0*2 + 7.0) / (2 + 1) = 3.0 / 3
    (True, 0.5, 1, 0.25, 0.375),  # (0.5*1 + 0.25) / (1 + 1) = 0.75 / 2
    (True, None, 0, 4.0, 4.0),
    (True, 2.0, 2, None, 2.0),
    (True, None, 0, None, None),
    (False, None, 0, 7.0, None),
])
def test_34_stored_implied_mean_carry_step(admitted, mean, n, implied, expected):
    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [
        _implied_row("2024-01-10", 1.0, False, admitted=admitted,
                     mean_prior_or_implied=mean, n_prior_implied=n,
                     implied_move_pct=implied)]
    value = _scan(batches=batches).panel_row["mean_prior_or_implied"]
    if expected is None:
        assert value is None
    else:
        assert value == pytest.approx(expected)


def test_35_stored_implied_mean_uses_latest_event_date_not_list_order():
    batches = _default_batches()
    earlier = _implied_row("2024-01-05", 1.0, False, admitted=True,
                           mean_prior_or_implied=100.0, n_prior_implied=1,
                           implied_move_pct=100.0)
    later = _implied_row("2024-01-10", 1.0, False, admitted=True,
                         mean_prior_or_implied=2.0, n_prior_implied=1,
                         implied_move_pct=4.0)
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [later, earlier]
    assert _scan(batches=batches).panel_row["mean_prior_or_implied"] == pytest.approx(3.0)


def test_36_stored_implied_mean_uses_skipped_latest_row_but_history_ignores_it():
    batches = _default_batches()
    older = _implied_row("2024-01-05", 2.0, False)
    latest = _implied_row("2024-01-10", None, True, admitted=True,
                          mean_prior_or_implied=2.0, n_prior_implied=1,
                          implied_move_pct=4.0)
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [older, latest]
    panel = _scan(batches=batches).panel_row
    assert panel["mean_prior_or_implied"] == pytest.approx(3.0)
    assert panel["n_prior"] == 1


def test_37_stored_implied_mean_is_none_but_present_without_rows():
    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = []
    panel = _scan(batches=batches).panel_row
    assert "mean_prior_or_implied" in panel
    assert panel["mean_prior_or_implied"] is None


def test_38_stored_implied_mean_ignores_rows_on_or_after_the_decision():
    batches = _default_batches()
    older = _implied_row("2024-01-10", 1.0, False, admitted=True,
                         mean_prior_or_implied=2.0, n_prior_implied=1,
                         implied_move_pct=4.0)
    at_decision = _implied_row("2024-02-14", 1.0, False, admitted=True,
                               mean_prior_or_implied=100.0, n_prior_implied=1,
                               implied_move_pct=100.0)
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [older, at_decision]
    assert _scan(batches=batches).panel_row["mean_prior_or_implied"] == pytest.approx(3.0)


def _without_implied_columns():
    kept = tuple(column for column in COMPUTED_MOVES_CONTRACT.columns
                 if column.name not in panel_row_inputs._IMPLIED_COLUMNS)
    return dataclasses.replace(COMPUTED_MOVES_CONTRACT, columns=kept)


def _without_implied_column(removed):
    kept = tuple(column for column in COMPUTED_MOVES_CONTRACT.columns
                 if column.name != removed)
    return dataclasses.replace(COMPUTED_MOVES_CONTRACT, columns=kept)


def test_39_v3_contract_missing_implied_columns_refuses_even_with_empty_window():
    contracts = {**_contracts(), COMPUTED_MOVES_TABLE_NAME: _without_implied_columns()}
    snapshot = _snapshot()
    batches = _default_batches()
    repo = _FakeRepository(snapshot, contracts, batches)
    with pytest.raises(DataError) as exc:
        scan_panel_row(repo, snapshot, _key(), history_start=_HISTORY_START,
                       decision_session=_DECISION)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {
        "table_name": COMPUTED_MOVES_TABLE_NAME,
        "missing_columns": list(panel_row_inputs._IMPLIED_COLUMNS)}

    # The zero-day window short-circuits the query but never the validation.
    batches = _default_batches()
    batches[("daily_market", "AAA")] = _dm_rows("AAA", ["2024-02-14"])
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = []
    repo = _FakeRepository(snapshot, contracts, batches)
    with pytest.raises(DataError) as exc:
        scan_panel_row(repo, snapshot, _key(), history_start=_DECISION,
                       decision_session=_DECISION)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {
        "table_name": COMPUTED_MOVES_TABLE_NAME,
        "missing_columns": list(panel_row_inputs._IMPLIED_COLUMNS)}


@pytest.mark.parametrize("removed", panel_row_inputs._IMPLIED_COLUMNS)
def test_39b_v3_contract_missing_each_implied_column_refuses(removed):
    contracts = {**_contracts(),
                 COMPUTED_MOVES_TABLE_NAME: _without_implied_column(removed)}
    snapshot = _snapshot()
    repo = _FakeRepository(snapshot, contracts, _default_batches())
    with pytest.raises(DataError) as exc:
        scan_panel_row(repo, snapshot, _key(), history_start=_HISTORY_START,
                       decision_session=_DECISION)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {
        "table_name": COMPUTED_MOVES_TABLE_NAME, "missing_columns": [removed]}


def test_40_row_missing_an_implied_key_refuses():
    batches = _default_batches()
    row = _implied_row("2024-01-10", 1.0, False, admitted=True)
    del row["n_prior_implied"]
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [row]
    with pytest.raises(DataError) as exc:
        _scan(batches=batches)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {
        "table_name": COMPUTED_MOVES_TABLE_NAME,
        "missing_columns": ["n_prior_implied"]}


def test_40b_row_missing_implied_keys_unions_across_rows_sorted():
    batches = _default_batches()
    earlier = _implied_row("2024-01-05", 1.0, False, admitted=True)
    del earlier["event_ordinal"]
    del earlier["implied_move_pct"]
    later = _implied_row("2024-01-10", 1.0, False, admitted=True)
    del later["n_prior_implied"]
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [earlier, later]
    with pytest.raises(DataError) as exc:
        _scan(batches=batches)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details == {
        "table_name": COMPUTED_MOVES_TABLE_NAME,
        "missing_columns": sorted(["event_ordinal", "implied_move_pct", "n_prior_implied"])}



@pytest.mark.parametrize("overrides,field", [
    ({"admitted": 1}, "admitted"),
    ({"n_prior_implied": -1, "mean_prior_or_implied": 2.0}, "n_prior_implied"),
    ({"n_prior_implied": True, "mean_prior_or_implied": 2.0}, "n_prior_implied"),
    ({"n_prior_implied": 1.5, "mean_prior_or_implied": 2.0}, "n_prior_implied"),
    ({"n_prior_implied": "2", "mean_prior_or_implied": 2.0}, "n_prior_implied"),
    ({"mean_prior_or_implied": float("nan"), "n_prior_implied": 1},
     "mean_prior_or_implied"),
    ({"implied_move_pct": float("inf")}, "implied_move_pct"),
    ({"mean_prior_or_implied": None, "n_prior_implied": 2}, "mean_prior_or_implied"),
    ({"mean_prior_or_implied": 3.0, "n_prior_implied": 0}, "mean_prior_or_implied"),
    ({"mean_prior_or_implied": True, "n_prior_implied": 1}, "mean_prior_or_implied"),
    ({"mean_prior_or_implied": "2.0", "n_prior_implied": 1}, "mean_prior_or_implied"),
    ({"implied_move_pct": True, "mean_prior_or_implied": 2.0, "n_prior_implied": 1},
     "implied_move_pct"),
    ({"implied_move_pct": "4.0", "mean_prior_or_implied": 2.0, "n_prior_implied": 1},
     "implied_move_pct"),
])
def test_41_bad_implied_values_refuse(overrides, field):
    batches = _default_batches()
    kwargs = {"admitted": True, **overrides}
    row = _implied_row("2024-01-10", 1.0, False, **kwargs)
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [row]
    with pytest.raises(DataError) as exc:
        _scan(batches=batches)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert exc.value.problem.details["field"] == field


def test_42_bundle_presence_of_stored_implied_mean():
    from engine.v2.scoring.nightly_source_bundle import _project_features

    carry = _implied_row("2024-01-10", 1.0, False, admitted=True,
                         mean_prior_or_implied=2.0, n_prior_implied=2,
                         implied_move_pct=5.0)
    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [carry]
    with_value = _scan(batches=batches).panel_row
    assert with_value["mean_prior_or_implied"] == pytest.approx(3.0)

    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [
        _implied_row("2024-01-10", 1.0, False, admitted=False)]
    without_value = _scan(batches=batches).panel_row
    assert without_value["mean_prior_or_implied"] is None

    for panel, expected in ((with_value, 3.0), (without_value, None)):
        vector, missing = _project_features({}, panel, ["mean_prior_or_implied"],
                                            pd.Timestamp("2024-02-14"))
        if expected is None:
            assert vector["mean_prior_or_implied"] is None
        else:
            assert vector["mean_prior_or_implied"] == pytest.approx(expected)
        assert missing["mean_prior_or_implied"] is False


def test_43_implied_columns_are_in_the_projection_and_real_contract():
    real_columns = {column.name for column in COMPUTED_MOVES_CONTRACT.columns}
    for name in panel_row_inputs._IMPLIED_COLUMNS:
        assert name in panel_row_inputs._COMPUTED_COLUMNS
        assert name in real_columns


def test_44_implied_carry_ignores_a_late_availability_that_history_respects():
    batches = _default_batches()
    batches[(COMPUTED_MOVES_TABLE_NAME, "AAA")] = [
        _implied_row("2024-01-10", 1.0, False, admitted=True,
                     available_as_of_date="2024-03-01",
                     mean_prior_or_implied=2.0, n_prior_implied=1,
                     implied_move_pct=4.0)]
    panel = _scan(batches=batches).panel_row
    # (m*n + i) / (n + 1) = (2.0*1 + 4.0) / (1 + 1) = 3.0, even though the
    # row's availability is after the 2024-02-14 decision (so it never feeds
    # the prior history).
    assert panel["mean_prior_or_implied"] == pytest.approx(3.0)
    assert panel["n_prior"] == 0


def test_45_gspc_rows_before_history_start_are_excluded_from_regime_inputs():
    """A ^GSPC row older than ``history_start`` never reaches the regime
    series, and the rows that do reach it are actually used: the composed
    ``spy_*`` values are the real ``regime.add_regime_features`` outcome over
    ONLY the in-window fixture rows (the early rows' sentinel closes must not
    shift the anchor or any value), and the windows the 44 in-window closes can
    compute are finite -- a panel that discarded the regime data entirely would
    show every one of them NaN."""
    in_window = _gspc_rows(periods=44, end="2024-02-14")  # 2024-01-02 .. 2024-02-14
    assert in_window[0]["date"] == str(_HISTORY_START.date())
    early = [{"date": day, "close_adj": 900.0 + index, "close_raw": 900.0 + index,
              "high_raw": 950.0 + index, "retrieved_at": "2024-02-01T00:00:00Z",
              "deleted": False, "source_kind": "legacy_gspc_csv",
              "source_hash": fake_hash("gspc-early")}
             for index, day in enumerate(
                 str(stamp.date()) for stamp in pd.date_range("2023-04-21", "2024-01-01"))]
    batches = _default_batches()
    batches[(PRICE_HISTORY_TABLE_NAME, REGIME_TICKER)] = early + in_window
    with_early = _scan(batches=batches)
    batches = _default_batches()
    batches[(PRICE_HISTORY_TABLE_NAME, REGIME_TICKER)] = in_window
    baseline = _scan(batches=batches)
    _rows_equal(with_early.panel_row, baseline.panel_row)
    assert with_early.panel_anchor == baseline.panel_anchor
    # In-window rows produce real features, never an all-NaN discard.
    for name in ("spy_ret21", "spy_vol5", "spy_vol20"):
        assert math.isfinite(baseline.panel_row[name]), baseline.panel_row[name]
    expected = regime.add_regime_features(
        pd.DataFrame({"date": [_EVENT], "decision": [_DECISION]}),
        pd.DataFrame({"date": pd.to_datetime([row["date"] for row in in_window]),
                      "close": [row["close_raw"] for row in in_window]}),
        as_of_column="decision").iloc[0]
    for name in panel_row_inputs._REGIME_COLUMNS:
        assert _nan_equal(baseline.panel_row[name], expected[name])


def test_46_history_start_includes_the_exact_boundary_day_not_the_day_before():
    """The regime read's ``history_start`` bound is inclusive at equality: a
    ^GSPC row dated exactly ``history_start`` (2024-01-02) reaches the regime
    series, the row dated the day before (2024-01-01) does not."""
    gspc = _gspc_rows(periods=32)  # consecutive closes 2024-01-01 .. 2024-02-01
    assert gspc[0]["date"] == "2024-01-01"
    snapshot = _snapshot()
    batches = _default_batches()
    batches[(PRICE_HISTORY_TABLE_NAME, REGIME_TICKER)] = gspc
    repo = _FakeRepository(snapshot, _contracts(), batches)
    rows = panel_row_inputs._read_regime_prices(repo, snapshot, _HISTORY_START, _DECISION)
    assert rows[0].date == "2024-01-02"
    assert "2024-01-01" not in {row.date for row in rows}


class _ByteBoundRepository(_FakeRepository):
    """The shared fake with ``Repository.scan``'s pre-decode byte bound added:
    a scan whose ``max_object_bytes`` is below the selected rows' rough object
    size refuses the repository's own bound error before any row is yielded,
    and every forwarded cap is recorded per table."""

    def __init__(self, snapshot, contracts, batches_by_key) -> None:
        super().__init__(snapshot, contracts, batches_by_key)
        self.byte_caps: list[tuple[str, object]] = []

    def scan(self, query, *, table_name, max_object_bytes=None):
        self.byte_caps.append((table_name, max_object_bytes))
        ticker = query.key_filter[0].values[0]
        rows = self._batches.get((table_name, ticker), [])
        selected = [row for row in rows if _bound_row_selected(
            row, query.key_filter, query.time_interval)]
        if max_object_bytes is not None and sum(len(str(row)) for row in selected) > max_object_bytes:
            from engine.v2.data import errors
            raise DataError(errors.make_problem("RESULT_LIMIT_EXCEEDED",
                                                "object exceeds max_object_bytes",
                                                details={"table_name": table_name}))
        yield _Batch(selected)


def _byte_bound_reads():
    snapshot = _snapshot()
    repo = _ByteBoundRepository(snapshot, _contracts(), _default_batches())
    return snapshot, repo


def test_47_regime_read_forwards_the_module_byte_cap():
    """The ^GSPC regime read is bounded by ``_REGIME_MAX_OBJECT_BYTES`` (the run-up
    read of ``key.ticker`` is not this reader's concern and stays uncapped)."""
    snapshot, repo = _byte_bound_reads()
    rows = panel_row_inputs._read_regime_prices(repo, snapshot, _HISTORY_START, _DECISION)
    assert rows and rows[0].close_raw is not None
    assert repo.byte_caps == [(PRICE_HISTORY_TABLE_NAME,
                               panel_row_inputs._REGIME_MAX_OBJECT_BYTES)]


def test_48_tiny_regime_byte_cap_refuses_before_decode(monkeypatch):
    """With the cap driven tiny, the regime read refuses the repository's own
    bound error instead of decoding the fixture's rows."""
    monkeypatch.setattr(panel_row_inputs, "_REGIME_MAX_OBJECT_BYTES", 1)
    snapshot, repo = _byte_bound_reads()
    with pytest.raises(DataError, match="object exceeds max_object_bytes") as caught:
        panel_row_inputs._read_regime_prices(repo, snapshot, _HISTORY_START, _DECISION)
    assert caught.value.code == "RESULT_LIMIT_EXCEEDED"
    assert repo.byte_caps == [(PRICE_HISTORY_TABLE_NAME, 1)]


def test_49_nonfinite_regime_close_refuses_in_both_readers():
    """The shared ``regime.validated_regime_closes`` rule: an infinite close is
    refused by the panel reader and by the research block builder alike."""
    from engine.v2.research import magnitude_panel_daily as research_daily
    rows = list(_gspc_rows(periods=25))
    rows[5] = {**rows[5], "close_raw": float("inf")}
    snapshot, repo = _byte_bound_reads()
    from engine.v2.contracts import PriceSeriesRow
    series = tuple(PriceSeriesRow(date=row["date"], close_adj=row["close_adj"],
                                  close_raw=row["close_raw"], high_raw=row["high_raw"],
                                  retrieved_at=row["retrieved_at"], source_hash=row["source_hash"])
                   for row in rows)
    with pytest.raises(DataError, match="regime source closes must be positive") as caught:
        panel_row_inputs._regime_from_prices(series, _EVENT, _DECISION)
    assert caught.value.code == "CONTRACT_MISMATCH"
    with pytest.raises(DataError, match="regime source closes must be positive") as caught:
        research_daily._spy_blocks(
            [{"event_id": "synthetic", "event_date": "2024-02-15", "forecasts": [
                {"horizon_sessions": 1, "source_session_ceiling": "2024-02-14"}]}],
            [{"date": row["date"], "close": row["close_raw"]} for row in rows])
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_50_regime_closes_validator_is_pure_and_exact():
    """The shared validator accepts finite positives and raises ``ValueError``
    (never a typed refusal) for a non-finite or non-positive value."""
    assert regime.validated_regime_closes([1.0, 4700.5]) == [1.0, 4700.5]
    for bad in (0.0, -1.0, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            regime.validated_regime_closes([1.0, bad])
