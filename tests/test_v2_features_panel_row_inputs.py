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
from engine.v2.data.computed_moves import build_rows
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
# path exercises ``max`` (not a tie): daily_state 02-05, regime 02-01, runup 01-20.
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
    # available_as_of_date follows the COMPUTED_MOVES_CONTRACT null policy:
    # a non-skipped row is knowable the day after its event, a skipped row
    # (null realized_move_pct) carries a null availability -- never "assume
    # available". Delayed-close fixtures override it with producer-derived
    # dates (see test_19).
    return [{"ticker": "AAA", "event_date": day, "realized_move_pct": move,
             "skipped": skipped,
             "available_as_of_date": None if skipped else
             (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()}
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


def _spy_rows(periods=20):
    days = [str(d.date()) for d in pd.date_range(end=_REGIME_SOURCE, periods=periods, freq="D")]
    return [{"ticker": "SPY", "date": pd.Timestamp(day), "spot": 500.0 + i,
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


def test_1_happy_path_all_four_reads_and_max_anchor():
    # add_regime_features needs j >= 21 (22 rows, anchor at the last) for a
    # non-NaN spy_ret21, so this test widens the SPY batch; every other test
    # keeps the default 20-row fixture.
    spy = _spy_rows(periods=22)
    batches = _default_batches()
    batches[("daily_market", "SPY")] = spy
    result = _scan(batches=batches)
    assert isinstance(result, PanelRowInputs)
    panel = result.panel_row
    # One key from each of the four sources.
    assert "im" in panel and panel["im"] == 2.0
    assert "n_prior" in panel and panel["n_prior"] == 2
    assert "spy_ret21" in panel
    assert "signed_streak" in panel
    # The three contributing dates differ; the anchor is the latest.
    assert result.panel_anchor == pd.Timestamp(_DAILY_STATE_SOURCE)
    # spy_* and runup_* keys survive as plain floats; the ema fallback is the mean.
    # regime.py's formula, mirrored over the same spots: every SPY date is
    # strictly before both the event and the decision, so the anchor is the
    # last row (j = len - 1) and spy_ret21 = (spot / closes[j - 21] - 1) * 100.
    spots = [row["spot"] for row in spy]
    expected_ret21 = (spots[-1] / spots[-1 - 21] - 1.0) * 100
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

        def scan(self, query, *, table_name):
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


def test_14_spy_columns_exist_in_the_real_daily_market_contract():
    """_SPY_COLUMNS must name real daily_market columns -- the fake
    repository's scan() does not validate this, so without this test a
    revert back to the old "close" bug would still pass every other test
    in this file."""
    from engine.v2.features import panel_row_inputs
    real_columns = {column.name for column in contract_for("daily_market").columns}
    assert set(panel_row_inputs._SPY_COLUMNS) <= real_columns


def test_15_null_spy_spot_raises_contract_mismatch():
    batches = _default_batches()
    spy_rows = _spy_rows()
    spy_rows[3] = {**spy_rows[3], "spot": None}  # daily_market.spot is nullable
    batches[("daily_market", "SPY")] = spy_rows
    with pytest.raises(DataError) as exc:
        _scan(batches=batches)
    assert exc.value.code == "CONTRACT_MISMATCH"


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
    repo = _FakeRepository(snapshot, _contracts(), batches)
    result = scan_panel_row(repo, snapshot, _key(),
                            history_start=_DECISION, decision_session=_DECISION)
    assert result.panel_row["n_prior"] == 0


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

    def scan(self, query, *, table_name):
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
        pd.DataFrame({"date": pd.to_datetime(["2024-01-08"]), "implied_move": [1.0]}),
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
