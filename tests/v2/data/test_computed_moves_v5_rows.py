"""``computed_moves.v5`` rows (issue #966): one row per considered event, with a reason.

An event the v4 builder silently dropped -- a ``forward_claim`` phantom, an
ORATS row without a session -- is now a row whose ``not_computed_reason`` says
why nothing was computed, and it must not disturb its confirmed neighbours'
``event_ordinal`` or ``mean_prior_or_implied``. The two ticker-level reasons
reach every row of the ticker. A confirmed row is the v4 row plus the reason,
and every considered event gets exactly one row (``[]`` only for an empty event
list); same-date rows collapse to one -- confirmed wins over unconfirmed, else
the first in input order -- so ``(ticker, event_date)`` stays unique.
"""
from __future__ import annotations

import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from engine.v2.data import manifests
from engine.v2.data.computed_moves import (
    ADMISSION_EVENTS,
    build_event_rows,
    build_rows,
    classify_event_reasons,
)
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_V5_CONTRACT
from engine.v2.data.errors import DataError

_BDAYS = pd.bdate_range("2024-01-01", periods=60)
_SD = _BDAYS.to_numpy(dtype="datetime64[ns]")
_SC = np.linspace(100.0, 130.0, len(_BDAYS))
_DAILY = pd.DataFrame({"date": _BDAYS,
                       "implied_move": [float(i) for i in range(len(_BDAYS))],
                       "src_iv": ["orats"] * len(_BDAYS)})
_KW = dict(computed_at="2026-09-25T00:00:00+00:00", source_hash="sha256:" + "a" * 64,
           capture_id="capture_test")

#: The exact key set of a v5 row: the contract's own column names.
_V5_COLUMNS = tuple(column.name for column in COMPUTED_MOVES_V5_CONTRACT.columns)

#: 16 ORATS-confirmed sessioned events (business days 10..25): at least
#: ``MIN_SCOREABLE``, and each one bracketable, so no ticker-level reason can
#: mask the per-event reasons these tests target.
_CONFIRMED = [(_BDAYS[i], "BMO", True) for i in range(10, 26)]

#: Saturday 2024-01-20, between ``_CONFIRMED[4]`` (Fri 2024-01-19) and
#: ``_CONFIRMED[5]`` (Mon 2024-01-22) -- an unconfirmed row a phantom or a
#: session-less event can take without disturbing the confirmed order.
_BETWEEN = pd.Timestamp("2024-01-20")


def _events(*rows):
    """One considered-event frame from ``(event_date, session, src_orats)`` triples."""
    return pd.DataFrame({"event_date": pd.to_datetime([day for day, _, _ in rows]),
                         "session": [session for _, session, _ in rows],
                         "src_orats": [orats for _, _, orats in rows]})


def _build(events, sd=_SD, sc=_SC, daily=_DAILY, **kw):
    return build_event_rows("AAPL", events, sd, sc, daily, **_KW, **kw)


def _assert_refused(row, reason, ordinal):
    """Every refused row: the reason, and the same no-outcome null/0/False shape."""
    assert row["not_computed_reason"] == reason
    assert row["event_ordinal"] == ordinal
    assert row["skipped"] is True
    assert row["admitted"] is False
    assert row["realized_move_pct"] is None
    assert row["available_as_of_date"] is None
    assert row["implied_move_pct"] is None
    assert row["mean_prior_or_implied"] is None
    assert row["n_prior_implied"] == 0


def _assert_refusal_parity(events, expected, *, sd=_SD, sc=_SC, daily=_DAILY, **kw):
    """``classify_event_reasons`` and ``build_event_rows`` refuse ALIKE (issue #966).

    For every input both entry points validate: CONTRACT_MISMATCH with the exact
    code's message, the SAME text out of both -- so a caller switching between
    them never meets two flavours of one refusal.
    """
    calls = (
        lambda: classify_event_reasons(events, sd, sc, **kw),
        lambda: build_event_rows("AAPL", events, sd, sc, daily, **_KW, **kw),
    )
    messages = []
    for call in calls:
        with pytest.raises(DataError) as caught:
            call()
        assert caught.value.code == "CONTRACT_MISMATCH"
        messages.append(caught.value.problem.message)
    assert messages == [expected, expected]


def test_prices_known_false_classifies_ticker_too_few_events_without_prices():
    no_dates, no_closes = np.array([], dtype="datetime64[ns]"), np.array([])
    considered = _events(*_CONFIRMED[:3])

    rows = _build(considered, sd=no_dates, sc=no_closes, prices_known=False)

    assert len(rows) == 3
    for position, row in enumerate(rows):
        _assert_refused(row, "ticker_too_few_events", position)  # never no_price_history
    assert classify_event_reasons(
        considered, no_dates, no_closes, prices_known=False) == ["ticker_too_few_events"] * 3


@pytest.mark.parametrize("entry", ["build", "classify"])
def test_prices_known_false_refuses_a_ticker_at_or_above_the_bar(entry):
    no_dates, no_closes = np.array([], dtype="datetime64[ns]"), np.array([])
    considered = _events(*_CONFIRMED)  # 16 confirmed events clear MIN_SCOREABLE
    def call():
        if entry == "build":
            return _build(considered, sd=no_dates, sc=no_closes, prices_known=False)
        return classify_event_reasons(considered, no_dates, no_closes, prices_known=False)
    with pytest.raises(DataError) as error:
        call()
    assert error.value.code == "CONTRACT_MISMATCH"
    assert "fewer than min_scoreable confirmed events" in error.value.problem.message


@pytest.mark.parametrize("entry", ["build", "classify"])
def test_prices_known_must_be_a_bool(entry):
    considered = _events(*_CONFIRMED[:3])
    def call():
        if entry == "build":
            return _build(considered, prices_known="no")
        return classify_event_reasons(considered, _SD, _SC, prices_known="no")
    with pytest.raises(DataError) as error:
        call()
    assert error.value.code == "CONTRACT_MISMATCH"
    assert "prices_known must be a bool" in error.value.problem.message


def test_phantom_forward_claim_is_not_orats_and_leaves_neighbours_unchanged():
    base = _build(_events(*_CONFIRMED))
    considered = _CONFIRMED[:5] + [(_BETWEEN, "BMO", False)] + _CONFIRMED[5:]
    rows = _build(_events(*considered))

    assert len(rows) == len(base) + 1
    _assert_refused(rows[5], "not_orats", 5)
    assert rows[5]["event_date"] == "2024-01-20"

    # the phantom shifts nothing: every confirmed neighbour is field-for-field
    # what it was without it, ordinal and prior mean included
    assert rows[:5] == base[:5]
    assert rows[6:] == base[5:]
    neighbour = rows[6]
    # the neighbour's own values, hard-coded: 2024-01-22 BMO counts its ordinal
    # (5 >= WARMUP, so admitted), measures close[15]/close[14] on _SC, and takes
    # its prior mean from the ONE earlier admitted event -- the 2024-01-19 row's
    # as-of anchor on the 2024-01-18 daily row, 13.0
    assert neighbour["event_ordinal"] == 5
    assert neighbour["admitted"] is True
    assert neighbour["realized_move_pct"] == pytest.approx(0.4746835443037975, rel=1e-9)
    assert neighbour["mean_prior_or_implied"] == 13.0
    assert neighbour["n_prior_implied"] == 1
    assert neighbour["not_computed_reason"] is None


@pytest.mark.parametrize("null", [None, float("nan")])
def test_confirmed_event_without_a_session_is_no_session(null):
    considered = _CONFIRMED[:5] + [(_BETWEEN, null, True)] + _CONFIRMED[5:]
    events = _events(*considered)
    rows = _build(events)

    _assert_refused(rows[5], "no_session", 5)
    assert rows[5]["event_date"] == "2024-01-20"
    assert rows[:5] + rows[6:] == _build(_events(*_CONFIRMED))
    assert classify_event_reasons(events, _SD, _SC)[5] == "no_session"


def test_three_confirmed_events_all_get_ticker_too_few_events():
    rows = _build(_events(*_CONFIRMED[:3]))

    assert len(rows) == 3
    for position, row in enumerate(rows):
        _assert_refused(row, "ticker_too_few_events", position)


def test_empty_price_history_gives_every_row_no_price_history():
    # 16 confirmed events clear MIN_SCOREABLE, so the missing closes are the only
    # thing that can fire: the reason is no_price_history, never ticker_too_few_events.
    rows = _build(_events(*_CONFIRMED), sd=np.array([], dtype="datetime64[ns]"),
                  sc=np.array([]))

    assert len(rows) == len(_CONFIRMED)
    for position, row in enumerate(rows):
        _assert_refused(row, "no_price_history", position)


def test_confirmed_rows_are_exactly_build_rows_on_the_confirmed_only_frame():
    late = _BDAYS[-1] + pd.Timedelta(days=3)  # past the last close: a v4 skip
    considered = [*_CONFIRMED, (late, "BMO", True)]
    events = _events(*considered)
    rows = _build(events)
    v4 = build_rows("AAPL", events, _SD, _SC, _DAILY, **_KW)

    assert len(rows) == len(v4) == len(considered)
    for row, old in zip(rows, v4):
        assert "not_computed_reason" not in old  # the v4 row dict is untouched
        reason = "no_price_window" if old["skipped"] else None
        assert row == {**old, "not_computed_reason": reason}
    assert {row["not_computed_reason"] for row in rows} == {None, "no_price_window"}


def test_one_row_per_considered_event():
    considered = (_CONFIRMED[:5]
                  + [(_BETWEEN, "BMO", False), (_BETWEEN + pd.Timedelta(days=1), None, True)]
                  + _CONFIRMED[5:])
    rows = _build(_events(*considered))

    assert len(rows) == len(considered)
    assert [row["event_date"] for row in rows] == [
        day.strftime("%Y-%m-%d") for day, _, _ in considered]
    assert len({(row["ticker"], row["event_date"]) for row in rows}) == len(considered)
    confirmed_after = len(_CONFIRMED) - 5
    assert [row["not_computed_reason"] for row in rows] == (
        [None] * 5 + ["not_orats", "no_session"] + [None] * confirmed_after)
    # ordinals count earlier CONFIRMED events only: the two unconfirmed rows both
    # see 5, and the confirmed event after them keeps the ordinal it had alone
    assert [row["event_ordinal"] for row in rows] == (
        list(range(5)) + [5, 5] + list(range(5, len(_CONFIRMED))))


#: The date a CONFIRMED event can carry with no price window bracketing it:
#: Friday 2023-12-29, three days before ``_BDAYS[0]``'s close, so there is no
#: earlier close to measure against. Confirmed like any other event, but not
#: computable -- and the classification, not a built row, must say so.
_EARLY = _BDAYS[0] - pd.Timedelta(days=3)

#: The exact reason list of the mixed frame below, position by position: the
#: early confirmed event has no window, the next five and the last eleven of
#: ``_CONFIRMED`` compute, the Saturday phantom is not ORATS and the Sunday
#: event has no session.
_MIXED_REASONS = (["no_price_window"] + [None] * 5 + ["not_orats", "no_session"]
                  + [None] * 11)


def _mixed_events():
    """One frame with every non-ticker-level reason, in date order."""
    return _events(*[(_EARLY, "BMO", True), *_CONFIRMED[:5],
                     (_BETWEEN, "BMO", False), (_BETWEEN + pd.Timedelta(days=1), None, True),
                     *_CONFIRMED[5:]])


def test_classify_event_reasons_returns_the_exact_reason_list():
    assert classify_event_reasons(_mixed_events(), _SD, _SC) == _MIXED_REASONS
    assert [row["not_computed_reason"] for row in _build(_mixed_events())] == _MIXED_REASONS


def test_classify_event_reasons_returns_the_ticker_level_lists():
    assert classify_event_reasons(_events(*_CONFIRMED[:3]), _SD, _SC) == (
        ["ticker_too_few_events"] * 3)
    assert classify_event_reasons(_events(*_CONFIRMED), np.array([], dtype="datetime64[ns]"),
                                  np.array([])) == (["no_price_history"] * len(_CONFIRMED))


def test_the_no_price_window_row_is_skipped_and_carries_that_reason():
    considered = [(_EARLY, "BMO", True), *_CONFIRMED]
    events = _events(*considered)
    rows = _build(events)
    v4 = build_rows("AAPL", events, _SD, _SC, _DAILY, **_KW)

    window = rows[0]
    assert window["event_date"] == "2023-12-29"
    assert window["not_computed_reason"] == "no_price_window"
    assert window["skipped"] is True
    assert window["realized_move_pct"] is None
    assert window["available_as_of_date"] is None
    assert window == {**v4[0], "not_computed_reason": "no_price_window"}
    assert [row["not_computed_reason"] for row in rows[1:]] == [None] * len(_CONFIRMED)
    assert [row["skipped"] for row in rows[1:]] == [False] * len(_CONFIRMED)


# --------------------------------------------------------------------------
# same-date collapse, the min_scoreable bar, hand-computed values, the contract
# --------------------------------------------------------------------------

#: Eight sessions from Friday 2024-03-01. Each close is an exact multiple of the
#: previous one by a binary fraction (x1.25, x0.875, x1.5, x0.75, x1.125, x1.25,
#: x0.75), so every realised move below is exact -- no rounding to argue about.
_D8 = pd.bdate_range("2024-03-01", periods=8)
_SD8 = _D8.to_numpy(dtype="datetime64[ns]")
_SC8 = np.array([128.0, 160.0, 140.0, 210.0, 157.5, 177.1875, 221.484375, 166.11328125])

#: The daily frame behind ``_DAILY``-style implied anchors: event i (i >= 1)
#: anchors on the row strictly before it, so its ``implied_move_pct`` is i.
_DAILY8 = pd.DataFrame({"date": _D8,
                        "implied_move": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        "src_iv": ["orats"] * len(_D8)})

#: One ORATS-confirmed BMO event per session of ``_D8``: 8 confirmed, 7 of them
#: bracketable (the first has no earlier close), so both admission bars clear.
_D8_EVENTS = [(_D8[i], "BMO", True) for i in range(len(_D8))]


def test_same_date_phantom_collapses_to_the_confirmed_row():
    base = _build(_events(*_CONFIRMED))
    # the unconfirmed twin sits FIRST, on 2024-01-19, and must still lose
    considered = [*_CONFIRMED[:4], (_CONFIRMED[4][0], "AMC", False), *_CONFIRMED[4:]]
    rows = _build(_events(*considered))

    assert len(rows) == len(base)  # one row for the two same-date events
    keys = [(row["ticker"], row["event_date"]) for row in rows]
    assert len(set(keys)) == len(keys)
    assert rows == base  # the confirmed row won, field for field
    assert [row["event_date"] for row in rows] == [
        day.strftime("%Y-%m-%d") for day, _, _ in _CONFIRMED]  # order preserved

    # the collapsed row's own values, hard-coded rather than read off another
    # builder call: 2024-01-19 BMO is the 5th confirmed event (ordinal 4 ==
    # WARMUP, so admitted), measures close[14]/close[13] on _SC, and has no
    # earlier admitted event behind its prior mean -- a computed outcome
    survivor = rows[4]
    assert survivor["event_date"] == "2024-01-19"
    assert survivor["event_ordinal"] == 4
    assert survivor["admitted"] is True
    assert survivor["realized_move_pct"] == pytest.approx(0.4769475357710652, rel=1e-9)
    assert survivor["mean_prior_or_implied"] is None
    assert survivor["n_prior_implied"] == 0
    assert survivor["not_computed_reason"] is None


def test_two_same_date_confirmed_rows_collapse_to_the_first_in_input_order():
    base = _build(_events(*_CONFIRMED))
    day = _CONFIRMED[4][0]  # 2024-01-19 BMO, duplicated as AMC right after it
    considered = [*_CONFIRMED[:5], (day, "AMC", True), *_CONFIRMED[5:]]
    rows = _build(_events(*considered))

    assert len(rows) == len(base)  # two confirmed, one row: the input's first wins
    assert rows == base  # and it is the BMO row, not the AMC twin

    # the collapsed row's own values, hard-coded: it is the BMO event of
    # 2024-01-19 (ordinal 4 == WARMUP, so admitted), measuring close[14]/close[13]
    # on _SC -- the AMC twin would measure close[15]/close[14] -- with no earlier
    # admitted event behind its prior mean and a computed outcome
    survivor = rows[4]
    assert survivor["event_date"] == "2024-01-19"
    assert survivor["event_ordinal"] == 4
    assert survivor["admitted"] is True
    assert survivor["realized_move_pct"] == pytest.approx(0.4769475357710652, rel=1e-9)
    assert survivor["mean_prior_or_implied"] is None
    assert survivor["n_prior_implied"] == 0
    assert survivor["not_computed_reason"] is None


def test_empty_event_list_returns_no_rows():
    empty = _events()

    assert _build(empty) == []
    assert classify_event_reasons(empty, _SD, _SC) == []


@pytest.mark.parametrize("bad", [True, -1])
def test_empty_events_with_invalid_min_scoreable_still_refuse(bad):
    # validation runs BEFORE the empty-input return: an empty frame is not a
    # licence to skip the contract -- both entry points refuse alike
    _assert_refusal_parity(_events(),
                           "min_scoreable must be a non-negative int", min_scoreable=bad)


def test_empty_events_missing_a_required_column_still_refuse():
    # likewise a column check precedes the empty-input return
    _assert_refusal_parity(pd.DataFrame({"event_date": []}),
                           "events lacks required columns: session, src_orats")


def test_min_scoreable_above_the_confirmed_count_refuses_every_row():
    considered = _events(*_D8_EVENTS)  # 8 confirmed, 7 computable: k >= 5 holds
    rows = _build(considered, sd=_SD8, sc=_SC8, daily=_DAILY8, min_scoreable=9)

    assert len(rows) == len(_D8_EVENTS)
    for position, row in enumerate(rows):
        _assert_refused(row, "ticker_too_few_events", position)
        assert set(row) == set(_V5_COLUMNS)
    # the SAME inputs compute when min_scoreable drops to the confirmed count,
    # so the refusal above is the min_scoreable branch and nothing else
    computed = _build(considered, sd=_SD8, sc=_SC8, daily=_DAILY8, min_scoreable=8)
    assert [row["not_computed_reason"] for row in computed].count(None) == 7


def test_confirmed_rows_carry_hand_computed_values():
    rows = _build(_events(*_D8_EVENTS), sd=_SD8, sc=_SC8, daily=_DAILY8, min_scoreable=5)

    assert [row["event_date"] for row in rows] == [
        "2024-03-01", "2024-03-04", "2024-03-05", "2024-03-06",
        "2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12"]
    assert len({(row["ticker"], row["event_date"]) for row in rows}) == len(rows)
    # as-of anchor of event i is the daily row strictly before it: i for i >= 1
    assert [row["implied_move_pct"] for row in rows] == [None, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

    # Hand-computed from the series and the session_move definition: BMO measures
    # the close of the event day against the previous close (event 0 has none, so
    # it is a v4 skip); available_as_of_date is that measured close plus one
    # CALENDAR day (Fri 03-08 -> Sat 03-09); ordinal counts confirmed events and
    # admitted is ordinal >= WARMUP (4); mean_prior_or_implied / n_prior_implied
    # are the mean and count over the implied values of EARLIER admitted events.
    expected = [
        # realized, available_as_of, ordinal, admitted, mean_prior, n_prior, reason
        (None,  None,         0, False, None, 0, "no_price_window"),
        (25.0,  "2024-03-05", 1, False, None, 0, None),
        (-12.5, "2024-03-06", 2, False, None, 0, None),
        (50.0,  "2024-03-07", 3, False, None, 0, None),
        (-25.0, "2024-03-08", 4, True,  None, 0, None),
        (12.5,  "2024-03-09", 5, True,  4.0,  1, None),
        (25.0,  "2024-03-12", 6, True,  4.5,  2, None),
        (-25.0, "2024-03-13", 7, True,  5.0,  3, None),
    ]
    for row, want in zip(rows, expected):
        realized, as_of, ordinal, admitted, mean_prior, n_prior, reason = want
        assert row["realized_move_pct"] == realized
        assert row["available_as_of_date"] == as_of
        assert row["event_ordinal"] == ordinal
        assert row["admitted"] is admitted
        assert row["mean_prior_or_implied"] == mean_prior
        assert row["n_prior_implied"] == n_prior
        assert row["not_computed_reason"] == reason
        assert row["skipped"] is (reason is not None)
    refused = [row for row in rows if row["not_computed_reason"] is not None]
    assert refused  # the comparison is not vacuous
    for row in refused:
        assert set(row) == set(_V5_COLUMNS)


def test_one_confirmed_amc_event_end_to_end():
    # 2024-03-01 AMC measures the NEXT close (160.0) against its own (128.0)
    # on _SC8: exactly +25.0%, available the day after that measured close
    # (Mon 2024-03-04 + 1 calendar day), clear of both bars with 6 confirmed
    considered = [(_D8[0], "AMC", True), *_D8_EVENTS[1:6]]
    events = _events(*considered)

    assert classify_event_reasons(events, _SD8, _SC8, min_scoreable=6) == [None] * 6
    rows = _build(events, sd=_SD8, sc=_SC8, daily=_DAILY8, min_scoreable=6)

    assert [row["not_computed_reason"] for row in rows] == [None] * 6
    amc = rows[0]
    assert amc["event_date"] == "2024-03-01"
    assert amc["realized_move_pct"] == 25.0
    assert amc["available_as_of_date"] == "2024-03-05"
    assert amc["skipped"] is False


def test_v5_contract_is_v4_plus_not_computed_reason():
    v4, v5 = COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_V5_CONTRACT

    assert v5.contract_id == "computed_moves.v5"
    assert v5.semantic_version == "5.0.0"
    assert [column.name for column in v5.columns] == (
        [column.name for column in v4.columns] + ["not_computed_reason"])
    added = v5.columns[-1]
    assert added.physical_type == "string"
    assert added.nullable is True
    assert added.null_policy == (
        "null exactly when the row is a computed outcome (skipped false); non-null "
        "exactly when skipped is true and realized_move_pct and available_as_of_date "
        "are null. Closed reason set: not_orats (no ORATS confirmation, e.g. a "
        "forward_claim phantom), no_session (ORATS-confirmed but session null), "
        "no_price_window (the v4 skipped case: a missing close on either side or a "
        "P->Q window wider than MAX_GAP_CALENDAR_DAYS), no_price_history (the "
        "ticker's close series is empty), ticker_too_few_events (fewer than "
        "MIN_SCOREABLE ORATS-confirmed sessioned events or fewer than "
        "ADMISSION_EVENTS computable ones); the last two are ticker-level and apply "
        "to every row of that ticker.")
    assert v5.primary_key == v4.primary_key == ("ticker", "event_date")
    assert v5.partition_columns == v4.partition_columns == ("ticker",)
    identity = ("contract_id", "definition_hash", "semantic_version", "columns")
    shared = [field.name for field in dataclasses.fields(v4) if field.name not in identity]
    assert [name for name in shared if getattr(v5, name) != getattr(v4, name)] == []


@pytest.mark.parametrize("bad", [True, -1, None])
def test_invalid_min_scoreable_is_contract_mismatch(bad):
    _assert_refusal_parity(_events(*_CONFIRMED[:3]),
                           "min_scoreable must be a non-negative int", min_scoreable=bad)


@pytest.mark.parametrize("bad", [2.5, "3"])
def test_non_int_min_scoreable_is_contract_mismatch(bad):
    # a float bar or a numeric string is not an int count: both entry points
    # refuse the type outright, same code and same message
    _assert_refusal_parity(_events(*_CONFIRMED[:3]),
                           "min_scoreable must be a non-negative int", min_scoreable=bad)


def test_np_int64_min_scoreable_is_accepted():
    # the store hands over numpy ints; they clear the same bar as a Python int
    events = _events(*_CONFIRMED)
    assert isinstance(np.int64(12), np.integer)

    assert classify_event_reasons(events, _SD, _SC, min_scoreable=np.int64(12)) == (
        [None] * len(_CONFIRMED))
    rows = _build(events, min_scoreable=np.int64(12))
    assert [row["not_computed_reason"] for row in rows] == [None] * len(_CONFIRMED)


def test_bad_min_scoreable_refuses_before_a_missing_column_on_one_frame():
    # _validate_inputs checks the bar FIRST: a frame carrying BOTH a bad
    # min_scoreable and a missing column raises the min_scoreable message,
    # exact text, out of both entry points
    for bad in (True, -1, 2.5):
        _assert_refusal_parity(pd.DataFrame({"event_date": []}),
                               "min_scoreable must be a non-negative int", min_scoreable=bad)


def test_bad_src_orats_refuses_before_an_unsorted_sd_on_one_frame():
    # src_orats is checked inside _validate_inputs, before _require_price_series,
    # so ONE frame carrying both defects raises the src_orats message first
    events = _events(*_CONFIRMED[:3])
    events["src_orats"] = pd.Series([None, True, True], dtype="object")
    unsorted_sd = _SD[::-1].copy()
    assert np.any(unsorted_sd[1:] < unsorted_sd[:-1])  # the sd really is unsorted
    _assert_refusal_parity(events, "events src_orats must hold a real bool per row",
                           sd=unsorted_sd, sc=_SC)


def test_min_scoreable_zero_is_valid_but_admission_still_applies():
    # 0 is a non-negative int: a ticker with at least ADMISSION_EVENTS computable
    # events computes every row, both entry points alike
    rows = _build(_events(*_CONFIRMED), min_scoreable=0)
    assert len(rows) == len(_CONFIRMED)
    assert [row["not_computed_reason"] for row in rows] == [None] * len(_CONFIRMED)
    assert classify_event_reasons(_events(*_CONFIRMED), _SD, _SC, min_scoreable=0) == (
        [None] * len(_CONFIRMED))

    # fewer than ADMISSION_EVENTS computable is STILL ticker_too_few_events:
    # min_scoreable=0 relaxes only the confirmed-count bar, never the other one
    too_few = _events(*_CONFIRMED[:3])  # 3 confirmed and computable < 5
    assert classify_event_reasons(too_few, _SD, _SC, min_scoreable=0) == (
        ["ticker_too_few_events"] * 3)
    rows = _build(too_few, min_scoreable=0)
    for position, row in enumerate(rows):
        _assert_refused(row, "ticker_too_few_events", position)


def test_missing_event_columns_or_a_price_length_mismatch_refuse():
    _assert_refusal_parity(pd.DataFrame({"event_date": []}),
                           "events lacks required columns: session, src_orats")
    _assert_refusal_parity(_events(*_CONFIRMED[:3]).drop(columns="session"),
                           "events lacks required columns: session")
    _assert_refusal_parity(_events(*_CONFIRMED[:3]),
                           "price series sd and sc differ in length", sc=_SC[:5])


def test_unsorted_sd_refuses_from_both_entry_points():
    # a close series that is not non-decreasing would bracket events against
    # the wrong side of history, so it refuses BEFORE any ticker-level branch,
    # same code and same message out of both entry points
    swapped = _SD.copy()
    swapped[[5, 6]] = swapped[[6, 5]]
    for bad_sd in (_SD[::-1].copy(), swapped):
        assert np.any(bad_sd[1:] < bad_sd[:-1])  # the input really is unsorted
        _assert_refusal_parity(_events(*_CONFIRMED[:3]),
                               "price dates sd must be ascending",
                               sd=bad_sd, sc=_SC)


def test_a_malformed_daily_frame_refuses_even_a_ticker_level_refusal():
    # checked BEFORE branching: the ticker-level reason below cannot mask it,
    # and an empty frame missing the columns is malformed all the same
    for bad_daily in (_DAILY.drop(columns="src_iv"), pd.DataFrame()):
        with pytest.raises(DataError) as caught:
            build_event_rows("AAPL", _events(*_CONFIRMED[:3]), _SD, _SC, bad_daily, **_KW)
        assert caught.value.code == "CONTRACT_MISMATCH"
        assert caught.value.problem.message == (
            "daily_market frame lacks date, implied_move or src_iv")


def test_unparseable_daily_dates_and_non_numeric_implied_move_refuse_both_tickers():
    # each malformed field keeps its OWN message, in BOTH populations: the
    # 3-event ticker (left alone ticker_too_few_events) and the 16-event
    # scoreable one -- the refusal fires before any branch either way
    bad_dates = pd.DataFrame({
        "date": ["2024-01-02", "not-a-date", "2024-01-04"],
        "implied_move": [1.0, 2.0, 3.0],
        "src_iv": ["orats", "orats", "orats"]})
    bad_implied = _DAILY.assign(implied_move=["bad"] * len(_DAILY))
    assert not pd.api.types.is_datetime64_any_dtype(bad_dates["date"])  # really strings
    assert not pd.api.types.is_numeric_dtype(bad_implied["implied_move"])
    cases = ((bad_dates, "daily_market dates are unparseable"),
             (bad_implied, "daily_market implied_move must be numeric"))
    for events in (_events(*_CONFIRMED[:3]), _events(*_CONFIRMED)):
        for bad_daily, message in cases:
            with pytest.raises(DataError) as caught:
                build_event_rows("AAPL", events, _SD, _SC, bad_daily, **_KW)
            assert caught.value.code == "CONTRACT_MISMATCH"
            assert caught.value.problem.message == message

    # ... while an all-null object column of None is NOT malformed: it is the
    # "no implied quote captured" frame, and every row still computes
    all_null = _DAILY.assign(implied_move=[None] * len(_DAILY))
    assert not pd.api.types.is_numeric_dtype(all_null["implied_move"])
    rows = _build(_events(*_CONFIRMED), daily=all_null)
    assert [row["implied_move_pct"] for row in rows] == [None] * len(_CONFIRMED)


def test_nat_daily_dates_refuse_both_populations_from_build_event_rows():
    # NaT parses without raising, so it needs its own check: it refuses with the
    # ONE unparseable-dates message before any branch, in the 3-event ticker
    # (left alone ticker_too_few_events) and the 16-event scoreable one
    nat_daily = _DAILY.copy()
    nat_daily.loc[1, "date"] = pd.NaT
    assert pd.api.types.is_datetime64_any_dtype(nat_daily["date"])
    assert nat_daily["date"].isna().any()
    for events in (_events(*_CONFIRMED[:3]), _events(*_CONFIRMED)):
        with pytest.raises(DataError) as caught:
            build_event_rows("AAPL", events, _SD, _SC, nat_daily, **_KW)
        assert caught.value.code == "CONTRACT_MISMATCH"
        assert caught.value.problem.message == "daily_market dates are unparseable"


def test_v4_build_rows_refuses_the_same_malformed_daily_frames():
    # the v4 entry point reaches the same _validate_daily, so each malformed
    # field refuses there with exactly the message build_event_rows gives
    bad_dates = pd.DataFrame({
        "date": ["2024-01-02", "not-a-date", "2024-01-04"],
        "implied_move": [1.0, 2.0, 3.0],
        "src_iv": ["orats", "orats", "orats"]})
    bad_implied = _DAILY.assign(implied_move=["bad"] * len(_DAILY))
    events = _events(*_CONFIRMED)
    cases = ((bad_dates, "daily_market dates are unparseable"),
             (bad_implied, "daily_market implied_move must be numeric"))
    for bad_daily, message in cases:
        for call in (
                lambda bad_daily=bad_daily: build_rows(
                    "AAPL", events, _SD, _SC, bad_daily, **_KW),
                lambda bad_daily=bad_daily: build_event_rows(
                    "AAPL", events, _SD, _SC, bad_daily, **_KW)):
            with pytest.raises(DataError) as caught:
                call()
            assert caught.value.code == "CONTRACT_MISMATCH"
            assert caught.value.problem.message == message


def test_object_dtype_implied_move_validates_per_cell_not_by_column_dtype():
    # pd.concat of an all-null object batch and a float batch can yield an
    # object column holding valid floats, so the contract check is per CELL:
    # numbers and None compute normally from both entry points ...
    values = [float(i) for i in range(len(_DAILY))]
    values[12] = None
    mixed = _DAILY.assign(implied_move=pd.Series(values, dtype=object))
    assert not pd.api.types.is_numeric_dtype(mixed["implied_move"])
    events = _events(*_CONFIRMED)
    anchors = [None if i == 12 else float(i) for i in range(9, 25)]
    for daily in (mixed, _DAILY.assign(
            implied_move=pd.Series([None] * len(_DAILY), dtype=object))):
        for call in (
                lambda daily=daily: build_event_rows(
                    "AAPL", events, _SD, _SC, daily, **_KW),
                lambda daily=daily: build_rows("AAPL", events, _SD, _SC, daily, **_KW)):
            rows = call()
            assert len(rows) == len(_CONFIRMED)
            assert all(not row["skipped"] for row in rows)
            assert [row["implied_move_pct"] for row in rows] == (
                anchors if daily is mixed else [None] * len(_CONFIRMED))

    # ... while a string or a bool cell still refuses with the ONE message,
    # out of build_event_rows and build_rows alike
    for bad in ("bad", True):
        bad_daily = _DAILY.assign(implied_move=pd.Series([bad] * len(_DAILY), dtype=object))
        for call in (
                lambda daily=bad_daily: build_event_rows(
                    "AAPL", events, _SD, _SC, daily, **_KW),
                lambda daily=bad_daily: build_rows("AAPL", events, _SD, _SC, daily, **_KW)):
            with pytest.raises(DataError) as caught:
                call()
            assert caught.value.code == "CONTRACT_MISMATCH"
            assert caught.value.problem.message == "daily_market implied_move must be numeric"


def test_bool_complex_and_object_complex_implied_move_refuse_both_populations():
    # a bool or complex COLUMN is never a real-number series (pandas counts
    # both as numeric, so the dtype alone must refuse), and a Python complex
    # CELL in an object column is not one either: each raises the ONE message
    # from build_event_rows, in the 3-event ticker (left alone
    # ticker_too_few_events) and the 16-event scoreable one -- while the float
    # dtype the store hands over still computes
    bad_series = (
        pd.Series([True, False] * (len(_DAILY) // 2), dtype="bool"),
        pd.Series([1.5 + 2.5j] * len(_DAILY), dtype="complex128"),
        pd.Series([1.5 + 2.5j] * len(_DAILY), dtype=object),
    )
    assert pd.api.types.is_bool_dtype(bad_series[0])
    assert pd.api.types.is_complex_dtype(bad_series[1])
    assert bad_series[2].dtype == object and isinstance(bad_series[2].iloc[0], complex)
    for series in bad_series:
        bad_daily = _DAILY.assign(implied_move=series)
        for events in (_events(*_CONFIRMED[:3]), _events(*_CONFIRMED)):
            with pytest.raises(DataError) as caught:
                build_event_rows("AAPL", events, _SD, _SC, bad_daily, **_KW)
            assert caught.value.code == "CONTRACT_MISMATCH"
            assert caught.value.problem.message == "daily_market implied_move must be numeric"

    assert pd.api.types.is_float_dtype(_DAILY["implied_move"])
    rows = _build(_events(*_CONFIRMED))
    assert [row["implied_move_pct"] for row in rows] == [float(i) for i in range(9, 25)]


def test_empty_daily_frames_validate_even_against_empty_inputs():
    # an EMPTY daily frame with the required columns is well-formed: rows still
    # build, with no as-of anchor anywhere (every implied value null)
    empty = pd.DataFrame({"date": pd.to_datetime([]),
                          "implied_move": pd.Series([], dtype="float64"),
                          "src_iv": pd.Series([], dtype="object")})
    rows = _build(_events(*_CONFIRMED), daily=empty)
    assert len(rows) == len(_CONFIRMED)
    assert [row["implied_move_pct"] for row in rows] == [None] * len(_CONFIRMED)

    # an EMPTY daily frame MISSING a column is malformed all the same:
    # CONTRACT_MISMATCH with the one daily-shape message, before any row builds
    with pytest.raises(DataError) as caught:
        _build(_events(*_CONFIRMED), daily=empty.drop(columns="src_iv"))
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert caught.value.problem.message == (
        "daily_market frame lacks date, implied_move or src_iv")

    # an EMPTY events frame does not skip the daily check either: build_event_rows
    # validates daily BEFORE the empty-input return, so the malformed frame wins
    with pytest.raises(DataError) as caught:
        _build(_events(), daily=pd.DataFrame())
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert caught.value.problem.message == (
        "daily_market frame lacks date, implied_move or src_iv")


@pytest.mark.parametrize("bad", [None, pd.NA, float("nan")])
def test_non_bool_src_orats_is_contract_mismatch(bad):
    events = _events(*_CONFIRMED[:3])
    events["src_orats"] = pd.Series([bad, True, True], dtype="object")
    _assert_refusal_parity(events, "events src_orats must hold a real bool per row")


@pytest.mark.parametrize("bad", ["False", 0, 1, np.nan])
def test_non_null_invalid_src_orats_is_contract_mismatch(bad):
    # a truthy-looking string, 0/1 ints and a float NaN are each NOT a real
    # bool verdict, so each refuses from both entry points, same message
    events = _events(*_CONFIRMED[:3])
    events["src_orats"] = pd.Series([bad, True, True], dtype="object")
    _assert_refusal_parity(events, "events src_orats must hold a real bool per row")


@pytest.mark.parametrize("bad", ["PM", "bmo", 5])
def test_session_outside_bmo_amc_is_contract_mismatch(bad):
    # the source schema's set is exactly BMO/AMC nullable: a typo, wrong case
    # or non-string refuses outright from both entry points, same message --
    # it must never reach _session_move_indexed and read as an AMC window
    events = _events(*_CONFIRMED[:3])
    events["session"] = pd.Series([bad, "BMO", "AMC"], dtype="object")
    _assert_refusal_parity(events, "events session must be BMO or AMC")


def test_string_event_date_is_contract_mismatch_from_both_entry_points():
    # the store converts with pd.to_datetime; a caller handing over strings must
    # meet one typed refusal, never an AttributeError from .to_datetime64()
    events = _events(*_CONFIRMED[:3])
    events["event_date"] = pd.Series(
        [day.strftime("%Y-%m-%d") for day, _, _ in _CONFIRMED[:3]], dtype="object")
    assert events["event_date"].dtype == object  # the input really is strings

    _assert_refusal_parity(events, "events event_date must be a datetime column")


def test_nat_event_date_is_contract_mismatch_from_both_entry_points():
    # NaT is a datetime-dtype cell no ordering or as-of lookup can place, so it
    # refuses right after the dtype check -- same code, same text, both entry points
    events = _events(*_CONFIRMED[:3])
    events.loc[1, "event_date"] = pd.NaT
    assert pd.api.types.is_datetime64_any_dtype(events["event_date"])
    assert events["event_date"].isna().any()

    _assert_refusal_parity(events, "events event_date must not hold NaT")


def test_intraday_event_date_is_contract_mismatch_from_both_entry_points():
    # a timestamp with a time component is not a calendar day, so it refuses
    # right after the NaT check -- same code, same text, both entry points
    events = _events(*_CONFIRMED[:3])
    events.loc[1, "event_date"] = events.loc[1, "event_date"] + pd.Timedelta(hours=9, minutes=30)
    assert pd.api.types.is_datetime64_any_dtype(events["event_date"])
    assert not events["event_date"].isna().any()
    assert (events["event_date"] != events["event_date"].dt.normalize()).any()

    _assert_refusal_parity(events, "events event_date must be a calendar day")


def test_reversed_event_dates_refuse_before_the_ticker_level_branch():
    # three confirmed events: left alone this frame would classify
    # ticker_too_few_events, so the ordering refusal has to fire FIRST in both
    # entry points -- a ticker-level verdict must not hide a descending frame
    _assert_refusal_parity(_events(*_CONFIRMED[:3][::-1]),
                           "event dates must be strictly ascending")


def test_reversed_daily_dates_refuse_even_a_ticker_level_refusal():
    # build_event_rows only: classify_event_reasons takes no daily frame to
    # order, and the three-event ticker below would otherwise take the
    # ticker_too_few_events verdict instead of this refusal
    reversed_daily = _DAILY.iloc[::-1].reset_index(drop=True)
    with pytest.raises(DataError) as caught:
        build_event_rows("AAPL", _events(*_CONFIRMED[:3]), _SD, _SC, reversed_daily, **_KW)
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert caught.value.problem.message == "rows must be strictly date-ascending"


def test_two_intraday_daily_timestamps_on_one_calendar_day_refuse_every_branch():
    # ascending TIMESTAMPS on one calendar day are a duplicate day for the
    # %Y-%m-%d as-of projection, so they refuse with the one daily-order
    # message in EVERY population branch: the ticker-level one (3 confirmed,
    # left alone ticker_too_few_events) and the computing one (16 confirmed)
    intraday = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-15 09:30:00", "2024-01-15 16:00:00"]),
        "implied_move": [5.0, 6.0],
        "src_iv": ["orats", "orats"],
    })
    assert intraday["date"].is_monotonic_increasing and intraday["date"].is_unique
    for events in (_events(*_CONFIRMED[:3]), _events(*_CONFIRMED)):
        for call in (
                lambda events=events: build_event_rows(
                    "AAPL", events, _SD, _SC, intraday, **_KW),
                lambda events=events: build_rows("AAPL", events, _SD, _SC, intraday, **_KW)):
            with pytest.raises(DataError) as caught:
                call()
            assert caught.value.code == "CONTRACT_MISMATCH"
            assert caught.value.problem.message == "rows must be strictly date-ascending"


# --------------------------------------------------------------------------
# row provenance: the call inputs, and only the v5 keys, on every row
# --------------------------------------------------------------------------

#: An empty close series: nothing of the ticker's price history exists, so
#: every considered event is ``no_price_history``.
_EMPTY_SD = np.array([], dtype="datetime64[ns]")
_EMPTY_SC = np.array([])

#: ``_mixed_events``' event dates, ISO, position by position.
_MIXED_DATES = (
    "2023-12-29", "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18",
    "2024-01-19", "2024-01-20", "2024-01-21", "2024-01-22", "2024-01-23",
    "2024-01-24", "2024-01-25", "2024-01-26", "2024-01-29", "2024-01-30",
    "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-05",
)

#: ``_CONFIRMED[:3]``'s event dates, ISO: the too-few ticker's rows.
_TOO_FEW_DATES = ("2024-01-15", "2024-01-16", "2024-01-17")


def test_every_row_carries_the_call_inputs_and_the_v5_key_set():
    # (ticker, events, sd, sc, expected ISO dates, expected reasons, provenance)
    # -- three DIFFERENT tickers: a mixed one holding a computed row
    # (null), a no_price_window row, a phantom row and a no_session row, one
    # below the admission bar (a ticker-level row), one with no price history
    # (the other ticker-level row)
    cases = (
        ("AAPL", _mixed_events(), _SD, _SC, _MIXED_DATES, list(_MIXED_REASONS),
         dict(computed_at="2026-09-25T00:00:00+00:00", source_hash="sha256:" + "a" * 64,
              capture_id="capture_test")),
        ("MSFT", _events(*_CONFIRMED[:3]), _SD, _SC, _TOO_FEW_DATES,
         ["ticker_too_few_events"] * len(_TOO_FEW_DATES),
         dict(computed_at="2026-10-02T08:30:00+00:00", source_hash="sha256:" + "b" * 64,
              capture_id="capture_other")),
        ("NVDA", _mixed_events(), _EMPTY_SD, _EMPTY_SC, _MIXED_DATES,
         ["no_price_history"] * len(_MIXED_DATES),
         dict(computed_at="2026-11-11T00:00:00+00:00", source_hash="sha256:" + "c" * 64,
              capture_id="capture_third")),
    )
    seen: set = set()
    for ticker, events, sd, sc, dates, reasons, provenance in cases:
        rows = build_event_rows(ticker, events, sd, sc, _DAILY, **provenance)

        assert len(rows) == len(dates) == len(reasons) > 0
        for row, day, reason in zip(rows, dates, reasons):
            assert set(row) == set(_V5_COLUMNS)
            assert row["ticker"] == ticker
            assert row["event_date"] == day
            assert row["not_computed_reason"] == reason  # exact, per row
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["event_date"])
            assert row["computed_at"] == provenance["computed_at"]
            assert row["source_hash"] == provenance["source_hash"]
            assert row["capture_id"] == provenance["capture_id"]
        seen.update(reasons)

    # the four row kinds the cases were built to cover are all actually present
    assert {None, "no_price_window", "not_orats"} <= seen
    assert "ticker_too_few_events" in seen


# --------------------------------------------------------------------------
# the two admission bars, isolated from each other
# --------------------------------------------------------------------------

#: Six CONFIRMED BMO events on the sessions BEFORE ``_BDAYS[0]`` (2023-12-18
#: through 2023-12-26): every one is unbracketable against ``_SD`` -- there is
#: no earlier close -- so this ticker can hold a confirmed count at or above
#: ``min_scoreable`` while its computable count stays below ``ADMISSION_EVENTS``.
_PRE_CLOSE_DAYS = ("2023-12-18", "2023-12-19", "2023-12-20",
                   "2023-12-21", "2023-12-22", "2023-12-26")
_PRE_CLOSE_EVENTS = [(pd.Timestamp(day), "BMO", True) for day in _PRE_CLOSE_DAYS]

#: The closes that WOULD bracket ``_PRE_CLOSE_EVENTS``: 40 business days ending
#: on the last of those events, with a positive close series, so each event has
#: an earlier close in this series alone.
_PRE_CLOSE_SD = pd.bdate_range("2023-11-01", periods=40).to_numpy(dtype="datetime64[ns]")
_PRE_CLOSE_SC = np.linspace(100.0, 130.0, 40)


def test_confirmed_below_min_scoreable_refuses_while_computable_clears_the_bar():
    events = _events(*_CONFIRMED[:6])  # 6 confirmed, all 6 computable: ADMISSION clears
    rows = _build(events, min_scoreable=12)

    assert len(rows) == 6
    for position, row in enumerate(rows):
        _assert_refused(row, "ticker_too_few_events", position)
    assert classify_event_reasons(events, _SD, _SC, min_scoreable=12) == (
        ["ticker_too_few_events"] * 6)
    # dropping the bar to the confirmed count is the ONLY change: they compute,
    # so the refusal above was min_scoreable and nothing else
    computed = _build(events, min_scoreable=6)
    assert [row["not_computed_reason"] for row in computed] == [None] * 6


def test_computable_below_admission_refuses_even_with_enough_confirmed():
    events = _events(*_PRE_CLOSE_EVENTS)  # 6 confirmed (>= 5), 0 computable (< 5)
    rows = _build(events, min_scoreable=5)

    assert len(rows) == 6
    for position, row in enumerate(rows):
        _assert_refused(row, "ticker_too_few_events", position)
    assert classify_event_reasons(events, _SD, _SC, min_scoreable=5) == (
        ["ticker_too_few_events"] * 6)
    # the SAME events compute once their closes exist: the refusal above was the
    # computable-count bar, not the confirmed count
    computed = _build(events, sd=_PRE_CLOSE_SD, sc=_PRE_CLOSE_SC, min_scoreable=5)
    assert [row["not_computed_reason"] for row in computed] == [None] * 6


def test_empty_price_history_beats_the_too_few_events_bar():
    events = _events(*_CONFIRMED[:3])  # 3 confirmed: below min_scoreable too
    rows = _build(events, sd=_EMPTY_SD, sc=_EMPTY_SC)

    assert len(rows) == 3
    for position, row in enumerate(rows):
        _assert_refused(row, "no_price_history", position)
    assert classify_event_reasons(events, _EMPTY_SD, _EMPTY_SC) == (
        ["no_price_history"] * 3)


def test_admission_boundaries_count_only_computable_and_min_scoreable_separately():
    # a phantom and a no_session event ride along NEITHER frame: neither is
    # confirmed nor computable, so neither counts toward either bar
    filler = [(_BETWEEN, "BMO", False), (_BETWEEN + pd.Timedelta(days=1), None, True)]
    below = _events(*(_CONFIRMED[: ADMISSION_EVENTS - 1] + filler))
    at_bar = _events(*(_CONFIRMED[:ADMISSION_EVENTS] + filler))
    refused_below = ["ticker_too_few_events"] * (ADMISSION_EVENTS + 1)
    refused_at_bar = ["ticker_too_few_events"] * (ADMISSION_EVENTS + 2)
    clears = [None] * ADMISSION_EVENTS + ["not_orats", "no_session"]

    # exactly ADMISSION_EVENTS-1 computable: the admission bar alone refuses,
    # even though confirmed == min_scoreable clears the other one
    assert classify_event_reasons(below, _SD, _SC,
                                  min_scoreable=ADMISSION_EVENTS - 1) == refused_below
    assert [row["not_computed_reason"] for row in
            _build(below, min_scoreable=ADMISSION_EVENTS - 1)] == refused_below

    # exactly ADMISSION_EVENTS computable: rows compute, and the phantom and
    # the no_session event kept their own per-event reasons throughout
    assert classify_event_reasons(at_bar, _SD, _SC,
                                  min_scoreable=ADMISSION_EVENTS - 1) == clears
    assert [row["not_computed_reason"] for row in
            _build(at_bar, min_scoreable=ADMISSION_EVENTS - 1)] == clears

    # the min_scoreable bar alone: confirmed == min_scoreable passes ...
    assert classify_event_reasons(at_bar, _SD, _SC,
                                  min_scoreable=ADMISSION_EVENTS) == clears
    assert [row["not_computed_reason"] for row in
            _build(at_bar, min_scoreable=ADMISSION_EVENTS)] == clears
    # ... and confirmed == min_scoreable - 1 fails
    assert classify_event_reasons(at_bar, _SD, _SC,
                                  min_scoreable=ADMISSION_EVENTS + 1) == refused_at_bar
    assert [row["not_computed_reason"] for row in
            _build(at_bar, min_scoreable=ADMISSION_EVENTS + 1)] == refused_at_bar


def test_reason_precedence_over_one_mixed_population_three_tickers():
    # ONE frame: a phantom, a no_session event and five computable confirmed
    # events; only the ticker's own price history (and the bars) change
    considered = [(_CONFIRMED[0][0], "BMO", False), (_CONFIRMED[1][0], None, True),
                  *_CONFIRMED[2:7]]
    events = _events(*considered)
    per_event = ["not_orats", "no_session"] + [None] * 5

    # empty closes: no_price_history is checked FIRST and overrides everything
    assert classify_event_reasons(events, _EMPTY_SD, _EMPTY_SC, min_scoreable=5) == (
        ["no_price_history"] * len(considered))
    assert [row["not_computed_reason"] for row in
            _build(events, sd=_EMPTY_SD, sc=_EMPTY_SC, min_scoreable=5)] == (
                ["no_price_history"] * len(considered))

    # closes that exist but bracket none of the events: 0 computable <
    # ADMISSION_EVENTS, so ticker_too_few_events overrides the per-event reasons
    far_sd = pd.bdate_range("2024-02-01", periods=10).to_numpy(dtype="datetime64[ns]")
    far_sc = np.linspace(100.0, 110.0, 10)
    too_few = ["ticker_too_few_events"] * len(considered)
    assert classify_event_reasons(events, far_sd, far_sc, min_scoreable=5) == too_few
    assert [row["not_computed_reason"] for row in
            _build(events, sd=far_sd, sc=far_sc, min_scoreable=5)] == too_few

    # the SAME frame on a good ticker: each event keeps its own reason
    assert classify_event_reasons(events, _SD, _SC, min_scoreable=5) == per_event
    assert [row["not_computed_reason"] for row in
            _build(events, min_scoreable=5)] == per_event


# --------------------------------------------------------------------------
# same-date collapse of UNCONFIRMED twins: the FIRST input event's reason
# --------------------------------------------------------------------------


def test_two_same_date_phantoms_collapse_to_one_row_with_the_first_reason():
    base = _build(_events(*_CONFIRMED))
    considered = (*_CONFIRMED[:5], (_BETWEEN, "BMO", False), (_BETWEEN, "AMC", False),
                  *_CONFIRMED[5:])
    rows = _build(_events(*considered))

    assert len(rows) == len(base) + 1  # two unconfirmed events, ONE row
    _assert_refused(rows[5], "not_orats", 5)  # the first input event's reason
    assert rows[5]["event_date"] == "2024-01-20"
    assert rows[:5] == base[:5]
    assert rows[6:] == base[5:]


def test_same_date_phantom_and_no_session_collapse_to_the_first_input_reason():
    base = _build(_events(*_CONFIRMED))
    phantom_first = (*_CONFIRMED[:5], (_BETWEEN, "BMO", False), (_BETWEEN, None, True),
                     *_CONFIRMED[5:])
    session_first = (*_CONFIRMED[:5], (_BETWEEN, None, True), (_BETWEEN, "AMC", False),
                     *_CONFIRMED[5:])

    rows = _build(_events(*phantom_first))
    assert len(rows) == len(base) + 1
    _assert_refused(rows[5], "not_orats", 5)  # the phantom came first
    assert rows[5]["event_date"] == "2024-01-20"

    rows = _build(_events(*session_first))
    assert len(rows) == len(base) + 1
    _assert_refused(rows[5], "no_session", 5)  # the sessionless event came first
    assert rows[5]["event_date"] == "2024-01-20"
    assert rows[:5] == base[:5]
    assert rows[6:] == base[5:]


# --------------------------------------------------------------------------
# the v5 contract's own identity (issue #966)
# --------------------------------------------------------------------------


def test_v5_definition_hash_is_the_placeholder_hash_of_v5():
    v5 = COMPUTED_MOVES_V5_CONTRACT
    placeholder = dataclasses.replace(v5, definition_hash="sha256:" + "0" * 64)

    assert v5.definition_hash == manifests.table_contract_hash(placeholder)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", v5.definition_hash)


def test_v4_definition_hash_is_still_the_pinned_literal():
    v4 = COMPUTED_MOVES_CONTRACT

    assert v4.definition_hash == (
        "sha256:c7f3d2c5a042403e9262bb6d63225c9e2ed6a969dd5b46dcca2813527f9f2d9f")
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", v4.definition_hash)


def test_only_the_six_rewritten_descriptors_differ_from_v4():
    v4, v5 = COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_V5_CONTRACT
    v4_by_name = {column.name: column for column in v4.columns}
    v5_by_name = {column.name: column for column in v5.columns}
    rewritten = {"implied_move_pct", "event_ordinal", "admitted", "skipped",
                 "mean_prior_or_implied", "n_prior_implied"}

    assert [column.name for column in v5.columns] == (
        [column.name for column in v4.columns] + ["not_computed_reason"])
    assert set(v5_by_name) == set(v4_by_name) | {"not_computed_reason"}
    # exactly the six contradicting v4 descriptors moved; every other shared
    # column descriptor is byte-identical to v4's
    assert {name for name, column in v4_by_name.items()
            if v5_by_name[name] != column} == rewritten

    # and those six carry EXACTLY this text -- pinned here rather than read
    # from _V5_TEXT_OVERRIDES, which is the thing under test
    assert v5_by_name["implied_move_pct"].null_policy == (
        "implied_move on the last daily_market row strictly before "
        "event_date whose src_iv is non-null, filled for no_price_window "
        "rows; null when no such row precedes the event or that row's "
        "implied_move is null or non-finite (never an older row); always "
        "null on not_orats, no_session and ticker-level rows")
    assert v5_by_name["event_ordinal"].observation_time_semantics == (
        "0-based count of earlier ORATS-confirmed sessioned events of the "
        "ticker; counts confirmed events only, so unconfirmed/no-outcome "
        "rows carry the count of earlier confirmed events")
    assert v5_by_name["admitted"].null_policy == (
        "event_ordinal >= 4 and the row is a confirmed event on a ticker "
        "that clears the bars; false on every row whose "
        "not_computed_reason is not_orats, no_session or ticker-level")
    assert v5_by_name["skipped"].null_policy == (
        "true exactly when not_computed_reason is non-null: no outcome "
        "was computed for this event")
    assert v5_by_name["mean_prior_or_implied"].null_policy == (
        "mean of the stored implied_move_pct of earlier admitted events, "
        "nulls skipped; null on a non-admitted row, on a "
        "not_orats/no_session/ticker-level row, or when no earlier "
        "admitted event has an implied value")
    assert v5_by_name["n_prior_implied"].null_policy == (
        "count of values behind mean_prior_or_implied; 0 when it is null, "
        "including every not_orats/no_session/ticker-level row")
