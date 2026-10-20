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
from engine.v2.data.computed_moves import build_event_rows, build_rows, classify_event_reasons
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
    assert neighbour["mean_prior_or_implied"] is not None  # the comparison is not vacuous
    assert neighbour["event_ordinal"] == 5 == base[5]["event_ordinal"]


def test_confirmed_event_without_a_session_is_no_session():
    considered = _CONFIRMED[:5] + [(_BETWEEN, None, True)] + _CONFIRMED[5:]
    rows = _build(_events(*considered))

    _assert_refused(rows[5], "no_session", 5)
    assert rows[5]["event_date"] == "2024-01-20"
    assert rows[:5] + rows[6:] == _build(_events(*_CONFIRMED))


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


def test_two_same_date_confirmed_rows_collapse_to_the_first_in_input_order():
    base = _build(_events(*_CONFIRMED))
    day = _CONFIRMED[4][0]  # 2024-01-19 BMO, duplicated as AMC right after it
    considered = [*_CONFIRMED[:5], (day, "AMC", True), *_CONFIRMED[5:]]
    rows = _build(_events(*considered))

    assert len(rows) == len(base)  # two confirmed, one row: the input's first wins
    assert rows == base  # and it is the BMO row, not the AMC twin


def test_empty_event_list_returns_no_rows():
    empty = _events()

    assert _build(empty) == []
    assert classify_event_reasons(empty, _SD, _SC) == []


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
        "exactly when skipped is true and realized_move_pct and "
        "available_as_of_date are null. Closed reason set: not_orats (no ORATS "
        "confirmation, e.g. a forward_claim phantom), no_session (ORATS-confirmed "
        "but session null), no_price_window (the v4 skipped case: a missing close "
        "on either side or a P->Q window wider than MAX_GAP_CALENDAR_DAYS), "
        "no_price_history (the ticker's close series is empty), "
        "ticker_too_few_events (fewer than MIN_SCOREABLE ORATS-confirmed sessioned "
        "events, or fewer than ADMISSION_EVENTS computable ones); the last two are "
        "ticker-level and apply to every row of that ticker.")
    assert v5.primary_key == v4.primary_key == ("ticker", "event_date")
    assert v5.partition_columns == v4.partition_columns == ("ticker",)
    identity = ("contract_id", "definition_hash", "semantic_version", "columns")
    shared = [field.name for field in dataclasses.fields(v4) if field.name not in identity]
    assert [name for name in shared if getattr(v5, name) != getattr(v4, name)] == []


@pytest.mark.parametrize("bad", [True, -1, None])
def test_invalid_min_scoreable_is_contract_mismatch(bad):
    _assert_refusal_parity(_events(*_CONFIRMED[:3]),
                           "min_scoreable must be a non-negative int", min_scoreable=bad)


def test_missing_event_columns_or_a_price_length_mismatch_refuse():
    _assert_refusal_parity(pd.DataFrame({"event_date": []}),
                           "events lacks required columns: session, src_orats")
    _assert_refusal_parity(_events(*_CONFIRMED[:3]).drop(columns="session"),
                           "events lacks required columns: session")
    _assert_refusal_parity(_events(*_CONFIRMED[:3]),
                           "price series sd and sc differ in length", sc=_SC[:5])


def test_a_malformed_daily_frame_refuses_even_a_ticker_level_refusal():
    # checked BEFORE branching: the ticker-level reason below cannot mask it,
    # and an empty frame missing the columns is malformed all the same
    for bad_daily in (_DAILY.drop(columns="src_iv"), pd.DataFrame()):
        with pytest.raises(DataError) as caught:
            build_event_rows("AAPL", _events(*_CONFIRMED[:3]), _SD, _SC, bad_daily, **_KW)
        assert caught.value.code == "CONTRACT_MISMATCH"
        assert caught.value.problem.message == (
            "daily_market frame lacks date, implied_move or src_iv")


@pytest.mark.parametrize("bad", [None, pd.NA, float("nan")])
def test_non_bool_src_orats_is_contract_mismatch(bad):
    events = _events(*_CONFIRMED[:3])
    events["src_orats"] = pd.Series([bad, True, True], dtype="object")
    _assert_refusal_parity(events, "events src_orats must hold a real bool per row")


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
    # (events, sd, sc, expected ISO dates, provenance) -- a mixed ticker, one
    # below the admission bar, and one with no price history at all
    cases = (
        (_mixed_events(), _SD, _SC, _MIXED_DATES,
         dict(computed_at="2026-09-25T00:00:00+00:00", source_hash="sha256:" + "a" * 64,
              capture_id="capture_test")),
        (_events(*_CONFIRMED[:3]), _SD, _SC, _TOO_FEW_DATES,
         dict(computed_at="2026-10-02T08:30:00+00:00", source_hash="sha256:" + "b" * 64,
              capture_id="capture_other")),
        (_mixed_events(), _EMPTY_SD, _EMPTY_SC, _MIXED_DATES,
         dict(computed_at="2026-11-11T00:00:00+00:00", source_hash="sha256:" + "c" * 64,
              capture_id="capture_third")),
    )
    for events, sd, sc, dates, provenance in cases:
        rows = build_event_rows("AAPL", events, sd, sc, _DAILY, **provenance)

        assert len(rows) == len(dates) > 0
        for row, day in zip(rows, dates):
            assert set(row) == set(_V5_COLUMNS)
            assert row["ticker"] == "AAPL"
            assert row["event_date"] == day
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["event_date"])
            assert row["computed_at"] == provenance["computed_at"]
            assert row["source_hash"] == provenance["source_hash"]
            assert row["capture_id"] == provenance["capture_id"]


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


def test_only_the_five_rewritten_descriptors_differ_from_v4():
    v4, v5 = COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_V5_CONTRACT
    v4_by_name = {column.name: column for column in v4.columns}
    v5_by_name = {column.name: column for column in v5.columns}
    rewritten = {"event_ordinal", "admitted", "skipped",
                 "mean_prior_or_implied", "n_prior_implied"}

    assert [column.name for column in v5.columns] == (
        [column.name for column in v4.columns] + ["not_computed_reason"])
    assert set(v5_by_name) == set(v4_by_name) | {"not_computed_reason"}
    # exactly the five contradicting v4 descriptors moved; every other shared
    # column descriptor is byte-identical to v4's
    assert {name for name, column in v4_by_name.items()
            if v5_by_name[name] != column} == rewritten