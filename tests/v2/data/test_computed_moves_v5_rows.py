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

import numpy as np
import pandas as pd
import pytest

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
        "no_price_history (the ticker has no daily_market rows or usable closes), "
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
    events = _events(*_CONFIRMED[:3])
    calls = (
        lambda: classify_event_reasons(events, _SD, _SC, min_scoreable=bad),
        lambda: build_event_rows("AAPL", events, _SD, _SC, _DAILY, min_scoreable=bad, **_KW),
    )

    for call in calls:
        with pytest.raises(DataError) as caught:
            call()
        assert caught.value.code == "CONTRACT_MISMATCH"
        assert caught.value.problem.message == "min_scoreable must be a non-negative int"


def test_missing_event_columns_or_a_price_length_mismatch_refuse():
    for frame in (pd.DataFrame({"event_date": []}),
                  _events(*_CONFIRMED[:3]).drop(columns="session")):
        with pytest.raises(DataError) as caught:
            classify_event_reasons(frame, _SD, _SC)
        assert caught.value.code == "CONTRACT_MISMATCH"

    with pytest.raises(DataError) as caught:
        build_event_rows("AAPL", _events(*_CONFIRMED[:3]), _SD, _SC[:5], _DAILY, **_KW)
    assert caught.value.code == "CONTRACT_MISMATCH"


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
    calls = (
        lambda: classify_event_reasons(events, _SD, _SC),
        lambda: build_event_rows("AAPL", events, _SD, _SC, _DAILY, **_KW),
    )

    for call in calls:
        with pytest.raises(DataError) as caught:
            call()
        assert caught.value.code == "CONTRACT_MISMATCH"
        assert caught.value.problem.message == "events src_orats must hold a real bool per row"