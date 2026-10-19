"""``computed_moves.v5`` rows (issue #966): one row per considered event, with a reason.

An event the v4 builder silently dropped -- a ``forward_claim`` phantom, an
ORATS row without a session -- is now a row whose ``not_computed_reason`` says
why nothing was computed, and it must not disturb its confirmed neighbours'
``event_ordinal`` or ``mean_prior_or_implied``. The two ticker-level reasons
reach every row of the ticker. A confirmed row is the v4 row plus the reason,
and every considered event gets exactly one row (never ``[]``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.v2.data.computed_moves import build_event_rows, build_rows

_BDAYS = pd.bdate_range("2024-01-01", periods=60)
_SD = _BDAYS.to_numpy(dtype="datetime64[ns]")
_SC = np.linspace(100.0, 130.0, len(_BDAYS))
_DAILY = pd.DataFrame({"date": _BDAYS,
                       "implied_move": [float(i) for i in range(len(_BDAYS))],
                       "src_iv": ["orats"] * len(_BDAYS)})
_KW = dict(computed_at="2026-09-25T00:00:00+00:00", source_hash="sha256:" + "a" * 64,
           capture_id="capture_test")

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


def _build(events, sd=_SD, sc=_SC):
    return build_event_rows("AAPL", events, sd, sc, _DAILY, **_KW)


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