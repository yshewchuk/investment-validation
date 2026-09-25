"""Spec s4b Change 2: the pure computed-move row builder.

``session_move`` is asserted identical to the untouched legacy pull's own
function; the row builder must keep every skipped event visible rather than
silently dropped (EXP-117's materiality rule), and must apply the same
"fewer than five computable events" admission rule the legacy
``build_ticker`` enforces by returning ``None``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data.pulls import computed_moves as legacy
from engine.v2.data.computed_moves import build_rows, session_move

_SESSIONS = np.array(
    ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
     "2024-01-09", "2024-01-10", "2024-01-11"],
    dtype="datetime64[ns]",
)
_CLOSES = np.array([100.0, 102.0, 101.0, 110.0, 105.0, 108.0, 104.0, 112.0])


def _daily(dates, implied):
    return pd.DataFrame({"date": pd.to_datetime(dates), "implied_move": implied})


def test_session_move_identical_to_legacy():
    for session in ("BMO", "AMC"):
        for event_date in ("2024-01-03", "2024-01-05", "2024-01-08"):
            t = np.datetime64(event_date)
            assert session_move(_SESSIONS, _CLOSES, t, session) == legacy.session_move(
                _SESSIONS, _CLOSES, t, session)


def test_build_rows_include_skipped_events():
    dates = ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
             "2024-01-09", "2024-02-01"]
    events = pd.DataFrame({"event_date": pd.to_datetime(dates),
                           "session": ["BMO"] * len(dates)})
    daily = _daily(dates, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    rows = build_rows("AAPL", events, _SESSIONS, _CLOSES, daily,
                      computed_at="2026-09-25T00:00:00+00:00",
                      source_hash="sha256:" + "a" * 64, capture_id="capture_test")

    assert len(rows) == len(events)  # never minus the skipped one
    skipped = [row for row in rows if row["skipped"]]
    assert len(skipped) == 1
    assert skipped[0]["realized_move_pct"] is None
    assert skipped[0]["implied_move_pct"] is None
    assert skipped[0]["event_date"] == "2024-02-01"
    assert all(row["capture_id"] == "capture_test" for row in rows)


def test_build_rows_empty_below_five_computable():
    dates = ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    events = pd.DataFrame({"event_date": pd.to_datetime(dates),
                           "session": ["BMO"] * len(dates)})
    daily = _daily(dates, [1.0, 2.0, 3.0, 4.0])

    # negative control: the legacy admission rule is the same one, ported
    assert legacy.build_ticker("AAPL", events, _SESSIONS, _CLOSES, daily) is None
    assert build_rows("AAPL", events, _SESSIONS, _CLOSES, daily,
                      computed_at="2026-09-25T00:00:00+00:00",
                      source_hash="sha256:" + "a" * 64, capture_id="capture_test") == []
