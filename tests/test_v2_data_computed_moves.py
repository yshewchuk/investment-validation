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
from engine.v2.data.computed_moves import _session_move_indexed, build_rows, session_move

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


def test_build_rows_skipped_first_event_never_shifts_the_ordinal():
    dates = ["2024-01-01", "2024-01-03", "2024-01-04", "2024-01-05",
             "2024-01-08", "2024-01-09"]
    events = pd.DataFrame({"event_date": pd.to_datetime(dates),
                           "session": ["BMO"] * len(dates)})
    daily = _daily(dates, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    rows = build_rows("AAPL", events, _SESSIONS, _CLOSES, daily,
                      computed_at="2026-09-25T00:00:00+00:00",
                      source_hash="sha256:" + "a" * 64, capture_id="capture_test")

    assert [row["skipped"] for row in rows] == [True] + [False] * 5
    assert rows[0]["quarter_ordinal"] == 0  # the sentinel, never a real ordinal
    assert rows[1]["quarter_ordinal"] == 1  # not shifted by the earlier skip


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


def test_build_rows_available_as_of_date_bmo():
    dates = ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
             "2024-01-09", "2024-02-01"]
    events = pd.DataFrame({"event_date": pd.to_datetime(dates),
                           "session": ["BMO"] * len(dates)})
    daily = _daily(dates, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    rows = build_rows("AAPL", events, _SESSIONS, _CLOSES, daily,
                      computed_at="2026-09-25T00:00:00+00:00",
                      source_hash="sha256:" + "a" * 64, capture_id="capture_test")

    expected = ["2024-01-04", "2024-01-05", "2024-01-06", "2024-01-09", "2024-01-10"]
    computable = [row for row in rows if not row["skipped"]]
    assert [row["available_as_of_date"] for row in computable] == expected
    skipped = [row for row in rows if row["skipped"]]
    assert skipped[0]["available_as_of_date"] is None


def test_build_rows_available_as_of_date_amc():
    dates = ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
             "2024-01-09", "2024-02-01"]
    events = pd.DataFrame({"event_date": pd.to_datetime(dates),
                           "session": ["AMC"] * len(dates)})
    daily = _daily(dates, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    rows = build_rows("AAPL", events, _SESSIONS, _CLOSES, daily,
                      computed_at="2026-09-25T00:00:00+00:00",
                      source_hash="sha256:" + "a" * 64, capture_id="capture_test")

    expected = ["2024-01-05", "2024-01-06", "2024-01-09", "2024-01-10", "2024-01-11"]
    computable = [row for row in rows if not row["skipped"]]
    assert [row["available_as_of_date"] for row in computable] == expected
    skipped = [row for row in rows if row["skipped"]]
    assert skipped[0]["available_as_of_date"] is None
    # realized_move_pct must independently agree with the untouched legacy formula
    for row, date in zip(computable, dates[:5]):
        t = np.datetime64(date)
        assert row["realized_move_pct"] == session_move(_SESSIONS, _CLOSES, t, "AMC")


def test_build_rows_amc_gap_within_limit_still_computes():
    sd = np.array(["2024-02-26", "2024-02-27", "2024-02-28", "2024-03-01",
                   "2024-03-06", "2024-03-07", "2024-03-08"], dtype="datetime64[ns]")
    sc = np.array([48.0, 49.0, 50.0, 52.0, 55.0, 56.0, 57.0])
    dates = ["2024-02-26", "2024-02-27", "2024-02-28", "2024-03-01", "2024-03-06"]
    events = pd.DataFrame({"event_date": pd.to_datetime(dates),
                           "session": ["AMC"] * len(dates)})
    daily = _daily(dates, [1.0, 2.0, 3.0, 4.0, 5.0])

    rows = build_rows("AAPL", events, sd, sc, daily,
                      computed_at="2026-09-25T00:00:00+00:00",
                      source_hash="sha256:" + "a" * 64, capture_id="capture_test")

    assert len(rows) == 5
    assert all(not row["skipped"] for row in rows)
    march1_row = rows[3]  # event_date == "2024-03-01"
    assert march1_row["event_date"] == "2024-03-01"
    # next close after 2024-03-01 AMC is 2024-03-06 -- a 5-calendar-day gap,
    # exactly at MAX_GAP_CALENDAR_DAYS, still bracketable (not "> limit")
    assert march1_row["available_as_of_date"] == "2024-03-07"
    assert march1_row["realized_move_pct"] == session_move(
        sd, sc, np.datetime64("2024-03-01"), "AMC")


def test_build_rows_gap_exceeds_limit_is_skipped_and_null():
    # Note: sd/sc carry one more session (2024-03-19) than `events` does --
    # 2024-03-18's own "next close" lookup needs it to exist, but 2024-03-19
    # is deliberately NOT one of the events, because it is sd's last entry
    # and would itself be skipped as "no post session" (a different skip
    # reason than the one this test targets), which would throw off the
    # computable count below.
    sd = np.array(["2024-02-26", "2024-02-27", "2024-02-28", "2024-03-01",
                   "2024-03-15", "2024-03-18", "2024-03-19"], dtype="datetime64[ns]")
    sc = np.array([48.0, 49.0, 50.0, 52.0, 70.0, 71.0, 72.0])
    dates = ["2024-02-26", "2024-02-27", "2024-02-28", "2024-03-01",
             "2024-03-15", "2024-03-18"]
    events = pd.DataFrame({"event_date": pd.to_datetime(dates),
                           "session": ["AMC"] * len(dates)})
    daily = _daily(dates, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    rows = build_rows("AAPL", events, sd, sc, daily,
                      computed_at="2026-09-25T00:00:00+00:00",
                      source_hash="sha256:" + "a" * 64, capture_id="capture_test")

    by_date = {row["event_date"]: row for row in rows}
    march1_row = by_date["2024-03-01"]
    # next close after 2024-03-01 AMC is 2024-03-15 -- a 14-calendar-day gap,
    # over MAX_GAP_CALENDAR_DAYS (5) -- must be skipped and null, never guessed
    assert march1_row["skipped"] is True
    assert march1_row["realized_move_pct"] is None
    assert march1_row["available_as_of_date"] is None
    # the other 5 events are ordinary 1-3 day gaps and must still compute
    assert len(rows) == 6
    assert sum(1 for row in rows if not row["skipped"]) == 5


def test_available_as_of_date_assertion_catches_a_planted_defect():
    """Negative control for the BMO/AMC available_as_of_date tests above:
    prove the comparison style they use actually rejects a wrong value,
    not just that it happens to pass on correct ones."""
    dates = ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
             "2024-01-09", "2024-02-01"]
    events = pd.DataFrame({"event_date": pd.to_datetime(dates),
                           "session": ["BMO"] * len(dates)})
    daily = _daily(dates, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    rows = build_rows("AAPL", events, _SESSIONS, _CLOSES, daily,
                      computed_at="2026-09-25T00:00:00+00:00",
                      source_hash="sha256:" + "a" * 64, capture_id="capture_test")
    computable = [row for row in rows if not row["skipped"]]
    correct = [row["available_as_of_date"] for row in computable]

    # Plant a defect: shift one computable row's date by a day, and null out
    # another computable row's date (the two wrong shapes a real regression
    # could take). Both must be REJECTED by the same comparison style
    # test_build_rows_available_as_of_date_bmo uses -- if this assertion
    # ever silently passed, that test's own assertions would be worthless.
    corrupted_shifted = list(correct)
    corrupted_shifted[0] = str(
        (pd.Timestamp(corrupted_shifted[0]) + pd.Timedelta(days=1)).date())
    assert corrupted_shifted != correct

    corrupted_null = list(correct)
    corrupted_null[1] = None
    assert corrupted_null != correct

    # And the real, uncorrupted output must still match itself exactly --
    # confirming the negative control above is about the PLANTED defect,
    # not about the comparison being trivially always-unequal.
    assert correct == [row["available_as_of_date"] for row in computable]
