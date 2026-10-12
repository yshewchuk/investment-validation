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
import pytest

from engine.data.pulls import computed_moves as legacy
from engine.v2.data.computed_moves import _session_move_indexed, build_rows, session_move
from engine.v2.data.errors import DataError

_SESSIONS = np.array(
    ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
     "2024-01-09", "2024-01-10", "2024-01-11"],
    dtype="datetime64[ns]",
)
_CLOSES = np.array([100.0, 102.0, 101.0, 110.0, 105.0, 108.0, 104.0, 112.0])


def _daily(dates, implied, src_iv="orats"):
    return pd.DataFrame({"date": pd.to_datetime(dates), "implied_move": implied,
                         "src_iv": [src_iv] * len(dates)})


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
    assert skipped[0]["implied_move_pct"] == 5.0
    assert skipped[0]["event_date"] == "2024-02-01"
    assert all(row["capture_id"] == "capture_test" for row in rows)


def test_build_rows_skipped_first_event_still_counts_in_the_ordinal():
    dates = ["2024-01-01", "2024-01-03", "2024-01-04", "2024-01-05",
             "2024-01-08", "2024-01-09"]
    events = pd.DataFrame({"event_date": pd.to_datetime(dates),
                           "session": ["BMO"] * len(dates)})
    daily = _daily(dates, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    rows = build_rows("AAPL", events, _SESSIONS, _CLOSES, daily,
                      computed_at="2026-09-25T00:00:00+00:00",
                      source_hash="sha256:" + "a" * 64, capture_id="capture_test")

    assert [row["skipped"] for row in rows] == [True] + [False] * 5
    assert [row["event_ordinal"] for row in rows] == [0, 1, 2, 3, 4, 5]  # a skip counts
    assert "quarter_ordinal" not in rows[0]


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


# --- event-history columns (computed_moves.v4, design #651) -------------------

_KW = dict(computed_at="2026-09-25T00:00:00+00:00", source_hash="sha256:" + "a" * 64,
           capture_id="capture_test")
_BDAYS = pd.bdate_range("2024-01-01", periods=60)


def _hist_build(event_idx, daily, drop_idx=()):
    """Real build_rows on a 60-business-day series; events/holes given as bday indexes."""
    keep = [i for i in range(len(_BDAYS)) if i not in set(drop_idx)]
    sd = _BDAYS[keep].to_numpy(dtype="datetime64[ns]")
    sc = np.linspace(100.0, 130.0, len(keep))
    events = pd.DataFrame({"event_date": _BDAYS[list(event_idx)],
                           "session": ["BMO"] * len(event_idx)})
    return build_rows("AAPL", events, sd, sc, daily, **_KW)


def _daily_idx(upto, overrides=None):
    """One daily row per business day 0..upto-1, implied_move == index, src_iv set."""
    daily = _daily(_BDAYS[:upto], [float(i) for i in range(upto)])
    for i, (implied, src) in (overrides or {}).items():
        daily.loc[i, "implied_move"] = implied
        daily.loc[i, "src_iv"] = src
    return daily


def test_ordinal_admission_and_prior_mean_exclude_the_warmup_events():
    rows = _hist_build(range(10, 17), _daily_idx(40))
    assert [r["event_ordinal"] for r in rows] == [0, 1, 2, 3, 4, 5, 6]
    assert [r["admitted"] for r in rows] == [False] * 4 + [True] * 3
    # the event at bday index k anchors on the row at k-1, whose implied_move is k-1
    assert [r["implied_move_pct"] for r in rows] == [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    got = [(r["mean_prior_or_implied"], r["n_prior_implied"]) for r in rows]
    # the first four events are never history: ordinal 4 has none, ordinal 5 sees only 13.0
    assert got == [(None, 0)] * 5 + [(13.0, 1), (13.5, 2)]


def test_skipped_event_keeps_its_implied_and_feeds_the_aggregate():
    # dropping bdays 14..16 leaves a 6-day price gap, so events 14, 15, 16 cannot be bracketed
    rows = _hist_build([10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21], _daily_idx(40),
                       drop_idx=(14, 15, 16))
    assert [r["skipped"] for r in rows[:8]] == [False] * 4 + [True] * 3 + [False]
    assert rows[4]["realized_move_pct"] is None
    assert rows[4]["implied_move_pct"] == 13.0
    assert rows[7]["event_ordinal"] == 7
    assert (rows[7]["mean_prior_or_implied"], rows[7]["n_prior_implied"]) == (14.0, 3)


def test_implied_anchor_ignores_a_newer_row_without_src_iv():
    daily = _daily_idx(40, {13: (99.0, None)})
    assert _hist_build(range(10, 15), daily)[4]["implied_move_pct"] == 12.0


def test_latest_src_iv_row_with_a_null_quote_gives_null_not_an_older_quote():
    daily = _daily_idx(40, {13: (float("nan"), "orats")})
    assert _hist_build(range(10, 15), daily)[4]["implied_move_pct"] is None


def test_no_eligible_row_gives_null_implied_and_a_null_aggregate():
    rows = _hist_build(range(10, 16), _daily(_BDAYS[:40], [1.0] * 40, src_iv=None))
    assert all(r["implied_move_pct"] is None for r in rows)
    assert all((r["mean_prior_or_implied"], r["n_prior_implied"]) == (None, 0) for r in rows)
    empty = pd.DataFrame(columns=["date", "implied_move", "src_iv"])
    assert _hist_build(range(10, 16), empty) == rows


def test_retrieved_at_never_changes_the_output():
    base = _daily_idx(40)
    first = base.assign(retrieved_at=[f"2026-01-{1 + i % 28:02d}" for i in range(40)])
    second = base.assign(retrieved_at=[f"2026-01-{28 - i % 28:02d}" for i in range(40)])
    assert _hist_build(range(10, 17), first) == _hist_build(range(10, 17), second)
    assert _hist_build(range(10, 17), first) == _hist_build(range(10, 17), base)


@pytest.mark.parametrize("event_idx", [[10, 11, 12, 12, 13, 14], [10, 12, 11, 13, 14, 15]])
def test_duplicate_or_unsorted_event_dates_refuse_with_contract_mismatch(event_idx):
    with pytest.raises(DataError) as err:
        _hist_build(event_idx, _daily_idx(40))
    assert err.value.code == "CONTRACT_MISMATCH"


def test_duplicate_or_unsorted_daily_dates_refuse_with_contract_mismatch():
    base = _daily_idx(40)
    for bad in (pd.concat([base, base.iloc[[5]]], ignore_index=True), base.iloc[::-1]):
        with pytest.raises(DataError) as err:
            _hist_build(range(10, 16), bad)
        assert err.value.code == "CONTRACT_MISMATCH"


def test_daily_frame_without_src_iv_refuses_with_contract_mismatch():
    with pytest.raises(DataError) as err:
        _hist_build(range(10, 16), _daily_idx(40).drop(columns=["src_iv"]))
    assert err.value.code == "CONTRACT_MISMATCH"


def test_events_before_the_first_close_count_in_the_ordinal_as_skipped_rows():
    # price history starts at business day 10; events at 5 and 7 predate it
    rows = _hist_build([5, 7, 12, 13, 14, 15, 16, 17], _daily_idx(40), drop_idx=range(10))
    assert [r["event_ordinal"] for r in rows] == list(range(8))
    assert [r["admitted"] for r in rows] == [False] * 4 + [True] * 4
    assert [r["skipped"] for r in rows] == [True, True] + [False] * 6
    assert all(r["realized_move_pct"] is None and r["available_as_of_date"] is None
               for r in rows[:2])
    # skipped pre-history events keep their as-of implied value (anchor = previous business day)
    assert [r["implied_move_pct"] for r in rows[:2]] == [4.0, 6.0]
    # ordinal 4 is the first admitted event: nothing earlier is admitted
    assert (rows[4]["mean_prior_or_implied"], rows[4]["n_prior_implied"]) == (None, 0)
    assert (rows[5]["mean_prior_or_implied"], rows[5]["n_prior_implied"]) == (13.0, 1)
