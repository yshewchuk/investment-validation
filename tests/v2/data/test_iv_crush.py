"""Signed native labels: synthetic anchors, omissions and provenance only."""
from __future__ import annotations

import copy
import math

import pandas as pd
import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.iv_crush import build_rows
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT


def build(events=None, daily=None, **kwargs):
    return build_rows("TEST", events if events is not None else [
        {"event_date": "2024-01-05", "session": "AMC"}],
        daily if daily is not None else [
            {"date": "2024-01-04", "iv30": 40.0},
            {"date": "2024-01-05", "iv30": 50.0},
            {"date": "2024-01-08", "iv30": 30.0}],
        source_id="synthetic-source", **kwargs)


@pytest.mark.parametrize("session,pre,post,value,available", [
    ("AMC", "2024-01-05", "2024-01-08", -40.0, "2024-01-09"),
    ("BMO", "2024-01-04", "2024-01-05", 25.0, "2024-01-06"),
])
def test_explicit_anchors_and_signed_relative_change(session, pre, post, value, available):
    row = build(events=[{"event_date": "2024-01-05", "session": session}])[0]
    assert (row["pre_date"], row["post_date"]) == (pre, post)
    assert row["crush_pct_iv30"] == pytest.approx(value)
    assert row["available_as_of_date"] == available
    assert row["skipped"] is False and row["skip_reason"] is None
    # The native table order and builder projection cannot silently diverge.
    assert list(row) == [c.name for c in IV_CRUSH_CONTRACT.columns]


@pytest.mark.parametrize("after,expected", [(0.0, -100.0), (-5.0, -110.0), (75.0, 50.0)])
def test_nonpositive_post_and_positive_change_preserve_existing_target(after, expected):
    row = build(daily=[{"date": "2024-01-05", "iv30": 50.0},
                       {"date": "2024-01-08", "iv30": after}])[0]
    assert row["crush_pct_iv30"] == pytest.approx(expected)
    assert not row["skipped"]  # Deliberately no new post-positive filter or clipping.


@pytest.mark.parametrize("post,reason", [("2024-01-10", None), ("2024-01-11", "gap_exceeds_limit")])
def test_five_day_weekend_holiday_gap_boundary(post, reason):
    row = build(daily=[{"date": "2024-01-05", "iv30": 50.0},
                       {"date": post, "iv30": 40.0}])[0]
    assert row["skip_reason"] == reason
    assert row["post_date"] == post
    assert row["pre_iv30"] == 50.0 and row["post_iv30"] == 40.0
    assert (row["crush_pct_iv30"] is None) == (reason is not None)
    assert (row["available_as_of_date"] is None) == (reason is not None)


@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), float("inf"), -float("inf"), True, "50"])
def test_invalid_pre_never_becomes_zero_label_or_fallback(value):
    row = build(daily=[{"date": "2024-01-04", "iv30": 70.0},
                       {"date": "2024-01-05", "iv30": value},
                       {"date": "2024-01-08", "iv30": 30.0}])[0]
    assert row["skip_reason"] == "invalid_pre_iv30"
    assert row["pre_date"] == "2024-01-05"
    assert row["crush_pct_iv30"] is None and row["available_as_of_date"] is None


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), False, "30"])
def test_invalid_post_does_not_search_later_valid_quote(value):
    row = build(daily=[{"date": "2024-01-05", "iv30": 50.0},
                       {"date": "2024-01-08", "iv30": value},
                       {"date": "2024-01-09", "iv30": 30.0}])[0]
    assert row["skip_reason"] == "invalid_post_iv30"
    assert row["post_date"] == "2024-01-08"
    assert row["crush_pct_iv30"] is None and row["available_as_of_date"] is None


def test_nonfinite_ratio_is_explicit_skip():
    row = build(daily=[{"date": "2024-01-05", "iv30": 1e-300},
                       {"date": "2024-01-08", "iv30": 1e300}])[0]
    assert row["skip_reason"] == "nonfinite_target"
    assert row["available_as_of_date"] is None


@pytest.mark.parametrize("daily,reason", [([], "missing_pre"),
    ([{"date": "2024-01-08", "iv30": 30}], "missing_pre"),
    ([{"date": "2024-01-05", "iv30": 50}], "missing_post")])
def test_missing_anchors_retain_event(daily, reason):
    rows = build(daily=daily)
    assert len(rows) == 1
    assert rows[0]["skip_reason"] == reason
    assert rows[0]["crush_pct_iv30"] is None
    assert rows[0]["available_as_of_date"] is None


@pytest.mark.parametrize("session,reason", [(None, "missing_session"), ("UNK", "unknown_session"),
                                           ("amc", "unknown_session"), (False, "unknown_session")])
def test_unknown_session_is_not_silently_bmo(session, reason):
    row = build(events=[{"event_date": "2024-01-05", "session": session}])[0]
    assert row["session"] is None and row["skip_reason"] == reason
    assert row["pre_date"] is None and row["post_date"] is None


@pytest.mark.parametrize("kwargs", [
    {"events": [{"event_date": "20240105", "session": "AMC"}]},
    {"events": [{"event_date": "2024-01-05T00:00:00", "session": "AMC"}]},
    {"events": [{"event_date": "2024-02-30", "session": "AMC"}]},
    {"events": [{"event_date": "2024-01-05"}]},
    {"events": [{"event_date": "2024-01-05", "session": "AMC", "ticker": "OTHER"}]},
    {"daily": [{"date": "2024-01-05", "iv30": object()}]},
    {"daily": [{"date": "2024-01-05", "iv30": 30, "ticker": "OTHER"}]},
    {"max_rows": True}, {"max_rows": 0},
])
def test_malformed_identity_or_shape_refuses_without_echoing_input(kwargs):
    with pytest.raises(DataError) as exc:
        build(**kwargs)
    assert exc.value.code == "CONTRACT_MISMATCH"
    assert "OTHER" not in str(exc.value)


@pytest.mark.parametrize("which", ["events", "daily"])
def test_duplicate_keys_refuse_even_when_equal(which):
    row = {"event_date": "2024-01-05", "session": "AMC"} if which == "events" else {
        "date": "2024-01-05", "iv30": 50}
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        build(**{which: [row, row.copy()]})


def test_row_bounds_do_not_truncate_silently():
    with pytest.raises(DataError, match="RESULT_LIMIT_EXCEEDED"):
        build(max_rows=2)
    assert len(build(max_rows=3)) == 1
    assert build(events=[]) == []


def test_event_row_cap_refuses_before_a_partial_result():
    events = ({"event_date": f"2024-01-{day:02}", "session": "AMC"} for day in range(1, 5))
    with pytest.raises(DataError, match="RESULT_LIMIT_EXCEEDED"):
        build(events=events, daily=[], max_rows=2)


@pytest.mark.parametrize("session", ["AMC", "BMO"])
def test_missing_event_day_uses_supplied_adjacent_rows(session):
    row = build(events=[{"event_date": "2024-01-06", "session": session}],
                daily=[{"date": "2024-01-05", "iv30": 50},
                       {"date": "2024-01-08", "iv30": 30}])[0]
    assert (row["pre_date"], row["post_date"], row["crush_pct_iv30"]) == (
        "2024-01-05", "2024-01-08", -40.0)


def test_invalid_scalars_remain_distinguishable_in_input_hash():
    hashes = set()
    for value in (None, float("nan"), float("inf"), "bad", False):
        row = build(daily=[{"date": "2024-01-05", "iv30": value},
                           {"date": "2024-01-08", "iv30": 30}])[0]
        hashes.add(row["input_hash"])
        assert row["skip_reason"] == "invalid_pre_iv30"
    assert len(hashes) == 5


def test_deterministic_projection_identity_and_no_mutation():
    events = [{"event_date": "2024-01-05", "session": "AMC"},
              {"event_date": "2024-01-08", "session": "BMO"}]
    daily = [{"date": "2024-01-05", "iv30": 50.0}, {"date": "2024-01-08", "iv30": 30.0}]
    original = copy.deepcopy((events, daily))
    first = build(events, daily)
    assert build(list(reversed(events)), list(reversed(daily))) == first
    assert (events, daily) == original
    assert len({r["input_hash"] for r in first}) == 1
    daily[1]["iv30"] = 31.0
    assert build(events, daily)[0]["input_hash"] != first[0]["input_hash"]
    assert first[0]["source_id"] == "synthetic-source"


def test_same_relative_target_for_different_iv_scale():
    for scale in (0.01, 1, 100):
        row = build(daily=[{"date": "2024-01-05", "iv30": scale * 50},
                           {"date": "2024-01-08", "iv30": scale * 30}])[0]
        assert row["crush_pct_iv30"] == pytest.approx(-40)


def test_legacy_arithmetic_parity_from_synthetic_rows_only():
    from engine.models.training.iv_crush import crush_frame

    events = [{"event_date": "2024-01-05", "session": "AMC"},
              {"event_date": "2024-01-08", "session": "BMO"},
              {"event_date": "2024-01-09", "session": "AMC"}]
    for post in (-5.0, 0.0, 30.0, float("nan")):
        daily = [{"date": "2024-01-05", "iv30": 50.0},
                 {"date": "2024-01-08", "iv30": post},
                 {"date": "2024-01-09", "iv30": 30.0},
                 {"date": "2024-01-10", "iv30": 20.0}]
        event_frame, daily_frame = pd.DataFrame(events), pd.DataFrame(daily)
        event_frame["ticker"] = daily_frame["ticker"] = "TEST"
        daily_frame["date"] = pd.to_datetime(daily_frame.date)
        for column in ("iv10", "exern_iv10", "exern_iv30"):
            daily_frame[column] = 1.0
        legacy = crush_frame(event_frame, daily_frame)
        for actual, expected in zip(build(events, daily), legacy.crush_pct_iv30):
            assert (actual["crush_pct_iv30"] is None) == math.isnan(expected)
            if not math.isnan(expected):
                assert actual["crush_pct_iv30"] == expected
