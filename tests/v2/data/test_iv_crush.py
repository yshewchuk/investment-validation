"""Signed native labels: synthetic anchors, omissions and provenance only."""
from __future__ import annotations

import copy
import hashlib
import json
import math

import pandas as pd
import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.iv_crush import build_rows
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT


def assert_refusal(error, code="CONTRACT_MISMATCH"):
    message, category = {
        "CONTRACT_MISMATCH": ("IV-crush input is malformed or ambiguous", "validation"),
        "RESULT_LIMIT_EXCEEDED": ("IV-crush input exceeds its row bound", "resource"),
    }[code]
    problem = error.problem
    assert (problem.code, problem.message, problem.category, problem.retryable) == (
        code, message, category, False)
    assert problem.details == {} and problem.stage is None
    assert str(error) == f"{code}: {message}"


def assert_skip(row, reason, *, session="AMC", pre=None, post=None, pre_iv=None, post_iv=None):
    assert row["ticker"] == "TEST" and row["event_date"] == "2024-01-05"
    assert (row["skipped"], row["skip_reason"], row["session"]) == (True, reason, session)
    assert row["crush_pct_iv30"] is None and row["available_as_of_date"] is None
    assert (row["pre_date"], row["post_date"], row["pre_iv30"], row["post_iv30"]) == (
        pre, post, pre_iv, post_iv)
    assert row["source_id"] == "synthetic-source"
    assert row["input_hash"].startswith("sha256:") and len(row["input_hash"]) == 71


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
    if reason is None:
        assert row["skipped"] is False
        assert row["crush_pct_iv30"] == pytest.approx(-20)
        assert row["available_as_of_date"] == "2024-01-11"
    else:
        assert_skip(row, reason, pre="2024-01-05", post=post, pre_iv=50, post_iv=40)


@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), float("inf"), -float("inf"), True, "50"])
def test_invalid_pre_never_becomes_zero_label_or_fallback(value):
    row = build(daily=[{"date": "2024-01-04", "iv30": 70.0},
                       {"date": "2024-01-05", "iv30": value},
                       {"date": "2024-01-08", "iv30": 30.0}])[0]
    assert row["skip_reason"] == "invalid_pre_iv30"
    assert row["pre_date"] == "2024-01-05"
    assert row["crush_pct_iv30"] is None and row["available_as_of_date"] is None
    assert_skip(row, "invalid_pre_iv30", pre="2024-01-05", post="2024-01-08",
                pre_iv=value if type(value) in (int, float) and math.isfinite(value) else None,
                post_iv=30)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), False, "30"])
def test_invalid_post_does_not_search_later_valid_quote(value):
    row = build(daily=[{"date": "2024-01-05", "iv30": 50.0},
                       {"date": "2024-01-08", "iv30": value},
                       {"date": "2024-01-09", "iv30": 30.0}])[0]
    assert row["skip_reason"] == "invalid_post_iv30"
    assert row["post_date"] == "2024-01-08"
    assert row["crush_pct_iv30"] is None and row["available_as_of_date"] is None
    assert_skip(row, "invalid_post_iv30", pre="2024-01-05", post="2024-01-08", pre_iv=50)


def test_nonfinite_ratio_is_explicit_skip():
    row = build(daily=[{"date": "2024-01-05", "iv30": 1e-300},
                       {"date": "2024-01-08", "iv30": 1e300}])[0]
    assert row["skip_reason"] == "nonfinite_target"
    assert row["available_as_of_date"] is None
    assert_skip(row, "nonfinite_target", pre="2024-01-05", post="2024-01-08",
                pre_iv=1e-300, post_iv=1e300)


@pytest.mark.parametrize("daily,reason", [([], "missing_pre"),
    ([{"date": "2024-01-08", "iv30": 30}], "missing_pre"),
    ([{"date": "2024-01-05", "iv30": 50}], "missing_post")])
def test_missing_anchors_retain_event(daily, reason):
    rows = build(daily=daily)
    assert len(rows) == 1
    assert rows[0]["skip_reason"] == reason
    assert rows[0]["crush_pct_iv30"] is None
    assert rows[0]["available_as_of_date"] is None
    assert_skip(rows[0], reason,
                pre="2024-01-05" if reason == "missing_post" else None,
                post="2024-01-08" if daily and reason == "missing_pre" else None,
                pre_iv=50 if reason == "missing_post" else None,
                post_iv=30 if daily and reason == "missing_pre" else None)


@pytest.mark.parametrize("session,reason", [(None, "missing_session"), ("UNK", "unknown_session"),
                                           ("amc", "unknown_session"), (False, "unknown_session")])
def test_unknown_session_is_not_silently_bmo(session, reason):
    row = build(events=[{"event_date": "2024-01-05", "session": session}])[0]
    assert row["session"] is None and row["skip_reason"] == reason
    assert row["pre_date"] is None and row["post_date"] is None
    assert_skip(row, reason, session=None)


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
    assert_refusal(exc.value)


@pytest.mark.parametrize("which", ["events", "daily"])
def test_duplicate_keys_refuse_even_when_equal(which):
    row = {"event_date": "2024-01-05", "session": "AMC"} if which == "events" else {
        "date": "2024-01-05", "iv30": 50}
    with pytest.raises(DataError) as exc:
        build(**{which: [row, row.copy()]})
    assert_refusal(exc.value)


def test_row_bounds_do_not_truncate_silently():
    with pytest.raises(DataError) as exc:
        build(max_rows=2)
    assert_refusal(exc.value, "RESULT_LIMIT_EXCEEDED")
    assert len(build(max_rows=3)) == 1
    assert build(events=[]) == []


def test_event_row_cap_refuses_before_a_partial_result():
    events = ({"event_date": f"2024-01-{day:02}", "session": "AMC"} for day in range(1, 5))
    with pytest.raises(DataError) as exc:
        build(events=events, daily=[], max_rows=2)
    assert_refusal(exc.value, "RESULT_LIMIT_EXCEEDED")


@pytest.mark.parametrize("session", ["AMC", "BMO"])
def test_missing_event_day_uses_supplied_adjacent_rows(session):
    row = build(events=[{"event_date": "2024-01-06", "session": session}],
                daily=[{"date": "2024-01-05", "iv30": 50},
                       {"date": "2024-01-08", "iv30": 30}])[0]
    assert (row["pre_date"], row["post_date"], row["crush_pct_iv30"]) == (
        "2024-01-05", "2024-01-08", -40.0)


def test_invalid_scalars_remain_distinguishable_in_input_hash():
    hashes = set()
    for value, encoded in ((None, None), (float("nan"), {"__nonfinite__": "nan"}),
                           (float("inf"), {"__nonfinite__": "inf"}), ("bad", "bad"), (False, False)):
        row = build(daily=[{"date": "2024-01-05", "iv30": value},
                           {"date": "2024-01-08", "iv30": 30}])[0]
        expected = {"schema_version": "iv_crush_inputs.v1", "ticker": "TEST",
                    "events": [{"event_date": "2024-01-05", "session": "AMC"}],
                    "daily": [{"date": "2024-01-05", "iv30": encoded},
                              {"date": "2024-01-08", "iv30": 30}]}
        assert row["input_hash"] == "sha256:" + hashlib.sha256(json.dumps(
            expected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
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
        actual_rows = build(events, daily)
        assert len(actual_rows) == len(legacy) == len(events) == 3
        assert [row["event_date"] for row in actual_rows] == [event["event_date"] for event in events]
        assert [row["ticker"] for row in actual_rows] == ["TEST"] * 3
        assert legacy.event_date.dt.strftime("%Y-%m-%d").tolist() == [
            event["event_date"] for event in events]
        for actual, expected in zip(actual_rows, legacy.crush_pct_iv30, strict=True):
            assert (actual["crush_pct_iv30"] is None) == math.isnan(expected)
            if not math.isnan(expected):
                assert actual["crush_pct_iv30"] == expected


@pytest.mark.parametrize("field,value", [
    (field, value) for field in ("ticker", "source_id")
    for value in (None, True, 42, "", " ", " padded", "padded ", "x" * 257)
] + [("max_rows", value) for value in (None, False, -1, 1.5, "2")])
def test_public_identity_and_limit_guards_have_exact_refusals(field, value):
    args = dict(ticker="TEST", events=[], daily=[], source_id="synthetic", max_rows=1)
    args[field] = value
    with pytest.raises(DataError) as exc:
        build_rows(**args)
    assert_refusal(exc.value)


@pytest.mark.parametrize("field", ["session", "iv30"])
def test_projection_string_length_boundary(field):
    events = [{"event_date": "2024-01-05", "session": "AMC"}]
    daily = [{"date": "2024-01-05", "iv30": 50}, {"date": "2024-01-08", "iv30": 30}]
    selected = events[0] if field == "session" else daily[0]
    selected[field] = "x" * 256
    row = build(events, daily)[0]
    assert row["skip_reason"] == ("unknown_session" if field == "session" else "invalid_pre_iv30")
    selected[field] += "x"
    with pytest.raises(DataError) as exc:
        build(events, daily)
    assert_refusal(exc.value)


def test_text_and_row_caps_accept_exact_boundary():
    events = [{"event_date": "2024-01-05", "session": "AMC"},
              {"event_date": "2024-01-08", "session": "BMO"}]
    daily = [{"date": "2024-01-05", "iv30": 50}, {"date": "2024-01-08", "iv30": 30}]
    rows = build_rows("T" * 256, events, daily, source_id="s" * 256, max_rows=2)
    assert len(rows) == 2
    assert [(r["ticker"], r["source_id"], r["crush_pct_iv30"]) for r in rows] == [
        ("T" * 256, "s" * 256, -40.0)] * 2


@pytest.mark.parametrize("field", ["events", "daily"])
def test_malformed_or_conflicting_projection_refusals(field):
    row = {"event_date": "2024-01-05", "session": "AMC"} if field == "events" else {
        "date": "2024-01-05", "iv30": 50}
    value_field = "session" if field == "events" else "iv30"
    for bad in (None, [None], [42], [row, {**row, value_field: "different"}]):
        args = dict(ticker="TEST", events=[], daily=[], source_id="synthetic")
        args[field] = bad
        with pytest.raises(DataError) as exc:
            build_rows(**args)
        assert_refusal(exc.value)


def test_empty_events_still_validate_daily_input():
    with pytest.raises(DataError) as exc:
        build(events=[], daily=[{"date": "20240105", "iv30": 50}])
    assert_refusal(exc.value)


@pytest.mark.parametrize("field", ["event_date", "date"])
@pytest.mark.parametrize("value", [None, 20240105, "2024-1-5", "2024-01-05Z"])
def test_noncanonical_projection_dates_refuse(field, value):
    kwargs = ({"events": [{field: value, "session": "AMC"}]} if field == "event_date"
              else {"daily": [{field: value, "iv30": 50}]})
    with pytest.raises(DataError) as exc:
        build(**kwargs)
    assert_refusal(exc.value)


def test_unrepresentable_next_day_refuses_complete_batch():
    with pytest.raises(DataError) as exc:
        build(events=[{"event_date": "9999-12-31", "session": "BMO"}],
              daily=[{"date": "9999-12-30", "iv30": 50}, {"date": "9999-12-31", "iv30": 30}])
    assert_refusal(exc.value)


def test_independent_projection_hash_and_complete_multievent_rows():
    # ASCII keys/strings and integer numbers make sorted compact stdlib JSON
    # exactly the specified canonical bytes, independent of the production helper.
    events = [{"event_date": "2024-01-05", "session": "AMC"},
              {"event_date": "2024-01-08", "session": "BMO"},
              {"event_date": "2024-01-09", "session": "AMC"},
              {"event_date": "2024-01-10", "session": None}]
    daily = [{"date": "2024-01-05", "iv30": 50}, {"date": "2024-01-08", "iv30": 30}]
    expected_input = {"schema_version": "iv_crush_inputs.v1", "ticker": "TEST",
                      "events": events, "daily": daily}
    expected_hash = "sha256:" + hashlib.sha256(json.dumps(
        expected_input, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    base = {"ticker": "TEST", "source_id": "synthetic-source", "input_hash": expected_hash}
    expected = [
        {**base, "event_date": "2024-01-05", "session": "AMC", "pre_date": "2024-01-05",
         "post_date": "2024-01-08", "pre_iv30": 50, "post_iv30": 30, "crush_pct_iv30": -40,
         "available_as_of_date": "2024-01-09", "skipped": False, "skip_reason": None},
        {**base, "event_date": "2024-01-08", "session": "BMO", "pre_date": "2024-01-05",
         "post_date": "2024-01-08", "pre_iv30": 50, "post_iv30": 30, "crush_pct_iv30": -40,
         "available_as_of_date": "2024-01-09", "skipped": False, "skip_reason": None},
        {**base, "event_date": "2024-01-09", "session": "AMC", "pre_date": "2024-01-08",
         "post_date": None, "pre_iv30": 30, "post_iv30": None, "crush_pct_iv30": None,
         "available_as_of_date": None, "skipped": True, "skip_reason": "missing_post"},
        {**base, "event_date": "2024-01-10", "session": None, "pre_date": None,
         "post_date": None, "pre_iv30": None, "post_iv30": None, "crush_pct_iv30": None,
         "available_as_of_date": None, "skipped": True, "skip_reason": "missing_session"},
    ]
    assert build(list(reversed(events)), list(reversed(daily))) == expected
    # Verify each semantic input dimension rather than only daily IV values.
    for changed in ("ticker", "event_date", "session", "daily_date", "iv30"):
        altered = copy.deepcopy(expected_input)
        if changed == "ticker":
            altered["ticker"] = "OTHER"
        elif changed in ("event_date", "session"):
            altered["events"][0][changed] = "2024-01-04" if changed == "event_date" else "BMO"
        else:
            altered["daily"][0]["date" if changed == "daily_date" else "iv30"] = (
                "2024-01-04" if changed == "daily_date" else 60)
        changed_hash = "sha256:" + hashlib.sha256(json.dumps(
            altered, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        rows = build_rows(altered["ticker"], altered["events"], altered["daily"], source_id="synthetic")
        assert all(row["input_hash"] == changed_hash != expected_hash for row in rows)


def test_pure_boundary_ignores_extras_and_does_not_read_declared_source(monkeypatch):
    import builtins
    import os
    import sqlite3
    from pathlib import Path

    from engine.v2.foundation import ArtifactStore

    events = [{"event_date": "2024-01-05", "session": "AMC"}]
    daily = [{"date": "2024-01-05", "iv30": 50}, {"date": "2024-01-08", "iv30": 30}]
    expected = build(events, daily)
    events[0].update(ticker="TEST", ignored=object())
    daily[0].update(ticker="TEST", ignored=object())

    def forbid(*args, **kwargs):
        raise AssertionError("pure target construction attempted I/O")

    with monkeypatch.context() as guard:
        for target, name in ((builtins, "open"), (os, "open"), (Path, "open"),
                             (sqlite3, "connect"), (ArtifactStore, "read_verified"),
                             (ArtifactStore, "publish_bytes")):
            guard.setattr(target, name, forbid)
        actual = build_rows("TEST", events, daily, source_id="unverified-source")
    assert actual == [{**row, "source_id": "unverified-source"} for row in expected]
