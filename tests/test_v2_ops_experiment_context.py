"""Synthetic as-of and leak coverage for the entry-relative feature context.

Slice 3 is pure: these tests hand ``ExperimentFeatureContext`` synthetic
five-column rows that combine a ``Repository.scan`` row's values with the
caller's selected ``SnapshotRef`` binding, and call ``feature`` directly. No
catalog, ledger, runner or filesystem touches this module -- a refusal here
leaves nothing behind.
"""
from __future__ import annotations

import decimal
import math
from datetime import datetime, timedelta, timezone

import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiments import ENTRY, ExperimentFeatureContext

SNAPSHOT = "snap-synthetic-1"
EVENT = "EVT-1"
ENTRY_AT = datetime(2026, 1, 15, 21, 0, tzinfo=timezone.utc)


def _row(feature, observed_at, value, *, event_id=EVENT, snapshot_id=SNAPSHOT):
    return {"snapshot_id": snapshot_id, "event_id": event_id, "feature": feature,
            "observed_at": observed_at, "value": value}


def _context(*rows, **changes):
    fields = {"snapshot_id": SNAPSHOT, "event_id": EVENT, "entry_at": ENTRY_AT,
              "observations": rows}
    fields.update(changes)
    return ExperimentFeatureContext(**fields)


def test_row_observed_at_the_entry_instant_is_returned():
    context = _context(_row("gap", ENTRY_AT, 0.25))
    assert context.feature("gap") == 0.25
    assert context.feature("gap", observed_at=ENTRY) == 0.25


def test_latest_pre_entry_observation_wins_among_many():
    context = _context(_row("gap", ENTRY_AT - timedelta(days=3), 0.10),
                       _row("gap", (ENTRY_AT - timedelta(days=1)).isoformat(), 0.20),
                       _row("gap", ENTRY_AT - timedelta(hours=4), 0.30))
    assert context.feature("gap") == 0.30


def test_iso_entry_string_is_accepted_and_bound():
    context = _context(_row("gap", ENTRY_AT, 0.25),
                       entry_at="2026-01-15T21:00:00+00:00")
    assert context.feature("gap") == 0.25


def test_entry_instant_in_another_utc_offset_is_accepted():
    offset = timezone(timedelta(hours=-5))
    context = _context(_row("gap", ENTRY_AT.astimezone(offset), 0.25))
    assert context.feature("gap") == 0.25


def test_post_entry_row_refuses_even_with_an_older_eligible_row():
    eligible = _row("gap", ENTRY_AT - timedelta(days=2), 0.10)
    passing = _context(eligible)
    assert passing.feature("gap") == 0.10
    leaking = _context(eligible, _row("gap", ENTRY_AT + timedelta(seconds=1), 0.99))
    with pytest.raises(OpsError) as excinfo:
        leaking.feature("gap")
    assert excinfo.value.code == "FEATURE_LOOKAHEAD"
    assert not excinfo.value.problem.retryable


def test_one_microsecond_past_entry_refuses_as_lookahead():
    eligible = _row("gap", ENTRY_AT - timedelta(days=2), 0.10)
    passing = _context(eligible)
    assert passing.feature("gap") == 0.10
    leaking = _context(eligible, _row("gap", ENTRY_AT + timedelta(microseconds=1), 0.99))
    with pytest.raises(OpsError) as excinfo:
        leaking.feature("gap")
    assert excinfo.value.code == "FEATURE_LOOKAHEAD"
    assert not excinfo.value.problem.retryable


def test_latest_tied_observations_with_agreeing_values_return_that_value():
    context = _context(_row("gap", ENTRY_AT - timedelta(days=1), 0.10),
                       _row("gap", ENTRY_AT, 0.30),
                       _row("gap", ENTRY_AT, 0.30))
    assert context.feature("gap") == 0.30


def test_latest_tied_observations_with_conflicting_values_are_refused():
    context = _context(_row("gap", ENTRY_AT, 0.30), _row("gap", ENTRY_AT, 0.31))
    with pytest.raises(OpsError) as excinfo:
        context.feature("gap")
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    assert not excinfo.value.problem.retryable


def test_latest_tied_nan_observations_agree_and_return_nan():
    context = _context(_row("gap", ENTRY_AT, float("nan")),
                       _row("gap", ENTRY_AT, float("nan")))
    value = context.feature("gap")
    assert isinstance(value, float) and math.isnan(value)


@pytest.mark.parametrize("values", (
    pytest.param((float("nan"), 0.30), id="nan-then-number"),
    pytest.param((0.30, float("nan")), id="number-then-nan"),
))
def test_latest_tied_nan_and_number_conflict(values):
    context = _context(_row("gap", ENTRY_AT, values[0]),
                       _row("gap", ENTRY_AT, values[1]))
    with pytest.raises(OpsError) as excinfo:
        context.feature("gap")
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    assert not excinfo.value.problem.retryable


def test_latest_tied_decimal_nan_observations_agree_and_return_nan():
    context = _context(_row("gap", ENTRY_AT, decimal.Decimal("NaN")),
                       _row("gap", ENTRY_AT, decimal.Decimal("NaN")))
    value = context.feature("gap")
    assert isinstance(value, decimal.Decimal) and value.is_nan()


@pytest.mark.parametrize("values", (
    pytest.param((decimal.Decimal("NaN"), 0.30), id="decimal-nan-then-number"),
    pytest.param((0.30, decimal.Decimal("NaN")), id="number-then-decimal-nan"),
))
def test_latest_tied_decimal_nan_and_number_conflict(values):
    context = _context(_row("gap", ENTRY_AT, values[0]),
                       _row("gap", ENTRY_AT, values[1]))
    with pytest.raises(OpsError) as excinfo:
        context.feature("gap")
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    assert not excinfo.value.problem.retryable


def test_other_events_and_features_do_not_trigger_the_leak_refusal():
    leak = ENTRY_AT + timedelta(days=1)
    context = _context(_row("gap", ENTRY_AT - timedelta(days=1), 0.10),
                       _row("gap", leak, 0.99, event_id="EVT-OTHER"),
                       _row("momentum", leak, 7.0))
    assert context.feature("gap") == 0.10
    with pytest.raises(OpsError) as excinfo:
        context.feature("momentum")
    assert excinfo.value.code == "FEATURE_LOOKAHEAD"


def test_row_from_another_snapshot_is_unresolved():
    with pytest.raises(OpsError) as excinfo:
        _context(_row("gap", ENTRY_AT, 0.10, snapshot_id="snap-other"))
    assert excinfo.value.code == "SNAPSHOT_UNRESOLVED"


def test_missing_requested_observation_uses_the_typed_missing_input_code():
    context = _context(_row("gap", ENTRY_AT - timedelta(days=1), 0.10))
    with pytest.raises(OpsError) as excinfo:
        context.feature("iv30")
    assert excinfo.value.code == "FEATURES_MISSING"


@pytest.mark.parametrize("case", (
    pytest.param(
        lambda: _context(_row("gap", ENTRY_AT, 0.10),
                         entry_at=datetime(2026, 1, 15, 21, 0)),
        id="naive-entry"),
    pytest.param(
        lambda: _context(_row("gap", "no-such-instant", 0.10)),
        id="malformed-observation-instant"),
    pytest.param(
        lambda: _context(_row("gap", ENTRY_AT, 0.10), snapshot_id=None),
        id="empty-snapshot-id"),
    pytest.param(
        lambda: _context({"snapshot_id": SNAPSHOT}).feature("gap"),
        id="incomplete-observation-mapping"),
    pytest.param(
        lambda: _context(_row("gap", ENTRY_AT, 0.10)).feature("gap",
                                                              observed_at=ENTRY_AT),
        id="unsupported-observed-at-request"),
))
def test_malformed_identity_time_and_request_inputs_are_typed(case):
    with pytest.raises(OpsError) as excinfo:
        case()
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
