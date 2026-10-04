"""Synthetic as-of and leak coverage for the entry-relative feature context.

Slice 3 is pure: these tests hand ``ExperimentFeatureContext`` synthetic
five-column rows that combine a ``Repository.scan`` row's values with the
caller's selected ``SnapshotRef`` binding, and call ``feature`` directly. No
catalog, ledger, runner or filesystem touches this module -- a refusal here
leaves nothing behind.
"""
from __future__ import annotations

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


def test_post_entry_row_refuses_even_with_an_older_eligible_row():
    context = _context(_row("gap", ENTRY_AT - timedelta(days=2), 0.10),
                       _row("gap", ENTRY_AT + timedelta(seconds=1), 0.99))
    with pytest.raises(OpsError) as excinfo:
        context.feature("gap")
    assert excinfo.value.code == "FEATURE_LOOKAHEAD"
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


def test_malformed_identity_time_and_request_inputs_are_typed():
    with pytest.raises(OpsError) as excinfo:
        _context(_row("gap", ENTRY_AT, 0.10), entry_at=datetime(2026, 1, 15, 21, 0))
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    with pytest.raises(OpsError) as excinfo:
        _context(_row("gap", "no-such-instant", 0.10))
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    with pytest.raises(OpsError) as excinfo:
        _context(_row("gap", ENTRY_AT, 0.10), snapshot_id=None)
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    with pytest.raises(OpsError) as excinfo:
        _context({"snapshot_id": SNAPSHOT}).feature("gap")
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
    context = _context(_row("gap", ENTRY_AT, 0.10))
    with pytest.raises(OpsError) as excinfo:
        context.feature("gap", observed_at=ENTRY_AT)
    assert excinfo.value.code == "INVALID_EXPERIMENT_SPEC"
