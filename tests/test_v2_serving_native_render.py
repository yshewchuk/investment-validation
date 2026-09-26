"""native_render: the native ScoreRecord -> serving display row translator.

Every case scores a real :class:`~engine.v2.contracts.ScoreRequest` through the
real :func:`engine.v2.scoring.application.score_one` with a real
list-appending observer, then feeds the resulting record (and the S9H
``analog_display_fields`` display channel) to
:func:`engine.v2.serving.native_render.native_display_row`. No guards are
monkeypatched and no fixture ids are hand-injected; the one post-hoc mutation
is test 4's stripped ``event_ref``, which no real input can produce.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from checks.phase4_real import _application_control_source, _request
from engine.v2.scoring import application
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs
from engine.v2.scoring.stages import DISPLAY_ANALOG_FIELDS, analog_display_fields
from engine.v2.serving.native_render import (
    NATIVE_NEVER_COMPUTED,
    join_analog_fields_by_request,
    native_display_row,
    native_row_key,
)


def _scored(bundle: SourceBundle | None = None):
    """One real score_one call with a real observer; ``(record, observations)``."""
    collected: list = []
    record = application.score_one(
        _request(), build_native_score_inputs(bundle or _application_control_source()),
        observer=collected.append)
    return record, collected


def _zero_match_bundle() -> SourceBundle:
    """The real application control, with an analog recipe that ran and matched
    no row: an all-unavailable query bucket (no market cap) against a one-row
    population, with no widening dimension left to drop onto it."""
    return replace(
        _application_control_source(),
        analog_recipe={
            "bucket_dimensions": ("mcap_bucket",),
            "widening_order": (),
            "min_analogs": 3,
            "alpha": 0.5,
            "bootstrap_draws": 64,
            "bootstrap_seed": 11,
            "ci_quantiles": (0.05, 0.95),
        },
        analog_source_rows=(
            {"row_id": "hist-1", "mcap_bucket": "1-10B", "realized_return": 0.1},
        ),
        analog_query={"mcap_bucket": None},
    )


def test_never_computed_band_is_absent_from_the_real_row():
    record, collected = _scored()
    row = native_display_row(record, analog_fields=analog_display_fields(collected))

    assert set(NATIVE_NEVER_COMPUTED).isdisjoint(row)
    assert row["strategy"] == "STR-THRU"
    assert row["gate_pass"] is True
    assert row["driver_prediction"] == pytest.approx(7.0)
    assert row["implied_move"] == pytest.approx(6.0)
    assert row["entry_cost"] == pytest.approx(5.0)


def test_no_chooser_selection_is_none_never_an_error_or_a_zero():
    record, collected = _scored()
    assert record.chooser_selection is None

    row = native_display_row(record, analog_fields=analog_display_fields(collected))

    assert row["chosen_strategy"] is None
    assert row["chosen_margin"] is None
    assert row["menu_size"] is None


def test_n_analogs_zero_is_present_but_absent_analog_fields_omits_the_key():
    record, collected = _scored(_zero_match_bundle())
    assert record.resolved_request.get("n_analogs") == 0
    fields = analog_display_fields(collected)
    assert fields == {"selected_row_ids": (), "contributing_row_ids": ()}

    ran = native_display_row(record, analog_fields=fields)
    assert "n_analogs" in ran and ran["n_analogs"] == 0

    did_not_run = native_display_row(record, analog_fields=None)
    assert "n_analogs" not in did_not_run
    for name in DISPLAY_ANALOG_FIELDS:
        assert did_not_run[name] == ()


def test_join_analog_fields_by_request_keys_each_group_verbatim():
    _, collected = _scored()
    joined = join_analog_fields_by_request({"phase4-event|cal-1": collected})
    assert joined == {"phase4-event|cal-1": analog_display_fields(collected)}


def test_native_row_key_refuses_a_malformed_event_ref():
    record, _ = _scored()
    malformed = replace(record, event_ref={})

    with pytest.raises(KeyError, match="native ScoreRecord.event_ref missing 'ticker'"):
        native_row_key(malformed)
