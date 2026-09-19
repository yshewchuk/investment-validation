"""R4 `_native()` residual, closed: checks/phase4_real.py's application/
completion/chooser side controls no longer build their NativeScoreInputs via
NativeScoreInputs.from_legacy_fields (the removed `checks.phase4_real._native`
helper). These tests prove each converted control reads inputs built the same
native way the main comparison does (SourceBundle -> build_native_score_inputs,
or -- for the chooser controls, which never touch NativeScoreInputs at all --
a directly-built ScoreRecord), and that a legacy answer field smuggled into
that construction is refused outright, not silently accepted."""
from __future__ import annotations

from dataclasses import replace

import pytest

from checks.phase4_real import (
    _application_control_source,
    _application_controls,
    _chooser_controls,
    _completion_controls,
    _request,
)
from engine.v2.scoring import application
from engine.v2.scoring.source_inputs import build_native_score_inputs
from engine.v2.scoring.stages import NativeScoreInputs


def test_phase4_real_no_longer_has_a_legacy_field_helper():
    from checks import phase4_real

    assert not hasattr(phase4_real, "_native")


def test_application_control_source_is_native_built_not_legacy_fields():
    inputs = build_native_score_inputs(_application_control_source())

    assert isinstance(inputs, NativeScoreInputs)
    # from_legacy_fields always stamps this exact marker and leaves geometry/
    # pricing None forever; build_native_score_inputs never does.
    assert inputs.source_ref != "compatibility-input"
    assert inputs.geometry is None and inputs.pricing is None
    assert inputs.forecast.get("models", {}).get("driver_prediction") == {
        "intercept": 7.0, "coefficients": {},
    }


def test_application_controls_all_true_from_native_inputs():
    assert _application_controls() == {
        "direct_batch_equal": True,
        "operational_time_excluded": True,
        "fill_changes_identity": True,
        "zero_is_not_missing": True,
        "financial_values_owned": True,
        "batch_resource_profile": True,
    }


def test_financial_values_are_owned_by_real_native_pricing_not_copied():
    """Change a quote and prove entry_cost_pct moves with it: the control's
    financial diagnostics come from a real native pricing stage, not a
    passed-through legacy value (which from_legacy_fields would have
    smuggled straight through unchanged)."""
    source = _application_control_source()
    cheaper = replace(source, raw_quotes={
        key: {"bid": 1.0, "ask": 2.0} for key in source.raw_quotes
    })
    record = application.score_one(_request(), build_native_score_inputs(cheaper))

    assert record.financial_diagnostics["entry_cost_pct"] != 5.0


def test_completion_controls_legacy_projection_reads_the_native_source():
    assert _completion_controls(_application_controls())["legacy_projection_owned"] is True


def test_chooser_controls_all_true_from_directly_built_score_records():
    assert _chooser_controls() == {
        "chooser_tie_control": True,
        "chooser_missing_competitor_control": True,
        "chooser_fallback_control": True,
        "chooser_no_regating_control": True,
    }


@pytest.mark.parametrize("answer_field,value", [
    ("gate_score", 0.7), ("entry_cost", 5.0), ("driver_prediction", 7.0),
])
def test_a_legacy_answer_field_smuggled_into_the_source_is_refused(answer_field, value):
    """Negative control: build_native_score_inputs refuses a legacy-computed
    answer field outright (SourceBundle._reject_answers) rather than silently
    accepting it -- the enforcement that makes a from_legacy_fields-style
    passthrough impossible on this path."""
    source = _application_control_source()
    tampered = replace(source, context={**source.context, answer_field: value})

    with pytest.raises(ValueError, match="calculated answer fields"):
        build_native_score_inputs(tampered)