import pytest

from engine.v2.scoring.nightly_source_bundle import assemble_nightly_source_bundle
from engine.v2.scoring.source_inputs import (
    AnswerFieldsInSource,
    build_native_score_inputs,
)
from engine.v2.scoring.stages import assemble_native_values


def _kwargs(**overrides):
    base = dict(
        source_ref="test:answer-named:v1",
        strategy="STR-THRU",
        as_of="2026-01-16",
        calendar_row={
            "ticker": "TEST", "event_date": "2026-01-15",
            "entry_date": "2026-01-15", "exit_date": "2026-01-16",
            "expiry": "2026-01-16", "spot": 100.0,
            "calendar_observed_through": "2026-01-09",
        },
        panel_row={"date": "2026-01-15", "signal": 1.5},
        panel_anchor="2026-01-10",
        tier4_row={"pred_abs_move": 0.05, "pred_abs_move_fold_start": "2026-01-01"},
        quote_rows=[
            {"right": "C", "strike": 100.0, "expiry": "2026-01-16",
             "bid": 1.0, "ask": 1.2, "observed_at": "2026-01-15",
             "chain_spot": 95.0},
            {"right": "P", "strike": 100.0, "expiry": "2026-01-16",
             "bid": 1.1, "ask": 1.3, "observed_at": "2026-01-15",
             "chain_spot": 95.0},
        ],
        quote_status=None,
        feature_names=("signal", "pred_abs_move", "entry_cost_pct"),
        forecast_recipes={"driver_prediction": {"intercept": 0.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:answer-named"},
        gate_recipe={"model": {"intercept": 0.0, "coefficients": {}}, "threshold": 0.0},
    )
    base.update(overrides)
    return base


def test_calculated_answer_name_in_feature_order_builds_and_scorer_computes_it():
    bundle = assemble_nightly_source_bundle(**_kwargs())
    assert "entry_cost_pct" not in bundle.feature_vector
    assert "entry_cost_pct" not in bundle.feature_missing_mask
    assert set(bundle.feature_missing_mask) == {"signal", "pred_abs_move"}
    values = assemble_native_values(
        build_native_score_inputs(bundle), strategy="STR-THRU")
    # Chain spot differs from the calendar close, so a close fallback fails.
    assert values["entry_cost_pct"] == pytest.approx((1.1 + 1.2) / 95.0 * 100.0)


def test_answer_name_staged_in_a_source_row_is_still_refused_with_paths():
    bundle = assemble_nightly_source_bundle(**_kwargs())
    bad = type(bundle)(**{**bundle.__dict__, "metadata": {"entry_cost_pct": 1.0}})
    with pytest.raises(AnswerFieldsInSource) as info:
        build_native_score_inputs(bad)
    assert info.value.fields == ["entry_cost_pct"]