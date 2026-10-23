"""The entry_rule branch of _gate_block: build-time declarations, synthetic only."""
import pytest

from engine.v2.models.lineage import DataDependency, Lineage
from engine.v2.models.trailing_cutoff_artifact import (
    PNL_HISTORY_ID,
    cutoff_month,
    make_trailing_cutoff_artifact,
)
from engine.v2.scoring.native_entry_rule import entry_rule_block
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs

EVENT_DATE = "2026-09-16"
MCAP = 2e10
_CONTEXT = {
    "ticker": "TEST",
    "event_date": EVENT_DATE,
    "entry_date": EVENT_DATE,
    "exit_date": "2026-09-17",
    "expiry": "2026-09-18",
    "spot": 100.0,
}


def _cutoff(*, month="2026-09-01", bar=0.5):
    return make_trailing_cutoff_artifact(
        month=month, cutoff=bar,
        lineage=Lineage(data=(DataDependency(
            table="synthetic.pnl-history", end_exclusive=month),)))


def _bundle(**overrides):
    values = dict(
        source_ref="synthetic:twin-p:entry-rule",
        strategy="TWIN-P",
        context=dict(_CONTEXT),
        raw_quotes={("P", 100.0, "2026-09-18"): {"bid": 1.0, "ask": 1.2}},
        feature_vector={"signal": 1.0},
        feature_missing_mask={"signal": False},
        model_identity={},
        forecast_recipes={"forecast_abs_move": {
            "intercept": 1.0, "coefficients": {"signal": 0.5}}},
        model_artifact_refs={"forecast_abs_move": "sha256:synthetic-size"},
        residual_recipe={},
        analog_recipe={},
        gate_recipe={"mode": "entry_rule", "mcap_usd": MCAP},
        trailing_cutoff_artifact=_cutoff(),
    )
    values.update(overrides)
    return SourceBundle(**values)


def test_gate_equals_entry_rule_block_over_the_declared_cutoff():
    artifact = _cutoff()
    inputs = build_native_score_inputs(_bundle(trailing_cutoff_artifact=artifact))

    assert inputs.gate == entry_rule_block(
        "TWIN-P", mcap_usd=MCAP, cutoff=artifact, month=None)


def test_absent_artifact_declares_the_event_month_key_with_no_payload():
    inputs = build_native_score_inputs(_bundle(trailing_cutoff_artifact=None))

    assert inputs.gate == entry_rule_block(
        "TWIN-P", mcap_usd=MCAP, cutoff=None, month=EVENT_DATE)
    assert inputs.gate["trailing_cutoff"] is None
    assert inputs.gate["trailing_cutoff_key"] == {
        "history_id": PNL_HISTORY_ID,
        "month": cutoff_month(EVENT_DATE),
    }


def test_absent_artifact_without_an_event_date_is_refused():
    context = {name: value for name, value in _CONTEXT.items()
               if name != "event_date"}

    with pytest.raises(ValueError, match="entry_rule gate needs context.event_date"):
        build_native_score_inputs(
            _bundle(trailing_cutoff_artifact=None, context=context))


def test_an_extra_recipe_member_is_refused():
    with pytest.raises(ValueError, match="takes only mode and mcap_usd"):
        build_native_score_inputs(_bundle(gate_recipe={
            "mode": "entry_rule", "mcap_usd": MCAP, "pin": False,
        }))


def test_an_undefined_entry_rule_strategy_is_refused():
    with pytest.raises(ValueError,
                        match="entry_rule gate is not defined for STR-THRU"):
        build_native_score_inputs(_bundle(
            strategy="STR-THRU",
            forecast_recipes={"driver_prediction": {
                "intercept": 5.0, "coefficients": {"signal": 2.0}}},
            model_artifact_refs={"driver_prediction": "sha256:synthetic-driver"},
        ))


def test_a_trailing_cutoff_without_an_entry_rule_recipe_is_refused():
    with pytest.raises(ValueError, match="requires an entry_rule gate_recipe"):
        build_native_score_inputs(_bundle(gate_recipe={
            "recipe_id": "linear-gate-v1",
            "model": {"intercept": 4.0, "coefficients": {"signal": -1.0}},
            "threshold": 0.0,
        }))


def test_a_numeric_string_mcap_usd_is_refused():
    with pytest.raises(ValueError,
                        match="gate_recipe.mcap_usd must be a finite number or None"):
        build_native_score_inputs(_bundle(gate_recipe={
            "mode": "entry_rule", "mcap_usd": "5",
        }))


def test_a_foreign_trailing_cutoff_value_is_refused():
    with pytest.raises(ValueError, match="must be a TrailingCutoffArtifact"):
        build_native_score_inputs(_bundle(trailing_cutoff_artifact=object()))