"""Causal cutoff selection for frozen scoring artifacts."""
from types import SimpleNamespace

import pytest

from engine.v2.foundation import to_document
from engine.v2.models.payoff_artifact import (
    PayoffLineArtifact,
    make_payoff_line_artifact,
    make_payoff_surface_artifact,
)
from engine.v2.models.recalibration_artifact import (
    RecalibrationMapArtifact,
    make_recalibration_map_artifact,
)
from engine.v2.ops.native.input_decoding import _load_native_score_inputs
from engine.v2.ops.native_runup_inputs import (
    _surface,
    latest_artifacts_at_or_before,
)
from engine.v2.ops.native_score_batch import _release_payoff_declarations
from engine.v2.scoring.native_payoff import RUNUP_TERMS
from engine.v2.scoring.nightly_source_bundle import NightlySourceBundleRefusal
from engine.v2.scoring.stages import (
    STAGE_NAMES,
    NativeScoreInputs,
    StageReceipt,
    _execute_model,
    _recalibrated_output,
    _surface_fit_from_artifact,
)


def test_latest_artifact_uses_data_cutoff_and_admits_equal_cutoff():
    before = SimpleNamespace(
        cutoff="2026-10-01", alpha=0.5, computed_at="2026-10-30T00:00:00Z")
    at = SimpleNamespace(
        cutoff="2026-10-15", alpha=0.5, computed_at="2026-10-01T00:00:00Z")
    after = SimpleNamespace(
        cutoff="2026-10-16", alpha=0.5, computed_at="2026-10-02T00:00:00Z")

    selected = latest_artifacts_at_or_before(
        (before, at, after), cutoff="2026-10-15", fill_alpha=0.5)

    assert selected == (at,)


def test_runup_surface_keeps_typed_refusal_when_no_causal_artifact_exists():
    binding = SimpleNamespace(payoff_artifacts={"STR-RUNUP": ()})

    with pytest.raises(NightlySourceBundleRefusal) as raised:
        _surface(binding, cutoff="2026-10-15", fill_alpha=0.5)

    assert raised.value.code == "MODEL_NOT_READY"


def test_scoring_stage_accepts_recorded_prior_cutoff_and_refuses_future():
    def surface(cutoff):
        return make_payoff_surface_artifact(
            {
                "n": 2,
                "resid_sd": 0.1,
                "r": None,
                "residuals": (0.1, -0.1),
                "coefficients": (0.0,) * len(RUNUP_TERMS),
            },
            alpha=0.5,
            cutoff=cutoff,
        )

    flags = []
    selected = _surface_fit_from_artifact(
        surface("2026-10-01"), "STR-RUNUP", 0.5, "2026-10-15", flags,
        "2026-10-01",
    )
    assert selected is not None
    assert flags == []

    future_flags = []
    future = _surface_fit_from_artifact(
        surface("2026-10-16"), "STR-RUNUP", 0.5, "2026-10-15", future_flags,
        "2026-10-16",
    )
    assert future is None
    assert future_flags == ["MODEL_NOT_READY"]


def test_runup_surface_binds_latest_data_cutoff_at_or_before_row():
    def surface(cutoff):
        return make_payoff_surface_artifact(
            {
                "n": 2,
                "resid_sd": 0.1,
                "r": None,
                "residuals": (0.1, -0.1),
                "coefficients": (0.0,) * len(RUNUP_TERMS),
            },
            alpha=0.5,
            cutoff=cutoff,
        )

    artifacts = tuple(surface(day) for day in (
        "2026-10-01", "2026-10-15", "2026-10-16"))
    binding = SimpleNamespace(payoff_artifacts={"STR-RUNUP": artifacts})

    assert _surface(binding, cutoff="2026-10-15", fill_alpha=0.5).cutoff == "2026-10-15"


def test_recalibration_stage_accepts_bound_prior_and_refuses_future_cutoff():
    def recalibration(cutoff):
        return make_recalibration_map_artifact(
            None, strategy="STR-THRU", alpha=0.5, cutoff=cutoff, min_pairs=10)

    flags = []
    output = {"win_model": 0.5}
    prior = _recalibrated_output(
        {"recalibration_artifact": recalibration("2026-10-01")},
        {"before": "2026-10-15", "recalibration_bound_cutoff": "2026-10-01"},
        "STR-THRU", {"fill": 0.5}, output, flags)
    assert prior == output
    assert flags == []

    future_flags = []
    future = _recalibrated_output(
        {"recalibration_artifact": recalibration("2026-10-16")},
        {"before": "2026-10-15", "recalibration_bound_cutoff": "2026-10-16"},
        "STR-THRU", {"fill": 0.5}, output, future_flags)
    assert future == {}
    assert future_flags == ["MODEL_NOT_READY"]

    missing_cutoff_flags = []
    missing_cutoff = _recalibrated_output(
        {"recalibration_artifact": recalibration("2026-10-16")},
        {"recalibration_bound_cutoff": "2026-10-16"},
        "STR-THRU", {"fill": 0.5}, output, missing_cutoff_flags)
    assert missing_cutoff == {}
    assert missing_cutoff_flags == ["MODEL_NOT_READY"]


def test_str_thru_declaration_binds_latest_line_and_recalibration_cutoffs():
    def line(cutoff):
        return make_payoff_line_artifact(
            {
                "n": 2,
                "resid_sd": 0.1,
                "r": None,
                "residuals": (0.1, -0.1),
                "intercept": 0.1,
                "slope": 0.2,
            },
            strategy="STR-THRU",
            driver="implied_t1",
            alpha=0.5,
            cutoff=cutoff,
        )

    def recalibration(cutoff):
        return make_recalibration_map_artifact(
            None, strategy="STR-THRU", alpha=0.5, cutoff=cutoff, min_pairs=10)

    lines = tuple(line(day) for day in (
        "2026-10-01", "2026-10-15", "2026-10-16"))
    maps = tuple(recalibration(day) for day in (
        "2026-10-01", "2026-10-15", "2026-10-16"))
    binding = SimpleNamespace(
        driver_residual_artifacts={},
        recalibration_artifacts={"STR-THRU": maps},
    )

    declarations = _release_payoff_declarations(
        binding, "STR-THRU", lines, "2026-10-20", {"entry_date": "2026-10-15"})

    assert declarations["payoff_artifact"].cutoff == "2026-10-15"
    assert declarations["recalibration_artifact"].cutoff == "2026-10-15"
    assert declarations["payoff_artifact_recipe"] == {
        "before": "2026-10-15",
        "bound_cutoff": "2026-10-15",
        "recalibration_bound_cutoff": "2026-10-15",
    }


def test_str_thru_refuses_when_no_eligible_line_or_recalibration_map():
    binding = SimpleNamespace(recalibration_artifacts={"STR-THRU": ()})
    row = {"entry_date": "2026-10-15"}

    with pytest.raises(NightlySourceBundleRefusal) as missing_line:
        _release_payoff_declarations(
            binding, "STR-THRU", (), "2026-10-20", row)
    assert missing_line.value.code == "MODEL_NOT_READY"

    line = make_payoff_line_artifact(
        {
            "n": 2,
            "resid_sd": 0.1,
            "r": None,
            "residuals": (0.1, -0.1),
            "intercept": 0.1,
            "slope": 0.2,
        },
        strategy="STR-THRU",
        driver="implied_t1",
        alpha=0.5,
        cutoff="2026-10-01",
    )
    with pytest.raises(NightlySourceBundleRefusal) as missing_map:
        _release_payoff_declarations(
            binding, "STR-THRU", (line,), "2026-10-20", row)
    assert missing_map.value.code == "MODEL_NOT_READY"


def test_runup_refuses_duplicate_artifacts_at_latest_eligible_cutoff():
    def surface():
        return make_payoff_surface_artifact(
            {
                "n": 2,
                "resid_sd": 0.1,
                "r": None,
                "residuals": (0.1, -0.1),
                "coefficients": (0.0,) * len(RUNUP_TERMS),
            },
            alpha=0.5,
            cutoff="2026-10-15",
        )

    binding = SimpleNamespace(
        payoff_artifacts={"STR-RUNUP": (surface(), surface())})
    with pytest.raises(NightlySourceBundleRefusal) as raised:
        _surface(binding, cutoff="2026-10-15", fill_alpha=0.5)
    assert raised.value.code == "MODEL_NOT_READY"


def test_roundtrip_restores_typed_artifacts_and_recalibrates_win_model():
    line = make_payoff_line_artifact(
        {
            "n": 2,
            "resid_sd": 0.1,
            "r": None,
            "residuals": (0.1, -0.1),
            "intercept": 0.1,
            "slope": 0.2,
        },
        strategy="STR-THRU",
        driver="implied_t1",
        alpha=0.5,
        cutoff="2026-10-15",
    )
    recalibration = make_recalibration_map_artifact(
        {
            "n": 10,
            "base_rate": 0.5,
            "x_thresholds": (0.0, 1.0),
            "y_thresholds": (0.2, 0.8),
        },
        strategy="STR-THRU",
        alpha=0.5,
        cutoff="2026-10-15",
        min_pairs=10,
    )
    receipts = tuple(
        StageReceipt(stage, "declared-input", "declared-output")
        for stage in STAGE_NAMES if stage != "diagnostics")
    inputs = NativeScoreInputs(
        context={}, features={}, forecast={}, geometry=None, pricing=None,
        analogs={}, simulation={}, gate={}, chooser={}, diagnostics={},
        source_ref="roundtrip-regression", stage_receipts=receipts,
        model={"payoff_artifact": line, "recalibration_artifact": recalibration},
    )

    doc = to_document(inputs)
    assert isinstance(doc["model"]["recalibration_artifact"], dict)

    loaded = _load_native_score_inputs(doc)
    assert isinstance(loaded.model["payoff_artifact"], PayoffLineArtifact)
    assert isinstance(loaded.model["recalibration_artifact"], RecalibrationMapArtifact)

    flags = []
    scored = _recalibrated_output(
        loaded.model,
        {"before": "2026-10-15", "recalibration_bound_cutoff": "2026-10-15"},
        "STR-THRU", {"fill": 0.5}, {"win_model": 0.25}, flags)
    assert flags == []
    assert scored["win_model"] == pytest.approx(0.35)


def test_str_thru_model_stage_scores_known_hand_computed_return():
    line = make_payoff_line_artifact(
        {
            "n": 2,
            "resid_sd": 0.0,
            "r": None,
            "residuals": (0.0,),
            "intercept": 0.05,
            "slope": 0.5,
        },
        strategy="STR-THRU",
        driver="implied_t1",
        alpha=0.5,
        cutoff="2026-10-15",
    )
    block = {
        "payoff_artifact": line,
        "payoff_recipe": {
            "before": "2026-10-15",
            "bound_cutoff": "2026-10-15",
            "seed": 7,
            "draw_count": 4,
        },
        "model_residual_rows": (
            {"prediction": 0.10, "residual": 0.0},
        ),
    }
    values = {
        "fill": 0.5,
        "driver_prediction": 0.10,
        "spot": 100.0,
        "entry_cost": 8.0,
    }
    flags = []

    output = _execute_model(
        SimpleNamespace(model=block), "STR-THRU", values, flags)

    # Mutating the intercept from 0.05 to 0.10 (or the slope from 0.5 to 0.6)
    # changes the terminal value, so exp_pnl_model would no longer be 0.25.
    assert flags == []
    assert output["exp_pnl_model"] == pytest.approx(0.25)
    assert output["win_model"] == pytest.approx(1.0)
