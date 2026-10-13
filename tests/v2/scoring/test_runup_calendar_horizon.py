"""RUNUP calendar-owned timing reaches native features; no real release/data."""
from copy import deepcopy

import numpy as np
import pytest

from engine.v2.scoring.nightly_source_bundle import (
    NightlySourceBundleRefusal,
    assemble_nightly_source_bundle,
)
from engine.v2.scoring.source_inputs import build_native_score_inputs
from tests.v2.scoring.test_nightly_source_bundle import _valid_kwargs


def _kwargs(**changes):
    kwargs = _valid_kwargs(strategy="STR-RUNUP", driver_name="im_t1",
                           feature_names=("days_before_print", "signal"))
    kwargs["calendar_row"]["days_before_print"] = 5
    kwargs.update(changes)
    return kwargs


@pytest.mark.parametrize("value", [0, 1, 5, 14.0, np.int64(5)])
def test_horizon_is_in_context_and_requested_features(value):
    kwargs = _kwargs()
    kwargs["calendar_row"]["days_before_print"] = value
    bundle = assemble_nightly_source_bundle(**kwargs)
    assert bundle.context["days_before_print"] == int(value)
    assert type(bundle.context["days_before_print"]) is int
    assert bundle.feature_vector == {"days_before_print": int(value), "signal": 1.5}
    assert bundle.feature_missing_mask == {"days_before_print": False, "signal": False}


@pytest.mark.parametrize("value", [None, True, False, "5", -1, 0.5, float("nan"),
                                   float("inf"), -float("inf"), [], 10**1000])
def test_invalid_staged_horizon_refuses_without_default(value):
    kwargs = _kwargs()
    kwargs["calendar_row"]["days_before_print"] = value
    with pytest.raises(NightlySourceBundleRefusal) as error:
        assemble_nightly_source_bundle(**kwargs)
    assert error.value.code == "INVALID_RUNUP_HORIZON"
    assert error.value.detail == "days_before_print must be a nonnegative session count"


def test_missing_horizon_never_falls_back_to_panel():
    kwargs = _kwargs()
    del kwargs["calendar_row"]["days_before_print"]
    kwargs["panel_row"]["days_before_print"] = 99
    with pytest.raises(NightlySourceBundleRefusal) as error:
        assemble_nightly_source_bundle(**kwargs)
    assert error.value.code == "MISSING_STAGED_INPUT"


@pytest.mark.parametrize("stamp", [None, "2099-01-01"])
def test_calendar_owns_horizon_over_stale_rows_without_reading_forecast_stamp(stamp):
    kwargs = _kwargs()
    kwargs["panel_row"]["days_before_print"] = 99
    kwargs["tier4_row"]["days_before_print"] = 88
    if stamp is not None:
        kwargs["tier4_row"]["days_before_print_fold_start"] = stamp
    original = deepcopy(kwargs)
    for _ in range(2):
        bundle = assemble_nightly_source_bundle(**kwargs)
        assert bundle.context["days_before_print"] == 5
        assert bundle.feature_vector["days_before_print"] == 5
        assert bundle.feature_missing_mask["days_before_print"] is False
    assert kwargs == original


def test_context_keeps_horizon_when_not_in_feature_order():
    bundle = assemble_nightly_source_bundle(**_kwargs(feature_names=("signal",)))
    assert bundle.context["days_before_print"] == 5
    assert "days_before_print" not in bundle.feature_vector
    assert "days_before_print" not in bundle.feature_missing_mask


def test_real_native_builder_receives_horizon():
    kwargs = _kwargs(
        forecast_recipes={
            "driver_prediction": {"intercept": 1.0, "coefficients": {"days_before_print": 1.0}},
            "runup_move_prediction": {"intercept": 2.0, "coefficients": {}},
        },
        gate_recipe={"model": {"intercept": 0.0, "coefficients": {}}, "threshold": 0.0},
        model_artifact_refs={"driver_prediction": "synthetic-implied", "runup_move_prediction": "synthetic-move"},
    )
    inputs = build_native_score_inputs(assemble_nightly_source_bundle(**kwargs))
    assert inputs.context["days_before_print"] == 5
    assert inputs.features["model_inputs"]["days_before_print"] == 5
    assert inputs.features["missing_mask"]["days_before_print"] is False


def test_through_omits_calendar_horizon_and_projects_panel_horizon():
    kwargs = _valid_kwargs(feature_names=("days_before_print",))
    kwargs["calendar_row"]["days_before_print"] = "not-a-horizon"
    kwargs["panel_row"]["days_before_print"] = 7.5
    bundle = assemble_nightly_source_bundle(**kwargs)
    assert "days_before_print" not in bundle.context
    assert bundle.feature_vector == {"days_before_print": 7.5}
