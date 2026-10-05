import math
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from engine.v2.models.contracts import ModelBinding
from engine.v2.scoring.frozen_executor import FrozenStageExecutor, FrozenStageRefusal
from engine.v2.scoring.nightly_source_bundle import (
    NightlySourceBundleRefusal,
    assemble_nightly_source_bundle,
    quote_domain_map,
    validated_as_of,
)
from engine.v2.scoring.source_inputs import build_native_score_inputs
from engine.v2.scoring.stages import assemble_native_values, flags_refuse


def _valid_kwargs(**overrides):
    base = dict(
        source_ref="test:nightly:v1",
        strategy="STR-THRU",
        as_of="2026-01-10",
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
             "bid": 1.0, "ask": 1.2, "observed_at": "2026-01-09"},
            {"right": "P", "strike": 100.0, "expiry": "2026-01-16",
             "bid": 1.1, "ask": 1.3, "observed_at": "2026-01-09"},
        ],
        quote_status=None,
        feature_names=("signal", "pred_abs_move", "missing_one"),
    )
    base.update(overrides)
    return base


def test_happy_path_builds_expected_bundle():
    bundle = assemble_nightly_source_bundle(**_valid_kwargs())
    assert bundle.context["ticker"] == "TEST"
    assert bundle.feature_vector == {"signal": 1.5, "pred_abs_move": 0.05}
    assert bundle.feature_missing_mask == {
        "signal": False, "pred_abs_move": False, "missing_one": True}
    assert "missing_one" not in bundle.feature_vector
    assert bundle.raw_quotes == {
        "C:100.0:2026-01-16": {"bid": 1.0, "ask": 1.2},
        "P:100.0:2026-01-16": {"bid": 1.1, "ask": 1.3},
    }
    assert bundle.model_identity == {}
    assert bundle.forecast_recipes == {}


def test_missing_calendar_row_refuses():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(calendar_row=None))
    assert exc.value.code == "MISSING_STAGED_INPUT"


def test_calendar_row_missing_key_refuses():
    calendar_row = dict(_valid_kwargs()["calendar_row"])
    calendar_row.pop("spot")
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(calendar_row=calendar_row))
    assert exc.value.code == "MISSING_STAGED_INPUT"
    assert "spot" in str(exc.value)


def test_missing_panel_row_refuses():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(panel_row=None))
    assert exc.value.code == "MISSING_STAGED_INPUT"


def test_missing_tier4_row_refuses():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(tier4_row=None))
    assert exc.value.code == "MISSING_STAGED_INPUT"


def test_panel_row_missing_date_refuses():
    # The real panel column is "date" (engine/features.py's _KEY_COLUMNS),
    # never "observed_at" -- panel.parquet has never carried that name.
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(panel_row={"signal": 1.5}))
    assert exc.value.code == "MISSING_STAGED_INPUT"
    assert "date" in str(exc.value)


def test_missing_quote_rows_refuses():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(quote_rows=None))
    assert exc.value.code == "MISSING_STAGED_INPUT"


def test_quote_row_missing_observed_at_refuses():
    # A quote row without its own observed_at must never reach raw_quotes
    # unchecked against as_of -- skipping it silently (the prior behavior)
    # is worse than "unchecked and flagged," it is genuinely unvalidated.
    quote_rows = [
        {"right": "C", "strike": 100.0, "expiry": "2026-01-16",
         "bid": 1.0, "ask": 1.2},  # no observed_at
    ]
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(quote_rows=quote_rows))
    assert exc.value.code == "MISSING_STAGED_INPUT"
    assert "quote_rows[0]" in str(exc.value)


def test_quote_rows_accepts_a_tuple():
    # quote_rows is typed as Sequence[Mapping[str, Any]], not list; a tuple
    # (a valid Sequence) must not be misread as an absent/wrong-shaped input.
    kwargs = _valid_kwargs()
    bundle = assemble_nightly_source_bundle(**_valid_kwargs(
        quote_rows=tuple(kwargs["quote_rows"])))
    assert bundle.raw_quotes == {
        "C:100.0:2026-01-16": {"bid": 1.0, "ask": 1.2},
        "P:100.0:2026-01-16": {"bid": 1.1, "ask": 1.3},
    }


def test_partial_feature_vector_marks_missing_mask():
    bundle = assemble_nightly_source_bundle(**_valid_kwargs())
    assert "missing_one" not in bundle.feature_vector
    assert bundle.feature_missing_mask["missing_one"] is True


# --- R3 (was R4-20/CodeRabbit BLOCK item 3): the bundle must not classify
# feature values itself; it passes them through exactly as staged and lets
# the real consumer (FrozenStageExecutor._row) decide missing vs. invalid. ---

_PARITY_VALUES = (
    None, float("nan"), np.float32("nan"), pd.NA, float("inf"),
    "nan", Decimal("NaN"), "abc", 1.0,
)


@pytest.mark.parametrize("raw", _PARITY_VALUES)
def test_feature_value_passes_through_unchanged_by_identity(raw):
    bundle = assemble_nightly_source_bundle(**_valid_kwargs(
        tier4_row={"pred_abs_move": raw, "pred_abs_move_fold_start": "2026-01-01"}))
    assert bundle.feature_vector["pred_abs_move"] is raw
    assert bundle.feature_missing_mask["pred_abs_move"] is False


def _executor_classification(value):
    """The real classification (FrozenStageExecutor._row, frozen_executor.py)
    for one raw feature value, independent of any bundle."""
    binding = ModelBinding(
        binding_id="parity-test", model_id="parity-test", role="parity-test",
        strategy_id="STR-THRU", decision_clock_id="entry-close",
        adapter="json-linear.v1", feature_order=("pred_abs_move",),
        output_names=("prediction",), members=(),
    )
    try:
        FrozenStageExecutor._row(binding, {"pred_abs_move": value})
    except FrozenStageRefusal as exc:
        return exc.code
    return "OK"


@pytest.mark.parametrize("raw", _PARITY_VALUES)
def test_bundle_feature_value_matches_real_executor_classification(raw):
    # Runs each raw value through bundle -> feature_vector, then through the
    # real executor classification, and asserts that equals running the SAME
    # classification directly on the untouched raw value -- proving the
    # bundle changed nothing about what the value means.
    bundle = assemble_nightly_source_bundle(**_valid_kwargs(
        tier4_row={"pred_abs_move": raw, "pred_abs_move_fold_start": "2026-01-01"}))
    bundled_value = bundle.feature_vector["pred_abs_move"]
    assert _executor_classification(bundled_value) == _executor_classification(raw)


# --- R4 (item 4): input validation ---

@pytest.mark.parametrize("spot", [0.0, -5.0, float("nan"), float("inf"), "abc"])
def test_invalid_spot_refuses(spot):
    calendar_row = dict(_valid_kwargs()["calendar_row"])
    calendar_row["spot"] = spot
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(calendar_row=calendar_row))
    assert exc.value.code == "INVALID_SPOT"


def test_feature_names_string_input_refuses():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(feature_names="signal"))
    assert exc.value.code == "INVALID_FEATURE_NAMES"


def test_feature_names_empty_string_entry_refuses():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(feature_names=("signal", "")))
    assert exc.value.code == "INVALID_FEATURE_NAMES"


def test_feature_names_non_str_entry_refuses():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(feature_names=("signal", 123)))
    assert exc.value.code == "INVALID_FEATURE_NAMES"


def test_feature_names_duplicate_refuses():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(feature_names=("signal", "signal")))
    assert exc.value.code == "INVALID_FEATURE_NAMES"


# --- R1 leakage (item 1): planted leaks, one per class ---

def test_panel_outcome_columns_matches_legacy():
    # Guards against hand-picking: the denylist must equal legacy's own
    # OUTCOME_COLUMNS (engine/features.py), not an independently-guessed set.
    from engine.features import OUTCOME_COLUMNS

    from engine.v2.scoring import nightly_source_bundle as nsb
    assert frozenset(OUTCOME_COLUMNS) == nsb._PANEL_OUTCOME_COLUMNS


def test_planted_leak_realized_outcome_move_is_rejected():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            feature_names=("signal", "move"),
            panel_row={"date": "2026-01-15", "signal": 1.5, "move": 0.02},
        ))
    assert exc.value.code == "LEAKED_FEATURE_NAME"
    assert "move" in str(exc.value)


def test_planted_leak_realized_outcome_abs_move_is_rejected():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            feature_names=("signal", "abs_move"),
            panel_row={"date": "2026-01-15", "signal": 1.5, "abs_move": 0.02},
        ))
    assert exc.value.code == "LEAKED_FEATURE_NAME"
    assert "abs_move" in str(exc.value)


def test_planted_leak_driver_name_is_rejected():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            driver_name="custom_driver",
            feature_names=("signal", "custom_driver"),
            panel_row={"date": "2026-01-15", "signal": 1.5, "custom_driver": 0.02},
        ))
    assert exc.value.code == "LEAKED_FEATURE_NAME"
    assert "custom_driver" in str(exc.value)


def test_planted_leak_tier4_fold_start_column_is_rejected():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            feature_names=("signal", "pred_abs_move_fold_start"),
        ))
    assert exc.value.code == "LEAKED_FEATURE_NAME"
    assert "pred_abs_move_fold_start" in str(exc.value)


def test_planted_leak_tier4_model_id_column_is_rejected():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            feature_names=("signal", "pred_abs_move_model_id"),
            tier4_row={
                "pred_abs_move": 0.05, "pred_abs_move_fold_start": "2026-01-01",
                "pred_abs_move_model_id": "m1",
            },
        ))
    assert exc.value.code == "LEAKED_FEATURE_NAME"
    assert "pred_abs_move_model_id" in str(exc.value)


def test_planted_leak_tier3_snapshot_is_rejected():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            feature_names=("signal", "tier3_snapshot"),
            tier4_row={
                "pred_abs_move": 0.05, "pred_abs_move_fold_start": "2026-01-01",
                "tier3_snapshot": "snap-1",
            },
        ))
    assert exc.value.code == "LEAKED_FEATURE_NAME"
    assert "tier3_snapshot" in str(exc.value)


def test_planted_leak_pred_iv_crush_30_band_is_rejected():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            feature_names=("signal", "pred_iv_crush_30_p10"),
            tier4_row={
                "pred_abs_move": 0.05, "pred_abs_move_fold_start": "2026-01-01",
                "pred_iv_crush_30_p10": 0.01,
            },
        ))
    assert exc.value.code == "LEAKED_FEATURE_NAME"
    assert "pred_iv_crush_30_p10" in str(exc.value)


def test_planted_answer_field_in_tier4_row_is_rejected():
    # source_inputs._ANSWER_FIELDS is still enforced independently of the
    # new leak denylist above (a calculated SCORING answer, not a raw
    # source-table column); "gate_pass" is not itself a panel/Tier-4 stamp
    # column, so it must still be caught by _reject_answers on the
    # assembled feature_vector.
    with pytest.raises(ValueError) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            feature_names=("signal", "gate_pass"),
            tier4_row={
                "pred_abs_move": 0.05, "pred_abs_move_fold_start": "2026-01-01",
                "gate_pass": 1.0, "gate_pass_fold_start": "2026-01-01",
            },
        ))
    assert "gate_pass" in str(exc.value)


def test_no_answer_field_in_clean_bundle():
    bundle = assemble_nightly_source_bundle(**_valid_kwargs())
    assert bundle.feature_vector


# --- R2 leakage (item 2): the real stamp/as-of contract ---
#
# Second Opus gate (re-review of 042f794): the panel's own "date" column is
# the EVENT date (engine/features.py::live_features sets "date": event_date),
# never an observation timestamp -- the baseline fixture's panel_row.date
# ("2026-01-15") equals calendar_row.event_date, exactly the real shape:
# a genuine upcoming event, dated strictly after as_of ("2026-01-10"), must
# be allowed. test_happy_path_builds_expected_bundle already proves this
# (it uses the unmodified baseline); test_future_event_dates_are_not_post_as_of
# documents it explicitly.

def test_planted_wrong_event_panel_row_is_rejected():
    # The one invariant this module CAN check without a persisted decision
    # anchor (see _checked_against_as_of's docstring for the escalated gap):
    # panel_row must describe the SAME event calendar_row names.
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            panel_row={"date": "2026-01-11", "signal": 1.5}))
    assert exc.value.code == "PANEL_ROW_WRONG_EVENT"
    assert "2026-01-11" in str(exc.value) and "2026-01-15" in str(exc.value)


def test_missing_fold_start_for_used_tier4_feature_refuses():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            tier4_row={"pred_abs_move": 0.05}))  # no pred_abs_move_fold_start
    assert exc.value.code == "MISSING_STAGED_INPUT"
    assert "pred_abs_move_fold_start" in str(exc.value)


def test_tier4_fold_start_after_as_of_is_rejected():
    # Real-shaped row from the brief: fold_start 2026-02-01 with as_of
    # 2026-01-10 must refuse.
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            as_of="2026-01-10",
            tier4_row={"pred_abs_move": 0.05, "pred_abs_move_fold_start": "2026-02-01"},
        ))
    assert exc.value.code == "POST_AS_OF_ROW"
    assert "pred_abs_move_fold_start" in str(exc.value)


def test_tier4_fold_start_before_as_of_is_allowed():
    bundle = assemble_nightly_source_bundle(**_valid_kwargs(
        as_of="2026-01-10",
        tier4_row={"pred_abs_move": 0.05, "pred_abs_move_fold_start": "2026-01-01"},
    ))
    assert bundle.feature_vector["pred_abs_move"] == 0.05


def test_tier4_band_column_uses_base_metric_fold_start():
    # Second Opus gate finding #2: a band column (pred_abs_move_p10) has no
    # "pred_abs_move_p10_fold_start" column in the real schema -- it is
    # stamped by its BASE metric's own fold_start. A wrong (band-suffixed)
    # key would raise MISSING_STAGED_INPUT even with the real column staged.
    bundle = assemble_nightly_source_bundle(**_valid_kwargs(
        feature_names=("signal", "pred_abs_move_p10"),
        tier4_row={
            "pred_abs_move_p10": 0.03,
            "pred_abs_move_fold_start": "2026-01-01",  # base metric's stamp
        },
    ))
    assert bundle.feature_vector["pred_abs_move_p10"] == 0.03


def test_tier4_band_column_fold_start_after_as_of_is_rejected():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            as_of="2026-01-10",
            feature_names=("signal", "pred_abs_move_p10"),
            tier4_row={
                "pred_abs_move_p10": 0.03,
                "pred_abs_move_fold_start": "2026-02-01",
            },
        ))
    assert exc.value.code == "POST_AS_OF_ROW"
    assert "pred_abs_move_fold_start" in str(exc.value)


# --- Second Opus gate finding #2: a NULL Tier-4 forecast is a missing
# value, not a leakage refusal. Measured against the real
# tier4_forecasts.parquet: 108,320 of 199,973 pred_abs_move rows are NULL,
# always paired with a NULL pred_abs_move_fold_start. ---

def test_null_tier4_forecast_with_missing_fold_start_is_allowed():
    bundle = assemble_nightly_source_bundle(**_valid_kwargs(
        tier4_row={"pred_abs_move": None}))  # no pred_abs_move_fold_start at all
    assert bundle.feature_vector["pred_abs_move"] is None
    assert bundle.feature_missing_mask["pred_abs_move"] is False


def test_null_tier4_forecast_with_nat_fold_start_is_allowed():
    bundle = assemble_nightly_source_bundle(**_valid_kwargs(
        tier4_row={"pred_abs_move": float("nan"), "pred_abs_move_fold_start": pd.NaT}))
    assert math.isnan(bundle.feature_vector["pred_abs_move"])
    assert bundle.feature_missing_mask["pred_abs_move"] is False


def test_real_shaped_mix_of_null_and_non_null_tier4_forecasts_assembles():
    # A real-shaped tier4_row: some forecasts null (with a null fold_start,
    # the measured real pairing), some genuinely fit (with a real,
    # in-range fold_start) -- the bundle must assemble both.
    bundle = assemble_nightly_source_bundle(**_valid_kwargs(
        feature_names=("signal", "pred_abs_move", "pred_im_t1_d14"),
        tier4_row={
            "pred_abs_move": 0.05, "pred_abs_move_fold_start": "2026-01-01",
            "pred_im_t1_d14": None, "pred_im_t1_d14_fold_start": None,
        },
    ))
    assert bundle.feature_vector["pred_abs_move"] == 0.05
    assert bundle.feature_vector["pred_im_t1_d14"] is None
    assert bundle.feature_missing_mask == {
        "signal": False, "pred_abs_move": False, "pred_im_t1_d14": False}


def test_planted_post_as_of_quote_row_is_rejected():
    quote_rows = [dict(row) for row in _valid_kwargs()["quote_rows"]]
    quote_rows[1]["observed_at"] = "2026-01-11"
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(quote_rows=quote_rows))
    assert exc.value.code == "POST_AS_OF_ROW"


def test_planted_post_as_of_calendar_observed_through_is_rejected():
    calendar_row = dict(_valid_kwargs()["calendar_row"])
    calendar_row["calendar_observed_through"] = "2026-01-11"
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(calendar_row=calendar_row))
    assert exc.value.code == "POST_AS_OF_ROW"


def test_future_event_dates_are_not_post_as_of():
    # Inverse control for the date invariant: the baseline calendar's
    # event_date/entry_date/exit_date/expiry (2026-01-15/16) all fall strictly
    # after as_of (2026-01-10), and assembling them without a POST_AS_OF_ROW
    # refusal proves event dates stay exempt (see also
    # test_happy_path_builds_expected_bundle).
    bundle = assemble_nightly_source_bundle(**_valid_kwargs())
    assert bundle.context["event_date"] == "2026-01-15"
    assert bundle.context["expiry"] == "2026-01-16"


def test_same_inputs_produce_equal_bundle():
    # `==` is a valid probe here ONLY because the baseline carries no NaN
    # pass-through value; see test_same_inputs_with_nan_are_field_identical
    # for the general case (ARCHITECTURE.md's determinism claim was fixed
    # to stop overstating `==` as always valid, second Opus gate finding #3).
    first = assemble_nightly_source_bundle(**_valid_kwargs())
    second = assemble_nightly_source_bundle(**_valid_kwargs())
    assert first == second


def test_same_inputs_with_nan_are_field_identical_but_not_equal():
    # Second Opus gate finding #3: with a pass-through NaN (a null Tier-4
    # forecast), two calls with identical arguments are NOT `==` -- NaN !=
    # NaN under IEEE 754 -- even though every field was built the same way.
    # Determinism means reproducibility of the underlying data, not that
    # bare `==` is a valid equality probe once a NaN is present.
    # Two DISTINCT NaN objects (not the same reused float): CPython's
    # container equality has an identity fast-path (`x is x` short-circuits
    # to True even for a NaN), so reusing one nan object across both calls
    # would silently pass `==` for the wrong reason and hide the real bug.
    first = assemble_nightly_source_bundle(**_valid_kwargs(
        tier4_row={"pred_abs_move": float("nan")}))
    second = assemble_nightly_source_bundle(**_valid_kwargs(
        tier4_row={"pred_abs_move": float("nan")}))
    assert first != second  # bare `==` misreads this as different
    assert math.isnan(first.feature_vector["pred_abs_move"])
    assert math.isnan(second.feature_vector["pred_abs_move"])
    assert first.context == second.context
    assert first.raw_quotes == second.raw_quotes
    assert first.feature_missing_mask == second.feature_missing_mask


def test_validated_as_of_rejects_none():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        validated_as_of(None)
    assert exc.value.code == "MISSING_STAGED_INPUT"


def test_validated_as_of_rejects_bare_number():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        validated_as_of(20260115)
    assert exc.value.code == "INVALID_DATE"


def test_validated_as_of_rejects_bool():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        validated_as_of(True)
    assert exc.value.code == "INVALID_DATE"


def test_validated_as_of_rejects_tz_aware():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        validated_as_of(pd.Timestamp("2026-01-15", tz="UTC"))
    assert exc.value.code == "INVALID_DATE"


def test_validated_as_of_rejects_unparseable():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        validated_as_of("not-a-date")
    assert exc.value.code == "INVALID_DATE"


def test_validated_as_of_accepts_naive_date():
    ts = validated_as_of("2026-01-15")
    assert isinstance(ts, pd.Timestamp)
    assert ts == pd.Timestamp("2026-01-15")


def test_quote_domain_map_rejects_incomplete_quote():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        quote_domain_map([{"right": "C", "strike": 100.0}])
    assert exc.value.code == "INVALID_QUOTE_DOMAIN"


def test_quote_domain_map_rejects_crossed_quote():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        quote_domain_map([{"right": "C", "strike": 100.0, "expiry": "2026-01-16",
                           "bid": 2.0, "ask": 1.0}])
    assert exc.value.code == "INVALID_QUOTE_DOMAIN"


def test_quote_domain_map_rejects_conflicting_duplicate():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        quote_domain_map([
            {"right": "C", "strike": 100.0, "expiry": "2026-01-16",
             "bid": 1.0, "ask": 1.2},
            {"right": "C", "strike": 100.0, "expiry": "2026-01-16",
             "bid": 1.1, "ask": 1.3},
        ])
    assert exc.value.code == "INVALID_QUOTE_DOMAIN"


def test_quote_domain_map_empty_domain_requires_status():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        quote_domain_map([], quote_status=None)
    assert exc.value.code == "MISSING_STAGED_INPUT"


def test_quote_domain_map_empty_domain_with_status_ok():
    assert quote_domain_map([], quote_status="empty") == {}
    assert quote_domain_map([], quote_status="not_reached") == {}


def test_quote_domain_map_none_rows_with_empty_status_still_refuses():
    # Behavior-identical to the original _quote_map: an empty *status* only
    # excuses an empty *list* ([]), not a wholly absent domain (None) -- the
    # two are not interchangeable, matching the original's `rows != []` check.
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        quote_domain_map(None, quote_status="not_reached")
    assert exc.value.code == "INVALID_QUOTE_DOMAIN"


def test_quote_domain_map_matches_capture_call_site():
    # capture_tier0_corpus.quote_domain_map is the SAME imported function
    # object as this module's, so comparing the two calls to each other
    # would compare the implementation with itself, vacuously. Assert
    # against an independently-computed expected map instead, covering both
    # the CALL->C normalization and a second strike at the same expiry.
    from tools.capture_tier0_corpus import quote_domain_map as capture_quote_domain_map

    rows = [
        {"right": "C", "strike": 100.0, "expiry": "2026-01-16",
         "bid": 1.0, "ask": 1.2},
        {"right": "P", "strike": 95.0, "expiry": "2026-01-16",
         "bid": 0.5, "ask": 0.7},
        {"right": "CALL", "strike": 105.0, "expiry": "2026-01-16",
         "bid": 0.2, "ask": 0.3},
    ]
    expected = {
        "C:100.0:2026-01-16": {"bid": 1.0, "ask": 1.2},
        "P:95.0:2026-01-16": {"bid": 0.5, "ask": 0.7},
        "C:105.0:2026-01-16": {"bid": 0.2, "ask": 0.3},
    }
    assert quote_domain_map(rows) == expected
    assert capture_quote_domain_map(rows) == expected


# The capture shim's message-fidelity regression
# (tools.capture_tier0_corpus's real quote_domain_map call site preserving
# .detail, not str(exc), through _captured_blocks) lives in
# tests/test_phase4_capture_strict.py::test_probe_preserves_incomplete_quote_message,
# which exercises the real call site through native_inputs_from_capture. A
# synthetic StrictTraceCaptureError built directly in this test file (as a
# prior version of this test did) never executes that call site's own
# exception translation, so it cannot catch a regression there.


def test_missing_panel_anchor_refuses():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(panel_anchor=None))
    assert exc.value.code == "MISSING_STAGED_INPUT"
    assert "panel_row.anchor" in exc.value.detail


def test_planted_post_as_of_panel_anchor_is_rejected():
    # The exact leak issue #53 reports: a forward/upcoming event (event_date
    # after as_of, a legitimate board-looking request) whose panel_row was
    # staged with a decision anchor that postdates as_of -- e.g. a
    # persisted panel.parquet row whose market-state features were computed
    # using data observed after this score's as_of. Before this fix, this
    # call silently succeeded.
    kwargs = _valid_kwargs(
        as_of="2026-01-02",
        calendar_row={
            "ticker": "TEST", "event_date": "2026-01-15",
            "entry_date": "2026-01-15", "exit_date": "2026-01-16",
            "expiry": "2026-01-16", "spot": 100.0,
            "calendar_observed_through": "2026-01-01",
        },
        panel_row={"date": "2026-01-15", "signal": 1.5},
        panel_anchor="2026-01-10",  # after as_of (2026-01-02)
        quote_rows=[
            {"right": "C", "strike": 100.0, "expiry": "2026-01-16",
             "bid": 1.0, "ask": 1.2, "observed_at": "2026-01-01"},
        ],
    )
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**kwargs)
    assert exc.value.code == "POST_AS_OF_ROW"
    assert "panel_row.anchor" in exc.value.detail


def test_forward_event_with_panel_anchor_at_as_of_is_allowed():
    # The correct live_features(as_of=...) shape for a genuine forward
    # event: the decision anchor equals the score's own as_of, strictly
    # before the future event_date. Proves the fix does not regress the
    # legitimate forward-scoring path.
    kwargs = _valid_kwargs(
        as_of="2026-01-02",
        calendar_row={
            "ticker": "TEST", "event_date": "2026-01-15",
            "entry_date": "2026-01-15", "exit_date": "2026-01-16",
            "expiry": "2026-01-16", "spot": 100.0,
            "calendar_observed_through": "2026-01-01",
        },
        panel_row={"date": "2026-01-15", "signal": 1.5},
        panel_anchor="2026-01-02",  # == as_of
        quote_rows=[
            {"right": "C", "strike": 100.0, "expiry": "2026-01-16",
             "bid": 1.0, "ask": 1.2, "observed_at": "2026-01-01"},
        ],
    )
    bundle = assemble_nightly_source_bundle(**kwargs)
    assert bundle.feature_vector["signal"] == 1.5


def test_panel_anchor_before_as_of_is_allowed():
    kwargs = _valid_kwargs(panel_anchor="2026-01-09")  # before as_of (2026-01-10)
    bundle = assemble_nightly_source_bundle(**kwargs)
    assert bundle.feature_vector["signal"] == 1.5


def test_historical_row_panel_anchor_equal_event_date_is_allowed():
    # The historical/realized-event shape: panel_anchor is panel_row["date"]
    # itself (ANCHOR_COLUMNS all equal date for a historical row), and
    # as_of is on/after the already-realized event.
    kwargs = _valid_kwargs(
        as_of="2026-01-20",
        calendar_row={
            "ticker": "TEST", "event_date": "2026-01-15",
            "entry_date": "2026-01-15", "exit_date": "2026-01-16",
            "expiry": "2026-01-16", "spot": 100.0,
            "calendar_observed_through": "2026-01-20",
        },
        panel_row={"date": "2026-01-15", "signal": 1.5},
        panel_anchor="2026-01-15",  # == panel_row["date"] == event_date
        quote_rows=[
            {"right": "C", "strike": 100.0, "expiry": "2026-01-16",
             "bid": 1.0, "ask": 1.2, "observed_at": "2026-01-20"},
        ],
    )
    bundle = assemble_nightly_source_bundle(**kwargs)
    assert bundle.feature_vector["signal"] == 1.5


# --- Issue #169: quote staleness provenance in context ---

def test_quote_max_age_sessions_and_earliest_quote_date_go_to_context():
    # stages._check_stale_quote needs the caller's session budget and one
    # canonical observation date (the earliest quote observed_at, so rows
    # disagreeing on date are treated conservatively) recorded in context --
    # while raw_quotes stays the bid/ask domain, never gaining provenance
    # keys of its own.
    quote_rows = [dict(row) for row in _valid_kwargs()["quote_rows"]]
    quote_rows[0]["observed_at"] = "2026-01-08"  # row[1] stays "2026-01-09"; both <= as_of
    bundle = assemble_nightly_source_bundle(**_valid_kwargs(
        quote_rows=quote_rows, quote_max_age_sessions=1))
    assert bundle.context["quote_max_age_sessions"] == 1
    assert bundle.context["quote_date"] == "2026-01-08"
    assert bundle.raw_quotes == {
        "C:100.0:2026-01-16": {"bid": 1.0, "ask": 1.2},
        "P:100.0:2026-01-16": {"bid": 1.1, "ask": 1.3},
    }
    assert all(set(quote) == {"bid", "ask"} for quote in bundle.raw_quotes.values())


def test_quote_latest_date_after_entry_refuses_assembled_bundle():
    # Gate round 3 regression (issue #169): a put observed AFTER entry but
    # within as_of must not hide behind the entry-day call. Assembly records
    # both provenance dates; scoring judges the latest observation on its own
    # date -- future-dated relative to entry -- and flags the quote unusable
    # as the non-advisory NO_CHAIN without raising. Pre-fix, quote_latest_date
    # was never recorded, so the entry-day call alone looked fresh. The
    # declared driver_prediction recipe (a pass-through, not an external
    # fixture) keeps build_native_score_inputs from refusing the undeclared
    # strategy contract, so the refusal is the quote's, not the recipe's.
    kwargs = _valid_kwargs(
        as_of="2026-01-16",
        quote_max_age_sessions=1,
        quote_rows=[
            {"right": "C", "strike": 100.0, "expiry": "2026-01-16",
             "bid": 1.0, "ask": 1.2, "observed_at": "2026-01-15"},
            {"right": "P", "strike": 100.0, "expiry": "2026-01-16",
             "bid": 1.1, "ask": 1.3, "observed_at": "2026-01-16"},
        ],
        forecast_recipes={"driver_prediction": {"intercept": 0.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:quote-latest-regression"},
        gate_recipe={"model": {"intercept": 0.0, "coefficients": {}}, "threshold": 0.0},
    )
    bundle = assemble_nightly_source_bundle(**kwargs)
    assert bundle.context["quote_date"] == "2026-01-15"
    assert bundle.context["quote_latest_date"] == "2026-01-16"
    values = assemble_native_values(
        build_native_score_inputs(bundle), strategy="STR-THRU")
    assert "NO_CHAIN" in values["flags"]
    assert "STALE_QUOTE" not in values["flags"]
    assert flags_refuse(values["flags"]) is True