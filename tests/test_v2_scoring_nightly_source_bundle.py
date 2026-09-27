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
        panel_row={"date": "2026-01-09", "signal": 1.5},
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
            panel_row={"date": "2026-01-09", "signal": 1.5, "move": 0.02},
        ))
    assert exc.value.code == "LEAKED_FEATURE_NAME"
    assert "move" in str(exc.value)


def test_planted_leak_realized_outcome_abs_move_is_rejected():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            feature_names=("signal", "abs_move"),
            panel_row={"date": "2026-01-09", "signal": 1.5, "abs_move": 0.02},
        ))
    assert exc.value.code == "LEAKED_FEATURE_NAME"
    assert "abs_move" in str(exc.value)


def test_planted_leak_driver_name_is_rejected():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            driver_name="custom_driver",
            feature_names=("signal", "custom_driver"),
            panel_row={"date": "2026-01-09", "signal": 1.5, "custom_driver": 0.02},
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

def test_planted_post_as_of_panel_row_is_rejected():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            panel_row={"date": "2026-01-11", "signal": 1.5}))
    assert exc.value.code == "POST_AS_OF_ROW"


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
    first = assemble_nightly_source_bundle(**_valid_kwargs())
    second = assemble_nightly_source_bundle(**_valid_kwargs())
    assert first == second


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
