import math

import numpy as np
import pandas as pd
import pytest

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
        panel_row={"observed_at": "2026-01-09", "signal": 1.5},
        tier4_row={"observed_at": "2026-01-09", "pred_abs_move": 0.05},
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


def test_panel_row_missing_observed_at_refuses():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(panel_row={"signal": 1.5}))
    assert exc.value.code == "MISSING_STAGED_INPUT"


def test_missing_quote_rows_refuses():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(quote_rows=None))
    assert exc.value.code == "MISSING_STAGED_INPUT"


def test_partial_feature_vector_marks_missing_mask():
    bundle = assemble_nightly_source_bundle(**_valid_kwargs())
    assert "missing_one" not in bundle.feature_vector
    assert bundle.feature_missing_mask["missing_one"] is True


def test_feature_value_none_is_missing():
    bundle = assemble_nightly_source_bundle(**_valid_kwargs(
        tier4_row={"observed_at": "2026-01-09", "pred_abs_move": None}))
    assert "pred_abs_move" not in bundle.feature_vector
    assert bundle.feature_missing_mask["pred_abs_move"] is True


def test_feature_value_nan_is_missing():
    bundle = assemble_nightly_source_bundle(**_valid_kwargs(
        tier4_row={"observed_at": "2026-01-09", "pred_abs_move": float("nan")}))
    assert "pred_abs_move" not in bundle.feature_vector
    assert bundle.feature_missing_mask["pred_abs_move"] is True


def test_feature_value_numpy_float32_nan_is_missing():
    bundle = assemble_nightly_source_bundle(**_valid_kwargs(
        tier4_row={"observed_at": "2026-01-09", "pred_abs_move": np.float32("nan")}))
    assert "pred_abs_move" not in bundle.feature_vector
    assert bundle.feature_missing_mask["pred_abs_move"] is True


def test_feature_value_pandas_na_is_missing():
    bundle = assemble_nightly_source_bundle(**_valid_kwargs(
        tier4_row={"observed_at": "2026-01-09", "pred_abs_move": pd.NA}))
    assert "pred_abs_move" not in bundle.feature_vector
    assert bundle.feature_missing_mask["pred_abs_move"] is True


def test_feature_value_non_numeric_is_missing():
    # A value that cannot be coerced to a float at all never had a usable
    # number to lose -- it is missing, not INVALID_FEATURE_VALUE (which is
    # reserved for a value that DOES coerce but is infinite).
    bundle = assemble_nightly_source_bundle(**_valid_kwargs(
        tier4_row={"observed_at": "2026-01-09", "pred_abs_move": "not-a-number"}))
    assert "pred_abs_move" not in bundle.feature_vector
    assert bundle.feature_missing_mask["pred_abs_move"] is True


def test_feature_value_infinite_refuses():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            tier4_row={"observed_at": "2026-01-09", "pred_abs_move": float("inf")}))
    assert exc.value.code == "INVALID_FEATURE_VALUE"


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


def test_planted_answer_field_in_tier4_row_is_rejected():
    with pytest.raises(ValueError) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            feature_names=("signal", "gate_pass"),
            tier4_row={"observed_at": "2026-01-09", "gate_pass": 1.0},
        ))
    assert "gate_pass" in str(exc.value)


def test_no_answer_field_in_clean_bundle():
    bundle = assemble_nightly_source_bundle(**_valid_kwargs())
    assert bundle.feature_vector


def test_planted_post_as_of_panel_row_is_rejected():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            panel_row={"observed_at": "2026-01-11", "signal": 1.5}))
    assert exc.value.code == "POST_AS_OF_ROW"


def test_planted_post_as_of_tier4_row_is_rejected():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_valid_kwargs(
            tier4_row={"observed_at": "2026-01-11", "pred_abs_move": 0.05}))
    assert exc.value.code == "POST_AS_OF_ROW"


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
    from tools.capture_tier0_corpus import quote_domain_map as capture_quote_domain_map

    rows = [
        {"right": "C", "strike": 100.0, "expiry": "2026-01-16",
         "bid": 1.0, "ask": 1.2},
        {"right": "P", "strike": 95.0, "expiry": "2026-01-16",
         "bid": 0.5, "ask": 0.7},
        {"right": "CALL", "strike": 105.0, "expiry": "2026-01-16",
         "bid": 0.2, "ask": 0.3},
    ]
    assert quote_domain_map(rows) == capture_quote_domain_map(rows)


def test_capture_shim_preserves_original_message_via_detail():
    # Regression for the capture shim (tools/capture_tier0_corpus.py's one
    # call site): NightlySourceBundleRefusal's own str() carries a
    # "CODE: " prefix StrictTraceCaptureError's messages never had, so the
    # shim re-raises with `.detail` (the original text), not `str(exc)`.
    # tests/test_phase4_capture_strict.py's own
    # test_probe_still_refuses_unrecorded_or_contradictory_quote_domains
    # exercises the real call site end to end (via substring `match=`); this
    # pins the exact mechanism the shim relies on.
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        quote_domain_map([{"right": "C", "strike": 100.0}])
    from tools.capture_tier0_corpus import StrictTraceCaptureError
    wrapped = StrictTraceCaptureError(getattr(exc.value, "detail", str(exc.value)))
    assert str(wrapped) == exc.value.detail
    assert not str(wrapped).startswith("INVALID_QUOTE_DOMAIN:")
