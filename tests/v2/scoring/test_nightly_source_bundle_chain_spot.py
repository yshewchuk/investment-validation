"""``chain_spot`` through the nightly source bundle: staged into ``context``,
never computed from, and refused (no bundle) when the quote rows are not
consistent about it.
"""
from __future__ import annotations

import pytest

from engine.v2.foundation import DocumentError
from engine.v2.scoring.financial import financial_diagnostics
from engine.v2.scoring.native_chooser import derive_chooser_columns
from engine.v2.scoring.nightly_source_bundle import (
    NightlySourceBundleRefusal,
    assemble_nightly_source_bundle,
    quote_domain_map,
)
from engine.v2.scoring.source_inputs import build_native_score_inputs
from engine.v2.scoring.stages import assemble_native_values

_AS_OF = "2026-01-10"
_CALENDAR_CLOSE = 100.0


def _rows(**overrides):
    base = [
        {"right": "C", "strike": 100.0, "expiry": "2026-01-16",
         "bid": 1.0, "ask": 1.2, "observed_at": "2026-01-09"},
        {"right": "P", "strike": 100.0, "expiry": "2026-01-16",
         "bid": 1.1, "ask": 1.3, "observed_at": "2026-01-09"},
    ]
    if overrides:
        for row in base:
            row.update(overrides)
    return base


def _kwargs(**overrides):
    base = dict(
        source_ref="test:chain-spot",
        strategy="STR-THRU",
        as_of=_AS_OF,
        calendar_row={
            "ticker": "TEST", "event_date": "2026-01-15",
            "entry_date": "2026-01-15", "exit_date": "2026-01-16",
            "expiry": "2026-01-16", "spot": _CALENDAR_CLOSE,
            "calendar_observed_through": "2026-01-09",
        },
        panel_row={"date": "2026-01-15", "signal": 1.5},
        panel_anchor=_AS_OF,
        tier4_row={"pred_abs_move": 0.05, "pred_abs_move_fold_start": "2026-01-01"},
        quote_rows=_rows(),
        quote_status=None,
        feature_names=("signal",),
        forecast_recipes={"driver_prediction": {"intercept": 0.0, "coefficients": {}}},
        model_artifact_refs={"driver_prediction": "sha256:chain-spot"},
        gate_recipe={"model": {"intercept": 0.0, "coefficients": {}}, "threshold": 0.0},
    )
    base.update(overrides)
    return base


def test_context_carries_the_staged_chain_spot():
    bundle = assemble_nightly_source_bundle(**_kwargs(quote_rows=_rows(chain_spot=87.5)))

    assert bundle.context["chain_spot"] == 87.5
    assert isinstance(bundle.context["chain_spot"], float)
    assert bundle.context["spot"] == _CALENDAR_CLOSE


@pytest.mark.parametrize("spot", [
    None, float("nan"), float("inf"), -float("inf"), 0.0, -10.0,
    True, "not-a-spot", 10**400,
])
def test_source_bundle_builder_refuses_invalid_chain_spot(spot):
    bundle = assemble_nightly_source_bundle(
        **_kwargs(quote_rows=_rows(chain_spot=87.5)))
    bundle.context["chain_spot"] = spot

    with pytest.raises(DocumentError) as excinfo:
        build_native_score_inputs(bundle)

    assert excinfo.value.code == "BAD_VALUE"
    assert excinfo.value.path == "$.context.chain_spot"
    assert bundle.context["chain_spot"] is spot


def test_source_bundle_builder_normalizes_chain_spot_without_mutating_bundle():
    bundle = assemble_nightly_source_bundle(
        **_kwargs(quote_rows=_rows(chain_spot=87.5)))
    bundle.context["chain_spot"] = "64.25"

    inputs = build_native_score_inputs(bundle)

    assert inputs.context["chain_spot"] == 64.25
    assert type(inputs.context["chain_spot"]) is float
    assert bundle.context["chain_spot"] == "64.25"


def test_chain_spot_drives_entry_cost_pct_while_calendar_spot_is_preserved():
    bundle = assemble_nightly_source_bundle(**_kwargs(quote_rows=_rows(chain_spot=87.5)))

    inputs = build_native_score_inputs(bundle)
    values = assemble_native_values(inputs, strategy="STR-THRU")

    assert inputs.context["chain_spot"] == 87.5
    assert inputs.context["spot"] == _CALENDAR_CLOSE
    # entry_cost_pct uses the staged chain spot while `spot` remains the calendar close.
    expected_pct = (1.1 + 1.2) / 87.5 * 100.0
    assert values["entry_cost_pct"] == pytest.approx(expected_pct)
    assert financial_diagnostics(values)["entry_cost_pct"] == pytest.approx(expected_pct)
    chooser = derive_chooser_columns({}, {}, "STR-THRU", values, {}, [])
    assert chooser is not None
    assert chooser["entry_cost_pct"] == pytest.approx(expected_pct)


def test_missing_chain_spot_leaves_entry_cost_pct_absent():
    bundle = assemble_nightly_source_bundle(**_kwargs())

    values = assemble_native_values(
        build_native_score_inputs(bundle), strategy="STR-THRU")

    assert "chain_spot" not in bundle.context
    assert values["entry_cost"] is not None
    assert values.get("entry_cost_pct") is None
    assert financial_diagnostics(values)["entry_cost_pct"] is None


def test_an_empty_slice_carries_no_chain_spot():
    bundle = assemble_nightly_source_bundle(
        **_kwargs(quote_rows=(), quote_status="empty"))

    assert "chain_spot" not in bundle.context


def test_the_key_is_omitted_when_no_row_carries_it():
    bundle = assemble_nightly_source_bundle(**_kwargs(quote_rows=_rows()))

    assert "chain_spot" not in bundle.context


def test_a_partially_present_chain_spot_refuses_the_bundle():
    rows = [dict(_rows()[0], chain_spot=87.5), _rows()[1]]

    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_kwargs(quote_rows=rows))
    assert exc.value.code == "INVALID_QUOTE_DOMAIN"
    assert "quote_rows[1]" in str(exc.value)


def test_a_chain_spot_missing_from_an_earlier_row_refuses_too():
    rows = [_rows()[0], dict(_rows()[1], chain_spot=87.5)]

    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_kwargs(quote_rows=rows))
    assert exc.value.code == "INVALID_QUOTE_DOMAIN"
    assert "quote_rows[0]" in str(exc.value)


def test_rows_that_disagree_on_the_chain_spot_refuse_the_bundle():
    rows = [dict(_rows()[0], chain_spot=87.5), dict(_rows()[1], chain_spot=87.6)]

    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_kwargs(quote_rows=rows))
    assert exc.value.code == "INVALID_QUOTE_DOMAIN"


@pytest.mark.parametrize("spot", [None, float("nan"), float("inf"), 0.0, -87.5, True,
                                  "not-a-spot", 10**400])
def test_an_unusable_chain_spot_refuses_the_bundle(spot):
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(**_kwargs(quote_rows=_rows(chain_spot=spot)))
    assert exc.value.code == "INVALID_QUOTE_DOMAIN"


def test_the_refusal_never_echoes_the_staged_spot_value():
    with pytest.raises(NightlySourceBundleRefusal) as exc:
        assemble_nightly_source_bundle(
            **_kwargs(quote_rows=[dict(_rows()[0], chain_spot=1234.5678),
                                  dict(_rows()[1], chain_spot="not-a-spot")]))

    assert "1234.5678" not in str(exc.value)
    assert "not-a-spot" not in str(exc.value)


def test_quote_domain_map_ignores_the_extra_row_key():
    without = quote_domain_map(_rows(), "recorded")
    with_spot = quote_domain_map(_rows(chain_spot=87.5), "recorded")

    assert with_spot == without
    assert all("chain_spot" not in quote for quote in with_spot.values())


def test_a_repeated_contract_still_merges_with_the_key_present():
    rows = _rows(chain_spot=87.5)
    rows.append(dict(rows[0], observed_at="2026-01-08"))

    domain = quote_domain_map(rows, "recorded")

    assert domain["C:100.0:2026-01-16"]["observed_date"] == "2026-01-08"
