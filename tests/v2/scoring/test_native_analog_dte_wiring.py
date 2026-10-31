"""The frozen analog stage sees the scorer-computed dte_entry (#681, #653)."""
import math
from dataclasses import replace

import pytest

from engine.v2.models.analog_artifact import make_board_analog_pool_artifact
from engine.v2.models.lineage import Lineage
from engine.v2.scoring.native_analog_query import analog_declarations
from engine.v2.scoring.stages import assemble_native_values
from tests.v2.scoring.test_scorer_computed_dates import _inputs


def _rows():
    # 35 rows in the "26-45" dte band, 25 in "4-10"; every other band equal.
    return [
        (f"r{i:03d}", "2025-01-01", "2025-02-01", "1-10B",
         "26-45" if i < 35 else "4-10", "ATM", "mid", 0.01 * (i % 7 - 3))
        for i in range(60)
    ]


def _analog_inputs(panel_row=None, **overrides):
    artifact = make_board_analog_pool_artifact(
        strategy="STR-THRU", alpha=0.5, cutoff="2026-10-07",
        population_edges=(0.9, 1.1), causal_edges=(0.9, 1.1),
        rows=_rows(), lineage=Lineage())
    declared = analog_declarations(
        [artifact], "STR-THRU", 0.5, "2026-10-07",
        panel_row or {"mcap_log": math.log(5e9)}, "snap-1", "req-key")
    block = {"analog_artifact": declared["analog_artifact"],
             "analog_artifact_recipe": declared["analog_artifact_recipe"],
             "query_features": declared["analog_query"]}
    return replace(_inputs(**overrides), analogs=block)


def test_analog_query_sees_the_priced_dte_entry():
    values = assemble_native_values(_analog_inputs())
    # Priced first leg expires 2026-11-20, quote date 2026-10-08: 43 days -> "26-45".
    assert values["dte_entry"] == 43
    assert "MODEL_NOT_READY" not in values["flags"]
    assert not [f for f in values["flags"] if "ANALOG" in f]
    # Only the 35 "26-45" rows match; a dte_band of None would keep all 60.
    assert values["n_analogs"] == 35


def test_unpriced_row_has_no_dte_band_and_keeps_the_whole_pool():
    # No quotes: pricing refuses, nothing writes dte_entry, the dte dimension
    # is dropped and the matcher keeps every ATM/"1-10B" row.
    values = assemble_native_values(_analog_inputs(quotes={}))
    assert "dte_entry" not in values
    assert "MISSING_PRICING_INPUT" in values["flags"]
    assert values["n_analogs"] == 60


_PANEL = {"mcap_log": math.log(5e10), "entry_im": 0.09, "mean_prior_or_implied": 0.06}


def _analogs_observation(spot, expiry=None):
    seen = []
    assemble_native_values(
        _analog_inputs(_PANEL, context_overrides={"spot": spot}, expiry=expiry),
        observer=seen.append)
    return next(o for o in seen if o.receipt.stage == "analogs")


@pytest.mark.parametrize(("spot", "expiry", "dte_band", "moneyness_band"), [
    (100.0, None, "26-45", "ATM"),           # 43 days
    (106.0, "2026-10-20", "11-25", ">5%"),   # 12 days
])
def test_receipt_records_the_completed_query_actually_used(
        spot, expiry, dte_band, moneyness_band):
    # The receipt's recorded query is the full completed one: the declared
    # mcap/ratio inputs AND the bands computed from the priced values.
    obs = _analogs_observation(spot, expiry)
    assert obs.input_document["inputs"]["query_features"] == {
        "mcap_bucket": ">=10B", "implied_ratio": pytest.approx(1.5),
        "dte_band": dte_band, "moneyness_band": moneyness_band}


def test_different_completed_bands_give_different_input_hashes():
    near = _analogs_observation(100.0)
    far = _analogs_observation(106.0)
    assert near.receipt.input_hash != far.receipt.input_hash
    assert near.receipt.input_hash == _analogs_observation(100.0).receipt.input_hash