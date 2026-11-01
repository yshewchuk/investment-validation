"""Pure query/declaration helpers for the frozen board analog matcher (#681).

Every case is synthetic: bucket edges, the implied-ratio rule, the
declaration pick, and one real-artifact run through the frozen stage. No
fixture tree, no ``data/`` read, nothing imported from ``tests/``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from engine.v2.models.analog_artifact import (
    board_analog_pool_key,
    make_board_analog_pool_artifact,
)
from engine.v2.models.lineage import Lineage
from engine.v2.scoring.native_analog import FROZEN_ANALOG_RECIPE_FIELDS
from engine.v2.scoring.native_analog_query import (
    BOOTSTRAP_DRAWS,
    CI_QUANTILES,
    MIN_ANALOGS,
    AmbiguousAnalogPool,
    analog_declarations,
    analog_query_base,
    complete_analog_query,
)
from engine.v2.scoring.stages import _execute_frozen_analogs


def test_query_base_buckets_and_ratio():
    panel_row = {
        "mcap_log": math.log(5e9),
        "entry_im": 0.06,
        "mean_prior_or_implied": 0.05,
    }
    query = analog_query_base(panel_row)
    assert query["mcap_bucket"] == "1-10B"
    assert query["implied_ratio"] == pytest.approx(1.2)

    # Edge cases, all in this one test: the bucket boundary is half-open
    # upward, a missing or overflowing market cap gives no bucket, and a
    # missing/zero/non-finite ratio (a bool is not a number) gives none.
    for panel, bucket, ratio in [
        ({"mcap_log": math.log(1.0001e9)}, "1-10B", None),
        ({"mcap_log": math.log(9.99e8)}, "<1B", None),
        ({"mcap_log": math.log(1.0001e10)}, ">=10B", None),
        ({}, None, None),
        ({"mcap_log": 1e9}, None, None),
        ({"entry_im": 0.06, "mean_prior_or_implied": 0.05}, None, pytest.approx(1.2)),
        ({"mcap_log": math.log(5e9), "entry_im": 0.06}, "1-10B", None),
        # The model feature "im" is never the ratio's numerator: only "im",
        # no entry_im, gives no ratio.
        ({"mcap_log": math.log(5e9), "im": 0.06, "mean_prior_or_implied": 0.05},
         "1-10B", None),
        ({"mcap_log": math.log(5e9), "entry_im": 0.06, "mean_prior_or_implied": 0.0},
         "1-10B", None),
        ({"mcap_log": math.log(5e9), "entry_im": float("nan"),
          "mean_prior_or_implied": 0.05}, "1-10B", None),
        ({"mcap_log": math.log(5e9), "entry_im": 0.06,
          "mean_prior_or_implied": True}, "1-10B", None),
    ]:
        got = analog_query_base(panel)
        assert got["mcap_bucket"] == bucket
        assert got["implied_ratio"] == ratio


@pytest.mark.parametrize(
    ("dte_entry", "band"),
    [
        (1, "1-3"),
        (3, "1-3"),
        (0, None),
        (2, "1-3"),
        (4, "4-10"),
        (10, "4-10"),
        (11, "11-25"),
        (25, "11-25"),
        (26, "26-45"),
        (45, "26-45"),
        (46, None),
        (3.5, None),
        (pytest.param(None, None, id="absent")),
    ],
)
def test_complete_query_dte_band(dte_entry, band):
    values = {} if dte_entry is None else {"dte_entry": dte_entry}
    query = {"mcap_bucket": "1-10B", "implied_ratio": None}
    out = complete_analog_query(query, values)
    assert out["dte_band"] == band
    assert out["mcap_bucket"] == "1-10B"
    assert out["implied_ratio"] is None


@pytest.mark.parametrize(
    ("strike", "band"),
    [
        (100.0, "ATM"),
        (101.99, "ATM"),
        (102.0, "2-5%"),
        (104.99, "2-5%"),
        (105.0, ">5%"),
        (90.0, ">5%"),
        (pytest.param(None, None, id="absent")),
    ],
)
def test_complete_query_moneyness_band(strike, band):
    values = {"spot": 100.0}
    if strike is not None:
        values["strike"] = strike
    out = complete_analog_query({"mcap_bucket": "1-10B", "implied_ratio": None}, values)
    assert out["moneyness_band"] == band


@pytest.mark.parametrize(
    ("im", "prior", "ratio"),
    [(0.09, 0.06, 1.5), (0.04, 0.08, 0.5), (0.12, 0.03, 4.0)],
)
def test_query_base_implied_ratio_is_entry_im_over_prior(im, prior, ratio):
    # Distinct ratios, none equal to 1.2 or its inverse: im/prior, not prior/im.
    got = analog_query_base({"entry_im": im, "mean_prior_or_implied": prior})
    assert got["implied_ratio"] == pytest.approx(ratio)


@pytest.mark.parametrize(
    ("spot", "strike", "band"),
    [
        (250.0, 254.9, "ATM"),
        (250.0, 255.0, "2-5%"),
        (250.0, 245.0, "2-5%"),
        (250.0, 262.4, "2-5%"),
        (250.0, 262.5, ">5%"),
        (250.0, 237.0, ">5%"),
        (50.0, 100.0, ">5%"),
        (50.0, 50.5, "ATM"),
    ],
)
def test_complete_query_moneyness_is_relative_to_spot(spot, strike, band):
    # Spots other than 100: a distance-from-100 or absolute-dollar rule fails.
    out = complete_analog_query({"mcap_bucket": "1-10B", "implied_ratio": None},
                                {"spot": spot, "strike": strike})
    assert out["moneyness_band"] == band


_BAD_NUMBERS = [
    pytest.param(None, id="none"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="-inf"),
    pytest.param("abc", id="string"),
    pytest.param(True, id="bool"),
]


@pytest.mark.parametrize("bad", _BAD_NUMBERS)
def test_query_base_drops_mcap_for_every_bad_mcap_log(bad):
    # Dropped (None), never bucketed: -inf would otherwise be exp(-inf)=0 -> "<1B".
    got = analog_query_base({"mcap_log": bad, "entry_im": 0.06, "mean_prior_or_implied": 0.05})
    assert got["mcap_bucket"] is None
    assert got["implied_ratio"] == pytest.approx(1.2)


@pytest.mark.parametrize("bad", _BAD_NUMBERS)
def test_query_base_drops_ratio_for_every_bad_entry_im_or_prior(bad):
    base = {"mcap_log": math.log(5e9)}
    assert analog_query_base({**base, "entry_im": bad, "mean_prior_or_implied": 0.05})[
        "implied_ratio"] is None
    assert analog_query_base({**base, "entry_im": 0.06, "mean_prior_or_implied": bad})[
        "implied_ratio"] is None


def test_query_base_ratio_edge_cases():
    base = {"mcap_log": math.log(5e9)}
    # A non-finite quotient (overflow) is dropped; a negative divisor is a valid number.
    assert analog_query_base({**base, "entry_im": 1e308, "mean_prior_or_implied": 1e-308})[
        "implied_ratio"] is None
    assert analog_query_base({**base, "entry_im": 0.06, "mean_prior_or_implied": -0.05})[
        "implied_ratio"] == pytest.approx(-1.2)
    assert analog_query_base({**base, "entry_im": 0.0, "mean_prior_or_implied": 0.05})[
        "implied_ratio"] == 0.0


def test_query_base_negative_mcap_log_is_a_small_cap():
    assert analog_query_base({"mcap_log": -3.0})["mcap_bucket"] == "<1B"


@pytest.mark.parametrize("bad", _BAD_NUMBERS)
def test_complete_query_drops_bands_for_every_bad_input(bad):
    query = {"mcap_bucket": "1-10B", "implied_ratio": None}
    assert complete_analog_query(query, {"dte_entry": bad})["dte_band"] is None
    for values in ({"spot": 100.0, "strike": bad}, {"spot": bad, "strike": 100.0}):
        assert complete_analog_query(query, values)["moneyness_band"] is None


@pytest.mark.parametrize(("spot", "strike"), [
    (-100.0, 100.0), (100.0, -100.0), (-100.0, -100.0),
    (0.0, 100.0), (100.0, 0.0), (0.0, 0.0),
])
def test_complete_query_non_positive_spot_or_strike_has_no_moneyness(spot, strike):
    # A non-positive price is invalid input, not a moneyness.
    out = complete_analog_query({"mcap_bucket": "1-10B", "implied_ratio": None},
                                {"spot": spot, "strike": strike})
    assert out["moneyness_band"] is None


def test_complete_query_zero_spot_is_none():
    out = complete_analog_query({"mcap_bucket": "1-10B", "implied_ratio": None},
                                {"spot": 0.0, "strike": 100.0})
    assert out["moneyness_band"] is None


@pytest.mark.parametrize(("dte_band", "moneyness_band"), [
    ("1-3", "ATM"), ("11-25", ">5%"), ("26-45", "2-5%")])
def test_complete_query_keeps_each_distinct_existing_band(dte_band, moneyness_band):
    # values would compute "26-45"/"ATM"; every given label survives unchanged.
    out = complete_analog_query(
        {"mcap_bucket": "1-10B", "implied_ratio": None,
         "dte_band": dte_band, "moneyness_band": moneyness_band},
        {"dte_entry": 30, "spot": 100.0, "strike": 100.0})
    assert out["dte_band"] == dte_band
    assert out["moneyness_band"] == moneyness_band


def test_complete_query_does_not_mutate_the_callers_query():
    query = {"mcap_bucket": "1-10B", "implied_ratio": 1.5}
    out = complete_analog_query(query, {"dte_entry": 5, "spot": 100.0, "strike": 101.0})
    assert out is not query
    assert query == {"mcap_bucket": "1-10B", "implied_ratio": 1.5}
    assert out["dte_band"] == "4-10" and out["moneyness_band"] == "ATM"
    # A second call with other values on the SAME query is not polluted by the first.
    again = complete_analog_query(query, {"dte_entry": 30, "spot": 100.0, "strike": 110.0})
    assert again["dte_band"] == "26-45" and again["moneyness_band"] == ">5%"


def test_complete_query_keeps_existing_and_passes_non_mapping():
    kept = complete_analog_query({"mcap_bucket": "1-10B", "implied_ratio": None,
                                  "dte_band": "1-3"},
                                 {"dte_entry": 30, "spot": 100.0, "strike": 100.0})
    assert kept["dte_band"] == "1-3"
    assert kept["moneyness_band"] == "ATM"
    kept = complete_analog_query({"mcap_bucket": "1-10B", "implied_ratio": None,
                                  "moneyness_band": "ATM"},
                                 {"dte_entry": 30, "spot": 100.0, "strike": 120.0})
    assert kept["moneyness_band"] == "ATM"
    assert kept["dte_band"] == "26-45"
    # A key that is present with value None is an explicit "unavailable": it
    # stays None and is never filled from the priced values.
    kept = complete_analog_query({"mcap_bucket": "1-10B", "implied_ratio": None,
                                  "dte_band": None, "moneyness_band": None},
                                 {"dte_entry": 30, "spot": 100.0, "strike": 101.0})
    assert kept["dte_band"] is None
    assert kept["moneyness_band"] is None


@pytest.mark.parametrize("not_a_mapping", [
    None, [("mcap_bucket", "1-10B")], (("mcap_bucket", "1-10B"),), "1-10B", 7])
def test_complete_query_returns_every_non_mapping_unchanged(not_a_mapping):
    # The evaluator then refuses it with INVALID_ANALOG_INPUT; completion never
    # converts, fills or wraps it.
    out = complete_analog_query(not_a_mapping, {"dte_entry": 5, "spot": 100.0, "strike": 101.0})
    assert out is not_a_mapping


@dataclass(frozen=True, kw_only=True)
class _FakeArtifact:
    """A stand-in pool carrying only what ``analog_declarations`` reads."""

    key: tuple
    content_hash: str


def test_declarations_pick_matching_pool():
    match_key = board_analog_pool_key("STR-THRU", 0.5, "2026-01-09")
    artifacts = [
        _FakeArtifact(key=board_analog_pool_key("STR-THRU", 0.5, "2025-12-31"),
                      content_hash="sha256:other"),
        _FakeArtifact(key=match_key, content_hash="sha256:picked"),
        _FakeArtifact(key=board_analog_pool_key("STR-THRU", 0.5, "2026-01-10"),
                      content_hash="sha256:later"),
    ]
    panel_row = {"mcap_log": math.log(5e9), "entry_im": 0.06, "mean_prior_or_implied": 0.05}
    declarations = analog_declarations(
        artifacts, "STR-THRU", 0.5, "2026-01-09", panel_row, "snap-1", "req-key")
    assert declarations["analog_artifact"] is artifacts[1]
    recipe = declarations["analog_artifact_recipe"]
    assert set(recipe) == set(FROZEN_ANALOG_RECIPE_FIELDS)
    assert recipe["cutoff"] == "2026-01-09"
    assert recipe["content_hash"] == "sha256:picked"
    assert recipe["seed_snapshot"] == "snap-1"
    assert recipe["request_key"] == "req-key"
    assert recipe["min_analogs"] == 30 and recipe["min_analogs"] == MIN_ANALOGS
    assert recipe["bootstrap_draws"] == 2000 and recipe["bootstrap_draws"] == BOOTSTRAP_DRAWS
    assert recipe["ci_quantiles"] == (0.05, 0.95)
    assert recipe["ci_quantiles"] == CI_QUANTILES
    assert declarations["analog_query"] == analog_query_base(panel_row)

    # A different cutoff or a different strategy declares no pool at all.
    assert analog_declarations(artifacts, "STR-THRU", 0.5, "2026-01-11",
                               panel_row, "snap-1", "req-key") == {}
    assert analog_declarations(artifacts, "OTHER-STR", 0.5, "2026-01-09",
                               panel_row, "snap-1", "req-key") == {}
    assert analog_declarations([], "STR-THRU", 0.5, "2026-01-09",
                               panel_row, "snap-1", "req-key") == {}


def test_declarations_use_the_given_alpha():
    panel_row = {"mcap_log": math.log(5e9)}
    half = _FakeArtifact(key=board_analog_pool_key("STR-THRU", 0.5, "2026-01-09"),
                         content_hash="sha256:half")
    quarter = _FakeArtifact(key=board_analog_pool_key("STR-THRU", 0.25, "2026-01-09"),
                            content_hash="sha256:quarter")
    picked = analog_declarations([half, quarter], "STR-THRU", 0.25, "2026-01-09",
                                 panel_row, "snap-1", "req-key")
    assert picked["analog_artifact"] is quarter
    assert picked["analog_artifact_recipe"]["content_hash"] == "sha256:quarter"
    assert analog_declarations([half], "STR-THRU", 0.25, "2026-01-09",
                               panel_row, "snap-1", "req-key") == {}


@pytest.mark.parametrize(("bucket", "ratio"), [
    ("<1B", 0.5), ("1-10B", 1.2), (">=10B", 2.5), (None, None)])
def test_complete_query_passes_mcap_bucket_and_ratio_through(bucket, ratio):
    # Distinct bucket/ratio values: overwriting them with a constant fails.
    out = complete_analog_query({"mcap_bucket": bucket, "implied_ratio": ratio},
                                {"dte_entry": 5, "spot": 100.0, "strike": 101.0})
    assert out["mcap_bucket"] == bucket
    assert out["implied_ratio"] == ratio
    assert out["dte_band"] == "4-10" and out["moneyness_band"] == "ATM"


def test_declarations_carry_the_given_snapshot_request_key_and_row():
    key = board_analog_pool_key("STR-THRU", 0.5, "2026-01-09")
    pool = _FakeArtifact(key=key, content_hash="sha256:p")
    small = {"mcap_log": math.log(5e8), "entry_im": 0.09, "mean_prior_or_implied": 0.06}
    large = {"mcap_log": math.log(5e10), "entry_im": 0.04, "mean_prior_or_implied": 0.08}
    first = analog_declarations([pool], "STR-THRU", 0.5, "2026-01-09",
                                small, "snap-A", "key-A")
    second = analog_declarations([pool], "STR-THRU", 0.5, "2026-01-09",
                                 large, "snap-B", "key-B")
    assert first["analog_artifact_recipe"]["seed_snapshot"] == "snap-A"
    assert first["analog_artifact_recipe"]["request_key"] == "key-A"
    assert second["analog_artifact_recipe"]["seed_snapshot"] == "snap-B"
    assert second["analog_artifact_recipe"]["request_key"] == "key-B"
    assert first["analog_query"] == {"mcap_bucket": "<1B", "implied_ratio": pytest.approx(1.5)}
    assert second["analog_query"] == {"mcap_bucket": ">=10B", "implied_ratio": pytest.approx(0.5)}


def _pool_rows():
    # A: dte "4-10" + ATM (40 rows); B: "4-10" + ">5%" (35); C: "26-45" + ATM (31).
    groups = [("4-10", "ATM", 40), ("4-10", ">5%", 35), ("26-45", "ATM", 31)]
    rows = []
    for dte, moneyness, count in groups:
        for _ in range(count):
            i = len(rows)
            rows.append((f"r{i:03d}", "2025-01-01", "2025-02-01", "1-10B", dte,
                         moneyness, "mid", 0.01 * (i % 7 - 3)))
    return rows


def _pool_block():
    artifact = make_board_analog_pool_artifact(
        strategy="STR-THRU", alpha=0.5, cutoff="2026-01-09",
        population_edges=(0.9, 1.1), causal_edges=(0.9, 1.1),
        rows=_pool_rows(), lineage=Lineage())
    declarations = analog_declarations(
        [artifact], "STR-THRU", 0.5, "2026-01-09",
        {"mcap_log": math.log(5e9)}, "snap-1", "req-key")
    block = {
        "analog_artifact": declarations["analog_artifact"],
        "analog_artifact_recipe": declarations["analog_artifact_recipe"],
        "query_features": declarations["analog_query"],
    }
    return block


def _n_analogs(values):
    flags: list[str] = []
    output = _execute_frozen_analogs(_pool_block(), "STR-THRU", values, flags)
    assert flags == []
    return output


@pytest.mark.parametrize(("dte", "strike", "n"), [
    (5, 101.0, 40),     # "4-10" + ATM
    (5, 110.0, 35),     # "4-10" + ">5%"; forcing moneyness to None would give 75
    (30, 101.0, 31),    # "26-45" + ATM; forcing dte to None would give 71
    (30, 110.0, 31),    # nothing matches both: moneyness widens away first
    (None, 101.0, 71),  # no dte: ATM rows of both dte bands
    (5, None, 75),      # no strike: every "4-10" row
    (None, None, 106),  # neither: the whole pool
])
def test_real_frozen_path_end_to_end(dte, strike, n):
    values = {"fill": 0.5, "spot": 100.0}
    if dte is not None:
        values["dte_entry"] = dte
    if strike is not None:
        values["strike"] = strike
    output = _n_analogs(values)
    assert output["n_analogs"] == n
    assert output["ci_low"] is not None and output["ci_low"] <= output["ci_high"]


def test_moneyness_overflow_is_unavailable_not_the_far_bucket():
    # Finite positive operands whose ratio overflows to inf: no band, never ">5%".
    out = complete_analog_query({"mcap_bucket": "1-10B", "implied_ratio": None},
                                {"spot": 1e-308, "strike": 1e308})
    assert out["moneyness_band"] is None


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_two_pools_with_the_same_key_are_ambiguous_in_either_order(order):
    key = board_analog_pool_key("STR-THRU", 0.5, "2026-01-09")
    pools = [_FakeArtifact(key=key, content_hash="sha256:one"),
             _FakeArtifact(key=key, content_hash="sha256:two")]
    with pytest.raises(AmbiguousAnalogPool):
        analog_declarations([pools[i] for i in order], "STR-THRU", 0.5, "2026-01-09",
                            {"mcap_log": math.log(5e9)}, "snap-1", "req-key")