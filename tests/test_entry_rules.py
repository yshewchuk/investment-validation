"""Arithmetic entry rules, and sizing a structure from a forecast.

Two properties carry most of the weight here, and both are about refusing
rather than deciding.

An entry rule has THREE outcomes. Collapsing "we could not tell" into "no" is
how a missing market cap becomes a silent permanent decline that looks like a
judgement; collapsing it into "yes" is worse.

Sizing has TWO, and the second is "decline". A forecast that cannot size a
structure must produce no structure, not a clipped one — a clipped width trades
a shape the forecast never asked for while the row claims to be forecast-sized.
"""
from __future__ import annotations

import math

import pytest

from engine.entry_rules import (
    MAX_REL_SPREAD,
    MCAP_FLOOR,
    TWIN_P_RULE,
    EntryRule,
    Term,
    rule_for,
)
from engine.forecast_sizing import (
    FORECAST_SIZED,
    PLATEAU_CENTRE,
    WIDTH_MAX,
    WIDTH_MIN,
    forecast_params,
    twin_p_params,
)


def facts(**over):
    """Entry-rule facts, with TWIN-P's geometry as the default.

    `peak` defaults to `2 * w` because that IS TWIN-P's peak; a caller
    overriding `w` alone therefore still describes a coherent seven-strike
    structure. A shape whose peak is not `2w` — TWIN-P5's tight wing — passes
    `peak` explicitly, which is the whole point of the rule reading it.
    """
    base = {"cost": 1.0, "w": 2.0, "rel_spread": 0.10, "mcap_usd": 50e9}
    base.update(over)
    if "peak" not in over:
        width = base.get("w")
        base["peak"] = 2.0 * width if isinstance(width, (int, float)) and width == width else None
    return base


class TestTheThreeOutcomes:
    def test_all_terms_holding_passes(self):
        verdict = TWIN_P_RULE.evaluate(facts())
        assert verdict.passed is True
        assert set(verdict.terms.values()) == {True}

    def test_a_failing_term_says_which(self):
        verdict = TWIN_P_RULE.evaluate(facts(cost=3.0))
        assert verdict.passed is False
        assert verdict.terms["reward"] is False
        assert verdict.terms["spread"] is True
        assert "peak payoff" in verdict.detail

    def test_every_failing_term_is_named_not_just_the_first(self):
        # The rule is reported term by term so each one's cost stays countable.
        verdict = TWIN_P_RULE.evaluate(facts(cost=3.0, rel_spread=0.9, mcap_usd=1e9))
        assert verdict.passed is False
        assert verdict.detail.count(";") == 2

    @pytest.mark.parametrize("absent", ["cost", "peak", "rel_spread", "mcap_usd"])
    def test_a_missing_fact_is_undetermined_not_a_rejection(self, absent):
        verdict = TWIN_P_RULE.evaluate(facts(**{absent: None}))
        assert verdict.passed is None
        assert absent in verdict.missing

    def test_a_nan_is_missing_not_a_number(self):
        # A NaN that compares False against every threshold would silently
        # reject, and the row would show a decision nobody made.
        verdict = TWIN_P_RULE.evaluate(facts(mcap_usd=float("nan")))
        assert verdict.passed is None
        assert "mcap_usd" in verdict.missing

    def test_an_infinite_fact_is_missing_too(self):
        verdict = TWIN_P_RULE.evaluate(facts(rel_spread=float("inf")))
        assert verdict.passed is None


class TestTheTwinPTerms:
    def test_reward_is_strict(self):
        assert TWIN_P_RULE.evaluate(facts(cost=2.0, w=2.0)).passed is False
        assert TWIN_P_RULE.evaluate(facts(cost=1.999, w=2.0)).passed is True

    def test_the_reward_term_reads_the_peak_not_the_width(self):
        """The same cost and spacing, two shapes, two answers.

        TWIN-P5's tight wing peaks at `a` where TWIN-P peaks at `2a`, so at
        `cost = 1.5` against `a = 2` the seven-strike shape still has reward
        beating risk and the tight five-strike shape does not. A rule written
        as `cost < w` cannot tell them apart and would admit the second at
        twice the risk it priced.
        """
        assert TWIN_P_RULE.evaluate(facts(cost=1.5, w=2.0, peak=4.0)).passed is True
        assert TWIN_P_RULE.evaluate(facts(cost=1.5, w=2.0, peak=2.0)).passed is False

    def test_a_zero_width_is_undetermined_not_a_free_pass(self):
        # cost < 0 is impossible, so `cost < w` with w = 0 would always reject —
        # but a zero width means the legs collapsed, which is a broken structure
        # and not a judgement about the trade.
        assert TWIN_P_RULE.evaluate(facts(w=0.0)).passed is None
        assert TWIN_P_RULE.evaluate(facts(peak=0.0)).passed is None

    def test_the_spread_threshold_is_inclusive(self):
        assert TWIN_P_RULE.evaluate(facts(rel_spread=MAX_REL_SPREAD)).passed is True
        assert TWIN_P_RULE.evaluate(facts(rel_spread=MAX_REL_SPREAD + 1e-9)).passed is False

    def test_the_mcap_floor_is_inclusive(self):
        assert TWIN_P_RULE.evaluate(facts(mcap_usd=MCAP_FLOOR)).passed is True
        assert TWIN_P_RULE.evaluate(facts(mcap_usd=MCAP_FLOOR - 1)).passed is False

    def test_the_thresholds_match_the_preregistered_experiment(self):
        # EXP-123's spec.yaml registered these before it ran. A later edit here
        # would silently change what "the registered rule" means.
        assert MAX_REL_SPREAD == 0.25
        assert MCAP_FLOOR == 10e9


class TestTheRegistry:
    def test_twin_p_has_a_rule_and_the_gated_strategies_do_not(self):
        assert rule_for("TWIN-P") is TWIN_P_RULE
        assert rule_for("STR-THRU") is None
        assert rule_for("nonsense") is None

    def test_a_rule_reports_the_evidence_that_registered_it(self):
        assert "EXP-123" in TWIN_P_RULE.evidence


class TestSizing:
    def test_the_forecast_lands_at_the_plateau_centre(self):
        # The payoff is flat at its maximum between w and 2w, so a forecast of
        # 7.5% should put 7.5% at 1.5w — not at a wing, not at the peak.
        params = twin_p_params(7.5)
        width = params["width_moneyness"]
        assert width * PLATEAU_CENTRE == pytest.approx(0.075)

    @pytest.mark.parametrize("bad", [None, float("nan"), 0.0, -3.0, "x"])
    def test_a_forecast_that_is_not_one_sizes_nothing(self, bad):
        assert twin_p_params(bad) is None

    def test_a_forecast_outside_the_bounds_is_refused_not_clipped(self):
        too_small = WIDTH_MIN * PLATEAU_CENTRE * 100 * 0.5
        too_large = WIDTH_MAX * PLATEAU_CENTRE * 100 * 2
        assert twin_p_params(too_small) is None
        assert twin_p_params(too_large) is None

    def test_the_bounds_themselves_are_inside(self):
        assert twin_p_params(WIDTH_MIN * PLATEAU_CENTRE * 100) is not None
        assert twin_p_params(WIDTH_MAX * PLATEAU_CENTRE * 100) is not None

    def test_only_declared_strategies_are_forecast_sized(self):
        assert set(FORECAST_SIZED) == {"TWIN-P", "TWIN-P5"}
        assert forecast_params("STR-THRU", 7.5) is None

    def test_the_parameter_is_one_the_factory_accepts(self):
        # A parameter the factory rejects would raise at score time, on the
        # board, for every row.
        from engine.structures import STRUCTURES

        STRUCTURES["TWIN-P"](**twin_p_params(7.5))


class TestARuleIsGeneral:
    """The mechanism, not just TWIN-P's instance of it."""

    def test_a_rule_with_no_terms_passes_vacuously(self):
        rule = EntryRule(strategy="X", terms=(), evidence="none")
        assert rule.evaluate({}).passed is True

    def test_undetermined_beats_failed_when_both_are_present(self):
        # A row that both fails one term and cannot evaluate another is
        # UNDETERMINED: the failing term might have been the only one, and
        # reporting a decision would overstate what is known.
        rule = EntryRule(
            strategy="X",
            terms=(
                Term("always_false", "no", lambda _f: False),
                Term("unknown", "?", lambda _f: None, needs=("absent",)),
            ),
            evidence="test",
        )
        assert rule.evaluate({}).passed is None
