# BAD_QUOTE's percentage threshold is measured against the staged chain
# spot, not the calendar-close geometry spot: legacy's denominator is the
# priced chain spot (engine/score.py:2440, ``result.spot = float(priced.spot)``)
# while native's ``pricing.spot`` stays the calendar close for strike
# geometry. Each fixture row puts the two bases on opposite sides of
# ``BAD_QUOTE_COST_PCT``, so a calendar-close-basis threshold classifies
# differently from the chain-spot basis.

from engine.v2.domain.generation import Geometry, Pricing
from engine.v2.scoring.stages import BAD_QUOTE_COST_PCT, _check_pricing_boundary


def _flags(chain_spot):
    geometry = Geometry("STR-THRU", 100.0, 10.0, (), None)
    pricing = Pricing("STR-THRU", 100.0, 35.0, (), None)
    values = {"entry_cost": 35.0, "chain_spot": chain_spot}
    flags: list[str] = []
    _check_pricing_boundary(values, geometry, pricing, flags)
    return flags


def test_bad_quote_threshold_uses_chain_spot_not_calendar_close():
    # entry_cost 35 is 35% of the calendar close (100) -- over the 30%
    # threshold -- but 28% of the staged chain spot (125): the chain basis
    # keeps the row the calendar-close basis would flag BAD_QUOTE.
    assert 35.0 / 100.0 * 100.0 > BAD_QUOTE_COST_PCT
    assert not (35.0 / 125.0 * 100.0 > BAD_QUOTE_COST_PCT)
    assert "BAD_QUOTE" not in _flags(125.0)


def test_bad_quote_threshold_still_raises_from_the_chain_spot():
    # Same calendar-close geometry, a chain spot that puts the cost over the
    # threshold (35/90 = 38.9%): the chain basis drives the flag too.
    assert "BAD_QUOTE" in _flags(90.0)