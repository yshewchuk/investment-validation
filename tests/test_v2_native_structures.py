import pytest

from engine.v2.domain.generation import generate, price


def _inputs():
    return {"spot": 100.0, "forecast_abs_move": 8.0, "expiry": "2026-10-01"}


def test_inventory_and_disabled_refusals_are_explicit():
    for strategy in ("CAL-P", "CND-P"):
        result = generate(strategy, _inputs())
        assert result.refusal == "UNVALIDATED_STRUCTURE"
        assert result.legs == ()


def test_native_factories_preserve_mirrored_ladders_and_quantity_balance():
    for strategy in ("TWIN-P", "TWIN-P5", "CND-PS", "BFLY-P", "BFLY-P5", "RAMP7", "CTR5"):
        result = generate(strategy, _inputs())
        assert result.legs
        assert sum((leg.quantity if leg.side == "buy" else -leg.quantity)
                   for leg in result.legs) == 0
        strikes = [leg.strike for leg in result.legs]
        assert all((2 * result.spot - strike) in strikes for strike in strikes)


def test_straddle_pricing_is_deterministic_and_fill_aware():
    geometry = generate("STR-THRU", _inputs())
    quotes = {
        ("C", 100.0, "2026-10-01"): {"bid": 2.0, "ask": 4.0},
        ("P", 100.0, "2026-10-01"): {"bid": 3.0, "ask": 5.0},
    }
    mid = price(geometry, quotes, 0.5)
    worst = price(geometry, quotes, 0.0)
    assert mid.entry_cost == 7.0
    assert worst.entry_cost == 9.0
    assert price(geometry, quotes, 0.5) == mid


def test_missing_quote_is_a_truthful_refusal():
    geometry = generate("STR-THRU", _inputs())
    with pytest.raises(ValueError, match="MISSING_QUOTE"):
        price(geometry, {}, 0.5)


def test_resolved_contracts_replace_theoretical_width_with_traded_spacing():
    inputs = {
        **_inputs(),
        "width": 2.0,
        "resolved_legs": (
            {"name": "down1", "right": "P", "side": "buy", "quantity": 1,
             "strike": 95.0},
            {"name": "atm", "right": "P", "side": "sell", "quantity": 2,
             "strike": 100.0},
            {"name": "up1", "right": "P", "side": "buy", "quantity": 1,
             "strike": 105.0},
        ),
    }
    geometry = generate("BFLY-P", inputs)
    assert geometry.width == 5.0
