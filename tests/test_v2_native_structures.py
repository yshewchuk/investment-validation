import pytest

from engine.v2.domain.generation import GeometryRefusal, generate, price


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


def test_fresh_cnd_ps_matches_legacy_five_leg_shape():
    geometry = generate("CND-PS", _inputs())

    assert [leg.name for leg in geometry.legs] == ["atm", "up1", "dn1", "up2", "dn2"]
    assert [leg.quantity for leg in geometry.legs] == [0.0, 1.0, 1.0, 1.0, 1.0]
    assert [leg.side for leg in geometry.legs] == ["buy", "sell", "sell", "buy", "buy"]
    assert [leg.strike for leg in geometry.legs] == [100.0, 104.0, 96.0, 108.0, 92.0]
    assert geometry.legs[1].strike + geometry.legs[2].strike == 2 * geometry.spot
    assert geometry.legs[3].strike + geometry.legs[4].strike == 2 * geometry.spot


def test_fresh_cnd_ps_atm_reference_prices_with_zero_cash_flow():
    geometry = generate("CND-PS", _inputs())
    quotes = {
        (leg.right, leg.strike, leg.expiry): {"bid": 1.0, "ask": 3.0}
        for leg in geometry.legs
    }

    priced = price(geometry, quotes, 0.5)

    assert priced.legs[0].name == "atm"
    assert priced.legs[0].fill == 2.0
    assert priced.legs[0].cash_flow == 0.0


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


def test_straddle_selects_common_listed_contract_from_raw_quote_domain():
    quotes = {
        ("C", 95.0, "2026-09-18"): {"bid": 1.0, "ask": 2.0},
        ("P", 95.0, "2026-09-18"): {"bid": 1.0, "ask": 2.0},
        ("C", 105.0, "2026-09-18"): {"bid": 1.0, "ask": 2.0},
        ("P", 105.0, "2026-09-18"): {"bid": 1.0, "ask": 2.0},
        ("C", 100.0, "2026-10-16"): {"bid": 2.0, "ask": 3.0},
        ("P", 100.0, "2026-10-16"): {"bid": 2.0, "ask": 3.0},
    }

    geometry = generate(
        "STR-THRU",
        {
            "spot": 101.0,
            "forecast_abs_move": 8.0,
            "event_date": "2026-09-16",
            "exit_date": "2026-09-17",
            "quotes": quotes,
        },
    )

    # Expiry resolves FIRST (earliest listed expiry on/after exit_date), and
    # only then is the strike nearest spot chosen within that expiry -- even
    # though the later 2026-10-16 expiry lists a strike (100) closer to spot
    # than either strike listed at the earlier, correct expiry.
    assert {leg.strike for leg in geometry.legs} == {105.0}
    assert {leg.expiry for leg in geometry.legs} == {"2026-09-18"}


def test_straddle_honors_requested_expiry_over_a_later_nearer_strike():
    """Reproduces the reported defect: a caller-supplied expiry must win
    even when a LATER expiry lists a strike closer to spot."""
    quotes = {
        ("C", 95.0, "2026-10-16"): {"bid": 1.0, "ask": 2.0},
        ("P", 95.0, "2026-10-16"): {"bid": 1.0, "ask": 2.0},
        ("C", 105.0, "2026-10-16"): {"bid": 1.0, "ask": 2.0},
        ("P", 105.0, "2026-10-16"): {"bid": 1.0, "ask": 2.0},
        ("C", 100.0, "2026-11-20"): {"bid": 2.0, "ask": 3.0},
        ("P", 100.0, "2026-11-20"): {"bid": 2.0, "ask": 3.0},
    }

    geometry = generate(
        "STR-THRU",
        {
            "spot": 100.0,
            "forecast_abs_move": 8.0,
            "expiry": "2026-10-16",
            "quotes": quotes,
        },
    )

    assert {leg.expiry for leg in geometry.legs} == {"2026-10-16"}
    # Within the requested expiry, 105 and 95 are equidistant from spot;
    # the tie-break is the lower strike.
    assert {leg.strike for leg in geometry.legs} == {95.0}


def test_straddle_refuses_when_requested_expiry_is_not_listed():
    quotes = {
        ("C", 100.0, "2026-10-16"): {"bid": 1.0, "ask": 2.0},
        ("P", 100.0, "2026-10-16"): {"bid": 1.0, "ask": 2.0},
    }

    with pytest.raises(GeometryRefusal, match="EXPIRY_NOT_LISTED"):
        generate(
            "STR-THRU",
            {
                "spot": 100.0,
                "forecast_abs_move": 8.0,
                "expiry": "2026-11-20",
                "quotes": quotes,
            },
        )


def test_straddle_amc_excludes_expiry_landing_on_event_date():
    """An AMC print happens after the close, so an expiry ON the event date
    is already dead when the news lands -- it must be excluded."""
    quotes = {
        ("C", 95.0, "2026-09-18"): {"bid": 1.0, "ask": 2.0},
        ("P", 95.0, "2026-09-18"): {"bid": 1.0, "ask": 2.0},
        ("C", 105.0, "2026-09-25"): {"bid": 1.0, "ask": 2.0},
        ("P", 105.0, "2026-09-25"): {"bid": 1.0, "ask": 2.0},
    }
    inputs = {
        "spot": 100.0,
        "forecast_abs_move": 8.0,
        "event_date": "2026-09-18",
        "session": "AMC",
        "quotes": quotes,
    }

    geometry = generate("STR-THRU", inputs)

    assert {leg.expiry for leg in geometry.legs} == {"2026-09-25"}
    assert {leg.strike for leg in geometry.legs} == {105.0}


def test_straddle_bmo_includes_expiry_landing_on_event_date():
    """A BMO print happens before the open, so an expiry ON the event date
    survives it and is the earliest eligible expiry."""
    quotes = {
        ("C", 95.0, "2026-09-18"): {"bid": 1.0, "ask": 2.0},
        ("P", 95.0, "2026-09-18"): {"bid": 1.0, "ask": 2.0},
        ("C", 105.0, "2026-09-25"): {"bid": 1.0, "ask": 2.0},
        ("P", 105.0, "2026-09-25"): {"bid": 1.0, "ask": 2.0},
    }
    inputs = {
        "spot": 100.0,
        "forecast_abs_move": 8.0,
        "event_date": "2026-09-18",
        "session": "BMO",
        "quotes": quotes,
    }

    geometry = generate("STR-THRU", inputs)

    assert {leg.expiry for leg in geometry.legs} == {"2026-09-18"}
    assert {leg.strike for leg in geometry.legs} == {95.0}


def test_straddle_unknown_session_matches_legacy_permissive_fallback():
    """Session unknown falls back to legacy's inclusive >= rule: an expiry
    landing exactly on the event date is NOT excluded."""
    quotes = {
        ("C", 95.0, "2026-09-18"): {"bid": 1.0, "ask": 2.0},
        ("P", 95.0, "2026-09-18"): {"bid": 1.0, "ask": 2.0},
        ("C", 105.0, "2026-09-25"): {"bid": 1.0, "ask": 2.0},
        ("P", 105.0, "2026-09-25"): {"bid": 1.0, "ask": 2.0},
    }
    inputs = {
        "spot": 100.0,
        "forecast_abs_move": 8.0,
        "event_date": "2026-09-18",
        "quotes": quotes,
    }

    geometry = generate("STR-THRU", inputs)

    assert {leg.expiry for leg in geometry.legs} == {"2026-09-18"}


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
