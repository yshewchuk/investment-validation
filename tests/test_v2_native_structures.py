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


# --------------------------------------------------------------------------
# Listed-strike generation for the symmetric put-ladder families.
#
# The defect: generate() built the ATM (and every other) leg from the raw
# continuous spot float for CND-PS/TWIN-P/TWIN-P5/BFLY-P/BFLY-P5/RAMP7/CTR5,
# so price()'s exact-key quote lookup could never match a real chain. The
# fix must reproduce legacy's own StrikeSelector rule (engine/structures.py)
# rather than invent a new one: `bracket(side="below")` for the anchor,
# `offset_from` to size the spacing off the grid, `mirror` (exact-listed-or-
# refuse) for every strike placed relative to another. Below: fixture-driven
# regressions for two different assembly shapes, a coarse-grid case that
# proves TWIN-P's chained-mirror construction is not interchangeable with
# the other families' independent-offset construction, and the two distinct
# refusal codes legacy has (NO_CHAIN-analogue vs COARSE_LADDER-analogue).


def test_cnd_ps_snaps_atm_to_the_listed_strike_from_fixture_000():
    """Fixture 000_CND-PS-LEN-2026-09-16 (corpus 20260922T031338Z.tmp):
    legacy priced this event's ATM reference leg at the listed strike 77.0
    for the 2026-09-18 expiry. spot/width are the values legacy's own
    record.spot / record.structure_width captured for this row; quote
    bid/ask below are synthetic, not the fixture's real market data.

    FAILS on origin/main: unpatched generate() builds the atm leg at raw
    spot (77.91), which is not a quotes key, so price() raises
    MISSING_QUOTE:atm (verified separately against origin/main's copy of
    this module -- see the PR/handback notes)."""
    grid = (73.0, 75.0, 77.0, 79.0, 81.0)
    expiry = "2026-09-18"
    quotes = {("P", strike, expiry): {"bid": 1.0, "ask": 2.0} for strike in grid}
    inputs = {"spot": 77.91, "width": 2.0, "expiry": expiry, "quotes": quotes}

    geometry = generate("CND-PS", inputs)

    assert geometry.legs[0].name == "atm"
    assert geometry.legs[0].strike == 77.0
    assert ("P", 77.0, expiry) in quotes

    priced = price(geometry, quotes, 0.5)
    assert priced.refusal is None
    assert [leg.strike for leg in priced.legs] == [77.0, 79.0, 75.0, 81.0, 73.0]


def test_bfly_p_snaps_to_the_listed_strikes_from_fixture_003():
    """Fixture 003_BFLY-P-RLGT-2026-09-14: legacy's atm/up1/dn1 were the
    listed strikes 7.5/10.0/5.0 on a sparse ($2.50-spaced) grid -- the same
    independent-offset-then-mirror shape as CND-PS's family, exercised on a
    different structure so the fix isn't a single-family patch."""
    grid = (5.0, 7.5, 10.0)
    expiry = "2026-09-18"
    quotes = {("P", strike, expiry): {"bid": 1.0, "ask": 2.0} for strike in grid}
    inputs = {"spot": 8.21, "width": 2.5, "expiry": expiry, "quotes": quotes}

    geometry = generate("BFLY-P", inputs)

    assert [(leg.name, leg.strike) for leg in geometry.legs] == [
        ("atm", 7.5), ("up1", 10.0), ("dn1", 5.0),
    ]
    priced = price(geometry, quotes, 0.5)
    assert priced.refusal is None


def test_twin_p_chained_mirror_matches_legacy_on_a_coarse_grid_that_would_diverge():
    """Proves TWIN-P's construction is NOT the independently-offset ladder
    the other five families use. On this grid, offsetting each wing
    independently by width*multiple (the wrong, easier-to-write rule) would
    put up2 at 109 (nearest listed to spot+2*width=110); legacy's actual
    rule -- offset_from finds up1 ONLY, everything else is an exact chained
    mirror off already-resolved strikes -- puts up2 at 112
    (mirror(atm=100, about=up1=106) = 2*106-100). A dense grid cannot tell
    these apart because both land on the same nearby strike; this one can
    and does."""
    grid = (76.0, 88.0, 94.0, 100.0, 106.0, 109.0, 112.0, 124.0)
    expiry = "2026-10-01"
    quotes = {("P", strike, expiry): {"bid": 1.0, "ask": 2.0} for strike in grid}
    inputs = {"spot": 101.0, "width": 5.0, "expiry": expiry, "quotes": quotes}

    geometry = generate("TWIN-P", inputs)
    by_name = {leg.name: leg.strike for leg in geometry.legs}

    # The independent-offset value this grid was built to distinguish from:
    naive_up2 = min((s for s in grid if s > 100.0), key=lambda s: abs(s - 110.0))
    assert naive_up2 == 109.0
    assert by_name["up2"] == 112.0
    assert by_name["up2"] != naive_up2

    assert by_name == {
        "atm": 100.0, "up1": 106.0, "dn1": 94.0,
        "up2": 112.0, "dn2": 88.0, "up3": 124.0, "dn3": 76.0,
    }
    priced = price(geometry, quotes, 0.5)
    assert priced.refusal is None


def test_condor_wings_are_exactly_evenly_spaced_in_dollars():
    """The defined-risk claim depends on (K4-K3) == (K2-K1) exactly -- an
    uneven condor pays a negative amount below the bottom strike. Every
    mirrored leg must therefore land at EXACTLY the anchor-symmetric dollar
    distance, not merely close to it."""
    grid = (73.0, 75.0, 77.0, 79.0, 81.0)
    expiry = "2026-09-18"
    quotes = {("P", strike, expiry): {"bid": 1.0, "ask": 2.0} for strike in grid}
    geometry = generate(
        "CND-PS", {"spot": 77.91, "width": 2.0, "expiry": expiry, "quotes": quotes},
    )
    by_name = {leg.name: leg.strike for leg in geometry.legs}
    assert by_name["up1"] - by_name["atm"] == by_name["atm"] - by_name["dn1"]
    assert by_name["up2"] - by_name["atm"] == by_name["atm"] - by_name["dn2"]


def test_ladder_refuses_rather_than_approximating_when_no_listed_strike_completes_the_mirror():
    """A mirror target that is not listed means the grid cannot carry the
    even-spacing shape at all -- legacy refuses (StructureError -> NO_CHAIN)
    rather than substituting the nearest strike, and so must native. This is
    a per-leg failure, distinct from the collision case below, and must
    raise a distinguishable code (not the collision code)."""
    grid = (100.0, 105.0)  # 95.0 (2*100-105) is deliberately absent
    expiry = "2026-01-01"
    quotes = {("P", strike, expiry): {"bid": 1.0, "ask": 2.0} for strike in grid}

    with pytest.raises(GeometryRefusal, match=r"^NO_LISTED_STRIKE:dn1$"):
        generate("BFLY-P", {"spot": 100.5, "width": 5.0, "expiry": expiry, "quotes": quotes})


def test_ladder_refuses_with_coarse_ladder_code_when_two_legs_collide_on_one_contract():
    """A ladder too coarse for the shape can independently snap two DIFFERENT
    legs onto the SAME listed contract (CND-PS's up1/up2 both nearest to the
    one strike a sparse grid lists above the anchor) -- legacy's
    LadderTooCoarse / COARSE_LADDER, a different failure from a single leg
    simply having no listed strike. The two codes must not collapse into
    one: that is the diagnosability gap the coordinator flagged."""
    grid = (95.0, 100.0, 105.0)  # only one strike listed above/below the anchor
    expiry = "2026-01-01"
    quotes = {("P", strike, expiry): {"bid": 1.0, "ask": 2.0} for strike in grid}

    with pytest.raises(GeometryRefusal, match=r"^COARSE_LADDER:"):
        generate("CND-PS", {"spot": 100.5, "width": 1.0, "expiry": expiry, "quotes": quotes})


def test_str_thru_and_str_runup_are_unaffected_by_the_ladder_grid_fix():
    """STR-THRU/STR-RUNUP already snapped correctly via _select_listed_
    straddle before this fix and must be byte-for-byte unchanged by it --
    the fix only touches the CND-PS/TWIN-P/TWIN-P5/BFLY-P/BFLY-P5/RAMP7/CTR5
    branches. Reuses the same deliberately coarse grid the TWIN-P test above
    proves is discriminating, to show straddle selection (nearest listed
    strike either direction, shared call/put strike) is untouched by the
    ladder-specific bracket/offset_from/mirror machinery."""
    grid = (76.0, 88.0, 94.0, 100.0, 106.0, 109.0, 112.0, 124.0)
    expiry = "2026-10-01"
    quotes = {}
    for strike in grid:
        quotes[("C", strike, expiry)] = {"bid": 1.0, "ask": 2.0}
        quotes[("P", strike, expiry)] = {"bid": 1.0, "ask": 2.0}

    geometry = generate("STR-THRU", {"spot": 101.0, "expiry": expiry, "quotes": quotes})

    # Nearest listed strike to spot=101.0 in EITHER direction is 100.0 (not
    # the ladder anchor's bracket-below rule, and not 106.0/109.0).
    assert {leg.strike for leg in geometry.legs} == {100.0}
    priced = price(geometry, quotes, 0.5)
    assert priced.refusal is None


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
