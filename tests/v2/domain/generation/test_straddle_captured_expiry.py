"""STR-THRU/STR-RUNUP honour a captured post_event_expiry like _expiry() does (issue 115)."""
import pytest

from engine.v2.domain.generation import GeometryRefusal, generate, has_resolvable_expiry

STRADDLES = ("STR-THRU", "STR-RUNUP")
_QUOTE = {"bid": 1.0, "ask": 2.0}
_CHAIN = {
    **{(right, strike, "2026-10-16"): _QUOTE for right in "CP" for strike in (95.0, 105.0)},
    **{(right, 100.0, "2026-11-20"): _QUOTE for right in "CP"},
    **{(right, 100.0, "2026-12-18"): _QUOTE for right in "CP"},
}
# Native selection (no captured expiry) picks 2026-11-20 for both strategies:
# first expiry on/after event_date, and first expiry with DTE >= 30 from entry_date.
_BASE = {
    "spot": 100.0,
    "forecast_abs_move": 8.0,
    "quotes": _CHAIN,
    "event_date": "2026-11-01",
    "entry_date": "2026-10-01",
}


def _expiries(strategy, **captured):
    return {leg.expiry for leg in generate(strategy, {**_BASE, **captured}).legs}


@pytest.mark.parametrize("strategy", STRADDLES)
def test_post_event_expiry_alone_is_honoured(strategy):
    assert _expiries(strategy, post_event_expiry="2026-10-16") == {"2026-10-16"}


@pytest.mark.parametrize("strategy", STRADDLES)
def test_expiry_alone_unchanged(strategy):
    assert _expiries(strategy, expiry="2026-10-16") == {"2026-10-16"}


@pytest.mark.parametrize("strategy", STRADDLES)
def test_expiry_wins_when_both_present(strategy):
    got = _expiries(strategy, expiry="2026-12-18", post_event_expiry="2026-10-16")
    assert got == {"2026-12-18"}


@pytest.mark.parametrize("strategy", STRADDLES)
def test_empty_expiry_falls_back_to_post_event_expiry(strategy):
    assert _expiries(strategy, expiry="", post_event_expiry="2026-10-16") == {"2026-10-16"}


@pytest.mark.parametrize("strategy", STRADDLES)
def test_no_captured_expiry_uses_native_selection(strategy):
    assert _expiries(strategy) == {"2026-11-20"}


@pytest.mark.parametrize("strategy", STRADDLES)
def test_unlisted_post_event_expiry_refuses_instead_of_reselecting(strategy):
    with pytest.raises(GeometryRefusal, match="EXPIRY_NOT_LISTED:2026-10-23"):
        generate(strategy, {**_BASE, "post_event_expiry": "2026-10-23"})


@pytest.mark.parametrize("strategy", STRADDLES)
def test_has_resolvable_expiry_true_for_post_event_expiry_only(strategy):
    assert has_resolvable_expiry(strategy, {**_BASE, "post_event_expiry": "2026-10-16"}, 100.0)
