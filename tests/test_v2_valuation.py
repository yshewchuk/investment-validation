from engine.v2.domain.valuation import multi_expiry_refusal, planned_exit_label, terminal_payoff


def test_terminal_payoff_and_horizon_are_owned_by_valuation():
    legs = (
        {"kind": "call", "strike": 100, "quantity": 1},
        {"kind": "put", "strike": 90, "quantity": -1},
    )
    assert terminal_payoff(legs, 110) == 10.0
    assert planned_exit_label("2026-09-18", "2026-09-18") == "terminal"
    assert planned_exit_label("2026-09-18", "2026-09-17") == "planned_exit"


def test_multi_expiry_terminal_refusal_is_explicit():
    legs = ({"expiry": "2026-09-18"}, {"expiry": "2026-09-25"})
    assert multi_expiry_refusal(legs) == "MULTI_EXPIRY_TERMINAL_UNSUPPORTED"


def test_terminal_payoff_applies_side_to_positive_quantities():
    legs = (
        {"right": "P", "strike": 95, "quantity": 1, "side": "buy"},
        {"right": "P", "strike": 100, "quantity": 2, "side": "sell"},
        {"right": "P", "strike": 105, "quantity": 1, "side": "buy"},
    )
    assert terminal_payoff(legs, 90) == 0.0
