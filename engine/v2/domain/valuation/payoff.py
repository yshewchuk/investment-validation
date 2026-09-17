"""Pure payoff and horizon semantics owned by the valuation layer."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

__all__ = ["multi_expiry_refusal", "planned_exit_label", "terminal_payoff"]


def planned_exit_label(expiry: str | None, exit_date: str | None) -> str:
    if expiry is None or exit_date is None:
        return "planned_exit"
    return "terminal" if str(expiry) == str(exit_date) else "planned_exit"


def multi_expiry_refusal(legs: Iterable[Mapping[str, Any]]) -> str | None:
    expiries = {str(leg.get("expiry")) for leg in legs if leg.get("expiry") is not None}
    return "MULTI_EXPIRY_TERMINAL_UNSUPPORTED" if len(expiries) > 1 else None


def terminal_payoff(legs: Iterable[Mapping[str, Any]], spot: float) -> float:
    """Return the signed intrinsic payoff for a frozen set of option legs."""
    total = 0.0
    for leg in legs:
        quantity = float(leg.get("quantity", leg.get("qty", 0.0)))
        side = leg.get("side")
        if side is not None:
            normalized_side = str(side).lower()
            if normalized_side in {"buy", "long"}:
                quantity = abs(quantity)
            elif normalized_side in {"sell", "short"}:
                quantity = -abs(quantity)
            else:
                raise ValueError(f"unsupported option side: {side}")
        strike = float(leg["strike"])
        kind = str(leg.get("kind", leg.get("right", "call"))).lower()
        intrinsic = max(float(spot) - strike, 0.0) if kind in {"call", "c"} else max(strike - float(spot), 0.0)
        total += quantity * intrinsic
    return total
