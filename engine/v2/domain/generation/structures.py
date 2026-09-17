"""Native deterministic structure generation and quote pricing."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

STRATEGIES = (
    "CAL-P", "STR-THRU", "STR-RUNUP", "CND-P", "TWIN-P", "TWIN-P5",
    "CND-PS", "BFLY-P", "BFLY-P5", "RAMP7", "CTR5",
)
DISABLED = {"CAL-P": "UNVALIDATED_STRUCTURE", "CND-P": "UNVALIDATED_STRUCTURE"}


class GeometryRefusal(ValueError):
    """A strategy cannot be generated from the supplied contract domain."""


class PricingRefusal(ValueError):
    """A generated structure cannot be priced from the supplied quotes."""


@dataclass(frozen=True)
class NativeLeg:
    name: str
    right: str
    side: str
    quantity: float
    strike: float
    expiry: str


@dataclass(frozen=True)
class Geometry:
    strategy: str
    spot: float
    width: float
    legs: tuple[NativeLeg, ...]
    refusal: str | None = None


@dataclass(frozen=True)
class PricedLeg:
    name: str
    right: str
    side: str
    quantity: float
    strike: float
    expiry: str
    bid: float
    ask: float
    fill: float
    cash_flow: float


@dataclass(frozen=True)
class Pricing:
    strategy: str
    spot: float
    entry_cost: float
    legs: tuple[PricedLeg, ...]
    refusal: str | None = None


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GeometryRefusal(f"{name} must be finite") from exc
    if not isfinite(result):
        raise GeometryRefusal(f"{name} must be finite")
    return result


def _expiry(inputs: Mapping[str, Any]) -> str:
    expiry = inputs.get("expiry") or inputs.get("post_event_expiry")
    if expiry is None:
        raise GeometryRefusal("MISSING_EXPIRY")
    return str(expiry)


def _resolved_width(legs: tuple[NativeLeg, ...]) -> float:
    """Derive the traded spacing from resolved contracts."""
    by_name = {leg.name: leg.strike for leg in legs}
    if "atm" in by_name and "up1" in by_name:
        return abs(by_name["up1"] - by_name["atm"])
    strikes = sorted({leg.strike for leg in legs})
    spacings = tuple(
        right - left for left, right in zip(strikes, strikes[1:])
        if right > left
    )
    return min(spacings) if spacings else 0.0


def _put_ladder(strategy: str, spot: float, width: float, expiry: str,
                pattern: tuple[tuple[int, float], ...]) -> tuple[NativeLeg, ...]:
    legs: list[NativeLeg] = []
    for index, (multiple, quantity) in enumerate(pattern):
        if multiple == 0:
            legs.append(NativeLeg("atm", "P", "buy" if quantity > 0 else "sell",
                                  abs(quantity), spot, expiry))
            continue
        amount = width * multiple
        for side, strike in (("up", spot + amount), ("down", spot - amount)):
            legs.append(NativeLeg(
                f"{side}{index}", "P", "buy" if quantity > 0 else "sell",
                abs(quantity), strike, expiry,
            ))
    return tuple(legs)


def generate(strategy: str, inputs: Mapping[str, Any]) -> Geometry:
    """Generate one strategy from explicit spot, forecast and expiry inputs."""
    if strategy not in STRATEGIES:
        raise GeometryRefusal("UNKNOWN_STRATEGY")
    if strategy in DISABLED:
        return Geometry(strategy, 0.0, 0.0, (), DISABLED[strategy])
    spot = _number(inputs.get("spot"), "spot")
    forecast = abs(_number(inputs.get("forecast_abs_move", inputs.get("forecast", 0.0)), "forecast"))
    divisor = {"TWIN-P": 1.5, "TWIN-P5": 1.0, "CND-PS": 2.0,
               "BFLY-P": 1.0, "BFLY-P5": 3.0, "RAMP7": 3.0,
               "CTR5": 2.0}.get(strategy, 1.0)
    width = _number(inputs.get("width", forecast / divisor / 100.0 * spot), "width")
    if width <= 0 and strategy not in {"STR-THRU", "STR-RUNUP"}:
        raise GeometryRefusal("ZERO_WIDTH")
    expiry = _expiry(inputs)
    resolved = inputs.get("resolved_legs")
    if resolved:
        legs = tuple(NativeLeg(
            str(leg.get("name", f"leg-{index}")),
            str(leg.get("right", "P")),
            str(leg.get("side", "buy")),
            float(leg.get("quantity", leg.get("qty", 0.0))),
            float(leg["strike"]),
            str(leg.get("expiry", expiry)),
        ) for index, leg in enumerate(resolved))
        return Geometry(strategy, spot, _resolved_width(legs), legs)
    if strategy in {"STR-THRU", "STR-RUNUP"}:
        strike = _number(inputs.get("strike", spot), "strike")
        legs = (NativeLeg("call", "C", "buy", 1.0, strike, expiry),
                NativeLeg("put", "P", "buy", 1.0, strike, expiry))
    else:
        patterns = {
            "CND-PS": ((1, -1.0), (2, 1.0)),
            "TWIN-P": ((0, 2.0), (1, -1.0), (2, -1.0), (4, 1.0)),
            "TWIN-P5": ((0, 2.0), (1, -2.0), (3, 1.0)),
            "BFLY-P": ((0, -2.0), (1, 1.0)),
            "BFLY-P5": ((0, -4.0), (1, 1.0), (3, 1.0)),
            "RAMP7": ((0, -2.0), (1, -1.0), (2, 1.0), (3, 1.0)),
            "CTR5": ((0, -2.0), (1, -1.0), (2, 2.0)),
        }
        legs = _put_ladder(strategy, spot, width, expiry, patterns[strategy])
    return Geometry(strategy, spot, width, legs)


def price(geometry: Geometry, quotes: Mapping[Any, Mapping[str, Any]],
          fill_alpha: float = 0.5) -> Pricing:
    """Price generated legs using the shared worst-to-best fill convention."""
    if geometry.refusal:
        return Pricing(geometry.strategy, geometry.spot, 0.0, (), geometry.refusal)
    alpha = _number(fill_alpha, "fill_alpha")
    if not 0.0 <= alpha <= 1.0:
        raise PricingRefusal("INVALID_FILL_ALPHA")
    priced: list[PricedLeg] = []
    for leg in geometry.legs:
        keys = ((leg.right, leg.strike, leg.expiry),
                (leg.right, str(leg.strike), leg.expiry),
                f"{leg.right}:{leg.strike}:{leg.expiry}")
        quote = next((quotes.get(key) for key in keys if key in quotes), None)
        if quote is None:
            raise PricingRefusal(f"MISSING_QUOTE:{leg.name}")
        bid = _number(quote.get("bid"), f"{leg.name}.bid")
        ask = _number(quote.get("ask"), f"{leg.name}.ask")
        if bid < 0 or ask < bid:
            raise PricingRefusal(f"INVALID_QUOTE:{leg.name}")
        fill = ask - alpha * (ask - bid) if leg.side == "buy" else bid + alpha * (ask - bid)
        cash = (-1.0 if leg.side == "buy" else 1.0) * fill * leg.quantity
        priced.append(PricedLeg(leg.name, leg.right, leg.side, leg.quantity,
                                leg.strike, leg.expiry, bid, ask, fill, cash))
    return Pricing(geometry.strategy, geometry.spot,
                   -sum(item.cash_flow for item in priced), tuple(priced))


__all__ = ["DISABLED", "Geometry", "GeometryRefusal", "NativeLeg", "PricedLeg",
           "Pricing", "PricingRefusal", "STRATEGIES", "generate", "price"]
