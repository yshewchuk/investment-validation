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

    def __init__(self, code: str, detail: str | None = None):
        super().__init__(code)
        self.code = code
        self.detail = detail


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
    detail: str | None = None


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


def _finite_float(value: Any, name: str) -> float:
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


def _quote_contracts(inputs: Mapping[str, Any]) -> tuple[tuple[str, float, str], ...]:
    """Return the raw contract keys available to a source-driven selector."""
    quotes = inputs.get("quotes")
    if not isinstance(quotes, Mapping):
        return ()
    contracts = []
    for key in quotes:
        if isinstance(key, tuple) and len(key) == 3:
            right, strike, expiry = key
        elif isinstance(key, str):
            parts = key.split(":")
            if len(parts) != 3:
                continue
            right, strike, expiry = parts
        else:
            continue
        try:
            right = str(right).upper()
            strike = float(strike)
            expiry = str(expiry)
        except (TypeError, ValueError):
            continue
        if right in {"C", "P"} and isfinite(strike) and expiry:
            contracts.append((right, strike, expiry))
    return tuple(sorted(set(contracts), key=lambda row: (row[2], row[1], row[0])))


def _resolve_straddle_expiry(inputs: Mapping[str, Any], expiries: list[str]) -> str:
    """Resolve the expiry to trade, mirroring legacy ``ExpirySelector``.

    Despite the name (kept to avoid touching call sites unnecessarily), this
    takes no straddle-specific input -- like legacy's own
    ``ExpirySelector.select`` (``engine/structures.py``), which chooses from
    the full chain's expiries regardless of the leg's ``right``. It is
    shared by the put-ladder expiry selection below (:func:`generate`'s
    non-straddle branch), which supplies put-only ``expiries`` from
    :func:`_listed_put_expiries` -- every enabled put-ladder strategy
    (TWIN-P, TWIN-P5, CND-PS, BFLY-P, BFLY-P5, RAMP7, CTR5) uses legacy's
    ``first_post_event`` kind, the same one this implements.

    - A caller-supplied ``expiry`` is legacy's ``fixed`` rule: it must match
      a listed expiry exactly (by calendar date), or this refuses via
      ``GeometryRefusal`` rather than silently substituting another expiry.
    - Otherwise, when an ``event_date`` is known, this mirrors
      legacy's ``first_post_event``: the earliest listed expiry on/after that
      date, with the AMC/BMO distinction applied when ``session`` is known
      (AMC excludes an expiry landing exactly on the event date, since it
      dies at the close before an after-close announcement). Session unknown
      falls back to legacy's permissive ``>=`` rule. No survivor is also a
      refusal, not a silent substitution.
    - With no date signal at all, the earliest listed expiry is used as a
      deterministic default.
    """
    requested = inputs.get("expiry")
    if requested is not None:
        target = str(requested)[:10]
        matches = [candidate for candidate in expiries if candidate[:10] == target]
        if not matches:
            raise GeometryRefusal(f"EXPIRY_NOT_LISTED:{target}")
        return matches[0]

    target_source = inputs.get("event_date")
    if target_source is None:
        return expiries[0]

    target = str(target_source)[:10]
    session = str(inputs.get("session") or "").upper()
    if session == "AMC":
        survivors = [candidate for candidate in expiries if candidate[:10] > target]
    else:
        survivors = [candidate for candidate in expiries if candidate[:10] >= target]
    if not survivors:
        raise GeometryRefusal(f"NO_EXPIRY_ON_OR_AFTER:{target}")
    return survivors[0]


def _select_listed_straddle(inputs: Mapping[str, Any], spot: float) -> tuple[float, str] | None:
    """Select a common listed strike and expiry from raw quote keys.

    This is deliberately a geometry operation. It sees the available contract
    domain and request dates, while pricing remains responsible for validating
    and consuming the bid/ask values.

    Mirrors legacy ``ExpirySelector`` (``engine/structures.py``): the EXPIRY
    is resolved first (:func:`_resolve_straddle_expiry`), and only then is a
    strike chosen within it. Choosing by strike distance first (the previous
    behaviour here) could return a strike from a LATER expiry than the one
    actually requested, whenever that later expiry happened to list a strike
    closer to spot.
    """
    contracts = _quote_contracts(inputs)
    common = {
        (strike, expiry)
        for right, strike, expiry in contracts
        if right == "C"
    } & {
        (strike, expiry)
        for right, strike, expiry in contracts
        if right == "P"
    }
    if not common:
        return None
    expiries = sorted({expiry for _, expiry in common})
    resolved_expiry = _resolve_straddle_expiry(inputs, expiries)
    pool = [(strike, expiry) for strike, expiry in common if expiry == resolved_expiry]
    strike, expiry = min(pool, key=lambda row: (abs(row[0] - spot), row[0]))
    return strike, expiry


def _listed_put_expiries(inputs: Mapping[str, Any]) -> list[str]:
    """Sorted distinct expiries with at least one listed put.

    The put-ladder counterpart to :func:`_select_listed_straddle`'s expiry
    candidates. Unlike that function this does not require a matching call
    at the same strike: every enabled put-ladder strategy (TWIN-P, TWIN-P5,
    CND-PS, BFLY-P, BFLY-P5, RAMP7, CTR5) trades puts alone, and legacy's own
    ``ExpirySelector.select`` (``engine/structures.py``) never filters by
    right either -- the per-leg *strike* selection that follows is what
    actually requires the right to be listed at the chosen expiry.
    """
    contracts = _quote_contracts(inputs)
    return sorted({expiry for right, _, expiry in contracts if right == "P"})


def has_resolvable_expiry(strategy: str, inputs: Mapping[str, Any], spot: float) -> bool:
    """Whether :func:`generate` can resolve an expiry for ``strategy``.

    True when ``inputs`` carries an explicit ``expiry``/``post_event_expiry``
    field, or -- when neither is captured -- a native selection off listed
    ``quotes`` finds a candidate (mirrors legacy ``ExpirySelector``'s
    ``first_post_event``/``fixed`` kinds via :func:`_resolve_straddle_expiry`,
    the only kinds any enabled strategy here uses).

    Used by the geometry gate (``engine/v2/scoring/stages.py``
    ``_resolve_geometry``) to decide whether ``MISSING_EXPIRY`` is a genuine
    refusal or only a captured-field gap that a chain already present in
    ``quotes`` can still resolve -- a row legacy failed to price (``NO_CHAIN``,
    ``COARSE_LADDER``, ``NO_FORECAST`` at sizing) never gets ``context
    ["expiry"]`` written (``engine/score.py`` ~2349-2358), even though the
    same row still carries ``quotes``, ``event_date``, ``exit_date`` and
    ``session``.

    This is deliberately an EXISTENCE check (is there any contract domain to
    try at all), not a full resolution: whether the date filter (e.g. no
    listed expiry survives an AMC print) then leaves a survivor is
    :func:`generate`'s job, and its refusal (``NO_EXPIRY_ON_OR_AFTER``,
    ``EXPIRY_NOT_LISTED``, ``COARSE_LADDER``, ``MISSING_CONTRACTS``) is more
    specific than a blanket ``MISSING_EXPIRY`` and is reported as such rather
    than collapsed into it -- the gate only short-circuits the case where
    there is nothing listed for native to even attempt.
    """
    if inputs.get("expiry") is not None or inputs.get("post_event_expiry") is not None:
        return True
    if strategy in {"STR-THRU", "STR-RUNUP"}:
        contracts = _quote_contracts(inputs)
        calls = {(strike, expiry) for right, strike, expiry in contracts if right == "C"}
        puts = {(strike, expiry) for right, strike, expiry in contracts if right == "P"}
        return bool(calls & puts)
    return bool(_listed_put_expiries(inputs))


def _grid_for_right(inputs: Mapping[str, Any], right: str, expiry: str) -> tuple[float, ...]:
    """Sorted distinct listed strikes for ``right`` at ``expiry``.

    Empty when the caller supplied no chain domain at all (``quotes`` absent,
    or nothing listed at this right/expiry) -- the degenerate case unit tests
    exercise without a chain, where generation falls back to the pre-existing
    continuous-spot arithmetic instead of refusing. Once a real chain is
    present, callers must use it: see :func:`_resolve_ladder_on_grid` and
    friends below.
    """
    contracts = _quote_contracts(inputs)
    return tuple(sorted({strike for r, strike, e in contracts
                          if r == right and e == expiry}))


def _bracket_below(grid: tuple[float, ...], spot: float) -> float | None:
    """Greatest listed strike at or below ``spot``.

    Mirrors legacy's ``StrikeSelector("bracket", side="below")``
    (``engine/structures.py`` ~308) -- the anchor every symmetric put ladder
    is built from. ``grid`` is sorted ascending, so the last surviving entry
    is the one nearest spot from below.
    """
    candidates = [strike for strike in grid if strike <= spot]
    return candidates[-1] if candidates else None


def _ladder_offset_from(grid: tuple[float, ...], anchor: float,
                        delta: float) -> float | None:
    """Nearest listed strike to ``anchor + delta``, strictly on ``delta``'s side.

    Mirrors legacy's ``StrikeSelector("offset_from", ...)`` (~348): candidates
    are restricted to strikes strictly above the anchor (``delta > 0``) or
    strictly below it (``delta < 0``), which excludes the anchor's own strike
    and keeps a lopsided grid from returning a nearer strike on the wrong
    side of it.
    """
    if delta == 0:
        return None
    candidates = [strike for strike in grid
                  if (strike > anchor if delta > 0 else strike < anchor)]
    if not candidates:
        return None
    target = anchor + delta
    return min(candidates, key=lambda strike: (abs(strike - target), strike))


def _ladder_mirror(grid: tuple[float, ...], ref: float, about: float,
                   tol: float = 1e-6) -> float | None:
    """``2*about - ref`` if and only if that strike is LISTED.

    Mirrors legacy's ``StrikeSelector("mirror", ...)`` (~371). Unlike
    ``offset_from`` this never snaps to the nearest strike: an unlisted
    mirror target means the ladder cannot carry the shape at exactly even
    spacing, so the caller must refuse rather than approximate -- an unevenly
    spaced condor pays ``(K4-K3) - (K2-K1) < 0`` below the bottom strike, and
    its loss is no longer capped at the debit.
    """
    target = 2.0 * about - ref
    for strike in grid:
        if abs(strike - target) <= tol:
            return strike
    return None


def _check_ladder_collisions(legs: tuple[NativeLeg, ...]) -> None:
    """Refuse when two DISTINCT legs resolved onto the same contract.

    Mirrors legacy's post-resolution collision check (``engine/structures.py``
    ~769-784, ``LadderTooCoarse``, flagged ``COARSE_LADDER`` by
    ``engine/score.py``): independently-snapped legs -- CND-PS's up1/up2 in
    particular -- can both land on the first listed strike above a coarse
    anchor even though each one resolved fine on its own. That is a distinct
    failure mode from any single leg lacking a listed strike (``NO_CHAIN`` in
    legacy, ``NO_LISTED_STRIKE`` here), so it gets its own code rather than
    reusing that one -- collapsing the two costs a diagnosability cycle.
    """
    seen: dict[tuple[str, float, str], list[str]] = {}
    for leg in legs:
        seen.setdefault((leg.right, leg.strike, leg.expiry), []).append(leg.name)
    collided = sorted(
        (key, names) for key, names in seen.items() if len(names) > 1
    )
    if collided:
        detail = "+".join(name for _, names in collided for name in names)
        raise GeometryRefusal("COARSE_LADDER", detail=detail)


#: family -> (anchor_qty, tail) for the shared independent-offset ladder
#: shape. Mirrors legacy's ``_symmetric_put_ladder`` factory
#: (``engine/structures.py`` ~1238): each tail multiple is snapped to the
#: grid INDEPENDENTLY via ``offset_from``, then exact-mirrored for its dn
#: twin. This is the shape CND-PS, BFLY-P, BFLY-P5, RAMP7 and CTR5 share --
#: TWIN-P/TWIN-P5 do NOT use it, see :func:`_resolve_twin_peak_on_grid`.
_LADDER_SPECS: dict[str, tuple[float, tuple[tuple[int, float], ...]]] = {
    "CND-PS": (0.0, ((1, -1.0), (2, 1.0))),
    "BFLY-P": (-2.0, ((1, 1.0),)),
    "BFLY-P5": (-4.0, ((1, 1.0), (3, 1.0))),
    "RAMP7": (-2.0, ((1, -1.0), (2, 1.0), (3, 1.0))),
    "CTR5": (-2.0, ((1, -1.0), (2, 2.0))),
}


def _resolve_ladder_on_grid(strategy: str, spot: float, width: float, expiry: str,
                            grid: tuple[float, ...]) -> tuple[NativeLeg, ...]:
    """CND-PS/BFLY-P/BFLY-P5/RAMP7/CTR5's shape, resolved on the listed grid."""
    anchor_qty, tail = _LADDER_SPECS[strategy]
    atm_strike = _bracket_below(grid, spot)
    if atm_strike is None:
        raise GeometryRefusal("NO_CHAIN", detail="NO_LISTED_STRIKE:atm")
    legs: list[NativeLeg] = [NativeLeg(
        "atm", "P", "buy" if anchor_qty >= 0 else "sell",
        abs(anchor_qty), atm_strike, expiry,
    )]
    for i, (mult, qty) in enumerate(tail, start=1):
        up = _ladder_offset_from(grid, atm_strike, width * mult)
        if up is None:
            raise GeometryRefusal("NO_CHAIN", detail=f"NO_LISTED_STRIKE:up{i}")
        dn = _ladder_mirror(grid, up, atm_strike)
        if dn is None:
            raise GeometryRefusal("NO_CHAIN", detail=f"NO_LISTED_STRIKE:dn{i}")
        side = "buy" if qty > 0 else "sell"
        legs.append(NativeLeg(f"up{i}", "P", side, abs(qty), up, expiry))
        legs.append(NativeLeg(f"dn{i}", "P", side, abs(qty), dn, expiry))
    resolved = tuple(legs)
    _check_ladder_collisions(resolved)
    return resolved


def _resolve_twin_peak_on_grid(spot: float, width: float, expiry: str,
                               grid: tuple[float, ...]) -> tuple[NativeLeg, ...]:
    """TWIN-P's seven-strike chained-mirror shape, on the listed grid.

    Mirrors legacy's ``twin_peak`` (``engine/structures.py`` ~1024) exactly:
    ``offset_from`` picks ONLY up1; every other strike is a chained exact
    ``mirror`` off already-resolved strikes (``up2 = mirror(atm, up1)``,
    ``up4 = mirror(atm, up2)``, ``dn{i} = mirror(up{i}, atm)``). This is NOT
    the independent-offset ladder shape above -- multiplying width by the
    multiple independently would not reproduce this chain on a coarse grid,
    only on a dense one where the two happen to agree.
    """
    atm = _bracket_below(grid, spot)
    if atm is None:
        raise GeometryRefusal("NO_CHAIN", detail="NO_LISTED_STRIKE:atm")
    up1 = _ladder_offset_from(grid, atm, width)
    if up1 is None:
        raise GeometryRefusal("NO_CHAIN", detail="NO_LISTED_STRIKE:up1")
    dn1 = _ladder_mirror(grid, up1, atm)
    if dn1 is None:
        raise GeometryRefusal("NO_CHAIN", detail="NO_LISTED_STRIKE:dn1")
    up2 = _ladder_mirror(grid, atm, up1)
    if up2 is None:
        raise GeometryRefusal("NO_CHAIN", detail="NO_LISTED_STRIKE:up2")
    dn2 = _ladder_mirror(grid, up2, atm)
    if dn2 is None:
        raise GeometryRefusal("NO_CHAIN", detail="NO_LISTED_STRIKE:dn2")
    up4_local = _ladder_mirror(grid, atm, up2)
    if up4_local is None:
        raise GeometryRefusal("NO_CHAIN", detail="NO_LISTED_STRIKE:up4")
    dn4 = _ladder_mirror(grid, up4_local, atm)
    if dn4 is None:
        raise GeometryRefusal("NO_CHAIN", detail="NO_LISTED_STRIKE:dn4")
    legs = (
        NativeLeg("atm", "P", "buy", 2.0, atm, expiry),
        NativeLeg("up1", "P", "sell", 1.0, up1, expiry),
        NativeLeg("up2", "P", "sell", 1.0, up2, expiry),
        NativeLeg("up4", "P", "buy", 1.0, up4_local, expiry),
        NativeLeg("dn1", "P", "sell", 1.0, dn1, expiry),
        NativeLeg("dn2", "P", "sell", 1.0, dn2, expiry),
        NativeLeg("dn4", "P", "buy", 1.0, dn4, expiry),
    )
    _check_ladder_collisions(legs)
    return legs


def _resolve_twin_peak_5_on_grid(spot: float, width: float, expiry: str,
                                 grid: tuple[float, ...]) -> tuple[NativeLeg, ...]:
    """TWIN-P5's five-strike chained-mirror shape (``wing_multiple=3``), on
    the listed grid. Mirrors legacy's ``twin_peak_5`` (~1133) default case:
    ``up1`` is the only ``offset_from`` selection, ``dn1`` mirrors it about
    ``atm``, and the wings are chained mirrors of ``up1``/``dn1`` about each
    other -- again not reproducible by independently offsetting each leg.
    """
    atm = _bracket_below(grid, spot)
    if atm is None:
        raise GeometryRefusal("NO_CHAIN", detail="NO_LISTED_STRIKE:atm")
    up1 = _ladder_offset_from(grid, atm, width)
    if up1 is None:
        raise GeometryRefusal("NO_CHAIN", detail="NO_LISTED_STRIKE:up1")
    dn1 = _ladder_mirror(grid, up1, atm)
    if dn1 is None:
        raise GeometryRefusal("NO_CHAIN", detail="NO_LISTED_STRIKE:dn1")
    up_wing = _ladder_mirror(grid, dn1, up1)
    if up_wing is None:
        raise GeometryRefusal("NO_CHAIN", detail="NO_LISTED_STRIKE:up_wing")
    dn_wing = _ladder_mirror(grid, up1, dn1)
    if dn_wing is None:
        raise GeometryRefusal("NO_CHAIN", detail="NO_LISTED_STRIKE:dn_wing")
    legs = (
        NativeLeg("atm", "P", "buy", 2.0, atm, expiry),
        NativeLeg("up1", "P", "sell", 2.0, up1, expiry),
        NativeLeg("dn1", "P", "sell", 2.0, dn1, expiry),
        NativeLeg("up_wing", "P", "buy", 1.0, up_wing, expiry),
        NativeLeg("dn_wing", "P", "buy", 1.0, dn_wing, expiry),
    )
    _check_ladder_collisions(legs)
    return legs


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


def _resolve_generate_expiry(strategy: str, inputs: Mapping[str, Any]) -> str:
    """Expiry for a non-straddle (put-ladder) leg, or a straddle leg whose
    caller supplied both ``strike`` and ``expiry`` already (``generate``
    handles the straddle quote-selection case itself, before calling this).

    A captured ``expiry``/``post_event_expiry`` is used as before
    (``_expiry``). Otherwise -- the state every row legacy failed to price
    arrives in -- falls back to a native selection off the listed puts
    (``_listed_put_expiries`` + ``_resolve_straddle_expiry``, the same
    ``first_post_event``/``fixed`` port every enabled put-ladder strategy's
    single expiry uses), or to ``_expiry``'s ``MISSING_EXPIRY`` refusal when
    nothing is listed at all (the pre-existing degenerate unit-test case).
    """
    if (strategy not in {"STR-THRU", "STR-RUNUP"}
            and inputs.get("expiry") is None
            and inputs.get("post_event_expiry") is None):
        put_expiries = _listed_put_expiries(inputs)
        if put_expiries:
            return _resolve_straddle_expiry(inputs, put_expiries)
    return _expiry(inputs)


def generate(strategy: str, inputs: Mapping[str, Any]) -> Geometry:
    """Generate one strategy from explicit spot, forecast and expiry inputs."""
    if strategy not in STRATEGIES:
        raise GeometryRefusal("UNKNOWN_STRATEGY")
    if strategy in DISABLED:
        return Geometry(strategy, 0.0, 0.0, (), DISABLED[strategy])
    spot = _finite_float(inputs.get("spot"), "spot")
    forecast = abs(_finite_float(inputs.get("forecast_abs_move", inputs.get("forecast", 0.0)), "forecast"))
    divisor = {"TWIN-P": 1.5, "TWIN-P5": 1.0, "CND-PS": 2.0,
               "BFLY-P": 1.0, "BFLY-P5": 3.0, "RAMP7": 3.0,
               "CTR5": 2.0}.get(strategy, 1.0)
    width = _finite_float(inputs.get("width", forecast / divisor / 100.0 * spot), "width")
    if width <= 0 and strategy not in {"STR-THRU", "STR-RUNUP"}:
        raise GeometryRefusal("ZERO_WIDTH")
    selected = (_select_listed_straddle(inputs, spot)
                if strategy in {"STR-THRU", "STR-RUNUP"}
                and (inputs.get("strike") is None or inputs.get("expiry") is None)
                else None)
    expiry = (str(selected[1]) if selected is not None
              else _resolve_generate_expiry(strategy, inputs))
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
        strike = _finite_float(
            inputs.get("strike", spot) if inputs.get("strike") is not None
            else selected[0] if selected is not None else spot,
            "strike",
        )
        legs = (NativeLeg("call", "C", "buy", 1.0, strike, expiry),
                NativeLeg("put", "P", "buy", 1.0, strike, expiry))
        return Geometry(strategy, spot, width, legs)
    legs = _resolve_put_ladder_legs(strategy, inputs, spot, width, expiry)
    # `width` here is the CONTINUOUS target the ladder was asked for, not
    # what the listed grid actually gave -- `_resolve_put_ladder_legs` snaps
    # every offset to the nearest listed strike (`_ladder_offset_from`), so
    # the realised atm-to-up1 spacing can differ from `width` on a grid
    # coarse enough for the snap to matter. `cost_over_width`'s denominator
    # must be the spacing the market actually offered (mirrors legacy's
    # `_structure_width`, engine/score.py, and the `resolved_legs` branch
    # above, which already does this via the same `_resolved_width` call) --
    # otherwise the ratio compares a real premium against a width nobody
    # ever traded.
    return Geometry(strategy, spot, _resolved_width(legs), legs)


def _resolve_put_ladder_legs(strategy: str, inputs: Mapping[str, Any], spot: float,
                             width: float, expiry: str) -> tuple[NativeLeg, ...]:
    """Dispatch CND-PS/TWIN-P/TWIN-P5/BFLY-P/BFLY-P5/RAMP7/CTR5 to the
    grid-based resolvers when a real chain is present for this right+expiry,
    else preserve the pre-existing continuous-spot arithmetic (the
    unit-test degenerate case with no ``quotes`` at all). Extracted out of
    :func:`generate` to keep its own branch count under the complexity
    budget; this is pure dispatch, no behaviour lives here that isn't also
    named by one of the ``_resolve_*_on_grid`` functions or ``_put_ladder``.
    """
    grid = _grid_for_right(inputs, "P", expiry)
    if strategy == "CND-PS":
        if grid:
            # Real chain domain present: legs MUST land on listed strikes,
            # or refuse -- see _resolve_ladder_on_grid's docstring. Never
            # snap to a nearby-but-different strike here.
            return _resolve_ladder_on_grid(strategy, spot, width, expiry, grid)
        # No chain domain supplied at all: unchanged pre-existing
        # continuous-spot arithmetic.
        return (
            NativeLeg("atm", "P", "buy", 0.0, spot, expiry),
            NativeLeg("up1", "P", "sell", 1.0, spot + width, expiry),
            NativeLeg("dn1", "P", "sell", 1.0, spot - width, expiry),
            NativeLeg("up2", "P", "buy", 1.0, spot + 2.0 * width, expiry),
            NativeLeg("dn2", "P", "buy", 1.0, spot - 2.0 * width, expiry),
        )
    if grid and strategy == "TWIN-P":
        return _resolve_twin_peak_on_grid(spot, width, expiry, grid)
    if grid and strategy == "TWIN-P5":
        return _resolve_twin_peak_5_on_grid(spot, width, expiry, grid)
    if grid and strategy in _LADDER_SPECS:
        return _resolve_ladder_on_grid(strategy, spot, width, expiry, grid)
    patterns = {
        "TWIN-P": ((0, 2.0), (1, -1.0), (2, -1.0), (4, 1.0)),
        "TWIN-P5": ((0, 2.0), (1, -2.0), (3, 1.0)),
        "BFLY-P": ((0, -2.0), (1, 1.0)),
        "BFLY-P5": ((0, -4.0), (1, 1.0), (3, 1.0)),
        "RAMP7": ((0, -2.0), (1, -1.0), (2, 1.0), (3, 1.0)),
        "CTR5": ((0, -2.0), (1, -1.0), (2, 2.0)),
    }
    return _put_ladder(strategy, spot, width, expiry, patterns[strategy])


def price(geometry: Geometry, quotes: Mapping[Any, Mapping[str, Any]],
          fill_alpha: float = 0.5) -> Pricing:
    """Price generated legs using the shared worst-to-best fill convention."""
    if geometry.refusal:
        return Pricing(geometry.strategy, geometry.spot, 0.0, (), geometry.refusal)
    alpha = _finite_float(fill_alpha, "fill_alpha")
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
        bid = _finite_float(quote.get("bid"), f"{leg.name}.bid")
        ask = _finite_float(quote.get("ask"), f"{leg.name}.ask")
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
