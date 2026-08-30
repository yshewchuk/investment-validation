"""Trade-structure definitions → leg lists, and the one pricing path.

Every strategy in the program is expressed as a :class:`Structure`: a small,
serializable spec that resolves against an option-chain snapshot into concrete
legs, which are then priced through a :class:`~engine.fills.FillModel`. Phases
1–6 all price through :func:`price_structure`; there is deliberately no second
pricing implementation to drift away from this one.

Three structures ship with Phase 0:

``put_calendar`` (CAL-P)
    Short a ~1 DTE front put, long a back put. **Both legs open together and
    close together** — at no point is the short put naked, which makes this a
    defined-risk debit structure whose max loss is approximately the net debit.
    Same-strike by default; ``back_moneyness`` makes it a diagonal.

``straddle_through`` (STR-THRU)
    Long ATM straddle bought shortly before the print, sold immediately after.

``straddle_runup`` (STR-RUNUP)
    Long ATM straddle bought early, sold immediately *before* the print.

Day rules are expressed in trading days relative to the print and resolved by
:mod:`engine.calendar`, because "shortly before the print" means T−1 for a BMO
announcement and T−0 for an AMC one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from engine.fills import FillModel

__all__ = [
    "ExpirySelector",
    "StrikeSelector",
    "LegSpec",
    "Structure",
    "ResolvedLeg",
    "StructurePrice",
    "ChainSnapshot",
    "put_calendar",
    "straddle_through",
    "straddle_runup",
    "STRUCTURES",
    "price_structure",
    "structure_return",
]

CALL, PUT = "C", "P"
BUY, SELL = "buy", "sell"


class StructureError(Exception):
    """A structure could not be resolved against the given chain."""


# --------------------------------------------------------------------------
# selectors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpirySelector:
    """How a leg picks its expiry out of the expiries present in a chain.

    ``kind``:

    ``"nearest_dte"``
        Expiry whose DTE is closest to ``target_dte``; ties break long, since
        an extra day of optionality is the safer error.
    ``"first_dte_at_least"``
        Earliest expiry with ``dte >= target_dte``.
    ``"first_post_event"``
        Earliest expiry that still exists on the far side of the print. This is
        the one that matters for every through-the-print structure: 430 events
        in the existing S2 trade set have an expiry at or before the earnings
        date, and pricing "the front expiry" without this filter silently books
        an option that expired before the event.

        **Session matters here.** For a BMO print the announcement lands before
        the open, so an expiry *on* the event date survives it. For an AMC print
        the announcement lands after the close, so an expiry on the event date
        died hours earlier — it must be excluded. Without the session the rule
        has to assume the permissive case, and the invariant ends up resting on
        a pull parameter (``dte=1,45`` happens to produce no dte-0 rows) rather
        than on the selector.
    """

    kind: str
    target_dte: int | None = None
    min_dte: int | None = None
    max_dte: int | None = None

    VALID = ("nearest_dte", "first_dte_at_least", "first_post_event")

    def __post_init__(self) -> None:
        if self.kind not in self.VALID:
            raise ValueError(f"unknown expiry selector {self.kind!r}")
        if self.kind in ("nearest_dte", "first_dte_at_least") and self.target_dte is None:
            raise ValueError(f"{self.kind} requires target_dte")

    def select(
        self,
        chain: pd.DataFrame,
        event_date: pd.Timestamp,
        session: str | None = None,
    ) -> pd.Timestamp:
        exp = chain[["expiry", "dte"]].drop_duplicates().sort_values("expiry")
        if self.min_dte is not None:
            exp = exp[exp["dte"] >= self.min_dte]
        if self.max_dte is not None:
            exp = exp[exp["dte"] <= self.max_dte]
        if exp.empty:
            raise StructureError(f"no expiry survives {self}")

        if self.kind == "first_post_event":
            # An AMC print happens after the close, so an expiry ON the event
            # date is already dead when the news lands. Unknown session falls
            # back to the inclusive rule, which is what the legacy trade sets
            # used, so behaviour only tightens where the session is known.
            if str(session).upper() == "AMC":
                post = exp[exp["expiry"] > event_date]
                if post.empty:
                    raise StructureError(
                        f"no expiry after the AMC event date {event_date.date()} "
                        "(an expiry on the event date dies at the close, before "
                        "the announcement)"
                    )
            else:
                post = exp[exp["expiry"] >= event_date]
                if post.empty:
                    raise StructureError(
                        f"no expiry on or after the event date {event_date.date()}"
                    )
            return pd.Timestamp(post.iloc[0]["expiry"])

        if self.kind == "first_dte_at_least":
            ok = exp[exp["dte"] >= self.target_dte]
            if ok.empty:
                raise StructureError(f"no expiry with dte >= {self.target_dte}")
            return pd.Timestamp(ok.iloc[0]["expiry"])

        # nearest_dte, ties broken toward the longer-dated expiry
        order = exp.assign(
            gap=(exp["dte"] - self.target_dte).abs(), neg_dte=-exp["dte"]
        ).sort_values(["gap", "neg_dte"])
        return pd.Timestamp(order.iloc[0]["expiry"])


@dataclass(frozen=True)
class StrikeSelector:
    """How a leg picks its strike once the expiry is fixed.

    ``kind``:

    ``"atm"``           strike closest to spot
    ``"moneyness"``     strike closest to ``moneyness * spot``
    ``"delta"``         strike whose |delta| is closest to ``target_delta``
    ``"fixed"``         exactly ``strike`` (used to replay a recorded trade)
    ``"same_as"``       reuse the strike already resolved for leg ``ref``
    """

    kind: str = "atm"
    moneyness: float | None = None
    target_delta: float | None = None
    strike: float | None = None
    ref: str | None = None

    VALID = ("atm", "moneyness", "delta", "fixed", "same_as")

    def __post_init__(self) -> None:
        if self.kind not in self.VALID:
            raise ValueError(f"unknown strike selector {self.kind!r}")
        need = {
            "moneyness": "moneyness",
            "delta": "target_delta",
            "fixed": "strike",
            "same_as": "ref",
        }.get(self.kind)
        if need is not None and getattr(self, need) is None:
            raise ValueError(f"{self.kind} requires {need}")

    def select(
        self,
        rows: pd.DataFrame,
        spot: float,
        resolved: dict[str, float],
    ) -> float:
        if rows.empty:
            raise StructureError("no chain rows at the selected expiry")

        if self.kind == "same_as":
            if self.ref not in resolved:
                raise StructureError(
                    f"leg {self.ref!r} must resolve before a same_as reference to it"
                )
            target = resolved[self.ref]
            if not np.isclose(rows["strike"], target).any():
                raise StructureError(
                    f"strike {target} from leg {self.ref!r} absent at this expiry"
                )
            return float(target)

        if self.kind == "fixed":
            hit = rows[np.isclose(rows["strike"], self.strike)]
            if hit.empty:
                raise StructureError(f"strike {self.strike} absent at this expiry")
            return float(self.strike)

        if self.kind == "delta":
            if "delta" not in rows or rows["delta"].isna().all():
                raise StructureError("delta selector needs a delta column")
            gap = (rows["delta"].abs() - abs(self.target_delta)).abs()
            return float(rows.loc[gap.idxmin(), "strike"])

        target = spot if self.kind == "atm" else self.moneyness * spot
        gap = (rows["strike"] - target).abs()
        return float(rows.loc[gap.idxmin(), "strike"])


# --------------------------------------------------------------------------
# specs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LegSpec:
    name: str
    right: str  # "C" or "P"
    side: str  # "buy" or "sell"
    expiry: ExpirySelector
    strike: StrikeSelector = field(default_factory=StrikeSelector)
    qty: float = 1.0

    def __post_init__(self) -> None:
        if self.right not in (CALL, PUT):
            raise ValueError(f"right must be {CALL!r} or {PUT!r}, got {self.right!r}")
        if self.side not in (BUY, SELL):
            raise ValueError(f"side must be {BUY!r} or {SELL!r}, got {self.side!r}")
        if self.qty <= 0:
            raise ValueError("qty must be positive; direction is carried by `side`")


@dataclass(frozen=True)
class Structure:
    """A named, serializable trade structure.

    ``entry_offset`` / ``exit_offset`` are in **trading days relative to the
    print**: ``-1`` is one trading day before the last pre-print close, ``0`` is
    the last pre-print close itself, and ``+1`` is the first post-print close.
    :func:`engine.calendar.resolve_offsets` maps those onto real dates using the
    BMO/AMC session, so the same spec means the same thing for both sessions.
    """

    name: str
    legs: tuple[LegSpec, ...]
    entry_offset: int
    exit_offset: int
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.legs:
            raise ValueError("a structure needs at least one leg")
        names = [leg.name for leg in self.legs]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate leg names in {self.name}: {names}")
        if self.exit_offset <= self.entry_offset:
            raise ValueError(
                f"{self.name}: exit_offset ({self.exit_offset}) must be after "
                f"entry_offset ({self.entry_offset})"
            )
        for leg in self.legs:
            if leg.strike.kind == "same_as" and leg.strike.ref not in names:
                raise ValueError(f"{self.name}: same_as ref {leg.strike.ref!r} is not a leg")
            if leg.strike.kind == "same_as" and names.index(leg.strike.ref) > names.index(leg.name):
                raise ValueError(
                    f"{self.name}: leg {leg.name!r} references {leg.strike.ref!r}, "
                    "which resolves later"
                )

    @property
    def holds_through_print(self) -> bool:
        return self.entry_offset <= 0 < self.exit_offset

    @property
    def has_short_leg(self) -> bool:
        return any(leg.side == SELL for leg in self.legs)

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable description, for report provenance and specs."""
        return {
            "name": self.name,
            "entry_offset": self.entry_offset,
            "exit_offset": self.exit_offset,
            "description": self.description,
            "params": dict(self.params),
            "legs": [
                {
                    "name": leg.name,
                    "right": leg.right,
                    "side": leg.side,
                    "qty": leg.qty,
                    "expiry": {k: v for k, v in vars(leg.expiry).items() if v is not None},
                    "strike": {k: v for k, v in vars(leg.strike).items() if v is not None},
                }
                for leg in self.legs
            ],
        }


# --------------------------------------------------------------------------
# resolution + pricing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainSnapshot:
    """One (ticker, obs_date) option chain, plus the event it is being traded around.

    ``rows`` carries the Tier-2 ``option_chains`` columns: expiry, dte, strike,
    right, bid, ask, mid, iv, delta, spot.

    ``session`` (BMO/AMC) is optional but load-bearing where present: it decides
    whether an expiry on the event date survives the print. Leaving it unset
    keeps the permissive legacy behaviour.
    """

    ticker: str
    obs_date: pd.Timestamp
    event_date: pd.Timestamp
    rows: pd.DataFrame
    spot: float | None = None
    session: str | None = None

    REQUIRED = ("expiry", "dte", "strike", "right", "bid", "ask")

    def __post_init__(self) -> None:
        missing = [c for c in self.REQUIRED if c not in self.rows.columns]
        if missing:
            raise StructureError(f"chain snapshot missing columns: {missing}")

    @property
    def spot_price(self) -> float:
        if self.spot is not None:
            return float(self.spot)
        if "spot" in self.rows.columns and self.rows["spot"].notna().any():
            return float(self.rows["spot"].dropna().iloc[0])
        raise StructureError("chain snapshot has no spot price")


@dataclass(frozen=True)
class ResolvedLeg:
    name: str
    right: str
    side: str
    qty: float
    expiry: pd.Timestamp
    strike: float
    dte: int
    bid: float
    ask: float
    price: float
    cash_flow: float
    wide_market: bool


@dataclass(frozen=True)
class StructurePrice:
    """The priced structure at one point in time.

    ``closing`` records which direction the trade was priced in. It matters:
    opening a long leg lifts the ask and closing it hits the bid, so a close
    priced as if it were an open overstates the exit by a full spread on every
    long leg. The flag keeps that distinction in the data rather than in a
    convention someone has to remember.
    """

    structure: str
    ticker: str
    obs_date: pd.Timestamp
    event_date: pd.Timestamp
    spot: float
    alpha: float
    legs: tuple[ResolvedLeg, ...]
    closing: bool = False

    @property
    def net_cash_flow(self) -> float:
        """Signed cash: negative when cash goes out, positive when it comes in."""
        return float(sum(leg.cash_flow for leg in self.legs))

    @property
    def cost(self) -> float:
        """Net debit to open — positive when you pay (the usual case here)."""
        if self.closing:
            raise StructureError("cost is an opening concept; use exit_value")
        return -self.net_cash_flow

    @property
    def exit_value(self) -> float:
        """Net cash received on closing — positive when the position is worth something."""
        if not self.closing:
            raise StructureError("exit_value is a closing concept; use cost")
        return self.net_cash_flow

    @property
    def any_wide_market(self) -> bool:
        return any(leg.wide_market for leg in self.legs)

    def leg(self, name: str) -> ResolvedLeg:
        for leg in self.legs:
            if leg.name == name:
                return leg
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "structure": self.structure,
            "ticker": self.ticker,
            "obs_date": str(self.obs_date.date()),
            "event_date": str(self.event_date.date()),
            "spot": self.spot,
            "alpha": self.alpha,
            "closing": self.closing,
            "net_cash_flow": self.net_cash_flow,
            "any_wide_market": self.any_wide_market,
            "legs": [vars(leg) | {"expiry": str(leg.expiry.date())} for leg in self.legs],
        }


#: Closing a position transacts the opposite side of every leg.
_OPPOSITE = {BUY: SELL, SELL: BUY}


def price_structure(
    structure: Structure,
    snapshot: ChainSnapshot,
    fill: FillModel,
    *,
    pin: Sequence[ResolvedLeg] | None = None,
    closing: bool = False,
) -> StructurePrice:
    """Resolve ``structure`` against ``snapshot`` and price it at ``fill``.

    ``pin`` re-uses the expiries and strikes of an earlier resolution — this is
    how a position is closed on the *same* contracts it was opened on, rather
    than re-running ATM selection against a post-print spot that has moved.
    Closing a structure without pinning is nearly always a bug.

    ``closing=True`` transacts the opposite side of every leg, so a long leg
    opened at the ask (alpha=0) closes at the bid rather than the ask.
    """
    rows = snapshot.rows
    spot = snapshot.spot_price
    pinned = {leg.name: leg for leg in (pin or ())}
    resolved_strikes: dict[str, float] = {}
    out: list[ResolvedLeg] = []

    for spec in structure.legs:
        if spec.name in pinned:
            expiry = pinned[spec.name].expiry
            strike = pinned[spec.name].strike
            at_expiry = rows[rows["expiry"] == expiry]
            if at_expiry.empty:
                raise StructureError(
                    f"{spec.name}: pinned expiry {expiry.date()} absent from the "
                    f"{snapshot.obs_date.date()} chain"
                )
        else:
            expiry = spec.expiry.select(rows, snapshot.event_date, snapshot.session)
            at_expiry = rows[rows["expiry"] == expiry]
            strike = spec.strike.select(at_expiry, spot, resolved_strikes)

        resolved_strikes[spec.name] = strike
        hit = at_expiry[
            np.isclose(at_expiry["strike"], strike) & (at_expiry["right"] == spec.right)
        ]
        if hit.empty:
            raise StructureError(
                f"{spec.name}: no {spec.right} at strike {strike} expiry "
                f"{expiry.date()} on {snapshot.obs_date.date()}"
            )
        row = hit.iloc[0]
        bid, ask = float(row["bid"]), float(row["ask"])
        side = _OPPOSITE[spec.side] if closing else spec.side
        out.append(
            ResolvedLeg(
                name=spec.name,
                right=spec.right,
                side=side,
                qty=spec.qty,
                expiry=pd.Timestamp(expiry),
                strike=float(strike),
                dte=int(row["dte"]),
                bid=bid,
                ask=ask,
                price=float(fill.price(side, bid, ask)),
                cash_flow=float(fill.cash_flow(side, bid, ask, spec.qty)),
                wide_market=bool(FillModel.is_wide(bid, ask)),
            )
        )

    return StructurePrice(
        structure=structure.name,
        ticker=snapshot.ticker,
        obs_date=pd.Timestamp(snapshot.obs_date),
        event_date=pd.Timestamp(snapshot.event_date),
        spot=spot,
        alpha=fill.alpha,
        legs=tuple(out),
        closing=closing,
    )


def structure_return(entry: StructurePrice, exit_: StructurePrice) -> dict[str, float]:
    """P&L of opening at ``entry`` and closing at ``exit_``.

    Return is quoted on the net debit, matching the convention every existing
    trade set uses (``ret = exit_val / cost - 1``). A structure opened for a net
    credit has no meaningful return-on-debit, so ``ret`` is ``nan`` there and
    ``pnl`` is the number to use.
    """
    if entry.closing:
        raise StructureError("entry must be priced with closing=False")
    if not exit_.closing:
        raise StructureError("exit must be priced with closing=True")

    entry_names = [leg.name for leg in entry.legs]
    exit_names = [leg.name for leg in exit_.legs]
    if entry_names != exit_names:
        raise StructureError(f"leg mismatch: {entry_names} vs {exit_names}")
    for a, b in zip(entry.legs, exit_.legs):
        if a.expiry != b.expiry or not np.isclose(a.strike, b.strike):
            raise StructureError(
                f"leg {a.name} changed contract between entry and exit "
                f"({a.strike}@{a.expiry.date()} → {b.strike}@{b.expiry.date()}); "
                "pass the entry legs as `pin=` when pricing the exit"
            )

    cost = entry.cost
    exit_value = exit_.exit_value
    pnl = exit_value - cost
    return {
        "cost": cost,
        "exit_value": exit_value,
        "pnl": pnl,
        "ret": pnl / cost if cost > 0 else float("nan"),
        "alpha_entry": entry.alpha,
        "alpha_exit": exit_.alpha,
    }


# --------------------------------------------------------------------------
# the three Phase-0 structures
# --------------------------------------------------------------------------


def put_calendar(
    back_dte: int = 20,
    front_dte: int = 1,
    back_moneyness: float | None = None,
    entry_offset: int = 0,
    exit_offset: int = 1,
) -> Structure:
    """CAL-P — short ~``front_dte`` put, long ~``back_dte`` put, opened and closed together.

    The front leg uses ``first_post_event`` so the short put is always still
    alive at the print; the back leg is same-strike unless ``back_moneyness``
    makes it a diagonal.

    Both offsets default to entry at the last pre-print close and exit at the
    first post-print close: the short leg is never carried naked, and it is
    never left outstanding past the session in which the event resolves.
    """
    front_strike = StrikeSelector(kind="atm")
    back_strike = (
        StrikeSelector(kind="same_as", ref="front_put")
        if back_moneyness is None
        else StrikeSelector(kind="moneyness", moneyness=back_moneyness)
    )
    return Structure(
        name="CAL-P",
        description=(
            "Put calendar: short front put + long back put, both legs opened "
            "together shortly before the print and closed together after it."
        ),
        legs=(
            LegSpec(
                name="front_put",
                right=PUT,
                side=SELL,
                expiry=ExpirySelector(kind="first_post_event", max_dte=max(front_dte * 4, 7)),
                strike=front_strike,
            ),
            LegSpec(
                name="back_put",
                right=PUT,
                side=BUY,
                expiry=ExpirySelector(kind="first_dte_at_least", target_dte=back_dte),
                strike=back_strike,
            ),
        ),
        entry_offset=entry_offset,
        exit_offset=exit_offset,
        params={
            "front_dte": front_dte,
            "back_dte": back_dte,
            "back_moneyness": back_moneyness,
        },
    )


def straddle_through(entry_offset: int = 0, exit_offset: int = 1) -> Structure:
    """STR-THRU — long ATM straddle bought before the print, sold right after."""
    expiry = ExpirySelector(kind="first_post_event")
    return Structure(
        name="STR-THRU",
        description="Long ATM straddle held through the print (earliest post-event expiry).",
        legs=(
            LegSpec("call", CALL, BUY, expiry, StrikeSelector("atm")),
            LegSpec("put", PUT, BUY, expiry, StrikeSelector(kind="same_as", ref="call")),
        ),
        entry_offset=entry_offset,
        exit_offset=exit_offset,
    )


def straddle_runup(entry_offset: int = -14, exit_offset: int = 0, target_dte: int = 30) -> Structure:
    """STR-RUNUP — long ATM straddle bought early, sold immediately *before* the print.

    ``exit_offset=0`` is the last pre-print close, so the position never sees
    the event: this harvests the IV run-up, not the move.
    """
    expiry = ExpirySelector(kind="first_dte_at_least", target_dte=target_dte)
    return Structure(
        name="STR-RUNUP",
        description="Long ATM straddle entered early and exited before the print (IV run-up).",
        legs=(
            LegSpec("call", CALL, BUY, expiry, StrikeSelector("atm")),
            LegSpec("put", PUT, BUY, expiry, StrikeSelector(kind="same_as", ref="call")),
        ),
        entry_offset=entry_offset,
        exit_offset=exit_offset,
        params={"target_dte": target_dte},
    )


#: Factories keyed by strategy code, so specs can name a structure as a string.
STRUCTURES = {
    "CAL-P": put_calendar,
    "STR-THRU": straddle_through,
    "STR-RUNUP": straddle_runup,
}
