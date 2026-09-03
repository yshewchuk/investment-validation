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

``put_condor`` (CND-P) was added later:
    Long the wings, short the two strikes straddling the money, all puts, all
    one expiry, evenly spaced. The first structure in the program that WANTS the
    print to be quiet — the other three are long the move or long the run-up.

Day rules are expressed in trading days relative to the print and resolved by
:mod:`engine.calendar`, because "shortly before the print" means T−1 for a BMO
announcement and T−0 for an AMC one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from engine.fills import MIN_MEANINGFUL_COST, FillModel

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
    "put_condor",
    "twin_peak",
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
    ``"fixed"``
        Exactly ``expiry``. Used to price a caller-specified expiry — the
        Phase 1 scoring API's ``expiry=`` argument — and to replay a recorded
        trade on the contract it actually traded.
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
    expiry: Any = None

    VALID = ("nearest_dte", "first_dte_at_least", "first_post_event", "fixed")

    def __post_init__(self) -> None:
        if self.kind not in self.VALID:
            raise ValueError(f"unknown expiry selector {self.kind!r}")
        if self.kind in ("nearest_dte", "first_dte_at_least") and self.target_dte is None:
            raise ValueError(f"{self.kind} requires target_dte")
        if self.kind == "fixed" and self.expiry is None:
            raise ValueError("fixed requires expiry")

    def select(
        self,
        chain: pd.DataFrame,
        event_date: pd.Timestamp,
        session: str | None = None,
    ) -> pd.Timestamp:
        exp = chain[["expiry", "dte"]].drop_duplicates().sort_values("expiry")

        # A caller-named expiry bypasses the DTE filters: they exist to *choose*
        # an expiry, and there is nothing left to choose.
        if self.kind == "fixed":
            wanted = pd.Timestamp(self.expiry)
            if not (exp["expiry"] == wanted).any():
                raise StructureError(f"expiry {wanted.date()} absent from this chain")
            return wanted

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
    ``"bracket"``       the listed strike immediately below (``side="below"``)
                        or above (``side="above"``) spot — the pair that
                        straddles the money
    ``"offset_from"``   the listed strike closest to ``ref``'s strike plus
                        ``moneyness * spot``, excluding ``ref``'s own strike
    ``"grid_step"``     ``steps`` positions along the listed grid from ``ref``
                        — the ticker's OWN granularity, not a share of spot
    ``"mirror"``        ``2 * about - ref``: the strike as far the other side of
                        ``about`` as ``ref`` is this side of it, and it must be
                        listed

    The last three exist for :func:`put_condor`, whose four strikes are one
    geometric object rather than four independent choices. ``bracket`` puts the
    two shorts either side of spot, ``offset_from`` sets the common spacing by
    snapping it to the listed grid, and ``mirror`` places each wing so that the
    spacing is EXACTLY even in dollars. Even spacing is not cosmetic: a condor
    whose lower gap exceeds its upper gap pays ``(K4-K3) - (K2-K1) < 0`` below
    the bottom strike, so its loss is no longer capped at the debit. Requiring
    the mirrored strike to be listed — rather than snapping to the nearest one —
    is what keeps the defined-risk claim a property of the structure instead of
    an approximation.

    Selection is restricted to the leg's own ``right`` where the caller passes
    one. Calls and puts do not share a delta at a strike, and a strike that is
    listed for one right and not the other would otherwise be chosen and then
    fail to price.
    """

    kind: str = "atm"
    moneyness: float | None = None
    target_delta: float | None = None
    strike: float | None = None
    ref: str | None = None
    side: str | None = None
    about: str | None = None
    steps: int | None = None

    VALID = ("atm", "moneyness", "delta", "fixed", "same_as",
             "bracket", "offset_from", "grid_step", "mirror")

    #: Legs this selector must resolve after, by field.
    REF_FIELDS = ("ref", "about")

    def __post_init__(self) -> None:
        if self.kind not in self.VALID:
            raise ValueError(f"unknown strike selector {self.kind!r}")
        need = {
            "moneyness": ("moneyness",),
            "delta": ("target_delta",),
            "fixed": ("strike",),
            "same_as": ("ref",),
            "bracket": ("side",),
            "offset_from": ("ref", "moneyness"),
            "grid_step": ("ref", "steps"),
            "mirror": ("ref", "about"),
        }.get(self.kind, ())
        for field_name in need:
            if getattr(self, field_name) is None:
                raise ValueError(f"{self.kind} requires {field_name}")
        if self.kind == "bracket" and self.side not in ("below", "above"):
            raise ValueError(f"bracket side must be 'below' or 'above', got {self.side!r}")

    @property
    def refs(self) -> tuple[str, ...]:
        """Leg names this selector reads, in no particular order."""
        return tuple(
            value for value in (getattr(self, f) for f in self.REF_FIELDS)
            if value is not None
        )

    @staticmethod
    def _grid(rows: pd.DataFrame) -> np.ndarray:
        """Sorted distinct strikes present in ``rows``."""
        return np.sort(np.unique(rows["strike"].to_numpy(dtype=float)))

    def select(
        self,
        rows: pd.DataFrame,
        spot: float,
        resolved: dict[str, float],
        *,
        right: str | None = None,
    ) -> float:
        if rows.empty:
            raise StructureError("no chain rows at the selected expiry")
        if right is not None and "right" in rows.columns:
            same_right = rows[rows["right"] == right]
            # A chain missing the right entirely is a data problem, not a
            # selection one; fall back so the leg's own lookup raises the
            # message that says which contract was absent.
            if not same_right.empty:
                rows = same_right

        if self.kind == "bracket":
            grid = self._grid(rows)
            side = grid[grid <= spot] if self.side == "below" else grid[grid > spot]
            if side.size == 0:
                raise StructureError(
                    f"no listed strike {self.side} spot {spot:.4f} at this expiry"
                )
            return float(side[-1] if self.side == "below" else side[0])

        if self.kind == "grid_step":
            if self.ref not in resolved:
                raise StructureError(
                    f"leg {self.ref!r} must resolve before a grid_step reference to it"
                )
            grid = self._grid(rows)
            hit = np.flatnonzero(np.isclose(grid, resolved[self.ref]))
            if hit.size == 0:
                raise StructureError(
                    f"anchor strike {resolved[self.ref]:.4f} is not on this expiry's grid"
                )
            target = int(hit[0]) + int(self.steps)
            if not 0 <= target < grid.size:
                raise StructureError(
                    f"{self.steps:+d} grid steps from {resolved[self.ref]:.4f} runs off "
                    f"the ladder ({grid.size} listed strikes)"
                )
            return float(grid[target])

        if self.kind == "offset_from":
            if self.ref not in resolved:
                raise StructureError(
                    f"leg {self.ref!r} must resolve before an offset_from reference to it"
                )
            anchor = resolved[self.ref]
            grid = self._grid(rows)
            # Candidates lie strictly on the side the offset points, so the sign
            # of `moneyness` is a guarantee rather than a hope. Excluding the
            # anchor keeps a spacing smaller than one grid step from snapping
            # back onto it and collapsing two of the condor's strikes into one;
            # excluding the far side keeps a lopsided grid (dense above spot,
            # sparse below) from returning a nearer strike in the wrong
            # direction and inverting the structure.
            grid = grid[grid > anchor] if self.moneyness > 0 else grid[grid < anchor]
            if grid.size == 0:
                side = "above" if self.moneyness > 0 else "below"
                raise StructureError(
                    f"no listed strike {side} {anchor:.4f} at this expiry"
                )
            target = anchor + self.moneyness * spot
            return float(grid[np.abs(grid - target).argmin()])

        if self.kind == "mirror":
            for name in (self.ref, self.about):
                if name not in resolved:
                    raise StructureError(
                        f"leg {name!r} must resolve before a mirror reference to it"
                    )
            target = 2.0 * resolved[self.about] - resolved[self.ref]
            grid = self._grid(rows)
            if not np.isclose(grid, target).any():
                raise StructureError(
                    f"mirrored strike {target:.4f} "
                    f"(2x{resolved[self.about]:.4f} - {resolved[self.ref]:.4f}) "
                    "is not listed at this expiry"
                )
            return float(target)

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

    ``decision_offset`` is the close the trade is **decided** on, on the same
    anchor. ``None`` — the default, and what every structure shipped with —
    means "decided at the entry close". That is unactionable for a structure
    that enters at the last pre-print close: the chain you price against only
    exists after that close has happened, so the prediction cannot be reached
    in time to place the trade. Setting ``decision_offset`` earlier than
    ``entry_offset`` buys lead time, at the cost of quoting a premium from a
    close you will not fill at. See ``guides/str_thru_t2_decision.md``.
    """

    name: str
    legs: tuple[LegSpec, ...]
    entry_offset: int
    exit_offset: int
    decision_offset: int | None = None
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
        if self.decision_offset is not None and self.decision_offset > self.entry_offset:
            raise ValueError(
                f"{self.name}: decision_offset ({self.decision_offset}) cannot be "
                f"after entry_offset ({self.entry_offset}) — a trade cannot be "
                "decided on information that only exists once it is already on"
            )
        # Every cross-leg strike reference — same_as, offset_from, mirror — must
        # name a real leg that resolves EARLIER, because resolution is a single
        # pass over `legs` in order and a forward reference reads an empty slot.
        for leg in self.legs:
            for ref in leg.strike.refs:
                if ref not in names:
                    raise ValueError(
                        f"{self.name}: leg {leg.name!r} references {ref!r}, "
                        "which is not a leg"
                    )
                if names.index(ref) > names.index(leg.name):
                    raise ValueError(
                        f"{self.name}: leg {leg.name!r} references {ref!r}, "
                        "which resolves later"
                    )

    @property
    def decided_at(self) -> int:
        """The offset the trade is decided on — ``entry_offset`` when unset.

        Read this rather than ``decision_offset`` anywhere a date is being
        resolved, so that "unset" resolves to the entry close in exactly one
        place instead of at every call site.
        """
        return self.entry_offset if self.decision_offset is None else self.decision_offset

    @property
    def decided_early(self) -> bool:
        """True when the decision close is strictly before the entry close."""
        return self.decided_at < self.entry_offset

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
            "decision_offset": self.decision_offset,
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
            strike = spec.strike.select(
                at_expiry, spot, resolved_strikes, right=spec.right
            )

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
    # A cost this small is floating-point noise around zero, not a price — see
    # MIN_MEANINGFUL_COST. `ret` on it is not a large return, it is division by
    # an artifact, and reports NaN precisely because the true cost is unknown
    # to be positive at all, not because it is provably a credit.
    meaningful = cost > MIN_MEANINGFUL_COST
    return {
        "cost": cost,
        "exit_value": exit_value,
        "pnl": pnl,
        "ret": pnl / cost if meaningful else float("nan"),
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
    decision_offset: int | None = None,
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
        decision_offset=decision_offset,
        params={
            "front_dte": front_dte,
            "back_dte": back_dte,
            "back_moneyness": back_moneyness,
        },
    )


def straddle_through(
    entry_offset: int = 0,
    exit_offset: int = 1,
    decision_offset: int | None = None,
) -> Structure:
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
        decision_offset=decision_offset,
    )


def straddle_runup(
    entry_offset: int = -14,
    exit_offset: int = 0,
    target_dte: int = 30,
    decision_offset: int | None = None,
) -> Structure:
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
        decision_offset=decision_offset,
        params={"target_dte": target_dte},
    )


def put_condor(
    width: float = 0.05,
    entry_offset: int = 0,
    exit_offset: int = 1,
    decision_offset: int | None = None,
) -> Structure:
    """CND-P — long put condor: long the wings, short the two strikes around spot.

    Four puts, one expiry (the first that survives the print), evenly spaced::

        K1 = K2 - w   BUY   long_lo    out of the money
        K2            SELL  short_lo   the listed strike at or below spot
        K3 = K2 + w   SELL  short_hi   the next one up, w above K2, above spot
        K4 = K3 + w   BUY   long_hi    in the money

    ``w`` is the common spacing: the listed strike nearest ``width * spot``
    above ``K2``, so the geometry snaps to the ticker's own strike grid and the
    wings are then MIRRORED to make the three gaps exactly equal. Payoff at
    expiry is a tent — zero at or beyond both wings, ``w`` flat between the two
    shorts, straight ramps in between — so the structure is long the debit and
    wins when the stock goes nowhere. That makes it the first structure here
    that is **short** the event: STR-THRU and STR-RUNUP pay for the move, CND-P
    is paid for it, and its gate therefore wants the prints where the implied
    move is much LARGER than the predicted one.

    **Why the even spacing is load-bearing.** Below ``K1`` the four legs settle
    to ``(K4 - K3) - (K2 - K1)``. Equal gaps make that zero, which is what caps
    the loss at the debit; a wider lower gap makes it negative and the "defined
    risk" claim is simply false. The mirror selector therefore requires the
    wing strike to be LISTED rather than snapping to the nearest one — an event
    whose grid cannot carry a symmetric condor is refused and counted, not
    priced as an approximately-symmetric one.

    Both shorts sit inside long strikes, so neither is naked; ``short_hi`` is
    in the money by construction, which is early-assignment exposure of the
    same kind EXP-102 measured on CAL-P's front put, and is a required output
    of any experiment that trades this.

    Executed in puts alone. A short iron condor — the same tent built from a
    put spread plus a call spread for a credit — is the put-call-parity
    equivalent, and the reason to prefer one expiry in one right here is that
    the replay prices what it can verify: four quotes off one chain, no
    cross-right basis.
    """
    if not width > 0:
        raise ValueError(f"width must be positive, got {width!r}")
    expiry = ExpirySelector(kind="first_post_event")
    return Structure(
        name="CND-P",
        description=(
            "Long put condor: short the two strikes straddling spot, long "
            "evenly spaced wings, one post-event expiry, held through the print."
        ),
        legs=(
            LegSpec("short_lo", PUT, SELL, expiry, StrikeSelector("bracket", side="below")),
            LegSpec(
                "short_hi", PUT, SELL, expiry,
                StrikeSelector("offset_from", ref="short_lo", moneyness=width),
            ),
            LegSpec(
                "long_lo", PUT, BUY, expiry,
                StrikeSelector("mirror", ref="short_hi", about="short_lo"),
            ),
            LegSpec(
                "long_hi", PUT, BUY, expiry,
                StrikeSelector("mirror", ref="short_lo", about="short_hi"),
            ),
        ),
        entry_offset=entry_offset,
        exit_offset=exit_offset,
        decision_offset=decision_offset,
        params={"width": width},
    )


def twin_peak(
    steps: int = 1,
    entry_offset: int = 0,
    exit_offset: int = 1,
    decision_offset: int | None = None,
) -> Structure:
    """TWIN-P — two mirrored put condors sharing a doubled at-the-money long.

    Seven strikes, eight contracts, all puts, one post-event expiry::

        A - 4w  BUY  1      A       BUY  2      A + 4w  BUY  1
        A - 2w  SELL 1      A + w   SELL 1
        A - w   SELL 1      A + 2w  SELL 1

    ``A`` is the listed strike at or below spot; ``w`` is ``steps`` positions
    along that ticker's OWN strike ladder, so the geometry is set by the
    granularity the name actually lists rather than by a share of spot. Every
    other strike is then exact mirror arithmetic off ``A`` and ``A + w``, and
    each one must be LISTED — an event whose ladder cannot carry all seven is
    refused, not approximated.

    **The payoff is twin-peaked**, which is the point::

        2w ┤   ┌──┐      ┌──┐
        1w ┤  ╱    └──┬──┘    ╲
         0 ┼──────────────────────
          -4w  -2w  0  +2w  +4w

    Flat at ``2w`` for a move of one to two ``w`` in EITHER direction, dipping
    to ``w`` at a dead-flat print, zero beyond ``±4w``. Net contracts sum to
    zero so the far tails cancel, the floor is exactly zero everywhere, and max
    loss is therefore the debit.

    That shape is the thesis: earnings moves are usually SMALL but rarely zero,
    and realized lands below implied about two thirds of the time (EXP-120:
    realized/implied median 0.634). CND-P put its maximum at zero move — the
    least likely single outcome. This puts the maximum on the modal one.

    The cost is eight legs, so sixteen spread crossings round trip against a
    debit the design deliberately keeps small. That is the risk, and it is why
    an experiment trading this belongs on names whose markets are tight enough
    for mid to mean something.
    """
    if int(steps) < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    expiry = ExpirySelector(kind="first_post_event")
    atm = StrikeSelector("bracket", side="below")
    return Structure(
        name="TWIN-P",
        description=(
            "Twin-peak put structure: doubled ATM long, four shorts at +/-w and "
            "+/-2w, wings at +/-4w, all puts, one post-event expiry."
        ),
        legs=(
            LegSpec("atm", PUT, BUY, expiry, atm, qty=2.0),
            LegSpec("up1", PUT, SELL, expiry,
                    StrikeSelector("grid_step", ref="atm", steps=int(steps))),
            LegSpec("up2", PUT, SELL, expiry,
                    StrikeSelector("mirror", ref="atm", about="up1")),
            LegSpec("up4", PUT, BUY, expiry,
                    StrikeSelector("mirror", ref="atm", about="up2")),
            LegSpec("dn1", PUT, SELL, expiry,
                    StrikeSelector("mirror", ref="up1", about="atm")),
            LegSpec("dn2", PUT, SELL, expiry,
                    StrikeSelector("mirror", ref="up2", about="atm")),
            LegSpec("dn4", PUT, BUY, expiry,
                    StrikeSelector("mirror", ref="up4", about="atm")),
        ),
        entry_offset=entry_offset,
        exit_offset=exit_offset,
        decision_offset=decision_offset,
        params={"steps": int(steps)},
    )


#: Factories keyed by strategy code, so specs can name a structure as a string.
STRUCTURES = {
    "CAL-P": put_calendar,
    "STR-THRU": straddle_through,
    "STR-RUNUP": straddle_runup,
    "CND-P": put_condor,
    "TWIN-P": twin_peak,
}
