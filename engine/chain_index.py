"""Numpy-indexed twin of :func:`engine.structures.price_structure`.

The legacy path re-filters the pandas chain for every leg of every structure.
This module builds, once per :class:`ChainSnapshot`, a numpy index of the rows
(expiries as int64 nanoseconds, quotes as arrays, per-expiry row groups split
by right) and resolves legs against it. Results must be bit-identical to the
legacy code: same strikes, prices, cash flows and flags, the same exception
types and messages, and the same order in which failures are raised.

Eligibility is decided before any leg runs. A snapshot whose columns the index
cannot reproduce exactly, and any call with unusual selectors or pins, is sent
to the legacy code whole, never mixed with it. Ineligible snapshots are
remembered as ``None`` so they are not re-checked.

Indexes live in a module registry keyed by ``id(snapshot)`` and are dropped by
a :func:`weakref.finalize` when the snapshot dies; an index never holds a strong
reference to its snapshot. ``ENGINE_CHAIN_INDEX_VERIFY=1`` re-checksums every
column the index read at build time on each indexed call, and raises
:class:`ChainSnapshotMutated` if the rows were changed in place.

Resolution and pricing are split: :func:`resolve_structure` yields a
:class:`Resolution` (contracts and quotes, no fill convention) and
:func:`price_resolution` prices it at any fill model or direction, so one
selection can be priced at several alphas without re-running it.
"""
from __future__ import annotations

import os
import threading
import weakref
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from engine.fills import FillModel
from engine.structures import (
    BUY,
    SELL,
    ChainSnapshot,
    ExpirySelector,
    LadderTooCoarse,
    ResolvedLeg,
    StrikeSelector,
    Structure,
    StructureError,
    StructurePrice,
    price_structure,
)

__all__ = [
    "ChainSnapshotMutated",
    "ResolvedContract",
    "Resolution",
    "resolve_structure",
    "price_resolution",
    "price_structure_indexed",
]

_VERIFY_ENV = "ENGINE_CHAIN_INDEX_VERIFY"
_NUM = (int, float, np.integer, np.floating)
_OPPOSITE = {BUY: SELL, SELL: BUY}


class ChainSnapshotMutated(RuntimeError):
    """The snapshot's rows changed after its index was built (verify mode)."""


@dataclass(frozen=True)
class ResolvedContract:
    name: str
    right: str
    side: str  # the structure's OPENING side
    qty: float
    expiry: pd.Timestamp
    strike: float
    dte: int
    bid: float
    ask: float


@dataclass(frozen=True)
class Resolution:
    structure: str
    ticker: str
    obs_date: pd.Timestamp
    event_date: pd.Timestamp
    spot: float
    legs: tuple[ResolvedContract, ...]
    _snapshot_ref: Any = field(default=None, compare=False, repr=False)


def _ns(value: Any) -> int:
    if not (isinstance(value, pd.Timestamp) and value.tz is None):
        raise TypeError("expected a tz-naive pd.Timestamp")
    return int(value.value)


def _num(value: Any) -> bool:
    return isinstance(value, _NUM) and bool(np.isfinite(value))


def _arr(rows: pd.DataFrame, name: str, kinds: str) -> np.ndarray:
    arr = rows[name].to_numpy()
    if arr.dtype.kind not in kinds:
        raise TypeError(f"column {name!r} has unsupported dtype {arr.dtype}")
    return arr


def _checksums(rows: pd.DataFrame) -> tuple[int, ...]:
    cols = [
        rows["expiry"].to_numpy().astype("datetime64[ns]").view(np.int64),
        rows["strike"].to_numpy(),
        rows["bid"].to_numpy(),
        rows["ask"].to_numpy(),
        rows["dte"].to_numpy(),
    ]
    if "delta" in rows.columns:
        cols.append(rows["delta"].to_numpy())
    crcs = [zlib.crc32(col.tobytes()) for col in cols]
    right = "\x00".join(rows["right"].to_numpy(dtype=object)).encode()
    return (*crcs, zlib.crc32(right))


class _Group:
    """Rows of one expiry (optionally one right), in original row order."""

    __slots__ = ("strike", "bid", "ask", "dte", "delta", "_grid")

    def __init__(self, strike, bid, ask, dte, delta) -> None:
        self.strike, self.bid, self.ask, self.dte, self.delta = strike, bid, ask, dte, delta
        self._grid: np.ndarray | None = None

    @property
    def size(self) -> int:
        return int(self.strike.size)

    @property
    def grid(self) -> np.ndarray:
        if self._grid is None:
            self._grid = np.sort(np.unique(self.strike))
        return self._grid


class _ChainIndex:
    """Numpy view of one snapshot's rows. Raises if the snapshot is not indexable."""

    def __init__(self, snapshot: ChainSnapshot) -> None:
        rows = snapshot.rows
        if rows.empty or not rows.index.is_unique or not rows.columns.is_unique:
            raise ValueError("chain is empty or has duplicate labels")
        exp = rows["expiry"]
        if not (isinstance(exp.dtype, np.dtype) and exp.dtype.kind == "M") or exp.isna().any():
            raise TypeError("expiry must be a datetime64 column without NaT")
        raw = exp.to_numpy()
        ns = raw.astype("datetime64[ns]")
        if not np.array_equal(ns.astype(raw.dtype), raw):
            raise OverflowError("expiry outside the nanosecond range")
        ex_ns = ns.view(np.int64)
        strike = _arr(rows, "strike", "fi")
        if not np.isfinite(strike).all():
            raise ValueError("non-finite strike")
        dte = _arr(rows, "dte", "if")
        if dte.dtype.kind == "f" and np.isnan(dte).any():
            raise ValueError("NaN dte")
        right = rows["right"].to_numpy(dtype=object)
        if not all(isinstance(v, str) for v in right):
            raise TypeError("non-string right")
        _ns(snapshot.event_date)

        self.strike = strike.astype(np.float64)
        self.bid = _arr(rows, "bid", "fiu")
        self.ask = _arr(rows, "ask", "fiu")
        self.dte = dte
        self.delta = _arr(rows, "delta", "fiu") if "delta" in rows.columns else None
        self.right = right
        self.crcs = _checksums(rows)

        uex, inv = np.unique(ex_ns, return_inverse=True)
        inv = inv.reshape(-1)
        order = np.argsort(inv, kind="stable")
        pos = np.split(order, np.cumsum(np.bincount(inv, minlength=uex.size))[:-1])
        for p in pos:
            if np.unique(dte[p]).size != 1:
                raise ValueError("an expiry carries more than one dte")
        self.uex = uex
        self.pos = pos
        self.ts = [pd.Timestamp(rows["expiry"].iloc[int(p[0])]) for p in pos]
        self.udte = dte[[int(p[0]) for p in pos]]
        if np.unique(self.udte).size != self.udte.size:
            raise ValueError("two expiries share a dte")
        self._memo: dict[str, tuple[str, Any]] = {}
        self._groups: dict[tuple[int, str | None], _Group] = {}

    # -- groups and expiries -------------------------------------------------

    def group(self, k: int, right: str | None) -> _Group:
        key = (k, right)
        g = self._groups.get(key)
        if g is None:
            p = self.pos[k]
            if right is not None:
                p = p[self.right[p] == right]
            g = _Group(
                self.strike[p], self.bid[p], self.ask[p], self.dte[p],
                None if self.delta is None else self.delta[p],
            )
            self._groups[key] = g
        return g

    def k_of(self, expiry: pd.Timestamp) -> int | None:
        ns = _ns(pd.Timestamp(expiry))
        k = int(np.searchsorted(self.uex, ns))
        return k if k < self.uex.size and int(self.uex[k]) == ns else None

    def select_expiry(self, sel: ExpirySelector, event_date: pd.Timestamp,
                      session: str | None) -> pd.Timestamp:
        key = repr(sel)
        hit = self._memo.get(key)
        if hit is None:
            try:
                hit = ("ok", self._expiry_rule(sel, event_date, session))
            except StructureError as exc:
                hit = ("err", str(exc))
            self._memo[key] = hit
        if hit[0] == "err":
            raise StructureError(hit[1])
        return hit[1]

    def _expiry_rule(self, sel: ExpirySelector, event_date: pd.Timestamp,
                     session: str | None) -> pd.Timestamp:
        if sel.kind == "fixed":
            wanted = pd.Timestamp(sel.expiry)
            if self.k_of(wanted) is None:
                raise StructureError(f"expiry {wanted.date()} absent from this chain")
            return wanted
        mask = np.ones(self.uex.size, dtype=bool)
        if sel.min_dte is not None:
            mask &= self.udte >= sel.min_dte
        if sel.max_dte is not None:
            mask &= self.udte <= sel.max_dte
        if not mask.any():
            raise StructureError(f"no expiry survives {sel}")
        if sel.kind == "first_post_event":
            ev = _ns(event_date)
            if str(session).upper() == "AMC":
                mask &= self.uex > ev
                if not mask.any():
                    raise StructureError(
                        f"no expiry after the AMC event date {event_date.date()} "
                        "(an expiry on the event date dies at the close, before "
                        "the announcement)"
                    )
            else:
                mask &= self.uex >= ev
                if not mask.any():
                    raise StructureError(
                        f"no expiry on or after the event date {event_date.date()}"
                    )
        elif sel.kind == "first_dte_at_least":
            mask &= self.udte >= sel.target_dte
            if not mask.any():
                raise StructureError(f"no expiry with dte >= {sel.target_dte}")
        else:  # nearest_dte: smallest gap, ties toward the longer-dated expiry
            idx = np.flatnonzero(mask)
            dte = self.udte[idx]
            gap = np.abs(dte - sel.target_dte)
            return self.ts[int(idx[np.lexsort((-dte, gap))[0]])]
        return self.ts[int(np.flatnonzero(mask)[0])]

    # -- strikes ---------------------------------------------------------------

    def select_strike(self, sel: StrikeSelector, k: int, spot: float,
                      resolved: dict[str, float], right: str | None) -> float:
        g = self.group(k, None)
        if g.size == 0:
            raise StructureError("no chain rows at the selected expiry")
        if right is not None:
            same_right = self.group(k, right)
            if same_right.size:
                g = same_right

        if sel.kind == "bracket":
            grid = g.grid
            side = grid[grid <= spot] if sel.side == "below" else grid[grid > spot]
            if side.size == 0:
                raise StructureError(
                    f"no listed strike {sel.side} spot {spot:.4f} at this expiry"
                )
            anchor = float(side[-1] if sel.side == "below" else side[0])
            if not sel.steps:
                return anchor
            hit = np.flatnonzero(np.isclose(grid, anchor))
            target = int(hit[0]) + int(sel.steps)
            if not 0 <= target < grid.size:
                raise StructureError(
                    f"{sel.steps:+d} steps from the bracketing strike {anchor:.4f} "
                    f"runs off the ladder ({grid.size} listed strikes)"
                )
            return float(grid[target])

        if sel.kind == "grid_step":
            if sel.ref not in resolved:
                raise StructureError(
                    f"leg {sel.ref!r} must resolve before a grid_step reference to it"
                )
            grid = g.grid
            hit = np.flatnonzero(np.isclose(grid, resolved[sel.ref]))
            if hit.size == 0:
                raise StructureError(
                    f"anchor strike {resolved[sel.ref]:.4f} is not on this expiry's grid"
                )
            target = int(hit[0]) + int(sel.steps)
            if not 0 <= target < grid.size:
                raise StructureError(
                    f"{sel.steps:+d} grid steps from {resolved[sel.ref]:.4f} runs off "
                    f"the ladder ({grid.size} listed strikes)"
                )
            return float(grid[target])

        if sel.kind == "offset_from":
            if sel.ref not in resolved:
                raise StructureError(
                    f"leg {sel.ref!r} must resolve before an offset_from reference to it"
                )
            anchor = resolved[sel.ref]
            grid = g.grid
            grid = grid[grid > anchor] if sel.moneyness > 0 else grid[grid < anchor]
            if grid.size == 0:
                side = "above" if sel.moneyness > 0 else "below"
                raise StructureError(f"no listed strike {side} {anchor:.4f} at this expiry")
            target = anchor + sel.moneyness * spot
            return float(grid[np.abs(grid - target).argmin()])

        if sel.kind == "mirror":
            for name in (sel.ref, sel.about):
                if name not in resolved:
                    raise StructureError(
                        f"leg {name!r} must resolve before a mirror reference to it"
                    )
            target = 2.0 * resolved[sel.about] - resolved[sel.ref]
            if not np.isclose(g.grid, target).any():
                raise StructureError(
                    f"mirrored strike {target:.4f} "
                    f"(2x{resolved[sel.about]:.4f} - {resolved[sel.ref]:.4f}) "
                    "is not listed at this expiry"
                )
            return float(target)

        if sel.kind == "same_as":
            if sel.ref not in resolved:
                raise StructureError(
                    f"leg {sel.ref!r} must resolve before a same_as reference to it"
                )
            target = resolved[sel.ref]
            if not np.isclose(g.strike, target).any():
                raise StructureError(
                    f"strike {target} from leg {sel.ref!r} absent at this expiry"
                )
            return float(target)

        if sel.kind == "fixed":
            if not np.isclose(g.strike, sel.strike).any():
                raise StructureError(f"strike {sel.strike} absent at this expiry")
            return float(sel.strike)

        if sel.kind == "delta":
            if g.delta is None or np.isnan(g.delta).all():
                raise StructureError("delta selector needs a delta column")
            gap = np.abs(np.abs(g.delta) - abs(sel.target_delta))
            return float(g.strike[np.nanargmin(gap)])

        target = spot if sel.kind == "atm" else sel.moneyness * spot
        gap = np.abs(g.strike - target)
        return float(g.strike[gap.argmin()])

    # -- legs ------------------------------------------------------------------

    def resolve(self, structure: Structure, snapshot: ChainSnapshot,
                pin_legs: tuple | None, spot: float) -> Resolution:
        pinned = {leg.name: leg for leg in (pin_legs or ())}
        resolved_strikes: dict[str, float] = {}
        contracts: list[ResolvedContract] = []
        for spec in structure.legs:
            if spec.name in pinned:
                expiry = pinned[spec.name].expiry
                strike = pinned[spec.name].strike
                k = self.k_of(expiry)
                if k is None:
                    raise StructureError(
                        f"{spec.name}: pinned expiry {expiry.date()} absent from the "
                        f"{snapshot.obs_date.date()} chain"
                    )
            else:
                expiry = self.select_expiry(spec.expiry, snapshot.event_date, snapshot.session)
                k = self.k_of(expiry)
                strike = self.select_strike(spec.strike, k, spot, resolved_strikes, spec.right)
            resolved_strikes[spec.name] = strike

            g = self.group(k, spec.right)
            hit = np.flatnonzero(np.isclose(g.strike, strike))
            if hit.size == 0:
                raise StructureError(
                    f"{spec.name}: no {spec.right} at strike {strike} expiry "
                    f"{expiry.date()} on {snapshot.obs_date.date()}"
                )
            i = int(hit[0])
            bid, ask = float(g.bid[i]), float(g.ask[i])
            dte = int(g.dte[i])
            FillModel.mid(bid, ask)  # raises the legacy quote errors at this leg
            contracts.append(ResolvedContract(
                spec.name, spec.right, spec.side, spec.qty, pd.Timestamp(expiry),
                float(strike), dte, bid, ask,
            ))

        # Same collision rule as price_structure: two legs on one contract refused.
        seen: dict[tuple[str, float, pd.Timestamp], list[str]] = {}
        for leg in contracts:
            seen.setdefault((leg.right, leg.strike, leg.expiry), []).append(leg.name)
        collided = sorted((names, key) for key, names in seen.items() if len(names) > 1)
        if collided:
            detail = "; ".join(
                f"{' and '.join(names)} both resolve to {right} {strike:g} "
                f"{pd.Timestamp(expiry).date()}"
                for names, (right, strike, expiry) in collided
            )
            raise LadderTooCoarse(
                f"{structure.name}: the listed strikes are too coarse for this "
                f"shape — {detail}"
            )

        return Resolution(
            structure.name, snapshot.ticker, pd.Timestamp(snapshot.obs_date),
            pd.Timestamp(snapshot.event_date), spot, tuple(contracts), weakref.ref(snapshot),
        )


# --------------------------------------------------------------------------
# registry and eligibility
# --------------------------------------------------------------------------

_INDEXES: dict[int, tuple[weakref.ref, _ChainIndex | None]] = {}
_LOCK = threading.Lock()


def _drop(key: int, ref: weakref.ref) -> None:
    # Runs from a finalizer, possibly inside a lock held by this thread: no lock here.
    entry = _INDEXES.get(key)
    if entry is not None and entry[0] is ref:
        del _INDEXES[key]


def _index_for(snapshot: ChainSnapshot) -> _ChainIndex | None:
    key = id(snapshot)
    with _LOCK:
        entry = _INDEXES.get(key)
        if entry is not None and entry[0]() is snapshot:
            return entry[1]
        try:
            index: _ChainIndex | None = _ChainIndex(snapshot)
        except Exception:
            index = None
        ref = weakref.ref(snapshot)
        _INDEXES[key] = (ref, index)
        weakref.finalize(snapshot, _drop, key, ref)
        return index


def _call_eligible(structure: Structure, snapshot: ChainSnapshot,
                   pin_legs: tuple | None, spot: float) -> bool:
    try:
        if not np.isfinite(spot):
            return False
        if snapshot.session is not None and not isinstance(snapshot.session, str):
            return False
        for leg in pin_legs or ():
            if not isinstance(leg.strike, _NUM):
                return False
            _ns(leg.expiry)
            str(leg.name)
        for spec in structure.legs:
            if spec.expiry.kind == "fixed":
                _ns(pd.Timestamp(spec.expiry.expiry))
            for value in (spec.expiry.min_dte, spec.expiry.max_dte, spec.expiry.target_dte):
                if value is not None and not _num(value):
                    return False
            sel = spec.strike
            for value in (sel.moneyness, sel.target_delta, sel.strike, sel.steps):
                if value is not None and not _num(value):
                    return False
        return True
    except Exception:
        return False


def _select_index(structure: Structure, snapshot: ChainSnapshot,
                  pin_legs: tuple | None, spot: float) -> _ChainIndex | None:
    try:
        index = _index_for(snapshot)
        if index is None or not _call_eligible(structure, snapshot, pin_legs, spot):
            return None
    except Exception:
        return None
    if os.environ.get(_VERIFY_ENV) == "1":
        try:
            current = _checksums(snapshot.rows)
        except Exception:
            current = None
        if current != index.crcs:
            raise ChainSnapshotMutated("chain snapshot rows changed after the index was built")
    return index


def _pin_legs(structure: Structure, snapshot: ChainSnapshot, pin: Any) -> tuple | None:
    if isinstance(pin, Resolution):
        if pin.ticker != snapshot.ticker or pin.structure != structure.name:
            raise ValueError(
                f"pin resolution is for {pin.structure} on {pin.ticker}, "
                f"not {structure.name} on {snapshot.ticker}"
            )
        return tuple(pin.legs)
    return None if pin is None else tuple(pin)


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def resolve_structure(structure: Structure, snapshot: ChainSnapshot, *,
                      pin: Any = None) -> Resolution:
    """Locate every leg of ``structure`` on ``snapshot``; quotes validated, no pricing."""
    pin_legs = _pin_legs(structure, snapshot, pin)
    spot = snapshot.spot_price
    index = _select_index(structure, snapshot, pin_legs, spot)
    if index is not None:
        return index.resolve(structure, snapshot, pin_legs, spot)
    sp = price_structure(structure, snapshot, FillModel(0.5), pin=pin_legs)
    return Resolution(
        sp.structure, sp.ticker, sp.obs_date, sp.event_date, sp.spot,
        tuple(
            ResolvedContract(leg.name, leg.right, leg.side, leg.qty, leg.expiry,
                             leg.strike, leg.dte, leg.bid, leg.ask)
            for leg in sp.legs
        ),
        weakref.ref(snapshot),
    )


def price_resolution(resolution: Resolution, snapshot: ChainSnapshot, fill: FillModel, *,
                     closing: bool = False) -> StructurePrice:
    """Price a :class:`Resolution` at ``fill``; ``closing`` transacts the opposite sides."""
    ref = resolution._snapshot_ref
    if ref is None or ref() is not snapshot:
        raise ValueError("resolution was not resolved on this chain snapshot")
    legs = []
    for leg in resolution.legs:
        side = _OPPOSITE[leg.side] if closing else leg.side
        legs.append(ResolvedLeg(
            name=leg.name, right=leg.right, side=side, qty=leg.qty, expiry=leg.expiry,
            strike=leg.strike, dte=leg.dte, bid=leg.bid, ask=leg.ask,
            price=float(fill.price(side, leg.bid, leg.ask)),
            cash_flow=float(fill.cash_flow(side, leg.bid, leg.ask, leg.qty)),
            wide_market=bool(FillModel.is_wide(leg.bid, leg.ask)),
        ))
    return StructurePrice(
        structure=resolution.structure, ticker=resolution.ticker,
        obs_date=resolution.obs_date, event_date=resolution.event_date,
        spot=resolution.spot, alpha=fill.alpha, legs=tuple(legs), closing=closing,
    )


def price_structure_indexed(structure: Structure, snapshot: ChainSnapshot, fill: FillModel, *,
                            pin: Any = None, closing: bool = False) -> StructurePrice:
    """Equal to :func:`price_structure`; uses the chain index when the call is eligible."""
    pin_legs = _pin_legs(structure, snapshot, pin)
    spot = snapshot.spot_price
    index = _select_index(structure, snapshot, pin_legs, spot)
    if index is None:
        return price_structure(structure, snapshot, fill, pin=pin_legs, closing=closing)
    resolution = index.resolve(structure, snapshot, pin_legs, spot)
    return price_resolution(resolution, snapshot, fill, closing=closing)
