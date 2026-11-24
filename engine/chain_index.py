"""Chain-index twin of :func:`engine.structures.price_structure`.

:func:`price_structure_indexed` repeats the ``price_structure`` leg loop. Expiry selection is
answered from a per-snapshot numpy index of the distinct expiries (dte plus memoised rule
results). Strike selection and the (expiry, right, strike) contract lookup are answered from
cached numpy arrays of the strike and right columns, grouped by expiry and right. The selection
rules are copied verbatim from :class:`engine.structures.StrikeSelector`; a case the arrays do
not cover falls back to the legacy DataFrame path for that leg. bid, ask, dte and delta are read
live from the snapshot, never cached. Pricing, the ``LadderTooCoarse`` check and the messages are
the same code, so results are bit-identical. The index is built lazily, lives with its snapshot,
and calls or snapshots outside the proven shape go whole to ``price_structure``. The checksums
cover expiry, dte, strike, right and index uniqueness on every call; a change raises
:class:`ChainSnapshotMutated`. ``ENGINE_CHAIN_INDEX_VERIFY=0`` disables the check.
"""
from __future__ import annotations

import os
import threading
import weakref
import zlib
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

__all__ = ["ChainSnapshotMutated", "price_structure_indexed"]

_VERIFY_ENV = "ENGINE_CHAIN_INDEX_VERIFY"
_NUM = (int, float, np.integer, np.floating)
_OPPOSITE = {BUY: SELL, SELL: BUY}


class ChainSnapshotMutated(RuntimeError):
    """The snapshot's rows changed after its index was built (verify mode)."""


def _ns(value: Any) -> int:
    if not (isinstance(value, pd.Timestamp) and value.tz is None):
        raise TypeError("expected a tz-naive pd.Timestamp")
    return int(value.value)


def _num(value: Any) -> bool:
    return isinstance(value, _NUM) and bool(np.isfinite(value))


def _checksums(rows: pd.DataFrame) -> tuple[Any, ...]:
    expiry = rows["expiry"].to_numpy().astype("datetime64[ns]").view(np.int64)
    return (
        zlib.crc32(expiry.tobytes()),
        zlib.crc32(rows["dte"].to_numpy().tobytes()),
        zlib.crc32(rows["strike"].to_numpy().tobytes()),
        zlib.crc32(pd.util.hash_array(rows["right"].to_numpy()).tobytes()),
        bool(rows.index.is_unique),
    )


class _Fallback(Exception):
    """Internal: this leg's strike selection must run the legacy DataFrame path."""


class _Group:
    """Rows of one (expiry, right) group in row order, with lazily built derived arrays."""

    def __init__(self, pos: np.ndarray, st: np.ndarray) -> None:
        self.pos = pos  # row positions, ascending
        self.st = st  # strike at each position
        self._grid: np.ndarray | None = None
        self._sorted: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def grid(self) -> np.ndarray:
        if self._grid is None:
            grid = np.sort(np.unique(self.st.astype(float)))
            grid.flags.writeable = False
            self._grid = grid
        return self._grid

    def ordered(self) -> tuple[np.ndarray, np.ndarray]:
        if self._sorted is None:
            order = np.argsort(self.st, kind="stable")
            self._sorted = (order, self.st[order])
        return self._sorted


class _StrikeData:
    """Strike and right arrays of one snapshot. Raises if the columns are not the proven shape."""

    def __init__(self, rows: pd.DataFrame, inv: np.ndarray) -> None:
        if not rows.index.is_unique:
            raise ValueError("chain index labels are not unique")
        for col in ("strike", "right", "bid", "ask", "dte"):
            if col not in rows.columns:
                raise KeyError(col)
        sdtype = rows["strike"].dtype
        if not (isinstance(sdtype, np.dtype) and sdtype.name in ("float64", "int64")):
            raise TypeError(f"column 'strike' has unsupported dtype {sdtype}")
        self.strike = rows["strike"].to_numpy()
        if not np.isfinite(self.strike).all():
            raise ValueError("non-finite strike")
        if pd.api.types.infer_dtype(rows["right"], skipna=False) != "string":
            raise TypeError("column 'right' must hold strings only")
        for col in ("bid", "ask"):
            if rows[col].dtype.kind not in "iuf":
                raise TypeError(f"column {col!r} has unsupported dtype {rows[col].dtype}")
        rvals, rinv = np.unique(rows["right"].to_numpy(), return_inverse=True)
        self.rcode = rinv.reshape(-1)
        self.rights = {str(v): i for i, v in enumerate(rvals)}
        self.rows = rows
        self.inv = inv
        self.groups: dict[tuple[int, int | None], _Group] = {}

    def group(self, e: int, r: int | None) -> _Group:
        key = (e, r)
        grp = self.groups.get(key)
        if grp is None:
            mask = self.inv == e
            if r is not None:
                mask &= self.rcode == r
            pos = np.flatnonzero(mask)
            grp = self.groups[key] = _Group(pos, self.strike[pos])
        return grp

    @staticmethod
    def find(grp: _Group, strike: Any) -> int | None:
        """Local index of the first row whose strike ``np.isclose``-matches ``strike``."""
        if not (isinstance(strike, _NUM) and np.isfinite(strike)):
            hit = np.flatnonzero(np.isclose(grp.st, strike))
            return int(hit[0]) if hit.size else None
        order, sorted_st = grp.ordered()
        tol = 1e-8 + 1e-5 * abs(strike)
        lo = np.searchsorted(sorted_st, strike - 2 * tol, "left")
        hi = np.searchsorted(sorted_st, strike + 2 * tol, "right")
        cand = order[lo:hi][np.isclose(sorted_st[lo:hi], strike)]
        return int(cand.min()) if cand.size else None

    def lookup(self, e: int, right: str, strike: Any) -> int | None:
        """Row position of the (expiry, right, strike) contract, or None."""
        rc = self.rights.get(right)
        if rc is None:
            return None
        grp = self.group(e, rc)
        loc = self.find(grp, strike)
        return None if loc is None else int(grp.pos[loc])

    def select_strike(self, sel: StrikeSelector, e: int, right: str, spot: float,
                      resolved: dict[str, float]) -> float:
        """Mirror of :meth:`StrikeSelector.select` on the cached arrays."""
        grp = self.group(e, None)
        if grp.pos.size == 0:
            raise StructureError("no chain rows at the selected expiry")
        rc = self.rights.get(right)
        if rc is not None:
            same_right = self.group(e, rc)
            if same_right.pos.size:
                grp = same_right
        grid, st = grp.grid, grp.st

        if sel.kind == "bracket":
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
            grid = grid[grid > anchor] if sel.moneyness > 0 else grid[grid < anchor]
            if grid.size == 0:
                side = "above" if sel.moneyness > 0 else "below"
                raise StructureError(
                    f"no listed strike {side} {anchor:.4f} at this expiry"
                )
            target = anchor + sel.moneyness * spot
            return float(grid[np.abs(grid - target).argmin()])

        if sel.kind == "mirror":
            for name in (sel.ref, sel.about):
                if name not in resolved:
                    raise StructureError(
                        f"leg {name!r} must resolve before a mirror reference to it"
                    )
            target = 2.0 * resolved[sel.about] - resolved[sel.ref]
            if not np.isclose(grid, target).any():
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
            if not np.isclose(st, target).any():
                raise StructureError(
                    f"strike {target} from leg {sel.ref!r} absent at this expiry"
                )
            return float(target)

        if sel.kind == "fixed":
            if not np.isclose(st, sel.strike).any():
                raise StructureError(f"strike {sel.strike} absent at this expiry")
            return float(sel.strike)

        if sel.kind == "delta":
            if "delta" not in self.rows.columns:
                raise StructureError("delta selector needs a delta column")
            dcol = self.rows["delta"].to_numpy()
            if dcol.dtype != np.float64:
                raise _Fallback
            dv = dcol[grp.pos]
            if np.isnan(dv).all():
                raise StructureError("delta selector needs a delta column")
            gap = np.abs(np.abs(dv) - abs(sel.target_delta))
            return float(st[int(np.nanargmin(gap))])

        target = spot if sel.kind == "atm" else sel.moneyness * spot
        gap = np.abs(st - target)
        return float(st[int(np.argmin(gap))])


class _ChainIndex:
    """Distinct expiries of one snapshot. Raises if the snapshot is not indexable."""

    def __init__(self, snapshot: ChainSnapshot) -> None:
        rows = snapshot.rows
        if rows.empty:
            raise ValueError("chain is empty")
        exp = rows["expiry"]
        if not (isinstance(exp.dtype, np.dtype) and exp.dtype.kind == "M") or exp.isna().any():
            raise TypeError("expiry must be a datetime64 column without NaT")
        raw = exp.to_numpy()
        ns = raw.astype("datetime64[ns]")
        if not np.array_equal(ns.astype(raw.dtype), raw):
            raise OverflowError("expiry outside the nanosecond range")
        dte = rows["dte"].to_numpy()
        if dte.dtype.kind not in "if":
            raise TypeError(f"column 'dte' has unsupported dtype {dte.dtype}")
        if dte.dtype.kind == "f" and np.isnan(dte).any():
            raise ValueError("NaN dte")
        _ns(snapshot.event_date)

        self.ex_ns = ns.view(np.int64)
        self.uex, first, inv = np.unique(self.ex_ns, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        if not np.array_equal(dte, dte[first][inv]):
            raise ValueError("an expiry carries more than one dte")
        self.udte = dte[first]
        if np.unique(self.udte).size != self.udte.size:
            raise ValueError("two expiries share a dte")
        self.ts = [pd.Timestamp(rows["expiry"].iloc[int(i)]) for i in first]
        self.inv = inv
        self.crcs = _checksums(rows)
        self._memo: dict[str, tuple[str, Any]] = {}
        try:
            self.sd: _StrikeData | None = _StrikeData(rows, inv)
        except Exception:
            self.sd = None  # every leg then takes the legacy path

    def expiry_pos(self, expiry: pd.Timestamp) -> int | None:
        ns = _ns(expiry)
        i = int(np.searchsorted(self.uex, ns))
        return i if i < self.uex.size and int(self.uex[i]) == ns else None

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
            if not np.any(self.uex == _ns(wanted)):
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
                        "(an expiry on the event date dies at the close, before the announcement)"
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


def _call_eligible(structure: Structure, snapshot: ChainSnapshot, pin: Any = None) -> bool:
    try:
        if snapshot.session is not None and not isinstance(snapshot.session, str):
            return False
        if not _num(snapshot.spot_price):
            return False
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
        for leg in pin or ():
            _ns(leg.expiry)
        return True
    except Exception:
        return False


def _select_index(structure: Structure, snapshot: ChainSnapshot,
                  pin: Any = None) -> _ChainIndex | None:
    try:
        index = _index_for(snapshot)
        if index is None or not _call_eligible(structure, snapshot, pin):
            return None
    except Exception:
        return None
    if os.environ.get(_VERIFY_ENV) != "0":
        try:
            current = _checksums(snapshot.rows)
        except Exception:
            current = None
        if current != index.crcs:
            raise ChainSnapshotMutated("chain snapshot rows changed after the index was built")
    return index


def price_structure_indexed(structure: Structure, snapshot: ChainSnapshot, fill: FillModel, *,
                            pin: Any = None, closing: bool = False) -> StructurePrice:
    """Equal to :func:`price_structure`; expiry, strike and contract lookup come from the index."""
    index = _select_index(structure, snapshot, pin)
    if index is None:
        return price_structure(structure, snapshot, fill, pin=pin, closing=closing)

    spot = snapshot.spot_price
    sd = index.sd
    rows = snapshot.rows
    pinned = {leg.name: leg for leg in (pin or ())}
    resolved_strikes: dict[str, float] = {}
    out: list[ResolvedLeg] = []

    for spec in structure.legs:
        fast = sd is not None  # strike selection and lookup from the cached arrays
        if spec.name in pinned:
            expiry = pinned[spec.name].expiry
            strike = pinned[spec.name].strike
            if fast:
                e = index.expiry_pos(expiry)
                present = e is not None
            else:
                at_expiry = rows[rows["expiry"] == expiry]
                present = not at_expiry.empty
            if not present:
                raise StructureError(
                    f"{spec.name}: pinned expiry {expiry.date()} absent from the "
                    f"{snapshot.obs_date.date()} chain"
                )
        else:
            expiry = index.select_expiry(spec.expiry, snapshot.event_date, snapshot.session)
            e = index.expiry_pos(expiry) if fast else None
            try:
                if e is None:
                    raise _Fallback
                strike = sd.select_strike(spec.strike, e, spec.right, spot, resolved_strikes)
            except _Fallback:
                fast = False
                at_expiry = rows[rows["expiry"] == expiry]
                strike = spec.strike.select(at_expiry, spot, resolved_strikes, right=spec.right)

        resolved_strikes[spec.name] = strike
        if fast:
            p = sd.lookup(e, spec.right, strike)
            found = p is not None
            if found:
                bid, ask = float(rows["bid"].to_numpy()[p]), float(rows["ask"].to_numpy()[p])
                dte = int(rows["dte"].to_numpy()[p])
        else:
            hit = at_expiry[
                np.isclose(at_expiry["strike"], strike) & (at_expiry["right"] == spec.right)
            ]
            found = not hit.empty
            if found:
                row = hit.iloc[0]
                bid, ask, dte = float(row["bid"]), float(row["ask"]), int(row["dte"])
        if not found:
            raise StructureError(
                f"{spec.name}: no {spec.right} at strike {strike} expiry "
                f"{expiry.date()} on {snapshot.obs_date.date()}"
            )
        side = _OPPOSITE[spec.side] if closing else spec.side
        out.append(
            ResolvedLeg(
                name=spec.name,
                right=spec.right,
                side=side,
                qty=spec.qty,
                expiry=pd.Timestamp(expiry),
                strike=float(strike),
                dte=dte,
                bid=bid,
                ask=ask,
                price=float(fill.price(side, bid, ask)),
                cash_flow=float(fill.cash_flow(side, bid, ask, spec.qty)),
                wide_market=bool(FillModel.is_wide(bid, ask)),
            )
        )

    # Two legs on the same contract are not a structure (see price_structure).
    contracts: dict[tuple[str, float, pd.Timestamp], list[str]] = {}
    for leg in out:
        contracts.setdefault((leg.right, leg.strike, leg.expiry), []).append(leg.name)
    collided = sorted((names, key) for key, names in contracts.items() if len(names) > 1)
    if collided:
        detail = "; ".join(
            f"{' and '.join(names)} both resolve to {right} {strike:g} "
            f"{pd.Timestamp(expiry).date()}"
            for names, (right, strike, expiry) in collided
        )
        raise LadderTooCoarse(
            f"{structure.name}: the listed strikes are too coarse for this shape — {detail}"
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
