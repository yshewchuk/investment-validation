"""Chain-index twin of :func:`engine.structures.price_structure` (expiry rule only).

:func:`price_structure_indexed` repeats the ``price_structure`` leg loop; only expiry selection
differs: it is answered from a per-snapshot numpy index of the distinct expiries (dte plus
memoised rule results). Strike selection, contract lookup, pricing, the ``LadderTooCoarse``
check and the messages are the same code, so results are bit-identical. The index is built
lazily, lives with its snapshot, and calls or snapshots outside the proven shape go whole to
``price_structure``. The expiry and dte columns are checksummed on every call; a change raises
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


def _checksums(rows: pd.DataFrame) -> tuple[int, int]:
    expiry = rows["expiry"].to_numpy().astype("datetime64[ns]").view(np.int64)
    return zlib.crc32(expiry.tobytes()), zlib.crc32(rows["dte"].to_numpy().tobytes())


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
        self.crcs = _checksums(rows)
        self._memo: dict[str, tuple[str, Any]] = {}

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


def _call_eligible(structure: Structure, snapshot: ChainSnapshot) -> bool:
    try:
        if snapshot.session is not None and not isinstance(snapshot.session, str):
            return False
        for spec in structure.legs:
            if spec.expiry.kind == "fixed":
                _ns(pd.Timestamp(spec.expiry.expiry))
            for value in (spec.expiry.min_dte, spec.expiry.max_dte, spec.expiry.target_dte):
                if value is not None and not _num(value):
                    return False
        return True
    except Exception:
        return False


def _select_index(structure: Structure, snapshot: ChainSnapshot) -> _ChainIndex | None:
    try:
        index = _index_for(snapshot)
        if index is None or not _call_eligible(structure, snapshot):
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
    """Equal to :func:`price_structure`; expiry selection uses the chain index when eligible."""
    spot = snapshot.spot_price
    index = _select_index(structure, snapshot)
    if index is None:
        return price_structure(structure, snapshot, fill, pin=pin, closing=closing)

    rows = snapshot.rows
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
            expiry = index.select_expiry(spec.expiry, snapshot.event_date, snapshot.session)
            at_expiry = rows[rows["expiry"] == expiry]
            strike = spec.strike.select(at_expiry, spot, resolved_strikes, right=spec.right)

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
