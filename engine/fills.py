"""The fill model as a first-class object.

Execution quality is the program's headline risk: worst-case fills (buy the
ask, sell the bid) turn every earnings-vol exposure negative, and mid fills
turn all three positive. That means no function anywhere may quietly assume a
fill convention — every P&L computation takes an explicit :class:`FillModel`,
and every result is reported at worst / mid / best plus the breakeven alpha.

``alpha`` interpolates linearly from worst (0) to best (1) execution::

    buy(bid, ask)  = ask - alpha * (ask - bid)     # alpha=0 → ask,  alpha=1 → bid
    sell(bid, ask) = bid + alpha * (ask - bid)     # alpha=0 → bid,  alpha=1 → ask

so ``alpha = 0.5`` is the mid on both sides, and ``alpha`` is directly readable
as "the fraction of the spread we capture".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Above this relative spread a quote is flagged rather than trusted. Half the
#: mid is already a market you cannot reliably work a limit order into.
WIDE_MARKET_RATIO = 0.5

#: Entry cost above this fraction of spot is a BAD_QUOTE, not a price: no real
#: straddle costs more than a third of the stock. EXP-117 measured the
#: historical distribution (p95 18.6%, p99 53.4%, max 169%) and found 0.7% of
#: STR-THRU entries above the ceiling — junk-quote small caps, rising with
#: time, exactly the class of the live CBAT 2026-08-31 row at 166.7% that
#: WIDE_MARKET flagged but did not remove. Registered in EXP-117 spec.yaml
#: before the historical rate was measured.
BAD_QUOTE_COST_PCT = 30.0

__all__ = [
    "FillModel",
    "WORST",
    "MID",
    "BEST",
    "WIDE_MARKET_RATIO",
    "BAD_QUOTE_COST_PCT",
    "breakeven_alpha",
    "ALPHA_SWEEP",
]


def _validate(bid, ask):
    """Reject quote rows that should never have reached a pricing path.

    A crossed (``bid > ask``) or negative quote means a bad chain row got past
    ingestion validation. Pricing it would silently produce a plausible-looking
    number, so this raises instead — the loud failure is the point.

    A zero bid against a positive ask is *not* an error: it is a real, common
    market state for deep-OTM and near-expiry options. It prices normally and
    is surfaced through :meth:`FillModel.is_wide`.
    """
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    if bid.shape != ask.shape:
        raise ValueError(f"bid/ask shape mismatch: {bid.shape} vs {ask.shape}")
    if np.any(np.isnan(bid)) or np.any(np.isnan(ask)):
        raise ValueError("NaN in bid/ask — validation should have caught this upstream")
    if np.any(bid < 0):
        raise ValueError("negative bid in a quote reaching the pricing path")
    if np.any(ask < 0):
        raise ValueError("negative ask in a quote reaching the pricing path")
    if np.any(bid > ask):
        raise ValueError("crossed quote (bid > ask) reaching the pricing path")
    return bid, ask


def _unwrap(value, like):
    """Return a Python float when the inputs were scalars, else the array."""
    return float(value) if np.isscalar(like) or np.ndim(like) == 0 else value


@dataclass(frozen=True)
class FillModel:
    """Linear worst→best execution model. ``alpha`` in [0, 1]."""

    alpha: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha):
            raise ValueError(f"alpha must be finite, got {self.alpha!r}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {self.alpha!r}")

    # -- single-leg prices ------------------------------------------------

    def buy(self, bid, ask):
        """Price paid to open/close a long leg."""
        b, a = _validate(bid, ask)
        return _unwrap(a - self.alpha * (a - b), bid)

    def sell(self, bid, ask):
        """Price received to open/close a short leg."""
        b, a = _validate(bid, ask)
        return _unwrap(b + self.alpha * (a - b), bid)

    def price(self, side: str, bid, ask):
        """Dispatch on ``side`` ∈ {``"buy"``, ``"sell"``, ``"long"``, ``"short"``}."""
        s = side.lower()
        if s in ("buy", "long", "+1", "1"):
            return self.buy(bid, ask)
        if s in ("sell", "short", "-1"):
            return self.sell(bid, ask)
        raise ValueError(f"unknown side {side!r}")

    def cash_flow(self, side: str, bid, ask, qty: float = 1.0):
        """Signed cash flow of transacting ``qty`` contracts.

        Negative = cash out (a purchase), positive = cash in (a sale). Summing
        ``cash_flow`` over a structure's legs at entry and exit is the whole
        P&L calculation, which is why every structure reduces to leg lists.
        """
        s = side.lower()
        if s in ("buy", "long", "+1", "1"):
            return -qty * np.asarray(self.buy(bid, ask))
        if s in ("sell", "short", "-1"):
            return qty * np.asarray(self.sell(bid, ask))
        raise ValueError(f"unknown side {side!r}")

    # -- diagnostics ------------------------------------------------------

    @staticmethod
    def mid(bid, ask):
        b, a = _validate(bid, ask)
        return _unwrap((b + a) / 2.0, bid)

    @staticmethod
    def is_wide(bid, ask, ratio: float = WIDE_MARKET_RATIO):
        """``(ask - bid) / mid > ratio`` — a quote too wide to lean on.

        A zero mid (both sides zero) counts as wide: there is no market at all.
        """
        b, a = _validate(bid, ask)
        m = (b + a) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(m > 0, (a - b) / np.where(m > 0, m, 1.0), np.inf)
        wide = rel > ratio
        # A predicate returns a bool, not a float — `_unwrap` casts to float.
        return bool(wide) if np.isscalar(bid) or np.ndim(bid) == 0 else wide

    def __str__(self) -> str:  # pragma: no cover - display only
        label = {0.0: "worst", 0.5: "mid", 1.0: "best"}.get(self.alpha)
        return f"FillModel(alpha={self.alpha:g}{', ' + label if label else ''})"


#: The three conventions every result is reported at, side by side.
WORST = FillModel(0.0)
MID = FillModel(0.5)
BEST = FillModel(1.0)

#: Default grid for the fill-quality degradation curve (Phase 2 headline stat).
ALPHA_SWEEP = tuple(round(x, 2) for x in np.linspace(0.0, 1.0, 21))


def breakeven_alpha(pnl_at_worst: float, pnl_at_best: float) -> float | None:
    """Alpha at which a linearly-interpolated P&L crosses zero.

    P&L is linear in alpha for any fixed set of legs and quotes, so the two
    endpoint evaluations determine the whole curve. Returns ``None`` when the
    strategy never crosses zero inside [0, 1] — either always profitable
    (report it as ``0.0`` at the call site if that reads better) or never.

    This number is the margin of safety on the mid-fill assumption: how much
    worse than mid execution can get before the edge is gone.
    """
    lo, hi = float(pnl_at_worst), float(pnl_at_best)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        return None
    root = -lo / (hi - lo)
    return float(root) if 0.0 <= root <= 1.0 else None
