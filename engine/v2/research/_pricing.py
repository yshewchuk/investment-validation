"""Pricing primitives replay needs, moved from legacy verbatim (bodies intact).

Supervisor decision (slice 7, OPEN 1): no legacy adapter and no entry in
``checks/legacy_adapters.json``. Every function this package needs from
``engine.structures`` / ``engine.fills`` / ``engine.calendar`` was checked for
an existing v2 port first; none exists for these names (``engine.v2.domain.
generation`` ports ``generate``/``price`` with a different API and is not
byte-compatible with ``price_structure``/``structure_return``, which the
slice's acceptance test requires). The needed code is therefore moved here
verbatim -- same names, same bodies -- so ``engine/v2/research/replay.py`` can
be byte-identical to ``engine/replay.py`` without importing a legacy module.

Deviations from a literal copy, all behavior-preserving:

* ``StrikeSelector.select`` is split into one private helper per selector kind
  so each function stays inside the §4.3 function-length/complexity budget;
  every branch and return value is unchanged.
* ``trading_calendar`` resolves the S&P daily series off the repo root
  (``INVESTING_PLAN_ROOT`` or this worktree) instead of importing
  ``engine.paths``; the parse body is unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# ==========================================================================
# engine/fills.py
# ==========================================================================

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

#: A structure's net cost below this floor is FLOATING-POINT ZERO, not a real
#: price. A multi-leg structure with offsetting long and short legs (CND-P: two
#: long, two short) can sum to a net debit that floating-point addition cannot
#: distinguish from zero at the extreme end of the fill grid, where the longs'
#: cheapest price and the shorts' richest price nearly cancel. EXP-121 found 65
#: of 18,388 CND-P best-fill (alpha=1.0) trades costing between 1e-17 and 2e-15
#: — pure summation noise, not a $0.0000000000000001 debit anyone could pay —
#: against a clean floor: the next real value above the noise band is exactly
#: $0.01. `ret = pnl / cost` on a noise-floor cost produced means in the
#: quadrillions of percent, which corrupted every headline statistic that
#: touches best fill. 1e-6 sits five orders of magnitude above the noise band
#: and five below the cheapest real price observed, so it cannot misclassify
#: either side.
MIN_MEANINGFUL_COST = 1e-6


def _validate(bid, ask):
    """Reject quote rows that should never have reached a pricing path."""
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
        """Signed cash flow of transacting ``qty`` contracts."""
        s = side.lower()
        if s in ("buy", "long", "+1", "1"):
            return -qty * np.asarray(self.buy(bid, ask))
        if s in ("sell", "short", "-1"):
            return qty * np.asarray(self.sell(bid, ask))
        raise ValueError(f"unknown side {side!r}")

    @staticmethod
    def mid(bid, ask):
        b, a = _validate(bid, ask)
        return _unwrap((b + a) / 2.0, bid)

    @staticmethod
    def is_wide(bid, ask, ratio: float = WIDE_MARKET_RATIO):
        """``(ask - bid) / mid > ratio`` — a quote too wide to lean on."""
        b, a = _validate(bid, ask)
        m = (b + a) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(m > 0, (a - b) / np.where(m > 0, m, 1.0), np.inf)
        wide = rel > ratio
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
    """Alpha at which a linearly-interpolated P&L crosses zero."""
    lo, hi = float(pnl_at_worst), float(pnl_at_best)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo == hi:
        return None
    root = -lo / (hi - lo)
    return float(root) if 0.0 <= root <= 1.0 else None


# ==========================================================================
# engine/calendar.py (the session-arithmetic half replay uses)
# ==========================================================================

BMO = "BMO"
AMC = "AMC"

BMO_CUTOFF = 1200

#: One-off NYSE closures that no holiday rule generates.
UNSCHEDULED_CLOSURES = {
    "2007-01-02",  # national day of mourning, President Ford
    "2012-10-29",  # Hurricane Sandy
    "2012-10-30",  # Hurricane Sandy
    "2018-12-05",  # national day of mourning, President G.H.W. Bush
    "2025-01-09",  # national day of mourning, President Carter
}


def _easter(year: int) -> pd.Timestamp:
    """Gregorian Easter Sunday (anonymous computus) — Good Friday is two days earlier."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lam = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lam) // 451
    month, day = divmod(h + lam - 7 * m + 114, 31)
    return pd.Timestamp(year=year, month=month, day=day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """``n``-th ``weekday`` (Mon=0) of a month; ``n=-1`` means the last one."""
    days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    days = days[(days.month == month) & (days.weekday == weekday)]
    return days[n if n < 0 else n - 1]


def _observed(date: pd.Timestamp) -> pd.Timestamp:
    """NYSE observation rule: Saturday → the Friday before, Sunday → the Monday after."""
    if date.weekday() == 5:
        return date - pd.Timedelta(days=1)
    if date.weekday() == 6:
        return date + pd.Timedelta(days=1)
    return date


def _observed_new_year(year: int) -> pd.Timestamp | None:
    """New Year's Day, which does *not* follow the Saturday→Friday rule."""
    day = pd.Timestamp(year=year, month=1, day=1)
    if day.weekday() == 5:
        return None
    return _observed(day)


def us_market_holidays(year: int) -> set[pd.Timestamp]:
    """The scheduled NYSE holidays for ``year``, with observation rules applied."""
    out = {
        _nth_weekday(year, 1, 0, 3),  # MLK Day (from 1998)
        _nth_weekday(year, 2, 0, 3),  # Presidents' Day
        _easter(year) - pd.Timedelta(days=2),  # Good Friday
        _nth_weekday(year, 5, 0, -1),  # Memorial Day
        _observed(pd.Timestamp(year=year, month=7, day=4)),
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(pd.Timestamp(year=year, month=12, day=25)),
    }
    new_year = _observed_new_year(year)
    if new_year is not None:
        out.add(new_year)
    if year >= 2022:  # Juneteenth became a market holiday in 2022
        out.add(_observed(pd.Timestamp(year=year, month=6, day=19)))
    return out


def projected_trading_days(start, end) -> pd.DatetimeIndex:
    """Weekdays in ``(start, end]`` that are not scheduled market holidays."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end <= start:
        return pd.DatetimeIndex([])
    days = pd.date_range(start + pd.Timedelta(days=1), end, freq="B")
    holidays: set[pd.Timestamp] = set()
    for year in range(start.year, end.year + 1):
        holidays |= us_market_holidays(year)
    return days[~days.isin(pd.DatetimeIndex(sorted(holidays)))]


@dataclass(frozen=True)
class PrintWindow:
    """The dates a structure actually trades on, for one event."""

    event_date: pd.Timestamp
    session: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    last_pre_print: pd.Timestamp
    first_post_print: pd.Timestamp
    #: The close whose information the trade is decided on. Defaults to
    #: ``entry_date`` — decide and enter at the same close.
    decision_date: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        if self.decision_date is None:
            object.__setattr__(self, "decision_date", self.entry_date)


class TradingCalendar:
    """Trading days, taken from the S&P 500 daily series (2006 → today)."""

    def __init__(self, days: Sequence[pd.Timestamp], *, observed_through=None):
        idx = pd.DatetimeIndex(pd.to_datetime(list(days))).normalize().unique().sort_values()
        if len(idx) == 0:
            raise ValueError("trading calendar cannot be empty")
        self.days = idx
        self._pos = {d: i for i, d in enumerate(idx)}
        #: Last day that came from real price history. Anything after it is
        #: rule-projected and could be wrong about an unscheduled closure.
        self.observed_through = (
            pd.Timestamp(observed_through).normalize() if observed_through is not None else idx[-1]
        )

    def is_projected(self, date) -> bool:
        return pd.Timestamp(date).normalize() > self.observed_through

    def __len__(self) -> int:
        return len(self.days)

    @property
    def first(self) -> pd.Timestamp:
        return self.days[0]

    @property
    def last(self) -> pd.Timestamp:
        return self.days[-1]

    def is_trading_day(self, date) -> bool:
        return pd.Timestamp(date).normalize() in self._pos

    def index_of(self, date, *, side: str = "exact") -> int:
        """Position of ``date``. ``side`` ∈ {exact, prev, next}."""
        d = pd.Timestamp(date).normalize()
        if side == "exact":
            if d not in self._pos:
                raise KeyError(f"{d.date()} is not a trading day")
            return self._pos[d]
        pos = int(self.days.searchsorted(d, side="left"))
        if side == "next":
            if pos >= len(self.days):
                raise KeyError(f"no trading day on or after {d.date()}")
            return pos
        if side == "prev":
            if pos < len(self.days) and self.days[pos] == d:
                return pos
            if pos == 0:
                raise KeyError(f"no trading day on or before {d.date()}")
            return pos - 1
        raise ValueError(f"unknown side {side!r}")

    def shift(self, date, n: int, *, side: str = "prev") -> pd.Timestamp:
        """``n`` trading days from ``date`` (negative = earlier)."""
        pos = self.index_of(date, side=side) + n
        if not 0 <= pos < len(self.days):
            raise KeyError(f"{n:+d} trading days from {pd.Timestamp(date).date()} is out of range")
        return self.days[pos]

    def last_pre_print(self, event_date, session: str) -> pd.Timestamp:
        """The last close that is strictly information-free about the print."""
        d = pd.Timestamp(event_date).normalize()
        if session == AMC:
            return self.days[self.index_of(d, side="prev")]
        if session == BMO:
            pos = self.index_of(d, side="prev")
            if self.days[pos] == d:
                pos -= 1
            if pos < 0:
                raise KeyError(f"no trading day before {d.date()}")
            return self.days[pos]
        raise ValueError(f"unknown session {session!r}")

    def first_post_print(self, event_date, session: str) -> pd.Timestamp:
        """The first close that already reflects the print."""
        d = pd.Timestamp(event_date).normalize()
        if session == BMO:
            return self.days[self.index_of(d, side="next")]
        if session == AMC:
            pos = self.index_of(d, side="next")
            if self.days[pos] == d:
                pos += 1
            if pos >= len(self.days):
                raise KeyError(f"no trading day after {d.date()}")
            return self.days[pos]
        raise ValueError(f"unknown session {session!r}")

    def resolve_offsets(
        self,
        event_date,
        session: str,
        entry_offset: int,
        exit_offset: int,
        decision_offset: int | None = None,
    ) -> PrintWindow:
        """Map a structure's ``(entry_offset, exit_offset)`` onto real dates."""
        pre = self.last_pre_print(event_date, session)
        post = self.first_post_print(event_date, session)

        def resolve(offset: int) -> pd.Timestamp:
            if offset <= 0:
                return self.shift(pre, offset)
            return self.shift(post, offset - 1)

        return PrintWindow(
            event_date=pd.Timestamp(event_date).normalize(),
            session=session,
            entry_date=resolve(entry_offset),
            exit_date=resolve(exit_offset),
            decision_date=(
                resolve(decision_offset) if decision_offset is not None else None
            ),
            last_pre_print=pre,
            first_post_print=post,
        )


def _repo_root() -> Path:
    """Repo root without importing legacy ``engine.paths`` (layer rule)."""
    env = os.environ.get("INVESTING_PLAN_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=2)
def trading_calendar(extend_days: int = 400) -> TradingCalendar:
    """Trading days from the cached S&P 500 daily series, extended forward.

    History comes from the index series; ``extend_days`` calendar days of
    rule-projected weekdays are appended so events past the end of the price
    history still resolve. The source path mirrors ``engine.paths.GSPC_DAILY``
    (``earnings_predictions/data/raw/polygon/gspc_daily.csv``) off this
    worktree's root, never the legacy module.
    """
    path = _repo_root() / "earnings_predictions" / "data" / "raw" / "polygon" / "gspc_daily.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — the trading calendar is derived from the S&P daily series"
        )
    # yfinance multi-header: row 0 is field names, rows 1-2 are ticker/blank.
    df = pd.read_csv(path, skiprows=3, header=None, usecols=[0], names=["date"])
    observed = pd.to_datetime(df["date"], errors="coerce").dropna()
    last = pd.Timestamp(observed.max()).normalize()
    future = projected_trading_days(last, last + pd.Timedelta(days=extend_days))
    return TradingCalendar(
        list(observed) + list(future), observed_through=last
    )


# ==========================================================================
# engine/structures.py
# ==========================================================================

CALL, PUT = "C", "P"
BUY, SELL = "buy", "sell"


class StructureError(Exception):
    """A structure could not be resolved against the given chain."""


class LadderTooCoarse(StructureError):
    """Two legs of one structure resolved onto the same contract."""


@dataclass(frozen=True)
class ExpirySelector:
    """How a leg picks its expiry out of the expiries present in a chain."""

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
            return self._first_post_event(exp, event_date, session)
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

    @staticmethod
    def _first_post_event(exp: pd.DataFrame, event_date, session) -> pd.Timestamp:
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


@dataclass(frozen=True)
class StrikeSelector:
    """How a leg picks its strike once the expiry is fixed."""

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
            if not same_right.empty:
                rows = same_right
        if self.kind == "bracket":
            return self._select_bracket(rows, spot)
        if self.kind == "grid_step":
            return self._select_grid_step(rows, resolved)
        if self.kind == "offset_from":
            return self._select_offset_from(rows, spot, resolved)
        if self.kind == "mirror":
            return self._select_mirror(rows, resolved)
        if self.kind == "same_as":
            return self._select_same_as(rows, resolved)
        if self.kind == "fixed":
            return self._select_fixed(rows)
        if self.kind == "delta":
            return self._select_delta(rows)
        target = spot if self.kind == "atm" else self.moneyness * spot
        gap = (rows["strike"] - target).abs()
        return float(rows.loc[gap.idxmin(), "strike"])

    def _select_bracket(self, rows: pd.DataFrame, spot: float) -> float:
        grid = self._grid(rows)
        side = grid[grid <= spot] if self.side == "below" else grid[grid > spot]
        if side.size == 0:
            raise StructureError(
                f"no listed strike {self.side} spot {spot:.4f} at this expiry"
            )
        anchor = float(side[-1] if self.side == "below" else side[0])
        if not self.steps:
            return anchor
        hit = np.flatnonzero(np.isclose(grid, anchor))
        target = int(hit[0]) + int(self.steps)
        if not 0 <= target < grid.size:
            raise StructureError(
                f"{self.steps:+d} steps from the bracketing strike {anchor:.4f} "
                f"runs off the ladder ({grid.size} listed strikes)"
            )
        return float(grid[target])

    def _select_grid_step(self, rows: pd.DataFrame, resolved: dict[str, float]) -> float:
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

    def _select_offset_from(self, rows: pd.DataFrame, spot: float,
                            resolved: dict[str, float]) -> float:
        if self.ref not in resolved:
            raise StructureError(
                f"leg {self.ref!r} must resolve before an offset_from reference to it"
            )
        anchor = resolved[self.ref]
        grid = self._grid(rows)
        grid = grid[grid > anchor] if self.moneyness > 0 else grid[grid < anchor]
        if grid.size == 0:
            side = "above" if self.moneyness > 0 else "below"
            raise StructureError(f"no listed strike {side} {anchor:.4f} at this expiry")
        target = anchor + self.moneyness * spot
        return float(grid[np.abs(grid - target).argmin()])

    def _select_mirror(self, rows: pd.DataFrame, resolved: dict[str, float]) -> float:
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

    def _select_same_as(self, rows: pd.DataFrame, resolved: dict[str, float]) -> float:
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

    def _select_fixed(self, rows: pd.DataFrame) -> float:
        hit = rows[np.isclose(rows["strike"], self.strike)]
        if hit.empty:
            raise StructureError(f"strike {self.strike} absent at this expiry")
        return float(self.strike)

    def _select_delta(self, rows: pd.DataFrame) -> float:
        if "delta" not in rows or rows["delta"].isna().all():
            raise StructureError("delta selector needs a delta column")
        gap = (rows["delta"].abs() - abs(self.target_delta)).abs()
        return float(rows.loc[gap.idxmin(), "strike"])


@dataclass(frozen=True)
class LegSpec:
    name: str
    right: str
    side: str
    expiry: ExpirySelector
    strike: StrikeSelector = field(default_factory=StrikeSelector)
    qty: float = 1.0

    def __post_init__(self) -> None:
        if self.right not in (CALL, PUT):
            raise ValueError(f"right must be {CALL!r} or {PUT!r}, got {self.right!r}")
        if self.side not in (BUY, SELL):
            raise ValueError(f"side must be {BUY!r} or {SELL!r}, got {self.side!r}")
        if self.qty < 0:
            raise ValueError("qty must not be negative; direction is carried by `side`")


@dataclass(frozen=True)
class Structure:
    """A named, serializable trade structure."""

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
        """The offset the trade is decided on — ``entry_offset`` when unset."""
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


@dataclass(frozen=True)
class ChainSnapshot:
    """One (ticker, obs_date) option chain, plus the event it is traded around."""

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
    """The priced structure at one point in time."""

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


def _resolve_leg(spec, snapshot, pinned, resolved_strikes):
    """Resolve one leg's expiry/strike/quote — extracted from price_structure."""
    rows = snapshot.rows
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
            at_expiry, snapshot.spot_price, resolved_strikes, right=spec.right
        )
    return expiry, strike, at_expiry


def _check_ladder_collisions(structure, legs) -> None:
    """Refuse when two distinct legs resolve onto the same contract."""
    contracts: dict[tuple[str, float, pd.Timestamp], list[str]] = {}
    for leg in legs:
        contracts.setdefault((leg.right, leg.strike, leg.expiry), []).append(leg.name)
    collided = sorted(
        (names, key) for key, names in contracts.items() if len(names) > 1
    )
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


def price_structure(
    structure: Structure,
    snapshot: ChainSnapshot,
    fill: FillModel,
    *,
    pin: Sequence[ResolvedLeg] | None = None,
    closing: bool = False,
) -> StructurePrice:
    """Resolve ``structure`` against ``snapshot`` and price it at ``fill``."""
    spot = snapshot.spot_price
    pinned = {leg.name: leg for leg in (pin or ())}
    resolved_strikes: dict[str, float] = {}
    out: list[ResolvedLeg] = []
    for spec in structure.legs:
        expiry, strike, at_expiry = _resolve_leg(spec, snapshot, pinned, resolved_strikes)
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
    _check_ladder_collisions(structure, out)
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
    """P&L of opening at ``entry`` and closing at ``exit_``."""
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
    meaningful = cost > MIN_MEANINGFUL_COST
    return {
        "cost": cost,
        "exit_value": exit_value,
        "pnl": pnl,
        "ret": pnl / cost if meaningful else float("nan"),
        "alpha_entry": entry.alpha,
        "alpha_exit": exit_.alpha,
    }


def put_calendar(
    back_dte: int = 20,
    front_dte: int = 1,
    back_moneyness: float | None = None,
    entry_offset: int = 0,
    exit_offset: int = 1,
    decision_offset: int | None = None,
) -> Structure:
    """CAL-P — short ~``front_dte`` put, long ~``back_dte`` put, open/closed together."""
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
    """STR-RUNUP — long ATM straddle bought early, sold immediately *before* the print."""
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
    """CND-P — long put condor: long the wings, short the two strikes around spot."""
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
    anchor_offset: int = 0,
    width_moneyness: float | None = None,
    entry_offset: int = 0,
    exit_offset: int = 1,
    decision_offset: int | None = None,
) -> Structure:
    """TWIN-P — two mirrored put condors sharing a doubled at-the-money long."""
    if int(steps) < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    if width_moneyness is not None and not width_moneyness > 0:
        raise ValueError(f"width_moneyness must be positive, got {width_moneyness!r}")
    expiry = ExpirySelector(kind="first_post_event")
    atm = StrikeSelector("bracket", side="below", steps=int(anchor_offset) or None)
    return Structure(
        name="TWIN-P",
        description=(
            "Twin-peak put structure: doubled ATM long, four shorts at +/-w and "
            "+/-2w, wings at +/-4w, all puts, one post-event expiry."
        ),
        legs=(
            LegSpec("atm", PUT, BUY, expiry, atm, qty=2.0),
            LegSpec("up1", PUT, SELL, expiry,
                    StrikeSelector("offset_from", ref="atm",
                                   moneyness=float(width_moneyness))
                    if width_moneyness is not None
                    else StrikeSelector("grid_step", ref="atm", steps=int(steps))),
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
        params={"steps": int(steps), "anchor_offset": int(anchor_offset),
                "width_moneyness": width_moneyness,
                "width_legs": ("atm", "up1"),
                "peak_multiple": 2.0},
    )


def twin_peak_5(
    wing_multiple: int = 3,
    steps: int = 1,
    anchor_offset: int = 0,
    width_moneyness: float | None = None,
    entry_offset: int = 0,
    exit_offset: int = 1,
    decision_offset: int | None = None,
) -> Structure:
    """TWIN-P5 — the twin peak on FIVE strikes instead of seven."""
    if int(wing_multiple) not in (2, 3):
        raise ValueError(f"wing_multiple must be 2 or 3, got {wing_multiple!r}")
    if int(steps) < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    if width_moneyness is not None and not width_moneyness > 0:
        raise ValueError(f"width_moneyness must be positive, got {width_moneyness!r}")
    m = int(wing_multiple)
    expiry = ExpirySelector(kind="first_post_event")
    atm = StrikeSelector("bracket", side="below", steps=int(anchor_offset) or None)
    up_wing = (StrikeSelector("mirror", ref="atm", about="up1") if m == 2
               else StrikeSelector("mirror", ref="dn1", about="up1"))
    dn_wing = (StrikeSelector("mirror", ref="atm", about="dn1") if m == 2
               else StrikeSelector("mirror", ref="up1", about="dn1"))
    return Structure(
        name="TWIN-P5",
        description=(
            f"Five-strike twin peak: doubled ATM long, doubled shorts at +/-a, "
            f"wings at +/-{m}a, all puts, one post-event expiry."
        ),
        legs=(
            LegSpec("atm", PUT, BUY, expiry, atm, qty=2.0),
            LegSpec("up1", PUT, SELL, expiry,
                    StrikeSelector("offset_from", ref="atm",
                                   moneyness=float(width_moneyness))
                    if width_moneyness is not None
                    else StrikeSelector("grid_step", ref="atm", steps=int(steps)),
                    qty=2.0),
            LegSpec("dn1", PUT, SELL, expiry,
                    StrikeSelector("mirror", ref="up1", about="atm"), qty=2.0),
            LegSpec("up_wing", PUT, BUY, expiry, up_wing),
            LegSpec("dn_wing", PUT, BUY, expiry, dn_wing),
        ),
        entry_offset=entry_offset,
        exit_offset=exit_offset,
        decision_offset=decision_offset,
        params={"wing_multiple": m, "steps": int(steps),
                "anchor_offset": int(anchor_offset),
                "width_moneyness": width_moneyness,
                "width_legs": ("atm", "up1"),
                "peak_multiple": 2.0 if m == 3 else 1.0},
    )


#: Marker for a leg that exists only so others can mirror about its strike.
REFERENCE_QTY = 0.0


def _symmetric_put_ladder(name: str, description: str, anchor_qty: int,
                          tail: tuple[tuple[int, int], ...], *,
                          width_moneyness: float | None = None,
                          steps: int = 1, anchor_offset: int = 0,
                          entry_offset: int = 0, exit_offset: int = 1,
                          decision_offset: int | None = None,
                          peak_multiple: float = 1.0) -> Structure:
    """The general symmetric all-put structure the EXP-133 enumeration found."""
    if width_moneyness is not None and not width_moneyness > 0:
        raise ValueError(f"width_moneyness must be positive, got {width_moneyness!r}")
    if anchor_qty + 2 * sum(q for _, q in tail) != 0:
        raise ValueError("contracts must sum to zero or the deep-ITM tail does not cancel")
    expiry = ExpirySelector(kind="first_post_event")
    atm = StrikeSelector("bracket", side="below", steps=int(anchor_offset) or None)
    legs: list[LegSpec] = []
    legs.append(LegSpec("atm", PUT, BUY if anchor_qty >= 0 else SELL, expiry, atm,
                        qty=float(abs(anchor_qty)) if anchor_qty else REFERENCE_QTY))
    for i, (mult, qty) in enumerate(tail, start=1):
        side = BUY if qty > 0 else SELL
        up = (StrikeSelector("offset_from", ref="atm",
                             moneyness=float(width_moneyness) * mult)
              if width_moneyness is not None
              else StrikeSelector("grid_step", ref="atm", steps=int(steps) * mult))
        legs.append(LegSpec(f"up{i}", PUT, side, expiry, up, qty=float(abs(qty))))
        legs.append(LegSpec(f"dn{i}", PUT, side, expiry,
                            StrikeSelector("mirror", ref=f"up{i}", about="atm"),
                            qty=float(abs(qty))))
    return Structure(
        name=name, description=description, legs=tuple(legs),
        entry_offset=entry_offset, exit_offset=exit_offset,
        decision_offset=decision_offset,
        params={"anchor_qty": anchor_qty, "tail": list(tail),
                "width_moneyness": width_moneyness, "steps": int(steps),
                "anchor_offset": int(anchor_offset),
                "width_legs": ("atm", "up1"), "peak_multiple": peak_multiple},
    )


def put_condor_strike(width_moneyness: float | None = None, inner: int = 1,
                      outer: int = 2, **kw) -> Structure:
    """CND-PS — a put condor whose axis of symmetry sits ON a listed strike."""
    return _symmetric_put_ladder(
        "CND-PS", "Put condor centred on a listed strike; short +/-inner, long "
        "+/-outer, all puts, one post-event expiry.",
        0, ((inner, -1), (outer, 1)), width_moneyness=width_moneyness,
        peak_multiple=float(outer - inner), **kw)


def put_butterfly(width_moneyness: float | None = None, **kw) -> Structure:
    """BFLY-P — the three-strike put butterfly: long the wings, short two at A."""
    return _symmetric_put_ladder(
        "BFLY-P", "Put butterfly: short two at the anchor, long both wings, "
        "all puts, one post-event expiry.",
        -2, ((1, 1),), width_moneyness=width_moneyness, peak_multiple=1.0, **kw)


def put_butterfly_wide(width_moneyness: float | None = None, inner: int = 1,
                       outer: int = 3, **kw) -> Structure:
    """BFLY-P5 — a five-strike butterfly: short four at A, long both pairs."""
    return _symmetric_put_ladder(
        "BFLY-P5", "Five-strike put butterfly: short four at the anchor, long "
        "both strike pairs, all puts, one post-event expiry.",
        -4, ((inner, 1), (outer, 1)), width_moneyness=width_moneyness,
        peak_multiple=float(inner + outer), **kw)


def ramp7(**kw) -> Structure:
    """RAMP7 — centre-peaked seven-strike put ladder, stepped ramps."""
    return _symmetric_put_ladder(
        "RAMP7", "Centre-peaked seven-strike put ladder: short two at the "
        "anchor, short the first pair, long the outer two pairs, all puts, "
        "one post-event expiry.",
        -2, ((1, -1), (2, 1), (3, 1)), peak_multiple=4.0, **kw)


def ctr5(**kw) -> Structure:
    """CTR5 — the centre-five: doubled wings on a five-strike put ladder."""
    return _symmetric_put_ladder(
        "CTR5", "Five-strike centre-peaked put ladder: short two at the "
        "anchor, short the inner pair, doubled long the outer pair, all "
        "puts, one post-event expiry.",
        -2, ((1, -1), (2, 2)), peak_multiple=3.0, **kw)


#: Factories keyed by strategy code, so specs can name a structure as a string.
STRUCTURES = {
    "CAL-P": put_calendar,
    "STR-THRU": straddle_through,
    "STR-RUNUP": straddle_runup,
    "CND-P": put_condor,
    "TWIN-P": twin_peak,
    "TWIN-P5": twin_peak_5,
    "CND-PS": put_condor_strike,
    "BFLY-P": put_butterfly,
    "BFLY-P5": put_butterfly_wide,
    "RAMP7": ramp7,
    "CTR5": ctr5,
}

#: The actionable book decides at the prior completed close.
D1_DECISION_OFFSET = -1
D1_STRATEGIES: tuple[str, ...] = tuple(
    strategy for strategy in STRUCTURES if strategy != "STR-RUNUP"
)

#: Explicit D−1 selection for a replay, training job, or future live cutover.
D1_DECISION_OFFSETS: dict[str, int] = {
    strategy: D1_DECISION_OFFSET for strategy in D1_STRATEGIES
}


def with_decision_offset(structure: Structure, decision_offset: int | None) -> Structure:
    """Return structure with a separately declared decision close."""
    return Structure(
        name=structure.name,
        legs=structure.legs,
        entry_offset=structure.entry_offset,
        exit_offset=structure.exit_offset,
        decision_offset=decision_offset,
        description=structure.description,
        params=dict(structure.params),
    )


def execution_variant_label(structure: Structure) -> str:
    """Stable variant label that distinguishes an early decision book."""
    parts = [f"e{structure.entry_offset:+d}", f"x{structure.exit_offset:+d}"]
    if structure.decided_early:
        parts.append(f"d{structure.decided_at:+d}")
    for key in sorted(structure.params):
        value = structure.params[key]
        if value is None:
            continue
        parts.append(f"{key}={value}")
    return "_".join(parts)


__all__ = [
    "ALPHA_SWEEP", "AMC", "BAD_QUOTE_COST_PCT", "BEST", "BMO",
    "CALL", "ChainSnapshot", "D1_DECISION_OFFSET", "D1_DECISION_OFFSETS",
    "D1_STRATEGIES", "ExpirySelector", "FillModel", "LadderTooCoarse",
    "LegSpec", "MID", "MIN_MEANINGFUL_COST", "PrintWindow", "PUT",
    "REFERENCE_QTY", "ResolvedLeg", "STRUCTURES", "Structure",
    "StructureError", "StructurePrice", "StrikeSelector", "TradingCalendar",
    "UNSCHEDULED_CLOSURES", "WORST", "WIDE_MARKET_RATIO", "breakeven_alpha",
    "ctr5", "execution_variant_label", "price_structure", "projected_trading_days",
    "put_butterfly", "put_butterfly_wide", "put_calendar", "put_condor",
    "put_condor_strike", "ramp7", "straddle_runup", "straddle_through",
    "structure_return", "trading_calendar", "twin_peak", "twin_peak_5",
    "us_market_holidays", "with_decision_offset",
]
