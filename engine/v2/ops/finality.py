"""Native v2 finality resolution for supervised EOD stages.

The v2 action owns the finality decision and its coverage calculations.  The
legacy module remains a compatibility source for older callers, but a normal
v2 invocation executes this implementation and records its own result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

MIN_FINAL_DAILY_SHARE = 0.80
MIN_FINAL_CHAIN_SHARE = 0.80


@dataclass(frozen=True)
class SessionFinality:
    date: str
    market_wide: bool
    daily_share: float
    chain_share: float
    is_final: bool
    detail: str
    tickers: int
    covered: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _legacy_compatibility(name, native):
    """Honor an explicitly monkeypatched legacy seam during transition tests."""
    from engine.v2.ops import legacy_adapter
    return legacy_adapter.finality_compatibility(name, native)


def _market_wide_complete(stamp: pd.Timestamp) -> bool:
    from engine.v2.ops import legacy_adapter
    return legacy_adapter.finality_market_wide_complete(stamp)


def _coverage_frame(table: str, column: str, stamp: pd.Timestamp):
    from engine.v2.ops import legacy_adapter
    return legacy_adapter.finality_coverage_frame(table, column, stamp)


def _coverage_sets(table: str, column: str, stamp: pd.Timestamp,
                   wanted: set[str], frame=None) -> tuple[set[str], set[str]]:
    """Return requested tickers carried at all and present on the target date.

    Session finality combines the carried sets from both tables before
    calculating either share. A ticker carried by only one table therefore
    remains in the shared denominator and counts as missing from the other.
    """
    if not wanted:
        return set(), set()
    frame = frame if frame is not None else _coverage_frame(table, column, stamp)
    if frame is None or frame.empty or column not in frame or "ticker" not in frame:
        return set(), set()
    carried = wanted & set(frame["ticker"].dropna().astype(str))
    if not carried:
        return set(), set()
    dates = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    got = set(frame.loc[dates == stamp, "ticker"].dropna().astype(str))
    return carried, carried & got


def _shared_coverage(daily_carried: set[str], daily_exact: set[str],
                     chain_carried: set[str], chain_exact: set[str]):
    carried = daily_carried | chain_carried
    if not carried:
        return 0.0, 0.0, 0
    covered = len(carried)
    return len(daily_exact) / covered, len(chain_exact) / covered, covered


def _native_session_finality(value, tickers: Iterable[str], *, frames=None,
                              market_wide=None):
    stamp = pd.Timestamp(value).normalize()
    wanted = {str(item) for item in tickers if item is not None and str(item)}
    frames = frames or {}
    daily_carried, daily_exact = _coverage_sets(
        "daily_market", "date", stamp, wanted, frames.get("daily_market"))
    chain_carried, chain_exact = _coverage_sets(
        "option_chains", "obs_date", stamp, wanted, frames.get("option_chains"))
    daily_share, chain_share, covered = _shared_coverage(
        daily_carried, daily_exact, chain_carried, chain_exact)
    if market_wide is None:
        market_wide = _market_wide_complete(stamp)
    final = bool(wanted) and covered > 0 and market_wide \
        and daily_share >= MIN_FINAL_DAILY_SHARE \
        and chain_share >= MIN_FINAL_CHAIN_SHARE
    failures = []
    if not market_wide:
        failures.append("market-wide summaries/cores missing")
    if daily_share < MIN_FINAL_DAILY_SHARE:
        failures.append(f"daily {daily_share:.0%} < {MIN_FINAL_DAILY_SHARE:.0%}")
    if chain_share < MIN_FINAL_CHAIN_SHARE:
        failures.append(f"chains {chain_share:.0%} < {MIN_FINAL_CHAIN_SHARE:.0%}")
    if not wanted:
        failures.append("empty ticker universe")
    elif not covered:
        failures.append("none of the requested tickers are carried in the store")
    return SessionFinality(
        date=str(stamp.date()), market_wide=market_wide,
        daily_share=float(daily_share), chain_share=float(chain_share),
        is_final=bool(final), detail="final" if not failures else "; ".join(failures),
        tickers=len(wanted), covered=int(covered))


def session_finality(value, tickers: Iterable[str], *, frames=None, market_wide=None):
    compatibility = _legacy_compatibility("session_finality", _native_session_finality)
    if compatibility is not _native_session_finality:
        result = compatibility(value, tickers, frames=frames)
        return SessionFinality(**result.as_dict())
    return _native_session_finality(value, tickers, frames=frames,
                                    market_wide=market_wide)


def _native_resolve_final_session(requested, tickers, *, calendar, max_sessions=15):
    stamp = pd.Timestamp(requested).normalize()
    frames = {
        "daily_market": _coverage_frame("daily_market", "date", stamp),
        "option_chains": _coverage_frame("option_chains", "obs_date", stamp),
    }
    for _ in range(max_sessions):
        if calendar.is_trading_day(stamp):
            result = session_finality(stamp, tickers, frames=frames)
            if result.is_final:
                return result
        stamp = calendar.shift(stamp, -1)
    raise RuntimeError(
        f"no final session at or before {pd.Timestamp(requested).date()} "
        f"within {max_sessions} trading sessions")


def resolve_final_session(requested, tickers, *, calendar, max_sessions=15):
    compatibility = _legacy_compatibility("resolve_final_session", _native_resolve_final_session)
    if compatibility is not _native_resolve_final_session:
        result = compatibility(requested, tickers, calendar=calendar, max_sessions=max_sessions)
        return SessionFinality(**result.as_dict())
    return _native_resolve_final_session(requested, tickers, calendar=calendar,
                                         max_sessions=max_sessions)


def _native_covered_tickers(value, tickers):
    stamp = pd.Timestamp(value).normalize()
    frames = {
        "daily_market": _coverage_frame("daily_market", "date", stamp),
        "option_chains": _coverage_frame("option_chains", "obs_date", stamp),
    }
    return [ticker for ticker in sorted({str(item) for item in tickers if item})
            if session_finality(stamp, (ticker,), frames=frames).is_final]


def covered_tickers(value, tickers):
    compatibility = _legacy_compatibility("covered_tickers", _native_covered_tickers)
    if compatibility is not _native_covered_tickers:
        return compatibility(value, tickers)
    return _native_covered_tickers(value, tickers)
