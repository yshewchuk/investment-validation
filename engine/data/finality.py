"""Positive evidence that a session close is complete enough to record.

A stale-data allowance answers a different question: whether an old board is
still useful. The ledger needs the stronger fact that the named session was
actually retrieved. A final session has all three independent proofs below:
ORATS published both market-wide files, and the requested universe is present
at that exact date in daily state and option chains.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

from engine.data import fetch, store

__all__ = [
    "MIN_FINAL_DAILY_SHARE", "MIN_FINAL_CHAIN_SHARE", "SessionFinality",
    "session_finality", "resolve_final_session",
]

# These match the existing nightly coverage floor. A lower number would permit
# a frozen decision when a material part of the book had no closing data.
MIN_FINAL_DAILY_SHARE = 0.80
MIN_FINAL_CHAIN_SHARE = 0.80


@dataclass(frozen=True)
class SessionFinality:
    """Auditable evidence for one date and ticker universe."""

    date: str
    market_wide: bool
    daily_share: float
    chain_share: float
    is_final: bool
    detail: str
    tickers: int

    def as_dict(self) -> dict:
        return asdict(self)


def _market_wide_complete(stamp: pd.Timestamp) -> bool:
    target = str(stamp.date())
    found = set()
    for entry in fetch.iter_cached("orats"):
        if entry.endpoint not in {"hist/summaries", "hist/cores"}:
            continue
        if str(entry.params.get("tradeDate")) != target:
            continue
        if int(entry.meta.get("status", 0)) == 200:
            found.add(entry.endpoint)
    return found == {"hist/summaries", "hist/cores"}


def _exact_share(table: str, column: str, stamp: pd.Timestamp, wanted: set[str]) -> float:
    if not wanted:
        return 0.0
    try:
        frame = store.read_table(
            table, years=[stamp.year], columns=["ticker", column]
        )
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return 0.0
    if frame.empty or column not in frame:
        return 0.0
    dates = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    got = set(frame.loc[dates == stamp, "ticker"].dropna().astype(str))
    return len(wanted & got) / len(wanted)


def session_finality(date, tickers: Iterable[str]) -> SessionFinality:
    """Return whether closing data exists for exactly one date."""
    stamp = pd.Timestamp(date).normalize()
    wanted = {str(t) for t in tickers if t is not None and str(t)}
    market_wide = _market_wide_complete(stamp)
    daily_share = _exact_share("daily_market", "date", stamp, wanted)
    chain_share = _exact_share("option_chains", "obs_date", stamp, wanted)
    is_final = (
        bool(wanted)
        and market_wide
        and daily_share >= MIN_FINAL_DAILY_SHARE
        and chain_share >= MIN_FINAL_CHAIN_SHARE
    )
    failures = []
    if not market_wide:
        failures.append("market-wide summaries/cores missing")
    if daily_share < MIN_FINAL_DAILY_SHARE:
        failures.append(f"daily {daily_share:.0%} < {MIN_FINAL_DAILY_SHARE:.0%}")
    if chain_share < MIN_FINAL_CHAIN_SHARE:
        failures.append(f"chains {chain_share:.0%} < {MIN_FINAL_CHAIN_SHARE:.0%}")
    if not wanted:
        failures.append("empty ticker universe")
    return SessionFinality(
        date=str(stamp.date()),
        market_wide=market_wide,
        daily_share=float(daily_share),
        chain_share=float(chain_share),
        is_final=bool(is_final),
        detail="final" if not failures else "; ".join(failures),
        tickers=len(wanted),
    )


def resolve_final_session(
    requested,
    tickers: Iterable[str],
    *,
    calendar,
    max_sessions: int = 15,
) -> SessionFinality:
    """Return the newest final session at or before the requested date."""
    stamp = pd.Timestamp(requested).normalize()
    for _ in range(max_sessions):
        if calendar.is_trading_day(stamp):
            result = session_finality(stamp, tickers)
            if result.is_final:
                return result
        stamp = calendar.shift(stamp, -1)
    raise RuntimeError(
        f"no final session at or before {pd.Timestamp(requested).date()} "
        f"within {max_sessions} trading sessions"
    )
