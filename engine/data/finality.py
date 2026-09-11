"""Positive evidence that a session close is complete enough to record.

A stale-data allowance answers a different question: whether an old board is
still useful. The ledger needs the stronger fact that the named session was
actually retrieved. A final session has all three independent proofs below:
ORATS published both market-wide files, and the COVERED universe is present at
that exact date in daily state and option chains.

"Covered" is doing real work in that sentence, and getting it wrong made this
gate refuse every session it was ever shown — see :func:`_exact_share`. The
share is measured against the tickers each table carries, not against every
ticker the caller asked about, because a name no data source covers cannot
ever produce a row and would otherwise veto the session forever.
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
    #: Tickers asked about.
    tickers: int
    #: Of those, how many the store carries at all — the denominator the
    #: shares above are actually measured against. Recorded on the row so a
    #: reader can tell a session judged on 132 covered names from one judged
    #: on 4, which the share alone cannot say.
    covered: int = 0

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


def _coverage_frame(table: str, column: str, stamp: pd.Timestamp):
    """``(ticker, column)`` for the window coverage is established over.

    ``{year - 1, year}`` rather than the stamp's own year, so a session in
    early January is not judged against a table holding two days of rows —
    and so a walk-back that crosses the new year sees December.
    """
    try:
        return store.read_table(
            table, years=sorted({stamp.year - 1, stamp.year}),
            columns=["ticker", column],
        )
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return None


def _exact_share(
    table: str, column: str, stamp: pd.Timestamp, wanted: set[str],
    frame=None,
) -> tuple[float, int]:
    """Share of the COVERED part of ``wanted`` present at exactly ``stamp``.

    Returns ``(share, n_covered)``.

    The denominator is the tickers this table carries at all in the read
    window, not every ticker asked about. A name the pipeline does not cover
    — an illiquid micro-cap no data source carries — has no row on any date,
    so counting it against a same-session check asks whether data arrived
    that was never going to arrive, and the answer is permanently no.

    Measured 2026-09-11, which is what this rule exists to fix: of the 213
    calendar names the nightly asked about for 2026-09-10, only 132 are
    carried at all; the missing 81 are identical in `daily_market` and
    `option_chains` (AENT, ALAR, BTTC, CMMB, EONR, …), so both raw shares
    came out at exactly 62% and the gate refused every session it was ever
    shown. Against the covered 132 the same night scores 100%.

    This is the denominator `validate_refresh`'s `ticker_coverage` check has
    always used — it divides by the tickers with rows, which is why it read
    47/47 = 100% on the night this gate read 62%. Two checks over one
    universe disagreeing that far meant one of them was counting the wrong
    thing.

    ``frame`` lets a caller walking several sessions read each table once
    instead of once per session — the content does not change between
    iterations, only the date compared against it.
    """
    if not wanted:
        return 0.0, 0
    if frame is None:
        frame = _coverage_frame(table, column, stamp)
    if frame is None or frame.empty or column not in frame:
        return 0.0, 0
    tickers = frame["ticker"].dropna().astype(str)
    covered = wanted & set(tickers)
    if not covered:
        # Nothing asked about is carried here at all. That is a coverage
        # failure, not a vacuous pass: dividing by an empty denominator is
        # how a store that lost the whole universe would report perfect
        # freshness.
        return 0.0, 0
    dates = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    got = set(frame.loc[dates == stamp, "ticker"].dropna().astype(str))
    return len(covered & got) / len(covered), len(covered)


def session_finality(date, tickers: Iterable[str], *, frames=None) -> SessionFinality:
    """Return whether closing data exists for exactly one date.

    ``frames`` is an optional ``{table: frame}`` of already-read coverage
    tables, so :func:`resolve_final_session` can walk several sessions
    without re-reading `option_chains` once per session.
    """
    stamp = pd.Timestamp(date).normalize()
    wanted = {str(t) for t in tickers if t is not None and str(t)}
    market_wide = _market_wide_complete(stamp)
    frames = frames or {}
    daily_share, daily_covered = _exact_share(
        "daily_market", "date", stamp, wanted, frames.get("daily_market"))
    chain_share, chain_covered = _exact_share(
        "option_chains", "obs_date", stamp, wanted, frames.get("option_chains"))
    covered = min(daily_covered, chain_covered)
    is_final = (
        bool(wanted)
        and covered > 0
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
    elif not covered:
        failures.append("none of the requested tickers are carried in the store")
    return SessionFinality(
        date=str(stamp.date()),
        market_wide=market_wide,
        daily_share=float(daily_share),
        chain_share=float(chain_share),
        is_final=bool(is_final),
        detail="final" if not failures else "; ".join(failures),
        tickers=len(wanted),
        covered=int(covered),
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
    # Read each coverage table once for the whole walk. Their content does
    # not change between iterations — only the date compared against it —
    # and re-reading option_chains fifteen times cost ~2 minutes of the
    # nightly's critical path for nothing.
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
        f"within {max_sessions} trading sessions"
    )
