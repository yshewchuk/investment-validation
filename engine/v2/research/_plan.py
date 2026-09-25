"""Event planning — pure calendar arithmetic, no quotes.

Split out of ``engine.v2.research.replay`` (review blocker: module fan-out)
with bodies unchanged, so the planning half stays byte-identical to
``engine/replay.py``'s.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.v2.research._pricing import Structure, TradingCalendar, trading_calendar

__all__ = ["SKIP_REASONS", "ReplayPlan", "plan_events"]

#: Why a planned trade produced no row. Counted rather than dropped: a replay
#: that silently loses 40% of its candidates is a replay whose headline number
#: is about the surviving 60%, and nobody can see which 60% that was.
SKIP_REASONS = (
    "no_entry_chain",
    "no_exit_chain",
    # Only ever non-zero for a structure decided before it enters: when the
    # decision close is the entry close, an event without a decision chain has
    # already been counted as `no_entry_chain`.
    "no_decision_chain",
    "structure_unresolved",
    "expiry_gone_at_exit",
    "bad_quote",
    "no_session",
    "calendar_out_of_range",
    "zero_cost",
)


@dataclass
class ReplayPlan:
    """Which dates each event would be traded on, before any chain is touched."""

    frame: pd.DataFrame
    skipped: dict[str, int] = field(default_factory=dict)

    @property
    def chain_keys(self) -> set[tuple[str, pd.Timestamp]]:
        """Every (ticker, date) chain this plan needs loaded."""
        keys = set(zip(self.frame["ticker"], self.frame["entry_date"]))
        keys |= set(zip(self.frame["ticker"], self.frame["exit_date"]))
        if "decision_date" in self.frame.columns:
            keys |= set(zip(self.frame["ticker"], self.frame["decision_date"]))
        return keys

    @property
    def years(self) -> list[int]:
        dates = pd.concat([self.frame["entry_date"], self.frame["exit_date"]])
        return sorted(pd.to_datetime(dates).dt.year.unique().tolist())


def plan_events(
    structure: Structure,
    events: pd.DataFrame,
    calendar: TradingCalendar | None = None,
) -> ReplayPlan:
    """Resolve every event's entry and exit dates for ``structure``."""
    cal = calendar or trading_calendar()
    rows: list[dict] = []
    skipped = {reason: 0 for reason in SKIP_REASONS}

    for event in events.itertuples(index=False):
        session = getattr(event, "session", None)
        if session is None or (isinstance(session, float) and np.isnan(session)) or pd.isna(session):
            skipped["no_session"] += 1
            continue
        event_date = pd.Timestamp(event.event_date).normalize()
        try:
            window = cal.resolve_offsets(
                event_date, str(session), structure.entry_offset, structure.exit_offset,
                decision_offset=structure.decision_offset,
            )
        except KeyError:
            skipped["calendar_out_of_range"] += 1
            continue
        rows.append(
            {
                "event_id": getattr(event, "event_id", f"{event.ticker}_{event_date.date()}"),
                "ticker": str(event.ticker),
                "event_date": event_date,
                "session": str(session),
                "decision_date": window.decision_date,
                "entry_date": window.entry_date,
                "exit_date": window.exit_date,
                "last_pre_print": window.last_pre_print,
                "first_post_print": window.first_post_print,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "event_id", "ticker", "event_date", "session", "decision_date",
            "entry_date", "exit_date", "last_pre_print", "first_post_print",
        ],
    )
    if len(frame):
        frame = frame.sort_values(["ticker", "event_date"]).reset_index(drop=True)
    return ReplayPlan(frame=frame, skipped=skipped)
