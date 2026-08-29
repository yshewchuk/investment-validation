"""Normalize the earnings calendar into Tier-2 ``earnings_events``.

The merge logic lives in :mod:`engine.calendar` (it is also what the live
calendar refresh uses); this module is the thin adapter that shapes it to the
Tier-2 schema. ORATS ``anncTod`` is authoritative for the session, the oquants
panel is the cross-check, and disagreements survive as ``date_agree=False``
rather than being resolved away.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from engine.calendar import build_calendar

__all__ = ["normalize", "session_coverage"]


def normalize(tickers: Iterable[str] | None = None) -> tuple[pd.DataFrame, dict]:
    cal = build_calendar(tickers=tickers)
    if cal.empty:
        return cal, {"rows": 0}

    out = cal.copy()
    out["year"] = pd.to_datetime(out["event_date"]).dt.year
    out["annc_tod"] = out["annc_tod"].astype("string")
    out["updated_at"] = out["updated_at"].astype("string")
    out["session"] = out["session"].astype("string")

    report = {
        "rows": int(len(out)),
        "tickers": int(out["ticker"].nunique()),
        "years": f"{int(out['year'].min())}–{int(out['year'].max())}",
        "both_sources": int(out["date_agree"].sum()),
        "orats_only": int((out["src_orats"] & ~out["src_oquants"]).sum()),
        "oquants_only": int((~out["src_orats"] & out["src_oquants"]).sum()),
        "session_known": int(out["session"].notna().sum()),
    }
    return out, report


def session_coverage(events: pd.DataFrame) -> dict:
    """BMO/AMC split, and how much of the calendar has no session at all."""
    counts = events["session"].value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}
