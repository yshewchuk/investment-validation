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
    for col in ("annc_tod", "updated_at", "session", "session_src"):
        out[col] = out[col].astype("string")

    today = pd.Timestamp.today().normalize()
    forward = out[pd.to_datetime(out["event_date"]) >= today]
    report = {
        "rows": int(len(out)),
        "tickers": int(out["ticker"].nunique()),
        "years": f"{int(out['year'].min())}–{int(out['year'].max())}",
        "both_sources": int(out["date_agree"].sum()),
        "orats_only": int((out["src_orats"] & ~out["src_oquants"]).sum()),
        "oquants_only": int((~out["src_orats"] & out["src_oquants"]).sum()),
        "session_known": int(out["session"].notna().sum()),
        "session_by_source": {
            str(k): int(v) for k, v in out["session_src"].value_counts().items()
        },
        # The forward slice is the one the monitoring board scores, and the one
        # ORATS cannot supply — reported separately so a calendar that is fat
        # with history and empty ahead cannot look healthy.
        "forward_rows": int(len(forward)),
        "forward_tickers": int(forward["ticker"].nunique()),
        "forward_session_known": int(forward["session"].notna().sum()),
        "forward_date_conflicts": int(forward["date_conflict"].sum()),
    }
    return out, report


def session_coverage(events: pd.DataFrame) -> dict:
    """BMO/AMC split, and how much of the calendar has no session at all."""
    counts = events["session"].value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}
