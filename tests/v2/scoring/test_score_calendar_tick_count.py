"""score_calendar reports the PROCESSED event count (1..N) to memprobe.tick."""
from __future__ import annotations

import pandas as pd

import engine.score as score_mod


def test_score_calendar_tick_receives_processed_counts(monkeypatch):
    events = pd.DataFrame(
        [
            {"event_id": 100 + n, "ticker": f"T{n}",
             "event_date": pd.Timestamp("2026-10-22"), "session": "AMC"}
            for n in range(1, 4)
        ]
    )
    monkeypatch.setattr(score_mod.store, "read_table", lambda *a, **k: events)
    ticks = []
    monkeypatch.setattr(
        score_mod.memprobe, "tick", lambda label, i: ticks.append((label, i))
    )

    class Engine:
        calendar = object()
        _live_features_cache = {}

    score_mod.score_calendar(
        "2026-10-08",
        strategies=[],
        scorer=Engine(),
        quote_max_age_sessions=None,
        progress_every=0,
    )
    assert ticks == [("score_calendar events", n) for n in (1, 2, 3)]